#!/usr/bin/env python3
"""
🚀 BART FINE-TUNING PIPELINE
============================
Fine-tune BART specifically for email classification using high-quality,
long-format email data with proper email structures.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
        DataCollatorWithPadding
    )
    from datasets import Dataset
    import torch.nn.functional as F
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Install with: pip install transformers datasets torch accelerate")
    sys.exit(1)

from components.enhanced_dataset_generator import EnhancedEmailDatasetGenerator

class BartFineTuner:
    """Fine-tune BART for email classification."""
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_to_id = {}
        self.id_to_label = {}
        
        print(f"🔧 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_config(self, config_path: str = None):
        """Load configuration."""
        base_dir = os.path.dirname(__file__)
        candidates = []
        if config_path:
            candidates.append(config_path)
        candidates.extend([
            os.path.normpath(os.path.join(base_dir, '..', 'config', 'model_config.json')),
            'production/config/model_config.json',
            'config/model_config.json',
        ])

        for path in candidates:
            try:
                if os.path.isfile(path):
                    with open(path, 'r') as f:
                        cfg = json.load(f)
                    print(f"🧩 Loaded config from: {path}")
                    return cfg
            except Exception:
                continue
        print("❌ Config not found, using defaults")
        return {
            "categories": ["Personal", "Spam", "Standard", "Urgent", "Work"],
            "bart": {"model_name": "facebook/bart-base"}
        }
    
    def setup_model_and_tokenizer(self, model_name: str = "facebook/bart-base"):
        """Initialize BART model and tokenizer for classification."""
        print(f"📥 Loading BART model: {model_name}")
        
        # Setup label mappings
        categories = self.config.get('categories', ["Personal", "Spam", "Standard", "Urgent", "Work"])
        self.label_to_id = {label: i for i, label in enumerate(categories)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(categories),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True  # Allow different number of labels
        )
        
        self.model.to(self.device)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {self.model.num_parameters():,}")
        print(f"   Labels: {categories}")
    
    def generate_training_data(self, 
                              total_samples: int = 5000,
                              use_existing_file: str = None) -> pd.DataFrame:
        """Generate or load training data."""
        
        if use_existing_file and os.path.exists(use_existing_file):
            print(f"📥 Loading existing dataset: {use_existing_file}")
            df = pd.read_csv(use_existing_file)
            print(f"   Loaded {len(df):,} emails")
            return df
        
        print(f"🎯 Generating {total_samples:,} training emails...")
        generator = EnhancedEmailDatasetGenerator()
        
        # Calculate samples per category
        samples_per_category = total_samples // 5
        
        df = generator.generate_comprehensive_dataset(
            spam_count=samples_per_category,
            work_count=samples_per_category,
            personal_count=samples_per_category,
            urgent_count=samples_per_category,
            standard_count=samples_per_category
        )
        
        # Save generated dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bart_training_dataset_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Dataset saved to: {filename}")
        
        return df
    
    def preprocess_emails(self, emails: List[str]) -> List[str]:
        """Preprocess emails for BART fine-tuning."""
        processed = []
        
        for email in emails:
            # Remove email headers for training (keep just subject and content)
            lines = email.split('\n')
            
            # Find where the actual content starts (after headers)
            content_start = 0
            for i, line in enumerate(lines):
                if line.startswith('Subject:'):
                    # Extract subject
                    subject = line.replace('Subject:', '').strip()
                    # Find content start (first empty line after headers)
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() == '':
                            content_start = j + 1
                            break
                    break
            
            # Combine subject and content
            if content_start < len(lines):
                content = '\n'.join(lines[content_start:]).strip()
                # Format for BART: "Classify email: [SUBJECT] [CONTENT]"
                processed_email = f"Classify this email: {subject} {content}"
            else:
                # Fallback if no proper structure found
                processed_email = f"Classify this email: {email}"
            
            # Truncate if too long (BART has token limits)
            if len(processed_email) > 3000:
                processed_email = processed_email[:3000] + "..."
            
            processed.append(processed_email)
        
        return processed
    
    def create_dataset(self, df: pd.DataFrame) -> Tuple[Dataset, Dataset, Dataset]:
        """Create training, validation, and test datasets."""
        print("📊 Creating datasets...")
        
        # Preprocess emails
        processed_emails = self.preprocess_emails(df['email'].tolist())
        labels = [self.label_to_id[cat] for cat in df['category'].tolist()]
        
        # Split data: 70% train, 15% validation, 15% test
        train_emails, temp_emails, train_labels, temp_labels = train_test_split(
            processed_emails, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        val_emails, test_emails, val_labels, test_labels = train_test_split(
            temp_emails, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        print(f"   Training samples: {len(train_emails):,}")
        print(f"   Validation samples: {len(val_emails):,}")
        print(f"   Test samples: {len(test_emails):,}")
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,  # Will be handled by data collator
                max_length=512,  # BART's max sequence length
                return_tensors=None
            )
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict({
            'text': train_emails,
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_emails,
            'labels': val_labels
        })
        
        test_dataset = Dataset.from_dict({
            'text': test_emails,
            'labels': test_labels
        })
        
        # Apply tokenization
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_training_arguments(self, output_dir: str = "./bart_finetuned") -> TrainingArguments:
        """Setup training arguments."""
        
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Training hyperparameters
            num_train_epochs=3,
            per_device_train_batch_size=8 if torch.cuda.is_available() else 2,
            per_device_eval_batch_size=16 if torch.cuda.is_available() else 4,
            gradient_accumulation_steps=2,
            
            # Optimization
            learning_rate=2e-5,
            weight_decay=0.01,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            
            # Learning rate scheduling
            warmup_steps=100,
            lr_scheduler_type="linear",
            
            # Evaluation and saving
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            
            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            report_to=None,  # Disable wandb/tensorboard
            
            # Performance
            dataloader_num_workers=2,
            remove_unused_columns=False,
            

            
            # Mixed precision (if available)
            fp16=torch.cuda.is_available(),
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        # Handle cases where predictions/labels come as tuples or lists of arrays
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]
        if isinstance(labels, (tuple, list)):
            labels = labels[0]

        # Ensure numpy arrays
        try:
            logits = np.array(predictions)
            label_ids = np.array(labels)
        except Exception:
            logits = predictions
            label_ids = labels

        # Argmax over last dimension
        try:
            pred_labels = np.argmax(logits, axis=-1)
        except Exception:
            logits = np.asarray(list(logits))
            pred_labels = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(label_ids, pred_labels)
        return {"accuracy": float(accuracy)}
    
    def fine_tune_model(self, train_dataset: Dataset, val_dataset: Dataset) -> Trainer:
        """Fine-tune the BART model."""
        print("🚀 Starting BART fine-tuning...")
        
        # Training arguments
        training_args = self.setup_training_arguments()
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        print(f"🔥 Training on {len(train_dataset):,} samples...")
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Learning rate: {training_args.learning_rate}")
        print(f"   Epochs: {training_args.num_train_epochs}")
        
        trainer.train()
        
        print("✅ Fine-tuning completed!")
        
        return trainer
    
    def evaluate_model(self, trainer: Trainer, test_dataset: Dataset) -> Dict:
        """Evaluate the fine-tuned model."""
        print("📊 Evaluating fine-tuned model...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Detailed classification report
        categories = list(self.label_to_id.keys())
        report = classification_report(
            y_true, y_pred, 
            target_names=categories, 
            output_dict=True
        )
        
        print(f"📈 EVALUATION RESULTS")
        print("=" * 40)
        print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=categories))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print("   ", " ".join(f"{cat:>8}" for cat in categories))
        for i, row in enumerate(cm):
            print(f"{categories[i]:>8}", " ".join(f"{val:>8}" for val in row))
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": y_pred.tolist(),
            "true_labels": y_true.tolist()
        }
    
    def save_model(self, trainer: Trainer, save_path: str = "./bart_email_classifier"):
        """Save the fine-tuned model."""
        print(f"💾 Saving fine-tuned model to: {save_path}")
        
        # Save model and tokenizer
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save label mappings
        label_mapping = {
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label
        }
        
        with open(f"{save_path}/label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print("✅ Model saved successfully!")
        return save_path
    
    def run_full_pipeline(self, 
                         dataset_size: int = 5000,
                         existing_dataset: str = None,
                         save_model_path: str = None) -> str:
        """Run the complete fine-tuning pipeline."""
        
        print("🚀 BART FINE-TUNING PIPELINE")
        print("=" * 50)
        
        # 1. Setup model
        bart_config = self.config.get("bart", {})
        model_name = bart_config.get("model_name", "facebook/bart-base")
        # Use base model for fine-tuning, not MNLI
        if "mnli" in model_name.lower():
            model_name = "facebook/bart-base"
        self.setup_model_and_tokenizer(model_name)
        
        # 2. Generate/load data
        df = self.generate_training_data(dataset_size, existing_dataset)
        
        # 3. Create datasets
        train_dataset, val_dataset, test_dataset = self.create_dataset(df)
        
        # 4. Fine-tune model
        trainer = self.fine_tune_model(train_dataset, val_dataset)
        
        # 5. Evaluate model
        results = self.evaluate_model(trainer, test_dataset)
        
        # 6. Save model
        if save_model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_model_path = f"./bart_email_classifier_{timestamp}"
        
        final_path = self.save_model(trainer, save_model_path)
        
        # 7. Save results
        results_file = f"{save_model_path}/training_results.json"
        training_summary = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "dataset_size": len(df),
            "training_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "final_accuracy": results["accuracy"],
            "detailed_results": results,
            "config": self.config
        }
        
        with open(results_file, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"\n🎉 FINE-TUNING PIPELINE COMPLETE!")
        print("=" * 50)
        print(f"Model saved to: {final_path}")
        print(f"Results saved to: {results_file}")
        print(f"Final accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        
        return final_path


def main():
    """Main fine-tuning function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune BART for email classification")
    parser.add_argument("--dataset-size", type=int, default=5000, 
                       help="Number of training samples to generate")
    parser.add_argument("--existing-dataset", type=str, default=None,
                       help="Path to existing dataset CSV file")
    parser.add_argument("--model-name", type=str, default="facebook/bart-base",
                       help="Base BART model to fine-tune")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save the fine-tuned model")
    
    args = parser.parse_args()
    
    # Create fine-tuner
    fine_tuner = BartFineTuner()
    
    # Override model name if specified
    if args.model_name != "facebook/bart-base":
        fine_tuner.config["bart"] = {"model_name": args.model_name}
    
    # Run pipeline
    try:
        model_path = fine_tuner.run_full_pipeline(
            dataset_size=args.dataset_size,
            existing_dataset=args.existing_dataset,
            save_model_path=args.save_path
        )
        print(f"\n✅ Success! Fine-tuned model available at: {model_path}")
        return 0
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
