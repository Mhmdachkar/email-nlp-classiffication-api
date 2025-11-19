#!/usr/bin/env python3
"""
🎯 BART ENSEMBLE EVALUATION & TUNING
====================================
Evaluates BART integration with the classical model and finds optimal weights.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
import pickle
import glob
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from performance_optimizer import PerformanceOptimizer
from components.bart_classifier import BartZeroShotClassifier

class BartEnsembleEvaluator:
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.optimizer = PerformanceOptimizer()
        self.classical_model = None
        self.scaler = None
        self.label_encoder = None
        self.bart = None
        self.categories = self.config.get('categories', [])
        
    def load_config(self, config_path: str = None):
        """Load configuration from robust set of candidate paths."""
        base_dir = os.path.dirname(__file__)
        candidates = []
        if config_path:
            candidates.append(config_path)
        # Relative to this script (preferred)
        candidates.append(os.path.normpath(os.path.join(base_dir, 'config', 'model_config.json')))
        # Project root common paths (when CWD is repo root)
        candidates.append(os.path.normpath(os.path.join(base_dir, '..', 'production', 'config', 'model_config.json')))
        candidates.append(os.path.normpath(os.path.join(base_dir, '..', 'config', 'model_config.json')))
        candidates.append('production/config/model_config.json')
        candidates.append('config/model_config.json')

        for path in candidates:
            try:
                if os.path.isfile(path):
                    with open(path, 'r') as f:
                        cfg = json.load(f)
                    print(f"🧩 Loaded config from: {path}")
                    return cfg
            except Exception as e:
                print(f"⚠️  Failed loading config at {path}: {e}")
                continue
        print("❌ Error loading config: checked candidates but none found")
        return {}
    
    def load_classical_model(self, models_dir: str = None):
        """Load the trained classical model from common search locations."""
        print("📂 Loading classical model...")
        try:
            base_dir = os.path.dirname(__file__)
            candidate_dirs = []
            if models_dir:
                candidate_dirs.append(models_dir)
            # Common locations relative to this script
            candidate_dirs.extend([
                os.path.join(base_dir, "models"),
                os.path.join(base_dir, "../models"),
                os.path.join(base_dir, "./../models"),
            ])
            # Normalize and dedupe
            candidate_dirs = list(dict.fromkeys([os.path.normpath(d) for d in candidate_dirs]))

            latest_model = latest_scaler = latest_encoder = None
            for dir_path in candidate_dirs:
                model_files = glob.glob(os.path.join(dir_path, "*calibrated_model_*.pkl"))
                scaler_files = glob.glob(os.path.join(dir_path, "*scaler_*.pkl"))
                encoder_files = glob.glob(os.path.join(dir_path, "*label_encoder_*.pkl"))
                if model_files and scaler_files and encoder_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    latest_scaler = max(scaler_files, key=os.path.getctime)
                    latest_encoder = max(encoder_files, key=os.path.getctime)
                    print(f"   Using models from: {dir_path}")
                    break

            if not all([latest_model, latest_scaler, latest_encoder]):
                print("❌ Classical model files not found in any known directory!")
                print("   Checked: ")
                for d in candidate_dirs:
                    print(f"   - {d}")
                return False

            print(f"   Model: {os.path.basename(latest_model)}")
            print(f"   Scaler: {os.path.basename(latest_scaler)}")
            print(f"   Encoder: {os.path.basename(latest_encoder)}")

            with open(latest_model, 'rb') as f:
                self.classical_model = pickle.load(f)
            with open(latest_scaler, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(latest_encoder, 'rb') as f:
                self.label_encoder = pickle.load(f)

            # Align categories order to label encoder for consistent probability axes
            self.categories = list(self.label_encoder.classes_)

            print("✅ Classical model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading classical model: {e}")
            return False
    
    def init_bart(self):
        """Initialize BART classifier."""
        print("🤖 Initializing BART classifier...")
        
        try:
            bart_cfg = self.config.get("bart", {})
            if not bart_cfg:
                print("❌ No BART config found")
                return False
            
            self.bart = BartZeroShotClassifier(
                model_name=bart_cfg.get("model_name", "facebook/bart-large-mnli"),
                candidate_labels=self.categories,
                multi_label=bart_cfg.get("multi_label", False),
                device=bart_cfg.get("device", -1),
                hypothesis_template=bart_cfg.get("hypothesis_template", "This text is about {}."),
            )
            print("✅ BART classifier initialized!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing BART: {e}")
            return False
    
    def generate_validation_data(self, examples_per_category=100):
        """Generate balanced validation data."""
        print(f"📊 Generating validation data ({examples_per_category} per category)...")
        
        # Use PerformanceOptimizer to generate data
        df = self.optimizer.generate_high_confidence_training_data()
        
        # Take a subset for validation
        validation_data = []
        for category in self.categories:
            category_data = df[df['category'] == category].sample(
                n=min(examples_per_category, len(df[df['category'] == category])),
                random_state=42
            )
            validation_data.append(category_data)
        
        validation_df = pd.concat(validation_data, ignore_index=True)
        
        print(f"✅ Generated {len(validation_df)} validation examples")
        print(f"   Distribution: {validation_df['category'].value_counts().to_dict()}")
        
        return validation_df
    
    def evaluate_classical_model(self, validation_df):
        """Evaluate classical model performance."""
        print("📈 Evaluating classical model...")
        
        # Extract features and make predictions
        features = self.optimizer.extract_optimized_features(validation_df['email'].values)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.classical_model.predict(features_scaled)
        probabilities = self.classical_model.predict_proba(features_scaled)
        
        # Convert predictions back to labels
        predicted_labels = [self.label_encoder.classes_[p] for p in predictions]
        true_labels = validation_df['category'].values
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Get confidence scores
        confidences = np.max(probabilities, axis=1)
        
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Average Confidence: {np.mean(confidences):.3f} ({np.mean(confidences)*100:.1f}%)")
        
        return {
            'predictions': predicted_labels,
            'probabilities': probabilities,
            'confidences': confidences,
            'accuracy': accuracy
        }
    
    def evaluate_bart_model(self, validation_df):
        """Evaluate BART model performance."""
        print("🤖 Evaluating BART model...")
        
        predictions = []
        confidences = []
        all_probabilities = []
        
        for email in validation_df['email'].values:
            try:
                pred, conf, probs = self.bart.classify(email)
                predictions.append(pred)
                confidences.append(conf)
                
                # Ensure all categories are represented
                prob_vector = []
                for category in self.categories:
                    prob_vector.append(probs.get(category, 0.0))
                all_probabilities.append(prob_vector)
                
            except Exception as e:
                print(f"   Error processing email: {e}")
                predictions.append(self.categories[0])  # Default
                confidences.append(0.0)
                all_probabilities.append([1.0/len(self.categories)] * len(self.categories))
        
        true_labels = validation_df['category'].values
        accuracy = accuracy_score(true_labels, predictions)
        
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Average Confidence: {np.mean(confidences):.3f} ({np.mean(confidences)*100:.1f}%)")
        
        return {
            'predictions': predictions,
            'probabilities': np.array(all_probabilities),
            'confidences': confidences,
            'accuracy': accuracy
        }
    
    def evaluate_ensemble(self, validation_df, classical_results, bart_results, weight):
        """Evaluate ensemble with given weight for BART."""
        print(f"⚖️  Evaluating ensemble (BART weight: {weight:.2f})...")
        
        # Combine probabilities
        classical_probs = classical_results['probabilities']
        bart_probs = bart_results['probabilities']
        
        ensemble_probs = (1 - weight) * classical_probs + weight * bart_probs
        
        # Get predictions
        ensemble_predictions = []
        ensemble_confidences = []
        
        for prob_vector in ensemble_probs:
            pred_idx = np.argmax(prob_vector)
            predicted_label = self.categories[pred_idx]
            confidence = prob_vector[pred_idx]
            
            ensemble_predictions.append(predicted_label)
            ensemble_confidences.append(confidence)
        
        true_labels = validation_df['category'].values
        accuracy = accuracy_score(true_labels, ensemble_predictions)
        
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Average Confidence: {np.mean(ensemble_confidences):.3f} ({np.mean(ensemble_confidences)*100:.1f}%)")
        
        return {
            'predictions': ensemble_predictions,
            'confidences': ensemble_confidences,
            'accuracy': accuracy,
            'weight': weight
        }
    
    def find_optimal_weight(self, validation_df, classical_results, bart_results):
        """Find optimal ensemble weight."""
        print("🎯 Finding optimal ensemble weight...")
        
        weights = np.arange(0.0, 1.1, 0.1)
        results = []
        
        for weight in weights:
            ensemble_result = self.evaluate_ensemble(
                validation_df, classical_results, bart_results, weight
            )
            results.append(ensemble_result)
        
        # Find best weight
        best_result = max(results, key=lambda x: x['accuracy'])
        
        print(f"\n🏆 OPTIMAL ENSEMBLE RESULTS")
        print("=" * 40)
        print(f"Best BART weight: {best_result['weight']:.2f}")
        print(f"Best accuracy: {best_result['accuracy']:.3f} ({best_result['accuracy']*100:.1f}%)")
        print(f"Best avg confidence: {np.mean(best_result['confidences']):.3f}")
        
        return best_result, results
    
    def analyze_data_sufficiency(self, validation_df):
        """Analyze if we have sufficient data for each category."""
        print("\n📊 DATA SUFFICIENCY ANALYSIS")
        print("=" * 40)
        
        category_counts = validation_df['category'].value_counts()
        total_examples = len(validation_df)
        
        print(f"Total validation examples: {total_examples}")
        print(f"Categories: {len(category_counts)}")
        print(f"Examples per category:")
        
        sufficient = True
        for category, count in category_counts.items():
            percentage = (count / total_examples) * 100
            status = "✅" if count >= 50 else "⚠️" if count >= 20 else "❌"
            print(f"  {status} {category}: {count} examples ({percentage:.1f}%)")
            
            if count < 50:
                sufficient = False
        
        print(f"\nData sufficiency: {'✅ Sufficient' if sufficient else '⚠️ Needs more data'}")
        
        if not sufficient:
            print("\n💡 RECOMMENDATIONS:")
            print("- Generate more training examples for underrepresented categories")
            print("- Use data augmentation techniques")
            print("- Consider synthetic data generation")
            print("- Collect more real-world examples")
        
        return sufficient
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("🚀 BART ENSEMBLE EVALUATION")
        print("=" * 50)
        
        # Load models
        if not self.load_classical_model():
            return False
        
        if not self.init_bart():
            return False
        
        # Generate validation data
        validation_df = self.generate_validation_data(examples_per_category=100)
        
        # Analyze data sufficiency
        self.analyze_data_sufficiency(validation_df)
        
        # Evaluate individual models
        print(f"\n📈 MODEL EVALUATIONS")
        print("=" * 40)
        classical_results = self.evaluate_classical_model(validation_df)
        bart_results = self.evaluate_bart_model(validation_df)
        
        # Find optimal ensemble
        print(f"\n⚖️  ENSEMBLE OPTIMIZATION")
        print("=" * 40)
        best_ensemble, all_results = self.find_optimal_weight(
            validation_df, classical_results, bart_results
        )
        
        # Detailed comparison
        print(f"\n📊 DETAILED COMPARISON")
        print("=" * 40)
        true_labels = validation_df['category'].values
        
        print("\nClassical Model:")
        print(classification_report(true_labels, classical_results['predictions'], 
                                  target_names=self.categories))
        
        print("\nBART Model:")
        print(classification_report(true_labels, bart_results['predictions'], 
                                  target_names=self.categories))
        
        print(f"\nOptimal Ensemble (weight={best_ensemble['weight']:.2f}):")
        print(classification_report(true_labels, best_ensemble['predictions'], 
                                  target_names=self.categories))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"bart_evaluation_results_{timestamp}.json"
        
        summary = {
            'timestamp': timestamp,
            'validation_size': len(validation_df),
            'classical_accuracy': classical_results['accuracy'],
            'bart_accuracy': bart_results['accuracy'],
            'optimal_weight': best_ensemble['weight'],
            'optimal_accuracy': best_ensemble['accuracy'],
            'improvement': best_ensemble['accuracy'] - classical_results['accuracy'],
            'config': self.config
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ EVALUATION COMPLETE")
        print("=" * 40)
        print(f"Results saved to: {results_file}")
        print(f"Classical accuracy: {classical_results['accuracy']:.3f}")
        print(f"BART accuracy: {bart_results['accuracy']:.3f}")
        print(f"Optimal ensemble: {best_ensemble['accuracy']:.3f} (weight={best_ensemble['weight']:.2f})")
        print(f"Improvement: {best_ensemble['accuracy'] - classical_results['accuracy']:+.3f}")
        
        return True

def main():
    """Main evaluation function."""
    evaluator = BartEnsembleEvaluator()
    success = evaluator.run_full_evaluation()
    
    if not success:
        print("❌ Evaluation failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())