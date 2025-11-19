#!/usr/bin/env python3
"""
🚀 ENHANCED BART COMPARISON
============================
Compare standard BART vs enhanced BART performance on email classification.
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
from components.enhanced_bart_classifier import EnhancedBartClassifier

class EnhancedBartComparison:
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.optimizer = PerformanceOptimizer()
        self.categories = self.config.get('categories', [])
        
    def load_config(self, config_path: str = None):
        """Load configuration from robust set of candidate paths."""
        base_dir = os.path.dirname(__file__)
        candidates = []
        if config_path:
            candidates.append(config_path)
        candidates.extend([
            os.path.normpath(os.path.join(base_dir, '..', 'config', 'model_config.json')),
            os.path.normpath(os.path.join(base_dir, '..', '..', 'production', 'config', 'model_config.json')),
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
        print("❌ Config not found")
        return {}
    
    def generate_test_examples(self):
        """Generate specific test examples that highlight classification challenges."""
        test_examples = [
            # Spam examples (should be easier to identify)
            ("CONGRATULATIONS! You've won $1,000,000! Click here NOW!", "Spam"),
            ("FREE VIAGRA! 80% discount! Order now before it expires!", "Spam"),
            ("Your account has been suspended. Click here to verify.", "Spam"),
            ("URGENT: Netflix payment failed. Update your card info.", "Spam"),
            ("Limited time offer! Get 95% off everything! Act fast!", "Spam"),
            
            # Work examples
            ("Team meeting scheduled for tomorrow at 2 PM in conference room A.", "Work"),
            ("Please find attached the quarterly budget report for review.", "Work"),
            ("Following up on our discussion about the project timeline.", "Work"),
            ("Performance review meeting scheduled for next week.", "Work"),
            ("Client presentation needs to be ready by Friday.", "Work"),
            
            # Personal examples
            ("Hey! Are you free this weekend? Want to grab coffee?", "Personal"),
            ("Thanks for the birthday gift! I love it so much!", "Personal"),
            ("How was your vacation? Can't wait to hear about it!", "Personal"),
            ("Movie night at my place this Saturday. Bring snacks!", "Personal"),
            ("Happy anniversary! Hope you have a wonderful day!", "Personal"),
            
            # Urgent examples
            ("URGENT: Server down. All systems offline. Need immediate help.", "Urgent"),
            ("CRITICAL: Data breach detected. Security team needed now.", "Urgent"),
            ("EMERGENCY: Production line stopped. Engineers required immediately.", "Urgent"),
            ("URGENT: CEO requested immediate meeting. All executives needed.", "Urgent"),
            ("CRITICAL: Website crashed. All transactions are failing.", "Urgent"),
            
            # Standard examples
            ("Please find attached the requested information.", "Standard"),
            ("Thank you for your time and consideration.", "Standard"),
            ("I hope this email finds you well.", "Standard"),
            ("Following up on our previous discussion.", "Standard"),
            ("Please let me know if you need any clarification.", "Standard"),
        ]
        
        return test_examples
    
    def evaluate_bart_variant(self, bart_classifier, test_examples, variant_name):
        """Evaluate a specific BART variant."""
        print(f"\n🤖 Evaluating {variant_name}...")
        
        predictions = []
        confidences = []
        true_labels = []
        
        for email_text, true_label in test_examples:
            true_labels.append(true_label)
            
            try:
                pred, conf, probs = bart_classifier.classify(email_text, self.categories)
                predictions.append(pred)
                confidences.append(conf)
                
                # Debug problematic cases
                if pred != true_label:
                    print(f"   ❌ '{email_text[:50]}...' → Predicted: {pred}, True: {true_label}, Conf: {conf:.3f}")
                
            except Exception as e:
                print(f"   Error: {e}")
                predictions.append(self.categories[0])
                confidences.append(0.0)
        
        accuracy = accuracy_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Average Confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
        
        return {
            'variant': variant_name,
            'predictions': predictions,
            'confidences': confidences,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence
        }
    
    def run_comparison(self):
        """Run complete comparison between BART variants."""
        print("🚀 ENHANCED BART COMPARISON")
        print("=" * 50)
        
        # Generate test examples
        test_examples = self.generate_test_examples()
        print(f"📊 Generated {len(test_examples)} test examples")
        
        bart_config = self.config.get("bart", {})
        model_name = bart_config.get("model_name", "facebook/bart-large-mnli")
        device = bart_config.get("device", -1)
        
        # Test Standard BART
        print("\n🤖 Initializing Standard BART...")
        try:
            standard_bart = BartZeroShotClassifier(
                model_name=model_name,
                candidate_labels=self.categories,
                device=device,
                hypothesis_template="This text is about {}."
            )
            standard_results = self.evaluate_bart_variant(
                standard_bart, test_examples, "Standard BART"
            )
        except Exception as e:
            print(f"❌ Standard BART failed: {e}")
            standard_results = None
        
        # Test Enhanced BART
        print("\n🚀 Initializing Enhanced BART...")
        try:
            enhanced_bart = EnhancedBartClassifier(
                model_name=model_name,
                device=device,
                use_enhanced_labels=True,
                use_preprocessing=True
            )
            enhanced_results = self.evaluate_bart_variant(
                enhanced_bart, test_examples, "Enhanced BART"
            )
        except Exception as e:
            print(f"❌ Enhanced BART failed: {e}")
            enhanced_results = None
        
        # Compare results
        print(f"\n📊 COMPARISON RESULTS")
        print("=" * 40)
        
        if standard_results and enhanced_results:
            print(f"Standard BART:")
            print(f"  Accuracy: {standard_results['accuracy']:.3f} ({standard_results['accuracy']*100:.1f}%)")
            print(f"  Avg Confidence: {standard_results['avg_confidence']:.3f}")
            
            print(f"\nEnhanced BART:")
            print(f"  Accuracy: {enhanced_results['accuracy']:.3f} ({enhanced_results['accuracy']*100:.1f}%)")
            print(f"  Avg Confidence: {enhanced_results['avg_confidence']:.3f}")
            
            improvement = enhanced_results['accuracy'] - standard_results['accuracy']
            conf_improvement = enhanced_results['avg_confidence'] - standard_results['avg_confidence']
            
            print(f"\nImprovement:")
            print(f"  Accuracy: {improvement:+.3f} ({improvement*100:+.1f}%)")
            print(f"  Confidence: {conf_improvement:+.3f}")
            
            # Detailed analysis
            print(f"\n📈 DETAILED ANALYSIS")
            print("=" * 40)
            
            true_labels = [label for _, label in test_examples]
            
            print("\nStandard BART Classification Report:")
            print(classification_report(true_labels, standard_results['predictions'], 
                                      target_names=self.categories, zero_division=0))
            
            print("\nEnhanced BART Classification Report:")
            print(classification_report(true_labels, enhanced_results['predictions'], 
                                      target_names=self.categories, zero_division=0))
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"enhanced_bart_comparison_{timestamp}.json"
            
            summary = {
                'timestamp': timestamp,
                'test_size': len(test_examples),
                'standard_accuracy': standard_results['accuracy'],
                'enhanced_accuracy': enhanced_results['accuracy'],
                'accuracy_improvement': improvement,
                'standard_confidence': standard_results['avg_confidence'],
                'enhanced_confidence': enhanced_results['avg_confidence'],
                'confidence_improvement': conf_improvement,
                'config': self.config
            }
            
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n✅ COMPARISON COMPLETE")
            print(f"Results saved to: {results_file}")
            
            if improvement > 0:
                print(f"🎉 Enhanced BART shows {improvement*100:.1f}% accuracy improvement!")
                print("💡 Recommendation: Enable enhanced BART in production config")
            else:
                print(f"⚠️  Enhanced BART shows {improvement*100:.1f}% accuracy change")
                print("💡 Consider further tuning or stick with standard BART")
            
            return True
        
        else:
            print("❌ Comparison failed due to initialization errors")
            return False

def main():
    """Main comparison function."""
    comparison = EnhancedBartComparison()
    success = comparison.run_comparison()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
