#!/usr/bin/env python3
"""
🚀 PRODUCTION EMAIL NLP API
===========================
Production-ready API for email classification using the improved model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import numpy as np
import pickle
import glob
import os
import json
import logging
from datetime import datetime
import time
from performance_optimizer import PerformanceOptimizer
from components.bart_classifier import BartZeroShotClassifier
from components.enhanced_bart_classifier import EnhancedBartClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per minute", "1000 per hour"]
)

class ProductionEmailClassifier:
    def __init__(self, config_dir="config"):
        self.optimizer = PerformanceOptimizer()
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.is_loaded = False
        self.config = self.load_config(config_dir)
        self.bart = None
        self.load_model()
        self._init_bart()
    
    def load_config(self, config_dir):
        """Load configuration files."""
        try:
            with open(os.path.join(config_dir, "model_config.json"), 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def load_model(self):
        """Load the production model."""
        try:
            models_dir = "models"
            
            # Find model files
            model_files = glob.glob(os.path.join(models_dir, "final_calibrated_model_*.pkl"))
            scaler_files = glob.glob(os.path.join(models_dir, "final_scaler_*.pkl"))
            encoder_files = glob.glob(os.path.join(models_dir, "final_label_encoder_*.pkl"))
            
            if not all([model_files, scaler_files, encoder_files]):
                logger.error("❌ Model files not found!")
                return False
            
            # Load latest files
            latest_model = max(model_files, key=os.path.getctime)
            latest_scaler = max(scaler_files, key=os.path.getctime)
            latest_encoder = max(encoder_files, key=os.path.getctime)
            
            logger.info(f"📂 Loading production model:")
            logger.info(f"   Model: {latest_model}")
            logger.info(f"   Scaler: {latest_scaler}")
            logger.info(f"   Encoder: {latest_encoder}")
            
            with open(latest_model, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(latest_scaler, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(latest_encoder, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.is_loaded = True
            logger.info("✅ Production model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

    def _init_bart(self):
        """Initialize optional BART zero-shot classifier based on config."""
        try:
            bart_cfg = (self.config or {}).get("bart", {})
            if not bart_cfg or not bart_cfg.get("enabled", False):
                logger.info("BART is disabled in config")
                return
            categories = (self.config or {}).get("categories", [])
            if not categories:
                logger.warning("No categories found for BART; disabling BART")
                return
            
            # Choose enhanced or standard BART classifier
            use_enhanced = bart_cfg.get("use_enhanced_classifier", True)
            
            if use_enhanced:
                self.bart = EnhancedBartClassifier(
                    model_name=bart_cfg.get("model_name", "facebook/bart-large-mnli"),
                    device=bart_cfg.get("device", -1),
                    use_enhanced_labels=bart_cfg.get("use_enhanced_labels", True),
                    use_preprocessing=bart_cfg.get("use_preprocessing", True),
                )
                logger.info("✅ Enhanced BART classifier initialized")
            else:
                self.bart = BartZeroShotClassifier(
                    model_name=bart_cfg.get("model_name", "facebook/bart-large-mnli"),
                    candidate_labels=categories,
                    multi_label=bart_cfg.get("multi_label", False),
                    device=bart_cfg.get("device", -1),
                    hypothesis_template=bart_cfg.get("hypothesis_template", "This text is about {}."),
                )
                logger.info("✅ Standard BART classifier initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BART classifier: {e}")
    
    def classify_email(self, email_text):
        """Classify email with production-ready error handling."""
        if not self.is_loaded:
            return {
                'error': 'Model not loaded',
                'category': None,
                'confidence': 0.0,
                'probabilities': {},
                'model_version': 'final_improved_calibrated',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Extract features
            features = self.optimizer.extract_optimized_features([email_text])
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get prediction and confidence
            predicted_category = self.label_encoder.classes_[prediction]
            confidence = max(probabilities)
            
            # Check confidence threshold
            threshold = self.config.get('confidence_threshold', 0.7)
            low_confidence = confidence < threshold
            
            # Create probabilities dictionary
            prob_dict = {}
            for i, category in enumerate(self.label_encoder.classes_):
                prob_dict[category] = float(probabilities[i])
            
            # Optional BART fusion when low confidence or enabled ensemble
            bart_cfg = (self.config or {}).get("bart", {})
            use_bart = self.bart is not None and (bart_cfg.get("enabled", False))
            if use_bart and (low_confidence or bart_cfg.get("always_ensemble", True)):
                try:
                    bart_label, bart_conf, bart_scores = self.bart.classify(
                        email_text,
                        candidate_labels=list(self.label_encoder.classes_),
                        multi_label=bart_cfg.get("multi_label", False),
                        hypothesis_template=bart_cfg.get("hypothesis_template", "This text is about {}."),
                    )
                    weight = float(bart_cfg.get("weight", 0.5))
                    # Normalize classical probs to dict with all labels
                    combined = {}
                    for label in self.label_encoder.classes_:
                        classic_p = prob_dict.get(label, 0.0)
                        bart_p = bart_scores.get(label, 0.0)
                        combined[label] = (1.0 - weight) * classic_p + weight * bart_p
                    # Re-select predicted label and confidence
                    predicted_category = max(combined, key=combined.get)
                    confidence = float(combined[predicted_category])
                    prob_dict = {k: float(v) for k, v in combined.items()}
                except Exception as bart_err:
                    logger.error(f"BART fusion failed: {bart_err}")
                    # proceed with classic outputs

            if low_confidence and not use_bart:
                predicted_category = 'Uncertain'

            return {
                'category': predicted_category,
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'model_version': 'final_improved_calibrated',
                'accuracy_target': '99.7%',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error classifying email: {e}")
            return {
                'error': str(e),
                'category': None,
                'confidence': 0.0,
                'probabilities': {},
                'model_version': 'final_improved_calibrated',
                'timestamp': datetime.now().isoformat()
            }

# Initialize classifier
classifier = ProductionEmailClassifier()

@app.route('/health', methods=['GET'])
@limiter.limit("10 per minute")
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier.is_loaded,
        'model_version': 'final_improved_calibrated',
        'accuracy_target': '99.7%',
        'timestamp': datetime.now().isoformat(),
        'uptime': time.time()
    })

@app.route('/classify', methods=['POST'])
@limiter.limit("100 per minute")
def classify_email():
    """Classify a single email."""
    try:
        data = request.get_json()
        
        if not data or 'email' not in data:
            return jsonify({'error': 'Email text is required'}), 400
        
        email_text = data['email'].strip()
        
        if not email_text:
            return jsonify({'error': 'Email text cannot be empty'}), 400
        
        if len(email_text) > 10000:
            return jsonify({'error': 'Email text too long (max 10000 characters)'}), 400
        
        # Classify email
        result = classifier.classify_email(email_text)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in classify endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_classify', methods=['POST'])
@limiter.limit("10 per minute")
def batch_classify():
    """Classify multiple emails."""
    try:
        data = request.get_json()
        
        if not data or 'emails' not in data:
            return jsonify({'error': 'Emails list is required'}), 400
        
        emails = data['emails']
        
        if not isinstance(emails, list):
            return jsonify({'error': 'Emails must be a list'}), 400
        
        if len(emails) > 100:
            return jsonify({'error': 'Maximum 100 emails per batch'}), 400
        
        results = []
        
        for i, email_text in enumerate(emails):
            if not isinstance(email_text, str):
                results.append({
                    'index': i,
                    'error': 'Email must be a string',
                    'category': None,
                    'confidence': 0.0
                })
                continue
            
            email_text = email_text.strip()
            
            if not email_text:
                results.append({
                    'index': i,
                    'error': 'Email text cannot be empty',
                    'category': None,
                    'confidence': 0.0
                })
                continue
            
            # Classify email
            result = classifier.classify_email(email_text)
            
            if 'error' in result:
                results.append({
                    'index': i,
                    'error': result['error'],
                    'category': None,
                    'confidence': 0.0
                })
            else:
                results.append({
                    'index': i,
                    'category': result['category'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'model_version': 'final_improved_calibrated',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error in batch_classify endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
@limiter.limit("10 per minute")
def model_info():
    """Get model information."""
    return jsonify({
        'model_version': 'final_improved_calibrated',
        'accuracy_target': '99.7%',
        'improvements': [
            'Work access requests correctly classified as Work',
            'Professional follow-up emails correctly classified as Work',
            'Clear distinction between Personal and Work emails',
            'Urgent emails correctly classified (not as Spam)',
            'Standard emails correctly classified (not as Spam)',
            '91.7% accuracy on previously problematic test cases'
        ],
        'categories': list(classifier.label_encoder.classes_) if classifier.label_encoder else [],
        'model_loaded': classifier.is_loaded,
        'confidence_threshold': classifier.config.get('confidence_threshold', 0.7),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == "__main__":
    # Load API configuration
    try:
        with open("config/api_config.json", 'r') as f:
            api_config = json.load(f)
    except:
        api_config = {
            "host": "0.0.0.0",
            "port": 8080,
            "debug": False
        }
    
    print("🚀 STARTING PRODUCTION EMAIL NLP API")
    print("=" * 50)
    print(f"🌐 Host: {api_config.get('host', '0.0.0.0')}")
    print(f"🔌 Port: {api_config.get('port', 8080)}")
    print(f"🔧 Debug: {api_config.get('debug', False)}")
    print(f"📊 Model Version: final_improved_calibrated")
    print(f"🎯 Target Accuracy: 99.7%")
    print(f"✨ Production Ready!")
    print("=" * 50)
    
    app.run(
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8080),
        debug=api_config.get('debug', False)
    )
