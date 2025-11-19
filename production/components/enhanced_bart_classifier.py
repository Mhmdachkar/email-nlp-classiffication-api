#!/usr/bin/env python3
"""
🚀 ENHANCED BART CLASSIFIER
============================
Improved BART implementation with better hypothesis templates, 
label mapping, and preprocessing for email classification.
"""

import re
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


class EnhancedBartClassifier:
    """Enhanced BART classifier optimized for email categorization."""
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: int = -1,
        use_enhanced_labels: bool = True,
        use_preprocessing: bool = True,
    ):
        if pipeline is None:
            raise ImportError("transformers is not installed")
        
        self.model_name = model_name
        self.device = device
        self.use_enhanced_labels = use_enhanced_labels
        self.use_preprocessing = use_preprocessing
        
        # Initialize the pipeline
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device,
        )
        
        # Enhanced label mappings for better semantic understanding
        self.label_mappings = {
            "Personal": {
                "enhanced_labels": [
                    "personal communication",
                    "informal conversation", 
                    "social interaction",
                    "friendly message",
                    "casual correspondence"
                ],
                "hypothesis_templates": [
                    "This is a personal {} between friends or family.",
                    "This message is {} and informal.",
                    "This text contains {} social communication."
                ]
            },
            "Work": {
                "enhanced_labels": [
                    "business communication",
                    "professional correspondence",
                    "work-related message",
                    "corporate communication",
                    "business meeting discussion"
                ],
                "hypothesis_templates": [
                    "This is a professional {} related to work.",
                    "This message discusses {} business matters.",
                    "This text contains {} workplace communication."
                ]
            },
            "Urgent": {
                "enhanced_labels": [
                    "urgent request",
                    "emergency communication",
                    "immediate action required",
                    "critical alert",
                    "time-sensitive message"
                ],
                "hypothesis_templates": [
                    "This message requires {} immediate attention.",
                    "This is an {} communication needing urgent action.",
                    "This text indicates {} critical importance."
                ]
            },
            "Spam": {
                "enhanced_labels": [
                    "promotional email",
                    "unsolicited advertisement",
                    "marketing message",
                    "fraudulent communication",
                    "suspicious offer"
                ],
                "hypothesis_templates": [
                    "This is {} promotional or unwanted content.",
                    "This message contains {} suspicious offers.",
                    "This text is {} unsolicited marketing."
                ]
            },
            "Standard": {
                "enhanced_labels": [
                    "routine communication",
                    "formal correspondence",
                    "standard business message",
                    "regular notification",
                    "informational update"
                ],
                "hypothesis_templates": [
                    "This is {} routine business correspondence.",
                    "This message is {} standard communication.",
                    "This text contains {} formal information."
                ]
            }
        }
    
    def preprocess_email(self, text: str) -> str:
        """Preprocess email text to improve BART understanding."""
        if not self.use_preprocessing:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize common email patterns
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        text = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', '[MONEY]', text)
        
        # Enhance context for BART
        # Add contextual markers for better understanding
        if any(word in text.lower() for word in ['urgent', 'critical', 'emergency', 'immediate']):
            text = f"URGENT: {text}"
        elif any(word in text.lower() for word in ['free', 'offer', 'click', 'win', 'prize']):
            text = f"PROMOTIONAL: {text}"
        elif any(word in text.lower() for word in ['meeting', 'project', 'team', 'work']):
            text = f"BUSINESS: {text}"
        elif any(word in text.lower() for word in ['hey', 'hi', 'thanks', 'love']):
            text = f"PERSONAL: {text}"
        
        return text
    
    def classify_with_enhanced_labels(
        self, 
        text: str, 
        original_categories: List[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Classify using enhanced semantic labels."""
        
        # Preprocess the text
        processed_text = self.preprocess_email(text)
        
        # Try multiple strategies and pick the best result
        strategies = []
        
        # Strategy 1: Enhanced labels with context
        if self.use_enhanced_labels:
            enhanced_labels = []
            for category in original_categories:
                if category in self.label_mappings:
                    enhanced_labels.extend(self.label_mappings[category]["enhanced_labels"])
            
            if enhanced_labels:
                result1 = self._pipeline(
                    processed_text,
                    enhanced_labels,
                    hypothesis_template="This message is {}."
                )
                
                # Map back to original categories
                label_to_category = {}
                for category in original_categories:
                    if category in self.label_mappings:
                        for label in self.label_mappings[category]["enhanced_labels"]:
                            label_to_category[label] = category
                
                # Aggregate scores by original category
                category_scores = {cat: 0.0 for cat in original_categories}
                for label, score in zip(result1["labels"], result1["scores"]):
                    if label in label_to_category:
                        category = label_to_category[label]
                        category_scores[category] += score
                
                # Normalize scores
                total_score = sum(category_scores.values())
                if total_score > 0:
                    category_scores = {k: v/total_score for k, v in category_scores.items()}
                
                best_category = max(category_scores, key=category_scores.get)
                strategies.append((best_category, category_scores[best_category], category_scores))
        
        # Strategy 2: Original labels with better hypothesis
        result2 = self._pipeline(
            processed_text,
            original_categories,
            hypothesis_template="This email message is about {}."
        )
        
        scores2 = {label: score for label, score in zip(result2["labels"], result2["scores"])}
        strategies.append((result2["labels"][0], result2["scores"][0], scores2))
        
        # Strategy 3: Context-aware hypothesis templates
        best_scores = {}
        for category in original_categories:
            if category in self.label_mappings:
                templates = self.label_mappings[category]["hypothesis_templates"]
                category_scores = []
                
                for template in templates:
                    try:
                        result = self._pipeline(
                            processed_text,
                            [category.lower()],
                            hypothesis_template=template
                        )
                        category_scores.append(result["scores"][0])
                    except:
                        continue
                
                if category_scores:
                    best_scores[category] = max(category_scores)
                else:
                    best_scores[category] = 0.0
        
        if best_scores:
            # Normalize
            total = sum(best_scores.values())
            if total > 0:
                best_scores = {k: v/total for k, v in best_scores.items()}
                best_cat = max(best_scores, key=best_scores.get)
                strategies.append((best_cat, best_scores[best_cat], best_scores))
        
        # Choose the strategy with highest confidence
        if strategies:
            best_strategy = max(strategies, key=lambda x: x[1])
            return best_strategy[0], best_strategy[1], best_strategy[2]
        
        # Fallback to simple classification
        return self.classify_simple(text, original_categories)
    
    def classify_simple(
        self, 
        text: str, 
        categories: List[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Simple classification fallback."""
        processed_text = self.preprocess_email(text)
        
        result = self._pipeline(
            processed_text,
            categories,
            hypothesis_template="This email is {}."
        )
        
        scores = {label: score for label, score in zip(result["labels"], result["scores"])}
        return result["labels"][0], result["scores"][0], scores
    
    def classify(
        self, 
        text: str, 
        candidate_labels: List[str]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Main classification method."""
        return self.classify_with_enhanced_labels(text, candidate_labels)
    
    def classify_batch(
        self, 
        texts: List[str], 
        candidate_labels: List[str]
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Classify multiple texts."""
        return [self.classify(text, candidate_labels) for text in texts]


def create_enhanced_bart_classifier(config: Dict) -> EnhancedBartClassifier:
    """Factory function to create enhanced BART classifier from config."""
    bart_config = config.get("bart", {})
    
    return EnhancedBartClassifier(
        model_name=bart_config.get("model_name", "facebook/bart-large-mnli"),
        device=bart_config.get("device", -1),
        use_enhanced_labels=bart_config.get("use_enhanced_labels", True),
        use_preprocessing=bart_config.get("use_preprocessing", True),
    )
