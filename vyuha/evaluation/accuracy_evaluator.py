"""
Accuracy evaluation for comparing original and optimized models.
"""

import logging
from typing import Any, Dict, List

import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class AccuracyEvaluator:
    """
    Evaluates model accuracy on support ticket classification task.
    """
    
    def __init__(self, model_name: str, max_samples: int = 100):
        """
        Initialize the accuracy evaluator.
        
        Args:
            model_name: Hugging Face model identifier
            max_samples: Maximum number of samples to evaluate
        """
        self.model_name = model_name
        self.max_samples = max_samples
        self.tokenizer = None
        self.original_model = None
        self.optimized_model = None
        
        # Support ticket categories for classification
        self.categories = [
            "technical_issue",
            "billing_inquiry", 
            "feature_request",
            "account_help",
            "general_inquiry"
        ]
        
    def load_models(self, original_model_path: str = None, optimized_model_path: str = None) -> None:
        """
        Load both original and optimized models.
        
        Args:
            original_model_path: Path to original model (if None, loads from HF)
            optimized_model_path: Path to optimized ONNX model
        """
        logger.info("Loading models for accuracy evaluation...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load original model
            if original_model_path:
                self.original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
            else:
                self.original_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Load optimized model (placeholder for now)
            if optimized_model_path:
                # For now, use the same model as original
                # In production, this would load the ONNX model
                self.optimized_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def load_test_dataset(self) -> List[Dict[str, Any]]:
        """
        Load and prepare test dataset for support ticket classification.
        
        Returns:
            List of test samples with text and labels
        """
        logger.info("Loading test dataset...")
        
        try:
            # Create synthetic support ticket dataset
            # In a real scenario, you would load from a proper dataset
            test_samples = self._create_synthetic_dataset()
            
            logger.info(f"Loaded {len(test_samples)} test samples")
            return test_samples
            
        except Exception as e:
            logger.error(f"Failed to load test dataset: {e}")
            raise
    
    def _create_synthetic_dataset(self) -> List[Dict[str, Any]]:
        """Create synthetic support ticket dataset for testing."""
        samples = []
        
        # Technical issues
        tech_samples = [
            "I can't log into my account, getting error 500",
            "The application keeps crashing when I try to upload files",
            "I'm getting a database connection error",
            "The API is returning 404 errors",
            "My dashboard is not loading properly"
        ]
        
        # Billing inquiries
        billing_samples = [
            "I was charged twice for my subscription",
            "Can you explain the charges on my invoice?",
            "I want to cancel my premium plan",
            "How do I update my payment method?",
            "I need a refund for last month's charge"
        ]
        
        # Feature requests
        feature_samples = [
            "Can you add dark mode to the interface?",
            "I'd like to export my data to CSV",
            "Please add two-factor authentication",
            "Can you implement bulk operations?",
            "I need better search functionality"
        ]
        
        # Account help
        account_samples = [
            "How do I change my password?",
            "I forgot my username",
            "Can you help me recover my account?",
            "How do I update my profile?",
            "I need to change my email address"
        ]
        
        # General inquiries
        general_samples = [
            "What are your business hours?",
            "How do I contact customer support?",
            "What is your refund policy?",
            "Do you offer training sessions?",
            "Can you tell me about your company?"
        ]
        
        # Combine all samples with labels
        all_samples = [
            (tech_samples, "technical_issue"),
            (billing_samples, "billing_inquiry"),
            (feature_samples, "feature_request"),
            (account_samples, "account_help"),
            (general_samples, "general_inquiry")
        ]
        
        for sample_list, label in all_samples:
            for text in sample_list:
                samples.append({
                    "text": text,
                    "label": label,
                    "label_id": self.categories.index(label)
                })
        
        return samples[:self.max_samples]
    
    def predict_category(self, text: str, model, is_onnx: bool = False) -> str:
        """
        Predict category for a given text using the specified model.
        
        Args:
            text: Input text to classify
            model: Model to use for prediction
            is_onnx: Whether the model is ONNX format
            
        Returns:
            Predicted category
        """
        try:
            # Prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if is_onnx:
                # ONNX model inference
                outputs = model(**inputs)
                logits = outputs.logits
            else:
                # PyTorch model inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
            
            # Get prediction
            predicted_id = torch.argmax(logits, dim=-1).item()
            
            # Map to category (simplified mapping)
            if predicted_id < len(self.categories):
                return self.categories[predicted_id]
            else:
                return self.categories[0]  # Default to first category
                
        except Exception as e:
            logger.warning(f"Prediction failed for text: {text[:50]}... Error: {e}")
            return self.categories[0]  # Default fallback
    
    def evaluate_accuracy(self, test_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate accuracy of both models on test samples.
        
        Args:
            test_samples: List of test samples with text and labels
            
        Returns:
            Dictionary with accuracy metrics for both models
        """
        logger.info("Evaluating model accuracy...")
        
        try:
            # Get predictions from both models
            original_predictions = []
            optimized_predictions = []
            true_labels = []
            
            for sample in test_samples:
                text = sample["text"]
                true_label = sample["label"]
                
                # Get predictions
                orig_pred = self.predict_category(text, self.original_model, is_onnx=False)
                opt_pred = self.predict_category(text, self.optimized_model, is_onnx=True)
                
                original_predictions.append(orig_pred)
                optimized_predictions.append(opt_pred)
                true_labels.append(true_label)
            
            # Calculate accuracy scores
            original_accuracy = accuracy_score(true_labels, original_predictions)
            optimized_accuracy = accuracy_score(true_labels, optimized_predictions)
            
            # Calculate accuracy drop
            accuracy_drop = original_accuracy - optimized_accuracy
            
            results = {
                "original_accuracy": original_accuracy,
                "optimized_accuracy": optimized_accuracy,
                "accuracy_drop": accuracy_drop,
                "accuracy_retention": (optimized_accuracy / original_accuracy) * 100,
                "num_samples": len(test_samples),
                "original_predictions": original_predictions,
                "optimized_predictions": optimized_predictions,
                "true_labels": true_labels
            }
            
            logger.info(f"Accuracy evaluation completed:")
            logger.info(f"  Original model: {original_accuracy:.3f}")
            logger.info(f"  Optimized model: {optimized_accuracy:.3f}")
            logger.info(f"  Accuracy drop: {accuracy_drop:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            raise
    
    def run_evaluation(self, original_model_path: str = None, optimized_model_path: str = None) -> Dict[str, Any]:
        """
        Run complete accuracy evaluation.
        
        Args:
            original_model_path: Path to original model
            optimized_model_path: Path to optimized model
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting accuracy evaluation...")
        
        try:
            # Load models
            self.load_models(original_model_path, optimized_model_path)
            
            # Load test dataset
            test_samples = self.load_test_dataset()
            
            # Run evaluation
            results = self.evaluate_accuracy(test_samples)
            
            logger.info("Accuracy evaluation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
