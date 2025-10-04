"""
Tests for the accuracy evaluator module.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from vyuha.evaluation.accuracy_evaluator import AccuracyEvaluator


class TestAccuracyEvaluator:
    """Test cases for AccuracyEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_name = "microsoft/DialoGPT-small"
        self.max_samples = 10
        self.evaluator = AccuracyEvaluator(self.model_name, self.max_samples)
    
    def test_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator.model_name == self.model_name
        assert self.evaluator.max_samples == self.max_samples
        assert self.evaluator.tokenizer is None
        assert self.evaluator.original_model is None
        assert self.evaluator.optimized_model is None
        assert len(self.evaluator.categories) == 5
    
    def test_categories(self):
        """Test category definitions."""
        expected_categories = [
            "technical_issue",
            "billing_inquiry", 
            "feature_request",
            "account_help",
            "general_inquiry"
        ]
        assert self.evaluator.categories == expected_categories
    
    @patch('vyuha.evaluation.accuracy_evaluator.AutoTokenizer')
    @patch('vyuha.evaluation.accuracy_evaluator.AutoModelForCausalLM')
    @patch('vyuha.evaluation.accuracy_evaluator.ORTModelForCausalLM')
    def test_load_models_success(self, mock_ort_class, mock_model_class, mock_tokenizer_class):
        """Test successful model loading."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock models
        mock_original_model = Mock()
        mock_optimized_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_original_model
        mock_ort_class.from_pretrained.return_value = mock_optimized_model
        
        # Test
        self.evaluator.load_models()
        
        # Assertions
        assert self.evaluator.tokenizer == mock_tokenizer
        assert self.evaluator.original_model == mock_original_model
        assert self.evaluator.optimized_model == mock_optimized_model
    
    @patch('vyuha.evaluation.accuracy_evaluator.AutoTokenizer')
    def test_load_models_failure(self, mock_tokenizer_class):
        """Test model loading failure."""
        # Mock failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
        
        # Test
        with pytest.raises(Exception, match="Network error"):
            self.evaluator.load_models()
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        samples = self.evaluator._create_synthetic_dataset()
        
        # Assertions
        assert len(samples) <= self.max_samples
        assert all("text" in sample for sample in samples)
        assert all("label" in sample for sample in samples)
        assert all("label_id" in sample for sample in samples)
        assert all(sample["label"] in self.evaluator.categories for sample in samples)
    
    def test_load_test_dataset(self):
        """Test test dataset loading."""
        samples = self.evaluator.load_test_dataset()
        
        # Assertions
        assert isinstance(samples, list)
        assert len(samples) > 0
        assert all("text" in sample for sample in samples)
        assert all("label" in sample for sample in samples)
        assert all("label_id" in sample for sample in samples)
    
    @patch('vyuha.evaluation.accuracy_evaluator.AutoTokenizer')
    @patch('vyuha.evaluation.accuracy_evaluator.AutoModelForCausalLM')
    def test_predict_category_pytorch(self, mock_model_class, mock_tokenizer_class):
        """Test category prediction with PyTorch model."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock()
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_logits = Mock()
        mock_logits.item.return_value = 0
        mock_outputs = Mock()
        mock_outputs.logits = mock_logits
        
        mock_model = Mock()
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Load models
        self.evaluator.load_models()
        
        # Test prediction
        text = "I need help with my account"
        prediction = self.evaluator.predict_category(text, self.evaluator.original_model, is_onnx=False)
        
        # Assertions
        assert prediction in self.evaluator.categories
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
    
    @patch('vyuha.evaluation.accuracy_evaluator.AutoTokenizer')
    @patch('vyuha.evaluation.accuracy_evaluator.ORTModelForCausalLM')
    def test_predict_category_onnx(self, mock_ort_class, mock_tokenizer_class):
        """Test category prediction with ONNX model."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock()
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock ONNX model
        mock_logits = Mock()
        mock_logits.item.return_value = 1
        mock_outputs = Mock()
        mock_outputs.logits = mock_logits
        
        mock_ort_model = Mock()
        mock_ort_model.return_value = mock_outputs
        mock_ort_class.from_pretrained.return_value = mock_ort_model
        
        # Load models
        self.evaluator.load_models()
        
        # Test prediction
        text = "I need help with billing"
        prediction = self.evaluator.predict_category(text, self.evaluator.optimized_model, is_onnx=True)
        
        # Assertions
        assert prediction in self.evaluator.categories
        mock_tokenizer.assert_called_once()
        mock_ort_model.assert_called_once()
    
    @patch('vyuha.evaluation.accuracy_evaluator.AutoTokenizer')
    @patch('vyuha.evaluation.accuracy_evaluator.AutoModelForCausalLM')
    @patch('vyuha.evaluation.accuracy_evaluator.ORTModelForCausalLM')
    def test_evaluate_accuracy(self, mock_ort_class, mock_model_class, mock_tokenizer_class):
        """Test accuracy evaluation."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock()
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock models
        mock_original_model = Mock()
        mock_optimized_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_original_model
        mock_ort_class.from_pretrained.return_value = mock_optimized_model
        
        # Mock predictions
        mock_logits = Mock()
        mock_logits.item.return_value = 0
        mock_outputs = Mock()
        mock_outputs.logits = mock_logits
        
        mock_original_model.return_value = mock_outputs
        mock_optimized_model.return_value = mock_outputs
        
        # Load models
        self.evaluator.load_models()
        
        # Create test samples
        test_samples = [
            {"text": "I need help", "label": "technical_issue", "label_id": 0},
            {"text": "Billing question", "label": "billing_inquiry", "label_id": 1}
        ]
        
        # Test evaluation
        results = self.evaluator.evaluate_accuracy(test_samples)
        
        # Assertions
        assert "original_accuracy" in results
        assert "optimized_accuracy" in results
        assert "accuracy_drop" in results
        assert "accuracy_retention" in results
        assert "num_samples" in results
        assert results["num_samples"] == len(test_samples)
        assert 0 <= results["original_accuracy"] <= 1
        assert 0 <= results["optimized_accuracy"] <= 1
    
    @patch('vyuha.evaluation.accuracy_evaluator.AutoTokenizer')
    @patch('vyuha.evaluation.accuracy_evaluator.AutoModelForCausalLM')
    @patch('vyuha.evaluation.accuracy_evaluator.ORTModelForCausalLM')
    def test_run_evaluation_success(self, mock_ort_class, mock_model_class, mock_tokenizer_class):
        """Test complete evaluation pipeline success."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock()
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock models
        mock_original_model = Mock()
        mock_optimized_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_original_model
        mock_ort_class.from_pretrained.return_value = mock_optimized_model
        
        # Mock predictions
        mock_logits = Mock()
        mock_logits.item.return_value = 0
        mock_outputs = Mock()
        mock_outputs.logits = mock_logits
        
        mock_original_model.return_value = mock_outputs
        mock_optimized_model.return_value = mock_outputs
        
        # Test evaluation
        results = self.evaluator.run_evaluation()
        
        # Assertions
        assert "original_accuracy" in results
        assert "optimized_accuracy" in results
        assert "status" not in results  # No error status
    
    @patch('vyuha.evaluation.accuracy_evaluator.AutoTokenizer')
    def test_run_evaluation_failure(self, mock_tokenizer_class):
        """Test evaluation pipeline failure."""
        # Mock failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
        
        # Test
        results = self.evaluator.run_evaluation()
        
        # Assertions
        assert "error" in results
        assert "status" in results
        assert results["status"] == "failed"
        assert "Network error" in results["error"]
    
    def test_predict_category_failure(self):
        """Test prediction failure handling."""
        # Test with no models loaded
        prediction = self.evaluator.predict_category("test text", None, is_onnx=False)
        assert prediction == self.evaluator.categories[0]  # Default fallback
    
    def test_predict_category_invalid_model(self):
        """Test prediction with invalid model."""
        # Mock model that raises exception
        mock_model = Mock()
        mock_model.side_effect = Exception("Model error")
        
        # Test
        prediction = self.evaluator.predict_category("test text", mock_model, is_onnx=False)
        assert prediction == self.evaluator.categories[0]  # Default fallback
