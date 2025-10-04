"""
Tests for the core optimizer module.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from vyuha.core.optimizer import ModelOptimizer


class TestModelOptimizer:
    """Test cases for ModelOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_name = "microsoft/DialoGPT-small"  # Smaller model for testing
        self.optimizer = ModelOptimizer(self.model_name, self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.model_name == self.model_name
        assert self.optimizer.output_dir == Path(self.temp_dir)
        assert self.optimizer.tokenizer is None
        assert self.optimizer.original_model is None
        assert self.optimizer.quantized_model is None
        assert self.optimizer.onnx_model is None
    
    @patch('vyuha.core.optimizer.AutoTokenizer')
    @patch('vyuha.core.optimizer.AutoModelForCausalLM')
    def test_load_model_success(self, mock_model_class, mock_tokenizer_class):
        """Test successful model loading."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test
        self.optimizer.load_model()
        
        # Assertions
        mock_tokenizer_class.from_pretrained.assert_called_once_with(self.model_name)
        mock_model_class.from_pretrained.assert_called_once()
        assert self.optimizer.tokenizer == mock_tokenizer
        assert self.optimizer.original_model == mock_model
    
    @patch('vyuha.core.optimizer.AutoTokenizer')
    @patch('vyuha.core.optimizer.AutoModelForCausalLM')
    def test_load_model_failure(self, mock_model_class, mock_tokenizer_class):
        """Test model loading failure."""
        # Mock failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")
        
        # Test
        with pytest.raises(Exception, match="Model not found"):
            self.optimizer.load_model()
    
    @patch('vyuha.core.optimizer.AutoTokenizer')
    @patch('vyuha.core.optimizer.AutoModelForCausalLM')
    def test_quantize_model(self, mock_model_class, mock_tokenizer_class):
        """Test model quantization."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Load model first
        self.optimizer.load_model()
        
        # Test quantization
        self.optimizer.quantize_model()
        
        # Assertions
        assert self.optimizer.quantized_model == mock_model
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
    
    @patch('vyuha.core.optimizer.AutoTokenizer')
    @patch('vyuha.core.optimizer.AutoModelForCausalLM')
    @patch('vyuha.core.optimizer.ORTModelForCausalLM')
    def test_export_to_onnx(self, mock_ort_class, mock_model_class, mock_tokenizer_class):
        """Test ONNX export."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_ort_model = Mock()
        mock_ort_class.from_pretrained.return_value = mock_ort_model
        
        # Load model first
        self.optimizer.load_model()
        
        # Test ONNX export
        onnx_path = self.optimizer.export_to_onnx()
        
        # Assertions
        assert self.optimizer.onnx_model == mock_ort_model
        mock_ort_model.save_pretrained.assert_called_once()
        assert onnx_path == str(Path(self.temp_dir) / "model.onnx")
    
    @patch('vyuha.core.optimizer.AutoTokenizer')
    @patch('vyuha.core.optimizer.AutoModelForCausalLM')
    @patch('vyuha.core.optimizer.ORTModelForCausalLM')
    def test_optimize_success(self, mock_ort_class, mock_model_class, mock_tokenizer_class):
        """Test complete optimization pipeline success."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_ort_model = Mock()
        mock_ort_class.from_pretrained.return_value = mock_ort_model
        
        # Test optimization
        results = self.optimizer.optimize()
        
        # Assertions
        assert results["status"] == "success"
        assert results["model_name"] == self.model_name
        assert results["output_dir"] == self.temp_dir
        assert "onnx_path" in results
        assert "quantized_model_path" in results
    
    @patch('vyuha.core.optimizer.AutoTokenizer')
    def test_optimize_failure(self, mock_tokenizer_class):
        """Test optimization pipeline failure."""
        # Mock failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
        
        # Test
        results = self.optimizer.optimize()
        
        # Assertions
        assert results["status"] == "failed"
        assert "error" in results
        assert "Network error" in results["error"]
    
    def test_get_model_info_no_model(self):
        """Test get_model_info when no model is loaded."""
        info = self.optimizer.get_model_info()
        assert "error" in info
        assert info["error"] == "Model not loaded"
    
    @patch('vyuha.core.optimizer.AutoTokenizer')
    @patch('vyuha.core.optimizer.AutoModelForCausalLM')
    def test_get_model_info_with_model(self, mock_model_class, mock_tokenizer_class):
        """Test get_model_info with loaded model."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model with parameters
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000
        mock_param1.element_size.return_value = 4
        
        mock_param2 = Mock()
        mock_param2.numel.return_value = 2000
        mock_param2.element_size.return_value = 4
        
        mock_model = Mock()
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        mock_model.buffers.return_value = []
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Load model
        self.optimizer.load_model()
        
        # Test get_model_info
        info = self.optimizer.get_model_info()
        
        # Assertions
        assert info["model_name"] == self.model_name
        assert info["model_type"] == "Mock"
        assert info["num_parameters"] == 3000  # 1000 + 2000
        assert info["model_size_mb"] > 0
    
    def test_calculate_model_size(self):
        """Test model size calculation."""
        # Test with no model
        size = self.optimizer._calculate_model_size()
        assert size == 0.0
        
        # Test with mock model
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_param.element_size.return_value = 4
        
        mock_buffer = Mock()
        mock_buffer.numel.return_value = 500
        mock_buffer.element_size.return_value = 4
        
        mock_model = Mock()
        mock_model.parameters.return_value = [mock_param]
        mock_model.buffers.return_value = [mock_buffer]
        
        self.optimizer.original_model = mock_model
        
        size = self.optimizer._calculate_model_size()
        expected_size = (1000 * 4 + 500 * 4) / (1024 * 1024)  # Convert to MB
        assert abs(size - expected_size) < 0.001
