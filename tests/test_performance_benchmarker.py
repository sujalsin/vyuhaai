"""
Tests for the performance benchmarker module.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from vyuha.evaluation.performance_benchmarker import PerformanceBenchmarker


class TestPerformanceBenchmarker:
    """Test cases for PerformanceBenchmarker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_name = "microsoft/DialoGPT-small"
        self.num_runs = 10
        self.benchmarker = PerformanceBenchmarker(self.model_name, self.num_runs)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test benchmarker initialization."""
        assert self.benchmarker.model_name == self.model_name
        assert self.benchmarker.num_runs == self.num_runs
        assert self.benchmarker.tokenizer is None
        assert self.benchmarker.original_model is None
        assert self.benchmarker.optimized_model is None
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    @patch('vyuha.evaluation.performance_benchmarker.AutoModelForCausalLM')
    @patch('vyuha.evaluation.performance_benchmarker.ORTModelForCausalLM')
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
        self.benchmarker.load_models()
        
        # Assertions
        assert self.benchmarker.tokenizer == mock_tokenizer
        assert self.benchmarker.original_model == mock_original_model
        assert self.benchmarker.optimized_model == mock_optimized_model
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    def test_load_models_failure(self, mock_tokenizer_class):
        """Test model loading failure."""
        # Mock failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
        
        # Test
        with pytest.raises(Exception, match="Network error"):
            self.benchmarker.load_models()
    
    def test_measure_model_size_file(self):
        """Test model size measurement for a single file."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test_model.bin"
        test_file.write_bytes(b"test data" * 1000)  # 8KB
        
        # Test
        results = self.benchmarker.measure_model_size(str(test_file))
        
        # Assertions
        assert results["size_bytes"] == 8000
        assert results["size_mb"] == 8000 / (1024 * 1024)
        assert results["num_files"] == 1
    
    def test_measure_model_size_directory(self):
        """Test model size measurement for a directory."""
        # Create test directory with files
        test_dir = Path(self.temp_dir) / "test_model"
        test_dir.mkdir()
        
        # Create multiple files
        (test_dir / "model.bin").write_bytes(b"data1" * 1000)
        (test_dir / "config.json").write_bytes(b"data2" * 500)
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "tokenizer.json").write_bytes(b"data3" * 200)
        
        # Test
        results = self.benchmarker.measure_model_size(str(test_dir))
        
        # Assertions
        expected_size = (5000 + 2500 + 1000)  # Sum of all file sizes
        assert results["size_bytes"] == expected_size
        assert results["size_mb"] == expected_size / (1024 * 1024)
        assert results["num_files"] == 3
    
    def test_measure_model_size_nonexistent(self):
        """Test model size measurement for nonexistent path."""
        # Test
        results = self.benchmarker.measure_model_size("/nonexistent/path")
        
        # Assertions
        assert results["size_bytes"] == 0
        assert results["size_mb"] == 0.0
        assert results["num_files"] == 0
        assert "error" in results
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    @patch('vyuha.evaluation.performance_benchmarker.AutoModelForCausalLM')
    def test_measure_inference_speed_pytorch(self, mock_model_class, mock_tokenizer_class):
        """Test inference speed measurement for PyTorch model."""
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
        mock_outputs = Mock()
        mock_model = Mock()
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Load models
        self.benchmarker.load_models()
        
        # Test
        results = self.benchmarker.measure_inference_speed(self.benchmarker.original_model, is_onnx=False)
        
        # Assertions
        assert "mean_inference_time_ms" in results
        assert "std_inference_time_ms" in results
        assert "min_inference_time_ms" in results
        assert "max_inference_time_ms" in results
        assert "p95_inference_time_ms" in results
        assert "num_runs" in results
        assert "throughput_per_second" in results
        assert results["num_runs"] == self.num_runs
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    @patch('vyuha.evaluation.performance_benchmarker.ORTModelForCausalLM')
    def test_measure_inference_speed_onnx(self, mock_ort_class, mock_tokenizer_class):
        """Test inference speed measurement for ONNX model."""
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
        mock_outputs = Mock()
        mock_ort_model = Mock()
        mock_ort_model.return_value = mock_outputs
        mock_ort_class.from_pretrained.return_value = mock_ort_model
        
        # Load models
        self.benchmarker.load_models()
        
        # Test
        results = self.benchmarker.measure_inference_speed(self.benchmarker.optimized_model, is_onnx=True)
        
        # Assertions
        assert "mean_inference_time_ms" in results
        assert "std_inference_time_ms" in results
        assert "min_inference_time_ms" in results
        assert "max_inference_time_ms" in results
        assert "p95_inference_time_ms" in results
        assert "num_runs" in results
        assert "throughput_per_second" in results
        assert results["num_runs"] == self.num_runs
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    @patch('vyuha.evaluation.performance_benchmarker.AutoModelForCausalLM')
    def test_measure_inference_speed_failure(self, mock_model_class, mock_tokenizer_class):
        """Test inference speed measurement failure."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock()
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model that raises exception
        mock_model = Mock()
        mock_model.side_effect = Exception("Model error")
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Load models
        self.benchmarker.load_models()
        
        # Test
        results = self.benchmarker.measure_inference_speed(self.benchmarker.original_model, is_onnx=False)
        
        # Assertions
        assert "error" in results
        assert results["mean_inference_time_ms"] == 0.0
        assert results["num_runs"] == 0
    
    @patch('vyuha.evaluation.performance_benchmarker.psutil.Process')
    def test_measure_memory_usage_pytorch(self, mock_process_class):
        """Test memory usage measurement for PyTorch model."""
        # Mock process
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process_class.return_value = mock_process
        
        # Mock model
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_param.element_size.return_value = 4
        
        mock_buffer = Mock()
        mock_buffer.numel.return_value = 500
        mock_buffer.element_size.return_value = 4
        
        mock_model = Mock()
        mock_model.parameters.return_value = [mock_param]
        mock_model.buffers.return_value = [mock_buffer]
        
        # Test
        results = self.benchmarker.measure_memory_usage(mock_model, is_onnx=False)
        
        # Assertions
        assert "baseline_memory_mb" in results
        assert "model_memory_mb" in results
        assert "total_memory_mb" in results
        assert results["baseline_memory_mb"] == 100.0
        assert results["model_memory_mb"] > 0
    
    @patch('vyuha.evaluation.performance_benchmarker.psutil.Process')
    def test_measure_memory_usage_onnx(self, mock_process_class):
        """Test memory usage measurement for ONNX model."""
        # Mock process
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 50 * 1024 * 1024  # 50MB
        mock_process_class.return_value = mock_process
        
        # Mock ONNX model
        mock_model = Mock()
        
        # Test
        results = self.benchmarker.measure_memory_usage(mock_model, is_onnx=True)
        
        # Assertions
        assert "baseline_memory_mb" in results
        assert "model_memory_mb" in results
        assert "total_memory_mb" in results
        assert results["baseline_memory_mb"] == 50.0
    
    @patch('vyuha.evaluation.performance_benchmarker.psutil.Process')
    def test_measure_memory_usage_failure(self, mock_process_class):
        """Test memory usage measurement failure."""
        # Mock process failure
        mock_process_class.side_effect = Exception("Process error")
        
        # Mock model
        mock_model = Mock()
        
        # Test
        results = self.benchmarker.measure_memory_usage(mock_model, is_onnx=False)
        
        # Assertions
        assert "error" in results
        assert results["baseline_memory_mb"] == 0.0
        assert results["model_memory_mb"] == 0.0
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    @patch('vyuha.evaluation.performance_benchmarker.AutoModelForCausalLM')
    def test_benchmark_original_model_success(self, mock_model_class, mock_tokenizer_class):
        """Test original model benchmarking success."""
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
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_param.element_size.return_value = 4
        
        mock_buffer = Mock()
        mock_buffer.numel.return_value = 500
        mock_buffer.element_size.return_value = 4
        
        mock_model = Mock()
        mock_model.parameters.return_value = [mock_param]
        mock_model.buffers.return_value = [mock_buffer]
        mock_model.return_value = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test
        results = self.benchmarker.benchmark_original_model()
        
        # Assertions
        assert results["model_type"] == "original"
        assert results["status"] == "success"
        assert "size" in results
        assert "speed" in results
        assert "memory" in results
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    def test_benchmark_original_model_failure(self, mock_tokenizer_class):
        """Test original model benchmarking failure."""
        # Mock failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
        
        # Test
        results = self.benchmarker.benchmark_original_model()
        
        # Assertions
        assert results["model_type"] == "original"
        assert results["status"] == "failed"
        assert "error" in results
        assert "Network error" in results["error"]
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    @patch('vyuha.evaluation.performance_benchmarker.ORTModelForCausalLM')
    def test_benchmark_optimized_model_success(self, mock_ort_class, mock_tokenizer_class):
        """Test optimized model benchmarking success."""
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
        mock_ort_model = Mock()
        mock_ort_model.return_value = Mock()
        mock_ort_class.from_pretrained.return_value = mock_ort_model
        
        # Test
        results = self.benchmarker.benchmark_optimized_model(self.temp_dir)
        
        # Assertions
        assert results["model_type"] == "optimized"
        assert results["status"] == "success"
        assert "size" in results
        assert "speed" in results
        assert "memory" in results
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    def test_benchmark_optimized_model_failure(self, mock_tokenizer_class):
        """Test optimized model benchmarking failure."""
        # Mock failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
        
        # Test
        results = self.benchmarker.benchmark_optimized_model(self.temp_dir)
        
        # Assertions
        assert results["model_type"] == "optimized"
        assert results["status"] == "failed"
        assert "error" in results
        assert "Network error" in results["error"]
    
    def test_calculate_improvements_success(self):
        """Test improvement calculations."""
        # Mock results
        original_results = {
            "status": "success",
            "size": {"size_mb": 100.0},
            "speed": {"mean_inference_time_ms": 50.0},
            "memory": {"model_memory_mb": 80.0}
        }
        
        optimized_results = {
            "status": "success",
            "size": {"size_mb": 25.0},
            "speed": {"mean_inference_time_ms": 20.0},
            "memory": {"model_memory_mb": 20.0}
        }
        
        # Test
        improvements = self.benchmarker._calculate_improvements(original_results, optimized_results)
        
        # Assertions
        assert "size_reduction_percent" in improvements
        assert "size_ratio" in improvements
        assert "speed_improvement_percent" in improvements
        assert "speed_ratio" in improvements
        assert "memory_reduction_percent" in improvements
        assert "memory_ratio" in improvements
        
        # Check specific values
        assert improvements["size_ratio"] == 4.0  # 100/25
        assert improvements["speed_ratio"] == 2.5  # 50/20
        assert improvements["memory_ratio"] == 4.0  # 80/20
    
    def test_calculate_improvements_failure(self):
        """Test improvement calculations with failed results."""
        # Mock failed results
        original_results = {"status": "failed"}
        optimized_results = {"status": "failed"}
        
        # Test
        improvements = self.benchmarker._calculate_improvements(original_results, optimized_results)
        
        # Assertions
        assert improvements == {}
    
    def test_calculate_improvements_zero_division(self):
        """Test improvement calculations with zero values."""
        # Mock results with zero values
        original_results = {
            "status": "success",
            "size": {"size_mb": 0.0},
            "speed": {"mean_inference_time_ms": 0.0},
            "memory": {"model_memory_mb": 0.0}
        }
        
        optimized_results = {
            "status": "success",
            "size": {"size_mb": 10.0},
            "speed": {"mean_inference_time_ms": 10.0},
            "memory": {"model_memory_mb": 10.0}
        }
        
        # Test
        improvements = self.benchmarker._calculate_improvements(original_results, optimized_results)
        
        # Assertions
        assert "size_ratio" in improvements
        assert "speed_ratio" in improvements
        assert "memory_ratio" in improvements
        assert improvements["size_ratio"] == float('inf')
        assert improvements["speed_ratio"] == 0.0
        assert improvements["memory_ratio"] == float('inf')
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    @patch('vyuha.evaluation.performance_benchmarker.AutoModelForCausalLM')
    @patch('vyuha.evaluation.performance_benchmarker.ORTModelForCausalLM')
    def test_run_complete_benchmark_success(self, mock_ort_class, mock_model_class, mock_tokenizer_class):
        """Test complete benchmark pipeline success."""
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
        
        # Mock model parameters for size calculation
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_param.element_size.return_value = 4
        
        mock_original_model.parameters.return_value = [mock_param]
        mock_original_model.buffers.return_value = []
        mock_original_model.return_value = Mock()
        
        mock_optimized_model.return_value = Mock()
        
        # Test
        results = self.benchmarker.run_complete_benchmark(optimized_model_path=self.temp_dir)
        
        # Assertions
        assert results["status"] == "success"
        assert "original" in results
        assert "optimized" in results
        assert "improvements" in results
        assert results["original"]["model_type"] == "original"
        assert results["optimized"]["model_type"] == "optimized"
    
    @patch('vyuha.evaluation.performance_benchmarker.AutoTokenizer')
    def test_run_complete_benchmark_failure(self, mock_tokenizer_class):
        """Test complete benchmark pipeline failure."""
        # Mock failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
        
        # Test
        results = self.benchmarker.run_complete_benchmark()
        
        # Assertions
        assert results["status"] == "failed"
        assert "error" in results
        assert "Network error" in results["error"]
