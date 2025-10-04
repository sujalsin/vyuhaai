"""
Integration tests for the complete Vyuha AI optimization pipeline.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from vyuha.cli import app


class TestIntegration:
    """Integration tests for the complete optimization pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.integration
    @patch('vyuha.cli.ModelOptimizer')
    @patch('vyuha.cli.PerformanceBenchmarker')
    @patch('vyuha.cli.AccuracyEvaluator')
    def test_complete_optimization_pipeline(self, mock_evaluator_class, mock_benchmarker_class, mock_optimizer_class):
        """Test the complete optimization pipeline end-to-end."""
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            "status": "success",
            "model_name": "microsoft/DialoGPT-small",
            "output_dir": self.temp_dir,
            "onnx_path": f"{self.temp_dir}/model.onnx",
            "quantized_model_path": f"{self.temp_dir}/quantized_model"
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock benchmarker
        mock_benchmarker = Mock()
        mock_benchmarker.run_complete_benchmark.return_value = {
            "status": "success",
            "original": {
                "model_type": "original",
                "size": {"size_mb": 100.0, "size_bytes": 104857600, "num_files": 3},
                "speed": {
                    "mean_inference_time_ms": 50.0,
                    "std_inference_time_ms": 5.0,
                    "min_inference_time_ms": 40.0,
                    "max_inference_time_ms": 60.0,
                    "p95_inference_time_ms": 55.0,
                    "num_runs": 100,
                    "throughput_per_second": 20.0
                },
                "memory": {
                    "baseline_memory_mb": 200.0,
                    "model_memory_mb": 80.0,
                    "total_memory_mb": 280.0
                }
            },
            "optimized": {
                "model_type": "optimized",
                "size": {"size_mb": 25.0, "size_bytes": 26214400, "num_files": 2},
                "speed": {
                    "mean_inference_time_ms": 20.0,
                    "std_inference_time_ms": 2.0,
                    "min_inference_time_ms": 18.0,
                    "max_inference_time_ms": 22.0,
                    "p95_inference_time_ms": 21.0,
                    "num_runs": 100,
                    "throughput_per_second": 50.0
                },
                "memory": {
                    "baseline_memory_mb": 200.0,
                    "model_memory_mb": 20.0,
                    "total_memory_mb": 220.0
                }
            },
            "improvements": {
                "size_reduction_percent": 75.0,
                "size_ratio": 4.0,
                "speed_improvement_percent": 60.0,
                "speed_ratio": 2.5,
                "memory_reduction_percent": 75.0,
                "memory_ratio": 4.0
            }
        }
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.run_evaluation.return_value = {
            "original_accuracy": 0.95,
            "optimized_accuracy": 0.94,
            "accuracy_drop": 0.01,
            "accuracy_retention": 98.9,
            "num_samples": 100
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        # Test complete pipeline
        result = self.runner.invoke(app, [
            "optimize",
            "--model", "microsoft/DialoGPT-small",
            "--task", "support_classification",
            "--output", self.temp_dir,
            "--samples", "100",
            "--runs", "100"
        ])
        
        # Assertions
        assert result.exit_code == 0
        assert "Optimization completed successfully" in result.output
        assert "4.0x smaller" in result.output
        assert "2.5x faster" in result.output
        assert "98.9% retained" in result.output
        
        # Verify all components were called
        mock_optimizer_class.assert_called_once_with("microsoft/DialoGPT-small", self.temp_dir)
        mock_benchmarker_class.assert_called_once_with("microsoft/DialoGPT-small", 100)
        mock_evaluator_class.assert_called_once_with("microsoft/DialoGPT-small", 100)
        
        # Verify method calls
        mock_optimizer.optimize.assert_called_once()
        mock_benchmarker.run_complete_benchmark.assert_called_once()
        mock_evaluator.run_evaluation.assert_called_once()
    
    @pytest.mark.integration
    @patch('vyuha.cli.ModelOptimizer')
    def test_optimization_pipeline_failure(self, mock_optimizer_class):
        """Test optimization pipeline failure handling."""
        # Mock optimizer failure
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            "status": "failed",
            "error": "Model not found"
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Test
        result = self.runner.invoke(app, [
            "optimize",
            "--model", "nonexistent/model"
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Optimization failed" in result.output
        assert "Model not found" in result.output
    
    @pytest.mark.integration
    @patch('vyuha.cli.ModelOptimizer')
    @patch('vyuha.cli.PerformanceBenchmarker')
    def test_optimization_with_benchmark_failure(self, mock_benchmarker_class, mock_optimizer_class):
        """Test optimization with benchmark failure."""
        # Mock optimizer success
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            "status": "success",
            "output_dir": self.temp_dir,
            "onnx_path": f"{self.temp_dir}/model.onnx"
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock benchmarker failure
        mock_benchmarker = Mock()
        mock_benchmarker.run_complete_benchmark.return_value = {
            "status": "failed",
            "error": "Benchmark failed"
        }
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Test
        result = self.runner.invoke(app, [
            "optimize",
            "--model", "microsoft/DialoGPT-small"
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Performance benchmarking failed" in result.output
        assert "Benchmark failed" in result.output
    
    @pytest.mark.integration
    @patch('vyuha.cli.ModelOptimizer')
    @patch('vyuha.cli.PerformanceBenchmarker')
    @patch('vyuha.cli.AccuracyEvaluator')
    def test_optimization_with_accuracy_evaluation_failure(self, mock_evaluator_class, mock_benchmarker_class, mock_optimizer_class):
        """Test optimization with accuracy evaluation failure."""
        # Mock optimizer success
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            "status": "success",
            "output_dir": self.temp_dir,
            "onnx_path": f"{self.temp_dir}/model.onnx"
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock benchmarker success
        mock_benchmarker = Mock()
        mock_benchmarker.run_complete_benchmark.return_value = {
            "status": "success",
            "original": {
                "size": {"size_mb": 100.0},
                "speed": {"mean_inference_time_ms": 50.0},
                "memory": {"model_memory_mb": 80.0}
            },
            "optimized": {
                "size": {"size_mb": 25.0},
                "speed": {"mean_inference_time_ms": 20.0},
                "memory": {"model_memory_mb": 20.0}
            },
            "improvements": {
                "size_ratio": 4.0,
                "speed_ratio": 2.5,
                "memory_ratio": 4.0
            }
        }
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Mock evaluator failure
        mock_evaluator = Mock()
        mock_evaluator.run_evaluation.return_value = {
            "error": "Accuracy evaluation failed",
            "status": "failed"
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        # Test
        result = self.runner.invoke(app, [
            "optimize",
            "--model", "microsoft/DialoGPT-small"
        ])
        
        # Assertions
        assert result.exit_code == 0  # Should still succeed
        assert "Optimization completed successfully" in result.output
        assert "Accuracy evaluation failed" in result.output
        assert "N/A" in result.output  # Accuracy should show N/A
    
    @pytest.mark.integration
    @patch('vyuha.cli.PerformanceBenchmarker')
    def test_benchmark_integration(self, mock_benchmarker_class):
        """Test benchmark command integration."""
        # Mock benchmarker
        mock_benchmarker = Mock()
        mock_benchmarker.benchmark_original_model.return_value = {
            "status": "success",
            "size": {"size_mb": 100.0, "size_bytes": 104857600, "num_files": 3},
            "speed": {
                "mean_inference_time_ms": 50.0,
                "std_inference_time_ms": 5.0,
                "min_inference_time_ms": 40.0,
                "max_inference_time_ms": 60.0,
                "p95_inference_time_ms": 55.0,
                "num_runs": 100,
                "throughput_per_second": 20.0
            },
            "memory": {
                "baseline_memory_mb": 200.0,
                "model_memory_mb": 80.0,
                "total_memory_mb": 280.0
            }
        }
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Test
        result = self.runner.invoke(app, [
            "benchmark",
            "--model", "microsoft/DialoGPT-small",
            "--runs", "100"
        ])
        
        # Assertions
        assert result.exit_code == 0
        assert "Model Performance Benchmark" in result.output
        assert "100.0 MB" in result.output
        assert "50.0 ms" in result.output
        assert "20.0 inferences/sec" in result.output
        
        # Verify benchmarker was called with correct parameters
        mock_benchmarker_class.assert_called_once_with("microsoft/DialoGPT-small", 100)
        mock_benchmarker.benchmark_original_model.assert_called_once()
    
    @pytest.mark.integration
    def test_cli_help_commands(self):
        """Test CLI help commands."""
        # Test main help
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Vyuha AI - Enterprise AI Model Optimization Platform" in result.output
        
        # Test optimize help
        result = self.runner.invoke(app, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "Optimize a model and generate comprehensive performance report" in result.output
        
        # Test benchmark help
        result = self.runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Benchmark a model's performance without optimization" in result.output
    
    @pytest.mark.integration
    def test_cli_parameter_validation(self):
        """Test CLI parameter validation."""
        # Test with invalid model
        result = self.runner.invoke(app, [
            "optimize",
            "--model", "",
            "--output", self.temp_dir
        ])
        # Should still run but may fail during execution
        
        # Test with invalid output directory
        result = self.runner.invoke(app, [
            "optimize",
            "--model", "microsoft/DialoGPT-small",
            "--output", "/nonexistent/path"
        ])
        # Should still run but may fail during execution
        
        # Test with invalid parameters
        result = self.runner.invoke(app, [
            "optimize",
            "--model", "microsoft/DialoGPT-small",
            "--samples", "-1",
            "--runs", "0"
        ])
        # Should still run but may fail during execution
    
    @pytest.mark.integration
    @patch('vyuha.cli.ModelOptimizer')
    @patch('vyuha.cli.PerformanceBenchmarker')
    @patch('vyuha.cli.AccuracyEvaluator')
    def test_verbose_logging(self, mock_evaluator_class, mock_benchmarker_class, mock_optimizer_class):
        """Test verbose logging mode."""
        # Mock all components
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            "status": "success",
            "output_dir": self.temp_dir,
            "onnx_path": f"{self.temp_dir}/model.onnx"
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        mock_benchmarker = Mock()
        mock_benchmarker.run_complete_benchmark.return_value = {
            "status": "success",
            "original": {
                "size": {"size_mb": 100.0},
                "speed": {"mean_inference_time_ms": 50.0},
                "memory": {"model_memory_mb": 80.0}
            },
            "optimized": {
                "size": {"size_mb": 25.0},
                "speed": {"mean_inference_time_ms": 20.0},
                "memory": {"model_memory_mb": 20.0}
            },
            "improvements": {
                "size_ratio": 4.0,
                "speed_ratio": 2.5,
                "memory_ratio": 4.0
            }
        }
        mock_benchmarker_class.return_value = mock_benchmarker
        
        mock_evaluator = Mock()
        mock_evaluator.run_evaluation.return_value = {
            "original_accuracy": 0.95,
            "optimized_accuracy": 0.94,
            "accuracy_retention": 98.9
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        # Test with verbose flag
        result = self.runner.invoke(app, [
            "optimize",
            "--model", "microsoft/DialoGPT-small",
            "--verbose"
        ])
        
        # Assertions
        assert result.exit_code == 0
        assert "Optimization completed successfully" in result.output
