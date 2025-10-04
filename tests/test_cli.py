"""
Tests for the CLI module.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from vyuha.cli import _display_benchmark_results, _generate_final_report, app


class TestCLI:
    """Test cases for CLI module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_app_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Vyuha AI - Enterprise AI Model Optimization Platform" in result.output
    
    def test_optimize_help(self):
        """Test optimize command help."""
        result = self.runner.invoke(app, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "Optimize a model and generate comprehensive performance report" in result.output
    
    def test_benchmark_help(self):
        """Test benchmark command help."""
        result = self.runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Benchmark a model's performance without optimization" in result.output
    
    @patch('vyuha.cli.ModelOptimizer')
    @patch('vyuha.cli.PerformanceBenchmarker')
    @patch('vyuha.cli.AccuracyEvaluator')
    def test_optimize_success(self, mock_evaluator_class, mock_benchmarker_class, mock_optimizer_class):
        """Test successful optimization command."""
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            "status": "success",
            "output_dir": self.temp_dir,
            "onnx_path": f"{self.temp_dir}/model.onnx"
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock benchmarker
        mock_benchmarker = Mock()
        mock_benchmarker.run_complete_benchmark.return_value = {
            "status": "success",
            "original": {
                "model_type": "original",
                "size": {"size_mb": 100.0},
                "speed": {"mean_inference_time_ms": 50.0},
                "memory": {"model_memory_mb": 80.0}
            },
            "optimized": {
                "model_type": "optimized",
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
        
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.run_evaluation.return_value = {
            "original_accuracy": 0.95,
            "optimized_accuracy": 0.94,
            "accuracy_retention": 98.9
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        # Test
        result = self.runner.invoke(app, [
            "optimize",
            "--model", "microsoft/DialoGPT-small",
            "--output", self.temp_dir
        ])
        
        # Assertions
        assert result.exit_code == 0
        assert "Optimization completed successfully" in result.output
        assert "4.0x smaller" in result.output
        assert "2.5x faster" in result.output
    
    @patch('vyuha.cli.ModelOptimizer')
    def test_optimize_failure(self, mock_optimizer_class):
        """Test optimization command failure."""
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
    
    @patch('vyuha.cli.PerformanceBenchmarker')
    def test_benchmark_success(self, mock_benchmarker_class):
        """Test successful benchmark command."""
        # Mock benchmarker
        mock_benchmarker = Mock()
        mock_benchmarker.benchmark_original_model.return_value = {
            "status": "success",
            "size": {"size_mb": 100.0},
            "speed": {
                "mean_inference_time_ms": 50.0,
                "throughput_per_second": 20.0
            },
            "memory": {
                "model_memory_mb": 80.0,
                "total_memory_mb": 100.0
            }
        }
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Test
        result = self.runner.invoke(app, [
            "benchmark",
            "--model", "microsoft/DialoGPT-small"
        ])
        
        # Assertions
        assert result.exit_code == 0
        assert "Model Performance Benchmark" in result.output
        assert "100.0 MB" in result.output
        assert "50.0 ms" in result.output
    
    @patch('vyuha.cli.PerformanceBenchmarker')
    def test_benchmark_failure(self, mock_benchmarker_class):
        """Test benchmark command failure."""
        # Mock benchmarker failure
        mock_benchmarker = Mock()
        mock_benchmarker.benchmark_original_model.return_value = {
            "status": "failed",
            "error": "Model not found"
        }
        mock_benchmarker_class.return_value = mock_benchmarker
        
        # Test
        result = self.runner.invoke(app, [
            "benchmark",
            "--model", "nonexistent/model"
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Benchmarking failed" in result.output
        assert "Model not found" in result.output
    
    def test_generate_final_report(self):
        """Test final report generation."""
        # Mock results
        optimization_results = {
            "status": "success",
            "output_dir": self.temp_dir,
            "onnx_path": f"{self.temp_dir}/model.onnx"
        }
        
        performance_results = {
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
        
        accuracy_results = {
            "original_accuracy": 0.95,
            "optimized_accuracy": 0.94,
            "accuracy_retention": 98.9
        }
        
        # Test (this will print to console, so we just check it doesn't raise)
        try:
            _generate_final_report(optimization_results, performance_results, accuracy_results)
            assert True  # If we get here, no exception was raised
        except Exception as e:
            pytest.fail(f"Report generation failed: {e}")
    
    def test_display_benchmark_results(self):
        """Test benchmark results display."""
        # Mock results
        results = {
            "status": "success",
            "size": {"size_mb": 100.0},
            "speed": {
                "mean_inference_time_ms": 50.0,
                "throughput_per_second": 20.0
            },
            "memory": {
                "model_memory_mb": 80.0,
                "total_memory_mb": 100.0
            }
        }
        
        # Test (this will print to console, so we just check it doesn't raise)
        try:
            _display_benchmark_results(results)
            assert True  # If we get here, no exception was raised
        except Exception as e:
            pytest.fail(f"Benchmark display failed: {e}")
    
    def test_optimize_with_custom_parameters(self):
        """Test optimize command with custom parameters."""
        # Mock all components
        with patch('vyuha.cli.ModelOptimizer') as mock_optimizer_class, \
             patch('vyuha.cli.PerformanceBenchmarker') as mock_benchmarker_class, \
             patch('vyuha.cli.AccuracyEvaluator') as mock_evaluator_class:
            
            # Mock optimizer
            mock_optimizer = Mock()
            mock_optimizer.optimize.return_value = {
                "status": "success",
                "output_dir": self.temp_dir,
                "onnx_path": f"{self.temp_dir}/model.onnx"
            }
            mock_optimizer_class.return_value = mock_optimizer
            
            # Mock benchmarker
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
            
            # Mock evaluator
            mock_evaluator = Mock()
            mock_evaluator.run_evaluation.return_value = {
                "original_accuracy": 0.95,
                "optimized_accuracy": 0.94,
                "accuracy_retention": 98.9
            }
            mock_evaluator_class.return_value = mock_evaluator
            
            # Test with custom parameters
            result = self.runner.invoke(app, [
                "optimize",
                "--model", "microsoft/DialoGPT-small",
                "--task", "support_classification",
                "--output", self.temp_dir,
                "--samples", "50",
                "--runs", "50",
                "--verbose"
            ])
            
            # Assertions
            assert result.exit_code == 0
            assert "Optimization completed successfully" in result.output
            
            # Check that components were called with correct parameters
            mock_optimizer_class.assert_called_once()
            mock_benchmarker_class.assert_called_once_with("microsoft/DialoGPT-small", 50)
            mock_evaluator_class.assert_called_once_with("microsoft/DialoGPT-small", 50)
    
    def test_benchmark_with_custom_parameters(self):
        """Test benchmark command with custom parameters."""
        # Mock benchmarker
        with patch('vyuha.cli.PerformanceBenchmarker') as mock_benchmarker_class:
            mock_benchmarker = Mock()
            mock_benchmarker.benchmark_original_model.return_value = {
                "status": "success",
                "size": {"size_mb": 100.0},
                "speed": {
                    "mean_inference_time_ms": 50.0,
                    "throughput_per_second": 20.0
                },
                "memory": {
                    "model_memory_mb": 80.0,
                    "total_memory_mb": 100.0
                }
            }
            mock_benchmarker_class.return_value = mock_benchmarker
            
            # Test with custom parameters
            result = self.runner.invoke(app, [
                "benchmark",
                "--model", "microsoft/DialoGPT-small",
                "--path", self.temp_dir,
                "--runs", "50"
            ])
            
            # Assertions
            assert result.exit_code == 0
            assert "Model Performance Benchmark" in result.output
            
            # Check that benchmarker was called with correct parameters
            mock_benchmarker_class.assert_called_once_with("microsoft/DialoGPT-small", 50)
