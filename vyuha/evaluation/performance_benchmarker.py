"""
Performance benchmarking for measuring model size and inference speed.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class PerformanceBenchmarker:
    """
    Benchmarks model performance including size and inference speed.
    """
    
    def __init__(self, model_name: str, num_runs: int = 100):
        """
        Initialize the performance benchmarker.
        
        Args:
            model_name: Hugging Face model identifier
            num_runs: Number of inference runs for speed testing
        """
        self.model_name = model_name
        self.num_runs = num_runs
        self.tokenizer = None
        self.original_model = None
        self.optimized_model = None
        
    def load_models(self, original_model_path: str = None, optimized_model_path: str = None) -> None:
        """
        Load both original and optimized models.
        
        Args:
            original_model_path: Path to original model (if None, loads from HF)
            optimized_model_path: Path to optimized ONNX model
        """
        logger.info("Loading models for performance benchmarking...")
        
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
    
    def measure_model_size(self, model_path: str) -> Dict[str, float]:
        """
        Measure model size in MB.
        
        Args:
            model_path: Path to model directory or file
            
        Returns:
            Dictionary with size metrics
        """
        logger.info(f"Measuring model size: {model_path}")
        
        try:
            path = Path(model_path)
            
            if path.is_file():
                # Single file
                size_bytes = path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                
                return {
                    "size_bytes": size_bytes,
                    "size_mb": size_mb,
                    "num_files": 1
                }
            else:
                # Directory - sum all files
                total_size = 0
                num_files = 0
                
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        num_files += 1
                
                size_mb = total_size / (1024 * 1024)
                
                return {
                    "size_bytes": total_size,
                    "size_mb": size_mb,
                    "num_files": num_files
                }
                
        except Exception as e:
            logger.error(f"Failed to measure model size: {e}")
            return {
                "size_bytes": 0,
                "size_mb": 0.0,
                "num_files": 0,
                "error": str(e)
            }
    
    def measure_inference_speed(self, model, is_onnx: bool = False) -> Dict[str, float]:
        """
        Measure inference speed for a model.
        
        Args:
            model: Model to benchmark
            is_onnx: Whether the model is ONNX format
            
        Returns:
            Dictionary with speed metrics
        """
        logger.info(f"Measuring inference speed (ONNX: {is_onnx})...")
        
        try:
            # Prepare test inputs
            test_texts = [
                "I need help with my account login",
                "Can you explain the billing charges?",
                "The application is not working properly",
                "How do I update my profile information?",
                "I want to request a new feature"
            ]
            
            # Warm up
            for text in test_texts[:2]:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                if is_onnx:
                    _ = model(**inputs)
                else:
                    with torch.no_grad():
                        _ = model(**inputs)
            
            # Benchmark
            times = []
            
            for _ in range(self.num_runs):
                text = np.random.choice(test_texts)
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                start_time = time.time()
                
                if is_onnx:
                    outputs = model(**inputs)
                else:
                    with torch.no_grad():
                        outputs = model(**inputs)
                
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # Convert to ms
                times.append(inference_time)
            
            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            p95_time = np.percentile(times, 95)
            
            results = {
                "mean_inference_time_ms": mean_time,
                "std_inference_time_ms": std_time,
                "min_inference_time_ms": min_time,
                "max_inference_time_ms": max_time,
                "p95_inference_time_ms": p95_time,
                "num_runs": self.num_runs,
                "throughput_per_second": 1000 / mean_time if mean_time > 0 else 0
            }
            
            logger.info(f"Inference speed measurement completed:")
            logger.info(f"  Mean time: {mean_time:.2f}ms")
            logger.info(f"  Throughput: {results['throughput_per_second']:.2f} inferences/sec")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to measure inference speed: {e}")
            return {
                "mean_inference_time_ms": 0.0,
                "std_inference_time_ms": 0.0,
                "min_inference_time_ms": 0.0,
                "max_inference_time_ms": 0.0,
                "p95_inference_time_ms": 0.0,
                "num_runs": 0,
                "throughput_per_second": 0.0,
                "error": str(e)
            }
    
    def measure_memory_usage(self, model, is_onnx: bool = False) -> Dict[str, float]:
        """
        Measure memory usage of a model.
        
        Args:
            model: Model to benchmark
            is_onnx: Whether the model is ONNX format
            
        Returns:
            Dictionary with memory metrics
        """
        logger.info(f"Measuring memory usage (ONNX: {is_onnx})...")
        
        try:
            # Get baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Load model and measure memory
            if is_onnx:
                # ONNX models are already loaded
                model_memory = baseline_memory
            else:
                # PyTorch model memory
                model_memory = baseline_memory
                
                # Calculate model parameter memory
                param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
                
                model_memory = param_memory + buffer_memory
            
            results = {
                "baseline_memory_mb": baseline_memory,
                "model_memory_mb": model_memory,
                "total_memory_mb": baseline_memory + model_memory
            }
            
            logger.info(f"Memory usage measurement completed:")
            logger.info(f"  Model memory: {model_memory:.2f}MB")
            logger.info(f"  Total memory: {results['total_memory_mb']:.2f}MB")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to measure memory usage: {e}")
            return {
                "baseline_memory_mb": 0.0,
                "model_memory_mb": 0.0,
                "total_memory_mb": 0.0,
                "error": str(e)
            }
    
    def benchmark_original_model(self, model_path: str = None) -> Dict[str, Any]:
        """
        Benchmark the original model.
        
        Args:
            model_path: Path to original model (if None, loads from HF)
            
        Returns:
            Dictionary with original model benchmarks
        """
        logger.info("Benchmarking original model...")
        
        try:
            # Load model if not already loaded
            if self.original_model is None:
                if model_path:
                    self.original_model = AutoModelForCausalLM.from_pretrained(model_path)
                else:
                    self.original_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Measure size
            if model_path:
                size_metrics = self.measure_model_size(model_path)
            else:
                # Estimate size from model parameters
                param_size = sum(p.numel() * p.element_size() for p in self.original_model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in self.original_model.buffers())
                total_size = param_size + buffer_size
                size_metrics = {
                    "size_bytes": total_size,
                    "size_mb": total_size / (1024 * 1024),
                    "num_files": 1
                }
            
            # Measure speed
            speed_metrics = self.measure_inference_speed(self.original_model, is_onnx=False)
            
            # Measure memory
            memory_metrics = self.measure_memory_usage(self.original_model, is_onnx=False)
            
            results = {
                "model_type": "original",
                "size": size_metrics,
                "speed": speed_metrics,
                "memory": memory_metrics,
                "status": "success"
            }
            
            logger.info("Original model benchmarking completed")
            return results
            
        except Exception as e:
            logger.error(f"Original model benchmarking failed: {e}")
            return {
                "model_type": "original",
                "status": "failed",
                "error": str(e)
            }
    
    def benchmark_optimized_model(self, model_path: str) -> Dict[str, Any]:
        """
        Benchmark the optimized model.
        
        Args:
            model_path: Path to optimized ONNX model
            
        Returns:
            Dictionary with optimized model benchmarks
        """
        logger.info("Benchmarking optimized model...")
        
        try:
            # Load model if not already loaded
            if self.optimized_model is None:
                self.optimized_model = ORTModelForCausalLM.from_pretrained(model_path)
            
            # Measure size
            size_metrics = self.measure_model_size(model_path)
            
            # Measure speed
            speed_metrics = self.measure_inference_speed(self.optimized_model, is_onnx=True)
            
            # Measure memory
            memory_metrics = self.measure_memory_usage(self.optimized_model, is_onnx=True)
            
            results = {
                "model_type": "optimized",
                "size": size_metrics,
                "speed": speed_metrics,
                "memory": memory_metrics,
                "status": "success"
            }
            
            logger.info("Optimized model benchmarking completed")
            return results
            
        except Exception as e:
            logger.error(f"Optimized model benchmarking failed: {e}")
            return {
                "model_type": "optimized",
                "status": "failed",
                "error": str(e)
            }
    
    def run_complete_benchmark(self, original_model_path: str = None, optimized_model_path: str = None) -> Dict[str, Any]:
        """
        Run complete performance benchmark for both models.
        
        Args:
            original_model_path: Path to original model
            optimized_model_path: Path to optimized model
            
        Returns:
            Complete benchmark results
        """
        logger.info("Starting complete performance benchmark...")
        
        try:
            # Load models
            self.load_models(original_model_path, optimized_model_path)
            
            # Benchmark original model
            original_results = self.benchmark_original_model(original_model_path)
            
            # Benchmark optimized model
            optimized_results = self.benchmark_optimized_model(optimized_model_path)
            
            # Calculate improvements
            improvements = self._calculate_improvements(original_results, optimized_results)
            
            results = {
                "original": original_results,
                "optimized": optimized_results,
                "improvements": improvements,
                "status": "success"
            }
            
            logger.info("Complete performance benchmark completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Complete performance benchmark failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _calculate_improvements(self, original_results: Dict[str, Any], optimized_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate improvement metrics between original and optimized models.
        
        Args:
            original_results: Results from original model
            optimized_results: Results from optimized model
            
        Returns:
            Dictionary with improvement metrics
        """
        try:
            improvements = {}
            
            # Size improvements
            if (original_results.get("status") == "success" and 
                optimized_results.get("status") == "success"):
                
                orig_size = original_results["size"]["size_mb"]
                opt_size = optimized_results["size"]["size_mb"]
                
                if orig_size > 0:
                    size_reduction = (orig_size - opt_size) / orig_size * 100
                    size_ratio = orig_size / opt_size if opt_size > 0 else float('inf')
                    improvements["size_reduction_percent"] = size_reduction
                    improvements["size_ratio"] = size_ratio
                
                # Speed improvements
                orig_speed = original_results["speed"]["mean_inference_time_ms"]
                opt_speed = optimized_results["speed"]["mean_inference_time_ms"]
                
                if orig_speed > 0 and opt_speed > 0:
                    speed_improvement = (orig_speed - opt_speed) / orig_speed * 100
                    speed_ratio = orig_speed / opt_speed
                    improvements["speed_improvement_percent"] = speed_improvement
                    improvements["speed_ratio"] = speed_ratio
                
                # Memory improvements
                orig_memory = original_results["memory"]["model_memory_mb"]
                opt_memory = optimized_results["memory"]["model_memory_mb"]
                
                if orig_memory > 0:
                    memory_reduction = (orig_memory - opt_memory) / orig_memory * 100
                    memory_ratio = orig_memory / opt_memory if opt_memory > 0 else float('inf')
                    improvements["memory_reduction_percent"] = memory_reduction
                    improvements["memory_ratio"] = memory_ratio
            
            return improvements
            
        except Exception as e:
            logger.error(f"Failed to calculate improvements: {e}")
            return {"error": str(e)}
