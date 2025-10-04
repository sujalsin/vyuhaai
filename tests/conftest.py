"""
Pytest configuration and fixtures for Vyuha AI tests.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.return_value = {
        "input_ids": Mock(),
        "attention_mask": Mock()
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock PyTorch model for testing."""
    model = Mock()
    
    # Mock parameters
    param1 = Mock()
    param1.numel.return_value = 1000
    param1.element_size.return_value = 4
    
    param2 = Mock()
    param2.numel.return_value = 2000
    param2.element_size.return_value = 4
    
    model.parameters.return_value = [param1, param2]
    model.buffers.return_value = []
    
    # Mock forward pass
    mock_outputs = Mock()
    mock_logits = Mock()
    mock_logits.item.return_value = 0
    mock_outputs.logits = mock_logits
    model.return_value = mock_outputs
    
    return model


@pytest.fixture
def mock_onnx_model():
    """Mock ONNX model for testing."""
    model = Mock()
    
    # Mock forward pass
    mock_outputs = Mock()
    mock_logits = Mock()
    mock_logits.item.return_value = 0
    mock_outputs.logits = mock_logits
    model.return_value = mock_outputs
    
    return model


@pytest.fixture
def sample_test_data():
    """Sample test data for evaluation."""
    return [
        {
            "text": "I need help with my account login",
            "label": "technical_issue",
            "label_id": 0
        },
        {
            "text": "Can you explain the billing charges?",
            "label": "billing_inquiry",
            "label_id": 1
        },
        {
            "text": "The application is not working properly",
            "label": "technical_issue",
            "label_id": 0
        },
        {
            "text": "How do I update my profile information?",
            "label": "account_help",
            "label_id": 3
        },
        {
            "text": "I want to request a new feature",
            "label": "feature_request",
            "label_id": 2
        }
    ]


@pytest.fixture
def mock_optimization_results():
    """Mock optimization results."""
    return {
        "status": "success",
        "model_name": "microsoft/DialoGPT-small",
        "output_dir": "/tmp/test_output",
        "onnx_path": "/tmp/test_output/model.onnx",
        "quantized_model_path": "/tmp/test_output/quantized_model"
    }


@pytest.fixture
def mock_performance_results():
    """Mock performance benchmark results."""
    return {
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


@pytest.fixture
def mock_accuracy_results():
    """Mock accuracy evaluation results."""
    return {
        "original_accuracy": 0.95,
        "optimized_accuracy": 0.94,
        "accuracy_drop": 0.01,
        "accuracy_retention": 98.9,
        "num_samples": 100,
        "original_predictions": ["technical_issue"] * 50 + ["billing_inquiry"] * 50,
        "optimized_predictions": ["technical_issue"] * 49 + ["billing_inquiry"] * 51,
        "true_labels": ["technical_issue"] * 50 + ["billing_inquiry"] * 50
    }


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ["slow", "integration"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
