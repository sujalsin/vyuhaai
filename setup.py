from setuptools import find_packages, setup

setup(
    name="vyuha-ai",
    version="0.1.0",
    description="AI Model Optimization Platform for Enterprise",
    author="Vyuha AI",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.12.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "vyuha=vyuha.cli:app",
        ],
    },
)
