#!/usr/bin/env python3
"""
Comprehensive test runner for Vyuha AI optimization platform.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False


def main():
    """Run comprehensive tests."""
    print("🚀 Vyuha AI - Comprehensive Test Suite")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Test results tracking
    test_results = []
    
    # 1. Install dependencies
    print("\n📦 Installing dependencies...")
    install_cmd = "pip install -r requirements.txt"
    test_results.append(("Dependencies", run_command(install_cmd, "Installing dependencies")))
    
    # 2. Install package in development mode
    print("\n🔧 Installing package in development mode...")
    install_dev_cmd = "pip install -e ."
    test_results.append(("Package Installation", run_command(install_dev_cmd, "Installing package")))
    
    # 3. Run linting
    print("\n🔍 Running linting...")
    lint_cmd = "python -m flake8 vyuha/ tests/ --max-line-length=100 --ignore=E203,W503"
    test_results.append(("Linting", run_command(lint_cmd, "Running flake8 linting")))
    
    # 4. Run unit tests
    print("\n🧪 Running unit tests...")
    unit_test_cmd = "python -m pytest tests/ -v -m unit --tb=short"
    test_results.append(("Unit Tests", run_command(unit_test_cmd, "Running unit tests")))
    
    # 5. Run integration tests
    print("\n🔗 Running integration tests...")
    integration_test_cmd = "python -m pytest tests/test_integration.py -v -m integration --tb=short"
    test_results.append(("Integration Tests", run_command(integration_test_cmd, "Running integration tests")))
    
    # 6. Run all tests with coverage
    print("\n📊 Running all tests with coverage...")
    coverage_cmd = "python -m pytest tests/ -v --cov=vyuha --cov-report=term-missing --cov-report=html:htmlcov --cov-fail-under=80"
    test_results.append(("Coverage Tests", run_command(coverage_cmd, "Running tests with coverage")))
    
    # 7. Test CLI help commands
    print("\n🖥️ Testing CLI help commands...")
    cli_help_cmd = "python -m vyuha.cli --help"
    test_results.append(("CLI Help", run_command(cli_help_cmd, "Testing CLI help")))
    
    # 8. Test CLI optimize help
    print("\n🔧 Testing optimize command help...")
    optimize_help_cmd = "python -m vyuha.cli optimize --help"
    test_results.append(("Optimize Help", run_command(optimize_help_cmd, "Testing optimize help")))
    
    # 9. Test CLI benchmark help
    print("\n📊 Testing benchmark command help...")
    benchmark_help_cmd = "python -m vyuha.cli benchmark --help"
    test_results.append(("Benchmark Help", run_command(benchmark_help_cmd, "Testing benchmark help")))
    
    # 10. Test package import
    print("\n📦 Testing package import...")
    import_test_cmd = "python -c 'import vyuha; print(f\"Vyuha AI version: {vyuha.__version__}\")'"
    test_results.append(("Package Import", run_command(import_test_cmd, "Testing package import")))
    
    # 11. Test module imports
    print("\n🔧 Testing module imports...")
    module_import_cmd = "python -c 'from vyuha.core.optimizer import ModelOptimizer; from vyuha.evaluation.accuracy_evaluator import AccuracyEvaluator; from vyuha.evaluation.performance_benchmarker import PerformanceBenchmarker; print(\"All modules imported successfully\")'"
    test_results.append(("Module Imports", run_command(module_import_cmd, "Testing module imports")))
    
    # Print summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("Your Vyuha AI optimization platform is ready for deployment!")
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
