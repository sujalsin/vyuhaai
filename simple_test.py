#!/usr/bin/env python3
"""
Simple test script for Vyuha AI optimization platform.
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
    """Run simple tests."""
    print("🚀 Vyuha AI - Simple Test Suite")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Test results tracking
    test_results = []
    
    # 1. Test package import
    print("\n📦 Testing package import...")
    import_test_cmd = "python -c 'import vyuha; print(f\"Vyuha AI version: {vyuha.__version__}\")'"
    test_results.append(("Package Import", run_command(import_test_cmd, "Testing package import")))
    
    # 2. Test CLI help (without running the full command)
    print("\n🖥️ Testing CLI help...")
    cli_help_cmd = "python -c 'from vyuha.cli import app; print(\"CLI module imported successfully\")'"
    test_results.append(("CLI Import", run_command(cli_help_cmd, "Testing CLI import")))
    
    # 3. Test core modules
    print("\n🔧 Testing core modules...")
    core_test_cmd = "python -c 'from vyuha.core.optimizer import ModelOptimizer; print(\"Core optimizer imported successfully\")'"
    test_results.append(("Core Optimizer", run_command(core_test_cmd, "Testing core optimizer")))
    
    # 4. Test evaluation modules
    print("\n📊 Testing evaluation modules...")
    eval_test_cmd = "python -c 'from vyuha.evaluation.accuracy_evaluator import AccuracyEvaluator; from vyuha.evaluation.performance_benchmarker import PerformanceBenchmarker; print(\"Evaluation modules imported successfully\")'"
    test_results.append(("Evaluation Modules", run_command(eval_test_cmd, "Testing evaluation modules")))
    
    # 5. Test basic functionality
    print("\n🧪 Testing basic functionality...")
    basic_test_cmd = "python -c 'from vyuha.core.optimizer import ModelOptimizer; optimizer = ModelOptimizer(\"test-model\", \"./test-output\"); print(\"Optimizer created successfully\")'"
    test_results.append(("Basic Functionality", run_command(basic_test_cmd, "Testing basic functionality")))
    
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
        print("Your Vyuha AI optimization platform is ready for development!")
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
