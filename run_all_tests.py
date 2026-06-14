#!/usr/bin/env python3
"""
Run all tests for the wine quality prediction project
"""

import sys
import os

# Add project root and tests directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tests'))


def run_test(module_name, test_func_name):
    """Helper to run a single test"""
    try:
        # Try importing from tests.module first
        try:
            module = __import__(f'tests.{module_name}', fromlist=[test_func_name])
        except ImportError:
            # Fall back to direct import
            module = __import__(module_name)
        
        test_func = getattr(module, test_func_name)
        test_func()
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("RUNNING ALL TESTS FOR WINE QUALITY PREDICTION")
    print("=" * 60)
    
    # List of tests (module name, function name)
    tests = [
        ("test_data_loader", "test_load_data"),
        ("test_preprocess", "test_create_target"),
        ("test_model", "test_model_training"),
        ("test_evaluate", "test_evaluate"),
    ]
    
    results = []
    
    for test_module, test_func in tests:
        print(f"\n📋 Running {test_module}.{test_func}...")
        passed, error = run_test(test_module, test_func)
        
        if passed:
            print(f"   ✅ PASSED")
            results.append(True)
        else:
            print(f"   ❌ FAILED: {error}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total = len(results)
    passed = sum(results)
    
    for i, (test_module, _) in enumerate(tests):
        status = "✅" if results[i] else "❌"
        print(f"{status} {test_module}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())