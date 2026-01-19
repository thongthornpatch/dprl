#!/usr/bin/env python3
"""
Test script to verify the basic setup is working correctly.

Run this on your laptop to verify everything is set up properly.

Usage:
    python scripts/test_setup.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_imports():
    """Test that basic imports work."""
    print("Testing imports...")

    try:
        import yaml
        print("  ✅ yaml imported")
    except ImportError:
        print("  ❌ yaml not found - run: pip install pyyaml")
        return False

    try:
        import numpy
        print("  ✅ numpy imported")
    except ImportError:
        print("  ⚠️  numpy not found (optional for now)")

    try:
        from utils.config_loader import load_config
        print("  ✅ config_loader imported")
    except ImportError as e:
        print(f"  ❌ config_loader import failed: {e}")
        return False

    return True


def test_config_loading():
    """Test that configs can be loaded."""
    print("\nTesting configuration loading...")

    from utils.config_loader import load_config

    # Test laptop config
    try:
        config = load_config('configs/config_laptop.yaml')
        print("  ✅ Laptop config loaded")

        # Verify key fields
        assert config['model']['name'] == 'gpt2'
        assert config['training']['algorithm'] == 'grpo'
        print("  ✅ Laptop config validated")

    except Exception as e:
        print(f"  ❌ Laptop config failed: {e}")
        return False

    # Test NSCC config
    try:
        config = load_config('configs/config_nscc.yaml')
        print("  ✅ NSCC config loaded")

        # Verify key fields
        assert config['model']['name'] == 'Salesforce/codegen-1B-mono'
        assert config['data']['num_problems'] == 199
        print("  ✅ NSCC config validated")

    except Exception as e:
        print(f"  ❌ NSCC config failed: {e}")
        return False

    return True


def test_logging():
    """Test logging utilities."""
    print("\nTesting logging utilities...")

    from utils.logging_utils import setup_logger, MetricsLogger

    try:
        # Test logger
        logger = setup_logger('test', log_file='logs/test_setup.log', verbose=False)
        logger.info("Test message")
        print("  ✅ Logger created")

        # Test metrics logger
        metrics = MetricsLogger()
        metrics.log('test_metric', 0.5, step=1)
        assert metrics.get_latest('test_metric') == 0.5
        print("  ✅ Metrics logger working")

    except Exception as e:
        print(f"  ❌ Logging failed: {e}")
        return False

    return True


def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")

    required_dirs = [
        'configs',
        'src/data',
        'src/models',
        'src/training',
        'src/rewards',
        'src/evaluation',
        'src/utils',
        'data/raw',
        'data/processed',
        'logs',
        'experiments',
        'scripts',
    ]

    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} not found")
            all_exist = False

    return all_exist


def test_sandbox_basic():
    """Test basic sandboxing concepts (without RestrictedPython yet)."""
    print("\nTesting basic code execution safety concepts...")

    try:
        # Test 1: Can we run safe code?
        safe_code = """
def solution(x):
    return x + 1
"""

        local_scope = {}
        exec(safe_code, {}, local_scope)
        result = local_scope['solution'](5)
        assert result == 6
        print("  ✅ Safe code execution works")

        # Test 2: Can we catch dangerous code?
        dangerous_code = """
import os
os.system('echo "This should be blocked"')
"""

        try:
            # We'll implement proper blocking later
            # For now, just show we can catch import attempts
            if 'import os' in dangerous_code:
                print("  ✅ Dangerous imports detected (basic check)")

        except Exception as e:
            print(f"  ⚠️  Sandbox not fully implemented yet: {e}")

        print("  ℹ️  Full sandboxing will be implemented in Phase 3")
        return True

    except Exception as e:
        print(f"  ❌ Code execution test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("SETUP VERIFICATION TEST")
    print("=" * 80)
    print()

    tests = [
        ("Imports", test_imports),
        ("Configuration Loading", test_config_loading),
        ("Logging Utilities", test_logging),
        ("Directory Structure", test_directory_structure),
        ("Sandbox Basics", test_sandbox_basic),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}  {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Run: python scripts/download_neocoder.py (coming soon)")
        print("  2. Implement reward function")
        print("  3. Implement GRPO training loop")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
