"""
Unit tests for numlib namespace functionality (autograd and cuda)

Tests cover:
- autograd namespace (on/off control, context manager, function calls)
- cuda namespace (availability, device management, memory info)
"""

import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon import numlib as nm


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, test_name, test_func):
        """Run a single test"""
        try:
            test_func()
            self.passed += 1
            print(f"✓ {test_name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((test_name, str(e)))
            print(f"✗ {test_name}: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((test_name, str(e)))
            print(f"✗ {test_name}: {type(e).__name__}: {e}")

    def report(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print(f"Tests run: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        if self.errors:
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print("=" * 60)
        return self.failed == 0


def test_namespace_interaction():
    """Test interactions between namespaces"""
    runner = TestRunner()

    print("\n=== Testing Namespace Interactions ===\n")

    def test_independence():
        nm.autograd.enable()
        if nm.cuda.is_available():
            nm.cuda.enable()
            assert nm.autograd.is_enabled() is True
            assert nm.cuda.is_enabled() is True

            nm.autograd.disable()
            assert nm.autograd.is_enabled() is False
            assert nm.cuda.is_enabled() is True

            nm.cuda.disable()
            assert nm.autograd.is_enabled() is False
            assert nm.cuda.is_enabled() is False

            nm.autograd.enable()  # Reset
        else:
            # Just test autograd
            nm.autograd.disable()
            assert nm.autograd.is_enabled() is False
            nm.autograd.enable()
            assert nm.autograd.is_enabled() is True

    runner.run_test("autograd and cuda are independent", test_independence)

    if nm.cuda.is_available():

        def test_nested_both():
            nm.autograd.enable()
            nm.cuda.disable()

            with nm.autograd.off:
                assert nm.autograd.is_enabled() is False
                with nm.cuda.gpu:
                    assert nm.autograd.is_enabled() is False
                    assert nm.cuda.is_enabled() is True
                assert nm.autograd.is_enabled() is False
                assert nm.cuda.is_enabled() is False

            assert nm.autograd.is_enabled() is True
            assert nm.cuda.is_enabled() is False

        runner.run_test("nested contexts with both namespaces", test_nested_both)

    return runner.report()
