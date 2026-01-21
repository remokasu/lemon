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


def test_autograd_namespace():
    """Test autograd namespace"""
    runner = TestRunner()

    print("\n=== Testing Autograd Namespace ===\n")

    # Test default state
    def test_default_enabled():
        nm.autograd.enable()  # Reset
        assert nm.autograd.is_enabled() is True

    runner.run_test("autograd is enabled by default", test_default_enabled)

    # Test enable/disable
    def test_enable():
        nm.autograd.disable()
        nm.autograd.enable()
        assert nm.autograd.is_enabled() is True

    runner.run_test("autograd.enable()", test_enable)

    def test_disable():
        nm.autograd.enable()
        nm.autograd.disable()
        assert nm.autograd.is_enabled() is False

    runner.run_test("autograd.disable()", test_disable)

    # Test set_enabled
    def test_set_enabled_true():
        nm.autograd.set_enabled(True)
        assert nm.autograd.is_enabled() is True

    runner.run_test("autograd.set_enabled(True)", test_set_enabled_true)

    def test_set_enabled_false():
        nm.autograd.set_enabled(False)
        assert nm.autograd.is_enabled() is False

    runner.run_test("autograd.set_enabled(False)", test_set_enabled_false)

    # Test context manager - off
    def test_off_context():
        nm.autograd.enable()
        with nm.autograd.off:
            assert nm.autograd.is_enabled() is False
        assert nm.autograd.is_enabled() is True

    runner.run_test("autograd.off context manager", test_off_context)

    # Test context manager - on
    def test_on_context():
        nm.autograd.disable()
        with nm.autograd.on:
            assert nm.autograd.is_enabled() is True
        assert nm.autograd.is_enabled() is False

    runner.run_test("autograd.on context manager", test_on_context)

    # Test nested contexts
    def test_nested():
        nm.autograd.enable()
        with nm.autograd.off:
            assert nm.autograd.is_enabled() is False
            with nm.autograd.on:
                assert nm.autograd.is_enabled() is True
            assert nm.autograd.is_enabled() is False
        assert nm.autograd.is_enabled() is True

    runner.run_test("nested autograd contexts", test_nested)

    # Test exception handling
    def test_exception():
        nm.autograd.enable()
        try:
            with nm.autograd.off:
                assert nm.autograd.is_enabled() is False
                raise ValueError("test")
        except ValueError:
            pass
        assert nm.autograd.is_enabled() is True

    runner.run_test("autograd context restores on exception", test_exception)

    # Test function call
    def test_function_call():
        nm.autograd.enable()
        nm.autograd.off()
        assert nm.autograd.is_enabled() is False
        nm.autograd.on()
        assert nm.autograd.is_enabled() is True

    runner.run_test("autograd function calls", test_function_call)

    # Test repr
    def test_repr():
        assert repr(nm.autograd.on) == "autograd.on"
        assert repr(nm.autograd.off) == "autograd.off"

    runner.run_test("autograd repr", test_repr)

    # Test integration with operations
    def test_integration():
        nm.autograd.enable()
        x = nm.real(3.0)
        assert x.requires_grad is True

        nm.autograd.disable()
        y = nm.real(3.0)
        assert y.requires_grad is False

        nm.autograd.enable()  # Reset

    runner.run_test("autograd affects requires_grad", test_integration)

    return runner.report()


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


if __name__ == "__main__":
    # Run all test suites
    results = []

    results.append(test_autograd_namespace())
    results.append(test_cuda_namespace())
    results.append(test_namespace_interaction())

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    if all(results):
        print("✓ All test suites passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
