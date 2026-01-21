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


def test_cuda_namespace():
    """Test cuda namespace"""
    runner = TestRunner()

    print("\n=== Testing CUDA Namespace ===\n")

    # Test availability check
    def test_is_available():
        result = nm.cuda.is_available()
        assert isinstance(result, bool)

    runner.run_test("cuda.is_available()", test_is_available)

    # Test device count
    def test_device_count():
        count = nm.cuda.device_count()
        assert isinstance(count, int)
        assert count >= 0

    runner.run_test("cuda.device_count()", test_device_count)

    # Test current_device when disabled
    def test_current_device_disabled():
        if nm.cuda.is_available():
            nm.cuda.disable()
        device = nm.cuda.current_device()
        assert device == -1

    runner.run_test("cuda.current_device() when disabled", test_current_device_disabled)

    # Test memory_info when disabled
    def test_memory_info_disabled():
        nm.cuda.disable()
        info = nm.cuda.memory_info()
        assert isinstance(info, dict)
        assert "error" in info

    runner.run_test("cuda.memory_info() when disabled", test_memory_info_disabled)

    # CUDA-specific tests (only if available)
    if nm.cuda.is_available():
        print("\n--- CUDA Available: Running GPU tests ---\n")

        def test_enable():
            nm.cuda.disable()
            nm.cuda.enable()
            assert nm.cuda.is_enabled() is True
            nm.cuda.disable()  # Cleanup

        runner.run_test("cuda.enable()", test_enable)

        def test_disable():
            nm.cuda.enable()
            nm.cuda.disable()
            assert nm.cuda.is_enabled() is False

        runner.run_test("cuda.disable()", test_disable)

        def test_enable_if_available():
            nm.cuda.disable()
            nm.cuda.enable_if_available()
            assert nm.cuda.is_enabled() is True
            nm.cuda.disable()  # Cleanup

        runner.run_test("cuda.enable_if_available()", test_enable_if_available)

        def test_gpu_context():
            nm.cuda.disable()
            with nm.cuda.gpu:
                assert nm.cuda.is_enabled() is True
            assert nm.cuda.is_enabled() is False

        runner.run_test("cuda.gpu context manager", test_gpu_context)

        def test_cpu_context():
            nm.cuda.enable()
            with nm.cuda.cpu:
                assert nm.cuda.is_enabled() is False
            assert nm.cuda.is_enabled() is True
            nm.cuda.disable()  # Cleanup

        runner.run_test("cuda.cpu context manager", test_cpu_context)

        def test_nested():
            nm.cuda.disable()
            with nm.cuda.gpu:
                assert nm.cuda.is_enabled() is True
                with nm.cuda.cpu:
                    assert nm.cuda.is_enabled() is False
                assert nm.cuda.is_enabled() is True
            assert nm.cuda.is_enabled() is False

        runner.run_test("nested cuda contexts", test_nested)

        def test_exception():
            nm.cuda.disable()
            try:
                with nm.cuda.gpu:
                    assert nm.cuda.is_enabled() is True
                    raise ValueError("test")
            except ValueError:
                pass
            assert nm.cuda.is_enabled() is False

        runner.run_test("cuda context restores on exception", test_exception)

        def test_current_device():
            nm.cuda.enable()
            device = nm.cuda.current_device()
            assert isinstance(device, int)
            assert device >= 0
            nm.cuda.disable()  # Cleanup

        runner.run_test("cuda.current_device() when enabled", test_current_device)

        def test_memory_info():
            nm.cuda.enable()
            info = nm.cuda.memory_info()
            assert isinstance(info, dict)
            assert "used" in info
            assert "total" in info
            assert info["used"] >= 0
            assert info["total"] >= 0
            nm.cuda.disable()  # Cleanup

        runner.run_test("cuda.memory_info() when enabled", test_memory_info)

        if nm.cuda.device_count() > 0:

            def test_set_device():
                nm.cuda.enable()
                nm.cuda.set_device(0)
                assert nm.cuda.current_device() == 0
                nm.cuda.disable()  # Cleanup

            runner.run_test("cuda.set_device(0)", test_set_device)

    else:
        print("\n--- CUDA Not Available: Skipping GPU tests ---\n")

        def test_enable_raises():
            try:
                nm.cuda.enable()
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "CUDA is not available" in str(e)

        runner.run_test("cuda.enable() raises without GPU", test_enable_raises)

        def test_gpu_context_raises():
            try:
                with nm.cuda.gpu:
                    pass
                assert False, "Should have raised RuntimeError"
            except RuntimeError:
                pass

        runner.run_test("cuda.gpu context raises without GPU", test_gpu_context_raises)

        def test_cpu_context_works():
            with nm.cuda.cpu:
                assert nm.cuda.is_enabled() is False

        runner.run_test("cuda.cpu context works without GPU", test_cpu_context_works)

        def test_set_device_raises():
            try:
                nm.cuda.set_device(0)
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "CUDA is not available" in str(e)

        runner.run_test("cuda.set_device() raises without GPU", test_set_device_raises)

    return runner.report()
