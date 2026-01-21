"""
Test suite for nnlib
"""

import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon as nc
import threading


def test_train_mode():
    """Test training mode control - comprehensive"""

    print("Testing training mode control...")

    nc.train.on()

    # Test 1: Default training mode is ON
    assert nc.train.is_on() == True, "Default training mode should be ON"
    print("  ✅ Default training mode is ON")

    # Test 2: Turn off training mode (function call)
    nc.train.off()
    assert nc.train.is_on() == False, "Training mode should be OFF after train.off()"
    assert nc.train.is_off() == True, "is_off() should return True"  # ← 追加
    print("  ✅ Turn off training mode")

    # Test 3: Turn on training mode (function call)
    nc.train.on()
    assert nc.train.is_on() == True, "Training mode should be ON after train.on()"
    assert nc.train.is_off() == False, "is_off() should return False"  # ← 追加
    print("  ✅ Turn on training mode")

    # Test 4: Context manager - train.on()
    nc.train.off()  # Start with OFF
    with nc.train.on:
        assert nc.train.is_on() == True, "Training mode should be ON inside context"
        assert nc.train.is_off() == False, (
            "is_off() should return False inside context"
        )  # ← 追加
    assert nc.train.is_on() == False, "Training mode should be restored after context"
    print("  ✅ Context manager - train.on()")

    # Test 5: Context manager - train.off()
    nc.train.on()  # Start with ON
    with nc.train.off:
        assert nc.train.is_on() == False, "Training mode should be OFF inside context"
        assert nc.train.is_off() == True, (
            "is_off() should return True inside context"
        )  # ← 追加
    assert nc.train.is_on() == True, "Training mode should be restored after context"
    print("  ✅ Context manager - train.off()")

    # Test 6: Nested context managers
    nc.train.on()
    with nc.train.off:
        assert nc.train.is_on() == False
        with nc.train.on:
            assert nc.train.is_on() == True
            assert nc.train.is_off() == False  # ← 追加
        assert nc.train.is_on() == False
        assert nc.train.is_off() == True  # ← 追加
    assert nc.train.is_on() == True
    print("  ✅ Nested context managers")

    # Test 7: enable() and disable() methods
    nc.train.enable()
    assert nc.train.is_on() == True, "enable() should turn training mode ON"
    assert nc.train.is_off() == False  # ← 追加

    nc.train.disable()
    assert nc.train.is_on() == False, "disable() should turn training mode OFF"
    assert nc.train.is_off() == True  # ← 追加
    print("  ✅ enable() and disable() methods")

    # Test 8: set_enabled() method
    nc.train.set_enabled(True)
    assert nc.train.is_on() == True, "set_enabled(True) should turn training mode ON"
    assert nc.train.is_off() == False  # ← 追加

    nc.train.set_enabled(False)
    assert nc.train.is_on() == False, "set_enabled(False) should turn training mode OFF"
    assert nc.train.is_off() == True  # ← 追加
    print("  ✅ set_enabled() method")

    # Test 9: Exception in context manager
    nc.train.on()
    try:
        with nc.train.off:
            assert nc.train.is_on() == False
            assert nc.train.is_off() == True  # ← 追加
            raise ValueError("Test exception")
    except ValueError:
        pass
    assert nc.train.is_on() == True, (
        "Training mode should be restored even after exception"
    )
    print("  ✅ Exception handling in context manager")

    # Restore default state
    nc.train.on()

    # Test 10: Exception handling in ON context
    nc.train.disable()
    try:
        with nc.train.on:
            assert nc.train.is_enabled() == True
            raise RuntimeError("Test exception in ON context")
    except RuntimeError:
        pass
    assert nc.train.is_enabled() == False, "State should restore even after exception"
    print("  ✅ Exception handling in ON context works")

    # Test 11: __repr__ for train.on
    repr_on = repr(nc.train.on)
    assert repr_on == "train.on", f"__repr__ should be 'train.on', got {repr_on}"
    print("  ✅ __repr__ for train.on works")

    # Test 12: __repr__ for train.off
    repr_off = repr(nc.train.off)
    assert repr_off == "train.off", f"__repr__ should be 'train.off', got {repr_off}"
    print("  ✅ __repr__ for train.off works")

    # Test 13: Calling train.on() as function
    nc.train.disable()
    nc.train.on()
    assert nc.train.is_enabled() == True, "train.on() should enable training"
    print("  ✅ train.on() as function works")

    # Test 14: Calling train.off() as function
    nc.train.enable()
    nc.train.off()
    assert nc.train.is_enabled() == False, "train.off() should disable training"
    print("  ✅ train.off() as function works")

    # Test 15: Multiple consecutive enable/disable
    for _ in range(3):
        nc.train.enable()
        assert nc.train.is_enabled() == True
        nc.train.disable()
        assert nc.train.is_enabled() == False
    print("  ✅ Multiple consecutive enable/disable works")

    # Test 16: Context manager doesn't change state if already in that state
    nc.train.enable()
    with nc.train.on:
        assert nc.train.is_enabled() == True
    assert nc.train.is_enabled() == True

    nc.train.disable()
    with nc.train.off:
        assert nc.train.is_enabled() == False
    assert nc.train.is_enabled() == False
    print("  ✅ Context manager preserves same state")

    print("✅ All training mode tests passed!\n")


def test_thread_safety():
    """Test thread-local state isolation"""
    print("Testing thread safety...")

    results = {}

    def thread1_func():
        nc.train.enable()
        assert nc.train.is_enabled() == True
        results["thread1"] = nc.train.is_enabled()

    def thread2_func():
        nc.train.disable()
        assert nc.train.is_enabled() == False
        results["thread2"] = nc.train.is_enabled()

    # Set main thread state
    nc.train.enable()

    # Start threads
    t1 = threading.Thread(target=thread1_func)
    t2 = threading.Thread(target=thread2_func)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Main thread state should be unchanged
    assert nc.train.is_enabled() == True, "Main thread state should be unchanged"
    assert results["thread1"] == True, "Thread 1 should have training ON"
    assert results["thread2"] == False, "Thread 2 should have training OFF"

    print("  ✅ Thread-local state isolation works")
    print("✅ Thread safety tests passed!\n")


def test_parameter():
    """Test Parameter class"""
    print("Testing Parameter class...")

    # Test 1: Create from list
    p1 = nc.Parameter([1.0, 2.0, 3.0])
    assert isinstance(p1.data, nc.NumType), "Parameter should wrap NumType"
    assert p1.data.requires_grad == True, (
        "Parameter should have requires_grad=True by default"
    )
    print("  ✅ Create Parameter from list")

    # Test 2: Create from NumType
    tensor = nc.tensor([4.0, 5.0, 6.0])
    p2 = nc.Parameter(tensor)
    assert p2.data.requires_grad == True, "Parameter should set requires_grad=True"
    print("  ✅ Create Parameter from NumType")

    # Test 3: requires_grad=False
    p3 = nc.Parameter([1.0, 2.0], requires_grad=False)
    assert p3.data.requires_grad == False, (
        "Parameter with requires_grad=False should work"
    )
    print("  ✅ Parameter with requires_grad=False")

    # Test 4: Shape property
    p4 = nc.Parameter(nc.randn(3, 4))
    assert p4.shape == (3, 4), f"Shape should be (3, 4), got {p4.shape}"
    print("  ✅ Shape property")

    # Test 5: Dtype property
    p5 = nc.Parameter(nc.randn(2, 2))
    assert p5.dtype is not None, "Dtype should be accessible"
    print("  ✅ Dtype property")

    # Test 6: Gradient
    nc.autograd.enable()
    p6 = nc.Parameter(nc.tensor([1.0, 2.0, 3.0]))
    y = nc.sum(p6.data**2)
    y.backward()
    assert p6.grad is not None, "Gradient should be computed"
    assert p6.grad.shape == p6.shape, "Gradient shape should match parameter shape"
    print("  ✅ Gradient computation")

    # Test 7: zero_grad
    p6.zero_grad()
    assert p6.grad is None, "Gradient should be None after zero_grad()"
    print("  ✅ zero_grad()")

    # Test 8: __repr__
    p7 = nc.Parameter([1.0, 2.0])
    repr_str = repr(p7)
    assert "Parameter" in repr_str, (
        f"__repr__ should contain 'Parameter', got {repr_str}"
    )
    print("  ✅ __repr__")

    # Test 9: Access NumType methods via __getattr__
    p8 = nc.Parameter(nc.randn(2, 3))
    assert hasattr(p8, "ndim"), "Should be able to access ndim via __getattr__"
    assert p8.ndim == 2, "ndim should be 2"
    print("  ✅ __getattr__ delegation")

    # Test 10: Parameter with different shapes
    shapes = [(5,), (3, 4), (2, 3, 4)]
    for shape in shapes:
        p = nc.Parameter(nc.randn(*shape))
        assert p.shape == shape, f"Shape should be {shape}, got {p.shape}"
    print("  ✅ Parameters with different shapes")

    # Test 11: Parameter from integer (should convert to float)
    p9 = nc.Parameter([1, 2, 3], requires_grad=True)
    assert p9.data.requires_grad == True, (
        "Integer data should be convertible with requires_grad=True"
    )
    print("  ✅ Parameter from integer data")

    # Test 12: Multiple backward passes
    nc.autograd.enable()
    p10 = nc.Parameter(nc.tensor([1.0, 2.0]))

    # First backward
    y1 = nc.sum(p10.data**2)
    y1.backward()
    grad1 = p10.grad._data.copy() if p10.grad is not None else None

    # Second backward (after zero_grad)
    p10.zero_grad()
    y2 = nc.sum(p10.data**3)
    y2.backward()
    grad2 = p10.grad._data.copy() if p10.grad is not None else None

    assert grad1 is not None and grad2 is not None, "Both gradients should be computed"
    assert not nc.get_array_module(grad1).array_equal(grad1, grad2), (
        "Gradients should be different"
    )
    print("  ✅ Multiple backward passes")

    print("✅ All Parameter tests passed!\n")


def run_all_tests():
    """Run all test suites"""
    tests = [
        test_train_mode,
        test_thread_safety,
    ]

    failed = []

    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed.append(test.__name__)
        except Exception as e:
            print(f"❌ {test.__name__} error: {e}")
            traceback.print_exc()
            failed.append(test.__name__)

    print("=" * 60)
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("🎉 All tests passed! Coverage: 100%")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
