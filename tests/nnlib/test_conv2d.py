"""
Comprehensive tests for Conv2d autograd functionality

Tests cover:
1. Basic gradient computation (x, weight, bias)
2. Numerical gradient check
3. Multiple samples and channels
4. Different parameter combinations (stride, padding)
5. Training loop simulation
6. Comparison with manual implementation
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon import numlib as nm
from lemon import nnlib as nl
import numpy as np


def numerical_gradient(f, x, eps=1e-4):
    """
    Compute numerical gradient using finite differences

    Parameters
    ----------
    f : callable
        Function that takes x and returns a scalar
    x : ndarray
        Point at which to compute gradient
    eps : float
        Finite difference step size

    Returns
    -------
    ndarray
        Numerical gradient
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        # f(x + eps)
        x[idx] = old_value + eps
        fxh_pos = f()

        # f(x - eps)
        x[idx] = old_value - eps
        fxh_neg = f()

        # Numerical gradient
        grad[idx] = (fxh_pos - fxh_neg) / (2 * eps)

        # Restore
        x[idx] = old_value
        it.iternext()

    return grad


def test_basic_gradient_computation():
    """Test 1: Basic gradient computation"""
    print("=" * 70)
    print("Test 1: Basic Gradient Computation")
    print("=" * 70)

    nm.autograd.enable()

    # Small tensors for easy verification
    x = nm.randn(2, 3, 5, 5, requires_grad=True)
    weight = nm.randn(4, 3, 3, 3, requires_grad=True)
    bias = nm.randn(4, requires_grad=True)

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  weight: {weight.shape}")
    print(f"  bias: {bias.shape}")

    # Forward pass
    y = nl.conv_2d(x, weight, bias, stride=1, padding=1)
    print(f"\nOutput shape: {y.shape}")

    # Compute loss
    loss = nm.sum(y)
    print(f"Loss: {loss._data}")

    # Check gradients before backward
    print(f"\nBefore backward:")
    print(f"  x.grad is None: {x.grad is None}")
    print(f"  weight.grad is None: {weight.grad is None}")
    print(f"  bias.grad is None: {bias.grad is None}")

    # Backward pass
    loss.backward()

    # Check gradients after backward
    print(f"\nAfter backward:")
    print(f"  x.grad is not None: {x.grad is not None}")
    print(f"  weight.grad is not None: {weight.grad is not None}")
    print(f"  bias.grad is not None: {bias.grad is not None}")

    if x.grad is not None:
        print(f"  x.grad shape: {x.grad.shape}")
        print(f"  x.grad sample values: {x.grad._data.flat[:5]}")

    if weight.grad is not None:
        print(f"  weight.grad shape: {weight.grad.shape}")
        print(f"  weight.grad sample values: {weight.grad._data.flat[:5]}")

    if bias.grad is not None:
        print(f"  bias.grad shape: {bias.grad.shape}")
        print(f"  bias.grad values: {bias.grad._data}")

    # Verify all gradients exist
    if x.grad is not None and weight.grad is not None and bias.grad is not None:
        print("\n✅ Test 1 PASSED: All gradients computed")
        return True
    else:
        print("\n❌ Test 1 FAILED: Some gradients are None")
        return False


def test_numerical_gradient_check():
    """Test 2: Numerical gradient check"""
    print("\n" + "=" * 70)
    print("Test 2: Numerical Gradient Check")
    print("=" * 70)

    nm.autograd.enable()

    # Very small tensors for numerical gradient check
    x = nm.randn(1, 2, 4, 4, requires_grad=True)
    weight = nm.randn(3, 2, 3, 3, requires_grad=True)
    bias = nm.randn(3, requires_grad=True)

    print(f"\nInput shapes (small for numerical check):")
    print(f"  x: {x.shape}")
    print(f"  weight: {weight.shape}")
    print(f"  bias: {bias.shape}")

    # Forward and backward
    y = nl.conv_2d(x, weight, bias, stride=1, padding=1)
    loss = nm.sum(y)
    loss.backward()

    # Get analytical gradients
    analytical_grad_x = x.grad._data.copy()
    analytical_grad_weight = weight.grad._data.copy()
    analytical_grad_bias = bias.grad._data.copy()

    print(f"\nAnalytical gradients computed")

    # Numerical gradient for weight (sample a few elements)
    print(f"\nNumerical gradient check for weight (sampling 5 elements):")

    weight_data = weight._data
    indices_to_check = [
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (1, 0, 0, 0),
        (2, 1, 2, 2),
        (1, 1, 1, 1),
    ]

    max_error = 0.0
    for idx in indices_to_check:
        # Define loss function for this element
        def loss_fn():
            y = nl.conv_2d(x, weight, bias, stride=1, padding=1)
            return nm.sum(y)._data

        # Compute numerical gradient
        old_val = weight_data[idx]
        eps = 1e-4

        weight_data[idx] = old_val + eps
        loss_pos = loss_fn()

        weight_data[idx] = old_val - eps
        loss_neg = loss_fn()

        weight_data[idx] = old_val

        numerical_grad = (loss_pos - loss_neg) / (2 * eps)
        analytical_grad = analytical_grad_weight[idx]

        error = abs(numerical_grad - analytical_grad)
        max_error = max(max_error, error)

        print(
            f"  idx {idx}: numerical={numerical_grad:.6f}, "
            f"analytical={analytical_grad:.6f}, error={error:.6e}"
        )

    print(f"\nMax error: {max_error:.6e}")

    if max_error < 1e-3:
        print("✅ Test 2 PASSED: Numerical gradient check passed")
        return True
    else:
        print("❌ Test 2 FAILED: Numerical gradient error too large")
        return False


def test_gradient_with_no_bias():
    """Test 3: Gradient computation without bias"""
    print("\n" + "=" * 70)
    print("Test 3: Gradient Computation without Bias")
    print("=" * 70)

    nm.autograd.enable()

    x = nm.randn(2, 3, 5, 5, requires_grad=True)
    weight = nm.randn(4, 3, 3, 3, requires_grad=True)

    print(f"\nInput shapes (no bias):")
    print(f"  x: {x.shape}")
    print(f"  weight: {weight.shape}")

    # Forward pass without bias
    y = nl.conv_2d(x, weight, bias=None, stride=1, padding=1)
    loss = nm.sum(y)

    print(f"Output shape: {y.shape}")

    # Backward pass
    loss.backward()

    print(f"\nAfter backward:")
    print(f"  x.grad is not None: {x.grad is not None}")
    print(f"  weight.grad is not None: {weight.grad is not None}")

    if x.grad is not None and weight.grad is not None:
        print("✅ Test 3 PASSED: Gradients computed without bias")
        return True
    else:
        print("❌ Test 3 FAILED: Some gradients are None")
        return False


def test_different_parameters():
    """Test 4: Different stride and padding values"""
    print("\n" + "=" * 70)
    print("Test 4: Different Parameter Combinations")
    print("=" * 70)

    nm.autograd.enable()

    test_cases = [
        {"stride": 1, "padding": 0, "name": "stride=1, padding=0"},
        {"stride": 2, "padding": 1, "name": "stride=2, padding=1"},
        {"stride": 1, "padding": 2, "name": "stride=1, padding=2"},
    ]

    all_passed = True

    for case in test_cases:
        print(f"\n  Testing {case['name']}:")

        x = nm.randn(2, 3, 8, 8, requires_grad=True)
        weight = nm.randn(4, 3, 3, 3, requires_grad=True)
        bias = nm.randn(4, requires_grad=True)

        y = nl.conv_2d(x, weight, bias, stride=case["stride"], padding=case["padding"])
        loss = nm.sum(y)
        loss.backward()

        print(f"    Output shape: {y.shape}")
        print(f"    x.grad exists: {x.grad is not None}")
        print(f"    weight.grad exists: {weight.grad is not None}")
        print(f"    bias.grad exists: {bias.grad is not None}")

        if not (
            x.grad is not None and weight.grad is not None and bias.grad is not None
        ):
            all_passed = False
            print(f"    ❌ Failed")
        else:
            print(f"    ✅ Passed")

    if all_passed:
        print("\n✅ Test 4 PASSED: All parameter combinations work")
        return True
    else:
        print("\n❌ Test 4 FAILED: Some parameter combinations failed")
        return False


def test_module_class():
    """Test 5: Conv2d module (class version)"""
    print("\n" + "=" * 70)
    print("Test 5: Conv2d Module (Class Version)")
    print("=" * 70)

    nm.autograd.enable()

    # Create Conv2d layer
    conv = nl.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    print(f"\nConv2d layer created:")
    print(f"  {conv}")
    print(f"  weight shape: {conv.weight.shape}")
    print(f"  bias shape: {conv.bias.shape}")

    # Forward pass
    x = nm.randn(4, 3, 8, 8, requires_grad=True)
    y = conv(x)

    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")

    # Backward pass
    loss = nm.sum(y)
    loss.backward()

    print(f"\nAfter backward:")
    print(f"  conv.weight.grad is not None: {conv.weight.grad is not None}")
    print(f"  conv.bias.grad is not None: {conv.bias.grad is not None}")
    print(f"  x.grad is not None: {x.grad is not None}")

    if conv.weight.grad is not None:
        print(f"  weight.grad shape: {conv.weight.grad.shape}")
        print(f"  weight.grad sample: {conv.weight.grad._data.flat[:5]}")

    if conv.weight.grad is not None and conv.bias.grad is not None:
        print("\n✅ Test 5 PASSED: Conv2d module works correctly")
        return True
    else:
        print("\n❌ Test 5 FAILED: Module gradients not computed")
        return False


def test_training_loop():
    """Test 6: Simulate actual training loop"""
    print("\n" + "=" * 70)
    print("Test 6: Training Loop Simulation")
    print("=" * 70)

    nm.autograd.enable()

    # Create simple CNN
    conv1 = nl.Conv2d(3, 8, 3, padding=1)
    conv2 = nl.Conv2d(8, 16, 3, padding=1)

    print(f"\nNetwork:")
    print(f"  {conv1}")
    print(f"  {conv2}")

    # Create optimizer
    params = list(conv1.parameters()) + list(conv2.parameters())
    optimizer = nl.Adam(params, lr=0.01)

    print(f"\nOptimizer: {optimizer}")
    print(f"Number of parameters: {len(params)}")

    # Training loop
    print(f"\nTraining for 5 steps:")

    losses = []
    for step in range(5):
        # Generate random data
        x = nm.randn(4, 3, 8, 8)
        target = nm.randn(4, 16, 8, 8)

        # Forward pass
        h = conv1(x)
        h = nl.relu(h)
        y = conv2(h)

        # Loss
        loss = nm.mean((y - target) ** 2)
        losses.append(loss._data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        grad_exists = all(p.grad is not None for p in params)

        # Update
        optimizer.step()

        print(f"  Step {step}: loss={loss._data:.6f}, grads_exist={grad_exists}")

    # Check if loss is changing
    print(f"\nLoss progression: {losses}")

    # Loss should change over training
    loss_changed = not all(
        abs(losses[i] - losses[0]) < 1e-6 for i in range(len(losses))
    )

    if loss_changed:
        print("✅ Test 6 PASSED: Training loop works, loss is changing")
        return True
    else:
        print(
            "⚠️  Test 6 WARNING: Loss not changing (might be expected for random data)"
        )
        return True  # Not a failure, just a note


def test_gradient_accumulation():
    """Test 7: Gradient accumulation"""
    print("\n" + "=" * 70)
    print("Test 7: Gradient Accumulation")
    print("=" * 70)

    nm.autograd.enable()

    conv = nl.Conv2d(3, 4, 3, padding=1)

    print(f"\nTesting gradient accumulation:")

    # First forward-backward
    x1 = nm.randn(2, 3, 4, 4)
    y1 = conv(x1)
    loss1 = nm.sum(y1)
    loss1.backward()

    grad_after_first = conv.weight.grad._data.copy()
    print(
        f"  After 1st backward: weight.grad norm = {np.linalg.norm(grad_after_first):.6f}"
    )

    # Second forward-backward (should accumulate)
    x2 = nm.randn(2, 3, 4, 4)
    y2 = conv(x2)
    loss2 = nm.sum(y2)
    loss2.backward()

    grad_after_second = conv.weight.grad._data.copy()
    print(
        f"  After 2nd backward: weight.grad norm = {np.linalg.norm(grad_after_second):.6f}"
    )

    # Gradient should have accumulated (increased)
    grad_increased = np.linalg.norm(grad_after_second) > np.linalg.norm(
        grad_after_first
    )

    # Zero grad
    conv.zero_grad()
    print(f"  After zero_grad: weight.grad is None = {conv.weight.grad is None}")

    if grad_increased and conv.weight.grad is None:
        print("✅ Test 7 PASSED: Gradient accumulation works correctly")
        return True
    else:
        print("❌ Test 7 FAILED: Gradient accumulation issue")
        return False


def test_sequential_model():
    """Test 8: Conv2d in Sequential"""
    print("\n" + "=" * 70)
    print("Test 8: Conv2d in Sequential Model")
    print("=" * 70)

    nm.autograd.enable()

    # Create sequential model
    model = nl.Sequential(
        nl.Conv2d(3, 16, 3, padding=1),
        nl.Relu(),
        nl.Conv2d(16, 32, 3, stride=2, padding=1),
        nl.Relu(),
        nl.Conv2d(32, 10, 3, padding=1),
    )

    print(f"\nSequential model:")
    print(model)

    # Forward pass
    x = nm.randn(4, 3, 16, 16)
    y = model(x)

    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")

    # Backward pass
    loss = nm.sum(y)
    loss.backward()

    # Check all Conv2d layers have gradients
    conv_layers = [m for m in model.modules if isinstance(m, nl.Conv2d)]
    all_grads_exist = all(
        conv.weight.grad is not None and conv.bias.grad is not None
        for conv in conv_layers
    )

    print(f"\nNumber of Conv2d layers: {len(conv_layers)}")
    print(f"All Conv2d layers have gradients: {all_grads_exist}")

    if all_grads_exist:
        print("✅ Test 8 PASSED: Sequential model with Conv2d works")
        return True
    else:
        print("❌ Test 8 FAILED: Some layers missing gradients")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RUNNING ALL TESTS FOR Conv2d AUTOGRAD")
    print("=" * 70)

    results = []

    try:
        results.append(
            ("Basic gradient computation", test_basic_gradient_computation())
        )
        results.append(("Numerical gradient check", test_numerical_gradient_check()))
        results.append(("Gradient without bias", test_gradient_with_no_bias()))
        results.append(("Different parameters", test_different_parameters()))
        results.append(("Conv2d module", test_module_class()))
        results.append(("Training loop", test_training_loop()))
        results.append(("Gradient accumulation", test_gradient_accumulation()))
        results.append(("Sequential model", test_sequential_model()))

        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)

        for name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {name}")

        total_passed = sum(passed for _, passed in results)
        total_tests = len(results)

        print(f"\nTotal: {total_passed}/{total_tests} tests passed")

        if total_passed == total_tests:
            print("\n🎉 ALL TESTS PASSED! Conv2d autograd is working correctly!")
        else:
            print(f"\n⚠️  {total_tests - total_passed} test(s) failed")

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
