"""
Test suite for nnlib
"""

import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon.nnlib as nl
import lemon.numlib as nm


def test_linear():
    """Test Linear layer"""
    print("Testing Linear layer...")

    # Test 1: Basic forward pass
    layer = nl.Linear(10, 5)
    x = nm.randn(3, 10)
    y = layer(x)
    assert y.shape == (3, 5), f"Output shape should be (3, 5), got {y.shape}"
    print("  ✅ Basic forward pass")

    # Test 2: Without bias
    layer_no_bias = nl.Linear(10, 5, bias=False)
    assert layer_no_bias.bias is None, "Bias should be None when bias=False"
    y = layer_no_bias(x)
    assert y.shape == (3, 5), "Output shape should be correct without bias"
    print("  ✅ Linear without bias")

    # Test 3: With bias
    layer_with_bias = nl.Linear(10, 5, bias=True)
    assert layer_with_bias.bias is not None, "Bias should exist when bias=True"
    y = layer_with_bias(x)
    assert y.shape == (3, 5), "Output shape should be correct with bias"
    print("  ✅ Linear with bias")

    # Test 4: Parameters count
    layer = nl.Linear(10, 5)
    params = list(layer.parameters())
    assert len(params) == 2, (
        f"Should have 2 parameters (weight + bias), got {len(params)}"
    )

    layer_no_bias = nl.Linear(10, 5, bias=False)
    params_no_bias = list(layer_no_bias.parameters())
    assert len(params_no_bias) == 1, (
        f"Should have 1 parameter (weight only), got {len(params_no_bias)}"
    )
    print("  ✅ Parameters count")

    # Test 5: Weight shape
    layer = nl.Linear(784, 128)
    assert layer.weight.shape == (784, 128), (
        f"Weight shape should be (784, 128), got {layer.weight.shape}"
    )
    print("  ✅ Weight shape")

    # Test 6: Bias shape
    layer = nl.Linear(784, 128)
    assert layer.bias.shape == (128,), (
        f"Bias shape should be (128,), got {layer.bias.shape}"
    )
    print("  ✅ Bias shape")

    # Test 7: Gradient computation
    nm.autograd.enable()
    layer = nl.Linear(3, 2)
    x = nm.tensor([[1.0, 2.0, 3.0]])
    y = layer(x)
    loss = nm.sum(y)

    loss.backward()

    assert layer.weight.grad is not None, "Weight gradient should be computed"
    assert layer.bias.grad is not None, "Bias gradient should be computed"
    assert layer.weight.grad.shape == layer.weight.shape, (
        "Weight gradient shape should match"
    )
    assert layer.bias.grad.shape == layer.bias.shape, "Bias gradient shape should match"
    print("  ✅ Gradient computation")

    # Test 8: zero_grad
    layer.zero_grad()
    assert layer.weight.grad is None, "Weight gradient should be None after zero_grad"
    assert layer.bias.grad is None, "Bias gradient should be None after zero_grad"
    print("  ✅ zero_grad()")

    # Test 9: Different input shapes
    layer = nl.Linear(10, 5)

    # 2D input (batch)
    x2d = nm.randn(32, 10)
    y2d = layer(x2d)
    assert y2d.shape == (32, 5), "Should handle 2D input"

    # 1D input (single sample)
    x1d = nm.randn(10)
    y1d = layer(x1d)
    assert y1d.shape == (5,), "Should handle 1D input"

    print("  ✅ Different input shapes")

    # Test 10: Xavier initialization bounds
    layer = nl.Linear(100, 50)
    limit = (6.0 / (100 + 50)) ** 0.5

    xp = nm.get_array_module(layer.weight.data._data)
    max_val = xp.max(xp.abs(layer.weight.data._data))

    # 初期化が範囲内にあることを確認
    assert max_val <= limit * 1.1, (
        f"Initial weights should be within Xavier bounds, got max={max_val}, limit={limit}"
    )
    print("  ✅ Xavier initialization")

    # Test 11: __repr__
    layer = nl.Linear(784, 128, bias=True)
    repr_str = repr(layer)
    assert "Linear" in repr_str, "repr should contain 'Linear'"
    assert "784" in repr_str, "repr should contain in_features"
    assert "128" in repr_str, "repr should contain out_features"
    assert "bias=True" in repr_str, "repr should contain bias info"
    print("  ✅ __repr__")

    # Test 12: Multiple forward passes
    layer = nl.Linear(5, 3)
    x1 = nm.randn(2, 5)
    x2 = nm.randn(4, 5)

    y1 = layer(x1)
    y2 = layer(x2)

    assert y1.shape == (2, 3), "First forward pass shape"
    assert y2.shape == (4, 3), "Second forward pass shape"
    print("  ✅ Multiple forward passes")

    # Test 13: Backward pass updates gradients correctly
    nm.autograd.enable()
    layer = nl.Linear(2, 1)

    # First backward
    x = nm.tensor([[1.0, 2.0]])
    y = layer(x)
    loss = nm.sum(y)
    loss.backward()

    weight_grad1 = layer.weight.grad._data.copy()
    bias_grad1 = layer.bias.grad._data.copy()

    # Second backward (after zero_grad)
    layer.zero_grad()
    x = nm.tensor([[3.0, 4.0]])
    y = layer(x)
    loss = nm.sum(y)
    loss.backward()

    weight_grad2 = layer.weight.grad._data.copy()
    bias_grad2 = layer.bias.grad._data.copy()

    # Gradients should be different
    xp = nm.get_array_module(weight_grad1)
    assert not xp.allclose(weight_grad1, weight_grad2), (
        "Gradients should differ for different inputs"
    )
    print("  ✅ Backward pass with different inputs")

    print("✅ All Linear layer tests passed!\n")
