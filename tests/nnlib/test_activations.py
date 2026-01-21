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


def test_activation_functions():
    """Test activation functions (functional API)"""

    print("Testing activation functions (functional)...")

    # ========================================
    # relu
    # ========================================
    print("  Testing relu()...")

    # Test 1: relu forward
    x = nm.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = nl.relu(x)
    expected = nm.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    xp = nm.get_array_module(y._data)
    assert xp.allclose(y._data, expected._data), "relu forward should be correct"
    print("    ✅ relu forward")

    # Test 2: relu gradient
    nm.autograd.enable()
    x = nm.tensor([-1.0, 0.5, 1.0], requires_grad=True)
    y = nl.relu(x)
    loss = nm.sum(y)  # 修正: スカラー化
    loss.backward()
    expected_grad = nm.tensor([0.0, 1.0, 1.0])
    assert xp.allclose(x.grad._data, expected_grad._data), (
        "relu gradient should be correct"
    )
    print("    ✅ relu gradient")

    # ========================================
    # sigmoid
    # ========================================
    print("  Testing sigmoid()...")

    # Test 3: sigmoid forward
    x = nm.tensor([0.0])
    y = nl.sigmoid(x)
    assert xp.allclose(y._data, [0.5], atol=1e-6), "sigmoid(0) should be 0.5"

    x = nm.tensor([-100.0, 0.0, 100.0])
    y = nl.sigmoid(x)
    assert y._data[0] < 0.01, "sigmoid(-100) should be close to 0"
    assert xp.allclose(y._data[1], 0.5, atol=1e-6), "sigmoid(0) should be 0.5"
    assert y._data[2] > 0.99, "sigmoid(100) should be close to 1"
    print("    ✅ sigmoid forward")

    # Test 4: sigmoid gradient - 修正: スカラーに変更
    nm.autograd.enable()
    x = nm.tensor(0.0, requires_grad=True)  # shape=() - スカラー
    y = nl.sigmoid(x)
    y.backward()  # OK
    assert xp.allclose(x.grad._data, 0.25, atol=1e-6), (
        "sigmoid gradient at 0 should be 0.25"
    )
    print("    ✅ sigmoid gradient")

    # ========================================
    # tanh
    # ========================================
    print("  Testing tanh()...")

    # Test 5: tanh forward
    x = nm.tensor([0.0])
    y = nl.tanh(x)
    assert xp.allclose(y._data, [0.0], atol=1e-6), "tanh(0) should be 0"

    x = nm.tensor([-1.0, 0.0, 1.0])
    y = nl.tanh(x)
    assert xp.allclose(y._data[0], -y._data[2], atol=1e-6), "tanh should be symmetric"
    print("    ✅ tanh forward")

    # Test 6: tanh gradient - 修正: スカラーに変更
    nm.autograd.enable()
    x = nm.tensor(0.0, requires_grad=True)  # shape=() - スカラー
    y = nl.tanh(x)
    y.backward()  # OK
    assert xp.allclose(x.grad._data, 1.0, atol=1e-6), "tanh gradient at 0 should be 1"
    print("    ✅ tanh gradient")

    # ========================================
    # softmax
    # ========================================
    print("  Testing softmax()...")

    # Test 7: softmax forward
    x = nm.tensor([[1.0, 2.0, 3.0]])
    y = nl.softmax(x)
    sum_y = nm.sum(y, axis=-1)
    assert xp.allclose(sum_y._data, [1.0], atol=1e-6), "softmax sum should be 1"
    assert xp.all(y._data > 0), "softmax outputs should be positive"
    print("    ✅ softmax forward")

    # Test 8: softmax with different axis
    x = nm.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = nl.softmax(x, axis=0)
    sum_y = nm.sum(y, axis=0)
    assert xp.allclose(sum_y._data, [1.0, 1.0], atol=1e-6), (
        "softmax sum along axis 0 should be 1"
    )
    print("    ✅ softmax with different axis")

    # Test 9: softmax gradient - 修正: スカラー化を追加
    nm.autograd.enable()
    x = nm.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = nl.softmax(x)
    loss = nm.sum(y)  # 修正: スカラー化
    loss.backward()
    assert x.grad is not None, "softmax gradient should be computed"
    print("    ✅ softmax gradient")

    print("✅ All activation function tests passed!\n")


def test_activation_modules():
    """Test activation modules (class-based API)"""

    print("Testing activation modules (class-based)...")

    # Test 1: Relu module
    relu = nl.Relu()
    x = nm.tensor([-1.0, 0.0, 1.0])
    y = relu(x)
    xp = nm.get_array_module(y._data)
    expected = nm.tensor([0.0, 0.0, 1.0])
    assert xp.allclose(y._data, expected._data), "Relu module should work"
    assert repr(relu) == "Relu()", "Relu repr should be correct"
    print("  ✅ Relu module")

    # Test 2: Sigmoid module
    sigmoid = nl.Sigmoid()
    x = nm.tensor([0.0])
    y = sigmoid(x)
    assert xp.allclose(y._data, [0.5], atol=1e-6), "Sigmoid module should work"
    assert repr(sigmoid) == "Sigmoid()", "Sigmoid repr should be correct"
    print("  ✅ Sigmoid module")

    # Test 3: Tanh module
    tanh_mod = nl.Tanh()
    x = nm.tensor([0.0])
    y = tanh_mod(x)
    assert xp.allclose(y._data, [0.0], atol=1e-6), "Tanh module should work"
    assert repr(tanh_mod) == "Tanh()", "Tanh repr should be correct"
    print("  ✅ Tanh module")

    # Test 4: Softmax module
    softmax = nl.Softmax(axis=-1)
    x = nm.tensor([[1.0, 2.0, 3.0]])
    y = softmax(x)
    sum_y = nm.sum(y, axis=-1)
    assert xp.allclose(sum_y._data, [1.0], atol=1e-6), "Softmax module should work"
    assert "Softmax" in repr(softmax), "Softmax repr should be correct"
    print("  ✅ Softmax module")

    # Test 5: Gradient through modules - 修正: スカラー化を追加
    nm.autograd.enable()
    relu = nl.Relu()
    x = nm.tensor([1.0, -1.0, 2.0], requires_grad=True)
    y = relu(x)
    loss = nm.sum(y)  # 修正: スカラー化
    loss.backward()
    expected_grad = nm.tensor([1.0, 0.0, 1.0])
    assert xp.allclose(x.grad._data, expected_grad._data), (
        "Gradient through Relu module should work"
    )
    print("  ✅ Gradient through modules")

    # Test 6: Modules in Sequential
    model = nl.Sequential(nl.Linear(10, 20), nl.Relu(), nl.Linear(20, 5), nl.Sigmoid())

    nm.autograd.enable()
    x = nm.randn(3, 10)
    y = model(x)

    assert y.shape == (3, 5), "Model output shape should be correct"
    assert xp.all(y._data >= 0) and xp.all(y._data <= 1), (
        "Sigmoid outputs should be in [0, 1]"
    )

    # Backward - 修正: スカラー化を追加
    loss = nm.sum(y)  # 修正: スカラー化
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "All parameters should have gradients"

    print("  ✅ Modules in Sequential")

    # Test 7: Functional vs Module equivalence - 修正: スカラー化を追加
    nm.autograd.enable()
    x = nm.tensor([1.0, -1.0, 2.0], requires_grad=True)

    # Functional
    y_func = nl.relu(x)
    loss_func = nm.sum(y_func)  # 修正: スカラー化
    loss_func.backward()
    grad_func = x.grad._data.copy()

    # Module
    x.zero_grad()
    relu_mod = nl.Relu()
    y_mod = relu_mod(x)
    loss_mod = nm.sum(y_mod)  # 修正: スカラー化
    loss_mod.backward()
    grad_mod = x.grad._data.copy()

    assert xp.allclose(grad_func, grad_mod), (
        "Functional and module should give same gradients"
    )
    print("  ✅ Functional vs Module equivalence")

    print("✅ All activation module tests passed!\n")


def test_activation_edge_cases():
    """Test activation function edge cases for coverage"""

    print("Testing activation edge cases...")

    # Test 1: Activations with autograd disabled
    nm.autograd.disable()

    x = nm.tensor([1.0, -1.0, 2.0])
    y_relu = nl.relu(x)
    assert y_relu.requires_grad == False, (
        "relu should not require grad when autograd is off"
    )

    y_sigmoid = nl.sigmoid(x)
    assert y_sigmoid.requires_grad == False, (
        "sigmoid should not require grad when autograd is off"
    )

    y_tanh = nl.tanh(x)
    assert y_tanh.requires_grad == False, (
        "tanh should not require grad when autograd is off"
    )

    y_softmax = nl.softmax(x)
    assert y_softmax.requires_grad == False, (
        "softmax should not require grad when autograd is off"
    )

    print("  ✅ Activations with autograd disabled")

    # Test 2: Activations with requires_grad=False
    nm.autograd.enable()

    x = nm.tensor([1.0, -1.0, 2.0], requires_grad=False)
    y_relu = nl.relu(x)
    assert y_relu.requires_grad == False, (
        "relu should not require grad when input doesn't"
    )

    y_sigmoid = nl.sigmoid(x)
    assert y_sigmoid.requires_grad == False, (
        "sigmoid should not require grad when input doesn't"
    )

    y_tanh = nl.tanh(x)
    assert y_tanh.requires_grad == False, (
        "tanh should not require grad when input doesn't"
    )

    y_softmax = nl.softmax(x)
    assert y_softmax.requires_grad == False, (
        "softmax should not require grad when input doesn't"
    )

    print("  ✅ Activations with requires_grad=False")

    # Test 3: Backward with None gradient (edge case)
    nm.autograd.enable()
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    y = nl.relu(x)

    # Manually set grad to None and call backward
    y.grad = None
    y._backward()  # Should handle None gracefully

    print("  ✅ Backward with None gradient")

    # Test 4: Multiple backward calls (gradient accumulation)
    nm.autograd.enable()
    x = nm.tensor([1.0, -1.0, 2.0], requires_grad=True)

    # First backward
    y1 = nl.relu(x)
    loss1 = nm.sum(y1)
    loss1.backward()
    grad1 = x.grad._data.copy()

    # Second backward without zero_grad (should accumulate)
    y2 = nl.relu(x)
    loss2 = nm.sum(y2)
    loss2.backward()
    grad2 = x.grad._data.copy()

    xp = nm.get_array_module(grad1)
    assert xp.allclose(grad2, grad1 * 2), "Gradients should accumulate"

    print("  ✅ Gradient accumulation")

    # Test 5: Softmax with different axes (more coverage)
    x = nm.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)

    y_axis0 = nl.softmax(x, axis=0)
    y_axis1 = nl.softmax(x, axis=1)
    y_axis2 = nl.softmax(x, axis=2)

    # Backward for each
    for y in [y_axis0, y_axis1, y_axis2]:
        x.zero_grad()
        loss = nm.sum(y)
        loss.backward()
        assert x.grad is not None, "Softmax gradient should work for all axes"

    print("  ✅ Softmax with multiple axes")

    # Test 6: Edge values for sigmoid
    x_extreme = nm.tensor([-1000.0, 1000.0], requires_grad=True)
    y_extreme = nl.sigmoid(x_extreme)
    loss_extreme = nm.sum(y_extreme)
    loss_extreme.backward()

    assert x_extreme.grad is not None, "Sigmoid should handle extreme values"
    print("  ✅ Sigmoid with extreme values")

    print("✅ All activation edge cases passed!\n")


def test_activation_gradient_flow():
    """Test gradient flow through activation functions using public API"""

    print("Testing activation gradient flow...")

    # ========================================
    # Test sigmoid gradient computation
    # ========================================
    nm.autograd.enable()

    # Test at multiple points
    x = nm.tensor([0.0, 1.0, -1.0, 2.0, -2.0], requires_grad=True)
    y = nl.sigmoid(x)
    loss = nm.sum(y)
    loss.backward()

    # Verify gradient exists
    assert x.grad is not None, "Sigmoid should propagate gradients"

    # Verify gradient values (σ'(x) = σ(x)(1-σ(x)))
    xp = nm.get_array_module(x._data)
    sigmoid_vals = 1.0 / (1.0 + xp.exp(-x._data))
    expected_grad = sigmoid_vals * (1 - sigmoid_vals)
    assert xp.allclose(x.grad._data, expected_grad, atol=1e-6), (
        "Sigmoid gradient should match σ(x)(1-σ(x))"
    )

    print("  ✅ sigmoid gradient computation")

    # ========================================
    # Test softmax gradient computation
    # ========================================
    x = nm.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = nl.softmax(x, axis=-1)

    # Use weighted sum for non-uniform gradient
    weights = nm.tensor([[0.1, 0.3, 0.6]])
    loss = nm.sum(y * weights)
    loss.backward()

    assert x.grad is not None, "Softmax should propagate gradients"

    # Gradient should sum to approximately 0 (softmax property)
    grad_sum = xp.sum(x.grad._data)
    assert abs(grad_sum) < 1e-5, "Softmax gradient should sum to ~0"

    print("  ✅ softmax gradient computation")

    # ========================================
    # Test gradient accumulation
    # ========================================
    x = nm.tensor([0.5], requires_grad=True)

    # First pass
    y1 = nl.sigmoid(x)
    loss1 = nm.sum(y1)
    loss1.backward()
    grad_after_first = float(x.grad._data[0])

    # Second pass (should accumulate)
    y2 = nl.sigmoid(x)
    loss2 = nm.sum(y2)
    loss2.backward()
    grad_after_second = float(x.grad._data[0])

    assert abs(grad_after_second - 2 * grad_after_first) < 1e-6, (
        "Gradients should accumulate correctly"
    )

    print("  ✅ gradient accumulation")

    # ========================================
    # Test zero gradient initialization
    # ========================================
    x = nm.tensor([1.0], requires_grad=True)
    x.grad = None  # Ensure clean state

    y = nl.sigmoid(x)
    loss = nm.sum(y)
    loss.backward()

    assert x.grad is not None, "Gradient should be created from None"
    assert x.grad._data.shape == x._data.shape, "Gradient shape should match input"

    print("  ✅ gradient initialization from None")

    # ========================================
    # Test chain rule through activations
    # ========================================
    x = nm.tensor([[1.0, 2.0]], requires_grad=True)

    # Chain: x -> sigmoid -> softmax -> loss
    y1 = nl.sigmoid(x)
    y2 = nl.softmax(y1, axis=-1)
    loss = nm.sum(y2)
    loss.backward()

    assert x.grad is not None, "Gradient should flow through chain"
    assert not xp.any(xp.isnan(x.grad._data)), "No NaN in gradients"
    assert not xp.any(xp.isinf(x.grad._data)), "No Inf in gradients"

    print("  ✅ chain rule through multiple activations")

    # ========================================
    # Test requires_grad propagation states
    # ========================================

    # State 1: autograd disabled
    nm.autograd.disable()
    x = nm.tensor([1.0], requires_grad=True)
    y = nl.sigmoid(x)
    assert not y.requires_grad, "autograd disabled -> no grad required"

    # State 2: input doesn't require grad
    nm.autograd.enable()
    x = nm.tensor([1.0], requires_grad=False)
    y = nl.sigmoid(x)
    assert not y.requires_grad, "input no grad -> output no grad"

    # State 3: normal operation
    x = nm.tensor([1.0], requires_grad=True)
    y = nl.sigmoid(x)
    assert y.requires_grad, "input requires grad -> output requires grad"

    print("  ✅ requires_grad propagation")

    # ========================================
    # Test numerical stability
    # ========================================

    # Large positive values
    x = nm.tensor([100.0], requires_grad=True)
    y = nl.sigmoid(x)
    loss = nm.sum(y)
    loss.backward()
    assert abs(y._data[0] - 1.0) < 1e-6, "sigmoid(100) should be ~1"
    assert abs(x.grad._data[0]) < 1e-6, "gradient should be ~0 for saturated sigmoid"

    # Large negative values
    x = nm.tensor([-100.0], requires_grad=True)
    y = nl.sigmoid(x)
    loss = nm.sum(y)
    loss.backward()
    assert abs(y._data[0]) < 1e-6, "sigmoid(-100) should be ~0"
    assert abs(x.grad._data[0]) < 1e-6, "gradient should be ~0 for saturated sigmoid"

    print("  ✅ numerical stability")

    print("✅ All activation gradient flow tests passed!\n")


def test_new_activation_functions():
    """Test newly added ONNX-compliant activation functions"""

    print("Testing newly added activation functions...")

    xp = nm.get_array_module(nm.tensor([0.0])._data)

    # ========================================
    # Selu
    # ========================================
    print("  Testing Selu...")
    x = nm.tensor([-1.0, 0.0, 1.0])
    y = nl.selu(x)

    # Test forward pass
    assert y.shape == x.shape, "Selu output shape should match input"
    assert y._data[1] == 0.0, "Selu(0) should be 0"
    assert y._data[2] > 0, "Selu of positive should be positive"

    # Test module
    selu_mod = nl.Selu()
    y_mod = selu_mod(x)
    assert xp.allclose(y._data, y_mod._data), "Selu function and module should match"
    assert "Selu" in repr(selu_mod), "Selu repr should be correct"

    # Test gradient
    nm.autograd.enable()
    x_grad = nm.tensor([1.0], requires_grad=True)
    y_grad = nl.selu(x_grad)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "Selu gradient should be computed"
    print("    ✅ Selu")

    # ========================================
    # Celu
    # ========================================
    print("  Testing Celu...")
    x = nm.tensor([-1.0, 0.0, 1.0])
    y = nl.celu(x, alpha=1.0)

    assert y.shape == x.shape, "Celu output shape should match input"
    assert y._data[1] == 0.0, "Celu(0) should be 0"
    assert y._data[2] == 1.0, "Celu(1) should be 1"

    # Test module
    celu_mod = nl.Celu(alpha=1.0)
    y_mod = celu_mod(x)
    assert xp.allclose(y._data, y_mod._data), "Celu function and module should match"

    # Test gradient
    x_grad = nm.tensor([1.0], requires_grad=True)
    y_grad = nl.celu(x_grad)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "Celu gradient should be computed"
    print("    ✅ Celu")

    # ========================================
    # Softplus
    # ========================================
    print("  Testing Softplus...")
    x = nm.tensor([-1.0, 0.0, 1.0])
    y = nl.softplus(x)

    assert y.shape == x.shape, "Softplus output shape should match input"
    assert xp.all(y._data > 0), "Softplus outputs should be positive"
    assert xp.allclose(y._data[1], xp.log(2.0), atol=1e-6), (
        "Softplus(0) should be ln(2)"
    )

    # Test module
    softplus_mod = nl.Softplus()
    y_mod = softplus_mod(x)
    assert xp.allclose(y._data, y_mod._data), (
        "Softplus function and module should match"
    )

    # Test gradient
    x_grad = nm.tensor([0.0], requires_grad=True)
    y_grad = nl.softplus(x_grad)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "Softplus gradient should be computed"
    # At x=0, gradient should be sigmoid(0) = 0.5
    assert xp.allclose(x_grad.grad._data, [0.5], atol=1e-6), (
        "Softplus gradient at 0 should be 0.5"
    )
    print("    ✅ Softplus")

    # ========================================
    # Softsign
    # ========================================
    print("  Testing Softsign...")
    x = nm.tensor([-1.0, 0.0, 1.0])
    y = nl.softsign(x)

    assert y.shape == x.shape, "Softsign output shape should match input"
    assert y._data[1] == 0.0, "Softsign(0) should be 0"
    assert xp.allclose(y._data[0], -0.5), "Softsign(-1) should be -0.5"
    assert xp.allclose(y._data[2], 0.5), "Softsign(1) should be 0.5"

    # Test module
    softsign_mod = nl.Softsign()
    y_mod = softsign_mod(x)
    assert xp.allclose(y._data, y_mod._data), (
        "Softsign function and module should match"
    )

    # Test gradient
    x_grad = nm.tensor([1.0], requires_grad=True)
    y_grad = nl.softsign(x_grad)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "Softsign gradient should be computed"
    print("    ✅ Softsign")

    # ========================================
    # HardSigmoid
    # ========================================
    print("  Testing HardSigmoid...")
    x = nm.tensor([-3.0, 0.0, 3.0])
    y = nl.hard_sigmoid(x, alpha=0.2, beta=0.5)

    assert y.shape == x.shape, "HardSigmoid output shape should match input"
    assert xp.allclose(y._data[0], 0.0), "HardSigmoid(-3) should be 0"
    assert xp.allclose(y._data[1], 0.5), "HardSigmoid(0) should be 0.5"
    assert xp.allclose(y._data[2], 1.0), "HardSigmoid(3) should be 1"

    # Test module
    hardsigmoid_mod = nl.HardSigmoid(alpha=0.2, beta=0.5)
    y_mod = hardsigmoid_mod(x)
    assert xp.allclose(y._data, y_mod._data), (
        "HardSigmoid function and module should match"
    )

    # Test gradient
    x_grad = nm.tensor([0.0], requires_grad=True)
    y_grad = nl.hard_sigmoid(x_grad)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "HardSigmoid gradient should be computed"
    print("    ✅ HardSigmoid")

    # ========================================
    # HardSwish
    # ========================================
    print("  Testing HardSwish...")
    x = nm.tensor([-3.0, 0.0, 3.0])
    y = nl.hard_swish(x)

    assert y.shape == x.shape, "HardSwish output shape should match input"
    assert xp.allclose(y._data[0], 0.0), "HardSwish(-3) should be 0"
    assert xp.allclose(y._data[2], 3.0), "HardSwish(3) should be 3"

    # Test module
    hardswish_mod = nl.HardSwish()
    y_mod = hardswish_mod(x)
    assert xp.allclose(y._data, y_mod._data), (
        "HardSwish function and module should match"
    )

    # Test gradient
    x_grad = nm.tensor([1.0], requires_grad=True)
    y_grad = nl.hard_swish(x_grad)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "HardSwish gradient should be computed"
    print("    ✅ HardSwish")

    # ========================================
    # Mish
    # ========================================
    print("  Testing Mish...")
    x = nm.tensor([-1.0, 0.0, 1.0])
    y = nl.mish(x)

    assert y.shape == x.shape, "Mish output shape should match input"
    assert xp.allclose(y._data[1], 0.0, atol=1e-6), "Mish(0) should be close to 0"

    # Test module
    mish_mod = nl.Mish()
    y_mod = mish_mod(x)
    assert xp.allclose(y._data, y_mod._data), "Mish function and module should match"

    # Test gradient
    x_grad = nm.tensor([1.0], requires_grad=True)
    y_grad = nl.mish(x_grad)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "Mish gradient should be computed"
    print("    ✅ Mish")

    # ========================================
    # ThresholdedRelu
    # ========================================
    print("  Testing ThresholdedRelu...")
    x = nm.tensor([0.5, 1.0, 1.5])
    y = nl.thresholded_relu(x, alpha=1.0)

    assert y.shape == x.shape, "ThresholdedRelu output shape should match input"
    assert xp.allclose(y._data[0], 0.0), "ThresholdedRelu(0.5, alpha=1) should be 0"
    assert xp.allclose(y._data[1], 0.0), "ThresholdedRelu(1.0, alpha=1) should be 0"
    assert xp.allclose(y._data[2], 1.5), "ThresholdedRelu(1.5, alpha=1) should be 1.5"

    # Test module
    thresholdedrelu_mod = nl.ThresholdedRelu(alpha=1.0)
    y_mod = thresholdedrelu_mod(x)
    assert xp.allclose(y._data, y_mod._data), (
        "ThresholdedRelu function and module should match"
    )

    # Test gradient
    x_grad = nm.tensor([1.5], requires_grad=True)
    y_grad = nl.thresholded_relu(x_grad, alpha=1.0)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "ThresholdedRelu gradient should be computed"
    print("    ✅ ThresholdedRelu")

    # ========================================
    # PRelu
    # ========================================
    print("  Testing PRelu...")

    # Test with single parameter
    prelu = nl.PRelu(num_parameters=1, init=0.25)
    x = nm.tensor([-1.0, 0.0, 1.0])
    y = prelu(x)

    assert y.shape == x.shape, "PRelu output shape should match input"
    assert xp.allclose(y._data[0], -0.25), "PRelu(-1) should be -0.25"
    assert y._data[1] == 0.0, "PRelu(0) should be 0"
    assert y._data[2] == 1.0, "PRelu(1) should be 1"

    # Test gradient
    nm.autograd.enable()
    prelu2 = nl.PRelu(num_parameters=1, init=0.25)
    x_grad = nm.tensor([-1.0, 1.0], requires_grad=True)
    y_grad = prelu2(x_grad)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "PRelu gradient should be computed"
    assert prelu2.slope.grad is not None, "PRelu slope gradient should be computed"

    # Test with multiple parameters
    prelu_multi = nl.PRelu(num_parameters=3, init=0.25)
    assert prelu_multi.slope.shape == (3,), "PRelu should have correct parameter shape"

    print("    ✅ PRelu")

    # ========================================
    # Test Elu (existing, but verify ONNX compliance)
    # ========================================
    print("  Testing Elu (ONNX-compliant)...")
    elu_mod = nl.Elu(alpha=1.0)
    x = nm.tensor([-1.0, 0.0, 1.0])
    y = elu_mod(x)

    assert y.shape == x.shape, "Elu output shape should match input"
    assert "Elu" in repr(elu_mod), "Elu repr should be correct"

    # Test gradient
    x_grad = nm.tensor([1.0], requires_grad=True)
    y_grad = nl.elu(x_grad)
    loss = nm.sum(y_grad)
    loss.backward()
    assert x_grad.grad is not None, "Elu gradient should be computed"
    print("    ✅ Elu")

    print("✅ All new activation function tests passed!\n")


def test_activation_onnx_compliance():
    """Test that activation functions work correctly for ONNX export"""

    print("Testing activation functions for ONNX compliance...")

    # Test all ONNX-compliant activation modules in a Sequential model
    try:
        model = nl.Sequential(
            nl.Linear(10, 20),
            nl.Relu(),
            nl.Linear(20, 15),
            nl.Gelu(),
            nl.Linear(15, 10),
            nl.Selu(),
            nl.Linear(10, 5),
            nl.Sigmoid(),
        )

        x = nm.randn(2, 10)
        y = model(x)

        assert y.shape == (2, 5), "Model output shape should be correct"
        print("  ✅ Sequential model with new activations")

        # Test gradient flow through all layers
        nm.autograd.enable()
        x_grad = nm.randn(2, 10, requires_grad=True)
        y_grad = model(x_grad)
        loss = nm.sum(y_grad)
        loss.backward()

        assert x_grad.grad is not None, "Gradient should flow through all activations"
        for param in model.parameters():
            assert param.grad is not None, "All parameters should have gradients"

        print("  ✅ Gradient flow through new activations")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise

    print("✅ All ONNX compliance tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running activation function tests")
    print("=" * 60)
    print()

    try:
        test_activation_functions()
        test_activation_modules()
        test_activation_edge_cases()
        test_activation_gradient_flow()
        test_new_activation_functions()
        test_activation_onnx_compliance()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"UNEXPECTED ERROR: {e}")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)
