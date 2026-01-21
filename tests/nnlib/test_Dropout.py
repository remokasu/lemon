import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon.nnlib as nl
import lemon.numlib as nm


def test_dropout_function():
    """Test dropout function (functional API)"""

    print("Testing dropout function (functional)...")

    xp = nm.np

    # Test 1: Dropout in training mode
    x = nm.ones(1000)
    y = nl.dropout(x, p=0.5, training=True)

    # About 50% should be zero
    zero_ratio = xp.sum(y._data == 0) / y.size
    assert 0.4 < zero_ratio < 0.6, f"Zero ratio should be around 0.5, got {zero_ratio}"

    # Non-zero elements should be scaled by 1/(1-p) = 2
    non_zero_values = y._data[y._data != 0]
    if len(non_zero_values) > 0:
        assert xp.allclose(non_zero_values, 2.0), (
            "Non-zero values should be scaled by 2"
        )

    print("  ✅ dropout in training mode")

    # Test 2: Dropout in evaluation mode (should be identity)
    x = nm.ones(100)
    y = nl.dropout(x, p=0.5, training=False)

    assert xp.allclose(y._data, x._data), "Dropout in eval mode should be identity"
    print("  ✅ dropout in evaluation mode")

    # Test 3: Dropout with p=0 (no dropout)
    x = nm.ones(100)
    y = nl.dropout(x, p=0.0, training=True)

    assert xp.allclose(y._data, x._data), "Dropout with p=0 should be identity"
    print("  ✅ dropout with p=0")

    # Test 4: Dropout gradient (修正)
    nm.autograd.enable()
    x = nm.ones(100)
    x.requires_grad = True  # 修正: プロパティとして設定
    y = nl.dropout(x, p=0.5, training=True)
    loss = nm.sum(y)
    loss.backward()

    assert x.grad is not None, "Dropout gradient should be computed"
    # Gradient should also be masked and scaled
    print("  ✅ dropout gradient")

    # Test 5: Dropout with different probabilities
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        x = nm.ones(10000)
        y = nl.dropout(x, p=p, training=True)
        zero_ratio = xp.sum(y._data == 0) / y.size

        # Allow some tolerance
        assert abs(zero_ratio - p) < 0.05, (
            f"Zero ratio should be around {p}, got {zero_ratio}"
        )

    print("  ✅ dropout with different probabilities")

    # Test 6: Invalid probability
    x = nm.ones(10)
    try:
        nl.dropout(x, p=1.5, training=True)
        assert False, "Should raise ValueError for p >= 1"
    except ValueError as e:
        assert "must be in [0, 1)" in str(e)

    try:
        nl.dropout(x, p=-0.1, training=True)
        assert False, "Should raise ValueError for p < 0"
    except ValueError as e:
        assert "must be in [0, 1)" in str(e)

    print("  ✅ dropout invalid probability")

    # Test 7: Dropout preserves shape
    shapes = [(10,), (5, 10), (3, 4, 5), (2, 3, 4, 5)]
    for shape in shapes:
        x = nm.randn(*shape)
        y = nl.dropout(x, p=0.5, training=True)
        assert y.shape == shape, f"Dropout should preserve shape {shape}"

    print("  ✅ dropout preserves shape")

    print("✅ All dropout function tests passed!\n")


def test_dropout_module():
    """Test Dropout module (class-based API)"""
    print("Testing Dropout module (class-based)...")

    xp = nm.np

    # Test 1: Dropout module basic (修正)
    dropout_layer = nl.Dropout(p=0.5)
    x = nm.ones(1000)

    # Training mode (修正: 関数として呼ぶ)
    nl.train.on()
    y = dropout_layer(x)
    zero_ratio = xp.sum(y._data == 0) / y.size
    assert 0.4 < zero_ratio < 0.6, "Dropout should be active in training mode"

    # Evaluation mode
    nl.train.off()
    y = dropout_layer(x)
    assert xp.allclose(y._data, x._data), "Dropout should be inactive in eval mode"

    print("  ✅ Dropout module with train mode")

    # Test 2: Dropout repr
    dropout_layer = nl.Dropout(p=0.3)
    assert "Dropout" in repr(dropout_layer), "Dropout repr should contain 'Dropout'"
    assert "0.3" in repr(dropout_layer), "Dropout repr should contain probability"
    print("  ✅ Dropout repr")

    # Test 3: Dropout in Sequential (修正)
    model = nl.Sequential(
        nl.Linear(10, 20), nl.Relu(), nl.Dropout(0.5), nl.Linear(20, 5)
    )

    x = nm.randn(3, 10)

    # Training mode
    nl.train.on()
    y1 = model(x)
    y2 = model(x)
    # Outputs should be different due to random dropout
    assert not xp.allclose(y1._data, y2._data), (
        "Dropout should produce different outputs"
    )

    # Evaluation mode
    nl.train.off()
    y1 = model(x)
    y2 = model(x)
    # Outputs should be identical (deterministic)
    assert xp.allclose(y1._data, y2._data), "Eval mode should be deterministic"

    print("  ✅ Dropout in Sequential")

    # Test 4: Gradient through Dropout (修正)
    nm.autograd.enable()

    dropout_layer = nl.Dropout(0.5)
    x = nm.randn(10)
    x.requires_grad = True  # 修正

    nl.train.on()
    y = dropout_layer(x)
    loss = nm.sum(y)
    loss.backward()

    assert x.grad is not None, "Gradient should flow through Dropout"
    print("  ✅ Gradient through Dropout")

    # Test 5: Dropout in complete training loop
    nm.autograd.enable()

    model = nl.Sequential(
        nl.Linear(5, 10), nl.Relu(), nl.Dropout(0.5), nl.Linear(10, 2)
    )

    x = nm.randn(4, 5)
    y_true = nm.randn(4, 2)

    nl.train.on()
    # Forward
    y_pred = model(x)
    loss = nl.mean_squared_error(y_pred, y_true)

    # Backward
    model.zero_grad()
    loss.backward()

    # Check gradients
    for param in model.parameters():
        assert param.grad is not None, "All parameters should have gradients"

    print("  ✅ Dropout in complete training loop")

    # Test 6: Invalid probability in constructor
    try:
        nl.Dropout(p=1.5)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "must be in [0, 1)" in str(e)

    print("  ✅ Dropout invalid probability in constructor")

    # Test 7: Default training mode
    dropout_layer = nl.Dropout(0.5)
    x = nm.ones(100)

    # Default should follow global train state (which is ON by default)
    nl.train.on()  # 明示的に設定
    y = dropout_layer(x)
    # Should apply dropout (training mode is on)
    assert not xp.allclose(y._data, x._data), "Default should be training mode"

    print("  ✅ Dropout default training mode")

    print("✅ All Dropout module tests passed!\n")


def test_dropout_edge_cases():
    """Test Dropout edge cases for coverage"""
    print("Testing Dropout edge cases...")

    xp = nm.np

    # Test 1: Dropout with autograd disabled
    nm.autograd.disable()
    x = nm.randn(10)
    y = nl.dropout(x, p=0.5, training=True)
    assert y.requires_grad == False, (
        "Dropout should not require grad when autograd is off"
    )
    print("  ✅ Dropout with autograd disabled")

    # Test 2: Dropout with requires_grad=False (修正)
    nm.autograd.enable()
    x = nm.randn(10)
    x.requires_grad = False  # 修正: プロパティとして設定
    y = nl.dropout(x, p=0.5, training=True)
    assert y.requires_grad == False, (
        "Dropout should not require grad when input doesn't"
    )
    print("  ✅ Dropout with requires_grad=False")

    # Test 3: Backward with None gradient (修正)
    nm.autograd.enable()
    x = nm.randn(10)
    x.requires_grad = True  # 修正
    y = nl.dropout(x, p=0.5, training=True)

    y.grad = None
    y._backward()  # Should handle None gracefully
    print("  ✅ Dropout backward with None gradient")

    # Test 4: Multiple backward passes (修正)
    nm.autograd.enable()
    x = nm.randn(10)
    x.requires_grad = True  # 修正

    # First backward
    y1 = nl.dropout(x, p=0.5, training=True)
    loss1 = nm.sum(y1)
    loss1.backward()
    grad1 = x.grad._data.copy()

    # Second backward (should accumulate)
    y2 = nl.dropout(x, p=0.5, training=True)
    loss2 = nm.sum(y2)
    loss2.backward()
    grad2 = x.grad._data.copy()

    assert not xp.allclose(grad1, grad2), "Gradients should accumulate"
    print("  ✅ Dropout gradient accumulation")

    # Test 5: Dropout with very small probability
    x = nm.ones(10000)
    y = nl.dropout(x, p=0.01, training=True)
    zero_ratio = xp.sum(y._data == 0) / y.size
    assert abs(zero_ratio - 0.01) < 0.01, "Very small p should work"
    print("  ✅ Dropout with very small p")

    # Test 6: Dropout with very large probability
    x = nm.ones(10000)
    y = nl.dropout(x, p=0.99, training=True)
    zero_ratio = xp.sum(y._data == 0) / y.size
    assert abs(zero_ratio - 0.99) < 0.01, "Very large p should work"
    print("  ✅ Dropout with very large p")

    # Test 7: Functional vs Module equivalence (修正)
    nm.autograd.enable()
    x = nm.randn(100)
    x.requires_grad = True  # 修正

    # Set seed for reproducibility
    nm.seed(42)
    y_func = nl.dropout(x, p=0.5, training=True)
    loss_func = nm.sum(y_func)
    loss_func.backward()
    grad_func = x.grad._data.copy()

    # Reset
    x.zero_grad()
    nm.seed(42)
    dropout_mod = nl.Dropout(0.5)
    nl.train.on()
    y_mod = dropout_mod(x)
    loss_mod = nm.sum(y_mod)
    loss_mod.backward()
    grad_mod = x.grad._data.copy()

    # Should be identical with same seed
    assert xp.allclose(y_func._data, y_mod._data), (
        "Functional and module should match with same seed"
    )
    assert xp.allclose(grad_func, grad_mod), "Gradients should match with same seed"
    print("  ✅ Functional vs Module equivalence")

    print("✅ All Dropout edge cases passed!\n")
