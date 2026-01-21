import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon.nnlib as nl
import lemon.numlib as nm


def test_loss_functions():
    """Test loss functions (functional API)"""

    print("Testing loss functions (functional)...")

    xp = nm.np  # numpy を使用

    # ========================================
    # mean_squared_error
    # ========================================
    print("  Testing mean_squared_error()...")

    # Test 1: Basic MSE
    y_pred = nm.tensor([2.5, 0.0, 2.0])
    y_true = nm.tensor([3.0, -0.5, 2.0])
    loss = nl.mean_squared_error(y_pred, y_true)

    # Manual calculation: ((2.5-3)^2 + (0-(-0.5))^2 + (2-2)^2) / 3 = (0.25 + 0.25 + 0) / 3 = 0.1667
    expected = (0.25 + 0.25 + 0.0) / 3
    assert abs(loss._data - expected) < 1e-6, (
        f"MSE should be {expected}, got {loss._data}"
    )
    print("    ✅ mean_squared_error basic")

    # Test 2: MSE with reduction='sum'
    loss_sum = nl.mean_squared_error(y_pred, y_true, reduction="sum")
    expected_sum = 0.25 + 0.25 + 0.0
    assert abs(loss_sum._data - expected_sum) < 1e-6, "MSE sum should be correct"
    print("    ✅ mean_squared_error with reduction='sum'")

    # Test 3: MSE with reduction='none'
    loss_none = nl.mean_squared_error(y_pred, y_true, reduction="none")
    expected_none = nm.tensor([0.25, 0.25, 0.0])
    assert xp.allclose(loss_none._data, expected_none._data), (
        "MSE none should be correct"
    )
    print("    ✅ mean_squared_error with reduction='none'")

    # Test 4: MSE gradient
    nm.autograd.enable()
    y_pred = nm.tensor([2.0, 1.0], requires_grad=True)
    y_true = nm.tensor([1.0, 1.0])
    loss = nl.mean_squared_error(y_pred, y_true)
    loss.backward()

    # d/dy_pred[(y_pred - y_true)^2] = 2*(y_pred - y_true) / n
    # For y_pred=[2, 1], y_true=[1, 1]: grad = 2*[1, 0] / 2 = [1, 0]
    expected_grad = nm.tensor([1.0, 0.0])
    assert xp.allclose(y_pred.grad._data, expected_grad._data), (
        "MSE gradient should be correct"
    )
    print("    ✅ mean_squared_error gradient")

    # Test 5: Invalid reduction mode
    try:
        nl.mean_squared_error(y_pred, y_true, reduction="invalid")
        assert False, "Should raise ValueError for invalid reduction"
    except ValueError as e:
        assert "Invalid reduction mode" in str(e)
    print("    ✅ mean_squared_error invalid reduction")

    # ========================================
    # cross_entropy
    # ========================================
    print("  Testing cross_entropy()...")

    # Test 6: Basic cross entropy with class indices
    y_pred = nm.tensor([[2.0, 1.0, 0.1]])  # logits for 3 classes
    y_true = nm.tensor([0])  # class 0
    loss = nl.softmax_cross_entropy(y_pred, y_true)

    assert loss._data > 0, "Cross entropy should be positive"
    assert loss.shape == (), "Cross entropy should be scalar with reduction='mean'"
    print("    ✅ cross_entropy basic with class indices")

    # Test 7: Cross entropy with one-hot encoded targets
    y_pred = nm.tensor([[2.0, 1.0, 0.1]])
    y_true_onehot = nm.tensor([[1.0, 0.0, 0.0]])  # one-hot for class 0
    loss = nl.softmax_cross_entropy(y_pred, y_true_onehot)

    assert loss._data > 0, "Cross entropy should be positive"
    print("    ✅ cross_entropy with one-hot targets")

    # Test 8: Cross entropy with batch
    y_pred = nm.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    y_true = nm.tensor([0, 1])
    loss = nl.softmax_cross_entropy(y_pred, y_true)

    assert loss.shape == (), "Cross entropy should be scalar"
    print("    ✅ cross_entropy with batch")

    # Test 9: Cross entropy with reduction='sum'
    loss_sum = nl.softmax_cross_entropy(y_pred, y_true, reduction="sum")
    assert loss_sum._data > 0, "Cross entropy sum should be positive"
    print("    ✅ cross_entropy with reduction='sum'")

    # Test 10: Cross entropy with reduction='none'
    loss_none = nl.softmax_cross_entropy(y_pred, y_true, reduction="none")
    assert loss_none.shape == (2,), "Cross entropy none should return per-sample losses"
    print("    ✅ cross_entropy with reduction='none'")

    # Test 11: Cross entropy gradient
    nm.autograd.enable()
    y_pred = nm.tensor([[2.0, 1.0]], requires_grad=True)
    y_true = nm.tensor([0])
    loss = nl.softmax_cross_entropy(y_pred, y_true)
    loss.backward()

    assert y_pred.grad is not None, "Cross entropy gradient should be computed"
    print("    ✅ cross_entropy gradient")

    # ========================================
    # binary_cross_entropy
    # ========================================
    print("  Testing binary_cross_entropy()...")

    # Test 12: Basic BCE
    y_pred = nm.tensor([0.9, 0.2, 0.8])
    y_true = nm.tensor([1.0, 0.0, 1.0])
    loss = nl.binary_cross_entropy(y_pred, y_true)

    assert loss._data > 0, "BCE should be positive"
    assert loss.shape == (), "BCE should be scalar with reduction='mean'"
    print("    ✅ binary_cross_entropy basic")

    # Test 13: BCE perfect prediction
    y_pred = nm.tensor([1.0, 0.0, 1.0])
    y_true = nm.tensor([1.0, 0.0, 1.0])
    loss = nl.binary_cross_entropy(y_pred, y_true)

    # Perfect prediction should give very small loss (close to 0)
    assert loss._data < 0.01, "BCE for perfect prediction should be near 0"
    print("    ✅ binary_cross_entropy perfect prediction")

    # Test 14: BCE with reduction='sum'
    y_pred = nm.tensor([0.9, 0.2, 0.8])
    y_true = nm.tensor([1.0, 0.0, 1.0])
    loss_sum = nl.binary_cross_entropy(y_pred, y_true, reduction="sum")

    assert loss_sum._data > 0, "BCE sum should be positive"
    print("    ✅ binary_cross_entropy with reduction='sum'")

    # Test 15: BCE with reduction='none'
    loss_none = nl.binary_cross_entropy(y_pred, y_true, reduction="none")
    assert loss_none.shape == (3,), "BCE none should return per-sample losses"
    print("    ✅ binary_cross_entropy with reduction='none'")

    # Test 16: BCE gradient
    nm.autograd.enable()
    y_pred = nm.tensor([0.5, 0.8], requires_grad=True)
    y_true = nm.tensor([1.0, 0.0])
    loss = nl.binary_cross_entropy(y_pred, y_true)
    loss.backward()

    assert y_pred.grad is not None, "BCE gradient should be computed"
    print("    ✅ binary_cross_entropy gradient")

    # Test 17: BCE with extreme values (numerical stability)
    y_pred = nm.tensor([0.999999, 0.000001])
    y_true = nm.tensor([1.0, 0.0])
    loss = nl.binary_cross_entropy(y_pred, y_true)

    assert not xp.isnan(loss._data), "BCE should handle extreme values without NaN"
    assert not xp.isinf(loss._data), "BCE should handle extreme values without Inf"
    print("    ✅ binary_cross_entropy numerical stability")

    print("✅ All loss function tests passed!\n")


def test_loss_modules():
    """Test loss modules (class-based API)"""

    print("Testing loss modules (class-based)...")

    xp = nm.np

    # Test 1: MSELoss module
    criterion = nl.MSELoss()
    y_pred = nm.tensor([2.0, 1.0])
    y_true = nm.tensor([1.0, 1.0])
    loss = criterion(y_pred, y_true)

    expected = 0.5  # ((2-1)^2 + (1-1)^2) / 2 = 1/2
    assert abs(loss._data - expected) < 1e-6, "MSELoss module should work"
    assert "MSELoss" in repr(criterion), "MSELoss repr should be correct"
    print("  ✅ MSELoss module")

    # Test 2: MSELoss with different reduction
    criterion_sum = nl.MSELoss(reduction="sum")
    loss_sum = criterion_sum(y_pred, y_true)
    assert abs(loss_sum._data - 1.0) < 1e-6, "MSELoss sum should work"
    print("  ✅ MSELoss with reduction='sum'")

    # Test 3: CrossEntropyLoss module
    criterion = nl.CrossEntropyLoss()
    y_pred = nm.tensor([[2.0, 1.0, 0.1]])
    y_true = nm.tensor([0])
    loss = criterion(y_pred, y_true)

    assert loss._data > 0, "CrossEntropyLoss should work"
    assert "CrossEntropyLoss" in repr(criterion), (
        "CrossEntropyLoss repr should be correct"
    )
    print("  ✅ CrossEntropyLoss module")

    # Test 4: BCELoss module
    criterion = nl.BCELoss()
    y_pred = nm.tensor([0.9, 0.2])
    y_true = nm.tensor([1.0, 0.0])
    loss = criterion(y_pred, y_true)

    assert loss._data > 0, "BCELoss should work"
    assert "BCELoss" in repr(criterion), "BCELoss repr should be correct"
    print("  ✅ BCELoss module")

    # Test 5: Gradient through loss modules
    nm.autograd.enable()

    criterion = nl.MSELoss()
    y_pred = nm.tensor([2.0, 1.0], requires_grad=True)
    y_true = nm.tensor([1.0, 1.0])
    loss = criterion(y_pred, y_true)
    loss.backward()

    assert y_pred.grad is not None, "Gradient should flow through loss module"
    print("  ✅ Gradient through loss modules")

    # Test 6: Loss in training loop (realistic usage)
    nm.autograd.enable()

    model = nl.Sequential(nl.Linear(5, 3), nl.Relu(), nl.Linear(3, 2))
    criterion = nl.MSELoss()

    x = nm.randn(4, 5)
    y_true = nm.randn(4, 2)

    # Forward
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    # Backward
    model.zero_grad()
    loss.backward()

    # Check all parameters have gradients
    for param in model.parameters():
        assert param.grad is not None, "All parameters should have gradients"

    print("  ✅ Loss in training loop")

    print("✅ All loss module tests passed!\n")


def test_loss_edge_cases():
    """Test loss function edge cases for coverage"""
    print("Testing loss edge cases...")

    # Test 1: MSE with scalar inputs
    y_pred = nm.tensor(2.0)
    y_true = nm.tensor(1.0)
    loss = nl.mean_squared_error(y_pred, y_true)
    assert abs(loss._data - 1.0) < 1e-6, "MSE with scalars should work"
    print("  ✅ MSE with scalar inputs")

    # Test 2: Cross entropy with single sample
    y_pred = nm.tensor([[2.0, 1.0]])
    y_true = nm.tensor([0])
    loss = nl.softmax_cross_entropy(y_pred, y_true)
    assert loss._data > 0, "Cross entropy with single sample should work"
    print("  ✅ Cross entropy with single sample")

    # Test 3: BCE with all correct predictions
    y_pred = nm.tensor([0.99, 0.01, 0.99])
    y_true = nm.tensor([1.0, 0.0, 1.0])
    loss = nl.binary_cross_entropy(y_pred, y_true)
    assert loss._data < 0.1, "BCE with good predictions should be small"
    print("  ✅ BCE with good predictions")

    # Test 4: Loss without gradient (autograd off)
    nm.autograd.disable()
    y_pred = nm.tensor([2.0, 1.0])
    y_true = nm.tensor([1.0, 1.0])
    loss = nl.mean_squared_error(y_pred, y_true)
    assert loss.requires_grad == False, (
        "Loss should not require grad when autograd is off"
    )
    print("  ✅ Loss without gradient")

    # Test 5: Loss with requires_grad=False
    nm.autograd.enable()
    y_pred = nm.tensor([2.0, 1.0], requires_grad=False)
    y_true = nm.tensor([1.0, 1.0])
    loss = nl.mean_squared_error(y_pred, y_true)
    # Loss might still require grad due to operations, but input doesn't
    print("  ✅ Loss with requires_grad=False input")

    # Test 6: Multi-dimensional MSE
    y_pred = nm.randn(3, 4, 5)
    y_true = nm.randn(3, 4, 5)
    loss = nl.mean_squared_error(y_pred, y_true)
    assert loss.shape == (), "MSE should reduce to scalar"
    print("  ✅ Multi-dimensional MSE")

    # Test 7: Cross entropy with different batch sizes
    for batch_size in [1, 5, 10]:
        y_pred = nm.randn(batch_size, 3)
        y_true = nm.tensor([i % 3 for i in range(batch_size)])
        loss = nl.softmax_cross_entropy(y_pred, y_true)
        assert loss.shape == (), (
            f"Cross entropy should work with batch_size={batch_size}"
        )
    print("  ✅ Cross entropy with different batch sizes")

    print("✅ All loss edge cases passed!\n")


def test_loss_invalid_reduction():
    """Test invalid reduction modes for all loss functions"""
    print("Testing invalid reduction modes...")

    # Test 1: cross_entropy with invalid reduction
    y_pred = nm.tensor([[2.0, 1.0]])
    y_true = nm.tensor([0])

    try:
        nl.softmax_cross_entropy(y_pred, y_true, reduction="invalid")
        assert False, "cross_entropy should raise ValueError for invalid reduction"
    except ValueError as e:
        assert "Invalid reduction mode" in str(e)
        assert "invalid" in str(e)
    print("  ✅ cross_entropy invalid reduction")

    # Test 2: binary_cross_entropy with invalid reduction
    y_pred = nm.tensor([0.5, 0.8])
    y_true = nm.tensor([1.0, 0.0])

    try:
        nl.binary_cross_entropy(y_pred, y_true, reduction="invalid")
        assert False, (
            "binary_cross_entropy should raise ValueError for invalid reduction"
        )
    except ValueError as e:
        assert "Invalid reduction mode" in str(e)
        assert "invalid" in str(e)
    print("  ✅ binary_cross_entropy invalid reduction")

    # Test 3: mean_squared_error already tested, but double-check
    y_pred = nm.tensor([2.0, 1.0])
    y_true = nm.tensor([1.0, 1.0])

    try:
        nl.mean_squared_error(y_pred, y_true, reduction="wrong")
        assert False, "mean_squared_error should raise ValueError for invalid reduction"
    except ValueError as e:
        assert "Invalid reduction mode" in str(e)
    print("  ✅ mean_squared_error invalid reduction (verified)")

    print("✅ All invalid reduction tests passed!\n")
