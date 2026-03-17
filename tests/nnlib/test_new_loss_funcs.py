import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon.nnlib as nl
import lemon.numlib as nm


def test_huber_loss():
    """Test HuberLoss"""
    print("Testing HuberLoss...")

    xp = nm.np

    # Test 1: small error (MSE region) -> 0.5 * error^2
    y_pred = nm.tensor([1.5])
    y_true = nm.tensor([1.0])
    loss = nl.huber_loss(y_pred, y_true, delta=1.0)
    expected = 0.5 * (0.5**2)  # 0.125
    assert abs(loss._data - expected) < 1e-6, f"Huber small error should be {expected}"
    print("  ✅ small error (MSE region)")

    # Test 2: large error (MAE region) -> delta * (|error| - 0.5 * delta)
    y_pred = nm.tensor([3.0])
    y_true = nm.tensor([0.0])
    loss = nl.huber_loss(y_pred, y_true, delta=1.0)
    expected = 1.0 * (3.0 - 0.5)  # 2.5
    assert abs(loss._data - expected) < 1e-6, f"Huber large error should be {expected}"
    print("  ✅ large error (MAE region)")

    # Test 3: error exactly at delta -> MSE region (<=)
    y_pred = nm.tensor([1.0])
    y_true = nm.tensor([0.0])
    loss = nl.huber_loss(y_pred, y_true, delta=1.0)
    expected = 0.5 * (1.0**2)  # 0.5
    assert abs(loss._data - expected) < 1e-6, "Huber at delta boundary should be MSE"
    print("  ✅ boundary case (error == delta)")

    # Test 4: reduction='sum'
    y_pred = nm.tensor([1.5, 3.0])
    y_true = nm.tensor([1.0, 0.0])
    loss_sum = nl.huber_loss(y_pred, y_true, delta=1.0, reduction="sum")
    expected_sum = 0.125 + 2.5  # 2.625
    assert abs(loss_sum._data - expected_sum) < 1e-6, "Huber sum should be correct"
    print("  ✅ reduction='sum'")

    # Test 5: reduction='none'
    loss_none = nl.huber_loss(y_pred, y_true, delta=1.0, reduction="none")
    assert loss_none.shape == (2,), "Huber none should return per-sample losses"
    print("  ✅ reduction='none'")

    # Test 6: invalid reduction
    try:
        nl.huber_loss(y_pred, y_true, reduction="invalid")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Invalid reduction mode" in str(e)
    print("  ✅ invalid reduction")

    # Test 7: gradient
    nm.autograd.enable()
    y_pred = nm.tensor([1.5, 3.0], requires_grad=True)
    y_true = nm.tensor([1.0, 0.0])
    loss = nl.huber_loss(y_pred, y_true, delta=1.0)
    loss.backward()
    assert y_pred.grad is not None, "Huber gradient should be computed"
    print("  ✅ gradient")

    # Test 8: HuberLoss module
    criterion = nl.HuberLoss(delta=1.0)
    y_pred = nm.tensor([1.5, 3.0])
    y_true = nm.tensor([1.0, 0.0])
    loss = criterion(y_pred, y_true)
    assert loss._data > 0, "HuberLoss module should work"
    assert "HuberLoss" in repr(criterion), "HuberLoss repr should be correct"
    print("  ✅ HuberLoss module")

    # Test 9: training loop
    nm.autograd.enable()
    model = nl.Sequential(nl.Linear(5, 1))
    criterion = nl.HuberLoss()
    x = nm.randn(4, 5)
    y_true = nm.randn(4, 1)
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None, "All parameters should have gradients"
    print("  ✅ training loop")

    print("✅ All HuberLoss tests passed!\n")


def test_focal_loss():
    """Test FocalLoss"""
    print("Testing FocalLoss...")

    xp = nm.np

    # Test 1: easy correct prediction -> small loss (down-weighted)
    y_pred_easy = nm.tensor([0.99])
    y_true_pos = nm.tensor([1.0])
    loss_easy = nl.focal_loss(y_pred_easy, y_true_pos, gamma=2.0, alpha=0.25)

    y_pred_hard = nm.tensor([0.5])
    loss_hard = nl.focal_loss(y_pred_hard, y_true_pos, gamma=2.0, alpha=0.25)

    assert loss_easy._data < loss_hard._data, (
        "Easy examples should have smaller focal loss"
    )
    print("  ✅ easy examples down-weighted vs hard examples")

    # Test 2: gamma=0 -> reduces to weighted BCE
    y_pred = nm.tensor([0.8])
    y_true = nm.tensor([1.0])
    loss_focal = nl.focal_loss(y_pred, y_true, gamma=0.0, alpha=1.0)
    loss_bce = nl.binary_cross_entropy(y_pred, y_true)
    # With gamma=0, alpha=1: focal = -log(p_t) = BCE
    assert abs(loss_focal._data - loss_bce._data) < 1e-5, (
        "gamma=0, alpha=1 should equal BCE"
    )
    print("  ✅ gamma=0 reduces to BCE")

    # Test 3: loss is positive
    y_pred = nm.tensor([0.3, 0.7, 0.9])
    y_true = nm.tensor([1.0, 0.0, 1.0])
    loss = nl.focal_loss(y_pred, y_true)
    assert loss._data > 0, "FocalLoss should be positive"
    print("  ✅ loss is positive")

    # Test 4: reduction modes
    loss_mean = nl.focal_loss(y_pred, y_true, reduction="mean")
    loss_sum = nl.focal_loss(y_pred, y_true, reduction="sum")
    loss_none = nl.focal_loss(y_pred, y_true, reduction="none")
    assert loss_none.shape == (3,), "none should return per-sample losses"
    assert abs(loss_sum._data - loss_none._data.sum()) < 1e-6, "sum should match"
    print("  ✅ reduction modes")

    # Test 5: invalid reduction
    try:
        nl.focal_loss(y_pred, y_true, reduction="invalid")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Invalid reduction mode" in str(e)
    print("  ✅ invalid reduction")

    # Test 6: gradient
    nm.autograd.enable()
    y_pred = nm.tensor([0.3, 0.7], requires_grad=True)
    y_true = nm.tensor([1.0, 0.0])
    loss = nl.focal_loss(y_pred, y_true)
    loss.backward()
    assert y_pred.grad is not None, "FocalLoss gradient should be computed"
    print("  ✅ gradient")

    # Test 7: FocalLoss module
    criterion = nl.FocalLoss(gamma=2.0, alpha=0.25)
    loss = criterion(y_pred, y_true)
    assert loss._data > 0, "FocalLoss module should work"
    assert "FocalLoss" in repr(criterion), "FocalLoss repr should be correct"
    print("  ✅ FocalLoss module")

    print("✅ All FocalLoss tests passed!\n")


def test_kl_div_loss():
    """Test KLDivLoss"""
    print("Testing KLDivLoss...")

    xp = nm.np

    # Test 1: identical distributions -> loss near 0
    probs = nm.tensor([[0.25, 0.25, 0.25, 0.25]])
    log_probs = nm.log(probs)
    loss = nl.kl_div_loss(log_probs, probs)
    assert abs(loss._data) < 1e-6, "KL(p||p) should be ~0"
    print("  ✅ identical distributions -> ~0")

    # Test 2: loss is non-negative
    p = nm.tensor([[0.7, 0.3]])
    q = nm.tensor([[0.4, 0.6]])
    log_q = nm.log(q)
    loss = nl.kl_div_loss(log_q, p)
    assert loss._data >= 0, "KL divergence should be non-negative"
    print("  ✅ non-negative")

    # Test 3: reduction modes
    p = nm.tensor([[0.7, 0.3], [0.4, 0.6]])
    log_q = nm.log(nm.tensor([[0.5, 0.5], [0.5, 0.5]]))
    loss_mean = nl.kl_div_loss(log_q, p, reduction="mean")
    loss_sum = nl.kl_div_loss(log_q, p, reduction="sum")
    loss_batchmean = nl.kl_div_loss(log_q, p, reduction="batchmean")
    loss_none = nl.kl_div_loss(log_q, p, reduction="none")
    assert loss_none.shape == (2, 2), "none should return per-element losses"
    assert abs(loss_batchmean._data - loss_sum._data / 2) < 1e-6, (
        "batchmean = sum / batch_size"
    )
    print("  ✅ reduction modes")

    # Test 4: invalid reduction
    try:
        nl.kl_div_loss(log_q, p, reduction="invalid")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Invalid reduction mode" in str(e)
    print("  ✅ invalid reduction")

    # Test 5: gradient
    nm.autograd.enable()
    q = nm.tensor([[0.5, 0.5]], requires_grad=True)
    log_q = nm.log(q)
    p = nm.tensor([[0.7, 0.3]])
    loss = nl.kl_div_loss(log_q, p)
    loss.backward()
    assert q.grad is not None, "KLDivLoss gradient should be computed"
    print("  ✅ gradient")

    # Test 6: KLDivLoss module
    criterion = nl.KLDivLoss()
    q = nm.tensor([[0.5, 0.5]])
    log_q = nm.log(q)
    p = nm.tensor([[0.7, 0.3]])
    loss = criterion(log_q, p)
    assert loss._data >= 0, "KLDivLoss module should work"
    assert "KLDivLoss" in repr(criterion), "KLDivLoss repr should be correct"
    print("  ✅ KLDivLoss module")

    print("✅ All KLDivLoss tests passed!\n")


def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING NEW LOSS FUNCTION TESTS")
    print("=" * 70 + "\n")

    try:
        test_huber_loss()
        test_focal_loss()
        test_kl_div_loss()

        print("=" * 70)
        print("✅ ALL NEW LOSS TESTS PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
