import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon.nnlib as nl
import lemon.numlib as nm


def test_step_scheduler():
    """Test StepScheduler"""
    print("Testing StepScheduler...")

    nm.autograd.enable()

    # Test 1: Basic StepScheduler for lr
    model = nl.Sequential(nl.Linear(10, 5))
    optimizer = nl.Adam(model.parameters(), lr=0.1)
    scheduler = nl.StepScheduler(optimizer, param_name="lr", step_size=3, gamma=0.5)

    assert optimizer.lr == 0.1, "Initial lr should be 0.1"

    # Epoch 1-3: lr should stay at 0.1 (last_epoch 0, 1, 2 -> all // 3 = 0)
    for i in range(3):
        scheduler.step()
        assert abs(optimizer.lr - 0.1) < 1e-6, f"Epoch {i + 1}: lr should still be 0.1"

    # Epoch 4: lr should drop to 0.05 (last_epoch 3 -> 3 // 3 = 1)
    scheduler.step()
    assert abs(optimizer.lr - 0.05) < 1e-6, "Epoch 4: lr should be 0.05"

    # Epoch 5-6: lr should stay at 0.05
    for i in range(2):
        scheduler.step()
        assert abs(optimizer.lr - 0.05) < 1e-6, (
            f"Epoch {i + 5}: lr should still be 0.05"
        )

    # Epoch 7: lr should drop to 0.025 (last_epoch 6 -> 6 // 3 = 2)
    scheduler.step()
    assert abs(optimizer.lr - 0.025) < 1e-6, "Epoch 7: lr should be 0.025"

    print("  ✅ Basic StepScheduler")

    # Test 2: StepLR convenience class
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.StepLR(optimizer, step_size=2, gamma=0.1)

    assert optimizer.lr == 0.01, "Initial lr should be 0.01"
    scheduler.step()
    assert abs(optimizer.lr - 0.01) < 1e-6, "Epoch 1: lr should still be 0.01"
    scheduler.step()
    assert abs(optimizer.lr - 0.01) < 1e-6, "Epoch 2: lr should still be 0.01"
    scheduler.step()
    assert abs(optimizer.lr - 0.001) < 1e-6, (
        "Epoch 3: lr should be 0.001 (after 2 steps)"
    )

    print("  ✅ StepLR convenience class")

    # Test 3: StepScheduler for momentum
    optimizer = nl.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = nl.StepScheduler(
        optimizer, param_name="momentum", step_size=2, gamma=0.9
    )

    assert optimizer.momentum == 0.9, "Initial momentum should be 0.9"
    scheduler.step()
    scheduler.step()
    assert abs(optimizer.momentum - 0.9) < 1e-6, (
        "Momentum should still be 0.9 after 2 steps"
    )
    scheduler.step()
    assert abs(optimizer.momentum - 0.81) < 1e-6, (
        "Momentum should be 0.81 after 3 steps (2//2=1)"
    )

    print("  ✅ StepScheduler for momentum")

    # Test 4: repr
    repr_str = repr(scheduler)
    assert "StepScheduler" in repr_str, "repr should contain 'StepScheduler'"
    print("  ✅ StepScheduler repr")

    print("✅ All StepScheduler tests passed!\n")


def test_cosine_annealing_scheduler():
    """Test CosineAnnealingScheduler"""
    print("Testing CosineAnnealingScheduler...")

    nm.autograd.enable()

    # Test 1: Basic CosineAnnealingScheduler
    model = nl.Sequential(nl.Linear(10, 5))
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.CosineAnnealingScheduler(
        optimizer, param_name="lr", T_max=10, eta_min=0
    )

    assert optimizer.lr == 0.01, "Initial lr should be 0.01"

    # First step returns base value (last_epoch=0 case)
    scheduler.step()
    assert abs(optimizer.lr - 0.01) < 1e-6, "After first step, lr should still be 0.01"

    # Second step starts cosine decay
    scheduler.step()
    assert optimizer.lr < 0.01, "lr should start decreasing"

    # Continue to middle
    for _ in range(3):
        scheduler.step()

    # At epoch 5 (last_epoch=4, which is close to T_max/2)
    mid_lr = optimizer.lr
    assert mid_lr < 0.01, "lr should decrease"
    assert mid_lr > 0, "lr should be above eta_min"

    # Continue to T_max
    for _ in range(5):
        scheduler.step()

    # At last_epoch=9 (< T_max=10), lr should be close to eta_min but not exactly
    assert optimizer.lr < mid_lr, "lr should continue decreasing"
    assert optimizer.lr > 0, "lr should be above eta_min (last_epoch < T_max)"

    # One more step to reach last_epoch >= T_max
    scheduler.step()
    assert abs(optimizer.lr - 0) < 1e-6, (
        "After last_epoch >= T_max, lr should be eta_min (0)"
    )

    print("  ✅ Basic CosineAnnealingScheduler")

    # Test 2: CosineAnnealingLR convenience class
    optimizer = nl.Adam(model.parameters(), lr=0.1)
    scheduler = nl.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.01)

    initial_lr = optimizer.lr
    # Run T_max steps (last_epoch goes from 0 to 4, still < T_max)
    for _ in range(5):
        scheduler.step()

    # At last_epoch=4 (< T_max=5), lr should be close to eta_min but not exactly
    assert optimizer.lr < initial_lr, "lr should decrease"
    assert optimizer.lr > 0.01, "lr should be above eta_min (last_epoch < T_max)"

    # One more step to reach last_epoch >= T_max
    scheduler.step()
    assert abs(optimizer.lr - 0.01) < 1e-6, (
        "At last_epoch >= T_max, lr should be eta_min (0.01)"
    )

    print("  ✅ CosineAnnealingLR convenience class")

    # Test 3: Cosine annealing for momentum
    optimizer = nl.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = nl.CosineAnnealingScheduler(
        optimizer, param_name="momentum", T_max=10, eta_min=0.5
    )

    initial_momentum = optimizer.momentum
    for _ in range(10):
        scheduler.step()

    # At last_epoch=9 (< T_max=10), momentum should be close to eta_min but not exactly
    assert optimizer.momentum < initial_momentum, "momentum should decrease"
    assert optimizer.momentum > 0.5, (
        "momentum should be above eta_min (last_epoch < T_max)"
    )

    # One more step to reach last_epoch >= T_max
    scheduler.step()
    assert abs(optimizer.momentum - 0.5) < 1e-6, (
        "At last_epoch >= T_max, momentum should be eta_min (0.5)"
    )

    print("  ✅ CosineAnnealingScheduler for momentum")

    # Test 4: repr
    repr_str = repr(scheduler)
    assert "CosineAnnealingScheduler" in repr_str, (
        "repr should contain 'CosineAnnealingScheduler'"
    )
    print("  ✅ CosineAnnealingScheduler repr")

    print("✅ All CosineAnnealingScheduler tests passed!\n")


def test_reduce_on_plateau_scheduler():
    """Test ReduceOnPlateauScheduler"""
    print("Testing ReduceOnPlateauScheduler...")

    nm.autograd.enable()

    # Test 1: Basic ReduceOnPlateauScheduler (loss - lower is better)
    model = nl.Sequential(nl.Linear(10, 5))
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.ReduceOnPlateauScheduler(
        optimizer, param_name="lr", better="<", patience=3, factor=0.5, verbose=False
    )

    assert optimizer.lr == 0.01, "Initial lr should be 0.01"

    # Improving losses - lr should not change
    losses = [1.0, 0.9, 0.8]
    for loss in losses:
        scheduler.step(loss)
        assert abs(optimizer.lr - 0.01) < 1e-6, "lr should not change when improving"

    # Plateau - lr should reduce after patience epochs
    plateau_losses = [0.81, 0.82, 0.83, 0.84]
    for loss in plateau_losses:
        scheduler.step(loss)

    assert abs(optimizer.lr - 0.005) < 1e-6, (
        "lr should be reduced to 0.005 after plateau"
    )

    print("  ✅ Basic ReduceOnPlateauScheduler (better='<')")

    # Test 2: ReduceLROnPlateau with mode='min'
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.1, verbose=False
    )

    # Improving
    scheduler.step(1.0)
    scheduler.step(0.9)
    assert abs(optimizer.lr - 0.01) < 1e-6, "lr should not change"

    # Plateau
    scheduler.step(0.91)
    scheduler.step(0.92)
    scheduler.step(0.93)
    assert abs(optimizer.lr - 0.001) < 1e-6, "lr should be reduced to 0.001"

    print("  ✅ ReduceLROnPlateau with mode='min'")

    # Test 3: ReduceLROnPlateau with mode='max' (accuracy)
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, verbose=False
    )

    # Improving accuracy
    scheduler.step(0.5)
    scheduler.step(0.6)
    assert abs(optimizer.lr - 0.01) < 1e-6, "lr should not change when improving"

    # Plateau (accuracy stops increasing)
    scheduler.step(0.59)
    scheduler.step(0.58)
    scheduler.step(0.57)
    assert abs(optimizer.lr - 0.005) < 1e-6, (
        "lr should be reduced after accuracy plateau"
    )

    print("  ✅ ReduceLROnPlateau with mode='max'")

    # Test 4: ReduceOnLossPlateau convenience class
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.ReduceOnLossPlateau(
        optimizer, param_name="lr", patience=2, factor=0.5, verbose=False
    )

    for loss in [1.0, 0.9, 0.91, 0.92, 0.93]:
        scheduler.step(loss)

    assert optimizer.lr < 0.01, "lr should have been reduced"

    print("  ✅ ReduceOnLossPlateau convenience class")

    # Test 5: ReduceOnMetricPlateau convenience class
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.ReduceOnMetricPlateau(
        optimizer, param_name="lr", patience=2, factor=0.5, verbose=False
    )

    for acc in [0.5, 0.6, 0.59, 0.58, 0.57]:
        scheduler.step(acc)

    assert optimizer.lr < 0.01, "lr should have been reduced"

    print("  ✅ ReduceOnMetricPlateau convenience class")

    # Test 6: min_value enforcement
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.ReduceLROnPlateau(
        optimizer, mode="min", patience=1, factor=0.1, min_lr=0.0001, verbose=False
    )

    # Trigger multiple reductions
    for i in range(10):
        scheduler.step(1.0 + i * 0.01)  # Always worse

    assert optimizer.lr >= 0.0001, "lr should not go below min_lr"
    assert abs(optimizer.lr - 0.0001) < 1e-6, "lr should be at min_lr"

    print("  ✅ min_value enforcement")

    # Test 7: repr
    repr_str = repr(scheduler)
    assert "ReduceLROnPlateau" in repr_str, "repr should contain 'ReduceLROnPlateau'"
    print("  ✅ ReduceOnPlateauScheduler repr")

    print("✅ All ReduceOnPlateauScheduler tests passed!\n")


def test_scheduler_state_dict():
    """Test scheduler state_dict and load_state_dict"""
    print("Testing scheduler state_dict...")

    nm.autograd.enable()

    # Test 1: StepScheduler state
    model = nl.Sequential(nl.Linear(10, 5))
    optimizer = nl.Adam(model.parameters(), lr=0.1)
    scheduler = nl.StepScheduler(optimizer, param_name="lr", step_size=2, gamma=0.5)

    # Run a few steps (last_epoch starts at -1, after 3 steps it's 2)
    for _ in range(3):
        scheduler.step()

    # Save state
    state = scheduler.state_dict()
    assert state["last_epoch"] == 2, "last_epoch should be 2 after 3 steps"
    assert state["base_value"] == 0.1, "base_value should be saved"
    assert state["param_name"] == "lr", "param_name should be saved"

    # Create new scheduler and load state
    optimizer2 = nl.Adam(model.parameters(), lr=0.1)
    scheduler2 = nl.StepScheduler(optimizer2, param_name="lr", step_size=2, gamma=0.5)
    scheduler2.load_state_dict(state)

    assert scheduler2.last_epoch == 2, "last_epoch should be restored"
    assert scheduler2.base_value == 0.1, "base_value should be restored"

    print("  ✅ StepScheduler state_dict")

    # Test 2: ReduceOnPlateauScheduler state
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    # Run some steps
    scheduler.step(1.0)
    scheduler.step(0.9)
    scheduler.step(0.91)

    # Save state
    state = scheduler.state_dict()
    assert "best" in state, "best should be in state_dict"
    assert "num_bad_epochs" in state, "num_bad_epochs should be in state_dict"

    # Create new scheduler and load state
    optimizer2 = nl.Adam(model.parameters(), lr=0.01)
    scheduler2 = nl.ReduceLROnPlateau(optimizer2, mode="min", patience=2, factor=0.5)
    scheduler2.load_state_dict(state)

    assert scheduler2.best == scheduler.best, "best should be restored"

    print("  ✅ ReduceOnPlateauScheduler state_dict")

    print("✅ All state_dict tests passed!\n")


def test_multiple_schedulers():
    """Test using multiple schedulers simultaneously"""
    print("Testing multiple schedulers...")

    nm.autograd.enable()

    # Test: Schedule both lr and momentum
    model = nl.Sequential(nl.Linear(10, 5))
    optimizer = nl.SGD(model.parameters(), lr=0.01, momentum=0.9)

    lr_scheduler = nl.StepLR(optimizer, step_size=2, gamma=0.5)
    momentum_scheduler = nl.StepScheduler(
        optimizer, param_name="momentum", step_size=3, gamma=0.9
    )

    initial_lr = optimizer.lr
    initial_momentum = optimizer.momentum

    # Step both schedulers
    for i in range(6):
        lr_scheduler.step()
        momentum_scheduler.step()

    # After 6 steps, last_epoch=5
    # lr_scheduler (step_size=2): 5 // 2 = 2 reductions
    expected_lr = initial_lr * (0.5**2)
    assert abs(optimizer.lr - expected_lr) < 1e-6, f"lr should be {expected_lr}"

    # momentum_scheduler (step_size=3): 5 // 3 = 1 reduction
    expected_momentum = initial_momentum * (0.9**1)
    assert abs(optimizer.momentum - expected_momentum) < 1e-6, (
        f"momentum should be {expected_momentum}"
    )

    print("  ✅ Multiple schedulers")

    print("✅ All multiple scheduler tests passed!\n")


def test_scheduler_edge_cases():
    """Test scheduler edge cases"""
    print("Testing scheduler edge cases...")

    nm.autograd.enable()

    # Test 1: Invalid parameter name
    model = nl.Sequential(nl.Linear(10, 5))
    optimizer = nl.Adam(model.parameters(), lr=0.01)

    try:
        scheduler = nl.StepScheduler(
            optimizer, param_name="invalid_param", step_size=2, gamma=0.5
        )
        assert False, "Should raise ValueError for invalid parameter"
    except ValueError as e:
        assert "does not have parameter" in str(e)

    print("  ✅ Invalid parameter name detection")

    # Test 2: ReduceLROnPlateau with invalid mode
    try:
        scheduler = nl.ReduceLROnPlateau(
            optimizer, mode="invalid", patience=2, factor=0.5
        )
        assert False, "Should raise ValueError for invalid mode"
    except ValueError as e:
        assert "mode must be" in str(e)

    print("  ✅ Invalid mode detection")

    # Test 3: ReduceOnPlateauScheduler with invalid better parameter
    try:
        scheduler = nl.ReduceOnPlateauScheduler(
            optimizer, param_name="lr", better="invalid", patience=2, factor=0.5
        )
        assert False, "Should raise ValueError for invalid better parameter"
    except ValueError as e:
        assert "'better' must be one of" in str(e)

    print("  ✅ Invalid better parameter detection")

    # Test 4: ReduceLROnPlateau with factor >= 1.0
    try:
        scheduler = nl.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=1.5)
        assert False, "Should raise ValueError for factor >= 1.0"
    except ValueError as e:
        assert "Factor should be < 1.0" in str(e)

    print("  ✅ Invalid factor detection")

    # Test 5: Scheduler base class get_value not implemented
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.Scheduler(optimizer, param_name="lr")

    try:
        scheduler.get_value()
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass

    print("  ✅ Scheduler base class NotImplementedError")

    # Test 6: get_last_value
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.StepLR(optimizer, step_size=2, gamma=0.5)

    last_value = scheduler.get_last_value()
    assert abs(last_value - 0.01) < 1e-6, "get_last_value should return current lr"

    # After 2 steps, last_epoch=1, 1//2=0, lr still 0.01
    scheduler.step()
    scheduler.step()

    last_value = scheduler.get_last_value()
    assert abs(last_value - 0.01) < 1e-6, "get_last_value should still return 0.01"

    # After 3rd step, last_epoch=2, 2//2=1, lr reduced to 0.005
    scheduler.step()

    last_value = scheduler.get_last_value()
    assert abs(last_value - 0.005) < 1e-6, (
        "get_last_value should return updated lr (0.005)"
    )

    print("  ✅ get_last_value")

    print("✅ All edge case tests passed!\n")


def test_scheduler_with_optimizer():
    """Test schedulers with different optimizers"""
    print("Testing schedulers with different optimizers...")

    nm.autograd.enable()

    model = nl.Sequential(nl.Linear(10, 5), nl.Relu(), nl.Linear(5, 2))

    # Test 1: StepLR with SGD
    optimizer = nl.SGD(model.parameters(), lr=0.1)
    scheduler = nl.StepLR(optimizer, step_size=2, gamma=0.5)

    x = nm.randn(8, 10)
    y = nm.tensor([i % 2 for i in range(8)])

    for epoch in range(5):
        y_pred = model(x)
        loss = nl.softmax_cross_entropy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    assert optimizer.lr < 0.1, "lr should have been reduced"
    print("  ✅ StepLR with SGD")

    # Test 2: CosineAnnealingLR with Adam
    model = nl.Sequential(nl.Linear(10, 5), nl.Relu(), nl.Linear(5, 2))
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(10):
        y_pred = model(x)
        loss = nl.softmax_cross_entropy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    assert optimizer.lr < 0.01, "lr should have decreased"
    print("  ✅ CosineAnnealingLR with Adam")

    # Test 3: ReduceLROnPlateau with Adam
    model = nl.Sequential(nl.Linear(10, 5), nl.Relu(), nl.Linear(5, 2))
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = nl.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    for epoch in range(10):
        y_pred = model(x)
        loss = nl.softmax_cross_entropy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Simulate plateau by passing same loss
        scheduler.step(1.0)

    assert optimizer.lr < 0.01, "lr should have been reduced due to plateau"
    print("  ✅ ReduceLROnPlateau with Adam")

    print("✅ All scheduler-optimizer integration tests passed!\n")


def run_all_tests():
    """Run all scheduler tests"""
    print("\n" + "=" * 70)
    print("RUNNING SCHEDULER TESTS")
    print("=" * 70 + "\n")

    try:
        test_step_scheduler()
        test_cosine_annealing_scheduler()
        test_reduce_on_plateau_scheduler()
        test_scheduler_state_dict()
        test_multiple_schedulers()
        test_scheduler_edge_cases()
        test_scheduler_with_optimizer()

        print("=" * 70)
        print("✅ ALL SCHEDULER TESTS PASSED!")
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
