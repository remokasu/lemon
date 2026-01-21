import sys
import os
import tempfile

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon.nnlib as nl
import lemon.numlib as nm
from lemon.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_sgd():
    """Test checkpoint save/load with SGD optimizer"""
    print("Testing checkpoint with SGD...")

    nm.autograd.enable()
    xp = nm.np

    # Create model and optimizer
    model = nl.Sequential(nl.Linear(10, 5), nl.Relu(), nl.Linear(5, 2))
    optimizer = nl.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

    # Training step to initialize velocity
    x = nm.randn(4, 10)
    y = model(x)
    loss = nm.sum(y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=5,
            loss=0.123,
            metadata={"test": "data"},
            verbose=False,
        )

        # Create new model and optimizer
        model2 = nl.Sequential(nl.Linear(10, 5), nl.Relu(), nl.Linear(5, 2))
        optimizer2 = nl.SGD(model2.parameters(), lr=0.1)  # Different lr

        # Load checkpoint
        info = load_checkpoint(checkpoint_path, model2, optimizer2, verbose=False)

        # Verify metadata
        assert info["epoch"] == 5, "Epoch should be 5"
        assert abs(info["loss"] - 0.123) < 1e-6, "Loss should be 0.123"
        assert info["metadata"]["test"] == "data", "Metadata should match"

        # Verify optimizer state
        assert optimizer2.lr == 0.01, "Learning rate should be restored"
        assert optimizer2.momentum == 0.9, "Momentum should be restored"
        assert optimizer2.weight_decay == 0.001, "Weight decay should be restored"
        assert optimizer2.velocity is not None, "Velocity should be restored"
        assert len(optimizer2.velocity) > 0, "Velocity list should not be empty"

        # Verify model parameters
        params1 = list(model.parameters())
        params2 = list(model2.parameters())
        for p1, p2 in zip(params1, params2):
            assert xp.allclose(p1.data._data, p2.data._data), "Parameters should match"

        print("  ✅ SGD checkpoint save/load")

    finally:
        os.remove(checkpoint_path)

    print("✅ Checkpoint SGD test passed!\n")


def test_checkpoint_adam():
    """Test checkpoint save/load with Adam optimizer"""
    print("Testing checkpoint with Adam...")

    nm.autograd.enable()
    xp = nm.np

    # Create model and optimizer
    model = nl.Sequential(nl.Linear(8, 4))
    optimizer = nl.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training steps to update m, v, and t
    x = nm.randn(3, 8)
    for _ in range(3):
        y = model(x)
        loss = nm.sum(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(
            checkpoint_path, model, optimizer, epoch=10, loss=0.456, verbose=False
        )

        # Create new model and optimizer
        model2 = nl.Sequential(nl.Linear(8, 4))
        optimizer2 = nl.Adam(model2.parameters(), lr=0.1)  # Different lr

        # Load checkpoint
        info = load_checkpoint(checkpoint_path, model2, optimizer2, verbose=False)

        # Verify optimizer state
        assert optimizer2.lr == 0.001, "Learning rate should be restored"
        assert optimizer2.weight_decay == 0.01, "Weight decay should be restored"
        assert optimizer2.t == 3, "Time step should be restored"
        assert len(optimizer2.m) > 0, "m should be restored"
        assert len(optimizer2.v) > 0, "v should be restored"

        # Verify m and v values
        for m1, m2 in zip(optimizer.m, optimizer2.m):
            assert xp.allclose(m1._data, m2._data), "m values should match"
        for v1, v2 in zip(optimizer.v, optimizer2.v):
            assert xp.allclose(v1._data, v2._data), "v values should match"

        print("  ✅ Adam checkpoint save/load")

    finally:
        os.remove(checkpoint_path)

    print("✅ Checkpoint Adam test passed!\n")


def test_checkpoint_multiple_optimizers():
    """Test checkpoint with various optimizers without code modification"""
    print("Testing checkpoint with multiple optimizers...")

    nm.autograd.enable()

    model = nl.Sequential(nl.Linear(6, 3))
    x = nm.randn(2, 6)

    optimizers_to_test = [
        ("RMSprop", nl.RMSprop(model.parameters(), lr=0.01, alpha=0.95)),
        ("Adagrad", nl.Adagrad(model.parameters(), lr=0.01, eps=1e-10)),
        ("AdamW", nl.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)),
    ]

    for opt_name, optimizer in optimizers_to_test:
        # Training step
        y = model(x)
        loss = nm.sum(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch=1, loss=0.1, verbose=False
            )

            # Load into new optimizer
            model2 = nl.Sequential(nl.Linear(6, 3))
            if opt_name == "RMSprop":
                optimizer2 = nl.RMSprop(model2.parameters(), lr=0.1)
            elif opt_name == "Adagrad":
                optimizer2 = nl.Adagrad(model2.parameters(), lr=0.1)
            elif opt_name == "AdamW":
                optimizer2 = nl.AdamW(model2.parameters(), lr=0.1)

            load_checkpoint(checkpoint_path, model2, optimizer2, verbose=False)

            # Verify lr was restored
            assert optimizer2.lr == optimizer.lr, f"{opt_name} lr should be restored"
            print(f"  ✅ {opt_name} checkpoint")

        finally:
            os.remove(checkpoint_path)

    print("✅ Multiple optimizers test passed!\n")


def test_checkpoint_with_scheduler():
    """Test checkpoint save/load with scheduler"""
    print("Testing checkpoint with scheduler...")

    nm.autograd.enable()
    xp = nm.np

    # Create model, optimizer, and scheduler
    model = nl.Sequential(nl.Linear(5, 3))
    optimizer = nl.SGD(model.parameters(), lr=0.1)
    scheduler = nl.StepScheduler(optimizer, param_name="lr", step_size=2, gamma=0.5)

    # Train for a few epochs
    x = nm.randn(2, 5)
    for epoch in range(3):
        y = model(x)
        loss = nm.sum(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Current lr should be 0.05 (0.1 * 0.5 after epoch 2)
    assert abs(optimizer.lr - 0.05) < 1e-6, "LR should be 0.05 after 3 epochs"

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=3,
            loss=0.789,
            schedulers=[scheduler],
            verbose=False,
        )

        # Create new model, optimizer, and scheduler
        model2 = nl.Sequential(nl.Linear(5, 3))
        optimizer2 = nl.SGD(model2.parameters(), lr=0.1)
        scheduler2 = nl.StepScheduler(
            optimizer2, param_name="lr", step_size=2, gamma=0.5
        )

        # Before loading, optimizer2 should have lr=0.1
        assert abs(optimizer2.lr - 0.1) < 1e-6, "Initial lr should be 0.1"

        # Load checkpoint
        load_checkpoint(checkpoint_path, model2, optimizer2, schedulers=[scheduler2], verbose=False)

        # After loading, lr should be restored to 0.05
        assert abs(optimizer2.lr - 0.05) < 1e-6, "LR should be restored to 0.05"
        assert scheduler2.last_epoch == 2, "Scheduler last_epoch should be restored"

        print("  ✅ Scheduler checkpoint save/load")

    finally:
        os.remove(checkpoint_path)

    print("✅ Checkpoint scheduler test passed!\n")


def test_checkpoint_without_scheduler():
    """Test checkpoint save/load without scheduler (scheduler=None)"""
    print("Testing checkpoint without scheduler...")

    nm.autograd.enable()

    model = nl.Sequential(nl.Linear(4, 2))
    optimizer = nl.SGD(model.parameters(), lr=0.05)

    # Save without scheduler
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=1,
            loss=0.5,
            schedulers=None,
            verbose=False,
        )

        model2 = nl.Sequential(nl.Linear(4, 2))
        optimizer2 = nl.SGD(model2.parameters(), lr=0.1)

        # Load without scheduler
        info = load_checkpoint(
            checkpoint_path, model2, optimizer2, schedulers=None, verbose=False
        )

        assert info["epoch"] == 1, "Epoch should be 1"
        assert optimizer2.lr == 0.05, "LR should be restored"

        print("  ✅ Checkpoint without scheduler")

    finally:
        os.remove(checkpoint_path)

    print("✅ Checkpoint without scheduler test passed!\n")


def test_checkpoint_edge_cases():
    """Test checkpoint edge cases"""
    print("Testing checkpoint edge cases...")

    nm.autograd.enable()

    # Test 1: Empty metadata
    model = nl.Sequential(nl.Linear(3, 2))
    optimizer = nl.SGD(model.parameters(), lr=0.01)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=0,
            loss=0.0,
            metadata=None,
            verbose=False,
        )

        model2 = nl.Sequential(nl.Linear(3, 2))
        optimizer2 = nl.SGD(model2.parameters(), lr=0.1)
        info = load_checkpoint(checkpoint_path, model2, optimizer2, verbose=False)

        assert info["metadata"] == {}, "Metadata should be empty dict"
        print("  ✅ Empty metadata")

    finally:
        os.remove(checkpoint_path)

    # Test 2: Verbose output
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        checkpoint_path = f.name

    try:
        print("    Testing verbose output:")
        save_checkpoint(
            checkpoint_path, model, optimizer, epoch=1, loss=0.1, verbose=True
        )
        load_checkpoint(checkpoint_path, model2, optimizer2, verbose=True)
        print("  ✅ Verbose output")

    finally:
        os.remove(checkpoint_path)

    print("✅ Checkpoint edge cases test passed!\n")


def run_all_tests():
    """Run all checkpoint tests"""
    print("=" * 60)
    print("Running Checkpoint Tests")
    print("=" * 60 + "\n")

    test_checkpoint_sgd()
    test_checkpoint_adam()
    test_checkpoint_multiple_optimizers()
    test_checkpoint_with_scheduler()
    test_checkpoint_without_scheduler()
    test_checkpoint_edge_cases()

    print("=" * 60)
    print("✅ ALL CHECKPOINT TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
