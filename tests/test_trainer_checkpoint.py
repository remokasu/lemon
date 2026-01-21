import sys
import os
import tempfile

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon as nc


class SimpleDataset(nc.Dataset):
    """Simple dataset for testing"""

    def __init__(self, size=20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return nc.randn(10), nc.tensor(idx % 2)


def test_trainer_checkpoint_basic():
    """Test basic Trainer checkpoint save/load"""
    print("Testing Trainer checkpoint basic...")

    dataset = SimpleDataset()
    model = nc.Sequential(nc.Linear(10, 5), nc.Relu(), nc.Linear(5, 2))
    optimizer = nc.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trainer = nc.Trainer(
        model=model, dataset=dataset, optimizer=optimizer, batch_size=4, verbose=0
    )

    # Train
    trainer.fit(epochs=2)

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        checkpoint_path = f.name

    try:
        trainer.save_checkpoint(checkpoint_path, epoch=2, loss=0.5, verbose=False)

        # Create new trainer
        model2 = nc.Sequential(nc.Linear(10, 5), nc.Relu(), nc.Linear(5, 2))
        optimizer2 = nc.SGD(model2.parameters(), lr=0.1, momentum=0.9)  # Match momentum

        trainer2 = nc.Trainer(
            model=model2, dataset=dataset, optimizer=optimizer2, batch_size=4, verbose=0
        )

        # Load checkpoint
        info = trainer2.load_checkpoint(checkpoint_path, verbose=False)

        assert info["epoch"] == 2, "Epoch should be 2"
        assert abs(info["loss"] - 0.5) < 1e-6, "Loss should be 0.5"
        assert abs(trainer2.optimizer.lr - 0.01) < 1e-6, "LR should be restored"
        assert trainer2.optimizer.momentum == 0.9, "Momentum should be restored"

        # Continue training
        trainer2.fit(epochs=1)

        print("  ✅ Trainer checkpoint basic")

    finally:
        os.remove(checkpoint_path)

    print("✅ Trainer checkpoint basic test passed!\n")


def test_trainer_checkpoint_with_scheduler():
    """Test Trainer checkpoint with scheduler"""
    print("Testing Trainer checkpoint with scheduler...")

    dataset = SimpleDataset()
    model = nc.Sequential(nc.Linear(10, 5), nc.Relu(), nc.Linear(5, 2))
    optimizer = nc.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = nc.StepScheduler(optimizer, param_name="lr", step_size=2, gamma=0.5)

    trainer = nc.Trainer(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        schedulers=[scheduler],
        batch_size=4,
        verbose=0,
    )

    # Train for 3 epochs
    trainer.fit(epochs=3)
    lr_after_3 = trainer.optimizer.lr

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        checkpoint_path = f.name

    try:
        trainer.save_checkpoint(checkpoint_path, epoch=3, loss=0.5, verbose=False)

        # Create new trainer
        model2 = nc.Sequential(nc.Linear(10, 5), nc.Relu(), nc.Linear(5, 2))
        optimizer2 = nc.SGD(model2.parameters(), lr=0.1, momentum=0.9)  # Match momentum
        scheduler2 = nc.StepScheduler(
            optimizer2, param_name="lr", step_size=2, gamma=0.5
        )

        trainer2 = nc.Trainer(
            model=model2,
            dataset=dataset,
            optimizer=optimizer2,
            schedulers=[scheduler2],
            batch_size=4,
            verbose=0,
        )

        # Load checkpoint
        info = trainer2.load_checkpoint(checkpoint_path, verbose=False)

        assert abs(trainer2.optimizer.lr - lr_after_3) < 1e-6, "LR should match"
        assert trainer2.schedulers[0].last_epoch == 2, "Scheduler epoch should match"

        # Continue training
        trainer2.fit(epochs=2)

        print("  ✅ Trainer checkpoint with scheduler")

    finally:
        os.remove(checkpoint_path)

    print("✅ Trainer checkpoint with scheduler test passed!\n")


def test_trainer_checkpoint_metadata():
    """Test Trainer checkpoint with metadata"""
    print("Testing Trainer checkpoint with metadata...")

    dataset = SimpleDataset(size=10)
    model = nc.Sequential(nc.Linear(10, 2))
    trainer = nc.Trainer(model=model, dataset=dataset, batch_size=4, verbose=0)

    trainer.fit(epochs=1)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        checkpoint_path = f.name

    try:
        # Save with metadata
        trainer.save_checkpoint(
            checkpoint_path,
            epoch=1,
            loss=0.5,
            metadata={"test_name": "test123", "version": 1},
            verbose=False,
        )

        # Load
        model2 = nc.Sequential(nc.Linear(10, 2))
        trainer2 = nc.Trainer(model=model2, dataset=dataset, batch_size=4, verbose=0)

        info = trainer2.load_checkpoint(checkpoint_path, verbose=False)

        assert info["metadata"]["test_name"] == "test123", "Metadata should match"
        assert info["metadata"]["version"] == 1, "Metadata should match"

        print("  ✅ Trainer checkpoint with metadata")

    finally:
        os.remove(checkpoint_path)

    print("✅ Trainer checkpoint with metadata test passed!\n")


def run_all_tests():
    """Run all Trainer checkpoint tests"""
    print("=" * 60)
    print("Running Trainer Checkpoint Tests")
    print("=" * 60 + "\n")

    test_trainer_checkpoint_basic()
    test_trainer_checkpoint_with_scheduler()
    test_trainer_checkpoint_metadata()

    print("=" * 60)
    print("✅ ALL LEARNER CHECKPOINT TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
