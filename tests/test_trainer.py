import sys
import os
import tempfile
import shutil

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon import numlib as nm
from lemon import nnlib as nl
from lemon.nnlib.data import Dataset, DataLoader
from lemon import trainer as tl


class SimpleDataset(Dataset):
    """Simple dataset for testing"""

    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # XOR problem
        x = nm.tensor([float(idx % 2), float((idx // 2) % 2)])
        y = nm.tensor((idx % 2) ^ ((idx // 2) % 2))
        return x, y


def test_history():
    """Test History class"""
    print("Testing History...")

    # Test 1: Basic history
    history = tl.History()
    history.add(loss=0.5, accuracy=0.8)
    history.add(loss=0.4, accuracy=0.85)
    history.add(loss=0.3, accuracy=0.9)

    assert len(history.loss) == 3, "Should have 3 loss values"
    assert history.loss[0] == 0.5, "First loss should be 0.5"
    assert history.accuracy[-1] == 0.9, "Last accuracy should be 0.9"
    print("  ✅ History basic")

    # Test 2: Save and load
    temp_dir = tempfile.mkdtemp()
    try:
        history_path = os.path.join(temp_dir, "history.json")
        history.save(history_path)

        loaded_history = tl.History.load(history_path)
        assert loaded_history.loss == history.loss, "Loaded history should match"
        assert loaded_history.accuracy == history.accuracy, (
            "Loaded accuracy should match"
        )
        print("  ✅ History save/load")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ All History tests passed!\n")


def test_accuracy_metric():
    """Test Accuracy metric"""
    print("Testing Accuracy metric...")

    metric = tl.Accuracy()

    # Test 1: Perfect accuracy
    y_pred = nm.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    y_true = nm.tensor([1, 0, 1])
    acc = metric(y_pred, y_true)

    assert acc == 100.0, f"Accuracy should be 100.0, got {acc}"
    assert metric.name() == "accuracy", "Metric name should be 'accuracy'"
    print("  ✅ Accuracy perfect")

    # Test 2: Partial accuracy
    y_pred2 = nm.tensor([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    y_true2 = nm.tensor([1, 0, 1])
    acc2 = metric(y_pred2, y_true2)

    expected_acc2 = (2 / 3) * 100.0
    assert abs(acc2 - expected_acc2) < 1e-4, f"Accuracy should be {expected_acc2:.2f}%, got {acc2}"
    print("  ✅ Accuracy partial")

    print("✅ All Accuracy metric tests passed!\n")


def test_progress_logger():
    """Test ProgressLogger"""
    print("Testing ProgressLogger...")

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Logger with file output
        logger = tl.ProgressLogger(experiment_dir=temp_dir, verbose=0)

        model = nl.Sequential(nl.Linear(2, 2))
        optimizer = nl.Adam(model.parameters(), lr=0.01)

        logger.start_training(
            model=model, optimizer=optimizer, epochs=5, batch_size=10, seed=42
        )

        # Log some epochs
        logger.start_epoch(0)
        logger.log_epoch(
            epoch=0,
            total_epochs=5,
            train_metrics={"loss": 0.5, "accuracy": 0.8},
            val_metrics={"loss": 0.6, "accuracy": 0.75},
            is_best=True,
            lr=0.01,
        )

        logger.start_epoch(1)
        logger.log_epoch(
            epoch=1,
            total_epochs=5,
            train_metrics={"loss": 0.4, "accuracy": 0.85},
            val_metrics={"loss": 0.5, "accuracy": 0.82},
            is_best=True,
        )

        logger.end_training(best_epoch=1, best_metrics={"loss": 0.5})

        # Check files were created
        assert os.path.exists(os.path.join(temp_dir, "training.csv")), (
            "CSV should exist"
        )
        assert os.path.exists(os.path.join(temp_dir, "metadata.json")), (
            "Metadata should exist"
        )
        print("  ✅ ProgressLogger with files")

        # Test 2: Logger without file output
        logger_no_file = tl.ProgressLogger(experiment_dir=None, verbose=0)
        logger_no_file.start_training(model, optimizer, epochs=2, batch_size=10)
        logger_no_file.start_epoch(0)
        logger_no_file.log_epoch(epoch=0, total_epochs=2, train_metrics={"loss": 0.5})
        logger_no_file.end_training()
        print("  ✅ ProgressLogger without files")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ All ProgressLogger tests passed!\n")


def test_trainer_basic():
    """Test basic Trainer functionality"""
    print("Testing Trainer basic...")

    dataset = SimpleDataset(size=50)
    model = nl.Sequential(nl.Linear(2, 4), nl.Relu(), nl.Linear(4, 2))

    # Test 1: Basic training
    trainer = tl.Trainer(model=model, dataset=dataset, batch_size=10, verbose=0)

    history = trainer.fit(epochs=2)

    assert len(history.loss) == 2, "Should have 2 epochs of loss"
    assert len(history.accuracy) == 2, "Should have 2 epochs of accuracy"
    print("  ✅ Trainer basic training")

    # Test 2: Training with validation split
    trainer2 = tl.Trainer(
        model=model, dataset=dataset, batch_size=10, validation_split=0.2, verbose=0
    )

    history2 = trainer2.fit(epochs=2)

    assert hasattr(history2, "val_loss"), "Should have validation loss"
    assert hasattr(history2, "val_accuracy"), "Should have validation accuracy"
    assert len(history2.val_loss) == 2, "Should have 2 epochs of val loss"
    print("  ✅ Trainer with validation split")

    print("✅ All Trainer basic tests passed!\n")


def test_trainer_evaluate_predict():
    """Test Trainer evaluate and predict"""
    print("Testing Trainer evaluate and predict...")

    train_dataset = SimpleDataset(size=80)
    test_dataset = SimpleDataset(size=20)

    model = nl.Sequential(nl.Linear(2, 4), nl.Relu(), nl.Linear(4, 2))

    trainer = tl.Trainer(model=model, dataset=train_dataset, batch_size=10, verbose=0)

    # Train
    trainer.fit(epochs=3)

    # Test 1: Evaluate
    test_metrics = trainer.evaluate(test_dataset)

    assert "loss" in test_metrics, "Should have loss"
    assert "accuracy" in test_metrics, "Should have accuracy"
    assert isinstance(test_metrics["loss"], float), "Loss should be float"
    print("  ✅ Trainer evaluate")

    # Test 2: Predict
    x_test = nm.randn(5, 2)
    predictions = trainer.predict(x_test)

    assert predictions.shape == (5, 2), (
        f"Predictions shape should be (5, 2), got {predictions.shape}"
    )
    print("  ✅ Trainer predict")

    print("✅ All Trainer evaluate/predict tests passed!\n")


def test_trainer_save_best():
    """Test Trainer save_best functionality"""
    print("Testing Trainer save_best...")

    temp_dir = tempfile.mkdtemp()

    try:
        dataset = SimpleDataset(size=50)
        model = nl.Sequential(nl.Linear(2, 4), nl.Relu(), nl.Linear(4, 2))

        trainer = tl.Trainer(
            model=model,
            dataset=dataset,
            batch_size=10,
            validation_split=0.2,
            experiment_dir=temp_dir,
            save_best=True,
            verbose=0,
        )

        trainer.fit(epochs=3)

        # Check best model was saved
        best_model_path = os.path.join(temp_dir, "best_model.onnx")
        assert os.path.exists(best_model_path), "Best model should be saved"
        print("  ✅ Trainer save_best")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ All Trainer save_best tests passed!\n")


def test_trainer_restore_best_weights():
    """Test Trainer restore_best_weights"""
    print("Testing Trainer restore_best_weights...")

    dataset = SimpleDataset(size=50)
    model = nl.Sequential(nl.Linear(2, 4), nl.Relu(), nl.Linear(4, 2))

    trainer = tl.Trainer(
        model=model,
        dataset=dataset,
        batch_size=10,
        validation_split=0.3,
        restore_best_weights=True,
        verbose=0,
    )

    history = trainer.fit(epochs=5)

    # Best metric should be stored
    assert trainer.best_metric < float("inf"), "Best metric should be set"
    print("  ✅ Trainer restore_best_weights")

    print("✅ All Trainer restore_best_weights tests passed!\n")


def test_trainer_with_scheduler():
    """Test Trainer with scheduler"""
    print("Testing Trainer with scheduler...")

    from src.lemon.nnlib.scheduler import StepLR

    dataset = SimpleDataset(size=50)
    model = nl.Sequential(nl.Linear(2, 4), nl.Relu(), nl.Linear(4, 2))
    optimizer = nl.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    trainer = tl.Trainer(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        schedulers=[scheduler],
        batch_size=10,
        verbose=0,
    )

    initial_lr = optimizer.lr
    trainer.fit(epochs=3)

    # LR should have changed after 2 epochs
    assert optimizer.lr < initial_lr, (
        f"LR should decrease, was {initial_lr}, now {optimizer.lr}"
    )
    print("  ✅ Trainer with scheduler")

    print("✅ All Trainer scheduler tests passed!\n")


def test_trainer_save_load_weights():
    """Test Trainer save/load weights"""
    print("Testing Trainer save/load weights...")

    temp_dir = tempfile.mkdtemp()

    try:
        dataset = SimpleDataset(size=50)
        model = nl.Sequential(nl.Linear(2, 4), nl.Relu(), nl.Linear(4, 2))

        trainer = tl.Trainer(model=model, dataset=dataset, batch_size=10, verbose=0)

        trainer.fit(epochs=2)

        # Save weights
        weights_path = os.path.join(temp_dir, "weights.npz")
        trainer.save_weights(weights_path)

        # Create new trainer and load weights
        model2 = nl.Sequential(nl.Linear(2, 4), nl.Relu(), nl.Linear(4, 2))
        trainer2 = tl.Trainer(model=model2, dataset=dataset, batch_size=10, verbose=0)

        trainer2.load_weights(weights_path)

        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            import numpy as np

            p1_data = p1.data._data if hasattr(p1.data, "_data") else p1.data
            p2_data = p2.data._data if hasattr(p2.data, "_data") else p2.data
            assert np.allclose(p1_data, p2_data), "Weights should match"

        print("  ✅ Trainer save/load weights")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ All Trainer save/load weights tests passed!\n")


def test_trainer_seed():
    """Test Trainer with seed"""
    print("Testing Trainer with seed...")

    dataset = SimpleDataset(size=50)

    # Train with seed
    model1 = nl.Sequential(nl.Linear(2, 4), nl.Relu(), nl.Linear(4, 2))
    trainer1 = tl.Trainer(
        model=model1, dataset=dataset, batch_size=10, seed=42, verbose=0
    )
    history1 = trainer1.fit(epochs=2)

    # Train with same seed
    model2 = nl.Sequential(nl.Linear(2, 4), nl.Relu(), nl.Linear(4, 2))
    trainer2 = tl.Trainer(
        model=model2, dataset=dataset, batch_size=10, seed=42, verbose=0
    )
    history2 = trainer2.fit(epochs=2)

    # Results should be similar (not exact due to initialization randomness)
    # Just check that both completed successfully
    assert len(history1.loss) == len(history2.loss), (
        "Both should have same number of epochs"
    )
    print("  ✅ Trainer with seed")

    print("✅ All Trainer seed tests passed!\n")


if __name__ == "__main__":
    test_history()
    test_accuracy_metric()
    test_progress_logger()
    test_trainer_basic()
    test_trainer_evaluate_predict()
    test_trainer_save_best()
    test_trainer_restore_best_weights()
    test_trainer_with_scheduler()
    test_trainer_save_load_weights()
    test_trainer_seed()
    print("=" * 50)
    print("All trainer tests completed!")
