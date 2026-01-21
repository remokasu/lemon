import sys
import os
import tempfile
import shutil
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon.datasets.vision import MNIST, CIFAR10
from lemon.datasets.tabular import Iris, CSVDataset
from lemon.nnlib.data import Dataset, DataLoader, TensorDataset, ConcatDataset, Subset, random_split, SupervisedDataSet, random_sample, split_dataset
from lemon import numlib as nm


def test_supervised_dataset():
    """Test SupervisedDataSet"""
    print("Testing SupervisedDataSet...")

    # Test 1: Create dataset
    dataset = SupervisedDataSet(input_dim=2, target_dim=1)

    # Test 2: Add samples
    dataset.add_sample([0, 0], [0])
    dataset.add_sample([0, 1], [1])
    dataset.add_sample([1, 0], [1])
    dataset.add_sample([1, 1], [0])

    assert len(dataset) == 4, f"Dataset should have 4 samples, got {len(dataset)}"
    print("  ✅ SupervisedDataSet add_sample and length")

    # Test 3: Dataset indexing
    x, y = dataset[0]
    assert x.shape == (2,), f"Input should be 2-dimensional, got {x.shape}"
    assert y.shape == (1,), f"Target should be 1-dimensional, got {y.shape}"
    print("  ✅ SupervisedDataSet indexing")

    # Test 4: Verify first sample values
    xp = nm.get_array_module(x._data if hasattr(x, "_data") else x)
    if xp.__name__ == "numpy":
        expected_x = np.array([0.0, 0.0])
        expected_y = np.array([0.0])
        assert np.allclose(x, expected_x), f"Expected {expected_x}, got {x}"
        assert np.allclose(y, expected_y), f"Expected {expected_y}, got {y}"
    print("  ✅ SupervisedDataSet values")

    # Test 5: Error on dimension mismatch
    try:
        dataset.add_sample([0, 0, 0], [0])  # Wrong input_dim (should be 2, not 3)
        assert False, "Should raise ValueError for input dimension mismatch"
    except ValueError as e:
        assert "input" in str(e).lower()
    print("  ✅ SupervisedDataSet input dimension validation")

    try:
        dataset.add_sample([0, 0], [0, 1])  # Wrong target_dim (should be 1, not 2)
        assert False, "Should raise ValueError for target dimension mismatch"
    except ValueError as e:
        assert "target" in str(e).lower()
    print("  ✅ SupervisedDataSet target dimension validation")

    print("✅ All SupervisedDataSet tests passed!\n")


def test_csvdataset():
    """Test CSVDataset"""
    print("Testing CSVDataset...")

    temp_dir = tempfile.mkdtemp()

    try:
        # Create a test CSV file with x/y columns (auto-detect mode)
        csv_path = os.path.join(temp_dir, "test.csv")

        # Test 1: CSV with auto-detect columns
        with open(csv_path, "w") as f:
            f.write("x0,x1,x2,y0\n")
            f.write("1.0,2.0,3.0,0\n")
            f.write("4.0,5.0,6.0,1\n")
            f.write("7.0,8.0,9.0,0\n")

        dataset = CSVDataset(root=temp_dir, csv_file="test.csv")

        assert len(dataset) == 3, f"Dataset should have 3 samples, got {len(dataset)}"
        print("  ✅ CSVDataset length")

        # Test 2: Dataset indexing
        x, y = dataset[0]
        assert x.shape == (3,), f"Features should be 3-dimensional, got {x.shape}"
        assert int(y) == 0, f"First label should be 0, got {y}"
        print("  ✅ CSVDataset indexing")

        # Test 3: Data values
        xp = nm.get_array_module(x._data if hasattr(x, "_data") else x)
        if xp.__name__ == "numpy":
            expected_x = np.array([1.0, 2.0, 3.0])
            assert np.allclose(x, expected_x), f"Expected {expected_x}, got {x}"
        print("  ✅ CSVDataset values")

        # Test 4: CSV with explicit column specification
        csv_path_explicit = os.path.join(temp_dir, "test_explicit.csv")
        with open(csv_path_explicit, "w") as f:
            f.write("feature1,feature2,feature3,label\n")
            f.write("1.0,2.0,3.0,0\n")
            f.write("4.0,5.0,6.0,1\n")

        dataset_explicit = CSVDataset(
            root=temp_dir,
            csv_file="test_explicit.csv",
            xs=["feature1", "feature2", "feature3"],
            ys=["label"],
        )

        assert len(dataset_explicit) == 2, (
            "Dataset with explicit columns should have 2 samples"
        )
        print("  ✅ CSVDataset with explicit columns")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ All CSVDataset tests passed!\n")


def test_concat_dataset():
    """Test ConcatDataset"""
    print("Testing ConcatDataset...")

    # Create test datasets
    X1 = nm.randn(50, 10)
    y1 = nm.randint(50, low=0, high=2)
    dataset1 = TensorDataset(X1, y1)

    X2 = nm.randn(30, 10)
    y2 = nm.randint(30, low=0, high=2)
    dataset2 = TensorDataset(X2, y2)

    # Test 1: Concatenate datasets
    concat_dataset = ConcatDataset([dataset1, dataset2])
    assert len(concat_dataset) == 80, (
        f"Concatenated dataset should have 80 samples, got {len(concat_dataset)}"
    )
    print("  ✅ ConcatDataset length")

    # Test 2: Access first dataset
    x, y = concat_dataset[0]
    x1, y1_val = dataset1[0]
    xp = nm.get_array_module(x._data if hasattr(x, "_data") else x)
    if xp.__name__ == "numpy":
        assert np.allclose(x, x1), "First item should come from first dataset"
        assert y == y1_val, "First label should match first dataset"
    print("  ✅ ConcatDataset first dataset access")

    # Test 3: Access second dataset
    x, y = concat_dataset[50]
    x2, y2_val = dataset2[0]
    if xp.__name__ == "numpy":
        assert np.allclose(x, x2), "Item 50 should be first item of second dataset"
        assert y == y2_val, "Label should match second dataset"
    print("  ✅ ConcatDataset second dataset access")

    # Test 4: Negative indexing
    x, y = concat_dataset[-1]
    x_last, y_last = dataset2[-1]
    if xp.__name__ == "numpy":
        assert np.allclose(x, x_last), "Last item should be last item of second dataset"
    print("  ✅ ConcatDataset negative indexing")

    # Test 5: Empty dataset error
    try:
        ConcatDataset([])
        assert False, "Should raise ValueError for empty datasets list"
    except ValueError as e:
        assert "empty" in str(e).lower()
    print("  ✅ ConcatDataset empty error")

    print("✅ All ConcatDataset tests passed!\n")


def test_subset():
    """Test Subset"""
    print("Testing Subset...")

    # Create test dataset
    X = nm.randn(100, 10)
    y = nm.randint(100, low=0, high=5)
    dataset = TensorDataset(X, y)

    # Test 1: Create subset
    indices = [0, 5, 10, 15, 20]
    subset = Subset(dataset, indices)
    assert len(subset) == 5, f"Subset should have 5 samples, got {len(subset)}"
    print("  ✅ Subset length")

    # Test 2: Access subset items
    x, y_val = subset[0]
    x_original, y_original = dataset[0]
    xp = nm.get_array_module(x._data if hasattr(x, "_data") else x)
    if xp.__name__ == "numpy":
        assert np.allclose(x, x_original), "First subset item should be from index 0"
        assert y_val == y_original, "First subset label should match"
    print("  ✅ Subset item access")

    # Test 3: Correct mapping
    x, y_val = subset[1]  # Should map to dataset[5]
    x_mapped, y_mapped = dataset[5]
    if xp.__name__ == "numpy":
        assert np.allclose(x, x_mapped), "Subset index 1 should map to dataset index 5"
        assert y_val == y_mapped, "Subset label should match mapped index"
    print("  ✅ Subset index mapping")

    print("✅ All Subset tests passed!\n")


def test_random_split():
    """Test random_split function"""
    print("Testing random_split...")

    # Create test dataset
    X = nm.randn(100, 10)
    y = nm.randint(100, low=0, high=5)
    dataset = TensorDataset(X, y)

    # Test 1: Split dataset
    train_dataset, val_dataset = random_split(dataset, lengths=[70, 30], seed=42)
    assert len(train_dataset) == 70, (
        f"Train dataset should have 70 samples, got {len(train_dataset)}"
    )
    assert len(val_dataset) == 30, (
        f"Val dataset should have 30 samples, got {len(val_dataset)}"
    )
    print("  ✅ random_split lengths")

    # Test 2: Reproducibility with seed
    train_dataset2, val_dataset2 = random_split(dataset, lengths=[70, 30], seed=42)
    # Should produce same split with same seed
    x1, _ = train_dataset[0]
    x2, _ = train_dataset2[0]
    xp = nm.get_array_module(x1._data if hasattr(x1, "_data") else x1)
    if xp.__name__ == "numpy":
        assert np.allclose(x1, x2), "Same seed should produce same split"
    print("  ✅ random_split reproducibility")

    # Test 3: Error on incorrect lengths
    try:
        random_split(dataset, lengths=[70, 40])  # Sum = 110 > 100
        assert False, "Should raise ValueError for incorrect lengths"
    except ValueError:
        pass
    print("  ✅ random_split length validation")

    print("✅ All random_split tests passed!\n")


def test_random_sample():
    """Test random_sample function"""
    print("Testing random_sample...")

    # Create test dataset
    X = nm.randn(100, 10)
    y = nm.randint(100, low=0, high=5)
    dataset = TensorDataset(X, y)

    # Test 1: Sample dataset
    sampled_dataset = random_sample(dataset, n=20, seed=42)
    assert len(sampled_dataset) == 20, (
        f"Sampled dataset should have 20 samples, got {len(sampled_dataset)}"
    )
    print("  ✅ random_sample length")

    # Test 2: Reproducibility with seed
    sampled_dataset2 = random_sample(dataset, n=20, seed=42)
    x1, _ = sampled_dataset[0]
    x2, _ = sampled_dataset2[0]
    xp = nm.get_array_module(x1._data if hasattr(x1, "_data") else x1)
    if xp.__name__ == "numpy":
        assert np.allclose(x1, x2), "Same seed should produce same sample"
    print("  ✅ random_sample reproducibility")

    # Test 3: n > dataset size returns all samples
    sampled_large = random_sample(dataset, n=150)  # n > dataset size
    assert len(sampled_large) == 100, "Should return all samples when n > dataset size"
    print("  ✅ random_sample with n > dataset size")

    print("✅ All random_sample tests passed!\n")


def test_split_dataset():
    """Test split_dataset function"""
    print("Testing split_dataset...")

    # Create test dataset
    X = nm.randn(100, 10)
    y = nm.randint(100, low=0, high=5)
    dataset = TensorDataset(X, y)

    # Test 1: Split into train/val/test with ratios
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, ratios=[0.7, 0.15, 0.15], seed=42
    )
    assert len(train_dataset) == 70, (
        f"Train dataset should have 70 samples, got {len(train_dataset)}"
    )
    assert len(val_dataset) == 15, (
        f"Val dataset should have 15 samples, got {len(val_dataset)}"
    )
    assert len(test_dataset) == 15, (
        f"Test dataset should have 15 samples, got {len(test_dataset)}"
    )
    print("  ✅ split_dataset with ratios")

    # Test 2: Split into train/val only
    train_dataset, val_dataset = split_dataset(dataset, ratios=[0.8, 0.2], seed=42)
    assert len(train_dataset) == 80, (
        f"Train dataset should have 80 samples, got {len(train_dataset)}"
    )
    assert len(val_dataset) == 20, (
        f"Val dataset should have 20 samples, got {len(val_dataset)}"
    )
    print("  ✅ split_dataset two-way split")

    # Test 3: Reproducibility
    train_dataset2, val_dataset2 = split_dataset(dataset, ratios=[0.8, 0.2], seed=42)
    x1, _ = train_dataset[0]
    x2, _ = train_dataset2[0]
    xp = nm.get_array_module(x1._data if hasattr(x1, "_data") else x1)
    if xp.__name__ == "numpy":
        assert np.allclose(x1, x2), "Same seed should produce same split"
    print("  ✅ split_dataset reproducibility")

    # Test 4: Error on invalid ratio
    try:
        split_dataset(dataset, ratios=[0.5, 0.6])  # Sum > 1.0
        assert False, "Should raise ValueError for invalid split ratio"
    except (ValueError, AssertionError):
        pass
    print("  ✅ split_dataset ratio validation")

    print("✅ All split_dataset tests passed!\n")


if __name__ == "__main__":
    test_supervised_dataset()
    test_csvdataset()
    test_concat_dataset()
    test_subset()
    test_random_split()
    test_random_sample()
    test_split_dataset()
    print("=" * 50)
    print("All dataset utility tests completed!")
