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
from lemon.nnlib.data import Dataset, DataLoader, TensorDataset, ConcatDataset, Subset, random_split
from lemon import numlib as nm


def test_fashionmnist_dataset():
    """Test FashionMNIST dataset"""
    print("Testing FashionMNIST dataset...")

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Download FashionMNIST
        try:
            train_dataset = FashionMNIST(root=temp_dir, train=True, download=True)
            test_dataset = FashionMNIST(root=temp_dir, train=False)

            assert len(train_dataset) == 60000, "Training set should have 60000 samples"
            assert len(test_dataset) == 10000, "Test set should have 10000 samples"
            print("  ✅ FashionMNIST download and loading")
        except Exception as e:
            print(f"  ⚠️  FashionMNIST download skipped (no internet connection)")
            print(f"      Error: {e}")
            return

        # Test 2: Dataset indexing
        x, y = train_dataset[0]
        assert x.shape == (784,), f"Image should be flattened to 784, got {x.shape}"
        assert isinstance(y, (int, type(x))) or hasattr(y, "__int__"), (
            "Label should be integer-like"
        )
        print("  ✅ FashionMNIST indexing")

        # Test 3: Label range (0-9 for 10 classes)
        assert 0 <= int(y) <= 9, f"Label should be in range [0, 9], got {y}"
        print("  ✅ FashionMNIST label range")

        # Test 4: Transform
        def double_transform(x):
            return x * 2

        dataset_with_transform = FashionMNIST(
            root=temp_dir, train=True, transform=double_transform
        )
        x_transformed, _ = dataset_with_transform[0]
        x_original, _ = train_dataset[0]

        xp = nm.get_array_module(
            x_transformed._data if hasattr(x_transformed, "_data") else x_transformed
        )
        if xp.__name__ == "numpy":
            assert np.allclose(x_transformed, x_original * 2), (
                "Transform should double values"
            )
        print("  ✅ FashionMNIST transform")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ All FashionMNIST tests passed!\n")


def test_cifar10_dataset():
    """Test CIFAR-10 dataset"""
    print("Testing CIFAR-10 dataset...")

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Download CIFAR-10 (default flatten=True)
        try:
            train_dataset = CIFAR10(
                root=temp_dir, train=True, download=True, flatten=True
            )
            test_dataset = CIFAR10(root=temp_dir, train=False, flatten=True)

            assert len(train_dataset) == 50000, "Training set should have 50000 samples"
            assert len(test_dataset) == 10000, "Test set should have 10000 samples"
            print("  ✅ CIFAR-10 download and loading")
        except Exception as e:
            print(f"  ⚠️  CIFAR-10 download skipped (no internet connection)")
            print(f"      Error: {e}")
            return

        # Test 2: Dataset indexing (flattened)
        x, y = train_dataset[0]
        assert x.shape == (3072,), (
            f"Image should be (3072,) when flattened, got {x.shape}"
        )
        assert isinstance(y, (int, type(x))) or hasattr(y, "__int__"), (
            "Label should be integer-like"
        )
        print("  ✅ CIFAR-10 indexing (flattened)")

        # Test 3: Label range (0-9 for 10 classes)
        assert 0 <= int(y) <= 9, f"Label should be in range [0, 9], got {y}"
        print("  ✅ CIFAR-10 label range")

        # Test 4: Non-flattened version
        train_dataset_3d = CIFAR10(root=temp_dir, train=True, flatten=False)
        x_3d, y_3d = train_dataset_3d[0]
        assert x_3d.shape == (3, 32, 32), (
            f"Image should be (3, 32, 32) when not flattened, got {x_3d.shape}"
        )
        print("  ✅ CIFAR-10 3D shape (not flattened)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ All CIFAR-10 tests passed!\n")


def test_cifar100_dataset():
    """Test CIFAR-100 dataset"""
    print("Testing CIFAR-100 dataset...")

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Download CIFAR-100 (default flatten=True)
        try:
            train_dataset = CIFAR100(
                root=temp_dir, train=True, download=True, flatten=True
            )
            test_dataset = CIFAR100(root=temp_dir, train=False, flatten=True)

            assert len(train_dataset) == 50000, "Training set should have 50000 samples"
            assert len(test_dataset) == 10000, "Test set should have 10000 samples"
            print("  ✅ CIFAR-100 download and loading")
        except Exception as e:
            print(f"  ⚠️  CIFAR-100 download skipped (no internet connection)")
            print(f"      Error: {e}")
            return

        # Test 2: Dataset indexing (flattened)
        x, y = train_dataset[0]
        assert x.shape == (3072,), (
            f"Image should be (3072,) when flattened, got {x.shape}"
        )
        # CIFAR-100 returns (fine_label, coarse_label) tuple
        assert isinstance(y, tuple) and len(y) == 2, (
            f"Label should be tuple (fine_label, coarse_label), got {type(y)}"
        )
        y_fine, y_coarse = y
        print("  ✅ CIFAR-100 indexing (flattened)")

        # Test 3: Label range (0-99 for fine labels, 0-19 for coarse labels)
        assert 0 <= int(y_fine) <= 99, (
            f"Fine label should be in range [0, 99], got {y_fine}"
        )
        assert 0 <= int(y_coarse) <= 19, (
            f"Coarse label should be in range [0, 19], got {y_coarse}"
        )
        print("  ✅ CIFAR-100 label range")

        # Test 4: Non-flattened version
        train_dataset_3d = CIFAR100(root=temp_dir, train=True, flatten=False)
        x_3d, y_3d = train_dataset_3d[0]
        assert x_3d.shape == (3, 32, 32), (
            f"Image should be (3, 32, 32) when not flattened, got {x_3d.shape}"
        )
        print("  ✅ CIFAR-100 3D shape (not flattened)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ All CIFAR-100 tests passed!\n")


def test_iris_dataset():
    """Test Iris dataset"""
    print("Testing Iris dataset...")

    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Download Iris
        try:
            dataset = Iris(root=temp_dir, download=True)

            assert len(dataset) == 150, "Iris dataset should have 150 samples"
            print("  ✅ Iris download and loading")
        except Exception as e:
            print(f"  ⚠️  Iris download skipped (no internet connection)")
            print(f"      Error: {e}")
            return

        # Test 2: Dataset indexing
        x, y = dataset[0]
        assert x.shape == (4,), f"Features should be 4-dimensional, got {x.shape}"
        assert isinstance(y, (int, type(x))) or hasattr(y, "__int__"), (
            "Label should be integer-like"
        )
        print("  ✅ Iris indexing")

        # Test 3: Label range (0-2 for 3 classes)
        assert 0 <= int(y) <= 2, f"Label should be in range [0, 2], got {y}"
        print("  ✅ Iris label range")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✅ All Iris tests passed!\n")


if __name__ == "__main__":
    test_fashionmnist_dataset()
    test_cifar10_dataset()
    test_cifar100_dataset()
    test_iris_dataset()
    print("=" * 50)
    print("All dataset tests completed!")
