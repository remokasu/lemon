import sys
import os
import numpy as np
import pytest

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon.datasets.vision import FashionMNIST, CIFAR10, CIFAR100
from lemon.datasets.tabular import Iris, CSVDataset
from lemon.nnlib.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    ConcatDataset,
    Subset,
    random_split,
)
from lemon import numlib as nm


@pytest.mark.network
def test_fashionmnist_dataset(dataset_root):
    """Test FashionMNIST dataset"""
    print("Testing FashionMNIST dataset...")

    root = dataset_root

    # Test 1: Download FashionMNIST
    try:
        train_dataset = FashionMNIST(root=root, train=True, download=True, flatten=True)
        test_dataset = FashionMNIST(root=root, train=False, flatten=True)

        assert len(train_dataset) == 60000, "Training set should have 60000 samples"
        assert len(test_dataset) == 10000, "Test set should have 10000 samples"
        print("  ✅ FashionMNIST download and loading")
    except (RuntimeError, OSError) as e:
        pytest.skip(f"FashionMNIST download failed: {e}")

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
        root=root, train=True, transform=double_transform, flatten=True
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

    print("✅ All FashionMNIST tests passed!\n")


@pytest.mark.network
def test_cifar10_dataset(dataset_root):
    """Test CIFAR-10 dataset"""
    print("Testing CIFAR-10 dataset...")

    root = dataset_root

    # Test 1: Download CIFAR-10 (default flatten=True)
    try:
        train_dataset = CIFAR10(root=root, train=True, download=True, flatten=True)
        test_dataset = CIFAR10(root=root, train=False, flatten=True)

        assert len(train_dataset) == 50000, "Training set should have 50000 samples"
        assert len(test_dataset) == 10000, "Test set should have 10000 samples"
        print("  ✅ CIFAR-10 download and loading")
    except (RuntimeError, OSError) as e:
        pytest.skip(f"CIFAR-10 download failed: {e}")

    # Test 2: Dataset indexing (flattened)
    x, y = train_dataset[0]
    assert x.shape == (3072,), f"Image should be (3072,) when flattened, got {x.shape}"
    assert isinstance(y, (int, type(x))) or hasattr(y, "__int__"), (
        "Label should be integer-like"
    )
    print("  ✅ CIFAR-10 indexing (flattened)")

    # Test 3: Label range (0-9 for 10 classes)
    assert 0 <= int(y) <= 9, f"Label should be in range [0, 9], got {y}"
    print("  ✅ CIFAR-10 label range")

    # Test 4: Non-flattened version
    train_dataset_3d = CIFAR10(root=root, train=True, flatten=False)
    x_3d, y_3d = train_dataset_3d[0]
    assert x_3d.shape == (3, 32, 32), (
        f"Image should be (3, 32, 32) when not flattened, got {x_3d.shape}"
    )
    print("  ✅ CIFAR-10 3D shape (not flattened)")

    print("✅ All CIFAR-10 tests passed!\n")


@pytest.mark.network
def test_cifar100_dataset(dataset_root):
    """Test CIFAR-100 dataset"""
    print("Testing CIFAR-100 dataset...")

    root = dataset_root

    # Test 1: Download CIFAR-100 (default flatten=True)
    try:
        train_dataset = CIFAR100(root=root, train=True, download=True, flatten=True)
        test_dataset = CIFAR100(root=root, train=False, flatten=True)

        assert len(train_dataset) == 50000, "Training set should have 50000 samples"
        assert len(test_dataset) == 10000, "Test set should have 10000 samples"
        print("  ✅ CIFAR-100 download and loading")
    except (RuntimeError, OSError) as e:
        pytest.skip(f"CIFAR-100 download failed: {e}")

    # Test 2: Dataset indexing (flattened)
    x, y = train_dataset[0]
    assert x.shape == (3072,), f"Image should be (3072,) when flattened, got {x.shape}"
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
    train_dataset_3d = CIFAR100(root=root, train=True, flatten=False)
    x_3d, y_3d = train_dataset_3d[0]
    assert x_3d.shape == (3, 32, 32), (
        f"Image should be (3, 32, 32) when not flattened, got {x_3d.shape}"
    )
    print("  ✅ CIFAR-100 3D shape (not flattened)")

    print("✅ All CIFAR-100 tests passed!\n")


@pytest.mark.network
def test_iris_dataset(dataset_root):
    """Test Iris dataset"""
    print("Testing Iris dataset...")

    root = dataset_root

    # Test 1: Download Iris
    try:
        dataset = Iris(root=root, download=True)

        assert len(dataset) == 150, "Iris dataset should have 150 samples"
        print("  ✅ Iris download and loading")
    except (RuntimeError, OSError) as e:
        pytest.skip(f"Iris download failed: {e}")

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

    print("✅ All Iris tests passed!\n")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "--run-network", "-v", "-s"]))
