import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon.transforms import Normalize, ToTensor, RandomNoise, RandomHorizontalFlip, Compose, Lambda
from lemon.nnlib.data import Dataset, TensorDataset
from lemon import numlib as nm


def test_normalize():
    """Test Normalize transform"""
    print("Testing Normalize transform...")

    # Test 1: Basic normalization
    transform = Normalize(mean=0.5, std=0.5)
    x = nm.tensor([0.0, 0.5, 1.0])
    x_normalized = transform(x)

    expected = nm.tensor([-1.0, 0.0, 1.0])
    xp = nm.get_array_module(
        x_normalized._data if hasattr(x_normalized, "_data") else x_normalized
    )
    if xp.__name__ == "numpy":
        assert np.allclose(x_normalized, expected), (
            f"Expected {expected}, got {x_normalized}"
        )
    print("  ✅ Normalize basic")

    # Test 2: Different mean and std
    transform2 = Normalize(mean=0.0, std=1.0)
    x2 = nm.tensor([1.0, 2.0, 3.0])
    x2_normalized = transform2(x2)

    expected2 = nm.tensor([1.0, 2.0, 3.0])  # (x - 0) / 1 = x
    if xp.__name__ == "numpy":
        assert np.allclose(x2_normalized, expected2), (
            f"Expected {expected2}, got {x2_normalized}"
        )
    print("  ✅ Normalize with different parameters")

    print("✅ All Normalize tests passed!\n")


def test_to_tensor():
    """Test ToTensor transform"""
    print("Testing ToTensor transform...")

    transform = ToTensor()

    # Test 1: Convert numpy array to tensor
    np_array = np.array([1.0, 2.0, 3.0])
    tensor = transform(np_array)

    assert hasattr(tensor, "_data") or hasattr(tensor, "shape"), (
        "Should convert to tensor-like object"
    )
    print("  ✅ ToTensor conversion")

    # Test 2: Already a tensor
    existing_tensor = nm.tensor([1.0, 2.0, 3.0])
    result = transform(existing_tensor)

    xp = nm.get_array_module(result._data if hasattr(result, "_data") else result)
    if xp.__name__ == "numpy":
        assert np.allclose(result, existing_tensor), "Should handle existing tensors"
    print("  ✅ ToTensor with existing tensor")

    print("✅ All ToTensor tests passed!\n")


def test_random_noise():
    """Test RandomNoise transform"""
    print("Testing RandomNoise transform...")

    # Test 1: Add noise
    transform = RandomNoise(std=0.1)
    x = nm.zeros(100)
    x_noisy = transform(x)

    # Noise should make values non-zero
    xp = nm.get_array_module(x_noisy._data if hasattr(x_noisy, "_data") else x_noisy)
    if xp.__name__ == "numpy":
        x_noisy_data = x_noisy._data if hasattr(x_noisy, "_data") else x_noisy
        x_data = x._data if hasattr(x, "_data") else x
        # Not all values should be exactly zero after adding noise
        assert not np.allclose(x_noisy_data, x_data), "Noise should be added"
        # Standard deviation should be roughly 0.1
        std_val = np.std(x_noisy_data)
        assert std_val > 0.05 and std_val < 0.2, (
            f"Noise std should be roughly 0.1, got {std_val}"
        )
    print("  ✅ RandomNoise adds noise")

    # Test 2: Different std
    transform2 = RandomNoise(std=0.5)
    x2 = nm.zeros(100)
    x2_noisy = transform2(x2)

    if xp.__name__ == "numpy":
        x2_noisy_data = x2_noisy._data if hasattr(x2_noisy, "_data") else x2_noisy
        x_noisy_data = x_noisy._data if hasattr(x_noisy, "_data") else x_noisy
        # Larger std should produce larger noise
        assert np.std(x2_noisy_data) > np.std(x_noisy_data), (
            "Larger std should produce larger noise"
        )
    print("  ✅ RandomNoise different std")

    print("✅ All RandomNoise tests passed!\n")


def test_random_horizontal_flip():
    """Test RandomHorizontalFlip transform"""
    print("Testing RandomHorizontalFlip transform...")

    # Test 1: Deterministic flip (p=1.0)
    transform = RandomHorizontalFlip(p=1.0)
    x = nm.tensor([[1, 2, 3], [4, 5, 6]])  # 2x3 image
    x_flipped = transform(x)

    expected = nm.tensor([[3, 2, 1], [6, 5, 4]])  # Horizontally flipped
    xp = nm.get_array_module(
        x_flipped._data if hasattr(x_flipped, "_data") else x_flipped
    )
    if xp.__name__ == "numpy":
        assert np.allclose(x_flipped, expected), f"Expected {expected}, got {x_flipped}"
    print("  ✅ RandomHorizontalFlip with p=1.0")

    # Test 2: No flip (p=0.0)
    transform_no_flip = RandomHorizontalFlip(p=0.0)
    x2 = nm.tensor([[1, 2, 3], [4, 5, 6]])
    x2_result = transform_no_flip(x2)

    if xp.__name__ == "numpy":
        assert np.allclose(x2_result, x2), "Should not flip with p=0.0"
    print("  ✅ RandomHorizontalFlip with p=0.0")

    # Test 3: Random flip (p=0.5) - statistical test
    transform_random = RandomHorizontalFlip(p=0.5)
    flip_count = 0
    trials = 100
    x3 = nm.tensor([[1, 2, 3], [4, 5, 6]])

    for _ in range(trials):
        x3_result = transform_random(x3)
        if xp.__name__ == "numpy":
            if np.allclose(x3_result, expected):  # Flipped
                flip_count += 1

    # Should flip roughly half the time (allow some variance)
    if xp.__name__ == "numpy":
        assert 30 < flip_count < 70, (
            f"Should flip roughly 50% of the time, got {flip_count}/{trials}"
        )
    print("  ✅ RandomHorizontalFlip with p=0.5")

    print("✅ All RandomHorizontalFlip tests passed!\n")


def test_lambda():
    """Test Lambda transform"""
    print("Testing Lambda transform...")

    # Test 1: Simple lambda
    transform = Lambda(lambda x: x * 2)
    x = nm.tensor([1.0, 2.0, 3.0])
    x_transformed = transform(x)

    expected = nm.tensor([2.0, 4.0, 6.0])
    xp = nm.get_array_module(
        x_transformed._data if hasattr(x_transformed, "_data") else x_transformed
    )
    if xp.__name__ == "numpy":
        assert np.allclose(x_transformed, expected), (
            f"Expected {expected}, got {x_transformed}"
        )
    print("  ✅ Lambda simple transform")

    # Test 2: More complex lambda
    def standardize(x):
        # Standardize using numpy for internal calculation
        x_data = x._data if hasattr(x, "_data") else x
        mean_val = np.mean(x_data)
        std_val = np.std(x_data)
        return nm.tensor((x_data - mean_val) / std_val)

    transform2 = Lambda(standardize)
    x2 = nm.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    x2_transformed = transform2(x2)

    # Should be roughly standardized (mean ~0, std ~1)
    if xp.__name__ == "numpy":
        x2_transformed_data = (
            x2_transformed._data if hasattr(x2_transformed, "_data") else x2_transformed
        )
        mean_val = float(np.mean(x2_transformed_data))
        std_val = float(np.std(x2_transformed_data))
        assert abs(mean_val) < 1e-6, f"Mean should be ~0, got {mean_val}"
        assert abs(std_val - 1.0) < 0.1, f"Std should be ~1, got {std_val}"
    print("  ✅ Lambda complex transform")

    print("✅ All Lambda tests passed!\n")


def test_compose():
    """Test Compose transform"""
    print("Testing Compose transform...")

    # Test 1: Compose multiple transforms
    transform = Compose(
        [
            Lambda(lambda x: x * 2),
            Lambda(lambda x: x + 1),
        ]
    )

    x = nm.tensor([1.0, 2.0, 3.0])
    x_transformed = transform(x)

    expected = nm.tensor([3.0, 5.0, 7.0])  # (x * 2) + 1
    xp = nm.get_array_module(
        x_transformed._data if hasattr(x_transformed, "_data") else x_transformed
    )
    if xp.__name__ == "numpy":
        assert np.allclose(x_transformed, expected), (
            f"Expected {expected}, got {x_transformed}"
        )
    print("  ✅ Compose multiple transforms")

    # Test 2: Compose with Normalize
    transform2 = Compose(
        [
            Normalize(mean=0.5, std=0.5),
            Lambda(lambda x: x * 2),
        ]
    )

    x2 = nm.tensor([0.0, 0.5, 1.0])
    x2_transformed = transform2(x2)

    # First: (x - 0.5) / 0.5 = [-1, 0, 1]
    # Then: * 2 = [-2, 0, 2]
    expected2 = nm.tensor([-2.0, 0.0, 2.0])
    if xp.__name__ == "numpy":
        assert np.allclose(x2_transformed, expected2), (
            f"Expected {expected2}, got {x2_transformed}"
        )
    print("  ✅ Compose with Normalize")

    # Test 3: Empty compose
    transform_empty = Compose([])
    x3 = nm.tensor([1.0, 2.0, 3.0])
    x3_transformed = transform_empty(x3)

    if xp.__name__ == "numpy":
        assert np.allclose(x3_transformed, x3), "Empty compose should return original"
    print("  ✅ Compose empty")

    print("✅ All Compose tests passed!\n")


def test_transforms_with_dataset():
    """Test transforms integrated with dataset"""
    print("Testing transforms with dataset...")

    # Create dataset
    X = nm.randn(10, 5)
    y = nm.randint(10, low=0, high=2)
    dataset = TensorDataset(X, y)

    # Test 1: Apply transform to dataset
    transform = Lambda(lambda x: x * 2)

    # Create a simple wrapper to apply transform
    class TransformedDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y

    transformed_dataset = TransformedDataset(dataset, transform)
    x_original, _ = dataset[0]
    x_transformed, _ = transformed_dataset[0]

    xp = nm.get_array_module(
        x_transformed._data if hasattr(x_transformed, "_data") else x_transformed
    )
    if xp.__name__ == "numpy":
        assert np.allclose(x_transformed, x_original * 2), (
            "Transform should be applied to dataset"
        )
    print("  ✅ Transforms with dataset")

    print("✅ All transform-dataset integration tests passed!\n")


if __name__ == "__main__":
    test_normalize()
    test_to_tensor()
    test_random_noise()
    test_random_horizontal_flip()
    test_lambda()
    test_compose()
    test_transforms_with_dataset()
    print("=" * 50)
    print("All transform tests completed!")
