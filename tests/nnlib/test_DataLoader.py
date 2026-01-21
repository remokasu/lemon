import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon import numlib as nm
from lemon.nnlib.data import TensorDataset, DataLoader, ConcatDataset, Subset, Dataset


def test_dataset():
    """Test Dataset classes"""

    print("Testing Dataset classes...")

    # Test 1: TensorDataset basic
    X = nm.randn(100, 10)
    y = nm.randint(100, low=0, high=5)

    dataset = TensorDataset(X, y)

    assert len(dataset) == 100, "Dataset length should be 100"
    print("  ✅ TensorDataset length")

    # Test 2: TensorDataset indexing
    x_sample, y_sample = dataset[0]
    assert x_sample.shape == (10,), (
        f"Sample shape should be (10,), got {x_sample.shape}"
    )
    print("  ✅ TensorDataset indexing")

    # Test 3: TensorDataset with multiple items
    X = nm.randn(50, 5)
    y = nm.randn(50, 2)
    z = nm.randn(50)

    dataset = TensorDataset(X, y, z)
    assert len(dataset) == 50, "Dataset length should be 50"

    x, y_val, z_val = dataset[0]
    assert x.shape == (5,), "First item shape"
    assert y_val.shape == (2,), "Second item shape"
    assert z_val.shape == (), "Third item shape"
    print("  ✅ TensorDataset with multiple items")

    # Test 4: Empty dataset error
    try:
        TensorDataset()
        assert False, "Should raise ValueError for empty dataset"
    except ValueError as e:
        assert "At least one tensor required" in str(e)
    print("  ✅ TensorDataset empty error")

    # Test 5: Size mismatch error
    X = nm.randn(100, 10)
    y = nm.randn(50, 2)

    try:
        TensorDataset(X, y)
        assert False, "Should raise ValueError for size mismatch"
    except ValueError as e:
        assert "same size" in str(e)
    print("  ✅ TensorDataset size mismatch error")

    print("✅ All Dataset tests passed!\n")


def test_dataloader():
    """Test DataLoader"""

    print("Testing DataLoader...")

    xp = nm.np

    # Test 1: Basic DataLoader
    X = nm.randn(100, 10)
    y = nm.randint(100, low=0, high=5)
    dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    assert len(loader) == 10, f"DataLoader length should be 10, got {len(loader)}"
    print("  ✅ DataLoader length")

    # Test 2: Iterate through DataLoader
    batch_count = 0
    for X_batch, y_batch in loader:
        batch_count += 1
        assert X_batch.shape[0] == 10, "Batch size should be 10"
        assert X_batch.shape[1] == 10, "Feature size should be 10"

    assert batch_count == 10, f"Should have 10 batches, got {batch_count}"
    print("  ✅ DataLoader iteration")

    # Test 3: DataLoader with shuffle
    loader_shuffle = DataLoader(dataset, batch_size=10, shuffle=True)

    batches1 = []
    batches2 = []

    for X_batch, _ in loader_shuffle:
        batches1.append(X_batch._data[0].copy())

    for X_batch, _ in loader_shuffle:
        batches2.append(X_batch._data[0].copy())

    # Batches should be different (with high probability)
    all_same = all(xp.allclose(b1, b2) for b1, b2 in zip(batches1, batches2))
    assert not all_same, "Shuffled batches should be different"
    print("  ✅ DataLoader with shuffle")

    # Test 4: DataLoader with drop_last
    X = nm.randn(95, 10)
    y = nm.randint(95, low=0, high=5)
    dataset = TensorDataset(X, y)

    loader_drop = DataLoader(dataset, batch_size=10, drop_last=True)

    assert len(loader_drop) == 9, (
        f"Should have 9 complete batches, got {len(loader_drop)}"
    )

    batch_count = 0
    for X_batch, _ in loader_drop:
        batch_count += 1
        assert X_batch.shape[0] == 10, "All batches should be complete"

    assert batch_count == 9, "Should iterate through 9 batches"
    print("  ✅ DataLoader with drop_last")

    # Test 5: DataLoader without drop_last (incomplete last batch)
    loader_keep = DataLoader(dataset, batch_size=10, drop_last=False)

    assert len(loader_keep) == 10, f"Should have 10 batches, got {len(loader_keep)}"

    batches = []
    for X_batch, _ in loader_keep:
        batches.append(X_batch.shape[0])

    assert batches[-1] == 5, "Last batch should have 5 samples"
    print("  ✅ DataLoader with incomplete last batch")

    # Test 6: Small batch size
    loader_small = DataLoader(dataset, batch_size=1)
    assert len(loader_small) == 95, "Should have 95 batches of size 1"
    print("  ✅ DataLoader with batch_size=1")

    # Test 7: Large batch size
    loader_large = DataLoader(dataset, batch_size=200)
    assert len(loader_large) == 1, "Should have 1 batch"

    for X_batch, _ in loader_large:
        assert X_batch.shape[0] == 95, "Batch should contain all samples"
    print("  ✅ DataLoader with large batch_size")

    print("✅ All DataLoader tests passed!\n")


def test_dataset_edge_cases():
    """Test Dataset edge cases for coverage"""

    print("Testing Dataset edge cases...")

    # Test 1: Base Dataset class methods raise NotImplementedError
    dataset = Dataset()

    try:
        len(dataset)
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass
    print("  ✅ Dataset __len__ NotImplementedError")

    try:
        dataset[0]
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass
    print("  ✅ Dataset __getitem__ NotImplementedError")

    print("✅ All Dataset edge cases passed!\n")


def test_dataloader_edge_cases():
    """Test DataLoader edge cases for coverage"""

    print("Testing DataLoader edge cases...")

    # Test 1: DataLoader with batch[0] having length != 2 (general case)
    # Create a dataset that returns 3 items
    class TripleDataset(Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return nm.randn(5), nm.randn(3), nm.tensor([idx])

    dataset = TripleDataset(20)
    loader = DataLoader(dataset, batch_size=4)

    for batch in loader:
        assert len(batch) == 3, "Should return 3 items"
        assert batch[0].shape[0] == 4 or batch[0].shape[0] == 20 % 4, "Batch size check"
        break
    print("  ✅ DataLoader with triple output (general case)")

    # Test 2: DataLoader with raw arrays (not NumType)
    class RawArrayDataset(Dataset):
        def __init__(self, size):
            self.size = size
            self.X = nm.np.random.randn(size, 5)
            self.y = nm.np.random.randint(0, 2, size)

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Return raw numpy arrays, not NumType
            return self.X[idx], self.y[idx]

    dataset = RawArrayDataset(10)
    loader = DataLoader(dataset, batch_size=3)

    for X_batch, y_batch in loader:
        # Should handle raw arrays
        assert X_batch.shape[0] <= 3, "Batch size should be <= 3"
        break
    print("  ✅ DataLoader with raw arrays")

    # Test 3: DataLoader with mixed types (NumType X, scalar y)
    X = nm.randn(15, 4)
    y = nm.np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # raw scalars

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=5)

    for X_batch, y_batch in loader:
        assert X_batch.shape == (5, 4) or X_batch.shape == (15 % 5, 4), "X batch shape"
        break
    print("  ✅ DataLoader with mixed types")

    print("✅ All DataLoader edge cases passed!\n")
