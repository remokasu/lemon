"""
CIFAR-100 dataset

The CIFAR-100 dataset consists of 60000 32x32 color images in 100 fine classes and 20 superclasses.
"""

import os
import tarfile
import urllib.request
import pickle
import lemon.numlib as nm
from lemon.datasets.vision._base import _DownloadableDataset


class CIFAR100(_DownloadableDataset):
    """
    CIFAR-100 dataset loader

    32x32 color images in 100 classes, with 600 images per class.
    Also includes 20 superclasses (coarse labels).

    Parameters
    ----------
    root : str
        Root directory where dataset exists or will be downloaded (default: './data')
    train : bool, optional
        If True, creates dataset from training set, otherwise from test set (default: True)
    download : bool, optional
        If True, downloads the dataset if it doesn't exist (default: False)
    transform : callable, optional
        A function/transform to apply to the data (default: None)
    flatten : bool, optional
        If True, flattens images to 1D (default: True)

    Examples
    --------
    >>> from lemon.datasets.vision import CIFAR100
    >>> train_dataset = CIFAR100(root='./data', train=True, download=True)
    >>> test_dataset = CIFAR100(root='./data', train=False)
    >>> X, y = train_dataset[0]
    >>> print(X.shape)  # (3072,) or (3, 32, 32) depending on flatten parameter
    >>> print(y)  # Tuple: (fine_label, coarse_label)

    # Access both fine and coarse labels
    >>> X, (y_fine, y_coarse) = train_dataset[0]
    >>> print(y_fine)  # 0-99 (e.g., apple)
    >>> print(y_coarse)  # 0-19 (e.g., fruit_and_vegetables)

    Notes
    -----
    Fine labels (100 classes):
    - Each of 100 specific classes (e.g., apple, aquarium_fish, baby, etc.)

    Coarse labels (20 superclasses):
    - aquatic_mammals, fish, flowers, food_containers, fruit_and_vegetables,
      household_electrical_devices, household_furniture, insects,
      large_carnivores, large_man-made_outdoor_things,
      large_natural_outdoor_scenes, large_omnivores_and_herbivores,
      medium_mammals, non-insect_invertebrates, people, reptiles,
      small_mammals, trees, vehicles_1, vehicles_2

    The dataset returns (fine_label, coarse_label) as a tuple.
    """

    @property
    def urls(self):
        """CIFAR-100 download URL"""
        return [
            {"cifar100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"}
        ]

    @property
    def filenames(self):
        """Required filenames for CIFAR-100 dataset"""
        return ["cifar-100-python.tar.gz"]

    def __init__(
        self, root="./data", train=True, download=False, transform=None, flatten=False
    ):
        self.base_root = root
        self.root = os.path.join(root, "cifar-100-python")
        self.train = train
        self.transform = transform
        self.flatten = flatten

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        # Load data
        self.data, self.targets, self.coarse_targets = self._load_data()

    def _check_exists(self) -> bool:
        """Check if dataset files exist"""
        return os.path.exists(self.root) and os.path.isdir(self.root)

    def download(self):
        """Download and extract CIFAR-100 dataset"""
        if self._check_exists():
            return

        os.makedirs(self.base_root, exist_ok=True)

        filename = "cifar-100-python.tar.gz"
        filepath = os.path.join(self.base_root, filename)
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

        if not os.path.exists(filepath):
            print(f"Downloading CIFAR-100...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    with open(filepath, "wb") as f:
                        f.write(response.read())
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise RuntimeError(f"Failed to download CIFAR-100: {e}")

        # Extract tar file
        print("Extracting files...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=self.base_root)
        print("Download complete!")

    def _load_data(self):
        """Load CIFAR-100 data from pickle files"""

        xp = nm.get_array_module(nm.zeros(1)._data)

        if self.train:
            # Load training batch
            file_path = os.path.join(self.root, "train")
            with open(file_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
                data = xp.array(batch[b"data"])
                fine_labels = xp.array(batch[b"fine_labels"], dtype=xp.int64)
                coarse_labels = xp.array(batch[b"coarse_labels"], dtype=xp.int64)
        else:
            # Load test batch
            file_path = os.path.join(self.root, "test")
            with open(file_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
                data = xp.array(batch[b"data"])
                fine_labels = xp.array(batch[b"fine_labels"], dtype=xp.int64)
                coarse_labels = xp.array(batch[b"coarse_labels"], dtype=xp.int64)

        # Normalize to [0, 1]
        data = data.astype(xp.float32) / 255.0

        # Reshape if not flattening: (N, 3072) -> (N, 3, 32, 32)
        if not self.flatten:
            data = data.reshape(-1, 3, 32, 32)

        return data, fine_labels, coarse_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y_fine = self.targets[index]
        y_coarse = self.coarse_targets[index]

        if self.transform:
            x = self.transform(x)

        # Return both fine and coarse labels as tuple
        return x, (y_fine, y_coarse)
