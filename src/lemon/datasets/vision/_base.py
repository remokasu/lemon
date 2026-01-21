"""
Base classes for vision datasets

Provides base classes for downloadable datasets and MNIST-format datasets.
"""

import os
import gzip
import urllib.request
import lemon.numlib as nm
from lemon.nnlib.data import Dataset


class _DownloadableDataset(Dataset):
    """
    Base class for datasets that can be downloaded from URLs

    Provides common download functionality with mirror support and error handling.
    Subclasses should override:
    - urls: Property returning list of mirror URL dictionaries
    - filenames: Property returning list of expected filenames
    - _load_data: Method to load the actual data files
    """

    @property
    def urls(self):
        """Return list of mirror URL dictionaries. Override in subclass."""
        raise NotImplementedError("Subclass must provide URLs")

    @property
    def filenames(self):
        """Return list of expected filenames. Override in subclass."""
        raise NotImplementedError("Subclass must provide filenames")

    def _check_exists(self):
        """Check if all required dataset files exist"""
        return all(
            os.path.exists(os.path.join(self.root, filename))
            for filename in self.filenames
        )

    def download(self):
        """Download dataset files with fallback mirrors"""
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        for mirror_idx, url_dict in enumerate(self.urls):
            print(f"Trying mirror {mirror_idx + 1}...")
            success = True

            for name, url in url_dict.items():
                filename = url.split("/")[-1]
                filepath = os.path.join(self.root, filename)

                if os.path.exists(filepath):
                    print(f"{filename} already exists, skipping...")
                    continue

                print(f"Downloading {filename}...")
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=30) as response:
                        with open(filepath, "wb") as f:
                            f.write(response.read())
                    print(f"Successfully downloaded {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    success = False
                    # Clean up partial downloads
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    break

            if success:
                print("Download complete!")
                return

        raise RuntimeError(
            f"Failed to download {self.__class__.__name__} from all mirrors"
        )


class _MNISTLikeDataset(_DownloadableDataset):
    """
    Base class for MNIST-format datasets (MNIST, Fashion-MNIST, etc.)

    These datasets use the IDX file format with gzip compression.
    Subclasses only need to provide URLs and filenames.
    """

    def _load_images(self, filepath: str, flatten: bool = False):
        """Load MNIST-format images from gzip file"""
        xp = nm.get_array_module(nm.zeros(1)._data)

        with gzip.open(filepath, "rb") as f:
            # Read header
            magic = int.from_bytes(f.read(4), "big")
            n_images = int.from_bytes(f.read(4), "big")
            n_rows = int.from_bytes(f.read(4), "big")
            n_cols = int.from_bytes(f.read(4), "big")

            # Read image data
            data = xp.frombuffer(f.read(), dtype=xp.uint8)

            # Reshape based on flatten parameter
            if flatten:
                data = data.reshape(n_images, n_rows * n_cols)
            else:
                # (N, 1, H, W) format for CNN
                data = data.reshape(n_images, 1, n_rows, n_cols)

            # Normalize to [0, 1]
            data = data.astype(xp.float32) / 255.0

            return data

    def _load_labels(self, filepath: str):
        """Load MNIST-format labels from gzip file"""
        xp = nm.get_array_module(nm.zeros(1)._data)

        with gzip.open(filepath, "rb") as f:
            # Read header
            magic = int.from_bytes(f.read(4), "big")
            n_labels = int.from_bytes(f.read(4), "big")

            # Read label data
            labels = xp.frombuffer(f.read(), dtype=xp.uint8)

            return labels.astype(xp.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y
