"""
AutoencoderDataset - Dataset wrapper for autoencoder training
"""

from lemon.nnlib.data.dataset import Dataset


class AutoencoderDataset(Dataset):
    """
    Dataset wrapper for autoencoder training.

    Converts (input, label) pairs to (input, input) pairs,
    since autoencoders learn to reconstruct their input.

    Args:
        dataset: Source dataset that returns (data, label) pairs

    Example:
        >>> from lemon.nnlib.data import AutoencoderDataset
        >>> import lemon as lm
        >>>
        >>> # Wrap MNIST dataset
        >>> mnist = lm.datasets.MNIST(root="./data", train=True)
        >>> ae_dataset = AutoencoderDataset(mnist)
        >>>
        >>> # Now returns (image, image) instead of (image, label)
        >>> x, y = ae_dataset[0]
        >>> assert (x == y).all()  # x and y are the same
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x, x  # Return (input, input) for autoencoder training
