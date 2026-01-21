"""
Dataset base classes

Provides base classes for creating custom datasets and supervised learning datasets.
"""

import lemon.numlib as nm


class Dataset:
    """
    Base class for all datasets

    Examples
    --------
    >>> class MyDataset(Dataset):
    ...     def __init__(self, data, labels):
    ...         self.data = data
    ...         self.labels = labels
    ...
    ...     def __len__(self):
    ...         return len(self.data)
    ...
    ...     def __getitem__(self, idx):
    ...         return self.data[idx], self.labels[idx]
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class SupervisedDataSet(Dataset):
    """
    PyBrain-style supervised dataset for adding samples one by one

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    target_dim : int
        Dimension of target values

    Examples
    --------
    >>> ds = SupervisedDataSet(2, 1)
    >>> ds.add_sample([0, 0], [0])
    >>> ds.add_sample([0, 1], [1])
    >>> ds.add_sample([1, 0], [1])
    >>> ds.add_sample([1, 1], [0])
    >>> print(len(ds))
    4
    >>> X, y = ds[0]
    >>> print(X.shape, y.shape)
    (2,) (1,)
    """

    def __init__(self, input_dim: int, target_dim: int):
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.inputs = []
        self.targets = []

    def add_sample(self, input_data, target_data):
        """
        Add a single sample to the dataset

        Parameters
        ----------
        input_data : array_like
            Input features (must match input_dim)
        target_data : array_like
            Target values (must match target_dim)
        """
        xp = nm.get_array_module(nm.zeros(1)._data)

        # Convert to arrays
        input_array = xp.array(input_data, dtype=xp.float32)
        target_array = xp.array(target_data, dtype=xp.float32)

        # Validate dimensions
        if input_array.shape[0] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {input_array.shape[0]}"
            )
        if target_array.shape[0] != self.target_dim:
            raise ValueError(
                f"Target dimension mismatch: expected {self.target_dim}, got {target_array.shape[0]}"
            )

        self.inputs.append(input_array)
        self.targets.append(target_array)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def get_data(self):
        """
        Get all data as stacked arrays

        Returns
        -------
        tuple
            (inputs, targets) where inputs is (n_samples, input_dim)
            and targets is (n_samples, target_dim)
        """
        xp = nm.get_array_module(nm.zeros(1)._data)
        if len(self.inputs) == 0:
            return xp.array([]), xp.array([])

        inputs = xp.stack(self.inputs)
        targets = xp.stack(self.targets)
        return inputs, targets
