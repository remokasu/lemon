"""
TensorDataset

Dataset wrapping tensors for easy batching and iteration.
"""

from lemon.nnlib.data.dataset import Dataset


class TensorDataset(Dataset):
    """
    Dataset wrapping tensors

    Parameters
    ----------
    *tensors : Tensor
        Tensors that have the same size of the first dimension

    Examples
    --------
    >>> import lemon.numlib as nm
    >>> X = nm.randn(100, 10)
    >>> y = nm.randint(100, low=0, high=2)
    >>> dataset = TensorDataset(X, y)
    >>> print(len(dataset))
    100
    >>> x, y = dataset[0]
    """

    def __init__(self, *tensors):
        if len(tensors) == 0:
            raise ValueError("At least one tensor required")

        # Check all tensors have same length
        size = tensors[0].shape[0] if hasattr(tensors[0], "shape") else len(tensors[0])
        for tensor in tensors[1:]:
            tensor_size = tensor.shape[0] if hasattr(tensor, "shape") else len(tensor)
            if tensor_size != size:
                raise ValueError(
                    "All tensors must have the same size in the first dimension"
                )

        self.tensors = tensors

    def __len__(self):
        return (
            self.tensors[0].shape[0]
            if hasattr(self.tensors[0], "shape")
            else len(self.tensors[0])
        )

    def __getitem__(self, idx):
        return tuple(
            tensor[idx] if hasattr(tensor, "__getitem__") else tensor
            for tensor in self.tensors
        )
