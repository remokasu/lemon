import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


def embedding(x, weight, padding_idx=None):
    """
    Embedding lookup (functional API)

    Maps integer indices to dense vectors via a lookup table.

    Parameters
    ----------
    x : Tensor
        Integer index tensor of any shape
    weight : Tensor
        Embedding table of shape (num_embeddings, embedding_dim)
    padding_idx : int, optional
        If given, gradient at padding_idx is zeroed (default: None)

    Returns
    -------
    Tensor
        Embedded tensor of shape (*x.shape, embedding_dim)
    """
    xp = nm.get_array_module(weight._data)
    indices = x._data.astype(int)
    output_data = weight._data[indices]
    result = nm._create_result(output_data)

    if not nm.autograd.is_enabled() or not weight.requires_grad:
        result.requires_grad = False
        return result

    result.requires_grad = True
    result._prev = (weight,)

    def _backward():
        if result.grad is None:
            return
        if not weight.requires_grad:
            return

        grad_weight = xp.zeros_like(weight._data)
        xp.add.at(grad_weight, indices, result.grad._data)

        if padding_idx is not None:
            grad_weight[padding_idx] = 0.0

        g = nm._create_result(grad_weight)
        if weight.grad is None:
            weight.grad = g
        else:
            weight.grad._data += g._data

    result._backward = _backward
    return result


class Embedding(Module):
    """
    Embedding layer — integer index to dense vector lookup table

    Parameters
    ----------
    num_embeddings : int
        Size of the vocabulary (number of unique indices)
    embedding_dim : int
        Dimension of each embedding vector
    padding_idx : int, optional
        Index whose gradient is always zeroed (default: None)

    Examples
    --------
    >>> emb = Embedding(1000, 64)
    >>> x = nm.tensor([1, 5, 3, 2])   # integer indices
    >>> y = emb(x)                     # shape: (4, 64)
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Initialize with small random values
        self.weight = Parameter(nm.randn(num_embeddings, embedding_dim) * 0.01)

        if padding_idx is not None:
            self.weight.data._data[padding_idx] = 0.0

    def forward(self, x):
        return embedding(x, self.weight.data, padding_idx=self.padding_idx)

    def __repr__(self):
        return (
            f"Embedding({self.num_embeddings}, {self.embedding_dim}"
            + (
                f", padding_idx={self.padding_idx}"
                if self.padding_idx is not None
                else ""
            )
            + ")"
        )
