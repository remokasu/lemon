import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


def _embedding_forward(weight, indices, padding_idx):
    return weight[indices], (weight.shape, indices, padding_idx)


def _embedding_backward(ctx, grad, needs_grad):
    weight_shape, indices, padding_idx = ctx
    xp = nm.get_array_module(grad)
    # 同じインデックスが何度も出てくるので、勾配は足し合わせる
    grad_weight = xp.zeros(weight_shape, dtype=grad.dtype)
    xp.add.at(grad_weight, indices, grad)
    if padding_idx is not None:
        grad_weight[padding_idx] = 0.0
    return (grad_weight,)


_embedding = nm.make_op(_embedding_forward, _embedding_backward)


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
    # インデックスは微分しないので、演算の入力ではなくパラメータとして渡す
    indices = x._data if isinstance(x, nm.NumType) else x
    indices = nm.get_array_module(weight._data).asarray(indices).astype(int)
    return _embedding(weight, indices=indices, padding_idx=padding_idx)


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
