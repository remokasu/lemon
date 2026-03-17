import math
import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


def _batched_matmul(x, y):
    """
    Batched matrix multiplication supporting N-dimensional tensors.
    Computes x @ y with autograd support.

    Parameters
    ----------
    x : Tensor  shape (..., m, k)
    y : Tensor  shape (..., k, n)

    Returns
    -------
    Tensor  shape (..., m, n)
    """
    xp = nm.get_array_module(x._data)
    output_data = xp.matmul(x._data, y._data)
    result = nm._create_result(output_data)

    if not nm.autograd.is_enabled() or not (x.requires_grad or y.requires_grad):
        result.requires_grad = False
        return result

    result.requires_grad = True
    result._prev = (x, y)

    def _backward():
        if result.grad is None:
            return
        grad = result.grad._data

        if x.requires_grad:
            # dL/dx = grad @ y^T
            grad_x = xp.matmul(grad, y._data.swapaxes(-1, -2))
            g = nm._create_result(grad_x)
            if x.grad is None:
                x.grad = g
            else:
                x.grad._data += g._data

        if y.requires_grad:
            # dL/dy = x^T @ grad
            grad_y = xp.matmul(x._data.swapaxes(-1, -2), grad)
            g = nm._create_result(grad_y)
            if y.grad is None:
                y.grad = g
            else:
                y.grad._data += g._data

    result._backward = _backward
    return result


class MultiHeadAttention(Module):
    """
    Multi-Head Self/Cross Attention

    Splits queries, keys, and values into multiple heads, applies
    scaled dot-product attention in parallel, then concatenates.

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_o

    Parameters
    ----------
    d_model : int
        Total embedding dimension
    num_heads : int
        Number of attention heads. d_model must be divisible by num_heads.
    dropout : float, optional
        Dropout on attention weights (default: 0.0)
    bias : bool, optional
        Whether to use bias in projections (default: True)

    Examples
    --------
    >>> attn = MultiHeadAttention(d_model=512, num_heads=8)
    >>> x = nm.randn(2, 10, 512)        # (batch, seq_len, d_model)
    >>> out = attn(x, x, x)             # self-attention
    >>> out = attn(q, k, v)             # cross-attention
    """

    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout

        scale = math.sqrt(2.0 / d_model)
        self.W_q = Parameter(nm.randn(d_model, d_model) * scale)
        self.W_k = Parameter(nm.randn(d_model, d_model) * scale)
        self.W_v = Parameter(nm.randn(d_model, d_model) * scale)
        self.W_o = Parameter(nm.randn(d_model, d_model) * scale)

        self.use_bias = bias
        if bias:
            self.b_q = Parameter(nm.zeros(d_model))
            self.b_k = Parameter(nm.zeros(d_model))
            self.b_v = Parameter(nm.zeros(d_model))
            self.b_o = Parameter(nm.zeros(d_model))
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = None

    def forward(self, query, key, value, mask=None):
        """
        Parameters
        ----------
        query : Tensor  shape (batch, seq_q, d_model)
        key   : Tensor  shape (batch, seq_k, d_model)
        value : Tensor  shape (batch, seq_k, d_model)
        mask  : Tensor, optional

        Returns
        -------
        Tensor  shape (batch, seq_q, d_model)
        """
        xp = nm.get_array_module(query._data)
        batch, seq_q, _ = query.shape
        seq_k = key.shape[1]

        # Linear projections: (batch, seq, d_model)
        Q = nm.matmul(query, self.W_q.data)
        K = nm.matmul(key, self.W_k.data)
        V = nm.matmul(value, self.W_v.data)

        if self.use_bias:
            Q = Q + self.b_q.data
            K = K + self.b_k.data
            V = V + self.b_v.data

        # Split heads: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        def split_heads(t, seq):
            return t.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        Q = split_heads(Q, seq_q)
        K = split_heads(K, seq_k)
        V = split_heads(V, seq_k)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        K_t = K.transpose(0, 1, 3, 2)  # (batch, heads, d_k, seq_k)
        scores = _batched_matmul(Q, K_t) / scale  # (batch, heads, seq_q, seq_k)

        if mask is not None:
            scores = nm.tensor(
                xp.where(
                    mask._data == 0, xp.full_like(scores._data, -1e9), scores._data
                )
            )

        from lemon.nnlib.activation.softmax import softmax

        attn_weights = softmax(scores, axis=-1)

        if self.dropout_p > 0.0 and nm.train.is_enabled():
            from lemon.nnlib.layer.dropout import dropout

            attn_weights = dropout(attn_weights, p=self.dropout_p)

        # (batch, heads, seq_q, d_k)
        context = _batched_matmul(attn_weights, V)

        # Merge heads: (batch, seq_q, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_q, self.d_model)

        # Output projection
        out = nm.matmul(context, self.W_o.data)
        if self.use_bias:
            out = out + self.b_o.data

        return out

    def __repr__(self):
        return (
            f"MultiHeadAttention(d_model={self.d_model}, num_heads={self.num_heads}, "
            f"dropout={self.dropout_p})"
        )
