import math
import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


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
            Q = Q + nm.broadcast_to(self.b_q.data, Q.shape)
            K = K + nm.broadcast_to(self.b_k.data, K.shape)
            V = V + nm.broadcast_to(self.b_v.data, V.shape)

        # Split heads: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        def split_heads(t, seq):
            return t.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        Q = split_heads(Q, seq_q)
        K = split_heads(K, seq_k)
        V = split_heads(V, seq_k)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        K_t = K.transpose(0, 1, 3, 2)  # (batch, heads, d_k, seq_k)
        scores = nm.matmul(Q, K_t) / scale  # (batch, heads, seq_q, seq_k)

        if mask is not None:
            # 計算グラフを切らないように、微分できる nm.where でマスクする
            # （新しい Tensor を作り直すと、Q と K に勾配が流れなくなる）
            mask_data = mask._data if isinstance(mask, nm.NumType) else mask
            masked = xp.broadcast_to(mask_data == 0, scores.shape)
            scores = nm.where(masked, -1e9, scores)

        from lemon.nnlib.activation.softmax import softmax

        attn_weights = softmax(scores, axis=-1)

        if self.dropout_p > 0.0 and nm.train.is_enabled():
            from lemon.nnlib.layer.dropout import dropout

            attn_weights = dropout(attn_weights, p=self.dropout_p)

        # (batch, heads, seq_q, d_k)
        context = nm.matmul(attn_weights, V)

        # Merge heads: (batch, seq_q, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_q, self.d_model)

        # Output projection
        out = nm.matmul(context, self.W_o.data)
        if self.use_bias:
            out = out + nm.broadcast_to(self.b_o.data, out.shape)

        return out

    def __repr__(self):
        return (
            f"MultiHeadAttention(d_model={self.d_model}, num_heads={self.num_heads}, "
            f"dropout={self.dropout_p})"
        )
