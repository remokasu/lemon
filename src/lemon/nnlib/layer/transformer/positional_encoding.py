import math
import lemon.numlib as nm
from lemon.nnlib.module import Module


class PositionalEncoding(Module):
    """
    Sinusoidal Positional Encoding

    Adds position information to token embeddings using sine/cosine functions.
    No learnable parameters.

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Parameters
    ----------
    d_model : int
        Embedding dimension
    max_len : int, optional
        Maximum sequence length (default: 5000)
    dropout : float, optional
        Dropout probability (default: 0.0)

    Examples
    --------
    >>> pe = PositionalEncoding(d_model=512, max_len=100)
    >>> x = nm.randn(8, 20, 512)   # (batch, seq_len, d_model)
    >>> y = pe(x)                   # same shape, position info added
    """

    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = dropout

        xp = nm.np
        pe = xp.zeros((max_len, d_model), dtype=xp.float32)
        position = xp.arange(0, max_len, dtype=xp.float32).reshape(-1, 1)
        div_term = xp.exp(
            xp.arange(0, d_model, 2, dtype=xp.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = xp.sin(position * div_term)
        pe[:, 1::2] = xp.cos(position * div_term[: d_model // 2])

        # (1, max_len, d_model) — broadcastable over batch
        self._pe = pe[xp.newaxis, :, :]

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape (batch, seq_len, d_model)
        """
        xp = nm.get_array_module(x._data)
        seq_len = x.shape[1]
        pe = nm.tensor(xp.asarray(self._pe[:, :seq_len, :]))
        out = x + pe

        if self.dropout_p > 0.0 and nm.train.is_enabled():
            from lemon.nnlib.layer.dropout import dropout

            out = dropout(out, p=self.dropout_p)

        return out

    def __repr__(self):
        return f"PositionalEncoding(d_model={self.d_model}, dropout={self.dropout_p})"
