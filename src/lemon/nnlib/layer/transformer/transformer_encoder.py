import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.sequential import Sequential
from lemon.nnlib.layer.linear import Linear
from lemon.nnlib.layer.layer_norm import LayerNorm
from lemon.nnlib.layer.dropout import dropout
from lemon.nnlib.layer.transformer.multi_head_attention import MultiHeadAttention
from lemon.nnlib.activation.relu import relu
from lemon.nnlib.activation.gelu import gelu


class TransformerEncoderLayer(Module):
    """
    Single Transformer Encoder Layer

    Applies multi-head self-attention followed by a position-wise
    feed-forward network, with residual connections and layer normalization.

        x = LayerNorm(x + MHA(x, x, x))
        x = LayerNorm(x + FFN(x))

    Parameters
    ----------
    d_model : int
        Embedding dimension
    num_heads : int
        Number of attention heads
    d_ff : int, optional
        Hidden dimension of feed-forward network (default: 4 * d_model)
    dropout : float, optional
        Dropout probability (default: 0.0)
    activation : str, optional
        Activation function: 'relu' or 'gelu' (default: 'relu')
    norm_first : bool, optional
        If True, apply LayerNorm before attention (Pre-LN).
        If False, apply after (Post-LN, original Transformer). (default: False)

    Examples
    --------
    >>> layer = TransformerEncoderLayer(d_model=512, num_heads=8)
    >>> x = nm.randn(2, 10, 512)
    >>> out = layer(x)  # shape: (2, 10, 512)
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff=None,
        dropout=0.0,
        activation="relu",
        norm_first=False,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout_p = dropout
        self.norm_first = norm_first

        if activation == "relu":
            self._activation = relu
        elif activation == "gelu":
            self._activation = gelu
        else:
            raise ValueError(f"activation must be 'relu' or 'gelu', got '{activation}'")
        self.activation = activation

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : Tensor  shape (batch, seq, d_model)
        mask : Tensor, optional  attention mask

        Returns
        -------
        Tensor  shape (batch, seq, d_model)
        """
        if self.norm_first:
            # Pre-LN (more stable training)
            x = x + self._sa_block(self.norm1(x), mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-LN (original Transformer)
            x = self.norm1(x + self._sa_block(x, mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, mask):
        out = self.self_attn(x, x, x, mask=mask)
        if self.dropout_p > 0.0 and nm.train.is_enabled():
            out = dropout(out, p=self.dropout_p)
        return out

    def _ff_block(self, x):
        out = self._activation(self.ff1(x))
        if self.dropout_p > 0.0 and nm.train.is_enabled():
            out = dropout(out, p=self.dropout_p)
        out = self.ff2(out)
        if self.dropout_p > 0.0 and nm.train.is_enabled():
            out = dropout(out, p=self.dropout_p)
        return out

    def __repr__(self):
        return (
            f"TransformerEncoderLayer(d_model={self.self_attn.d_model}, "
            f"num_heads={self.self_attn.num_heads}, "
            f"activation='{self.activation}', norm_first={self.norm_first})"
        )


class TransformerEncoder(Module):
    """
    Stack of TransformerEncoderLayers

    Parameters
    ----------
    encoder_layer : TransformerEncoderLayer
        A single encoder layer instance (used as template)
    num_layers : int
        Number of encoder layers to stack
    norm : Module, optional
        Final normalization layer (default: None)

    Examples
    --------
    >>> layer = TransformerEncoderLayer(d_model=512, num_heads=8)
    >>> encoder = TransformerEncoder(layer, num_layers=6)
    >>> x = nm.randn(2, 10, 512)
    >>> out = encoder(x)  # shape: (2, 10, 512)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        import copy

        self.layers = [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : Tensor  shape (batch, seq, d_model)
        mask : Tensor, optional

        Returns
        -------
        Tensor  shape (batch, seq, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        if self.norm is not None:
            params.extend(self.norm.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self):
        return (
            f"TransformerEncoder(num_layers={self.num_layers}, layer={self.layers[0]})"
        )
