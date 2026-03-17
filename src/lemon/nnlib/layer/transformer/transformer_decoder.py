import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.layer.linear import Linear
from lemon.nnlib.layer.layer_norm import LayerNorm
from lemon.nnlib.layer.dropout import dropout
from lemon.nnlib.layer.transformer.multi_head_attention import MultiHeadAttention
from lemon.nnlib.activation.relu import relu
from lemon.nnlib.activation.gelu import gelu


class TransformerDecoderLayer(Module):
    """
    Single Transformer Decoder Layer

    Applies masked multi-head self-attention, cross-attention over encoder
    output, and a position-wise feed-forward network, with residual
    connections and layer normalization.

        x = LayerNorm(x + MaskedMHA(x, x, x))
        x = LayerNorm(x + CrossMHA(x, memory, memory))
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
        If True, apply LayerNorm before attention (Pre-LN). (default: False)

    Examples
    --------
    >>> layer = TransformerDecoderLayer(d_model=512, num_heads=8)
    >>> tgt = nm.randn(2, 10, 512)
    >>> memory = nm.randn(2, 20, 512)
    >>> out = layer(tgt, memory)  # shape: (2, 10, 512)
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
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout_p = dropout
        self.norm_first = norm_first

        if activation == "relu":
            self._activation = relu
        elif activation == "gelu":
            self._activation = gelu
        else:
            raise ValueError(f"activation must be 'relu' or 'gelu', got '{activation}'")
        self.activation = activation

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Parameters
        ----------
        tgt : Tensor  shape (batch, tgt_seq, d_model)
        memory : Tensor  shape (batch, src_seq, d_model)
        tgt_mask : Tensor, optional  causal mask for target self-attention
        memory_mask : Tensor, optional  mask for cross-attention

        Returns
        -------
        Tensor  shape (batch, tgt_seq, d_model)
        """
        if self.norm_first:
            # Pre-LN
            tgt = tgt + self._sa_block(self.norm1(tgt), tgt_mask)
            tgt = tgt + self._ca_block(self.norm2(tgt), memory, memory_mask)
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            # Post-LN (original Transformer)
            tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask))
            tgt = self.norm2(tgt + self._ca_block(tgt, memory, memory_mask))
            tgt = self.norm3(tgt + self._ff_block(tgt))
        return tgt

    def _sa_block(self, x, mask):
        out = self.self_attn(x, x, x, mask=mask)
        if self.dropout_p > 0.0 and nm.train.is_enabled():
            out = dropout(out, p=self.dropout_p)
        return out

    def _ca_block(self, x, memory, mask):
        out = self.cross_attn(x, memory, memory, mask=mask)
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
            f"TransformerDecoderLayer(d_model={self.self_attn.d_model}, "
            f"num_heads={self.self_attn.num_heads}, "
            f"activation='{self.activation}', norm_first={self.norm_first})"
        )


class TransformerDecoder(Module):
    """
    Stack of TransformerDecoderLayers

    Parameters
    ----------
    decoder_layer : TransformerDecoderLayer
        A single decoder layer instance (used as template)
    num_layers : int
        Number of decoder layers to stack
    norm : Module, optional
        Final normalization layer (default: None)

    Examples
    --------
    >>> layer = TransformerDecoderLayer(d_model=512, num_heads=8)
    >>> decoder = TransformerDecoder(layer, num_layers=6)
    >>> tgt = nm.randn(2, 10, 512)
    >>> memory = nm.randn(2, 20, 512)
    >>> out = decoder(tgt, memory)  # shape: (2, 10, 512)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        import copy

        self.layers = [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Parameters
        ----------
        tgt : Tensor  shape (batch, tgt_seq, d_model)
        memory : Tensor  shape (batch, src_seq, d_model)
        tgt_mask : Tensor, optional
        memory_mask : Tensor, optional

        Returns
        -------
        Tensor  shape (batch, tgt_seq, d_model)
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        if self.norm is not None:
            tgt = self.norm(tgt)
        return tgt

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
            f"TransformerDecoder(num_layers={self.num_layers}, layer={self.layers[0]})"
        )
