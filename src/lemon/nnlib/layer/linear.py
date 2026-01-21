from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter

import lemon.numlib as nm


def linear(x, weight, bias=None):
    """
    Linear transformation (fully connected layer)

    y = x @ weight + bias

    Parameters
    ----------
    x : Tensor
        Input tensor (..., in_features)
    weight : Tensor
        Weight matrix (in_features, out_features)
    bias : Tensor, optional
        Bias vector (out_features,)

    Returns
    -------
    Tensor
        Output tensor (..., out_features)

    Examples
    --------
    >>> x = nm.randn(32, 784)
    >>> weight = nm.randn(784, 128)
    >>> bias = nm.randn(128)
    >>> y = nl.linear(x, weight, bias)
    >>> y.shape
    (32, 128)
    """
    output = x @ weight

    if bias is not None:
        output = output + bias

    return output


class Linear(Module):
    """
    Linear layer (fully connected layer)

    y = x @ W + b

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool, optional
        If True, adds a learnable bias (default: True)

    Attributes
    ----------
    weight : Parameter
        Weight matrix (in_features, out_features)
    bias : Parameter or None
        Bias vector (out_features,)

    Examples
    --------
    >>> import numlib as nm
    >>> layer = nl.Linear(784, 128)
    >>> x = nm.randn(32, 784)  # batch_size=32
    >>> y = layer(x)
    >>> y.shape
    (32, 128)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier uniform initialization
        # Range: [-sqrt(6 / (in + out)), sqrt(6 / (in + out))]
        limit = (6.0 / (in_features + out_features)) ** 0.5
        self.weight = Parameter(
            nm.rand(in_features, out_features, low=-limit, high=limit)
        )

        if bias:
            self.bias = Parameter(nm.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        return linear(
            x, self.weight.data, self.bias.data if self.bias is not None else None
        )

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
