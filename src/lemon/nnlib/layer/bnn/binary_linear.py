import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter
from lemon.nnlib.activation.sign import sign


def binary_linear(x, weight, bias=None):
    """
    Binary linear transformation

    Binarizes weights with sign function (STE) then applies linear transform.

    Parameters
    ----------
    x : Tensor
        Input tensor (..., in_features)
    weight : Tensor
        Real-valued weight matrix (in_features, out_features)
    bias : Tensor, optional
        Bias vector (out_features,)

    Returns
    -------
    Tensor
        Output tensor (..., out_features)
    """
    w_b = sign(weight)
    output = x @ w_b
    if bias is not None:
        output = output + bias
    return output


class BinaryLinear(Module):
    """
    Binary Linear layer (BNN)

    Stores real-valued weights for gradient accumulation.
    During forward pass, weights are binarized to {-1, +1} via sign function.
    Gradients flow back through the Straight-Through Estimator.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool, optional
        If True, adds a learnable (real-valued) bias (default: True)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.BinaryLinear(784, 256),
    ...     nl.Sign(),
    ...     nl.BinaryLinear(256, 10),
    ... )
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Real-valued weights initialized near 0 for balanced binarization
        limit = (6.0 / (in_features + out_features)) ** 0.5
        self.weight = Parameter(
            nm.rand(in_features, out_features, low=-limit, high=limit)
        )

        if bias:
            self.bias = Parameter(nm.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        return binary_linear(
            x,
            self.weight.data,
            self.bias.data if self.bias is not None else None,
        )

    def __repr__(self):
        return (
            f"BinaryLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )
