from lemon.nnlib import Module
import lemon.numlib as nm


def softplus(x):
    """
    Softplus activation function

    Softplus(x) = log(exp(x) + 1)

    ONNX operator: Softplus
    Reference: https://onnx.ai/onnx/operators/onnx__Softplus.html

    Parameters
    ----------
    x : Tensor
        Input tensor

    Returns
    -------
    Tensor
        Output tensor

    Examples
    --------
    >>> x = nm.tensor([-1.0, 0.0, 1.0])
    >>> y = softplus(x)
    >>> # y ≈ [0.313, 0.693, 1.313]

    Notes
    -----
    For numerical stability, uses the identity:
    softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    """
    # Numerical stability: avoid overflow for large x
    return nm.log(1 + nm.exp(-nm.abs(x))) + nm.maximum(x, 0)


class Softplus(Module):
    """
    Softplus activation module (ONNX-compliant)

    For direct usage, prefer softplus(x).

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.Softplus(),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return softplus(x)

    def __repr__(self):
        return "Softplus()"
