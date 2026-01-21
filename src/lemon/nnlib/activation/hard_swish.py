from lemon.nnlib import Module
import lemon.numlib as nm


def hard_swish(x):
    """
    HardSwish activation function

    HardSwish(x) = x * max(0, min(1, (x + 3) / 6))

    ONNX operator: HardSwish
    Reference: https://onnx.ai/onnx/operators/onnx__HardSwish.html

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
    >>> x = nm.tensor([-3.0, 0.0, 3.0])
    >>> y = hard_swish(x)
    """
    return x * nm.maximum(0.0, nm.minimum(1.0, (x + 3) / 6))


class HardSwish(Module):
    """
    HardSwish activation module (ONNX-compliant)

    For direct usage, prefer hard_swish(x).

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.HardSwish(),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return hard_swish(x)

    def __repr__(self):
        return "HardSwish()"
