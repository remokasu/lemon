import lemon.numlib as nm
from lemon.nnlib.module import Module


def silu(x):
    """
    SiLU (Sigmoid Linear Unit) / Swish activation function

    SiLU(x) = x * sigmoid(x)

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
    >>> x = nm.tensor([-2.0, 0.0, 2.0])
    >>> y = silu(x)
    >>> # y ≈ [-0.238, 0.0, 1.762]

    Notes
    -----
    Used in modern architectures like LLaMA, PaLM, and Stable Diffusion.
    Smooth, non-monotonic, and self-gated.
    """
    xp = nm.get_array_module(x._data)
    sig = 1.0 / (1.0 + xp.exp(-x._data))
    output_data = x._data * sig
    result = nm._create_result(output_data)

    if not nm.autograd.is_enabled() or not x.requires_grad:
        result.requires_grad = False
        return result

    result.requires_grad = True
    result._prev = (x,)

    def _backward():
        if result.grad is None:
            return
        if x.requires_grad:
            # d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            #                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            grad = result.grad._data * sig * (1.0 + x._data * (1.0 - sig))
            g = nm._create_result(grad)
            if x.grad is None:
                x.grad = g
            else:
                x.grad._data += g._data

    result._backward = _backward
    return result


class Silu(Module):
    """
    SiLU (Sigmoid Linear Unit) activation module

    SiLU(x) = x * sigmoid(x)

    Also known as Swish. Used in LLaMA, PaLM, etc.

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.Silu(),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return silu(x)

    def __repr__(self):
        return "Silu()"
