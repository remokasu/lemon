import lemon.numlib as nm
from lemon.nnlib.module import Module


def _silu_forward(x):
    xp = nm.get_array_module(x)
    sig = 1.0 / (1.0 + xp.exp(-x))
    return x * sig, (x, sig)


def _silu_backward(ctx, grad, needs_grad):
    x, sig = ctx
    # d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    return (grad * sig * (1.0 + x * (1.0 - sig)),)


_silu = nm.make_op(_silu_forward, _silu_backward)


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
    return _silu(x)


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
