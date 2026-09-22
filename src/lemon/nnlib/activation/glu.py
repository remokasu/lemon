import lemon.numlib as nm
from lemon.nnlib.module import Module


def _glu_forward(x, dim):
    xp = nm.get_array_module(x)

    ndim = x.ndim
    if dim < 0:
        dim = ndim + dim

    size = x.shape[dim]
    if size % 2 != 0:
        raise ValueError(f"Size along dim={dim} must be even, got {size}")

    half = size // 2

    # Split along dim
    idx_a = [slice(None)] * ndim
    idx_b = [slice(None)] * ndim
    idx_a[dim] = slice(0, half)
    idx_b[dim] = slice(half, size)

    a = x[tuple(idx_a)]
    b = x[tuple(idx_b)]
    sig_b = 1.0 / (1.0 + xp.exp(-b))
    return a * sig_b, (a, sig_b, dim)


def _glu_backward(ctx, grad, needs_grad):
    a, sig_b, dim = ctx
    xp = nm.get_array_module(grad)
    # dL/da = grad * sigmoid(b)
    grad_a = grad * sig_b
    # dL/db = grad * a * sigmoid(b) * (1 - sigmoid(b))
    grad_b = grad * a * sig_b * (1.0 - sig_b)
    return (xp.concatenate([grad_a, grad_b], axis=dim),)


_glu = nm.make_op(_glu_forward, _glu_backward)


def glu(x, dim=-1):
    """
    GLU (Gated Linear Unit) activation function

    Splits x into two halves along `dim`, then:
        GLU(x) = x1 * sigmoid(x2)

    Parameters
    ----------
    x : Tensor
        Input tensor. Size along `dim` must be even.
    dim : int, optional
        Dimension to split along (default: -1)

    Returns
    -------
    Tensor
        Output tensor with half the size of x along `dim`

    Examples
    --------
    >>> x = nm.randn(2, 8)
    >>> y = glu(x)   # shape: (2, 4)

    Notes
    -----
    Used in gated CNNs and as the basis for SwiGLU (SiLU + GLU).
    """
    return _glu(x, dim=dim)


class Glu(Module):
    """
    GLU (Gated Linear Unit) activation module

    GLU(x) = x[:half] * sigmoid(x[half:])

    Splits the last dimension in half by default.

    Parameters
    ----------
    dim : int, optional
        Dimension to split along (default: -1)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 40),  # doubled for GLU
    ...     nl.Glu(),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return glu(x, dim=self.dim)

    def __repr__(self):
        return f"Glu(dim={self.dim})"
