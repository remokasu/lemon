import lemon.numlib as nm
from lemon.nnlib.module import Module


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
    xp = nm.get_array_module(x._data)

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

    a_data = x._data[tuple(idx_a)]
    b_data = x._data[tuple(idx_b)]
    sig_b = 1.0 / (1.0 + xp.exp(-b_data))
    output_data = a_data * sig_b
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
            grad = result.grad._data
            # dL/da = grad * sigmoid(b)
            grad_a = grad * sig_b
            # dL/db = grad * a * sigmoid(b) * (1 - sigmoid(b))
            grad_b = grad * a_data * sig_b * (1.0 - sig_b)

            grad_x = xp.concatenate([grad_a, grad_b], axis=dim)
            g = nm._create_result(grad_x)
            if x.grad is None:
                x.grad = g
            else:
                x.grad._data += g._data

    result._backward = _backward
    return result


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
