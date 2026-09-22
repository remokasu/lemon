import lemon.numlib as nm
from lemon.nnlib.module import Module


def _sign_forward(x):
    xp = nm.get_array_module(x)
    output = xp.sign(x)
    # sign(0) = 0 in numpy; BNN convention uses +1
    output = xp.where(output == 0, xp.ones_like(output), output)
    return output, x


def _sign_backward(x, grad, needs_grad):
    # STE: pass gradient through, zeroed where |x| > 1
    xp = nm.get_array_module(x)
    mask = (xp.abs(x) <= 1.0).astype(x.dtype)
    return (grad * mask,)


_sign = nm.make_op(_sign_forward, _sign_backward)


def sign(x):
    """
    Sign activation with Straight-Through Estimator (STE)

    Forward : sign(x) = +1 if x >= 0 else -1
    Backward: gradient passes through unchanged where |x| <= 1 (STE clip)

    Parameters
    ----------
    x : Tensor
        Input tensor

    Returns
    -------
    Tensor
        Binary tensor with values in {-1, +1}

    Notes
    -----
    The Straight-Through Estimator treats the sign function as the identity
    during backpropagation (clipped to |x| <= 1), allowing gradients to flow
    through the non-differentiable sign operation.
    Used as the core building block of Binary Neural Networks (BNN).
    """
    return _sign(x)


class Sign(Module):
    """
    Sign activation module (Binary Neural Network activation)

    Forward : sign(x) = +1 if x >= 0 else -1
    Backward: Straight-Through Estimator (STE)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.BinaryLinear(784, 256),
    ...     nl.Sign(),
    ...     nl.BinaryLinear(256, 10),
    ... )
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sign(x)

    def __repr__(self):
        return "Sign()"
