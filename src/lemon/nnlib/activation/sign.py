import lemon.numlib as nm
from lemon.nnlib.module import Module


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
    xp = nm.get_array_module(x._data)
    output_data = xp.sign(x._data)
    # sign(0) = 0 in numpy; BNN convention uses +1
    output_data = xp.where(output_data == 0, xp.ones_like(output_data), output_data)
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
            # STE: pass gradient through, zeroed where |x| > 1
            mask = (xp.abs(x._data) <= 1.0).astype(x._data.dtype)
            grad = result.grad._data * mask
            g = nm._create_result(grad)
            if x.grad is None:
                x.grad = g
            else:
                x.grad._data += g._data

    result._backward = _backward
    return result


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
