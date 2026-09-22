import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


def _rms_norm_forward(x, weight, eps):
    xp = nm.get_array_module(x)

    ms = xp.mean(x**2, axis=-1, keepdims=True)
    rms_inv = 1.0 / xp.sqrt(ms + eps)
    x_norm = x * rms_inv

    output = x_norm * weight if weight is not None else x_norm
    return output, (x, x_norm, rms_inv, weight)


def _rms_norm_backward(ctx, grad, needs_grad):
    x, x_norm, rms_inv, weight = ctx
    xp = nm.get_array_module(grad)
    N = x.shape[-1]

    grad_x = grad_w = None

    if needs_grad[1]:
        grad_w = xp.sum(grad * x_norm, axis=tuple(range(x.ndim - 1)))

    if needs_grad[0]:
        grad_norm = grad * weight if weight is not None else grad
        # d/dx [x * rms_inv] = rms_inv - x^2 * rms_inv^3 / N
        grad_x = rms_inv * (
            grad_norm - x * rms_inv**2 * xp.sum(grad_norm * x, axis=-1, keepdims=True) / N
        )

    return grad_x, grad_w


_rms_norm = nm.make_op(_rms_norm_forward, _rms_norm_backward)


def rms_norm(x, weight=None, eps=1e-8):
    """
    RMS Normalization (functional API)

    Normalizes by the root mean square of the last dimension.
    Unlike LayerNorm, does NOT subtract the mean.

        RMSNorm(x) = x / RMS(x) * weight
        RMS(x) = sqrt(mean(x^2) + eps)

    Parameters
    ----------
    x : Tensor
        Input tensor
    weight : Tensor, optional
        Learnable scale parameter (gamma), shape (d_model,)
    eps : float, optional
        Value added for numerical stability (default: 1e-8)

    Returns
    -------
    Tensor
        Normalized tensor with same shape as input
    """
    return _rms_norm(x, weight, eps=eps)


class RMSNorm(Module):
    """
    RMS Normalization

    Simpler and faster than LayerNorm: normalizes by root mean square
    without subtracting the mean. Used in LLaMA, Mistral, PaLM 2, etc.

        RMSNorm(x) = x / RMS(x) * weight
        RMS(x) = sqrt(mean(x^2) + eps)

    Parameters
    ----------
    d_model : int
        Size of the last dimension to normalize over
    eps : float, optional
        Numerical stability constant (default: 1e-8)
    elementwise_affine : bool, optional
        If True, adds learnable scale weight (default: True)

    Examples
    --------
    >>> norm = RMSNorm(512)
    >>> x = nm.randn(2, 10, 512)
    >>> y = norm(x)  # shape: (2, 10, 512)
    """

    def __init__(self, d_model, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Parameter(nm.ones(d_model))
        else:
            self.weight = None

    def forward(self, x):
        return rms_norm(
            x,
            weight=self.weight.data if self.elementwise_affine else None,
            eps=self.eps,
        )

    def __repr__(self):
        return (
            f"RMSNorm({self.d_model}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine})"
        )
