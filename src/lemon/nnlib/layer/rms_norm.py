import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


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
    xp = nm.get_array_module(x._data)

    ms = xp.mean(x._data ** 2, axis=-1, keepdims=True)
    rms_inv = 1.0 / xp.sqrt(ms + eps)
    x_norm_data = x._data * rms_inv

    if weight is not None:
        output_data = x_norm_data * weight._data
    else:
        output_data = x_norm_data

    result = nm._create_result(output_data)

    requires_grad_list = [x.requires_grad]
    if weight is not None:
        requires_grad_list.append(weight.requires_grad)

    if not nm.autograd.is_enabled() or not any(requires_grad_list):
        result.requires_grad = False
        return result

    result.requires_grad = True
    prev_list = [x]
    if weight is not None:
        prev_list.append(weight)
    result._prev = tuple(prev_list)

    saved_x_norm = x_norm_data
    saved_rms_inv = rms_inv
    saved_has_weight = weight is not None

    def _backward():
        if result.grad is None:
            return

        grad_out = result.grad._data
        N = x.shape[-1]

        # Gradient w.r.t. weight
        if saved_has_weight and weight.requires_grad:
            grad_w = xp.sum(grad_out * saved_x_norm,
                            axis=tuple(range(x.ndim - 1)))
            g = nm._create_result(grad_w)
            if weight.grad is None:
                weight.grad = g
            else:
                weight.grad._data += g._data

        # Gradient w.r.t. input x
        if x.requires_grad:
            if saved_has_weight:
                grad_norm = grad_out * weight._data
            else:
                grad_norm = grad_out

            # d/dx [x * rms_inv] = rms_inv - x^2 * rms_inv^3 / N
            grad_x = saved_rms_inv * (
                grad_norm
                - x._data * saved_rms_inv ** 2
                * xp.sum(grad_norm * x._data, axis=-1, keepdims=True) / N
            )
            g = nm._create_result(grad_x)
            if x.grad is None:
                x.grad = g
            else:
                x.grad._data += g._data

    result._backward = _backward
    return result


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
