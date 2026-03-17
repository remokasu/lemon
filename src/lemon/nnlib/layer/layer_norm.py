import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """
    Layer Normalization (functional API)

    Normalizes over the last len(normalized_shape) dimensions.
    Unlike BatchNorm, statistics are computed per sample, not per batch.

    Parameters
    ----------
    x : Tensor
        Input tensor
    normalized_shape : tuple of int
        Shape of the dimensions to normalize over (last dims of x)
    weight : Tensor, optional
        Learnable scale parameter (gamma)
    bias : Tensor, optional
        Learnable shift parameter (beta)
    eps : float, optional
        Value added for numerical stability (default: 1e-5)

    Returns
    -------
    Tensor
        Normalized tensor with same shape as input
    """
    xp = nm.get_array_module(x._data)

    # Axes to normalize over (last len(normalized_shape) dims)
    ndim = x.ndim
    n_norm = len(normalized_shape)
    axes = tuple(range(ndim - n_norm, ndim))

    mean = xp.mean(x._data, axis=axes, keepdims=True)
    var = xp.var(x._data, axis=axes, keepdims=True)
    x_norm_data = (x._data - mean) / xp.sqrt(var + eps)

    if weight is not None and bias is not None:
        output_data = x_norm_data * weight._data + bias._data
    else:
        output_data = x_norm_data

    result = nm._create_result(output_data)

    requires_grad_list = [x.requires_grad]
    if weight is not None:
        requires_grad_list.append(weight.requires_grad)
    if bias is not None:
        requires_grad_list.append(bias.requires_grad)

    if not nm.autograd.is_enabled() or not any(requires_grad_list):
        result.requires_grad = False
        return result

    result.requires_grad = True
    prev_list = [x]
    if weight is not None:
        prev_list.append(weight)
    if bias is not None:
        prev_list.append(bias)
    result._prev = tuple(prev_list)

    saved_x_norm = x_norm_data
    saved_var = var
    saved_axes = axes
    saved_eps = eps
    saved_has_weight = weight is not None
    saved_has_bias = bias is not None

    def _backward():
        if result.grad is None:
            return

        grad_out = result.grad._data
        std_inv = 1.0 / xp.sqrt(saved_var + saved_eps)
        N = 1
        for a in saved_axes:
            N *= x.shape[a]

        # Gradient w.r.t. weight (gamma)
        if saved_has_weight and weight.requires_grad:
            grad_w = xp.sum(grad_out * saved_x_norm, axis=tuple(range(ndim - n_norm)))
            g = nm._create_result(grad_w)
            if weight.grad is None:
                weight.grad = g
            else:
                weight.grad._data += g._data

        # Gradient w.r.t. bias (beta)
        if saved_has_bias and bias.requires_grad:
            grad_b = xp.sum(grad_out, axis=tuple(range(ndim - n_norm)))
            g = nm._create_result(grad_b)
            if bias.grad is None:
                bias.grad = g
            else:
                bias.grad._data += g._data

        # Gradient w.r.t. input x
        if x.requires_grad:
            if saved_has_weight:
                grad_norm = grad_out * weight._data
            else:
                grad_norm = grad_out

            grad_var = xp.sum(
                grad_norm
                * (x._data - xp.mean(x._data, axis=saved_axes, keepdims=True))
                * (-0.5)
                * (std_inv**3),
                axis=saved_axes,
                keepdims=True,
            )
            grad_mean = xp.sum(
                grad_norm * (-std_inv), axis=saved_axes, keepdims=True
            ) + grad_var * xp.mean(
                -2.0 * (x._data - xp.mean(x._data, axis=saved_axes, keepdims=True)),
                axis=saved_axes,
                keepdims=True,
            )
            grad_x_data = (
                grad_norm * std_inv
                + grad_var
                * 2.0
                * (x._data - xp.mean(x._data, axis=saved_axes, keepdims=True))
                / N
                + grad_mean / N
            )

            g = nm._create_result(grad_x_data)
            if x.grad is None:
                x.grad = g
            else:
                x.grad._data += g._data

    result._backward = _backward
    return result


class LayerNorm(Module):
    """
    Layer Normalization

    Normalizes over the last len(normalized_shape) dimensions of the input.
    Commonly used in Transformers and RNNs.

    Parameters
    ----------
    normalized_shape : int or tuple of int
        Shape of the dimensions to normalize over
    eps : float, optional
        Value added for numerical stability (default: 1e-5)
    elementwise_affine : bool, optional
        If True, adds learnable weight and bias (default: True)

    Examples
    --------
    >>> layer = LayerNorm(512)
    >>> x = nm.randn(32, 10, 512)
    >>> y = layer(x)  # normalizes over last dim (512)
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(nm.ones(*self.normalized_shape))
            self.bias = Parameter(nm.zeros(*self.normalized_shape))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        return layer_norm(
            x,
            self.normalized_shape,
            weight=self.weight.data if self.elementwise_affine else None,
            bias=self.bias.data if self.elementwise_affine else None,
            eps=self.eps,
        )

    def __repr__(self):
        return (
            f"LayerNorm({self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine})"
        )
