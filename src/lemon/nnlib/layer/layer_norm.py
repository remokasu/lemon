import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


def _layer_norm_forward(x, weight, bias, normalized_shape, eps):
    xp = nm.get_array_module(x)

    # Axes to normalize over (last len(normalized_shape) dims)
    ndim = x.ndim
    n_norm = len(normalized_shape)
    axes = tuple(range(ndim - n_norm, ndim))

    mean = xp.mean(x, axis=axes, keepdims=True)
    var = xp.var(x, axis=axes, keepdims=True)
    x_centered = x - mean
    std_inv = 1.0 / xp.sqrt(var + eps)
    x_norm = x_centered * std_inv

    # weight と bias は、それぞれ渡されたものだけを掛ける・足す
    output = x_norm
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias

    return output, (x_centered, x_norm, std_inv, axes, weight)


def _layer_norm_backward(ctx, grad, needs_grad):
    x_centered, x_norm, std_inv, axes, weight = ctx
    xp = nm.get_array_module(grad)
    batch_axes = tuple(range(grad.ndim - len(axes)))
    N = 1
    for a in axes:
        N *= grad.shape[a]

    grad_x = grad_w = grad_b = None

    if needs_grad[1]:
        grad_w = xp.sum(grad * x_norm, axis=batch_axes)
    if needs_grad[2]:
        grad_b = xp.sum(grad, axis=batch_axes)

    if needs_grad[0]:
        grad_norm = grad * weight if weight is not None else grad
        grad_var = xp.sum(
            grad_norm * x_centered * (-0.5) * (std_inv**3), axis=axes, keepdims=True
        )
        grad_mean = xp.sum(grad_norm * (-std_inv), axis=axes, keepdims=True) + grad_var * xp.mean(
            -2.0 * x_centered, axis=axes, keepdims=True
        )
        grad_x = grad_norm * std_inv + grad_var * 2.0 * x_centered / N + grad_mean / N

    return grad_x, grad_w, grad_b


_layer_norm = nm.make_op(_layer_norm_forward, _layer_norm_backward)


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
    return _layer_norm(x, weight, bias, normalized_shape=normalized_shape, eps=eps)


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
