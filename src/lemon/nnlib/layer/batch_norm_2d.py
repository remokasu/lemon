import numpy as np
import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter
from lemon.nnlib.train_control import train


def _batch_norm_2d_forward(x, gamma, beta, running_mean, running_var, training, momentum, eps):
    # 移動平均・移動分散が NumType なら、中身の配列を直接（その場で）更新する
    if isinstance(running_mean, nm.NumType):
        running_mean = running_mean._data
    if isinstance(running_var, nm.NumType):
        running_var = running_var._data

    if x.ndim != 4:
        raise ValueError(
            f"Expected 4D input (N, C, H, W), got {x.ndim}D input with shape {x.shape}"
        )

    xp = nm.get_array_module(x)

    # Calculate over batch, height, width dimensions (keep channel dimension)
    axes = (0, 2, 3)
    param_shape = (1, -1, 1, 1)

    has_running = running_mean is not None and running_var is not None
    if training or not has_running:
        # Batch statistics
        mean = xp.mean(x, axis=axes, keepdims=True)
        var = xp.var(x, axis=axes, keepdims=True)
        batch_stats = True

        # Update running statistics (in-place, outside of autograd)
        if training and has_running:
            running_mean[:] = (1 - momentum) * running_mean + momentum * mean.reshape(-1)
            running_var[:] = (1 - momentum) * running_var + momentum * var.reshape(-1)
    else:
        # Running statistics (constants)
        mean = xp.asarray(running_mean).reshape(param_shape)
        var = xp.asarray(running_var).reshape(param_shape)
        batch_stats = False

    x_centered = x - mean
    std_inv = 1.0 / xp.sqrt(var + eps)
    x_norm = x_centered * std_inv

    # gamma と beta は、それぞれ渡されたものだけを掛ける・足す
    gamma_b = gamma.reshape(param_shape) if gamma is not None else None
    output = x_norm
    if gamma_b is not None:
        output = output * gamma_b
    if beta is not None:
        output = output + beta.reshape(param_shape)

    return output, (x_centered, x_norm, std_inv, axes, gamma_b, batch_stats)


def _batch_norm_2d_backward(ctx, grad, needs_grad):
    x_centered, x_norm, std_inv, axes, gamma_b, batch_stats = ctx
    xp = nm.get_array_module(grad)

    grad_x = grad_gamma = grad_beta = None

    if needs_grad[1]:
        grad_gamma = xp.sum(grad * x_norm, axis=axes)
    if needs_grad[2]:
        grad_beta = xp.sum(grad, axis=axes)

    if needs_grad[0]:
        grad_normalized = grad * gamma_b if gamma_b is not None else grad
        if batch_stats:
            # 平均と分散もバッチ（x）から計算しているので、その分も微分する
            N = 1
            for a in axes:
                N *= grad.shape[a]
            grad_var = xp.sum(
                grad_normalized * x_centered * (-0.5) * (std_inv**3),
                axis=axes,
                keepdims=True,
            )
            grad_mean = xp.sum(
                grad_normalized * (-std_inv), axis=axes, keepdims=True
            ) + grad_var * xp.mean(-2.0 * x_centered, axis=axes, keepdims=True)
            grad_x = (
                grad_normalized * std_inv
                + grad_var * 2.0 * x_centered / N
                + grad_mean / N
            )
        else:
            # 移動平均・移動分散は定数なので、正規化は x の1次関数
            grad_x = grad_normalized * std_inv

    return grad_x, grad_gamma, grad_beta


_batch_norm_2d = nm.make_op(_batch_norm_2d_forward, _batch_norm_2d_backward)


def batch_norm_2d(
    x,
    gamma=None,
    beta=None,
    running_mean=None,
    running_var=None,
    training=True,
    momentum=0.1,
    eps=1e-5,
):
    """
    2D Batch Normalization (functional API with autograd support)

    Applies Batch Normalization over a 4D input.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (N, C, H, W)
    gamma : Tensor, optional
        Scale parameter of shape (C,)
    beta : Tensor, optional
        Shift parameter of shape (C,)
    running_mean : array, optional
        Running mean of shape (C,). Will be updated if training=True.
    running_var : array, optional
        Running variance of shape (C,). Will be updated if training=True.
    training : bool, optional
        Whether in training mode (default: True)
    momentum : float, optional
        Momentum for running statistics update (default: 0.1)
    eps : float, optional
        Value added for numerical stability (default: 1e-5)

    Returns
    -------
    Tensor
        Normalized tensor with same shape as input

    Examples
    --------
    >>> x = nm.randn(32, 64, 28, 28, requires_grad=True)
    >>> gamma = nm.ones(64, requires_grad=True)
    >>> beta = nm.zeros(64, requires_grad=True)
    >>> y = batch_norm_2d(x, gamma, beta, training=True)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    """
    return _batch_norm_2d(
        x,
        gamma,
        beta,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=eps,
    )


class BatchNorm2d(Module):
    """
    2D Batch Normalization layer

    [前のdocstringと同じ]
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.gamma = Parameter(nm.ones(num_features))
            self.beta = Parameter(nm.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

        if self.track_running_stats:
            self.running_mean = nm.zeros(num_features)
            self.running_var = nm.ones(num_features)
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, x):
        """Forward pass using functional API"""
        training = train.is_enabled()

        if self.track_running_stats and training:
            self.num_batches_tracked += 1

        return batch_norm_2d(
            x,
            gamma=self.gamma.data if self.affine else None,
            beta=self.beta.data if self.affine else None,
            running_mean=self.running_mean if self.track_running_stats else None,
            running_var=self.running_var if self.track_running_stats else None,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def __repr__(self):
        return (
            f"BatchNorm2d(num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats})"
        )
