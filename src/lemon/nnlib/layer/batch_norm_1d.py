import numpy as np
import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter
from lemon.nnlib.train_control import train


def _batch_norm_1d_forward(x, gamma, beta, running_mean, running_var, training, momentum, eps):
    # 移動平均・移動分散が NumType なら、中身の配列を直接（その場で）更新する
    if isinstance(running_mean, nm.NumType):
        running_mean = running_mean._data
    if isinstance(running_var, nm.NumType):
        running_var = running_var._data

    xp = nm.get_array_module(x)

    if x.ndim == 2:
        # (N, C)
        axes = (0,)
        param_shape = (1, -1)
    elif x.ndim == 3:
        # (N, C, L)
        axes = (0, 2)
        param_shape = (1, -1, 1)
    else:
        raise ValueError(
            f"Expected 2D or 3D input, got {x.ndim}D input with shape {x.shape}"
        )

    has_running = running_mean is not None and running_var is not None
    if training or not has_running:
        # Batch statistics
        mean = xp.mean(x, axis=axes, keepdims=True)
        var = xp.var(x, axis=axes, keepdims=True)
        batch_stats = True

        # Update running statistics (in-place, outside of autograd)
        if training and has_running:
            mean_flat = mean.reshape(-1)
            var_flat = var.reshape(-1)
            if len(mean_flat) != len(running_mean):
                raise ValueError(
                    f"Shape mismatch: mean has {len(mean_flat)} elements, "
                    f"but running_mean has {len(running_mean)} elements"
                )
            running_mean[:] = (1 - momentum) * running_mean + momentum * mean_flat
            running_var[:] = (1 - momentum) * running_var + momentum * var_flat
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


def _batch_norm_1d_backward(ctx, grad, needs_grad):
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


_batch_norm_1d = nm.make_op(_batch_norm_1d_forward, _batch_norm_1d_backward)


def batch_norm_1d(
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
    1D Batch Normalization (functional API with autograd support)

    Applies Batch Normalization over a 2D or 3D input.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (N, C) or (N, C, L)
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
    >>> x = nm.randn(32, 100, requires_grad=True)
    >>> gamma = nm.ones(100, requires_grad=True)
    >>> beta = nm.zeros(100, requires_grad=True)
    >>> y = batch_norm_1d(x, gamma, beta, training=True)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    """
    return _batch_norm_1d(
        x,
        gamma,
        beta,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=eps,
    )


class BatchNorm1d(Module):
    """
    1D Batch Normalization layer (without cache)
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
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, x):
        """Forward pass using functional API"""

        # Validation
        if x.ndim not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

        C = x.shape[1]
        if C != self.num_features:
            raise ValueError(
                f"Expected input with {self.num_features} features, got {C}"
            )

        training = train.is_enabled()

        # Handle eval mode without tracking
        if not training and not self.track_running_stats:
            raise RuntimeError(
                "BatchNorm1d was created with track_running_stats=False "
                "but is being used in evaluation mode"
            )

        # Update num_batches
        if self.track_running_stats and training:
            if self.num_batches_tracked is None:
                self.num_batches_tracked = 0
            self.num_batches_tracked += 1

        # 関数版を呼び出し
        return batch_norm_1d(
            x,
            gamma=self.gamma.data if self.affine else None,
            beta=self.beta.data if self.affine else None,
            running_mean=self.running_mean if self.track_running_stats else None,
            running_var=self.running_var if self.track_running_stats else None,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def parameters(self):
        """Return list of learnable parameters"""
        params = []
        if self.affine:
            params.append(self.gamma)
            params.append(self.beta)
        return params

    def zero_grad(self):
        """Zero out gradients of parameters"""
        for param in self.parameters():
            if hasattr(param, "zero_grad"):
                param.zero_grad()
            elif hasattr(param, "grad"):
                param.grad = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return (
            f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats})"
        )
