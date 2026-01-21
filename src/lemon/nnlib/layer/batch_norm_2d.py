import numpy as np
import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter
from lemon.nnlib.train_control import train


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
    if x.ndim != 4:
        raise ValueError(
            f"Expected 4D input (N, C, H, W), got {x.ndim}D input with shape {x.shape}"
        )

    xp = nm.get_array_module(x._data)

    # Calculate over batch, height, width dimensions (keep channel dimension)
    axes = (0, 2, 3)

    # Calculate statistics
    if training:
        # Calculate batch statistics
        mean = nm.mean(x, axis=axes, keepdims=True)
        var = nm.var(x, axis=axes, keepdims=True)

        # Update running statistics
        if running_mean is not None and running_var is not None:
            mean_scalar = xp.squeeze(mean)
            var_scalar = xp.squeeze(var)

            running_mean[:] = (1 - momentum) * running_mean + momentum * mean_scalar
            running_var[:] = (1 - momentum) * running_var + momentum * var_scalar
    else:
        # Use running statistics
        if running_mean is not None and running_var is not None:
            mean = running_mean.reshape(1, -1, 1, 1)
            var = running_var.reshape(1, -1, 1, 1)
        else:
            # Fallback to batch statistics
            mean = xp.mean(x._data, axis=axes, keepdims=True)
            var = xp.var(x._data, axis=axes, keepdims=True)

    # Normalize
    x_normalized_data = (x._data - mean) / xp.sqrt(var + eps)

    # Apply affine transformation if gamma/beta provided
    if gamma is not None and beta is not None:
        gamma_data = gamma._data.reshape(1, -1, 1, 1)
        beta_data = beta._data.reshape(1, -1, 1, 1)
        output_data = x_normalized_data * gamma_data + beta_data
    else:
        output_data = x_normalized_data

    result = nm._create_result(output_data)

    # Setup autograd
    requires_grad_list = [x.requires_grad]
    if gamma is not None:
        requires_grad_list.append(gamma.requires_grad)
    if beta is not None:
        requires_grad_list.append(beta.requires_grad)

    if not nm.autograd.is_enabled() or not any(requires_grad_list):
        result.requires_grad = False
        return result

    result.requires_grad = True

    prev_list = [x]
    if gamma is not None:
        prev_list.append(gamma)
    if beta is not None:
        prev_list.append(beta)
    result._prev = tuple(prev_list)

    # Save for backward
    saved_x_normalized = x_normalized_data
    saved_mean = mean
    saved_var = var
    saved_x_shape = x.shape
    saved_has_gamma = gamma is not None
    saved_has_beta = beta is not None

    def _backward():
        if result.grad is None:
            return

        grad_output = result.grad._data

        # Gradient w.r.t. gamma
        if saved_has_gamma and gamma.requires_grad:
            grad_gamma = xp.sum(grad_output * saved_x_normalized, axis=axes)
            grad_gamma_result = nm._create_result(grad_gamma)
            if gamma.grad is None:
                gamma.grad = grad_gamma_result
            else:
                gamma.grad._data += grad_gamma_result._data

        # Gradient w.r.t. beta
        if saved_has_beta and beta.requires_grad:
            grad_beta = xp.sum(grad_output, axis=axes)
            grad_beta_result = nm._create_result(grad_beta)
            if beta.grad is None:
                beta.grad = grad_beta_result
            else:
                beta.grad._data += grad_beta_result._data

        # Gradient w.r.t. input x
        if x.requires_grad:
            if saved_has_gamma:
                gamma_data = gamma._data.reshape(1, -1, 1, 1)
                grad_normalized = grad_output * gamma_data
            else:
                grad_normalized = grad_output

            # Backprop through normalization
            N = saved_x_shape[0] * saved_x_shape[2] * saved_x_shape[3]

            std_inv = 1.0 / xp.sqrt(saved_var + eps)

            grad_var = xp.sum(
                grad_normalized * (x._data - saved_mean) * (-0.5) * (std_inv**3),
                axis=axes,
                keepdims=True,
            )
            grad_mean = xp.sum(
                grad_normalized * (-std_inv), axis=axes, keepdims=True
            ) + grad_var * xp.mean(
                -2.0 * (x._data - saved_mean), axis=axes, keepdims=True
            )

            grad_x_data = (
                grad_normalized * std_inv
                + grad_var * 2.0 * (x._data - saved_mean) / N
                + grad_mean / N
            )

            grad_x = nm._create_result(grad_x_data)
            if x.grad is None:
                x.grad = grad_x
            else:
                x.grad._data += grad_x._data

    result._backward = _backward
    return result


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
