import numpy as np
import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter
from lemon.nnlib.train_control import train


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
    xp = nm.get_array_module(x._data)

    if x.ndim == 2:
        # (N, C)
        axes = (0,)
    elif x.ndim == 3:
        # (N, C, L)
        axes = (0, 2)
    else:
        raise ValueError(
            f"Expected 2D or 3D input, got {x.ndim}D input with shape {x.shape}"
        )

    # Calculate statistics
    if training:
        # Calculate batch statistics
        mean = xp.mean(x._data, axis=axes, keepdims=True)
        var = xp.var(x._data, axis=axes, keepdims=True)

        # Update running statistics (in-place, outside of autograd)
        if running_mean is not None and running_var is not None:
            # Squeeze to get scalar values for each feature
            mean_scalar = xp.squeeze(mean)
            var_scalar = xp.squeeze(var)

            # Convert to numpy if needed
            if hasattr(mean_scalar, "_data"):
                mean_scalar = mean_scalar._data
            if hasattr(var_scalar, "_data"):
                var_scalar = var_scalar._data

            # Ensure proper shape (1D array)
            mean_scalar = np.atleast_1d(mean_scalar)
            var_scalar = np.atleast_1d(var_scalar)

            # Flatten if necessary
            if mean_scalar.ndim > 1:
                mean_scalar = mean_scalar.flatten()
            if var_scalar.ndim > 1:
                var_scalar = var_scalar.flatten()

            # Ensure correct size
            if len(mean_scalar) != len(running_mean):
                raise ValueError(
                    f"Shape mismatch: mean_scalar has {len(mean_scalar)} elements, "
                    f"but running_mean has {len(running_mean)} elements"
                )

            # Update running statistics
            running_mean[:] = (1 - momentum) * running_mean + momentum * mean_scalar
            running_var[:] = (1 - momentum) * running_var + momentum * var_scalar
    else:
        # Use running statistics
        if running_mean is not None and running_var is not None:
            # Convert numpy arrays to proper shape for broadcasting
            mean = np.array(running_mean)
            var = np.array(running_var)

            # Add dimensions for broadcasting
            if x.ndim == 2:
                mean = mean.reshape(1, -1)
                var = var.reshape(1, -1)
            elif x.ndim == 3:
                mean = mean.reshape(1, -1, 1)
                var = var.reshape(1, -1, 1)
        else:
            # Fallback to batch statistics
            mean = xp.mean(x._data, axis=axes, keepdims=True)
            var = xp.var(x._data, axis=axes, keepdims=True)

    # Normalize
    x_normalized_data = (x._data - mean) / xp.sqrt(var + eps)

    # Apply affine transformation if gamma/beta provided
    if gamma is not None and beta is not None:
        # Reshape gamma and beta for broadcasting
        if x.ndim == 2:
            gamma_data = gamma._data.reshape(1, -1)
            beta_data = beta._data.reshape(1, -1)
        elif x.ndim == 3:
            gamma_data = gamma._data.reshape(1, -1, 1)
            beta_data = beta._data.reshape(1, -1, 1)

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
    saved_axes = axes
    saved_has_gamma = gamma is not None
    saved_has_beta = beta is not None
    saved_eps = eps

    def _backward():
        if result.grad is None:
            return

        grad_output = result.grad._data

        # Gradient w.r.t. gamma
        if saved_has_gamma and gamma.requires_grad:
            grad_gamma = xp.sum(grad_output * saved_x_normalized, axis=saved_axes)
            grad_gamma_result = nm._create_result(grad_gamma)
            if gamma.grad is None:
                gamma.grad = grad_gamma_result
            else:
                gamma.grad._data += grad_gamma_result._data

        # Gradient w.r.t. beta
        if saved_has_beta and beta.requires_grad:
            grad_beta = xp.sum(grad_output, axis=saved_axes)
            grad_beta_result = nm._create_result(grad_beta)
            if beta.grad is None:
                beta.grad = grad_beta_result
            else:
                beta.grad._data += grad_beta_result._data

        # Gradient w.r.t. input x
        if x.requires_grad:
            if saved_has_gamma:
                if x.ndim == 2:
                    gamma_data = gamma._data.reshape(1, -1)
                elif x.ndim == 3:
                    gamma_data = gamma._data.reshape(1, -1, 1)
                grad_normalized = grad_output * gamma_data
            else:
                grad_normalized = grad_output

            # Backprop through normalization
            N = saved_x_shape[0]
            if x.ndim == 3:
                N = N * saved_x_shape[2]  # Total number of elements per feature

            std_inv = 1.0 / xp.sqrt(saved_var + saved_eps)

            # Gradients for variance and mean
            grad_var = xp.sum(
                grad_normalized * (x._data - saved_mean) * (-0.5) * (std_inv**3),
                axis=saved_axes,
                keepdims=True,
            )

            grad_mean = xp.sum(
                grad_normalized * (-std_inv), axis=saved_axes, keepdims=True
            ) + grad_var * xp.mean(
                -2.0 * (x._data - saved_mean), axis=saved_axes, keepdims=True
            )

            # Gradient for x
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
