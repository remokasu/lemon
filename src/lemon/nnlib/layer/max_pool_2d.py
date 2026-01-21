import lemon.numlib as nm
from lemon.nnlib.module import Module


def max_pool_2d(x, kernel_size, stride=None, padding=0):
    """
    2D Max pooling (functional API with autograd support)

    Applies a 2D max pooling over an input signal.
    This function supports autograd when used with numlib tensors.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (N, C, H, W)
    kernel_size : int or tuple
        Size of the pooling window
    stride : int or tuple, optional
        Stride (default: kernel_size)
    padding : int or tuple, optional
        Padding (default: 0)

    Returns
    -------
    Tensor
        Output tensor of shape (N, C, H_out, W_out)

    Examples
    --------
    >>> x = nm.randn(4, 16, 28, 28, requires_grad=True)
    >>> y = max_pool_2d(x, kernel_size=2, stride=2)
    >>> y.shape
    (4, 16, 14, 14)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    >>> # x.grad is computed

    Notes
    -----
    This implementation uses im2col + argmax for efficiency.
    The gradient only flows through the maximum value locations.
    """
    xp = nm.get_array_module(x._data)
    N, C, H, W = x.shape

    # Normalize parameters
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    kernel_h, kernel_w = kernel_size

    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding)

    # Calculate output dimensions
    out_h = (H + 2 * padding[0] - kernel_h) // stride[0] + 1
    out_w = (W + 2 * padding[1] - kernel_w) // stride[1] + 1

    # Use im2col to extract patches
    col = nm.im2col(x._data, kernel_h, kernel_w, stride=stride[0], padding=padding[0])
    col = col.reshape(N, C, kernel_h * kernel_w, out_h * out_w)

    # Take max along kernel dimension
    argmax = xp.argmax(col, axis=2)  # (N, C, out_h*out_w)

    # Get max values
    max_vals = xp.zeros((N, C, out_h * out_w), dtype=x._data.dtype)
    for n in range(N):
        for c in range(C):
            for pos in range(out_h * out_w):
                max_vals[n, c, pos] = col[n, c, argmax[n, c, pos], pos]

    # Reshape output
    output_data = max_vals.reshape(N, C, out_h, out_w)

    # Create result using _create_result
    result = nm._create_result(output_data)

    # Gradient computation
    if not nm.autograd.is_enabled() or not x.requires_grad:
        result.requires_grad = False
        return result

    result.requires_grad = True
    result._prev = (x,)

    # Save variables for backward
    saved_col_shape = col.shape
    saved_argmax = argmax
    saved_x_shape = x.shape
    saved_kernel_h = kernel_h
    saved_kernel_w = kernel_w
    saved_stride = stride
    saved_padding = padding

    def _backward():
        if result.grad is None:
            return

        grad_output = result.grad._data  # (N, C, out_h, out_w)
        grad_output_flat = grad_output.reshape(N, C, out_h * out_w)

        # Create gradient for col
        grad_col = xp.zeros(saved_col_shape, dtype=grad_output.dtype)

        # Distribute gradients only to max positions
        for n in range(N):
            for c in range(C):
                for pos in range(out_h * out_w):
                    max_idx = saved_argmax[n, c, pos]
                    grad_col[n, c, max_idx, pos] = grad_output_flat[n, c, pos]

        # Reshape grad_col for col2im
        grad_col = grad_col.reshape(N, C * kernel_h * kernel_w, out_h * out_w)

        # Use col2im to convert back to input gradient
        grad_x_data = nm.col2im(
            grad_col,
            saved_x_shape,
            saved_kernel_h,
            saved_kernel_w,
            stride=saved_stride[0],
            padding=saved_padding[0],
        )

        grad_x = nm._create_result(grad_x_data)
        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data += grad_x._data

    result._backward = _backward
    return result


class MaxPool2d(Module):
    """
    2D Max pooling layer

    Applies a 2D max pooling over an input signal.

    Parameters
    ----------
    kernel_size : int or tuple
        Size of the pooling window
    stride : int or tuple, optional
        Stride of the pooling window (default: kernel_size)
    padding : int or tuple, optional
        Zero-padding added to both sides (default: 0)

    Examples
    --------
    >>> import numlib as nm
    >>> import lemon as lm
    >>> pool = nl.MaxPool2d(kernel_size=2, stride=2)
    >>> x = nm.randn(32, 16, 28, 28)
    >>> y = pool(x)
    >>> y.shape
    (32, 16, 14, 14)

    >>> # With gradient computation
    >>> x = nm.randn(4, 3, 8, 8, requires_grad=True)
    >>> pool = nl.MaxPool2d(2, 2)
    >>> y = pool(x)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    >>> # x.grad is computed

    Notes
    -----
    Max pooling is commonly used in CNNs to downsample feature maps
    while preserving the most important features.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()

        # Normalize kernel_size to tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.kernel_h, self.kernel_w = kernel_size

        # Default stride = kernel_size
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        # Normalize padding to tuple
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

    def forward(self, x):
        """
        Forward pass of max pooling

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W)

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, H_out, W_out)
        """
        return max_pool_2d(x, self.kernel_size, self.stride, self.padding)

    def __repr__(self):
        return (
            f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding})"
        )
