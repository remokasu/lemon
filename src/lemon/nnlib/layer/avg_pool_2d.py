from lemon.nnlib.module import Module
import lemon.numlib as nm


def avg_pool_2d(x, kernel_size, stride=None, padding=0):
    """
    2D Average pooling (functional API with autograd support)

    Applies a 2D average pooling over an input signal.
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
    >>> y = avg_pool_2d(x, kernel_size=2, stride=2)
    >>> y.shape
    (4, 16, 14, 14)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    >>> # x.grad is computed

    Notes
    -----
    This implementation uses im2col + mean for efficiency.
    The gradient is distributed equally to all positions in each pooling window.
    Uses numlib's mean function which supports autograd.
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

    # Use im2col
    col_data = nm.im2col(
        x._data, kernel_h, kernel_w, stride=stride[0], padding=padding[0]
    )
    col_data = col_data.reshape(N, C, kernel_h * kernel_w, out_h * out_w)

    # Take mean using numlib (supports autograd)
    col = nm.tensor(col_data)
    out = nm.mean(col, axis=2)  # (N, C, out_h*out_w)
    out = nm.reshape(out, (N, C, out_h, out_w))

    return out


class AvgPool2d(Module):
    """
    2D Average pooling layer

    Applies a 2D average pooling over an input signal.

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
    >>> pool = nl.AvgPool2d(kernel_size=2, stride=2)
    >>> x = nm.randn(32, 16, 28, 28)
    >>> y = pool(x)
    >>> y.shape
    (32, 16, 14, 14)

    >>> # With gradient computation
    >>> x = nm.randn(4, 3, 8, 8, requires_grad=True)
    >>> pool = nl.AvgPool2d(2, 2)
    >>> y = pool(x)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    >>> # x.grad is computed

    Notes
    -----
    Average pooling is commonly used in CNNs to downsample feature maps
    while preserving spatial information more smoothly than max pooling.
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
        Forward pass of average pooling

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W)

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, H_out, W_out)
        """
        return avg_pool_2d(x, self.kernel_size, self.stride, self.padding)

    def __repr__(self):
        return (
            f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding})"
        )
