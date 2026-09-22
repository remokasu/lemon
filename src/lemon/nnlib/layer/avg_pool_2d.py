from lemon.nnlib.module import Module
import lemon.numlib as nm



def _avg_pool_2d_forward(x, kernel_h, kernel_w, stride, padding, out_h, out_w):
    N, C = x.shape[:2]
    col = nm.im2col(x, kernel_h, kernel_w, stride=stride, padding=padding)
    col = col.reshape(N, C, kernel_h * kernel_w, out_h * out_w)
    output = col.mean(axis=2).reshape(N, C, out_h, out_w)
    return output, (x.shape, kernel_h, kernel_w, stride, padding)


def _avg_pool_2d_backward(ctx, grad, needs_grad):
    x_shape, kernel_h, kernel_w, stride, padding = ctx
    xp = nm.get_array_module(grad)
    N, C = x_shape[:2]
    K = kernel_h * kernel_w
    # 窓の平均の勾配: 出力の勾配を窓の各位置に 1/K ずつ配る
    grad_flat = grad.reshape(N, C, 1, -1) / K
    grad_col = xp.broadcast_to(grad_flat, (N, C, K, grad_flat.shape[-1]))
    grad_col = grad_col.reshape(N, C * K, -1)
    grad_x = nm.col2im(grad_col, x_shape, kernel_h, kernel_w, stride=stride, padding=padding)
    return (grad_x,)


_avg_pool_2d = nm.make_op(_avg_pool_2d_forward, _avg_pool_2d_backward)


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

    # im2col で窓を取り出して平均する（勾配は _avg_pool_2d_backward で入力に戻す）
    return _avg_pool_2d(
        x,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride=stride,
        padding=padding,
        out_h=out_h,
        out_w=out_w,
    )


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
