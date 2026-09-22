import lemon.numlib as nm
from lemon.nnlib.module import Module


def _max_pool_2d_forward(x, kernel_size, stride, padding):
    xp = nm.get_array_module(x)
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

    # 最大値プーリングの padding は -inf で埋める（0 で埋めると、入力が負のとき
    # 埋めた 0 が最大値に選ばれてしまう）。埋めてから padding なしで im2col する
    if padding[0] > 0 or padding[1] > 0:
        x = xp.pad(
            x,
            ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
            constant_values=-xp.inf,
        )

    # Use im2col to extract patches
    col = nm.im2col(x, kernel_h, kernel_w, stride=stride, padding=0)
    col = col.reshape(N, C, kernel_h * kernel_w, out_h * out_w)

    # Take max along kernel dimension
    argmax = xp.argmax(col, axis=2)  # (N, C, out_h*out_w)
    max_vals = xp.take_along_axis(col, argmax[:, :, None, :], axis=2)[:, :, 0, :]
    output = max_vals.reshape(N, C, out_h, out_w)

    ctx = (col.shape, argmax, x.shape, kernel_h, kernel_w, stride, padding)  # x は padding 後
    return output, ctx


def _max_pool_2d_backward(ctx, grad, needs_grad):
    col_shape, argmax, x_shape, kernel_h, kernel_w, stride, padding = ctx
    xp = nm.get_array_module(grad)
    N, C, _, n_pos = col_shape

    # Distribute gradients only to max positions
    # 各 (n, c, pos) の最大値の位置は1つだけなので、代入で足りる
    grad_col = xp.zeros(col_shape, dtype=grad.dtype)
    n_idx, c_idx, pos_idx = xp.meshgrid(
        xp.arange(N), xp.arange(C), xp.arange(n_pos), indexing="ij"
    )
    grad_col[n_idx, c_idx, argmax, pos_idx] = grad.reshape(N, C, n_pos)

    # Use col2im to convert back to input gradient (padding 後の形), then crop the padding
    grad_col = grad_col.reshape(N, C * kernel_h * kernel_w, n_pos)
    grad_x = nm.col2im(grad_col, x_shape, kernel_h, kernel_w, stride=stride, padding=0)
    ph, pw = padding
    grad_x = grad_x[:, :, ph : x_shape[2] - ph, pw : x_shape[3] - pw]
    return (grad_x,)


_max_pool_2d = nm.make_op(_max_pool_2d_forward, _max_pool_2d_backward)


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
    return _max_pool_2d(x, kernel_size=kernel_size, stride=stride, padding=padding)


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
