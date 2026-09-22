import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


def _conv_2d_forward(x, weight, bias, stride, padding, dilation, groups):
    xp = nm.get_array_module(x)
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, kernel_h, kernel_w = weight.shape

    # Validate groups parameter
    if C_in % groups != 0:
        raise ValueError(f"in_channels ({C_in}) must be divisible by groups ({groups})")
    if C_out % groups != 0:
        raise ValueError(
            f"out_channels ({C_out}) must be divisible by groups ({groups})"
        )
    if C_in // groups != C_in_per_group:
        raise ValueError(
            f"weight shape mismatch: expected C_in/groups={C_in // groups}, got {C_in_per_group}"
        )

    # Normalize parameters to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Calculate output dimensions
    out_h = (H + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_w = (W + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1

    # 各グループの (im2col した入力, 平らにした重み)。groups == 1 なら1組だけ
    C_in_g = C_in // groups
    C_out_g = C_out // groups
    output = xp.zeros((N, C_out, out_h * out_w), dtype=x.dtype)
    cols, weight_flats = [], []
    for g in range(groups):
        col = nm.im2col(
            x[:, g * C_in_g : (g + 1) * C_in_g, :, :],
            kernel_h,
            kernel_w,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )  # (N, C_in_g*K*K, out_h*out_w)
        weight_flat = weight[g * C_out_g : (g + 1) * C_out_g].reshape(C_out_g, -1)
        for i in range(N):
            output[i, g * C_out_g : (g + 1) * C_out_g] = weight_flat @ col[i]
        cols.append(col)
        weight_flats.append(weight_flat)

    output = output.reshape(N, C_out, out_h, out_w)

    # Add bias
    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)

    ctx = (x.shape, weight.shape, cols, weight_flats, kernel_h, kernel_w, stride, padding, dilation, groups)
    return output, ctx


def _conv_2d_backward(ctx, grad, needs_grad):
    x_shape, weight_shape, cols, weight_flats, kernel_h, kernel_w, stride, padding, dilation, groups = ctx
    xp = nm.get_array_module(grad)
    N, C_in, H, W = x_shape
    C_out = weight_shape[0]
    C_in_g = C_in // groups
    C_out_g = C_out // groups

    grad_x = xp.zeros(x_shape, dtype=grad.dtype) if needs_grad[0] else None
    grad_weight = xp.zeros(weight_shape, dtype=grad.dtype) if needs_grad[1] else None
    grad_bias = xp.sum(grad, axis=(0, 2, 3)) if needs_grad[2] else None

    grad_flat = grad.reshape(N, C_out, -1)
    for g in range(groups):
        grad_g = grad_flat[:, g * C_out_g : (g + 1) * C_out_g]  # (N, C_out_g, out_h*out_w)
        col, weight_flat = cols[g], weight_flats[g]

        if needs_grad[1]:
            grad_weight_flat = xp.zeros_like(weight_flat)
            for i in range(N):
                grad_weight_flat = grad_weight_flat + grad_g[i] @ col[i].T
            grad_weight[g * C_out_g : (g + 1) * C_out_g] = grad_weight_flat.reshape(
                C_out_g, C_in_g, kernel_h, kernel_w
            )

        if needs_grad[0]:
            grad_col = xp.zeros_like(col)
            for i in range(N):
                grad_col[i] = weight_flat.T @ grad_g[i]
            grad_x[:, g * C_in_g : (g + 1) * C_in_g] = nm.col2im(
                grad_col,
                (N, C_in_g, H, W),
                kernel_h,
                kernel_w,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

    return grad_x, grad_weight, grad_bias


_conv_2d = nm.make_op(_conv_2d_forward, _conv_2d_backward)


def conv_2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    2D Convolution (functional API with autograd support)

    Applies a 2D convolution over an input signal.
    This function supports autograd when used with numlib tensors.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (N, C_in, H, W)
    weight : Tensor
        Weight tensor of shape (C_out, C_in/groups, kernel_h, kernel_w)
    bias : Tensor, optional
        Bias tensor of shape (C_out,)
    stride : int or tuple, optional
        Stride (default: 1)
    padding : int or tuple, optional
        Padding (default: 0)
    dilation : int or tuple, optional
        Dilation (default: 1)
    groups : int, optional
        Number of blocked connections from input channels to output channels (default: 1)

    Returns
    -------
    Tensor
        Output tensor of shape (N, C_out, H_out, W_out)

    Examples
    --------
    >>> # Standard convolution
    >>> x = nm.randn(4, 3, 28, 28, requires_grad=True)
    >>> weight = nm.randn(16, 3, 3, 3, requires_grad=True)
    >>> y = conv_2d(x, weight, stride=1, padding=1)

    >>> # Depthwise convolution (groups = in_channels)
    >>> x = nm.randn(4, 32, 28, 28)
    >>> weight = nm.randn(32, 1, 3, 3)  # Note: C_in/groups = 32/32 = 1
    >>> y = conv_2d(x, weight, stride=1, padding=1, groups=32)

    Notes
    -----
    This implementation uses im2col + matrix multiplication for efficiency.
    Autograd is fully supported through numlib's automatic differentiation.

    When groups > 1:
    - Input channels are divided into 'groups' groups
    - Output channels are divided into 'groups' groups
    - Each output group only connects to its corresponding input group
    - This reduces parameters by a factor of 'groups'
    """
    return _conv_2d(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class Conv2d(Module):
    """
    2D Convolutional layer

    Applies a 2D convolution over an input signal composed of several input planes.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced by the convolution
    kernel_size : int or tuple
        Size of the convolving kernel
    stride : int or tuple, optional
        Stride of the convolution (default: 1)
    padding : int or tuple, optional
        Zero-padding added to both sides of the input (default: 0)
    dilation : int or tuple, optional
        Spacing between kernel elements (default: 1)
    groups : int, optional
        Number of blocked connections from input channels to output channels (default: 1)
        When groups=1: standard convolution
        When groups=in_channels: depthwise convolution
    bias : bool, optional
        If True, adds a learnable bias to the output (default: True)

    Attributes
    ----------
    weight : Parameter
        Learnable weights of shape (out_channels, in_channels/groups, kernel_h, kernel_w)
    bias : Parameter or None
        Learnable bias of shape (out_channels,)

    Examples
    --------
    >>> import numlib as nm
    >>> import lemon as lm
    >>>
    >>> # Standard convolution
    >>> conv = nl.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    >>> x = nm.randn(32, 3, 28, 28)
    >>> y = conv(x)
    >>> y.shape
    (32, 16, 28, 28)

    >>> # Depthwise convolution (MobileNet style)
    >>> depthwise = nl.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
    >>> x = nm.randn(4, 32, 28, 28)
    >>> y = depthwise(x)
    >>> y.shape
    (4, 32, 28, 28)

    >>> # Grouped convolution (ResNeXt style)
    >>> grouped = nl.Conv2d(64, 128, kernel_size=3, padding=1, groups=4)
    >>> x = nm.randn(4, 64, 28, 28)
    >>> y = grouped(x)
    >>> y.shape
    (4, 128, 28, 28)

    Notes
    -----
    The convolution operation is implemented using im2col for efficiency.
    Output spatial dimensions are calculated as:
        out_h = (in_h + 2*padding - dilation*(kernel_h-1) - 1) // stride + 1
        out_w = (in_w + 2*padding - dilation*(kernel_w-1) - 1) // stride + 1

    When groups > 1:
    - in_channels and out_channels must be divisible by groups
    - Each group processes in_channels/groups input channels
    - Each group produces out_channels/groups output channels
    - Parameters are reduced by a factor of groups
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
    ):
        super().__init__()

        # Validate groups parameter
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by groups ({groups})"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        # Normalize kernel_size to tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.kernel_h, self.kernel_w = kernel_size

        # Normalize stride to tuple
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        # Normalize padding to tuple
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        # Normalize dilation to tuple
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation

        # He initialization for ReLU networks
        # For grouped convolution, fan_in is per group
        fan_in = (in_channels // groups) * self.kernel_h * self.kernel_w
        std = (2.0 / fan_in) ** 0.5

        # Weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
        self.weight = Parameter(
            nm.randn(out_channels, in_channels // groups, self.kernel_h, self.kernel_w)
            * std
        )

        if bias:
            self.bias = Parameter(nm.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        """
        Forward pass of convolution

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C_in, H, W)

        Returns
        -------
        Tensor
            Output tensor of shape (N, C_out, H_out, W_out)
        """
        # Call the functional API
        return conv_2d(
            x,
            self.weight.data,
            self.bias.data if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def __repr__(self):
        return (
            f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None})"
        )
