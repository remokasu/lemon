import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


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
    # Get array module
    xp = nm.get_array_module(x._data)
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

    if groups == 1:
        # Standard convolution (original implementation)
        col = nm.im2col(
            x._data,
            kernel_h,
            kernel_w,
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
        )  # (N, C_in*K*K, out_h*out_w)

        weight_flat = weight._data.reshape(C_out, -1)  # (C_out, C_in*K*K)

        output_data = xp.zeros((N, C_out, out_h * out_w), dtype=x._data.dtype)
        for i in range(N):
            output_data[i] = weight_flat @ col[i]  # (C_out, out_h*out_w)

        output_data = output_data.reshape(N, C_out, out_h, out_w)

        # Save for backward
        saved_col = col
        saved_weight_flat = weight_flat

    else:
        # Grouped convolution
        C_in_per_group = C_in // groups
        C_out_per_group = C_out // groups

        output_data = xp.zeros((N, C_out, out_h, out_w), dtype=x._data.dtype)
        saved_col_groups = []
        saved_weight_flat_groups = []

        for g in range(groups):
            # Extract input channels for this group
            x_group = x._data[:, g * C_in_per_group : (g + 1) * C_in_per_group, :, :]

            # Convert to column matrix
            col_group = nm.im2col(
                x_group,
                kernel_h,
                kernel_w,
                stride=stride[0],
                padding=padding[0],
                dilation=dilation[0],
            )  # (N, C_in_per_group*K*K, out_h*out_w)

            # Extract weight for this group
            weight_group = weight._data[
                g * C_out_per_group : (g + 1) * C_out_per_group, :, :, :
            ]
            weight_flat_group = weight_group.reshape(C_out_per_group, -1)

            # Perform convolution for this group
            for i in range(N):
                output_data[
                    i, g * C_out_per_group : (g + 1) * C_out_per_group, :, :
                ] = (weight_flat_group @ col_group[i]).reshape(
                    C_out_per_group, out_h, out_w
                )

            saved_col_groups.append(col_group)
            saved_weight_flat_groups.append(weight_flat_group)

        # Save for backward
        saved_col = saved_col_groups
        saved_weight_flat = saved_weight_flat_groups

    # Create result
    result = nm._create_result(output_data)

    # Add bias
    if bias is not None:
        bias_reshaped = bias._data.reshape(1, -1, 1, 1)
        result_data_with_bias = result._data + bias_reshaped
        result = nm._create_result(result_data_with_bias)

    # Gradient computation
    if not nm.autograd.is_enabled() or not (
        x.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        result.requires_grad = False
        return result

    result.requires_grad = True
    result._prev = (x, weight) if bias is None else (x, weight, bias)

    # Save variables for backward
    saved_x_shape = x.shape
    saved_kernel_h = kernel_h
    saved_kernel_w = kernel_w
    saved_stride = stride
    saved_padding = padding
    saved_dilation = dilation
    saved_groups = groups

    def _backward():
        if result.grad is None:
            return

        grad_output = result.grad._data  # (N, C_out, out_h, out_w)

        if saved_groups == 1:
            # Standard convolution backward
            grad_output_flat = grad_output.reshape(N, C_out, -1)

            if weight.requires_grad:
                grad_weight_flat = xp.zeros_like(saved_weight_flat)
                for i in range(N):
                    grad_weight_flat += grad_output_flat[i] @ saved_col[i].T
                grad_weight = grad_weight_flat.reshape(weight.shape)
                grad_weight_result = nm._create_result(grad_weight)
                if weight.grad is None:
                    weight.grad = grad_weight_result
                else:
                    weight.grad._data += grad_weight_result._data

            if x.requires_grad:
                grad_col = xp.zeros_like(saved_col)
                for i in range(N):
                    grad_col[i] = saved_weight_flat.T @ grad_output_flat[i]
                grad_x_data = nm.col2im(
                    grad_col,
                    saved_x_shape,
                    saved_kernel_h,
                    saved_kernel_w,
                    stride=saved_stride[0],
                    padding=saved_padding[0],
                    dilation=saved_dilation[0],
                )
                grad_x = nm._create_result(grad_x_data)
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad._data += grad_x._data

        else:
            # Grouped convolution backward
            C_in_per_group = C_in // saved_groups
            C_out_per_group = C_out // saved_groups

            if weight.requires_grad:
                grad_weight_data = xp.zeros_like(weight._data)

            if x.requires_grad:
                grad_x_data = xp.zeros(saved_x_shape, dtype=x._data.dtype)

            for g in range(saved_groups):
                grad_output_group = grad_output[
                    :, g * C_out_per_group : (g + 1) * C_out_per_group, :, :
                ]
                grad_output_flat_group = grad_output_group.reshape(
                    N, C_out_per_group, -1
                )

                if weight.requires_grad:
                    grad_weight_flat_group = xp.zeros_like(saved_weight_flat[g])
                    for i in range(N):
                        grad_weight_flat_group += (
                            grad_output_flat_group[i] @ saved_col[g][i].T
                        )
                    grad_weight_data[
                        g * C_out_per_group : (g + 1) * C_out_per_group, :, :, :
                    ] = grad_weight_flat_group.reshape(
                        C_out_per_group, C_in_per_group, saved_kernel_h, saved_kernel_w
                    )

                if x.requires_grad:
                    grad_col_group = xp.zeros_like(saved_col[g])
                    for i in range(N):
                        grad_col_group[i] = (
                            saved_weight_flat[g].T @ grad_output_flat_group[i]
                        )

                    x_group_shape = (
                        N,
                        C_in_per_group,
                        saved_x_shape[2],
                        saved_x_shape[3],
                    )
                    grad_x_group_data = nm.col2im(
                        grad_col_group,
                        x_group_shape,
                        saved_kernel_h,
                        saved_kernel_w,
                        stride=saved_stride[0],
                        padding=saved_padding[0],
                        dilation=saved_dilation[0],
                    )
                    grad_x_data[
                        :, g * C_in_per_group : (g + 1) * C_in_per_group, :, :
                    ] = grad_x_group_data

            if weight.requires_grad:
                grad_weight_result = nm._create_result(grad_weight_data)
                if weight.grad is None:
                    weight.grad = grad_weight_result
                else:
                    weight.grad._data += grad_weight_result._data

            if x.requires_grad:
                grad_x = nm._create_result(grad_x_data)
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad._data += grad_x._data

        # Gradient w.r.t. bias (same for both cases)
        if bias is not None and bias.requires_grad:
            grad_bias_data = xp.sum(grad_output, axis=(0, 2, 3))
            grad_bias = nm._create_result(grad_bias_data)
            if bias.grad is None:
                bias.grad = grad_bias
            else:
                bias.grad._data += grad_bias._data

    result._backward = _backward
    return result


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
