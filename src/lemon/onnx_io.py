"""
onnx_io - ONNX model export utilities

This library provides functionality to convert nnlib models to ONNX format for
interoperability with other machine learning frameworks and deployment platforms.
It traces the computational graph and generates ONNX operators.

Key Features
------------
- Export nnlib models to ONNX format
- Support for common layers (Linear, Conv2D, BatchNorm, etc.)
- Computational graph tracing
- Model validation and verification
"""

from typing import List, Optional, Dict, Tuple, Any

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
    from onnx.checker import check_model

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    helper = None
    TensorProto = None
    check_model = None

import lemon.numlib as nm
import lemon.nnlib as nl


def _check_onnx_available():
    """Check if ONNX is available, raise ImportError if not"""
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX is not installed. Please install it with:\n"
            "  pip install onnx onnxruntime\n"
            "or\n"
            "  pip install lemon[onnx]"
        )


IR_VERSION = 11


# ==============================
# Decorator for Layer Registration
# ==============================


def handles(*layer_types):
    """レイヤータイプとトレーサーメソッドを関連付けるデコレータ"""

    def decorator(method):
        if not hasattr(method, "_handles_layers"):
            method._handles_layers = []
        method._handles_layers.extend(layer_types)
        return method

    return decorator


# ==============================
# Model Saving & Loading (ONNX)
# ==============================


# ==============================
# Helper Functions
# ==============================


def _normalize_kernel_attrs(kernel_size, stride, padding, dilation=None):
    """カーネル属性をONNX形式に正規化"""
    attrs = {}

    # kernel_size
    if isinstance(kernel_size, int):
        attrs["kernel_shape"] = [kernel_size, kernel_size]
    else:
        attrs["kernel_shape"] = list(kernel_size)

    # stride
    if isinstance(stride, int):
        attrs["strides"] = [stride, stride]
    else:
        attrs["strides"] = list(stride)

    # padding (ONNX形式: [top, left, bottom, right])
    if isinstance(padding, int):
        attrs["pads"] = [padding, padding, padding, padding]
    else:
        attrs["pads"] = [padding[0], padding[1], padding[0], padding[1]]

    # dilation (オプション)
    if dilation is not None:
        if isinstance(dilation, int):
            attrs["dilations"] = [dilation, dilation]
        else:
            attrs["dilations"] = list(dilation)

    return attrs


def _infer_input_shape(model):
    """
    Infer input shape from the first Linear layer

    Parameters
    ----------
    model : Module
        The model to inspect

    Returns
    -------
    tuple
        Input shape (without batch dimension)
    """
    # For Sequential models
    if hasattr(model, "modules"):
        for module in model.modules:
            if hasattr(module, "in_features"):
                return (module.in_features,)

    # Fallback
    raise ValueError("Cannot infer input shape. Please provide sample_input.")


# ==============================
# Computation Graph Tracer
# ==============================


class _GraphNode:
    """Represents a single operation in the computation graph"""

    def __init__(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}


class _ComputationGraphTracer:
    """Traces the forward pass to build an ONNX computation graph"""

    def __init__(self):
        self.nodes: List[_GraphNode] = []
        self.parameters: Dict[str, np.ndarray] = {}
        self.tensor_counter = 0
        self.layer_counter = 0
        self.input_name = "input"
        self.output_name = "output"

        # Build layer tracer registry from decorated methods
        self.layer_tracers = {}
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, "_handles_layers"):
                for layer_type in method._handles_layers:
                    self.layer_tracers[layer_type] = method

    def _get_unique_name(self, prefix: str = "tensor") -> str:
        name = f"{prefix}_{self.tensor_counter}"
        self.tensor_counter += 1
        return name

    def _get_layer_prefix(self, layer_type: str) -> str:
        """Get a unique prefix for the current layer"""
        prefix = f"{layer_type}_layer{self.layer_counter}"
        return prefix

    def add_node(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attributes: Optional[Dict[str, Any]] = None,
    ):
        node = _GraphNode(op_type, inputs, outputs, attributes)
        self.nodes.append(node)
        return node

    def add_parameter(self, name: str, value: np.ndarray):
        self.parameters[name] = value

    @handles(nl.Linear)
    def trace_linear(self, layer, input_name: str, output_name: str) -> str:
        # Extract weights and bias
        weight = nm.as_numpy(layer.weight.data)

        # Use layer counter for unique naming
        prefix = self._get_layer_prefix("linear")
        weight_name = f"{prefix}_weight"
        self.add_parameter(weight_name, weight)

        # MatMul
        matmul_output = self._get_unique_name("matmul_out")
        self.add_node("MatMul", [input_name, weight_name], [matmul_output])

        # Add bias if exists
        if layer.bias is not None:
            bias = nm.as_numpy(layer.bias.data)
            bias_name = f"{prefix}_bias"
            self.add_parameter(bias_name, bias)
            self.add_node("Add", [matmul_output, bias_name], [output_name])
        else:
            self.add_node("Identity", [matmul_output], [output_name])

        self.layer_counter += 1
        return output_name

    @handles(nl.Conv2d)
    def trace_conv2d(self, layer, input_name: str, output_name: str) -> str:
        """Trace Conv2d layer"""
        weight = nm.as_numpy(layer.weight.data)

        prefix = self._get_layer_prefix("conv")
        weight_name = f"{prefix}_weight"
        self.add_parameter(weight_name, weight)

        # Prepare attributes using helper function
        dilation = (
            layer.dilation
            if hasattr(layer, "dilation") and layer.dilation != 1
            else None
        )
        attrs = _normalize_kernel_attrs(
            layer.kernel_size, layer.stride, layer.padding, dilation
        )

        if hasattr(layer, "groups") and layer.groups != 1:
            attrs["group"] = layer.groups

        # Conv operation
        if layer.bias is not None:
            bias = nm.as_numpy(layer.bias.data)
            bias_name = f"{prefix}_bias"
            self.add_parameter(bias_name, bias)
            self.add_node(
                "Conv", [input_name, weight_name, bias_name], [output_name], attrs
            )
        else:
            self.add_node("Conv", [input_name, weight_name], [output_name], attrs)

        self.layer_counter += 1
        return output_name

    @handles(nl.MaxPool2d)
    def trace_maxpool2d(self, layer, input_name: str, output_name: str) -> str:
        """Trace MaxPool2d layer"""
        attrs = _normalize_kernel_attrs(layer.kernel_size, layer.stride, layer.padding)

        self.add_node("MaxPool", [input_name], [output_name], attrs)
        self.layer_counter += 1
        return output_name

    @handles(nl.AvgPool2d)
    def trace_avgpool2d(self, layer, input_name: str, output_name: str) -> str:
        """Trace AvgPool2d layer"""
        attrs = _normalize_kernel_attrs(layer.kernel_size, layer.stride, layer.padding)

        self.add_node("AveragePool", [input_name], [output_name], attrs)
        self.layer_counter += 1
        return output_name

    @handles(nl.AdaptiveAvgPool2d)
    def trace_adaptiveavgpool2d(self, layer, input_name: str, output_name: str) -> str:
        """
        Trace AdaptiveAvgPool2d layer

        AdaptiveAvgPool2dは入力サイズに応じて自動的にkernel_sizeとstrideを計算する必要がありますが、
        ONNXでは静的なグラフが必要なため、一般的なアプローチとして GlobalAveragePool を使用するか、
        特定の出力サイズ(1x1など)の場合のみ対応します。
        """
        if layer.output_size == (1, 1) or layer.output_size == 1:
            # Global Average Pooling
            self.add_node("GlobalAveragePool", [input_name], [output_name])
        else:
            raise NotImplementedError(
                f"AdaptiveAvgPool2d with output_size={layer.output_size} is not supported in ONNX export. "
                "Only output_size=(1, 1) is supported (converts to GlobalAveragePool)."
            )
        self.layer_counter += 1
        return output_name

    @handles(nl.GlobalAveragePooling2d)
    def trace_globalaveragepooling2d(
        self, layer, input_name: str, output_name: str
    ) -> str:
        """Trace GlobalAveragePooling2d layer"""
        self.add_node("GlobalAveragePool", [input_name], [output_name])
        self.layer_counter += 1
        return output_name

    @handles(nl.Flatten)
    def trace_flatten(self, layer, input_name: str, output_name: str) -> str:
        """Trace Flatten layer"""
        # Flatten to (batch_size, -1)
        # start_dim=1 means keep batch dimension and flatten the rest
        start_dim = getattr(layer, "start_dim", 1)
        self.add_node("Flatten", [input_name], [output_name], {"axis": start_dim})
        self.layer_counter += 1
        return output_name

    # =============================
    # Activation functions
    # =============================
    @handles(nl.Relu)
    def trace_relu(self, input_name: str, output_name: str) -> str:
        self.add_node("Relu", [input_name], [output_name])
        return output_name

    @handles(nl.Sigmoid)
    def trace_sigmoid(self, input_name: str, output_name: str) -> str:
        self.add_node("Sigmoid", [input_name], [output_name])
        return output_name

    @handles(nl.Tanh)
    def trace_tanh(self, input_name: str, output_name: str) -> str:
        self.add_node("Tanh", [input_name], [output_name])
        return output_name

    @handles(nl.Softmax)
    def trace_softmax(self, layer, input_name: str, output_name: str) -> str:
        axis = layer.axis if hasattr(layer, "axis") else -1
        self.add_node("Softmax", [input_name], [output_name], {"axis": axis})
        return output_name

    @handles(nl.LeakyRelu)
    def trace_leaky_relu(self, layer, input_name: str, output_name: str) -> str:
        """Trace LeakyRelu layer"""
        alpha = layer.alpha if hasattr(layer, "alpha") else 0.01
        self.add_node("LeakyRelu", [input_name], [output_name], {"alpha": alpha})
        return output_name

    @handles(nl.Elu)
    def trace_elu(self, layer, input_name: str, output_name: str) -> str:
        """Trace Elu layer"""
        alpha = layer.alpha
        self.add_node("Elu", [input_name], [output_name], {"alpha": alpha})
        return output_name

    @handles(nl.Selu)
    def trace_selu(self, layer, input_name: str, output_name: str) -> str:
        """Trace Selu layer"""
        alpha = layer.alpha if hasattr(layer, "alpha") else 1.67326324
        gamma = layer.gamma if hasattr(layer, "gamma") else 1.05070098
        self.add_node(
            "Selu", [input_name], [output_name], {"alpha": alpha, "gamma": gamma}
        )
        return output_name

    @handles(nl.Celu)
    def trace_celu(self, layer, input_name: str, output_name: str) -> str:
        """Trace Celu layer"""
        alpha = layer.alpha if hasattr(layer, "alpha") else 1.0
        self.add_node("Celu", [input_name], [output_name], {"alpha": alpha})
        return output_name

    @handles(nl.HardSigmoid)
    def trace_hard_sigmoid(self, layer, input_name: str, output_name: str) -> str:
        """Trace HardSigmoid layer"""
        alpha = layer.alpha if hasattr(layer, "alpha") else 0.2
        beta = layer.beta if hasattr(layer, "beta") else 0.5
        self.add_node(
            "HardSigmoid", [input_name], [output_name], {"alpha": alpha, "beta": beta}
        )
        return output_name

    @handles(nl.HardSwish)
    def trace_hard_swish(self, layer, input_name: str, output_name: str) -> str:
        """Trace HardSwish layer"""
        self.add_node("HardSwish", [input_name], [output_name])
        return output_name

    @handles(nl.ThresholdedRelu)
    def trace_thresholded_relu(self, layer, input_name: str, output_name: str) -> str:
        """Trace ThresholdedRelu layer"""
        alpha = layer.alpha if hasattr(layer, "alpha") else 1.0
        self.add_node("ThresholdedRelu", [input_name], [output_name], {"alpha": alpha})
        return output_name

    @handles(nl.Gelu)
    def trace_gelu(self, layer, input_name: str, output_name: str) -> str:
        """
        Trace GELU layer

        ONNX doesn't have native GELU, so we use the tanh approximation:
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        """
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715

        # Create constant tensors
        three_name = self._get_unique_name("three")
        self.add_parameter(three_name, np.array([3.0], dtype=np.float32))

        coeff_name = self._get_unique_name("coeff")
        self.add_parameter(coeff_name, np.array([coeff], dtype=np.float32))

        scale_name = self._get_unique_name("scale")
        self.add_parameter(scale_name, np.array([sqrt_2_over_pi], dtype=np.float32))

        one_name = self._get_unique_name("one")
        self.add_parameter(one_name, np.array([1.0], dtype=np.float32))

        half_name = self._get_unique_name("half")
        self.add_parameter(half_name, np.array([0.5], dtype=np.float32))

        # x³
        x_cubed = self._get_unique_name("x_cubed")
        self.add_node("Pow", [input_name, three_name], [x_cubed])

        # 0.044715 * x³
        coeff_x_cubed = self._get_unique_name("coeff_x_cubed")
        self.add_node("Mul", [x_cubed, coeff_name], [coeff_x_cubed])

        # x + 0.044715 * x³
        sum_term = self._get_unique_name("sum_term")
        self.add_node("Add", [input_name, coeff_x_cubed], [sum_term])

        # √(2/π) * (x + 0.044715 * x³)
        scaled_term = self._get_unique_name("scaled_term")
        self.add_node("Mul", [sum_term, scale_name], [scaled_term])

        # tanh(...)
        tanh_out = self._get_unique_name("tanh_out")
        self.add_node("Tanh", [scaled_term], [tanh_out])

        # 1 + tanh(...)
        one_plus_tanh = self._get_unique_name("one_plus_tanh")
        self.add_node("Add", [one_name, tanh_out], [one_plus_tanh])

        # x * (1 + tanh(...))
        x_times_term = self._get_unique_name("x_times_term")
        self.add_node("Mul", [input_name, one_plus_tanh], [x_times_term])

        # 0.5 * x * (1 + tanh(...))
        self.add_node("Mul", [x_times_term, half_name], [output_name])

        return output_name

    def trace_silu(self, layer, input_name: str, output_name: str) -> str:
        """
        Trace SiLU/Swish layer

        SiLU(x) = x * sigmoid(x)
        """
        sigmoid_out = self._get_unique_name("sigmoid_out")
        self.add_node("Sigmoid", [input_name], [sigmoid_out])
        self.add_node("Mul", [input_name, sigmoid_out], [output_name])
        return output_name

    @handles(nl.Mish)
    def trace_mish(self, layer, input_name: str, output_name: str) -> str:
        """
        Trace Mish layer

        ONNX opset 18+ has native Mish operator
        """
        self.add_node("Mish", [input_name], [output_name])
        return output_name

    @handles(nl.PRelu)
    def trace_prelu(self, layer, input_name: str, output_name: str) -> str:
        """Trace PRelu layer"""
        slope = nm.as_numpy(layer.slope.data)
        slope_name = self._get_unique_name("prelu_slope")
        self.add_parameter(slope_name, slope)
        self.add_node("PRelu", [input_name, slope_name], [output_name])
        return output_name

    @handles(nl.Softplus)
    def trace_softplus(self, layer, input_name: str, output_name: str) -> str:
        """
        Trace Softplus layer

        ONNX has native Softplus operator
        """
        self.add_node("Softplus", [input_name], [output_name])
        return output_name

    @handles(nl.Softsign)
    def trace_softsign(self, layer, input_name: str, output_name: str) -> str:
        """
        Trace Softsign layer

        ONNX has native Softsign operator
        """
        self.add_node("Softsign", [input_name], [output_name])
        return output_name

    # =============================
    # Regularization layers
    # ============================
    @handles(nl.Dropout)
    def trace_dropout(self, layer, input_name: str, output_name: str) -> str:
        # In inference mode, dropout is identity
        self.add_node("Identity", [input_name], [output_name])
        return output_name

    @handles(nl.Dropout2d)
    def trace_dropout2d(self, layer, input_name: str, output_name: str) -> str:
        """Trace Dropout2d layer (inference mode: identity)"""
        # In inference mode, dropout is identity
        self.add_node("Identity", [input_name], [output_name])
        return output_name

    @handles(nl.BatchNorm1d)
    def trace_batchnorm1d(self, layer, input_name: str, output_name: str) -> str:
        # Extract parameters
        if layer.affine:
            scale = nm.as_numpy(layer.gamma.data)
            bias = nm.as_numpy(layer.beta.data)
        else:
            scale = np.ones(layer.num_features, dtype=np.float32)
            bias = np.zeros(layer.num_features, dtype=np.float32)

        if layer.track_running_stats:
            mean = nm.as_numpy(layer.running_mean)
            var = nm.as_numpy(layer.running_var)
        else:
            raise ValueError("Cannot export BatchNorm1d without running stats")

        # Add parameters with layer prefix
        prefix = self._get_layer_prefix("bn1d")
        scale_name = f"{prefix}_scale"
        bias_name = f"{prefix}_bias"
        mean_name = f"{prefix}_mean"
        var_name = f"{prefix}_var"

        self.add_parameter(scale_name, scale)
        self.add_parameter(bias_name, bias)
        self.add_parameter(mean_name, mean)
        self.add_parameter(var_name, var)

        # BatchNormalization node
        self.add_node(
            "BatchNormalization",
            [input_name, scale_name, bias_name, mean_name, var_name],
            [output_name],
            {"epsilon": layer.eps, "momentum": layer.momentum},
        )

        self.layer_counter += 1
        return output_name

    @handles(nl.BatchNorm2d)
    def trace_batchnorm2d(self, layer, input_name: str, output_name: str) -> str:
        """Trace BatchNorm2d layer"""
        # Extract parameters
        if layer.affine:
            scale = nm.as_numpy(layer.gamma.data)
            bias = nm.as_numpy(layer.beta.data)
        else:
            scale = np.ones(layer.num_features, dtype=np.float32)
            bias = np.zeros(layer.num_features, dtype=np.float32)

        if layer.track_running_stats:
            mean = nm.as_numpy(layer.running_mean)
            var = nm.as_numpy(layer.running_var)
        else:
            raise ValueError("Cannot export BatchNorm2d without running stats")

        # Add parameters with layer prefix
        prefix = self._get_layer_prefix("bn2d")
        scale_name = f"{prefix}_scale"
        bias_name = f"{prefix}_bias"
        mean_name = f"{prefix}_mean"
        var_name = f"{prefix}_var"

        self.add_parameter(scale_name, scale)
        self.add_parameter(bias_name, bias)
        self.add_parameter(mean_name, mean)
        self.add_parameter(var_name, var)

        # BatchNormalization node
        self.add_node(
            "BatchNormalization",
            [input_name, scale_name, bias_name, mean_name, var_name],
            [output_name],
            {"epsilon": layer.eps, "momentum": layer.momentum},
        )

        self.layer_counter += 1
        return output_name

    def trace_sequential(self, model, input_name: str, output_name: str) -> str:
        """Sequential モデルをトレース"""
        import inspect

        current_tensor = input_name

        for i, module in enumerate(model.modules):
            tracer = self.layer_tracers.get(type(module))

            if tracer is None:
                raise ValueError(
                    f"Unsupported layer type: {type(module).__name__}. "
                    f"Please implement a trace method with @handles({type(module).__name__}) decorator."
                )

            # 最後のモジュールは出力名を使用
            if i == len(model.modules) - 1:
                next_tensor = output_name
            else:
                next_tensor = self._get_unique_name(f"layer{i}_out")

            self.layer_counter += 1

            # トレーサーを呼び出し（layerを持つものと持たないものに対応）
            import inspect

            sig = inspect.signature(tracer)
            param_count = len(
                [
                    p
                    for p in sig.parameters.values()
                    if p.default == inspect.Parameter.empty
                ]
            )

            if param_count == 3:  # layer, input_name, output_name (selfは除く)
                current_tensor = tracer(module, current_tensor, next_tensor)
            elif param_count == 2:  # input_name, output_name (selfは除く)
                current_tensor = tracer(current_tensor, next_tensor)
            else:
                raise TypeError(
                    f"Unexpected tracer signature for {type(module).__name__}: {sig}"
                )

        return current_tensor

    def trace(self, model, sample_input):
        """Main tracing function"""
        # Reset state
        self.nodes = []
        self.parameters = {}
        self.tensor_counter = 0

        # Trace the model
        if isinstance(model, nl.Sequential):
            final_output = self.trace_sequential(
                model, self.input_name, self.output_name
            )
        else:
            raise NotImplementedError(
                f"ONNX export not implemented for model type: {type(model).__name__}. "
                "Currently only Sequential is supported."
            )

        return self.input_name, self.output_name


# ==============================
# ONNX Model Builder
# ==============================


class _ONNXModelBuilder:
    """Builds an ONNX model from a traced computation graph"""

    def __init__(self, tracer: _ComputationGraphTracer):
        self.tracer = tracer

    def _numpy_dtype_to_onnx(self, np_dtype) -> int:
        """Convert NumPy dtype to ONNX TensorProto data type"""
        dtype_map = {
            np.float16: TensorProto.FLOAT16,
            np.float32: TensorProto.FLOAT,
            np.float64: TensorProto.DOUBLE,
            np.int8: TensorProto.INT8,
            np.int16: TensorProto.INT16,
            np.int32: TensorProto.INT32,
            np.int64: TensorProto.INT64,
            np.uint8: TensorProto.UINT8,
            np.uint16: TensorProto.UINT16,
            np.uint32: TensorProto.UINT32,
            np.uint64: TensorProto.UINT64,
            np.bool_: TensorProto.BOOL,
        }
        return dtype_map.get(np_dtype, TensorProto.FLOAT)

    def _create_initializer(self, name: str, value: np.ndarray):
        """Create an ONNX initializer (constant tensor)"""
        return helper.make_tensor(
            name=name,
            data_type=self._numpy_dtype_to_onnx(value.dtype),
            dims=value.shape,
            vals=value.flatten().tolist(),
        )

    def _create_value_info(self, name: str, shape: Tuple[int, ...], dtype=np.float32):
        """Create an ONNX ValueInfoProto"""
        return helper.make_tensor_value_info(
            name, self._numpy_dtype_to_onnx(dtype), shape
        )

    def build(
        self,
        model_name: str,
        input_shape: Tuple[int, ...],
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        ir_version: Optional[int] = None,
    ):
        """
        Build the ONNX model

        Parameters
        ----------
        model_name : str
            Name of the model
        input_shape : Tuple[int, ...]
            Shape of the input tensor
        dynamic_axes : Optional[Dict[str, Dict[int, str]]], optional
            Dynamic axes specification
        ir_version : Optional[int], optional
            ONNX IR version to use. If None, uses ONNX library default.
            Common values:
            - None: Use ONNX default (recommended for latest features)
            - 11: Compatible with ONNX Runtime 1.23.x
            - 10: Maximum compatibility with older ONNX Runtime versions
            - 8: Very broad compatibility

        Returns
        -------
        onnx.ModelProto
            The built ONNX model
        """
        # Apply dynamic axes if specified
        if dynamic_axes and "input" in dynamic_axes:
            input_shape = list(input_shape)
            for axis, name in dynamic_axes["input"].items():
                input_shape[axis] = None
            input_shape = tuple(input_shape)

        # Create input
        inputs = [self._create_value_info(self.tracer.input_name, input_shape)]

        # Create output (dynamic shape)
        output_shape = [None] * len(input_shape)
        outputs = [
            self._create_value_info(self.tracer.output_name, tuple(output_shape))
        ]

        # Create initializers
        initializers = []
        for name, value in self.tracer.parameters.items():
            initializers.append(self._create_initializer(name, value))

        # Create nodes
        nodes = []
        for graph_node in self.tracer.nodes:
            # Build attributes dictionary directly for make_node
            attrs = {}
            for attr_name, attr_value in graph_node.attributes.items():
                attrs[attr_name] = attr_value

            # Create node with attributes as keyword arguments
            node = helper.make_node(
                graph_node.op_type,
                inputs=graph_node.inputs,
                outputs=graph_node.outputs,
                **attrs,  # Pass attributes as keyword arguments
            )
            nodes.append(node)

        # Create graph
        graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)

        # Create model
        model = helper.make_model(graph, producer_name="nnlib")
        model.opset_import[0].version = 13

        # Set IR version if specified
        if ir_version is not None:
            model.ir_version = ir_version

        # Check model validity
        check_model(model)

        return model


# ==============================
# Main Export Function
# ==============================


def _export_onnx_internal(
    model,
    filepath: str,
    sample_input,
    model_name: str = "lemon_model",
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ir_version: Optional[int] = None,
    verbose: bool = True,
):
    """
    Internal function to export a nnlib model to ONNX format

    This function should be called from neurolib's export_model()
    """
    if verbose:
        print("  Tracing computation graph...")

    # Ensure inference mode
    prev_train_state = nl.train.is_enabled()
    nl.train.disable()

    try:
        # Trace the computation graph
        tracer = _ComputationGraphTracer()
        tracer.trace(model, sample_input)

        if verbose:
            print(f"  Traced {len(tracer.nodes)} operations")
            print(f"  Found {len(tracer.parameters)} parameters")

        # Build ONNX model
        if verbose:
            print("  Building ONNX model...")

        builder = _ONNXModelBuilder(tracer)
        onnx_model = builder.build(
            model_name=model_name,
            input_shape=tuple(sample_input.shape),
            dynamic_axes=dynamic_axes,
            ir_version=ir_version,
        )

        # Save to file
        onnx.save(onnx_model, filepath)
    finally:
        # Restore train state
        nl.train.set_enabled(prev_train_state)


def export_model(
    model,
    filepath: str,
    sample_input: Optional[nm.NumType] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
    dynamic_batch: bool = True,
    model_name: str = "lemon_model",
    ir_version: Optional[int] = IR_VERSION,
    verbose: bool = True,
):
    """
    Export model to ONNX format

    Parameters
    ----------
    model : Module
        Model to export
    filepath : str
        Path to save the model (should end with .onnx)
    sample_input : nm.NumType, optional
        Sample input for shape inference. Required if input_shape is None.
    input_shape : Tuple[int, ...], optional
        Shape of input tensor. If None, inferred from sample_input.
    dynamic_batch : bool, optional
        Whether to use dynamic batch size (default: True)
    model_name : str, optional
        Name of the model (default: "lemon_model")
    ir_version : int, optional
        ONNX IR version to use. If None, uses ONNX library default.
        Common values:
        - None: Use ONNX default (recommended for latest features)
        - 11: Compatible with ONNX Runtime 1.23.x
        - 10: Maximum compatibility with older ONNX Runtime versions
        - 8: Very broad compatibility
    verbose : bool, optional
        Print export information (default: True)

    Examples
    --------
    >>> model = Sequential(Linear(10, 5), ReLU())
    >>> # Use default (latest)
    >>> export_model(model, 'model.onnx', sample_input=nm.randn(1, 10))
    >>>
    >>> # Force IR version for compatibility
    >>> export_model(model, 'model.onnx', sample_input=nm.randn(1, 10), ir_version=11)
    """
    _check_onnx_available()

    if sample_input is None and input_shape is None:
        raise ValueError("Either sample_input or input_shape must be provided")

    if sample_input is None:
        sample_input = nm.zeros(input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    if verbose:
        print(f"Exporting model to {filepath}...")

    _export_onnx_internal(
        model,
        filepath,
        sample_input,
        model_name=model_name,
        dynamic_axes=dynamic_axes,
        ir_version=ir_version,
        verbose=verbose,
    )

    if verbose:
        print("  Model exported successfully")


def load_model(model, filepath: str, verbose: bool = False):
    """
    Load model parameters from ONNX file

    Parameters
    ----------
    model : Module
        Model to load parameters into
    filepath : str
        Path to the ONNX model file
    verbose : bool, optional
        Print loading information

    Examples
    --------
    >>> model = Sequential(Linear(10, 5), ReLU())
    >>> load_model(model, 'model.onnx')
    """
    _check_onnx_available()

    if verbose:
        print(f"Loading model from {filepath}...")

    onnx_model = onnx.load(filepath)
    initializers = {
        init.name: onnx.numpy_helper.to_array(init)
        for init in onnx_model.graph.initializer
    }

    if verbose:
        print(f"  Found {len(initializers)} initializers in ONNX file")

    # Collect all parameters from the model in order
    param_list = []
    param_names = []

    for module in model.modules if hasattr(model, "modules") else [model]:
        if isinstance(module, nl.Linear):
            param_list.append(module.weight.data)
            param_names.append("Linear.weight")
            if module.bias is not None:
                param_list.append(module.bias.data)
                param_names.append("Linear.bias")

        elif isinstance(module, nl.Conv2d):
            param_list.append(module.weight.data)
            param_names.append("Conv2d.weight")
            if module.bias is not None:
                param_list.append(module.bias.data)
                param_names.append("Conv2d.bias")

        elif isinstance(module, (nl.BatchNorm1d, nl.BatchNorm2d)):
            if module.affine:
                param_list.append(module.gamma.data)
                param_names.append("BatchNorm.gamma")
                param_list.append(module.beta.data)
                param_names.append("BatchNorm.beta")
            # Note: running_mean and running_var are handled separately

    # Separate weights, biases, and batchnorm params
    onnx_weights = sorted([k for k in initializers.keys() if "weight" in k])
    onnx_biases = sorted(
        [k for k in initializers.keys() if "bias" in k and "bn" not in k]
    )
    onnx_bn_scales = sorted([k for k in initializers.keys() if "bn_scale" in k])
    onnx_bn_biases = sorted([k for k in initializers.keys() if "bn_bias" in k])
    onnx_bn_means = sorted([k for k in initializers.keys() if "bn_mean" in k])
    onnx_bn_vars = sorted([k for k in initializers.keys() if "bn_var" in k])

    # Load Linear/Conv2d parameters
    weight_idx = 0
    bias_idx = 0
    bn_idx = 0

    for i, (param, name) in enumerate(zip(param_list, param_names)):
        if "weight" in name and "BatchNorm" not in name:
            if weight_idx < len(onnx_weights):
                weight_data = initializers[onnx_weights[weight_idx]]
                param._data = nm.get_array_module(param._data).asarray(weight_data)
                if verbose:
                    print(f"  Loaded {name} from {onnx_weights[weight_idx]}")
                weight_idx += 1

        elif "bias" in name and "BatchNorm" not in name:
            if bias_idx < len(onnx_biases):
                bias_data = initializers[onnx_biases[bias_idx]]
                param._data = nm.get_array_module(param._data).asarray(bias_data)
                if verbose:
                    print(f"  Loaded {name} from {onnx_biases[bias_idx]}")
                bias_idx += 1

        elif "gamma" in name:
            if bn_idx < len(onnx_bn_scales):
                scale_data = initializers[onnx_bn_scales[bn_idx]]
                param._data = nm.get_array_module(param._data).asarray(scale_data)
                if verbose:
                    print(f"  Loaded {name} from {onnx_bn_scales[bn_idx]}")

        elif "beta" in name:
            if bn_idx < len(onnx_bn_biases):
                bias_data = initializers[onnx_bn_biases[bn_idx]]
                param._data = nm.get_array_module(param._data).asarray(bias_data)
                if verbose:
                    print(f"  Loaded {name} from {onnx_bn_biases[bn_idx]}")
                bn_idx += 1  # Increment only after loading both gamma and beta

    # Load BatchNorm running statistics
    bn_module_idx = 0
    for module in model.modules if hasattr(model, "modules") else [model]:
        if (
            isinstance(module, (nl.BatchNorm1d, nl.BatchNorm2d))
            and module.track_running_stats
        ):
            if bn_module_idx < len(onnx_bn_means):
                mean_data = initializers[onnx_bn_means[bn_module_idx]]
                module.running_mean[:] = nm.get_array_module(
                    module.running_mean
                ).asarray(mean_data)
                if verbose:
                    print(
                        f"  Loaded BatchNorm running_mean from {onnx_bn_means[bn_module_idx]}"
                    )

                var_data = initializers[onnx_bn_vars[bn_module_idx]]
                module.running_var[:] = nm.get_array_module(module.running_var).asarray(
                    var_data
                )
                if verbose:
                    print(
                        f"  Loaded BatchNorm running_var from {onnx_bn_vars[bn_module_idx]}"
                    )

                bn_module_idx += 1

    if verbose:
        loaded_params = weight_idx + bias_idx + bn_idx * 2
        print(
            f"  Loaded {loaded_params} parameters + {bn_module_idx * 2} running stats"
        )


# ==============================
# Export All Public APIs
# ==============================

__all__ = [
    "export_model",
    "load_model",
]
