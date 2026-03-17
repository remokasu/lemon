"""
onnxlib.tracer - ONNX tracing infrastructure

Pure engine: no dependency on nnlib or any specific layer type.
Layer-specific tracers register themselves via register_tracer().
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


IR_VERSION = 11


def check_onnx_available():
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX is not installed. Please install it with:\n"
            "  pip install onnx onnxruntime"
        )


# ==============================
# Global Tracer Registry
# ==============================

_TRACER_REGISTRY: Dict[type, Any] = {}


def register_tracer(layer_type):
    """
    Register an ONNX tracer function for a layer type.

    The tracer function must have signature:
        fn(tracer: ComputationGraphTracer, layer, input_name: str, output_name: str) -> str

    Usage
    -----
    @register_tracer(SomeLayer)
    def trace_some_layer(tracer, layer, input_name, output_name):
        tracer.add_node("SomeOp", [input_name], [output_name])
        return output_name
    """
    def decorator(fn):
        _TRACER_REGISTRY[layer_type] = fn
        return fn
    return decorator


# ==============================
# Graph Node
# ==============================

class GraphNode:
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


# ==============================
# Computation Graph Tracer
# ==============================

class ComputationGraphTracer:
    """
    Traces a Sequential model's forward pass and builds an ONNX computation graph.

    Tracer functions registered via register_tracer() are called for each layer.
    Each tracer function has signature:
        fn(tracer, layer, input_name: str, output_name: str) -> str
    """

    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.parameters: Dict[str, np.ndarray] = {}
        self.tensor_counter = 0
        self.layer_counter = 0
        self.input_name = "input"
        self.output_name = "output"

    def unique_name(self, prefix: str = "tensor") -> str:
        name = f"{prefix}_{self.tensor_counter}"
        self.tensor_counter += 1
        return name

    def layer_prefix(self, layer_type: str) -> str:
        return f"{layer_type}_layer{self.layer_counter}"

    def add_node(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> GraphNode:
        node = GraphNode(op_type, inputs, outputs, attributes)
        self.nodes.append(node)
        return node

    def add_parameter(self, name: str, value: np.ndarray):
        self.parameters[name] = value

    def trace_sequential(self, model, input_name: str, output_name: str) -> str:
        current = input_name

        for i, module in enumerate(model.modules):
            fn = _TRACER_REGISTRY.get(type(module))
            if fn is None:
                raise ValueError(
                    f"Unsupported layer type: {type(module).__name__}. "
                    f"Implement a tracer with @register_tracer({type(module).__name__})."
                )

            next_name = output_name if i == len(model.modules) - 1 \
                else self.unique_name(f"layer{i}_out")

            current = fn(self, module, current, next_name)

        return current

    def trace(self, model, sample_input):
        self.nodes = []
        self.parameters = {}
        self.tensor_counter = 0
        self.layer_counter = 0

        import lemon.nnlib as nl
        if isinstance(model, nl.Sequential):
            self.trace_sequential(model, self.input_name, self.output_name)
        else:
            raise NotImplementedError(
                f"ONNX export not implemented for model type: {type(model).__name__}. "
                "Currently only Sequential is supported."
            )

        return self.input_name, self.output_name


# ==============================
# ONNX Model Builder
# ==============================

class ONNXModelBuilder:
    def __init__(self, tracer: ComputationGraphTracer):
        self.tracer = tracer

    def _dtype_to_onnx(self, np_dtype) -> int:
        dtype_map = {
            np.float16: TensorProto.FLOAT16,
            np.float32: TensorProto.FLOAT,
            np.float64: TensorProto.DOUBLE,
            np.int8:    TensorProto.INT8,
            np.int16:   TensorProto.INT16,
            np.int32:   TensorProto.INT32,
            np.int64:   TensorProto.INT64,
            np.uint8:   TensorProto.UINT8,
            np.uint16:  TensorProto.UINT16,
            np.uint32:  TensorProto.UINT32,
            np.uint64:  TensorProto.UINT64,
            np.bool_:   TensorProto.BOOL,
        }
        return dtype_map.get(np_dtype, TensorProto.FLOAT)

    def _make_initializer(self, name: str, value: np.ndarray):
        return helper.make_tensor(
            name=name,
            data_type=self._dtype_to_onnx(value.dtype),
            dims=value.shape,
            vals=value.flatten().tolist(),
        )

    def _make_value_info(self, name: str, shape: Tuple, dtype=np.float32):
        return helper.make_tensor_value_info(
            name, self._dtype_to_onnx(dtype), shape
        )

    def build(
        self,
        model_name: str,
        input_shape: Tuple,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        ir_version: Optional[int] = None,
    ):
        if dynamic_axes and "input" in dynamic_axes:
            input_shape = list(input_shape)
            for axis in dynamic_axes["input"]:
                input_shape[axis] = None
            input_shape = tuple(input_shape)

        inputs = [self._make_value_info(self.tracer.input_name, input_shape)]
        output_shape = tuple(None for _ in input_shape)
        outputs = [self._make_value_info(self.tracer.output_name, output_shape)]

        initializers = [
            self._make_initializer(name, value)
            for name, value in self.tracer.parameters.items()
        ]

        nodes = [
            helper.make_node(
                node.op_type,
                inputs=node.inputs,
                outputs=node.outputs,
                **node.attributes,
            )
            for node in self.tracer.nodes
        ]

        graph = helper.make_graph(
            nodes, model_name, inputs, outputs, initializers
        )
        model = helper.make_model(graph, producer_name="lemon")
        model.opset_import[0].version = 13

        if ir_version is not None:
            model.ir_version = ir_version

        check_model(model)
        return model
