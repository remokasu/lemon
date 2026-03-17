"""
onnxlib - ONNX model export/import for lemon

Usage
-----
    from lemon.onnxlib import export_model, load_model
"""

from typing import Optional, Dict, Tuple

import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import (
    check_onnx_available,
    ComputationGraphTracer,
    ONNXModelBuilder,
    IR_VERSION,
    ONNX_AVAILABLE,
)

# Register all built-in tracers by importing the sub-packages
from lemon.onnxlib import layer      # noqa: F401
from lemon.onnxlib import activation  # noqa: F401


def _infer_input_shape(model):
    """Infer input shape from the first layer with in_features."""
    if hasattr(model, "modules"):
        for module in model.modules:
            if hasattr(module, "in_features"):
                return (module.in_features,)
    raise ValueError("Cannot infer input shape. Please provide sample_input.")


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
    Export a lemon model to ONNX format.

    Parameters
    ----------
    model : Sequential
        Model to export (must be nl.Sequential)
    filepath : str
        Output path (e.g. 'model.onnx')
    sample_input : Tensor, optional
        Sample input for shape inference
    input_shape : tuple, optional
        Input shape. Used if sample_input is None.
    dynamic_batch : bool
        Use dynamic batch size (default: True)
    model_name : str
        ONNX model name (default: 'lemon_model')
    ir_version : int, optional
        ONNX IR version (default: 11)
    verbose : bool
        Print progress (default: True)

    Examples
    --------
    >>> model = nl.Sequential(nl.Linear(10, 5), nl.Relu())
    >>> export_model(model, 'model.onnx', sample_input=nm.randn(1, 10))
    """
    check_onnx_available()

    if sample_input is None and input_shape is None:
        raise ValueError("Either sample_input or input_shape must be provided")
    if sample_input is None:
        sample_input = nm.zeros(input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    if verbose:
        print(f"Exporting model to {filepath}...")

    prev_train = nl.train.is_enabled()
    nl.train.disable()

    try:
        tracer = ComputationGraphTracer()
        tracer.trace(model, sample_input)

        if verbose:
            print(f"  Traced {len(tracer.nodes)} operations")
            print(f"  Found {len(tracer.parameters)} parameters")

        builder = ONNXModelBuilder(tracer)
        onnx_model = builder.build(
            model_name=model_name,
            input_shape=tuple(sample_input.shape),
            dynamic_axes=dynamic_axes,
            ir_version=ir_version,
        )

        import onnx
        onnx.save(onnx_model, filepath)
    finally:
        nl.train.set_enabled(prev_train)

    if verbose:
        print("  Model exported successfully")


def load_model(model, filepath: str, verbose: bool = False):
    """
    Load model parameters from an ONNX file.

    Parameters
    ----------
    model : Sequential
        Model to load parameters into
    filepath : str
        Path to the ONNX model file
    verbose : bool
        Print loading information (default: False)

    Examples
    --------
    >>> model = nl.Sequential(nl.Linear(10, 5), nl.Relu())
    >>> load_model(model, 'model.onnx')
    """
    check_onnx_available()

    import onnx
    if verbose:
        print(f"Loading model from {filepath}...")

    onnx_model = onnx.load(filepath)
    initializers = {
        init.name: onnx.numpy_helper.to_array(init)
        for init in onnx_model.graph.initializer
    }

    if verbose:
        print(f"  Found {len(initializers)} initializers")

    modules = model.modules if hasattr(model, "modules") else [model]

    onnx_weights = sorted(k for k in initializers if "weight" in k)
    onnx_biases  = sorted(k for k in initializers if "bias" in k and "bn" not in k)
    onnx_bn_scales = sorted(k for k in initializers if "bn_scale" in k)
    onnx_bn_biases = sorted(k for k in initializers if "bn_bias" in k)
    onnx_bn_means  = sorted(k for k in initializers if "bn_mean" in k)
    onnx_bn_vars   = sorted(k for k in initializers if "bn_var" in k)

    w_idx = b_idx = bn_idx = 0

    for module in modules:
        if isinstance(module, nl.Linear):
            if w_idx < len(onnx_weights):
                module.weight.data._data = nm.get_array_module(
                    module.weight.data._data).asarray(initializers[onnx_weights[w_idx]])
                w_idx += 1
            if module.bias is not None and b_idx < len(onnx_biases):
                module.bias.data._data = nm.get_array_module(
                    module.bias.data._data).asarray(initializers[onnx_biases[b_idx]])
                b_idx += 1

        elif isinstance(module, nl.Conv2d):
            if w_idx < len(onnx_weights):
                module.weight.data._data = nm.get_array_module(
                    module.weight.data._data).asarray(initializers[onnx_weights[w_idx]])
                w_idx += 1
            if module.bias is not None and b_idx < len(onnx_biases):
                module.bias.data._data = nm.get_array_module(
                    module.bias.data._data).asarray(initializers[onnx_biases[b_idx]])
                b_idx += 1

        elif isinstance(module, (nl.BatchNorm1d, nl.BatchNorm2d)):
            if module.affine and bn_idx < len(onnx_bn_scales):
                module.gamma.data._data = nm.get_array_module(
                    module.gamma.data._data).asarray(initializers[onnx_bn_scales[bn_idx]])
                module.beta.data._data = nm.get_array_module(
                    module.beta.data._data).asarray(initializers[onnx_bn_biases[bn_idx]])
            if module.track_running_stats and bn_idx < len(onnx_bn_means):
                module.running_mean[:] = nm.get_array_module(
                    module.running_mean).asarray(initializers[onnx_bn_means[bn_idx]])
                module.running_var[:] = nm.get_array_module(
                    module.running_var).asarray(initializers[onnx_bn_vars[bn_idx]])
            bn_idx += 1

    if verbose:
        print(f"  Loaded {w_idx} weights, {b_idx} biases, {bn_idx} BN layers")


__all__ = ["export_model", "load_model"]
