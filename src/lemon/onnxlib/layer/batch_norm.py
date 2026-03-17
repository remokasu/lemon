import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


def _trace_batchnorm(tracer, layer, input_name, output_name, prefix_key):
    if layer.affine:
        scale = nm.as_numpy(layer.gamma.data)
        bias = nm.as_numpy(layer.beta.data)
    else:
        scale = np.ones(layer.num_features, dtype=np.float32)
        bias = np.zeros(layer.num_features, dtype=np.float32)

    if not layer.track_running_stats:
        raise ValueError("Cannot export BatchNorm without running stats")

    mean = nm.as_numpy(layer.running_mean)
    var = nm.as_numpy(layer.running_var)

    prefix = tracer.layer_prefix(prefix_key)
    scale_name, bias_name = f"{prefix}_scale", f"{prefix}_bias"
    mean_name, var_name = f"{prefix}_mean", f"{prefix}_var"

    tracer.add_parameter(scale_name, scale)
    tracer.add_parameter(bias_name, bias)
    tracer.add_parameter(mean_name, mean)
    tracer.add_parameter(var_name, var)

    tracer.add_node(
        "BatchNormalization",
        [input_name, scale_name, bias_name, mean_name, var_name],
        [output_name],
        {"epsilon": layer.eps, "momentum": layer.momentum},
    )
    tracer.layer_counter += 1
    return output_name


@register_tracer(nl.BatchNorm1d)
def trace_batchnorm1d(tracer, layer, input_name, output_name):
    return _trace_batchnorm(tracer, layer, input_name, output_name, "bn1d")


@register_tracer(nl.BatchNorm2d)
def trace_batchnorm2d(tracer, layer, input_name, output_name):
    return _trace_batchnorm(tracer, layer, input_name, output_name, "bn2d")
