import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.LayerNorm)
def trace_layer_norm(tracer, layer, input_name, output_name):
    """
    LayerNorm via basic ops (opset 13 compatible):
    ReduceMean → Sub → Pow → ReduceMean → Add(eps) → Sqrt → Div → Mul+Add
    """
    prefix = tracer.layer_prefix("layernorm")
    axis = -len(layer.normalized_shape)

    two_name = f"{prefix}_two"
    eps_name = f"{prefix}_eps"
    tracer.add_parameter(two_name, np.array([2.0], dtype=np.float32))
    tracer.add_parameter(eps_name, np.array([layer.eps], dtype=np.float32))

    mean_out = tracer.unique_name("ln_mean")
    tracer.add_node("ReduceMean", [input_name], [mean_out],
                    {"axes": [axis], "keepdims": 1})

    x_centered = tracer.unique_name("ln_centered")
    tracer.add_node("Sub", [input_name, mean_out], [x_centered])

    x_sq = tracer.unique_name("ln_sq")
    tracer.add_node("Pow", [x_centered, two_name], [x_sq])

    var_out = tracer.unique_name("ln_var")
    tracer.add_node("ReduceMean", [x_sq], [var_out],
                    {"axes": [axis], "keepdims": 1})

    var_eps = tracer.unique_name("ln_var_eps")
    tracer.add_node("Add", [var_out, eps_name], [var_eps])

    std_out = tracer.unique_name("ln_std")
    tracer.add_node("Sqrt", [var_eps], [std_out])

    x_normed = tracer.unique_name("ln_normed")
    tracer.add_node("Div", [x_centered, std_out], [x_normed])

    if layer.elementwise_affine:
        scale = nm.as_numpy(layer.weight.data)
        bias = nm.as_numpy(layer.bias.data)
        scale_name, bias_name = f"{prefix}_scale", f"{prefix}_bias"
        tracer.add_parameter(scale_name, scale)
        tracer.add_parameter(bias_name, bias)
        scaled = tracer.unique_name("ln_scaled")
        tracer.add_node("Mul", [x_normed, scale_name], [scaled])
        tracer.add_node("Add", [scaled, bias_name], [output_name])
    else:
        tracer.add_node("Identity", [x_normed], [output_name])

    tracer.layer_counter += 1
    return output_name


@register_tracer(nl.RMSNorm)
def trace_rms_norm(tracer, layer, input_name, output_name):
    """
    RMSNorm: Pow → ReduceMean → Add(eps) → Sqrt → Div → Mul(weight)
    """
    prefix = tracer.layer_prefix("rmsnorm")

    two_name = f"{prefix}_two"
    eps_name = f"{prefix}_eps"
    tracer.add_parameter(two_name, np.array([2.0], dtype=np.float32))
    tracer.add_parameter(eps_name, np.array([layer.eps], dtype=np.float32))

    x_sq = tracer.unique_name("rms_x_sq")
    tracer.add_node("Pow", [input_name, two_name], [x_sq])

    x_ms = tracer.unique_name("rms_ms")
    tracer.add_node("ReduceMean", [x_sq], [x_ms], {"axes": [-1], "keepdims": 1})

    x_ms_eps = tracer.unique_name("rms_ms_eps")
    tracer.add_node("Add", [x_ms, eps_name], [x_ms_eps])

    rms = tracer.unique_name("rms_rms")
    tracer.add_node("Sqrt", [x_ms_eps], [rms])

    x_normed = tracer.unique_name("rms_normed")
    tracer.add_node("Div", [input_name, rms], [x_normed])

    if layer.elementwise_affine:
        weight = nm.as_numpy(layer.weight.data)
        weight_name = f"{prefix}_weight"
        tracer.add_parameter(weight_name, weight)
        tracer.add_node("Mul", [x_normed, weight_name], [output_name])
    else:
        tracer.add_node("Identity", [x_normed], [output_name])

    tracer.layer_counter += 1
    return output_name
