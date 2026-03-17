import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.Linear)
def trace_linear(tracer, layer, input_name, output_name):
    weight = nm.as_numpy(layer.weight.data)
    prefix = tracer.layer_prefix("linear")
    weight_name = f"{prefix}_weight"
    tracer.add_parameter(weight_name, weight)

    matmul_out = tracer.unique_name("matmul_out")
    tracer.add_node("MatMul", [input_name, weight_name], [matmul_out])

    if layer.bias is not None:
        bias = nm.as_numpy(layer.bias.data)
        bias_name = f"{prefix}_bias"
        tracer.add_parameter(bias_name, bias)
        tracer.add_node("Add", [matmul_out, bias_name], [output_name])
    else:
        tracer.add_node("Identity", [matmul_out], [output_name])

    tracer.layer_counter += 1
    return output_name
