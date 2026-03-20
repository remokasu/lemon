import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.BinaryLinear)
def trace_binary_linear(tracer, layer, input_name, output_name):
    # Binarize weights at export time (inference uses fixed binary weights)
    weight = np.sign(nm.as_numpy(layer.weight.data)).astype(np.float32)
    weight[weight == 0] = 1.0  # sign(0) -> +1

    prefix = tracer.layer_prefix("binary_linear")
    weight_name = f"{prefix}_weight"
    tracer.add_parameter(weight_name, weight)

    matmul_out = tracer.unique_name("bin_matmul_out")
    tracer.add_node("MatMul", [input_name, weight_name], [matmul_out])

    if layer.bias is not None:
        bias = nm.as_numpy(layer.bias.data).astype(np.float32)
        bias_name = f"{prefix}_bias"
        tracer.add_parameter(bias_name, bias)
        tracer.add_node("Add", [matmul_out, bias_name], [output_name])
    else:
        tracer.add_node("Identity", [matmul_out], [output_name])

    tracer.layer_counter += 1
    return output_name
