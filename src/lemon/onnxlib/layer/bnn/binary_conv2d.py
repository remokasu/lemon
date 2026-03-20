import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.BinaryConv2d)
def trace_binary_conv2d(tracer, layer, input_name, output_name):
    # Binarize weights at export time
    weight = np.sign(nm.as_numpy(layer.weight.data)).astype(np.float32)
    weight[weight == 0] = 1.0

    prefix = tracer.layer_prefix("binary_conv")
    weight_name = f"{prefix}_weight"
    tracer.add_parameter(weight_name, weight)

    stride = list(layer.stride) if not isinstance(layer.stride, int) else [layer.stride, layer.stride]
    padding_val = layer.padding
    if isinstance(padding_val, int):
        pads = [padding_val, padding_val, padding_val, padding_val]
    else:
        pads = [padding_val[0], padding_val[1], padding_val[0], padding_val[1]]
    dilation = list(layer.dilation) if not isinstance(layer.dilation, int) else [layer.dilation, layer.dilation]

    attrs = {
        "kernel_shape": [layer.kernel_h, layer.kernel_w],
        "strides": stride,
        "pads": pads,
        "dilations": dilation,
    }
    if layer.groups != 1:
        attrs["group"] = layer.groups

    if layer.bias is not None:
        bias = nm.as_numpy(layer.bias.data).astype(np.float32)
        bias_name = f"{prefix}_bias"
        tracer.add_parameter(bias_name, bias)
        tracer.add_node("Conv", [input_name, weight_name, bias_name], [output_name], attrs)
    else:
        tracer.add_node("Conv", [input_name, weight_name], [output_name], attrs)

    tracer.layer_counter += 1
    return output_name
