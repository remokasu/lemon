import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


def _normalize_kernel_attrs(kernel_size, stride, padding, dilation=None):
    attrs = {}
    attrs["kernel_shape"] = [kernel_size, kernel_size] if isinstance(kernel_size, int) else list(kernel_size)
    attrs["strides"] = [stride, stride] if isinstance(stride, int) else list(stride)
    if isinstance(padding, int):
        attrs["pads"] = [padding, padding, padding, padding]
    else:
        attrs["pads"] = [padding[0], padding[1], padding[0], padding[1]]
    if dilation is not None:
        attrs["dilations"] = [dilation, dilation] if isinstance(dilation, int) else list(dilation)
    return attrs


@register_tracer(nl.Conv2d)
def trace_conv2d(tracer, layer, input_name, output_name):
    weight = nm.as_numpy(layer.weight.data)
    prefix = tracer.layer_prefix("conv")
    weight_name = f"{prefix}_weight"
    tracer.add_parameter(weight_name, weight)

    dilation = layer.dilation if hasattr(layer, "dilation") and layer.dilation != 1 else None
    attrs = _normalize_kernel_attrs(layer.kernel_size, layer.stride, layer.padding, dilation)
    if hasattr(layer, "groups") and layer.groups != 1:
        attrs["group"] = layer.groups

    if layer.bias is not None:
        bias = nm.as_numpy(layer.bias.data)
        bias_name = f"{prefix}_bias"
        tracer.add_parameter(bias_name, bias)
        tracer.add_node("Conv", [input_name, weight_name, bias_name], [output_name], attrs)
    else:
        tracer.add_node("Conv", [input_name, weight_name], [output_name], attrs)

    tracer.layer_counter += 1
    return output_name
