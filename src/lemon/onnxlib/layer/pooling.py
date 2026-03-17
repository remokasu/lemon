import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


def _pool_attrs(kernel_size, stride, padding):
    attrs = {}
    attrs["kernel_shape"] = [kernel_size, kernel_size] if isinstance(kernel_size, int) else list(kernel_size)
    attrs["strides"] = [stride, stride] if isinstance(stride, int) else list(stride)
    if isinstance(padding, int):
        attrs["pads"] = [padding, padding, padding, padding]
    else:
        attrs["pads"] = [padding[0], padding[1], padding[0], padding[1]]
    return attrs


@register_tracer(nl.MaxPool2d)
def trace_maxpool2d(tracer, layer, input_name, output_name):
    tracer.add_node("MaxPool", [input_name], [output_name],
                    _pool_attrs(layer.kernel_size, layer.stride, layer.padding))
    tracer.layer_counter += 1
    return output_name


@register_tracer(nl.AvgPool2d)
def trace_avgpool2d(tracer, layer, input_name, output_name):
    tracer.add_node("AveragePool", [input_name], [output_name],
                    _pool_attrs(layer.kernel_size, layer.stride, layer.padding))
    tracer.layer_counter += 1
    return output_name


@register_tracer(nl.AdaptiveAvgPool2d)
def trace_adaptive_avgpool2d(tracer, layer, input_name, output_name):
    if layer.output_size in ((1, 1), 1):
        tracer.add_node("GlobalAveragePool", [input_name], [output_name])
    else:
        raise NotImplementedError(
            f"AdaptiveAvgPool2d with output_size={layer.output_size} is not supported. "
            "Only output_size=(1,1) is supported."
        )
    tracer.layer_counter += 1
    return output_name


@register_tracer(nl.GlobalAveragePooling2d)
def trace_global_avgpool2d(tracer, layer, input_name, output_name):
    tracer.add_node("GlobalAveragePool", [input_name], [output_name])
    tracer.layer_counter += 1
    return output_name
