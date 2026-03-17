import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.Dropout)
def trace_dropout(tracer, layer, input_name, output_name):
    # Inference mode: identity
    tracer.add_node("Identity", [input_name], [output_name])
    return output_name


@register_tracer(nl.Dropout2d)
def trace_dropout2d(tracer, layer, input_name, output_name):
    tracer.add_node("Identity", [input_name], [output_name])
    return output_name
