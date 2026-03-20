import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.Sign)
def trace_sign(tracer, layer, input_name, output_name):
    tracer.add_node("Sign", [input_name], [output_name])
    return output_name
