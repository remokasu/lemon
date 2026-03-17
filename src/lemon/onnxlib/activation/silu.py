import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.Silu)
def trace_silu(tracer, layer, input_name, output_name):
    """SiLU(x) = x * sigmoid(x)"""
    sig = tracer.unique_name("silu_sig")
    tracer.add_node("Sigmoid", [input_name], [sig])
    tracer.add_node("Mul", [input_name, sig], [output_name])
    return output_name
