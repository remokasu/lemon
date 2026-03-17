import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.Glu)
def trace_glu(tracer, layer, input_name, output_name):
    """GLU(x) = x[:half] * sigmoid(x[half:])  via Split + Sigmoid + Mul"""
    a_name = tracer.unique_name("glu_a")
    b_name = tracer.unique_name("glu_b")
    tracer.add_node("Split", [input_name], [a_name, b_name], {"axis": layer.dim})

    sig_name = tracer.unique_name("glu_sig")
    tracer.add_node("Sigmoid", [b_name], [sig_name])
    tracer.add_node("Mul", [a_name, sig_name], [output_name])
    return output_name
