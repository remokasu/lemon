import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.Flatten)
def trace_flatten(tracer, layer, input_name, output_name):
    start_dim = getattr(layer, "start_dim", 1)
    tracer.add_node("Flatten", [input_name], [output_name], {"axis": start_dim})
    tracer.layer_counter += 1
    return output_name
