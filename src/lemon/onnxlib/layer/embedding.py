import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.Embedding)
def trace_embedding(tracer, layer, input_name, output_name):
    """
    Embedding via ONNX Gather op:
        output = Gather(weight, indices, axis=0)
    """
    prefix = tracer.layer_prefix("embedding")
    weight = nm.as_numpy(layer.weight.data)
    weight_name = f"{prefix}_weight"
    tracer.add_parameter(weight_name, weight)

    tracer.add_node("Gather", [weight_name, input_name], [output_name], {"axis": 0})
    tracer.layer_counter += 1
    return output_name
