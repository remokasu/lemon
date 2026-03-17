import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.PositionalEncoding)
def trace_positional_encoding(tracer, layer, input_name, output_name):
    """
    PositionalEncoding: Add pre-computed constant PE table (sliced to seq_len).
    Since seq_len may vary, we store the full PE table and use Slice at runtime.
    For static export simplicity, we Add the full PE table truncated to max_len.
    In practice, inputs should be padded/truncated to a fixed length.
    """
    prefix = tracer.layer_prefix("pe")

    # Store the full PE table as a constant parameter
    # pe shape: (1, max_len, d_model)
    pe_data = layer._pe  # numpy array (1, max_len, d_model)
    pe_name = f"{prefix}_table"
    tracer.add_parameter(pe_name, pe_data.astype(np.float32))

    # Add PE to input: input + pe_table (broadcast over batch)
    tracer.add_node("Add", [input_name, pe_name], [output_name])
    tracer.layer_counter += 1
    return output_name
