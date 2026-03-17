import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer, _TRACER_REGISTRY


def _trace_ff_block(tracer, layer, input_name, prefix):
    """Feed-forward block: Linear → activation → Linear"""
    linear_tracer = _TRACER_REGISTRY[nl.Linear]

    ff1_out = tracer.unique_name(f"{prefix}_ff1_out")
    linear_tracer(tracer, layer.ff1, input_name, ff1_out)

    act_out = tracer.unique_name(f"{prefix}_act_out")
    act_tracer = _TRACER_REGISTRY.get(type(layer._activation))
    if act_tracer is not None:
        # activation is a Module
        act_tracer(tracer, layer._activation, ff1_out, act_out)
    else:
        # activation is a function — map to known Module type
        from lemon.nnlib.activation.relu import relu
        from lemon.nnlib.activation.gelu import gelu
        if layer._activation is relu:
            tracer.add_node("Relu", [ff1_out], [act_out])
        else:
            # gelu approximation
            _trace_gelu_nodes(tracer, ff1_out, act_out)

    ff2_out = tracer.unique_name(f"{prefix}_ff2_out")
    linear_tracer(tracer, layer.ff2, act_out, ff2_out)
    return ff2_out


def _trace_gelu_nodes(tracer, input_name, output_name):
    sqrt_2_over_pi = np.array([0.7978845608028654], dtype=np.float32)
    coeff = np.array([0.044715], dtype=np.float32)
    three = np.array([3.0], dtype=np.float32)
    one = np.array([1.0], dtype=np.float32)
    half = np.array([0.5], dtype=np.float32)

    n = tracer.unique_name
    p = tracer.add_parameter

    three_n = n("gelu_three"); p(three_n, three)
    coeff_n = n("gelu_coeff"); p(coeff_n, coeff)
    scale_n = n("gelu_scale"); p(scale_n, sqrt_2_over_pi)
    one_n   = n("gelu_one");   p(one_n, one)
    half_n  = n("gelu_half");  p(half_n, half)

    x_cubed = n("gelu_x3")
    tracer.add_node("Pow", [input_name, three_n], [x_cubed])
    cx3 = n("gelu_cx3")
    tracer.add_node("Mul", [x_cubed, coeff_n], [cx3])
    sum_t = n("gelu_sum")
    tracer.add_node("Add", [input_name, cx3], [sum_t])
    scaled = n("gelu_scaled")
    tracer.add_node("Mul", [sum_t, scale_n], [scaled])
    tanh_o = n("gelu_tanh")
    tracer.add_node("Tanh", [scaled], [tanh_o])
    one_p = n("gelu_1ptanh")
    tracer.add_node("Add", [one_n, tanh_o], [one_p])
    xmul = n("gelu_xmul")
    tracer.add_node("Mul", [input_name, one_p], [xmul])
    tracer.add_node("Mul", [xmul, half_n], [output_name])


@register_tracer(nl.TransformerEncoderLayer)
def trace_transformer_encoder_layer(tracer, layer, input_name, output_name):
    """
    TransformerEncoderLayer decomposed into ONNX primitives.
    Supports Post-LN and Pre-LN (norm_first) variants.
    Mask is not supported in static ONNX export.
    """
    prefix = tracer.layer_prefix("tel")
    mha_tracer = _TRACER_REGISTRY[nl.MultiHeadAttention]
    ln_tracer = _TRACER_REGISTRY[nl.LayerNorm]

    if layer.norm_first:
        # Pre-LN: norm → attn → residual → norm → ffn → residual
        norm1_out = tracer.unique_name(f"{prefix}_norm1_out")
        ln_tracer(tracer, layer.norm1, input_name, norm1_out)

        sa_out = tracer.unique_name(f"{prefix}_sa_out")
        mha_tracer(tracer, layer.self_attn, norm1_out, sa_out)

        res1 = tracer.unique_name(f"{prefix}_res1")
        tracer.add_node("Add", [input_name, sa_out], [res1])

        norm2_out = tracer.unique_name(f"{prefix}_norm2_out")
        ln_tracer(tracer, layer.norm2, res1, norm2_out)

        ff_out = _trace_ff_block(tracer, layer, norm2_out, prefix)

        tracer.add_node("Add", [res1, ff_out], [output_name])
    else:
        # Post-LN: attn → residual → norm → ffn → residual → norm
        sa_out = tracer.unique_name(f"{prefix}_sa_out")
        mha_tracer(tracer, layer.self_attn, input_name, sa_out)

        res1_pre = tracer.unique_name(f"{prefix}_res1_pre")
        tracer.add_node("Add", [input_name, sa_out], [res1_pre])

        norm1_out = tracer.unique_name(f"{prefix}_norm1_out")
        ln_tracer(tracer, layer.norm1, res1_pre, norm1_out)

        ff_out = _trace_ff_block(tracer, layer, norm1_out, prefix)

        res2_pre = tracer.unique_name(f"{prefix}_res2_pre")
        tracer.add_node("Add", [norm1_out, ff_out], [res2_pre])

        ln_tracer(tracer, layer.norm2, res2_pre, output_name)

    tracer.layer_counter += 1
    return output_name


@register_tracer(nl.TransformerEncoder)
def trace_transformer_encoder(tracer, layer, input_name, output_name):
    """
    TransformerEncoder: loop through all encoder layers.
    """
    enc_layer_tracer = _TRACER_REGISTRY[nl.TransformerEncoderLayer]
    current = input_name

    for i, enc_layer in enumerate(layer.layers):
        next_name = tracer.unique_name(f"enc_layer{i}_out")
        enc_layer_tracer(tracer, enc_layer, current, next_name)
        current = next_name

    if layer.norm is not None:
        norm_tracer = _TRACER_REGISTRY.get(type(layer.norm))
        if norm_tracer is not None:
            norm_tracer(tracer, layer.norm, current, output_name)
        else:
            tracer.add_node("Identity", [current], [output_name])
    else:
        tracer.add_node("Identity", [current], [output_name])

    tracer.layer_counter += 1
    return output_name
