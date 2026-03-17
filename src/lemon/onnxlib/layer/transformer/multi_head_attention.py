import numpy as np
import lemon.numlib as nm
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.MultiHeadAttention)
def trace_multi_head_attention(tracer, layer, input_name, output_name):
    """
    MultiHeadAttention decomposed into ONNX primitives.

    Uses Reshape with shape [0, 0, num_heads, d_k] where 0 means
    "copy this dimension from input" — handles dynamic batch and seq_len.
    """
    prefix = tracer.layer_prefix("mha")
    d_model = layer.d_model
    num_heads = layer.num_heads
    d_k = layer.d_k

    # --- Store weight/bias parameters ---
    W_q = nm.as_numpy(layer.W_q.data)
    W_k = nm.as_numpy(layer.W_k.data)
    W_v = nm.as_numpy(layer.W_v.data)
    W_o = nm.as_numpy(layer.W_o.data)
    tracer.add_parameter(f"{prefix}_W_q", W_q)
    tracer.add_parameter(f"{prefix}_W_k", W_k)
    tracer.add_parameter(f"{prefix}_W_v", W_v)
    tracer.add_parameter(f"{prefix}_W_o", W_o)

    if layer.use_bias:
        b_q = nm.as_numpy(layer.b_q.data)
        b_k = nm.as_numpy(layer.b_k.data)
        b_v = nm.as_numpy(layer.b_v.data)
        b_o = nm.as_numpy(layer.b_o.data)
        tracer.add_parameter(f"{prefix}_b_q", b_q)
        tracer.add_parameter(f"{prefix}_b_k", b_k)
        tracer.add_parameter(f"{prefix}_b_v", b_v)
        tracer.add_parameter(f"{prefix}_b_o", b_o)

    # Shape constants for split_heads: [0, 0, num_heads, d_k]
    split_shape_name = f"{prefix}_split_shape"
    tracer.add_parameter(
        split_shape_name,
        np.array([0, 0, num_heads, d_k], dtype=np.int64)
    )

    # Shape constants for merge_heads: [0, 0, d_model]
    merge_shape_name = f"{prefix}_merge_shape"
    tracer.add_parameter(
        merge_shape_name,
        np.array([0, 0, d_model], dtype=np.int64)
    )

    # Scale constant
    import math
    scale_val = np.array([1.0 / math.sqrt(d_k)], dtype=np.float32)
    scale_name = f"{prefix}_scale"
    tracer.add_parameter(scale_name, scale_val)

    def project(in_name, W_name, b_name=None):
        """Linear projection: MatMul + optional Add"""
        mm_out = tracer.unique_name(f"{prefix}_mm")
        tracer.add_node("MatMul", [in_name, W_name], [mm_out])
        if b_name is not None:
            add_out = tracer.unique_name(f"{prefix}_bias_add")
            tracer.add_node("Add", [mm_out, b_name], [add_out])
            return add_out
        return mm_out

    def split_heads(in_name, label):
        """(batch, seq, d_model) → (batch, num_heads, seq, d_k)"""
        reshaped = tracer.unique_name(f"{prefix}_{label}_reshaped")
        tracer.add_node("Reshape", [in_name, split_shape_name], [reshaped])
        transposed = tracer.unique_name(f"{prefix}_{label}_transposed")
        tracer.add_node("Transpose", [reshaped], [transposed], {"perm": [0, 2, 1, 3]})
        return transposed

    def merge_heads(in_name):
        """(batch, num_heads, seq_q, d_k) → (batch, seq_q, d_model)"""
        transposed = tracer.unique_name(f"{prefix}_ctx_transposed")
        tracer.add_node("Transpose", [in_name], [transposed], {"perm": [0, 2, 1, 3]})
        merged = tracer.unique_name(f"{prefix}_ctx_merged")
        tracer.add_node("Reshape", [transposed, merge_shape_name], [merged])
        return merged

    # --- Linear projections ---
    b_q_name = f"{prefix}_b_q" if layer.use_bias else None
    b_k_name = f"{prefix}_b_k" if layer.use_bias else None
    b_v_name = f"{prefix}_b_v" if layer.use_bias else None

    Q = project(input_name, f"{prefix}_W_q", b_q_name)
    K = project(input_name, f"{prefix}_W_k", b_k_name)
    V = project(input_name, f"{prefix}_W_v", b_v_name)

    # --- Split heads ---
    Q_h = split_heads(Q, "Q")
    K_h = split_heads(K, "K")
    V_h = split_heads(V, "V")

    # --- Scaled dot-product attention ---
    # K^T: (batch, heads, d_k, seq_k)
    K_t = tracer.unique_name(f"{prefix}_K_t")
    tracer.add_node("Transpose", [K_h], [K_t], {"perm": [0, 1, 3, 2]})

    # scores = Q @ K^T
    scores_raw = tracer.unique_name(f"{prefix}_scores_raw")
    tracer.add_node("MatMul", [Q_h, K_t], [scores_raw])

    # scores * scale
    scores = tracer.unique_name(f"{prefix}_scores")
    tracer.add_node("Mul", [scores_raw, scale_name], [scores])

    # Softmax
    attn_w = tracer.unique_name(f"{prefix}_attn_w")
    tracer.add_node("Softmax", [scores], [attn_w], {"axis": -1})

    # context = attn_w @ V
    context_h = tracer.unique_name(f"{prefix}_context_h")
    tracer.add_node("MatMul", [attn_w, V_h], [context_h])

    # --- Merge heads ---
    context = merge_heads(context_h)

    # --- Output projection ---
    out = project(context, f"{prefix}_W_o",
                  f"{prefix}_b_o" if layer.use_bias else None)

    tracer.add_node("Identity", [out], [output_name])
    tracer.layer_counter += 1
    return output_name
