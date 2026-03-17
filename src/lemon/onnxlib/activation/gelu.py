import numpy as np
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.Gelu)
def trace_gelu(tracer, layer, input_name, output_name):
    """
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    n = tracer.unique_name
    p = tracer.add_parameter

    three_n = n("gelu_three"); p(three_n, np.array([3.0], dtype=np.float32))
    coeff_n = n("gelu_coeff"); p(coeff_n, np.array([0.044715], dtype=np.float32))
    scale_n = n("gelu_scale"); p(scale_n, np.array([0.7978845608028654], dtype=np.float32))
    one_n   = n("gelu_one");   p(one_n,   np.array([1.0], dtype=np.float32))
    half_n  = n("gelu_half");  p(half_n,  np.array([0.5], dtype=np.float32))

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
    return output_name
