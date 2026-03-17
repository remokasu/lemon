import numpy as np
import lemon.nnlib as nl
from lemon.onnxlib.tracer import register_tracer


@register_tracer(nl.Relu)
def trace_relu(tracer, layer, input_name, output_name):
    tracer.add_node("Relu", [input_name], [output_name])
    return output_name


@register_tracer(nl.Sigmoid)
def trace_sigmoid(tracer, layer, input_name, output_name):
    tracer.add_node("Sigmoid", [input_name], [output_name])
    return output_name


@register_tracer(nl.Tanh)
def trace_tanh(tracer, layer, input_name, output_name):
    tracer.add_node("Tanh", [input_name], [output_name])
    return output_name


@register_tracer(nl.Softmax)
def trace_softmax(tracer, layer, input_name, output_name):
    axis = getattr(layer, "axis", -1)
    tracer.add_node("Softmax", [input_name], [output_name], {"axis": axis})
    return output_name


@register_tracer(nl.LeakyRelu)
def trace_leaky_relu(tracer, layer, input_name, output_name):
    tracer.add_node("LeakyRelu", [input_name], [output_name],
                    {"alpha": getattr(layer, "alpha", 0.01)})
    return output_name


@register_tracer(nl.Elu)
def trace_elu(tracer, layer, input_name, output_name):
    tracer.add_node("Elu", [input_name], [output_name], {"alpha": layer.alpha})
    return output_name


@register_tracer(nl.Selu)
def trace_selu(tracer, layer, input_name, output_name):
    tracer.add_node("Selu", [input_name], [output_name], {
        "alpha": getattr(layer, "alpha", 1.67326324),
        "gamma": getattr(layer, "gamma", 1.05070098),
    })
    return output_name


@register_tracer(nl.Celu)
def trace_celu(tracer, layer, input_name, output_name):
    tracer.add_node("Celu", [input_name], [output_name],
                    {"alpha": getattr(layer, "alpha", 1.0)})
    return output_name


@register_tracer(nl.HardSigmoid)
def trace_hard_sigmoid(tracer, layer, input_name, output_name):
    tracer.add_node("HardSigmoid", [input_name], [output_name], {
        "alpha": getattr(layer, "alpha", 0.2),
        "beta":  getattr(layer, "beta",  0.5),
    })
    return output_name


@register_tracer(nl.HardSwish)
def trace_hard_swish(tracer, layer, input_name, output_name):
    """HardSwish(x) = x * HardSigmoid(x) = x * clip((x+3)/6, 0, 1)"""
    n, p = tracer.unique_name, tracer.add_parameter
    three = n("hs_three"); p(three, np.array([3.0], dtype=np.float32))
    six   = n("hs_six");   p(six,   np.array([6.0], dtype=np.float32))
    zero  = n("hs_zero");  p(zero,  np.array([0.0], dtype=np.float32))
    one   = n("hs_one");   p(one,   np.array([1.0], dtype=np.float32))

    x3    = n("hs_x3");    tracer.add_node("Add",  [input_name, three], [x3])
    x3d6  = n("hs_x3d6");  tracer.add_node("Div",  [x3, six],           [x3d6])
    clamp = n("hs_clamp"); tracer.add_node("Clip", [x3d6, zero, one],   [clamp])
    tracer.add_node("Mul", [input_name, clamp], [output_name])
    return output_name


@register_tracer(nl.Mish)
def trace_mish(tracer, layer, input_name, output_name):
    """Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))"""
    n, p = tracer.unique_name, tracer.add_parameter
    one = n("mish_one"); p(one, np.array([1.0], dtype=np.float32))

    exp_x  = n("mish_exp");  tracer.add_node("Exp",      [input_name], [exp_x])
    sp     = n("mish_sp");   tracer.add_node("Add",      [exp_x, one], [sp])
    ln_sp  = n("mish_ln");   tracer.add_node("Log",      [sp],         [ln_sp])
    tanh_  = n("mish_tanh"); tracer.add_node("Tanh",     [ln_sp],      [tanh_])
    tracer.add_node("Mul", [input_name, tanh_], [output_name])
    return output_name


@register_tracer(nl.PRelu)
def trace_prelu(tracer, layer, input_name, output_name):
    import lemon.numlib as nm
    slope = nm.as_numpy(layer.slope.data)
    slope_name = tracer.unique_name("prelu_slope")
    tracer.add_parameter(slope_name, slope)
    tracer.add_node("PRelu", [input_name, slope_name], [output_name])
    return output_name


@register_tracer(nl.Softplus)
def trace_softplus(tracer, layer, input_name, output_name):
    tracer.add_node("Softplus", [input_name], [output_name])
    return output_name


@register_tracer(nl.Softsign)
def trace_softsign(tracer, layer, input_name, output_name):
    tracer.add_node("Softsign", [input_name], [output_name])
    return output_name


@register_tracer(nl.ThresholdedRelu)
def trace_thresholded_relu(tracer, layer, input_name, output_name):
    tracer.add_node("ThresholdedRelu", [input_name], [output_name],
                    {"alpha": getattr(layer, "alpha", 1.0)})
    return output_name
