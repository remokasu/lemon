from lemon.nnlib import Module
import lemon.numlib as nm


def sigmoid(x):
    """
    Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))
    """
    return 1 / (1 + nm.exp(-x))


class Sigmoid(Module):
    """
    Sigmoid activation module (wrapper for Sequential)

    For direct usage, prefer sigmoid(x).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sigmoid(x)

    def __repr__(self):
        return "Sigmoid()"
