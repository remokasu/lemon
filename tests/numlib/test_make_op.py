import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon import numlib as nm
import numpy as np


def _square_forward(x):
    return x * x, x


def _square_backward(x, grad, needs_grad):
    return (grad * 2 * x,)


square = nm.make_op(_square_forward, _square_backward)


def _scaled_add_forward(x, y, scale):
    return scale * x + y, None


def _scaled_add_backward(ctx, grad, needs_grad):
    # ctx は使わない。needs_grad が False の入力には None を返してよい
    return (grad * 2.0 if needs_grad[0] else None, grad if needs_grad[1] else None)


scaled_add = nm.make_op(_scaled_add_forward, _scaled_add_backward)


class TestMakeOp:
    """公開 API make_op のテスト"""

    def test_value_and_gradient(self):
        x = nm.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = square(x)
        np.testing.assert_allclose(y._data, [1.0, 4.0, 9.0])
        nm.sum(y).backward()
        np.testing.assert_allclose(x.grad._data, [2.0, 4.0, 6.0])

    def test_params_are_passed_as_keywords(self):
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        y = nm.tensor([10.0, 20.0], requires_grad=True)
        z = scaled_add(x, y, scale=2.0)
        np.testing.assert_allclose(z._data, [12.0, 24.0])
        nm.sum(z).backward()
        np.testing.assert_allclose(x.grad._data, [2.0, 2.0])
        np.testing.assert_allclose(y.grad._data, [1.0, 1.0])

    def test_only_inputs_requiring_grad_get_gradients(self):
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        y = nm.tensor([10.0, 20.0], requires_grad=False)
        nm.sum(scaled_add(x, y, scale=2.0)).backward()
        assert x.grad is not None
        assert y.grad is None

    def test_none_input_is_allowed(self):
        """省略できる入力（bias など）に None を渡せる"""

        def forward(x, b):
            return (x if b is None else x + b), None

        def backward(ctx, grad, needs_grad):
            return grad, (grad if needs_grad[1] else None)

        op = nm.make_op(forward, backward)
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        nm.sum(op(x, None)).backward()
        np.testing.assert_allclose(x.grad._data, [1.0, 1.0])

    def test_raw_input_is_constant(self):
        """NumType でない入力は定数として扱う"""
        assert square(np.array([1.0, 2.0])).requires_grad is False

    def test_autograd_off(self):
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        with nm.autograd.off:
            assert square(x).requires_grad is False

    def test_accumulate_onto_readonly_gradient(self):
        """勾配が read-only の view から先に入っていても、累積で落ちない"""
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        (nm.sum(square(x)) + nm.sum(x)).backward()
        np.testing.assert_allclose(x.grad._data, [3.0, 5.0])

    def test_math_type_is_preserved(self):
        v = nm.vector([1.0, 2.0, 3.0])
        assert isinstance(square(v), nm.Vector)
        nm.sum(square(v)).backward()
        assert isinstance(v.grad, nm.Vector)

    def test_wrong_gradient_shape_raises(self):
        bad = nm.make_op(lambda x: (x.sum(), None), lambda c, g, n: (np.ones(3),))
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(nm.GradientError):
            bad(x).backward()

    def test_wrong_number_of_gradients_raises(self):
        bad = nm.make_op(lambda x: (x * 1.0, None), lambda c, g, n: (g, g))
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(nm.GradientError):
            nm.sum(bad(x)).backward()
