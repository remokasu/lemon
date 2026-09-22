import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon import numlib as nm
from lemon import nnlib as nl
import numpy as np


class TestImplicitConversion1D:
    """NumType でない1次元の入力を暗黙に変換しても、形が変わらないことのテスト"""

    @pytest.mark.parametrize(
        "other",
        [np.arange(3.0), [0.0, 1.0, 2.0], (0.0, 1.0, 2.0)],
        ids=["ndarray", "list", "tuple"],
    )
    def test_auto_convert_keeps_shape(self, other):
        """1次元の入力は (n,) の Tensor になる"""
        x = nm._auto_convert(other)
        assert type(x) is nm.Tensor
        assert x.shape == (3,)

    @pytest.mark.parametrize(
        "other",
        [np.arange(3.0), [0.0, 1.0, 2.0], (0.0, 1.0, 2.0)],
        ids=["ndarray", "list", "tuple"],
    )
    @pytest.mark.parametrize(
        "op",
        [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b],
        ids=["add", "sub", "mul"],
    )
    def test_binary_op_keeps_shape(self, op, other):
        """Tensor (n,) と1次元の入力の演算は (n,) になる（(n, n) にならない）"""
        t = nm.tensor(np.arange(3.0))
        assert op(t, other).shape == (3,)

    def test_reflected_op_keeps_shape(self):
        """ndarray が左にあっても (n,) になる"""
        t = nm.tensor(np.arange(3.0))
        assert (np.arange(3.0) + t).shape == (3,)

    def test_value_and_gradient(self):
        """値と勾配が正しい"""
        x = nm.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = nm.sum(x * np.ones(3))
        y.backward()

        assert float(y.data) == pytest.approx(6.0)
        np.testing.assert_allclose(x.grad.data, [1.0, 1.0, 1.0])

    def test_unary_op_keeps_shape(self):
        """単項演算に1次元の ndarray を渡しても (n,) になる"""
        assert nm.exp(np.arange(3.0)).shape == (3,)


class TestLossWithNdarrayTarget:
    """正解ラベルを ndarray で渡しても、損失の値が正しいことのテスト"""

    def setup_method(self):
        self.pred_np = np.array([0.2, 0.7, 0.9])
        self.target_np = np.array([0.0, 1.0, 1.0])

    def test_mse_loss(self):
        pred = nm.tensor(self.pred_np)
        expected = np.mean((self.pred_np - self.target_np) ** 2)
        loss = nl.MSELoss()(pred, self.target_np)
        assert float(loss.data) == pytest.approx(expected)

    def test_bce_loss(self):
        pred = nm.tensor(self.pred_np)
        p, t = self.pred_np, self.target_np
        expected = -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
        loss = nl.BCELoss()(pred, self.target_np)
        assert float(loss.data) == pytest.approx(expected, rel=1e-6)


class TestRawInputIsConstant:
    """NumType でない入力（ndarray など）は定数なので、勾配を追跡しないことのテスト"""

    @pytest.mark.parametrize(
        "fn",
        [nm.exp, nm.sin, nm.sqrt, nm.tanh, nm.abs,
         nm.sum, nm.mean, lambda a: nm.reshape(a, (3, 1)),
         lambda a: nm.get_item(a, 0), lambda a: nm.clip(a, 0, 1)],
        ids=["exp", "sin", "sqrt", "tanh", "abs", "sum", "mean", "reshape",
             "get_item", "clip"],
    )
    def test_raw_ndarray_does_not_require_grad(self, fn):
        assert fn(np.arange(1.0, 4.0)).requires_grad is False

    def test_tensor_input_still_requires_grad(self):
        t = nm.tensor(np.arange(1.0, 4.0), requires_grad=True)
        assert nm.exp(t).requires_grad is True
        assert nm.sum(t).requires_grad is True
