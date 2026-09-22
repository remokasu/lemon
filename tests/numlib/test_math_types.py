import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon import numlib as nm
import numpy as np


class TestTensorStaysTensor:
    """Tensor の演算結果に、形だけで数学の型をつけないことのテスト"""

    @pytest.mark.parametrize(
        "op",
        [
            lambda t: t + t,
            lambda t: t * 2.0,
            lambda t: nm.exp(t),
            lambda t: t.T,
            lambda t: t @ nm.tensor(np.ones((4, 1))),
            lambda t: nm.sum(t, axis=1, keepdims=True),
            lambda t: t[0:1],
            lambda t: nm.reshape(t, (12, 1)),
            lambda t: t + np.ones((3, 4)),
        ],
        ids=["add", "scalar_mul", "exp", "transpose", "matmul", "sum_keepdims",
             "slice", "reshape", "add_ndarray"],
    )
    def test_result_is_tensor(self, op):
        t = nm.tensor(np.ones((3, 4)))
        assert type(op(t)) is nm.Tensor

    def test_implicit_2d_conversion_is_tensor(self):
        """2次元の ndarray / list を暗黙に変換しても Matrix にならない"""
        assert type(nm._auto_convert(np.ones((2, 3)))) is nm.Tensor
        assert type(nm._auto_convert([[1.0, 2.0], [3.0, 4.0]])) is nm.Tensor


class TestMathTypesPreserved:
    """明示した数学の型どうしの演算で、型が数学のルールどおりになることのテスト"""

    def test_matrix_elementwise(self):
        m = nm.matrix(np.ones((3, 4)))
        assert isinstance(m + m, nm.Matrix)
        assert isinstance(nm.exp(m), nm.Matrix)

    def test_scalar_multiplication(self):
        v = nm.vector([1.0, 2.0, 3.0])
        assert isinstance(nm.real(2.0) * v, nm.Vector)
        assert isinstance(v * 2.0, nm.Vector)

    def test_n_by_1_matrix_is_vector(self):
        """n×1 行列は列ベクトルそのもの"""
        m = nm.matrix(np.ones((3, 4)))
        assert isinstance(m @ nm.matrix(np.ones((4, 1))), nm.Vector)


class TestOrientation:
    """数学の型から1次元の結果が出る演算で、向きを保つことのテスト"""

    def setup_method(self):
        self.m = nm.matrix(np.arange(12.0).reshape(3, 4))

    def test_row_is_row_vector(self):
        """A[i] は行ベクトル eᵢᵀA"""
        for row in (self.m[1], self.m[1, :], self.m[-1]):
            assert isinstance(row, nm.RowVector)
            assert row.shape == (1, 4)
        np.testing.assert_array_equal(self.m[1]._data, [[4, 5, 6, 7]])

    def test_column_is_vector(self):
        """A[:, j] は列ベクトル Aeⱼ"""
        col = self.m[:, 2]
        assert isinstance(col, nm.Vector)
        np.testing.assert_array_equal(col._data, [[2], [6], [10]])

    def test_partial_row_is_row_vector(self):
        assert isinstance(self.m[1, 1:3], nm.RowVector)

    def test_vector_component_is_scalar(self):
        """ベクトルの成分はスカラー"""
        v = nm.vector([10.0, 20.0, 30.0])
        r = nm.rowvector([10.0, 20.0, 30.0])
        assert isinstance(v[1], nm.Scalar) and float(v[1]._data) == 20.0
        assert isinstance(r[1], nm.Scalar) and float(r[1]._data) == 20.0

    def test_vector_slice_keeps_type(self):
        assert isinstance(nm.vector([1.0, 2.0, 3.0])[1:], nm.Vector)
        assert isinstance(nm.rowvector([1.0, 2.0, 3.0])[1:], nm.RowVector)

    @pytest.mark.parametrize("fn", [nm.sum, nm.mean, nm.amax, nm.amin])
    def test_reduction_orientation(self, fn):
        """1ᵀA は行ベクトル、A1 は列ベクトル"""
        assert isinstance(fn(self.m, axis=0), nm.RowVector)
        assert isinstance(fn(self.m, axis=1), nm.Vector)

    def test_vector_reduction_is_scalar(self):
        """ベクトルの成分の和はスカラー"""
        assert isinstance(nm.sum(nm.vector([1.0, 2.0, 3.0]), axis=0), nm.Scalar)

    def test_row_gradient(self):
        """行を取り出したときの勾配が正しい位置に入る"""
        m = nm.matrix(np.arange(6.0).reshape(2, 3))
        nm.sum(m[1] * 2.0).backward()
        np.testing.assert_array_equal(m.grad._data, [[0, 0, 0], [2, 2, 2]])


class TestGradientType:
    """勾配は変数と同じ空間の元なので、同じ型になることのテスト"""

    def test_vector_gradient(self):
        v = nm.vector([1.0, 2.0, 3.0])
        nm.sum(nm.exp(v)).backward()
        assert isinstance(v.grad, nm.Vector)

    def test_matrix_gradient(self):
        m = nm.matrix(np.ones((2, 3)))
        nm.sum(m * m).backward()
        assert isinstance(m.grad, nm.Matrix)

    def test_tensor_gradient(self):
        t = nm.tensor(np.ones(3), requires_grad=True)
        nm.sum(t * t).backward()
        assert type(t.grad) is nm.Tensor


class TestDotIsInnerProduct:
    """dot はベクトルどうしの内積で、いつもスカラーになることのテスト"""

    MAKERS = {
        "Vector": nm.vector,
        "RowVector": nm.rowvector,
        "Tensor": lambda d: nm.tensor(d),
    }

    @pytest.mark.parametrize("xt", MAKERS)
    @pytest.mark.parametrize("yt", MAKERS)
    def test_dot_value_and_gradient(self, xt, yt):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        x = self.MAKERS[xt](a.copy())
        y = self.MAKERS[yt](b.copy())
        x.requires_grad = True
        y.requires_grad = True

        z = nm.dot(x, y)
        assert z.shape == ()
        assert float(z._data) == pytest.approx(32.0)

        z.backward()
        assert x.grad.shape == x.shape and y.grad.shape == y.shape
        np.testing.assert_allclose(np.ravel(x.grad._data), b)
        np.testing.assert_allclose(np.ravel(y.grad._data), a)
