import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon import numlib as nm
import numpy as np


def _t(*shape):
    return nm.tensor(np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape))


class TestElementwiseRules:
    """
    要素ごとの演算は、数学的に定義される組み合わせだけを許す。
    形が (n₁, …, n_k) の Tensor は ℝ^(n₁×…×n_k) の元。和は同じ空間の元どうしだけ。
    """

    # ---- 定義される演算 ----

    @pytest.mark.parametrize(
        "op",
        [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a * b,  # アダマール積
            lambda a, b: a / b,
            lambda a, b: nm.maximum(a, b),
            lambda a, b: a**b,
        ],
        ids=["add", "sub", "hadamard", "div", "maximum", "pow"],
    )
    def test_same_shape_is_defined(self, op):
        assert op(_t(2, 3), _t(2, 3)).shape == (2, 3)

    @pytest.mark.parametrize(
        "op",
        [
            lambda x: 2.0 * x,
            lambda x: x * 2.0,
            lambda x: x / 2.0,
            lambda x: 1.0 / x,  # 成分ごとの逆数のスカラー倍
            lambda x: nm.maximum(x, 0),  # 成分ごとの関数（ReLU）
            lambda x: x**2,
            lambda x: 2**x,
        ],
        ids=["scalar_mul_left", "scalar_mul_right", "div_by_scalar", "scalar_div",
             "maximum_scalar", "pow_scalar", "scalar_pow"],
    )
    def test_scalar_multiplication_and_componentwise_maps(self, op):
        assert op(_t(3)).shape == (3,)

    def test_scalar_plus_scalar(self):
        assert float((nm.real(1.0) + nm.real(2.0))._data) == 3.0

    def test_math_types_of_same_space(self):
        v = nm.vector([1.0, 2.0])
        m = nm.matrix(np.ones((2, 2)))
        assert isinstance(v + v, nm.Vector)
        assert isinstance(m * m, nm.Matrix)

    # ---- 定義されない演算 ----

    @pytest.mark.parametrize(
        "op",
        [lambda x: x + 1.0, lambda x: 1.0 + x, lambda x: x - 1.0, lambda x: 1.0 - x],
        ids=["x+c", "c+x", "x-c", "c-x"],
    )
    def test_scalar_addition_is_undefined(self, op):
        with pytest.raises(nm.TypeMismatchError):
            op(_t(3))

    @pytest.mark.parametrize(
        "a_shape, b_shape",
        [((2, 3), (3,)), ((3,), (3, 1)), ((1, 3, 1), (2, 1, 4)), ((2, 3), (2, 1))],
    )
    def test_implicit_broadcasting_is_undefined(self, a_shape, b_shape):
        for op in (lambda a, b: a + b, lambda a, b: a * b, nm.maximum):
            with pytest.raises(nm.DimensionError):
                op(_t(*a_shape), _t(*b_shape))

    def test_tensor_and_math_type_do_not_mix(self):
        with pytest.raises(nm.TypeMismatchError):
            nm.tensor(np.ones((3, 1))) + nm.vector([1.0, 2.0, 3.0])

    def test_vector_and_row_vector_do_not_add(self):
        with pytest.raises(nm.DimensionError):
            nm.vector([1.0, 2.0, 3.0]) + nm.rowvector([1.0, 2.0, 3.0])

    # ---- 明示的に書けば計算できる ----

    def test_explicit_broadcast_and_constant_vector(self):
        X = nm.tensor(np.ones((2, 3)), requires_grad=True)
        b = nm.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        c = nm.real(5.0, requires_grad=True)

        y = X + nm.broadcast_to(b, X.shape) + c * nm.ones_like(X)
        np.testing.assert_allclose(y._data, [[7, 8, 9], [7, 8, 9]])

        nm.sum(y).backward()
        np.testing.assert_allclose(b.grad._data, [2, 2, 2])
        assert float(c.grad._data) == 6.0

    def test_ones_like_is_constant_of_same_type(self):
        for x in (nm.tensor(np.ones(3), requires_grad=True), nm.vector([1.0, 2.0])):
            one = nm.ones_like(x)
            assert type(one) is type(x)
            assert one.requires_grad is False


class TestLiterals:
    """リテラルは、形がまったく同じときだけ相手と同じ空間の元として読む"""

    def test_list_literal_adopts_matrix_space(self):
        m = nm.matrix([[1.0, 2.0], [3.0, 4.0]])
        for r in (m + [[1, 2], [3, 4]], [[1, 2], [3, 4]] + m, m ** [[1, 2], [3, 4]]):
            assert isinstance(r, nm.Matrix)

    def test_literal_with_different_shape_is_undefined(self):
        with pytest.raises(nm.TypeMismatchError):
            nm.vector([1.0, 2.0]) + np.ones(2)  # (2,) は (2, 1) と別の空間

    def test_explicit_tensor_does_not_adopt(self):
        with pytest.raises(nm.TypeMismatchError):
            nm.matrix(np.ones((2, 2))) + nm.tensor(np.ones((2, 2)))


class TestDifferentiableCast:
    """型の変換は恒等写像なので、勾配がそのまま（形を戻して）流れる"""

    @pytest.mark.parametrize(
        "make, cast",
        [
            (lambda: nm.tensor(np.array([1.0, 2.0]), requires_grad=True), nm.vector),
            (lambda: nm.vector([1.0, 2.0]), nm.tensor),
            (lambda: nm.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True), nm.matrix),
            (lambda: nm.vector([1.0, 2.0]), nm.rowvector),
        ],
        ids=["tensor_to_vector", "vector_to_tensor", "tensor_to_matrix", "vector_to_rowvector"],
    )
    def test_gradient_flows_through_cast(self, make, cast):
        x = make()
        y = cast(x)
        nm.sum(y * y).backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        np.testing.assert_allclose(np.ravel(x.grad._data), 2 * np.ravel(x._data))

    def test_explicit_requires_grad_makes_new_leaf(self):
        x = nm.tensor(np.ones(3), requires_grad=True)
        y = nm.tensor(x, requires_grad=True)
        nm.sum(y).backward()
        assert x.grad is None


class TestNumpyUfuncs:
    """NumPy の関数を通しても、微分できる演算に回り、形も変わらない"""

    @pytest.mark.parametrize(
        "f, expected",
        [
            (lambda W: np.ones((4, 2)) @ W, 4.0),
            (lambda W: np.maximum(np.zeros((2, 3)), W), 1.0),
            (lambda W: np.ones((2, 3)) * W, 1.0),
            (lambda W: np.ones((2, 3)) - W, -1.0),
        ],
        ids=["matmul", "maximum", "multiply", "subtract"],
    )
    def test_gradient_flows(self, f, expected):
        W = nm.tensor(np.ones((2, 3)), requires_grad=True)
        nm.sum(f(W)).backward()
        np.testing.assert_allclose(W.grad._data, expected)

    def test_non_differentiable_ufunc_keeps_shape(self):
        y = np.floor(nm.tensor(np.array([1.5, -2.5, 3.5])))
        assert type(y) is nm.Tensor
        assert y.shape == (3,)
