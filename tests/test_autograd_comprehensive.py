"""
test_autograd_comprehensive.py

自動微分の包括的テストスイート
- 全ての微分可能な型と演算子のテスト
- 複雑な式での動作確認
- autograd.off時の動作確認
- 勾配の数値的検証
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from lemon.numlib import *
import pytest


# =============================================================================
# ヘルパー関数
# =============================================================================


def numerical_gradient(f, x, eps=1e-5):
    """
    数値微分で勾配を計算

    Parameters
    ----------
    f : callable
        スカラー値を返す関数
    x : NumType
        入力
    eps : float
        微小変化量

    Returns
    -------
    ndarray
        数値微分で計算した勾配
    """
    x_data = x._data.copy()
    grad = np.zeros_like(x_data)

    it = np.nditer(x_data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index

        # f(x + eps)
        x_data[idx] += eps
        x_plus = type(x)(x_data, requires_grad=False)
        f_plus = float(f(x_plus)._data)

        # f(x - eps)
        x_data[idx] -= 2 * eps
        x_minus = type(x)(x_data, requires_grad=False)
        f_minus = float(f(x_minus)._data)

        # 中心差分
        grad[idx] = (f_plus - f_minus) / (2 * eps)

        # 元に戻す
        x_data[idx] += eps
        it.iternext()

    return grad


def check_gradient(f, x, rtol=1e-3, atol=1e-5):
    """
    自動微分と数値微分の勾配を比較

    Parameters
    ----------
    f : callable
        スカラー値を返す関数
    x : NumType
        入力（requires_grad=True）
    rtol : float
        相対許容誤差
    atol : float
        絶対許容誤差

    Returns
    -------
    bool
        勾配が一致すればTrue
    """
    # 自動微分
    x.zero_grad()
    y = f(x)
    y.backward()
    auto_grad = x.grad._data

    # 数値微分
    num_grad = numerical_gradient(f, x)

    # 比較
    return np.allclose(auto_grad, num_grad, rtol=rtol, atol=atol)


# =============================================================================
# テスト: 基本的な算術演算
# =============================================================================


class TestBasicArithmetic:
    """基本的な算術演算のテスト"""

    def test_addition(self):
        """加算の勾配テスト"""
        x = real(3.0)
        y = real(2.0)

        z = x + y
        z.backward()

        assert x.grad._data == 1.0, "dx should be 1"
        assert y.grad._data == 1.0, "dy should be 1"

    def test_subtraction(self):
        """減算の勾配テスト"""
        x = real(5.0)
        y = real(2.0)

        z = x - y
        z.backward()

        assert x.grad._data == 1.0, "dx should be 1"
        assert y.grad._data == -1.0, "dy should be -1"

    def test_multiplication(self):
        """乗算の勾配テスト"""
        x = real(3.0)
        y = real(4.0)

        z = x * y
        z.backward()

        assert x.grad._data == 4.0, "dx should be y"
        assert y.grad._data == 3.0, "dy should be x"

    def test_division(self):
        """除算の勾配テスト"""
        x = real(6.0)
        y = real(2.0)

        z = x / y
        z.backward()

        assert np.isclose(x.grad._data, 0.5), "dx should be 1/y"
        assert np.isclose(y.grad._data, -1.5), "dy should be -x/y^2"

    def test_power(self):
        """べき乗の勾配テスト"""
        x = real(2.0)

        # x^2
        z = x**2
        z.backward()
        assert np.isclose(x.grad._data, 4.0), "d(x^2)/dx should be 2x"

        # x^3
        x.zero_grad()
        z = x**3
        z.backward()
        assert np.isclose(x.grad._data, 12.0), "d(x^3)/dx should be 3x^2"

    def test_negation(self):
        """符号反転の勾配テスト"""
        x = real(5.0)

        z = -x
        z.backward()

        assert x.grad._data == -1.0, "d(-x)/dx should be -1"


# =============================================================================
# テスト: 数学関数
# =============================================================================


class TestMathFunctions:
    """数学関数のテスト"""

    def test_exp(self):
        """指数関数の勾配テスト"""
        x = real(2.0)

        y = exp(x)
        y.backward()

        expected = np.exp(2.0)
        assert np.isclose(x.grad._data, expected), f"d(exp(x))/dx should be exp(x)"

    def test_log(self):
        """対数関数の勾配テスト"""
        x = real(2.0)

        y = log(x)
        y.backward()

        assert np.isclose(x.grad._data, 0.5), "d(log(x))/dx should be 1/x"

    def test_sqrt(self):
        """平方根の勾配テスト"""
        x = real(4.0)

        y = sqrt(x)
        y.backward()

        assert np.isclose(x.grad._data, 0.25), "d(sqrt(x))/dx should be 1/(2*sqrt(x))"

    def test_sin(self):
        """正弦関数の勾配テスト"""
        x = real(1.0)

        y = sin(x)
        y.backward()

        expected = np.cos(1.0)
        assert np.isclose(x.grad._data, expected), "d(sin(x))/dx should be cos(x)"

    def test_cos(self):
        """余弦関数の勾配テスト"""
        x = real(1.0)

        y = cos(x)
        y.backward()

        expected = -np.sin(1.0)
        assert np.isclose(x.grad._data, expected), "d(cos(x))/dx should be -sin(x)"

    def test_tan(self):
        """正接関数の勾配テスト"""
        x = real(0.5)

        y = tan(x)
        y.backward()

        expected = 1.0 / (np.cos(0.5) ** 2)
        assert np.isclose(x.grad._data, expected), "d(tan(x))/dx should be sec^2(x)"

    def test_tanh(self):
        """双曲線正接の勾配テスト"""
        x = real(0.5)

        y = tanh(x)
        y.backward()

        tanh_val = np.tanh(0.5)
        expected = 1.0 - tanh_val**2
        assert np.isclose(x.grad._data, expected), "d(tanh(x))/dx should be 1-tanh^2(x)"


# =============================================================================
# テスト: テンソル演算
# =============================================================================


class TestTensorOperations:
    """テンソル演算のテスト"""

    def test_vector_addition(self):
        """ベクトル加算の勾配テスト"""
        x = vector([1.0, 2.0, 3.0])
        y = vector([4.0, 5.0, 6.0])

        z = x + y
        loss = sum(z)
        loss.backward()

        assert np.allclose(x.grad._data, np.ones((3, 1))), "dx should be ones"
        assert np.allclose(y.grad._data, np.ones((3, 1))), "dy should be ones"

    def test_matrix_multiplication(self):
        """行列積の勾配テスト"""
        A = matrix([[1.0, 2.0], [3.0, 4.0]])
        B = matrix([[5.0, 6.0], [7.0, 8.0]])

        C = A @ B
        loss = sum(C)
        loss.backward()

        # dL/dA = dL/dC @ B^T
        expected_dA = np.ones((2, 2)) @ B._data.T
        assert np.allclose(A.grad._data, expected_dA), "Gradient of A is incorrect"

        # dL/dB = A^T @ dL/dC
        expected_dB = A._data.T @ np.ones((2, 2))
        assert np.allclose(B.grad._data, expected_dB), "Gradient of B is incorrect"

    def test_element_wise_multiplication(self):
        """要素ごとの乗算の勾配テスト"""
        x = vector([2.0, 3.0, 4.0])
        y = vector([5.0, 6.0, 7.0])

        z = x * y
        loss = sum(z)
        loss.backward()

        assert np.allclose(x.grad._data, y._data), "dx should be y"
        assert np.allclose(y.grad._data, x._data), "dy should be x"

    def test_sum_reduction(self):
        """和の削減の勾配テスト"""
        x = matrix([[1.0, 2.0], [3.0, 4.0]])

        y = sum(x)
        y.backward()

        assert np.allclose(x.grad._data, np.ones((2, 2))), "Gradient should be all ones"

    def test_mean_reduction(self):
        """平均の削減の勾配テスト"""
        x = matrix([[2.0, 4.0], [6.0, 8.0]])

        y = mean(x)
        y.backward()

        expected = np.ones((2, 2)) / 4
        assert np.allclose(x.grad._data, expected), "Gradient should be 1/n"

    def test_reshape(self):
        """リシェイプの勾配テスト"""
        x = vector([1.0, 2.0, 3.0, 4.0])

        y = reshape(x, (2, 2))
        loss = sum(y)
        loss.backward()

        assert np.allclose(x.grad._data, np.ones((4, 1))), (
            "Gradient shape should match input"
        )

    def test_transpose(self):
        """転置の勾配テスト"""
        x = matrix([[1.0, 2.0], [3.0, 4.0]])

        y = x.T
        loss = sum(y)
        loss.backward()

        assert np.allclose(x.grad._data, np.ones((2, 2))), (
            "Gradient should flow through transpose"
        )


# =============================================================================
# テスト: 複雑な式
# =============================================================================


class TestComplexExpressions:
    """複雑な式のテスト"""

    def test_polynomial(self):
        """多項式: f(x) = 3x^3 + 2x^2 - 5x + 1"""
        x = real(2.0)

        y = 3 * x**3 + 2 * x**2 - 5 * x + 1
        y.backward()

        # df/dx = 9x^2 + 4x - 5 = 9(4) + 4(2) - 5 = 36 + 8 - 5 = 39
        expected = 39.0
        assert np.isclose(x.grad._data, expected), (
            f"Polynomial gradient should be {expected}"
        )

    def test_chain_rule(self):
        """連鎖律: f(x) = sin(x^2)"""
        x = real(1.0)

        y = sin(x**2)
        y.backward()

        # df/dx = cos(x^2) * 2x
        expected = np.cos(1.0) * 2.0
        assert np.isclose(x.grad._data, expected), "Chain rule gradient is incorrect"

    def test_nested_operations(self):
        """入れ子の演算: f(x) = exp(log(x^2 + 1))"""
        x = real(2.0)

        y = exp(log(x**2 + 1))
        y.backward()

        # f(x) = x^2 + 1, so df/dx = 2x
        expected = 4.0
        assert np.isclose(x.grad._data, expected, rtol=1e-3), (
            "Nested operations gradient is incorrect"
        )

    def test_multi_variable_function(self):
        """多変数関数: f(x,y) = x^2*y + y^3"""
        x = real(3.0)
        y = real(2.0)

        z = x**2 * y + y**3
        z.backward()

        # df/dx = 2xy = 2(3)(2) = 12
        # df/dy = x^2 + 3y^2 = 9 + 12 = 21
        assert np.isclose(x.grad._data, 12.0), "df/dx should be 2xy"
        assert np.isclose(y.grad._data, 21.0), "df/dy should be x^2 + 3y^2"

    def test_neural_network_like(self):
        """ニューラルネットワーク風: σ(Wx + b)"""
        # 入力
        x = vector([1.0, 2.0])

        # パラメータ
        W = matrix([[0.5, -0.3], [0.2, 0.8]])
        b = vector([0.1, -0.1])

        # 順伝播
        z = W @ x + b
        a = tanh(z)
        loss = sum(a)

        # 逆伝播
        loss.backward()

        # 勾配が計算されているか確認
        assert x.grad is not None, "Input gradient should be computed"
        assert W.grad is not None, "Weight gradient should be computed"
        assert b.grad is not None, "Bias gradient should be computed"

        # 形状確認
        assert x.grad.shape == x.shape
        assert W.grad.shape == W.shape
        assert b.grad.shape == b.shape


# =============================================================================
# テスト: 数値微分との比較
# =============================================================================


class TestNumericalGradient:
    """数値微分との比較テスト"""

    def test_simple_function(self):
        """単純な関数: f(x) = x^2"""
        x = vector([1.0, 2.0, 3.0])

        def f(x):
            return sum(x**2)

        assert check_gradient(f, x), "Gradient check failed for x^2"

    def test_complex_function(self):
        """複雑な関数: f(x) = sum(sin(x) * exp(x))"""
        x = vector([0.5, 1.0, 1.5])

        def f(x):
            return sum(sin(x) * exp(x))

        assert check_gradient(f, x), "Gradient check failed for complex function"

    def test_matrix_function(self):
        """行列関数: f(A) = sum(A @ A^T)"""
        A = matrix([[1.0, 2.0], [3.0, 4.0]])

        def f(A):
            return sum(A @ A.T)

        assert check_gradient(f, A), "Gradient check failed for matrix function"


# =============================================================================
# テスト: autograd.off の動作
# =============================================================================


class TestAutogradOff:
    """autograd.off時の動作テスト"""

    def test_autograd_off_context(self):
        """autograd.offコンテキストマネージャーのテスト"""
        x = real(2.0, requires_grad=True)

        # autograd on
        y1 = x**2
        assert y1.requires_grad, "Should require grad by default"

        # autograd off
        with autograd.off:
            y2 = x**2
            assert not y2.requires_grad, "Should not require grad in autograd.off"

        # autograd on (restored)
        y3 = x**2
        assert y3.requires_grad, "Should require grad after context exit"

    def test_no_gradient_computation(self):
        """autograd.off時に勾配が計算されないことを確認"""
        x = real(3.0, requires_grad=True)

        with autograd.off:
            y = x**3 + 2 * x**2 - x + 5
            assert not y.requires_grad, "Result should not require grad"

            # backwardを呼んでもエラーが出る
            try:
                y.backward()
                assert False, "Should raise RuntimeError"
            except RuntimeError:
                pass  # Expected

    def test_mixed_operations(self):
        """autograd on/offを混在させたテスト"""
        x = real(2.0)

        # on
        y1 = x**2

        # off
        with autograd.off:
            y2 = y1 + x
            assert not y2.requires_grad, "Should not require grad"

        # on again
        y3 = y1 * real(3.0)
        y3.backward()

        # y1の勾配は計算されるが、y2には影響しない
        assert x.grad is not None, "Gradient should be computed"


# =============================================================================
# テスト: requires_grad=False
# =============================================================================


class TestRequiresGradFalse:
    """requires_grad=Falseのテスト"""

    def test_no_grad_input(self):
        """requires_grad=Falseの入力"""
        x = real(2.0, requires_grad=False)

        y = x**2 + 3 * x

        # ✅ yもrequires_grad=Falseになる
        assert not y.requires_grad, "Result should not require grad"

        # ✅ backward呼び出しはエラー
        try:
            y.backward()
            assert False, "Should raise RuntimeError"
        except RuntimeError as e:
            assert "does not require gradients" in str(e)

    def test_mixed_requires_grad(self):
        """requires_gradの混在"""
        x = real(2.0, requires_grad=True)
        y = real(3.0, requires_grad=False)

        z = x * y
        z.backward()

        # xの勾配のみ計算される
        assert x.grad is not None, "x should have gradient"
        assert y.grad is None, "y should not have gradient"

    def test_all_false_requires_grad(self):
        """全てrequires_grad=Falseの場合"""
        with autograd.off:
            x = real(2.0)
            y = real(3.0)

            z = x * y + x**2

            assert not z.requires_grad, "Result should not require grad"


# =============================================================================
# テスト: エッジケース
# =============================================================================


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_zero_gradient(self):
        """定数に対する勾配のテスト"""
        # ✅ autograd.offで定数を作成
        with autograd.off:
            y = real(10.0)

        # yに依存しない変数xを作成
        x = real(5.0)

        # yだけに依存する計算
        z = y**2

        # ✅ zはrequires_grad=False（yがFalseなので）
        assert not z.requires_grad, "Constant operation should not require grad"

        # ✅ xに依存する計算
        w = x**2
        assert w.requires_grad, "Should require grad"

    def test_independent_variables(self):
        """独立変数のテスト"""
        x = real(2.0)
        y = real(3.0)

        # xだけに依存
        z1 = x**2
        z1.backward()

        assert x.grad is not None, "x should have gradient"
        assert y.grad is None, "y should not have gradient (independent)"

    def test_multiple_backward(self):
        """複数回のbackward呼び出し"""
        x = real(2.0)

        y = x**2
        y.backward()

        grad1 = x.grad._data.copy()

        # 2回目
        y = x**2
        y.backward()

        grad2 = x.grad._data

        # 勾配が蓄積される
        assert np.isclose(grad2, grad1 * 2), "Gradients should accumulate"

    def test_zero_grad(self):
        """zero_gradのテスト"""
        x = real(3.0)

        y = x**2
        y.backward()

        assert x.grad is not None

        x.zero_grad()

        assert x.grad is None, "Gradient should be None after zero_grad"

    def test_broadcast_gradient(self):
        """ブロードキャストの勾配テスト"""
        x = vector([1.0, 2.0, 3.0])
        y = real(2.0)

        z = x * y
        loss = sum(z)
        loss.backward()

        # yの勾配はxの要素の和
        expected = np.sum(x._data)
        assert np.isclose(y.grad._data, expected), "Broadcast gradient incorrect"

    def test_detached_computation(self):
        """切り離された計算のテスト"""
        x = real(2.0)
        y = x**2

        # yのデータだけを使って新しい計算（勾配は流れない）
        with autograd.off:
            z = real(y._data)

        w = z * 3

        assert not w.requires_grad, "Detached tensor should not require grad"


# =============================================================================
# テストスイート実行
# =============================================================================


def run_all_tests():
    """全テストを実行"""
    import pytest

    # pytestを使って実行
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
