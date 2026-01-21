import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon.numlib import *


def numerical_gradient(f, x, eps=1e-5):
    """
    数値微分で勾配を計算

    Parameters
    ----------
    f : callable
        関数
    x : float
        評価点
    eps : float
        微小量

    Returns
    -------
    float
        数値微分による勾配
    """
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_add():
    """加算の勾配テスト"""
    x0 = real(1.0, requires_grad=True)
    x1 = real(2.0, requires_grad=True)
    y = x0 + x1
    y.backward()

    # d(x0 + x1)/dx0 = 1
    # d(x0 + x1)/dx1 = 1
    assert np.isclose(float(x0.grad), 1.0)
    assert np.isclose(float(x1.grad), 1.0)


def test_sub():
    """減算の勾配テスト"""
    x0 = real(1.0, requires_grad=True)
    x1 = real(2.0, requires_grad=True)
    y = x0 - x1
    y.backward()

    # d(x0 - x1)/dx0 = 1
    # d(x0 - x1)/dx1 = -1
    assert np.isclose(float(x0.grad), 1.0)
    assert np.isclose(float(x1.grad), -1.0)


def test_mul():
    """乗算の勾配テスト"""
    x0_val = 1.0
    x1_val = 2.0

    x0 = real(x0_val, requires_grad=True)
    x1 = real(x1_val, requires_grad=True)
    y = x0 * x1
    y.backward()

    # d(x0 * x1)/dx0 = x1
    # d(x0 * x1)/dx1 = x0
    assert np.isclose(float(x0.grad), x1_val)
    assert np.isclose(float(x1.grad), x0_val)

    # 数値微分との比較
    def f0(x):
        return x * x1_val

    def f1(x):
        return x0_val * x

    grad0_numerical = numerical_gradient(f0, x0_val)
    grad1_numerical = numerical_gradient(f1, x1_val)

    assert np.isclose(float(x0.grad), grad0_numerical, rtol=1e-4)
    assert np.isclose(float(x1.grad), grad1_numerical, rtol=1e-4)


def test_div():
    """除算の勾配テスト"""
    x0_val = 1.0
    x1_val = 2.0

    x0 = real(x0_val, requires_grad=True)
    x1 = real(x1_val, requires_grad=True)
    y = x0 / x1
    y.backward()

    # d(x0 / x1)/dx0 = 1/x1
    # d(x0 / x1)/dx1 = -x0/x1^2
    expected_grad0 = 1.0 / x1_val
    expected_grad1 = -x0_val / (x1_val**2)

    assert np.isclose(float(x0.grad), expected_grad0)
    assert np.isclose(float(x1.grad), expected_grad1)

    # 数値微分との比較
    def f0(x):
        return x / x1_val

    def f1(x):
        return x0_val / x

    grad0_numerical = numerical_gradient(f0, x0_val)
    grad1_numerical = numerical_gradient(f1, x1_val)

    assert np.isclose(float(x0.grad), grad0_numerical, rtol=1e-4)
    assert np.isclose(float(x1.grad), grad1_numerical, rtol=1e-4)


def test_pow():
    """累乗の勾配テスト"""
    x0_val = 3.0
    x1_val = 2.0

    x0 = real(x0_val, requires_grad=True)
    x1 = real(x1_val, requires_grad=True)
    y = x0**x1
    y.backward()

    # d(x0^x1)/dx0 = x1 * x0^(x1-1)
    # d(x0^x1)/dx1 = x0^x1 * log(x0)
    expected_grad0 = x1_val * (x0_val ** (x1_val - 1))
    expected_grad1 = (x0_val**x1_val) * np.log(x0_val)

    assert np.isclose(float(x0.grad), expected_grad0, rtol=1e-4)
    assert np.isclose(float(x1.grad), expected_grad1, rtol=1e-4)

    # 数値微分との比較
    def f0(x):
        return x**x1_val

    def f1(x):
        return x0_val**x

    grad0_numerical = numerical_gradient(f0, x0_val)
    grad1_numerical = numerical_gradient(f1, x1_val)

    assert np.isclose(float(x0.grad), grad0_numerical, rtol=1e-4)
    assert np.isclose(float(x1.grad), grad1_numerical, rtol=1e-4)


def test_log():
    """対数の勾配テスト"""
    x0_val = 3.0

    x0 = real(x0_val, requires_grad=True)
    y = log(x0)
    y.backward()

    # d log(x)/dx = 1/x
    expected_grad = 1.0 / x0_val
    assert np.isclose(float(x0.grad), expected_grad)

    # 数値微分との比較
    def f(x):
        return np.log(x)

    grad_numerical = numerical_gradient(f, x0_val)
    assert np.isclose(float(x0.grad), grad_numerical, rtol=1e-4)


def test_sqrt():
    """平方根の勾配テスト"""
    x0_val = 10.0

    x0 = real(x0_val, requires_grad=True)
    y = sqrt(x0)
    y.backward()

    # d sqrt(x)/dx = 1/(2*sqrt(x))
    expected_grad = 1.0 / (2 * np.sqrt(x0_val))
    assert np.isclose(float(x0.grad), expected_grad)

    # 数値微分との比較
    def f(x):
        return np.sqrt(x)

    grad_numerical = numerical_gradient(f, x0_val)
    assert np.isclose(float(x0.grad), grad_numerical, rtol=1e-4)


def test_exp():
    """指数関数の勾配テスト"""
    x0_val = 10.0

    x0 = real(x0_val, requires_grad=True)
    y = exp(x0)
    y.backward()

    # d exp(x)/dx = exp(x)
    expected_grad = np.exp(x0_val)
    assert np.isclose(float(x0.grad), expected_grad)

    # 数値微分との比較
    def f(x):
        return np.exp(x)

    grad_numerical = numerical_gradient(f, x0_val)
    assert np.isclose(float(x0.grad), grad_numerical, rtol=1e-4)


def test_sin():
    """正弦の勾配テスト"""
    x0_val = 10.0

    x0 = real(x0_val, requires_grad=True)
    y = sin(x0)
    y.backward()

    # d sin(x)/dx = cos(x)
    expected_grad = np.cos(x0_val)
    assert np.isclose(float(x0.grad), expected_grad)

    # 数値微分との比較
    def f(x):
        return np.sin(x)

    grad_numerical = numerical_gradient(f, x0_val)
    assert np.isclose(float(x0.grad), grad_numerical, rtol=1e-4)


def test_cos():
    """余弦の勾配テスト"""
    x0_val = 10.0

    x0 = real(x0_val, requires_grad=True)
    y = cos(x0)
    y.backward()

    # d cos(x)/dx = -sin(x)
    expected_grad = -np.sin(x0_val)
    assert np.isclose(float(x0.grad), expected_grad)

    # 数値微分との比較
    def f(x):
        return np.cos(x)

    grad_numerical = numerical_gradient(f, x0_val)
    assert np.isclose(float(x0.grad), grad_numerical, rtol=1e-4)


def test_tan():
    """正接の勾配テスト"""
    x0_val = 10.0

    x0 = real(x0_val, requires_grad=True)
    y = tan(x0)
    y.backward()

    # d tan(x)/dx = 1/cos^2(x) = sec^2(x)
    expected_grad = 1.0 / (np.cos(x0_val) ** 2)
    assert np.isclose(float(x0.grad), expected_grad)

    # 数値微分との比較
    def f(x):
        return np.tan(x)

    grad_numerical = numerical_gradient(f, x0_val)
    assert np.isclose(float(x0.grad), grad_numerical, rtol=1e-4)


def test_tanh():
    """双曲線正接の勾配テスト"""
    x0_val = 10.0

    x0 = real(x0_val, requires_grad=True)
    y = tanh(x0)
    y.backward()

    # d tanh(x)/dx = 1 - tanh^2(x)
    expected_grad = 1.0 - np.tanh(x0_val) ** 2
    assert np.isclose(float(x0.grad), expected_grad)

    # 数値微分との比較
    def f(x):
        return np.tanh(x)

    grad_numerical = numerical_gradient(f, x0_val)
    assert np.isclose(float(x0.grad), grad_numerical, rtol=1e-4)


def test_log2():
    """底2の対数の勾配テスト"""
    x0_val = 10.0

    x0 = real(x0_val, requires_grad=True)
    y = log2(x0)
    y.backward()

    # d log2(x)/dx = 1/(x * ln(2))
    expected_grad = 1.0 / (x0_val * np.log(2))
    assert np.isclose(float(x0.grad), expected_grad, rtol=1e-4)

    # 数値微分との比較
    def f(x):
        return np.log2(x)

    grad_numerical = numerical_gradient(f, x0_val)
    assert np.isclose(float(x0.grad), grad_numerical, rtol=1e-4)


def test_log10():
    """底10の対数の勾配テスト"""
    x0_val = 10.0

    x0 = real(x0_val, requires_grad=True)
    y = log10(x0)
    y.backward()

    # d log10(x)/dx = 1/(x * ln(10))
    expected_grad = 1.0 / (x0_val * np.log(10))
    assert np.isclose(float(x0.grad), expected_grad, rtol=1e-4)

    # 数値微分との比較
    def f(x):
        return np.log10(x)

    grad_numerical = numerical_gradient(f, x0_val)
    assert np.isclose(float(x0.grad), grad_numerical, rtol=1e-4)


def test_all_gradients():
    """全ての勾配テストを実行"""
    print("\n" + "=" * 60)
    print("Testing Gradients with Numerical Differentiation")
    print("=" * 60)

    tests = [
        ("Addition", test_add),
        ("Subtraction", test_sub),
        ("Multiplication", test_mul),
        ("Division", test_div),
        ("Power", test_pow),
        ("Logarithm", test_log),
        ("Square Root", test_sqrt),
        ("Exponential", test_exp),
        ("Sine", test_sin),
        ("Cosine", test_cos),
        ("Tangent", test_tan),
        ("Hyperbolic Tangent", test_tanh),
        ("Log2", test_log2),
        ("Log10", test_log10),
    ]

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name} gradient test passed")
        except AssertionError as e:
            print(f"✗ {name} gradient test FAILED: {e}")
            raise

    print("\n" + "=" * 60)
    print("All gradient tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_all_gradients()
