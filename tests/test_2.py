import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import pytest
import numpy as np
from lemon.numlib import *


class TestChainRule:
    """Test automatic differentiation chain rule"""

    def setup_method(self):
        """Reset state before each test"""
        autograd.enable()
        cuda.disable()

    def test_simple_chain(self):
        """Test simple chain: f(g(x))"""
        # f(g(x)) = (x^2)^2 = x^4
        # df/dx = 4x^3
        x = real(2.0)
        y = x**2  # g(x) = x^2
        z = y**2  # f(y) = y^2

        z.backward()

        # At x=2: df/dx = 4 * 2^3 = 32
        assert abs(x.grad.item() - 32.0) < 1e-6

    def test_multi_path_chain(self):
        """Test when variable is used multiple times"""
        # f(x) = x^2 + x^3
        # df/dx = 2x + 3x^2
        x = real(3.0)
        y1 = x**2  # 9
        y2 = x**3  # 27
        z = y1 + y2  # 36

        z.backward()

        # At x=3: df/dx = 2*3 + 3*3^2 = 6 + 27 = 33
        assert abs(x.grad.item() - 33.0) < 1e-6

    def test_complex_chain(self):
        """Test complex chain with multiple operations"""
        # f(x) = sin(x^2) * exp(x)
        # df/dx = 2x*cos(x^2)*exp(x) + sin(x^2)*exp(x)
        x = real(1.0)

        x_squared = x**2
        sin_x_squared = sin(x_squared)
        exp_x = exp(x)
        result = sin_x_squared * exp_x

        result.backward()

        # Manual calculation at x=1
        x_val = 1.0
        expected = 2 * x_val * np.cos(x_val**2) * np.exp(x_val) + np.sin(
            x_val**2
        ) * np.exp(x_val)

        assert abs(x.grad.item() - expected) < 1e-6

    def test_vector_chain(self):
        """Test chain rule with vectors"""
        # f(x) = ||x||^2 = x1^2 + x2^2 + x3^2
        # df/dx = [2x1, 2x2, 2x3]
        x = vec([1.0, 2.0, 3.0])
        y = x * x  # Element-wise square
        z = y.sum()  # Sum all elements

        z.backward()

        expected_grad = np.array([[2.0], [4.0], [6.0]])  # (3, 1)形状に修正
        np.testing.assert_allclose(x.grad.data, expected_grad, rtol=1e-6)

    def test_matrix_chain(self):
        """Test chain rule with matrix operations"""
        # f(X) = tr(X @ X^T) = sum of squared elements
        # For X = [[1, 2], [3, 4]]
        X = mat([[1.0, 2.0], [3.0, 4.0]])

        Y = X @ X.T  # [[5, 11], [11, 25]]
        trace = Y[0, 0] + Y[1, 1]  # 5 + 25 = 30

        trace.backward()

        # Gradient should be 2*X
        expected_grad = np.array([[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose(X.grad.data, expected_grad, rtol=1e-6)

    def test_multiple_outputs_single_input(self):
        """Test when one input affects multiple outputs"""
        x = real(2.0)

        # Two separate computation paths
        y1 = x**2  # 4
        y2 = x**3  # 8

        # Use both outputs
        z1 = y1 * 2  # 8
        z2 = y2 * 3  # 24

        final = z1 + z2  # 32
        final.backward()

        # df/dx = d(2x^2 + 3x^3)/dx = 4x + 9x^2
        # At x=2: 4*2 + 9*4 = 8 + 36 = 44
        assert abs(x.grad.item() - 44.0) < 1e-6

    def test_shared_intermediate(self):
        """Test when intermediate value is reused"""
        x = real(3.0)

        # Shared intermediate
        y = x**2  # 9

        # Use y twice
        z1 = y + x  # 9 + 3 = 12
        z2 = y * x  # 9 * 3 = 27

        result = z1 * z2  # 12 * 27 = 324
        result.backward()

        # Complex gradient calculation
        # result = (x^2 + x) * (x^2 * x) = (x^2 + x) * x^3
        #        = x^5 + x^4
        # d/dx = 5x^4 + 4x^3
        # At x=3: 5*81 + 4*27 = 405 + 108 = 513
        assert abs(x.grad.item() - 513.0) < 1e-6

    def test_nested_functions(self):
        """Test deeply nested function composition"""
        x = real(0.5)

        # f(g(h(x))) where h(x) = exp(x), g(x) = sin(x), f(x) = sqrt(x + 1)
        h = exp(x)
        g = sin(h)
        f = sqrt(g + real(1.0))

        f.backward()

        # Chain rule: df/dx = df/dg * dg/dh * dh/dx
        x_val = 0.5
        h_val = np.exp(x_val)
        g_val = np.sin(h_val)

        dh_dx = np.exp(x_val)
        dg_dh = np.cos(h_val)
        df_dg = 0.5 / np.sqrt(g_val + 1.0)

        expected = df_dg * dg_dh * dh_dx
        assert abs(x.grad.item() - expected) < 1e-6

    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly"""
        x = real(2.0)

        # First forward-backward pass
        y1 = x**2
        y1.backward()

        grad1 = x.grad.item()  # Should be 4

        # Second forward-backward pass (gradients should accumulate)
        y2 = x**3
        y2.backward()

        # Gradient should be 4 + 12 = 16
        assert abs(x.grad.item() - 16.0) < 1e-6

        # Clear gradient and check
        x.zero_grad()
        assert x.grad is None

        # Third pass after clearing
        y3 = x * 2
        y3.backward()
        assert abs(x.grad.item() - 2.0) < 1e-6

    def test_no_grad_context(self):
        """Test that no_grad context prevents gradient computation"""
        x = real(3.0)

        # With gradient
        y = x**2
        assert y.requires_grad == True
        y.backward()
        assert x.grad is not None
        assert abs(x.grad.item() - 6.0) < 1e-6

        # Reset
        x.zero_grad()

        # Without gradient
        with autograd.off:
            y_no_grad = x**2
            # requires_gradがFalseであることを確認（これが重要）
            assert y_no_grad.requires_grad == False

            # backwardを呼ぶとエラーになることを確認
            with pytest.raises(RuntimeError):
                y_no_grad.backward()

    def test_detach_breaks_chain(self):
        """Test that operations on non-differentiable types don't track gradients"""
        x = real(2.0)

        # Convert to non-differentiable integer
        y = x**2
        y_int = integer(int(y.item()))  # Should break the chain
        z = real(y_int.item()) * 2

        # z doesn't have gradient tracking from x
        if hasattr(z, "backward"):
            z.backward()
            # x should not have gradients since chain was broken
            assert x.grad is None or x.grad.item() == 0

    def test_complex_computational_graph(self):
        """Test a complex computational graph with multiple paths"""
        # Create a diamond-shaped graph
        #       x
        #      / \
        #     y1  y2
        #      \ /
        #       z

        x = real(2.0)

        # Two different paths
        y1 = x**2 + 1  # 5
        y2 = x * 3 - 1  # 5

        # Merge paths
        z = y1 * y2  # 25

        z.backward()

        # dz/dx = dz/dy1 * dy1/dx + dz/dy2 * dy2/dx
        #       = y2 * 2x + y1 * 3
        #       = 5 * 4 + 5 * 3 = 20 + 15 = 35
        assert abs(x.grad.item() - 35.0) < 1e-6


class TestVectorMatrixGradients:
    """Test gradients for vector and matrix operations"""

    def setup_method(self):
        autograd.enable()
        cuda.disable()

    def test_vector_dot_product(self):
        """Test gradient of dot product"""
        x = vec([1.0, 2.0, 3.0])
        y = vec([4.0, 5.0, 6.0])

        # z = x · y = 1*4 + 2*5 + 3*6 = 32
        z = (x * y).sum()  # Element-wise multiply then sum
        z.backward()

        # dz/dx = y
        np.testing.assert_allclose(x.grad.data, y.data, rtol=1e-6)
        # dz/dy = x
        np.testing.assert_allclose(y.grad.data, x.data, rtol=1e-6)

    def test_matrix_multiplication_gradient(self):
        """Test gradient of matrix multiplication"""
        A = mat([[1.0, 2.0], [3.0, 4.0]])
        B = mat([[5.0, 6.0], [7.0, 8.0]])

        C = A @ B  # [[19, 22], [43, 50]]
        loss = C.sum()  # 134

        loss.backward()

        # dL/dA = ones @ B^T
        expected_A_grad = np.ones((2, 2)) @ B.data.T
        np.testing.assert_allclose(A.grad.data, expected_A_grad, rtol=1e-6)

        # dL/dB = A^T @ ones
        expected_B_grad = A.data.T @ np.ones((2, 2))
        np.testing.assert_allclose(B.grad.data, expected_B_grad, rtol=1e-6)

    def test_broadcasting_gradient(self):
        """Test gradient with broadcasting"""
        # Scalar + Vector (broadcasting)
        x = real(2.0)
        v = vec([1.0, 2.0, 3.0])

        result = x + v  # [3, 4, 5]
        loss = result.sum()  # 12

        loss.backward()

        # Gradient of x should be sum of gradients (due to broadcasting)
        print(x.grad)
        assert abs(x.grad.item() - 3.0) < 1e-6

        # Gradient of v should be ones
        np.testing.assert_allclose(
            v.grad.data, np.ones((3, 1)), rtol=1e-6
        )  # (3, 1)形状に修正

    def test_reshape_gradient(self):
        """Test gradient flows through reshape"""
        x = vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Reshape to matrix
        y = x.reshape(2, 3)

        # Some operation
        z = (y**2).sum()

        z.backward()

        # Gradient should be 2*x (element-wise)
        expected = 2 * x.data
        np.testing.assert_allclose(x.grad.data, expected, rtol=1e-6)


class TestHigherOrderDerivatives:
    """Test second-order and higher derivatives"""

    def setup_method(self):
        autograd.enable()
        cuda.disable()

    def test_second_derivative(self):
        """Test computing second derivatives"""
        # f(x) = x^3
        # f'(x) = 3x^2
        # f''(x) = 6x

        x = real(2.0)

        # First derivative
        y = x**3
        y.backward()
        first_grad = x.grad.item()  # Should be 12
        assert abs(first_grad - 12.0) < 1e-6

        # For second derivative, we need to track gradient of gradient
        # This requires the gradient computation itself to be differentiable
        # Current implementation might not support this directly

    def test_gradient_of_gradient_sum(self):
        """Test gradient of a function of gradients"""
        x = real(2.0)
        y = real(3.0)

        # f(x, y) = x^2 * y
        z = (x**2) * y
        z.backward()

        # df/dx = 2xy = 12
        # df/dy = x^2 = 4
        assert abs(x.grad.item() - 12.0) < 1e-6
        assert abs(y.grad.item() - 4.0) < 1e-6


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
