"""
Comprehensive test suite for the numerical computation library
"""

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
import numpy as np
from lemon.numlib import *


# ==============================
# GPU Tests
# ==============================


import pytest
from lemon.numlib import cuda, autograd, mat, vec, real


class TestCudaNamespace:
    """Test suite for CUDA namespace"""

    def test_cuda_enable_disable(self):
        """Test GPU enable/disable functionality"""
        if cuda.is_available():
            # enable/disable メソッド
            cuda.enable()
            assert cuda.is_enabled() == True

            cuda.disable()
            assert cuda.is_enabled() == False

            # gpu/cpu ショートカット
            cuda.gpu()
            assert cuda.is_enabled() == True

            cuda.cpu()
            assert cuda.is_enabled() == False
        else:
            # GPUが利用できない場合
            with pytest.raises(RuntimeError):
                cuda.enable()

            with pytest.raises(RuntimeError):
                cuda.gpu()

            # CPUは常に動作
            cuda.cpu()
            assert cuda.is_enabled() == False

    def test_cuda_context_manager(self):
        """Test CUDA context manager functionality"""
        if cuda.is_available():
            # GPU context
            cuda.disable()
            with cuda.gpu:
                assert cuda.is_enabled() == True
            assert cuda.is_enabled() == False

            # CPU context
            cuda.enable()
            with cuda.cpu:
                assert cuda.is_enabled() == False
            assert cuda.is_enabled() == True

    def test_cuda_device_info(self):
        """Test CUDA device information methods"""
        if cuda.is_available():
            cuda.enable()

            # Device count
            count = cuda.device_count()
            assert isinstance(count, int)
            assert count >= 1

            # Current device
            device_id = cuda.current_device()
            assert isinstance(device_id, int)
            assert device_id >= 0

            # Memory info
            mem_info = cuda.memory_info()
            assert isinstance(mem_info, dict)
            assert "used" in mem_info
            assert "total" in mem_info

            cuda.disable()
            assert cuda.current_device() == -1
        else:
            assert cuda.device_count() == 0
            assert cuda.current_device() == -1

            mem_info = cuda.memory_info()
            assert mem_info == {"error": "GPU not enabled"}


class TestAutogradNamespace:
    """Test suite for Autograd namespace"""

    def test_autograd_enable_disable(self):
        """Test gradient computation enable/disable"""
        # enable/disable メソッド
        autograd.enable()
        assert autograd.is_enabled() == True

        autograd.disable()
        assert autograd.is_enabled() == False

        # on/off ショートカット
        autograd.on()
        assert autograd.is_enabled() == True

        autograd.off()
        assert autograd.is_enabled() == False

    def test_autograd_context_manager(self):
        """Test autograd context manager functionality"""
        # on context
        autograd.disable()
        with autograd.on:
            assert autograd.is_enabled() == True
        assert autograd.is_enabled() == False

        # off context
        autograd.enable()
        with autograd.off:
            assert autograd.is_enabled() == False
        assert autograd.is_enabled() == True

    def test_autograd_with_tensors(self):
        """Test autograd affects tensor creation"""
        autograd.on()
        x = real(2.0)
        assert x.requires_grad == True

        with autograd.off:
            y = real(3.0)
            # Note: 現在の実装では、scalar作成時は
            # is_grad_enabled() をチェックするように修正が必要かも

        autograd.off()
        z = vec([1, 2, 3])
        # requires_grad のデフォルト値をチェック


class TestCudaAutogradIntegration:
    """Integration tests for CUDA and Autograd namespaces"""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset to known state before and after each test"""
        # Setup
        cuda.disable()
        autograd.enable()

        yield

        # Teardown
        cuda.disable()
        autograd.enable()

    def test_initial_state(self):
        """Test that initial state is as expected"""
        assert cuda.is_enabled() == False
        assert autograd.is_enabled() == True

    def test_function_calls(self):
        """Test function call style"""
        # Test autograd
        autograd.off()
        assert autograd.is_enabled() == False

        autograd.on()
        assert autograd.is_enabled() == True

        # Test cuda if available
        if cuda.is_available():
            cuda.gpu()
            assert cuda.is_enabled() == True

            cuda.cpu()
            assert cuda.is_enabled() == False

    def test_context_managers(self):
        """Test context manager style"""
        # Autograd context
        with autograd.off:
            assert autograd.is_enabled() == False
        assert autograd.is_enabled() == True

        with autograd.on:
            assert autograd.is_enabled() == True
        assert autograd.is_enabled() == True

        # CUDA context if available
        if cuda.is_available():
            with cuda.gpu:
                assert cuda.is_enabled() == True
            assert cuda.is_enabled() == False

            with cuda.cpu:
                assert cuda.is_enabled() == False
            assert cuda.is_enabled() == False

    def test_combined_contexts(self):
        """Test combining CUDA and autograd contexts"""
        if not cuda.is_available():
            pytest.skip("CUDA not available")

        # Start from known state
        assert cuda.is_enabled() == False
        assert autograd.is_enabled() == True

        # Combined context
        with cuda.gpu, autograd.off:
            assert cuda.is_enabled() == True
            assert autograd.is_enabled() == False

        # Both should be restored
        assert cuda.is_enabled() == False
        assert autograd.is_enabled() == True


# ==============================
# Scalar Type Tests
# ==============================


def test_boolean_creation():
    """Test Boolean scalar creation"""
    b1 = boolean(True)
    b2 = boolean(False)
    assert isinstance(b1, Boolean)
    assert isinstance(b2, Boolean)
    assert b1.item() == True
    assert b2.item() == False


def test_integer_creation():
    """Test Integer scalar creation with different kinds"""
    i8 = int8(10)
    i16 = int16(100)
    i32 = int32(1000)
    i64 = int64(10000)

    assert isinstance(i8, Integer)
    assert i8.kind == 8
    assert i8.signed == True

    u8 = uint8(10)
    assert u8.signed == False


def test_real_creation():
    """Test Real scalar creation"""
    r16 = real16(1.5)
    r32 = real32(2.5)
    r64 = real64(3.5)

    assert isinstance(r64, Real)
    assert r64.kind == 64
    assert abs(r64.item() - 3.5) < 1e-10


def test_complex_creation():
    """Test Complex scalar creation"""
    c1 = cmplx(1, 2)
    c2 = cmplx64(3, 4)
    c3 = cmplx128(5, 6)

    assert isinstance(c1, Complex)
    assert c1.real.item() == 1.0
    assert c1.imag.item() == 2.0


# ==============================
# Scalar Arithmetic Tests
# ==============================


def test_scalar_addition():
    """Test scalar addition and type promotion"""
    i = integer(5)
    r = real(2.5)

    result = i + r
    assert isinstance(result, Real)
    assert abs(result.item() - 7.5) < 1e-10


def test_scalar_division():
    """Test scalar division promotes to Real"""
    i1 = integer(10)
    i2 = integer(3)

    result = i1 / i2
    assert isinstance(result, Real)
    assert abs(result.item() - 3.333333) < 1e-5


def test_scalar_type_promotion():
    """Test type promotion hierarchy: Boolean < Integer < Real < Complex"""
    b = boolean(True)
    i = integer(2)
    r = real(3.0)
    c = cmplx(4, 1)

    # Boolean + Integer -> Integer
    result1 = b + i
    assert isinstance(result1, Integer)

    # Integer + Real -> Real
    result2 = i + r
    assert isinstance(result2, Real)

    # Real + Complex -> Complex
    result3 = r + c
    assert isinstance(result3, Complex)


def test_scalar_comparison():
    """Test scalar comparison operations"""
    i1 = integer(5)
    i2 = integer(10)

    assert i1 < i2
    assert i2 > i1
    assert i1 <= i2
    assert i2 >= i1
    assert i1 != i2
    assert i1 == integer(5)


# ==============================
# Integer Bitwise Operations Tests
# ==============================


def test_integer_bitwise_and():
    """Test bitwise AND operation"""
    i1 = integer(12)  # 1100 in binary
    i2 = integer(10)  # 1010 in binary

    result = i1 & i2  # Should be 1000 = 8
    assert isinstance(result, Integer)
    assert result.item() == 8


def test_integer_bitwise_or():
    """Test bitwise OR operation"""
    i1 = integer(12)  # 1100
    i2 = integer(10)  # 1010

    result = i1 | i2  # Should be 1110 = 14
    assert result.item() == 14


def test_integer_bitwise_xor():
    """Test bitwise XOR operation"""
    i1 = integer(12)  # 1100
    i2 = integer(10)  # 1010

    result = i1 ^ i2  # Should be 0110 = 6
    assert result.item() == 6


def test_integer_shift_operations():
    """Test left and right shift operations"""
    i = integer(4)

    left = i << integer(2)  # 4 << 2 = 16
    right = i >> integer(1)  # 4 >> 1 = 2

    assert left.item() == 16
    assert right.item() == 2


def test_integer_invert():
    """Test bitwise NOT operation"""
    i = int8(5)  # 00000101
    result = ~i  # Should invert all bits

    assert isinstance(result, Integer)
    # Result depends on bit width and signedness


def test_integer_binary_representation():
    """Test bin, oct, hex properties"""
    i = integer(255)

    assert "11111111" in i.bin
    assert i.oct == oct(255)
    assert i.hex == hex(255)


# ==============================
# Vector Tests
# ==============================


def test_vector_creation():
    """Test Vector creation"""
    v1 = vector([1, 2, 3])
    v2 = vec([4, 5, 6])

    assert isinstance(v1, Vector)
    assert isinstance(v2, Vector)
    assert v1.shape == (3, 1)


def test_vector_transpose():
    """Test Vector transpose to RowVector"""
    v = vector([1, 2, 3])
    rv = v.T

    assert isinstance(rv, RowVector)
    assert (rv.T._data == v._data).all()


def test_vector_arithmetic():
    """Test basic vector arithmetic"""
    v1 = vec([1, 2, 3])
    v2 = vec([4, 5, 6])

    # Addition
    result = v1 + v2
    assert np.allclose(result._data, [[5], [7], [9]])  # (3, 1)形状

    # Subtraction
    result = v1 - v2
    assert np.allclose(result._data, [[-3], [-3], [-3]])  # (3, 1)形状


def test_vector_dot_product():
    """Test Vector inner product"""
    v1 = vector([1, 2, 3])
    v2 = vector([4, 5, 6])

    # Inner product: v1.T @ v2
    result = v1.T @ v2
    assert isinstance(result, Scalar)
    assert result.item() == 32  # 1*4 + 2*5 + 3*6


def test_vector_outer_product():
    """Test Vector outer product"""
    v1 = vector([1, 2, 3])
    v2 = vector([4, 5, 6])

    # Outer product: v1 @ v2.T
    result = v1 @ v2.T
    assert isinstance(result, Matrix)
    assert result.shape == (3, 3)


def test_vector_invalid_operations():
    """Test that invalid Vector operations raise errors"""
    v1 = vector([1, 2, 3])
    v2 = vector([4, 5, 6])

    # Cannot multiply column vector with column vector
    with pytest.raises(ValueError):
        v1 @ v2


# ==============================
# RowVector Tests
# ==============================


def test_rowvector_creation():
    """Test RowVector creation"""
    rv1 = rowvector([1, 2, 3])
    rv2 = rowvec([4, 5, 6])

    assert isinstance(rv1, RowVector)
    assert isinstance(rv2, RowVector)


def test_rowvector_transpose():
    """Test RowVector transpose to Vector"""
    rv = rowvector([1, 2, 3])
    v = rv.T

    assert isinstance(v, Vector)
    assert (v.T._data == rv._data).all()


def test_rowvector_vector_product():
    """Test RowVector @ Vector (inner product)"""
    rv = rowvector([1, 2, 3])
    v = vector([4, 5, 6])

    result = rv @ v
    assert isinstance(result, Scalar)
    assert result.item() == 32


def test_rowvector_invalid_operations():
    """Test invalid RowVector operations"""
    rv1 = rowvector([1, 2, 3])
    rv2 = rowvector([4, 5, 6])

    # Cannot multiply row vector with row vector
    with pytest.raises(ValueError):
        rv1 @ rv2


# ==============================
# Matrix Tests
# ==============================


def test_matrix_creation():
    """Test Matrix creation"""
    m1 = matrix([[1, 2], [3, 4]])
    m2 = mat([[5, 6], [7, 8]])

    assert isinstance(m1, Matrix)
    assert isinstance(m2, Matrix)
    assert m1.shape == (2, 2)


def test_matrix_transpose():
    """Test Matrix transpose"""
    m = matrix([[1, 2, 3], [4, 5, 6]])
    mt = m.T

    assert isinstance(mt, Matrix)
    assert mt.shape == (3, 2)
    assert mt._data[0, 0] == 1
    assert mt._data[1, 0] == 2


def test_matrix_vector_multiply():
    """Test Matrix @ Vector"""
    m = matrix([[1, 2], [3, 4]])
    v = vector([1, 2])

    result = m @ v
    assert isinstance(result, Vector)
    assert result._data.tolist() == [[5], [11]]


def test_matrix_matrix_multiply():
    """Test Matrix @ Matrix"""
    m1 = matrix([[1, 2], [3, 4]])
    m2 = matrix([[5, 6], [7, 8]])

    result = m1 @ m2
    assert isinstance(result, Matrix)
    assert result.shape == (2, 2)
    assert result._data[0, 0] == 19


def test_rowvector_matrix_multiply():
    """Test RowVector @ Matrix"""
    rv = rowvector([1, 2])
    m = matrix([[1, 2, 3], [4, 5, 6]])

    result = rv @ m
    assert isinstance(result, RowVector)
    assert result._data.tolist() == [[9, 12, 15]]


def test_matrix_indexing():
    """Test Matrix indexing returns correct types"""
    m = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Single element -> Scalar
    elem = m[0, 0]
    assert isinstance(elem, Scalar)
    assert elem.item() == 1

    # Row -> Vector
    row = m[0]
    assert isinstance(row, Tensor)
    assert row._data.tolist() == [1, 2, 3]

    # Submatrix -> Matrix
    submat = m[0:2, 0:2]
    assert isinstance(submat, Matrix)


def test_matrix_invalid_operations():
    """Test invalid Matrix operations"""
    m = matrix([[1, 2], [3, 4]])
    rv = rowvector([1, 2])
    v = vector([1, 2])

    # Cannot multiply matrix with row vector
    with pytest.raises(ValueError):
        m @ rv

    # Cannot left-multiply matrix by column vector
    with pytest.raises(ValueError):
        v @ m


# ==============================
# Tensor Tests
# ==============================


def test_tensor_creation():
    """Test Tensor creation"""
    t1 = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    t2 = ten([1, 2, 3, 4])

    assert isinstance(t1, Tensor)
    assert t1.shape == (2, 2, 2)
    assert t2.shape == (4,)


def test_tensor_arithmetic():
    """Test Tensor arithmetic"""
    t1 = tensor([[1, 2], [3, 4]])
    t2 = tensor([[5, 6], [7, 8]])

    result = t1 + t2
    assert isinstance(result, Tensor)


def test_tensor_reshape():
    """Test Tensor reshape returns correct types"""
    t = tensor([1, 2, 3, 4, 5, 6])

    # Reshape to matrix
    m = t.reshape(2, 3)
    assert isinstance(m, Matrix)
    assert m.shape == (2, 3)

    # Reshape to vector
    v = m.reshape(6)
    assert isinstance(v, Tensor)


# ==============================
# Mathematical Functions Tests
# ==============================


def test_exp_function():
    """Test exponential function"""
    r = real(1.0)
    result = exp(r)

    assert isinstance(result, Real)
    assert abs(result.item() - np.e) < 1e-10


def test_log_function():
    """Test logarithm function"""
    r = real(np.e)
    result = log(r)

    assert isinstance(result, Real)
    assert abs(result.item() - 1.0) < 1e-10


def test_sqrt_function():
    """Test square root function"""
    r = real(4.0)
    result = sqrt(r)

    assert isinstance(result, Real)
    assert abs(result.item() - 2.0) < 1e-10


def test_trig_functions():
    """Test trigonometric functions"""
    r = real(0.0)

    assert abs(sin(r).item() - 0.0) < 1e-10
    assert abs(cos(r).item() - 1.0) < 1e-10
    assert abs(tan(r).item() - 0.0) < 1e-10


def test_inverse_trig_functions():
    """Test inverse trigonometric functions"""
    r = real(0.5)

    result = asin(r)
    assert isinstance(result, Real)

    result2 = acos(r)
    assert isinstance(result2, Real)

    result3 = atan(r)
    assert isinstance(result3, Real)


def test_hyperbolic_functions():
    """Test hyperbolic functions"""
    r = real(1.0)

    sinh_result = sinh(r)
    cosh_result = cosh(r)
    tanh_result = tanh(r)

    assert isinstance(sinh_result, Real)
    assert isinstance(cosh_result, Real)
    assert isinstance(tanh_result, Real)


def test_math_functions_preserve_type():
    """Test that math functions preserve Vector/Matrix types"""
    v = vector([1.0, 2.0, 3.0])
    result = sin(v)

    assert isinstance(result, Vector)
    assert result.shape == v.shape

    m = matrix([[1.0, 2.0], [3.0, 4.0]])
    result2 = exp(m)

    assert isinstance(result2, Matrix)
    assert result2.shape == m.shape


def test_maximum_minimum():
    """Test maximum and minimum functions"""
    v1 = vector([1, 5, 3])
    v2 = vector([4, 2, 6])

    max_result = maximum(v1, v2)
    assert isinstance(max_result, Vector)
    assert max_result._data.tolist() == [[4], [5], [6]]

    min_result = minimum(v1, v2)
    assert min_result._data.tolist() == [[1], [2], [3]]


def test_abs_function():
    """Test absolute value function"""
    v = vector([-1, 2, -3])
    result = abs(v)

    assert isinstance(result, Vector)
    assert result._data.tolist() == [[1], [2], [3]]

    # Complex abs returns Real
    c = cmplx(3, 4)
    result2 = abs(c)
    assert isinstance(result2, Real)
    assert abs(result2.item() - 5.0) < 1e-10


# ==============================
# Utility Functions Tests
# ==============================


def test_dot_function():
    """Test dot product utility function"""
    v1 = vector([1, 2, 3])
    v2 = vector([4, 5, 6])

    result = dot(v1, v2)
    assert isinstance(result, Scalar)
    assert result.item() == 32


def test_reshape_function():
    """Test reshape utility function"""
    v = vector([1, 2, 3, 4])
    m = reshape(v, 2, 2)

    assert isinstance(m, Matrix)
    assert m.shape == (2, 2)


def test_transpose_function():
    """Test transpose utility function"""
    v = vector([1, 2, 3])
    rv = transpose(v)

    assert isinstance(rv, RowVector)

    m = matrix([[1, 2], [3, 4]])
    mt = transpose(m)

    assert isinstance(mt, Matrix)
    assert mt.shape == (2, 2)


def test_mean_function():
    """Test mean utility function"""
    v = vector([1.0, 2.0, 3.0, 4.0])
    result = mean(v)

    assert isinstance(result, Real)
    assert abs(result.item() - 2.5) < 1e-10


# ==============================
# Factory Functions Tests
# ==============================


def test_zeros_creation():
    """Test zeros factory functions"""
    t = zeros(2, 3)
    assert isinstance(t, Tensor)
    assert t.shape == (2, 3)
    assert (t._data == 0).all()

    v = zeros_vector(5)
    assert isinstance(v, Vector)
    assert v.shape == (5, 1)

    m = zeros_matrix((3, 4))
    assert isinstance(m, Matrix)
    assert m.shape == (3, 4)


def test_ones_creation():
    """Test ones factory functions"""
    t = ones(2, 3)
    assert isinstance(t, Tensor)
    assert (t._data == 1).all()

    v = ones_vector(5)
    assert isinstance(v, Vector)

    m = ones_matrix((3, 4))
    assert isinstance(m, Matrix)


def test_random_creation():
    """Test random factory functions"""
    t = rand(2, 3)
    assert isinstance(t, Tensor)
    assert t.shape == (2, 3)

    v = random_vector(5)
    assert isinstance(v, Vector)

    m = random_matrix((3, 4))
    assert isinstance(m, Matrix)


def test_randn_creation():
    """Test randn (normal distribution)"""
    t = randn(100)
    assert isinstance(t, Tensor)
    # Mean should be close to 0
    assert abs(t._data.mean()) < 0.5


def test_randint_creation():
    """Test randint creation"""
    t = randint(2, 3, low=0, high=10)
    assert isinstance(t, Tensor)
    assert t.shape == (2, 3)
    assert t._data.min() >= 0
    assert t._data.max() < 10


# ==============================
# Compound Assignment Tests
# ==============================


def test_scalar_compound_assignment():
    """Test compound assignment operators for scalars"""
    i = integer(10)
    i += integer(5)
    assert i.item() == 15

    i -= integer(3)
    assert i.item() == 12

    i *= integer(2)
    assert i.item() == 24

    i //= integer(4)
    assert i.item() == 6


def test_tensor_compound_assignment():
    """Test compound assignment operators for tensors"""
    v = vector([1, 2, 3])
    v += vector([1, 1, 1])
    assert v._data.tolist() == [[2], [3], [4]]

    v *= 2
    assert v._data.tolist() == [[4], [6], [8]]


# ==============================
# Edge Cases and Error Handling
# ==============================


def test_scalar_from_zero_dim_array():
    """Test creating scalar from 0-dimensional array"""
    arr = np.array(5.0)
    r = real(arr)
    assert isinstance(r, Real)
    assert r.item() == 5.0


def test_vector_from_scalar():
    """Test creating vector from scalar creates 1-element vector"""
    s = real(5.0)
    v = vector(s)
    assert isinstance(v, Vector)
    assert v.shape == (1, 1)


def test_dimension_mismatch_errors():
    """Test that dimension mismatches raise appropriate errors"""
    m = matrix([[1, 2], [3, 4]])
    v = vector([1, 2, 3])  # Wrong size

    with pytest.raises((ValueError, Exception)):
        m @ v


def test_type_conversion():
    """Test type conversions between scalar types"""
    i = integer(10)
    r = real(i)
    assert isinstance(r, Real)
    assert r.item() == 10.0


# ==============================
# Integration Tests
# ==============================


def test_complex_expression():
    """Test complex mathematical expression"""
    x = real(1.0)
    y = real(2.0)
    z = real(0.0)

    # From the original example
    result = x**2 + 2 * x * y + sin(y * z) + exp(z)

    assert isinstance(result, Real)
    expected = 1.0 + 4.0 + 0.0 + 1.0  # 1 + 4 + sin(0) + exp(0)
    assert abs(result.item() - expected) < 1e-10


def test_linear_algebra_workflow():
    """Test typical linear algebra workflow"""
    # Create a system Ax = b
    A = matrix([[2, 1], [1, 3]])
    b = vector([5, 6])

    # Solve using direct computation (not implemented yet, but test matrix ops)
    result = A @ b
    assert isinstance(result, Vector)


def test_mixed_operations():
    """Test operations mixing different types"""
    s = real(2.0)
    v = vector([1.0, 2.0, 3.0])

    # Scalar * Vector
    result = s * v
    assert isinstance(result, Vector)
    assert result._data.tolist() == [[2.0], [4.0], [6.0]]


# ==============================
# Run all tests
# ==============================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
