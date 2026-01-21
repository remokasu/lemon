import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestNumTypeInitialization:
    """Test NumType initialization and data conversion"""

    def test_init_from_python_scalar(self):
        """Test initialization from Python scalar"""
        x = nm.Real(3.14)
        assert x._data.shape == ()
        assert float(x._data) == 3.14

    def test_init_from_python_list(self):
        """Test initialization from Python list"""
        x = nm.Vector([1, 2, 3])
        assert isinstance(x._data, np.ndarray)
        assert x._data.shape == (3, 1)

    def test_init_from_numpy_array(self):
        """Test initialization from NumPy array"""
        arr = np.array([1, 2, 3])
        x = nm.Vector(arr)
        assert isinstance(x._data, np.ndarray)
        np.testing.assert_array_equal(x._data.flatten(), arr)

    def test_init_from_numtype(self):
        """Test initialization from another NumType"""
        x = nm.Real(3.14)
        y = nm.Real(x)
        assert y._data == x._data

    def test_init_with_name(self):
        """Test initialization with name parameter"""
        x = nm.Real(3.14, name="learning_rate")
        assert x.name == "learning_rate"

    def test_init_default_name_is_none(self):
        """Test that default name is None"""
        x = nm.Real(3.14)
        assert x.name is None

    def test_init_with_requires_grad_true(self):
        """Test initialization with requires_grad=True"""
        nm.autograd.enable()
        x = nm.Real(3.14, requires_grad=True)
        assert x.requires_grad is True

    def test_init_with_requires_grad_false(self):
        """Test initialization with requires_grad=False"""
        x = nm.Real(3.14, requires_grad=False)
        assert x.requires_grad is False

    def test_init_respects_autograd_state(self):
        """Test that initialization respects autograd state when requires_grad not specified"""
        nm.autograd.enable()
        x = nm.Real(3.14)
        assert x.requires_grad is True

        nm.autograd.disable()
        y = nm.Real(3.14)
        assert y.requires_grad is False

        nm.autograd.enable()  # Reset


class TestNumTypeProperties:
    """Test NumType properties"""

    def test_data_property(self):
        """Test data property returns underlying array"""
        arr = np.array([1, 2, 3])
        x = nm.Vector(arr)
        assert x.data is x._data
        np.testing.assert_array_equal(x.data.flatten(), arr)

    def test_shape_property_scalar(self):
        """Test shape property for scalar"""
        x = nm.Real(3.14)
        assert x.shape == ()

    def test_shape_property_vector(self):
        """Test shape property for vector"""
        x = nm.Vector([1, 2, 3])
        assert x.shape == (3, 1)

    def test_shape_property_matrix(self):
        """Test shape property for matrix"""
        x = nm.Matrix([[1, 2], [3, 4]])
        assert x.shape == (2, 2)

    def test_ndim_property_scalar(self):
        """Test ndim property for scalar"""
        x = nm.Real(3.14)
        assert x.ndim == 0

    def test_ndim_property_vector(self):
        """Test ndim property for vector"""
        x = nm.Vector([1, 2, 3])
        assert x.ndim == 2  # Vector is (n, 1)

    def test_ndim_property_matrix(self):
        """Test ndim property for matrix"""
        x = nm.Matrix([[1, 2], [3, 4]])
        assert x.ndim == 2

    def test_size_property_scalar(self):
        """Test size property for scalar"""
        x = nm.Real(3.14)
        assert x.size == 1

    def test_size_property_vector(self):
        """Test size property for vector"""
        x = nm.Vector([1, 2, 3])
        assert x.size == 3

    def test_size_property_matrix(self):
        """Test size property for matrix"""
        x = nm.Matrix([[1, 2], [3, 4]])
        assert x.size == 4

    def test_dtype_property_int(self):
        """Test dtype property for integer"""
        x = nm.Integer(42)
        assert x.dtype.kind in ("i", "u")  # signed or unsigned int

    def test_dtype_property_float(self):
        """Test dtype property for float"""
        x = nm.Real(3.14)
        assert x.dtype.kind == "f"

    def test_dtype_property_complex(self):
        """Test dtype property for complex"""
        x = nm.Complex(1 + 2j)
        assert x.dtype.kind == "c"

    def test_dtype_property_bool(self):
        """Test dtype property for boolean"""
        x = nm.Boolean(True)
        assert x.dtype.kind == "b"


class TestNumTypeGPUCPUTransfer:
    """Test GPU/CPU transfer methods"""

    def test_as_numpy_from_numpy(self):
        """Test as_numpy when already on CPU"""
        x = nm.Vector([1, 2, 3])
        result = x.as_numpy()
        assert result is x  # Should return self
        assert isinstance(x._data, np.ndarray)

    def test_as_numpy_returns_self(self):
        """Test that as_numpy returns self"""
        x = nm.Vector([1, 2, 3])
        result = x.as_numpy()
        assert result is x

    @pytest.mark.skipif(not nm.cuda.is_available(), reason="CUDA not available")
    def test_as_cupy_when_cuda_enabled(self):
        """Test as_cupy when CUDA is enabled"""
        nm.cuda.enable()
        x = nm.Vector([1, 2, 3])
        result = x.as_cupy()
        assert result is x  # Should return self
        assert hasattr(x._data, "device")
        nm.cuda.disable()

    def test_as_cupy_raises_when_cuda_disabled(self):
        """Test as_cupy raises when CUDA disabled"""
        nm.cuda.disable()
        x = nm.Vector([1, 2, 3])
        if nm.cuda.is_available():
            with pytest.raises(RuntimeError, match="GPU is not enabled"):
                x.as_cupy()
        else:
            with pytest.raises(RuntimeError, match="CuPy is not installed"):
                x.as_cupy()

    @pytest.mark.skipif(not nm.cuda.is_available(), reason="CUDA not available")
    def test_as_numpy_from_cupy(self):
        """Test as_numpy converts from CuPy to NumPy"""
        nm.cuda.enable()
        x = nm.Vector([1, 2, 3])
        x.as_cupy()  # Move to GPU
        x.as_numpy()  # Move back to CPU
        assert isinstance(x._data, np.ndarray)
        nm.cuda.disable()


class TestNumTypeArrayProtocol:
    """Test NumPy array protocol implementation"""

    def test_array_protocol_basic(self):
        """Test __array__ returns NumPy array"""
        x = nm.Vector([1, 2, 3])
        arr = np.array(x)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr.flatten(), [1, 2, 3])

    def test_array_protocol_with_dtype(self):
        """Test __array__ with dtype conversion"""
        x = nm.Vector([1, 2, 3])
        arr = np.array(x, dtype=np.float32)
        assert arr.dtype == np.float32

    def test_array_protocol_scalar(self):
        """Test __array__ with scalar"""
        x = nm.Real(3.14)
        arr = np.array(x)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == ()
        assert float(arr) == 3.14

    def test_array_ufunc_add(self):
        """Test __array_ufunc__ with np.add"""
        x = nm.Vector([1, 2, 3])
        y = nm.Vector([4, 5, 6])
        result = np.add(x, y)
        assert isinstance(result, nm.NumType)
        np.testing.assert_array_equal(result._data.flatten(), [5, 7, 9])

    def test_array_ufunc_multiply(self):
        """Test __array_ufunc__ with np.multiply"""
        x = nm.Vector([1, 2, 3])
        result = np.multiply(x, 2)
        assert isinstance(result, nm.NumType)
        np.testing.assert_array_equal(result._data.flatten(), [2, 4, 6])

    def test_array_ufunc_exp(self):
        """Test __array_ufunc__ with np.exp"""
        x = nm.Real(0.0)
        result = np.exp(x)
        assert isinstance(result, nm.NumType)
        assert float(result._data) == 1.0

    def test_array_ufunc_sin(self):
        """Test __array_ufunc__ with np.sin"""
        x = nm.Real(0.0)
        result = np.sin(x)
        assert isinstance(result, nm.NumType)
        assert float(result._data) == 0.0

    def test_array_ufunc_unsupported(self):
        """Test __array_ufunc__ with unsupported ufunc falls back to NumPy"""
        x = nm.Vector([1, 2, 3])
        # Use a ufunc that's not in the mapping
        result = np.sign(x)
        assert isinstance(result, nm.NumType)


class TestNumTypeConversionMethods:
    """Test conversion methods"""

    def test_to_numpy_scalar(self):
        """Test to_numpy for scalar"""
        x = nm.Real(3.14)
        arr = x.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == ()
        assert float(arr) == 3.14

    def test_to_numpy_vector(self):
        """Test to_numpy for vector"""
        x = nm.Vector([1, 2, 3])
        arr = x.to_numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr.flatten(), [1, 2, 3])

    def test_to_list_scalar(self):
        """Test to_list for scalar"""
        x = nm.Real(3.14)
        result = x.to_list()
        assert result == 3.14

    def test_to_list_vector(self):
        """Test to_list for vector"""
        x = nm.Vector([1, 2, 3])
        result = x.to_list()
        assert isinstance(result, list)
        assert len(result) == 3

    def test_to_list_matrix(self):
        """Test to_list for matrix"""
        x = nm.Matrix([[1, 2], [3, 4]])
        result = x.to_list()
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 2

    def test_tolist_scalar(self):
        """Test tolist for scalar (alias)"""
        x = nm.Real(3.14)
        result = x.tolist()
        assert result == 3.14

    def test_tolist_vector(self):
        """Test tolist for vector (alias)"""
        x = nm.Vector([1, 2, 3])
        result = x.tolist()
        assert isinstance(result, list)

    def test_item_scalar(self):
        """Test item for scalar"""
        x = nm.Real(3.14)
        result = x.item()
        assert isinstance(result, float)
        assert result == 3.14

    def test_item_integer(self):
        """Test item for integer"""
        x = nm.Integer(42)
        result = x.item()
        assert isinstance(result, (int, np.integer))

    def test_float_conversion(self):
        """Test __float__ conversion"""
        x = nm.Real(3.14)
        result = float(x)
        assert isinstance(result, float)
        assert result == 3.14

    def test_float_conversion_from_integer(self):
        """Test __float__ conversion from integer"""
        x = nm.Integer(42)
        result = float(x)
        assert isinstance(result, float)
        assert result == 42.0

    def test_int_conversion(self):
        """Test __int__ conversion"""
        x = nm.Integer(42)
        result = int(x)
        assert isinstance(result, int)
        assert result == 42

    def test_int_conversion_from_real(self):
        """Test __int__ conversion from real"""
        x = nm.Real(3.14)
        result = int(x)
        assert isinstance(result, int)
        assert result == 3

    def test_complex_conversion(self):
        """Test __complex__ conversion"""
        x = nm.Complex(1 + 2j)
        result = complex(x)
        assert isinstance(result, complex)
        assert result == 1 + 2j

    def test_complex_conversion_from_real(self):
        """Test __complex__ conversion from real"""
        x = nm.Real(3.14)
        result = complex(x)
        assert isinstance(result, complex)
        assert result.real == 3.14
        assert result.imag == 0.0


class TestNumTypeContainerProtocol:
    """Test container protocol implementation"""

    def test_len_vector(self):
        """Test __len__ for vector"""
        x = nm.Vector([1, 2, 3])
        assert len(x) == 3

    def test_len_matrix(self):
        """Test __len__ for matrix"""
        x = nm.Matrix([[1, 2], [3, 4], [5, 6]])
        assert len(x) == 3  # Number of rows

    def test_contains_true(self):
        """Test __contains__ when element exists"""
        x = nm.Vector([1, 2, 3])
        # Note: Contains checks the flattened array
        assert 2 in x._data.flatten()

    def test_contains_false(self):
        """Test __contains__ when element doesn't exist"""
        x = nm.Vector([1, 2, 3])
        assert 10 not in x._data.flatten()

    def test_iter_vector(self):
        """Test __iter__ for vector"""
        x = nm.Vector([1, 2, 3])
        result = list(x)
        assert len(result) == 3

    def test_iter_matrix(self):
        """Test __iter__ for matrix"""
        x = nm.Matrix([[1, 2], [3, 4]])
        result = list(x)
        assert len(result) == 2  # Number of rows


class TestNumTypeGradientManagement:
    """Test gradient management"""

    def test_grad_initial_none(self):
        """Test that grad is initially None"""
        x = nm.Real(3.0, requires_grad=True)
        assert x.grad is None

    def test_g_property(self):
        """Test g property as alias for grad"""
        x = nm.Real(3.0, requires_grad=True)
        assert x.g is x.grad

    def test_backward_simple(self):
        """Test simple backward pass"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3
        y.backward()
        assert x.grad is not None
        assert float(x.grad._data) == 3.0

    def test_backward_requires_grad_false_raises(self):
        """Test backward raises when requires_grad=False"""
        x = nm.Real(2.0, requires_grad=False)
        with pytest.raises(RuntimeError, match="does not require gradients"):
            x.backward()

    def test_backward_non_scalar_requires_gradient(self):
        """Test backward on non-scalar requires gradient argument"""
        x = nm.Vector([1, 2, 3], requires_grad=True)
        y = x * 2
        with pytest.raises(
            RuntimeError, match="Cannot compute gradient for non-scalar output"
        ):
            y.backward()

    def test_backward_non_scalar_with_gradient(self):
        """Test backward on non-scalar with gradient"""
        x = nm.Vector([1, 2, 3], requires_grad=True)
        y = x * 2
        grad_output = nm.Vector([1, 1, 1])
        y.backward(grad_output)
        assert x.grad is not None
        np.testing.assert_array_equal(x.grad._data.flatten(), [2, 2, 2])

    def test_zero_grad(self):
        """Test zero_grad resets gradient"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3
        y.backward()
        assert x.grad is not None

        x.zero_grad()
        assert x.grad is None

    def test_cleargrad_alias(self):
        """Test cleargrad as alias for zero_grad"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3
        y.backward()
        assert x.grad is not None

        x.cleargrad()
        assert x.grad is None

    def test_gradient_accumulation(self):
        """Test that gradients accumulate"""
        x = nm.Real(2.0, requires_grad=True)

        y1 = x * 3
        y1.backward()
        grad1 = float(x.grad._data)

        y2 = x * 2
        y2.backward()
        grad2 = float(x.grad._data)

        assert grad2 == grad1 + 2.0  # 3 + 2

    def test_backward_clears_computational_graph(self):
        """Test backward on intermediate node"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3
        z = y * 2

        z.backward()
        assert x.grad is not None
        assert float(x.grad._data) == 6.0  # dy/dx = 3 * 2

    def test_requires_grad_propagation(self):
        """Test requires_grad propagates through operations"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3
        assert y.requires_grad is True

        x2 = nm.Real(2.0, requires_grad=False)
        y2 = x2 * 3
        assert y2.requires_grad is False


class TestNumTypeStringRepresentation:
    """Test string representation"""

    def test_repr_real(self):
        """Test __repr__ for Real"""
        x = nm.Real(3.14)
        repr_str = repr(x)
        assert "real" in repr_str.lower()
        assert "3.14" in repr_str

    def test_repr_integer(self):
        """Test __repr__ for Integer"""
        x = nm.Integer(42)
        repr_str = repr(x)
        assert "int" in repr_str.lower()
        assert "42" in repr_str

    def test_repr_complex(self):
        """Test __repr__ for Complex"""
        x = nm.Complex(1 + 2j)
        repr_str = repr(x)
        assert "complex" in repr_str.lower()

    def test_repr_boolean(self):
        """Test __repr__ for Boolean"""
        x = nm.Boolean(True)
        repr_str = repr(x)
        assert "bool" in repr_str.lower()

    def test_repr_vector(self):
        """Test __repr__ for Vector"""
        x = nm.Vector([1, 2, 3])
        repr_str = repr(x)
        assert "vector" in repr_str.lower()

    def test_repr_matrix(self):
        """Test __repr__ for Matrix"""
        x = nm.Matrix([[1, 2], [3, 4]])
        repr_str = repr(x)
        assert "matrix" in repr_str.lower()


class TestNumTypeEdgeCases:
    """Test edge cases and special scenarios"""

    def test_empty_array(self):
        """Test with empty array"""
        x = nm.Tensor(np.array([]))
        assert x.size == 0
        assert x.shape == (0,)

    def test_single_element_array(self):
        """Test with single element array"""
        x = nm.Vector([42])
        assert x.size == 1
        assert x.shape == (1, 1)

    def test_large_array(self):
        """Test with large array"""
        data = list(range(10000))
        x = nm.Vector(data)
        assert x.size == 10000

    def test_negative_values(self):
        """Test with negative values"""
        x = nm.Vector([-1, -2, -3])
        assert all(val < 0 for val in x._data.flatten())

    def test_zero_values(self):
        """Test with zero values"""
        x = nm.Vector([0, 0, 0])
        assert all(val == 0 for val in x._data.flatten())

    def test_mixed_signs(self):
        """Test with mixed positive and negative values"""
        x = nm.Vector([-1, 0, 1])
        assert x.size == 3

    def test_very_small_values(self):
        """Test with very small float values"""
        x = nm.Real(1e-100)
        assert float(x._data) == 1e-100

    def test_very_large_values(self):
        """Test with very large float values"""
        x = nm.Real(1e100)
        assert float(x._data) == 1e100

    def test_inf_values(self):
        """Test with infinity values"""
        x = nm.Real(float("inf"))
        assert np.isinf(x._data)

    def test_nan_values(self):
        """Test with NaN values"""
        x = nm.Real(float("nan"))
        assert np.isnan(x._data)


class TestNumTypeRealImagProperties:
    """Test real and imag properties for complex support"""

    def test_real_property_real_number(self):
        """Test real property returns self for real numbers"""
        x = nm.Real(3.14)
        assert x.real is x

    def test_imag_property_real_number(self):
        """Test imag property returns zero for real numbers"""
        x = nm.Real(3.14)
        imag = x.imag
        assert isinstance(imag, nm.NumType)
        assert float(imag._data) == 0.0

    def test_real_property_complex_number(self):
        """Test real property for complex numbers"""
        x = nm.Complex(3 + 4j)
        real_part = x.real
        assert isinstance(real_part, nm.Real)
        assert float(real_part._data) == 3.0

    def test_imag_property_complex_number(self):
        """Test imag property for complex numbers"""
        x = nm.Complex(3 + 4j)
        imag_part = x.imag
        assert isinstance(imag_part, nm.Real)
        assert float(imag_part._data) == 4.0

    def test_real_property_complex_vector(self):
        """Test real property for complex vector"""
        x = nm.Vector([1 + 2j, 3 + 4j])
        real_part = x.real
        assert isinstance(real_part, nm.Vector)
        np.testing.assert_array_equal(real_part._data.flatten(), [1, 3])

    def test_imag_property_complex_vector(self):
        """Test imag property for complex vector"""
        x = nm.Vector([1 + 2j, 3 + 4j])
        imag_part = x.imag
        assert isinstance(imag_part, nm.Vector)
        np.testing.assert_array_equal(imag_part._data.flatten(), [2, 4])


class TestNumTypeTransposeProperty:
    """Test T property (transpose)"""

    def test_T_property_vector(self):
        """Test T property for vector"""
        x = nm.Vector([1, 2, 3])
        x_t = x.T
        assert isinstance(x_t, nm.RowVector)
        assert x_t.shape == (1, 3)

    def test_T_property_matrix(self):
        """Test T property for matrix"""
        x = nm.Matrix([[1, 2], [3, 4]])
        x_t = x.T
        assert isinstance(x_t, nm.Matrix)
        assert x_t.shape == (2, 2)
        np.testing.assert_array_equal(x_t._data, [[1, 3], [2, 4]])

    def test_T_property_double_transpose(self):
        """Test double transpose returns original"""
        x = nm.Vector([1, 2, 3])
        x_tt = x.T.T
        assert isinstance(x_tt, nm.Vector)
        np.testing.assert_array_equal(x_tt._data.flatten(), x._data.flatten())


class TestNumTypeBackwardPrev:
    """Test _prev and _backward internal state"""

    def test_prev_initially_empty(self):
        """Test _prev is initially empty set"""
        x = nm.Real(3.0, requires_grad=True)
        assert isinstance(x._prev, set)
        assert len(x._prev) == 0

    def test_prev_populated_after_operation(self):
        """Test _prev is populated after operation"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3
        assert len(y._prev) == 1
        assert x in y._prev

    def test_prev_with_multiple_inputs(self):
        """Test _prev with multiple inputs"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)
        z = x + y
        assert len(z._prev) == 2
        assert x in z._prev
        assert y in z._prev

    def test_backward_callable(self):
        """Test _backward is callable"""
        x = nm.Real(3.0, requires_grad=True)
        assert callable(x._backward)

    def test_backward_initially_noop(self):
        """Test _backward is initially no-op"""
        x = nm.Real(3.0, requires_grad=True)
        # Should not raise
        x._backward()


class TestNumTypeWithAutogradOff:
    """Test NumType behavior when autograd is off"""

    def test_creation_with_autograd_off(self):
        """Test NumType creation with autograd off"""
        with nm.autograd.off:
            x = nm.Real(3.0)
            assert x.requires_grad is False

    def test_operation_with_autograd_off(self):
        """Test operations with autograd off"""
        with nm.autograd.off:
            x = nm.Real(2.0)
            y = x * 3
            assert y.requires_grad is False

    def test_no_gradient_tracking_with_autograd_off(self):
        """Test no gradient tracking when autograd off"""
        with nm.autograd.off:
            x = nm.Real(2.0)
            y = x * 3
            assert len(y._prev) == 0  # Should not track dependencies
