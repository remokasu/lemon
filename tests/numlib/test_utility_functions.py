import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestOneHot:
    """Test one-hot encoding"""

    def test_one_hot_basic(self):
        """Test basic one-hot encoding"""
        labels = [0, 1, 2]
        result = nm.one_hot(labels, num_classes=3)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(result._data, expected)

    def test_one_hot_infer_num_classes(self):
        """Test one-hot with inferred num_classes"""
        labels = [0, 1, 2, 1]
        result = nm.one_hot(labels)
        assert result.shape == (4, 3)

    def test_one_hot_tensor_input(self):
        """Test one-hot with tensor input"""
        labels = nm.Tensor([0, 1, 2])
        result = nm.one_hot(labels, num_classes=3)
        assert result.shape == (3, 3)

    def test_one_hot_repeated_labels(self):
        """Test one-hot with repeated labels"""
        labels = [0, 0, 1, 1, 2]
        result = nm.one_hot(labels, num_classes=3)
        assert result.shape == (5, 3)
        # Check first two rows are identical
        np.testing.assert_array_equal(result._data[0], result._data[1])

    def test_one_hot_larger_num_classes(self):
        """Test one-hot with num_classes > max(labels)"""
        labels = [0, 1]
        result = nm.one_hot(labels, num_classes=5)
        assert result.shape == (2, 5)
        expected = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result._data, expected)


class TestEye:
    """Test identity matrix creation"""

    def test_eye_square(self):
        """Test creating square identity matrix"""
        I = nm.eye(3)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(I._data, expected)

    def test_eye_rectangular(self):
        """Test creating rectangular identity matrix"""
        I = nm.eye(3, 5)
        assert I.shape == (3, 5)
        # Check diagonal is 1
        for i in range(3):
            assert I._data[i, i] == 1

    def test_eye_dtype(self):
        """Test eye with specific dtype"""
        I = nm.eye(3, dtype=np.int32)
        assert I.dtype == np.int32

    def test_eye_single_element(self):
        """Test eye with n=1"""
        I = nm.eye(1)
        assert I.shape == (1, 1)
        assert I._data[0, 0] == 1


class TestArange:
    """Test arange (evenly spaced values)"""

    def test_arange_stop_only(self):
        """Test arange with stop only"""
        result = nm.arange(5)
        expected = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result._data, expected)

    def test_arange_start_stop(self):
        """Test arange with start and stop"""
        result = nm.arange(2, 7)
        expected = np.array([2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result._data, expected)

    def test_arange_start_stop_step(self):
        """Test arange with start, stop, and step"""
        result = nm.arange(0, 10, 2)
        expected = np.array([0, 2, 4, 6, 8])
        np.testing.assert_array_equal(result._data, expected)

    def test_arange_float_step(self):
        """Test arange with float step"""
        result = nm.arange(0, 1, 0.2)
        expected = np.array([0, 0.2, 0.4, 0.6, 0.8])
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_arange_negative_step(self):
        """Test arange with negative step"""
        result = nm.arange(5, 0, -1)
        expected = np.array([5, 4, 3, 2, 1])
        np.testing.assert_array_equal(result._data, expected)

    def test_arange_dtype(self):
        """Test arange with specific dtype"""
        result = nm.arange(5, dtype=np.float32)
        assert result.dtype == np.float32


class TestLinspace:
    """Test linspace (linearly spaced values)"""

    def test_linspace_basic(self):
        """Test basic linspace"""
        result = nm.linspace(0, 10, 5)
        expected = np.array([0, 2.5, 5, 7.5, 10])
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_linspace_single_value(self):
        """Test linspace with num=1"""
        result = nm.linspace(5, 10, 1)
        assert result.shape == (1,)
        assert result._data[0] == 5

    def test_linspace_two_values(self):
        """Test linspace with num=2"""
        result = nm.linspace(0, 10, 2)
        expected = np.array([0, 10])
        np.testing.assert_array_equal(result._data, expected)

    def test_linspace_negative_range(self):
        """Test linspace with negative range"""
        result = nm.linspace(-5, 5, 11)
        expected = np.linspace(-5, 5, 11)
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_linspace_dtype(self):
        """Test linspace with specific dtype"""
        result = nm.linspace(0, 10, 5, dtype=np.float32)
        assert result.dtype == np.float32


class TestConcatenate:
    """Test concatenate"""

    def test_concatenate_vectors_axis0(self):
        """Test concatenating vectors along axis 0"""
        v1 = nm.Vector([1, 2])
        v2 = nm.Vector([3, 4])
        result = nm.concatenate([v1, v2], axis=0)
        assert result.shape == (4, 1)
        expected = np.array([[1], [2], [3], [4]])
        np.testing.assert_array_equal(result._data, expected)

    def test_concatenate_matrices_axis0(self):
        """Test concatenating matrices along axis 0 (vertically)"""
        m1 = nm.Matrix([[1, 2], [3, 4]])
        m2 = nm.Matrix([[5, 6]])
        result = nm.concatenate([m1, m2], axis=0)
        assert result.shape == (3, 2)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(result._data, expected)

    def test_concatenate_matrices_axis1(self):
        """Test concatenating matrices along axis 1 (horizontally)"""
        m1 = nm.Matrix([[1, 2], [3, 4]])
        m2 = nm.Matrix([[5], [6]])
        result = nm.concatenate([m1, m2], axis=1)
        assert result.shape == (2, 3)
        expected = np.array([[1, 2, 5], [3, 4, 6]])
        np.testing.assert_array_equal(result._data, expected)

    def test_concatenate_multiple_tensors(self):
        """Test concatenating more than 2 tensors"""
        t1 = nm.Tensor([1, 2])
        t2 = nm.Tensor([3, 4])
        t3 = nm.Tensor([5, 6])
        result = nm.concatenate([t1, t2, t3], axis=0)
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result._data, expected)


class TestStack:
    """Test stack"""

    def test_stack_vectors_axis0(self):
        """Test stacking vectors along new axis 0"""
        v1 = nm.Vector([1, 2, 3])
        v2 = nm.Vector([4, 5, 6])
        result = nm.stack([v1, v2], axis=0)
        assert result.shape == (2, 3, 1)

    def test_stack_matrices_axis0(self):
        """Test stacking matrices along new axis 0"""
        m1 = nm.Matrix([[1, 2], [3, 4]])
        m2 = nm.Matrix([[5, 6], [7, 8]])
        result = nm.stack([m1, m2], axis=0)
        assert result.shape == (2, 2, 2)

    def test_stack_tensors_axis1(self):
        """Test stacking tensors along axis 1"""
        t1 = nm.Tensor([1, 2, 3])
        t2 = nm.Tensor([4, 5, 6])
        result = nm.stack([t1, t2], axis=1)
        assert result.shape == (3, 2)

    def test_stack_multiple_tensors(self):
        """Test stacking more than 2 tensors"""
        t1 = nm.Tensor([1, 2])
        t2 = nm.Tensor([3, 4])
        t3 = nm.Tensor([5, 6])
        result = nm.stack([t1, t2, t3], axis=0)
        assert result.shape == (3, 2)


class TestHelperFunctions:
    """Test helper functions"""

    def test_get_array_module_numpy(self):
        """Test get_array_module returns numpy for numpy array"""
        x = nm.Vector([1, 2, 3])
        module = nm.get_array_module(x._data)
        assert module is np

    @pytest.mark.skipif(not nm.cuda.is_available(), reason="CUDA not available")
    def test_get_array_module_cupy(self):
        """Test get_array_module returns cupy for cupy array"""
        nm.cuda.enable()
        x = nm.Vector([1, 2, 3])
        x.as_cupy()
        module = nm.get_array_module(x._data)
        assert module.__name__ == "cupy"
        nm.cuda.disable()

    def test_as_numpy_function(self):
        """Test as_numpy function"""
        x = nm.Vector([1, 2, 3])
        result = nm.as_numpy(x._data)
        assert isinstance(result, np.ndarray)

    def test_as_cupy_function_no_cuda(self):
        """Test as_cupy raises when CUDA not available"""
        nm.cuda.disable()
        x = nm.Vector([1, 2, 3])
        if not nm.cuda.is_available():
            with pytest.raises(RuntimeError):
                nm.as_cupy(x._data)

    def test_auto_convert_python_int(self):
        """Test _auto_convert with Python int"""
        result = nm._auto_convert(42)
        assert isinstance(result, nm.Integer)

    def test_auto_convert_python_float(self):
        """Test _auto_convert with Python float"""
        result = nm._auto_convert(3.14)
        assert isinstance(result, nm.Real)

    def test_auto_convert_python_complex(self):
        """Test _auto_convert with Python complex"""
        result = nm._auto_convert(1 + 2j)
        assert isinstance(result, nm.Complex)

    def test_auto_convert_python_bool(self):
        """Test _auto_convert with Python bool"""
        result = nm._auto_convert(True)
        assert isinstance(result, nm.Boolean)

    def test_auto_convert_list_to_vector(self):
        """Test _auto_convert with 1D list"""
        result = nm._auto_convert([1, 2, 3])
        assert isinstance(result, nm.Vector)

    def test_auto_convert_2d_list_to_matrix(self):
        """Test _auto_convert with 2D list"""
        result = nm._auto_convert([[1, 2], [3, 4]])
        assert isinstance(result, nm.Matrix)

    def test_auto_convert_numtype_returns_same(self):
        """Test _auto_convert with NumType returns same object"""
        x = nm.Real(3.14)
        result = nm._auto_convert(x)
        assert result is x

    def test_create_result_0d(self):
        """Test _create_result with 0D array creates scalar"""
        arr = np.array(3.14)
        result = nm._create_result(arr)
        assert isinstance(result, nm.Scalar)

    def test_create_result_1d(self):
        """Test _create_result with 1D array creates tensor"""
        arr = np.array([1, 2, 3])
        result = nm._create_result(arr)
        assert isinstance(result, nm.Tensor)
        assert result.ndim == 1

    def test_create_result_2d_column(self):
        """Test _create_result with (n, 1) array creates Vector"""
        arr = np.array([[1], [2], [3]])
        result = nm._create_result(arr)
        assert isinstance(result, nm.Vector)

    def test_create_result_2d_row(self):
        """Test _create_result with (1, n) array creates RowVector"""
        arr = np.array([[1, 2, 3]])
        result = nm._create_result(arr)
        assert isinstance(result, nm.RowVector)

    def test_create_result_2d_matrix(self):
        """Test _create_result with (m, n) array creates Matrix"""
        arr = np.array([[1, 2], [3, 4]])
        result = nm._create_result(arr)
        assert isinstance(result, nm.Matrix)

    def test_create_result_3d(self):
        """Test _create_result with 3D array creates Tensor"""
        arr = np.array([[[1, 2], [3, 4]]])
        result = nm._create_result(arr)
        assert isinstance(result, nm.Tensor)
        assert result.ndim == 3


class TestRandomFunctions:
    """Test random number generation functions"""

    def test_rand_shape(self):
        """Test rand creates correct shape"""
        result = nm.rand(3, 4)
        assert result.shape == (3, 4)

    def test_rand_range(self):
        """Test rand values in correct range"""
        result = nm.rand(100, low=0.0, high=1.0)
        assert np.all(result._data >= 0.0)
        assert np.all(result._data <= 1.0)

    def test_rand_custom_range(self):
        """Test rand with custom range"""
        result = nm.rand(100, low=-5.0, high=5.0)
        assert np.all(result._data >= -5.0)
        assert np.all(result._data <= 5.0)

    def test_randn_shape(self):
        """Test randn creates correct shape"""
        result = nm.randn(3, 4)
        assert result.shape == (3, 4)

    def test_randn_distribution(self):
        """Test randn follows normal distribution (approximately)"""
        result = nm.randn(10000)
        mean = np.mean(result._data)
        std = np.std(result._data)
        assert abs(mean) < 0.1  # Should be close to 0
        assert abs(std - 1.0) < 0.1  # Should be close to 1

    def test_randint_shape(self):
        """Test randint creates correct shape"""
        result = nm.randint(3, 4, low=0, high=10)
        assert result.shape == (3, 4)

    def test_randint_range(self):
        """Test randint values in correct range"""
        result = nm.randint(100, low=5, high=10)
        assert np.all(result._data >= 5)
        assert np.all(result._data < 10)

    def test_seed_reproducibility(self):
        """Test seed makes random generation reproducible"""
        nm.seed(42)
        r1 = nm.rand(3, 3)
        nm.seed(42)
        r2 = nm.rand(3, 3)
        np.testing.assert_array_equal(r1._data, r2._data)

    def test_random_vector(self):
        """Test random_vector factory function"""
        v = nm.random_vector(5)
        assert isinstance(v, nm.Vector)
        assert v.shape == (5, 1)

    def test_random_matrix(self):
        """Test random_matrix factory function"""
        m = nm.random_matrix((3, 4))
        assert isinstance(m, nm.Matrix)
        assert m.shape == (3, 4)

    def test_random_tensor(self):
        """Test random_tensor factory function"""
        t = nm.random_tensor((2, 3, 4))
        assert isinstance(t, nm.Tensor)
        assert t.shape == (2, 3, 4)


class TestZerosOnes:
    """Test zeros and ones creation functions"""

    def test_zeros_shape(self):
        """Test zeros creates correct shape"""
        result = nm.zeros(3, 4)
        assert result.shape == (3, 4)
        assert np.all(result._data == 0)

    def test_zeros_tuple_shape(self):
        """Test zeros with tuple shape"""
        result = nm.zeros((3, 4))
        assert result.shape == (3, 4)

    def test_zeros_dtype(self):
        """Test zeros with specific dtype"""
        result = nm.zeros(3, 4, dtype=np.int32)
        assert result.dtype == np.int32

    def test_zeros_vector(self):
        """Test zeros_vector"""
        v = nm.zeros_vector(5)
        assert isinstance(v, nm.Vector)
        assert v.shape == (5, 1)
        assert np.all(v._data == 0)

    def test_zeros_matrix(self):
        """Test zeros_matrix"""
        m = nm.zeros_matrix((3, 4))
        assert isinstance(m, nm.Matrix)
        assert m.shape == (3, 4)

    def test_ones_shape(self):
        """Test ones creates correct shape"""
        result = nm.ones(3, 4)
        assert result.shape == (3, 4)
        assert np.all(result._data == 1)

    def test_ones_vector(self):
        """Test ones_vector"""
        v = nm.ones_vector(5)
        assert isinstance(v, nm.Vector)
        assert v.shape == (5, 1)
        assert np.all(v._data == 1)

    def test_ones_matrix(self):
        """Test ones_matrix"""
        m = nm.ones_matrix((3, 4))
        assert isinstance(m, nm.Matrix)
        assert m.shape == (3, 4)
        assert np.all(m._data == 1)
