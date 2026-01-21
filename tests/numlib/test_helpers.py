"""
Unit tests for type detection utilities and helper functions

Tests cover:
- Type cache initialization
- Type detection functions (_is_np_bool, _is_np_int, _is_np_float, _is_np_complex)
- Auto conversion functions (_auto_scalar, _auto_convert, _create_result)
- Array module functions (get_array_module, as_numpy, as_cupy)
- Utility functions (ones_like, zeros_like)
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon import numlib as nm
import numpy as np

# Import internal functions directly from the module
from lemon.numlib import (
    _is_np_bool,
    _is_np_int,
    _is_np_float,
    _is_np_complex,
    _auto_scalar,
    _auto_convert,
    _create_result,
    ones_like,
    zeros_like,
)


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, test_name, test_func):
        """Run a single test"""
        try:
            test_func()
            self.passed += 1
            print(f"✓ {test_name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((test_name, str(e)))
            print(f"✗ {test_name}: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((test_name, str(e)))
            print(f"✗ {test_name}: {type(e).__name__}: {e}")

    def report(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print(f"Tests run: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        if self.errors:
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print("=" * 60)
        return self.failed == 0


def test_type_detection():
    """Test type detection utilities"""
    runner = TestRunner()

    print("\n=== Testing Type Detection ===\n")

    # Test _is_np_bool
    def test_is_np_bool_python():
        assert _is_np_bool(True) is True
        assert _is_np_bool(False) is True

    runner.run_test("_is_np_bool with Python bool", test_is_np_bool_python)

    def test_is_np_bool_numpy():
        assert _is_np_bool(np.bool_(True)) is True
        assert _is_np_bool(np.array(True)) is True

    runner.run_test("_is_np_bool with NumPy bool", test_is_np_bool_numpy)

    def test_is_np_bool_false():
        assert _is_np_bool(1) is False
        assert _is_np_bool(1.0) is False
        assert _is_np_bool("True") is False

    runner.run_test("_is_np_bool returns False for non-bool", test_is_np_bool_false)

    # Test _is_np_int
    def test_is_np_int_python():
        assert _is_np_int(42) is True
        assert _is_np_int(-10) is True
        assert _is_np_int(0) is True

    runner.run_test("_is_np_int with Python int", test_is_np_int_python)

    def test_is_np_int_numpy():
        assert _is_np_int(np.int32(42)) is True
        assert _is_np_int(np.int64(-10)) is True
        assert _is_np_int(np.uint8(255)) is True
        assert _is_np_int(np.array(42)) is True

    runner.run_test("_is_np_int with NumPy int", test_is_np_int_numpy)

    def test_is_np_int_false():
        assert _is_np_int(1.0) is False
        # Note: Python bool is subclass of int, so True/False return True
        # This is expected behavior
        assert _is_np_int("42") is False
        assert _is_np_int(1 + 2j) is False

    runner.run_test("_is_np_int returns False for non-int", test_is_np_int_false)

    # Test _is_np_float
    def test_is_np_float_python():
        assert _is_np_float(3.14) is True
        assert _is_np_float(-2.5) is True
        assert _is_np_float(0.0) is True

    runner.run_test("_is_np_float with Python float", test_is_np_float_python)

    def test_is_np_float_numpy():
        assert _is_np_float(np.float32(3.14)) is True
        assert _is_np_float(np.float64(-2.5)) is True
        assert _is_np_float(np.array(3.14)) is True

    runner.run_test("_is_np_float with NumPy float", test_is_np_float_numpy)

    def test_is_np_float_false():
        assert _is_np_float(1) is False
        assert _is_np_float(True) is False
        assert _is_np_float("3.14") is False

    runner.run_test("_is_np_float returns False for non-float", test_is_np_float_false)

    # Test _is_np_complex
    def test_is_np_complex_python():
        assert _is_np_complex(1 + 2j) is True
        assert _is_np_complex(complex(3, 4)) is True
        assert _is_np_complex(5j) is True

    runner.run_test("_is_np_complex with Python complex", test_is_np_complex_python)

    def test_is_np_complex_numpy():
        assert _is_np_complex(np.complex64(1 + 2j)) is True
        assert _is_np_complex(np.complex128(3 + 4j)) is True
        assert _is_np_complex(np.array(1 + 2j)) is True

    runner.run_test("_is_np_complex with NumPy complex", test_is_np_complex_numpy)

    def test_is_np_complex_false():
        assert _is_np_complex(1) is False
        assert _is_np_complex(1.0) is False
        assert _is_np_complex("1+2j") is False

    runner.run_test(
        "_is_np_complex returns False for non-complex", test_is_np_complex_false
    )

    return runner.report()


def test_auto_scalar():
    """Test _auto_scalar function"""
    runner = TestRunner()

    print("\n=== Testing _auto_scalar ===\n")

    # Test with Python types
    def test_python_bool():
        result = _auto_scalar(True)
        assert isinstance(result, nm.Boolean)

    runner.run_test("_auto_scalar(bool) -> Boolean", test_python_bool)

    def test_python_int():
        result = _auto_scalar(42)
        assert isinstance(result, nm.Integer)

    runner.run_test("_auto_scalar(int) -> Integer", test_python_int)

    def test_python_float():
        result = _auto_scalar(3.14)
        assert isinstance(result, nm.Real)

    runner.run_test("_auto_scalar(float) -> Real", test_python_float)

    def test_python_complex():
        result = _auto_scalar(1 + 2j)
        assert isinstance(result, nm.Complex)

    runner.run_test("_auto_scalar(complex) -> Complex", test_python_complex)

    # Test with NumPy types
    def test_numpy_bool():
        result = _auto_scalar(np.bool_(True))
        assert isinstance(result, nm.Boolean)

    runner.run_test("_auto_scalar(np.bool_) -> Boolean", test_numpy_bool)

    def test_numpy_int():
        result = _auto_scalar(np.int64(42))
        assert isinstance(result, nm.Integer)

    runner.run_test("_auto_scalar(np.int64) -> Integer", test_numpy_int)

    def test_numpy_float():
        result = _auto_scalar(np.float64(3.14))
        assert isinstance(result, nm.Real)

    runner.run_test("_auto_scalar(np.float64) -> Real", test_numpy_float)

    def test_numpy_complex():
        result = _auto_scalar(np.complex128(1 + 2j))
        assert isinstance(result, nm.Complex)

    runner.run_test("_auto_scalar(np.complex128) -> Complex", test_numpy_complex)

    # Test with NumPy arrays (0-dimensional)
    def test_numpy_array_bool():
        result = _auto_scalar(np.array(True))
        assert isinstance(result, nm.Boolean)

    runner.run_test("_auto_scalar(np.array(bool)) -> Boolean", test_numpy_array_bool)

    def test_numpy_array_int():
        result = _auto_scalar(np.array(42))
        assert isinstance(result, nm.Integer)

    runner.run_test("_auto_scalar(np.array(int)) -> Integer", test_numpy_array_int)

    def test_numpy_array_float():
        result = _auto_scalar(np.array(3.14))
        assert isinstance(result, nm.Real)

    runner.run_test("_auto_scalar(np.array(float)) -> Real", test_numpy_array_float)

    def test_numpy_array_complex():
        result = _auto_scalar(np.array(1 + 2j))
        assert isinstance(result, nm.Complex)

    runner.run_test(
        "_auto_scalar(np.array(complex)) -> Complex", test_numpy_array_complex
    )

    # Test with existing Scalar
    def test_scalar_passthrough():
        x = nm.real(3.14)
        result = _auto_scalar(x)
        assert result is x

    runner.run_test("_auto_scalar(Scalar) returns same object", test_scalar_passthrough)

    # Test requires_grad parameter
    def test_requires_grad_true():
        result = _auto_scalar(3.14, requires_grad=True)
        assert result.requires_grad is True

    runner.run_test("_auto_scalar with requires_grad=True", test_requires_grad_true)

    def test_requires_grad_false():
        result = _auto_scalar(3.14, requires_grad=False)
        assert result.requires_grad is False

    runner.run_test("_auto_scalar with requires_grad=False", test_requires_grad_false)

    return runner.report()


def test_auto_convert():
    """Test _auto_convert function"""
    runner = TestRunner()

    print("\n=== Testing _auto_convert ===\n")

    # Test with scalars
    def test_python_scalar():
        result = _auto_convert(3.14)
        assert isinstance(result, nm.Scalar)

    runner.run_test("_auto_convert(scalar) -> Scalar", test_python_scalar)

    # Test with 1D arrays
    def test_1d_list():
        result = _auto_convert([1, 2, 3])
        assert isinstance(result, nm.Vector)
        assert result.shape == (3, 1)

    runner.run_test("_auto_convert(1D list) -> Vector", test_1d_list)

    def test_1d_numpy():
        result = _auto_convert(np.array([1, 2, 3]))
        assert isinstance(result, nm.Vector)
        assert result.shape == (3, 1)

    runner.run_test("_auto_convert(1D numpy) -> Vector", test_1d_numpy)

    # Test with 2D arrays
    def test_2d_list():
        result = _auto_convert([[1, 2], [3, 4]])
        assert isinstance(result, nm.Matrix)
        assert result.shape == (2, 2)

    runner.run_test("_auto_convert(2D list) -> Matrix", test_2d_list)

    def test_2d_numpy():
        result = _auto_convert(np.array([[1, 2], [3, 4]]))
        assert isinstance(result, nm.Matrix)
        assert result.shape == (2, 2)

    runner.run_test("_auto_convert(2D numpy) -> Matrix", test_2d_numpy)

    # Test with 3D+ arrays
    def test_3d_list():
        result = _auto_convert([[[1, 2]], [[3, 4]]])
        assert isinstance(result, nm.Tensor)
        assert result.ndim == 3

    runner.run_test("_auto_convert(3D list) -> Tensor", test_3d_list)

    # Test with NumType passthrough
    def test_numtype_passthrough():
        x = nm.vector([1, 2, 3])
        result = _auto_convert(x)
        assert result is x

    runner.run_test(
        "_auto_convert(NumType) returns same object", test_numtype_passthrough
    )

    # Test requires_grad parameter
    def test_requires_grad_true():
        result = _auto_convert([1, 2, 3], requires_grad=True)
        assert result.requires_grad is True

    runner.run_test("_auto_convert with requires_grad=True", test_requires_grad_true)

    def test_requires_grad_false():
        result = _auto_convert([1, 2, 3], requires_grad=False)
        assert result.requires_grad is False

    runner.run_test("_auto_convert with requires_grad=False", test_requires_grad_false)

    # Test 0-dimensional arrays
    def test_0d_numpy():
        result = _auto_convert(np.array(42))
        assert isinstance(result, nm.Scalar)

    runner.run_test("_auto_convert(0D numpy) -> Scalar", test_0d_numpy)

    return runner.report()


def test_create_result():
    """Test _create_result function"""
    runner = TestRunner()

    print("\n=== Testing _create_result ===\n")

    # Test 0-dimensional
    def test_0d():
        result = _create_result(np.array(42))
        assert isinstance(result, nm.Scalar)

    runner.run_test("_create_result(0D) -> Scalar", test_0d)

    # Test 1-dimensional
    def test_1d():
        result = _create_result(np.array([1, 2, 3]))
        assert isinstance(result, nm.Tensor)
        assert result.ndim == 1

    runner.run_test("_create_result(1D) -> Tensor", test_1d)

    # Test 2-dimensional - column vector
    def test_2d_column():
        result = _create_result(np.array([[1], [2], [3]]))
        assert isinstance(result, nm.Vector)
        assert result.shape == (3, 1)

    runner.run_test("_create_result(2D column) -> Vector", test_2d_column)

    # Test 2-dimensional - row vector
    def test_2d_row():
        result = _create_result(np.array([[1, 2, 3]]))
        assert isinstance(result, nm.RowVector)
        assert result.shape == (1, 3)

    runner.run_test("_create_result(2D row) -> RowVector", test_2d_row)

    # Test 2-dimensional - matrix
    def test_2d_matrix():
        result = _create_result(np.array([[1, 2], [3, 4]]))
        assert isinstance(result, nm.Matrix)
        assert result.shape == (2, 2)

    runner.run_test("_create_result(2D matrix) -> Matrix", test_2d_matrix)

    # Test 3-dimensional
    def test_3d():
        result = _create_result(np.array([[[1, 2]], [[3, 4]]]))
        assert isinstance(result, nm.Tensor)
        assert result.ndim == 3

    runner.run_test("_create_result(3D) -> Tensor", test_3d)

    # Test requires_grad inheritance
    def test_requires_grad():
        nm.autograd.enable()
        result = _create_result(np.array([1, 2, 3]))
        assert result.requires_grad is True

        nm.autograd.disable()
        result = _create_result(np.array([1, 2, 3]))
        assert result.requires_grad is False

        nm.autograd.enable()  # Reset

    runner.run_test("_create_result inherits requires_grad", test_requires_grad)

    return runner.report()


def test_array_module_functions():
    """Test array module helper functions"""
    runner = TestRunner()

    print("\n=== Testing Array Module Functions ===\n")

    # Test get_array_module
    def test_get_array_module_numpy():
        x = nm.vector([1, 2, 3])
        xp = nm.get_array_module(x._data)
        assert xp is np

    runner.run_test("get_array_module returns np", test_get_array_module_numpy)

    # Test as_numpy
    def test_as_numpy_from_numpy():
        x = nm.vector([1, 2, 3])
        result = nm.as_numpy(x._data)
        assert isinstance(result, np.ndarray)
        # Verify it's actually a NumPy array, not CuPy
        assert type(result).__module__ == "numpy"

    runner.run_test("as_numpy from numpy array", test_as_numpy_from_numpy)

    def test_as_numpy_from_python():
        result = nm.as_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)

    runner.run_test("as_numpy from Python list", test_as_numpy_from_python)

    # Test as_cupy
    if nm.cuda.is_available():

        def test_as_cupy_enabled():
            nm.cuda.enable()
            x = np.array([1, 2, 3])
            result = nm.as_cupy(x)
            assert hasattr(result, "device")
            nm.cuda.disable()

        runner.run_test("as_cupy when CUDA enabled", test_as_cupy_enabled)

    def test_as_cupy_disabled_raises():
        nm.cuda.disable()
        if nm.cuda.is_available():
            # Should raise when CUDA available but not enabled
            try:
                nm.as_cupy(np.array([1, 2, 3]))
                assert False, "Should raise RuntimeError"
            except RuntimeError as e:
                assert "GPU is not enabled" in str(e)
        else:
            # Should raise when CUDA not available
            try:
                nm.as_cupy(np.array([1, 2, 3]))
                assert False, "Should raise RuntimeError"
            except RuntimeError as e:
                assert "CuPy is not installed" in str(e)

    runner.run_test("as_cupy raises when disabled", test_as_cupy_disabled_raises)

    return runner.report()


def test_utility_functions():
    """Test utility functions like ones_like and zeros_like"""
    runner = TestRunner()

    print("\n=== Testing Utility Functions ===\n")

    # Test ones_like
    def test_ones_like_scalar():
        x = nm.real(3.14)
        result = ones_like(x)
        assert isinstance(result, nm.Real)
        assert float(result.item()) == 1.0

    runner.run_test("ones_like(scalar)", test_ones_like_scalar)

    def test_ones_like_vector():
        x = nm.vector([1, 2, 3])
        result = ones_like(x)
        assert isinstance(result, nm.Vector)
        assert result.shape == x.shape
        assert np.all(result._data == 1.0)

    runner.run_test("ones_like(vector)", test_ones_like_vector)

    def test_ones_like_matrix():
        x = nm.matrix([[1, 2], [3, 4]])
        result = ones_like(x)
        assert isinstance(result, nm.Matrix)
        assert result.shape == x.shape
        assert np.all(result._data == 1.0)

    runner.run_test("ones_like(matrix)", test_ones_like_matrix)

    # Test zeros_like
    def test_zeros_like_scalar():
        x = nm.real(3.14)
        result = zeros_like(x)
        assert isinstance(result, nm.Real)
        assert float(result.item()) == 0.0

    runner.run_test("zeros_like(scalar)", test_zeros_like_scalar)

    def test_zeros_like_vector():
        x = nm.vector([1, 2, 3])
        result = zeros_like(x)
        assert isinstance(result, nm.Vector)
        assert result.shape == x.shape
        assert np.all(result._data == 0.0)

    runner.run_test("zeros_like(vector)", test_zeros_like_vector)

    def test_zeros_like_matrix():
        x = nm.matrix([[1, 2], [3, 4]])
        result = zeros_like(x)
        assert isinstance(result, nm.Matrix)
        assert result.shape == x.shape
        assert np.all(result._data == 0.0)

    runner.run_test("zeros_like(matrix)", test_zeros_like_matrix)

    return runner.report()


if __name__ == "__main__":
    # Run all test suites
    results = []

    results.append(test_type_detection())
    results.append(test_auto_scalar())
    results.append(test_auto_convert())
    results.append(test_create_result())
    results.append(test_array_module_functions())
    results.append(test_utility_functions())

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")

    print("=" * 60)
    if all(results):
        print("✓ All test suites passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
