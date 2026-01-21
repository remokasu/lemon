import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon.numlib import *
import numpy as np


def test_one_hot():
    """Test one_hot encoding"""
    print("\n" + "=" * 60)
    print("Testing One-Hot Encoding")
    print("=" * 60)

    # Test 1: Basic one-hot encoding
    labels = [0, 1, 2, 1]
    one_hot_encoded = one_hot(labels, num_classes=3)

    expected = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )

    assert isinstance(one_hot_encoded, Tensor)
    assert one_hot_encoded.shape == (4, 3)
    assert np.allclose(one_hot_encoded._data, expected)
    print("✓ Basic one-hot encoding test passed")

    # Test 2: Infer num_classes
    labels = [0, 1, 2, 3]
    one_hot_encoded = one_hot(labels)

    assert one_hot_encoded.shape == (4, 4)
    assert np.allclose(one_hot_encoded._data, np.eye(4))
    print("✓ Inferred num_classes test passed")

    # Test 3: With NumType input
    labels_tensor = vector([0, 1, 2, 1])
    one_hot_encoded = one_hot(labels_tensor, num_classes=3)

    assert one_hot_encoded.shape == (4, 3)
    assert np.allclose(one_hot_encoded._data, expected)
    print("✓ NumType input test passed")


def test_eye():
    """Test identity matrix creation"""
    print("\n" + "=" * 60)
    print("Testing Identity Matrix")
    print("=" * 60)

    # Test 1: Square identity matrix
    I = eye(3)

    assert isinstance(I, Matrix)
    assert I.shape == (3, 3)
    assert np.allclose(I._data, np.eye(3))
    print("✓ Square identity matrix test passed")

    # Test 2: Rectangular identity matrix
    I = eye(3, 5)

    assert I.shape == (3, 5)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(I._data, expected)
    print("✓ Rectangular identity matrix test passed")

    # Test 3: With dtype
    I = eye(2, dtype=np.int32)

    assert I._data.dtype == np.int32
    print("✓ Identity matrix with dtype test passed")


def test_arange():
    """Test arange function"""
    print("\n" + "=" * 60)
    print("Testing Arange")
    print("=" * 60)

    # Test 1: Single argument (stop)
    x = arange(5)

    assert isinstance(x, Tensor)
    assert x.shape == (5,)  # 1次元配列
    assert np.allclose(x._data, np.array([0, 1, 2, 3, 4]))
    print("✓ Arange with stop test passed")

    # Test 2: Start and stop
    x = arange(2, 7)

    assert x.shape == (5,)  # 1次元配列
    assert np.allclose(x._data, np.array([2, 3, 4, 5, 6]))
    print("✓ Arange with start and stop test passed")

    # Test 3: Start, stop, and step
    x = arange(0, 10, 2)

    assert x.shape == (5,)  # 1次元配列
    assert np.allclose(x._data, np.array([0, 2, 4, 6, 8]))
    print("✓ Arange with step test passed")

    # Test 4: Float values
    x = arange(0.0, 1.0, 0.25)

    assert x.shape == (4,)  # 1次元配列
    assert np.allclose(x._data, np.array([0.0, 0.25, 0.5, 0.75]))
    print("✓ Arange with float test passed")


def test_linspace():
    """Test linspace function"""
    print("\n" + "=" * 60)
    print("Testing Linspace")
    print("=" * 60)

    # Test 1: Basic linspace
    x = linspace(0, 1, 5)

    assert isinstance(x, Tensor)
    assert x.shape == (5,)  # 1次元配列
    assert np.allclose(x._data, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    print("✓ Basic linspace test passed")

    # Test 2: Different range
    x = linspace(-1, 1, 11)

    assert x.shape == (11,)  # 1次元配列
    expected = np.linspace(-1, 1, 11)
    assert np.allclose(x._data, expected)
    print("✓ Linspace with different range test passed")

    # Test 3: With dtype
    x = linspace(0, 10, 5, dtype=np.int32)

    assert x._data.dtype == np.int32
    print("✓ Linspace with dtype test passed")


def test_concatenate():
    # Test 3: Concatenate column vectors vertically
    v1 = vector([1, 2, 3])  # (3, 1)
    v2 = vector([4, 5, 6])  # (3, 1)
    v3 = concatenate([v1, v2], axis=0)  # (6, 1)

    assert v3.shape == (6, 1)
    expected = np.array([[1], [2], [3], [4], [5], [6]])
    assert np.allclose(v3._data, expected)
    print("✓ Concatenate column vectors test passed")

    # Test 4: Concatenate column vectors horizontally
    v4 = concatenate([v1, v2], axis=1)  # (3, 2) - becomes a Matrix
    assert v4.shape == (3, 2)
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    assert np.allclose(v4._data, expected)
    print("✓ Concatenate vectors horizontally test passed")


def test_stack():
    """Test stack function"""
    print("\n" + "=" * 60)
    print("Testing Stack")
    print("=" * 60)

    # Test 1: Stack along axis 0
    a = matrix([[1, 2], [3, 4]])
    b = matrix([[5, 6], [7, 8]])
    c = stack([a, b], axis=0)

    assert isinstance(c, Tensor)
    assert c.shape == (2, 2, 2)
    expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert np.allclose(c._data, expected)
    print("✓ Stack along axis 0 test passed")

    # Test 2: Stack along axis 1
    c = stack([a, b], axis=1)

    assert c.shape == (2, 2, 2)
    expected = np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]]])
    assert np.allclose(c._data, expected)
    print("✓ Stack along axis 1 test passed")

    # Test 3: Stack vectors (列ベクトル)
    v1 = vector([1, 2, 3])  # shape: (3, 1)
    v2 = vector([4, 5, 6])  # shape: (3, 1)
    v3 = stack([v1, v2], axis=0)

    assert v3.shape == (2, 3, 1)  # 2つの(3,1)ベクトルをaxis=0でスタック
    expected = np.array([[[1], [2], [3]], [[4], [5], [6]]])
    assert np.allclose(v3._data, expected)
    print("✓ Stack vectors along axis 0 test passed")

    # Test 4: Stack vectors along axis 1 (横に並べる)
    v4 = stack([v1, v2], axis=1)

    assert v4.shape == (3, 2, 1)  # axis=1でスタック
    expected = np.array([[[1], [4]], [[2], [5]], [[3], [6]]])
    assert np.allclose(v4._data, expected)
    print("✓ Stack vectors along axis 1 test passed")

    # Test 5: もし1次元配列として扱いたい場合
    # flatten してからスタック
    v1_flat = v1.reshape(-1)  # (3,)
    v2_flat = v2.reshape(-1)  # (3,)
    v5 = stack([v1_flat, v2_flat], axis=0)

    assert v5.shape == (2, 3)
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(v5._data, expected)
    print("✓ Stack flattened vectors test passed")


def test_bmm():
    """Test batch matrix multiplication"""
    print("\n" + "=" * 60)
    print("Testing Batch Matrix Multiplication")
    print("=" * 60)

    # Test 1: Basic batch matmul
    batch_size = 2
    n, m, p = 3, 4, 5

    a_data = np.random.randn(batch_size, n, m)
    b_data = np.random.randn(batch_size, m, p)

    a = tensor(a_data)
    b = tensor(b_data)
    c = bmm(a, b)

    assert isinstance(c, Tensor)
    assert c.shape == (batch_size, n, p)

    # Verify correctness
    expected = a_data @ b_data
    assert np.allclose(c._data, expected)
    print("✓ Basic batch matmul test passed")

    # Test 2: Error on wrong dimensions
    try:
        a_2d = matrix([[1, 2], [3, 4]])
        b_3d = tensor(np.random.randn(2, 2, 2))
        bmm(a_2d, b_3d)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "must be a 3D tensor" in str(e)
        print("✓ Dimension check test passed")

    # Test 3: Error on batch size mismatch
    try:
        a = tensor(np.random.randn(2, 3, 4))
        b = tensor(np.random.randn(3, 4, 5))
        bmm(a, b)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Batch size mismatch" in str(e)
        print("✓ Batch size check test passed")


def test_einsum():
    """Test Einstein summation"""
    print("\n" + "=" * 60)
    print("Testing Einstein Summation")
    print("=" * 60)

    # Test 1: Matrix multiplication
    a = matrix([[1, 2], [3, 4]])
    b = matrix([[5, 6], [7, 8]])
    c = einsum("ij,jk->ik", a, b)

    assert isinstance(c, Tensor)
    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(c._data, expected)
    print("✓ Matrix multiplication test passed")

    # Test 2: Trace
    a = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    trace = einsum("ii->", a)

    assert trace._data.shape == ()
    assert np.allclose(trace._data, 15)  # 1 + 5 + 9
    print("✓ Trace test passed")

    # Test 3: Element-wise multiplication
    a = matrix([[1, 2], [3, 4]])
    b = matrix([[5, 6], [7, 8]])
    c = einsum("ij,ij->ij", a, b)

    expected = np.array([[5, 12], [21, 32]])
    assert np.allclose(c._data, expected)
    print("✓ Element-wise multiplication test passed")

    # Test 4: Batch matrix multiplication
    a = tensor(np.random.randn(2, 3, 4))
    b = tensor(np.random.randn(2, 4, 5))
    c = einsum("bij,bjk->bik", a, b)

    assert c.shape == (2, 3, 5)
    expected = a._data @ b._data
    assert np.allclose(c._data, expected)
    print("✓ Batch matrix multiplication test passed")

    # Test 5: Outer product
    # Vectorは(n, 1)形状なので、flattenするか2次元用の添字を使用
    a = vector([1, 2, 3])  # (3, 1)
    b = vector([4, 5])  # (2, 1)

    # 方法1: flattenしてから計算
    a_flat = a.reshape(-1)  # (3,)
    b_flat = b.reshape(-1)  # (2,)
    c = einsum("i,j->ij", a_flat, b_flat)

    expected = np.array([[4, 5], [8, 10], [12, 15]])
    assert np.allclose(c._data, expected)
    print("✓ Outer product test passed")

    # Test 6: Vector内積（2次元のまま）
    v1 = vector([1, 2, 3])  # (3, 1)
    v2 = vector([4, 5, 6])  # (3, 1)

    # 2次元用の添字を使用: (3,1).T @ (3,1) = スカラー
    inner = einsum("ji,ji->", v1, v2)  # sum over both dimensions

    expected = 32  # 1*4 + 2*5 + 3*6
    assert np.allclose(inner._data, expected)
    print("✓ Vector inner product (2D) test passed")

    # Test 7: Vector外積（2次元のまま）
    # (3,1) @ (1,2) -> (3,2)
    v1 = vector([1, 2, 3])  # (3, 1)
    v2_row = rowvec([4, 5])  # (1, 2)
    outer = einsum("ij,jk->ik", v1, v2_row)

    expected = np.array([[4, 5], [8, 10], [12, 15]])
    assert np.allclose(outer._data, expected)
    print("✓ Vector outer product (2D) test passed")


def test_tensordot():
    """Test tensor dot product"""
    print("\n" + "=" * 60)
    print("Testing Tensor Dot Product")
    print("=" * 60)

    # Test 1: Matrix multiplication (axes=1)
    a = matrix([[1, 2], [3, 4]])
    b = matrix([[5, 6], [7, 8]])
    c = tensordot(a, b, axes=1)

    assert isinstance(c, Tensor)
    # tensordot with axes=1 is like a @ b.T
    expected = np.tensordot(a._data, b._data, axes=1)
    assert np.allclose(c._data, expected)
    print("✓ Tensordot with axes=1 test passed")

    # Test 2: Inner product (axes=2)
    a = matrix([[1, 2], [3, 4]])
    b = matrix([[5, 6], [7, 8]])
    c = tensordot(a, b, axes=2)

    expected = np.tensordot(a._data, b._data, axes=2)
    assert np.allclose(c._data, expected)
    print("✓ Tensordot with axes=2 test passed")

    # Test 3: Specified axes
    a = tensor(np.random.randn(3, 4, 5))
    b = tensor(np.random.randn(4, 5, 6))
    c = tensordot(a, b, axes=([1, 2], [0, 1]))

    expected = np.tensordot(a._data, b._data, axes=([1, 2], [0, 1]))
    assert c.shape == (3, 6)
    assert np.allclose(c._data, expected)
    print("✓ Tensordot with specified axes test passed")

    # Test 4: Vector dot product
    # Vectorは(n, 1)形状なので、適切に処理する必要がある
    a = vector([1, 2, 3])  # (3, 1)
    b = vector([4, 5, 6])  # (3, 1)

    # 方法1: flattenしてから計算
    a_flat = a.reshape(-1)  # (3,)
    b_flat = b.reshape(-1)  # (3,)
    c = tensordot(a_flat, b_flat, axes=1)

    expected = 32  # 1*4 + 2*5 + 3*6
    assert np.allclose(c._data, expected)
    print("✓ Vector dot product (flattened) test passed")

    # 方法2: 明示的な軸指定で2次元のまま計算
    # (3, 1)と(3, 1)の最初の次元で縮約
    c2 = tensordot(a, b, axes=([0], [0]))
    expected2 = np.array([[32]])  # (1, 1)の結果
    assert np.allclose(c2._data, expected2)
    print("✓ Vector dot product (2D with axes) test passed")

    # Test 5: Vector outer product
    a = vector([1, 2, 3])  # (3, 1)
    b = vector([4, 5])  # (2, 1)

    # axes=0 で外積
    c = tensordot(a, b, axes=0)
    expected = np.array([[[[4], [5]]], [[[8], [10]]], [[[12], [15]]]])  # (3, 1, 2, 1)
    assert c.shape == (3, 1, 2, 1)
    assert np.allclose(c._data, expected)
    print("✓ Vector outer product test passed")


def test_utility_functions():
    """Run all utility function tests"""
    print("\n" + "=" * 70)
    print(" " * 20 + "UTILITY FUNCTIONS TESTS")
    print("=" * 70)

    test_one_hot()
    test_eye()
    test_arange()
    test_linspace()
    test_concatenate()
    test_stack()
    test_bmm()
    test_einsum()
    test_tensordot()

    print("\n" + "=" * 70)
    print(" " * 15 + "ALL UTILITY FUNCTIONS TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_utility_functions()
