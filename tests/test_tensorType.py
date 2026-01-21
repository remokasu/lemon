import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import pytest
import numpy as np
from lemon.numlib import *


class TestMathematicalTypeCorrectness:
    """
    原則:
    1. Vector は列ベクトル (n, 1)
    2. RowVector は行ベクトル (1, n)
    3. Matrix は2次元配列 (m, n)
    4. Tensor は任意次元配列
    """

    def setup_method(self):
        """各テストの前に実行"""
        autograd.enable()

    # ========================================
    # 基本的な型と形状の保証
    # ========================================

    def test_vector_is_column_vector(self):
        """Vectorは常に列ベクトル(n, 1)である"""
        v = vector([1, 2, 3])
        assert isinstance(v, Vector)
        assert v.shape == (3, 1)
        assert v.ndim == 2

        # スカラーから作成
        v_scalar = vector(5)
        assert v_scalar.shape == (1, 1)

    def test_rowvector_is_row_vector(self):
        """RowVectorは常に行ベクトル(1, n)である"""
        rv = rowvec([1, 2, 3])
        assert isinstance(rv, RowVector)
        assert rv.shape == (1, 3)
        assert rv.ndim == 2

    def test_matrix_is_2d(self):
        """Matrixは常に2次元配列である"""
        m = matrix([[1, 2], [3, 4]])
        assert isinstance(m, Matrix)
        assert m.ndim == 2

        # 1次元入力も2次元になる
        m_1d = matrix([1, 2, 3])
        assert m_1d.shape == (1, 3)
        assert m_1d.ndim == 2

    # ========================================
    # 転置の数学的正しさ
    # ========================================

    def test_vector_transpose_gives_rowvector(self):
        """列ベクトルの転置は行ベクトル"""
        v = vector([1, 2, 3])
        vT = v.T

        assert isinstance(vT, RowVector)
        assert vT.shape == (1, 3)

    def test_rowvector_transpose_gives_vector(self):
        """行ベクトルの転置は列ベクトル"""
        rv = rowvec([1, 2, 3])
        rvT = rv.T

        assert isinstance(rvT, Vector)
        assert rvT.shape == (3, 1)

    def test_matrix_transpose_preserves_matrix(self):
        """行列の転置は行列のまま"""
        m = matrix([[1, 2, 3], [4, 5, 6]])
        mT = m.T

        assert isinstance(mT, Matrix)
        assert mT.shape == (3, 2)

    def test_double_transpose_identity(self):
        """二重転置は恒等変換"""
        v = vector([1, 2, 3])
        assert isinstance(v.T.T, Vector)
        np.testing.assert_array_equal(v._data, v.T.T._data)

        m = matrix([[1, 2], [3, 4]])
        assert isinstance(m.T.T, Matrix)
        np.testing.assert_array_equal(m._data, m.T.T._data)

    # ========================================
    # 行列積の数学的正しさ
    # ========================================

    def test_matrix_vector_product(self):
        """行列×ベクトル = ベクトル"""
        m = matrix([[1, 2], [3, 4]])  # (2, 2)
        v = vector([5, 6])  # (2, 1)

        result = m @ v  # (2, 2) @ (2, 1) = (2, 1)
        assert isinstance(result, Vector)
        assert result.shape == (2, 1)

    def test_rowvector_matrix_product(self):
        """行ベクトル×行列 = 行ベクトル"""
        rv = rowvec([1, 2])  # (1, 2)
        m = matrix([[3, 4], [5, 6]])  # (2, 2)

        result = rv @ m  # (1, 2) @ (2, 2) = (1, 2)
        assert isinstance(result, RowVector)
        assert result.shape == (1, 2)

    def test_vector_rowvector_outer_product(self):
        """ベクトル×行ベクトル = 行列（外積）"""
        v = vector([1, 2, 3])  # (3, 1)
        rv = rowvec([4, 5])  # (1, 2)

        result = v @ rv  # (3, 1) @ (1, 2) = (3, 2)
        assert isinstance(result, Matrix)
        assert result.shape == (3, 2)

    def test_rowvector_vector_inner_product(self):
        """行ベクトル×ベクトル = スカラー（内積）"""
        rv = rowvec([1, 2, 3])  # (1, 3)
        v = vector([4, 5, 6])  # (3, 1)

        result = rv @ v  # (1, 3) @ (3, 1) = スカラー
        assert isinstance(result, Scalar)
        assert result._data.shape == () or result._data.shape == (1, 1)

    def test_matrix_matrix_product(self):
        """行列×行列 = 行列"""
        m1 = matrix([[1, 2], [3, 4]])  # (2, 2)
        m2 = matrix([[5, 6], [7, 8]])  # (2, 2)

        result = m1 @ m2
        assert isinstance(result, Matrix)
        assert result.shape == (2, 2)

    # ========================================
    # 要素ごと演算の型保存
    # ========================================

    def test_vector_elementwise_preserves_type(self):
        """ベクトルの要素ごと演算は型を保存"""
        v1 = vector([1, 2, 3])
        v2 = vector([4, 5, 6])

        assert isinstance(v1 + v2, Vector)
        assert isinstance(v1 - v2, Vector)
        assert isinstance(v1 * v2, Vector)
        assert isinstance(v1 / v2, Vector)

    def test_matrix_elementwise_preserves_type(self):
        """行列の要素ごと演算は型を保存"""
        m1 = matrix([[1, 2], [3, 4]])
        m2 = matrix([[5, 6], [7, 8]])

        assert isinstance(m1 + m2, Matrix)
        assert isinstance(m1 - m2, Matrix)
        assert isinstance(m1 * m2, Matrix)
        assert isinstance(m1 / m2, Matrix)

    # ========================================
    # ブロードキャスティングの数学的正しさ
    # ========================================

    def test_scalar_vector_broadcasting(self):
        """スカラー×ベクトル = ベクトル"""
        s = real(2.0)
        v = vector([1, 2, 3])

        result = s * v
        assert isinstance(result, Vector)
        assert result.shape == (3, 1)

    def test_vector_matrix_broadcasting(self):
        """ベクトルと行列のブロードキャスト"""
        v = vector([1, 2])  # (2, 1)
        m = matrix([[3, 4, 5], [6, 7, 8]])  # (2, 3)

        # (2, 1) + (2, 3) = (2, 3)
        result = v + m
        assert isinstance(result, Matrix)
        assert result.shape == (2, 3)

    # ========================================
    # 数学的に無効な操作の検出
    # ========================================

    def test_vector_vector_matmul_invalid(self):
        """ベクトル @ ベクトル は無効"""
        v1 = vector([1, 2, 3])
        v2 = vector([4, 5, 6])

        with pytest.raises(ValueError):
            v1 @ v2

    def test_dimension_mismatch_in_matmul(self):
        """次元が合わない行列積はエラー"""
        m1 = matrix([[1, 2, 3]])  # (1, 3)
        m2 = matrix([[4, 5]])  # (1, 2)

        with pytest.raises(ValueError):
            m1 @ m2

    # ========================================
    # インデックスアクセスの数学的意味
    # ========================================

    def test_matrix_row_access(self):
        """行列の行アクセスは1次元配列"""
        m = matrix([[1, 2, 3], [4, 5, 6]])
        row = m[0]

        # 注: 現在の実装ではTensorになる（型情報を失う）
        # これは設計上の選択
        assert row.shape == (3,) or row.shape == (1, 3)

    def test_matrix_column_access(self):
        """行列の列アクセス"""
        m = matrix([[1, 2, 3], [4, 5, 6]])
        col = m[:, 0]

        # 列は(n, 1)形状になるべき
        assert col.shape == (2,) or col.shape == (2, 1)

    # ========================================
    # 数学的恒等式の検証
    # ========================================

    def test_transpose_product_identity(self):
        """(AB)^T = B^T A^T"""
        A = matrix([[1, 2], [3, 4]])
        B = matrix([[5, 6], [7, 8]])

        left = (A @ B).T
        right = B.T @ A.T

        np.testing.assert_array_almost_equal(left._data, right._data)

    def test_inner_product_commutativity(self):
        """内積の可換性: ⟨u, v⟩ = ⟨v, u⟩"""
        u = vector([1, 2, 3])
        v = vector([4, 5, 6])

        inner1 = u.T @ v  # RowVector @ Vector
        inner2 = v.T @ u  # RowVector @ Vector

        np.testing.assert_almost_equal(inner1._data, inner2._data)
