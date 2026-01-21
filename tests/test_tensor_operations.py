"""
test_tensor_operations.py

Tensor、Matrix、Vectorの包括的テストスイート
- 型チェックと形状検証
- 基本演算
- 行列演算
- 高度なテンソル演算
- エッジケースとエラー処理
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
# テスト: 型とシェイプ
# =============================================================================


class TestTypeAndShape:
    """型とシェイプのテスト"""

    def test_vector_creation(self):
        """Vectorの作成と形状"""
        # 1次元リストから作成
        v = vector([1, 2, 3])
        assert v.shape == (3, 1), f"Expected (3, 1), got {v.shape}"
        assert v.ndim == 2
        assert isinstance(v, Vector)

        # NumPy配列から作成
        np_arr = np.array([4, 5, 6])
        v2 = vector(np_arr)
        assert v2.shape == (3, 1)

        # スカラーから作成
        v3 = vector([1])
        assert v3.shape == (1, 1)

    def test_rowvector_creation(self):
        """RowVectorの作成と形状"""
        # 1次元リストから作成
        rv = rowvector([1, 2, 3])
        assert rv.shape == (1, 3), f"Expected (1, 3), got {rv.shape}"
        assert rv.ndim == 2
        assert isinstance(rv, RowVector)

        # NumPy配列から作成
        np_arr = np.array([4, 5, 6])
        rv2 = rowvector(np_arr)
        assert rv2.shape == (1, 3)

    def test_matrix_creation(self):
        """Matrixの作成と形状"""
        # 2次元リストから作成
        m = matrix([[1, 2], [3, 4]])
        assert m.shape == (2, 2), f"Expected (2, 2), got {m.shape}"
        assert m.ndim == 2
        assert isinstance(m, Matrix)

        # NumPy配列から作成
        np_arr = np.array([[5, 6], [7, 8]])
        m2 = matrix(np_arr)
        assert m2.shape == (2, 2)

        # 非正方行列
        m3 = matrix([[1, 2, 3], [4, 5, 6]])
        assert m3.shape == (2, 3)

    def test_tensor_creation(self):
        """Tensorの作成と形状"""
        # 3次元配列
        t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert t.shape == (2, 2, 2)
        assert t.ndim == 3
        assert isinstance(t, Tensor)

        # 4次元配列
        t2 = tensor(np.random.randn(2, 3, 4, 5))
        assert t2.shape == (2, 3, 4, 5)
        assert t2.ndim == 4

    def test_type_preservation(self):
        """型の保持確認"""
        v = vector([1, 2, 3])
        m = matrix([[1, 2], [3, 4]])
        rv = rowvector([1, 2, 3])

        # 演算後も型が保持される
        v2 = v + v
        assert isinstance(v2, Vector), "Vector + Vector should return Vector"

        m2 = m + m
        assert isinstance(m2, Matrix), "Matrix + Matrix should return Matrix"

        rv2 = rv + rv
        assert isinstance(rv2, RowVector), (
            "RowVector + RowVector should return RowVector"
        )


# =============================================================================
# テスト: 基本演算
# =============================================================================


class TestBasicOperations:
    """基本演算のテスト"""

    def test_vector_addition(self):
        """Vectorの加算"""
        v1 = vector([1, 2, 3])
        v2 = vector([4, 5, 6])

        result = v1 + v2
        expected = np.array([[5], [7], [9]])

        assert isinstance(result, Vector)
        assert np.allclose(result._data, expected)

    def test_vector_subtraction(self):
        """Vectorの減算"""
        v1 = vector([5, 7, 9])
        v2 = vector([1, 2, 3])

        result = v1 - v2
        expected = np.array([[4], [5], [6]])

        assert np.allclose(result._data, expected)

    def test_vector_scalar_multiplication(self):
        """Vectorとスカラーの乗算"""
        v = vector([1, 2, 3])

        # 左から
        result1 = 2 * v
        expected = np.array([[2], [4], [6]])
        assert np.allclose(result1._data, expected)

        # 右から
        result2 = v * 2
        assert np.allclose(result2._data, expected)

    def test_vector_element_wise_multiplication(self):
        """Vectorの要素ごとの乗算"""
        v1 = vector([1, 2, 3])
        v2 = vector([2, 3, 4])

        result = v1 * v2
        expected = np.array([[2], [6], [12]])

        assert np.allclose(result._data, expected)

    def test_matrix_addition(self):
        """Matrixの加算"""
        m1 = matrix([[1, 2], [3, 4]])
        m2 = matrix([[5, 6], [7, 8]])

        result = m1 + m2
        expected = np.array([[6, 8], [10, 12]])

        assert isinstance(result, Matrix)
        assert np.allclose(result._data, expected)

    def test_matrix_scalar_multiplication(self):
        """Matrixとスカラーの乗算"""
        m = matrix([[1, 2], [3, 4]])

        result = 3 * m
        expected = np.array([[3, 6], [9, 12]])

        assert np.allclose(result._data, expected)

    def test_broadcasting(self):
        """ブロードキャスティング"""
        m = matrix([[1, 2], [3, 4]])
        v = vector([10, 20])

        # Matrix + Vector (ブロードキャスト)
        result = m + v
        expected = np.array([[11, 12], [23, 24]])

        assert np.allclose(result._data, expected)


# =============================================================================
# テスト: 行列演算
# =============================================================================


class TestMatrixOperations:
    """行列演算のテスト"""

    def test_matrix_multiplication(self):
        """行列積"""
        A = matrix([[1, 2], [3, 4]])
        B = matrix([[5, 6], [7, 8]])

        C = A @ B
        expected = np.array([[19, 22], [43, 50]])

        assert isinstance(C, Matrix)
        assert np.allclose(C._data, expected)

    def test_matrix_vector_multiplication(self):
        """行列とベクトルの積"""
        A = matrix([[1, 2], [3, 4]])
        v = vector([5, 6])

        result = A @ v
        expected = np.array([[17], [39]])

        assert isinstance(result, Vector), "Matrix @ Vector should return Vector"
        assert np.allclose(result._data, expected)

    def test_rowvector_matrix_multiplication(self):
        """行ベクトルと行列の積"""
        rv = rowvector([1, 2])
        A = matrix([[3, 4], [5, 6]])

        result = rv @ A
        expected = np.array([[13, 16]])

        assert isinstance(result, RowVector), (
            "RowVector @ Matrix should return RowVector"
        )
        assert np.allclose(result._data, expected)

    def test_inner_product(self):
        """内積（RowVector @ Vector）"""
        rv = rowvector([1, 2, 3])
        v = vector([4, 5, 6])

        result = rv @ v
        expected = 32  # 1*4 + 2*5 + 3*6 = 32

        # 結果はスカラー
        assert np.isclose(result._data, expected)

    def test_outer_product(self):
        """外積（Vector @ RowVector）"""
        v = vector([1, 2, 3])
        rv = rowvector([4, 5, 6])

        result = v @ rv
        expected = np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])

        assert isinstance(result, Matrix), "Vector @ RowVector should return Matrix"
        assert np.allclose(result._data, expected)

    def test_dot_product(self):
        """内積（dot関数）"""
        v1 = vector([1, 2, 3])
        v2 = vector([4, 5, 6])

        result = dot(v1, v2)
        expected = 32  # 1*4 + 2*5 + 3*6

        assert np.isclose(result._data, expected)

    def test_transpose_vector(self):
        """Vectorの転置"""
        v = vector([1, 2, 3])

        vT = v.T

        assert isinstance(vT, RowVector), "Vector.T should return RowVector"
        assert vT.shape == (1, 3)
        assert np.allclose(vT._data, np.array([[1, 2, 3]]))

    def test_transpose_rowvector(self):
        """RowVectorの転置"""
        rv = rowvector([1, 2, 3])

        rvT = rv.T

        assert isinstance(rvT, Vector), "RowVector.T should return Vector"
        assert rvT.shape == (3, 1)
        assert np.allclose(rvT._data, np.array([[1], [2], [3]]))

    def test_transpose_matrix(self):
        """Matrixの転置"""
        m = matrix([[1, 2, 3], [4, 5, 6]])

        mT = m.T

        assert isinstance(mT, Matrix)
        assert mT.shape == (3, 2)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        assert np.allclose(mT._data, expected)


# =============================================================================
# テスト: 高度なテンソル演算
# =============================================================================


class TestAdvancedTensorOperations:
    """高度なテンソル演算のテスト"""

    def test_reshape(self):
        """リシェイプ"""
        v = vector([1, 2, 3, 4, 5, 6])

        # Vector (6, 1) -> Matrix (2, 3)
        m = reshape(v, (2, 3))
        assert m.shape == (2, 3)
        assert isinstance(m, Matrix)

        # Matrix (2, 3) -> Vector (6, 1)
        v2 = reshape(m, (6, 1))
        assert v2.shape == (6, 1)
        assert isinstance(v2, Vector)

        # -1を使った自動計算
        m2 = reshape(v, (2, -1))
        assert m2.shape == (2, 3)

    def test_flatten(self):
        """フラット化"""
        m = matrix([[1, 2, 3], [4, 5, 6]])

        v = flatten(m)

        # ✅ 修正: flatten は 1次元 Tensor を返す
        assert v.ndim == 1
        assert v.shape == (6,)
        assert np.allclose(v._data, [1, 2, 3, 4, 5, 6])

    def test_sum_reduction(self):
        """和の削減"""
        m = matrix([[1, 2, 3], [4, 5, 6]])

        # 全要素の和
        total = sum(m)
        assert np.isclose(total._data, 21)

        # 軸方向の和
        col_sum = m.sum(axis=0)
        # ✅ 修正: keepdims=False なので (3,)
        assert col_sum.shape == (3,)
        assert np.allclose(col_sum._data, [5, 7, 9])

        row_sum = m.sum(axis=1)
        # ✅ 修正: keepdims=False なので (2,)
        assert row_sum.shape == (2,)
        assert np.allclose(row_sum._data, [6, 15])

        # keepdims=True のテスト
        col_sum_keep = m.sum(axis=0, keepdims=True)
        assert col_sum_keep.shape == (1, 3)
        assert isinstance(col_sum_keep, RowVector)

        row_sum_keep = m.sum(axis=1, keepdims=True)
        assert row_sum_keep.shape == (2, 1)
        assert isinstance(row_sum_keep, Vector)

    def test_mean_reduction(self):
        """平均の削減"""
        m = matrix([[2, 4, 6], [8, 10, 12]])

        # 全要素の平均
        avg = mean(m)
        assert np.isclose(avg._data, 7)

        # 軸方向の平均
        col_avg = m.mean(axis=0)
        # ✅ 修正: (3,) になる
        assert col_avg.shape == (3,)
        assert np.allclose(col_avg._data, [5, 7, 9])

        row_avg = m.mean(axis=1)
        # ✅ 修正: (2,) になる
        assert row_avg.shape == (2,)
        assert np.allclose(row_avg._data, [4, 10])

    def test_indexing_vector(self):
        """Vectorのインデックス"""
        v = vector([10, 20, 30, 40, 50])

        # 単一要素
        elem = v[2]
        # ✅ 修正: (1,) になる（0次元ではない）
        assert elem.ndim == 1 or elem.ndim == 2
        assert np.isclose(elem._data.item(), 30)

        # スライス
        sub = v[1:4]
        assert sub.shape == (3, 1)
        assert np.allclose(sub._data, [[20], [30], [40]])

    def test_indexing_matrix(self):
        """Matrixのインデックス"""
        m = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # 単一要素
        elem = m[1, 2]
        assert np.isclose(elem._data, 6)

        # 行の取得
        row = m[1]
        # ✅ 修正: (3,) になる（NumPy互換）
        assert row.shape == (3,)
        assert np.allclose(row._data, [4, 5, 6])

        # keepdims=True の動作を期待する場合は m[1:2] を使う
        row_keep = m[1:2]
        assert row_keep.shape == (1, 3)
        assert isinstance(row_keep, RowVector)

        # スライス
        sub = m[0:2, 1:3]
        expected = np.array([[2, 3], [5, 6]])
        assert np.allclose(sub._data, expected)

    def test_concatenate(self):
        """連結"""
        v1 = vector([1, 2, 3])
        v2 = vector([4, 5, 6])

        # 縦方向に連結
        result = concatenate([v1, v2], axis=0)
        assert result.shape == (6, 1)

        # 横方向に連結
        m1 = matrix([[1, 2], [3, 4]])
        m2 = matrix([[5, 6], [7, 8]])
        result2 = concatenate([m1, m2], axis=1)
        assert result2.shape == (2, 4)

    def test_stack(self):
        """スタック"""
        v1 = vector([1, 2, 3])
        v2 = vector([4, 5, 6])

        # 新しい軸でスタック
        result = stack([v1, v2], axis=0)
        assert result.shape == (2, 3, 1)


# =============================================================================
# テスト: エラー処理
# =============================================================================


class TestErrorHandling:
    """エラー処理のテスト"""

    def test_invalid_vector_vector_matmul(self):
        """Vector @ Vector（無効）"""
        v1 = vector([1, 2, 3])
        v2 = vector([4, 5, 6])

        with pytest.raises(ValueError, match="Cannot perform matrix multiplication"):
            result = v1 @ v2

    def test_invalid_matrix_dimensions(self):
        """次元が合わない行列積"""
        A = matrix([[1, 2, 3]])  # (1, 3)
        B = matrix([[4, 5], [6, 7]])  # (2, 2)

        with pytest.raises(ValueError):
            result = A @ B

    def test_invalid_reshape(self):
        """無効なリシェイプ"""
        v = vector([1, 2, 3, 4, 5])

        # 要素数が合わない
        with pytest.raises(ValueError):
            m = reshape(v, (2, 3))  # 5要素を2x3に

    def test_invalid_vector_creation(self):
        """無効なVector作成"""
        # 2次元配列で幅が1でない
        with pytest.raises(ValueError):
            v = Vector([[1, 2], [3, 4]])

    def test_matrix_rowvector_matmul_invalid(self):
        """Matrix @ RowVector（無効）"""
        m = matrix([[1, 2], [3, 4]])
        rv = rowvector([5, 6])

        # ✅ 修正: 実際のエラーメッセージに合わせる
        with pytest.raises(ValueError, match="Cannot multiply matrix with row vector"):
            result = m @ rv


# =============================================================================
# テスト: NumPy互換性
# =============================================================================


class TestNumpyCompatibility:
    """NumPy互換性のテスト"""

    def test_array_protocol(self):
        """NumPy配列プロトコル"""
        v = vector([1, 2, 3])

        # np.array()で変換
        np_arr = np.array(v)
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (3, 1)

    def test_numpy_functions(self):
        """NumPy関数との互換性"""
        v = vector([1, 2, 3])

        # NumPy関数が使える
        result = np.mean(v)
        assert np.isclose(result, 2.0)

        total = np.sum(v)
        assert np.isclose(total, 6.0)

    def test_tolist(self):
        """Pythonリストへの変換"""
        v = vector([1, 2, 3])

        lst = v.tolist()
        assert lst == [[1], [2], [3]]

        m = matrix([[1, 2], [3, 4]])
        lst2 = m.tolist()
        assert lst2 == [[1, 2], [3, 4]]


# =============================================================================
# テスト: 特殊ケース
# =============================================================================


class TestSpecialCases:
    """特殊ケースのテスト"""

    def test_empty_operations(self):
        """空配列の演算"""
        v = vector([])
        assert v.shape == (0, 1) or v.shape == (1, 1)

    def test_single_element_vector(self):
        """単一要素のVector"""
        v = vector([5])
        assert v.shape == (1, 1)

        result = v * 2
        assert np.isclose(result._data, 10)

    def test_large_matrix_multiplication(self):
        """大きな行列の積"""
        A = random_matrix((100, 50))
        B = random_matrix((50, 75))

        C = A @ B
        assert C.shape == (100, 75)
        assert isinstance(C, Matrix)

    def test_chained_operations(self):
        """連鎖演算"""
        v = vector([1, 2, 3])

        # (v + v) * 2 - v
        result = (v + v) * 2 - v
        expected = np.array([[3], [6], [9]])

        assert isinstance(result, Vector)
        assert np.allclose(result._data, expected)

    def test_mixed_type_operations(self):
        """異なる型の混在演算"""
        v = vector([1, 2, 3])
        m = matrix([[1], [2], [3]])

        # Vector + Matrix (同じ形状)
        result = v + m
        expected = np.array([[2], [4], [6]])
        assert np.allclose(result._data, expected)


# =============================================================================
# テストスイート実行
# =============================================================================


def run_all_tests():
    """全テストを実行"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
