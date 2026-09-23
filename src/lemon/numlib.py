# -----------------------------------------------------------------------------
# numlib - MIT License
#
# Copyright (c) 2025 @remokasu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""
numlib - A NumPy/CuPy-based numerical computation library with automatic differentiation

This library provides a Tensor class for array operations with automatic gradient computation,
supporting both CPU (NumPy) and GPU (CuPy) backends. It includes basic mathematical operations,
linear algebra functions, and neural network building blocks.

Key Features
------------
- Automatic differentiation for gradient-based optimization
- GPU acceleration support via CuPy
- NumPy-compatible API for tensor operations
- Gradient computation control and device management
"""

__version__ = "0.0.2"
__author__ = "@remokasu"
__email__ = "0w0.ebi.kaitai@gmail.com"
__homepage__ = "https://github.com/remokasu/numlib"
__license__ = "MIT"

import threading
import numpy as np
import builtins
from typing import Any, Union, Tuple

# ==============================
# GPU Support Detection
# ==============================
try:
    import cupy as cp

    ArrayType = Union[np.ndarray, cp.ndarray]
except ImportError:
    cp = None
    ArrayType = np.ndarray

_cuda_enabled = False


def cuda_available() -> bool:
    """
    Check if CUDA is available on the system

    Returns
    -------
    bool
        True if CUDA is available and at least one GPU is detected, False otherwise
    """
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def is_gpu_enabled() -> bool:
    """
    Check if GPU acceleration is currently enabled

    Returns
    -------
    bool
        True if GPU mode is enabled, False if running in CPU mode
    """
    return _cuda_enabled


def enable_gpu() -> None:
    """
    Enable GPU acceleration for tensor operations

    Raises
    ------
    RuntimeError
        If CUDA is not available or no GPU is detected
    """
    global _cuda_enabled
    if not cuda_available():
        raise RuntimeError("CUDA is not available or no GPU detected")
    _cuda_enabled = True


def disable_gpu() -> None:
    """
    Disable GPU acceleration and switch to CPU mode for tensor operations
    """
    global _cuda_enabled
    _cuda_enabled = False


class CudaControl:
    """CUDA state control"""

    def __init__(self, device: str):
        self.device = device

    def __call__(self):
        if self.device == "cpu":
            disable_gpu()
        else:
            enable_gpu()
        return self

    def __enter__(self):
        self.prev_state = is_gpu_enabled()
        try:
            if self.device == "cpu":
                disable_gpu()
            else:
                enable_gpu()
        except RuntimeError:
            disable_gpu()
            raise
        return self

    def __exit__(self, *args):
        if self.prev_state:
            enable_gpu()
        else:
            disable_gpu()


class CudaNamespace:
    """CUDA/GPU control namespace"""

    def __init__(self):
        self._cpu = CudaControl("cpu")
        self._gpu = CudaControl("gpu")

    @property
    def cpu(self):
        return self._cpu

    @property
    def gpu(self):
        return self._gpu

    def is_available(self) -> bool:
        return cuda_available()

    def is_enabled(self) -> bool:
        return is_gpu_enabled()

    def enable(self) -> None:
        enable_gpu()

    def enable_if_available(self) -> None:
        if cuda_available():
            enable_gpu()

    def disable(self) -> None:
        disable_gpu()

    def device_count(self) -> int:
        if not cuda_available():
            return 0
        try:
            return cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError:
            return 0

    def current_device(self) -> int:
        if not is_gpu_enabled():
            return -1
        try:
            return cp.cuda.runtime.getDevice()
        except cp.cuda.runtime.CUDARuntimeError:
            return -1

    def set_device(self, device_id: int) -> None:
        if not cuda_available():
            raise RuntimeError("CUDA is not available")
        cp.cuda.Device(device_id).use()

    def memory_info(self) -> dict:
        if not is_gpu_enabled():
            return {"error": "GPU not enabled"}
        try:
            mempool = cp.get_default_memory_pool()
            return {
                "used": mempool.used_bytes(),
                "total": mempool.total_bytes(),
                "used_gb": mempool.used_bytes() / (1024**3),
                "total_gb": mempool.total_bytes() / (1024**3),
            }
        except Exception as e:
            return {"error": str(e)}


cuda = CudaNamespace()


# ==============================
# Array Module Getter
# ==============================
def get_array_module(x: ArrayType) -> Any:
    """
    Get the appropriate array module (numpy or cupy) for the given array.

    Parameters
    ----------
    x : ArrayType
        Input array (numpy or cupy)

    Returns
    -------
    module
        numpy or cupy module

    Notes
    -----
    どの装置にあるかは配列そのものが決める（`nm.cuda` の設定は、新しく作る配列を
    どこに置くかだけを決める）。そのため GPU を有効にしていなくても、cupy の配列には
    cupy を返す。
    """
    return cp if (cp is not None and isinstance(x, cp.ndarray)) else np


def _index_device_name(k) -> str:
    """インデックスに使う配列がどの装置にあるか（配列でなければ None）"""
    if cp is not None and isinstance(k, cp.ndarray):
        return "GPU"
    if isinstance(k, np.ndarray):
        return "CPU"
    return None


def _check_index_device(x, key) -> None:
    """インデックスに使う配列の装置が、値の装置と食い違っていればエラーにする"""
    keys = key if isinstance(key, tuple) else (key,)
    x_device = _device_name(x)
    for k in keys:
        k_device = _index_device_name(k)
        if k_device is not None and k_device != x_device:
            raise TypeMismatchError(
                "indexing",
                f"{_operand_name(x)} on {x_device}",
                f"index array on {k_device}",
                hint=(
                    "The value and the index must be on the same device. Move one "
                    "of them explicitly, e.g. nm.to_gpu(x) or nm.to_cpu(x)"
                ),
            )


def _device_name(x) -> str:
    """値がどの装置にあるか（"CPU" / "GPU"）"""
    return "GPU" if (cp is not None and isinstance(x._data, cp.ndarray)) else "CPU"


def _check_device_match(name, *xs) -> None:
    """
    入力の装置が食い違っていれば TypeMismatchError を出す

    順伝播の計算が TypeError で失敗したときだけ呼ぶ（成功する経路には判定を入れない）。
    """
    tensors = [x for x in xs if isinstance(x, NumType)]
    if not tensors:
        return
    left = tensors[0]
    left_device = _device_name(left)
    # 実際に装置が食い違っている値を出す（where や concatenate のように 3 つ以上取る
    # 演算で、先頭 2 つがたまたま同じ装置のことがある）
    right = next((t for t in tensors[1:] if _device_name(t) != left_device), None)
    if right is None:
        return
    raise TypeMismatchError(
        name,
        f"{_operand_name(left)} on {left_device}",
        f"{_operand_name(right)} on {_device_name(right)}",
        hint=(
            "Values on CPU and GPU cannot be combined. Move one of them explicitly, "
            "e.g. nm.to_gpu(x) or nm.to_cpu(x)"
        ),
    )


def _array_module_for(data: Any) -> Any:
    """
    data を入れる配列のモジュール（numpy / cupy）を返す

    すでに配列なら、その配列のモジュールに従う（渡された配列は動かさない、という numlib の
    決まりに合わせる。cuda.gpu の中で numpy の値を作っても cupy に化けない）。
    配列でなければ、今の CPU / GPU の設定に従う。
    """
    if isinstance(data, np.ndarray):
        return np
    if cp is not None and isinstance(data, cp.ndarray):
        return cp
    return cp if _cuda_enabled and cp else np


def as_numpy(x: ArrayType) -> np.ndarray:
    """
    Convert array to CPU (numpy array).

    Parameters
    ----------
    x : ArrayType
        Input array

    Returns
    -------
    np.ndarray
        NumPy array on CPU
    """
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def as_cupy(x: ArrayType) -> ArrayType:
    """
    Convert array to GPU (cupy array).

    Parameters
    ----------
    x : ArrayType
        Input array

    Returns
    -------
    ArrayType
        CuPy array on GPU

    Raises
    ------
    RuntimeError
        If CuPy is not installed or GPU is not enabled
    """
    if not cp:
        raise RuntimeError("CuPy is not installed")
    if not _cuda_enabled:
        raise RuntimeError("GPU is not enabled. Call enable_gpu() first")
    if cp is not None:
        return cp.asarray(x)
    return x


# ==============================
# Automatic Differentiation Control
# ==============================


# スレッドセーフな状態管理（必要なら切り替え可能）
_grad_state = threading.local()
_grad_state.enabled = True


def set_grad_enabled(mode: bool):
    """
    Enable or disable gradient computation globally

    Parameters
    ----------
    mode : bool
        True to enable gradient computation, False to disable
    """
    _grad_state.enabled = mode


def get_grad_enabled() -> bool:
    """
    Get current gradient computation state

    Returns
    -------
    bool
        True if gradient computation is enabled, False otherwise
    """
    return getattr(_grad_state, "enabled", True)


class AutogradControl:
    """Single object that acts as both function and context manager"""

    def __init__(self, enable: bool, name: str):
        self.enable = enable
        self.name = name

    def __call__(self):
        """Function call: set gradient state globally"""
        set_grad_enabled(self.enable)

    def __enter__(self):
        """Context manager: temporarily set gradient state"""
        self.prev = get_grad_enabled()
        set_grad_enabled(self.enable)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore previous state even if exception occurred"""
        set_grad_enabled(self.prev)

    def __repr__(self):
        return f"autograd.{self.name}"


class AutogradNamespace:
    """Automatic differentiation control namespace"""

    def __init__(self):
        self._on = AutogradControl(True, "on")
        self._off = AutogradControl(False, "off")

    @property
    def on(self):
        return self._on

    @property
    def off(self):
        return self._off

    def is_enabled(self) -> bool:
        return get_grad_enabled()

    def enable(self) -> None:
        set_grad_enabled(True)

    def disable(self) -> None:
        set_grad_enabled(False)

    def set_enabled(self, mode: bool) -> None:
        set_grad_enabled(mode)


autograd = AutogradNamespace()


# ==============================
# Higher-Order Gradient Helpers
# ==============================
# 逆伝播の閉包 `_backward` は 2 つのモードを持つ:
#
#   _backward()                   1 階: 勾配の式を生の配列で計算する（速い経路）
#   _backward(create_graph=True)  高階: 勾配の式を numlib の演算で計算し、
#                                 勾配の計算そのものも計算グラフに残す
#
# 高階のモードに対応しているかは `_backward` のシグネチャ（引数を 1 つ取るか）で表す
# （_supports_create_graph）。引数を取らない `_backward`（make_op の演算、まだ対応して
# いない演算）は 1 階専用とみなし、create_graph=True の逆伝播の前にエラーにする。


def _backward_built_under_off(create_graph=False):
    """
    番兵: 勾配を追跡する値から autograd.off の中で作ったので、グラフを持たない

    off の中では、この番兵を持つ値から作った値にも引き継ぐ（off の中で合成した関数の
    値も見分けられるように）。on に戻ってから使った値には引き継がない（off の中の値を
    定数として使った、と読む。detach() と同じ意味）。
    """


def _backward_graph_freed(create_graph=False):
    """番兵: backward(retain_graph=False) で計算グラフを解放した"""


def _supports_create_graph(fn):
    """
    _backward が高階のモード（引数 create_graph）に対応していれば True

    約束: 高階に対応した `_backward` は ``def _backward(create_graph=False)`` の形で、
    create_graph=True のときは勾配を numlib の演算で計算して `_accumulate_graph` で
    足し込む。引数を取らない `_backward` は 1 階専用。
    """
    code = getattr(fn, "__code__", None)
    return code is not None and code.co_argcount >= 1


def _as_constant(data, like):
    """生の配列 data を、like と同じ種類（数学の型か Tensor か）・同じ dtype の定数にする"""
    return _create_result(
        data.astype(like.dtype, copy=False), requires_grad=False, math=_is_math(like)
    )


# 高階用の勾配の式で使う定数。Python の float にしておくと、リテラルとして相手の精度で
# 読まれる（NumPy のスカラーのままだと float32 の勾配を float64 に上げてしまう）
_LN2 = float(np.log(2.0))
_LN10 = float(np.log(10.0))


def _spread(s, like):
    """s がスカラーで like が配列なら、s を like と同じ空間の元（s * ones_like(like)）にする"""
    if s.shape == like.shape:
        return s
    return s * ones_like(like)


def _accumulate_graph(x, g):
    """
    高階のモードで、勾配 g（計算グラフ付き）を x.grad に足し込む

    勾配は変数と同じ空間の元なので、g の型を x の型にそろえる（型の変換は恒等写像なので
    グラフはつながったまま）。1 階と違って x.grad._data は書き換えず、新しいノードを作る。
    """
    tx = type(x)
    tg = type(g)
    if tg is not tx and tx in _ARRAY_KINDS and tg in _ARRAY_KINDS:
        g = _construct_or_cast(tx, g)
    x.grad = g if x.grad is None else x.grad + g


def _no_graph_hint(y):
    """requires_grad=False の y を微分しようとしたときの hint（off の中で作った y だけ）"""
    if y._backward is _backward_built_under_off:
        return (
            "This value was computed inside nm.autograd.off, so it has no "
            "computation graph. Compute it inside nm.autograd.on. To treat it "
            "as a constant on purpose, use y.detach()"
        )
    return None


def _check_autograd_for_create_graph():
    if not autograd.is_enabled():
        raise GradientError(
            "create_graph=True is not available inside nm.autograd.off",
            hint=(
                "Higher-order gradients need a computation graph for the "
                "gradient itself. Use create_graph=True inside nm.autograd.on"
            ),
        )


def _topological_order(root):
    """root から _prev をたどったノードを、葉が先になる順に並べる（深いグラフでも反復で行う）"""
    topo = []
    visited = {id(root)}
    stack = [(root, iter(getattr(root, "_prev", ())))]
    while stack:
        v, children = stack[-1]
        for child in children:
            if id(child) not in visited:
                visited.add(id(child))
                stack.append((child, iter(getattr(child, "_prev", ()))))
                break
        else:
            stack.pop()
            topo.append(v)
    return topo


def _check_graph_not_freed(topo):
    for node in topo:
        if node._backward is _backward_graph_freed:
            raise GradientError(
                "The computation graph has already been freed",
                hint=(
                    "backward() frees the graph unless retain_graph=True. Pass "
                    "retain_graph=True to the earlier backward(), or compute the "
                    "value again"
                ),
            )


def _check_create_graph(topo):
    """
    create_graph=True の逆伝播の前に、1 階専用のノードがないかをまとめて調べる

    途中まで x.grad に足し込んでから失敗しないよう、逆伝播を始める前に調べる。
    """
    _check_graph_not_freed(topo)
    for node in topo:
        if node._data.dtype.kind == "c":
            raise GradientError(
                "Higher-order gradient is not supported for complex numbers",
                hint=(
                    "Higher-order gradients of complex numbers are not supported "
                    "yet. Use create_graph=False (first-order gradients of complex "
                    "numbers work as before)"
                ),
            )
        if not node._prev or _supports_create_graph(node._backward):
            continue
        qualname = getattr(node._backward, "__qualname__", "")
        if qualname.startswith("make_op."):
            raise GradientError(
                "Higher-order gradient is not supported for `make_op` operations",
                hint=(
                    "Operations created with nm.make_op only support first-order "
                    "gradients. Use create_graph=False, or express the operation "
                    "with numlib functions"
                ),
            )
        name = qualname.split(".<locals>")[0] or "unknown"
        raise GradientError(
            f"Higher-order gradient is not supported yet for `{name}`",
            hint=(
                "Supported so far: element-wise operations (+, -, *, /, exp, log, "
                "sin, tanh, maximum, ...), pow, sum, mean, reshape, transpose, "
                "broadcast_to, sum_to and type conversions (nm.vector, nm.tensor, "
                "...). Other operations (matmul, dot, indexing, ...) will be "
                "supported later. Use create_graph=False for first-order gradients"
            ),
        )


def _backprop(root, seed, topo, create_graph):
    """
    topo（_topological_order の結果）を逆順にたどって勾配を流す

    create_graph=False は 1 階の速い経路（生の配列）、True は勾配を numlib の演算で
    計算して計算グラフに残す（呼ぶ前に _check_create_graph で調べておくこと）。
    """
    # 中間ノードの勾配は今回の逆伝播のぶんだけにする（retain_graph で
    # 繰り返したとき、前回の値に足し込まれないように）。葉の勾配は累積する
    for node in topo:
        if node._prev:
            node.grad = None

    if create_graph:
        # 勾配の型は _accumulate_graph が足し込むたびに変数の型にそろえる。種もそろえる
        tr, ts = type(root), type(seed)
        if ts is not tr and tr in _ARRAY_KINDS and ts in _ARRAY_KINDS:
            seed = _construct_or_cast(tr, seed)
        root.grad = seed
        for node in reversed(topo):
            # 葉の _backward は何もしないので呼ばない（引数を取らないものもある）
            if node._prev:
                node._backward(True)
        return

    root.grad = seed
    for node in reversed(topo):
        node._backward()

    # 勾配は変数と同じ空間の元なので、数学の型の変数には同じ型の勾配を持たせる
    for node in topo:
        g = node.grad
        if (
            g is not None
            and isinstance(node, _MATRIX_TYPES)
            and type(g) is not type(node)
            and g.shape == node.shape
        ):
            node.grad = _create_result(g._data, requires_grad=g.requires_grad, math=True)


# ==============================
# Operation Factory Functions
# ==============================


def _make_binary_op(
    forward_fn,
    grad_x_fn,
    grad_y_fn,
    save_data=True,
    kind="map",
    name="operation",
    *,
    grad_x_graph,
    grad_y_graph,
):
    """
    二項演算のファクトリ関数

    ボイラープレートを削減し、新しい演算の追加を容易にする。

    Parameters
    ----------
    forward_fn : callable(x_data, y_data) -> ndarray
        順伝播: result = forward_fn(x._data, y._data)
    grad_x_fn : callable(grad, x_data, y_data, result_data) -> ndarray
        x の勾配計算（1 階用。生の配列）
    grad_y_fn : callable(grad, x_data, y_data, result_data) -> ndarray
        y の勾配計算（1 階用。生の配列）
    save_data : bool, optional
        x._data, y._data を保存するか (デフォルト: True)
    kind : {"add", "mul", "div", "map"}, optional
        数学的に定義される組み合わせの種類（_check_elementwise を参照）
    name : str, optional
        エラーメッセージに出す演算の名前
    grad_x_graph : callable(grad, x, y, result) -> NumType
        x の勾配計算（高階用。引数はすべて NumType のノードで、numlib の演算で書く）。
        create_graph=True の逆伝播で使い、勾配の計算も計算グラフに残る
    grad_y_graph : callable(grad, x, y, result) -> NumType
        y の勾配計算（高階用）

    Returns
    -------
    callable
        二項演算関数

    Examples
    --------
    >>> add = _make_binary_op(
    ...     forward_fn=lambda x, y: x + y,
    ...     grad_x_fn=lambda g, x, y, r: g,
    ...     grad_y_fn=lambda g, x, y, r: g,
    ...     grad_x_graph=lambda g, x, y, r: g,
    ...     grad_y_graph=lambda g, x, y, r: g,
    ... )
    """

    def binary_op(x, y):
        # 型変換（リテラルは、形がまったく同じなら相手と同じ空間の元として読む）
        x, y = _convert_operands(x, y)

        # 数学的に定義されない組み合わせ（暗黙のブロードキャストなど）はエラー
        _check_elementwise(kind, name, x, y)

        # 計算
        try:
            result_data = forward_fn(x._data, y._data)
        except TypeError:
            # 装置（CPU / GPU）が食い違っているなら、分かるエラーにする
            _check_device_match(name, x, y)
            raise
        math = _is_math(x, y)

        # 勾配追跡の早期判定
        x_req = x.requires_grad
        y_req = y.requires_grad

        if not (autograd.is_enabled() and (x_req or y_req)):
            result = _create_result(result_data, math=math)
            result.requires_grad = False
            if x_req or y_req or (
                (
                    x._backward is _backward_built_under_off
                    or y._backward is _backward_built_under_off
                )
                and not autograd.is_enabled()
            ):
                result._backward = _backward_built_under_off
            return result

        # 勾配が必要な場合
        result = _create_result(result_data, math=math)
        result.requires_grad = True

        # _prev を構築
        if x_req and y_req:
            result._prev = (x, y)
        elif x_req:
            result._prev = (x,)
        else:
            result._prev = (y,)

        # クロージャ変数の保存
        if save_data:
            x_data = x._data
            y_data = y._data
        x_shape = x.shape
        y_shape = y.shape
        result_data_saved = result_data if not save_data else None

        def _backward(create_graph=False):
            if result.grad is None:
                return

            if create_graph:
                g = result.grad
                if x_req:
                    _accumulate_graph(
                        x, sum_to(grad_x_graph(g, x, y, result), x_shape)
                    )
                if y_req:
                    _accumulate_graph(
                        y, sum_to(grad_y_graph(g, x, y, result), y_shape)
                    )
                return

            grad_data = result.grad._data

            if x_req:
                grad_x_raw = grad_x_fn(
                    grad_data,
                    x_data if save_data else x._data,
                    y_data if save_data else y._data,
                    result_data_saved
                    if result_data_saved is not None
                    else result._data,
                )
                grad_x = sum_to(_create_result(grad_x_raw), x_shape)
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad._data = x.grad._data + grad_x._data

            if y_req:
                grad_y_raw = grad_y_fn(
                    grad_data,
                    x_data if save_data else x._data,
                    y_data if save_data else y._data,
                    result_data_saved
                    if result_data_saved is not None
                    else result._data,
                )
                grad_y = sum_to(_create_result(grad_y_raw), y_shape)
                if y.grad is None:
                    y.grad = grad_y
                else:
                    y.grad._data = y.grad._data + grad_y._data

        result._backward = _backward
        return result

    return binary_op


def _make_unary_op(forward_fn, grad_fn, save_input=False, save_output=True, *, grad_graph):
    """
    単項演算のファクトリ関数

    ボイラープレートを削減し、新しい演算の追加を容易にする。

    Parameters
    ----------
    forward_fn : callable(xp, x_data) -> ndarray
        順伝播: result = forward_fn(xp, x._data)
    grad_fn : callable(grad, x_data, result_data, xp) -> ndarray
        勾配計算（1 階用。生の配列）: grad_x = grad_fn(grad, x._data, result._data, xp)
    save_input : bool, optional
        x._data を保存するか (デフォルト: False)
    save_output : bool, optional
        result._data を保存するか (デフォルト: True)
    grad_graph : callable(grad, x, result) -> NumType
        勾配計算（高階用。引数はすべて NumType のノードで、numlib の演算で書く）。
        create_graph=True の逆伝播で使い、勾配の計算も計算グラフに残る

    Returns
    -------
    callable
        単項演算関数

    Examples
    --------
    >>> exp = _make_unary_op(
    ...     forward_fn=lambda xp, x: xp.exp(x),
    ...     grad_fn=lambda g, x, r, xp: g * r,  # d exp(x)/dx = exp(x) = r
    ...     grad_graph=lambda g, x, r: g * r,
    ...     save_input=False,
    ...     save_output=True
    ... )
    """

    def unary_op(x):
        if not isinstance(x, NumType):
            x = _auto_convert(x, requires_grad=False)

        xp = get_array_module(x._data)
        result_data = forward_fn(xp, x._data)
        result = _create_result(result_data, math=_is_math(x))

        # 早期リターン
        if not (autograd.is_enabled() and x.requires_grad):
            result.requires_grad = False
            if x.requires_grad or (
                x._backward is _backward_built_under_off and not autograd.is_enabled()
            ):
                result._backward = _backward_built_under_off
            return result

        result.requires_grad = True
        result._prev = (x,)

        # 必要な値だけ保存
        x_data = x._data if save_input else None
        result_data_saved = result._data if save_output else None

        def _backward(create_graph=False):
            if result.grad is None:
                return

            if create_graph:
                _accumulate_graph(x, grad_graph(result.grad, x, result))
                return

            grad_data = result.grad._data

            # 保存してない場合は再計算
            _x_data = x_data if save_input else x._data
            _result_data = result_data_saved if save_output else result._data

            grad_x_data = grad_fn(grad_data, _x_data, _result_data, xp)
            grad_x = _create_result(grad_x_data)

            if x.grad is None:
                x.grad = grad_x
            else:
                x.grad._data = x.grad._data + grad_x._data

        result._backward = _backward
        return result

    return unary_op


def make_op(forward, backward):
    """
    順伝播と逆伝播から、自動微分できる演算を作る（公開 API）

    演算を作る側は、生の配列（numpy / cupy）で順伝播と勾配の式だけを書く。
    入力の変換、勾配を追跡するかの判定、計算グラフの構築、勾配の形のチェック、
    勾配の累積は、ここで1か所にまとめて行う。

    Parameters
    ----------
    forward : callable(*arrays, **params) -> (result_array, ctx)
        順伝播。位置引数は入力の配列（入力が None ならそのまま None）。
        キーワード引数は、微分しないパラメータ（stride や eps など）。
        結果の配列と、逆伝播で使う値 ctx（何でもよい）のタプルを返す。
    backward : callable(ctx, grad, needs_grad) -> tuple
        逆伝播。grad は出力についての勾配の配列、needs_grad は入力ごとに
        勾配が必要かどうかの bool のタプル。入力ごとの勾配の配列を、入力と同じ
        順番のタプルで返す。勾配が要らない入力には None を返してよい。

    Returns
    -------
    callable(*inputs, **params) -> NumType
        位置引数に入力（NumType、配列、None）、キーワード引数にパラメータを取る演算。
        NumType でない入力は定数として扱う（勾配を追跡しない）。

    Examples
    --------
    >>> def _forward(x):
    ...     y = x * x
    ...     return y, x
    >>> def _backward(x, grad, needs_grad):
    ...     return (grad * 2 * x,)
    >>> square = nm.make_op(_forward, _backward)
    >>> x = nm.tensor([1.0, 2.0], requires_grad=True)
    >>> nm.sum(square(x)).backward()
    >>> x.grad  # [2.0, 4.0]
    """

    def op(*inputs, **params):
        xs = tuple(
            None
            if a is None
            else (
                a if isinstance(a, NumType) else _auto_convert(a, requires_grad=False)
            )
            for a in inputs
        )
        result_data, ctx = forward(
            *(None if a is None else a._data for a in xs), **params
        )
        result = _create_result(
            result_data, math=_is_math(*(a for a in xs if a is not None))
        )

        needs_grad = tuple(a is not None and a.requires_grad for a in xs)
        if not (autograd.is_enabled() and builtins.any(needs_grad)):
            result.requires_grad = False
            if builtins.any(needs_grad) or (
                not autograd.is_enabled()
                and builtins.any(a is not None and a._backward is _backward_built_under_off for a in xs)
            ):
                result._backward = _backward_built_under_off
            return result

        result.requires_grad = True
        result._prev = tuple(a for a, n in zip(xs, needs_grad) if n)

        def _backward():
            if result.grad is None:
                return
            grads = backward(ctx, result.grad._data, needs_grad)
            if len(grads) != len(xs):
                raise GradientError(
                    f"backward returned {len(grads)} gradients for {len(xs)} inputs"
                )
            for a, need, g in zip(xs, needs_grad, grads):
                if not need or g is None:
                    continue
                if g.shape != a.shape:
                    raise GradientError(
                        f"gradient shape {g.shape} does not match input shape {a.shape}"
                    )
                # 勾配の配列は read-only の view のこともあるので、+= ではなく新しい配列を作る
                if a.grad is None:
                    a.grad = _create_result(g)
                else:
                    a.grad._data = a.grad._data + g

        result._backward = _backward
        return result

    return op


# ==============================
# Factory-Created Operations
# ==============================
# Operations created using _make_binary_op and _make_unary_op
# Organized in logical order for better maintainability


# ------------------------------
# 1. Basic Binary Operations
# ------------------------------

add = _make_binary_op(
    forward_fn=lambda x, y: x + y,
    grad_x_fn=lambda g, x, y, r: g,
    grad_y_fn=lambda g, x, y, r: g,
    grad_x_graph=lambda g, x, y, r: g,
    grad_y_graph=lambda g, x, y, r: g,
    kind="add",
    name="addition",
)


sub = _make_binary_op(
    forward_fn=lambda x, y: x - y,
    grad_x_fn=lambda g, x, y, r: g,
    grad_y_fn=lambda g, x, y, r: -g,
    grad_x_graph=lambda g, x, y, r: g,
    grad_y_graph=lambda g, x, y, r: -g,
    kind="add",
    name="subtraction",
)


mul = _make_binary_op(
    forward_fn=lambda x, y: x * y,
    grad_x_fn=lambda g, x, y, r: g * y,
    grad_y_fn=lambda g, x, y, r: g * x,
    grad_x_graph=lambda g, x, y, r: g * y,
    grad_y_graph=lambda g, x, y, r: g * x,
    kind="mul",
    name="multiplication",
)


div = _make_binary_op(
    forward_fn=lambda x, y: x / y,
    grad_x_fn=lambda g, x, y, r: g / y,
    grad_y_fn=lambda g, x, y, r: -g * x / (y * y),
    grad_x_graph=lambda g, x, y, r: g / y,
    grad_y_graph=lambda g, x, y, r: -(g * x) / (y * y),
    kind="div",
    name="division",
)


# ------------------------------
# 2. Basic Unary Operations
# ------------------------------

neg = _make_unary_op(
    forward_fn=lambda xp, x: -x,
    grad_fn=lambda g, x, r, xp: -g,
    grad_graph=lambda g, x, r: -g,
    save_input=False,
    save_output=False,
)

# |x| は x = 0 で微分できない。ほとんど至る所での微分 sign(x) を使い（x = 0 では 0）、
# sign(x) の微分は 0（定数）として扱う
absolute = _make_unary_op(
    forward_fn=lambda xp, x: xp.absolute(x),
    grad_fn=lambda g, x, r, xp: g * xp.sign(x),
    grad_graph=lambda g, x, r: g * _as_constant(get_array_module(x._data).sign(x._data), x),
    save_input=True,
    save_output=False,
)

# Alias for backward compatibility
abs = absolute

sqrt = _make_unary_op(
    forward_fn=lambda xp, x: xp.sqrt(x),
    grad_fn=lambda g, x, r, xp: g / (2 * r),  # d sqrt(x)/dx = 1/(2*sqrt(x))
    grad_graph=lambda g, x, r: g / (2 * r),
    save_input=False,
    save_output=True,
)


# ------------------------------
# 3. Exponential and Logarithmic Functions
# ------------------------------

exp = _make_unary_op(
    forward_fn=lambda xp, x: xp.exp(x),
    grad_fn=lambda g, x, r, xp: g * r,  # d exp(x)/dx = exp(x) = r
    grad_graph=lambda g, x, r: g * r,
    save_input=False,
    save_output=True,
)

log = _make_unary_op(
    forward_fn=lambda xp, x: xp.log(x),
    grad_fn=lambda g, x, r, xp: g / x,  # d log(x)/dx = 1/x
    grad_graph=lambda g, x, r: g / x,
    save_input=True,
    save_output=False,
)

expm1 = _make_unary_op(
    forward_fn=lambda xp, x: xp.expm1(x),
    grad_fn=lambda g, x, r, xp: g * xp.exp(x),  # d (exp(x)-1)/dx = exp(x)
    grad_graph=lambda g, x, r: g * exp(x),
    save_input=True,
    save_output=False,
)

log1p = _make_unary_op(
    forward_fn=lambda xp, x: xp.log1p(x),
    grad_fn=lambda g, x, r, xp: g / (1 + x),  # d log(1+x)/dx = 1/(1+x)
    grad_graph=lambda g, x, r: g / (ones_like(x) + x),
    save_input=True,
    save_output=False,
)

log2 = _make_unary_op(
    forward_fn=lambda xp, x: xp.log2(x),
    grad_fn=lambda g, x, r, xp: g / (x * xp.log(2)),  # d log2(x)/dx = 1/(x*ln(2))
    grad_graph=lambda g, x, r: g / (x * _LN2),
    save_input=True,
    save_output=False,
)

log10 = _make_unary_op(
    forward_fn=lambda xp, x: xp.log10(x),
    grad_fn=lambda g, x, r, xp: g / (x * xp.log(10)),  # d log10(x)/dx = 1/(x*ln(10))
    grad_graph=lambda g, x, r: g / (x * _LN10),
    save_input=True,
    save_output=False,
)


# ------------------------------
# 4. Trigonometric Functions
# ------------------------------

sin = _make_unary_op(
    forward_fn=lambda xp, x: xp.sin(x),
    grad_fn=lambda g, x, r, xp: g * xp.cos(x),  # d sin(x)/dx = cos(x)
    grad_graph=lambda g, x, r: g * cos(x),
    save_input=True,
    save_output=False,
)

cos = _make_unary_op(
    forward_fn=lambda xp, x: xp.cos(x),
    grad_fn=lambda g, x, r, xp: -g * xp.sin(x),  # d cos(x)/dx = -sin(x)
    grad_graph=lambda g, x, r: -(g * sin(x)),
    save_input=True,
    save_output=False,
)

tan = _make_unary_op(
    forward_fn=lambda xp, x: xp.tan(x),
    grad_fn=lambda g, x, r, xp: g / (xp.cos(x) ** 2),  # d tan(x)/dx = 1/cos²(x)
    grad_graph=lambda g, x, r: g / cos(x) ** 2,
    save_input=True,
    save_output=False,
)

arcsin = _make_unary_op(
    forward_fn=lambda xp, x: xp.arcsin(x),
    grad_fn=lambda g, x, r, xp: g / xp.sqrt(1 - x**2),  # d arcsin(x)/dx = 1/sqrt(1-x²)
    grad_graph=lambda g, x, r: g / sqrt(ones_like(x) - x**2),
    save_input=True,
    save_output=False,
)

arccos = _make_unary_op(
    forward_fn=lambda xp, x: xp.arccos(x),
    grad_fn=lambda g, x, r, xp: (
        -g / xp.sqrt(1 - x**2)
    ),  # d arccos(x)/dx = -1/sqrt(1-x²)
    grad_graph=lambda g, x, r: -g / sqrt(ones_like(x) - x**2),
    save_input=True,
    save_output=False,
)

arctan = _make_unary_op(
    forward_fn=lambda xp, x: xp.arctan(x),
    grad_fn=lambda g, x, r, xp: g / (1 + x**2),  # d arctan(x)/dx = 1/(1+x²)
    grad_graph=lambda g, x, r: g / (ones_like(x) + x**2),
    save_input=True,
    save_output=False,
)


# ------------------------------
# 5. Hyperbolic Functions
# ------------------------------

sinh = _make_unary_op(
    forward_fn=lambda xp, x: xp.sinh(x),
    grad_fn=lambda g, x, r, xp: g * xp.cosh(x),  # d sinh(x)/dx = cosh(x)
    grad_graph=lambda g, x, r: g * cosh(x),
    save_input=True,
    save_output=False,
)

cosh = _make_unary_op(
    forward_fn=lambda xp, x: xp.cosh(x),
    grad_fn=lambda g, x, r, xp: g * xp.sinh(x),  # d cosh(x)/dx = sinh(x)
    grad_graph=lambda g, x, r: g * sinh(x),
    save_input=True,
    save_output=False,
)

tanh = _make_unary_op(
    forward_fn=lambda xp, x: xp.tanh(x),
    grad_fn=lambda g, x, r, xp: g * (1 - r * r),  # d tanh(x)/dx = 1 - tanh²(x)
    grad_graph=lambda g, x, r: g * (ones_like(r) - r * r),
    save_input=False,
    save_output=True,
)

arcsinh = _make_unary_op(
    forward_fn=lambda xp, x: xp.arcsinh(x),
    grad_fn=lambda g, x, r, xp: g / xp.sqrt(x**2 + 1),  # d arcsinh(x)/dx = 1/sqrt(x²+1)
    grad_graph=lambda g, x, r: g / sqrt(x**2 + ones_like(x)),
    save_input=True,
    save_output=False,
)

arccosh = _make_unary_op(
    forward_fn=lambda xp, x: xp.arccosh(x),
    grad_fn=lambda g, x, r, xp: g / xp.sqrt(x**2 - 1),  # d arccosh(x)/dx = 1/sqrt(x²-1)
    grad_graph=lambda g, x, r: g / sqrt(x**2 - ones_like(x)),
    save_input=True,
    save_output=False,
)

arctanh = _make_unary_op(
    forward_fn=lambda xp, x: xp.arctanh(x),
    grad_fn=lambda g, x, r, xp: g / (1 - x**2),  # d arctanh(x)/dx = 1/(1-x²)
    grad_graph=lambda g, x, r: g / (ones_like(x) - x**2),
    save_input=True,
    save_output=False,
)


# ------------------------------
# 6. Comparison and Special Functions
# ------------------------------

# maximum / minimum は a = b で微分できない。ほとんど至る所での微分を使い、a = b では
# 両方に半分ずつ（0.5）流す。マスクの微分は 0（定数）として扱う（1 階も高階も同じ決まり）


def _maximum_mask(a, b, r):
    """max(a, b) の a についての微分: a > b で 1、a == b で 0.5、a < b で 0（形は r と同じ）"""
    mask = get_array_module(r).zeros_like(r)
    mask[a > b] = 1.0
    mask[a == b] = 0.5
    return mask


def _minimum_mask(a, b, r):
    """min(a, b) の a についての微分: a < b で 1、a == b で 0.5、a > b で 0（形は r と同じ）"""
    mask = get_array_module(r).zeros_like(r)
    mask[a < b] = 1.0
    mask[a == b] = 0.5
    return mask


maximum = _make_binary_op(
    forward_fn=lambda x, y: get_array_module(x).maximum(x, y),
    grad_x_fn=lambda g, x, y, r: g * _maximum_mask(x, y, r),
    grad_y_fn=lambda g, x, y, r: g * _maximum_mask(y, x, r),
    grad_x_graph=lambda g, x, y, r: g
    * _as_constant(_maximum_mask(x._data, y._data, r._data), g),
    grad_y_graph=lambda g, x, y, r: g
    * _as_constant(_maximum_mask(y._data, x._data, r._data), g),
    save_data=True,
    kind="map",
    name="maximum",
)


minimum = _make_binary_op(
    forward_fn=lambda x, y: get_array_module(x).minimum(x, y),
    grad_x_fn=lambda g, x, y, r: g * _minimum_mask(x, y, r),
    grad_y_fn=lambda g, x, y, r: g * _minimum_mask(y, x, r),
    grad_x_graph=lambda g, x, y, r: g
    * _as_constant(_minimum_mask(x._data, y._data, r._data), g),
    grad_y_graph=lambda g, x, y, r: g
    * _as_constant(_minimum_mask(y._data, x._data, r._data), g),
    save_data=True,
    kind="map",
    name="minimum",
)


# Special math functions
square = _make_unary_op(
    forward_fn=lambda xp, x: x**2,
    grad_fn=lambda g, x, r, xp: g * 2 * x,  # d x²/dx = 2x
    grad_graph=lambda g, x, r: g * (2 * x),
    save_input=True,
    save_output=False,
)

reciprocal = _make_unary_op(
    forward_fn=lambda xp, x: 1.0 / x,
    grad_fn=lambda g, x, r, xp: -g / (x**2),  # d (1/x)/dx = -1/x²
    grad_graph=lambda g, x, r: -g / x**2,
    save_input=True,
    save_output=False,
)


# Helper functions for atan2
def _atan2_grad_y(g, y, x, r):
    """Gradient for atan2 w.r.t. y: d atan2(y,x)/dy = x/(x²+y²)"""
    return g * x / (x**2 + y**2)


def _atan2_grad_x(g, y, x, r):
    """Gradient for atan2 w.r.t. x: d atan2(y,x)/dx = -y/(x²+y²)"""
    return g * (-y) / (x**2 + y**2)


def _atan2_denominator(y, x, r):
    """高階用: x² + y²。片方がスカラーなら ones_like で相手の空間の元にしてから足す"""
    return _spread(x**2, r) + _spread(y**2, r)


atan2 = _make_binary_op(
    forward_fn=lambda y, x: get_array_module(y).arctan2(y, x),
    grad_x_fn=_atan2_grad_y,  # Note: x in _make_binary_op corresponds to first arg (y)
    grad_y_fn=_atan2_grad_x,  # Note: y in _make_binary_op corresponds to second arg (x)
    grad_x_graph=lambda g, y, x, r: g * x / _atan2_denominator(y, x, r),
    grad_y_graph=lambda g, y, x, r: -(g * y) / _atan2_denominator(y, x, r),
    save_data=True,
    kind="map",
    name="atan2",
)


# ------------------------------
# 7. NumPy Compatibility Aliases
# ------------------------------

# Aliases for inverse trigonometric functions (NumPy compatibility)
asin = arcsin
acos = arccos
atan = arctan

# Aliases for inverse hyperbolic functions (NumPy compatibility)
asinh = arcsinh
acosh = arccosh
atanh = arctanh


# ==============================
# Type Detection Utilities (Optimized)
# ==============================
_TYPE_CACHE = None


def _init_type_cache():
    """型判定用のキャッシュを初期化（一度だけ実行）- Optimized"""
    global _TYPE_CACHE
    if _TYPE_CACHE is not None:
        return

    _TYPE_CACHE = {
        "int_types": (
            int,
            np.integer,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
        "float_types": (float, np.floating, np.float16, np.float32, np.float64),
        "complex_types": (complex, np.complexfloating, np.complex64, np.complex128),
        "bool_types": (bool, np.bool_),
    }

    # CuPy対応
    if cp is not None:
        _TYPE_CACHE["int_types"] += (
            cp.integer,
            cp.int8,
            cp.int16,
            cp.int32,
            cp.int64,
            cp.uint8,
            cp.uint16,
            cp.uint32,
            cp.uint64,
        )
        _TYPE_CACHE["float_types"] += (cp.floating, cp.float16, cp.float32, cp.float64)
        _TYPE_CACHE["complex_types"] += (
            cp.complexfloating,
            cp.complex64,
            cp.complex128,
        )
        _TYPE_CACHE["bool_types"] += (cp.bool_,)


def _is_np_bool(value: Any) -> bool:
    """Check if value is numpy/cupy bool - Optimized"""
    _init_type_cache()
    if isinstance(value, _TYPE_CACHE["bool_types"]):
        return True
    if not isinstance(value, (np.ndarray, (cp.ndarray if cp else type(None)))):
        return False
    return value.ndim == 0 and value.dtype.kind == "b"


def _is_np_int(value: Any) -> bool:
    """Check if value is numpy/cupy integer - Optimized"""
    _init_type_cache()
    if isinstance(value, _TYPE_CACHE["int_types"]):
        return True
    if not isinstance(value, (np.ndarray, (cp.ndarray if cp else type(None)))):
        return False
    return value.ndim == 0 and value.dtype.kind in ("i", "u")


def _is_np_float(value: Any) -> bool:
    """Check if value is numpy/cupy float - Optimized"""
    _init_type_cache()
    if isinstance(value, _TYPE_CACHE["float_types"]):
        return True
    if not isinstance(value, (np.ndarray, (cp.ndarray if cp else type(None)))):
        return False
    return value.ndim == 0 and value.dtype.kind == "f"


def _is_np_complex(value: Any) -> bool:
    """Check if value is numpy/cupy complex - Optimized"""
    _init_type_cache()
    if isinstance(value, _TYPE_CACHE["complex_types"]):
        return True
    if not isinstance(value, (np.ndarray, (cp.ndarray if cp else type(None)))):
        return False
    return value.ndim == 0 and value.dtype.kind == "c"


# ==============================
# Exception Classes
# ==============================


class CastError(TypeError):
    """Exception raised when type casting is not allowed"""

    def __init__(self, from_type, to_type, hint=None):
        self.message = f"Cannot cast `{from_type}` to `{to_type}`"
        if hint:
            self.message += f"\n\n  Hint: {hint}\n"
        super().__init__(self.message)


def _check_scalar_astype(x, dtype):
    """Raise CastError when a scalar type is cast to another category of dtype"""
    categories = {Boolean: "b", Integer: "iu", Real: "f", Complex: "c"}
    allowed = categories.get(type(x))
    dtype = np.dtype(dtype)
    if allowed is None or dtype.kind in allowed:
        return
    bits = dtype.itemsize * 8
    factory = {
        "b": "boolean",
        "i": f"int{bits}",
        "u": f"uint{bits}",
        "f": f"real{bits}",
        "c": f"cmplx{bits}",
    }.get(dtype.kind)
    hint = (
        f"{type(x).__name__}.astype() only changes precision within the same kind"
        f" of number."
    )
    if factory and isinstance(x, Complex):
        # A complex number has no single real value; make the user pick one
        hint += (
            " A complex number has no single real value: choose x.real, x.imag"
            f" or nm.abs(x), e.g. nm.{factory}(x.real)."
        )
    elif factory:
        hint += f" To convert, use nm.{factory}(x)."
    raise CastError(type(x).__name__, dtype.name, hint=hint)


class NumlibError(Exception):
    """Base class for all numlib errors"""

    pass


class DimensionError(NumlibError, ValueError):
    """Dimension mismatch in operations (also a ValueError for backward compatibility)"""

    def __init__(
        self,
        operation,
        left_shape,
        right_shape,
        left_type=None,
        right_type=None,
        hint=None,
    ):
        self.operation = operation
        self.left_shape = left_shape
        self.right_shape = right_shape

        msg = f"\nCannot perform {operation} with incompatible shapes\n"
        msg += f"  Left:  {self._format_shape(left_shape, left_type)}\n"
        msg += f"  Right: {self._format_shape(right_shape, right_type)}\n"

        if operation == "matmul":
            msg += f"\n  For matrix multiplication A @ B:\n"
            msg += f"    Inner dimensions must match\n"
            msg += f"    Got: {left_shape[-1]} != {right_shape[0]}\n"

        if hint:
            msg += f"\n  Hint: {hint}\n"

        super().__init__(msg)

    def _format_shape(self, shape, type_name=None):
        if type_name:
            type_str = type_name
        elif len(shape) == 0:
            type_str = "Scalar"
        elif len(shape) == 1:
            type_str = f"Vector[{shape[0]}]"
        elif len(shape) == 2:
            type_str = f"Matrix[{shape[0]}x{shape[1]}]"
        else:
            type_str = f"Tensor{list(shape)}"
        return type_str


class GradientError(NumlibError, RuntimeError):
    """Gradient computation error (also a RuntimeError for backward compatibility)"""

    def __init__(self, message, shape=None, suggestions=None, hint=None):
        msg = f"\n{message}\n"

        if shape is not None:
            msg += f"  Output shape: {shape}\n"

        if suggestions:
            msg += "\n  Possible solutions:\n"
            for i, suggestion in enumerate(suggestions, 1):
                msg += f"    {i}. {suggestion}\n"

        if hint:
            msg += f"\n  Hint: {hint}\n"

        super().__init__(msg)


class TypeMismatchError(NumlibError, ValueError):
    """Type mismatch in operations"""

    def __init__(self, operation, left_type, right_type, hint=None):
        msg = f"\nCannot perform {operation} between incompatible types\n"
        msg += f"  Left:  {left_type}\n"
        msg += f"  Right: {right_type}\n"

        if hint:
            msg += f"\n  Hint: {hint}\n"

        super().__init__(msg)


# ==============================
# Base NumType Class (Optimized)
# ==============================


def ones_like(x):
    """
    Create a tensor of ones with the same shape and type as input

    Parameters
    ----------
    x : Tensor
        Input tensor

    Returns
    -------
    Tensor
        Tensor filled with ones, matching the shape and type of x
    """
    xp = get_array_module(x._data)
    data = xp.ones_like(x._data)
    # 定数なので勾配は追跡しない。スカラーは kind を中身の dtype に合わせる
    if isinstance(x, Scalar):
        return _auto_scalar(data, requires_grad=False)
    return type(x)(data, requires_grad=False)


def zeros_like(x):
    """
    Create a tensor of zeros with the same shape and type as input

    Parameters
    ----------
    x : Tensor
        Input tensor

    Returns
    -------
    Tensor
        Tensor filled with zeros, matching the shape and type of x
    """
    xp = get_array_module(x._data)
    data = xp.zeros_like(x._data)
    # 定数なので勾配は追跡しない。スカラーは kind を中身の dtype に合わせる
    if isinstance(x, Scalar):
        return _auto_scalar(data, requires_grad=False)
    return type(x)(data, requires_grad=False)


class NumType:
    """
    Base class for all numerical types with automatic differentiation support.
    """

    __slots__ = ("_data", "name", "grad", "requires_grad", "_prev", "_backward")

    def __init__(self, data: Any, requires_grad: bool = True, name: str = None):
        self._data = self._convert_data(data)
        self.name = name
        # autodiff
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = set()
        self._backward = lambda: None

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        """
        Compute gradients via backpropagation.

        Parameters
        ----------
        gradient : NumType, optional
            Gradient tensor. If None, assumes scalar output with gradient=1.
        retain_graph : bool, optional
            If False (default), the computation graph is freed after backward.
            If True, the graph is retained for multiple backward calls.
        create_graph : bool, optional
            If True, the gradients are computed with numlib operations and
            ``x.grad`` carries a computation graph, so it can be
            differentiated again (higher-order derivatives). Implies
            ``retain_graph=True``. If False (default), the fast first-order
            path (raw arrays) is used.

        Raises
        ------
        GradientError
            If ``self`` does not require gradients, if ``create_graph=True``
            is used inside ``nm.autograd.off``, or if the graph contains an
            operation that supports only first-order gradients (``make_op``
            operations, complex numbers, operations not yet supported).

        Examples
        --------
        >>> x = nm.real(2.0, requires_grad=True)
        >>> y = x ** 3
        >>> y.backward(create_graph=True)   # x.grad = 3x² = 12 (with a graph)
        >>> x.grad.backward()               # x.grad += d(3x²)/dx = 6x = 12
        """
        if not self.requires_grad:
            raise GradientError(
                f"{type(self).__name__} does not require gradients",
                hint=_no_graph_hint(self),
            )

        if create_graph:
            _check_autograd_for_create_graph()
            # 高階微分では勾配の計算が元のグラフを参照するので、グラフは解放しない
            retain_graph = True

        topo = _topological_order(self)
        if create_graph:
            _check_create_graph(topo)

        # 勾配の初期化
        if gradient is None:
            if self.shape != ():
                raise GradientError(
                    "Cannot compute gradient for non-scalar output",
                    shape=self.shape,
                    suggestions=[
                        "Sum the output: output.sum().backward()",
                        "Take mean: output.mean().backward()",
                        f"Provide gradient explicitly: output.backward(gradient=ones({self.shape}))",
                    ],
                )
            gradient = ones_like(self)

        _backprop(self, gradient, topo, create_graph)

        # retain_graph=Falseの場合、計算グラフを解放する。解放したノードには番兵を
        # 置き、あとで nm.grad などが黙って 0 を返さないようにする（葉には置かない）
        if not retain_graph:
            for node in topo:
                if node._prev:
                    node._prev = set()
                    node._backward = _backward_graph_freed

    @property
    def g(self):
        return self.grad

    def zero_grad(self):
        """勾配をリセット"""
        self.grad = None

    def cleargrad(self):
        return self.zero_grad()

    def _like(self, data, requires_grad=False):
        """
        自分と同じ型・同じ属性（kind / signed / name）で、中身だけ data にした値を作る

        サブクラスのスロット（Integer.kind / signed、Real.kind ...）はそのまま写し、
        __init__ を通さないので dtype が derive され直さない。
        """
        cls = type(self)
        out = object.__new__(cls)
        for klass in cls.__mro__[:-1]:
            if klass is NumType:
                break
            for slot in klass.__dict__.get("__slots__", ()):
                if hasattr(self, slot):
                    setattr(out, slot, getattr(self, slot))
        NumType.__init__(out, data, requires_grad=requires_grad, name=self.name)
        return out

    def detach(self):
        """
        Return a new object detached from the computational graph.

        The result has the same type (``Vector``, ``Matrix``, ``Real`` ...),
        the same ``kind`` / ``signed`` and the same value as ``self``, but
        ``requires_grad=False`` and no graph history, so ``backward()`` stops
        there. Only this path is cut; other paths from ``self`` still carry
        gradients.

        Returns
        -------
        NumType
            Object of ``type(self)`` sharing the underlying array with ``self``.

        Notes
        -----
        The array is shared, not copied. In-place operations (``+=``,
        ``__setitem__`` ...) on either object are visible through the other.
        Use ``x.detach().copy()`` for an independent copy.

        Examples
        --------
        >>> y = x * x.detach()             # dy/dx = x, not 2x
        >>> y = x + (round_op(x) - x).detach()  # straight-through estimator
        """
        # Subclass slots (Integer.kind/signed, Real.kind, ...) are copied as is,
        # bypassing __init__ so that the dtype is never re-derived.
        return self._like(self._data)

    def _convert_data(self, data: Any) -> ArrayType:
        """Convert input to appropriate array type - Optimized"""
        if isinstance(data, NumType):
            return data._data

        xp = cp if _cuda_enabled and cp else np

        if isinstance(data, (np.ndarray, (cp.ndarray if cp else type(None)))):
            return data

        return xp.asarray(data)

    # ==============================
    # Properties
    # ==============================

    @property
    def data(self) -> ArrayType:
        """Get underlying array data"""
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array"""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Total number of elements"""
        return self._data.size

    @property
    def dtype(self):
        """Data type of the array"""
        return self._data.dtype

    # ==============================
    # GPU/CPU Transfer
    # ==============================

    def as_numpy(self) -> "NumType":
        """Transfer data to CPU"""
        self._data = as_numpy(self._data)
        return self

    def as_cupy(self) -> "NumType":
        """Transfer data to GPU"""
        self._data = as_cupy(self._data)
        return self

    # ==============================
    # NumPy Array Protocol (for ecosystem compatibility)
    # ==============================

    def __array__(self, dtype=None):
        """
        NumPy array protocol for ecosystem compatibility.

        This allows NumType objects to work seamlessly with:
        - matplotlib (plotting)
        - pandas (DataFrames)
        - scikit-learn (machine learning)
        - scipy (scientific computing)

        Parameters
        ----------
        dtype : numpy.dtype, optional
            If specified, convert to this dtype

        Returns
        -------
        numpy.ndarray
            The underlying NumPy array
        """
        data = as_numpy(self._data)
        if dtype is None:
            return data
        else:
            return data.astype(dtype)

    def __array_wrap__(self, result):
        """
        NumPy array wrap protocol.

        This is called when a NumPy ufunc is applied to this object.
        """
        # 微分しない NumPy の関数の結果を包む。形は変えず、数学の型は self が数学の型のときだけ
        if not isinstance(result, (np.ndarray, (cp.ndarray if cp else type(None)))):
            result = get_array_module(self._data).asarray(result)
        return _create_result(result, requires_grad=False, math=_is_math(self))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        NumPy universal function protocol.

        This allows NumPy ufuncs to work with NumType objects while
        maintaining automatic differentiation capability.
        """
        if method == "__call__":
            # Map NumPy ufuncs to our differentiable functions
            ufunc_map = {
                np.add: add,
                np.subtract: sub,
                np.multiply: mul,
                np.divide: div,
                np.power: pow,
                np.matmul: matmul,
                np.maximum: maximum,
                np.minimum: minimum,
                np.arctan2: atan2,
                np.negative: neg,
                np.absolute: absolute,
                np.exp: exp,
                np.log: log,
                np.log2: log2,
                np.log10: log10,
                np.sqrt: sqrt,
                np.sin: sin,
                np.cos: cos,
                np.tan: tan,
                np.arcsin: asin,
                np.arccos: acos,
                np.arctan: atan,
                np.sinh: sinh,
                np.cosh: cosh,
                np.tanh: tanh,
                np.arcsinh: asinh,
                np.arccosh: acosh,
                np.arctanh: atanh,
            }

            if ufunc in ufunc_map:
                # Use our differentiable version（NumType でない入力の変換は、演算の側で
                # 定数として行う）
                func = ufunc_map[ufunc]
                return func(*inputs)
            else:
                # Fall back to NumPy
                arrays = []
                for inp in inputs:
                    if isinstance(inp, NumType):
                        arrays.append(inp._data)
                    else:
                        arrays.append(inp)

                result = ufunc(*arrays, **kwargs)

                if isinstance(result, tuple):
                    return tuple(self.__array_wrap__(r) for r in result)
                else:
                    return self.__array_wrap__(result)
        else:
            return NotImplemented

    # ==============================
    # Conversion Methods
    # ==============================

    def to_numpy(self):
        """
        Explicitly convert to NumPy array.

        Returns
        -------
        numpy.ndarray
            The underlying NumPy array
        """
        return as_numpy(self._data)

    def to_list(self):
        """
        Convert to Python list.

        Returns
        -------
        list
            Python list representation
        """
        return as_numpy(self._data).tolist()

    def __float__(self):
        """
        float() での変換をサポート

        Examples
        --------
        >>> x = real(3.14)
        >>> float(x)
        3.14
        """
        return float(self.item())

    def __int__(self):
        """
        int() での変換をサポート

        Examples
        --------
        >>> x = integer(42)
        >>> int(x)
        42
        """
        return int(self.item())

    def __complex__(self):
        """
        complex() での変換をサポート

        Examples
        --------
        >>> x = cmplx(1+2j)
        >>> complex(x)
        (1+2j)
        """
        return complex(self.item())

    def tolist(self):
        """
        NumPy配列をPythonリストに変換

        Returns
        -------
        list or scalar
            Pythonリストまたはスカラー値

        Examples
        --------
        >>> x = real(3.14)
        >>> x.tolist()
        3.14

        >>> v = vector([1, 2, 3])
        >>> v.tolist()
        [1, 2, 3]

        >>> m = matrix([[1, 2], [3, 4]])
        >>> m.tolist()
        [[1, 2], [3, 4]]
        """
        return self._data.tolist()

    def item(self):
        """
        Get Python scalar (for 0-dimensional arrays).

        Returns
        -------
        scalar
            Python scalar value
        """
        return as_numpy(self._data).item()

    # ==============================
    # String Representation
    # ==============================

    def __repr__(self) -> str:
        return f"{self.__class__.__name__.lower()}({self._data})"

    # ==============================
    # Container Protocol
    # ==============================

    def __len__(self):
        return len(self._data)

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return iter(self._data)


# ==============================
# Tensor Type (Optimized)
# ==============================


class Tensor(NumType):
    """
    Base class for all tensor types (n-dimensional arrays) with automatic differentiation.

    Supports:
    - Arbitrary dimensional arrays
    - Element-wise operations
    - Broadcasting
    - Automatic differentiation
    """

    __slots__ = ()

    def __init__(
        self, data: Any, dtype=None, requires_grad: bool = None, name: str = None
    ):
        if isinstance(data, NumType):
            data = data._data
        if isinstance(data, (np.ndarray, (cp.ndarray if cp else type(None)))):
            if dtype is not None:
                data = data.astype(dtype)
        else:
            xp = cp if _cuda_enabled and cp else np
            data = xp.asarray(data, dtype=dtype)

        if requires_grad is None:
            requires_grad = autograd.is_enabled()

        NumType.__init__(self, data, requires_grad, name)
        self._check_ndim()

    def _check_ndim(self):
        """Override in subclasses to enforce dimensionality"""
        pass

    # ==============================
    # Properties
    # ==============================

    @property
    def rank(self):
        """Number of dimensions (alias for ndim)"""
        return self._data.ndim

    @property
    def T(self):
        """Transpose property"""
        return self.transpose()

    @property
    def real(self):
        """Real part of complex tensor"""
        xp = get_array_module(self._data)
        if xp.iscomplexobj(self._data):
            return type(self)(self._data.real, requires_grad=self.requires_grad)
        return self

    @property
    def imag(self):
        """Imaginary part of complex tensor"""
        xp = get_array_module(self._data)
        if xp.iscomplexobj(self._data):
            return type(self)(self._data.imag, requires_grad=self.requires_grad)
        return type(self)(xp.zeros_like(self._data), requires_grad=False)

    # ==============================
    # Arithmetic operators
    # ==============================

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __floordiv__(self, other):
        return floordiv(self, other)

    def __rfloordiv__(self, other):
        return floordiv(other, self)

    def __mod__(self, other):
        return mod(self, other)

    def __rmod__(self, other):
        return mod(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __rpow__(self, other):
        return pow(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __neg__(self):
        return neg(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return abs(self)

    # ==============================
    # In-place arithmetic operators
    # ==============================

    def __iadd__(self, other):
        self._data += other._data if isinstance(other, NumType) else other
        return self

    def __isub__(self, other):
        self._data -= other._data if isinstance(other, NumType) else other
        return self

    def __imul__(self, other):
        self._data *= other._data if isinstance(other, NumType) else other
        return self

    def __itruediv__(self, other):
        if isinstance(self, Integer):
            raise TypeError(
                "In-place true division not supported for Integer. "
                "Use //= for floor division or convert to Real first."
            )
        self._data /= other._data if isinstance(other, NumType) else other
        return self

    def __ifloordiv__(self, other):
        self._data //= other._data if isinstance(other, NumType) else other
        return self

    def __imod__(self, other):
        self._data %= other._data if isinstance(other, NumType) else other
        return self

    def __ipow__(self, other):
        self._data **= other._data if isinstance(other, NumType) else other
        return self

    # ==============================
    # Bitwise operators
    # ==============================

    def __and__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        return type(self)(self._data & other_data)

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        return type(self)(self._data | other_data)

    def __ror__(self, other):
        return self.__or__(other)

    def __xor__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        return type(self)(self._data ^ other_data)

    def __rxor__(self, other):
        return self.__xor__(other)

    def __lshift__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        return type(self)(self._data << other_data)

    def __rlshift__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        return type(self)(other_data << self._data)

    def __rshift__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        return type(self)(self._data >> other_data)

    def __rrshift__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        return type(self)(other_data >> self._data)

    def __invert__(self):
        return type(self)(~self._data)

    # ==============================
    # In-place bitwise operators
    # ==============================

    def __iand__(self, other):
        self._data &= other._data if isinstance(other, NumType) else other
        return self

    def __ior__(self, other):
        self._data |= other._data if isinstance(other, NumType) else other
        return self

    def __ixor__(self, other):
        self._data ^= other._data if isinstance(other, NumType) else other
        return self

    def __ilshift__(self, other):
        self._data <<= other._data if isinstance(other, NumType) else other
        return self

    def __irshift__(self, other):
        self._data >>= other._data if isinstance(other, NumType) else other
        return self

    # ==============================
    # Comparison operators
    # ==============================

    def __eq__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        try:
            return self._data == other_data
        except TypeError:
            _check_device_match("comparison", self, other)
            raise

    def __ne__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        try:
            return self._data != other_data
        except TypeError:
            _check_device_match("comparison", self, other)
            raise

    def __lt__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        try:
            return self._data < other_data
        except TypeError:
            _check_device_match("comparison", self, other)
            raise

    def __le__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        try:
            return self._data <= other_data
        except TypeError:
            _check_device_match("comparison", self, other)
            raise

    def __gt__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        try:
            return self._data > other_data
        except TypeError:
            _check_device_match("comparison", self, other)
            raise

    def __ge__(self, other):
        other_data = other._data if isinstance(other, NumType) else other
        try:
            return self._data >= other_data
        except TypeError:
            _check_device_match("comparison", self, other)
            raise

    # ==============================
    # Indexing
    # ==============================

    def __getitem__(self, key):
        """Differentiable indexing"""
        return get_item(self, key)

    def __setitem__(self, key, value):
        value_data = value._data if isinstance(value, NumType) else value
        self._data[key] = value_data

    # ==============================
    # Array manipulation methods
    # ==============================

    def reshape(self, *shape, order="C"):
        """Reshape (differentiable)"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape, order=order)

    def transpose(self, *axes):
        """Transpose (differentiable)"""
        if len(axes) == 0:
            return transpose(self)
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            return transpose(self, axes[0])
        else:
            return transpose(self, axes)

    def flatten(self, order="C"):
        """Flatten (NumPy compatible)"""
        return flatten(self)

    def ravel(self, order="C"):
        """Ravel (NumPy compatible)"""
        return flatten(self)

    def squeeze(self, axis=None):
        """Squeeze (NumPy compatible)"""
        xp = get_array_module(self._data)
        result = xp.squeeze(self._data, axis=axis)
        return _create_result(result, math=_is_math(self))

    def _kind_kwargs(self, dtype):
        """
        kind / signed that Integer / Real / Complex need to keep ``dtype``.

        ``dtype`` must be of the same category as ``self`` (``copy`` passes
        its own dtype, ``astype`` checks it with ``_check_scalar_astype``).
        """
        kind = dtype.itemsize * 8
        if isinstance(self, Integer):
            return {"kind": kind, "signed": dtype.kind == "i"}
        if isinstance(self, (Real, Complex)):
            return {"kind": kind}
        return {}

    def copy(self, order="C"):
        """Copy (NumPy compatible)"""
        return type(self)(
            self._data.copy(),
            requires_grad=self.requires_grad,
            **self._kind_kwargs(self._data.dtype),
        )

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        """
        Cast to dtype (NumPy compatible)

        Scalar types (Integer / Real / Complex / Boolean) only change precision
        within their own category. Casting across categories would silently
        change the value (1.5 -> 1, 1+2j -> 1) or ignore ``dtype``, so it raises
        ``CastError``; use a factory such as ``nm.int32(x)`` instead.

        Raises
        ------
        CastError
            If ``self`` is a scalar type and ``dtype`` belongs to another category.
        """
        _check_scalar_astype(self, dtype)
        result = self._data.astype(dtype, copy=copy)
        return type(self)(
            result,
            requires_grad=self.requires_grad,
            **self._kind_kwargs(result.dtype),
        )

    # ==============================
    # Reduction methods
    # ==============================

    def sum(self, axis=None, keepdims=False, dtype=None, out=None, **kwargs):
        """Sum along axis (differentiable, NumPy compatible)"""
        return sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False, dtype=None, out=None, **kwargs):
        """Mean along axis (differentiable, NumPy compatible)"""
        return mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False, out=None, **kwargs):
        """Maximum (NumPy compatible)"""
        xp = get_array_module(self._data)
        axis, keepdims = _math_reduce_args(self, axis, keepdims)
        result = xp.max(self._data, axis=axis, keepdims=keepdims)
        if isinstance(result, (np.ndarray, (cp.ndarray if cp else type(None)))):
            return _create_result(result, math=_is_math(self))
        return _auto_scalar(result)

    def min(self, axis=None, keepdims=False, out=None, **kwargs):
        """Minimum (NumPy compatible)"""
        xp = get_array_module(self._data)
        axis, keepdims = _math_reduce_args(self, axis, keepdims)
        result = xp.min(self._data, axis=axis, keepdims=keepdims)
        if isinstance(result, (np.ndarray, (cp.ndarray if cp else type(None)))):
            return _create_result(result, math=_is_math(self))
        return _auto_scalar(result)

    def all(self, axis=None, keepdims=False):
        """Logical AND reduction"""
        axis, keepdims = _math_reduce_args(self, axis, keepdims)
        result = self._data.all(axis=axis, keepdims=keepdims)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return (
            _create_result(result, math=_is_math(self))
            if result.ndim > 0
            else bool(result)
        )

    def any(self, axis=None, keepdims=False):
        """Logical OR reduction"""
        axis, keepdims = _math_reduce_args(self, axis, keepdims)
        result = self._data.any(axis=axis, keepdims=keepdims)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return (
            _create_result(result, math=_is_math(self))
            if result.ndim > 0
            else bool(result)
        )

    def argmax(self, axis=None, out=None, **kwargs):
        """Argmax (NumPy compatible)"""
        xp = get_array_module(self._data)
        result = xp.argmax(self._data, axis=axis)
        if isinstance(result, (np.ndarray, (cp.ndarray if cp else type(None)))):
            return _create_result(result)
        return int(result)

    def argmin(self, axis=None, out=None, **kwargs):
        """Argmin (NumPy compatible)"""
        xp = get_array_module(self._data)
        result = xp.argmin(self._data, axis=axis)
        if isinstance(result, (np.ndarray, (cp.ndarray if cp else type(None)))):
            return _create_result(result)
        return int(result)

    # ==============================
    # Other methods
    # ==============================

    def dot(self, other):
        """Dot product (differentiable)"""
        return dot(self, other)

    def clip(self, min=None, max=None, out=None, **kwargs):
        """Clip values (NumPy compatible)"""
        xp = get_array_module(self._data)
        result = xp.clip(self._data, min, max)
        return _create_result(result, math=_is_math(self))

    def round(self, decimals=0, out=None):
        """Round (NumPy compatible)"""
        xp = get_array_module(self._data)
        result = xp.round(self._data, decimals)
        return _create_result(result, math=_is_math(self))

    # Placeholder for future implementation
    def std(self, axis=None, keepdims=False, ddof=0, dtype=None, out=None, **kwargs):
        """Standard deviation (not yet implemented)"""
        raise NotImplementedError("std is not yet implemented")

    def var(self, axis=None, keepdims=False, ddof=0, dtype=None, out=None, **kwargs):
        """Variance (not yet implemented)"""
        raise NotImplementedError("var is not yet implemented")

    # ==============================
    # String representation
    # ==============================

    def _formatted_repr(self, prefix="tensor"):
        xp = get_array_module(self._data)
        formatted = xp.array2string(
            self._data,
            precision=4,
            suppress_small=False,
            separator=", ",
            threshold=100,
            max_line_width=120,
            prefix=f"{prefix}(",
        )
        return f"{prefix}({formatted})"

    def __repr__(self):
        return self._formatted_repr("tensor")


# ==============================
# Vector Type (1-dimensional) - Optimized
# ==============================


class Vector(Tensor):
    """Column vector (n×1) with automatic differentiation."""

    __slots__ = ()

    def __init__(
        self, data: Any, dtype=None, requires_grad: bool = None, name: str = None
    ):
        if requires_grad is None:
            requires_grad = autograd.is_enabled()

        super().__init__(data, dtype=dtype, requires_grad=requires_grad, name=name)

    def _check_ndim(self):
        """Vectorの形状を(n, 1)に正規化"""
        # 【重要】self._dataへのアクセスを最小化
        data = self._data
        ndim = data.ndim

        if ndim == 2:
            # すでに2次元の場合、形状チェックのみ
            if data.shape[1] != 1:
                if data.shape[0] == 1:
                    # (1, n) -> (n, 1)に転置（高速）
                    self._data = data.T
                else:
                    raise ValueError(f"Vector must have shape (n, 1), got {data.shape}")
            # すでに(n, 1)なら何もしない（高速パス）
            return

        # ndim == 0 または 1の場合のみreshape
        xp = get_array_module(data)

        if ndim == 0:
            # スカラー -> (1, 1)
            self._data = data.reshape(1, 1)
        elif ndim == 1:
            # 1次元配列 -> (n, 1) - 最も頻繁なケース
            self._data = data.reshape(-1, 1)
        else:
            raise ValueError(f"Vector must be 0, 1 or 2-dimensional, got {ndim}D")

    def __matmul__(self, other):
        """Vector @ X behavior"""
        if isinstance(other, Vector):
            # (n×1) @ (n×1) is invalid
            raise TypeMismatchError(
                "matrix multiplication",
                f"Vector[{len(self)}]",
                f"Vector[{len(other)}]",
                hint="Use dot() for inner product, or x @ y.T for outer product",
            )
        elif isinstance(other, RowVector):
            # (n×1) @ (1×m) = (n×m) → Matrix (outer product)
            xp = get_array_module(self._data)
            result = xp.outer(self._data, other._data)
            return Matrix(result)
        elif isinstance(other, Matrix):
            # (n×1) @ (n×m) is mathematically invalid
            raise TypeMismatchError(
                "matrix multiplication",
                f"Vector[{len(self)}]",
                f"Matrix[{other.shape[0]}x{other.shape[1]}]",
                hint="Transpose the vector first: x.T @ y (this treats x as a row vector)",
            )
        else:
            return matmul(self, other)

    def __rmatmul__(self, other):
        """X @ Vector behavior"""
        if isinstance(other, Matrix):
            # (m×n) @ (n×1) = (m×1) → Vector
            return matmul(other, self)
        elif isinstance(other, RowVector):
            # (1×n) @ (n×1) = scalar (inner product)
            return matmul(other, self)
        else:
            return matmul(other, self)

    def __repr__(self):
        return self._formatted_repr("vector")


# ==============================
# RowVector Type (1-dimensional, but represents row) - Optimized
# ==============================


class RowVector(Tensor):
    """Row vector (1×n) with automatic differentiation."""

    __slots__ = ()

    def __init__(
        self, data: Any, dtype=None, requires_grad: bool = None, name: str = None
    ):
        if requires_grad is None:
            requires_grad = autograd.is_enabled()

        super().__init__(data, dtype=dtype, requires_grad=requires_grad, name=name)

    def _check_ndim(self):
        """RowVectorの形状を(1, n)に正規化"""
        xp = get_array_module(self._data)

        if self._data.ndim == 0:
            # スカラー -> (1, 1)
            self._data = xp.reshape(self._data, (1, 1))
        elif self._data.ndim == 1:
            # 1次元配列 -> (1, n)
            self._data = xp.reshape(self._data, (1, -1))
        elif self._data.ndim == 2:
            if self._data.shape[0] != 1:
                if self._data.shape[1] == 1:
                    # (n, 1) -> (1, n)に転置
                    self._data = self._data.T
                else:
                    raise ValueError(
                        f"RowVector must have shape (1, n), got {self._data.shape}"
                    )
        else:
            raise ValueError(
                f"RowVector must be 0, 1 or 2-dimensional, got {self._data.ndim}D"
            )

    def __matmul__(self, other):
        """RowVector @ X behavior"""
        if isinstance(other, Vector):
            # (1×n) @ (n×1) = scalar (inner product)
            return matmul(self, other)
        elif isinstance(other, RowVector):
            # (1×n) @ (1×m) is invalid
            raise ValueError(
                "Cannot multiply row vector with row vector. "
                "Use rv @ v.T or rv @ matrix."
            )
        elif isinstance(other, Matrix):
            # (1×n) @ (n×m) = (1×m) → RowVector
            return matmul(self, other)
        else:
            return matmul(self, other)

    def __rmatmul__(self, other):
        """X @ RowVector behavior"""
        if isinstance(other, Vector):
            # (n×1) @ (1×m) = (n×m) → Matrix (outer product)
            xp = get_array_module(self._data)
            result = xp.outer(other._data, self._data)
            return Matrix(result)
        elif isinstance(other, Matrix):
            # Matrix @ RowVector is unusual
            raise ValueError(
                "Cannot left-multiply row vector by matrix. "
                "Use M @ rv.T for column vector multiplication."
            )
        else:
            return matmul(other, self)

    def __repr__(self):
        return self._formatted_repr("rowvector")


# ==============================
# Matrix Type (2-dimensional) - Optimized
# ==============================


class Matrix(Tensor):
    """2-dimensional tensor (m×n matrix) with automatic differentiation."""

    __slots__ = ()

    def __init__(
        self, data: Any, dtype=None, requires_grad: bool = None, name: str = None
    ):
        if requires_grad is None:
            requires_grad = autograd.is_enabled()

        super().__init__(data, dtype=dtype, requires_grad=requires_grad, name=name)

    def _check_ndim(self):
        """Matrixが2次元であることを確認"""
        if self._data.ndim == 1:
            # 1次元配列 -> (1, n)の行ベクトルとして扱う
            xp = get_array_module(self._data)
            self._data = xp.reshape(self._data, (1, -1))
        elif self._data.ndim != 2:
            raise ValueError(f"Matrix must be 2-dimensional, got {self._data.ndim}D")

    def __matmul__(self, other):
        """Matrix @ X behavior"""
        if isinstance(other, Vector):
            # (m×n) @ (n×1) = (m×1) → Vector
            return matmul(self, other)
        elif isinstance(other, RowVector):
            # Matrix @ RowVector is unusual
            raise ValueError(
                "Cannot multiply matrix with row vector. "
                "Use M @ rv.T for column vector multiplication."
            )
        elif isinstance(other, Matrix):
            # (m×n) @ (n×p) = (m×p) → Matrix
            return matmul(self, other)
        else:
            return matmul(self, other)

    def __rmatmul__(self, other):
        """X @ Matrix behavior"""
        if isinstance(other, RowVector):
            # (1×n) @ (n×m) = (1×m) → RowVector
            return matmul(other, self)
        elif isinstance(other, Vector):
            # Vector @ Matrix is invalid
            raise ValueError(
                "Cannot left-multiply matrix by column vector. "
                "Use v.T @ M for row vector multiplication."
            )
        else:
            return matmul(other, self)

    def __getitem__(self, key):
        """Matrix indexing (differentiable)"""
        return get_item(self, key)

    def __repr__(self):
        return self._formatted_repr("matrix")


# ==============================
# Scalar Type (Base class for 0-dimensional tensors) - Optimized
# ==============================


class Scalar(Tensor):
    """
    Base class for 0-dimensional tensors (scalars).

    Provides type promotion and automatic casting between
    different scalar types (Boolean, Integer, Real, Complex).
    """

    __slots__ = ()
    _priority = 0  # Type promotion priority

    def __init__(self, data: Any, requires_grad: bool = None, name: str = None):
        # Convert to 0-dimensional array
        if isinstance(data, NumType):
            data = data._data

        if requires_grad is None:
            requires_grad = autograd.is_enabled()

        xp = _array_module_for(data)

        # If already ndarray
        if isinstance(data, (np.ndarray, (cp.ndarray if cp else type(None)))):
            if data.ndim != 0:
                data = data.reshape(())
        else:
            # Convert scalar value to 0-dimensional array
            data = xp.asarray(data)
            if data.ndim != 0:
                data = data.reshape(())

        NumType.__init__(self, data, requires_grad, name)

    def _check_ndim(self):
        """Enforce 0-dimensional constraint"""
        if self._data.ndim != 0:
            raise ValueError(f"Scalar must be 0-dimensional, got {self._data.ndim}D")

    @classmethod
    def _get_priority(cls) -> int:
        """Get type priority for promotion"""
        return cls._priority

    def _promote_types(self, other: Any) -> Tuple["Scalar", "Scalar"]:
        """型プロモーション（requires_grad保持版）"""
        if not isinstance(other, Scalar):
            if type(other) in _PY_NUMBER_TYPES:
                # リテラルは相手（self）の精度で読む
                other = _literal_with_precision_of(
                    other, _auto_scalar(other), self
                )
            else:
                other = _auto_scalar(other)

        if type(self) is type(other):
            return self, other

        if self._get_priority() >= other._get_priority():
            target_type = type(self)
        else:
            target_type = type(other)

        # 変換先の精度は、両方を合わせた dtype に従う（float32 が黙って float64 にならない）
        target_dtype = np.result_type(self._data.dtype, other._data.dtype)
        self_conv = _as_scalar_type(target_type, self, target_dtype)
        other_conv = _as_scalar_type(target_type, other, target_dtype)

        return self_conv, other_conv

    @classmethod
    def _convert(cls, value: "Scalar") -> "Scalar":
        """Convert scalar to this type"""
        return cls(value._data)

    # ==============================
    # Arithmetic operations with type promotion (Optimized)
    # ==============================

    def __add__(self, other):
        # Let Tensor handle if other is Tensor (not Scalar)
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return add(self_conv, other_conv)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return sub(self_conv, other_conv)

    def __rsub__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return sub(other_conv, self_conv)

    def __mul__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return mul(self_conv, other_conv)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        # Division promotes to Real or Complex
        result = div(self_conv, other_conv)
        if isinstance(self_conv, Complex) or isinstance(other_conv, Complex):
            if not isinstance(result, Complex):
                result = Complex(result._data, kind=result._data.dtype.itemsize * 8)
        else:
            if not isinstance(result, Real):
                result = Real(result._data, kind=result._data.dtype.itemsize * 8)
        return result

    def __rtruediv__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        result = div(other_conv, self_conv)
        if isinstance(self_conv, Complex) or isinstance(other_conv, Complex):
            if not isinstance(result, Complex):
                result = Complex(result._data, kind=result._data.dtype.itemsize * 8)
        else:
            if not isinstance(result, Real):
                result = Real(result._data, kind=result._data.dtype.itemsize * 8)
        return result

    def __floordiv__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return floordiv(self_conv, other_conv)

    def __rfloordiv__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return floordiv(other_conv, self_conv)

    def __mod__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return mod(self_conv, other_conv)

    def __rmod__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return mod(other_conv, self_conv)

    def __pow__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return pow(self_conv, other_conv)

    def __rpow__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        self_conv, other_conv = self._promote_types(other)
        return pow(other_conv, self_conv)

    def __neg__(self):
        return neg(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return abs(self)

    # ==============================
    # Scalar-specific comparison (returns Python bool) - Optimized
    # ==============================

    def __eq__(self, other):
        if not isinstance(other, Scalar):
            other = _auto_scalar(other)
        return bool(self._data == other._data)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, Scalar):
            other = _auto_scalar(other)
        return bool(self._data < other._data)

    def __le__(self, other):
        if not isinstance(other, Scalar):
            other = _auto_scalar(other)
        return bool(self._data <= other._data)

    def __gt__(self, other):
        if not isinstance(other, Scalar):
            other = _auto_scalar(other)
        return bool(self._data > other._data)

    def __ge__(self, other):
        if not isinstance(other, Scalar):
            other = _auto_scalar(other)
        return bool(self._data >= other._data)

    # ==============================
    # Compound assignments (Optimized)
    # ==============================

    def __iadd__(self, other):
        self._data += other._data if isinstance(other, NumType) else other
        return self

    def __isub__(self, other):
        self._data -= other._data if isinstance(other, NumType) else other
        return self

    def __imul__(self, other):
        self._data *= other._data if isinstance(other, NumType) else other
        return self

    def __itruediv__(self, other):
        self._data /= other._data if isinstance(other, NumType) else other
        return self

    def __ifloordiv__(self, other):
        self._data //= other._data if isinstance(other, NumType) else other
        return self

    def __imod__(self, other):
        self._data %= other._data if isinstance(other, NumType) else other
        return self

    def __ipow__(self, other):
        self._data **= other._data if isinstance(other, NumType) else other
        return self

    def __repr__(self):
        return f"scalar({self._data})"


# ==============================
# Boolean (NOT differentiable)
# ==============================


class Boolean(Scalar):
    """
    Boolean scalar type (NOT differentiable).

    Used for masks, conditions, and logical operations.
    """

    __slots__ = ()
    _priority = 1

    def __init__(self, data: Any, requires_grad: bool = None, name: str = None):
        if isinstance(data, NumType):
            data = data._data

        xp = _array_module_for(data)
        data = xp.bool_(data)

        # Booleanは常に微分不可
        if requires_grad is None:
            requires_grad = False

        if requires_grad:
            import warnings

            warnings.warn(
                "Boolean type cannot have gradients. Setting requires_grad=False",
                UserWarning,
                stacklevel=2,
            )
            requires_grad = False

        super().__init__(data, requires_grad, name)

    def __repr__(self):
        return f"bool({self._data})"


# ==============================
# Integer (NOT differentiable) - Optimized
# ==============================


class Integer(Scalar):
    """
    Integer scalar type (NOT differentiable).

    Used for indices, counts, and discrete values.

    **Overflow behavior**: Wraps around like C (two's complement).
    - uint8(-1) → 255
    - int8(128) → -128
    - uint16(65536) → 0
    """

    __slots__ = ("kind", "signed")
    _priority = 2

    def __init__(
        self,
        data: Any,
        kind: int = 64,
        signed: bool = True,
        requires_grad: bool = None,
        name: str = None,
    ):
        if isinstance(data, NumType):
            if isinstance(data, Complex):
                raise CastError("Complex", "Integer")
            data = data._data

        xp = _array_module_for(data)

        # Determine dtype from kind and signed
        if signed:
            dtype_map = {8: xp.int8, 16: xp.int16, 32: xp.int32, 64: xp.int64}
        else:
            dtype_map = {8: xp.uint8, 16: xp.uint16, 32: xp.uint32, 64: xp.uint64}

        if kind not in dtype_map:
            raise ValueError(f"Integer kind must be 8, 16, 32, or 64, got {kind}")

        # C-style wrapping behavior
        if isinstance(data, (int, float, np.number)):
            # CRITICAL: Always go through int64 (signed) first!
            # This allows negative numbers to be represented in two's complement
            data = xp.array(int(data), dtype=xp.int64)
        else:
            data = xp.asarray(data)

        # Now cast to target dtype (wrapping occurs here)
        data = data.astype(dtype_map[kind])

        self.kind = kind
        self.signed = signed

        # Integerは常に微分不可
        if requires_grad is None:
            requires_grad = False

        if requires_grad:
            import warnings

            warnings.warn(
                "Integer type cannot have gradients. Setting requires_grad=False",
                UserWarning,
                stacklevel=2,
            )
            requires_grad = False

        super().__init__(data, requires_grad, name)

    # ==============================
    # Arithmetic operations with type promotion (Optimized)
    # Integer同士ならIntegerを返す
    # ==============================

    def __add__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        # int を Integer に変換
        if isinstance(other, int):
            other = Integer(other, kind=self.kind, signed=self.signed)

        if isinstance(other, Integer):
            result_kind = builtins.max(self.kind, other.kind)
            result_signed = self.signed or other.signed
            result = add(self, other)
            return Integer(result._data, kind=result_kind, signed=result_signed)

        self_conv, other_conv = self._promote_types(other)
        return add(self_conv, other_conv)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        if isinstance(other, int):
            other = Integer(other, kind=self.kind, signed=self.signed)

        if isinstance(other, Integer):
            result_kind = builtins.max(self.kind, other.kind)
            result_signed = self.signed or other.signed
            result = sub(self, other)
            return Integer(result._data, kind=result_kind, signed=result_signed)

        self_conv, other_conv = self._promote_types(other)
        return sub(self_conv, other_conv)

    def __rsub__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        if isinstance(other, int):
            other = Integer(other, kind=self.kind, signed=self.signed)

        if isinstance(other, Integer):
            result_kind = builtins.max(self.kind, other.kind)
            result_signed = self.signed or other.signed
            result = sub(other, self)
            return Integer(result._data, kind=result_kind, signed=result_signed)

        self_conv, other_conv = self._promote_types(other)
        return sub(other_conv, self_conv)

    def __mul__(self, other):
        if isinstance(other, Tensor) and not isinstance(other, Scalar):
            return NotImplemented

        if isinstance(other, int):
            other = Integer(other, kind=self.kind, signed=self.signed)

        if isinstance(other, Integer):
            result_kind = builtins.max(self.kind, other.kind)
            result_signed = self.signed or other.signed
            result = mul(self, other)
            return Integer(result._data, kind=result_kind, signed=result_signed)

        self_conv, other_conv = self._promote_types(other)
        return mul(self_conv, other_conv)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        raise TypeError(
            "In-place true division not supported for Integer. "
            "Use //= for floor division or convert to Real first."
        )

    # ==============================
    # Bitwise operations (Integer-specific) - Optimized
    # ==============================

    def __and__(self, other):
        if isinstance(other, (Real, Complex)):
            raise TypeError(
                f"unsupported operand type(s) for &: 'Integer' and '{type(other).__name__}'"
            )
        if not isinstance(other, Integer):
            other = Integer(other)
        return Integer(self._data & other._data, kind=self.kind, signed=self.signed)

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        if isinstance(other, (Real, Complex)):
            raise TypeError(
                f"unsupported operand type(s) for |: 'Integer' and '{type(other).__name__}'"
            )
        if not isinstance(other, Integer):
            other = Integer(other)
        return Integer(self._data | other._data, kind=self.kind, signed=self.signed)

    def __ror__(self, other):
        return self.__or__(other)

    def __xor__(self, other):
        if isinstance(other, (Real, Complex)):
            raise TypeError(
                f"unsupported operand type(s) for ^: 'Integer' and '{type(other).__name__}'"
            )
        if not isinstance(other, Integer):
            other = Integer(other)
        return Integer(self._data ^ other._data, kind=self.kind, signed=self.signed)

    def __rxor__(self, other):
        return self.__xor__(other)

    def __lshift__(self, other):
        if isinstance(other, (Real, Complex)):
            raise TypeError(
                f"unsupported operand type(s) for <<: 'Integer' and '{type(other).__name__}'"
            )
        if not isinstance(other, Integer):
            other = Integer(other)
        return Integer(self._data << other._data, kind=self.kind, signed=self.signed)

    def __rlshift__(self, other):
        if not isinstance(other, Integer):
            other = Integer(other)
        return Integer(other._data << self._data, kind=self.kind, signed=self.signed)

    def __rshift__(self, other):
        if isinstance(other, (Real, Complex)):
            raise TypeError(
                f"unsupported operand type(s) for >>: 'Integer' and '{type(other).__name__}'"
            )
        if not isinstance(other, Integer):
            other = Integer(other)
        return Integer(self._data >> other._data, kind=self.kind, signed=self.signed)

    def __rrshift__(self, other):
        if not isinstance(other, Integer):
            other = Integer(other)
        return Integer(other._data >> self._data, kind=self.kind, signed=self.signed)

    def __invert__(self):
        return Integer(~self._data, kind=self.kind, signed=self.signed)

    # ==============================
    # Binary representation properties
    # ==============================

    @property
    def bin(self):
        """Binary representation"""
        xp = get_array_module(self._data)
        return f"{xp.binary_repr(self._data.item(), width=self.kind)}"

    @property
    def oct(self):
        """Octal representation"""
        return oct(self._data.item())

    @property
    def hex(self):
        """Hexadecimal representation"""
        return hex(self._data.item())

    def __repr__(self):
        type_name = f"int{self.kind}" if self.signed else f"uint{self.kind}"
        return f"{type_name}({self._data})"


# ==============================
# Real (Differentiable) - Optimized
# ==============================


class Real(Scalar):
    """
    Real (floating-point) scalar type (differentiable).
    """

    __slots__ = ("kind",)
    _priority = 3

    def __init__(
        self, data: Any, kind: int = 64, requires_grad: bool = None, name: str = None
    ):
        if isinstance(data, NumType):
            if isinstance(data, Complex):
                raise CastError("Complex", "Real")
            data = data._data

        xp = _array_module_for(data)

        if isinstance(data, (np.ndarray, (cp.ndarray if cp else type(None)))) and (
            kind == 64 and data.dtype == xp.float64
        ):
            # 最適パス: 型変換不要
            pass
        else:
            # kind と中身の dtype が必ず一致するよう、配列でも検証して変換する
            dtype_map = {16: xp.float16, 32: xp.float32, 64: xp.float64}
            if hasattr(xp, "float128"):
                dtype_map[128] = xp.float128
            if kind not in dtype_map:
                raise ValueError(
                    f"Real kind must be 16, 32, 64 (or 128 if available), got {kind}"
                )
            data = xp.asarray(data, dtype=dtype_map[kind])

        self.kind = kind

        if requires_grad is None:
            requires_grad = autograd.is_enabled()

        super().__init__(data, requires_grad, name)

    def __repr__(self):
        if self.kind == 64:
            return f"real({self._data})"
        return f"real{self.kind}({self._data})"


# ==============================
# Complex (Differentiable) - Optimized
# ==============================


class Complex(Scalar):
    """
    Complex scalar type (differentiable).

    Supports automatic differentiation for complex-valued functions.
    """

    __slots__ = ("kind",)
    _priority = 4

    def __init__(
        self, *data, kind: int = 128, requires_grad: bool = None, name: str = None
    ):
        if len(data) == 1:
            value = data[0]
        elif len(data) == 2:
            value = complex(data[0], data[1])
        else:
            raise ValueError("Complex takes 1 or 2 arguments")

        if isinstance(value, NumType):
            value = value._data

        xp = _array_module_for(value)

        # Determine dtype from kind
        dtype_map = {64: xp.complex64, 128: xp.complex128}
        if hasattr(xp, "complex256"):
            dtype_map[256] = xp.complex256

        if kind not in dtype_map:
            raise ValueError(
                f"Complex kind must be 64, 128 (or 256 if available), got {kind}"
            )

        value = xp.asarray(value, dtype=dtype_map[kind])
        self.kind = kind

        # Complexはautograd.is_enabled()に従う
        if requires_grad is None:
            requires_grad = autograd.is_enabled()

        super().__init__(value, requires_grad, name)

    @property
    def real(self):
        """Real part"""
        data = self._data.real
        return Real(
            data, kind=data.dtype.itemsize * 8, requires_grad=self.requires_grad
        )

    @property
    def imag(self):
        """Imaginary part"""
        data = self._data.imag
        return Real(
            data, kind=data.dtype.itemsize * 8, requires_grad=self.requires_grad
        )

    def __repr__(self):
        if self.kind == 128:
            return f"complex({self._data})"
        return f"complex{self.kind}({self._data})"


# ==============================
# Auto Conversion Functions
# ==============================
# グローバルキャッシュ（起動時に一度だけ初期化）
_ARRAY_TYPES = None
_SCALAR_TYPES = None


def _init_type_cache_fast():
    """型チェック用キャッシュの高速初期化"""
    global _ARRAY_TYPES, _SCALAR_TYPES

    if _ARRAY_TYPES is not None:
        return

    # 配列型のタプル
    _ARRAY_TYPES = (np.ndarray,) + ((cp.ndarray,) if cp else ())

    # スカラー型のタプル（Python組み込み + NumPy）
    _SCALAR_TYPES = {
        "python": (bool, int, float, complex),
        "numpy": (np.bool_, np.integer, np.floating, np.complexfloating),
        "cupy": ()
        if cp is None
        else (cp.bool_, cp.integer, cp.floating, cp.complexfloating),
    }


# 初期化を実行
_init_type_cache_fast()


def _auto_scalar(data: Any, requires_grad: bool = False) -> Scalar:
    """
    高速版: 適切なスカラー型に自動変換

    最適化ポイント:
    1. 型チェックの順序を最適化（頻度の高いものを先に）
    2. 早期リターンの徹底
    """
    # すでにScalarなら即座に返す
    if isinstance(data, Scalar):
        return data

    # NumPy/CuPy配列の場合（最も一般的）
    if isinstance(data, _ARRAY_TYPES):
        # dtype.kindで直接分岐（最速）
        dtype = data.dtype
        kind = dtype.kind
        # 値の精度は変えない: Scalar の kind は中身の dtype から決める
        # （float32 の和が float64 になるような、黙った昇格を起こさない）
        bits = dtype.itemsize * 8

        if kind == "f":  # float（最も頻繁）
            return Real(data, kind=bits, requires_grad=requires_grad)
        elif kind in ("i", "u"):  # int, uint
            return Integer(data, kind=bits, signed=kind == "i", requires_grad=False)
        elif kind == "c":  # complex
            return Complex(data, kind=bits, requires_grad=requires_grad)
        elif kind == "b":  # bool
            return Boolean(data, requires_grad=False)

    # Python組み込み型（2番目に一般的）
    data_type = type(data)

    if data_type is float:  # 最も頻繁
        return Real(data, requires_grad=requires_grad)
    elif data_type is int:
        return Integer(data, requires_grad=False)
    elif data_type is complex:
        return Complex(data, requires_grad=requires_grad)
    elif data_type is bool:
        return Boolean(data, requires_grad=False)

    # NumPyスカラー型（低頻度）。配列と同じく、中身の dtype から kind を決める
    if _is_np_float(data):
        return Real(data, kind=data.dtype.itemsize * 8, requires_grad=requires_grad)
    elif _is_np_int(data):
        return Integer(
            data,
            kind=data.dtype.itemsize * 8,
            signed=data.dtype.kind == "i",
            requires_grad=False,
        )
    elif _is_np_complex(data):
        return Complex(data, kind=data.dtype.itemsize * 8, requires_grad=requires_grad)
    elif _is_np_bool(data):
        return Boolean(data, requires_grad=False)

    # デフォルト
    return Real(data, requires_grad=requires_grad)


def _auto_convert(data: Any, requires_grad: bool = None) -> NumType:
    """
    高速版: データを適切なNumTypeに自動変換

    最適化ポイント:
    1. 早期リターンの徹底
    2. 型チェックの最小化
    3. 配列変換の削減
    """
    # 【最速パス】NumTypeなら即座に返す
    if isinstance(data, NumType):
        return data

    # 【高速パス】Python組み込みスカラー
    data_type = type(data)
    if data_type in (bool, int, float, complex):
        if requires_grad is None:
            requires_grad = False
        return _auto_scalar(data, requires_grad=requires_grad)

    # 【高速パス】NumPy/CuPy配列
    if isinstance(data, _ARRAY_TYPES):
        ndim = data.ndim

        if requires_grad is None:
            requires_grad = ndim > 0  # 配列はTrue、0次元はFalse

        # 次元に応じて分岐（最も高速）
        if ndim == 0:
            return _auto_scalar(data, requires_grad=requires_grad)
        else:
            # 暗黙の変換では数学の型（Vector / Matrix）にしない。形も変えない。
            # 1次元を Vector (n, 1) にすると、Tensor (n,) との演算が
            # (n, n) にブロードキャストされてしまう
            return Tensor(data, requires_grad=requires_grad)

    # 【中速パス】NumPyスカラー型
    if isinstance(data, (_SCALAR_TYPES["numpy"] + _SCALAR_TYPES["cupy"])):
        if requires_grad is None:
            requires_grad = False
        return _auto_scalar(data, requires_grad=requires_grad)

    # 【低速パス】リスト/タプル（配列化が必要）
    if isinstance(data, (list, tuple)):
        xp = cp if _cuda_enabled and cp else np
        arr = xp.asarray(data)
        ndim = arr.ndim

        if requires_grad is None:
            requires_grad = ndim > 0

        if ndim == 0:
            return _auto_scalar(arr, requires_grad=requires_grad)
        else:
            return Tensor(arr, requires_grad=requires_grad)

    # 【最低速パス】その他（最後の手段）
    xp = cp if _cuda_enabled and cp else np
    arr = xp.asarray(data)
    ndim = arr.ndim

    if requires_grad is None:
        requires_grad = ndim > 0

    if ndim == 0:
        return _auto_scalar(arr, requires_grad=requires_grad)
    else:
        return Tensor(arr, requires_grad=requires_grad)


def _create_result(
    data: ArrayType, requires_grad: bool = True, math: bool = False
) -> NumType:
    """
    超高速版: オブジェクト生成を最小限に（RowVector対応版）

    ルール:
    - 0次元 → Scalar（自動判定）
    - 2次元 かつ math=True（入力がすべて数学の型）:
      - (n, 1) → Vector（列ベクトル。n×1 行列は列ベクトルそのもの）
      - (1, n) → RowVector（行ベクトル）
      - (n, m) → Matrix
    - それ以外 → Tensor（Tensor の演算結果に、形だけで数学の型をつけない）
    """
    ndim = data.ndim

    # 数学の型どうしの演算: 2次元は形で Matrix/Vector/RowVector に決める
    if ndim == 2 and math:
        rows, cols = data.shape

        # (n, 1) → Vector（列ベクトル）
        if cols == 1:
            result = object.__new__(Vector)
        # (1, n) → RowVector（行ベクトル）
        elif rows == 1:
            result = object.__new__(RowVector)
        # (n, m) → Matrix
        else:
            result = object.__new__(Matrix)

        result._data = data
        result.name = None
        result.grad = None
        result.requires_grad = autograd.is_enabled() and requires_grad
        result._prev = set()
        result._backward = lambda: None
        return result

    # 0次元（スカラー）
    if ndim == 0:
        return _auto_scalar(data, requires_grad=autograd.is_enabled() and requires_grad)

    # それ以外 → Tensor
    result = object.__new__(Tensor)
    result._data = data
    result.name = None
    result.grad = None
    result.requires_grad = autograd.is_enabled() and requires_grad
    result._prev = set()
    result._backward = lambda: None
    return result


# 数学の型（明示したときだけ使う型）
_MATRIX_TYPES = (Vector, RowVector, Matrix)
# 配列の型（勾配の型をそろえるときに使う。スカラーの型は含めない）
_ARRAY_KINDS = (Tensor, Vector, RowVector, Matrix)
_MATH_TYPES = (Vector, RowVector, Matrix, Scalar)
_PY_SCALAR_TYPES = (bool, int, float, complex, np.number)


def _is_math(*xs) -> bool:
    """入力がすべて数学の型かスカラーの数値なら True（Tensor が1つでも混ざれば False）"""
    for x in xs:
        if isinstance(x, NumType):
            if not isinstance(x, _MATH_TYPES):
                return False
        elif not isinstance(x, _PY_SCALAR_TYPES):
            return False
    return True


def _is_scalar_operand(x) -> bool:
    """スカラー（0次元）の値か"""
    return isinstance(x, NumType) and x._data.ndim == 0


def _operand_name(x) -> str:
    return f"{type(x).__name__}{list(x.shape)}"


def _check_elementwise(kind, name, x, y):
    """
    要素ごとの二項演算が数学的に定義されるかを調べ、定義されなければ例外を出す

    形が (n₁, …, n_k) の Tensor は ℝ^(n₁×…×n_k) の元、Vector / RowVector / Matrix は
    行列の空間の元として扱う。和は同じ空間の元どうしでしか定義されない。

    kind:
      "add": 和・差。スカラーどうしか、同じ空間の元どうしだけ
      "mul": 積。スカラー倍か、同じ空間の元どうし（アダマール積）
      "div": 商。x / c（スカラー倍）、c / x（成分ごとの逆数のスカラー倍）、同じ空間の元どうし
      "map": 成分ごとの関数（maximum, atan2, べき乗など）。スカラーはどちら側でもよい
    """
    x_scalar = _is_scalar_operand(x)
    y_scalar = _is_scalar_operand(y)
    if x_scalar and y_scalar:
        return
    if x_scalar or y_scalar:
        if kind == "add":
            raise TypeMismatchError(
                name,
                _operand_name(x),
                _operand_name(y),
                hint=(
                    "Adding a scalar to a vector, matrix or tensor is not defined. "
                    "To add c to every component, write c * nm.ones_like(x)"
                ),
            )
        return

    # どちらもスカラーでない: 同じ空間の元どうしだけ
    if isinstance(x, _MATRIX_TYPES) != isinstance(y, _MATRIX_TYPES):
        raise TypeMismatchError(
            name,
            _operand_name(x),
            _operand_name(y),
            hint=(
                "Tensor and Vector / RowVector / Matrix belong to different spaces. "
                "Convert explicitly, e.g. nm.vector(t) or nm.tensor(v)"
            ),
        )
    if x.shape != y.shape:
        raise DimensionError(
            name,
            x.shape,
            y.shape,
            left_type=_operand_name(x),
            right_type=_operand_name(y),
            hint=(
                "Element-wise operations need identical shapes. "
                "Broadcast explicitly, e.g. nm.broadcast_to(y, x.shape)"
            ),
        )


_PY_NUMBER_TYPES = (bool, int, float, complex)


def _as_scalar_type(target_type, x, dtype):
    """
    x を target_type のスカラーにする。精度（kind）は dtype に合わせる

    呼ぶのは `Scalar._promote_types` だけで、target_type は優先度の高いほうの型。
    Boolean は最低優先度なので、ここに Boolean が来るのは両方 Boolean のときだけだが、
    その場合は呼び出し側が早く返している。
    """
    if type(x) is target_type and x._data.dtype == dtype:
        return x
    bits = dtype.itemsize * 8
    if target_type is Integer:
        return Integer(
            x._data, kind=bits, signed=dtype.kind != "u", requires_grad=False
        )
    return target_type(x._data, kind=bits, requires_grad=x.requires_grad)


def _literal_with_precision_of(value, node, other):
    """
    Python の数のリテラルを、相手の精度に合わせた定数として読み直す

    リテラルの `2` は実数であって「float64 という表現」ではないので、相手が float32 の元なら
    float32 の定数として読む。種類（bool < int < float < complex）はリテラルと相手の大きいほう、
    精度は相手に合わせる（種類が上がるときは既定の 64 ビット）。dtype の決め方は
    `np.result_type` に任せる（NumPy の「弱いスカラー」と同じ規則）。

    ndarray や list のリテラルは「形を持つ定数」なので、今までどおり dtype を変えない。
    """
    if type(value) not in _PY_NUMBER_TYPES:
        return node
    xp = get_array_module(other._data)
    target = np.result_type(other._data.dtype, value)
    # 精度も装置も相手に合っていれば、そのまま使う
    if target == node._data.dtype and (cp is None or xp is get_array_module(node._data)):
        return node
    return _auto_scalar(xp.asarray(value, dtype=target), requires_grad=False)


def _convert_operands(x, y):
    """
    二項演算の入力を NumType にする。NumType でない入力（リテラル）は定数として変換し、
    形がまったく同じなら相手と同じ空間の元として読む。Python の数のリテラルは、
    相手と同じ精度（dtype）で読む
    """
    x_literal = not isinstance(x, NumType)
    y_literal = not isinstance(y, NumType)
    x_value, y_value = x, y
    if x_literal:
        x = _auto_convert(x, requires_grad=False)
    if y_literal:
        y = _auto_convert(y, requires_grad=False)
    if x_literal and not y_literal:
        x = _literal_in_space_of(_literal_with_precision_of(x_value, x, y), y)
    elif y_literal and not x_literal:
        y = _literal_in_space_of(_literal_with_precision_of(y_value, y, x), x)
    return x, y


def _literal_in_space_of(literal, other):
    """
    リテラル（ndarray や list を変換した Tensor）を、形がまったく同じときだけ
    相手（Vector / RowVector / Matrix）と同じ空間の元として読む。形は変えない。
    """
    if (
        isinstance(other, _MATRIX_TYPES)
        and type(literal) is Tensor
        and literal.shape == other.shape
    ):
        return _create_result(literal._data, requires_grad=False, math=True)
    return literal


def _math_reduce_args(x, axis, keepdims):
    """
    数学の型を軸に沿って縮約するときの axis と keepdims を返す

    向きを保つ: 列ごとの和 1ᵀA は行ベクトル、行ごとの和 A1 は列ベクトル。
    結果が 1×1 になる縮約（ベクトルの成分の和など）はスカラーにする。
    """
    if axis is None or not isinstance(x, _MATRIX_TYPES):
        return axis, keepdims
    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(a % x.ndim for a in axes)
    kept_shape = tuple(1 if i in axes else s for i, s in enumerate(x.shape))
    if builtins.all(s == 1 for s in kept_shape):
        return None, False
    return axis, True


def _math_key(x, key):
    """
    数学の型のインデックスで向きを保つための key を返す

    - v[i]（Vector / RowVector）→ i 番目の成分（スカラー）
    - A[i] / A[i, :] → i 行目（行ベクトル eᵢᵀA）
    - A[:, j] → j 列目（列ベクトル Aeⱼ）
    整数とスライス以外を含む key（配列によるインデックスなど）はそのまま返す。
    """
    if not isinstance(x, _MATRIX_TYPES):
        return key
    k = key if isinstance(key, tuple) else (key,)
    if len(k) > 2 or not builtins.all(
        isinstance(e, (int, np.integer, slice)) and not isinstance(e, bool) for e in k
    ):
        return key

    def keep(e):
        # 整数 i を、次元を残すスライス i:i+1 に置き換える
        if isinstance(e, slice):
            return e
        i = int(e)
        return slice(i, i + 1 if i != -1 else None)

    if len(k) == 1:
        if isinstance(x, Vector):
            k = (k[0], 0) if not isinstance(k[0], slice) else (k[0], slice(None))
        elif isinstance(x, RowVector):
            k = (0, k[0]) if not isinstance(k[0], slice) else (slice(None), k[0])
        else:
            k = (k[0], slice(None))

    row, col = k
    if isinstance(row, slice) == isinstance(col, slice):
        # 両方整数（スカラー）か、両方スライス（2次元のまま）
        return k
    return (keep(row), keep(col))


# ==============================
# 算術演算（Arithmetic Operations）
# ==============================
# Note: Basic operations (add, sub, mul, div) are now in the
# "Factory-Created Operations" section after the factory functions.


def pow(x, y):
    """
    べき乗 xʸ（自動微分対応。高階微分にも対応）

    Parameters
    ----------
    x : NumType or array_like
        底
    y : NumType, int, float or array_like
        指数。Python の数（リテラル）なら定数として扱う

    Returns
    -------
    NumType
        xʸ

    Notes
    -----
    微分は数学の定義に従う（1 階も高階も同じ決まり）。

    - x についての微分 y·x^(y-1): y = 0 のとき、0⁰ = 1 の決まりで xʸ は x について定数
      関数 1 なので、微分は x = 0 を含めてどこでも 0
    - y についての微分 xʸ·ln x: x > 0 では定義どおり。x = 0 かつ y > 0 では極限の 0。
      x < 0（ln x が定義されない）と、x = 0 かつ y ≤ 0（xʸ が y について微分できない）
      では nan
    """
    # ───────────────────────────────────────────────────────────
    # yがPythonリテラル（int/float）の場合の特殊処理
    # ───────────────────────────────────────────────────────────
    if isinstance(y, (int, float)):
        y_value = float(y)

        if not isinstance(x, NumType):
            x = _auto_convert(x, requires_grad=False)

        xp = get_array_module(x._data)

        # よく使われる指数を直接処理
        if y_value == 2.0:
            result_data = x._data * x._data
        elif y_value == 3.0:
            result_data = x._data * x._data * x._data
        elif y_value == 0.5:
            result_data = xp.sqrt(x._data)
        elif y_value == -1.0:
            result_data = 1.0 / x._data
        elif y_value == 1.0:
            result_data = x._data.copy()
        elif y_value == 0.0:
            result_data = xp.ones_like(x._data)
        else:
            result_data = xp.power(x._data, y_value)

        # 勾配不要なら早期リターン
        if not (autograd.is_enabled() and x.requires_grad):
            result = _create_result(result_data, math=_is_math(x, y))
            result.requires_grad = False
            if x.requires_grad or (
                x._backward is _backward_built_under_off and not autograd.is_enabled()
            ):
                result._backward = _backward_built_under_off
            return result

        # 勾配必要
        result = _create_result(result_data, math=_is_math(x, y))
        result.requires_grad = True
        result._prev = (x,)  # yはリテラルなので含まない

        x_shape = x.shape
        x_data = x._data

        def _backward(create_graph=False):
            if result.grad is None:
                return

            if create_graph:
                g = result.grad
                if y_value == 0.0:
                    # x⁰ は定数関数 1 なので、微分はどこでも 0
                    _accumulate_graph(x, zeros_like(x))
                elif y_value == 1.0:
                    _accumulate_graph(x, g)
                else:
                    _accumulate_graph(
                        x, g * (y_value * x ** (y_value - 1))
                    )
                return

            grad_data = result.grad._data
            # d(x^y)/dx = y * x^(y-1)。y = 0 は定数関数なので 0（x = 0 で 0·∞ にしない）
            if y_value == 0.0:
                grad_x_raw = xp.zeros_like(grad_data)
            else:
                grad_x_raw = grad_data * y_value * xp.power(x_data, y_value - 1)
            grad_x = sum_to(_create_result(grad_x_raw), x_shape)

            if x.grad is None:
                x.grad = grad_x
            else:
                x.grad._data = x.grad._data + grad_x._data

        result._backward = _backward
        return result

    # ───────────────────────────────────────────────────────────
    # yがNumTypeの場合（一般的なケース）
    # ───────────────────────────────────────────────────────────

    x, y = _convert_operands(x, y)

    # べき乗は成分ごとの関数: スカラーはどちら側でもよく、配列どうしは同じ形だけ
    _check_elementwise("map", "power", x, y)

    xp = get_array_module(x._data)

    # yがスカラーNumTypeの場合、特殊ケースをチェック
    y_value = None
    is_special_case = False

    if y._data.ndim == 0:
        try:
            y_value = float(y._data)

            if y_value == 2.0:
                result_data = x._data * x._data
                is_special_case = True
            elif y_value == 3.0:
                result_data = x._data * x._data * x._data
                is_special_case = True
            elif y_value == 0.5:
                result_data = xp.sqrt(x._data)
                is_special_case = True
            elif y_value == -1.0:
                result_data = 1.0 / x._data
                is_special_case = True
            elif y_value == 1.0:
                result_data = x._data.copy()
                is_special_case = True
            elif y_value == 0.0:
                result_data = xp.ones_like(x._data)
                is_special_case = True
        except (TypeError, ValueError):
            pass

    if not is_special_case:
        try:
            result_data = xp.power(x._data, y._data)
        except TypeError:
            _check_device_match("power", x, y)
            raise

    # 勾配追跡
    x_req = x.requires_grad
    y_req = y.requires_grad

    if not (autograd.is_enabled() and (x_req or y_req)):
        result = _create_result(result_data, math=_is_math(x, y))
        result.requires_grad = False
        if x_req or y_req or (
            (
                x._backward is _backward_built_under_off
                or y._backward is _backward_built_under_off
            )
            and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result = _create_result(result_data, math=_is_math(x, y))
    result.requires_grad = True

    # _prev を構築
    if x_req and y_req:
        result._prev = (x, y)
    elif x_req:
        result._prev = (x,)
    else:  # y_req
        result._prev = (y,)

    x_data = x._data
    y_data = y._data
    x_shape = x.shape
    y_shape = y.shape
    result_data_saved = result_data

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            g = result.grad
            if x_req:
                _accumulate_graph(x, sum_to(_pow_grad_x_graph(g, x, y), x_shape))
            if y_req:
                _accumulate_graph(
                    y, sum_to(_pow_grad_y_graph(g, x, y, result), y_shape)
                )
            return

        grad_data = result.grad._data

        # d(x^y)/dx = y * x^(y-1)
        if x_req:
            # y = 0 では xʸ は x について定数（0⁰ = 1）なので、x = 0 でも微分は 0。
            # その点だけ底を 1 にして 0·∞ を避ける（y = 0 なので値は 0 になる）
            zero_base = (y_data == 0) & (x_data == 0)
            base = xp.where(zero_base, 1, x_data) if zero_base.any() else x_data
            grad_x_raw = grad_data * y_data * xp.power(base, y_data - 1)
            grad_x = sum_to(_create_result(grad_x_raw), x_shape)
            if x.grad is None:
                x.grad = grad_x
            else:
                x.grad._data = x.grad._data + grad_x._data

        # d(x^y)/dy = x^y * log(x)
        if y_req:
            positive = x_data > 0
            if positive.all():
                grad_y_raw = grad_data * result_data_saved * xp.log(x_data)
            else:
                # 正でない x では ln x を計算しない。その点は x = 0 かつ y > 0 なら
                # 極限の 0、それ以外は定義されないので nan（途中の inf·0 は捨てる値）
                safe_x = xp.where(positive, x_data, 1)
                with np.errstate(invalid="ignore"):
                    grad_y_raw = grad_data * result_data_saved * xp.log(safe_x)
                grad_y_raw = xp.where(
                    positive, grad_y_raw, _pow_grad_y_fill(xp, x_data, y_data)
                )
            grad_y = sum_to(_create_result(grad_y_raw), y_shape)

            if y.grad is None:
                y.grad = grad_y
            else:
                y.grad._data = y.grad._data + grad_y._data

    result._backward = _backward
    return result


def _pow_grad_y_fill(xp, x_data, y_data):
    """xʸ の y についての微分の、x ≤ 0 での値: x = 0 かつ y > 0 なら 0、それ以外は nan"""
    return xp.where((x_data == 0) & (y_data > 0), 0.0, xp.nan)


def _pow_grad_x_graph(g, x, y):
    """高階用: xʸ の x についての勾配 g·y·x^(y-1)（y = 0 かつ x = 0 の点は 0）"""
    base = x
    zero_base = (y._data == 0) & (x._data == 0)
    if zero_base.any():
        # その点だけ底を 1 にずらして 0·∞ を避ける（y = 0 なので値は 0 のまま）
        base = _spread(x, g)
        base = base + _as_constant(zero_base, base)
    return g * y * base ** (y - ones_like(y))


def _pow_grad_y_graph(g, x, y, r):
    """高階用: xʸ の y についての勾配 g·xʸ·ln x（x ≤ 0 の点は _pow_grad_y_fill の値）"""
    x_data = x._data
    positive = x_data > 0
    if positive.all():
        return g * r * log(x)
    xp = get_array_module(x_data)
    # 正でない点は底を 1 にして ln を有限にし、定数のマスクで消してから決まりの値を足す
    # （その点の xʸ が inf のときの inf·0 = nan は、決まりの値でも nan なので警告だけ止める）
    base = x + _as_constant(xp.where(positive, 0, 1 - x_data), x)
    fill = xp.where(positive, 0.0, _pow_grad_y_fill(xp, x_data, y._data))
    with np.errstate(invalid="ignore"):
        return g * r * log(base) * _as_constant(positive, x) + _as_constant(fill, g)


# Note: neg, absolute (abs) are now in the "Factory-Created Operations" section


def _sum_to_shape(g, shape):
    """バッチ行列積の勾配 g を、共有していた先頭の軸について足し合わせて shape にする"""
    extra = g.ndim - len(shape)
    if extra > 0:
        g = g.sum(axis=tuple(range(extra)))
    return g


def matmul(x, y):
    """行列乗算（自動微分対応・汎用版）"""
    # 型変換（定数は requires_grad=False）
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)
    if not isinstance(y, NumType):
        y = _auto_convert(y, requires_grad=False)

    xp = get_array_module(x._data)
    if cp is not None:
        _check_device_match("matrix multiplication", x, y)

    # 特別なケースの処理とエラーチェック
    if isinstance(x, Vector) and isinstance(y, Vector):
        # Vector @ Vector は無効
        raise TypeMismatchError(
            "matrix multiplication",
            f"Vector[{len(x)}]",
            f"Vector[{len(y)}]",
            hint="Use dot() for inner product, or x @ y.T for outer product",
        )
    elif isinstance(x, Vector) and isinstance(y, Matrix):
        # Vector @ Matrix は無効
        raise TypeMismatchError(
            "matrix multiplication",
            f"Vector[{len(x)}]",
            f"Matrix[{y.shape[0]}x{y.shape[1]}]",
            hint="Transpose the vector first: x.T @ y (this treats x as a row vector)",
        )
    elif isinstance(x, RowVector) and isinstance(y, Vector):
        # RowVector @ Vector -> スカラー（内積）
        if x.shape[1] != y.shape[0]:
            raise DimensionError(
                "matrix multiplication",
                x.shape,
                y.shape,
                left_type=f"RowVector[1x{x.shape[1]}]",
                right_type=f"Vector[{y.shape[0]}]",
                hint="Inner dimensions must match",
            )
        result_data = xp.dot(x._data.flatten(), y._data.flatten())
    elif isinstance(x, Vector) and isinstance(y, RowVector):
        # Vector @ RowVector -> 行列（外積）
        result_data = x._data @ y._data
    elif isinstance(x, RowVector) and isinstance(y, Matrix):
        # RowVector @ Matrix -> RowVector
        result_data = x._data @ y._data
    elif isinstance(x, Matrix) and isinstance(y, Vector):
        # Matrix @ Vector -> Vector
        if x.shape[1] != y.shape[0]:
            raise DimensionError(
                "matrix multiplication",
                x.shape,
                y.shape,
                left_type=f"Matrix[{x.shape[0]}x{x.shape[1]}]",
                right_type=f"Vector[{y.shape[0]}]",
                hint="Inner dimensions must match",
            )
        result_data = x._data @ y._data
    else:
        # 通常の行列積（N次元はバッチ行列積: 最後の2軸で行列積、それより前の軸はバッチ）
        # 次元チェック: x の最後の軸と、y の最後から2番目の軸（y が1次元ならその軸）
        if x.ndim == 0 or y.ndim == 0:
            raise DimensionError("matrix multiplication", x.shape, y.shape)
        k_y = y.shape[-2] if y.ndim >= 2 else y.shape[0]
        if x.shape[-1] != k_y:
            # より良いヒントを生成
            hint = None
            if len(x.shape) == 2 and len(y.shape) == 2:
                if x.shape[-1] == y.shape[-1]:
                    hint = f"Try transposing the right matrix: x @ y.T (would give {x.shape[0]}x{y.shape[1]})"
                elif x.shape[0] == y.shape[0]:
                    hint = f"Try transposing the left matrix: x.T @ y (would give {x.shape[1]}x{y.shape[1]})"

            raise DimensionError("matrix multiplication", x.shape, y.shape, hint=hint)

        # バッチの軸: 同じ形か、片方がもう片方の後ろの軸だけを持つ（Cᵢ = A Bᵢ）ときだけ定義される。
        # サイズ 1 の軸を暗黙に広げることはしない
        batch_x = x.shape[:-2] if x.ndim >= 2 else ()
        batch_y = y.shape[:-2] if y.ndim >= 2 else ()
        short, long_ = sorted((batch_x, batch_y), key=len)
        if long_[len(long_) - len(short) :] != short:
            raise DimensionError(
                "matrix multiplication",
                x.shape,
                y.shape,
                hint=(
                    f"Batch dimensions {batch_x} and {batch_y} do not match. "
                    "Use nm.broadcast_to to replicate a batch explicitly"
                ),
            )
        result_data = x._data @ y._data

    # 結果の型を決定
    if not isinstance(result_data, (np.ndarray, (cp.ndarray if cp else type(None)))):
        result_data = xp.asarray(result_data)

    # 0次元配列（スカラー）の場合
    if result_data.ndim == 0:
        result = _auto_scalar(result_data)
    # 特別なケースの結果型を明示的に指定
    elif isinstance(x, RowVector) and isinstance(y, Matrix):
        result = RowVector(result_data)
    elif isinstance(x, Matrix) and isinstance(y, Vector):
        result = Vector(result_data)
    elif isinstance(x, Vector) and isinstance(y, RowVector):
        result = Matrix(result_data)
    else:
        result = _create_result(result_data, math=_is_math(x, y))

    # 勾配追跡の早期判定
    x_req = x.requires_grad
    y_req = y.requires_grad

    if not (autograd.is_enabled() and (x_req or y_req)):
        result.requires_grad = False
        if x_req or y_req or (
            (
                x._backward is _backward_built_under_off
                or y._backward is _backward_built_under_off
            )
            and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result.requires_grad = True

    # _prev を構築
    if x_req and y_req:
        result._prev = (x, y)
    elif x_req:
        result._prev = (x,)
    else:  # y_req
        result._prev = (y,)

    # 順伝播時に形状情報を保存
    x_shape = x.shape
    y_shape = y.shape
    x_data = x._data
    y_data = y._data
    is_rowvec_vec = isinstance(x, RowVector) and isinstance(y, Vector)
    is_vec_rowvec = isinstance(x, Vector) and isinstance(y, RowVector)
    is_rowvec_mat = isinstance(x, RowVector) and isinstance(y, Matrix)
    is_mat_vec = isinstance(x, Matrix) and isinstance(y, Vector)

    def _backward():
        grad = result.grad

        # RowVector @ Vector（内積）の特別処理
        if is_rowvec_vec:
            if x_req:
                # grad_x = grad * y.T (スカラー × Vector.T -> RowVector)
                grad_x_data = grad._data * y_data.T
                grad_x = RowVector(grad_x_data)
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad._data = x.grad._data + grad_x._data

            if y_req:
                # grad_y = x.T * grad (RowVector.T × スカラー -> Vector)
                grad_y_data = x_data.T * grad._data
                grad_y = Vector(grad_y_data)
                if y.grad is None:
                    y.grad = grad_y
                else:
                    y.grad._data = y.grad._data + grad_y._data

        # Vector @ RowVector（外積）の特別処理
        elif is_vec_rowvec:
            grad_data = grad._data
            if x_req:
                # grad_x = grad @ y (Matrix @ RowVector.T -> Vector)
                grad_x_data = grad_data @ y_data.T
                grad_x = Vector(grad_x_data)
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad._data = x.grad._data + grad_x._data

            if y_req:
                # grad_y = x.T @ grad (Vector.T @ Matrix -> RowVector)
                grad_y_data = x_data.T @ grad_data
                grad_y = RowVector(grad_y_data)
                if y.grad is None:
                    y.grad = grad_y
                else:
                    y.grad._data = y.grad._data + grad_y._data

        # RowVector @ Matrix の特別処理
        elif is_rowvec_mat:
            grad_data = grad._data
            if x_req:
                # grad_x = grad @ y.T (RowVector @ Matrix.T -> RowVector)
                grad_x_data = grad_data @ y_data.T
                grad_x = RowVector(grad_x_data)
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad._data = x.grad._data + grad_x._data

            if y_req:
                # grad_y = x.T @ grad (RowVector.T @ RowVector -> Matrix)
                grad_y_data = x_data.T @ grad_data
                grad_y = Matrix(grad_y_data)
                if y.grad is None:
                    y.grad = grad_y
                else:
                    y.grad._data = y.grad._data + grad_y._data

        # Matrix @ Vector の特別処理
        elif is_mat_vec:
            grad_data = grad._data
            if x_req:
                # grad_x = grad @ y.T (Vector @ Vector.T -> Matrix)
                grad_x_data = grad_data @ y_data.T
                grad_x = Matrix(grad_x_data)
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad._data = x.grad._data + grad_x._data

            if y_req:
                # grad_y = x.T @ grad (Matrix.T @ Vector -> Vector)
                grad_y_data = x_data.T @ grad_data
                grad_y = Vector(grad_y_data)
                if y.grad is None:
                    y.grad = grad_y
                else:
                    y.grad._data = y.grad._data + grad_y._data

        # 通常の場合の勾配計算（行列積・バッチ行列積・1次元を含む）
        else:
            # 1次元の入力は行列とみなす: x (k,) → (1, k)、y (k,) → (k, 1)
            grad_data = grad._data
            xd = x_data[None, :] if x_data.ndim == 1 else x_data
            yd = y_data[:, None] if y_data.ndim == 1 else y_data
            if x_data.ndim == 1 and y_data.ndim == 1:
                grad_data = xp.reshape(grad_data, (1, 1))
            elif x_data.ndim == 1:
                grad_data = xp.expand_dims(grad_data, -2)
            elif y_data.ndim == 1:
                grad_data = xp.expand_dims(grad_data, -1)

            if x_req:
                # ∂L/∂X = G Yᵀ（最後の2軸を転置）。バッチを共有していた分は足し合わせる
                grad_x_data = grad_data @ xp.swapaxes(yd, -1, -2)
                grad_x_data = _sum_to_shape(grad_x_data, xd.shape).reshape(x_shape)
                grad_x = _create_result(grad_x_data)
                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad._data = x.grad._data + grad_x._data

            if y_req:
                # ∂L/∂Y = Xᵀ G
                grad_y_data = xp.swapaxes(xd, -1, -2) @ grad_data
                grad_y_data = _sum_to_shape(grad_y_data, yd.shape).reshape(y_shape)
                grad_y = _create_result(grad_y_data)
                if y.grad is None:
                    y.grad = grad_y
                else:
                    y.grad._data = y.grad._data + grad_y._data

    result._backward = _backward
    return result


def dot(x, y):
    """内積（自動微分対応）"""
    # 型変換（定数は requires_grad=False）
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)
    if not isinstance(y, NumType):
        y = _auto_convert(y, requires_grad=False)

    xp = get_array_module(x._data)
    if cp is not None:
        _check_device_match("dot", x, y)

    # Vectorの特別処理
    x_is_vector = isinstance(x, (Vector, RowVector))
    y_is_vector = isinstance(y, (Vector, RowVector))

    # ベクトルどうし（Vector / RowVector / 1次元 Tensor）は内積なので、結果はいつもスカラー。
    # 外積が必要なら Vector @ RowVector を使う
    inner = (x_is_vector or x._data.ndim == 1) and (y_is_vector or y._data.ndim == 1)

    if inner:
        result_data = xp.dot(x._data.reshape(-1), y._data.reshape(-1))
    elif x_is_vector or y_is_vector:
        # 片方だけがVector/RowVector
        x_data = x._data.flatten() if x_is_vector else x._data
        y_data = y._data.flatten() if y_is_vector else y._data
        result_data = xp.dot(x_data, y_data)
    else:
        # 通常のdot
        result_data = xp.dot(x._data, y._data)

    if not isinstance(result_data, (np.ndarray, (cp.ndarray if cp else type(None)))):
        result_data = xp.asarray(result_data)

    result = _create_result(result_data, math=_is_math(x, y))

    # 勾配追跡の早期判定
    x_req = x.requires_grad
    y_req = y.requires_grad

    if not (autograd.is_enabled() and (x_req or y_req)):
        result.requires_grad = False
        if x_req or y_req or (
            (
                x._backward is _backward_built_under_off
                or y._backward is _backward_built_under_off
            )
            and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result.requires_grad = True

    # _prev を構築
    if x_req and y_req:
        result._prev = (x, y)
    elif x_req:
        result._prev = (x,)
    else:  # y_req
        result._prev = (y,)

    # 順伝播時に値を保存
    x_data = x._data
    y_data = y._data
    x_shape = x.shape
    y_shape = y.shape

    def _backward():
        grad = result.grad
        grad_data = grad._data

        if x_req:
            if inner:
                # 内積の勾配: 相手のベクトルを自分の形にしたもの
                grad_x_data = (grad_data * y_data.reshape(-1)).reshape(x_shape)
            elif x_is_vector and not y_is_vector:
                # Vectorと他の型
                grad_x_data = grad_data * y_data
                grad_x_data = xp.reshape(grad_x_data, x_shape)
            elif x._data.ndim == 1 and y._data.ndim == 1:
                # 1次元どうし
                grad_x_data = grad_data * y_data
            else:
                # 一般的なケース
                grad_x_data = grad_data @ y_data.T

            grad_x = _create_result(grad_x_data)
            if x.grad is None:
                x.grad = grad_x
            else:
                x.grad._data = x.grad._data + grad_x._data

        if y_req:
            if inner:
                grad_y_data = (grad_data * x_data.reshape(-1)).reshape(y_shape)
            elif y_is_vector and not x_is_vector:
                # Vectorと他の型
                grad_y_data = (
                    x_data.T @ grad_data if x_data.ndim > 1 else grad_data * x_data
                )
                grad_y_data = xp.reshape(grad_y_data, y_shape)
            elif x._data.ndim == 1 and y._data.ndim == 1:
                # 1次元どうし
                grad_y_data = grad_data * x_data
            else:
                # 一般的なケース
                grad_y_data = x_data.T @ grad_data

            grad_y = _create_result(grad_y_data)
            if y.grad is None:
                y.grad = grad_y
            else:
                y.grad._data = y.grad._data + grad_y._data

    result._backward = _backward
    return result


# ==============================
# リダクション演算（Reduction Operations）
# ==============================


def _broadcast_like(v, x):
    """
    v を x と同じ空間の元に広げる（高階用の勾配の式で使う）

    0 次元はスカラー倍で、それ以外は broadcast_to で広げる。数学の型は x に合わせる。
    """
    if v.shape == x.shape:
        return v
    if v._data.ndim == 0:
        return v * ones_like(x)
    out = broadcast_to(v, x.shape)
    tx, to = type(x), type(out)
    if to is not tx and tx in _ARRAY_KINDS and to in _ARRAY_KINDS:
        out = _construct_or_cast(tx, out)
    return out


def _unreduce_graph(g, original_shape, axis, keepdims):
    """高階用: 縮約で消えた軸を大きさ 1 の軸として戻す（そのあと broadcast_to で広げる）"""
    if axis is None or keepdims:
        return g
    ndim = len(original_shape)
    axes = {a % ndim for a in (axis if isinstance(axis, tuple) else (axis,))}
    kept = tuple(1 if i in axes else n for i, n in enumerate(original_shape))
    return reshape(g, kept)


def sum(x, axis=None, keepdims=False):
    """和（自動微分対応）"""
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)
    axis, keepdims = _math_reduce_args(x, axis, keepdims)

    xp = get_array_module(x._data)
    result_data = xp.sum(x._data, axis=axis, keepdims=keepdims)

    if not isinstance(result_data, (np.ndarray, (cp.ndarray if cp else type(None)))):
        result_data = xp.asarray(result_data)

    result = _create_result(result_data, math=_is_math(x))

    # AND条件: autograd.offなら即終了
    if not autograd.is_enabled():
        result.requires_grad = False
        if x.requires_grad or x._backward is _backward_built_under_off:
            result._backward = _backward_built_under_off
        return result

    # AND条件: x.requires_grad=Falseなら終了
    if not x.requires_grad:
        result.requires_grad = False
        return result

    result.requires_grad = True
    result._prev = (x,)
    original_shape = x.shape

    def _backward(create_graph=False):
        if create_graph:
            if result.grad is not None:
                g = _unreduce_graph(result.grad, original_shape, axis, keepdims)
                _accumulate_graph(x, broadcast_to(g, original_shape))
            return

        grad = result.grad
        grad_data = grad._data

        if axis is not None and not keepdims:
            # 削減された次元を復元
            if isinstance(axis, int):
                grad_data = xp.expand_dims(grad_data, axis)
            else:
                for ax in sorted(axis):
                    grad_data = xp.expand_dims(grad_data, ax)

        # 元の形状にブロードキャスト
        grad_x_data = xp.broadcast_to(grad_data, original_shape)
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


def mean(x, axis=None, keepdims=False):
    """平均（自動微分対応）"""
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)
    axis, keepdims = _math_reduce_args(x, axis, keepdims)

    xp = get_array_module(x._data)
    result_data = xp.mean(x._data, axis=axis, keepdims=keepdims)

    if not isinstance(result_data, (np.ndarray, (cp.ndarray if cp else type(None)))):
        result_data = xp.asarray(result_data)

    result = _create_result(result_data, math=_is_math(x))

    # AND条件: autograd.offなら即終了
    if not autograd.is_enabled():
        result.requires_grad = False
        if x.requires_grad or x._backward is _backward_built_under_off:
            result._backward = _backward_built_under_off
        return result

    # AND条件: x.requires_grad=Falseなら終了
    if not x.requires_grad:
        result.requires_grad = False
        return result

    result.requires_grad = True
    result._prev = (x,)
    original_shape = x.shape

    # 平均を取った要素数を計算
    if axis is None:
        n = x._data.size
    else:
        if isinstance(axis, int):
            n = x._data.shape[axis]
        else:
            n = 1
            for ax in axis:
                n *= x._data.shape[ax]

    def _backward(create_graph=False):
        if create_graph:
            if result.grad is not None:
                g = result.grad / n
                g = _unreduce_graph(g, original_shape, axis, keepdims)
                _accumulate_graph(x, broadcast_to(g, original_shape))
            return

        grad = result.grad
        grad_data = grad._data / n

        if axis is not None and not keepdims:
            # 削減された次元を復元
            if isinstance(axis, int):
                grad_data = xp.expand_dims(grad_data, axis)
            else:
                for ax in sorted(axis):
                    grad_data = xp.expand_dims(grad_data, ax)

        # 元の形状にブロードキャスト
        grad_x_data = xp.broadcast_to(grad_data, original_shape)
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


# ==============================
# 形状操作（Shape Operations）
# ==============================


def reshape(x, *shape, order="C"):
    """
    リシェイプ（自動微分対応・NumPy互換）

    Parameters
    ----------
    x : NumType
        入力テンソル
    *shape : int or tuple
        新しい形状
        - -1 を含むことができる（自動計算）
    order : {'C', 'F', 'A'}, optional
        メモリレイアウト（デフォルト: 'C'）

    Returns
    -------
    NumType
        リシェイプされたテンソル

    Examples
    --------
    >>> v = vector([1, 2, 3, 4, 5, 6])
    >>> reshape(v, 2, 3)        # (2, 3)
    >>> reshape(v, (2, 3))      # (2, 3)
    >>> reshape(v, 2, -1)       # (2, 3) - 自動計算
    >>> reshape(v, -1)          # (6,) - フラット化
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    # 引数の正規化
    if len(shape) == 1:
        if isinstance(shape[0], (tuple, list)):
            # reshape(x, (2, 2))
            shape = tuple(shape[0])
        elif isinstance(shape[0], int):
            # reshape(x, -1) -> (shape[0],)
            shape = (shape[0],)
        else:
            shape = tuple(shape)
    else:
        # reshape(x, 2, 2)
        shape = tuple(shape)

    # NumPyのreshapeを使用（-1の処理も自動）
    result_data = x._data.reshape(shape, order=order)
    result = _create_result(result_data, math=_is_math(x))

    # AND条件: autograd.offなら即終了
    if not autograd.is_enabled():
        result.requires_grad = False
        if x.requires_grad or x._backward is _backward_built_under_off:
            result._backward = _backward_built_under_off
        return result

    # AND条件: x.requires_grad=Falseなら終了
    if not x.requires_grad:
        result.requires_grad = False
        return result

    result.requires_grad = True
    result._prev = (x,)
    original_shape = x.shape

    def _backward(create_graph=False):
        if create_graph:
            if result.grad is not None:
                _accumulate_graph(x, reshape(result.grad, original_shape, order=order))
            return

        grad = result.grad
        # 順伝播と同じ order で戻す（order="F" の並べ替えの逆写像）
        grad_x_data = grad._data.reshape(original_shape, order=order)
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


def flatten(x):
    """
    テンソルを1次元に平坦化

    Parameters
    ----------
    x : NumType
        入力テンソル

    Returns
    -------
    Vector
        1次元ベクトル

    Examples
    --------
    >>> m = matrix([[1, 2], [3, 4]])
    >>> v = flatten(m)
    >>> v.shape  # (4,)
    """
    return reshape(x, -1)


def ravel(x):
    """flatten のエイリアス"""
    return flatten(x)


def transpose(x, axes=None):
    """
    Transpose a tensor by permuting its dimensions

    Parameters
    ----------
    x : Tensor
        Input tensor
    axes : tuple of int, optional
        Permutation of axes. If None, reverses the dimensions

    Returns
    -------
    Tensor
        Transposed tensor with automatic gradient tracking if enabled
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    # 特別なケース - Vector/RowVector
    if isinstance(x, Vector):
        # Vector (n, 1) -> RowVector (1, n)
        result = RowVector(x._data.T)

        if not (autograd.is_enabled() and x.requires_grad):
            result.requires_grad = False
            if x.requires_grad or (
                x._backward is _backward_built_under_off and not autograd.is_enabled()
            ):
                result._backward = _backward_built_under_off
            return result

        result.requires_grad = True
        result._prev = (x,)

        def _backward(create_graph=False):
            if result.grad is None:
                return
            if create_graph:
                _accumulate_graph(x, transpose(result.grad))
            elif x.grad is None:
                x.grad = Vector(result.grad._data.T)
            else:
                x.grad._data = x.grad._data + result.grad._data.T

        result._backward = _backward
        return result

    if isinstance(x, RowVector):
        # RowVector (1, n) -> Vector (n, 1)
        result = Vector(x._data.T)

        if not (autograd.is_enabled() and x.requires_grad):
            result.requires_grad = False
            if x.requires_grad or (
                x._backward is _backward_built_under_off and not autograd.is_enabled()
            ):
                result._backward = _backward_built_under_off
            return result

        result.requires_grad = True
        result._prev = (x,)

        def _backward(create_graph=False):
            if result.grad is None:
                return
            if create_graph:
                _accumulate_graph(x, transpose(result.grad))
            elif x.grad is None:
                x.grad = RowVector(result.grad._data.T)
            else:
                x.grad._data = x.grad._data + result.grad._data.T

        result._backward = _backward
        return result

    #  一般的なケース
    xp = get_array_module(x._data)
    ndim = x._data.ndim

    if axes is None:
        result_data = x._data.T
        inv_axes = None  # Tの逆はT
    else:
        axes_norm = tuple(int(ax) % ndim for ax in axes)
        result_data = x._data.transpose(axes_norm)

        # 逆転置の軸を事前計算
        inv_axes = [0] * ndim
        for i, ax in enumerate(axes_norm):
            inv_axes[ax] = i
        inv_axes = tuple(inv_axes)

    # 型を保持
    result = (
        Matrix(result_data) if isinstance(x, Matrix) else _create_result(result_data)
    )

    # 勾配不要なら即座にリターン
    if not (autograd.is_enabled() and x.requires_grad):
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result.requires_grad = True
    result._prev = (x,)

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            _accumulate_graph(x, transpose(result.grad, inv_axes))
            return

        # 逆転置
        grad_x_data = (
            result.grad._data.T
            if inv_axes is None
            else result.grad._data.transpose(inv_axes)
        )
        grad_x = (
            Matrix(grad_x_data)
            if isinstance(x, Matrix)
            else _create_result(grad_x_data)
        )

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


# ==============================
# インデックス操作（Indexing Operations）
# ==============================


def _has_int_array_index(key) -> bool:
    """key に整数の配列（同じ要素を重複して選べるインデックス）が含まれるなら True"""
    for e in key if isinstance(key, tuple) else (key,):
        if isinstance(e, (list, np.ndarray)) or (
            cp is not None and isinstance(e, cp.ndarray)
        ):
            kind = e.dtype.kind if hasattr(e, "dtype") else np.asarray(e).dtype.kind
            if kind in ("i", "u"):
                return True
    return False


def _scatter_add(xp, target, key, values):
    """target[key] += values を、重複したインデックスも足し合わせて行う"""
    if xp is np:
        np.add.at(target, key, values)
    else:
        import cupyx

        cupyx.scatter_add(target, key, values)


def _index_add(g, key, shape):
    """
    形 shape の零の元の key の位置に g を足した値（`get_item` の随伴）

    取り出し `x[key]` は選択行列 P をかける線形写像なので、その随伴は Pᵀ にあたるこの演算。
    2 つは互いに随伴なので、高階の勾配の式でも組にして使える。同じ要素を複数回選ぶ
    インデックスでは、上書きせずに足し合わせる。
    """
    xp = get_array_module(g._data)
    out_data = xp.zeros(shape, dtype=g._data.dtype)
    if _has_int_array_index(key):
        _scatter_add(xp, out_data, key, g._data)
    else:
        out_data[key] = g._data

    result = _create_result(out_data, math=_is_math(g))

    if not (autograd.is_enabled() and g.requires_grad):
        result.requires_grad = False
        if g.requires_grad or (
            g._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result.requires_grad = True
    result._prev = (g,)

    def _backward(create_graph=False):
        if result.grad is None:
            return
        if create_graph:
            _accumulate_graph(g, get_item(result.grad, key))
            return
        grad_g_data = result.grad._data[key]
        grad_g = _create_result(grad_g_data)
        if g.grad is None:
            g.grad = grad_g
        else:
            g.grad._data = g.grad._data + grad_g._data

    result._backward = _backward
    return result


def get_item(x, key):
    """インデックス（自動微分対応・超高速版）"""
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    # インデックスに NumType が含まれていれば中身を使う（0次元は Python の数値にする）
    def _unwrap(k):
        if isinstance(k, NumType):
            return k._data.item() if k._data.ndim == 0 else k._data
        return k

    key = tuple(_unwrap(k) for k in key) if isinstance(key, tuple) else _unwrap(key)

    key = _math_key(x, key)
    x_data = x._data
    if cp is not None:
        # 配列のインデックスは、値と同じ装置のものだけ許す（cupy は host の配列を
        # 黙って device へコピーしてしまうので、こちらで止める）
        _check_index_device(x, key)
    result_data = x_data[key]

    if not isinstance(result_data, (np.ndarray, (cp.ndarray if cp else type(None)))):
        xp = get_array_module(x_data)
        result_data = xp.asarray(result_data)

    result = _create_result(result_data, math=_is_math(x))

    if not (autograd.is_enabled() and x.requires_grad):
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result.requires_grad = True
    result._prev = (x,)

    # クロージャで保持する変数を最小化
    x_shape = x.shape
    result_grad_shape = result.shape

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            _accumulate_graph(x, _index_add(result.grad, key, x_shape))
            return

        xp = get_array_module(x_data)
        grad_x_data = xp.zeros(x_shape, dtype=result.grad._data.dtype)
        if _has_int_array_index(key):
            # 整数配列のインデックスは同じ要素を何度も選べる。取り出しは選択行列 P をかける
            # 線形写像なので、勾配は Pᵀ: 同じ要素への勾配は上書きせずに足し合わせる
            _scatter_add(xp, grad_x_data, key, result.grad._data)
        else:
            # スライスや bool マスクでは同じ要素は1度しか選ばれないので、代入で足りる
            grad_x_data[key] = result.grad._data
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


# ==============================
# 数学関数（Mathematical Functions）
# ==============================
# Note: Mathematical functions (exp, log, sqrt, sin, cos, tan, etc.)
# are now in the "Factory-Created Operations" section after the factory functions.


def random_mask(x, p=0.5, training=True):
    """
    Apply random mask to tensor (自動微分対応)

    Randomly zeros some elements of the input tensor with probability p
    during training. Uses inverted dropout scaling (scale by 1/(1-p)).

    This is a low-level primitive used to implement dropout layers.

    Parameters
    ----------
    x : Tensor
        Input tensor
    p : float
        Probability of zeroing each element (default: 0.5)
    training : bool
        Whether to apply masking (default: True)

    Returns
    -------
    Tensor
        Output tensor with random mask applied

    Examples
    --------
    >>> x = nm.randn(10, 20, requires_grad=True)
    >>> y = random_mask(x, p=0.5, training=True)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    if not training or p == 0:
        return x

    if p < 0 or p >= 1:
        raise ValueError(f"Mask probability must be in [0, 1), got {p}")

    xp = get_array_module(x._data)

    # Generate mask
    mask = xp.random.rand(*x.shape) > p

    # Inverted dropout: scale by 1/(1-p) during training
    scale = 1.0 / (1.0 - p)

    # Apply mask and scale
    result_data = x._data * mask * scale
    result = _create_result(result_data, math=_is_math(x))

    # AND条件: autograd.offなら即終了
    if not autograd.is_enabled():
        result.requires_grad = False
        if x.requires_grad or x._backward is _backward_built_under_off:
            result._backward = _backward_built_under_off
        return result

    # AND条件: x.requires_grad=Falseなら終了
    if not x.requires_grad:
        result.requires_grad = False
        return result

    # ここに来るのは勾配計算が必要な場合のみ
    result.requires_grad = True
    result._prev = (x,)

    # Save mask for backward
    mask_scaled = mask * scale

    def _backward():
        grad = result.grad
        if grad is None:
            return
        grad_x_data = grad._data * mask_scaled
        grad_x = _create_result(grad_x_data)
        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


def random_mask_channel(x, p=0.5, training=True):
    """
    Apply random mask to entire channels (自動微分対応)

    Randomly zeros entire channels of the input tensor with probability p
    during training. Uses inverted dropout scaling (scale by 1/(1-p)).

    This is a low-level primitive used to implement channel dropout (dropout2d).

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (N, C, H, W)
    p : float
        Probability of zeroing each channel (default: 0.5)
    training : bool
        Whether to apply masking (default: True)

    Returns
    -------
    Tensor
        Output tensor with random channel mask applied

    Examples
    --------
    >>> x = nm.randn(4, 16, 28, 28, requires_grad=True)
    >>> y = random_mask_channel(x, p=0.5, training=True)
    >>> loss = nm.sum(y)
    >>> loss.backward()

    Notes
    -----
    Unlike random_mask which zeros individual elements,
    random_mask_channel zeros entire channels. This is more effective for
    convolutional layers where adjacent pixels are strongly correlated.
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    if not training or p == 0:
        return x

    if p < 0 or p >= 1:
        raise ValueError(f"Mask probability must be in [0, 1), got {p}")

    xp = get_array_module(x._data)
    N, C, H, W = x.shape

    # Generate mask for channels (not individual elements)
    # Shape: (N, C, 1, 1) - dropout entire channels
    mask = xp.random.rand(N, C, 1, 1) > p

    # Inverted dropout: scale by 1/(1-p) during training
    scale = 1.0 / (1.0 - p)

    # Apply mask and scale
    # Broadcasting: (N, C, 1, 1) * (N, C, H, W)
    result_data = x._data * mask * scale
    result = _create_result(result_data, math=_is_math(x))

    # AND条件: autograd.offなら即終了
    if not autograd.is_enabled():
        result.requires_grad = False
        if x.requires_grad or x._backward is _backward_built_under_off:
            result._backward = _backward_built_under_off
        return result

    # AND条件: x.requires_grad=Falseなら終了
    if not x.requires_grad:
        result.requires_grad = False
        return result

    # ここに来るのは勾配計算が必要な場合のみ
    result.requires_grad = True
    result._prev = (x,)

    # Save mask for backward
    mask_scaled = mask * scale

    def _backward():
        grad = result.grad
        if grad is None:
            return
        grad_x_data = grad._data * mask_scaled
        grad_x = _create_result(grad_x_data)
        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


# Note: NumPy compatibility aliases (asin, acos, atan, asinh, acosh, atanh)
# are now in the "Factory-Created Operations" section after the factory functions.


# ==============================
# その他の関数（微分なし）
# ==============================


def amax(x, axis=None, keepdims=False):
    """
    Maximum value along axis (reduction operation)

    Returns the maximum of an array or maximum along an axis.

    Parameters
    ----------
    x : Tensor
        Input tensor
    axis : int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    Tensor
        Maximum values (gradient not tracked)

    Examples
    --------
    >>> x = nm.array([[1, 2], [3, 4]])
    >>> nm.amax(x)
    4
    >>> nm.amax(x, axis=0)
    array([3, 4])
    >>> nm.amax(x, axis=1, keepdims=True)
    array([[2], [4]])

    See Also
    --------
    amin : Minimum along axis
    maximum : Element-wise maximum
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)
    axis, keepdims = _math_reduce_args(x, axis, keepdims)

    xp = get_array_module(x._data)
    result_data = xp.max(x._data, axis=axis, keepdims=keepdims)
    result = _create_result(result_data, math=_is_math(x))

    # 勾配は不要（定数として扱う）
    result.requires_grad = False
    return result


def amin(x, axis=None, keepdims=False):
    """
    Minimum value along axis (reduction operation)

    Returns the minimum of an array or minimum along an axis.

    Parameters
    ----------
    x : Tensor
        Input tensor
    axis : int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    Tensor
        Minimum values (gradient not tracked)

    Examples
    --------
    >>> x = nm.array([[1, 2], [3, 4]])
    >>> nm.amin(x)
    1
    >>> nm.amin(x, axis=0)
    array([1, 2])
    >>> nm.amin(x, axis=1, keepdims=True)
    array([[1], [3]])

    See Also
    --------
    amax : Maximum along axis
    minimum : Element-wise minimum
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)
    axis, keepdims = _math_reduce_args(x, axis, keepdims)

    xp = get_array_module(x._data)
    result_data = xp.min(x._data, axis=axis, keepdims=keepdims)
    result = _create_result(result_data, math=_is_math(x))

    # 勾配は不要（定数として扱う）
    result.requires_grad = False
    return result


# Note: maximum, minimum are now in the "Factory-Created Operations" section


def floordiv(x, y):
    """床除算（微分不可能）"""
    x, y = _convert_operands(x, y)
    _check_elementwise("map", "floordiv", x, y)

    result = x._data // y._data
    return _create_result(result, requires_grad=False, math=_is_math(x, y))


def mod(x, y):
    """剰余（微分不可能）"""
    # 型変換（定数は requires_grad=False）
    x, y = _convert_operands(x, y)
    _check_elementwise("map", "mod", x, y)

    result = x._data % y._data
    return _create_result(result, requires_grad=False, math=_is_math(x, y))


# Note: atan2 is now in the "Factory-Created Operations" section


# ==============================
# Factory Functions - Tensors
# ==============================


def _construct_or_cast(cls, data, **kwargs):
    """
    cls の値を作る。data が NumType なら型の変換として扱う。

    型の変換（例: Tensor (n,) → Vector (n, 1)）は、ℝⁿ と ℝⁿˣ¹ の自然な同型、
    つまり形を並べ替えるだけの恒等写像なので、計算グラフをつないだまま勾配を流す。
    requires_grad を明示したときは、今までどおり新しい葉（グラフから切り離した値）を作る。
    """
    if not isinstance(data, NumType) or "requires_grad" in kwargs:
        return cls(data, **kwargs)

    x = data
    result = cls(x._data, requires_grad=False, **kwargs)
    if not (autograd.is_enabled() and x.requires_grad):
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result.requires_grad = True
    result._prev = (x,)
    x_shape = x.shape

    def _backward(create_graph=False):
        if result.grad is None:
            return
        if create_graph:
            # 勾配の型は _accumulate_graph が x の型に戻す
            _accumulate_graph(x, reshape(result.grad, x_shape))
            return
        g = result.grad._data.reshape(x_shape)
        if x.grad is None:
            x.grad = _create_result(g)
        else:
            x.grad._data = x.grad._data + g

    result._backward = _backward
    return result


def tensor(data, **kwargs) -> Tensor:
    """
    Create a tensor.

    Parameters
    ----------
    data : array_like
        Input data
    dtype : dtype, optional
        Data type

    Returns
    -------
    Tensor
        Created tensor
    """
    return _construct_or_cast(Tensor, data, **kwargs)


def ten(data, **kwargs) -> Tensor:
    """Alias for tensor()"""
    return _construct_or_cast(Tensor, data, **kwargs)


def vector(data, **kwargs) -> Vector:
    """
    Create a column vector.

    Parameters
    ----------
    data : array_like
        Input data (1D)
    dtype : dtype, optional
        Data type

    Returns
    -------
    Vector
        Created vector
    """
    return _construct_or_cast(Vector, data, **kwargs)


def vec(data, **kwargs) -> Vector:
    """Alias for vector()"""
    return _construct_or_cast(Vector, data, **kwargs)


def rowvector(data, **kwargs) -> RowVector:
    """
    Create a row vector.

    Parameters
    ----------
    data : array_like
        Input data (1D)
    dtype : dtype, optional
        Data type

    Returns
    -------
    RowVector
        Created row vector
    """
    return _construct_or_cast(RowVector, data, **kwargs)


def rowvec(data, **kwargs) -> RowVector:
    """Alias for rowvector()"""
    return _construct_or_cast(RowVector, data, **kwargs)


def matrix(data, **kwargs) -> Matrix:
    """
    Create a matrix.

    Parameters
    ----------
    data : array_like
        Input data (2D)
    dtype : dtype, optional
        Data type

    Returns
    -------
    Matrix
        Created matrix
    """
    return _construct_or_cast(Matrix, data, **kwargs)


def mat(data, **kwargs) -> Matrix:
    """Alias for matrix()"""
    return _construct_or_cast(Matrix, data, **kwargs)


# ==============================
# Factory Functions - Scalars
# ==============================


def boolean(data) -> Boolean:
    """Create a Boolean scalar"""
    return Boolean(data)


def integer(data, kind: int = 64, signed: bool = True) -> Integer:
    """Create an Integer scalar"""
    return Integer(data, kind=kind, signed=signed)


def int8(data) -> Integer:
    """Create an 8-bit signed integer"""
    return Integer(data, kind=8, signed=True)


def int16(data) -> Integer:
    """Create a 16-bit signed integer"""
    return Integer(data, kind=16, signed=True)


def int32(data) -> Integer:
    """Create a 32-bit signed integer"""
    return Integer(data, kind=32, signed=True)


def int64(data) -> Integer:
    """Create a 64-bit signed integer"""
    return Integer(data, kind=64, signed=True)


def uint8(data) -> Integer:
    """Create an 8-bit unsigned integer"""
    return Integer(data, kind=8, signed=False)


def uint16(data) -> Integer:
    """Create a 16-bit unsigned integer"""
    return Integer(data, kind=16, signed=False)


def uint32(data) -> Integer:
    """Create a 32-bit unsigned integer"""
    return Integer(data, kind=32, signed=False)


def uint64(data) -> Integer:
    """Create a 64-bit unsigned integer"""
    return Integer(data, kind=64, signed=False)


def real(data, kind: int = 64, requires_grad: bool = None, name: str = None) -> Real:
    """Create a Real scalar"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    return Real(data, kind=kind, requires_grad=requires_grad, name=name)


def real16(data, requires_grad: bool = None, name: str = None) -> Real:
    """Create a 16-bit float"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    return Real(data, kind=16, requires_grad=requires_grad, name=name)


def real32(data, requires_grad: bool = None, name: str = None) -> Real:
    """Create a 32-bit float"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    return Real(data, kind=32, requires_grad=requires_grad, name=name)


def real64(data, requires_grad: bool = None, name: str = None) -> Real:
    """Create a 64-bit float"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    return Real(data, kind=64, requires_grad=requires_grad, name=name)


def real128(data, requires_grad: bool = None, name: str = None) -> Real:
    """Create a 128-bit float (if available)"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    return Real(data, kind=128, requires_grad=requires_grad, name=name)


def cmplx(
    *data, kind: int = 128, requires_grad: bool = None, name: str = None
) -> Complex:
    """Create a Complex scalar"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    return Complex(*data, kind=kind, requires_grad=requires_grad, name=name)


def cmplx64(*data, requires_grad: bool = None, name: str = None) -> Complex:
    """Create a 64-bit complex number"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    return Complex(*data, kind=64, requires_grad=requires_grad, name=name)


def cmplx128(*data, requires_grad: bool = None, name: str = None) -> Complex:
    """Create a 128-bit complex number"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    return Complex(*data, kind=128, requires_grad=requires_grad, name=name)


def cmplx256(*data, requires_grad: bool = None, name: str = None) -> Complex:
    """Create a 256-bit complex number (if available)"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    return Complex(*data, kind=256, requires_grad=requires_grad, name=name)


# ==============================
# Zeros Functions
# ==============================


def zeros(*shape, dtype=None, requires_grad=None):
    """
    Create a tensor filled with zeros.

    Parameters
    ----------
    *shape : int or tuple
        Shape of the array
    dtype : dtype, optional
        Data type
    requires_grad : bool, optional
        If True, gradients will be computed for this tensor.
        If None, follows autograd.is_enabled() (default: None)

    Returns
    -------
    Tensor
        Tensor filled with zeros

    Examples
    --------
    >>> b = zeros(128)  # requires_grad follows autograd state
    >>> bias = zeros(10, requires_grad=True)  # explicitly enabled
    """
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(xp.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def zeros_vector(size: int, dtype=None, requires_grad=None):
    """Create a vector filled with zeros"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    return Vector(xp.zeros(size, dtype=dtype), requires_grad=requires_grad)


def zeros_matrix(shape: Tuple[int, int], dtype=None, requires_grad=None):
    """Create a matrix filled with zeros"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    return Matrix(xp.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def zeros_tensor(shape: Tuple[int, ...], dtype=None, requires_grad=None):
    """Create a tensor filled with zeros"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    return Tensor(xp.zeros(shape, dtype=dtype), requires_grad=requires_grad)


# ==============================
# Ones Functions
# ==============================


def ones(*shape, dtype=None, requires_grad=None):
    """
    Create a tensor filled with ones.

    Parameters
    ----------
    *shape : int or tuple
        Shape of the array
    dtype : dtype, optional
        Data type
    requires_grad : bool, optional
        If True, gradients will be computed for this tensor.
        If None, follows autograd.is_enabled() (default: None)

    Returns
    -------
    Tensor
        Tensor filled with ones

    Examples
    --------
    >>> W = ones(784, 128)  # requires_grad follows autograd state
    >>> mask = ones(100, requires_grad=False)  # explicitly disabled
    """
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(xp.ones(shape, dtype=dtype), requires_grad=requires_grad)


def ones_vector(size: int, dtype=None, requires_grad=None):
    """Create a vector filled with ones"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    return Vector(xp.ones(size, dtype=dtype), requires_grad=requires_grad)


def ones_matrix(shape: Tuple[int, int], dtype=None, requires_grad=None):
    """Create a matrix filled with ones"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    return Matrix(xp.ones(shape, dtype=dtype), requires_grad=requires_grad)


def ones_tensor(shape: Tuple[int, ...], dtype=None, requires_grad=None):
    """Create a tensor filled with ones"""
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    return Tensor(xp.ones(shape, dtype=dtype), requires_grad=requires_grad)


# ==============================
# Random Functions
# ==============================


def rand(*shape, low=0.0, high=1.0, requires_grad=None) -> Tensor:
    """
    Create a tensor with random values from uniform distribution.

    Parameters
    ----------
    *shape : int or tuple
        Shape of the array
    low : float, optional
        Lower bound
    high : float, optional
        Upper bound
    requires_grad : bool, optional
        If True, gradients will be computed for this tensor.
        If None, follows autograd.is_enabled() (default: None)

    Returns
    -------
    Tensor
        Tensor with random values
    """
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(xp.random.uniform(low, high, shape), requires_grad=requires_grad)


def randn(*shape, requires_grad=None) -> Tensor:
    """
    Create a tensor with random values from standard normal distribution.

    Parameters
    ----------
    *shape : int or tuple
        Shape of the array
    requires_grad : bool, optional
        If True, gradients will be computed for this tensor.
        If None, follows autograd.is_enabled() (default: None)

    Returns
    -------
    Tensor
        Tensor with random values
    """
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(xp.random.randn(*shape), requires_grad=requires_grad)


def randint(*shape, low=0, high=10, requires_grad=None) -> Tensor:
    """
    Create a tensor with random integers.

    Parameters
    ----------
    *shape : int or tuple
        Shape of the array
    low : int, optional
        Lower bound (inclusive)
    high : int, optional
        Upper bound (exclusive)
    requires_grad : bool, optional
        If True, gradients will be computed for this tensor.
        If None, follows autograd.is_enabled() (default: None)

    Returns
    -------
    Tensor
        Tensor with random integers
    """
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(xp.random.randint(low, high, shape), requires_grad=requires_grad)


def random_vector(
    size: int, low=0.0, high=1.0, dtype=None, requires_grad=None
) -> Vector:
    """
    Create a vector with random values

    Parameters
    ----------
    size : int
        Size of the vector
    low : float, optional
        Lower bound
    high : float, optional
        Upper bound
    dtype : dtype, optional
        Data type
    requires_grad : bool, optional
        If True, gradients will be computed for this tensor.
        If None, follows autograd.is_enabled() (default: None)

    Returns
    -------
    Vector
        Vector with random values
    """
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    data = xp.random.uniform(low, high, size)
    return Vector(data, dtype=dtype, requires_grad=requires_grad)


def random_matrix(
    shape: Tuple[int, int], low=0.0, high=1.0, dtype=None, requires_grad=None
) -> Matrix:
    """
    Create a matrix with random values

    Parameters
    ----------
    shape : Tuple[int, int]
        Shape of the matrix
    low : float, optional
        Lower bound
    high : float, optional
        Upper bound
    dtype : dtype, optional
        Data type
    requires_grad : bool, optional
        If True, gradients will be computed for this tensor.
        If None, follows autograd.is_enabled() (default: None)

    Returns
    -------
    Matrix
        Matrix with random values
    """
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    data = xp.random.uniform(low, high, shape)
    return Matrix(data, dtype=dtype, requires_grad=requires_grad)


def random_tensor(
    shape: Tuple[int, ...], low=0.0, high=1.0, dtype=None, requires_grad=None
) -> Tensor:
    """
    Create a tensor with random values

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the tensor
    low : float, optional
        Lower bound
    high : float, optional
        Upper bound
    dtype : dtype, optional
        Data type
    requires_grad : bool, optional
        If True, gradients will be computed for this tensor.
        If None, follows autograd.is_enabled() (default: None)

    Returns
    -------
    Tensor
        Tensor with random values
    """
    if requires_grad is None:
        requires_grad = autograd.is_enabled()

    xp = cp if _cuda_enabled and cp else np
    data = xp.random.uniform(low, high, shape)
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)


# ==============================
# Random State Control
# ==============================


def seed(s: int) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    s : int
        Random seed value

    Examples
    --------
    >>> seed(42)
    >>> a = rand(2, 3)
    >>> seed(42)
    >>> b = rand(2, 3)
    >>> # a and b will be identical
    """
    xp = cp if _cuda_enabled and cp else np
    xp.random.seed(s)


def get_random_state():
    """
    Get the current random state.

    Returns
    -------
    state : tuple
        Current random state
    """
    xp = cp if _cuda_enabled and cp else np
    return xp.random.get_state()


def set_random_state(state):
    """
    Set the random state.

    Parameters
    ----------
    state : tuple
        Random state obtained from get_random_state()
    """
    xp = cp if _cuda_enabled and cp else np
    xp.random.set_state(state)


# ==============================
# Utility Functions
# ==============================


def one_hot(labels, num_classes=None) -> Tensor:
    """
    Convert class labels to one-hot encoding.

    Parameters
    ----------
    labels : array_like
        Class labels (1D array)
    num_classes : int, optional
        Number of classes. If None, inferred from labels.

    Returns
    -------
    Tensor
        One-hot encoded tensor

    Examples
    --------
    >>> labels = [0, 1, 2, 1]
    >>> one_hot(labels, num_classes=3)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 1., 0.]])
    """
    # Get the array module (np or cp)
    if isinstance(labels, NumType):
        xp = get_array_module(labels._data)
        labels_array = labels._data
    else:
        # labelsが生のnumpy/cupy配列の場合
        if cp is not None and isinstance(labels, cp.ndarray):
            xp = cp
            labels_array = labels
        else:
            xp = np
            labels_array = np.asarray(labels)

    # Flatten and convert to int
    labels_array = labels_array.flatten()

    if labels_array.dtype.kind in ("U", "S", "O"):  # String or object type
        labels_array = labels_array.astype(int)
    labels_array = labels_array.astype(int)

    # Infer num_classes if not provided
    if num_classes is None:
        num_classes = int(labels_array.max()) + 1

    # Create one-hot encoding
    one_hot_array = xp.eye(num_classes, dtype=xp.float32)[labels_array]

    return Tensor(one_hot_array, requires_grad=False)


def eye(n: int, m: int = None, dtype=None) -> Matrix:
    """
    Create an identity matrix.

    Parameters
    ----------
    n : int
        Number of rows
    m : int, optional
        Number of columns (default: n)
    dtype : dtype, optional
        Data type

    Returns
    -------
    Matrix
        Identity matrix
    """
    xp = cp if _cuda_enabled and cp else np
    if m is None:
        m = n
    return Matrix(xp.eye(n, m, dtype=dtype))


def arange(*args, **kwargs) -> Tensor:
    """
    Create a tensor with evenly spaced values.

    Parameters
    ----------
    start : number, optional
        Start of interval
    stop : number
        End of interval
    step : number, optional
        Spacing between values
    dtype : dtype, optional
        Data type

    Returns
    -------
    Tensor
        Array of evenly spaced values
    """
    xp = cp if _cuda_enabled and cp else np
    return Tensor(xp.arange(*args, **kwargs))


def linspace(start, stop, num=50, dtype=None) -> Tensor:
    """
    Create a tensor with evenly spaced values over an interval.

    Parameters
    ----------
    start : number
        Start of interval
    stop : number
        End of interval
    num : int, optional
        Number of samples
    dtype : dtype, optional
        Data type

    Returns
    -------
    Tensor
        Array of evenly spaced values
    """
    xp = cp if _cuda_enabled and cp else np
    return Tensor(xp.linspace(start, stop, num, dtype=dtype))


def concatenate(tensors, axis=0) -> Tensor:
    """
    Concatenate tensors along an axis (autograd version).

    Parameters
    ----------
    tensors : sequence of NumType
        Tensors to concatenate
    axis : int, optional
        Axis along which to concatenate

    Returns
    -------
    Tensor
        Concatenated tensor
    """
    # Convert to NumType if needed
    tensors = [
        _auto_convert(t, requires_grad=False) if not isinstance(t, NumType) else t
        for t in tensors
    ]

    arrays = [t._data for t in tensors]
    xp = get_array_module(arrays[0])
    try:
        result_data = xp.concatenate(arrays, axis=axis)
    except TypeError:
        _check_device_match("concatenate", *tensors)
        raise
    result = Tensor(result_data)

    # Check if any input requires grad
    requires_grads = [t.requires_grad for t in tensors]
    if not (autograd.is_enabled() and builtins.any(requires_grads)):
        result.requires_grad = False
        if builtins.any(requires_grads) or (
            not autograd.is_enabled()
            and builtins.any(t._backward is _backward_built_under_off for t in tensors)
        ):
            result._backward = _backward_built_under_off
        return result

    result.requires_grad = True
    result._prev = tuple(t for t in tensors if t.requires_grad)

    # Save info for backward
    axis_normalized = axis if axis >= 0 else len(result_data.shape) + axis

    def _concat_split_indices():
        """つなぎ目の位置（最後の要素は要らない）"""
        indices = []
        offset = 0
        for t in tensors:
            offset += t._data.shape[axis_normalized]
            indices.append(offset)
        return indices[:-1]

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            parts = split(result.grad, _concat_split_indices(), axis=axis_normalized)
            for t, part in zip(tensors, parts):
                if t.requires_grad:
                    _accumulate_graph(t, part)
            return

        grad_data = result.grad._data
        grad_splits = xp.split(
            grad_data, _concat_split_indices(), axis=axis_normalized
        )

        for t, grad_split in zip(tensors, grad_splits):
            if t.requires_grad:
                grad_t = _create_result(grad_split)
                if t.grad is None:
                    t.grad = grad_t
                else:
                    t.grad._data = t.grad._data + grad_t._data

    result._backward = _backward
    return result


def stack(tensors, axis=0) -> Tensor:
    """
    Stack tensors along a new axis (autograd version).

    Parameters
    ----------
    tensors : sequence of NumType
        Tensors to stack
    axis : int, optional
        Axis along which to stack

    Returns
    -------
    Tensor
        Stacked tensor
    """
    # Convert to NumType if needed
    tensors = [
        _auto_convert(t, requires_grad=False) if not isinstance(t, NumType) else t
        for t in tensors
    ]

    arrays = [t._data for t in tensors]
    xp = get_array_module(arrays[0])
    try:
        result_data = xp.stack(arrays, axis=axis)
    except TypeError:
        _check_device_match("stack", *tensors)
        raise
    result = Tensor(result_data)

    # Check if any input requires grad
    requires_grads = [t.requires_grad for t in tensors]
    if not (autograd.is_enabled() and builtins.any(requires_grads)):
        result.requires_grad = False
        if builtins.any(requires_grads) or (
            not autograd.is_enabled()
            and builtins.any(t._backward is _backward_built_under_off for t in tensors)
        ):
            result._backward = _backward_built_under_off
        return result

    result.requires_grad = True
    result._prev = tuple(t for t in tensors if t.requires_grad)

    # Save info for backward
    axis_normalized = axis if axis >= 0 else len(result_data.shape) + axis + 1

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            parts = split(result.grad, len(tensors), axis=axis_normalized)
            for t, part in zip(tensors, parts):
                if t.requires_grad:
                    _accumulate_graph(t, squeeze(part, axis=axis_normalized))
            return

        grad_data = result.grad._data
        # unstack along the stacked axis
        grad_unstacked = xp.split(grad_data, len(tensors), axis=axis_normalized)

        for t, grad_slice in zip(tensors, grad_unstacked):
            if t.requires_grad:
                # Remove the stacked dimension by squeezing
                grad_slice_squeezed = xp.squeeze(grad_slice, axis=axis_normalized)
                grad_t = _create_result(grad_slice_squeezed)
                if t.grad is None:
                    t.grad = grad_t
                else:
                    t.grad._data = t.grad._data + grad_t._data

    result._backward = _backward
    return result


def bmm(a, b) -> Tensor:
    """
    Batch matrix multiplication.

    Parameters
    ----------
    a : Tensor
        First batch of matrices (batch, n, m)
    b : Tensor
        Second batch of matrices (batch, m, p)

    Returns
    -------
    Tensor
        Batch matrix product (batch, n, p)
    """
    a_data = a._data if isinstance(a, NumType) else a
    b_data = b._data if isinstance(b, NumType) else b

    if a_data.ndim != 3:
        raise ValueError(f"a must be a 3D tensor, got {a_data.ndim}D")
    if b_data.ndim != 3:
        raise ValueError(f"b must be a 3D tensor, got {b_data.ndim}D")

    batch_a, n, m = a_data.shape
    batch_b, m2, p = b_data.shape

    if batch_a != batch_b:
        raise ValueError(f"Batch size mismatch: {batch_a} vs {batch_b}")
    if m != m2:
        raise ValueError(f"Inner dimensions mismatch: {m} vs {m2}")

    result = a_data @ b_data
    return Tensor(result)


def einsum(subscripts: str, *operands) -> Tensor:
    """
    Einstein summation convention.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation
    *operands : NumType
        Arrays for the operation

    Returns
    -------
    Tensor
        Result of Einstein summation

    Examples
    --------
    >>> a = matrix([[1, 2], [3, 4]])
    >>> b = matrix([[5, 6], [7, 8]])
    >>> einsum('ij,jk->ik', a, b)  # Matrix multiplication
    >>> einsum('ii->', a)  # Trace
    >>> v1 = vector([1, 2, 3])
    >>> v2 = vector([4, 5])
    >>> einsum('i,j->ij', v1, v2)  # Outer product (auto-flatten)
    >>> einsum('ij,jk->ik', v1, rowvec([4, 5]))  # Use 2D shape
    """
    arrays = []
    xp = cp if _cuda_enabled and cp else np

    # 添字文字列を解析して、各オペランドの期待次元数を取得
    # "ij,jk->ik" -> ["ij", "jk"]
    # "i,j->ij" -> ["i", "j"]
    subscripts_parts = subscripts.split("->")
    input_subscripts = subscripts_parts[0].split(",")

    for i, op in enumerate(operands):
        if isinstance(op, NumType):
            xp = get_array_module(op._data)

            # 期待される次元数を取得
            if i < len(input_subscripts):
                expected_ndim = len(input_subscripts[i].strip())
                actual_ndim = op._data.ndim

                # Vector/RowVectorで、1次元添字が指定されている場合のみflatten
                if (
                    isinstance(op, (Vector, RowVector))
                    and expected_ndim == 1
                    and actual_ndim == 2
                ):
                    arrays.append(op._data.flatten())
                else:
                    arrays.append(op._data)
            else:
                arrays.append(op._data)
        else:
            arrays.append(op)

    result = xp.einsum(subscripts, *arrays)
    return Tensor(result)


def tensordot(a, b, axes=2) -> Tensor:
    """
    Compute tensor dot product along specified axes.

    Parameters
    ----------
    a, b : NumType
        Tensors to dot
    axes : int or (2,) array_like
        Axes to sum over

    Returns
    -------
    Tensor
        Tensor dot product
    """
    a_data = a._data if isinstance(a, NumType) else a
    b_data = b._data if isinstance(b, NumType) else b

    xp = get_array_module(a_data)
    result = xp.tensordot(a_data, b_data, axes=axes)

    return Tensor(result)


# ==============================
# Broadcasting Operations (ブロードキャスト操作)
# ==============================


def broadcast_to(x, shape):
    """
    Broadcast tensor to specified shape

    Parameters
    ----------
    x : NumType
        Input tensor
    shape : tuple
        Target shape

    Returns
    -------
    NumType
        Broadcasted tensor

    Examples
    --------
    >>> x = tensor([1, 2, 3])
    >>> y = broadcast_to(x, (2, 3))
    >>> y.shape
    (2, 3)
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    xp = get_array_module(x._data)

    # 既に目的の形状の場合はそのまま返す
    if x.shape == shape:
        return x

    # ブロードキャスト
    result_data = xp.broadcast_to(x._data, shape)

    # 勾配追跡の判定
    if not (autograd.is_enabled() and x.requires_grad):
        result = _create_result(result_data, math=_is_math(x))
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    # 勾配が必要な場合
    result = _create_result(result_data, math=_is_math(x))
    result.requires_grad = True
    result._prev = (x,)

    x_shape = x.shape

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            _accumulate_graph(x, sum_to(result.grad, x_shape))
            return

        # broadcast_toの逆操作はsum_to相当
        grad_data = result.grad._data
        grad_x = sum_to(_create_result(grad_data), x_shape)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


def sum_to(x, shape):
    """
    Sum elements along axes to output an array of a given shape

    Parameters
    ----------
    x : NumType
        Input tensor
    shape : tuple
        Target shape

    Returns
    -------
    NumType
        Summed tensor with target shape

    Notes
    -----
    This is the inverse operation of broadcast_to for gradient computation
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    xp = get_array_module(x._data)

    # 既に目的の形状の場合
    if x.shape == shape:
        return x

    # 軸の計算
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    # shapeが1の軸を見つける
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])

    # 合計を実行
    result_data = x._data.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        result_data = result_data.squeeze(lead_axis)

    # 勾配追跡
    if not (autograd.is_enabled() and x.requires_grad):
        result = _create_result(result_data, math=_is_math(x))
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result = _create_result(result_data, math=_is_math(x))
    result.requires_grad = True
    result._prev = (x,)

    x_shape = x.shape

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            # 結果の形 shape は x_shape の後ろの軸にそろっているので、そのまま広げられる
            _accumulate_graph(x, broadcast_to(result.grad, x_shape))
            return

        # sum_toの逆操作はbroadcast_to
        grad_data = result.grad._data
        grad_x_data = xp.broadcast_to(grad_data.reshape(shape), x_shape)
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


# ==============================
# Clipping Operations (クリッピング操作)
# ==============================


def clip(x, min_val=None, max_val=None):
    """
    Clip (limit) the values in a tensor

    Parameters
    ----------
    x : NumType
        Input tensor
    min_val : float, optional
        Minimum value
    max_val : float, optional
        Maximum value

    Returns
    -------
    NumType
        Clipped tensor
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    xp = get_array_module(x._data)
    result_data = xp.clip(x._data, min_val, max_val)

    if not (autograd.is_enabled() and x.requires_grad):
        result = _create_result(result_data, math=_is_math(x))
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result = _create_result(result_data, math=_is_math(x))
    result.requires_grad = True
    result._prev = (x,)

    x_data = x._data

    def _clip_mask():
        """クリップ範囲の中だけ勾配を通すマスク（1 階と高階で同じ決まり）"""
        mask = xp.ones_like(x_data)
        if min_val is not None:
            mask = mask * (x_data >= min_val)
        if max_val is not None:
            mask = mask * (x_data <= max_val)
        return mask

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            g = result.grad
            _accumulate_graph(x, g * _as_constant(_clip_mask(), g))
            return

        grad_data = result.grad._data

        # クリップ範囲内の要素のみ勾配を通す
        grad_x_data = grad_data * _clip_mask()
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


# ==============================
# Dimension Operations (次元操作)
# ==============================


def expand_dims(x, axis):
    """
    Expand the shape of a tensor

    Parameters
    ----------
    x : NumType
        Input tensor
    axis : int
        Position where new axis is placed

    Returns
    -------
    NumType
        Tensor with expanded dimensions
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    xp = get_array_module(x._data)
    result_data = xp.expand_dims(x._data, axis)

    if not (autograd.is_enabled() and x.requires_grad):
        result = _create_result(result_data, math=_is_math(x))
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result = _create_result(result_data, math=_is_math(x))
    result.requires_grad = True
    result._prev = (x,)

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            _accumulate_graph(x, squeeze(result.grad, axis))
            return

        grad_data = result.grad._data
        grad_x_data = xp.squeeze(grad_data, axis)
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


def squeeze(x, axis=None):
    """
    Remove single-dimensional entries from the shape

    Parameters
    ----------
    x : NumType
        Input tensor
    axis : int or tuple of ints, optional
        Selects a subset of single-dimensional entries

    Returns
    -------
    NumType
        Squeezed tensor
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    xp = get_array_module(x._data)
    result_data = xp.squeeze(x._data, axis)

    if not (autograd.is_enabled() and x.requires_grad):
        result = _create_result(result_data, math=_is_math(x))
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result = _create_result(result_data, math=_is_math(x))
    result.requires_grad = True
    result._prev = (x,)

    x_shape = x.shape

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            _accumulate_graph(x, reshape(result.grad, x_shape))
            return

        grad_data = result.grad._data
        grad_x_data = grad_data.reshape(x_shape)
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


# ==============================
# Statistical Operations (統計演算)
# ==============================


def var(x, axis=None, keepdims=False, ddof=0):
    """
    Compute variance along the specified axis

    Parameters
    ----------
    x : NumType
        Input tensor
    axis : int or tuple, optional
        Axis along which to compute variance
    keepdims : bool
        Keep reduced dimensions
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    NumType
        Variance
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)
    axis, keepdims = _math_reduce_args(x, axis, keepdims)

    xp = get_array_module(x._data)

    # 分散の計算
    result_data = xp.var(x._data, axis=axis, keepdims=keepdims, ddof=ddof)

    if not (autograd.is_enabled() and x.requires_grad):
        result = _create_result(result_data, math=_is_math(x))
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result = _create_result(result_data, math=_is_math(x))
    result.requires_grad = True
    result._prev = (x,)

    # 平均を計算（勾配計算用）
    x_mean = xp.mean(x._data, axis=axis, keepdims=True)
    x_shape = x.shape

    def _var_count():
        """割る数 N（ddof を引いたもの）"""
        n = (
            x._data.size
            if axis is None
            else x._data.shape[axis]
            if isinstance(axis, int)
            else np.prod([x._data.shape[ax] for ax in axis])
        )
        return n - ddof

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            g = _unreduce_graph(result.grad, x.shape, axis, keepdims)
            m = mean(x, axis=axis, keepdims=True) if axis is not None else mean(x)
            centered = x - _broadcast_like(m, x)
            _accumulate_graph(
                x, (2.0 / _var_count()) * (centered * _broadcast_like(g, x))
            )
            return

        grad_data = result.grad._data

        # 分散の勾配: 2 * (x - mean) / N
        if not keepdims and axis is not None:
            grad_data = xp.expand_dims(grad_data, axis=axis)

        N = (
            x._data.size
            if axis is None
            else x._data.shape[axis]
            if isinstance(axis, int)
            else np.prod([x._data.shape[ax] for ax in axis])
        )
        N = N - ddof

        grad_x_data = 2.0 * (x._data - x_mean) * grad_data / N
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


def std(x, axis=None, keepdims=False, ddof=0):
    """
    Compute standard deviation along the specified axis

    Parameters
    ----------
    x : NumType
        Input tensor
    axis : int or tuple, optional
        Axis along which to compute standard deviation
    keepdims : bool
        Keep reduced dimensions
    ddof : int
        Delta degrees of freedom

    Returns
    -------
    NumType
        Standard deviation
    """
    # 標準偏差 = sqrt(分散)
    variance = var(x, axis=axis, keepdims=keepdims, ddof=ddof)
    return sqrt(variance)


# ==============================
# Advanced Mathematical Operations (高度な数学演算)
# ==============================


def logsumexp(x, axis=None, keepdims=False):
    """
    Compute log(sum(exp(x))) in a numerically stable way

    Parameters
    ----------
    x : NumType
        Input tensor
    axis : int or tuple, optional
        Axis along which to compute
    keepdims : bool
        Keep reduced dimensions

    Returns
    -------
    NumType
        log(sum(exp(x)))
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)
    axis, keepdims = _math_reduce_args(x, axis, keepdims)

    xp = get_array_module(x._data)

    # 数値的安定性のため最大値を引く
    x_max = xp.max(x._data, axis=axis, keepdims=True)
    x_shifted = x._data - x_max

    # exp -> sum -> log
    sum_exp = xp.sum(xp.exp(x_shifted), axis=axis, keepdims=keepdims)

    if not keepdims:
        # 軸を指定しない全体の縮約はスカラー（nm.sum / nm.mean と同じ）
        x_max = xp.squeeze(x_max, axis=axis) if axis is not None else x_max.reshape(())

    result_data = xp.log(sum_exp) + x_max

    if not (autograd.is_enabled() and x.requires_grad):
        result = _create_result(result_data, math=_is_math(x))
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result = _create_result(result_data, math=_is_math(x))
    result.requires_grad = True
    result._prev = (x,)

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            # logsumexp の勾配は softmax = exp(x - r)（r は出力のノード）
            g = _unreduce_graph(result.grad, x.shape, axis, keepdims)
            r_kept = _unreduce_graph(result, x.shape, axis, keepdims)
            soft = exp(x - _broadcast_like(r_kept, x))
            _accumulate_graph(x, _broadcast_like(g, x) * soft)
            return

        grad_data = result.grad._data

        # logsumexpの勾配: softmax
        if not keepdims and axis is not None:
            grad_data = xp.expand_dims(grad_data, axis=axis)

        # logsumexp の勾配は softmax = exp(x - r)。全体の縮約（axis=None）のときも
        # r を引く（前は exp(r) を掛けていて、形も値も間違っていた）
        r_expanded = (
            xp.expand_dims(result_data, axis=axis)
            if axis is not None and not keepdims
            else result_data
        )
        softmax_vals = xp.exp(x._data - r_expanded)
        grad_x_data = grad_data * softmax_vals
        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


def where(condition, x, y):
    """
    Return elements from x or y depending on condition

    Parameters
    ----------
    condition : NumType or array-like
        Condition array
    x : NumType
        Values where condition is True
    y : NumType
        Values where condition is False

    Returns
    -------
    NumType
        Selected elements
    """
    # 型変換
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)
    if not isinstance(y, NumType):
        y = _auto_convert(y, requires_grad=False)

    xp = get_array_module(x._data)

    # conditionをnumpy/cupy配列に変換
    if isinstance(condition, NumType):
        cond_data = condition._data
    else:
        cond_data = xp.asarray(condition)

    try:
        result_data = xp.where(cond_data, x._data, y._data)
    except TypeError:
        _check_device_match("where", condition, x, y)
        raise

    x_req = x.requires_grad
    y_req = y.requires_grad

    if not (autograd.is_enabled() and (x_req or y_req)):
        result = _create_result(result_data, math=_is_math(x, y))
        result.requires_grad = False
        if x_req or y_req or (
            (
                x._backward is _backward_built_under_off
                or y._backward is _backward_built_under_off
            )
            and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result = _create_result(result_data, math=_is_math(x, y))
    result.requires_grad = True

    if x_req and y_req:
        result._prev = (x, y)
    elif x_req:
        result._prev = (x,)
    else:
        result._prev = (y,)

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            g = result.grad
            zero = zeros_like(g)
            if x_req:
                _accumulate_graph(x, where(cond_data, g, zero))
            if y_req:
                _accumulate_graph(y, where(cond_data, zero, g))
            return

        grad_data = result.grad._data

        if x_req:
            grad_x_data = xp.where(cond_data, grad_data, 0)
            grad_x = _create_result(grad_x_data)
            if x.grad is None:
                x.grad = grad_x
            else:
                x.grad._data = x.grad._data + grad_x._data

        if y_req:
            grad_y_data = xp.where(cond_data, 0, grad_data)
            grad_y = _create_result(grad_y_data)
            if y.grad is None:
                y.grad = grad_y
            else:
                y.grad._data = y.grad._data + grad_y._data

    result._backward = _backward
    return result


# ==============================
# Tensor Manipulation Operations (テンソル操作)
# ==============================


def split(x, indices_or_sections, axis=0):
    """
    Split a tensor into multiple sub-tensors

    Parameters
    ----------
    x : NumType
        Input tensor
    indices_or_sections : int or list
        If int, number of equal splits
        If list, indices for splits
    axis : int
        Axis along which to split

    Returns
    -------
    list of NumType
        List of sub-tensors
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    xp = get_array_module(x._data)

    # 分割
    split_arrays = xp.split(x._data, indices_or_sections, axis=axis)

    if not (autograd.is_enabled() and x.requires_grad):
        results = [
            _create_result(arr, requires_grad=False, math=_is_math(x))
            for arr in split_arrays
        ]
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            for result in results:
                result._backward = _backward_built_under_off
        return results

    # 各分割に対してTensorを作成
    results = []
    split_infos = []  # 勾配計算用の情報を保存

    for i, arr in enumerate(split_arrays):
        result = _create_result(arr, math=_is_math(x))
        result.requires_grad = True
        result._prev = (x,)

        # 分割情報を保存
        split_info = {
            "index": i,
            "axis": axis,
            "total_splits": len(split_arrays),
            "shape": x.shape,
        }
        split_infos.append(split_info)

        def make_backward(info):
            def _backward(create_graph=False):
                if results[info["index"]].grad is None:
                    return

                if create_graph:
                    # 自分のぶんの勾配と、ほかは 0 をつなぐ（随伴は concatenate）
                    parts = []
                    for j in range(info["total_splits"]):
                        if j == info["index"]:
                            parts.append(results[j].grad)
                        else:
                            shape = list(info["shape"])
                            shape[info["axis"]] = split_arrays[j].shape[info["axis"]]
                            parts.append(
                                _as_constant(
                                    xp.zeros(tuple(shape), dtype=x._data.dtype),
                                    results[info["index"]].grad,
                                )
                            )
                    _accumulate_graph(
                        x, concatenate(parts, axis=info["axis"])
                    )
                    return

                # 勾配を結合するためのリストを作成
                grad_parts = []
                for j in range(info["total_splits"]):
                    if j == info["index"]:
                        grad_parts.append(results[j].grad._data)
                    else:
                        # 他の部分はゼロ
                        shape = list(info["shape"])
                        shape[info["axis"]] = split_arrays[j].shape[info["axis"]]
                        grad_parts.append(xp.zeros(tuple(shape), dtype=x._data.dtype))

                # 勾配を結合
                grad_x_data = xp.concatenate(grad_parts, axis=info["axis"])
                grad_x = _create_result(grad_x_data)

                if x.grad is None:
                    x.grad = grad_x
                else:
                    x.grad._data = x.grad._data + grad_x._data

            return _backward

        result._backward = make_backward(split_info)
        results.append(result)

    return results


def tile(x, reps):
    """
    Construct a tensor by repeating x

    Parameters
    ----------
    x : NumType
        Input tensor
    reps : tuple or int
        Number of repetitions along each axis

    Returns
    -------
    NumType
        Tiled tensor
    """
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    # repsを正規化（intの場合はtupleに）
    if isinstance(reps, int):
        reps = (reps,)

    xp = get_array_module(x._data)
    result_data = xp.tile(x._data, reps)

    if not (autograd.is_enabled() and x.requires_grad):
        result = _create_result(result_data, math=_is_math(x))
        result.requires_grad = False
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result = _create_result(result_data, math=_is_math(x))
    result.requires_grad = True
    result._prev = (x,)

    x_shape = x.shape

    def _tile_padded():
        """x の形と reps の長さをそろえた (形, 繰り返し回数) を返す"""
        ndim_diff = len(reps) - len(x_shape)
        padded_x_shape = (1,) * ndim_diff + x_shape if ndim_diff > 0 else x_shape
        padded_reps = (1,) * (-ndim_diff) + reps if ndim_diff < 0 else reps
        return ndim_diff, padded_x_shape, padded_reps

    def _backward(create_graph=False):
        if result.grad is None:
            return

        if create_graph:
            # 繰り返した軸を分けて足すのは、reshape と sum の組み合わせで書ける
            ndim_diff, padded_x_shape, padded_reps = _tile_padded()
            g = result.grad
            for axis, (orig_size, rep_count) in enumerate(
                zip(padded_x_shape, padded_reps)
            ):
                if rep_count > 1:
                    new_shape = list(g.shape)
                    new_shape[axis] = rep_count
                    new_shape.insert(axis + 1, orig_size)
                    g = sum(reshape(g, tuple(new_shape)), axis=axis)
            if ndim_diff > 0:
                g = reshape(g, x_shape)
            _accumulate_graph(x, g)
            return

        grad_data = result.grad._data

        # tileの逆操作: 繰り返された部分を合計
        # 元の形状と繰り返し回数から、どの軸でどれだけ繰り返されたかを計算

        # repsとx_shapeの長さを合わせる
        # repsが長い場合: xに先頭次元を追加
        # x_shapeが長い場合: repsに1を追加
        ndim_diff = len(reps) - len(x_shape)
        if ndim_diff > 0:
            # xの形状を拡張（先頭に1を追加）
            padded_x_shape = (1,) * ndim_diff + x_shape
        else:
            padded_x_shape = x_shape

        if ndim_diff < 0:
            # repsを拡張（先頭に1を追加）
            padded_reps = (1,) * (-ndim_diff) + reps
        else:
            padded_reps = reps

        # 勾配を元の形状に戻す
        grad_x_data = grad_data

        # 各軸について逆方向に処理
        for axis, (orig_size, rep) in enumerate(zip(padded_x_shape, padded_reps)):
            if rep > 1:
                # その軸に沿ってreshapeして合計
                # 現在の形状を取得
                current_shape = grad_x_data.shape

                # reshapeのための新しい形状を作成
                new_shape = list(current_shape)
                new_shape[axis] = rep
                new_shape.insert(axis + 1, orig_size)

                # reshape -> sum
                grad_x_data = grad_x_data.reshape(new_shape)
                grad_x_data = grad_x_data.sum(axis=axis)

        # 元の形状に戻す（先頭の余分な次元を削除）
        if ndim_diff > 0:
            # 先頭の余分な次元を削除
            for _ in range(ndim_diff):
                grad_x_data = grad_x_data.squeeze(0)

        grad_x = _create_result(grad_x_data)

        if x.grad is None:
            x.grad = grad_x
        else:
            x.grad._data = x.grad._data + grad_x._data

    result._backward = _backward
    return result


# ==============================
# im2col / col2im (for convolution)
# ==============================


def _pair(v):
    """整数なら (v, v)、(縦, 横) ならそのまま返す"""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def im2col(img, kernel_h, kernel_w, stride=1, padding=0, dilation=1):
    """
    Convert image to column matrix for convolution (optimized version)

    Transforms 4D input tensor into 2D matrix where each column contains
    the values in a receptive field. This allows convolution to be implemented
    as matrix multiplication.

    Parameters
    ----------
    img : ndarray
        Input image of shape (N, C, H, W)
    kernel_h : int
        Kernel height
    kernel_w : int
        Kernel width
    stride : int or (int, int), optional
        Stride (default: 1). 縦と横で別々の値を (縦, 横) で渡せる
    padding : int or (int, int), optional
        Padding (default: 0). 縦と横で別々の値を (縦, 横) で渡せる
    dilation : int or (int, int), optional
        Dilation (default: 1). 縦と横で別々の値を (縦, 横) で渡せる

    Returns
    -------
    ndarray
        Column matrix of shape (N, C*kernel_h*kernel_w, out_h*out_w)

    Examples
    --------
    >>> import numlib as nm
    >>> x = nm.randn(2, 3, 8, 8)
    >>> col = nm.im2col(x._data, 3, 3, stride=1, padding=1)
    >>> col.shape
    (2, 27, 64)  # 2 samples, 3*3*3=27 features, 8*8=64 positions

    Notes
    -----
    This optimized version uses advanced indexing and broadcasting
    to avoid Python loops, making it significantly faster.
    """
    xp = get_array_module(img)
    N, C, H, W = img.shape
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    # Calculate output dimensions
    out_h = (H + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1

    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        img = xp.pad(
            img,
            [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)],
            mode="constant",
            constant_values=0,
        )

    # Create index arrays for advanced indexing (vectorized)
    # Generate all positions where we need to sample

    # Starting positions for each output location
    i0 = xp.arange(kernel_h) * dil_h  # (kernel_h,)
    i1 = xp.arange(kernel_w) * dil_w  # (kernel_w,)
    i0 = xp.repeat(i0, kernel_w)  # (kernel_h * kernel_w,)
    i1 = xp.tile(i1, kernel_h)  # (kernel_h * kernel_w,)

    # Output positions
    j0 = xp.arange(out_h) * stride_h  # (out_h,)
    j1 = xp.arange(out_w) * stride_w  # (out_w,)

    # Combine to get all sampling positions
    # Broadcasting: (kernel_h*kernel_w, 1) + (1, out_h) -> (kernel_h*kernel_w, out_h)
    i = i0.reshape(-1, 1) + j0.reshape(1, -1)  # (kernel_h*kernel_w, out_h)
    j = i1.reshape(-1, 1) + j1.reshape(1, -1)  # (kernel_h*kernel_w, out_w)

    # Reshape for indexing: (kernel_h*kernel_w, out_h, out_w)
    i = xp.repeat(i[:, :, xp.newaxis], out_w, axis=2)
    j = xp.repeat(j[:, xp.newaxis, :], out_h, axis=1)

    # Extract all patches at once using advanced indexing
    # img: (N, C, H_pad, W_pad)
    # We want: img[:, :, i, j] where i, j are the positions

    # Create channel indices
    k = xp.arange(C)  # (C,)

    # Extract using advanced indexing
    # Result shape: (N, C, kernel_h*kernel_w, out_h, out_w)
    col = img[:, k[:, None, None, None], i[None, None, :, :, :], j[None, None, :, :, :]]

    # Reshape to (N, C*kernel_h*kernel_w, out_h*out_w)
    col = col.reshape(N, C * kernel_h * kernel_w, out_h * out_w)

    return col


def col2im(col, input_shape, kernel_h, kernel_w, stride=1, padding=0, dilation=1):
    """
    Convert column matrix back to image (fully vectorized version)

    stride, padding, dilation は im2col と同じく、整数か (縦, 横) のタプル。
    """
    xp = get_array_module(col)
    N, C, H, W = input_shape
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    # Calculate output dimensions
    out_h = (H + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1

    # Reshape col to (N, C, kernel_h*kernel_w, out_h, out_w)
    col = col.reshape(N, C, kernel_h * kernel_w, out_h, out_w)

    # Initialize output image (with padding)
    H_pad = H + 2 * pad_h
    W_pad = W + 2 * pad_w
    img = xp.zeros((N, C, H_pad, W_pad), dtype=col.dtype)

    # Create index arrays
    i0 = xp.arange(kernel_h) * dil_h
    i1 = xp.arange(kernel_w) * dil_w
    i0 = xp.repeat(i0, kernel_w)  # (kernel_h*kernel_w,)
    i1 = xp.tile(i1, kernel_h)  # (kernel_h*kernel_w,)

    j0 = xp.arange(out_h) * stride_h  # (out_h,)
    j1 = xp.arange(out_w) * stride_w  # (out_w,)

    # Create all position indices
    i = i0.reshape(-1, 1, 1) + j0.reshape(1, -1, 1)  # (kernel_h*kernel_w, out_h, 1)
    j = i1.reshape(-1, 1, 1) + j1.reshape(1, 1, -1)  # (kernel_h*kernel_w, 1, out_w)
    i = xp.broadcast_to(i, (kernel_h * kernel_w, out_h, out_w))
    j = xp.broadcast_to(j, (kernel_h * kernel_w, out_h, out_w))

    # Create batch and channel index arrays for vectorized add.at
    n_idx = xp.arange(N)[:, None, None, None, None]  # (N, 1, 1, 1, 1)
    c_idx = xp.arange(C)[None, :, None, None, None]  # (1, C, 1, 1, 1)
    i_idx = i[None, None, :, :, :]  # (1, 1, K*K, out_h, out_w)
    j_idx = j[None, None, :, :, :]  # (1, 1, K*K, out_h, out_w)

    # Broadcast all indices
    n_idx = xp.broadcast_to(n_idx, (N, C, kernel_h * kernel_w, out_h, out_w))
    c_idx = xp.broadcast_to(c_idx, (N, C, kernel_h * kernel_w, out_h, out_w))
    i_idx = xp.broadcast_to(i_idx, (N, C, kernel_h * kernel_w, out_h, out_w))
    j_idx = xp.broadcast_to(j_idx, (N, C, kernel_h * kernel_w, out_h, out_w))

    # Flatten everything for add.at
    n_flat = n_idx.ravel()
    c_flat = c_idx.ravel()
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    col_flat = col.ravel()

    # Single vectorized add.at call
    xp.add.at(img, (n_flat, c_flat, i_flat, j_flat), col_flat)

    # Remove padding
    return img[:, :, pad_h : H_pad - pad_h, pad_w : W_pad - pad_w]


# ==============================
# 装置の移動（Device Transfer）
# ==============================


def _to_device(x, to_gpu_device):
    """to_gpu / to_cpu の本体。装置を移した新しい値を返す（微分できる恒等写像）"""
    if not isinstance(x, NumType):
        x = _auto_convert(x, requires_grad=False)

    on_gpu = cp is not None and isinstance(x._data, cp.ndarray)
    if on_gpu == to_gpu_device:
        return x  # すでにその装置にある

    if to_gpu_device:
        if cp is None:
            raise RuntimeError(
                "CuPy is not installed, so values cannot be moved to the GPU."
                " Install cupy (e.g. pip install cupy-cuda12x) to use nm.to_gpu()"
            )
        data = cp.asarray(x._data)
    else:
        data = as_numpy(x._data)

    result = x._like(data)

    if not (autograd.is_enabled() and x.requires_grad):
        if x.requires_grad or (
            x._backward is _backward_built_under_off and not autograd.is_enabled()
        ):
            result._backward = _backward_built_under_off
        return result

    result.requires_grad = True
    result._prev = (x,)

    def _backward(create_graph=False):
        if result.grad is None:
            return
        if create_graph:
            # 装置を戻すのも恒等写像。グラフはつながったまま
            _accumulate_graph(x, _to_device(result.grad, not to_gpu_device))
            return
        g = (
            as_numpy(result.grad._data)
            if to_gpu_device
            else cp.asarray(result.grad._data)
        )
        if x.grad is None:
            x.grad = x._like(g)
        else:
            x.grad._data = x.grad._data + g

    result._backward = _backward
    return result


def to_gpu(x):
    """
    Move a value to the GPU (CuPy).

    Moving between devices is an identity map, so gradients flow through it and
    are moved back to the device of ``x``.

    Parameters
    ----------
    x : NumType or array_like
        Input value. A value that is already on the GPU is returned as is.

    Returns
    -------
    NumType
        Value of ``type(x)`` holding a CuPy array.

    Raises
    ------
    RuntimeError
        If CuPy is not installed.

    Notes
    -----
    numlib never moves arrays between devices on its own: a new array (from a
    list or from ``nm.randn``) is created on the current device (``nm.cuda``),
    and an array that is passed in stays where it is. Use this function to move
    it explicitly. Combining a CPU value and a GPU value in one operation raises
    ``TypeMismatchError``.

    Examples
    --------
    >>> x = nm.tensor(np.ones((2, 2)))   # CPU
    >>> with nm.cuda.gpu:
    ...     y = nm.tensor([[1.0, 2.0], [3.0, 4.0]])   # GPU
    ...     z = nm.to_gpu(x) * y
    """
    return _to_device(x, True)


def to_cpu(x):
    """
    Move a value to the CPU (NumPy).

    Moving between devices is an identity map, so gradients flow through it and
    are moved back to the device of ``x``.

    Parameters
    ----------
    x : NumType or array_like
        Input value. A value that is already on the CPU is returned as is.

    Returns
    -------
    NumType
        Value of ``type(x)`` holding a NumPy array.

    See Also
    --------
    to_gpu : Move a value to the GPU.
    as_numpy : Get the raw NumPy array out of a value.
    """
    return _to_device(x, False)


# ==============================
# 高階微分（Higher-Order Derivatives）
# ==============================


def grad(y, x, create_graph=False):
    """
    スカラー y の、変数 x についての勾配 ∇ₓy を返す（x.grad には足し込まない）

    Parameters
    ----------
    y : NumType
        スカラー（形が ()）の値
    x : NumType or list of NumType or tuple of NumType
        変数（``requires_grad=True``）。リストかタプルなら、同じ順番のリストを返す
    create_graph : bool, optional
        True なら勾配の計算も計算グラフに残し、戻り値をさらに微分できる
        （高階微分）。False（デフォルト）なら戻り値は定数（``requires_grad=False``）

    Returns
    -------
    NumType or list of NumType
        x と同じ型・同じ形の勾配（``Vector`` の変数には ``Vector`` の勾配）。
        y が x に依存しないときは 0（数学的に微分は 0 で定義されている）

    Raises
    ------
    GradientError
        y がスカラーでない、x が ``requires_grad=True`` でない、x が ``Integer`` /
        ``Boolean``（微分できない型）、y を ``nm.autograd.off`` の中で作った、
        計算グラフが解放済み、``create_graph=True`` を ``nm.autograd.off`` の中で
        使った、``create_graph=True`` で 1 階専用の演算（``make_op``、複素数、
        まだ対応していない演算）が計算グラフに入っている、のどれか

    Notes
    -----
    計算グラフは解放しない（同じ y について何度でも呼べる）。グラフは y への参照が
    なくなったときに消える。

    y を ``nm.autograd.off`` の中で作った場合はエラーになるが、off の中で作った値を
    外で使った場合（``with off: c = f(x)`` のあとの ``2 * c``）は、その値を定数として
    扱ったことになり 0 が返る（``detach()`` と同じ意味）。

    微分できない点（``abs`` の 0、``maximum`` / ``minimum`` の等しい点）は、1 階の
    逆伝播と同じく、ほとんど至る所での微分の決まりを高階でも使う。

    Examples
    --------
    >>> x = nm.real(2.0, requires_grad=True)
    >>> y = x ** 3
    >>> dy = nm.grad(y, x, create_graph=True)    # 3x² = 12
    >>> d2y = nm.grad(dy, x, create_graph=True)  # 6x = 12
    >>> nm.grad(d2y, x)                          # 6
    """
    single = not isinstance(x, (list, tuple))
    xs = [x] if single else list(x)
    for v in xs:
        _check_grad_variable(v, create_graph)
    if not isinstance(y, NumType):
        raise GradientError(
            f"nm.grad needs a NumType output, got {type(y).__name__}",
            hint="Compute y from x with numlib operations",
        )
    if y.shape != ():
        raise GradientError(
            "nm.grad needs a scalar output",
            shape=y.shape,
            hint="The gradient is defined for a scalar y. Sum it first: nm.grad(nm.sum(y), x)",
        )
    if create_graph:
        _check_autograd_for_create_graph()

    grads = _grad_impl(y, xs, None, create_graph)
    return grads[0] if single else grads


def _check_grad_variable(v, create_graph):
    """nm.grad の変数 v が微分できるものかを調べる"""
    if not isinstance(v, NumType):
        raise GradientError(
            f"nm.grad needs NumType variables, got {type(v).__name__}",
            hint="Create the variable with numlib, e.g. nm.vector(data, requires_grad=True)",
        )
    if isinstance(v, (Integer, Boolean)):
        raise GradientError(
            f"{type(v).__name__} is not differentiable",
            hint="Use a real variable, e.g. nm.real(x) or x.astype(float)",
        )
    if not v.requires_grad:
        raise GradientError(
            f"The variable ({type(v).__name__}) does not require gradients",
            hint="Create the variable with requires_grad=True",
        )
    if create_graph and v._data.dtype.kind == "c":
        raise GradientError(
            "Higher-order gradient is not supported for complex numbers",
            hint="Use create_graph=False (first-order gradients of complex numbers work as before)",
        )


def _grad_impl(y, xs, seed, create_graph):
    """
    y の xs についての勾配のリストを返す。seed は y の勾配（None なら 1）

    逆伝播の閉包はすべて .grad に足し込む作りなので、関係するノードの .grad を
    退避して None にしてから逆伝播し、変数の .grad を読んだあとで元に戻す。
    """
    if not y.requires_grad:
        if y._backward is _backward_built_under_off:
            raise GradientError(
                "y has no computation graph", hint=_no_graph_hint(y)
            )
        # y は x に依存しない（定数）ので、微分は 0
        return [zeros_like(v) for v in xs]

    topo = _topological_order(y)
    if create_graph:
        _check_create_graph(topo)  # 解放済みのグラフも調べる
    else:
        _check_graph_not_freed(topo)
    if seed is None:
        seed = ones_like(y)

    saved = [(node, node.grad) for node in topo]
    saved += [(v, v.grad) for v in xs]
    for node, _ in saved:
        node.grad = None
    try:
        _backprop(y, seed, topo, create_graph)
        out = []
        for v in xs:
            g = v.grad
            if g is None:
                # v は y の計算に使われていない
                g = zeros_like(v)
            else:
                tv, tg = type(v), type(g)
                if tg is not tv and tv in _ARRAY_KINDS and tg in _ARRAY_KINDS:
                    g = _construct_or_cast(tv, g) if create_graph else tv(g._data)
                if not create_graph:
                    g = g.detach()
            out.append(g)
    finally:
        for node, g in reversed(saved):
            node.grad = g
    return out


# ==============================
# Export All Public APIs
# ==============================

__all__ = [
    # ==================== Gradient Control ====================
    "autograd",
    "make_op",
    "grad",
    # ==================== GPU Control ====================
    "cuda",
    # ==================== Type ====================
    # Base types
    "NumType",
    "Tensor",
    "Vector",
    "RowVector",
    "Matrix",
    # Scalar types
    "Scalar",
    "Boolean",
    "Integer",
    "Real",
    "Complex",
    # ==================== Factory Functions - Tensors ====================
    "tensor",
    "ten",
    "vector",
    "vec",
    "rowvector",
    "rowvec",
    "matrix",
    "mat",
    # ==================== Factory Functions - Scalars ====================
    "boolean",
    "integer",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "real",
    "real16",
    "real32",
    "real64",
    "real128",
    "cmplx",
    "cmplx64",
    "cmplx128",
    "cmplx256",
    # ==================== Initialization Functions ====================
    # Zeros
    "zeros",
    "zeros_vector",
    "zeros_matrix",
    "zeros_tensor",
    # Ones
    "ones",
    "ones_vector",
    "ones_matrix",
    "ones_tensor",
    # Random
    "rand",
    "randn",
    "randint",
    "random_vector",
    "random_matrix",
    "random_tensor",
    # Like functions
    "ones_like",
    "zeros_like",
    # Random control
    "seed",
    "get_random_state",
    "set_random_state",
    # ==================== Basic Operations ====================
    "add",
    "sub",
    "mul",
    "div",
    "floordiv",
    "mod",
    "pow",
    "neg",
    "abs",
    # ==================== Matrix Operations ====================
    "matmul",
    "dot",
    "bmm",
    "einsum",
    "tensordot",
    # ==================== Reduction Operations ====================
    "sum",
    "mean",
    "amax",
    "amin",
    "maximum",
    "minimum",
    # ==================== Shape Operations ====================
    "reshape",
    "flatten",
    "ravel",
    "transpose",
    # ==================== Indexing Operations ====================
    "get_item",
    # ==================== Mathematical Functions ====================
    # Exponential and logarithmic
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    # Trigonometric functions
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    # Hyperbolic functions
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    # Random masking functions
    "random_mask",
    "random_mask_channel",
    # ==================== Utility Functions ====================
    "one_hot",
    "eye",
    "arange",
    "linspace",
    "concatenate",
    "stack",
    # Broadcasting operations
    "broadcast_to",
    "sum_to",
    # Clipping
    "clip",
    # Dimension operations
    "expand_dims",
    "squeeze",
    # Statistical operations
    "var",
    "std",
    "logsumexp",
    # Conditional operations
    "where",
    # Tensor manipulation
    "split",
    "tile",
    # ==================== Helper Functions ====================
    "get_array_module",
    "as_numpy",
    "to_gpu",
    "to_cpu",
    "as_cupy",
    # ==================== im2col / col2im ====================
    "im2col",
    "col2im",
    # =====================Exception =====================
    "CastError",
    # =================== For Testing ===================
    "_auto_convert",
    "_create_result",
    "_is_np_bool",
    "_is_np_int",
    "_is_np_float",
    "_is_np_complex",
    "_auto_scalar",
    "np",
    "cp",
]
