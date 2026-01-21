import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter
from lemon.nnlib.init import uniform_
import math


class LSTMCell(Module):
    """
    LSTM (Long Short-Term Memory) セル

    単一タイムステップのLSTM計算を実行する。
    入力ゲート、忘却ゲート、セルゲート、出力ゲートの4つのゲートを持つ。

    Parameters
    ----------
    input_size : int
        入力特徴量の次元数
    hidden_size : int
        隠れ状態とセル状態の次元数
    bias : bool, optional
        バイアス項を使用するか (デフォルト: True)

    Attributes
    ----------
    weight_ih : Parameter
        入力から隠れ状態への重み行列 (input_size, 4*hidden_size)
        4つのゲート（input, forget, cell, output）の重みを連結
    weight_hh : Parameter
        隠れ状態から隠れ状態への重み行列 (hidden_size, 4*hidden_size)
    bias_ih : Parameter
        入力側のバイアス (4*hidden_size,)
    bias_hh : Parameter
        隠れ状態側のバイアス (4*hidden_size,)

    Examples
    --------
    >>> import lemon.numlib as nm
    >>> from lemon.nnlib.layer.rnn import LSTMCell
    >>>
    >>> # LSTMセルの作成
    >>> cell = LSTMCell(input_size=10, hidden_size=20)
    >>>
    >>> # 入力、隠れ状態、セル状態
    >>> x = nm.randn(3, 10)   # (batch, input_size)
    >>> h = nm.randn(3, 20)   # (batch, hidden_size)
    >>> c = nm.randn(3, 20)   # (batch, hidden_size)
    >>>
    >>> # 1タイムステップの計算
    >>> h_next, c_next = cell(x, (h, c))
    >>> h_next.shape
    (3, 20)
    >>> c_next.shape
    (3, 20)

    Notes
    -----
    LSTMの計算式:
        i_t = σ(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  # 入力ゲート
        f_t = σ(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  # 忘却ゲート
        g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)  # セルゲート
        o_t = σ(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  # 出力ゲート
        c_t = f_t * c_{t-1} + i_t * g_t  # セル状態更新
        h_t = o_t * tanh(c_t)  # 隠れ状態更新

    実装では効率化のため、4つのゲートの重みを連結して一度に計算。
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        # 重み行列: 4つのゲート（input, forget, cell, output）を連結
        self.weight_ih = Parameter(nm.zeros(input_size, 4 * hidden_size))
        self.weight_hh = Parameter(nm.zeros(hidden_size, 4 * hidden_size))

        if bias:
            self.bias_ih = Parameter(nm.zeros(4 * hidden_size))
            self.bias_hh = Parameter(nm.zeros(4 * hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None

        self._reset_parameters()

    def _reset_parameters(self):
        """パラメータを初期化"""
        # PyTorchと同じ初期化方法: uniform(-sqrt(k), sqrt(k)), k = 1/hidden_size
        stdv = 1.0 / math.sqrt(self.hidden_size)

        uniform_(self.weight_ih, -stdv, stdv)
        uniform_(self.weight_hh, -stdv, stdv)

        if self.use_bias:
            uniform_(self.bias_ih, -stdv, stdv)
            uniform_(self.bias_hh, -stdv, stdv)

            # 忘却ゲートのバイアスを1.0に初期化（勾配消失を防ぐ）
            # bias_ih[hidden_size:2*hidden_size] が忘却ゲートのバイアス
            self.bias_ih.data._data[self.hidden_size:2*self.hidden_size] = 1.0
            self.bias_hh.data._data[self.hidden_size:2*self.hidden_size] = 1.0

    def forward(self, x, hx=None):
        """
        順伝播

        Parameters
        ----------
        x : Tensor
            入力テンソル (batch, input_size)
        hx : tuple of Tensor, optional
            (h_0, c_0) のタプル。それぞれ (batch, hidden_size)
            Noneの場合、ゼロで初期化される

        Returns
        -------
        h_next : Tensor
            次の隠れ状態 (batch, hidden_size)
        c_next : Tensor
            次のセル状態 (batch, hidden_size)
        """
        if hx is None:
            batch_size = x.shape[0]
            h = nm.zeros(batch_size, self.hidden_size)
            c = nm.zeros(batch_size, self.hidden_size)
        else:
            h, c = hx

        # 線形変換: x @ W_ih + h @ W_hh + bias
        gates = x @ self.weight_ih.data + h @ self.weight_hh.data

        if self.use_bias:
            gates = gates + self.bias_ih.data + self.bias_hh.data

        # ゲートを分割: (batch, 4*hidden_size) -> 4 x (batch, hidden_size)
        # gates[:, 0:h] = input gate
        # gates[:, h:2h] = forget gate
        # gates[:, 2h:3h] = cell gate
        # gates[:, 3h:4h] = output gate
        i = gates[:, 0:self.hidden_size]
        f = gates[:, self.hidden_size:2*self.hidden_size]
        g = gates[:, 2*self.hidden_size:3*self.hidden_size]
        o = gates[:, 3*self.hidden_size:4*self.hidden_size]

        # 活性化関数を適用
        i = 1 / (1 + nm.exp(-i))  # sigmoid(i)
        f = 1 / (1 + nm.exp(-f))  # sigmoid(f)
        g = nm.tanh(g)             # tanh(g)
        o = 1 / (1 + nm.exp(-o))  # sigmoid(o)

        # セル状態と隠れ状態を更新
        c_next = f * c + i * g
        h_next = o * nm.tanh(c_next)

        return h_next, c_next

    def __repr__(self):
        return f"LSTMCell({self.input_size}, {self.hidden_size})"
