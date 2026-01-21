import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter
from lemon.nnlib.init import uniform_
import math


class RNNCell(Module):
    """
    Vanilla RNN セル

    単一タイムステップの単純なRNN計算を実行する。

    Parameters
    ----------
    input_size : int
        入力特徴量の次元数
    hidden_size : int
        隠れ状態の次元数
    bias : bool, optional
        バイアス項を使用するか (デフォルト: True)
    nonlinearity : str, optional
        活性化関数。'tanh' または 'relu' (デフォルト: 'tanh')

    Attributes
    ----------
    weight_ih : Parameter
        入力から隠れ状態への重み行列 (input_size, hidden_size)
    weight_hh : Parameter
        隠れ状態から隠れ状態への重み行列 (hidden_size, hidden_size)
    bias_ih : Parameter
        入力側のバイアス (hidden_size,)
    bias_hh : Parameter
        隠れ状態側のバイアス (hidden_size,)

    Examples
    --------
    >>> import lemon.numlib as nm
    >>> from lemon.nnlib.layer.rnn import RNNCell
    >>>
    >>> # RNNセルの作成
    >>> cell = RNNCell(input_size=10, hidden_size=20)
    >>>
    >>> # 入力と隠れ状態
    >>> x = nm.randn(3, 10)   # (batch, input_size)
    >>> h = nm.randn(3, 20)   # (batch, hidden_size)
    >>>
    >>> # 1タイムステップの計算
    >>> h_next = cell(x, h)
    >>> h_next.shape
    (3, 20)
    >>>
    >>> # ReLU活性化関数を使用
    >>> cell_relu = RNNCell(10, 20, nonlinearity='relu')

    Notes
    -----
    RNNの計算式:
        h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

    または nonlinearity='relu' の場合:
        h_t = relu(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

    Vanilla RNNは勾配消失問題を抱えやすいため、長い系列では
    LSTMやGRUの使用を推奨。
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias
        self.nonlinearity = nonlinearity

        if nonlinearity not in ['tanh', 'relu']:
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got '{nonlinearity}'")

        # 重み行列
        self.weight_ih = Parameter(nm.zeros(input_size, hidden_size))
        self.weight_hh = Parameter(nm.zeros(hidden_size, hidden_size))

        if bias:
            self.bias_ih = Parameter(nm.zeros(hidden_size))
            self.bias_hh = Parameter(nm.zeros(hidden_size))
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

    def forward(self, x, h=None):
        """
        順伝播

        Parameters
        ----------
        x : Tensor
            入力テンソル (batch, input_size)
        h : Tensor, optional
            前の隠れ状態 (batch, hidden_size)
            Noneの場合、ゼロで初期化される

        Returns
        -------
        h_next : Tensor
            次の隠れ状態 (batch, hidden_size)
        """
        if h is None:
            batch_size = x.shape[0]
            h = nm.zeros(batch_size, self.hidden_size)

        # 線形変換: x @ W_ih + h @ W_hh + bias
        output = x @ self.weight_ih.data + h @ self.weight_hh.data

        if self.use_bias:
            output = output + self.bias_ih.data + self.bias_hh.data

        # 活性化関数を適用
        if self.nonlinearity == 'tanh':
            h_next = nm.tanh(output)
        else:  # relu
            h_next = nm.maximum(0, output)

        return h_next

    def __repr__(self):
        return f"RNNCell({self.input_size}, {self.hidden_size}, nonlinearity='{self.nonlinearity}')"
