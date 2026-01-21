import lemon.numlib as nm


class Parameter:
    """
    学習可能なパラメータ

    本質的には numlib の Tensor のラッパー。
    Optimizer がパラメータを認識するためのマーカーとして機能する。

    Parameters
    ----------
    data : array_like or NumType
        パラメータの初期値
    requires_grad : bool, optional
        勾配計算を有効にするか（デフォルト: True）

    Examples
    --------
    >>> import numlib as nm
    >>> w = Parameter(nm.randn(10, 5))
    >>> b = Parameter(nm.zeros(5))
    >>> y = x @ w + b
    """

    def __init__(self, data, requires_grad: bool = True):
        # NumType に変換
        if isinstance(data, nm.NumType):
            self.data = data
        else:
            self.data = nm.tensor(data, requires_grad=requires_grad)

        # requires_grad を強制的に設定
        if requires_grad:
            self.data.requires_grad = True

    def __repr__(self):
        return f"Parameter({self.data})"

    def zero_grad(self):
        self.data.zero_grad()

    @property
    def grad(self):
        return self.data.grad

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getattr__(self, name):
        """その他の属性は data に委譲"""
        return getattr(self.data, name)

    # ========== 算術演算子 ==========
    def __add__(self, other):
        return self.data + (other.data if isinstance(other, Parameter) else other)

    def __radd__(self, other):
        return other + self.data

    def __sub__(self, other):
        return self.data - (other.data if isinstance(other, Parameter) else other)

    def __rsub__(self, other):
        return other - self.data

    def __mul__(self, other):
        return self.data * (other.data if isinstance(other, Parameter) else other)

    def __rmul__(self, other):
        return other * self.data

    def __truediv__(self, other):
        return self.data / (other.data if isinstance(other, Parameter) else other)

    def __rtruediv__(self, other):
        return other / self.data

    def __floordiv__(self, other):
        return self.data // (other.data if isinstance(other, Parameter) else other)

    def __rfloordiv__(self, other):
        return other // self.data

    def __mod__(self, other):
        return self.data % (other.data if isinstance(other, Parameter) else other)

    def __rmod__(self, other):
        return other % self.data

    def __pow__(self, other):
        return self.data ** (other.data if isinstance(other, Parameter) else other)

    def __rpow__(self, other):
        return other ** self.data

    def __matmul__(self, other):
        return self.data @ (other.data if isinstance(other, Parameter) else other)

    def __rmatmul__(self, other):
        return other @ self.data

    def __neg__(self):
        return -self.data

    def __pos__(self):
        return +self.data

    def __abs__(self):
        return abs(self.data)

    # ========== 累算代入演算子 ==========
    def __iadd__(self, other):
        self.data += other.data if isinstance(other, Parameter) else other
        return self

    def __isub__(self, other):
        self.data -= other.data if isinstance(other, Parameter) else other
        return self

    def __imul__(self, other):
        self.data *= other.data if isinstance(other, Parameter) else other
        return self

    def __itruediv__(self, other):
        self.data /= other.data if isinstance(other, Parameter) else other
        return self

    def __ifloordiv__(self, other):
        self.data //= other.data if isinstance(other, Parameter) else other
        return self

    def __imod__(self, other):
        self.data %= other.data if isinstance(other, Parameter) else other
        return self

    def __ipow__(self, other):
        self.data **= other.data if isinstance(other, Parameter) else other
        return self

    # ========== 比較演算子 ==========
    def __eq__(self, other):
        return self.data == (other.data if isinstance(other, Parameter) else other)

    def __ne__(self, other):
        return self.data != (other.data if isinstance(other, Parameter) else other)

    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, Parameter) else other)

    def __le__(self, other):
        return self.data <= (other.data if isinstance(other, Parameter) else other)

    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, Parameter) else other)

    def __ge__(self, other):
        return self.data >= (other.data if isinstance(other, Parameter) else other)

    # ========== インデックス演算子 ==========
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value.data if isinstance(value, Parameter) else value

    # ========== その他 ==========
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        return item in self.data
