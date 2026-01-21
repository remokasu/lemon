# レイヤー (Layers)

## 関数版とクラス版の違い

多くのレイヤーには**関数版**（小文字）と**クラス版**（大文字）の2つの形式があります。

### 関数版（例: `linear(x, weight, bias)`）
- パラメータを明示的に渡す必要がある
- 状態を持たない

```python
# パラメータを手動管理する場合
weight = nm.randn(784, 128)
bias = nm.randn(128)
output = linear(x, weight, bias)
```

### クラス版（例: `Linear(in_features, out_features)`）
- パラメータを自動的に初期化・管理
- `Module`を継承し、`Sequential`などで使用可能
- 内部的には関数版を呼び出している

```python
# パラメータを自動管理する場合
layer = Linear(in_features=784, out_features=128)
output = layer(x)
```

### クラス版のみのレイヤー

一部のレイヤー（`Flatten`など）は、パラメータを持たず状態変換のみを行うため、クラス版のみが提供されています。

```python
flatten = Flatten()
output = flatten(x)
```

## 独自のレイヤーを追加する

### 関数版の実装

```python
import src.numlib as nm

def my_layer(x, weight, bias=None):
    """
    独自のレイヤー関数
    """
    output = x @ weight
    if bias is not None:
        output = output + bias
    return output
```

### クラス版の実装（パラメータあり）

```python
from src.lemon.nnlib import Module
from src.lemon.nnlib.parameter import Parameter
import src.numlib as nm

class MyLayer(Module):
    """
    独自のレイヤー
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # パラメータの初期化
        self.weight = Parameter(nm.randn(in_features, out_features))
        self.bias = Parameter(nm.zeros(out_features))

    def forward(self, x):
        return my_layer(x, self.weight.data, self.bias.data)

    def __repr__(self):
        return f"MyLayer(in_features={self.in_features}, out_features={self.out_features})"
```

### クラス版の実装（パラメータなし）

```python
from src.lemon.nnlib import Module
import src.numlib as nm

class MyTransformLayer(Module):
    """
    パラメータを持たないレイヤー
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 独自の処理（例: 2倍にする）
        return x * 2

    def __repr__(self):
        return f"{self.__class__.__name__}()"
```

### 自動微分に対応した独自演算

numlibに定義されていない演算を使う場合は、自動微分に対応させる必要があります。詳細は[activation/README.md](../activation/README.md)の「独自の数学関数を定義する場合」を参照してください。

## 利用可能なレイヤー

### 基本レイヤー
- `linear(x, weight, bias)`, `Linear` - 全結合層（パラメータあり）
- `Flatten` - 平坦化層（パラメータなし、クラス版のみ）

### 畳み込みレイヤー
- `conv_2d(x, weight, bias, ...)`, `Conv2d` - 2D畳み込み層（パラメータあり）

### プーリングレイヤー
- `max_pool_2d(x, ...)`, `MaxPool2d` - 最大プーリング
- `avg_pool_2d(x, ...)`, `AvgPool2d` - 平均プーリング
- `adaptive_avg_pool_2d(x, ...)`, `AdaptiveAvgPool2d` - 適応的平均プーリング
- `global_average_pool_2d(x)`, `GlobalAveragePooling2d` - 大域平均プーリング

### 正規化レイヤー
- `batch_norm_1d(x, ...)`, `BatchNorm1d` - 1Dバッチ正規化（パラメータあり）
- `batch_norm_2d(x, ...)`, `BatchNorm2d` - 2Dバッチ正規化（パラメータあり）

### 正則化レイヤー
- `dropout(x, p, training)`, `Dropout` - ドロップアウト
- `dropout_2d(x, p, training)`, `Dropout2d` - 2Dドロップアウト
