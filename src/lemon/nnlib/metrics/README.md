# メトリクス (Metrics)

## 利用可能なメトリクス

### 分類メトリクス

- `Accuracy` - 多クラス分類精度（argmax方式）
- `BinaryAccuracy` - 二値分類精度（閾値0.5）
- `TopKAccuracy` - Top-K精度

### 回帰メトリクス

- `MAE` - Mean Absolute Error（平均絶対誤差）
- `MSE` - Mean Squared Error（平均二乗誤差）
- `RMSE` - Root Mean Squared Error（二乗平均平方根誤差）

## 使用例

### 分類メトリクス

```python
import lemon.nnlib as nl
import lemon.trainlib as tl

# Accuracy（多クラス分類）
model = nl.Sequential([
    nl.Linear(784, 128),
    nl.Relu(),
    nl.Linear(128, 10)
])

trainer = tl.Trainer(
    model=model,
    dataset=train_dataset,
    loss=nl.CrossEntropyLoss(),
    metrics=[nl.Accuracy()]  # 0-100%で返る
)

history = trainer.fit(epochs=10)
```

### 回帰メトリクス

```python
# 複数のメトリクスを指定
trainer = tl.Trainer(
    model=model,
    dataset=train_dataset,
    loss=nl.MSELoss(),
    metrics=[nl.MAE(), nl.MSE(), nl.RMSE()]
)

history = trainer.fit(epochs=50)
```

### Top-K Accuracy

```python
# ImageNetなどの大規模分類で使用
trainer = tl.Trainer(
    model=model,
    dataset=imagenet_dataset,
    loss=nl.CrossEntropyLoss(),
    metrics=[
        nl.Accuracy(),           # Top-1
        nl.TopKAccuracy(k=5)     # Top-5
    ]
)
```

## カスタムメトリクスの作成

```python
import lemon.numlib as nm
from lemon.nnlib.metrics import Metric

class MyCustomMetric(Metric):
    def __call__(self, y_pred, y_true):
        # カスタムロジック
        result = nm.mean(nm.abs(y_pred - y_true))
        return float(result.item())

    def name(self):
        return "custom_metric"

# 使用
trainer = tl.Trainer(
    model=model,
    dataset=dataset,
    metrics=[MyCustomMetric()]
)
```

## メトリクスの特徴

### 戻り値の形式

- **Accuracyメトリクス**: 0-100のパーセンテージ（`Accuracy`, `BinaryAccuracy`, `TopKAccuracy`）
- **回帰メトリクス**: 実数値（`MAE`, `MSE`, `RMSE`）

### 内部処理

すべてのメトリクスは以下の処理を行います:

1. 予測値と正解値を受け取る
2. 評価指標を計算
3. Python float値として返す（Tensorではなく）

これにより、メトリクスは自動微分グラフから独立し、メモリ効率が向上します。
