# Installation Guide / インストールガイド

## Quick Install / クイックインストール

### From source (development mode) / ソースから（開発モード）

```bash
# Clone repository / リポジトリをクローン
git clone https://github.com/remokasu/lemon.git
cd lemon

# Install in editable mode / 編集可能モードでインストール
pip install -e .
```

### With optional dependencies / オプション依存関係付き

```bash
# GPU support / GPU対応
pip install -e ".[gpu]"

# Plotting support / プロット機能
pip install -e ".[plot]"

# ONNX export/import / ONNX エクスポート/インポート
pip install -e ".[onnx]"

# All optional dependencies / すべてのオプション
pip install -e ".[all]"

# Development dependencies / 開発用依存関係
pip install -e ".[dev]"
```

## Usage / 使い方

```python
import lemon as lm

# Create model / モデル作成
model = lm.Sequential(
    lm.Linear(784, 128),
    lm.Relu(),
    lm.Linear(128, 10)
)

# Optimizer / オプティマイザ
optimizer = lm.Adam(model.parameters(), lr=0.001)

# Training loop / 訓練ループ
for epoch in range(10):
    y = model(x)
    loss = lm.softmax_cross_entropy(y, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Verify Installation / インストール確認

```bash
python -c "import lemon; print(lemon.__version__)"
```

## Requirements / 必須要件

- Python >= 3.8
- numpy >= 1.20.0

## Optional Requirements / オプション要件

- **GPU**: cupy >= 9.0.0
- **Plotting**: matplotlib >= 3.0.0, scikit-learn >= 0.24.0
- **ONNX**: onnx >= 1.10.0, onnxruntime >= 1.10.0
