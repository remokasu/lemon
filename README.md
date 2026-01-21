# Lemon

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange.svg)](https://numpy.org/)

数値計算/テンソル/自動微分/ニューラルネットワークライブラリ

GPU対応、ONNX対応

## インストール

```bash
pip install -e .
```

## 依存パッケージ

### 必須
- `numpy>=1.20.0`

### オプション

**GPU対応** (`cupy>=9.0.0`):
```bash
pip install -e ".[gpu]"
```

**可視化** (`matplotlib>=3.0.0`, `scikit-learn>=0.24.0`):
```bash
pip install -e ".[plot]"
```

**ONNX対応** (`onnx>=1.10.0`, `onnxruntime>=1.10.0`):
```bash
pip install -e ".[onnx]"
```

**全てインストール**:
```bash
pip install -e ".[all]"
```

## 機能

### レイヤー
`Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `GlobalAveragePooling2d`, `Dropout`, `Dropout2d`, `Flatten`, `BatchNorm1d`, `BatchNorm2d`

### RNNレイヤー
`RNN`, `LSTM`, `GRU`, `RNNCell`, `LSTMCell`, `GRUCell`

### 活性化関数
`Relu`, `Gelu`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyRelu`, `Elu`, `Selu`, `Celu`, `Softplus`, `Softsign`, `HardSigmoid`, `HardSwish`, `Mish`, `ThresholdedRelu`, `PRelu`

### 損失関数
`MSELoss`, `CrossEntropyLoss`, `BCELoss`

### オプティマイザ
`SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`

### スケジューラ
`StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`

### メトリクス
`Accuracy`, `BinaryAccuracy`, `TopKAccuracy`, `MAE`, `MSE`, `RMSE`

### 組み込みデータセット
`MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100`,  `CSVDataset`

