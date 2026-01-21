# Examples / サンプル集

This directory contains executable examples demonstrating the features of the library.
ライブラリの機能を実演する実行可能なサンプル集。

## Structure / 構成

```
example/
├── linear_regression/    # Linear regression basics / 線形回帰の基礎
├── xor/                  # XOR classification with different approaches / 異なる手法でのXOR分類
├── mnist/                # MNIST digit classification / MNIST数字分類
├── cifar100/             # CIFAR-100 image classification / CIFAR-100画像分類
├── scheduler/            # Learning rate schedulers / 学習率スケジューラ
└── save_load/            # Model save/load in ONNX format / ONNXフォーマットでのモデル保存・読み込み
```

## Getting Started / はじめに

**Recommended order / 推奨順序:**

1. `linear_regression/main.py` - Gradient descent basics
2. `xor/supervised_dataset.py` - Neural network basics
3. `mnist/simple.py` - Image classification
4. `cifar100/main.py` - Advanced classification

## Examples / サンプル

### linear_regression/

**main.py**
- Fit y = wx + b using gradient descent
- Manual parameter updates
- 勾配降下法でy = wx + bをフィット

### xor/

**supervised_dataset.py**
- Manual data entry with SupervisedDataSet
- Using Trainer for training
- SupervisedDataSetで手動データ入力

**csv_dataset.py**
- Load data from CSV file
- Auto-detect input/output columns
- CSVファイルからデータ読み込み

**manual_training.py**
- Low-level training loop without Trainer
- Manual network definition and parameter updates
- Trainerを使わない低レベル訓練ループ

### mnist/

**simple.py**
- Basic 2-layer network (784 -> 128 -> 10)
- Using Trainer with validation split
- 基本的な2層ネットワーク

**with_scheduler.py**
- Deeper network with dropout
- Learning rate scheduling with StepScheduler
- スケジューラ付きの深いネットワーク

### cifar100/

**main.py**
- 100-class color image classification
- Deep network (3072 -> 1024 -> 512 -> 256 -> 100)
- GPU support
- 100クラスカラー画像分類

### scheduler/

**step_scheduler.py**
- Decay learning rate by factor every N epochs
- Nエポックごとに学習率を減衰

**cosine_annealing.py**
- Smooth cosine decay for momentum
- モーメンタムのスムーズなコサイン減衰

**reduce_on_plateau.py**
- Adaptive learning rate based on validation loss
- 検証損失に基づく適応的な学習率

### save_load/

**basic.py**
- Save model to ONNX format
- Load and verify outputs match
- ONNXフォーマットでモデル保存・検証

## Notes / 注意

- All examples use `import lemon as lm` for consistency
- Trainer-based examples minimize print output
- Each directory is self-contained
- すべてのサンプルは一貫性のため`import lemon as lm`を使用
- Trainer使用例ではprint出力を最小化
- 各ディレクトリは独立
