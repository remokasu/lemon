"""
Sequential MNIST - LSTM による系列分類
======================================

MNIST画像(28×28)を時系列データとして扱い、LSTMで分類する例。
"""

import lemon as lm


# CUDAを有効化
lm.cuda.enable_if_available()


# Data / データ
def reshape_to_sequence(x):
    """(1, 28, 28) → (28, 28)"""
    return x.reshape(28, 28)


train_dataset = lm.datasets.MNIST(root="./data", train=True, download=True, transform=reshape_to_sequence)
test_dataset = lm.datasets.MNIST(root="./data", train=False, transform=reshape_to_sequence)

# サンプル数を制限 (高速化のため)
train_set = lm.Subset(train_dataset, list(range(1000)))
test_set = lm.Subset(test_dataset, list(range(200)))


# Model: LSTM(28, 128) -> Linear(128, 10)
model = lm.Sequential(
    lm.LSTM(input_size=28, hidden_size=128, num_layers=2, dropout=0.2,
            batch_first=True, return_sequences=False),
    lm.Linear(128, 10)
)

# Training / 訓練
trainer = lm.Trainer(
    model=model,
    dataset=train_set,
    loss=lm.CrossEntropyLoss(),
    optimizer=lm.Adam(model.parameters(), lr=0.001),
    batch_size=32,
    metrics=[lm.Accuracy()],
)

trainer.fit(epochs=5)

# Test evaluation / テスト評価
lm.train.disable()
correct = 0
total = 0

for x, y_true in test_set:
    x = lm.expand_dims(x, axis=0)  # (28, 28) → (1, 28, 28)
    y_pred = model(x)
    pred = y_pred.argmax(axis=1)._data[0]
    correct += int(pred == y_true)
    total += 1

test_acc = 100.0 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")
