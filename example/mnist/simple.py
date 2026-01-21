"""
MNIST - Simple / MNIST - シンプル
================================

Basic 2-layer network (784 -> 128 -> 10).
基本的な2層ネットワーク (784 -> 128 -> 10)。
"""

import lemon as lm

# Data / データ
train_dataset = lm.datasets.MNIST(root="./data", train=True, download=True)
test_dataset = lm.datasets.MNIST(root="./data", train=False)

train_set, val_set = lm.random_split(train_dataset, [50000, 10000])

train_loader = lm.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = lm.DataLoader(val_set, batch_size=1000, shuffle=False)
test_loader = lm.DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_dataset)}")

# Model: 784 -> 128 -> 10 / モデル: 784 -> 128 -> 10
model = lm.Sequential(lm.Flatten(), lm.Linear(784, 128), lm.Relu(), lm.Linear(128, 10))

# Training / 訓練
trainer = lm.Trainer(
    model=model,
    dataset=train_dataset,
    loss=lm.CrossEntropyLoss(),
    optimizer=lm.Adam(model.parameters(), lr=0.001),
    batch_size=64,
    validation_split=0.1667,  # 10000/60000
    metrics=[lm.Accuracy()],
)

trainer.fit(epochs=5)

# Test evaluation / テスト評価
lm.train.disable()
correct = 0
total = 0
for x, t in test_loader:
    x = x.reshape(x.shape[0], -1)
    y = model(x)
    pred = y.argmax(axis=1)
    correct += lm.sum(pred == t).item()
    total += len(t)

test_acc = 100.0 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")
