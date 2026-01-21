"""
CIFAR-100 - 100-class Image Classification / CIFAR-100 - 100クラス画像分類
========================================================================

Deep network for 100-class color images.
100クラスカラー画像用深層ネットワーク。
"""

import lemon as lm

lm.cuda.enable_if_available()
print(f"GPU: {lm.cuda.is_enabled()}")

# Data / データ
transform = lm.Normalize(mean=0.5, std=0.5)
train_data = lm.CIFAR100(
    root="./data", train=True, download=True, transform=transform, flatten=True
)
test_data = lm.CIFAR100(root="./data", train=False, transform=transform, flatten=True)

print(f"Train: {len(train_data)}, Test: {len(test_data)}")


# Wrapper to extract fine labels / 詳細ラベルを抽出するラッパー
class FineLabelDataset(lm.Dataset):
    def __init__(self, cifar100_dataset):
        self.dataset = cifar100_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, (y_fine, y_coarse) = self.dataset[idx]
        return x, y_fine


train_data_fine = FineLabelDataset(train_data)
test_data_fine = FineLabelDataset(test_data)

# Model: 3072 -> 1024 -> 512 -> 256 -> 100
# モデル: 3072 -> 1024 -> 512 -> 256 -> 100
model = lm.Sequential(
    lm.Linear(3072, 1024),
    lm.Relu(),
    lm.Dropout(0.3),
    lm.Linear(1024, 512),
    lm.Relu(),
    lm.Dropout(0.3),
    lm.Linear(512, 256),
    lm.Relu(),
    lm.Dropout(0.2),
    lm.Linear(256, 100),
)

# Training with Trainer / Trainerで訓練
trainer = lm.Trainer(
    model=model,
    dataset=train_data_fine,
    loss=lm.CrossEntropyLoss(),
    optimizer=lm.Adam(model.parameters(), lr=0.001),
    batch_size=128,
    validation_split=0.1,
    metrics=[lm.Accuracy()],
)

trainer.fit(epochs=10)

# Test evaluation / テスト評価
lm.train.disable()
test_loader = lm.DataLoader(test_data_fine, batch_size=1000, shuffle=False)

correct = 0
total = 0
for x, y in test_loader:
    pred = model(x)
    pred_labels = pred.argmax(axis=1)
    correct += lm.sum(pred_labels == y).item()
    total += len(y)

test_acc = 100.0 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")
