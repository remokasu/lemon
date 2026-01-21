"""
Autoencoder の基本例
====================

MNIST 画像を 784次元 → 32次元 → 784次元 で圧縮・復元する。
"""

import lemon as lm
import numpy as np
import matplotlib.pyplot as plt


# Data (flatten=True で 784次元のベクトルとして取得)
train_dataset = lm.datasets.MNIST(root="./data", train=True, download=True, flatten=True)
train_set = lm.AutoencoderDataset(lm.Subset(train_dataset, list(range(5000))))


# Model: 784 -> 128 -> 32 -> 128 -> 784
model = lm.Sequential(
    lm.Linear(784, 128),
    lm.Relu(),
    lm.Linear(128, 32),      # Bottleneck
    lm.Relu(),
    lm.Linear(32, 128),
    lm.Relu(),
    lm.Linear(128, 784),
    lm.Sigmoid()
)


# Training
trainer = lm.Trainer(
    model=model,
    dataset=train_set,
    loss=lm.MSELoss(),
    optimizer=lm.Adam(model.parameters(), lr=0.001),
    batch_size=128,
    metrics=[],
)

trainer.fit(epochs=10)


# Test: 復元結果を確認
lm.train.disable()

test_dataset = lm.datasets.MNIST(root="./data", train=False, flatten=True)
test_images = [test_dataset[i][0] for i in range(5)]
test_batch = lm.tensor(np.stack(test_images, axis=0))

reconstructed = model(test_batch)
error = lm.mean((test_batch - reconstructed) ** 2)


# 画像で結果を表示
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i in range(5):
    # 元画像
    axes[0, i].imshow(test_batch[i]._data.reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')

    # 復元画像
    axes[1, i].imshow(reconstructed[i]._data.reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original', size=12)
axes[1, 0].set_ylabel('Reconstructed', size=12)

plt.tight_layout()
plt.savefig('./example/autoencoder/autoencoder_result.png', dpi=100, bbox_inches='tight')
print("Saved reconstructed images to './example/autoencoder/autoencoder_result.png'")

