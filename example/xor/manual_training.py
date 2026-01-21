"""
XOR - Manual Training Loop / XOR - 手動訓練ループ
================================================

Low-level API without Trainer.
Trainerを使わない低レベルAPI。

Related: supervised_dataset.py, csv_dataset.py
"""

import lemon as lm

# Dataset / データセット
dataset = lm.SupervisedDataSet(2, 1)
dataset.add_sample([0, 0], [0])
dataset.add_sample([0, 1], [1])
dataset.add_sample([1, 0], [1])
dataset.add_sample([1, 1], [0])

loader = lm.DataLoader(dataset, batch_size=4)

# Network parameters / ネットワークパラメータ
W1 = lm.rand(2, 8, low=-0.1, high=0.1)
b1 = lm.zeros(8)
W2 = lm.rand(8, 1, low=-0.1, high=0.1)
b2 = lm.zeros(1)


def network(x):
    h = lm.relu(x @ W1 + b1)
    return lm.sigmoid(h @ W2 + b2)


# Training / 訓練
learning_rate = 0.5
epochs = 1000

for epoch in range(epochs):
    epoch_loss = 0

    for X_batch, y_batch in loader:
        output = network(X_batch)
        loss = lm.mean_squared_error(output, y_batch)
        epoch_loss += loss.item()

        loss.backward()

        with lm.autograd.off:
            W1 -= learning_rate * W1.grad
            b1 -= learning_rate * b1.grad
            W2 -= learning_rate * W2.grad
            b2 -= learning_rate * b2.grad

        W1.grad = b1.grad = W2.grad = b2.grad = None

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {epoch_loss / len(loader):.6f}")

# Evaluation / 評価
print("\nPredictions:")
with lm.autograd.off:
    for i in range(len(dataset)):
        x, y = dataset[i]
        output = network(x.reshape(1, -1))
        predicted = 1 if output.item() > 0.5 else 0
        x_data = x._data if hasattr(x, "_data") else x
        y_val = int(y[0].item()) if hasattr(y[0], "item") else int(y[0])
        print(
            f"Input: {x_data}, Target: {y_val}, Predicted: {predicted}, Output: {output.item():.4f}"
        )
