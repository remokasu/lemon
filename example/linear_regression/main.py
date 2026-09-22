"""
Linear Regression / 線形回帰
============================

Fit y = wx + b to data from y = 2x + 3.
y = 2x + 3 のデータに y = wx + b をフィット。
"""

import lemon as lm

# Data: y = 2x + 3 + noise / データ: y = 2x + 3 + ノイズ
lm.seed(42)
X = lm.rand(50, 1, low=0, high=1)
# スカラーは全成分に足せないので、定数ベクトル 3·1 を明示する
# A scalar cannot be added to a vector; write the constant vector 3·1 explicitly
one = lm.ones_like(X)
y = 2 * X + 3 * one + lm.rand(50, 1, low=-0.1, high=0.1)

# Parameters / パラメータ
w = lm.Real(0.0, requires_grad=True)
b = lm.Real(0.0, requires_grad=True)

# Training / 訓練
learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    y_pred = w * X + b * one  # y = w x + b·1
    loss = lm.mean((y_pred - y) ** 2)

    loss.backward()

    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

    w.zero_grad()
    b.zero_grad()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1:3d} | Loss: {loss.item():.6f} | w: {w.item():.4f} | b: {b.item():.4f}"
        )

print(f"\nFinal: w={w.item():.4f}, b={b.item():.4f}")
print(f"True:  w=2.0000, b=3.0000")
