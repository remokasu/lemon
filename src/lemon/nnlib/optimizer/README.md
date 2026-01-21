# オプティマイザ (Optimizers)

## 利用可能なオプティマイザ

- `SGD` - 確率的勾配降下法
- `Adam` - Adaptive Moment Estimation
- `AdamW` - Adam with Weight Decay
- `RMSprop` - Root Mean Square Propagation
- `Adagrad` - Adaptive Gradient

## 使用例

```python
from src.lemon.nnlib.optimizer import Adam

# オプティマイザの初期化
optimizer = Adam(model.parameters(), learning_rate=0.001)

# 学習ループ
for epoch in range(num_epochs):
    predictions = model.forward(x)
    loss = criterion.forward(predictions, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
