"""
XOR - SupervisedDataSet / XOR - SupervisedDataSet使用
====================================================

Manual data entry with SupervisedDataSet.
SupervisedDataSetで手動データ入力。
"""

import lemon as nc

# Dataset / データセット
dataset = nc.SupervisedDataSet(2, 1)
dataset.add_sample([0, 0], [0])
dataset.add_sample([0, 1], [1])
dataset.add_sample([1, 0], [1])
dataset.add_sample([1, 1], [0])

# Model / モデル
model = nc.Sequential(nc.Linear(2, 8), nc.Relu(), nc.Linear(8, 1), nc.Sigmoid())

# Training / 訓練
trainer = nc.Trainer(
    model=model,
    dataset=dataset,
    loss=nc.MSELoss(),
    optimizer=nc.SGD(model.parameters(), lr=0.5),
    metrics=[nc.BinaryAccuracy()],
    batch_size=4,
    validation_split=0.0,
)

trainer.fit(epochs=1000)

# Evaluation / 評価
nc.train.disable()
print("\nPredictions:")
for i in range(len(dataset)):
    x, y = dataset[i]
    output = trainer.predict(x.reshape(1, -1))
    predicted = 1 if output.item() > 0.5 else 0
    print(
        f"Input: {x}, Target: {int(y[0].item())}, Predicted: {predicted}, Output: {output.item():.4f}"
    )
