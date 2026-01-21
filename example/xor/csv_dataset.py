"""
XOR - CSVDataset / XOR - CSVDataset使用
======================================

Load data from CSV file.
CSVファイルからデータ読み込み。

Related: supervised_dataset.py, manual_training.py
"""

from pathlib import Path


import lemon as lm

import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dataset from CSV / CSVからデータセット
dataset = lm.datasets.CSVDataset(csv_file=str(Path(__file__).parent / "xor_data.csv"))

# Model / モデル
model = lm.Sequential(lm.Linear(2, 8), lm.Relu(), lm.Linear(8, 1), lm.Sigmoid())

# Training / 訓練
trainer = lm.Trainer(
    model=model,
    dataset=dataset,
    loss=lm.MSELoss(),
    optimizer=lm.SGD(model.parameters(), lr=0.5),
    metrics=[lm.BinaryAccuracy()],
    batch_size=4,
    validation_split=0.0,
)

trainer.fit(epochs=1000)

# Evaluation / 評価
lm.train.disable()
print("\nPredictions:")
for i in range(len(dataset)):
    x, y = dataset[i]
    output = trainer.predict(x.reshape(1, -1))
    predicted = 1 if output.item() > 0.5 else 0
    target = int(y.item()) if y.shape == () else int(y[0].item())
    print(
        f"Input: {x}, Target: {target}, Predicted: {predicted}, Output: {output.item():.4f}"
    )
