"""
Scheduler - ReduceOnPlateauScheduler / スケジューラ - ReduceOnPlateauScheduler
==============================================================================

Reduce learning rate when validation metric plateaus.
検証メトリクスが停滞したら学習率を削減。
"""

import lemon as lm

model = lm.Sequential(lm.Linear(10, 10))
optimizer = lm.Adam(model.parameters(), lr=0.01)

# Reduce when validation loss plateaus / 検証損失が停滞したら削減
scheduler = lm.ReduceOnPlateauScheduler(
    optimizer,
    param_name="lr",
    better="<",  # Lower is better / 小さいほど良い
    patience=3,
    factor=0.5,
    verbose=True,
)

# Simulated validation losses / シミュレートされた検証損失
val_losses = [1.0, 0.9, 0.8, 0.81, 0.82, 0.83, 0.84, 0.7, 0.71, 0.72]

print("ReduceOnPlateauScheduler:")
for epoch, val_loss in enumerate(val_losses):
    scheduler.step(val_loss)
    print(f"  Epoch {epoch + 1}: val_loss = {val_loss:.4f}, lr = {optimizer.lr:.4f}")
