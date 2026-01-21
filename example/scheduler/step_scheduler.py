"""
Scheduler - StepScheduler / スケジューラ - StepScheduler
=======================================================

Decay learning rate by factor every N epochs.
Nエポックごとに学習率を係数倍で減衰。
"""

import lemon as lm

model = lm.Sequential(lm.Linear(10, 10))
optimizer = lm.Adam(model.parameters(), lr=0.1)

# Decay by 0.5 every 3 epochs / 3エポックごとに0.5倍
scheduler = lm.StepScheduler(optimizer, param_name="lr", step_size=3, gamma=0.5)

print("StepScheduler (lr decay every 3 epochs):")
for epoch in range(10):
    scheduler.step()
    print(f"  Epoch {epoch + 1}: lr = {optimizer.lr:.4f}")
