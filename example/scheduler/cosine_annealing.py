"""
Scheduler - CosineAnnealingScheduler / スケジューラ - CosineAnnealingScheduler
==============================================================================

Smooth cosine decay for learning rate or momentum.
学習率やモーメンタムのスムーズなコサイン減衰。
"""

import lemon as lm

model = lm.Sequential(lm.Linear(10, 10))
optimizer = lm.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Smooth cosine decay / スムーズなコサイン減衰
scheduler = lm.CosineAnnealingScheduler(
    optimizer,
    param_name="momentum",
    T_max=10,
    eta_min=0.5,
)

print("CosineAnnealingScheduler (momentum):")
for epoch in range(10):
    scheduler.step()
    print(f"  Epoch {epoch + 1}: momentum = {optimizer.momentum:.4f}")
