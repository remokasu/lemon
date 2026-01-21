"""
Save/Load - Basic / 保存・読み込み - 基本
========================================

Save model to ONNX, load, and verify.
ONNXでモデル保存、読み込み、検証。
"""

import lemon as lm
import numpy as np
import os
import tempfile

# Create model / モデル作成
model = lm.Sequential(lm.Linear(10, 20), lm.Relu(), lm.Linear(20, 5))

# Save to ONNX / ONNXに保存
output_dir = tempfile.mkdtemp()
output_path = os.path.join(output_dir, "saved_model.onnx")
lm.export_model(model, output_path, sample_input=lm.randn(1, 10), verbose=True)
print(f"Saved to {output_path}\n")

# Load model / モデル読み込み
loaded_model = lm.Sequential(lm.Linear(10, 20), lm.Relu(), lm.Linear(20, 5))
lm.load_model(loaded_model, output_path, verbose=True)
print()

# Verify / 検証
lm.train.off()
test_input = lm.randn(3, 10)

output_original = model(test_input)
output_loaded = loaded_model(test_input)

diff = lm.as_numpy(output_original - output_loaded)
max_diff = np.abs(diff).max()

if max_diff < 1e-5:
    print(f"Model loaded successfully! Max difference: {max_diff:.2e}")
else:
    print(f"Warning: outputs differ. Max difference: {max_diff:.2e}")
