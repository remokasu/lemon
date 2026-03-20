"""
Binary Neural Network (BNN) - MNIST
=====================================

Comparison of a standard network vs. a Binary Neural Network on MNIST.

BNN replaces floating-point weights with binary values {-1, +1}.
Real-valued weights are kept internally for gradient accumulation,
but only binary weights are used during the forward pass.

Standard model  : Linear weights (float32)
BNN model       : Binary weights {-1, +1} via Sign + STE

Key points
----------
- BNN weights are 32x smaller in theory (1 bit vs 32 bit per weight)
- Multiply-accumulate operations become XNOR + popcount on hardware
- Accuracy trades off slightly against the standard model
- Use Sign activation between binary layers (not Relu)
"""

import lemon as lm
import lemon.numlib as nm

# -----------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------
train_dataset = lm.datasets.MNIST(root="./data", train=True, download=True)
test_dataset  = lm.datasets.MNIST(root="./data", train=False)

test_loader = lm.DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
print()


def evaluate(model, loader):
    lm.train.disable()
    correct = total = 0
    for x, t in loader:
        x = x.reshape(x.shape[0], -1)
        y = model(x)
        pred = y.argmax(axis=1)
        correct += lm.sum(pred == t).item()
        total += len(t)
    return 100.0 * correct / total


# -----------------------------------------------------------------------
# Standard model (baseline)
# -----------------------------------------------------------------------
print("=" * 50)
print("Standard model  (float32 weights)")
print("=" * 50)

standard_model = lm.Sequential(
    lm.Flatten(),
    lm.Linear(784, 256),
    lm.Relu(),
    lm.Linear(256, 128),
    lm.Relu(),
    lm.Linear(128, 10),
)

standard_params = sum(p.data.size for p in standard_model.parameters())
print(f"Parameters: {standard_params:,}")

standard_trainer = lm.Trainer(
    model=standard_model,
    dataset=train_dataset,
    loss=lm.CrossEntropyLoss(),
    optimizer=lm.Adam(standard_model.parameters(), lr=0.001),
    batch_size=64,
    validation_split=0.1667,
    metrics=[lm.Accuracy()],
)

standard_trainer.fit(epochs=5)
standard_acc = evaluate(standard_model, test_loader)
print(f"Test Accuracy (Standard): {standard_acc:.2f}%\n")


# -----------------------------------------------------------------------
# BNN model
# -----------------------------------------------------------------------
print("=" * 50)
print("BNN model  (binary weights {-1, +1})")
print("=" * 50)

# Architecture notes:
#   - Use Sign (not Relu) between binary layers
#   - First layer: standard Linear + BatchNorm (common BNN practice)
#     to normalize input before binarization
#   - Last layer: standard Linear for real-valued logits
bnn_model = lm.Sequential(
    lm.Flatten(),
    lm.Linear(784, 256),           # real-valued input projection
    lm.BatchNorm1d(256),
    lm.Sign(),                      # binarize activations
    lm.BinaryLinear(256, 128),
    lm.BatchNorm1d(128),
    lm.Sign(),
    lm.BinaryLinear(128, 64),
    lm.BatchNorm1d(64),
    lm.Sign(),
    lm.Linear(64, 10),             # real-valued output projection
)

bnn_params = sum(p.data.size for p in bnn_model.parameters())
print(f"Parameters: {bnn_params:,}")

bnn_trainer = lm.Trainer(
    model=bnn_model,
    dataset=train_dataset,
    loss=lm.CrossEntropyLoss(),
    optimizer=lm.Adam(bnn_model.parameters(), lr=0.001),
    batch_size=64,
    validation_split=0.1667,
    metrics=[lm.Accuracy()],
)

bnn_trainer.fit(epochs=5)
bnn_acc = evaluate(bnn_model, test_loader)
print(f"Test Accuracy (BNN): {bnn_acc:.2f}%\n")


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("=" * 50)
print("Summary")
print("=" * 50)
print(f"{'Model':<20} {'Params':>10} {'Test Acc':>10}")
print("-" * 42)
print(f"{'Standard':<20} {standard_params:>10,} {standard_acc:>9.2f}%")
print(f"{'BNN':<20} {bnn_params:>10,} {bnn_acc:>9.2f}%")
print()

binary_weight_params = sum(
    p.data.size
    for name, layer in zip(range(len(bnn_model.modules)), bnn_model.modules)
    if isinstance(layer, lm.BinaryLinear)
    for p in [layer.weight]
)
print(f"Binary weight params: {binary_weight_params:,}")
print(f"Effective bit size  : {binary_weight_params} bits  (vs {binary_weight_params * 32} bits in float32)")
print(f"Compression ratio   : 32x for binary layers")
