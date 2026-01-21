"""
VGG16 で CIFAR-10 分類
=======================

VGG16 を使って CIFAR-10 データセットを分類する例。
"""

import lemon as lm


# CUDAを有効化
lm.cuda.enable_if_available()


# VGG設定
VGG_CONFIGS = {
    'vgg11': [1, 1, 2, 2, 2],
    'vgg13': [2, 2, 2, 2, 2],
    'vgg16': [2, 2, 3, 3, 3],
    'vgg19': [2, 2, 4, 4, 4],
}


def make_vgg_layers(config, in_channels=3):
    """VGGの特徴抽出レイヤーを作成"""
    layers = []
    channels = [64, 128, 256, 512, 512]

    for num_convs, out_channels in zip(config, channels):
        for _ in range(num_convs):
            layers.append(lm.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(lm.Relu())
            in_channels = out_channels
        layers.append(lm.MaxPool2d(kernel_size=2, stride=2))

    return layers


class VGG(lm.Module):
    """VGGモデル"""

    def __init__(self, config_name='vgg16', num_classes=10, in_channels=3):
        super().__init__()
        config = VGG_CONFIGS[config_name]

        # 特徴抽出層
        self.features = lm.Sequential(*make_vgg_layers(config, in_channels))

        # 分類層 (CIFAR-10用に簡略化)
        self.classifier = lm.Sequential(
            lm.AdaptiveAvgPool2d((1, 1)),
            lm.Flatten(),
            lm.Linear(512, 512),
            lm.Relu(),
            lm.Dropout(0.5),
            lm.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Data / データ
train_dataset = lm.datasets.CIFAR10(root="./data", train=True, download=True)
test_dataset = lm.datasets.CIFAR10(root="./data", train=False)

# サンプル数を制限 (高速化のため)
train_set = lm.Subset(train_dataset, list(range(1000)))
test_set = lm.Subset(test_dataset, list(range(200)))

# Model
model = VGG('vgg16', num_classes=10)

# Training / 訓練
trainer = lm.Trainer(
    model=model,
    dataset=train_set,
    loss=lm.CrossEntropyLoss(),
    optimizer=lm.Adam(model.parameters(), lr=0.001),
    batch_size=32,
    metrics=[lm.Accuracy()],
)

trainer.fit(epochs=5)

# Test evaluation / テスト評価
lm.train.disable()
correct = 0
total = 0

for x, y_true in test_set:
    x = lm.expand_dims(x, axis=0)  # (3, 32, 32) → (1, 3, 32, 32)
    y_pred = model(x)
    pred = y_pred.argmax(axis=1)._data[0]
    correct += int(pred == y_true)
    total += 1

test_acc = 100.0 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")
