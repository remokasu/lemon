"""
ResNet18 で CIFAR-10 分類
=========================

ResNet18 を使って CIFAR-10 データセットを分類する例。
"""

import lemon as lm


# CUDAを有効化
# lm.cuda.enable_if_available()


class BasicBlock(lm.Module):
    """ResNetのBasicBlock"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = lm.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = lm.BatchNorm2d(out_channels)
        self.conv2 = lm.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = lm.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = lm.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = lm.relu(out)
        return out


class ResNet(lm.Module):
    """ResNetモデル"""

    def __init__(self, layers, num_classes=10, in_channels=3):
        super().__init__()
        self.in_channels = 64

        # CIFAR-10用に最初の層を簡略化
        self.conv1 = lm.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = lm.BatchNorm2d(64)

        # Residualブロック
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        # 分類ヘッド
        self.avgpool = lm.AdaptiveAvgPool2d((1, 1))
        self.fc = lm.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = lm.Sequential(
                lm.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                lm.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return lm.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = lm.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = lm.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        return x


def ResNet18(num_classes=10):
    return ResNet([2, 2, 2, 2], num_classes)


# Data / データ
train_dataset = lm.datasets.CIFAR10(root="./data", train=True, download=True)
test_dataset = lm.datasets.CIFAR10(root="./data", train=False)

# サンプル数を制限 (高速化のため)
train_set = lm.Subset(train_dataset, list(range(1000)))
test_set = lm.Subset(test_dataset, list(range(200)))

# Model
model = ResNet18(num_classes=10)

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
