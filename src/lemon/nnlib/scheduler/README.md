# Scheduler API 仕様書

## 概要

Schedulerは訓練中にoptimizerのハイパーパラメータを動的に調整する。

---

## Scheduler (基底クラス)

### コンストラクタ

```python
Scheduler(optimizer, param_name, last_epoch=-1)
```

**パラメータ:**

| 名前 | 型 | 説明 |
|------|-----|------|
| `optimizer` | Optimizer | スケジュール対象のoptimizer |
| `param_name` | str | スケジュールするパラメータ名 |
| `last_epoch` | int | 最後のエポック番号（デフォルト: -1） |

### メソッド

#### `step(metric=None)`

スケジューラを1ステップ進める。

**パラメータ:**
- `metric` (float, optional): メトリクス値（ReduceOnPlateauScheduler系で必要）

#### `get_value()`

現在のエポックでのパラメータ値を計算。サブクラスで実装。

**戻り値:** float

#### `get_last_value()`

最後に設定されたパラメータ値を取得。

**戻り値:** float

#### `state_dict()`

スケジューラの状態を辞書で返す。

**戻り値:** dict

#### `load_state_dict(state_dict)`

スケジューラの状態を復元。

**パラメータ:**
- `state_dict` (dict): 状態辞書

---

## StepScheduler

段階的にパラメータを減衰。

### コンストラクタ

```python
StepScheduler(optimizer, param_name, step_size, gamma=0.1, last_epoch=-1)
```

**パラメータ:**

| 名前 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `optimizer` | Optimizer | - | スケジュール対象 |
| `param_name` | str | - | パラメータ名 |
| `step_size` | int | - | 減衰間隔（エポック数） |
| `gamma` | float | 0.1 | 減衰率 |
| `last_epoch` | int | -1 | 最後のエポック番号 |

### 数式

```
param = base_param * (gamma ** (epoch // step_size))
```

### 例

```python
optimizer = nc.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = nc.StepScheduler(optimizer, param_name='lr', step_size=30, gamma=0.1)

for epoch in range(100):
    train()
    scheduler.step()
```

---

## StepLR

学習率を段階的に減衰。`StepScheduler`の`param_name='lr'`固定版。

### コンストラクタ

```python
StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

**パラメータ:**

| 名前 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `optimizer` | Optimizer | - | スケジュール対象 |
| `step_size` | int | - | 減衰間隔（エポック数） |
| `gamma` | float | 0.1 | 減衰率 |
| `last_epoch` | int | -1 | 最後のエポック番号 |

### 例

```python
optimizer = nc.Adam(model.parameters(), lr=0.01)
scheduler = nc.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train()
    scheduler.step()
```

---

## CosineAnnealingScheduler

コサインカーブでパラメータを減衰。

### コンストラクタ

```python
CosineAnnealingScheduler(optimizer, param_name, T_max, eta_min=0, last_epoch=-1)
```

**パラメータ:**

| 名前 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `optimizer` | Optimizer | - | スケジュール対象 |
| `param_name` | str | - | パラメータ名 |
| `T_max` | int | - | 最大エポック数 |
| `eta_min` | float | 0 | 最小値 |
| `last_epoch` | int | -1 | 最後のエポック番号 |

### 数式

```
param = eta_min + (base_param - eta_min) * (1 + cos(π * epoch / T_max)) / 2
```

### 例

```python
optimizer = nc.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = nc.CosineAnnealingScheduler(optimizer, param_name='momentum', T_max=100, eta_min=0.5)

for epoch in range(100):
    train()
    scheduler.step()
```

---

## CosineAnnealingLR

学習率をコサインカーブで減衰。`CosineAnnealingScheduler`の`param_name='lr'`固定版。

### コンストラクタ

```python
CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
```

**パラメータ:**

| 名前 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `optimizer` | Optimizer | - | スケジュール対象 |
| `T_max` | int | - | 最大エポック数 |
| `eta_min` | float | 0 | 最小学習率 |
| `last_epoch` | int | -1 | 最後のエポック番号 |

### 例

```python
optimizer = nc.Adam(model.parameters(), lr=0.01)
scheduler = nc.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

for epoch in range(100):
    train()
    scheduler.step()
```

---

## ReduceOnPlateauScheduler

メトリクスが改善しない場合にパラメータを減衰。

### コンストラクタ

```python
ReduceOnPlateauScheduler(
    optimizer,
    param_name,
    better,
    factor=0.1,
    patience=10,
    threshold=1e-4,
    threshold_mode='rel',
    cooldown=0,
    min_value=0,
    verbose=False
)
```

**パラメータ:**

| 名前 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `optimizer` | Optimizer | - | スケジュール対象 |
| `param_name` | str | - | パラメータ名 |
| `better` | str | - | 改善方向: `'<'`, `'>'`, `'lower'`, `'higher'`, `'minimize'`, `'maximize'` |
| `factor` | float | 0.1 | 減衰率（< 1.0） |
| `patience` | int | 10 | 許容エポック数 |
| `threshold` | float | 1e-4 | 改善の最小変化量 |
| `threshold_mode` | str | 'rel' | `'rel'`（相対）または`'abs'`（絶対） |
| `cooldown` | int | 0 | 待機期間（エポック数） |
| `min_value` | float | 0 | 下限値 |
| `verbose` | bool | False | メッセージ表示 |

### step()の引数

```python
scheduler.step(metric)
```

**必須:** `metric` (float) - 監視するメトリクス値

### 例

```python
optimizer = nc.Adam(model.parameters(), lr=0.01)
scheduler = nc.ReduceOnPlateauScheduler(
    optimizer, param_name='lr', better='<', patience=5, factor=0.5
)

for epoch in range(100):
    train()
    val_loss = validate()
    scheduler.step(val_loss)
```

---

## ReduceLROnPlateau

学習率をメトリクスベースで減衰。`ReduceOnPlateauScheduler`の`param_name='lr'`固定版。

### コンストラクタ

```python
ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10,
    threshold=1e-4,
    threshold_mode='rel',
    cooldown=0,
    min_lr=0,
    verbose=False
)
```

**パラメータ:**

| 名前 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `optimizer` | Optimizer | - | スケジュール対象 |
| `mode` | str | 'min' | `'min'`（loss用）または`'max'`（accuracy用） |
| `factor` | float | 0.1 | 減衰率（< 1.0） |
| `patience` | int | 10 | 許容エポック数 |
| `threshold` | float | 1e-4 | 改善の最小変化量 |
| `threshold_mode` | str | 'rel' | `'rel'`または`'abs'` |
| `cooldown` | int | 0 | 待機期間 |
| `min_lr` | float | 0 | 学習率の下限 |
| `verbose` | bool | False | メッセージ表示 |

### step()の引数

```python
scheduler.step(metric)
```

**必須:** `metric` (float) - 監視するメトリクス値

### 例

```python
# Loss監視
optimizer = nc.Adam(model.parameters(), lr=0.01)
scheduler = nc.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

for epoch in range(100):
    train()
    val_loss = validate()
    scheduler.step(val_loss)

# Accuracy監視
scheduler = nc.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.1)

for epoch in range(100):
    train()
    val_acc = validate()
    scheduler.step(val_acc)
```

---

## ReduceOnLossPlateau

loss監視専用。`ReduceOnPlateauScheduler`の`better='<'`固定版。

### コンストラクタ

```python
ReduceOnLossPlateau(
    optimizer,
    param_name,
    factor=0.1,
    patience=10,
    threshold=1e-4,
    threshold_mode='rel',
    cooldown=0,
    min_value=0,
    verbose=False
)
```

**パラメータ:** `better`以外は`ReduceOnPlateauScheduler`と同じ

### 例

```python
optimizer = nc.Adam(model.parameters(), lr=0.01)
scheduler = nc.ReduceOnLossPlateau(optimizer, param_name='lr', patience=5, factor=0.5)

for epoch in range(100):
    train()
    val_loss = validate()
    scheduler.step(val_loss)
```

---

## ReduceOnMetricPlateau

metric監視専用。`ReduceOnPlateauScheduler`の`better='>'`固定版。

### コンストラクタ

```python
ReduceOnMetricPlateau(
    optimizer,
    param_name,
    factor=0.1,
    patience=10,
    threshold=1e-4,
    threshold_mode='rel',
    cooldown=0,
    min_value=0,
    verbose=False
)
```

**パラメータ:** `better`以外は`ReduceOnPlateauScheduler`と同じ

### 例

```python
optimizer = nc.Adam(model.parameters(), lr=0.01)
scheduler = nc.ReduceOnMetricPlateau(optimizer, param_name='lr', patience=10, factor=0.5)

for epoch in range(100):
    train()
    val_acc = validate()
    scheduler.step(val_acc)
```

---

## Trainerとの統合

### 単一scheduler

```python
trainer = nc.Trainer(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    schedulers=[scheduler],
    batch_size=64,
    validation_split=0.2
)

history = trainer.fit(epochs=50)
```

### 複数scheduler

```python
trainer = nc.Trainer(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    schedulers=[lr_scheduler, momentum_scheduler],
    batch_size=64,
    validation_split=0.2
)

history = trainer.fit(epochs=50)
```

**注:** メトリクスベースscheduler（`ReduceLROnPlateau`等）は自動的にvalidation lossを受け取る。`validation_split > 0`が必要。

---

## 状態の保存と復元

```python
# 保存
state = scheduler.state_dict()

# 復元
new_scheduler = nc.StepLR(optimizer, step_size=10, gamma=0.1)
new_scheduler.load_state_dict(state)
```

---

## クラス一覧

### 汎用クラス（param_name指定）

| クラス | 説明 |
|--------|------|
| `StepScheduler` | 段階的減衰 |
| `CosineAnnealingScheduler` | コサイン減衰 |
| `ReduceOnPlateauScheduler` | メトリクスベース減衰 |
| `ReduceOnLossPlateau` | loss監視（`better='<'`固定） |
| `ReduceOnMetricPlateau` | metric監視（`better='>'`固定） |

### 学習率専用クラス（param_name='lr'固定）

| クラス | 説明 |
|--------|------|
| `StepLR` | 段階的な学習率減衰 |
| `CosineAnnealingLR` | コサインカーブの学習率減衰 |
| `ReduceLROnPlateau` | メトリクスベースの学習率減衰 |
