"""
Backward Speed Benchmark / 逆伝播の速さのベンチマーク
======================================================

Measures one training step (zero_grad -> forward -> loss -> backward -> step)
of two MLPs, to check that first-order backward (create_graph=False) has not
become slower (SPEC-0001 AC-9: within +-5%).

2 つの MLP の学習 1 ステップ（zero_grad → 順伝播 → 損失 → backward → step）を測る。
1 階の逆伝播（create_graph=False）が遅くなっていないかを確かめる（SPEC-0001 AC-9: ±5% 以内）。

- mnist-mlp: Linear(784, 128) -> Relu -> Linear(128, 10), batch 64（代表）
- small-mlp: Linear(32, 32) -> Tanh -> Linear(32, 32) -> Tanh -> Linear(32, 10), batch 8
  （配列が小さく Python のオーバーヘッドが支配的。閉包の分岐のコストが一番見える）

Data is random (nm.seed(0)); nothing is downloaded. Run on CPU with one thread:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python example/benchmark/backward_speed.py --save before.json
    # ... change the code ...
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python example/benchmark/backward_speed.py --compare before.json

Timing: 20 warm-up steps, then 7 rounds of 200 steps. The median of the
per-round mean step time is reported. The time of backward() alone is also
recorded (for investigation only, not used for pass / fail).
"""

import argparse
import json
import statistics
import time

import lemon as lm
import lemon.numlib as nm

WARMUP = 20
ROUNDS = 7
STEPS = 200


def _mnist_mlp():
    model = lm.Sequential(lm.Linear(784, 128), lm.Relu(), lm.Linear(128, 10))
    return model, 64, 784, 10


def _small_mlp():
    model = lm.Sequential(
        lm.Linear(32, 32), lm.Tanh(), lm.Linear(32, 32), lm.Tanh(), lm.Linear(32, 10)
    )
    return model, 8, 32, 10


MODELS = {"mnist-mlp": _mnist_mlp, "small-mlp": _small_mlp}


def measure(build, rounds=ROUNDS, steps=STEPS):
    """1 ステップの時間と backward() だけの時間（秒）を、ラウンドごとの平均の中央値で返す"""
    nm.seed(0)
    model, batch, n_in, n_out = build()
    loss_fn = lm.CrossEntropyLoss()
    optimizer = lm.SGD(model.parameters(), lr=0.01)
    x = nm.randn(batch, n_in, requires_grad=False)
    t = nm.randint(batch, low=0, high=n_out, requires_grad=False)

    def step():
        optimizer.zero_grad()
        loss = loss_fn(model(x), t)
        start = time.perf_counter()
        loss.backward()
        elapsed = time.perf_counter() - start
        optimizer.step()
        return elapsed

    for _ in range(WARMUP):
        step()

    step_means, backward_means = [], []
    for _ in range(rounds):
        backward_total = 0.0
        start = time.perf_counter()
        for _ in range(steps):
            backward_total += step()
        step_means.append((time.perf_counter() - start) / steps)
        backward_means.append(backward_total / steps)
    return {
        "step": statistics.median(step_means),
        "backward": statistics.median(backward_means),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--save", help="結果を JSON に保存する")
    parser.add_argument("--compare", help="保存した JSON（変更前）と比べる")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    nm.cuda.cpu()
    results = {}
    for name, build in MODELS.items():
        results[name] = measure(build, rounds=args.rounds)
        r = results[name]
        print(
            f"{name:10s} step {r['step'] * 1e3:8.3f} ms   "
            f"backward {r['backward'] * 1e3:8.3f} ms"
        )

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved: {args.save}")

    if args.compare:
        with open(args.compare) as f:
            before = json.load(f)
        ok = True
        for name, r in results.items():
            ratio = r["step"] / before[name]["step"]
            passed = 0.95 <= ratio <= 1.05
            ok = ok and passed
            print(
                f"{name:10s} after/before = {ratio:.3f} "
                f"(backward {r['backward'] / before[name]['backward']:.3f})  "
                f"{'PASS' if passed else 'FAIL'}"
            )
        print("AC-9:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
