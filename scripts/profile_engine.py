"""Where does training time go? Evidence for DESIGN.md §3 (engine bottlenecks).

    uv run python -m scripts.profile_engine

1. Phase breakdown of full training runs (cProfile cumulative times of the
   outermost call of each phase): forward, loss, backward, optimizer, pruner.
2. Micro-benchmark of the Adam update on one 1024x1024 parameter, isolating the
   cost of the per-element bias correction β^t.

Writes results/engine_profile.json. Timings are machine-dependent.
"""
from __future__ import annotations

import cProfile
import json
import pstats
import time
from pathlib import Path

import numpy as np

from train.data import load_digits_dataset
from train.experiment import RunConfig, run

PHASES = {  # phase -> (file suffix, function name) of its outermost call
    "forward": ("nn/layers.py", "forward"),
    "loss": ("engine/functional.py", "softmax_cross_entropy"),
    "backward": ("engine/tensor.py", "backward"),
    "optimizer": ("nn/optim.py", "step"),
    "pruner": ("prune/pruner.py", "on_step"),
}


def phase_breakdown(cfg: RunConfig, data) -> dict:
    run(RunConfig(epochs=1, hidden=cfg.hidden), data)          # warm-up
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    run(cfg, data)
    prof.disable()
    total = time.perf_counter() - t0
    stats = pstats.Stats(prof).stats
    out = {"total_s": total}
    for phase, (suffix, fn) in PHASES.items():
        # outermost call = the one with the largest cumulative time (nested calls are inside it)
        cums = [v[3] for (f, _, name), v in stats.items() if f.endswith(suffix) and name == fn]
        out[phase + "_s"] = max(cums, default=0.0)
    return out


def median_ms(fn, repeats=20, warmup=3) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def adam_microbenchmark() -> dict:
    from nn.optim import Adam
    from nn.parameter import Parameter

    rng = np.random.default_rng(0)
    shape = (1024, 1024)
    p = Parameter(rng.normal(size=shape), prunable=True)
    p.set_mask(rng.random(shape) < 0.1)
    opt = Adam([p])
    g = rng.normal(size=shape)

    def our_step():
        p.grad = np.where(p.mask, g, 0.0)
        opt.step()

    t = rng.integers(1, 500, size=shape)
    p1, p2 = 0.9 ** t, 0.999 ** t
    return {
        "shape": list(shape),
        "our_adam_step_ms": median_ms(our_step),
        "elementwise_pow_beta_t_ms": median_ms(lambda: (0.9 ** t, 0.999 ** t)),
        "running_product_beta_t_ms": median_ms(lambda: (np.multiply(p1, 0.9, out=p1), np.multiply(p2, 0.999, out=p2))),
    }


def main() -> None:
    data = load_digits_dataset(split_seed=0)
    configs = {
        "self-prune 64-128-128-10, batch 64": RunConfig(),
        "dense 64-128-128-10, batch 64": RunConfig(sparsity=0.0),
        "dense 64-128-128-10, batch 512": RunConfig(sparsity=0.0, batch_size=512),
        "dense 64-1024-1024-10, batch 64 (5 epochs)": RunConfig(sparsity=0.0, hidden=[1024, 1024], epochs=5),
    }
    report = {"phases": {}, "adam": None}
    print(f"{'configuration':45s} {'total':>7s} " + " ".join(f"{p:>10s}" for p in PHASES))
    for name, cfg in configs.items():
        r = phase_breakdown(cfg, data)
        report["phases"][name] = r
        cells = " ".join(f"{r[p + '_s']:6.2f}s{100 * r[p + '_s'] / r['total_s']:3.0f}%" for p in PHASES)
        print(f"{name:45s} {r['total_s']:6.2f}s {cells}")

    a = adam_microbenchmark()
    report["adam"] = a
    print(f"\nAdam on one {a['shape'][0]}x{a['shape'][1]} parameter (10% dense):")
    print(f"  full step (as implemented)          {a['our_adam_step_ms']:6.2f} ms")
    print(f"  of which β1^t, β2^t via array power {a['elementwise_pow_beta_t_ms']:6.2f} ms")
    print(f"  same via running products           {a['running_product_beta_t_ms']:6.2f} ms")

    out = Path("results/engine_profile.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
