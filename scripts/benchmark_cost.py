"""Part 4 (cost): does the self-pruned model actually do less work?

    uv run python -m scripts.benchmark_cost

Loads results/part3_self_prune/model.npz and reports:
  1. correctness  -- sparse forward == dense forward of the masked model
  2. MACs / bytes -- hardware-independent work and memory
  3. wall-clock   -- dense BLAS vs our CSR kernel vs compacted dense, several batch sizes
  4. scaling      -- CSR kernel time vs density on random masks (is work ∝ nnz?)

BLAS and NumPy are pinned to ONE thread so that the dense baseline and our
single-threaded sparse kernel are compared like for like. Wall-clock numbers are
machine-dependent; MAC and byte counts are exact.
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from prune.sparse import SparseMLP, compact, dense_forward, dense_macs_per_sample  # noqa: E402
from train.data import load_digits_dataset  # noqa: E402
from train.plotting import plt  # noqa: E402


def median_time(fn, repeats: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="results/part3_self_prune/model.npz")
    ap.add_argument("--out", default="results/part4_cost")
    args = ap.parse_args()

    z = np.load(args.model)
    n_layers = sum(1 for k in z.files if k.startswith("W"))
    W = [z[f"W{i}"] for i in range(n_layers)]
    b = [z[f"b{i}"] for i in range(n_layers)]
    M = [z[f"M{i}"].astype(bool) for i in range(n_layers)]
    shapes = [w.shape for w in W]
    data = load_digits_dataset(split_seed=0)
    X, y = data.test.X, data.test.y

    # 1. correctness ------------------------------------------------------------
    sparse = SparseMLP.from_arrays(W, b, M)
    Wc, bc, removed = compact(W, b)
    ref = dense_forward(W, b, X)
    err_csr = float(np.abs(sparse.forward(X) - ref).max())
    err_compact = float(np.abs(dense_forward(Wc, bc, X) - ref).max())
    acc = float((sparse.predict(X) == y).mean())
    assert err_csr < 1e-9 and err_compact < 1e-9, "sparse paths disagree with the dense model"

    # 2. MACs and bytes -----------------------------------------------------------
    dense_macs = dense_macs_per_sample(shapes)
    sparse_macs = sparse.macs_per_sample()
    compact_macs = dense_macs_per_sample([w.shape for w in Wc])
    dense_bytes = sum(w.nbytes + x.nbytes for w, x in zip(W, b))
    compact_bytes = sum(w.nbytes + x.nbytes for w, x in zip(Wc, bc))

    print(f"model: {' -> '.join(str(s[0]) for s in shapes)} -> {shapes[-1][1]}, "
          f"active weights {int(sum(m.sum() for m in M))}/{sum(w.size for w in W)}")
    print(f"correctness: max |sparse - dense| = {err_csr:.2e}, |compact - dense| = {err_compact:.2e}, "
          f"test acc via sparse path = {acc:.4f}")
    print(f"compaction removed hidden units {removed} -> shapes {[w.shape for w in Wc]}")
    print("\nwork per sample (multiply-accumulates) and weight memory:")
    print(f"  dense    {dense_macs:6d} MACs  {dense_bytes:7d} bytes")
    print(f"  CSR      {sparse_macs:6d} MACs  {sparse.nbytes():7d} bytes   "
          f"({dense_macs / sparse_macs:.1f}x fewer MACs, {dense_bytes / sparse.nbytes():.1f}x less memory)")
    print(f"  compact  {compact_macs:6d} MACs  {compact_bytes:7d} bytes   "
          f"({dense_macs / compact_macs:.2f}x fewer MACs)")

    # 3. wall-clock ------------------------------------------------------------------
    timings = []
    print("\nwall-clock per forward pass (median of 50, single-threaded):")
    print(f"  {'batch':>6} | {'dense BLAS':>11} | {'CSR (ours)':>11} | {'compact BLAS':>12} | CSR/dense | compact speed-up")
    for n in (1, 64, 359, 4096):
        Xn = np.tile(X, (int(np.ceil(n / len(X))), 1))[:n]
        t_dense = median_time(lambda: dense_forward(W, b, Xn))
        t_csr = median_time(lambda: sparse.forward(Xn))
        t_comp = median_time(lambda: dense_forward(Wc, bc, Xn))
        timings.append({"batch": n, "dense_s": t_dense, "csr_s": t_csr, "compact_s": t_comp})
        print(f"  {n:6d} | {t_dense * 1e6:9.1f}us | {t_csr * 1e6:9.1f}us | {t_comp * 1e6:10.1f}us | "
              f"{t_csr / t_dense:8.2f}x | {t_dense / t_comp:.2f}x")

    # 4. scaling of our kernel with density ------------------------------------------
    rng = np.random.default_rng(0)
    densities = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
    scaling = []
    Xs = np.tile(X, (3, 1))
    t_blas = median_time(lambda: dense_forward(W, b, Xs), repeats=30)
    print(f"\nCSR kernel time vs density (random masks, batch {len(Xs)}; dense BLAS = {t_blas * 1e6:.0f}us):")
    for d in densities:
        masks = [rng.random(w.shape) < d for w in W]
        Wr = [rng.normal(size=w.shape) * m for w, m in zip(W, masks)]
        sp = SparseMLP.from_arrays(Wr, b, masks)
        t = median_time(lambda: sp.forward(Xs), repeats=30)
        scaling.append({"density": d, "nnz": sp.macs_per_sample(), "csr_s": t})
        print(f"  density {d:4.2f}: nnz {sp.macs_per_sample():6d}  {t * 1e6:9.1f}us  "
              f"({t / t_blas:.1f}x dense BLAS)")
    slope, intercept = np.polyfit([s["nnz"] for s in scaling], [s["csr_s"] for s in scaling], 1)
    pred = slope * np.array([s["nnz"] for s in scaling]) + intercept
    r2 = 1 - np.sum((np.array([s["csr_s"] for s in scaling]) - pred) ** 2) / np.var([s["csr_s"] for s in scaling]) / len(scaling)
    print(f"  linear fit time = {slope * 1e9:.2f} ns/nnz * nnz + {intercept * 1e6:.1f}us  (R^2 = {r2:.4f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "python": platform.python_version(), "numpy": np.__version__, "threads": 1},
        "correctness": {"max_abs_err_csr": err_csr, "max_abs_err_compact": err_compact, "test_acc": acc},
        "macs_per_sample": {"dense": dense_macs, "csr": sparse_macs, "compact": compact_macs},
        "weight_bytes": {"dense": dense_bytes, "csr": sparse.nbytes(), "compact": compact_bytes},
        "compaction_removed_units": removed,
        "compact_shapes": [list(w.shape) for w in Wc],
        "wall_clock": timings,
        "csr_scaling": {"points": scaling, "dense_blas_s": t_blas, "ns_per_nnz": slope * 1e9,
                        "intercept_s": intercept, "r2": r2},
    }
    (out / "cost.json").write_text(json.dumps(report, indent=2))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot([s["nnz"] for s in scaling], [s["csr_s"] * 1e6 for s in scaling], "o-", label="CSR kernel (ours)")
    a1.axhline(t_blas * 1e6, color="grey", linestyle="--", label="dense BLAS (any density)")
    a1.set(xlabel="non-zero weights", ylabel="µs per forward (batch 1077)",
           title="Our sparse kernel: time ∝ non-zeros")
    a1.legend()
    a1.grid(alpha=0.3)
    batches = [t["batch"] for t in timings]
    for key, label in (("dense_s", "dense BLAS"), ("csr_s", "CSR (ours)"), ("compact_s", "compacted dense")):
        a2.plot(batches, [t[key] * 1e6 for t in timings], "o-", label=label)
    a2.set(xscale="log", yscale="log", xlabel="batch size", ylabel="µs per forward",
           title="Self-pruned model (90%): wall-clock")
    a2.legend()
    a2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "cost.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote {out}/cost.json and {out}/cost.png")


if __name__ == "__main__":
    main()
