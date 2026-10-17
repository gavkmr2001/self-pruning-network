"""Part 4 (evidence): sparsity–accuracy Pareto sweep over methods and seeds.

    uv run python -m scripts.pareto_sweep            # ~3-5 min on a laptop CPU
    uv run python -m scripts.pareto_sweep --seeds 3  # quick look

Every run uses train/experiment.py::run (the Part 3 code) with only the method
and target sparsity changed. Seeds change the weight initialisation and batch
order; the data split is fixed (split_seed = 0) and shared by all runs.

Analysis plan -- fixed BEFORE the sweep was run (see README):
  * primary comparison: saliency vs magnitude, both global ranking, no regrowth
    (isolates the importance criterion);
  * primary metric: test accuracy; primary sparsity level: 95%;
  * paired by seed; exact sign-flip permutation test; bootstrap 95% CI;
    Holm correction across all sparsity levels > 0;
  * secondary: test loss (a continuous metric, less quantised than accuracy
    on 359 images), and the full method (saliency + regrowth) vs magnitude.

Outputs (results/part4_pareto/): runs.csv (one row per run, the raw numbers),
summary.csv, comparisons.json, comparisons.md, pareto.png.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train.data import load_digits_dataset
from train.experiment import RunConfig, run
from train.plotting import plt
from train.stats import bootstrap_ci, holm, paired_permutation_test

METHODS = {
    "magnitude-global": dict(criterion="magnitude", scope="global", regrow_fraction=0.0),
    "magnitude-layer": dict(criterion="magnitude", scope="layer", regrow_fraction=0.0),
    "saliency-global": dict(criterion="saliency", scope="global", regrow_fraction=0.0),
    "saliency-global+regrow": dict(criterion="saliency", scope="global", regrow_fraction=0.3),
}
SPARSITIES = [0.0, 0.5, 0.75, 0.9, 0.95, 0.97, 0.98]
COMPARISONS = [  # (A, B, label)
    ("saliency-global", "magnitude-global", "criterion: saliency vs magnitude (global, no regrowth)"),
    ("saliency-global+regrow", "magnitude-global", "full method: saliency+regrowth vs magnitude (global)"),
    ("saliency-global+regrow", "saliency-global", "regrowth: on vs off (saliency, global)"),
    ("magnitude-global", "magnitude-layer", "magnitude: global vs layer-wise"),
]
PRIMARY = ("saliency-global", "magnitude-global", 0.95, "test_acc")
FIELDS = ["method", "target_sparsity", "seed", "achieved_sparsity", "active_weights",
          "test_acc", "test_loss", "val_acc", "val_loss"]


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return {(r["method"], float(r["target_sparsity"]), int(r["seed"])): r for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/part4_pareto")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.csv"
    done = load_existing(runs_path)
    data = load_digits_dataset(split_seed=0)

    # ------------------------------------------------------------------ runs
    new_file = not runs_path.exists()
    with runs_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for sparsity in SPARSITIES:
            # At 0% sparsity no pruning happens, so all methods are the same dense run.
            methods = ["dense"] if sparsity == 0.0 else list(METHODS)
            for method in methods:
                for seed in range(args.seeds):
                    key = (method, sparsity, seed)
                    if key in done:
                        continue
                    kwargs = {} if method == "dense" else METHODS[method]
                    r = run(RunConfig(sparsity=sparsity, seed=seed, **kwargs), data)
                    row = {"method": method, "target_sparsity": sparsity, "seed": seed,
                           "achieved_sparsity": r["achieved_sparsity"], "active_weights": r["active_weights"],
                           "test_acc": r["test_acc"], "test_loss": r["test_loss"],
                           "val_acc": r["final_val_acc"], "val_loss": r["history"][-1]["val_loss"]}
                    writer.writerow(row)
                    f.flush()
                    done[key] = {k: str(v) for k, v in row.items()}
                    print(f"{method:24s} s={sparsity:.2f} seed={seed}: test acc {r['test_acc']:.4f} "
                          f"loss {r['test_loss']:.4f} (achieved {r['achieved_sparsity']:.4f})")

    rows = [r for r in done.values() if int(r["seed"]) < args.seeds]

    def metric(method, sparsity, name):
        m = "dense" if sparsity == 0.0 else method
        vals = sorted(((int(r["seed"]), float(r[name])) for r in rows
                       if r["method"] == m and float(r["target_sparsity"]) == sparsity))
        return np.array([v for _, v in vals])

    # --------------------------------------------------------------- summary
    summary = []
    for method in METHODS:
        for s in SPARSITIES:
            acc, loss = metric(method, s, "test_acc"), metric(method, s, "test_loss")
            ach = metric(method, s, "achieved_sparsity")
            summary.append({"method": method, "target_sparsity": s, "achieved_sparsity_mean": ach.mean(),
                            "n_seeds": len(acc), "test_acc_mean": acc.mean(), "test_acc_std": acc.std(ddof=1),
                            "test_loss_mean": loss.mean(), "test_loss_std": loss.std(ddof=1)})
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # ----------------------------------------------------------- comparisons
    comparisons = []
    for a, b, label in COMPARISONS:
        block = []
        for s in SPARSITIES[1:]:
            entry = {"A": a, "B": b, "label": label, "sparsity": s}
            for name in ("test_acc", "test_loss"):
                d = metric(a, s, name) - metric(b, s, name)
                lo, hi = bootstrap_ci(d)
                entry[name] = {"mean_A": float(metric(a, s, name).mean()), "mean_B": float(metric(b, s, name).mean()),
                               "mean_diff": float(d.mean()), "ci95": [lo, hi],
                               "p_perm": paired_permutation_test(d),
                               "A_better": int(np.sum(d > 0) if name == "test_acc" else np.sum(d < 0)),
                               "ties": int(np.sum(d == 0)), "n": len(d)}
            block.append(entry)
        for name in ("test_acc", "test_loss"):
            for entry, padj in zip(block, holm([e[name]["p_perm"] for e in block])):
                entry[name]["p_holm"] = padj
        comparisons.extend(block)
    (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2))

    lines = [f"# Paired comparisons ({args.seeds} seeds, test set = {len(data.test)} images)\n",
             "diff = A − B. Accuracy in percentage points; positive = A better. Loss: negative = A better.",
             "p = exact paired sign-flip permutation test; p_holm = Holm-corrected across sparsity levels.\n"]
    for a, b, label in COMPARISONS:
        lines.append(f"## {label}\n")
        lines.append("| sparsity | acc A | acc B | Δacc (pts) [95% CI] | A wins/ties/n | p | p_holm | Δloss [95% CI] | p_holm(loss) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for e in (c for c in comparisons if c["A"] == a and c["B"] == b):
            acc, loss = e["test_acc"], e["test_loss"]
            lines.append(
                f"| {e['sparsity']:.2f} | {acc['mean_A']:.4f} | {acc['mean_B']:.4f} | "
                f"{100 * acc['mean_diff']:+.2f} [{100 * acc['ci95'][0]:+.2f}, {100 * acc['ci95'][1]:+.2f}] | "
                f"{acc['A_better']}/{acc['ties']}/{acc['n']} | {acc['p_perm']:.3f} | {acc['p_holm']:.3f} | "
                f"{loss['mean_diff']:+.4f} [{loss['ci95'][0]:+.4f}, {loss['ci95'][1]:+.4f}] | {loss['p_holm']:.3f} |")
        lines.append("")
    pa, pb, ps, pm = PRIMARY
    prim = next(c for c in comparisons if c["A"] == pa and c["B"] == pb and c["sparsity"] == ps)[pm]
    lines.append("## Pre-registered primary comparison\n")
    lines.append(f"{pa} vs {pb} at {ps:.0%} sparsity, {pm}: {100 * prim['mean_A']:.2f}% vs {100 * prim['mean_B']:.2f}%, "
                 f"Δ = {100 * prim['mean_diff']:+.2f} pts, 95% CI [{100 * prim['ci95'][0]:+.2f}, {100 * prim['ci95'][1]:+.2f}], "
                 f"p = {prim['p_perm']:.3f} (Holm {prim['p_holm']:.3f}), A better on {prim['A_better']}/{prim['n']} seeds.")
    (out / "comparisons.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ------------------------------------------------------------------ plot
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(len(SPARSITIES))
    offsets = np.linspace(-0.15, 0.15, len(METHODS))
    for off, method in zip(offsets, METHODS):
        accs = [metric(method, s, "test_acc") for s in SPARSITIES]
        losses = [metric(method, s, "test_loss") for s in SPARSITIES]
        ax_acc.errorbar(xs + off, [a.mean() for a in accs], yerr=[a.std(ddof=1) for a in accs],
                        marker="o", capsize=3, label=method)
        ax_loss.errorbar(xs + off, [l.mean() for l in losses], yerr=[l.std(ddof=1) for l in losses],
                         marker="o", capsize=3, label=method)
    for ax, ylabel in ((ax_acc, "test accuracy"), (ax_loss, "test cross-entropy")):
        ax.set_xticks(xs, [f"{s:.0%}" for s in SPARSITIES])
        ax.set(xlabel="achieved sparsity (fraction of weights removed)", ylabel=ylabel)
        ax.grid(alpha=0.3)
    ax_acc.set_title(f"Sparsity–accuracy trade-off (mean ± std, {args.seeds} seeds)")
    ax_loss.set_title("Test loss (lower is better)")
    ax_acc.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pareto.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote {out}/runs.csv, summary.csv, comparisons.json, comparisons.md, pareto.png")


if __name__ == "__main__":
    main()
