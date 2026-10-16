"""Part 3: train an MLP that prunes itself to a target sparsity while it learns.

    uv run python -m scripts.self_prune                      # saliency + regrowth, 90% sparsity
    uv run python -m scripts.self_prune --criterion magnitude --regrow 0 --out results/part3_magnitude

Writes <out>/{summary.json, learning_curve.png, schedule.png}. The trained weights
are saved as <out>/model.npz (masks + values) for the Part 4 sparse inference benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train.data import load_digits_dataset
from train.experiment import RunConfig, run
from train.plotting import plot_learning_curve, plot_sparsity_schedule


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--criterion", default="saliency", choices=["magnitude", "saliency", "taylor1"])
    ap.add_argument("--scope", default="global", choices=["global", "layer"])
    ap.add_argument("--regrow", type=float, default=0.3, help="initial regrowth fraction (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/part3_self_prune")
    args = ap.parse_args()

    cfg = RunConfig(sparsity=args.sparsity, criterion=args.criterion, scope=args.scope,
                    regrow_fraction=args.regrow, seed=args.seed)
    data = load_digits_dataset(split_seed=0)
    print(f"config: {cfg}")
    result = run(cfg, data, verbose=True)
    model = result.pop("model")

    print("\npruning events (step: target -> achieved, pruned / grown):")
    for e in result["events"]:
        print(f"  step {e['step']:4d}: target {e['target_sparsity']:.4f} -> achieved "
              f"{e['achieved_sparsity']:.4f}   pruned {e['pruned']:5d}  grown {e['grown']:4d}")

    print("\nper-layer result:")
    for i, layer in enumerate(result["layers"]):
        print(f"  layer {i} {tuple(layer['shape'])}: {layer['active']:5d} active "
              f"({layer['density']:.3f} dense), exact zeros stored = {layer['stored_exact_zeros']}")
    print(f"\nachieved sparsity {result['achieved_sparsity']:.4f} "
          f"({result['active_weights']}/{result['prunable_weights']} weights active)")
    print(f"test loss {result['test_loss']:.4f}  test acc {result['test_acc']:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    arrays = {}
    for i, lin in enumerate(model.linears()):
        arrays[f"W{i}"], arrays[f"b{i}"], arrays[f"M{i}"] = lin.weight.data, lin.bias.data, lin.weight.mask
    np.savez(out / "model.npz", **arrays)
    title = (f"Part 3: self-pruning to {cfg.sparsity:.0%} ({cfg.criterion}, {cfg.scope}, "
             f"regrow {cfg.regrow_fraction})")
    plot_learning_curve(result["history"], str(out / "learning_curve.png"), title)
    plot_sparsity_schedule(result["events"], str(out / "schedule.png"))
    print(f"wrote {out}/summary.json, model.npz, learning_curve.png, schedule.png")


if __name__ == "__main__":
    main()
