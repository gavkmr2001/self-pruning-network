"""Part 2: train a dense MLP on `digits` with our own Adam and mini-batch loop.

    uv run python -m scripts.train_dense

Writes results/part2_dense/{history.json, learning_curve.png}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn import MLP
from nn.optim import Adam
from train.data import load_digits_dataset
from train.loop import evaluate, train
from train.plotting import plot_learning_curve

HIDDEN = [128, 128]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/part2_dense")
    args = ap.parse_args()

    data = load_digits_dataset(split_seed=0)
    model = MLP([data.num_features, *HIDDEN, data.num_classes], activation="relu", seed=args.seed)
    optimizer = Adam(model.parameters(), lr=args.lr)
    print(model)
    print(f"train/val/test = {len(data.train)}/{len(data.val)}/{len(data.test)}, "
          f"prunable weights = {model.num_prunable()}")

    history = train(model, data, optimizer, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    test_loss, test_acc = evaluate(model, data.test)
    print(f"test loss {test_loss:.4f}  test acc {test_acc:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(
        {"config": vars(args) | {"hidden": HIDDEN}, "history": history,
         "test_loss": test_loss, "test_acc": test_acc}, indent=2))
    plot_learning_curve(history, str(out / "learning_curve.png"), "Part 2: dense MLP on digits (Adam)")
    print(f"wrote {out}/history.json and {out}/learning_curve.png")


if __name__ == "__main__":
    main()
