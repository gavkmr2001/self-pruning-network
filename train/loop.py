"""Mini-batch training loop with per-epoch evaluation and a per-step hook.

The hook (``on_step``) is how Part 3 plugs pruning in without changing this
loop: it is called after backward() and BEFORE optimizer.step(), when both
``param.grad`` and ``param.dense_grad`` hold this step's gradients, so a
pruner can score connections and update masks. The optimizer then reads the
new masks in the same step.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from engine.functional import softmax_cross_entropy
from engine.tensor import Tensor, no_grad
from nn.layers import Module
from nn.optim import Optimizer
from train.data import Dataset, Split, minibatches


class NonFiniteLossError(RuntimeError):
    pass


def evaluate(model: Module, split: Split, batch_size: int = 1024) -> tuple[float, float]:
    """(mean loss, accuracy) without building a graph."""
    total_loss, correct = 0.0, 0
    with no_grad():
        for start in range(0, len(split), batch_size):
            X, y = split.X[start:start + batch_size], split.y[start:start + batch_size]
            logits = model(Tensor(X))
            total_loss += float(softmax_cross_entropy(logits, y).data) * len(y)
            correct += int((logits.data.argmax(axis=1) == y).sum())
    return total_loss / len(split), correct / len(split)


def train(
    model: Module,
    data: Dataset,
    optimizer: Optimizer,
    epochs: int,
    batch_size: int = 64,
    seed: int = 0,
    on_step: Callable[[int, int], None] | None = None,
    on_epoch: Callable[[int, dict], None] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Train for ``epochs`` epochs; returns one history record per epoch."""
    rng = np.random.default_rng(seed)
    steps_per_epoch = int(np.ceil(len(data.train) / batch_size))
    history, step = [], 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss_sum, correct, seen = 0.0, 0, 0
        for X, y in minibatches(data.train, batch_size, rng):
            optimizer.zero_grad()
            logits = model(Tensor(X))
            loss = softmax_cross_entropy(logits, y)
            if not np.isfinite(loss.data):
                raise NonFiniteLossError(f"loss became {loss.data} at epoch {epoch}, step {step}")
            loss.backward()
            if on_step is not None:
                on_step(step, steps_per_epoch * epochs)
            optimizer.step()

            loss_sum += float(loss.data) * len(y)
            correct += int((logits.data.argmax(axis=1) == y).sum())
            seen += len(y)
            step += 1

        val_loss, val_acc = evaluate(model, data.val)
        record = {
            "epoch": epoch,
            "step": step,
            "train_loss": loss_sum / seen,     # running average over the epoch (weights change during it)
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": time.perf_counter() - t0,
        }
        if hasattr(model, "sparsity"):
            record["sparsity"] = model.sparsity()
        if on_epoch is not None:
            on_epoch(epoch, record)
        history.append(record)
        if verbose:
            sp = f"  sparsity {record['sparsity']:.3f}" if "sparsity" in record else ""
            print(
                f"epoch {epoch:3d} | train loss {record['train_loss']:.4f} acc {record['train_acc']:.4f}"
                f" | val loss {val_loss:.4f} acc {val_acc:.4f}{sp}"
            )
    return history
