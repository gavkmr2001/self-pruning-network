"""Matplotlib helpers (headless backend, so scripts run without a display)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_learning_curve(history: list[dict], path: str, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], label="validation")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(alpha=0.3)

    if any(h.get("sparsity", 0.0) > 0 for h in history):
        ax_sp = ax_acc.twinx()
        ax_sp.plot(epochs, [h["sparsity"] for h in history], color="grey", linestyle="--", label="sparsity")
        ax_sp.set(ylabel="sparsity", ylim=(0, 1))
        ax_sp.legend(loc="center right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
