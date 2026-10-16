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


def plot_sparsity_schedule(events: list[dict], path: str) -> None:
    """Target vs achieved sparsity per event, per-layer density, and prune/grow counts."""
    steps = [e["step"] for e in events]
    fig, (ax_s, ax_n) = plt.subplots(1, 2, figsize=(11, 4))

    ax_s.plot(steps, [e["target_sparsity"] for e in events], "o-", label="target (cubic)")
    ax_s.plot(steps, [e["achieved_sparsity"] for e in events], "x--", label="achieved")
    n_layers = len(events[0]["layer_density"])
    for i in range(n_layers):
        ax_s.plot(steps, [1 - e["layer_density"][i] for e in events], alpha=0.6, label=f"layer {i} sparsity")
    ax_s.set(xlabel="training step", ylabel="sparsity", title="Sparsity schedule", ylim=(0, 1))
    ax_s.legend(fontsize=8)
    ax_s.grid(alpha=0.3)

    ax_n.bar(steps, [e["pruned"] for e in events], width=8, label="pruned")
    ax_n.bar(steps, [-e["grown"] for e in events], width=8, label="grown (negative)")
    ax_n.set(xlabel="training step", ylabel="weights per event", title="Mask changes per event")
    ax_n.axhline(0, color="black", linewidth=0.5)
    ax_n.legend()
    ax_n.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
