"""One configurable training run (dense or self-pruning), shared by the Part 3
script and the Part 4 sweep so that every reported number comes from the same code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from nn import MLP
from nn.optim import Adam
from prune import CubicSchedule, Pruner
from train.data import Dataset, load_digits_dataset
from train.loop import evaluate, train


@dataclass
class RunConfig:
    sparsity: float = 0.9               # final fraction of prunable weights removed (0 = dense)
    criterion: str = "saliency"         # magnitude | saliency | taylor1
    scope: str = "global"               # global | layer
    regrow_fraction: float = 0.3        # initial share of each event's budget re-grown (0 = off)
    seed: int = 0                       # model init + batch order (the data split is fixed)
    hidden: list = field(default_factory=lambda: [128, 128])
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    prune_start_epoch: int = 5          # dense warm-up before the first event
    prune_end_epoch: int = 30           # last event; remaining epochs fine-tune a fixed mask
    prune_every_epochs: int = 1
    min_density: float = 0.01
    ema_decay: float = 0.9


def run(cfg: RunConfig, data: Dataset | None = None, verbose: bool = False) -> dict:
    data = data if data is not None else load_digits_dataset(split_seed=0)
    model = MLP([data.num_features, *cfg.hidden, data.num_classes], activation="relu", seed=cfg.seed)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    pruner = None
    if cfg.sparsity > 0:
        steps_per_epoch = int(np.ceil(len(data.train) / cfg.batch_size))
        schedule = CubicSchedule(
            cfg.sparsity,
            start_step=cfg.prune_start_epoch * steps_per_epoch,
            end_step=cfg.prune_end_epoch * steps_per_epoch,
            every=cfg.prune_every_epochs * steps_per_epoch,
        )
        pruner = Pruner(model, schedule, criterion=cfg.criterion, scope=cfg.scope,
                        regrow_fraction=cfg.regrow_fraction, ema_decay=cfg.ema_decay,
                        min_density=cfg.min_density)

    history = train(model, data, optimizer, epochs=cfg.epochs, batch_size=cfg.batch_size, seed=cfg.seed,
                    on_step=pruner.on_step if pruner else None, verbose=verbose)
    test_loss, test_acc = evaluate(model, data.test)

    layers = []
    for lin in model.linears():
        W = lin.weight
        layers.append({
            "shape": list(W.shape),
            "active": W.num_active,
            "density": W.density,
            "stored_exact_zeros": int((W.data == 0.0).sum()),
        })
    return {
        "config": asdict(cfg),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "final_val_acc": history[-1]["val_acc"],
        "achieved_sparsity": model.sparsity(),
        "active_weights": model.num_active(),
        "prunable_weights": model.num_prunable(),
        "layers": layers,
        "history": history,
        "events": pruner.events if pruner else [],
        "model": model,
    }
