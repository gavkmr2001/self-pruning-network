# AQUA — The Self-Pruning Network

A from-scratch reverse-mode autodiff engine in pure Python + NumPy, an MLP trained
with a hand-written Adam optimizer, and a pruning system that removes the network's
least important connections *during* training under a hard sparsity budget.

> **Restrictions honoured:** no PyTorch / TensorFlow / JAX / Keras / HuggingFace or
> any autodiff / pruning / NN library. **scikit-learn is used only to load a standard
> dataset** — never for models, training, or gradients.

## Layout

| Folder    | Contents                                                        |
|-----------|-----------------------------------------------------------------|
| `engine/` | `Tensor`, differentiable ops, backward pass, gradient checking  |
| `nn/`     | Parameters (with masks), layers, MLP, optimizers                |
| `prune/`  | Importance criteria, sparsity schedule, pruner, sparse inference |
| `train/`  | Data loading and the mini-batch training loop                   |
| `tests/`  | Gradient-check suite and masked-weight correctness tests        |
| `scripts/`| One entry point per part of the challenge                       |
| `results/`| Committed raw numbers and plots                                 |

## Install

```bash
uv sync                      # or: pip install -r requirements.txt
```

## Commands

```bash
uv run pytest -v                    # gradient checks + masking tests
uv run python -m scripts.train_dense  # Part 2: dense MLP + Adam on digits (~10 s)
uv run python -m scripts.self_prune   # Part 3: self-pruning to 90% sparsity (~2 s)
```

## Part 2 — dense baseline

`digits` (1,797 8×8 images, 10 classes; loaded with `sklearn.datasets.load_digits`,
the only use of scikit-learn), stratified 60/20/20 train/val/test split, features
standardised with training-set statistics. MLP 64-128-128-10 (ReLU, He init for
hidden layers, Xavier for the logit layer), Adam lr 1e-3, batch 64, 40 epochs, seed 0.

| | value |
|---|---|
| test accuracy | 0.9861 |
| final val accuracy | 0.9583 |
| prunable weights | 25,856 |

Learning curve and raw per-epoch numbers: `results/part2_dense/`.

## Part 3 — self-pruning run

Same model, data and optimizer as Part 2. While training, the network removes its own
connections: dense warm-up for 5 epochs, then one mask update per epoch following a cubic
sparsity ramp to 90% at epoch 30, then 10 epochs of fine-tuning with a fixed mask.

* **Importance criterion:** saliency `S = ½ · E[g²] · w²` — second-order Taylor estimate of
  the loss increase from removing a weight, with the diagonal Fisher (EMA of squared
  gradients w.r.t. the effective weight) standing in for the Hessian. Global ranking across
  layers (scores are in loss units), with a 1% per-layer floor.
* **Regrowth:** a cosine-decaying share (30% → 0) of each event's budget is re-grown from
  previously pruned weights with the largest `|E[dL/dW_eff]|`.
* **Honest masking:** pruned weights are exact zeros in storage and in the forward pass;
  the optimizer holds them at 0 and gives revived weights fresh Adam state.

| seed 0 | dense (Part 2) | self-pruned |
|---|---|---|
| active weights | 25,856 | 2,586 (sparsity 0.9000) |
| test accuracy | 0.9861 | 0.9666 |
| test loss | 0.0505 | 0.0784 |
| final val accuracy / loss | 0.9583 / 0.167 | 0.9694 / 0.112 |

Per-layer density after pruning: 13.9% (64→128), 6.0% (128→128), 36.7% (128→10).
These are single-seed numbers; Part 4 reports means and variance over seeds.

Outputs in `results/part3_self_prune/`: `summary.json` (config, per-epoch history, every
pruning event), `learning_curve.png`, `schedule.png`, `model.npz` (weights + masks).
