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


