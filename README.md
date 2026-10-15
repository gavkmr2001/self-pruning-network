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
uv run pytest -v             # gradient checks + masking tests
```


