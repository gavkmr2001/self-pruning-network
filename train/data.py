"""Dataset loading and splitting.

scikit-learn is used ONLY to load the raw arrays of the standard `digits`
dataset (``sklearn.datasets.load_digits``: 1,797 8x8 grey-scale images, 10
classes, shipped inside the sklearn package so no download is needed).
Splitting, normalisation, batching -- everything after loading -- is NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Dataset:
    train: Split
    val: Split
    test: Split
    num_features: int
    num_classes: int


def stratified_split(y: np.ndarray, fractions=(0.6, 0.2, 0.2), seed: int = 0):
    """Index arrays for train/val/test with the same class proportions in each."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in fractions]
    bounds = np.cumsum(fractions)[:-1]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        cuts = np.round(bounds * len(idx)).astype(int)
        for part, chunk in zip(parts, np.split(idx, cuts)):
            part.append(chunk)
    return [rng.permutation(np.concatenate(p)) for p in parts]


def load_digits_dataset(split_seed: int = 0) -> Dataset:
    from sklearn.datasets import load_digits  # loading only

    raw = load_digits()
    X, y = raw.data.astype(np.float64), raw.target.astype(np.int64)
    tr, va, te = stratified_split(y, seed=split_seed)

    # Standardise with TRAIN statistics only (no leakage from val/test).
    # Border pixels are constant 0 in train; the std floor maps them to 0
    # instead of dividing by zero.
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    Xn = (X - mean) / std

    return Dataset(
        train=Split(Xn[tr], y[tr]),
        val=Split(Xn[va], y[va]),
        test=Split(Xn[te], y[te]),
        num_features=X.shape[1],
        num_classes=int(y.max()) + 1,
    )


def minibatches(split: Split, batch_size: int, rng: np.random.Generator):
    """Shuffled mini-batches; the last partial batch is kept (every sample is seen)."""
    order = rng.permutation(len(split))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield split.X[idx], split.y[idx]
