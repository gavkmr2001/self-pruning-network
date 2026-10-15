"""Fused, numerically stable operations that are clearer as a single graph node."""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor, as_tensor


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax via the log-sum-exp trick (plain NumPy, no graph).

    log softmax(x)_k = x_k - m - log(sum_j exp(x_j - m)),  m = max_j x_j.
    Shifting by m leaves the result unchanged mathematically but keeps every
    exp() argument <= 0, so nothing overflows; the largest term is exp(0) = 1,
    so the sum is >= 1 and the log never sees 0.
    """
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(x))


def softmax_cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean cross-entropy of integer class ``targets`` under softmax(``logits``).

    L = (1/N) * sum_i [ logsumexp(z_i) - z_{i, y_i} ]

    Fusing softmax with the loss gives the famously simple gradient
        dL/dz_i = (softmax(z_i) - onehot(y_i)) / N,
    which avoids back-propagating through a separate softmax Jacobian (and the
    1/p blow-up of d log p / d p when p underflows to 0).
    """
    logits = as_tensor(logits)
    z = logits.data
    if z.ndim != 2:
        raise ValueError(f"logits must be (N, C), got {z.shape}")
    y = np.asarray(targets, dtype=np.int64)
    n = z.shape[0]
    if y.shape != (n,):
        raise ValueError(f"targets must be ({n},), got {y.shape}")

    logp = log_softmax(z)
    loss = -logp[np.arange(n), y].mean()

    def backward_fn(g):
        dz = np.exp(logp)
        dz[np.arange(n), y] -= 1.0
        return (g * dz / n,)

    return Tensor._make(np.asarray(loss), (logits,), backward_fn, "softmax_xent")
