"""Mask-aware optimizers: SGD with momentum, and Adam(W).

What an optimizer must do with a masked weight
----------------------------------------------
``param.grad`` is exactly 0 at a pruned weight (dL/dW = G ⊙ M). A zero
gradient is NOT enough to keep the weight still, because both optimizers
carry state:

* Momentum: v <- μ v + g. With g = 0 the old velocity keeps pushing, so the
  weight keeps moving for ~1/(1-μ) steps after it was "removed".
* Adam: m <- β1 m + (1-β1) g, v <- β2 v + (1-β2) g². With g = 0 the update
  m̂ / (sqrt(v̂) + ε) is still non-zero, so the stored weight drifts away from 0.

And when a connection is revived, stale state corrupts it in two ways:

* stale m (decays like 0.9^k) shoves it in a direction that was relevant
  before it was pruned;
* stale v (decays like 0.999^k, i.e. it lingers for thousands of steps) makes
  its effective learning rate tiny, so it can barely learn.

The subtle one: even after zeroing m and v on revival, using the *global* step
count t in Adam's bias correction is wrong. Adam's correction 1/(1-β^t)
assumes the moments have been accumulating for t steps. A revived weight's
moments have been accumulating for 1 step, so its first update is
[(1-β1)/(1-β1^t)] / sqrt[(1-β2)/(1-β2^t)] times a genuine first Adam step:
≈ 2x at t = 500, approaching (1-β1)/sqrt(1-β2) ≈ 3.16x for t ≫ 1/(1-β2).
The error persists for the next ~1/(1-β2) = 1000 steps while v̂ is
under-corrected.

The rule implemented here
-------------------------
For every entry where the mask is 0, at every step:
    the weight is held at exactly 0, its state (velocity / m, v) is zeroed,
    and (Adam) its per-element step count is reset to 0.
For active entries the update is ordinary SGD-momentum / Adam, with Adam's
bias correction using the *per-element* count of steps the weight has been
alive. Consequence: a revived connection is optimised exactly as if it were a
brand-new parameter initialised at 0, and pruning one entry never changes how
any other entry is updated (both optimizers are element-wise).

The optimizer reads the current mask at each step, so it needs no notification
from the pruner -- prune, revive, or both, between any two steps.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from nn.parameter import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float):
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            active = p.mask
            self._reset_state(i, ~active)
            self._update(i, p, active)
            p.apply_mask_()  # weights of pruned entries stay a real 0.0

    def _reset_state(self, i: int, dead: np.ndarray) -> None:
        raise NotImplementedError

    def _update(self, i: int, p: Parameter, active: np.ndarray) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """v <- μ v + g (+ λ w);  w <- w - lr v."""

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum, self.weight_decay = momentum, weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def _reset_state(self, i, dead):
        self.velocity[i][dead] = 0.0

    def _update(self, i, p, active):
        g = p.grad
        if self.weight_decay:
            g = g + self.weight_decay * p.data
        v = self.velocity[i]
        v[active] = self.momentum * v[active] + g[active]
        p.data[active] -= self.lr * v[active]


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with per-element step counts; decoupled weight decay (AdamW)."""

    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = [np.zeros(p.shape, dtype=np.int64) for p in self.params]

    def _reset_state(self, i, dead):
        self.m[i][dead] = 0.0
        self.v[i][dead] = 0.0
        self.t[i][dead] = 0

    def _update(self, i, p, active):
        b1, b2 = self.beta1, self.beta2
        g = p.grad[active]
        t = self.t[i][active] + 1
        self.t[i][active] = t

        m = b1 * self.m[i][active] + (1 - b1) * g
        v = b2 * self.v[i][active] + (1 - b2) * g * g
        self.m[i][active], self.v[i][active] = m, v

        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        w = p.data[active]
        if self.weight_decay:
            w = w * (1 - self.lr * self.weight_decay)
        p.data[active] = w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
