"""Importance criteria: how a connection earns the right to survive.

All scores estimate (or proxy) the loss increase  ΔL_i = L(w with w_i := 0) − L(w);
higher score = more important = kept longer.

Derivation of the saliency criterion
------------------------------------
Removing weight i is the perturbation δ = −w_i e_i. Second-order Taylor:

    ΔL_i ≈ g_i δ_i + ½ H_ii δ_i²  =  −g_i w_i + ½ H_ii w_i²          (1)

where g = ∂L/∂w and H the Hessian of the *expected* loss. Two facts about
the training regime turn (1) into something we can compute:

(a) Near a (local) minimum the expected gradient E[g_i] → 0, so the first-order
    term vanishes and the curvature term ½ H_ii w_i² dominates. A purely
    first-order score |E[g_i] w_i| therefore becomes noise late in training
    (we keep it as the ``taylor1`` ablation).

(b) For the cross-entropy loss of a model whose predictions match the data
    distribution, the Hessian equals the Fisher information F = E[g gᵀ]
    (Gauss–Newton/Fisher identity). We estimate its diagonal from the
    mini-batch gradients we already have:  F_ii ≈ EMA_t[g_i(t)²].

Plugging in gives the Optimal-Brain-Damage-style saliency

    S_i = ½ · F_ii · w_i²  ≈ ΔL_i.                                       (2)

The same formula arises from the first-order view averaged over batches:
E[(g_i w_i)²] = w_i² E[g_i²] = w_i² F_ii, i.e. (2) is (half) the mean squared
first-order Taylor term (Molchanov et al., 2019). Either way, S_i is in units
of *loss*, so scores are comparable across layers -- unlike |w|, whose scale
depends on each layer's fan-in and initialisation. That is what makes global
(cross-layer) ranking principled for S and ad hoc for |w|.

Magnitude |w_i| is (2) under the extra assumption that every weight has the
same curvature F_ii. Saliency drops that assumption: a large weight on a
rarely-active input (small F_ii) can be removed cheaply; a small weight on a
pixel that matters for every image may not.

Regrowth score
--------------
For a pruned weight w_i = 0, so S_i = 0 by construction and it could never
win a place back on S. What matters instead is how fast the loss would fall
if the connection were re-added: to first order, |∂L/∂w_eff,i| = |G_i| per
unit of weight. We use |EMA_t[G_i]|, where G is the gradient w.r.t. the
*effective* weight (``param.dense_grad``) -- which is non-zero at pruned
entries, unlike ``param.grad``. Averaging over steps (the EMA) keeps
connections whose gradient is consistently signed and rejects ones that
merely had one noisy batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from nn.parameter import Parameter


@dataclass
class GradStats:
    """Exponential moving averages of the dense gradient G = dL/dW_eff."""

    grad_ema: np.ndarray       # EMA of G     -> E[g]   (first-order term, regrowth)
    sq_grad_ema: np.ndarray    # EMA of G²    -> diag Fisher F
    steps: int = 0
    decay: float = field(default=0.9)

    @classmethod
    def zeros(cls, shape, decay: float) -> "GradStats":
        return cls(np.zeros(shape), np.zeros(shape), 0, decay)

    def update(self, G: np.ndarray) -> None:
        d = self.decay
        self.grad_ema = d * self.grad_ema + (1 - d) * G
        self.sq_grad_ema = d * self.sq_grad_ema + (1 - d) * G * G
        self.steps += 1

    def corrected(self) -> tuple[np.ndarray, np.ndarray]:
        """Bias-corrected averages (EMAs start at 0, same issue as Adam's moments)."""
        c = 1.0 - self.decay ** max(self.steps, 1)
        return self.grad_ema / c, self.sq_grad_ema / c


Criterion = Callable[[Parameter, GradStats], np.ndarray]


def magnitude(p: Parameter, stats: GradStats) -> np.ndarray:
    """Baseline: |w|. Equivalent to (2) if all weights shared one curvature."""
    return np.abs(p.data)


def saliency(p: Parameter, stats: GradStats) -> np.ndarray:
    """½ · F_ii · w_i²  -- second-order Taylor with the diagonal-Fisher Hessian."""
    _, fisher = stats.corrected()
    return 0.5 * fisher * p.data ** 2


def taylor1(p: Parameter, stats: GradStats) -> np.ndarray:
    """|E[g] · w| -- first-order Taylor only (ablation; weak near convergence)."""
    g_mean, _ = stats.corrected()
    return np.abs(g_mean * p.data)


CRITERIA: dict[str, Criterion] = {"magnitude": magnitude, "saliency": saliency, "taylor1": taylor1}


def regrowth_score(p: Parameter, stats: GradStats) -> np.ndarray:
    """|E[G]|: first-order loss decrease per unit weight if the connection existed."""
    g_mean, _ = stats.corrected()
    return np.abs(g_mean)
