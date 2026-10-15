"""Trainable parameters that carry a binary pruning mask.

The masked forward pass
-----------------------
A layer never uses ``W`` directly. It uses the *effective* weight

    W_eff = W ⊙ M,      M ∈ {0, 1}  (a constant: no gradient flows into M)

so a pruned connection (M_ij = 0) contributes an exact 0.0 to the forward
computation, whatever value is stored in W_ij. We additionally keep the
invariant W_ij == 0 wherever M_ij == 0 (``apply_mask_``) so that the stored
weights *are* the sparse model and can be exported to a sparse format directly.

Two gradients of a masked weight
--------------------------------
Let G = dL/dW_eff (what the downstream ops send back). By the chain rule

    dL/dW = G ⊙ M          -> ``param.grad``        (exactly 0 where pruned)
    dL/dW_eff = G          -> ``param.dense_grad``  (generally non-zero where pruned)

``grad`` is the true derivative of the loss w.r.t. the stored parameter: a
pruned weight does not affect the loss, so its derivative is 0, and the
optimizer must not move it. ``dense_grad`` answers a different question --
"how would the loss change if this connection existed?" -- which is what a
regrowth criterion needs. Mixing the two up is the classic bug: using G to
update pruned weights silently revives them, and using G ⊙ M for regrowth
means a pruned weight can never come back.
"""
from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data, prunable: bool = False):
        super().__init__(data, requires_grad=True)
        self.prunable = prunable
        self.mask = np.ones(self.shape, dtype=bool)
        self.dense_grad: np.ndarray | None = None

    # -------------------------------------------------------------- masking
    def masked(self) -> Tensor:
        """W ⊙ M as a graph node. Its backward records G = dL/dW_eff in
        ``dense_grad`` and passes G ⊙ M on to ``grad``.

        The mask is captured by value at forward time, so changing it between
        forward and backward cannot leak into this step's gradients."""
        m = self.mask.copy()

        def backward_fn(g):
            self.dense_grad = g.copy() if self.dense_grad is None else self.dense_grad + g
            return (g * m,)

        return Tensor._make(self.data * m, (self,), backward_fn, "mask")

    def set_mask(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} != parameter shape {self.shape}")
        self.mask = mask.copy()
        self.apply_mask_()

    def apply_mask_(self) -> None:
        """Enforce the invariant W == 0 wherever M == 0 (a real zero, not a small weight)."""
        self.data[~self.mask] = 0.0

    # ----------------------------------------------------------- reporting
    @property
    def num_active(self) -> int:
        return int(self.mask.sum())

    @property
    def density(self) -> float:
        return self.num_active / self.size

    def zero_grad(self) -> None:
        self.grad = None
        self.dense_grad = None

    def __repr__(self) -> str:
        tag = f", density={self.density:.3f}" if self.prunable else ""
        return f"Parameter(shape={self.shape}{tag})"
