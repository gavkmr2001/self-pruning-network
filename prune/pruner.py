"""The pruner: collects gradient statistics every step and rewrites masks on schedule.

Plugged into the training loop through ``train(..., on_step=pruner.on_step)``,
which runs after backward() and before optimizer.step().

One mask-update event at step t (target sparsity s_t, N prunable weights):

  1. target_active = N − round(s_t · N)
  2. n_grow = round(f_t · target_active) if regrowth is on, else 0, where the
     regrowth fraction f_t decays from f_0 to 0 along a cosine over the schedule
  3. DROP: keep the (target_active − n_grow) currently-active weights with the
     highest importance score; prune the rest.
  4. GROW: among weights that were ALREADY pruned before this event, revive the
     n_grow with the highest regrowth score |E[G]|. (Weights dropped in step 3
     are not eligible -- otherwise a weight could be dropped and re-grown in
     the same event, which is a no-op that wastes the regrowth budget.)
  5. Active count is now exactly target_active.

  Early in the schedule the regrowth budget (f_t · target_active) exceeds the
  pool of pruned weights, so the whole pool is re-grown and the event in effect
  re-selects the full pruned set from current scores instead of locking in
  decisions made on a barely-trained network. We measured the alternative
  (budget capped at f_t · |pruned pool|) on the validation set, 5 seeds: no
  better at 90% and slightly worse at 97% sparsity, so we keep the RigL-style
  budget relative to the active count.

Ranking scope:
  * "global": one ranking across all layers (meaningful for scores in units of
    loss, like saliency);
  * "layer": each layer pruned to s_t separately (uniform sparsity).
  ``min_density`` reserves each layer's top-scoring weights so no layer can be
  pruned to nothing (which would disconnect the network).

Stability of regrowth
---------------------
* A revived weight restarts at 0, so reviving never changes the network's
  output at that instant; it only gives the optimizer a new direction.
* The optimizer gives it fresh state (see nn/optim.py), so it is trained like
  a newly created parameter.
* The gradient the optimizer sees this step was computed under the OLD mask,
  so ``grad`` is 0 at a just-revived entry. Feeding that 0 to Adam would burn
  the weight's first bias-corrected step on a fake zero gradient: at t = 2,
  m̂ = 0.1g/(1−0.9²) ≈ 0.53g and v̂ = 0.001g²/(1−0.999²) ≈ 0.50g², so its first
  real step would be ≈ 0.74 lr instead of lr. We therefore copy ``dense_grad`` --
  exactly the gradient a weight at 0 would have received (tested in
  tests/test_nn.py) -- into ``grad`` at revived entries.
* Revived weights are at 0 and so have the lowest saliency; they get until the
  next event (``every`` steps) to prove themselves. The regrowth fraction
  decays to 0, so the mask converges and the last phase of training is pure
  fine-tuning with a fixed topology.
"""
from __future__ import annotations

import math

import numpy as np

from nn.layers import Module
from nn.parameter import Parameter
from prune.criteria import CRITERIA, GradStats, regrowth_score
from prune.schedule import CubicSchedule


class Pruner:
    def __init__(
        self,
        model: Module,
        schedule: CubicSchedule,
        criterion: str = "saliency",
        scope: str = "global",
        regrow_fraction: float = 0.0,
        ema_decay: float = 0.9,
        min_density: float = 0.01,
    ):
        if criterion not in CRITERIA:
            raise ValueError(f"unknown criterion {criterion!r}; choose from {sorted(CRITERIA)}")
        if scope not in ("global", "layer"):
            raise ValueError("scope must be 'global' or 'layer'")
        self.params: list[Parameter] = model.prunable_parameters()
        self.schedule = schedule
        self.criterion_name = criterion
        self.criterion = CRITERIA[criterion]
        self.scope = scope
        self.regrow_fraction = regrow_fraction
        self.min_density = min_density
        self.stats = [GradStats.zeros(p.shape, ema_decay) for p in self.params]
        self.total = sum(p.size for p in self.params)
        self.events: list[dict] = []

    # ---------------------------------------------------------- training hook
    def on_step(self, step: int, total_steps: int) -> None:
        for p, st in zip(self.params, self.stats):
            if p.dense_grad is not None:
                st.update(p.dense_grad)
        if self.schedule.is_update_step(step):
            self.update_masks(step)

    # ---------------------------------------------------------------- events
    def regrow_fraction_at(self, step: int) -> float:
        return self.regrow_fraction * 0.5 * (1.0 + math.cos(math.pi * self.schedule.progress(step)))

    def update_masks(self, step: int) -> dict:
        s = self.schedule.sparsity_at(step)
        old_masks = [p.mask.copy() for p in self.params]
        importance = [self.criterion(p, st) for p, st in zip(self.params, self.stats)]
        growth = [regrowth_score(p, st) for p, st in zip(self.params, self.stats)]

        if self.scope == "global":
            groups = [list(range(len(self.params)))]
        else:
            groups = [[i] for i in range(len(self.params))]

        new_masks = [m.copy() for m in old_masks]
        n_grown = 0
        for group in groups:
            n_total = sum(self.params[i].size for i in group)
            target_active = n_total - int(round(s * n_total))
            n_active_before = sum(int(old_masks[i].sum()) for i in group)
            n_pruned_before = n_total - n_active_before
            n_grow = int(round(self.regrow_fraction_at(step) * target_active))
            # If fewer weights are active than the target (sparsity went DOWN, e.g. a
            # mask set externally), the shortfall must be grown, not just kept.
            n_grow = max(n_grow, target_active - n_active_before)
            n_grow = min(n_grow, n_pruned_before)
            n_reserved = sum(math.ceil(self.min_density * self.params[i].size) for i in group)
            if target_active - n_grow < n_reserved and self.min_density > 0:
                raise ValueError(
                    f"min_density={self.min_density} reserves {n_reserved} weights but only "
                    f"{target_active - n_grow} may be kept at sparsity {s:.4f}; lower min_density")
            keep = self._select(group, importance, old_masks, target_active - n_grow, reserve=True)
            grow = self._select(group, growth, [~m for m in old_masks], n_grow, reserve=False)
            for i in group:
                new_masks[i] = keep[i] | grow[i]
            n_grown += n_grow

        for p, old, new in zip(self.params, old_masks, new_masks):
            revived = new & ~old
            p.set_mask(new)
            if p.grad is not None and revived.any() and p.dense_grad is not None:
                p.grad[revived] = p.dense_grad[revived]

        event = {
            "step": step,
            "target_sparsity": s,
            "achieved_sparsity": self.sparsity(),
            "pruned": int(sum((o & ~n).sum() for o, n in zip(old_masks, new_masks))),
            "grown": n_grown,
            "layer_density": [p.density for p in self.params],
        }
        self.events.append(event)
        return event

    def _select(self, group, scores, eligible, k, reserve):
        """Boolean masks marking the k highest-scoring eligible entries in ``group``.

        With ``reserve``, each layer's top ceil(min_density · size) eligible entries
        are taken first, so no layer can be emptied by a global ranking."""
        chosen = {i: np.zeros(self.params[i].shape, dtype=bool) for i in group}
        if k <= 0:
            return chosen
        flat_scores, owners, positions = [], [], []
        for i in group:
            idx = np.flatnonzero(eligible[i])
            sc = scores[i].ravel()[idx].astype(np.float64)
            if reserve and self.min_density > 0 and len(idx):
                n_res = min(len(idx), math.ceil(self.min_density * self.params[i].size))
                top = np.argsort(-sc, kind="stable")[:n_res]
                sc = sc.copy()
                sc[top] = np.inf
            flat_scores.append(sc)
            owners.append(np.full(len(idx), i))
            positions.append(idx)
        flat_scores = np.concatenate(flat_scores)
        owners, positions = np.concatenate(owners), np.concatenate(positions)
        k = min(k, len(flat_scores))
        # Stable sort on -score: ties broken by position, so results are deterministic.
        top = np.argsort(-flat_scores, kind="stable")[:k]
        for j in top:
            chosen[owners[j]].ravel()[positions[j]] = True
        return chosen

    # ------------------------------------------------------------- reporting
    def sparsity(self) -> float:
        return 1.0 - sum(p.num_active for p in self.params) / self.total
