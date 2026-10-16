"""Sparsity schedules: how sparse the network should be at each training step.

Cubic schedule (Zhu & Gupta, 2017)
----------------------------------
    s(t) = s_f + (s_i - s_f) * (1 - (t - t0) / (t1 - t0))^3      for t0 <= t <= t1

s(t) starts at s_i, ends at s_f, and its slope is steepest at t0 and zero at
t1. The reasoning:

* Early, the dense network is highly redundant: many weights can go at once
  at little cost, and there is a lot of training left to recover.
* Late, every remaining weight matters more and there is little time to
  recover, so each event removes only a little.
* Pruning *gradually* (an event every ``every`` steps) lets the surviving
  weights adapt between events, unlike one-shot pruning at the end, which
  must remove everything at once with no recovery. One-shot is the special
  case t0 == t1 (followed by fine-tuning, if any steps remain).
"""
from __future__ import annotations


class CubicSchedule:
    def __init__(self, final_sparsity: float, start_step: int, end_step: int, every: int, initial_sparsity: float = 0.0):
        if not 0.0 <= initial_sparsity <= final_sparsity < 1.0:
            raise ValueError("need 0 <= initial_sparsity <= final_sparsity < 1")
        if not 0 <= start_step <= end_step:
            raise ValueError("need 0 <= start_step <= end_step")
        if every < 1:
            raise ValueError("every must be >= 1")
        self.final, self.initial = final_sparsity, initial_sparsity
        self.start, self.end, self.every = start_step, end_step, every

    def sparsity_at(self, step: int) -> float:
        if step < self.start:
            return 0.0
        if step >= self.end:
            return self.final
        progress = (step - self.start) / (self.end - self.start)
        return self.final + (self.initial - self.final) * (1.0 - progress) ** 3

    def is_update_step(self, step: int) -> bool:
        """Mask updates every ``every`` steps in [start, end]; the end step always
        updates so the final sparsity is hit exactly."""
        if step < self.start or step > self.end:
            return False
        return step == self.end or (step - self.start) % self.every == 0

    def progress(self, step: int) -> float:
        if self.end == self.start:
            return 1.0
        return min(max((step - self.start) / (self.end - self.start), 0.0), 1.0)

    def __repr__(self) -> str:
        return (f"CubicSchedule({self.initial}->{self.final}, steps {self.start}..{self.end}, "
                f"every {self.every})")
