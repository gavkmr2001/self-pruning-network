"""Small, dependency-free statistics for comparing methods across random seeds.

Paired design: method A and method B are run with the SAME seed (same initial
weights, same batch order, same data split), so the per-seed difference
d_s = metric_A(s) − metric_B(s) cancels most seed-to-seed variation. We test
whether the mean of d is different from 0.

* ``paired_permutation_test`` -- exact sign-flip test. Under H0 (no
  difference) each d_s is equally likely to be +|d_s| or −|d_s|; we enumerate
  all 2^n sign patterns (1,024 for n = 10) and report the fraction whose
  |mean| is >= the observed |mean|. No normality assumption.
* ``bootstrap_ci`` -- percentile bootstrap confidence interval for the mean
  difference.
* ``holm`` -- Holm–Bonferroni correction, because we test several sparsity
  levels and would otherwise find "significant" noise by chance.
"""
from __future__ import annotations

import itertools

import numpy as np


def paired_permutation_test(diffs, max_exact: int = 16, n_random: int = 100_000, seed: int = 0) -> float:
    """Two-sided p-value for H0: mean(diffs) == 0 (exact for n <= max_exact)."""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    observed = abs(d.mean())
    if n == 0 or np.all(d == 0):
        return 1.0
    if n <= max_exact:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([1.0, -1.0], size=(n_random, n))
    means = np.abs((signs * d).mean(axis=1))
    return float(np.mean(means >= observed - 1e-12))


def bootstrap_ci(values, level: float = 0.95, n_boot: int = 20_000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def holm(pvalues) -> list[float]:
    """Holm–Bonferroni adjusted p-values (controls family-wise error rate)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()
