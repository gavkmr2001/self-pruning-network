"""Finite-difference gradient checking for the autodiff engine.

For a function f of tensors x_1..x_k with (possibly non-scalar) output y, we
reduce to a scalar with a fixed random projection R:

    L(x) = sum(f(x) * R)

so that dL/dx = J^T R is a random vector-Jacobian product -- exactly what
``backward`` computes. A random R (rather than all-ones) makes it unlikely that
errors in different output entries cancel out.

Each input entry is perturbed with a central difference

    dL/dx_i ~= (L(x + h e_i) - L(x - h e_i)) / (2h),

whose truncation error is O(h^2). With float64 and h = 1e-6 that is ~1e-12,
and the round-off error is ~eps_machine * |L| / h ~ 1e-10, so a correct
analytic gradient agrees to roughly 1e-8 relative error or better.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from engine.tensor import Tensor


def numerical_grads(f: Callable[..., Tensor], inputs: Sequence[np.ndarray], R: np.ndarray, h: float = 1e-6):
    """Central-difference gradient of sum(f(*inputs) * R) with respect to each input."""
    arrays = [np.array(x, dtype=np.float64) for x in inputs]

    def L() -> float:
        return float((f(*[Tensor(a) for a in arrays]).data * R).sum())

    grads = []
    for a in arrays:
        g = np.zeros_like(a)
        it = np.nditer(a, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = a[idx]
            a[idx] = orig + h
            plus = L()
            a[idx] = orig - h
            minus = L()
            a[idx] = orig
            g[idx] = (plus - minus) / (2 * h)
        grads.append(g)
    return grads


def analytic_grads(f: Callable[..., Tensor], inputs: Sequence[np.ndarray], R: np.ndarray):
    tensors = [Tensor(np.array(x, dtype=np.float64), requires_grad=True) for x in inputs]
    out = f(*tensors)
    out.backward(R)
    return [t.grad if t.grad is not None else np.zeros_like(t.data) for t in tensors]


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| / max(|a|, |b|, 1e-8): scale-free, and safe when both are ~0."""
    denom = max(np.abs(a).max(initial=0.0), np.abs(b).max(initial=0.0), 1e-8)
    return float(np.abs(a - b).max(initial=0.0) / denom)


def gradcheck(f: Callable[..., Tensor], *inputs: np.ndarray, seed: int = 0, h: float = 1e-6, tol: float = 1e-6) -> float:
    """Raise AssertionError if analytic and numerical gradients of ``f`` disagree.

    Returns the worst relative error across all inputs (useful for reporting).
    """
    out_shape = f(*[Tensor(np.asarray(x, dtype=np.float64)) for x in inputs]).shape
    R = np.random.default_rng(seed).normal(size=out_shape)

    ana = analytic_grads(f, inputs, R)
    num = numerical_grads(f, inputs, R, h=h)

    worst = 0.0
    for i, (a, n) in enumerate(zip(ana, num)):
        assert a.shape == n.shape, f"input {i}: grad shape {a.shape} != input shape {n.shape}"
        err = relative_error(a, n)
        worst = max(worst, err)
        assert err < tol, f"input {i}: relative error {err:.3e} >= {tol:.0e}\nanalytic:\n{a}\nnumerical:\n{n}"
    return worst
