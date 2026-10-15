"""Finite-difference gradient checks for every differentiable operation in the engine.

Inputs that hit a non-differentiable point (ReLU at 0) or a singularity
(division by ~0, log of <= 0) are kept away from it, because a central
difference straddling a kink measures the average of the two one-sided slopes
rather than either of them.
"""
import numpy as np
import pytest

from engine.functional import softmax_cross_entropy
from engine.gradcheck import gradcheck, relative_error

rng = np.random.default_rng(42)


def normal(*shape):
    return rng.normal(size=shape)


def away_from_zero(*shape, low=0.5, high=2.0):
    """Random values with |x| in [low, high] and random sign."""
    return rng.uniform(low, high, size=shape) * rng.choice([-1.0, 1.0], size=shape)


def positive(*shape):
    return rng.uniform(0.5, 2.0, size=shape)


# (name, function, inputs). Shapes deliberately exercise broadcasting.
CASES = [
    # element-wise, same shape
    ("add", lambda a, b: a + b, [normal(3, 4), normal(3, 4)]),
    ("sub", lambda a, b: a - b, [normal(3, 4), normal(3, 4)]),
    ("mul", lambda a, b: a * b, [normal(3, 4), normal(3, 4)]),
    ("div", lambda a, b: a / b, [normal(3, 4), away_from_zero(3, 4)]),
    ("neg", lambda a: -a, [normal(3, 4)]),
    ("pow3", lambda a: a ** 3, [normal(3, 4)]),
    ("pow_half", lambda a: a ** 0.5, [positive(3, 4)]),
    # broadcasting: (N, D) op (D,)
    ("add_bcast_row", lambda a, b: a + b, [normal(5, 3), normal(3)]),
    ("sub_bcast_row", lambda a, b: a - b, [normal(5, 3), normal(3)]),
    ("mul_bcast_row", lambda a, b: a * b, [normal(5, 3), normal(3)]),
    ("div_bcast_row", lambda a, b: a / b, [normal(5, 3), away_from_zero(3)]),
    # broadcasting: (N, D) op (N, 1), (1, D) op (N, 1), (N, D) op scalar
    ("add_bcast_col", lambda a, b: a + b, [normal(5, 3), normal(5, 1)]),
    ("mul_outer", lambda a, b: a * b, [normal(1, 4), normal(3, 1)]),
    ("div_bcast_scalar", lambda a, b: a / b, [normal(2, 3), away_from_zero()]),
    ("mul_bcast_3d", lambda a, b: a * b, [normal(2, 3, 4), normal(3, 1)]),
    # reflected ops with Python constants
    ("rsub_rdiv", lambda a: 2.0 - 3.0 / a, [away_from_zero(4)]),
    # matmul
    ("matmul", lambda a, b: a @ b, [normal(4, 3), normal(3, 5)]),
    ("matmul_batched", lambda a, b: a @ b, [normal(2, 4, 3), normal(2, 3, 5)]),
    ("matmul_bcast_batch", lambda a, b: a @ b, [normal(2, 4, 3), normal(3, 5)]),
    ("linear_layer", lambda x, w, b: x @ w + b, [normal(6, 4), normal(4, 3), normal(3)]),
    # reductions
    ("sum_all", lambda a: a.sum(), [normal(3, 4)]),
    ("sum_axis0", lambda a: a.sum(axis=0), [normal(3, 4)]),
    ("sum_axis1_keep", lambda a: a.sum(axis=1, keepdims=True), [normal(3, 4)]),
    ("sum_axis_neg_tuple", lambda a: a.sum(axis=(0, -1)), [normal(2, 3, 4)]),
    ("mean_all", lambda a: a.mean(), [normal(3, 4)]),
    ("mean_axis1", lambda a: a.mean(axis=1), [normal(3, 4)]),
    ("mean_axis0_keep", lambda a: a.mean(axis=0, keepdims=True), [normal(3, 4)]),
    # shape ops
    ("reshape", lambda a: a.reshape(4, 3) * np.arange(12.0).reshape(4, 3), [normal(3, 4)]),
    ("transpose", lambda a: a.T, [normal(3, 4)]),
    ("transpose_axes", lambda a: a.transpose(2, 0, 1), [normal(2, 3, 4)]),
    # unary maths and non-linearities
    ("exp", lambda a: a.exp(), [normal(3, 4)]),
    ("log", lambda a: a.log(), [positive(3, 4)]),
    ("relu", lambda a: a.relu(), [away_from_zero(3, 4, low=0.1)]),
    ("tanh", lambda a: a.tanh(), [normal(3, 4)]),
    ("sigmoid", lambda a: a.sigmoid(), [normal(3, 4) * 3]),
    # loss: gradient w.r.t. logits
    ("softmax_xent", lambda z: softmax_cross_entropy(z, [0, 2, 1, 2, 0]), [normal(5, 3)]),
    ("softmax_xent_large_logits", lambda z: softmax_cross_entropy(z, [1, 0, 3]), [normal(3, 4) * 20]),
    # composition: a full 2-layer MLP loss, gradients w.r.t. every parameter
    (
        "mlp_2layer",
        lambda x, w1, b1, w2, b2: softmax_cross_entropy(((x @ w1 + b1).tanh()) @ w2 + b2, [0, 1, 2, 1]),
        [normal(4, 5), normal(5, 6), normal(6), normal(6, 3), normal(3)],
    ),
    # a tensor reused along several paths (gradient accumulation)
    ("reuse", lambda a: (a * a + a.sigmoid() * a).sum(axis=0), [normal(3, 4)]),
]


@pytest.mark.parametrize("name,f,inputs", CASES, ids=[c[0] for c in CASES])
def test_gradcheck(name, f, inputs):
    err = gradcheck(f, *inputs)
    assert err < 1e-6


def test_gradcheck_catches_a_wrong_gradient():
    """The checker itself must fail on a deliberately broken op (no false passes)."""
    from engine.tensor import Tensor

    def bad_square(a):
        return Tensor._make(a.data ** 2, (a,), lambda g: (g * a.data,), "bad_square")  # missing factor 2

    with pytest.raises(AssertionError):
        gradcheck(bad_square, normal(3))


def test_engine_unbroadcasts_for_ops_that_do_not():
    """A raw op whose backward returns the full (N, D) gradient for a (D,) parent
    is still correct, because backward() sums over broadcast axes generically --
    so no individual op can forget the reduction."""
    from engine.tensor import Tensor

    def naive_add(a, b):
        out = Tensor(a.data + b.data, requires_grad=True)
        out._parents, out._backward_fn = (a, b), lambda g: (g, g)
        return out

    assert gradcheck(naive_add, normal(5, 3), normal(3)) < 1e-6


def test_relative_error_is_scale_free():
    a = np.array([1e-3, 2e-3])
    assert relative_error(a, a * (1 + 1e-9)) < 1e-8
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0
