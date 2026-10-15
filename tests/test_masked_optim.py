"""Part 1 correctness requirement: masked weights under a stateful optimizer.

Each test drives a Parameter with hand-chosen gradients (``p.grad = ...``) so
the optimizer behaviour is isolated from the rest of the engine; the final
tests run the full forward/backward/step loop.

Contrast implementations (``naive_*``) reproduce the common mistakes so the
tests show not only that ours is right but that the alternatives are wrong.
"""
import numpy as np
import pytest

from engine.functional import softmax_cross_entropy
from engine.tensor import Tensor
from nn import MLP, Parameter
from nn.optim import SGD, Adam

LR, B1, B2, EPS = 1e-2, 0.9, 0.999, 1e-8


def reference_adam(grads, lr=LR, w0=0.0):
    """Textbook Adam on one scalar, from a fresh state. Returns the trajectory."""
    w, m, v, out = w0, 0.0, 0.0, []
    for t, g in enumerate(grads, start=1):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        w = w - lr * (m / (1 - B1 ** t)) / (np.sqrt(v / (1 - B2 ** t)) + EPS)
        out.append(w)
    return np.array(out)


def naive_adam_step(w, g, m, v, t, lr=LR):
    """Mask-unaware Adam with a global step count (what most quick versions do)."""
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    w = w - lr * (m / (1 - B1 ** t)) / (np.sqrt(v / (1 - B2 ** t)) + EPS)
    return w, m, v


def scalar_param(value=0.5):
    return Parameter(np.array([value]), prunable=True)


def prune(p):
    p.set_mask(np.zeros(p.shape, dtype=bool))


def revive(p):
    p.set_mask(np.ones(p.shape, dtype=bool))


def step_with(opt, p, g):
    """One optimizer step with gradient g as autodiff would deliver it (G ⊙ M)."""
    p.grad = np.where(p.mask, g, 0.0)
    opt.step()


# --------------------------------------------------------- reference behaviour
def test_adam_matches_textbook_on_dense_parameters():
    rng = np.random.default_rng(0)
    grads = rng.normal(size=50)
    p = scalar_param(0.3)
    opt = Adam([p], lr=LR)
    traj = []
    for g in grads:
        step_with(opt, p, g)
        traj.append(p.data[0])
    assert np.allclose(traj, reference_adam(grads, w0=0.3), rtol=0, atol=1e-12)


def test_sgd_momentum_matches_textbook():
    grads = [1.0, -0.5, 2.0, 0.25]
    p = scalar_param(1.0)
    opt = SGD([p], lr=0.1, momentum=0.9)
    w, v = 1.0, 0.0
    for g in grads:
        step_with(opt, p, g)
        v = 0.9 * v + g
        w -= 0.1 * v
        assert np.isclose(p.data[0], w)


# ------------------------------------------------------------ pruned = frozen
@pytest.mark.parametrize("make_opt", [lambda ps: Adam(ps, lr=LR), lambda ps: SGD(ps, lr=0.1, momentum=0.9)], ids=["adam", "sgd"])
def test_pruned_weight_stays_exactly_zero_despite_accumulated_momentum(make_opt):
    p = scalar_param(0.5)
    opt = make_opt([p])
    for _ in range(20):
        step_with(opt, p, 1.0)          # build up a large momentum
    prune(p)
    for _ in range(100):
        step_with(opt, p, 1.0)          # autodiff delivers 0 here
        assert p.data[0] == 0.0         # a real zero, every single step


def test_naive_adam_lets_a_pruned_weight_drift():
    """Contrast: relying on 'the gradient is zero' alone. The stored value walks
    away from 0 because the momentum m keeps producing updates."""
    w, m, v = 0.5, 0.0, 0.0
    for t in range(1, 21):
        w, m, v = naive_adam_step(w, 1.0, m, v, t)
    w = 0.0                             # "prune" by zeroing the value once
    for t in range(21, 41):
        w, m, v = naive_adam_step(w, 0.0, m, v, t)   # masked gradient is 0
    assert abs(w) > 0.05                # drifted far from the "pruned" 0


# ---------------------------------------------------------- revived = fresh
def test_revived_weight_follows_a_fresh_adam_trajectory_exactly():
    """After prune -> revive, the weight's future must be indistinguishable from a
    brand-new parameter initialised at 0 that sees the same gradients."""
    rng = np.random.default_rng(1)
    p = scalar_param(0.5)
    opt = Adam([p], lr=LR)
    for g in rng.normal(size=300) + 2.0:     # long history -> large global t, big m, v
        step_with(opt, p, g)
    prune(p)
    for _ in range(37):
        step_with(opt, p, 5.0)
    revive(p)

    future = rng.normal(size=40) - 1.0
    traj = []
    for g in future:
        step_with(opt, p, g)
        traj.append(p.data[0])
    assert np.allclose(traj, reference_adam(future, w0=0.0), rtol=0, atol=1e-12)


def test_first_step_after_revival_has_textbook_size():
    """A fresh Adam's first step is lr * g / (|g| + eps) ≈ lr * sign(g)."""
    p = scalar_param(0.5)
    opt = Adam([p], lr=LR)
    for _ in range(500):
        step_with(opt, p, 1.0)
    prune(p)
    step_with(opt, p, 0.0)
    revive(p)
    step_with(opt, p, -3.0)
    assert np.isclose(p.data[0], LR, rtol=1e-6)


@pytest.mark.parametrize("t_global", [500, 10_000])
def test_naive_reset_with_global_step_count_oversteps(t_global):
    """Contrast: zeroing m, v on revival but keeping the global t. The first
    step is then lr * [(1-β1)/(1-β1^t)] / sqrt[(1-β2)/(1-β2^t)] instead of lr:
    ≈ 1.98x at t=500, → (1-β1)/sqrt(1-β2) ≈ 3.16x as t grows."""
    w, _, _ = naive_adam_step(0.0, -3.0, 0.0, 0.0, t_global)
    ratio = w / LR
    expected = ((1 - B1) / (1 - B1 ** t_global)) / np.sqrt((1 - B2) / (1 - B2 ** t_global))
    assert np.isclose(ratio, expected, rtol=1e-6)
    assert ratio > 1.9


def test_naive_stale_second_moment_freezes_a_revived_weight():
    """Contrast: keeping stale v. After a history of large gradients, a revived
    weight receiving small gradients barely moves -- its effective LR collapses."""
    w, m, v = 0.5, 0.0, 0.0
    for t in range(1, 301):
        w, m, v = naive_adam_step(w, 100.0, m, v, t)
    w, m = 0.0, 0.0                       # prune, then revive 30 steps later with stale v
    t0 = 330
    for k in range(20):
        w, m, v = naive_adam_step(w, 1.0, m, v, t0 + k)
    fresh = reference_adam([1.0] * 20)[-1]
    assert abs(w) < 0.1 * abs(fresh)      # moved <10% as far as it should have


def test_sgd_revived_weight_starts_with_zero_velocity():
    p = scalar_param(1.0)
    opt = SGD([p], lr=0.1, momentum=0.9)
    for _ in range(30):
        step_with(opt, p, 1.0)
    prune(p)
    step_with(opt, p, 1.0)
    revive(p)
    step_with(opt, p, 2.0)
    assert np.isclose(p.data[0], -0.1 * 2.0)   # pure first step, no inherited velocity


# ------------------------------------------------ masking is element-local
def test_masking_some_entries_does_not_change_updates_of_others():
    rng = np.random.default_rng(2)
    init = rng.normal(size=(4, 5))
    grads = rng.normal(size=(30, 4, 5))
    mask = rng.random((4, 5)) > 0.5

    dense, sparse = Parameter(init.copy(), prunable=True), Parameter(init.copy(), prunable=True)
    sparse.set_mask(mask)
    od, os_ = Adam([dense], lr=LR), Adam([sparse], lr=LR)
    for g in grads:
        step_with(od, dense, g)
        step_with(os_, sparse, g)
    assert np.array_equal(sparse.data[mask], dense.data[mask])
    assert np.all(sparse.data[~mask] == 0.0)


def test_partial_revival_resets_only_revived_entries():
    p = Parameter(np.ones((2, 2)), prunable=True)
    opt = Adam([p], lr=LR)
    for _ in range(10):
        step_with(opt, p, np.ones((2, 2)))
    mask = np.array([[True, False], [True, True]])
    p.set_mask(mask)
    step_with(opt, p, np.ones((2, 2)))
    assert opt.t[0][0, 1] == 0 and opt.m[0][0, 1] == 0.0 and opt.v[0][0, 1] == 0.0
    assert np.all(opt.t[0][mask] == 11)
    p.set_mask(np.ones((2, 2), dtype=bool))
    step_with(opt, p, np.ones((2, 2)))
    assert opt.t[0][0, 1] == 1 and np.all(opt.t[0][mask] == 12)


def test_weight_decay_does_not_touch_pruned_entries():
    p = Parameter(np.full((3,), 2.0), prunable=True)
    p.set_mask(np.array([True, False, True]))
    for opt in (Adam([p], lr=LR, weight_decay=0.1), SGD([p], lr=0.1, weight_decay=0.1)):
        step_with(opt, p, np.zeros(3))
        assert p.data[1] == 0.0


# ------------------------------------------------- full engine integration
def test_end_to_end_training_with_prune_and_revive():
    """Real forward/backward/step: pruned weights stay 0 during training, the
    model still learns, and nothing becomes NaN across prune/revive events."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = MLP([6, 16, 2], seed=0)
    opt = Adam(model.parameters(), lr=1e-2)
    W = model.linears()[0].weight
    mask = rng.random(W.shape) > 0.5

    losses = []
    for step in range(300):
        if step == 100:
            W.set_mask(mask)
        if step == 200:
            W.set_mask(np.ones(W.shape, dtype=bool))
        opt.zero_grad()
        loss = softmax_cross_entropy(model(Tensor(X)), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
        if 100 <= step < 200:
            assert np.all(W.data[~mask] == 0.0)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < 0.5 * losses[0]
    assert np.all(opt.t[0][~mask] == 100)   # revived entries have lived exactly 100 steps
