"""Schedule, criteria and the pruner (Part 3)."""
import numpy as np
import pytest

from engine.functional import softmax_cross_entropy
from engine.tensor import Tensor, no_grad
from nn import MLP, Parameter
from nn.optim import Adam
from prune import CRITERIA, CubicSchedule, GradStats, Pruner, magnitude, regrowth_score, saliency
from train.data import load_digits_dataset
from train.loop import train


# ----------------------------------------------------------------- schedule
def test_cubic_schedule_endpoints_shape_and_monotonicity():
    s = CubicSchedule(0.9, start_step=100, end_step=300, every=10)
    assert s.sparsity_at(0) == 0.0 and s.sparsity_at(99) == 0.0
    assert s.sparsity_at(100) == 0.0
    assert np.isclose(s.sparsity_at(200), 0.9 * (1 - 0.5 ** 3))   # 87.5% of the way at half time
    assert s.sparsity_at(300) == 0.9 and s.sparsity_at(10_000) == 0.9
    values = [s.sparsity_at(t) for t in range(0, 400)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    increments = np.diff([s.sparsity_at(t) for t in range(100, 301, 10)])
    assert all(b <= a + 1e-12 for a, b in zip(increments, increments[1:]))   # front-loaded


def test_schedule_update_steps_include_the_end():
    s = CubicSchedule(0.5, start_step=10, end_step=35, every=10)
    assert [t for t in range(50) if s.is_update_step(t)] == [10, 20, 30, 35]


def test_one_shot_is_start_equals_end():
    s = CubicSchedule(0.8, start_step=50, end_step=50, every=1)
    assert s.sparsity_at(49) == 0.0 and s.sparsity_at(50) == 0.8
    assert [t for t in range(100) if s.is_update_step(t)] == [50]


# ----------------------------------------------------------------- criteria
def test_grad_stats_are_bias_corrected_averages():
    st = GradStats.zeros((2,), decay=0.9)
    for _ in range(3):
        st.update(np.array([2.0, -1.0]))
    g, g2 = st.corrected()
    assert np.allclose(g, [2.0, -1.0]) and np.allclose(g2, [4.0, 1.0])


def test_saliency_formula_and_its_difference_from_magnitude():
    p = Parameter(np.array([[2.0, 0.5]]), prunable=True)
    st = GradStats.zeros(p.shape, decay=0.9)
    st.update(np.array([[0.01, 1.0]]))   # big weight, tiny gradients | small weight, big gradients
    assert np.allclose(saliency(p, st), 0.5 * np.array([[0.01 ** 2 * 4.0, 1.0 * 0.25]]))
    assert magnitude(p, st)[0, 0] > magnitude(p, st)[0, 1]      # magnitude keeps the big weight
    assert saliency(p, st)[0, 0] < saliency(p, st)[0, 1]        # saliency keeps the busy one


def test_regrowth_score_uses_the_dense_gradient_mean():
    p = Parameter(np.zeros((1, 3)), prunable=True)
    st = GradStats.zeros(p.shape, decay=0.5)
    st.update(np.array([[1.0, -1.0, 3.0]]))
    st.update(np.array([[1.0, 1.0, -3.0]]))   # entry 1 and 2 flip sign: inconsistent
    score = regrowth_score(p, st)
    assert score[0, 0] > score[0, 1]


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def test_saliency_predicts_true_loss_increase_better_than_magnitude():
    """The derivation's claim, checked empirically: rank-correlate each criterion
    with the measured loss increase from zeroing single weights."""
    data = load_digits_dataset()
    model = MLP([64, 128, 128, 10], seed=0)   # the Part 3 architecture (see notes: narrow nets show less gap)
    params = model.prunable_parameters()
    stats = [GradStats.zeros(p.shape, 0.9) for p in params]

    def hook(step, total):
        for p, s in zip(params, stats):
            s.update(p.dense_grad)

    train(model, data, Adam(model.parameters(), lr=1e-3), epochs=10, on_step=hook, verbose=False)
    X, y = data.train.X, data.train.y

    def loss():
        with no_grad():
            return float(softmax_cross_entropy(model(Tensor(X)), y).data)

    base = loss()
    rng = np.random.default_rng(0)
    true, scores = [], {k: [] for k in CRITERIA}
    for p, st in zip(params, stats):
        sc = {k: f(p, st) for k, f in CRITERIA.items()}
        for j in rng.choice(p.size, 100, replace=False):
            w = p.data.flat[j]
            p.data.flat[j] = 0.0
            true.append(loss() - base)
            p.data.flat[j] = w
            for k in CRITERIA:
                scores[k].append(sc[k].flat[j])
    rho = {k: _spearman(np.array(v), np.array(true)) for k, v in scores.items()}
    assert rho["saliency"] > rho["magnitude"] + 0.05
    assert rho["saliency"] > 0.5


# ------------------------------------------------------------------ pruner
def _run(criterion="saliency", scope="global", regrow=0.0, sparsity=0.9, epochs=6, seed=0, min_density=0.01):
    data = load_digits_dataset()
    model = MLP([64, 32, 10], seed=seed)
    pruner = Pruner(model, CubicSchedule(sparsity, 17, 17 * 4, 17), criterion=criterion, scope=scope,
                    regrow_fraction=regrow, min_density=min_density)
    hist = train(model, data, Adam(model.parameters(), lr=3e-3), epochs=epochs, seed=seed,
                 on_step=pruner.on_step, verbose=False)
    return model, pruner, hist


@pytest.mark.parametrize("criterion", sorted(CRITERIA))
@pytest.mark.parametrize("scope", ["global", "layer"])
def test_pruner_hits_target_sparsity_exactly_with_real_zeros(criterion, scope):
    model, pruner, hist = _run(criterion=criterion, scope=scope)
    total = model.num_prunable()
    assert model.num_active() == total - round(0.9 * total)
    for p in model.prunable_parameters():
        assert np.all(p.data[~p.mask] == 0.0)
        if scope == "layer":
            assert p.num_active == p.size - round(0.9 * p.size)
    assert np.all(np.isfinite([h["train_loss"] for h in hist]))
    assert hist[-1]["val_acc"] > 0.6     # mechanism test (6 short epochs), not an accuracy benchmark


def test_without_regrowth_masks_only_shrink():
    data = load_digits_dataset()
    model = MLP([64, 32, 10], seed=0)
    pruner = Pruner(model, CubicSchedule(0.9, 0, 60, 10), criterion="saliency")
    seen = [m.copy() for m in (p.mask for p in model.prunable_parameters())]

    def hook(step, total):
        pruner.on_step(step, total)
        for i, p in enumerate(model.prunable_parameters()):
            assert not np.any(p.mask & ~seen[i])      # nothing revived
            seen[i] = p.mask.copy()

    train(model, data, Adam(model.parameters()), epochs=4, on_step=hook, verbose=False)
    assert all(e["grown"] == 0 for e in pruner.events)


def test_regrowth_revives_previously_pruned_weights_and_keeps_budget():
    model, pruner, _ = _run(regrow=0.3)
    grown_events = [e for e in pruner.events if e["grown"] > 0]
    assert grown_events, "regrowth never happened"
    assert pruner.events[-1]["grown"] == 0           # fraction has decayed to 0 at the end
    for e in pruner.events:
        assert np.isclose(e["achieved_sparsity"], round(e["target_sparsity"] * pruner.total) / pruner.total)


def test_grown_weights_were_pruned_before_the_event_not_during_it():
    data = load_digits_dataset()
    model = MLP([64, 32, 10], seed=0)
    pruner = Pruner(model, CubicSchedule(0.8, 0, 80, 20), criterion="saliency", regrow_fraction=0.5)
    params = model.prunable_parameters()

    def hook(step, total):
        before = [p.mask.copy() for p in params]
        pruner.on_step(step, total)
        if pruner.schedule.is_update_step(step):
            for p, old in zip(params, before):
                revived = p.mask & ~old
                assert not np.any(revived & old)         # revived ones were pruned before the event
                assert np.all(p.data[revived] == 0.0)    # revived at exactly 0: forward unchanged

    train(model, data, Adam(model.parameters()), epochs=5, on_step=hook, verbose=False)


def test_revived_weight_gets_its_dense_gradient_for_the_first_adam_step():
    """At the event step, autodiff computed grad under the OLD mask (0 at a revived
    entry). The pruner substitutes dense_grad, so the first Adam step of a revived
    weight is the textbook ±lr instead of a wasted zero-gradient step."""
    model = MLP([4, 3, 2], seed=0)
    W = model.linears()[0].weight
    mask = np.ones(W.shape, dtype=bool)
    mask[0, 0] = False
    W.set_mask(mask)
    pruner = Pruner(model, CubicSchedule(0.0, 0, 0, 1), criterion="magnitude", regrow_fraction=1.0, min_density=0.0)
    opt = Adam(model.parameters(), lr=1e-2)

    X = Tensor(np.random.default_rng(0).normal(size=(8, 4)))
    softmax_cross_entropy(model(X), [0, 1] * 4).backward()
    G = W.dense_grad[0, 0]
    assert W.grad[0, 0] == 0.0 and G != 0.0
    # sparsity 0 at step 0 with regrowth 1.0 -> everything active again
    pruner.on_step(0, 1)
    assert W.mask[0, 0] and W.grad[0, 0] == G
    opt.step()
    assert np.isclose(W.data[0, 0], -1e-2 * np.sign(G), rtol=1e-5)


def test_min_density_keeps_every_layer_connected():
    model, pruner, _ = _run(criterion="magnitude", sparsity=0.95, min_density=0.03)
    for p in model.prunable_parameters():
        assert p.num_active >= np.ceil(0.03 * p.size)


def test_infeasible_min_density_is_an_error_not_a_silent_violation():
    with pytest.raises(ValueError, match="min_density"):
        _run(criterion="magnitude", sparsity=0.99, min_density=0.05)


def test_pruning_is_deterministic():
    m1, p1, _ = _run(regrow=0.3, seed=1)
    m2, p2, _ = _run(regrow=0.3, seed=1)
    for a, b in zip(m1.prunable_parameters(), m2.prunable_parameters()):
        assert np.array_equal(a.mask, b.mask) and np.array_equal(a.data, b.data)


def test_unknown_criterion_is_rejected():
    with pytest.raises(ValueError):
        Pruner(MLP([2, 2], seed=0), CubicSchedule(0.5, 0, 10, 1), criterion="vibes")
