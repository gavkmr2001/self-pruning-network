"""Layers, initialisation, and the masked forward/backward contract of Parameter."""
import numpy as np
import pytest

from engine.functional import softmax_cross_entropy
from engine.gradcheck import relative_error
from engine.tensor import Tensor
from nn import MLP, Linear, Parameter

rng = np.random.default_rng(7)


def loss_of(model, X, y) -> Tensor:
    return softmax_cross_entropy(model(Tensor(X)), y)


def numerical_param_grad(model, X, y, p: Parameter, h=1e-6) -> np.ndarray:
    """Central differences w.r.t. the *stored* values of p (mask held fixed)."""
    g = np.zeros_like(p.data)
    for idx in np.ndindex(p.shape):
        orig = p.data[idx]
        p.data[idx] = orig + h
        plus = float(loss_of(model, X, y).data)
        p.data[idx] = orig - h
        minus = float(loss_of(model, X, y).data)
        p.data[idx] = orig
        g[idx] = (plus - minus) / (2 * h)
    return g


# ------------------------------------------------------------------ structure
def test_mlp_structure_and_parameter_discovery():
    model = MLP([4, 8, 6, 3], seed=0)
    assert [type(layer).__name__ for layer in model.layers] == ["Linear", "ReLU", "Linear", "ReLU", "Linear"]
    assert len(model.parameters()) == 6                  # 3 weights + 3 biases
    assert len(model.prunable_parameters()) == 3         # only weights are prunable
    assert model.num_prunable() == 4 * 8 + 8 * 6 + 6 * 3
    assert model.sparsity() == 0.0


def test_linear_forward_matches_numpy():
    layer = Linear(5, 3, rng=rng)
    layer.bias.data[:] = rng.normal(size=3)
    X = rng.normal(size=(4, 5))
    assert np.allclose(layer(Tensor(X)).data, X @ layer.weight.data + layer.bias.data)


def test_same_seed_same_model():
    a, b = MLP([4, 8, 3], seed=3), MLP([4, 8, 3], seed=3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert np.array_equal(pa.data, pb.data)


def test_mlp_parameter_gradients_match_finite_differences():
    model = MLP([5, 7, 4], activation="tanh", seed=1)
    for p in model.parameters():
        p.data += 0.1 * rng.normal(size=p.shape)          # make biases non-zero too
    X, y = rng.normal(size=(6, 5)), rng.integers(0, 4, size=6)
    model.zero_grad()
    loss_of(model, X, y).backward()
    for p in model.parameters():
        assert relative_error(p.grad, numerical_param_grad(model, X, y, p)) < 1e-6


# ---------------------------------------------------------------- init
def test_he_init_preserves_activation_scale_through_depth():
    """With He init the mean squared activation stays O(1) after 10 ReLU layers;
    with a naive N(0, 0.01) init it collapses towards 0."""
    width, depth = 256, 10
    X = Tensor(rng.normal(size=(512, width)))
    he = MLP([width] * (depth + 1), seed=0)
    h = X
    for layer in he.layers:
        h = layer(h)
    ratio = (h.data ** 2).mean() / (X.data ** 2).mean()
    assert 0.2 < ratio < 5.0

    tiny = MLP([width] * (depth + 1), seed=0)
    for p in tiny.prunable_parameters():
        p.data[:] = np.random.default_rng(0).normal(0, 0.01, size=p.shape)
    h = X
    for layer in tiny.layers:
        h = layer(h)
    assert (h.data ** 2).mean() < 1e-10


def test_initial_loss_matches_logit_variance_prediction():
    """E[loss] ≈ log C + s²/2 · (C-1)/C for i.i.d. logits of variance s² (random labels)."""
    C = 10
    model = MLP([20, 64, 64, C], seed=0)
    X, y = rng.normal(size=(4096, 20)), rng.integers(0, C, size=4096)
    logits = model(Tensor(X)).data
    s2 = logits.var(axis=1).mean() * C / (C - 1)
    predicted = np.log(C) + s2 / 2 * (C - 1) / C
    actual = float(softmax_cross_entropy(Tensor(logits), y).data)
    assert np.isfinite(actual)
    assert abs(actual - predicted) < 0.25
    assert actual < np.log(C) + 1.5   # not confidently wrong at step 0


# ------------------------------------------------------------ masking contract
def make_pruned_model():
    model = MLP([5, 6, 3], seed=2)
    W = model.linears()[0].weight
    mask = np.ones(W.shape, dtype=bool)
    mask[0, :] = False                 # prune every connection out of input 0
    mask[2, 3] = mask[4, 1] = False
    W.set_mask(mask)
    return model, W, mask


def test_set_mask_writes_real_zeros():
    _, W, mask = make_pruned_model()
    assert np.all(W.data[~mask] == 0.0)
    assert W.num_active == mask.sum()


def test_pruned_weight_has_no_effect_on_forward_even_if_storage_is_corrupted():
    model, W, mask = make_pruned_model()
    X = rng.normal(size=(4, 5))
    before = model(Tensor(X)).data.copy()
    W.data[~mask] = 123.0              # simulate a buggy optimizer writing into pruned slots
    assert np.array_equal(model(Tensor(X)).data, before)


def test_grad_of_pruned_weight_is_exactly_zero():
    model, W, mask = make_pruned_model()
    X, y = rng.normal(size=(8, 5)), rng.integers(0, 3, size=8)
    loss_of(model, X, y).backward()
    assert np.all(W.grad[~mask] == 0.0)
    assert np.any(W.grad[mask] != 0.0)


def test_grad_of_pruned_weight_matches_finite_differences():
    """Perturbing a pruned weight's storage does not change the loss, so its true
    derivative is 0; the finite-difference check confirms grad = G ⊙ M everywhere."""
    model, W, mask = make_pruned_model()
    X, y = rng.normal(size=(8, 5)), rng.integers(0, 3, size=8)
    loss_of(model, X, y).backward()
    assert relative_error(W.grad, numerical_param_grad(model, X, y, W)) < 1e-6


def test_dense_grad_is_the_gradient_a_revived_connection_would_receive():
    """dense_grad at a pruned entry equals the ordinary gradient that entry gets
    if it is un-masked at its current value 0 -- i.e. the regrowth signal."""
    model, W, mask = make_pruned_model()
    X, y = rng.normal(size=(8, 5)), rng.integers(0, 3, size=8)
    loss_of(model, X, y).backward()
    dense = W.dense_grad.copy()
    assert np.any(dense[~mask] != 0.0)
    assert np.allclose(dense[mask], W.grad[mask])

    model.zero_grad()
    W.set_mask(np.ones(W.shape, dtype=bool))     # revive at value 0: forward unchanged
    loss_of(model, X, y).backward()
    assert np.allclose(W.grad, dense)


def test_mask_change_between_forward_and_backward_does_not_leak():
    model, W, mask = make_pruned_model()
    X, y = rng.normal(size=(8, 5)), rng.integers(0, 3, size=8)
    loss = loss_of(model, X, y)
    W.mask = np.ones(W.shape, dtype=bool)        # mutate after forward
    loss.backward()
    assert np.all(W.grad[~mask] == 0.0)


def test_set_mask_rejects_wrong_shape():
    W = Parameter(np.ones((3, 2)), prunable=True)
    with pytest.raises(ValueError):
        W.set_mask(np.ones((2, 3), dtype=bool))


def test_sparsity_counts_weights_only():
    model, W, mask = make_pruned_model()
    removed = (~mask).sum()
    assert model.num_active() == model.num_prunable() - removed
    assert np.isclose(model.sparsity(), removed / model.num_prunable())
