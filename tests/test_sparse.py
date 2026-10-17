"""Sparse inference path: CSR correctness, MAC accounting, exact neuron compaction."""
import numpy as np
import pytest

from nn import MLP
from prune.sparse import CSRLayer, SparseMLP, compact, dense_forward, dense_macs_per_sample

rng = np.random.default_rng(0)


def random_sparse(shape, density):
    mask = rng.random(shape) < density
    return rng.normal(size=shape) * mask, mask


@pytest.mark.parametrize("density", [1.0, 0.3, 0.05, 0.0])
def test_csr_layer_matches_dense_matmul(density):
    W, mask = random_sparse((7, 5), density)
    b = rng.normal(size=5)
    layer = CSRLayer.from_dense(W, b, mask)
    X = rng.normal(size=(11, 7))
    assert np.allclose(layer.forward(X), X @ W + b, atol=1e-12)
    assert np.array_equal(layer.to_dense(), W)
    assert layer.nnz == mask.sum()


def test_csr_handles_empty_output_units():
    W = np.zeros((4, 3))
    W[1, 0], W[3, 2] = 2.0, -1.0           # output unit 1 has no inputs at all
    layer = CSRLayer.from_dense(W, np.array([0.5, 0.25, 0.0]))
    X = rng.normal(size=(6, 4))
    assert np.allclose(layer.forward(X), X @ W + [0.5, 0.25, 0.0])


def test_mask_not_values_defines_sparsity():
    """A live weight that happens to be exactly 0.0 is still stored (mask wins);
    a pruned slot is never stored even if storage were corrupted."""
    W = np.array([[0.0, 1.0], [2.0, 3.0]])
    mask = np.array([[True, True], [False, True]])
    layer = CSRLayer.from_dense(W * mask, np.zeros(2), mask)
    assert layer.nnz == 3


def test_sparse_mlp_equals_masked_model_forward():
    model = MLP([10, 16, 12, 4], seed=0)
    for p in model.prunable_parameters():
        p.set_mask(rng.random(p.shape) < 0.2)
    sp = SparseMLP.from_model(model)
    X = rng.normal(size=(9, 10))
    from engine.tensor import Tensor, no_grad
    with no_grad():
        ref = model(Tensor(X)).data
    assert np.allclose(sp.forward(X), ref, atol=1e-12)
    assert sp.macs_per_sample() == model.num_active()


def test_mac_and_byte_accounting():
    shapes = [(64, 128), (128, 10)]
    assert dense_macs_per_sample(shapes) == 64 * 128 + 128 * 10
    W, mask = random_sparse((64, 128), 0.1)
    layer = CSRLayer.from_dense(W, np.zeros(128), mask)
    nnz = int(mask.sum())
    assert layer.macs_per_sample() == nnz
    assert layer.nbytes() == (128 + 1) * 8 + nnz * 4 + nnz * 8 + 128 * 8


def test_compaction_is_exact_and_removes_disconnected_units():
    W0, W1, W2 = rng.normal(size=(5, 6)), rng.normal(size=(6, 4)), rng.normal(size=(4, 3))
    b0, b1, b2 = rng.normal(size=6), rng.normal(size=4), rng.normal(size=3)
    b0[2] = 1.5                 # unit 2 of layer 0 will be input-dead with relu(b)=1.5 > 0
    W0[:, 2] = 0.0              # -> its constant activation must be folded into b1
    W1[4, :] = 0.0              # unit 4 of layer 0 is output-dead
    W1[:, 1] = 0.0              # unit 1 of layer 1 input-dead (bias may be negative)
    W, b = [W0, W1, W2], [b0, b1, b2]
    Wc, bc, removed = compact(W, b)
    X = rng.normal(size=(20, 5))
    assert np.allclose(dense_forward(Wc, bc, X), dense_forward(W, b, X), atol=1e-12)
    assert removed == [2, 1]
    assert [w.shape for w in Wc] == [(5, 4), (4, 3), (3, 3)]


def test_compaction_cascades_to_a_fixed_point():
    """Removing a unit can disconnect a unit in the previous layer."""
    W0 = rng.normal(size=(3, 2))
    W1 = np.array([[1.0, 0.0], [0.0, 0.0]])    # layer-0 unit 1 has no outputs
    W2 = np.array([[1.0], [0.0]])              # layer-1 unit 1 has no outputs
    b = [np.zeros(2), np.zeros(2), np.zeros(1)]
    Wc, bc, removed = compact([W0, W1, W2], b)
    X = rng.normal(size=(5, 3))
    assert np.allclose(dense_forward(Wc, bc, X), dense_forward([W0, W1, W2], b, X))
    assert removed == [1, 1]
