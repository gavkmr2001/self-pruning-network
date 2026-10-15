"""Hand-derived (analytic) checks for matmul, reductions, non-linearities and the loss."""
import numpy as np
import pytest

from engine.functional import log_softmax, softmax, softmax_cross_entropy
from engine.tensor import Tensor

rng = np.random.default_rng(0)


def test_matmul_gradients_match_closed_form():
    A = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    B = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    G = rng.normal(size=(4, 2))
    (A @ B).backward(G)
    assert np.allclose(A.grad, G @ B.data.T)
    assert np.allclose(B.grad, A.data.T @ G)


def test_linear_layer_with_broadcast_bias():
    # y = sum(X @ W + b): dy/dW = X^T @ 1, dy/db = N * 1
    N, D, H = 6, 4, 3
    X = Tensor(rng.normal(size=(N, D)))
    W = Tensor(rng.normal(size=(D, H)), requires_grad=True)
    b = Tensor(np.zeros(H), requires_grad=True)
    (X @ W + b).sum().backward()
    assert np.allclose(W.grad, X.data.T @ np.ones((N, H)))
    assert np.allclose(b.grad, np.full(H, float(N)))


def test_sum_and_mean_over_axes():
    x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    x.sum(axis=0).backward(np.arange(4.0))           # each row receives [0,1,2,3]
    assert np.allclose(x.grad, np.tile(np.arange(4.0), (3, 1)))
    x.zero_grad()
    x.mean(axis=1, keepdims=True).backward(np.ones((3, 1)))
    assert np.allclose(x.grad, np.full((3, 4), 0.25))
    x.zero_grad()
    x.mean().backward()
    assert np.allclose(x.grad, np.full((3, 4), 1 / 12))


def test_reshape_and_transpose_route_gradients_back():
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    y = x.T.reshape(6) * Tensor(np.arange(6.0))
    y.sum().backward()
    # x.T.reshape(6) lists x in column-major order: x00 x10 x01 x11 x02 x12
    assert np.allclose(x.grad, np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]))


def test_relu_tanh_sigmoid_derivatives():
    x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    x = Tensor(x_np, requires_grad=True)
    x.relu().sum().backward()
    assert np.array_equal(x.grad, np.array([0.0, 0.0, 0.0, 1.0, 1.0]))  # relu'(0) := 0

    x.zero_grad()
    x.tanh().sum().backward()
    assert np.allclose(x.grad, 1 - np.tanh(x_np) ** 2)

    x.zero_grad()
    x.sigmoid().sum().backward()
    s = 1 / (1 + np.exp(-x_np))
    assert np.allclose(x.grad, s * (1 - s))


def test_sigmoid_is_stable_for_huge_inputs():
    with np.errstate(over="raise"):
        out = Tensor(np.array([-1000.0, 1000.0])).sigmoid().data
    assert np.allclose(out, [0.0, 1.0])


def test_log_softmax_is_stable_and_normalised():
    z = np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        lp = log_softmax(z)
    assert np.all(np.isfinite(lp))
    assert np.allclose(softmax(z).sum(axis=1), 1.0)
    assert np.allclose(lp[0], [0.0, -1000.0, -2000.0])


def test_softmax_cross_entropy_value_and_gradient():
    z = rng.normal(size=(5, 4))
    y = np.array([0, 3, 1, 1, 2])
    logits = Tensor(z, requires_grad=True)
    loss = softmax_cross_entropy(logits, y)

    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    assert np.isclose(loss.data, -np.log(p[np.arange(5), y]).mean())

    loss.backward()
    onehot = np.eye(4)[y]
    assert np.allclose(logits.grad, (p - onehot) / 5)


def test_uniform_logits_give_log_num_classes():
    loss = softmax_cross_entropy(Tensor(np.zeros((8, 10))), np.zeros(8, dtype=int))
    assert np.isclose(loss.data, np.log(10))


def test_extreme_logits_do_not_produce_nan():
    logits = Tensor(np.array([[1e4, -1e4], [-1e4, 1e4]]), requires_grad=True)
    loss = softmax_cross_entropy(logits, [1, 1])  # first row is confidently wrong
    loss.backward()
    assert np.isfinite(loss.data) and np.isclose(loss.data, 1e4)  # (2e4 + 0) / 2
    assert np.all(np.isfinite(logits.grad))


def test_matmul_rejects_vectors():
    with pytest.raises(ValueError):
        Tensor(np.ones(3)) @ Tensor(np.ones((3, 2)))
