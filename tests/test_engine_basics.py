"""Hand-derived (analytic) checks of the core engine: broadcasting, accumulation, ordering."""
import numpy as np

from engine.tensor import Tensor, _unbroadcast


def test_unbroadcast_prepended_and_stretched_axes():
    g = np.ones((4, 3))
    assert np.array_equal(_unbroadcast(g, (3,)), np.full(3, 4.0))          # (D,) -> (N, D)
    assert np.array_equal(_unbroadcast(g, (4, 1)), np.full((4, 1), 3.0))   # (N,1) -> (N, D)
    assert np.array_equal(_unbroadcast(g, (1, 3)), np.full((1, 3), 4.0))   # (1,D) -> (N, D)
    assert np.array_equal(_unbroadcast(np.ones((2, 4, 3)), ()), np.array(24.0))  # scalar


def test_bias_broadcast_add_backprop():
    # y = sum(X + b) with X:(N,D), b:(D,)  =>  dy/db_j = N, dy/dX_ij = 1
    N, D = 5, 3
    X = Tensor(np.random.default_rng(0).normal(size=(N, D)), requires_grad=True)
    b = Tensor(np.zeros(D), requires_grad=True)
    y = X + b
    y.backward(np.ones((N, D)))
    assert b.grad.shape == (D,)
    assert np.array_equal(b.grad, np.full(D, float(N)))
    assert np.array_equal(X.grad, np.ones((N, D)))


def test_gradient_accumulates_when_a_tensor_is_reused():
    # y = x*x + x  =>  dy/dx = 2x + 1  (x feeds three edges of the graph)
    x = Tensor(np.array([3.0, -2.0]), requires_grad=True)
    y = x * x + x
    y.backward(np.ones(2))
    assert np.allclose(x.grad, 2 * x.data + 1)


def test_diamond_graph():
    # a -> b = 2a, c = 3a -> d = b * c = 6a^2  =>  dd/da = 12a
    a = Tensor(np.array([1.5]), requires_grad=True)
    b, c = a * 2.0, a * 3.0
    (b * c).backward(np.ones(1))
    assert np.allclose(a.grad, 12 * a.data)


def test_leaf_grads_accumulate_across_backward_calls_until_zeroed():
    x = Tensor(np.array([2.0]), requires_grad=True)
    (x * 3.0).backward(np.ones(1))
    (x * 3.0).backward(np.ones(1))
    assert np.allclose(x.grad, [6.0])
    x.zero_grad()
    (x * 3.0).backward(np.ones(1))
    assert np.allclose(x.grad, [3.0])


def test_deep_graph_has_no_recursion_limit():
    x = Tensor(np.array([1.0]), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward(np.ones(1))
    assert np.allclose(x.grad, [1.0])


def test_constants_do_not_receive_gradients():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    c = Tensor(np.array([5.0, 7.0]))  # requires_grad=False
    (x * c).backward(np.ones(2))
    assert c.grad is None
    assert np.allclose(x.grad, c.data)


def test_division_and_reflected_ops():
    x = Tensor(np.array([2.0, 4.0]), requires_grad=True)
    y = 1.0 / x - 3.0 * x + (2.0 - x)   # dy/dx = -1/x^2 - 3 - 1
    y.backward(np.ones(2))
    assert np.allclose(x.grad, -1 / x.data**2 - 4.0)
