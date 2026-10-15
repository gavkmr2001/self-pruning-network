"""A minimal reverse-mode automatic differentiation engine on top of NumPy.

Design
------
* A ``Tensor`` wraps a float64 ``np.ndarray`` and remembers the tensors it was
  computed from (``_parents``) plus a closure ``_backward_fn`` that maps the
  gradient of the output to the gradients of each parent (the local
  vector-Jacobian product).
* ``backward()`` builds a topological order of the graph *iteratively* (no
  recursion limit on deep graphs), then walks it in reverse, summing every
  gradient contribution that reaches a node before propagating it further.
* Broadcasting: if ``x`` of shape (D,) was broadcast to (N, D) in the forward
  pass, each of its elements was used N times, so its gradient is the *sum* of
  the N incoming gradients. ``_unbroadcast`` performs exactly that reduction.
* Only leaves (and tensors that call ``retain_grad()``) store ``.grad``.
  Intermediate gradients live in a dict for the duration of one backward pass.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.float64


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` over the axes that broadcasting created, returning ``shape``.

    Two cases, following NumPy's broadcasting rules:
      1. Leading axes that were prepended (e.g. (D,) -> (N, D)): sum them away.
      2. Axes of size 1 that were stretched (e.g. (N, 1) -> (N, D)): sum with
         keepdims so the size-1 axis is preserved.
    """
    if grad.shape == shape:
        return grad
    n_extra = grad.ndim - len(shape)
    if n_extra > 0:
        grad = grad.sum(axis=tuple(range(n_extra)))
    stretched = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


def as_tensor(x) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    # Makes `ndarray <op> Tensor` dispatch to Tensor's reflected operators.
    __array_priority__ = 1000

    def __init__(self, data, requires_grad: bool = False, _parents: tuple = (), _op: str = ""):
        self.data = np.asarray(data, dtype=DTYPE)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._parents = _parents
        self._backward_fn = None  # callable(out_grad) -> tuple of parent grads
        self._op = _op
        self._retain_grad = False

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def __repr__(self) -> str:
        op = f", op={self._op}" if self._op else ""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{op})"

    def numpy(self) -> np.ndarray:
        return self.data

    def zero_grad(self) -> None:
        self.grad = None

    def retain_grad(self) -> "Tensor":
        """Ask an intermediate (non-leaf) tensor to keep its gradient after backward()."""
        self._retain_grad = True
        return self

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    @staticmethod
    def _make(data, parents: tuple, backward_fn, op: str) -> "Tensor":
        """Create an op output; only record the graph if some parent needs grads."""
        needs_grad = any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs_grad, _parents=parents if needs_grad else (), _op=op)
        if needs_grad:
            out._backward_fn = backward_fn
        return out

    # ------------------------------------------------------- element-wise ops
    def __add__(self, other) -> "Tensor":
        other = as_tensor(other)
        return Tensor._make(self.data + other.data, (self, other), lambda g: (g, g), "add")

    def __sub__(self, other) -> "Tensor":
        other = as_tensor(other)
        return Tensor._make(self.data - other.data, (self, other), lambda g: (g, -g), "sub")

    def __mul__(self, other) -> "Tensor":
        other = as_tensor(other)
        a, b = self.data, other.data
        return Tensor._make(a * b, (self, other), lambda g: (g * b, g * a), "mul")

    def __truediv__(self, other) -> "Tensor":
        other = as_tensor(other)
        a, b = self.data, other.data
        # d(a/b)/da = 1/b ;  d(a/b)/db = -a/b^2
        return Tensor._make(a / b, (self, other), lambda g: (g / b, -g * a / (b * b)), "div")

    def __neg__(self) -> "Tensor":
        return Tensor._make(-self.data, (self,), lambda g: (-g,), "neg")

    def __pow__(self, p) -> "Tensor":
        if isinstance(p, Tensor):
            raise TypeError("Only constant (non-Tensor) exponents are supported.")
        a = self.data
        return Tensor._make(a ** p, (self,), lambda g: (g * p * a ** (p - 1),), f"pow{p}")

    # Reflected versions so that `2 * t`, `1 - t`, `1 / t` all work.
    def __radd__(self, other) -> "Tensor":
        return as_tensor(other) + self

    def __rsub__(self, other) -> "Tensor":
        return as_tensor(other) - self

    def __rmul__(self, other) -> "Tensor":
        return as_tensor(other) * self

    def __rtruediv__(self, other) -> "Tensor":
        return as_tensor(other) / self

    # ---------------------------------------------------------------- matmul
    def __matmul__(self, other) -> "Tensor":
        """C = A @ B.  dL/dA = G @ B^T,  dL/dB = A^T @ G.

        Leading (batch) dimensions follow NumPy broadcasting; ``backward`` then
        sums the gradient over any batch axes a parent was broadcast along.
        """
        other = as_tensor(other)
        a, b = self.data, other.data
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("matmul expects tensors with ndim >= 2; reshape vectors explicitly")

        def backward_fn(g):
            return g @ np.swapaxes(b, -1, -2), np.swapaxes(a, -1, -2) @ g

        return Tensor._make(a @ b, (self, other), backward_fn, "matmul")

    def __rmatmul__(self, other) -> "Tensor":
        return as_tensor(other) @ self

    # ------------------------------------------------------------ reductions
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        """Every input element contributes once to its output slot, so the
        gradient is the output gradient copied (broadcast) back to the input."""
        shape = self.shape

        def backward_fn(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)  # restore the reduced axes as size 1
            return (np.broadcast_to(g, shape),)

        return Tensor._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), backward_fn, "sum")

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        count = self.size if axis is None else int(np.prod([self.shape[i] for i in np.atleast_1d(axis)]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # -------------------------------------------------------- shape handling
    def reshape(self, *shape) -> "Tensor":
        old = self.shape
        return Tensor._make(self.data.reshape(*shape), (self,), lambda g: (g.reshape(old),), "reshape")

    def transpose(self, *axes) -> "Tensor":
        axes = axes or tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(axes))
        return Tensor._make(self.data.transpose(axes), (self,), lambda g: (g.transpose(inverse),), "transpose")

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # ---------------------------------------------------------- unary maths
    def exp(self) -> "Tensor":
        out = np.exp(self.data)
        return Tensor._make(out, (self,), lambda g: (g * out,), "exp")

    def log(self) -> "Tensor":
        a = self.data
        return Tensor._make(np.log(a), (self,), lambda g: (g / a,), "log")

    # ------------------------------------------------------ non-linearities
    def relu(self) -> "Tensor":
        # Subgradient convention: d relu(x)/dx at x == 0 is taken to be 0.
        positive = self.data > 0
        return Tensor._make(self.data * positive, (self,), lambda g: (g * positive,), "relu")

    def tanh(self) -> "Tensor":
        out = np.tanh(self.data)
        return Tensor._make(out, (self,), lambda g: (g * (1.0 - out * out),), "tanh")

    def sigmoid(self) -> "Tensor":
        # Branch-free stable form: never calls exp() on a large positive number.
        x = self.data
        e = np.exp(-np.abs(x))
        out = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        return Tensor._make(out, (self,), lambda g: (g * out * (1.0 - out),), "sigmoid")

    # ------------------------------------------------------------- backward
    def _topological_order(self) -> list["Tensor"]:
        """Parents-before-children ordering, computed iteratively (no recursion)."""
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self, grad=None) -> None:
        """Reverse-mode sweep. Leaf gradients ACCUMULATE across calls (call zero_grad)."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        grads: dict[int, np.ndarray] = {id(self): np.asarray(grad, dtype=DTYPE)}

        for node in reversed(self._topological_order()):
            g = grads.pop(id(node), None)
            if g is None:  # no path from this node to the output
                continue
            is_leaf = node._backward_fn is None
            if is_leaf or node._retain_grad:
                node.grad = g.copy() if node.grad is None else node.grad + g
            if is_leaf:
                continue
            for parent, pg in zip(node._parents, node._backward_fn(g)):
                if pg is None or not parent.requires_grad:
                    continue
                pg = _unbroadcast(np.asarray(pg, dtype=DTYPE), parent.shape)
                key = id(parent)
                grads[key] = pg if key not in grads else grads[key] + pg
