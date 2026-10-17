"""Sparse-aware inference: a forward pass whose work is proportional to the number
of SURVIVING weights, not a dense matmul that multiplies by zeros.

Storage: CSR (compressed sparse row) of Wᵀ
------------------------------------------
A layer computes y = x @ W with W of shape (in, out). We store Wᵀ (out, in) in
CSR form -- one "row" per OUTPUT unit listing only its live inputs:

    indptr  (out+1,)  row j's entries live at positions indptr[j] : indptr[j+1]
    indices (nnz,)    input index of each live weight
    values  (nnz,)    the weight itself

    y[:, j] = sum_k  x[:, indices[k]] * values[k]     for k in row j

Kernel: activations are kept feature-major, H = Xᵀ of shape (in, N), so the
gather H[indices] copies whole contiguous rows (gathering columns of X was
1.4-1.7x slower at large batch: strided memory access). Multiply each gathered
row by its weight, then sum each output unit's segment with np.add.reduceat.
Multiply-adds = N · nnz exactly, and the kernel's time scales linearly with
nnz (see scripts/benchmark_cost.py).

Why it still loses to BLAS here: NumPy must materialise the (nnz, N) product
array in memory and read it back for the reduction -- two full memory passes
per layer -- whereas a compiled sparse kernel fuses gather, multiply and add in
registers, and dense BLAS uses cache blocking and SIMD on contiguous data.

Honesty about speed
-------------------
"Fewer multiplies" is not automatically "faster". A dense BLAS matmul is
blocked, vectorised and cache-friendly; an unstructured sparse kernel does
gathers with irregular memory access and index overhead per non-zero. For
small layers like ours, dense BLAS is often faster in wall-clock even at 90%
sparsity. We therefore report three things separately: (1) MACs and bytes
(hardware-independent), (2) our sparse kernel's time vs density (does the
work really shrink?), (3) wall-clock vs NumPy's dense BLAS (is it actually
faster here?). The dense-times-mask forward used during training is
explicitly NOT counted as a speed-up: it does the full dense work.

Neuron compaction (exact, structured)
-------------------------------------
Unstructured pruning can leave whole hidden units disconnected:

* no live INCOMING weights: its pre-activation is the constant b_j, so its
  activation relu(b_j) is a constant. Fold relu(b_j) · W_next[j, :] into the
  next layer's bias and delete the unit.
* no live OUTGOING weights: it cannot affect anything downstream; delete it.

Both are exact rewrites (outputs unchanged to float precision) and turn
unstructured sparsity into smaller DENSE matrices, which BLAS does run faster.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CSRLayer:
    """y = x @ W + b with W stored as CSR of Wᵀ (rows = output units)."""

    indptr: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    bias: np.ndarray
    in_features: int
    out_features: int

    @classmethod
    def from_dense(cls, W: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> "CSRLayer":
        live = (W != 0.0) if mask is None else mask.astype(bool)
        WT, liveT = W.T, live.T
        counts = liveT.sum(axis=1)
        indptr = np.zeros(W.shape[1] + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        rows, cols = np.nonzero(liveT)            # row-major order: grouped by output unit
        return cls(indptr, cols.astype(np.int32), WT[rows, cols].astype(np.float64),
                   np.asarray(b, dtype=np.float64).copy(), W.shape[0], W.shape[1])

    @property
    def nnz(self) -> int:
        return int(self.values.size)

    def to_dense(self) -> np.ndarray:
        W = np.zeros((self.in_features, self.out_features))
        for j in range(self.out_features):
            s, e = self.indptr[j], self.indptr[j + 1]
            W[self.indices[s:e], j] = self.values[s:e]
        return W

    def forward_t(self, H: np.ndarray) -> np.ndarray:
        """Feature-major forward: H is (in, N), returns (out, N)."""
        out = np.empty((self.out_features, H.shape[1]))
        out[:] = self.bias[:, None]
        if self.nnz == 0:
            return out
        products = H[self.indices] * self.values[:, None]       # (nnz, N): the N·nnz multiplies
        nonempty = np.flatnonzero(np.diff(self.indptr) > 0)
        # reduceat sums products[indptr[j]:indptr[j+1]] for each non-empty output unit j
        out[nonempty] += np.add.reduceat(products, self.indptr[nonempty], axis=0)
        return out

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.forward_t(np.ascontiguousarray(X.T)).T

    def macs_per_sample(self) -> int:
        return self.nnz

    def nbytes(self) -> int:
        return self.indptr.nbytes + self.indices.nbytes + self.values.nbytes + self.bias.nbytes


class SparseMLP:
    """Inference-only MLP: CSR layers with ReLU between them (no autodiff graph)."""

    def __init__(self, layers: list[CSRLayer]):
        self.layers = layers

    @classmethod
    def from_arrays(cls, weights, biases, masks=None) -> "SparseMLP":
        masks = masks if masks is not None else [None] * len(weights)
        return cls([CSRLayer.from_dense(W, b, M) for W, b, M in zip(weights, biases, masks)])

    @classmethod
    def from_model(cls, model) -> "SparseMLP":
        lins = model.linears()
        return cls.from_arrays([l.weight.data for l in lins], [l.bias.data for l in lins],
                               [l.weight.mask for l in lins])

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.ascontiguousarray(X.T)          # transpose once; stay feature-major inside
        for i, layer in enumerate(self.layers):
            h = layer.forward_t(h)
            if i < len(self.layers) - 1:
                np.maximum(h, 0.0, out=h)
        return h.T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)

    def macs_per_sample(self) -> int:
        return sum(layer.macs_per_sample() for layer in self.layers)

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers)

    def dense_weights(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return [l.to_dense() for l in self.layers], [l.bias.copy() for l in self.layers]


def dense_macs_per_sample(shapes) -> int:
    return int(sum(i * o for i, o in shapes))


def dense_forward(weights, biases, X: np.ndarray) -> np.ndarray:
    """Reference dense forward with NumPy/BLAS matmul (what an unpruned model costs)."""
    h = X
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W + b
        if i < len(weights) - 1:
            h = np.maximum(h, 0.0)
    return h


def compact(weights: list[np.ndarray], biases: list[np.ndarray]) -> tuple[list, list, list[int]]:
    """Remove hidden units that are disconnected (no live inputs or no live outputs).

    Exact for ReLU MLPs: see module docstring. Iterates to a fixed point because
    removing a unit can disconnect another. Returns new weights, biases and the
    number of units removed per hidden layer."""
    W = [w.copy() for w in weights]
    b = [x.copy() for x in biases]
    removed = [0] * (len(W) - 1)
    changed = True
    while changed:
        changed = False
        for l in range(len(W) - 1):              # hidden layer l = outputs of W[l], inputs of W[l+1]
            no_in = ~np.any(W[l] != 0.0, axis=0)
            no_out = ~np.any(W[l + 1] != 0.0, axis=1)
            drop = no_in | no_out
            if not drop.any():
                continue
            # constant activations of input-dead units flow into the next bias
            const = np.maximum(b[l][no_in & ~no_out], 0.0)
            b[l + 1] = b[l + 1] + const @ W[l + 1][no_in & ~no_out]
            keep = ~drop
            W[l], b[l] = W[l][:, keep], b[l][keep]
            W[l + 1] = W[l + 1][keep, :]
            removed[l] += int(drop.sum())
            changed = True
    return W, b, removed
