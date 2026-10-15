"""Composable layers: Module base class, Linear, activations, Sequential, MLP.

Initialisation
--------------
For y = x W with W_ij ~ N(0, σ²) and independent zero-mean inputs,
Var(y_j) = fan_in · σ² · E[x²].

* Hidden layers feed a ReLU. ReLU zeroes half of a symmetric pre-activation,
  so E[relu(z)²] = Var(z) / 2. Keeping E[x²] constant from layer to layer
  therefore needs fan_in · σ² / 2 = 1, i.e. **He: σ = sqrt(2 / fan_in)**.
  Smaller σ makes activations (and gradients) shrink geometrically with depth;
  larger makes them explode.
* The output layer produces logits that go straight into softmax, with no
  ReLU halving, so we use **Xavier/Glorot: σ = sqrt(2 / (fan_in + fan_out))**,
  which balances forward and backward variance and keeps the initial logits
  O(1). For logits z with small i.i.d. variance s², a second-order expansion
  of log-sum-exp gives E[loss] ≈ log(C) + s²/2 (minus a small (C-1)/C factor),
  so the network starts near the uniform-prediction loss log(C) instead of
  being confidently wrong. (For a 64-wide last hidden layer s² ≈ 1.7, so the
  start is ~log(C) + 0.8 nats; a scaled-down output layer would get closer
  to log(C) but would also make output weights look "unimportant" to a
  magnitude criterion early on, so we keep Xavier.)
* tanh/sigmoid hidden layers are near-linear around 0 (no halving), so they
  also get Xavier.
* Biases start at 0: the random weights already break symmetry.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from engine.tensor import Tensor
from nn.parameter import Parameter


class Module:
    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        seen: set[int] = set()
        for p in self._walk():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        return params

    def _walk(self) -> Iterator[Parameter]:
        for value in vars(self).values():
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                if isinstance(item, Parameter):
                    yield item
                elif isinstance(item, Module):
                    yield from item._walk()

    def prunable_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters() if p.prunable]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x) -> Tensor:
        return self.forward(x)

    def forward(self, x) -> Tensor:
        raise NotImplementedError


def init_weight(fan_in: int, fan_out: int, scheme: str, rng: np.random.Generator) -> np.ndarray:
    if scheme == "he":
        std = np.sqrt(2.0 / fan_in)
    elif scheme == "xavier":
        std = np.sqrt(2.0 / (fan_in + fan_out))
    else:
        raise ValueError(f"unknown init scheme {scheme!r}")
    return rng.normal(0.0, std, size=(fan_in, fan_out))


class Linear(Module):
    """y = x @ (W ⊙ M) + b.  W has shape (in, out); only W is prunable."""

    def __init__(self, in_features: int, out_features: int, init: str = "he", rng: np.random.Generator | None = None):
        rng = rng if rng is not None else np.random.default_rng()
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(init_weight(in_features, out_features, init, rng), prunable=True)
        self.bias = Parameter(np.zeros(out_features), prunable=False)

    def forward(self, x) -> Tensor:
        return x @ self.weight.masked() + self.bias

    def __repr__(self) -> str:
        return f"Linear({self.in_features}, {self.out_features}, density={self.weight.density:.3f})"


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


ACTIVATIONS = {"relu": (ReLU, "he"), "tanh": (Tanh, "xavier")}


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"{type(self).__name__}(\n  {inner}\n)"


class MLP(Sequential):
    """Linear -> act -> ... -> Linear (logits). ``sizes`` = [in, hidden..., out]."""

    def __init__(self, sizes: Sequence[int], activation: str = "relu", seed: int = 0):
        if len(sizes) < 2:
            raise ValueError("MLP needs at least an input and an output size")
        act_cls, hidden_init = ACTIVATIONS[activation]
        rng = np.random.default_rng(seed)
        layers: list[Module] = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            is_last = i == len(sizes) - 2
            layers.append(Linear(fan_in, fan_out, init="xavier" if is_last else hidden_init, rng=rng))
            if not is_last:
                layers.append(act_cls())
        super().__init__(*layers)
        self.sizes = list(sizes)

    def linears(self) -> list[Linear]:
        return [layer for layer in self.layers if isinstance(layer, Linear)]

    def num_prunable(self) -> int:
        return sum(p.size for p in self.prunable_parameters())

    def num_active(self) -> int:
        return sum(p.num_active for p in self.prunable_parameters())

    def sparsity(self) -> float:
        """Fraction of prunable connections (weights, not biases) that are removed."""
        return 1.0 - self.num_active() / self.num_prunable()
