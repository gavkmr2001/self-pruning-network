"""Data pipeline and training loop."""
import numpy as np

from engine.tensor import Tensor, no_grad
from nn import MLP
from nn.optim import Adam
from train.data import Split, load_digits_dataset, minibatches, stratified_split
from train.loop import evaluate, train


def test_stratified_split_is_disjoint_complete_and_balanced():
    y = np.repeat(np.arange(4), [50, 30, 20, 100])
    tr, va, te = stratified_split(y, seed=0)
    all_idx = np.concatenate([tr, va, te])
    assert len(all_idx) == len(y) and len(np.unique(all_idx)) == len(y)
    for part, frac in ((tr, 0.6), (va, 0.2), (te, 0.2)):
        counts = np.bincount(y[part], minlength=4)
        assert np.allclose(counts, frac * np.bincount(y), atol=1)


def test_digits_is_standardised_with_train_statistics_only():
    data = load_digits_dataset()
    assert data.num_features == 64 and data.num_classes == 10
    assert len(data.train) + len(data.val) + len(data.test) == 1797
    assert np.allclose(data.train.X.mean(axis=0), 0, atol=1e-9)
    assert not np.allclose(data.test.X.mean(axis=0), 0, atol=1e-3)   # test stats were not used
    assert np.all(np.isfinite(data.train.X))


def test_minibatches_cover_every_sample_once():
    split = Split(np.arange(10.0).reshape(10, 1), np.arange(10))
    batches = list(minibatches(split, 4, np.random.default_rng(0)))
    assert [len(b[1]) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate([b[1] for b in batches])) == list(range(10))


def test_no_grad_builds_no_graph():
    w = Tensor(np.ones((2, 2)), requires_grad=True)
    with no_grad():
        out = Tensor(np.ones((1, 2))) @ w
    assert not out.requires_grad and out._parents == ()
    assert (Tensor(np.ones((1, 2))) @ w).requires_grad   # re-enabled afterwards


def test_training_is_reproducible_and_learns():
    data = load_digits_dataset()

    def run():
        model = MLP([64, 32, 10], seed=0)
        hist = train(model, data, Adam(model.parameters(), lr=1e-2), epochs=3, seed=0, verbose=False)
        return hist, model

    h1, m1 = run()
    h2, m2 = run()
    assert [h["train_loss"] for h in h1] == [h["train_loss"] for h in h2]
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert np.array_equal(p1.data, p2.data)
    assert h1[-1]["val_acc"] > 0.85
    assert h1[-1]["train_loss"] < h1[0]["train_loss"]


def test_on_step_hook_sees_gradients_before_the_update():
    data = load_digits_dataset()
    model = MLP([64, 16, 10], seed=0)
    calls = []

    def hook(step, total):
        W = model.linears()[0].weight
        calls.append((step, total, W.grad is not None, W.dense_grad is not None))

    train(model, data, Adam(model.parameters()), epochs=1, batch_size=256, on_step=hook, verbose=False)
    assert len(calls) == 5                      # ceil(1078 / 256)
    assert all(total == 5 and g and dg for _, total, g, dg in calls)
    assert [c[0] for c in calls] == list(range(5))


def test_evaluate_matches_manual_accuracy():
    data = load_digits_dataset()
    model = MLP([64, 16, 10], seed=0)
    _, acc = evaluate(model, data.val, batch_size=100)    # batching must not change the result
    pred = model(Tensor(data.val.X)).data.argmax(axis=1)
    assert acc == (pred == data.val.y).mean()
