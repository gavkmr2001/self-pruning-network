# DESIGN

Answers to the four design questions, with pointers to the code, tests and committed numbers
that back each statement. Numbers come from `results/`; timings are from one laptop CPU
(single-threaded BLAS where stated) and are machine-dependent.

---

## 1. The importance criterion and why it approximates the loss change

**Goal.** Rank connections by ΔL_i = L(w with w_i := 0) − L(w), the loss increase from
removing connection i. Measuring it exactly costs one pass over the data per weight
(25,856 passes per pruning event), so we approximate it.

**Derivation.** Removing w_i is the perturbation δ = −w_i·e_i. A second-order Taylor
expansion of the loss around the current weights gives

```
ΔL_i ≈ g_i·δ_i + ½·H_ii·δ_i²  =  −g_i·w_i + ½·H_ii·w_i²                 (1)
```

with g = ∂L/∂w and H the Hessian (off-diagonal terms vanish because only one coordinate moves).

1. *The first-order term fades.* Training drives the expected gradient E[g_i] towards 0, so
   −E[g_i]·w_i becomes small and noisy exactly when pruning decisions matter most. A purely
   first-order score |g·w| is kept only as the `taylor1` ablation.
2. *Curvature from gradients we already have.* For cross-entropy, the Hessian is approximated
   by the Fisher information F = E[g·gᵀ] (exact when the model's predictive distribution
   matches the data; the standard Gauss–Newton/Fisher approximation otherwise). Its diagonal
   is F_ii = E[g_i²], which we estimate with a bias-corrected EMA (decay 0.9) of the squared
   gradient w.r.t. the effective weight.

Substituting into (1):

```
S_i = ½ · F_ii · w_i²        (prune/criteria.py::saliency)                (2)
```

the Optimal-Brain-Damage form with the Fisher in place of the Hessian. The same expression is
half the batch-averaged *squared* first-order term: E[(g_i·w_i)²] = w_i²·E[g_i²] = w_i²·F_ii
(the Molchanov et al. 2019 view), so the first-order noise is not ignored but turned into a
variance estimate.

**Why it beats magnitude.** |w_i| is (2) under the extra assumption that every weight has the
same curvature F_ii. Saliency drops it: a large weight on a pixel that is almost always 0 has
F_ii ≈ 0 and is cheap to remove; a small weight on an input every example uses is not. S_i is
also in *units of loss*, so scores from different layers are comparable and a single global
ranking is principled; |w| depends on each layer's fan-in and initialisation scale.

**Checked, not assumed.** We zeroed sampled weights one at a time and measured the true ΔL on
the training set (`tests/test_prune.py::test_saliency_predicts_true_loss_increase_better_than_magnitude`).
Spearman rank correlation with the measured ΔL, 64-128-128-10 network:

| trained for | magnitude | taylor1 | saliency |
|---|---|---|---|
| 3 epochs | 0.31 | 0.38 | **0.42** |
| 10 epochs | 0.46 | 0.55 | **0.60** |
| 40 epochs | 0.59 | 0.70 | **0.78** |

Caveat: on a narrow, still-underfit 64-32-10 network all three were ≈ 0.55. The gap appears
in the over-parameterised networks where pruning is worthwhile.

**Regrowth score.** A pruned weight has w_i = 0, so S_i = 0 by construction and it could never
return on saliency. For pruned weights we use the first-order rate of loss decrease if the
connection existed, |E[∂L/∂W_eff,i]|, from an EMA of the dense gradient (§2). Averaging rewards
consistently signed gradients over one noisy batch.

**Schedule.** Cubic ramp s(t) = s_f·(1 − (1 − progress)³) (Zhu & Gupta 2017), one event per
epoch from epoch 5 to 30, then 10 epochs with a fixed mask. Front-loaded: 87.5% of the pruning
is done by half-time, while redundancy is high and there is time to recover; gradual, so the
surviving weights adapt between events. One-shot pruning is the special case start = end.

**What the evidence says** (Part 4, 10 seeds, paired permutation test, Holm-corrected): no
difference from global magnitude up to 90% sparsity; at 98% saliency is +2.14 accuracy points
(Holm p = 0.035), and its test loss is lower on all 10 seeds at 95–98%. The pre-registered
primary test (accuracy at 95%, +0.70 points) is not significant after correction. Regrowth adds
no significant benefit on this task. See README §Part 4.

---

## 2. "The gradient of a masked weight" — what the engine computes and why

Layers never use the stored weight W directly; they use W_eff = W ⊙ M with a constant 0/1 mask
(`nn/parameter.py::Parameter.masked`). Let G = ∂L/∂W_eff. The chain rule gives two different,
both meaningful, quantities:

| | value | at a pruned entry | consumer |
|---|---|---|---|
| `param.grad` | ∂L/∂W = G ⊙ M | **exactly 0** | the optimizer |
| `param.dense_grad` | ∂L/∂W_eff = G | generally ≠ 0 | regrowth scoring |

**`grad` is the right thing for the optimizer** because it is the true derivative of the loss
with respect to the stored parameter: a pruned weight does not influence the loss, so its
derivative is 0 (finite differences confirm it: `tests/test_nn.py::test_grad_of_pruned_weight_matches_finite_differences`).
**`dense_grad` is the right thing for regrowth** because it answers "how would the loss move if
this connection existed?"; it equals the ordinary gradient the entry receives if un-masked at
value 0 (`test_dense_grad_is_the_gradient_a_revived_connection_would_receive`). Using G for
updates silently revives pruned weights; using G ⊙ M for regrowth means nothing can ever return.
The mask is captured by value at forward time, so a mask change between forward and backward
cannot leak into that step's gradients.

**A zero gradient does not freeze a weight under a stateful optimizer.** Momentum and Adam's
first moment keep producing updates after the gradient goes to 0, and Adam's slowly decaying
second moment v makes a revived weight's effective learning rate tiny for hundreds of steps. The
subtle failure: resetting m and v on revival but keeping the global step count t. Bias
correction 1/(1 − βᵗ) then assumes t steps of accumulation when there was one, so the first
update is [(1−β1)/(1−β1ᵗ)] / √[(1−β2)/(1−β2ᵗ)] times too large: ≈ 2× at t = 500, approaching
(1−β1)/√(1−β2) ≈ 3.16× for large t, and distorted for about 1/(1−β2) = 1,000 further steps.

**Our rule** (`nn/optim.py`): at every step, wherever M = 0, the weight is held at exactly 0.0
and its optimizer state (velocity, or m, v and a **per-element** step count) is reset to 0. Active
entries get textbook Adam with bias correction from their own count of live steps. Consequences,
each with a test in `tests/test_masked_optim.py`:

* a pruned weight stays exactly 0 every step despite accumulated momentum (Adam and SGD);
* a revived weight's trajectory equals a fresh Adam started at 0 with the same gradients, to
  1e-12, after 300 steps of history and 37 steps pruned;
* masking some entries never changes how other entries are updated;
* weight decay never touches pruned entries;
* contrast tests show the naive variants drift, stall, or over-step.

**One more ordering detail** (`prune/pruner.py`). Masks change inside the training hook, after
`backward()` and before `optimizer.step()`. The `grad` of a weight revived at that moment was
computed under the old mask and is 0; passing it to Adam would spend the weight's first
bias-corrected step on a fake zero (its first real step would be ≈ 0.74·lr). The pruner copies
`dense_grad` into `grad` at revived entries; by the property above that is exactly the gradient
the weight at 0 should receive (`test_revived_weight_gets_its_dense_gradient_for_the_first_adam_step`).

---

## 3. Where the autodiff engine bottlenecks, and how to optimise it

Measured with `uv run python -m scripts.profile_engine` (cProfile; `results/engine_profile.json`):

| configuration | total | forward | backward | optimizer | pruner |
|---|---|---|---|---|---|
| self-prune 64-128-128-10, batch 64, 40 ep | 0.57 s | 10% | 16% | **49%** | 21% |
| dense, batch 64 | 0.47 s | 12% | 19% | **64%** | — |
| dense, batch 512 | 0.12 s | 23% | 26% | 43% | — |
| dense 64-1024-1024-10, batch 64, 5 ep | 1.78 s | 7% | 10% | **83%** | — |

**Finding 1: the optimizer, not autodiff, dominates.** On a 1024×1024 parameter one Adam step
takes ~17 ms, and **8.9 ms of it is computing β1ᵗ and β2ᵗ as element-wise array powers** (the
price of per-element step counts). Keeping running products β^t ← β·β^t per element (reset to 1
on revival) does the same in 0.25 ms (35× faster). Rewriting the update with in-place ufuncs
brings the whole step to ~3 ms in a micro-benchmark (≈ 5×). Boolean-mask fancy indexing
(`x[active]`, which copies) costs another ~1.2×; computing densely and re-applying the mask is
cheaper at these sizes. *Not applied in the submission:* running products round differently from
`β**t` in the last bits, which would perturb every committed result; we chose bit-reproducibility
of the reported numbers and document the optimisation instead.

**Finding 2: per-node interpreter overhead dominates small ops.** A graph node costs ~1.5 µs
for forward plus backward (closure creation, `_unbroadcast` checks, dict bookkeeping, the
topological sort) versus ~0.2 µs for the raw NumPy op. With 8-row batches or tiny layers this
is the cost; at batch 512 the matmuls amortise it (forward+backward share rises to ~50%).
Remedies, in order of payoff:

1. **Trace once, replay.** The MLP's graph is identical every step. Record the op sequence
   and topological order once, then replay forward and backward without rebuilding closures or
   re-sorting (what `torch.compile` / JAX `jit` do).
2. **Fuse ops.** `x @ W_eff + b → relu` as one node with a hand-written backward (one VJP
   instead of four); the softmax-cross-entropy is already fused for this reason.
3. **Avoid materialising W ⊙ M every step.** Since stored pruned weights are invariantly 0,
   the forward can use W directly; the mask is then only needed in backward (G ⊙ M) and in the
   optimizer. That saves one full-size multiply and copy per layer per step, at the price of the
   defence-in-depth property tested in `test_pruned_weight_has_no_effect_on_forward_even_if_storage_is_corrupted`.
4. **float32** halves memory traffic (float64 is used here for gradient checking).

**Finding 3: the pruner's global ranking.** Each event concatenates and fully sorts all scores,
O(N log N). Only the top-k boundary matters, so `np.argpartition` (O(N)) suffices; events are
also infrequent (once per epoch), so this matters only for much larger N.

**At scale** the profile changes: matmuls dominate and are BLAS/GPU-bound, activation memory
for backward becomes the constraint (fix: recompute instead of storing, mixed precision), and
optimizer state is 2× the parameters (fix: fused optimizer kernels, sharded state).

---

## 4. Serving a self-pruned model in a multi-tenant inference service at scale

**Principle: pruning happens in training; serving loads a frozen artefact.** The mask is never
changed online. The export pipeline turns a trained checkpoint into the cheapest exact
representation and gates it on quality before any traffic sees it.

**1. Export and compile per model.**
- Apply exact compaction first: unstructured pruning disconnected 54 of 256 hidden units in our
  90% model, and removing them is an exact rewrite to smaller dense matrices (1.48× fewer MACs,
  measured 1.05–1.35× faster than dense; outputs identical).
- Then pick the kernel *by measurement*, per model and batch size. On our CPU benchmark the
  remaining unstructured sparsity did not pay: the CSR path does 10× fewer MACs and its time
  scales linearly with non-zeros, but it is 2.7–32× slower than dense BLAS. So at this size we
  would serve compacted dense. Unstructured CSR becomes worthwhile only at very high sparsity
  with compiled kernels (fused gather-multiply-accumulate). For speed on accelerators, retrain
  with **structured** constraints instead (whole neurons, blocks, or 2:4 N:M patterns that GPU
  sparse tensor cores execute natively) and quantise (int8) on top.
- Store versioned, checksummed artefacts (weights, masks/compacted shapes, kernel choice,
  metrics) in a model registry.

**2. Quality gate before rollout.** Re-evaluate the exported artefact on a held-out set and
require (a) accuracy within the agreed budget of the dense model and (b) bit-level or
tolerance-level agreement between the training-time masked forward and the serving kernel
(our benchmark asserts max |sparse − dense| < 1e-9 before timing anything). Then canary or
shadow traffic, comparing prediction agreement and latency against the current version, with
automatic rollback.

**3. Multi-tenancy is where sparsity pays even without a speed-up.** Many tenants means many
models. Memory, not FLOPs, decides how many models stay resident on a host: our 90% model needs
35 KB as CSR (5.9× less) or 141 KB compacted, versus 209 KB dense. Smaller models mean more
tenants per host and fewer cold loads, and hot models fit in CPU cache. Use per-tenant model
routing with an LRU cache of loaded models, bounded per host; place tenants by consistent
hashing so each host keeps its working set warm, and replicate hot tenants.

**4. Throughput at thousands of requests per second.**
- Compute is not the limit at this model size: the measured single-threaded dense forward is
  ~6 µs at batch 1 and ~28 µs at batch 64, i.e. hundreds of thousands to millions of samples per
  second per core. The budget goes to network I/O, (de)serialisation, scheduling and tail latency.
- **Dynamic batching** per model, flushing on size or a few-millisecond deadline to stay within
  the latency SLO. Batching amortises per-call overhead (our dense per-sample cost falls ~23×
  from batch 1 to 4096: 6.2 µs to 0.26 µs), and the batch-size–latency curve is measured per kernel.
- Stateless replicas behind a load balancer with autoscaling on queue depth and p99 latency;
  one worker process per core with pinned BLAS threads (we benchmark single-threaded for exactly
  this reason) to avoid oversubscription.
- **Isolation:** per-tenant rate limits and quotas, so one tenant's burst cannot starve others,
  plus admission control and load shedding under overload.

**5. Observability.** Per-tenant and per-model p50/p99 latency, throughput, batch fill rate,
cache hit rate and error rate. Prediction-distribution drift monitors, because a pruned model
can degrade differently under shift than its dense parent. Keep the dense model deployable as a
fallback.

---

## Limitations and next steps

- One small dataset (`digits`, 359 test images: 1 image = 0.28 points), a fixed split, one
  architecture. Next: Fashion-MNIST, varied splits, more seeds.
- Cost is measured on CPU with NumPy kernels; a compiled sparse kernel and structured (N:M)
  pruning are the obvious next experiments for wall-clock gains.
- The optimizer and per-node optimisations in §3 are measured but not applied, to keep the
  committed results bit-reproducible.
