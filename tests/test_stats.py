"""Statistics helpers used for the Part 4 baseline comparison."""
import numpy as np

from train.stats import bootstrap_ci, holm, paired_permutation_test


def test_permutation_test_exact_small_case():
    # d = [1, 1, 1]: of the 8 sign patterns only all-plus and all-minus reach |mean| = 1
    assert np.isclose(paired_permutation_test([1.0, 1.0, 1.0]), 2 / 8)


def test_permutation_test_no_effect_and_strong_effect():
    assert paired_permutation_test([0.0] * 10) == 1.0
    assert paired_permutation_test([0.5, -0.5] * 5) == 1.0
    assert paired_permutation_test(np.full(10, 0.01)) == 2 / 1024   # smallest possible with n = 10


def test_permutation_test_is_calibrated_under_the_null():
    rng = np.random.default_rng(0)
    pvals = [paired_permutation_test(rng.normal(size=10)) for _ in range(300)]
    assert 0.02 < np.mean(np.array(pvals) < 0.05) < 0.09   # ~5% false positives


def test_bootstrap_ci_contains_true_mean_and_shrinks_with_n():
    rng = np.random.default_rng(1)
    small, large = rng.normal(2.0, 1.0, 10), rng.normal(2.0, 1.0, 1000)
    lo_s, hi_s = bootstrap_ci(small)
    lo_l, hi_l = bootstrap_ci(large)
    assert lo_l < 2.0 < hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> monotone 0.09, 0.5*1=0.5
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])
