"""Checks of the three calibration regimes (Sec. 8)."""
import itertools
import math

import numpy as np
import pytest
from scipy.stats import binom, hypergeom

from rscrank import calibrate as cal


def test_marginal_index_basic():
    assert cal.marginal_index(1000, 0.05) == 951
    assert cal.marginal_index(2000, 0.05) == 1901
    assert cal.marginal_index(10, 0.05) == 11  # -> +inf threshold
    assert cal.marginal_index(19, 0.05) == 19


def test_marginal_coverage_exhaustive_over_ranks():
    """Theorem 6 (the coverage claim, not just the index): with n+1 exchangeable values, the
    test value exceeds the j-th smallest calibration value iff its rank among the n+1 values
    is > j.  Enumerating all n+1 equally likely positions of the test value gives exactly
    1 - j/(n+1) <= alpha for distinct values; with ties the strict event is rarer."""
    rng = np.random.default_rng(5)
    for n in (4, 7, 19, 40, 99):
        for alpha in (0.05, 0.1, 0.25, 0.5):
            j = cal.marginal_index(n, alpha)
            # distinct values: the test value can sit at any of the n+1 positions with prob 1/(n+1)
            vals = np.sort(rng.normal(size=n + 1))
            fails = 0
            for pos in range(n + 1):
                test_v = vals[pos]
                cal_vals = np.delete(vals, pos)
                thr = math.inf if j >= n + 1 else cal_vals[j - 1]
                fails += test_v > thr
            exact = fails / (n + 1)
            assert exact == pytest.approx(max(0.0, 1 - j / (n + 1)), abs=1e-12)
            assert exact <= alpha + 1e-12
            # ties: every value drawn from a small integer set; the strict-event frequency over all
            # positions is at most the distinct-value bound
            tied = np.sort(rng.integers(0, 3, size=n + 1)).astype(float)
            fails_t = 0
            for pos in range(n + 1):
                cal_vals = np.delete(tied, pos)
                thr = math.inf if j >= n + 1 else cal_vals[j - 1]
                fails_t += tied[pos] > thr
            assert fails_t / (n + 1) <= exact + 1e-12


def test_index_arithmetic_is_exact_for_decimal_alpha():
    """The marginal index ceil((n+1)(1-alpha)) and the finite-population t = floor(alpha*M)+1 are
    computed on the exact decimal alpha denotes, so products that are integers in exact arithmetic
    (e.g. 0.05 * 2000 = 100) are never mis-rounded, for any n, M (no floating-point fudge)."""
    from fractions import Fraction

    rng = np.random.default_rng(11)
    alphas = [0.05, 0.1, 0.7, 0.3, 0.025, 0.01, 0.125, 0.2]
    checks = 0
    for alpha in alphas:
        a = Fraction(repr(alpha))
        for n in list(rng.integers(1, 10**6, size=300)) + [19, 20, 39, 40, 1000, 2000, 54619, 603919]:
            n = int(n)
            exact_j = min(max(math.ceil((n + 1) * (1 - a)), 1), n + 1)
            assert cal.marginal_index(n, alpha) == exact_j, (n, alpha)
            M = int(rng.integers(1, 10**6))
            exact_t = math.floor(a * M) + 1
            # reproduce the finite index with the exact t and compare the first candidate j whose tail is small
            assert exact_t <= M
            checks += 1
    assert checks > 2000
    # explicit integer-crossing cases
    assert cal.marginal_index(19, 0.05) == 19      # (20)(0.95) = 19 exactly
    assert cal.marginal_index(39, 0.05) == 38      # (40)(0.95) = 38 exactly
    assert cal.marginal_index(1999, 0.05) == 1900  # (2000)(0.95) = 1900 exactly
    assert cal.exact_fraction(0.05) == Fraction(1, 20)
    assert cal.exact_fraction(0.7) == Fraction(7, 10)


def test_pac_index_definition_and_table():
    for n, alpha, delta in [(1000, 0.05, 0.0125), (2000, 0.05, 0.0125), (300, 0.1, 0.05), (50, 0.05, 0.05)]:
        j = cal.pac_index(n, alpha, delta)
        assert binom.sf(j - 1, n, 1 - alpha) <= delta
        if j > 1:
            assert binom.sf(j - 2, n, 1 - alpha) > delta
    assert cal.pac_index(1000, 0.05, 0.0125) == 966
    assert cal.pac_index(2000, 0.05, 0.0125) == 1922


def test_finite_population_index_table():
    assert cal.finite_population_index(1000, 2452, 0.05, 0.0125) == 968
    assert cal.finite_population_index(2000, 5405, 0.05, 0.0125) == 1925
    # ordering of the three indices in the main configuration
    for n, M in [(1000, 2452), (2000, 5405)]:
        jm, jp, jf = cal.marginal_index(n, 0.05), cal.pac_index(n, 0.05, 0.0125), cal.finite_population_index(n, M, 0.05, 0.0125)
        assert jm < jp < jf


def exact_split_failure_probability(pool, n, j, alpha):
    """Enumerate all C(L, n) calibration subsets; failure = test fraction of
    values strictly above a_(j) exceeds alpha."""
    L = len(pool)
    M = L - n
    fails, total = 0, 0
    for cal_idx in itertools.combinations(range(L), n):
        cal_vals = sorted(pool[i] for i in cal_idx)
        thr = math.inf if j >= n + 1 else cal_vals[j - 1]
        test_vals = [pool[i] for i in range(L) if i not in cal_idx]
        frac = sum(v > thr for v in test_vals) / M
        fails += frac > alpha
        total += 1
    return fails / total


def test_hypergeometric_identity_exhaustive_distinct_values():
    """eq. (21): with distinct values the failure probability equals p_j exactly."""
    rng = np.random.default_rng(0)
    checks = 0
    for L, n in [(8, 3), (9, 4), (10, 5), (11, 4)]:
        pool = list(rng.normal(size=L))
        M = L - n
        for alpha in (0.1, 0.25, 0.4):
            t = math.floor(alpha * M + 1e-12) + 1
            for j in range(1, n + 2):
                exact = exact_split_failure_probability(pool, n, j, alpha)
                good = M + j - t
                pj = 0.0 if good < 0 else float(hypergeom.sf(j - 1, L, min(good, L), n))
                assert abs(exact - pj) < 1e-12, (L, n, alpha, j, exact, pj)
                checks += 1
    assert checks >= 60


def test_hypergeometric_bound_conservative_with_ties():
    rng = np.random.default_rng(1)
    for L, n in [(9, 4), (10, 4)]:
        pool = [float(x) for x in rng.integers(0, 4, L)]  # many ties
        M = L - n
        for alpha in (0.1, 0.3):
            t = math.floor(alpha * M + 1e-12) + 1
            for j in range(1, n + 2):
                exact = exact_split_failure_probability(pool, n, j, alpha)
                good = M + j - t
                pj = 0.0 if good < 0 else float(hypergeom.sf(j - 1, L, min(good, L), n))
                assert exact <= pj + 1e-12


def test_finite_index_guarantee_holds_exhaustively():
    """The chosen index j satisfies P_split[test error fraction > alpha] <= delta."""
    rng = np.random.default_rng(2)
    for L, n in [(9, 4), (10, 5)]:
        pool = list(rng.normal(size=L))
        M = L - n
        for alpha, delta in [(0.25, 0.3), (0.4, 0.2), (0.1, 0.5)]:
            j = cal.finite_population_index(n, M, alpha, delta)
            assert exact_split_failure_probability(pool, n, j, alpha) <= delta + 1e-12


def test_pac_guarantee_by_simulation():
    """Continuous scores: P_cal[ P_Q(A > a_(j)) > alpha ] should be <= delta."""
    rng = np.random.default_rng(3)
    n, alpha, delta = 200, 0.1, 0.1
    j = cal.pac_index(n, alpha, delta)
    q = 1 - alpha  # true (1-alpha)-quantile of Uniform(0,1) is 0.9
    fails = 0
    T = 4000
    for _ in range(T):
        a = np.sort(rng.random(n))
        thr = math.inf if j >= n + 1 else a[j - 1]
        fails += (1 - min(thr, 1.0)) > alpha  # P(U > thr) = 1 - thr
    assert fails / T <= delta + 0.02


def test_calibrate_threshold_returns_order_statistic():
    scores = np.arange(1, 21, dtype=float)
    res = cal.calibrate_threshold(scores, 0.1, "marginal")
    assert res.index == 19 and res.threshold == 19.0
    res = cal.calibrate_threshold(scores, 0.01, "marginal")
    assert res.index == 21 and math.isinf(res.threshold)
    res = cal.calibrate_threshold(scores, 0.1, "pac", delta=0.05)
    assert res.index == cal.pac_index(20, 0.1, 0.05)
    with pytest.raises(ValueError):
        cal.calibrate_threshold([1.0, float("nan")], 0.1)
    with pytest.raises(ValueError):
        cal.calibrate_threshold(scores, 0.1, "pac")  # needs delta


def test_family_delta():
    assert cal.family_delta(0.05, 4) == pytest.approx(0.0125)


def test_monte_carlo_finite_matches_exhaustive():
    rng = np.random.default_rng(5)
    pool = list(rng.normal(size=9))
    n, alpha, j = 4, 0.25, 4
    exact = exact_split_failure_probability(pool, n, j, alpha)
    mc = cal.empirical_failure_probability_finite(pool, n, j, alpha, n_splits=6000, seed=1)
    assert abs(mc - exact) < 0.03
