"""Checks of the checkpoint-grid critical value (size_critical_value / grid_critical_value).

Brute force over every unsafe set of a given size, monotonicity in the size, the last-winner identity with a
shared scale, and the controller under random orders restricted to a checkpoint grid.
"""
import itertools
import math

import numpy as np
import pytest

from rscrank import core


def brute_size_value(s, mu, sigma, k, c):
    n = len(s)
    top = set(core.true_topk(s, k).tolist())
    best = -math.inf
    for V in itertools.combinations(range(n), c):
        if top - set(V):
            best = max(best, core.lambda_star(V, s, mu, sigma, k))
    return best


def random_instance(rng, n, ties, shared):
    if ties:
        s = rng.integers(0, 4, size=n).astype(float)
        mu = rng.integers(-3, 4, size=n).astype(float)
    else:
        s = rng.normal(size=n)
        mu = s + rng.normal(scale=1.5, size=n)
    sigma = np.full(n, float(rng.uniform(0.5, 2.0))) if shared else rng.uniform(0.3, 2.0, size=n)
    return s, mu, sigma


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_size_value_equals_bruteforce(seed):
    rng = np.random.default_rng(seed)
    checks = 0
    for _ in range(150):
        n = int(rng.integers(3, 9))
        k = int(rng.integers(1, n))
        s, mu, sigma = random_instance(rng, n, ties=bool(rng.integers(0, 2)), shared=bool(rng.integers(0, 2)))
        for c in range(k, n):
            assert core.size_critical_value(s, mu, sigma, k, c) == brute_size_value(s, mu, sigma, k, c)
            checks += 1
    assert checks > 300


def test_size_value_monotone_and_last_size_is_A():
    rng = np.random.default_rng(7)
    for _ in range(300):
        n = int(rng.integers(3, 10))
        k = int(rng.integers(1, n))
        s, mu, sigma = random_instance(rng, n, ties=bool(rng.integers(0, 2)), shared=bool(rng.integers(0, 2)))
        vals = [core.size_critical_value(s, mu, sigma, k, c) for c in range(k, n)]
        assert all(a <= b for a, b in zip(vals, vals[1:]))
        assert vals[-1] == core.critical_value(s, mu, sigma, k, 1)


def test_shared_scale_last_winner_identity_and_grid_equality():
    """With a shared sigma, A_{k,1} = (s_(k+1) - mu_[H_k]) / sigma, and the grid value equals A_{k,1} whenever the
    last winner in the mu order sits at or above the last checkpoint below N."""
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(600):
        n = int(rng.integers(4, 12))
        k = int(rng.integers(1, n - 1))
        s, mu, sigma = random_instance(rng, n, ties=False, shared=True)
        order = core.teacher_order(s)
        H = core.H_k(s, mu, k)
        mu_sorted = mu[core.teacher_order(mu)]
        A = core.critical_value(s, mu, sigma, k, 1)
        assert A == pytest.approx((s[order[k]] - mu_sorted[H - 1]) / sigma[0], abs=1e-12)
        for c in range(k, n):
            g = core.size_critical_value(s, mu, sigma, k, c)
            if H <= c:
                assert g == A
                hits += 1
    assert hits > 500


def test_grid_controller_safe_at_grid_value_and_witness_below():
    """Every order restricted to the grid is safe at the grid value; just below it, a nested chain of grid-size
    subsets of the maximising set makes the controller stop missing a winner."""
    rng = np.random.default_rng(5)
    n_err = 0
    for _ in range(250):
        n = int(rng.integers(4, 9))
        k = int(rng.integers(1, n - 1))
        s, mu, sigma = random_instance(rng, n, ties=bool(rng.integers(0, 2)), shared=bool(rng.integers(0, 2)))
        sizes = sorted(set(int(x) for x in rng.integers(k, n, size=2)) | {n})
        g = core.grid_critical_value(s, mu, sigma, k, sizes)
        top = set(core.true_topk(s, k).tolist())
        if not math.isfinite(g):
            continue
        # safety: every nested chain with sizes on the grid never stops on an unsafe set at lambda = g
        for perm in itertools.islice(itertools.permutations(range(n)), 60):
            for c in sizes:
                V = list(perm[:c])
                if c < n and (top - set(V)):
                    assert not g < core.lambda_star(V, s, mu, sigma, k)
        # witness: at lambda just below g some grid-size unsafe set satisfies the rule
        lam = float(np.nextafter(g, -np.inf))
        c_star = max(c for c in sizes if k <= c < n)
        found = any((top - set(V)) and lam < core.lambda_star(V, s, mu, sigma, k)
                    for V in itertools.combinations(range(n), c_star))
        assert found
        n_err += 1
    assert n_err > 100
