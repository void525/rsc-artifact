"""The blocker-batch controller depends on lambda (review round 3, W3/Q2).

A trajectory frozen at one lambda does not certify the controller re-executed at another; the controller's own
critical value (core.blocker_critical_value) does.  These tests check, by re-executing the controller at every
breakpoint of lambda on random small instances, that the controller errs iff lambda < blocker_critical_value(q),
and they reproduce the reviewer's counterexample.
"""
import math

import numpy as np
import pytest

from rscrank import core
from rscrank.policies import mu_descending


def blocker_schedule(first, min_batch):
    base = core.blocker_batch_schedule(min_batch=min_batch)

    def sched(round_idx, observed, order, mu_by_id, sg_by_id, lam, k):
        return first if round_idx == 0 else base(round_idx, observed, order, mu_by_id, sg_by_id, lam, k)

    sched.adaptive = True
    return sched


def run_blocker(s, mu, sigma, k, lam, first, min_batch):
    ids = list(range(len(s)))
    oracle = core.TeacherOracle(ids, s)
    res = core.run_controller(ids, mu, sigma, lam, k, oracle, mu_descending(), blocker_schedule(first, min_batch))
    truth = [int(p) for p in core.true_topk(np.asarray(s, float), k, ids)]
    return res, list(res.topk_ids) != truth


def frozen_trajectory_value(s, mu, sigma, k, first, min_batch, lam_ref):
    """What a frozen-trajectory calibration would compute: A^pi of the trajectory generated at lam_ref."""
    ids = list(range(len(s)))
    oracle = core.TeacherOracle(ids, s)
    res = core.run_controller(ids, mu, sigma, lam_ref, k, oracle, mu_descending(), blocker_schedule(first, min_batch))
    traj = res.trajectory if not res.early_stopped else res.trajectory
    if not res.early_stopped:
        traj = traj[:-1]
    prefixes = core.prefixes_from_trajectory(traj, ids)
    return core.trajectory_critical_value(np.asarray(s, float), np.asarray(mu, float), np.asarray(sigma, float), k, prefixes, 1, ids)


def test_reviewer_counterexample():
    s, mu, sigma = np.array([0.0, 1.0, 10.0]), np.array([10.0, -5.0, -10.0]), np.ones(3)
    k, first, min_batch = 1, 1, 1
    a_blk = core.blocker_critical_value(s, mu, sigma, k, None, first, min_batch)
    assert a_blk == 10.0
    assert core.critical_value(s, mu, sigma, k, 1) == 11.0
    # the trajectory generated at the reference lambda 11 has critical value 5 ...
    assert frozen_trajectory_value(s, mu, sigma, k, first, min_batch, 11.0) == 5.0
    # ... yet the controller re-executed at lambda = 6 >= 5 stops early and misses the winner d3
    _, err6 = run_blocker(s, mu, sigma, k, 6.0, first, min_batch)
    assert err6
    _, err10 = run_blocker(s, mu, sigma, k, 10.0, first, min_batch)
    assert not err10
    _, err_below = run_blocker(s, mu, sigma, k, float(np.nextafter(10.0, -np.inf)), first, min_batch)
    assert err_below


def _breakpoints(s, mu, sigma, k):
    """All values (tau(V) - mu_j)/sigma at which the controller's decisions can change, for V any mu-prefix."""
    n = len(s)
    order = sorted(range(n), key=lambda i: (-mu[i], i))
    pts = set()
    for m in range(k, n):
        tau = sorted((s[i] for i in order[:m]), reverse=True)[k - 1]
        for j in range(n):
            pts.add(float((tau - mu[j]) / sigma[j]))
    pts.add(core.critical_value(s, mu, sigma, k, 1))
    out = set()
    for p in pts:
        out.update({p, float(np.nextafter(p, -np.inf)), float(np.nextafter(p, np.inf))})
    return sorted(out)


@pytest.mark.parametrize("seed", range(6))
def test_blocker_controller_errs_iff_lambda_below_its_critical_value(seed):
    rng = np.random.default_rng(1000 + seed)
    checked = 0
    for _ in range(250):
        n = int(rng.integers(4, 13))
        k = int(rng.integers(1, min(4, n)))
        first = int(rng.integers(k, n))
        min_batch = int(rng.integers(1, 4))
        if rng.random() < 0.3:  # ties in scores and predictions
            s = rng.integers(0, 4, n).astype(float)
            mu = rng.integers(0, 4, n).astype(float)
        else:
            s = rng.normal(size=n)
            mu = s + rng.normal(scale=1.0, size=n)
        sigma = np.full(n, float(rng.uniform(0.2, 2.0)))
        a_blk = core.blocker_critical_value(s, mu, sigma, k, None, first, min_batch)
        assert a_blk <= core.critical_value(s, mu, sigma, k, 1)
        for lam in _breakpoints(s, mu, sigma, k) + [-1e6, 1e6]:
            _, err = run_blocker(s, mu, sigma, k, lam, first, min_batch)
            assert err == (lam < a_blk), (s, mu, sigma, k, first, min_batch, lam, a_blk)
            checked += 1
    assert checked > 1000


def test_frozen_trajectory_underestimates_on_some_instances():
    """The frozen-trajectory value never exceeds the controller's critical value and is strictly smaller on some
    instances, which is why the blocker controller needs its own statistic."""
    rng = np.random.default_rng(7)
    strictly = 0
    for _ in range(400):
        n = int(rng.integers(5, 12))
        k = int(rng.integers(1, 3))
        first, min_batch = k, 1
        s = rng.normal(size=n)
        mu = s + rng.normal(scale=1.5, size=n)
        sigma = np.ones(n)
        a_blk = core.blocker_critical_value(s, mu, sigma, k, None, first, min_batch)
        frozen = frozen_trajectory_value(s, mu, sigma, k, first, min_batch, math.inf)
        assert frozen <= a_blk
        strictly += frozen < a_blk
    assert strictly > 0
