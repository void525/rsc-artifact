"""The round-by-round controller of scripts/09b_throughput.py must reproduce core.run_controller exactly."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from rscrank import core, policies

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("thr09b", ROOT / "scripts" / "09b_throughput.py")
thr09b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(thr09b)  # type: ignore[union-attr]


class _Oracle:
    def __init__(self, s_by_id):
        self.s, self.n_calls = s_by_id, 0

    def score(self, ids):
        self.n_calls += len(ids)
        return [self.s[i] for i in ids]


def _instance(rng, N=40, k=5):
    ids = [f"d{i}" for i in range(N)]
    mu = rng.normal(size=N)
    s = mu + rng.normal(scale=0.7, size=N)
    sigma = np.full(N, float(rng.uniform(0.5, 1.5)))
    return ids, mu, s, sigma, k


def _run_step(ids, mu, sigma, lam, k, policy, sched, s_by_id):
    c = thr09b.StepController(ids, mu, sigma, lam, k, policy, sched)
    calls, rounds = 0, 0
    while True:
        b = c.next_batch()
        if not b:
            break
        c.observe(b, [s_by_id[i] for i in b])
        calls += len(b)
        rounds += 1
    return calls, rounds, c.topk(), c.early


def test_step_controller_matches_run_controller_fixed_and_blocker_schedules():
    rng = np.random.default_rng(7)
    for trial in range(60):
        ids, mu, s, sigma, k = _instance(rng)
        s_by_id = dict(zip(ids, s))
        lam = float(rng.uniform(-0.5, 3.0))
        for sched in (core.default_batch_schedule(k, 8, 4), thr09b.blocker_sched(12, 6)):
            for pol_name in ("mu", "cheap_raw"):
                feats = {"cheap_raw": dict(zip(ids, rng.normal(size=len(ids))))}
                pol_a = policies.make_policy(pol_name, feats, 0)
                pol_b = policies.make_policy(pol_name, feats, 0)
                ref = core.run_controller(ids, mu, sigma, lam, k, _Oracle(s_by_id), pol_a, sched)
                calls, rounds, top, early = _run_step(ids, mu, sigma, lam, k, pol_b, sched, s_by_id)
                assert calls == ref.n_calls, (trial, pol_name, calls, ref.n_calls)
                assert rounds == ref.n_rounds
                assert list(top) == list(ref.topk_ids)
                assert early == ref.early_stopped
