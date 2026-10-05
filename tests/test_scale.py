"""Main-configuration scale (N=100, k=10): safety under random legal policies and
the r-miss frontier, plus a timing sanity check for the replay evaluator."""
import math
import time

import numpy as np

from rscrank import core
from rscrank.replay import MethodSpec, query_statistics, replay_dataset
from rscrank.synthetic import make_dataset


def test_r_miss_frontier_at_scale():
    rng = np.random.default_rng(0)
    data = make_dataset(40, seed=1, N=100, noise=2.0)
    k = 10
    for b in data:
        st = query_statistics(b, k, r_values=(1, 2, 3))
        assert st["A_k3"] <= st["A_k2"] <= st["A_k1"] <= st["B_top"] <= st["B_all"]
        truth = set(b.true_topk_ids(k))
        for r in (1, 2, 3):
            lam = st[f"A_k{r}"]
            for _ in range(3):
                perm = list(rng.permutation(b.ids))
                oracle = core.TeacherOracle(b.ids, b.s)
                res = core.run_controller(b.ids, b.mu, b.sigma, lam, k, oracle, lambda rem, obs, m, perm=perm: [i for i in perm if i in rem], core.default_batch_schedule(k))
                missed = len(truth - set(res.topk_ids))
                assert missed < r  # lambda >= A_{k,r} forbids missing r or more winners
        # r = 1 with the main schedule: exact list
        oracle = core.TeacherOracle(b.ids, b.s)
        res = core.run_controller(b.ids, b.mu, b.sigma, st["A_k1"], k, oracle, lambda rem, obs, m: sorted(rem, key=lambda i: -m[i]), core.default_batch_schedule(k))
        assert res.topk_ids == b.true_topk_ids(k)


def test_replay_throughput():
    data = make_dataset(300, seed=2, N=100)
    t0 = time.time()
    recs = replay_dataset(data, MethodSpec("rsc", "guardrail", lam=2.0, policy="mu"), 10)
    dt = time.time() - t0
    assert len(recs) == 300
    assert dt < 30.0  # replay is cheap: ~ms per query on CPU
