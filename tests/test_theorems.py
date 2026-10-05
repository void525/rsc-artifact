"""Exhaustive and exact checks of Theorem 1 / Theorem 2 and their corollaries.

These tests are the machine-checked counterpart of paper/sections/03_method.tex.
They do NOT prove the theorems; they falsify them on thousands of small
instances (floats and exact rationals), including ties at the top-k boundary.
"""
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from rscrank import core
from rscrank.synthetic import small_instances

# --------------------------------------------------------------------------- #
# Independent exact (rational) reference implementation -- written separately
# from rscrank.core on purpose.
# --------------------------------------------------------------------------- #
def exact_order(s):
    return sorted(range(len(s)), key=lambda i: (-s[i], i))


def exact_A(s, mu, sigma, k, r):
    order = exact_order(s)
    top = order[:k]
    b = s[order[k + r - 1]]
    z = sorted(((b - mu[i]) / sigma[i] for i in top), reverse=True)
    return z[r - 1]


def exact_can_stop(V, s, mu, sigma, k, lam):
    if len(V) < k:
        return False
    tau = sorted((s[i] for i in V), reverse=True)[k - 1]
    rest = [j for j in range(len(s)) if j not in V]
    if not rest:
        return True
    return tau > max(mu[j] + lam * sigma[j] for j in rest)


def exact_unsafe_sets(s, k, r):
    n = len(s)
    top = set(exact_order(s)[:k])
    for size in range(k, n):
        for V in itertools.combinations(range(n), size):
            if len(top - set(V)) >= r:
                yield V


# --------------------------------------------------------------------------- #
def valid_r(k, N):
    return range(1, min(k, N - k) + 1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closed_form_equals_bruteforce_max_over_unsafe_sets(seed):
    """eq. (9)/(12): max_{V unsafe} lambda*(V) == A_{k,r}(q), exactly (diff 0)."""
    n_sets = 0
    n_checks = 0
    for s, mu, sigma, k in small_instances(seed=seed, n_instances=120, N_range=(3, 8)):
        for r in valid_r(k, len(s)):
            a = core.critical_value(s, mu, sigma, k, r)
            bf, cnt = core.brute_force_critical_value(s, mu, sigma, k, r)
            n_sets += cnt
            n_checks += 1
            assert bf == a, f"closed form {a} != brute force {bf} (seed={seed}, k={k}, r={r})"
    assert n_checks > 100 and n_sets > 5000


def test_iff_exact_rationals():
    """Theorem 2 with exact arithmetic: at lambda = A no unsafe set can stop;
    at lambda = A - eps at least one unsafe set can stop."""
    rng = np.random.default_rng(7)
    checks = 0
    for _ in range(80):
        N = int(rng.integers(3, 7))
        k = int(rng.integers(1, N))
        s = [Fraction(int(x), 2) for x in rng.integers(-6, 7, N)]
        mu = [Fraction(int(x), 3) for x in rng.integers(-9, 10, N)]
        sigma = [Fraction(int(x), 2) for x in rng.integers(1, 5, N)]
        for r in valid_r(k, N):
            A = exact_A(s, mu, sigma, k, r)
            eps = Fraction(1, 1000)
            unsafe = list(exact_unsafe_sets(s, k, r))
            assert not any(exact_can_stop(V, s, mu, sigma, k, A) for V in unsafe)
            assert any(exact_can_stop(V, s, mu, sigma, k, A - eps) for V in unsafe)
            # and the float implementation agrees with the exact value
            af = core.critical_value(np.array(s, float), np.array(mu, float), np.array(sigma, float), k, r)
            assert abs(af - float(A)) < 1e-9
            checks += 1
    assert checks >= 100


def test_sufficiency_direction_with_ties_at_boundary():
    """Explicit ties at s_(k) = s_(k+1): the strict rule still never stops unsafely
    at lambda = A (tie-break by id)."""
    s = np.array([5.0, 5.0, 5.0, 3.0, 3.0, 1.0])
    mu = np.array([4.0, 6.0, 2.0, 3.5, 2.0, 0.0])
    sigma = np.ones(6)
    for k in range(1, 6):
        for r in valid_r(k, 6):
            A = core.critical_value(s, mu, sigma, k, r)
            top = core.true_topk(s, k)
            for V in core.unsafe_sets(6, k, top, r):
                rest = [j for j in range(6) if j not in V]
                assert not core.can_stop(s[list(V)], mu[rest], sigma[rest], A, k)


def test_frontier_monotone_and_dominance():
    """eq. (13): A_{k,r+1} <= A_{k,r};  eq. (14): A_{k,1} <= B_top <= B_all."""
    for s, mu, sigma, k in small_instances(seed=3, n_instances=200, N_range=(3, 10)):
        N = len(s)
        prev = None
        for r in valid_r(k, N):
            a = core.critical_value(s, mu, sigma, k, r)
            if prev is not None:
                assert a <= prev + 1e-12
            prev = a
        a1 = core.critical_value(s, mu, sigma, k, 1)
        bt = core.B_top(s, mu, sigma, k)
        ba = core.B_all(s, mu, sigma)
        assert a1 <= bt + 1e-12 <= ba + 2e-12


def test_controller_is_safe_at_or_above_A_for_random_policies_and_schedules():
    rng = np.random.default_rng(11)
    runs = 0
    for s, mu, sigma, k in small_instances(seed=5, n_instances=150, N_range=(3, 9)):
        N = len(s)
        ids = [f"d{j}" for j in rng.permutation(N)]  # arbitrary external ids
        A = core.critical_value(s, mu, sigma, k, 1, ids)
        truth = [ids[p] for p in core.true_topk(s, k, ids)]
        for lam in (A, A + 0.1, A + 3.0, math.inf):
            for trial in range(4):
                perm = rng.permutation(N)
                key = {ids[p]: float(t) for t, p in enumerate(perm)}

                def policy(rem, obs, mu_by_id, key=key):
                    return sorted(rem, key=lambda i: key[i])

                b0, b1 = int(rng.integers(1, N + 1)), int(rng.integers(1, N + 1))
                oracle = core.TeacherOracle(ids, s)
                res = core.run_controller(ids, mu, sigma, lam, k, oracle, policy, lambda t: b0 if t == 0 else b1)
                assert res.topk_ids == truth
                assert res.n_calls == len(res.scored_ids) == oracle.n_calls
                runs += 1
    assert runs > 1000


def test_controller_is_safe_at_A_kr_for_r_ge_2():
    """Sec. 8 implementation note for general r: at lambda = A_{k,r} exactly (float), no legal
    policy/schedule ever stops missing >= r winners; at nextafter(A_{k,r}, -inf) the witness
    order (score everything except the r worst-covered winners first) misses exactly r winners."""
    rng = np.random.default_rng(23)
    safe_runs, witness_hits = 0, 0
    for s, mu, sigma, k in small_instances(seed=17, n_instances=200, N_range=(4, 9)):
        N = len(s)
        ids = list(range(N))
        truth = set(core.true_topk(s, k, ids).tolist())
        for r in valid_r(k, N):
            if r < 2:
                continue
            A = core.critical_value(s, mu, sigma, k, r, ids)
            for trial in range(4):
                perm = rng.permutation(N)
                key = {ids[p]: float(t) for t, p in enumerate(perm)}
                b0, b1 = int(rng.integers(1, N + 1)), int(rng.integers(1, N + 1))
                oracle = core.TeacherOracle(ids, s)
                res = core.run_controller(ids, mu, sigma, A, k, oracle, lambda rem, obs, m, key=key: sorted(rem, key=lambda i: key[i]), lambda t: b0 if t == 0 else b1)
                assert len(truth - set(res.topk_ids)) < r, (r, A, res)
                safe_runs += 1
            # witness of the necessity proof at lambda just below A_{k,r}
            order = core.teacher_order(s, ids)
            top = order[:k]
            z = (s[order[k + r - 1]] - mu[top]) / sigma[top]
            W = [int(top[i]) for i in np.argsort(-z, kind="stable")[:r]]
            others = [i for i in ids if i not in W]
            lam = float(np.nextafter(A, -math.inf))
            if len(others) >= k:
                oracle = core.TeacherOracle(ids, s)
                res = core.run_controller(ids, mu, sigma, lam, k, oracle, lambda rem, obs, m, others=others, W=W: [i for i in others if i in rem] + [i for i in W if i in rem], lambda t: len(others) if t == 0 else 1)
                assert res.early_stopped and len(truth - set(res.topk_ids)) >= r
                witness_hits += 1
    assert safe_runs > 300 and witness_hits > 50


def test_necessity_direction_realised_by_a_legal_policy():
    """For lambda < A_{k,1} there is a legal scoring order (score everything except
    the argmax winner w first) on which the controller stops early and misses w."""
    found = 0
    for s, mu, sigma, k in small_instances(seed=9, n_instances=100, N_range=(3, 8), tie_frac=0.0):
        N = len(s)
        ids = list(range(N))
        A = core.critical_value(s, mu, sigma, k, 1, ids)
        order = core.teacher_order(s, ids)
        top = order[:k]
        z = (s[order[k]] - mu[top]) / sigma[top]
        w = int(top[int(np.argmax(z))])
        lam = A - 1e-6
        others = [i for i in ids if i != w]

        def policy(rem, obs, mu_by_id):
            return [i for i in others if i in rem] + ([w] if w in rem else [])

        oracle = core.TeacherOracle(ids, s)
        res = core.run_controller(ids, mu, sigma, lam, k, oracle, policy, lambda t: N - 1 if t == 0 else 1)
        if N - 1 >= k:
            assert res.early_stopped and w not in res.topk_ids
            found += 1
    assert found > 50


def test_trajectory_threshold_below_A_and_protects_only_its_own_trajectory():
    rng = np.random.default_rng(3)
    below, transfer_fail = 0, 0
    for s, mu, sigma, k in small_instances(seed=21, n_instances=150, N_range=(4, 8), tie_frac=0.0):
        N = len(s)
        ids = list(range(N))
        A = core.critical_value(s, mu, sigma, k, 1, ids)
        # trajectory pi: mu-descending in batches of 1
        pi = [int(p) for p in core.teacher_order(mu, ids)]
        prefixes = [pi[: t + 1] for t in range(N - 1)]
        Api = core.trajectory_critical_value(s, mu, sigma, k, prefixes, 1, ids)
        assert Api <= A + 1e-12
        if Api < A - 1e-9:
            below += 1
            lam = Api if Api > -math.inf else -1e9
            # protects pi itself
            oracle = core.TeacherOracle(ids, s)
            res = core.run_controller(ids, mu, sigma, lam, k, oracle, lambda rem, obs, m: [i for i in pi if i in rem], lambda t: 1)
            assert res.topk_ids == [ids[p] for p in core.true_topk(s, k, ids)]
            # converse of Prop. 5: just below A^pi the same trajectory stops at an unsafe prefix
            if Api > -math.inf:
                oracle = core.TeacherOracle(ids, s)
                res_lo = core.run_controller(ids, mu, sigma, float(np.nextafter(Api, -math.inf)), k, oracle, lambda rem, obs, m: [i for i in pi if i in rem], lambda t: 1)
                assert res_lo.early_stopped and res_lo.topk_ids != [ids[p] for p in core.true_topk(s, k, ids)]
            # but some other legal order can fail at the same lambda
            for _ in range(20):
                perm = [int(x) for x in rng.permutation(N)]
                oracle = core.TeacherOracle(ids, s)
                res = core.run_controller(ids, mu, sigma, lam, k, oracle, lambda rem, obs, m, perm=perm: [i for i in perm if i in rem], lambda t: 1)
                if res.topk_ids != [ids[p] for p in core.true_topk(s, k, ids)]:
                    transfer_fail += 1
                    break
    assert below > 20
    assert transfer_fail > 0  # policy change without recalibration can break the guarantee


def test_policy_uniform_cost_counterexample_N_over_k():
    """Sec. 7.3: winners have s=1, mu=-c; others s=0, mu=0; sigma=1, c>1.
    Any lambda >= A = c forces N calls under every order, while a fixed
    winner-first order with a smaller lambda stops after k calls."""
    N, k, c = 12, 3, 2.5
    s = np.array([1.0] * k + [0.0] * (N - k))
    mu = np.array([-c] * k + [0.0] * (N - k))
    sigma = np.ones(N)
    ids = list(range(N))
    A = core.critical_value(s, mu, sigma, k, 1, ids)
    assert A == pytest.approx(c)
    winners_first = lambda rem, obs, m: sorted(rem)
    oracle = core.TeacherOracle(ids, s)
    res = core.run_controller(ids, mu, sigma, A, k, oracle, winners_first, lambda t: k if t == 0 else 1)
    assert res.n_calls == N and not res.early_stopped
    # trajectory-specific threshold for winners-first is small (< 1) -> k calls
    prefixes = [list(range(t + 1)) for t in range(N - 1)]
    Api = core.trajectory_critical_value(s, mu, sigma, k, prefixes, 1, ids)
    assert Api < 1.0
    oracle = core.TeacherOracle(ids, s)
    res2 = core.run_controller(ids, mu, sigma, max(Api, 0.0) + 1e-9, k, oracle, winners_first, lambda t: k if t == 0 else 1)
    assert res2.n_calls == k and res2.topk_ids == list(range(k))
    assert res.n_calls / res2.n_calls == pytest.approx(N / k)


def test_arithmetic_example_of_section_7_2():
    s = np.array([9.0, 8.0, 7.0, 1.0])
    mu = np.array([8.0, 6.0, 6.0, -100.0])
    sigma = np.ones(4)
    assert core.critical_value(s, mu, sigma, 2, 1) == pytest.approx(1.0)
    assert core.B_top(s, mu, sigma, 2) == pytest.approx(2.0)
    assert core.B_all(s, mu, sigma) == pytest.approx(101.0)
    U = core.guardrail(mu, sigma, 1.0)
    assert np.allclose(U, [9, 7, 7, -99])
    assert core.can_stop(s[:2], mu[2:], sigma[2:], 1.0, 2)
    assert not core.can_stop(s[:2], mu[2:], sigma[2:], 2.0, 2)  # tau=8, U=(8,-98): not strict


def test_cert_lower_bound_is_a_lower_bound_on_calls_of_correct_early_stops():
    rng = np.random.default_rng(1)
    for s, mu, sigma, k in small_instances(seed=13, n_instances=100, N_range=(3, 8)):
        N = len(s)
        ids = list(range(N))
        A = core.critical_value(s, mu, sigma, k, 1, ids)
        lam = A + float(rng.uniform(0, 1))
        lb = core.cert_lower_bound(s, mu, sigma, k, lam, ids)
        for _ in range(5):
            perm = [int(x) for x in rng.permutation(N)]
            oracle = core.TeacherOracle(ids, s)
            res = core.run_controller(ids, mu, sigma, lam, k, oracle, lambda rem, obs, m, perm=perm: [i for i in perm if i in rem], lambda t: 1)
            if res.early_stopped:
                assert res.n_calls >= lb


def test_H_k_prefix_characterisation():
    for s, mu, sigma, k in small_instances(seed=17, n_instances=100, N_range=(3, 9)):
        N = len(s)
        ids = list(range(N))
        H = core.H_k(s, mu, k, ids)
        cheap = core.teacher_order(mu, ids)
        top = set(core.true_topk(s, k, ids).tolist())
        for m in range(k, N + 1):
            contains = top <= set(cheap[:m].tolist())
            assert contains == (m >= H)


def test_invariants_raise():
    s = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        core.critical_value(s, s, np.array([1.0, 0.0, 1.0]), 1, 1)
    with pytest.raises(ValueError):
        core.critical_value(np.array([1.0, np.nan, 3.0]), s, np.ones(3), 1, 1)
    with pytest.raises(ValueError):
        core.critical_value(s, s, np.ones(3), 3, 1)  # k must be < N
    with pytest.raises(ValueError):
        core.critical_value(s, s, np.ones(3), 2, 2)  # r <= min(k, N-k) = 1
    with pytest.raises(ValueError):
        core.teacher_order(s, ids=["a", "a", "b"])
    with pytest.raises(KeyError):
        core.TeacherOracle(["a", "b"], [1.0, 2.0]).score(["zz"])
    with pytest.raises(ValueError):
        core.TeacherOracle(["a", "b"], [1.0, float("nan")])
    with pytest.raises(ValueError):
        core.TeacherOracle(["a", "b"], [1.0, 2.0]).score(["a", "a"])


# --------------------------------------------------------------------------- #
# Proposition 10: cost anatomy of the blocker-first (guardrail-descending) schedule
# --------------------------------------------------------------------------- #
def _instance(rng, n=20, k=5):
    s = rng.normal(0, 1, n)
    mu = s + rng.normal(0, 0.8, n)
    sigma = np.full(n, float(rng.uniform(0.5, 1.5)))
    ids = [f"d{i:02d}" for i in range(n)]
    return s, mu, sigma, ids, k


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_cost_anatomy_predicts_the_controller_exactly(seed):
    """calls(mu-descending controller) == cert_core + band, then rounded up to the next checkpoint,
    on every covered query (lambda >= A); and cert_core is a lower bound for *any* policy."""
    from rscrank.policies import make_policy, random_order

    rng = np.random.default_rng(seed)
    for _ in range(40):
        s, mu, sigma, ids, k = _instance(rng)
        A = core.critical_value(s, mu, sigma, k, 1, ids)
        for lam in (A, A + 0.3, A + 2.0):
            for first, rest in ((k, 1), (8, 4), (12, 6)):
                sched = core.default_batch_schedule(k, first, rest)
                cps = core.checkpoints_from_schedule(len(s), k, sched)
                ana = core.cost_anatomy(s, mu, sigma, k, lam, ids, cps)
                oracle = core.TeacherOracle(ids, s)
                res = core.run_controller(ids, mu, sigma, lam, k, oracle, make_policy("mu"), sched)
                assert res.n_calls == int(ana["calls_predicted"]), (res.n_calls, ana, lam, first, rest)
                assert ana["m_star"] == ana["cert_core"] + ana["band"] or ana["m_star"] == len(s)
                # instance optimality: no legal policy stops earlier than the certification core
                for pol in ("mu", "cheap_raw", "bm25", "random"):
                    feats = {"cheap_raw": {i: float(m) for i, m in zip(ids, mu)}, "bm25": {i: float(-x) for i, x in zip(ids, mu)}, "length": {i: 1.0 for i in ids}}
                    o2 = core.TeacherOracle(ids, s)
                    r2 = core.run_controller(ids, mu, sigma, lam, k, o2, make_policy(pol, feats, seed), sched)
                    assert r2.n_calls >= min(ana["cert_core"], len(s)), (pol, r2.n_calls, ana)


def test_cost_anatomy_and_policies_share_the_identifier_tie_rule():
    """Prop. 10(b) claims the U-descending schedule with ties by identifier *is* the mu policy.
    With integer ids >= 10 (numeric order != string order) and exactly tied mu values, the
    anatomy's predicted calls must still equal the controller's calls."""
    from rscrank.policies import make_policy

    rng = np.random.default_rng(31)
    checked = 0
    for _ in range(300):
        n = int(rng.integers(12, 40))
        k = int(rng.integers(2, 6))
        s = np.sort(rng.normal(size=n))[::-1]
        # mu with many exact ties: quantise to a coarse grid
        mu = np.round(s + rng.normal(0, 0.8, n), 0)
        sigma = np.ones(n)
        ids = [int(x) for x in rng.permutation(np.arange(5, 5 + n))]  # numeric ids >= 5, unsorted
        A = core.critical_value(s, mu, sigma, k, 1, ids)
        for lam in (A, A + 0.5):
            for first, rest in ((k, 1), (8, 4)):
                sched = core.default_batch_schedule(k, first, rest)
                cps = core.checkpoints_from_schedule(n, k, sched)
                ana = core.cost_anatomy(s, mu, sigma, k, lam, ids, cps)
                oracle = core.TeacherOracle(ids, s)
                res = core.run_controller(ids, mu, sigma, lam, k, oracle, make_policy("mu"), sched)
                assert res.n_calls == int(ana["calls_predicted"]), (res.n_calls, ana)
                checked += 1
    assert checked >= 1000


def test_cost_anatomy_uses_the_controllers_normalised_comparison_at_lambda_eq_A():
    """At lambda == A_{k,1}(q) exactly (a calibrated lambda often *is* some query's A) and with
    quantised inputs, the guardrail form mu + lambda*sigma >= theta and the controller's normalised
    form (theta - mu)/sigma <= lambda can disagree by one document; the anatomy must use the latter
    so that calls_predicted == the controller's calls on every covered query."""
    from rscrank.policies import make_policy

    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(1500):
        n = int(rng.integers(6, 16))
        k = int(rng.integers(1, min(6, n)))
        s = rng.normal(size=n)
        if rng.random() < 0.4:
            s = np.round(s, 0)
        mu = s + rng.normal(0, 1.0, n)
        if rng.random() < 0.5:
            mu = np.round(mu, 0)
        sigma = np.full(n, float(np.exp(rng.normal())))
        ids = [int(x) for x in rng.permutation(np.arange(7, 7 + n))]
        A = core.critical_value(s, mu, sigma, k, 1, ids)
        sched = core.default_batch_schedule(k, k, 1)
        cps = core.checkpoints_from_schedule(n, k, sched)
        ana = core.cost_anatomy(s, mu, sigma, k, A, ids, cps)
        res = core.run_controller(ids, mu, sigma, A, k, core.TeacherOracle(ids, s), make_policy("mu"), sched)
        assert res.n_calls == int(ana["calls_predicted"]), (res.n_calls, ana)
        assert core.cert_lower_bound(s, mu, sigma, k, A, ids) == int(ana["cert_core"])
        checked += 1
    assert checked == 1500


def test_order_statistics_monotone_in_pointwise_order_prop3b():
    """Prop. 3(b): if a_i <= b_i for every calibration query, then every order statistic of a is
    <= the same order statistic of b (hence lambda_A <= lambda_top <= lambda_all for any index)."""
    from rscrank.calibrate import calibrate_threshold

    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(5, 60))
        a = rng.normal(size=n)
        b = a + rng.exponential(size=n) * (rng.random(n) < 0.7)  # b >= a pointwise, many ties
        c = b + rng.exponential(size=n)
        sa, sb, sc = np.sort(a), np.sort(b), np.sort(c)
        assert np.all(sa <= sb) and np.all(sb <= sc)
        for alpha in (0.05, 0.1, 0.3):
            for regime in ("marginal", "pac", "finite"):
                ta = calibrate_threshold(a, alpha, regime, delta=0.05, M=n)
                tb = calibrate_threshold(b, alpha, regime, delta=0.05, M=n)
                tc = calibrate_threshold(c, alpha, regime, delta=0.05, M=n)
                assert ta.threshold <= tb.threshold <= tc.threshold


def test_cert_core_is_a_stopping_state_and_a_lower_bound():
    rng = np.random.default_rng(7)
    for _ in range(200):
        s, mu, sigma, ids, k = _instance(rng, n=14, k=4)
        A = core.critical_value(s, mu, sigma, k, 1, ids)
        lam = A + float(rng.uniform(0, 1))
        ana = core.cost_anatomy(s, mu, sigma, k, lam, ids)
        order = core.teacher_order(s, ids)
        theta = s[order[k - 1]]
        U = core.guardrail(mu, sigma, lam)
        Vstar = sorted(set(order[:k].tolist()) | {int(j) for j in np.where(U >= theta)[0]})
        assert len(Vstar) == int(ana["cert_core"]) == core.cert_lower_bound(s, mu, sigma, k, lam, ids)
        if len(Vstar) < len(s):
            assert core.lambda_star(Vstar, s, mu, sigma, k) > lam  # V* really does stop


def test_mu_binned_length_is_a_legal_policy_and_changes_the_order():
    from rscrank.policies import make_policy

    rng = np.random.default_rng(3)
    s, mu, sigma, ids, k = _instance(rng, n=30, k=10)
    lengths = {i: float(rng.integers(20, 400)) for i in ids}
    feats = {"length": lengths, "cheap_raw": {i: float(m) for i, m in zip(ids, mu)}, "bm25": {i: 0.0 for i in ids}}
    pol = make_policy("mu_binned_length", feats)
    mu_by_id = {i: float(m) for i, m in zip(ids, mu)}
    out = pol(list(ids), {}, mu_by_id)
    assert sorted(out) == sorted(ids)                      # permutation of the remaining ids
    assert out != make_policy("mu")(list(ids), {}, mu_by_id)  # genuinely different schedule
    A = core.critical_value(s, mu, sigma, k, 1, ids)
    oracle = core.TeacherOracle(ids, s)
    res = core.run_controller(ids, mu, sigma, A, k, oracle, pol, core.default_batch_schedule(k))
    truth = [ids[p] for p in core.true_topk(s, k, ids)]
    assert res.topk_ids == truth                            # protected at lambda = A like any other policy
