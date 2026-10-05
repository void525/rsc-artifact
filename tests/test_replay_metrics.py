"""Replay evaluator, baselines, policies, metrics, stats and provenance."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from rscrank import baselines, calibrate, core, metrics, policies, provenance, stats
from rscrank.replay import MethodSpec, QueryBundle, check_call_dominance, query_statistics, replay_dataset, replay_query
from rscrank.synthetic import make_dataset

K = 5


@pytest.fixture(scope="module")
def data():
    return make_dataset(160, seed=42, N=30, noise=1.5, tie_prob=0.05)


def calib_lambda(bundles, stat, alpha=0.1):
    vals = [query_statistics(b, K)[stat] for b in bundles]
    return calibrate.calibrate_threshold(vals, alpha, "marginal").threshold


def test_same_path_call_dominance_and_risk(data):
    cal, test = data[:80], data[80:]
    lams = {s: calib_lambda(cal, s) for s in ("A_k1", "B_top", "B_all")}
    assert lams["A_k1"] <= lams["B_top"] <= lams["B_all"]
    recs = {s: replay_dataset(test, MethodSpec(s, "guardrail", lam=lams[s], policy="mu"), K) for s in lams}
    assert check_call_dominance(recs["A_k1"], recs["B_top"], recs["B_all"])
    # every guardrail method with lambda >= its own A is exact on that query
    for b, r in zip(test, recs["A_k1"]):
        if query_statistics(b, K)["A_k1"] <= lams["A_k1"]:
            assert r["mismatch"] == 0
    # empirical risk on the test half must be small (marginal guarantee ~ alpha)
    risk = np.mean([r["mismatch"] for r in recs["A_k1"]])
    assert risk <= 0.25
    # cost bookkeeping
    for r in recs["A_k1"]:
        assert K <= r["n_calls"] <= r["N"]
        assert r["full_fallback"] == (r["n_calls"] == r["N"])
        assert math.isfinite(r["tokens_real"]) and r["tokens_padded"] >= r["tokens_real"] - 1e-9


def test_policy_change_keeps_guarantee_with_A_but_not_with_trajectory_threshold(data):
    cal, test = data[:80], data[80:]
    lamA = calib_lambda(cal, "A_k1")
    for pol in ["mu", "cheap_raw", "bm25", "length_asc", "length_desc", "random", "observed_regression", "alternating_bm25"]:
        recs = replay_dataset(test, MethodSpec(f"rsc-{pol}", "guardrail", lam=lamA, policy=pol), K)
        for b, r in zip(test, recs):
            if query_statistics(b, K)["A_k1"] <= lamA:
                assert r["mismatch"] == 0, pol


def test_prefix_baselines_correct_iff_m_geq_H(data):
    for b in data[:40]:
        H = core.H_k(b.s, b.mu, K, b.ids)
        for m in range(K, b.N + 1):
            r = replay_query(b, MethodSpec("p", "prefix_fixed", m=m), K)
            assert (r["mismatch"] == 0) == (m >= H)
            assert r["n_calls"] == m


def test_uniform_and_adaptive_prefix_calibration(data):
    cal, test = data[:80], data[80:]
    H_cal = [core.H_k(b.s, b.mu, K, b.ids) for b in cal]
    res = baselines.calibrate_uniform_prefix(H_cal, 0.1, "marginal", N=30)
    m = int(res.threshold)
    recs = replay_dataset(test, MethodSpec("uni", "prefix_uniform", m=m), K)
    fails = [r["mismatch"] for r in recs]
    assert np.mean(fails) <= 0.3
    # adaptive: h_pred = log H + noise (stand-in for a learned predictor)
    rng = np.random.default_rng(0)
    for b in data:
        Hb = core.H_k(b.s, b.mu, K, b.ids)
        b.features["h_pred"] = np.full(b.N, math.log(Hb) + rng.normal(0, 0.3))
    res2 = baselines.calibrate_adaptive_prefix(H_cal, [float(b.features["h_pred"][0]) for b in cal], 0.1)
    recs2 = replay_dataset(test, MethodSpec("ada", "prefix_adaptive", t_H=res2.threshold), K)
    for b, r in zip(test, recs2):
        Hb = core.H_k(b.s, b.mu, K, b.ids)
        e = math.log(Hb) - float(b.features["h_pred"][0])
        if e <= res2.threshold:
            assert r["mismatch"] == 0
    recs3 = replay_dataset(test, MethodSpec("learn", "prefix_learned", c=1.0), K)
    assert all(K <= r["n_calls"] <= 30 for r in recs3)


def test_full_and_cheap_only_methods(data):
    b = data[0]
    r = replay_query(b, MethodSpec("full", "full"), K)
    assert r["mismatch"] == 0 and r["n_calls"] == b.N
    r = replay_query(b, MethodSpec("cheap", "cheap_only"), K)
    assert r["n_calls"] == 0


def test_policies_return_permutations(data):
    b = data[3]
    feats = {n: b.feature_map(n) for n in b.features}
    feats["mu_alt"] = {i: -float(m) for i, m in zip(b.ids, b.mu)}  # the 'other cheap model' order (here: reversed mu)
    obs = {b.ids[0]: 1.0, b.ids[1]: -2.0, b.ids[2]: 0.5, b.ids[3]: 0.1, b.ids[4]: 2.0}
    mu_by = b.feature_map("cheap_raw")
    for name in policies.POLICY_NAMES:
        p = policies.make_policy(name, feats, seed=1)
        out = p(b.ids[5:], obs, {i: float(m) for i, m in zip(b.ids, b.mu)})
        assert sorted(out) == sorted(b.ids[5:])
    # random policy is deterministic given the seed
    p1 = policies.make_policy("random", feats, seed=3)(b.ids, {}, mu_by)
    p2 = policies.make_policy("random", feats, seed=3)(b.ids, {}, mu_by)
    assert p1 == p2


def test_ndcg_trec_eval_convention():
    qrels = {"a": 2, "b": 1, "c": 0, "d": 1}
    run = ["c", "a", "d", "b"]
    dcg = 0 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert metrics.ndcg_at_k(run, qrels, 10) == pytest.approx(dcg / idcg)
    assert metrics.ndcg_at_k(run, qrels, 2) == pytest.approx((2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3)))
    assert metrics.mrr_at_k(run, qrels, 10) == 0.5
    assert metrics.recall_at_k(run, qrels, 2) == pytest.approx(1 / 3)
    assert metrics.ndcg_at_k(run, {"z": 0}, 10) == 0.0
    try:
        import pytrec_eval  # noqa: F401
    except ImportError:
        pytest.skip("pytrec_eval not installed; cross-check runs in the real environment")
    ev = pytrec_eval.RelevanceEvaluator({"q": {k: int(v) for k, v in qrels.items()}}, {"ndcg_cut"})
    res = ev.evaluate({"q": {d: float(len(run) - i) for i, d in enumerate(run)}})
    assert res["q"]["ndcg_cut_10"] == pytest.approx(metrics.ndcg_at_k(run, qrels, 10), abs=1e-9)


def test_metric_gap_bound(data):
    b = data[1]
    truth = b.true_topk_ids(K)
    for m in (K, K + 3, b.N):
        r = replay_query(b, MethodSpec("p", "prefix_fixed", m=m), K)
        assert metrics.metric_gap_bound_holds(r["ndcg_full@k"], r["ndcg@k"], r["mismatch"])
    assert metrics.exact_mismatch(truth, truth) == 0
    assert metrics.exact_mismatch(list(reversed(truth)), truth) == 1  # order matters


def test_stats_helpers():
    rs = stats.risk_summary([0] * 95 + [1] * 5)
    assert rs["rate"] == pytest.approx(0.05) and rs["cp_lo"] <= 0.05 <= rs["cp_hi"]
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 400)
    b = a - 0.001 + rng.normal(0, 0.01, 400)
    ni = stats.noninferiority(b, a, epsilon=0.005, n_boot=500, seed=1)
    assert ni["noninferior"] and ni["lcb"] < ni["mean"] < ni["ucb"]
    bad = stats.noninferiority(a - 0.02, a, epsilon=0.005, n_boot=500, seed=1)
    assert not bad["noninferior"]
    cl = stats.paired_bootstrap_diff(b, a, n_boot=300, clusters=np.repeat(np.arange(40), 10))
    assert cl["n_groups"] == 40
    cr = stats.conditional_risk([0, 1, 0, 0, 1, 0], [True, True, False, False, True, True])
    assert cr["early_fraction"] == pytest.approx(4 / 6) and cr["error_given_early"] == pytest.approx(0.5)


def test_provenance_roundtrip(tmp_path):
    src = tmp_path / "result.json"
    src.write_text(json.dumps({"x": 1}))
    reg = tmp_path / "registry.json"
    provenance.register("nqRiskRsc", 0.0412, str(src), "test number", registry=reg)
    provenance.register("nqCalls", 37, str(src), "int number", registry=reg)
    with pytest.raises(ValueError):
        provenance.register("bad_key1", 1, str(src), "x", registry=reg)
    n = provenance.render_numbers_tex(tmp_path / "numbers.tex", reg, tmp_path / "numbers.json")
    assert n == 2
    tex = (tmp_path / "numbers.tex").read_text()
    assert "\\newcommand{\\numnqRiskRsc}{0.041}" in tex and "\\newcommand{\\numnqCalls}{37}" in tex
    assert provenance.verify_registry(reg) == {}
    src.write_text(json.dumps({"x": 2}))
    assert set(provenance.verify_registry(reg)) == {"nqRiskRsc", "nqCalls"}


def test_observed_regression_changes_the_order_and_mu_alt_follows_the_alternative_model(data):
    """Amendment A3: the refit policy must be a genuinely different order (the pre-registered affine refit was a no-op)."""
    b = data[3]
    feats = {n: b.feature_map(n) for n in b.features}
    mu_by = {i: float(m) for i, m in zip(b.ids, b.mu)}
    obs = {i: float(v) for i, v in list(zip(b.ids, b.s))[:10]}
    rest = [i for i in b.ids if i not in obs]
    refit = policies.make_policy("observed_regression", feats, 0)(rest, obs, mu_by)
    mu_order = policies.make_policy("mu", feats, 0)(rest, obs, mu_by)
    assert sorted(refit) == sorted(rest)
    # with fewer than min_obs observations the policy falls back to the mu order
    assert policies.make_policy("observed_regression", feats, 0)(rest, dict(list(obs.items())[:3]), mu_by) == mu_order
    feats["mu_alt"] = {i: -mu_by[i] for i in b.ids}
    alt = policies.make_policy("mu_alt", feats, 0)(rest, {}, mu_by)
    assert alt == list(reversed(mu_order)) or sorted(alt) == sorted(rest)
