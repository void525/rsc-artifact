"""Render paper/numbers.tex from the provenance registry (node N12/N13).

Also registers *analytic* constants that the paper quotes (calibration
indices, pair counts, throughput requirement), computed here from the frozen
manifests rather than typed by hand.  Refuses to run in --synthetic mode (smoke
numbers must never reach the manuscript); the smoke pipeline renders a preview
into runs/smoke/paper_preview/numbers.tex instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank.calibrate import family_delta, finite_population_index, marginal_index, pac_index  # noqa: E402
from rscrank.io import read_json, write_json  # noqa: E402
from rscrank.provenance import macro_key, register, render_numbers_tex, verify_registry  # noqa: E402


def main() -> int:
    args = base_parser("paper numbers").parse_args()
    cfg = load_config(args)
    synthetic = bool(cfg["smoke"]["synthetic"])
    reg_path = ROOT / cfg["paths"]["runs"] / "provenance" / "registry.json"
    pr = cfg["protocol"]
    a0 = pr["alpha_main"]
    delta_cfg = family_delta(pr["delta_family"], pr["n_main_configs"])
    # analytic constants from the frozen splits manifest
    sm = ROOT / cfg["paths"]["runs"] / "data" / "splits_manifest.json"
    if sm.exists():
        man = read_json(sm)
        for d, m in man["domains"].items():
            if m.get("n_cal"):
                n, M = int(m["n_cal"]), int(m["n_test"])
                pre = macro_key(d)
                register(f"{pre}NCal", n, str(sm), f"{d} calibration queries", 0, reg_path)
                register(f"{pre}NTestSplit", M, str(sm), f"{d} test queries", 0, reg_path)
                register(f"{pre}IdxMarginal", marginal_index(n, a0), str(sm), f"{d} marginal index at alpha={a0}", 0, reg_path)
                register(f"{pre}IdxPac", pac_index(n, a0, delta_cfg), str(sm), f"{d} PAC index at alpha={a0}, delta={delta_cfg}", 0, reg_path)
                register(f"{pre}IdxFinite", finite_population_index(n, M, a0, delta_cfg), str(sm), f"{d} finite-population index", 0, reg_path)
            if m.get("n_train") is not None:
                register(f"{macro_key(d)}NTrain", int(m["n_train"]), str(sm), f"{d} training queries after dedup", 0, reg_path)
                register(f"{macro_key(d)}NDev", int(m["n_dev"]), str(sm), f"{d} dev queries", 0, reg_path)
    cm = ROOT / cfg["paths"]["runs"] / "candidates" / "manifest.json"
    if cm.exists():
        man = read_json(cm)
        total_q = sum(v["n_queries"] for v in man["domains"].values())
        register("CoreQueries", total_q, str(cm), "total queries with candidates", 0, reg_path)
        register("CorePairs", total_q * man["N"], str(cm), "total (query, candidate) pairs", 0, reg_path)
        register("CandidateN", man["N"], str(cm), "candidates per query", 0, reg_path)
        register("CandidateBackend", man["backend"], str(cm), "first-stage backend", 0, reg_path)
        for d, v in man["domains"].items():
            register(f"{macro_key(d)}BmTwoFiveNdcg", v["bm25_ndcg10_test"], str(cm), f"{d} BM25 nDCG@10 on the test split", 4, reg_path)
            register(f"{macro_key(d)}RecallAtN", v[f"recall@{man['N']}_test"], str(cm), f"{d} first-stage recall@N", 4, reg_path)
    for t in (cfg["teachers"]["main"], cfg["teachers"]["second"]):
        pm = ROOT / cfg["paths"]["runs"] / "pilot" / f"throughput_{t}.json"
        if pm.exists():
            v = read_json(pm)
            pre = macro_key(t)
            register(f"{pre}PairsPerS", v["v_measured_pairs_per_s"], str(pm), f"measured teacher throughput {t} (pilot sample)", 1, reg_path)
            register(f"{pre}PeakMemGb", v.get("peak_mem_gb", float("nan")), str(pm), f"peak GPU memory {t}", 1, reg_path)
            rec = v.get("recommendation") or {}
            register(f"{pre}BatchMode", str(rec.get("mode", "")), str(pm), f"batching mode chosen by the pilot for {t}", 0, reg_path)
            register(f"{pre}BatchSize", int(rec.get("batch_size", 0)), str(pm), f"micro-batch rows chosen by the pilot for {t}", 0, reg_path)
            ran = [r for r in v.get("settings", []) if not r.get("oom")]
            fast = [r for r in ran if r["mode"] == "fast"]
            can = [r for r in ran if r["mode"] == "canonical"]
            if fast:
                register(f"{pre}FastBestPairsPerS", max(r["pairs_per_s"] for r in fast), str(pm), f"best fast-mode throughput {t}", 1, reg_path)
                register(f"{pre}FastMaxAbsDiff", max(r["invariance"]["max_abs_diff"] for r in fast), str(pm), f"fast mode: max |score difference| between batch compositions {t}", 3, reg_path)
                register(f"{pre}FastMaxTopkFlips", max(r["invariance"]["topk_order_changed_queries"] for r in fast), str(pm), f"fast mode: max number of audited queries (of {v['invariance_protocol']['queries']}) whose top-k order changed with the batch composition {t}", 0, reg_path)
                register(f"{pre}PilotInvQueries", int(v["invariance_protocol"]["queries"]), str(pm), "queries in the pilot batch-invariance set", 0, reg_path)
                register(f"{pre}PilotInvPairs", int(v["invariance_protocol"]["pairs"]), str(pm), "pairs in the pilot batch-invariance set", 0, reg_path)
            if can:
                register(f"{pre}CanonicalMaxAbsDiff", max(r["invariance"]["max_abs_diff"] for r in can), str(pm), f"canonical mode: max |score difference| between batch compositions {t}", 3, reg_path)
                register(f"{pre}CanonicalBestPairsPerS", max(r["pairs_per_s"] for r in can), str(pm), f"best canonical-mode throughput {t}", 1, reg_path)
        tm = ROOT / cfg["paths"]["runs"] / "teacher" / t / "manifest.json"
        if tm.exists():
            v = read_json(tm)
            pre = macro_key(t)
            register(f"{pre}CacheHours", v["seconds"] / 3600, str(tm), f"GPU hours spent caching {t}", 1, reg_path)
            register(f"{pre}CachePairs", v["pairs"], str(tm), f"query-document pairs cached for {t} (all splits and domains)", 0, reg_path)
            register(f"{pre}CachePairsPerS", v["pairs"] / v["seconds"] if v.get("seconds") else float("nan"), str(tm), f"effective caching throughput of {t} = cached pairs / wall-clock seconds (incl. tokenisation and I/O)", 1, reg_path)
        dm = ROOT / cfg["paths"]["runs"] / "teacher" / t / "determinism.json"
        if dm.exists():
            v = read_json(dm)
            pre = macro_key(t)
            register(f"{pre}DetMaxDiff", v["max_abs_diff"], str(dm), f"max |re-score - cache| {t}", 4, reg_path)
            register(f"{pre}DetTopkChanged", v["topk_order_changed_queries"], str(dm), f"queries whose top-k order changed on re-scoring {t}", 0, reg_path)
    for t in (cfg["teachers"]["main"], cfg["teachers"]["second"]):
        for d in main_domains(cfg):
            lm = ROOT / cfg["paths"]["runs"] / "latency" / t / f"{d}.json"
            if lm.exists():
                v = read_json(lm)
                pre = macro_key(d, t)
                for reg in ("controlled", "tuned"):
                    R = v["regimes"].get(reg, {})
                    for m in ("full", "rsc", "rsc_blockers", "prefix_uniform", "prefix_adaptive"):
                        if m in R and isinstance(R[m], dict):
                            register(macro_key(pre, "Lat", reg, m, "Ms"), 1000 * R[m]["latency_mean_s"], str(lm), f"{t}/{d} {reg} mean latency {m} (ms)", 0, reg_path)
                            register(macro_key(pre, "LatPninetyfive", reg, m, "Ms"), 1000 * R[m]["latency_p95_s"], str(lm), f"{t}/{d} {reg} p95 latency {m} (ms)", 0, reg_path)
                            if R[m].get("agree_with_cache") is not None:
                                register(macro_key(pre, "LatAgree", reg, m), R[m]["agree_with_cache"], str(lm), f"{t}/{d} {reg} fraction of queries whose live top-k equals the cached teacher top-k ({m})", 3, reg_path)
                    for var in ("rsc", "rsc_blockers"):
                        key = f"{var}_vs_full_latency_reduction"
                        if key in R:
                            register(macro_key(pre, "LatReduction", reg, var, "Pct"), 100 * R[key], str(lm), f"{t}/{d} {reg} latency reduction {var} vs full (%)", 1, reg_path)
    # aggregate latency ratios over the measured configurations (ratio = method mean latency / full-teacher mean latency)
    if not synthetic:
        lat_files = sorted(f for f in (ROOT / cfg["paths"]["runs"] / "latency").glob("*/*.json") if f.name != "aggregate.json")
        ratios = {}
        agree_by_teacher = {}
        for lf in lat_files:
            v = read_json(lf)
            for reg in ("controlled", "tuned"):
                R = v["regimes"].get(reg, {})
                if "full" not in R:
                    continue
                for m in ("rsc", "rsc_blockers", "prefix_uniform", "prefix_adaptive"):
                    if m in R and isinstance(R[m], dict):
                        ratios.setdefault((reg, m), []).append(R[m]["latency_mean_s"] / R["full"]["latency_mean_s"])
                if reg == "tuned" and R["full"].get("agree_with_cache") is not None:
                    agree_by_teacher.setdefault(v["teacher"], []).append(R["full"]["agree_with_cache"])
        agg_path = ROOT / cfg["paths"]["runs"] / "latency" / "aggregate.json"
        write_json(agg_path, {"sources": [str(f.relative_to(ROOT)) for f in lat_files], "ratios": {f"{reg}/{m}": xs for (reg, m), xs in ratios.items()}, "tuned_full_agree_by_teacher": agree_by_teacher})
        for (reg, m), xs in ratios.items():
            register(macro_key("AggLatRatio", reg, m, "Min"), min(xs), str(agg_path), f"smallest ratio of mean {m} latency to the full teacher's ({reg} regime) over measured configurations", 2, reg_path)
            register(macro_key("AggLatRatio", reg, m, "Max"), max(xs), str(agg_path), f"largest ratio of mean {m} latency to the full teacher's ({reg} regime) over measured configurations", 2, reg_path)
            register(macro_key("AggLatRatio", reg, m, "N"), len(xs), str(agg_path), f"number of configurations with a {m} latency measurement ({reg})", 0, reg_path)
        for t, xs in agree_by_teacher.items():
            register(macro_key("AggTunedAgree", t, "Min"), min(xs), str(agg_path), f"{t}: smallest fraction of queries whose tuned-regime full-teacher top-k equals the cached (micro-batch 16) top-k", 2, reg_path)
            register(macro_key("AggTunedAgree", t, "Max"), max(xs), str(agg_path), f"{t}: largest such fraction", 2, reg_path)
    # N18: N = 200 vs N = 100 on the same test subset (runs/N200/compare.json = main heads, list-length
    # feature clamped; runs/N200h/compare.json = heads retrained at N = 200). Both use the full calibration split.
    if not synthetic:
        def _rsc_rows(res):
            r = res.get("rsc")
            if isinstance(r, dict) and "N200" in r and "N100" in r:
                return r
            return None

        cmp_path = ROOT / "runs" / "N200" / "compare.json"
        fr200, fr100, red200, risk200, drop = [], [], [], [], []
        if cmp_path.exists():
            C = read_json(cmp_path)
            for key, res in C["configs"].items():
                t, d = key.split("/")
                pre = macro_key(d, t)
                for name, r in res.items():
                    if isinstance(r, dict) and "N200" in r:
                        for nn in ("N200", "N100"):
                            register(f"{pre}{macro_key(nn)}{macro_key(name)}Calls", r[nn]["calls_mean"], str(cmp_path), f"{key}: mean calls of {name} at {nn} on the shared test subset", 1, reg_path)
                            register(f"{pre}{macro_key(nn)}{macro_key(name)}CallsPct", 100 * r[nn]["calls_frac_of_N"], str(cmp_path), f"{key}: calls of {name} as % of N at {nn}", 1, reg_path)
                            register(f"{pre}{macro_key(nn)}{macro_key(name)}Risk", r[nn]["risk"], str(cmp_path), f"{key}: exact-mismatch risk of {name} at {nn}", 4, reg_path)
                        if name == "rsc":
                            fr200.append(100 * r["N200"]["calls_frac_of_N"])
                            fr100.append(100 * r["N100"]["calls_frac_of_N"])
                            red200.append(100 * (1 - r["N200"]["calls_frac_of_N"]))
                            risk200.append(r["N200"]["risk"])
                            drop.append(100 * (r["N100"]["calls_frac_of_N"] - r["N200"]["calls_frac_of_N"]))
                register(f"{pre}NTwoHundredSubsetTest", res["n_subset_test"], str(cmp_path), f"{key}: test queries in the N = 200 subset", 0, reg_path)
                if "E9_N200" in res:
                    register(f"{pre}NTwoHundredOvershoot", res["E9_N200"].get("overshoot_mean", float("nan")), str(cmp_path), f"{key}: mean batching overshoot at N = 200 (covered queries)", 1, reg_path)
                    register(f"{pre}NTwoHundredBand", res["E9_N200"].get("band_mean", float("nan")), str(cmp_path), f"{key}: mean band at N = 200 (covered queries)", 2, reg_path)
            if fr200:
                register("AggNTwoHundredRscCallsPctMin", min(fr200), str(cmp_path), "smallest RSC calls as % of N at N = 200 over the subset configurations (main heads, length feature clamped)", 1, reg_path)
                register("AggNTwoHundredRscCallsPctMax", max(fr200), str(cmp_path), "largest RSC calls as % of N at N = 200 (main heads, length feature clamped)", 1, reg_path)
                register("AggNOneHundredSubsetRscCallsPctMin", min(fr100), str(cmp_path), "smallest RSC calls as % of N at N = 100 on the same queries", 1, reg_path)
                register("AggNOneHundredSubsetRscCallsPctMax", max(fr100), str(cmp_path), "largest RSC calls as % of N at N = 100 on the same queries", 1, reg_path)
                register("AggNTwoHundredReductionPctMin", min(red200), str(cmp_path), "smallest RSC call reduction (%) at N = 200 (main heads, length feature clamped)", 1, reg_path)
                register("AggNTwoHundredReductionPctMax", max(red200), str(cmp_path), "largest RSC call reduction (%) at N = 200 (main heads, length feature clamped)", 1, reg_path)
                register("AggNTwoHundredRscRiskMin", min(risk200), str(cmp_path), "smallest RSC exact-mismatch risk at N = 200 (main heads, length feature clamped)", 4, reg_path)
                register("AggNTwoHundredRscRiskMax", max(risk200), str(cmp_path), "largest RSC exact-mismatch risk at N = 200 (main heads, length feature clamped)", 4, reg_path)
                register("AggNTwoHundredCallsPctDropMin", min(drop), str(cmp_path), "smallest drop, in percentage points, of RSC calls as % of N from N = 100 to N = 200 on the same queries (main heads)", 1, reg_path)
                register("AggNTwoHundredCallsPctDropMax", max(drop), str(cmp_path), "largest drop, in percentage points, of RSC calls as % of N from N = 100 to N = 200 on the same queries (main heads)", 1, reg_path)
        h_path = ROOT / "runs" / "N200h" / "compare.json"
        h_fr, h_risk, h_drop = [], [], []
        if h_path.exists():
            H = read_json(h_path)
            for res in H["configs"].values():
                r = _rsc_rows(res)
                if r is None:
                    continue
                h_fr.append(100 * r["N200"]["calls_frac_of_N"])
                h_risk.append(r["N200"]["risk"])
                h_drop.append(100 * (r["N100"]["calls_frac_of_N"] - r["N200"]["calls_frac_of_N"]))
            if h_fr:
                register("AggNTwoHundredHeadsRscCallsPctMin", min(h_fr), str(h_path), "smallest RSC calls as % of N at N = 200 with heads retrained at N = 200", 1, reg_path)
                register("AggNTwoHundredHeadsRscCallsPctMax", max(h_fr), str(h_path), "largest RSC calls as % of N at N = 200 with heads retrained at N = 200", 1, reg_path)
                register("AggNTwoHundredHeadsRscRiskMin", min(h_risk), str(h_path), "smallest RSC exact-mismatch risk at N = 200 with heads retrained at N = 200", 4, reg_path)
                register("AggNTwoHundredHeadsRscRiskMax", max(h_risk), str(h_path), "largest RSC exact-mismatch risk at N = 200 with heads retrained at N = 200", 4, reg_path)
                register("AggNTwoHundredHeadsCallsPctDropMin", min(h_drop), str(h_path), "smallest drop, in percentage points, of RSC calls as % of N from N = 100 to N = 200 (heads retrained at N = 200)", 1, reg_path)
                register("AggNTwoHundredHeadsCallsPctDropMax", max(h_drop), str(h_path), "largest drop, in percentage points, of RSC calls as % of N from N = 100 to N = 200 (heads retrained at N = 200)", 1, reg_path)
        both_fr = fr200 + h_fr
        both_risk = risk200 + h_risk
        if both_fr:
            both_path = ROOT / "runs" / "N200" / "heads_aggregate.json"
            write_json(both_path, {"sources": [str(p.relative_to(ROOT)) for p in (cmp_path, h_path) if p.exists()], "calls_pct": both_fr, "risk": both_risk})
            register("AggNTwoHundredBothRscCallsPctMin", min(both_fr), str(both_path), "smallest RSC calls as % of N at N = 200 over clamped main heads and heads retrained at N = 200", 1, reg_path)
            register("AggNTwoHundredBothRscCallsPctMax", max(both_fr), str(both_path), "largest RSC calls as % of N at N = 200 over clamped main heads and heads retrained at N = 200", 1, reg_path)
            register("AggNTwoHundredBothRscRiskMin", min(both_risk), str(both_path), "smallest RSC exact-mismatch risk at N = 200 over both head settings", 4, reg_path)
            register("AggNTwoHundredBothRscRiskMax", max(both_risk), str(both_path), "largest RSC exact-mismatch risk at N = 200 over both head settings", 4, reg_path)
    # policy envelope (review round 1): trajectory-specific critical values per order vs the rank-sufficient value
    if not synthetic:
        env_files = sorted((ROOT / cfg["paths"]["runs"] / "policy_envelope").glob("*/*.json")) if (ROOT / cfg["paths"]["runs"] / "policy_envelope").exists() else []
        shares, untested, envA, shares_r, bonf_extra, bonf_calls, rule_tr, rule_rsc = [], [], [], [], [], [], {}, {}
        for ef in env_files:
            v = read_json(ef)
            pre = macro_key(v["domain"], v["teacher"])
            th, ps, cl = v["thresholds"], v["premium_split"], v["calls_under_mu_order"]
            register(f"{pre}EnvLambdaMu", th["lambda_mu"], str(ef), f"{v['teacher']}/{v['domain']}: threshold calibrated on the mu-trajectory statistic", 3, reg_path)
            register(f"{pre}EnvLambdaEnv", th["lambda_env"], str(ef), f"{v['teacher']}/{v['domain']}: threshold calibrated on the envelope (max over the {len(th['env_orders'])} tested orders/schedules) of the trajectory statistics", 3, reg_path)
            register(f"{pre}EnvLambdaRsc", th["lambda_rsc"], str(ef), f"{v['teacher']}/{v['domain']}: rank-sufficient threshold", 3, reg_path)
            register(f"{pre}EnvCallsMu", cl["mu"]["calls_mean"], str(ef), f"{v['teacher']}/{v['domain']}: mean calls under the mu order at the mu-trajectory threshold", 1, reg_path)
            register(f"{pre}EnvCallsEnv", cl["env"]["calls_mean"], str(ef), f"{v['teacher']}/{v['domain']}: mean calls under the mu order at the envelope threshold", 1, reg_path)
            register(f"{pre}EnvCallsRsc", cl["rsc"]["calls_mean"], str(ef), f"{v['teacher']}/{v['domain']}: mean calls under the mu order at the rank-sufficient threshold", 1, reg_path)
            register(f"{pre}EnvPremiumCalls", ps["total_premium_calls"], str(ef), f"{v['teacher']}/{v['domain']}: policy-uniform premium in calls (RSC - mu-trajectory threshold)", 1, reg_path)
            register(f"{pre}EnvPremiumUntestedCalls", ps["spent_on_untested_states_calls"], str(ef), f"{v['teacher']}/{v['domain']}: part of the premium beyond the envelope of the tested orders (calls)", 2, reg_path)
            if ps["share_tested"] is not None:
                register(f"{pre}EnvShareTestedPct", 100 * ps["share_tested"], str(ef), f"{v['teacher']}/{v['domain']}: share of the premium needed to protect the tested orders simultaneously (%)", 0, reg_path)
                shares.append(100 * ps["share_tested"])
            untested.append(ps["spent_on_untested_states_calls"])
            envA.append(th["env_equals_A_frac_test"])
            register(f"{pre}EnvEqualsAPct", 100 * th["env_equals_A_frac_test"], str(ef), f"{v['teacher']}/{v['domain']}: test queries on which the envelope of the tested orders attains the rank-sufficient value (%)", 0, reg_path)
            register(f"{pre}EnvUnionAboveLamMu", th["env_test_frac_above_lambda_mu"], str(ef), f"{v['teacher']}/{v['domain']}: fraction of test queries whose envelope exceeds the mu-trajectory threshold (union failure of the transferred trajectory threshold, from statistics)", 4, reg_path)
            register(f"{pre}EnvNOrders", len(th["env_orders"]), str(ef), "number of orders/schedules in the envelope", 0, reg_path)
            if "bonferroni" in v:
                bo = v["bonferroni"]
                register(f"{pre}BonfK", bo["K"], str(ef), "number of tested trajectories K in the Bonferroni per-order scheme", 0, reg_path)
                register(f"{pre}BonfCallsMu", bo["calls_mu_order"], str(ef), f"{v['teacher']}/{v['domain']}: mean calls under the mu order with the mu-trajectory threshold calibrated at alpha/K (Bonferroni union guarantee over the K tested orders)", 1, reg_path)
                register(f"{pre}BonfUnionFail", bo["union_test_frac_fail"], str(ef), f"{v['teacher']}/{v['domain']}: union failure fraction of the K Bonferroni per-order thresholds on the test split", 4, reg_path)
                register(f"{pre}BonfExtraCallsVsRsc", -bo["rsc_minus_bonf_calls"], str(ef), f"{v['teacher']}/{v['domain']}: extra calls of the Bonferroni per-order scheme over the single rank-sufficient threshold (mu order)", 1, reg_path)
                bonf_extra.append(-bo["rsc_minus_bonf_calls"]); bonf_calls.append(bo["calls_mu_order"])
            for rname, rv in v.get("selection_rules", {}).items():
                register(f"{pre}Rule{macro_key(rname)}RiskRsc", rv["risk_rsc"], str(ef), f"{v['teacher']}/{v['domain']}: RSC risk under the query-dependent order-selection rule '{rname}'", 4, reg_path)
                register(f"{pre}Rule{macro_key(rname)}RiskTrajRecal", rv["risk_per_order_recalibrated_traj"], str(ef), f"{v['teacher']}/{v['domain']}: risk of per-order recalibrated trajectory thresholds under the rule '{rname}'", 4, reg_path)
                rule_tr.setdefault(rname, []).append(rv["risk_per_order_recalibrated_traj"]); rule_rsc.setdefault(rname, []).append(rv["risk_rsc"])
            if "realistic_envelope" in v:
                re_ = v["realistic_envelope"]
                register(f"{pre}EnvRealisticNOrders", len(re_["orders"]), str(ef), "orders in the realistic envelope (mu-based and adaptive orders only)", 0, reg_path)
                register(f"{pre}EnvRealisticUnionAboveLamMu", re_["env_test_frac_above_lambda_mu"], str(ef), f"{v['teacher']}/{v['domain']}: union failure of the transferred mu-trajectory threshold over the realistic orders only", 4, reg_path)
                if ps.get("share_realistic") is not None:
                    register(f"{pre}EnvShareRealisticPct", 100 * ps["share_realistic"], str(ef), f"{v['teacher']}/{v['domain']}: share of the premium needed to protect the realistic orders simultaneously (%)", 0, reg_path)
                    shares_r.append(100 * ps["share_realistic"])
            mu_o = v["orders"].get("mu", {})
            if mu_o:
                register(f"{pre}TrajNegInfPct", 100 * mu_o["neg_inf_frac_test"], str(ef), f"{v['teacher']}/{v['domain']}: test queries whose mu trajectory has no unsafe checkpoint (A^mu = -inf, excluded from PoU) (%)", 1, reg_path)
        if env_files:
            agg_path = ROOT / cfg["paths"]["runs"] / "policy_envelope" / "aggregate.json"
            write_json(agg_path, {"sources": [str(f.relative_to(ROOT)) for f in env_files], "share_tested_pct": shares, "untested_calls": untested, "env_equals_A_frac": envA})
            if shares:
                register("AggEnvShareTestedPctMin", min(shares), str(agg_path), "smallest share of the policy-uniform premium needed for the tested orders (%)", 0, reg_path)
                register("AggEnvShareTestedPctMax", max(shares), str(agg_path), "largest share of the policy-uniform premium needed for the tested orders (%)", 0, reg_path)
            register("AggEnvPremiumUntestedCallsMin", min(untested), str(agg_path), "smallest part of the premium beyond the tested envelope (calls)", 2, reg_path)
            register("AggEnvPremiumUntestedCallsMax", max(untested), str(agg_path), "largest part of the premium beyond the tested envelope (calls)", 2, reg_path)
            if shares_r:
                register("AggEnvShareRealisticPctMin", min(shares_r), str(agg_path), "smallest share of the premium needed for the realistic orders (%)", 0, reg_path)
                register("AggEnvShareRealisticPctMax", max(shares_r), str(agg_path), "largest share of the premium needed for the realistic orders (%)", 0, reg_path)
            if rule_tr:
                allv = [x for xs in rule_tr.values() for x in xs]
                register("AggRuleRiskTrajRecalMin", min(allv), str(agg_path), "smallest risk of per-order recalibrated trajectory thresholds under the two realistic query-dependent selection rules", 4, reg_path)
                register("AggRuleRiskTrajRecalMax", max(allv), str(agg_path), "largest risk of per-order recalibrated trajectory thresholds under the two realistic query-dependent selection rules", 4, reg_path)
                allr = [x for xs in rule_rsc.values() for x in xs]
                register("AggRuleRiskRscMax", max(allr), str(agg_path), "largest RSC risk under the two realistic query-dependent selection rules", 4, reg_path)
            if bonf_extra:
                register("AggBonfExtraCallsVsRscMin", min(bonf_extra), str(agg_path), "smallest extra calls of the Bonferroni per-order scheme over the rank-sufficient threshold", 1, reg_path)
                register("AggBonfExtraCallsVsRscMax", max(bonf_extra), str(agg_path), "largest extra calls of the Bonferroni per-order scheme over the rank-sufficient threshold", 1, reg_path)
                register("AggBonfCallsMuMin", min(bonf_calls), str(agg_path), "smallest mean calls of the Bonferroni mu-trajectory threshold (alpha/K)", 1, reg_path)
                register("AggBonfCallsMuMax", max(bonf_calls), str(agg_path), "largest mean calls of the Bonferroni mu-trajectory threshold (alpha/K)", 1, reg_path)
            register("AggEnvEqualsAPctMin", 100 * min(envA), str(agg_path), "smallest share of test queries on which the tested envelope attains A_k1 (%)", 0, reg_path)
            register("AggEnvEqualsAPctMax", 100 * max(envA), str(agg_path), "largest share of test queries on which the tested envelope attains A_k1 (%)", 0, reg_path)
    # throughput under concurrency (A4): per configuration and per concurrency, GPU-seconds-per-query ratios to the full teacher
    if not synthetic:
        thr_files = sorted((ROOT / cfg["paths"]["runs"] / "throughput").glob("*/*.json")) if (ROOT / cfg["paths"]["runs"] / "throughput").exists() else []
        agg_thr = {}
        for tf in thr_files:
            v = read_json(tf)
            pre = macro_key(v["domain"], v["teacher"])
            for m, R in v["methods"].items():
                for c, r in R.items():
                    f = v["methods"].get("full", {}).get(c)
                    register(macro_key(pre, "Thr", "C" + c, m, "Qps"), r["throughput_qps"], str(tf), f"{v['teacher']}/{v['domain']}: throughput of {m} with {c} queries in flight (queries/s)", 2, reg_path)
                    register(macro_key(pre, "Thr", "C" + c, m, "GpuMs"), 1000 * r["gpu_s_per_query"], str(tf), f"{v['teacher']}/{v['domain']}: GPU-seconds per query of {m} with {c} in flight (ms)", 0, reg_path)
                    register(macro_key(pre, "Thr", "C" + c, m, "LatMs"), 1000 * r["latency_mean_s"], str(tf), f"{v['teacher']}/{v['domain']}: mean per-query latency under load of {m} with {c} in flight (ms)", 0, reg_path)
                    if f and m != "full":
                        ratio = r["gpu_s_per_query"] / f["gpu_s_per_query"]
                        register(macro_key(pre, "Thr", "C" + c, m, "Ratio"), ratio, str(tf), f"{v['teacher']}/{v['domain']}: GPU-seconds per query of {m} / full, {c} in flight", 2, reg_path)
                        agg_thr.setdefault((c, m), []).append(ratio)
                        agg_thr.setdefault((c, m, v["teacher"]), []).append(ratio)
                    if r.get("dummy_row_fraction") is not None:
                        register(macro_key(pre, "Thr", "C" + c, m, "DummyPct"), 100 * r["dummy_row_fraction"], str(tf), f"{v['teacher']}/{v['domain']}: dummy-row fraction of {m} with {c} in flight (%)", 0, reg_path)
        if agg_thr:
            agg_path = ROOT / cfg["paths"]["runs"] / "throughput" / "aggregate.json"
            write_json(agg_path, {"sources": [str(f.relative_to(ROOT)) for f in thr_files], "gpu_s_ratio_vs_full": {"/".join(["C" + k_[0]] + list(k_[1:])): xs for k_, xs in agg_thr.items()}})
            for key_, xs in agg_thr.items():
                if len(key_) == 2:
                    c, m = key_
                    register(macro_key("AggThr", "C" + c, m, "RatioMin"), min(xs), str(agg_path), f"smallest GPU-seconds-per-query ratio {m}/full with {c} in flight over configurations", 2, reg_path)
                    register(macro_key("AggThr", "C" + c, m, "RatioMax"), max(xs), str(agg_path), f"largest GPU-seconds-per-query ratio {m}/full with {c} in flight over configurations", 2, reg_path)
                    register(macro_key("AggThr", "C" + c, m, "N"), len(xs), str(agg_path), f"configurations measured ({m}, {c} in flight)", 0, reg_path)
                else:
                    c, m, t = key_
                    register(macro_key("AggThr", t, "C" + c, m, "RatioMin"), min(xs), str(agg_path), f"{t}: smallest GPU-seconds-per-query ratio {m}/full with {c} in flight over its domains", 2, reg_path)
                    register(macro_key("AggThr", t, "C" + c, m, "RatioMax"), max(xs), str(agg_path), f"{t}: largest GPU-seconds-per-query ratio {m}/full with {c} in flight over its domains", 2, reg_path)
    # review round 3 (amendment A7, scripts/18_order_free_baselines.py): declared-family envelopes with the blocker
    # controllers' own critical values, Bonferroni at (alpha/K, delta/K), set-completion prefixes under every order,
    # same-sample covered comparison.  Savings are registered as positive magnitudes with the direction in the key.
    if not synthetic:
        r3_files = sorted((ROOT / cfg["paths"]["runs"] / "review3").glob("*/*.json")) if (ROOT / cfg["paths"]["runs"] / "review3").exists() else []
        A3 = {}

        def _acc(key, val):
            A3.setdefault(key, []).append(val)

        for rf in r3_files:
            v = read_json(rf)
            pre = macro_key(v["domain"], v["teacher"])
            cfgname = f"{v['teacher']}/{v['domain']}"
            env, bonf, lad, po = v["envelope"], v["bonferroni"], v["calls_under_mu"], v["per_order"]
            e13, eR9 = env["all13"], env["realistic9"]
            register(f"{pre}RThreeEnvLambda", e13["lambda"], str(rf), f"{cfgname}: threshold calibrated on the envelope over the 13 declared policies (ten orders, three blocker controllers by their own critical values)", 3, reg_path)
            register(f"{pre}RThreeEnvTenLambda", env["orders10"]["lambda"], str(rf), f"{cfgname}: envelope threshold over the ten lambda-independent orders (equals the round-2 envelope threshold)", 3, reg_path)
            register(f"{pre}RThreeEnvCallsMu", lad["env_all13"], str(rf), f"{cfgname}: mean calls under the mu order at the 13-policy envelope threshold", 1, reg_path)
            register(f"{pre}RThreeEnvRealCallsMu", lad["env_realistic9"], str(rf), f"{cfgname}: mean calls under the mu order at the realistic 9-policy envelope threshold", 1, reg_path)
            register(f"{pre}RThreeTrajCallsMu", lad["traj_mu"], str(rf), f"{cfgname}: mean calls under the mu order at the mu-trajectory threshold", 1, reg_path)
            register(f"{pre}RThreeRscCallsMu", lad["rsc"], str(rf), f"{cfgname}: mean calls under the mu order at the rank-sufficient threshold", 1, reg_path)
            register(f"{pre}RThreeRscMinusEnvCalls", e13["rsc_minus_env_calls"], str(rf), f"{cfgname}: extra calls of RSC over the 13-policy envelope threshold (mu order)", 2, reg_path)
            register(f"{pre}RThreeEnvUnion", e13["test_union_stat"], str(rf), f"{cfgname}: test fraction failing under at least one of the 13 declared policies at the envelope threshold", 4, reg_path)
            register(f"{pre}RThreeBonfCallsMu", bonf["all13"]["calls_mu"], str(rf), f"{cfgname}: mean calls under the mu order of the mu-trajectory threshold at (alpha/13, delta/13)", 1, reg_path)
            _acc("RscMinusEnv", e13["rsc_minus_env_calls"])
            _acc("RscMinusEnvReal", eR9["rsc_minus_env_calls"])
            _acc("EnvSharePct", 100 * e13["share_of_premium"])
            _acc("EnvRealSharePct", 100 * eR9["share_of_premium"])
            _acc("EnvUnion", e13["test_union_stat"])
            _acc("BonfCalls", bonf["all13"]["calls_mu"])
            _acc("BonfTenCalls", bonf["orders10"]["calls_mu"])
            _acc("PremiumCalls", lad["rsc"] - lad["traj_mu"])
            _acc("EnvRaised", float(e13["lambda"] > env["orders10"]["lambda"] + 1e-12))
            _acc("ReplayEqualsStat", float(e13["replay_equals_stat_per_query"]))
            _acc("BlkChecks", v["blocker_closed_form_check"]["n_checks"])
            _acc("BlkMismatch", v["blocker_closed_form_check"]["n_mismatch"])
            for bname, bv in v["blocker_vs_frozen"].items():
                _acc("BlkTrueAboveFrozenPct", 100 * bv["test_frac_true_above_frozen"])
            r2 = v.get("round2_envelope_check", {})
            _acc("RoundTwoEnvEqualsTen", float(bool(r2.get("equal"))))
            for key_, val_ in r2.items():
                if key_.startswith("test_frac_") and key_.endswith("_above_round2_lambda"):
                    _acc("BlkExposedRoundTwoPct", 100 * val_)
            for pol, row in po.items():
                pk = macro_key(pol)
                register(f"{pre}RThreeScCalls{pk}", row["sc_adaptive"]["calls_mean"], str(rf), f"{cfgname}: mean calls of the set-completion adaptive prefix under order {pol} (16/8)", 1, reg_path)
                register(f"{pre}RThreeScRisk{pk}", row["sc_adaptive"]["risk"], str(rf), f"{cfgname}: risk of the set-completion adaptive prefix under order {pol}", 4, reg_path)
                register(f"{pre}RThreeRscCalls{pk}", row["rsc"]["calls_mean"], str(rf), f"{cfgname}: mean calls of RSC under order {pol} (16/8)", 1, reg_path)
                d = row["diff_sc_adaptive_minus_rsc"]
                _acc(f"ScMinusRsc{pk}", d["mean"])
                _acc(f"ScMinusRscLcb{pk}", d["lcb"])
                _acc(f"ScMinusRscUcb{pk}", d["ucb"])
                _acc("ScRiskAnyOrder", row["sc_adaptive"]["risk"])
                _acc("ScUniformRiskAnyOrder", row["sc_uniform"]["risk"])
                _acc(f"EnvMinusRsc{pk}", row["diff_env_all13_minus_rsc"]["mean"])
                _acc(f"TrajMinusRsc{pk}", row["diff_traj_minus_rsc"]["mean"])
                if pol != "mu":
                    _acc("CellsRscCheaper", float(d["lcb"] > 0))
                    _acc("CellsScCheaper", float(d["ucb"] < 0))
                    _acc("CellsTie", float(d["lcb"] <= 0 <= d["ucb"]))
            _acc("ScUnion", v["union_over_orders"]["sc_adaptive"])
            _acc("ScBaseFail", v["set_completion_base_failure"]["sc_adaptive"])
            _acc("RscUnion", v["union_over_orders"]["rsc"])
            cp = v["covered_paired"]
            register(f"{pre}RThreeCovBudget", cp["adaptive_budget_mean"], str(rf), f"{cfgname}: mean adaptive-prefix budget on covered test queries", 1, reg_path)
            register(f"{pre}RThreeCovCore", cp["cert_core_mean"], str(rf), f"{cfgname}: mean certification core on the same covered queries", 1, reg_path)
            _acc("CovBudgetBelowCore", -cp["diff_budget_minus_core"]["mean"])
            _acc("CovBudgetBelowCoreNearZero", -cp["diff_budget_minus_core"]["ucb"])
        if r3_files:
            agg3 = ROOT / cfg["paths"]["runs"] / "review3" / "aggregate.json"
            write_json(agg3, {"sources": [str(f.relative_to(ROOT)) for f in r3_files], "values": A3})

            def _mm(name, key, digits, desc, neg=False):
                xs = [(-x if neg else x) for x in A3[key]]
                register(f"AggRThree{name}Min", min(xs), str(agg3), f"smallest {desc}", digits, reg_path)
                register(f"AggRThree{name}Max", max(xs), str(agg3), f"largest {desc}", digits, reg_path)

            _mm("RscMinusEnv", "RscMinusEnv", 2, "extra calls of RSC over the 13-policy envelope threshold (mu order) over configurations")
            _mm("RscMinusEnvReal", "RscMinusEnvReal", 2, "extra calls of RSC over the realistic 9-policy envelope threshold")
            _mm("EnvSharePct", "EnvSharePct", 0, "share of the policy-uniform premium needed by the 13-policy envelope (%)")
            _mm("EnvRealSharePct", "EnvRealSharePct", 0, "share of the premium needed by the realistic 9-policy envelope (%)")
            _mm("EnvUnion", "EnvUnion", 4, "test union failure over the 13 declared policies at the envelope threshold")
            _mm("BonfCalls", "BonfCalls", 1, "mean calls of the 13-policy Bonferroni scheme at (alpha/13, delta/13) under the mu order")
            _mm("BonfTenCalls", "BonfTenCalls", 1, "mean calls of the 10-order Bonferroni scheme at (alpha/10, delta/10) under the mu order")
            _mm("PremiumCalls", "PremiumCalls", 1, "policy-uniform premium (RSC minus mu-trajectory threshold calls, mu order)")
            _mm("BlkTrueAboveFrozenPct", "BlkTrueAboveFrozenPct", 0, "share of test queries on which a frozen blocker trajectory underestimates the controller's critical value (%)")
            _mm("BlkExposedRoundTwoPct", "BlkExposedRoundTwoPct", 1, "share of test queries on which a re-executed blocker controller errs at the round-2 envelope threshold (%)")
            _mm("ScUnion", "ScUnion", 4, "union over the ten orders of the set-completion adaptive prefix failures")
            _mm("RscUnion", "RscUnion", 4, "union over the ten orders of RSC failures (review-3 replay)")
            _mm("ScBaseFail", "ScBaseFail", 4, "fraction of test queries whose calibrated adaptive set misses a winner")
            _mm("CovBudgetBelowCore", "CovBudgetBelowCore", 1, "mean calls by which the adaptive budget undercuts the certification core on the same covered queries")
            register("AggRThreeCovBudgetBelowCoreNearZeroMin", min(A3["CovBudgetBelowCoreNearZero"]), str(agg3), "smallest end nearest zero of the paired 95% interval of (core - adaptive budget) on covered queries", 1, reg_path)
            register("AggRThreeScRiskAnyOrderMax", max(A3["ScRiskAnyOrder"]), str(agg3), "largest risk of the set-completion adaptive prefix under any order and configuration", 4, reg_path)
            register("AggRThreeEnvRaisedCount", int(sum(A3["EnvRaised"])), str(agg3), "configurations in which the blocker controllers' own critical values raise the envelope threshold above the ten-order envelope", 0, reg_path)
            register("AggRThreeReplayEqualsStatCount", int(sum(A3["ReplayEqualsStat"])), str(agg3), "configurations in which the replayed union over the 13 declared policies equals the statistic on every test query", 0, reg_path)
            register("AggRThreeRoundTwoEnvEqualsTenCount", int(sum(A3["RoundTwoEnvEqualsTen"])), str(agg3), "configurations in which the round-2 envelope threshold equals the ten-order envelope threshold", 0, reg_path)
            register("AggRThreeBlkChecks", int(sum(A3["BlkChecks"])), str(agg3), "re-executions of the blocker controllers checking the closed-form critical value (at the value and one ulp below)", 0, reg_path)
            register("AggRThreeBlkMismatch", int(sum(A3["BlkMismatch"])), str(agg3), "mismatches between the closed-form blocker critical value and the re-executed controller", 0, reg_path)
            register("AggRThreeCellsRscCheaper", int(sum(A3["CellsRscCheaper"])), str(agg3), "non-mu (order, configuration) cells in which RSC needs fewer calls than the set-completion prefix (paired 95% interval excludes zero)", 0, reg_path)
            register("AggRThreeCellsScCheaper", int(sum(A3["CellsScCheaper"])), str(agg3), "non-mu cells in which the set-completion prefix needs fewer calls than RSC (paired 95% interval excludes zero)", 0, reg_path)
            register("AggRThreeCellsTie", int(sum(A3["CellsTie"])), str(agg3), "non-mu cells whose paired 95% interval contains zero", 0, reg_path)
            register("AggRThreeCellsTotal", len(A3["CellsTie"]), str(agg3), "non-mu (order, configuration) cells compared", 0, reg_path)
            for pol in pr["policies"]:
                pk = macro_key(pol)
                xs = A3[f"ScMinusRsc{pk}"]
                if max(xs) < 0:
                    register(f"AggRThreeScSaves{pk}Min", -max(xs), str(agg3), f"smallest saving (calls) of the set-completion prefix over RSC under order {pol}", 1, reg_path)
                    register(f"AggRThreeScSaves{pk}Max", -min(xs), str(agg3), f"largest saving (calls) of the set-completion prefix over RSC under order {pol}", 1, reg_path)
                elif min(xs) > 0:
                    register(f"AggRThreeRscSaves{pk}Min", min(xs), str(agg3), f"smallest saving (calls) of RSC over the set-completion prefix under order {pol}", 1, reg_path)
                    register(f"AggRThreeRscSaves{pk}Max", max(xs), str(agg3), f"largest saving (calls) of RSC over the set-completion prefix under order {pol}", 1, reg_path)
                ev = [-x for x in A3[f"EnvMinusRsc{pk}"]]
                tv = [-x for x in A3[f"TrajMinusRsc{pk}"]]
                register(f"AggRThreeEnvSaves{pk}Max", max(ev), str(agg3), f"largest saving of the 13-policy envelope threshold over RSC under order {pol}", 1, reg_path)
                register(f"AggRThreeTrajSaves{pk}Max", max(tv), str(agg3), f"largest saving of the transferred mu-trajectory threshold over RSC under order {pol}", 1, reg_path)
        # repeated timed passes of the C = 32 throughput cells (runs/throughput = pass 1, runs/throughput_rep/pass*)
        rep_dirs = sorted((ROOT / cfg["paths"]["runs"] / "throughput_rep").glob("pass*")) if (ROOT / cfg["paths"]["runs"] / "throughput_rep").exists() else []
        thr_ok = []
        for t in teachers(cfg):
            for d in main_domains(cfg):
                paths = [ROOT / cfg["paths"]["runs"] / "throughput" / t / f"{d}.json"] + [rd / t / f"{d}.json" for rd in rep_dirs]
                if len(paths) < 3 or not all(p.exists() for p in paths):
                    continue
                passes = [read_json(p) for p in paths]
                if not all("full" in P_["methods"] and "32" in P_["methods"]["full"] for P_ in passes):
                    continue
                thr_ok.append((t, d, paths, passes))
        if thr_ok and len(thr_ok) == len(teachers(cfg)) * len(main_domains(cfg)):
            rep = {}
            for t, d, paths, passes in thr_ok:
                pre = macro_key(d, t)
                for m in ("rsc", "rsc_blockers", "prefix_uniform", "prefix_adaptive"):
                    if not all(m in P_["methods"] and "32" in P_["methods"][m] for P_ in passes):
                        continue
                    ratios = [P_["methods"][m]["32"]["gpu_s_per_query"] / P_["methods"]["full"]["32"]["gpu_s_per_query"] for P_ in passes]
                    p95 = [P_["methods"][m]["32"]["latency_p95_s"] / P_["methods"]["full"]["32"]["latency_p95_s"] for P_ in passes]
                    cheap = [P_["methods"][m]["32"]["cheap_s"] / P_["methods"][m]["32"]["wall_s"] for P_ in passes]
                    mean = sum(ratios) / len(ratios)
                    src = str(paths[-1])
                    register(macro_key(pre, "ThrRep", m, "Ratio"), mean, src, f"{t}/{d}: GPU-seconds per query of {m} / full at C = 32, mean of {len(ratios)} timed passes", 2, reg_path)
                    register(macro_key(pre, "ThrRep", m, "Spread"), (max(ratios) - min(ratios)) / 2, src, f"{t}/{d}: half-range over the timed passes of that ratio", 2, reg_path)
                    rep.setdefault((t, m, "ratio"), []).append(mean)
                    rep.setdefault((t, m, "spread"), []).append((max(ratios) - min(ratios)) / 2)
                    rep.setdefault((m, "spread"), []).append((max(ratios) - min(ratios)) / 2)
                    rep.setdefault((t, m, "p95"), []).append(sum(p95) / len(p95))
                    rep.setdefault((t, m, "cheap"), []).append(100 * sum(cheap) / len(cheap))
                fq = [P_["methods"]["full"]["32"]["throughput_qps"] for P_ in passes]
                register(macro_key(pre, "ThrRep", "full", "Qps"), sum(fq) / len(fq), str(paths[-1]), f"{t}/{d}: full-teacher throughput at C = 32 (queries/s), mean of {len(fq)} passes", 2, reg_path)
            agg_rep = ROOT / cfg["paths"]["runs"] / "throughput_rep" / "aggregate.json"
            write_json(agg_rep, {"sources": [str(p.relative_to(ROOT)) for _, _, ps, _ in thr_ok for p in ps], "values": {"/".join(k_): v_ for k_, v_ in rep.items()}, "n_passes": 1 + len(rep_dirs)})
            register("ThrRepNPasses", 1 + len(rep_dirs), str(agg_rep), "timed passes per C = 32 throughput cell", 0, reg_path)
            register("AggThrRepSpreadMax", max(x for k_, xs in rep.items() if k_[-1] == "spread" and len(k_) == 2 for x in xs), str(agg_rep), "largest half-range over the timed passes of any C = 32 GPU-seconds ratio", 2, reg_path)
            for k_, xs in rep.items():
                if len(k_) == 3:
                    t, m, what = k_
                    digits = 0 if what == "cheap" else 2
                    register(macro_key("AggThrRep", t, m, what, "Min"), min(xs), str(agg_rep), f"{t}: smallest {what} of {m} at C = 32 over domains (mean of passes)", digits, reg_path)
                    register(macro_key("AggThrRep", t, m, what, "Max"), max(xs), str(agg_rep), f"{t}: largest {what} of {m} at C = 32 over domains (mean of passes)", digits, reg_path)
                else:
                    m, what = k_
                    register(macro_key("AggThrRep", m, what, "Max"), max(xs), str(agg_rep), f"largest half-range over passes of the C = 32 GPU-seconds ratio of {m} over configurations", 2, reg_path)
    # D-DISTILL lever: development calls at alpha_sel of the selected head configuration, frozen vs distilled cheap model
    if not synthetic:
        from rscrank.data import cheap_model_for

        def _dev_calls(report_path):
            d = read_json(report_path)
            sel = d["selected"]
            for r in d["grid"]:
                if r["lr"] == sel["lr"] and r["wd"] == sel["wd"]:
                    return r["calls_sigma_head"] if sel["use_sigma_head"] else r["calls_sigma_one"]
            return None

        for t in teachers(cfg):
            adopted = cheap_model_for(cfg, t)
            main_rep = ROOT / cfg["paths"]["runs"] / "heads" / t / "report.json"
            other_rep = ROOT / "runs" / ("frozen_eval" if adopted == "minilm-distilled" else "distilled_eval") / "heads" / t / "report.json"
            if main_rep.exists() and other_rep.exists():
                c_main, c_other = _dev_calls(main_rep), _dev_calls(other_rep)
                c_frozen, c_dist = (c_other, c_main) if adopted == "minilm-distilled" else (c_main, c_other)
                pre = macro_key(t)
                register(f"{pre}DistillDevCallsFrozen", c_frozen, str(main_rep if adopted != "minilm-distilled" else other_rep), f"{t}: dev calls at alpha_sel with the frozen MiniLM heads", 1, reg_path)
                register(f"{pre}DistillDevCallsDistilled", c_dist, str(main_rep if adopted == "minilm-distilled" else other_rep), f"{t}: dev calls at alpha_sel with the distilled MiniLM heads", 1, reg_path)
                register(f"{pre}DistillDevGainPct", 100 * (1 - c_dist / c_frozen), str(main_rep), f"{t}: relative dev call reduction of the distilled cheap model (D-DISTILL rule: adopt iff >= {100 * cfg['cheap']['distill']['min_rel_gain']:.0f} %)", 1, reg_path)
                register(f"{pre}DistillAdopted", "distilled" if adopted == "minilm-distilled" else "frozen", str(ROOT / "configs" / "main.yaml"), f"{t}: cheap model adopted by D-DISTILL", 0, reg_path)
        register("DistillMinRelGainPct", 100 * cfg["cheap"]["distill"]["min_rel_gain"], str(ROOT / "configs" / "main.yaml"), "pre-registered minimum relative dev gain for adopting the distilled cheap model (%)", 0, reg_path)
    # 'at most' statements must round UP: ceiling variants of the aggregate maxima (source: the stats summary)
    if not synthetic:
        import math as _m

        summ_path = ROOT / cfg["paths"]["runs"] / "stats" / "summary.json"
        reg_now = read_json(reg_path)
        for key, digits in (("AggPolicyMaxRisk", 4), ("AggRiskFiniteMax", 4), ("AggRiskMarginalMax", 4), ("AggBandMax", 2), ("AggNdcgAbsDiffMax", 4), ("AggTrajTransferMaxRisk", 4), ("AggAdaptiveTransferMaxRisk", 3), ("AggOvershootMax", 1), ("AggOracleGapMax", 1)):
            if key in reg_now:
                val = float(reg_now[key]["value"])
                up = _m.ceil(val * 10**digits - 1e-9) / 10**digits
                register(f"{key}Ceil", up, reg_now[key]["source"], f"{reg_now[key]['description']} -- rounded UP for 'at most' statements", digits, reg_path)
        # H4: the smallest one-sided 97.5% lower bound of the nDCG@10 difference (RSC - full) over the main configurations,
        # rounded DOWN for 'at least' statements
        lcbs = [(k_, float(r_["value"])) for k_, r_ in reg_now.items() if k_.endswith("NdcgDeltaLcb") and not k_.startswith("Agg")]
        if lcbs:
            k_min, v_min = min(lcbs, key=lambda kv: kv[1])
            register("AggNdcgDeltaLcbMinFloor", _m.floor(v_min * 10**4 + 1e-9) / 10**4, reg_now[k_min]["source"], "smallest one-sided 97.5% bootstrap lower bound of Delta nDCG@10 over the main configurations, rounded down", 4, reg_path)
    register("AlphaMain", a0, str(ROOT / "configs" / "main.yaml"), "main risk level", 3, reg_path)
    register("DeltaFamily", pr["delta_family"], str(ROOT / "configs" / "main.yaml"), "family-wise calibration failure budget", 3, reg_path)
    register("DeltaCfg", delta_cfg, str(ROOT / "configs" / "main.yaml"), "per-configuration delta", 4, reg_path)
    register("KMain", pr["k"], str(ROOT / "configs" / "main.yaml"), "k", 0, reg_path)
    register("NonInfEps", pr["noninferiority_epsilon"], str(ROOT / "configs" / "main.yaml"), "non-inferiority margin", 3, reg_path)
    out_tex = (ROOT / "runs" / "smoke" / "paper_preview" / "numbers.tex") if synthetic else (ROOT / "paper" / "numbers.tex")
    out_json = out_tex.with_suffix(".json")
    n = render_numbers_tex(out_tex, reg_path, out_json)
    problems = verify_registry(reg_path)
    write_json(ROOT / cfg["paths"]["runs"] / "provenance" / "verify.json", {"n_numbers": n, "problems": problems})
    log(f"{n} numbers -> {out_tex}; stale/missing sources: {len(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
