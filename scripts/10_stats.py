"""Statistics over replay records (node N12): risk intervals, non-inferiority,
conditional risk, paired cost comparisons, failure-mechanism buckets, and the
pre-registered gate evaluation -> runs/stats/<teacher>/<domain>.json, runs/stats/summary.json.
Also registers every headline number in the provenance registry (paper macros).
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank.io import ensure_dir, read_json, read_jsonl, write_json  # noqa: E402
from rscrank.provenance import macro_key, register  # noqa: E402
from rscrank.stats import conditional_risk, noninferiority, paired_bootstrap_diff, risk_summary  # noqa: E402


def by_method(path):
    d = defaultdict(dict)
    for r in read_jsonl(path):
        d[r["method"]][r["qid"]] = r
    return d


def aligned(a, b, key):
    q = sorted(set(a) & set(b))
    return np.array([a[x][key] for x in q], float), np.array([b[x][key] for x in q], float), q


def main() -> int:
    ap = base_parser("statistics")
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k, a0, eps, nboot = pr["k"], pr["alpha_main"], pr["noninferiority_epsilon"], pr["bootstrap"]
    primary = pr.get("primary_regime", "finite")
    synthetic = bool(cfg["smoke"]["synthetic"])
    reg_path = ROOT / cfg["paths"]["runs"] / "provenance" / "registry.json"
    # the statistics are the first registrar of every pipeline run: start from an empty registry so that keys the
    # current code no longer produces cannot linger (12_paper_numbers.py re-registers the latency/config numbers)
    if reg_path.exists():
        reg_path.unlink()
    overall = {"configs": {}, "gates": {}, "synthetic": synthetic}
    for teacher in teachers(cfg):
        for domain in main_domains(cfg):
            rd = ROOT / cfg["paths"]["runs"] / "replay" / teacher / domain
            if not (rd / "E1_main.jsonl").exists():
                continue
            summ = read_json(rd / "summary.json")
            E1 = by_method(rd / "E1_main.jsonl")
            tag = f"a{a0}_{primary}"
            rsc, full = E1[f"rsc_{tag}"], E1["full"]
            uni, ada = E1[f"prefix_uniform_{tag}"], E1[f"prefix_adaptive_{tag}"]
            out = {"teacher": teacher, "domain": domain, "k": k, "alpha": a0, "primary_regime": primary, "n_test": len(rsc)}
            # risk with intervals per regime
            out["risk"] = {}
            for regime in pr["regimes"]:
                m = E1.get(f"rsc_a{a0}_{regime}", {})
                if m:
                    out["risk"][regime] = risk_summary([r["mismatch"] for r in m.values()])
                    out["risk"][regime]["full_scoring_fraction"] = float(np.mean([r["full_fallback"] for r in m.values()]))
            out["conditional"] = conditional_risk([r["mismatch"] for r in rsc.values()], [r["early_stopped"] for r in rsc.values()])
            out["E1_rounds_mean"] = float(np.mean([r["rounds"] for r in rsc.values()]))
            out["E1_rounds_p95"] = float(np.quantile([r["rounds"] for r in rsc.values()], 0.95))
            # cost
            c_r, c_f, _ = aligned(rsc, full, "n_calls")
            out["calls"] = {"rsc_mean": float(c_r.mean()), "full_mean": float(c_f.mean()), "reduction": float(1 - c_r.mean() / c_f.mean())}
            for name, other in (("uniform_prefix", uni), ("adaptive_prefix", ada), ("btop", E1[f"btop_{tag}"]), ("ball", E1[f"ball_{tag}"]), ("traj", E1[f"traj_{tag}"])):
                x, y, _ = aligned(other, rsc, "n_calls")
                out["calls"][name] = {"mean": float(x.mean()), "risk": float(np.mean([r["mismatch"] for r in other.values()])), "paired_diff_vs_rsc": paired_bootstrap_diff(x, y, n_boot=nboot, seed=cfg["seed"])}
            # quality
            if "ndcg@k" in next(iter(rsc.values())):
                nr, nf, _ = aligned(rsc, full, "ndcg@k")
                out["ndcg"] = {"rsc": float(nr.mean()), "full": float(nf.mean()), "noninferiority": noninferiority(nr, nf, epsilon=eps, n_boot=nboot, seed=cfg["seed"])}
                nu, _, _ = aligned(uni, full, "ndcg@k")
                na, _, _ = aligned(ada, full, "ndcg@k")
                out["ndcg"]["uniform_prefix"], out["ndcg"]["adaptive_prefix"] = float(nu.mean()), float(na.mean())
                # distribution of the nDCG change on the MISMATCHED queries (review round 1, R1): |Phi_F - Phi_RSC| <= ell_1
                # per query is trivial; what matters is how large the change is when a mismatch happens
                q_mis = sorted(q for q, r in rsc.items() if r["mismatch"] and q in full)
                if q_mis:
                    d = np.array([rsc[q]["ndcg@k"] - full[q]["ndcg@k"] for q in q_mis], float)
                    out["ndcg"]["on_mismatch"] = {"n": len(q_mis), "mean_delta": float(d.mean()), "mean_abs_delta": float(np.abs(d).mean()), "max_abs_delta": float(np.abs(d).max()),
                                                  "p95_abs_delta": float(np.quantile(np.abs(d), 0.95)), "frac_zero": float(np.mean(d == 0)), "frac_negative": float(np.mean(d < 0))}
            # alpha curve
            out["alpha_curve"] = {str(a): {reg: {"risk": summ["experiments"]["E1"][f"rsc_a{a}_{reg}"]["risk"], "calls": summ["experiments"]["E1"][f"rsc_a{a}_{reg}"]["calls_mean"]} for reg in pr["regimes"]} for a in pr["alpha_grid"]}
            out["prefix_curve"] = {m: {"risk": v["risk"], "calls": v["calls_mean"]} for m, v in summ["experiments"]["E1"].items() if m.startswith("prefix_fixed_")}
            # policies
            E1s = summ["experiments"]["E1"]
            if "cheap_only" in E1s:
                out["ndcg"]["cheap_only"] = E1s["cheap_only"].get("ndcg@k")
            if "prefix_learned" in E1s:
                out["calls"]["learned_budget"] = {"mean": E1s["prefix_learned"]["calls_mean"], "risk": E1s["prefix_learned"]["risk"], "ndcg": E1s["prefix_learned"].get("ndcg@k")}
            if "E2" in summ["experiments"]:
                E2 = summ["experiments"]["E2"]
                if "union_over_orders" in E2:
                    out["union_over_orders"] = E2["union_over_orders"]

                def _opt(prefix, p):
                    return {f"{prefix}_risk": E2[f"{prefix}_{p}"]["risk"], f"{prefix}_calls": E2[f"{prefix}_{p}"]["calls_mean"]} if f"{prefix}_{p}" in E2 else {}

                out["policies"] = {p: {"rsc_risk": E2[f"rsc_{p}"]["risk"], "rsc_calls": E2[f"rsc_{p}"]["calls_mean"], "traj_risk": E2[f"traj_{p}"]["risk"], "traj_calls": E2[f"traj_{p}"]["calls_mean"],
                                       **_opt("traj_recal", p), **_opt("prefix_adaptive", p), **_opt("prefix_adaptive_recal", p), **_opt("prefix_uniform", p), **_opt("prefix_uniform_recal", p)} for p in pr["policies"]}
                # the mu order is the prefix baselines' own order: their E1 numbers are their "transferred" numbers there
                out["policies"]["mu"].update({"prefix_adaptive_risk": out["calls"]["adaptive_prefix"]["risk"], "prefix_adaptive_calls": out["calls"]["adaptive_prefix"]["mean"], "prefix_uniform_risk": out["calls"]["uniform_prefix"]["risk"], "prefix_uniform_calls": out["calls"]["uniform_prefix"]["mean"]})
                out["policy_uniform_max_risk"] = max(v["rsc_risk"] for v in out["policies"].values())
                out["traj_transfer_max_risk"] = max(v["traj_risk"] for v in out["policies"].values())
                pa = [v["prefix_adaptive_risk"] for p, v in out["policies"].items() if "prefix_adaptive_risk" in v and p != "mu"]
                if pa:
                    out["adaptive_prefix_transfer_max_risk"] = max(pa)
                    out["adaptive_prefix_transfer_violations"] = int(sum(1 for x in pa if x > a0))
                    out["adaptive_prefix_transfer_n_orders"] = len(pa)
            for e in ("E3", "E4", "E5", "E6", "E7", "E8", "E9"):
                if e in summ["experiments"]:
                    out[e] = summ["experiments"][e]
            if "boundary_tie_fraction" in summ:
                out["boundary_tie_fraction"] = summ["boundary_tie_fraction"]
                out["boundary_tie_count"] = summ["boundary_tie_count"]
            if "E9" in out and (rd / "E9_anatomy.jsonl").exists():
                # baselines on the same population as the anatomy (covered queries), so that the anatomy figure compares like with like
                covered = {r["qid"] for r in read_jsonl(rd / "E9_anatomy.jsonl") if r.get("covered")}
                for name, recs_ in (("adaptive_prefix", ada), ("uniform_prefix", uni), ("traj", E1[f"traj_{tag}"]), ("full", E1[f"full_{tag}"] if f"full_{tag}" in E1 else None)):
                    if recs_:
                        vals = [r["n_calls"] for q, r in recs_.items() if q in covered]
                        if vals:
                            out["E9"][f"{name}_calls_covered"] = float(np.mean(vals))
            # cheap-model ablation (D-DISTILL): the variant NOT adopted for this teacher, replayed with its own calibration
            from rscrank.data import cheap_model_for
            adopted = cheap_model_for(cfg, teacher)
            other = "frozen_eval" if adopted == "minilm-distilled" else "distilled_eval"
            abl = ROOT / "runs" / other / "replay" / teacher / domain / "summary.json"  # configs/<other>.yaml: paths.runs = runs/<other>
            if abl.exists() and not synthetic:
                A = read_json(abl)["experiments"]["E1"]
                ra, aa = A[f"rsc_{tag}"], A[f"prefix_adaptive_{tag}"]
                out["cheap_variant_ablation"] = {"adopted": adopted, "other": ("minilm" if other == "frozen_eval" else "minilm-distilled"), "rsc_risk": ra["risk"], "rsc_calls": ra["calls_mean"], "adaptive_prefix_calls": aa["calls_mean"], "adaptive_prefix_risk": aa["risk"], "ndcg_rsc": ra.get("ndcg@k"), "source": str(abl)}
            # failure mechanism buckets (gap_k terciles, H_k buckets)
            stats_rows = {r["qid"]: r for r in read_jsonl(ROOT / cfg["paths"]["runs"] / "calib" / teacher / domain / f"stats_test_k{k}.jsonl")}
            gaps = np.array([stats_rows[q]["gap_k"] for q in rsc if q in stats_rows])
            qs = [q for q in rsc if q in stats_rows]
            if len(qs) >= 3:
                t1, t2 = np.quantile(gaps, [1 / 3, 2 / 3])
                buckets = {"gap_low": [q for q in qs if stats_rows[q]["gap_k"] <= t1], "gap_mid": [q for q in qs if t1 < stats_rows[q]["gap_k"] <= t2], "gap_high": [q for q in qs if stats_rows[q]["gap_k"] > t2]}
                Hs = [stats_rows[q]["H_k"] for q in qs]
                h1, h2 = np.quantile(Hs, [1 / 3, 2 / 3])
                buckets.update({"H_low": [q for q in qs if stats_rows[q]["H_k"] <= h1], "H_mid": [q for q in qs if h1 < stats_rows[q]["H_k"] <= h2], "H_high": [q for q in qs if stats_rows[q]["H_k"] > h2]})
                out["buckets"] = {b: {"n": len(v), "calls": float(np.mean([rsc[q]["n_calls"] for q in v])) if v else None, "full_fallback": float(np.mean([rsc[q]["full_fallback"] for q in v])) if v else None, "risk": float(np.mean([rsc[q]["mismatch"] for q in v])) if v else None} for b, v in buckets.items()}
            # gates (pre-registered, Sec. 18)
            # Gate on the regime whose guarantee is about THIS test split (finite population by default).
            # The marginal regime is expected to sit *at* alpha, so a raw "<= alpha" test on it would fail
            # half the time by construction: we only require that it is not *significantly* above alpha.
            gates = {
                "risk_within_alpha_primary": bool(out["risk"][primary]["rate"] <= a0),
                "risk_marginal_not_significantly_above": bool(out["risk"]["marginal"]["cp_lo"] <= a0),
                "calls_reduction_ge_target": bool(out["calls"]["reduction"] >= cfg["success_targets"]["calls_reduction_min"]),
                "efficiency_not_failed": bool(out["calls"]["rsc_mean"] <= 0.9 * out["calls"]["full_mean"]),
                "beats_adaptive_prefix_calls": bool(out["calls"]["adaptive_prefix"]["paired_diff_vs_rsc"]["lcb"] > 0),
                "beats_uniform_prefix_calls": bool(out["calls"]["uniform_prefix"]["paired_diff_vs_rsc"]["lcb"] > 0),
                "ndcg_noninferior": bool(out.get("ndcg", {}).get("noninferiority", {}).get("noninferior", False)),
                "policy_uniform_risk_ok": bool(out.get("policy_uniform_max_risk", 1.0) <= a0 + 0.02),
            }
            out["gates"] = gates
            ensure_dir(ROOT / cfg["paths"]["runs"] / "stats" / teacher)
            write_json(ROOT / cfg["paths"]["runs"] / "stats" / teacher / f"{domain}.json", out)
            overall["configs"][f"{teacher}/{domain}"] = {"gates": gates, "risk": out["risk"][primary]["rate"], "risk_marginal": out["risk"]["marginal"]["rate"], "calls_reduction": out["calls"]["reduction"], "ndcg_rsc": out.get("ndcg", {}).get("rsc"), "ndcg_full": out.get("ndcg", {}).get("full")}
            log(f"{teacher}/{domain}: risk={out['risk'][primary]['rate']:.4f} calls {out['calls']['rsc_mean']:.1f}/{out['calls']['full_mean']:.0f} (-{100*out['calls']['reduction']:.1f}%) adaptive-prefix {out['calls']['adaptive_prefix']['mean']:.1f} gates={gates}")
            # provenance (real runs only)
            if not synthetic:
                src = str(ROOT / cfg["paths"]["runs"] / "stats" / teacher / f"{domain}.json")
                pre = macro_key(domain, teacher)
                register(f"{pre}RiskRsc", out["risk"][primary]["rate"], src, f"{teacher}/{domain} exact-mismatch risk of RSC at alpha={a0} ({primary} regime)", 4, reg_path)
                register(f"{pre}RiskRscHi", out["risk"][primary]["cp_hi"], src, "Clopper-Pearson upper bound of the risk", 4, reg_path)
                register(f"{pre}RiskRscMarginal", out["risk"]["marginal"]["rate"], src, "exact-mismatch risk with the marginal (split-conformal) threshold", 4, reg_path)
                register(f"{pre}CallsRsc", out["calls"]["rsc_mean"], src, "mean teacher calls RSC", 1, reg_path)
                register(f"{pre}CallsFull", out["calls"]["full_mean"], src, "mean teacher calls full", 1, reg_path)
                register(f"{pre}CallsReductionPct", 100 * out["calls"]["reduction"], src, "percent reduction in teacher calls vs full", 1, reg_path)
                register(f"{pre}CallsUniform", out["calls"]["uniform_prefix"]["mean"], src, "mean teacher calls uniform calibrated prefix", 1, reg_path)
                register(f"{pre}CallsAdaptive", out["calls"]["adaptive_prefix"]["mean"], src, "mean teacher calls query-adaptive calibrated prefix", 1, reg_path)
                register(f"{pre}CallsBtop", out["calls"]["btop"]["mean"], src, "mean teacher calls B_top guardrail", 1, reg_path)
                register(f"{pre}CallsBall", out["calls"]["ball"]["mean"], src, "mean teacher calls B_all guardrail", 1, reg_path)
                register(f"{pre}CallsTraj", out["calls"]["traj"]["mean"], src, "mean teacher calls fixed-trajectory calibration", 1, reg_path)
                register(f"{pre}EarlyFrac", out["conditional"]["early_fraction"], src, "fraction of queries that stopped early", 3, reg_path)
                register(f"{pre}RoundsRsc", out["E1_rounds_mean"], src, "mean teacher rounds of RSC with the main 16/8 schedule", 1, reg_path)
                if "E3" in out:
                    if "union_over_schedules" in out["E3"]:
                        for key_, v in out["E3"]["union_over_schedules"].items():
                            register(f"{pre}UnionSchedRisk{macro_key(key_)}", v["union_risk"], src, f"fraction of test queries failing under at least one of the {v['n_schedules']} tested batch schedules ({key_})", 4, reg_path)
                    for m, v in out["E3"].items():
                        if isinstance(v, dict) and m.startswith("traj_blockers"):
                            register(f"{pre}{macro_key(m)}Risk", v["risk"], src, f"risk of the fixed-trajectory threshold (16/8 checkpoints) transferred to the {m} schedule", 4, reg_path)
                            register(f"{pre}{macro_key(m)}Calls", v["calls_mean"], src, f"calls of the fixed-trajectory threshold transferred to the {m} schedule", 1, reg_path)
                        if isinstance(v, dict) and m.startswith("rsc_blockers"):
                            register(f"{pre}{macro_key(m)}Calls", v["calls_mean"], src, f"mean calls of RSC with the {m} schedule (E3b)", 1, reg_path)
                            register(f"{pre}{macro_key(m)}Rounds", v["rounds_mean"], src, f"mean rounds of RSC with the {m} schedule (E3b)", 1, reg_path)
                            register(f"{pre}{macro_key(m)}Risk", v["risk"], src, f"risk of RSC with the {m} schedule (E3b)", 4, reg_path)
                register(f"{pre}ErrGivenEarly", out["conditional"]["error_given_early"], src, "error rate conditional on early stop", 4, reg_path)
                if "ndcg" in out:
                    if "on_mismatch" in out["ndcg"]:
                        om = out["ndcg"]["on_mismatch"]
                        register(f"{pre}NdcgMismatchMeanAbsDelta", om["mean_abs_delta"], src, "mean |nDCG@10 change| on the queries where RSC's list differs from the teacher's", 4, reg_path)
                        register(f"{pre}NdcgMismatchMaxAbsDelta", om["max_abs_delta"], src, "largest |nDCG@10 change| on a mismatched query", 3, reg_path)
                        register(f"{pre}NdcgMismatchFracZero", om["frac_zero"], src, "fraction of mismatched queries with no nDCG@10 change (swap among equally relevant documents)", 3, reg_path)
                        register(f"{pre}NdcgMismatchFracNegative", om["frac_negative"], src, "fraction of mismatched queries where RSC's nDCG@10 is lower than the teacher's", 3, reg_path)
                    register(f"{pre}NdcgRsc", out["ndcg"]["rsc"], src, "nDCG@10 RSC", 4, reg_path)
                    register(f"{pre}NdcgFull", out["ndcg"]["full"], src, "nDCG@10 full teacher", 4, reg_path)
                    register(f"{pre}NdcgDeltaLcb", out["ndcg"]["noninferiority"]["lcb"], src, "paired bootstrap LCB of nDCG difference", 4, reg_path)
                if "boundary_tie_fraction" in out:
                    register(f"{pre}BoundaryTieFrac", out["boundary_tie_fraction"], src, "fraction of test queries whose cached teacher scores tie exactly at the top-k boundary (s_(k) = s_(k+1))", 4, reg_path)
                    register(f"{pre}BoundaryTieCount", out["boundary_tie_count"], src, "number of test queries with an exact tie at the top-k boundary", 0, reg_path)
                if "cheap_only" in out.get("ndcg", {}) and out["ndcg"]["cheap_only"] is not None:
                    register(f"{pre}NdcgCheapOnly", out["ndcg"]["cheap_only"], src, "nDCG@k of the cheap-model ranking alone (no teacher calls)", 4, reg_path)
                if "learned_budget" in out["calls"]:
                    register(f"{pre}CallsLearned", out["calls"]["learned_budget"]["mean"], src, "mean teacher calls of the dev-tuned learned budget m(q) = ceil(c e^h(q)) (no guarantee)", 1, reg_path)
                    register(f"{pre}RiskLearned", out["calls"]["learned_budget"]["risk"], src, "risk of the dev-tuned learned budget", 4, reg_path)
                if "union_over_orders" in out:
                    for key_, v in out["union_over_orders"].items():
                        register(f"{pre}UnionRisk{macro_key(key_)}", v["union_risk"], src, f"fraction of test queries that fail under at least one of the {v['n_orders']} tested orders ({key_})", 4, reg_path)
                    register(f"{pre}UnionOrders", out["union_over_orders"]["rsc"]["n_orders"], src, "number of orders in the union-over-orders analysis", 0, reg_path)
                if "policies" in out:
                    register(f"{pre}PolicyMaxRisk", out["policy_uniform_max_risk"], src, "max risk of RSC over all scoring policies (no recalibration)", 4, reg_path)
                    register(f"{pre}TrajTransferMaxRisk", out["traj_transfer_max_risk"], src, "max risk of the fixed-trajectory threshold under other policies", 4, reg_path)
                    if "adaptive_prefix_transfer_max_risk" in out:
                        register(f"{pre}AdaptiveTransferMaxRisk", out["adaptive_prefix_transfer_max_risk"], src, "max risk of the query-adaptive calibrated prefix budget applied to another scoring order (E2b)", 3, reg_path)
                        register(f"{pre}AdaptiveTransferViolations", out["adaptive_prefix_transfer_violations"], src, "number of alternative orders under which the transferred adaptive prefix exceeds alpha (E2b)", 0, reg_path)
                        register(f"{pre}AdaptiveTransferOrders", out["adaptive_prefix_transfer_n_orders"], src, "number of alternative orders tested in E2b", 0, reg_path)
                        for p in ("alternating_bm25", "mu_binned_length", "cheap_raw", "bm25"):
                            v = out["policies"].get(p, {})
                            if "prefix_adaptive_risk" in v:
                                register(f"{pre}AdaptiveTransferRisk{macro_key(p)}", v["prefix_adaptive_risk"], src, f"risk of the transferred adaptive prefix under the {p} order (E2b)", 3, reg_path)
                                register(f"{pre}RscRisk{macro_key(p)}", v["rsc_risk"], src, f"risk of RSC under the {p} order (E2)", 4, reg_path)
                # per-experiment macros so that the text never has to type a grid value (E2 calls, E4-E8, alpha curve)
                if "policies" in out:
                    for p_, v in out["policies"].items():
                        register(f"{pre}RscCalls{macro_key(p_)}", v["rsc_calls"], src, f"mean RSC calls under the {p_} order (E2)", 1, reg_path)
                        if "prefix_adaptive_recal_risk" in v:
                            register(f"{pre}AdaptiveRecalRisk{macro_key(p_)}", v["prefix_adaptive_recal_risk"], src, f"risk of the adaptive prefix recalibrated for the {p_} order (E2b)", 3, reg_path)
                            register(f"{pre}AdaptiveRecalCalls{macro_key(p_)}", v["prefix_adaptive_recal_calls"], src, f"calls of the adaptive prefix recalibrated for the {p_} order (E2b)", 1, reg_path)
                if "alpha_curve" in out:
                    for a_, regs in out["alpha_curve"].items():
                        for reg_, v in regs.items():
                            if isinstance(v, dict) and "risk" in v:
                                register(f"{pre}Alpha{macro_key(a_)}{macro_key(reg_)}Risk", v["risk"], src, f"RSC risk at alpha={a_} ({reg_} regime)", 4, reg_path)
                                register(f"{pre}Alpha{macro_key(a_)}{macro_key(reg_)}Calls", v["calls"], src, f"RSC mean calls at alpha={a_} ({reg_} regime)", 1, reg_path)
                for e_ in ("E4", "E5", "E6", "E7", "E8"):
                    for m_, v in out.get(e_, {}).items():
                        if isinstance(v, dict) and "risk" in v:
                            register(f"{pre}{macro_key(e_)}{macro_key(m_)}Risk", v["risk"], src, f"{e_} {m_}: exact-mismatch risk", 4, reg_path)
                            register(f"{pre}{macro_key(e_)}{macro_key(m_)}Calls", v["calls_mean"], src, f"{e_} {m_}: mean teacher calls", 1, reg_path)
                            if e_ == "E5" and "risk_ell_r" in v:
                                register(f"{pre}{macro_key(e_)}{macro_key(m_)}RiskEllR", v["risk_ell_r"], src, f"{e_} {m_}: realised r-miss risk", 4, reg_path)
                if "cheap_variant_ablation" in out:
                    ab = out["cheap_variant_ablation"]
                    register(f"{pre}AblOtherCallsRsc", ab["rsc_calls"], src, f"mean RSC calls with the not-adopted cheap model ({ab['other']})", 1, reg_path)
                    register(f"{pre}AblOtherRiskRsc", ab["rsc_risk"], src, f"RSC risk with the not-adopted cheap model ({ab['other']})", 4, reg_path)
                    register(f"{pre}AblOtherCallsAdaptive", ab["adaptive_prefix_calls"], src, f"adaptive-prefix calls with the not-adopted cheap model ({ab['other']})", 1, reg_path)
                register(f"{pre}NTest", len(rsc), src, "number of test queries", 0, reg_path)
                if "E9" in out:
                    e9 = out["E9"]
                    register(f"{pre}CertCore", e9["cert_core_mean"], src, "mean |V*| over covered test queries: teacher calls any correct early stop needs at this lambda", 1, reg_path)
                    register(f"{pre}Band", e9["band_mean"], src, "mean guardrail band (extra calls of the guardrail-descending order over |V*|) over covered test queries", 2, reg_path)
                    register(f"{pre}Overshoot", e9["overshoot_mean"], src, "mean extra calls caused by batching, over covered test queries", 1, reg_path)
                    register(f"{pre}OracleGap", e9["oracle_gap_mean"], src, "mean calls above the instance-optimal certification core, over covered test queries (= band + overshoot)", 1, reg_path)
                    register(f"{pre}CoveredFrac", e9["covered_frac"], src, "fraction of test queries covered at the calibrated lambda (A_k1 <= lambda)", 3, reg_path)
                    register(f"{pre}CallsRscCovered", e9["calls_mean"], src, "mean RSC calls over covered test queries", 1, reg_path)
                    if e9.get("price_of_uniformity_mean") is not None:
                        register(f"{pre}PriceUniform", e9["price_of_uniformity_mean"], src, "mean A_k1 - A^pi (price of policy-uniformity, threshold units)", 3, reg_path)
    # global gate summary (Sec. 18.2)
    if overall["configs"]:
        g = overall["configs"]
        overall["gates"] = {
            "all_risk_within_alpha": all(v["gates"]["risk_within_alpha_primary"] for v in g.values()),
            "no_marginal_risk_violation": all(v["gates"]["risk_marginal_not_significantly_above"] for v in g.values()),
            "all_calls_reduction_ge_target": all(v["gates"]["calls_reduction_ge_target"] for v in g.values()),
            "any_efficiency_failure": any(not v["gates"]["efficiency_not_failed"] for v in g.values()),
            "beats_adaptive_prefix_everywhere": all(v["gates"]["beats_adaptive_prefix_calls"] for v in g.values()),
            "ndcg_noninferior_everywhere": all(v["gates"]["ndcg_noninferior"] for v in g.values()),
            "policy_uniform_everywhere": all(v["gates"]["policy_uniform_risk_ok"] for v in g.values()),
        }
        overall["primary_regime"] = primary
        overall["gates"]["beats_uniform_prefix_everywhere"] = all(v["gates"]["beats_uniform_prefix_calls"] for v in g.values())
        G = overall["gates"]
        valid_everywhere = G["all_risk_within_alpha"] and G["no_marginal_risk_violation"] and G["policy_uniform_everywhere"]  # D-CLAIMS: every level asserts validity under every scoring policy
        overall["claim_level"] = ("full" if valid_everywhere and G["beats_adaptive_prefix_everywhere"] and G["beats_uniform_prefix_everywhere"] and G["all_calls_reduction_ge_target"] and G["ndcg_noninferior_everywhere"] and not G["any_efficiency_failure"]
                                  else "policy-uniform-protection-with-cost" if valid_everywhere and G["ndcg_noninferior_everywhere"] and not G["any_efficiency_failure"]
                                  else "theory-plus-negative-efficiency" if valid_everywhere
                                  else "validity-failure")
        # aggregate numbers over the reported configurations (for the abstract / intro; provenance = summary.json)
        if not synthetic:
            src = str(ROOT / cfg["paths"]["runs"] / "stats" / "summary.json")
            per = {}
            for key in g:
                t, d = key.split("/")
                per[key] = read_json(ROOT / cfg["paths"]["runs"] / "stats" / t / f"{d}.json")
            agg = {
                "NConfigs": (len(per), "number of reported teacher x domain configurations", 0),
                "AggRiskFiniteMax": (max(v["risk"][primary]["rate"] for v in per.values()), "largest RSC risk (primary regime) over configurations", 4),
                "AggRiskFiniteMin": (min(v["risk"][primary]["rate"] for v in per.values()), "smallest RSC risk (primary regime) over configurations", 4),
                "AggRiskMarginalMax": (max(v["risk"]["marginal"]["rate"] for v in per.values()), "largest RSC risk (marginal regime) over configurations", 4),
                "AggCallsReductionPctMin": (100 * min(v["calls"]["reduction"] for v in per.values()), "smallest call reduction (%) over configurations", 1),
                "AggCallsReductionPctMax": (100 * max(v["calls"]["reduction"] for v in per.values()), "largest call reduction (%) over configurations", 1),
                "AggCallsReductionPctMean": (100 * float(np.mean([v["calls"]["reduction"] for v in per.values()])), "mean call reduction (%) over configurations", 1),
                "AggPolicyMaxRisk": (max(v.get("policy_uniform_max_risk", 0.0) for v in per.values()), "largest RSC risk over all configurations and all scoring orders (no recalibration)", 4),
                "AggTrajTransferMaxRisk": (max(v.get("traj_transfer_max_risk", 0.0) for v in per.values()), "largest risk of the transferred fixed-trajectory threshold over configurations and orders", 4),
                "AggAdaptiveGapUcbMin": (min(-v["calls"]["adaptive_prefix"]["paired_diff_vs_rsc"]["ucb"] for v in per.values()), "smallest lower confidence bound (nearest zero) on how many fewer calls the adaptive prefix needs than RSC", 1),
                "AggTrajGapUcbMin": (min(-v["calls"]["traj"]["paired_diff_vs_rsc"]["ucb"] for v in per.values()), "smallest lower confidence bound (nearest zero) on how many fewer calls the fixed-trajectory threshold needs than RSC", 1),
                "AggAdaptiveGapCallsMin": (min(v["calls"]["rsc_mean"] - v["calls"]["adaptive_prefix"]["mean"] for v in per.values()), "smallest (RSC - adaptive prefix) call difference over configurations", 1),
                "AggAdaptiveGapCallsMax": (max(v["calls"]["rsc_mean"] - v["calls"]["adaptive_prefix"]["mean"] for v in per.values()), "largest (RSC - adaptive prefix) call difference over configurations", 1),
                "AggTrajGapCallsMin": (min(v["calls"]["rsc_mean"] - v["calls"]["traj"]["mean"] for v in per.values()), "smallest (RSC - fixed-trajectory) call difference", 1),
                "AggTrajGapCallsMax": (max(v["calls"]["rsc_mean"] - v["calls"]["traj"]["mean"] for v in per.values()), "largest (RSC - fixed-trajectory) call difference", 1),
                "AggNdcgAbsDiffMax": (max(abs(v["ndcg"]["rsc"] - v["ndcg"]["full"]) for v in per.values() if "ndcg" in v), "largest |nDCG@10 RSC - full| over configurations", 4),
                "AggPoUMin": (min(v["E9"]["price_of_uniformity_mean"] for v in per.values() if "E9" in v), "smallest mean price of policy-uniformity (threshold units)", 3),
                "AggPoUMax": (max(v["E9"]["price_of_uniformity_mean"] for v in per.values() if "E9" in v), "largest mean price of policy-uniformity (threshold units)", 3),
                "AggOracleGapMin": (min(v["E9"]["oracle_gap_mean"] for v in per.values() if "E9" in v), "smallest mean oracle gap (calls above |V*|, covered queries)", 1),
                "AggOracleGapMax": (max(v["E9"]["oracle_gap_mean"] for v in per.values() if "E9" in v), "largest mean oracle gap (calls above |V*|, covered queries)", 1),
                "AggOvershootMin": (min(v["E9"]["overshoot_mean"] for v in per.values() if "E9" in v), "smallest mean batching overshoot (covered queries)", 1),
                "AggOvershootMax": (max(v["E9"]["overshoot_mean"] for v in per.values() if "E9" in v), "largest mean batching overshoot (covered queries)", 1),
                "AggBandMax": (max(v["E9"]["band_mean"] for v in per.values() if "E9" in v), "largest mean guardrail band (covered queries)", 2),
                "AggCoveredFracMin": (min(v["E9"]["covered_frac"] for v in per.values() if "E9" in v), "smallest covered fraction", 3),
                "AggCoveredFracMax": (max(v["E9"]["covered_frac"] for v in per.values() if "E9" in v), "largest covered fraction", 3),
                "AggAdaptivePrefixRiskMin": (min(v["calls"]["adaptive_prefix"]["risk"] for v in per.values()), "smallest realised risk of the adaptive prefix", 4),
                "AggAdaptivePrefixRiskMax": (max(v["calls"]["adaptive_prefix"]["risk"] for v in per.values()), "largest realised risk of the adaptive prefix", 4),
            }
            un = [v["union_over_orders"] for v in per.values() if "union_over_orders" in v]
            if un:
                for key_ in ("rsc", "traj", "traj_recal", "prefix_adaptive_recal", "prefix_adaptive"):
                    xs = [u[key_]["union_risk"] for u in un if key_ in u]
                    if xs:
                        agg[f"AggUnionRisk{macro_key(key_)}Min"] = (min(xs), f"smallest union-over-orders risk of {key_} over configurations", 4)
                        agg[f"AggUnionRisk{macro_key(key_)}Max"] = (max(xs), f"largest union-over-orders risk of {key_} over configurations", 4)
            om_all = [v["ndcg"]["on_mismatch"] for v in per.values() if "on_mismatch" in v.get("ndcg", {})]
            if om_all:
                agg["AggNdcgMismatchMaxAbsDelta"] = (max(o["max_abs_delta"] for o in om_all), "largest |nDCG@10 change| on any mismatched query over configurations", 3)
                agg["AggNdcgMismatchMeanAbsDeltaMax"] = (max(o["mean_abs_delta"] for o in om_all), "largest configuration mean of |nDCG@10 change| on mismatched queries", 4)
                agg["AggNdcgMismatchFracZeroMin"] = (min(o["frac_zero"] for o in om_all), "smallest fraction of mismatched queries with unchanged nDCG@10", 3)
                agg["AggNdcgMismatchFracZeroMax"] = (max(o["frac_zero"] for o in om_all), "largest fraction of mismatched queries with unchanged nDCG@10", 3)
            if any("cheap_only" in v.get("ndcg", {}) for v in per.values()):
                agg["AggNdcgCheapOnlyMin"] = (min(v["ndcg"]["cheap_only"] for v in per.values() if v.get("ndcg", {}).get("cheap_only") is not None), "smallest cheap-only nDCG@k", 3)
                agg["AggNdcgCheapOnlyMax"] = (max(v["ndcg"]["cheap_only"] for v in per.values() if v.get("ndcg", {}).get("cheap_only") is not None), "largest cheap-only nDCG@k", 3)
            tr = [v for v in per.values() if "adaptive_prefix_transfer_max_risk" in v]
            if tr:
                agg["AggAdaptiveTransferMaxRisk"] = (max(v["adaptive_prefix_transfer_max_risk"] for v in tr), "largest risk of the transferred adaptive prefix budget over configurations and alternative orders (E2b)", 3)
                agg["AggAdaptiveTransferViolationsTotal"] = (int(sum(v["adaptive_prefix_transfer_violations"] for v in tr)), "total (configuration, order) cells in which the transferred adaptive prefix exceeds alpha (E2b)", 0)
                agg["AggAdaptiveTransferCellsTotal"] = (int(sum(v["adaptive_prefix_transfer_n_orders"] for v in tr)), "total (configuration, order) cells tested in E2b", 0)
                agg["AggAdaptiveTransferViolationsMinPerCfg"] = (min(v["adaptive_prefix_transfer_violations"] for v in tr), "fewest violating orders in any configuration (E2b)", 0)
                agg["AggAdaptiveTransferViolationsMaxPerCfg"] = (max(v["adaptive_prefix_transfer_violations"] for v in tr), "most violating orders in any configuration (E2b)", 0)
            for key, (val, desc, dig) in agg.items():
                register(key, val, src, desc, dig, reg_path)
            register("ClaimLevel", overall["claim_level"], src, "claim level chosen by the pre-registered gates", 0, reg_path)
    write_json(ROOT / cfg["paths"]["runs"] / "stats" / "summary.json", overall)
    log(f"claim level: {overall.get('claim_level')} ; gates: {overall.get('gates')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
