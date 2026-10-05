"""Policy envelope analysis (review round 1, R1/R3): how much of the policy-uniform premium protects states that the
tested orders actually reach?  -> runs/policy_envelope/<teacher>/<domain>.json

For every calibration and test query and every scoring order pi in protocol.policies (and the blocker schedules of the
mu order) we compute the trajectory-specific critical value A^pi_{k,1}(q) (the smallest lambda at which no unsafe
checkpoint of that order passes the stop rule; -inf if the order never visits an unsafe checkpoint with |V| >= k) next
to the rank-sufficient A_{k,1}(q) (the supremum over ALL subsets).  Reported per configuration:
  * per order: fraction of test queries with A^pi = -inf, mean of A_{k,1} - A^pi over finite values (order-specific
    price of uniformity), fraction of queries where the order attains the worst case (A^pi = A_{k,1});
  * the envelope E(q) = max over the tested orders of A^pi(q): its calibrated threshold lambda_env (same regime, same
    delta) next to lambda_RSC (quantile of A_{k,1}) and lambda_mu (quantile of A^mu), and the mean calls of the
    controller under the mu order at each of the three thresholds -- the share of the premium 'spent' on states the
    tested orders reach vs on untested states;
  * the per-order thresholds lambda_pi and the fraction of test queries with A^pi(q) > lambda_mu (the transferred
    fixed-trajectory failure rate, recomputed from statistics rather than from replays).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank.calibrate import calibrate_threshold, family_delta  # noqa: E402
from rscrank.io import ensure_dir, read_json, write_json  # noqa: E402
from rscrank.policies import make_policy  # noqa: E402
from rscrank.replay import MethodSpec, query_statistics, replay_dataset  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("rep08", Path(__file__).resolve().parent / "08_replay_eval.py")
rep08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep08)  # type: ignore[union-attr]


def traj_stat(b, k, policy, sched):
    oracle = core.TeacherOracle(b.ids, b.s)
    res = core.run_controller(b.ids, b.mu, b.sigma, math.inf, k, oracle, policy, sched)
    prefixes = core.prefixes_from_trajectory(res.trajectory[:-1], b.ids)
    return core.trajectory_critical_value(b.s, b.mu, b.sigma, k, prefixes, 1, b.ids)


def blocker_sched(bf, br):
    base = core.blocker_batch_schedule(min_batch=br)

    def sched(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_):
        return bf if round_idx == 0 else base(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_)

    sched.adaptive = True  # type: ignore[attr-defined]
    return sched


def main() -> int:
    ap = base_parser("policy envelope analysis")
    ap.add_argument("--teacher", default=None)
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k0, a0 = pr["k"], pr["alpha_main"]
    primary = pr.get("primary_regime", "finite")
    delta_cfg = family_delta(pr["delta_family"], pr["n_main_configs"])
    bf, br = pr["batch_first"], pr["batch_rest"]
    orders = list(pr["policies"])
    for teacher in ([args.teacher] if args.teacher else teachers(cfg)):
        for domain in main_domains(cfg):
            lams = read_json(ROOT / cfg["paths"]["runs"] / "calib" / teacher / domain / "lambdas.json")
            cal = rep08.load_bundles(cfg, teacher, domain, "cal", k0, lams=lams)
            test = rep08.load_bundles(cfg, teacher, domain, "test", k0, lams=lams)
            out = {"teacher": teacher, "domain": domain, "k": k0, "alpha": a0, "regime": primary, "orders": {}, "n_cal": len(cal), "n_test": len(test)}
            variants = [(p, "fixed") for p in orders] + [("mu", "blockers32_16"), ("mu", "blockers16_16")]
            stats = {"cal": {}, "test": {}}
            for split, bundles in (("cal", cal), ("test", test)):
                A = np.array([query_statistics(b, k0)["A_k1"] for b in bundles])
                stats[split]["A"] = A
                for pol, sch in variants:
                    name = pol if sch == "fixed" else f"{pol}_{sch}"
                    vals = []
                    for b in bundles:
                        feats = {n: b.feature_map(n) for n in b.features}
                        if pol == "mu_alt" and "mu_alt" not in feats:
                            vals.append(float("nan"))
                            continue
                        policy = make_policy(pol, feats, cfg["seed"])
                        if sch == "fixed":
                            sched = core.default_batch_schedule(k0, bf, br)
                        else:
                            f_, r_ = (32, 16) if sch == "blockers32_16" else (16, 16)
                            sched = blocker_sched(f_, r_)
                        vals.append(traj_stat(b, k0, policy, sched))
                    stats[split][name] = np.array(vals, dtype=float)
                log(f"{teacher}/{domain}/{split}: trajectory statistics for {len(variants)} orders on {len(bundles)} queries")
            A_cal, A_test = stats["cal"]["A"], stats["test"]["A"]
            lam_rsc = calibrate_threshold(list(A_cal), a0, primary, delta=delta_cfg, M=len(test)).threshold
            names = [n for n in stats["cal"] if n != "A"]
            finite_names = [n for n in names if not np.isnan(stats["cal"][n]).any()]
            env_cal = np.max(np.vstack([stats["cal"][n] for n in finite_names]), axis=0)
            env_test = np.max(np.vstack([stats["test"][n] for n in finite_names]), axis=0)
            lam_env = calibrate_threshold(list(env_cal), a0, primary, delta=delta_cfg, M=len(test)).threshold
            lam_mu = calibrate_threshold(list(stats["cal"]["mu"]), a0, primary, delta=delta_cfg, M=len(test)).threshold
            for n in finite_names:
                tc, tt = stats["cal"][n], stats["test"][n]
                lam_pi = calibrate_threshold(list(tc), a0, primary, delta=delta_cfg, M=len(test)).threshold
                fin = np.isfinite(tt)
                out["orders"][n] = {
                    "neg_inf_frac_test": float(np.mean(~fin)),
                    "pou_mean_finite": float(np.mean(A_test[fin] - tt[fin])) if fin.any() else None,
                    "attains_worst_case_frac": float(np.mean(np.isclose(tt, A_test, rtol=0, atol=1e-12))),
                    "lambda_pi": lam_pi,
                    "test_frac_above_lambda_mu": float(np.mean(tt > lam_mu)),
                    "test_frac_above_lambda_rsc": float(np.mean(tt > lam_rsc)),
                    "test_frac_above_own_lambda": float(np.mean(tt > lam_pi)),
                }
            out["thresholds"] = {"lambda_rsc": lam_rsc, "lambda_env": lam_env, "lambda_mu": lam_mu,
                                 "env_orders": finite_names,
                                 "env_test_frac_above_lambda_env": float(np.mean(env_test > lam_env)),
                                 "env_test_frac_above_lambda_mu": float(np.mean(env_test > lam_mu)),
                                 "A_test_frac_above_lambda_rsc": float(np.mean(A_test > lam_rsc)),
                                 "A_test_frac_above_lambda_env": float(np.mean(A_test > lam_env)),
                                 "A_test_frac_above_lambda_mu": float(np.mean(A_test > lam_mu)),
                                 "env_equals_A_frac_test": float(np.mean(np.isclose(env_test, A_test, rtol=0, atol=1e-12)))}
            # (round-2 R3 W1) the fair per-order competitor: K per-order trajectory thresholds each calibrated at alpha/K (Bonferroni),
            # which also bounds the union over the K tested orders -- its calls under the mu order vs RSC's single threshold at alpha
            K = len(finite_names)
            lam_bonf = {n: calibrate_threshold(list(stats["cal"][n]), a0 / K, primary, delta=delta_cfg, M=len(test)).threshold for n in finite_names}
            union_bonf = float(np.mean(np.any(np.vstack([stats["test"][n] > lam_bonf[n] for n in finite_names]), axis=0)))
            out["bonferroni"] = {"K": K, "alpha_per_order": a0 / K, "lambda_mu_bonf": lam_bonf["mu"], "union_test_frac_fail": union_bonf,
                                 "per_order_lambda": lam_bonf}
            # (round-2 R3 W2) envelope over the REALISTIC orders only (those a system would deploy: mu-based and adaptive orders, not BM25/length/random)
            realistic = [n for n in finite_names if n in ("mu", "cheap_raw", "observed_regression", "alternating_bm25", "mu_binned_length", "mu_alt", "mu_blockers16_16", "mu_blockers32_16")]
            envR_cal = np.max(np.vstack([stats["cal"][n] for n in realistic]), axis=0)
            envR_test = np.max(np.vstack([stats["test"][n] for n in realistic]), axis=0)
            lam_envR = calibrate_threshold(list(envR_cal), a0, primary, delta=delta_cfg, M=len(test)).threshold
            out["realistic_envelope"] = {"orders": realistic, "lambda_env_realistic": lam_envR, "env_test_frac_above_lambda_mu": float(np.mean(envR_test > lam_mu)),
                                         "env_equals_A_frac_test": float(np.mean(np.isclose(envR_test, A_test, rtol=0, atol=1e-12)))}
            # calls of the controller under the mu order at the thresholds (same 16/8 schedule)
            calls = {}
            for nm, lam in (("rsc", lam_rsc), ("env", lam_env), ("mu", lam_mu), ("mu_bonf", lam_bonf["mu"]), ("env_realistic", lam_envR)):
                recs = replay_dataset(test, MethodSpec(f"c_{nm}", "guardrail", lam=lam, policy="mu", batch_first=bf, batch_rest=br), k0)
                calls[nm] = {"calls_mean": float(np.mean([r["n_calls"] for r in recs])), "risk": float(np.mean([r["mismatch"] for r in recs]))}
            out["calls_under_mu_order"] = calls
            prem = calls["rsc"]["calls_mean"] - calls["mu"]["calls_mean"]
            out["premium_split"] = {"total_premium_calls": prem, "spent_on_tested_orders_calls": calls["env"]["calls_mean"] - calls["mu"]["calls_mean"],
                                    "spent_on_untested_states_calls": calls["rsc"]["calls_mean"] - calls["env"]["calls_mean"],
                                    "share_tested": ((calls["env"]["calls_mean"] - calls["mu"]["calls_mean"]) / prem) if prem > 0 else None,
                                    "spent_on_realistic_orders_calls": calls["env_realistic"]["calls_mean"] - calls["mu"]["calls_mean"],
                                    "share_realistic": ((calls["env_realistic"]["calls_mean"] - calls["mu"]["calls_mean"]) / prem) if prem > 0 else None}
            out["bonferroni"]["calls_mu_order"] = calls["mu_bonf"]["calls_mean"]
            out["bonferroni"]["rsc_minus_bonf_calls"] = calls["rsc"]["calls_mean"] - calls["mu_bonf"]["calls_mean"]
            # (round-2 R1) two realistic QUERY-DEPENDENT selection rules between orders, evaluated from the E2 per-query records:
            # the risk of the per-order recalibrated trajectory thresholds under the rule (between the mixture and the union) vs RSC
            e2 = ROOT / cfg["paths"]["runs"] / "replay" / teacher / domain / "E2_policies.jsonl"
            if e2.exists():
                recs = {}
                for line in open(e2, encoding="utf-8"):
                    r = __import__("json").loads(line)
                    recs.setdefault(r["method"], {})[r["qid"]] = r["mismatch"]
                feat = {b.qid: (float(np.median(b.features["length"])), float(np.max(b.features["bm25"]))) for b in test}
                med_len = float(np.median([v[0] for v in feat.values()]))
                med_bm = float(np.median([v[1] for v in feat.values()]))
                rules = {"long_lists_to_shortest_first_window": lambda q: "mu_binned_length" if feat[q][0] > med_len else "mu",
                         "strong_bm25_to_alternating": lambda q: "alternating_bm25" if feat[q][1] > med_bm else "mu"}
                out["selection_rules"] = {}
                for name, sel in rules.items():
                    v_rsc, v_tr = [], []
                    for q in feat:
                        p_ = sel(q)
                        k_rsc = f"rsc_{p_}"
                        k_tr = f"traj_{p_}" if p_ == "mu" else f"traj_recal_{p_}"
                        if q in recs.get(k_rsc, {}) and q in recs.get(k_tr, {}):
                            v_rsc.append(recs[k_rsc][q])
                            v_tr.append(recs[k_tr][q])
                    if v_rsc:
                        out["selection_rules"][name] = {"n": len(v_rsc), "risk_rsc": float(np.mean(v_rsc)), "risk_per_order_recalibrated_traj": float(np.mean(v_tr))}
            od = ensure_dir(ROOT / cfg["paths"]["runs"] / "policy_envelope" / teacher)
            write_json(od / f"{domain}.json", out)
            log(f"{teacher}/{domain}: lambda_mu {lam_mu:.3f} < lambda_env {lam_env:.3f} <= lambda_rsc {lam_rsc:.3f}; calls mu {calls['mu']['calls_mean']:.1f} env {calls['env']['calls_mean']:.1f} rsc {calls['rsc']['calls_mean']:.1f}; premium split tested/untested = {out['premium_split']['spent_on_tested_orders_calls']:.2f}/{out['premium_split']['spent_on_untested_states_calls']:.2f}; realistic-env calls {calls['env_realistic']['calls_mean']:.1f}; Bonferroni(K={K}) mu-threshold calls {calls['mu_bonf']['calls_mean']:.1f} union fail {union_bonf:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
