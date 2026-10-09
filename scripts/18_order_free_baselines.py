"""Review round 3 analyses (amendment A7) -> runs/review3/<teacher>/<domain>.json

(1) Set-completion calibrated prefixes (W1/Q1).  The calibrated prefix set S(q) -- the first m (uniform) or m(q)
    (query-adaptive) documents of the mu order -- is fixed from cheap information before any teacher call.  The
    scheduler then scores in ANY order and batch schedule, and the controller stops only once S(q) is contained in
    the scored set V, returning the ordered top-k of V.  Its error event {T_k(q) not in S(q)} is a function of the
    query alone, so its risk bound holds under every order; its cost under an order is the first checkpoint at which
    S(q) is complete (one-shot execution, all of S(q) in one round, costs |S(q)| under every order).  Replayed under
    the ten E2 orders with the 16/8 schedule next to RSC, the transferred trajectory threshold and the envelope
    threshold (same orders, same schedule, paired differences).
(2) Envelopes over a DECLARED policy family (W2/W3/Q2).  The trajectory statistic A^pi is valid for trajectories
    that do not depend on lambda (the ten orders under the fixed 16/8 schedule); the blocker schedules choose their
    second round from the guardrails, so they enter through the critical value of the controller re-executed at each
    lambda (core.blocker_critical_value), never through a trajectory frozen at a reference lambda.  Envelopes: the
    ten orders; the ten orders and the three blocker controllers; the realistic subsets.  The replayed union of
    failures over the declared family at the envelope threshold is checked against the statistic, query by query.
(3) Per-policy (Bonferroni) competitor at (alpha/K, delta/K): K trajectory thresholds jointly valid with
    probability 1 - delta (union bound over the K finite-population statements), hence a union risk <= alpha.
(4) Same-sample cost comparison (W5): on the covered test queries, the adaptive prefix budget m(q) against the
    certification core |V*| of the same queries (paired bootstrap).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank.baselines import adaptive_prefix_budget  # noqa: E402
from rscrank.calibrate import calibrate_threshold, family_delta  # noqa: E402
from rscrank.io import ensure_dir, read_json, write_json  # noqa: E402
from rscrank.policies import make_policy  # noqa: E402
from rscrank.replay import MethodSpec, query_statistics, replay_query  # noqa: E402
from rscrank.stats import paired_bootstrap_diff  # noqa: E402

_spec = importlib.util.spec_from_file_location("rep08", Path(__file__).resolve().parent / "08_replay_eval.py")
rep08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep08)  # type: ignore[union-attr]

BLOCKERS = {"blockers16_8": (16, 8), "blockers16_16": (16, 16), "blockers32_16": (32, 16)}
REALISTIC_ORDERS = ("mu", "cheap_raw", "observed_regression", "alternating_bm25", "mu_binned_length", "mu_alt")


def traj_stat(b, k, policy, bf, br):
    """A^pi_{k,1} of a lambda-independent trajectory (fixed schedule): generated once at lambda = +inf."""
    oracle = core.TeacherOracle(b.ids, b.s)
    res = core.run_controller(b.ids, b.mu, b.sigma, math.inf, k, oracle, policy, core.default_batch_schedule(k, bf, br))
    prefixes = core.prefixes_from_trajectory(res.trajectory[:-1], b.ids)
    return core.trajectory_critical_value(b.s, b.mu, b.sigma, k, prefixes, 1, b.ids)


def set_completion(b, S_ids, k, policy, bf, br):
    """Score in the policy's order with the bf/br schedule; stop once S is scored; ordered top-k of V."""
    oracle = core.TeacherOracle(b.ids, b.s)
    mu_by_id = {i: float(m) for i, m in zip(b.ids, b.mu)}
    observed, remaining, rounds, need = {}, list(b.ids), 0, set(S_ids)
    sched = core.default_batch_schedule(k, bf, br)
    while remaining:
        order = list(policy(remaining, observed, mu_by_id))
        batch = order[: max(1, int(sched(rounds)))]
        for i, sc in zip(batch, oracle.score(batch)):
            observed[i] = sc
        rounds += 1
        remaining = [i for i in remaining if i not in observed]
        if need.issubset(observed.keys()):
            break
    ids_ = list(observed)
    sc = np.array([observed[i] for i in ids_], float)
    pred = [ids_[p] for p in core.teacher_order(sc, ids_)[:k]]
    return pred, oracle.n_calls, rounds


def mu_prefix(b, m):
    return [b.ids[p] for p in core.teacher_order(b.mu, b.ids)[: max(1, min(m, b.N))]]


def calib(vals, alpha, regime, delta, M):
    return calibrate_threshold(list(vals), alpha, regime, delta=delta, M=M).threshold


def boot(a, b, seed):
    d = paired_bootstrap_diff(a, b, n_boot=2000, seed=seed)
    return {"mean": d["mean"], "lcb": d["lcb"], "ucb": d["ucb"]}


FROZEN_SPLIT_SEED = 20261006  # declared once, before any metric on this split was computed; not chosen among draws


def frozen_family_resplit(cal, test, domain: str, split_dir: Path):
    """One uniform redraw of the fixed pool (the current calibration set union the test set).

    The policy family is already frozen. The draw is written to disk before any
    threshold or risk is computed, and a later run reuses that file rather than
    drawing again.
    """
    path = split_dir / f"{domain}.json"
    pool = sorted(list(cal) + list(test), key=lambda b: str(b.qid))
    by_qid = {str(b.qid): b for b in pool}
    if len(by_qid) != len(pool):
        raise SystemExit(f"{domain}: duplicate qids in the fixed pool")
    n_cal = len(cal)
    if path.exists():
        rec = read_json(path)
        if rec.get("seed") != FROZEN_SPLIT_SEED or rec.get("n_cal") != n_cal:
            raise SystemExit(f"{path} was drawn under a different seed or size; refusing to redraw")
        cal_q, test_q = rec["cal"], rec["test"]
    else:
        domain_code = {"nq": 1, "hotpotqa": 2, "msmarco": 3}[domain]
        rng = np.random.default_rng(np.random.SeedSequence([FROZEN_SPLIT_SEED, domain_code]))
        perm = rng.permutation(len(pool))
        cal_q = [str(pool[i].qid) for i in sorted(perm[:n_cal].tolist())]
        test_q = [str(pool[i].qid) for i in sorted(perm[n_cal:].tolist())]
        write_json(path, {"seed": FROZEN_SPLIT_SEED, "domain": domain, "n_cal": n_cal, "n_test": len(test_q),
                          "population": "previous calibration union test; train and development sets untouched",
                          "drawn_once": True, "cal": cal_q, "test": test_q})
    if set(cal_q) | set(test_q) != set(by_qid) or set(cal_q) & set(test_q):
        raise SystemExit(f"{domain}: saved split is not a partition of the fixed pool")
    return [by_qid[q] for q in cal_q], [by_qid[q] for q in test_q]


def main() -> int:
    ap = base_parser("review round 3: set-completion prefixes, declared-family envelopes, Bonferroni at delta/K")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--limit", type=int, default=0, help="dry run on the first LIMIT calibration and test queries (output under runs/review3_dry)")
    ap.add_argument("--frozen-family-split", action="store_true",
                    help="after the policy family is frozen, redraw the fixed pool once (seed 20261006) and recompute")
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k0, a0 = pr["k"], pr["alpha_main"]
    primary = pr.get("primary_regime", "finite")
    delta_cfg = family_delta(pr["delta_family"], pr["n_main_configs"])
    bf, br = pr["batch_first"], pr["batch_rest"]
    orders = list(pr["policies"])
    seed = int(cfg["seed"])
    for teacher in ([args.teacher] if args.teacher else teachers(cfg)):
        for domain in ([args.domain] if args.domain else main_domains(cfg)):
            lams = read_json(ROOT / cfg["paths"]["runs"] / "calib" / teacher / domain / "lambdas.json")
            cal = rep08.load_bundles(cfg, teacher, domain, "cal", k0, lams=lams)
            test = rep08.load_bundles(cfg, teacher, domain, "test", k0, lams=lams)
            if args.frozen_family_split:
                if args.limit:
                    raise SystemExit("the frozen-family split is one draw of the full pool; --limit is not allowed")
                cal, test = frozen_family_resplit(cal, test, domain, ensure_dir(ROOT / cfg["paths"]["runs"] / "a8_split" / "splits"))
                log(f"{teacher}/{domain}: frozen-family split seed {FROZEN_SPLIT_SEED}, n_cal {len(cal)}, n_test {len(test)}")
            elif args.limit:
                cal, test = cal[: args.limit], test[: args.limit]
            M = len(test)
            lam_rsc = rep08.thr(lams, k0, a0, primary, "A_k1")
            lam_mu = rep08.thr(lams, k0, a0, primary, "traj_mu")
            m_uni = int(min(rep08.thr(lams, k0, a0, primary, "H_k"), test[0].N))
            t_H = rep08.thr(lams, k0, a0, primary, "e_H")
            out = {"teacher": teacher, "domain": domain, "k": k0, "alpha": a0, "regime": primary, "delta_cfg": delta_cfg,
                   "n_cal": len(cal), "n_test": M, "schedule": [bf, br],
                   "thresholds": {"lambda_rsc": lam_rsc, "lambda_mu": lam_mu, "m_uniform": m_uni, "t_H": t_H}}

            # ---- (2) statistics: A_k1, A^pi for the ten lambda-independent orders, blocker critical values ----------------
            stats = {}
            for split, bundles in (("cal", cal), ("test", test)):
                rows_q = [query_statistics(b, k0) for b in bundles]
                st = {"A": np.array([r["A_k1"] for r in rows_q]),
                      "B_top": np.array([r["B_top"] for r in rows_q], float),
                      "B_all": np.array([r["B_all"] for r in rows_q], float),
                      "H_k": np.array([r["H_k"] for r in rows_q], float),
                      "h_pred": np.array([float(b.features["h_pred"][0]) for b in bundles], float)}
                for pol in orders:
                    vals = []
                    for b in bundles:
                        feats = {n: b.feature_map(n) for n in b.features}
                        if pol == "mu_alt" and "mu_alt" not in feats:
                            vals.append(float("nan"))
                            continue
                        vals.append(traj_stat(b, k0, make_policy(pol, feats, seed), bf, br))
                    st[pol] = np.array(vals, float)
                for name, (f_, r_) in BLOCKERS.items():
                    st[name] = np.array([core.blocker_critical_value(b.s, b.mu, b.sigma, k0, b.ids, f_, r_) for b in bundles], float)
                    # the value a frozen-trajectory calibration would have used: the trajectory generated at lambda = +inf
                    frozen = []
                    for b in bundles:
                        oracle = core.TeacherOracle(b.ids, b.s)
                        sched = rep08_blocker(f_, r_)
                        res = core.run_controller(b.ids, b.mu, b.sigma, math.inf, k0, oracle, make_policy("mu", {}, seed), sched)
                        frozen.append(core.trajectory_critical_value(b.s, b.mu, b.sigma, k0, core.prefixes_from_trajectory(res.trajectory[:-1], b.ids), 1, b.ids))
                    st[name + "_frozen"] = np.array(frozen, float)
                stats[split] = st
                log(f"{teacher}/{domain}/{split}: statistics for {len(orders)} orders + {len(BLOCKERS)} blocker controllers on {len(bundles)} queries")
            assert all(not np.isnan(stats[s_][p]).any() for s_ in stats for p in orders), "mu_alt missing for some queries"
            if args.frozen_family_split:
                from rscrank.baselines import calibrate_adaptive_prefix, calibrate_uniform_prefix
                lam_rsc = calib(stats["cal"]["A"], a0, primary, delta_cfg, M)
                lam_mu = calib(stats["cal"]["mu"], a0, primary, delta_cfg, M)
                m_uni = int(min(calibrate_uniform_prefix(stats["cal"]["H_k"], a0, primary, delta=delta_cfg, M=M, N=test[0].N).threshold, test[0].N))
                t_H = calibrate_adaptive_prefix(stats["cal"]["H_k"], stats["cal"]["h_pred"], a0, primary, delta=delta_cfg, M=M).threshold
                out["thresholds"] = {"lambda_rsc": lam_rsc, "lambda_mu": lam_mu, "m_uniform": m_uni, "t_H": t_H,
                                     "calibrated_on": "frozen-family split", "split_seed": FROZEN_SPLIT_SEED}
                if lam_rsc + 1e-9 < calib(np.maximum.reduce([stats["cal"][m] for m in list(orders) + list(BLOCKERS)]), a0, primary, delta_cfg, M):
                    raise SystemExit(f"{teacher}/{domain}: envelope threshold exceeds A; the statistics are not aligned")
                log(f"{teacher}/{domain}: recalibrated on the new split, lambda RSC {lam_rsc:.4f}, trajectory {lam_mu:.4f}")

            # check the closed form against the re-executed blocker controller on every test query, at the value
            # itself (must be correct) and just below it (must err, when finite)
            bad, n_chk = 0, 0
            for name, (f_, r_) in BLOCKERS.items():
                for b, a_blk in zip(test, stats["test"][name]):
                    for lam_, want_err in ((a_blk, False), (float(np.nextafter(a_blk, -np.inf)), True)):
                        if not math.isfinite(lam_):
                            continue
                        rec = replay_query(b, MethodSpec("chk", "guardrail", lam=lam_, policy="mu", batch_first=f_, batch_rest=r_, schedule="blockers"), k0)
                        n_chk += 1
                        bad += int(bool(rec["mismatch"]) != want_err)
            out["blocker_closed_form_check"] = {"n_checks": n_chk, "n_mismatch": bad}
            log(f"{teacher}/{domain}: blocker closed form checked on {n_chk} re-executions, {bad} mismatches")
            out["blocker_vs_frozen"] = {name: {"test_frac_true_above_frozen": float(np.mean(stats["test"][name] > stats["test"][name + "_frozen"])),
                                               "test_frac_true_above_lambda_mu": float(np.mean(stats["test"][name] > lam_mu)),
                                               "test_frac_frozen_above_lambda_mu": float(np.mean(stats["test"][name + "_frozen"] > lam_mu))}
                                        for name in BLOCKERS}

            # ---- envelopes over declared families ------------------------------------------------------------------------
            fams = {"orders10": list(orders), "all13": list(orders) + list(BLOCKERS),
                    "realistic6": list(REALISTIC_ORDERS), "realistic9": list(REALISTIC_ORDERS) + list(BLOCKERS)}
            old_env = None
            old_path = ROOT / cfg["paths"]["runs"] / "policy_envelope" / teacher / f"{domain}.json"
            # A threshold from the previous split is not a quantity on this split.
            if old_path.exists() and not args.frozen_family_split:
                old_env = read_json(old_path)["thresholds"]["lambda_env"]
            out["thresholds"]["lambda_env_round2"] = old_env
            env = {}
            for fname, members in fams.items():
                e_cal = np.max(np.vstack([stats["cal"][m] for m in members]), axis=0)
                e_test = np.max(np.vstack([stats["test"][m] for m in members]), axis=0)
                lam_e = calib(e_cal, a0, primary, delta_cfg, M)
                env[fname] = {"members": members, "lambda": lam_e, "test_union_stat": float(np.mean(e_test > lam_e)),
                              "test_frac_env_equals_A": float(np.mean(np.isclose(e_test, stats["test"]["A"], rtol=0, atol=1e-12))),
                              "test_frac_env_above_lambda_mu": float(np.mean(e_test > lam_mu)), "_e_test": e_test}
            # the old (round-2) envelope threshold re-used for the re-executed blocker controllers
            if old_env is not None:
                out["round2_envelope_check"] = {"lambda_env_round2": old_env, "lambda_orders10": env["orders10"]["lambda"],
                                                "equal": bool(abs(old_env - env["orders10"]["lambda"]) < 1e-12),
                                                **{f"test_frac_{n}_above_round2_lambda": float(np.mean(stats["test"][n] > old_env)) for n in BLOCKERS}}

            # ---- (3) per-policy Bonferroni at (alpha/K, delta/K) over the declared family of 13 ------------------------------
            bonf = {}
            for fname in ("orders10", "all13"):
                members = fams[fname]
                K = len(members)
                lam_b = {m: calib(stats["cal"][m], a0 / K, primary, delta_cfg / K, M) for m in members}
                union_b = float(np.mean(np.any(np.vstack([stats["test"][m] > lam_b[m] for m in members]), axis=0)))
                bonf[fname] = {"K": K, "alpha_per": a0 / K, "delta_per": delta_cfg / K, "lambda_mu": lam_b["mu"], "lambda_per_policy": lam_b,
                               "test_union_stat": union_b, "lambda_mu_is_inf": bool(math.isinf(lam_b["mu"]))}

            # ---- replays under the ten orders: RSC, trajectory (transferred), envelope thresholds, set-completion ---------
            S_uni = {b.qid: mu_prefix(b, m_uni) for b in test}
            m_ad = {b.qid: adaptive_prefix_budget(float(b.features["h_pred"][0]), t_H, k0, b.N) for b in test}
            S_ad = {b.qid: mu_prefix(b, m_ad[b.qid]) for b in test}
            truth = {b.qid: b.true_topk_ids(k0) for b in test}
            base_fail = {"sc_adaptive": float(np.mean([not set(truth[q]).issubset(S_ad[q]) for q in truth])),
                         "sc_uniform": float(np.mean([not set(truth[q]).issubset(S_uni[q]) for q in truth]))}
            lam_set = {"rsc": lam_rsc, "traj": lam_mu, "env_all13": env["all13"]["lambda"], "env_realistic9": env["realistic9"]["lambda"]}
            per_order, fail_by = {}, {}
            for pol in orders:
                row = {}
                calls = {}
                for name, lam_ in lam_set.items():
                    recs = [replay_query(b, MethodSpec(f"{name}_{pol}", "guardrail", lam=lam_, policy=pol, policy_seed=seed, batch_first=bf, batch_rest=br), k0) for b in test]
                    calls[name] = np.array([r["n_calls"] for r in recs], float)
                    mis = np.array([r["mismatch"] for r in recs], float)
                    row[name] = {"risk": float(mis.mean()), "calls_mean": float(calls[name].mean()), "rounds_mean": float(np.mean([r["rounds"] for r in recs]))}
                    fail_by.setdefault(name, []).append(mis)
                for name, S in (("sc_adaptive", S_ad), ("sc_uniform", S_uni)):
                    res = []
                    for b in test:
                        policy = make_policy(pol, {n: b.feature_map(n) for n in b.features}, seed)
                        pred, n_calls, rounds = set_completion(b, S[b.qid], k0, policy, bf, br)
                        res.append((float(pred != truth[b.qid]), n_calls, rounds))
                    mis = np.array([x[0] for x in res])
                    calls[name] = np.array([x[1] for x in res], float)
                    row[name] = {"risk": float(mis.mean()), "calls_mean": float(calls[name].mean()), "rounds_mean": float(np.mean([x[2] for x in res]))}
                    fail_by.setdefault(name, []).append(mis)
                for name in ("sc_adaptive", "sc_uniform", "traj", "env_all13"):
                    row[f"diff_{name}_minus_rsc"] = boot(calls[name], calls["rsc"], seed)
                per_order[pol] = row
                log(f"{teacher}/{domain}/{pol}: RSC {row['rsc']['calls_mean']:.1f} ({row['rsc']['risk']:.3f}) | SC-adaptive {row['sc_adaptive']['calls_mean']:.1f} ({row['sc_adaptive']['risk']:.3f}) | SC-uniform {row['sc_uniform']['calls_mean']:.1f} | env13 {row['env_all13']['calls_mean']:.1f} | traj {row['traj']['calls_mean']:.1f}")
            out["per_order"] = per_order
            out["union_over_orders"] = {name: float(np.mean(np.any(np.vstack(v), axis=0))) for name, v in fail_by.items()}
            out["set_completion_base_failure"] = base_fail
            out["one_shot"] = {"sc_adaptive_calls_mean": float(np.mean(list(m_ad.values()))), "sc_uniform_calls": m_uni}

            # replayed union over the declared family of 13 at its envelope threshold, against the statistic
            lam13 = env["all13"]["lambda"]
            fails = [np.vstack(fail_by["env_all13"])]
            for name, (f_, r_) in BLOCKERS.items():
                fails.append(np.array([[float(replay_query(b, MethodSpec("e13", "guardrail", lam=lam13, policy="mu", batch_first=f_, batch_rest=r_, schedule="blockers"), k0)["mismatch"]) for b in test]]))
            replay_union = np.any(np.vstack(fails), axis=0)
            stat_union = env["all13"]["_e_test"] > lam13
            env["all13"]["test_union_replay"] = float(np.mean(replay_union))
            env["all13"]["replay_equals_stat_per_query"] = bool(np.array_equal(replay_union, stat_union))

            # calls under the mu order (16/8) at every threshold, for the premium ladder
            def mu_calls(lam_):
                return float(np.mean([replay_query(b, MethodSpec("c", "guardrail", lam=lam_, policy="mu", batch_first=bf, batch_rest=br), k0)["n_calls"] for b in test]))

            ladder = {"traj_mu": mu_calls(lam_mu), "rsc": per_order["mu"]["rsc"]["calls_mean"]}
            for fname in fams:
                ladder[f"env_{fname}"] = mu_calls(env[fname]["lambda"])
            for fname in bonf:
                ladder[f"bonf_{fname}"] = mu_calls(bonf[fname]["lambda_mu"])
                bonf[fname]["calls_mu"] = ladder[f"bonf_{fname}"]
            prem = ladder["rsc"] - ladder["traj_mu"]
            for fname in fams:
                env[fname]["calls_mu"] = ladder[f"env_{fname}"]
                env[fname]["share_of_premium"] = ((ladder[f"env_{fname}"] - ladder["traj_mu"]) / prem) if prem > 0 else None
                env[fname]["rsc_minus_env_calls"] = ladder["rsc"] - ladder[f"env_{fname}"]
            for fname in env:
                env[fname].pop("_e_test", None)
            out["envelope"] = env
            out["bonferroni"] = bonf
            out["calls_under_mu"] = ladder

            # ---- (4) covered queries: adaptive budget vs certification core, same queries ------------------------------
            cps = core.checkpoints_from_schedule(test[0].N, k0, core.default_batch_schedule(k0, bf, br))
            rows = [core.cost_anatomy(b.s, b.mu, b.sigma, k0, lam_rsc, b.ids, cps) for b in test]
            cov = [i for i, a in enumerate(rows) if a["covered"]]
            m_cov = np.array([m_ad[test[i].qid] for i in cov], float)
            core_cov = np.array([rows[i]["cert_core"] for i in cov], float)
            rsc_cov = np.array([rows[i]["calls_predicted"] for i in cov], float)
            out["covered_paired"] = {"n_covered": len(cov), "covered_frac": len(cov) / len(test),
                                     "adaptive_budget_mean": float(m_cov.mean()), "cert_core_mean": float(core_cov.mean()), "rsc_calls_mean": float(rsc_cov.mean()),
                                     "diff_budget_minus_core": boot(m_cov, core_cov, seed), "diff_budget_minus_rsc": boot(m_cov, rsc_cov, seed),
                                     "frac_budget_below_core": float(np.mean(m_cov < core_cov))}
            cp = out["covered_paired"]
            if args.frozen_family_split:
                from rscrank.stats import conditional_risk, noninferiority, risk_summary
                meta = read_json(ROOT / "models" / "heads" / teacher / "meta.json")

                def replay_mu(lam_):
                    return [replay_query(b, MethodSpec("h", "guardrail", lam=lam_, policy="mu", policy_seed=seed, batch_first=bf, batch_rest=br), k0) for b in test]

                lam_top = calib(stats["cal"]["B_top"], a0, primary, delta_cfg, M)
                lam_all = calib(stats["cal"]["B_all"], a0, primary, delta_cfg, M)
                rec_rsc, rec_top, rec_all = replay_mu(lam_rsc), replay_mu(lam_top), replay_mu(lam_all)
                rsc_calls = np.array([r["n_calls"] for r in rec_rsc], float)
                ad_calls = np.array([m_ad[b.qid] for b in test], float)
                ad_mis = np.array([float(not set(truth[b.qid]).issubset(set(S_ad[b.qid]))) for b in test])
                uni_mis = np.array([float(not set(truth[b.qid]).issubset(set(S_uni[b.qid]))) for b in test])
                rec_learned = [replay_query(b, MethodSpec("hl", "prefix_learned", c=float(meta["learned_budget_c"])), k0) for b in test]
                nd_rsc = np.array([r["ndcg@k"] for r in rec_rsc], float)
                nd_full = np.array([r["ndcg_full@k"] for r in rec_rsc], float)
                ni = noninferiority(nd_rsc, nd_full, epsilon=float(pr["noninferiority_epsilon"]), n_boot=2000, seed=seed)
                cond = conditional_risk([r["mismatch"] for r in rec_rsc], [r["early_stopped"] for r in rec_rsc])
                covered_rows = [rows[i] for i in cov]
                out["headline"] = {
                    "split_seed": FROZEN_SPLIT_SEED,
                    "rsc": {"risk": float(np.mean([r["mismatch"] for r in rec_rsc])), "calls": float(rsc_calls.mean()),
                            "cp_hi": risk_summary([r["mismatch"] for r in rec_rsc])["cp_hi"],
                            "early_fraction": cond["early_fraction"], "error_given_early": cond["error_given_early"]},
                    "btop": {"risk": float(np.mean([r["mismatch"] for r in rec_top])), "calls": float(np.mean([r["n_calls"] for r in rec_top]))},
                    "ball": {"risk": float(np.mean([r["mismatch"] for r in rec_all])), "calls": float(np.mean([r["n_calls"] for r in rec_all]))},
                    "adaptive_once": {"risk": float(ad_mis.mean()), "calls": float(ad_calls.mean()), "paired_diff_vs_rsc": boot(ad_calls, rsc_calls, seed)},
                    "uniform_once": {"risk": float(uni_mis.mean()), "calls": float(m_uni)},
                    "learned": {"risk": float(np.mean([r["mismatch"] for r in rec_learned])), "calls": float(np.mean([r["n_calls"] for r in rec_learned]))},
                    "ndcg": {"rsc": float(np.nanmean(nd_rsc)), "full": float(np.nanmean(nd_full)), "lcb": ni["lcb"]},
                    "anatomy": {"calls_mean": float(np.mean([a["calls_predicted"] for a in covered_rows])),
                                "cert_core_mean": float(np.mean([a["cert_core"] for a in covered_rows])),
                                "band_mean": float(np.mean([a["band"] for a in covered_rows])),
                                "overshoot_mean": float(np.mean([a["overshoot"] for a in covered_rows]))},
                }
            out_name = "review3_dry" if args.limit else ("a8_split" if args.frozen_family_split else "review3")
            od = ensure_dir(ROOT / cfg["paths"]["runs"] / out_name / teacher)
            write_json(od / f"{domain}.json", out)
            log(f"{teacher}/{domain}: env13 lambda {lam13:.3f}, calls mu traj {ladder['traj_mu']:.1f} env13 {ladder['env_all13']:.1f} rsc {ladder['rsc']:.1f}; replay==stat {env['all13']['replay_equals_stat_per_query']}")
    return 0


def rep08_blocker(first, rest):
    base = core.blocker_batch_schedule(min_batch=rest)

    def sched(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_):
        return first if round_idx == 0 else base(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_)

    sched.adaptive = True  # type: ignore[attr-defined]
    return sched


if __name__ == "__main__":
    sys.exit(main())
