"""Cached-replay experiment matrix (node N10) -> runs/replay/<teacher>/<domain>/<exp>.jsonl + summary.json

Experiments (Sec. 16):
  E1 main      : full / cheap_only / RSC(A_k1) / B_top / B_all / fixed-trajectory / uniform prefix /
                 query-adaptive prefix / learned budget / fixed prefix curve, for every alpha and regime
  E2 policies  : RSC and fixed-trajectory thresholds under 8 scoring orders (no recalibration)
  E3 batches   : batch sizes 8/16/32
  E4 k grid    : k in k_grid (thresholds calibrated per k)
  E5 r grid    : A_{k,r} thresholds, r-miss risk
  E6 N grid    : N=50 prefix of the candidate list (recalibrated); N=200 needs the extended candidate set (optional node)
  E7 cross     : source-domain thresholds applied to FiQA / SciFact (empirical stress test only)
  E8 sigma     : sigma head vs sigma = 1 (both calibrated)
Every record keeps qid so 10_stats.py can bucket by gap/H_k/length.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank.assemble import bundles_from_split, load_split  # noqa: E402
from rscrank.baselines import calibrate_adaptive_prefix, calibrate_uniform_prefix  # noqa: E402
from rscrank.calibrate import calibrate_threshold, family_delta  # noqa: E402
from rscrank.data import cheap_model_for, features_dir_for, load_pool_qrels  # noqa: E402
from rscrank.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from rscrank.policies import make_policy  # noqa: E402
from rscrank.replay import MethodSpec, check_call_dominance, query_statistics, replay_dataset  # noqa: E402
from rscrank.stats import conditional_risk  # noqa: E402


def thr(lams, k, alpha, regime, stat):
    d = lams["k"][str(k)]["alpha"][str(alpha)][regime][stat]
    if d.get("threshold_is_inf"):
        return math.inf
    if d.get("threshold_is_neg_inf"):
        return -math.inf
    return float(d["threshold"])


def _check_fingerprint(cfg, teacher, domain, split, fz, lams=None):
    """Refuse to replay frozen predictions that were not produced by the calibrated system (same cheap model
    and head weights as the thresholds, and as this config): guards against a stale or foreign data/frozen file."""
    if "fingerprint" not in fz.files:
        raise SystemExit(f"frozen file for {teacher}/{domain}/{split} carries no fingerprint; re-run scripts/07_freeze_calibrate.py")
    fp = json.loads(str(fz["fingerprint"]))
    want_cheap = cheap_model_for(cfg, teacher)
    if fp.get("cheap_model") != want_cheap or fp.get("teacher") != teacher:
        raise SystemExit(f"frozen {teacher}/{domain}/{split}: cheap model {fp.get('cheap_model')!r} != config {want_cheap!r}; re-run 07 with this config")
    if lams is not None and lams.get("fingerprint") and lams["fingerprint"].get("heads_sha256") != fp.get("heads_sha256"):
        raise SystemExit(f"frozen {teacher}/{domain}/{split} and lambdas.json come from different heads ({fp.get('heads_sha256')[:12]} vs {lams['fingerprint'].get('heads_sha256', '')[:12]}); re-run 07")
    return fp


def load_bundles(cfg, teacher, domain, split, k, N_cap=None, sigma_one=False, lams=None):
    sd = load_split(features_dir_for(P(cfg, "features"), cheap_model_for(cfg, teacher), teacher), P(cfg, "teacher") / teacher, domain, split)
    fz = np.load(P(cfg, "frozen") / teacher / domain / f"{split}.npz")
    _check_fingerprint(cfg, teacher, domain, split, fz, lams)
    qrels = load_pool_qrels(cfg, domain, P(cfg, "data_raw"))
    sig = None if sigma_one else fz["sigma_q"]
    bundles = bundles_from_split(sd, fz["mu"], fz["q_index"], sig, k, qrels, fz["h_q"], N_cap=N_cap)
    # policy "mu_alt" (amendment A3): the order of the cheap-model variant NOT adopted for this teacher (its frozen
    # predictions live under runs/<other>_eval/frozen); the guardrail keeps the adopted heads.  Same candidates, same
    # teacher cache, so the flat mu aligns with the same q_index.
    adopted = cheap_model_for(cfg, teacher)
    alt = ROOT / "runs" / ("frozen_eval" if adopted == "minilm-distilled" else "distilled_eval") / "frozen" / teacher / domain / f"{split}.npz"
    if alt.exists():
        fa = np.load(alt)
        if fa["mu"].shape == fz["mu"].shape and np.array_equal(fa["q_index"], fz["q_index"]):
            mu_alt = fa["mu"].astype(float)
            by_q = {}
            for b in bundles:
                by_q[b.qid] = b
            for i, qid in enumerate(sd.qids):
                if qid in by_q:
                    sel = np.where(fz["q_index"] == i)[0][: len(by_q[qid].ids)]
                    by_q[qid].features["mu_alt"] = mu_alt[sel]
    return bundles


def summarize(recs, alpha=None):
    if not recs:
        return {}
    m = np.array([r["mismatch"] for r in recs], float)
    e = np.array([r["early_stopped"] for r in recs], bool)
    out = {"n": len(recs), "risk": float(m.mean()), "calls_mean": float(np.mean([r["n_calls"] for r in recs])), "calls_p50": float(np.median([r["n_calls"] for r in recs])), "calls_p95": float(np.quantile([r["n_calls"] for r in recs], 0.95)), "rounds_mean": float(np.mean([r["rounds"] for r in recs])), "full_fallback": float(np.mean([r["full_fallback"] for r in recs])), "overlap": float(np.mean([r["overlap"] for r in recs]))}
    out.update(conditional_risk(m, e))
    for key in ("ell_1", "ell_2", "ell_3", "ndcg@k", "ndcg_full@k", "mrr@k", "recall@k", "tokens_real", "tokens_padded"):
        vals = [r[key] for r in recs if key in r and r[key] == r[key]]
        if vals:
            out[key] = float(np.mean(vals))
    if alpha is not None:
        out["alpha"] = alpha
        out["risk_within_alpha"] = bool(out["risk"] <= alpha)
    return out


def strip(recs):
    for r in recs:
        r.pop("pred", None)
    return recs


def main() -> int:
    ap = base_parser("replay experiments")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--exp", default="E1,E2,E3,E4,E5,E6,E7,E8,E9")
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k0, a0 = pr["k"], pr["alpha_main"]
    primary = pr.get("primary_regime", "finite")   # the regime whose guarantee is about THIS test split
    exps = set(args.exp.split(","))
    delta_cfg = family_delta(pr["delta_family"], pr["n_main_configs"])
    tlist = [args.teacher] if args.teacher else teachers(cfg)
    for teacher in tlist:
        meta = read_json(P(cfg, "models") / "heads" / teacher / "meta.json")
        for domain in main_domains(cfg):
            lams = read_json(ROOT / cfg["paths"]["runs"] / "calib" / teacher / domain / "lambdas.json")
            out_dir = ensure_dir(ROOT / cfg["paths"]["runs"] / "replay" / teacher / domain)
            summary = {"teacher": teacher, "domain": domain, "k": k0, "alpha_main": a0, "primary_regime": primary, "experiments": {}, "fingerprint": lams.get("fingerprint"), "lambdas_file": str((ROOT / cfg["paths"]["runs"] / "calib" / teacher / domain / "lambdas.json").relative_to(ROOT))}
            if (out_dir / "summary.json").exists() and exps != set("E1,E2,E3,E4,E5,E6,E7,E8,E9".split(",")):
                # a partial run (--exp) updates only the selected experiments and keeps the others' summaries
                prev = read_json(out_dir / "summary.json")
                summary["experiments"] = {e: v for e, v in prev.get("experiments", {}).items() if e not in exps}
            test = load_bundles(cfg, teacher, domain, "test", k0, lams=lams)
            bf, br = pr["batch_first"], pr["batch_rest"]
            log(f"{teacher}/{domain}: {len(test)} test queries")

            if "E1" in exps or "E3" in exps:
                # exact ties of the cached teacher scores at the top-k boundary (review round 1): the ordered-list target
                # then depends on the identifier tie-break; report how often that happens
                ties = []
                for b in test:
                    ss = np.sort(np.asarray(b.s, float))[::-1]
                    ties.append(bool(ss[k0 - 1] == ss[k0]))
                summary["boundary_tie_fraction"] = float(np.mean(ties))
                summary["boundary_tie_count"] = int(np.sum(ties))
            if "E1" in exps:
                recs_all, S = [], {}
                base = {"full": MethodSpec("full", "full"), "cheap_only": MethodSpec("cheap_only", "cheap_only")}
                for name, spec in base.items():
                    recs = strip(replay_dataset(test, spec, k0))
                    S[name] = summarize(recs)
                    recs_all += recs
                for m in sorted(set([k0, 20, 30, 40, 50, 60, 70, 80, 90, 100] + [test[0].N])):
                    if k0 <= m <= test[0].N:
                        recs = strip(replay_dataset(test, MethodSpec(f"prefix_fixed_{m}", "prefix_fixed", m=m), k0))
                        S[f"prefix_fixed_{m}"] = summarize(recs)
                        recs_all += recs
                for alpha in pr["alpha_grid"]:
                    for regime in pr["regimes"]:
                        tag = f"a{alpha}_{regime}"
                        specs = {
                            f"rsc_{tag}": MethodSpec(f"rsc_{tag}", "guardrail", lam=thr(lams, k0, alpha, regime, "A_k1"), policy="mu", batch_first=bf, batch_rest=br),
                            f"btop_{tag}": MethodSpec(f"btop_{tag}", "guardrail", lam=thr(lams, k0, alpha, regime, "B_top"), policy="mu", batch_first=bf, batch_rest=br),
                            f"ball_{tag}": MethodSpec(f"ball_{tag}", "guardrail", lam=thr(lams, k0, alpha, regime, "B_all"), policy="mu", batch_first=bf, batch_rest=br),
                            f"traj_{tag}": MethodSpec(f"traj_{tag}", "guardrail", lam=thr(lams, k0, alpha, regime, "traj_mu"), policy="mu", batch_first=bf, batch_rest=br),
                            f"prefix_uniform_{tag}": MethodSpec(f"prefix_uniform_{tag}", "prefix_uniform", m=int(min(thr(lams, k0, alpha, regime, "H_k"), test[0].N))),
                            f"prefix_adaptive_{tag}": MethodSpec(f"prefix_adaptive_{tag}", "prefix_adaptive", t_H=thr(lams, k0, alpha, regime, "e_H")),
                        }
                        got = {}
                        for name, spec in specs.items():
                            recs = strip(replay_dataset(test, spec, k0))
                            got[name] = recs
                            S[name] = summarize(recs, alpha)
                            recs_all += recs
                        S[f"dominance_{tag}"] = check_call_dominance(got[f"rsc_{tag}"], got[f"btop_{tag}"], got[f"ball_{tag}"])
                recs = strip(replay_dataset(test, MethodSpec("prefix_learned", "prefix_learned", c=meta["learned_budget_c"]), k0))
                S["prefix_learned"] = summarize(recs, a0)
                recs_all += recs
                write_jsonl(out_dir / "E1_main.jsonl", recs_all)
                summary["experiments"]["E1"] = S
                log(f"E1 done: rsc risk={S[f'rsc_a{a0}_marginal']['risk']:.4f} calls={S[f'rsc_a{a0}_marginal']['calls_mean']:.1f} | uniform prefix calls={S[f'prefix_uniform_a{a0}_marginal']['calls_mean']:.1f} | adaptive prefix calls={S[f'prefix_adaptive_a{a0}_marginal']['calls_mean']:.1f}")

            if "E2" in exps:
                S, recs_all = {}, []
                lamA, lamT = thr(lams, k0, a0, primary, "A_k1"), thr(lams, k0, a0, primary, "traj_mu")
                for pol in pr["policies"]:
                    for name, lam in (("rsc", lamA), ("traj", lamT)):
                        recs = strip(replay_dataset(test, MethodSpec(f"{name}_{pol}", "guardrail", lam=lam, policy=pol, policy_seed=cfg["seed"], batch_first=bf, batch_rest=br), k0))
                        S[f"{name}_{pol}"] = summarize(recs, a0)
                        recs_all += recs
                # fixed-trajectory recalibrated for each other policy (fair comparison, Sec. 14.4)
                cal = load_bundles(cfg, teacher, domain, "cal", k0, lams=lams)
                for pol in pr["policies"]:
                    if pol == "mu":
                        continue
                    vals = []
                    for b in cal:
                        oracle = core.TeacherOracle(b.ids, b.s)
                        from rscrank.policies import make_policy

                        res = core.run_controller(b.ids, b.mu, b.sigma, math.inf, k0, oracle, make_policy(pol, {n: b.feature_map(n) for n in b.features}, cfg["seed"]), core.default_batch_schedule(k0, bf, br))
                        prefixes = core.prefixes_from_trajectory(res.trajectory[:-1], b.ids)
                        vals.append(core.trajectory_critical_value(b.s, b.mu, b.sigma, k0, prefixes, 1, b.ids))
                    lam_re = calibrate_threshold(vals, a0, primary, delta=delta_cfg, M=len(test)).threshold
                    recs = strip(replay_dataset(test, MethodSpec(f"traj_recal_{pol}", "guardrail", lam=lam_re, policy=pol, policy_seed=cfg["seed"], batch_first=bf, batch_rest=br), k0))
                    S[f"traj_recal_{pol}"] = summarize(recs, a0)
                    recs_all += recs
                # E2b (amendment 2026-09-22, docs/PREREGISTRATION.md): the calibrated PREFIX baselines under the same
                # alternative orders, (i) transferred -- the mu-calibrated budget m / m(q) applied to the first documents
                # of the other order -- and (ii) recalibrated on that order's horizon H_k^pi.  This mirrors the
                # transferred / recalibrated treatment of the fixed-trajectory threshold above.
                from rscrank.baselines import horizon_under_order

                L0 = lams["k"][str(k0)]["alpha"][str(a0)][primary]
                m_uni = int(min(thr(lams, k0, a0, primary, "H_k"), test[0].N))
                t_H = thr(lams, k0, a0, primary, "e_H")
                for pol in pr["policies"]:
                    if pol in ("mu", "observed_regression"):
                        continue  # observed_regression has no order before any score is observed (it starts as mu)
                    for name, spec in (("prefix_uniform", MethodSpec(f"prefix_uniform_{pol}", "prefix_uniform", m=m_uni, policy=pol, policy_seed=cfg["seed"])),
                                       ("prefix_adaptive", MethodSpec(f"prefix_adaptive_{pol}", "prefix_adaptive", t_H=t_H, policy=pol, policy_seed=cfg["seed"]))):
                        recs = strip(replay_dataset(test, spec, k0))
                        S[spec.name] = summarize(recs, a0)
                        recs_all += recs
                    # recalibrated: horizon of the true winners under this order on the calibration split
                    Hs, hp = [], []
                    for b in cal:
                        order = list(make_policy(pol, {n: b.feature_map(n) for n in b.features}, cfg["seed"])(list(b.ids), {}, {i: float(v) for i, v in zip(b.ids, b.mu)}))
                        Hs.append(horizon_under_order(order, b.true_topk_ids(k0)))
                        hp.append(float(b.features["h_pred"][0]))
                    m_re = calibrate_uniform_prefix(Hs, a0, primary, delta=delta_cfg, M=len(test), N=test[0].N)
                    e_re = calibrate_adaptive_prefix(Hs, hp, a0, primary, delta=delta_cfg, M=len(test))
                    for spec in (MethodSpec(f"prefix_uniform_recal_{pol}", "prefix_uniform", m=int(min(m_re.threshold, test[0].N)), policy=pol, policy_seed=cfg["seed"]),
                                 MethodSpec(f"prefix_adaptive_recal_{pol}", "prefix_adaptive", t_H=e_re.threshold, policy=pol, policy_seed=cfg["seed"])):
                        recs = strip(replay_dataset(test, spec, k0))
                        S[spec.name] = summarize(recs, a0)
                        recs_all += recs
                # E2c: union-over-orders risk.  A serving system may pick the order per query (by load, batch shape, cache
                # state); per-order calibration bounds each order's risk in isolation, the relevant quantity for a system
                # that switches is the fraction of queries that fail under AT LEAST ONE of the tested orders.  For RSC the
                # error event is (almost) order-invariant (A_k1 > lambda), so its union stays within the uncovered fraction.
                by_qid = {}
                for r in recs_all:
                    m = r["method"]
                    for prefix, key in (("traj_recal_", "traj_recal"), ("prefix_adaptive_recal_", "prefix_adaptive_recal"), ("prefix_uniform_recal_", "prefix_uniform_recal"),
                                        ("rsc_", "rsc"), ("traj_", "traj"), ("prefix_adaptive_", "prefix_adaptive"), ("prefix_uniform_", "prefix_uniform")):
                        if m.startswith(prefix):
                            pol_ = m[len(prefix):]
                            if pol_ in pr["policies"]:
                                by_qid.setdefault(key, {}).setdefault(r["qid"], []).append(bool(r["mismatch"]))
                            break
                U = {}
                for key, d in by_qid.items():
                    n_q = len(d)
                    n_orders = max(len(v) for v in d.values())
                    U[key] = {"union_risk": float(np.mean([any(v) for v in d.values()])), "n_queries": n_q, "n_orders": n_orders,
                              "mean_per_order_risk": float(np.mean([np.mean(v) for v in d.values()]))}
                S["union_over_orders"] = U
                write_jsonl(out_dir / "E2_policies.jsonl", recs_all)
                summary["experiments"]["E2"] = S
                log("E2c union-over-orders risk: " + ", ".join(f"{k_}: {v['union_risk']:.4f} ({v['n_orders']} orders)" for k_, v in U.items()))
                log("E2 done: " + ", ".join(f"{p}: rsc {S['rsc_' + p]['risk']:.3f}/{S['rsc_' + p]['calls_mean']:.0f} traj {S['traj_' + p]['risk']:.3f}" for p in pr["policies"]))
                log("E2b done: " + ", ".join(f"{p}: adaptive-prefix transferred {S['prefix_adaptive_' + p]['risk']:.3f}/{S['prefix_adaptive_' + p]['calls_mean']:.0f} recal {S['prefix_adaptive_recal_' + p]['risk']:.3f}/{S['prefix_adaptive_recal_' + p]['calls_mean']:.0f}" for p in pr["policies"] if p not in ("mu", "observed_regression")))

            if "E3" in exps:
                S, recs_all = {}, []
                lamA = thr(lams, k0, a0, primary, "A_k1")
                for b_ in pr["batch_grid"]:
                    recs = strip(replay_dataset(test, MethodSpec(f"rsc_batch{b_}", "guardrail", lam=lamA, policy="mu", batch_first=b_, batch_rest=b_), k0))
                    S[f"rsc_batch{b_}"] = summarize(recs, a0)
                    recs_all += recs
                # E3b (amendment A2): blocker-batch schedule -- first round bf documents, then one round per blocker set
                # (all unscored documents whose guardrail reaches the current k-th score, at least `rest`).  Same lambda,
                # legal policy; it trades a few extra calls for far fewer rounds (what a GPU wants).
                lamT = thr(lams, k0, a0, primary, "traj_mu")
                for first_, rest_ in ((bf, br), (bf, bf), (2 * bf, bf)):
                    name = f"rsc_blockers{first_}_{rest_}"
                    recs = strip(replay_dataset(test, MethodSpec(name, "guardrail", lam=lamA, policy="mu", batch_first=first_, batch_rest=rest_, schedule="blockers"), k0))
                    S[name] = summarize(recs, a0)
                    recs_all += recs
                    # the fixed-trajectory threshold was calibrated on the checkpoints of the 16/8 schedule; the blocker
                    # schedule evaluates the rule at different prefix sizes of the same order (a schedule change, no
                    # recalibration) -- review round 1 asked for exactly this kind of transfer
                    recs = strip(replay_dataset(test, MethodSpec(f"traj_blockers{first_}_{rest_}", "guardrail", lam=lamT, policy="mu", batch_first=first_, batch_rest=rest_, schedule="blockers"), k0))
                    S[f"traj_blockers{first_}_{rest_}"] = summarize(recs, a0)
                    recs_all += recs
                for b_ in pr["batch_grid"]:
                    recs = strip(replay_dataset(test, MethodSpec(f"traj_batch{b_}", "guardrail", lam=lamT, policy="mu", batch_first=b_, batch_rest=b_), k0))
                    S[f"traj_batch{b_}"] = summarize(recs, a0)
                    recs_all += recs
                # union over the tested schedules (16/8 main + fixed batches + blocker schedules) for RSC vs the transferred trajectory threshold
                by_q = {"rsc": {}, "traj": {}}
                for r in recs_all:
                    key = "rsc" if r["method"].startswith("rsc_") else "traj"
                    by_q[key].setdefault(r["qid"], []).append(bool(r["mismatch"]))
                main_recs = strip(replay_dataset(test, MethodSpec("rsc_main_sched", "guardrail", lam=lamA, policy="mu", batch_first=bf, batch_rest=br), k0))
                for r in main_recs:
                    by_q["rsc"].setdefault(r["qid"], []).append(bool(r["mismatch"]))
                traj_main = strip(replay_dataset(test, MethodSpec("traj_main_sched", "guardrail", lam=lamT, policy="mu", batch_first=bf, batch_rest=br), k0))
                for r in traj_main:
                    by_q["traj"].setdefault(r["qid"], []).append(bool(r["mismatch"]))
                S["union_over_schedules"] = {key: {"union_risk": float(np.mean([any(v) for v in d.values()])), "n_schedules": max(len(v) for v in d.values()), "n_queries": len(d)} for key, d in by_q.items()}
                write_jsonl(out_dir / "E3_batches.jsonl", recs_all)
                summary["experiments"]["E3"] = S
                log("E3 done: " + ", ".join(f"{m}: {v['calls_mean']:.1f} calls / {v['rounds_mean']:.2f} rounds / risk {v['risk']:.3f}" for m, v in S.items() if isinstance(v, dict) and "calls_mean" in v)
                    + f"; union over schedules: rsc {S['union_over_schedules']['rsc']['union_risk']:.4f} traj {S['union_over_schedules']['traj']['union_risk']:.4f}")

            if "E4" in exps:
                S, recs_all = {}, []
                for k in pr["k_grid"]:
                    if str(k) not in lams["k"] or k >= test[0].N:
                        continue
                    tk = load_bundles(cfg, teacher, domain, "test", k, lams=lams)
                    for name, stat in (("rsc", "A_k1"), ("btop", "B_top"), ("ball", "B_all")):
                        recs = strip(replay_dataset(tk, MethodSpec(f"{name}_k{k}", "guardrail", lam=thr(lams, k, a0, primary, stat), policy="mu", batch_first=max(k + 6, k), batch_rest=max(k // 2 + 3, 1)), k))
                        S[f"{name}_k{k}"] = summarize(recs, a0)
                        recs_all += recs
                    recs = strip(replay_dataset(tk, MethodSpec(f"prefix_uniform_k{k}", "prefix_uniform", m=int(min(thr(lams, k, a0, primary, "H_k"), tk[0].N))), k))
                    S[f"prefix_uniform_k{k}"] = summarize(recs, a0)
                    recs_all += recs
                write_jsonl(out_dir / "E4_kgrid.jsonl", recs_all)
                summary["experiments"]["E4"] = S

            if "E5" in exps:
                S, recs_all = {}, []
                for r in pr["r_grid"]:
                    key = f"A_k{r}"
                    if key not in lams["k"][str(k0)]["alpha"][str(a0)][primary]:
                        continue
                    recs = strip(replay_dataset(test, MethodSpec(f"rsc_r{r}", "guardrail", lam=thr(lams, k0, a0, primary, key), policy="mu", batch_first=bf, batch_rest=br), k0))
                    S[f"rsc_r{r}"] = summarize(recs, a0)
                    S[f"rsc_r{r}"]["risk_ell_r"] = float(np.mean([x[f"ell_{r}"] for x in recs]))
                    recs_all += recs
                write_jsonl(out_dir / "E5_rgrid.jsonl", recs_all)
                summary["experiments"]["E5"] = S

            if "E6" in exps:
                S, recs_all = {}, []
                for Ncap in [n for n in cfg["candidates"]["N_grid"] if n < test[0].N]:
                    cal = load_bundles(cfg, teacher, domain, "cal", k0, N_cap=Ncap, lams=lams)
                    tN = load_bundles(cfg, teacher, domain, "test", k0, N_cap=Ncap, lams=lams)
                    lamA = calibrate_threshold([query_statistics(b, k0)["A_k1"] for b in cal], a0, primary, delta=delta_cfg, M=len(tN)).threshold
                    mH = calibrate_uniform_prefix([core.H_k(b.s, b.mu, k0, b.ids) for b in cal], a0, primary, delta=delta_cfg, M=len(tN), N=Ncap).threshold
                    recs = strip(replay_dataset(tN, MethodSpec(f"rsc_N{Ncap}", "guardrail", lam=lamA, policy="mu", batch_first=bf, batch_rest=br), k0))
                    S[f"rsc_N{Ncap}"] = summarize(recs, a0)
                    recs_all += recs
                    recs = strip(replay_dataset(tN, MethodSpec(f"prefix_uniform_N{Ncap}", "prefix_uniform", m=int(min(mH, Ncap))), k0))
                    S[f"prefix_uniform_N{Ncap}"] = summarize(recs, a0)
                    recs_all += recs
                write_jsonl(out_dir / "E6_Ngrid.jsonl", recs_all)
                summary["experiments"]["E6"] = S
                summary["experiments"]["E6"]["note"] = "N=200 requires the extended candidate set (optional node N10b)"

            if "E7" in exps:
                S, recs_all = {}, []
                for cd in cfg["domains"]["cross"]:
                    fz = P(cfg, "frozen") / teacher / cd / "test.npz"
                    if not fz.exists():
                        continue
                    tc = load_bundles(cfg, teacher, cd, "test", k0, lams=lams)
                    for regime in pr["regimes"]:
                        lamA = thr(lams, k0, a0, regime, "A_k1")
                        recs = strip(replay_dataset(tc, MethodSpec(f"rsc_{cd}_{regime}", "guardrail", lam=lamA, policy="mu", batch_first=bf, batch_rest=br), k0))
                        S[f"rsc_{cd}_{regime}"] = summarize(recs, a0)
                        recs_all += recs
                    mH = int(min(thr(lams, k0, a0, primary, "H_k"), tc[0].N))
                    recs = strip(replay_dataset(tc, MethodSpec(f"prefix_uniform_{cd}", "prefix_uniform", m=mH), k0))
                    S[f"prefix_uniform_{cd}"] = summarize(recs, a0)
                    recs_all += recs
                    recs = strip(replay_dataset(tc, MethodSpec(f"prefix_adaptive_{cd}", "prefix_adaptive", t_H=thr(lams, k0, a0, primary, "e_H")), k0))
                    S[f"prefix_adaptive_{cd}"] = summarize(recs, a0)
                    recs_all += recs
                    # scale-free variant (review round 1): sigma = 1 everywhere, threshold calibrated on the SOURCE domain's
                    # calibration split with sigma = 1, transferred to the cross domain -- does dropping the learned scale
                    # head make the rank-sufficient certificate transfer?
                    cal1 = load_bundles(cfg, teacher, domain, "cal", k0, sigma_one=True, lams=lams)
                    lam1 = calibrate_threshold([query_statistics(b, k0)["A_k1"] for b in cal1], a0, primary, delta=delta_cfg, M=len(test)).threshold
                    tc1 = load_bundles(cfg, teacher, cd, "test", k0, sigma_one=True, lams=lams)
                    recs = strip(replay_dataset(tc1, MethodSpec(f"rsc_sigma_one_{cd}", "guardrail", lam=lam1, policy="mu", batch_first=bf, batch_rest=br), k0))
                    S[f"rsc_sigma_one_{cd}"] = summarize(recs, a0)
                    recs_all += recs
                write_jsonl(out_dir / "E7_cross.jsonl", recs_all)
                summary["experiments"]["E7"] = S

            if "E8" in exps:
                S, recs_all = {}, []
                cal1 = load_bundles(cfg, teacher, domain, "cal", k0, sigma_one=True, lams=lams)
                t1 = load_bundles(cfg, teacher, domain, "test", k0, sigma_one=True, lams=lams)
                lam1 = calibrate_threshold([query_statistics(b, k0)["A_k1"] for b in cal1], a0, primary, delta=delta_cfg, M=len(t1)).threshold
                recs = strip(replay_dataset(t1, MethodSpec("rsc_sigma_one", "guardrail", lam=lam1, policy="mu", batch_first=bf, batch_rest=br), k0))
                S["rsc_sigma_one"] = summarize(recs, a0)
                recs_all += recs
                recs = strip(replay_dataset(test, MethodSpec("rsc_sigma_head", "guardrail", lam=thr(lams, k0, a0, primary, "A_k1"), policy="mu", batch_first=bf, batch_rest=br), k0))
                S["rsc_sigma_head"] = summarize(recs, a0)
                # (round-2 R2) 'global-estimate' coverage stop: one calibrated radius for every unscored document, sigma = 1 --
                # the Fast-Forward-style rule (k-th verified score vs a single global upper-bound estimate), calibrated on the
                # same calibration split via B_all with sigma = 1
                lamB1 = calibrate_threshold([query_statistics(b, k0)["B_all"] for b in cal1], a0, primary, delta=delta_cfg, M=len(t1)).threshold
                recs = strip(replay_dataset(t1, MethodSpec("ball_sigma_one", "guardrail", lam=lamB1, policy="mu", batch_first=bf, batch_rest=br), k0))
                S["ball_sigma_one"] = summarize(recs, a0)
                recs_all += recs
                S["use_sigma_head_selected"] = bool(meta["use_sigma_head"])
                recs_all += recs
                write_jsonl(out_dir / "E8_sigma.jsonl", recs_all)
                summary["experiments"]["E8"] = S

            if "E9" in exps:
                lamA = thr(lams, k0, a0, primary, "A_k1")
                sched = core.default_batch_schedule(k0, bf, br)
                rows = []
                for b in test:
                    cps = core.checkpoints_from_schedule(b.N, k0, sched)
                    a = core.cost_anatomy(b.s, b.mu, b.sigma, k0, lamA, b.ids, cps)
                    a["qid"] = b.qid
                    rows.append(a)
                recs = strip(replay_dataset(test, MethodSpec("rsc_anatomy", "guardrail", lam=lamA, policy="mu", batch_first=bf, batch_rest=br), k0))
                calls = {r["qid"]: r["n_calls"] for r in recs}
                cov = [a for a in rows if a["covered"]]
                exact = all(int(a["calls_predicted"]) == calls[a["qid"]] for a in cov)
                st_path = ROOT / cfg["paths"]["runs"] / "calib" / teacher / domain / f"stats_test_k{k0}.jsonl"
                pou = [float(r["A_k1"]) - float(r["traj_mu"]) for r in read_jsonl(st_path) if math.isfinite(float(r.get("traj_mu", float("-inf"))))] if st_path.exists() else []
                # The decomposition calls = |V*| + band + overshoot is a statement about COVERED queries (A_k1 <= lambda,
                # Prop. 11); on an uncovered query the guardrail stop can be unsafe and the decomposition does not apply
                # (the controller usually still stops early there -- that is where the alpha-fraction of errors lives).
                # All anatomy means are therefore reported over covered queries; all-query means are kept under *_allq.
                S = {
                    "lambda": lamA, "regime": primary, "n": len(rows), "n_covered": len(cov), "covered_frac": float(np.mean([a["covered"] for a in rows])),
                    "population": "covered test queries (A_k1 <= lambda); on uncovered queries the guardrail stop can be unsafe and Prop. 11 does not apply",
                    "cert_core_mean": float(np.mean([a["cert_core"] for a in cov])), "band_mean": float(np.mean([a["band"] for a in cov])),
                    "overshoot_mean": float(np.mean([a["overshoot"] for a in cov])), "calls_mean": float(np.mean([calls[a["qid"]] for a in cov])),
                    "oracle_gap_mean": float(np.mean([calls[a["qid"]] - a["cert_core"] for a in cov])),
                    "calls_mean_allq": float(np.mean(list(calls.values()))),
                    "cert_core_mean_allq": float(np.mean([a["cert_core"] for a in rows])),
                    "oracle_gap_mean_allq": float(np.mean([calls[a["qid"]] - a["cert_core"] for a in rows])),
                    "anatomy_exact_on_covered": bool(exact),
                    "winners_below_theta_mean": float(np.mean([a["winners_below_theta"] for a in cov])),
                    "price_of_uniformity_mean": float(np.mean(pou)) if pou else None,
                    "price_of_uniformity_p90": float(np.quantile(pou, 0.9)) if pou else None,
                    "price_of_uniformity_zero_frac": float(np.mean([p <= 0 for p in pou])) if pou else None,
                }
                write_jsonl(out_dir / "E9_anatomy.jsonl", rows)
                summary["experiments"]["E9"] = S
                log(f"E9 done (covered queries, {S['covered_frac']:.3f}): cert_core {S['cert_core_mean']:.1f} + band {S['band_mean']:.2f} + overshoot {S['overshoot_mean']:.2f} = calls {S['calls_mean']:.1f} (exact on covered: {exact}); all-query calls {S['calls_mean_allq']:.1f}; PoU mean {S['price_of_uniformity_mean']}")

            write_json(out_dir / "summary.json", summary)
            log(f"summary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
