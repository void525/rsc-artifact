"""Freeze the system (heads + candidates + teacher cache) and calibrate every
threshold family on the calibration split of each main domain (node N09).

Outputs per (teacher, domain):
  data/frozen/<teacher>/<domain>/<split>.npz      mu (flat), q_index, sigma_q, h_q  for split in cal,test (+ cross-domain test)
  runs/calib/<teacher>/<domain>/stats_<split>.jsonl   per-query statistics (A_k,r, B_top, B_all, H_k, traj_mu, gap)
  runs/calib/<teacher>/<domain>/lambdas.json          thresholds for every (k, alpha, regime, statistic)
"""
from __future__ import annotations

import json
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank.assemble import bundles_from_split, load_split, pair_matrix, query_features_for_split  # noqa: E402
from rscrank.baselines import calibrate_adaptive_prefix, calibrate_uniform_prefix  # noqa: E402
from rscrank.calibrate import calibrate_threshold, family_delta  # noqa: E402
from rscrank.data import domain_list, cheap_model_for, features_dir_for, load_pool_qrels  # noqa: E402
from rscrank.heads import BudgetHead, MeanHead, ScaleHead  # noqa: E402
from rscrank.io import ensure_dir, read_json, write_json, write_jsonl  # noqa: E402
from rscrank.replay import query_statistics  # noqa: E402


def system_fingerprint(cfg, teacher) -> dict:
    """What the frozen predictions and the thresholds are a function of: the cheap model, the head weights
    (sha256 of the three npz files + meta.json) and the teacher.  Stored in every frozen npz and in lambdas.json;
    scripts/08_replay_eval.py refuses to combine files with different fingerprints."""
    hd = P(cfg, "models") / "heads" / teacher
    h = hashlib.sha256()
    for name in ("mu.npz", "sigma.npz", "budget.npz", "meta.json"):
        h.update((hd / name).read_bytes())
    return {"teacher": teacher, "cheap_model": cheap_model_for(cfg, teacher), "heads_sha256": h.hexdigest(), "heads_dir": str(hd.relative_to(ROOT)) if hd.is_relative_to(ROOT) else str(hd)}


def freeze_split(cfg, teacher, heads, domain, split, k):
    sd = load_split(features_dir_for(P(cfg, "features"), cheap_model_for(cfg, teacher), teacher), P(cfg, "teacher") / teacher, domain, split)
    if sd is None or sd.t is None:
        return None, None
    X, y, q = pair_matrix(sd, k + 1)
    mu = heads["mu"].predict(X)
    Z = query_features_for_split(sd, mu, q, k)
    ok = ~np.isnan(Z).any(1)
    sigma_q = np.ones(sd.Q)
    if heads["meta"]["use_sigma_head"]:
        sigma_q[ok] = heads["sigma"].predict(Z[ok])
    h_q = np.zeros(sd.Q)
    h_q[ok] = heads["budget"].predict(Z[ok])
    out = ensure_dir(P(cfg, "frozen") / teacher / domain)
    # mu is stored in float64: the calibrated statistics are computed from exactly these values and the controller replays them (docs/THEORY.md Sec. 8 requires identical stored inputs on both sides)
    fp = system_fingerprint(cfg, teacher)
    np.savez_compressed(out / f"{split}.npz", mu=np.asarray(mu, dtype=np.float64), q_index=q, sigma_q=np.asarray(sigma_q, dtype=np.float64), h_q=np.asarray(h_q, dtype=np.float64),
                        fingerprint=np.array(json.dumps(fp, sort_keys=True)), qids=np.asarray(sd.qids, dtype=str), use_sigma_head=np.array(bool(heads["meta"]["use_sigma_head"])))
    return sd, {"mu": mu, "q": q, "sigma_q": sigma_q, "h_q": h_q}


def stats_for(sd, fz, k, r_grid, batch_first, batch_rest, qrels):
    bundles = bundles_from_split(sd, fz["mu"], fz["q"], fz["sigma_q"], k, qrels, fz["h_q"])
    rows = []
    for b in bundles:
        st = query_statistics(b, k, r_grid)
        # trajectory-specific statistic for the main policy (mu-descending, main batch schedule)
        order = [b.ids[p] for p in core.teacher_order(b.mu, b.ids)]
        sched = core.default_batch_schedule(k, batch_first, batch_rest)
        prefixes, cur, t = [], [], 0
        while len(cur) < b.N:
            cur = cur + order[len(cur) : len(cur) + sched(t)]
            t += 1
            if len(cur) < b.N:
                prefixes.append([b.ids.index(i) for i in cur])
        st["traj_mu"] = core.trajectory_critical_value(b.s, b.mu, b.sigma, k, prefixes, 1, b.ids)
        st["h_pred"] = float(b.features["h_pred"][0])
        st["e_H"] = math.log(st["H_k"]) - st["h_pred"]
        st["sigma_q"] = float(b.sigma[0])
        rows.append(st)
    return rows, bundles


def main() -> int:
    ap = base_parser("freeze + calibrate")
    ap.add_argument("--teacher", default=None)
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    tlist = [args.teacher] if args.teacher else teachers(cfg)
    delta_cfg = family_delta(pr["delta_family"], pr["n_main_configs"])
    for teacher in tlist:
        hd = P(cfg, "models") / "heads" / teacher
        meta = read_json(hd / "meta.json")
        heads = {"mu": MeanHead.load(hd / "mu.npz", meta["kappa"]), "sigma": ScaleHead.load(hd / "sigma.npz"), "budget": BudgetHead.load(hd / "budget.npz"), "meta": meta}
        for domain in domain_list(cfg):
            is_main = domain in main_domains(cfg)
            qrels = load_pool_qrels(cfg, domain, P(cfg, "data_raw"))
            out_dir = ensure_dir(ROOT / cfg["paths"]["runs"] / "calib" / teacher / domain)
            lam = {"teacher": teacher, "domain": domain, "delta_cfg": delta_cfg, "k": {}, "fingerprint": system_fingerprint(cfg, teacher)}
            per_split = {}
            for split in (("cal", "test") if is_main else ("test",)):
                # freeze (mu, sigma_q, h_q) once per split at the main k; the frozen predictions do not depend on k,
                # so every k in the grid (including those listed before the main k) reuses them
                sd, fz = freeze_split(cfg, teacher, heads, domain, split, pr["k"])
                if sd is None:
                    continue
                per_split[("sd", split)], per_split[("fz", split)] = sd, fz
                for k in pr["k_grid"]:
                    if not (1 <= k < sd.N):
                        continue
                    r_grid = [r for r in pr["r_grid"] if 1 <= r <= min(k, sd.N - k)]
                    rows, _ = stats_for(sd, fz, k, r_grid, pr["batch_first"], pr["batch_rest"], qrels)
                    write_jsonl(out_dir / f"stats_{split}_k{k}.jsonl", rows)
                    if split == "cal":
                        n = len(rows)
                        M = None
                        if ("sd", "test") not in per_split:
                            sdt, fzt = freeze_split(cfg, teacher, heads, domain, "test", pr["k"])
                            per_split[("sd", "test")], per_split[("fz", "test")] = sdt, fzt
                        M = sum(1 for _ in bundles_from_split(per_split[("sd", "test")], per_split[("fz", "test")]["mu"], per_split[("fz", "test")]["q"], per_split[("fz", "test")]["sigma_q"], k))
                        lam["k"].setdefault(str(k), {"n_cal": n, "M_test": M, "alpha": {}})
                        for alpha in pr["alpha_grid"]:
                            entry = {}
                            for regime in pr["regimes"]:
                                kw = {"regime": regime, "delta": delta_cfg if regime != "marginal" else None, "M": M if regime == "finite" else None}
                                stats_keys = [f"A_k{r}" for r in r_grid] + ["B_top", "B_all", "traj_mu"]
                                entry[regime] = {s: calibrate_threshold([row[s] for row in rows], alpha, **kw).as_dict() for s in stats_keys}
                                entry[regime]["H_k"] = calibrate_uniform_prefix([row["H_k"] for row in rows], alpha, N=sd.N, **kw).as_dict()
                                entry[regime]["e_H"] = calibrate_adaptive_prefix([row["H_k"] for row in rows], [row["h_pred"] for row in rows], alpha, **kw).as_dict()
                            lam["k"][str(k)]["alpha"][str(alpha)] = entry
                    log(f"{teacher}/{domain}/{split} k={k}: {len(rows)} queries; mean A_k1={np.mean([r['A_k1'] for r in rows]):.3f} mean H_k={np.mean([r['H_k'] for r in rows]):.1f}")
            if is_main:
                write_json(out_dir / "lambdas.json", lam)
    return 0


if __name__ == "__main__":
    sys.exit(main())
