"""Score coverage with a scale head fitted to its own target, and the risk-calls curve (amendment A10 (ii), (iii))
-> runs/own_target/<teacher>/<domain>.json, runs/own_target/aggregate.json

(ii) B_all calibrates the residuals of every candidate, while the deployed scale head is fitted to the winners'
     residuals.  A head with the architecture, query features z(q), training data (train+dev of the main domains),
     optimiser settings and seed of the deployed one is fitted to log(eps + RMS over all candidates of s_j - mu_j),
     with the frozen mean head.  The same code first refits the deployed (winners') head and must reproduce its
     stored predictions, so the two heads differ in the target alone.  B_all is calibrated with the new head on the
     A8 calibration set and replayed on the A8 test set (deployed order, 16/8).
(iii) For alpha in {0.01, 0.02, 0.05, 0.10, 0.20}: finite-population thresholds on the A8 calibration set (delta_cfg)
     for RSC, B_top, B_all with its own head, the trajectory threshold and the adaptive and uniform prefixes; risk and
     mean calls on the A8 test set, deployed order.
No other head, threshold or method changes.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank.assemble import bundles_from_split, load_split, query_features_for_split  # noqa: E402
from rscrank.baselines import adaptive_prefix_budget, calibrate_adaptive_prefix, calibrate_uniform_prefix  # noqa: E402
from rscrank.calibrate import family_delta  # noqa: E402
from rscrank.data import cheap_model_for, features_dir_for, load_pool_qrels  # noqa: E402
from rscrank.heads import EPS, MeanHead, ScaleHead  # noqa: E402
from rscrank.io import read_json, write_json  # noqa: E402
from rscrank.policies import make_policy  # noqa: E402
from rscrank.replay import MethodSpec, query_statistics, replay_query  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


s06 = _load("s06", "06_train_heads.py")
rep08 = _load("rep08", "08_replay_eval.py")
s18 = _load("s18", "18_order_free_baselines.py")

ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20)


def all_candidate_targets(parts, offs, mu, q_index, k):
    """rho_all(q) = eps + RMS over every candidate of s_j - mu_j, aligned with s06.query_targets."""
    rho, valid = [], []
    for sd, off in zip(parts, offs):
        for i in range(sd.Q):
            sel = np.where(q_index == i + off)[0]
            if len(sel) < k + 1:
                rho.append(1.0)
                valid.append(False)
                continue
            e = sd.t["score"][i, : len(sel)].astype(float) - mu[sel]
            rho.append(EPS + float(np.sqrt(np.mean(e * e))))
            valid.append(True)
    return np.asarray(rho), np.asarray(valid)


def fit_scale(cfg, Ztr, ytr, Zdv, ydv):
    hc = cfg["heads"]
    sh = ScaleHead(Ztr.shape[1], hidden=32, seed=cfg["seed"])
    info = sh.fit(Ztr, ytr, Zdv, ydv, lr=3e-3, epochs=max(hc["epochs"], 10), patience=hc["patience"], batch_size=64, seed=cfg["seed"])
    return sh, info


def train_heads(cfg, teacher, k):
    hd = P(cfg, "models") / "heads" / teacher
    meta = read_json(hd / "meta.json")
    mh = MeanHead.load(hd / "mu.npz", meta["kappa"])
    deployed = ScaleHead.load(hd / "sigma.npz")
    train_parts, dev_parts = s06.load_pool(cfg, teacher, "train"), s06.load_pool(cfg, teacher, "dev")
    Xtr, _, qtr, offs_tr = s06.stacked(train_parts, k)
    Xdv, _, qdv, offs_dv = s06.stacked(dev_parts, k)
    mu_tr, mu_dv = mh.predict(Xtr), mh.predict(Xdv)
    Ztr, rho_tr, _, vtr = s06.query_targets(train_parts, offs_tr, mu_tr, qtr, k)
    Zdv, rho_dv, _, vdv = s06.query_targets(dev_parts, offs_dv, mu_dv, qdv, k)
    # the same code path must reproduce the deployed head before it is trusted with another target
    refit, _ = fit_scale(cfg, Ztr[vtr], rho_tr[vtr], Zdv[vdv], rho_dv[vdv])
    repro = float(np.max(np.abs(refit.predict(Zdv[vdv]) - deployed.predict(Zdv[vdv]))))
    if repro > 1e-9:
        raise SystemExit(f"{teacher}: refitting the deployed scale head does not reproduce it (max diff {repro:.3g})")
    rall_tr, wtr = all_candidate_targets(train_parts, offs_tr, mu_tr, qtr, k)
    rall_dv, wdv = all_candidate_targets(dev_parts, offs_dv, mu_dv, qdv, k)
    assert np.array_equal(wtr, vtr) and np.array_equal(wdv, vdv)
    own, info = fit_scale(cfg, Ztr[vtr], rall_tr[vtr], Zdv[vdv], rall_dv[vdv])
    report = {"teacher": teacher, "use_sigma_head_deployed": bool(meta["use_sigma_head"]), "refit_max_abs_diff": repro,
              "own_head_best_dev_loss": info["best_dev_loss"], "own_head_epochs_run": info["epochs_run"],
              "train_queries": int(vtr.sum()), "dev_queries": int(vdv.sum()),
              "median_rho_winners_dev": float(np.median(rho_dv[vdv])), "median_rho_all_dev": float(np.median(rall_dv[vdv]))}
    return deployed, own, bool(meta["use_sigma_head"]), report


def bundles_with_scale(cfg, teacher, domain, split, k, head, use_head):
    fdir = features_dir_for(P(cfg, "features"), cheap_model_for(cfg, teacher), teacher)
    sd = load_split(fdir, P(cfg, "teacher") / teacher, domain, split)
    fz = np.load(P(cfg, "frozen") / teacher / domain / f"{split}.npz")
    sig = np.ones(sd.Q)
    if use_head:
        Z = query_features_for_split(sd, fz["mu"], fz["q_index"], k)
        ok = ~np.isnan(Z).any(1)
        sig[ok] = head.predict(Z[ok])
    qrels = load_pool_qrels(cfg, domain, P(cfg, "data_raw"))
    return bundles_from_split(sd, fz["mu"], fz["q_index"], sig, k, qrels, fz["h_q"]), sig, fz


def main() -> int:
    ap = base_parser("own-target scale head for B_all and the risk-calls curve")
    ap.add_argument("--teacher", default=None)
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k, a0 = int(pr["k"]), float(pr["alpha_main"])
    primary = pr.get("primary_regime", "finite")
    delta_cfg = family_delta(pr["delta_family"], pr["n_main_configs"])
    bf, br, seed = pr["batch_first"], pr["batch_rest"], int(cfg["seed"])
    a8 = ROOT / cfg["paths"]["runs"] / "a8_split"
    out_root = ROOT / cfg["paths"]["runs"] / "own_target"
    agg = read_json(out_root / "aggregate.json") if (out_root / "aggregate.json").exists() else {}
    for teacher in ([args.teacher] if args.teacher else teachers(cfg)):
        deployed, own, use_head, report = train_heads(cfg, teacher, k)
        log(f"{teacher}: deployed head reproduced (max diff {report['refit_max_abs_diff']:.1e}); own-target head dev loss {report['own_head_best_dev_loss']:.4f}")
        for d in main_domains(cfg):
            lams = read_json(ROOT / cfg["paths"]["runs"] / "calib" / teacher / d / "lambdas.json")
            cal = rep08.load_bundles(cfg, teacher, d, "cal", k, lams=lams)
            test = rep08.load_bundles(cfg, teacher, d, "test", k, lams=lams)
            cal, test = s18.frozen_family_resplit(cal, test, d, a8 / "splits")
            M, N = len(test), test[0].N
            # the deployed head applied through this code path gives the frozen scales exactly
            for split in ("cal", "test"):
                _, sig_dep, fz = bundles_with_scale(cfg, teacher, d, split, k, deployed, use_head)
                if not np.array_equal(sig_dep, fz["sigma_q"]):
                    raise SystemExit(f"{teacher}/{d}/{split}: deployed scales are not reproduced")
            own_b = {}
            for split in ("cal", "test"):
                bs, _, _ = bundles_with_scale(cfg, teacher, d, split, k, own, True)
                own_b[split] = bs
            own_cal, own_test = s18.frozen_family_resplit(own_b["cal"], own_b["test"], d, a8 / "splits")
            one_b = {}
            for split in ("cal", "test"):
                bs, _, _ = bundles_with_scale(cfg, teacher, d, split, k, None, False)
                one_b[split] = bs
            one_cal, one_test = s18.frozen_family_resplit(one_b["cal"], one_b["test"], d, a8 / "splits")

            stat = {"dep": [query_statistics(b, k) for b in cal], "own": [query_statistics(b, k) for b in own_cal], "one": [query_statistics(b, k) for b in one_cal]}
            traj_cal = [s18.traj_stat(b, k, make_policy("mu", {n: b.feature_map(n) for n in b.features}, seed), bf, br) for b in cal]
            H_cal = [r["H_k"] for r in stat["dep"]]
            h_cal = [float(b.features["h_pred"][0]) for b in cal]
            truth = [set(b.true_topk_ids(k)) for b in test]

            def replay(bundles, lam_):
                recs = [replay_query(b, MethodSpec("o", "guardrail", lam=lam_, policy="mu", policy_seed=seed, batch_first=bf, batch_rest=br), k) for b in bundles]
                return {"lambda": lam_, "risk": float(np.mean([r["mismatch"] for r in recs])), "calls": float(np.mean([r["n_calls"] for r in recs]))}

            def calib(vals, alpha):
                return s18.calib(vals, alpha, primary, delta_cfg, M)

            saved = read_json(a8 / teacher / f"{d}.json")
            if abs(calib([r["A_k1"] for r in stat["dep"]], a0) - float(saved["thresholds"]["lambda_rsc"])) > 1e-12:
                raise SystemExit(f"{teacher}/{d}: RSC threshold does not match the stored post-freeze value")
            out = {"teacher": teacher, "domain": d, "k": k, "delta_cfg": delta_cfg, "n_cal": len(cal), "n_test": M, "head_report": report,
                   "ball_variants": {"winners_head": replay(test, calib([r["B_all"] for r in stat["dep"]], a0)),
                                     "own_head": replay(own_test, calib([r["B_all"] for r in stat["own"]], a0)),
                                     "sigma_one": replay(one_test, calib([r["B_all"] for r in stat["one"]], a0))},
                   "curve": {}}
            for alpha in ALPHAS:
                row = {"rsc": replay(test, calib([r["A_k1"] for r in stat["dep"]], alpha)),
                       "btop": replay(test, calib([r["B_top"] for r in stat["dep"]], alpha)),
                       "ball_own": replay(own_test, calib([r["B_all"] for r in stat["own"]], alpha)),
                       "traj": replay(test, calib(traj_cal, alpha))}
                m_uni = int(min(calibrate_uniform_prefix(H_cal, alpha, primary, delta=delta_cfg, M=M, N=N).threshold, N))
                t_H = calibrate_adaptive_prefix(H_cal, h_cal, alpha, primary, delta=delta_cfg, M=M).threshold
                m_ad = [adaptive_prefix_budget(float(b.features["h_pred"][0]), t_H, k, b.N) for b in test]
                row["adaptive_once"] = {"risk": float(np.mean([not t.issubset(s18.mu_prefix(b, m)) for t, b, m in zip(truth, test, m_ad)])), "calls": float(np.mean(m_ad))}
                row["uniform_once"] = {"risk": float(np.mean([not t.issubset(s18.mu_prefix(b, m_uni)) for t, b in zip(truth, test)])), "calls": float(m_uni)}
                out["curve"][f"{alpha:g}"] = row
            c5 = out["curve"]["0.05"]
            if abs(c5["rsc"]["calls"] - float(saved["calls_under_mu"]["rsc"])) > 1e-9 or abs(c5["traj"]["calls"] - float(saved["calls_under_mu"]["traj_mu"])) > 1e-9:
                raise SystemExit(f"{teacher}/{d}: alpha = 0.05 does not reproduce the stored deployed-order calls")
            path = out_root / teacher / f"{d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, out)
            agg[f"{teacher}/{d}"] = {"ball_variants": out["ball_variants"], "rsc_alpha05": c5["rsc"], "btop_alpha05": c5["btop"]}
            write_json(out_root / "aggregate.json", agg)
            v = out["ball_variants"]
            log(f"{teacher}/{d}: B_all calls winners-head {v['winners_head']['calls']:.1f} / own head {v['own_head']['calls']:.1f} / sigma=1 {v['sigma_one']['calls']:.1f}; "
                f"RSC {c5['rsc']['calls']:.1f}; risks {v['own_head']['risk']:.4f} (own)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
