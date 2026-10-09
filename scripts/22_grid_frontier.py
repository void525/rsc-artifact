"""Critical value of every policy on the deployed checkpoint grid -> runs/a8_split/grid_frontier.json

On the post-freeze split (seed 20261006), for each teacher and main domain:
  * the last-winner identity A_{k,1}(q) = (s_(k+1) - mu_[H_k]) / sigma(q) (shared scale), checked query by query;
  * the grid critical value A^G(q) = max{lambda*(V): V unsafe, |V| in G} for the deployed 16/8 grid G (Proposition
    "Checkpoint grids"), which protects every order and every score-dependent choice that keeps the 16/8 batch sizes;
  * its finite-population threshold (same regime, alpha and delta_cfg as RSC), the deployed-order calls and risk, the
    risk of the ten tested orders at that threshold, and the grid witness risk at the trajectory, envelope, grid and
    RSC thresholds;
  * the all-subset witness risk, the fraction of test queries with A_{k,1}(q) > lambda, at the trajectory, envelope
    and RSC thresholds (Table 1 of the paper).
Thresholds of the other methods are read from runs/a8_split/<teacher>/<domain>.json; nothing there is changed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank.calibrate import calibrate_threshold, family_delta  # noqa: E402
from rscrank.io import read_json, write_json  # noqa: E402
from rscrank.replay import MethodSpec, query_statistics, replay_query  # noqa: E402
from rscrank.stats import paired_bootstrap_diff  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


rep08 = _load("rep08", "08_replay_eval.py")
s18 = _load("s18", "18_order_free_baselines.py")


def main() -> int:
    ap = base_parser("checkpoint-grid critical value on the post-freeze split")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--skip-orders", action="store_true", help="do not replay the ten orders at the grid threshold")
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k0, a0 = pr["k"], pr["alpha_main"]
    primary = pr.get("primary_regime", "finite")
    delta_cfg = family_delta(pr["delta_family"], pr["n_main_configs"])
    bf, br = pr["batch_first"], pr["batch_rest"]
    seed = int(cfg["seed"])
    orders = list(pr["policies"])
    out_path = ROOT / cfg["paths"]["runs"] / "a8_split" / "grid_frontier.json"
    results = read_json(out_path) if out_path.exists() else {}
    for teacher in ([args.teacher] if args.teacher else teachers(cfg)):
        for domain in ([args.domain] if args.domain else main_domains(cfg)):
            lams = read_json(ROOT / cfg["paths"]["runs"] / "calib" / teacher / domain / "lambdas.json")
            cal = rep08.load_bundles(cfg, teacher, domain, "cal", k0, lams=lams)
            test = rep08.load_bundles(cfg, teacher, domain, "test", k0, lams=lams)
            cal, test = s18.frozen_family_resplit(cal, test, domain, ROOT / cfg["paths"]["runs"] / "a8_split" / "splits")
            saved = read_json(ROOT / cfg["paths"]["runs"] / "a8_split" / teacher / f"{domain}.json")
            M, N = len(test), test[0].N
            grid = core.checkpoints_from_schedule(N, k0, core.default_batch_schedule(k0, bf, br))
            c_star = max(c for c in grid if k0 <= c < N)

            def stats_of(bundles):
                A, G, H, ident = [], [], [], []
                for b in bundles:
                    if not np.all(b.sigma == b.sigma[0]):
                        raise SystemExit(f"{teacher}/{domain}/{b.qid}: sigma is not query-shared")
                    a = core.critical_value(b.s, b.mu, b.sigma, k0, 1, b.ids)
                    h = core.H_k(b.s, b.mu, k0, b.ids)
                    order = core.teacher_order(b.s, b.ids)
                    mu_sorted = b.mu[core.teacher_order(b.mu, b.ids)]
                    ident.append(abs(a - (b.s[order[k0]] - mu_sorted[h - 1]) / b.sigma[0]))
                    A.append(a)
                    G.append(core.grid_critical_value(b.s, b.mu, b.sigma, k0, grid, b.ids))
                    H.append(h)
                return np.array(A, float), np.array(G, float), np.array(H, int), float(np.max(ident))

            A_cal, G_cal, H_cal, id_cal = stats_of(cal)
            A_te, G_te, H_te, id_te = stats_of(test)
            assert np.all(G_cal <= A_cal) and np.all(G_te <= A_te)

            def calib(v):
                return calibrate_threshold(list(v), a0, primary, delta=delta_cfg, M=M).threshold

            lam_rsc, lam_grid = calib(A_cal), calib(G_cal)
            th = saved["thresholds"]
            if abs(lam_rsc - float(th["lambda_rsc"])) > 1e-12:
                raise SystemExit(f"{teacher}/{domain}: RSC threshold does not match the stored post-freeze value")
            lam_mu, lam_env = float(th["lambda_mu"]), float(saved["envelope"]["all13"]["lambda"])

            def replay_mu(lam_, pol="mu"):
                return [replay_query(b, MethodSpec("g", "guardrail", lam=lam_, policy=pol, policy_seed=seed, batch_first=bf, batch_rest=br), k0) for b in test]

            rec_grid, rec_rsc = replay_mu(lam_grid), replay_mu(lam_rsc)
            calls_grid = np.array([r["n_calls"] for r in rec_grid], float)
            calls_rsc = np.array([r["n_calls"] for r in rec_rsc], float)
            d = paired_bootstrap_diff(calls_rsc, calls_grid, n_boot=2000, seed=seed)
            per_order = {}
            if not args.skip_orders:
                for pol in orders:
                    rr = replay_mu(lam_grid, pol)
                    per_order[pol] = {"risk": float(np.mean([r["mismatch"] for r in rr])), "calls": float(np.mean([r["n_calls"] for r in rr]))}
            rec = {
                "teacher": teacher, "domain": domain, "k": k0, "alpha": a0, "regime": primary, "delta_cfg": delta_cfg,
                "n_cal": len(cal), "n_test": M, "grid": grid, "last_checkpoint_below_N": c_star,
                "identity_max_abs_diff": max(id_cal, id_te),
                "cal_frac_grid_below_A": float(np.mean(G_cal < A_cal)), "test_frac_grid_below_A": float(np.mean(G_te < A_te)),
                "cal_frac_H_above_cstar": float(np.mean(H_cal > c_star)), "test_frac_H_above_cstar": float(np.mean(H_te > c_star)),
                "grid_below_A_iff_H_above_cstar": bool(np.array_equal(G_cal < A_cal, H_cal > c_star) and np.array_equal(G_te < A_te, H_te > c_star)),
                "lambda": {"traj_mu": lam_mu, "env13": lam_env, "grid": lam_grid, "rsc": lam_rsc},
                "grid_witness_risk": {name: float(np.mean(G_te > v)) for name, v in (("traj_mu", lam_mu), ("env13", lam_env), ("grid", lam_grid), ("rsc", lam_rsc))},
                "all_subset_witness_risk_at_grid": float(np.mean(A_te > lam_grid)),
                "all_subset_witness_risk": {name: float(np.mean(A_te > v)) for name, v in (("traj_mu", lam_mu), ("env13", lam_env), ("rsc", lam_rsc))},
                "deployed": {"grid": {"calls": float(calls_grid.mean()), "risk": float(np.mean([r["mismatch"] for r in rec_grid]))},
                             "rsc": {"calls": float(calls_rsc.mean()), "risk": float(np.mean([r["mismatch"] for r in rec_rsc]))},
                             "rsc_minus_grid_calls": {"mean": d["mean"], "lcb": d["lcb"], "ucb": d["ucb"]}},
                "orders_at_grid_threshold": per_order,
            }
            results[f"{teacher}/{domain}"] = rec
            write_json(out_path, results)
            log(f"{teacher}/{domain}: c*={c_star}, grid<A on test {rec['test_frac_grid_below_A']:.4f} (H>c* {rec['test_frac_H_above_cstar']:.4f}), "
                f"lambda grid {lam_grid:.4f} vs rsc {lam_rsc:.4f}, calls grid {calls_grid.mean():.2f} vs rsc {calls_rsc.mean():.2f}, "
                f"grid witness at env {rec['grid_witness_risk']['env13']:.4f}, identity {rec['identity_max_abs_diff']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
