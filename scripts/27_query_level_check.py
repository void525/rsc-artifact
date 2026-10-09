"""Recompute the post-freeze thresholds, witness risks and committed-set results from the shipped query-level
statistics, without teacher scores or a GPU, and compare them with the shipped aggregates.

Inputs (all in the artifact):
  runs/calib/<teacher>/<domain>/stats_{cal,test}_k10.jsonl   one line per query of the fixed pool: A_k1..A_k3, B_top,
                                                               B_all, H_k, the deployed-trajectory value traj_mu, h_pred
  runs/a8_split/splits/<domain>.json                          the post-freeze calibration/test partition (query ids)
  runs/a8_split/<teacher>/<domain>.json, runs/a8_split/grid_frontier.json   the aggregates to compare with

  python scripts/27_query_level_check.py        exit code 0 iff every recomputed value equals the shipped one
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, base_parser, load_config, main_domains, teachers  # noqa: E402

from rscrank.baselines import adaptive_prefix_budget, calibrate_adaptive_prefix, calibrate_uniform_prefix  # noqa: E402
from rscrank.calibrate import calibrate_threshold, family_delta  # noqa: E402
from rscrank.io import read_json, read_jsonl  # noqa: E402


def main() -> int:
    args = base_parser("query-level reproduction of the post-freeze aggregates").parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k, alpha = int(pr["k"]), float(pr["alpha_main"])
    regime = pr.get("primary_regime", "finite")
    delta = family_delta(pr["delta_family"], pr["n_main_configs"])
    runs = ROOT / cfg["paths"]["runs"]
    grid = read_json(runs / "a8_split" / "grid_frontier.json")
    bad = 0
    print(f"{'configuration':<20} {'check':<34} {'recomputed':>12} {'shipped':>12}")
    for t in teachers(cfg):
        for d in main_domains(cfg):
            rows = {}
            for s in ("cal", "test"):
                for r in read_jsonl(runs / "calib" / t / d / f"stats_{s}_k{k}.jsonl"):
                    rows[str(r["qid"])] = r
            split = read_json(runs / "a8_split" / "splits" / f"{d}.json")
            cal, test = [rows[q] for q in split["cal"]], [rows[q] for q in split["test"]]
            M, N = len(test), int(test[0]["N"])
            agg = read_json(runs / "a8_split" / t / f"{d}.json")
            th, head = agg["thresholds"], agg["headline"]

            def calib(key):
                return calibrate_threshold([r[key] for r in cal], alpha, regime, delta=delta, M=M).threshold

            lam_rsc, lam_mu = calib("A_k1"), calib("traj_mu")
            m_uni = int(min(calibrate_uniform_prefix([r["H_k"] for r in cal], alpha, regime, delta=delta, M=M, N=N).threshold, N))
            t_H = calibrate_adaptive_prefix([r["H_k"] for r in cal], [r["h_pred"] for r in cal], alpha, regime, delta=delta, M=M).threshold
            m_ad = [adaptive_prefix_budget(r["h_pred"], t_H, k, N) for r in test]
            A_test = np.array([r["A_k1"] for r in test])
            wit = grid[f"{t}/{d}"]["all_subset_witness_risk"]
            checks = [
                ("lambda RSC", lam_rsc, th["lambda_rsc"]),
                ("lambda one trajectory", lam_mu, th["lambda_mu"]),
                ("uniform prefix m", m_uni, th["m_uniform"]),
                ("adaptive prefix t_H", t_H, th["t_H"]),
                ("witness risk at lambda RSC", float(np.mean(A_test > lam_rsc)), wit["rsc"]),
                ("witness risk at lambda traj.", float(np.mean(A_test > lam_mu)), wit["traj_mu"]),
                ("witness risk at lambda env.", float(np.mean(A_test > agg["envelope"]["all13"]["lambda"])), wit["env13"]),
                ("adaptive prefix risk", float(np.mean([r["H_k"] > m for r, m in zip(test, m_ad)])), head["adaptive_once"]["risk"]),
                ("adaptive prefix calls", float(np.mean(m_ad)), head["adaptive_once"]["calls"]),
                ("uniform prefix risk", float(np.mean([r["H_k"] > m_uni for r in test])), head["uniform_once"]["risk"]),
            ]
            for name, got, want in checks:
                ok = (got == want) if isinstance(want, (int, np.integer)) or (isinstance(want, float) and math.isinf(want)) else abs(float(got) - float(want)) <= 1e-12
                bad += not ok
                print(f"{t + '/' + d:<20} {name:<34} {float(got):>12.6f} {float(want):>12.6f}{'' if ok else '   MISMATCH'}")
    print("every recomputed value equals the shipped one" if bad == 0 else f"{bad} mismatches")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
