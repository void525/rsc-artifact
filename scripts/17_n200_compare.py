"""N18 summary: N = 200 vs N = 100 on the same pre-specified test subset -> runs/N200/compare.json

Reads the N = 200 replay (configs/n200.yaml; runs/N200/replay/<teacher>/<domain>/E1_main.jsonl, summary.json) and the
main N = 100 replay restricted to the same test queries (data/splits/extended_subset.json), and reports per teacher and
domain: mean calls, calls / N, exact-mismatch risk and early-stop fraction of RSC, the adaptive prefix and the uniform
prefix at alpha = 0.05 (primary regime), plus the E9 anatomy means at N = 200 when present.  Registers macros
<Domain><Teacher>NTwoHundred... via the provenance registry (12_paper_numbers.py picks them up from compare.json).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, base_parser, load_config, log  # noqa: E402

from rscrank.io import read_json, read_jsonl, write_json  # noqa: E402


def by_method(path):
    d = defaultdict(dict)
    for r in read_jsonl(path):
        d[r["method"]][r["qid"]] = r
    return d


def summarize(recs, N):
    calls = np.array([r["n_calls"] for r in recs.values()], float)
    mis = np.array([r["mismatch"] for r in recs.values()], float)
    early = np.array([r["early_stopped"] for r in recs.values()], float)
    return {"n": len(recs), "calls_mean": float(calls.mean()), "calls_frac_of_N": float(calls.mean() / N), "risk": float(mis.mean()), "early_fraction": float(early.mean())}


def main() -> int:
    ap = base_parser("N=200 comparison")
    ap.add_argument("--n200-runs", default="runs/N200", help="runs dir of the N = 200 pipeline (runs/N200 = frozen heads with clamped length feature; runs/N200h = heads retrained at N = 200)")
    ap.add_argument("--out", default=None, help="output json (default <n200-runs>/compare.json)")
    args = ap.parse_args()
    N200_RUNS = ROOT / args.n200_runs
    cfg = load_config(args)  # the MAIN config (N = 100 paths)
    subset = read_json(ROOT / "data" / "splits" / "extended_subset.json")["subset"]
    a0 = cfg["protocol"]["alpha_main"]
    primary = cfg["protocol"].get("primary_regime", "finite")
    tag = f"a{a0}_{primary}"
    out = {"alpha": a0, "regime": primary, "configs": {}}
    for t in cfg["teachers"]:
        for d in ("nq", "hotpotqa"):
            f200 = N200_RUNS / "replay" / t / d / "E1_main.jsonl"
            f100 = ROOT / cfg["paths"]["runs"] / "replay" / t / d / "E1_main.jsonl"
            if not f200.exists() or not f100.exists():
                log(f"{t}/{d}: missing {f200 if not f200.exists() else f100}; skipping")
                continue
            E200, E100 = by_method(f200), by_method(f100)
            qids = set(subset[d]["test"])
            res = {"n_subset_test": len(qids)}
            for name, key in (("rsc", f"rsc_{tag}"), ("adaptive_prefix", f"prefix_adaptive_{tag}"), ("uniform_prefix", f"prefix_uniform_{tag}"), ("btop", f"btop_{tag}"), ("traj", f"traj_{tag}")):
                r200 = {q: r for q, r in E200.get(key, {}).items()}
                r100 = {q: r for q, r in E100.get(key, {}).items() if q in qids}
                common = set(r200) & set(r100)
                if not common:
                    continue
                res[name] = {"N200": summarize({q: r200[q] for q in common}, 200), "N100": summarize({q: r100[q] for q in common}, 100), "n_common": len(common)}
            s200 = N200_RUNS / "replay" / t / d / "summary.json"
            if s200.exists():
                S = read_json(s200)["experiments"]
                if "E9" in S:
                    res["E9_N200"] = {k: S["E9"][k] for k in ("cert_core_mean", "band_mean", "overshoot_mean", "calls_mean", "covered_frac", "anatomy_exact_on_covered") if k in S["E9"]}
                if "E3" in S:
                    res["E3_N200"] = {m: {"calls_mean": v["calls_mean"], "rounds_mean": v["rounds_mean"], "risk": v["risk"]} for m, v in S["E3"].items() if isinstance(v, dict) and "calls_mean" in v}
            lam200 = N200_RUNS / "calib" / t / d / "lambdas.json"
            if lam200.exists():
                L = read_json(lam200)["k"][str(cfg["protocol"]["k"])]["alpha"][str(a0)][primary]["A_k1"]
                res["lambda_N200"] = L.get("threshold")
                res["n_cal_N200"] = read_json(lam200)["k"][str(cfg["protocol"]["k"])].get("n_cal")
            out["configs"][f"{t}/{d}"] = res
            r = res.get("rsc", {})
            if r:
                log(f"{t}/{d}: RSC calls N=200 {r['N200']['calls_mean']:.1f} ({100 * r['N200']['calls_frac_of_N']:.0f} % of N) risk {r['N200']['risk']:.4f} | N=100 same queries {r['N100']['calls_mean']:.1f} ({100 * r['N100']['calls_frac_of_N']:.0f} %) risk {r['N100']['risk']:.4f}")
    out["n200_runs"] = str(N200_RUNS.relative_to(ROOT))
    write_json(Path(args.out) if args.out else (N200_RUNS / "compare.json"), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
