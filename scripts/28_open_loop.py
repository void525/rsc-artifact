"""Open-loop serving at fixed arrival rates (amendment A11) -> runs/open_loop/<teacher>/<domain>.json

The closed-loop harness of scripts/09b_throughput.py admits a new query whenever one finishes, so it measures cost per
query but not what a deployment sees at a given traffic level.  Here queries arrive by a Poisson process at a fixed
offered rate and wait in a first-come first-served queue; the server is the one of 09b: at most C queries in flight,
admission between rounds, and every round scores the pending pairs of all queries in flight in one canonical call, so
every score equals the cached teacher score.  The cheap stage runs at admission inside the serving loop.

Per domain: 300 queries drawn once from the post-freeze (A8) test set with the A8 thresholds; the capacity c_full is
the closed-loop throughput of the full teacher at C = 32 on those queries, measured first; offered rates are
rho * c_full.  One arrival sequence per (domain, pass) is shared by all methods.

  python scripts/28_open_loop.py --teacher qwen3-4b --domains nq,hotpotqa,msmarco
  python scripts/28_open_loop.py --teacher qwen3-4b --domains msmarco --pass-index 2
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log  # noqa: E402

from rscrank.data import cheap_model_for, distilled_model_dir  # noqa: E402
from rscrank.heads import BudgetHead, MeanHead, ScaleHead  # noqa: E402
from rscrank.io import ensure_dir, read_json, write_json  # noqa: E402
from rscrank.models import get_cheap, get_teacher  # noqa: E402
from rscrank.store import load_candidates, load_docstore  # noqa: E402

_spec = importlib.util.spec_from_file_location("tp09b", Path(__file__).resolve().parent / "09b_throughput.py")
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)  # type: ignore[union-attr]

SEED = 20261009  # declared in amendment A11 before any open-loop run
DOMAIN_CODE = {"nq": 1, "hotpotqa": 2, "msmarco": 3}
METHODS = ("full", "rsc", "rsc_blockers", "prefix_adaptive")
RHOS = (0.6, 0.9, 1.2, 1.5)
C_MAX = 32
N_QUERIES, N_WARM = 300, 5


def serve_open(make_job, arrivals, teacher, c_max):
    """Open-loop server: query i arrives at arrivals[i] seconds; returns the finished jobs and the token counters."""
    before = dict(getattr(teacher, "counters", {}) or {})
    n = len(arrivals)
    queue, active, jobs = [], [], [None] * n
    nxt = 0
    tp.sync()
    t0 = time.perf_counter()
    while nxt < n or queue or active:
        now = time.perf_counter() - t0
        while nxt < n and arrivals[nxt] <= now:
            queue.append(nxt)
            nxt += 1
        if not queue and not active:
            time.sleep(max(0.0, arrivals[nxt] - now))
            continue
        while queue and len(active) < c_max:
            i = queue.pop(0)
            t_admit = time.perf_counter() - t0
            j = make_job(i)  # runs the cheap stage on the GPU, inside the serving loop
            j.t_arr, j.t_admit = float(arrivals[i]), t_admit
            jobs[i] = j
            active.append(j)
        pairs, owners = [], []
        for j in active:
            for d in j.pending:
                pairs.append((j.query, j.docs[d]))
                owners.append((j, d))
        if pairs:
            sc = teacher.score_pairs(pairs)
            tp.sync()
            by_job = {}
            for (j, d), s in zip(owners, sc):
                by_job.setdefault(id(j), (j, [], []))
                by_job[id(j)][1].append(d)
                by_job[id(j)][2].append(float(s))
            for j, ids, scs in by_job.values():
                j.observe(ids, scs)
        now = time.perf_counter() - t0
        still = []
        for j in active:
            if j.done:
                j.t_done = now
            else:
                still.append(j)
        active = still
    c = getattr(teacher, "counters", {}) or {}
    fw = {k: c.get(k, 0) - before.get(k, 0) for k in ("forward_passes", "rows", "dummy_rows", "real_tokens", "padded_tokens")}
    return jobs, fw


def main() -> int:
    ap = base_parser("open-loop serving at fixed arrival rates")
    ap.add_argument("--teacher", default="qwen3-4b")
    ap.add_argument("--pass-index", type=int, default=1)
    ap.add_argument("--rhos", default=",".join(str(r) for r in RHOS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n-queries", type=int, default=N_QUERIES, help="debug only; the registered run uses 300")
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k, N = int(pr["k"]), int(cfg["candidates"]["N"])
    bf, br = pr["batch_first"], pr["batch_rest"]
    rhos = [float(x) for x in args.rhos.split(",") if x]
    methods = [m for m in args.methods.split(",") if m]
    tname = args.teacher
    teacher = get_teacher(tname, dict(cfg["teachers"].get(tname, {}) or {}))
    ccfg = {kk: v for kk, v in cfg["cheap"].items() if kk not in ("distill", "model_by_teacher")}
    cheap_name = cheap_model_for(cfg, tname)
    if cheap_name == "minilm-distilled":
        cheap = get_cheap("minilm-distilled", {**ccfg, "path": str(distilled_model_dir(P(cfg, "models"), tname)), "name": f"minilm-distilled:{tname}"})
    else:
        cheap = get_cheap(cheap_name, ccfg)
    hd = P(cfg, "models") / "heads" / tname
    meta = read_json(hd / "meta.json")
    heads = {"mu": MeanHead.load(hd / "mu.npz", meta["kappa"]), "sigma": ScaleHead.load(hd / "sigma.npz"), "budget": BudgetHead.load(hd / "budget.npz")}
    runs = ROOT / cfg["paths"]["runs"]
    doms = [d for d in (args.domains.split(",") if args.domains else cfg["domains"]["main"]) if d]
    for domain in doms:
        th = read_json(runs / "a8_split" / tname / f"{domain}.json")["thresholds"]
        lam, tH, m_uni = float(th["lambda_rsc"]), float(th["t_H"]), int(th["m_uniform"])
        test_q = read_json(runs / "a8_split" / "splits" / f"{domain}.json")["test"]
        cands = {}
        for s in ("cal", "test"):
            for r in load_candidates(P(cfg, "candidates") / domain / f"{s}.jsonl"):
                cands[str(r["qid"])] = r
        docstore = load_docstore(P(cfg, "candidates") / domain / "docstore.jsonl")
        cache_top = {}
        for s in ("cal", "test"):
            cache_top.update(tp.cached_topk(P(cfg, "teacher") / tname / domain / f"{s}.npz", k) or {})
        rng = np.random.default_rng(np.random.SeedSequence([SEED, DOMAIN_CODE[domain]]))
        pick = rng.choice(len(test_q), size=N_WARM + N_QUERIES, replace=False)
        warm = [cands[test_q[i]] for i in pick[:N_WARM]]
        measure = [cands[test_q[i]] for i in pick[N_WARM:]][: args.n_queries]

        def job(method, cand):
            b0, b1 = (32, 16) if method == "rsc_blockers" else (bf, br)
            return tp.QueryJob(method, cand, docstore, cheap, heads, meta, k, lam, tH, m_uni, b0, b1, N, "mu", cfg["seed"])

        out_path = ensure_dir(runs / "open_loop" / tname) / (f"{domain}.json" if args.pass_index == 1 else f"{domain}_pass{args.pass_index}.json")
        result = read_json(out_path) if out_path.exists() else {}
        if result.get("n_queries") != len(measure) or result.get("pass") != args.pass_index:
            result = {"teacher": tname, "domain": domain, "pass": args.pass_index, "n_queries": len(measure), "c_max": C_MAX,
                      "thresholds": {"lambda_rsc": lam, "t_H": tH}, "split": "post-freeze (A8) test set", "rhos": rhos, "cells": {}}
        if "c_full" not in result:
            tp.serve([job("full", c) for c in warm], teacher, C_MAX, dict(getattr(teacher, "counters", {}) or {}))
            jobs = [job("full", c) for c in measure]
            wall, _ = tp.serve(jobs, teacher, C_MAX, dict(getattr(teacher, "counters", {}) or {}))
            result["c_full"] = len(jobs) / wall
            write_json(out_path, result)
        c_full = float(result["c_full"])
        log(f"{tname}/{domain} pass {args.pass_index}: closed-loop full-teacher capacity {c_full:.3f} q/s at C = {C_MAX}")
        arr_rng = np.random.default_rng(np.random.SeedSequence([SEED, DOMAIN_CODE[domain], args.pass_index]))
        unit = np.cumsum(arr_rng.exponential(1.0, size=len(measure)))  # unit-rate arrival times, rescaled per rate
        for rho in rhos:
            for method in methods:
                key = f"{method}@{rho:g}"
                if key in result["cells"]:
                    continue
                rate = rho * c_full
                tp.serve([job(method, c) for c in warm], teacher, C_MAX, dict(getattr(teacher, "counters", {}) or {}))  # warm-up, untimed
                jobs, fw = serve_open(lambda i, m=method: job(m, measure[i]), unit / rate, teacher, C_MAX)
                lat = np.array([j.t_done - j.t_arr for j in jobs])
                wait = np.array([j.t_admit - j.t_arr for j in jobs])
                span = max(j.t_done for j in jobs) - float(unit[0] / rate)
                achieved = len(jobs) / span
                agree = [list(j.topk()) == cache_top[j.qid] for j in jobs if j.qid in cache_top]
                result["cells"][key] = {
                    "method": method, "rho": rho, "offered_qps": rate, "achieved_qps": achieved, "overloaded": bool(achieved < 0.95 * rate),
                    "latency_p50_s": float(np.median(lat)), "latency_p95_s": float(np.quantile(lat, 0.95)), "latency_p99_s": float(np.quantile(lat, 0.99)),
                    "latency_mean_s": float(lat.mean()), "queue_wait_mean_s": float(wait.mean()), "cheap_mean_s": float(np.mean([j.cheap_s for j in jobs])),
                    "calls_mean": float(np.mean([j.calls for j in jobs])), "forward_passes": fw["forward_passes"],
                    "dummy_row_fraction": (fw["dummy_rows"] / fw["rows"]) if fw["rows"] else None,
                    "token_utilisation": (fw["real_tokens"] / fw["padded_tokens"]) if fw["padded_tokens"] else None,
                    "agree_with_cache": float(np.mean(agree)) if agree else None, "agree_n": len(agree)}
                write_json(out_path, result)
                r = result["cells"][key]
                log(f"{tname}/{domain} rho={rho:g} {method}: offered {rate:.2f} achieved {achieved:.2f} q/s, p50 {r['latency_p50_s']:.1f} p95 {r['latency_p95_s']:.1f} p99 {r['latency_p99_s']:.1f} s, "
                    f"wait {r['queue_wait_mean_s']:.1f} s, calls {r['calls_mean']:.1f}, util {r['token_utilisation']:.2f}, agree {r['agree_with_cache']:.3f}{' OVERLOADED' if r['overloaded'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
