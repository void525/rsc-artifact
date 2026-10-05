"""Throughput under concurrent queries on the GPU (node N11, amendment A4) -> runs/throughput/<teacher>/<domain>.json

The sequential measurement of scripts/09_system_latency.py answers "how long does ONE query take when it has the
GPU to itself"; there, a controller that scores in several small rounds pays a fixed-shape micro-batch per round and
saved teacher calls are idle GPU time, not reclaimed capacity.  A serving system keeps several queries in flight and
fills each teacher forward pass with pairs from all of them.  This script simulates that server: C queries are in
flight; in every round the pending pairs of all active queries are scored in ONE ``score_pairs`` call (canonical
fixed-shape micro-batches, so every score is bit-identical to the cached teacher and the controllers' decisions are
exactly those of the replay); finished queries leave and new ones are admitted.  Reported per (method, C):
throughput (queries/s), GPU-seconds per query (= wall-clock of the whole run / queries, the quantity a capacity
planner pays for), per-query latency (admission -> completion; mean/p50/p95), mean teacher calls, forward passes,
dummy-row fraction and the agreement of the live top-k with the cached full-teacher top-k.

Methods: full, rsc (pre-registered 16-then-8 schedule), rsc_blockers (amendment A2: first 32, then blocker batches
>= 16), prefix_uniform, prefix_adaptive.  Same 300 test queries per configuration as scripts/09_system_latency.py
(same seed), 5 untimed warm-up queries, CUDA-synchronised timing, cheap stage timed separately and included.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, teachers  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank.baselines import adaptive_prefix_budget  # noqa: E402
from rscrank.data import cheap_model_for, distilled_model_dir  # noqa: E402
from rscrank.heads import BudgetHead, MeanHead, ScaleHead, pair_features, query_level_features  # noqa: E402
from rscrank.io import ensure_dir, read_json, write_json  # noqa: E402
from rscrank.models import get_cheap, get_teacher  # noqa: E402
from rscrank.store import load_candidates, load_docstore  # noqa: E402

_spec = importlib.util.spec_from_file_location("lat09", Path(__file__).resolve().parent / "09_system_latency.py")
lat09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lat09)  # type: ignore[union-attr]
sync, cached_topk, make_rsc_policy = lat09.sync, lat09.cached_topk, lat09.make_rsc_policy

METHODS = ("full", "rsc", "rsc_blockers", "prefix_uniform", "prefix_adaptive")


class StepController:
    """Round-by-round version of core.run_controller (same policy / schedule / stop rule / tie rule).

    ``next_batch()`` returns the ids to score in the next round (empty when finished); ``observe(ids, scores)``
    feeds the teacher scores back.  Equivalence with run_controller is covered by tests/test_throughput_controller.py.
    """

    def __init__(self, ids, mu, sigma, lam, k, policy, batch_schedule):
        self.ids = list(ids)
        self.mu_by_id = {i: float(m) for i, m in zip(self.ids, mu)}
        self.sg_by_id = {i: float(g) for i, g in zip(self.ids, sigma)}
        self.lam, self.k, self.policy, self.sched = float(lam), int(k), policy, batch_schedule
        self.observed = {}
        self.remaining = list(self.ids)
        self.rounds = 0
        self.done = False
        self.early = False

    def next_batch(self):
        if self.done or not self.remaining:
            self.done = True
            return []
        order = list(self.policy(self.remaining, self.observed, self.mu_by_id))
        if getattr(self.sched, "adaptive", False):
            b = max(1, int(self.sched(self.rounds, self.observed, order, self.mu_by_id, self.sg_by_id, self.lam, self.k)))
        else:
            b = max(1, int(self.sched(self.rounds)))
        return order[:b]

    def observe(self, ids, scores):
        for i, sc in zip(ids, scores):
            if not math.isfinite(sc):
                raise ValueError("oracle returned NaN/inf")
            self.observed[i] = float(sc)
        self.rounds += 1
        self.remaining = [i for i in self.remaining if i not in self.observed]
        if not self.remaining:
            self.done = True
            return
        if len(self.observed) >= self.k and core.can_stop(list(self.observed.values()), [self.mu_by_id[j] for j in self.remaining], [self.sg_by_id[j] for j in self.remaining], self.lam, self.k):
            self.done, self.early = True, True

    def topk(self):
        items = list(self.observed.items())
        ids = [i for i, _ in items]
        sc = np.array([v for _, v in items], dtype=float)
        order = core.teacher_order(sc, ids)
        return [ids[p] for p in order[: self.k]]


def blocker_sched(bf, br):
    base = core.blocker_batch_schedule(min_batch=br)

    def sched(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_):
        return bf if round_idx == 0 else base(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_)

    sched.adaptive = True  # type: ignore[attr-defined]
    return sched


class QueryJob:
    """One query's state in the simulated server."""

    def __init__(self, method, cand, docstore, cheap, heads, meta, k, lam, tH, m_uni, bf, br, N, policy_name, seed):
        self.method, self.qid, self.query = method, cand["qid"], cand["query"]
        self.ids = [x["docid"] for x in cand["docs"]][:N]
        self.docs = {i: (docstore[i]["title"], docstore[i]["text"]) for i in self.ids}
        self.k = k
        self.observed = {}
        self.pending = []
        self.ctrl = None
        self.t_admit = None
        self.t_done = None
        self.cheap_s = 0.0
        if method == "full":
            self.pending = list(self.ids)
            return
        # cheap stage (shared by RSC and the prefix methods); timed and added to the wall-clock accounting
        sync()
        t0 = time.perf_counter()
        emb, logit, ntok = cheap.features(self.query, [self.docs[i] for i in self.ids])
        dlen = np.array([len((self.docs[i][0] + " " + self.docs[i][1]).split()) for i in self.ids], float)
        bm25 = np.array([x["bm25"] for x in cand["docs"]][:N])
        X = pair_features(emb, logit, bm25, np.array([x["rank"] for x in cand["docs"]][:N]), float(len(self.query.split())), dlen, N)
        mu = heads["mu"].predict(X)
        z = query_level_features(logit, mu, dlen, float(len(self.query.split())), k)
        sigma = float(heads["sigma"].predict(z[None])[0]) if meta["use_sigma_head"] else 1.0
        sync()
        self.cheap_s = time.perf_counter() - t0
        if method.startswith("rsc"):
            feats = {"cheap_raw": dict(zip(self.ids, np.asarray(logit, float))), "bm25": dict(zip(self.ids, bm25.astype(float))), "length": dict(zip(self.ids, dlen)), "tokens": dict(zip(self.ids, np.asarray(ntok, float)))}
            policy = make_rsc_policy(policy_name, feats, seed)
            sched = blocker_sched(bf, br) if method == "rsc_blockers" else core.default_batch_schedule(k, bf, br)
            self.ctrl = StepController(self.ids, mu, np.full(len(self.ids), sigma), lam, k, policy, sched)
            self.pending = self.ctrl.next_batch()
        else:
            if method == "prefix_uniform":
                m = m_uni
            else:
                h = float(heads["budget"].predict(z[None])[0])
                m = adaptive_prefix_budget(h, tH, k, len(self.ids))
            self.pending = [self.ids[p] for p in core.teacher_order(mu, self.ids)][: max(k, m)]

    def observe(self, ids, scores):
        for i, s in zip(ids, scores):
            self.observed[i] = float(s)
        if self.ctrl is not None:
            self.ctrl.observe(ids, scores)
            self.pending = [] if self.ctrl.done else self.ctrl.next_batch()
        else:
            self.pending = []

    @property
    def done(self):
        return not self.pending

    def topk(self):
        if self.ctrl is not None:
            return self.ctrl.topk()
        items = list(self.observed.items())
        ids = [i for i, _ in items]
        sc = np.array([v for _, v in items], dtype=float)
        order = core.teacher_order(sc, ids)
        return [ids[p] for p in order[: self.k]]

    @property
    def calls(self):
        return len(self.observed)


def serve(jobs, teacher, C, counters_before):
    """Run the simulated server over ``jobs`` with C queries in flight; returns wall-clock seconds and per-job stats."""
    queue = list(jobs)
    active = []
    sync()
    t_start = time.perf_counter()
    while queue or active:
        while queue and len(active) < C:
            j = queue.pop(0)
            j.t_admit = time.perf_counter()
            active.append(j)
        pairs, owners = [], []
        for j in active:
            for i in j.pending:
                pairs.append((j.query, j.docs[i]))
                owners.append((j, i))
        if pairs:
            sc = teacher.score_pairs(pairs)
            sync()
            by_job = {}
            for (j, i), s in zip(owners, sc):
                by_job.setdefault(id(j), (j, [], []))
                by_job[id(j)][1].append(i)
                by_job[id(j)][2].append(float(s))
            for _, (j, ids, scs) in by_job.items():
                j.observe(ids, scs)
        still = []
        now = time.perf_counter()
        for j in active:
            if j.done:
                j.t_done = now
            else:
                still.append(j)
        active = still
    sync()
    total = time.perf_counter() - t_start
    c = getattr(teacher, "counters", {}) or {}
    fw = {kk: c.get(kk, 0) - counters_before.get(kk, 0) for kk in ("forward_passes", "rows", "dummy_rows", "real_tokens", "padded_tokens")}
    return total, fw


def main() -> int:
    ap = base_parser("throughput under concurrency")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--concurrency", default="1,8,32", help="comma-separated numbers of queries in flight")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tune-queries", type=int, default=30, help="skip the same tuning queries as 09_system_latency.py (same sample)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--policy", default="mu")
    ap.add_argument("--first", type=int, default=32, help="first round of the blocker schedule")
    ap.add_argument("--rest", type=int, default=16, help="minimum blocker batch")
    ap.add_argument("--out-dir", default=None, help="output directory relative to the repo root (default: <runs>/throughput); repeated timed passes write elsewhere")
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k, a0 = pr["k"], pr["alpha_main"]
    N = cfg["candidates"]["N"]
    primary = pr.get("primary_regime", "finite")
    Cs = [int(x) for x in args.concurrency.split(",") if x]
    methods = [m for m in args.methods.split(",") if m]
    out_root = (ROOT / args.out_dir) if args.out_dir else (ROOT / cfg["paths"]["runs"] / "throughput")
    ccfg = {kk: v for kk, v in cfg["cheap"].items() if kk not in ("distill", "model_by_teacher")}
    shared_cheap = None
    doms = list(cfg["domains"]["main"]) + [d for d in (cfg["domains"].get("extra", []) or []) if d not in cfg["domains"]["main"]]
    for tname in ([args.teacher] if args.teacher else teachers(cfg)):
        tcfg = dict(cfg["teachers"].get(tname, {}) or {})
        teacher = get_teacher(tname, tcfg)
        cheap_name = cheap_model_for(cfg, tname)
        if cheap_name == "minilm-distilled":
            cheap = get_cheap("minilm-distilled", {**ccfg, "path": str(distilled_model_dir(P(cfg, "models"), tname)), "name": f"minilm-distilled:{tname}"})
        else:
            shared_cheap = shared_cheap or get_cheap(cheap_name, ccfg)
            cheap = shared_cheap
        hd = P(cfg, "models") / "heads" / tname
        meta = read_json(hd / "meta.json")
        heads = {"mu": MeanHead.load(hd / "mu.npz", meta["kappa"]), "sigma": ScaleHead.load(hd / "sigma.npz"), "budget": BudgetHead.load(hd / "budget.npz")}
        for domain in doms:
            lam_path = ROOT / cfg["paths"]["runs"] / "calib" / tname / domain / "lambdas.json"
            if not lam_path.exists():
                log(f"{tname}/{domain}: no calibration; skipping")
                continue
            lams = read_json(lam_path)
            L = lams["k"][str(k)]["alpha"][str(a0)][primary]

            def _thr(e):
                return math.inf if e.get("threshold_is_inf") else (-math.inf if e.get("threshold_is_neg_inf") else float(e["threshold"]))

            lam, tH = _thr(L["A_k1"]), _thr(L["e_H"])
            m_uni = int(min(L["H_k"]["threshold"] if not L["H_k"]["threshold_is_inf"] else N, N))
            cands = load_candidates(P(cfg, "candidates") / domain / "test.jsonl")
            docstore = load_docstore(P(cfg, "candidates") / domain / "docstore.jsonl")
            cache_top = cached_topk(P(cfg, "teacher") / tname / domain / "test.npz", k)
            rng = np.random.default_rng(cfg["seed"] + 99)
            sel = rng.choice(len(cands), size=min(len(cands), pr["latency_queries"] + args.tune_queries), replace=False)
            measure = [cands[i] for i in sel[args.tune_queries :]]
            warm = measure[: max(1, args.warmup)]
            result = {"teacher": tname, "domain": domain, "k": k, "alpha": a0, "N": N, "n_queries": len(measure), "policy": args.policy, "cheap_model": cheap.name,
                      "teacher_settings": teacher.settings() if hasattr(teacher, "settings") else {"name": teacher.name}, "blocker_schedule": {"first": args.first, "min_batch": args.rest},
                      "concurrency": Cs, "methods": {}}
            bf, br = pr["batch_first"], pr["batch_rest"]

            def make_jobs(method, cs):
                out = []
                for c in cs:
                    if method == "rsc_blockers":
                        out.append(QueryJob(method, c, docstore, cheap, heads, meta, k, lam, tH, m_uni, args.first, args.rest, N, args.policy, cfg["seed"]))
                    else:
                        out.append(QueryJob(method, c, docstore, cheap, heads, meta, k, lam, tH, m_uni, bf, br, N, args.policy, cfg["seed"]))
                return out

            for method in methods:
                R = {}
                for C in Cs:
                    serve(make_jobs(method, warm), teacher, C, dict(getattr(teacher, "counters", {}) or {}))  # warm-up, untimed
                    jobs = make_jobs(method, measure)  # cheap stage runs here (timed inside each job)
                    cheap_total = sum(j.cheap_s for j in jobs)
                    before = dict(getattr(teacher, "counters", {}) or {})
                    total, fw = serve(jobs, teacher, C, before)
                    lat = np.array([j.t_done - j.t_admit for j in jobs])
                    calls = np.array([j.calls for j in jobs], float)
                    agree = [list(j.topk()) == cache_top[j.qid] for j in jobs if cache_top is not None and j.qid in cache_top]
                    wall = total + cheap_total
                    R[str(C)] = {"wall_s": wall, "teacher_wall_s": total, "cheap_s": cheap_total, "throughput_qps": len(jobs) / wall, "gpu_s_per_query": wall / len(jobs),
                                 "latency_mean_s": float(lat.mean()), "latency_p50_s": float(np.median(lat)), "latency_p95_s": float(np.quantile(lat, 0.95)),
                                 "calls_mean": float(calls.mean()), "forward_passes": fw["forward_passes"], "rows": fw["rows"], "dummy_rows": fw["dummy_rows"],
                                 "dummy_row_fraction": (fw["dummy_rows"] / fw["rows"]) if fw["rows"] else None, "padded_tokens": fw["padded_tokens"], "real_tokens": fw["real_tokens"],
                                 "agree_with_cache": (float(np.mean(agree)) if agree else None), "agree_n": len(agree)}
                    log(f"{tname}/{domain}/{method}/C={C}: {R[str(C)]['throughput_qps']:.2f} q/s, {1000 * R[str(C)]['gpu_s_per_query']:.0f} ms GPU per query, latency mean {1000 * R[str(C)]['latency_mean_s']:.0f} ms, calls {calls.mean():.1f}, fw {fw['forward_passes']}, dummy {100 * (R[str(C)]['dummy_row_fraction'] or 0):.0f} %, agree {R[str(C)]['agree_with_cache']}")
                result["methods"][method] = R
                out = ensure_dir(out_root / tname) / f"{domain}.json"
                write_json(out, result)  # checkpoint after every method
            for C in Cs:
                f = result["methods"].get("full", {}).get(str(C))
                if f:
                    for m in methods:
                        r = result["methods"].get(m, {}).get(str(C))
                        if r:
                            r["gpu_s_ratio_vs_full"] = r["gpu_s_per_query"] / f["gpu_s_per_query"]
                            r["throughput_gain_vs_full"] = r["throughput_qps"] / f["throughput_qps"] - 1
            out = ensure_dir(out_root / tname) / f"{domain}.json"
            write_json(out, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
