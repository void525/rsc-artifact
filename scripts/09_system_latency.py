"""Real end-to-end rerank-stage latency on the GPU (node N11) -> runs/latency/<teacher>/<domain>.json

For a sample of test queries, runs the *live* pipeline (cheap model -> heads ->
controller -> teacher calls in batches) and the baselines with the real teacher,
in two regimes: controlled (identical microbatch) and tuned (best batch size per
method from a small grid measured on the first --tune-queries queries).
Every (regime, method) is warmed up on --warmup (>= 5) queries before timing; with CUDA every timed
region is bracketed by torch.cuda.synchronize().  Reports mean / p50 / p95 latency, teacher calls,
peak GPU memory, and ``agree_with_cache``: the fraction of timed runs whose live top-k equals the
cached full-teacher top-k (data/teacher/<t>/<domain>/test.npz, fixed total order with docid ties).
--policy selects the RSC scoring policy (mu; mu_binned_length when rscrank.policies provides it).
Cached-replay numbers are never used here.  BM25 retrieval latency is measured separately when
the pyserini index is available and reported as a constant per query.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, teachers  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank import policies as policies_mod  # noqa: E402
from rscrank.baselines import adaptive_prefix_budget  # noqa: E402
from rscrank.data import cheap_model_for, distilled_model_dir  # noqa: E402
from rscrank.heads import BudgetHead, MeanHead, ScaleHead, pair_features, query_level_features  # noqa: E402
from rscrank.io import ensure_dir, read_json, write_json  # noqa: E402
from rscrank.models import get_cheap, get_teacher  # noqa: E402
from rscrank.store import load_candidates, load_docstore, load_store  # noqa: E402

METHODS = ("full", "rsc", "prefix_uniform", "prefix_adaptive")


def _torch_cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


_TORCH_CUDA = _torch_cuda()


def sync() -> None:
    if _TORCH_CUDA is not None:
        _TORCH_CUDA.cuda.synchronize()


class LiveOracle:
    """Teacher oracle that really calls the model (counts calls and time)."""

    def __init__(self, model, query, docs_by_id):
        self.model, self.query, self.docs = model, query, docs_by_id
        self.n_calls, self.seconds = 0, 0.0
        self._seen = {}

    def score(self, ids):
        need = [i for i in ids if i not in self._seen]
        if need:
            sync()
            t0 = time.perf_counter()
            sc = self.model.score(self.query, [self.docs[i] for i in need])
            sync()
            self.seconds += time.perf_counter() - t0
            self.n_calls += len(need)
            self._seen.update(zip(need, sc))
        return [self._seen[i] for i in ids]

    @property
    def scored_ids(self):
        return list(self._seen)


def make_rsc_policy(name: str, feats: dict, seed: int):
    """mu -> mu_descending(); other names through rscrank.policies.make_policy (imported defensively)."""
    if name == "mu":
        return policies_mod.mu_descending()
    fn = getattr(policies_mod, name, None)
    make = getattr(policies_mod, "make_policy", None)
    if make is not None:
        try:
            return make(name, feats, seed)
        except (ValueError, KeyError, TypeError):
            pass
    if callable(fn):
        try:
            return fn()
        except TypeError:
            pass
    raise SystemExit(f"policy {name!r} is not provided by rscrank.policies (available: {getattr(policies_mod, 'POLICY_NAMES', [])})")


def run_query(method, teacher, cheap, heads, meta, q, cand, docstore, k, lam, tH, m_uniform, bf, br, N, policy_name="mu", seed=0, schedule="fixed"):
    ids = [x["docid"] for x in cand["docs"]][:N]
    docs = {i: (docstore[i]["title"], docstore[i]["text"]) for i in ids}
    sync()
    t0 = time.perf_counter()
    if method == "full":
        oracle = LiveOracle(teacher, q, docs)
        sc = oracle.score(ids)
        order = core.teacher_order(np.array(sc), ids)
        top = [ids[p] for p in order[:k]]
        sync()
        return {"latency": time.perf_counter() - t0, "teacher_s": oracle.seconds, "calls": oracle.n_calls, "top": top}
    # cheap stage (shared by RSC and prefix methods)
    emb, logit, ntok = cheap.features(q, [docs[i] for i in ids])
    dlen = np.array([len((docs[i][0] + " " + docs[i][1]).split()) for i in ids], float)
    bm25 = np.array([x["bm25"] for x in cand["docs"]][:N])
    X = pair_features(emb, logit, bm25, np.array([x["rank"] for x in cand["docs"]][:N]), float(len(q.split())), dlen, N)
    mu = heads["mu"].predict(X)
    z = query_level_features(logit, mu, dlen, float(len(q.split())), k)
    sigma = float(heads["sigma"].predict(z[None])[0]) if meta["use_sigma_head"] else 1.0
    sync()
    t_cheap = time.perf_counter() - t0
    oracle = LiveOracle(teacher, q, docs)
    if method.startswith("rsc"):
        feats = {"cheap_raw": dict(zip(ids, np.asarray(logit, float))), "bm25": dict(zip(ids, bm25.astype(float))), "length": dict(zip(ids, dlen)), "tokens": dict(zip(ids, np.asarray(ntok, float)))}
        policy = make_rsc_policy(policy_name, feats, seed)
        if schedule == "blockers":  # amendment A2: first round bf documents, then one round per blocker set (>= br)
            base = core.blocker_batch_schedule(min_batch=br)

            def sched(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_):
                return bf if round_idx == 0 else base(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_)

            sched.adaptive = True  # type: ignore[attr-defined]
        else:
            sched = core.default_batch_schedule(k, bf, br)
        res = core.run_controller(ids, mu, np.full(len(ids), sigma), lam, k, oracle, policy, sched)
        top = res.topk_ids
    else:
        if method == "prefix_uniform":
            m = m_uniform
        else:
            h = float(heads["budget"].predict(z[None])[0])
            m = adaptive_prefix_budget(h, tH, k, len(ids))
        cheap_order = [ids[p] for p in core.teacher_order(mu, ids)][: max(k, m)]
        sc = oracle.score(cheap_order)
        order = core.teacher_order(np.array(sc), cheap_order)
        top = [cheap_order[p] for p in order[:k]]
    sync()
    return {"latency": time.perf_counter() - t0, "cheap_s": t_cheap, "teacher_s": oracle.seconds, "calls": oracle.n_calls, "top": top}


def cached_topk(path: Path, k: int):
    """qid -> ordered cached full-teacher top-k docids (None if the cache is missing)."""
    if not path.exists():
        return None
    st = load_store(path)
    out = {}
    for i, qid in enumerate(st["qids"]):
        n = int(st["valid"][i].sum())
        ids = [str(x) for x in st["docids"][i, :n]]
        sc = st["score"][i, :n].astype(float)
        if n == 0 or not np.all(np.isfinite(sc)):
            continue
        out[str(qid)] = [ids[p] for p in core.teacher_order(sc, ids)[: min(k, n)]]
    return out


def main() -> int:
    ap = base_parser("system latency")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--tune-queries", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5, help="untimed warm-up queries per (regime, method); at least 5")
    ap.add_argument("--policy", default="mu", help="RSC scoring policy: mu (default) or mu_binned_length if rscrank.policies provides it")
    ap.add_argument("--mode", default=None, choices=[None, "fast", "canonical"], help="override teachers.<t>.mode for the live teacher")
    ap.add_argument("--methods", default=",".join(METHODS), help="comma-separated subset of full,rsc,prefix_uniform,prefix_adaptive")
    ap.add_argument("--regimes", default="controlled,tuned", help="comma-separated subset of controlled,tuned")
    ap.add_argument("--schedule", default="fixed", choices=["fixed", "blockers"], help="RSC batch schedule: fixed first/rest or blocker batches (amendment A2)")
    ap.add_argument("--first", type=int, default=None, help="override protocol.batch_first for RSC")
    ap.add_argument("--rest", type=int, default=None, help="override protocol.batch_rest (fixed) / min batch (blockers) for RSC")
    ap.add_argument("--merge", action="store_true", help="merge into an existing runs/latency/<t>/<d>.json instead of overwriting it (RSC variants are stored as method rsc_<schedule>)")
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k, a0 = pr["k"], pr["alpha_main"]
    N = cfg["candidates"]["N"]
    n_warm = max(5, int(args.warmup))
    _ids = ["a", "b", "c"]
    make_rsc_policy(args.policy, {f: dict(zip(_ids, (1.0, 2.0, 3.0))) for f in ("cheap_raw", "bm25", "length", "tokens")}, cfg["seed"])  # fail fast
    ccfg = {kk: v for kk, v in cfg["cheap"].items() if kk not in ("distill", "model_by_teacher")}
    shared_cheap = None  # the frozen cheap model, loaded once if any teacher uses it
    primary = pr.get("primary_regime", "finite")
    doms = list(cfg["domains"]["main"]) + [d for d in (cfg["domains"].get("extra", []) or []) if d not in cfg["domains"]["main"]]
    for tname in ([args.teacher] if args.teacher else teachers(cfg)):
        tcfg = dict(cfg["teachers"].get(tname, {}) or {})
        if args.mode:
            tcfg["mode"] = args.mode
        teacher = get_teacher(tname, tcfg)
        cheap_name = cheap_model_for(cfg, tname)  # per-teacher choice of the D-DISTILL lever
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
                log(f"{tname}/{domain}: no calibration ({lam_path}); skipping")
                continue
            lams = read_json(lam_path)
            L = lams["k"][str(k)]["alpha"][str(a0)][primary]  # thresholds of the primary regime (D-REGIME)
            def _thr(e):
                return math.inf if e.get("threshold_is_inf") else (-math.inf if e.get("threshold_is_neg_inf") else float(e["threshold"]))
            lam, tH = _thr(L["A_k1"]), _thr(L["e_H"])
            m_uni = int(min(L["H_k"]["threshold"] if not L["H_k"]["threshold_is_inf"] else N, N))
            cands = load_candidates(P(cfg, "candidates") / domain / "test.jsonl")
            docstore = load_docstore(P(cfg, "candidates") / domain / "docstore.jsonl")
            cache_top = cached_topk(P(cfg, "teacher") / tname / domain / "test.npz", k)
            rng = np.random.default_rng(cfg["seed"] + 99)
            sel = rng.choice(len(cands), size=min(len(cands), pr["latency_queries"] + args.tune_queries), replace=False)
            tune, measure = [cands[i] for i in sel[: args.tune_queries]], [cands[i] for i in sel[args.tune_queries :]]
            warm = (tune or measure)[:n_warm]
            result = {"teacher": tname, "domain": domain, "k": k, "alpha": a0, "N": N, "synthetic": bool(cfg["smoke"]["synthetic"]), "n_queries": len(measure), "regimes": {}, "policy": args.policy, "warmup_queries": len(warm), "cuda_sync": _TORCH_CUDA is not None, "cheap_model": cheap.name, "teacher_settings": teacher.settings() if hasattr(teacher, "settings") else {"name": teacher.name}, "cache_for_agreement": cache_top is not None}
            batch_grid = [8, 16, 32, 64] if not cfg["smoke"]["synthetic"] else [4, 8]

            bf_, br_ = (args.first if args.first is not None else pr["batch_first"]), (args.rest if args.rest is not None else pr["batch_rest"])

            def rq(method, c):
                return run_query(method, teacher, cheap, heads, meta, c["query"], c, docstore, k, lam, tH, m_uni, bf_, br_, N, args.policy, cfg["seed"], args.schedule)

            methods = [m for m in args.methods.split(",") if m]
            out = ensure_dir(ROOT / cfg["paths"]["runs"] / "latency" / tname) / f"{domain}.json"
            if args.merge and out.exists():
                prev = read_json(out)
                result["regimes"] = prev.get("regimes", {})
                result["variants"] = prev.get("variants", {})
            result.setdefault("variants", {})
            for regime in [r for r in args.regimes.split(",") if r]:
                R = result["regimes"].get(regime, {}) if args.merge else {}
                for method in methods:
                    store_as = method if not (method.startswith("rsc") and args.schedule != "fixed") else f"rsc_{args.schedule}"
                    result["variants"][store_as] = {"schedule": args.schedule, "batch_first": bf_, "batch_rest_or_min": br_, "policy": args.policy}
                    if regime == "controlled":
                        bs = tcfg.get("batch_size", 8)
                    else:  # tune the teacher microbatch per method on the tune queries
                        best = None
                        for b_ in batch_grid:
                            teacher.batch_size = b_
                            for c in warm:  # warm-up for this batch size (untimed)
                                rq(method, c)
                            sync()
                            t0 = time.perf_counter()
                            for c in tune:
                                rq(method, c)
                            sync()
                            dt = time.perf_counter() - t0
                            if best is None or dt < best[0]:
                                best = (dt, b_)
                        bs = best[1]
                    teacher.batch_size = bs
                    for c in warm:  # warm-up before timing (untimed)
                        rq(method, c)
                    lat, calls, teacher_s, agree = [], [], [], []
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                    except Exception:  # noqa: BLE001
                        pass
                    for rep in range(pr["latency_repeats"]):
                        for c in measure:
                            r = rq(method, c)
                            lat.append(r["latency"])
                            calls.append(r["calls"])
                            teacher_s.append(r["teacher_s"])
                            if cache_top is not None and c["qid"] in cache_top:
                                agree.append(list(r["top"]) == cache_top[c["qid"]])
                    lat = np.array(lat)
                    method = store_as  # noqa: PLW2901  (RSC variants keep the base method name for timing, stored under the variant key)
                    R[method] = {"batch_size": bs, "latency_mean_s": float(lat.mean()), "latency_p50_s": float(np.median(lat)), "latency_p95_s": float(np.quantile(lat, 0.95)), "teacher_s_mean": float(np.mean(teacher_s)), "calls_mean": float(np.mean(calls)), "throughput_qps_sequential": float(len(lat) / lat.sum()), "agree_with_cache": (float(np.mean(agree)) if agree else None), "agree_with_cache_n": len(agree)}
                    try:
                        import torch

                        if torch.cuda.is_available():
                            R[method]["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    except Exception:  # noqa: BLE001
                        pass
                    ag = R[method]["agree_with_cache"]
                    log(f"{tname}/{domain}/{regime}/{method}: mean {R[method]['latency_mean_s']*1000:.1f} ms, p95 {R[method]['latency_p95_s']*1000:.1f} ms, calls {R[method]['calls_mean']:.1f}, agree_with_cache {('n/a' if ag is None else f'{ag:.3f}')}")
                result["regimes"][regime] = R
                if "rsc" in R and "full" in R:
                    result["regimes"][regime]["rsc_vs_full_latency_reduction"] = 1 - R["rsc"]["latency_mean_s"] / R["full"]["latency_mean_s"]
                for key_ in [k_ for k_ in R if k_.startswith("rsc_") and isinstance(R[k_], dict)]:  # RSC variants (not the *_reduction floats)
                    if "full" in R:
                        result["regimes"][regime][f"{key_}_vs_full_latency_reduction"] = 1 - R[key_]["latency_mean_s"] / R["full"]["latency_mean_s"]
                write_json(out, result)  # checkpoint after every regime
            write_json(out, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
