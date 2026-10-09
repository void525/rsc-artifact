"""Matched serving run: one canonical teacher, one load, four interfaces.

The comparison the deployment question asks for, on the same queries and the same
closed-loop server as scripts/09b_throughput.py (C queries in flight, one canonical
``score_pairs`` call per round):

* ``submit_s`` — the client submits only the calibrated adaptive set S(q).
* ``set_<order>`` — the full pool is walked in an external order, in the paper's
  16/8 chunks; the method may stop only once S(q) has been scored.
* ``rsc_<order>`` — the same order and the same chunks, stopping by the
  rank-sufficient certificate at the already calibrated lambda.
* ``env_<order>`` — the same, at the 13-policy envelope threshold.

Orders are ``bm25`` (first-stage arrival) and ``mu_alt`` (the other cheap model).
Both are already members of the declared family; this script does not refit a
threshold. A chunk is atomic: documents scored after the stop condition became
true inside that chunk are counted as in-flight work that could not be cancelled.

Post-hoc measurement. It does not change any calibration statistic.
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
from rscrank.policies import feature_order  # noqa: E402
from rscrank.store import load_candidates, load_docstore  # noqa: E402

_spec = importlib.util.spec_from_file_location("thr09b", Path(__file__).resolve().parent / "09b_throughput.py")
thr09b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(thr09b)  # type: ignore[union-attr]
sync, cached_topk, serve = thr09b.sync, thr09b.cached_topk, thr09b.serve

_spec08 = importlib.util.spec_from_file_location("rep08", Path(__file__).resolve().parent / "08_replay_eval.py")
rep08 = importlib.util.module_from_spec(_spec08)
_spec08.loader.exec_module(rep08)  # type: ignore[union-attr]

ORDERS = ("bm25", "mu_alt")


class ExternalChunks:
    """Walk a fixed order in atomic chunks. The tail of the chunk after the stop is in-flight."""

    def __init__(self, order, batch_first, batch_rest, stop_fn):
        self.order = list(order)
        self.bf, self.br, self.stop_fn = int(batch_first), int(batch_rest), stop_fn
        self.cursor, self.rounds, self.inflight = 0, 0, 0
        self.observed = {}
        self.done = False

    def next_batch(self):
        if self.done or self.cursor >= len(self.order):
            self.done = True
            return []
        b = self.bf if self.rounds == 0 else self.br
        batch = self.order[self.cursor : self.cursor + b]
        self.cursor += len(batch)
        return batch

    def observe(self, ids, scores):
        stop_at = None
        for n, (i, s) in enumerate(zip(ids, scores)):
            if not math.isfinite(s):
                raise ValueError("oracle returned NaN/inf")
            self.observed[i] = float(s)
            if stop_at is None and self.stop_fn(self.observed):
                stop_at = n
        if stop_at is not None:
            self.inflight += len(ids) - stop_at - 1
            self.done = True
        self.rounds += 1
        if self.cursor >= len(self.order):
            self.done = True


def _topk(observed, k):
    items = list(observed.items())
    ids = [i for i, _ in items]
    sc = np.array([v for _, v in items], dtype=float)
    order = core.teacher_order(sc, ids)
    return [ids[p] for p in order[:k]]


class MatchedJob:
    """One query under one of the four interfaces. Attribute names match scripts/09b serve()."""

    def __init__(self, kind, cand, docstore, cheap, heads, meta, k, lam, tH, N, bf, br, mu_alt):
        self.kind, self.qid, self.query = kind, cand["qid"], cand["query"]
        self.ids = [x["docid"] for x in cand["docs"]][:N]
        self.docs = {i: (docstore[i]["title"], docstore[i]["text"]) for i in self.ids}
        self.k = k
        self.observed = {}
        self.pending = []
        self.ctrl = None
        self.t_admit = None
        self.t_done = None
        self.cheap_s = 0.0
        self.inflight = 0
        docs_in = [self.docs[i] for i in self.ids]
        sync()
        t0 = time.perf_counter()
        emb, logit, ntok = cheap.features(self.query, docs_in)
        dlen = np.array([len((self.docs[i][0] + " " + self.docs[i][1]).split()) for i in self.ids], float)
        bm25 = np.array([x["bm25"] for x in cand["docs"]][:N], float)
        ranks = np.array([x["rank"] for x in cand["docs"]][:N])
        X = pair_features(emb, logit, bm25, ranks, float(len(self.query.split())), dlen, N)
        mu = heads["mu"].predict(X)
        z = query_level_features(logit, mu, dlen, float(len(self.query.split())), k)
        sigma = float(heads["sigma"].predict(z[None])[0]) if meta["use_sigma_head"] else 1.0
        sync()
        self.cheap_s = time.perf_counter() - t0
        mu_by_id = {i: float(m) for i, m in zip(self.ids, mu)}
        pos = core.teacher_order(mu, self.ids)
        h = float(heads["budget"].predict(z[None])[0])
        m = adaptive_prefix_budget(h, tH, k, len(self.ids))
        committed = [self.ids[p] for p in pos[:m]]
        self.committed_n = len(committed)
        if kind == "submit_s":
            self.pending = list(committed)
            return
        order_name = kind.split("_", 1)[1]
        if order_name == "bm25":
            feat = {i: float(v) for i, v in zip(self.ids, bm25)}
            external = feature_order(feat, True)(self.ids, {}, mu_by_id)
        elif order_name == "mu_alt":
            if self.qid not in mu_alt:
                raise KeyError(f"mu_alt missing for {self.qid}")
            alt = mu_alt[self.qid]
            external = feature_order(alt, True)(self.ids, {}, mu_by_id)
        else:
            raise ValueError(kind)
        if kind.startswith("set_"):
            want = set(committed)

            def stop_fn(observed, want=want):
                return want.issubset(observed.keys())
        else:
            sig = {i: sigma for i in self.ids}
            lam_ = float(lam)

            def stop_fn(observed, mu_by_id=mu_by_id, sig=sig, lam_=lam_, k=k, ids=list(self.ids)):
                if len(observed) < k:
                    return False
                rest = [i for i in ids if i not in observed]
                if not rest:
                    return True
                return core.can_stop([observed[i] for i in observed], [mu_by_id[i] for i in rest], [sig[i] for i in rest], lam_, k)

        self.ctrl = ExternalChunks(external, bf, br, stop_fn)
        self.pending = self.ctrl.next_batch()

    def observe(self, ids, scores):
        for i, s in zip(ids, scores):
            self.observed[i] = float(s)
        if self.ctrl is None:
            self.pending = []
            return
        self.ctrl.observe(ids, scores)
        self.inflight = self.ctrl.inflight
        self.pending = [] if self.ctrl.done else self.ctrl.next_batch()

    @property
    def done(self):
        return not self.pending

    def topk(self):
        return _topk(self.observed, self.k)

    @property
    def calls(self):
        return len(self.observed)


def _thr(entry):
    if entry.get("threshold_is_inf"):
        return math.inf
    if entry.get("threshold_is_neg_inf"):
        return -math.inf
    return float(entry["threshold"])


def _env_lambda(teacher, domain):
    path = ROOT / "runs" / "review3" / teacher / f"{domain}.json"
    env = read_json(path)["envelope"]["all13"]["lambda"]
    if isinstance(env, str):
        return math.inf if env == "Infinity" else float(env)
    return float(env)


def _mu_alt(cfg, teacher, domain, k):
    bundles = rep08.load_bundles(cfg, teacher, domain, "test", k)
    out = {}
    for b in bundles:
        if "mu_alt" not in b.features:
            continue
        out[b.qid] = {i: float(v) for i, v in zip(b.ids, np.asarray(b.features["mu_alt"], float))}
    return out


def _half_range(xs):
    xs = [float(x) for x in xs]
    return 0.5 * (max(xs) - min(xs))


def aggregate(pass_roots, out_path):
    """Mean and half-range over timed passes, for every teacher/domain/method."""
    acc = {}
    for root in pass_roots:
        for path in sorted(Path(root).glob("*/*.json")):
            rec = read_json(path)
            key = (rec["teacher"], rec["domain"])
            acc.setdefault(key, {}).setdefault("methods", {})
            for method, cell in rec["methods"].items():
                acc[key]["methods"].setdefault(method, []).append(cell)
            acc[key]["meta"] = {kk: rec[kk] for kk in ("teacher", "domain", "k", "alpha", "N", "n_queries", "concurrency", "orders", "note")}
    summary = {"passes": len(pass_roots), "configs": []}
    for key, blob in acc.items():
        methods = {}
        for method, cells in blob["methods"].items():
            def col(name, cells=cells):
                return [c[name] for c in cells]
            methods[method] = {
                "n_passes": len(cells),
                "calls_mean": float(np.mean(col("calls_mean"))),
                "calls_half_range": _half_range(col("calls_mean")),
                "inflight_mean": float(np.mean(col("inflight_mean"))),
                "inflight_half_range": _half_range(col("inflight_mean")),
                "risk": float(np.mean(col("risk"))),
                "gpu_s_per_query": float(np.mean(col("gpu_s_per_query"))),
                "gpu_s_half_range": _half_range(col("gpu_s_per_query")),
                "latency_p95_s": float(np.mean(col("latency_p95_s"))),
                "latency_p95_half_range": _half_range(col("latency_p95_s")),
                "throughput_qps": float(np.mean(col("throughput_qps"))),
                "dummy_row_fraction": float(np.mean(col("dummy_row_fraction"))),
            }
        submit = methods.get("submit_s", {}).get("gpu_s_per_query")
        for method, row in methods.items():
            if submit:
                row["gpu_s_ratio_vs_submit_s"] = row["gpu_s_per_query"] / submit
                row["p95_ratio_vs_submit_s"] = row["latency_p95_s"] / methods["submit_s"]["latency_p95_s"]
        summary["configs"].append({**blob["meta"], "methods": methods})
    write_json(out_path, summary)
    return summary


def main() -> int:
    ap = base_parser("matched serving comparison")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--orders", default=",".join(ORDERS))
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--tune-queries", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out-dir", default="runs/matched_service")
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k, a0 = pr["k"], pr["alpha_main"]
    N = cfg["candidates"]["N"]
    primary = pr.get("primary_regime", "finite")
    bf, br = pr["batch_first"], pr["batch_rest"]
    orders = [o for o in args.orders.split(",") if o]
    kinds = ["submit_s"] + [f"{stem}_{order}" for order in orders for stem in ("set", "rsc", "env")]
    out_root = ROOT / args.out_dir
    ccfg = {kk: v for kk, v in cfg["cheap"].items() if kk not in ("distill", "model_by_teacher")}
    doms = [args.domain] if args.domain else list(cfg["domains"]["main"]) + [d for d in (cfg["domains"].get("extra", []) or []) if d not in cfg["domains"]["main"]]
    pass_roots = [out_root / f"pass{i}" for i in range(1, args.passes + 1)]
    shared_cheap = None
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
                lam_rsc, tH = _thr(L["A_k1"]), _thr(L["e_H"])
                lam_env = _env_lambda(tname, domain)
                mu_alt = _mu_alt(cfg, tname, domain, k)
                cands = load_candidates(P(cfg, "candidates") / domain / "test.jsonl")
                docstore = load_docstore(P(cfg, "candidates") / domain / "docstore.jsonl")
                cache_top = cached_topk(P(cfg, "teacher") / tname / domain / "test.npz", k)
                rng = np.random.default_rng(cfg["seed"] + 99)
                sel = rng.choice(len(cands), size=min(len(cands), pr["latency_queries"] + args.tune_queries), replace=False)
                measure = [cands[i] for i in sel[args.tune_queries :]]
                if args.limit_queries:
                    measure = measure[: args.limit_queries]
                warm = measure[: max(1, min(args.warmup, len(measure)))]

                def make_jobs(kind, cs, lam_rsc=lam_rsc, lam_env=lam_env, tH=tH, mu_alt=mu_alt):
                    lam = lam_env if kind.startswith("env_") else lam_rsc
                    return [MatchedJob(kind, c, docstore, cheap, heads, meta, k, lam, tH, N, bf, br, mu_alt) for c in cs]

                for pass_i, pass_root in enumerate(pass_roots, start=1):
                    result = {
                        "teacher": tname, "domain": domain, "k": k, "alpha": a0, "N": N, "n_queries": len(measure),
                        "concurrency": args.concurrency, "orders": orders, "pass": pass_i,
                        "lambda_rsc": lam_rsc, "lambda_env": lam_env, "batch_first": bf, "batch_rest": br,
                        "cheap_model": cheap.name,
                        "note": "post-hoc matched serving run; thresholds are the existing finite-population calibrations, not refit",
                        "methods": {},
                    }
                    for kind in kinds:
                        serve(make_jobs(kind, warm), teacher, args.concurrency, dict(getattr(teacher, "counters", {}) or {}))
                        jobs = make_jobs(kind, measure)
                        cheap_total = sum(j.cheap_s for j in jobs)
                        before = dict(getattr(teacher, "counters", {}) or {})
                        total, fw = serve(jobs, teacher, args.concurrency, before)
                        lat = np.array([j.t_done - j.t_admit for j in jobs])
                        calls = np.array([j.calls for j in jobs], float)
                        inflight = np.array([j.inflight for j in jobs], float)
                        agree = [list(j.topk()) == cache_top[j.qid] for j in jobs if cache_top is not None and j.qid in cache_top]
                        wall = total + cheap_total
                        cell = {
                            "wall_s": wall, "teacher_wall_s": total, "cheap_s": cheap_total,
                            "throughput_qps": len(jobs) / wall, "gpu_s_per_query": wall / len(jobs),
                            "latency_mean_s": float(lat.mean()), "latency_p50_s": float(np.median(lat)), "latency_p95_s": float(np.quantile(lat, 0.95)),
                            "calls_mean": float(calls.mean()), "inflight_mean": float(inflight.mean()),
                            "committed_n_mean": float(np.mean([j.committed_n for j in jobs])),
                            "risk": float(1.0 - np.mean(agree)) if agree else None,
                            "forward_passes": fw["forward_passes"], "rows": fw["rows"], "dummy_rows": fw["dummy_rows"],
                            "dummy_row_fraction": (fw["dummy_rows"] / fw["rows"]) if fw["rows"] else None,
                            "calls": calls.tolist(), "inflight": inflight.tolist(), "qids": [j.qid for j in jobs],
                        }
                        result["methods"][kind] = cell
                        log(f"pass {pass_i} {tname}/{domain}/{kind}/C={args.concurrency}: {cell['throughput_qps']:.2f} q/s, {1000 * cell['gpu_s_per_query']:.0f} ms/q, p95 {1000 * cell['latency_p95_s']:.0f} ms, calls {cell['calls_mean']:.1f}, inflight {cell['inflight_mean']:.2f}, risk {cell['risk']}")
                        write_json(ensure_dir(pass_root / tname) / f"{domain}.json", result)
    aggregate(pass_roots, out_root / "aggregate.json")
    log(f"wrote {out_root / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
