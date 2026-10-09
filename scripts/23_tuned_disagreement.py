"""What the tuned micro-batch changes in the teacher's own top-k -> runs/latency_tuned_diff/<teacher>/<domain>.json

Re-scores the 300 timed queries of scripts/09_system_latency.py (same sample: seed + 99, the first 30 drawn are the
tuning queries) with the full teacher at the micro-batch the tuned regime selected for the full teacher, and compares
each list with the canonical cache (data/teacher/<t>/<domain>/test.npz):
  * ordered top-k agreement (reproduces agree_with_cache of the tuned regime),
  * set agreement: the same k documents in another order versus a change of membership,
  * for each disagreement, the canonical boundary gap s_(k) - s_(k+1) against the largest score change on the pool,
  * nDCG@10 of both lists against the BEIR qrels.
Nothing here enters a calibration; the canonical cache stays the target of every risk statement.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank.data import load_pool_qrels  # noqa: E402
from rscrank.io import ensure_dir, read_json, write_json  # noqa: E402
from rscrank.metrics import ndcg_at_k  # noqa: E402
from rscrank.models import get_teacher  # noqa: E402
from rscrank.store import load_candidates, load_docstore, load_store  # noqa: E402


def main() -> int:
    ap = base_parser("tuned micro-batch: membership versus order changes of the teacher top-k")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--tune-queries", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args)
    pr = cfg["protocol"]
    k, N = pr["k"], cfg["candidates"]["N"]
    for tname in ([args.teacher] if args.teacher else teachers(cfg)):
        tcfg = dict(cfg["teachers"].get(tname, {}) or {})
        teacher = get_teacher(tname, tcfg)
        for domain in ([args.domain] if args.domain else main_domains(cfg)):
            lat = read_json(ROOT / cfg["paths"]["runs"] / "latency" / tname / f"{domain}.json")
            bs_tuned = int(lat["regimes"]["tuned"]["full"]["batch_size"])
            bs_canon = int(tcfg.get("batch_size", 8))
            cands = load_candidates(P(cfg, "candidates") / domain / "test.jsonl")
            docstore = load_docstore(P(cfg, "candidates") / domain / "docstore.jsonl")
            st = load_store(P(cfg, "teacher") / tname / domain / "test.npz")
            cache = {}
            for i, qid in enumerate(st["qids"]):
                n = int(st["valid"][i].sum())
                cache[str(qid)] = ([str(x) for x in st["docids"][i, :n]], st["score"][i, :n].astype(float))
            qrels = load_pool_qrels(cfg, domain, P(cfg, "data_raw"))
            rng = np.random.default_rng(cfg["seed"] + 99)
            sel = rng.choice(len(cands), size=min(len(cands), pr["latency_queries"] + args.tune_queries), replace=False)
            measure = [cands[i] for i in sel[args.tune_queries:]]
            teacher.batch_size = bs_tuned
            rows, t0 = [], time.perf_counter()
            for c in measure:
                ids = [x["docid"] for x in c["docs"]][:N]
                c_ids, c_sc = cache[str(c["qid"])]
                if c_ids != ids:
                    raise SystemExit(f"{tname}/{domain}/{c['qid']}: cached pool differs from the candidate list")
                sc = np.asarray(teacher.score(c["query"], [(docstore[i]["title"], docstore[i]["text"]) for i in ids]), float)
                o_t, o_c = core.teacher_order(sc, ids), core.teacher_order(c_sc, ids)
                top_t, top_c = [ids[p] for p in o_t[:k]], [ids[p] for p in o_c[:k]]
                qr = qrels.get(str(c["qid"]), {})
                rows.append({
                    "qid": str(c["qid"]),
                    "ordered_equal": top_t == top_c,
                    "set_equal": set(top_t) == set(top_c),
                    "n_swapped_out": len(set(top_c) - set(top_t)),
                    "boundary_gap": float(c_sc[o_c[k - 1]] - c_sc[o_c[k]]),
                    "max_abs_diff": float(np.max(np.abs(sc - c_sc))),
                    "median_abs_diff": float(np.median(np.abs(sc - c_sc))),
                    "score_iqr": float(np.subtract(*np.percentile(c_sc, [75, 25]))),
                    "ndcg_tuned": ndcg_at_k(top_t, qr, 10) if qr else None,
                    "ndcg_canonical": ndcg_at_k(top_c, qr, 10) if qr else None,
                })
            secs = time.perf_counter() - t0
            arr = lambda key: np.array([r[key] for r in rows], float)  # noqa: E731
            ordered, sets = arr("ordered_equal").astype(bool), arr("set_equal").astype(bool)
            mis = ~ordered
            member = ~sets
            gap, mx = arr("boundary_gap"), arr("max_abs_diff")
            nd = [(r["ndcg_tuned"], r["ndcg_canonical"]) for r in rows if r["ndcg_tuned"] is not None]
            nd_t, nd_c = np.array([a for a, _ in nd], float), np.array([b for _, b in nd], float)
            summ = {
                "teacher": tname, "domain": domain, "k": k, "N": N, "n_queries": len(rows),
                "batch_size_tuned": bs_tuned, "batch_size_canonical": bs_canon, "seconds": secs,
                "ordered_agreement": float(ordered.mean()),
                "agree_with_cache_reported": lat["regimes"]["tuned"]["full"]["agree_with_cache"],
                "set_agreement": float(sets.mean()),
                "frac_mismatch": float(mis.mean()),
                "frac_order_only": float((mis & sets).mean()),
                "frac_membership": float(member.mean()),
                "share_of_mismatches_order_only": float((mis & sets).sum() / max(1, mis.sum())),
                "membership_swapped_docs_mean": float(arr("n_swapped_out")[member].mean()) if member.any() else 0.0,
                "membership_boundary_gap_median": float(np.median(gap[member])) if member.any() else None,
                "membership_gap_below_max_diff": float(np.mean(gap[member] <= mx[member])) if member.any() else None,
                "all_boundary_gap_median": float(np.median(gap)),
                "max_abs_diff_median": float(np.median(mx)), "max_abs_diff_max": float(np.max(mx)),
                "median_abs_diff_median": float(np.median(arr("median_abs_diff"))),
                "score_iqr_median": float(np.median(arr("score_iqr"))),
                "ndcg10_tuned": float(nd_t.mean()) if len(nd) else None,
                "ndcg10_canonical": float(nd_c.mean()) if len(nd) else None,
                "ndcg10_max_abs_query_change": float(np.max(np.abs(nd_t - nd_c))) if len(nd) else None,
                "ndcg10_frac_queries_changed": float(np.mean(np.abs(nd_t - nd_c) > 1e-12)) if len(nd) else None,
            }
            od = ensure_dir(ROOT / cfg["paths"]["runs"] / "latency_tuned_diff" / tname)
            write_json(od / f"{domain}.json", {"summary": summ, "queries": rows})
            log(f"{tname}/{domain}: bs {bs_tuned} vs {bs_canon}; ordered {summ['ordered_agreement']:.3f} (reported {summ['agree_with_cache_reported']:.3f}), "
                f"set {summ['set_agreement']:.3f}, order-only share {summ['share_of_mismatches_order_only']:.3f}, "
                f"nDCG@10 {summ['ndcg10_tuned']:.4f} vs {summ['ndcg10_canonical']:.4f}, {secs:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
