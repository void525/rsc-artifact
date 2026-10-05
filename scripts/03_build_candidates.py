"""BM25 top-N candidates for every query of every split -> data/candidates/<domain>/<split>.jsonl,
docstore.jsonl, and runs/candidates/manifest.json with the BM25 sanity check (node N04).

Backend: configs/main.yaml candidates.backend (default bm25s).  ``pyserini`` is optional: ANY
exception while importing pyserini or retrieving with it (no Java 21, no index download, ...) is
logged and the domain falls back to bm25s (decision ladder D-CAND); the manifest records the
requested and the effective backend, the reason, BM25 parameters and index/retrieval times.
bm25s parameters (optional keys, read with defaults): candidates.k1 (0.9), candidates.b (0.4),
candidates.stemmer ("porter"), candidates.n_threads (all CPUs), candidates.bm25s_chunk_docs (200000).
Domains: domains.main + cross + extra; the sanity check uses the pool qrels (test; dev for msmarco).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, domains, load_config, log  # noqa: E402

from rscrank.candidates import REFERENCE_BM25_FLAT_NDCG10, needed_docids, retrieve, write_candidates, write_docstore  # noqa: E402
from rscrank.data import domain_list, load_pool_qrels, pool_split  # noqa: E402
from rscrank.io import read_json, read_jsonl, sha256_file, write_json  # noqa: E402
from rscrank.metrics import ndcg_at_k, recall_at_k  # noqa: E402


def corpus_iter(raw: Path, domain: str, with_title: bool):
    for r in read_jsonl(raw / domain / "corpus.jsonl"):
        title, text = r.get("title", "") or "", r.get("text", "") or ""
        if with_title:
            yield str(r["_id"]), title, text
        else:
            yield str(r["_id"]), (f"{title} {text}".strip() if title else text)


def bm25s_kwargs(cfg) -> dict:
    c = cfg["candidates"]
    return {"k1": float(c.get("k1", 0.9)), "b": float(c.get("b", 0.4)), "stemmer": str(c.get("stemmer", "porter")), "n_threads": c.get("n_threads"), "chunk_docs": int(c.get("bm25s_chunk_docs", 200_000)), "log": log}


def run_domain(cfg, backend: str, raw: Path, d: str, queries: dict, N: int):
    """(hits, info) with the pyserini -> bm25s fallback."""
    if backend == "pyserini":
        try:
            hits, info = retrieve("pyserini", d, None, queries, N, threads=cfg["candidates"].get("n_threads"))
            info["backend_requested"] = "pyserini"
            return hits, info
        except BaseException as e:  # noqa: BLE001  (pyserini/JVM failures can surface as SystemExit)
            if isinstance(e, KeyboardInterrupt):
                raise
            reason = f"{type(e).__name__}: {e}"[:500]
            log(f"{d}: pyserini failed ({reason}); falling back to bm25s (ladder D-CAND)")
            hits, info = retrieve("bm25s", d, corpus_iter(raw, d, False), queries, N, **bm25s_kwargs(cfg))
            info.update(backend_requested="pyserini", fallback_reason=reason)
            return hits, info
    if backend == "bm25s":
        hits, info = retrieve("bm25s", d, corpus_iter(raw, d, False), queries, N, **bm25s_kwargs(cfg))
    else:
        hits, info = retrieve(backend, d, corpus_iter(raw, d, False), queries, N)
    info["backend_requested"] = backend
    return hits, info


def main() -> int:
    ap = base_parser("build candidates")
    ap.add_argument("--N", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args)
    raw, splits_dir, out_dir = P(cfg, "data_raw"), P(cfg, "splits"), P(cfg, "candidates")
    N = args.N or cfg["candidates"]["N"]
    backend = cfg["candidates"].get("backend", "bm25s")
    doms = domains(cfg, args) if args.synthetic else domain_list(cfg, only=args.domains)
    man_path = ROOT / cfg["paths"]["runs"] / "candidates" / "manifest.json"
    man = {"backend": backend, "backend_requested": backend, "N": N, "domains": {}}
    if args.domains and man_path.exists():
        # a partial run (--domains) updates its domains and keeps the others' entries (same N only)
        old = read_json(man_path)
        if old.get("N") == N:
            man["domains"] = {d: v for d, v in old.get("domains", {}).items() if d not in doms}
    for d in doms:
        queries = {}
        per_split = {}
        for split in ("train", "dev", "cal", "test"):
            f = splits_dir / d / f"{split}.jsonl"
            if f.exists():
                per_split[split] = {r["qid"]: r["text"] for r in read_jsonl(f)}
                queries.update(per_split[split])
        log(f"{d}: retrieving top-{N} for {len(queries)} queries with {backend}")
        t0 = time.time()
        hits, rinfo = run_domain(cfg, backend, raw, d, queries, N)
        files = []
        for split, qs in per_split.items():
            files.append(write_candidates(out_dir, d, split, qs, hits))
        need = needed_docids(files)
        ds = write_docstore(out_dir, d, corpus_iter(raw, d, True), need)
        # sanity on the test split against the pool qrels (test; dev for msmarco)
        qrels = load_pool_qrels(cfg, d, raw)
        test_q = per_split.get("test", {})
        nd, rc, short = [], [], 0
        for q in queries:
            if len(hits.get(q, [])) < N:
                short += 1
        for q in test_q:
            ranked = [h[0] for h in hits.get(q, [])]
            nd.append(ndcg_at_k(ranked, qrels.get(q, {}), 10))
            rc.append(recall_at_k(ranked, qrels.get(q, {}), N))
        info = {"n_queries": len(queries), "n_docstore": len(need), "files": {str(f.relative_to(ROOT)): sha256_file(f) for f in files + [ds]}, "bm25_ndcg10_test": float(sum(nd) / max(1, len(nd))), f"recall@{N}_test": float(sum(rc) / max(1, len(rc))), "queries_with_fewer_than_N": short, "n_test_queries": len(test_q), "pool_split": pool_split(cfg, d), "backend": rinfo.get("backend", backend), "retrieval": rinfo, "seconds_total": round(time.time() - t0, 1)}
        ref = REFERENCE_BM25_FLAT_NDCG10.get(d)
        if info["backend"] == "pyserini" and ref is not None:
            info["reference_ndcg10_pyserini_2cr"] = ref
            info["sanity_ok"] = abs(info["bm25_ndcg10_test"] - ref) <= cfg["candidates"]["sanity_tolerance_ndcg"]
        else:
            if ref is not None:
                info["reference_ndcg10_pyserini_2cr"] = ref  # informational (different analyzer)
            info["sanity_ok"] = True  # no exact reference for other backends; value is recorded
        man["domains"][d] = info
        log(f"{d}: backend={info['backend']} nDCG@10(BM25)={info['bm25_ndcg10_test']:.4f} recall@{N}={info[f'recall@{N}_test']:.4f} short={short} ({info['seconds_total']}s)")
    eff = sorted({v["backend"] for v in man["domains"].values()})
    man["backend"] = "+".join(eff) if eff else backend  # effective backend(s): used by 12_paper_numbers.py
    man["ok"] = all(v["sanity_ok"] for v in man["domains"].values())
    write_json(man_path, man)
    return 0 if man["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
