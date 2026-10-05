"""Frozen cheap-model features for every (query, candidate) pair -> data/features/<domain>/<split>.npz (node N06).

Stores: emb [Q,N,d] float16, logit [Q,N] f32, ntok [Q,N] i32 (cheap tokenizer length),
dlen [Q,N] i32 (document words), bm25 [Q,N] f32, rank [Q,N] i32, qlen [Q] i32, valid mask.
Pairs of --chunk-queries queries are scored together with length-sorted batches (order restored).
Idempotent: existing outputs built from the same candidates are skipped unless --force.

cheap.model = minilm            -> data/features/<domain>/<split>.npz, runs/features/manifest.json
cheap.model = minilm-distilled  -> one model per teacher (models/cheap_distilled/<teacher>/, from
                                   scripts/06b_distill_cheap.py) and per-teacher features
                                   data/features_distilled/<teacher>/<domain>/<split>.npz,
                                   runs/features_distilled/<teacher>/manifest.json (docs/MODELS.md)
--limit-queries N (debug) featurises only the first N queries of each split; such stores are
marked incomplete in their meta and are recomputed by a later run without the limit.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, domains, load_config, log, teachers  # noqa: E402

from rscrank.data import distilled_model_dir, domain_list, features_dir_for  # noqa: E402
from rscrank.io import read_json, sha256_file, write_json  # noqa: E402
from rscrank.models import get_cheap  # noqa: E402
from rscrank.store import load_candidates, load_docstore, save_store, store_meta  # noqa: E402


def rel(p: Path) -> str:
    """Repo-relative key for manifests; absolute when paths.* point outside the repo."""
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def build_split(model, rows, docstore, N, chunk_q, prefix):
    Q = len(rows)
    emb = np.zeros((Q, N, model.dim), dtype=np.float16)
    logit = np.full((Q, N), np.nan, dtype=np.float32)
    ntok = np.zeros((Q, N), dtype=np.int32)
    dlen = np.zeros((Q, N), dtype=np.int32)
    bm25 = np.zeros((Q, N), dtype=np.float32)
    rank = np.zeros((Q, N), dtype=np.int32)
    qlen = np.zeros(Q, dtype=np.int32)
    qids, docids = [], []
    t0, done_pairs = time.time(), 0
    for i, r in enumerate(rows):
        ds = [x["docid"] for x in r["docs"]][:N]
        docs = [(docstore[x]["title"], docstore[x]["text"]) for x in ds]
        dlen[i, : len(ds)] = [len((t + " " + x).split()) for t, x in docs]
        bm25[i, : len(ds)] = [x["bm25"] for x in r["docs"][:N]]
        rank[i, : len(ds)] = [x["rank"] for x in r["docs"][:N]]
        qlen[i] = len(r["query"].split())
        qids.append(r["qid"])
        docids.append(ds)
    for s in range(0, Q, chunk_q):
        pairs, where = [], []
        for i in range(s, min(Q, s + chunk_q)):
            for j, d in enumerate(docids[i]):
                pairs.append((rows[i]["query"], (docstore[d]["title"], docstore[d]["text"])))
                where.append((i, j))
        if not pairs:
            continue
        e, lg, nt = model.features_pairs(pairs)
        ii = np.array([w[0] for w in where])
        jj = np.array([w[1] for w in where])
        emb[ii, jj] = np.asarray(e).astype(np.float16)
        logit[ii, jj] = lg
        ntok[ii, jj] = nt
        done_pairs += len(pairs)
        dt = max(1e-9, time.time() - t0)
        log(f"{prefix}: {min(Q, s + chunk_q)}/{Q} queries, {done_pairs} pairs, {done_pairs / dt:.1f} pairs/s")
    arrays = {"emb": emb, "logit": logit, "ntok": ntok, "dlen": dlen, "bm25": bm25, "rank": rank, "qlen": qlen}
    return qids, docids, arrays, done_pairs, time.time() - t0


def run_model(cfg, args, model, feat_dir: Path, man_path: Path, extra_meta: dict) -> int:
    cand_dir = P(cfg, "candidates")
    N = cfg["candidates"]["N"]
    doms = domains(cfg, args) if args.synthetic else domain_list(cfg, only=args.domains)
    man = {"model": model.name, "dim": model.dim, "files": {}, "features_dir": rel(feat_dir), **extra_meta}
    complete = True
    for d in doms:
        if not (cand_dir / d / "docstore.jsonl").exists():
            log(f"{d}: no candidates yet; skipping")
            continue
        docstore = load_docstore(cand_dir / d / "docstore.jsonl")
        for split in ("train", "dev", "cal", "test"):
            cf = cand_dir / d / f"{split}.jsonl"
            if not cf.exists():
                continue
            out = feat_dir / d / f"{split}.npz"
            cand_hash = sha256_file(cf)
            if out.exists() and not args.force:
                meta = store_meta(out)
                if meta.get("cand_sha256") == cand_hash and meta.get("model") == model.name and meta.get("limit_queries") == args.limit_queries:
                    log(f"skip existing {out}")
                    man["files"][rel(out)] = sha256_file(out)
                    complete = complete and meta.get("limit_queries") is None
                    continue
                log(f"{out} is stale (candidates, model or --limit-queries changed); recomputing")
            rows = load_candidates(cf)
            if args.limit_queries:
                rows = rows[: args.limit_queries]
                complete = False
            qids, docids, arrays, n_pairs, secs = build_split(model, rows, docstore, N, args.chunk_queries, f"{d}/{split}")
            meta = {"model": model.name, "cand_sha256": cand_hash, **extra_meta}
            if args.limit_queries:
                meta.update(limit_queries=args.limit_queries, complete=False)
            save_store(out, qids, docids, arrays, N, meta)
            man["files"][rel(out)] = sha256_file(out)
            log(f"wrote {out} ({len(rows)} queries, {n_pairs} pairs, {n_pairs / max(1e-9, secs):.1f} pairs/s)")
    man["complete"] = complete
    write_json(man_path, man)
    return 0


def main() -> int:
    ap = base_parser("cheap features")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--chunk-queries", type=int, default=64, help="queries whose pairs are batched together")
    ap.add_argument("--limit-queries", type=int, default=None, help="debug: only the first N queries per split")
    ap.add_argument("--teacher", default=None, help="minilm-distilled only: restrict to this teacher")
    args = ap.parse_args()
    cfg = load_config(args)
    name = cfg["cheap"]["model"]
    ccfg = {k: v for k, v in cfg["cheap"].items() if k != "distill"}
    if name != "minilm-distilled":
        model = get_cheap(name, ccfg)
        return run_model(cfg, args, model, features_dir_for(P(cfg, "features"), name), ROOT / cfg["paths"]["runs"] / "features" / "manifest.json", {})
    rc = 0
    for t in ([args.teacher] if args.teacher else teachers(cfg)):
        mdir = distilled_model_dir(P(cfg, "models"), t)
        if not (mdir / "meta.json").exists():
            log(f"no distilled cheap model for {t} at {mdir} (run scripts/06b_distill_cheap.py --teacher {t}); skipping")
            rc = 1
            continue
        meta = read_json(mdir / "meta.json")
        model = get_cheap("minilm-distilled", {**ccfg, "path": str(mdir), "name": f"minilm-distilled:{t}"})
        fdir = features_dir_for(P(cfg, "features"), name, t)
        rc = max(rc, run_model(cfg, args, model, fdir, ROOT / cfg["paths"]["runs"] / "features_distilled" / t / "manifest.json", {"teacher": t, "distilled_meta_sha256": sha256_file(mdir / "meta.json"), "distilled_from": meta.get("base_model")}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
