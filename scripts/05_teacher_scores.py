"""Full teacher score cache for every (query, candidate) pair (node N07), plus the
throughput/batching pilot (node N05) and the determinism audit (Sec. 11.3).

  python scripts/05_teacher_scores.py --teacher qwen3-4b --pilot                 # N05: sweep, measure, recommend
  python scripts/05_teacher_scores.py --teacher qwen3-4b --apply-recommendation  # print the YAML keys to set
  python scripts/05_teacher_scores.py --teacher qwen3-4b --splits cal,test       # N07: cache (resumable)
  python scripts/05_teacher_scores.py --teacher qwen3-4b --audit                 # determinism re-scoring

Caching scores the pairs of --chunk-queries queries (default 32) with one ``score_pairs`` call
(length-sorted batches across queries in ``fast`` mode; fixed-shape microbatches in ``canonical``
mode, see src/rscrank/models.py) and checkpoints after every chunk, so a killed job resumes at the
last finished chunk.  The candidates-file hash guard is unchanged: an existing cache built from
different candidates is never overwritten silently.

Outputs: data/teacher/<teacher>/<domain>/<split>.npz  (score [Q,N] f32, ntok [Q,N] i32, valid; meta:
         teacher, cand_sha256, settings), runs/teacher/<teacher>/manifest.json,
         runs/pilot/throughput_<teacher>.json, runs/teacher/<teacher>/determinism.json
Debug: --limit-queries N scores only the first N queries of each split (store marked incomplete).
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, domains, load_config, log  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank.data import domain_list  # noqa: E402
from rscrank.io import ensure_dir, read_json, sha256_file, write_json  # noqa: E402
from rscrank.models import get_teacher  # noqa: E402
from rscrank.store import load_candidates, load_docstore, load_store, save_store, store_meta  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _torch():
    try:
        import torch

        return torch
    except Exception:  # noqa: BLE001
        return None


def _cuda_on(model) -> bool:
    t = _torch()
    return bool(t is not None and t.cuda.is_available() and str(getattr(model, "device", "")).startswith("cuda"))


def _sync(model) -> None:
    if _cuda_on(model):
        _torch().cuda.synchronize()


def _is_oom(e: BaseException) -> bool:
    t = _torch()
    if t is not None and isinstance(e, getattr(t.cuda, "OutOfMemoryError", ())):
        return True
    return isinstance(e, MemoryError) or "out of memory" in str(e).lower()


def _free(model) -> None:
    gc.collect()
    if _cuda_on(model):
        _torch().cuda.empty_cache()


def settings_of(model) -> Dict[str, Any]:
    return model.settings() if hasattr(model, "settings") else {"name": getattr(model, "name", "?")}


def settings_key(s: Dict[str, Any]) -> Dict[str, Any]:
    """The part of the settings that defines the scoring function (for resume/audit comparisons)."""
    keys = ("class", "name", "mode", "batch_size", "max_tokens_per_batch", "max_length", "dtype", "attn", "instruction", "query_max_length")
    return {k: s.get(k) for k in keys if k in s}


def rel(p: Path) -> str:
    """Repo-relative key for manifests; absolute when paths.* point outside the repo."""
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _as_stored(scores) -> np.ndarray:
    """Scores rounded to the cache's storage dtype (float32), as float64."""
    return np.asarray(scores, dtype=np.float32).astype(np.float64)


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _pairs_for(rows, docstore, N):
    pairs, where = [], []
    for i, r in rows:
        for j, x in enumerate(r["docs"][:N]):
            t = docstore[x["docid"]]
            pairs.append((r["query"], (t["title"], t["text"])))
            where.append((i, j))
    return pairs, where


# --------------------------------------------------------------------------- #
# N07: cache
# --------------------------------------------------------------------------- #
def score_split(model, rows, docstore, N, out: Path, chunk_q: int, log_prefix: str, cand_hash: str, meta_extra: Dict[str, Any]):
    Q = len(rows)
    ensure_dir(out.parent)
    part = out.with_suffix(".partial.npz")
    score = np.full((Q, N), np.nan, dtype=np.float32)
    ntok = np.zeros((Q, N), dtype=np.int32)
    s_now = settings_of(model)
    key_now = settings_key(s_now)
    start, notes = 0, {}
    if part.exists():
        with np.load(part, allow_pickle=False) as zf:  # closed before os.replace (Windows file locking)
            z = {k: zf[k] for k in zf.files}
        pmeta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
        if z["score"].shape != (Q, N) or pmeta.get("cand_sha256") not in (None, cand_hash):
            stale = part.with_name(part.stem + f".stale-{int(time.time())}.npz")
            os.replace(part, stale)
            log(f"{log_prefix}: partial checkpoint belongs to other candidates; moved to {stale.name}, starting fresh")
        else:
            start = int(z["progress"])
            score[:start] = z["score"][:start]
            ntok[:start] = z["ntok"][:start]
            if pmeta.get("settings_key") not in (None, key_now):
                log(f"{log_prefix}: WARNING resuming with different teacher settings {pmeta.get('settings_key')} -> {key_now}")
                notes["resumed_with_different_settings"] = {"before": pmeta.get("settings_key"), "after": key_now, "at_query": start}
            log(f"{log_prefix}: resuming at query {start}/{Q}")
    t0, n_pairs = time.time(), 0
    pmeta_json = json.dumps({"cand_sha256": cand_hash, "settings_key": key_now})
    for s in range(start, Q, chunk_q):
        e = min(Q, s + chunk_q)
        pairs, where = _pairs_for([(i, rows[i]) for i in range(s, e)], docstore, N)
        if pairs:
            sc, nt = model.score_pairs(pairs, return_ntok=True)
            ii = np.fromiter((w[0] for w in where), dtype=np.int64, count=len(where))
            jj = np.fromiter((w[1] for w in where), dtype=np.int64, count=len(where))
            score[ii, jj] = np.asarray(sc, dtype=np.float32)
            ntok[ii, jj] = np.asarray(nt, dtype=np.int32)
            n_pairs += len(pairs)
        _atomic_savez(part, score=score, ntok=ntok, progress=np.array(e), meta_json=np.array(pmeta_json))
        dt = max(1e-9, time.time() - t0)
        rate = n_pairs / dt
        remaining = sum(min(N, len(rows[i]["docs"])) for i in range(e, Q))
        eta = remaining / rate / 60 if rate > 0 else float("nan")
        log(f"{log_prefix}: {e}/{Q} queries, {n_pairs} pairs this session, {rate:.2f} pairs/s, ETA {eta:.1f} min")
    qids = [r["qid"] for r in rows]
    docids = [[x["docid"] for x in r["docs"]][:N] for r in rows]
    meta = {"teacher": model.name, "cand_sha256": cand_hash, "settings": s_now, **meta_extra, **notes}
    save_store(out, qids, docids, {"score": score, "ntok": ntok}, N, meta)
    part.unlink(missing_ok=True)
    return n_pairs, time.time() - t0, notes


def cache(model, cfg, args, tname: str) -> int:
    cand_dir = P(cfg, "candidates")
    teacher_dir = P(cfg, "teacher") / tname
    N = cfg["candidates"]["N"]
    chunk_q = max(1, min(args.chunk_queries, args.ckpt or args.chunk_queries))
    doms = domains(cfg, args) if args.synthetic else domain_list(cfg, only=args.domains)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    man_path = ROOT / cfg["paths"]["runs"] / "teacher" / tname / "manifest.json"
    man = read_json(man_path) if man_path.exists() else {"teacher": tname, "model": model.name, "files": {}, "pairs": 0, "seconds": 0.0}
    man["settings"] = settings_of(model)
    for d in doms:
        if not (cand_dir / d / "docstore.jsonl").exists():
            log(f"{d}: no candidates; skipping")
            continue
        docstore = load_docstore(cand_dir / d / "docstore.jsonl")
        for split in splits:
            cf = cand_dir / d / f"{split}.jsonl"
            out = teacher_dir / d / f"{split}.npz"
            if not cf.exists():
                continue
            cand_hash = sha256_file(cf)
            if out.exists():
                meta = store_meta(out)
                lim = meta.get("limit_queries")
                if meta.get("cand_sha256") == cand_hash and (lim is None or lim == args.limit_queries):
                    old_key = settings_key(meta.get("settings", {}))
                    if old_key and old_key != settings_key(settings_of(model)):
                        log(f"skip existing {out} -- WARNING it was cached with {old_key}, the config now says {settings_key(settings_of(model))}; delete the file explicitly to re-cache (N07 step 4)")
                    else:
                        log(f"skip existing {out}")
                    continue
                if lim is None and meta.get("cand_sha256") != cand_hash:
                    raise SystemExit(f"{out} was built from different candidates ({cf} changed). Teacher caches are expensive: delete the stale file explicitly if the candidate change is intended.")
                log(f"{out} is a --limit-queries debug store; recomputing")
            rows = load_candidates(cf)
            extra: Dict[str, Any] = {}
            if args.limit_queries:
                rows = rows[: args.limit_queries]
                extra = {"limit_queries": args.limit_queries, "complete": False}
            n_pairs, secs, notes = score_split(model, rows, docstore, N, out, chunk_q, f"{tname}/{d}/{split}", cand_hash, extra)
            key = rel(out)
            man["files"][key] = {"sha256": sha256_file(out), "queries": len(rows), "pairs": n_pairs, "seconds": round(secs, 1), "settings_key": settings_key(settings_of(model)), **extra, **notes}
            man["pairs"] += n_pairs
            man["seconds"] += secs
            write_json(man_path, man)
    # coverage check over every existing store of this teacher
    missing, incomplete, keys = 0, [], set()
    for d in doms:
        for split in ("train", "dev", "cal", "test"):
            f = teacher_dir / d / f"{split}.npz"
            if f.exists():
                st = load_store(f)
                missing += int(np.sum(np.isnan(st["score"][st["valid"]])))
                meta = store_meta(f)
                if meta.get("limit_queries") is not None:
                    incomplete.append(rel(f))
                if meta.get("settings"):
                    keys.add(json.dumps(settings_key(meta["settings"]), sort_keys=True))
    man["missing_scores"] = missing
    man["incomplete_debug_stores"] = incomplete
    man["n_distinct_settings"] = len(keys)
    if len(keys) > 1:
        log(f"WARNING: stores of {tname} were built with {len(keys)} different settings; see manifest files[*].settings_key")
    man["ok"] = missing == 0 and not incomplete
    write_json(man_path, man)
    log(f"teacher cache manifest -> {man_path}; missing={missing} incomplete={len(incomplete)}")
    return 0 if missing == 0 else 1


# --------------------------------------------------------------------------- #
# invariance measurement (pilot and audit)
# --------------------------------------------------------------------------- #
def _topk_flips(ids_by_q: Dict[Any, List[str]], a_by_q: Dict[Any, np.ndarray], b_by_q: Dict[Any, np.ndarray], k: int) -> int:
    flips = 0
    for q, ids in ids_by_q.items():
        kk = min(k, len(ids) - 1)
        if kk < 1:
            continue
        oa = core.teacher_order(a_by_q[q], ids)[:kk]
        ob = core.teacher_order(b_by_q[q], ids)[:kk]
        flips += int(not np.array_equal(oa, ob))
    return flips


def _compare(a: np.ndarray, b: np.ndarray, groups: List[Tuple[Any, int]], ids: List[str], k: int) -> Dict[str, Any]:
    """a, b aligned pair scores; groups[i] = query key of pair i."""
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    by_q: Dict[Any, List[int]] = {}
    for i, g in enumerate(groups):
        by_q.setdefault(g, []).append(i)
    flips = _topk_flips({q: [ids[i] for i in ix] for q, ix in by_q.items()}, {q: np.asarray(a)[ix] for q, ix in by_q.items()}, {q: np.asarray(b)[ix] for q, ix in by_q.items()}, k)
    return {"pairs": int(len(d)), "queries": len(by_q), "max_abs_diff": float(d.max()) if len(d) else None, "mean_abs_diff": float(d.mean()) if len(d) else None, "bit_identical_pairs": int(np.sum(d == 0)), "topk_order_changed_queries": flips, "k": k}


def _score_in_composition(model, pairs: List, filler: List, seed: int) -> np.ndarray:
    """Scores of ``pairs`` when scored inside a shuffled mix with ``filler`` (other batch-mates, other order)."""
    mixed = list(pairs) + list(filler)
    order = np.random.default_rng(seed).permutation(len(mixed))
    sc = model.score_pairs([mixed[i] for i in order])
    back = np.empty(len(mixed), dtype=np.float64)
    back[order] = np.asarray(sc, dtype=np.float64)
    return back[: len(pairs)]


# --------------------------------------------------------------------------- #
# N05: pilot
# --------------------------------------------------------------------------- #
def _pilot_pool(cfg, cand_dir, N, max_rows=300):
    pool = []  # (domain, qid, docid, query, (title, text), n_words)
    for d in cfg["domains"]["main"]:
        cf = cand_dir / d / "cal.jsonl"
        if not cf.exists():
            continue
        docstore = load_docstore(cand_dir / d / "docstore.jsonl")
        for r in load_candidates(cf)[:max_rows]:
            for x in r["docs"][:N]:
                t = docstore[x["docid"]]
                pool.append((d, r["qid"], x["docid"], r["query"], (t["title"], t["text"]), len((t["title"] + " " + t["text"]).split())))
    return pool


def pilot(model, cfg, cand_dir, tname: str, args) -> int:
    """Stratified throughput/memory/batch-invariance sweep over mode x batch_size; recommendation rule of N05."""
    rng = np.random.default_rng(cfg["seed"])
    N = cfg["candidates"]["N"]
    k = int(cfg["protocol"]["k"])
    tcfg = cfg["teachers"].get(tname, {}) or {}
    pool = _pilot_pool(cfg, cand_dir, N)
    if not pool:
        raise SystemExit("no candidates yet; run 03_build_candidates first")
    lens = np.array([p[5] for p in pool])
    q1, q2 = np.quantile(lens, [1 / 3, 2 / 3])
    strata_all = {"short": [p for p in pool if p[5] <= q1], "medium": [p for p in pool if q1 < p[5] <= q2], "long": [p for p in pool if p[5] > q2]}
    n_each = max(1, int(cfg["budget"]["pilot_pairs"]) // 3)
    strata = {}
    for name, ps in strata_all.items():
        if ps:
            idx = rng.choice(len(ps), size=min(n_each, len(ps)), replace=False)
            strata[name] = [ps[i] for i in sorted(idx)]
    # batch-invariance set: ~inv_queries queries x ~inv_pairs/inv_queries candidates, plus filler pairs
    by_q: Dict[Tuple[str, str], List] = {}
    for p in pool:
        by_q.setdefault((p[0], p[1]), []).append(p)
    qkeys = list(by_q)
    n_iq = min(args.pilot_queries, len(qkeys))
    per_q = max(2, int(math.ceil(args.pilot_invariance_pairs / max(1, n_iq))))
    inv = []
    for qi in rng.choice(len(qkeys), size=n_iq, replace=False):
        cands = by_q[qkeys[qi]]
        sel = rng.choice(len(cands), size=min(per_q, len(cands)), replace=False)
        inv += [cands[j] for j in sorted(sel)]
    inv_set = {(p[0], p[1], p[2]) for p in inv}
    rest = [p for ps in strata.values() for p in ps if (p[0], p[1], p[2]) not in inv_set]
    filler = [rest[i] for i in rng.permutation(len(rest))[: len(inv)]]
    inv_pairs = [(p[3], p[4]) for p in inv]
    inv_groups = [(p[0], p[1]) for p in inv]
    inv_ids = [p[2] for p in inv]
    filler_pairs = [(p[3], p[4]) for p in filler]

    cuda = _cuda_on(model)
    torch = _torch()
    total_gb = None
    if cuda:
        total_gb = torch.cuda.get_device_properties(torch.device(model.device)).total_memory / 2**30
    vram_limit = args.vram_limit_gb if args.vram_limit_gb else (0.875 * total_gb if total_gb else None)
    orig = (getattr(model, "mode", "fast"), getattr(model, "batch_size", 1), getattr(model, "max_tokens_per_batch", None))
    mtpb_fast = tcfg.get("max_tokens_per_batch", orig[2])
    grid = [(m, int(b)) for m in ("fast", "canonical") for b in args.pilot_batch_sizes.split(",") if b.strip()]
    records = []
    for mode, bs in grid:
        model.mode, model.batch_size = mode, bs
        model.max_tokens_per_batch = mtpb_fast if mode == "fast" else None
        rec: Dict[str, Any] = {"mode": mode, "batch_size": bs, "max_tokens_per_batch": model.max_tokens_per_batch, "oom": False}
        try:
            warm = [(p[3], p[4]) for ps in strata.values() for p in ps[: max(1, min(bs, 4))]]
            model.score_pairs(warm)  # warm-up (kernels, allocator)
            _sync(model)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            if hasattr(model, "reset_counters"):
                model.reset_counters()
            per, tot_n, tot_t = {}, 0, 0.0
            for name, ps in strata.items():
                t0 = time.perf_counter()
                model.score_pairs([(p[3], p[4]) for p in ps])
                _sync(model)
                dt = time.perf_counter() - t0
                per[name] = {"pairs": len(ps), "seconds": round(dt, 3), "pairs_per_s": round(len(ps) / max(dt, 1e-9), 3), "mean_words": float(np.mean([p[5] for p in ps]))}
                tot_n += len(ps)
                tot_t += dt
            rec["strata"] = per
            rec["pairs"] = tot_n
            rec["seconds"] = round(tot_t, 3)
            rec["pairs_per_s"] = round(tot_n / max(tot_t, 1e-9), 3)
            if cuda:
                rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if hasattr(model, "counters"):
                c = dict(model.counters)
                rec["counters"] = c
                rec["padding_efficiency"] = round(c["real_tokens"] / max(1, c["padded_tokens"]), 4)
            a = _as_stored(model.score_pairs(inv_pairs))
            b = _as_stored(_score_in_composition(model, inv_pairs, filler_pairs, cfg["seed"] + bs))
            rec["invariance"] = _compare(a, b, inv_groups, inv_ids, k)
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e):
                raise
            rec.update(oom=True, error=f"{type(e).__name__}: {str(e)[:300]}")
            log(f"pilot {mode}/bs={bs}: OOM ({rec['error'][:120]}); continuing")
        _free(model)
        rec["fits"] = (not rec["oom"]) and (vram_limit is None or rec.get("peak_mem_gb") is None or rec["peak_mem_gb"] <= vram_limit)
        records.append(rec)
        if not rec["oom"]:
            log(f"pilot {mode:9s} bs={bs:3d}: {rec['pairs_per_s']:8.2f} pairs/s peak={rec.get('peak_mem_gb')} GB invariance max|d|={rec['invariance']['max_abs_diff']:.3g} topk-flips={rec['invariance']['topk_order_changed_queries']}/{rec['invariance']['queries']}")
    model.mode, model.batch_size, model.max_tokens_per_batch = orig

    ran = [r for r in records if not r["oom"]]
    fast_fit = [r for r in ran if r["mode"] == "fast" and r["fits"]]
    can_fit = [r for r in ran if r["mode"] == "canonical" and r["fits"]]
    best_fast = max(fast_fit, key=lambda r: r["pairs_per_s"]) if fast_fit else None
    can_stable = [r for r in can_fit if r["invariance"]["max_abs_diff"] == 0.0]
    best_can = max(can_stable, key=lambda r: r["pairs_per_s"]) if can_stable else None
    if best_can is not None and (best_fast is None or best_can["pairs_per_s"] >= 0.67 * best_fast["pairs_per_s"]):
        chosen, rule = best_can, "canonical: bit-stable (max|diff| == 0) and throughput >= 0.67 x best fast"
    elif best_fast is not None:
        chosen, rule = best_fast, "fastest fast setting that fits in memory" + ("" if best_can is None else f" (best bit-stable canonical {best_can['pairs_per_s']:.2f} < 0.67 x {best_fast['pairs_per_s']:.2f})")
    elif can_fit:
        chosen, rule = max(can_fit, key=lambda r: r["pairs_per_s"]), "no fast setting fits; fastest canonical setting that fits"
    else:
        chosen, rule = None, "no setting fits in memory"
    v = float(chosen["pairs_per_s"]) if chosen else 0.0
    core_pairs = None
    cm = ROOT / cfg["paths"]["runs"] / "candidates" / "manifest.json"
    if cm.exists():
        cmd = read_json(cm)
        core_pairs = int(sum(x["n_queries"] for x in cmd["domains"].values()) * cfg["candidates"]["N"])
    res: Dict[str, Any] = {
        "teacher": tname,
        "model": model.name,
        "device": str(getattr(model, "device", "cpu")),
        "settings_base": settings_of(model),
        "sample": {"pairs": sum(len(v_) for v_ in strata.values()), "strata_sizes": {k_: len(v_) for k_, v_ in strata.items()}, "word_tertiles": [float(q1), float(q2)], "pool_pairs": len(pool)},
        "invariance_protocol": {"queries": n_iq, "pairs": len(inv_pairs), "filler_pairs": len(filler_pairs), "composition_a": "score_pairs(invariance pairs, query order)", "composition_b": "score_pairs(shuffle(invariance pairs + filler pairs))", "k": k},
        "settings": records,
        "recommendation": ({"mode": chosen["mode"], "batch_size": chosen["batch_size"], "max_tokens_per_batch": chosen["max_tokens_per_batch"]} if chosen else None),
        "recommendation_rule": rule,
        "fast_bit_stable": bool(all(r["invariance"]["max_abs_diff"] == 0.0 for r in ran if r["mode"] == "fast")) if any(r["mode"] == "fast" for r in ran) else None,
        "canonical_bit_stable": bool(all(r["invariance"]["max_abs_diff"] == 0.0 for r in ran if r["mode"] == "canonical")) if any(r["mode"] == "canonical" for r in ran) else None,
        "vram_limit_gb": vram_limit,
        "gpu_total_gb": round(total_gb, 2) if total_gb else None,
        "strata": chosen["strata"] if chosen else {},
        "v_measured_pairs_per_s": round(v, 3),
        "v_min_pairs_per_s": cfg["budget"]["v_min_pairs_per_s"],
        "core_pairs": core_pairs,
        "projected_hours_core": round(core_pairs / v / 3600, 2) if (core_pairs and v > 0) else None,
        "budget_hours_core": cfg["budget"]["teacher_core_hours"],
        "ok": bool(v >= float(cfg["budget"]["v_min_pairs_per_s"])),
    }
    if chosen and chosen.get("peak_mem_gb") is not None:
        res["peak_mem_gb"] = chosen["peak_mem_gb"]
    out = ensure_dir(ROOT / cfg["paths"]["runs"] / "pilot") / f"throughput_{tname}.json"
    write_json(out, res)
    log(f"pilot: recommended {res['recommendation']} ({rule}); {v:.2f} pairs/s (min {cfg['budget']['v_min_pairs_per_s']}), projected core hours {res['projected_hours_core']}, fast bit-stable={res['fast_bit_stable']} -> {out}")
    if args.apply_recommendation:
        print_recommendation(res, tname)
    return 0 if res["ok"] else 1


def print_recommendation(res: Dict[str, Any], tname: str) -> None:
    rec = res.get("recommendation")
    print(f"# recommended by runs/pilot/throughput_{tname}.json -- rule: {res.get('recommendation_rule')}")
    print("# configs/main.yaml keys to set (this script does not edit configs; graphctl/the agent applies them):")
    if not rec:
        print("# (no recommendation: no setting fitted in memory)")
        return
    for key in ("mode", "batch_size", "max_tokens_per_batch"):
        if rec.get(key) is not None:
            print(f"teachers.{tname}.{key}: {rec[key]}")


# --------------------------------------------------------------------------- #
# determinism audit
# --------------------------------------------------------------------------- #
def audit(model, cfg, cand_dir, teacher_dir, tname: str, args) -> int:
    """Re-score ~teachers.determinism_pairs cached cal pairs under a different batch composition."""
    rng = np.random.default_rng(cfg["seed"] + 1)
    N = cfg["candidates"]["N"]
    k = int(cfg["protocol"]["k"])
    n_target = int(cfg["teachers"].get("determinism_pairs", 1000))
    avail = []
    doms = list(cfg["domains"]["main"]) + list(cfg["domains"].get("extra", []) or [])
    stores, rows_by, docstores, metas = {}, {}, {}, {}
    for d in doms:
        f = teacher_dir / d / "cal.npz"
        if not f.exists():
            continue
        stores[d] = load_store(f)
        metas[d] = store_meta(f)
        rows_by[d] = {r["qid"]: r for r in load_candidates(cand_dir / d / "cal.jsonl")}
        docstores[d] = load_docstore(cand_dir / d / "docstore.jsonl")
        avail += [(d, qi) for qi in range(len(stores[d]["qids"])) if stores[d]["valid"][qi].sum() > 1]
    res: Dict[str, Any] = {"teacher": tname, "settings": settings_of(model)}
    s_key = settings_key(res["settings"])
    res["mode"] = s_key.get("mode")
    if not avail:
        res.update(queries_checked=0, pairs_checked=0, max_abs_diff=None, mean_abs_diff=None, bit_stable=None, topk_order_changed_queries=0, ok=False, reason="no cached cal stores")
        write_json(ROOT / cfg["paths"]["runs"] / "teacher" / tname / "determinism.json", res)
        log("determinism audit: no cached cal stores")
        return 1
    n_q = min(len(avail), max(1, int(math.ceil(n_target / N))))
    chosen = [avail[i] for i in sorted(rng.choice(len(avail), size=n_q, replace=False))]
    pairs, cached, groups, ids = [], [], [], []
    for d, qi in chosen:
        st = stores[d]
        r = rows_by[d][st["qids"][qi]]
        n = int(st["valid"][qi].sum())
        ds = list(st["docids"][qi, :n])
        assert ds == [x["docid"] for x in r["docs"]][:n], "cache/candidates docid mismatch"
        for j, did in enumerate(ds):
            t = docstores[d][did]
            pairs.append((r["query"], (t["title"], t["text"])))
            cached.append(float(st["score"][qi, j]))
            groups.append((d, qi))
            ids.append(did)
    cached = np.asarray(cached, dtype=np.float64)
    # composition 1: all selected pairs, shuffled, in one cross-query score_pairs call (different chunk mates than the cache)
    perm = rng.permutation(len(pairs))
    sc = model.score_pairs([pairs[i] for i in perm])
    re1 = np.empty(len(pairs), dtype=np.float64)
    re1[perm] = _as_stored(sc)  # compare in the cache's storage precision (float32)
    comp1 = _compare(cached, re1, groups, ids, k)
    # composition 2: per-query calls in reversed document order + single-pair calls for 4 documents per query
    re2 = re1.copy()
    single_idx = []
    for g in dict.fromkeys(groups):
        ix = [i for i, gg in enumerate(groups) if gg == g]
        q = pairs[ix[0]][0]
        rev = ix[::-1]
        s2 = model.score(q, [pairs[i][1] for i in rev])
        re2[rev] = _as_stored(s2)
        for i in ix[:4]:
            single_idx.append((i, float(_as_stored(model.score(q, [pairs[i][1]]))[0])))
    comp2 = _compare(cached, re2, groups, ids, k)
    single_d = np.array([abs(v - cached[i]) for i, v in single_idx]) if single_idx else np.zeros(0)
    all_d = np.concatenate([np.abs(re1 - cached), np.abs(re2 - cached), single_d])
    flips = max(comp1["topk_order_changed_queries"], comp2["topk_order_changed_queries"])
    cache_keys = {d: settings_key(metas[d].get("settings", {})) for d in metas}
    res.update(
        queries_checked=len(chosen),
        pairs_checked=int(len(all_d)),
        max_abs_diff=float(all_d.max()),
        mean_abs_diff=float(all_d.mean()),
        bit_stable=bool(all_d.max() == 0.0),
        topk_order_changed_queries=int(flips),
        k=k,
        compositions={"cross_query_shuffled": comp1, "per_query_reversed": comp2, "single_pair_max_abs_diff": float(single_d.max()) if len(single_d) else None},
        cache_settings=cache_keys,
        settings_match_cache=all(v == s_key for v in cache_keys.values() if v),
    )
    res["ok"] = bool(res["queries_checked"] > 0 and flips <= max(1, 0.02 * res["queries_checked"]))
    write_json(ROOT / cfg["paths"]["runs"] / "teacher" / tname / "determinism.json", res)
    log(f"determinism audit: mode={res['mode']} queries={res['queries_checked']} max|d|={res['max_abs_diff']:.3g} bit_stable={res['bit_stable']} topk-changed={flips} ok={res['ok']}")
    if not res["settings_match_cache"]:
        log("WARNING: audit settings differ from the settings recorded in the cached stores")
    return 0 if res["ok"] else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = base_parser("teacher scores")
    ap.add_argument("--teacher", default=None, help="teacher key from configs (default: main teacher)")
    ap.add_argument("--pilot", action="store_true", help="N05 sweep: mode x batch size -> runs/pilot/throughput_<teacher>.json")
    ap.add_argument("--apply-recommendation", action="store_true", help="print the YAML keys recommended by the pilot (no config is written)")
    ap.add_argument("--audit", action="store_true", help="determinism re-scoring of cached cal pairs")
    ap.add_argument("--splits", default="train,dev,cal,test")
    ap.add_argument("--chunk-queries", type=int, default=32, help="queries scored together (and checkpointed) per score_pairs call")
    ap.add_argument("--ckpt", type=int, default=None, help="deprecated: checkpoint at least every this many queries")
    ap.add_argument("--limit-queries", type=int, default=None, help="debug: only the first N queries per split")
    ap.add_argument("--device", default=None, help="override teachers.<t>.device (e.g. cpu)")
    ap.add_argument("--pilot-batch-sizes", default="8,16,32")
    ap.add_argument("--pilot-queries", type=int, default=10, help="queries in the batch-invariance set")
    ap.add_argument("--pilot-invariance-pairs", type=int, default=256)
    ap.add_argument("--vram-limit-gb", type=float, default=None, help="settings above this peak VRAM do not 'fit' (default 0.875 x GPU memory)")
    args = ap.parse_args()
    cfg = load_config(args)
    tname = args.teacher or cfg["teachers"]["main"]
    if args.apply_recommendation and not args.pilot:
        pm = ROOT / cfg["paths"]["runs"] / "pilot" / f"throughput_{tname}.json"
        if not pm.exists():
            raise SystemExit(f"{pm} missing: run --pilot first")
        print_recommendation(read_json(pm), tname)
        return 0
    tcfg = dict(cfg["teachers"].get(tname, {}) or {})
    if args.device:
        tcfg["device"] = args.device
    model = get_teacher(tname, tcfg)
    cand_dir = P(cfg, "candidates")
    teacher_dir = P(cfg, "teacher") / tname
    if args.pilot:
        return pilot(model, cfg, cand_dir, tname, args)
    if args.audit:
        return audit(model, cfg, cand_dir, teacher_dir, tname, args)
    return cache(model, cfg, args, tname)


if __name__ == "__main__":
    sys.exit(main())
