"""Datasets: BEIR loading/downloading, training questions, de-duplication and
frozen splits (Sec. 12).

Directory contract (relative to repo root, configurable in configs/main.yaml)::

    data/raw/beir/<domain>/corpus.jsonl, queries.jsonl, qrels/{train,dev,test}.tsv
    data/raw/train_questions/<domain>.jsonl          # {qid, text, source}
    data/splits/<domain>/{train,dev,cal,test}.jsonl   # {qid, text, source}
    data/features/<domain>/<split>.npz                 # frozen MiniLM features (cheap.model: minilm)
    data/features_distilled/<teacher>/<domain>/<split>.npz   # cheap.model: minilm-distilled (one model per teacher,
    models/cheap_distilled/<teacher>/                  #   HF format + meta.json, written by scripts/06b_distill_cheap.py)

Domains: configs/main.yaml domains.main + domains.cross + domains.extra (optional web-query
domains, e.g. [msmarco]).  The BEIR pool that is split into cal/test is qrels/<pool split>.tsv with
pool split = domains.test_pool_split[domain] (``test`` except BEIR msmarco, which evaluates on ``dev``).

Splits are frozen by seed and recorded with hashes in runs/data/splits_manifest.json.
Nothing in this module looks at teacher scores or retrieval results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .io import ensure_dir, normalize_text, read_jsonl, write_jsonl

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
BEIR_HF = "BeIR/{name}"
NQ_OPEN_HF = ("google-research-datasets/nq_open", "nq_open/train-00000-of-00001.parquet")
BEIR_DOMAINS = ("nq", "hotpotqa", "fiqa", "scifact", "msmarco")
# BEIR pool sizes as published (verified 2026-09-20; docs/FACTS.md).  msmarco: the dev pool (BEIR's evaluation split).
BEIR_TEST_QUERIES = {"nq": 3452, "hotpotqa": 7405, "fiqa": 648, "scifact": 300, "msmarco": 6980}
BEIR_CORPUS_SIZE = {"nq": 2681468, "hotpotqa": 5233329, "fiqa": 57638, "scifact": 5183, "msmarco": 8841823}
DEFAULT_POOL_SPLIT = {"msmarco": "dev"}

__all__ = [
    "download_beir",
    "download_beir_info",
    "load_beir",
    "load_qrels",
    "load_train_questions",
    "dedup",
    "make_splits",
    "domain_list",
    "pool_split",
    "is_cross_domain_only",
    "load_pool_qrels",
    "cheap_model_for",
    "features_dir_for",
    "distilled_model_dir",
    "BEIR_DOMAINS",
    "BEIR_TEST_QUERIES",
    "BEIR_CORPUS_SIZE",
]


# --------------------------------------------------------------------------- #
# Config helpers (domains.main / cross / extra, pool split)
# --------------------------------------------------------------------------- #
def domain_list(cfg: Dict[str, Any], include_cross: bool = True, include_extra: bool = True, only: Optional[str] = None) -> List[str]:
    """main + (cross) + (extra) domains, de-duplicated, config order; ``only`` = comma-separated filter."""
    dcfg = cfg.get("domains", {})
    ds = list(dcfg.get("main", []) or [])
    if include_cross:
        ds += list(dcfg.get("cross", []) or [])
    if include_extra:
        ds += list(dcfg.get("extra", []) or [])
    ds = list(dict.fromkeys(ds))
    if only:
        want = {d.strip() for d in str(only).split(",") if d.strip()}
        ds = [d for d in ds if d in want]
    return ds


def pool_split(cfg: Optional[Dict[str, Any]], domain: str) -> str:
    """BEIR qrels split whose queries form the cal/test pool (domains.test_pool_split)."""
    m = ((cfg or {}).get("domains", {}) or {}).get("test_pool_split", {}) or {}
    return str(m.get(domain, DEFAULT_POOL_SPLIT.get(domain, "test")))


def is_cross_domain_only(cfg: Dict[str, Any], domain: str) -> bool:
    sc = (cfg.get("splits", {}) or {}).get(domain, {}) or {}
    return bool(sc.get("cross_domain_only", False)) or domain in (cfg.get("domains", {}).get("cross", []) or [])


def cheap_model_for(cfg: Dict[str, Any], teacher: Optional[str] = None) -> str:
    """The cheap model used with ``teacher``: ``cheap.model`` unless ``cheap.model_by_teacher`` overrides it.

    The distillation lever (D-DISTILL) is decided per teacher on development data, because the distilled
    model regresses onto that teacher's scores; a teacher for which the lever missed the pre-registered
    gain keeps the frozen model while another may adopt the distilled one."""
    c = cfg.get("cheap", {}) or {}
    by = c.get("model_by_teacher") or {}
    return str(by.get(teacher, c.get("model", "minilm"))) if teacher else str(c.get("model", "minilm"))


def features_dir_for(features_dir: Path, cheap_model: str, teacher: Optional[str] = None) -> Path:
    """Directory holding <domain>/<split>.npz cheap features for ``cheap_model``.

    minilm (frozen) -> paths.features (data/features); minilm-distilled -> a per-teacher sibling
    directory <paths.features>/../features_distilled/<teacher> (the distilled model depends on the teacher)."""
    features_dir = Path(features_dir)
    if cheap_model == "minilm-distilled":
        if not teacher:
            raise ValueError("minilm-distilled features are per teacher: pass teacher=")
        return features_dir.parent / "features_distilled" / teacher
    return features_dir


def distilled_model_dir(models_dir: Path, teacher: str) -> Path:
    """models/cheap_distilled/<teacher> (HF format + meta.json)."""
    return Path(models_dir) / "cheap_distilled" / teacher


def load_pool_qrels(cfg: Optional[Dict[str, Any]], domain: str, raw_dir: Path) -> Dict[str, Dict[str, int]]:
    """qrels of the cal/test pool (qrels/<pool split>.tsv), e.g. msmarco -> dev."""
    s = pool_split(cfg, domain)
    return load_beir(domain, raw_dir, splits=(s,), load_corpus=False)[2].get(s, {})


# --------------------------------------------------------------------------- #
# BEIR
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_snapshot_sha(path: str) -> Optional[str]:
    parts = Path(path).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def _download_ukp(domain: str, raw_dir: Path, force: bool) -> Dict[str, Any]:
    url = BEIR_URL.format(name=domain)
    zpath = Path(raw_dir) / f"{domain}.zip"
    if force or not zpath.exists():
        part = zpath.with_name(zpath.name + ".part")
        print(f"[data] downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        os.replace(part, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(Path(raw_dir))
    out = Path(raw_dir) / domain
    if not (out / "corpus.jsonl").exists():
        raise RuntimeError("zip did not contain corpus.jsonl")
    return {"source": "ukp", "url": url, "zip_sha256": _sha256(zpath), "zip_bytes": zpath.stat().st_size}


def _parquet_rows(path: str, columns: List[str]) -> Iterable[Dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    cols = [c for c in columns if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=50_000, columns=cols):
        yield from batch.to_pylist()


def _download_hf(domain: str, out: Path) -> Dict[str, Any]:
    """BeIR/<name> (corpus/queries parquet) + BeIR/<name>-qrels (tsv) via huggingface_hub (no `datasets`)."""
    from huggingface_hub import HfApi, hf_hub_download

    repo, qrepo = BEIR_HF.format(name=domain), BEIR_HF.format(name=domain) + "-qrels"
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    shas = {}
    for kind in ("corpus", "queries"):
        parts = sorted(f for f in files if f.startswith(f"{kind}/") and f.endswith(".parquet"))
        if not parts:
            raise RuntimeError(f"{repo}: no {kind} parquet files")
        rows = []
        for f in parts:
            p = hf_hub_download(repo, f, repo_type="dataset")
            shas[repo] = _hf_snapshot_sha(p)
            rows.append(p)

        def gen(paths=rows, kind=kind):
            for p in paths:
                for r in _parquet_rows(p, ["_id", "title", "text"]):
                    if kind == "corpus":
                        yield {"_id": str(r["_id"]), "title": r.get("title") or "", "text": r.get("text") or ""}
                    else:
                        yield {"_id": str(r["_id"]), "text": r.get("text") or ""}

        write_jsonl(out / f"{kind}.jsonl", gen())
    ensure_dir(out / "qrels")
    split_map = {"validation": "dev"}
    for f in HfApi().list_repo_files(qrepo, repo_type="dataset"):
        if f.endswith(".tsv"):
            p = hf_hub_download(qrepo, f, repo_type="dataset")
            shas[qrepo] = _hf_snapshot_sha(p)
            name = Path(f).stem
            shutil.copyfile(p, out / "qrels" / f"{split_map.get(name, name)}.tsv")
    return {"source": "hf", "repos": shas}


def download_beir_info(domain: str, raw_dir: Path, force: bool = False) -> Tuple[Path, Dict[str, Any]]:
    """Download BEIR ``domain`` into raw_dir/<domain> -> (path, provenance).

    UKP mirror first (live as of 2026-09-20); fallback: the Hugging Face hub repositories
    BeIR/<name> and BeIR/<name>-qrels read with huggingface_hub + pyarrow (honours HF_ENDPOINT)."""
    out = ensure_dir(Path(raw_dir) / domain)
    prov_file = out / "provenance.json"
    if not force and (out / "corpus.jsonl").exists() and (out / "queries.jsonl").exists():
        if prov_file.exists():
            with open(prov_file, "r", encoding="utf-8") as f:
                return out, {**json.load(f), "reused": True}
        return out, {"source": "existing"}
    try:
        prov = _download_ukp(domain, Path(raw_dir), force)
    except Exception as e:  # noqa: BLE001
        print(f"[data] UKP download failed ({type(e).__name__}: {e}); trying the Hugging Face hub", flush=True)
        prov = _download_hf(domain, out)
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=1, sort_keys=True)
    return out, prov


def download_beir(domain: str, raw_dir: Path, force: bool = False) -> Path:
    return download_beir_info(domain, raw_dir, force)[0]


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header and header[0].strip().lower() not in ("query-id", "query_id", "qid"):
            # no header line
            qid, did, sc = header[:3]
            qrels[qid][did] = int(float(sc))
        for row in reader:
            if len(row) < 3:
                continue
            qrels[row[0]][row[1]] = int(float(row[2]))
    return dict(qrels)


def load_beir(domain: str, raw_dir: Path, splits: Iterable[str] = ("test",), load_corpus: bool = True) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, Dict[str, int]]]]:
    d = Path(raw_dir) / domain
    corpus: Dict[str, Dict[str, str]] = {}
    if load_corpus:
        for r in read_jsonl(d / "corpus.jsonl"):
            corpus[str(r["_id"])] = {"title": r.get("title", "") or "", "text": r.get("text", "") or ""}
    queries = {str(r["_id"]): r["text"] for r in read_jsonl(d / "queries.jsonl")}
    qrels = {}
    for s in splits:
        p = d / "qrels" / f"{s}.tsv"
        if p.exists():
            qrels[s] = load_qrels(p)
    return corpus, queries, qrels


# --------------------------------------------------------------------------- #
# Training questions (question text only; no answers are used)
# --------------------------------------------------------------------------- #
def _nq_open_train_questions() -> Tuple[List[str], str]:
    """NQ-Open train questions in dataset order (parquet via huggingface_hub; `datasets` fallback)."""
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(NQ_OPEN_HF[0], NQ_OPEN_HF[1], repo_type="dataset")
        return [r["question"] for r in _parquet_rows(p, ["question"])], f"hf:{NQ_OPEN_HF[0]}@{_hf_snapshot_sha(p)}"
    except Exception as e:  # noqa: BLE001
        print(f"[data] nq_open parquet download failed ({type(e).__name__}: {e}); trying `datasets`", flush=True)
    try:
        from datasets import load_dataset  # lazy
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(f"cannot read the NQ-Open training questions: the parquet path needs `pyarrow` + `huggingface_hub` and the fallback needs `datasets` ({e}). Install one of them, set HF_ENDPOINT if the hub is blocked, or drop the file in data/raw/train_questions/nq.jsonl (decision D-DATA-2).") from e

    ds = load_dataset(NQ_OPEN_HF[0], split="train")
    return [r["question"] for r in ds], f"datasets:{NQ_OPEN_HF[0]}"


def load_train_questions(domain: str, raw_dir: Path, train_q_dir: Path, max_n: Optional[int] = None) -> List[Dict[str, str]]:
    """nq       -> NQ-Open train questions (google-research-datasets/nq_open on the HF hub,
                   CC BY-SA 3.0), cached to data/raw/train_questions/nq.jsonl
       hotpotqa -> BEIR hotpotqa queries that appear in qrels/train.tsv
       msmarco  -> BEIR msmarco queries that appear in qrels/train.tsv (optional web-query domain)
       fiqa/scifact -> BEIR train qrels queries (cross-domain: not used for training by default)
    """
    train_q_dir = ensure_dir(train_q_dir)
    cache = train_q_dir / f"{domain}.jsonl"
    if cache.exists():
        rows = list(read_jsonl(cache))
        return rows[:max_n] if max_n else rows
    rows: List[Dict[str, str]] = []
    if domain == "nq":
        questions, _src = _nq_open_train_questions()
        for i, q in enumerate(questions):
            rows.append({"qid": f"nqopen-train-{i}", "text": q, "source": "nq_open/train"})
    else:
        _, queries, qrels = load_beir(domain, raw_dir, splits=("train",), load_corpus=False)
        train_qrels = qrels.get("train", {})
        for qid in sorted(train_qrels):
            if qid in queries:
                rows.append({"qid": qid, "text": queries[qid], "source": f"beir/{domain}/train"})
    write_jsonl(cache, rows)
    return rows[:max_n] if max_n else rows


# --------------------------------------------------------------------------- #
# De-duplication
# --------------------------------------------------------------------------- #
_token_re = re.compile(r"[a-z0-9]+")


def _grams(text: str, n: int = 3) -> Set[str]:
    toks = _token_re.findall(text.lower())
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def dedup(protected: List[Dict[str, str]], candidates: List[Dict[str, str]], jaccard_threshold: float = 0.8) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Remove from ``candidates`` (train/dev) every query that is an exact
    normalised duplicate of, or a near-duplicate (3-gram Jaccard >= threshold)
    of, any ``protected`` query (cal/test), and internal exact duplicates.
    Returns (kept, stats).  Rules are fixed before any result is observed."""
    prot_norm = {normalize_text(r["text"]) for r in protected}
    index: Dict[str, List[int]] = defaultdict(list)
    prot_grams: List[Set[str]] = []
    for i, r in enumerate(protected):
        g = _grams(r["text"])
        prot_grams.append(g)
        for x in g:
            index[x].append(i)
    kept, seen = [], set()
    stats = {"exact_internal": 0, "exact_vs_protected": 0, "near_vs_protected": 0, "kept": 0}
    for r in candidates:
        nt = normalize_text(r["text"])
        if nt in seen:
            stats["exact_internal"] += 1
            continue
        if nt in prot_norm:
            stats["exact_vs_protected"] += 1
            continue
        g = _grams(r["text"])
        cand_ids: Set[int] = set()
        for x in g:
            cand_ids.update(index.get(x, ()))
        near = False
        for i in cand_ids:
            pg = prot_grams[i]
            inter = len(g & pg)
            union = len(g) + len(pg) - inter
            if union and inter / union >= jaccard_threshold:
                near = True
                break
        if near:
            stats["near_vs_protected"] += 1
            continue
        seen.add(nt)
        kept.append(r)
    stats["kept"] = len(kept)
    return kept, stats


# --------------------------------------------------------------------------- #
# Frozen splits
# --------------------------------------------------------------------------- #
def make_splits(domain: str, raw_dir: Path, train_q_dir: Path, out_dir: Path, seed: int, n_cal: int, n_train: int, n_dev: int, test_pool_split: str = "test", cross_domain_only: bool = False, jaccard_threshold: float = 0.8) -> Dict[str, object]:
    """Create frozen splits for one domain.

    * cal/test: uniform random split (without replacement) of the BEIR pool
      qrels/<test_pool_split>.tsv (queries with at least one positive qrel); the
      remainder is the final test set.  ``test_pool_split`` comes from
      configs/main.yaml domains.test_pool_split (``dev`` for msmarco).
    * train/dev: from training questions (never from the pool), after
      de-duplication against cal+test; sizes n_train / n_dev.
    """
    rng = np.random.default_rng(seed)
    _, queries, qrels = load_beir(domain, raw_dir, splits=(test_pool_split,), load_corpus=False)
    if test_pool_split not in qrels:
        raise FileNotFoundError(f"{Path(raw_dir) / domain / 'qrels' / (test_pool_split + '.tsv')} missing (pool split of {domain})")
    pool_ids = sorted(q for q in qrels.get(test_pool_split, {}) if q in queries and any(v > 0 for v in qrels[test_pool_split][q].values()))
    perm = rng.permutation(len(pool_ids))
    out = ensure_dir(Path(out_dir) / domain)
    rows = lambda ids, src: [{"qid": q, "text": queries[q], "source": src} for q in ids]
    manifest: Dict[str, object] = {"domain": domain, "seed": seed, "pool_split": test_pool_split, "pool_size": len(pool_ids)}
    if cross_domain_only:
        test_ids = pool_ids
        write_jsonl(out / "test.jsonl", rows(test_ids, f"beir/{domain}/{test_pool_split}"))
        manifest.update(n_test=len(test_ids), n_cal=0)
        protected = rows(test_ids, "")
    else:
        n_cal = min(n_cal, len(pool_ids) - 1)
        cal_ids = [pool_ids[i] for i in sorted(perm[:n_cal])]
        test_ids = [pool_ids[i] for i in sorted(perm[n_cal:])]
        write_jsonl(out / "cal.jsonl", rows(cal_ids, f"beir/{domain}/{test_pool_split}"))
        write_jsonl(out / "test.jsonl", rows(test_ids, f"beir/{domain}/{test_pool_split}"))
        manifest.update(n_cal=len(cal_ids), n_test=len(test_ids))
        protected = rows(cal_ids + test_ids, "")
    if n_train + n_dev > 0:
        tq = load_train_questions(domain, raw_dir, train_q_dir)
        tq_perm = rng.permutation(len(tq))
        tq = [tq[i] for i in tq_perm]
        kept, stats = dedup(protected, tq, jaccard_threshold)
        train = kept[:n_train]
        dev = kept[n_train : n_train + n_dev]
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "dev.jsonl", dev)
        manifest.update(n_train=len(train), n_dev=len(dev), dedup=stats, train_source_total=len(tq))
    return manifest
