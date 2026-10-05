"""First-stage candidate generation G: BM25 top-N per query (Sec. 13.1).

Backends (configs/main.yaml: candidates.backend):
  * bm25s    -- default.  Pure Python/numpy BM25 (bm25s + PyStemmer), Windows-safe, no Java.
                Anserini-like analysis: lowercase, regex tokens ``(?u)\\b\\w\\w+\\b``, bm25s' English stop list
                (= Lucene's 33-word default English stop set), PyStemmer ``porter`` (candidates.stemmer),
                Lucene BM25 (k1 = 0.9, b = 0.4; idf = log(1 + (N - df + 0.5)/(df + 0.5)), tf/(tf + k1(1-b+b dl/avgdl))).
                Exactly N candidates per query whenever the corpus has >= N documents; ranking is by
                descending score with ties broken by ascending corpus position (corpus.jsonl order), and
                queries that match fewer than N documents are filled with zero-score documents in corpus
                order.  (bm25s' own ``retrieve`` resolves ties by argpartition order, which is arbitrary.)
                Retrieval runs on all CPU threads (ThreadPoolExecutor; no multiprocessing).
  * pyserini -- optional: prebuilt Anserini index ``beir-v1.0.0-<domain>.flat`` (needs Java 21+);
                reference nDCG@10 of this exact index (pyserini 2CR, verified 2026-09-20):
                nq 0.3055, hotpotqa 0.6330, fiqa 0.2361, scifact 0.6789.  Any exception during the
                pyserini import or retrieval makes scripts/03_build_candidates.py fall back to bm25s (D-CAND).
  * python   -- tiny reference BM25 for smoke tests only (small corpora)

Output contract::

    data/candidates/<domain>/docstore.jsonl      {docid, title, text}
    data/candidates/<domain>/<split>.jsonl       {qid, query, docs: [{docid, rank, bm25}]}

The candidate set is fixed once and shared by every method; it is never
re-selected using teacher scores or qrels.
"""
from __future__ import annotations

import math
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .io import ensure_dir, read_jsonl, write_jsonl

REFERENCE_BM25_FLAT_NDCG10 = {"nq": 0.3055, "hotpotqa": 0.6330, "fiqa": 0.2361, "scifact": 0.6789}
BM25_K1, BM25_B = 0.9, 0.4
_tok = re.compile(r"[a-z0-9]+")

Hits = Dict[str, List[Tuple[str, float]]]

__all__ = ["PythonBM25", "retrieve", "retrieve_with_backend", "bm25s_search", "topn_deterministic", "write_candidates", "needed_docids", "write_docstore", "REFERENCE_BM25_FLAT_NDCG10"]


class PythonBM25:
    """Minimal BM25 (Robertson/Lucene-style idf) for smoke tests."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1, self.b = k1, b
        self.doc_len: Dict[str, int] = {}
        self.postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.avgdl = 0.0
        self.N = 0

    @staticmethod
    def tokens(text: str) -> List[str]:
        return _tok.findall(text.lower())

    def index(self, docs: Iterable[Tuple[str, str]]) -> None:
        total = 0
        for docid, text in docs:
            toks = self.tokens(text)
            self.doc_len[docid] = len(toks)
            total += len(toks)
            for t, c in Counter(toks).items():
                self.postings[t][docid] = c
        self.N = len(self.doc_len)
        self.avgdl = total / max(1, self.N)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = defaultdict(float)
        for t in sorted(set(self.tokens(query))):  # sorted: summation order must not depend on hash seeds
            plist = self.postings.get(t)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for docid, tf in plist.items():
                dl = self.doc_len[docid]
                scores[docid] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return ranked


# --------------------------------------------------------------------------- #
# Deterministic top-N
# --------------------------------------------------------------------------- #
def topn_deterministic(scores: np.ndarray, N: int) -> np.ndarray:
    """Positions of the N best documents: descending score, ties by ascending position.

    Assumes non-negative scores (true for Lucene BM25).  Documents with score 0 (no query term)
    fill the list in corpus order when fewer than N documents match."""
    scores = np.asarray(scores)
    n = len(scores)
    N = min(int(N), n)
    if N <= 0:
        return np.zeros(0, dtype=np.int64)
    nz = np.flatnonzero(scores > 0)  # ascending positions
    if len(nz) >= N:
        s = scores[nz]
        thr = s[np.argpartition(-s, N - 1)[:N]].min()
        above = nz[s > thr]
        tie = nz[s == thr][: N - len(above)]  # smallest positions among the boundary ties
        sel = np.concatenate([above, tie])
    else:
        head = np.arange(min(n, N + len(nz)))
        fill = head[~np.isin(head, nz, assume_unique=True)][: N - len(nz)]
        sel = np.concatenate([nz, fill])
    return sel[np.lexsort((sel, -scores[sel].astype(np.float64)))]


# --------------------------------------------------------------------------- #
# bm25s backend
# --------------------------------------------------------------------------- #
def _default_threads() -> int:
    return max(1, os.cpu_count() or 1)


def bm25s_search(corpus_iter: Iterable[Tuple[str, str]], queries: Dict[str, str], N: int, k1: float = BM25_K1, b: float = BM25_B, stemmer: str = "porter", stopwords: str = "en", n_threads: Optional[int] = None, chunk_docs: int = 200_000, log: Optional[Callable[[str], None]] = None) -> Tuple[Hits, Dict[str, Any]]:
    """Index ``corpus_iter`` (docid, text) with bm25s and return (hits, info).

    The corpus is streamed and tokenized in chunks of ``chunk_docs`` documents with one shared
    vocabulary, so raw texts are never all in memory at once (see docs/MODELS.md for RAM numbers)."""
    import bm25s  # lazy
    import Stemmer
    from bm25s.tokenization import Tokenizer

    log = log or (lambda m: None)
    n_threads = int(n_threads) if n_threads else _default_threads()
    tok = Tokenizer(lower=True, stopwords=stopwords, stemmer=Stemmer.Stemmer(stemmer))
    info: Dict[str, Any] = {"backend": "bm25s", "bm25s_version": getattr(bm25s, "__version__", "?"), "k1": k1, "b": b, "method": "lucene", "stopwords": f"bm25s '{stopwords}' ({len(tok.stopwords or [])} words)", "stemmer": f"PyStemmer {stemmer}", "token_pattern": r"(?u)\b\w\w+\b (lowercased)", "tie_break": "descending score, then ascending corpus position", "n_threads": n_threads, "N": int(N)}
    t0 = time.time()
    ids: List[str] = []
    corpus_tokens: List[List[int]] = []
    buf: List[str] = []

    def flush():
        if buf:
            corpus_tokens.extend(tok.tokenize(buf, update_vocab=True, return_as="ids", show_progress=False, allow_empty=False))
            buf.clear()

    for docid, text in corpus_iter:
        ids.append(docid)
        buf.append(text)
        if len(buf) >= chunk_docs:
            flush()
            log(f"bm25s: tokenized {len(ids)} documents")
    flush()
    vocab = tok.get_vocab_dict()
    info["tokenize_seconds"] = round(time.time() - t0, 2)
    t1 = time.time()
    retriever = bm25s.BM25(k1=k1, b=b, method="lucene")
    retriever.index((corpus_tokens, dict(vocab)), show_progress=False)
    n_docs = int(retriever.scores["num_docs"])
    info.update(n_docs=n_docs, vocab_size=len(vocab), avg_doc_len=float(np.mean([len(x) for x in corpus_tokens])) if corpus_tokens else 0.0, index_seconds=round(time.time() - t1, 2))
    del corpus_tokens
    log(f"bm25s: indexed {n_docs} documents, vocab {len(vocab)} ({info['tokenize_seconds']}s tokenize + {info['index_seconds']}s index)")

    qids = list(queries)
    q_tokens = tok.tokenize([queries[q] for q in qids], update_vocab=False, return_as="ids", show_progress=False, allow_empty=False)
    V = len(vocab)
    q_tokens = [[t for t in x if 0 <= t < V] for x in q_tokens]
    t2 = time.time()

    def one(token_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        if token_ids:
            sc = retriever.get_scores_from_ids(token_ids)
        else:
            sc = np.zeros(n_docs, dtype=np.float32)
        top = topn_deterministic(sc, N)
        return top, sc[top]

    if n_threads > 1 and len(qids) > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            results = list(ex.map(one, q_tokens, chunksize=16))
    else:
        results = [one(x) for x in q_tokens]
    out: Hits = {}
    n_fill = 0
    for q, (top, sc) in zip(qids, results):
        out[q] = [(ids[int(d)], float(s)) for d, s in zip(top, sc)]
        n_fill += int(np.any(sc <= 0))
    info.update(retrieve_seconds=round(time.time() - t2, 2), n_queries=len(qids), queries_empty_after_tokenization=int(sum(1 for x in q_tokens if not x)), queries_with_zero_score_fill=n_fill)
    return out, info


# --------------------------------------------------------------------------- #
# pyserini backend (optional)
# --------------------------------------------------------------------------- #
def _pyserini_search(domain: str, queries: Dict[str, str], N: int, index_name: Optional[str] = None, threads: Optional[int] = None) -> Tuple[Hits, Dict[str, Any]]:
    from pyserini.search.lucene import LuceneSearcher  # lazy; needs Java 21+

    threads = int(threads) if threads else _default_threads()
    name = index_name or f"beir-v1.0.0-{domain}.flat"
    t0 = time.time()
    searcher = LuceneSearcher.from_prebuilt_index(name)
    searcher.set_bm25(BM25_K1, BM25_B)
    t1 = time.time()
    qids = list(queries)
    out: Hits = {}
    B = 256
    for i in range(0, len(qids), B):
        chunk = qids[i : i + B]
        res = searcher.batch_search([queries[q] for q in chunk], chunk, k=N, threads=threads)
        for q in chunk:
            out[q] = [(h.docid, float(h.score)) for h in res[q]]
    info = {"backend": "pyserini", "index": name, "k1": BM25_K1, "b": BM25_B, "n_threads": threads, "N": int(N), "index_seconds": round(t1 - t0, 2), "retrieve_seconds": round(time.time() - t1, 2), "n_queries": len(qids)}
    return out, info


def retrieve(backend: str, domain: str, corpus_iter, queries: Dict[str, str], N: int, **kw) -> Tuple[Hits, Dict[str, Any]]:
    """(hits, info) for one domain.  kw: bm25s -> k1, b, stemmer, stopwords, n_threads, chunk_docs, log;
    pyserini -> index_name, threads."""
    if backend == "pyserini":
        return _pyserini_search(domain, queries, N, kw.get("index_name"), kw.get("threads"))
    if backend == "bm25s":
        keys = ("k1", "b", "stemmer", "stopwords", "n_threads", "chunk_docs", "log")
        return bm25s_search(corpus_iter, queries, N, **{k: v for k, v in kw.items() if k in keys and v is not None})
    if backend == "python":
        t0 = time.time()
        bm = PythonBM25()
        bm.index(corpus_iter)
        hits = {q: bm.search(t, N) for q, t in queries.items()}
        return hits, {"backend": "python", "k1": bm.k1, "b": bm.b, "N": int(N), "n_docs": bm.N, "seconds": round(time.time() - t0, 2)}
    raise ValueError(f"unknown backend {backend}")


def retrieve_with_backend(backend: str, domain: str, corpus_iter, queries: Dict[str, str], N: int, **kw) -> Hits:
    """Backward-compatible wrapper: hits only."""
    return retrieve(backend, domain, corpus_iter, queries, N, **kw)[0]


def write_candidates(out_dir: Path, domain: str, split: str, queries: Dict[str, str], hits: Hits) -> Path:
    out = ensure_dir(Path(out_dir) / domain) / f"{split}.jsonl"
    rows = []
    for q in sorted(queries):
        docs = [{"docid": d, "rank": r + 1, "bm25": s} for r, (d, s) in enumerate(hits.get(q, []))]
        rows.append({"qid": q, "query": queries[q], "docs": docs})
    write_jsonl(out, rows)
    return out


def needed_docids(cand_files: Sequence[Path]) -> set:
    need = set()
    for p in cand_files:
        for r in read_jsonl(p):
            need.update(d["docid"] for d in r["docs"])
    return need


def write_docstore(out_dir: Path, domain: str, corpus_iter, need: set) -> Path:
    out = ensure_dir(Path(out_dir) / domain) / "docstore.jsonl"
    rows = ({"docid": docid, "title": title, "text": text} for docid, title, text in corpus_iter if docid in need)
    write_jsonl(out, rows)
    return out
