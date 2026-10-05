"""Assemble aligned arrays from the feature store and a teacher store into
training matrices and replay bundles."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .heads import pair_features, query_level_features
from .replay import QueryBundle
from .store import load_store

__all__ = ["SplitData", "load_split", "pair_matrix", "bundles_from_split"]


class SplitData:
    """Container for one (domain, split): feature store + teacher store, aligned by qid."""

    def __init__(self, feat: Dict[str, np.ndarray], teach: Optional[Dict[str, np.ndarray]], domain: str, split: str):
        self.domain, self.split = domain, split
        self.f = feat
        self.t = teach
        if teach is not None:
            if list(feat["qids"]) != list(teach["qids"]):
                raise ValueError(f"qid order mismatch between features and teacher store for {domain}/{split}")
            if not np.array_equal(feat["docids"], teach["docids"]):
                raise ValueError(f"docid mismatch between features and teacher store for {domain}/{split}")
        self.Q, self.N = feat["logit"].shape
        self.qids = list(feat["qids"])

    def n_valid(self, i: int) -> int:
        return int(self.f["valid"][i].sum())


def load_split(features_dir: Path, teacher_dir: Optional[Path], domain: str, split: str) -> Optional[SplitData]:
    ff = Path(features_dir) / domain / f"{split}.npz"
    if not ff.exists():
        return None
    tf = Path(teacher_dir) / domain / f"{split}.npz" if teacher_dir is not None else None
    teach = load_store(tf) if (tf is not None and tf.exists()) else None
    return SplitData(load_store(ff), teach, domain, split)


def pair_matrix(sd: SplitData, min_docs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X [P,d], y [P] (teacher score), q_index [P]; only valid pairs of queries with >= min_docs candidates."""
    Xs, ys, qs = [], [], []
    for i in range(sd.Q):
        n = sd.n_valid(i)
        if n < min_docs:
            continue
        X = pair_features(sd.f["emb"][i, :n].astype(np.float32), sd.f["logit"][i, :n], sd.f["bm25"][i, :n], sd.f["rank"][i, :n], float(sd.f["qlen"][i]), sd.f["dlen"][i, :n], sd.N)
        Xs.append(X.astype(np.float32))
        if sd.t is not None:
            ys.append(sd.t["score"][i, :n].astype(np.float32))
        qs.append(np.full(n, i, dtype=np.int32))
    X = np.concatenate(Xs, 0) if Xs else np.zeros((0, 1), np.float32)
    y = np.concatenate(ys, 0) if ys else np.zeros(0, np.float32)
    q = np.concatenate(qs, 0) if qs else np.zeros(0, np.int32)
    return X, y, q


def bundles_from_split(sd: SplitData, mu_flat: np.ndarray, q_index: np.ndarray, sigma_by_query: Optional[np.ndarray], k: int, qrels: Optional[Dict[str, Dict[str, int]]] = None, h_pred: Optional[np.ndarray] = None, N_cap: Optional[int] = None) -> List[QueryBundle]:
    """Build replay bundles.  ``mu_flat`` aligns with ``q_index`` from pair_matrix;
    ``sigma_by_query`` (len Q) or None for sigma = 1.  ``N_cap`` truncates the
    candidate list (prefix of the BM25 order) for the N-sensitivity experiment."""
    out = []
    for i in range(sd.Q):
        sel = np.where(q_index == i)[0]
        if len(sel) == 0:
            continue
        n = len(sel)
        if N_cap is not None:
            n = min(n, N_cap)
            sel = sel[:n]
        if n < k + 1:
            continue
        ids = list(sd.f["docids"][i, :n])
        s = sd.t["score"][i, :n].astype(float)
        mu = mu_flat[sel].astype(float)
        sigma = np.full(n, float(sigma_by_query[i]) if sigma_by_query is not None else 1.0)
        feats = {"cheap_raw": sd.f["logit"][i, :n].astype(float), "bm25": sd.f["bm25"][i, :n].astype(float), "length": sd.f["dlen"][i, :n].astype(float), "tokens": (sd.t["ntok"][i, :n] if "ntok" in sd.t else sd.f["ntok"][i, :n]).astype(float)}
        if h_pred is not None:
            feats["h_pred"] = np.full(n, float(h_pred[i]))
        qr = qrels.get(sd.qids[i], {}) if qrels else {}
        out.append(QueryBundle(sd.qids[i], ids, s, mu, sigma, feats, qr))
    return out


def query_features_for_split(sd: SplitData, mu_flat: np.ndarray, q_index: np.ndarray, k: int) -> np.ndarray:
    Z = []
    for i in range(sd.Q):
        sel = np.where(q_index == i)[0]
        n = len(sel)
        if n == 0:
            Z.append(np.full(22, np.nan))
            continue
        Z.append(query_level_features(sd.f["logit"][i, :n], mu_flat[sel], sd.f["dlen"][i, :n], float(sd.f["qlen"][i]), k))
    return np.asarray(Z, dtype=np.float64)
