"""Evaluation metrics (Sec. 15).

* exact ordered-list mismatch  (eq. 31, the *main* metric),
* number of missed winners m(q) and the r-miss indicator ell_r,
* Overlap@k (diagnostic only),
* nDCG@k / MRR@k / Recall@k with the trec_eval convention (linear gain,
  log2 discount, ideal DCG from *all* qrels of the query) -- the real pipeline
  must use pytrec_eval; the pure-Python version here is cross-checked against
  it in tests/test_metrics.py whenever pytrec_eval is importable.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

__all__ = [
    "exact_mismatch",
    "missed_winners",
    "ell_r",
    "overlap_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "recall_at_k",
    "metric_gap_bound_holds",
]


def exact_mismatch(pred: Sequence, truth: Sequence) -> int:
    """1 if the ordered lists differ (set-equal but re-ordered also counts)."""
    return int(list(pred) != list(truth))


def missed_winners(pred: Sequence, truth: Sequence) -> int:
    return len(set(truth) - set(pred))


def ell_r(pred: Sequence, truth: Sequence, r: int) -> int:
    return int(missed_winners(pred, truth) >= r)


def overlap_at_k(pred: Sequence, truth: Sequence) -> float:
    k = len(truth)
    return len(set(pred) & set(truth)) / k


def _dcg(gains: Iterable[float]) -> float:
    return float(sum(g / math.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(ranked_ids: Sequence, qrels: Mapping, k: int) -> float:
    """trec_eval ndcg_cut_k: gain = relevance grade (linear), discount log2(i+2),
    IDCG from the sorted grades of ALL judged documents of the query.  Returns
    0.0 when the query has no positive-relevance document (trec_eval convention)."""
    gains = [float(qrels.get(d, 0)) for d in list(ranked_ids)[:k]]
    ideal = sorted((float(v) for v in qrels.values() if float(v) > 0), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg <= 0:
        return 0.0
    return _dcg(gains) / idcg


def mrr_at_k(ranked_ids: Sequence, qrels: Mapping, k: int) -> float:
    for i, d in enumerate(list(ranked_ids)[:k]):
        if float(qrels.get(d, 0)) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids: Sequence, qrels: Mapping, k: int) -> float:
    rel = {d for d, v in qrels.items() if float(v) > 0}
    if not rel:
        return 0.0
    return len(rel & set(list(ranked_ids)[:k])) / len(rel)


def metric_gap_bound_holds(m_full: float, m_rsc: float, mismatch: int) -> bool:
    """eq. 23:  |M_F(q) - M_RSC(q)| <= ell_1(q)  for any metric in [0,1] that
    depends only on the ordered top-k."""
    return abs(m_full - m_rsc) <= mismatch + 1e-12
