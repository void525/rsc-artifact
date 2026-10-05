"""Prefix-budget baselines (Sec. 14) and their calibration.

All prefix methods score the first ``m`` documents of the *cheap order*
(descending mu, ties by id) with the teacher and return the ordered teacher
top-k among them.  They are correct for query q iff ``m >= H_k(q)``.

* fixed prefix            : m constant (cost-quality curve),
* learned query budget    : m(q) = clip(ceil(c * exp(h(q))))  -- no guarantee,
* uniform calibrated      : m = order statistic of H_k on the calibration set,
* query-adaptive calibrated: m(q) = clip(ceil(exp(h(q) + t_H))) with t_H the
  calibrated order statistic of the residual  e(q) = log H_k(q) - h(q)  (eq. 30).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

from .calibrate import CalibrationResult, calibrate_threshold
from .core import H_k, teacher_order

__all__ = [
    "prefix_topk",
    "horizon_under_order",
    "calibrate_uniform_prefix",
    "adaptive_prefix_budget",
    "calibrate_adaptive_prefix",
    "learned_budget",
]


def prefix_topk(ids: Sequence, s_by_id: Dict, mu: np.ndarray, m: int, k: int, order: Optional[Sequence] = None):
    """Score the first m of the cheap order (or of ``order``, a full list of ids) and return the ordered top-k (ids).
    ``s_by_id`` is consulted only for the m prefix documents (the evaluator
    enforces this by passing an oracle-backed mapping)."""
    ids = list(ids)
    mu = np.asarray(mu, dtype=float)
    N = len(ids)
    m = int(min(max(m, k), N))
    if order is None:
        cheap = teacher_order(mu, ids)[:m]
        chosen = [ids[p] for p in cheap]
    else:
        chosen = list(order)[:m]
    sc = np.array([s_by_id[i] for i in chosen], dtype=float)
    order_sc = teacher_order(sc, chosen)
    return [chosen[p] for p in order_sc[:k]], m


def horizon_under_order(order: Sequence, winners: Sequence) -> int:
    """H_k^pi(q): worst 1-based position of a true winner in the scoring order ``order`` (the analogue of
    core.H_k for an arbitrary order; a prefix of that order contains T_k iff its length >= this value)."""
    pos = {i: r for r, i in enumerate(order, start=1)}
    return int(max(pos[w] for w in winners))


def calibrate_uniform_prefix(H_values: Sequence[int], alpha: float, regime: str = "marginal", delta: Optional[float] = None, M: Optional[int] = None, N: Optional[int] = None) -> CalibrationResult:
    """Order statistic of H_k over calibration queries; +inf maps to N (score all)."""
    res = calibrate_threshold(np.asarray(H_values, dtype=float), alpha, regime, delta, M)
    if math.isinf(res.threshold) and N is not None:
        res.threshold = float(N)
    return res


def adaptive_prefix_budget(h_pred: float, t_H: float, k: int, N: int) -> int:
    """m(q) = clip_[k,N]( ceil( exp(h(q) + t_H) ) )  (eq. 30)."""
    if math.isinf(t_H):
        return N
    val = math.exp(h_pred + t_H)
    return int(min(max(math.ceil(val - 1e-9), k), N))


def calibrate_adaptive_prefix(H_values: Sequence[int], h_pred: Sequence[float], alpha: float, regime: str = "marginal", delta: Optional[float] = None, M: Optional[int] = None) -> CalibrationResult:
    """Calibrate the residual e(q) = log H_k(q) - h(q)."""
    H = np.asarray(H_values, dtype=float)
    h = np.asarray(h_pred, dtype=float)
    if H.shape != h.shape:
        raise ValueError("shape mismatch")
    if np.any(H < 1):
        raise ValueError("H_k must be >= 1")
    e = np.log(H) - h
    return calibrate_threshold(e, alpha, regime, delta, M)


def learned_budget(h_pred: float, c: float, k: int, N: int) -> int:
    """Uncalibrated learned budget m(q) = clip(ceil(c * exp(h(q))))."""
    return int(min(max(math.ceil(c * math.exp(h_pred) - 1e-9), k), N))
