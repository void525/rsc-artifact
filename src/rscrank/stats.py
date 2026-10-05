"""Statistical summaries for the results tables (Sec. 15.4).

* risk estimates with exact (Clopper-Pearson) and Wilson intervals,
* paired query-level bootstrap for a mean difference (e.g. Delta nDCG) with
  optional cluster resampling, returning the lower confidence bound used in the
  pre-registered non-inferiority test  LCB(Delta) > -epsilon  (eq. 32),
* the conditional-risk decomposition of Sec. 9.2:
      overall error, early-stop fraction, error given early stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import beta

__all__ = ["risk_summary", "paired_bootstrap_diff", "noninferiority", "conditional_risk"]


def risk_summary(indicators: Sequence[int], conf: float = 0.95) -> Dict[str, float]:
    x = np.asarray(indicators, dtype=float)
    n, k = len(x), float(np.sum(x))
    p = k / n if n else float("nan")
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    z = 1.959963984540054 if abs(conf - 0.95) < 1e-9 else float(abs(np.round(np.sqrt(2) * _erfinv(conf), 6)))
    denom = 1 + z * z / n if n else 1
    centre = (p + z * z / (2 * n)) / denom if n else float("nan")
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom if n else float("nan")
    return {"n": int(n), "failures": int(k), "rate": p, "cp_lo": lo, "cp_hi": hi, "wilson_lo": float(centre - half), "wilson_hi": float(centre + half)}


def _erfinv(y: float) -> float:
    from scipy.special import erfinv

    return float(erfinv(y))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0, clusters: Optional[Sequence] = None, conf: float = 0.95) -> Dict[str, float]:
    """Bootstrap the mean of paired differences d = a - b over queries (or over
    clusters of queries when ``clusters`` is given).  Percentile interval."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align")
    d = a - b
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = [np.array([i]) for i in range(len(d))]
    else:
        cl = np.asarray(clusters)
        groups = [np.where(cl == c)[0] for c in np.unique(cl)]
    G = len(groups)
    sums = np.array([d[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    means = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, G, G)
        means[t] = sums[idx].sum() / sizes[idx].sum()
    alpha = 1 - conf
    return {
        "mean": float(d.mean()),
        "lcb": float(np.quantile(means, alpha / 2)),
        "ucb": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(len(d)),
        "n_groups": int(G),
        "n_boot": int(n_boot),
    }


def noninferiority(a: Sequence[float], b: Sequence[float], epsilon: float = 0.005, **kw) -> Dict[str, float]:
    """Pre-registered test: method a is non-inferior to reference b iff
    LCB(mean(a-b)) > -epsilon."""
    out = paired_bootstrap_diff(a, b, **kw)
    out["epsilon"] = epsilon
    out["noninferior"] = bool(out["lcb"] > -epsilon)
    return out


def conditional_risk(mismatch: Sequence[int], early: Sequence[bool]) -> Dict[str, float]:
    m = np.asarray(mismatch, dtype=float)
    e = np.asarray(early, dtype=bool)
    n = len(m)
    out = {"overall_error": float(m.mean()) if n else float("nan"), "early_fraction": float(e.mean()) if n else float("nan")}
    out["error_given_early"] = float(m[e].mean()) if e.any() else float("nan")
    out["error_given_full"] = float(m[~e].mean()) if (~e).any() else float("nan")
    return out
