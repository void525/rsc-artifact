"""Calibration of a scalar threshold from query-level scores (Sec. 8).

Three statistical regimes, all operating on the sorted calibration scores
``a_(1) <= ... <= a_(n)`` with the convention ``a_(n+1) = +inf``:

* ``marginal``      -- exchangeable queries, split-conformal index
                       j = ceil((n+1)(1-alpha))                              (Sec. 8.2)
* ``pac``           -- iid future queries, training-conditional (PAC) index
                       j = min{ j : P[Binomial(n, 1-alpha) >= j] <= delta }  (eq. 19)
* ``finite``        -- a fixed pool of L = n + M queries split uniformly at
                       random; controls the *test-set* error fraction:
                       j = min{ j : P[Hypergeom(L, M+j-t, n) >= j] <= delta },
                       t = floor(alpha*M) + 1                                (eq. 21)

The calibration unit is the **query**, never the (query, document) pair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Literal, Optional, Sequence

import numpy as np
from scipy.stats import binom, hypergeom

Regime = Literal["marginal", "pac", "finite"]

__all__ = [
    "exact_fraction",
    "marginal_index",
    "pac_index",
    "finite_population_index",
    "calibration_index",
    "CalibrationResult",
    "calibrate_threshold",
    "family_delta",
    "empirical_failure_probability_finite",
]


def exact_fraction(x: float) -> Fraction:
    """The decimal number a float *denotes* (its shortest round-trip repr), as an exact rational.

    ``Fraction(repr(0.05)) == 1/20`` whereas ``Fraction(0.05)`` is the binary approximation
    3602879701896397/72057594037927936.  Index arithmetic below is done on this exact value, so
    products such as (n+1)(1-alpha) or alpha*M that are integers in exact arithmetic are treated
    as integers for every n, M and every decimal alpha (no floating-point fudge factor)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    return Fraction(repr(float(x)))


def marginal_index(n: int, alpha: float) -> int:
    """j = ceil((n+1)(1-alpha)) in exact rational arithmetic; returns n+1 (=> +inf) when alpha is too small."""
    if n < 1 or not (0 < alpha < 1):
        raise ValueError("n >= 1 and 0 < alpha < 1 required")
    a = exact_fraction(alpha)
    j = int(math.ceil((n + 1) * (1 - a)))  # math.ceil is exact on Fraction
    return min(max(j, 1), n + 1)


def pac_index(n: int, alpha: float, delta: float) -> int:
    """Smallest j in 1..n+1 with P[Bin(n, 1-alpha) >= j] <= delta.

    Guarantee (iid calibration + future queries): with probability >= 1-delta
    over the calibration draw, P_Q(A(Q) > a_(j)) <= alpha.
    """
    if n < 1 or not (0 < alpha < 1) or not (0 < delta < 1):
        raise ValueError("bad arguments")
    p = 1.0 - alpha
    # sf(j-1) = P[X >= j]
    for j in range(1, n + 2):
        if binom.sf(j - 1, n, p) <= delta:
            return j
    return n + 1  # unreachable: P[X >= n+1] = 0


def finite_population_index(n: int, M: int, alpha: float, delta: float) -> int:
    """Smallest j in 1..n+1 with p_j <= delta where
    p_j = P[X >= j],  X ~ Hypergeom(L = n+M, successes = M+j-t, draws = n),
    t = floor(alpha*M) + 1.

    Guarantee (uniform split of a fixed pool): with probability >= 1-delta over
    the split, the fraction of *test* queries with A(q) > a_(j) is <= alpha.
    """
    if n < 1 or M < 1 or not (0 < alpha < 1) or not (0 < delta < 1):
        raise ValueError("bad arguments")
    L = n + M
    t = int(math.floor(exact_fraction(alpha) * M)) + 1  # exact: floor(alpha*M) + 1, alpha as the decimal it denotes
    for j in range(1, n + 2):
        good = M + j - t  # marked pool elements; 1 <= good <= L always holds since t <= M for alpha < 1
        # P[X >= j] where X = number of calibration points among the `good` smallest
        p = float(hypergeom.sf(j - 1, L, good, n))
        if p <= delta:
            return j
    return n + 1


def calibration_index(regime: Regime, n: int, alpha: float, delta: Optional[float] = None, M: Optional[int] = None) -> int:
    if regime == "marginal":
        return marginal_index(n, alpha)
    if regime == "pac":
        if delta is None:
            raise ValueError("pac regime needs delta")
        return pac_index(n, alpha, delta)
    if regime == "finite":
        if delta is None or M is None:
            raise ValueError("finite regime needs delta and M")
        return finite_population_index(n, M, alpha, delta)
    raise ValueError(f"unknown regime {regime!r}")


@dataclass
class CalibrationResult:
    regime: str
    alpha: float
    delta: Optional[float]
    n: int
    M: Optional[int]
    index: int
    threshold: float  # +inf means "always score everything"; -inf means "stop at the first checkpoint" (every calibration value was -inf up to the index)
    n_infinite: int   # number of calibration scores that were +inf/-inf (0 for A_{k,r}; -inf is legitimate for the trajectory statistic A^pi when no prefix is unsafe)

    def as_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["threshold"] = None if math.isinf(self.threshold) else self.threshold
        d["threshold_is_inf"] = bool(math.isinf(self.threshold) and self.threshold > 0)      # +inf only
        d["threshold_is_neg_inf"] = bool(math.isinf(self.threshold) and self.threshold < 0)  # -inf (JSON has no infinities)
        return d


def calibrate_threshold(
    scores: Sequence[float],
    alpha: float,
    regime: Regime = "marginal",
    delta: Optional[float] = None,
    M: Optional[int] = None,
) -> CalibrationResult:
    """Sort the query-level calibration scores and pick the order statistic."""
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or len(a) < 1:
        raise ValueError("scores must be a non-empty 1-D array")
    if np.any(np.isnan(a)):
        raise ValueError("NaN in calibration scores")
    n = len(a)
    j = calibration_index(regime, n, alpha, delta, M)
    a_sorted = np.sort(a)
    thr = math.inf if j >= n + 1 else float(a_sorted[j - 1])
    return CalibrationResult(regime, alpha, delta, n, M, j, thr, int(np.sum(np.isinf(a))))


def family_delta(delta_family: float, n_configs: int) -> float:
    """Bonferroni split of a family-wise calibration-failure budget (Sec. 8.6)."""
    if n_configs < 1:
        raise ValueError("n_configs >= 1")
    return delta_family / n_configs


def empirical_failure_probability_finite(pool_values: Sequence[float], n: int, j: int, alpha: float, n_splits: int = 2000, seed: int = 0) -> float:
    """Monte-Carlo estimate of P_split[ test-fraction(A > a_(j)) > alpha ] for a
    fixed pool -- used by tests to check the hypergeometric bound."""
    pool = np.asarray(pool_values, dtype=float)
    L = len(pool)
    M = L - n
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(n_splits):
        perm = rng.permutation(L)
        cal, test = pool[perm[:n]], pool[perm[n:]]
        thr = math.inf if j >= n + 1 else np.sort(cal)[j - 1]
        frac = float(np.mean(test > thr))
        fails += frac > alpha
    return fails / n_splits
