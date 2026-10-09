"""Core mathematics of RSC-Rank (Rank-Sufficient Calibration).

Everything in this module is pure NumPy and deterministic.  It implements the
objects defined in the paper (docs/THEORY.md, paper/sections/03_method.tex):

* the fixed teacher order  (descending score, ties broken by ascending doc id);
* the rank-sufficient critical values  A_{k,r}(q)  (Theorem 1 / Theorem 2);
* the score-coverage statistics  B_top(q), B_all(q)  (eq. 14) and the
  cheap-order horizon  H_k(q)  (eq. 29);
* the guardrail  U_i(lambda) = mu_i + lambda * sigma_i  and the strict stopping
  rule  tau(V) > max_{j not in V} U_j(lambda)  (eq. 3-4);
* the online controller that only sees teacher scores through an oracle
  (the full teacher cache is never handed to the controller);
* the trajectory-specific critical value  A^pi(q)  used by the
  "fixed-trajectory calibration" baseline;
* the offline diagnostic lower bound  C*_cert(q)  (eq. 17);
* exhaustive brute-force reference implementations used only by tests.

Conventions
-----------
Arrays are 1-D and aligned: ``s[i]``, ``mu[i]``, ``sigma[i]`` refer to the
candidate whose external identifier is ``ids[i]``.  ``ids`` must be unique and
sortable; when ``ids`` is None the position index is used.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "teacher_order",
    "true_topk",
    "kth_max",
    "critical_value",
    "A_kr",
    "B_top",
    "B_all",
    "H_k",
    "guardrail",
    "can_stop",
    "lambda_star",
    "TeacherOracle",
    "ControllerResult",
    "run_controller",
    "trajectory_critical_value",
    "blocker_critical_value",
    "cert_lower_bound",
    "unsafe_sets",
    "brute_force_critical_value",
    "validate_instance",
    "cost_anatomy",
    "checkpoints_from_schedule",
]


# --------------------------------------------------------------------------- #
# Ordering primitives
# --------------------------------------------------------------------------- #
def _as_ids(n: int, ids: Optional[Sequence]) -> np.ndarray:
    if ids is None:
        return np.arange(n)
    ids = np.asarray(ids)
    if ids.shape != (n,):
        raise ValueError(f"ids must have shape ({n},), got {ids.shape}")
    if len(np.unique(ids)) != n:
        raise ValueError("candidate identifiers must be unique")
    return ids


def validate_instance(s: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
    """Raise ValueError on any violated implementation invariant (Sec. 11.1)."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if not (s.shape == mu.shape == sigma.shape) or s.ndim != 1:
        raise ValueError("s, mu, sigma must be 1-D arrays of equal length")
    if s.shape[0] < 2:
        raise ValueError("need at least two candidates")
    for name, arr in (("s", s), ("mu", mu), ("sigma", sigma)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive")


def teacher_order(s: np.ndarray, ids: Optional[Sequence] = None) -> np.ndarray:
    """Positions sorted by descending score; ties broken by ascending id.

    This is the *fixed total order* used everywhere (Sec. 3.2).  Returns an
    array of positions ``p`` such that ``s[p[0]] >= s[p[1]] >= ...``.
    """
    s = np.asarray(s, dtype=float)
    ids = _as_ids(len(s), ids)
    # lexsort sorts by the last key first: primary = -s (descending s),
    # secondary = ids ascending.
    return np.lexsort((ids, -s))


def true_topk(s: np.ndarray, k: int, ids: Optional[Sequence] = None) -> np.ndarray:
    """Positions of the ordered teacher top-k (the list T_k(q))."""
    if not (1 <= k < len(s)):
        raise ValueError("require 1 <= k < N")
    return teacher_order(s, ids)[:k]


def kth_max(values: Iterable[float], k: int) -> float:
    """k-th largest value of a collection (k = 1 is the maximum)."""
    v = np.sort(np.asarray(list(values), dtype=float))[::-1]
    if k < 1 or k > len(v):
        raise ValueError("k out of range")
    return float(v[k - 1])


# --------------------------------------------------------------------------- #
# Query-level statistics
# --------------------------------------------------------------------------- #
def critical_value(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    r: int = 1,
    ids: Optional[Sequence] = None,
) -> float:
    """Rank-sufficient critical value A_{k,r}(q)  (eq. 5 for r=1, eq. 10 in general).

        A_{k,r}(q) = rthmax_{i in T_k(q)}  ( s_(k+r) - mu_i ) / sigma_i

    where s_(k+r) is the (k+r)-th largest teacher score and rthmax is the r-th
    largest value.  Requires 1 <= r <= min(k, N-k).
    """
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    validate_instance(s, mu, sigma)
    n = len(s)
    if not (1 <= k < n):
        raise ValueError("require 1 <= k < N")
    if not (1 <= r <= min(k, n - k)):
        raise ValueError("require 1 <= r <= min(k, N-k)")
    order = teacher_order(s, ids)
    top = order[:k]
    boundary = s[order[k + r - 1]]  # s_(k+r), 1-indexed
    z = (boundary - mu[top]) / sigma[top]
    return kth_max(z, r)


def A_kr(s, mu, sigma, k, r=1, ids=None) -> float:  # short alias used in scripts
    return critical_value(s, mu, sigma, k, r, ids)


def B_top(s, mu, sigma, k, ids=None) -> float:
    """Score-coverage statistic over the true top-k:  max_{i in T_k} (s_i - mu_i)/sigma_i."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    validate_instance(s, mu, sigma)
    top = true_topk(s, k, ids)
    return float(np.max((s[top] - mu[top]) / sigma[top]))


def B_all(s, mu, sigma, ids=None) -> float:
    """Score-coverage statistic over all candidates:  max_i (s_i - mu_i)/sigma_i."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    validate_instance(s, mu, sigma)
    return float(np.max((s - mu) / sigma))


def H_k(s: np.ndarray, mu: np.ndarray, k: int, ids: Optional[Sequence] = None) -> int:
    """Cheap-order horizon: worst 1-indexed rank (by descending mu, ties by id)
    among the true teacher top-k  (eq. 29).  A prefix of length m of the cheap
    order contains T_k(q) iff m >= H_k(q)."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    top = set(true_topk(s, k, ids).tolist())
    cheap_order = teacher_order(mu, ids)  # same tie rule applied to mu
    worst = 0
    for rank, pos in enumerate(cheap_order, start=1):
        if pos in top:
            worst = rank
    return int(worst)


def guardrail(mu: np.ndarray, sigma: np.ndarray, lam: float) -> np.ndarray:
    """U_i(lambda) = mu_i + lambda * sigma_i (eq. 3).  lambda may be +inf."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.isinf(lam):
        return np.full_like(mu, np.inf if lam > 0 else -np.inf)
    return mu + lam * sigma


def can_stop(scored_scores: Sequence[float], mu_unscored: Sequence[float], sigma_unscored: Sequence[float], lam: float, k: int) -> bool:
    """Strict stopping rule (eq. 4):  |V| >= k  and  tau(V) > max_{j not in V} U_j(lambda).

    Implementation note (floating point).  The rule is evaluated in *normalised
    units*:  stop  iff  lambda < lambda*(V) = min_{j not in V} (tau(V) - mu_j)/sigma_j.
    For sigma_j > 0 this is algebraically identical to the guardrail form, and
    because IEEE rounding is monotone, the float value of lambda*(V) for an
    unsafe V is <= the float value of A_{k,r}(q) computed by
    :func:`critical_value` with the same expression.  Hence the implemented
    controller is *exactly* consistent with the calibrated statistic: no
    rounding can turn ``lambda >= A`` into an unsafe stop.

    If nothing is unscored the caller should return the full teacher order
    instead; here we return True for convenience (min over empty set = +inf).
    """
    if len(scored_scores) < k:
        return False
    if len(mu_unscored) == 0:
        return True
    if np.isposinf(lam):
        return False
    tau = kth_max(scored_scores, k)
    mu_u = np.asarray(mu_unscored, dtype=float)
    sg_u = np.asarray(sigma_unscored, dtype=float)
    lam_star = float(np.min((tau - mu_u) / sg_u))
    return bool(lam < lam_star)


def lambda_star(V: Sequence[int], s: np.ndarray, mu: np.ndarray, sigma: np.ndarray, k: int) -> float:
    """lambda*(V) = min_{j not in V} (tau(V) - mu_j)/sigma_j  for a scored set V
    (positions).  The stopping rule holds at lambda iff lambda < lambda*(V)."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    V = list(V)
    if len(V) < k:
        raise ValueError("|V| must be >= k")
    rest = [j for j in range(len(s)) if j not in set(V)]
    if not rest:
        return math.inf
    tau = kth_max(s[V], k)
    return float(np.min((tau - mu[rest]) / sigma[rest]))


# --------------------------------------------------------------------------- #
# Oracle + online controller
# --------------------------------------------------------------------------- #
class TeacherOracle:
    """Hides the full teacher scores behind a request interface.

    The controller can only call ``score(ids)``.  Every call is counted, ids are
    validated (unknown id, duplicate id, or NaN score raise), and a scored id is
    never re-scored (cached).  The evaluator, not the controller, keeps the
    reference to the underlying array.
    """

    def __init__(self, ids: Sequence, scores: Sequence[float], cost_fn: Optional[Callable] = None):
        ids = list(ids)
        scores = np.asarray(scores, dtype=float)
        if len(ids) != len(scores):
            raise ValueError("ids and scores length mismatch")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate identifiers in oracle")
        if not np.all(np.isfinite(scores)):
            raise ValueError("teacher scores contain NaN/inf")
        self._table: Dict = {i: float(x) for i, x in zip(ids, scores)}
        self._seen: Dict = {}
        self.n_calls = 0
        self.n_requests = 0
        self.cost = 0.0
        self._cost_fn = cost_fn

    def score(self, ids: Sequence) -> List[float]:
        ids = list(ids)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate identifiers in a single request")
        out = []
        self.n_requests += 1
        for i in ids:
            if i in self._seen:
                out.append(self._seen[i])
                continue
            if i not in self._table:
                raise KeyError(f"unknown identifier requested: {i!r}")
            v = self._table[i]
            self._seen[i] = v
            self.n_calls += 1
            if self._cost_fn is not None:
                self.cost += float(self._cost_fn(i))
            out.append(v)
        return out

    @property
    def scored_ids(self) -> List:
        return list(self._seen.keys())


@dataclass
class ControllerResult:
    topk_ids: List
    n_calls: int
    n_rounds: int
    early_stopped: bool
    scored_ids: List = field(default_factory=list)
    trajectory: List[List] = field(default_factory=list)  # batches of ids in request order


def default_batch_schedule(k: int, first: Optional[int] = None, rest: Optional[int] = None) -> Callable[[int], int]:
    """Batch sizes used by the main configuration: first 16 then 8 for k=10
    (Sec. 11).  Generalised as first=max(k+6,k), rest=max(k//2+3, 1) unless given."""
    f = first if first is not None else max(k + 6, k)
    r = rest if rest is not None else max(k // 2 + 3, 1)

    def sched(round_idx: int) -> int:
        return f if round_idx == 0 else r

    return sched


def blocker_batch_schedule(min_batch: int = 8, max_batch: Optional[int] = None) -> Callable[..., int]:
    """State-dependent batch size ("blocker batches"): once |V| >= k, score in ONE round every unscored
    document whose guardrail reaches the current k-th observed score, i.e. the documents that block the
    stop rule -- exactly the set that any correct stop must score anyway (Prop. 10/11, {j : U_j >= tau(V)}
    with tau(V) <= theta = s_(k)); at least ``min_batch`` (so that progress is made when few block) and at
    most ``max_batch``.  Before |V| >= k it returns ``min_batch``.  The decision uses only cheap information
    and observed scores, so it is a *legal* policy: the calibrated lambda needs no recalibration (Thm 1).
    Fewer, larger rounds are what a GPU wants; the same lambda protects them (that is the point)."""

    def sched(round_idx: int, observed: Dict, order: Sequence, mu_by_id: Dict, sg_by_id: Dict, lam: float, k: int) -> int:
        n_rem = len(order)
        if len(observed) < k or math.isinf(lam):
            b = min_batch if not math.isinf(lam) else n_rem
        else:
            tau = kth_max(observed.values(), k)
            # normalised comparison, same expression as can_stop: j blocks iff (tau - mu_j)/sigma_j <= lam
            b = sum(1 for j in order if (tau - mu_by_id[j]) / sg_by_id[j] <= lam)
            b = max(b, min_batch)
        if max_batch is not None:
            b = min(b, max_batch)
        return max(1, min(b, n_rem))

    sched.adaptive = True  # type: ignore[attr-defined]
    return sched


def run_controller(
    ids: Sequence,
    mu: np.ndarray,
    sigma: np.ndarray,
    lam: float,
    k: int,
    oracle: TeacherOracle,
    policy: Callable[..., List],
    batch_schedule: Optional[Callable[..., int]] = None,
) -> ControllerResult:
    """Online algorithm of Sec. 11 for one query.

    ``policy(remaining_ids, observed: dict id->score, mu_by_id, features)``
    returns the remaining ids in the order they should be scored next; it may
    use only cheap information and already-observed teacher scores.
    ``batch_schedule(round_idx)`` gives the number of documents to score in
    that round; a schedule with attribute ``adaptive`` is called as
    ``batch_schedule(round_idx, observed, order, mu_by_id, sg_by_id, lam, k)``
    (see ``blocker_batch_schedule``).  The controller never sees unscored teacher scores.
    """
    ids = list(ids)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if len(ids) != len(mu) or len(mu) != len(sigma):
        raise ValueError("ids/mu/sigma length mismatch")
    if np.any(sigma <= 0) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError("invalid cheap predictions")
    if not (1 <= k < len(ids)):
        raise ValueError("require 1 <= k < N")
    if batch_schedule is None:
        batch_schedule = default_batch_schedule(k)

    mu_by_id = {i: float(m) for i, m in zip(ids, mu)}
    sg_by_id = {i: float(g) for i, g in zip(ids, sigma)}
    observed: Dict = {}
    remaining = list(ids)
    trajectory: List[List] = []
    rounds = 0
    while remaining:
        order = list(policy(remaining, observed, mu_by_id))
        if sorted(map(str, order)) != sorted(map(str, remaining)):
            raise ValueError("policy must return a permutation of the remaining ids")
        if getattr(batch_schedule, "adaptive", False):
            b = max(1, int(batch_schedule(rounds, observed, order, mu_by_id, sg_by_id, lam, k)))
        else:
            b = max(1, int(batch_schedule(rounds)))
        batch = order[:b]
        scores = oracle.score(batch)
        if len(scores) != len(batch):
            raise ValueError("oracle returned wrong number of scores")
        for i, sc in zip(batch, scores):
            if not math.isfinite(sc):
                raise ValueError("oracle returned NaN/inf")
            observed[i] = sc
        trajectory.append(list(batch))
        rounds += 1
        remaining = [i for i in remaining if i not in observed]
        if not remaining:
            break
        if len(observed) >= k:
            if can_stop(list(observed.values()), [mu_by_id[j] for j in remaining], [sg_by_id[j] for j in remaining], lam, k):
                top = _ordered_topk_from_dict(observed, k)
                return ControllerResult(top, oracle.n_calls, rounds, True, list(observed.keys()), trajectory)
    top = _ordered_topk_from_dict(observed, k)
    return ControllerResult(top, oracle.n_calls, rounds, False, list(observed.keys()), trajectory)


def _ordered_topk_from_dict(observed: Dict, k: int) -> List:
    items = list(observed.items())
    ids = [i for i, _ in items]
    sc = np.array([v for _, v in items], dtype=float)
    order = teacher_order(sc, ids)
    return [ids[p] for p in order[:k]]


# --------------------------------------------------------------------------- #
# Trajectory-specific critical value (fixed-trajectory calibration baseline)
# --------------------------------------------------------------------------- #
def trajectory_critical_value(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    prefixes: Sequence[Sequence[int]],
    r: int = 1,
    ids: Optional[Sequence] = None,
) -> float:
    """A^pi_{k,r}(q): smallest lambda such that no *unsafe prefix* of the given
    trajectory can satisfy the stopping rule.  ``prefixes`` are the nested
    scored sets (positions) after each round, excluding the full set.
    Returns -inf if the trajectory has no unsafe prefix with |V| >= k.
    Always <= A_{k,r}(q) since prefixes are a subset of all unsafe sets."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    top = set(true_topk(s, k, ids).tolist())
    n = len(s)
    best = -math.inf
    for V in prefixes:
        V = list(V)
        if len(V) < k or len(V) >= n:
            continue
        missing = len(top - set(V))
        if missing >= r:
            best = max(best, lambda_star(V, s, mu, sigma, k))
    return float(best)


def blocker_critical_value(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    ids: Optional[Sequence] = None,
    first: int = 16,
    min_batch: int = 8,
) -> float:
    """Critical value of the lambda-dependent blocker-batch controller (mu-descending order, query-shared sigma).

    The controller scores the first ``first`` documents of the mu order, then in one round every unscored document
    whose guardrail reaches the current k-th observed score (at least ``min_batch``; ``blocker_batch_schedule``).
    Its second round depends on lambda, so the trajectory critical value of a trajectory generated at one lambda
    does not protect the controller re-executed at another.  With a shared sigma the blockers are a contiguous
    mu-prefix of the unscored documents and the rule holds after the second round, so the controller re-executed
    at lambda errs on q iff lambda < blocker_critical_value(q), where, with m_T the mu-rank of the last winner,
    P_0 the first round and tau_0 = tau(P_0):
        -inf                                if m_T <= first;
        lambda*(P_0)                        if first < m_T <= first + min_batch;
        (tau_0 - mu_[m_T]) / sigma          otherwise (the last winner must itself be a blocker).
    The value is at most A_{k,1}(q)."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    validate_instance(s, mu, sigma)
    n = len(s)
    if not np.all(sigma == sigma[0]):
        raise ValueError("blocker_critical_value needs a query-shared sigma")
    if not (1 <= k < n) or first < k or min_batch < 1:
        raise ValueError("require 1 <= k < N, first >= k and min_batch >= 1")
    ids_arr = _as_ids(n, ids)
    mu_order = np.lexsort((ids_arr, -mu))
    rank = np.empty(n, dtype=int)
    rank[mu_order] = np.arange(1, n + 1)
    m_T = int(np.max(rank[true_topk(s, k, ids)]))
    if m_T <= first:
        return -math.inf
    tau0 = kth_max(s[mu_order[:first]], k)
    sg = float(sigma[0])
    if m_T <= first + min_batch:
        return float((tau0 - mu[mu_order[first]]) / sg)
    return float((tau0 - mu[mu_order[m_T - 1]]) / sg)


def size_critical_value(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    c: int,
    ids: Optional[Sequence] = None,
) -> float:
    """max{ lambda*(V) : |V| = c, V misses a winner }  for k <= c < N  (r = 1), without enumeration.

    Write W for the N - c unscored documents.  If the k-th scored document has teacher rank p, then tau(V) = s_(p),
    W holds exactly p - k documents of rank < p (so at least one winner) and the rest from ranks > p.  For fixed p
    the best W takes, in each of the two rank ranges, the documents with the largest (s_(p) - mu_j)/sigma_j, so the
    value is a maximum over at most N - c ranks p.  c = N - 1 gives A_{k,1}(q)."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    validate_instance(s, mu, sigma)
    n = len(s)
    if not (1 <= k < n) or not (k <= c < n):
        raise ValueError("require 1 <= k <= c < N")
    m = n - c
    order = teacher_order(s, ids)
    best = -math.inf
    for p in range(k + 1, min(k + m, n) + 1):
        hi, lo = p - k, m - (p - k)
        if lo > n - p:
            continue
        tau = s[order[p - 1]]
        z_hi = np.sort((tau - mu[order[: p - 1]]) / sigma[order[: p - 1]])[::-1]
        val = float(z_hi[hi - 1])
        if lo > 0:
            z_lo = np.sort((tau - mu[order[p:]]) / sigma[order[p:]])[::-1]
            val = min(val, float(z_lo[lo - 1]))
        best = max(best, val)
    return float(best)


def grid_critical_value(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    checkpoints: Sequence[int],
    ids: Optional[Sequence] = None,
) -> float:
    """Critical value of every policy that evaluates the rule only at scored-set sizes in ``checkpoints`` (r = 1).

    Adding documents to an unsafe set while keeping one winner out cannot lower lambda*, so the worst checkpoint is
    the last one below N; -inf if no checkpoint size lies in [k, N)."""
    n = len(np.asarray(s))
    eligible = [int(c) for c in checkpoints if k <= int(c) < n]
    if not eligible:
        return -math.inf
    return size_critical_value(s, mu, sigma, k, max(eligible), ids)


def prefixes_from_trajectory(trajectory: Sequence[Sequence], ids: Sequence) -> List[List[int]]:
    """Turn a list of id-batches into nested position prefixes."""
    pos = {i: p for p, i in enumerate(ids)}
    out, cur = [], []
    for batch in trajectory:
        cur = cur + [pos[i] for i in batch]
        out.append(list(cur))
    return out


# --------------------------------------------------------------------------- #
# Offline diagnostic (eq. 17)
# --------------------------------------------------------------------------- #
def cert_lower_bound(s, mu, sigma, k, lam, ids=None) -> int:
    """C*_cert(q) = | T_k  U  { j : U_j(lambda) >= s_(k) } |  -- a lower bound on
    the number of teacher calls of any correctly-certified stop.  Offline only."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    order = teacher_order(s, ids)
    top = set(order[:k].tolist())
    sk = s[order[k - 1]]
    # normalised form, consistent with can_stop and cost_anatomy (docs/THEORY.md Sec. 8)
    reach = ((sk - mu) / sigma) <= lam if not np.isinf(lam) else np.full(len(s), lam > 0)
    need = top | {int(j) for j in np.where(reach)[0]}
    return int(len(need))


# --------------------------------------------------------------------------- #
# Brute force references (tests only; exponential)
# --------------------------------------------------------------------------- #
def unsafe_sets(n: int, k: int, top: Iterable[int], r: int = 1) -> Iterable[Tuple[int, ...]]:
    """All V subset-neq [n] with |V| >= k and |T_k \\ V| >= r."""
    top = set(top)
    for size in range(k, n):
        for V in itertools.combinations(range(n), size):
            if len(top - set(V)) >= r:
                yield V


def brute_force_critical_value(s, mu, sigma, k, r=1, ids=None) -> Tuple[float, int]:
    """max over unsafe V of lambda*(V); returns (value, number_of_unsafe_sets)."""
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    top = true_topk(s, k, ids).tolist()
    best, count = -math.inf, 0
    for V in unsafe_sets(len(s), k, top, r):
        count += 1
        best = max(best, lambda_star(V, s, mu, sigma, k))
    return float(best), count


# --------------------------------------------------------------------------- #
# Cost anatomy of the blocker-first schedule (Proposition 10 in docs/THEORY.md)
# --------------------------------------------------------------------------- #
def checkpoints_from_schedule(N: int, k: int, schedule: Callable[[int], int]) -> List[int]:
    """Cumulative scored-set sizes at which the controller evaluates the rule."""
    out, cur, t = [], 0, 0
    while cur < N:
        cur = min(N, cur + max(1, int(schedule(t))))
        t += 1
        out.append(cur)
    return out


def cost_anatomy(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k: int,
    lam: float,
    ids: Optional[Sequence] = None,
    checkpoints: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Decompose the teacher calls of the guardrail-descending ("blocker-first") schedule.

    Definitions (Proposition 10).  Let theta = s_(k) and
        V* = T_k  U  {j : U_j(lambda) >= theta}            (the certification core, eq. 17)
    For a query with lambda >= A_{k,1}(q) ("covered"):
      (a) every legal policy that stops early scores at least |V*| documents, and V* itself
          satisfies the stopping rule, so |V*| is the instance-optimal cost;
      (b) the one-at-a-time U-descending policy stops after exactly min(N, |V*| + band) calls,
          where band = #{ j not in T_k : U_j < theta and j precedes the last winner in U order };
      (c) with checkpoints c_1 < c_2 < ... the cost is the first checkpoint >= that number,
          i.e. an extra "overshoot" of at most max batch size - 1.
    Returns the three terms and the resulting predicted call count (all integers as floats).
    The function needs the full teacher scores: it is an offline diagnostic, never online.
    """
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    validate_instance(s, mu, sigma)
    n = len(s)
    if not (1 <= k < n):
        raise ValueError("require 1 <= k < N")
    ids_arr = _as_ids(n, ids)
    order = teacher_order(s, ids)
    top = order[:k]
    theta = float(s[order[k - 1]])
    U = guardrail(mu, sigma, lam)
    A = critical_value(s, mu, sigma, k, 1, ids)
    covered = bool(lam >= A)
    # "U_j >= theta" is decided in the *normalised* form (theta - mu_j)/sigma_j <= lambda, the same
    # expression (and rounding) the controller uses in can_stop (Sec. 8 of docs/THEORY.md); the
    # guardrail form mu_j + lambda*sigma_j >= theta can differ by one document at lambda == A(q).
    reach = ((theta - mu) / sigma) <= lam if not np.isinf(lam) else np.full(n, lam > 0)
    core = set(int(i) for i in top) | {int(j) for j in np.where(reach)[0]}
    # U-descending order (ties by identifier).  With a shared sigma and finite lambda this order *is* the
    # mu-descending main policy in exact arithmetic; sort by mu directly in that case so that a
    # floating-point tie of mu + lambda*sigma between distinct mu values cannot reorder the documents.
    shared_sigma = bool(np.all(sigma == sigma[0]))
    u_order = list(np.lexsort((ids_arr, -mu))) if (shared_sigma and np.isfinite(lam)) else list(np.lexsort((ids_arr, -U)))
    pos = {int(p): r for r, p in enumerate(u_order)}  # 0-based position in the U order
    last_winner = max(pos[int(i)] for i in top)
    band = [int(p) for p in u_order[: last_winner + 1] if int(p) not in set(int(i) for i in top) and not reach[int(p)]]
    m_star = min(n, max(last_winner + 1, int(np.sum(reach)), k))
    if checkpoints:
        cps = [c for c in checkpoints if c >= k]
        calls = next((c for c in cps if c >= m_star), n)
    else:
        calls = m_star
    return {
        "covered": covered,
        "A_k1": float(A),
        "lambda": float(lam),
        "theta": theta,
        "cert_core": float(len(core)),          # |V*| = C*_cert(q), the instance-optimal cost
        "band": float(len(band)),               # extra calls of the blocker-first schedule
        "m_star": float(m_star),                # |V*| + band (capped at N)
        "calls_predicted": float(calls),        # after batching
        "overshoot": float(calls - m_star),
        "N": float(n),
        "winners_below_theta": float(int(np.sum(~reach[top]))),
    }
