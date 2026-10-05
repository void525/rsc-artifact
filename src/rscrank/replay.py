"""Cached-replay evaluator (Sec. 13).

Given per-query bundles that hold the *complete* teacher cache, this module
replays any method while exposing teacher scores to the online controller only
through :class:`rscrank.core.TeacherOracle`.  Replay time is **not** online
latency; latency is measured separately by scripts/09_system_latency.py.

The evaluator produces one flat record per (method, query) with risk, quality
and cost fields so that scripts/10_stats.py can aggregate them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import core
from .baselines import adaptive_prefix_budget, learned_budget, prefix_topk
from .metrics import exact_mismatch, missed_winners, mrr_at_k, ndcg_at_k, overlap_at_k, recall_at_k
from .policies import Policy, make_policy

__all__ = ["QueryBundle", "query_statistics", "MethodSpec", "replay_query", "replay_dataset", "check_call_dominance"]


@dataclass
class QueryBundle:
    qid: str
    ids: List
    s: np.ndarray                       # full teacher scores (evaluator-only)
    mu: np.ndarray
    sigma: np.ndarray
    features: Dict[str, np.ndarray] = field(default_factory=dict)   # cheap_raw, bm25, length, tokens, h_pred
    qrels: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.s = np.asarray(self.s, dtype=float)
        self.mu = np.asarray(self.mu, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if np.isscalar(self.sigma) or self.sigma.ndim == 0:
            self.sigma = np.full_like(self.mu, float(self.sigma))
        core.validate_instance(self.s, self.mu, self.sigma)
        if len(self.ids) != len(self.s):
            raise ValueError("ids/s mismatch")

    @property
    def N(self) -> int:
        return len(self.ids)

    def feature_map(self, name: str) -> Dict:
        arr = self.features[name]
        return {i: float(v) for i, v in zip(self.ids, arr)}

    def true_topk_ids(self, k: int) -> List:
        return [self.ids[p] for p in core.true_topk(self.s, k, self.ids)]


def query_statistics(b: QueryBundle, k: int, r_values: Sequence[int] = (1,)) -> Dict[str, float]:
    """All offline query-level statistics needed for calibration."""
    out = {"qid": b.qid, "N": b.N, "k": k}
    for r in r_values:
        if 1 <= r <= min(k, b.N - k):
            out[f"A_k{r}"] = core.critical_value(b.s, b.mu, b.sigma, k, r, b.ids)
    out["B_top"] = core.B_top(b.s, b.mu, b.sigma, k, b.ids)
    out["B_all"] = core.B_all(b.s, b.mu, b.sigma, b.ids)
    out["H_k"] = core.H_k(b.s, b.mu, k, b.ids)
    order = core.teacher_order(b.s, b.ids)
    out["gap_k"] = float(b.s[order[k - 1]] - b.s[order[k]])
    return out


@dataclass
class MethodSpec:
    """What to replay.  ``kind`` in {full, cheap_only, guardrail, prefix_fixed,
    prefix_learned, prefix_uniform, prefix_adaptive}.  For ``guardrail`` the
    threshold ``lam`` may come from A, B_top, B_all or a trajectory statistic
    -- the controller does not care, which is the point of the experiment."""
    name: str
    kind: str
    lam: float = math.inf
    policy: str = "mu"
    policy_seed: int = 0
    batch_first: Optional[int] = None
    batch_rest: Optional[int] = None
    schedule: str = "fixed"          # "fixed" (first/rest) or "blockers" (state-dependent, core.blocker_batch_schedule)
    m: Optional[int] = None          # prefix_fixed / prefix_uniform
    t_H: Optional[float] = None      # prefix_adaptive
    c: Optional[float] = None        # prefix_learned
    extra: Dict = field(default_factory=dict)


def _cost_fields(b: QueryBundle, scored_ids: Sequence, trajectory: Sequence[Sequence]) -> Dict[str, float]:
    tok = b.feature_map("tokens") if "tokens" in b.features else None
    real = float(sum(tok[i] for i in scored_ids)) if tok else float("nan")
    padded = 0.0
    if tok:
        for batch in trajectory:
            if batch:
                padded += len(batch) * max(tok[i] for i in batch)
    return {"tokens_real": real, "tokens_padded": padded if tok else float("nan")}


def replay_query(b: QueryBundle, spec: MethodSpec, k: int, r_values: Sequence[int] = (1, 2, 3)) -> Dict:
    truth = b.true_topk_ids(k)
    oracle = core.TeacherOracle(b.ids, b.s)
    early, rounds, trajectory = False, 0, []
    if spec.kind == "full":
        oracle.score(b.ids)
        pred = truth
        rounds, trajectory = 1, [list(b.ids)]
    elif spec.kind == "cheap_only":
        cheap = core.teacher_order(b.mu, b.ids)[:k]
        pred = [b.ids[p] for p in cheap]
        rounds, trajectory = 0, []
    elif spec.kind == "guardrail":
        feats = {n: b.feature_map(n) for n in b.features}
        policy: Policy = make_policy(spec.policy, feats, spec.policy_seed)
        if spec.schedule == "blockers":
            first = spec.batch_first if spec.batch_first is not None else max(k + 6, k)
            sched = core.blocker_batch_schedule(min_batch=first if spec.batch_rest is None else spec.batch_rest)
            # the first round scores `first` documents like the main schedule; afterwards blocker batches
            base = sched

            def sched(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_, _base=base, _first=first):  # noqa: E306
                return _first if round_idx == 0 else _base(round_idx, observed, order, mu_by_id, sg_by_id, lam_, k_)

            sched.adaptive = True  # type: ignore[attr-defined]
        else:
            sched = core.default_batch_schedule(k, spec.batch_first, spec.batch_rest)
        res = core.run_controller(b.ids, b.mu, b.sigma, spec.lam, k, oracle, policy, sched)
        pred, early, rounds, trajectory = res.topk_ids, res.early_stopped, res.n_rounds, res.trajectory
    elif spec.kind in ("prefix_fixed", "prefix_uniform", "prefix_learned", "prefix_adaptive"):
        if spec.kind in ("prefix_fixed", "prefix_uniform"):
            m = int(spec.m)
        elif spec.kind == "prefix_learned":
            m = learned_budget(float(b.features["h_pred"][0]), spec.c, k, b.N)
        else:
            m = adaptive_prefix_budget(float(b.features["h_pred"][0]), spec.t_H, k, b.N)
        if spec.policy and spec.policy != "mu":
            # prefix of another scoring order (E2b): the order a policy would follow before any teacher score is observed
            feats = {n: b.feature_map(n) for n in b.features}
            order = list(make_policy(spec.policy, feats, spec.policy_seed)(list(b.ids), {}, {i: float(v) for i, v in zip(b.ids, b.mu)}))
            chosen = order[: max(k, min(m, b.N))]
        else:
            order = None
            cheap = core.teacher_order(b.mu, b.ids)[: max(k, min(m, b.N))]
            chosen = [b.ids[p] for p in cheap]
        sc = oracle.score(chosen)
        s_by_id = dict(zip(chosen, sc))
        pred, m_used = prefix_topk(b.ids, s_by_id, b.mu, m, k, order=order)
        rounds, trajectory, early = 1, [list(chosen)], m_used < b.N
    else:
        raise ValueError(f"unknown method kind {spec.kind}")

    rec = {
        "qid": b.qid,
        "method": spec.name,
        "kind": spec.kind,
        "N": b.N,
        "k": k,
        "n_calls": oracle.n_calls,
        "rounds": rounds,
        "early_stopped": bool(early),
        "full_fallback": bool(oracle.n_calls >= b.N),
        "mismatch": exact_mismatch(pred, truth),
        "missed": missed_winners(pred, truth),
        "overlap": overlap_at_k(pred, truth),
        "pred": list(pred),
    }
    for r in r_values:
        rec[f"ell_{r}"] = int(rec["missed"] >= r)
    rec.update(_cost_fields(b, oracle.scored_ids, trajectory))
    if b.qrels:
        rec["ndcg@k"] = ndcg_at_k(pred, b.qrels, k)
        rec["ndcg_full@k"] = ndcg_at_k(truth, b.qrels, k)
        rec["mrr@k"] = mrr_at_k(pred, b.qrels, k)
        rec["recall@k"] = recall_at_k(pred, b.qrels, k)
    return rec


def replay_dataset(bundles: Iterable[QueryBundle], spec: MethodSpec, k: int, r_values: Sequence[int] = (1, 2, 3)) -> List[Dict]:
    return [replay_query(b, spec, k, r_values) for b in bundles]


def check_call_dominance(recs_A: Sequence[Dict], recs_top: Sequence[Dict], recs_all: Sequence[Dict]) -> bool:
    """eq. 15: on the same path, Calls_A <= Calls_top <= Calls_all per query."""
    by = lambda recs: {r["qid"]: r["n_calls"] for r in recs}
    a, t, al = by(recs_A), by(recs_top), by(recs_all)
    return all(a[q] <= t[q] <= al[q] for q in a)
