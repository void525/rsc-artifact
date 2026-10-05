"""Legal document-scoring policies (Sec. 3.4).

A policy is a callable ``policy(remaining_ids, observed, mu_by_id) -> list``
returning the remaining ids in the order they should be scored.  It may use
cheap per-document features (bound at construction time) and the teacher
scores already observed, but never unscored teacher scores.  All policies are
deterministic given their seed.
"""
from __future__ import annotations

import hashlib
import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

Policy = Callable[[Sequence, Dict, Dict[object, float]], List]

__all__ = [
    "mu_descending",
    "mu_binned_length",
    "feature_order",
    "random_order",
    "observed_regression",
    "alternating",
    "make_policy",
    "POLICY_NAMES",
]


def _sort_ids(ids: Sequence, key: Dict[object, float], descending: bool = True) -> List:
    """Sort by ``key`` (descending by default); ties broken by ascending identifier.

    The tie rule is the *natural* order of the identifiers (numeric for ints, lexicographic
    for strings) -- exactly the order ``numpy.lexsort`` uses in ``core.teacher_order`` /
    ``core.cost_anatomy`` -- so that the paper's fixed total order and the policies agree
    for every identifier type.  Mixed-type identifiers (not used anywhere) fall back to the
    string order."""
    ids = list(ids)
    try:
        return sorted(ids, key=lambda i: ((-key[i] if descending else key[i]), i))
    except TypeError:  # mixed identifier types
        return sorted(ids, key=lambda i: ((-key[i] if descending else key[i]), str(i)))


def mu_descending() -> Policy:
    """Main policy: score documents in descending order of the learned prediction mu."""

    def policy(remaining, observed, mu_by_id):
        return _sort_ids(remaining, mu_by_id, descending=True)

    return policy


def feature_order(feature_by_id: Dict[object, float], descending: bool = True) -> Policy:
    """Order by any fixed cheap feature: raw cheap score, BM25 score, or document length
    (length-ascending = short documents first)."""

    def policy(remaining, observed, mu_by_id):
        return _sort_ids(remaining, feature_by_id, descending=descending)

    return policy


def random_order(seed: int = 0) -> Policy:
    """Uniformly random order fixed per query by the seed and the id set."""

    def policy(remaining, observed, mu_by_id):
        ids = sorted(remaining, key=str)
        # stable (process-independent) hash of the id set; Python's hash() is salted per process
        h = int(hashlib.sha256("|".join(map(str, ids)).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng([seed, len(ids), h])
        perm = rng.permutation(len(ids))
        return [ids[p] for p in perm]

    return policy


def observed_regression(features: Optional[Dict[str, Dict[object, float]]] = None, min_obs: int = 8, ridge: float = 1.0) -> Policy:
    """Adaptive re-ranking of the remaining documents from the teacher scores observed so far (amendment A3).

    Once >= ``min_obs`` teacher scores are observed, fit a ridge regression of the observed scores on the
    standardised cheap features [mu, cheap_raw, bm25, log(1+length)] (whatever is available) and order the
    remaining documents by the refit prediction.  Unlike an affine refit of s on mu alone -- which preserves the
    mu order whenever the fitted slope is positive and was therefore a no-op -- the multi-feature refit changes
    the order per query.  Before ``min_obs`` observations the mu order is used.  Uses only cheap information and
    observed scores: a legal policy."""
    features = features or {}
    cols = [name for name in ("cheap_raw", "bm25", "length") if name in features]

    def _x(i, mu_by_id):
        row = [mu_by_id[i]]
        for name in cols:
            v = float(features[name][i])
            row.append(math.log1p(max(v, 0.0)) if name == "length" else v)
        return row

    def policy(remaining, observed, mu_by_id):
        if len(observed) >= min_obs:
            obs_ids = list(observed)
            X = np.array([_x(i, mu_by_id) for i in obs_ids], dtype=float)
            y = np.array([observed[i] for i in obs_ids], dtype=float)
            m, sd = X.mean(axis=0), X.std(axis=0)
            sd[sd == 0] = 1.0
            Xs = (X - m) / sd
            A = Xs.T @ Xs + ridge * np.eye(Xs.shape[1])
            w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
            R = (np.array([_x(i, mu_by_id) for i in remaining], dtype=float) - m) / sd
            pred = R @ w
            key = {i: float(v) for i, v in zip(remaining, pred)}
            return _sort_ids(remaining, key, descending=True)
        return _sort_ids(remaining, mu_by_id, descending=True)

    return policy


def alt_model_order(mu_alt_by_id: Dict[object, float]) -> Policy:
    """Order by a *different* cheap model's prediction (e.g. an upgraded or downgraded first stage) while the
    guardrail keeps the calibrated heads: the 'ordering model changed, calibration did not' scenario."""
    return feature_order(mu_alt_by_id, True)


def mu_binned_length(length_by_id: Dict[object, float], window: int = 24) -> Policy:
    """Cost-aware blocker-first policy (Sec. 16, E2/E3).

    Within the ``window`` most promising remaining documents (by mu, i.e. by the guardrail when
    sigma is shared) the shortest documents go first, so a round fills its GPU micro-batches with
    similar, short inputs; everything after the window keeps mu order.  This changes *which*
    documents a round scores, i.e. it is a genuinely different legal policy -- the kind of
    scheduling change a systems team makes for throughput, and the kind that invalidates a
    trajectory-specific threshold but not a policy-uniform one."""

    def policy(remaining, observed, mu_by_id):
        head = _sort_ids(remaining, mu_by_id, descending=True)
        w = max(1, int(window))
        front, back = head[:w], head[w:]
        try:
            front = sorted(front, key=lambda i: (float(length_by_id[i]), -mu_by_id[i], i))
        except TypeError:  # mixed identifier types
            front = sorted(front, key=lambda i: (float(length_by_id[i]), -mu_by_id[i], str(i)))
        return front + back

    return policy


def alternating(other_feature_by_id: Dict[object, float]) -> Policy:
    """Interleave mu order with another cheap order (e.g. BM25): positions
    0,2,4,... from mu, 1,3,5,... from the other feature, without duplicates."""

    def policy(remaining, observed, mu_by_id):
        a = _sort_ids(remaining, mu_by_id, True)
        b = _sort_ids(remaining, other_feature_by_id, True)
        out, seen = [], set()
        for x, y in zip(a, b):
            for z in (x, y):
                if z not in seen:
                    seen.add(z)
                    out.append(z)
        return out

    return policy


POLICY_NAMES = ["mu", "cheap_raw", "bm25", "length_asc", "length_desc", "random", "observed_regression", "alternating_bm25", "mu_binned_length", "mu_alt"]


def make_policy(name: str, features: Optional[Dict[str, Dict[object, float]]] = None, seed: int = 0) -> Policy:
    """Factory used by the replay evaluator.  ``features`` maps feature name ->
    {id: value}; required keys depend on the policy."""
    features = features or {}
    if name == "mu":
        return mu_descending()
    if name == "cheap_raw":
        return feature_order(features["cheap_raw"], True)
    if name == "bm25":
        return feature_order(features["bm25"], True)
    if name == "length_asc":
        return feature_order(features["length"], False)
    if name == "length_desc":
        return feature_order(features["length"], True)
    if name == "random":
        return random_order(seed)
    if name == "observed_regression":
        return observed_regression(features)
    if name == "mu_alt":
        return alt_model_order(features["mu_alt"])
    if name == "alternating_bm25":
        return alternating(features["bm25"])
    if name == "mu_binned_length":
        return mu_binned_length(features["length"], window=int((features.get("_window") or {}).get("window", 24)) if isinstance(features.get("_window"), dict) else 24)
    raise ValueError(f"unknown policy {name!r}; choose from {POLICY_NAMES}")
