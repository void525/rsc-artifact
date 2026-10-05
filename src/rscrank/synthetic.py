"""Synthetic instances for unit tests and the end-to-end smoke pipeline.

Nothing here is a retrieval result.  The generator produces query bundles whose
teacher scores, cheap predictions and per-query scales have a controllable
relationship so that every code path (early stop, full fallback, ties,
negative correlation, huge irrelevant residuals) is exercised.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .replay import QueryBundle

__all__ = ["make_bundle", "make_dataset", "small_instances"]


def make_bundle(rng: np.random.Generator, qid: str, N: int = 100, noise: float = 1.0, shared_sigma: bool = True, tie_prob: float = 0.0, outlier_prob: float = 0.05, n_rel: int = 3) -> QueryBundle:
    ids = [f"{qid}-d{j:03d}" for j in range(N)]
    # teacher scores: a few clearly relevant docs, the rest spread out
    s = rng.normal(-2.0, 2.0, N)
    rel = rng.choice(N, size=n_rel, replace=False)
    s[rel] += rng.uniform(3, 6, n_rel)
    if tie_prob > 0:  # inject exact ties
        for j in range(1, N):
            if rng.random() < tie_prob:
                s[j] = s[j - 1]
    q_scale = float(np.exp(rng.normal(np.log(noise), 0.4)))
    mu = s + rng.normal(0, q_scale, N)
    # occasional huge underestimation of an irrelevant document (drives B_all up, not A)
    if rng.random() < outlier_prob:
        j = int(rng.choice(np.setdiff1d(np.arange(N), rel)))
        mu[j] = s[j] - 50.0
    sigma = np.full(N, q_scale * rng.uniform(0.8, 1.25)) if shared_sigma else np.abs(rng.normal(q_scale, 0.2 * q_scale, N)) + 1e-3
    cheap_raw = mu + rng.normal(0, 0.5, N)
    bm25 = np.clip(rng.normal(10, 4, N) + 0.8 * s, 0, None)
    length = rng.integers(30, 500, N).astype(float)
    tokens = np.minimum(length * 1.3 + 40, 512)
    qrels = {ids[j]: 1 for j in rel[: max(1, n_rel - 1)]}
    feats = {"cheap_raw": cheap_raw, "bm25": bm25, "length": length, "tokens": tokens}
    return QueryBundle(qid, ids, s, mu, sigma, feats, qrels)


def make_dataset(n_queries: int, seed: int = 0, N: int = 100, **kw) -> List[QueryBundle]:
    rng = np.random.default_rng(seed)
    return [make_bundle(rng, f"q{i:05d}", N=N, **kw) for i in range(n_queries)]


def small_instances(seed: int = 0, n_instances: int = 200, N_range=(4, 9), tie_frac: float = 0.3):
    """Small float instances for exhaustive tests: returns list of (s, mu, sigma, k)."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_instances):
        N = int(rng.integers(N_range[0], N_range[1] + 1))
        k = int(rng.integers(1, N))
        s = rng.normal(0, 3, N)
        if rng.random() < tie_frac:
            # random ties, including ties at the top-k boundary
            for _ in range(int(rng.integers(1, N))):
                a, b = rng.integers(0, N, 2)
                s[a] = s[b]
        mu = s + rng.normal(0, 2, N)
        sigma = np.exp(rng.normal(0, 0.5, N)) if rng.random() < 0.5 else np.full(N, float(np.exp(rng.normal(0, 0.5))))
        out.append((s, mu, sigma, k))
    return out
