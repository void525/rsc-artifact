"""RSC-Rank: Rank-Sufficient Calibration for policy-uniform web reranking.

Public API (see individual modules for details):
    core       -- critical values, guardrail, controller, oracle, brute force refs
    calibrate  -- marginal / PAC / finite-population order-statistic indices
    policies   -- legal scoring policies
    baselines  -- prefix-budget baselines
    replay     -- cached replay evaluator
    metrics    -- exact mismatch, nDCG (trec_eval convention), etc.
    stats      -- risk intervals, paired bootstrap, non-inferiority
    provenance -- number registry -> paper/numbers.tex
    synthetic  -- synthetic instance generator for smoke tests
"""
__version__ = "0.1.0"

from . import baselines, calibrate, core, metrics, policies, provenance, replay, stats, synthetic  # noqa: F401
