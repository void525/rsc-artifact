"""Shared helpers for pipeline scripts: config loading, synthetic overrides, paths, logging."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

try:  # Windows consoles default to a legacy code page; our logs contain a few non-ASCII characters
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SYNTHETIC_OVERRIDES: Dict[str, Any] = {
    "paths": {"data_raw": "runs/smoke/data/raw/beir", "train_questions": "runs/smoke/data/raw/train_questions", "splits": "runs/smoke/data/splits", "candidates": "runs/smoke/data/candidates", "features": "runs/smoke/data/features", "teacher": "runs/smoke/data/teacher", "frozen": "runs/smoke/data/frozen", "models": "runs/smoke/models", "runs": "runs/smoke/runs"},
    "domains": {"main": ["nq", "hotpotqa"], "cross": ["fiqa", "scifact"], "extra": []},  # the smoke run never needs the optional extra domain
    "splits": {"nq": {"n_cal": 60, "n_train": 120, "n_dev": 40}, "hotpotqa": {"n_cal": 60, "n_train": 120, "n_dev": 40}, "fiqa": {"cross_domain_only": True}, "scifact": {"cross_domain_only": True}, "dedup_jaccard": 0.8},
    "candidates": {"backend": "python", "N": 30, "N_grid": [15, 30], "extended_subset_queries": 20, "sanity_tolerance_ndcg": 1.0},
    "cheap": {"model": "fake", "batch_size": 64},
    "teachers": {"main": "fake", "second": "fake2", "fake": {"seed": 0}, "fake2": {"seed": 1}, "determinism_pairs": 50},
    "heads": {"hidden": 16, "kappa": 1.0, "lr_grid": [0.003], "wd_grid": [0.0], "epochs": 8, "patience": 3, "batch_size": 64, "selection_alpha": 0.1, "pooled_across_domains": True},
    "protocol": {"k": 5, "k_grid": [1, 5], "alpha_main": 0.1, "alpha_grid": [0.1, 0.2], "r_grid": [1, 2], "regimes": ["marginal", "pac", "finite"], "delta_family": 0.2, "n_main_configs": 4, "batch_first": 8, "batch_rest": 4, "batch_grid": [4, 8], "policies": ["mu", "cheap_raw", "bm25", "length_asc", "random", "observed_regression"], "noninferiority_epsilon": 0.02, "bootstrap": 200, "latency_queries": 10, "latency_repeats": 1},
    "smoke": {"synthetic": True},
}


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "configs" / "main.yaml"))
    p.add_argument("--synthetic", action="store_true", help="smoke mode: tiny synthetic data + fake models under runs/smoke/")
    p.add_argument("--domains", default=None, help="comma-separated subset of domains")
    return p


def load_config(args) -> Dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if getattr(args, "synthetic", False):
        cfg = deep_update(cfg, SYNTHETIC_OVERRIDES)
    cfg["_root"] = str(ROOT)
    return cfg


def P(cfg: Dict[str, Any], key: str) -> Path:
    return ROOT / cfg["paths"][key]


def domains(cfg: Dict[str, Any], args, include_cross: bool = True):
    ds = list(cfg["domains"]["main"]) + list(cfg["domains"].get("extra", []) or []) + (list(cfg["domains"]["cross"]) if include_cross else [])
    ds = list(dict.fromkeys(ds))  # de-duplicate, order preserving
    if getattr(args, "domains", None):
        want = [d.strip() for d in args.domains.split(",") if d.strip()]
        ds = [d for d in ds if d in want]
    return ds


def teachers(cfg: Dict[str, Any]):
    return [cfg["teachers"]["main"], cfg["teachers"]["second"]]


def main_domains(cfg: Dict[str, Any]):
    """Domains that get their own calibration + test split (main + optional extra, e.g. msmarco)."""
    return list(dict.fromkeys(list(cfg["domains"]["main"]) + list(cfg["domains"].get("extra", []) or [])))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def runs_dir(cfg: Dict[str, Any], *parts: str) -> Path:
    p = P(cfg, "runs").joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
