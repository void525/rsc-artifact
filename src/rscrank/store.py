"""Aligned per-(query, candidate) array stores (NPZ) used by the pipeline.

A store for (domain, split) holds arrays aligned on axis 0 = query (in the
order of the candidates file) and axis 1 = candidate rank (1..N).  Queries with
fewer than N candidates are right-padded and flagged in ``valid``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .io import ensure_dir, read_jsonl

__all__ = ["load_candidates", "load_docstore", "save_store", "load_store"]


def load_candidates(path: Path) -> List[Dict]:
    return list(read_jsonl(path))


def load_docstore(path: Path) -> Dict[str, Dict[str, str]]:
    return {r["docid"]: {"title": r.get("title", ""), "text": r.get("text", "")} for r in read_jsonl(path)}


def save_store(path: Path, qids: List[str], docids: List[List[str]], arrays: Dict[str, np.ndarray], N: int, meta: Optional[Dict] = None) -> None:
    ensure_dir(Path(path).parent)
    Q = len(qids)
    doc_arr = np.full((Q, N), "", dtype=object)
    valid = np.zeros((Q, N), dtype=bool)
    for i, ds in enumerate(docids):
        doc_arr[i, : len(ds)] = ds
        valid[i, : len(ds)] = True
    payload = {"qids": np.asarray(qids, dtype=str), "docids": doc_arr.astype(str), "valid": valid, "N": np.array(N)}
    payload.update(arrays)
    if meta:
        import json

        payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **payload)


def load_store(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["qids"] = out["qids"].astype(str)
    out["docids"] = out["docids"].astype(str)
    return out


def store_meta(path: Path) -> Dict:
    """Read only the meta_json of a store (cheap)."""
    import json

    z = np.load(path, allow_pickle=False)
    if "meta_json" in z.files:
        return json.loads(str(z["meta_json"]))
    return {}
