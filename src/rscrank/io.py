"""Small I/O helpers shared by the pipeline scripts: JSONL, manifests, hashing."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

__all__ = ["read_jsonl", "write_jsonl", "sha256_file", "sha256_text", "write_manifest", "read_json", "write_json", "ensure_dir", "normalize_text"]


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=True, default=_default)
    os.replace(tmp, path)


def _default(o):
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(t: str) -> str:
    """Canonical form for exact de-duplication: lowercase, collapse whitespace,
    strip punctuation at the ends."""
    import re

    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" ?.!,;:'\"")
    return t


def write_manifest(path: Path, stage: str, inputs: Optional[List[Path]] = None, outputs: Optional[List[Path]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    man = {
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": [{"path": str(p), "sha256": sha256_file(Path(p)) if Path(p).is_file() else None} for p in (inputs or [])],
        "outputs": [{"path": str(p), "sha256": sha256_file(Path(p)) if Path(p).is_file() else None, "bytes": Path(p).stat().st_size if Path(p).exists() else None} for p in (outputs or [])],
    }
    if extra:
        man.update(extra)
    write_json(path, man)
    return man
