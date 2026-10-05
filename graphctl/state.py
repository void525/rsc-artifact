"""Persistent state of the graph loop (graph/state.json) and helpers."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import secrets
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from .lock import file_lock

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "graph"
STATE_PATH = GRAPH_DIR / "state.json"
LEDGER_PATH = GRAPH_DIR / "ledger.md"
BLOCKERS_PATH = GRAPH_DIR / "blockers.md"
HALT_PATH = GRAPH_DIR / "HALT"
RUNS = ROOT / "runs"

NODE_STATES = ("pending", "in_progress", "verified", "done", "blocked", "skipped")
LOCK_PATH = GRAPH_DIR / ".state.lock"

# Ledger prefixes that count as "the loop moved" (see progress_hash).
# NB: "VERIFY" is deliberately NOT here -- a verify that keeps failing is not progress, and counting it
# would make the no-progress self-pause unreachable (the loop would spin forever with nobody watching).
_PROGRESS_PREFIXES = ("START", "DONE", "FAIL", "BLOCK", "UNBLOCK", "SKIP", "JOB", "LADDER", "DECIDED", "RESUME", "NOTE")


def _empty_state() -> Dict[str, Any]:
    return {"version": 1, "status": "RUNNING", "nodes": {}, "followups": {"streak": 0, "last_progress_hash": ""}, "history": []}


@contextmanager
def state_txn(timeout: float = 60.0):
    """Load -> modify -> save graph/state.json under a file lock (parallel subagents).

    Yields the state dict.  ``state["_locked"]`` is not written to disk; callers that care can read
    ``txn_unsynchronised()`` afterwards.  The state is saved only when the body does not raise, so a
    crashing command never leaves a half-written transition behind."""
    with file_lock(LOCK_PATH, timeout=timeout) as acquired:
        st = load_state()
        if not acquired:
            # Extremely rare (another graphctl held the lock for >60 s).  Record it rather than hide it.
            print(f"warning: graph/state.json written without the file lock (busy > {timeout:g} s)", file=sys.stderr)
        yield st
        save_state(st)


def py_cmd() -> str:
    """How to spell the project interpreter in a message (PowerShell and bash both accept this).

    Before ``bootstrap.py`` has run there is no ``.venv``; fall back to the *absolute* path of the
    interpreter currently running graphctl rather than the bare name ``python``, which on Windows may
    resolve to the Microsoft Store stub or to a different conda environment."""
    port = ROOT / ".venv" / "Scripts" / "python.exe"
    if port.exists():
        return "./.venv/Scripts/python.exe"
    exe = sys.executable or "python"
    return f'"{exe}"' if " " in exe else exe


def now_iso() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def load_state() -> Dict[str, Any]:
    """Read graph/state.json, tolerating a truncated file (power cut / Windows Update reboot).

    A corrupt state file must never make every graphctl command crash for the rest of the run: the
    broken file is moved aside and an empty state is returned, which `next` then re-derives from the
    graph (nodes whose outputs exist verify immediately)."""
    if not STATE_PATH.exists():
        return _empty_state()
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
    except Exception as e:  # noqa: BLE001  (json error, decoding error, truncated file)
        bad = STATE_PATH.with_name(f"state.corrupt-{_dt.datetime.now():%Y%m%d-%H%M%S}.json")
        try:
            shutil.copy2(STATE_PATH, bad)
        except OSError:
            bad = None
        print(f"warning: graph/state.json was unreadable ({e}); starting from an empty state"
              + (f" (copy kept at {bad.name})" if bad else ""), file=sys.stderr)
        st = _empty_state()
        append_ledger(f"STATE RECOVERED: graph/state.json was unreadable ({e})")
        return st
    if not isinstance(st, dict) or "nodes" not in st:
        return _empty_state()
    st.setdefault("status", "RUNNING")
    st.setdefault("followups", {"streak": 0, "last_progress_hash": ""})
    st.setdefault("history", [])
    return st


def save_state(state: Dict[str, Any]) -> None:
    """Atomic, crash-safe write.

    The temp file name is unique per process: two graphctl processes that both lost the lock must not
    write the same ``state.tmp``.  ``os.replace`` is atomic on NTFS as well, but it raises
    PermissionError if the destination is momentarily open by another process (antivirus, a Windows
    indexer, another graphctl reader), so it is retried briefly.  NOT VERIFIED ON WINDOWS from this
    Linux sandbox -- the retry loop is the defensive measure."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(f"state.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    last = None
    for attempt in range(10):
        try:
            os.replace(tmp, STATE_PATH)
            return
        except OSError as e:  # Windows: destination briefly open elsewhere
            last = e
            time.sleep(0.05 * (attempt + 1))
    try:
        tmp.unlink()
    except OSError:
        pass
    raise OSError(f"could not replace {STATE_PATH}: {last}")


def node_state(state: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    ns = state.setdefault("nodes", {}).setdefault(
        node_id, {"status": "pending", "iterations": 0, "notes": [], "verify": None, "review": None})
    ns.setdefault("status", "pending")
    ns.setdefault("iterations", 0)
    ns.setdefault("jobs", [])
    ns.setdefault("notes", [])
    return ns


def append_ledger(line: str) -> None:
    """One short append per call: on both POSIX and Windows a single small write to a file opened in
    append mode is not interleaved with other writers, so parallel subagents cannot corrupt a line."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER_PATH.exists()
    text = ("# Ledger (append-only, machine + agent written)\n\n" if new else "") + f"- {now_iso()} {line}\n"
    try:
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:  # a full disk must not abort the command that was making progress
        print(f"warning: could not append to graph/ledger.md: {e}", file=sys.stderr)


def _ledger_progress_count() -> int:
    """Number of *meaningful* ledger entries.

    Deliberately a count of matching lines and not the file size: the byte size changes whenever any
    observability line is appended (it used to include one line per finished subagent), which reset the
    no-progress detector every turn and made the self-pause unreachable."""
    if not LEDGER_PATH.exists():
        return 0
    n = 0
    try:
        with open(LEDGER_PATH, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                body = line.lstrip("- ").strip()
                # "- <iso timestamp> WORD ..."
                parts = body.split(" ", 1)
                if len(parts) == 2 and parts[1].split(":")[0].split(" ")[0].upper() in _PROGRESS_PREFIXES:
                    n += 1
    except OSError:
        return 0
    return n


def progress_hash(state: Dict[str, Any]) -> str:
    """Hash of everything that counts as progress: node statuses/iterations, the number of meaningful
    ledger entries and running-job heartbeats."""
    h = hashlib.sha256()
    for nid in sorted(state.get("nodes", {})):
        ns = state["nodes"][nid]
        h.update(f"{nid}:{ns.get('status')}:{ns.get('iterations', 0)}:{len(ns.get('notes', []))}".encode())
    h.update(f"ledger:{_ledger_progress_count()}".encode())
    jobs_dir = RUNS / "jobs"
    if jobs_dir.exists():
        try:
            entries = sorted(p for p in jobs_dir.iterdir() if p.is_dir())
        except OSError:
            entries = []
        for d in entries:
            hb = d / "heartbeat"
            try:
                if hb.exists():
                    h.update(f"{d.name}:{hb.read_text(encoding='utf-8', errors='ignore').strip()}".encode())
            except OSError:
                pass
    return h.hexdigest()[:16]


def needs_you_path() -> Path:
    """Resolved at call time: tests (and any relocated checkout) monkeypatch ROOT."""
    return ROOT / "NEEDS_YOU.txt"


def write_needs_you(reason: str, what_human_can_do: str) -> None:
    """One loud file at the repo root: the human's watch-file when the loop cannot continue alone.
    (graph/blockers.md keeps the history; this file is the thing nobody can miss.)"""
    text = (
        "需要你处理（The autonomous loop is waiting for a human）\n"
        f"时间 / time: {now_iso()}\n\n"
        f"原因 / reason:\n{reason.strip()}\n\n"
        f"怎么继续 / how to continue:\n{what_human_can_do.strip()}\n\n"
        "细节见 graph/blockers.md 与 graph/ledger.md；处理完后在 Cursor 里说 `继续`（或运行 graphctl resume）。\n"
    )
    try:
        needs_you_path().write_text(text, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def clear_needs_you() -> None:
    try:
        needs_you_path().unlink()
    except OSError:
        pass


def write_blocker(node_id: str, reason: str, what_human_can_do: Optional[str] = None) -> None:
    BLOCKERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not BLOCKERS_PATH.exists()
    head = ("# Blockers\n\nThe agent writes here ONLY when a decision is genuinely outside its authority\n"
            "(see graph/decisions.md).  Everything else it decides itself.\n\n") if new else ""
    body = f"{head}## {now_iso()} -- {node_id}\n\n{reason.strip()}\n\n"
    if what_human_can_do:
        body += f"**What the human can do:** {what_human_can_do.strip()}\n\n"
    try:
        with open(BLOCKERS_PATH, "a", encoding="utf-8") as f:
            f.write(body)
    except OSError as e:
        print(f"warning: could not write graph/blockers.md: {e}", file=sys.stderr)
    # blockers.md is the history; NEEDS_YOU.txt at the repo root is the file the human cannot miss
    write_needs_you(reason, what_human_can_do or "See graph/blockers.md.")
