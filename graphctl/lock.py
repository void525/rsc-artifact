"""Cross-platform advisory file lock (no third-party dependency).

Several subagents may run graphctl commands at the same time; every read-modify-write of
graph/state.json happens inside ``state_txn()`` which uses this lock.

Design notes
  * ``O_CREAT | O_EXCL`` is the portable atomic "create if absent" primitive (works on NTFS and on
    every POSIX filesystem we care about; it is *not* safe over SMB/NFS, which this project never uses).
  * The lock file carries a unique token (pid + random).  The holder only ever deletes a lock file whose
    token is still its own, so breaking a stale lock can never make two processes delete each other's
    fresh locks in a loop.
  * A lock older than ``STALE_S`` is broken: an agent killed by a Windows Update reboot or by Cursor's
    idle-shell reaper must not wedge the five-week loop.
  * If the lock cannot be taken within ``timeout`` the context manager still runs the body, but yields
    ``False`` so the caller can record that the write was unsynchronised.  Blocking forever would be worse.
"""
from __future__ import annotations

import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path

STALE_S = 120.0


def _token() -> str:
    return f"{os.getpid()}:{secrets.token_hex(8)}"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


@contextmanager
def file_lock(path: Path, timeout: float = 60.0, poll: float = 0.05):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mine = _token()
    t0 = time.time()
    acquired = False
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, mine.encode("ascii"))
            finally:
                os.close(fd)
            acquired = True
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                continue  # the holder released it between the two calls: retry at once
            if age > STALE_S:
                # Break the stale lock, but only the exact file we just inspected.
                stale_token = _read(path)
                try:
                    if _read(path) == stale_token:
                        path.unlink()
                except OSError:
                    pass
                continue
            if time.time() - t0 > timeout:
                break  # never block the loop forever: run unlocked and say so
            time.sleep(poll)
        except OSError:
            # e.g. a transient Windows sharing violation while another process unlinks the file
            if time.time() - t0 > timeout:
                break
            time.sleep(poll)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                if _read(path) == mine:
                    path.unlink()
            except OSError:
                pass
