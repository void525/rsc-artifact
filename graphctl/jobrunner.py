"""Detached runner: executes a job's command, streams output to log.txt, touches heartbeat every 20 s,
writes status.json at exit and releases the GPU lock.  Invoked by graphctl.jobs.start; not meant to be
called by hand."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HEARTBEAT_S = 20


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_json_atomic(p: Path, obj: dict) -> None:
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(tmp, p)
                return
            except OSError:
                time.sleep(0.05 * (attempt + 1))
    except OSError:
        pass
    try:
        tmp.unlink()
    except OSError:
        pass


def _finish(d: Path, meta: dict, log, payload: dict) -> None:
    try:
        log.flush()
        log.close()
    except Exception:  # noqa: BLE001
        pass
    _write_json_atomic(d / "status.json", payload)
    _release(d, meta)


def main(job_dir: str) -> int:
    d = Path(job_dir)
    meta = json.loads((d / "cmd.json").read_text(encoding="utf-8"))
    log = open(d / "log.txt", "a", encoding="utf-8", errors="replace")
    log.write(f"=== job {meta['name']} started {_stamp()} ===\n")
    log.write("cmd: " + " ".join(str(c) for c in meta["cmd"]) + "\n\n")
    log.flush()
    if (d / "kill.request").exists():  # killed between `jobs start` and the runner getting scheduled
        log.write("kill requested before start; not launching\n")
        _finish(d, meta, log, {"exit_code": -9, "finished_at": _stamp(), "killed": True})
        return -9
    # Force UTF-8 in the child: a single non-ASCII progress character must not kill a 30-hour job on a
    # Windows console whose active code page is cp936/cp1252.
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "GRAPHCTL_JOB": str(meta["name"])}
    kwargs = {"cwd": meta.get("cwd") or None, "stdout": log, "stderr": subprocess.STDOUT,
              "stdin": subprocess.DEVNULL, "env": env}
    if os.name != "nt":
        kwargs["start_new_session"] = True  # own process group so `jobs kill` can kill the whole tree
    try:
        p = subprocess.Popen([str(c) for c in meta["cmd"]], **kwargs)
    except Exception as e:  # noqa: BLE001
        log.write(f"FAILED TO START: {e}\n")
        _finish(d, meta, log, {"exit_code": 127, "finished_at": _stamp(), "error": str(e)})
        return 127
    _write_json_atomic(d / "child.json", {"pid": p.pid})
    hb = d / "heartbeat"
    while True:
        try:
            hb.write_text(_stamp(), encoding="utf-8")
        except OSError:
            pass  # a full disk must not orphan the child; `status` will report the job stale
        try:
            rc = p.wait(timeout=HEARTBEAT_S)
            break
        except subprocess.TimeoutExpired:
            if (d / "kill.request").exists() and p.poll() is None:
                continue  # `jobs kill` signals the process itself; we just keep the heartbeat honest
            continue
    log.write(f"\n=== exit code {rc} at {_stamp()} ===\n")
    _finish(d, meta, log, {"exit_code": rc, "finished_at": _stamp()})
    return rc if isinstance(rc, int) and 0 <= rc < 256 else (0 if rc == 0 else 1)


def _release(d: Path, meta: dict) -> None:
    if meta.get("gpu"):
        lock = d.parent / "GPU.lock"
        try:
            if lock.exists() and lock.read_text(encoding="utf-8").strip() == meta["name"]:
                lock.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
