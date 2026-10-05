"""Cross-platform background job manager for long-running steps (hours).

    python -m graphctl jobs start NAME [--gpu] [--node NODE] -- python scripts/05_teacher_scores.py ...
    python -m graphctl jobs status [NAME]
    python -m graphctl jobs wait NAME --timeout 600
    python -m graphctl jobs tail NAME [--lines 40]
    python -m graphctl jobs kill NAME

Each job lives in runs/jobs/NAME/ with cmd.json, log.txt, heartbeat (touched every 20 s by the
detached runner), status.json (exit code) and, for GPU jobs, holds runs/jobs/GPU.lock so that only one
GPU job runs at a time.  Jobs are detached from the agent's shell, so the agent may end its turn while
they run; the stop hook keeps the loop alive and ``wait`` re-attaches to progress.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .state import RUNS, now_iso

JOBS_DIR = RUNS / "jobs"
GPU_LOCK = JOBS_DIR / "GPU.lock"
HEARTBEAT_STALE_S = 180        # no heartbeat for this long -> the runner is gone
STARTING_GRACE_S = 90          # cmd.json written but no heartbeat yet (slow Windows process creation)
RUNNER_GRACE_S = 30            # the runner opens log.txt first thing; no log by then means it never ran
PID_GRACE_S = 45               # only trust pid-liveness once the heartbeat is this old
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ACTIVE_STATES = ("running", "starting")


def job_dir(name: str) -> Path:
    if not _NAME_RE.match(str(name or "")) or name in (".", ".."):
        raise ValueError("job name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63} (no separators, no '..')")
    return JOBS_DIR / name


def _read_json(p: Path) -> Optional[Dict]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001  (missing, truncated mid-write, or bad encoding)
        return None


def _write_json_atomic(p: Path, obj: Dict) -> None:
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, p)
            return
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    try:
        tmp.unlink()
    except OSError:
        pass


def _pid_alive(pid: Optional[int]) -> Optional[bool]:
    """True/False if we can tell, None if we cannot.  Used only as a *secondary* staleness signal, so a
    wrong answer costs at most one re-dispatch.  The Windows branch is NOT VERIFIED from this Linux
    sandbox; it fails closed (returns None) if ctypes is unavailable or OpenProcess errors."""
    if not pid or pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
            k = ctypes.windll.kernel32
            h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False  # ERROR_INVALID_PARAMETER -> no such process
            code = ctypes.c_ulong()
            got = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return bool(got) and code.value == STILL_ACTIVE
        except Exception:  # noqa: BLE001
            return None
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None


def _holder_alive(name: str) -> bool:
    """True when the job's child process still exists (cross-platform, fails safe to True)."""
    try:
        info = _read_json(job_dir(name) / "child.json") or {}
        pid = int(info.get("pid") or 0)
    except Exception:  # noqa: BLE001
        return True
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except Exception:  # noqa: BLE001
        return True


def gpu_busy() -> Optional[str]:
    """Name of the job holding the GPU lock, or None (stale locks are cleared).

    ``starting`` counts as busy: a GPU job that has written cmd.json and the lock but has not yet
    produced its first heartbeat must not have its lock stolen, or two teacher-scoring jobs would run
    on the same 5090 at once."""
    if not GPU_LOCK.exists():
        return None
    try:
        holder = GPU_LOCK.read_text(encoding="utf-8").strip()
    # A stale heartbeat (frozen machine, full disk) must NOT free the GPU while the process is really
    # gone is unproven: two teacher jobs on one card means OOM and a corrupted cache.
    except OSError:
        return None
    if not holder:
        _unlink(GPU_LOCK)
        return None
    try:
        st = status(holder)
    except ValueError:
        st = {"state": "unknown"}
    if st["state"] in ACTIVE_STATES:
        return holder
    _unlink(GPU_LOCK)
    return None


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def normalize_cmd(cmd: List[str]) -> List[str]:
    """`python ...` / `python3 ...` always means the interpreter that runs graphctl
    (on Windows the bare name may resolve to the Store stub or to another environment)."""
    cmd = [str(c) for c in cmd]
    if cmd and Path(cmd[0]).name.lower() in ("python", "python3", "python.exe", "python3.exe", "py", "py.exe"):
        cmd[0] = sys.executable
    return cmd


def child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for a detached child.

    * UTF-8 is forced so a progress line with a non-ASCII character cannot kill a 30-hour job on a
      Windows console whose active code page is cp936/cp1252.
    * PYTHONPATH carries the directory that holds the graphctl package, so
      ``python -m graphctl.jobrunner`` resolves even if the detached process does not inherit the
      repository as its working directory."""
    pkg_parent = str(Path(__file__).resolve().parents[1])
    old = os.environ.get("PYTHONPATH", "")
    env = {**os.environ,
           "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": pkg_parent + (os.pathsep + old if old else "")}
    env.update(extra or {})
    return env


def start(name: str, cmd: List[str], gpu: bool = False, cwd: Optional[Path] = None,
          env: Optional[Dict[str, str]] = None) -> Dict:
    cmd = normalize_cmd(cmd)
    if not cmd:
        raise RuntimeError("empty command")
    d = job_dir(name)
    if d.exists() and status(name)["state"] in ACTIVE_STATES:
        raise RuntimeError(f"job {name} is already {status(name)['state']}; use `jobs status {name}` or pick another name")
    if gpu:
        holder = gpu_busy()
        if holder and holder != name:
            raise RuntimeError(f"GPU is held by job {holder}; wait for it (python -m graphctl jobs wait {holder})")
    d.mkdir(parents=True, exist_ok=True)
    for f in ("status.json", "heartbeat", "log.txt", "child.json", "kill.request"):
        _unlink(d / f)
    meta = {"name": name, "cmd": cmd, "gpu": gpu, "cwd": str(cwd or Path.cwd()), "started_at": now_iso()}
    _write_json_atomic(d / "cmd.json", meta)
    if gpu:
        GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        GPU_LOCK.write_text(name, encoding="utf-8")
    runner = [sys.executable, "-m", "graphctl.jobrunner", str(d)]
    kwargs: Dict = {"cwd": str(cwd or Path.cwd()), "env": child_env(env), "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: survives the agent's terminal being closed or
        # reaped by Cursor's idle-session cleanup.  NOT VERIFIED from this Linux sandbox.
        kwargs["creationflags"] = 0x00000008 | 0x00000200
    else:
        kwargs["start_new_session"] = True
    try:
        p = subprocess.Popen(runner, **kwargs)
    except Exception as e:  # noqa: BLE001
        if gpu:
            _unlink(GPU_LOCK)
        raise RuntimeError(f"could not spawn the detached runner: {e}") from e
    (d / "runner.pid").write_text(str(p.pid), encoding="utf-8")
    return {"name": name, "runner_pid": p.pid, "dir": str(d), "gpu": gpu}


def status(name: str) -> Dict:
    d = job_dir(name)
    if not d.exists():
        return {"name": name, "state": "unknown"}
    meta = _read_json(d / "cmd.json") or {}
    out = {"name": name, "gpu": meta.get("gpu", False), "started_at": meta.get("started_at"), "cmd": meta.get("cmd")}
    st = _read_json(d / "status.json")
    if st is not None and "exit_code" in st:
        out.update(state="finished" if st.get("exit_code") == 0 else "failed",
                   exit_code=st.get("exit_code"), finished_at=st.get("finished_at"))
        if st.get("killed"):
            out["killed"] = True
        return out
    runner_pid = None
    try:
        runner_pid = int((d / "runner.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    hb = d / "heartbeat"
    try:
        age = time.time() - hb.stat().st_mtime if hb.exists() else None
    except OSError:
        age = None
    if age is None:
        try:
            age0 = time.time() - (d / "cmd.json").stat().st_mtime
        except OSError:
            age0 = 1e9
        # No heartbeat yet.  The runner opens log.txt as its very first action, so a missing log after
        # RUNNER_GRACE_S means the detached process never ran at all (bad interpreter, package not
        # importable, blocked by security software) -- that must not look like "starting" for minutes.
        grace = STARTING_GRACE_S if (d / "log.txt").exists() else RUNNER_GRACE_S
        if age0 < grace and _pid_alive(runner_pid) is not False:
            out.update(state="starting", age_s=round(age0))
        else:
            out.update(state="stale", age_s=round(age0),
                       reason="the detached runner never produced a heartbeat"
                              if (d / "log.txt").exists() else "the detached runner never started")
        return out
    alive = _pid_alive(runner_pid)
    if age < HEARTBEAT_STALE_S and not (age > PID_GRACE_S and alive is False):
        out.update(state="running", heartbeat_age_s=round(age))
    else:
        out.update(state="stale", heartbeat_age_s=round(age),
                   reason="no heartbeat" if alive is not False else "the detached runner is gone")
    return out


def list_jobs() -> List[Dict]:
    if not JOBS_DIR.exists():
        return []
    out = []
    for p in sorted(JOBS_DIR.iterdir()):
        if not p.is_dir():
            continue
        try:
            out.append(status(p.name))
        except ValueError:
            continue
    return out


def tail(name: str, lines: int = 40, max_bytes: int = 512_000) -> str:
    """Last ``lines`` lines of the job log, reading only the tail of the file.

    A 30-hour scoring job writes a progress line per batch; the log can reach gigabytes and reading it
    whole (the previous behaviour) would stall or OOM every single `graphctl next`."""
    p = job_dir(name) / "log.txt"
    if not p.exists():
        return "(no log yet)"
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # drop the partial first line
            data = f.read()
    except OSError as e:
        return f"(log unreadable: {e})"
    text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def wait(name: str, timeout: float = 600.0, poll: float = 15.0) -> Dict:
    t0 = time.time()
    poll = max(0.05, float(poll))
    while True:
        st = status(name)
        if st["state"] in ("finished", "failed", "stale", "unknown"):
            return st
        if time.time() - t0 >= timeout:
            st["timed_out_after_s"] = round(time.time() - t0)
            return st
        time.sleep(min(poll, max(0.05, timeout - (time.time() - t0))))


def kill(name: str) -> Dict:
    """Stop a detached job: ask the runner to stop, kill the child tree, then kill the runner.

    Order matters.  Killing only the child (the previous behaviour) left the runner alive, and it then
    overwrote status.json with the child's own exit code; killing before the child exists left the job
    running under a status.json that claimed it was dead."""
    d = job_dir(name)
    if not d.exists():
        return {"name": name, "killed": False, "error": "no such job"}
    errors = []
    try:
        (d / "kill.request").write_text(now_iso(), encoding="utf-8")  # honoured by jobrunner before spawn
    except OSError as e:
        errors.append(str(e))
    child = (_read_json(d / "child.json") or {}).get("pid")
    try:
        runner = int((d / "runner.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        runner = None
    for pid, tree in ((child, True), (runner, False)):
        if not pid:
            continue
        try:
            if os.name == "nt":
                # /T kills the whole tree, /F forces it.  NOT VERIFIED from this Linux sandbox.
                subprocess.run(["taskkill", "/PID", str(pid)] + (["/T"] if tree else []) + ["/F"],
                               capture_output=True, timeout=30)
            else:
                try:
                    os.killpg(os.getpgid(pid), 15)
                except (ProcessLookupError, PermissionError):
                    os.kill(pid, 15)
        except Exception as e:  # noqa: BLE001
            errors.append(f"pid {pid}: {e}")
    time.sleep(0.4)
    if _read_json(d / "status.json") is None:
        _write_json_atomic(d / "status.json", {"exit_code": -9, "finished_at": now_iso(), "killed": True})
    meta = _read_json(d / "cmd.json") or {}
    try:
        if meta.get("gpu") and GPU_LOCK.exists() and GPU_LOCK.read_text(encoding="utf-8").strip() == name:
            _unlink(GPU_LOCK)
    except OSError:
        pass
    return {"name": name, "killed": True, "errors": errors} if errors else {"name": name, "killed": True}


def format_cmd(cmd: List[str]) -> str:
    """Human-readable command for the ledger, quoted the way the *local* shell would need it."""
    parts = []
    for c in (str(x) for x in cmd):
        if not c or any(ch in c for ch in ' \t"\''):
            parts.append('"' + c.replace('"', r'\"') + '"')
        else:
            parts.append(c)
    return " ".join(parts)
