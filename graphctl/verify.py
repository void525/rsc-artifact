"""Machine verification of a node (the 'checker' of the diamond).

Step types in graph.yaml::

    - cmd: python -m pytest -q tests          # exit code 0 required
      timeout: 1800
    - file_exists: runs/env/report.json
    - json_check: {path: runs/env/report.json, expr: "d['ok'] is True"}
    - review: N02_theory                       # fresh-context verifier verdict file must say pass

Results are written to runs/verify/<node>.json.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List

from .state import ROOT, RUNS, now_iso

VERIFY_DIR = RUNS / "verify"


_PY_PREFIX = re.compile(r"^(python3?|py)(\.exe)?(?=\s)")

# A safe namespace for the small boolean assertions in graph.yaml's json_check steps.
EVAL_NS = {"len": len, "min": min, "max": max, "abs": abs, "all": all, "any": any, "sum": sum,
           "round": round, "sorted": sorted, "float": float, "int": int, "str": str, "isinstance": isinstance}


def _bind_interpreter(cmd: str) -> str:
    """Bind a leading `python `/`python3 `/`py ` to the interpreter running graphctl.

    On Windows a bare ``python`` may be the Microsoft Store stub, the conda *base* environment, or
    nothing at all; every verify step must run in the project's .venv."""
    stripped = cmd.strip()
    m = _PY_PREFIX.match(stripped)
    if not m:
        return cmd
    return f'"{sys.executable}"' + stripped[m.end():]


def _kill_tree(p: "subprocess.Popen") -> None:
    """Kill a timed-out verify command *and its children*.

    ``subprocess.run(..., shell=True, timeout=...)`` only kills the shell, leaving e.g. a wedged pytest
    running for the rest of the five weeks.  The Windows branch is NOT VERIFIED from this Linux sandbox."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, timeout=30)
        else:
            os.killpg(os.getpgid(p.pid), 9)
    except Exception:  # noqa: BLE001
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass


def _run_cmd(cmd: str, timeout: int) -> Dict[str, Any]:
    """Run one verify command from the repository root.

    ``shell=True`` is deliberate: graph.yaml steps are written as shell one-liners.  On Windows this is
    cmd.exe (COMSPEC), not PowerShell, so the steps must stay free of PowerShell-only syntax -- they
    are plain ``<interpreter> <script> <flags>`` lines, which both shells accept."""
    t0 = time.time()
    cmd = _bind_interpreter(cmd)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    kwargs: Dict[str, Any] = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True
    try:
        p = subprocess.Popen(cmd, shell=True, cwd=str(ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                             encoding="utf-8", errors="replace", env=env, **kwargs)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "exit_code": None, "seconds": 0.0, "tail": f"could not start: {e}"}
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(p)
        try:
            out, _ = p.communicate(timeout=20)
        except Exception:  # noqa: BLE001
            out = ""
        return {"ok": False, "exit_code": None, "seconds": round(time.time() - t0, 1),
                "tail": f"TIMEOUT after {timeout}s (process tree killed)\n" + (out or "")[-2000:]}
    return {"ok": p.returncode == 0, "exit_code": p.returncode,
            "seconds": round(time.time() - t0, 1), "tail": (out or "")[-3000:]}


def review_verdict(node_id: str) -> Dict[str, Any]:
    p = VERIFY_DIR / f"{node_id}.review.json"
    if not p.exists():
        return {"ok": False, "reason": f"missing {p.relative_to(ROOT)} (the verifier subagent must write it)"}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"unreadable review file: {e}"}
    ok = str(d.get("verdict", "")).lower() == "pass"
    return {"ok": ok, "verdict": d.get("verdict"), "summary": d.get("summary"), "issues": d.get("issues", []), "reviewer": d.get("reviewer")}


def run_steps(node: Dict[str, Any]) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = node.get("verify", [])
    results = []
    all_ok = True
    for step in steps:
        if "cmd" in step:
            r = {"type": "cmd", "cmd": step["cmd"], **_run_cmd(step["cmd"], int(step.get("timeout", 1800)))}
        elif "file_exists" in step:
            p = ROOT / step["file_exists"]
            r = {"type": "file_exists", "path": step["file_exists"], "ok": p.exists() and (p.stat().st_size > 0 if p.is_file() else True)}
        elif "json_check" in step:
            spec = step["json_check"]
            p = ROOT / spec["path"]
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                ok = bool(eval(spec["expr"], {"__builtins__": {}}, {"d": d, **EVAL_NS}))
                r = {"type": "json_check", "path": spec["path"], "expr": spec["expr"], "ok": ok}
            except Exception as e:  # noqa: BLE001
                r = {"type": "json_check", "path": spec["path"], "expr": spec["expr"], "ok": False, "error": str(e)}
        elif "review" in step:
            r = {"type": "review", **review_verdict(step["review"])}
        else:
            r = {"type": "unknown", "ok": False, "step": step}
        results.append(r)
        all_ok = all_ok and bool(r.get("ok"))
    report = {"node": node["id"], "ok": all_ok, "at": now_iso(), "steps": results}
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFY_DIR / f"{node['id']}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def summarize(report: Dict[str, Any]) -> str:
    lines = [f"VERIFY {report['node']}: {'PASS' if report['ok'] else 'FAIL'} ({report['at']})"]
    for r in report["steps"]:
        flag = "ok " if r.get("ok") else "FAIL"
        if r["type"] == "cmd":
            lines.append(f"  [{flag}] cmd: {r['cmd']}  (exit={r.get('exit_code')}, {r.get('seconds')}s)")
            if not r.get("ok"):
                lines.append("        --- output tail ---")
                lines.extend("        " + l for l in (r.get("tail") or "").splitlines()[-25:])
        elif r["type"] == "file_exists":
            lines.append(f"  [{flag}] file_exists: {r['path']}")
        elif r["type"] == "json_check":
            lines.append(f"  [{flag}] json_check: {r['path']} :: {r['expr']}" + (f"  ({r['error']})" if r.get("error") else ""))
        elif r["type"] == "review":
            lines.append(f"  [{flag}] fresh-context review: verdict={r.get('verdict')} {r.get('reason', '')} {r.get('summary', '')}")
            for iss in r.get("issues", []) or []:
                lines.append(f"        - {iss}")
        else:
            lines.append(f"  [{flag}] {r}")
    return "\n".join(lines)
