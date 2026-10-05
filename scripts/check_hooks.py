"""Prove that the Cursor hooks really run on THIS machine (node N00) -> runs/env/hooks_check.json.

The `stop` hook is the whole autonomy mechanism: if Cursor cannot execute the command string in
`.cursor/hooks.json`, every hook fails silently (Cursor passes failing hooks) and the loop stops after
the first turn with nobody watching.  The command must work in the shell Cursor uses -- PowerShell on
Windows (pwsh if present), /bin/sh elsewhere -- so this script runs each hook exactly that way with a
sample payload and checks that valid JSON comes back.

    python scripts/check_hooks.py           # report (exit 1 if any hook is broken)
    python scripts/check_hooks.py --fix     # rewrite the commands with the absolute interpreter path and re-test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".cursor" / "hooks.json"
PAYLOAD = {
    "sessionStart": {"session_id": "check", "is_background_agent": False, "composer_mode": "agent"},
    "beforeShellExecution": {"command": "echo hello", "cwd": str(ROOT), "sandbox": False},
    "stop": {"status": "completed", "loop_count": 0},
    "subagentStop": {"status": "completed", "subagent_type": "verifier"},
}


def shells():
    if os.name == "nt":
        for exe in ("pwsh", "powershell"):
            if shutil.which(exe):
                yield exe, [exe, "-NoProfile", "-Command"]
        yield "cmd", [os.environ.get("COMSPEC", "cmd.exe"), "/c"]
    else:
        yield "sh", ["/bin/sh", "-c"]


def run_hook(command: str, payload: dict) -> dict:
    out = {"command": command, "attempts": []}
    for name, prefix in shells():
        try:
            p = subprocess.run(prefix + [command], input=json.dumps(payload), cwd=str(ROOT), capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace")
            ok_json = False
            try:
                json.loads((p.stdout or "").strip().lstrip("﻿") or "{}")
                ok_json = True
            except Exception:  # noqa: BLE001
                ok_json = False
            attempt = {"shell": name, "exit": p.returncode, "stdout": (p.stdout or "")[:400], "stderr": (p.stderr or "")[-300:], "json_ok": ok_json}
            out["attempts"].append(attempt)
            if p.returncode == 0 and ok_json:
                out["ok"], out["shell"] = True, name
                return out
        except Exception as e:  # noqa: BLE001
            out["attempts"].append({"shell": name, "error": str(e)[:200]})
    out["ok"] = False
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="on failure, rewrite the commands with the absolute interpreter path")
    args = ap.parse_args()
    if not HOOKS.exists():
        print("no .cursor/hooks.json -- the autonomous loop will not resume itself")
        return 1
    cfg = json.loads(HOOKS.read_text(encoding="utf-8"))
    report, changed = {}, False
    for event, entries in (cfg.get("hooks") or {}).items():
        for i, entry in enumerate(entries):
            cmd = entry.get("command", "")
            res = run_hook(cmd, PAYLOAD.get(event, {}))
            if not res["ok"] and args.fix:
                # same script, but with this interpreter's absolute path (quoted for both shells)
                tail = cmd.split("-u", 1)[1].strip() if "-u" in cmd else cmd.split(" ", 1)[1].strip()
                fixed = f'"{sys.executable}" -u {tail}'
                res_fixed = run_hook(fixed, PAYLOAD.get(event, {}))
                if res_fixed["ok"]:
                    entry["command"], res, changed = fixed, res_fixed, True
                    res["fixed_to_absolute_path"] = True
            report[f"{event}[{i}]"] = res
    if changed:
        HOOKS.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"rewrote {HOOKS} with absolute interpreter paths")
    ok = all(r.get("ok") for r in report.values())
    out = ROOT / "runs" / "env" / "hooks_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ok": ok, "platform": os.name, "hooks": report}, indent=1), encoding="utf-8")
    for k, r in report.items():
        print(f"[{'ok ' if r.get('ok') else 'FAIL'}] {k}: {r['command']}" + (f"  (shell: {r.get('shell')})" if r.get("ok") else ""))
        if not r.get("ok"):
            print("      " + json.dumps(r["attempts"])[:600])
    if not ok:
        print("\nThe stop hook is what makes the loop continue without you. Run `python scripts/check_hooks.py --fix`;\n"
              "if that still fails, put the absolute interpreter path into .cursor/hooks.json by hand and re-run.")
    print(f"report -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
