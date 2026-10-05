"""Plan B: drive the graph loop from a terminal with the Cursor CLI (no IDE window).

    ./.venv/Scripts/python.exe scripts/run_headless.py                 # loop until GRAPH DONE
    ...                                     --dry-run                  # print what it would send
    ...                                     --agent-cmd "claude -p"    # any headless coding agent

Each iteration sends one short prompt to a *fresh* agent process (the graph state lives in files, so a
clean context is enough), waits for it, then re-reads the state.  Use this when the IDE session is not
available; do not run it at the same time as an IDE agent -- one driver at a time (a lock file in
runs/headless/ enforces it).  Requires the Cursor CLI and `agent login`:

    Windows   irm 'https://cursor.com/install?win32=true' | iex
    Linux/mac curl https://cursor.com/install -fsS | bash
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "headless"

PROMPT = """You are the main agent of the RSC-Rank graph loop (read AGENTS.md first; it is short).
Run `{py} -m graphctl ladder --apply` then `{py} -m graphctl next`, and execute the dispatched node(s)
end to end: split the work into parallel subagents, verify with `{py} -m graphctl verify <node>`, get the
fresh-context review where the node requires one, then `{py} -m graphctl done <node> --note "..."`.
Long work goes into detached jobs (`{py} -m graphctl jobs start NAME --node <node> [--gpu] -- CMD`) and you
end the turn instead of waiting. Never ask the human anything: graph/decisions.md pre-authorises every
routine decision, and the only human actions are the two OpenReview steps. Stop this turn after at most
two nodes are finished (the driver will call you again immediately)."""


def detect_agent() -> list:
    for exe in (os.environ.get("RSC_AGENT_CMD"), "agent", "cursor-agent"):
        if exe and shutil.which(exe.split()[0]):
            return exe.split()
    return []


def graphctl(*args) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "graphctl", *args], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)


def _plan(out: str) -> dict:
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-cmd", default=None, help="headless agent command (default: cursor `agent`)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--turn-timeout", type=int, default=3600)
    ap.add_argument("--poll", type=int, default=120, help="seconds to sleep while only background jobs run")
    ap.add_argument("--min-gap", type=float, default=5.0, help="minimum seconds between turns")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "driver.lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < 3 * args.turn_timeout and not args.dry_run:
        print(f"another driver seems active ({lock}); delete it if that is wrong")
        return 4
    base = (args.agent_cmd.split() if args.agent_cmd else detect_agent())
    if not base and not args.dry_run:
        print("No headless agent CLI found. Install the Cursor CLI and run `agent login`, or pass --agent-cmd.\n"
              "  Windows: irm 'https://cursor.com/install?win32=true' | iex")
        return 4
    py = "./.venv/Scripts/python.exe" if (ROOT / ".venv" / "Scripts" / "python.exe").exists() else sys.executable
    prompt = PROMPT.format(py=py)
    if args.dry_run:
        print(" ".join(base or ["<agent>"]), "-p --force --trust --output-format text\n---\n" + prompt)
        return 0
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        for it in range(1, args.max_iters + 1):
            t0 = time.time()
            lock.write_text(str(os.getpid()), encoding="utf-8")
            graphctl("ladder", "--apply")
            nxt = graphctl("next", "--json")
            plan = _plan(nxt.stdout)
            if plan.get("status") == "DONE" or "GRAPH DONE" in nxt.stdout:
                print("GRAPH DONE -- see graph/HANDOFF.md")
                return 0
            if nxt.returncode == 3:                       # graph/HALT present: a human paused the run
                print(f"[{it}] halted (graph/HALT); waiting {args.poll}s for `graphctl resume`")
                time.sleep(args.poll)
                continue
            if not plan.get("plan"):
                if plan.get("waiting"):
                    print(f"[{it}] only background jobs running; sleeping {args.poll}s")
                    time.sleep(args.poll)
                    continue
                print(f"[{it}] nothing ready (see graph/blockers.md):\n{nxt.stdout[-1500:]}")
                return 3
            cmd = base + ["-p", "--force", "--trust", "--output-format", "text"] \
                + (["--model", args.model] if args.model else []) + [prompt]
            log = OUT / f"iter_{it:04d}.log"
            print(f"[{it}] dispatching {plan['plan']} -> {log}")
            with open(log, "w", encoding="utf-8", errors="replace") as f:
                try:
                    subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=f,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   timeout=args.turn_timeout, env=env)
                except subprocess.TimeoutExpired:
                    f.write("\n[driver] turn timed out; continuing\n")
                except OSError as e:
                    f.write(f"\n[driver] could not launch the agent: {e}\n")
                    print(f"[{it}] could not launch the agent: {e}")
                    return 4
            gap = args.min_gap - (time.time() - t0)       # never hot-loop if the agent returns at once
            if gap > 0:
                time.sleep(gap)
        print("max iterations reached")
        return 5
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
