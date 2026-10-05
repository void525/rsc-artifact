"""Command-line interface: {PY} -m graphctl <command> [...]

Exit codes (the stop hook and scripts/run_headless.py rely on them):
  0  success / there is work to do (dispatched nodes or nodes waiting on a background job)
  1  refused or failed (verify failed, `done` refused, job failed, bad usage)
  2  nothing to do right now and nothing running (the loop is stalled -- see graph/blockers.md)
  3  the loop is halted (graph/HALT present)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import jobs, render, verify
from .graph import conflicts, days_until, load_graph, node_by_id, ready_nodes, topo_order
from .state import (BLOCKERS_PATH, GRAPH_DIR, HALT_PATH, LEDGER_PATH, ROOT, append_ledger, clear_needs_you,
                    load_state, node_state, now_iso, progress_hash, py_cmd, state_txn, write_blocker)

try:  # Windows consoles default to a legacy code page
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

EXIT_OK, EXIT_FAIL, EXIT_IDLE, EXIT_HALTED = 0, 1, 2, 3
ACTIVE = ("in_progress", "verified")

PY = py_cmd()


def _current_node(state: Dict[str, Any]) -> Optional[str]:
    for nid, ns in sorted(state.get("nodes", {}).items()):
        if ns.get("status") in ACTIVE:
            return nid
    return None


def _all_current(state: Dict[str, Any]) -> List[str]:
    return [nid for nid, ns in sorted(state.get("nodes", {}).items()) if ns.get("status") in ACTIVE]


def _job_states(names) -> List[Dict[str, Any]]:
    out = []
    for n in names or []:
        try:
            out.append(jobs.status(n))
        except Exception:  # noqa: BLE001
            out.append({"name": n, "state": "unknown"})
    return out


def cmd_status(args) -> int:
    g, st = load_graph(), load_state()
    if getattr(args, "zh", False):
        out = GRAPH_DIR / "STATUS_zh.md"
        out.write_text(render.status_zh(g, st), encoding="utf-8")
        print(f"wrote {out}")
    print(f"graph status: {st.get('status')}   (HALT file: {'present' if HALT_PATH.exists() else 'absent'})")
    print(render.deadline_line(g))
    print(render.running_jobs_line())
    print("-" * 78)
    for nid in topo_order(g):
        n = node_by_id(g, nid)
        ns = st["nodes"].get(nid, {})
        s = ns.get("status", "pending")
        lf = n.get("latest_finish")
        late = ""
        if lf and s not in ("done", "skipped"):
            d = days_until(lf)
            late = f"  [latest finish {lf}: {d:+.1f} d]" if d < 3 else ""
        deps = ",".join(n.get("depends_on", [])) or "-"
        js = ",".join(j["name"] + "(" + j["state"] + ")" for j in _job_states(ns.get("jobs"))
                      if j.get("state") in jobs.ACTIVE_STATES)
        print(f"{nid:<22} {s:<12} it={ns.get('iterations', 0)}  deps={deps}{'  jobs=' + js if js else ''}{late}")
    if args.verbose:
        print("-" * 78)
        print(json.dumps(st.get("followups"), indent=1))
    return EXIT_OK


def _stuck_hours(g) -> float:
    return float(g.get("loop", {}).get("stuck_node_hours", 12))


def _hours_since(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        import datetime as _dt
        return (_dt.datetime.now() - _dt.datetime.fromisoformat(str(iso))).total_seconds() / 3600.0
    except Exception:  # noqa: BLE001
        return None


def _reap_stuck(g, st, max_it: int) -> List[str]:
    """A node left `in_progress` with no running job and no movement for `loop.stuck_node_hours` is
    recycled: back to pending (its iteration was already counted) or blocked once the retry budget is
    spent.  Without this a single unfinishable node keeps a `max_parallel` slot forever and the
    iteration cap -- which only `fail` applies -- is never reached, so the loop can never terminate."""
    limit, recycled = _stuck_hours(g), []
    for nid, ns in sorted(st.get("nodes", {}).items()):
        if ns.get("status") not in ACTIVE:
            continue
        if any(j.get("state") in jobs.ACTIVE_STATES for j in _job_states(ns.get("jobs"))):
            continue
        h = _hours_since(ns.get("touched_at") or ns.get("started_at"))
        if h is None or h < limit:
            continue
        ns["notes"].append(f"STUCK: no progress for {h:.1f} h while in_progress; recycled by `next`")
        if ns.get("iterations", 0) >= max_it:
            ns["status"] = "blocked"
            write_blocker(nid, f"Node sat in_progress for {h:.1f} h with no background job and no state change, "
                               f"and its {max_it} iterations are spent.",
                          f"Read runs/verify/{nid}.json and graph/ledger.md, fix the cause, then "
                          f"`{PY} -m graphctl unblock {nid}`.")
            append_ledger(f"BLOCK {nid}: stuck in_progress for {h:.1f} h (iterations exhausted)")
        else:
            ns["status"] = "pending"
            append_ledger(f"NOTE {nid} recycled: stuck in_progress for {h:.1f} h")
        recycled.append(nid)
    return recycled


def cmd_next(args) -> int:
    """Print the parallel work plan: every node that can be worked on right now.

    Nodes whose background job is still running are listed separately as WAITING -- the agent does
    CPU nodes meanwhile instead of blocking on the GPU."""
    g = load_graph()
    if HALT_PATH.exists():
        print(f"HALT file present (graph/HALT). Run `{PY} -m graphctl resume` to continue.")
        return EXIT_HALTED
    max_par = int(getattr(args, "max", None) or g.get("loop", {}).get("max_parallel_nodes", 3))
    max_it = int(g.get("loop", {}).get("max_iterations_per_node", 6))
    payload: Dict[str, Any] = {}
    with state_txn() as st:
        if st.get("status") == "DONE":
            print("GRAPH DONE. Read graph/HANDOFF.md for the remaining human action (OpenReview upload).")
            if getattr(args, "json", False):
                print(json.dumps({"plan": [], "waiting": [], "status": "DONE", "max_parallel": max_par}))
            return EXIT_OK
        recycled = _reap_stuck(g, st, max_it)
        holder = jobs.gpu_busy()
        waiting, active = [], []
        for nid, ns in sorted(st["nodes"].items()):
            if ns.get("status") not in ACTIVE:
                continue
            try:
                node = node_by_id(g, nid)
            except KeyError:
                continue  # a node that was removed from graph.yaml: ignore the leftover state entry
            js = [j for j in _job_states(ns.get("jobs")) if j.get("state") in jobs.ACTIVE_STATES]
            (waiting if js else active).append({"id": nid, "title": node.get("title", ""), "jobs": js, "node": node})
        plan = [w["node"] for w in active]
        if args.node:
            forced = node_by_id(g, args.node)
            fs = st["nodes"].get(args.node, {}).get("status", "pending")
            if fs in ("done", "skipped"):
                print(f"REFUSED: {args.node} is already {fs}. Use `unblock`/edit the state deliberately.")
                return EXIT_FAIL
            if fs == "blocked":
                print(f"REFUSED: {args.node} is blocked (see graph/blockers.md). "
                      f"Run `{PY} -m graphctl unblock {args.node}` first.")
                return EXIT_FAIL
            if args.node in [w["id"] for w in waiting]:
                print(f"note: {args.node} has a running job; it is listed as WAITING, not dispatched.")
            elif forced["id"] not in [n["id"] for n in plan]:
                plan.append(forced)
        gpu_in_plan = bool(holder) or any("gpu" in p.get("resources", []) for p in plan) \
            or any("gpu" in w["node"].get("resources", []) for w in waiting)
        ready = [n for n in ready_nodes(g, st, gpu_busy=bool(holder))
                 if st["nodes"].get(n["id"], {}).get("status", "pending") == "pending"]
        for n in ready:
            if len(plan) >= max_par:
                break
            if "gpu" in n.get("resources", []) and gpu_in_plan:
                continue  # one GPU node at a time
            if any(conflicts(n, p) for p in plan):
                continue  # two nodes must not write the same area in parallel
            plan.append(n)
            gpu_in_plan = gpu_in_plan or "gpu" in n.get("resources", [])
        iteration_of = {}
        for n in plan:
            ns = node_state(st, n["id"])
            if ns["status"] == "pending":
                ns["status"] = "in_progress"
                ns["iterations"] = ns.get("iterations", 0) + 1
                ns["started_at"] = ns.get("started_at") or now_iso()
                ns["dispatches"] = 0
                ns["touched_at"] = now_iso()      # only a real state change refreshes the stuck clock
                append_ledger(f"START {n['id']} (iteration {ns['iterations']})")
            else:
                # Re-dispatch of a node that is already in progress: count it.  A node that keeps being
                # handed out without ever reaching `done` (e.g. a review verdict that never appears) is
                # the classic silent infinite loop -- after `loop.max_redispatches` it is failed, which
                # walks the normal retry/block path and eventually reaches the human via blockers.md.
                ns["dispatches"] = int(ns.get("dispatches", 0)) + 1
                if ns["dispatches"] > int(g.get("loop", {}).get("max_redispatches", 6)):
                    ns["notes"].append(f"NO-PROGRESS: dispatched {ns['dispatches']} times without completing")
                    ns["status"] = "pending"
                    ns["dispatches"] = 0
                    ns["iterations"] = ns.get("iterations", 0) + 1
                    ns["touched_at"] = now_iso()
                    append_ledger(f"FAIL {n['id']} (iteration {ns['iterations']}): dispatched repeatedly without completing")
                    if ns["iterations"] >= int(g.get("loop", {}).get("max_iterations_per_node", 6)):
                        ns["status"] = "blocked"
                        write_blocker(n["id"], "The node was dispatched again and again without ever reaching `done` "
                                               "(most often: a required fresh-context review verdict never arrived).",
                                      f"Read runs/verify/{n['id']}.json and graph/ledger.md; fix the cause, then `{PY} -m graphctl unblock {n['id']}`.")
                        append_ledger(f"BLOCK {n['id']}: repeated dispatch without completion")
            iteration_of[n["id"]] = ns.get("iterations", 1)
        if not plan and not waiting:
            # A blocked *optional* node must never keep the graph from finishing: everything else can
            # be done and the loop would otherwise STALL one step before GRAPH DONE.  Blocked required
            # nodes still stall (correctly), because the paper cannot be finished without them.
            for n in g["nodes"]:
                ns = st["nodes"].get(n["id"], {})
                if n.get("optional") and ns.get("status") == "blocked":
                    ns["status"] = "skipped"
                    ns.setdefault("notes", []).append("auto-skipped: optional node was blocked and nothing else is left")
                    append_ledger(f"SKIP {n['id']}: optional node was blocked; dropped so the graph can finish")
            remaining = [n["id"] for n in g["nodes"]
                         if st["nodes"].get(n["id"], {}).get("status", "pending") not in ("done", "skipped")]
            if not remaining:
                st["status"] = "DONE"
                append_ledger("GRAPH DONE")
                print("All nodes done. GRAPH DONE. See graph/HANDOFF.md.")
                if getattr(args, "json", False):
                    print(json.dumps({"plan": [], "waiting": [], "status": "DONE", "max_parallel": max_par}))
                return EXIT_OK
            blocked = [nid for nid, ns in sorted(st["nodes"].items()) if ns.get("status") == "blocked"]
            waspreviously = st.get("status")
            st["status"] = "STALLED"
            msg = ("No node is ready. Blocked: " + (", ".join(blocked) or "none")
                   + ". Remaining: " + ", ".join(remaining))
            print(msg)
            print(f"Unblock (`{PY} -m graphctl unblock <node>`) or read graph/blockers.md; "
                  f"then `{PY} -m graphctl resume`.")
            if waspreviously != "STALLED":
                # The stop hook stays silent once the graph is STALLED, so the human must be able to
                # find out why from graph/blockers.md alone.
                append_ledger("STALLED: " + msg)
                write_blocker("loop", msg,
                              f"Unblock a node (`{PY} -m graphctl unblock <node>`) or adjust graph/decisions.md, "
                              f"then `{PY} -m graphctl resume` and send `{PY} -m graphctl next` to the agent.")
            if getattr(args, "json", False):
                print(json.dumps({"plan": [], "waiting": [], "status": "STALLED", "blocked": blocked,
                                  "remaining": remaining, "max_parallel": max_par}))
            return EXIT_IDLE
        st["status"] = "RUNNING"
        text = render.dispatch(g, st, plan, waiting, iteration_of, recycled)
        payload = {"plan": [n["id"] for n in plan],
                   "waiting": [{"id": w["id"], "jobs": [j["name"] for j in w["jobs"]]} for w in waiting],
                   "recycled": recycled, "gpu_held_by": holder, "status": "RUNNING", "max_parallel": max_par}
    if getattr(args, "json", False):
        print(json.dumps(payload))
        return EXIT_OK
    print(text)
    return EXIT_OK


def cmd_brief(args) -> int:
    """Full briefing for one node (what a worker subagent should start from)."""
    g, st = load_graph(), load_state()
    n = node_by_id(g, args.node)
    ns = node_state(st, args.node)
    print(render.briefing(g, st, n, ns.get("iterations", 1), _job_states(ns.get("jobs"))))
    return EXIT_OK


def _touch(st, node_id: str) -> Dict[str, Any]:
    ns = node_state(st, node_id)
    ns["touched_at"] = now_iso()
    return ns


def cmd_verify(args) -> int:
    g = load_graph()
    n = node_by_id(g, args.node)
    rep = verify.run_steps(n)
    print(verify.summarize(rep))
    with state_txn() as st:
        ns = _touch(st, args.node)
        ns["verify"] = {"ok": rep["ok"], "at": rep["at"]}
        if rep["ok"] and ns["status"] == "in_progress":
            ns["status"] = "verified"
        elif not rep["ok"] and ns["status"] == "verified":
            ns["status"] = "in_progress"  # a previously verified node that regressed is not "verified"
    append_ledger(f"VERIFY {args.node}: {'PASS' if rep['ok'] else 'FAIL'}")
    return EXIT_OK if rep["ok"] else EXIT_FAIL


def cmd_done(args) -> int:
    g = load_graph()
    n = node_by_id(g, args.node)
    st0 = load_state()
    running = [j for j in _job_states(st0["nodes"].get(args.node, {}).get("jobs"))
               if j.get("state") in jobs.ACTIVE_STATES]
    if running and not getattr(args, "force", False):
        print(f"REFUSED: {args.node} still has a background job running: "
              + ", ".join(j["name"] for j in running))
        print(f"Wait for it (`{PY} -m graphctl jobs wait {running[0]['name']} --timeout 540`), kill it, "
              f"or pass --force if it is genuinely unrelated.")
        return EXIT_FAIL
    rep = verify.run_steps(n)
    if not rep["ok"]:
        print(verify.summarize(rep))
        print(f"\nREFUSED: {args.node} does not pass machine verification. Fix and re-run.")
        return EXIT_FAIL
    rv = None
    if n.get("requires_review"):
        rv = verify.review_verdict(args.node)
        if not rv["ok"]:
            print(f"REFUSED: fresh-context review missing or not 'pass': {rv}")
            print("Spawn the `verifier` subagent (clean context) with the review prompt from the node spec;")
            print(f"it must write runs/verify/{args.node}.review.json with verdict 'pass'")
            print(f"(or reply with that JSON, which you then store verbatim with "
                  f"`{PY} -m graphctl record-review {args.node} --file <path>` / `--stdin`).")
            return EXIT_FAIL
    with state_txn() as st:
        ns = _touch(st, args.node)
        if rv is not None:
            ns["review"] = {"verdict": rv.get("verdict"), "summary": rv.get("summary")}
        ns["status"] = "done"
        ns["finished_at"] = now_iso()
        ns["jobs"] = [j["name"] for j in _job_states(ns.get("jobs")) if j.get("state") in jobs.ACTIVE_STATES]
        if args.note:
            ns["notes"].append(args.note)
    append_ledger(f"DONE {args.node}: {args.note or ''}")
    print(f"DONE {args.node}. Next: {PY} -m graphctl next")
    return EXIT_OK


def cmd_fail(args) -> int:
    g = load_graph()
    node_by_id(g, args.node)  # fail loudly on a typo instead of creating a phantom node
    max_it = int(g.get("loop", {}).get("max_iterations_per_node", 6))
    with state_txn() as st:
        ns = _touch(st, args.node)
        ns["notes"].append(f"FAIL: {args.reason}")
        ns["verify"] = None
        ns["jobs"] = [j["name"] for j in _job_states(ns.get("jobs")) if j.get("state") in jobs.ACTIVE_STATES]
        iterations = ns.get("iterations", 0)
        append_ledger(f"FAIL {args.node} (iteration {iterations}): {args.reason}")
        if iterations >= max_it:
            ns["status"] = "blocked"
            write_blocker(args.node, f"Exceeded {max_it} iterations. Last reason: {args.reason}",
                          f"Read runs/verify and the ledger; adjust graph/decisions.md or configs; then "
                          f"`{PY} -m graphctl unblock {args.node}`.")
            print(f"{args.node} BLOCKED after {iterations} iterations. "
                  f"Other ready nodes will be offered by `next`.")
            return EXIT_OK
        ns["status"] = "pending"
    print(f"{args.node} reset to pending (iteration {iterations}/{max_it}). "
          f"`next` will re-offer it with the failure noted.")
    return EXIT_OK


def cmd_block(args) -> int:
    g = load_graph()
    node_by_id(g, args.node)
    with state_txn() as st:
        ns = _touch(st, args.node)
        ns["status"] = "blocked"
        ns["notes"].append(f"BLOCKED: {args.reason}")
    write_blocker(args.node, args.reason, args.human)
    append_ledger(f"BLOCK {args.node}: {args.reason}")
    print(f"{args.node} blocked; recorded in graph/blockers.md. `next` will offer other ready nodes.")
    return EXIT_OK


def cmd_unblock(args) -> int:
    clear_needs_you()
    g = load_graph()
    node_by_id(g, args.node)
    with state_txn() as st:
        ns = _touch(st, args.node)
        ns["status"] = "pending"
        ns["iterations"] = 0
        st["status"] = "RUNNING"
        st.setdefault("followups", {})["streak"] = 0
    append_ledger(f"UNBLOCK {args.node}")
    print(f"{args.node} unblocked. Send `{PY} -m graphctl next` to the agent.")
    return EXIT_OK


def cmd_skip(args) -> int:
    g = load_graph()
    n = node_by_id(g, args.node)
    if not n.get("optional"):
        print(f"REFUSED: {args.node} is not optional. Use the decision ladder "
              f"(docs/DECISION_LADDER.md) to reduce its scope instead.")
        return EXIT_FAIL
    with state_txn() as st:
        ns = _touch(st, args.node)
        ns["status"] = "skipped"
        ns["notes"].append(f"SKIPPED: {args.reason}")
    append_ledger(f"SKIP {args.node}: {args.reason}")
    print(f"{args.node} skipped.")
    return EXIT_OK


def cmd_followup(args) -> int:
    """Used by the Cursor stop hook: prints the follow-up message, or nothing to let the loop rest.

    This function must never raise and must never print a traceback: empty stdout makes the stop hook
    return {} and the five-week loop simply stops, with no human anywhere near it."""
    try:
        return _followup()
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        # Degraded but alive: the agent can always re-derive everything from `next`.
        print(f"[graph-loop] Continue autonomously. `{PY} -m graphctl followup` failed internally "
              f"(see the stderr above); run `{PY} -m graphctl next` and keep working. "
              f"If graph/state.json is corrupt, `{PY} -m graphctl status -v` will say so.")
        return EXIT_OK


def _followup() -> int:
    g = load_graph()
    if HALT_PATH.exists():
        return EXIT_OK
    maxs = int(g.get("loop", {}).get("max_followups_without_progress", 8))
    try:
        running = [j["name"] for j in jobs.list_jobs() if j["state"] in jobs.ACTIVE_STATES]
    except Exception:  # noqa: BLE001
        running = []
    # Short lock timeout on purpose: the Cursor `stop` hook times out at 30 s and a dropped
    # hook output silently ends the loop, so never wait 60 s for another graphctl here.
    with state_txn(timeout=8.0) as st:
        if st.get("status") in ("DONE", "HALTED"):
            return EXIT_OK
        fu = st.setdefault("followups", {"streak": 0, "last_progress_hash": ""})
        h = progress_hash(st)
        fu["streak"] = 0 if h != fu.get("last_progress_hash") else fu.get("streak", 0) + 1
        fu["last_progress_hash"] = h
        fu["total"] = fu.get("total", 0) + 1
        fu["last_at"] = now_iso()
        streak, status = fu["streak"], st.get("status")
        if streak > maxs and not running and st.get("status") != "STALLED":   # only on the transition
            st["status"] = "STALLED"
            status = "STALLED"
            reason = (f"{streak} consecutive follow-ups produced no progress (no node status change, no ledger "
                      f"entry, no running job). The loop paused itself.")
            stalled_now = True
        else:
            stalled_now = False
        cur = _current_node(st)
    if stalled_now:
        # One last, explicit message: try to unblock yourself once, and only then leave it to the human.
        print(f"[graph-loop] STALLED: {reason} Before you stop: read graph/blockers.md and graph/ledger.md, and try "
              f"ONE concrete unblocking action (a workaround from graph/decisions.md, `{PY} -m graphctl unblock <node>`, "
              f"or skipping an optional node). If the blocker is genuinely on the human-only list, make sure NEEDS_YOU.txt "
              f"and graph/blockers.md say exactly what you need, write a one-line summary in graph/ledger.md, and stop.")
        write_blocker("loop", reason,
                      f"Open Cursor, read graph/ledger.md and the last briefing, fix the cause, then run "
                      f"`{PY} -m graphctl resume` and send: `{PY} -m graphctl next`.")
        append_ledger("STALLED: " + reason)
        return EXIT_OK
    if status == "STALLED" and not running:
        return EXIT_OK
    print(render.followup(g, cur, streak, running[0] if (running and not cur) else None, maxs))
    return EXIT_OK


def cmd_halt(args) -> int:
    HALT_PATH.write_text(f"halted {now_iso()}: {args.reason or ''}\n", encoding="utf-8")
    with state_txn() as st:
        st["status"] = "HALTED"
    append_ledger(f"HALT: {args.reason or ''}")
    print(f"Halted. The stop hook will no longer re-prompt. `{PY} -m graphctl resume` to continue.")
    return EXIT_OK


def cmd_resume(args) -> int:
    clear_needs_you()
    if HALT_PATH.exists():
        try:
            HALT_PATH.unlink()
        except OSError as e:
            print(f"could not remove {HALT_PATH}: {e}")
            return EXIT_FAIL
    with state_txn() as st:
        st["status"] = "RUNNING"
        st.setdefault("followups", {})["streak"] = 0
        st["followups"]["last_progress_hash"] = ""
    append_ledger("RESUME")
    print(f"Resumed. Send `{PY} -m graphctl next` to the agent (or wait for the stop hook).")
    return EXIT_OK


def cmd_ledger(args) -> int:
    append_ledger(args.text)
    print("recorded")
    return EXIT_OK


REVIEW_KEYS = ("node", "reviewer", "verdict", "summary", "issues", "checked", "at")


def cmd_review_template(args) -> int:
    """Print the JSON the verifier must write."""
    tpl = {"node": args.node, "reviewer": "verifier (fresh context)", "verdict": "pass|fail",
           "summary": "one paragraph", "issues": ["..."], "checked": ["what you re-derived / re-ran"],
           "at": now_iso()}
    print(json.dumps(tpl, indent=1))
    print(f"\nWrite it to runs/verify/{args.node}.review.json")
    print(f"(a read-only verifier subagent that cannot write files replies with this JSON instead; the parent "
          f"then stores it verbatim: `{PY} -m graphctl record-review {args.node} --stdin`)")
    return EXIT_OK


def cmd_record_review(args) -> int:
    """Store a verifier's verdict JSON verbatim at runs/verify/<node>.review.json.

    Needed because `.cursor/agents/verifier.md` declares `readonly: true`: depending on how Cursor
    enforces that flag the subagent may be unable to create the file itself, and every
    `requires_review` node (N01, N02, N09, N12-N15) would then be impossible to finish."""
    node_by_id(load_graph(), args.node)
    raw = sys.stdin.read() if args.stdin else Path(args.file).read_text(encoding="utf-8-sig")
    raw = raw.strip().lstrip("﻿")
    if raw.startswith("```"):  # a subagent that wrapped the JSON in a fence
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0] if "\n" in raw else raw
    try:
        d = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        print(f"REFUSED: not valid JSON ({e}). Expected the schema from `graphctl review-template {args.node}`.")
        return EXIT_FAIL
    if not isinstance(d, dict) or str(d.get("verdict", "")).lower() not in ("pass", "fail"):
        print("REFUSED: the JSON needs a \"verdict\" of \"pass\" or \"fail\".")
        return EXIT_FAIL
    d["node"] = args.node
    d.setdefault("reviewer", "verifier")
    d.setdefault("at", now_iso())
    d["recorded_by"] = "graphctl record-review"
    d["recorded_at"] = now_iso()
    out = verify.VERIFY_DIR / f"{args.node}.review.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    append_ledger(f"REVIEW {args.node}: {d['verdict']}")
    print(f"wrote {out} (verdict {d['verdict']})")
    return EXIT_OK if str(d["verdict"]).lower() == "pass" else EXIT_FAIL


def _ladder_fires(cond: Dict[str, Any]) -> bool:
    if not cond:
        return False  # an empty `when` would otherwise fire on every turn, forever
    if "date_after" in cond and days_until(cond["date_after"]) > 0:
        return False
    if "node_not_done" in cond:
        st = load_state()["nodes"].get(cond["node_not_done"], {}).get("status")
        if st in ("done", "skipped"):
            return False
    if "json" in cond:
        try:
            d = json.loads((ROOT / cond["json"]["path"]).read_text(encoding="utf-8"))
            if not bool(eval(cond["json"]["expr"], {"__builtins__": {}}, {"d": d, **verify.EVAL_NS})):
                return False
        except Exception:  # noqa: BLE001  (file missing / unreadable / expression false)
            return False
    return True


def cmd_ladder(args) -> int:
    """Evaluate and (optionally) apply the decision ladder rules in graph/graph.yaml.

    This is the first command of every turn, so it must never raise: a rule whose target file is
    missing is reported and skipped, not allowed to abort the turn."""
    g = load_graph()
    rules = g.get("ladder", [])
    with state_txn() as st:
        applied = st.setdefault("ladder_applied", [])
        fired = [r for r in rules if r["id"] not in applied and _ladder_fires(r.get("when", {}))]
        if not fired:
            print("ladder: no rule fires now")
            return EXIT_OK
        rc = EXIT_OK
        for r in fired:
            print(f"LADDER RULE {r['id']}: {r.get('note', '')}")
            for s in r.get("set", []):
                print(f"   set {s['file']} :: {s['key']} = {s['value']!r}")
            if args.apply and (args.rule in (None, r["id"])):
                try:
                    for s in r.get("set", []):
                        _yaml_set(ROOT / s["file"], s["key"], s["value"])
                except Exception as e:  # noqa: BLE001
                    print(f"   NOT APPLIED: {e}")
                    append_ledger(f"LADDER FAILED {r['id']}: {e}")
                    rc = EXIT_FAIL
                    continue
                applied.append(r["id"])
                append_ledger(f"LADDER APPLIED {r['id']}: {r.get('note', '')}")
                print("   applied")
    return rc


def _fmt_yaml_scalar(value) -> str:
    return yaml.safe_dump(value, default_flow_style=True).strip().rstrip("...").strip()


def _yaml_set(path: Path, dotted: str, value) -> None:
    """Set ``a.b.c: value`` in a YAML file **without losing its comments or layout**.

    configs/main.yaml documents every key and the decision ladder edits it unattended, so a PyYAML
    round-trip (which strips comments) is the last resort.  Handles both block style::

        protocol:
          latency_queries: 300      # comment

    and the inline flow mappings this project uses::

        splits:
          nq:       {n_cal: 1000, n_train: 4000, n_dev: 1000}
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    keys = dotted.split(".")

    def find(key: str, depth: int, lo: int, hi: int):
        want = 2 * depth
        for i in range(lo, hi):
            line = lines[i]
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            ind = len(line) - len(line.lstrip())
            if ind < want and i > lo:
                return None, i
            if ind == want and re.match(rf"\s*{re.escape(key)}\s*:", line):
                return i, hi
        return None, hi

    lo, hi, pos = 0, len(lines), None
    for d, key in enumerate(keys):
        pos, hi_new = find(key, d, lo, hi)
        if pos is None:
            break
        rest = lines[pos].split(":", 1)[1] if ":" in lines[pos] else ""
        if "{" in rest and d < len(keys) - 1:           # inline flow mapping: patch inside the braces
            inner = ".".join(keys[d + 1:])
            if "." in inner:
                pos = None
                break
            m = re.search(rf"(\{{[^}}]*\b{re.escape(inner)}\s*:\s*)([^,}}]*)", lines[pos])
            if not m:
                body = re.search(r"\{([^}]*)\}", lines[pos])
                if not body:
                    pos = None
                    break
                items = [x for x in body.group(1).split(",") if x.strip()] + [f" {inner}: {_fmt_yaml_scalar(value)}"]
                lines[pos] = lines[pos][: body.start()] + "{" + ",".join(items) + "}" + lines[pos][body.end():]
            else:
                lines[pos] = lines[pos][: m.start(2)] + _fmt_yaml_scalar(value) + lines[pos][m.end(2):]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        lo, hi = pos + 1, hi_new if hi_new > pos else hi

    if pos is not None:                                  # plain scalar line: keep the trailing comment
        line = lines[pos]
        head, rest = line.split(":", 1)
        comment = "  " + rest[rest.index("#"):].strip() if "#" in rest else ""
        lines[pos] = f"{head}: {_fmt_yaml_scalar(value)}{comment}"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    d = yaml.safe_load(text) or {}                       # key absent: full round-trip (comments lost)
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")


def cmd_jobs(args) -> int:
    if args.jobs_cmd == "start":
        # argparse.REMAINDER swallows every flag placed after NAME, so re-parse them here.
        cmd = list(args.cmd)
        gpu = bool(args.gpu)
        node = getattr(args, "node", None)
        while cmd:
            head = cmd[0]
            if head == "--gpu":
                gpu, cmd = True, cmd[1:]
            elif head == "--node":
                node, cmd = (cmd[1] if len(cmd) > 1 else None), cmd[2:]
            elif head.startswith("--node="):
                node, cmd = head.split("=", 1)[1], cmd[1:]
            elif head == "--":
                cmd = cmd[1:]
                break
            else:
                break
        args.node = node
        if not cmd:
            print("usage: graphctl jobs start NAME [--gpu] [--node NODE] -- CMD ...")
            return EXIT_FAIL
        if node:
            try:
                node_by_id(load_graph(), node)
            except KeyError:
                print(f"REFUSED: --node {node} is not a node in graph/graph.yaml")
                return EXIT_FAIL
        try:
            info = jobs.start(args.name, cmd, gpu=gpu, cwd=ROOT)
        except (RuntimeError, ValueError) as e:
            print(f"REFUSED: {e}")
            return EXIT_FAIL
        # Poll briefly: a command that cannot start (bad path, missing module) must be reported now,
        # not discovered hours later by a `wait` that returns instantly.
        logfile = jobs.job_dir(args.name) / "log.txt"
        stj = jobs.status(args.name)
        for _ in range(40):
            if stj.get("state") in ("running", "failed", "finished"):
                break
            time.sleep(0.25)
            stj = jobs.status(args.name)
        if stj.get("state") == "starting" and not logfile.exists():
            jobs.kill(args.name)
            print(json.dumps(stj))
            print(f"REFUSED: the detached runner for {args.name} never opened its log in 10 s -- it did not "
                  f"start. Check that `{sys.executable} -m graphctl.jobrunner` is importable and that the "
                  f"interpreter path is right, then retry.")
            return EXIT_FAIL
        if stj.get("state") == "failed":
            print(json.dumps(stj))
            print("--- log tail ---")
            print(jobs.tail(args.name, 20))
            print("REFUSED: the job failed immediately (see the log tail above); fix the command and retry.")
            return EXIT_FAIL
        if node:
            with state_txn() as st:
                ns = _touch(st, node)
                if args.name not in ns["jobs"]:
                    ns["jobs"].append(args.name)
                if ns["status"] == "pending":  # otherwise `next` would dispatch it again in parallel
                    ns["status"] = "in_progress"
                    ns["iterations"] = ns.get("iterations", 0) + 1
                    ns["started_at"] = ns.get("started_at") or now_iso()
        append_ledger(f"JOB START {args.name}{' [gpu]' if gpu else ''}"
                      f"{' node=' + node if node else ''}: {jobs.format_cmd(cmd)}")
        print(json.dumps({**info, "state": stj.get("state")}))
        return EXIT_OK
    if args.jobs_cmd == "status":
        try:
            items = [jobs.status(args.name)] if args.name else jobs.list_jobs()
        except ValueError as e:
            print(f"REFUSED: {e}")
            return EXIT_FAIL
        for it in items:
            print(json.dumps(it))
        return EXIT_OK
    if args.jobs_cmd == "wait":
        st = jobs.wait(args.name, timeout=args.timeout, poll=args.poll)
        print(json.dumps(st))
        print("--- log tail ---")
        print(jobs.tail(args.name, 25))
        state = st["state"]
        if state == "finished":
            append_ledger(f"JOB FINISHED {args.name}")
            return EXIT_OK
        if state == "failed":
            append_ledger(f"JOB FAILED {args.name} exit={st.get('exit_code')}")
            return EXIT_FAIL
        if state == "stale":
            append_ledger(f"JOB STALE {args.name} ({st.get('reason', 'no heartbeat')})")
            print(f"The detached runner is gone. Re-start the job or `{PY} -m graphctl fail <node> --reason ...`.")
            return EXIT_FAIL
        if state == "unknown":
            print(f"No such job {args.name}. `{PY} -m graphctl jobs status` lists them.")
            return EXIT_FAIL
        return EXIT_IDLE  # still running after --timeout: end the turn, the stop hook brings you back
    if args.jobs_cmd == "tail":
        print(jobs.tail(args.name, args.lines))
        return EXIT_OK
    if args.jobs_cmd == "kill":
        print(json.dumps(jobs.kill(args.name)))
        append_ledger(f"JOB KILLED {args.name}")
        return EXIT_OK
    return EXIT_FAIL


def cmd_reset(args) -> int:
    from .state import STATE_PATH
    if not args.yes:
        print("This wipes graph/state.json (not runs/). Re-run with --yes. "
              "Add --all to also clear graph/ledger.md, graph/blockers.md, graph/HALT and runs/jobs/.")
        return EXIT_FAIL
    removed = []
    for p in [STATE_PATH] + ([LEDGER_PATH, BLOCKERS_PATH, HALT_PATH, GRAPH_DIR / "STATUS_zh.md"] if args.all else []):
        if p.exists():
            p.unlink()
            removed.append(p.name)
    if args.all:
        import shutil
        for d in (ROOT / "runs" / "jobs", ROOT / "runs" / "verify"):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d.name + "/")
    if not args.all:
        append_ledger("RESET state.json")
    print("state reset: " + (", ".join(removed) or "nothing to remove"))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graphctl", description="graph-loop controller")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("-v", "--verbose", action="store_true"); s.add_argument("--zh", action="store_true", help="also write the Chinese dashboard graph/STATUS_zh.md"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("next"); s.add_argument("--node"); s.add_argument("--max", type=int, default=None, help="max nodes to dispatch in parallel"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_next)
    s = sub.add_parser("brief"); s.add_argument("node"); s.set_defaults(fn=cmd_brief)
    s = sub.add_parser("verify"); s.add_argument("node"); s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("done"); s.add_argument("node"); s.add_argument("--note", default=""); s.add_argument("--force", action="store_true", help="allow `done` while a background job of this node is still running"); s.set_defaults(fn=cmd_done)
    s = sub.add_parser("fail"); s.add_argument("node"); s.add_argument("--reason", required=True); s.set_defaults(fn=cmd_fail)
    s = sub.add_parser("block"); s.add_argument("node"); s.add_argument("--reason", required=True); s.add_argument("--human", default=None); s.set_defaults(fn=cmd_block)
    s = sub.add_parser("unblock"); s.add_argument("node"); s.set_defaults(fn=cmd_unblock)
    s = sub.add_parser("skip"); s.add_argument("node"); s.add_argument("--reason", required=True); s.set_defaults(fn=cmd_skip)
    s = sub.add_parser("followup"); s.set_defaults(fn=cmd_followup)
    s = sub.add_parser("halt"); s.add_argument("--reason", default=""); s.set_defaults(fn=cmd_halt)
    s = sub.add_parser("resume"); s.set_defaults(fn=cmd_resume)
    s = sub.add_parser("ledger"); s.add_argument("text"); s.set_defaults(fn=cmd_ledger)
    s = sub.add_parser("review-template"); s.add_argument("node"); s.set_defaults(fn=cmd_review_template)
    s = sub.add_parser("record-review", help="store a verifier's verdict JSON at runs/verify/<node>.review.json")
    s.add_argument("node"); s.add_argument("--file", default=None); s.add_argument("--stdin", action="store_true"); s.set_defaults(fn=cmd_record_review)
    s = sub.add_parser("ladder"); s.add_argument("--apply", action="store_true"); s.add_argument("--rule", default=None); s.set_defaults(fn=cmd_ladder)
    s = sub.add_parser("reset"); s.add_argument("--yes", action="store_true"); s.add_argument("--all", action="store_true", help="also clear ledger/blockers/HALT/runs jobs+verify"); s.set_defaults(fn=cmd_reset)
    j = sub.add_parser("jobs"); js = j.add_subparsers(dest="jobs_cmd", required=True)
    a = js.add_parser("start"); a.add_argument("name"); a.add_argument("--gpu", action="store_true"); a.add_argument("--node", default=None, help="attach the job to a graph node (next shows it as WAITING)"); a.add_argument("cmd", nargs=argparse.REMAINDER)
    a = js.add_parser("status"); a.add_argument("name", nargs="?")
    a = js.add_parser("wait"); a.add_argument("name"); a.add_argument("--timeout", type=float, default=540); a.add_argument("--poll", type=float, default=15)
    a = js.add_parser("tail"); a.add_argument("name"); a.add_argument("--lines", type=int, default=40)
    a = js.add_parser("kill"); a.add_argument("name")
    j.set_defaults(fn=cmd_jobs)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "cmd", None) == "record-review" and not args.stdin and not args.file:
        print("record-review needs --file PATH or --stdin")
        return EXIT_FAIL
    try:
        return int(args.fn(args) or 0)
    except BrokenPipeError:  # e.g. piped into `head`
        try:
            sys.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        return EXIT_OK
    except KeyError as e:
        print(f"unknown node {e}. `{PY} -m graphctl status` lists every node id.", file=sys.stderr)
        return EXIT_FAIL
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
