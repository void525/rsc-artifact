"""Text rendering: node briefing (for `next`) and the stop-hook follow-up."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import jobs
from .graph import days_until, node_by_id, read_spec, topo_order
from .state import BLOCKERS_PATH, LEDGER_PATH, py_cmd


PY = py_cmd()


def _counts(g, state) -> Dict[str, int]:
    c = {"done": 0, "in_progress": 0, "pending": 0, "blocked": 0, "skipped": 0, "verified": 0}
    for n in g["nodes"]:
        s = state["nodes"].get(n["id"], {}).get("status", "pending")
        c[s] = c.get(s, 0) + 1
    return c


def deadline_line(g: Dict[str, Any]) -> str:
    dl = g.get("deadlines", {})
    parts = []
    for key in ("abstract_registration", "full_paper", "experiment_freeze", "writing_freeze"):
        if key in dl:
            parts.append(f"{key.replace('_', ' ')} in {days_until(dl[key]):.1f} d")
    return "; ".join(parts)


def running_jobs_line() -> str:
    try:
        js = [j for j in jobs.list_jobs() if j["state"] in ("running", "starting", "stale")]
        holder = jobs.gpu_busy()
    except OSError as e:
        return f"Running jobs: (runs/jobs unreadable: {e})"
    if not js:
        return "Running jobs: none | GPU: free"
    return ("Running jobs: " + ", ".join(f"{j['name']}({j['state']})" for j in js)
            + f" | GPU: {'held by ' + holder if holder else 'free'}")


def tail_file(path, n: int = 12, max_bytes: int = 200_000) -> str:
    """Last n lines, reading only the tail: graph/ledger.md grows for five weeks."""
    if not path.exists():
        return "(none)"
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()
            data = f.read()
    except OSError as e:
        return f"(unreadable: {e})"
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-n:])


def briefing(g: Dict[str, Any], state: Dict[str, Any], node: Dict[str, Any], iteration: int,
             job_states: Optional[List[Dict[str, Any]]] = None) -> str:
    max_it = g.get("loop", {}).get("max_iterations_per_node", 6)
    lf = node.get("latest_finish")
    lf_txt = f"; this node's latest finish {lf} ({days_until(lf):.1f} d left)" if lf else ""
    c = _counts(g, state)
    lines = [
        "=" * 78,
        f"GRAPH LOOP :: NODE {node['id']} -- {node.get('title', '')}   (iteration {iteration}/{max_it})",
        "=" * 78,
        f"Deadlines: {deadline_line(g)}{lf_txt}",
        running_jobs_line(),
        f"Graph: {c['done']} done, {c['in_progress']} in progress, {c['verified']} verified, {c['pending']} pending, {c['blocked']} blocked, {c['skipped']} skipped",
        f"Resources: {', '.join(node.get('resources', [])) or 'cpu'} | requires fresh-context review: {'YES' if node.get('requires_review') else 'no'}",
        _jobs_line(job_states),
        "",
        f"NOTE: inside the spec below, `python` and `PY` both mean **{PY}** (the project interpreter);",
        "      on Windows the shell is PowerShell: run commands one per line, no `&&`, no shell redirects (`>`).",
        "",
        f"--- NODE SPEC ({node['spec']}) ---",
        read_spec(node).strip(),
        "",
        f"--- MACHINE VERIFICATION (run: {PY} -m graphctl verify " + node["id"] + ") ---",
    ]
    for step in node.get("verify", []):
        if "cmd" in step:
            lines.append(f"  - cmd: {step['cmd']}")
        elif "file_exists" in step:
            lines.append(f"  - file exists: {step['file_exists']}")
        elif "json_check" in step:
            lines.append(f"  - json: {step['json_check']['path']} :: {step['json_check']['expr']}")
        elif "review" in step:
            lines.append(f"  - fresh-context review verdict file: runs/verify/{step['review']}.review.json (verdict: pass)")
    fan = node.get("fanout")
    if fan:
        lines += ["", "--- FAN-OUT PLAN (spawn these as parallel subagents; then reduce) ---"]
        for f in fan:
            lines.append(f"  * [{f.get('agent', 'general')}] {f['task']}")
    lines += [
        "",
        "--- PROTOCOL FOR THIS NODE (the diamond: split -> workers -> checker -> merge) ---",
        "  1. Read the spec, the relevant configs and graph/decisions.md. Decide; do not ask the human.",
        "  2. Split the work as in the fan-out plan (subagents run in parallel, each with a clean context).",
        f"  3. Do the work. Long commands (> 5 min) go through `{PY} -m graphctl jobs start NAME [--gpu] -- CMD`;",
        f"     then `{PY} -m graphctl jobs wait NAME --timeout 600` (re-run after a tool timeout; ending the turn is fine,",
        "     the stop hook brings you back).",
        f"  4. Check: `{PY} -m graphctl verify " + node["id"] + "`. If a fresh-context review is required, spawn the",
        "     `verifier` subagent with the review prompt from the spec; it writes runs/verify/" + node["id"] + ".review.json.",
        f"  5. Merge: fix everything the checker found, re-verify, then `{PY} -m graphctl done " + node["id"] + " --note \"...\"`.",
        f"     On an unrecoverable problem: `{PY} -m graphctl fail " + node["id"] + " --reason \"...\"` (retries) or",
        f"     `{PY} -m graphctl block " + node["id"] + " --reason \"...\"` (only for decisions outside your authority).",
        "",
        "--- LEDGER TAIL ---",
        tail_file(LEDGER_PATH, 8),
        "--- OPEN BLOCKERS ---",
        tail_file(BLOCKERS_PATH, 6),
    ]
    return "\n".join(lines)


def _jobs_line(job_states: Optional[List[Dict[str, Any]]]) -> str:
    if not job_states:
        return "This node has no background job attached."
    parts = []
    for j in job_states:
        extra = ""
        if j.get("state") == "stale":
            extra = "  <- the detached runner is gone; re-start it or `fail` the node"
        elif j.get("state") == "failed":
            extra = f"  <- exit {j.get('exit_code')}; read `jobs tail {j['name']}`"
        parts.append(f"{j['name']}({j.get('state')}){extra}")
    return "Jobs attached to this node: " + "; ".join(parts)


def dispatch(g: Dict[str, Any], state: Dict[str, Any], plan: List[Dict[str, Any]], waiting: List[Dict[str, Any]],
             iteration_of: Dict[str, int], recycled: Optional[List[str]] = None) -> str:
    """The parallel work plan for this turn: split -> workers -> checker -> merge."""
    c = _counts(g, state)
    lines = [
        "=" * 78,
        f"GRAPH LOOP :: DISPATCH {len(plan)} NODE(S) IN PARALLEL" + (f"  |  {len(waiting)} NODE(S) WAITING ON JOBS" if waiting else ""),
        "=" * 78,
        f"Deadlines: {deadline_line(g)}",
        running_jobs_line(),
        f"Graph: {c['done']} done, {c['in_progress']} in progress, {c['pending']} pending, {c['blocked']} blocked, {c['skipped']} skipped",
        "",
    ]
    if recycled:
        lines += [f"!! RECYCLED (sat in_progress with no job and no progress): {', '.join(recycled)}"
                  "  -- find out why before re-running them the same way.", ""]
    if plan:
        lines.append("--- WORK NOW (one subagent per node, all launched in ONE message) ---")
        for n in plan:
            nid = n["id"]
            lines.append(f"  [{nid}] {n.get('title','')}   (iteration {iteration_of.get(nid, 1)}; resources: {', '.join(n.get('resources', ['cpu']))}"
                         f"{'; fresh-context review required' if n.get('requires_review') else ''})")
            lines.append(f"       brief:  {PY} -m graphctl brief {nid}        spec: {n['spec']}")
            if n.get("latest_finish"):
                lines.append(f"       latest finish {n['latest_finish']} ({days_until(n['latest_finish']):+.1f} d)")
        lines += [
            "",
            "Protocol for each dispatched node (the diamond):",
            f"  1. Spawn a worker subagent whose first instruction is: `{PY} -m graphctl brief <node>` and follow it.",
            "     Launch every worker in a single message so they run in parallel; give each only file paths, not your memory.",
            "  2. Long work (> 5 min): the worker starts a detached job",
            f"     `{PY} -m graphctl jobs start NAME --node <node> [--gpu] -- CMD` and returns immediately.",
            f"  3. When a worker returns: `{PY} -m graphctl verify <node>`; if the node requires a review, spawn the",
            "     `verifier` subagent (clean context) with the node's review prompt -> runs/verify/<node>.review.json.",
            f"  4. `{PY} -m graphctl done <node> --note \"...\"` (refused unless checks pass), then loop.",
        ]
    if waiting:
        lines += ["", "--- WAITING ON BACKGROUND JOBS (do not block the turn on these) ---"]
        for w in waiting:
            js = ", ".join(f"{j['name']}({j['state']})" for j in w["jobs"])
            lines.append(f"  [{w['id']}] {w.get('title','')}: {js}")
            lines.append(f"       progress: {PY} -m graphctl jobs tail {w['jobs'][0]['name']} --lines 20")
            lines.append(f"       block until it moves: {PY} -m graphctl jobs wait {w['jobs'][0]['name']} --timeout 540")
    lines += ["", "--- LEDGER TAIL ---", tail_file(LEDGER_PATH, 6), "--- OPEN BLOCKERS ---", tail_file(BLOCKERS_PATH, 4)]
    return "\n".join(lines)


def status_zh(g: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Chinese one-page dashboard for the human (graph/STATUS_zh.md); nobody needs to read it to work."""
    c = _counts(g, state)
    dl = g.get("deadlines", {})
    total = max(1, len(g["nodes"]))
    lines = [
        f"# 进度看板（由 `{PY} -m graphctl status --zh` 自动生成，请勿手改）",
        "",
        f"- 生成时间：{__import__('datetime').datetime.now().replace(microsecond=0).isoformat()}",
        f"- 图状态：**{state.get('status')}**；节点完成 {c['done']}/{total}（进行中 {c['in_progress']}，待办 {c['pending']}，受阻 {c['blocked']}，跳过 {c['skipped']}）",
        f"- 摘要注册截止：{dl.get('abstract_registration','?')}（剩 {days_until(dl['abstract_registration']):.1f} 天）" if dl.get("abstract_registration") else "",
        f"- 全文截止：{dl.get('full_paper','?')}（剩 {days_until(dl['full_paper']):.1f} 天）" if dl.get("full_paper") else "",
        f"- 后台任务：{running_jobs_line()}",
        "",
        "## 需要你做的事（只有两件）",
        "1. **10 月 18 日 AoE 前**：在 OpenReview 注册摘要 —— 材料见 `graph/HANDOFF_ABSTRACT.md`（生成后才有）。",
        "2. **10 月 25 日 AoE 前**：上传 `submission/main.pdf` —— 步骤见 `graph/HANDOFF.md`（生成后才有）。",
        "",
        "## 节点",
        "",
        "| 节点 | 状态 | 迭代 | 最晚完成 |",
        "|---|---|---:|---|",
    ]
    for nid in topo_order(g):
        n = node_by_id(g, nid)
        ns = state["nodes"].get(nid, {})
        lines.append(f"| {nid} — {n.get('title','')} | {ns.get('status','pending')} | {ns.get('iterations',0)} | {n.get('latest_finish','')} |")
    blockers = BLOCKERS_PATH.read_text(encoding='utf-8', errors='replace') if BLOCKERS_PATH.exists() else ""
    lines += ["", "## 需要人工处理的阻塞项", "", ("（无）" if not blockers.strip() else "见 `graph/blockers.md`（最后几条）：\n\n```\n" + "\n".join(blockers.splitlines()[-12:]) + "\n```")]
    return "\n".join([l for l in lines if l is not None])


def followup(g: Dict[str, Any], current: Optional[str], streak: int, waiting_job: Optional[str],
             maxs: int = 8) -> str:
    dl = deadline_line(g)
    head = "[graph-loop] Continue autonomously; do not wait for or ask the human (graph/decisions.md pre-authorises every routine decision)."
    if waiting_job:
        return (
            f"{head} Background job `{waiting_job}` is still running. Run `{PY} -m graphctl jobs wait {waiting_job} --timeout 540`, "
            f"then `{PY} -m graphctl next` and act on the briefing (CPU-only nodes can proceed meanwhile). Deadlines: {dl}."
        )
    if current:
        return (
            f"{head} Current node: {current}. Run `{PY} -m graphctl next` and follow the briefing to completion "
            f"(verify -> review -> done). Deadlines: {dl}. Follow-ups without recorded progress: {streak}/{maxs} -- "
            f"end every turn with a graphctl action (verify/done/fail/jobs) or `{PY} -m graphctl ledger \"what you did\"`; "
            f"at {maxs} the loop pauses itself."
        )
    return f"{head} Run `{PY} -m graphctl next` to pick the next ready node and execute it end-to-end. Deadlines: {dl}."
