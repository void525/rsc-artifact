# CURSOR_SETUP.md — how this repo drives Cursor
*(Cursor facts verified 2026-09-20 against cursor.com/docs and the 3.21.x release notes; graphctl behaviour verified
against `tests/test_graphctl.py` on Linux. Points that could only be reasoned about, not executed, on Windows are marked
**untested on Windows**.)*

`PY` below always means **`./.venv/Scripts/python.exe`** — that one spelling runs in PowerShell *and* in bash,
because `bootstrap.py` creates exactly that path on every OS (a real file on Windows, a symlink or a tiny `sh`
launcher on POSIX). Nothing in this project depends on `python` being on PATH.

---

## 0. Order of operations (the whole setup)

1. Open the folder in **Cursor 3.x** on the machine with the RTX 5090.
2. Cursor Settings → Agents → Approvals & Execution → run mode **“Run Everything”** (second choice: “Auto-review”).
   *This is the one switch a repository cannot set for you.* Without it the agent stops for approval every few minutes
   and the five-week run dies the first night.
3. Power plan → never sleep; pause Windows Update restarts; ≥ 150 GB free disk.
4. Send the agent the kick-off message from `START_HERE.md`. Its first command is `python bootstrap.py`
   (on Windows without `python` on PATH: `& "$env:USERPROFILE\anaconda3\python.exe" bootstrap.py`).
5. Everything after that is automatic until one of the two OpenReview steps.

**Before `bootstrap.py` has run there is no `.venv`, so the hooks cannot execute.** Cursor fails open on a hook
that exits non-zero, so the session still works — but the `stop` hook cannot bring the agent back yet. This only
matters for the very first turn, which is the one turn the human is watching anyway.

---

## 1. What the repository configures for you

| file | what it does |
|---|---|
| `AGENTS.md` (root) | Cursor reads it in every conversation: the loop, the rhythm of a turn, where things are. |
| `.cursor/rules/*.mdc` | `00-mission`, `10-graph-loop`, `20-science-integrity` are `alwaysApply: true`; `30-python-style`, `40-paper-writing`, `50-experiments` are glob-scoped. |
| `.cursor/agents/*.md` | nine subagents (below). Frontmatter keys used: `name`, `description`, `model: inherit`, `readonly`. |
| `.cursor/skills/*/SKILL.md` | four skills: `/run-graph`, `/verify-node`, `/review-round`, `/graph-status` (folder name == `name`). Skills replaced the deprecated custom commands. |
| `.cursor/hooks.json` | the four hooks below. |
| `.cursor/permissions.json` | natural-language guidance for the Auto-review classifier. Advisory only. |
| `.cursor/cli.json` | permissions for the headless `agent` CLI (Plan B). |

There is **no** `.cursor/sandbox.json` and **no** `.cursor/environment.json` — see §6 for why they were removed.

### Subagents

Parallelism in Cursor = several Task calls **in one message**; each subagent gets its own context.
The parent delegates from the `description` field, so those are written as “use this for …”.

| agent | readonly | used for |
|---|---|---|
| `verifier` | yes | the fresh-context check of a finished node → `runs/verify/<node>.review.json` |
| `reviewer` | yes | simulated WWW PC reviewer in N15 |
| `auditor` | yes | numbers / anonymity / format / policy audit |
| `planner` | yes | re-scoping when a node is late or blocked |
| `researcher`, `theorist`, `implementer`, `experimenter`, `writer` | no | the actual work |

> **The readonly escape hatch.** A `readonly` subagent may be unable to create its output file.
> `graphctl done` refuses without `runs/verify/<node>.review.json`, and seven nodes require a review
> (N01, N02, N09, N12, N13, N14, N15) — so the loop must not depend on the subagent being able to write.
> Each read-only agent is therefore instructed to *reply* with its JSON if it cannot write it, and the parent
> stores that reply verbatim with `PY -m graphctl record-review <node> --stdin`. Nothing is rewritten in between.

### Hooks (`.cursor/hooks.json`, `version: 1`)

Hook commands run from the project root; on Windows Cursor executes them through PowerShell (pwsh preferred),
and the stdin payload may carry a UTF-8 BOM — `.cursor/hooks/_hookio.py` strips it, prints pure-ASCII JSON,
flushes and sleeps ~50 ms (very fast hooks have had their output dropped).

| event | script | effect |
|---|---|---|
| `sessionStart` | `session_start.py` | returns `{"additional_context": …}` with `graphctl status`, so a fresh or compacted context starts oriented. |
| `beforeShellExecution` | `guard_shell.py` | `{"command","cwd","sandbox"}` → `{"permission":"allow"\|"deny","user_message","agent_message"}` (snake_case). Denies about a dozen patterns: deleting `data/raw` / `data/teacher` / `data/candidates` / `models/heads`, `rm -rf /` `~` `$HOME` `*`, `git push --force`, `git reset --hard`, `git clean -f`, `sudo`/`runas`, power and format commands, anything that looks like an OpenReview upload, `pip uninstall torch`, `graphctl reset --yes`, and `graphctl halt` (only the human halts the run — the agent escalates with `graphctl block`). Everything else is allowed. |
| `stop` | `stop_followup.py` | the engine of the unattended loop. Prints `{"followup_message": …}` from `graphctl followup`, which Cursor auto-submits; `loop_limit: null` removes the default cap of 5. It returns `{}` — i.e. lets the agent rest — only when the human pressed stop (`status: "aborted"`), when `graph/HALT` exists, when the graph is DONE/HALTED, or when the loop has paused itself (below). |
| `subagentStop` | `subagent_stop.py` | one line in `runs/subagents.log`. Deliberately *not* the ledger: the ledger feeds the no-progress detector. |

### The self-pause

`graphctl followup` hashes node statuses + iteration counts + meaningful ledger lines + running-job heartbeats.
After `loop.max_followups_without_progress` (8) follow-ups with an unchanged hash *and* no running job, it sets the
graph to `STALLED`, writes `graph/blockers.md`, and goes quiet — better a paused loop than one burning tokens in a
circle. `PY -m graphctl resume` restarts it. `graphctl next` also writes a blocker whenever it stalls, so
`graph/blockers.md` is always the single place the human has to look.

---

## 2. Long jobs

Foreground terminal commands in Cursor time out (~10 min reported) and get moved to the background, and an idle
session can kill background shells. So **anything longer than five minutes is a detached process**:

```
PY -m graphctl jobs start teacher_qwen_cal --node N07_teacher_cache --gpu -- python scripts/05_teacher_scores.py ...
PY -m graphctl jobs status | jobs tail NAME --lines 40 | jobs wait NAME --timeout 540 | jobs kill NAME
```

`jobs start` spawns `graphctl.jobrunner` with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows
(**untested on Windows**) and `start_new_session=True` on POSIX, so it survives the terminal, the turn and the
session. The runner streams to `runs/jobs/NAME/log.txt`, touches `heartbeat` every 20 s, writes `status.json` on
exit and releases `runs/jobs/GPU.lock`. `--gpu` takes that lock; a second `--gpu` job is refused. A node with a
running job is shown by `next` as **WAITING** and is not dispatched — the agent does CPU nodes meanwhile.

---

## 3. Headless Plan B (no IDE window)

```powershell
irm 'https://cursor.com/install?win32=true' | iex     # Cursor CLI on Windows
agent login
./.venv/Scripts/python.exe scripts/run_headless.py     # --dry-run to see the prompt first
```

The driver loops `agent -p --force --trust --output-format text "<kick-off prompt>"` against a *fresh* agent each
turn (all state is in files), sleeps while only background jobs run, respects `graph/HALT`, and holds
`runs/headless/driver.lock` so it cannot race an IDE session. The CLI reads the same `AGENTS.md`, rules, subagents
and hooks; `--force --trust` bypasses `.cursor/cli.json`, which is there for hand-driven CLI use.
**Run one driver at a time — never the IDE agent and the headless driver together.**

---

## 4. If the loop stalls

```
PY -m graphctl status -v            # graph status, per-node state, attached jobs, deadlines
type graph\blockers.md              # the only place the agent asks for a human
PY -m graphctl jobs status          # is something actually running?
PY -m graphctl jobs tail NAME --lines 60
```

* **`graph status: STALLED`** → read `graph/blockers.md`, fix the cause, `PY -m graphctl unblock <node>` if a node is
  blocked, then `PY -m graphctl resume` and send `PY -m graphctl next` to the agent.
* **`graph status: HALTED` / `graph/HALT` present** → `PY -m graphctl resume`.
* **A job shows `stale`** → its detached runner is gone (reboot, kill, disk full). Re-start the job; the GPU lock is
  released automatically once the job is stale.
* **A node sits `in_progress` for hours with no job** → `next` recycles it after `loop.stuck_node_hours` (12) and
  eventually blocks it, so the loop can always terminate. You can force it with `PY -m graphctl fail <node> --reason …`.
* **The agent stopped and does not come back** → the `stop` hook is the only thing that restarts it. Check
  `.venv/Scripts/python.exe` exists, run `PY -m graphctl followup` by hand (it should print a message), and check the
  run mode is still “Run Everything”. Sending any message to the agent also restarts the loop.
* **`graph/state.json` corrupted** (power cut) → graphctl moves it to `graph/state.corrupt-*.json` and continues from an
  empty state; every finished node re-verifies immediately from the files it produced, so almost nothing is lost.

---

## 5. Context

Cursor compacts context automatically. Nothing in this project lives in the conversation: the `sessionStart` hook plus
`graphctl status` / `next` / `brief` make any fresh context self-sufficient. Give subagents **file paths, not summaries**.

---

## 6. Two Cursor features this project deliberately does not use

* **Cloud / background agents (`.cursor/environment.json` — deleted).** Cloud agents run in a container with no GPU,
  and the entire point of this repo is the local RTX 5090. Keeping a cloud-agent environment file invited someone to
  launch a run that would silently produce CPU-only or synthetic results. Its `install` line also used `&&` and
  `|| true`, which the project's own Windows rule forbids.
* **The workspace sandbox (`.cursor/sandbox.json` — deleted).** Its schema is unverified, sandboxing is not applied the
  same way on Windows, and it contained `networkPolicy.default: "deny"` with a hand-written host allowlist. Hugging
  Face serves large files from rotating CDN hosts (`cdn-lfs*`, Xet storage), and PyPI/CTAN/BEIR mirrors move too: one
  missing host would kill a multi-hour download at 3 a.m. with nobody watching. Network safety here comes from the
  `beforeShellExecution` guard and from `.cursor/cli.json`, not from an allowlist that can only be wrong.
