# RISKS.md — what can go wrong and what the agent does (pre-decided)

## A. Science and schedule

| risk | detection | response |
|---|---|---|
| RTX 5090 not visible to torch (cu126 wheel / driver) | N00 env check; `bootstrap.py` warns if `torch.cuda.is_available()` is False | install cu128+ wheel; update driver hint in blockers if still failing |
| Teacher throughput far below plan | N05 pilot | ladder L1 (smaller train/dev); L2/L3 later |
| bf16 non-determinism changes top-k order on re-scoring | N07 audit | eager attention / fp32 audit; re-cache with deterministic setting; state in paper |
| HF hub / UKP mirror unreachable | N03/N05 | hf-mirror.com; DPR question file; retry with backoff |
| pyserini needs Java; Windows pain | N04 | bm25s fallback (recorded) |
| RSC loses to the adaptive calibrated prefix | N12 gates | claim level "policy-uniform-protection-with-cost"; paper centres on theory + guarantee + cost |
| Savings < 10% | N12 gates | claim level "theory-plus-negative-efficiency" |
| Closer prior work found | N01 review | re-scope per docs/LITERATURE.md §D; cite prominently |
| Page overflow | check_paper | move proofs/grids to appendix |
| Deadline slips | ladder L5 | write with what exists; never fabricate |
| Human forgets abstract registration | N16 handoff | the packet says it in the first line; the paper cannot be submitted otherwise |

## B. Running unattended for five weeks on one Windows box

Each row: how it shows up, how it is *detected*, and what recovers it. Recovery is automatic unless the
"human" column says otherwise; anything needing a human is written to `graph/blockers.md`, which is the only
file the human has to watch.

| risk | symptom | detection | recovery | human? |
|---|---|---|---|---|
| **Machine sleeps / hibernates** | everything freezes, heartbeats stop, clocks jump | on wake, `jobs status` shows `stale` (heartbeat older than 180 s) and the GPU lock is released | the agent re-starts the job; partial caches are resumable (every stage is idempotent with a manifest) | prevention only: power plan → never sleep (START_HERE §1.3) |
| **Windows Update reboot** | Cursor and every detached job are gone | after restart `graph/state.json` still holds the graph; jobs are `stale`; `graph/state.corrupt-*.json` appears if the write was cut mid-way | reopen Cursor and send `PY -m graphctl next`; the `sessionStart` hook re-orients the agent, finished nodes re-verify from their files, unfinished jobs restart | yes — reopening Cursor. Prevention: pause update restarts for the month |
| **Cursor kills an idle background shell** | a long command dies silently mid-turn | it cannot: every job > 5 min is a *detached* process (`DETACHED_PROCESS`/`start_new_session`), not a terminal command | none needed; `jobs wait` re-attaches to progress | no |
| **Cursor session dies / is closed / crashes** | no `stop` hook, so no follow-up, so no more turns | `graph/STATUS_zh.md` stops updating; `jobs status` still shows running jobs | detached jobs keep running; reopen Cursor — `sessionStart` injects the state and any message restarts the loop | yes — reopening Cursor |
| **Stop hook drops its output** (known issue with very fast hooks) | one missed turn | none | `_hookio.emit()` flushes and sleeps 50 ms; the next hook fires on the following turn; nothing is lost | no |
| **The loop spins without progress** | turns go by, nothing changes | `graphctl followup` hashes node states + iterations + meaningful ledger lines + job heartbeats (a *failing* `verify` is deliberately not progress); 8 identical hashes with no job running | one last explicit message ("try one unblocking action"), then `STALLED` + `graph/blockers.md` + **`NEEDS_YOU.txt`** at the repo root, and no more re-prompting | yes — `graphctl resume` |
| **A node is dispatched again and again without finishing** (e.g. a required review verdict never arrives) | the same node in every dispatch block, nothing completes | `next` counts re-dispatches of an already in-progress node (`loop.max_redispatches`, 6) | the node is failed (iteration + 1) and after its iterations are spent it is `blocked` → blockers.md + NEEDS_YOU.txt | only once it is blocked |
| **A node is stuck `in_progress` for hours** | it holds a `max_parallel` slot and never retries | `next` looks at `touched_at`, which only a real state change refreshes; more than `loop.stuck_node_hours` (12) with no running job | recycled to `pending` (its iteration is already counted), and `blocked` once the 6 iterations are spent | no |
| **The Cursor hooks do not run on this machine** (wrong shell, wrong interpreter path) | the agent stops after every turn and waits | `scripts/check_hooks.py` runs each hook command through the shell Cursor uses; it is a verification step of node N00 | `--fix` rewrites `.cursor/hooks.json` with the absolute interpreter path; until then the human just says "继续" | first run only |
| **A blocked *optional* node at the very end** | everything is finished but the loop says STALLED | `next` sees nothing dispatchable and only optional blocked nodes remain | those are auto-skipped with a ledger line so `GRAPH DONE` is reached | no |
| **Stale GPU lock** (job killed, runner gone, reboot) | no GPU job can start: "GPU is held by …" | `gpu_busy()` asks the holder's real state; a `starting` job keeps the lock, a `stale`/`finished`/`failed`/unknown one loses it | the lock is deleted automatically on the next `next`/`jobs start` | no |
| **Two GPU jobs at once → OOM, corrupted cache** | CUDA OOM, garbage scores | `--gpu` jobs take `runs/jobs/GPU.lock`; `next` never dispatches a second GPU node while one is held, planned or waiting | refused at `jobs start`; regression-tested | no |
| **HF download interrupted** (network drop, CDN hiccup, mirror change) | a stage aborts with a partial file | the stage's manifest is not written, so the node's `json_check` fails | stages are resumable and idempotent: re-run downloads only what is missing; `hf-mirror.com` is the recorded fallback (D-DATA-2). **No network allowlist is enforced** (see docs/CURSOR_SETUP.md §6) precisely so a rotating HF CDN host cannot kill a run | no |
| **Disk full** (teacher caches + HF cache + logs on one SSD) | writes fail everywhere | `append_ledger`, `write_blocker` and the job heartbeat swallow OSError and warn instead of crashing, so the failure surfaces as a failed verify rather than a dead loop | the agent frees `runs/smoke`, `runs/jobs/*/log.txt` of finished jobs and the pip cache — never `data/raw`, `data/teacher`, `data/candidates` or `models/heads` (guard_shell blocks that) | prevention: ≥ 150 GB free |
| **A job log grows to gigabytes** | `jobs tail` used to read the whole file | `tail()` seeks to the last 512 KiB; `next`/`brief` never read a whole log | none | no |
| **`graph/state.json` truncated by a crash** | JSON decode error on every command | `load_state()` catches it | the file is copied to `graph/state.corrupt-*.json`, the loop restarts from an empty state and re-verifies finished nodes from their outputs | no |
| **Two subagents write state at once** | lost node transitions | every mutation goes through `state_txn()` → `graph/.state.lock` (O_EXCL, token-owned, stale after 120 s) | serialised; a lock older than 120 s is broken so a killed agent cannot wedge the loop | no |
| **The detached runner cannot start** (broken venv, security software) | a 30-hour job that never actually runs | `jobs start` waits 10 s for `log.txt`, and `status` reports `stale` ("never started") after 30 s | `jobs start` refuses loudly and releases the GPU lock | no |
| **Windows code page mangles output** | `UnicodeEncodeError` kills a job hours in | — | every child process gets `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`; every file is opened with `encoding="utf-8"`; hooks emit ASCII-only JSON | no |
| **Clock/timezone confusion around the AoE deadline** | wrong "days left" | `days_until()` computes in UTC+8 explicitly and the deadlines in `graph.yaml` carry `-12:00` (AoE) | none | no |
| **The agent halts itself or resets the state** | the run ends quietly | `guard_shell` denies `graphctl halt` and `graphctl reset --yes` from the agent | escalation is `graphctl block <node> --reason … --human …`, which keeps other nodes moving | no |
