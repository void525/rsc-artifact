"""Tests for the graph-loop controller (graphctl) and the three Cursor hooks.

Fast, offline, and Windows-safe: every test runs against a throwaway repository under ``tmp_path``
and monkeypatches graphctl's module-level paths, so the real ``graph/state.json``, ``graph/ledger.md``
and ``runs/`` are never touched.  No test asserts on POSIX-only behaviour.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from graphctl import cli, graph as gmod, jobs, render, state, verify  # noqa: E402
from graphctl.lock import file_lock  # noqa: E402

HOOKS = REPO / ".cursor" / "hooks"

# --------------------------------------------------------------------------------------- fixtures

MINI_GRAPH = """
version: 1
project: test
deadlines:
  abstract_registration: "2026-10-18T23:59:00-12:00"
  full_paper: "2026-10-25T23:59:00-12:00"
loop:
  max_iterations_per_node: 3
  max_followups_without_progress: 2
  max_parallel_nodes: 2
  stuck_node_hours: 12
nodes:
  - id: A
    title: root
    spec: graph/nodes/A.md
    depends_on: []
    resources: [cpu]
    priority: 0
    writes: ["runs/a/**"]
    verify:
      - file_exists: runs/a/ok.txt
  - id: B
    title: cpu one
    spec: graph/nodes/B.md
    depends_on: [A]
    resources: [cpu]
    priority: 5
    writes: ["runs/b/**"]
    verify:
      - file_exists: runs/b/ok.txt
  - id: C
    title: cpu two, same write area as B
    spec: graph/nodes/C.md
    depends_on: [A]
    resources: [cpu]
    priority: 6
    writes: ["runs/b/sub/**"]
    verify:
      - file_exists: runs/c/ok.txt
  - id: D
    title: gpu one
    spec: graph/nodes/D.md
    depends_on: [A]
    resources: [gpu]
    priority: 7
    writes: ["runs/d/**"]
    verify:
      - file_exists: runs/d/ok.txt
  - id: E
    title: gpu two
    spec: graph/nodes/E.md
    depends_on: [A]
    resources: [gpu]
    priority: 8
    writes: ["runs/e/**"]
    verify:
      - file_exists: runs/e/ok.txt
  - id: F
    title: reviewed node
    spec: graph/nodes/F.md
    depends_on: [B]
    resources: [cpu]
    priority: 9
    requires_review: true
    writes: ["runs/f/**"]
    verify:
      - file_exists: runs/f/ok.txt
      - review: F
  - id: G
    title: optional node
    spec: graph/nodes/G.md
    depends_on: [A]
    resources: [cpu]
    priority: 90
    optional: true
    writes: ["runs/g/**"]
    verify:
      - file_exists: runs/g/ok.txt
ladder:
  - id: L_json
    note: "fires when the pilot json says not ok"
    when: {json: {path: runs/pilot.json, expr: "d['ok'] is False"}}
    set:
      - {file: configs/main.yaml, key: splits.nq.n_train, value: 2000}
  - id: L_date
    note: "fires after a date that has passed"
    when: {date_after: "2000-01-01T00:00:00+08:00", node_not_done: A}
    set: []
"""


def _write_repo(root: Path) -> None:
    (root / "graph" / "nodes").mkdir(parents=True)
    (root / "graph" / "graph.yaml").write_text(MINI_GRAPH, encoding="utf-8")
    for nid in "ABCDEFG":
        (root / "graph" / "nodes" / f"{nid}.md").write_text(f"# node {nid}\nDo the thing.\n", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "main.yaml").write_text("splits:\n  nq:\n    n_train: 4000\n", encoding="utf-8")
    (root / "runs").mkdir()


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A throwaway repository with graphctl's module-level paths pointed at it."""
    root = tmp_path / "repo"
    root.mkdir()
    _write_repo(root)
    _patch_paths(monkeypatch, root)
    return root


def _patch_paths(monkeypatch, root: Path) -> None:
    graph_dir, runs = root / "graph", root / "runs"
    for mod, name, val in [
        (state, "ROOT", root), (state, "GRAPH_DIR", graph_dir), (state, "RUNS", runs),
        (state, "STATE_PATH", graph_dir / "state.json"), (state, "LEDGER_PATH", graph_dir / "ledger.md"),
        (state, "BLOCKERS_PATH", graph_dir / "blockers.md"), (state, "HALT_PATH", graph_dir / "HALT"),
        (state, "LOCK_PATH", graph_dir / ".state.lock"),
        (gmod, "GRAPH_PATH", graph_dir / "graph.yaml"), (gmod, "ROOT", root), (gmod, "GRAPH_DIR", graph_dir),
        (jobs, "JOBS_DIR", runs / "jobs"), (jobs, "GPU_LOCK", runs / "jobs" / "GPU.lock"), (jobs, "RUNS", runs),
        (verify, "ROOT", root), (verify, "RUNS", runs), (verify, "VERIFY_DIR", runs / "verify"),
        (render, "LEDGER_PATH", graph_dir / "ledger.md"), (render, "BLOCKERS_PATH", graph_dir / "blockers.md"),
        (render, "PY", "PY"),
        (cli, "ROOT", root), (cli, "GRAPH_DIR", graph_dir), (cli, "HALT_PATH", graph_dir / "HALT"),
        (cli, "LEDGER_PATH", graph_dir / "ledger.md"), (cli, "BLOCKERS_PATH", graph_dir / "blockers.md"),
        (cli, "PY", "PY"),
    ]:
        monkeypatch.setattr(mod, name, val)


def run(*argv) -> int:
    """Invoke the CLI exactly as the agent does."""
    return cli.main(list(argv))


def st() -> dict:
    return state.load_state()


def set_status(node: str, status: str, **kw) -> None:
    with state.state_txn() as s:
        ns = state.node_state(s, node)
        ns["status"] = status
        ns.update(kw)


def finish(node: str) -> None:
    set_status(node, "done")


# ------------------------------------------------------------------------------- graph loading

def test_graph_loads_and_topologically_orders(repo):
    g = gmod.load_graph()
    order = gmod.topo_order(g)
    assert order[0] == "A"
    assert order.index("A") < order.index("B") < order.index("F")
    assert len(order) == 7


def test_graph_rejects_cycles_and_unknown_dependencies(repo):
    p = repo / "graph" / "graph.yaml"
    p.write_text(MINI_GRAPH.replace("  - id: B\n    title: cpu one\n    spec: graph/nodes/B.md\n    depends_on: [A]",
                                    "  - id: B\n    title: cpu one\n    spec: graph/nodes/B.md\n    depends_on: [F]"),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        gmod.load_graph()
    p.write_text(MINI_GRAPH.replace("depends_on: [A]\n    resources: [cpu]\n    priority: 5",
                                    "depends_on: [ZZ]\n    resources: [cpu]\n    priority: 5"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown node"):
        gmod.load_graph()


def test_write_conflicts_are_component_wise(repo):
    g = gmod.load_graph()
    b, c, d = (gmod.node_by_id(g, x) for x in "BCD")
    assert gmod.conflicts(b, c)        # runs/b/** vs runs/b/sub/**
    assert not gmod.conflicts(b, d)
    # "runs/replay/**" must not collide with "runs/replay_N200/**"
    assert not gmod.conflicts({"writes": ["runs/replay/**"]}, {"writes": ["runs/replay_N200/**"]})
    assert gmod.conflicts({"writes": ["runs/replay/**"]}, {"writes": ["runs/**"]})


# -------------------------------------------------------------------------------- next / dispatch

def test_next_dispatches_root_then_respects_max_parallel(repo):
    assert run("next", "--json") == 0
    assert st()["nodes"]["A"]["status"] == "in_progress"
    assert st()["nodes"]["A"]["iterations"] == 1
    finish("A")
    assert run("next") == 0
    s = st()["nodes"]
    dispatched = [n for n in "BCDEG" if s.get(n, {}).get("status") == "in_progress"]
    assert len(dispatched) == 2, dispatched            # loop.max_parallel_nodes
    assert dispatched == ["B", "D"], dispatched        # C conflicts with B, E is the second GPU node


def test_next_never_dispatches_two_gpu_nodes(repo):
    finish("A")
    finish("B")
    run("next", "--max", "5")
    gpu = [n for n in "DE" if st()["nodes"].get(n, {}).get("status") == "in_progress"]
    assert gpu == ["D"]


def test_next_respects_the_gpu_lock_held_by_a_job(repo):
    finish("A")
    jobs.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    (jobs.JOBS_DIR / "held").mkdir()
    (jobs.JOBS_DIR / "held" / "cmd.json").write_text(json.dumps({"name": "held", "gpu": True}), encoding="utf-8")
    (jobs.JOBS_DIR / "held" / "heartbeat").write_text("now", encoding="utf-8")
    jobs.GPU_LOCK.write_text("held", encoding="utf-8")
    assert jobs.gpu_busy() == "held"
    run("next", "--max", "5")
    assert st()["nodes"].get("D", {}).get("status", "pending") == "pending"
    assert st()["nodes"]["B"]["status"] == "in_progress"


def test_node_with_a_running_job_is_waiting_not_dispatched(repo, capsys):
    finish("A")
    set_status("B", "in_progress", jobs=["bjob"])
    d = jobs.JOBS_DIR / "bjob"
    d.mkdir(parents=True)
    d.joinpath("cmd.json").write_text(json.dumps({"name": "bjob", "gpu": False}), encoding="utf-8")
    d.joinpath("heartbeat").write_text("now", encoding="utf-8")
    run("next", "--json")
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["waiting"] == [{"id": "B", "jobs": ["bjob"]}]
    assert "B" not in out["plan"]


def test_next_returns_zero_while_only_jobs_are_running(repo):
    """The normal state during the 30-hour teacher cache must not look like a failure."""
    for n in "ABCDEG":
        finish(n)
    set_status("F", "in_progress", jobs=["fjob"])
    d = jobs.JOBS_DIR / "fjob"
    d.mkdir(parents=True)
    d.joinpath("cmd.json").write_text(json.dumps({"name": "fjob"}), encoding="utf-8")
    d.joinpath("heartbeat").write_text("now", encoding="utf-8")
    assert run("next") == 0


def test_next_stalls_with_a_blocker_when_nothing_is_ready(repo, capsys):
    run("next")                       # A -> in_progress
    run("block", "A", "--reason", "needs a human")
    assert run("next") == cli.EXIT_IDLE
    assert st()["status"] == "STALLED"
    assert "No node is ready" in (state.BLOCKERS_PATH).read_text(encoding="utf-8")


def test_next_reports_graph_done(repo, capsys):
    for n in "ABCDEFG":
        finish(n)
    assert run("next") == 0
    assert st()["status"] == "DONE"
    assert "GRAPH DONE" in capsys.readouterr().out
    assert run("next") == 0          # idempotent


def test_blocked_optional_node_does_not_prevent_graph_done(repo, capsys):
    """Found by the end-to-end dry-run: one blocked optional node used to stall the loop one step
    before GRAPH DONE, with the finished submission packet sitting there unannounced."""
    for n in "ABCDEF":
        finish(n)
    run("block", "G", "--reason", "needs a human")
    assert run("next") == 0
    assert st()["status"] == "DONE"
    assert st()["nodes"]["G"]["status"] == "skipped"
    assert "GRAPH DONE" in capsys.readouterr().out


def test_blocked_required_node_still_stalls(repo):
    finish("A")
    run("block", "B", "--reason", "needs a human")
    for n in "CDEG":
        finish(n)
    assert run("next") == cli.EXIT_IDLE
    assert st()["nodes"]["B"]["status"] == "blocked"


def test_jobs_start_reports_a_runner_that_cannot_start(repo, monkeypatch, capsys):
    """`python -m graphctl.jobrunner` must be importable from the detached process; if it is not, the
    job would sit in `starting` and a 30-hour cache would never actually run."""
    real = jobs.child_env
    monkeypatch.setattr(jobs, "child_env", lambda extra=None: {k: v for k, v in real(extra).items()
                                                               if k != "PYTHONPATH"})
    rc = run("jobs", "start", "nostart", "--", sys.executable, "-c", "pass")
    assert rc == cli.EXIT_FAIL
    assert "never opened its log" in capsys.readouterr().out


def test_next_recycles_a_node_stuck_in_progress(repo):
    """Without this the loop can never terminate: an in_progress node keeps its slot forever."""
    run("next")
    set_status("A", "in_progress", touched_at="2020-01-01T00:00:00")
    run("next")
    assert st()["nodes"]["A"]["status"] == "in_progress"      # recycled to pending, then re-dispatched
    assert st()["nodes"]["A"]["iterations"] == 2
    for _ in range(4):
        set_status("A", "in_progress", touched_at="2020-01-01T00:00:00")
        run("next")
    assert st()["nodes"]["A"]["status"] == "blocked"


def test_next_force_node_refuses_done_and_blocked(repo):
    finish("A")
    assert run("next", "--node", "A") == cli.EXIT_FAIL
    run("block", "B", "--reason", "x")
    assert run("next", "--node", "B") == cli.EXIT_FAIL


def test_next_is_halted_by_the_halt_file(repo):
    state.HALT_PATH.write_text("stop", encoding="utf-8")
    assert run("next") == cli.EXIT_HALTED


# --------------------------------------------------------------------------------------- verify

def test_verify_step_types(repo):
    (repo / "runs" / "a").mkdir(parents=True)
    node = {"id": "X", "verify": [
        {"file_exists": "runs/a/ok.txt"},
        {"json_check": {"path": "runs/a/x.json", "expr": "d['ok'] is True and len(d['xs']) == 2"}},
        {"cmd": "python -c \"import sys;sys.exit(0)\"", "timeout": 60},
    ]}
    rep = verify.run_steps(node)
    assert rep["ok"] is False and rep["steps"][0]["ok"] is False
    (repo / "runs" / "a" / "ok.txt").write_text("x", encoding="utf-8")
    (repo / "runs" / "a" / "x.json").write_text(json.dumps({"ok": True, "xs": [1, 2]}), encoding="utf-8")
    rep = verify.run_steps(node)
    assert rep["ok"] is True, rep
    assert (verify.VERIFY_DIR / "X.json").exists()


def test_verify_file_exists_rejects_empty_files(repo):
    (repo / "runs" / "a").mkdir(parents=True)
    (repo / "runs" / "a" / "ok.txt").write_text("", encoding="utf-8")
    assert verify.run_steps({"id": "X", "verify": [{"file_exists": "runs/a/ok.txt"}]})["ok"] is False


def test_verify_json_check_survives_a_missing_or_broken_file(repo):
    rep = verify.run_steps({"id": "X", "verify": [{"json_check": {"path": "nope.json", "expr": "d['a']"}}]})
    assert rep["ok"] is False and "error" in rep["steps"][0]


def test_verify_cmd_failure_and_nonzero_exit(repo):
    rep = verify.run_steps({"id": "X", "verify": [{"cmd": "python -c \"import sys;sys.exit(3)\"", "timeout": 60}]})
    assert rep["ok"] is False and rep["steps"][0]["exit_code"] == 3


def test_verify_binds_a_leading_python_to_this_interpreter():
    assert verify._bind_interpreter("python -m pytest -q") == f'"{sys.executable}" -m pytest -q'
    assert verify._bind_interpreter("python3 scripts/x.py") == f'"{sys.executable}" scripts/x.py'
    assert verify._bind_interpreter("pythonesque --x") == "pythonesque --x"
    assert verify._bind_interpreter("latexmk -pdf") == "latexmk -pdf"


def test_verify_cmd_actually_runs_in_the_project_interpreter(repo):
    rep = verify.run_steps({"id": "X", "verify": [
        {"cmd": "python -c \"import sys;print(sys.executable)\"", "timeout": 60}]})
    assert sys.executable in rep["steps"][0]["tail"]


def test_verify_times_out_and_reports_it(repo):
    rep = verify.run_steps({"id": "X", "verify": [
        {"cmd": "python -c \"import time;time.sleep(30)\"", "timeout": 2}]})
    assert rep["ok"] is False and "TIMEOUT" in rep["steps"][0]["tail"]


def test_verify_marks_the_node_verified_and_unverifies_on_regression(repo):
    run("next")
    (repo / "runs" / "a").mkdir(parents=True)
    (repo / "runs" / "a" / "ok.txt").write_text("x", encoding="utf-8")
    assert run("verify", "A") == 0
    assert st()["nodes"]["A"]["status"] == "verified"
    (repo / "runs" / "a" / "ok.txt").unlink()
    assert run("verify", "A") == cli.EXIT_FAIL
    assert st()["nodes"]["A"]["status"] == "in_progress"


# ------------------------------------------------------------------------------------ done / review

def test_done_is_refused_without_verification(repo):
    run("next")
    assert run("done", "A", "--note", "nope") == cli.EXIT_FAIL
    assert st()["nodes"]["A"]["status"] == "in_progress"


def test_done_is_refused_without_a_fresh_context_review(repo, capsys):
    for n, d in (("A", "a"), ("B", "b"), ("F", "f")):
        (repo / "runs" / d).mkdir(parents=True)
        (repo / "runs" / d / "ok.txt").write_text("x", encoding="utf-8")
    finish("A")
    finish("B")
    set_status("F", "in_progress")
    assert run("done", "F") == cli.EXIT_FAIL
    assert "fresh-context review" in capsys.readouterr().out
    verify.VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    (verify.VERIFY_DIR / "F.review.json").write_text(json.dumps({"verdict": "fail", "issues": ["x"]}), encoding="utf-8")
    assert run("done", "F") == cli.EXIT_FAIL
    (verify.VERIFY_DIR / "F.review.json").write_text(json.dumps({"verdict": "pass", "summary": "ok"}), encoding="utf-8")
    assert run("done", "F") == 0
    assert st()["nodes"]["F"]["status"] == "done"
    assert st()["nodes"]["F"]["review"]["verdict"] == "pass"


def test_record_review_stores_a_readonly_verifiers_verdict(repo, monkeypatch):
    (repo / "runs" / "f").mkdir(parents=True)
    (repo / "runs" / "f" / "ok.txt").write_text("x", encoding="utf-8")
    payload = json.dumps({"verdict": "pass", "summary": "re-derived", "issues": []})
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("﻿" + payload))
    assert run("record-review", "F", "--stdin") == 0
    d = json.loads((verify.VERIFY_DIR / "F.review.json").read_text(encoding="utf-8"))
    assert d["verdict"] == "pass" and d["node"] == "F" and d["recorded_by"]
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("not json"))
    assert run("record-review", "F", "--stdin") == cli.EXIT_FAIL


def test_done_refuses_while_a_job_of_that_node_runs(repo):
    (repo / "runs" / "a").mkdir(parents=True)
    (repo / "runs" / "a" / "ok.txt").write_text("x", encoding="utf-8")
    set_status("A", "in_progress", jobs=["aj"])
    d = jobs.JOBS_DIR / "aj"
    d.mkdir(parents=True)
    d.joinpath("cmd.json").write_text(json.dumps({"name": "aj"}), encoding="utf-8")
    d.joinpath("heartbeat").write_text("now", encoding="utf-8")
    assert run("done", "A") == cli.EXIT_FAIL
    assert run("done", "A", "--force") == 0


# --------------------------------------------------------------------------------- fail / block / skip

def test_fail_retries_then_blocks_after_max_iterations(repo):
    max_it = 3
    for i in range(1, max_it + 1):
        run("next")
        assert st()["nodes"]["A"]["iterations"] == i
        run("fail", "A", "--reason", f"try {i}")
        expected = "pending" if i < max_it else "blocked"
        assert st()["nodes"]["A"]["status"] == expected, (i, st()["nodes"]["A"])
    assert "Exceeded 3 iterations" in state.BLOCKERS_PATH.read_text(encoding="utf-8")


def test_fail_then_retry_then_success(repo):
    run("next")
    run("fail", "A", "--reason", "one")
    run("next")
    run("fail", "A", "--reason", "two")
    run("next")
    (repo / "runs" / "a").mkdir(parents=True)
    (repo / "runs" / "a" / "ok.txt").write_text("x", encoding="utf-8")
    assert run("done", "A", "--note", "third time") == 0
    assert st()["nodes"]["A"]["iterations"] == 3


def test_block_and_unblock(repo):
    run("next")
    assert run("block", "A", "--reason", "human only", "--human", "do X") == 0
    assert st()["nodes"]["A"]["status"] == "blocked"
    assert "do X" in state.BLOCKERS_PATH.read_text(encoding="utf-8")
    assert run("unblock", "A") == 0
    assert st()["nodes"]["A"]["status"] == "pending" and st()["nodes"]["A"]["iterations"] == 0


def test_skip_only_works_on_optional_nodes(repo):
    assert run("skip", "A", "--reason", "x") == cli.EXIT_FAIL
    assert run("skip", "G", "--reason", "out of time") == 0
    assert st()["nodes"]["G"]["status"] == "skipped"


def test_unknown_node_is_a_clean_error_not_a_phantom(repo, capsys):
    assert run("fail", "ZZ", "--reason", "typo") == cli.EXIT_FAIL
    assert "unknown node" in capsys.readouterr().err
    assert "ZZ" not in st()["nodes"]


# --------------------------------------------------------------------------------------- ladder

def test_ladder_evaluates_and_applies(repo, capsys):
    assert run("ladder") == 0
    out = capsys.readouterr().out
    assert "L_date" in out and "L_json" not in out          # the json rule's file does not exist yet
    (repo / "runs" / "pilot.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
    assert run("ladder", "--apply") == 0
    cfg = (repo / "configs" / "main.yaml").read_text(encoding="utf-8")
    assert "n_train: 2000" in cfg
    assert set(st()["ladder_applied"]) == {"L_json", "L_date"}
    run("ladder")
    assert "no rule fires now" in capsys.readouterr().out   # applied rules never fire twice


def test_ladder_survives_a_missing_target_file(repo, capsys):
    (repo / "runs" / "pilot.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
    (repo / "configs" / "main.yaml").unlink()
    assert run("ladder", "--apply") == cli.EXIT_FAIL        # reported, not raised
    assert "NOT APPLIED" in capsys.readouterr().out
    assert "L_json" not in st().get("ladder_applied", [])


def test_ladder_never_fires_an_empty_condition():
    assert cli._ladder_fires({}) is False


# ------------------------------------------------------------------------------------- followup

def test_followup_normal_case_prints_a_message(repo, capsys):
    run("next")
    assert run("followup") == 0
    assert "graph-loop" in capsys.readouterr().out


def test_followup_is_silent_when_done_or_halted(repo, capsys):
    with state.state_txn() as s:
        s["status"] = "DONE"
    assert run("followup") == 0
    assert capsys.readouterr().out.strip() == ""
    with state.state_txn() as s:
        s["status"] = "RUNNING"
    state.HALT_PATH.write_text("x", encoding="utf-8")
    assert run("followup") == 0
    assert capsys.readouterr().out.strip() == ""


def test_followup_self_pauses_after_a_no_progress_streak(repo, capsys):
    run("next")
    capsys.readouterr()
    maxs = 2  # loop.max_followups_without_progress in MINI_GRAPH
    for _ in range(maxs + 1):
        run("followup")
        assert capsys.readouterr().out.strip()
    run("followup")                                  # streak > maxs -> self-pause
    stall_msg = capsys.readouterr().out
    assert "STALLED" in stall_msg and "unblock" in stall_msg   # one loud last message ...
    assert st()["status"] == "STALLED"
    assert (state.ROOT / "NEEDS_YOU.txt").exists()
    assert "paused itself" in state.BLOCKERS_PATH.read_text(encoding="utf-8")
    run("followup")
    assert capsys.readouterr().out.strip() == ""     # ... then silence
    run("resume")
    capsys.readouterr()
    run("followup")
    assert capsys.readouterr().out.strip()           # resume revives the loop


def test_followup_streak_resets_on_real_progress(repo, capsys):
    run("next")
    run("followup")
    run("followup")
    assert st()["followups"]["streak"] == 1
    run("ledger", "DECIDED: something")
    run("followup")
    assert st()["followups"]["streak"] == 0


def test_followup_never_raises_even_with_a_corrupt_state(repo, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_followup", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert run("followup") == 0
    assert "Continue autonomously" in capsys.readouterr().out   # the loop must not die silently


def test_followup_points_at_a_running_job_when_no_node_is_active(repo, capsys):
    d = jobs.JOBS_DIR / "long"
    d.mkdir(parents=True)
    d.joinpath("cmd.json").write_text(json.dumps({"name": "long"}), encoding="utf-8")
    d.joinpath("heartbeat").write_text("now", encoding="utf-8")
    run("followup")
    assert "jobs wait long" in capsys.readouterr().out


# ----------------------------------------------------------------------------------------- jobs

def test_job_end_to_end(repo):
    info = jobs.start("j1", [sys.executable, "-c", "import time;print('hello');time.sleep(0.6);print('bye')"])
    assert info["name"] == "j1"
    assert jobs.status("j1")["state"] in ("starting", "running")
    out = jobs.wait("j1", timeout=30, poll=0.1)
    assert out["state"] == "finished" and out["exit_code"] == 0
    tail = jobs.tail("j1")
    assert "hello" in tail and "bye" in tail


def test_job_failed_start_is_detected(repo):
    jobs.start("bad", ["this-binary-does-not-exist-xyz"])
    out = jobs.wait("bad", timeout=30, poll=0.1)
    assert out["state"] == "failed" and out["exit_code"] == 127
    assert "FAILED TO START" in jobs.tail("bad")


def test_job_nonzero_exit_is_failed(repo):
    jobs.start("boom", [sys.executable, "-c", "import sys;sys.exit(4)"])
    out = jobs.wait("boom", timeout=30, poll=0.1)
    assert out["state"] == "failed" and out["exit_code"] == 4


def test_gpu_lock_is_held_while_starting_and_released_at_exit(repo):
    jobs.start("g1", [sys.executable, "-c", "import time;time.sleep(0.5)"], gpu=True)
    assert jobs.gpu_busy() == "g1"                       # must hold even before the first heartbeat
    with pytest.raises(RuntimeError, match="GPU is held"):
        jobs.start("g2", [sys.executable, "-c", "pass"], gpu=True)
    jobs.wait("g1", timeout=30, poll=0.1)
    assert jobs.gpu_busy() is None
    assert not jobs.GPU_LOCK.exists()


def test_stale_gpu_lock_is_cleared(repo):
    jobs.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jobs.GPU_LOCK.write_text("ghost", encoding="utf-8")
    assert jobs.gpu_busy() is None                       # unknown holder -> lock released
    d = jobs.JOBS_DIR / "dead"
    d.mkdir()
    d.joinpath("cmd.json").write_text(json.dumps({"name": "dead", "gpu": True}), encoding="utf-8")
    hb = d / "heartbeat"
    hb.write_text("old", encoding="utf-8")
    old = time.time() - (jobs.HEARTBEAT_STALE_S + 60)
    os.utime(hb, (old, old))
    jobs.GPU_LOCK.write_text("dead", encoding="utf-8")
    assert jobs.status("dead")["state"] == "stale"
    assert jobs.gpu_busy() is None and not jobs.GPU_LOCK.exists()


def test_job_that_never_produced_a_heartbeat_goes_stale(repo):
    d = jobs.JOBS_DIR / "ghost"
    d.mkdir(parents=True)
    c = d / "cmd.json"
    c.write_text(json.dumps({"name": "ghost"}), encoding="utf-8")
    assert jobs.status("ghost")["state"] == "starting"
    old = time.time() - (jobs.STARTING_GRACE_S + 30)
    os.utime(c, (old, old))
    assert jobs.status("ghost")["state"] == "stale"


def test_job_kill_stops_a_long_process_and_frees_the_gpu(repo):
    jobs.start("kme", [sys.executable, "-c", "import time;time.sleep(120)"], gpu=True)
    for _ in range(60):
        if jobs.status("kme")["state"] == "running":
            break
        time.sleep(0.1)
    assert jobs.kill("kme")["killed"] is True
    assert jobs.status("kme")["state"] == "failed"
    assert jobs.gpu_busy() is None
    time.sleep(0.6)
    assert jobs.status("kme")["state"] == "failed"   # the runner must not resurrect it


def test_job_tail_reads_only_the_tail_of_a_huge_log(repo):
    d = jobs.JOBS_DIR / "big"
    d.mkdir(parents=True)
    d.joinpath("cmd.json").write_text(json.dumps({"name": "big"}), encoding="utf-8")
    with open(d / "log.txt", "w", encoding="utf-8") as f:
        for i in range(60000):
            f.write(f"line {i} " + "x" * 40 + "\n")
    out = jobs.tail("big", lines=3, max_bytes=8192)
    assert out.count("\n") == 2 and out.strip().endswith("x" * 40)
    assert "line 59999" in out


def test_job_names_are_validated(repo):
    for bad in ("", "..", "a/b", "a b", "a\\b"):
        with pytest.raises(ValueError):
            jobs.job_dir(bad)
    assert jobs.job_dir("teacher_qwen3-4b.cal").name == "teacher_qwen3-4b.cal"


def test_normalize_cmd_binds_python_to_this_interpreter():
    assert jobs.normalize_cmd(["python", "-V"])[0] == sys.executable
    assert jobs.normalize_cmd(["python3.exe", "-V"])[0] == sys.executable
    assert jobs.normalize_cmd(["latexmk", "-pdf"])[0] == "latexmk"


def test_jobs_start_through_the_cli_attaches_the_node_and_marks_it_in_progress(repo, capsys):
    rc = run("jobs", "start", "cj", "--node", "A", "--", sys.executable, "-c", "import time;time.sleep(0.4)")
    assert rc == 0
    ns = st()["nodes"]["A"]
    assert ns["jobs"] == ["cj"] and ns["status"] == "in_progress" and ns["iterations"] == 1
    assert run("jobs", "wait", "cj", "--timeout", "30", "--poll", "0.1") == 0


def test_jobs_start_rejects_an_unknown_node(repo):
    assert run("jobs", "start", "x", "--node", "ZZ", "--", sys.executable, "-c", "pass") == cli.EXIT_FAIL


def test_jobs_wait_exit_codes(repo):
    assert run("jobs", "wait", "nosuchjob", "--timeout", "1", "--poll", "0.1") == cli.EXIT_FAIL
    jobs.start("w1", [sys.executable, "-c", "import time;time.sleep(20)"])
    assert run("jobs", "wait", "w1", "--timeout", "0.5", "--poll", "0.1") == cli.EXIT_IDLE
    jobs.kill("w1")


# ----------------------------------------------------------------------------------------- lock

def test_file_lock_serialises_two_writers(tmp_path):
    lock = tmp_path / "l.lock"
    counter = tmp_path / "n.txt"
    counter.write_text("0", encoding="utf-8")
    errors = []

    def worker():
        for _ in range(40):
            try:
                with file_lock(lock, timeout=20) as ok:
                    assert ok
                    n = int(counter.read_text(encoding="utf-8"))
                    time.sleep(0.0005)
                    counter.write_text(str(n + 1), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    assert counter.read_text(encoding="utf-8") == "160"
    assert not lock.exists()


def test_file_lock_breaks_a_stale_lock(tmp_path):
    lock = tmp_path / "l.lock"
    lock.write_text("99999:deadbeef", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    with file_lock(lock, timeout=2) as ok:
        assert ok is True
    assert not lock.exists()


def test_file_lock_yields_false_rather_than_blocking_forever(tmp_path):
    lock = tmp_path / "l.lock"
    lock.write_text("1:fresh", encoding="utf-8")
    t0 = time.time()
    with file_lock(lock, timeout=0.3, poll=0.05) as ok:
        assert ok is False
    assert time.time() - t0 < 5
    assert lock.exists()          # someone else's lock must survive our giving up


def test_state_txn_round_trips_and_recovers_from_corruption(repo, capsys):
    with state.state_txn() as s:
        state.node_state(s, "A")["status"] = "in_progress"
    assert st()["nodes"]["A"]["status"] == "in_progress"
    state.STATE_PATH.write_text("{not json", encoding="utf-8")
    s = state.load_state()
    assert s["nodes"] == {} and s["status"] == "RUNNING"
    assert list(state.GRAPH_DIR.glob("state.corrupt-*.json"))


def test_progress_hash_ignores_non_progress_ledger_noise(repo):
    with state.state_txn() as s:
        state.node_state(s, "A")
    h0 = state.progress_hash(st())
    with open(state.LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write("- 2026-09-21T00:00:00 SUBAGENT STOP status=completed type=verifier\n")
    assert state.progress_hash(st()) == h0
    state.append_ledger("DONE A: real progress")
    assert state.progress_hash(st()) != h0


# -------------------------------------------------------------------------------- status / render

def test_status_and_zh_dashboard(repo, capsys):
    run("next")
    assert run("status", "--zh", "-v") == 0
    out = (state.GRAPH_DIR / "STATUS_zh.md").read_text(encoding="utf-8")
    assert "进度看板" in out and "A — root" in out
    assert "graph status: RUNNING" in capsys.readouterr().out


def test_brief_contains_the_spec_and_the_verification_list(repo, capsys):
    run("next")
    run("brief", "A")
    out = capsys.readouterr().out
    assert "Do the thing." in out and "runs/a/ok.txt" in out and "NODE A" in out


def test_reset_clears_state(repo):
    run("next")
    assert state.STATE_PATH.exists()
    assert run("reset") == cli.EXIT_FAIL
    assert run("reset", "--yes") == 0
    assert not state.STATE_PATH.exists()


# ----------------------------------------------------------------------------------------- hooks

def _hook_repo(tmp_path: Path) -> Path:
    root = tmp_path / "hookrepo"
    root.mkdir()
    _write_repo(root)
    shutil.copytree(REPO / "graphctl", root / "graphctl",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(HOOKS, root / ".cursor" / "hooks", ignore=shutil.ignore_patterns("__pycache__"))
    return root


def _run_hook(root: Path, name: str, payload: dict, bom: bool = True) -> dict:
    raw = ("﻿" if bom else "") + json.dumps(payload, ensure_ascii=False)
    p = subprocess.run([sys.executable, "-u", str(root / ".cursor" / "hooks" / name)],
                       input=raw.encode("utf-8"), cwd=str(root), capture_output=True, timeout=120)
    out = p.stdout.decode("utf-8", errors="replace").strip()
    assert out, f"{name} printed nothing (stderr: {p.stderr.decode('utf-8', 'replace')[:800]})"
    return json.loads(out.splitlines()[-1])


@pytest.mark.parametrize("bom", [True, False])
def test_stop_hook_returns_a_followup_message(tmp_path, bom):
    root = _hook_repo(tmp_path)
    (root / "graph" / "AUTORUN").write_text("on\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "graphctl", "next"], cwd=str(root), capture_output=True, timeout=120)
    got = _run_hook(root, "stop_followup.py", {"status": "completed", "loop_count": 1}, bom=bom)
    assert "followup_message" in got and "graph-loop" in got["followup_message"]


def test_stop_hook_respects_aborted_and_halt(tmp_path):
    root = _hook_repo(tmp_path)
    assert _run_hook(root, "stop_followup.py", {"status": "aborted"}) == {}
    (root / "graph" / "HALT").write_text("x", encoding="utf-8")
    assert _run_hook(root, "stop_followup.py", {"status": "completed"}) == {}


def test_stop_hook_is_off_unless_autorun(tmp_path):
    root = _hook_repo(tmp_path)
    subprocess.run([sys.executable, "-m", "graphctl", "next"], cwd=str(root), capture_output=True, timeout=120)
    assert _run_hook(root, "stop_followup.py", {"status": "completed"}) == {}


def test_stop_hook_prefixes_errors(tmp_path):
    root = _hook_repo(tmp_path)
    (root / "graph" / "AUTORUN").write_text("on\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "graphctl", "next"], cwd=str(root), capture_output=True, timeout=120)
    got = _run_hook(root, "stop_followup.py", {"status": "error"})
    assert got["followup_message"].startswith("[graph-loop] The previous turn ended with an error")


def test_stop_hook_keeps_the_loop_alive_when_the_graph_file_is_broken(tmp_path):
    """A broken graph.yaml must produce a diagnostic follow-up, not a silent end of the five-week run."""
    root = _hook_repo(tmp_path)
    (root / "graph" / "AUTORUN").write_text("on\n", encoding="utf-8")
    (root / "graph" / "graph.yaml").write_text("nodes: []\n", encoding="utf-8")
    got = _run_hook(root, "stop_followup.py", {"status": "completed"})
    assert "Continue autonomously" in got.get("followup_message", "")


def test_stop_hook_emits_empty_json_when_graphctl_cannot_run_at_all(tmp_path):
    root = _hook_repo(tmp_path)
    shutil.rmtree(root / "graphctl")
    assert _run_hook(root, "stop_followup.py", {"status": "completed"}) == {}


def test_session_start_hook_injects_context(tmp_path):
    root = _hook_repo(tmp_path)
    got = _run_hook(root, "session_start.py", {"session_id": "x"})
    assert "additional_context" in got and "graphctl" in got["additional_context"]


def test_subagent_stop_hook_is_quiet_and_does_not_pollute_the_ledger(tmp_path):
    root = _hook_repo(tmp_path)
    assert _run_hook(root, "subagent_stop.py", {"status": "completed", "subagent_type": "verifier"}) == {}
    assert not (root / "graph" / "ledger.md").exists()
    assert "verifier" in (root / "runs" / "subagents.log").read_text(encoding="utf-8")


GUARD_TABLE = [
    ("python -m pytest -q tests", "allow"),
    ("./.venv/Scripts/python.exe -m graphctl next", "allow"),
    ("pip install torch --index-url https://download.pytorch.org/whl/cu128", "allow"),
    ("git add -A; git commit -m 'N02: theory'", "allow"),
    ("latexmk -pdf main.tex", "allow"),
    ("Remove-Item -Recurse runs/smoke", "allow"),
    ("rm -rf /", "deny"),
    ("rm -rf ~/", "deny"),
    ("rm -rf $HOME", "deny"),
    ("rm -fr /*", "deny"),
    ("rm -rf data/teacher", "deny"),
    ("Remove-Item -Recurse -Force data\\raw", "deny"),
    ("del models\\heads\\qwen3-4b", "deny"),
    ("git push --force origin main", "deny"),
    ("git reset --hard HEAD~1", "deny"),
    ("git clean -fd", "deny"),
    ("sudo apt install texlive", "deny"),
    ("shutdown /r /t 0", "deny"),
    ("python upload_to_openreview.py --submit", "deny"),
    ("pip uninstall torch -y", "deny"),
    ("python -m graphctl reset --yes", "deny"),
    ("python -m graphctl halt", "deny"),
]


@pytest.mark.parametrize("cmd,expected", GUARD_TABLE)
def test_guard_shell_allow_deny_table(cmd, expected):
    raw = "﻿" + json.dumps({"command": cmd, "cwd": str(REPO), "sandbox": "danger-full-access"})
    p = subprocess.run([sys.executable, "-u", str(HOOKS / "guard_shell.py")],
                       input=raw.encode("utf-8"), capture_output=True, timeout=60)
    got = json.loads(p.stdout.decode("utf-8").strip().splitlines()[-1])
    assert got["permission"] == expected, (cmd, got)
    if expected == "deny":
        assert got.get("user_message") and got.get("agent_message")


def test_hooks_emit_empty_json_for_an_empty_or_broken_payload():
    for name in ("guard_shell.py", "subagent_stop.py"):
        p = subprocess.run([sys.executable, "-u", str(HOOKS / name)], input=b"", capture_output=True, timeout=60)
        assert json.loads(p.stdout.decode("utf-8").strip()) is not None


BASHISM_FILES = ["AGENTS.md", "START_HERE.md"] + \
    [str(p.relative_to(REPO)) for p in sorted((REPO / "graph" / "nodes").glob("*.md"))] + \
    [str(p.relative_to(REPO)) for p in sorted((REPO / ".cursor").rglob("*.md"))] + \
    [str(p.relative_to(REPO)) for p in sorted((REPO / ".cursor" / "rules").glob("*.mdc"))]


def test_no_bash_isms_in_what_the_agent_is_told_to_run():
    """The Windows terminal is PowerShell 5.1: no `&&`, no `$(...)`, no `export`."""
    import re
    bad = []
    for rel in BASHISM_FILES:
        for i, line in enumerate((REPO / rel).read_text(encoding="utf-8").splitlines(), 1):
            if "no `&&`" in line or "used `&&`" in line or "has no `&&`" in line:
                continue
            if re.search(r"&&|\$\((?!\.\.\.)|(^|\s)export\s+\w+=", line):
                bad.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not bad, "bash-isms in agent-facing instructions:\n" + "\n".join(bad)


def test_cursor_has_no_unused_or_unverified_config_files():
    """sandbox.json / environment.json were removed on purpose (docs/CURSOR_SETUP.md section 6)."""
    assert not (REPO / ".cursor" / "sandbox.json").exists()
    assert not (REPO / ".cursor" / "environment.json").exists()


def test_hooks_json_is_well_formed_and_points_at_existing_scripts():
    cfg = json.loads((REPO / ".cursor" / "hooks.json").read_text(encoding="utf-8"))
    assert cfg["version"] == 1
    for event, entries in cfg["hooks"].items():
        for e in entries:
            assert e["command"].startswith("./.venv/Scripts/python.exe -u .cursor/hooks/")
            script = e["command"].split()[-1]
            assert (REPO / script).exists(), script
    assert cfg["hooks"]["stop"][0]["loop_limit"] is None


def test_agent_and_skill_frontmatter_is_valid():
    import yaml as _yaml
    for f in sorted((REPO / ".cursor" / "agents").glob("*.md")):
        head = f.read_text(encoding="utf-8").split("---")[1]
        fm = _yaml.safe_load(head)
        assert fm["name"] == f.stem, f
        assert isinstance(fm.get("description"), str) and len(fm["description"]) > 40, f
        assert fm.get("model") == "inherit", f
        assert set(fm) <= {"name", "description", "model", "readonly", "is_background"}, f
    for f in sorted((REPO / ".cursor" / "skills").glob("*/SKILL.md")):
        fm = _yaml.safe_load(f.read_text(encoding="utf-8").split("---")[1])
        assert fm["name"] == f.parent.name, f
        assert set(fm) <= {"name", "description"}, f
        assert isinstance(fm.get("description"), str) and len(fm["description"]) > 30, f


def test_every_source_file_opens_text_as_utf8():
    """Windows defaults to the ANSI code page for text I/O; a missing encoding= corrupts Chinese docs."""
    import re
    bad = []
    for f in list((REPO / "graphctl").glob("*.py")) + list((REPO / ".cursor" / "hooks").glob("*.py")) \
            + [REPO / "bootstrap.py", REPO / "scripts" / "run_headless.py"]:
        src = f.read_text(encoding="utf-8")
        for m in re.finditer(r"(?<!os\.)\b(open|read_text|write_text)\(", src):
            line = src[m.start():src.find("\n", m.start())]
            if '"rb"' in line or '"wb"' in line or "encoding=" in line:
                continue
            # multi-line call: look ahead to the closing paren
            chunk = src[m.start():m.start() + 400]
            head = chunk[:chunk.find(")") + 1] if ")" in chunk else chunk
            if "encoding=" in head or '"rb"' in head or '"wb"' in head:
                continue
            bad.append(f"{f.relative_to(REPO)}: {line.strip()[:90]}")
    assert not bad, "text I/O without encoding='utf-8':\n" + "\n".join(bad)
