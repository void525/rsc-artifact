"""End-to-end smoke run of the whole pipeline on synthetic data (CPU, ~1-3 min).

    python scripts/smoke_pipeline.py            # all stages + paper preview build
    python scripts/smoke_pipeline.py --no-latex # skip the LaTeX build

Everything is written under runs/smoke/.  This is the closed-loop check that
the code paths used by the real experiments work together; it says nothing
about retrieval quality.  Exit code 0 = every stage and check passed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STAGES = [
    ["scripts/01_download_data.py"],
    ["scripts/02_build_splits.py"],
    ["scripts/03_build_candidates.py"],
    ["scripts/04_cheap_features.py"],
    ["scripts/05_teacher_scores.py", "--teacher", "fake"],
    ["scripts/05_teacher_scores.py", "--teacher", "fake2"],
    ["scripts/05_teacher_scores.py", "--teacher", "fake", "--pilot"],
    ["scripts/05_teacher_scores.py", "--teacher", "fake", "--audit"],
    ["scripts/06_train_heads.py", "--teacher", "fake"],
    ["scripts/06_train_heads.py", "--teacher", "fake2"],
    ["scripts/06b_distill_cheap.py"],
    ["scripts/07_freeze_calibrate.py"],
    ["scripts/08_replay_eval.py"],
    ["scripts/09_system_latency.py"],
    ["scripts/10_stats.py"],
    ["scripts/11_make_tables_figs.py"],
    ["scripts/12_paper_numbers.py"],
]


def run(cmd, timeout=1800):
    t0 = time.time()
    p = subprocess.run([PY] + cmd + ["--synthetic"], cwd=str(ROOT), capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    ok = p.returncode == 0
    print(f"[{'ok ' if ok else 'FAIL'}] {' '.join(cmd)}  ({dt:.1f}s)")
    if not ok:
        print((p.stdout + p.stderr)[-4000:])
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-latex", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete runs/smoke first")
    args = ap.parse_args()
    smoke = ROOT / "runs" / "smoke"
    if args.fresh and smoke.exists():
        shutil.rmtree(smoke)
    smoke.mkdir(parents=True, exist_ok=True)
    for st in STAGES:
        if not run(st):
            print("SMOKE FAILED at", st)
            return 1
    checks = [smoke / "runs" / "stats" / "summary.json", smoke / "paper_preview" / "numbers.tex", smoke / "paper_preview" / "tables" / "tab_main.tex", smoke / "paper_preview" / "figures" / "fig_risk_cost.pdf"]
    for c in checks:
        if not c.exists():
            print("missing", c)
            return 1
    if not args.no_latex:
        build = smoke / "paper_build"
        if build.exists():
            shutil.rmtree(build)
        shutil.copytree(ROOT / "paper", build, ignore=shutil.ignore_patterns("*.pdf", "*.aux", "*.log", "*.bbl", "*.blg", "*.fdb_latexmk", "*.fls", "*.out"))
        shutil.copy(smoke / "paper_preview" / "numbers.tex", build / "numbers.tex")
        for sub in ("tables", "figures"):
            for f in (smoke / "paper_preview" / sub).glob("*"):
                shutil.copy(f, build / sub / f.name)
        (build / "SMOKE_PREVIEW").write_text("synthetic numbers -- not the manuscript\n")
        p = subprocess.run([PY, "tools/check_paper.py", "--build", "--allow-placeholders", "--preview"], cwd=str(build), capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(p.stdout[-3000:])
        if p.returncode != 0:
            print(p.stderr[-3000:])
            print("SMOKE FAILED: paper preview build")
            return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
