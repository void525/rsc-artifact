"""Merge one simulated review round into merged.md and update runs/reviews/converged.json.

    python paper/tools/merge_reviews.py runs/reviews/round3 [--json] [--converged FILE]

Inputs in <round-dir>:
  r*.json   one per reviewer: {"score": 1-5, "confidence": 1-5, "summary": str,
            "strengths": [...], "weaknesses": [...], "blocking": [...], "minor": [...],
            "questions": [...]}
  audit.json  the numbers/compliance auditor: {"number_mismatches": [...],
            "anonymity_violations": [...], "format_issues": [...], "policy_issues": [...]}

Outputs:
  <round-dir>/merged.md          scores, blocking issues grouped, action list by severity
  runs/reviews/converged.json    {"round": k, "scores": [...], "blocking_issues": n,
                                  "converged": bool, ...}

Convergence rule (graph/nodes/N15_review_loop.md): converged iff no blocking issue in any
review AND every score >= 4 AND the auditor is clean; or round >= 5 with zero blocking
issues (review blockers plus auditor number mismatches and anonymity violations).

Stdlib only, utf-8 everywhere, runs on Windows and Linux.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

AUDIT_FIELDS = ("number_mismatches", "anonymity_violations", "format_issues", "policy_issues")
AUDIT_SEVERITY = {"number_mismatches": 0, "anonymity_violations": 0, "format_issues": 2, "policy_issues": 2}
SEVERITY_NAME = {0: "blocking", 1: "blocking", 2: "major", 3: "minor", 4: "question"}


def as_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, dict):
        return [f"{k}: {json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x}" for k, x in v.items()]
    return [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in v]


def round_number(d: Path) -> int:
    m = re.search(r"(\d+)\s*$", d.name)
    return int(m.group(1)) if m else 0


def load_reviews(d: Path) -> Tuple[List[dict], List[str]]:
    problems: List[str] = []
    out: List[dict] = []
    for f in sorted(d.glob("r*.json")):
        if f.name == "response.json":
            continue
        try:
            js = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:  # noqa: BLE001
            problems.append(f"{f.name}: unreadable ({e})")
            continue
        if not isinstance(js, dict):
            problems.append(f"{f.name}: not a JSON object")
            continue
        js["_file"] = f.name
        score = js.get("score")
        if not isinstance(score, (int, float)):
            problems.append(f"{f.name}: missing or non-numeric 'score'")
        out.append(js)
    return out, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="merge one review round")
    ap.add_argument("round_dir")
    ap.add_argument("--converged", default=None, help="path of converged.json (default: <round-dir>/../converged.json)")
    ap.add_argument("--json", action="store_true", help="print only the JSON summary")
    a = ap.parse_args(argv)

    d = Path(a.round_dir)
    if not d.is_dir():
        print(json.dumps({"error": f"no such round directory: {d}"}))
        return 2
    k = round_number(d)
    reviews, problems = load_reviews(d)

    audit_file = d / "audit.json"
    audit: Dict[str, List[str]] = {}
    audit_present = audit_file.exists()
    if audit_present:
        try:
            raw = json.loads(audit_file.read_text(encoding="utf-8"))
            audit = {f: as_list(raw.get(f)) for f in AUDIT_FIELDS}
            extra = [kk for kk in raw if kk not in AUDIT_FIELDS and not kk.startswith("_")]
            if extra:
                problems.append(f"audit.json: ignored extra fields {extra}")
        except (ValueError, OSError) as e:  # noqa: BLE001
            problems.append(f"audit.json: unreadable ({e})")
            audit_present = False
    if not audit_present:
        problems.append("audit.json is missing: the round cannot be declared converged")
        audit = {f: [] for f in AUDIT_FIELDS}

    scores = [r["score"] for r in reviews if isinstance(r.get("score"), (int, float))]
    review_blocking = sum(len(as_list(r.get("blocking"))) for r in reviews)
    hard_audit = len(audit["number_mismatches"]) + len(audit["anonymity_violations"])
    audit_total = sum(len(audit[f]) for f in AUDIT_FIELDS)
    audit_clean = audit_present and audit_total == 0
    blocking_issues = review_blocking + hard_audit

    converged = bool(
        (reviews and review_blocking == 0 and scores and len(scores) == len(reviews)
         and all(s >= 4 for s in scores) and audit_clean)
        or (k >= 5 and blocking_issues == 0 and audit_present)
    )

    # ---- action list, most severe first ------------------------------------
    actions: List[Tuple[int, str, str]] = []           # (severity, source, text)
    for r in reviews:
        who = r.get("reviewer") or r["_file"].replace(".json", "")
        for t in as_list(r.get("blocking")):
            actions.append((0, who, t))
    for f in AUDIT_FIELDS:
        for t in audit[f]:
            actions.append((AUDIT_SEVERITY[f], f"audit/{f}", t))
    for r in reviews:
        who = r.get("reviewer") or r["_file"].replace(".json", "")
        for t in as_list(r.get("weaknesses")):
            actions.append((2, who, t))
        for t in as_list(r.get("minor")):
            actions.append((3, who, t))
        for t in as_list(r.get("questions")):
            actions.append((4, who, t))
    actions.sort(key=lambda x: x[0])

    # ---- merged.md ----------------------------------------------------------
    L: List[str] = []
    L.append(f"# Review round {k} -- merged verdict")
    L.append("")
    L.append(f"Merged {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from `{d.as_posix()}` "
             f"({len(reviews)} review(s){', auditor present' if audit_present else ', AUDITOR MISSING'}).")
    L.append("")
    L.append(f"**Converged: {'YES' if converged else 'NO'}** "
             f"(blocking issues: {blocking_issues}; scores: {', '.join(str(s) for s in scores) or 'none'}; "
             f"auditor {'clean' if audit_clean else 'not clean' if audit_present else 'missing'}).")
    L.append("")
    L.append("## Scores")
    L.append("")
    L.append("| review | score | confidence | blocking | summary |")
    L.append("|---|---|---|---|---|")
    for r in reviews:
        s = r.get("score", "?")
        c = r.get("confidence", "?")
        nb = len(as_list(r.get("blocking")))
        summ = " ".join(str(r.get("summary", "")).split())[:160]
        L.append(f"| {r.get('reviewer') or r['_file']} | {s} | {c} | {nb} | {summ} |")
    if scores:
        L.append("")
        L.append(f"Mean score {sum(scores) / len(scores):.2f}, min {min(scores)}, max {max(scores)}.")
    L.append("")
    L.append("## Blocking issues")
    L.append("")
    blocking_any = False
    for r in reviews:
        items = as_list(r.get("blocking"))
        if items:
            blocking_any = True
            L.append(f"**{r.get('reviewer') or r['_file']}**")
            L += [f"- {t}" for t in items]
            L.append("")
    for f in AUDIT_FIELDS:
        if audit[f]:
            blocking_any = blocking_any or AUDIT_SEVERITY[f] == 0
            L.append(f"**auditor / {f.replace('_', ' ')}**")
            L += [f"- {t}" for t in audit[f]]
            L.append("")
    if not blocking_any:
        L.append("None reported.")
        L.append("")
    L.append("## Action list (most severe first)")
    L.append("")
    if actions:
        for i, (sev, who, t) in enumerate(actions, start=1):
            L.append(f"{i}. **[{SEVERITY_NAME[sev]}]** ({who}) {' '.join(str(t).split())}")
    else:
        L.append("Nothing to act on.")
    L.append("")
    L.append("## Strengths reported")
    L.append("")
    any_str = False
    for r in reviews:
        items = as_list(r.get("strengths"))
        if items:
            any_str = True
            L.append(f"**{r.get('reviewer') or r['_file']}**")
            L += [f"- {t}" for t in items]
            L.append("")
    if not any_str:
        L.append("None reported.")
        L.append("")
    if problems:
        L.append("## Merge problems")
        L.append("")
        L += [f"- {p}" for p in problems]
        L.append("")
    L.append("---")
    L.append("")
    L.append("Next: address every blocking and major item, record it in `response.md` "
             "(issue -> change -> file), rebuild with "
             "`python tools/check_paper.py --build --strict`, and run the next round "
             "(cap: 5 rounds or the writing freeze).")
    (d / "merged.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    summary = {
        "round": k,
        "scores": scores,
        "blocking_issues": blocking_issues,
        "converged": converged,
        "review_blocking": review_blocking,
        "n_reviews": len(reviews),
        "audit_present": audit_present,
        "audit_clean": audit_clean,
        "audit_counts": {f: len(audit[f]) for f in AUDIT_FIELDS},
        "round_dir": d.as_posix(),
        "merged_md": (d / "merged.md").as_posix(),
        "merged_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "problems": problems,
    }
    conv = Path(a.converged) if a.converged else d.parent / "converged.json"
    conv.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if conv.exists():
        try:
            prev = json.loads(conv.read_text(encoding="utf-8"))
            history = [h for h in prev.get("history", []) if h.get("round") != k]
            history.append({kk: prev[kk] for kk in ("round", "scores", "blocking_issues", "converged") if kk in prev})
            history = [h for h in history if h.get("round") != k]
        except (ValueError, OSError):
            history = []
    summary["history"] = sorted(history, key=lambda h: h.get("round", 0))
    conv.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    if a.json:
        print(json.dumps(summary))
    else:
        print(json.dumps(summary, indent=1))
        print("WROTE:", (d / "merged.md").as_posix(), "and", conv.as_posix())
        print("RESULT:", "CONVERGED" if converged else f"NOT CONVERGED ({blocking_issues} blocking)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
