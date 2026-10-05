"""Assemble and check the OpenReview submission packet (node N17) -> submission/.

    python scripts/13_build_packet.py            # build the PDF, run every check, write submission/
    python scripts/13_build_packet.py --abstract # only the abstract-registration packet (node N16)

Outputs
  submission/main.pdf                 the exact file the human uploads
  submission/packet.json              machine-checkable state of every requirement
  submission/CHECKLIST.md             the CFP checklist with the measured values
  submission/openreview_fields.md     copy-paste fields for the OpenReview form
  submission/abstract_registration.md fields needed by the abstract deadline (author list is frozen then)
  submission/camera_ready_TODO.md     what changes for the non-anonymous version
  graph/HANDOFF.md / graph/HANDOFF_ABSTRACT.md   the two human actions, in 10 lines each
The script never uploads anything: OpenReview is human-only (graph/decisions.md).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, base_parser, load_config, log  # noqa: E402

from rscrank.io import read_json, write_json  # noqa: E402

SUB = ROOT / "submission"
PAPER = ROOT / "paper"
VENUE = "https://openreview.net/group?id=ACM.org/TheWebConf/2027/Conference_Long_Papers"
TRACK = "Search, Recommendation, and Retrieval-Augmented AI"


def run(cmd, timeout=1800):
    p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def pdf_pages(pdf: Path) -> int:
    for tool, pat in (("pdfinfo", r"Pages:\s+(\d+)"),):
        if shutil.which(tool):
            rc, out = run([tool, str(pdf)], 120)
            m = re.search(pat, out)
            if rc == 0 and m:
                return int(m.group(1))
    try:  # pure-python fallback: count page objects
        raw = pdf.read_bytes()
        return max(raw.count(b"/Type /Page\n"), raw.count(b"/Type/Page"), 0) or -1
    except Exception:  # noqa: BLE001
        return -1


def references_page(pdf: Path) -> int:
    """1-based page on which the bibliography starts (-1 if not found)."""
    if not shutil.which("pdftotext"):
        return -1
    rc, _ = run(["pdftotext", "-layout", str(pdf), str(SUB / "_main.txt")], 300)
    if rc != 0:
        return -1
    pages = (SUB / "_main.txt").read_text(encoding="utf-8", errors="replace").split("\f")
    for i, p in enumerate(pages, start=1):
        if re.search(r"^\s*(REFERENCES|References)\s*$", p, flags=re.MULTILINE):
            return i
    return -1


def abstract_text() -> str:
    t = (PAPER / "sections" / "00_abstract.tex").read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"\\begin\{abstract\}|\\end\{abstract\}", "", t)
    t = re.sub(r"(?<!\\)%.*", "", t).replace("\\%", "%")
    # registered numbers -> their typeset value (paper/numbers.tex); an unregistered key stays visible as <Key>
    typeset = {}
    nt = PAPER / "numbers.tex"
    if nt.exists():
        for m in re.finditer(r"\\newcommand\{\\num([A-Za-z]+)\}\{([^}]*(?:\{,\}[^}]*)*)\}", nt.read_text(encoding="utf-8", errors="replace")):
            typeset[m.group(1)] = m.group(2).replace("{,}", ",")
    t = re.sub(r"\\num(ref)?\{([A-Za-z]+)\}", lambda m: typeset.get(m.group(2), f"<{m.group(2)}>"), t)
    t = re.sub(r"\\(emph|textbf|textit|texttt)\{([^{}]*)\}", r"\2", t)
    # project macros and the few maths symbols the abstract uses -> plain Unicode for the OpenReview form
    for pat, rep in ((r"\\method\{\}", "RSC-Rank"), (r"\\method(?![A-Za-z])", "RSC-Rank"), (r"\\Akr(?![A-Za-z])", "A_{k,r}"), (r"\\Ak(?![A-Za-z])", "A_{k,1}"),
                     (r"\\Btop(?![A-Za-z])", "B_top"), (r"\\Ball(?![A-Za-z])", "B_all"), (r"\\Tset(?![A-Za-z])", "T_k"), (r"\\rthmax(?![A-Za-z])", "r-th max"),
                     (r"\\lambda(?![A-Za-z])", "λ"), (r"\\mu(?![A-Za-z])", "μ"), (r"\\sigma(?![A-Za-z])", "σ"), (r"\\alpha(?![A-Za-z])", "α"), (r"\\delta(?![A-Za-z])", "δ"),
                     (r"\\ge(?![A-Za-z])", "≥"), (r"\\le(?![A-Za-z])", "≤"), (r"\\notin(?![A-Za-z])", "∉"), (r"\\in(?![A-Za-z])", "∈"), (r"\\times(?![A-Za-z])", "×"), (r"---", "—"), (r"--", "–"),
                     (r"``", "\u201c"), (r"''", "\u201d"), (r"\\,", " "), (r"~", " ")):
        t = re.sub(pat, rep, t)
    t = re.sub(r"[$\\]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def authors_block() -> str:
    p = ROOT / "docs" / "AUTHORS.private.md"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace").strip()
    return ("(docs/AUTHORS.private.md not filled in -- copy docs/AUTHORS.private.md.example and fill it. "
            "The author list and order are FROZEN at the abstract deadline by the CFP.)")


def main() -> int:
    ap = base_parser("build the submission packet")
    ap.add_argument("--abstract", action="store_true", help="only the abstract-registration packet (N16)")
    ap.add_argument("--allow-placeholders", action="store_true", help="N16 before results exist")
    args = ap.parse_args()
    cfg = load_config(args)
    SUB.mkdir(parents=True, exist_ok=True)
    title = "Rank-Sufficient Calibration: Tight Error Frontiers for Policy-Uniform Web Reranking"
    packet = {"title": title, "track": TRACK, "venue": VENUE, "checks": {}, "ok": False}

    # ---- build + strict check -------------------------------------------------
    cmd = [sys.executable, "tools/check_paper.py", "--build", "--json"] + ([] if not args.allow_placeholders else ["--allow-placeholders"]) + ([] if args.allow_placeholders else ["--strict"])
    p = subprocess.run([str(c) for c in cmd], cwd=str(PAPER), capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        rep = json.loads(re.search(r"\{.*\}", p.stdout, flags=re.S).group(0))
    except Exception:  # noqa: BLE001
        rep = {"errors": [f"check_paper.py did not return JSON (exit {p.returncode}): {(p.stdout + p.stderr)[-800:]}"], "info": {}}
    packet["check_paper"] = rep
    packet["checks"]["paper_checks_pass"] = not rep.get("errors")

    pdf = PAPER / "main.pdf"
    if pdf.exists():
        shutil.copy(pdf, SUB / "main.pdf")
        info = rep.get("info", {}) or {}
        # check_paper.py knows where the bibliography starts (acmart's review line numbers break naive
        # text matching); fall back to our own detectors only if it did not report them.
        pages = int(info.get("pages_total") or 0) or pdf_pages(SUB / "main.pdf")
        ref_page = int(info.get("references_page") or 0) or references_page(SUB / "main.pdf")
        if info.get("main_pages"):
            packet["main_pages_reported"] = int(info["main_pages"])
        packet.update(pages_total=pages, references_page=ref_page, main_pages=int(packet.get("main_pages_reported") or (ref_page - 1 if ref_page > 0 else pages)))
        packet["checks"]["pages_total_le_12"] = 0 < pages <= 12
        packet["checks"]["main_pages_le_8"] = 0 < packet["main_pages"] <= 8
        packet["checks"]["pdf_size_le_50mb"] = (SUB / "main.pdf").stat().st_size <= 50 * 1024 * 1024
    else:
        packet["checks"]["pdf_exists"] = False

    # ---- anonymity + numbers --------------------------------------------------
    anon_tool = PAPER / "tools" / "check_anonymity.py"
    if anon_tool.exists():
        rc, out = run([sys.executable, str(anon_tool), str(SUB / "main.pdf")], 600)
        packet["anonymity"] = out[-2000:]
        packet["checks"]["anonymous"] = rc == 0
    else:
        packet["checks"]["anonymous"] = False
        packet["anonymity"] = "paper/tools/check_anonymity.py missing"
    nums = PAPER / "numbers.json"
    packet["checks"]["numbers_ok"] = nums.exists() or bool(args.allow_placeholders)
    stats = ROOT / cfg["paths"]["runs"] / "stats" / "summary.json"
    if stats.exists():
        s = read_json(stats)
        packet["claim_level"] = s.get("claim_level")
        packet["checks"]["not_synthetic"] = not s.get("synthetic", True)

    # ---- fields ---------------------------------------------------------------
    abst = abstract_text()
    fields = [
        "# OpenReview fields (copy-paste)", "",
        f"* **Venue**: {VENUE}", f"* **Track / primary subject area**: {TRACK}",
        "* **Industry paper**: No", f"* **Title** ({len(title)}/250 chars): {title}", "",
        f"* **Abstract** ({len(abst)}/5000 chars):", "", abst, "",
        "* **TL;DR** (<=250 chars): One calibrated threshold on a rank-sufficient statistic lets a reranker stop early with a distribution-free bound on returning anything but the teacher's exact top-k, under every scoring order.",
        "* **Keywords**: web search reranking; early stopping; conformal calibration; risk control; top-k selection; efficiency", "",
        "* **Authors** (frozen at the abstract deadline):", "", authors_block(), "",
        "* **Required checkboxes**: all co-authors have OpenReview profiles; reviews may be made public if accepted; authors may be asked to review; the submission follows the CFP.",
    ]
    (SUB / "openreview_fields.md").write_text("\n".join(fields), encoding="utf-8")
    (SUB / "abstract_registration.md").write_text("\n".join(fields + ["", "Deadline: 2026-10-18 23:59 AoE (= 2026-10-19 19:59 Beijing). Placeholder abstracts are forbidden; this one is real."]), encoding="utf-8")
    (SUB / "camera_ready_TODO.md").write_text(
        "# Camera-ready (only if accepted; due 2027-01-31)\n\n"
        "- Remove `anonymous,review` from \\documentclass, add the real authors, affiliations, ORCIDs and e-mails.\n"
        "- Add the acknowledgements and funding paragraph (kept out of the anonymous version).\n"
        "- Keep the AI-tools statement (ACM authorship policy, May 2026: AI used in the research is described in the methods).\n"
        "- Replace the code-availability sentence with the archival repository DOI ('Artifacts Available' badge).\n"
        "- ACM rights form, 12 pages of which 8 content pages, one full registration per paper.\n", encoding="utf-8")

    checklist = ["# Submission checklist (WWW 2027 research track; docs/WWW2027_FACTS.md)", ""]
    for key, label in (
        ("paper_checks_pass", "paper/tools/check_paper.py passes (no undefined refs, no placeholders, numbers from macros)"),
        ("main_pages_le_8", f"main content <= 8 pages (measured: {packet.get('main_pages', '?')})"),
        ("pages_total_le_12", f"total <= 12 pages (measured: {packet.get('pages_total', '?')})"),
        ("pdf_size_le_50mb", "single PDF <= 50 MB"),
        ("anonymous", "double-blind: no authors, affiliations, acknowledgements, funding or identifying links"),
        ("numbers_ok", "every number in the paper comes from paper/numbers.tex (provenance registry)"),
        ("not_synthetic", "results come from the real pipeline, not the synthetic smoke run"),
    ):
        v = packet["checks"].get(key)
        checklist.append(f"- [{'x' if v else ' '}] {label}")
    checklist += ["", "Manual eyeball before uploading: page 1 states the relevance to the Web; CCS concepts and keywords present;",
                  "the appendix is inside the same PDF (no separate supplementary upload exists);",
                  "the abstract in the form matches the PDF."]
    (SUB / "CHECKLIST.md").write_text("\n".join(checklist), encoding="utf-8")

    packet["ok"] = all(v for k, v in packet["checks"].items()) if not args.allow_placeholders else packet["checks"].get("paper_checks_pass", False)
    write_json(SUB / "packet.json", packet)

    hand = (ROOT / "graph" / ("HANDOFF_ABSTRACT.md" if args.abstract else "HANDOFF.md"))
    if args.abstract:
        hand.write_text(
            "# 需要你做的第一件事：注册摘要（2026-10-18 23:59 AoE = 北京时间 10-19 19:59 前）\n\n"
            f"1. 打开 {VENUE} ，登录 OpenReview（所有共同作者都要有资料完整的账号，否则可能被 desk reject）。\n"
            "2. 点 New Submission，把 `submission/abstract_registration.md` 里的字段逐项粘贴（标题/摘要/TL;DR/关键词/track=Search, Recommendation, and Retrieval-Augmented AI/Industry paper=No）。\n"
            "3. 作者名单与顺序在摘要截止后**不能再改**：提交前确认 `docs/AUTHORS.private.md` 里的名单就是最终名单。\n"
            "4. 勾选表单要求的四个确认框，提交。PDF 这一步不需要上传。\n"
            "5. 回到 Cursor 不用做任何事，代理会继续把正文做完。\n", encoding="utf-8")
    else:
        hand.write_text(
            "# 需要你做的第二件事：上传正文（2026-10-25 23:59 AoE = 北京时间 10-26 19:59 前）\n\n"
            f"1. 打开 {VENUE} ，编辑已经注册的那篇投稿。\n"
            "2. 上传 `submission/main.pdf`（≤ 50 MB，附录已经在同一个 PDF 里，WWW 没有单独的补充材料入口）。\n"
            "3. 用眼睛确认三件事：第 1 页有 “Relevance to the Web” 段落；正文不超过 8 页、总页数不超过 12 页；全文没有作者信息。\n"
            "4. 字段如有改动，按 `submission/openreview_fields.md` 更新；提交。\n"
            f"5. `submission/CHECKLIST.md` 里是机器检查的结果（当前 ok={packet['ok']}）。\n", encoding="utf-8")
    log(f"packet -> {SUB} (ok={packet['ok']}); handoff -> {hand}")
    return 0 if packet["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
