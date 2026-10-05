"""Build and check the manuscript against the WWW 2027 rules and this project's integrity rules.

    cd paper && python tools/check_paper.py --build [--strict] [--allow-placeholders] [--json] [--preview]

Checks (E = error, W = warning; --strict turns page/number/anonymity warnings into errors):
  E  LaTeX build succeeds (latexmk -pdf), no undefined references/citations
  E  main text <= 8 pages (up to the References heading), total <= 12 pages
  E  no unresolved \numref keys (main.missing) unless --allow-placeholders
  E  no \PH{...} placeholders in sources unless --allow-placeholders
  E  first page contains the Web-relevance paragraph ("Relevance to the Web")
  E  documentclass options sigconf, anonymous, review present
  W/E hand-typed result numbers in results/analysis sections (numbers not inside \numref)
  W/E anonymity: names/emails/affiliations from docs/AUTHORS.private.md, common identifying strings
  W  forbidden phrases (SOTA, "novel framework", "guaranteed" outside theorem text)
  W  overfull hboxes > 10pt

Writes check_report.json and prints ONE JSON object with the keys `errors`, `warnings`, `info`;
`info` always carries pages_total, main_pages, references_page, undefined_refs, placeholders,
hand_typed_numbers, anonymity_leaks and overfull_gt10 (scripts/13_build_packet.py consumes them).
Exit 0 iff `errors` is empty.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(HERE))
from pdf_text import pages_text  # noqa: E402

# Every section whose numbers must come from the provenance registry.  Method/guarantee sections are
# excluded because they legitimately contain mathematical constants (k, N, indices, exponents).
RESULT_SECTIONS = ["sections/00_abstract.tex", "sections/01_intro.tex", "sections/06_setup.tex", "sections/07_results.tex",
                   "sections/08_analysis.tex", "sections/09_limitations.tex", "sections/10_conclusion.tex", "sections/91_appendix_extra.tex"]
ALLOWED_NUMBERS = {"0.01", "0.025", "0.05", "0.10", "0.1", "1", "2", "3", "5", "8", "10", "16", "20", "32", "50", "100", "200", "2000", "0.005", "4", "8/16/32", "384", "512", "2027", "2026", "0", "6", "7", "12", "13"}
FORBIDDEN = [r"\bSOTA\b", r"state[- ]of[- ]the[- ]art", r"novel framework", r"\bsignificantly\b(?!.*bootstrap)", r"\balways\b.*\bguarantee"]


def run(cmd, cwd=PAPER, timeout=1200):
    return subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")


def strip_math(s: str) -> str:
    """Remove math: a worked example like $\lambda\ge 101$ is mathematics, not a measured result."""
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
    s = re.sub(r"(?s)\\begin\{(equation|align|gather|multline|array|eqnarray)\*?\}.*?\\end\{\1\*?\}", " ", s)
    s = re.sub(r"(?s)\$\$.*?\$\$", " ", s)
    s = re.sub(r"(?s)(?<!\\)\$.*?(?<!\\)\$", " ", s)
    return s


def allowed_extra(text: str) -> set:
    """Per-file allowlist: a line `% check_paper: allow 5090, 128` marks constants that are not results."""
    out = set()
    for m in re.finditer(r"%\s*check_paper:\s*allow\s+([^\n]*)", text):
        out |= set(re.findall(r"\d+(?:\.\d+)?", m.group(1)))   # only the numbers; the rest of the line is a comment
    return out


def strip_tex(s: str) -> str:
    s = re.sub(r"(?m)%.*$", "", s)
    s = re.sub(r"\\numref\{[^}]*\}", " NUM ", s)
    s = re.sub(r"\\(ref|label|cite|eqref|input|includegraphics|taborph|figorph|texttt|PH|setthm)\*?(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", " ", s)
    s = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", s)
    return s


def references_page(pages) -> int:
    """1-based page on which the bibliography starts, -1 if it cannot be located.

    acmart's `review` option prints line numbers in both columns, so the extracted text
    rarely has "References" alone on a line: we look for the heading followed by the first
    bracketed entry, and fall back to the first page carrying both [1] and [2].
    """
    for i, t in enumerate(pages, start=1):
        if re.search(r"(?im)^\s*references\s*$", t):
            return i
    for i, t in enumerate(pages, start=1):
        m = re.search(r"(?i)\breferences\b", t)
        if m and re.search(r"\[\s*1\s*\]", t[m.end():]):
            return i
    for i, t in enumerate(pages, start=1):
        if re.search(r"\[\s*1\s*\]", t) and re.search(r"\[\s*2\s*\]", t):
            return i
    return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--allow-placeholders", action="store_true")
    ap.add_argument("--json", action="store_true", help="print only the JSON report (no RESULT line)")
    ap.add_argument("--preview", action="store_true", help="smoke preview build: numbers are synthetic, anonymity/hand-typed checks relaxed")
    a = ap.parse_args()
    errors, warnings, info = [], [], {}
    info.update(pages_total=-1, main_pages=-1, references_page=-1, undefined_refs=[], placeholders={},
                hand_typed_numbers={}, anonymity_leaks={}, overfull_gt10=0, unresolved_numrefs=[])
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    if not re.search(r"\\documentclass\[[^\]]*sigconf[^\]]*\]\{acmart\}", main_tex) or "anonymous" not in main_tex.split("\\begin{document}")[0] or "review" not in main_tex.split("\\begin{document}")[0]:
        errors.append("documentclass must be [sigconf,anonymous,review]{acmart}")
    if a.build:
        for f in ("main.missing",):
            try:
                (PAPER / f).unlink()
            except OSError:
                pass
        # -g forces every rule to run: without it latexmk can report "up to date" and
        # main.missing (written by \numref) would never be regenerated.
        p = run("latexmk -pdf -g -interaction=nonstopmode -halt-on-error -f main.tex")
        log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace") if (PAPER / "main.log").exists() else ""
        if p.returncode != 0 or not (PAPER / "main.pdf").exists():
            errs = [l for l in log.splitlines() if l.startswith("!")]
            errors.append("LaTeX build failed: " + " | ".join(errs[:5]))
        und = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' on page", log)))
        info["undefined_refs"] = und
        if und or "There were undefined references" in log:
            errors.append("undefined references/citations: " + (", ".join(und[:20]) if und else "see main.log"))
        over = [l for l in log.splitlines() if l.startswith("Overfull \\hbox")]
        big = [l for l in over if re.search(r"\((\d+\.\d+)pt too wide\)", l) and float(re.search(r"\((\d+\.\d+)pt too wide\)", l).group(1)) > 10]
        if big:
            warnings.append(f"{len(big)} overfull hboxes > 10pt")
        info["overfull_gt10"] = len(big)
    pdf = PAPER / "main.pdf"
    if pdf.exists():
        pages = pages_text(pdf)
        total = len(pages)
        ref_page = references_page(pages)
        main_pages = ref_page if ref_page > 0 else total
        info.update(pages_total=total, references_page=ref_page, main_pages=main_pages,
                    main_pages_upper_bound=main_pages)
        if total > 12:
            errors.append(f"total pages {total} > 12")
        if main_pages > 8:
            (errors if a.strict else warnings).append(f"main text may exceed 8 pages (References heading first seen on page {main_pages}); verify by eye")
        if ref_page < 0:
            warnings.append("could not locate the References page in the PDF text; main_pages is the total page count")
        first = pages[0] if pages else ""
        if "relevance to the web" not in first.lower():
            errors.append("first page lacks the 'Relevance to the Web' paragraph (WWW CFP requirement)")
        miss = PAPER / "main.missing"
        missing = sorted(set(l.strip() for l in miss.read_text(encoding="utf-8").splitlines() if l.strip())) if miss.exists() else []
        info["unresolved_numrefs"] = missing
        if missing and not a.allow_placeholders:
            errors.append(f"{len(missing)} unresolved \\numref keys: {', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''}")
    elif a.build:
        errors.append("main.pdf not produced")
    # sources
    src_files = sorted((PAPER / "sections").glob("*.tex"))
    ph = {}
    for f in src_files:
        s = f.read_text(encoding="utf-8")
        n = len(re.findall(r"\\PH\{", s))
        if n:
            ph[f.name] = n
    info["placeholders"] = ph
    if ph and not a.allow_placeholders:
        errors.append("placeholders \\PH{...} remain in: " + ", ".join(f"{k}({v})" for k, v in ph.items()))
    # hand-typed numbers in results sections
    typed = {}
    for rel in RESULT_SECTIONS:
        f = PAPER / rel
        if not f.exists():
            continue
        raw = f.read_text(encoding="utf-8")
        s = strip_tex(strip_math(raw))
        extra = allowed_extra(raw)
        nums = re.findall(r"(?<![\w.\\-])\d+(?:\.\d+)?%?(?![\w.])", s)
        bad = [n for n in nums if n.rstrip("%") not in ALLOWED_NUMBERS and n.rstrip("%") not in extra]
        if bad:
            typed[rel] = sorted(set(bad))
    info["hand_typed_numbers"] = typed
    if typed and not a.preview:
        (errors if a.strict else warnings).append("hand-typed numbers in results sections (use \\numref): " + json.dumps(typed))
    # forbidden phrases
    forb = {}
    for f in src_files:
        s = strip_tex(f.read_text(encoding="utf-8"))
        hits = [pat for pat in FORBIDDEN if re.search(pat, s, flags=re.IGNORECASE)]
        if hits:
            forb[f.name] = hits
    if forb:
        warnings.append("forbidden phrases: " + json.dumps(forb))
    # anonymity
    # Identity strings are NEVER hard-coded here: this file ships inside the anonymous artifact.
    # They come from docs/AUTHORS.private.md (git-ignored) and docs/anonymity_patterns.txt (optional).
    ident: list = []
    pat_file = PAPER.parent / "docs" / "anonymity_patterns.txt"
    if pat_file.exists():
        ident += [l.strip() for l in pat_file.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    priv = PAPER.parent / "docs" / "AUTHORS.private.md"
    if priv.exists():
        for m in re.findall(r"name:\s*(.+)|email:\s*(.+)|affiliation:\s*(.+)", priv.read_text(encoding="utf-8")):
            ident += [x.strip() for x in m if x and x.strip() and not x.strip().startswith("#")]
    leaks = {}
    for f in src_files + [PAPER / "main.tex"]:
        s = f.read_text(encoding="utf-8")
        hit = [w for w in set(ident) if w and w.lower() in s.lower()]
        if hit:
            leaks[f.name] = hit
    if re.search(r"\\author\{(?!Anonymous)", main_tex):
        leaks["main.tex"] = leaks.get("main.tex", []) + ["non-anonymous \\author"]
    info["anonymity_leaks"] = leaks
    if leaks and not a.preview:
        (errors if a.strict else warnings).append("anonymity: " + json.dumps(leaks))
    report = {"errors": errors, "warnings": warnings, "info": info, "strict": a.strict}
    (PAPER / "check_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if a.json:
        print(json.dumps(report))
    else:
        print(json.dumps(report, indent=1))
        print("RESULT:", "OK" if not errors else f"{len(errors)} ERROR(S)")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
