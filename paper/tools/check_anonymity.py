"""Double-blind check of the manuscript sources and (optionally) the built PDF.

    python tools/check_anonymity.py [main.pdf] [--json] [--paper-dir DIR] [--authors FILE]

Scans paper/*.tex, paper/sections/*.tex, paper/refs.bib and, when a PDF is given, its
extracted text and its document metadata (Title/Author/Subject/Keywords/Creator/Producer/
XMP) for:

  * every non-trivial string from docs/AUTHORS.private.md (names, e-mails, affiliations)
  * e-mail addresses whose domain is not a public venue/tool domain
  * acknowledgement and funding sections ("acknowledg*", "funded by", grant numbers)
  * GitHub / GitLab / personal URLs that carry a user or organisation name
  * \\author{...} / \\affiliation{...} content other than the anonymous placeholder
  * first-person self-citation ("our previous work", "in our earlier paper", "we have shown in [n]")

Prints a JSON report; exit code 1 if anything was found, 0 otherwise.
Stdlib only, utf-8 everywhere, runs on Windows and Linux.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent

# domains that may legitimately appear in a double-blind paper (venues, tools, standards)
PUBLIC_DOMAINS = {
    "acm.org", "ieee.org", "arxiv.org", "openreview.net", "aclanthology.org", "aclweb.org",
    "neurips.cc", "icml.cc", "iclr.cc", "mlr.press", "proceedings.mlr.press", "jmlr.org",
    "springer.com", "sciencedirect.com", "elsevier.com", "doi.org", "dx.doi.org", "crossref.org",
    "huggingface.co", "github.io", "python.org", "numpy.org", "scipy.org", "pytorch.org",
    "trec.nist.gov", "nist.gov", "vldb.org", "sigir.org", "thewebconf.org", "www2027.thewebconf.org",
    "projecteuclid.org", "ijcai.org", "auai.org", "example.org", "example.com", "anonymous.invalid",
    "wikipedia.org", "latex-project.org", "ctan.org", "tug.org", "kaggle.com", "zenodo.org",
}
ANON_AUTHOR = re.compile(r"^\s*Anonymous\s+Author\(s\)\s*$", re.I)
ANON_INSTITUTION = re.compile(r"Anonymous\s+Institution", re.I)

ACK_PATTERNS = [
    (r"(?i)\\section\*?\{[^}]*acknowledg", "acknowledgement section"),
    (r"(?i)\\begin\{acks\}", "acks environment"),
    (r"(?i)\backnowledg(e|ement|ments|ments?)\b", "acknowledgement wording"),
    (r"(?i)\b(this work was|research was|partially) (funded|supported) by\b", "funding statement"),
    (r"(?i)\b(grant|award)\s*(no\.?|number|#)\s*[A-Z0-9][-A-Z0-9/]{3,}", "grant number"),
    (r"(?i)\bNSF\s+(grant|award)\b|\bERC\b\s+(grant|advanced)|\bNSFC\b", "funder name"),
]
SELF_CITE_PATTERNS = [
    r"(?i)\bour (previous|prior|earlier|own) (work|paper|study|system|method|results?)\b",
    r"(?i)\bin our (previous|prior|earlier) (work|paper|study)\b",
    r"(?i)\bwe (have )?(previously|earlier) (showed|shown|proposed|introduced|published)\b",
    r"(?i)\bas we (showed|proposed|introduced) in\b",
    r"(?i)\b(extends?|builds? (up)?on) our\b",
]
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s{}\"'\\,)]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
CODEHOST_RE = re.compile(r"(?i)\b(?:https?://)?(?:www\.)?(github\.com|gitlab\.com|bitbucket\.org|gitee\.com|codeberg\.org|huggingface\.co)/([A-Za-z0-9_.\-]+)(?:/([A-Za-z0-9_.\-]+))?")
CODEHOST_ANON_OWNERS = {"anonymous", "anonymous-authors", "anon", "anonymised", "anonymized", "blind", "double-blind"}


def macro_args(text: str, name: str) -> List[str]:
    """Every balanced argument of \\<name>{...}, including nested occurrences."""
    out: List[str] = []
    for m in re.finditer(r"\\" + re.escape(name) + r"\s*\{", text):
        depth, j = 1, m.end()
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        out.append(text[m.end():j - 1])
    return out


def private_strings(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<!--"):
            continue
        m = re.match(r"(?i)^[-*\s]*(name|email|e-mail|affiliation|institution|orcid|author)\s*:\s*(.+)$", line)
        if m:
            val = m.group(2).strip().strip("`\"'")
            if len(val) >= 4 and "example" not in val.lower() and not val.startswith("<"):
                out.append(val)
    return sorted(set(out))


def pdf_metadata(pdf: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        from pypdf import PdfReader

        r = PdfReader(str(pdf))
        for k, v in (r.metadata or {}).items():
            meta[str(k).lstrip("/")] = str(v)
        try:
            xmp = r.xmp_metadata
            if xmp is not None and getattr(xmp, "stream", None) is not None:
                meta["XMP"] = xmp.stream.get_data().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        import shutil
        import subprocess

        if shutil.which("pdfinfo"):
            out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
            for line in out.splitlines():
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
    return meta


def pdf_text(pdf: Path) -> str:
    sys.path.insert(0, str(HERE))
    try:
        from pdf_text import pages_text  # noqa: PLC0415

        return "\n".join(pages_text(pdf))
    except Exception:  # noqa: BLE001
        return ""


def scan(name: str, text: str, priv: List[str], leaks: Dict[str, List[str]]) -> None:
    def add(what: str) -> None:
        leaks.setdefault(name, [])
        if what not in leaks[name]:
            leaks[name].append(what)

    low = text.lower()
    for p in priv:
        if p.lower() in low:
            add(f"private string from AUTHORS.private.md: {p!r}")
    for m in EMAIL_RE.finditer(text):
        addr = m.group(0)
        dom = addr.split("@")[-1].lower().rstrip(".")
        if not any(dom == d or dom.endswith("." + d) for d in PUBLIC_DOMAINS):
            add(f"e-mail address with a non-public domain: {addr}")
    for pat, what in ACK_PATTERNS:
        m = re.search(pat, text)
        if m:
            add(f"{what}: {text[max(0, m.start() - 30):m.end() + 40].strip()[:120]!r}")
    for m in CODEHOST_RE.finditer(text):
        owner = (m.group(2) or "").lower()
        if owner and owner not in CODEHOST_ANON_OWNERS:
            add(f"code-hosting URL with an owner name: {m.group(0)}")
    for pat in SELF_CITE_PATTERNS:
        m = re.search(pat, text)
        if m:
            add(f"first-person self-citation: {text[max(0, m.start() - 20):m.end() + 40].strip()[:120]!r}")
    for m in URL_RE.finditer(text):
        u = m.group(0)
        host = re.sub(r"^(https?://)?(www\.)?", "", u).split("/")[0].lower()
        if any(host == d or host.endswith("." + d) for d in PUBLIC_DOMAINS):
            continue
        if re.search(r"(?i)/~[A-Za-z0-9._\-]+", u) or re.search(r"(?i)\b[a-z0-9\-]+\.(github\.io|gitlab\.io|netlify\.app|vercel\.app)\b", u):
            add(f"personal URL: {u}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="double-blind check of the manuscript")
    ap.add_argument("pdf", nargs="?", default=None, help="optional built PDF to scan as well")
    ap.add_argument("--paper-dir", default=str(PAPER))
    ap.add_argument("--authors", default=str(ROOT / "docs" / "AUTHORS.private.md"))
    ap.add_argument("--json", action="store_true", help="print only the JSON report")
    a = ap.parse_args(argv)

    paper = Path(a.paper_dir)
    priv = private_strings(Path(a.authors))
    leaks: Dict[str, List[str]] = {}
    scanned: List[str] = []

    sources = sorted(paper.glob("*.tex")) + sorted((paper / "sections").glob("*.tex"))
    bib = paper / "refs.bib"
    if bib.exists():
        sources.append(bib)
    for f in sources:
        text = f.read_text(encoding="utf-8", errors="replace")
        rel = f.relative_to(paper).as_posix()
        scanned.append(rel)
        scan(rel, text, priv, leaks)

    # \author / \affiliation must stay on the anonymous placeholder
    main_tex = paper / "main.tex"
    if main_tex.exists():
        s = main_tex.read_text(encoding="utf-8", errors="replace")
        s_nc = re.sub(r"(?m)(?<!\\)%.*$", "", s)
        for v in macro_args(s_nc, "author"):
            if not ANON_AUTHOR.match(v):
                leaks.setdefault("main.tex", []).append(f"non-anonymous \\author{{{v[:80]}}}")
        for name in ("institution", "affiliation"):
            for v in macro_args(s_nc, name):
                plain = re.sub(r"\\[a-zA-Z@]+\s*(\{[^{}]*\})?", " ", v).strip()
                if name == "affiliation":
                    v = plain          # the inner \institution{...} is checked on its own
                if v.strip() and not ANON_INSTITUTION.search(v):
                    leaks.setdefault("main.tex", []).append(f"non-anonymous \\{name}{{{v.strip()[:80]}}}")
        for f_ in ("\\email{", "\\orcid{"):
            if f_ in s_nc:
                leaks.setdefault("main.tex", []).append(f"identifying macro present: {f_}...}}")
        if "anonymous" not in s_nc.split("\\begin{document}")[0]:
            leaks.setdefault("main.tex", []).append("documentclass is missing the 'anonymous' option")

    pdf_info: Dict[str, object] = {}
    if a.pdf:
        p = Path(a.pdf)
        if not p.is_absolute() and not p.exists():
            p = paper / a.pdf
        if not p.exists():
            leaks.setdefault(str(a.pdf), []).append("PDF not found")
        else:
            scanned.append(p.name)
            txt = pdf_text(p)
            if not txt:
                leaks.setdefault(p.name, []).append("could not extract text from the PDF (install pypdf or pdftotext)")
            scan(p.name, txt, priv, leaks)
            meta = pdf_metadata(p)
            pdf_info = {"metadata_fields": sorted(meta)}
            blob = "\n".join(f"{k}: {v}" for k, v in meta.items())
            scan(p.name + " [metadata]", blob, priv, leaks)
            for k in ("Author", "Title", "Subject", "Keywords"):
                v = (meta.get(k) or "").strip()
                if k == "Author" and v and not ANON_AUTHOR.match(v):
                    leaks.setdefault(p.name + " [metadata]", []).append(f"PDF /Author = {v[:80]!r}")

    report = {
        "leaks": leaks,
        "leak_count": sum(len(v) for v in leaks.values()),
        "info": {
            "scanned": scanned,
            "private_strings_loaded": len(priv),
            "authors_file": a.authors if Path(a.authors).exists() else None,
            "pdf": str(a.pdf) if a.pdf else None,
            **pdf_info,
        },
    }
    if a.json:
        print(json.dumps(report))
    else:
        print(json.dumps(report, indent=1))
        print("RESULT:", "OK (anonymous)" if not leaks else f"{report['leak_count']} POSSIBLE LEAK(S)")
    return 0 if not leaks else 1


if __name__ == "__main__":
    sys.exit(main())
