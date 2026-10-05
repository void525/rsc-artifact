"""Extract per-page text from a PDF (pypdf, fallback pdftotext).  Used by check_paper.py and the reviewer/auditor subagents.

    python tools/pdf_text.py [main.pdf] [--out main.txt]
Prints the text with '=== page N ===' separators (or writes it to --out).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def pages_text(pdf: Path) -> List[str]:
    try:
        from pypdf import PdfReader

        r = PdfReader(str(pdf))
        return [(p.extract_text() or "") for p in r.pages]
    except Exception:  # noqa: BLE001
        if shutil.which("pdftotext"):
            out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
            pages = out.split("\f")
            # pdftotext terminates every page with a form feed, so the split leaves one empty
            # trailing element that is not a page (an off-by-one that made a 12-page PDF count as 13)
            if pages and pages[-1].strip() == "":
                pages = pages[:-1]
            return pages
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default="main.pdf")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pages = pages_text(Path(a.pdf))
    txt = "\n".join(f"=== page {i + 1} ===\n{t}" for i, t in enumerate(pages))
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"{len(pages)} pages -> {a.out}")
    else:
        sys.stdout.write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
