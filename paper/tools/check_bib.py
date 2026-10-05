"""Structural (and optionally online) integrity check of paper/refs.bib.

    python tools/check_bib.py [--json] [--online] [--runs-dir DIR] [--bib FILE] [--tex-dir DIR]

Offline checks (all structural problems are errors; exit 1):
  * every key used in a \\cite{...} of paper/*.tex and paper/sections/*.tex exists in refs.bib
  * every entry carries at least one persistent identifier: doi, eprint or url
  * no duplicate keys and no duplicate normalised titles
  * the required fields of the entry type are present and non-empty
  * warnings: suspicious years (non-numeric, before 1900, or more than one year ahead),
    entries never cited, arXiv ids that do not look like arXiv ids

With --online the titles are additionally verified against the arXiv API (entries with an
``eprint``) and Crossref (entries with a ``doi``); every answer is cached as JSON under
``<runs-dir>/bibcheck/`` so repeated runs do not hit the network.  A title that disagrees
with the record is an error; a network failure is a warning, never an error.

Runs on Windows and Linux, stdlib only (numpy/scipy are not needed here), utf-8 everywhere.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent

REQUIRED: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    # entry type -> tuple of requirements; each requirement is a tuple of alternatives
    "article": (("author",), ("title",), ("journal",), ("year",)),
    "inproceedings": (("author",), ("title",), ("booktitle",), ("year",)),
    "incollection": (("author",), ("title",), ("booktitle",), ("year",)),
    "book": (("author", "editor"), ("title",), ("publisher",), ("year",)),
    "phdthesis": (("author",), ("title",), ("school",), ("year",)),
    "mastersthesis": (("author",), ("title",), ("school",), ("year",)),
    "techreport": (("author",), ("title",), ("institution",), ("year",)),
    "misc": (("author",), ("title",), ("year",)),
    "unpublished": (("author",), ("title",), ("note",)),
}
IDENTIFIER_FIELDS = ("doi", "eprint", "url")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_bib(text: str) -> Tuple[List[dict], List[str]]:
    """Return (entries, problems).  Each entry: {type, key, fields, line}."""
    entries: List[dict] = []
    problems: List[str] = []
    i, n = 0, len(text)
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        m = re.compile(r"@(\w+)\s*\{\s*", re.S).match(text, at)
        if not m:
            i = at + 1
            continue
        etype = m.group(1).lower()
        if etype in ("comment", "preamble", "string"):
            i = m.end()
            continue
        # find the matching closing brace of the entry
        depth, j = 1, m.end()
        while j < n and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        body = text[m.end():j - 1]
        line = text.count("\n", 0, at) + 1
        key, _, rest = body.partition(",")
        key = key.strip()
        if not key:
            problems.append(f"line {line}: @{etype} entry without a key")
        entries.append({"type": etype, "key": key, "fields": parse_fields(rest), "line": line})
        i = j
    return entries, problems


def parse_fields(body: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        m = re.compile(r"\s*(\w+)\s*=\s*", re.S).match(body, i)
        if not m:
            break
        name = m.group(1).lower()
        i = m.end()
        if i < n and body[i] == "{":
            depth, j = 1, i + 1
            while j < n and depth:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            value = body[i + 1:j - 1]
            i = j
        elif i < n and body[i] == '"':
            j = body.find('"', i + 1)
            j = n if j < 0 else j
            value = body[i + 1:j]
            i = j + 1
        else:
            j = i
            while j < n and body[j] not in ",\n":
                j += 1
            value = body[i:j]
            i = j
        fields[name] = " ".join(value.split())
        while i < n and body[i] in " \t\r\n":
            i += 1
        if i < n and body[i] == ",":
            i += 1
    return fields


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", t)          # latex commands
    t = re.sub(r"[{}$\\~^_]", " ", t)
    t = re.sub(r"[^0-9a-zA-Z]+", " ", t.lower())
    return " ".join(t.split())


def cited_keys(tex_dirs: List[Path]) -> Dict[str, List[str]]:
    used: Dict[str, List[str]] = {}
    for d in tex_dirs:
        for f in sorted(d.glob("*.tex")):
            s = f.read_text(encoding="utf-8", errors="replace")
            s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
            for m in re.finditer(r"\\(?:cite|citep|citet|citeauthor|citeyear|footcite)\*?(?:\[[^\]]*\])*\{([^}]*)\}", s):
                for k in m.group(1).split(","):
                    k = k.strip()
                    if k:
                        used.setdefault(k, []).append(f.name)
    return used


# --------------------------------------------------------------------------- #
# online verification
# --------------------------------------------------------------------------- #
_LAST_CALL: Dict[str, float] = {}
_MIN_GAP = {"export.arxiv.org": 6.0, "api.crossref.org": 0.2}


def _get(url: str, accept: str = "*/*", timeout: float = 30.0, retries: int = 4) -> str:
    """GET with a per-host throttle and backoff; both APIs rate-limit hard (429/406)."""
    host = urllib.parse.urlparse(url).netloc
    delay = 2.0
    for attempt in range(retries):
        gap = _MIN_GAP.get(host, 0.2) - (time.monotonic() - _LAST_CALL.get(host, -1e9))
        if gap > 0:
            time.sleep(gap)
        _LAST_CALL[host] = time.monotonic()
        req = urllib.request.Request(url, headers={
            "User-Agent": "rsc-rank-bibcheck/1.0 (anonymous WWW submission metadata check)",
            "Accept": accept,
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (406, 429, 500, 502, 503) and attempt + 1 < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise urllib.error.URLError("retries exhausted")


def arxiv_title(eprint: str, cache: dict) -> Tuple[str, str]:
    """(title, status).  status in {ok, miss, error}."""
    eid = eprint.strip().split("v")[0] if re.match(r"^\d{4}\.\d{4,5}v?\d*$", eprint.strip()) else eprint.strip()
    ck = f"arxiv:{eid}"
    if ck in cache:
        return cache[ck].get("title", ""), cache[ck].get("status", "miss")
    try:
        xml = _get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": eid, "max_results": 1}),
                   accept="application/atom+xml")
        m = re.search(r"<entry>.*?<title>(.*?)</title>", xml, flags=re.S)
        out = {"title": " ".join(m.group(1).split()) if m else "", "status": "ok" if m else "miss"}
    except (urllib.error.URLError, OSError, ValueError) as e:  # noqa: BLE001
        out = {"title": "", "status": "error", "detail": str(e)[:200]}
    if out["status"] != "error":
        cache[ck] = out
    return out["title"], out["status"]


def crossref_title(doi: str, cache: dict) -> Tuple[str, str]:
    ck = f"doi:{doi.lower()}"
    if ck in cache:
        return cache[ck].get("title", ""), cache[ck].get("status", "miss")
    try:
        js = json.loads(_get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""), accept="application/json"))
        titles = js.get("message", {}).get("title") or []
        out = {"title": " ".join(titles[0].split()) if titles else "", "status": "ok" if titles else "miss"}
    except (urllib.error.URLError, OSError, ValueError) as e:  # noqa: BLE001
        out = {"title": "", "status": "error", "detail": str(e)[:200]}
    if out["status"] != "error":
        cache[ck] = out
    return out["title"], out["status"]


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="check paper/refs.bib")
    ap.add_argument("--bib", default=str(PAPER / "refs.bib"))
    ap.add_argument("--tex-dir", action="append", default=None, help="directory with .tex files (repeatable)")
    ap.add_argument("--runs-dir", default=os.environ.get("RSCRANK_RUNS_DIR", str(ROOT / "runs")))
    ap.add_argument("--online", action="store_true", help="verify titles against arXiv/Crossref (cached)")
    ap.add_argument("--json", action="store_true", help="print only the JSON report")
    a = ap.parse_args(argv)

    errors: List[str] = []
    warnings: List[str] = []
    bib = Path(a.bib)
    if not bib.exists():
        print(json.dumps({"errors": [f"missing bib file {bib}"], "warnings": [], "info": {}}))
        return 1
    entries, problems = parse_bib(bib.read_text(encoding="utf-8", errors="replace"))
    errors += problems

    tex_dirs = [Path(d) for d in (a.tex_dir or [str(PAPER), str(PAPER / "sections")])]
    used = cited_keys([d for d in tex_dirs if d.is_dir()])

    by_key: Dict[str, dict] = {}
    dup_keys: List[str] = []
    for e in entries:
        if e["key"] in by_key:
            dup_keys.append(e["key"])
        else:
            by_key[e["key"]] = e
    if dup_keys:
        errors.append("duplicate bib keys: " + ", ".join(sorted(set(dup_keys))))

    titles: Dict[str, List[str]] = {}
    for e in entries:
        t = norm_title(e["fields"].get("title", ""))
        if t:
            titles.setdefault(t, []).append(e["key"])
    dup_titles = {t: ks for t, ks in titles.items() if len(ks) > 1}
    if dup_titles:
        errors.append("duplicate normalised titles: " + json.dumps({t: ks for t, ks in sorted(dup_titles.items())}))

    missing_keys = sorted(k for k in used if k not in by_key)
    if missing_keys:
        errors.append("cited keys missing from the bib: " + ", ".join(f"{k} ({','.join(sorted(set(used[k])))})" for k in missing_keys))

    no_id, missing_fields = [], {}
    this_year = _dt.date.today().year
    bad_years, bad_eprints = [], []
    for e in entries:
        f = e["fields"]
        if not any(f.get(x) for x in IDENTIFIER_FIELDS):
            no_id.append(e["key"])
        req = REQUIRED.get(e["type"])
        if req is None:
            warnings.append(f"{e['key']}: unknown entry type @{e['type']}")
        else:
            miss = [" or ".join(alt) for alt in req if not any(f.get(x) for x in alt)]
            if miss:
                missing_fields[e["key"]] = miss
        y = f.get("year", "")
        if not re.fullmatch(r"\d{4}", y or "") or not (1900 <= int(y or 0) <= this_year + 1):
            bad_years.append(f"{e['key']}={y or '<none>'}")
        ep = f.get("eprint", "")
        if ep and not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?|[a-z-]+(\.[A-Z]{2})?/\d{7}", ep):
            bad_eprints.append(f"{e['key']}={ep}")
    if no_id:
        errors.append("entries without doi/eprint/url: " + ", ".join(sorted(no_id)))
    if missing_fields:
        errors.append("missing required fields: " + json.dumps(missing_fields, sort_keys=True))
    if bad_years:
        warnings.append("suspicious years: " + ", ".join(sorted(bad_years)))
    if bad_eprints:
        warnings.append("suspicious arXiv ids: " + ", ".join(sorted(bad_eprints)))
    uncited = sorted(k for k in by_key if k not in used)
    if uncited:
        warnings.append(f"{len(uncited)} entries are never cited: " + ", ".join(uncited))

    online: Dict[str, dict] = {}
    if a.online:
        cache_dir = Path(a.runs_dir) / "bibcheck"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "title_cache.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
        mismatches, unreachable = [], []
        for e in entries:
            f, key = e["fields"], e["key"]
            want = norm_title(f.get("title", ""))
            got, status, src = "", "skip", ""
            # Crossref first (publisher record, no hard rate limit); arXiv for preprint-only entries.
            if f.get("doi"):
                got, status = crossref_title(f["doi"], cache)
                src = "crossref"
            if status != "ok" and f.get("eprint"):
                got, status = arxiv_title(f["eprint"], cache)
                src = "arxiv"
            if status == "ok":
                g = norm_title(got)
                agree = bool(g) and (g == want or g.startswith(want[:60]) or want.startswith(g[:60]))
                online[key] = {"source": src, "record_title": got, "agrees": agree}
                if not agree:
                    mismatches.append(f"{key}: bib={f.get('title', '')!r} record={got!r}")
            elif status in ("error", "miss"):
                unreachable.append(f"{key}({src or 'no id'}:{status})")
        cache_file.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
        if mismatches:
            errors.append("title mismatches against the record: " + " | ".join(mismatches))
        if unreachable:
            warnings.append("could not verify online: " + ", ".join(unreachable))

    report = {
        "errors": errors,
        "warnings": warnings,
        "info": {
            "bib": str(bib),
            "entries": len(entries),
            "cited_keys": len(used),
            "uncited_entries": uncited,
            "online_checked": len(online),
            "online": online,
        },
    }
    if a.json:
        print(json.dumps(report))
    else:
        print(json.dumps(report, indent=1))
        print("RESULT:", "OK" if not errors else f"{len(errors)} ERROR(S)")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
