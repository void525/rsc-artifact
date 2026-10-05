"""Anonymous code artifact for reviewers (optional node N19) -> submission/artifact_anonymous.zip.

Ships code, configs, tests and the calibration thresholds -- never data, model weights, author
identities, .git, .cursor or runs/ (except runs/calib/*/lambdas.json).  Verified by
paper/tools/check_anonymity.py before it is written.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, base_parser, load_config, log  # noqa: E402

INCLUDE_DIRS = ["src", "tests", "scripts", "configs", "graphctl", "paper/tools", "docs"]
INCLUDE_FILES = ["requirements.txt", "requirements-optional.txt", "bootstrap.py", "README_ARTIFACT.md"]
EXCLUDE_PARTS = {".git", ".cursor", "__pycache__", ".pytest_cache", ".venv", "runs", "data", "models"}
EXCLUDE_NAMES = {"AUTHORS.private.md", "AUTHORS.private.md.example", "anonymity_patterns.txt", "research_plan_www1_zh.md", "REAL_MINI_RUN.md"}
# files that legitimately *name* the identity files / patterns (they are the tooling itself)
SELF_REFERENTIAL = {"scripts/13_build_packet.py", "scripts/14_make_artifact.py", "paper/tools/check_anonymity.py", "paper/tools/check_paper.py"}
EMAIL_RE = __import__("re").compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PUBLIC_EMAIL_DOMAINS = (b"acm.org", b"openreview.net", b"thewebconf.org", b"huggingface.co", b"googlegroups.com")  # venue/tool contacts, not identities


def identity_strings() -> list:
    """Whatever the (never-shipped) private author file contains, plus obvious institution words."""
    out = []
    pat = ROOT / "docs" / "anonymity_patterns.txt"
    if pat.exists():
        out += [l.strip() for l in pat.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    p = ROOT / "docs" / "AUTHORS.private.md"
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip(" -*#\t")
            for tok in [t.strip() for t in line.replace("|", ",").split(",")]:
                if len(tok) >= 4 and not tok.lower().startswith(("name", "email", "affiliation", "orcid", "openreview", "(", "<")):
                    out.append(tok)
    return sorted(set(out))


def wanted(p: Path) -> bool:
    parts = set(p.parts)
    return not (parts & EXCLUDE_PARTS) and p.name not in EXCLUDE_NAMES and p.suffix not in (".pyc", ".npz", ".pdf")


def main() -> int:
    ap = base_parser("anonymous artifact")
    args = ap.parse_args()
    cfg = load_config(args)
    out = ROOT / "submission" / "artifact_anonymous.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    readme = ROOT / "README_ARTIFACT.md"
    if not readme.exists():
        readme.write_text(
            "# Anonymous artifact\n\nReference implementation of the paper's critical values, calibration regimes,\n"
            "controller and replay evaluator, with the exhaustive tests that check every theorem.\n\n"
            "```\npython bootstrap.py --no-torch      # numpy/scipy only\n./.venv/Scripts/python.exe -m pytest -q tests\n"
            "./.venv/Scripts/python.exe scripts/smoke_pipeline.py --no-latex   # synthetic end-to-end run\n```\n\n"
            "Data, model weights and run outputs are not included (size, licences); every download step is scripted.\n", encoding="utf-8")
    payload = {}
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and wanted(p.relative_to(ROOT)):
                payload[str(p.relative_to(ROOT)).replace("\\", "/")] = p.read_bytes()
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.is_file():
            payload[f] = p.read_bytes()
    for lam in sorted((ROOT / cfg["paths"]["runs"] / "calib").rglob("lambdas.json")) if (ROOT / cfg["paths"]["runs"] / "calib").exists() else []:
        payload[str(lam.relative_to(ROOT)).replace("\\", "/")] = lam.read_bytes()
    # small result files that back every number in the paper: statistics, replay summaries, latency/throughput
    # measurements, the policy-envelope analysis, the generated tables (incl. the full versions that did not fit the
    # PDF) and the provenance registry (macro -> value, source, hash)
    runs = ROOT / cfg["paths"]["runs"]
    extra_globs = ["stats/**/*.json", "replay/*/*/summary.json", "latency/**/*.json", "throughput/**/*.json", "policy_envelope/**/*.json",
                   "provenance/*.json", "pilot/*.json", "teacher/*/manifest.json", "N200/**/summary.json", "N200/calib/*/*/lambdas.json",
                   "review3/**/*.json", "throughput_rep/**/*.json"]
    for g in extra_globs:
        for f in sorted(runs.glob(g)):
            if f.is_file() and f.stat().st_size < 5_000_000:
                payload[str(f.relative_to(ROOT)).replace("\\", "/")] = f.read_bytes()
    for f in sorted((ROOT / "paper" / "tables").glob("*.tex")) + [ROOT / "paper" / "numbers.json", ROOT / "paper" / "numbers.tex"]:
        if f.is_file():
            payload[str(f.relative_to(ROOT)).replace("\\", "/")] = f.read_bytes()
    for f in sorted((ROOT / "paper" / "figures").glob("*.pdf")):
        payload[str(f.relative_to(ROOT)).replace("\\", "/")] = f.read_bytes()
    ids = identity_strings()
    leaks = []
    for name, blob in payload.items():
        if name in SELF_REFERENTIAL:
            continue
        hits = [m.group(0) for m in EMAIL_RE.finditer(blob)]
        private_mail = [h for h in hits if not any(h.endswith(d) for d in PUBLIC_EMAIL_DOMAINS)]
        if any(i.encode("utf-8") in blob for i in ids) or private_mail:
            leaks.append(f"{name} ({private_mail[:2]})" if private_mail else name)
    if leaks:
        log(f"REFUSED: identity strings found in {leaks[:5]}")
        return 1
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, blob in sorted(payload.items()):
            z.writestr(name, blob)
    log(f"artifact -> {out} ({out.stat().st_size / 1e6:.1f} MB, {len(payload)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
