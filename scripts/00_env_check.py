"""Environment check -> runs/env/report.json  (node N00).

Checks Python, NumPy/SciPy, PyYAML, pytest, torch (+CUDA, GPU name, memory, sm
capability), transformers, datasets, pytrec_eval, pyserini/java or bm25s, LaTeX
(latexmk + acmart), free disk, and runs the unit tests.  ``--strict`` exits
non-zero when a *required* item is missing.  Missing optional items are
reported so the decision ladder can pick fallbacks (docs/DECISION_LADDER.md).
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, base_parser, load_config, log  # noqa: E402


def try_import(name):
    try:
        m = importlib.import_module(name)
        return {"ok": True, "version": getattr(m, "__version__", "?")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200]}


def run(cmd, timeout=600):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        return {"ok": p.returncode == 0, "exit": p.returncode, "tail": (p.stdout + p.stderr)[-1500:]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def main() -> int:
    ap = base_parser("environment check")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args)
    rep = {"python": sys.version, "python_exe": sys.executable, "platform": platform.platform(), "cwd": str(ROOT), "packages": {}, "gpu": {}, "tools": {}, "disk": {}}
    for pkg in ("numpy", "scipy", "yaml", "pytest", "torch", "transformers", "huggingface_hub", "pyarrow", "datasets", "pytrec_eval", "pyserini", "bm25s", "Stemmer", "sentence_transformers", "matplotlib", "pandas"):
        rep["packages"][pkg] = try_import(pkg)
    # GPU
    if rep["packages"]["torch"]["ok"]:
        import torch

        g = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            g.update(name=p.name, total_mem_gb=round(p.total_memory / 2**30, 1), capability=f"{p.major}.{p.minor}", arch_list=torch.cuda.get_arch_list())
            g["sm120_supported"] = any("sm_120" in a for a in torch.cuda.get_arch_list())
            try:
                x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
                g["bf16_matmul_ok"] = bool(torch.isfinite((x @ x).float().sum()).item())
            except Exception as e:  # noqa: BLE001
                g["bf16_matmul_ok"] = False
                g["error"] = str(e)[:300]
        rep["gpu"] = g
    # tools
    rep["tools"]["java"] = run("java -version", 60)
    rep["tools"]["latexmk"] = run("latexmk -version", 60)
    rep["tools"]["pdflatex"] = run("pdflatex --version", 60)
    rep["tools"]["tectonic"] = run("tectonic --version", 60)
    rep["tools"]["kpsewhich_acmart"] = run("kpsewhich acmart.cls", 60)
    rep["tools"]["git"] = run("git --version", 60)
    rep["tools"]["cursor_agent_cli"] = {"ok": shutil.which("agent") is not None}
    du = shutil.disk_usage(str(ROOT))
    rep["disk"] = {"free_gb": round(du.free / 2**30, 1), "total_gb": round(du.total / 2**30, 1), "enough_for_real_run": bool(du.free / 2**30 >= 120)}
    rep["hf_cache"] = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE") or str(Path.home() / ".cache" / "huggingface")
    if not args.skip_tests:
        rep["unit_tests"] = run(f'"{sys.executable}" -m pytest -q tests', 1800)
    # requirements
    required = ["numpy", "scipy", "yaml", "pytest"]
    real_required = ["torch", "transformers", "huggingface_hub", "pyarrow"] if not args.synthetic else []
    missing = [p for p in required + real_required if not rep["packages"][p]["ok"]]
    rep["missing_required"] = missing
    rep["candidate_backend_available"] = {"pyserini": rep["packages"]["pyserini"]["ok"] and rep["tools"]["java"]["ok"], "bm25s": rep["packages"]["bm25s"]["ok"] and rep["packages"]["Stemmer"]["ok"]}
    rep["latex_ok"] = (rep["tools"]["latexmk"]["ok"] or rep["tools"]["pdflatex"]["ok"] or rep["tools"]["tectonic"]["ok"]) and (rep["tools"]["kpsewhich_acmart"]["ok"] or (ROOT / "paper" / "acmart.cls").exists())
    rep["ok"] = not missing and rep.get("unit_tests", {"ok": True})["ok"] and (args.synthetic or rep["gpu"].get("cuda_available", False))
    out = ROOT / cfg["paths"]["runs"] / "env" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log(f"env report -> {out}  ok={rep['ok']}  missing={missing}  gpu={rep['gpu'].get('name')}")
    if not rep["ok"]:
        log("NOT OK. If torch/transformers are missing: pip install per docs/CURSOR_SETUP.md (torch cu128 wheels for RTX 50xx).")
    return 0 if (rep["ok"] or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
