#!/usr/bin/env python3
"""One-shot environment bootstrap (standard library only; Windows / Linux / macOS).

    python bootstrap.py                # create ./.venv, install requirements + CUDA torch
    python bootstrap.py --cpu-torch    # CPU wheel (CI / no GPU)
    python bootstrap.py --no-torch     # skip torch entirely
    python bootstrap.py --python C:/Users/me/anaconda3/python.exe

Why a venv and not conda: every tool in this repo (and the Cursor hooks) calls the interpreter by the
*fixed relative path* ``./.venv/Scripts/python.exe`` -- on Windows that is where venv puts it, and on
POSIX this script creates that exact path as a symlink to ``.venv/bin/python``.  One spelling works in
PowerShell and in bash, so nothing depends on PATH, on conda activation, or on the Windows Store stub.
Idempotent: re-running only installs what is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
CUDA_INDEXES = ["https://download.pytorch.org/whl/cu128", "https://download.pytorch.org/whl/cu129", "https://download.pytorch.org/whl/cu130"]


def venv_python(v: Path = VENV) -> Path:
    return v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def portable_python(v: Path = VENV) -> Path:
    """The single spelling used everywhere: .venv/Scripts/python.exe (symlinked on POSIX)."""
    return v / "Scripts" / "python.exe"


def child_env() -> dict:
    """pip and every child run UTF-8: on a Chinese/Japanese Windows install the console code page is
    cp936/cp932 and package metadata or a progress bar can abort an otherwise fine install."""
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def run(cmd, **kw) -> int:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    kw.setdefault("env", child_env())
    return subprocess.run([str(c) for c in cmd], cwd=str(ROOT), **kw).returncode


def ok(py: Path, code: str, cwd: Path = None) -> bool:
    return subprocess.run([str(py), "-c", code], capture_output=True, cwd=str(cwd or ROOT),
                          env=child_env()).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-torch", action="store_true")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--python", default=None, help="interpreter to build the venv with (default: this one)")
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    base = Path(args.python) if args.python else Path(sys.executable)
    if not args.python and sys.version_info < (3, 10):
        print(f"ERROR: this interpreter is {platform.python_version()}; pass --python <path to Python 3.10-3.13>")
        return 2
    if args.recreate and VENV.exists():
        shutil.rmtree(VENV)

    if os.name == "nt" and len(str(ROOT)) > 100:
        # site-packages paths inside .venv routinely exceed MAX_PATH-260 for torch/transformers.
        print(f"WARNING: this folder's path is {len(str(ROOT))} characters long. On Windows, enable long paths\n"
              f"         (Win+R -> gpedit, or `New-ItemProperty -Path \"HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\" "
              f"-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`)\n"
              f"         or move the project to something like C:\\rsc, or the torch install may fail.")

    if not venv_python().exists():
        print(f"creating {VENV} with {base}")
        if base == Path(sys.executable):
            venv.EnvBuilder(with_pip=True, clear=False).create(str(VENV))
        elif run([base, "-m", "venv", str(VENV)]) != 0:
            print("ERROR: could not create the virtual environment")
            return 2
    py = venv_python()

    # one portable spelling of the interpreter: .venv/Scripts/python.exe on every OS
    port = portable_python()
    if not port.exists():
        port.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(os.path.relpath(py, port.parent), port)
        except (OSError, NotImplementedError):  # no symlink permission: tiny launcher instead
            port.write_text(f'#!/bin/sh\nexec "{py}" "$@"\n', encoding="utf-8")
            os.chmod(port, 0o755)
    print(f"interpreter: {py}\nportable spelling: {port}")

    run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    if run([py, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]) != 0:
        print("ERROR: core requirements failed to install")
        return 2

    if not args.no_torch and not ok(py, "import torch"):
        if args.cpu_torch:
            run([py, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
        else:
            for idx in CUDA_INDEXES:  # RTX 50xx (sm_120) needs cu128 or newer
                print(f"installing torch from {idx}")
                if run([py, "-m", "pip", "install", "torch", "--index-url", idx]) == 0:
                    break
            else:
                print("WARNING: no CUDA wheel installed; falling back to the default index (may be CPU-only)")
                run([py, "-m", "pip", "install", "torch"])

    opt = ROOT / "requirements-optional.txt"
    if opt.exists():
        for pkg in opt.read_text(encoding="utf-8").splitlines():
            pkg = pkg.split("#")[0].strip()
            if pkg:
                run([py, "-m", "pip", "install", pkg])  # failures are fine

    # Everything the loop needs must import through the *portable* spelling, from the repo root.
    problems = [name for name, code in (("yaml", "import yaml"), ("numpy", "import numpy"),
                                        ("graphctl", "import graphctl, graphctl.cli"))
                if not ok(port if port.exists() else py, code)]
    if problems:
        print(f"ERROR: the new environment cannot import: {', '.join(problems)}")
        return 2

    info = {
        "python": str(py),
        "portable_python": str(port),
        "version": subprocess.run([str(py), "-c", "import platform;print(platform.python_version())"], capture_output=True, text=True).stdout.strip(),
        "torch": subprocess.run([str(py), "-c", "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"], capture_output=True, text=True).stdout.strip() or "not installed",
        "platform": platform.platform(),
        "created_at": __import__("datetime").datetime.now().replace(microsecond=0).isoformat(),
    }
    out = ROOT / "runs" / "env"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bootstrap.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    (out / "python_path.txt").write_text(str(py) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=1))
    if "True" not in info["torch"] and not args.no_torch and not args.cpu_torch:
        print("\nWARNING: torch reports no CUDA. Check the driver and re-run with a newer wheel index;\n"
              "         node N00's verification asserts gpu.cuda_available is True.")
    print("\nNext:  ./.venv/Scripts/python.exe -m graphctl next        (see AGENTS.md)")
    print("       (PowerShell and bash both accept that exact spelling; nothing depends on PATH.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
