"""Figure of the open-loop serving runs (amendment A11): p95 latency against offered load, one panel per domain.

  python scripts/29_open_loop_figure.py   -> paper/figures/fig_open_loop.pdf (from runs/open_loop/<teacher>/<domain>.json)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, base_parser, load_config  # noqa: E402

from rscrank.io import read_json  # noqa: E402

COL = {"rsc": "#0B7A75", "sc": "#2F6DB5", "full": "#4B5563", "grid": "#D9DDE3", "text": "#2B2F36"}
DOMAINS = (("nq", "NQ"), ("hotpotqa", "HotpotQA"), ("msmarco", "MS~MARCO"))
SERIES = (
    ("full", "full teacher", dict(color=COL["full"], marker="s", ms=3.0, lw=0.9)),
    ("rsc", r"\textsc{RSC-Rank}, $16/8$", dict(color=COL["rsc"], marker="o", ms=2.8, lw=0.9, ls=(0, (2.2, 1.4)))),
    ("rsc_blockers", r"\textsc{RSC-Rank}, blocker", dict(color=COL["rsc"], marker="o", ms=3.4, lw=1.3)),
    ("prefix_adaptive", "adaptive prefix", dict(color=COL["sc"], marker="D", ms=3.0, lw=1.0)),
)


def main() -> int:
    ap = base_parser("open-loop figure")
    ap.add_argument("--teacher", default="qwen3-4b")
    args = ap.parse_args()
    cfg = load_config(args)
    src = ROOT / cfg["paths"]["runs"] / "open_loop" / args.teacher
    runs = {d: read_json(src / f"{d}.json") for d, _ in DOMAINS}
    import matplotlib

    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "font.family": "serif", "text.usetex": True,
                         "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}\usepackage[libertine]{newtxmath}",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.4, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0,
                         "ytick.major.size": 2.0, "ytick.minor.size": 1.2, "axes.edgecolor": "#6B7280",
                         "xtick.color": COL["text"], "ytick.color": COL["text"], "axes.labelcolor": COL["text"]})
    fig, axes = plt.subplots(1, 3, figsize=(3.33, 1.42), sharey=True, gridspec_kw={"wspace": 0.08})
    rhos = sorted(runs[DOMAINS[0][0]]["rhos"])
    lo = min(c["latency_p95_s"] for r in runs.values() for c in r["cells"].values())
    hi = max(c["latency_p95_s"] for r in runs.values() for c in r["cells"].values())
    for ax, (d, title) in zip(axes, DOMAINS):
        cells = runs[d]["cells"]
        for m, _, st in SERIES:
            pts = [cells.get(f"{m}@{rho:g}") for rho in rhos]
            xs = [rho for rho, c in zip(rhos, pts) if c]
            ys = [c["latency_p95_s"] for c in pts if c]
            ax.plot(xs, ys, color=st["color"], lw=st["lw"], ls=st.get("ls", "-"), marker=st["marker"], ms=st["ms"],
                    mfc=st["color"], mec="white", mew=0.4, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(rhos)
        ax.set_xticklabels([f"{r:g}" for r in rhos])
        ax.set_xlim(rhos[0] - 0.15, rhos[-1] + 0.15)
        ax.set_title(title, fontsize=6.8, pad=2.5, color=COL["text"])
        ax.grid(axis="y", which="major", color=COL["grid"], lw=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylim(lo / 1.25, hi * 1.25)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if lo / 1.25 <= t <= hi * 1.25]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([str(t) for t in ticks])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("p95 latency (s)")
    fig.text(0.53, -0.02, r"offered load ($\times$ closed-loop capacity of the full teacher)", ha="center", va="top", fontsize=7, color=COL["text"])
    handles = [Line2D([0], [0], color=st["color"], lw=st["lw"], ls=st.get("ls", "-"), marker=st["marker"], ms=st["ms"], mec="white", mew=0.4) for _, _, st in SERIES]
    labels = [lab for _, lab, _ in SERIES]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.53, 0.955), handletextpad=0.3,
               columnspacing=1.0, handlelength=1.8, borderaxespad=0.0)
    out = ROOT / "paper" / "figures" / "fig_open_loop.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
