"""Train the mean head, the query-shared scale head and the budget head for one
teacher; select hyper-parameters by *teacher calls at the target risk* on a
temporary split of the dev set (Sec. 10.6) -> models/heads/<teacher>/, runs/heads/<teacher>/report.json (node N08).

Training data: train+dev of the main domains (pooled).  The calibration and
test splits are never touched here.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, main_domains  # noqa: E402

from rscrank import core  # noqa: E402
from rscrank.assemble import bundles_from_split, load_split, pair_matrix, query_features_for_split  # noqa: E402
from rscrank.calibrate import calibrate_threshold  # noqa: E402
from rscrank.data import cheap_model_for, features_dir_for  # noqa: E402
from rscrank.heads import BudgetHead, MeanHead, ScaleHead, rho_target  # noqa: E402
from rscrank.io import ensure_dir, write_json  # noqa: E402
from rscrank.replay import MethodSpec, query_statistics, replay_dataset  # noqa: E402


def load_pool(cfg, teacher, split):
    parts = []
    fdir = features_dir_for(P(cfg, "features"), cheap_model_for(cfg, teacher), teacher)
    for d in main_domains(cfg):
        sd = load_split(fdir, P(cfg, "teacher") / teacher, d, split)
        if sd is not None and sd.t is not None:
            parts.append(sd)
    if not parts:
        raise SystemExit(f"no {split} data with teacher scores for {teacher}")
    return parts


def stacked(parts, k):
    Xs, ys, qs, offs = [], [], [], []
    off = 0
    for sd in parts:
        X, y, q = pair_matrix(sd, k + 1)
        Xs.append(X)
        ys.append(y)
        qs.append(q + off)
        offs.append(off)
        off += sd.Q
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(qs), offs


def sample_weights(q_index):
    # query-equal weighting: 1/N_q per pair
    counts = np.bincount(q_index)
    return 1.0 / counts[q_index]


def query_targets(parts, offs, mu, q_index, k):
    """rho(q), log H_k(q), sigma features z(q) for every query in parts."""
    Z, rho, logH, valid = [], [], [], []
    for sd, off in zip(parts, offs):
        Zs = query_features_for_split(sd, mu, q_index - off, k)  # local index
        for i in range(sd.Q):
            sel = np.where(q_index == i + off)[0]
            if len(sel) < k + 1:
                valid.append(False)
                Z.append(np.zeros(22)); rho.append(1.0); logH.append(0.0)
                continue
            s = sd.t["score"][i, : len(sel)].astype(float)
            m = mu[sel]
            top = core.true_topk(s, k)
            rho.append(rho_target(s, m, top))
            logH.append(math.log(core.H_k(s, m, k)))
            Z.append(Zs[i])
            valid.append(True)
    return np.asarray(Z), np.asarray(rho), np.asarray(logH), np.asarray(valid)


def main() -> int:
    ap = base_parser("train heads")
    ap.add_argument("--teacher", default=None)
    args = ap.parse_args()
    cfg = load_config(args)
    teacher = args.teacher or cfg["teachers"]["main"]
    k = cfg["protocol"]["k"]
    hc = cfg["heads"]
    train_parts, dev_parts = load_pool(cfg, teacher, "train"), load_pool(cfg, teacher, "dev")
    Xtr, ytr, qtr, offs_tr = stacked(train_parts, k)
    Xdv, ydv, qdv, offs_dv = stacked(dev_parts, k)
    log(f"{teacher}: train pairs {len(ytr)} ({len(np.unique(qtr))} queries); dev pairs {len(ydv)} ({len(np.unique(qdv))} queries)")
    wtr = sample_weights(qtr)
    # dev temp split for selection: half temp-cal, half temp-test (by query, seeded)
    rng = np.random.default_rng(cfg["seed"] + 7)
    dev_q = np.unique(qdv)
    perm = rng.permutation(len(dev_q))
    tcal_q = set(dev_q[perm[: len(dev_q) // 2]].tolist())
    report = {"teacher": teacher, "k": k, "grid": [], "selection_alpha": hc["selection_alpha"]}
    best = None
    out_dir = ensure_dir(P(cfg, "models") / "heads" / teacher)
    for lr, wd in itertools.product(hc["lr_grid"], hc["wd_grid"]):
        mh = MeanHead(Xtr.shape[1], hidden=hc["hidden"], kappa=hc["kappa"], seed=cfg["seed"])
        hist = []
        info = mh.fit(Xtr, ytr, Xdv, ydv, sample_weight=wtr, lr=lr, weight_decay=wd, epochs=hc["epochs"], patience=hc["patience"], batch_size=hc["batch_size"], seed=cfg["seed"], log=hist)
        mu_tr, mu_dv = mh.predict(Xtr), mh.predict(Xdv)
        Ztr, rho_tr, logH_tr, vtr = query_targets(train_parts, offs_tr, mu_tr, qtr, k)
        Zdv, rho_dv, logH_dv, vdv = query_targets(dev_parts, offs_dv, mu_dv, qdv, k)
        sh = ScaleHead(Ztr.shape[1], hidden=32, seed=cfg["seed"])
        sh.fit(Ztr[vtr], rho_tr[vtr], Zdv[vdv], rho_dv[vdv], lr=3e-3, epochs=max(hc["epochs"], 10), patience=hc["patience"], batch_size=64, seed=cfg["seed"])
        sig_dv = sh.predict(Zdv)
        # selection: calls at alpha on dev temp-test, with sigma head and with sigma = 1
        calls = {}
        risks = {}
        for use_sigma in (True, False):
            bundles = []
            for sd, off in zip(dev_parts, offs_dv):
                sig = sig_dv[off : off + sd.Q] if use_sigma else None
                bundles += bundles_from_split(sd, mu_dv, qdv - off, sig, k)
            # map bundle -> temp split by global query index
            gq = {}
            for sd, off in zip(dev_parts, offs_dv):
                for i, qid in enumerate(sd.qids):
                    gq[qid] = i + off
            cal_b = [b for b in bundles if gq[b.qid] in tcal_q]
            test_b = [b for b in bundles if gq[b.qid] not in tcal_q]
            A = [query_statistics(b, k)["A_k1"] for b in cal_b]
            lam = calibrate_threshold(A, hc["selection_alpha"], "marginal").threshold
            recs = replay_dataset(test_b, MethodSpec("sel", "guardrail", lam=lam, policy="mu", batch_first=cfg["protocol"]["batch_first"], batch_rest=cfg["protocol"]["batch_rest"]), k)
            calls[use_sigma] = float(np.mean([r["n_calls"] for r in recs]))
            risks[use_sigma] = float(np.mean([r["mismatch"] for r in recs]))
        g = {"lr": lr, "wd": wd, "best_dev_huber": info["best_dev_loss"], "epochs_run": info["epochs_run"], "calls_sigma_head": calls[True], "calls_sigma_one": calls[False], "risk_sigma_head": risks[True], "risk_sigma_one": risks[False], "history": hist}
        report["grid"].append(g)
        log(f"grid lr={lr} wd={wd}: dev huber {info['best_dev_loss']:.4f}; calls@alpha sigma-head {calls[True]:.2f} / sigma=1 {calls[False]:.2f}")
        score = min(calls[True], calls[False])
        if best is None or score < best[0]:
            best = (score, lr, wd, mh, sh, calls[True] <= calls[False], mu_tr, mu_dv, Ztr, Zdv, logH_tr, logH_dv, vtr, vdv)
    score, lr, wd, mh, sh, use_sigma, mu_tr, mu_dv, Ztr, Zdv, logH_tr, logH_dv, vtr, vdv = best
    # budget head (adaptive prefix baseline) on the selected mu
    bh = BudgetHead(Ztr.shape[1], hidden=32, seed=cfg["seed"])
    bh.fit(Ztr[vtr], logH_tr[vtr], Zdv[vdv], logH_dv[vdv], lr=3e-3, epochs=max(hc["epochs"], 10), patience=hc["patience"], batch_size=64, seed=cfg["seed"])
    # learned-budget constant c: smallest c on dev such that risk <= selection alpha (tuned, not calibrated)
    h_dv = bh.predict(Zdv)
    H_dv = np.exp(logH_dv)
    c_grid = np.linspace(1.0, 4.0, 31)
    c_star = None
    for c in c_grid:
        m = np.clip(np.ceil(c * np.exp(h_dv[vdv]) - 1e-9), k, None)
        if np.mean(H_dv[vdv] > m) <= hc["selection_alpha"]:
            c_star = float(c)
            break
    mh.save(out_dir / "mu.npz")
    sh.save(out_dir / "sigma.npz")
    bh.save(out_dir / "budget.npz")
    meta = {"teacher": teacher, "k": k, "lr": lr, "wd": wd, "use_sigma_head": bool(use_sigma), "learned_budget_c": c_star if c_star is not None else 4.0, "d_in": int(Xtr.shape[1]), "z_dim": int(Ztr.shape[1]), "hidden": hc["hidden"], "kappa": hc["kappa"], "train_pairs": int(len(ytr)), "dev_pairs": int(len(ydv))}
    write_json(out_dir / "meta.json", meta)
    report.update(selected=meta)
    write_json(ROOT / cfg["paths"]["runs"] / "heads" / teacher / "report.json", report)
    log(f"selected lr={lr} wd={wd} use_sigma_head={use_sigma} c={meta['learned_budget_c']}; heads -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
