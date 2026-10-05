"""Independent re-computation of the calibration thresholds written by the pipeline.

    python paper/tools/check_calibration.py [--runs-dir DIR] [--json] [--verbose]

For every ``<runs-dir>/calib/<teacher>/<domain>/lambdas.json`` this tool checks, with its
own arithmetic (it deliberately does **not** import ``rscrank.calibrate`` -- that module is
the thing being checked):

  1. the three order-statistic indices, recomputed from the stored (n, M, alpha, delta):
       marginal  j = ceil((n+1)(1-alpha))                          (exact rational)
       pac       j = min{ j : P[Bin(n, 1-alpha) >= j] <= delta }
       finite    j = min{ j : P[Hypergeom(n+M, M+j-t, n) >= j] <= delta },
                 t = floor(alpha M) + 1
  2. every stored threshold equals the corresponding order statistic of the stored
     per-query statistics in ``stats_cal_k<k>.jsonl`` (a_(j), with a_(n+1) = +inf);
  3. the index ordering marginal <= pac <= finite for each (k, alpha);
  4. A_k1 <= B_top <= B_all for every query of every stats file (Proposition 3).

Exit 1 on any mismatch, 0 otherwise; with no calibration files it prints
"nothing to check yet" and exits 0.  Stdlib + numpy, utf-8 everywhere, Windows and Linux.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REGIMES = ("marginal", "pac", "finite")
TOL = 1e-9          # absolute tolerance when a tail probability sits exactly on delta


# --------------------------------------------------------------------------- #
# own arithmetic: log-binomial coefficients and the two tail probabilities
# --------------------------------------------------------------------------- #
def _log_comb_row(n: int) -> np.ndarray:
    """log C(n, x) for x = 0..n, via log-gamma (no scipy, no rscrank)."""
    x = np.arange(n + 1, dtype=float)
    lg = np.vectorize(math.lgamma, otypes=[float])
    return lg(n + 1.0) - lg(x + 1.0) - lg(n - x + 1.0)


def binom_upper_tail(n: int, p: float) -> np.ndarray:
    """Array T with T[j] = P[Bin(n, p) >= j] for j = 0..n+1."""
    if p <= 0.0:
        t = np.zeros(n + 2)
        t[0] = 1.0
        return t
    if p >= 1.0:
        t = np.ones(n + 2)
        t[n + 1] = 0.0
        return t
    x = np.arange(n + 1, dtype=float)
    logpmf = _log_comb_row(n) + x * math.log(p) + (n - x) * math.log1p(-p)
    pmf = np.exp(logpmf)
    tail = np.zeros(n + 2)
    tail[:n + 1] = np.cumsum(pmf[::-1])[::-1]     # tail[j] = sum_{x>=j} pmf[x]
    return np.clip(tail, 0.0, 1.0)


def hypergeom_upper_tail(L: int, good: int, draws: int, j: int) -> float:
    """P[X >= j] for X ~ Hypergeom(population L, successes `good`, `draws` draws)."""
    good = max(0, min(good, L))
    lo = max(0, draws - (L - good))
    hi = min(draws, good)
    if j <= lo:
        return 1.0
    if j > hi:
        return 0.0
    x = np.arange(j, hi + 1, dtype=float)
    lg = np.vectorize(math.lgamma, otypes=[float])

    def lcomb(nn, kk):
        return lg(nn + 1.0) - lg(kk + 1.0) - lg(nn - kk + 1.0)

    logp = lcomb(float(good), x) + lcomb(float(L - good), draws - x) - math.lgamma(L + 1.0) + math.lgamma(draws + 1.0) + math.lgamma(L - draws + 1.0)
    return float(np.clip(np.exp(logp).sum(), 0.0, 1.0))


def marginal_index(n: int, alpha: float) -> int:
    """ceil((n+1)(1-alpha)) in exact rational arithmetic, clipped to 1..n+1."""
    a = Fraction(str(alpha))
    val = (n + 1) * (1 - a)
    j = -((-val.numerator) // val.denominator)      # ceil of a Fraction
    return min(max(int(j), 1), n + 1)


def pac_index(n: int, alpha: float, delta: float) -> Tuple[int, float]:
    tail = binom_upper_tail(n, 1.0 - alpha)
    for j in range(1, n + 2):
        if tail[j] <= delta + TOL:
            return j, float(tail[j])
    return n + 1, 0.0


def finite_index(n: int, M: int, alpha: float, delta: float) -> Tuple[int, float]:
    L = n + M
    from fractions import Fraction
    t = int(math.floor(Fraction(repr(float(alpha))) * M)) + 1  # exact: the decimal alpha denotes, as in rscrank.calibrate

    def p_of(j: int) -> float:
        return hypergeom_upper_tail(L, M + j - t, n, j)

    lo, hi = 1, n + 1                                # p_of is non-increasing in j
    while lo < hi:
        mid = (lo + hi) // 2
        if p_of(mid) <= delta + TOL:
            hi = mid
        else:
            lo = mid + 1
    if lo > 1 and p_of(lo - 1) <= delta + TOL:       # monotonicity was violated: linear scan
        for j in range(1, n + 2):
            if p_of(j) <= delta + TOL:
                return j, p_of(j)
    return lo, p_of(lo)


def order_statistic(values: Sequence[float], j: int) -> float:
    a = np.sort(np.asarray(values, dtype=float))
    return math.inf if j >= len(a) + 1 else float(a[j - 1])


# --------------------------------------------------------------------------- #
def read_jsonl(p: Path) -> List[dict]:
    rows = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_dominance(rows: List[dict], where: str, errors: List[str], checked: Dict[str, int]) -> None:
    bad = []
    for r in rows:
        a, bt, ba = r.get("A_k1"), r.get("B_top"), r.get("B_all")
        if a is None or bt is None or ba is None:
            continue
        checked["dominance"] += 1
        if not (a <= bt <= ba):
            bad.append(f"{r.get('qid', '?')}: A_k1={a!r} B_top={bt!r} B_all={ba!r}")
    if bad:
        errors.append(f"{where}: Proposition 3 ordering A_k1 <= B_top <= B_all violated on {len(bad)} query(ies): " + "; ".join(bad[:5]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="independently re-check the calibration thresholds")
    ap.add_argument("--runs-dir", default=os.environ.get("RSCRANK_RUNS_DIR", str(ROOT / "runs")))
    ap.add_argument("--frozen-dir", default=None, help="directory of the frozen predictions (default: data/frozen for the main runs dir, <runs>/frozen otherwise)")
    ap.add_argument("--json", action="store_true", help="print only the JSON report")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    runs = Path(a.runs_dir)
    if not runs.is_dir():
        print(json.dumps({"errors": [f"runs dir {runs} does not exist"], "warnings": [], "status": "error"}, indent=None if a.json else 1))
        return 1
    calib = runs / "calib"
    lam_files = sorted(calib.glob("*/*/lambdas.json")) if calib.is_dir() else []
    stats_files = sorted(calib.glob("*/*/stats_*.jsonl")) if calib.is_dir() else []
    if not lam_files and not stats_files:
        report = {"errors": [], "warnings": [], "status": "nothing to check yet",
                  "info": {"runs_dir": str(runs), "calib_dir": str(calib), "lambdas_files": 0}}
        print(json.dumps(report) if a.json else json.dumps(report, indent=1))
        if not a.json:
            print("RESULT: nothing to check yet (no", calib.as_posix(), "files)")
        return 0

    errors: List[str] = []
    warnings: List[str] = []
    checked = {"index": 0, "threshold": 0, "ordering": 0, "dominance": 0}
    details: List[dict] = []

    for lam in lam_files:
        domain_dir = lam.parent
        teacher, domain = domain_dir.parent.name, domain_dir.name
        tag = f"{teacher}/{domain}"
        try:
            L = json.loads(lam.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:  # noqa: BLE001
            errors.append(f"{tag}: unreadable lambdas.json ({e})")
            continue
        delta_cfg = L.get("delta_cfg")
        # ---- 0. the frozen predictions on disk must come from the same system as the thresholds ------
        fp = L.get("fingerprint")
        frozen_root = Path(a.frozen_dir) if a.frozen_dir else (runs.parent / "data" / "frozen" if runs.name == "runs" else runs / "frozen")
        frozen_dir = frozen_root / teacher / domain
        for split in ("cal", "test"):
            fzp = frozen_dir / f"{split}.npz"
            if not fzp.exists():
                warnings.append(f"{tag}: frozen predictions {fzp} not found (fingerprint check skipped)")
                continue
            try:
                import numpy as _np

                with _np.load(fzp) as fz:
                    if "fingerprint" not in fz.files:
                        errors.append(f"{tag}/{split}: frozen file has no fingerprint (stale file from an older 07 run?)")
                    else:
                        got = json.loads(str(fz["fingerprint"]))
                        if fp and got.get("heads_sha256") != fp.get("heads_sha256"):
                            errors.append(f"{tag}/{split}: frozen predictions come from heads {got.get('heads_sha256', '')[:12]} but lambdas.json from {fp.get('heads_sha256', '')[:12]} -- data/frozen was overwritten by another configuration")
                        if fp and got.get("cheap_model") != fp.get("cheap_model"):
                            errors.append(f"{tag}/{split}: cheap model {got.get('cheap_model')} in the frozen file vs {fp.get('cheap_model')} in lambdas.json")
                        checked["fingerprint"] = checked.get("fingerprint", 0) + 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"{tag}/{split}: cannot read frozen file ({e})")
        for kstr, kblock in sorted((L.get("k") or {}).items()):
            k = int(kstr)
            stats_path = domain_dir / f"stats_cal_k{k}.jsonl"
            if not stats_path.exists():
                errors.append(f"{tag} k={k}: {stats_path.name} is missing but lambdas.json has thresholds for it")
                continue
            rows = read_jsonl(stats_path)
            values: Dict[str, List[float]] = {}
            for r in rows:
                for key, v in r.items():
                    if isinstance(v, (int, float)) and key not in ("N", "k"):
                        values.setdefault(key, []).append(float(v))
            n_rows = len(rows)
            if kblock.get("n_cal") is not None and kblock["n_cal"] != n_rows:
                errors.append(f"{tag} k={k}: lambdas.json says n_cal={kblock['n_cal']} but {stats_path.name} has {n_rows} queries")
            for astr, ablock in sorted((kblock.get("alpha") or {}).items()):
                alpha = float(astr)
                idx_by_regime: Dict[str, int] = {}
                for regime in REGIMES:
                    rblock = ablock.get(regime) or {}
                    for stat, e in sorted(rblock.items()):
                        n = int(e["n"])
                        delta = e.get("delta")
                        if delta is None:
                            delta = delta_cfg
                        stored_idx = int(e["index"])
                        # ---- 1. the index ------------------------------------
                        if regime == "marginal":
                            want = marginal_index(n, alpha)
                            pval = None
                        elif regime == "pac":
                            want, pval = pac_index(n, alpha, float(delta))
                        else:
                            M = e.get("M") or kblock.get("M_test")
                            if M is None:
                                errors.append(f"{tag} k={k} alpha={alpha} {regime}/{stat}: no M for the finite regime")
                                continue
                            want, pval = finite_index(n, int(M), alpha, float(delta))
                        checked["index"] += 1
                        if want != stored_idx:
                            near = pval is not None and abs(pval - float(delta)) <= 1e-7
                            msg = (f"{tag} k={k} alpha={alpha} {regime}/{stat}: stored index {stored_idx}, "
                                   f"recomputed {want}" + (f" (tail p={pval:.12g} vs delta={delta})" if pval is not None else ""))
                            (warnings if near else errors).append(msg + (" [boundary case, within 1e-7 of delta]" if near else ""))
                        idx_by_regime.setdefault(regime, stored_idx)
                        # ---- 2. the threshold is the order statistic ---------
                        if stat not in values:
                            errors.append(f"{tag} k={k}: statistic {stat!r} has a threshold but no column in {stats_path.name}")
                            continue
                        if len(values[stat]) != n:
                            errors.append(f"{tag} k={k} {regime}/{stat}: n={n} in lambdas.json but {len(values[stat])} values in {stats_path.name}")
                            continue
                        want_thr = order_statistic(values[stat], stored_idx)
                        got_inf = bool(e.get("threshold_is_inf")) or bool(e.get("threshold_is_neg_inf"))
                        got_sign = 1 if e.get("threshold_is_inf") else -1
                        got_thr = (got_sign * math.inf) if (got_inf and e.get("threshold") is None) else e.get("threshold")
                        checked["threshold"] += 1
                        if got_inf:
                            if not (math.isinf(want_thr) and (want_thr > 0) == (got_sign > 0)):
                                errors.append(f"{tag} k={k} alpha={alpha} {regime}/{stat}: stored threshold is {'+' if got_sign > 0 else '-'}inf, order statistic a_({stored_idx}) = {want_thr!r}")
                        elif got_thr is None:
                            errors.append(f"{tag} k={k} alpha={alpha} {regime}/{stat}: threshold is null but threshold_is_inf is false")
                        elif math.isinf(want_thr) or not (float(got_thr) == want_thr or math.isclose(float(got_thr), want_thr, rel_tol=1e-12, abs_tol=1e-12)):
                            errors.append(f"{tag} k={k} alpha={alpha} {regime}/{stat}: stored threshold {got_thr!r} != a_({stored_idx}) = {want_thr!r}")
                # ---- 3. index ordering -----------------------------------
                if all(r in idx_by_regime for r in REGIMES):
                    checked["ordering"] += 1
                    m, pc, fi = (idx_by_regime[r] for r in REGIMES)
                    if not (m <= pc <= fi):
                        errors.append(f"{tag} k={k} alpha={alpha}: index ordering marginal<=pac<=finite violated ({m}, {pc}, {fi})")
                    details.append({"teacher": teacher, "domain": domain, "k": k, "alpha": alpha,
                                    "marginal": m, "pac": pc, "finite": fi})
            check_dominance(rows, f"{tag} {stats_path.name}", errors, checked)

    # ---- 4. dominance on every other stats file (test splits, cross-domain) --
    for sp in stats_files:
        if sp.name.startswith("stats_cal_k"):
            continue                                   # already done above
        tag = f"{sp.parent.parent.name}/{sp.parent.name}"
        try:
            check_dominance(read_jsonl(sp), f"{tag} {sp.name}", errors, checked)
        except (ValueError, OSError) as e:  # noqa: BLE001
            errors.append(f"{tag} {sp.name}: unreadable ({e})")

    report = {
        "errors": errors,
        "warnings": warnings,
        "status": "ok" if not errors else "mismatch",
        "info": {
            "runs_dir": str(runs),
            "lambdas_files": len(lam_files),
            "stats_files": len(stats_files),
            "checks": checked,
            "indices": details if a.verbose else details[:8],
        },
    }
    if a.json:
        print(json.dumps(report))
    else:
        print(json.dumps(report, indent=1))
        print("RESULT:", "OK" if not errors else f"{len(errors)} MISMATCH(ES)")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
