"""Fresh queries that no design decision has seen (amendment A10 (i)) -> data/fresh/, runs/fresh/

Per main domain, 2000 queries are drawn from the training-question source after removing every query of the frozen
splits, then go through the pipeline of the main tables with nothing refitted: BM25 candidates with the effective
backend and parameters of scripts/03_build_candidates.py, cheap features with the cheap model adopted for each teacher,
canonical-shape teacher scores, the frozen heads, and the thresholds frozen on the post-freeze (A8) calibration set.

  python scripts/25_fresh_queries.py --stage splits
  python scripts/25_fresh_queries.py --stage candidates
  python scripts/25_fresh_queries.py --stage features
  python scripts/25_fresh_queries.py --stage teacher --teacher qwen3-4b      (resumable)
  python scripts/25_fresh_queries.py --stage teacher --teacher bge-m3
  python scripts/25_fresh_queries.py --stage eval                            -> runs/fresh/<teacher>/<domain>.json

Every stage skips outputs that already exist for the same inputs.  Nothing here fits a head, chooses a threshold or
changes the protocol.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log, main_domains, teachers  # noqa: E402

from rscrank.baselines import adaptive_prefix_budget  # noqa: E402
from rscrank.calibrate import calibrate_threshold, family_delta  # noqa: E402
from rscrank.candidates import needed_docids, write_candidates, write_docstore  # noqa: E402
from rscrank.data import cheap_model_for, dedup, distilled_model_dir, features_dir_for  # noqa: E402
from rscrank.heads import BudgetHead, MeanHead, ScaleHead  # noqa: E402
from rscrank.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from rscrank.models import get_cheap, get_teacher  # noqa: E402
from rscrank.replay import MethodSpec, query_statistics, replay_query  # noqa: E402
from rscrank.stats import risk_summary  # noqa: E402
from rscrank.store import load_candidates, load_docstore, save_store, store_meta  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


s03 = _load("s03", "03_build_candidates.py")
s04 = _load("s04", "04_cheap_features.py")
s05 = _load("s05", "05_teacher_scores.py")
s07 = _load("s07", "07_freeze_calibrate.py")
rep08 = _load("rep08", "08_replay_eval.py")
s18 = _load("s18", "18_order_free_baselines.py")

FRESH = ROOT / "data" / "fresh"
SPLIT = "fresh"
FRESH_SEED = 20261009  # declared in amendment A10 before any fresh query was drawn
DOMAIN_CODE = {"nq": 1, "hotpotqa": 2, "msmarco": 3}  # the domain codes of the A8 redraw
M_FRESH = 2000


def fresh_cfg(cfg):
    """The same configuration with every data path of the fresh split under data/fresh/."""
    c = copy.deepcopy(cfg)
    for key in ("candidates", "features", "teacher", "frozen"):
        c["paths"][key] = str(FRESH / key)
    return c


def _free_gpu():
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def stage_splits(cfg):
    man_path = FRESH / "splits" / "manifest.json"
    man = read_json(man_path) if man_path.exists() else {"seed": FRESH_SEED, "n_per_domain": M_FRESH, "jaccard_threshold": 0.8, "domains": {}}
    for d in main_domains(cfg):
        out = FRESH / "splits" / d / f"{SPLIT}.jsonl"
        if out.exists():
            log(f"{d}: {out.relative_to(ROOT)} exists; not redrawn")
            continue
        frozen = [r for s in ("train", "dev", "cal", "test") for r in read_jsonl(P(cfg, "splits") / d / f"{s}.jsonl")]
        frozen_ids = {r["qid"] for r in frozen}
        source_all = list(read_jsonl(P(cfg, "train_questions") / f"{d}.jsonl"))
        source = [r for r in source_all if r["qid"] not in frozen_ids]
        rng = np.random.default_rng(np.random.SeedSequence([FRESH_SEED, DOMAIN_CODE[d]]))
        perm = rng.permutation(len(source))
        examined = [source[i] for i in perm[: 4 * M_FRESH]]
        kept, st = dedup(frozen, examined, 0.8)
        if len(kept) < M_FRESH:
            raise SystemExit(f"{d}: only {len(kept)} of {len(examined)} examined queries survive de-duplication")
        rows = [{"qid": r["qid"], "text": r["text"], "source": r.get("source", "")} for r in kept[:M_FRESH]]
        write_jsonl(out, rows)
        man["domains"][d] = {"source_size": len(source_all), "after_removing_frozen_ids": len(source), "examined_in_draw_order": len(examined),
                             "dedup_on_examined": st, "n": len(rows), "sha256": sha256_file(out)}
        write_json(man_path, man)
        log(f"{d}: {len(rows)} fresh queries (examined {len(examined)}, dedup {st})")


def stage_candidates(cfg):
    raw, N = P(cfg, "data_raw"), int(cfg["candidates"]["N"])
    main_man = read_json(ROOT / cfg["paths"]["runs"] / "candidates" / "manifest.json")
    out_dir = FRESH / "candidates"
    man_path = out_dir / "manifest.json"
    man = read_json(man_path) if man_path.exists() else {"N": N, "domains": {}}
    for d in main_domains(cfg):
        cf, ds = out_dir / d / f"{SPLIT}.jsonl", out_dir / d / "docstore.jsonl"
        if cf.exists() and ds.exists():
            log(f"{d}: fresh candidates exist")
            continue
        backend = main_man["domains"][d]["backend"]  # the backend the main tables actually used
        queries = {r["qid"]: r["text"] for r in read_jsonl(FRESH / "splits" / d / f"{SPLIT}.jsonl")}
        t0 = time.time()
        hits, info = s03.run_domain(cfg, backend, raw, d, queries, N)
        f = write_candidates(out_dir, d, SPLIT, queries, hits)
        need = needed_docids([f])
        write_docstore(out_dir, d, s03.corpus_iter(raw, d, True), need)
        short = sum(1 for q in queries if len(hits.get(q, [])) < N)
        man["domains"][d] = {"backend": info.get("backend", backend), "n_queries": len(queries), "n_docstore": len(need),
                             "queries_with_fewer_than_N": short, "seconds": round(time.time() - t0, 1),
                             "files": {str(p.relative_to(ROOT)): sha256_file(p) for p in (f, ds)}}
        write_json(man_path, man)
        log(f"{d}: candidates for {len(queries)} fresh queries ({short} with fewer than {N}; {man['domains'][d]['seconds']} s)")


def stage_features(cfg):
    N = int(cfg["candidates"]["N"])
    ccfg = {k: v for k, v in cfg["cheap"].items() if k != "distill"}
    built = set()
    for t in teachers(cfg):
        name = cheap_model_for(cfg, t)
        fdir = features_dir_for(FRESH / "features", name, t)
        if fdir in built:
            continue
        if name == "minilm-distilled":
            mdir = distilled_model_dir(P(cfg, "models"), t)
            model = get_cheap("minilm-distilled", {**ccfg, "path": str(mdir), "name": f"minilm-distilled:{t}"})
            extra = {"teacher": t, "distilled_meta_sha256": sha256_file(mdir / "meta.json"), "distilled_from": read_json(mdir / "meta.json").get("base_model")}
        else:
            model = get_cheap(name, ccfg)
            extra = {}
        for d in main_domains(cfg):
            cf = FRESH / "candidates" / d / f"{SPLIT}.jsonl"
            out = fdir / d / f"{SPLIT}.npz"
            h = sha256_file(cf)
            if out.exists() and store_meta(out).get("cand_sha256") == h and store_meta(out).get("model") == model.name:
                log(f"skip existing {out.relative_to(ROOT)}")
                continue
            rows = load_candidates(cf)
            docstore = load_docstore(FRESH / "candidates" / d / "docstore.jsonl")
            qids, docids, arrays, n_pairs, secs = s04.build_split(model, rows, docstore, N, 64, f"{name}:{d}/{SPLIT}")
            save_store(out, qids, docids, arrays, N, {"model": model.name, "cand_sha256": h, **extra})
            log(f"wrote {out.relative_to(ROOT)} ({len(rows)} queries, {n_pairs} pairs, {n_pairs / max(secs, 1e-9):.0f} pairs/s)")
        built.add(fdir)
        del model
        _free_gpu()


def stage_teacher(cfg, tname, chunk_q):
    N = int(cfg["candidates"]["N"])
    model = get_teacher(tname, dict(cfg["teachers"].get(tname, {}) or {}))
    want = s05.settings_key(s05.settings_of(model))
    for d in main_domains(cfg):
        ref = s05.settings_key(store_meta(P(cfg, "teacher") / tname / d / "test.npz").get("settings", {}))
        if ref != want:
            raise SystemExit(f"{tname}/{d}: teacher settings {want} differ from those of the main cache {ref}")
    man_path = FRESH / "teacher" / tname / "manifest.json"
    man = read_json(man_path) if man_path.exists() else {"teacher": tname, "settings_key": want, "files": {}}
    for d in main_domains(cfg):
        cf = FRESH / "candidates" / d / f"{SPLIT}.jsonl"
        out = FRESH / "teacher" / tname / d / f"{SPLIT}.npz"
        h = sha256_file(cf)
        if out.exists() and store_meta(out).get("cand_sha256") == h:
            log(f"skip existing {out.relative_to(ROOT)}")
            continue
        rows = load_candidates(cf)
        docstore = load_docstore(FRESH / "candidates" / d / "docstore.jsonl")
        n_pairs, secs, notes = s05.score_split(model, rows, docstore, N, out, chunk_q, f"{tname}/{d}/{SPLIT}", h, {})
        man["files"][str(out.relative_to(ROOT))] = {"sha256": sha256_file(out), "queries": len(rows), "pairs": n_pairs, "seconds": round(secs, 1), **notes}
        write_json(man_path, man)


def _risk(mis):
    s = risk_summary(list(mis))
    return {"risk": s["rate"], "cp_lo": s["cp_lo"], "cp_hi": s["cp_hi"], "failures": s["failures"], "n": s["n"]}


def stage_eval(cfg):
    pr = cfg["protocol"]
    k, a0 = int(pr["k"]), float(pr["alpha_main"])
    primary = pr.get("primary_regime", "finite")
    delta_cfg = family_delta(pr["delta_family"], pr["n_main_configs"])
    bf, br, seed = pr["batch_first"], pr["batch_rest"], int(cfg["seed"])
    cfg_f = fresh_cfg(cfg)
    a8 = ROOT / cfg["paths"]["runs"] / "a8_split"
    for d in main_domains(cfg):  # the drawn identifiers, shipped with the results (the texts stay in data/fresh/)
        rows = list(read_jsonl(FRESH / "splits" / d / f"{SPLIT}.jsonl"))
        write_json(ROOT / cfg["paths"]["runs"] / "fresh" / "splits" / f"{d}.json",
                   {"seed": FRESH_SEED, "domain": d, "n": len(rows), "source": rows[0]["source"], "qids": [r["qid"] for r in rows]})
    for teacher in teachers(cfg):
        hd = P(cfg, "models") / "heads" / teacher
        meta = read_json(hd / "meta.json")
        heads = {"mu": MeanHead.load(hd / "mu.npz", meta["kappa"]), "sigma": ScaleHead.load(hd / "sigma.npz"), "budget": BudgetHead.load(hd / "budget.npz"), "meta": meta}
        for d in main_domains(cfg):
            s07.freeze_split(cfg_f, teacher, heads, d, SPLIT, k)
            lams = read_json(ROOT / cfg["paths"]["runs"] / "calib" / teacher / d / "lambdas.json")
            cal = rep08.load_bundles(cfg, teacher, d, "cal", k, lams=lams)
            test = rep08.load_bundles(cfg, teacher, d, "test", k, lams=lams)
            cal, test = s18.frozen_family_resplit(cal, test, d, a8 / "splits")
            fresh = rep08.load_bundles(cfg_f, teacher, d, SPLIT, k, lams=lams)
            saved = read_json(a8 / teacher / f"{d}.json")
            M = len(test)
            st_cal = [query_statistics(b, k) for b in cal]
            A_cal = [r["A_k1"] for r in st_cal]
            lam = {"rsc_finite": s18.calib(A_cal, a0, primary, delta_cfg, M),
                   "rsc_pac": calibrate_threshold(A_cal, a0, "pac", delta=delta_cfg).threshold,
                   "rsc_marginal": calibrate_threshold(A_cal, a0, "marginal").threshold,
                   "traj": float(saved["thresholds"]["lambda_mu"]),
                   "env13": float(saved["envelope"]["all13"]["lambda"]),
                   "btop": s18.calib([r["B_top"] for r in st_cal], a0, primary, delta_cfg, M),
                   "ball": s18.calib([r["B_all"] for r in st_cal], a0, primary, delta_cfg, M)}
            if abs(lam["rsc_finite"] - float(saved["thresholds"]["lambda_rsc"])) > 1e-12:
                raise SystemExit(f"{teacher}/{d}: RSC threshold does not match the stored post-freeze value")

            def replay(bundles, lam_):
                return [replay_query(b, MethodSpec("f", "guardrail", lam=lam_, policy="mu", policy_seed=seed, batch_first=bf, batch_rest=br), k) for b in bundles]

            for name in ("btop", "ball"):  # the deployed coverage thresholds reproduce the stored A8 headline
                calls_test = float(np.mean([r["n_calls"] for r in replay(test, lam[name])]))
                if abs(calls_test - float(saved["headline"][name]["calls"])) > 1e-9:
                    raise SystemExit(f"{teacher}/{d}: {name} threshold does not reproduce the stored A8 calls")
            st_f = [query_statistics(b, k) for b in fresh]
            A_f = np.array([r["A_k1"] for r in st_f], float)
            out = {"teacher": teacher, "domain": d, "k": k, "alpha": a0, "delta_cfg": delta_cfg, "n_fresh": len(fresh),
                   "n_cal": len(cal), "n_test_a8": M, "thresholds_from": "A8 calibration set", "lambda": lam, "methods": {}}
            for name, lam_ in lam.items():
                recs = replay(fresh, lam_)
                out["methods"][name] = {**_risk([r["mismatch"] for r in recs]), "calls": float(np.mean([r["n_calls"] for r in recs])),
                                        "witness": float(np.mean(A_f > lam_)),
                                        "early_fraction": float(np.mean([r["early_stopped"] for r in recs]))}
            t_H, m_uni = float(saved["thresholds"]["t_H"]), int(saved["thresholds"]["m_uniform"])
            truth = [set(b.true_topk_ids(k)) for b in fresh]
            m_ad = [adaptive_prefix_budget(float(b.features["h_pred"][0]), t_H, k, b.N) for b in fresh]
            out["methods"]["adaptive_once"] = {**_risk([float(not t.issubset(s18.mu_prefix(b, m))) for t, b, m in zip(truth, fresh, m_ad)]), "calls": float(np.mean(m_ad))}
            out["methods"]["uniform_once"] = {**_risk([float(not t.issubset(s18.mu_prefix(b, m_uni))) for t, b in zip(truth, fresh)]), "calls": float(m_uni)}
            # where the fresh queries sit relative to the post-freeze test set
            A_t = np.array([query_statistics(b, k)["A_k1"] for b in test], float)
            out["shift"] = {"witness_rsc_finite_test_a8": float(np.mean(A_t > lam["rsc_finite"])),
                            "A_k1_median_fresh": float(np.median(A_f)), "A_k1_median_test_a8": float(np.median(A_t)),
                            "H_k_mean_fresh": float(np.mean([r["H_k"] for r in st_f])),
                            "H_k_mean_test_a8": float(np.mean([query_statistics(b, k)["H_k"] for b in test]))}
            path = ROOT / cfg["paths"]["runs"] / "fresh" / teacher / f"{d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, out)
            m = out["methods"]
            log(f"{teacher}/{d}: fresh n={len(fresh)} | RSC deployed risk {m['rsc_finite']['risk']:.4f} (CP hi {m['rsc_finite']['cp_hi']:.4f}) "
                f"witness {m['rsc_finite']['witness']:.4f} calls {m['rsc_finite']['calls']:.1f} | PAC {m['rsc_pac']['risk']:.4f} | "
                f"traj {m['traj']['risk']:.4f}/{m['traj']['witness']:.4f} | env {m['env13']['risk']:.4f}/{m['env13']['witness']:.4f} | "
                f"adaptive prefix {m['adaptive_once']['risk']:.4f} at {m['adaptive_once']['calls']:.1f}")


def main() -> int:
    ap = base_parser("fresh queries never used in any design decision")
    ap.add_argument("--stage", required=True, choices=["splits", "candidates", "features", "teacher", "eval"])
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--chunk-queries", type=int, default=32)
    args = ap.parse_args()
    cfg = load_config(args)
    if args.stage == "splits":
        stage_splits(cfg)
    elif args.stage == "candidates":
        stage_candidates(cfg)
    elif args.stage == "features":
        stage_features(cfg)
    elif args.stage == "teacher":
        for t in ([args.teacher] if args.teacher else teachers(cfg)):
            stage_teacher(cfg, t, args.chunk_queries)
            _free_gpu()
    else:
        stage_eval(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
