"""Optional lever (pre-registered, node N08; cheap.distill in configs/main.yaml): distil the MiniLM
cross-encoder on TRAIN teacher scores -> models/cheap_distilled/<teacher>/ (Hugging Face format) + meta.json.

Regression of the cross-encoder logit onto the cached teacher score s(q, d) of every TRAIN pair of the
main (+extra) domains, Huber loss (delta 1.0), optional query-equal weighting (--query-equal), AdamW with
linear warm-up/decay, early stopping on the DEV Huber loss (the best dev epoch is what is saved).
Only the train and dev splits are ever read (asserted); cal and test are never touched.
Adoption is decided on dev by the rule in docs/PREREGISTRATION.md (cheap.distill.min_rel_gain); to use
the model set ``cheap.model: minilm-distilled`` and re-run scripts/04_cheap_features.py, which writes
per-teacher features to data/features_distilled/<teacher>/ (docs/MODELS.md).

  python scripts/06b_distill_cheap.py --teacher qwen3-4b
  python scripts/06b_distill_cheap.py --teacher qwen3-4b --max-train-pairs 300 --epochs 1   # quick check
--synthetic exits 0 with a "skipped in synthetic mode" message.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, load_config, log  # noqa: E402

from rscrank.data import distilled_model_dir, domain_list  # noqa: E402
from rscrank.io import ensure_dir, sha256_file, write_json  # noqa: E402
from rscrank.store import load_candidates, load_docstore, load_store  # noqa: E402

ALLOWED_SPLITS = ("train", "dev")  # never cal/test


def _doc_text(title: str, text: str) -> str:
    title = (title or "").strip()
    return f"{title} {text}".strip() if title else (text or "")


def load_pairs(cfg, teacher: str, split: str, doms: List[str], N: int) -> Tuple[List[Tuple[str, str]], np.ndarray, np.ndarray, Dict[str, str]]:
    """(query, doc_text) pairs, teacher targets, query ids (int) and input file hashes for one split."""
    assert split in ALLOWED_SPLITS, f"distillation may only read {ALLOWED_SPLITS}, not {split!r}"
    cand_dir, teacher_dir = P(cfg, "candidates"), P(cfg, "teacher") / teacher
    pairs, y, g, hashes = [], [], [], {}
    gid = 0
    for d in doms:
        cf, tf = cand_dir / d / f"{split}.jsonl", teacher_dir / d / f"{split}.npz"
        if not (cf.exists() and tf.exists()):
            continue
        assert cf.stem in ALLOWED_SPLITS and tf.stem in ALLOWED_SPLITS
        st = load_store(tf)
        rows = {r["qid"]: r for r in load_candidates(cf)}
        docstore = load_docstore(cand_dir / d / "docstore.jsonl")
        hashes[str(tf.relative_to(ROOT))] = sha256_file(tf)
        for qi, qid in enumerate(st["qids"]):
            r = rows[str(qid)]
            n = int(st["valid"][qi].sum())
            for j in range(n):
                s = float(st["score"][qi, j])
                if not math.isfinite(s):
                    continue
                t = docstore[str(st["docids"][qi, j])]
                pairs.append((r["query"], _doc_text(t["title"], t["text"])))
                y.append(s)
                g.append(gid)
            gid += 1
    return pairs, np.asarray(y, dtype=np.float32), np.asarray(g, dtype=np.int64), hashes


def weights_for(groups: np.ndarray, query_equal: bool) -> np.ndarray:
    if not query_equal or len(groups) == 0:
        return np.ones(len(groups), dtype=np.float32)
    cnt = np.bincount(groups)
    w = 1.0 / cnt[groups]
    return (w * len(w) / w.sum()).astype(np.float32)


def main() -> int:
    ap = base_parser("distil the cheap cross-encoder on train teacher scores")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-train-pairs", type=int, default=None, help="subsample train pairs (debug / quick check)")
    ap.add_argument("--max-dev-pairs", type=int, default=None)
    ap.add_argument("--query-equal", action="store_true", help="weight pairs by 1/(candidates of their query)")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=1, help="epochs without dev improvement before stopping")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="output directory (default models/cheap_distilled/<teacher>)")
    args = ap.parse_args()
    if args.synthetic:
        print("06b_distill_cheap: skipped in synthetic mode (optional lever; the smoke pipeline uses fake models)", flush=True)
        return 0
    cfg = load_config(args)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dcfg = dict(cfg["cheap"].get("distill", {}) or {})
    teacher = args.teacher or cfg["teachers"]["main"]
    epochs = int(args.epochs or dcfg.get("epochs", 2))
    lr = float(args.lr or dcfg.get("lr", 2e-5))
    bs = int(args.batch_size or dcfg.get("batch_size", 64))
    max_len = int(args.max_length or dcfg.get("max_length", 256))
    seed = int(cfg["seed"])
    base = cfg["cheap"].get("model_id", "cross-encoder/ms-marco-MiniLM-L6-v2")
    N = cfg["candidates"]["N"]
    doms = domain_list(cfg, include_cross=False, only=args.domains)  # main + extra (never cross-domain)
    out_dir = Path(args.out) if args.out else distilled_model_dir(P(cfg, "models"), teacher)
    log(f"distilling {base} on {teacher} train scores (domains {doms}); cheap.distill.enabled={dcfg.get('enabled')}")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr_pairs, tr_y, tr_g, tr_hash = load_pairs(cfg, teacher, "train", doms, N)
    dv_pairs, dv_y, dv_g, dv_hash = load_pairs(cfg, teacher, "dev", doms, N)
    if not tr_pairs or not dv_pairs:
        raise SystemExit(f"need cached TRAIN and DEV teacher scores for {teacher} (scripts/05_teacher_scores.py --splits train,dev)")
    if args.max_train_pairs and len(tr_pairs) > args.max_train_pairs:
        sel = np.sort(rng.choice(len(tr_pairs), size=args.max_train_pairs, replace=False))
        tr_pairs, tr_y, tr_g = [tr_pairs[i] for i in sel], tr_y[sel], tr_g[sel]
    if args.max_dev_pairs and len(dv_pairs) > args.max_dev_pairs:
        sel = np.sort(rng.choice(len(dv_pairs), size=args.max_dev_pairs, replace=False))
        dv_pairs, dv_y, dv_g = [dv_pairs[i] for i in sel], dv_y[sel], dv_g[sel]
    tr_w, dv_w = weights_for(tr_g, args.query_equal), weights_for(dv_g, args.query_equal)
    log(f"train pairs {len(tr_pairs)} ({len(np.unique(tr_g))} queries), dev pairs {len(dv_pairs)} ({len(np.unique(dv_g))} queries)")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base).to(device)
    huber = torch.nn.HuberLoss(reduction="none", delta=args.huber_delta)

    def encode(batch):
        enc = tok([q for q, _ in batch], [d for _, d in batch], padding=True, truncation="longest_first", max_length=max_len, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    def dev_loss() -> float:
        model.eval()
        order = np.argsort([len(d) for _, d in dv_pairs])  # length-sorted evaluation batches
        tot, wsum = 0.0, 0.0
        with torch.inference_mode():
            for s in range(0, len(order), bs):
                ix = order[s : s + bs]
                pred = model(**encode([dv_pairs[i] for i in ix])).logits.view(-1).float()
                y = torch.as_tensor(dv_y[ix], device=device)
                w = torch.as_tensor(dv_w[ix], device=device)
                tot += float((huber(pred, y) * w).sum())
                wsum += float(w.sum())
        return tot / max(wsum, 1e-12)

    steps_per_epoch = int(math.ceil(len(tr_pairs) / bs))
    total = max(1, steps_per_epoch * epochs)
    warm = max(1, int(0.1 * total))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_factor(t: int) -> float:  # linear warm-up over 10% of the steps, then linear decay to 0
        if t < warm:
            return (t + 1) / warm
        return max(0.0, (total - t) / max(1, total - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    ensure_dir(out_dir)
    (out_dir / "meta.json").unlink(missing_ok=True)  # meta.json marks a finished model; absent while training
    t0 = time.time()
    before = dev_loss()
    log(f"dev Huber before distillation: {before:.4f}")
    best, best_epoch, bad, history = math.inf, -1, 0, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr_pairs))
        run, wsum = 0.0, 0.0
        for s in range(0, len(perm), bs):
            ix = perm[s : s + bs]
            pred = model(**encode([tr_pairs[i] for i in ix])).logits.view(-1).float()
            y = torch.as_tensor(tr_y[ix], device=device)
            w = torch.as_tensor(tr_w[ix], device=device)
            loss = (huber(pred, y) * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach()) * float(w.sum())
            wsum += float(w.sum())
        dl = dev_loss()
        history.append({"epoch": ep + 1, "train_huber": run / max(wsum, 1e-12), "dev_huber": dl, "seconds": round(time.time() - t0, 1)})
        log(f"epoch {ep + 1}/{epochs}: train Huber {history[-1]['train_huber']:.4f}, dev Huber {dl:.4f}")
        if dl < best - 1e-6:
            best, best_epoch, bad = dl, ep + 1, 0
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        else:
            bad += 1
            if bad > args.patience:
                log("early stopping on dev Huber loss")
                break
    if best_epoch < 0:  # never improved on the base model: keep the last epoch but record it
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
    import transformers

    meta: Dict[str, Any] = {
        "teacher": teacher,
        "base_model": base,
        "domains": doms,
        "splits_used": list(ALLOWED_SPLITS),
        "teacher_stores": {**tr_hash, **dv_hash},
        "n_train_pairs": len(tr_pairs),
        "n_train_queries": int(len(np.unique(tr_g))),
        "n_dev_pairs": len(dv_pairs),
        "n_dev_queries": int(len(np.unique(dv_g))),
        "max_train_pairs": args.max_train_pairs,
        "target": "raw cached teacher score",
        "loss": f"Huber(delta={args.huber_delta})",
        "query_equal": bool(args.query_equal),
        "epochs_max": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "dev_huber_before": before,
        "best_dev_huber": best if best_epoch > 0 else None,
        "improved_on_base": bool(best_epoch > 0 and best < before),
        "history": history,
        "lr": lr,
        "batch_size": bs,
        "max_length": max_len,
        "seed": seed,
        "device": device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "seconds": round(time.time() - t0, 1),
    }
    write_json(out_dir / "meta.json", meta)
    log(f"saved distilled cheap model -> {out_dir} (best dev Huber {meta['best_dev_huber']}, before {before:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
