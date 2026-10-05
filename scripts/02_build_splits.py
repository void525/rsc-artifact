"""Freeze query splits (train/dev/cal/test) per domain -> runs/data/splits_manifest.json (node N03).

Domains: domains.main + domains.cross + domains.extra.  The cal/test pool of a domain is the BEIR
qrels split domains.test_pool_split[domain] (test; dev for msmarco).
Dedup happens here, before any retrieval or teacher scores exist.  Re-running
with the same seed reproduces identical files (hashes in the manifest).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, domains, load_config, log  # noqa: E402

from rscrank.data import domain_list, is_cross_domain_only, make_splits, pool_split  # noqa: E402
from rscrank.io import read_jsonl, sha256_file, write_json  # noqa: E402


def main() -> int:
    args = base_parser("build frozen splits").parse_args()
    cfg = load_config(args)
    raw, tq, out = P(cfg, "data_raw"), P(cfg, "train_questions"), P(cfg, "splits")
    man = {"seed": cfg["seed"], "domains": {}, "files": {}}
    doms = domain_list(cfg, only=args.domains) if not args.synthetic else domains(cfg, args)
    for d in doms:
        sc = cfg["splits"].get(d, {}) or {}
        cross = is_cross_domain_only(cfg, d)
        m = make_splits(d, raw, tq, out, cfg["seed"], n_cal=sc.get("n_cal", 0), n_train=0 if cross else sc.get("n_train", 0), n_dev=0 if cross else sc.get("n_dev", 0), test_pool_split=pool_split(cfg, d), cross_domain_only=cross, jaccard_threshold=float(cfg["splits"].get("dedup_jaccard", 0.8)))
        man["domains"][d] = m
        for f in sorted((out / d).glob("*.jsonl")):
            man["files"][str(f.relative_to(ROOT))] = {"sha256": sha256_file(f), "n": sum(1 for _ in read_jsonl(f))}
        log(f"{d}: {m}")
    # disjointness audit across all splits of all domains (by normalised text)
    from rscrank.io import normalize_text

    seen = {}
    clashes = []
    for d in doms:
        for split in ("train", "dev", "cal", "test"):
            f = out / d / f"{split}.jsonl"
            if not f.exists():
                continue
            for r in read_jsonl(f):
                key = normalize_text(r["text"])
                if key in seen and seen[key] != (d, split):
                    clashes.append({"text": r["text"][:80], "a": seen[key], "b": (d, split)})
                seen.setdefault(key, (d, split))
    man["cross_split_text_clashes"] = clashes[:50]
    man["n_clashes"] = len(clashes)
    man["ok"] = len([c for c in clashes if c["a"][0] == c["b"][0] and {c["a"][1], c["b"][1]} & {"cal", "test"} and {c["a"][1], c["b"][1]} & {"train", "dev"}]) == 0
    write_json(ROOT / cfg["paths"]["runs"] / "data" / "splits_manifest.json", man)
    log(f"splits manifest written; leakage-free={man['ok']} (clashes across domains are allowed and listed)")
    return 0 if man["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
