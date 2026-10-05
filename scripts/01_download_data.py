"""Download BEIR datasets and training questions -> runs/data/download_manifest.json (node N03).

Domains: domains.main + domains.cross + domains.extra (e.g. [msmarco]).  Sizes are checked against
the published BEIR numbers of the cal/test pool split (domains.test_pool_split: msmarco -> dev).
Training questions are cached for main and extra domains.  Sources: UKP zip mirror, fallback
Hugging Face hub (BeIR/<name>, BeIR/<name>-qrels; google-research-datasets/nq_open), recorded per domain.

--synthetic writes a tiny BEIR-format dataset per domain under runs/smoke/ so that
the entire pipeline can be exercised without network or GPU.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, P, base_parser, domains, load_config, log  # noqa: E402

from rscrank.data import BEIR_CORPUS_SIZE, BEIR_TEST_QUERIES, domain_list, download_beir_info, load_beir, load_train_questions, pool_split  # noqa: E402
from rscrank.io import ensure_dir, write_jsonl, write_json  # noqa: E402

VOCAB = [f"w{i}" for i in range(400)] + ["web", "search", "query", "page", "site", "link", "user", "data", "index", "rank", "news", "video", "music", "game", "phone", "city", "river", "mountain", "book", "film"]


def make_synthetic_domain(domain: str, out: Path, rng: np.random.Generator, n_docs: int, n_queries: int, n_train_q: int) -> None:
    ensure_dir(out / "qrels")
    docs = []
    for i in range(n_docs):
        L = int(rng.integers(20, 120))
        words = rng.choice(VOCAB, size=L)
        docs.append({"_id": f"{domain}-doc{i}", "title": f"Title {i} " + " ".join(rng.choice(VOCAB, 2)), "text": " ".join(words)})
    write_jsonl(out / "corpus.jsonl", docs)
    queries, qrels_test, qrels_train = [], [], []
    for j in range(n_queries + n_train_q):
        src = int(rng.integers(0, n_docs))
        dw = docs[src]["text"].split()
        q = " ".join(rng.choice(dw, size=min(6, len(dw)), replace=False))
        qid = f"{domain}-q{j}"
        queries.append({"_id": qid, "text": q})
        rel = [(src, 1)]
        if rng.random() < 0.3:
            rel.append((int(rng.integers(0, n_docs)), 1))
        target = qrels_test if j < n_queries else qrels_train
        for d, g in rel:
            target.append((qid, docs[d]["_id"], g))
    write_jsonl(out / "queries.jsonl", queries)
    for name, rows in (("test", qrels_test), ("train", qrels_train)):
        with open(out / "qrels" / f"{name}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["query-id", "corpus-id", "score"])
            w.writerows(rows)


def main() -> int:
    args = base_parser("download data").parse_args()
    cfg = load_config(args)
    raw = P(cfg, "data_raw")
    tq = P(cfg, "train_questions")
    man = {"synthetic": bool(args.synthetic), "domains": {}}
    if args.synthetic:
        rng = np.random.default_rng(cfg["seed"])
        for d in domains(cfg, args):
            make_synthetic_domain(d, raw / d, rng, n_docs=500, n_queries=180 if d in cfg["domains"]["main"] else 60, n_train_q=300 if d in cfg["domains"]["main"] else 0)
            man["domains"][d] = {"n_docs": 500}
        # NQ train questions in smoke mode come from the synthetic train qrels (like hotpotqa)
        ensure_dir(tq)
        for d in cfg["domains"]["main"]:
            _, queries, qrels = load_beir(d, raw, splits=("train",), load_corpus=False)
            rows = [{"qid": q, "text": queries[q], "source": f"synthetic/{d}/train"} for q in sorted(qrels.get("train", {}))]
            write_jsonl(tq / f"{d}.jsonl", rows)
            man["domains"][d]["train_questions"] = len(rows)
    else:
        for d in domain_list(cfg, only=args.domains):
            log(f"downloading BEIR {d}")
            out, prov = download_beir_info(d, raw)
            ps = pool_split(cfg, d)
            _, queries, qrels = load_beir(d, raw, splits=(ps,), load_corpus=False)
            n_test = len([q for q in qrels.get(ps, {}) if q in queries])
            with open(out / "corpus.jsonl", "r", encoding="utf-8") as f:
                n_corpus = sum(1 for _ in f)
            man["domains"][d] = {"pool_split": ps, "n_test_queries": n_test, "expected_test": BEIR_TEST_QUERIES.get(d), "n_corpus": n_corpus, "expected_corpus": BEIR_CORPUS_SIZE.get(d), "provenance": prov}
            if n_test != BEIR_TEST_QUERIES.get(d) or n_corpus != BEIR_CORPUS_SIZE.get(d):
                log(f"WARNING: {d} sizes differ from published BEIR sizes; record and continue (the frozen manifest is the truth for the paper)")
        for d in domain_list(cfg, include_cross=False, only=args.domains):
            log(f"training questions for {d}")
            rows = load_train_questions(d, raw, tq)
            man["domains"][d]["train_questions"] = len(rows)
    out = ROOT / cfg["paths"]["runs"] / "data" / "download_manifest.json"
    write_json(out, man)
    log(f"manifest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
