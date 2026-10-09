# Anonymous artifact — Rank-Sufficient Calibration (WWW 2027 submission)

Reference implementation of the rank-sufficient critical values, the three calibration regimes, the online controller,
the replay evaluator, the live-latency and throughput harnesses, and the exhaustive tests that check every theorem of
the paper on small instances. Every number in the manuscript is a generated macro (`paper/numbers.tex`,
`paper/numbers_extra.tex`) whose source file is recorded next to it. `runs/provenance/registry.json` stores the
SHA-256 of the source of every number in `paper/numbers.tex`; paths are relative to the repository root, so the check
below runs from a fresh copy of this artifact.

## Reproduce without data or GPUs (minutes)
`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere:
```
python bootstrap.py --no-torch                                  # numpy/scipy only
./.venv/Scripts/python.exe -m pytest -q tests                   # 84 tests: brute force over every unsafe set for Theorems 1-2
                                                                # and Propositions 3-7, the hypergeometric identity of Theorem 10
./.venv/Scripts/python.exe scripts/smoke_pipeline.py --no-latex # synthetic end-to-end run of every script
./.venv/Scripts/python.exe -c "import sys; sys.path.insert(0, 'src'); from rscrank.provenance import verify_registry; print(verify_registry() or 'every registered number matches its shipped source')"
./.venv/Scripts/python.exe scripts/27_query_level_check.py      # thresholds, Table 1 witness risks and the committed-set
                                                                # results of Table 2, recomputed from query-level statistics
```

## Where each table and figure comes from
All of them are regenerated from the shipped result files, without data or GPUs:
```
python scripts/19_review3_tables_figs.py && python scripts/21_matched_numbers.py && python scripts/24_extra_numbers.py
```

| Paper | Script | Result files |
|---|---|---|
| Table 1 (witness risk) | `24_extra_numbers.py`; from query-level data, `27_query_level_check.py` | `runs/a8_split/grid_frontier.json` (from `22_grid_frontier.py`), `runs/a8_split/<teacher>/<domain>.json`, `runs/calib/<teacher>/<domain>/stats_{cal,test}_k10.jsonl` |
| Table 2, Table 4a, Figures 3-4 | `19_review3_tables_figs.py` | `runs/a8_split/<teacher>/<domain>.json`, `runs/stats/` |
| Table 3 | `19_review3_tables_figs.py` | `runs/throughput/`, `runs/throughput_rep/pass2|pass3/` |
| Table 4b | `24_extra_numbers.py` | `runs/stats/<teacher>/<domain>.json` |
| Table 4c | `12_paper_numbers.py` | `runs/data/splits_manifest.json` (split sizes; the post-freeze query ids are in `runs/a8_split/splits/`) |
| Table 4d | `21_matched_numbers.py` | `runs/matched_service/aggregate.json` |
| Proposition 6 numbers | `22_grid_frontier.py` -> `24_extra_numbers.py` | `runs/a8_split/grid_frontier.json` |
| Tuned micro-batch decomposition | `23_tuned_disagreement.py` (GPU) -> `24_extra_numbers.py` | `runs/latency_tuned_diff/<teacher>/<domain>.json` |
| Table 5, own-target scale head for B_all | `26_own_target_scale.py` -> `24_extra_numbers.py` | `runs/own_target/<teacher>/<domain>.json` (curve per configuration, B_all with three scale heads) |
| Fresh queries (Section 7, Appendix B) | `25_fresh_queries.py` (every stage needs the BEIR data; `features` and `teacher` need a GPU) -> `24_extra_numbers.py` | `runs/fresh/<teacher>/<domain>.json`, drawn ids in `runs/fresh/splits/`, pipeline manifests in `data/fresh/**/manifest.json` |
| Figures 1-2 | TikZ | `paper/figures/fig_example.tex`, `paper/figures/fig_overview.tex` |

## Full pipeline (data + GPU)
`scripts/00_env_check.py` … `scripts/13_build_packet.py`, in order; `configs/main.yaml` holds every setting
(protocol, thresholds, seeds). Data (BEIR NQ / HotpotQA / FiQA / SciFact, MS MARCO passage) and model weights
(Qwen3-Reranker-4B, bge-reranker-v2-m3, cross-encoder/ms-marco-MiniLM-L6-v2) are downloaded by the scripts and are
not included (size, licences). `docs/PREREGISTRATION.md` is the pre-registration with its append-only amendment log
(A1–A10); `docs/THEORY.md` the mathematical statements and proofs with pointers to the tests that check them.
`scripts/18_order_free_baselines.py` computes the set-completion prefixes, the declared-family envelopes and Bonferroni.
`scripts/18_order_free_baselines.py --frozen-family-split` is amendment A8: one redraw, seed 20261006, reused from
`runs/a8_split/splits/`, with outputs in `runs/a8_split/<teacher>/<domain>.json`. `scripts/19_review3_tables_figs.py`
reads those outputs when all six files exist.

## Results shipped
* `runs/stats/<teacher>/<domain>.json`, `runs/stats/summary.json` — all statistics, Clopper–Pearson intervals, paired
  bootstrap bounds, gates and the claim level.
* `runs/replay/<teacher>/<domain>/summary.json` — every replay experiment (E1–E9, E2b/E2c, E3b), all methods.
* `runs/latency/`, `runs/throughput/` — live GPU measurements (sequential latency; throughput under concurrency).
* `runs/policy_envelope/` — trajectory-specific critical values per order vs the rank-sufficient value (round 2).
* `runs/review3_presplit/` — the review-3 outputs from before the A8 redraw.
* `runs/a8_split/PROTOCOL.txt`, `runs/a8_split/splits/*.json` — the single post-freeze split (seed 20261006), written before metrics.
* `runs/a8_split/<teacher>/<domain>.json` and `runs/a8_split/headline_aggregate.json` — the finite-population call, risk, envelope and anatomy numbers in the current tables. `runs/review3/` currently holds a copy of these files so `scripts/12_paper_numbers.py` reads them.
* `runs/a8_split/grid_frontier.json` — the last-winner identity, the checkpoint-grid critical value and its threshold, and the all-subset witness risk at the three guardrail thresholds, on the same split.
* `runs/latency_tuned_diff/` — the 300 timed queries re-scored at the tuned micro-batch and compared with the canonical cache (order-only changes, membership changes, boundary gaps, nDCG@10).
* `runs/fresh/` — amendment A10 (i): 2,000 queries per domain drawn from the training-question sources after removing every frozen query, scored through the same pipeline and evaluated at the frozen post-freeze thresholds (risk with Clopper–Pearson interval, witness risk, calls, for every method).
* `runs/own_target/` — amendment A10 (ii)-(iii): B_all with a scale head fitted to its own target (next to the winners' head and sigma = 1), and the risk–calls curve for alpha in {0.01, 0.02, 0.05, 0.10, 0.20}.
* `runs/throughput_rep/pass2|pass3/` — the repeated timed passes of the C = 32 throughput cells, on the preceding split.
* `runs/matched_service/` — the matched serving run on the preceding split (submit-only set, full-pool set completion, RSC and the envelope; BM25 and the other cheap model; three passes at C = 32). Not re-timed after A8.
* `runs/calib/<teacher>/<domain>/lambdas.json` — calibrated thresholds for every k, alpha and regime.
* `runs/calib/<teacher>/<domain>/stats_{cal,test}_k10.jsonl` — one line per query of every pool (k = 10): A_k1–A_k3, B_top, B_all, H_k, the deployed-trajectory critical value, the budget-head prediction and the scale. The post-freeze split is a partition of these queries (`runs/a8_split/splits/`), so every threshold and witness risk can be recomputed without teacher scores.
* `paper/tables/*.tex` — all generated tables, including the full versions (`*_full.tex`) that did not fit the PDF.
* `runs/N200/` — the N = 200 sensitivity run (when present).

## Use of AI tools
Coding assistants implemented and ran the calibration, replay, analysis and figure code under the authors' direction,
and were also used to critique drafts internally (the amendment log in `docs/PREREGISTRATION.md` records which
changes followed). Every reported number is an output of that code, checked by the tests, the provenance registry and
the manuscript checker. No model selected queries, judged relevance, annotated documents, or is an author.
