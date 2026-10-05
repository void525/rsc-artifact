# Anonymous artifact — Rank-Sufficient Calibration (WWW 2027 submission)

Reference implementation of the rank-sufficient critical values, the three calibration regimes, the online controller,
the replay evaluator, the live-latency and throughput harnesses, and the exhaustive tests that check every theorem of
the paper on small instances. Every number in the manuscript is a generated macro (`paper/numbers.tex`) whose source
file and hash are recorded in `runs/provenance/registry.json` / `paper/numbers.json`.

## Reproduce without data or GPUs (minutes)
```
python bootstrap.py --no-torch                                  # numpy/scipy only
./.venv/Scripts/python.exe -m pytest -q tests                   # 160+ tests, incl. brute-force checks of Theorems 1-3, 6-8
./.venv/Scripts/python.exe scripts/smoke_pipeline.py --no-latex # synthetic end-to-end run of every script
```

## Full pipeline (data + GPU)
`scripts/00_env_check.py` … `scripts/13_build_packet.py`, in order; `configs/main.yaml` holds every setting
(protocol, thresholds, seeds). Data (BEIR NQ / HotpotQA / FiQA / SciFact, MS MARCO passage) and model weights
(Qwen3-Reranker-4B, bge-reranker-v2-m3, cross-encoder/ms-marco-MiniLM-L6-v2) are downloaded by the scripts and are
not included (size, licences). `docs/PREREGISTRATION.md` is the pre-registration with its append-only amendment log
(A1–A7); `docs/THEORY.md` the mathematical statements and proofs with pointers to the tests that check them.
`scripts/18_order_free_baselines.py` (amendment A7) computes the set-completion prefixes under every order, the
envelopes over declared policy families (blocker schedules through `rscrank.core.blocker_critical_value`), per-policy
Bonferroni at (alpha/K, delta/K) and the same-sample anatomy comparison; `scripts/19_review3_tables_figs.py` renders
the corresponding tables and figures.

## Results shipped
* `runs/stats/<teacher>/<domain>.json`, `runs/stats/summary.json` — all statistics, Clopper–Pearson intervals, paired
  bootstrap bounds, gates and the claim level.
* `runs/replay/<teacher>/<domain>/summary.json` — every replay experiment (E1–E9, E2b/E2c, E3b), all methods.
* `runs/latency/`, `runs/throughput/` — live GPU measurements (sequential latency; throughput under concurrency).
* `runs/policy_envelope/` — trajectory-specific critical values per order vs the rank-sufficient value (round 2).
* `runs/review3/<teacher>/<domain>.json` — set-completion prefixes, RSC, envelope and trajectory thresholds under the
  ten orders (paired differences), the envelopes over the ten orders and the thirteen declared policies, the blocker
  closed-form check, Bonferroni at (alpha/K, delta/K), and the covered-query comparison.
* `runs/throughput_rep/pass2|pass3/` — the repeated timed passes of the C = 32 throughput cells.
* `runs/calib/<teacher>/<domain>/lambdas.json` — calibrated thresholds for every k, alpha and regime.
* `paper/tables/*.tex` — all generated tables, including the full versions (`*_full.tex`) that did not fit the PDF.
* `runs/N200/` — the N = 200 sensitivity run (when present).

## Use of AI tools
The code and the experiment runs were produced by agentic LLM coding assistants under author supervision inside this
repository, with machine checks (tests, provenance registry, manuscript checker). No LLM was used as a judge,
annotator, reviewer or author.
