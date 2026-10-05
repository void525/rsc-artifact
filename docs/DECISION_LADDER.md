# DECISION_LADDER.md — pre-approved degradations (applied by `python -m graphctl ladder --apply`)

Order of cuts when time or compute binds. Each step keeps: all guarantees, cal/test sizes, all baselines, N = 100, k = 10.

| id | trigger | action (config keys) | paper consequence |
|---|---|---|---|
| L1 | pilot v_measured < v_min (2.7532 pairs/s) | train 4000→2000, dev 1000→500 per domain | state sizes |
| L2 | N07 not done by Oct 2 | second teacher → Qwen3-Reranker-0.6B | second-teacher section says so |
| L3 | N07 not done by Oct 6 | n_main_configs 4→2 (single teacher); δ_cfg = 0.025 | family guarantee over 2 configs |
| L4 | N11 not done by Oct 12 | latency 300→100 queries, 3→1 repeats | note in setup |
| L5 | Oct 20 | experiment freeze; claim level from existing stats | limitations |
| L6 | pilot headroom < 30% or Oct 6 | skip the optional MS MARCO domain (N20) | "two domains" stated in setup |
| L7 | distillation gain < 10% on dev | keep the frozen cheap model (rule inside node N08, not a yaml rule) | reported as an ablation |

Manual (agent-decided, logged with DECIDED:) fallbacks not in the yaml:
- D-CAND: pyserini unavailable → bm25s (record parameters; no 2CR sanity reference).
- D-DATA-2: HF hub unreachable → hf-mirror.com → DPR question file.
- VRAM: OOM → batch 8→4 → max_length 512→384 (state L_max).
- Determinism audit fails → try `attn: eager`, then fp32 for the audit; re-cache with the deterministic setting.
- LaTeX missing after one install attempt → keep building through `paper/tools/check_paper.py` in Docker-less mode is impossible;
  block only the writing nodes with reason "latex" and continue experiments.
- Optional nodes (N18, N19) are skipped whenever the GPU or the calendar is needed elsewhere.

Never: reduce cal/test, drop the adaptive-prefix baseline, drop a policy, shorten the review loop below 2 rounds, fabricate.
