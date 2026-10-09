# PREREGISTRATION.md — fixed before any result exists (2026-09-20)

## Hypotheses
- H1 (validity): with λ calibrated from A_{k,1} on the calibration split, the exact ordered-mismatch risk on the test split is
  ≤ α (marginal: in expectation; PAC/finite: with probability ≥ 1−δ_cfg), *for every* legal scoring policy without recalibration.
- H2 (efficiency): mean teacher calls of RSC-Rank are ≥ 30% below the full teacher at α = 0.05 on both main domains and both teachers.
- H3 (strong baseline): RSC-Rank uses fewer teacher calls than the query-adaptive calibrated prefix at the same α and regime
  (paired bootstrap LCB of the difference > 0). H3 may fail; the paper then reports it.
- H4 (quality): nDCG@10 of RSC-Rank is non-inferior to the full teacher with margin 0.005 (paired query bootstrap LCB > −0.005).
- H5 (cost of uniformity): the fixed-trajectory threshold uses fewer calls than A on its own trajectory but violates α under at
  least one other policy without recalibration; RSC-Rank does not.
- H6 (latency): tuned-regime mean latency of RSC-Rank is ≥ 20% below the full teacher; p95 reported.

## Design (frozen)
Datasets, models, N = 100, k = 10, α grid, regimes, δ_family = 0.05 over 4 configs, batch schedule 16/8, policies, baselines,
metrics, bootstrap 2000, non-inferiority margin 0.005, model-selection criterion (calls at α_sel on dev temp-split) — all in
configs/main.yaml and graph/decisions.md. Cross-domain (FiQA/SciFact) with source-domain thresholds is a stress test only.

## Primary regime
The **finite-population** guarantee (random split of a fixed BEIR pool) is primary for every headline number; marginal and
PAC are reported next to it. Rationale: the test set is the rest of a fixed pool, not a sample of future traffic.

## Gates (evaluated by scripts/10_stats.py; the claim level follows mechanically)
risk_within_alpha_primary · risk_marginal_not_significantly_above (CP lower bound ≤ α) · calls_reduction_ge_target (30%) ·
efficiency_not_failed (≤ 90% of N) · beats_adaptive_prefix_calls · beats_uniform_prefix_calls · ndcg_noninferior ·
policy_uniform_risk_ok (max risk over all scoring policies ≤ α + 0.02).

## Pre-registered optional levers (decided by rule, never by taste; all selected on dev only)
- **L-DISTILL**: fine-tune the cheap MiniLM on TRAIN teacher scores; adopt iff dev calls at α improve ≥ 10% relative.
- **L-MSMARCO**: add MS MARCO passage dev (real web queries) as an extra domain iff the pilot leaves ≥ 30% GPU headroom.
- **L-CANONICAL**: batch-invariant teacher micro-batching iff it is bit-stable and ≥ 2/3 as fast as the best fast setting.
- **L-POLICY**: the cost-aware `mu_binned_length` schedule is reported in the policy table; it is never the main policy.
Each lever's outcome (adopted / not adopted, with the measured numbers) is reported in the paper.

## Cost anatomy (E9, pre-registered analysis)
Per covered query the calls decompose exactly as |V*| + band + batching overshoot (Prop. 10 in docs/THEORY.md); we report
the three terms, the oracle gap (calls − |V*|) and the price of policy-uniformity A_{k,1} − A^π.

## Amendments (append-only; date, reason, effect)
- **A1 (2026-09-22, E2b: calibrated prefix baselines under alternative scoring orders).** *When:* after the main E1/E2
  replay results of all six configurations had been computed (RSC, fixed-trajectory and the prefix baselines under the
  mu order; RSC and the fixed-trajectory threshold under the nine orders). *Reason:* the query-adaptive calibrated prefix
  is the cheapest calibrated method under the mu order in every configuration, and its guarantee is tied to that order
  (its statistic H_k is the worst mu-rank of a winner). The pre-registered E2 measured order changes only for the
  guardrail methods, so the strongest baseline's order-specificity was never measured. *Effect:* E2 additionally
  replays the uniform and the query-adaptive calibrated prefix under every alternative order, (i) *transferred* (the
  mu-calibrated budget applied to the first documents of the other order) and (ii) *recalibrated* on that order's
  horizon H_k^pi -- exactly the transferred/recalibrated treatment E2 already gives the fixed-trajectory threshold.
  `observed_regression` is excluded (it has no order before scores are observed). No threshold of any existing method
  changes; no gate or claim level depends on E2b; it is reported as a separate table. The amendment is disclosed in the paper.
- **A2 (2026-09-22, E3b + latency: blocker-batch schedule).** *When:* after the controlled-regime latency measurement of the
  main (16 then 8) schedule on the Qwen configurations showed that RSC's fewer teacher calls do *not* translate into lower
  wall-clock latency with the deterministic (canonical) teacher: each small sequential round costs one or more fixed-shape
  micro-batches, so RSC took 2-2.5x the full teacher's time while the one-shot prefix methods were 20-30 % faster.
  *Reason:* the guarantee is policy-uniform, so the schedule may be changed without recalibration; Prop. 10/11 identify the
  documents any correct stop must score (those whose guardrail reaches the current k-th score, the "blockers"). *Effect:*
  a state-dependent batch schedule `core.blocker_batch_schedule` (first round as before, then one round per blocker set,
  at least `rest` documents) is added as an RSC variant: E3b replays it (calls, rounds, risk) next to the fixed schedules,
  and the latency experiment measures it in the controlled regime (`--schedule blockers`, stored as method `rsc_blockers`).
  The pre-registered main schedule (16/8) remains the main configuration in every table; the variant is reported beside it.
  Its decisions use only cheap information and observed scores (legal policy), so the calibrated lambda and all
  guarantees are unchanged. The amendment is disclosed in the paper.
- **A3 (2026-09-23, E2 policies; from the round-1 simulated review).** *When:* after the first internal review round.
  *Finding:* the pre-registered adaptive order "refit on observed scores" fitted s ~ a + b*mu on the observed
  documents and re-ordered the rest by a + b*mu; whenever the fitted slope is positive (essentially always) this
  preserves the mu order, so the policy was a no-op (identical numbers to the mu row in every configuration).
  *Change:* (i) `observed_regression` now fits a ridge regression of the observed teacher scores on the standardised
  cheap features [mu, raw cheap logit, BM25, log length] once >= 8 scores are observed and orders the remaining
  documents by the refit prediction (a genuinely adaptive legal order); (ii) a tenth order `mu_alt` is added: the
  order of the cheap-model variant NOT adopted for the teacher (distilled MiniLM for Qwen3-4B, frozen MiniLM for
  bge-m3) while the guardrail keeps the adopted heads -- the "ordering model changed, calibration did not" scenario;
  (iii) E2 additionally reports the *union-over-orders* risk (fraction of test queries that fail under at least one
  tested order) for the single RSC threshold, the transferred fixed-trajectory threshold, and the per-order
  recalibrated fixed-trajectory thresholds and prefix budgets. All E2 numbers are recomputed; the pre-registered
  eight unchanged orders keep their definitions. Disclosed in the paper.
- **A4 (2026-09-23, throughput under concurrency; from the round-1 simulated review).** *When:* after the first
  internal review round. *Finding:* the pre-registered latency protocol serves one query at a time, so a saved
  teacher call is idle GPU time; reviewers asked for the regime in which saved calls are reclaimed. *Change:*
  `scripts/09b_throughput.py` simulates a server with C in {1, 8, 32} queries in flight whose pending pairs are
  scored together in canonical fixed-shape micro-batches (bit-identical scores, identical controller decisions); it
  reports throughput (queries/s), GPU-seconds per query, per-query latency under load, forward passes and dummy-row
  fraction for full scoring, RSC (16/8), RSC (blocker batches), and the two calibrated prefixes, on the same 300
  test queries per configuration. H6 is re-stated for this regime as an exploratory (not pre-registered) outcome.
- **A5 (2026-09-23, family-wise delta over six configurations; from the round-1 simulated review).** *Finding:* the
  PAC / finite-population thresholds were calibrated with delta_cfg = delta_family / 4 = 0.0125 for the four
  pre-registered main configurations, and MS MARCO (added as a third main domain by node N20) reused the same
  delta_cfg "as a separate statement" while the abstract and conclusion aggregate over all six configurations, so the
  family-wise failure probability of the six reported finite-population claims was 6 x 0.0125 = 0.075, not 0.05.
  *Change:* `protocol.n_main_configs` = 6 (delta_cfg = 0.05 / 6), MS MARCO is a member of the main family; all
  thresholds are recalibrated (07) and every replay (E1-E9), the statistics, tables and numbers are regenerated;
  the ablation configurations use the same delta_cfg. The marginal-regime thresholds do not depend on delta and are
  unchanged. Disclosed in the paper.
- **A6 (2026-09-23, fair per-order competitor and realistic envelope; from the round-2 simulated review).** *Finding:* the
  round-1 revision compared one rank-sufficient threshold at alpha with the UNION of K per-order thresholds each calibrated at
  alpha -- a quantity that is expected to grow with K and that nobody who knows the K orders would deploy. *Change:* (i) the
  per-order competitor is now also calibrated at alpha/K (Bonferroni), which bounds the union over the K tested trajectories;
  its calls under the mu order are reported next to the single threshold's (E2d, `scripts/16_policy_envelope.py`); (ii) the
  envelope is additionally computed over the *realistic* orders only (mu-based and adaptive orders: mu, raw cheap score,
  refit, alternating, shortest-first window, mu', and the two blocker schedules), with the corresponding share of the
  premium; (iii) E8 gains a 'global-estimate' coverage stop (B_all with sigma = 1, one calibrated radius for every unscored
  document -- the Fast-Forward-style rule) as an executed baseline; (iv) paired call differences are reported with both
  interval ends and 'cheaper by at least' statements use the end nearest zero. The union-over-orders quantity is described
  as the worst case over query-dependent order selection (a system whose order selection is independent of the query is
  exposed to a mixture, which per-order calibration at alpha does bound). Disclosed in the paper.
- **A7 (2026-10-05, order-free baselines, lambda-dependent schedules and repeated timing; from the round-3 simulated review).**
  *Findings:* (i) the round-2 envelope (E2d) entered the two blocker schedules through trajectories generated at
  lambda = +inf, but the blocker schedule chooses its second round from the guardrails, so the critical value of a trajectory
  frozen at one threshold does not protect the controller re-executed at another (counterexample k=1, s=(0,1,10),
  mu=(10,-5,-10), sigma=1, b0=b=1); the frozen values are dominated by the mu-order 16/8 checkpoints, so the round-2 envelope
  threshold equals the envelope of the ten lambda-independent orders in all six configurations, while the re-executed blocker
  controllers were not covered by it (1.8-2.8 % of test queries exposed); (ii) the per-order Bonferroni competitor split alpha
  by K but kept delta_cfg per order, so its union statement held with confidence 1 - K delta_cfg; (iii) no calibrated prefix
  was evaluated with a fixed candidate set under other orders (set completion); (iv) the anatomy comparison of the
  certification core (covered queries) with the adaptive prefix (all queries) used different samples; (v) the throughput
  cells had one timed pass. *Change:* `core.blocker_critical_value` (closed form, checked against re-executions in
  `tests/test_blocker_critical.py` and on every test query); `scripts/18_order_free_baselines.py` computes the envelopes over
  the ten orders and over the thirteen declared policies (blocker schedules by their own critical value), per-policy
  Bonferroni at (alpha/K, delta/K), the set-completion prefixes (adaptive and uniform) under every order with paired
  differences to RSC, and the same-sample covered comparison; two further timed passes of every C = 32 throughput cell
  (`scripts/09b_throughput.py --out-dir runs/throughput_rep/pass2|pass3`). No threshold of RSC, the trajectory threshold, the
  coverage rules or the prefixes changes. Disclosed in the paper (Appendix B).
- **A8 (2026-10-06, one split after the policy family was frozen; from the round-3 simulated review).** *Finding:* logging the thirteen policies before an A7 recomputation on the same split does not make that family a statistic fixed before the split, so Theorem 9 did not cover the envelope row. *Change:* the ten orders and three blocker schedules stay as already named. One uniform redraw of each domain's fixed pool (the previous calibration set union the test set), same `n_cal`, seed `20261006` with a fixed domain code, written to `runs/a8_split/splits/` before any metric and not redrawn. `scripts/18_order_free_baselines.py --frozen-family-split` recomputes calibration and the call/risk comparisons from the cached scores. Train and development sets are untouched. Serving timings, sequential latency, and the marginal/PAC headline numbers were not repeated; the paper says so. Disclosed in Appendix B.
- **A9 (2026-10-08, three diagnostics on the A8 split; from the round-7 simulated review; logged after the fact on 2026-10-09).** *What:* computed afterwards from the frozen scores, heads and A8 thresholds, selecting nothing: (i) the all-subset witness risk P(A_k1(q) > lambda) at the trajectory, envelope and RSC thresholds; (ii) the last-winner identity and the critical value of every policy paced by the deployed 16/8 checkpoint grid, with its finite-population threshold (`scripts/22_grid_frontier.py`); (iii) a re-scoring of the 300 timed queries at the tuned micro-batch against the canonical cache (`scripts/23_tuned_disagreement.py`). No threshold, head, policy family or protocol choice changed. Disclosed in Section 6 and Appendix B.
- **A10 (2026-10-09, fresh queries and an own-target scale head; from the round-7 simulated review; registered before any fresh query was drawn and before the new head was trained).**
  *(i) Fresh queries.* Per main domain, 2000 queries drawn uniformly (seed 20261009 with the fixed domain code of A8) from the domain's training-question source (`data/raw/train_questions/<domain>.jsonl`), after removing every query of the frozen train/dev/cal/test splits by id and by the split de-duplication rule (exact normalised match or 3-gram Jaccard >= 0.8 against those queries) and internal exact duplicates; written to `data/fresh/splits/<domain>/fresh.jsonl` before retrieval. Same BM25 backend and parameters, N = 100; cheap features with the cheap model adopted for each teacher; canonical-shape teacher scores; frozen heads, no refit. Thresholds all frozen on the A8 calibration set: RSC at the marginal, PAC (delta_cfg) and finite-population (deployed) regimes; the trajectory threshold, the thirteen-policy envelope, B_top and B_all at their deployed thresholds; adaptive and uniform prefixes with their deployed parameters. Reported: ordered top-k risk on the deployed mu-order 16/8 with a Clopper-Pearson 95 % interval, the witness risk P(A_k1(q) > lambda), mean calls. Theorems 8-9 assume calibration and future queries exchangeable or i.i.d.; these queries come from another split of the source the heads were trained on, so the run measures transfer to new queries of the same domain, and whatever risk results is reported. No choice is made on these queries.
  *(ii) Own-target scale head for B_all.* A scale head with the architecture, features z(q), training data (train+dev of the main domains), optimiser settings and seed of the existing head, fitted to log(eps + RMS over all candidates of s_j - mu_j) with the frozen mu; B_all is calibrated with it on the A8 calibration set (finite regime, alpha, delta_cfg) and replayed on the A8 test set (deployed order, 16/8). Reported next to B_all with the winners'-residual head and with sigma = 1. No other method changes.
  *(iii) Risk-calls curve.* For alpha in {0.01, 0.02, 0.05, 0.10, 0.20}, finite-population thresholds on the A8 calibration set (delta_cfg) for RSC, B_top, B_all with its own head, the trajectory threshold and the adaptive and uniform prefixes; risk and mean calls on the A8 test set, deployed order. Calls only: serving time was measured at alpha = 0.05 alone.
  *(iv) Service contract.* No new measurement: the paper states the interface under which the stopping module cannot name the documents and reports its value from the A7 comparisons and the matched server.
