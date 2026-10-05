# CLAIMS.md — the three contributions and the sentences each claim level permits

## Contributions (fixed wording for the introduction)
- **C1 — Tight rank-sufficient critical values.** For the strict stop rule τ(V) > max_{j∉V}(μ_j + λσ_j), the worst case over
  the exponentially many scored subsets that miss at least r winners equals the closed-form query statistic
  A_{k,r}(q) = rthmax_{i∈T_k}(s_(k+r) − μ_i)/σ_i (Theorems 1–2). Score-coverage statistics (B_top, B_all) are provably more
  conservative (Prop. 3).
- **C2 — Policy-uniform risk control.** Calibrating A_{k,r} on queries yields one λ that protects every legal scoring order at once,
  in three regimes: marginal, PAC, and an exact finite-population guarantee for a random split of a fixed benchmark
  (Theorems 6–8). No per-policy recalibration.
- **C3 — Benefit and cost, measured.** Policy-uniformity can cost up to N/k calls versus a trajectory-specific threshold (Prop. 4),
  and on a covered query the calls decompose exactly into the instance-optimal certification core |V*|, the guardrail band beta(q)
  and batching overshoot, so the distance from instance optimality and the price of policy-uniformity A_{k,1} - A^pi are measured,
  not asserted (Prop. 10); experiments with two frozen teachers, two web QA domains (NQ, HotpotQA) plus MS MARCO passage as an additional
  web-query domain (D-MSMARCO, enabled by the N05 pilot), nine scoring policies (the main mu order and eight alternatives),
  strong calibrated prefix baselines and real GPU latency quantify when the uniform guarantee pays for itself.

## Claim levels (runs/stats/summary.json → claim_level)
**full** — "RSC-Rank meets the target risk in every configuration and regime, reduces teacher calls by \num…% at α = 0.05 with
non-inferior nDCG@10, and needs fewer calls than the query-adaptive calibrated prefix; the same threshold remains valid under all
eight scoring policies, whereas trajectory-specific calibration does not."
**policy-uniform-protection-with-cost** — "RSC-Rank meets the target risk in every configuration and under every scoring
policy without recalibration; its cost lies between the fixed-trajectory threshold and the calibrated prefix baselines — the
price of policy-uniformity is \num… calls on average, which we characterise." (No superiority claim over the adaptive prefix.)
**theory-plus-negative-efficiency** — "The guarantees hold, but on these teachers and domains the certified savings are below
10%: the cheap predictor's residual scale on winners is too large. We report this as a negative result and quantify the
predictor quality required by the frontier."

## Never-claims
SOTA; "guaranteed" for empirical results; improvement of relevance over the teacher; validity after any change of
F, G, μ, σ, N, k or numerical backend; cross-domain guarantees; per-query guarantees; global novelty beyond docs/LITERATURE.md §D.

**validity-failure** — a validity gate (risk within alpha in the primary regime, marginal risk not significantly above alpha, or policy-uniform risk under every scoring order) failed in some configuration; no guarantee claim may be made for that configuration and the failure is reported as the main result of the empirical section. (Added 2026-09-22 when the claim-level rule was made to consult the policy-uniform gate explicitly, as D-CLAIMS requires.)

## Revision after the round-3 review (amendment A7, 2026-10-05)
The introduction's C2/C3 now read: **C2 — what one calibration protects** (one threshold for every policy, Theorems 7–9 in
the paper's reading-order numbering; a trajectory threshold protects its trajectory; lambda-dependent schedules need their own
critical value, Proposition 5; the ladder one trajectory / declared family / all scored subsets) and **C3 — what it costs,
against the right alternatives** (premium over one trajectory and over the envelope of the thirteen tested policies; coverage
and per-policy Bonferroni at (alpha/K, delta/K); the set-completion prefix as the order-free alternative, cheaper when its set
is scored first and dearer once the order departs from the cheap model; sequential latency and C = 32 throughput). Permitted
new sentences: "the certificate needs fewer calls than the committed set in <AggRThreeCellsRscCheaper> of
<AggRThreeCellsTotal> order-configuration cells away from the deployed order"; never: "RSC-Rank is cheaper than the
calibrated prefix" without the order qualifier, or "only at alpha/K" as a necessity statement.
