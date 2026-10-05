# Bandit / weak–strong-oracle neighbours of RSC-Rank (read 2026-09-21, full PDFs)

Scope: the three works that share RSC-Rank's *stop test* — "k-th best verified/lower score beats the best upper bound
outside" — but not its uncertainty model. Every statement below comes from the PDF text retrieved on 2026-09-21
(arXiv:2601.20989v1, arXiv:2602.02827v2, icml.cc/2012/papers/359.pdf). Section/equation numbers refer to those versions.
RSC-Rank notation for the five columns: (1) cross-query calibrated λ; (2) iff-tightness over all unsafe subsets
(Thm. 1–2: λ ≥ A_{k,r}(q) ⇔ no V ∈ U_{k,r}(q) can stop); (3) policy-uniform guarantee (one λ valid for every legal scoring
order); (4) r-miss frontier A_{k,r}; (5) finite-population (hypergeometric random-split) regime.

---

## 1. Oettershagen 2026 — Top-k on a Budget: Adaptive Ranking with Weak and Strong Oracles
- **Key:** `oettershagen2026topk` — arXiv:2601.20989 (v1, 28 Jan 2026; "arXiv preprint", no venue stated in the PDF).
- **Summary.** Items x ∈ D with unknown v(x) ∈ [0,1]; a *weak oracle* returns i.i.d. noisy estimates (mean v(x), bounded
  variance), a *strong oracle* returns v(x) exactly. Goal: exact top-k set w.p. ≥ 1−δ ((δ,k)-PAC, Sec. 3) with few strong
  calls. N weak pulls per item give intervals [L(x),U(x)] (empirical Bernstein) made *jointly* valid by a Bonferroni bound
  (event E). STC (Alg. 1) strong-queries every ambiguous item; ACE (Alg. 2) queries adaptively; ACE-W (Alg. 3) also allocates
  the weak budget. Cost in the near-tie mass m(η) = |{x : |v(x)−t*| ≤ η}|: ≤ m(4ε_max) strong calls (Thm. 2, 5, 7) vs a
  lower bound Ω(m(ε_max)) (Thm. 4); experiments on synthetic data and Shapley data valuation.
- **Exact stopping rule / guarantee.** ACE (Alg. 2 l.4–8): S_k ← k items with largest U; i ← argmin_{S_k} L; j ← argmax_{∉S_k} U;
  **return S_k iff L(i) ≥ U(j)**; Thm. 5: on E the output is T*_k. The baseline "TA-Certify" (Sec. 8) is the closest to our
  rule: it scores items in weak-estimate order and "stops as soon as the worst verified top-k item is guaranteed (under the
  weak CIs) to exceed any remaining unverified item" (k-th verified score ≥ max unverified U); it has no theorem of its own.
- **Uncertainty model.** Per-item CIs from repeated weak pulls, valid by concentration + union bound on one instance; δ per instance.
- **Lacks vs RSC-Rank.** (1) no — ε is analytic, nothing calibrated on held-out queries; (2) no — "on E every stop is
  correct" is sufficiency only, no worst case over unsafe scored subsets; (3) no — guarantee is for ACE's own rule (and
  TA-Certify's fixed order); (4) no — exact top-k only (Sec. 7 leaves other targets open); (5) no.

## 2. Col-Bandit 2026 — Query-Time Top-K Estimation for Late-Interaction Retrieval
- **Key:** `pony2026colbandit` — arXiv:2602.02827 (v2, 2 Jun 2026; IBM Research Israel; no venue stated in the PDF).
- **Summary.** ColBERT MaxSim reranking as *finite-population* top-K identification: the N×T MaxSim matrix is revealed in
  batches of query-token columns (shared random permutation π of [T]); each document keeps
  [LCB_i, UCB_i] = [max(LB_hard, Ŝ_i − r_i), min(UB_hard, Ŝ_i + r_i)] with empirical Bernstein–Serfling radius
  r_i = α_ef · T σ̂_i √(2 log(cNT/δ)/n_i) · ρ_{n_i} (Eq. 7–8); documents are eliminated round by round and the ≤ K+M survivors
  rescored exactly. Knob α_ef trades compute for fidelity (deployed 0.2: no certificate); ~8× fewer MaxSim FLOPs, Overlap@5 ≥ 0.90.
- **Exact stopping rule / guarantee.** Alg. 1 l.4–5: τ_r ← K-th largest LCB_i over the active set; **A_{r+1} ← {i : UCB_i ≥ τ_r}**;
  loop ends when |A_r| ≤ K+M or all T columns are revealed. Thm. 1 (App. C): for α_ef = 1, uniform-without-replacement reveals
  and Eq. 7 taken as a valid per-row radius, P(T̂_K = T*_K) ≥ 1−δ; Remark 2: c = 1 and the O(1/n) term are "folded into the
  radius rather than proved conservative", int8 elimination carries no PAC claim.
- **Uncertainty model.** Within-query finite-population concentration (Bardenet–Maillard) on partial token sums of one
  scorer; union bound over N documents × T reveal counts; no cheap-model prediction, no held-out queries.
- **Lacks vs RSC-Rank.** (1) no — α_ef is hand-set ("calibrated" only informally, Sec. 5.3); (2) no — sufficiency on the
  1−δ event; (3) no — one fixed reveal schedule; (4) no — exact set only; (5) *partial* — the finite population is the T
  tokens of one query, not a pool of calibration/test queries; no hypergeometric split guarantee.

## 3. LUCB 2012 — PAC Subset Selection in Stochastic Multi-armed Bandits
- **Key:** `kalyanakrishnan2012lucb` — Kalyanakrishnan, Tewari, Auer, Stone, ICML 2012, pp. 655–662 (icml.cc/2012/papers/359.pdf).
- **Summary.** Explore-m: select the m arms with the highest means of an n-armed Bernoulli bandit; (ε,m,δ)-optimal = w.p.
  ≥ 1−δ every returned arm satisfies p_a ≥ p_m − ε. LUCB separates a stopping rule from a greedy sampling strategy (sample
  the two critical arms h*_t, l*_t each round); expected sample complexity O(H^{ε/2} log(H^{ε/2}/δ)) (Thm. 6), matching-order
  worst-case lower bound (Sec. 4).
- **Exact stopping rule / guarantee (Sec. 3.1).** High_t = m arms with highest empirical means; h*_t = argmin_{High_t}(p̂ − β),
  l*_t = argmax_{Low_t}(p̂ + β); **terminate iff (p̂_{l*} + β(u_{l*},t)) − (p̂_{h*} − β(u_{h*},t)) < ε** (ε = 0 is the strict
  k-th-LCB-vs-best-UCB test). Thm. 1: for *any* sampling strategy, if Σ_t Σ_u exp(−2uβ(u,t)²) ≤ δ/n, the probability of
  returning a non-(ε,m)-optimal arm is ≤ δ (Hoeffding + union bound over rounds, arms, pull counts).
- **Uncertainty model.** Stochastic rewards, one pull per round; anytime-valid Hoeffding radii β(u,t); per-instance δ.
- **Lacks vs RSC-Rank.** (1) no — β is analytic; (2) no — Thm. 1 is sufficiency (a mistake implies a violated bound), no
  worst-case-over-unsafe-states identity; (3) *strategy-uniform but not our notion* — Thm. 1 holds for every sampling
  strategy because the bounds are anytime-valid, whereas RSC-Rank has one fixed prediction per item and proves
  policy-uniformity through the all-subset statistic; (4) no — ε is a value slack, not a rank slack; (5) no.

---

### What the three share with RSC-Rank, and what they do not (one-paragraph summary for §02_related)
All three stop (or eliminate) with the comparison "k-th best inside vs best upper bound outside" — LUCB Sec. 3.1,
ACE Alg. 2 line 7 / TA-Certify Sec. 8, Col-Bandit Alg. 1 line 5. Their upper bounds are stochastic confidence radii made
jointly valid by a union bound on a *single instance* (Hoeffding, empirical Bernstein, Bernstein–Serfling); the guarantee
is "P(correct) ≥ 1−δ on this instance" and follows from sufficiency of the interval-dominance argument. None of them
(i) calibrates a scale on held-out queries, (ii) shows that the stopping threshold equals the worst case over all unsafe
scored subsets (the iff of Thm. 1–2), (iii) states validity of one threshold under every scoring order (Prop. 5 / Thm. 6–8),
(iv) treats an r-miss target, or (v) gives a finite-population split guarantee over queries. Col-Bandit's "finite
population" is the token dimension of one query; Oettershagen's TA-Certify is our fixed μ-descending policy with
per-item CIs in place of μ_j + λσ_j. The paper's current paragraph "Top-k identification with weak and strong oracles"
in `paper/sections/02_related.tex` states exactly this and is consistent with the PDFs; one addition is warranted:
TA-Certify should be named explicitly as the sequential (scan-in-weak-order) variant of the test, and Col-Bandit's
finite-population CIs should be distinguished from our finite-population *split* regime so that the word is not read as
the same thing.

### Second-pass sanity check (2026-09-21, later session)
Re-located in the extracted texts: ACE = Alg. 2 lines 4–8 ("S_k ← set of k items with largest upper bounds", "if
L(i) ≥ U(j) then return S_k"), Thm. 5 and its proof ("v(x) ≥ L(x) ≥ L(i) ≥ U(j) ≥ U(y) ≥ v(y)"); TA-Certify = Sec. 8
("ranks items by the weak estimates, then queries the strong oracle sequentially in that order … stops as soon as the
worst verified top-k item is guaranteed (under the weak CIs) to exceed any remaining unverified item"); LUCB Thm. 1
("Let S … be an arbitrary sampling strategy … the probability that it returns a non-(ε,m)-optimal arm is at most δ");
Col-Bandit Eq. 7 (α_ef · T σ̂_i · ρ_{n_i}), the "K-th largest lower bound" elimination and Remark 2. Two consequences for
02_related.tex (reported, not edited): (i) its Oettershagen sentence ("stopping when the worst *verified* top-k item is
guaranteed to exceed every unverified item") quotes the TA-Certify baseline of Sec. 8, whereas ACE (Alg. 2, Thm. 5)
stops on L(worst IN by U-ranking) ≥ U(best OUT) with items that need not have been strong-queried; (ii) "give no
statement … about its validity under other scoring orders" is not accurate for LUCB, whose Thm. 1 holds for an
arbitrary sampling strategy (anytime-valid bounds) — the honest contrast is that LUCB's strategy-uniformity comes from
per-arm anytime-valid CIs, ours from one calibrated all-subset statistic with a single prediction per item.
