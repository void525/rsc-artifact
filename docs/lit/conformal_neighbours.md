# Conformal / risk-control neighbours of RSC-Rank (read 2026-09-21)

Full PDF text was retrieved for eight of the nine works in §1–9 (arXiv PDFs via curl + pdftotext); Conformal@K is abstract-only
(Crossref record for doi 10.1145/3828550; the ACM DL body returns 403 to non-browser clients). §10–11 (added in the
second pass, 2026-09-21) are the two conformal-IR works cited in 02_related.tex that had no block yet; both are
**abstract-level** (arXiv API abstracts; venue/DOI verified on the ACL Anthology and Crossref). Section/equation numbers
refer to the arXiv versions named in each block. Column legend as in `bandit_neighbours.md`:
(1) cross-query calibrated λ | (2) iff-tightness over all unsafe subsets | (3) policy-uniform guarantee | (4) r-miss
frontier | (5) finite-population regime. "n/a" = the concept does not apply because the method has no stopping rule.

---

## 1. Two-stage Risk Control with Application to Ranked Retrieval — Xu, Ying, Guo, Wei
- **Key:** `xu2025twostage` — IJCAI 2025, pp. 9104–9111, doi 10.24963/ijcai.2025/1012; arXiv:2404.17769 (text read).
- **Summary.** Two sequential tuning parameters, λ (retrieval) and γ (ranking), on finite grids in [0,1]; losses L^(1)_i(λ),
  L^(2)_i(λ,γ) non-increasing and right-continuous (Sec. 2). Ranked retrieval: Ĉ^(1)(X;λ) = {d : M_retrieval(q,d) ≥ 1−λ}
  (Eq. 13), loss 1 − |Y ∩ Ĉ^(1)|/|Y| (Eq. 14); Ĉ^(2)(X;λ,γ) = {d : M_rank(q,d) ≥ 1−γ} ∩ Ĉ^(1) (Eq. 15), loss 1 − nDCG_mod
  (Eq. 16); among feasible pairs the empirical second-stage set size is minimised (Eq. 17–18). MSLR-Web10K, Yahoo LTRC.
- **Exact guarantee.** LTT (Thm. 1): FWER-controlling grid test returns R with P(∀(λ,γ)∈R: R^(1)(λ) ≤ α_1, R^(2)(λ,γ) ≤ α_2)
  ≥ 1−δ. Two-stage CRC (Thm. 2): E R^(1)(λ̂) ≤ α_1 finite-sample, lim sup E R^(2)(λ̂,γ̂) ≤ α_2 asymptotically; sample-splitting
  (Thm. 3): finite-sample control of both expected risks. Marginal over i.i.d. (query, relevance-set) draws.
- **Uncertainty model.** Distribution-free (LTT / CRC) over exchangeable queries; losses use *ground-truth relevance*, not
  agreement with a frozen teacher.
- **Lacks vs RSC-Rank.** (1) yes — one (λ,γ) for all queries; (2) no — the calibrated object is a *score cutoff* (a prefix),
  no stop rule, no worst case over scored subsets; (3) n/a — everything above the cutoff is scored; (4) no — miss-fraction /
  nDCG losses, not an r-miss rank event; (5) no. Our "calibrated prefix" baseline family.

## 2. Conformal@K — Fan, Liu, Pan
- **Key:** `fan2026conformalk` — ACM TOIS 44(6), 2026, pp. 1–42, doi 10.1145/3828550 (Crossref: published 2026-07-31).
  **Abstract-level only**; body UNVERIFIED.
- **Summary (abstract).** "A model-agnostic calibration layer that provides distribution-free, finite-sample control of
  Top-K miss-risk in two-stage recommendation and retrieval. Given arbitrary retrieval and re-ranking scorers, Conformal@K
  selects monotone budgets so that, at level 1−α, the probability that no relevant item appears in the returned Top-K is
  controlled whenever a feasible parameter exists; otherwise, a transparent best-effort mode reports the residual gap."
  Extensions: overlapping-group guarantees via smoothed self-normalised estimates, joint two-stage calibration
  (retrieve-miss and final miss@K), importance-weighted and windowed variants for covariate shift / temporal dependence.
  Data: MSLR-WEB10K, Gowalla, Foursquare (display cap K_max = 50).
- **Exact guarantee.** As stated in the abstract: P(no relevant item in the returned Top-K) ≤ α under a calibrated
  monotone *budget* (retrieval depth / K). Theorem numbers UNVERIFIED (body not read).
- **Uncertainty model.** Distribution-free calibration over users/queries with relevance labels.
- **Lacks vs RSC-Rank.** (1) yes (a budget for all queries); (2) no — budget = prefix size, no scored-subset worst case
  (from the abstract; UNVERIFIED for the body); (3) n/a; (4) no — the event is "miss every relevant item", not
  "miss ≥ r of the teacher's top-k"; (5) no evidence in the abstract.

## 3. CertBind from Multimodal Connectivity to Certifiable Retrieval Decisions — Cao et al.
- **Key:** `cao2026certbind` — arXiv:2608.06516 (v1, 6 Aug 2026; the PDF carries an AAAI-2027 copyright line; venue
  otherwise UNVERIFIED — cite as arXiv).
- **Summary.** A "multiscale certificate chain" for frozen multimodal connector graphs: native anchors fix the target
  gauge (Thm. 1), stratified conformal p-values + Holm control edge flags (Thm. 2), coordinate-median recovery over
  declared paths with a split-conformal residual radius (Thm. 3), then a query-scale top-k certificate: Direct / Certified / Abstain.
- **Exact guarantee (Thm. 3).** Owner residuals Z_i = max_p ‖y_{i,p} − y*_i‖_∞; ε̂ = split-conformal order statistic at rank
  ⌈(N+1)(1−δ_p)⌉; ρ := √d · ε̂ (Eq. 17) gives P(‖ŷ − y*‖_2 ≤ ρ) ≥ 1−δ_p (Eq. 18). With unit-norm gallery scores s_j = ⟨ŷ,g_j⟩,
  ℓ_j = s_j − ρ, u_j = s_j + ρ, τ_k = k-th largest ℓ: C_k = {j : u_j ≥ τ_k} = {j : s_j ≥ s_(k) − 2ρ} (Eq. 19, 24),
  P(S*_k ⊆ C_k) ≥ 1−δ_p (Eq. 20); **|C_k| = k ⇔ gap s_(k) − s_(k+1) > 2ρ** (Eq. 21); P(Certified ∩ {Ŝ_k ≠ S*_k}) ≤ δ_p (Eq. 22).
- **Uncertainty model.** One conformal radius for *all* items of a query (score coverage with a common margin); no per-item
  scale, no expensive oracle, no sequential scoring.
- **Lacks vs RSC-Rank.** (1) yes — one radius across queries; (2) no — the certificate is score coverage (s_j ∈ [ℓ_j,u_j] ∀j),
  our B_all-type statistic, which Prop. 3 shows is more conservative than A_{k,1}; nothing is scored sequentially, so no
  worst case over scored subsets; (3) n/a — no stopping, no policy; (4) no — set equality only; (5) no.

## 4. Transductive Conformal Inference for Full Ranking — Fermanian, Humbert, Blanchard
- **Key:** `fermanian2025transductive` — NeurIPS 2025 (stated on the PDF), arXiv:2501.11384 (v3 read).
- **Summary.** n+m items are scored by a black-box ranker; the relative ranks of n of them are known; the target is the
  rank of each new item among all n+m. Since the calibration items' full ranks depend on the unknown test ranks, the
  paper bounds them (Eq. 6) and uses proxy scores S_i = max_{r∈[R_i^-,R_i^+]} s(X_i, r, ·) (Eq. 7).
- **Exact guarantee.** Thm. 3.5: with k = ⌈(1−α+δ)(n+1)⌉ and prediction set Ĉ_k(x) = {r : s(x,r,·) ≤ S_(k)} (Eq. 8),
  P(R_{n+i} ∈ Ĉ_k(X_{n+i})) ≥ 1−α for every test item. Prop. 3.8: the false coverage proportion over the m test items is
  ≤ α + λ_{n,m} with probability ≥ 1−β−δ. Prop. 4.1–4.2 give explicit / calibrated rank envelopes.
- **Uncertainty model.** Exchangeability of items within one ranking problem; no cost model.
- **Lacks vs RSC-Rank.** (1) no cross-*query* calibration (calibration is within one ranking instance); (2) n/a — no
  stopping rule; (3) n/a; (4) no; (5) *different* finite-population flavour: the transductive bound is over n+m items of
  one instance, not a random split of a fixed benchmark of queries. Orthogonal to our problem, as 02_related.tex says.

## 5. Recommendation Systems with Distribution-Free Reliability Guarantees — Angelopoulos, Krauth, Bates, Wang, Jordan
- **Key:** `angelopoulos2023recsys` — COPA 2023, PMLR 204 (proceedings.mlr.press/v204/angelopoulos23a.html verified);
  arXiv:2207.01609 (v1 read).
- **Summary.** From a pairwise ranking model π̂, item scores s_i(X) = mean_{j≠i} π̂_{i,j}(X); the returned set is
  T_λ(X) = {i : s_i(X) ≥ λ} (Sec. 2.2). The error is the false discovery proportion FDP(Ŝ,Y) = |Ŝ ∩ Y_(m+1:K)| / max(|Ŝ|,1)
  (items not among the true top-m). λ is chosen by Learn-then-Test with fixed-sequence testing over Λ = (.99, .98, …).
  A diversity-optimising extension keeps the same FDR certificate. Data: Yahoo! LTR and MS MARCO.
- **Exact guarantee.** Eq. (1): P(FDR(T_λ̂) > α) < δ with FDR = E[FDP]; the test rejects H_{0,λ}: FDR(T_λ) > α when the
  empirical FDP plus a Hoeffding term sqrt(log(1/δ)/(2n)) is below α (Eq. 2, Alg. 1).
- **Uncertainty model.** Distribution-free over i.i.d. (X,Y) with ground-truth top-m labels.
- **Lacks vs RSC-Rank.** (1) yes — one λ for all queries; (2) no — score threshold ⇒ prefix, no scored-subset worst case;
  (3) n/a — everything is scored by the model, nothing is skipped; (4) no; (5) no.

## 6. Conformal Thinking: Risk Control for Reasoning on a Compute Budget — Wang, Suresh, Zhang, More, Jurayj, Van Durme, Farajtabar, Khashabi, Nalisnick
- **Key:** `wang2026conformalthinking` — ICML 2026, PMLR 306 (stated on the PDF), arXiv:2602.03814 (v2 read).
- **Summary.** Adaptive-thinking LLMs stop when an uncertainty signal s_t = u(x, r_{1:t}) (Eq. 2) crosses an *upper*
  threshold λ⁺ (stop when confident) or falls below a *dynamic lower* threshold λ⁻(t; c,s,l,u) (Eq. 12–13, a sigmoid of
  tokens spent — stop unsolvable instances early). Losses in [0,1] measure false-positive stops, false-negative stops and
  wasted steps (Eq. 8–11).
- **Exact guarantee.** Eq. 14: P_V(R(λ̂⁺, ĉ) ≤ ε) ≥ 1−δ via the UCB procedure of Bates et al. (RCPS) assuming risks are
  monotone in the hyper-parameters (Sec. 4.4); Alg. 1 enumerates (signal, threshold) pairs, keeps those whose adjusted
  risk ≤ ε, and picks the most efficient. The two-step (upper then lower) selection is stated to lose some rigour
  ("though with some loss of theoretical rigor"; App. C gives the failure mode).
- **Uncertainty model.** Distribution-free over i.i.d. validation instances for one fixed reasoning procedure.
- **Lacks vs RSC-Rank.** (1) yes; (2) no — a calibrated *exit threshold* on one trajectory, no worst case over states
  of other procedures; (3) no — validity is for the fixed sampling/exit procedure that was calibrated; (4) no; (5) no.

## 7. Fast yet Safe: Early-Exiting with Risk Control — Jazbec, Timans, Hadži Veljković, Sakmann, Zhang, Naesseth, Nalisnick
- **Key:** `jazbec2024fastyetsafe` — NeurIPS 2024 (stated on the PDF), arXiv:2405.20915 (v2 read).
- **Summary.** An early-exit network exits at the first layer whose confidence ĉ_l ≥ λ (Eq. 1); assuming marginal
  monotonicity of performance across exits (Eq. 2), the paper defines a *performance-gap risk*
  R_G(λ) = E[ℓ(ô_λ,y) − ℓ(ô_L,y)] (Eq. 6) and an unsupervised *consistency risk* R_C(λ) with the full model's prediction
  as label (Eq. 8), and controls them post hoc.
- **Exact guarantee.** Prop. 1: λ̂_CRC = min{λ : (n/(n+1)) R̂(λ) + B/(n+1) ≤ ε} (Eq. 10) gives E R(λ̂_CRC) ≤ ε (conformal
  risk control). Prop. 2: λ̂_UCB = min{λ : R̂⁺(λ') < ε ∀λ' ≥ λ} (Eq. 12) gives P(R(λ̂_UCB) ≤ ε) ≥ 1−δ (RCPS). Default λ̂ = 1
  (no early exit) if no feasible threshold exists.
- **Uncertainty model.** Distribution-free over i.i.d. calibration inputs for one fixed exit sequence.
- **Lacks vs RSC-Rank.** (1) yes; (2) no — the loss is monotone in λ along *one* exit trajectory, no all-subset
  statement; (3) no — a different exit policy needs re-calibration (the paper's Eq. 1 fixes the policy); (4) no; (5) no.
  This is exactly our fixed-trajectory baseline family (Prop. 5).

## 8. Consistent Accelerated Inference via Confident Adaptive Transformers (CATs) — Schuster, Fisch, Jaakkola, Barzilay
- **Key:** `schuster2021cats` — EMNLP 2021, pp. 4962–4979, doi 10.18653/v1/2021.emnlp-main.406; arXiv:2104.08803 (read).
- **Summary.** An amortised model G with early prediction heads F_k and meta "consistency" classifiers M_k exits at the
  first layer with M_k(x) > τ_k (Eq. 2). Goal: P(G(X_{n+1}) = F(X_{n+1})) ≥ 1−ε (Eq. 1) using *unlabelled* exchangeable
  calibration data. Conformal prediction is inverted: first build a set C_ε(x) ⊇ I(x) (the inconsistent layers) with
  probability ≥ 1−ε (Eq. 6), then exit at the first layer not in it (Prop. 4.1).
- **Exact guarantee.** Independent calibration (Thm. 4.4): per-layer quantiles τ_k^ind at level α_k = ω_k ε (weighted
  Bonferroni). **Shared calibration (Sec. 4.2.2, Alg. 1):** the statistic M_max(x) = max{M_k(x) : F_k(x) ≠ F(x)} (Eq. 9)
  — the worst-case meta-confidence over the *inconsistent* layers of one input — and a single threshold
  τ_share = Quantile(1−ε, {M_max(x_i)} ∪ {∞}) shared by all layers; Thm. 4.6 states validity for any number of layers.
- **Uncertainty model.** Split-conformal over exchangeable inputs; validity is w.r.t. the full model F (like ours w.r.t.
  the frozen teacher).
- **Lacks vs RSC-Rank.** (1) yes — one τ across inputs; (2) *partial analogue only*: M_max is a max over the unsafe
  states of one fixed exit sequence (the layers), i.e. the analogue of our trajectory-specific A^π_{k,1} (THEORY §5), not
  of the all-subset A_{k,1}; Remark 4.2 notes Eq. (6) is "stricter than necessary", i.e. no tightness claim; (3) no —
  the exit order is the layer order; (4) no — a binary consistency event; (5) no.

## 9. ORCA: Online Reasoning Calibration — Zhou, Wang, Wu, Zhu, Shi, Wang, Wilson, Jaakkola, Bates
- **Key:** `zhou2026orca` — COLM 2026 (stated on the PDF), arXiv:2604.01170 (v2 read).
- **Summary.** Conformal / LTT calibration of an LLM reasoning *stopping rule* whose probe score s_t = f(φ_t; W_{t−1}) is
  produced by a calibration module updated by test-time training inside each instance (fast weights reset per instance).
  Stopping time τ_λ(x;U) = min{t ≤ T : s_t ≥ λ} (App. A, item v). Reports 40–67 % token savings at δ = 0.1 on Qwen2.5-32B
  and OOD generalisation.
- **Exact guarantee.** Lemma A.1: intra-instance adaptation preserves inter-instance exchangeability of the 0–1 risks
  R_i(λ). Thm. A.2: fixed-sequence testing of H_j : r(λ_j) ≥ δ with binomial p-values p_j = P(Binom(n,δ) ≤ n r̂_n(λ_j))
  (Eq. 22) at level ε returns λ* with P_Dcal(r(λ*) ≤ δ) ≥ 1−ε (Eq. 23). Remark A.3: marginal, not prompt-conditional.
- **Uncertainty model.** Distribution-free over i.i.d. prompts for the *entire deployed procedure* (Eq. 17), including
  its own online updates.
- **Lacks vs RSC-Rank.** (1) yes; (2) no — the calibrated object is the threshold on one probe trajectory; (3) no —
  Thm. A.2 covers the one procedure A_λ that was run on calibration (changing the procedure changes the risk curve);
  (4) no; (5) no.

## 10. Streamlining Conformal Information Retrieval via Score Refinement — Intrator, Cohen, Kelner, Goldenberg, Rivlin, Freedman
- **Key:** `intrator2024streamlining` — FEVER workshop 2024 (EMNLP), pp. 186–191, doi 10.18653/v1/2024.fever-1.22
  (ACL Anthology 2024.fever-1.22; author order there differs from arXiv:2410.02914 — the Anthology order is used).
  **Abstract-level only** (arXiv API); body UNVERIFIED.
- **Summary (abstract).** Conformal prediction "retriev[es] sets guaranteed to include relevant information, yet existing
  approaches produce large-sized sets"; the paper "introduce[s] a score refinement method that applies a simple monotone
  transformation to retrieval scores, leading to significantly smaller conformal sets while maintaining their
  statistical guarantees" (BEIR benchmarks).
- **Exact guarantee.** Split-conformal coverage of the relevant document by the retrieved set (as stated in the abstract;
  theorem numbers UNVERIFIED). The set is a score cutoff, i.e. a calibrated prefix of the retrieval list.
- **Uncertainty model.** Distribution-free over exchangeable queries with relevance labels; no expensive oracle, no stopping.
- **Lacks vs RSC-Rank.** (1) yes (one conformal cutoff across queries); (2) no — prefix, no scored-subset worst case;
  (3) n/a — no scoring policy; (4) no; (5) no evidence in the abstract.

## 11. Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction — Chakraborty, Yang, Khashabi, Lawrie, Duh
- **Key:** `chakraborty2026rag` — ECIR 2026, LNCS "Advances in Information Retrieval", pp. 537–546,
  doi 10.1007/978-3-032-21300-6_45 (Crossref); arXiv:2511.17908 (v2). **Abstract-level only**; body UNVERIFIED.
- **Summary (abstract).** "Context engineering through conformal prediction, a coverage-controlled filtering framework
  that removes irrelevant content while preserving recall of supporting evidence", with embedding- and LLM-based scoring
  functions on NeuCLIR and RAGTIME; "conformal filtering consistently meets its target coverage, ensuring that a specified
  fraction of relevant snippets are retained, and reduces retained context by 2–3x relative to unfiltered retrieval".
- **Exact guarantee.** Coverage (recall of relevant snippets) at a target level via a conformal score threshold on the
  filter (abstract; theorem numbers UNVERIFIED). Again a calibrated cutoff, applied after retrieval.
- **Uncertainty model.** Distribution-free over exchangeable queries with snippet relevance labels; no stopping.
- **Lacks vs RSC-Rank.** (1) yes; (2) no — cutoff/prefix; (3) n/a; (4) no; (5) no evidence in the abstract.

---

### Reading across the eleven
Every work here calibrates one scalar across instances (column 1), so RSC-Rank may **not** claim to be the first
cross-query distribution-free calibration in ranking or early exit. What none of them has is columns 2–5: the calibrated
object is a *prefix size / score cutoff* (Two-stage RC, Conformal@K, Angelopoulos et al., Intrator et al., Chakraborty
et al.), a *common score radius* (CertBind, Eq. 24 — our B_all-type coverage statistic), or the *exit threshold of one
fixed procedure* (Fast yet Safe, Conformal Thinking, ORCA, CATs). CATs' shared-calibration statistic M_max (Eq. 9) is the closest formal cousin of a
"worst case over unsafe states" but ranges over the layers of a single exit sequence — the fixed-trajectory analogue
A^π, which Prop. 5 shows is not policy-uniform. CertBind's certificate |C_k| = k ⇔ gap > 2ρ (Eq. 21) is the σ-shared,
no-stopping special case of the coverage statistic, not the rank-sufficient A_{k,1}. No work states an r-miss frontier
or a finite-population split guarantee over queries. 02_related.tex's paragraphs "Calibrated early exit …" and
"Conformal and risk-controlled retrieval" are consistent with the PDFs; two precise additions are recommended in
`novelty_sweep_2026-09.md` (§ "What 02_related.tex should add").

### Second-pass sanity check (2026-09-21, later session)
Every section/equation pointer in §1–9 was re-located in the extracted PDF texts (CATs Eq. 9 / Remark 4.2 / Thm. 4.4, 4.6;
CertBind Eq. 17, 21, 22, 24; Fast yet Safe Eq. 10, 12 / Prop. 1–2; ORCA Lemma A.1 / Thm. A.2 / Eq. 22–23 / Remark A.3;
Conformal Thinking Eq. 12–14 and the "loss of theoretical rigor" sentence; Two-stage RC Eq. 13, 15, 17 / Thm. 1–3;
Angelopoulos et al. Eq. 1–2 and fixed-sequence testing; Transductive Thm. 3.5 / Prop. 3.8 / Eq. 7–8). Two wording
issues in 02_related.tex surfaced (reported, not edited): (i) Angelopoulos et al. control the **false discovery rate**
(their abstract, Sec. 2.2, Eq. 1), not "recall or coverage" as the sentence groups them; (ii) Fast yet Safe and Conformal
Thinking calibrate with CRC / RCPS-UCB (Bates et al. 2021), only ORCA uses LTT fixed-sequence testing — "with conformal
risk control and Learn-then-Test" is right for the family, not for each member.
