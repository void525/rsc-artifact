# LITERATURE.md — verified neighbours and the claims boundary (checked 2026-09-20)

Sources verified via arXiv API, Crossref, ACL Anthology, PMLR, OpenReview metadata. Full texts of Conformal@K (ACM DL 403)
and IDCR (no arXiv) were not readable; their abstracts were. N01 must read the PDFs of the three nearest neighbours.
N01 (2026-09-21) read the PDFs of the three nearest neighbours and of 21 further works; per-paper notes with exact stopping
rules, guarantees and section/equation references are in docs/lit/bandit_neighbours.md, docs/lit/conformal_neighbours.md,
docs/lit/efficiency_neighbours.md; the 2025–2026 sweep (arXiv API + Semantic Scholar) is in docs/lit/novelty_sweep_2026-09.md
— **no closer neighbour found**; one new adjacent work (ConANN, PVLDB 2025) added to §B.3h and refs.bib.
Second pass (2026-09-21, later session, after the first worker session died): all section/equation pointers re-checked
against the extracted texts, one more full text read (PAC-CF), 40 further arXiv queries, verdict unchanged; see §G for the
file map and counts (22 works full text, 10 abstract only, 7 metadata only).

## A. The ten works named in the plan
| # | Work | Venue | Ids | What it does | Has our threshold? |
|---|---|---|---|---|---|
| a | Two-stage Risk Control with Application to Ranked Retrieval — Xu, Ying, Guo, Wei | IJCAI 2025 pp. 9104–9111 | doi 10.24963/ijcai.2025/1012; arXiv:2404.17769 | LTT/CRC calibration of two cutoff sizes (retrieval set, ranking set) with joint expected-risk control (MSLR-Web, Yahoo LTRC) | No — calibrated *prefix sizes*; = our uniform-prefix baseline family |
| b | Transductive Conformal Inference for Full Ranking — Fermanian, Humbert, Blanchard | NeurIPS 2025 | arXiv:2501.11384 | Prediction sets for the ranks of new items from partially known relative ranks; FCP control | No |
| c | Relevance Scores Calibration for Ranked List Truncation via TMP Adapter — Posokhov et al. | Findings ACL 2025 | doi 10.18653/v1/2025.findings-acl.402 | Adapter + threshold-margin loss so one threshold truncates lists | No (training-time, no guarantee) |
| d | AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking — Yoon, Kim, Cho, Hwang | NeurIPS 2025 | arXiv:2505.18512 | TrueSkill posterior drives which subsets an LLM listwise reranker re-scores and when to stop | No (heuristic; listwise) |
| e | Generate to Accelerate: Improved Reranking via LLM-Generated Pivot Documents — Sinhababu, Bharati, Ganguly, Mitra | arXiv (v3 2026-08-27) | arXiv:2604.09492 | LLM-generated pivot document's score as a query-dependent truncation cutoff (up to 66% fewer pointwise calls) | No (heuristic; generation cost must be counted) |
| f | IDCR: Information-Directed Conformal Retrieval — Nanivadekar et al. | UAI 2026, PMLR 337:4712–4732 | OpenReview 4HAtvSTMHJ | Selects retrieved docs to shrink downstream conformal prediction-set volume | No (unrelated objective) |
| g | CertBind from Multimodal Connectivity to Certifiable Retrieval Decisions — Cao et al. | arXiv 2026-08-06 | arXiv:2608.06516 | Split-conformal residual radius ρ; certified top-k iff s_(k+1) < s_(k) − 2ρ else abstain | Adjacent: uniform-radius *score coverage*, no early stopping, no per-item σ, no all-subset quantifier |
| h | Conformal@K: Distribution-Free Top-K Miss-Risk Control for Recommendation with Overlapping-Group and Two-Stage Guarantees — Fan, Liu, Pan | ACM TOIS 44(6), 2026 | doi 10.1145/3828550 | Calibrated monotone budgets for miss@K with group/two-stage guarantees | No per abstract (budget selection = calibrated prefix); body UNVERIFIED |
| i | Consistent Accelerated Inference via Confident Adaptive Transformers (CATs) — Schuster, Fisch, Jaakkola, Barzilay | EMNLP 2021 | arXiv:2104.08803 | Calibrated early exit consistent with the full model w.p. ≥ 1−ε | No (classification) |
| j | ORCA: Online Reasoning Calibration — Zhou et al. | COLM 2026 | arXiv:2604.01170 | CRC-calibrated LLM sampling/stopping with test-time-trained calibration module | No |

## B. Nearest structural neighbours (must be cited and differentiated explicitly)
1. **Top-k on a Budget: Adaptive Ranking with Weak and Strong Oracles — Oettershagen** (arXiv:2601.20989, Jan 2026). Weak oracle gives per-item CIs [L,U]; strong oracle certifies; the baseline **TA-Certify** (Sec. 8) stops "as soon as the worst verified top-k item is guaranteed to exceed any remaining unverified item" — the same k-th-verified-vs-best-UCB test (his main algorithm ACE, Alg. 2 / Thm. 5, stops when the smallest lower bound among the k items with the largest upper bounds is ≥ the largest upper bound outside, verified or not). Differences: CIs assumed jointly valid from repeated weak pulls (Hoeffding/empirical-Bernstein), *no single λ calibrated across queries*, no query-level statistic, no iff/tightness over all unsafe subsets, no conformal/PAC/finite-population regime, no policy-uniform statement, no r-miss frontier.
2. **Col-Bandit: Query-Time Top-K Estimation for Late-Interaction Retrieval** (arXiv:2602.02827). LUCB-style elimination inside ColBERT MaxSim with finite-population Bernstein–Serfling CIs; single scorer; no cross-query calibration.
3. **PAC Subset Selection in Stochastic Multi-armed Bandits (LUCB)** — Kalyanakrishnan, Tewari, Auer, Stone, ICML 2012. The ancestor of the comparison "k-th best LCB vs best UCB outside" (exactly: UCB(best outside High_t) − LCB(worst inside High_t) < ε, our form at ε = 0). Note: its Thm. 1 holds for "an arbitrary sampling strategy" because the per-arm bounds are anytime-valid — strategy-uniform in that sense, so the paper must not say LUCB gives *no* statement about other orders; the contrast is anytime-valid per-arm CIs vs one calibrated all-subset statistic.
3b. **Fast-Forward indexes / Efficient Neural Ranking using Forward Indexes — Leonhardt, Rudra, Khosla, A. Anand, A. Anand**, WWW 2022 (doi 10.1145/3485447.3511955, arXiv:2110.06051), Sec. 4.4.
   Same venue, same inequality: interpolated sparse+dense reranking stops once the sparse score of the next
   document plus an *estimated* maximum dense score cannot beat the current k-th score. Exact only if the true
   maximum is known; in practice the running maximum is used, justified by a DKW-style argument that assumes iid
   scores (which the authors say does not hold). One global bound, no calibration, structural (not black-box) teacher.
   **Threat: HIGH** (same venue, same test) — must be cited explicitly in the "stopping rules with predicted bounds"
   paragraph as the nearest reranking ancestor.
3c. **Everest / Top-K Deep Video Analytics: A Probabilistic Approach — Lai, Han, Liu, Zhang, Lo, Kao**, SIGMOD 2021
   (doi 10.1145/3448016.3452786, arXiv:2003.00773). Cheap proxy (mixture-density network) gives each item a score
   distribution; an expensive oracle is probed in a loop until the returned top-K equals the exact top-K with
   probability at least a user threshold. Same "cheap proxy + expensive oracle + certify top-K" loop; the guarantee
   is model-based (it inherits the proxy's probability model), not distribution-free, and there is no cross-query
   calibration and no order-uniformity. **Threat: med–high.**
3d. **MPro / Minimal Probing — Chang & Hwang**, SIGMOD 2002 (doi 10.1145/564691.564731). Expensive predicates are
   probed only for objects that are in the current top-k by upper bound; returns the exact top-k. Our rule is
   MPro/TA-style with calibrated soft bounds in place of worst-case hard bounds. **Threat: med (ancestor).**
3e. **Probabilistic TA — Theobald, Weikum, Schenkel**, VLDB 2004 (vldb.org/conf/2004/RS17P3.PDF, pp. 648–659).
   Score-distribution convolutions decide when a candidate can be dropped, with high-probability precision/recall
   relative to the exact top-k. Direct ancestor of probabilistic early termination; model-based, not distribution-free.
   **Threat: med.**
3f. **Threshold estimation for dynamic pruning — Petri, Moffat, Mackenzie, Culpepper, Beck**, SIGIR 2019
   (doi 10.1145/3331184.3331207) and **Mallia, Siedlaczek, Sun, Suel**, CIKM 2020 (doi 10.1145/3340531.3412080).
   Predict the final k-th score to initialise pruning; over-prediction is repaired by re-executing the query, so
   safety comes from re-execution, not from calibration. Learned *threshold*, not learned per-document bounds.
   **Threat: low–med.**
3g. **SUPG / Approximate Selection with Guarantees using Proxies — Kang, Gan, Bailis, Hashimoto, Zaharia**,
   PVLDB 13(12) 2020 (doi 10.14778/3407790.3407804): proxy plus sampled oracle labels with precision/recall targets
   — *selection*, not top-k, and the target is a label-defined rate. **Threat: low–med.**
3h. **Learned early termination in ANN search — Li, Zhang, Andersen, He**, SIGMOD 2020 (doi 10.1145/3318464.3380600)
   and **DARTH — Chatzakis, Papakonstantinou, Palpanas**, PACMMOD 3(4) 2025 (doi 10.1145/3749160): a learned or
   declarative recall target stops the search early. Same "learn when to stop" shape, different guarantee
   (recall of an index scan, not exactness w.r.t. a frozen teacher, and no cross-query distribution-free calibration).
   **Threat: low–med.** Added 2026-09-21: **ConANN — Horchidan, Zeiher, Boström, Carbone**, PVLDB 19(1):29–42, 2025
   (doi 10.14778/3772181.3772184; abstract-level): conformal risk control of the false-negative rate of IVF probing,
   "dynamically adapts the cluster probes required per query" — the *calibrated* member of this family (a budget, not a
   stop test over scored subsets; no teacher). **Threat: low–med**; must be cited in the ANN sentence of 02_related.tex.
4. Deterministic ancestors: **WAND** (Broder et al., CIKM 2003), **Block-Max WAND** (Ding & Suel, SIGIR 2011), **Threshold Algorithm** (Fagin, Lotem, Naor, PODS 2001 / JCSS 2003), Turtle & Flood 1995: skip items whose upper bound cannot beat the current k-th score (σ = 0 analogue).
5. Early-exit / cascade IR lineage: Cambazoglu et al. WSDM 2010; Wang, Lin, Metzler SIGIR 2011; Chen et al. SIGIR 2017; QuickScorer SIGIR 2015; Lucchese et al. SIGIR 2020; Busolin et al. SIGIR 2021, SIGIR 2025 (cross-encoder early exit, doi 10.1145/3726302.3729962); Ranked List Truncation for LLM reranking (Meng et al., SIGIR 2024, arXiv:2404.18185) = learned query-adaptive prefix (no guarantee).
6. Risk-controlled early exit (non-ranking): Fast yet Safe (Jazbec et al., NeurIPS 2024, arXiv:2405.20915); Conformal Thinking (ICML 2026, arXiv:2602.03814); Beyond Greedy Exits (NeurIPS 2025, arXiv:2509.23666) — calibrate exit thresholds for one fixed policy = our fixed-trajectory baseline family.
7. Conformal set sizing in retrieval/RAG: Intrator et al. FEVER 2024 (arXiv:2410.02914; doi 10.18653/v1/2024.fever-1.22, abstract-level block in docs/lit/conformal_neighbours.md §10); Chakraborty et al. ECIR 2026 (arXiv:2511.17908; doi 10.1007/978-3-032-21300-6_45, §11); Angelopoulos et al. COPA 2023 (arXiv:2207.01609 — controls FDR, not recall); Guo et al. arXiv:2306.07188 (`guo2023isrr`); CAS (arXiv:2608.20771, `zhu2026cas`).
8. Also seen: Best Arm Identification with LLM Judges (arXiv:2601.21471, top-1; `ao2026bai`); Adaptive Re-Ranking (arXiv:2606.25249; `genc2026adaptive`); Learning Cascade Ranking as One Network (ICML 2025, arXiv:2503.09492; `wang2025cascadeonenetwork`); Guaranteed Top-Adaptive-K (withdrawn ICLR 2026 submission — no identifier, not in the bib); Prof-K (arXiv:2608.12573; `dziarmaga2026profk`); Optimal Top-k Identification from Pairwise Comparisons (ICML 2026, arXiv:2607.08979; `goldberger2026topk`); Beyond Greedy Exits (NeurIPS 2025, arXiv:2509.23666; `bajpai2025beyondgreedy`).
9. Added by the second-pass sweep (2026-09-21): **PAC-CF** (arXiv:2604.14345, `qian2026paccf`, full text) — one conformal
   score-gap margin calibrated across tasks on a max-over-frontiers deficit of the best verifier-valid candidate (Eq. 7–9),
   used as a top-K *pruning* rule in LLM tree search; prefix-like, existential guarantee, no oracle in the loop, no
   stopping — the nearest new calibrated-margin rule, **threat: low–med** (cite next to CertBind). Conformal Cascade
   (arXiv:2607.25018, `dou2026conformalcascade`): conformal set size as the deferral rule of an LLM cascade — fixed-procedure
   family (§B.6). BlitzRank (ICML 2026, arXiv:2602.05448, `agrawal2026blitzrank`): deterministic certificate that the top-m
   is determined by the transitive closure of k-wise LLM comparisons — the σ = 0 analogue in the comparison-oracle world.
   DCR (arXiv:2601.23128): tighter transductive full-ranking sets via Negative-Hypergeometric rank distributions — orthogonal,
   like Fermanian et al. **TAR stopping rules** — Di Nunzio, WWW '26 Companion (doi 10.1145/3774905.3795598,
   `dinunzio2026tarstopping`, abstract-level): "practical stopping rules based on conservative (lower-bound) recall
   estimates" that fuse calibrated ranker probabilities with a probability sample — a per-query sampling-based recall
   certificate for screening a ranked list (SUPG-like), human relevance as target, no frozen teacher, no cross-query
   calibration; an adjacent family at our venue that the map had not listed. **Threat: low.**

## C. Differentiation table (completed in N01, 2026-09-21, from the PDFs — details in docs/lit/*.md; one row per paper since the second pass)
Columns: (1) one λ calibrated across queries, distribution-free; (2) the threshold *equals* the worst case over all
unsafe scored subsets (iff, Thm. 1–2); (3) one threshold valid under every legal scoring order; (4) r-miss frontier
A_{k,r}; (5) finite-population (hypergeometric random-split) guarantee over queries. "n/a" = no stopping rule / no
scoring policy exists in the method. Retrieval: F = full text read, A = abstract only, M = metadata only (ACM DL 403);
for M rows the justification is secondary-source based and UNVERIFIED against the paper itself.
| Work (retrieval) | (1) cross-query λ | (2) iff over unsafe subsets | (3) policy-uniform | (4) r-miss | (5) finite-pop. | Justification (section / equation of the paper) |
|---|---|---|---|---|---|---|
| Oettershagen 2026, arXiv:2601.20989 (F) | no | no | no | no | no | per-item empirical-Bernstein CIs + Bonferroni on one instance (Sec. 3); ACE stops iff L(worst IN) ≥ U(best OUT) (Alg. 2 l.7); Thm. 5 is sufficiency on event E; TA-Certify (Sec. 8) = our μ-order with CIs; exact top-k only (Sec. 7) |
| Col-Bandit 2026, arXiv:2602.02827 (F) | no | no | no | no | partial (within-query tokens) | drop i iff UCB_i < K-th largest LCB (Alg. 1 l.4–5); radius Eq. 7 with hand-set α_ef; Thm. 1 (App. C) δ-PAC at α_ef = 1 only; Bernstein–Serfling over the T tokens of one query, not over queries |
| LUCB 2012, ICML (F) | no | no | strategy-uniform only | no | no | stop iff UCB(l*) − LCB(h*) < ε (Sec. 3.1); Thm. 1 holds for any sampling strategy because β(u,t) is anytime-valid (Hoeffding + union bound) — not a calibrated statistic; ε is a value slack, not a rank slack |
| CertBind 2026, arXiv:2608.06516 (F) | yes (one radius ρ) | no (score coverage) | n/a | no | no | ρ = √d · split-conformal quantile (Eq. 17); C_k = {j : s_j ≥ s_(k) − 2ρ} (Eq. 24); Certified iff gap > 2ρ (Eq. 21); P(wrong Certified) ≤ δ_p (Eq. 22); nothing is scored sequentially |
| Two-stage Risk Control, IJCAI 2025 (F) | yes (λ,γ) | no | n/a | no | no | score cutoffs Eq. 13/15 (prefixes); LTT Thm. 1 FWER over the grid; CRC Thm. 2–3 expected risk; losses use relevance labels (Eq. 14, 16) |
| Conformal@K, TOIS 2026 (A) | yes (budget) | no (abstract) | n/a | no | UNVERIFIED (body unread) | abstract: "selects monotone budgets so that … the probability that no relevant item appears in the returned Top-K is controlled" — budget = prefix size |
| Transductive conformal full ranking, NeurIPS 2025 (F) | no (within one instance) | n/a | n/a | no | different (n+m items of one instance) | proxy scores Eq. 7, set Eq. 8, Thm. 3.5 marginal validity, Prop. 3.8 FCP; no cost model, no stopping |
| Angelopoulos et al. RecSys/COPA 2023 (F) | yes | no | n/a | no | no | T_λ = {i : s_i ≥ λ} (Sec. 2.2); LTT with fixed-sequence testing, Eq. 1–2; controls the **false discovery rate** w.r.t. true top-m labels (precision-type, not recall) |
| Intrator et al., FEVER 2024 (A) | yes (one conformal cutoff) | no (prefix) | n/a | no | no | abstract: monotone score transformation yields "significantly smaller conformal sets while maintaining their statistical guarantees" — a calibrated score cutoff on the retrieval list; body UNVERIFIED |
| Chakraborty et al., ECIR 2026 (A) | yes (one coverage threshold) | no (cutoff) | n/a | no | no | abstract: "coverage-controlled filtering framework … ensuring that a specified fraction of relevant snippets are retained"; a calibrated filter after retrieval; body UNVERIFIED |
| Conformal Thinking, ICML 2026 (F) | yes | no | no (one procedure) | no | no | upper λ⁺ and dynamic lower λ⁻ (Eq. 12–13) calibrated by UCB/RCPS, Eq. 14; two-step selection "with some loss of theoretical rigor" (Sec. 4.4) |
| Fast yet Safe, NeurIPS 2024 (F) | yes | no | no (fixed exit sequence) | no | no | exit rule Eq. 1; performance-gap / consistency risks Eq. 6, 8; Prop. 1 (CRC, Eq. 10), Prop. 2 (UCB, Eq. 12); monotone in λ along one trajectory |
| CATs, EMNLP 2021 (F) | yes | trajectory-max only (not all subsets) | no (layer order) | no | no | shared statistic M_max(x) = max{M_k(x) : F_k(x) ≠ F(x)} (Eq. 9) = analogue of our A^π; Thm. 4.4/4.6 validity of C_ε ⊇ I(x); Remark 4.2: condition "stricter than necessary" |
| ORCA, COLM 2026 (F) | yes | no | no (one deployed procedure) | no | no | τ_λ = min{t : s_t ≥ λ} (App. A (v)); Lemma A.1 exchangeability under TTT; Thm. A.2 LTT-FST, Eq. 23; Remark A.3 marginal |
| Fast-Forward indexes, WWW 2022 (F) | no (running max) | no | no (sparse order) | no | no | s_best = α φ_S(d_{i−1}) + (1−α) s_D (Eq. 7); stop iff s_best ≤ s_min (Alg. 2 l.8); Thm. 4.1 exact with true max; Thm. 4.3 DKW under an i.i.d. assumption the authors say "can not be ensured" (Sec. 4.4.1) |
| Everest, SIGMOD 2021 (F) | no (model posterior) | no | no (greedy order) | no | no | p̂ = Σ_{W: R̂ = Top-K(W)} Pr(W) (Eq. 1); loop until p̂_i ≥ thres (Sec. 3.3); certain-result condition; guarantee inherits the CMDN's distributions |
| MPro, SIGMOD 2002 (M) | n/a (exact bounds) | n/a | n/a | no | no | expensive predicates probed only for objects still in the top-k by upper bound ("necessary probes"), exact result — secondary description (LITERATURE §B.3d); body not retrieved, UNVERIFIED |
| Threshold Algorithm, JCSS 2003 (M) | n/a (exact bounds) | n/a | n/a | no | no | TA "stops … only when it is certain that no more top-k results can be found" (Theobald 2004 §1.1, read in full); instance-optimal for exact top-k; body not retrieved |
| Probabilistic TA, VLDB 2004 (F) | no (score-distribution model) | no | no (sorted access) | no | no | convolution bounds on P[Σ_{i∉E(d)} S_i ≤ v] to "predict when it is safe, with high probability, to drop candidate items" (abstract, Sec. 3); precision/recall guarantees relative to exact top-k (§1.1) |
| Petri et al., SIGIR 2019 (A) | no (predicted k-th score) | no | no | no | no | term-/query-dependent features predict "the final value of that threshold", used "right from the commencement"; "if the estimated threshold is too high the query will need to be re-executed" (abstract) |
| Mallia et al., CIKM 2020 (A) | no (predicted k-th score) | no | no | no | no | reimplements and compares parametric, sampling and ML estimators of the rank-k score for MaxScore/WAND/BMW (abstract); no stop rule of its own |
| SUPG, PVLDB 2020 (F) | per-query sampled labels | no | n/a (selection) | no | no | RT/PT queries with failure probability (Sec. 1, "Accuracy guarantees"); proxy threshold τ from importance-sampled oracle labels; label-defined rate, not exactness w.r.t. a teacher |
| Li et al., SIGMOD 2020 (A) | no | no | no | no | no | GBDT models "learn and predict when to stop searching for a certain query" from the intermediate result, "under the same high accuracy targets" (abstract); regression, no distribution-free statement |
| DARTH, PACMMOD 2025 (F) | no | no | no | no | no | recall predictor invoked at selected search points, search terminated at the declared target recall (§1); meets targets empirically, 6.8×/13.6× average speed-ups |
| ConANN, PVLDB 2025 (A) | yes (CRC budget) | no (budget) | n/a | no | no | conformal risk control of the IVF false-negative rate, "dynamically adapts the cluster probes required per query" (abstract) — a calibrated probe budget, no stop test over scored subsets, no teacher; body UNVERIFIED |
| WAND, CIKM 2003 (M) | n/a (exact) | n/a | n/a | no | no | skip d iff Σ_{t∈q∩d} UB_t < current k-th score (Col-Bandit §2.2; Petri 2019 abstract); body not retrieved |
| Block-Max WAND, SIGIR 2011 (M) | n/a (exact) | n/a | n/a | no | no | same skip test with block-level maxima (Col-Bandit §2.2; Mallia 2020 abstract); body not retrieved |
| Cascade ranking, Wang–Lin–Metzler SIGIR 2011 (F) | no | no | no | no | no | boosting (Alg. 1, generalising AdaRank) learns per-stage rankers *and* pruning fractions offline (§1, §3); no correctness statement |
| Cost-aware cascade, Chen et al. SIGIR 2017 (A) | no | no | no | no | no | LambdaMART/GBDT cascades balancing "feature costs and the number of documents being re-ranked in each stage" (abstract); no guarantee |
| Early exit, Cambazoglu et al. WSDM 2010 (M) | no | no | no | no | no | title/venue only; placed by Lucchese 2020's abstract in the lineage that "early terminate[s] the evaluation of documents unlikely to be ranked among the top-k"; UNVERIFIED |
| QuickScorer, Lucchese et al. SIGIR 2015 (M) | no | no | no | no | no | fast traversal of additive tree ensembles (title; Lucchese 2020 abstract: "make faster the traversal"); not a stopping rule; body not retrieved |
| Query-level early exit, Lucchese et al. SIGIR 2020 (A) | no | no | no | no | no | per-query decision to stop the ensemble traversal for all candidates; up to 2.2× speed-up and +7.5 % nDCG@10 (abstract); heuristic, no risk statement |
| Learned early exit, Busolin et al. SIGIR 2021 (M) | no | no | no | no | no | "Learning Early Exit Strategies for Additive Ranking Ensembles" (title, Crossref); the *learned* member of the lineage; body not retrieved |
| SEE, Busolin et al. SIGIR 2025 (A) | no | no | no | no | no | "a novel non-learned strategy" exiting query–document pairs by token-embedding similarity, best before the first transformer block, up to 3.5× (abstract); no guarantee |
| Ranked List Truncation, Meng et al. SIGIR 2024 (F) | no | no | no | no | no | cut-off k = f(list) ∈ {1..|L|} (Eq. 1); "supervised RLT methods do not show a clear advantage over … a fixed re-ranking depth"; learned prefix, no guarantee |
| Gen-PART, Sinhababu et al. 2026 (F) | no | no | no | no | no | LLM pivot D* inserted at position p with θ(Q,D_p) ≥ θ(Q,D*) ≥ θ(Q,D_{p+1}) (Eq. 4) sets the cut-off (§4.2); heuristic, generation cost extra |
| AcuRank, NeurIPS 2025 (F) | no | no | no | no | no | TrueSkill posterior; stop when |C| < τ (τ = 10), top-k unchanged, or budget exhausted (Alg. 1 l.6, §3.3); listwise teacher; hand-set ε, τ |
| TMP adapter, Findings ACL 2025 (F) | training-time only | no | no | no | no | threshold-margin penalty (Eq. 1–4) so one validation-chosen threshold transfers ("threshold consistency"); no statistical guarantee |
| PAC-CF, arXiv:2604.14345 (F, second pass) | yes (one margin across tasks) | no (max over the frontiers of one trajectory of a top-1 deficit; pruning set, no stop) | no (fixed controller) | no (existential: ≥ 1 valid survives, Cor. 1) | no | nonconformity C_path = (max_C Ȳ − max_P Ȳ)₊, task score = max over frontiers (Eq. 7); margin = k_Q-th order statistic (Eq. 8); keep {a : s* − s(a) ≤ L̂_Q + ε} (Eq. 9); Prop. 4 marginal; no expensive oracle in the loop |

## D. What we may claim (and may not)
May: first *tight, closed-form* critical value equating the worst case over all unsafe scored subsets of a strict k-th-vs-guardrail
stop rule to a query-level statistic; the r-miss generalisation; a single calibrated threshold protecting *every* legal scoring
order (policy-uniform) in three regimes incl. an exact finite-population (hypergeometric) split guarantee; a lower-bound example
showing policy-uniformity can cost N/k; the exact cost anatomy |V*| + band + batching with the instance-optimal certification core
and the measured price of policy-uniformity (Prop. 10); an implementation-level exactness note (normalised comparison).
May not: first calibrated early exit; first risk-controlled retrieval; first k-th-vs-bound stopping test — the test itself is
inherited from TA/MPro (Fagin 2001/2003, Chang & Hwang 2002), WAND/BMW, LUCB, Oettershagen 2026 and, in reranking at our own
venue, Fast-Forward indexes (Leonhardt et al., WWW 2022); first "cheap proxy + expensive oracle certifies top-K" loop (Everest,
SIGMOD 2021); first *learned* stopping signal in retrieval (Petri 2019, Mallia 2020, Li 2020, DARTH 2025, SUPG 2020);
first conformal method in IR; "SOTA reranking efficiency" (depends on results vs the query-adaptive calibrated prefix).
Also may not (added 2026-09-21 after reading the PDFs): first *cross-query* distribution-free calibration of a stop/exit
threshold — CATs (EMNLP 2021), Fast yet Safe (NeurIPS 2024), Conformal Thinking (ICML 2026) and ORCA (COLM 2026) all do
this for one fixed procedure, and Two-stage RC / Conformal@K / Angelopoulos et al. do it for prefix sizes; first
"worst case over unsafe states" statistic in *any* form — CATs' shared calibration M_max (their Eq. 9) is a max over the
inconsistent layers of one exit sequence, i.e. our trajectory-specific A^π (THEORY §5), so the claim must be stated as
"over *all scored subsets* / *every* legal policy", not merely "worst case"; first certificate of the form "top-k is
certified iff the observed gap clears a calibrated margin" — CertBind's Eq. 21 (gap > 2ρ, one radius, no stopping) is
exactly the σ-shared coverage special case B_all that Prop. 3 dominates; first distribution-free stop in ANN search —
ConANN (PVLDB 2025) calibrates the IVF probe budget with conformal risk control.
The paragraph "stopping rules with predicted bounds" in paper/sections/02_related.tex states this inheritance explicitly; the
novelty rests only on (i) cross-query distribution-free calibration of learned per-document bounds, (ii) the rank-sufficient
statistic A_{k,r}(q) with its iff/tightness and policy-uniform validity, (iii) the finite-population regime, (iv) the cost anatomy.
Consistency check (N01, 2026-09-21): every "yes" in column (1) of §C is a prefix size, a common radius, or a fixed-procedure
exit threshold; no row has a "yes" in columns (2)–(5) (Col-Bandit's "partial" in (5) is within-query, not over queries;
Transductive's finite population is the items of one instance). The seven concrete sentences that 02_related.tex should add
are listed in docs/lit/novelty_sweep_2026-09.md.
Second-pass check (2026-09-21, later session): the three rows added since (Intrator 2024, Chakraborty 2026, PAC-CF 2026) keep
the pattern — their "yes" in column (1) is a conformal score cutoff / coverage threshold / one score-gap margin, and PAC-CF's
max-over-frontiers deficit (its Eq. 7) is, like CATs' M_max, a trajectory-level worst case (our A^π), not the all-subset
A_{k,r}; its guarantee is existential (≥ 1 valid candidate survives), so it is neither an exact-top-k nor an r-miss statement.
Two further "may not" items follow: RSC-Rank may **not** claim the first *calibrated margin* pruning rule for top-K candidate
sets (PAC-CF, arXiv:2604.14345; CertBind, arXiv:2608.06516) — it may claim the first such margin that is *rank-sufficient*
(iff over all unsafe scored subsets) and *policy-uniform*; and it may **not** claim the first certificate of a top-m set from
LLM comparisons (BlitzRank, ICML 2026, deterministic transitive closure) — ours is statistical and about a frozen pointwise
teacher. The claims in "May:" above stand unchanged.

## E. Canonical references (verified ids) — see paper/refs.bib
Vovk et al. 2005 (doi 10.1007/b106715); Lei et al. JASA 2018 (10.1080/01621459.2017.1307116); Angelopoulos & Bates FnT 2023
(10.1561/2200000101); Conformal Risk Control ICLR 2024 (arXiv:2208.02814); Learn then Test, Annals of Applied Statistics 19(2) 2025
(10.1214/24-AOAS1998); RCPS JACM 2021 (10.1145/3478535); Wilks 1941 (10.1214/aoms/1177731788); Park et al. ICLR 2020 (arXiv:2001.00106);
Vovk 2013 ML (10.1007/s10994-013-5355-6); Serfling 1974 (10.1214/aos/1176342611); Bardenet & Maillard 2015 (10.3150/14-BEJ605);
BEIR NeurIPS 2021 D&B (arXiv:2104.08663); Qwen3 Embedding (arXiv:2506.05176); BGE M3 Findings ACL 2024 (10.18653/v1/2024.findings-acl.137);
Sentence-BERT (10.18653/v1/D19-1410); MiniLM (arXiv:2002.10957); Pyserini SIGIR 2021 (10.1145/3404835.3463238); pytrec_eval SIGIR 2018
(10.1145/3209978.3210065); NQ TACL 2019 (10.1162/tacl_a_00276); HotpotQA EMNLP 2018 (10.18653/v1/D18-1259); FiQA WWW'18 companion
(10.1145/3184558.3192301); SciFact EMNLP 2020 (10.18653/v1/2020.emnlp-main.609); MS MARCO (arXiv:1611.09268, key `bajaj2016msmarco`,
added to refs.bib 2026-09-21 — it was listed here but missing from the bib); Wang–Lin–Metzler SIGIR 2011
(10.1145/2009916.2009934); Fagin–Lotem–Naor JCSS 2003 (10.1016/S0022-0000(03)00026-6); Anytime ranking, Mackenzie–Petri–Moffat,
TOIS 40(1) **2022** (10.1145/3467890; Crossref dates the issue 2022-01-31 — corrected from "2021"; key `mackenzie2022anytime`, added).
Bardenet & Maillard 2015 (Bernoulli 21(3), 10.3150/14-BEJ605) was likewise listed here but absent from the bib; added as
`bardenet2015concentration` (Crossref carries no page range, so none is given).

## F. Direct ancestors added to paper/refs.bib (verified 2026-09-21)
leonhardt2022fastforward (10.1145/3485447.3511955, arXiv:2110.06051); lai2021everest (10.1145/3448016.3452786,
arXiv:2003.00773); chang2002mpro (10.1145/564691.564731); theobald2004topk (vldb.org/conf/2004/RS17P3.PDF);
petri2019threshold (10.1145/3331184.3331207); mallia2020threshold (10.1145/3340531.3412080);
kang2020supg (10.14778/3407790.3407804); li2020learnedet (10.1145/3318464.3380600); chatzakis2025darth (10.1145/3749160).
All nine are cited in the "stopping rules with predicted bounds" paragraph of 02_related.tex.
Persistent identifiers were also added to the entries that previously carried only `note = {arXiv:...}`
(oettershagen2026topk, pony2026colbandit, sinhababu2026generate, cao2026certbind, zhang2025qwen3embedding,
nanivadekar2026idcr, kalyanakrishnan2012lucb and twelve note-to-eprint conversions), so that
`paper/tools/check_bib.py` passes: every entry now has a doi, an eprint or a url.
N01 (2026-09-21) added lucchese2015quickscorer (10.1145/2766462.2767733), busolin2021earlyexit (10.1145/3404835.3463088),
horchidan2025conann (10.14778/3772181.3772184); completed the author list of wang2026conformalthinking (nine authors, PMLR 306)
and the pages of fan2026conformalk (1--42). These three new keys are not yet cited in 02_related.tex (check_bib warns, not errors).
Second pass (2026-09-21, later session): intrator2024streamlining gained its DOI (10.18653/v1/2024.fever-1.22), pages and the
ACL-Anthology author order; chakraborty2026rag gained its DOI (10.1007/978-3-032-21300-6_45) and pages; fifteen entries were
added for works named in §B.6–B.9 / §E (bajpai2025beyondgreedy, guo2023isrr, zhu2026cas, ao2026bai, dziarmaga2026profk,
goldberger2026topk, genc2026adaptive, wang2025cascadeonenetwork, bajaj2016msmarco, mackenzie2022anytime, bardenet2015concentration,
qian2026paccf, dou2026conformalcascade, agrawal2026blitzrank, dinunzio2026tarstopping), every field taken from the arXiv
`id_list` record or Crossref. `paper/tools/check_bib.py`: 0 errors, exit 0, 75 entries, 22 never-cited warnings.

## G. Where the evidence lives (file map; sweep dated 2026-09-21)
| File | Content | Retrieval status |
|---|---|---|
| docs/lit/bandit_neighbours.md | Oettershagen 2026, Col-Bandit 2026, LUCB 2012: stop rules, guarantees, uncertainty models, five-column "lacks" | 3 full text (arXiv v1/v2 PDFs, icml.cc PDF) |
| docs/lit/conformal_neighbours.md | Two-stage RC, Conformal@K, CertBind, Transductive full ranking, Angelopoulos et al. 2023, Conformal Thinking, Fast yet Safe, CATs, ORCA, Intrator 2024, Chakraborty 2026 | 8 full text, 3 abstract only (Conformal@K: ACM DL 403; Intrator, Chakraborty: arXiv abstracts + Anthology/Crossref metadata) |
| docs/lit/efficiency_neighbours.md | WAND, BMW, TA, MPro, Probabilistic TA; cascades 2011/2017; early exits 2010/2015/2020/2021/2025; RLT 2024; Gen-PART 2026; AcuRank 2025; TMP 2025; Fast-Forward 2022; Everest 2021; Petri 2019; Mallia 2020; SUPG 2020; Li 2020; DARTH 2025; ConANN 2025 | 10 full text, 7 abstract only, 7 metadata only (dl.acm.org / ScienceDirect 403 — descriptions from named secondary sources, marked UNVERIFIED) |
| docs/lit/novelty_sweep_2026-09.md | Verdict; method (first pass: 36 arXiv `all:` queries + 12 Semantic Scholar queries, 32 + 65 records; second pass: 40 arXiv `all:`/`ti:`/`abs:` queries, 195 records; Semantic Scholar search rate-limited (429) → Crossref keyword fallback, 9 queries, 308 DOIs); candidate tables with one-line reasons; bibliography notes; 15 sentences of 02_related.tex that need attention | PAC-CF full text; all other candidates at abstract level |
Sweep dates: 2026-09-21 (both passes). Raw feeds: /tmp/lit/sweep_arxiv.json, /tmp/lit/sweep_s2.json, /tmp/lit3/sweep2_arxiv.json
(scratch, not versioned). Totals across the four files: 22 works read in full text, 10 at abstract level, 7 at metadata level
(§C has one row per work, 39 rows). **Closer neighbour with the same closed-form all-subset critical value: none found.**
The differentiation columns (2)–(5) are "no"/"n/a" in every row; every "yes" in column (1) is a prefix size, a score cutoff,
a common radius, a probe budget, a fixed-procedure exit threshold or a top-1 score-gap margin.
