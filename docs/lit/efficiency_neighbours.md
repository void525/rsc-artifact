# Efficiency / early-exit IR neighbours of RSC-Rank (read 2026-09-21)

Retrieval status per paper is stated in each block: **[full text]** = PDF read (arXiv / author / VLDB / ACL copy);
**[abstract]** = only the abstract (Semantic Scholar record) was readable; **[metadata]** = title/venue/pages verified
via Crossref, body and abstract not retrievable (dl.acm.org returns 403 to non-browser clients) — for these the
description is taken from *secondary* sources that were read in full (named in the block) and is marked as such.
Column legend as in `bandit_neighbours.md`: (1) cross-query calibrated λ | (2) iff-tightness over all unsafe subsets |
(3) policy-uniform guarantee | (4) r-miss frontier | (5) finite-population regime.

---

## A. Deterministic bound-based pruning (σ = 0 analogue of our guardrail)

### A1. WAND — Broder, Carmel, Herscovici, Soffer, Zien, CIKM 2003 **[metadata]**
- **Key:** `broder2003wand` — doi 10.1145/956863.956944, pp. 426–434 (Crossref).
- **Description (secondary: Col-Bandit §2.2 "Bound-based skipping"; Petri et al. 2019 abstract).** Dynamic pruning
  for disjunctive top-k queries: each term carries a per-term score upper bound; a document is fully evaluated only if
  the sum of upper bounds of its matching terms can exceed the current threshold = "the k-th highest similarity score
  identified so far" (Petri et al.). Rank-safe: exact top-k with respect to the full scoring function.
- **Stopping/skip rule.** skip d iff Σ_{t∈q∩d} UB_t < θ_k (current k-th score). Guarantee: exact by construction; no
  statistics. Body not retrieved — equation numbering UNVERIFIED.
- **Uncertainty model.** None — deterministic per-term upper bounds (σ = 0 analogue).
- **Lacks vs RSC-Rank.** (1) n/a (exact bounds, nothing calibrated); (2) n/a; (3) n/a; (4) no; (5) no.

### A2. Block-Max WAND — Ding & Suel, SIGIR 2011 **[metadata]**
- **Key:** `ding2011bmw` — doi 10.1145/2009916.2010048, pp. 993–1002 (Crossref).
- **Description (secondary: Col-Bandit §2.2; Mallia et al. 2020 abstract lists BMW among the algorithms whose
  performance depends on a good initial threshold).** Same skip test as WAND with tighter *block-level* maximum scores
  stored in the inverted lists. Exact. Body not retrieved — description UNVERIFIED against the PDF.
- **Uncertainty model.** None — deterministic block-level upper bounds.
- **Lacks vs RSC-Rank.** identical to A1.

### A3. Threshold Algorithm — Fagin, Lotem, Naor, JCSS 66(4), 2003 (PODS 2001) **[metadata]**
- **Key:** `fagin2003optimal` — doi 10.1016/S0022-0000(03)00026-6, pp. 614–656 (Crossref).
- **Description (secondary: Theobald et al. 2004 §1.1, read in full; Oettershagen 2026 §2).** Sorted access on each
  attribute list, random access for the missing attributes; maintain the threshold τ = aggregate of the last seen local
  scores; "TA … stops scanning index lists only when it is certain that no more top-k results can be found" (Theobald
  §1.1), i.e. when the k-th best aggregated score seen ≥ τ. Instance-optimal for exact top-k. Body not retrieved
  (ScienceDirect 403) — the description is secondary and UNVERIFIED against the paper itself.
- **Uncertainty model.** None — the threshold is an exact bound on every unseen aggregate.
- **Lacks vs RSC-Rank.** (1) n/a; (2) n/a (instance-optimality is a cost statement, not a calibration tightness);
  (3) n/a; (4) no; (5) no.

### A4. Minimal Probing (MPro) — Chang & Hwang, SIGMOD 2002 **[metadata]**
- **Key:** `chang2002mpro` — doi 10.1145/564691.564731, pp. 346–357 (Crossref: title "Minimal probing").
- **Description (secondary: docs/LITERATURE.md §B.3d, verified 2026-09-20; body not re-read here).** Expensive
  predicates are probed only for objects that are in the current top-k by upper bound ("necessary probes"); returns the
  exact top-k with a minimal number of probes. Description UNVERIFIED against the PDF in this pass (Semantic Scholar
  holds no abstract for this DOI; dl.acm.org 403).
- **Uncertainty model.** None — hard upper bounds on the unevaluated expensive predicates.
- **Lacks vs RSC-Rank.** (1) n/a — hard upper bounds; (2) n/a; (3) n/a; (4) no; (5) no.

### A5. Probabilistic TA — Theobald, Weikum, Schenkel, VLDB 2004 **[full text]**
- **Key:** `theobald2004topk` — vldb.org/conf/2004/RS17P3.PDF, pp. 648–659.
- **Summary (abstract + §1.1).** "A family of approximate top-k algorithms based on probabilistic arguments. When
  scanning index lists … in descending order of local scores, various forms of convolution and derived bounds are
  employed to predict when it is safe, with high probability, to drop candidate items and to prune the index scans."
  Score distributions per index list (histograms / Poisson estimates) are convolved to bound
  P[Σ_{i∉E(d)} S_i ≤ v] for the unseen dimensions; a candidate is dropped when its probability of still qualifying for
  the top-k falls below a user threshold ε; guarantees are "about the error relative to 'exactly top-k' queries,
  translatable into guarantees about query-result precision and recall" (§1.1).
- **Uncertainty model.** Model-based (assumed score distributions per list); no held-out calibration; per-query error.
- **Lacks vs RSC-Rank.** (1) no — ε is a model probability, not calibrated across queries; (2) no; (3) no — the scan
  order is the sorted-access order; (4) no; (5) no.

## B. Learned / cascade / early-exit rerankers (no distribution-free guarantee)

### B1. Cascade ranking — Wang, Lin, Metzler, SIGIR 2011 **[full text]**
- **Key:** `wang2011cascade` — doi 10.1145/2009916.2009934, pp. 105–114.
- **Summary.** "A cascade ranking model … a sequence of increasingly complex ranking functions that progressively prune
  and refine the set of candidate documents"; a boosting algorithm (generalising AdaRank) learns per-stage rankers *and*
  the fraction of candidates to prune at each stage, directly optimising an effectiveness–efficiency trade-off
  (abstract, §1). Stopping = the cascade depth and pruning fractions learnt offline; no correctness guarantee.
- **Uncertainty model.** None — pruning fractions are training-time hyper-parameters (Alg. 1, boosting); no risk statement.
- **Lacks vs RSC-Rank.** (1) no; (2) no; (3) no; (4) no; (5) no. Learned prefix schedule, not a certified stop.

### B2. Cost-aware cascade — Chen, Gallagher, Blanco, Culpepper, SIGIR 2017 **[abstract]**
- **Key:** `chen2017cascade` — doi 10.1145/3077136.3080819, pp. 445–454.
- **Summary (abstract).** Cascades that "directly leverage … LambdaMART and Gradient Boosted Decision Trees" and
  balance "feature costs and the number of documents being re-ranked in each stage of the cascade … to maximize both
  efficiency and effectiveness." No guarantee.
- **Uncertainty model.** None (abstract; body UNVERIFIED).
- **Lacks vs RSC-Rank.** all five: no.

### B3. Early exit for additive ensembles — Cambazoglu et al., WSDM 2010 **[metadata]**; QuickScorer — Lucchese et al., SIGIR 2015 **[metadata]**; query-level early exit — Lucchese et al., SIGIR 2020 **[abstract]**; learned early exit — Busolin et al., SIGIR 2021 **[metadata]**; cross-encoder early exit (SEE) — Busolin et al., SIGIR 2025 **[abstract]**
- **Keys:** `cambazoglu2010early` (10.1145/1718487.1718538, pp. 411–420); `lucchese2015quickscorer` (10.1145/2766462.2767733,
  pp. 73–82, added to refs.bib); `lucchese2020query` (10.1145/3397271.3401256); `busolin2021earlyexit`
  (10.1145/3404835.3463088, pp. 2217–2221, added); `busolin2025earlyexit` (10.1145/3726302.3729962, pp. 2534–2544).
- **Summary.** Lucchese 2020 (abstract): prior work "early terminate[s] the evaluation of documents that are unlikely to
  be ranked among the top-k" during the traversal of an additive tree ensemble; the 2020 paper decides *per query*
  whether to stop the traversal for all candidates, reporting up to 2.2× speed-up and up to +7.5 % nDCG@10. Busolin 2025
  (abstract): "Similarity-based Early Exit (SEE), a novel non-learned strategy that exploits the similarities between
  query and document token embeddings to early-terminate the inference of documents that will most likely be
  non-relevant", best applied before the first transformer block; up to 3.5× speed-up "with a limited loss in ranking
  effectiveness" on 17 datasets. Cambazoglu 2010, QuickScorer 2015 and Busolin 2021: titles/venues verified; bodies not
  retrieved; described in Lucchese 2020's abstract as the lineage of "faster traversal" and "early termination" of
  additive ensembles.
- **Stopping rule / guarantee.** Heuristic or learned exits; effectiveness measured empirically; no risk statement.
  Note for 02_related.tex: of the three keys it cites as "learned early exits", only Busolin 2021 (uncited) *learns* the
  exit; SEE (Busolin 2025) is by its own abstract "a novel non-learned strategy", and Lucchese 2020's abstract does not
  describe a learned exit; Cambazoglu 2010 is metadata-only here (UNVERIFIED).
- **Uncertainty model.** None — score/rank thresholds or classifiers, no calibrated error statement.
- **Lacks vs RSC-Rank.** all five: no. (Different object as well: exits *inside* one document's scoring, whereas
  RSC-Rank stops scoring *further documents*.)

### B4. Ranked List Truncation for LLM-based re-ranking — Meng, Arabzadeh, Askari, Aliannejadi, de Rijke, SIGIR 2024 **[full text]**
- **Key:** `meng2024rlt` — doi 10.1145/3626772.3657864, pp. 141–151; arXiv:2404.18185.
- **Summary.** Reproducibility study of 8 RLT methods (fixed-k, Greedy-k, Surprise, Choppy, AttnCut, BiCut, LeCut, …)
  in the "retrieve-then-re-rank" setting (TREC-DL 19/20; BM25/SPLADE++/RepLLaMA × RankLLaMA/monoT5). Cut-off
  k = f([x_1..x_|L|]) ∈ {1..|L|} (Eq. 1) predicted from retrieval scores / list statistics. Findings: oracle per-query
  cut-offs beat every fixed depth (paired t-test), but "supervised RLT methods do not show a clear advantage over using a
  fixed re-ranking depth"; ≈30 % of queries need no reranking with RepLLaMA, ≈5 % with BM25.
- **Stopping rule / guarantee.** Learned query-adaptive prefix; none.
- **Uncertainty model.** None — supervised/unsupervised cut-off predictors evaluated by effectiveness/efficiency only.
- **Lacks vs RSC-Rank.** all five: no. This is the uncalibrated ancestor of our "query-adaptive calibrated prefix" baseline.

### B5. Generate to Accelerate — Sinhababu, Bharati, Ganguly, Mitra, arXiv 2026 **[full text]**
- **Key:** `sinhababu2026generate` — arXiv:2604.09492 (v3, 27 Aug 2026; no venue on the PDF).
- **Summary.** An LLM generates a pseudo-relevant pivot document D* for the query; Gen-PART scores D* with the
  first-stage retriever θ and inserts it into the ranked list at position p with θ(Q,D_p) ≥ θ(Q,D*) ≥ θ(Q,D_{p+1})
  (Eq. 4); p is the pointwise reranking cut-off (§4.2). Listwise variants use D* as a comparison pivot. Reported:
  up to 66 % fewer pointwise inferences, up to 2.95× listwise speed-up, comparable/improved effectiveness.
- **Stopping rule / guarantee.** Query-dependent prefix cut-off from a heuristic pivot; no guarantee; generation cost
  of D* is extra.
- **Uncertainty model.** None — the pivot's first-stage score is used as a point estimate of the relevance boundary.
- **Lacks vs RSC-Rank.** all five: no.

### B6. AcuRank — Yoon, Kim, Cho, Hwang, NeurIPS 2025 **[full text]**
- **Key:** `yoon2025acurank` — arXiv:2505.18512 (v2; "39th Conference on Neural Information Processing Systems" on the PDF).
- **Summary.** Listwise LLM reranking with a Bayesian TrueSkill posterior (μ_i, σ_i) per document, initialised from
  first-stage scores (σ_i = μ_i/3). Each round selects the uncertain set C = {D_i : ε < s_i < 1−ε} with
  s_i = P(x_i > t(k)), partitions it by μ into groups of size m, reranks each group, and updates the posterior (Alg. 1).
- **Stopping rule (Alg. 1 line 6, §3.3).** Stop when (i) |C| < τ (default τ = 10), or (ii) the top-k set is unchanged for
  a fixed number of iterations, or (iii) the call budget is exhausted. No guarantee; the reranker is listwise (a
  document's score depends on its context), so the teacher is not a fixed pointwise function.
- **Uncertainty model.** Bayesian (TrueSkill posterior per document); probabilities are model-internal, never calibrated
  against held-out queries.
- **Lacks vs RSC-Rank.** (1) no — ε, τ are hand-set; (2) no; (3) no; (4) no; (5) no.

### B7. TMP Adapter — Posokhov, Masliukhin, Skrylnikov, Tirskikh, Makhnytkina, Findings of ACL 2025 **[full text]**
- **Key:** `posokhov2025tmp` — doi 10.18653/v1/2025.findings-acl.402, pp. 7728–7734.
- **Summary.** A residual/GeLU adapter on bi-encoder cosine scores trained with an additive Threshold Margin Penalty
  TMP = w_pos P_pos + w_neg P_neg − w_m R_m (Eq. 1–4) with learnable margins m⁺, m⁻, so that one *global* truncation
  threshold transfers across queries and datasets (FiQA, NFCorpus, Robust04). Training-time calibration; the reported
  goal is "threshold consistency" (deviation of the validation-optimal threshold from the test-optimal one).
- **Stopping rule / guarantee.** Global score threshold; no statistical guarantee.
- **Uncertainty model.** None — the adapter reshapes scores so that a single validation-chosen threshold transfers.
- **Lacks vs RSC-Rank.** (1) *training-time* single threshold, no distribution-free calibration; (2)–(5) no.

## C. Direct ancestors of the "cheap bound + expensive oracle" test (LITERATURE.md §B.3b–3h)

### C1. Fast-Forward indexes — Leonhardt, Rudra, Khosla, A. Anand, A. Anand, WWW 2022 **[full text]**
- **Key:** `leonhardt2022fastforward` — doi 10.1145/3485447.3511955, pp. 266–276; arXiv:2110.06051.
- **Summary (§4.4).** Interpolated score φ = α φ_S + (1−α) φ_D over the sparse-ranked list; after the priority queue of
  size k is full, s_best = α φ_S(q,d_{i−1}) + (1−α) s_D (Eq. 7) with s_D an upper limit on dense scores; **Alg. 2 line 8:
  stop iff s_best ≤ s_min** (the current k-th score). Thm. 4.1: with the *true* maximum s_D the returned scores are the
  exact top-k. Since dense scores are unnormalised, s_D is approximated by the running maximum; Thm. 4.3 (DKW) bounds
  P(F(z) < 1−ε) ≤ e^{−2nε²} for the maximum z of an i.i.d. sample — and the authors state the i.i.d. assumption "can not
  be ensured" because scores are sorted by sparse score. Table 4 (Passage'19, k_S = 5000): identical RR@10 with early
  stopping at 72 ms (TCT-ColBERT) / 52 ms (ANCE) per query vs 114 ms for plain Fast-Forward interpolation.
- **Lacks vs RSC-Rank.** (1) no — one running global maximum, nothing calibrated on queries; (2) no; (3) no — sparse
  order only; (4) no; (5) no. Same venue, same inequality as Eq. (stop) with μ_j + λσ_j replaced by a global constant.

### C2. Everest — Lai, Han, Liu, Zhang, Lo, Kao, SIGMOD 2021 **[full text]**
- **Key:** `lai2021everest` — doi 10.1145/3448016.3452786, pp. 1037–1050; arXiv:2003.00773.
- **Summary (§3).** A convolutional mixture-density network gives each frame a score *distribution* (uncertain relation
  D_0); Phase 2 repeatedly picks the uncertain frame f* = argmax E[X_f] (expected confidence gain), obtains its exact
  score from the oracle detector, and recomputes p̂ = Pr(R̂ = R) under possible-world semantics (Eq. 1); **the loop
  "is repeated until a Top-K result R̂_i … has a confidence p̂_i that exceeds thres"** (§3.3), subject to the
  certain-result condition (all returned frames oracle-scored). Speed-ups 14.3–20.6× vs baselines.
- **Uncertainty model.** Model-based: p̂ is computed *from the proxy's own distributions*; the guarantee inherits the
  CMDN's calibration and is per query.
- **Lacks vs RSC-Rank.** (1) no; (2) no — p̂ ≥ thres is a sufficiency stop on a model posterior; (3) no — one greedy
  selection order; (4) no — exact set only; (5) no.

### C3. Threshold estimation for dynamic pruning — Petri, Moffat, Mackenzie, Culpepper, Beck, SIGIR 2019 **[abstract]**; Mallia, Siedlaczek, Sun, Suel, CIKM 2020 **[abstract]**
- **Keys:** `petri2019threshold` (10.1145/3331184.3331207, pp. 485–494); `mallia2020threshold` (10.1145/3340531.3412080, pp. 2141–2144).
- **Summary (abstracts).** Petri: predict "the final value of that threshold" (the k-th score) from term/query features
  and start pruning with it; "if the estimated threshold is too high the query will need to be re-executed in order to
  assure the correct answer", so prediction must be risk-sensitive. Mallia: compares parametric, sampling and ML
  estimators of the rank-k score by error, over-estimate likelihood and end-to-end effect on MaxScore/WAND/BMW.
- **Uncertainty model.** Point prediction of the k-th score (regression / sampling estimators); correctness restored by
  re-execution, not by a probabilistic statement (abstracts; bodies UNVERIFIED).
- **Lacks vs RSC-Rank.** (1) no — a learned *threshold*, safety by re-execution; (2)–(5) no.

### C4. SUPG — Kang, Gan, Bailis, Hashimoto, Zaharia, PVLDB 13(12), 2020 **[full text]**
- **Key:** `kang2020supg` — doi 10.14778/3407790.3407804, pp. 1990–2003; arXiv:2004.00827.
- **Summary (§1).** Approximate *selection* with a cheap proxy and a budget of oracle labels: recall-target (RT) and
  precision-target (PT) queries "subject to an oracle label budget and a failure probability"; the algorithm samples
  records to label (importance sampling on proxy scores) and estimates "a proxy confidence threshold τ for which it is
  safe to return all records with proxy score above τ", using confidence intervals and multiple-testing corrections.
- **Lacks vs RSC-Rank.** (1) *per-query* sampled labels, not cross-query calibration; (2) no; (3) n/a (selection, no
  scoring order); (4) no; (5) no — and the target is a label-defined rate, not exactness w.r.t. a frozen teacher.

### C5. Learned early termination in ANN — Li, Zhang, Andersen, He, SIGMOD 2020 **[abstract]**; DARTH — Chatzakis, Papakonstantinou, Palpanas, PACMMOD 3(4) 2025 **[full text, arXiv:2505.19001]**
- **Keys:** `li2020learnedet` (10.1145/3318464.3380600, pp. 2539–2554); `chatzakis2025darth` (10.1145/3749160).
- **Summary.** Li: gradient-boosted trees "learn and predict when to stop searching for a certain query" using the
  intermediate result as a runtime feature; same accuracy with up to 7.1× lower latency. DARTH: the user declares a
  target recall; a recall predictor is invoked at selected points of the HNSW/IVF search and the search is terminated
  early once the predicted recall reaches the target; "effectively meets user-defined recall targets" empirically,
  6.8×/13.6× average speed-ups.
- **Uncertainty model.** Regression (GBDT) point predictions of "how much more search" (Li) / of the current recall
  (DARTH); accuracy targets are met empirically, no distribution-free statement. Note for 02_related.tex: Li 2020's
  abstract says the models "learn and predict when to stop searching" under "high accuracy targets" — the sentence
  "stops when a predicted recall target is reached" describes DARTH exactly and Li 2020 only approximately.
- **Lacks vs RSC-Rank.** (1) no — regression targets, no distribution-free calibration; (2)–(5) no.

### C6. ConANN — Horchidan, Zeiher, Boström, Carbone, PVLDB 19(1), 2025 **[abstract]** (added in the second pass)
- **Key:** `horchidan2025conann` — doi 10.14778/3772181.3772184, pp. 29–42 (Crossref; Semantic Scholar record
  `journals/pvldb/HorchidanZBC25`; no arXiv id found). Abstract-level; body UNVERIFIED.
- **Summary (abstract).** "The first framework to provide formal, distribution-free error guarantees for IVF-based ANN
  search by leveraging recent advances in Conformal Risk Control"; "tightly controls approximation error, achieving a
  worst-case False Negative Rate deviation within 0.03 percentage points of the target", "dynamically adapts the cluster
  probes required per query", integrated into FAISS.
- **Stopping rule / guarantee.** Conformal risk control (Angelopoulos et al. 2024) of the false-negative rate of the
  probed clusters ⇒ a calibrated per-query *probe budget*; no teacher, no stop test over scored subsets (abstract only).
- **Uncertainty model.** Distribution-free over exchangeable queries (CRC); the controlled loss is index recall, not
  agreement with a frozen scorer.
- **Lacks vs RSC-Rank.** (1) yes (one CRC-calibrated budget across queries); (2) no — budget/prefix analogue; (3) n/a —
  no scoring order; (4) no; (5) no evidence in the abstract. The calibrated member of the C5 family.

---

### Reading across the lineage
The comparison "k-th score obtained so far vs. an upper bound on what is unscored" is the invariant of this whole
line: exact bounds (WAND/BMW/TA/MPro), model-based bounds (Probabilistic TA, Everest), a running empirical maximum
(Fast-Forward, WWW 2022), or a predicted k-th score repaired by re-execution (Petri, Mallia). What none of them does is
calibrate the bound: either it is exact by construction, or its correctness is inherited from a model, or safety comes
from re-running the query. The learned-prefix family (cascades, RLT, Gen-PART, TMP, AcuRank, early exits) has no risk
statement at all. Hence the paragraph "Stopping rules with predicted bounds" in 02_related.tex is accurate; the only
additions warranted are the two extra keys added here (QuickScorer 2015, Busolin 2021) if the early-exit sentence is
meant to cover the additive-ensemble lineage completely.

### Second-pass sanity check (2026-09-21, later session)
The "[full text]" claims were re-checked against the extracted texts on disk (Fast-Forward Eq. 7 / Alg. 2 / Thm. 4.1,
4.3 and the "can not be ensured" sentence; Everest Eq. 1 and "exceeds thres" (§3.3); AcuRank Alg. 1 line 6 and τ = 10;
RLT Eq. 1 and "do not show a clear advantage"; Gen-PART pivot / 66 %; TMP Eq. 1–4 and "threshold consistency"; DARTH
target recall / 6.8× / 13.6×; Theobald §1.1 quotes; Wang 2011 boosting / "progressively prune"); SUPG (arXiv:2004.00827)
was re-fetched and its §1 quotes confirmed. Nothing had to be corrected; "Uncertainty model" lines and one block (C6,
ConANN, abstract-level) were added so that every paper covered here has the same five fields. Retrieval status of this
file: 10 full text (A5, B1, B4–B7, C1, C2, C4, C5-DARTH), 7 abstract only (B2, B3-Lucchese 2020, B3-Busolin 2025,
C3 ×2, C5-Li 2020, C6), 7 metadata only (A1–A4, B3-Cambazoglu 2010, B3-QuickScorer, B3-Busolin 2021).
