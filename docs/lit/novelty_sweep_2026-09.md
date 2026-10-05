# Novelty sweep 2025–2026 (run 2026-09-21)

## Verdict
**No closer neighbour found.** No 2025–2026 paper retrieved in this sweep states a closed-form *all-subset* critical value
(worst case over every scored subset that misses a true winner equated to a query-level statistic) for a calibrated
stop rule, nor a policy-uniform guarantee, an r-miss frontier, or a finite-population split guarantee for early-stopped
reranking. The nearest structural neighbours remain the three in `bandit_neighbours.md` (Oettershagen 2026, Col-Bandit
2026, LUCB 2012) and, among calibrated methods, CertBind 2026 (common-radius score coverage) and CATs 2021 (max over the
unsafe states of one fixed exit sequence). One *new* adjacent work surfaced — **ConANN (PVLDB 2025)**, conformal risk
control for the number of probed IVF clusters in ANN search — which belongs to the "learned/declarative early termination
in ANN" sentence of 02_related.tex (same family as Li 2020 / DARTH 2025, now with a distribution-free budget). It is not
closer than the known neighbours (calibrated *budget*, no stop test over scored subsets, recall of an index scan).
**Re-confirmed in a second pass on 2026-09-21** (40 further arXiv queries over `all:`/`ti:`/`abs:`, 195 unique 2025+ ids;
§ "Supplementary sweep" below): the nearest new item, **PAC-CF** (arXiv:2604.14345, full text read), calibrates one
score-gap margin across tasks on a max-over-frontiers deficit of the best *valid* candidate — a calibrated pruning set
with an existential guarantee, not a closed-form all-subset critical value for a k-th-vs-guardrail stop rule; no paper
found states our A_{k,r}, a policy-uniform guarantee, an r-miss frontier or a finite-population split guarantee.

## Method
- **arXiv API** (https://export.arxiv.org/api/query, `all:` field search, max_results = 50, newest first; 36 queries,
  3.2 s apart; entries before 2025-01-01 discarded). Note: the plain-http endpoint redirects (301) and Python `urllib`
  received HTTP 406 for every query regardless of headers, so queries were issued with `curl -H "Accept:
  application/atom+xml"`. Queries included the five requested ones ("conformal reranking early", "certified top-k early
  termination", "weak strong oracle top-k", "conformal early stopping reranking", "calibrated early exit ranking
  guarantee", "top-k identification proxy oracle guarantee") plus broader two/three-term variants ("conformal AND
  reranking", "early stopping AND reranking", "weak oracle AND strong oracle", "conformal AND top-k AND exact",
  "PAC AND top-k AND reranking", "risk control AND reranking", "conformal AND cascade AND ranking", "conformal AND
  cross-encoder", "conformal AND anytime AND ranking", "rank-sufficient", "policy-uniform", …). 32 unique 2025+ ids.
- **Semantic Scholar** (`/graph/v1/paper/search`, year 2025-2026, limit 50, fields title/abstract/externalIds/venue;
  12 queries: "conformal early stopping reranking", "certified top-k early termination", "weak strong oracle top-k
  identification", "calibrated early exit ranking guarantee", "top-k identification proxy oracle guarantee",
  "conformal reranking efficiency guarantee", "risk control reranking cost", "distribution-free early stopping top-k
  retrieval", "stopping rule reranking calibrated threshold policy", "adaptive reranking budget statistical guarantee
  top-k exact", "conformal prediction cascade retrieval cost", "cheap model expensive reranker certify top-k").
  65 unique records.
- **Re-check of the "also seen" items** of docs/LITERATURE.md §B.8 by arXiv `id_list`.
- Google Scholar / ACM DL full-text search were not available (403 to non-browser clients).

## Candidates examined (one line each; "not closer" = none of columns 2–5 of the differentiation table is present)
| id | title (year) | why it is / is not closer |
|---|---|---|
| arXiv:2601.20989 | Top-k on a Budget: Adaptive Ranking with Weak and Strong Oracles (2026) | known nearest neighbour (see bandit_neighbours.md §1); per-instance joint CIs, no cross-query calibration, no iff, no policy-uniformity |
| arXiv:2602.02827 | Col-Bandit (2026) | known neighbour (§2); LUCB-style elimination inside MaxSim; finite population = tokens of one query |
| arXiv:2608.06516 | CertBind (2026) | known neighbour; common conformal radius ⇒ score coverage, certificate gap > 2ρ (Eq. 21); no stopping |
| doi 10.14778/3772181.3772184 | **ConANN: Conformal Approximate Nearest Neighbor Search** — Horchidan, Zeiher, Boström, Carbone, PVLDB 19(1):29–42 (2025) | NEW adjacent: conformal risk control of the false-negative rate of IVF probing, "dynamically adapts the cluster probes required per query"; a calibrated search *budget* (prefix analogue) for ANN recall, not a stop test over scored subsets; no teacher, no policy-uniformity. Add to the ANN sentence of 02_related.tex; entry `horchidan2025conann` added to refs.bib (abstract-level; body not read) |
| arXiv:2601.21471 | Best Arm Identification with LLM Judges and Limited Human (2026) | cheap biased proxy + selectively observed ground truth, top-1 BAI with anytime-valid confidence sequences; per-instance, top-1 only, no calibration across instances — not closer |
| arXiv:2608.12573 | Prof-K: Probabilistic One-Pass Filtering for Efficient Top-k Selection (2026) | random-sample threshold + exact top-k on a buffer, P(exact) ≥ 1−ε; all elements are read exactly, no cheap predictor, no calibration — not closer |
| arXiv:2607.08979 | Optimal Top-k Identification from Pairwise Comparisons (ICML 2026) | fixed-confidence top-k from noisy pairwise comparisons, asymptotic optimality; single stochastic oracle — not closer |
| arXiv:2606.25249 | Adaptive Re-Ranking (2026) | per-query *routing* among BM25 / MiniLM / BGE rerankers via a trained classifier; no guarantee — not closer |
| arXiv:2604.09492 | Generate to Accelerate (2026) | LLM pivot document sets the reranking cut-off; heuristic — not closer (efficiency_neighbours.md §B5) |
| arXiv:2505.18512 | AcuRank (NeurIPS 2025) | TrueSkill uncertainty drives listwise re-scoring; stop when few uncertain docs; no guarantee — not closer |
| arXiv:2609.00588 | Quit While You're Ahead: Quit for Efficient Candidate Generation in MT Reranking (2026) | incremental generate-and-rerank, stop "when the best reranking score stabilizes"; heuristic, MT — not closer |
| arXiv:2607.14157 | Certified Domain Consistency for Multi-Domain Retrieval (C3R) (2026) | RCPS-based per-domain contamination budgets with abstention; a set-size/budget certificate, not a stop rule — not closer |
| arXiv:2608.20771 | CAS: Conformalized Agentic Search (2026) | APS-based dynamic document truncation for search agents (calibrated prefix) — not closer |
| arXiv:2602.03814 | Conformal Thinking (ICML 2026) | calibrated upper/lower exit thresholds for one reasoning procedure — fixed-trajectory family (conformal_neighbours.md §6) |
| arXiv:2604.01170 | ORCA (COLM 2026) | LTT-calibrated stopping threshold of one deployed procedure with test-time training — fixed-trajectory family (§9) |
| arXiv:2604.01413 | Adaptive Stopping for Multi-Turn LLM Reasoning (MiCP) (2026) | per-turn error budgets with an overall CP coverage guarantee for multi-turn pipelines; fixed procedure, no top-k — not closer |
| arXiv:2512.05325 | LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning (2025) | probe on hidden states + split conformal control of premature exits; one exit sequence — not closer |
| arXiv:2602.03043 | SAFE-KD: Risk-Controlled Early-Exit Distillation for Vision Backbones (2026) | CRC-calibrated per-exit thresholds with selective risk; classification, one exit sequence — not closer |
| arXiv:2509.23666 | Beyond Greedy Exits (NeurIPS 2025) | bandit-adapted exit threshold for early-exit DNNs; no distribution-free stop statistic — not closer |
| arXiv:2602.01476 | Conformal Prediction for Early Stopping in Mixed Integer Optimization (2026) | NN estimates optimality gap, conformal threshold on solver state; single trajectory per instance, not ranking — not closer |
| arXiv:2512.09850 | Conformal bandits (2025) | CP inside regret-minimising bandit policies; coverage of rewards, no top-k stop — not closer |
| arXiv:2607.16209 | Shapley Context Pruning (2026) | Monte-Carlo Shapley attribution with sample-complexity bounds for preserving top-K subsets; a training/attribution bound, not a stop rule — not closer |
| arXiv:2505.23059 | State Machine Reasoning … Overthinking in IR (EMNLP 2025) | discrete Refine/Rerank/Stop actions for LLM reasoning in IR; heuristic — not closer |
| arXiv:2306.07188 | Inference-time Stochastic Ranking with Risk Control (2023, v3) | utility/fairness risk control of PL sampling; no efficiency stop — not closer |
| arXiv:2606.28620 | Reproducing FACTER (2026) | fairness via conformal thresholding in LLM recommendation; no stopping — not closer |
| arXiv:2602.18389, 2506.12287 | weak–strong distance-oracle clustering (2026, 2025) | same two-oracle *model* applied to k-means/k-center; not top-k ranking — not closer, but confirms the "weak/strong oracle" vocabulary is the one to search |
| others (25 arXiv hits) | e.g. "conformal" matching "conformation"/"conforming" (AlloGen, VAMOS, EquiCPI, FastCSP, crystals), requirements engineering, ASR, OT | false positives of the `all:` search — discarded |
| S2 noise (≈45 records) | generic conformal-prediction papers (online CP, DP-CP, robust CP, CP for GNNs, …) | no ranking / stopping content — discarded |

## Supplementary sweep (second pass, 2026-09-21, later session)
**Verdict unchanged: no closer neighbour found.** The first pass used phrase-conjunction `all:` queries only; this pass
added 40 arXiv API queries over `all:`, `ti:` and `abs:` fields (issued with `curl`, 3.2 s apart, max_results = 50,
newest first, entries before 2025-01-01 discarded): "conformal" ∧ "early exit"; "risk control" ∧ "early exit"; "top-k" ∧
"conformal" ∧ "guarantee"; "cross-encoder" ∧ "early termination"; "reranking" ∧ "budget" ∧ "guarantee"; "certified" ∧
"reranking"; "exact top-k" ∧ "proxy"; "top-k" ∧ "guarantee" ∧ "learned" ∧ "approximate"; "surrogate" ∧ "top-k" ∧
"guarantee"; "PAC" ∧ "reranking"; "conformal" ∧ "retrieval" ∧ "budget"; "distribution-free" ∧ "ranking" ∧ "stopping";
"stopping" ∧ "cross-encoder"; "conformal" ∧ "cascade"; "conformal" ∧ "pruning" ∧ "retrieval"; "top-k" ∧ "cheap" ∧
"expensive" ∧ "guarantee"; "learning to stop" ∧ "ranking"; "multi-fidelity" ∧ "top-k"; "prediction-powered" ∧ "ranking";
"upper confidence" ∧ "top-k" ∧ "retrieval"; "anytime" ∧ "reranking"; "reranking depth" ∧ "guarantee"; "probabilistic
guarantee" ∧ "top-k" ∧ "reranking"; "certificate" ∧ "top-k" ∧ "retrieval"; ti:"early" ∧ ti:"reranking" / "re-ranking";
ti:"conformal" ∧ ti:"ranking" / "retrieval"; ti:"top-k" ∧ ti:"guarantee" / "oracle"; abs:"stop" ∧ abs:"reranking" ∧
abs:"guarantee"; abs:"conformal" ∧ abs:"rerank" / "re-rank"; abs:"conformal" ∧ abs:"top-k" ∧ abs:"retrieval"; abs:"risk
control" ∧ abs:"ranking"; abs:"early stopping" ∧ abs:"ranking" ∧ abs:"guarantee"; abs:"cascade" ∧ abs:"ranking" ∧
abs:"guarantee" ∧ abs:"distribution-free"; abs:"weak oracle"; abs:"strong oracle" ∧ abs:"ranking"; abs:"stopping rule" ∧
abs:"top-k". Result: 195 unique 2025+ ids (raw feed in /tmp/lit3/sweep2_arxiv.json), of which ~170 are off-topic
("conformal" field theory / conformations / conformity, physics cascades, finance rankings, generic conformal
prediction). Semantic Scholar answered HTTP 429 (rate limit, no API key) to every query of this pass at 45 s spacing,
so S2 coverage rests on the 12 queries / 65 records of the first pass above, which already included the five
requested S2 queries; a Crossref keyword sweep was run as a fallback (below). One candidate (PAC-CF) was read in full
text (arXiv PDF v4); the others at abstract level.

| id | title (year) | why it is / is not closer |
|---|---|---|
| arXiv:2604.14345 (v4, 6 Sep 2026) | **PAC-CF: Calibrating Irreversible Frontier Pruning in LLM-Guided Search** — Qian, Chen, Wang (2026) | closest new item, **not closer**. Calibrated *score-gap* filter for top-K pruning in LLM tree search: nonconformity C_path(p) = (max_{a∈C(p)} Ȳ(a) − max_{a∈P(p)} Ȳ(a))₊, the deficit of the best *verifier-valid* child below the best-scored child, maximised over the frontiers of a task (Eq. 7); margin L̂_Q = k_Q-th order statistic (Eq. 8); keep set K_t = {a : s*_t − s_t(a) ≤ L̂_Q + ε} (Eq. 9); Prop. 4 marginal task coverage, Cor. 1 "at least one valid continuation survives". Shared shape: one margin calibrated across tasks on a max-over-states deficit. Differences: no expensive oracle in the loop (verifier labels only at calibration), no stop rule over scored subsets (a pruning set, prefix-like), top-1-relative deficit without per-item σ, existential target (≥ 1 valid survives) rather than the exact top-k or an r-miss frontier, max over the frontiers of *one* trajectory (our A^π shape, not the all-subset A_{k,r}), no policy-uniform or finite-population statement. Entry `qian2026paccf` added to refs.bib |
| arXiv:2607.25018 | Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference — Dou, Lian, Li (2026) | conformal prediction-set size as the deferral rule of an LLM cascade ("accept when the calibrated set collapses to a single answer"), 1−Kα per-tier union bound; classification-style cascade, fixed procedure — the fixed-trajectory family of §B.6, not closer. Entry `dou2026conformalcascade` added |
| arXiv:2602.05448 | BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs — Agrawal, Nguyen, Kiela (ICML 2026 spotlight) | k-wise LLM comparisons aggregated into a tournament graph; "formalize[s] when the current top-m output is certifiably determined" by transitive closure — a *deterministic* certificate from comparisons, no cheap predictor, no calibration, no statistics — not closer (the σ = 0 analogue in the comparison-oracle world). Entry `agrawal2026blitzrank` added |
| arXiv:2601.23128 | Distribution-informed Efficient Conformal Prediction for Full Ranking — Liao et al. (2026) | follow-up of Fermanian et al.: exact Negative-Hypergeometric rank distributions tighten the transductive full-ranking sets; same orthogonal problem (rank prediction sets, no cost, no stopping) — not closer |
| arXiv:2607.16279 | Risk-Aware Facial Retrieval (RA-FR) (2026) | conformal (Hoeffding) calibration of retrieval *set sizes* so the ground truth is included at level α — calibrated prefix — not closer |
| arXiv:2510.17543 | Reliable Inference in Edge-Cloud Model Cascades via Conformal Alignment (2025) | conformal-alignment test decides which inputs the edge model may answer; fixed two-tier procedure, coverage target — not closer |
| arXiv:2510.02480 | Controlling the Risk of Corrupted Contexts for LMs via Early-Exiting (2025) | DFRC-calibrated early exit of attention heads; one fixed exit sequence — not closer |
| arXiv:2511.07396, 2605.18796, 2604.23577, 2605.16401 | C3PO; UCCI; RouteNLP; CADS (2025–2026) | LLM / model cascades with calibrated or cost-constrained deferral thresholds (regret to the most powerful model, isotonic error probability, conformal threshold initialisation, conformal complexity routing); classification/generation, fixed cascade order, no top-k — not closer |
| arXiv:2608.03199 | SieveIVF: Threshold-Aware IVF Execution (2026) | stops IVF probing after W consecutive partitions without a qualifying (above-threshold) candidate; heuristic system stop, no guarantee — not closer |
| arXiv:2511.22273 | UCB for Large-Scale Pure Exploration: Beyond Sub-Gaussianity (2025) | best-arm identification with UCB under heavy tails; single stochastic oracle, no proxy — not closer |
| arXiv:2505.10160 | One-Stage Top-k Learning-to-Defer (2025) | surrogate-loss consistency for deferring to k experts; training objective, no stop rule — not closer |
| arXiv:2606.13221 | Conformal Elo Estimation for LLM Evaluation (2026) | split-conformal intervals on Elo residuals; evaluation, no retrieval stop — not closer |
| arXiv:2603.10950 | Selective prediction for molecular structure retrieval from mass spectra (2026) | risk–coverage selective prediction over candidate rankings; abstention, no early stop — not closer |
| arXiv:2608.01039 | Lower-Bound Representations for Trajectory Similarity Learning (2026) | admissible (exact) lower bounds for trajectory distances used to prune retrieval — the deterministic-bound family (§A of efficiency_neighbours.md), no calibration — not closer |
| arXiv:2603.04816, 2608.25115, 2606.29328, 2605.24634 | Scaling laws for cross-encoder reranking; PACE (pressure-adaptive reranking budget); GeoRAG; calibrated interaction for composed image retrieval (2026) | reranking budgets / context selection / conformal set size as an ambiguity measure; no certified stop — not closer |
| remaining ≈170 ids | conformal field theory, protein conformations, LLM conformity, physics/finance "cascades" and "rankings", generic conformal-prediction methodology, unrelated software-engineering papers | false positives of the `all:`/`abs:` search — discarded |

**Crossref fallback for the rate-limited Semantic Scholar pass.** Because `api.semanticscholar.org/graph/v1/paper/search`
answered 429 throughout, the same nine keyword queries ("conformal reranking", "certified top-k early termination", "weak
strong oracle top-k", "conformal early stopping", "calibrated early exit ranking", "top-k identification proxy oracle",
"conformal prediction retrieval early termination", "risk control reranking", "top-k selection proxy oracle guarantee")
were run against `api.crossref.org/works` (filter from-pub-date 2025-01-01, proceedings/journal articles, 40 rows each;
308 unique DOIs, raw feed /tmp/lit3/sweep2_crossref.json). Crossref's relevance ranking is weak (most hits are
"conformal" algebras, "early termination" in decoding/hardware, "early stopping" in training), and the AcuRank NeurIPS
record confirms the feed reaches the right venues. Abstracts of the three relevant hits were read via the S2 paper-by-DOI
endpoint (which was not rate-limited):

| id | title (year) | why it is / is not closer |
|---|---|---|
| doi 10.1145/3774905.3795598 | Early Recall Estimation and Stopping in Technology-Assisted Reviews via Ranker-Sampler Fusion — Di Nunzio, WWW '26 Companion pp. 333–340 | TAR stopping: Bayesian / model-assisted (GREG + Horvitz–Thompson) recall estimators that fuse calibrated ranker probabilities with a small probability sample, and "practical stopping rules based on conservative (lower-bound) recall estimates". Same venue, same "stop scanning a ranked list" shape, but the target is recall of *human* relevance judgements at 0.90–0.95, the guarantee is per-query and sampling-based (SUPG-like), no expensive frozen teacher, no cheap-bound stop test, no cross-query calibration — not closer. Entry `dinunzio2026tarstopping` added; the TAR stopping-rule family (adjacent, at our venue) was absent from the map and is worth one clause in 02_related.tex |
| doi 10.5220/0014803000004018 | When to Rerank? Adaptive Reranking Decisions for Efficient IR (ARDS) — Gonzalez-Gamella et al., ICEIS 2026 | binary classifier decides per query whether to rerank at all (retrieval-signal features); no guarantee — the RLT / Adaptive Re-Ranking family, not closer |
| doi 10.1109/LSP.2026.3720982 | A KL Certificate for Best-of-N Reranking in Language-Model Inference — Zhang & Yang, IEEE SPL 2026 | proves a KL-divergence upper bound for best-of-N selection; a distribution-shift certificate, no stopping or top-k rule — false positive of "certificate + reranking" |
| other Crossref hits | CAE²-CRC (risk-aware early exit with CRC), CalexNet (early-exit calibration training), "Reasoning Is Not a Race" (NeurIPS 2025, LLM reasoning stop), Esc (early-stopping checker for index tuning, PVLDB 2025), ANSMET (ISCA 2025, ANN early termination in hardware) | fixed-procedure exits, training methods or system heuristics; none has a stop test over scored subsets — not closer |

## Bibliography notes (fields that could not be verified — left out of refs.bib)
- `oettershagen2026topk`, `pony2026colbandit`, `sinhababu2026generate`, `cao2026certbind`: arXiv preprints; no venue on the
  PDFs (CertBind carries an AAAI-2027 copyright line but no acceptance evidence; kept as arXiv).
- `fan2026conformalk`: pages 1–42 and publication date 2026-07-31 from Crossref; body unread — no theorem numbers cited.
- `horchidan2025conann` (new): title/authors/volume/issue/pages from Crossref; abstract from Semantic Scholar; no arXiv id found.
- `lucchese2015quickscorer`, `busolin2021earlyexit` (new): Crossref title/authors/pages/venue; abstracts not retrievable.
- `wang2026conformalthinking`: author list completed from the PDF (nine authors; previously "Xi Wang and others").
- `kalyanakrishnan2012lucb`: no DOI exists for ICML 2012 papers; the icml.cc URL is the persistent identifier.
- `theobald2004topk`: no DOI; VLDB 2004 proceedings URL retained.
- Classic ACM papers (WAND, BMW, MPro, Cambazoglu 2010, QuickScorer, Busolin 2021, TA/JCSS): bodies not retrievable
  (dl.acm.org / ScienceDirect 403); metadata verified via Crossref; descriptions in efficiency_neighbours.md are marked
  as secondary-source based.
- `paper/tools/check_bib.py` (offline): 0 errors, exit 0 (60 entries; 7 never-cited warnings). `--online`: 39 DOI entries
  verified against Crossref, all titles agree; the 17 arXiv-only entries could not be checked by the script (arXiv API
  answers HTTP 406 to `urllib`, a warning) and were verified by hand with `curl` + `id_list` — all 17 titles agree with the
  arXiv records (arXiv spells BEIR's title "Heterogenous"; the bib keeps the published spelling "Heterogeneous").

### Second pass (2026-09-21, later session) — bibliography changes
- Corrected/completed from the primary records: `intrator2024streamlining` — DOI 10.18653/v1/2024.fever-1.22, pp. 186–191
  and the ACL-Anthology author order (Intrator, Cohen, Kelner, Goldenberg, Rivlin, Freedman; arXiv lists Kelner before
  Cohen); `chakraborty2026rag` — DOI 10.1007/978-3-032-21300-6_45, pp. 537–546 (Crossref, LNCS "Advances in Information
  Retrieval").
- Added (all fields from the arXiv API `id_list` record or Crossref; none invented): `bajpai2025beyondgreedy`
  (arXiv:2509.23666, "Accepted as poster in NeurIPS 2025"), `guo2023isrr` (arXiv:2306.07188, preprint, no venue found),
  `zhu2026cas` (arXiv:2608.20771), `ao2026bai` (arXiv:2601.21471), `dziarmaga2026profk` (arXiv:2608.12573),
  `goldberger2026topk` (arXiv:2607.08979, "accepted for ICML 2026"), `genc2026adaptive` (arXiv:2606.25249),
  `wang2025cascadeonenetwork` (arXiv:2503.09492, "Accepted by ICML 2025"), `bajaj2016msmarco` (arXiv:1611.09268),
  `mackenzie2022anytime` (doi 10.1145/3467890, TOIS 40(1):1–32 — Crossref dates it 2022, not 2021 as LITERATURE.md §E
  said), `bardenet2015concentration` (doi 10.3150/14-BEJ605, Bernoulli 21(3) 2015 — listed in LITERATURE.md §E but absent
  from the bib; Crossref gives no page range, so none is recorded), `qian2026paccf` (arXiv:2604.14345),
  `dou2026conformalcascade` (arXiv:2607.25018), `agrawal2026blitzrank` (arXiv:2602.05448, "ICML 2026 spotlight"),
  `dinunzio2026tarstopping` (doi 10.1145/3774905.3795598, WWW '26 Companion pp. 333–340, Crossref). Venue
  strings are taken from the arXiv comment field only when it states acceptance; otherwise the entry is an arXiv preprint.
- Not added (no persistent identifier): "Guaranteed Top-Adaptive-K" (withdrawn ICLR 2026 submission, LITERATURE.md §B.8).
- `paper/tools/check_bib.py` after this pass: 0 errors, exit 0, 75 entries; 22 never-cited warnings (the new adjacent
  works plus bates2021rcps, busolin2021earlyexit, horchidan2025conann, lucchese2015quickscorer, nanivadekar2026idcr,
  posokhov2025tmp, vangysel2018pytrec) — warnings, not errors; the writer node decides which to cite.

## What paper/sections/02_related.tex should say that it currently does not (read, not edited)
1. *Weak/strong-oracle paragraph:* name Oettershagen's **TA-Certify** baseline explicitly — "scores items in weak-estimate
   order and stops as soon as the worst verified top-k item is guaranteed under the weak CIs to exceed every unverified
   item" — since it is literally our μ-descending policy with per-item CIs; and add that his guarantee is *per instance*
   (joint CIs from repeated pulls) whereas ours is *per query distribution* (one calibrated λ), so that the reader does
   not conflate the two notions of "confidence".
2. *Same paragraph:* say that Col-Bandit's "finite population" is the token dimension of one query's MaxSim rows
   (Bernstein–Serfling within a query), which is a different object from our finite-population *split* of a fixed
   benchmark of queries (Thm. 8); the current text mentions Col-Bandit only as "applies this inside late-interaction
   retrieval".
3. *Calibrated early-exit paragraph:* CATs' shared calibration uses M_max, the maximum meta-confidence over the
   *inconsistent* layers of an input — a worst case over the unsafe states of one fixed exit sequence. It deserves one
   clause as the closest existing analogue of a "max over unsafe states" statistic, and the contrast that it ranges
   over a trajectory (our A^π) rather than over all scored subsets (our A_{k,r}); CATs' own Remark 4.2 says the set
   condition is "stricter than necessary".
4. *Conformal retrieval paragraph:* state CertBind's certificate explicitly — Certified iff the observed top-k gap
   exceeds 2ρ (their Eq. 21) — because it is the one-radius, no-stopping special case of our score-coverage statistic
   B_all, and Prop. 3 then says exactly how much A_{k,1} gains over it.
5. *ANN sentence:* add ConANN (PVLDB 2025) — distribution-free control of IVF false-negative rate via conformal risk
   control — as the calibrated member of the "learned early termination in ANN" family; otherwise the sentence implies
   that family has no distribution-free variant.
6. *Efficient neural reranking paragraph:* if the additive-ensemble lineage is meant, cite QuickScorer (SIGIR 2015) and
   Busolin et al. SIGIR 2021 (learned early exit) in addition to Cambazoglu 2010 / Lucchese 2020 / Busolin 2025 — both
   keys are now in refs.bib and will otherwise show up as "never cited" warnings.
7. *Bandit paragraph, precision:* LUCB's stop is (UCB of best outside) − (LCB of worst inside) < ε with an additive
   slack ε; the k-th-LCB-vs-best-UCB form is its ε = 0 case. The current wording ("stops when the k-th best lower
   confidence bound exceeds the best upper bound outside") is correct only for ε = 0 and could say so.

Added in the second pass (2026-09-21, later session), sentences that are **not supported** as written:
8. *Weak/strong-oracle paragraph:* "Oettershagen … stopping when the worst verified top-k item is guaranteed to exceed
   every unverified item" quotes the TA-Certify *baseline* (Sec. 8). His algorithm ACE (Alg. 2, Thm. 5) stops when the
   smallest lower bound among the k items with the largest upper bounds is ≥ the largest upper bound outside; those
   items need not have been strong-queried ("verified"). Attribute the quoted rule to TA-Certify, or describe ACE.
9. *Same paragraph:* "give no statement … about its validity under other scoring orders" is false for LUCB — Thm. 1
   holds for "an arbitrary sampling strategy". Say instead that LUCB's strategy-uniformity comes from anytime-valid
   per-arm confidence bounds (union bound over rounds and pull counts), whereas ours comes from one calibrated
   all-subset statistic with a single prediction per item. Also, Col-Bandit's radii are Bernstein–Serfling
   (finite-population), not "Hoeffding or empirical-Bernstein".
10. *Conformal retrieval paragraph:* Angelopoulos et al. control the **false discovery rate** of the returned set
    (abstract, Sec. 2.2, Eq. 1) — a precision-type error — so "calibrate cutoffs for recall or coverage" misdescribes
    that citation (it is accurate for Intrator et al. and Chakraborty et al., whose abstracts speak of sets
    "guaranteed to include relevant information" / "target coverage … fraction of relevant snippets retained").
11. *Same paragraph:* "Conformal@K … with LTT-style guarantees" — the abstract (the only text available; body 403) says
    "selects monotone budgets" and "distribution-free, finite-sample control"; it does not name LTT. Mark as UNVERIFIED
    or soften to "calibrated budgets".
12. *Efficient neural reranking paragraph:* "learned early exits in additive ensembles and cross-encoders
    \cite{cambazoglu2010early,lucchese2020query,busolin2025earlyexit}" — Busolin et al. 2025 (SEE) is by its own abstract
    "a novel non-learned strategy"; Lucchese 2020's abstract describes query-level early exit without a learned exit
    classifier; Cambazoglu 2010 is metadata-only (UNVERIFIED). The paper that *learns* exits is Busolin et al. SIGIR 2021
    (`busolin2021earlyexit`, in refs.bib, uncited). Either drop "learned" or cite the 2021 paper.
13. *Stopping-rules paragraph:* "learned and declarative early termination in ANN search … stops when a predicted recall
    target is reached" describes DARTH exactly; Li et al. 2020's abstract says their GBDT models "learn and predict when
    to stop searching for a certain query" under fixed "accuracy targets" — a predicted amount of search, not a
    predicted recall. Minor; one clause fixes it.
14. *Calibrated early-exit paragraph:* "with conformal risk control and Learn-then-Test" — Fast yet Safe uses CRC and the
    RCPS/UCB procedure (Prop. 1–2), Conformal Thinking uses the RCPS/UCB procedure of Bates et al. (Eq. 14), only ORCA
    uses LTT fixed-sequence testing (Thm. A.2). Correct for the family, imprecise per member; harmless if kept generic.
15. *Stopping-rules paragraph, first sentence:* "Minimal probing and the threshold algorithm apply it to expensive
    predicates" — the expensive-predicate setting is MPro's; TA (Fagin–Lotem–Naor) aggregates sorted lists with random
    accesses. Both bodies are unretrieved (metadata-only), so the sentence is UNVERIFIED rather than wrong; the
    Theobald §1.1 description of TA supports "stops … only when it is certain that no more top-k results can be found".

Verified neighbours that the section should mention but does not (second pass): PAC-CF (arXiv:2604.14345, calibrated
score-gap margin for top-K pruning — the nearest *new* calibrated-margin rule, prefix-like, no oracle in the loop) in
the conformal-retrieval paragraph next to CertBind; Conformal Cascade (arXiv:2607.25018) in the calibrated-early-exit
family; TMP adapter (`posokhov2025tmp`, one of the ten named works in the plan) in the efficient-reranking paragraph as
the training-time single-threshold ancestor of ranked-list truncation; Beyond Greedy Exits (`bajpai2025beyondgreedy`)
if the early-exit family sentence is meant to be complete; ConANN, TA-Certify, Col-Bandit's within-query finite
population, CATs' M_max, CertBind's Eq. 21, QuickScorer/Busolin 2021 and LUCB's ε as in items 1–7 above.
