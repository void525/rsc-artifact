# WRITING_GUIDE.md — how a strong WWW paper reads (for the writer/reviewer subagents)

**Spine of the paper (three sentences the reader must remember).**
1. We do not calibrate all teacher scores; we calibrate the condition that rules out a wrong stop.
2. The worst case over all unsafe scored subsets collapses to a closed-form query statistic — exactly, not approximately.
3. One calibrated threshold protects every scoring order; we prove what that costs and measure whether it pays.

**Page 1.** Title; abstract (numbers via macros); introduction with the Web problem in the first paragraph (query-time reranking cost
vs. result stability in web search; schedules change in production), a labelled paragraph "Relevance to the Web", a
worked mini-example (k=2, s=(9,8,7,1), μ=(8,6,6,−100): A=1, B_top=2, B_all=101), contributions C1–C3, and the claims boundary
sentence (what is not claimed).

**Related work.** Six families in ≤ 0.9 page: (i) deterministic rank-safe pruning (WAND/BMW/TA); (ii) **stopping rules with
predicted bounds** — the ancestors of our test: MPro and TA with exact bounds, probabilistic TA, Fast-Forward indexes
(Leonhardt et al., WWW 2022 — same venue, same inequality with an *estimated* maximum dense score), Everest (SIGMOD 2021),
threshold prediction for dynamic pruning (Petri 2019, Mallia 2020), SUPG, learned/declarative early termination in ANN search
(Li 2020, DARTH 2025). Say plainly what is inherited (the k-th-vs-bound test) and what is new (cross-query distribution-free
calibration of a *rank-sufficient* statistic, all-subset tightness, policy-uniform validity, the finite-population regime,
the cost anatomy); (iii) bandit top-k and the weak/strong-oracle line (LUCB; Oettershagen 2026; Col-Bandit) — same test,
different uncertainty model, no cross-query calibration; (iv) calibrated early exit / risk-controlled adaptive compute
(CATs, Fast-yet-Safe, Conformal Thinking, ORCA) — fixed policy; (v) conformal/risk-controlled retrieval set sizing
(Two-stage RC, Conformal@K, CertBind, COPA'23) — coverage or budgets, not all-subset tightness; (vi) efficient LLM reranking
(RLT, Generate-to-Accelerate, AcuRank, early-exit CEs).

**Method.** Definitions → stop rule → Theorem 1 with a 6-line proof sketch → Theorem 2 → Prop. 3 (why it saves) → Prop. 4 (what it
costs) → the algorithm box → implementation note (normalised comparison) → Prop. 10 (cost anatomy: |V*| + band + batching) with
the definition of PoU(q) = A_{k,1} − A^π_{k,1}. Theorem statements verbatim from docs/THEORY.md, **with its numbering**: the
paper pins each statement to the number it has in THEORY.md via `\setthm{n}` in paper/main.tex, so Theorem 1/2, Prop. 3/4/5,
Theorems 6/7/8 and Prop. 9/10 mean the same objects in both files. Facts A/B and Lemma C use their own lettered counter.

**Guarantees.** Theorems 6–8 stated with their exact assumptions; the "what is not guaranteed" list; the calibration indices
(951/966/968 etc.) shown as a small table via macros.

**Results.** Main table (risk with CP interval, calls, Δcalls LCB vs RSC, nDCG, early fraction) for 4 configs × main baselines;
Fig. risk–cost frontier with α curve and the fixed-prefix curve; policy table (RSC vs trajectory-specific, with and without
recalibration); k/r/batch/N grids compact; cross-domain as stress test; sigma ablation; latency table (controlled + tuned, p50/p95).
Every sentence with a number cites a macro; every comparative claim cites a paired interval.

**Analysis.** Mechanism buckets (gap_k, H_k, length); conditional error given early stop; break-even (offline cost / per-query
savings); the N/k cost example vs. measured cost of uniformity; failure modes.

**Compliance paragraphs.** The Web-relevance paragraph is on page 1 (desk-reject otherwise). The ACM authorship policy
(updated 2026-05-14) requires AI used *in the research itself* — code, experiments, figures — to be described **in the
methods section**: that is the anonymous "Use of AI tools" paragraph in 06_setup.tex, with the detail in 91_appendix_extra.tex.
Writing assistance no longer needs its own disclosure. Keep both paragraphs free of vendor names and of anything identifying.

**Limitations.** Pointwise teachers only; guarantees tied to the frozen system; finite-population statement ≠ future traffic;
no relevance gain; compute budget; single GPU; what would change with a listwise teacher.

**Reviewer traps to avoid.** Overclaiming novelty (cite LUCB/Oettershagen/WAND early); hiding the adaptive-prefix baseline;
treating replay time as latency; averaging early-stop errors away; omitting the strict α=0.01 result; unexplained +∞ thresholds;
page-9 theorems; missing Web paragraph.
