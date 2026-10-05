# WWW 2027 — verified submission facts (checked 2026-09-20)

Primary source: official CFP https://www2027.thewebconf.org/research-track-papers/ (wording checked in the raw HTML).
Everything below was verified on 2026-09-20 unless a line says **UNVERIFIED**; re-check the CFP page once more in
N16 (things can change). The ACM authorship section was re-checked after the 2026-05-14 policy revision.

## Venue and dates
- The ACM Web Conference 2027, Dublin, Ireland, 10–14 May 2027. https://www2027.thewebconf.org/
- Research-track full papers: **abstract 18 Oct 2026**, **full paper 25 Oct 2026**, both "end-of-day Anywhere on Earth (AoE)"
  (https://www2027.thewebconf.org/important-dates/). OpenReview's clock: abstract 2026-10-19 13:00 UTC, paper 2026-10-26 13:00 UTC.
  "No extensions, regardless of circumstances." Placeholder abstracts are forbidden.
- Reviews to authors 21 Dec 2026; rebuttal 21–31 Dec 2026 (non-interactive); notification 4 Jan 2027; camera-ready 31 Jan 2027.
- Short papers (separate venue): abstract 9 Nov, paper 16 Nov 2026 — not our target.

## Format
- Main paper **8 pages** + references + optional appendix, **12 pages total**, single PDF (OpenReview limit 50 MB).
- First 8 pages must be self-contained; reviewers are not required to read beyond page 8.
- LaTeX: `\documentclass[sigconf, anonymous, review]{acmart}` (ACM template; acmart v2.21 shipped in paper/). Double column, English.
- No separate supplementary upload: the appendix lives inside the PDF.
- Code/data: "strongly encouraged" to be released after review; ACM "Artifacts Available" badge for accepted papers (archival repo + DOI).

## Tracks (12) — ours: **"Search, Recommendation, and Retrieval-Augmented AI"** (chairs: Hamed Zamani, Xiangnan He)
Others: Brave New Ideas for the Future of the Web; Web Economics and Digital Society; Graph Algorithms and Modeling for the
Web; Responsible Web; Security and Privacy; Semantics and Knowledge; Social Networks and Social Media; Web Infrastructure and
Agentic Systems; User Modeling, Personalization and Agentic Web Users; Web Mining, Multimedia and Multilingual Content Analysis;
Evaluation, Human Computation, and Resources.

## Relevance requirement (desk-reject risk)
Every submission must clearly state **on the first page** how the work is relevant to the Web and to the chosen track.
Papers that merely use a Web artifact without addressing a Web research challenge are out of scope and desk-rejected.
Our statement: web search reranking at query time — result stability vs. compute; guarantees that survive changes of the
document-scoring schedule of a production reranker.

## Anonymity, preprints, dual submission
- Double-blind: no names, affiliations, identifying acknowledgements; own prior work cited in third person; organisations that
  provided data/hosted experiments may be named.
- Anonymised preprints (arXiv) allowed "without citing it".
- No dual submission to peer-reviewed venues with proceedings. Author list and order **frozen at the abstract deadline**.

## Submission system
- OpenReview: https://openreview.net/group?id=ACM.org/TheWebConf/2027/Conference_Long_Papers
- All authors need up-to-date OpenReview profiles (affiliation history, conflicts); a form checkbox confirms it (desk-reject otherwise).
- Max 7 submissions per author across research tracks. Authors may be asked to review; late reviews → desk rejection.
- Form fields: title, abstract, TL;DR, keywords, primary subject area (track), checkboxes (public reviews if accepted;
  authors-as-reviewers), industry-paper flag. Decisions: Accept (Oral) / Accept (Poster). Contact: pcchairs-www2027@acm.org.

## Generative-AI policy  (**updated 2026-05-14 — the 2023 wording no longer applies**)
- WWW: LLMs cannot be authors; authors are solely responsible for everything in the paper. Reviewers must not upload
  submissions to LLMs or any other non-confidential service.
- WWW defers to the **ACM Policy on Authorship**, whose generative-AI section was revised on **2026-05-14**. Under the
  current text:
  - Generative-AI systems **are not authors** (unchanged).
  - **AI used in the research itself** — generating or substantially writing code, running or designing experiments,
    producing figures, analysing data, an agentic assistant doing any of that — **must be described in the methods
    section of the paper**, at the level of detail a reader needs to judge and reproduce the work. It is part of the
    methodology, not a disclosure footnote.
  - **AI writing assistance no longer requires disclosure**: using a generative tool to draft, rephrase, translate,
    shorten or copy-edit prose is treated like spelling and grammar tools and needs no acknowledgement. (This replaces
    the 2023 rule that any generated *text* had to be disclosed in the acknowledgements or an appendix.)
  - Authors remain fully responsible for correctness, originality, citation accuracy and the absence of plagiarism,
    whatever produced the text.
  https://www.acm.org/publications/policies/new-acm-policy-on-authorship
- **What this project does** (D-LLM-DISCLOSURE in graph/decisions.md): this paper's code and experiments were written and
  run by agentic LLM assistants under author supervision, so that is *research* use → an anonymous "Use of AI tools"
  paragraph belongs in the **experimental-setup section of the main text** (with details in the appendix): which
  assistants, what they produced, that every number comes from the released code, and that no LLM was used as a judge,
  annotator, reviewer or author. Nothing has to be said about prose assistance, and nothing identifying may appear
  (double-blind).

## Other
- Ethics section encouraged; ACM policy on research involving human participants applies (not our case — no human study).
- Review criteria: originality, technical merit, impact, execution, presentation, related work, reproducibility, ethics.
- Accepted papers: every paper presents a poster; a subset gets oral slots; one full/main in-person registration per paper.
- Best Paper Award chairs: Alan Smeaton (DCU), Grace Hui Yang (Georgetown). **UNVERIFIED** — the names come from a
  2026-09-20 read of the organisation page and the *selection process is not published anywhere*; do not write any
  claim about how best papers are chosen, and re-check the names in N16 before they appear in any text.
- ORCID required for all authors at publication time.
