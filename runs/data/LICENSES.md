# Data sources and licences (recorded at node N03, 2026-09-21)

All corpora, queries and qrels come from the BEIR benchmark distribution (UKP mirror zips; sha256 in
`runs/data/download_manifest.json`). Sizes match the published BEIR numbers exactly
(NQ: 2,681,468 docs / 3,452 test queries; HotpotQA: 5,233,329 / 7,405; FiQA-2018: 57,638 / 648; SciFact: 5,183 / 300).

| domain | role | source | licence (as published by the data owners / BEIR cards) |
|---|---|---|---|
| nq | main (cal/test pool = BEIR test split) | BEIR `nq` (derived from Natural Questions) | Natural Questions: CC BY-SA 3.0; BEIR card: cc-by-sa-4.0 |
| nq training questions | train/dev queries (question text only) | HF `google-research-datasets/nq_open` train split (87,925 questions) | CC BY-SA 3.0 (NQ-Open) |
| hotpotqa | main (cal/test pool = BEIR test split) | BEIR `hotpotqa` | HotpotQA questions and processed Wikipedia corpus: CC BY-SA 4.0 |
| hotpotqa training questions | train/dev queries | BEIR `hotpotqa` queries appearing in `qrels/train.tsv` (85,000) | CC BY-SA 4.0 |
| fiqa | cross-domain stress test only | BEIR `fiqa` (FiQA-2018) | BEIR card: cc-by-sa-4.0 |
| scifact | cross-domain stress test only | BEIR `scifact` | claims/labels CC BY 4.0; abstract corpus ODC-By 1.0 (component licences retained) |
| msmarco (optional web-query domain, N20) | additional domain (cal/test pool = BEIR dev split, 6,980 queries) | BEIR `msmarco` (MS MARCO passage ranking, 8,841,823 passages) | MS MARCO terms: non-commercial research use only (Microsoft); BEIR card: cc-by-sa-4.0 for the packaging |
| msmarco training questions | train/dev queries | BEIR `msmarco` queries in `qrels/train.tsv` (502,939; de-duplicated against cal+test) | same as above |

Notes.
- No new human annotation, no user study, no LLM judge; relevance metrics use the public qrels only.
- Only question *text* is used from the training sources; answers are never read.
- The code licence of any wrapper repository does not replace the original data licences above.
