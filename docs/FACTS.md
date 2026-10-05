# FACTS.md — verified facts about models, data, tools, hardware (checked 2026-09-20)

## Teachers and cheap model
**Qwen/Qwen3-Reranker-4B** (https://huggingface.co/Qwen/Qwen3-Reranker-4B) — Apache-2.0; base Qwen3-4B-Base; 4.02B params bf16;
36 layers, hidden 2560; context 32k (config 40960). Official template (verbatim):
- prefix: `<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n`
- body: `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`; default instruction "Given a web search query, retrieve relevant passages that answer the query"
- suffix: `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
- Card score = softmax over the two logits [no, yes] at the last position → P(yes) = sigmoid(logit_yes − logit_no).
  The Sentence-Transformers integration (≥ 5.4) returns the **raw logit difference** by default. We use s = logit_yes − logit_no
  (a strictly monotone transform of P(yes); identical ranking; token ids yes=9693, no=2152). Padding side must be **left**
  (score read at logits[:, −1, :]); tokenizer_config does not set it. Card recommends flash_attention_2 + fp16; weights are bf16;
  transformers ≥ 4.51 required. Variants: 0.6B (28 layers) and 8B. No official "Qwen3.5-Reranker" exists as of 2026-09-20
  (only Qwen3-VL-Reranker-2B/8B multimodal, Jan 2026).

**BAAI/bge-reranker-v2-m3** (https://huggingface.co/BAAI/bge-reranker-v2-m3) — Apache-2.0; XLM-R-large-style, 568M params,
24 layers, hidden 1024; score = raw logit (`logits.view(-1)`), optional sigmoid; max_position 8194; card snippet uses max_length 512.

**cross-encoder/ms-marco-MiniLM-L6-v2** — Apache-2.0; BERT 6 layers, hidden 384, 22.7M params; single logit, Identity activation;
max 512 tokens. We use the pooled [CLS] (pooler_output, 384-d) + the logit as cheap features.

## First-stage
Pyserini prebuilt BEIR indexes: `beir-v1.0.0-{nq,hotpotqa,fiqa,scifact}.flat` (title+text concatenated) and `.multifield`;
hosted at https://huggingface.co/datasets/castorini/prebuilt-indexes-beir; list https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md.
Reference BM25 nDCG@10 (pyserini 2CR, flat): nq 0.3055, hotpotqa 0.6330, fiqa 0.2361, scifact 0.6789 (multifield: 0.3285, 0.6027, 0.2361, 0.6647).
Pyserini 2.4.0 (2026-08-27) needs Java 21. Fallback `bm25s` (+PyStemmer): k1 0.9, b 0.4, Porter, English stopwords.

## Datasets (BEIR; sizes verified from qrels/corpora)
| domain | test queries (qrels rows) | corpus | other splits | licence (HF card) |
|---|---|---|---|---|
| nq | 3,452 (4,201) | 2,681,468 | none (BEIR nq has only test) | cc-by-sa-4.0 |
| hotpotqa | 7,405 (14,810) | 5,233,329 | dev 5,447; train 85,000 | cc-by-sa-4.0 |
| fiqa | 648 (1,706) | 57,638 | dev 500; train 5,500 | cc-by-sa-4.0 |
| scifact | 300 (339) | 5,183 | train 809 | cc-by-sa-4.0 (SciFact itself: claims CC BY 4.0, abstracts ODC-By 1.0) |
BEIR zips: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip (live 2026-09-20; nq 498 MB, hotpotqa 654 MB,
fiqa 18 MB, scifact 2.8 MB). HF alternative: `BeIR/{name}` (corpus/queries) + `BeIR/{name}-qrels`. Loader: `beir` 2.2.0 or our own parser.
Evaluator: BEIR uses pytrec_eval (trec_eval ndcg_cut = **linear gain**, log2 discount); our pure-Python nDCG matches it (test).
Training questions: NQ-Open `google-research-datasets/nq_open` (train 87,925; CC BY-SA 3.0) — questions only; fallback DPR
`nq-train.qa.csv` (79,168 lines; DPR repo CC-BY-NC 4.0). HotpotQA train from BEIR train qrels (85k queries). NQ original:
307,373 train / 7,830 dev (CC BY-SA 3.0); HotpotQA original train 90,447 (CC BY-SA 4.0).

## Hardware / software
RTX 5090: 32 GB GDDR7 (NVIDIA). Blackwell consumer (sm_120) needs **PyTorch ≥ 2.7 with cu128** (or cu129/cu130) wheels — cu126
wheels do not include sm_120 (`pip install torch --index-url https://download.pytorch.org/whl/cu128`). flash-attn 2.8.x wheels
include sm_120 kernels (Linux only, keyed by torch version); README does not list Blackwell — treat as optional; default `sdpa`.
Latest torch 2.14.0 (2026-09-02). Windows: no flash-attn wheels; use sdpa.

## Budget arithmetic (planning, not measurement)
Core queries 8000 + 2000 + 3452 + 7405 + 648 + 300 = 21,805; × N=100 → 2,180,500 pairs for the main teacher.
220 GPU-hours budget → v_min = 2.7532 pairs/s. A 4B bf16 model at 512 tokens does ≈ 4.1 TFLOP/pair; at an effective 100 TFLOPS
that is ≈ 24 pairs/s → ≈ 25 h. The pilot (N05) replaces these numbers with measurements.
