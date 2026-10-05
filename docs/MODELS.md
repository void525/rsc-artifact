# MODELS.md — teachers, cheap model, batching modes, first stage, extra domain

Code: `src/rscrank/models.py`, `src/rscrank/candidates.py`, `src/rscrank/data.py`,
`scripts/03_build_candidates.py`, `04_cheap_features.py`, `05_teacher_scores.py`, `06b_distill_cheap.py`,
`09_system_latency.py`. Tests: `tests/test_models.py`. Evidence from a real CPU run: `docs/REAL_MINI_RUN.md`.

## 1. Teachers

### Qwen3-Reranker (main teacher `qwen3-4b`; `qwen3-0.6b` is the ladder-L2 fallback and the CPU test model)
The input construction is the model card's *transformers* snippet, verbatim
(https://huggingface.co/Qwen/Qwen3-Reranker-4B, fetched 2026-09-20):

```
text   = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
         instruction = "Given a web search query, retrieve relevant passages that answer the query"
ids    = tokenizer(texts, padding=False, truncation='longest_first', return_attention_mask=False,
                   max_length=max_length - len(prefix_tokens) - len(suffix_tokens))["input_ids"]
seq    = prefix_tokens + ids + suffix_tokens          # tokenizer(padding_side='left')
prefix = <|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the
         Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n
suffix = <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```
`doc = "{title} {text}"` (title omitted when empty). The whole body string is tokenized in one call (the old
code tokenized head and document separately, so tokens at the boundary could differ from the card).
`max_length = 512` (`teachers.<t>.max_length`); long documents are cut at the end of the body, the suffix is never cut.

**Score.** `s = z_yes − z_no` at the last position (the raw logit difference the card's Sentence-Transformers
path returns; the card's `P(yes) = softmax([z_no, z_yes])[1] = sigmoid(s)`, a strictly monotone transform).
We run only the decoder body (`model.model(..., use_cache=False)`, final RMSNorm included) and apply the two
lm_head rows to the last hidden state in float32: `z = h.float() @ W[[yes, no]].float().T`. This is the same
number as `logits[:, -1, yes] − logits[:, -1, no]` up to rounding, and more precise: with bf16 weights the full
vocabulary logits are rounded to bf16 (spacing 0.0625 at |z| ≈ 10) before the subtraction, whereas here the
bf16 hidden state and bf16 weights are multiplied exactly in fp32. It also never allocates the
`[B, L, 151669]` logits tensor (2.5 GB at B=16, L=512 in bf16).

**Positions.** Left padding (the last position is every row's last real token). Real tokens get position ids
`0..n−1` (`cumsum(mask) − 1`, as HF `generate` does for left-padded batches). RoPE is relative, so this is the
card's forward pass in exact arithmetic (the card's plain `model(**inputs)` call offsets positions by the pad
count); it removes a dependence of RoPE rounding on the batch's padding.

### BGE reranker (`bge-m3` = BAAI/bge-reranker-v2-m3)
FlagEmbedding's `FlagReranker.compute_score` (the card's primary usage; code read 2026-09-20): query tokens
truncated to `query_max_length = 3/4 · max_length` (384), document tokens appended with
`truncation='only_second'` to `max_length = 512`: `<s> q </s></s> d </s>`; score = raw logit (`logits.view(-1)`).
For queries shorter than the budget this equals the card's plain-transformers snippet
(`tokenizer(pairs, padding=True, truncation=True, max_length=512)`), which the test checks token-for-token.
The 2-layer classification head runs in float32 on the encoder's `<s>` state (fp16 on the GPU otherwise
rounds the logit to fp16). Right padding (XLM-R position ids are derived from non-pad tokens).

### Validation (CPU, float32; `python -m pytest -q tests/test_models.py`)
| check | result on this sandbox (2 CPUs) |
|---|---|
| Qwen3-Reranker-0.6B vs the card's reference code (6 pairs, one truncated at 512 tokens) | identical token ids; max abs logit-difference error 2.7e-5 (< 1e-3); sigmoid(s) = card P(yes) within 1e-4 |
| Qwen fast vs canonical | max abs diff 2.7e-5 (< 1e-4) |
| Qwen canonical, same pairs inside a shuffled mix with other pairs | **bit-identical** |
| bge-reranker-v2-m3 vs the card's transformers snippet | identical token ids; max abs diff 3.8e-6; card example −8.1838 / 5.2650 (card, fp16: −8.1875 / 5.2617) |
| bge canonical under a different composition | **bit-identical**; fast mode: 1.4e-6 differences (not bit-stable) |
| MiniLM features vs `AutoModelForSequenceClassification` | logits 4.8e-7, pooled [CLS] 6e-8, token counts equal |
| relevant document outranks an obviously irrelevant one (3 queries) | all three models |
The network tests are skipped when `RSC_SKIP_NET=1` or the hub (`HF_ENDPOINT`) does not answer a HEAD request.
Run time: Qwen test ≈ 51 s, BGE ≈ 40 s, MiniLM ≈ 5 s on 2 CPU cores (downloads ≈ 3.5 GB once).

## 2. Batching modes (`teachers.<t>.mode`)
| mode | forward-pass shapes | a pair's score depends on batch-mates? | use |
|---|---|---|---|
| `fast` | pairs sorted by token length (descending, ties by input index); greedy batches with ≤ `batch_size` rows and ≤ `max_tokens_per_batch` padded tokens (rows × padded length; padded length rounded up to a multiple of 8); order restored | yes, at the rounding level (different padded shapes select different GEMM/attention kernels and reduction splits) | throughput |
| `canonical` | every pass is exactly `[batch_size, L]`, L = smallest multiple of 64 **strictly greater** than the pair's own length; a batch only contains pairs with the same L; missing rows are dummy copies of the batch's first row (outputs discarded) | no: the pair's tensor slice, pad count and all kernel shapes are functions of the pair alone | reproducible caches |
"Strictly greater" guarantees ≥ 1 pad token per row, so Hugging Face never switches to the mask-free SDPA
(`is_causal`) kernel for batches that happen to have no padding. `score(query, docs)` is `score_pairs` on that
query's pairs in the same mode; `score_pairs(pairs)` accepts pairs of many queries (used for caching, which
scores 32 queries = 3,200 pairs per call). GPU caveat: canonical mode removes shape-dependence; bit-stability
additionally requires deterministic kernels for fixed shapes (true for cuBLAS GEMMs and SDPA forward on one
stream). It is *measured*, not assumed: the pilot and the audit report `max_abs_diff`.

**Pilot (N05)** `python scripts/05_teacher_scores.py --teacher qwen3-4b --pilot` sweeps
`mode ∈ {fast, canonical} × batch_size ∈ {8,16,32}` (fast uses `max_tokens_per_batch` from the config) on a
stratified sample of `budget.pilot_pairs` cal pairs (word-length tertiles), records pairs/s (per stratum and
total), peak VRAM, padding efficiency, OOMs (caught per setting), and batch invariance: ~256 pairs of 10 queries
scored alone vs. inside a shuffled mix with 256 other pairs (`max_abs_diff`, bit-identical pairs, queries whose
top-k order changes). Recommendation rule: **canonical** if its fastest bit-stable setting (max |Δ| = 0) reaches
≥ 0.67 × the best fast throughput; otherwise the **fastest fast setting that fits in memory** (no OOM and peak
VRAM ≤ 0.875 × GPU memory, i.e. 28 GB on the 32 GB card; `--vram-limit-gb`). Written to
`runs/pilot/throughput_<t>.json` with `v_measured_pairs_per_s` of the recommended setting,
`projected_hours_core`, `ok = v ≥ budget.v_min_pairs_per_s`, `fast_bit_stable`, `canonical_bit_stable`.
`--apply-recommendation` prints the keys to set (it never edits configs):
```
teachers.qwen3-4b.mode: canonical
teachers.qwen3-4b.batch_size: 16
```
**Audit (N07)** `--audit` re-scores ⌈`teachers.determinism_pairs`/N⌉ cached cal queries (all candidates)
(i) shuffled in one cross-query call and (ii) per query in reversed order plus 4 single-pair calls, compares
with the cache in its float32 storage precision and writes `runs/teacher/<t>/determinism.json`
(`mode, bit_stable, max_abs_diff, mean_abs_diff, queries_checked, topk_order_changed_queries, ok`) with
`ok = queries_checked > 0 and topk_order_changed_queries ≤ max(1, 0.02 · queries_checked)`.
Top-k order uses the paper's fixed total order (`core.teacher_order`, docid ties).

**Cache format** (unchanged): `data/teacher/<t>/<domain>/<split>.npz` with `score [Q,N] f32`, `ntok [Q,N] i32`,
`valid`; meta now also stores `settings` (mode, batch size, dtype, attn, versions, GPU). Checkpoint after every
chunk (`<split>.partial.npz`, written atomically); a partial from other candidates is set aside, a resume with
different settings is logged and recorded. `--limit-queries N` (debug) stores are flagged `complete: false`,
make the manifest `ok: false`, and are recomputed by a full run.

## 3. Cheap model and the distilled lever
`MiniLMCheap` (cross-encoder/ms-marco-MiniLM-L6-v2): pooled [CLS] (384-d, the classifier input) + raw logit +
token count; `features_pairs(pairs)` batches pairs of many queries sorted by length (04 uses 64 queries per call).

`cheap.model: minilm-distilled` (optional, pre-registered lever) — layout, one model per teacher because the
regression target is that teacher's score:
```
models/cheap_distilled/<teacher>/            HF format (config.json, model.safetensors, tokenizer) + meta.json
data/features_distilled/<teacher>/<domain>/<split>.npz     features from that model (same store format)
runs/features_distilled/<teacher>/manifest.json
```
Helpers: `rscrank.data.features_dir_for(P(cfg,"features"), cfg["cheap"]["model"], teacher)` (returns
`data/features` for `minilm`, `data/features_distilled/<teacher>` for `minilm-distilled`) and
`rscrank.data.distilled_model_dir(P(cfg,"models"), teacher)`. `get_cheap("minilm-distilled", {"path": ...})`.
`scripts/06b_distill_cheap.py --teacher <t>`: Huber regression (δ = 1) of the MiniLM logit on the cached TRAIN
teacher scores of the main (+extra) domains, optional `--query-equal` weighting, AdamW (lr 2e-5, bs 64,
max_length 256, 2 epochs from `cheap.distill`), linear warm-up/decay, early stopping on DEV Huber loss (best dev
epoch saved; `meta.json` written last, so a missing meta.json means "unfinished"). Only `train`/`dev` are read
(asserted in the loader). Then `python scripts/04_cheap_features.py` with `cheap.model: minilm-distilled`.
Scripts that read features (06/07/08) must use `features_dir_for(...)` for the distilled variant (see the
integration notes in the hand-off); 09 already does.

## 4. First stage (`candidates.backend`)
**bm25s (default).** Lucene BM25, k1 = 0.9, b = 0.4 (`candidates.k1/b`), lowercase + regex tokens
`(?u)\b\w\w+\b`, bm25s' English stop list (the 33-word Lucene default set), PyStemmer `porter`
(`candidates.stemmer`; the documented protocol says Porter; `english` = Porter2 also works), index text =
`title + " " + text` (like the pyserini *flat* index). Determinism: bm25s' own `retrieve` resolves ties by
`argpartition` order; we take the full score vector (`get_scores_from_ids`) and rank by descending score, ties
by ascending corpus position (corpus.jsonl order). **Exactly N candidates** whenever the corpus has ≥ N
documents: queries that match fewer than N documents (or whose tokens are all stopwords/unknown) are filled
with zero-score documents in corpus order (counted in the manifest as `queries_with_zero_score_fill`,
`queries_empty_after_tokenization`). Retrieval uses all CPU threads (`ThreadPoolExecutor`, no multiprocessing:
Windows-safe; `candidates.n_threads`). The corpus is streamed and tokenized in chunks of
`candidates.bm25s_chunk_docs` (200k) with one shared vocabulary, so raw texts are never all in memory.
The manifest records backend requested/effective, parameters, tokenize/index/retrieve seconds, vocabulary size,
average document length. On SciFact (300 test queries): nDCG@10 = 0.6713 (porter) / 0.6762 (english) vs the
pyserini flat 2CR reference 0.6789; top-N scores identical to stock bm25s (max diff 0).

**pyserini (optional).** `beir-v1.0.0-<domain>.flat`, BM25(0.9, 0.4). Any exception during the import or the
retrieval (no Java 21, index download failure, JVM crash surfacing as SystemExit) is logged and the domain falls
back to bm25s; the manifest records `backend_requested: pyserini`, `backend: bm25s`, `fallback_reason`.

**Memory and time (bm25s).** Measured on this sandbox (2 slow cores) with synthetic corpora of
HotpotQA-like length (46 words/document), streamed from an iterator; peak RSS of the whole process:

| documents | vocabulary | tokenize | index | retrieve 200 queries | peak RSS (of which raw texts) |
|---|---|---|---|---|---|
| 200,000 | 179,655 | 3.2 s | 4.2 s | 0.09 s | 0.70 GB (0.25 GB) |
| 600,000 | 350,669 | 12.8 s | 13.6 s | 0.17 s | 1.83 GB (0.56 GB) |

The index structures cost about **2.1 GB per million documents** at this length (token-id lists ~8 B/token; the
score/doc/vocab flat arrays plus the CSC copy ~30 B per (document, term) pair) and tokenize+index about
**25-45 s per million documents** on two slow cores. Extrapolated, *not* measured: **HotpotQA** (5.23 M
documents) ~11-13 GB peak - plan for 16 GB free RAM; **MS MARCO** (8.84 M passages, ~56 words) ~20-24 GB. A real
corpus has a far larger vocabulary than these synthetic ones (millions of surface forms), which adds roughly
0.5-1 GB of Python dictionaries per 3 M distinct words, so the planning numbers in
`graph/nodes/N04_candidates.md` (10-30 min, 10-20 GB for HotpotQA) remain the safe bound. If RAM is short:
close other programs, index title+text truncated to 256 words (and record it), lower
`candidates.bm25s_chunk_docs` (this only shrinks the streamed text buffer: 0.25 GB per 200k documents), or use
pyserini (its index stays on disk). Retrieval itself is cheap: one dense score vector per query
(4 bytes x corpus size = 21 MB for HotpotQA) per worker thread.

## 5. Extra web-query domain (`domains.extra`, e.g. `[msmarco]`; off by default)
`rscrank.data.domain_list(cfg)` = main + cross + extra. BEIR msmarco: UKP zip (1.08 GB, 8,841,823 passages) or
HF `BeIR/msmarco` (parquet) + `BeIR/msmarco-qrels` (train/dev/test tsv). The cal/test pool is
`domains.test_pool_split.msmarco = dev` (6,980 queries, BEIR's evaluation split); `make_splits` takes the pool
split from the config and records it (`pool_split` in the splits manifest). Training questions: queries of
BEIR msmarco `qrels/train.tsv` (502,939), de-duplicated against cal+test exactly like the other domains.
Sizes from `splits.msmarco`. 01/02/03/04/05 handle it; 06/07/08/10–12 read `domains.main` and load qrels with
`splits=("test",)` — they must use `domain_list` / `rscrank.data.load_pool_qrels(cfg, domain, raw)` before
msmarco is enabled (hand-off notes).
The Hugging Face fallback of `download_beir` and the NQ-Open training questions now use `huggingface_hub` +
`pyarrow` (no `datasets` dependency); `datasets` remains a fallback for NQ-Open only.

## 6. Hardware notes (Windows 11 + RTX 5090, or WSL)
* Torch with CUDA 12.8+ wheels (sm_120); `attn: sdpa` (flash-attn has no Windows wheels). transformers ≥ 4.51
  (Qwen3); both the ≥ 4.56 `dtype=` and the older `torch_dtype=` keyword are handled.
* bf16 on the GPU for Qwen (`dtype: bfloat16`), fp16 for BGE; the heads of both run in fp32.
* On CPU use `dtype: float32` (bf16/fp16 on CPU is slow and triggers a warning).
* `HF_ENDPOINT=https://hf-mirror.com` if the hub is unreachable (tests and downloads honour it).
* Files are opened with UTF-8 and checkpoints are replaced atomically after the handle is closed (Windows
  refuses to replace open files).
