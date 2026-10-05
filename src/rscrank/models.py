"""Frozen models (Sec. 10.1-10.3) behind two tiny interfaces.

TeacherScorer.score(query, docs)            -> list[float]      (pointwise, deterministic)
TeacherScorer.score_pairs(pairs)            -> list[float]      pairs = [(query, (title, text))], many queries
CheapModel.features(query, docs)            -> (emb [n,d], logit [n], n_tokens [n])
CheapModel.features_pairs(pairs)            -> same, for pairs spanning many queries

Real backends (torch/transformers are imported lazily inside the classes):

* Qwen3RerankerScorer -- Qwen/Qwen3-Reranker-{0.6B,4B} (Apache-2.0).  Input construction is the
  model card's transformers snippet verbatim (checked 2026-09-20):
      text   = "<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {doc}"
      ids    = tokenizer(texts, padding=False, truncation='longest_first', return_attention_mask=False,
                         max_length=max_length - len(prefix_tokens) - len(suffix_tokens))["input_ids"]
      seq    = prefix_tokens + ids + suffix_tokens            (left padding, tokenizer(padding_side='left'))
  The whole body string is tokenized in one call, so boundary tokens are exactly the card's.
  Score s = z_yes - z_no at the last position, i.e. the raw logit difference that the card's
  Sentence-Transformers path returns (the card's P(yes) = sigmoid(s)).  We run only the decoder
  body (``model.model(..., use_cache=False)``) and apply the two lm_head rows ("yes", "no") to the
  last hidden state in float32: s = (h.float() @ W[[yes, no]].float().T) @ [1, -1].  This is the same
  number as ``logits[:, -1, yes] - logits[:, -1, no]`` up to rounding, and more precise than the
  bf16 full-vocabulary logits (which are rounded to 8 mantissa bits before the subtraction); it
  also never materialises a [B, L, 151669] logits tensor.  Real tokens get position ids 0..n-1
  (as in HF ``generate`` for left-padded batches); RoPE is relative, so this equals the card's
  forward pass in exact arithmetic and removes the dependence of RoPE rounding on the pad count.
* BGERerankerScorer   -- BAAI/bge-reranker-v2-m3 (Apache-2.0), raw logit.  Input construction is
  FlagEmbedding's FlagReranker.compute_score (the model card's primary usage): query tokens
  truncated to query_max_length = 3/4 * max_length, then ``<s> q </s></s> d </s>`` with
  truncation='only_second' at max_length = 512.  For queries shorter than the budget this equals
  the card's plain-transformers snippet (tokenizer(pairs, truncation=True, max_length=512)).  The
  2-layer classification head runs in float32 on the encoder's <s> state (same value up to rounding).
* MiniLMCheap         -- cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0): pooled [CLS]
  representation (384-d, the classifier input) + raw logit; max_length 512, 'longest_first'
  truncation as in Sentence-Transformers' CrossEncoder.  Also loads a distilled copy from a path.

Batching (teachers): ``mode`` = "fast" or "canonical" (configs/main.yaml teachers.<t>.mode).
  fast      : sort pairs by token length (descending, ties by input index), greedy batches with
              <= batch_size rows and <= max_tokens_per_batch padded tokens (rows x padded length,
              padded length rounded up to a multiple of 8); original order restored.  Maximum
              throughput; a score may depend on its batch-mates at the rounding level (different
              padded shapes select different kernels).  Measured by the pilot (N05).
  canonical : every forward pass has the fixed shape [batch_size, L] with L = the smallest multiple
              of 64 strictly greater than the pair's own length (so every row carries >= 1 pad
              token and HF never switches to the mask-free SDPA path for some batches only); a
              batch only holds pairs of the same L; missing rows are dummy copies of the batch's
              first row whose outputs are discarded.  A pair's input tensor slice, pad count and
              kernel shapes are therefore functions of the pair alone, so its score cannot depend
              on its batch-mates (verified bit-exact on CPU; audited on the GPU by
              scripts/05_teacher_scores.py --pilot / --audit).
Synthetic backends (FakeTeacher / FakeCheap) generate deterministic hash-based scores with a
shared latent so that the smoke pipeline exercises every stage.
Facts about the model cards were verified on 2026-09-20 (docs/FACTS.md, docs/MODELS.md).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "TeacherScorer",
    "CheapModel",
    "Qwen3RerankerScorer",
    "BGERerankerScorer",
    "MiniLMCheap",
    "FakeTeacher",
    "FakeCheap",
    "get_teacher",
    "get_cheap",
    "plan_fast",
    "plan_canonical",
    "canonical_length",
    "format_qwen_instruction",
    "QWEN_PREFIX",
    "QWEN_SUFFIX",
    "QWEN_INSTRUCTION",
    "QWEN_TEMPLATE",
    "TEACHER_MODES",
]

QWEN_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_TEMPLATE = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
TEACHER_MODES = ("fast", "canonical")
CANONICAL_MULTIPLE = 64  # canonical padded lengths are multiples of this
FAST_PAD_MULTIPLE = 8  # fast-mode padded lengths are rounded up to this (tensor-core friendly)

Doc = Tuple[str, str]
Pair = Tuple[str, Doc]


def _doc_text(title: str, text: str) -> str:
    title = (title or "").strip()
    return f"{title} {text}".strip() if title else (text or "")


def format_qwen_instruction(instruction: Optional[str], query: str, doc: str) -> str:
    """The model card's ``format_instruction`` (verbatim semantics)."""
    if instruction is None:
        instruction = QWEN_INSTRUCTION
    return QWEN_TEMPLATE.format(instruction=instruction, query=query, doc=doc)


# --------------------------------------------------------------------------- #
# Batch planning (pure functions; unit-tested without any model)
# --------------------------------------------------------------------------- #
def _round_up(n: int, m: int) -> int:
    return int(-(-int(n) // m) * m) if m > 1 else int(n)


def canonical_length(n: int, multiple: int = CANONICAL_MULTIPLE) -> int:
    """Smallest multiple of ``multiple`` strictly greater than ``n`` (>= 1 pad token per row)."""
    return (int(n) // multiple + 1) * multiple


def plan_fast(lengths: Sequence[int], batch_size: int, max_tokens_per_batch: Optional[int] = None, pad_multiple: int = FAST_PAD_MULTIPLE) -> List[Tuple[List[int], int]]:
    """Length-sorted greedy batches -> [(indices, padded_length)]; every index appears exactly once.

    Pairs are visited by (-length, index); a batch grows while it has <= batch_size rows and
    rows * padded_length(first row) <= max_tokens_per_batch.  A single over-long pair forms its own batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))
    plans: List[Tuple[List[int], int]] = []
    cur: List[int] = []
    L = 0
    for i in order:
        if cur and len(cur) + 1 <= batch_size and (max_tokens_per_batch is None or (len(cur) + 1) * L <= max_tokens_per_batch):
            cur.append(i)
            continue
        if cur:
            plans.append((cur, L))
        cur, L = [i], _round_up(max(1, int(lengths[i])), pad_multiple)
    if cur:
        plans.append((cur, L))
    return plans


def plan_canonical(lengths: Sequence[int], batch_size: int, multiple: int = CANONICAL_MULTIPLE) -> List[Tuple[List[int], int]]:
    """Fixed-shape batches -> [(indices, L)] with len(indices) <= batch_size (caller adds dummy rows).

    L = canonical_length(own length) for every member, so the padded shape of a pair never depends
    on which other pairs are scored with it.  Buckets are processed longest first; inside a bucket
    pairs keep their input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    buckets: Dict[int, List[int]] = {}
    for i, n in enumerate(lengths):
        buckets.setdefault(canonical_length(n, multiple), []).append(i)
    plans: List[Tuple[List[int], int]] = []
    for L in sorted(buckets, reverse=True):
        idx = buckets[L]
        for s in range(0, len(idx), batch_size):
            plans.append((idx[s : s + batch_size], L))
    return plans


def _group_by_query(pairs: Sequence[Pair]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, (q, _) in enumerate(pairs):
        groups.setdefault(q, []).append(i)
    return groups


def _dtype_kwargs(torch_dtype) -> Dict[str, Any]:
    """transformers >= 4.56 renamed ``torch_dtype`` to ``dtype`` (5.x warns on the old name)."""
    import transformers

    try:
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    except Exception:  # noqa: BLE001
        major, minor = 5, 0
    return {"dtype": torch_dtype} if (major, minor) >= (4, 56) else {"torch_dtype": torch_dtype}


def _resolve_dtype(torch, dtype: Union[str, Any]):
    if isinstance(dtype, str):
        if not hasattr(torch, dtype):
            raise ValueError(f"unknown dtype {dtype!r}")
        return getattr(torch, dtype)
    return dtype


def _env_info(torch, device: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"torch": torch.__version__}
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            info["gpu"] = torch.cuda.get_device_name(torch.device(device))
            info["cuda"] = torch.version.cuda
        except Exception:  # noqa: BLE001
            pass
    return info


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
class TeacherScorer:
    """Pointwise teacher.  Subclasses implement ``score``/``n_tokens`` (and may override
    ``score_pairs`` for cross-query batching)."""

    name = "abstract"
    mode = "fast"
    batch_size = 1
    max_tokens_per_batch: Optional[int] = None

    def score(self, query: str, docs: Sequence[Doc]) -> List[float]:  # docs = [(title, text)]
        raise NotImplementedError

    def n_tokens(self, query: str, docs: Sequence[Doc]) -> List[int]:
        raise NotImplementedError

    def score_pairs(self, pairs: Sequence[Pair], return_ntok: bool = False):
        """Score (query, (title, text)) pairs that may span many queries.  Default: group by query
        and call ``score`` once per query; output order = input order."""
        out = [0.0] * len(pairs)
        nt = [0] * len(pairs)
        for q, idx in _group_by_query(pairs).items():
            docs = [pairs[i][1] for i in idx]
            for i, v in zip(idx, self.score(q, docs)):
                out[i] = float(v)
            if return_ntok:
                for i, v in zip(idx, self.n_tokens(q, docs)):
                    nt[i] = int(v)
        return (out, nt) if return_ntok else out

    def settings(self) -> Dict[str, Any]:
        """Everything that defines the scoring function (recorded in cache manifests)."""
        return {"class": type(self).__name__, "name": self.name, "mode": self.mode, "batch_size": self.batch_size, "max_tokens_per_batch": self.max_tokens_per_batch}


class CheapModel:
    name = "abstract"
    dim = 0

    def features(self, query: str, docs: Sequence[Doc]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def features_pairs(self, pairs: Sequence[Pair]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Features for pairs spanning many queries (default: group by query); input order kept."""
        n = len(pairs)
        emb = np.zeros((n, self.dim), dtype=np.float64)  # float64 keeps FakeCheap values bit-identical
        logit = np.zeros(n, dtype=np.float64)
        ntok = np.zeros(n, dtype=np.int64)
        for q, idx in _group_by_query(pairs).items():
            e, lg, nt = self.features(q, [pairs[i][1] for i in idx])
            emb[idx] = e
            logit[idx] = lg
            ntok[idx] = nt
        return emb, logit, ntok


# --------------------------------------------------------------------------- #
# Shared machinery for tokenized transformer teachers
# --------------------------------------------------------------------------- #
class _TokenizedTeacher(TeacherScorer):
    """encode_pairs -> token id lists; _forward(rows, L) -> scores; batching per ``mode``."""

    torch: Any = None
    device = "cpu"
    max_length = 512
    dtype_name = "float32"
    attn: Optional[str] = None

    def _init_batching(self, batch_size: int, mode: str, max_tokens_per_batch: Optional[int]) -> None:
        if mode not in TEACHER_MODES:
            raise ValueError(f"mode must be one of {TEACHER_MODES}, got {mode!r}")
        self.mode = mode
        self.batch_size = int(batch_size)
        self.max_tokens_per_batch = int(max_tokens_per_batch) if max_tokens_per_batch else None
        self.counters = {"forward_passes": 0, "rows": 0, "dummy_rows": 0, "real_tokens": 0, "padded_tokens": 0}

    def reset_counters(self) -> None:
        for k in self.counters:
            self.counters[k] = 0

    # subclasses -------------------------------------------------------------
    def encode_pairs(self, pairs: Sequence[Pair]) -> List[List[int]]:
        raise NotImplementedError

    def _forward(self, rows: List[List[int]], L: int) -> np.ndarray:
        raise NotImplementedError

    # public -----------------------------------------------------------------
    def plan(self, lengths: Sequence[int]) -> List[Tuple[List[int], int]]:
        if self.mode == "canonical":
            return plan_canonical(lengths, self.batch_size, CANONICAL_MULTIPLE)
        cap = self.max_tokens_per_batch
        return plan_fast(lengths, self.batch_size, cap, FAST_PAD_MULTIPLE)

    def score_seqs(self, seqs: List[List[int]]) -> np.ndarray:
        n = len(seqs)
        out = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return out
        lengths = [len(s) for s in seqs]
        fill = self.batch_size if self.mode == "canonical" else None
        with self.torch.inference_mode():
            for idx, L in self.plan(lengths):
                rows = [seqs[i] for i in idx]
                n_dummy = 0
                if fill is not None and len(rows) < fill:
                    n_dummy = fill - len(rows)
                    rows = rows + [rows[0]] * n_dummy  # dummy copies of the first row; outputs discarded
                s = self._forward(rows, L)
                if not np.all(np.isfinite(s[: len(idx)])):
                    raise RuntimeError(f"{self.name}: non-finite teacher score (batch rows={len(rows)}, L={L}, lengths={[lengths[i] for i in idx]})")
                out[idx] = s[: len(idx)]
                c = self.counters
                c["forward_passes"] += 1
                c["rows"] += len(rows)
                c["dummy_rows"] += n_dummy
                c["real_tokens"] += sum(len(r) for r in rows[: len(idx)])
                c["padded_tokens"] += len(rows) * L
        return out

    def score_pairs(self, pairs: Sequence[Pair], return_ntok: bool = False):
        seqs = self.encode_pairs(pairs)
        sc = [float(v) for v in self.score_seqs(seqs)]
        return (sc, [len(s) for s in seqs]) if return_ntok else sc

    def score(self, query: str, docs: Sequence[Doc]) -> List[float]:
        return self.score_pairs([(query, d) for d in docs])

    def n_tokens(self, query: str, docs: Sequence[Doc]) -> List[int]:
        return [len(s) for s in self.encode_pairs([(query, d) for d in docs])]

    def settings(self) -> Dict[str, Any]:
        s = super().settings()
        s.update({"max_length": self.max_length, "dtype": self.dtype_name, "attn": self.attn, "device": str(self.device), "canonical_multiple": CANONICAL_MULTIPLE, "fast_pad_multiple": FAST_PAD_MULTIPLE})
        s.update(_env_info(self.torch, self.device))
        return s

    def _load_common(self, device: Optional[str], dtype: str) -> None:
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_name = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")
        if not str(self.device).startswith("cuda") and self.dtype_name in ("bfloat16", "float16"):
            print(f"[models] warning: {self.dtype_name} on {self.device} is slow and imprecise; use dtype float32 for CPU runs", flush=True)


# --------------------------------------------------------------------------- #
# Qwen3-Reranker
# --------------------------------------------------------------------------- #
class Qwen3RerankerScorer(_TokenizedTeacher):
    def __init__(self, model_id: str = "Qwen/Qwen3-Reranker-4B", max_length: int = 512, batch_size: int = 8, dtype: str = "bfloat16", attn: str = "sdpa", device: Optional[str] = None, instruction: str = QWEN_INSTRUCTION, mode: str = "fast", max_tokens_per_batch: Optional[int] = None):
        self._load_common(device, dtype)
        torch = self.torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.model_id = model_id
        self.attn = attn
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw: Dict[str, Any] = dict(_dtype_kwargs(_resolve_dtype(torch, dtype)))
        if attn:
            kw["attn_implementation"] = attn
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to(self.device).eval()
        self.max_length = int(max_length)
        self.instruction = instruction
        self._init_batching(batch_size, mode, max_tokens_per_batch)
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self.pad_id = int(self.tok.pad_token_id)
        self.prefix_ids = self.tok.encode(QWEN_PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(QWEN_SUFFIX, add_special_tokens=False)
        if self.max_length - len(self.prefix_ids) - len(self.suffix_ids) < 16:
            raise ValueError("max_length too small for the Qwen3-Reranker template")
        # decoder body + the two lm_head rows in float32
        self.backbone = self.model.get_decoder() if hasattr(self.model, "get_decoder") else self.model.model
        head = self.model.get_output_embeddings()
        rows = torch.tensor([self.yes_id, self.no_id], device=head.weight.device)
        self.w_yes_no = head.weight.detach().index_select(0, rows).to(device=self.device, dtype=torch.float32).clone()
        b = getattr(head, "bias", None)
        self.b_yes_no = b.detach().index_select(0, rows).to(device=self.device, dtype=torch.float32).clone() if b is not None else None

    def encode_pairs(self, pairs: Sequence[Pair]) -> List[List[int]]:
        if not pairs:
            return []
        texts = [format_qwen_instruction(self.instruction, q, _doc_text(t, x)) for q, (t, x) in pairs]
        enc = self.tok(texts, padding=False, truncation="longest_first", return_attention_mask=False, max_length=self.max_length - len(self.prefix_ids) - len(self.suffix_ids))
        return [self.prefix_ids + list(ids) + self.suffix_ids for ids in enc["input_ids"]]

    def _forward(self, rows: List[List[int]], L: int) -> np.ndarray:
        torch = self.torch
        B = len(rows)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        for r, s in enumerate(rows):  # left padding: the last position is every row's last real token
            ids[r, L - len(s) :] = torch.as_tensor(s, dtype=torch.long)
            att[r, L - len(s) :] = 1
        pos = att.cumsum(-1) - 1
        pos.masked_fill_(att == 0, 1)
        dev = self.device
        out = self.backbone(input_ids=ids.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev), use_cache=False)
        h = out.last_hidden_state[:, -1, :].float()
        z = h @ self.w_yes_no.T  # [B, 2] = (z_yes, z_no) in float32
        if self.b_yes_no is not None:
            z = z + self.b_yes_no
        return (z[:, 0] - z[:, 1]).double().cpu().numpy()

    def settings(self) -> Dict[str, Any]:
        s = super().settings()
        s.update({"model_id": self.model_id, "instruction": self.instruction, "score": "z_yes - z_no at the last position (fp32 two-row head)", "yes_id": self.yes_id, "no_id": self.no_id, "padding_side": "left"})
        return s


# --------------------------------------------------------------------------- #
# BGE reranker
# --------------------------------------------------------------------------- #
def _find_sub(seq: Sequence[int], sub: Sequence[int], start: int = 0) -> int:
    n, m = len(seq), len(sub)
    for i in range(start, n - m + 1):
        if list(seq[i : i + m]) == list(sub):
            return i
    raise ValueError("sub-sequence not found")


def _pair_template(tok) -> Tuple[List[int], List[int], List[int]]:
    """Special tokens around (a, b) in ``tok(a, b)``: prefix, middle, suffix (version-independent)."""
    a = tok("alpha", add_special_tokens=False)["input_ids"]
    b = tok("omega", add_special_tokens=False)["input_ids"]
    full = tok("alpha", "omega")["input_ids"]
    p = _find_sub(full, a)
    r = _find_sub(full, b, p + len(a))
    return list(full[:p]), list(full[p + len(a) : r]), list(full[r + len(b) :])


class BGERerankerScorer(_TokenizedTeacher):
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512, batch_size: int = 16, dtype: str = "float16", device: Optional[str] = None, mode: str = "fast", max_tokens_per_batch: Optional[int] = None, query_max_length: Optional[int] = None, attn: Optional[str] = None):
        self._load_common(device, dtype)
        torch = self.torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model_id
        self.model_id = model_id
        self.attn = attn
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw: Dict[str, Any] = dict(_dtype_kwargs(_resolve_dtype(torch, dtype)))
        if attn:
            kw["attn_implementation"] = attn
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, **kw).to(self.device).eval()
        self.max_length = int(max_length)
        self.query_max_length = int(query_max_length) if query_max_length else self.max_length * 3 // 4  # FlagEmbedding default
        self._init_batching(batch_size, mode, max_tokens_per_batch)
        self.pad_id = int(self.tok.pad_token_id)
        self.tpl_prefix, self.tpl_mid, self.tpl_suffix = _pair_template(self.tok)
        self.n_special = len(self.tpl_prefix) + len(self.tpl_mid) + len(self.tpl_suffix)
        self.encoder = getattr(self.model, self.model.base_model_prefix)
        import copy

        self.head32 = copy.deepcopy(self.model.classifier).to(device=self.device, dtype=torch.float32).eval()

    def encode_pairs(self, pairs: Sequence[Pair]) -> List[List[int]]:
        if not pairs:
            return []
        uq = list(dict.fromkeys(q for q, _ in pairs))
        q_ids = dict(zip(uq, self.tok(uq, add_special_tokens=False, truncation=True, max_length=self.query_max_length)["input_ids"]))
        d_ids = self.tok([_doc_text(t, x) for _, (t, x) in pairs], add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        out = []
        for (q, _), d in zip(pairs, d_ids):
            qi = list(q_ids[q])
            budget = self.max_length - len(qi) - self.n_special  # truncation='only_second'
            d = list(d)[: max(0, budget)]
            out.append(self.tpl_prefix + qi + self.tpl_mid + d + self.tpl_suffix)
        return out

    def _forward(self, rows: List[List[int]], L: int) -> np.ndarray:
        torch = self.torch
        B = len(rows)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        for r, s in enumerate(rows):  # right padding (XLM-R position ids are derived from non-pad tokens)
            ids[r, : len(s)] = torch.as_tensor(s, dtype=torch.long)
            att[r, : len(s)] = 1
        dev = self.device
        hs = self.encoder(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
        logits = self.head32(hs[:, :1, :].float()).view(-1)
        return logits.double().cpu().numpy()

    def settings(self) -> Dict[str, Any]:
        s = super().settings()
        s.update({"model_id": self.model_id, "query_max_length": self.query_max_length, "score": "raw logit (fp32 classification head)", "truncation": "query<=3/4*max_length, only_second", "padding_side": "right"})
        return s


# --------------------------------------------------------------------------- #
# MiniLM cheap model
# --------------------------------------------------------------------------- #
class MiniLMCheap(CheapModel):
    dim = 384

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2", max_length: int = 512, batch_size: int = 64, device: Optional[str] = None, name: Optional[str] = None, max_tokens_per_batch: Optional[int] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name or str(model_id)
        self.model_id = str(model_id)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device).eval()
        self.base = getattr(self.model, self.model.base_model_prefix)
        self.dim = int(self.model.config.hidden_size)
        self.max_length, self.batch_size = int(max_length), int(batch_size)
        self.max_tokens_per_batch = max_tokens_per_batch
        self.pad_id = int(self.tok.pad_token_id)

    def encode_pairs(self, pairs: Sequence[Pair]) -> Dict[str, List[List[int]]]:
        qs = [q for q, _ in pairs]
        ds = [_doc_text(t, x) for _, (t, x) in pairs]
        enc = self.tok(qs, ds, padding=False, truncation="longest_first", max_length=self.max_length)
        return {k: [list(v) for v in enc[k]] for k in ("input_ids", "token_type_ids") if k in enc}

    def features_pairs(self, pairs: Sequence[Pair]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        n = len(pairs)
        emb = np.zeros((n, self.dim), dtype=np.float32)
        logits = np.zeros(n, dtype=np.float64)
        if n == 0:
            return emb, logits, np.zeros(0, dtype=np.int64)
        enc = self.encode_pairs(pairs)
        ids_all = enc["input_ids"]
        tt_all = enc.get("token_type_ids")
        ntok = np.array([len(x) for x in ids_all], dtype=np.int64)
        with torch.inference_mode():
            for idx, L in plan_fast(ntok.tolist(), self.batch_size, self.max_tokens_per_batch, pad_multiple=1):
                B = len(idx)
                ids = torch.full((B, L), self.pad_id, dtype=torch.long)
                att = torch.zeros((B, L), dtype=torch.long)
                tt = torch.zeros((B, L), dtype=torch.long)
                for r, i in enumerate(idx):
                    m = len(ids_all[i])
                    ids[r, :m] = torch.as_tensor(ids_all[i], dtype=torch.long)
                    att[r, :m] = 1
                    if tt_all is not None:
                        tt[r, :m] = torch.as_tensor(tt_all[i], dtype=torch.long)
                kw = {"input_ids": ids.to(self.device), "attention_mask": att.to(self.device)}
                if tt_all is not None:
                    kw["token_type_ids"] = tt.to(self.device)
                out = self.base(**kw)
                pooled = out.pooler_output  # tanh(W h_cls): the 384-d representation fed to the classifier
                logit = self.model.classifier(pooled).view(-1)
                emb[idx] = pooled.float().cpu().numpy()
                logits[idx] = logit.float().cpu().numpy()
        return emb, logits, ntok

    def features(self, query: str, docs: Sequence[Doc]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.features_pairs([(query, d) for d in docs])


# --------------------------------------------------------------------------- #
# Synthetic backends (smoke tests only)
# --------------------------------------------------------------------------- #
def _latent(query: str, doc: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}||{query}||{doc[:200]}".encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "little") / 2**64
    # inverse-normal via Box-Muller with a second hash
    v = int.from_bytes(h[8:16], "little") / 2**64
    return math.sqrt(-2 * math.log(max(u, 1e-12))) * math.cos(2 * math.pi * v)


def _overlap(query: str, doc: str) -> float:
    q = set(query.lower().split())
    d = set(doc.lower().split())
    return len(q & d) / max(1, len(q))


class FakeTeacher(TeacherScorer):
    name = "fake-teacher"

    def __init__(self, seed: int = 0, noise: float = 0.3):
        self.seed, self.noise = seed, noise

    def score(self, query, docs):
        out = []
        for t, x in docs:
            d = _doc_text(t, x)
            out.append(4.0 * _overlap(query, d) + 1.5 * _latent(query, d, "latent") + self.noise * _latent(query, d, f"tn{self.seed}"))
        return out

    def n_tokens(self, query, docs):
        return [min(512, 12 + len(_doc_text(t, x).split()) + len(query.split())) for t, x in docs]


class FakeCheap(CheapModel):
    name = "fake-cheap"
    dim = 16

    def __init__(self, seed: int = 0, noise: float = 1.0):
        self.seed, self.noise = seed, noise

    def features(self, query, docs):
        embs, logits, ntok = [], [], []
        for t, x in docs:
            d = _doc_text(t, x)
            lat = _latent(query, d, "latent")
            ov = _overlap(query, d)
            logit = 4.0 * ov + 1.5 * lat + self.noise * _latent(query, d, f"cn{self.seed}")
            e = np.array([_latent(query, d, f"e{j}") for j in range(self.dim)])
            e[0] = lat
            e[1] = ov
            embs.append(e)
            logits.append(logit)
            ntok.append(min(512, 8 + len(d.split()) + len(query.split())))
        if not embs:
            return np.zeros((0, self.dim)), np.zeros(0), np.zeros(0, dtype=np.int64)
        return np.stack(embs, 0), np.asarray(logits, float), np.asarray(ntok)


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
_QWEN_KEYS = ("max_length", "batch_size", "dtype", "attn", "device", "instruction", "mode", "max_tokens_per_batch")
_BGE_KEYS = ("max_length", "batch_size", "dtype", "device", "mode", "max_tokens_per_batch", "query_max_length", "attn")
QWEN_IDS = {"qwen3-4b": "Qwen/Qwen3-Reranker-4B", "qwen3-reranker-4b": "Qwen/Qwen3-Reranker-4B", "qwen3-0.6b": "Qwen/Qwen3-Reranker-0.6B", "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B", "qwen3-8b": "Qwen/Qwen3-Reranker-8B"}


def get_teacher(name: str, cfg: Optional[Dict] = None) -> TeacherScorer:
    """Teacher by config key.  ``cfg`` = configs/main.yaml teachers.<name> (unknown keys ignored;
    ``model_id`` overrides the hub id, e.g. a local snapshot directory)."""
    cfg = dict(cfg or {})
    if name in ("fake", "fake-teacher", "fake2"):
        return FakeTeacher(seed=cfg.get("seed", 1 if name == "fake2" else 0), noise=0.3 if name != "fake2" else 0.8)
    if name in QWEN_IDS:
        return Qwen3RerankerScorer(cfg.get("model_id", QWEN_IDS[name]), **{k: v for k, v in cfg.items() if k in _QWEN_KEYS and v is not None})
    if name in ("bge-m3", "bge-reranker-v2-m3"):
        return BGERerankerScorer(cfg.get("model_id", "BAAI/bge-reranker-v2-m3"), **{k: v for k, v in cfg.items() if k in _BGE_KEYS and v is not None})
    raise ValueError(f"unknown teacher {name}")


def get_cheap(name: str, cfg: Optional[Dict] = None) -> CheapModel:
    """Cheap model by name.  ``minilm-distilled`` needs cfg['path'] (models/cheap_distilled/<teacher>)."""
    cfg = dict(cfg or {})
    keys = ("max_length", "batch_size", "device", "max_tokens_per_batch")
    if name in ("fake", "fake-cheap"):
        return FakeCheap(seed=cfg.get("seed", 0))
    if name in ("minilm", "ms-marco-MiniLM-L6-v2"):
        return MiniLMCheap(cfg.get("model_id", "cross-encoder/ms-marco-MiniLM-L6-v2"), **{k: v for k, v in cfg.items() if k in keys and v is not None})
    if name in ("minilm-distilled",):
        path = cfg.get("path")
        if not path:
            raise ValueError("cheap model 'minilm-distilled' needs cfg['path'] = models/cheap_distilled/<teacher> (see docs/MODELS.md)")
        return MiniLMCheap(str(path), name=cfg.get("name", f"minilm-distilled:{path}"), **{k: v for k, v in cfg.items() if k in keys and v is not None})
    raise ValueError(f"unknown cheap model {name}")
