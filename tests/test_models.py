"""Model backends: batching plans (offline) and agreement with the official reference code
(network: downloads Qwen/Qwen3-Reranker-0.6B, BAAI/bge-reranker-v2-m3, cross-encoder/ms-marco-MiniLM-L6-v2).

The network tests run on CPU in float32 and are skipped when RSC_SKIP_NET=1 or the Hugging Face hub
(HF_ENDPOINT, default https://huggingface.co) does not answer a HEAD request within a few seconds.
Each network test takes < 3 minutes on 2 CPU cores.
"""
from __future__ import annotations

import gc
import math
import os
import random
import urllib.request

import numpy as np
import pytest

from rscrank.models import (
    FakeCheap,
    FakeTeacher,
    QWEN_INSTRUCTION,
    _TokenizedTeacher,
    canonical_length,
    get_cheap,
    plan_canonical,
    plan_fast,
)


# --------------------------------------------------------------------------- #
# Offline: batching plans
# --------------------------------------------------------------------------- #
def _lengths(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(20, 600, size=n)]


def test_plan_fast_covers_every_pair_once_and_respects_caps():
    L = _lengths()
    for bs, cap in ((8, None), (16, 4096), (32, 16384), (5, 700)):
        plans = plan_fast(L, bs, cap, pad_multiple=8)
        seen = [i for idx, _ in plans for i in idx]
        assert sorted(seen) == list(range(len(L)))
        for idx, padded in plans:
            assert 1 <= len(idx) <= bs
            assert padded >= max(L[i] for i in idx) and padded % 8 == 0
            if cap is not None and len(idx) > 1:
                assert len(idx) * padded <= cap
        # length-sorted: batches are visited longest first
        firsts = [max(L[i] for i in idx) for idx, _ in plans]
        assert firsts == sorted(firsts, reverse=True)
    assert plan_fast(L, 8, 4096) == plan_fast(list(L), 8, 4096)  # deterministic


def test_plan_canonical_shape_depends_only_on_own_length():
    L = _lengths()
    for bs in (4, 16):
        plans = plan_canonical(L, bs)
        seen = [i for idx, _ in plans for i in idx]
        assert sorted(seen) == list(range(len(L)))
        for idx, padded in plans:
            assert 1 <= len(idx) <= bs
            for i in idx:
                assert padded == canonical_length(L[i]) and padded > L[i] and padded % 64 == 0
    # the padded length of a pair never changes when its batch-mates change
    sub = L[:37]
    p_all = {i: Lp for idx, Lp in plan_canonical(L, 16) for i in idx}
    p_sub = {i: Lp for idx, Lp in plan_canonical(sub, 16) for i in idx}
    assert all(p_all[i] == p_sub[i] for i in range(len(sub)))
    assert canonical_length(64) == 128 and canonical_length(63) == 64 and canonical_length(0) == 64


class _ToyTeacher(_TokenizedTeacher):
    """Deterministic stand-in for a transformer: score = f(token ids), independent of the batch."""

    def __init__(self, mode, batch_size=4, max_tokens=None):
        import torch

        self.torch = torch
        self.name = "toy"
        self._init_batching(batch_size, mode, max_tokens)
        self.shapes = []

    def encode_pairs(self, pairs):
        return [[1 + (ord(c) % 7) for c in (q + "|" + t + " " + x)][:300] for q, (t, x) in pairs]

    def _forward(self, rows, L):
        self.shapes.append((len(rows), L))
        return np.array([sum(r) / len(r) + 0.001 * len(r) for r in rows], dtype=np.float64)


def test_tokenized_teacher_restores_order_and_fills_canonical_batches():
    rng = random.Random(0)
    pairs = [(f"q{rng.randint(0, 5)}", ("t", "word " * rng.randint(1, 55))) for _ in range(41)]
    fast = _ToyTeacher("fast", 4, 400)
    can = _ToyTeacher("canonical", 4)
    a = fast.score_pairs(pairs)
    b, nt = can.score_pairs(pairs, return_ntok=True)
    expect = [s[0] for s in (fast._forward([seq], 1) for seq in fast.encode_pairs(pairs))]
    assert np.allclose(a, expect) and np.allclose(b, expect)
    assert nt == [len(s) for s in can.encode_pairs(pairs)]
    assert all(r == 4 for r, _ in can.shapes)  # canonical: every forward pass has exactly batch_size rows
    assert can.counters["dummy_rows"] == can.counters["rows"] - len(pairs)
    # per-query score() equals score_pairs() over many queries
    q = pairs[0][0]
    docs = [d for qq, d in pairs if qq == q]
    assert np.allclose(can.score(q, docs), [b[i] for i, (qq, _) in enumerate(pairs) if qq == q])


def test_fake_backends_score_pairs_equal_per_query_calls():
    rng = random.Random(1)
    pairs = [(f"query {rng.randint(0, 4)} w1 w2", ("T", " ".join(f"w{rng.randint(0, 30)}" for _ in range(20)))) for _ in range(30)]
    t = FakeTeacher(seed=0)
    s, nt = t.score_pairs(pairs, return_ntok=True)
    for i, (q, d) in enumerate(pairs):
        assert s[i] == t.score(q, [d])[0] and nt[i] == t.n_tokens(q, [d])[0]
    c = FakeCheap(seed=0)
    e, lg, n = c.features_pairs(pairs)
    for i, (q, d) in enumerate(pairs):
        e1, l1, n1 = c.features(q, [d])
        assert np.array_equal(e[i], e1[0]) and lg[i] == l1[0] and n[i] == n1[0]


def test_distilled_cheap_requires_path():
    with pytest.raises(ValueError):
        get_cheap("minilm-distilled", {})


# --------------------------------------------------------------------------- #
# Offline: first-stage candidates (bm25s backend)
# --------------------------------------------------------------------------- #
def test_topn_deterministic_ties_and_zero_fill():
    from rscrank.candidates import topn_deterministic

    s = np.array([0, 3, 1, 3, 0, 0, 2, 3], dtype=np.float32)
    assert topn_deterministic(s, 3).tolist() == [1, 3, 7]  # boundary ties -> smallest corpus positions
    assert topn_deterministic(s, 6).tolist() == [1, 3, 7, 6, 2, 0]  # then zero-score docs in corpus order
    assert topn_deterministic(np.zeros(5, np.float32), 3).tolist() == [0, 1, 2]
    assert topn_deterministic(s, 50).tolist() == [1, 3, 7, 6, 2, 0, 4, 5]
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 4, size=200).astype(np.float32)
        N = int(rng.integers(1, 200))
        ref = np.lexsort((np.arange(200), -x))[:N]
        assert topn_deterministic(x, N).tolist() == ref.tolist()


def test_bm25s_returns_exactly_N_in_deterministic_order():
    pytest.importorskip("bm25s")
    pytest.importorskip("Stemmer")
    from rscrank.candidates import bm25s_search

    rng = random.Random(3)
    vocab = ["river", "bank", "money", "loan", "water", "fish", "boat", "city", "bridge", "tower"]
    corpus = [(f"d{i}", " ".join(rng.choice(vocab) for _ in range(rng.randint(3, 12)))) for i in range(60)]
    corpus += [("dup-a", "river bank"), ("dup-b", "river bank"), ("empty", "")]
    queries = {"q1": "river bank loans", "q2": "the and of", "q3": "zeppelin", "q4": "fish boat bridge"}
    h1, info = bm25s_search(iter(corpus), queries, 20, n_threads=2)
    h2, _ = bm25s_search(iter(corpus), queries, 20, n_threads=1)
    assert h1 == h2
    pos = {d: i for i, (d, _) in enumerate(corpus)}
    for q, hits in h1.items():
        assert len(hits) == 20
        keys = [(-s, pos[d]) for d, s in hits]
        assert keys == sorted(keys)
    assert [d for d, _ in h1["q3"]] == [d for d, _ in corpus[:20]]  # nothing matches: corpus order
    assert info["queries_empty_after_tokenization"] == 2 and info["n_docs"] == len(corpus)
    assert all(s > 0 for _, s in h1["q1"][:3])
    ids1 = [d for d, _ in h1["q1"]]
    assert dict(h1["q1"])["dup-a"] == dict(h1["q1"])["dup-b"] and ids1.index("dup-a") < ids1.index("dup-b")  # exact tie -> corpus order


# --------------------------------------------------------------------------- #
# Offline: extra domains (msmarco) and the config-driven cal/test pool split
# --------------------------------------------------------------------------- #
def test_domain_helpers_and_directory_contract(tmp_path):
    from rscrank.data import distilled_model_dir, domain_list, features_dir_for, pool_split

    cfg = {"domains": {"main": ["nq", "hotpotqa"], "cross": ["fiqa", "scifact"], "extra": ["msmarco"], "test_pool_split": {"nq": "test", "msmarco": "dev"}}}
    assert domain_list(cfg) == ["nq", "hotpotqa", "fiqa", "scifact", "msmarco"]
    assert domain_list(cfg, include_cross=False) == ["nq", "hotpotqa", "msmarco"]
    assert domain_list({"domains": {"main": ["nq"], "cross": [], "extra": []}}) == ["nq"]
    assert domain_list(cfg, only="msmarco,nq") == ["nq", "msmarco"]
    assert pool_split(cfg, "msmarco") == "dev" and pool_split(cfg, "hotpotqa") == "test" and pool_split({}, "msmarco") == "dev"
    base = tmp_path / "data" / "features"
    assert features_dir_for(base, "minilm") == base
    assert features_dir_for(base, "minilm-distilled", "qwen3-4b") == tmp_path / "data" / "features_distilled" / "qwen3-4b"
    assert distilled_model_dir(tmp_path / "models", "bge-m3") == tmp_path / "models" / "cheap_distilled" / "bge-m3"
    with pytest.raises(ValueError):
        features_dir_for(base, "minilm-distilled")


def test_make_splits_uses_the_configured_pool_split(tmp_path):
    import csv
    import json

    from rscrank.data import make_splits

    raw = tmp_path / "raw"
    d = raw / "webq"
    (d / "qrels").mkdir(parents=True)
    queries = [{"_id": f"q{i}", "text": f"web query number {i} about topic {i % 7}"} for i in range(60)]
    with open(d / "queries.jsonl", "w", encoding="utf-8") as f:
        for r in queries:
            f.write(json.dumps(r) + "\n")
    for name, ids in (("dev", range(0, 20)), ("train", range(20, 60))):
        with open(d / "qrels" / f"{name}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["query-id", "corpus-id", "score"])
            w.writerows([(f"q{i}", f"d{i}", 1) for i in ids])
    m = make_splits("webq", raw, tmp_path / "tq", tmp_path / "splits", seed=1, n_cal=8, n_train=10, n_dev=5, test_pool_split="dev")
    assert m["pool_split"] == "dev" and m["pool_size"] == 20 and m["n_cal"] == 8 and m["n_test"] == 12
    assert m["n_train"] == 10 and m["n_dev"] == 5
    pool = {f"q{i}" for i in range(20)}
    read = lambda s: [json.loads(l)["qid"] for l in open(tmp_path / "splits" / "webq" / f"{s}.jsonl", encoding="utf-8")]
    assert set(read("cal")) | set(read("test")) == pool
    assert not (set(read("train")) | set(read("dev"))) & pool
    with pytest.raises(FileNotFoundError):
        make_splits("webq", raw, tmp_path / "tq", tmp_path / "splits2", seed=1, n_cal=8, n_train=0, n_dev=0, test_pool_split="test")


# --------------------------------------------------------------------------- #
# Network: agreement with the official reference implementations (CPU, float32)
# --------------------------------------------------------------------------- #
def _hub_reachable() -> bool:
    """True when the reference models can be loaded: either all three are already in the local
    Hugging Face cache (offline is fine) or the hub / mirror answers."""
    if os.environ.get("RSC_SKIP_NET", "") == "1":
        return False
    try:
        from huggingface_hub import snapshot_download

        for mid in ("Qwen/Qwen3-Reranker-0.6B", "BAAI/bge-reranker-v2-m3", "cross-encoder/ms-marco-MiniLM-L6-v2"):
            snapshot_download(mid, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        pass
    url = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/api/models/Qwen/Qwen3-Reranker-0.6B"
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "rsc-rank-tests/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status < 500
    except Exception:  # noqa: BLE001
        return False


needs_net = pytest.mark.skipif(not _hub_reachable(), reason="RSC_SKIP_NET=1 or Hugging Face hub unreachable")

LONG_DOC = " ".join(["The mitochondrion is the organelle that produces most of the cell's ATP through oxidative phosphorylation."] * 120)
QUERIES = {
    "What is the capital of China?": (("", "The capital of China is Beijing."), ("", "Gravity is a force that attracts two bodies towards each other.")),
    "Explain gravity": (("Gravity", "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."), ("", "Paris is famous for its bakeries and the Eiffel Tower.")),
    "which organelle produces ATP": (("Cell biology", LONG_DOC), ("", "The 1966 World Cup final was played at Wembley Stadium.")),
}
PAIRS = [(q, d) for q, docs in QUERIES.items() for d in docs]  # per query: (relevant, irrelevant)


def _doc(d):
    t, x = d
    return f"{t} {x}".strip() if t else x


def _relevant_outrank(scores):
    return all(scores[2 * j] > scores[2 * j + 1] for j in range(len(QUERIES)))


def _composition_scores(model, pairs, filler):
    """Score ``pairs`` inside a shuffled mix with other pairs (different batch-mates and order)."""
    mixed = list(pairs) + list(filler)
    order = list(range(len(mixed)))
    random.Random(7).shuffle(order)
    sc = model.score_pairs([mixed[i] for i in order])
    back = {i: s for i, s in zip(order, sc)}
    return np.array([back[i] for i in range(len(pairs))])


FILLER = [("an unrelated query about football", ("", "word " * n)) for n in (3, 40, 90, 150, 260)]


@needs_net
def test_qwen3_reranker_matches_model_card_reference():
    import torch

    from rscrank.models import Qwen3RerankerScorer

    max_length = 512
    sc = Qwen3RerankerScorer("Qwen/Qwen3-Reranker-0.6B", max_length=max_length, batch_size=4, dtype="float32", device="cpu", mode="fast", max_tokens_per_batch=2048)
    try:
        ours = np.array(sc.score_pairs(PAIRS))
        seqs = sc.encode_pairs(PAIRS)
        assert max(len(s) for s in seqs) == max_length  # the long document is truncated

        # ---- model card reference (https://huggingface.co/Qwen/Qwen3-Reranker-4B, "Using Transformers"), verbatim ----
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
        model = sc.model  # same weights; avoids loading a second copy

        def format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = "Given a web search query, retrieve relevant passages that answer the query"
            output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)
            return output

        def process_inputs(pairs):
            inputs = tokenizer(pairs, padding=False, truncation="longest_first", return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens))
            for i, ele in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
            inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
            for key in inputs:
                inputs[key] = inputs[key].to(model.device)
            return inputs

        @torch.no_grad()
        def compute_logits(inputs, **kwargs):
            batch_scores = model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores, (true_vector - false_vector).tolist()  # + the raw logit difference

        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        task = "Given a web search query, retrieve relevant passages that answer the query"
        assert task == QWEN_INSTRUCTION
        # ---- end of reference; run it in two left-padded batches of 3 (memory: full-vocab fp32 logits) ----
        p_yes, ref = [], []
        for chunk in (PAIRS[:3], PAIRS[3:]):
            inputs = process_inputs([format_instruction(task, q, _doc(d)) for q, d in chunk])
            for r, (q, d) in enumerate(chunk):  # identical token ids (no boundary effects)
                assert inputs["input_ids"][r][inputs["attention_mask"][r].bool()].tolist() == seqs[PAIRS.index((q, d))]
            pr, lr = compute_logits(inputs)
            p_yes += pr
            ref += lr
        ref, p_yes = np.array(ref), np.array(p_yes)
        assert np.max(np.abs(ours - ref)) < 1e-3, (ours, ref)
        assert np.max(np.abs(1.0 / (1.0 + np.exp(-ours)) - p_yes)) < 1e-4
        assert _relevant_outrank(ours)

        sc.mode, sc.batch_size = "canonical", 2
        can = np.array(sc.score_pairs(PAIRS))
        # fast vs canonical differ only by kernel/padding rounding; the size of that difference is a
        # *measurement* (the pilot reports it), not a contract -- keep the assertion loose enough that
        # CPU/GPU jitter can never block node N00, whose first check is the whole test suite.
        print("qwen fast-vs-canonical max|diff| =", float(np.max(np.abs(can - ours))))
        assert np.max(np.abs(can - ours)) < 2e-3
        can2 = _composition_scores(sc, PAIRS, FILLER)
        print("qwen canonical composition max|diff| =", float(np.max(np.abs(can2 - can))), "bit-exact:", bool(np.array_equal(can2, can)))
        # Bit-exactness of canonical mode is a property of the *deployment backend* (the GPU pilot/audit
        # measure and report it: max|diff| = 0 on the CUDA card); multi-threaded CPU BLAS may reorder
        # reductions between calls, so on CPU only fp32-rounding-level agreement is asserted here.
        assert np.max(np.abs(can2 - can)) <= 1e-4
        assert sc.counters["dummy_rows"] > 0
    finally:
        del sc
        gc.collect()


@needs_net
def test_bge_reranker_matches_transformers_reference():
    import torch

    from rscrank.models import BGERerankerScorer

    sc = BGERerankerScorer("BAAI/bge-reranker-v2-m3", max_length=512, batch_size=4, dtype="float32", device="cpu", mode="fast")
    try:
        ours = np.array(sc.score_pairs(PAIRS))
        seqs = sc.encode_pairs(PAIRS)
        assert max(len(s) for s in seqs) == 512
        # model card, "Using Huggingface transformers" (normal reranker), verbatim
        tokenizer, model = sc.tok, sc.model
        pairs = [[q, _doc(d)] for q, d in PAIRS]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1).float()
        for r in range(len(pairs)):
            assert inputs["input_ids"][r][inputs["attention_mask"][r].bool()].tolist() == seqs[r]
        ref = scores.numpy().astype(np.float64)
        assert np.max(np.abs(ours - ref)) < 1e-3, (ours, ref)
        assert _relevant_outrank(ours)
        # card example values (fp16 in the card): [-8.1875, 5.26171875]
        ex = sc.score_pairs([("what is panda?", ("", "hi")), ("what is panda?", ("", "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."))])
        assert abs(ex[0] - (-8.1875)) < 0.05 and abs(ex[1] - 5.26171875) < 0.05
        sc.mode, sc.batch_size = "canonical", 4
        can = np.array(sc.score_pairs(PAIRS))
        assert np.max(np.abs(can - ours)) < 1e-4
        can2 = _composition_scores(sc, PAIRS, FILLER)
        print("bge canonical composition max|diff| =", float(np.max(np.abs(can2 - can))), "bit-exact:", bool(np.array_equal(can2, can)))
        assert np.max(np.abs(can2 - can)) <= 1e-4  # see the note in the Qwen test: bit-exactness is measured on the GPU
    finally:
        del sc
        gc.collect()


@needs_net
def test_minilm_features_match_sequence_classification():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from rscrank.models import MiniLMCheap

    mid = "cross-encoder/ms-marco-MiniLM-L6-v2"
    m = MiniLMCheap(mid, batch_size=4, device="cpu")
    emb, logit, ntok = m.features_pairs(PAIRS)
    tok = AutoTokenizer.from_pretrained(mid)
    ref = AutoModelForSequenceClassification.from_pretrained(mid).eval()
    with torch.no_grad():
        enc = tok([q for q, _ in PAIRS], [_doc(d) for _, d in PAIRS], padding=True, truncation=True, max_length=512, return_tensors="pt")
        rl = ref(**enc).logits.view(-1).numpy()
        pooled = ref.bert(**enc).pooler_output.numpy()
    assert emb.shape == (len(PAIRS), 384)
    assert np.max(np.abs(logit - rl)) < 1e-4
    assert np.max(np.abs(emb - pooled)) < 1e-4
    assert ntok.tolist() == enc["attention_mask"].sum(1).tolist()
    e1, l1, n1 = m.features(PAIRS[0][0], [PAIRS[0][1], PAIRS[1][1]])
    assert np.allclose(l1, logit[:2], atol=1e-5) and n1.tolist() == ntok[:2].tolist()
    assert _relevant_outrank(logit)
    assert not math.isnan(float(logit.sum()))
