#!/usr/bin/env python3
"""Real SOTA retrieval baselines over the pooled FreshBench-RAG corpus.

Retrievers (all downloaded public checkpoints, CPU):
  bm25    rank_bm25.BM25Okapi                              (sparse lexical)
  dpr     facebook/dpr-*-multiset-base                     (dense, Karpukhin+20)
  bge     BAAI/bge-base-en-v1.5                            (SOTA bi-encoder)
  splade  naver/splade-cocondenser-ensembledistil         (SOTA learned sparse)
  colbert colbert-ir/colbertv2.0 via pylate                (SOTA late interaction)
  hybrid  RRF(bm25, bge)                                   (fusion)

Any retriever whose checkpoint fails to load is SKIPPED and logged — never
replaced by a fake ranking. Outputs:
  data/processed/retrieval.json          {retriever: {qid: [doc_id,...]}} (+oracle)
  data/processed/retrieval_metrics.json  recall@k / hit@k per retriever x split
"""
import argparse
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
_TOK = re.compile(r"[A-Za-z0-9]+")


def tok(s):
    return _TOK.findall(s.lower())


def load():
    questions = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()]
    corpus = [json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines()]
    qrels = json.loads((PROC / "qrels.json").read_text())
    return questions, corpus, qrels


# ---- per-question temporal admissibility ---------------------------------
# The Method and Appendix C both promise that a question is never scored against
# a document that could not have existed when it was asked. The builder enforced
# that only against the question a document was COLLECTED FOR, then pooled every
# survivor, so ranking the pool admitted post-question evidence for every OTHER
# question. `retrieval_full.py` already carried the mask; the main index never
# applied it. Measured consequence before this fix, from the shipped rankings:
# 11.66% of Hybrid contexts and 34.28% of questions, and the exposure was
# concentrated in one arm of the treatment variable (Static 19.75% vs Fresh
# 4.77%). See pipeline/temporal_leakage_audit.py.
LAG = 3  # days; matches configs/datasets.yaml drop_docs_after_question_date_days
_MASK = {"on": True, "rows": None, "qindex": None}


def parse_date(s):
    from datetime import date
    return date.fromisoformat(str(s)[:10])


def build_time_mask(questions, corpus):
    """(n_q, n_d) bool: may question i legally see document j?"""
    dday = np.array([parse_date(d["publish_date"]).toordinal() if d.get("publish_date")
                     else -1 for d in corpus])
    qday = np.array([parse_date(q["date"]).toordinal() for q in questions])
    undated = dday < 0                       # undated docs cannot be shown to violate
    return (dday[None, :] <= (qday[:, None] + LAG)) | undated[None, :]


def mask_for(qid):
    if not _MASK["on"] or _MASK["rows"] is None:
        return None
    i = _MASK["qindex"].get(qid)
    return None if i is None else _MASK["rows"][i]


def topk_ids(scores, ids, k, qid=None):
    """Top-k under the per-question time mask.

    Inadmissible documents are pushed to -inf BEFORE the top-k cut, so masking
    changes the ranking rather than just deleting rows from it -- filtering an
    already-computed top-5 would silently return fewer than k contexts and
    change the retrieval budget along with the admissibility.
    """
    m = mask_for(qid)
    if m is not None:
        scores = np.where(m, scores, -np.inf)
        k = min(k, int(m.sum()))
        if k <= 0:
            return []
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [ids[i] for i in idx if np.isfinite(scores[i])]


# ----------------------------- retrievers ---------------------------------
def r_bm25(questions, corpus, ids, k, cfg):
    from rank_bm25 import BM25Okapi
    c = cfg["bm25"]
    bm = BM25Okapi([tok(d["text"]) for d in corpus], k1=c["k1"], b=c["b"])
    out, scores_by_q = {}, {}
    for q in questions:
        s = np.asarray(bm.get_scores(tok(q["question"])))
        out[q["qid"]] = topk_ids(s, ids, k, q["qid"])
        scores_by_q[q["qid"]] = s
    return out, scores_by_q


def r_bge(questions, corpus, ids, k, cfg):
    from sentence_transformers import SentenceTransformer
    c = cfg["bge"]
    m = SentenceTransformer(c["model"])
    demb = m.encode([d["text"] for d in corpus], batch_size=64,
                    normalize_embeddings=True, show_progress_bar=True)
    qemb = m.encode([c["query_prefix"] + q["question"] for q in questions], batch_size=64,
                    normalize_embeddings=True, show_progress_bar=True)
    sims = qemb @ demb.T
    out, scores_by_q = {}, {}
    for i, q in enumerate(questions):
        out[q["qid"]] = topk_ids(sims[i], ids, k, q["qid"])
        scores_by_q[q["qid"]] = sims[i]
    return out, scores_by_q


def r_dpr(questions, corpus, ids, k, cfg):
    import torch
    from transformers import (DPRContextEncoder, DPRContextEncoderTokenizerFast,
                              DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast)
    c = cfg["dpr"]
    qtok = DPRQuestionEncoderTokenizerFast.from_pretrained(c["q_model"])
    qenc = DPRQuestionEncoder.from_pretrained(c["q_model"]).eval()
    ctok = DPRContextEncoderTokenizerFast.from_pretrained(c["ctx_model"])
    cenc = DPRContextEncoder.from_pretrained(c["ctx_model"]).eval()

    @torch.no_grad()
    def embed(texts, tokz, enc, bs=32):
        vecs = []
        for i in range(0, len(texts), bs):
            b = tokz(texts[i:i + bs], truncation=True, max_length=256,
                     padding=True, return_tensors="pt")
            vecs.append(enc(**b).pooler_output.cpu().numpy())
        return np.vstack(vecs)

    demb = embed([d["text"] for d in corpus], ctok, cenc)
    qemb = embed([q["question"] for q in questions], qtok, qenc)
    sims = qemb @ demb.T
    out, scores_by_q = {}, {}
    for i, q in enumerate(questions):
        out[q["qid"]] = topk_ids(sims[i], ids, k, q["qid"])
        scores_by_q[q["qid"]] = sims[i]
    return out, scores_by_q


def r_splade(questions, corpus, ids, k, cfg):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    c = cfg["splade"]
    tokz = AutoTokenizer.from_pretrained(c["model"])
    model = AutoModelForMaskedLM.from_pretrained(c["model"]).eval()

    @torch.no_grad()
    def splade_vecs(texts, bs=16):
        vecs = []
        for i in range(0, len(texts), bs):
            b = tokz(texts[i:i + bs], truncation=True, max_length=256,
                     padding=True, return_tensors="pt")
            logits = model(**b).logits                       # (B,L,V)
            rep = torch.log1p(torch.relu(logits))
            rep = torch.max(rep * b["attention_mask"].unsqueeze(-1), dim=1).values
            vecs.append(rep.cpu().numpy().astype(np.float32))
        return np.vstack(vecs)

    demb = splade_vecs([d["text"] for d in corpus])
    qemb = splade_vecs([q["question"] for q in questions])
    sims = qemb @ demb.T
    out, scores_by_q = {}, {}
    for i, q in enumerate(questions):
        out[q["qid"]] = topk_ids(sims[i], ids, k, q["qid"])
        scores_by_q[q["qid"]] = sims[i]
    return out, scores_by_q


def r_colbert(questions, corpus, ids, k, cfg):
    from pylate import indexes, models, retrieve
    c = cfg["colbert"]
    model = models.ColBERT(model_name_or_path=c["model"])
    index = indexes.PLAID(index_name="fbrag_colbert", override=True)
    demb = model.encode([d["text"] for d in corpus], batch_size=32,
                        is_query=False, show_progress_bar=True)
    index.add_documents(documents_ids=ids, documents_embeddings=demb)
    retriever = retrieve.ColBERT(index=index)
    qemb = model.encode([q["question"] for q in questions], batch_size=32,
                        is_query=True, show_progress_bar=True)
    res = retriever.retrieve(queries_embeddings=qemb, k=k)
    out = {q["qid"]: [r["id"] for r in res[i]] for i, q in enumerate(questions)}
    return out, None


def r_hybrid(questions, ids, k, cfg, sparse_scores, dense_ranks):
    """Reciprocal Rank Fusion of bm25 scores and bge ranked lists."""
    rrf_k = cfg["hybrid"]["rrf_k"]
    out = {}
    for q in questions:
        qid = q["qid"]
        fused = defaultdict(float)
        # bm25 ranks from scores
        # mask the sparse side too: the dense list arrives already masked, so
        # fusing it with an unmasked BM25 order would re-admit exactly the
        # documents the mask exists to exclude
        ss = sparse_scores[qid]
        m = mask_for(qid)
        if m is not None:
            ss = np.where(m, ss, -np.inf)
        bm_order = np.argsort(ss)[::-1]
        for rank, di in enumerate(bm_order[:100]):
            if np.isfinite(ss[di]):
                fused[ids[di]] += 1.0 / (rrf_k + rank + 1)
        for rank, did in enumerate(dense_ranks[qid][:100]):
            fused[did] += 1.0 / (rrf_k + rank + 1)
        out[qid] = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]
    return out


def score(ranked, questions, qrels, k):
    by = defaultdict(lambda: {"recall": [], "hit": []})
    for q in questions:
        gold = set(qrels[q["qid"]])
        got = set(ranked[q["qid"]][:k])
        rec = len(got & gold) / max(1, len(gold))
        by[q["split"]]["recall"].append(rec)
        by[q["split"]]["hit"].append(1.0 if got & gold else 0.0)
        by["all"]["recall"].append(rec)
        by["all"]["hit"].append(1.0 if got & gold else 0.0)
    return {s: {"recall@k": round(float(np.mean(v["recall"])), 4),
                "hit@k": round(float(np.mean(v["hit"])), 4),
                "n": len(v["recall"])} for s, v in by.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-mask", action="store_true",
                    help="reproduce the pre-fix unmasked rankings")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of retrievers to run, e.g. --only bm25 bge")
    ap.add_argument("--out-suffix", default="",
                    help="write retrieval<suffix>.json instead of overwriting")
    args = ap.parse_args()
    rcfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text())
    cfg, k = rcfg["retrievers"], rcfg["top_k"]
    questions, corpus, qrels = load()
    ids = [d["doc_id"] for d in corpus]
    _MASK["on"] = not args.no_mask
    if _MASK["on"]:
        _MASK["rows"] = build_time_mask(questions, corpus)
        _MASK["qindex"] = {q["qid"]: i for i, q in enumerate(questions)}
        adm = _MASK["rows"].sum(1)
        print(f"per-question time mask ON (lag {LAG}d): "
              f"{adm.min()}-{adm.max()} admissible docs per question "
              f"(median {int(np.median(adm))} of {len(corpus)})")
    else:
        print("per-question time mask OFF -- reproduces the pre-fix rankings")
    print(f"{len(questions)} questions, {len(corpus)} docs, top_k={k}")

    ranked, skipped = {}, {}
    sparse_scores = dense_ranks = None

    def run(name, fn):
        if args.only and name not in args.only:
            return None
        print(f"== {name} ==", flush=True)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            skipped[name] = f"{type(e).__name__}: {e}"
            print(f"  !! SKIP {name}: {e}", file=sys.stderr)
            traceback.print_exc()
            return None

    res = run("bm25", lambda: r_bm25(questions, corpus, ids, k, cfg))
    if res:
        ranked["bm25"], sparse_scores = res
    res = run("dpr", lambda: r_dpr(questions, corpus, ids, k, cfg))
    if res:
        ranked["dpr"] = res[0]
    res = run("bge", lambda: r_bge(questions, corpus, ids, k, cfg))
    if res:
        ranked["bge"], _ = res
        dense_ranks = ranked["bge"]
    res = run("splade", lambda: r_splade(questions, corpus, ids, k, cfg))
    if res:
        ranked["splade"] = res[0]
    res = run("colbert", lambda: r_colbert(questions, corpus, ids, k, cfg))
    if res:
        ranked["colbert"] = res[0]
    if sparse_scores is not None and dense_ranks is not None:
        ranked["hybrid"] = run("hybrid", lambda: r_hybrid(questions, ids, k, cfg,
                                                          sparse_scores, dense_ranks))
    # The provided reference evidence is not exempt: 28 qrel contexts are dated
    # after their own question. An "oracle" that can see the future is not the
    # ceiling the paper wants it to be.
    admissible = set()
    if _MASK["on"]:
        for i, q in enumerate(questions):
            row = _MASK["rows"][i]
            admissible |= {(q["qid"], ids[j]) for j in np.nonzero(row)[0]}
    ranked["oracle"] = {
        q["qid"]: [d for d in qrels[q["qid"]]
                   if not _MASK["on"] or (q["qid"], d) in admissible][:k]
        for q in questions}

    metrics = {r: score(ranked[r], questions, qrels, k) for r in ranked}
    metrics["_skipped"] = skipped
    sfx = args.out_suffix
    (PROC / f"retrieval{sfx}.json").write_text(json.dumps(ranked))
    metrics["_mask"] = {"applied": _MASK["on"], "lag_days": LAG}
    (PROC / f"retrieval_metrics{sfx}.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== retrieval recall@%d ===" % k)
    for r in ranked:
        m = metrics[r]
        print(f"{r:8s} all recall={m['all']['recall@k']:.3f} hit={m['all']['hit@k']:.3f} "
              f"| fresh hit={m['fresh']['hit@k']:.3f} static hit={m['static']['hit@k']:.3f}")
    if skipped:
        print("SKIPPED:", skipped)


if __name__ == "__main__":
    main()
