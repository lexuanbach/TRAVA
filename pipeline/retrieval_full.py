#!/usr/bin/env python3
"""Retrieval over the FULL distractor-rich corpus, with a real oracle.

Two threats to validity this answers
------------------------------------
(1) "The corpus is the union of the questions' own gold sets at ~1 doc:1 question,
    so no retrieval conclusion transfers."  We re-run over `corpus_full.jsonl`
    (26,611 docs; 19,919 real same-period news distractors from unsampled
    RealTimeQA weeks) instead of the 6,692-doc planted pool.

(2) "Gold-Evidence is `qrels[qid][:5]` unranked -- a seventh baseline retriever,
    not an oracle (coverage 0.22, beaten by Hybrid), so the difficulty control
    controls nothing."  We add `oracle_real`: only the question's ANSWER-BEARING
    gold documents, ranked by lexical match, top-k.  That is an actual ceiling.

Temporal invariant
------------------
The pooled index spans 2022-2026, so an unmasked search would let a 2022 question
retrieve 2026 news.  Every retriever here applies a per-question mask
    admissible(d, q)  <=>  publish_date(d) <= t_q + lag   (undated gold: always)
which generalises the paper's per-question timestamp invariant to a shared index
and matches what a deployed system can actually see.

Retrievers: bm25 (sparse), bge (dense), hybrid (RRF) -- one representative of each
family, plus oracle_real and the legacy oracle for comparability.  DPR/SPLADE/
ColBERT are not re-run at this corpus size (cost); this is stated in the paper.

Outputs (data/processed/):
  retrieval_full.json          {retriever: {qid: [doc_id,...]}}
  retrieval_full_metrics.json  recall@k / hit@k per retriever x split
"""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import yaml

from build_dataset import parse_date
from metrics import gold_in_text
from retrieval import tok

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
LAG = 3  # days; matches configs/datasets.yaml drop_docs_after_question_date_days


def load():
    questions = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()]
    corpus = [json.loads(l) for l in (PROC / "corpus_full.jsonl").read_text().splitlines()]
    qrels = json.loads((PROC / "qrels_full.json").read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text())
    return questions, corpus, qrels, cfg


def build_time_mask(questions, corpus):
    """(n_q, n_d) bool: can question i legally see document j?"""
    dday = np.array([
        parse_date(d["publish_date"]).toordinal() if d["publish_date"] else -1
        for d in corpus
    ])
    qday = np.array([parse_date(q["date"]).toordinal() for q in questions])
    undated = dday < 0                                   # only gold survives undated
    mask = (dday[None, :] <= (qday[:, None] + LAG)) | undated[None, :]
    return mask


def topk_masked(scores, mask, ids, k):
    s = np.where(mask, scores, -np.inf)
    kk = min(k, int(mask.sum()))
    if kk <= 0:
        return []
    idx = np.argpartition(s, -kk)[-kk:]
    idx = idx[np.argsort(s[idx])[::-1]]
    return [ids[i] for i in idx if np.isfinite(s[i])]


def score_run(ranked, questions, qrels, k):
    out = {}
    for split in ("all", "static", "fresh"):
        qs = [q for q in questions if split == "all" or q["split"] == split]
        rec, hit = [], []
        for q in qs:
            gold = set(qrels[q["qid"]])
            got = set(ranked.get(q["qid"], [])[:k])
            if not gold:
                continue
            rec.append(len(gold & got) / len(gold))
            hit.append(float(bool(gold & got)))
        out[split] = {
            "recall@k": round(float(np.mean(rec)), 4) if rec else 0.0,
            "hit@k": round(float(np.mean(hit)), 4) if hit else 0.0,
            "n": len(rec),
        }
    return out


def main():
    questions, corpus, qrels, cfg = load()
    k = cfg["top_k"]
    ids = [d["doc_id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    print(f"corpus={len(corpus)} questions={len(questions)} k={k}")

    mask = build_time_mask(questions, corpus)
    print(f"time mask: mean admissible docs/question = {mask.sum(1).mean():.0f} "
          f"({mask.mean() * 100:.1f}% of index)")

    ranked = {}

    # ---- BM25 --------------------------------------------------------------
    from rank_bm25 import BM25Okapi
    c = cfg["retrievers"]["bm25"]
    print("bm25: indexing...")
    bm = BM25Okapi([tok(t) for t in texts], k1=c["k1"], b=c["b"])
    bm25_scores = np.zeros((len(questions), len(corpus)), dtype=np.float32)
    for i, q in enumerate(questions):
        bm25_scores[i] = bm.get_scores(tok(q["question"]))
        if i % 200 == 0:
            print(f"  bm25 {i}/{len(questions)}")
    ranked["bm25"] = {q["qid"]: topk_masked(bm25_scores[i], mask[i], ids, k)
                      for i, q in enumerate(questions)}

    # ---- BGE ---------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    c = cfg["retrievers"]["bge"]
    print("bge: encoding corpus...")
    m = SentenceTransformer(c["model"])
    demb = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    qemb = m.encode([c["query_prefix"] + q["question"] for q in questions],
                    batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    bge_scores = (qemb @ demb.T).astype(np.float32)
    ranked["bge"] = {q["qid"]: topk_masked(bge_scores[i], mask[i], ids, k)
                     for i, q in enumerate(questions)}

    # ---- Hybrid (RRF over the masked rankings) -----------------------------
    rrf_k = cfg["retrievers"]["hybrid"].get("rrf_k", 60)
    depth = max(50, k * 10)
    hyb = {}
    for i, q in enumerate(questions):
        fused = {}
        for scores in (bm25_scores[i], bge_scores[i]):
            top = topk_masked(scores, mask[i], ids, depth)
            for rank, did in enumerate(top):
                fused[did] = fused.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
        hyb[q["qid"]] = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]
    ranked["hybrid"] = hyb

    # ---- Oracles -----------------------------------------------------------
    # legacy: first k gold docs, unranked (what the paper called "Gold-Evidence")
    ranked["oracle"] = {q["qid"]: qrels[q["qid"]][:k] for q in questions}
    # real: ONLY answer-bearing gold docs, ranked by BM25 among themselves
    by_id = {d["doc_id"]: d for d in corpus}
    idpos = {d: i for i, d in enumerate(ids)}
    oracle_real, n_empty = {}, 0
    for i, q in enumerate(questions):
        bearing = [d for d in qrels[q["qid"]]
                   if gold_in_text(q["gold_answer"], by_id[d]["text"])]
        bearing.sort(key=lambda d: -bm25_scores[i][idpos[d]])
        oracle_real[q["qid"]] = bearing[:k]
        if not bearing:
            n_empty += 1
    ranked["oracle_real"] = oracle_real
    print(f"oracle_real: {n_empty}/{len(questions)} questions have NO answer-bearing "
          f"gold doc ({n_empty / len(questions) * 100:.1f}%)")

    metrics = {r: score_run(ranked[r], questions, qrels, k) for r in ranked}
    metrics["_corpus"] = {
        "docs": len(corpus),
        "mean_admissible_per_q": float(mask.sum(1).mean()),
        "oracle_real_empty": n_empty,
    }
    (PROC / "retrieval_full.json").write_text(json.dumps(ranked))
    (PROC / "retrieval_full_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n=== retrieval@{k} on the FULL corpus ({len(corpus)} docs) ===")
    for r in ranked:
        m = metrics[r]
        print(f"{r:12s} all recall={m['all']['recall@k']:.3f} hit={m['all']['hit@k']:.3f} "
              f"| static hit={m['static']['hit@k']:.3f} fresh hit={m['fresh']['hit@k']:.3f}")


if __name__ == "__main__":
    main()
