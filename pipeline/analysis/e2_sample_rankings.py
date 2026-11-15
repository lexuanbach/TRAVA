#!/usr/bin/env python3
"""E2 Phase A: question-disjoint sample, full-corpus Hybrid rankings, bundle.

Design: results/analysis/PREREG_E2_E3_E6.md, section E2.

1. Sample. Pool = qids of data/processed/questions_full.jsonl not in
   questions.jsonl (expected 3,298). Stratum = year of `date`. ONE
   random.Random(20260917) instance; for each year in ascending order,
   rng.sample(sorted(qids of that year), 200) with Python string sort. Stored
   rows are the full questions_full rows plus
     inclusion_prob = 200 / N_year
     in_new_quiz    = quiz id (qid prefix) not among the 199 quizzes of questions.jsonl
   written in (year, qid) order.
2. Rankings over the frozen corpus_full.jsonl (26,611 docs) for the original
   878 + new 800: Hybrid top-5 with the +3-day mask applied before fusion and
   unmasked, reproducing pipeline/retrieval_full.py (imported, not modified):
     BM25 scores float32 (BM25Okapi k1/b from configs/retrieval.yaml),
     BGE sims (qemb @ demb.T).astype(float32), embeddings cached to .npy,
     per retriever topk_masked(depth = max(50, 10k)), RRF k=60, top-5.
   The unmasked arm uses an all-True mask in the same code path. The original
   878 are encoded in the same call as the original run (their own 878-row
   list); the new 800 are encoded as a separate list.
   Validation: original 878 masked == retrieval_full.json['hybrid'].
3. Stats (no F1) -> results/analysis/e2_sample_stats.json.
4. Bundle -> data/processed/analysis/e2_bundle/.
No generation; no model outputs are read.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time

import numpy as np
import yaml

from retr_cache import (OUTD, PIPE, PROC, RES, ROOT, bge_encode_cached, by_regime,
                        date_ord, exposure_block, read_jsonl, regime, sha, token_audit)

import retrieval_full as RF  # noqa: E402  (imported, not modified)
from clusters import quiz_id  # noqa: E402
from metrics import gold_in_text  # noqa: E402
from retrieval import tok  # noqa: E402

SEED = 20260917
PER_YEAR = 200
EXPECTED_POOL = {2022: 558, 2023: 1093, 2024: 1095, 2025: 552}
ARMS = ("hybrid_full_masked", "hybrid_full_unmasked")
VIS = 600


def year(q):
    return int(str(q["date"])[:4])


def draw_sample(orig, full):
    orig_ids = {q["qid"] for q in orig}
    orig_quizzes = {quiz_id(q["qid"]) for q in orig}
    assert len(orig_quizzes) == 199, len(orig_quizzes)
    pool = [q for q in full if q["qid"] not in orig_ids]
    by_year = collections.defaultdict(list)
    for q in pool:
        by_year[year(q)].append(q)
    counts = {y: len(v) for y, v in sorted(by_year.items())}
    if len(pool) != 3298 or counts != EXPECTED_POOL:
        raise SystemExit(f"pool differs from the preregistered expectation: n={len(pool)} {counts}")
    rng = random.Random(SEED)
    rows = []
    for y in sorted(by_year):
        byid = {q["qid"]: q for q in by_year[y]}
        pick = rng.sample(sorted(byid), PER_YEAR)
        for qid in sorted(pick):
            r = dict(byid[qid])
            r["inclusion_prob"] = PER_YEAR / counts[y]
            r["in_new_quiz"] = quiz_id(qid) not in orig_quizzes
            rows.append(r)
    return rows, counts, orig_quizzes


def hybrid_rrf(bm_row, bge_row, mask_row, ids, k, rrf_k):
    """The Hybrid loop of retrieval_full.main, verbatim in logic."""
    depth = max(50, k * 10)
    fused = {}
    for scores in (bm_row, bge_row):
        top = RF.topk_masked(scores, mask_row, ids, depth)
        for rank, did in enumerate(top):
            fused[did] = fused.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]


def compare(mine, ref, qids):
    bad = [q for q in qids if list(mine[q]) != list(ref[q])]
    return {"n": len(qids), "exact_ordered_match": len(qids) - len(bad),
            "set_match": sum(set(mine[q]) == set(ref[q]) for q in qids),
            "first_mismatches": bad[:10]}


def support_block(qs, rk, text_by_id, weights=None):
    n = len(qs)
    hits = [any(gold_in_text(q["gold_answer"], text_by_id[d][:VIS]) for d in rk[q["qid"]]) for q in qs]
    out = {"n": n, "n_supported": int(sum(hits)),
           "share": round(sum(hits) / n, 6) if n else None}
    if weights is not None and n:
        w = [weights[q["qid"]] for q in qs]
        out["share_ipw"] = round(sum(wi * h for wi, h in zip(w, hits)) / sum(w), 6)
    return out


def hit_block(qs, rk, gold_by_q):
    cov = [q for q in qs if gold_by_q.get(q["qid"])]
    if not cov:
        return {"n_covered": 0, "hit@5": None}
    hit = [bool(set(rk[q["qid"]][:5]) & set(gold_by_q[q["qid"]])) for q in cov]
    rec = [len(set(rk[q["qid"]][:5]) & set(gold_by_q[q["qid"]])) / len(set(gold_by_q[q["qid"]])) for q in cov]
    return {"n_covered": len(cov), "hit@5": round(float(np.mean(hit)), 6),
            "recall@5": round(float(np.mean(rec)), 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    rcfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text())
    k = rcfg["top_k"]
    rrf_k = rcfg["retrievers"]["hybrid"].get("rrf_k", 60)
    orig = read_jsonl(PROC / "questions.jsonl")
    full = read_jsonl(PROC / "questions_full.jsonl")

    # ---- 1. sample ------------------------------------------------------------
    new, pool_counts, orig_quizzes = draw_sample(orig, full)
    OUTD.mkdir(parents=True, exist_ok=True)
    new_path = OUTD / "e2_new_questions.jsonl"
    new_path.write_text("".join(json.dumps(r) + "\n" for r in new))
    print(f"sampled {len(new)} new questions; pool by year {pool_counts}", flush=True)

    # ---- 2. rankings on corpus_full ---------------------------------------------
    corpus = read_jsonl(PROC / "corpus_full.jsonl")
    ids = [d["doc_id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    union = orig + new
    qids_o = [q["qid"] for q in orig]
    print(f"corpus_full={len(corpus)} questions={len(union)}", flush=True)

    c = rcfg["retrievers"]["bm25"]
    t0 = time.time()
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([tok(t) for t in texts], k1=c["k1"], b=c["b"])
    bm25_scores = np.zeros((len(union), len(corpus)), dtype=np.float32)
    for i, q in enumerate(union):
        bm25_scores[i] = bm.get_scores(tok(q["question"]))
    print(f"bm25: {time.time() - t0:.0f}s", flush=True)

    cb = rcfg["retrievers"]["bge"]
    demb, dinfo = bge_encode_cached(texts, "corpus_full", device=args.device)
    qemb_o, qinfo_o = bge_encode_cached([cb["query_prefix"] + q["question"] for q in orig],
                                        "questions_main878", device=args.device)
    qemb_n, qinfo_n = bge_encode_cached([cb["query_prefix"] + q["question"] for q in new],
                                        "questions_e2new800", device=args.device)
    print(f"bge: {dinfo}\n{qinfo_o}\n{qinfo_n}", flush=True)
    bge_scores = np.vstack([(qemb_o @ demb.T).astype(np.float32),
                            (qemb_n @ demb.T).astype(np.float32)])

    mask = RF.build_time_mask(union, corpus)
    allowed = np.ones(len(corpus), dtype=bool)
    rankings = {a: {} for a in ARMS}
    diag = {"bm25": {}, "bge": {}}
    for i, q in enumerate(union):
        qid = q["qid"]
        rankings["hybrid_full_masked"][qid] = hybrid_rrf(bm25_scores[i], bge_scores[i], mask[i], ids, k, rrf_k)
        rankings["hybrid_full_unmasked"][qid] = hybrid_rrf(bm25_scores[i], bge_scores[i], allowed, ids, k, rrf_k)
        diag["bm25"][qid] = RF.topk_masked(bm25_scores[i], mask[i], ids, k)
        diag["bge"][qid] = RF.topk_masked(bge_scores[i], mask[i], ids, k)

    ref = json.loads((PROC / "retrieval_full.json").read_text())
    validation = {
        "original878_masked_vs_retrieval_full": {
            "hybrid": compare(rankings["hybrid_full_masked"], ref["hybrid"], qids_o),
            "bm25_top5": compare(diag["bm25"], ref["bm25"], qids_o),
            "bge_top5": compare(diag["bge"], ref["bge"], qids_o)},
        "bge_device": {"corpus": dinfo.get("actual_device"), "q_orig": qinfo_o.get("actual_device"),
                       "q_new": qinfo_n.get("actual_device")},
    }
    print(json.dumps(validation, indent=2), flush=True)
    ok = validation["original878_masked_vs_retrieval_full"]["hybrid"]["exact_ordered_match"] == len(orig)
    rk_path = OUTD / "e2_rankings.json"
    rk_path.write_text(json.dumps(rankings))

    # ---- 3. stats ------------------------------------------------------------------
    doc = {d["doc_id"]: d for d in corpus}
    doc_date = {d: v["publish_date"] for d, v in doc.items()}
    text_by_id = {d: v["text"] for d, v in doc.items()}
    qrels = json.loads((PROC / "qrels_full.json").read_text())
    src = collections.defaultdict(list)
    for d in corpus:
        src[d.get("source_qid")].append(d["doc_id"])
    ipw = {q["qid"]: 1.0 / q["inclusion_prob"] for q in new}
    samples = {"original": orig, "new": new}

    def composition(qs):
        return {"n": len(qs), "by_year": dict(sorted(collections.Counter(year(q) for q in qs).items())),
                "by_regime": dict(collections.Counter(regime(q) for q in qs)),
                "n_quizzes": len({quiz_id(q["qid"]) for q in qs}),
                "in_new_quiz": {"n_questions": sum(quiz_id(q["qid"]) not in orig_quizzes for q in qs),
                                "n_quizzes": len({quiz_id(q["qid"]) for q in qs
                                                  if quiz_id(q["qid"]) not in orig_quizzes})}}

    def subsets(name, qs):
        out = {"all": qs,
               "static": [q for q in qs if regime(q) == "static"],
               "fresh": [q for q in qs if regime(q) == "fresh"]}
        if name == "original":
            out["common_years_2022_2025"] = [q for q in qs if year(q) <= 2025]
            out["fresh_2025_only"] = [q for q in qs if year(q) == 2025]
        else:
            out["in_new_quiz"] = [q for q in qs if q["in_new_quiz"]]
            out["in_original_quizzes"] = [q for q in qs if not q["in_new_quiz"]]
        for y in sorted({year(q) for q in qs}):
            out[f"year_{y}"] = [q for q in qs if year(q) == y]
        return out

    exposure, support, hits = {}, {}, {}
    for name, qs in samples.items():
        subs = subsets(name, qs)
        w = ipw if name == "new" else None
        exposure[name] = {a: {s: exposure_block(v, rankings[a], doc_date, (0, 3),
                                                {"masked": rankings["hybrid_full_masked"],
                                                 "unmasked": rankings["hybrid_full_unmasked"]})
                              for s, v in subs.items()} for a in ARMS}
        support[name] = {a: {s: support_block(v, rankings[a], text_by_id, w) for s, v in subs.items()}
                         for a in ARMS}
        hits[name] = {a: {"qrels_full": {s: hit_block(v, rankings[a], qrels) for s, v in subs.items()},
                          "source_qid_docs": {s: hit_block(v, rankings[a], src) for s, v in subs.items()}}
                      for a in ARMS}

    def evidence(qs, is_new):
        n_src = [len(src.get(q["qid"], [])) for q in qs]
        n_src_adm = [sum(date_ord(doc_date[d]) <= date_ord(q["date"]) + 3 for d in src.get(q["qid"], []))
                     for q in qs]
        n_bearing = [sum(gold_in_text(q["gold_answer"], text_by_id[d]) for d in src.get(q["qid"], []))
                     for q in qs]
        out = {"n_questions_in_qrels_full": sum(q["qid"] in qrels for q in qs),
               "n_questions_with_source_docs_in_corpus_full": sum(x > 0 for x in n_src),
               "mean_source_docs": round(float(np.mean(n_src)), 3),
               "n_questions_with_admissible_source_doc(+3d)": sum(x > 0 for x in n_src_adm),
               "n_questions_with_answer_bearing_source_doc(fulltext)": sum(x > 0 for x in n_bearing)}
        if not is_new:
            qr_docs = [d for q in qs for d in qrels.get(q["qid"], [])]
            out["qrels_full_docs_present_in_corpus_full"] = f"{sum(d in doc for d in qr_docs)}/{len(qr_docs)}"
            out["source_docs_subset_of_qrels_full"] = sum(set(src.get(q["qid"], [])) <= set(qrels[q["qid"]])
                                                          for q in qs)
        return out

    stats = {
        "design": "results/analysis/PREREG_E2_E3_E6.md section E2 (Phase A, no generation)",
        "sampling": {"seed": SEED, "per_year": PER_YEAR, "pool_by_year": pool_counts,
                     "pool_n": sum(pool_counts.values()),
                     "procedure": "one random.Random(20260917); years ascending; "
                                  "rng.sample(sorted(qids_in_year), 200)",
                     "inclusion_prob": {y: PER_YEAR / n for y, n in pool_counts.items()}},
        "composition": {n: composition(qs) for n, qs in samples.items()},
        "validation": validation,
        "validation_all_exact": ok,
        "mask": {"lag_days": RF.LAG,
                 "mean_admissible_docs_per_question": {
                     "original": float(mask[:len(orig)].sum(1).mean()),
                     "new": float(mask[len(orig):].sum(1).mean())}},
        "definitions": {
            "exposure": "ctx_after_q_date[_plus3]: share of shown (qid, doc) contexts with publish_date > "
                        "q_date [+3]; questions_any_after_*: share of questions with >= 1 such doc",
            "changed_context": "topk_differs_vs_unmasked in the masked arm (ordered / set)",
            "support": "metrics.gold_in_text(gold_answer, text[:600]) for any shown passage "
                       "(share_ipw for the new sample weights by 1/inclusion_prob)",
            "hit@5 qrels_full": "hit against qrels_full.json (covers only the original 878)",
            "hit@5 source_qid_docs": "SUPPLEMENTARY: hit against corpus_full docs whose source_qid is the "
                                     "question (the evidence collected for it); not the qrels definition",
            "regime": "static = year(date) < 2025, fresh = year(date) >= 2025",
        },
        "exposure": exposure,
        "support_visible600": support,
        "retrieval_hits": hits,
        "evidence_in_corpus_full": {"original": evidence(orig, False), "new": evidence(new, True)},
        "sources": {p: sha(PROC / p) for p in ("questions.jsonl", "questions_full.jsonl", "corpus_full.jsonl",
                                               "qrels_full.json", "retrieval_full.json")},
        "outputs": {"e2_new_questions.jsonl": sha(new_path), "e2_rankings.json": sha(rk_path)},
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "e2_sample_stats.json").write_text(json.dumps(stats, indent=2))
    for name in samples:
        for a in ARMS:
            e, s = exposure[name][a]["all"], support[name][a]["all"]
            print(f"{name:8s} {a:22s} ctx>q {e['ctx_after_q_date']['share']:.4f} ctx>q+3 "
                  f"{e['ctx_after_q_date_plus3']['share']:.4f} q_any>q+3 "
                  f"{e['questions_any_after_q_date_plus3']['share']:.4f} changed(vs unmasked) "
                  f"{e['topk_differs_vs_unmasked']['ordered_share']:.4f} support {s['share']:.4f}", flush=True)

    if args.no_bundle:
        return
    if not ok:
        raise SystemExit("validation failed: not building the E2 bundle")

    # ---- 4. bundle ------------------------------------------------------------------
    B = OUTD / "e2_bundle"
    B.mkdir(parents=True, exist_ok=True)
    bq = [dict(q, e2_sample="original") for q in orig] + [dict(q, e2_sample="new") for q in new]
    allq = [q["qid"] for q in bq]
    assert len(set(allq)) == len(allq) == 1678
    bundle_rk = {"closed_book": {q: [] for q in allq},
                 "hybrid_full_masked": {q: rankings["hybrid_full_masked"][q] for q in allq},
                 "hybrid_full_unmasked": {q: rankings["hybrid_full_unmasked"][q] for q in allq}}
    need = {d for r in bundle_rk.values() for v in r.values() for d in v}
    subset = {d: {"doc_id": d, "publish_date": doc[d].get("publish_date", ""), "text": doc[d]["text"][:VIS]}
              for d in need}
    audit = token_audit(bq, bundle_rk, subset)
    if any(v["n_over_limit"] for v in audit["arms"].values()):
        raise SystemExit("token audit: prompts over 4096 tokens; not writing bundle")
    (B / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in bq))
    with (B / "corpus_subset.jsonl").open("w") as f:
        for d in ids:
            if d in need:
                f.write(json.dumps(subset[d]) + "\n")
    (B / "rankings.json").write_text(json.dumps(bundle_rk))
    manifest = {
        "experiment": "E2 Phase A question-disjoint replication (PREREG_E2_E3_E6.md)",
        "n_questions": len(bq), "n_original": len(orig), "n_new": len(new),
        "arms": list(bundle_rk),
        "n_docs_needed": len(need), "n_docs_written": len(subset),
        "generation_note": "run gen_hf_notrunc.py with --cond-prefix '' (arm names are full condition "
                           "names; closed_book has empty lists)",
        "sampling": stats["sampling"],
        "validation": {
            "original878_masked_equals_retrieval_full_hybrid":
                validation["original878_masked_vs_retrieval_full"]["hybrid"]["exact_ordered_match"],
            "n": len(orig), "bge_device": validation["bge_device"]},
        "token_audit": audit,
        "sources": {
            **{f"data/processed/{p}": sha(PROC / p) for p in (
                "questions.jsonl", "questions_full.jsonl", "corpus_full.jsonl", "qrels_full.json",
                "retrieval_full.json", "analysis/e2_new_questions.jsonl", "analysis/e2_rankings.json")},
            "configs/retrieval.yaml": sha(ROOT / "configs" / "retrieval.yaml"),
            "pipeline/retrieval_full.py": sha(PIPE / "retrieval_full.py"),
            "pipeline/analysis/e2_sample_rankings.py": sha(PIPE / "analysis" / "e2_sample_rankings.py"),
            "pipeline/analysis/retr_cache.py": sha(PIPE / "analysis" / "retr_cache.py"),
            "pipeline/analysis/gen_hf_notrunc.py": sha(PIPE / "analysis" / "gen_hf_notrunc.py"),
        },
        "bundle_files": {p: sha(B / p) for p in ("questions.jsonl", "corpus_subset.jsonl", "rankings.json")},
        "note": "corpus_subset stores only the first 600 characters, exactly the text the prompt shows",
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (B / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k2: manifest[k2] for k2 in ("n_docs_needed", "validation", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
