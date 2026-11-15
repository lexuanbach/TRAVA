#!/usr/bin/env python3
"""E3 (admissibility window): Hybrid top-5 rankings for w in {0,1,3,7} and unmasked.

Design: results/analysis/PREREG_E2_E3_E6.md, section E3. A document is
admissible for a question iff publish_date <= q_date + w (undated documents
are admissible); the mask is applied before BM25 top-100 / BGE top-5 RRF(k=60)
fusion, exactly as pipeline/retrieval.py does. That module is imported and its
module-level `LAG` / `_MASK` state is overridden per arm; nothing is edited.

  BM25  : retrieval.r_bm25 (raw float64 scores per question)
  BGE   : the r_bge encode call (batch 64, normalized), cached to .npy;
          sims = qemb @ demb.T; dense list = retrieval.topk_ids(sims[i], k=5)
  Hybrid: retrieval.r_hybrid(questions, ids, 5, cfg, bm25_scores, dense_lists)

Validation (reported, never silently fixed):
  w3       == data/processed/retrieval_masked.json['hybrid']  (per qid)
  unmasked == data/processed/retrieval.json['hybrid']         (per qid)
  plus the bm25 / bge top-5 lists of both files as diagnostics.

Outputs
  data/processed/analysis/e3_rankings.json   {"w0","w1","w3","w7","unmasked"}: {qid: [doc_id]}
  results/analysis/e3_exposure.json          exposure / context-change stats (no F1)
  data/processed/analysis/e3_bundle/         questions.jsonl, corpus_subset.jsonl,
                                             rankings.json, manifest.json
No generation, no model outputs are read.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import yaml

from retr_cache import (OUTD, PIPE, PROC, RES, ROOT, bge_encode_cached, by_regime,
                        exposure_block, read_jsonl, sha, token_audit)

import retrieval as R  # noqa: E402  (pipeline/retrieval.py, imported, not modified)

WINDOWS = (0, 1, 3, 7)
ARMS = [f"w{w}" for w in WINDOWS] + ["unmasked"]
LAGS = (0, 1, 3, 7)


def set_mask(questions, corpus, w):
    if w is None:
        R._MASK["on"], R._MASK["rows"], R._MASK["qindex"] = False, None, None
        return None
    R.LAG = w
    rows = R.build_time_mask(questions, corpus)
    R._MASK["on"], R._MASK["rows"] = True, rows
    R._MASK["qindex"] = {q["qid"]: i for i, q in enumerate(questions)}
    return rows


def compare(mine, ref, qids):
    n_eq = sum(list(mine[q]) == list(ref[q]) for q in qids)
    n_set = sum(set(mine[q]) == set(ref[q]) for q in qids)
    bad = [q for q in qids if list(mine[q]) != list(ref[q])]
    return {"n": len(qids), "exact_ordered_match": n_eq, "set_match": n_set,
            "first_mismatches": bad[:10]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None,
                    help="BGE device; default = SentenceTransformer auto choice (as the original runs)")
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    rcfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text())
    cfg, k = rcfg["retrievers"], rcfg["top_k"]
    questions, corpus, _qrels = R.load()
    ids = [d["doc_id"] for d in corpus]
    qids = [q["qid"] for q in questions]
    print(f"{len(questions)} questions, {len(corpus)} docs, k={k}", flush=True)

    # ---- BM25 (mask off: only the raw scores are used) ---------------------
    set_mask(questions, corpus, None)
    t0 = time.time()
    bm25_top_unmasked, bm25_scores = R.r_bm25(questions, corpus, ids, k, cfg)
    print(f"bm25 scores: {time.time() - t0:.0f}s", flush=True)

    # ---- BGE (cached) ------------------------------------------------------
    c = cfg["bge"]
    demb, dinfo = bge_encode_cached([d["text"] for d in corpus], "corpus_main", device=args.device)
    qemb, qinfo = bge_encode_cached([c["query_prefix"] + q["question"] for q in questions],
                                    "questions_main878", device=args.device)
    print(f"bge corpus: {dinfo}\nbge questions: {qinfo}", flush=True)
    sims = qemb @ demb.T

    rankings, diag, admissible = {}, {}, {}
    for w in list(WINDOWS) + [None]:
        arm = "unmasked" if w is None else f"w{w}"
        rows = set_mask(questions, corpus, w)
        if rows is not None:
            adm = rows.sum(1)
            admissible[arm] = {"min": int(adm.min()), "median": float(np.median(adm)),
                               "max": int(adm.max()), "n_docs": len(corpus)}
        dense = {q["qid"]: R.topk_ids(sims[i], ids, k, q["qid"]) for i, q in enumerate(questions)}
        bm25_top = {q["qid"]: R.topk_ids(bm25_scores[q["qid"]], ids, k, q["qid"]) for q in questions}
        rankings[arm] = R.r_hybrid(questions, ids, k, cfg, bm25_scores, dense)
        diag[arm] = {"bm25": bm25_top, "bge": dense}
    set_mask(questions, corpus, None)
    R.LAG = 3  # restore module default

    # ---- validation ----------------------------------------------------------
    ref_m = json.loads((PROC / "retrieval_masked.json").read_text())
    ref_u = json.loads((PROC / "retrieval.json").read_text())
    validation = {
        "w3_vs_retrieval_masked": {
            "hybrid": compare(rankings["w3"], ref_m["hybrid"], qids),
            "bm25_top5": compare(diag["w3"]["bm25"], ref_m["bm25"], qids),
            "bge_top5": compare(diag["w3"]["bge"], ref_m["bge"], qids)},
        "unmasked_vs_retrieval": {
            "hybrid": compare(rankings["unmasked"], ref_u["hybrid"], qids),
            "bm25_top5": compare(diag["unmasked"]["bm25"], ref_u["bm25"], qids),
            "bge_top5": compare(diag["unmasked"]["bge"], ref_u["bge"], qids)},
        "bge_device": {"corpus": dinfo.get("actual_device"), "questions": qinfo.get("actual_device")},
    }
    print(json.dumps(validation, indent=2), flush=True)
    ok = (validation["w3_vs_retrieval_masked"]["hybrid"]["exact_ordered_match"] == len(qids)
          and validation["unmasked_vs_retrieval"]["hybrid"]["exact_ordered_match"] == len(qids))

    OUTD.mkdir(parents=True, exist_ok=True)
    rk_path = OUTD / "e3_rankings.json"
    rk_path.write_text(json.dumps({a: rankings[a] for a in ARMS}))

    # ---- exposure stats (no F1) ---------------------------------------------
    doc_date = {d["doc_id"]: d["publish_date"] for d in corpus}
    ref = {"w3": rankings["w3"], "unmasked": rankings["unmasked"]}
    exposure = {a: by_regime(questions, lambda qs, a=a: exposure_block(qs, rankings[a], doc_date,
                                                                       LAGS, ref))
                for a in ARMS}
    stats = {
        "design": "results/analysis/PREREG_E2_E3_E6.md section E3",
        "definitions": {
            "ctx_after_q_date_plusL": "share of shown (qid, doc) contexts with publish_date > q_date + L days",
            "questions_any_after_q_date_plusL": "share of questions with >= 1 such shown doc",
            "n_questions_lt5_contexts": "questions shown fewer than 5 documents",
            "topk_differs_vs_X": "questions whose ordered (resp. set) top-5 differs from arm X",
            "regime": "static = year(date) < 2025, fresh = year(date) >= 2025",
        },
        "validation": validation,
        "validation_all_exact": ok,
        "admissible_docs_per_question": admissible,
        "arms": exposure,
        "sources": {p: sha(PROC / p) for p in ("questions.jsonl", "corpus.jsonl",
                                               "retrieval.json", "retrieval_masked.json")},
        "rankings_sha256": sha(rk_path),
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "e3_exposure.json").write_text(json.dumps(stats, indent=2))
    for a in ARMS:
        e = exposure[a]["all"]
        print(f"{a:9s} ctx>q {e['ctx_after_q_date']['share']:.4f} ctx>q+3 "
              f"{e['ctx_after_q_date_plus3']['share']:.4f} q_any>q+3 "
              f"{e['questions_any_after_q_date_plus3']['share']:.4f} short {e['n_questions_lt5_contexts']} "
              f"diff_w3 {e['topk_differs_vs_w3']['ordered_n']} diff_unm "
              f"{e['topk_differs_vs_unmasked']['ordered_n']}", flush=True)

    if args.no_bundle:
        return
    if not ok:
        raise SystemExit("validation failed: not building the E3 bundle")

    # ---- bundle ----------------------------------------------------------------
    B = OUTD / "e3_bundle"
    B.mkdir(parents=True, exist_ok=True)
    bundle_rk = {"closed_book": {q: [] for q in qids}}
    for a in ARMS:
        bundle_rk[f"hybrid_{a}"] = {q: rankings[a][q] for q in qids}
    need = {d for r in bundle_rk.values() for v in r.values() for d in v}
    subset = {d["doc_id"]: {"doc_id": d["doc_id"], "publish_date": d.get("publish_date", ""),
                            "text": d["text"][:600]} for d in corpus if d["doc_id"] in need}
    assert len(subset) == len(need), "ranked doc ids missing from corpus.jsonl"
    audit = token_audit(questions, bundle_rk, subset)
    if any(v["n_over_limit"] for v in audit["arms"].values()):
        raise SystemExit("token audit: prompts over 4096 tokens; not writing bundle")
    (B / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    with (B / "corpus_subset.jsonl").open("w") as f:
        for d in corpus:
            if d["doc_id"] in need:
                f.write(json.dumps(subset[d["doc_id"]]) + "\n")
    (B / "rankings.json").write_text(json.dumps(bundle_rk))
    manifest = {
        "experiment": "E3 admissibility window (PREREG_E2_E3_E6.md)",
        "n_questions": len(questions),
        "arms": list(bundle_rk),
        "n_docs_needed": len(need), "n_docs_written": len(subset),
        "generation_note": "run gen_hf_notrunc.py with --cond-prefix '' (arm names are full condition "
                           "names; closed_book has empty lists)",
        "validation": {
            "w3_equals_retrieval_masked_hybrid": validation["w3_vs_retrieval_masked"]["hybrid"]["exact_ordered_match"],
            "unmasked_equals_retrieval_hybrid": validation["unmasked_vs_retrieval"]["hybrid"]["exact_ordered_match"],
            "n": len(qids),
            "bge_device": validation["bge_device"],
        },
        "exposure_summary": {a: {"ctx_after_q_date": exposure[a]["all"]["ctx_after_q_date"]["share"],
                                 "ctx_after_q_date_plus3": exposure[a]["all"]["ctx_after_q_date_plus3"]["share"],
                                 "topk_differs_vs_w3": exposure[a]["all"]["topk_differs_vs_w3"]["ordered_n"]}
                             for a in ARMS},
        "token_audit": audit,
        "sources": {
            **{f"data/processed/{p}": sha(PROC / p) for p in ("questions.jsonl", "corpus.jsonl",
                                                              "retrieval.json", "retrieval_masked.json",
                                                              "analysis/e3_rankings.json")},
            "configs/retrieval.yaml": sha(ROOT / "configs" / "retrieval.yaml"),
            "pipeline/retrieval.py": sha(PIPE / "retrieval.py"),
            "pipeline/analysis/e3_window_rankings.py": sha(PIPE / "analysis" / "e3_window_rankings.py"),
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
