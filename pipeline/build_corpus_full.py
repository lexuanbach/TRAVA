#!/usr/bin/env python3
"""Build the FULL (distractor-rich) TRAVA retrieval corpus.

Threat to validity this answers
-------------------------------
The main corpus (`corpus.jsonl`, 6,692 docs) pools the evidence of ONLY the 878
sampled questions, so it sits at ~1 doc : 1 question and contains essentially no
document that competes with a question's own gold evidence.  Retrieval on it
measures "can you find the planted needle in a 6.7k pool where nothing else
covers the event", which does not transfer to deployment.

This script pools the evidence of ALL raw RealTimeQA questions (~4,927), not just
the 878 sampled ones.  The extra documents are real, same-period, same-domain
news from the same weekly quiz stream -- i.e. genuine distractors rather than
synthetic noise.  The 878 evaluation questions and their qrels are UNCHANGED, so
the only thing that varies vs. the main analysis is the size and difficulty of
the index.

Temporal invariant
------------------
The pooled corpus spans 2022-2026, so an unmasked index would let a 2022 question
retrieve a 2026 document.  `retrieval_full.py` therefore applies a per-question
time mask (publish_date <= t_question + lag), which is both the honest analogue
of the paper's existing timestamp invariant and what a deployed index looks like.
We record `publish_date` for every document here so that mask is computable.
Undated documents (~0.3% of raw) are admitted ONLY if they are some evaluation
question's own gold evidence (dropping them would silently change qrels/recall);
they are never added as distractors, since their date cannot be validated.

Outputs (data/processed/):
  corpus_full.jsonl : {doc_id, text, publish_date, source_qid, url, is_gold}
  qrels_full.json   : {qid: [doc_id, ...]}  -- same gold sets, remapped to new ids
  corpus_full_stats.json
"""
import json
from collections import defaultdict
from pathlib import Path

from build_dataset import RAW, load_cfg, norm_key, parse_date

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def load_all_gcs():
    """qid -> raw search_result list, for EVERY raw question (not just sampled)."""
    idx = {}
    for f in sorted(RAW.glob("*_gcs.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            idx[d["question_id"]] = d.get("search_result", []) or []
    return idx


def main():
    cfg = load_cfg()
    ccfg = cfg["corpus"]
    lo, hi = ccfg["min_doc_chars"], ccfg["max_doc_chars"]
    lag_days = ccfg["drop_docs_after_question_date_days"]

    questions = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()]
    eval_qids = {q["qid"] for q in questions}
    qdate = {q["qid"]: parse_date(q["date"]) for q in questions}
    old_qrels = json.loads((PROC / "qrels.json").read_text())

    # Rebuild the gold key set exactly as build_dataset.py did, so we can remap
    # the SAME gold documents onto the new id space (and flag them as gold).
    gcs = load_all_gcs()
    gold_keys = defaultdict(set)          # qid -> {norm_key}
    for q in questions:
        qd = qdate[q["qid"]]
        for r in gcs.get(q["qid"], []):
            text = (r.get("text") or "").strip()
            if len(text) < lo:
                continue
            text = text[:hi]
            pub = parse_date(r.get("publish_date"))
            if pub is not None and (pub - qd).days > lag_days:
                continue                   # same timestamp invariant as the main build
            gold_keys[q["qid"]].add(norm_key(text))
    all_gold_keys = set().union(*gold_keys.values()) if gold_keys else set()

    # Pool EVERY raw question's evidence into one index.
    corpus, key2id = {}, {}
    n_seen = n_undated_dropped = 0
    for qid, results in gcs.items():
        for r in results:
            text = (r.get("text") or "").strip()
            if len(text) < lo:
                continue
            text = text[:hi]
            n_seen += 1
            k = norm_key(text)
            pub = parse_date(r.get("publish_date"))
            is_gold = k in all_gold_keys
            if pub is None and not is_gold:
                n_undated_dropped += 1     # cannot time-validate a distractor
                continue
            if k in key2id:
                if is_gold:
                    corpus[key2id[k]]["is_gold"] = True
                continue
            did = f"f{len(key2id):06d}"
            key2id[k] = did
            corpus[did] = {
                "doc_id": did,
                "text": text,
                "publish_date": pub.strftime("%Y-%m-%d") if pub else "",
                "source_qid": qid,
                "url": r.get("url", ""),
                "is_gold": is_gold,
            }

    # Remap qrels onto the new id space. Every gold doc must survive.
    qrels, missing = {}, 0
    for q in questions:
        ids = [key2id[k] for k in gold_keys[q["qid"]] if k in key2id]
        if not ids:
            missing += 1
        qrels[q["qid"]] = ids

    with (PROC / "corpus_full.jsonl").open("w") as f:
        for d in corpus.values():
            f.write(json.dumps(d) + "\n")
    (PROC / "qrels_full.json").write_text(json.dumps(qrels))

    n_gold = sum(1 for d in corpus.values() if d["is_gold"])
    old_n = sum(len(v) for v in old_qrels.values())
    new_n = sum(len(v) for v in qrels.values())
    stats = {
        "corpus_docs_full": len(corpus),
        "corpus_docs_main": 6692,
        "expansion_factor": round(len(corpus) / 6692, 2),
        "gold_docs": n_gold,
        "distractor_docs": len(corpus) - n_gold,
        "raw_questions_pooled": len(gcs),
        "eval_questions": len(questions),
        "gold_links_main": old_n,
        "gold_links_full": new_n,
        "questions_with_no_gold": missing,
        "raw_docs_seen": n_seen,
        "undated_distractors_dropped": n_undated_dropped,
        "note": "Distractors are real same-period news from unsampled RealTimeQA "
                "weeks. qrels are unchanged; only the index grows.",
    }
    (PROC / "corpus_full_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if missing:
        print(f"WARNING: {missing} questions lost their gold docs in the remap")


if __name__ == "__main__":
    main()
