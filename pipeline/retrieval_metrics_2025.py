#!/usr/bin/env python3
"""Recompute retrieval metrics at the 2025 boundary, for BOTH indexes.

Why this exists: `questions.jsonl` carries a `split` field that was written when
`fresh_after` was 2026-01-01, so it encodes the CALENDAR split (532 static / 346
fresh). Every headline table in the paper uses the shared 2025 boundary (404/474).
Anything that scored per-slice off `q["split"]` -- retrieval.py, and the first cut
of retrieval_full.py -- therefore reported Static/Fresh retrieval quality at a
*different* boundary from the F1 it was being compared against.

This recomputes recall@5 / hit@5 by question YEAR (the ground truth) at the 2025
boundary for the main and full corpora. Writes results/retrieval_2025.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
BOUNDARY = 2025
K = 5


def score(ranked, questions, qrels):
    out = {}
    for name, sel in (("all", lambda y: True),
                      ("static", lambda y: y < BOUNDARY),
                      ("fresh", lambda y: y >= BOUNDARY)):
        rec, hit = [], []
        for q in questions:
            if not sel(q["year"]):
                continue
            gold = set(qrels.get(q["qid"], []))
            if not gold:
                continue
            got = set(ranked.get(q["qid"], [])[:K])
            rec.append(len(gold & got) / len(gold))
            hit.append(float(bool(gold & got)))
        out[name] = {"recall@5": round(float(np.mean(rec)), 4),
                     "hit@5": round(float(np.mean(hit)), 4), "n": len(rec)}
    return out


def main():
    questions = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()]
    res = {"boundary": BOUNDARY,
           "n_static": sum(q["year"] < BOUNDARY for q in questions),
           "n_fresh": sum(q["year"] >= BOUNDARY for q in questions)}

    for tag, rfile, qfile in (("main", "retrieval.json", "qrels.json"),
                              ("full", "retrieval_full.json", "qrels_full.json")):
        rp, qp = PROC / rfile, PROC / qfile
        if not rp.exists():
            continue
        ranked = json.loads(rp.read_text())
        qrels = json.loads(qp.read_text())
        res[tag] = {r: score(ranked[r], questions, qrels) for r in ranked
                    if not r.startswith("_")}

    (RES / "retrieval_2025.json").write_text(json.dumps(res, indent=2))
    print(f"boundary {BOUNDARY}: {res['n_static']} static / {res['n_fresh']} fresh\n")
    for tag in ("main", "full"):
        if tag not in res:
            continue
        print(f"=== {tag} corpus ===")
        print(f"{'retriever':12s} {'recall@5':>9s} {'hit@5':>7s} {'hit@5 S':>8s} {'hit@5 F':>8s}")
        for r, m in res[tag].items():
            print(f"{r:12s} {m['all']['recall@5']:9.3f} {m['all']['hit@5']:7.3f} "
                  f"{m['static']['hit@5']:8.3f} {m['fresh']['hit@5']:8.3f}")
        print()


if __name__ == "__main__":
    main()
