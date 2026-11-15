#!/usr/bin/env python3
"""Aggregate the distractor-corpus controls on the 32B primary.

Compares, on the SAME questions and model:
  main index (6,692 docs, ~1 doc:1 question, gold planted)
  full index (26,611 docs, 19,919 same-period news distractors, time-masked)
and reports the real answer-bearing oracle where available.

Skips any condition whose generation file is absent or incomplete, so it can be run
while the GPU queue is still draining. Writes results/distractor.json.
"""
import json
from pathlib import Path

import numpy as np

from metrics import token_f1

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
GEN = PROC / "generations_qwen32b"
RNG = np.random.default_rng(0)
BOUNDARY = 2025
# condition -> (label, index it was generated against)
PAIRS = [("hybrid", "hybrid_full", "Hybrid"), ("bm25", "bm25_full", "BM25")]


def load(cond, Q):
    f = GEN / f"{cond}.jsonl"
    if not f.exists():
        return None
    out = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["qid"] in Q:
            out[r["qid"]] = token_f1(r["prediction"], Q[r["qid"]]["gold_answer"]) * 100
    return out


def paired(d, n=2000):
    d = np.asarray(d, float)
    if len(d) < 2:
        return None
    bs = np.array([RNG.choice(d, len(d)).mean() for _ in range(n)])
    return {"diff": round(float(d.mean()), 2),
            "ci95": [round(float(np.percentile(bs, 2.5)), 2),
                     round(float(np.percentile(bs, 97.5)), 2)],
            "two_sided_p": round(float(2 * min((bs <= 0).mean(), (bs >= 0).mean())), 4),
            "n": len(d)}


def slice_mean(scores, Q, which):
    qs = [q for q in scores if (Q[q]["year"] < BOUNDARY) == (which == "static")]
    return round(float(np.mean([scores[q] for q in qs])), 2), len(qs)


def main():
    Q = {q["qid"]: q for q in
         (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines())}
    cb = load("closed_book", Q)
    res = {"generator": "Qwen2.5-32B", "boundary": BOUNDARY,
           "closed_book": {"static": slice_mean(cb, Q, "static")[0],
                           "fresh": slice_mean(cb, Q, "fresh")[0]},
           "rows": {}}

    for main_c, full_c, label in PAIRS:
        m, f = load(main_c, Q), load(full_c, Q)
        if not m or not f:
            print(f"skip {label}: full={'-' if not f else len(f)}")
            continue
        common = sorted(set(m) & set(f) & set(cb))
        if len(common) < 800:
            print(f"skip {label}: only {len(common)}/878 complete")
            continue
        row = {"n": len(common)}
        for sl in ("static", "fresh"):
            qs = [q for q in common if (Q[q]["year"] < BOUNDARY) == (sl == "static")]
            row[sl] = {
                "main_f1": round(float(np.mean([m[q] for q in qs])), 2),
                "full_f1": round(float(np.mean([f[q] for q in qs])), 2),
                "delta_full_minus_main": paired([f[q] - m[q] for q in qs]),
                "full_minus_closedbook": paired([f[q] - cb[q] for q in qs]),
            }
        # does the cutoff stay flat under retrieval on the harder index?
        s = [q for q in common if Q[q]["year"] < BOUNDARY]
        fr = [q for q in common if Q[q]["year"] >= BOUNDARY]
        row["cutoff_gap_full"] = round(float(np.mean([f[q] for q in s]) -
                                             np.mean([f[q] for q in fr])), 2)
        row["cutoff_gap_main"] = round(float(np.mean([m[q] for q in s]) -
                                             np.mean([m[q] for q in fr])), 2)
        res["rows"][label] = row

    orc = load("oracle_real", Q)
    if orc:
        rf = json.loads((PROC / "retrieval_full.json").read_text())
        answerable = [q for q in orc if rf["oracle_real"].get(q)]
        res["oracle_real"] = {
            "n_answerable": len(answerable),
            "pct_answerable": round(len(answerable) / len(Q) * 100, 1),
            "f1_on_answerable": round(float(np.mean([orc[q] for q in answerable])), 2),
            "closedbook_on_answerable": round(
                float(np.mean([cb[q] for q in answerable])), 2),
            "note": "A real oracle only exists for questions with >=1 answer-bearing "
                    "gold doc; the rest have no supporting evidence at all.",
        }

    (RES / "distractor.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
