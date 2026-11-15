#!/usr/bin/env python3
"""Paired retrieval-minus-closed-book gains for EVERY retriever on the primary.

The manuscript previously wrote "paired gains of +6 to +9 on Static and +11 to +14
on Fresh (all p<0.01)" while robustness.json only ever tested bm25/hybrid/oracle,
and both stated ranges excluded real values (DPR +4.6, BM25 Fresh +10.6).  This
computes all six with a paired bootstrap so the prose can quote the real range.

Writes results/rcb_all.json.
"""
import json
from pathlib import Path

import numpy as np

from metrics import token_f1

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RNG = np.random.default_rng(0)
CONDS = ["bm25", "dpr", "splade", "colbert", "bge", "hybrid", "oracle"]


def paired_boot(d, n=2000):
    d = np.asarray(d, float)
    m = d.mean()
    bs = np.array([RNG.choice(d, len(d)).mean() for _ in range(n)])
    return {"gain": round(float(m), 2),
            "ci95": [round(float(np.percentile(bs, 2.5)), 2),
                     round(float(np.percentile(bs, 97.5)), 2)],
            "one_sided_p": round(float((bs <= 0).mean()), 4), "n": len(d)}


def main():
    Q = {q["qid"]: q for q in
         (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines())}
    gd = PROC / "generations_qwen32b"

    def load(c):
        f = gd / f"{c}.jsonl"
        if not f.exists():
            return None
        return {r["qid"]: token_f1(r["prediction"], Q[r["qid"]]["gold_answer"]) * 100
                for r in (json.loads(l) for l in f.read_text().splitlines())
                if r["qid"] in Q}

    cb = load("closed_book")
    out = {"generator": "Qwen2.5-32B", "boundary": 2025}
    for c in CONDS:
        r = load(c)
        if not r:
            continue
        row = {}
        for slice_ in ("static", "fresh"):
            qs = [q for q in Q if Q[q]["year"] < 2025] if slice_ == "static" \
                else [q for q in Q if Q[q]["year"] >= 2025]
            qs = [q for q in qs if q in cb and q in r]
            row[slice_] = paired_boot([r[q] - cb[q] for q in qs])
        out[c] = row
    (RES / "rcb_all.json").write_text(json.dumps(out, indent=2))

    print(f"{'retriever':10s} {'Static gain':>22s}  {'Fresh gain':>22s}")
    for c in CONDS:
        if c not in out:
            continue
        s, f = out[c]["static"], out[c]["fresh"]
        print(f"{c:10s} {s['gain']:+6.2f} [{s['ci95'][0]:+5.1f},{s['ci95'][1]:+5.1f}] p={s['one_sided_p']:.3f}  "
              f"{f['gain']:+6.2f} [{f['ci95'][0]:+5.1f},{f['ci95'][1]:+5.1f}] p={f['one_sided_p']:.3f}")
    sg = [out[c]["static"]["gain"] for c in CONDS if c in out and c != "oracle"]
    fg = [out[c]["fresh"]["gain"] for c in CONDS if c in out and c != "oracle"]
    pm = max(max(out[c]["static"]["one_sided_p"], out[c]["fresh"]["one_sided_p"])
             for c in CONDS if c in out and c != "oracle")
    print(f"\nRETRIEVER RANGE  Static {min(sg):+.1f}..{max(sg):+.1f}   "
          f"Fresh {min(fg):+.1f}..{max(fg):+.1f}   max p = {pm:.4f}")


if __name__ == "__main__":
    main()
