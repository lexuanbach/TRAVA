#!/usr/bin/env python3
"""Build a second temporal source: FreshQA (Vu et al., 2024).

The review asked us to replicate on a second temporal QA source. FreshQA is the
right one to pick precisely because it does NOT construct freshness the way we
do, so agreement between the two is evidence about the phenomenon rather than
about one dataset's idiosyncrasies:

  RealTimeQA (ours)   freshness = the question's PUBLICATION DATE. A weekly news
                      quiz; every question is new when written.
  FreshQA             freshness = the answer's VOLATILITY and its EFFECTIVE
                      YEAR. Questions are hand-written and static; what changes
                      is whether the correct answer has moved since pretraining.

FreshQA also differs in ways that matter for scoring: it ships up to ten
acceptable answers per question, so alias handling is built in rather than
approximated, and it marks 149 FALSE-PREMISE questions, which we exclude from
the main contrast and report separately (a false premise is a different failure
mode from a stale fact).

We keep the split at the same 2025 boundary the rest of the paper uses:
`effective_year` before 2025 is Static, 2025 or later is Fresh. There is no
retrieval corpus, so this source tests the CLOSED-BOOK claim only -- which is
the claim the paper's headline rests on.

Input:  the FreshQA CSV export (--csv), default /tmp/freshqa.csv
Output: data/processed/freshqa.jsonl
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_YEAR = 2025


def parse_year(s: str):
    """`effective_year` is a year, or a phrase like 'before 2022'."""
    s = (s or "").strip()
    if not s:
        return None
    m = re.search(r"(\d{4})", s)
    if not m:
        return None
    y = int(m.group(1))
    return y - 1 if s.lower().startswith("before") else y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("/tmp/freshqa.csv"))
    ap.add_argument("--output", type=Path,
                    default=ROOT / "data/processed/freshqa.jsonl")
    args = ap.parse_args()

    txt = args.csv.read_text(encoding="utf-8")
    i = txt.index("id,split,question")          # skip the sheet's banner rows
    rows = list(csv.DictReader(io.StringIO(txt[i:])))

    out = []
    dropped = collections.Counter()
    for r in rows:
        answers = [r.get(f"answer_{k}", "").strip() for k in range(10)]
        answers = [a for a in answers if a]
        year = parse_year(r.get("effective_year", ""))
        fp = str(r.get("false_premise", "")).strip().lower() in ("true", "1", "yes")
        if not answers:
            dropped["no_answer"] += 1
            continue
        if year is None:
            dropped["no_effective_year"] += 1
            continue
        if not (r.get("question") or "").strip():
            dropped["no_question"] += 1
            continue
        out.append({
            "qid": f"freshqa_{r['id']}",
            "question": r["question"].strip(),
            # FreshQA ships several acceptable answers; the first is canonical
            # and the rest are aliases we score against with a max.
            "gold_answer": answers[0],
            "gold_answers": answers,
            "effective_year": year,
            "year": year,
            "slice": "fresh" if year >= BOUNDARY_YEAR else "static",
            "fact_type": (r.get("fact_type") or "").strip(),
            "false_premise": fp,
            "num_hops": (r.get("num_hops") or "").strip(),
            "split": (r.get("split") or "").strip(),
            "source": "freshqa",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for d in out:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    ft = collections.Counter(d["fact_type"] for d in out)
    sl = collections.Counter(d["slice"] for d in out)
    yr = collections.Counter(d["year"] for d in out)
    n_fp = sum(1 for d in out if d["false_premise"])
    n_ans = collections.Counter(len(d["gold_answers"]) for d in out)
    print(f"wrote {len(out)} questions to {args.output}")
    print(f"  dropped: {dict(dropped) or 'none'}")
    print(f"  slice:       {dict(sl)}  (boundary {BOUNDARY_YEAR})")
    print(f"  fact_type:   {dict(ft)}")
    print(f"  false premise: {n_fp} (excluded from the main contrast)")
    print(f"  effective_year: {dict(sorted(yr.items()))}")
    print(f"  answers/question: {dict(sorted(n_ans.items()))}")
    print(f"  usable for the main contrast: "
          f"{sum(1 for d in out if not d['false_premise'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
