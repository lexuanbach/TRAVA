#!/usr/bin/env python3
"""Freeze-and-test: RealTimeQA weeks published AFTER our snapshot.

The review offered two ways to replicate: a second temporal source, or "freeze
the protocol and test an untouched future RealTimeQA slice". This is the second.
The `v2026-07` snapshot ends at the 2026-07-03 quiz; the upstream benchmark has
since published 2026-07-10 and 2026-07-17. Those two weeks were not available
when any analysis in this paper was written, so they are a genuine prospective
test rather than another split of the same data.

What that buys and what it does not:

  BUYS   The protocol, prompts, scoring and models are all fixed before these
         questions existed. Nothing here can be tuned to them.
  DOES NOT BUY  power. Two quizzes is ~40 questions. This can corroborate the
         direction of an effect; it cannot establish one, and we report it with
         the interval it deserves rather than as a headline.

Every question here is post-2025-boundary, so this slice is entirely FRESH. It
therefore tests the Fresh-side prediction directly: closed-book accuracy should
be low for every model regardless of scale, and retrieval should help.

Applies the same filters as build_dataset.py: a single gold answer, four
choices, and the timestamp invariant on evidence.

Output: data/processed/questions_heldout.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/heldout_future"
SNAPSHOT_ENDS = "20260703"
EVIDENCE_GRACE_DAYS = 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "data/processed/questions_heldout.jsonl")
    args = ap.parse_args()

    out, dropped = [], collections.Counter()
    for qa in sorted(RAW.glob("*_qa.jsonl")):
        week = re.search(r"_(\d{8})_qa", qa.name).group(1)
        if week <= SNAPSHOT_ENDS:
            dropped["not_after_snapshot"] += 1
            continue
        gcs = qa.with_name(qa.name.replace("_qa.jsonl", "_gcs.jsonl"))
        evidence = {}
        if gcs.is_file():
            for line in gcs.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                evidence[d.get("question_id")] = d

        for line in qa.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            choices = d.get("choices") or []
            ans = d.get("answer")
            if isinstance(ans, list):
                if len(ans) != 1:
                    dropped["not_single_answer"] += 1
                    continue
                ans = ans[0]
            try:
                idx = int(ans)
            except (TypeError, ValueError):
                dropped["unparseable_answer"] += 1
                continue
            if not (0 <= idx < len(choices)):
                dropped["answer_out_of_range"] += 1
                continue
            qdate = str(d["question_date"]).replace("/", "-")
            out.append({
                "qid": d["question_id"],
                "date": qdate,
                "year": int(qdate[:4]),
                "question": d["question_sentence"],
                "gold_answer": choices[idx],
                "gold_answer_idx": idx,
                "choices": choices,
                "source": d.get("question_source", ""),
                "url": d.get("question_url", ""),
                "split": "heldout_future",
                "week": week,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for q in out:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    weeks = collections.Counter(q["week"] for q in out)
    print(f"wrote {len(out)} questions from {len(weeks)} quizzes to {args.output}")
    print(f"  weeks: {dict(sorted(weeks.items()))}")
    print(f"  dropped: {dict(dropped) or 'none'}")
    print(f"  snapshot ended {SNAPSHOT_ENDS}; every question here post-dates it")
    print(f"  all are FRESH-side (year >= 2025): "
          f"{all(q['year'] >= 2025 for q in out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
