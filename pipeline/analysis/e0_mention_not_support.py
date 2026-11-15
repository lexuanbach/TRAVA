#!/usr/bin/env python3
"""E0 task 4: window-matched "mention without support" passages (E1 seed list).

A passage is listed when the exact-string label computed on the SAME window the
human annotator saw says YES, but the resolved human label is not YES (NO or
PARTIAL; the strict human definition used by support_label_validation.py).

  Task B pack   (annotation/pack_2026-08-02):          window text.replace("\\n"," ")[:700]
  masked pack   (annotation/masked_support_2026-08-04): window " ".join(text.split())[:700]

Rows carry identifiers and labels only -- no question, answer or passage text.
`gold_in_generator_window_600` says whether the gold string also lies in the
600-char prefix the generators received.

Writes results/analysis/e0_mention_not_support.jsonl.
"""
from __future__ import annotations

import collections
import json

from _e0util import OUT
import e0_human_alignment as H


def main() -> int:
    qs, corpus = H.load_questions(), H.load_corpus()
    rows_b = H.packb_rows(qs, corpus)
    rows_m = H.masked_rows(qs, corpus, H.load_mhs())
    picks = ([(r, "string_w700nl") for r in rows_b] + [(r, "string_w700ws") for r in rows_m])
    out = []
    for r, win in picks:
        if r[win] and not r["human_bin"]:
            out.append({"pack": r["pack"], "item_id": r["item_id"],
                        "evidence_id": r["evidence_id"], "qid": r["qid"],
                        "doc_id": r["doc_id"], "human_label": r["human"],
                        "gold_in_generator_window_600": r["string_w600"]})
    f = OUT / "e0_mention_not_support.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec) + "\n")
    by = collections.Counter(r["pack"] for r in out)
    uniq = {p: len({(r["qid"], r["doc_id"]) for r in out if r["pack"] == p}) for p in by}
    lab = {p: dict(collections.Counter(r["human_label"] for r in out if r["pack"] == p)) for p in by}
    vis = {p: sum(r["gold_in_generator_window_600"] for r in out if r["pack"] == p) for p in by}
    print(f"rows by pack {dict(by)}; unique (qid, doc) {uniq}; human labels {lab}; "
          f"gold also in 600-char window {vis}")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
