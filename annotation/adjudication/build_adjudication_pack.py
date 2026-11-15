#!/usr/bin/env python3
"""Build the third-party adjudication pack from the three annotators' labels.

Majority vote over three raters resolved every item, but nobody reviewed the
competing labels. That is consensus, not adjudication, and the difference
matters on exactly the items where raters disagreed -- which are the items the
automatic metric is most likely to be wrong about too.

WHAT GOES IN THE PACK. Only fields that passed reliability: Task A `verdict`
(alpha 0.90) and Task B `supports` (alpha 0.76). Freshness is EXCLUDED. It
scored alpha 0.47, below the protocol threshold, and adjudicating 453 contested
rows of an unreliable construct would manufacture authoritative-looking labels
on a definition the annotators demonstrably could not apply the same way. That
field needs a re-pilot with a rewritten decision rule, not a referee.

THREE THINGS THIS PACK DELIBERATELY HIDES.
  * WHO said what. Rater identity invites deference to the rater the
    coordinator trusts.
  * HOW MANY said what. This is the important one. An adjudicator shown "2 chose
    CORRECT, 1 chose PARTIAL" will ratify the majority, and adjudication then
    adds nothing that majority vote did not already give. Candidate labels are
    listed in an order fixed by a hash of the item id, so it is stable across
    rebuilds but carries no information about the vote.
  * The automatic scores. The whole point is a verdict the metric did not
    influence.

Outputs annotation/adjudication/pack/.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "annotation" / "pack_2026-08-02"
OUT = Path(__file__).resolve().parent / "pack"
# field -> (source csv stem, key columns, reliability alpha, usable)
FIELDS = [
    ("verdict", "task_a_correctness", ("item_id",)),
    ("supports", "task_b_evidence", ("item_id", "evidence_id")),
]
EXCLUDED = {"freshness": "alpha 0.47, below threshold; needs a re-pilot"}


def load(stem, keys, col):
    out = {}
    for f in sorted(glob.glob(str(SRC / f"{stem}__annotator?.csv"))):
        who = Path(f).name.split("__annotator")[1][0]
        for r in csv.DictReader(open(f, encoding="utf-8")):
            out.setdefault(tuple(r[k] for k in keys), {})[who] = r[col].strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    orig_a = {r["item_id"]: r for r in
              csv.DictReader((SRC / "task_a_correctness.csv").open(encoding="utf-8"))}
    orig_b = {(r["item_id"], r["evidence_id"]): r for r in
              csv.DictReader((SRC / "task_b_evidence.csv").open(encoding="utf-8"))}

    manifest = {"source_pack": SRC.name, "excluded_fields": EXCLUDED, "fields": {}}
    for col, stem, keys in FIELDS:
        labels = load(stem, keys, col)
        contested = {k: v for k, v in labels.items() if len(set(v.values())) > 1}
        rows = []
        for k in sorted(contested):
            votes = contested[k]
            # order candidates by a hash of the key, NOT by vote count
            cands = sorted(set(votes.values()),
                           key=lambda c: hashlib.sha1(
                               (k[0] + c).encode()).hexdigest())
            src = orig_a[k[0]] if stem == "task_a_correctness" else orig_b[k]
            row = {"item_id": k[0], "field": col,
                   "candidates": " | ".join(cands),
                   "question": src["question"],
                   "question_date": src["question_date"],
                   "reference_answer": src["reference_answer"]}
            if stem == "task_a_correctness":
                row["system_answer"] = src["system_answer"]
            else:
                row["evidence_id"] = k[1]
                row["evidence_date"] = src["evidence_date"]
                row["evidence_excerpt"] = src["evidence_excerpt"]
            row["adjudicated"] = ""
            row["reason"] = ""
            rows.append(row)
        dest = args.out / f"adjudicate_{col}.csv"
        cols = list(rows[0]) if rows else []
        with dest.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        manifest["fields"][col] = {
            "file": dest.name, "n_contested": len(rows),
            "n_total": len(labels),
            "pct_contested": round(100 * len(rows) / max(len(labels), 1), 1),
        }
        print(f"{col:9s} {len(rows):4d} contested of {len(labels)} -> {dest.name}")

    # the key stays with the coordinator; it is what lets us compare
    # adjudicated labels against the majority they were hidden from
    keyf = args.out / "adjudication_metadata.jsonl"
    with keyf.open("w", encoding="utf-8") as fh:
        for col, stem, keys in FIELDS:
            for k, votes in sorted(load(stem, keys, col).items()):
                if len(set(votes.values())) > 1:
                    fh.write(json.dumps({"item_id": k[0],
                                         "evidence_id": k[1] if len(k) > 1 else "",
                                         "field": col, "votes": votes}) + "\n")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {keyf.name} (COORDINATOR ONLY) and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
