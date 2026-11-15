#!/usr/bin/env python3
"""Ingest a returned annotation file, discarding spreadsheet damage.

Annotators work in Excel, and Excel silently rewrites what it thinks it
understands. The first returned pack came back with:

  * every `question_date` and `evidence_date` reformatted from ISO 8601 to US
    locale (2026-06-07 -> 6/7/2026), which is lossy and ambiguous;
  * trailing spaces appended to 11-18 currency-like answers ('$4' -> '$4 ').

The join keys (`item_id`, `evidence_id`) and the text the annotator actually
read (`question`, `evidence_excerpt`) came back byte-identical, so the labels are
usable -- but only if we refuse to trust any column Excel may have touched.

This script therefore takes ONLY the label columns from the returned file, joins
them to the pristine pack by key, and writes a clean per-annotator file. It
fails loudly rather than guessing if the keys do not line up.

Usage:
  python annotation/ingest_filled.py --from <dir> --annotator A
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "annotation" / "pack_2026-08-02"
TASKS = {
    "task_a_correctness.csv": (("item_id",), ("verdict", "confidence", "notes")),
    "task_b_evidence.csv": (("item_id", "evidence_id"),
                            ("supports", "freshness", "notes")),
}
VALID = {
    "verdict": {"CORRECT", "INCORRECT", "PARTIAL", "ABSTAIN", "UNCLEAR"},
    "confidence": {"HIGH", "LOW"},
    "supports": {"YES", "PARTIAL", "NO"},
    "freshness": {"CURRENT", "SUPERSEDED", "UNDATED", "NOT_APPLICABLE",
                  "ANACHRONISTIC"},
}


def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", required=True, type=Path)
    ap.add_argument("--annotator", required=True,
                    help="single letter, e.g. A")
    ap.add_argument("--pack", type=Path, default=PACK)
    args = ap.parse_args()

    src = args.src if args.src.name.startswith("pack_") else \
        next(iter(sorted(args.src.glob("pack_*"))), args.src)
    rc = 0
    for fname, (keys, labels) in TASKS.items():
        f_in, f_orig = src / fname, args.pack / fname
        if not f_in.is_file():
            print(f"skip {fname}: not in {src}")
            continue
        filled, orig = read(f_in), read(f_orig)
        print(f"=== {fname}")
        if len(filled) != len(orig):
            print(f"  FAIL row count {len(filled)} != {len(orig)}")
            rc = 1
            continue

        # keys must line up row for row; anything else means the file was
        # re-sorted or edited structurally and the labels cannot be trusted
        bad = [i for i, (a, b) in enumerate(zip(filled, orig))
               if tuple(a.get(k, "") for k in keys) != tuple(b[k] for k in keys)]
        if bad:
            print(f"  FAIL {len(bad)} rows have mismatched keys "
                  f"(first at row {bad[0] + 2})")
            rc = 1
            continue
        print(f"  keys line up on all {len(orig)} rows")

        # report, but do not import, every column the spreadsheet altered
        touched = []
        for c in orig[0]:
            if c in labels:
                continue
            n = sum(1 for a, b in zip(filled, orig)
                    if (a.get(c) or "") != (b.get(c) or ""))
            if n:
                touched.append((c, n))
        for c, n in touched:
            print(f"  discarded spreadsheet-modified column {c!r} ({n} rows)")

        # validate labels
        out_rows, empty = [], 0
        for a, b in zip(filled, orig):
            row = {k: b[k] for k in keys}
            for c in labels:
                v = (a.get(c) or "").strip()
                if c in VALID and v and v.upper() not in VALID[c]:
                    print(f"  FAIL unrecognised {c}={v!r} at {row}")
                    rc = 1
                row[c] = v.upper() if c in VALID else v
            if all(not row.get(c) for c in labels if c in VALID):
                empty += 1
            out_rows.append(row)
        print(f"  {len(out_rows) - empty}/{len(out_rows)} rows carry a label")

        dest = args.pack / f"{fname[:-4]}__annotator{args.annotator}.csv"
        with dest.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(keys) + list(labels))
            w.writeheader()
            w.writerows(out_rows)
        print(f"  wrote {dest.name}")

    if rc:
        print("\nFAILED: nothing downstream should use these files")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
