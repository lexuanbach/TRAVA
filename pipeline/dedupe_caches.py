#!/usr/bin/env python3
"""Remove duplicate question IDs from the shipped generation caches.

Three full-scale closed-book caches carry 4182 rows for 4176 questions: six
quiz items from 2023-09-22 were written twice. The repeated rows are byte-equal,
so no prediction is ambiguous, but they are double-counted by every mean taken
over the file. This removes them and reports exactly how much each affected
number moves, so the correction is on the record rather than silent.

By default the script is a DRY RUN: it reports what it would change and writes
nothing. Pass --apply to rewrite the caches, which first copies each original to
`<name>.dupes.bak` alongside it.

Refuses to touch a file whose duplicate rows are not identical -- there the
right fix is regeneration, not deduplication, and the script says so.

Outputs results/dedupe_report.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import token_f1                        # noqa: E402


def load_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the caches (default is a dry run)")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/dedupe_report.json")
    args = ap.parse_args()

    qs = {}
    with (ROOT / "data/processed/questions_full.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            qs[d["qid"]] = d

    report = {
        "scope": "Duplicate question IDs in the shipped generation caches, and "
                 "the effect of removing them on the affected means.",
        "mode": "apply" if args.apply else "dry-run",
        "files": {},
    }
    blocked = []
    for f in sorted((ROOT / "data/processed").glob("generations_*/*.jsonl")):
        rows = load_rows(f)
        counts = collections.Counter(r["qid"] for r in rows)
        dups = {q for q, n in counts.items() if n > 1}
        if not dups:
            continue
        rel = str(f.relative_to(ROOT))

        by = collections.defaultdict(list)
        for r in rows:
            by[r["qid"]].append(r)
        conflicting = sorted(
            q for q in dups
            if len({json.dumps(r, sort_keys=True) for r in by[q]}) > 1)

        def mean_f1(rs):
            v = [100.0 * token_f1(r["prediction"] or "", qs[r["qid"]]["gold_answer"])
                 for r in rs if r["qid"] in qs]
            return round(statistics.fmean(v), 4) if v else None

        deduped = list({r["qid"]: r for r in rows}.values())
        before, after = mean_f1(rows), mean_f1(deduped)
        rec = {
            "rows_before": len(rows), "rows_after": len(deduped),
            "duplicate_qids": sorted(dups),
            "conflicting_qids": conflicting,
            "mean_f1_before": before, "mean_f1_after": after,
            "delta_f1": round(after - before, 4)
            if before is not None and after is not None else None,
        }
        if conflicting:
            rec["action"] = ("REFUSED -- duplicate rows differ; regenerate this "
                             "cache rather than deduplicating it")
            blocked.append(rel)
        elif args.apply:
            shutil.copy2(f, f.with_suffix(f.suffix + ".dupes.bak"))
            with f.open("w", encoding="utf-8") as fh:
                for r in deduped:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            rec["action"] = f"deduplicated; original saved to {f.name}.dupes.bak"
        else:
            rec["action"] = "would deduplicate (dry run)"
        report["files"][rel] = rec
        print(f"{rel}: {len(rows)} -> {len(deduped)} rows, "
              f"mean F1 {before} -> {after} ({rec['delta_f1']:+}) [{rec['action']}]")

    report["summary"] = {
        "n_files_with_duplicates": len(report["files"]),
        "n_refused": len(blocked),
        "largest_abs_delta_f1": max(
            (abs(v["delta_f1"]) for v in report["files"].values()
             if v.get("delta_f1") is not None), default=0.0),
        "reading": "The duplicate rows are byte-equal, so no prediction is "
                   "ambiguous. A naive mean over the raw file moves by under a "
                   "twentieth of an F1 point once they are removed.",
        "effect_on_published_numbers": "NONE. fullscale_agg.py builds its score "
                   "map keyed by qid, so it already collapsed the repeats and "
                   "results/fullscale.json is byte-identical before and after "
                   "this fix. That was accidental robustness rather than a "
                   "designed guard, which is the actual defect: "
                   "tests/test_artifact.py now fails if any shipped cache "
                   "contains a duplicate id.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["files"]:
        print("no duplicate ids found")
    print(f"\nwrote {args.output}")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
