#!/usr/bin/env python3
"""Build an exact-input (600-character) human annotation pack — review R2.

Human Task B and the masked pack showed 700-character excerpts, while generators saw the
first 600 characters of each passage. No human statistic therefore validates relation support
on the exact model input. This script re-issues the SAME passages with the excerpt the
generator actually received, so a future human pass is directly comparable, row by row,
with the existing labels.

It only writes a new pack under annotation/exact_input_600_<date>/; no existing file is
touched, and no labels are produced here.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
ANN = ROOT / "annotation"
OUT = ANN / f"exact_input_600_{date.today().isoformat()}"
DOC_CHARS = 600
sys.path.insert(0, str(ROOT / "pipeline"))


def load_corpus() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    out.setdefault(d["doc_id"], d)
    return out


def main() -> int:
    corpus = load_corpus()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"created": date.today().isoformat(), "doc_chars": DOC_CHARS,
                "purpose": "exact-input (600-char) re-annotation of the same passages the "
                           "700-char packs used; review/analysis-review.md R2",
                "sources": {}, "rows": {}}

    for pack, csv_name, fields in (
            ("pack_2026-08-02", "task_b_evidence.csv",
             ["item_id", "question", "question_date", "reference_answer", "evidence_id",
              "evidence_date", "evidence_excerpt", "supports", "freshness", "notes"]),
            ("masked_support_2026-08-04", "task_support_masked.csv",
             ["item_id", "question", "question_date", "reference_answer", "evidence_id",
              "evidence_date", "evidence_excerpt", "supports", "notes"])):
        src = ANN / pack / csv_name
        if not src.is_file():
            print(f"skip {src}: missing", file=sys.stderr)
            continue
        rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
        out_rows, changed, missing = [], 0, 0
        for r in rows:
            doc = corpus.get(r["evidence_id"])
            if doc is None:
                missing += 1
                continue
            # what the generator was shown: the raw first 600 characters (prompt keeps newlines;
            # they are collapsed here only so the text fits one CSV cell, as the old packs did)
            shown = " ".join(doc.get("text", "")[:DOC_CHARS].split())
            changed += shown != " ".join(r["evidence_excerpt"].split())
            new = {k: r.get(k, "") for k in fields}
            new["evidence_excerpt"] = shown
            new["supports"] = ""
            if "freshness" in new:
                new["freshness"] = ""
            new["notes"] = ""
            out_rows.append(new)
        dst = OUT / csv_name.replace(".csv", "_600.csv")
        with dst.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
        manifest["sources"][pack] = csv_name
        manifest["rows"][pack] = {"rows_in": len(rows), "rows_out": len(out_rows),
                                  "excerpt_text_differs_from_700_pack": changed,
                                  "evidence_ids_missing_from_corpus": missing}
        print(f"{pack}: {len(out_rows)} rows, {changed} excerpts differ from the 700-char version")

    (OUT / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "README.md").write_text(f"""# Exact-input (600-character) re-annotation pack

Created {manifest['created']} by `pipeline/analysis/r2b_build_exact_input_pack.py`.

**Why.** The existing packs showed annotators 700-character excerpts, while the generators
received the first 600 characters of each passage. Agreement statistics computed against the
700-character labels therefore measure agreement with legacy labels, not validation of support
on the exact model input (`review/analysis-review.md`, R2).

**What this is.** The same passages, same item ids, same column layout, with
`evidence_excerpt` replaced by exactly the text the generator was shown (600 characters,
whitespace collapsed to fit one CSV cell). Labels are blank and are to be filled by
annotators who have not seen the earlier labels.

**Protocol.** Use `annotation/GUIDELINES.md` unchanged, except that the excerpt truncation is
600 characters, not 700. Two independent annotators plus blind adjudication, as before. Do not
show annotators the previous labels, and do not reuse an annotator for rows they labelled
before, or record the overlap if that is unavoidable.

**Analysis after collection.** Recompute the passage-level and question-level agreement and the
human-label support interaction on these labels, and report them beside the legacy-window
numbers in `results/analysis/r2_human_alignment.json`. Only then can a claim of exact-input
validation be made.

Row counts: {json.dumps(manifest['rows'], indent=2)}
""")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
