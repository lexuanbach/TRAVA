#!/usr/bin/env python3
"""Build the freshness re-pilot pack.

The first freshness pass scored Krippendorff alpha 0.471 over three raters,
below the 0.6 threshold, so no human freshness or staleness number is reportable.
Diagnosing WHY changed the design twice.

FINDING 1 -- most rows asked an undefined question. Agreement depends almost
entirely on whether the evidence is about the question at all:

    evidence supports the answer (YES/PARTIAL)   alpha 0.569   n=320
    evidence is off-topic (NO)                   alpha 0.293   n=572

Nearly two thirds of the rows asked "is this document current for this
question's date?" about a document with no bearing on the question -- a
retrieved article about Nichelle Nichols filed against a question about Cindy
Williams. There is no correct answer, so raters split between "it cannot go
stale" and "it is old". The re-pilot therefore only asks freshness where the
evidence is on topic, and gives an explicit escape hatch for the rest so the
case is recorded rather than guessed.

FINDING 2 -- the SUPERSEDED/NOT_APPLICABLE rule was metaphysical. "Could this
fact change?" invites a judgement about the world. The v2 rule asks instead
about the sentence as written: does it assert an ongoing state, or a completed
dated event? That is checkable from the text, which is what a coding rule has to
be. Even on-topic rows only reached 0.569, so scoping alone is not enough.

The pack is deliberately SMALL. This is a pilot to test whether the rewritten
rule clears 0.6, not a re-run of the study; collecting 900 rows again under an
unvalidated rule would waste three people's time.

Outputs annotation/freshness_repilot/pack/.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "annotation" / "pack_2026-08-02"
OUT = Path(__file__).resolve().parent / "pack"


def whole_sentences(text: str, cap: int = 700) -> str:
    """Trim to the last complete sentence at or before `cap`.

    The v2 rule classifies a SENTENCE by its grammatical aspect, and the source
    pack cannot support that: 98.4% of its excerpts sit exactly at the 700-char
    cap, only 4.6% end at a sentence boundary, and 77.3% are cut mid-word. A
    truncated sentence can also read as the wrong aspect -- "...opened its third
    plant in 2019 and is now" looks like a STATE claim when the claim is an
    EVENT -- which would inject exactly the noise this re-pilot exists to remove.

    We trim rather than extend, so the annotator still sees a SUBSET of what the
    generator was shown and no label describes evidence the model never had.
    Costs a median of 65 characters; no row loses its last complete sentence.
    """
    s = text[:cap]
    ends = list(re.finditer(r"[.!?][\u201d\"\')\]]?(?=\s|$)", s))
    return s[:ends[-1].end()].rstrip() if ends else s.rstrip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150,
                    help="pilot size; enough to estimate alpha to about +/-0.08")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    orig = {(r["item_id"], r["evidence_id"]): r for r in
            csv.DictReader((SRC / "task_b_evidence.csv").open(encoding="utf-8"))}
    per = {}
    for f in sorted(glob.glob(str(SRC / "task_b_evidence__annotator?.csv"))):
        who = Path(f).name.split("__annotator")[1][0]
        for r in csv.DictReader(open(f, encoding="utf-8")):
            per.setdefault((r["item_id"], r["evidence_id"]), {})[who] = r

    def maj(votes, col):
        c = collections.Counter(v[col] for v in votes.values())
        t, n = c.most_common(1)[0]
        return t if n * 2 > len(votes) else None

    on_topic = [k for k, v in per.items() if maj(v, "supports") in ("YES", "PARTIAL")]
    contested = [k for k in on_topic
                 if len({v["freshness"] for v in per[k].values()}) > 1]
    clean = [k for k in on_topic if k not in contested]

    # Oversample the rows that split last time -- they carry the information
    # about whether the new rule actually fixes anything -- but keep a straight
    # sample too so the pilot alpha is not purely an estimate on hard cases.
    rng = random.Random(args.seed)
    n_hard = min(len(contested), int(args.n * 0.6))
    n_easy = min(len(clean), args.n - n_hard)
    chosen = sorted(rng.sample(contested, n_hard)) + sorted(rng.sample(clean, n_easy))
    rng.shuffle(chosen)

    rows = []
    for k in chosen:
        o = orig[k]
        rows.append({
            "item_id": k[0], "evidence_id": k[1],
            "question": o["question"], "question_date": o["question_date"],
            "reference_answer": o["reference_answer"],
            "evidence_date": o["evidence_date"],
            "evidence_excerpt": whole_sentences(o["evidence_excerpt"]),
            "claim_type": "", "freshness": "", "notes": "",
        })
    dest = args.out / "freshness_repilot.csv"
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    man = {
        "why": "freshness scored alpha 0.471 in the first pass; this pilot tests "
               "a rewritten decision rule before any re-run",
        "scope": "on-topic evidence only (supports majority YES or PARTIAL)",
        "diagnostic_alphas_from_pass_1": {
            "all_rows": 0.471, "on_topic_only": 0.569, "off_topic_only": 0.293},
        "n_items": len(rows),
        "composition": {"previously_contested": n_hard, "previously_unanimous": n_easy},
        "sampling_note": ("contested rows are oversampled, so the pilot alpha is "
                          "a LOWER bound on what the rule would score on a "
                          "straight sample -- that is the point, but do not "
                          "quote it as a population figure"),
        "seed": args.seed,
        "excerpt_handling": (
            "trimmed to the last COMPLETE sentence within the original "
            "700-character window. The source pack cut 98.4% of excerpts at the "
            "cap and 77.3% mid-word, which a sentence-aspect rule cannot be "
            "applied to. Trimming keeps the excerpt a subset of what the "
            "generator saw; median cost 65 characters, no row left without a "
            "complete sentence."),
        "new_column": ("claim_type is collected as a check on the rule: if two "
                       "raters agree on claim_type but differ on freshness, the "
                       "labels are wrong; if they differ on claim_type, the rule "
                       "itself is still ambiguous"),
        "target": "Krippendorff alpha >= 0.6 on freshness before any re-run",
    }
    (args.out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"wrote {dest} ({len(rows)} rows: {n_hard} previously contested, "
          f"{n_easy} previously unanimous)")
    print(f"on-topic pool was {len(on_topic)} of {len(per)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
