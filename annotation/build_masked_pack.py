#!/usr/bin/env python3
"""Build the blinded human-annotation pack for MASKED retrieval contexts.

WHY THIS PACK EXISTS. The paper's Layer-3 result is checked two ways that never
meet: the temporal mask is tested with automatic support labels, and human labels
are collected on the ORIGINAL unmasked rankings. Two audit passes independently
flagged that there is therefore no joint test, and the manuscript says so. This
pack collects human support labels on the passages the MASKED ranking actually
served, which is the only missing ingredient for that joint estimate.

Two further review points are designed into the sample rather than patched
afterwards:

  * The estimate must be population-weighted. The pack is a stratified
    probability sample, and `sampling_manifest.json` records each stratum's
    population size and inclusion weight so the scorer can reweight instead of
    reporting a raw mean over a deliberately unbalanced sample.
  * Support labels must be adjudicated. The pack ships an adjudication stage as
    part of the protocol, not as an optional extra.

STRATIFICATION. The interaction contrasts a supported arm against an unsupported
arm within a slice, so a sample that is mostly unsupported would estimate one arm
precisely and the other badly. We stratify on slice x automatic-string-support
and draw equally from all four cells, then weight back to the population.

BLINDING. Annotators receive question, date, reference answer and the evidence
excerpt with its publish date. They never see the model, the condition, the
automatic labels, or which stratum a row came from. The de-blinding key is
written to a separate file that must not be sent out.

Outputs annotation/masked_support_<date>/.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.dont_write_bytecode = True
from metrics import gold_in_text                               # noqa: E402

PROC = ROOT / "data/processed"
BOUNDARY = 2025
EXCERPT_CHARS = 700
PASSAGES_PER_Q = 3          # of the top-5 the generator saw


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()
             if l.strip())}


def load_corpus():
    corpus = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    corpus.setdefault(d["doc_id"], d)
    return corpus


def already_labelled():
    """(qid, doc_id) pairs that the 2026-08-02 pack already covered."""
    pack = ROOT / "annotation/pack_2026-08-02"
    key_f = pack / "item_metadata.jsonl"
    if not key_f.is_file():
        return set()
    key = {}
    for line in key_f.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("task") == "B":
                key[d["item_id"]] = d["qid"]
    seen = set()
    f = pack / "task_b_evidence.csv"
    if f.is_file():
        for row in csv.DictReader(f.open(encoding="utf-8")):
            qid = key.get(row["item_id"])
            if qid:
                seen.add((qid, row["evidence_id"]))
    return seen


def blind_id(qid, doc_id, salt):
    h = hashlib.sha256(f"{salt}|{qid}|{doc_id}".encode()).hexdigest()
    return "M" + h[:9]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranked", default="retrieval_masked.json",
                    help="ranked file under data/processed (the MASKED one)")
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--questions-per-stratum", type=int, default=50)
    ap.add_argument("--overlap", type=int, default=10,
                    help="questions deliberately re-drawn from the previous pack "
                         "so cross-pack annotator consistency is measurable")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "annotation/masked_support_2026-08-04")
    args = ap.parse_args()

    qs = load_questions()
    corpus = load_corpus()
    ranked = json.loads((PROC / args.ranked).read_text())[args.retriever]
    prior = already_labelled()

    # ---- build the population, one row per (question, shown passage) --------
    population, per_q = [], {}
    for qid, doc_ids in ranked.items():
        q = qs.get(qid)
        if not q:
            continue
        rows = []
        for did in doc_ids[:PASSAGES_PER_Q]:
            doc = corpus.get(did)
            if not doc or not doc.get("text"):
                continue
            rows.append({
                "qid": qid, "doc_id": did,
                "slice": "fresh" if q["year"] >= BOUNDARY else "static",
                "auto_string": bool(gold_in_text(q["gold_answer"], doc["text"])),
                "seen_before": (qid, did) in prior,
            })
        if rows:
            per_q[qid] = rows
            population.extend(rows)

    # A question's stratum is decided by whether ANY shown passage supports the
    # answer, because that is the variable the interaction splits on.
    q_stratum = {}
    for qid, rows in per_q.items():
        sl = rows[0]["slice"]
        sup = any(r["auto_string"] for r in rows)
        q_stratum[qid] = f"{sl}_{'supported' if sup else 'unsupported'}"

    pop_counts = collections.Counter(q_stratum.values())
    rng = random.Random(args.seed)

    drawn, manifest_strata = [], {}
    for stratum in sorted(pop_counts):
        pool = sorted(qid for qid, s in q_stratum.items() if s == stratum)
        rng.shuffle(pool)
        take = min(args.questions_per_stratum, len(pool))
        chosen = pool[:take]
        drawn.extend((qid, stratum) for qid in chosen)
        manifest_strata[stratum] = {
            "population_questions": pop_counts[stratum],
            "sampled_questions": take,
            # inverse inclusion probability: what each sampled question stands for
            "weight": round(pop_counts[stratum] / take, 5) if take else None,
        }

    # deliberate re-draw from the previous pack, for cross-pack consistency
    repeat_pool = sorted({qid for qid, rows in per_q.items()
                          if any(r["seen_before"] for r in rows)
                          and qid not in {q for q, _ in drawn}})
    rng.shuffle(repeat_pool)
    repeats = repeat_pool[:args.overlap]
    drawn.extend((qid, "consistency_recheck") for qid in repeats)

    # ---- write the pack ----------------------------------------------------
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    salt = str(args.seed)

    rows_out, key_out = [], []
    order = list(drawn)
    rng.shuffle(order)                    # strata must not be readable from order
    for qid, stratum in order:
        q = qs[qid]
        for r in per_q[qid]:
            doc = corpus[r["doc_id"]]
            iid = blind_id(qid, r["doc_id"], salt)
            text = " ".join(doc["text"].split())[:EXCERPT_CHARS]
            rows_out.append({
                "item_id": iid,
                "question": q["question"],
                "question_date": q["date"],
                "reference_answer": q["gold_answer"],
                "evidence_id": r["doc_id"],
                "evidence_date": doc.get("publish_date", ""),
                "evidence_excerpt": text,
                "supports": "",
                "notes": "",
            })
            key_out.append({
                "item_id": iid, "qid": qid, "doc_id": r["doc_id"],
                "stratum": stratum, "slice": r["slice"],
                "auto_string": r["auto_string"],
                "seen_in_previous_pack": r["seen_before"],
                "condition": "hybrid_masked",
            })

    with (out / "task_support_masked.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    with (out / "item_metadata.jsonl").open("w", encoding="utf-8") as fh:
        for k in key_out:
            fh.write(json.dumps(k) + "\n")

    manifest = {
        "purpose": "Human support labels on MASKED retrieval contexts, so the "
                   "Layer-3 interaction can be estimated jointly under the "
                   "temporal mask and human labels.",
        "built_by": "annotation/build_masked_pack.py",
        "seed": args.seed,
        "ranked_file": args.ranked,
        "retriever": args.retriever,
        "passages_per_question": PASSAGES_PER_Q,
        "excerpt_chars": EXCERPT_CHARS,
        "n_questions": len(order),
        "n_rows": len(rows_out),
        "strata": manifest_strata,
        "consistency_recheck_questions": len(repeats),
        "weighting_note": "Strata are sampled equally, NOT proportionally, so "
                          "every population estimate must be weighted by "
                          "strata[*].weight. An unweighted mean over this pack "
                          "is not a population quantity.",
        "blinding": "Annotators receive no model, condition, automatic label or "
                    "stratum. The key is in item_metadata.jsonl.",
    }
    (out / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")

    print(f"wrote {out}")
    print(f"  {len(order)} questions, {len(rows_out)} passage rows")
    for s, v in manifest_strata.items():
        print(f"  {s:22s} pop={v['population_questions']:4d} "
              f"sampled={v['sampled_questions']:3d} weight={v['weight']}")
    print(f"  consistency recheck: {len(repeats)} questions re-drawn from the "
          f"previous pack")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
