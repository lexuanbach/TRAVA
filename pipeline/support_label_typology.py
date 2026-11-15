#!/usr/bin/env python3
"""Where the automatic support labels disagree with humans, and whether it matters.

This computes the error typology behind the kappa values: when exact-string
containment or entailment disagrees with a human passage label, what KIND of
disagreement is it, and does the rate depend on the generator whose context the
passage came from or on the slice the question falls in?

This matters because L3 is the paper's surviving result and it is stratified by
these labels. If the labels failed differentially by slice, the interaction would
be confounded with the boundary rather than independent of it.

We classify every human-labelled passage into one of four cells:

  AGREE_SUPPORT      human says the passage supports the answer, automatic agrees
  AGREE_NO           both say it does not
  AUTO_ONLY          automatic fires, human says no        (false positive)
  HUMAN_ONLY         human says yes, automatic misses it   (false negative)

and, for the two disagreement cells, records the passage so the failure mode can
be read rather than guessed. Rates are then broken down by slice and by the
generator whose retrieved context the passage was drawn from.

Outputs results/support_label_typology.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.dont_write_bytecode = True
from metrics import gold_in_text                                # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
PACK = ROOT / "annotation/pack_2026-08-02"
BOUNDARY = 2025

PARAMS = {"qwen1_5b": 1.5, "qwen3b": 3.0, "qwen7b": 7.0, "qwen32b": 32.0,
          "qwen72b": 72.0, "mistral_large": 123.0}


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
                    corpus.setdefault(d["doc_id"], d.get("text", ""))
    return corpus


def human_passage_labels():
    """(item_id, evidence_id) -> majority human label, plus the key row."""
    key = {}
    for line in (PACK / "item_metadata.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("task") == "B":
                key[d["item_id"]] = d
    votes = collections.defaultdict(list)
    for letter in "ABC":
        f = PACK / f"task_b_evidence__annotator{letter}.csv"
        if not f.is_file():
            continue
        for row in csv.DictReader(f.open(encoding="utf-8")):
            lab = (row.get("supports") or "").strip().upper()
            if lab:
                votes[(row["item_id"], row["evidence_id"])].append(lab)
    out = {}
    for k, vs in votes.items():
        out[k] = collections.Counter(vs).most_common(1)[0][0]
    return out, key


def nli_labels():
    """(qid, doc_id) -> True when the entailment model says entailed."""
    f = RES / "nli_support_labels.jsonl"
    lab = {}
    if f.is_file():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                lab[(r["qid"], r.get("doc_id"))] = \
                    str(r.get("nli_label", "")).startswith("entail")
    return lab


def rate_ci(hits, n, B, seed):
    """Bootstrap interval for a proportion, so small cells are not overread."""
    if n == 0:
        return None
    rng = random.Random(seed)
    xs = [1] * hits + [0] * (n - hits)
    draws = sorted(statistics.fmean([xs[rng.randrange(n)] for _ in range(n)])
                   for _ in range(B))
    return [round(draws[int(0.025 * B)], 4), round(draws[int(0.975 * B)], 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--strict", action="store_true",
                    help="count human PARTIAL as unsupported (default: as supported)")
    ap.add_argument("--output", type=Path,
                    default=RES / "support_label_typology.json")
    args = ap.parse_args()

    qs = load_questions()
    corpus = load_corpus()
    human, key = human_passage_labels()
    nli = nli_labels()

    rows = []
    for (item_id, ev_id), hlab in human.items():
        k = key.get(item_id)
        if not k:
            continue
        qid = k["qid"]
        q = qs.get(qid)
        if not q:
            continue
        text = corpus.get(ev_id, "")
        if not text:
            continue
        h_sup = hlab == "YES" if args.strict else hlab in ("YES", "PARTIAL")
        rows.append({
            "qid": qid, "doc_id": ev_id,
            "model": k.get("model", ""),
            "slice": "fresh" if q["year"] >= BOUNDARY else "static",
            "human": hlab, "human_supported": h_sup,
            "string": bool(gold_in_text(q["gold_answer"], text)),
            "nli": nli.get((qid, ev_id)),
            "gold": q["gold_answer"],
            "text": text[:400],
        })

    def cell(r, auto):
        a = r[auto]
        if a is None:
            return None
        if r["human_supported"] and a:
            return "agree_support"
        if not r["human_supported"] and not a:
            return "agree_no"
        return "auto_only" if a else "human_only"

    out = {
        "scope": "Typology of disagreements between the automatic support labels "
                 "and human passage labels, broken down by slice and by the "
                 "generator whose retrieved context the passage came from.",
        "human_partial_counts_as": "unsupported" if args.strict else "supported",
        "n_labelled_passages": len(rows),
        "why": ("L3 is stratified by these labels, so a failure rate that "
                "differed by slice would confound the interaction with the "
                "boundary. These breakdowns test that directly."),
        "labels": {},
    }

    for auto in ("string", "nli"):
        usable = [r for r in rows if cell(r, auto) is not None]
        counts = collections.Counter(cell(r, auto) for r in usable)
        n = len(usable)
        disagree = counts["auto_only"] + counts["human_only"]
        rec = {
            "n_passages": n,
            "cells": dict(counts),
            "disagreement_rate": round(disagree / n, 4) if n else None,
            "disagreement_ci95": rate_ci(disagree, n, args.B, args.seed),
            "by_slice": {},
            "by_model": {},
        }
        # does the label fail differentially across the boundary?
        for sl in ("static", "fresh"):
            sub = [r for r in usable if r["slice"] == sl]
            d = sum(1 for r in sub if cell(r, auto) in ("auto_only", "human_only"))
            rec["by_slice"][sl] = {
                "n": len(sub),
                "disagreement_rate": round(d / len(sub), 4) if sub else None,
                "ci95": rate_ci(d, len(sub), args.B, args.seed),
                "cells": dict(collections.Counter(cell(r, auto) for r in sub)),
            }
        if rec["by_slice"]["static"]["disagreement_rate"] is not None and \
           rec["by_slice"]["fresh"]["disagreement_rate"] is not None:
            rec["slice_difference_pts"] = round(
                100 * (rec["by_slice"]["static"]["disagreement_rate"]
                       - rec["by_slice"]["fresh"]["disagreement_rate"]), 2)
        # and does it depend on the generator the context came from?
        for m in sorted({r["model"] for r in usable if r["model"]}):
            sub = [r for r in usable if r["model"] == m]
            d = sum(1 for r in sub if cell(r, auto) in ("auto_only", "human_only"))
            rec["by_model"][m] = {
                "n": len(sub), "params_B": PARAMS.get(m),
                "disagreement_rate": round(d / len(sub), 4) if sub else None,
            }
        # readable examples of each failure mode, so the typology can be checked
        rec["examples"] = {}
        for mode in ("auto_only", "human_only"):
            ex = [r for r in usable if cell(r, auto) == mode][:3]
            rec["examples"][mode] = [
                {"qid": r["qid"], "gold": r["gold"], "human": r["human"],
                 "excerpt": r["text"][:220]} for r in ex]
        out["labels"][auto] = rec

    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{out['n_labelled_passages']} labelled passages")
    for auto, rec in out["labels"].items():
        print(f"\n{auto}: disagreement {rec['disagreement_rate']:.3f} "
              f"{rec['disagreement_ci95']} over {rec['n_passages']} passages")
        print(f"  cells: {rec['cells']}")
        for sl, v in rec["by_slice"].items():
            print(f"  {sl:6s} n={v['n']:4d} disagreement={v['disagreement_rate']}")
        print(f"  Static minus Fresh: {rec.get('slice_difference_pts')} points")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
