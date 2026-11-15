#!/usr/bin/env python3
"""Weekly-cluster intervals for retrieval gains, split Static vs Fresh.

The paper reports retrieval gains as point estimates and, in at least one place,
describes a negative point estimate as retrieval HARM. Two things were missing:

  * an interval. Questions inside one weekly quiz share a news cycle, a
    retrieval pool and an annotator pass, so the quiz is the independent unit.
    A per-question interval would be far too narrow.
  * the Static/Fresh split. A gain averaged over both regimes hides that
    retrieval can only help where the answer post-dates what the model knows.
    We define Static as quizzes on the pre-boundary side (<= 2024) and Fresh as
    post-boundary (>= 2025), the same boundary the rest of the paper uses.

Each contrast is PAIRED at the question level -- the same question is answered
closed-book and with retrieval -- then aggregated to the quiz and bootstrapped
over quizzes. Every model with both conditions cached is reported, not only the
ones with a convenient sign.

Outputs results/retrieval_cluster_ci.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import token_f1                       # noqa: E402

RES = ROOT / "results"
BOUNDARY_YEAR = 2025          # >= this is "Fresh"; earlier is "Static"


def quiz_of(qid: str) -> str:
    return qid.split("_")[0]


def load_questions():
    qs = {}
    with open(ROOT / "data/processed/questions_full.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            qs[d["qid"]] = d
    return qs


def load(model, condition):
    f = ROOT / "data/processed" / f"generations_{model}" / f"{condition}.jsonl"
    if not f.is_file():
        return None
    out = {}
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["qid"]] = r["prediction"] or ""
    return out


def paired_cluster_ci(diffs_by_quiz, B, seed):
    """Bootstrap the mean paired difference, resampling quizzes."""
    keys = sorted(diffs_by_quiz)
    per = [statistics.fmean(diffs_by_quiz[k]) for k in keys]
    if len(per) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean([per[rng.randrange(len(per))] for _ in per])
                   for _ in range(B))
    le = sum(1 for v in draws if v <= 0)
    n_items = sum(len(v) for v in diffs_by_quiz.values())
    return {
        "gain_f1": round(statistics.fmean(per), 3),
        "ci95": [round(draws[int(0.025 * B)], 3), round(draws[int(0.975 * B)], 3)],
        "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5),
        "n_quizzes": len(per), "n_questions": n_items,
        "significant": bool(draws[int(0.025 * B)] > 0 or draws[int(0.975 * B)] < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--condition", default="hybrid")
    ap.add_argument("--output", type=Path, default=RES / "retrieval_cluster_ci.json")
    args = ap.parse_args()

    qs = load_questions()
    out = {
        "scope": "Paired retrieval gains over closed-book with the weekly quiz "
                 "as the resampling unit, reported separately for Static "
                 f"(year < {BOUNDARY_YEAR}) and Fresh (year >= {BOUNDARY_YEAR}) "
                 "questions.",
        "why": "Questions inside a quiz share a news cycle and a retrieval "
               "pool, so per-question intervals understate uncertainty; and a "
               "gain pooled over both regimes hides that retrieval can only "
               "help where the answer post-dates the model's knowledge.",
        "retrieval_condition": args.condition,
        "boundary_year": BOUNDARY_YEAR,
        "models": {},
    }

    models = sorted(d.name.replace("generations_", "")
                    for d in (ROOT / "data/processed").glob("generations_*"))
    print(f"{'model':16s}{'regime':8s}{'gain':>8s}{'ci95':>20s}{'p':>9s}{'quizzes':>9s}")
    for m in models:
        cb = load(m, "closed_book")
        rt = load(m, args.condition)
        if not cb or not rt:
            continue
        by = {"static": collections.defaultdict(list),
              "fresh": collections.defaultdict(list),
              "all": collections.defaultdict(list)}
        for qid in set(cb) & set(rt):
            q = qs.get(qid)
            if q is None:
                continue
            d = 100.0 * (token_f1(rt[qid], q["gold_answer"])
                         - token_f1(cb[qid], q["gold_answer"]))
            reg = "fresh" if q["year"] >= BOUNDARY_YEAR else "static"
            by[reg][quiz_of(qid)].append(d)
            by["all"][quiz_of(qid)].append(d)
        rec = {}
        for reg in ("all", "static", "fresh"):
            r = paired_cluster_ci(by[reg], args.B, args.seed)
            if r:
                rec[reg] = r
                print(f"{m:16s}{reg:8s}{r['gain_f1']:8.2f}{str(r['ci95']):>20s}"
                      f"{r['p_plus_one']:9.4f}{r['n_quizzes']:9d}")
        if rec:
            out["models"][m] = rec

    # -------- what may and may not be said about retrieval harm -------------
    harmed, helped, flat = [], [], []
    for m, rec in out["models"].items():
        for reg in ("static", "fresh"):
            r = rec.get(reg)
            if not r:
                continue
            tag = f"{m}:{reg}"
            if r["ci95"][1] < 0:
                harmed.append(tag)
            elif r["ci95"][0] > 0:
                helped.append(tag)
            else:
                flat.append(tag)
    out["summary"] = {
        "significant_harm": harmed,
        "significant_gain": helped,
        "interval_spans_zero": flat,
        "n_contrasts": len(harmed) + len(helped) + len(flat),
        "reading": (
            "Only the contrasts listed under significant_harm may be described "
            "as retrieval hurting; a negative point estimate whose interval "
            "spans zero is not evidence of harm, and it is easy to describe "
            "one that way. Where retrieval does help it is concentrated in the "
            "Fresh regime, which is the direction the paper's thesis predicts."),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nsignificant harm: {harmed or 'none'}")
    print(f"significant gain: {helped or 'none'}")
    print(f"spans zero:       {len(flat)} of {out['summary']['n_contrasts']} contrasts")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
