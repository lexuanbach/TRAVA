#!/usr/bin/env python3
"""How much of the shipped retrieval violates the paper's own timestamp invariant.

The Method claims that every admitted document satisfies the timestamp invariant,
and Appendix C strengthens that to "a question is never scored against a document
that could not have existed when it was asked". The builder does filter documents
against the date of the question they were COLLECTED FOR, then pools everything;
the main retrievers rank that pool for EVERY question with no per-question mask.
`retrieval_full.py` has the mask (`build_time_mask`); `retrieval.py` never calls it.

This script measures the consequence, read-only, from the shipped
`retrieval.json`, `corpus.jsonl` and `questions.jsonl`. It reports, per retriever
and per regime:

  * contexts dated more than LAG days after their question
  * questions with at least one such context
  * how far past the question the offending evidence is dated

The regime split is the point. If anachronistic evidence were spread evenly it
would be noise; if it concentrates in one arm of the treatment variable, it is a
confound aligned with the effect the paper reports, and every downstream analysis
computed from these contexts inherits it.

Outputs results/temporal_leakage_audit.json.

Usage:
    python pipeline/temporal_leakage_audit.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LAG = 3  # days; matches configs/datasets.yaml drop_docs_after_question_date_days


def parse_date(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def regime_of(q) -> str:
    """Static = before the 2025 evaluation boundary, Fresh = on/after it.

    These are calendar labels, not training-exposure labels; the audit uses the
    same split the paper's headline contrast uses.
    """
    return "fresh" if int(q["year"]) >= 2025 else "static"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lag", type=int, default=LAG)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/temporal_leakage_audit.json")
    args = ap.parse_args()

    questions = {q["qid"]: q for q in
                 (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines())}
    docs = {d["doc_id"]: d for d in
            (json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines())}
    ranked = json.loads((PROC / "retrieval.json").read_text())

    qday = {qid: parse_date(q["date"]).toordinal() for qid, q in questions.items()}
    dday = {did: (parse_date(d["publish_date"]).toordinal()
                  if d.get("publish_date") else None) for did, d in docs.items()}

    out = {
        "scope": "Read-only audit of the shipped rankings against the paper's "
                 "stated per-question timestamp invariant.",
        "lag_days": args.lag,
        "n_questions": len(questions),
        "n_docs": len(docs),
        "invariant": "publish_date <= question_date + lag_days",
        "retrievers": {},
    }

    for retriever, per_q in ranked.items():
        if not isinstance(per_q, dict) or retriever.startswith("_"):
            continue
        tot = {"contexts": 0, "bad": 0, "questions": 0, "bad_questions": 0}
        by_regime = {r: {"contexts": 0, "bad": 0, "questions": 0, "bad_questions": 0}
                     for r in ("static", "fresh")}
        overshoot = []
        for qid, dids in per_q.items():
            q = questions.get(qid)
            if q is None:
                continue
            reg = regime_of(q)
            limit = qday[qid] + args.lag
            bad = 0
            for did in dids:
                dd = dday.get(did)
                if dd is None:            # undated documents cannot violate it
                    continue
                if dd > limit:
                    bad += 1
                    overshoot.append(dd - qday[qid])
            for bucket in (tot, by_regime[reg]):
                bucket["contexts"] += len(dids)
                bucket["bad"] += bad
                bucket["questions"] += 1
                bucket["bad_questions"] += 1 if bad else 0

        def pct(a, b):
            return round(100 * a / b, 2) if b else None

        out["retrievers"][retriever] = {
            "all": {**tot,
                    "pct_contexts_anachronistic": pct(tot["bad"], tot["contexts"]),
                    "pct_questions_affected": pct(tot["bad_questions"], tot["questions"])},
            **{r: {**v,
                   "pct_contexts_anachronistic": pct(v["bad"], v["contexts"]),
                   "pct_questions_affected": pct(v["bad_questions"], v["questions"])}
               for r, v in by_regime.items()},
            "days_past_question": {
                "median": statistics.median(overshoot) if overshoot else None,
                "p90": (sorted(overshoot)[int(0.9 * len(overshoot))]
                        if overshoot else None),
                "max": max(overshoot) if overshoot else None,
            },
        }

    # The subset of questions whose HYBRID context set is entirely admissible.
    # Every cached generation for those questions was conditioned only on
    # evidence that could have existed, so any analysis restricted to them is
    # leakage-free without regenerating a single model answer.
    clean_qids = []
    hyb = ranked.get("hybrid", {})
    for qid, dids in hyb.items():
        if qid not in questions:
            continue
        limit = qday[qid] + args.lag
        if all((dday.get(d) is None or dday[d] <= limit) for d in dids):
            clean_qids.append(qid)
    out["leakage_free_subset_hybrid"] = {
        "n": len(clean_qids),
        "of": len(hyb),
        "by_regime": {r: sum(1 for q in clean_qids if regime_of(questions[q]) == r)
                      for r in ("static", "fresh")},
        "regime_totals": {r: sum(1 for q in hyb if q in questions
                                 and regime_of(questions[q]) == r)
                          for r in ("static", "fresh")},
        "qids": sorted(clean_qids),
        "use": "restrict any cached-generation analysis to these qids for a "
               "leakage-free estimate of the same estimand",
    }

    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'retriever':10s}{'ctx bad':>12s}{'%':>7s}{'q affected':>13s}{'%':>7s}"
          f"{'static %':>10s}{'fresh %':>9s}")
    for r, v in out["retrievers"].items():
        print(f"{r:10s}{v['all']['bad']:>6d}/{v['all']['contexts']:<5d}"
              f"{v['all']['pct_contexts_anachronistic']:>7.2f}"
              f"{v['all']['bad_questions']:>7d}/{v['all']['questions']:<5d}"
              f"{v['all']['pct_questions_affected']:>7.2f}"
              f"{v['static']['pct_contexts_anachronistic']:>10.2f}"
              f"{v['fresh']['pct_contexts_anachronistic']:>9.2f}")
    s = out["leakage_free_subset_hybrid"]
    print(f"\nleakage-free hybrid subset: {s['n']}/{s['of']} questions "
          f"(static {s['by_regime']['static']}/{s['regime_totals']['static']}, "
          f"fresh {s['by_regime']['fresh']}/{s['regime_totals']['fresh']})")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
