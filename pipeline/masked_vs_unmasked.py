#!/usr/bin/env python3
"""What the per-question time mask actually changed.

Fixing the mask is only half the answer. The other half is the prevalence
AND the effect. This compares the shipped (unmasked) rankings against the
regenerated masked ones, per retriever and per regime:

  * residual leakage after the fix, which must be zero
  * retrieval quality before and after -- if recall collapses, the mask has
    removed evidence the questions needed, and that is itself a finding about
    the corpus rather than about the retrievers
  * churn: how many questions received a different context set at all, since
    that is the population whose cached generations are no longer valid

The churn number is the one that governs what has to be regenerated. A question
whose top-5 is unchanged has a cached answer that is still exactly right.

Outputs results/masked_vs_unmasked.json.

Usage:
    python pipeline/masked_vs_unmasked.py
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LAG = 3


def pdate(s):
    return date.fromisoformat(str(s)[:10])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", type=Path, default=PROC / "retrieval.json")
    ap.add_argument("--after", type=Path, default=PROC / "retrieval_masked.json")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/masked_vs_unmasked.json")
    args = ap.parse_args()

    questions = {q["qid"]: q for q in
                 (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines())}
    docs = {d["doc_id"]: d for d in
            (json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines())}
    qrels = json.loads((PROC / "qrels.json").read_text())
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())

    qday = {q: pdate(v["date"]).toordinal() for q, v in questions.items()}
    dday = {d: (pdate(v["publish_date"]).toordinal() if v.get("publish_date") else None)
            for d, v in docs.items()}

    def regime(qid):
        return "fresh" if int(questions[qid]["year"]) >= 2025 else "static"

    def bad_count(qid, dids):
        lim = qday[qid] + LAG
        return sum(1 for d in dids if dday.get(d) is not None and dday[d] > lim)

    def quality(per_q, qids):
        rec, hit = [], []
        for qid in qids:
            gold = set(qrels.get(qid, []))
            got = set(per_q.get(qid, []))
            if not gold:
                continue
            rec.append(len(gold & got) / len(gold))
            hit.append(1.0 if gold & got else 0.0)
        n = len(rec)
        return {"recall@5": round(sum(rec) / n, 4) if n else None,
                "hit@5": round(sum(hit) / n, 4) if n else None, "n": n}

    out = {"scope": "Effect of applying the per-question temporal mask to the "
                    "main index, measured against the shipped rankings.",
           "lag_days": LAG, "retrievers": {}}

    common = [r for r in after if r in before and isinstance(after[r], dict)]
    for r in common:
        b, a = before[r], after[r]
        qids = [q for q in a if q in questions and q in b]
        residual = sum(bad_count(q, a[q]) for q in qids)
        prior = sum(bad_count(q, b[q]) for q in qids)
        changed = [q for q in qids if list(b[q]) != list(a[q])]
        setchanged = [q for q in qids if set(b[q]) != set(a[q])]
        entry = {
            "anachronistic_contexts_before": prior,
            "anachronistic_contexts_after": residual,
            "questions_with_changed_ranking": len(changed),
            "questions_with_changed_context_set": len(setchanged),
            "pct_questions_context_set_changed": round(100 * len(setchanged) / len(qids), 2),
            "quality_before": {g: quality(b, [q for q in qids if regime(q) == g])
                               for g in ("static", "fresh")},
            "quality_after": {g: quality(a, [q for q in qids if regime(q) == g])
                              for g in ("static", "fresh")},
            "changed_by_regime": {
                g: {"changed": sum(1 for q in setchanged if regime(q) == g),
                    "of": sum(1 for q in qids if regime(q) == g)}
                for g in ("static", "fresh")},
            "qids_context_set_changed": sorted(setchanged),
        }
        entry["quality_before"]["all"] = quality(b, qids)
        entry["quality_after"]["all"] = quality(a, qids)
        out["retrievers"][r] = entry

    hyb = out["retrievers"].get("hybrid", {})
    unchanged = sorted(set(after.get("hybrid", {})) - set(hyb.get("qids_context_set_changed", [])))
    out["cached_generations_still_valid_hybrid"] = {
        "n": len(unchanged),
        "of": len(after.get("hybrid", {})),
        "by_regime": {g: sum(1 for q in unchanged if q in questions and regime(q) == g)
                      for g in ("static", "fresh")},
        "use": "for these questions the retrieved context set is identical "
               "before and after the fix, so every cached model answer remains "
               "valid and no regeneration is required",
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'retr':8s}{'bad before':>12s}{'bad after':>11s}{'ctx changed':>13s}"
          f"{'static R@5':>22s}{'fresh R@5':>20s}")
    for r, v in out["retrievers"].items():
        qb, qa = v["quality_before"], v["quality_after"]
        print(f"{r:8s}{v['anachronistic_contexts_before']:>12d}"
              f"{v['anachronistic_contexts_after']:>11d}"
              f"{v['pct_questions_context_set_changed']:>12.1f}%"
              f"{qb['static']['recall@5']:>12.3f}->{qa['static']['recall@5']:<8.3f}"
              f"{qb['fresh']['recall@5']:>10.3f}->{qa['fresh']['recall@5']:<8.3f}")
    c = out["cached_generations_still_valid_hybrid"]
    print(f"\nhybrid questions whose contexts are unchanged: {c['n']}/{c['of']} "
          f"(static {c['by_regime']['static']}, fresh {c['by_regime']['fresh']})")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
