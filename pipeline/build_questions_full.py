#!/usr/bin/env python3
"""Build the FULL question set (no per-year sampling cap).

Why: the paper evaluates 878 of the 4,927 usable RealTimeQA questions. That cap
was chosen to bound an API budget -- but the closed-book cliff, which is what the
cap now costs us, needs no API and no corpus at all: closed-book means no context.
At n=404/474 our minimum detectable effect is 7.8 F1 i.i.d. and 10.6 cluster-robust,
against an observed 4.9. Removing the cap takes the slices to ~3.7k/1.2k and the MDE
to ~3.8 (~5.1 cluster-robust), which is the only *legitimate* way to sharpen the
null: it adds data rather than conditioning on a post-treatment covariate.

Filters match build_dataset.py exactly (single gold answer, >=2 choices, parseable
date, and >=1 surviving evidence document under the timestamp invariant), so the
output is a strict SUPERSET of questions.jsonl -- same definition, no sampling.

Output: data/processed/questions_full.jsonl  (+ questions_full_stats.json)
"""
import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from build_dataset import load_cfg, load_gcs_index, load_questions, norm_key, parse_date

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def main():
    cfg = load_cfg()
    ccfg = cfg["corpus"]
    lo = ccfg["min_doc_chars"]
    max_lag = timedelta(days=ccfg["drop_docs_after_question_date_days"])

    qs = load_questions(cfg)          # same filters, no sampling
    gcs = load_gcs_index()
    print(f"usable questions (pre evidence filter): {len(qs)}")

    kept, n_gold = [], {}
    for q in qs:
        ids = set()
        for r in gcs.get(q["qid"], []):
            text = (r.get("text") or "").strip()
            if len(text) < lo:
                continue
            pub = parse_date(r.get("publish_date"))
            if pub is not None and pub > q["_dt"] + max_lag:
                continue              # timestamp invariant, as in the main build
            ids.add(norm_key(text[:ccfg["max_doc_chars"]]))
        if ids:
            q = dict(q)
            q["split"] = "fresh" if q["year"] >= 2025 else "static"   # 2025 boundary
            n_gold[q["qid"]] = len(ids)
            kept.append(q)

    with (PROC / "questions_full.jsonl").open("w") as f:
        for q in kept:
            f.write(json.dumps({k: v for k, v in q.items() if k != "_dt"}) + "\n")

    old = {json.loads(l)["qid"] for l in (PROC / "questions.jsonl").read_text().splitlines()}
    by_year = Counter(q["year"] for q in kept)
    stats = {
        "n_questions_full": len(kept),
        "n_questions_sampled": len(old),
        "expansion": round(len(kept) / len(old), 2),
        "by_year": dict(sorted(by_year.items())),
        "boundary_2025": {"static": sum(1 for q in kept if q["year"] < 2025),
                          "fresh": sum(1 for q in kept if q["year"] >= 2025)},
        "is_superset_of_sampled": old.issubset({q["qid"] for q in kept}),
        "mean_gold_docs": round(sum(n_gold.values()) / len(n_gold), 2),
        "note": "Same filters as questions.jsonl; the per-year sampling cap is removed. "
                "The `split` field here IS the 2025 analysis boundary (unlike "
                "questions.jsonl, whose split field is the 2026 calendar boundary).",
    }
    (PROC / "questions_full_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if not stats["is_superset_of_sampled"]:
        print("WARNING: not a superset of the sampled set -- filters diverged")


if __name__ == "__main__":
    main()
