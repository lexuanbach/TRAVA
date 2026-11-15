#!/usr/bin/env python3
"""Do the headline retrieval results survive when no context is anachronistic?

The main index was not time-masked, so 11.66% of Hybrid contexts and 34.28% of
questions were conditioned on evidence dated after the question, concentrated in
the Static arm (19.75% vs 4.77%). `retrieval.py` now applies the mask, but the
cached generations were produced under the old rankings and we cannot re-query
nine models offline to replace them.

We do not have to. Every cached generation records the exact `doc_ids` it saw.
So for each (model, question) we can decide from the cache itself whether that
answer was conditioned on admissible evidence, and restrict the estimand to the
model-questions where it was. That is a leakage-free estimate of the same
quantity, on real cached answers, with no regeneration.

What this buys and what it does not:

  BUYS   an unbiased-by-anachronism estimate of the retrieval gain by regime,
         and a direct before/after comparison on the SAME models
  DOES NOT BUY  the full-sample estimate. The clean subset is SELECTED: Static
         questions are far likelier to be dropped than Fresh ones, so the clean
         Static arm is thinner and may differ from the full Static arm in ways
         other than admissibility. We report the selection rates beside every
         number rather than burying them, and we report the full-sample estimate
         next to the clean one so the reader sees the move, not just the result.

Outputs results/leakage_free_reanalysis.json.

Usage:
    python pipeline/leakage_free_reanalysis.py
    python pipeline/leakage_free_reanalysis.py --condition hybrid --lag 3
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import token_f1  # noqa: E402


def pdate(s):
    return date.fromisoformat(str(s)[:10])


def boot_ci(vals, n=2000, seed=0):
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    xs = sorted(statistics.fmean([vals[rng.randrange(len(vals))] for _ in vals])
                for _ in range(n))
    return [round(xs[int(0.025 * n)], 3), round(xs[int(0.975 * n)], 3)]


def paired_p(diffs, n=2000, seed=0):
    """Two-sided paired bootstrap p for mean(diffs) == 0."""
    if len(diffs) < 2:
        return None
    rng = random.Random(seed)
    m = statistics.fmean(diffs)
    centred = [d - m for d in diffs]
    hits = sum(1 for _ in range(n)
               if abs(statistics.fmean([centred[rng.randrange(len(centred))]
                                        for _ in centred])) >= abs(m))
    return round((hits + 1) / (n + 1), 4)


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", default="hybrid")
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/leakage_free_reanalysis.json")
    args = ap.parse_args()

    questions = {q["qid"]: q for q in load_jsonl(PROC / "questions.jsonl")}
    docs = {d["doc_id"]: d for d in load_jsonl(PROC / "corpus.jsonl")}
    qday = {q: pdate(v["date"]).toordinal() for q, v in questions.items()}
    dday = {d: (pdate(v["publish_date"]).toordinal() if v.get("publish_date") else None)
            for d, v in docs.items()}

    def admissible(qid, doc_ids):
        lim = qday[qid] + args.lag
        return all(dday.get(d) is None or dday[d] <= lim for d in (doc_ids or []))

    def regime(qid):
        return "fresh" if int(questions[qid]["year"]) >= 2025 else "static"

    out = {
        "scope": "Retrieval gain by regime, estimated on all cached "
                 "model-questions and on the subset whose retrieved contexts "
                 "were all temporally admissible.",
        "condition": args.condition, "lag_days": args.lag,
        "estimand": "mean token-F1(RAG) - token-F1(closed book), per question, "
                    "paired within model",
        "models": {},
    }

    for mdir in sorted(PROC.glob("generations_*")):
        cb_p, rag_p = mdir / "closed_book.jsonl", mdir / f"{args.condition}.jsonl"
        if not (cb_p.is_file() and rag_p.is_file()):
            continue
        cb = {r["qid"]: r for r in load_jsonl(cb_p)}
        rag = {r["qid"]: r for r in load_jsonl(rag_p)}
        shared = [q for q in rag if q in cb and q in questions]
        if len(shared) < 20:
            continue

        rows = []
        for qid in shared:
            gold = questions[qid]["gold_answer"]
            g = token_f1(rag[qid].get("prediction", ""), gold) - \
                token_f1(cb[qid].get("prediction", ""), gold)
            rows.append((qid, regime(qid), g, admissible(qid, rag[qid].get("doc_ids"))))

        entry = {"model": rag[shared[0]].get("model"), "n_questions": len(rows)}
        for scope_name, keep in (("all", lambda r: True),
                                 ("leakage_free", lambda r: r[3])):
            block = {}
            for reg in ("static", "fresh"):
                v = [r[2] for r in rows if r[1] == reg and keep(r)]
                block[reg] = {
                    "n": len(v),
                    "mean_gain_f1": round(statistics.fmean(v), 3) if v else None,
                    "ci95": boot_ci(v),
                    "p_paired": paired_p(v),
                }
            s = block["static"]["mean_gain_f1"]
            f = block["fresh"]["mean_gain_f1"]
            block["fresh_minus_static"] = (round(f - s, 3)
                                           if s is not None and f is not None else None)
            entry[scope_name] = block
        entry["retention"] = {
            reg: {"kept": sum(1 for r in rows if r[1] == reg and r[3]),
                  "of": sum(1 for r in rows if r[1] == reg)}
            for reg in ("static", "fresh")}
        out["models"][mdir.name.replace("generations_", "")] = entry

    # Does the headline direction survive, model by model?
    # the harm-on-Static claim specifically: which models lose it?
    harm_lost = [m for m, e in out["models"].items()
                 if (e["all"]["static"]["mean_gain_f1"] or 0) < 0
                 and (e["leakage_free"]["static"]["mean_gain_f1"] or 0) >= 0]
    out["static_harm_disappears_when_leakage_removed"] = sorted(harm_lost)

    survives, flipped = [], []
    for m, e in out["models"].items():
        a, c = e.get("all", {}).get("fresh_minus_static"), \
            e.get("leakage_free", {}).get("fresh_minus_static")
        if a is None or c is None:
            continue
        (survives if (a > 0) == (c > 0) else flipped).append(m)
    out["direction_of_fresh_minus_static"] = {
        "same_sign_after_restriction": sorted(survives),
        "sign_flipped_after_restriction": sorted(flipped),
        "n_models": len(survives) + len(flipped),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"condition={args.condition}   gain = token-F1(RAG) - token-F1(closed book)\n")
    print(f"{'model':14s}{'static all':>12s}{'static clean':>14s}"
          f"{'fresh all':>11s}{'fresh clean':>13s}{'kept S/F':>14s}")
    for m, e in out["models"].items():
        a, c, r = e["all"], e["leakage_free"], e["retention"]
        def g(b, k):
            v = b[k]["mean_gain_f1"]
            return f"{v:+.2f}" if v is not None else "  --"
        kept = (f"{r['static']['kept']}/{r['static']['of']} "
                f"{r['fresh']['kept']}/{r['fresh']['of']}")
        print(f"{m:14s}{g(a,'static'):>12s}{g(c,'static'):>14s}"
              f"{g(a,'fresh'):>11s}{g(c,'fresh'):>13s}{kept:>16s}")
    if harm_lost:
        print(f"\nmodels where retrieval HARMED Static in the full sample but "
              f"does not on admissible evidence: {sorted(harm_lost)}")
    d = out["direction_of_fresh_minus_static"]
    print(f"\nfresh-minus-static keeps its sign on {len(d['same_sign_after_restriction'])}"
          f"/{d['n_models']} models; flips on {d['sign_flipped_after_restriction']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
