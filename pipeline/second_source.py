#!/usr/bin/env python3
"""Does the boundary effect replicate on a second temporal source?

FreshQA (Vu et al., 2024) is a deliberately awkward replication target, which is
why it is the right one. It shares almost nothing with our benchmark except the
idea that some answers go stale:

  ours      freshness = the question's publication date; questions are weekly
            news items; one gold answer; retrieval evidence ships with it.
  FreshQA   freshness = the answer's effective year and its volatility class;
            questions are hand-written and static; up to ten acceptable
            answers; no evidence corpus at all.

If the closed-book boundary gap is a property of models meeting post-cutoff
facts, it should appear in both. If it is a property of how RealTimeQA writes
news quizzes, it should not.

Three analyses:

  1. The Static/Fresh gap per model at the same 2025 boundary, with the
     `effective_year` standing in for the question date. Scored against ALL
     acceptable answers (max token-F1), which is the alias handling our main
     benchmark can only approximate.
  2. The scale association, refit on this source with the same exact
     permutation test used in the main paper.
  3. The volatility axis FreshQA gives us and RealTimeQA cannot. This turns out
     to be partly structural rather than free: a NEVER-changing fact has no
     post-2025 effective year by definition, so all 150 of them are Static and
     no boundary gap can be computed for them at all. We use them as a NEGATIVE
     CONTROL instead -- accuracy on facts that cannot have gone stale is a
     model's baseline on this source, and the Fresh-side deficit has to be
     read against it. For fast- and slow-changing facts a gap is computable,
     with the caveat that the cells are unbalanced (fast: 16 static / 137
     fresh; slow: 112 / 36).

False-premise questions are excluded from every contrast and reported
separately: answering "no, that premise is false" is a different capability
from recalling a fact, and mixing them would confound the comparison.

Outputs results/second_source.json.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import token_f1                       # noqa: E402

RES = ROOT / "results"
PROC = ROOT / "data/processed"
BOUNDARY_YEAR = 2025

PARAMS_B = {"qwen": 0.5, "qwen1_5b": 1.5, "qwen3b": 3.0, "qwen7b": 7.0,
            "qwen32b": 32.0, "qwen72b": 72.0, "mistral_large": 123.0,
            "llama3_8b": 8.0}


def best_f1(pred, golds):
    """FreshQA ships several acceptable answers; credit the best match."""
    return max((token_f1(pred or "", g) for g in golds), default=0.0)


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in
             (PROC / "freshqa.jsonl").read_text().splitlines() if l.strip())}


def load_preds(model):
    f = PROC / f"generations_{model}" / "freshqa.jsonl"
    if not f.is_file():
        return None
    return {r["qid"]: r["prediction"] or "" for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def boot_diff(a, b, B, seed):
    """Unpaired difference of means over items (FreshQA has no quiz clusters).

    FreshQA questions are independently authored, so the item IS the unit here
    -- unlike our benchmark, where weekly quizzes induce clustering. This is a
    genuine difference between the sources, not an inconsistency.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean([a[rng.randrange(len(a))] for _ in a])
        - statistics.fmean([b[rng.randrange(len(b))] for _ in b])
        for _ in range(B))
    le = sum(1 for x in draws if x <= 0)
    return {"diff": round(statistics.fmean(a) - statistics.fmean(b), 3),
            "static_mean": round(statistics.fmean(a), 3),
            "fresh_mean": round(statistics.fmean(b), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5),
            "n": [len(a), len(b)]}


def ols_slope(x, y):
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx


def exact_perm(x, y):
    obs = abs(ols_slope(x, y))
    n_ge = tot = 0
    for perm in itertools.permutations(y):
        tot += 1
        if abs(ols_slope(x, list(perm))) >= obs - 1e-12:
            n_ge += 1
    return {"p_exact": round(n_ge / tot, 6), "n_permutations": tot,
            "n_at_least_as_extreme": n_ge}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--output", type=Path, default=RES / "second_source.json")
    args = ap.parse_args()

    qs = load_questions()
    models = [m for m in PARAMS_B if load_preds(m)]
    if not models:
        print("no FreshQA generations found; run pipeline/gen_freshqa.py first")
        return 1
    models.sort(key=lambda m: PARAMS_B[m])

    out = {
        "scope": "Replication of the closed-book boundary contrast on FreshQA, "
                 "a temporal QA source that defines freshness by answer "
                 "volatility and effective year rather than by publication date.",
        "source": "FreshQA (Vu et al. 2024), CSV export shipped as "
                  "data/raw/freshqa_v2024-04-26.csv",
        "boundary_year": BOUNDARY_YEAR,
        "scoring": "max token-F1 over all acceptable answers per question",
        "excluded": "false-premise questions (a different failure mode)",
        "n_questions_total": len(qs),
        "n_questions_used": sum(1 for q in qs.values() if not q["false_premise"]),
        "models": {},
    }

    # ---------------- 1. the Static/Fresh gap per model ---------------------
    print(f"{'model':16s}{'params':>8s}{'static':>9s}{'fresh':>9s}"
          f"{'gap':>8s}{'ci95':>20s}{'p':>9s}")
    for m in models:
        pr = load_preds(m)
        st, fr = [], []
        for qid, q in qs.items():
            if q["false_premise"] or qid not in pr:
                continue
            s = 100.0 * best_f1(pr[qid], q["gold_answers"])
            (fr if q["year"] >= BOUNDARY_YEAR else st).append(s)
        r = boot_diff(st, fr, args.B, args.seed)
        if not r:
            continue
        r["params_B"] = PARAMS_B[m]
        out["models"][m] = r
        print(f"{m:16s}{PARAMS_B[m]:8.1f}{r['static_mean']:9.2f}"
              f"{r['fresh_mean']:9.2f}{r['diff']:8.2f}{str(r['ci95']):>20s}"
              f"{r['p_plus_one']:9.4f}")

    # ---------------- 2. the scale association on this source ---------------
    pts = [(m, math.log10(PARAMS_B[m]), out["models"][m]["diff"])
           for m in models if m in out["models"]]
    if len(pts) >= 3:
        x = [p[1] for p in pts]
        y = [p[2] for p in pts]
        slope = ols_slope(x, y)
        perm = exact_perm(x, y) if len(x) <= 8 else None
        out["scale_slope"] = {
            "n_models": len(pts),
            "slope_f1_per_decade": round(slope, 4),
            "exact_permutation": perm,
            "per_model_gap": {p[0]: p[2] for p in pts},
        }
        print(f"\nscale slope on FreshQA: {slope:.3f} F1/decade "
              f"(n={len(pts)}"
              f"{', exact p=' + str(perm['p_exact']) if perm else ''})")

    # ---------------- 3. the volatility axis --------------------------------
    vol = {}
    for m in models:
        pr = load_preds(m)
        per = {}
        for ft in ("never-changing", "slow-changing", "fast-changing"):
            st, fr = [], []
            for qid, q in qs.items():
                if q["false_premise"] or q["fact_type"] != ft or qid not in pr:
                    continue
                s = 100.0 * best_f1(pr[qid], q["gold_answers"])
                (fr if q["year"] >= BOUNDARY_YEAR else st).append(s)
            r = boot_diff(st, fr, args.B, args.seed)
            if r:
                per[ft] = r
        if per:
            vol[m] = per
    out["by_volatility"] = vol
    out["volatility_crosstab"] = {
        "note": "never-changing facts are ALL Static by construction, so their "
                "boundary gap is undefined rather than zero; the cells for the "
                "other two classes are unbalanced.",
        "counts": {ft: {sl: sum(1 for q in qs.values()
                                if not q["false_premise"]
                                and q["fact_type"] == ft
                                and (("fresh" if q["year"] >= BOUNDARY_YEAR
                                      else "static") == sl))
                        for sl in ("static", "fresh")}
                   for ft in ("never-changing", "slow-changing", "fast-changing")},
    }

    # ---- negative control: facts that cannot have gone stale ---------------
    nc = {}
    for m in models:
        pr = load_preds(m)
        v = [100.0 * best_f1(pr[qid], q["gold_answers"])
             for qid, q in qs.items()
             if not q["false_premise"] and q["fact_type"] == "never-changing"
             and qid in pr]
        if v:
            nc[m] = {"mean_f1": round(statistics.fmean(v), 3), "n": len(v),
                     "params_B": PARAMS_B[m]}
    out["never_changing_control"] = {
        "why": "A never-changing fact cannot be stale, so accuracy here is the "
               "model's baseline on this source. If Fresh-side accuracy were "
               "low simply because FreshQA is hard, this would be low too.",
        "per_model": nc,
    }
    if nc:
        print(f"\nnegative control -- never-changing facts (no staleness possible):")
        for m, v in sorted(nc.items(), key=lambda kv: kv[1]["params_B"]):
            fresh = out["models"].get(m, {}).get("fresh_mean")
            print(f"  {m:16s} never-changing {v['mean_f1']:6.2f}   "
                  f"fresh {fresh if fresh is not None else float('nan'):6.2f}")
    print(f"\nStatic-minus-Fresh gap by fact volatility:")
    print(f"{'model':16s}{'never':>12s}{'slow':>12s}{'fast':>12s}")
    for m, per in vol.items():
        row = f"{m:16s}"
        for ft in ("never-changing", "slow-changing", "fast-changing"):
            row += f"{per[ft]['diff']:12.2f}" if ft in per else f"{'-':>12s}"
        print(row)

    # ---------------- 4. false-premise questions, reported separately -------
    fp = {}
    for m in models:
        pr = load_preds(m)
        v = [100.0 * best_f1(pr[qid], q["gold_answers"])
             for qid, q in qs.items() if q["false_premise"] and qid in pr]
        if v:
            fp[m] = {"mean_f1": round(statistics.fmean(v), 3), "n": len(v)}
    out["false_premise_questions"] = {
        "note": "excluded from every contrast above; recognising a false "
                "premise is a different capability from recalling a fact",
        "per_model": fp,
    }

    # ---------------- verdict ----------------------------------------------
    sig = [m for m, r in out["models"].items() if r["ci95"][0] > 0]
    neg = [m for m, r in out["models"].items() if r["ci95"][1] < 0]
    sl = out.get("scale_slope", {})
    out["verdict"] = {
        "models_with_a_positive_gap": sig,
        "models_with_a_negative_gap": neg,
        "n_models": len(out["models"]),
        "replicates_the_direction": bool(sl.get("slope_f1_per_decade", 0) > 0),
        "reading": (
            "FreshQA shares no questions, no answer format, no notion of "
            "freshness and no evidence corpus with our benchmark. Whatever "
            "agrees across the two is unlikely to be an artifact of either."),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\npositive gap (CI excludes 0): {sig or 'none'}")
    print(f"negative gap (CI excludes 0): {neg or 'none'}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
