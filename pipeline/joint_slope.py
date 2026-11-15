#!/usr/bin/env python
"""Joint quiz block-bootstrap for the capability-slope association.

Identification problem this addresses: the cross-model slope was
estimated by permuting seven model-level point estimates. Because every model is
evaluated on the *same* 878 questions, the model-level gaps carry correlated
measurement error, and a permutation over seven summary numbers neither
propagates the uncertainty in each gap nor respects that correlation.

This script instead resamples RealTimeQA weekly quizzes (clusters) *jointly*
across all models: on each replicate the same resampled set of quizzes is used to
recompute every model's boundary gap, and the slope of gap on log10(parameters)
is refit. The resulting interval therefore reflects both the sampling error in the
gaps and their cross-model correlation.

It also reports the year-pair sensitivity: the
slope and the per-model gap under every admissible pre/post year contrast, so the
dependence on the 2023-vs-2026 choice is visible rather than asserted.

Outputs results/joint_slope.json.

Usage:
    python pipeline/joint_slope.py
    python pipeline/joint_slope.py --replicates 10000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from metrics import token_f1  # noqa: E402
from clusters import quiz_id  # canonical resampling unit

# The seven open-weight rungs, in the order the paper reports them.
LADDER = [
    ("qwen", "Qwen2.5-0.5B", 0.5),
    ("qwen1_5b", "Qwen2.5-1.5B", 1.5),
    ("qwen3b", "Qwen2.5-3B", 3.0),
    ("qwen7b", "Qwen2.5-7B", 7.0),
    ("qwen32b", "Qwen2.5-32B", 32.0),
    ("qwen72b", "Qwen2.5-72B", 72.0),
    ("mistral_large", "Mistral-Large-2", 123.0),
]
BOUNDARY = date(2025, 1, 1)


def load_questions() -> dict[str, dict]:
    path = ROOT / "data/processed/questions.jsonl"
    return {json.loads(l)["qid"]: json.loads(l) for l in path.open(encoding="utf-8")}


def load_scores(model: str, questions: dict[str, dict]) -> dict[str, float]:
    """Per-question closed-book token-F1 for one model."""
    path = ROOT / f"data/processed/generations_{model}/closed_book.jsonl"
    out = {}
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        q = questions.get(rec["qid"])
        if q:
            out[rec["qid"]] = 100.0 * token_f1(rec["prediction"], q["gold_answer"])
    return out


def ols_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def gap_for(scores: dict[str, float], qids_pre: list[str], qids_post: list[str]) -> float:
    pre = [scores[q] for q in qids_pre if q in scores]
    post = [scores[q] for q in qids_post if q in scores]
    if not pre or not post:
        return float("nan")
    return statistics.fmean(pre) - statistics.fmean(post)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--output", type=Path, default=ROOT / "results/joint_slope.json")
    args = ap.parse_args()

    questions = load_questions()
    scores = {m: load_scores(m, questions) for m, _, _ in LADDER}
    logp = [math.log10(p) for _, _, p in LADDER]

    # Quizzes are the resampling unit: RealTimeQA ships one dated quiz per week.
    # CANONICAL CLUSTER UNIT: the weekly quiz id is the prefix of the question id
    # (`20260515_12` -> `20260515`), which is what cluster_robust.py uses and what
    # the setup section reports. Grouping by `q["date"]` instead splits 86 weekly
    # batches that carry two dates, inflating the cluster count 199 -> 282 and
    # understating the correlation the block bootstrap exists to capture.
    quiz_of = {qid: quiz_id(qid) for qid in questions}
    quizzes = sorted({d for d in quiz_of.values()})
    by_quiz = defaultdict(list)
    for qid, d in quiz_of.items():
        by_quiz[d].append(qid)

    def split_by_boundary(qids):
        pre = [q for q in qids if questions[q]["date"] < BOUNDARY.isoformat()]
        post = [q for q in qids if questions[q]["date"] >= BOUNDARY.isoformat()]
        return pre, post

    all_pre, all_post = split_by_boundary(list(questions))
    observed_gaps = [gap_for(scores[m], all_pre, all_post) for m, _, _ in LADDER]
    observed_slope = ols_slope(logp, observed_gaps)

    # ---- joint quiz bootstrap -------------------------------------------------
    rng = random.Random(args.seed)
    slopes, gap_draws = [], [[] for _ in LADDER]
    for _ in range(args.replicates):
        drawn = [quizzes[rng.randrange(len(quizzes))] for _ in range(len(quizzes))]
        qids = [q for d in drawn for q in by_quiz[d]]
        pre, post = split_by_boundary(qids)
        if not pre or not post:
            continue
        gaps = [gap_for(scores[m], pre, post) for m, _, _ in LADDER]
        if any(math.isnan(g) for g in gaps):
            continue
        for i, g in enumerate(gaps):
            gap_draws[i].append(g)
        slopes.append(ols_slope(logp, gaps))

    slopes.sort()
    lo, hi = slopes[int(0.025 * len(slopes))], slopes[int(0.975 * len(slopes))]
    p_two_sided = 2 * min(
        sum(1 for s in slopes if s <= 0) / len(slopes),
        sum(1 for s in slopes if s >= 0) / len(slopes),
    )

    report = {
        "scope": ("Slope of the closed-book boundary gap on log10(parameters) across the seven "
                  "open-weight rungs, with uncertainty from a joint block bootstrap over weekly "
                  "quizzes. The same resampled quizzes are used for every model on each replicate, "
                  "so the interval reflects the correlation induced by evaluating all models on the "
                  "same questions."),
        "boundary": BOUNDARY.isoformat(),
        "n_quizzes": len(quizzes),
        "n_questions": len(questions),
        "replicates_used": len(slopes),
        "models": [
            {"key": m, "name": n, "params_B": p,
             "observed_gap_f1": round(g, 4),
             "gap_ci95": [round(sorted(d)[int(0.025 * len(d))], 4),
                          round(sorted(d)[int(0.975 * len(d))], 4)] if d else None}
            for (m, n, p), g, d in zip(LADDER, observed_gaps, gap_draws)
        ],
        "slope_f1_per_decade": {
            "observed": round(observed_slope, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "bootstrap_median": round(statistics.median(slopes), 4),
            # Plus-one estimator: with B replicates the smallest attainable
            # two-sided value is 2/(B+1), so a raw 0.0 would claim more
            # resolution than the simulation supplies.
            "two_sided_p_plus_one": round(min(1.0, p_two_sided if p_two_sided > 0
                                              else 2.0 / (len(slopes) + 1)), 5),
            "p_resolution_floor": round(2.0 / (len(slopes) + 1), 5),
            "p_note": ("Reported with the plus-one estimator. We quote the interval "
                       "as the primary evidence; the p-value cannot resolve below "
                       "the floor."),
            "interpretation": ("An interval excluding zero indicates the gap increases with scale "
                               "across this ladder. It does not identify capability as the cause: "
                               "the ladder changes model family at the top rung, and pretraining "
                               "cutoffs are undisclosed for several rungs."),
        },
    }

    # ---- year-pair sensitivity ------------------------------------------------
    years = sorted({questions[q]["date"][:4] for q in questions})
    by_year = defaultdict(list)
    for qid, q in questions.items():
        by_year[q["date"][:4]].append(qid)
    # only contrasts with one year on each side of the analysis boundary
    # are pre/post contrasts. Same-side pairs (e.g. 2022 vs 2023, both Static;
    # 2025 vs 2026, both Fresh) measure year-difficulty drift, not the boundary,
    # and were previously pooled into the sensitivity range.
    BOUNDARY_YEAR = BOUNDARY.year
    pairs, same_side = {}, {}
    for pre_y in years:
        for post_y in years:
            if pre_y >= post_y:
                continue
            cross = int(pre_y) < BOUNDARY_YEAR <= int(post_y)
            gaps = [gap_for(scores[m], by_year[pre_y], by_year[post_y]) for m, _, _ in LADDER]
            if any(math.isnan(g) for g in gaps):
                continue
            entry = {
                "slope_f1_per_decade": round(ols_slope(logp, gaps), 4),
                "gaps": {m: round(g, 3) for (m, _, _), g in zip(LADDER, gaps)},
                "n_pre": len(by_year[pre_y]), "n_post": len(by_year[post_y]),
            }
            (pairs if cross else same_side)[f"{pre_y}_vs_{post_y}"] = entry
    slopes_by_pair = [v["slope_f1_per_decade"] for v in pairs.values()]
    report["year_pair_sensitivity"] = {
        "note": ("Slope under every genuine CROSS-BOUNDARY year contrast (one year on each "
                 "side of the analysis boundary). Same-side pairs are reported separately "
                 "under `same_side_contrasts`: they measure year-difficulty drift, not the "
                 "boundary, and pooling them into this range was an error in an earlier "
                 "version. Reported so the dependence on the 2023-vs-2026 choice is visible."),
        "pairs": pairs,
        "same_side_contrasts": same_side,
        "slope_range": [round(min(slopes_by_pair), 4), round(max(slopes_by_pair), 4)],
        "slope_median": round(statistics.median(slopes_by_pair), 4),
        "n_pairs": len(pairs),
        "sign_agreement": round(sum(1 for s in slopes_by_pair if s > 0) / len(slopes_by_pair), 3),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["slope_f1_per_decade"]
    print(f"joint quiz bootstrap over {len(quizzes)} quizzes, {len(slopes)} replicates")
    print(f"  slope = {s['observed']:.2f} F1/decade  95% CI [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]  "
          f"two-sided p = {s['two_sided_p_plus_one']:.5f} "
          f"(plus-one; floor {s['p_resolution_floor']:.5f})")
    print("\n  per-model boundary gap (observed [95% CI]):")
    for m in report["models"]:
        ci = m["gap_ci95"]
        print(f"    {m['name']:18s} {m['params_B']:6.1f}B  {m['observed_gap_f1']:6.2f} "
              f"[{ci[0]:6.2f}, {ci[1]:6.2f}]")
    ys = report["year_pair_sensitivity"]
    print(f"\n  year-pair sensitivity: {ys['n_pairs']} contrasts, slope range "
          f"[{ys['slope_range'][0]:.2f}, {ys['slope_range'][1]:.2f}], median {ys['slope_median']:.2f}, "
          f"positive in {100*ys['sign_agreement']:.0f}% of contrasts")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
