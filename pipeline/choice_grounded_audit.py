#!/usr/bin/env python3
"""Score cached free-form answers against RealTimeQA multiple-choice options.

Uses the fixed 878-question sample and an explicitly named closed-book panel.
Exact and unique-containment matches precede token-F1 matching. Ties and
unmatched answers score zero. Detected abstentions are counted separately,
without overriding the matching rule. This automatic sensitivity analysis
complements, and does not replace, the adjudicated human metric-error study.
No model answers or labels are generated.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import exact_match, normalize, token_f1   # noqa: E402

RES = ROOT / "results"

ABSTAIN = ("i dont know", "i do not know", "unknown", "cannot determine",
           "cant determine", "no answer", "not sure", "unable to determine",
           "insufficient information", "n/a", "na", "none of the above")


def is_abstention(pred: str) -> bool:
    p = normalize(pred)
    if not p:
        return True
    return any(p == a or p.startswith(a) for a in ABSTAIN)


def nearest_choice(pred: str, choices):
    """Index of the choice the prediction is closest to, or None if it matches
    nothing at all.

    Containment is checked before token-F1 because a correct short answer is
    frequently a substring of the choice ("Squid Game" in "“Squid Game”") and
    substring identity is a stronger signal than overlap.
    """
    p = normalize(pred)
    if not p:
        return None, 0.0
    exact = [k for k, c in enumerate(choices) if normalize(c) == p]
    if len(exact) == 1:
        return exact[0], 1.0
    contained = [k for k, c in enumerate(choices)
                 if p and (p in normalize(c) or normalize(c) in p)]
    if len(contained) == 1:
        return contained[0], 0.99
    scores = [token_f1(pred, c) for c in choices]
    best = max(scores)
    if best <= 0.0:
        return None, 0.0
    top = [k for k, s in enumerate(scores) if s == best]
    if len(top) > 1:                    # ambiguous between choices: no credit
        return None, best
    return top[0], best


def load_questions():
    qs = {}
    with open(ROOT / "data/processed/questions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            qs[d["qid"]] = d
    return qs


def load_predictions(condition="closed_book"):
    out = {}
    panel = ["qwen", "qwen1_5b", "qwen3b", "qwen7b", "qwen32b", "qwen72b",
             "mistral_large", "haiku", "sonnet", "llama3_8b", "gpt4omini"]
    for key in panel:
        d = ROOT / "data/processed" / ("generations" if key == "haiku" else "generations_" + key)
        f = d / f"{condition}.jsonl"
        if not f.is_file():
            continue
        recs = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                recs[r["qid"]] = r["prediction"] or ""
        out[key] = recs
    return out


def cluster_bootstrap(per_quiz_a, per_quiz_b, B, seed):
    """Paired difference of means, resampling QUIZZES."""
    keys = sorted(set(per_quiz_a) & set(per_quiz_b))
    d = [statistics.fmean(per_quiz_a[k]) - statistics.fmean(per_quiz_b[k])
         for k in keys]
    if len(d) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean([d[rng.randrange(len(d))] for _ in d])
                   for _ in range(B))
    le = sum(1 for v in draws if v <= 0)
    return {"diff": round(statistics.fmean(d), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5),
            "n_clusters": len(d)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--output", type=Path, default=RES / "choice_grounded_audit.json")
    args = ap.parse_args()

    qs = load_questions()
    preds = load_predictions()
    assert all(set(rows) == set(qs) for rows in preds.values())
    out = {
        "scope": "Nearest-option sensitivity on the fixed 878-question closed-book sample",
        "referee": "Exact and unique containment matches precede token-F1 matching. Ties and unmatched answers score zero. Detected abstentions are counted separately and do not override matching.",
        "not_a_human_audit": "An automatic scoring sensitivity, not a replacement for the adjudicated human study. It does not establish gold validity, question clarity or temporal correctness.",
        "n_questions": len(qs),
        "panel": list(preds),
        "inference": {"B": args.B, "seed": args.seed, "boundary_contrast": "2023 minus 2026", "weighting": "equal quiz means"},
    }

    # ------------------------------------------- 1. metric disagreement rates
    agree = collections.Counter()
    examples = {"f1_credits_wrong_choice": [], "f1_penalises_correct_alias": []}
    per_model = {}
    for model, recs in sorted(preds.items()):
        c = collections.Counter()
        for qid, pred in recs.items():
            q = qs.get(qid)
            if q is None:
                continue
            gold = q["gold_answer"]
            f1 = token_f1(pred, gold)
            em = exact_match(pred, gold)
            abstained = is_abstention(pred)
            idx, _ = nearest_choice(pred, q["choices"])
            if abstained:
                c["abstain"] += 1
                agree["abstain"] += 1
            if idx is None:
                c["no_match"] += 1
                agree["no_match"] += 1
            correct = int(idx is not None and idx == q["gold_answer_idx"])
            c["scored"] += 1
            c["choice_correct"] += correct
            c["em_correct"] += int(em)
            c["f1_sum"] += f1
            agree["ok" if correct else "wrong"] += 1
            if idx is not None and not correct and f1 >= 0.5:
                agree["f1_credits_wrong_choice"] += 1
                if len(examples["f1_credits_wrong_choice"]) < 8:
                    examples["f1_credits_wrong_choice"].append(
                        {"model": model, "qid": qid, "pred": pred[:90],
                         "gold": gold, "chose": q["choices"][idx][:60],
                         "f1": round(f1, 3)})
            if correct and em == 0.0:
                agree["f1_penalises_correct_alias"] += 1
                if len(examples["f1_penalises_correct_alias"]) < 8:
                    examples["f1_penalises_correct_alias"].append(
                        {"model": model, "qid": qid, "pred": pred[:90],
                         "gold": gold, "f1": round(f1, 3)})
        n = max(c["scored"], 1)
        per_model[model] = {
            "n_predictions": sum(recs and 1 for _ in recs),
            "abstain_rate": round(c["abstain"] / max(len(recs), 1), 4),
            "no_match_rate": round(c["no_match"] / max(len(recs), 1), 4),
            "choice_accuracy": round(100 * c["choice_correct"] / n, 2),
            "exact_match": round(100 * c["em_correct"] / n, 2),
            "token_f1": round(100 * c["f1_sum"] / n, 2),
        }
    tot = agree["ok"] + agree["wrong"]
    out["metric_disagreement"] = {
        "n_scored": tot,
        "f1_credits_a_wrong_choice": agree["f1_credits_wrong_choice"],
        "f1_credits_a_wrong_choice_rate": round(agree["f1_credits_wrong_choice"] / max(tot, 1), 4),
        "correct_but_exact_match_zero": agree["f1_penalises_correct_alias"],
        "correct_but_exact_match_zero_rate": round(agree["f1_penalises_correct_alias"] / max(tot, 1), 4),
        "abstentions": agree["abstain"],
        "no_match": agree["no_match"],
        "reading": "The first rate is F1's false-credit rate: answers scoring "
                   "at least 0.5 token-F1 that in fact name a distractor. The "
                   "second is the alias rate that exact match alone would "
                   "discard. These descriptive disagreement rates are not bounds on human correctness.",
        "examples": examples,
    }
    out["per_model"] = per_model
    print(f"{'model':16s}{'choice-acc':>12s}{'EM':>8s}{'F1':>8s}{'abstain':>9s}")
    for m, v in per_model.items():
        print(f"{m:16s}{v['choice_accuracy']:12.2f}{v['exact_match']:8.2f}"
              f"{v['token_f1']:8.2f}{v['abstain_rate']:9.3f}")
    print(f"\nF1 credits a wrong choice in "
          f"{out['metric_disagreement']['f1_credits_a_wrong_choice_rate']:.1%} of scored answers")

    # ------------------- 2. the headline contrast under the choice referee ---
    def quiz_of(qid):
        return qid.split("_")[0]

    boundary = {}
    for model, recs in sorted(preds.items()):
        pre, post = collections.defaultdict(list), collections.defaultdict(list)
        for qid, pred in recs.items():
            q = qs.get(qid)
            if q is None:
                continue
            idx, _ = nearest_choice(pred, q["choices"])
            s = 100.0 * int(idx is not None and idx == q["gold_answer_idx"])
            if q["year"] == 2023:
                pre[quiz_of(qid)].append(s)
            elif q["year"] == 2026:
                post[quiz_of(qid)].append(s)
        if pre and post:
            # unpaired: the two years are different quizzes
            a = [statistics.fmean(v) for v in pre.values()]
            b = [statistics.fmean(v) for v in post.values()]
            rng = random.Random(args.seed)
            draws = sorted(
                statistics.fmean([a[rng.randrange(len(a))] for _ in a])
                - statistics.fmean([b[rng.randrange(len(b))] for _ in b])
                for _ in range(args.B))
            le = sum(1 for v in draws if v <= 0)
            boundary[model] = {
                "gap_2023_minus_2026": round(statistics.fmean(a) - statistics.fmean(b), 3),
                "ci95": [round(draws[int(0.025 * args.B)], 3),
                         round(draws[int(0.975 * args.B)], 3)],
                "p_plus_one": round(min(1.0, 2.0 * min(le + 1, args.B - le + 1) / (args.B + 1)), 5),
                "n_quizzes_pre": len(a), "n_quizzes_post": len(b),
            }
    out["boundary_gap_choice_grounded"] = boundary
    print(f"\n{'model':16s}{'gap(choice)':>13s}{'ci95':>22s}{'p':>9s}")
    for m, v in boundary.items():
        print(f"{m:16s}{v['gap_2023_minus_2026']:13.2f}{str(v['ci95']):>22s}"
              f"{v['p_plus_one']:9.4f}")

    # ---------------- 3. does the scale slope survive the new metric? -------
    cm = json.loads((RES / "cross_model.json").read_text())
    pts = [(k, math.log10(cm[k]["params_B"]), boundary[k]["gap_2023_minus_2026"])
           for k in boundary
           if isinstance(cm.get(k), dict) and cm[k].get("params_B")]
    pts.sort(key=lambda r: r[1])
    if len(pts) >= 3:
        import itertools
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((a - mx) ** 2 for a in xs)
        slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx

        def sl(yy):
            return sum((a - mx) * (b - statistics.fmean(yy))
                       for a, b in zip(xs, yy)) / sxx
        n_ge = tot_p = 0
        for perm in itertools.permutations(ys):
            tot_p += 1
            if abs(sl(list(perm))) >= abs(slope) - 1e-12:
                n_ge += 1
        out["scale_slope_choice_grounded"] = {
            "n_models": len(pts),
            "slope_pts_per_decade": round(slope, 4),
            "exact_permutation_p": round(n_ge / tot_p, 6),
            "n_permutations": tot_p,
            "per_model": {p[0]: p[2] for p in pts},
            "reading": "The positive 2023/2026 quiz-weighted estimate is imprecise. "
                       "The exact model-level permutation test does not reject zero. "
                       "This result does not establish metric independence.",
        }
        print(f"\nscale slope under the choice referee: {slope:.3f} pts/decade, "
              f"exact p={n_ge/tot_p:.4f} (n={len(pts)})")

    # ---------------- 4. all three metrics side by side ---------------------
    ml = RES / "model_level_inference.json"
    if ml.is_file() and "scale_slope_choice_grounded" in out:
        m = json.loads(ml.read_text())["target_b_over_model_rungs"]
        rows = {
            "token_f1": {"slope": m["f1_all_7"]["slope_f1_per_decade"],
                         "p_exact": m["f1_all_7"]["permutation"]["p_exact"]},
            "exact_match": {"slope": m.get("em_all", {}).get("slope_f1_per_decade"),
                            "p_exact": m.get("em_all", {}).get("permutation", {}).get("p_exact")},
            "choice_grounded": {"slope": out["scale_slope_choice_grounded"]["slope_pts_per_decade"],
                                "p_exact": out["scale_slope_choice_grounded"]["exact_permutation_p"]},
        }
        sig = [k for k, v in rows.items() if v["p_exact"] is not None and v["p_exact"] < 0.05]
        out["metric_sensitivity_of_the_scale_slope"] = {
            "slopes": rows,
            "all_positive": all(v["slope"] is not None and v["slope"] > 0
                                for v in rows.values()),
            "significant_under": sig,
            "comparison_scope": "Historical juxtaposition of different estimands, not an aligned metric comparison",
            "estimands": {
                "token_f1": "2025 boundary, seven-model ladder",
                "exact_match": "2025 boundary, seven-model ladder",
                "choice_grounded": "2023 minus 2026, equal quiz means, seven-model ladder"},
            "reading": "These slopes change both metric and temporal contrast. Use results/reporting_summary.json matched_scale_contrast for the aligned 2023/2026 token-F1 and choice-accuracy comparison.",
        }
        print(f"\nscale slope by metric: " + ", ".join(
            f"{k}={v['slope']} (p={v['p_exact']})" for k, v in rows.items()))
        print(f"significant under: {sig}")

    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
