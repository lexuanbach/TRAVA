#!/usr/bin/env python3
"""Stale-index negative control as a date-defined difference-in-differences.

The stale index (`retrieval_stale.json`) removes only documents published on or
after 2025-01-01. For Static questions (dated before 2025) the removed documents
post-date the question, so the stale index should leave Static answers roughly
unchanged; for Fresh questions it removes the contemporaneous coverage. The
contrast is defined by question DATE, not by gold-string support grouping.

Per generator (Qwen2.5-3B, Qwen2.5-32B, Mistral-Large-2; 878 common questions):

  drop_static = mean(hybrid - hybrid_stale) over Static questions
  drop_fresh  = mean(hybrid - hybrid_stale) over Fresh questions
  DiD         = drop_fresh - drop_static

and the same for the leave-one-out index (hybrid - hybrid_loo), where both
regimes are expected to drop because LOO removes the annotated gold documents
regardless of date. Static and Fresh questions sit in disjoint quizzes, so each
bootstrap draw resamples the full quiz set once and recomputes both regimes
(and all three models) from the same draw. The three-model average DiD uses the
same draws.

Step 0 reproduces `results/stale_replication.json` exactly by calling that
script's own `scores` and `paired_block_bootstrap` with its stored B and seed.

Writes results/analysis/stale_placebo.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as C  # noqa: E402
import stale_replication as SR  # noqa: E402  (pipeline/ is on sys.path via C)

MODELS = [(k, n) for k, n, _ in SR.MODELS]
CONDS = SR.CONDS
CONTRASTS = {"stale": ("hybrid", "hybrid_stale"), "loo": ("hybrid", "hybrid_loo")}


def reproduce(stored: dict, questions: dict) -> dict:
    B = stored["bootstrap"]["B"]
    seed = stored["bootstrap"]["seed"]
    fresh = {q for q, v in questions.items() if str(v["date"])[:4] >= SR.BOUNDARY}
    rep = {"B": B, "seed": seed, "estimator": "question-pooled means; paired "
           "weekly-quiz block bootstrap over Fresh questions only "
           "(stale_replication.paired_block_bootstrap, python `random`)",
           "models": {}}
    all_ok = True
    for key, name in MODELS:
        sc = {c: SR.scores(key, c, questions) for c in CONDS}
        common = sorted(set.intersection(*(set(sc[c]) for c in CONDS)))
        cf = [q for q in common if q in fresh]
        cs = [q for q in common if q not in fresh]
        st = stored["models"][key]
        m = {"n_common": len(common), "n_fresh": len(cf), "contrasts": {},
             "f1_means_match": True}
        for c in CONDS:
            for reg, ids in (("static", cs), ("fresh", cf)):
                val = round(statistics.fmean(sc[c][q] for q in ids), 2)
                if val != st["f1"][c][reg]:
                    m["f1_means_match"] = False
        for label, worse in (("hybrid_minus_stale", "hybrid_stale"),
                             ("hybrid_minus_loo", "hybrid_loo"),
                             ("stale_minus_closedbook", "closed_book")):
            better = "hybrid" if label != "stale_minus_closedbook" else "hybrid_stale"
            lo, hi, p = SR.paired_block_bootstrap(sc[better], sc[worse], cf, B, seed)
            diff = round(round(statistics.fmean(sc[better][q] for q in cf), 2)
                         - round(statistics.fmean(sc[worse][q] for q in cf), 2), 2)
            s = st["fresh_contrasts"][label]
            ok = (diff == s["diff_f1"] and [lo, hi] == s["ci95"]
                  and p == s["two_sided_p_plus_one"])
            all_ok &= ok
            m["contrasts"][label] = {"recomputed": {"diff_f1": diff, "ci95": [lo, hi],
                                                    "p": p},
                                     "stored": {"diff_f1": s["diff_f1"], "ci95": s["ci95"],
                                                "p": s["two_sided_p_plus_one"]},
                                     "exact_match": ok}
        all_ok &= m["f1_means_match"] and len(common) == st["n_common"] \
            and len(cf) == st["n_fresh"]
        rep["models"][key] = m
    rep["all_exact_match"] = bool(all_ok)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=C.B_DEFAULT)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    questions = C.load_questions()
    stored = json.loads((C.ROOT / "results/stale_replication.json").read_text())
    rep = reproduce(stored, questions)
    print("reproduction of results/stale_replication.json:",
          "EXACT" if rep["all_exact_match"] else "MISMATCH")

    # common question set across all models x conditions
    scores = {}
    for key, _ in MODELS:
        for c in CONDS:
            rows = C.load_rows(key, c)
            scores[(key, c)] = C.f1_scores(rows, questions)
    common = sorted(set.intersection(*(set(v) for v in scores.values())))
    qb = C.QuizBoot(common, B=args.B, seed=args.seed)
    fresh = np.array([C.is_fresh(questions[q]) for q in common])
    static = ~fresh
    allq = np.ones(len(common), dtype=bool)

    def vec(key, c):
        return np.array([scores[(key, c)][q] for q in common])

    out = {
        "scope": ("Date-defined negative control for the stale index and the "
                  "leave-one-out index: retrieval-condition drop on Static vs "
                  "Fresh questions and their difference (DiD)."),
        "boundary": "Static = date year < 2025; Fresh = date year >= 2025 (from `date`)",
        "bootstrap": {"B": args.B, "seed": args.seed, "unit": "weekly quiz id "
                      "(clusters.quiz_id); one resample of the full quiz set per draw, "
                      "shared by both regimes, both contrasts and all three models",
                      "ci": "percentile 2.5/97.5",
                      "p": "two-sided, 2*min(#<=0+1, #>=0+1)/(B+1)"},
        "weightings": {"question_pooled": "mean over questions",
                       "quiz_weighted": "mean over quizzes of within-quiz means; a "
                       "quiz enters a regime's denominator only if it has questions "
                       "in that regime (Static and Fresh quizzes are disjoint)"},
        "n": {"common_questions": len(common), **{
            "static": qb.counts(static), "fresh": qb.counts(fresh),
            "all": qb.counts(allq)}},
        "reproduction_of_stale_replication_json": rep,
        "models": {},
        "three_model_average": {},
    }

    draws_store = {w: {} for w in C.WEIGHTINGS}
    for key, name in MODELS:
        entry = {"name": name, "condition_means": {}, "contrasts": {}}
        for w in C.WEIGHTINGS:
            entry["condition_means"][w] = {
                c: {reg: round(qb.est(w, vec(key, c), m)[0], 2)
                    for reg, m in (("static", static), ("fresh", fresh))}
                for c in CONDS}
        for cname, (a, b) in CONTRASTS.items():
            d = vec(key, a) - vec(key, b)
            entry["contrasts"][f"hybrid_minus_{cname}"] = {}
            for w in C.WEIGHTINGS:
                os_, ds = qb.est(w, d, static)
                of, df = qb.est(w, d, fresh)
                draws_store[w][(key, cname)] = (os_, ds, of, df)
                entry["contrasts"][f"hybrid_minus_{cname}"][w] = {
                    "drop_static": C.summarize(os_, ds),
                    "drop_fresh": C.summarize(of, df),
                    "did_fresh_minus_static": C.summarize(of - os_, df - ds),
                }
        out["models"][key] = entry

    for cname in CONTRASTS:
        out["three_model_average"][f"hybrid_minus_{cname}"] = {}
        for w in C.WEIGHTINGS:
            parts = [draws_store[w][(k, cname)] for k, _ in MODELS]
            os_ = np.mean([p[0] for p in parts]); ds = np.mean([p[1] for p in parts], axis=0)
            of = np.mean([p[2] for p in parts]); df = np.mean([p[3] for p in parts], axis=0)
            out["three_model_average"][f"hybrid_minus_{cname}"][w] = {
                "drop_static": C.summarize(os_, ds),
                "drop_fresh": C.summarize(of, df),
                "did_fresh_minus_static": C.summarize(of - os_, df - ds),
                "note": "unweighted average over Qwen2.5-3B, Qwen2.5-32B, "
                        "Mistral-Large-2 of the per-model quantity, same quiz draws",
            }

    out["reading"] = (
        "The stale index is a date-defined intervention. drop_static is the "
        "placebo arm (removed documents post-date the question). A DiD interval "
        "excluding zero says the stale-index cost is larger on Fresh than on "
        "Static questions. A Static interval that includes zero is NOT an "
        "equivalence result; read its width as the size of Static effect the data "
        "cannot exclude. The LOO index removes annotated gold documents in both "
        "regimes, so both drops are expected to be positive; its DiD is reported "
        "for comparison, not as a placebo.")
    p = C.write_json(out, "stale_placebo.json")

    for key, name in MODELS:
        for cname in CONTRASTS:
            for w in C.WEIGHTINGS:
                r = out["models"][key]["contrasts"][f"hybrid_minus_{cname}"][w]
                print(f"{name:16s} {cname:5s} {w:15s} static {r['drop_static']['estimate']:+6.2f} "
                      f"{r['drop_static']['ci95']}  fresh {r['drop_fresh']['estimate']:+6.2f} "
                      f"{r['drop_fresh']['ci95']}  DiD {r['did_fresh_minus_static']['estimate']:+6.2f} "
                      f"{r['did_fresh_minus_static']['ci95']}")
    for cname in CONTRASTS:
        for w in C.WEIGHTINGS:
            r = out["three_model_average"][f"hybrid_minus_{cname}"][w]
            print(f"{'3-model avg':16s} {cname:5s} {w:15s} static {r['drop_static']['estimate']:+6.2f} "
                  f"{r['drop_static']['ci95']}  fresh {r['drop_fresh']['estimate']:+6.2f} "
                  f"{r['drop_fresh']['ci95']}  DiD {r['did_fresh_minus_static']['estimate']:+6.2f} "
                  f"{r['did_fresh_minus_static']['ci95']}")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
