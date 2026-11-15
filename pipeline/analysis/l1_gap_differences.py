#!/usr/bin/env python3
"""Layer-1 closed-book gaps: shared boundary vs conservative pair, estimated jointly.

The paper and analysis compare a shared-boundary gap (Static < 2025 vs Fresh
>= 2025) with a conservative 2023-vs-2026 gap. Comparing a significant estimate
with a non-significant one is not a test of their difference, and both gaps use
overlapping questions (2023 is Static, 2026 is Fresh). This script therefore
computes, from the SAME weekly-quiz draws (full quiz set resampled once):

  gap_shared = F1(year < 2025) - F1(year >= 2025)
  gap_cons   = F1(2023) - F1(2026)
  diff       = gap_shared - gap_cons

(a) on the sampled 878 questions for every generator with a closed_book cache;
(b) on the uncapped 4,176-question closed_book_full caches (Qwen2.5-32B,
    Qwen2.5-72B, GPT-4o-mini), after reproducing results/fullscale.json with
    that script's own estimator (question-pooled means; Static and Fresh quiz
    sets resampled independently; B=4000; crc32-seeded numpy RNG per label);
(c) year-by-year closed-book means and an answer-permutation null for the
    Qwen2.5 0.5B/1.5B/3B/7B rungs (other generators as reference): predictions
    are shuffled among questions of the same year and answer-length bucket
    (gold answer whitespace words 1 / 2 / 3+), 1,000 times, to ask how far each
    year's observed F1 sits above what type-matched but question-unrelated
    answers score (the "floor" reading).

Writes results/analysis/l1_gap_differences.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as C  # noqa: E402
import metrics as M  # noqa: E402
import fullscale_agg as FA  # noqa: E402

SMALL_QWEN = ["qwen", "qwen1_5b", "qwen3b", "qwen7b"]
FULL_GENS = ["qwen32b", "qwen72b", "gpt4omini"]
N_PERM = 1000


def closed_book_generators():
    gens = sorted(p.parent.name.replace("generations_", "")
                  for p in C.PROC.glob("generations_*/closed_book.jsonl"))
    if (C.PROC / "generations" / "closed_book.jsonl").is_file():
        gens.append("haiku")
    return gens


def gap_block(qb, v, qs_list, weighting):
    years = np.array([C.year_of(q) for q in qs_list])
    masks = {"static": years < C.BOUNDARY_YEAR, "fresh": years >= C.BOUNDARY_YEAR,
             "y2023": years == 2023, "y2026": years == 2026}
    e = {k: qb.est(weighting, v, m) for k, m in masks.items()}
    gs = (e["static"][0] - e["fresh"][0], e["static"][1] - e["fresh"][1])
    gc = (e["y2023"][0] - e["y2026"][0], e["y2023"][1] - e["y2026"][1])
    return {
        "means": {k: round(e[k][0], 2) for k in masks},
        "gap_shared_static_minus_fresh": C.summarize(*gs),
        "gap_cons_2023_minus_2026": C.summarize(*gc),
        "diff_gap_shared_minus_gap_cons": C.summarize(gs[0] - gc[0], gs[1] - gc[1]),
        "counts": {k: qb.counts(m) for k, m in masks.items()},
    }


def by_year_block(qb, v, qs_list, weighting):
    years = np.array([C.year_of(q) for q in qs_list])
    return {str(y): C.summarize(*qb.est(weighting, v, years == y),
                                extra=qb.counts(years == y))
            for y in sorted(set(years.tolist()))}


# ---------------------------------------------------------------- (a) sampled
def part_a(args, questions):
    qids = sorted(questions)
    qlist = [questions[q] for q in qids]
    qb = C.QuizBoot(qids, B=args.B, seed=args.seed)
    stored = json.loads((C.ROOT / "results/cross_model.json").read_text())
    out = {"n_questions": len(qids), "n_quizzes": qb.nq, "generators": {}}
    for g in closed_book_generators():
        rows = C.load_rows(g, "closed_book")
        sc = C.f1_scores(rows, questions)
        if set(sc) != set(qids):
            out["generators"][g] = {"skipped": f"only {len(sc)}/{len(qids)} scored"}
            continue
        v = np.array([sc[q] for q in qids])
        entry = {w: gap_block(qb, v, qlist, w) for w in C.WEIGHTINGS}
        if g in stored and "cliff" in stored[g]:
            cl = stored[g]["cliff"]
            m = entry["question_pooled"]["means"]
            recomputed = {"static_f1": round(m["static"], 1), "fresh_f1": round(m["fresh"], 1),
                          "gap": round(round(m["static"], 1) - round(m["fresh"], 1), 1)}
            by_year_ok = all(
                abs(round(float(v[np.array([C.year_of(x) for x in qlist]) == int(y)].mean()), 2)
                    - yy["f1"]) < 0.006
                for y, yy in stored[g]["closed_book_by_year"].items())
            entry["reproduction_cross_model_json"] = {
                "stored": {k: cl[k] for k in ("static_f1", "fresh_f1", "gap")},
                "recomputed_question_pooled": recomputed,
                "static_fresh_match": all(abs(recomputed[k] - cl[k]) < 0.051
                                          for k in ("static_f1", "fresh_f1")),
                "gap_match": abs(recomputed["gap"] - cl["gap"]) < 0.051
                or abs((m["static"] - m["fresh"]) - cl["gap"]) < 0.051,
                "by_year_match": by_year_ok,
            }
        out["generators"][g] = entry
    return out


# ---------------------------------------------------------------- (b) full
def part_b(args, questions_full, sampled_ids):
    stored = json.loads((C.ROOT / "results/fullscale.json").read_text())
    qids = sorted(questions_full)
    qlist = [questions_full[q] for q in qids]
    qb = C.QuizBoot(qids, B=args.B, seed=args.seed + 1)
    mism = sum(1 for q in qlist if int(q["year"]) != C.year_of(q))
    out = {"n_questions": len(qids), "n_quizzes": qb.nq,
           "year_field_vs_date_mismatches": mism,
           "stored_estimator": ("fullscale_agg.block_boot: question-pooled means; the two "
                                "groups' quiz sets resampled INDEPENDENTLY; B=4000; "
                                "numpy RNG seeded by crc32(label); percentile CI"),
           "generators": {}}
    for g in FULL_GENS:
        rows = C.load_rows(g, "closed_book_full")
        sc = C.f1_scores(rows, questions_full)
        entry = {"n_scored": len(sc)}
        # reproduction with the stored estimator (fullscale_agg's own function)
        Qf = questions_full

        def contrast(sel_a, sel_b, label, restrict=None):
            A = [q for q in sc if sel_a(Qf[q]["year"]) and (restrict is None or q in restrict)]
            Bq = [q for q in sc if sel_b(Qf[q]["year"]) and (restrict is None or q in restrict)]
            return FA.block_boot([sc[q] for q in A], [FA.batch_of(q) for q in A],
                                 [sc[q] for q in Bq], [FA.batch_of(q) for q in Bq],
                                 f"{g}:{label}")

        rec = {"boundary_2025": contrast(lambda y: y < 2025, lambda y: y >= 2025, "bnd"),
               "clean_2023_vs_2026": contrast(lambda y: y == 2023, lambda y: y == 2026, "cln"),
               "sampled_subset_only": contrast(lambda y: y < 2025, lambda y: y >= 2025, "sub",
                                               restrict=sampled_ids)}
        st = stored[g]
        match = {}
        for k, r in rec.items():
            s = st[k]
            match[k] = all(r[f] == s[f] for f in ("diff", "se", "ci95", "p_two_sided", "n",
                                                   "n_clusters"))
        by_year = {str(y): round(float(np.mean([sc[q] for q in sc if Qf[q]["year"] == y])), 2)
                   for y in sorted({Qf[q]["year"] for q in sc})}
        match["by_year"] = by_year == st["by_year"]
        entry["reproduction_fullscale_json"] = {"recomputed": rec, "by_year": by_year,
                                                "exact_match": match,
                                                "all_exact": all(match.values())}
        if set(sc) != set(qids):
            entry["joint"] = {"skipped": f"only {len(sc)}/{len(qids)} scored"}
        else:
            v = np.array([sc[q] for q in qids])
            entry["joint"] = {w: gap_block(qb, v, qlist, w) for w in C.WEIGHTINGS}
        out["generators"][g] = entry
    return out


# ---------------------------------------------------------------- (c) floor
def _tok(s):
    return M.normalize(s).split()


def _f1_tokens(p, g):
    # identical logic to metrics.token_f1 on pre-normalised token lists
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def part_c(args, questions):
    qids = sorted(questions)
    qlist = [questions[q] for q in qids]
    years = np.array([C.year_of(q) for q in qlist])
    bucket = np.array([min(len(q["gold_answer"].split()), 3) for q in qlist])
    gold_t = [_tok(q["gold_answer"]) for q in qlist]
    qtok = [set(_tok(q["question"])) for q in qlist]
    qb = C.QuizBoot(qids, B=args.B, seed=args.seed + 2)
    cells = {}
    for i, (y, b) in enumerate(zip(years, bucket)):
        cells.setdefault((int(y), int(b)), []).append(i)
    out = {"n_permutations": N_PERM, "perm_seed": args.seed + 3,
           "null_definition": ("within each (year, answer-length bucket) cell, closed-book "
                               "predictions are permuted among questions; null F1 = "
                               "token F1 of a type-matched prediction written for a "
                               "different question. p_one_sided_perm = (1 + #null >= "
                               "observed)/(1 + N)."),
           "diagnostics": ("exact_match_pct: normalised prediction == normalised gold. "
                           "echo_stripped_f1: token F1 after removing tokens that occur "
                           "in the question text from both prediction and gold (questions "
                           "whose gold tokens all occur in the question are excluded and "
                           "counted). The shuffle null is weak against answers assembled "
                           "from question words or generic high-prior entities; these "
                           "diagnostics are descriptive and have no null of their own."),
           "cells": "(date year, gold-answer whitespace word count bucket 1/2/3+)",
           "cell_sizes": {f"{y}/{b}": len(ix) for (y, b), ix in sorted(cells.items())},
           "headline_generators": SMALL_QWEN, "generators": {}}
    gens = SMALL_QWEN + [g for g in closed_book_generators() if g not in SMALL_QWEN]
    for g in gens:
        rows = C.load_rows(g, "closed_book")
        if not set(qids) <= set(rows):
            continue
        pred_t = [_tok(rows[q].get("prediction") or "") for q in qids]
        obs = np.array([100.0 * _f1_tokens(pred_t[i], gold_t[i]) for i in range(len(qids))])
        chk = np.array([100.0 * M.token_f1(rows[q].get("prediction") or "",
                                           questions[q]["gold_answer"]) for q in qids])
        assert np.allclose(obs, chk)
        mats = {}
        for key, ix in cells.items():
            mats[key] = np.array([[100.0 * _f1_tokens(pred_t[j], gold_t[i]) for j in ix]
                                  for i in ix])
        rng = np.random.default_rng(args.seed + 3)
        null = np.zeros((N_PERM, len(qids)))
        for p in range(N_PERM):
            for key, ix in cells.items():
                perm = rng.permutation(len(ix))
                null[p, ix] = mats[key][np.arange(len(ix)), perm]
        groups = {str(y): years == y for y in sorted(set(years.tolist()))}
        groups["static_lt2025"] = years < C.BOUNDARY_YEAR
        groups["fresh_ge2025"] = years >= C.BOUNDARY_YEAR
        groups["all"] = np.ones(len(qids), bool)
        # question-echo diagnostic: the shuffle null breaks the pairing with the
        # question, so answers built from question words also count as "above
        # null". Echo-stripped F1 drops tokens that occur in the question from
        # both prediction and gold before scoring.
        stripped = np.array([100.0 * _f1_tokens([t for t in pred_t[i] if t not in qtok[i]],
                                                [t for t in gold_t[i] if t not in qtok[i]])
                             if [t for t in gold_t[i] if t not in qtok[i]] else np.nan
                             for i in range(len(qids))])
        em = np.array([float(pred_t[i] == gold_t[i]) for i in range(len(qids))])
        perm_out = {}
        for name, m in groups.items():
            o = float(obs[m].mean())
            nd = null[:, m].mean(axis=1)
            sd = float(nd.std(ddof=1))
            perm_out[name] = {
                "exact_match_pct": round(100.0 * float(em[m].mean()), 2),
                "echo_stripped_f1": round(float(np.nanmean(stripped[m])), 2),
                "n_gold_all_tokens_in_question": int(np.isnan(stripped[m]).sum()),
                "n": int(m.sum()), "observed_f1": round(o, 2),
                "null_mean_f1": round(float(nd.mean()), 2),
                "null_sd": round(sd, 2),
                "null_95_range": [round(float(x), 2) for x in np.percentile(nd, [2.5, 97.5])],
                "excess_over_null": round(o - float(nd.mean()), 2),
                "z": round((o - float(nd.mean())) / sd, 2) if sd > 0 else None,
                "p_one_sided_perm": round((1 + int((nd >= o).sum())) / (N_PERM + 1), 4),
                "observed_over_null_ratio": round(o / float(nd.mean()), 2) if nd.mean() > 0 else None,
            }
        # gaps against their permutation nulls
        def gap_null(ma, mb):
            o = float(obs[ma].mean() - obs[mb].mean())
            nd = null[:, ma].mean(axis=1) - null[:, mb].mean(axis=1)
            return {"observed": round(o, 2), "null_mean": round(float(nd.mean()), 2),
                    "null_95_range": [round(float(x), 2) for x in np.percentile(nd, [2.5, 97.5])],
                    "excess_gap_observed_minus_null": round(o - float(nd.mean()), 2)}
        perm_gaps = {"gap_shared": gap_null(groups["static_lt2025"], groups["fresh_ge2025"]),
                     "gap_cons": gap_null(groups["2023"], groups["2026"])}
        entry = {"permutation_null": perm_out, "gaps_vs_null": perm_gaps}
        if g in SMALL_QWEN:
            entry["by_year"] = {w: by_year_block(qb, obs, qlist, w) for w in C.WEIGHTINGS}
        out["generators"][g] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=C.B_DEFAULT)
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    questions = C.load_questions("questions.jsonl")
    questions_full = C.load_questions("questions_full.jsonl")
    res = {
        "scope": ("Joint weekly-quiz bootstrap of the shared-boundary gap, the "
                  "conservative 2023-2026 gap and their difference; reproduction of "
                  "stored cross_model.json and fullscale.json gaps; year-level "
                  "answer-permutation nulls for small Qwen rungs."),
        "boundary": "Static = date year < 2025; Fresh = date year >= 2025 (from `date`)",
        "bootstrap": {"B": args.B, "seed_sampled": args.seed, "seed_full": args.seed + 1,
                      "seed_by_year": args.seed + 2,
                      "unit": "weekly quiz id; full quiz set resampled once per draw",
                      "ci": "percentile 2.5/97.5"},
        "weightings": {"question_pooled": "mean over questions",
                       "quiz_weighted": "mean over quizzes of within-quiz means, each "
                       "year/regime using only quizzes that contain questions of that "
                       "year/regime"},
        "caution": ("A difference interval that includes zero does not show the two "
                    "gaps are equal; no equivalence margin is specified."),
    }
    res["a_sampled_878"] = part_a(args, questions)
    sampled_ids = set(questions)
    res["b_full_4176"] = part_b(args, questions_full, sampled_ids)
    res["c_floor_permutation"] = part_c(args, questions)
    p = C.write_json(res, "l1_gap_differences.json")

    for part in ("a_sampled_878",):
        for g, e in res[part]["generators"].items():
            if "skipped" in e:
                continue
            for w in C.WEIGHTINGS:
                r = e[w]
                print(f"(a) {g:14s} {w:15s} shared {r['gap_shared_static_minus_fresh']['estimate']:+6.2f} "
                      f"{r['gap_shared_static_minus_fresh']['ci95']} cons "
                      f"{r['gap_cons_2023_minus_2026']['estimate']:+6.2f} "
                      f"{r['gap_cons_2023_minus_2026']['ci95']} diff "
                      f"{r['diff_gap_shared_minus_gap_cons']['estimate']:+6.2f} "
                      f"{r['diff_gap_shared_minus_gap_cons']['ci95']}"
                      + (f"  repro={e['reproduction_cross_model_json']['static_fresh_match']}/"
                         f"{e['reproduction_cross_model_json']['gap_match']}/"
                         f"{e['reproduction_cross_model_json']['by_year_match']}"
                         if w == "question_pooled" and "reproduction_cross_model_json" in e else ""))
    for g, e in res["b_full_4176"]["generators"].items():
        print(f"(b) {g} reproduction exact: {e['reproduction_fullscale_json']['exact_match']}")
        for w in C.WEIGHTINGS:
            r = e["joint"][w]
            print(f"(b) {g:14s} {w:15s} shared {r['gap_shared_static_minus_fresh']['estimate']:+6.2f} "
                  f"{r['gap_shared_static_minus_fresh']['ci95']} cons "
                  f"{r['gap_cons_2023_minus_2026']['estimate']:+6.2f} "
                  f"{r['gap_cons_2023_minus_2026']['ci95']} diff "
                  f"{r['diff_gap_shared_minus_gap_cons']['estimate']:+6.2f} "
                  f"{r['diff_gap_shared_minus_gap_cons']['ci95']}")
    for g, e in res["c_floor_permutation"]["generators"].items():
        pn = e["permutation_null"]
        print(f"(c) {g:14s} " + "  ".join(
            f"{y}: {pn[y]['observed_f1']:.1f} vs null {pn[y]['null_mean_f1']:.1f} "
            f"(z {pn[y]['z']}; EM {pn[y]['exact_match_pct']:.1f}; strip {pn[y]['echo_stripped_f1']:.1f})"
            for y in ("2022", "2023", "2024", "2025", "2026")))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
