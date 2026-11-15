#!/usr/bin/env python3
"""E5: does the audit change a retrieval-policy decision? (cached outputs only)

Implements the E5 section of results/analysis/PREREG_E2_E3_E6.md exactly. No
generation, no network; every score comes from cached predictions in
data/processed/generations_<model>/{closed_book,hybrid,hybrid_masked}.jsonl.

Candidate policies (information available before answering: question date):
  CB          always closed-book
  RAG         always masked Hybrid (hybrid_masked)
  DATE_b      closed-book if question date < b, masked Hybrid otherwise,
              b in {2022-07-01, ..., 2026-01-01} (8 values)

Procedures:
  naive   choose between CB and RAG by dev-fold mean F1
  audit   choose among CB, RAG and all DATE_b by dev-fold mean F1
Selection uses the QUIZ-WEIGHTED dev mean (mean over dev quizzes of within-quiz
question means). Exact ties go to the simpler policy in the order
CB, RAG, DATE_b by increasing b. Ties are evaluated exactly: each question's
token F1 (a rational 200*n_common/(len_pred+len_gold)) is recovered as a
Fraction from the float returned by metrics.token_f1, so float round-off can
neither create nor break a tie.

Validation: 20 repeats of 5-fold cross-fitting grouped by weekly quiz. Repeat r
shuffles the sorted quiz-id list with random.Random(20260917 + r) and cuts it
into 5 contiguous folds of near-equal size (sizes 40/40/40/40/39 for 199
quizzes). For each fold the procedures are fitted on the other 4 folds and
applied to the held-out fold's questions. Each question's cross-fitted test F1
is the mean over the 20 repeats.

Endpoint (primary): per-question cross-fitted F1(audit) - F1(naive). Quiz-
weighted estimate is primary, question-pooled is reported alongside; 95% CI from
a paired weekly-quiz bootstrap (B=5000, seed 20260917,
analysis_common.QuizBoot), two-sided plus-one bootstrap p, Holm across the nine
generators on the quiz-weighted p.

Secondary (leakage-audit variant): the naive procedure with RAG scored on
unmasked `hybrid` versus the same procedure with RAG scored on `hybrid_masked`.
Readout: whether the CB-vs-RAG choice changes (selection counts, fold-fits that
differ) and the paired cross-fitted F1 difference (masked-scored minus
unmasked-scored), labelled secondary.

Also reported (descriptive): full-sample mean F1 of every candidate policy, the
per-question oracle max(CB, RAG) as a ceiling that is NOT an achievable policy,
Static/Fresh subgroup results (harm check: Static F1 of the chosen procedure vs
CB), and retrieval call rate as the cost proxy. Latency is not claimed.

Writes results/analysis/e5_decision.json and results/analysis/e5_decision.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as C  # noqa: E402  (puts pipeline/ on sys.path)
from routing_baselines import holm  # noqa: E402

GENERATORS = ["qwen1_5b", "qwen3b", "qwen7b", "qwen32b", "qwen72b",
              "mistral_large", "llama3_8b", "gpt4omini", "sonnet"]
BOUNDS = ["2022-07-01", "2023-01-01", "2023-07-01", "2024-01-01",
          "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"]
POLICIES = ["CB", "RAG"] + [f"DATE_{b}" for b in BOUNDS]   # tie-break order
NAIVE = ["CB", "RAG"]
AUDIT = POLICIES
LEAK_NAIVE = ["CB", "RAG_UNMASKED"]
SEED = 20260917
REPEATS = 20
FOLDS = 5
ALPHA = 0.05
N_EXPECTED = 878


# --------------------------------------------------------------------------
def exact_f1(v: float) -> Fraction:
    """Recover the exact rational F1 (0-100 scale) from its float value."""
    fr = Fraction(v).limit_denominator(10 ** 5)
    if abs(float(fr) - v) > 1e-9:
        raise AssertionError(f"could not recover an exact rational for {v!r}")
    return fr


def date_of(q: dict) -> str:
    return str(q["date"])[:10]


def retrieves(policy: str, date: str) -> bool:
    if policy == "CB":
        return False
    if policy in ("RAG", "RAG_UNMASKED"):
        return True
    return date >= policy[len("DATE_"):]


def folds_for(quizzes: list[str], repeat: int) -> list[list[str]]:
    shuf = sorted(quizzes)
    random.Random(SEED + repeat).shuffle(shuf)
    n = len(shuf)
    sizes = [n // FOLDS + (1 if i < n % FOLDS else 0) for i in range(FOLDS)]
    out, start = [], 0
    for s in sizes:
        out.append(shuf[start:start + s])
        start += s
    return out


def select(cands: list[str], dev: list[str], qmean: dict) -> tuple[str, int]:
    """Policy maximising the quiz-weighted dev mean; exact ties -> earliest in
    `cands` (the prespecified simplicity order). Returns (policy, n_at_max).
    All candidates share the same dev quizzes, so comparing sums of quiz means
    is equivalent to comparing quiz-weighted means."""
    vals = [(p, sum((qmean[p][z] for z in dev), Fraction(0))) for p in cands]
    best_p, best_v = vals[0]
    for p, v in vals[1:]:
        if v > best_v:
            best_p, best_v = p, v
    return best_p, sum(1 for _, v in vals if v == best_v)


def holm_step_ci(draws, rank: int, m: int, nd: int = 2):
    """Percentile CI at the Holm step level alpha/(m - rank)."""
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    a = ALPHA / (m - rank)
    lo, hi = np.percentile(d, [100 * a / 2, 100 * (1 - a / 2)])
    return {"level": round(1 - a, 5), "ci": [round(float(lo), nd), round(float(hi), nd)]}


def freq(counter: dict, total: int) -> dict:
    return {p: {"n": n, "share": round(n / total, 3)}
            for p, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))}


# --------------------------------------------------------------------------
def analyse_generator(model: str, questions: dict, qids: list[str],
                      boot: C.QuizBoot) -> dict:
    rows = {c: C.load_rows(model, c) for c in ("closed_book", "hybrid", "hybrid_masked")}
    for c, r in rows.items():
        missing = [q for q in qids if q not in r]
        if missing:
            raise AssertionError(f"{model}/{c}: {len(missing)} questions missing")
    empty = {c: sum(1 for q in qids if not (rows[c][q].get("prediction") or "").strip())
             for c in rows}
    raw = {c: C.f1_scores(rows[c], questions) for c in rows}
    dates = {q: date_of(questions[q]) for q in qids}
    # Exact rationals for every per-question F1; all floats below are derived
    # from them, so identical actions give exactly identical scores.
    ex = {c: {q: exact_f1(raw[c][q]) for q in qids} for c in raw}
    fl = {c: {q: float(ex[c][q]) for q in qids} for c in raw}
    f_cb, f_rag, f_unm = fl["closed_book"], fl["hybrid_masked"], fl["hybrid"]

    def cond_of(p: str, q: str) -> str:
        if p == "RAG_UNMASKED":
            return "hybrid"
        return "hybrid_masked" if retrieves(p, dates[q]) else "closed_book"

    def score(p: str, q: str) -> float:
        return fl[cond_of(p, q)][q]

    def ex_score(p: str, q: str) -> Fraction:
        return ex[cond_of(p, q)][q]

    quizzes = sorted({C.quiz_id(q) for q in qids})
    by_quiz = {z: [q for q in qids if C.quiz_id(q) == z] for z in quizzes}

    qmean = {p: {z: sum((ex_score(p, q) for q in by_quiz[z]), Fraction(0)) / len(by_quiz[z])
                 for z in quizzes}
             for p in POLICIES + ["RAG_UNMASKED"]}

    # ---- cross-fitting ---------------------------------------------------
    procs = ("naive", "audit", "naive_unmasked")
    cands = {"naive": NAIVE, "audit": AUDIT, "naive_unmasked": LEAK_NAIVE}
    acc = {k: {q: Fraction(0) for q in qids} for k in procs}
    ret = {k: {q: 0.0 for q in qids} for k in procs}
    counts = {k: {} for k in procs}
    ties = {k: 0 for k in procs}
    differ_audit_naive = 0
    action_disagree = 0          # held-out (question, repeat) pairs where
                                 # audit and naive differ in retrieve/not
    differ_leak = 0
    per_repeat_qw = []
    fits = []
    for r in range(REPEATS):
        rep_scores = {k: {} for k in ("naive", "audit")}
        for i, fold in enumerate(folds_for(quizzes, r)):
            fold_set = set(fold)
            dev = [z for z in quizzes if z not in fold_set]
            chosen = {}
            for k in procs:
                p, n_max = select(cands[k], dev, qmean)
                chosen[k] = p
                counts[k][p] = counts[k].get(p, 0) + 1
                ties[k] += n_max > 1
            # the leakage comparison asks whether RAG vs CB flips
            leak_flip = (chosen["naive"] == "RAG") != (chosen["naive_unmasked"] == "RAG_UNMASKED")
            differ_leak += leak_flip
            differ_audit_naive += chosen["audit"] != chosen["naive"]
            fits.append({"repeat": r, "fold": i, "n_test_quizzes": len(fold), **chosen})
            for z in fold:
                for q in by_quiz[z]:
                    for k in procs:
                        acc[k][q] += ex_score(chosen[k], q)
                        ret[k][q] += retrieves(chosen[k], dates[q])
                    action_disagree += (retrieves(chosen["audit"], dates[q])
                                        != retrieves(chosen["naive"], dates[q]))
                    for k in ("naive", "audit"):
                        rep_scores[k][q] = score(chosen[k], q)
        d_rep = [rep_scores["audit"][q] - rep_scores["naive"][q] for q in qids]
        per_repeat_qw.append(boot.quizw(d_rep, [True] * len(qids))[0])
    cf = {k: np.array([float(acc[k][q] / REPEATS) for q in qids]) for k in procs}
    cr = {k: np.array([ret[k][q] / REPEATS for q in qids]) for k in procs}
    n_fits = REPEATS * FOLDS

    allm = np.ones(len(qids), dtype=bool)
    static = np.array([not C.is_fresh(questions[q]) for q in qids])
    fresh = ~static
    v_cb = np.array([f_cb[q] for q in qids])
    v_rag = np.array([f_rag[q] for q in qids])
    v_unm = np.array([f_unm[q] for q in qids])

    def both(v, mask=allm, nd=2):
        return {w: C.summarize(*boot.est(w, v, mask), nd=nd) for w in C.WEIGHTINGS}

    def point(v, mask=allm, nd=3):
        return {w: round(boot.est(w, v, mask)[0], nd) for w in C.WEIGHTINGS}

    diff = cf["audit"] - cf["naive"]
    obs_qw, draws_qw = boot.quizw(diff, allm)

    # ---- full-sample descriptive ----------------------------------------
    full = {}
    for p in POLICIES + ["RAG_UNMASKED"]:
        v = np.array([score(p, q) for q in qids])
        full[p] = {**point(v),
                   "retrieval_call_rate": round(float(np.mean(
                       [retrieves(p, dates[q]) for q in qids])), 3)}
    full["RAG_UNMASKED"]["note"] = ("unmasked hybrid; candidate only in the "
                                    "secondary leakage variant")
    oracle = np.maximum(v_cb, v_rag)
    in_sample = {k: select(cands[k], quizzes, qmean)[0] for k in procs}

    # ---- subgroups --------------------------------------------------------
    sub = {}
    for name, mask in (("static", static), ("fresh", fresh)):
        sub[name] = {
            **boot.counts(mask),
            "mean_f1": {"CB": point(v_cb, mask), "RAG": point(v_rag, mask),
                        "naive_crossfit": point(cf["naive"], mask),
                        "audit_crossfit": point(cf["audit"], mask)},
            "retrieval_call_rate": {"naive": round(float(cr["naive"][mask].mean()), 3),
                                    "audit": round(float(cr["audit"][mask].mean()), 3)},
            "paired": {"audit_minus_naive": both(cf["audit"] - cf["naive"], mask),
                       "naive_minus_CB": both(cf["naive"] - v_cb, mask),
                       "audit_minus_CB": both(cf["audit"] - v_cb, mask)},
        }
    sub["static"]["harm_check"] = (
        "Static F1 of each chosen procedure minus always-closed-book on the same "
        "Static questions (negative = retrieval-policy harm on Static)")

    leak_diff = cf["naive"] - cf["naive_unmasked"]
    return {
        "n_questions": len(qids), "n_quizzes": len(quizzes),
        "empty_predictions": empty,
        "full_sample_descriptive": {
            "label": "in-sample means over all 878 questions; descriptive, not held-out",
            "policies": full,
            "oracle_max_CB_RAG": {**point(oracle),
                                  "label": ("per-question max(CB, masked RAG): a "
                                            "descriptive ceiling that uses the "
                                            "answer; NOT an achievable policy")},
            "in_sample_selection": in_sample,
        },
        "cross_fitted": {
            "selection_counts": {k: freq(counts[k], n_fits) for k in procs},
            "fits_decided_by_exact_tie": ties,
            "fits_where_audit_choice_differs_from_naive": differ_audit_naive,
            "heldout_action_disagreement": {
                "n": action_disagree, "of": len(qids) * REPEATS,
                "rate": round(action_disagree / (len(qids) * REPEATS), 4),
                "note": ("held-out (question, repeat) instances where the audit "
                         "and naive choices differ in whether retrieval is used; "
                         "0 means the selections differ at most in label")},
            "n_fits": n_fits,
            "test_f1": {"naive": both(cf["naive"]), "audit": both(cf["audit"]),
                        "CB_fixed": both(v_cb), "RAG_fixed": both(v_rag)},
            "retrieval_call_rate": {k: round(float(cr[k].mean()), 3)
                                    for k in ("naive", "audit")},
            "per_repeat_quiz_weighted_diff": {
                "min": round(min(per_repeat_qw), 3),
                "median": round(statistics.median(per_repeat_qw), 3),
                "max": round(max(per_repeat_qw), 3)},
        },
        "primary": {
            "estimand": "cross-fitted F1(audit) - F1(naive), mean over 20 repeats per question",
            **both(diff),
            "all_selections_identical": differ_audit_naive == 0,
        },
        "_draws_qw": draws_qw,
        "_obs_qw": obs_qw,
        "subgroups": sub,
        "leakage_variant": {
            "label": "secondary",
            "question": "Does the admissibility correction change the CB-vs-RAG choice?",
            "selection_counts_masked": freq(counts["naive"], n_fits),
            "selection_counts_unmasked": freq(counts["naive_unmasked"], n_fits),
            "fits_where_CB_vs_RAG_choice_differs": differ_leak,
            "n_fits": n_fits,
            "in_sample_selection": {"masked": in_sample["naive"],
                                    "unmasked": in_sample["naive_unmasked"]},
            "test_f1": {"naive_masked": both(cf["naive"]),
                        "naive_unmasked": both(cf["naive_unmasked"])},
            "retrieval_call_rate": {"masked": round(float(cr["naive"].mean()), 3),
                                    "unmasked": round(float(cr["naive_unmasked"].mean()), 3)},
            "masked_minus_unmasked": {
                **both(leak_diff),
                "note": ("cross-fitted F1 of the naive procedure run on masked "
                         "vs unmasked RAG outputs; unmasked outputs may use "
                         "post-question evidence, so this is not a deployable "
                         "utility contrast"),
            },
        },
        "fits": fits,
    }


def verdict(rec: dict) -> str:
    p = rec["primary"]
    est = p["quiz_weighted"]["estimate"]
    if p["all_selections_identical"]:
        return "decision unchanged (identical selections in every fit)"
    if p["p_holm"] < ALPHA:
        return ("audit changes the decision and improves held-out utility"
                if est > 0 else "failure: resolved negative difference")
    if est < 0:
        return "failure: negative point estimate, not resolved after Holm"
    if rec["cross_fitted"]["heldout_action_disagreement"]["n"] == 0:
        return ("decision unchanged or not resolved: selections differ in label "
                "only, every held-out action identical")
    return "decision unchanged or not resolved after Holm"


def fmt_counts(c: dict, n: int) -> str:
    return ", ".join(f"{p} {v['n']}/{n}" for p, v in c.items())


def fmt_ci(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['estimate']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def write_md(out: dict, path: Path) -> None:
    L = ["# E5: does the audit change a retrieval-policy decision?", "",
         "Cached outputs only; 878 questions, 199 weekly quizzes; 20 x 5-fold "
         "quiz-grouped cross-fitting (100 fits per generator). Token F1, 0-100. "
         "Primary = quiz-weighted cross-fitted F1(audit) - F1(naive), 95% paired "
         "quiz bootstrap CI (B=5000), Holm across 9 generators. Prereg: "
         "`results/analysis/PREREG_E2_E3_E6.md` (E5).", "",
         "## Primary", "",
         "| generator | naive picks | audit picks | naive F1 | audit F1 | "
         "diff QW [95% CI] | p | Holm p | diff pooled [95% CI] | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, rec in out["generators"].items():
        cfit = rec["cross_fitted"]
        n = cfit["n_fits"]
        pr = rec["primary"]
        L.append("| {m} | {nv} | {au} | {fn:.2f} | {fa:.2f} | {dq} | {p:.4f} | {ph:.4f} | {dp} | {v} |".format(
            m=m, nv=fmt_counts(cfit["selection_counts"]["naive"], n),
            au=fmt_counts(cfit["selection_counts"]["audit"], n),
            fn=cfit["test_f1"]["naive"]["quiz_weighted"]["estimate"],
            fa=cfit["test_f1"]["audit"]["quiz_weighted"]["estimate"],
            dq=fmt_ci(pr["quiz_weighted"]), p=pr["quiz_weighted"]["p_two_sided_boot"],
            ph=pr["p_holm"], dp=fmt_ci(pr["question_pooled"]), v=rec["verdict"]))
    L += ["", "## Static harm check and cost (cross-fitted, quiz-weighted)", "",
          "| generator | Static CB | Static naive | Static audit | audit-CB Static [95% CI] | "
          "Fresh RAG | Fresh naive | Fresh audit | call rate naive | call rate audit |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, rec in out["generators"].items():
        s, fr = rec["subgroups"]["static"], rec["subgroups"]["fresh"]
        cr = rec["cross_fitted"]["retrieval_call_rate"]
        L.append("| {m} | {a:.2f} | {b:.2f} | {c:.2f} | {d} | {e:.2f} | {f:.2f} | {g:.2f} | {h:.3f} | {i:.3f} |".format(
            m=m, a=s["mean_f1"]["CB"]["quiz_weighted"],
            b=s["mean_f1"]["naive_crossfit"]["quiz_weighted"],
            c=s["mean_f1"]["audit_crossfit"]["quiz_weighted"],
            d=fmt_ci(s["paired"]["audit_minus_CB"]["quiz_weighted"]),
            e=fr["mean_f1"]["RAG"]["quiz_weighted"],
            f=fr["mean_f1"]["naive_crossfit"]["quiz_weighted"],
            g=fr["mean_f1"]["audit_crossfit"]["quiz_weighted"],
            h=cr["naive"], i=cr["audit"]))
    L += ["", "## Secondary: leakage-audit variant (naive procedure, unmasked vs masked RAG)", "",
          "| generator | picks (unmasked) | picks (masked) | fits where CB-vs-RAG flips | "
          "masked - unmasked QW [95% CI] |", "|---|---|---|---|---|"]
    for m, rec in out["generators"].items():
        lv = rec["leakage_variant"]
        n = lv["n_fits"]
        L.append(f"| {m} | {fmt_counts(lv['selection_counts_unmasked'], n)} | "
                 f"{fmt_counts(lv['selection_counts_masked'], n)} | "
                 f"{lv['fits_where_CB_vs_RAG_choice_differs']}/{n} | "
                 f"{fmt_ci(lv['masked_minus_unmasked']['quiz_weighted'])} |")
    L += ["", "## Descriptive (full sample, not held out)", "",
          "| generator | " + " | ".join(POLICIES) + " | oracle max(CB,RAG)* |",
          "|---|" + "---|" * (len(POLICIES) + 1)]
    for m, rec in out["generators"].items():
        fs = rec["full_sample_descriptive"]
        L.append(f"| {m} | " + " | ".join(
            f"{fs['policies'][p]['quiz_weighted']:.2f}" for p in POLICIES)
            + f" | {fs['oracle_max_CB_RAG']['quiz_weighted']:.2f} |")
    L += ["", "*Per-question oracle uses the answer: a descriptive ceiling, not an "
          "achievable policy. Quiz-weighted means.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-B", type=int, default=C.B_DEFAULT)
    args = ap.parse_args()

    questions = C.load_questions()
    qids = sorted(questions)
    if len(qids) != N_EXPECTED:
        raise AssertionError(f"expected {N_EXPECTED} questions, found {len(qids)}")
    boot = C.QuizBoot(qids, B=args.B, seed=SEED)

    out = {
        "experiment": "E5",
        "prereg": "results/analysis/PREREG_E2_E3_E6.md#e5",
        "design": {
            "generators": GENERATORS,
            "policies": {"CB": "always closed-book", "RAG": "always masked Hybrid",
                         **{f"DATE_{b}": f"closed-book if question date < {b}, "
                            "masked Hybrid otherwise" for b in BOUNDS}},
            "naive_candidates": NAIVE, "audit_candidates": AUDIT,
            "selection": ("quiz-weighted dev mean token F1; exact ties (rational "
                          "arithmetic) broken toward CB, RAG, DATE_b by increasing b"),
            "cross_fitting": (f"{REPEATS} repeats x {FOLDS} folds over sorted quiz "
                              f"ids shuffled with random.Random({SEED} + repeat), "
                              "contiguous near-equal folds; per-question test F1 "
                              "averaged over repeats"),
            "bootstrap": (f"paired weekly-quiz bootstrap, B={args.B}, "
                          f"numpy default_rng({SEED}) (analysis_common.QuizBoot); "
                          "two-sided plus-one p; Holm across 9 generators on the "
                          "quiz-weighted p"),
            "decision_rule": ("positive quiz-weighted difference with Holm p < 0.05 "
                              "(equivalently the Holm step-level CI excludes 0) = "
                              "audit changes the decision and improves held-out "
                              "utility; identical selections or unresolved = "
                              "decision unchanged or not resolved; negative = failure"),
            "static_fresh": "Static = date year < 2025, Fresh = year >= 2025",
            "endpoint": "token F1 (pipeline/metrics.py), 0-100",
            "scope": "internal validation, not a prospective holdout; latency not claimed",
        },
        "generators": {},
    }

    for m in GENERATORS:
        out["generators"][m] = analyse_generator(m, questions, qids, boot)
        print(f"done {m}", flush=True)

    # ---- Holm across generators on the primary (quiz-weighted) p ---------
    ms = list(out["generators"])
    ps = [out["generators"][m]["primary"]["quiz_weighted"]["p_two_sided_boot"] for m in ms]
    adj = holm(ps)
    order = sorted(range(len(ms)), key=lambda i: ps[i])
    for rank, i in enumerate(order):
        rec = out["generators"][ms[i]]
        rec["primary"]["p_holm"] = adj[i]
        rec["primary"]["holm_step_ci_quiz_weighted"] = holm_step_ci(
            rec["_draws_qw"], rank, len(ms))
    lps = [out["generators"][m]["leakage_variant"]["masked_minus_unmasked"]
           ["quiz_weighted"]["p_two_sided_boot"] for m in ms]
    for m, a in zip(ms, holm(lps)):
        out["generators"][m]["leakage_variant"]["masked_minus_unmasked"][
            "quiz_weighted"]["p_holm_within_secondary"] = a
    for m in ms:
        rec = out["generators"][m]
        rec.pop("_draws_qw")
        rec.pop("_obs_qw")
        rec["verdict"] = verdict(rec)

    out["summary"] = {
        "n_generators": len(ms),
        "n_all_selections_identical": sum(
            out["generators"][m]["primary"]["all_selections_identical"] for m in ms),
        "n_positive_holm_significant": sum(
            1 for m in ms if out["generators"][m]["verdict"].startswith("audit changes")),
        "n_negative_point_estimate": sum(
            1 for m in ms if out["generators"][m]["primary"]["quiz_weighted"]["estimate"] < 0),
        "verdicts": {m: out["generators"][m]["verdict"] for m in ms},
        "leakage_variant_fits_flipping": {
            m: out["generators"][m]["leakage_variant"]["fits_where_CB_vs_RAG_choice_differs"]
            for m in ms},
    }

    p = C.write_json(out, "e5_decision.json")
    write_md(out, C.OUT / "e5_decision.md")
    print(f"wrote {p}")
    print(f"wrote {C.OUT / 'e5_decision.md'}")
    for m in ms:
        rec = out["generators"][m]
        pr = rec["primary"]
        cfit = rec["cross_fitted"]
        print(f"{m:14s} naive[{fmt_counts(cfit['selection_counts']['naive'], 100)}] "
              f"audit[{fmt_counts(cfit['selection_counts']['audit'], 100)}] "
              f"QW {fmt_ci(pr['quiz_weighted'])} p={pr['quiz_weighted']['p_two_sided_boot']} "
              f"holm={pr['p_holm']} pooled {fmt_ci(pr['question_pooled'])} -> {rec['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
