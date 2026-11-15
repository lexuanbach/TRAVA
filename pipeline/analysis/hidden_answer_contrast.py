#!/usr/bin/env python3
"""Answer retrieved but not shown: does the retrieval gain follow document-level
retrieval of an answer-bearing article, or the answer being visible?

Every ranked-context generator in this repository shows the model only the first
600 characters of each passage (DOC_CHARS = 600 in local_generate.py,
local_generate_gguf.py, gen_gguf_ranked.py, generate.py, gen_openai.py).
Support labels elsewhere scanned the full stored text. Each question is
classified from the ranked doc_ids stored in the generation row:

  SHOWN  : the gold string (metrics.gold_in_text) occurs in the first 600
           characters of at least one ranked passage
  HIDDEN : not SHOWN, but the gold string occurs somewhere in the full stored
           text of at least one ranked passage
  NONE   : the gold string occurs in no ranked passage

Corpus text: data/processed/corpus.jsonl, with corpus_full.jsonl only for doc
ids absent from corpus.jsonl (support_interaction.load_corpus). For hybrid and
hybrid_masked every ranked doc id is present in corpus.jsonl.

Gain = 100 * (token_f1(rag) - token_f1(closed_book)), per question.
Contrasts: SHOWN - HIDDEN and HIDDEN - NONE, per regime (Static < 2025 <= Fresh,
from `date`) and pooled, with a weekly-quiz bootstrap (B=5000) that recomputes
every class mean inside each draw.

The classes are NOT randomised: where an answer sits in an article, and which
articles are retrieved, depend on the question. The covariate block compares the
classes on year, answer length, gold-string offset and closed-book F1 so the
selection can be judged. This is observational.

Writes results/analysis/hidden_answer_contrast.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as C  # noqa: E402
from metrics import gold_in_text  # noqa: E402
import support_interaction as SI  # noqa: E402

DOC_CHARS = 600
CLASSES = ("SHOWN", "HIDDEN", "NONE")
RAG_CONDS = ("hybrid", "hybrid_masked")
GENERATORS = ["gpt4omini", "llama3_8b", "mistral_large", "qwen1_5b", "qwen32b",
              "qwen3b", "qwen72b", "qwen7b", "sonnet", "haiku"]
PANEL9 = GENERATORS[:9]   # the nine generators globbed by support_interaction.py


def first_visible_offset(gold: str, text: str) -> int | None:
    """Smallest raw-character prefix length k with gold_in_text(gold, text[:k]).

    Binary search; gold_in_text on prefixes is monotone except in rare
    normalisation edge cases, so the result is verified and falls back to a
    linear scan when the search lands on a non-matching prefix."""
    if not gold_in_text(gold, text):
        return None
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if gold_in_text(gold, text[:mid]):
            hi = mid
        else:
            lo = mid + 1
    if gold_in_text(gold, text[:lo]) and not (lo > 1 and gold_in_text(gold, text[:lo - 1])):
        return lo
    for k in range(1, len(text) + 1):
        if gold_in_text(gold, text[:k]):
            return k
    return None


def classify(questions, corpus, rows):
    info = {}
    for qid, r in rows.items():
        q = questions.get(qid)
        if q is None:
            continue
        gold = q["gold_answer"]
        shown_any = full_any = False
        offsets, ranks, n_bearing = [], [], 0
        shown_not_full = False
        for rank, did in enumerate(r.get("doc_ids") or [], 1):
            text = corpus.get(did, "")
            if not text:
                continue
            s = gold_in_text(gold, text[:DOC_CHARS])
            f = gold_in_text(gold, text)
            if s and not f:
                shown_not_full = True
            shown_any |= s
            if f:
                full_any = True
                n_bearing += 1
                ranks.append(rank)
                off = first_visible_offset(gold, text)
                if off is not None:
                    offsets.append(off)
        cls = "SHOWN" if shown_any else ("HIDDEN" if full_any else "NONE")
        info[qid] = {"cls": cls, "min_offset": min(offsets) if offsets else None,
                     "best_rank": min(ranks) if ranks else None,
                     "n_bearing": n_bearing, "shown_not_full": shown_not_full}
    return info


def describe(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "mean": round(float(v.mean()), 2),
            "sd": round(float(v.std(ddof=1)), 2) if len(v) > 1 else None,
            "median": round(float(med), 1), "iqr": [round(float(q1), 1), round(float(q3), 1)]}


def smd(a, b):
    a = np.asarray([x for x in a if x is not None], float)
    b = np.asarray([x for x in b if x is not None], float)
    if len(a) < 2 or len(b) < 2:
        return None
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return round(float((a.mean() - b.mean()) / sp), 3) if sp > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=C.B_DEFAULT)
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()

    questions = C.load_questions()
    corpus = SI.load_corpus()
    qids = sorted(questions)
    qb = C.QuizBoot(qids, B=args.B, seed=args.seed)
    fresh = np.array([C.is_fresh(questions[q]) for q in qids])
    regimes = {"static": ~fresh, "fresh": fresh, "pooled": np.ones(len(qids), bool)}

    out = {
        "scope": ("Retrieval gain by whether an answer-bearing ranked passage exists "
                  "and whether the gold string falls inside the 600-character "
                  "prefix the generator was shown."),
        "definitions": {
            "SHOWN": "gold_in_text(gold, text[:600]) for >=1 ranked passage",
            "HIDDEN": "not SHOWN, but gold_in_text(gold, full stored text) for >=1 ranked passage",
            "NONE": "gold string in no ranked passage",
            "gain": "100*(token_f1(rag) - token_f1(closed_book))",
            "regimes": "static: date year < 2025; fresh: >= 2025; pooled: all 878",
        },
        "bootstrap": {"B": args.B, "seed": args.seed,
                      "unit": "weekly quiz id; full quiz set resampled once per draw, "
                              "all classes/regimes/generators recomputed from the same draw",
                      "ci": "percentile 2.5/97.5"},
        "weightings": {"question_pooled": "mean over questions in the class",
                       "quiz_weighted": "mean over quizzes containing >=1 question of "
                       "the class of the within-quiz class mean; each class has its own "
                       "quiz denominator, so a contrast compares two differently "
                       "composed quiz sets"},
        "not_causal": ("Class membership is not randomised (answer position within an "
                       "article and which articles are retrieved depend on the "
                       "question). See covariates."),
        "conditions": {},
    }

    for cond in RAG_CONDS:
        # classification uses the ranked doc_ids, identical across generators
        ref_rows = None
        per_gen = {}
        cls_info = None
        for g in GENERATORS:
            cb, rg = C.load_rows(g, "closed_book"), C.load_rows(g, cond)
            if not cb or not rg:
                continue
            if cls_info is None:
                cls_info = classify(questions, corpus, rg)
                ref_rows = {q: r.get("doc_ids") for q, r in rg.items()}
            else:
                diffs = sum(1 for q, r in rg.items() if r.get("doc_ids") != ref_rows.get(q))
                assert diffs == 0, f"{g}/{cond}: ranked doc_ids differ from reference"
            assert set(rg) >= set(qids) and set(cb) >= set(qids), f"{g}: incomplete"
            f_rag = C.f1_scores(rg, questions)
            f_cb = C.f1_scores(cb, questions)
            gain = np.array([f_rag[q] - f_cb[q] for q in qids])
            cbv = np.array([f_cb[q] for q in qids])
            ragv = np.array([f_rag[q] for q in qids])
            per_gen[g] = (gain, cbv, ragv)
        if cls_info is None:
            continue
        cls = np.array([cls_info[q]["cls"] for q in qids])
        masks = {c: cls == c for c in CLASSES}

        cond_out = {"generators": list(per_gen), "in_support_interaction_panel":
                    [g for g in per_gen if g in PANEL9],
                    "class_counts": {}, "n_shown_but_not_in_full_text":
                    int(sum(v["shown_not_full"] for v in cls_info.values())),
                    "per_generator": {}, "covariates": {}}
        for reg, rm in regimes.items():
            cond_out["class_counts"][reg] = {c: qb.counts(rm & masks[c]) for c in CLASSES}

        for g, (gain, cbv, ragv) in per_gen.items():
            gout = {}
            for reg, rm in regimes.items():
                rout = {}
                for w in C.WEIGHTINGS:
                    est = {c: qb.est(w, gain, rm & masks[c]) for c in CLASSES}
                    wout = {f"gain_{c}": C.summarize(*est[c], extra=qb.counts(rm & masks[c]))
                            for c in CLASSES}
                    for a, b in (("SHOWN", "HIDDEN"), ("HIDDEN", "NONE")):
                        wout[f"{a}_minus_{b}"] = C.summarize(est[a][0] - est[b][0],
                                                             est[a][1] - est[b][1])
                    rout[w] = wout
                rout["levels_question_pooled"] = {
                    c: {"closed_book_f1": round(float(cbv[rm & masks[c]].mean()), 2)
                        if (rm & masks[c]).any() else None,
                        "rag_f1": round(float(ragv[rm & masks[c]].mean()), 2)
                        if (rm & masks[c]).any() else None}
                    for c in CLASSES}
                gout[reg] = rout
            cond_out["per_generator"][g] = gout

        # covariates (descriptive), pooled over regimes and by regime
        ans_words = np.array([len(questions[q]["gold_answer"].split()) for q in qids])
        years = np.array([C.year_of(questions[q]) for q in qids])
        cb_panel = np.mean([v[1] for v in per_gen.values()], axis=0)
        for reg, rm in regimes.items():
            cov = {}
            for c in CLASSES:
                m = rm & masks[c]
                idx = np.where(m)[0]
                cov[c] = {
                    "n": int(m.sum()),
                    "share_fresh": round(float(fresh[m].mean()), 3) if m.any() else None,
                    "year_counts": {str(y): int((years[m] == y).sum()) for y in sorted(set(years))},
                    "answer_words": describe(ans_words[m]),
                    "gold_first_visible_offset_chars": describe(
                        [cls_info[qids[i]]["min_offset"] for i in idx]),
                    "best_rank_of_answer_bearing_passage": describe(
                        [cls_info[qids[i]]["best_rank"] for i in idx]),
                    "n_answer_bearing_passages_full_text": describe(
                        [cls_info[qids[i]]["n_bearing"] for i in idx]),
                    "closed_book_f1_panel_mean": describe(cb_panel[m]),
                    "closed_book_f1_by_generator": {
                        g: round(float(v[1][m].mean()), 2) if m.any() else None
                        for g, v in per_gen.items()},
                }
            sh, hi = rm & masks["SHOWN"], rm & masks["HIDDEN"]
            cov["standardized_mean_difference_HIDDEN_minus_SHOWN"] = {
                "year": smd(years[hi], years[sh]),
                "answer_words": smd(ans_words[hi], ans_words[sh]),
                "gold_first_visible_offset_chars": smd(
                    [cls_info[qids[i]]["min_offset"] for i in np.where(hi)[0]],
                    [cls_info[qids[i]]["min_offset"] for i in np.where(sh)[0]]),
                "best_rank": smd(
                    [cls_info[qids[i]]["best_rank"] for i in np.where(hi)[0]],
                    [cls_info[qids[i]]["best_rank"] for i in np.where(sh)[0]]),
                "closed_book_f1_panel_mean": smd(cb_panel[hi], cb_panel[sh]),
            }
            cond_out["covariates"][reg] = cov
        cond_out["covariate_notes"] = (
            "gold_first_visible_offset_chars = smallest raw-character prefix length of "
            "any ranked passage whose normalised text contains the normalised gold "
            "string (the end position of the first occurrence); SHOWN <= 600 by "
            "construction, HIDDEN > 600. closed_book_f1_panel_mean averages the "
            "closed-book F1 of the generators listed for this condition. Standardized "
            "mean differences use the pooled SD (HIDDEN minus SHOWN).")
        out["conditions"][cond] = cond_out

        # console summary
        cc = cond_out["class_counts"]["pooled"]
        print(f"\n=== {cond}: SHOWN {cc['SHOWN']['n_questions']} / HIDDEN "
              f"{cc['HIDDEN']['n_questions']} / NONE {cc['NONE']['n_questions']}")
        for g in per_gen:
            for reg in ("static", "fresh", "pooled"):
                for w in C.WEIGHTINGS:
                    r = cond_out["per_generator"][g][reg][w]
                    print(f"  {g:13s} {reg:6s} {w:15s} S {r['gain_SHOWN']['estimate']:+6.1f} "
                          f"H {r['gain_HIDDEN']['estimate']:+6.1f} N {r['gain_NONE']['estimate']:+6.1f} "
                          f"S-H {r['SHOWN_minus_HIDDEN']['estimate']:+6.1f} {r['SHOWN_minus_HIDDEN']['ci95']} "
                          f"H-N {r['HIDDEN_minus_NONE']['estimate']:+6.1f} {r['HIDDEN_minus_NONE']['ci95']}")

    # panel-level summary
    summ = {}
    for cond, co in out["conditions"].items():
        s = {}
        for reg in ("static", "fresh", "pooled"):
            for w in C.WEIGHTINGS:
                sh = [co["per_generator"][g][reg][w]["SHOWN_minus_HIDDEN"] for g in co["generators"]]
                hn = [co["per_generator"][g][reg][w]["HIDDEN_minus_NONE"] for g in co["generators"]]
                s[f"{reg}/{w}"] = {
                    "SHOWN_minus_HIDDEN_range": [min(x["estimate"] for x in sh),
                                                 max(x["estimate"] for x in sh)],
                    "SHOWN_minus_HIDDEN_ci_excludes_zero": sum(bool(x["ci_excludes_zero"]) for x in sh),
                    "HIDDEN_minus_NONE_range": [min(x["estimate"] for x in hn),
                                                max(x["estimate"] for x in hn)],
                    "HIDDEN_minus_NONE_ci_excludes_zero": sum(bool(x["ci_excludes_zero"]) for x in hn),
                    "HIDDEN_minus_NONE_negative_ci_excludes_zero": sum(
                        bool(x["ci_excludes_zero"]) and x["estimate"] < 0 for x in hn),
                    "n_generators": len(sh)}
        summ[cond] = s
    out["summary"] = summ
    p = C.write_json(out, "hidden_answer_contrast.json")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
