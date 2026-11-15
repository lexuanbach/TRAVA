#!/usr/bin/env python3
"""The temporal-mask counterfactual, on the same questions.

WHAT THIS REPLACES. The paper withdrew its Static-harm result using the subset
whose ORIGINAL contexts happened to be admissible -- 179 of 404 Static questions
against 398 of 474 Fresh. That compares different questions before and after,
so it diagnoses a problem rather than correcting for it, and it is
right that the causal-sounding wording ("most of the harm was evidence from the
future") is not identified by it.

This runs the actual counterfactual: the SAME questions and generators, with
contexts re-ranked under publish_date <= question_date + 3 days, so every
comparison is paired within question.

  Delta_mask = F1(masked RAG)   - F1(unmasked RAG)     paired, same question
  Delta_RAG  = F1(masked RAG)   - F1(closed-book)      paired, same question

Both are quiz-clustered. The support interaction is then recomputed on masked
contexts, because the sharper point is that this result
was still being computed from the contaminated rankings.

COVERAGE. Eight of the nine Hybrid-panel generators are re-runnable: seven
local, plus Claude Sonnet through the API. GPT-4o-mini is not (no OpenAI key),
and the script reports it as absent rather than silently dropping it. Every
per-model record carries its own question count and a completeness flag, so a
short run can never be read as though it were the full panel.

Outputs results/masked_counterfactual.json.
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
sys.dont_write_bytecode = True
from metrics import gold_in_text, token_f1                       # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
BOUNDARY = 2025
PANEL = ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
         "mistral_large", "", "sonnet", "gpt4omini"]        # "" is Haiku
# Both API generators are now regenerated under the mask: Sonnet once Anthropic
# credit was restored, GPT-4o-mini once an OpenAI key was available. Nothing in
# the Hybrid panel is now excluded for want of access, so this list is empty and
# the coverage block says so rather than implying an untested gap.
API_ONLY = []
LADDER_B = {"qwen1_5b": 1.5, "qwen3b": 3.0, "qwen7b": 7.0, "qwen32b": 32.0,
            "qwen72b": 72.0, "mistral_large": 123.0}


def quiz_of(q):
    return q.split("_")[0]


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in
             (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}


def load(model, cond):
    # Haiku lives in the bare generations/ directory, matching aggregate.py.
    # A f"generations_{model}" path silently misses it and drops the model.
    f = PROC / (f"generations_{model}" if model else "generations") / f"{cond}.jsonl"
    if not f.is_file():
        return {}
    return {r["qid"]: r for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def _summarise(obs, draws):
    draws = sorted(draws)
    n = len(draws)
    le = sum(1 for v in draws if v <= 0)
    return {"delta": round(obs, 3),
            "ci95": [round(draws[int(0.025 * n)], 3), round(draws[int(0.975 * n)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 5)}


def paired(per_q, B, seed):
    """Quiz-clustered bootstrap of a within-question difference.

    Reported under BOTH weightings, because they are different estimands and
    the difference is not small. Averaging questions weights a 12-question quiz
    six times as heavily as a 2-question one; averaging quiz means weights every
    week equally. On Sonnet's Static slice the same contrast is -4.97 pooled
    over questions and -7.92 over quiz means, and only the second excludes zero.

    The quiz-weighted value is primary: it matches the independent unit the
    paper argues for throughout, and it is what pipeline/retrieval_cluster_ci.py
    already reports, so the two analyses are directly comparable. Reporting only
    one of them is exactly the kind of undisclosed analytic choice this paper
    audits, so both travel together.
    """
    by = collections.defaultdict(list)
    for q, v in per_q:
        by[q].append(v)
    cl = list(by.values())
    if len(cl) < 3:
        return None
    flat = [v for c in cl for v in c]
    obs_item = statistics.fmean(flat)
    obs_quiz = statistics.fmean([statistics.fmean(c) for c in cl])
    rng = random.Random(seed)
    d_item, d_quiz = [], []
    for _ in range(B):
        idx = [rng.randrange(len(cl)) for _ in cl]
        d_item.append(statistics.fmean([v for i in idx for v in cl[i]]))
        d_quiz.append(statistics.fmean([statistics.fmean(cl[i]) for i in idx]))
    out = _summarise(obs_quiz, d_quiz)
    out["question_weighted"] = _summarise(obs_item, d_item)
    out["weighting"] = "quiz-mean (primary); question-pooled under question_weighted"
    out["n_questions"] = len(flat)
    out["n_quizzes"] = len(cl)
    return out


def interaction_ci(units, B, seed):
    """Quiz-clustered bootstrap of (mean supported gain - mean unsupported gain).

    The masked support interaction needs uncertainty, which was
    previously reported as two bare means. Resampling quizzes -- not questions --
    keeps the two arms coupled: a quiz contributes all of its supported and all
    of its unsupported questions or none of them, so the interval reflects the
    fact that a whole weekly quiz can be easy or hard.
    """
    cl = [(a, b) for a, b in units.values()]
    if len(cl) < 3:
        return None
    sup = [v for a, _ in cl for v in a]
    uns = [v for _, b in cl for v in b]
    if len(sup) < 10 or len(uns) < 10:
        return None
    obs = statistics.fmean(sup) - statistics.fmean(uns)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        sa, sb = [], []
        for _ in cl:
            a, b = cl[rng.randrange(len(cl))]
            sa.extend(a); sb.extend(b)
        if sa and sb:
            draws.append(statistics.fmean(sa) - statistics.fmean(sb))
    if len(draws) < B // 2:
        return None
    draws.sort()
    n = len(draws)
    le = sum(1 for v in draws if v <= 0)
    return {"interaction": round(obs, 3),
            "ci95": [round(draws[int(0.025 * n)], 3), round(draws[int(0.975 * n)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 5),
            "n_quizzes": len(cl)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--output", type=Path, default=RES / "masked_counterfactual.json")
    ap.add_argument("--include-partial", action="store_true",
                    help="fold short runs into the panel as diagnostics "
                         "(default: exclude them and name them under coverage)")
    args = ap.parse_args()

    qs = load_questions()
    corpus = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    corpus.setdefault(d["doc_id"], d.get("text", ""))

    out = {
        "scope": "Same-question counterfactual: Hybrid retrieval re-ranked under "
                 "the per-query temporal mask, against the shipped rankings.",
        "estimands": {
            "delta_mask": "F1(masked RAG) - F1(unmasked RAG), paired within question",
            "delta_rag_masked": "F1(masked RAG) - F1(closed-book), paired within question",
            "delta_rag_unmasked": "F1(unmasked RAG) - F1(closed-book), paired within question",
        },
        "why": ("The selected-subset analysis compared different questions "
                "(179/404 Static retained against 398/474 Fresh) and so could "
                "not identify the mask's effect. These contrasts are within "
                "question."),
        "coverage": {"regenerated": [], "api_only_not_regenerated": API_ONLY},
        "per_model": {},
    }

    out["coverage"]["incomplete_excluded"] = []
    for m in PANEL:
        cb, un, ma = load(m, "closed_book"), load(m, "hybrid"), load(m, "hybrid_masked")
        if not ma:
            continue
        # A run short of the full question set is a development diagnostic, not a
        # panel member. Folding partial caches into the inferential
        # panel and let a still-growing largest rung carry the scale slope; the
        # frozen panel is complete runs only, and the rest are named here.
        if not args.include_partial and len(ma) < len(qs):
            out["coverage"]["incomplete_excluded"].append(
                {"model": m, "n_masked": len(ma), "n_questions": len(qs)})
            continue
        out["coverage"]["regenerated"].append(m)
        common = sorted(set(cb) & set(un) & set(ma) & set(qs))
        rec = {"n_questions": len(common)}
        # A run cut short leaves a prefix of questions.jsonl. That file is not
        # date-ordered, so the prefix is close to representative, but it is a
        # different question set from the complete models and must be labelled
        # rather than compared as though it were the same panel.
        if len(ma) < len(qs):
            rec["INCOMPLETE"] = True
            rec["coverage_fraction"] = round(len(ma) / len(qs), 3)
            static_share = sum(1 for q in common if qs[q]["year"] < BOUNDARY) / max(len(common), 1)
            rec["static_share"] = round(static_share, 3)
        f = lambda r, q: 100.0 * token_f1(r[q]["prediction"] or "", qs[q]["gold_answer"])
        for regime in ("all", "static", "fresh"):
            sel = [q for q in common
                   if regime == "all"
                   or ((qs[q]["year"] >= BOUNDARY) == (regime == "fresh"))]
            if len(sel) < 20:
                continue
            rec[regime] = {
                "delta_mask": paired([(quiz_of(q), f(ma, q) - f(un, q)) for q in sel],
                                     args.B, args.seed),
                "delta_rag_masked": paired([(quiz_of(q), f(ma, q) - f(cb, q)) for q in sel],
                                           args.B, args.seed),
                "delta_rag_unmasked": paired([(quiz_of(q), f(un, q) - f(cb, q)) for q in sel],
                                             args.B, args.seed),
                "n": len(sel),
            }
        # how often did the mask actually change what the model saw?
        changed = sum(1 for q in common
                      if (ma[q].get("doc_ids") or []) != (un[q].get("doc_ids") or []))
        rec["context_set_changed"] = changed
        rec["pct_context_set_changed"] = round(100 * changed / max(len(common), 1), 1)
        out["per_model"][m] = rec

    # ---- does the Static harm survive the mask? --------------------------
    harmed = {m: r["static"]["delta_rag_masked"] for m, r in out["per_model"].items()
              if "static" in r}
    out["static_harm_under_mask"] = {
        "per_model": {m: {"delta": v["delta"], "ci95": v["ci95"]}
                      for m, v in harmed.items()},
        "n_harmed_ci_excludes_zero": sum(1 for v in harmed.values() if v["ci95"][1] < 0),
        "n_models": len(harmed),
    }

    # ---- support interaction, recomputed on MASKED contexts --------------
    nli = {}
    f = RES / "nli_support_labels.jsonl"
    if f.is_file():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                nli.setdefault(r["qid"], False)
                nli[r["qid"]] |= r["nli_label"].startswith("entail")

    inter = {}
    for m in out["coverage"]["regenerated"]:
        cb, ma = load(m, "closed_book"), load(m, "hybrid_masked")
        common = sorted(set(cb) & set(ma) & set(qs))
        for label in ("string", "nli"):
            units = collections.defaultdict(lambda: ([], []))
            for q in common:
                if qs[q]["year"] >= BOUNDARY:
                    continue                     # Static only, as in the paper
                if label == "nli":
                    sup = nli.get(q)
                    if sup is None:
                        continue
                else:
                    texts = [corpus.get(d, "") for d in (ma[q].get("doc_ids") or [])]
                    sup = any(gold_in_text(qs[q]["gold_answer"], t) for t in texts if t)
                g = 100.0 * (token_f1(ma[q]["prediction"] or "", qs[q]["gold_answer"])
                             - token_f1(cb[q]["prediction"] or "", qs[q]["gold_answer"]))
                units[quiz_of(q)][0 if sup else 1].append(g)
            sup_all = [v for a, _ in units.values() for v in a]
            uns_all = [v for _, b in units.values() for v in b]
            if len(sup_all) < 10 or len(uns_all) < 10:
                continue
            rec_i = {
                "gain_supported": round(statistics.fmean(sup_all), 3),
                "gain_unsupported": round(statistics.fmean(uns_all), 3),
                "interaction": round(statistics.fmean(sup_all) - statistics.fmean(uns_all), 3),
                "n_supported": len(sup_all), "n_unsupported": len(uns_all),
            }
            ci = interaction_ci(units, args.B, args.seed)
            if ci:
                rec_i["ci95"] = ci["ci95"]
                rec_i["p_plus_one"] = ci["p_plus_one"]
                rec_i["n_quizzes"] = ci["n_quizzes"]
            inter.setdefault(label, {})[m] = rec_i
    out["support_interaction_masked"] = inter

    # side-by-side with the unmasked interaction, so mask sensitivity is
    # visible per model rather than only in the slope
    unmasked_ref = {}
    f_si = RES / "support_interaction.json"
    if f_si.is_file():
        d_si = json.loads(f_si.read_text())
        for label, per_m in (d_si.get("labels") or {}).items():
            for m, rec in per_m.items():
                if "static" in rec:
                    unmasked_ref.setdefault(label, {})[m] = {
                        "interaction": rec["static"]["interaction"],
                        "gain_unsupported": rec["static"]["gain_unsupported"]}
    comp = {}
    for label, per_m in inter.items():
        for m, v in per_m.items():
            u = (unmasked_ref.get(label) or {}).get(m)
            if not u:
                continue
            comp.setdefault(label, {})[m] = {
                "interaction_unmasked": u["interaction"],
                "interaction_masked": v["interaction"],
                "shift": round(v["interaction"] - u["interaction"], 3),
                "unsupported_unmasked": u["gain_unsupported"],
                "unsupported_masked": v["gain_unsupported"],
            }
    out["support_interaction_mask_sensitivity"] = comp
    if comp:
        print(f"\n{'model':15s}{'interaction':>13s}{'masked':>9s}{'shift':>8s}   (string, Static)")
        for m, v in (comp.get("string") or {}).items():
            print(f"{m:15s}{v['interaction_unmasked']:13.2f}"
                  f"{v['interaction_masked']:9.2f}{v['shift']:+8.2f}")

    # unsupported-cost slope on masked contexts, over published-count rungs
    slopes = {}
    for label, per in inter.items():
        pts = sorted((LADDER_B[m], v["gain_unsupported"])
                     for m, v in per.items() if m in LADDER_B)
        if len(pts) >= 4:
            x = [math.log10(a) for a, _ in pts]
            y = [b for _, b in pts]
            mx = statistics.fmean(x)
            sxx = sum((a - mx) ** 2 for a in x)
            slope = sum((a - mx) * (b - statistics.fmean(y))
                        for a, b in zip(x, y)) / sxx
            import itertools
            fit = lambda yy: sum((a - mx) * (b - statistics.fmean(yy))
                                 for a, b in zip(x, yy)) / sxx
            ge = tot = 0
            for pm in itertools.permutations(y):
                tot += 1
                ge += abs(fit(list(pm))) >= abs(slope) - 1e-12
            slopes[label] = {"slope_per_decade": round(slope, 3),
                             "p_exact": round(ge / tot, 5), "n_rungs": len(y),
                             "points": [{"params_B": a, "gain_unsupported": b}
                                        for a, b in pts]}
    out["unsupported_cost_vs_scale_masked"] = slopes

    print(f"{'model':15s}{'n':>5s}{'ctx chg':>9s}{'d_mask':>9s}"
          f"{'RAG|mask':>10s}{'RAG|unmask':>12s}   (Static)")
    for m, r in out["per_model"].items():
        s = r.get("static")
        if not s:
            continue
        mark = "*" if r.get("INCOMPLETE") else " "
        print(f"{m:14s}{mark}{r['n_questions']:5d}{r['pct_context_set_changed']:8.1f}%"
              f"{s['delta_mask']['delta']:9.2f}{s['delta_rag_masked']['delta']:10.2f}"
              f"{s['delta_rag_unmasked']['delta']:12.2f}")
    inc = [m for m, r in out["per_model"].items() if r.get("INCOMPLETE")]
    if inc:
        print(f"  * incomplete run, partial question set: {', '.join(inc)}")
    h = out["static_harm_under_mask"]
    print(f"\nStatic harm under the mask: {h['n_harmed_ci_excludes_zero']} of "
          f"{h['n_models']} generators have a negative interval")
    for label, s in slopes.items():
        print(f"unsupported-cost slope, masked, {label}: "
              f"{s['slope_per_decade']:+.2f}/decade p={s['p_exact']} "
              f"({s['n_rungs']} rungs)")
    if not out["coverage"]["regenerated"]:
        print("no masked generations yet; run scripts/run_masked_panel.sh")
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
