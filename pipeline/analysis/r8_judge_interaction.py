#!/usr/bin/env python3
"""R8(3): the support interaction recomputed with the R8 judge label on the
exact shown text, side by side with the string and NLI labels at 600 chars.

ESTIMAND (unchanged from pipeline/support_interaction.py)

    gain(q)      = 100 * (F1(RAG answer) - F1(closed-book answer))
    interaction  = mean gain over SUPPORTED questions
                 - mean gain over UNSUPPORTED questions

SUPPORT LABELS COMPARED, all on the passages the generator actually saw
(`corpus[d]["text"][:600]`, the five Hybrid passages of the condition):

    string600        metrics.gold_in_text(gold, text[:600])          (E0)
    nli600           DeBERTa-v3-large entailment, premise text[:600] (E0 / R1)
    judge600_YES     r8 judge label == YES
    judge600_YESPART r8 judge label in {YES, PARTIAL}

CONDITIONS: unmasked Hybrid top-5 (generations_*/hybrid.jsonl) and masked
Hybrid top-5 (hybrid_masked.jsonl), every generator with a closed-book cache and
that condition. Regimes: static (year < 2025), fresh (year >= 2025) and pooled.
Weightings: question_pooled (primary) and quiz_weighted. One joint weekly-quiz
bootstrap (analysis_common.QuizBoot, B=5000, seed 20260920) drives every label,
model, regime and condition, so all contrasts are jointly resampled.

Side by side with the STORED values of results/analysis/e0_visible_support.json
(string@600, unmasked and masked) and results/analysis/r1_masked_nli.json
(entailment@600 on the masked contexts), which use their own scripts' seeds and
question-pooled estimator; those are reproduced here under their own seeds as a
check before the common-seed recomputation is reported.

Also: how many questions change support arm between labels, per condition and
regime, and which generators' interaction sign/interval changes.

Writes results/analysis/r8_judge_interaction.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

import numpy as np

from _e0util import OUT, PROC, ROOT, dump, load_json, meta, repro_record, window_fn
from analysis_common import QuizBoot, WEIGHTINGS, summarize

import masked_counterfactual as MC                                # noqa: E402
import support_interaction as SI                                  # noqa: E402
from clusters import quiz_id                                      # noqa: E402
from metrics import token_f1                                      # noqa: E402

SEED = 20260920
B_DEFAULT = 5000
VIS = window_fn(600)
JUDGE = OUT / "r8_judge_labels.jsonl"
# every generator with closed_book + hybrid + hybrid_masked ("" is Haiku, which
# lives in the bare generations/ directory; masked_counterfactual.PANEL order)
PANEL = ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
         "mistral_large", "", "sonnet", "gpt4omini"]
NAME = {"": "haiku"}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


LABEL_ORDER = ("YES", "PARTIAL", "NO")


def conservative(r):
    """Tie-break toward NO (see r8_judge_validation.conservative)."""
    mx = max(r["probs"].values())
    return [l for l in LABEL_ORDER if r["probs"][l] == mx][-1]


def load_gen(model, cond):
    d = PROC / (f"generations_{model}" if model else "generations")
    f = d / f"{cond}.jsonl"
    if not f.is_file():
        return {}
    return {r["qid"]: r for r in read_jsonl(f)}


def load_corpus_text():
    out = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    out.setdefault(d["doc_id"], d.get("text", ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=B_DEFAULT)
    args = ap.parse_args()
    t0 = time.time()

    qs = SI.load_questions()
    corpus = load_corpus_text()
    ret_u = load_json(PROC / "retrieval.json")["hybrid"]
    ret_m = load_json(PROC / "retrieval_masked.json")["hybrid"]
    judge = {(r["qid"], r["doc_id"]): r for r in read_jsonl(JUDGE) if r["window"] == "w600"}

    nli_u = {}
    for r in read_jsonl(OUT / "nli_support_labels_600.jsonl"):
        nli_u[(r["qid"], r["doc_id"])] = r["nli_label"].startswith("entail")
    nli_m = {}
    for r in read_jsonl(OUT / "r1_nli_masked_600.jsonl"):
        nli_m[(r["qid"], r["doc_id"])] = r["label"].startswith("entail")

    qids = sorted(qs)
    docs = {"unmasked": {q: list(ret_u[q][:5]) for q in qids},
            "masked": {q: list(ret_m[q][:5]) for q in qids}}

    # -------- question-level support under each label, per condition --------
    def q_support(cond):
        out = {}
        for q in qids:
            dd = docs[cond][q]
            out[q] = {
                "string600": any(VIS(qs[q]["gold_answer"], corpus.get(d, "")) for d in dd),
                "nli600": any((nli_u if cond == "unmasked" else nli_m).get((q, d), False)
                              for d in dd),
                "judge600_YES": any(judge[(q, d)]["label"] == "YES" for d in dd),
                "judge600_YESPART": any(judge[(q, d)]["label"] in ("YES", "PARTIAL") for d in dd),
            }
        return out

    sup = {c: q_support(c) for c in ("unmasked", "masked")}
    LABELS = ("string600", "nli600", "judge600_YES", "judge600_YESPART")

    # -------- support-rate and arm-change tables ---------------------------
    def regime_mask(reg):
        if reg == "pooled":
            return {q: True for q in qids}
        return {q: (qs[q]["year"] >= 2025) == (reg == "fresh") for q in qids}

    arms = {}
    for cond in ("unmasked", "masked"):
        rec = {}
        for reg in ("static", "fresh", "pooled"):
            m = regime_mask(reg)
            sel = [q for q in qids if m[q]]
            r = {"n_questions": len(sel),
                 "support_rate": {lab: round(sum(sup[cond][q][lab] for q in sel) / len(sel), 4)
                                  for lab in LABELS},
                 "n_supported": {lab: sum(sup[cond][q][lab] for q in sel) for lab in LABELS},
                 "arm_changes_vs_judge600_YES": {}}
            for lab in ("string600", "nli600", "judge600_YESPART"):
                diff = [q for q in sel if sup[cond][q][lab] != sup[cond][q]["judge600_YES"]]
                r["arm_changes_vs_judge600_YES"][lab] = {
                    "n_changed": len(diff),
                    "rate": round(len(diff) / len(sel), 4),
                    "other_yes_judge_no": sum(1 for q in diff if sup[cond][q][lab]),
                    "judge_yes_other_no": sum(1 for q in diff if not sup[cond][q][lab])}
            r["pairwise_agreement"] = {
                f"{a}_vs_{b}": round(sum(1 for q in sel if sup[cond][q][a] == sup[cond][q][b]) / len(sel), 4)
                for i, a in enumerate(LABELS) for b in LABELS[i + 1:]}
            rec[reg] = r
        arms[cond] = rec
    # masked-vs-unmasked arm changes under the judge (context set changes)
    arms["masked_vs_unmasked_same_label"] = {
        lab: {"n_questions_changing_support": sum(
            1 for q in qids if sup["unmasked"][q][lab] != sup["masked"][q][lab])}
        for lab in LABELS}

    # -------- tie-break sensitivity on the question-level arms -------------
    tie = {}
    for cond in ("unmasked", "masked"):
        chg = {"judge600_YES": [], "judge600_YESPART": []}
        for q in qids:
            dd = docs[cond][q]
            cy = any(conservative(judge[(q, d)]) == "YES" for d in dd)
            cp = any(conservative(judge[(q, d)]) in ("YES", "PARTIAL") for d in dd)
            if cy != sup[cond][q]["judge600_YES"]:
                chg["judge600_YES"].append(q)
            if cp != sup[cond][q]["judge600_YESPART"]:
                chg["judge600_YESPART"].append(q)
        tie[cond] = {k: {"n_questions_changing_arm": len(v), "qids": v} for k, v in chg.items()}
    tie["note"] = ("exact float32 ties between two label probabilities are resolved toward the "
                   "earlier of (YES, PARTIAL, NO) by the stored label; this block resolves them "
                   "toward the later one and reports how many question-level arms move")

    # -------- the interaction, common seed, both estimators ----------------
    boot = QuizBoot(qids, B=args.B, seed=SEED)
    qi_year = np.array([qs[q]["year"] for q in qids])
    reg_masks = {"static": qi_year < 2025, "fresh": qi_year >= 2025,
                 "pooled": np.ones(len(qids), dtype=bool)}

    per = {}
    for cond, rag_cond in (("unmasked", "hybrid"), ("masked", "hybrid_masked")):
        per[cond] = {}
        for m in PANEL:
            cb, rt = load_gen(m, "closed_book"), load_gen(m, rag_cond)
            if not cb or not rt:
                continue
            common = [q for q in qids if q in cb and q in rt]
            if len(common) < len(qids):
                pass                                  # recorded below
            gain = np.array([
                100.0 * (token_f1(rt[q]["prediction"] or "", qs[q]["gold_answer"])
                         - token_f1(cb[q]["prediction"] or "", qs[q]["gold_answer"]))
                if q in cb and q in rt else np.nan for q in qids])
            have = np.isfinite(gain)
            gain0 = np.nan_to_num(gain)
            rec = {"n_questions_with_both_caches": int(have.sum())}
            for lab in LABELS:
                s = np.array([sup[cond][q][lab] for q in qids])
                rec[lab] = {}
                for reg, rm in reg_masks.items():
                    blk = {}
                    for w in WEIGHTINGS:
                        ms = rm & have & s
                        mu = rm & have & (~s)
                        if ms.sum() < 5 or mu.sum() < 5:
                            blk[w] = None
                            continue
                        o_s, d_s = boot.est(w, gain0, ms)
                        o_u, d_u = boot.est(w, gain0, mu)
                        blk[w] = summarize(o_s - o_u, d_s - d_u, 3, extra={
                            "gain_supported": round(o_s, 3),
                            "gain_unsupported": round(o_u, 3),
                            "n_supported": int(ms.sum()), "n_unsupported": int(mu.sum()),
                            "n_quizzes": boot.counts(rm & have)["n_quizzes"]})
                    rec[lab][reg] = blk
            per[cond][NAME.get(m, m)] = rec

    # -------- reproduction of the stored, script-seeded values -------------
    rep = {}
    e0 = load_json(OUT / "e0_visible_support.json")
    r1 = load_json(OUT / "r1_masked_nli.json")

    def did_boot_pack(cond, lab, model, regime, B, seed):
        """support_interaction.did_boot on the same units, that script's seed."""
        cb, rt = load_gen(model, "hybrid" if cond == "unmasked" else "hybrid_masked")[:0] or {}, {}
        return None

    # (i) unmasked string600 vs e0_visible_support (support_interaction.py seed)
    unm = {}
    for m in PANEL:
        cb, rt = load_gen(m, "closed_book"), load_gen(m, "hybrid")
        if not cb or not rt:
            continue
        for regime in ("static", "fresh"):
            units = collections.defaultdict(lambda: ([], []))
            # support_interaction.py iterates qs in FILE order; did_boot's draw
            # sequence depends on list(units), so the order must match to
            # reproduce the stored intervals, not only the point estimates.
            for q in qs:
                if (qs[q]["year"] >= 2025) != (regime == "fresh") or q not in cb or q not in rt:
                    continue
                s = sup["unmasked"][q]["string600"]
                g = 100.0 * (token_f1(rt[q]["prediction"] or "", qs[q]["gold_answer"])
                             - token_f1(cb[q]["prediction"] or "", qs[q]["gold_answer"]))
                units[SI.quiz_of(q)][0 if s else 1].append(g)
            r = SI.did_boot(dict(units), 5000, 20260802)
            if r:
                unm.setdefault(NAME.get(m, m), {})[regime] = r
    stored_unm = {m: {reg: e0["support_interaction_unmasked_hybrid"]["per_model"][m][reg]["new_visible600"]
                      for reg in ("static", "fresh")
                      if reg in e0["support_interaction_unmasked_hybrid"]["per_model"][m]}
                  for m in e0["support_interaction_unmasked_hybrid"]["per_model"]}
    rep["e0_visible_support.string600_unmasked"] = repro_record(
        stored_unm,
        {m: {reg: {k: unm[m][reg][k] for k in
                   ("interaction", "ci95", "p_plus_one", "gain_supported",
                    "gain_unsupported", "n_supported", "n_unsupported", "n_quizzes")}
             for reg in stored_unm[m]} for m in stored_unm},
        "e0_visible_support.json support_interaction_unmasked_hybrid per_model "
        "new_visible600 (string@600, question-pooled, seed 20260802)")

    # (ii) masked NLI600 vs r1_masked_nli (masked_counterfactual.py seed, Static)
    msk_nli = {}
    for m in PANEL:
        cb, rt = load_gen(m, "closed_book"), load_gen(m, "hybrid_masked")
        if not cb or not rt:
            continue
        units = collections.defaultdict(lambda: ([], []))
        for q in qids:
            if qs[q]["year"] >= MC.BOUNDARY or q not in cb or q not in rt:
                continue
            s = sup["masked"][q]["nli600"]
            g = 100.0 * (token_f1(rt[q]["prediction"] or "", qs[q]["gold_answer"])
                         - token_f1(cb[q]["prediction"] or "", qs[q]["gold_answer"]))
            units[MC.quiz_of(q)][0 if s else 1].append(g)
        sup_all = [v for a, _ in units.values() for v in a]
        uns_all = [v for _, b in units.values() for v in b]
        ci = MC.interaction_ci(units, 5000, 20260803)
        msk_nli[NAME.get(m, m)] = {
            "interaction": round(statistics.fmean(sup_all) - statistics.fmean(uns_all), 3),
            "ci95": ci["ci95"], "p_plus_one": ci["p_plus_one"],
            "gain_supported": round(statistics.fmean(sup_all), 3),
            "gain_unsupported": round(statistics.fmean(uns_all), 3),
            "n_supported": len(sup_all), "n_unsupported": len(uns_all),
            "n_quizzes": ci["n_quizzes"]}
    stored_msk = {m: v["new_masked_contexts_premise600"]
                  for m, v in r1["support_interaction_masked_nli"]["per_model"].items()}
    rep["r1_masked_nli.nli600_masked_static"] = repro_record(
        stored_msk,
        {m: {k: msk_nli[m][k] for k in stored_msk[m]} for m in stored_msk},
        "r1_masked_nli.json support_interaction_masked_nli per_model "
        "new_masked_contexts_premise600 (entailment@600 on masked contexts, Static, seed 20260803)")

    # (iii) masked string600 vs e0_visible_support (masked_counterfactual seed, Static)
    msk_str = {}
    for m in PANEL:
        cb, rt = load_gen(m, "closed_book"), load_gen(m, "hybrid_masked")
        if not cb or not rt:
            continue
        units = collections.defaultdict(lambda: ([], []))
        for q in qids:
            if qs[q]["year"] >= MC.BOUNDARY or q not in cb or q not in rt:
                continue
            s = sup["masked"][q]["string600"]
            g = 100.0 * (token_f1(rt[q]["prediction"] or "", qs[q]["gold_answer"])
                         - token_f1(cb[q]["prediction"] or "", qs[q]["gold_answer"]))
            units[MC.quiz_of(q)][0 if s else 1].append(g)
        sup_all = [v for a, _ in units.values() for v in a]
        uns_all = [v for _, b in units.values() for v in b]
        ci = MC.interaction_ci(units, 5000, 20260803)
        msk_str[NAME.get(m, m)] = {
            "interaction": round(statistics.fmean(sup_all) - statistics.fmean(uns_all), 3),
            "ci95": ci["ci95"], "p_plus_one": ci["p_plus_one"],
            "gain_supported": round(statistics.fmean(sup_all), 3),
            "gain_unsupported": round(statistics.fmean(uns_all), 3),
            "n_supported": len(sup_all), "n_unsupported": len(uns_all),
            "n_quizzes": ci["n_quizzes"]}
    stored_mstr = {m: v["new_visible600"]
                   for m, v in e0["masked_hybrid_support_interaction"]["per_model"].items()}
    rep["e0_visible_support.string600_masked_static"] = repro_record(
        stored_mstr,
        {m: {k: msk_str[m][k] for k in stored_mstr[m]} for m in stored_mstr},
        "e0_visible_support.json masked_hybrid_support_interaction per_model "
        "new_visible600 (string@600 on masked contexts, Static, seed 20260803)")

    # -------- side-by-side table -------------------------------------------
    def cell(cond, model, lab, reg, w="question_pooled"):
        v = per[cond].get(model, {}).get(lab, {}).get(reg, {}).get(w)
        return None if v is None else {"interaction": v["estimate"], "ci95": v["ci95"],
                                       "excl0": v["ci_excludes_zero"]}

    table = {}
    for cond in ("unmasked", "masked"):
        table[cond] = {}
        for model in sorted(per[cond]):
            table[cond][model] = {}
            for reg in ("static", "fresh", "pooled"):
                table[cond][model][reg] = {
                    "question_pooled": {lab: cell(cond, model, lab, reg) for lab in LABELS},
                    "quiz_weighted": {lab: cell(cond, model, lab, reg, "quiz_weighted")
                                      for lab in LABELS},
                }
    stored_side = {
        "unmasked_string600_e0": {m: {reg: {"interaction": v["interaction"], "ci95": v["ci95"]}
                                      for reg, v in stored_unm[m].items()} for m in stored_unm},
        "masked_nli600_r1_static": {m: {"interaction": v["interaction"], "ci95": v["ci95"]}
                                    for m, v in stored_msk.items()},
        "masked_string600_e0_static": {
            m: {"interaction": v["new_visible600"]["interaction"],
                "ci95": v["new_visible600"]["ci95"]}
            for m, v in e0["masked_hybrid_support_interaction"]["per_model"].items()},
    }

    # summary: does the judge label move the conclusion?
    summary = {}
    for cond in ("unmasked", "masked"):
        summary[cond] = {}
        for reg in ("static", "fresh", "pooled"):
            summary[cond][reg] = {}
            for w in WEIGHTINGS:
                s = {}
                for lab in LABELS:
                    vals, excl, n = [], 0, 0
                    for model in per[cond]:
                        v = per[cond][model][lab][reg][w]
                        if v:
                            n += 1
                            vals.append(v["estimate"])
                            excl += bool(v["ci_excludes_zero"])
                    s[lab] = {"n_models": n,
                              "median_interaction": round(float(np.median(vals)), 3) if vals else None,
                              "min": round(min(vals), 3) if vals else None,
                              "max": round(max(vals), 3) if vals else None,
                              "n_ci_excludes_zero": excl}
                # per-model judge-minus-other differences (same questions,
                # same draws; the interval is a joint resample of the two labels)
                for other in ("string600", "nli600"):
                    diffs = []
                    for model in per[cond]:
                        a = per[cond][model]["judge600_YES"][reg][w]
                        b = per[cond][model][other][reg][w]
                        if a and b:
                            diffs.append(round(a["estimate"] - b["estimate"], 3))
                    s[f"judge600_YES_minus_{other}"] = {
                        "per_model": diffs,
                        "median": round(float(np.median(diffs)), 3) if diffs else None,
                        "min": min(diffs) if diffs else None,
                        "max": max(diffs) if diffs else None,
                        "n_positive": sum(1 for x in diffs if x > 0)}
                summary[cond][reg][w] = s

    out = {
        "step": "R8(3) -- support interaction under the R8 judge label on the exact shown text",
        "meta": meta([JUDGE, PROC / "retrieval.json", PROC / "retrieval_masked.json",
                      OUT / "nli_support_labels_600.jsonl", OUT / "r1_nli_masked_600.jsonl"]),
        "estimand": "gain(supported) - gain(unsupported), token F1 points, "
                    "gain = 100*(F1 RAG - F1 closed-book)",
        "not_causal": (
            "Support is not randomised and every label here uses the gold answer. A positive "
            "interaction says the retrieval gain differs by labelled support status; it does not "
            "say that supplying support would produce that gain, and it is not a deployable rule. "
            "The judge label is an LLM prediction on the shown text, calibrated against legacy "
            "700-character human labels (r8_judge_validation.json), not verified semantics."),
        "labels": {
            "string600": "metrics.gold_in_text(gold, text[:600]) on any of the 5 shown passages",
            "nli600": "DeBERTa-v3-large entailment, premise text[:600] (unmasked: "
                      "nli_support_labels_600.jsonl; masked: r1_nli_masked_600.jsonl)",
            "judge600_YES": "R8 judge label YES on any of the 5 shown passages",
            "judge600_YESPART": "R8 judge label YES or PARTIAL on any of the 5 shown passages"},
        "bootstrap": {"B": args.B, "seed": SEED,
                      "unit": "weekly quiz (clusters.quiz_id); ONE multiplicity matrix drives "
                              "every label, model, regime and condition (analysis_common.QuizBoot)",
                      "estimators": list(WEIGHTINGS),
                      "primary": "question_pooled"},
        "support_rates_and_arm_changes": arms,
        "tie_break_sensitivity_question_arms": tie,
        "per_model": per,
        "side_by_side_question_pooled": table,
        "stored_reference_values": stored_side,
        "reproduction_of_stored_values": rep,
        "summary_by_regime_and_weighting": summary,
        "runtime_total_sec": round(time.time() - t0, 1),
    }
    dump(out, OUT / "r8_judge_interaction.json")

    print("repro:", {k: v["reproduced_exactly"] for k, v in rep.items()})
    for cond in ("unmasked", "masked"):
        print("=" * 110)
        for model in sorted(per[cond]):
            for reg in ("static", "fresh", "pooled"):
                row = [f"{cond:9s}{model:14s}{reg:7s}"]
                for lab in LABELS:
                    v = per[cond][model][lab][reg]["question_pooled"]
                    row.append(f"{lab[:17]:18s}{v['estimate']:7.2f}{str(v['ci95']):>20s}"
                               if v else f"{lab[:17]:18s}{'n/a':>27s}")
                print("  ".join(row))
    print(json.dumps(summary, indent=1))
    print("wrote", OUT / "r8_judge_interaction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
