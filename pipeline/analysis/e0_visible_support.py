#!/usr/bin/env python3
"""E0 task 1: string support labels recomputed on the text the generator saw.

Every generator renders each retrieved passage as text[:600]. The string support
label in the Layer-3 analyses was computed with metrics.gold_in_text on the FULL
stored document (~3,000 chars). This script re-runs, unchanged except for that
label, the string-label paths of

  * pipeline/support_interaction.py      unmasked Hybrid, Static and Fresh
  * pipeline/masked_counterfactual.py    support interaction on masked Hybrid
  * pipeline/support_conditioned.py      support-conditioned decomposition
  * pipeline/routing_baselines.py        GOLD_STRING diagnostic policy
  * pipeline/k_ablation.py               k=3 and k=5 ONLY (k=10 is invalid:
                                         72.8% of its prompts were right-
                                         truncated at 1,536 tokens)

Order of operations for each: (1) rerun the ORIGINAL full-text setting and
compare it with the stored results/*.json value; (2) rerun with the label on
text[:600]. Each script's own seed, B and estimator are kept. Modules are
imported and patched in memory (their `gold_in_text` name) -- no file under
pipeline/ is modified.

Outputs results/analysis/e0_visible_support.json and raw reruns under
results/analysis/e0_raw/.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import statistics
import time

from _e0util import (GEN_DOC_CHARS, OUT, PROC, RAW, RES, dump, load_json, meta,
                     repro_record, run_main, window_fn)

import k_ablation as KA                     # noqa: E402
import masked_counterfactual as MC          # noqa: E402
import routing_baselines as RB              # noqa: E402
import support_conditioned as SC            # noqa: E402
import support_interaction as SI            # noqa: E402

FULL = window_fn(None)
VIS = window_fn(GEN_DOC_CHARS)
KEEP = ("interaction", "ci95", "p_plus_one", "gain_supported", "gain_unsupported",
        "n_supported", "n_unsupported", "n_quizzes")


def side(old, new, keys=KEEP):
    rec = {"old_fulltext": {k: old.get(k) for k in keys if k in old} if old else None,
           "new_visible600": {k: new.get(k) for k in keys if k in new} if new else None}
    if old and new and "interaction" in old and "interaction" in new:
        rec["interaction_change"] = round(new["interaction"] - old["interaction"], 3)
        rec["ci_excludes_zero"] = {"old": old["ci95"][0] > 0 or old["ci95"][1] < 0,
                                   "new": new["ci95"][0] > 0 or new["ci95"][1] < 0}
    return rec


def load_corpus_text():
    corpus = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    corpus.setdefault(d["doc_id"], d.get("text", ""))
    return corpus


def transitions(qs, docs_by_q, corpus, boundary=2025):
    """Question-level support under full vs visible text, by regime."""
    tab = {"static": collections.Counter(), "fresh": collections.Counter()}
    s2u = {"static": [], "fresh": []}
    u2s = {"static": [], "fresh": []}
    for qid in sorted(docs_by_q):
        q = qs.get(qid)
        if not q:
            continue
        reg = "fresh" if q["year"] >= boundary else "static"
        texts = [corpus.get(d, "") for d in docs_by_q[qid]]
        a = any(FULL(q["gold_answer"], t) for t in texts if t)
        b = any(VIS(q["gold_answer"], t) for t in texts if t)
        tab[reg][("S" if a else "U") + "->" + ("S" if b else "U")] += 1
        if a and not b:
            s2u[reg].append(qid)
        if b and not a:
            u2s[reg].append(qid)
    out = {}
    for reg in ("static", "fresh"):
        c = tab[reg]
        out[reg] = {"S->S": c["S->S"], "S->U": c["S->U"], "U->S": c["U->S"],
                    "U->U": c["U->U"],
                    "n_supported_old": c["S->S"] + c["S->U"],
                    "n_supported_new": c["S->S"] + c["U->S"],
                    "qids_supported_to_unsupported": s2u[reg],
                    "qids_unsupported_to_supported": u2s[reg]}
    out["total_S->U"] = sum(out[r]["S->U"] for r in ("static", "fresh"))
    out["total_U->S"] = sum(out[r]["U->S"] for r in ("static", "fresh"))
    return out


def slope_exact(pts):
    """Same OLS-on-log10-params slope and exhaustive permutation p as the scripts."""
    pts = sorted(pts)
    x = [math.log10(a) for a, _ in pts]
    y = [b for _, b in pts]
    mx = statistics.fmean(x)
    sxx = sum((a - mx) ** 2 for a in x)
    fit = lambda yy: sum((a - mx) * (b - statistics.fmean(yy))
                         for a, b in zip(x, yy)) / sxx
    obs = fit(y)
    ge = tot = 0
    for perm in itertools.permutations(y):
        tot += 1
        ge += abs(fit(list(perm))) >= abs(obs) - 1e-12
    return {"slope_per_decade": round(obs, 3), "p_exact": round(ge / tot, 5),
            "n_rungs": len(y)}


# --------------------------------------------------------------------------
# masked_counterfactual.py: the support-interaction block, reproduced verbatim
# (lines ~270-300 of the script) with the label function as a parameter. The
# rest of that script (paired mask deltas) does not use the support label.
# --------------------------------------------------------------------------
def masked_support_interaction(label_fn, corpus, B, seed):
    qs = MC.load_questions()
    inter = {}
    regenerated = []
    for m in MC.PANEL:
        ma = MC.load(m, "hybrid_masked")
        if not ma or len(ma) < len(qs):
            continue
        regenerated.append(m)
    for m in regenerated:
        cb, ma = MC.load(m, "closed_book"), MC.load(m, "hybrid_masked")
        common = sorted(set(cb) & set(ma) & set(qs))
        units = collections.defaultdict(lambda: ([], []))
        for q in common:
            if qs[q]["year"] >= MC.BOUNDARY:
                continue
            texts = [corpus.get(d, "") for d in (ma[q].get("doc_ids") or [])]
            sup = any(label_fn(qs[q]["gold_answer"], t) for t in texts if t)
            g = 100.0 * (MC.token_f1(ma[q]["prediction"] or "", qs[q]["gold_answer"])
                         - MC.token_f1(cb[q]["prediction"] or "", qs[q]["gold_answer"]))
            units[MC.quiz_of(q)][0 if sup else 1].append(g)
        sup_all = [v for a, _ in units.values() for v in a]
        uns_all = [v for _, b in units.values() for v in b]
        if len(sup_all) < 10 or len(uns_all) < 10:
            continue
        rec = {"gain_supported": round(statistics.fmean(sup_all), 3),
               "gain_unsupported": round(statistics.fmean(uns_all), 3),
               "interaction": round(statistics.fmean(sup_all) - statistics.fmean(uns_all), 3),
               "n_supported": len(sup_all), "n_unsupported": len(uns_all)}
        ci = MC.interaction_ci(units, B, seed)
        if ci:
            rec["ci95"] = ci["ci95"]
            rec["p_plus_one"] = ci["p_plus_one"]
            rec["n_quizzes"] = ci["n_quizzes"]
        inter[m] = rec
    pts = [(MC.LADDER_B[m], v["gain_unsupported"]) for m, v in inter.items()
           if m in MC.LADDER_B]
    slope = slope_exact(pts) if len(pts) >= 4 else None
    if slope:
        slope["points"] = [{"params_B": a, "gain_unsupported": b} for a, b in sorted(pts)]
    return inter, slope


# --------------------------------------------------------------------------
# routing_baselines.py: the GOLD_STRING policy and its paired contrast against
# the better fixed policy, using the script's own loaders and estimators.
# Every estimator call re-seeds, so this equals the full script's value.
# --------------------------------------------------------------------------
def routing_gold_string(label_fn, B, seed):
    qs = RB.load_questions()
    corpus = RB.load_corpus()
    ranked = json.loads((PROC / "retrieval.json").read_text())
    models = sorted(d.name.replace("generations_", "")
                    for d in PROC.glob("generations_*") if d.is_dir())
    out = {}
    for m in models:
        cb, rt = RB.load_gen(m, "closed_book"), RB.load_gen(m, "hybrid")
        if not cb or not rt:
            continue
        common = sorted(q for q in qs if q in cb and q in rt)
        f_cb = {q: RB.token_f1(cb[q]["prediction"] or "", qs[q]["gold_answer"]) * 100
                for q in common}
        f_rt = {q: RB.token_f1(rt[q]["prediction"] or "", qs[q]["gold_answer"]) * 100
                for q in common}
        gs = {}
        for q in common:
            docs = ranked.get("hybrid", {}).get(q, [])[:5]
            texts = [corpus.get(d, {}).get("text", "") for d in docs]
            gs[q] = any(label_fn(qs[q]["gold_answer"], t) for t in texts if t)
        score = lambda q, use: (f_rt if use else f_cb)[q]
        pol = {"never": {q: score(q, False) for q in common},
               "always": {q: score(q, True) for q in common},
               "gold_string": {q: score(q, gs[q]) for q in common},
               "perq_best": {q: max(f_cb[q], f_rt[q]) for q in common}}
        stats = {k: RB.cluster_mean_ci([(RB.quiz_of(q), v[q]) for q in common], B, seed)
                 for k, v in pol.items()}
        ref = max(("always", "never"), key=lambda k: stats[k]["mean"])
        d = RB.paired_delta([(RB.quiz_of(q), pol["gold_string"][q] - pol[ref][q])
                             for q in common], B, seed)
        base, ceil, g = stats[ref]["mean"], stats["perq_best"]["mean"], stats["gold_string"]["mean"]
        out[m] = {"reference": ref,
                  "reference_mean": base,
                  "perq_best_mean": ceil,
                  "gold_string_policy": stats["gold_string"],
                  "gold_string_vs_reference": d,
                  "gold_string_retrieve_rate": round(sum(gs.values()) / len(common), 4),
                  "gold_string_share_of_ceiling_headroom":
                      round((g - base) / (ceil - base), 3) if ceil > base else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    args = ap.parse_args()
    t_start = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus_text()
    qs = SI.load_questions()
    runtimes = {}
    repro = {}

    # ---------------- support_interaction.py (unmasked Hybrid) -------------
    f_old, f_new = RAW / "support_interaction__fulltext.json", RAW / "support_interaction__vis600.json"
    _, runtimes["support_interaction_full"] = run_main(
        SI, ["-B", args.B, "--output", f_old], RAW / "support_interaction__fulltext.log")
    _, runtimes["support_interaction_vis600"] = run_main(
        SI, ["-B", args.B, "--output", f_new], RAW / "support_interaction__vis600.log",
        gold_in_text=VIS)
    stored, old, new = load_json(RES / "support_interaction.json"), load_json(f_old), load_json(f_new)
    repro["support_interaction.py"] = repro_record(stored, old, "entire output JSON (string and nli labels)")
    si = {"script": "pipeline/support_interaction.py", "label": "string",
          "condition": "hybrid (unmasked), k=5",
          "seed": 20260802, "B": args.B,
          "weighting": ("QUESTION-POOLED: gain(S) and gain(U) are means over all "
                        "questions in each arm, every question weight 1. CI: "
                        "quiz-cluster bootstrap (resample quizzes with replacement, "
                        "both arms recomputed per draw, percentile interval). Not "
                        "quiz-weighted. Haiku (bare generations/) is not in this "
                        "script's model glob."),
          "per_model": {}}
    for m in old["labels"]["string"]:
        si["per_model"][m] = {reg: side(old["labels"]["string"][m].get(reg),
                                        new["labels"]["string"][m].get(reg))
                              for reg in ("static", "fresh")}
    si["unsupported_cost_vs_scale"] = {
        "old_fulltext": {k: old["unsupported_cost_vs_scale"]["string"][k]
                         for k in ("slope_per_decade", "p_exact", "n_rungs", "points")},
        "new_visible600": {k: new["unsupported_cost_vs_scale"]["string"][k]
                           for k in ("slope_per_decade", "p_exact", "n_rungs", "points")}}
    si["summary"] = {"old_fulltext": old["summary"]["string"],
                     "new_visible600": new["summary"]["string"]}
    hyb_docs = {q: r.get("doc_ids") or [] for q, r in SI.load_gen("qwen32b", "hybrid").items()}
    # doc_ids are identical across generators (checked in the log below)
    ident = all({q: r.get("doc_ids") or [] for q, r in SI.load_gen(m, "hybrid").items()} == hyb_docs
                for m in old["labels"]["string"])
    si["transitions"] = transitions(qs, hyb_docs, corpus)
    si["transitions"]["doc_ids_identical_across_generators"] = ident

    # ---------------- masked_counterfactual.py (masked Hybrid) --------------
    t0 = time.time()
    m_old, s_old = masked_support_interaction(FULL, corpus, args.B, 20260803)
    m_new, s_new = masked_support_interaction(VIS, corpus, args.B, 20260803)
    runtimes["masked_support_interaction_both"] = round(time.time() - t0, 1)
    stored_mc = load_json(RES / "masked_counterfactual.json")
    repro["masked_counterfactual.py"] = repro_record(
        {"support_interaction_masked.string": stored_mc["support_interaction_masked"]["string"],
         "unsupported_cost_vs_scale_masked.string": {
             k: stored_mc["unsupported_cost_vs_scale_masked"]["string"][k]
             for k in ("slope_per_decade", "p_exact", "n_rungs", "points")}},
        {"support_interaction_masked.string": m_old,
         "unsupported_cost_vs_scale_masked.string": s_old},
        "support_interaction_masked['string'] and its unsupported-cost slope "
        "(the only label-dependent block; recomputed with the script's own "
        "loaders and interaction_ci)")
    dump({"string_fulltext": m_old, "slope_fulltext": s_old,
          "string_vis600": m_new, "slope_vis600": s_new},
         RAW / "masked_support_interaction__both.json")
    mas_docs = {q: r.get("doc_ids") or [] for q, r in MC.load("qwen32b", "hybrid_masked").items()}
    mc = {"script": "pipeline/masked_counterfactual.py", "label": "string",
          "condition": "hybrid_masked (publish_date <= question_date + 3d), k=5",
          "slice": "Static only, as in the script", "seed": 20260803, "B": args.B,
          "weighting": ("QUESTION-POOLED arm means (the script's quiz-weighted "
                        "primary applies to its paired mask deltas, NOT to this "
                        "support interaction); quiz-cluster bootstrap CI. Haiku "
                        "is included (key '' = Haiku)."),
          "per_model": {("haiku" if m == "" else m): side(m_old.get(m), m_new.get(m))
                        for m in m_old},
          "unsupported_cost_vs_scale_masked": {"old_fulltext": s_old, "new_visible600": s_new},
          "transitions": transitions(qs, mas_docs, corpus)}

    # ---------------- support_conditioned.py --------------------------------
    c_old, c_new = RAW / "support_conditioned__fulltext.json", RAW / "support_conditioned__vis600.json"
    _, runtimes["support_conditioned_full"] = run_main(
        SC, ["-B", args.B, "--output", c_old], RAW / "support_conditioned__fulltext.log")
    _, runtimes["support_conditioned_vis600"] = run_main(
        SC, ["-B", args.B, "--output", c_new], RAW / "support_conditioned__vis600.log",
        gold_in_text=VIS)
    stored, old, new = load_json(RES / "support_conditioned.json"), load_json(c_old), load_json(c_new)
    repro["support_conditioned.py"] = repro_record(stored, old, "entire output JSON")
    sc_keys = ("n_questions", "support_rate", "acc_given_support", "acc_given_no_support",
               "closed_book_acc", "override_when_supported", "override_when_unsupported",
               "synthesis_failure_rate", "paired_gain_given_support")
    sc = {"script": "pipeline/support_conditioned.py", "label": "string",
          "condition": "hybrid (unmasked), k=5", "seed": 20260801, "B": args.B,
          "correct_threshold": "token F1 >= 0.5",
          "no_interaction_estimand": ("this script has no gain(S)-gain(U) "
                                      "interaction; the side-by-side fields are its "
                                      "own quantities"),
          "weighting": {
              "acc_given_support / acc_given_no_support / closed_book_acc":
                  "QUIZ-WEIGHTED: mean of per-quiz accuracy over quizzes having >=1 "
                  "question in that arm; CI = i.i.d. bootstrap over those quiz means; "
                  "field n = number of quizzes",
              "support_rate / override_* / rescue_rate / synthesis_failure_rate":
                  "QUESTION-POOLED proportions, no CI",
              "paired_gain_given_support":
                  "QUESTION-POOLED mean of 100*(correct_RAG - correct_CB) over "
                  "supported questions; CI = i.i.d. QUESTION bootstrap (NOT "
                  "quiz-clustered); n = questions"},
          "per_model": {}}
    for m in old["models"]:
        sc["per_model"][m] = {}
        for reg in ("static", "fresh"):
            o, n = old["models"][m].get(reg), new["models"][m].get(reg)
            if not o:
                continue
            rec = {"old_fulltext": {k: o.get(k) for k in sc_keys},
                   "new_visible600": {k: n.get(k) for k in sc_keys}}
            for tag, r in (("old_fulltext", o), ("new_visible600", n)):
                ns = round(r["support_rate"] * r["n_questions"])
                rec[tag]["n_supported"] = ns
                rec[tag]["n_unsupported"] = r["n_questions"] - ns
            pg_o, pg_n = o.get("paired_gain_given_support"), n.get("paired_gain_given_support")
            if pg_o and pg_n:
                rec["paired_gain_given_support_change"] = round(pg_n["mean"] - pg_o["mean"], 3)
            sc["per_model"][m][reg] = rec
    sc["mechanism"] = {"old_fulltext": {k: v for k, v in old["mechanism"].items() if k != "reading"},
                       "new_visible600": {k: v for k, v in new["mechanism"].items() if k != "reading"}}

    # ---------------- routing_baselines.py GOLD_STRING ----------------------
    t0 = time.time()
    r_old = routing_gold_string(FULL, args.B, 20260802)
    r_new = routing_gold_string(VIS, args.B, 20260802)
    runtimes["routing_gold_string_both"] = round(time.time() - t0, 1)
    stored_rb = load_json(RES / "routing_baselines.json")
    repro["routing_baselines.py"] = repro_record(
        {m: {"reference": v["reference"], "gold_string": v["policies"]["gold_string"],
             "vs_reference.gold_string": v["vs_reference"].get("gold_string"),
             "perq_best": v["policies"]["perq_best"]["mean"]}
         for m, v in stored_rb["models"].items()},
        {m: {"reference": v["reference"],
             "gold_string": v["gold_string_policy"],
             "vs_reference.gold_string": v["gold_string_vs_reference"],
             "perq_best": v["perq_best_mean"]}
         for m, v in r_old.items()},
        "policies.gold_string, vs_reference.gold_string, reference, perq_best mean "
        "(label-dependent entries; recomputed with the script's own estimators)")
    dump({"fulltext": r_old, "vis600": r_new}, RAW / "routing_gold_string__both.json")
    rb = {"script": "pipeline/routing_baselines.py", "policy": "gold_string",
          "condition": "hybrid (unmasked), retrieval.json top-5", "seed": 20260802,
          "B": args.B,
          "weighting": ("QUIZ-WEIGHTED: policy mean = mean of per-quiz means over "
                        "199 quizzes; paired delta vs the better fixed policy is "
                        "also a mean of quiz-mean differences; CI = bootstrap over "
                        "quiz means. Ceiling (perq_best) does not depend on the label."),
          "per_model": {}}
    for m in r_old:
        o, n = r_old[m], r_new[m]
        rb["per_model"][m] = {
            "reference": o["reference"], "reference_mean": o["reference_mean"],
            "perq_best_mean": o["perq_best_mean"],
            "old_fulltext": {"policy": o["gold_string_policy"],
                             "vs_reference": o["gold_string_vs_reference"],
                             "retrieve_rate": o["gold_string_retrieve_rate"],
                             "share_of_ceiling_headroom": o["gold_string_share_of_ceiling_headroom"]},
            "new_visible600": {"policy": n["gold_string_policy"],
                               "vs_reference": n["gold_string_vs_reference"],
                               "retrieve_rate": n["gold_string_retrieve_rate"],
                               "share_of_ceiling_headroom": n["gold_string_share_of_ceiling_headroom"]},
            "delta_vs_reference_change": round(n["gold_string_vs_reference"]["delta"]
                                               - o["gold_string_vs_reference"]["delta"], 3)}
    rb["n_models_gold_string_below_reference"] = {
        "old_fulltext": sum(1 for v in r_old.values() if v["gold_string_vs_reference"]["delta"] < 0),
        "new_visible600": sum(1 for v in r_new.values() if v["gold_string_vs_reference"]["delta"] < 0)}
    rb["transitions"] = transitions(qs, {q: v[:5] for q, v in json.loads(
        (PROC / "retrieval.json").read_text())["hybrid"].items()}, corpus)

    # ---------------- k_ablation.py, k=3 and k=5 only -----------------------
    arms_no10 = [a for a in KA.ARMS if a[0] != "k10"]
    k_old, k_new = RAW / "k_ablation__k3k5__fulltext.json", RAW / "k_ablation__k3k5__vis600.json"
    _, runtimes["k_ablation_full"] = run_main(
        KA, ["-B", args.B, "--output", k_old], RAW / "k_ablation__k3k5__fulltext.log",
        ARMS=arms_no10)
    _, runtimes["k_ablation_vis600"] = run_main(
        KA, ["-B", args.B, "--output", k_new], RAW / "k_ablation__k3k5__vis600.log",
        ARMS=arms_no10, gold_in_text=VIS)
    stored, old, new = load_json(RES / "k_ablation.json"), load_json(k_old), load_json(k_new)
    strip = lambda d: {"per_model": {m: {a: r[a] for a in ("k3", "k5") if a in r}
                                     for m, r in d["per_model"].items()},
                       "partition_shift": {a: d["partition_shift"][a] for a in ("k3", "k5")}}
    repro["k_ablation.py"] = repro_record(strip(stored), strip(old),
                                          "per_model k3/k5 and partition_shift k3/k5")
    ka = {"script": "pipeline/k_ablation.py", "label": "string",
          "condition": "hybrid_masked_k3 / hybrid_masked (k5)", "slice": "Static only",
          "seed": 20260804, "B": args.B,
          "weighting": ("QUESTION-POOLED arm means; quiz-cluster bootstrap CI. "
                        "Support is computed from the ranking file's doc lists "
                        "(identical to the generation doc_ids for k3/k5)."),
          "k10": ("NOT REPORTED: 639/878 (72.8%) k=10 prompts exceeded the "
                  "1,536-token limit in local_generate.py and were right-truncated, "
                  "removing the question; the arm is invalid pending a rerun."),
          "per_model": {m: {a: side(old["per_model"][m].get(a), new["per_model"][m].get(a),
                                     KEEP + ("excludes_zero",))
                            for a in ("k3", "k5")} for m in old["per_model"]},
          "partition_shift": {"old_fulltext": {a: old["partition_shift"][a] for a in ("k3", "k5")},
                              "new_visible600": {a: new["partition_shift"][a] for a in ("k3", "k5")}}}
    ka["transitions"] = {}
    for arm, rfile, _ in arms_no10:
        rk = json.loads((PROC / rfile).read_text())["hybrid"]
        ka["transitions"][arm] = transitions(qs, rk, corpus)

    out = {
        "step": "E0 task 1 -- string support label on generator-visible text",
        "meta": meta([PROC / "retrieval.json", PROC / "retrieval_masked.json",
                      PROC / "retrieval_masked_k3.json"]),
        "label_definitions": {
            "old_fulltext": "metrics.gold_in_text(gold, full stored corpus text), any of the shown docs",
            "new_visible600": "metrics.gold_in_text(gold, text[:600]), any of the shown docs; "
                              "text[:600] is exactly the per-passage prefix every generator "
                              "rendered (DOC_CHARS=600); same SQuAD normalisation",
        },
        "reproduction_of_stored_results": repro,
        "runtimes_sec": runtimes,
        "support_interaction_unmasked_hybrid": si,
        "masked_hybrid_support_interaction": mc,
        "support_conditioned": sc,
        "routing_gold_string": rb,
        "k_ablation_k3_k5": ka,
        "not_causal": ("Support remains a gold-conditioned, non-randomised "
                       "partition; these are descriptive associations."),
    }
    out["runtimes_sec"]["total"] = round(time.time() - t_start, 1)
    dump(out, OUT / "e0_visible_support.json")

    # ---------------- console summary ---------------------------------------
    print("reproduction:", {k: v["reproduced_exactly"] for k, v in repro.items()})
    print(f"\n{'unmasked hybrid':16s}{'regime':>7s}{'old':>9s}{'new':>9s}  old CI / new CI")
    for m, r in si["per_model"].items():
        for reg, v in r.items():
            print(f"{m:16s}{reg:>7s}{v['old_fulltext']['interaction']:9.2f}"
                  f"{v['new_visible600']['interaction']:9.2f}  "
                  f"{v['old_fulltext']['ci95']} / {v['new_visible600']['ci95']}")
    print("transitions hybrid:", {r: (si['transitions'][r]['S->U'], si['transitions'][r]['U->S'])
                                  for r in ("static", "fresh")})
    print(f"wrote {OUT / 'e0_visible_support.json'}  ({out['runtimes_sec']['total']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
