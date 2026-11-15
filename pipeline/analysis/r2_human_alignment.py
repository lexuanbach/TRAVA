#!/usr/bin/env python3
"""R2(a,b): human agreement of the 600-char automatic labels, restricted honestly.

WHAT HUMANS SAW. Task B annotators saw text.replace('\\n',' ')[:700]; the masked
pack showed ' '.join(text.split())[:700]. Generators saw raw text[:600]. No new
human labels exist for the 600-char excerpts, so every statistic here is
agreement with LEGACY 700-CHARACTER HUMAN LABELS. It is not validation of
relation support on the exact generator input.

(a) Passage level, for string@600, NLI@600 (undated) and NLI@600 (dated
    hypothesis; see r2_dated_nli.py), against strict YES and YES+PARTIAL:
      (i)  all labelled passages;
      (ii) excluding passages whose gold string occurs in the human excerpt but
           not in text[:600] (i.e. only in characters 600-700 of the excerpt).
    This exclusion is a LOWER BOUND on window differences: a human judgement can
    also rest on non-string content in characters 600-700.
    Metrics: precision, recall, Cohen's kappa, raw agreement; 95% percentile CIs
    from the quiz-clustered bootstrap of e0_human_alignment.py /
    support_label_validation.py (same draws for all four statistics; the
    agreement interval reproduces e0 exactly on all rows).
(b) Question level. Humans labelled exactly 3 passages per question (both
    packs). The automatic label is recomputed on EXACTLY those passages (same
    doc ids) and reported separately from the 5-passage label used in the
    analyses. The 3-passage comparison is a restricted annotation subset.
    Unweighted sample agreement and design-weighted estimates:
      Task B     300 sampled items (generator x question), 75 per
                 regime x full-text-auto-support cell, weight_to_population
                 22.8/25.68/25.2/31.68 (population 7,902 = 9 generators x 878
                 questions). Labels do not depend on the generator (duplicate
                 items show the same 3 passages; verified), so the item-weighted
                 estimate is a population estimate over the 878 questions.
                 The per-question convention of human_support_interaction.py
                 (each distinct question once, its cell's item weight) is
                 reported alongside.
      masked     200 stratified questions, stratum weights 3.82/4.26/4.8/4.68
                 (population 878); the 10 consistency_recheck questions have no
                 inclusion weight and are excluded from weighted estimates.
    The sampling strata were drawn on the FULL-TEXT string label; those
    weights remain the valid design weights.
Also reruns pipeline/support_label_typology.py (appendix app:typology) at 600
characters after reproducing its stored output.

Outputs results/analysis/r2_human_alignment.json (+ r2_raw/).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time
from pathlib import Path

from _e0util import (GEN_DOC_CHARS, OUT, PROC, RES, ROOT, dump, load_json, meta,
                     repro_record, run_main, window_fn)

import e0_human_alignment as E0H                                  # noqa: E402
import human_support_interaction as HSI                           # noqa: E402
import support_label_typology as SLT                              # noqa: E402
from score_annotations import cohens_kappa                        # noqa: E402

RAW = OUT / "r2_raw"
SRV = OUT / "r1_raw" / "server_out"
E0_NLI600 = OUT / "nli_support_labels_600.jsonl"
PACK_B, PACK_M = E0H.PACK_B, E0H.PACK_M
SEED_B, SEED_M = 20260803, 20260804
VIS = window_fn(GEN_DOC_CHARS)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def server_labels(job):
    return {(r["qid"], r["doc_id"]): r["label"].startswith("entail")
            for r in read_jsonl(SRV / f"{job}.jsonl")}


# ---------------------------------------------------------------------------
# one bootstrap, four statistics
# ---------------------------------------------------------------------------
def _stats(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n <= 0:
        return None
    po = (tp + tn) / n
    pa_yes, ph_yes = (tp + fp) / n, (tp + fn) / n
    pe = pa_yes * ph_yes + (1 - pa_yes) * (1 - ph_yes)
    return {"agreement": po,
            "cohens_kappa": (po - pe) / (1 - pe) if pe != 1 else float("nan"),
            "precision": tp / (tp + fp) if tp + fp > 0 else float("nan"),
            "recall": tp / (tp + fn) if tp + fn > 0 else float("nan")}


def agreement_block(units, B, seed):
    """units: list of (cluster, auto_bool, human_bool, weight), in row order.

    Clusters are quizzes, resampled with replacement exactly as
    support_label_validation.py's boot_rate does (same RNG call sequence), so
    with unit weights the agreement interval equals the stored one.
    """
    by = collections.OrderedDict()
    for c, a, h, w in units:
        by.setdefault(c, []).append((a, h, w))
    cl = list(by.values())

    def conf(rows):
        tp = fp = fn = tn = 0.0
        for a, h, w in rows:
            if a and h:
                tp += w
            elif a and not h:
                fp += w
            elif h:
                fn += w
            else:
                tn += w
        return tp, fp, fn, tn

    obs_c = conf([r for c in cl for r in c])
    obs = _stats(*obs_c)
    rng = random.Random(seed)
    draws = collections.defaultdict(list)
    for _ in range(B):
        pool = []
        for _ in cl:
            pool.extend(cl[rng.randrange(len(cl))])
        s = _stats(*conf(pool))
        for k, v in s.items():
            if v == v:                      # drop NaN draws
                draws[k].append(v)
    out = {"n_units": len(units), "n_clusters": len(cl),
           "sum_weights": round(sum(w for *_, w in units), 3),
           "confusion_weighted" if any(w != 1 for *_, w in units) else "confusion":
               {"auto_yes_human_yes": round(obs_c[0], 3), "auto_yes_human_no": round(obs_c[1], 3),
                "auto_no_human_yes": round(obs_c[2], 3), "auto_no_human_no": round(obs_c[3], 3)}}
    for k, v in obs.items():
        d = sorted(draws[k])
        m = len(d)
        out[k] = {"estimate": round(v, 4),
                  "ci95": [round(d[int(0.025 * m)], 4), round(d[int(0.975 * m)], 4)] if m else None,
                  "n_valid_draws": m}
    return out


def passage_metrics(rows, auto_key, B, seed, exclude_key=None):
    sel = [r for r in rows if r.get(auto_key) is not None and not (exclude_key and r[exclude_key])]
    out = {}
    for strict, name in ((True, "human_YES_only"), (False, "human_YES_or_PARTIAL")):
        hb = "human_bin" if strict else "human_bin_loose"
        blk = agreement_block([(r["quiz"], bool(r[auto_key]), bool(r[hb]), 1.0) for r in sel], B, seed)
        a = ["Y" if r[auto_key] else "N" for r in sel]
        h = ["Y" if r[hb] else "N" for r in sel]
        k_ref = cohens_kappa(a, h)
        assert abs(k_ref - blk["cohens_kappa"]["estimate"]) < 1e-4, (k_ref, blk["cohens_kappa"])
        out[name] = blk
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    args = ap.parse_args()
    t_start = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    qs, corpus = E0H.load_questions(), E0H.load_corpus()
    mhs = E0H.load_mhs()
    e0_ha = load_json(OUT / "e0_human_alignment.json")

    nli_mps = E0H.nli_map(E0_NLI600)
    nli_cuda = server_labels("undated600")
    nli_dated = server_labels("dated600")
    ret_u = load_json(PROC / "retrieval.json")["hybrid"]
    ret_m = load_json(PROC / "retrieval_masked.json")["hybrid"]

    # ================= (a) passage level ======================================
    rows_b = E0H.packb_rows(qs, corpus)
    rows_m = E0H.masked_rows(qs, corpus, mhs)
    for r in rows_b:
        k = (r["qid"], r["doc_id"])
        r["nli600"], r["nli600_cuda"], r["nli600_dated"] = nli_mps.get(k), nli_cuda.get(k), nli_dated.get(k)
        r["gold_only_in_600_700"] = bool(r["string_w700nl"] and not r["string_w600"])
    for r in rows_m:
        k = (r["qid"], r["doc_id"])
        r["nli600"], r["nli600_dated"] = nli_cuda.get(k), nli_dated.get(k)
        r["gold_only_in_600_700"] = bool(r["string_w700ws"] and not r["string_w600"])
    assert all(r["nli600"] is not None and r["nli600_dated"] is not None for r in rows_b + rows_m)

    def affected(rows):
        sel = [r for r in rows if r["gold_only_in_600_700"]]
        return {"n_passage_rows": len(sel),
                "n_unique_qid_doc": len({(r["qid"], r["doc_id"]) for r in sel}),
                "n_questions": len({r["qid"] for r in sel}),
                "human_label_distribution": dict(collections.Counter(r["human"] for r in sel)),
                "rows": [{"item_id": r["item_id"], "qid": r["qid"], "doc_id": r["doc_id"], "human": r["human"],
                          "string600": r["string_w600"], "nli600": r["nli600"],
                          "nli600_dated": r["nli600_dated"]} for r in sel],
                "n_rows_gold_in_600_but_not_in_human_excerpt":
                    sum(1 for r in rows if r["string_w600"] and not r[
                        "string_w700nl" if "string_w700nl" in r else "string_w700ws"])}

    labels_b = {"string600": "string_w600", "nli600": "nli600", "nli600_dated": "nli600_dated",
                "nli600_cuda_device_check": "nli600_cuda",
                "legacy_string_700_as_shown_to_humans": "string_w700nl"}
    labels_m = {"string600": "string_w600", "nli600": "nli600", "nli600_dated": "nli600_dated",
                "legacy_string_700_as_shown_to_humans": "string_w700ws"}
    pa = {}
    for pack, rows, labels, seed in (("task_b", rows_b, labels_b, SEED_B),
                                     ("masked_pack", rows_m, labels_m, SEED_M)):
        blk = {"n_passage_rows": len(rows), "n_questions": len({r["qid"] for r in rows}),
               "rows_gold_string_only_in_chars_600_700": affected(rows), "by_label": {}}
        for name, key in labels.items():
            blk["by_label"][name] = {
                "i_all_rows": passage_metrics(rows, key, args.B, seed),
                "ii_excluding_gold_only_in_600_700": passage_metrics(rows, key, args.B, seed,
                                                                     exclude_key="gold_only_in_600_700")}
        pa[pack] = blk

    # reproduction against e0 (all rows)
    e0b = e0_ha["a_task_b_pack"]
    e0m = e0_ha["a_masked_pack"]
    rep = {}
    for pack, name, e0rec in (("task_b", "string600", e0b["string_label_by_window"]["w600"]),
                              ("task_b", "nli600", e0b["nli_label"]["premise600"]),
                              ("masked_pack", "string600", e0m["string_label_by_window"]["w600"])):
        mine = pa[pack]["by_label"][name]["i_all_rows"]
        rep[f"{pack}.{name}"] = repro_record(
            {s: {"agreement": e0rec[s]["agreement"], "kappa": e0rec[s]["cohens_kappa"],
                 "precision": e0rec[s]["precision"], "recall": e0rec[s]["recall"],
                 "confusion": e0rec[s]["confusion"]} for s in e0rec},
            {s: {"agreement": {"rate": mine[s]["agreement"]["estimate"], "ci95": mine[s]["agreement"]["ci95"]},
                 "kappa": mine[s]["cohens_kappa"]["estimate"], "precision": mine[s]["precision"]["estimate"],
                 "recall": mine[s]["recall"]["estimate"], "confusion": mine[s]["confusion"]} for s in mine},
            f"e0_human_alignment.json {pack} {name} (all rows) vs this script")

    # ================= (b) question level =====================================
    strict, loose, qweight, _, _ = HSI.human_question_labels()
    cell_w = HSI.load_cell_weights()
    keyb = {r["item_id"]: r for r in read_jsonl(PACK_B / "item_metadata.jsonl") if r.get("task") == "B"}
    shown_b = collections.defaultdict(set)
    items_b = collections.defaultdict(set)
    for r in rows_b:
        shown_b[r["qid"]].add(r["doc_id"])
        items_b[r["item_id"]].add(r["doc_id"])
    dup_same = all(len({frozenset(items_b[i]) for i in items_b if keyb[i]["qid"] == q}) == 1 for q in shown_b)
    affected_q_b = {r["qid"] for r in rows_b if r["gold_only_in_600_700"]}

    def any_str(q, docs):
        return any(VIS(qs[q]["gold_answer"], corpus.get(d, "")) for d in docs if corpus.get(d))

    def any_lab(lab, q, docs):
        return any(lab[(q, d)] for d in docs)

    qlab_b = {
        "string600_shown3": {q: any_str(q, shown_b[q]) for q in strict},
        "nli600_shown3": {q: any_lab(nli_mps, q, shown_b[q]) for q in strict},
        "nli600_dated_shown3": {q: any_lab(nli_dated, q, shown_b[q]) for q in strict},
        "string600_5passages": {q: any_str(q, ret_u[q][:5]) for q in strict},
        "nli600_5passages": {q: any_lab(nli_mps, q, ret_u[q][:5]) for q in strict},
        "nli600_dated_5passages": {q: any_lab(nli_dated, q, ret_u[q][:5]) for q in strict},
    }
    labelled_b = sorted(set(strict) & set(qs))

    def q_block(qids, auto, human, weight_of, B, seed):
        return agreement_block([(E0H.quiz_of(q), bool(auto[q]), bool(human[q]), weight_of(q)) for q in qids],
                               B, seed)

    qb = {}
    for name, auto in qlab_b.items():
        rec = {"passages": "the 3 passages humans labelled" if name.endswith("shown3")
               else "all 5 Hybrid passages shown to the generator (humans labelled only 3)"}
        for hname, human in (("human_YES_only", strict), ("human_YES_or_PARTIAL", loose)):
            items = sorted(keyb)
            rec[hname] = {
                "unweighted_sample_questions": q_block(labelled_b, auto, human, lambda q: 1.0, args.B, SEED_B),
                "unweighted_excluding_600_700_questions": q_block(
                    [q for q in labelled_b if q not in affected_q_b], auto, human, lambda q: 1.0, args.B, SEED_B),
                "design_weighted_items_population_estimate": agreement_block(
                    [(E0H.quiz_of(keyb[i]["qid"]), bool(auto[keyb[i]["qid"]]), bool(human[keyb[i]["qid"]]),
                      cell_w[(keyb[i]["regime"], bool(keyb[i]["auto_supported"]))]) for i in items],
                    args.B, SEED_B),
                "hsi_convention_question_weighted": q_block(labelled_b, auto, human, lambda q: qweight[q],
                                                            args.B, SEED_B),
            }
        qb[name] = rec
    e0_q_b = e0_ha["b_human_support_interaction"]["question_level_agreement_with_human_strict"]
    rep["task_b.question_level.string600_shown3_rate"] = repro_record(
        {"rate": e0_q_b["auto_600_shown3"]["rate_vs_human_strict"]},
        {"rate": qb["string600_shown3"]["human_YES_only"]["unweighted_sample_questions"]["agreement"]["estimate"]},
        "e0 auto_600_shown3 question-level agreement")
    rep["task_b.question_level.string600_5_rate"] = repro_record(
        {"rate": e0_q_b["auto_600_5docs"]["rate_vs_human_strict"]},
        {"rate": qb["string600_5passages"]["human_YES_only"]["unweighted_sample_questions"]["agreement"]["estimate"]},
        "e0 auto_600_5docs question-level agreement")

    # masked pack
    key_m = mhs.load_key()
    q_sup, q_loose, q_stratum, q_slice = {}, {}, {}, {}
    for r in rows_m:
        q_sup[r["qid"]] = q_sup.get(r["qid"], False) or r["human_bin"]
        q_loose[r["qid"]] = q_loose.get(r["qid"], False) or r["human_bin_loose"]
        q_stratum[r["qid"]], q_slice[r["qid"]] = r["stratum"], r["slice"]
    shown_m = collections.defaultdict(list)
    for iid, k in key_m.items():
        shown_m[k["qid"]].append(k["doc_id"])
    assert all(shown_m[q] == ret_m[q][:3] for q in shown_m)
    man_m = load_json(PACK_M / "sampling_manifest.json")
    w_m = {s: v["weight"] for s, v in man_m["strata"].items()}
    affected_q_m = {r["qid"] for r in rows_m if r["gold_only_in_600_700"]}
    qlab_m = {
        "string600_shown3": {q: any_str(q, shown_m[q]) for q in q_sup},
        "nli600_shown3": {q: any_lab(nli_cuda, q, shown_m[q]) for q in q_sup},
        "nli600_dated_shown3": {q: any_lab(nli_dated, q, shown_m[q]) for q in q_sup},
        "string600_5passages": {q: any_str(q, ret_m[q][:5]) for q in q_sup},
        "nli600_5passages": {q: any_lab(nli_cuda, q, ret_m[q][:5]) for q in q_sup},
        "nli600_dated_5passages": {q: any_lab(nli_dated, q, ret_m[q][:5]) for q in q_sup},
    }
    all_m = list(q_sup)                                   # score_masked_support.py order
    strat_m = [q for q in all_m if q_stratum[q] != "consistency_recheck"]
    qm = {}
    for name, auto in qlab_m.items():
        rec = {"passages": "the 3 passages humans labelled (masked top-3)" if name.endswith("shown3")
               else "all 5 masked Hybrid passages shown to the generator (humans labelled only 3)"}
        for hname, human in (("human_YES_only", q_sup), ("human_YES_or_PARTIAL", q_loose)):
            rec[hname] = {
                "unweighted_sample_all_210": q_block(all_m, auto, human, lambda q: 1.0, args.B, SEED_M),
                "unweighted_stratified_200": q_block(strat_m, auto, human, lambda q: 1.0, args.B, SEED_M),
                "unweighted_excluding_600_700_questions": q_block(
                    [q for q in all_m if q not in affected_q_m], auto, human, lambda q: 1.0, args.B, SEED_M),
                "design_weighted_200_population_estimate": q_block(
                    strat_m, auto, human, lambda q: w_m[q_stratum[q]], args.B, SEED_M),
            }
        qm[name] = rec
    e0_q_m = e0_ha["b_masked_human_support"]["question_level_agreement_with_human_strict"]
    rep["masked_pack.question_level.string600_shown3_rate"] = repro_record(
        {"rate": e0_q_m["auto_600_shown3"]["rate"]},
        {"rate": qm["string600_shown3"]["human_YES_only"]["unweighted_sample_all_210"]["agreement"]["estimate"]},
        "e0 masked auto_600_shown3 question-level agreement")

    # ================= typology rerun at 600 ==================================
    resdir = RAW / "resdir_nli600_mps"
    resdir.mkdir(parents=True, exist_ok=True)
    link = resdir / "nli_support_labels.jsonl"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(E0_NLI600, resdir))
    resdir_d = RAW / "resdir_nli600_dated_unmasked"
    resdir_d.mkdir(parents=True, exist_ok=True)
    dated_rows = read_jsonl(SRV / "dated600.jsonl")
    unmasked_pairs = {(q, d) for q in qs for d in ret_u[q][:5]}
    with (resdir_d / "nli_support_labels.jsonl").open("w", encoding="utf-8") as fh:
        for r in dated_rows:
            if (r["qid"], r["doc_id"]) in unmasked_pairs:
                fh.write(json.dumps({"qid": r["qid"], "doc_id": r["doc_id"], "nli_label": r["label"],
                                     "probs": r["probs"], "premise_chars": 600,
                                     "hypothesis": "dated"}) + "\n")
    typ = {}
    rts = {}
    f0 = RAW / "support_label_typology__stored_rerun.json"
    _, rts["stored"] = run_main(SLT, ["--output", f0], RAW / "support_label_typology__stored_rerun.log")
    rep["support_label_typology.py"] = repro_record(load_json(RES / "support_label_typology.json"),
                                                    load_json(f0), "entire output JSON")
    for tag, patches in (("w600_undated", {"gold_in_text": VIS, "RES": resdir}),
                         ("w600_dated", {"gold_in_text": VIS, "RES": resdir_d})):
        f = RAW / f"support_label_typology__{tag}.json"
        _, rts[tag] = run_main(SLT, ["--output", f], RAW / f"support_label_typology__{tag}.log", **patches)
        typ[tag] = load_json(f)
    old_t = load_json(f0)

    def typ_summary(d):
        return {lab: {"disagreement_rate": v["disagreement_rate"], "ci95": v["disagreement_ci95"],
                      "cells": v["cells"],
                      "by_slice": {s: {k: x[k] for k in ("n", "disagreement_rate", "ci95")}
                                   for s, x in v["by_slice"].items()},
                      "by_model_range": [min(x["disagreement_rate"] for x in v["by_model"].values()),
                                         max(x["disagreement_rate"] for x in v["by_model"].values())]}
                for lab, v in d["labels"].items()}

    out = {
        "step": "R2(a,b) -- human agreement of 600-char automatic labels; legacy 700-char human window",
        "meta": meta([E0_NLI600, SRV / "undated600.jsonl", SRV / "dated600.jsonl",
                      PACK_B / "item_metadata.jsonl", PACK_M / "item_metadata.jsonl",
                      PACK_B / "sampling_manifest.json", PACK_M / "sampling_manifest.json"]),
        "what_humans_saw": {
            "task_b": "text.replace('\\n',' ')[:700] (900/900 excerpts verified in e0)",
            "masked_pack": "' '.join(text.split())[:700] (630/630 excerpts verified in e0)",
            "generators": "raw text[:600]",
            "claim_scope": ("Agreement with LEGACY 700-character human labels. No human judged the exact "
                            "600-character generator input; these numbers do not validate relation "
                            "support on the exact model input.")},
        "labels": {
            "string600": "metrics.gold_in_text(gold, text[:600])",
            "nli600": ("DeBERTa-v3-large NLI, premise text[:600], undated hypothesis. Task B: the e0 MPS labels "
                       "(nli_support_labels_600.jsonl); masked pack: r1_nli_masked_600.jsonl (CUDA). CUDA and "
                       "MPS agree on 4,390/4,390 unmasked labels (r1_masked_nli.json)."),
            "nli600_dated": "same, hypothesis 'As of <question date>, the answer to ... is ...' (r2_dated_nli.py)",
            "legacy_string_700_as_shown_to_humans": "gold_in_text on the exact human excerpt window (reference)"},
        "a_passage_level": {
            "seeds": {"task_b": SEED_B, "masked_pack": SEED_M}, "B": args.B,
            "ci": "quiz-clustered percentile bootstrap (support_label_validation.py resampling)",
            "weighting": "unweighted passage rows (Task B 900 rows incl. 138 repeated passages; masked pack 630)",
            "exclusion_is_lower_bound": ("only rows where the GOLD STRING is visible to humans but not to the "
                                         "generator are excluded; other semantic content in chars 600-700 "
                                         "can also change a human judgement and is not detectable this way"),
            **pa},
        "b_question_level": {
            "restricted_subset_statement": (
                "Humans labelled 3 passages per question; generators saw 5. '*_shown3' labels are computed on "
                "exactly the 3 human-labelled doc ids and form a RESTRICTED ANNOTATION SUBSET. '*_5passages' "
                "labels are the analysis labels; comparing them with a 3-passage human label mixes coverage."),
            "task_b": {
                "n_questions": len(labelled_b), "n_items": len(keyb),
                "passages_per_item": 3, "duplicate_items_show_identical_passages": dup_same,
                "human_label": "any labelled passage YES (strict) / YES or PARTIAL; majority of 3, adjudicated",
                "weights": {f"{r}|supported={s}": w for (r, s), w in cell_w.items()},
                "weighting_note": ("design_weighted_items_population_estimate uses the pack's item design "
                                   "(300 items, weight_to_population); hsi_convention_question_weighted "
                                   "gives each distinct question its cell's item weight once, as "
                                   "human_support_interaction.py does"),
                "questions_with_600_700_gold_rows": sorted(affected_q_b),
                "seed": SEED_B, "B": args.B, "by_label": qb},
            "masked_pack": {
                "n_questions": len(all_m), "n_stratified": len(strat_m),
                "n_consistency_recheck_unweighted_only": len(all_m) - len(strat_m),
                "weights": w_m, "questions_with_600_700_gold_rows": sorted(affected_q_m),
                "seed": SEED_M, "B": args.B, "by_label": qm},
        },
        "typology_app_typology_rerun": {
            "script": "pipeline/support_label_typology.py (seed 20260804, B 2000, PARTIAL counts as supported)",
            "old_fulltext_string_nli1500": typ_summary(old_t),
            "w600_undated": typ_summary(typ["w600_undated"]),
            "w600_dated": typ_summary(typ["w600_dated"]),
            "runtimes_sec": rts},
        "reproduction": rep,
        "runtime_total_sec": round(time.time() - t_start, 1),
    }
    dump(out, OUT / "r2_human_alignment.json")

    print("repro:", {k: v["reproduced_exactly"] for k, v in rep.items()})
    for pack in ("task_b", "masked_pack"):
        print(pack, {k: v for k, v in pa[pack]["rows_gold_string_only_in_chars_600_700"].items() if k != "rows"})
        for name, v in pa[pack]["by_label"].items():
            for sub in ("i_all_rows", "ii_excluding_gold_only_in_600_700"):
                for s in ("human_YES_only", "human_YES_or_PARTIAL"):
                    x = v[sub][s]
                    print(f"  {name:38s}{sub[:6]:7s}{s[6:13]:8s} n={x['n_units']:4d} "
                          f"P {x['precision']['estimate']:.3f} R {x['recall']['estimate']:.3f} "
                          f"k {x['cohens_kappa']['estimate']:.3f} {x['cohens_kappa']['ci95']} "
                          f"A {x['agreement']['estimate']:.3f}")
    for pack, qq in (("task_b", qb), ("masked", qm)):
        for name, v in qq.items():
            x = v["human_YES_only"]
            ks = [k for k in x]
            print(f"Q {pack:7s}{name:24s}" + "  ".join(
                f"{k[:14]} A {x[k]['agreement']['estimate']:.3f} k {x[k]['cohens_kappa']['estimate']:.3f}"
                for k in ks))
    print("wrote", OUT / "r2_human_alignment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
