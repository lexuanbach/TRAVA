#!/usr/bin/env python3
"""R8(2): calibrate the R8 judge against the existing human labels.

The judge (r8_judge_run.py) labelled the exact 600-character generator input and
the 700-character excerpt each pack's annotators actually saw. This script

  (a) passage level, on the human-annotated rows of BOTH packs, at BOTH windows:
      accuracy, precision and recall of YES, Cohen's kappa, each with the
      quiz-clustered 95% percentile bootstrap of support_label_validation.py
      (imported from r2_human_alignment.agreement_block, same seeds, same draw
      sequence, so the intervals are comparable with the stored ones), under the
      strict-YES and the YES-or-PARTIAL binarisation, in ONE table together with
      the string@600, NLI@600, dated-NLI@600 and legacy string@700 labels of
      results/analysis/r2_human_alignment.json;
  (b) the judge's 600-vs-700 disagreement on those same rows (ternary confusion
      and both binarisations);
  (c) question level: judge-derived support (any shown passage YES) against the
      human question-level label, on the restricted 3-passage annotation subset
      and on all 5 passages the generator was shown;
  (d) sanity: the stored human-label statistics this compares against are
      recomputed from scratch and diffed against r2_human_alignment.json; the
      judge label distribution, the rate at which the unconstrained greedy token
      is not a label, and the free-generation check are reported.

Binarisation convention. The judge emits the same ternary label as the humans,
so it is binarised the same way the human label is in each column: "strict YES"
= YES only for both sides, "YES or PARTIAL" = YES or PARTIAL for both sides.
string/NLI labels are binary already and are compared exactly as in R2.

Nothing is modified. Writes results/analysis/r8_judge_validation.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from _e0util import OUT, PROC, dump, load_json, meta, repro_record

import e0_human_alignment as E0H                                  # noqa: E402
import human_support_interaction as HSI                           # noqa: E402
import r2_human_alignment as R2                                   # noqa: E402

JUDGE = OUT / "r8_judge_labels.jsonl"
SEED_B, SEED_M = R2.SEED_B, R2.SEED_M
PACK_B, PACK_M = E0H.PACK_B, E0H.PACK_M


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


LABEL_ORDER = ("YES", "PARTIAL", "NO")


def conservative(r):
    """Tie-break toward NO instead of toward YES.

    The judge label is the argmax of the three renormalised label probabilities
    in (YES, PARTIAL, NO) order, so an exact float32 tie resolves toward the
    earlier (more positive) label. This resolves it toward the later one, and is
    reported as a sensitivity."""
    mx = max(r["probs"].values())
    return [l for l in LABEL_ORDER if r["probs"][l] == mx][-1]


def judge_map():
    """(qid, doc_id, window) -> row."""
    return {(r["qid"], r["doc_id"], r["window"]): r for r in read_jsonl(JUDGE)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    args = ap.parse_args()
    t0 = time.time()

    J = judge_map()
    all_judge = list(J.values())
    qs, corpus = E0H.load_questions(), E0H.load_corpus()
    mhs = E0H.load_mhs()
    r2 = load_json(OUT / "r2_human_alignment.json")
    ret_u = load_json(PROC / "retrieval.json")["hybrid"]
    ret_m = load_json(PROC / "retrieval_masked.json")["hybrid"]

    nli_u = E0H.nli_map(R2.E0_NLI600)                       # unmasked, e0 MPS
    nli_cuda = R2.server_labels("undated600")               # all 4,959 pairs
    nli_dated = R2.server_labels("dated600")

    # ---------------- rows, exactly as R2 builds them ---------------------
    rows_b = E0H.packb_rows(qs, corpus)
    rows_m = E0H.masked_rows(qs, corpus, mhs)
    for r in rows_b:
        k = (r["qid"], r["doc_id"])
        r["nli600"], r["nli600_dated"] = nli_u.get(k), nli_dated.get(k)
        r["gold_only_in_600_700"] = bool(r["string_w700nl"] and not r["string_w600"])
    for r in rows_m:
        k = (r["qid"], r["doc_id"])
        r["nli600"], r["nli600_dated"] = nli_cuda.get(k), nli_dated.get(k)
        r["gold_only_in_600_700"] = bool(r["string_w700ws"] and not r["string_w600"])

    W700 = {"task_b": "w700nl", "masked_pack": "w700ws"}
    missing = []
    for pack, rows in (("task_b", rows_b), ("masked_pack", rows_m)):
        for r in rows:
            for w in ("w600", W700[pack]):
                j = J.get((r["qid"], r["doc_id"], w))
                if j is None:
                    missing.append((pack, r["qid"], r["doc_id"], w))
                    continue
                tag = "judge600" if w == "w600" else "judge700"
                r[f"{tag}_raw"] = j["label"]
                r[f"{tag}_prob"] = j["prob"]
                r[f"{tag}_yes"] = j["label"] == "YES"
                r[f"{tag}_yes_or_partial"] = j["label"] in ("YES", "PARTIAL")
    assert not missing, missing[:10]

    # ---------------- (a) passage level, one table ------------------------
    # Column key -> (attribute for strict-YES comparison,
    #                attribute for YES-or-PARTIAL comparison)
    def cols(pack):
        legacy = "string_w700nl" if pack == "task_b" else "string_w700ws"
        return collections.OrderedDict([
            ("judge600", ("judge600_yes", "judge600_yes_or_partial")),
            ("judge700", ("judge700_yes", "judge700_yes_or_partial")),
            ("string600", ("string_w600", "string_w600")),
            ("nli600", ("nli600", "nli600")),
            ("nli600_dated", ("nli600_dated", "nli600_dated")),
            ("legacy_string_700_as_shown_to_humans", (legacy, legacy)),
        ])

    def block(rows, auto_strict, auto_loose, seed, exclude=False):
        sel = [r for r in rows if not (exclude and r["gold_only_in_600_700"])]
        out = {}
        for strict, name, akey in ((True, "human_YES_only", auto_strict),
                                   (False, "human_YES_or_PARTIAL", auto_loose)):
            hb = "human_bin" if strict else "human_bin_loose"
            units = [(r["quiz"], bool(r[akey]), bool(r[hb]), 1.0) for r in sel]
            out[name] = R2.agreement_block(units, args.B, seed)
        return out

    passage = {}
    for pack, rows, seed in (("task_b", rows_b, SEED_B), ("masked_pack", rows_m, SEED_M)):
        blk = {"n_passage_rows": len(rows),
               "n_unique_qid_doc": len({(r["qid"], r["doc_id"]) for r in rows}),
               "n_questions": len({r["qid"] for r in rows}),
               "human_window": r2["what_humans_saw"]["task_b" if pack == "task_b" else "masked_pack"],
               "by_label": {}}
        for name, (a_s, a_l) in cols(pack).items():
            blk["by_label"][name] = {
                "i_all_rows": block(rows, a_s, a_l, seed),
                "ii_excluding_gold_only_in_600_700": block(rows, a_s, a_l, seed, exclude=True)}
        # judge ternary confusion against the human ternary label
        for tag in ("judge600", "judge700"):
            blk[f"{tag}_ternary_confusion_vs_human"] = {
                f"human_{h}|judge_{a}": n for (h, a), n in sorted(
                    collections.Counter((r["human"], r[f"{tag}_raw"]) for r in rows).items())}
            blk[f"{tag}_label_distribution"] = dict(
                collections.Counter(r[f"{tag}_raw"] for r in rows))
        blk["human_label_distribution"] = dict(collections.Counter(r["human"] for r in rows))
        passage[pack] = blk

    # ---------------- (b) 600 vs 700, judge held fixed --------------------
    window_cmp = {}
    for pack, rows in (("task_b", rows_b), ("masked_pack", rows_m)):
        uniq = {}
        for r in rows:
            uniq[(r["qid"], r["doc_id"])] = r
        u = list(uniq.values())
        diff = [r for r in u if r["judge600_raw"] != r["judge700_raw"]]
        window_cmp[pack] = {
            "n_unique_passages": len(u),
            "n_rows": len(rows),
            "n_label_changes_600_vs_700_unique_passages": len(diff),
            "n_label_changes_600_vs_700_rows": sum(
                1 for r in rows if r["judge600_raw"] != r["judge700_raw"]),
            "rate_unique": round(len(diff) / len(u), 4),
            "ternary_confusion_600_rows_700_cols": {
                f"{a}->{b}": n for (a, b), n in sorted(collections.Counter(
                    (r["judge600_raw"], r["judge700_raw"]) for r in u).items())},
            "binarised_changes": {
                "strict_YES": sum(1 for r in u if r["judge600_yes"] != r["judge700_yes"]),
                "YES_or_PARTIAL": sum(1 for r in u
                                      if r["judge600_yes_or_partial"] != r["judge700_yes_or_partial"])},
            "changes_among_gold_only_in_600_700_rows": {
                "n_such_unique_passages": sum(1 for r in u if r["gold_only_in_600_700"]),
                "n_changed": sum(1 for r in u if r["gold_only_in_600_700"]
                                 and r["judge600_raw"] != r["judge700_raw"]),
                "detail": [{"qid": r["qid"], "doc_id": r["doc_id"], "human": r["human"],
                            "judge600": r["judge600_raw"], "judge700": r["judge700_raw"]}
                           for r in u if r["gold_only_in_600_700"]]},
            "examples_changed": [{"qid": r["qid"], "doc_id": r["doc_id"], "human": r["human"],
                                  "judge600": r["judge600_raw"], "judge700": r["judge700_raw"]}
                                 for r in diff[:25]],
        }

    # ---------------- (c) question level ----------------------------------
    strict_b, loose_b, qweight_b, _, _ = HSI.human_question_labels()
    cell_w = HSI.load_cell_weights()
    keyb = {r["item_id"]: r for r in read_jsonl(PACK_B / "item_metadata.jsonl")
            if r.get("task") == "B"}
    shown3_b = collections.defaultdict(set)
    for r in rows_b:
        shown3_b[r["qid"]].add(r["doc_id"])
    labelled_b = sorted(set(strict_b) & set(qs))

    def any_judge(q, docs, loose=False):
        key = "yes_or_partial" if loose else "yes"
        vals = []
        for d in docs:
            j = J.get((q, d, "w600"))
            if j is None:
                return None
            vals.append(j["label"] in (("YES", "PARTIAL") if loose else ("YES",)))
        return any(vals)

    def any_nli(lab, q, docs):
        return any(lab.get((q, d), False) for d in docs)

    def any_str(q, docs):
        return any(R2.VIS(qs[q]["gold_answer"], corpus.get(d, "")) for d in docs if corpus.get(d))

    qlab_b = {}
    for tag, docs_of in (("shown3", lambda q: sorted(shown3_b[q])),
                         ("5passages", lambda q: ret_u[q][:5])):
        qlab_b[f"judge600_YES_{tag}"] = {q: any_judge(q, docs_of(q)) for q in labelled_b}
        qlab_b[f"judge600_YESorPARTIAL_{tag}"] = {q: any_judge(q, docs_of(q), True) for q in labelled_b}
        qlab_b[f"string600_{tag}"] = {q: any_str(q, docs_of(q)) for q in labelled_b}
        qlab_b[f"nli600_{tag}"] = {q: any_nli(nli_u, q, docs_of(q)) for q in labelled_b}

    def q_block(qids, auto, human, weight_of, seed):
        return R2.agreement_block(
            [(E0H.quiz_of(q), bool(auto[q]), bool(human[q]), weight_of(q)) for q in qids],
            args.B, seed)

    affected_q_b = {r["qid"] for r in rows_b if r["gold_only_in_600_700"]}
    qb = {}
    for name, auto in qlab_b.items():
        rec = {"passages": ("the 3 passages humans labelled (RESTRICTED ANNOTATION SUBSET)"
                            if name.endswith("shown3") else
                            "all 5 unmasked Hybrid passages shown to the generator")}
        for hname, human in (("human_YES_only", strict_b), ("human_YES_or_PARTIAL", loose_b)):
            rec[hname] = {
                "unweighted_sample_questions": q_block(labelled_b, auto, human, lambda q: 1.0, SEED_B),
                "unweighted_excluding_600_700_questions": q_block(
                    [q for q in labelled_b if q not in affected_q_b], auto, human,
                    lambda q: 1.0, SEED_B),
                "design_weighted_items_population_estimate": R2.agreement_block(
                    [(E0H.quiz_of(keyb[i]["qid"]), bool(auto[keyb[i]["qid"]]),
                      bool(human[keyb[i]["qid"]]),
                      cell_w[(keyb[i]["regime"], bool(keyb[i]["auto_supported"]))])
                     for i in sorted(keyb)], args.B, SEED_B),
                "hsi_convention_question_weighted": q_block(labelled_b, auto, human,
                                                            lambda q: qweight_b[q], SEED_B),
            }
        qb[name] = rec

    # masked pack
    key_m = mhs.load_key()
    q_sup, q_loose, q_stratum = {}, {}, {}
    for r in rows_m:
        q_sup[r["qid"]] = q_sup.get(r["qid"], False) or r["human_bin"]
        q_loose[r["qid"]] = q_loose.get(r["qid"], False) or r["human_bin_loose"]
        q_stratum[r["qid"]] = r["stratum"]
    shown3_m = collections.defaultdict(list)
    for iid, k in key_m.items():
        shown3_m[k["qid"]].append(k["doc_id"])
    assert all(shown3_m[q] == ret_m[q][:3] for q in shown3_m)
    w_m = {s: v["weight"] for s, v in
           load_json(PACK_M / "sampling_manifest.json")["strata"].items()}
    all_m = list(q_sup)
    strat_m = [q for q in all_m if q_stratum[q] != "consistency_recheck"]
    affected_q_m = {r["qid"] for r in rows_m if r["gold_only_in_600_700"]}

    qlab_m = {}
    for tag, docs_of in (("shown3", lambda q: shown3_m[q]),
                         ("5passages", lambda q: ret_m[q][:5])):
        qlab_m[f"judge600_YES_{tag}"] = {q: any_judge(q, docs_of(q)) for q in all_m}
        qlab_m[f"judge600_YESorPARTIAL_{tag}"] = {q: any_judge(q, docs_of(q), True) for q in all_m}
        qlab_m[f"string600_{tag}"] = {q: any_str(q, docs_of(q)) for q in all_m}
        qlab_m[f"nli600_{tag}"] = {q: any_nli(nli_cuda, q, docs_of(q)) for q in all_m}
    qm = {}
    for name, auto in qlab_m.items():
        rec = {"passages": ("the 3 masked passages humans labelled (RESTRICTED ANNOTATION SUBSET)"
                            if name.endswith("shown3") else
                            "all 5 masked Hybrid passages shown to the generator")}
        for hname, human in (("human_YES_only", q_sup), ("human_YES_or_PARTIAL", q_loose)):
            rec[hname] = {
                "unweighted_sample_all_210": q_block(all_m, auto, human, lambda q: 1.0, SEED_M),
                "unweighted_stratified_200": q_block(strat_m, auto, human, lambda q: 1.0, SEED_M),
                "unweighted_excluding_600_700_questions": q_block(
                    [q for q in all_m if q not in affected_q_m], auto, human, lambda q: 1.0, SEED_M),
                "design_weighted_200_population_estimate": q_block(
                    strat_m, auto, human, lambda q: w_m[q_stratum[q]], SEED_M),
            }
        qm[name] = rec

    # ---------------- (d) sanity / reproduction ---------------------------
    rep = {}
    for pack in ("task_b", "masked_pack"):
        for name in ("string600", "nli600", "nli600_dated",
                     "legacy_string_700_as_shown_to_humans"):
            stored = r2["a_passage_level"][pack]["by_label"][name]
            mine = passage[pack]["by_label"][name]
            rep[f"r2_passage.{pack}.{name}"] = repro_record(
                {sub: {s: {k: stored[sub][s][k]["estimate"] for k in
                           ("agreement", "cohens_kappa", "precision", "recall")}
                       for s in stored[sub]} for sub in stored},
                {sub: {s: {k: mine[sub][s][k]["estimate"] for k in
                           ("agreement", "cohens_kappa", "precision", "recall")}
                       for s in mine[sub]} for sub in mine},
                f"r2_human_alignment.json a_passage_level.{pack}.{name} point estimates")
    for pack, mine_q, stored_key in (("task_b", qb, "task_b"), ("masked_pack", qm, "masked_pack")):
        for name in ("string600", "nli600"):
            for tag, sub in (("shown3", "shown3"), ("5passages", "5passages")):
                s_name = f"{name}_{sub}"
                if s_name not in r2["b_question_level"][stored_key]["by_label"]:
                    continue
                skey = ("unweighted_sample_questions" if pack == "task_b"
                        else "unweighted_sample_all_210")
                stored = r2["b_question_level"][stored_key]["by_label"][s_name][
                    "human_YES_only"][skey]
                mine = mine_q[f"{name}_{tag}"]["human_YES_only"][skey]
                rep[f"r2_question.{pack}.{s_name}"] = repro_record(
                    {k: stored[k]["estimate"] for k in
                     ("agreement", "cohens_kappa", "precision", "recall")},
                    {k: mine[k]["estimate"] for k in
                     ("agreement", "cohens_kappa", "precision", "recall")},
                    f"r2_human_alignment.json b_question_level.{stored_key}.{s_name} "
                    f"human_YES_only {skey}")

    gen_check = None
    gc = OUT / "r8_raw" / "r8_judge_gencheck.jsonl"
    if gc.is_file():
        jobs = {j["job_id"]: j for j in read_jsonl(OUT / "r8_bundle" / "r8_jobs.jsonl")}
        lab_by_job = {}
        for r in read_jsonl(OUT / "r8_raw" / "r8_judge_labels_raw.jsonl"):
            lab_by_job[r["job_id"]] = r["label"]
        rows = read_jsonl(gc)
        agree = sum(1 for r in rows if r["parsed"] == lab_by_job.get(r["job_id"]))
        gen_check = {
            "n": len(rows),
            "n_free_generation_equals_constrained_label": agree,
            "rate": round(agree / len(rows), 4) if rows else None,
            "n_free_text_not_a_label": sum(1 for r in rows if r["parsed"] not in
                                           ("YES", "PARTIAL", "NO")),
            "free_text_distribution": dict(collections.Counter(r["text"] for r in rows).most_common(10)),
            "disagreements": [{"job_id": r["job_id"], "free": r["text"],
                               "constrained": lab_by_job.get(r["job_id"])}
                              for r in rows if r["parsed"] != lab_by_job.get(r["job_id"])][:20]}

    judge_diag = {
        "n_rows": len(all_judge),
        "model": sorted({r["model"] for r in all_judge}),
        "label_distribution_overall": dict(collections.Counter(r["label"] for r in all_judge)),
        "label_distribution_by_window": {
            w: dict(collections.Counter(r["label"] for r in all_judge if r["window"] == w))
            for w in sorted({r["window"] for r in all_judge})},
        "unparseable": {
            "definition": ("the unconstrained greedy next token is not the first token of "
                           "YES / PARTIAL / NO; the stored label is always the argmax of the "
                           "renormalised three-label distribution, so no row is dropped"),
            "n_argmax_outside_label_set": sum(1 for r in all_judge if not r["argmax_in_label_set"]),
            "rate": round(sum(1 for r in all_judge if not r["argmax_in_label_set"]) / len(all_judge), 5),
            "n_argmax_label_differs_from_constrained": sum(
                1 for r in all_judge if r["argmax_in_label_set"] and r["argmax_label"] != r["label"]),
            "argmax_tokens_outside_label_set": dict(collections.Counter(
                r["argmax_token"] for r in all_judge if not r["argmax_in_label_set"]).most_common(10)),
            "label_mass_min": round(min(r["label_mass"] for r in all_judge), 5),
            "label_mass_mean": round(sum(r["label_mass"] for r in all_judge) / len(all_judge), 5),
            "n_label_mass_below_0.5": sum(1 for r in all_judge if r["label_mass"] < 0.5)},
        "exact_ties": {
            "definition": ("rows where the top two of the three renormalised label "
                           "probabilities are exactly equal in float32; the stored label is the "
                           "first in (YES, PARTIAL, NO) order, which is why the unconstrained "
                           "vocabulary argmax can name the other member of the tie"),
            "n": sum(1 for r in all_judge
                     if sorted(r["probs"].values())[-1] == sorted(r["probs"].values())[-2]),
            "pairs": dict(collections.Counter(
                "/".join(sorted(k for k, v in r["probs"].items()
                                if v == max(r["probs"].values())))
                for r in all_judge
                if sorted(r["probs"].values())[-1] == sorted(r["probs"].values())[-2])),
            "n_on_annotated_rows": sum(
                1 for r in all_judge
                if sorted(r["probs"].values())[-1] == sorted(r["probs"].values())[-2]
                and (r["in_pack"] if "in_pack" in r else
                     ("taskb_pack" in r["condition"] or "masked_pack" in r["condition"]))),
            "n_strict_YES_affected": sum(
                1 for r in all_judge
                if sorted(r["probs"].values())[-1] == sorted(r["probs"].values())[-2]
                and "YES" in {k for k, v in r["probs"].items() if v == max(r["probs"].values())})},
        "prob_quantiles": {
            q: round(sorted(r["prob"] for r in all_judge)[int(q * (len(all_judge) - 1))], 4)
            for q in (0.01, 0.05, 0.25, 0.5, 0.75)},
        "free_generation_check": gen_check,
    }
    # -- tie sensitivity: point estimates under the conservative tie-break ---
    tie_sens = {}
    for pack, rows in (("task_b", rows_b), ("masked_pack", rows_m)):
        rec = {}
        for tag, w in (("judge600", "w600"),
                       ("judge700", W700[pack])):
            changed = 0
            cons = {}
            for r in rows:
                j = J[(r["qid"], r["doc_id"], w)]
                c = conservative(j)
                cons[(r["qid"], r["doc_id"])] = c
                changed += c != j["label"]
            sub = {"n_rows_relabelled_by_conservative_tie_break": changed}
            for strict, name in ((True, "human_YES_only"), (False, "human_YES_or_PARTIAL")):
                hb = "human_bin" if strict else "human_bin_loose"
                pos = ("YES",) if strict else ("YES", "PARTIAL")
                tp = fp = fn = tn = 0
                for r in rows:
                    a = cons[(r["qid"], r["doc_id"])] in pos
                    h = bool(r[hb])
                    tp, fp, fn, tn = (tp + (a and h), fp + (a and not h),
                                      fn + (h and not a), tn + (not a and not h))
                st = R2._stats(tp, fp, fn, tn)
                sub[name] = {k: round(v, 4) for k, v in st.items()}
                sub[name]["delta_vs_stored_tie_break"] = {
                    k: round(st[k] - passage[pack]["by_label"][tag]["i_all_rows"][name][k]["estimate"], 4)
                    for k in st}
            rec[tag] = sub
        tie_sens[pack] = rec
    judge_diag["tie_break_sensitivity_point_estimates"] = tie_sens

    # judge600 vs judge700 over all annotated passages (both packs together)
    both = []
    for pack, rows in (("task_b", rows_b), ("masked_pack", rows_m)):
        seen = set()
        for r in rows:
            k = (r["qid"], r["doc_id"])
            if k in seen:
                continue
            seen.add(k)
            both.append(r)
    judge_diag["judge600_vs_judge700_all_annotated_unique_passages"] = {
        "n": len(both),
        "n_same_ternary": sum(1 for r in both if r["judge600_raw"] == r["judge700_raw"]),
        "agreement": round(sum(1 for r in both if r["judge600_raw"] == r["judge700_raw"]) / len(both), 4),
        "n_changes": sum(1 for r in both if r["judge600_raw"] != r["judge700_raw"])}

    out = {
        "step": "R8(2) -- judge-vs-human calibration on the annotated rows, both windows",
        "meta": meta([JUDGE, OUT / "r8_judge_bundle_manifest.json"]),
        "what_this_establishes": (
            "An INDEPENDENT AUTOMATIC JUDGE (open-weight LLM) labelled the exact 600-character "
            "generator input with the human Task B `supports` rubric, and the 700-character "
            "excerpt the annotators saw. The agreement numbers calibrate that judge against the "
            "LEGACY 700-character human labels. They do NOT constitute human judgement of the "
            "600-character input: no human has seen it. The judge@600 column is the label of "
            "record for the exact model input; the judge@700 column is what makes the "
            "comparison with humans window-matched."),
        "judge": {"model": judge_diag["model"],
                  "decoding": "greedy, output constrained to YES / PARTIAL / NO by scoring the "
                              "three label continuations at the first generated token",
                  "bundle_manifest": "results/analysis/r8_judge_bundle_manifest.json",
                  "labels_file": "results/analysis/r8_judge_labels.jsonl"},
        "binarisation": ("strict: YES only, on BOTH sides; loose: YES or PARTIAL, on both sides. "
                         "string/NLI labels are binary and are compared exactly as in R2."),
        "B": args.B, "seeds": {"task_b": SEED_B, "masked_pack": SEED_M},
        "ci": "quiz-clustered percentile bootstrap (support_label_validation.py resampling, "
              "imported from r2_human_alignment.agreement_block)",
        "a_passage_level": passage,
        "b_window_600_vs_700": window_cmp,
        "c_question_level": {
            "restricted_subset_statement": r2["b_question_level"]["restricted_subset_statement"],
            "task_b": {"n_questions": len(labelled_b), "n_items": len(keyb),
                       "weights": {f"{r}|supported={s}": w for (r, s), w in cell_w.items()},
                       "by_label": qb},
            "masked_pack": {"n_questions": len(all_m), "n_stratified": len(strat_m),
                            "weights": w_m, "by_label": qm}},
        "d_judge_diagnostics": judge_diag,
        "reproduction_of_r2_stored_statistics": rep,
        "runtime_total_sec": round(time.time() - t0, 1),
    }
    dump(out, OUT / "r8_judge_validation.json")

    print("repro:", {k: v["reproduced_exactly"] for k, v in rep.items()})
    for pack in ("task_b", "masked_pack"):
        print("=" * 100)
        print(pack, passage[pack]["n_passage_rows"], "rows")
        for name, v in passage[pack]["by_label"].items():
            for s in ("human_YES_only", "human_YES_or_PARTIAL"):
                x = v["i_all_rows"][s]
                print(f"  {name:38s}{s[6:]:18s} n={x['n_units']:4d} "
                      f"A {x['agreement']['estimate']:.4f}{x['agreement']['ci95']} "
                      f"P {x['precision']['estimate']:.4f} R {x['recall']['estimate']:.4f} "
                      f"k {x['cohens_kappa']['estimate']:.4f}{x['cohens_kappa']['ci95']}")
        print("  600vs700:", {k: v for k, v in window_cmp[pack].items()
                              if k not in ("examples_changed", "changes_among_gold_only_in_600_700_rows",
                                           "ternary_confusion_600_rows_700_cols")})
    for pack, qq in (("task_b", qb), ("masked", qm)):
        for name, v in qq.items():
            x = v["human_YES_only"]
            k0 = "unweighted_sample_questions" if pack == "task_b" else "unweighted_sample_all_210"
            print(f"Q {pack:7s}{name:34s} A {x[k0]['agreement']['estimate']:.4f} "
                  f"k {x[k0]['cohens_kappa']['estimate']:.4f} "
                  f"P {x[k0]['precision']['estimate']:.4f} R {x[k0]['recall']['estimate']:.4f}")
    print(json.dumps(judge_diag["unparseable"], indent=1))
    print("wrote", OUT / "r8_judge_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
