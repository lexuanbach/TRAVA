#!/usr/bin/env python3
"""R2(c): dated-hypothesis NLI sensitivity (a sensitivity, NOT a validation).

The e0 NLI label uses the hypothesis
    The answer to "<question>" is <gold>.
which carries no date. This relabels the same 600-character premises with
    As of <Month D, YYYY>, the answer to "<question>" is <gold>.
(date = questions.jsonl 'date', the date the temporal mask uses), with the same
checkpoint, tokenizer call, max_length and batching (GPU server, CUDA; token ids
from the local tokenizer; see r1_build_nli_bundle.py), on
    unmasked  retrieval.json['hybrid'][:5]          4,390 tuples
    masked    retrieval_masked.json['hybrid'][:5]   4,390 tuples
and recomputes
  * the unmasked NLI support interaction (pipeline/support_interaction.py, label
    file swapped in memory; seed 20260802, B 5000, Static and Fresh,
    question-pooled arm means, quiz-cluster CI) and its unsupported-cost slope,
    after reproducing the e0 undated 600-char values exactly;
  * the masked NLI support interaction (masked_counterfactual.py block via
    r1_masked_nli.masked_interaction; seed 20260803, Static only) and slope,
    after reproducing r1's undated masked values exactly;
  * human agreement for dated vs undated labels (computed in
    r2_human_alignment.py; copied here side by side);
  * the routing GOLD_NLI diagnostic with the dated label.
No human ever judged whether a passage supports the DATED proposition, and the
NLI model was not trained for temporal scoping, so nothing here shows that the
label resolves time correctly.

Outputs results/analysis/r2_dated_nli.json, results/analysis/r2_nli_dated_600.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time

from _e0util import OUT, PROC, dump, load_json, meta, repro_record, run_main

import support_interaction as SI                                  # noqa: E402
import r1_masked_nli as R1                                        # noqa: E402

RAW = OUT / "r2_raw"
SRV = OUT / "r1_raw" / "server_out"
MANIFEST = OUT / "r1_raw" / "bundle_manifest.json"
DATED_OUT = OUT / "r2_nli_dated_600.jsonl"
E0_NLI600 = OUT / "nli_support_labels_600.jsonl"
MONTH_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|"
                      r"October|November|December)\b")
YEAR_RE = re.compile(r"\b(19|20)\d\d\b")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_resdir(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    f = path / "nli_support_labels.jsonl"
    if f.is_symlink() or f.exists():
        f.unlink()
    with f.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    args = ap.parse_args()
    t_start = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    qs = SI.load_questions()
    manifest = load_json(MANIFEST)
    tup = {(t["qid"], t["doc_id"]): t for t in manifest["tuples"]}
    ret_u = load_json(PROC / "retrieval.json")["hybrid"]
    ret_m = load_json(PROC / "retrieval_masked.json")["hybrid"]

    dated, chk_d = R1.load_server_labels("dated600", qs, manifest)
    undated, chk_u = R1.load_server_labels("undated600", qs, manifest)
    with DATED_OUT.open("w", encoding="utf-8") as fh:
        for k in sorted(dated):
            r, t = dated[k], tup[k]
            fh.write(json.dumps({"qid": k[0], "doc_id": k[1], "sets": t["sets"], "rank": t["rank"],
                                 "premise_sha256": r["premise_sha256"], "premise_chars": 600,
                                 "hypothesis": "dated", "hypothesis_sha256": r["hypothesis_sha256"],
                                 "question_date": qs[k[0]]["date"],
                                 "label": r["label"], "probs": r["probs"]}) + "\n")

    # ---- label transitions ----------------------------------------------------
    def ctx_trans(keys):
        c = collections.Counter(f"{undated[k]['label']}->{dated[k]['label']}" for k in keys)
        return dict(sorted(c.items()))

    u_keys = [(q, d) for q in sorted(qs) for d in ret_u[q][:5]]
    m_keys = [(q, d) for q in sorted(qs) for d in ret_m[q][:5]]
    has_date = {q: bool(MONTH_RE.search(qs[q]["question"]) or YEAR_RE.search(qs[q]["question"])) for q in qs}

    def q_any(lab, ret):
        return {q: any(lab[(q, d)]["label"].startswith("entail") for d in ret[q][:5]) for q in qs}

    qu_u, qd_u = q_any(undated, ret_u), q_any(dated, ret_u)
    qu_m, qd_m = q_any(undated, ret_m), q_any(dated, ret_m)

    def q_trans(a, b):
        out = {}
        for reg in ("static", "fresh"):
            for grp in ("all", "question_text_has_explicit_date", "no_explicit_date"):
                c = collections.Counter()
                for q in qs:
                    if (qs[q]["year"] >= SI.BOUNDARY_YEAR) != (reg == "fresh"):
                        continue
                    if grp != "all" and has_date[q] != (grp == "question_text_has_explicit_date"):
                        continue
                    c[("E" if a[q] else "N") + "->" + ("E" if b[q] else "N")] += 1
                out[f"{reg}|{grp}"] = {**dict(c), "n": sum(c.values()),
                                       "n_supported_undated": c["E->E"] + c["E->N"],
                                       "n_supported_dated": c["E->E"] + c["N->E"]}
        return out

    # ---- unmasked interaction via support_interaction.py ----------------------
    rd_und = RAW / "resdir_nli600_mps_e0"
    rd_und.mkdir(parents=True, exist_ok=True)
    link = rd_und / "nli_support_labels.jsonl"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(E0_NLI600, rd_und))
    rd_dat = write_resdir(RAW / "resdir_nli600_dated_unmasked_si",
                          [{"qid": q, "doc_id": d, "nli_label": dated[(q, d)]["label"],
                            "probs": dated[(q, d)]["probs"]} for q, d in u_keys])
    rts = {}
    f_u, f_d = RAW / "support_interaction__nli600_undated.json", RAW / "support_interaction__nli600_dated.json"
    _, rts["unmasked_undated"] = run_main(SI, ["-B", args.B, "--output", f_u],
                                          RAW / "support_interaction__nli600_undated.log", RES=rd_und)
    _, rts["unmasked_dated"] = run_main(SI, ["-B", args.B, "--output", f_d],
                                        RAW / "support_interaction__nli600_dated.log", RES=rd_dat)
    o, n = load_json(f_u), load_json(f_d)
    e0n = load_json(OUT / "e0_nli_visible.json")["support_interaction_nli"]
    keep = ("interaction", "ci95", "p_plus_one", "gain_supported", "gain_unsupported",
            "n_supported", "n_unsupported", "n_quizzes")
    rep = {"unmasked_undated_equals_e0_nli_visible": repro_record(
        {m: {reg: v[reg]["new_nli_premise600"] for reg in v} for m, v in e0n["per_model"].items()},
        {m: {reg: {k: o["labels"]["nli"][m][reg][k] for k in keep} for reg in o["labels"]["nli"][m]}
         for m in o["labels"]["nli"]},
        "e0_nli_visible.json support_interaction_nli.per_model[*].new_nli_premise600"),
        "server_label_files": {"dated600": chk_d, "undated600": chk_u}}
    unmasked = {}
    for m in o["labels"]["nli"]:
        unmasked[m] = {}
        for reg in ("static", "fresh"):
            a, b = o["labels"]["nli"][m].get(reg), n["labels"]["nli"][m].get(reg)
            unmasked[m][reg] = {"undated_600": {k: a[k] for k in keep} if a else None,
                                "dated_600": {k: b[k] for k in keep} if b else None,
                                "change_dated_minus_undated": round(b["interaction"] - a["interaction"], 3)
                                if a and b else None,
                                "ci_excludes_zero": {"undated": R1.excludes_zero(a), "dated": R1.excludes_zero(b)}}

    # ---- masked interaction -----------------------------------------------------
    r1 = load_json(OUT / "r1_masked_nli.json")["support_interaction_masked_nli"]
    m_und, sl_und = R1.masked_interaction(lambda q, row: qu_m[q], args.B, R1.SEED_MC)
    m_dat, sl_dat = R1.masked_interaction(lambda q, row: qd_m[q], args.B, R1.SEED_MC)
    rep["masked_undated_equals_r1_new_masked600"] = repro_record(
        {m: v["new_masked_contexts_premise600"] for m, v in r1["per_model"].items()},
        {("haiku" if m == "" else m): {k: v[k] for k in keep if k in v} for m, v in m_und.items()},
        "r1_masked_nli.json per_model[*].new_masked_contexts_premise600")
    masked = {("haiku" if m == "" else m): {
        "undated_600": {k: m_und[m][k] for k in keep if k in m_und[m]},
        "dated_600": {k: m_dat[m][k] for k in keep if k in m_dat[m]},
        "change_dated_minus_undated": round(m_dat[m]["interaction"] - m_und[m]["interaction"], 3),
        "ci_excludes_zero": {"undated": R1.excludes_zero(m_und[m]), "dated": R1.excludes_zero(m_dat[m])}}
        for m in m_und}

    # ---- routing GOLD_NLI --------------------------------------------------------
    rb_d = R1.routing_gold_nli(qd_u, args.B, R1.SEED_RB)
    rb_u_json = load_json(OUT / "r1_masked_nli.json")["routing_gold_nli"]["per_model"]

    # ---- human agreement (from r2_human_alignment.json) -------------------------
    ha = load_json(OUT / "r2_human_alignment.json")

    def pick(blk):
        return {k: (blk[k]["estimate"], blk[k]["ci95"]) for k in ("precision", "recall", "cohens_kappa", "agreement")}

    human = {"source": "results/analysis/r2_human_alignment.json (legacy 700-char human labels)",
             "passage_level": {}, "question_level": {}}
    for pack in ("task_b", "masked_pack"):
        for lab in ("nli600", "nli600_dated"):
            for sub in ("i_all_rows", "ii_excluding_gold_only_in_600_700"):
                for hs in ("human_YES_only", "human_YES_or_PARTIAL"):
                    human["passage_level"].setdefault(pack, {}).setdefault(lab, {}).setdefault(sub, {})[hs] = \
                        pick(ha["a_passage_level"][pack]["by_label"][lab][sub][hs])
    qsets = {"task_b": ("unweighted_sample_questions", "design_weighted_items_population_estimate"),
             "masked_pack": ("unweighted_sample_all_210", "design_weighted_200_population_estimate")}
    for pack, subs in qsets.items():
        for lab in ("nli600_shown3", "nli600_dated_shown3", "nli600_5passages", "nli600_dated_5passages"):
            for sub in subs:
                human["question_level"].setdefault(pack, {}).setdefault(lab, {})[sub] = \
                    pick(ha["b_question_level"][pack]["by_label"][lab]["human_YES_only"][sub])

    prob_shift = {}
    for name, keys in (("unmasked", u_keys), ("masked", m_keys)):
        d_ent = [dated[k]["probs"]["entailment"] - undated[k]["probs"]["entailment"] for k in keys]
        d_ent.sort()
        prob_shift[name] = {"mean_entailment_prob_change": round(sum(d_ent) / len(d_ent), 4),
                            "median": round(d_ent[len(d_ent) // 2], 4),
                            "n_entail_undated": sum(1 for k in keys if undated[k]["label"] == "entailment"),
                            "n_entail_dated": sum(1 for k in keys if dated[k]["label"] == "entailment")}

    out = {
        "step": "R2(c) -- dated-hypothesis NLI sensitivity",
        "status": ("SENSITIVITY ONLY. Date resolution is not validated: no human judged the dated "
                   "proposition, and the NLI model is not known to scope entailment by date."),
        "meta": meta([MANIFEST, DATED_OUT, SRV / "dated600.jsonl", SRV / "undated600.jsonl", E0_NLI600]),
        "hypotheses": {"undated": 'The answer to "<question>" is <gold>.',
                       "dated": 'As of <Month D, YYYY>, the answer to "<question>" is <gold>.',
                       "date_source": "questions.jsonl 'date' (the mask's question date)",
                       "example": f"As of {R1.hyp_dated(qs['20260515_12']).split('As of ')[1]}"},
        "labeller": {"model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli @b3546ea",
                     "premise": "text[:600]", "max_length": 384, "batch": 1, "device": "CUDA fp32 (H100 MIG)",
                     "token_ids": manifest.get("tokenizer_local")},
        "label_file": str(DATED_OUT.name),
        "reproduction": rep,
        "context_label_transitions_undated_to_dated": {"unmasked": ctx_trans(u_keys), "masked": ctx_trans(m_keys)},
        "entailment_probability_shift": prob_shift,
        "question_level_transitions": {"unmasked": q_trans(qu_u, qd_u), "masked": q_trans(qu_m, qd_m),
                                       "explicit_date_rule": "question text contains a month name or a 19xx/20xx year",
                                       "n_questions_with_explicit_date": sum(has_date.values())},
        "unmasked_nli_interaction": {
            "script": "pipeline/support_interaction.py (label 'nli'), label file swapped in memory",
            "seed": 20260802, "B": args.B,
            "weighting": "QUESTION-POOLED arm means; quiz-cluster bootstrap CI",
            "per_model": unmasked,
            "unsupported_cost_vs_scale": {"undated_600": o["unsupported_cost_vs_scale"].get("nli"),
                                          "dated_600": n["unsupported_cost_vs_scale"].get("nli")},
            "summary": {"undated_600": o["summary"]["nli"], "dated_600": n["summary"]["nli"]},
            "runtimes_sec": rts},
        "masked_nli_interaction": {
            "script": "masked_counterfactual.py support-interaction block (via r1_masked_nli.py), Static only",
            "seed": R1.SEED_MC, "B": args.B,
            "weighting": "QUESTION-POOLED arm means; quiz-cluster bootstrap CI",
            "per_model": masked,
            "unsupported_cost_vs_scale_masked": {"undated_600": sl_und, "dated_600": sl_dat}},
        "routing_gold_nli": {
            "weighting": "QUIZ-WEIGHTED (routing_baselines.py estimators)", "seed": R1.SEED_RB,
            "per_model": {m: {"reference": rb_d[m]["reference"],
                              "undated_600_vs_reference": rb_u_json[m]["new_premise600"]["vs_reference"],
                              "dated_600_vs_reference": rb_d[m]["gold_nli_vs_reference"],
                              "retrieve_rate_undated": rb_u_json[m]["new_premise600"]["retrieve_rate"],
                              "retrieve_rate_dated": rb_d[m]["gold_nli_retrieve_rate"]} for m in rb_d}},
        "human_agreement_undated_vs_dated": human,
        "runtime_total_sec": round(time.time() - t_start, 1),
    }
    dump(out, OUT / "r2_dated_nli.json")
    print("repro:", {k: v.get("reproduced_exactly") for k, v in rep.items() if isinstance(v, dict) and "compared" in v})
    print(f"\n{'model':15s}{'reg':>7s}{'undated':>9s}{'dated':>9s}  dated CI  (unmasked)")
    for m, r in unmasked.items():
        for reg, v in r.items():
            print(f"{m:15s}{reg:>7s}{v['undated_600']['interaction']:9.2f}{v['dated_600']['interaction']:9.2f}  "
                  f"{v['dated_600']['ci95']} n_sup {v['undated_600']['n_supported']}->{v['dated_600']['n_supported']}")
    print(f"\n{'model':15s}{'undated':>9s}{'dated':>9s}  dated CI  (masked, Static)")
    for m, v in masked.items():
        print(f"{m:15s}{v['undated_600']['interaction']:9.2f}{v['dated_600']['interaction']:9.2f}  "
              f"{v['dated_600']['ci95']} n_sup {v['undated_600']['n_supported']}->{v['dated_600']['n_supported']}")
    print("slopes unmasked:", o["unsupported_cost_vs_scale"]["nli"]["slope_per_decade"],
          n["unsupported_cost_vs_scale"]["nli"]["slope_per_decade"], n["unsupported_cost_vs_scale"]["nli"]["p_exact"])
    print("slopes masked:", sl_und["slope_per_decade"], sl_dat["slope_per_decade"], sl_dat["p_exact"])
    print("wrote", OUT / "r2_dated_nli.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
