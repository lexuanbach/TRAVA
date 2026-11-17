#!/usr/bin/env python
"""Build the versioned numerical ledger and check it byte for byte with --check."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCHEMA = ["claim_id", "value", "ci95_or_note", "table_or_figure", "provenance", "status"]

# Inferential fields that were superseded by cluster-robust two-sided testing.
# Any row carrying these must be versioned, not silently retained.
SUPERSEDED_KEYS = ("one_sided_p", "estimated_cutoff")


def rows_from_json() -> tuple[list[list[str]], list[str]]:
    rows: list[list[str]] = []
    notes: list[str] = []

    def add(claim_id, value, ci, table, provenance, status="real"):
        rows.append([claim_id, f"{value}" if not isinstance(value, str) else value,
                     ci, table, provenance, status])

    # --- cross-model closed-book gaps, now with joint-bootstrap intervals -------
    js = RESULTS / "joint_slope.json"
    if js.exists():
        d = json.loads(js.read_text(encoding="utf-8"))
        s = d["slope_f1_per_decade"]
        pval = s.get("two_sided_p_plus_one", s.get("two_sided_p"))
        floor = s.get("p_resolution_floor")
        pnote = (f"two-sided p<={pval} (resolution floor {floor})" if floor
                 else f"two-sided p={pval}")
        add("capability_slope_f1_per_decade", s["observed"],
            f"[{s['ci95'][0]}, {s['ci95'][1]}] joint quiz bootstrap, {pnote}",
            "tab:crossmodel", "results/joint_slope.json:slope_f1_per_decade")
        for m in d["models"]:
            ci = m["gap_ci95"]
            add(f"{m['key']}_boundary_gap_f1", m["observed_gap_f1"],
                f"[{ci[0]}, {ci[1]}] joint quiz bootstrap" if ci else "n/a",
                "tab:crossmodel", f"results/joint_slope.json:models[{m['key']}]")
        ys = d["year_pair_sensitivity"]
        add("capability_slope_year_pair_range",
            f"[{ys['slope_range'][0]}, {ys['slope_range'][1]}]",
            f"median {ys['slope_median']}, positive in {ys['sign_agreement']} of {ys['n_pairs']} contrasts",
            "tab:crossmodel", "results/joint_slope.json:year_pair_sensitivity")
    else:
        notes.append("results/joint_slope.json missing; capability-slope rows omitted")

    # --- verified post-training slice (the confirmatory contrast) --------------
    vs = RESULTS / "verified_slice.json"
    if vs.exists():
        d = json.loads(vs.read_text(encoding="utf-8"))
        for key, m in d["models"].items():
            add(f"{key}_verified_gap_2023_vs_2026", m["gap_2023_vs_2026"],
                f"[{m['ci95'][0]}, {m['ci95'][1]}] quiz bootstrap, "
                f"p={m['two_sided_p_plus_one']}, excludes_zero={m['excludes_zero']}",
                "tab:crossmodel", f"results/verified_slice.json:models.{key}")
    else:
        notes.append("results/verified_slice.json missing; verified-slice rows omitted")

    # --- stale-failure audit ---------------------------------------------------
    sa = RESULTS / "stale_audit_report.json"
    if sa.exists():
        d = json.loads(sa.read_text(encoding="utf-8"))
        pr = d["precision_of_automatic_label"]
        add("stale_label_precision", pr["point"],
            f"Wilson95 [{pr['wilson95'][0]}, {pr['wilson95'][1]}], {pr['k']}/{pr['n']} "
            "adjudicated by one unblinded author",
            "tab:staleaudit", "results/stale_audit_report.json:precision_of_automatic_label")
        for key, m in d["corrected_rates"].items():
            add(f"{key}_stale_faithful_pct_corrected", m["corrected_superseded_pct"],
                f"automatic rule said {m['automatic_grounded_outdated_pct']}%; "
                f"CI {m['corrected_ci95_pct']}; precision from {m['precision_source']}",
                "tab:staletax", f"results/stale_audit_report.json:corrected_rates.{key}")
        for key, v in d.get("per_model_precision", {}).items():
            add(f"{key}_stale_label_precision", v["point"],
                f"{v['k']}/{v['n']} adjudicated, Wilson95 {v['wilson95']}",
                "tab:staleaudit",
                f"results/stale_audit_report.json:per_model_precision.{key}")

    else:
        notes.append("results/stale_audit_report.json missing; audit rows omitted")

    # --- stale/LOO replication across the ladder --------------------------------
    sr = RESULTS / "stale_replication.json"
    if sr.exists():
        d = json.loads(sr.read_text(encoding="utf-8"))
        for key, m in d["models"].items():
            for label, c in m["fresh_contrasts"].items():
                add(f"{key}_fresh_{label}", c["diff_f1"],
                    f"[{c['ci95'][0]}, {c['ci95'][1]}] quiz block bootstrap, "
                    f"p={c['two_sided_p_plus_one']}, excludes_zero={c['excludes_zero']}",
                    "tab:heldout", f"results/stale_replication.json:models.{key}.{label}")
    else:
        notes.append("results/stale_replication.json missing; replication rows omitted")

    # --- adaptive-retrieval policies -------------------------------------------
    ap = RESULTS / "adaptive_policy.json"
    if ap.exists():
        d = json.loads(ap.read_text(encoding="utf-8"))
        for key, m in d["models"].items():
            for policy in ("never", "always", "temporal", "abstain", "oracle_perq"):
                add(f"{key}_policy_{policy}_f1", m[policy]["f1_all"],
                    f"retrieval rate {m[policy]['retrieval_rate']}",
                    "tab:adaptive", f"results/adaptive_policy.json:models.{key}.{policy}")
    else:
        notes.append("results/adaptive_policy.json missing; adaptive-policy rows omitted")

    # --- main closed-book / RAG table ------------------------------------------
    mr = RESULTS / "main_results.json"
    if mr.exists():
        d = json.loads(mr.read_text(encoding="utf-8"))
        for cond, rec in (d.items() if isinstance(d, dict) else []):
            if not isinstance(rec, dict):
                continue
            for split in ("static", "fresh"):
                node = rec.get(split)
                if isinstance(node, dict) and "f1" in node:
                    f1 = node["f1"]
                    val = f1.get("mean") if isinstance(f1, dict) else f1
                    ci = (f"[{f1['ci95'][0]}, {f1['ci95'][1]}]"
                          if isinstance(f1, dict) and "ci95" in f1 else "")
                    add(f"{cond}_{split}_f1", val, ci, "tab:main",
                        f"results/main_results.json:{cond}.{split}.f1")

    # --- cluster-robust inference (the values that supersede one-sided p) -------
    cr = RESULTS / "cluster_robust.json"
    if cr.exists():
        d = json.loads(cr.read_text(encoding="utf-8"))
        def walk(node, path):
            if isinstance(node, dict):
                if "p" in node and isinstance(node.get("p"), (int, float)):
                    add(f"{path}_cluster_robust_p", node["p"],
                        "two-sided, quiz-clustered", "tab:did",
                        f"results/cluster_robust.json:{path}")
                for k, v in node.items():
                    walk(v, f"{path}.{k}" if path else k)
        walk(d, "")

    # --- model-level inference over the seven rungs ---------------
    mli = RESULTS / "model_level_inference.json"
    if mli.exists():
        d = json.loads(mli.read_text(encoding="utf-8"))
        tb = d["target_b_over_model_rungs"]
        for key, label in (("f1_all_7", "f1"), ("f1_qwen_only_6", "qwen_only"),
                           ("em_all", "em")):
            b = tb.get(key)
            if not b or "slope_f1_per_decade" not in b:
                continue
            pm = b["permutation"]
            add(f"model_level_slope_{label}", b["slope_f1_per_decade"],
                f"OLS t CI {b['ols_ci95_t']} on n={b['n_models']}, R2={b['r2']}, "
                f"exact permutation p={pm['p_exact']} "
                f"({pm['n_at_least_as_extreme']}/{pm['n_permutations']})",
                "sec:results",
                f"results/model_level_inference.json:target_b_over_model_rungs.{key}")
        loo = tb["f1_all_7"]["loo_range"]
        add("model_level_slope_loo_range", f"[{loo[0]}, {loo[1]}]",
            f"leave-one-model-out; sign stable: {tb['f1_all_7']['loo_sign_stable']}",
            "sec:results",
            "results/model_level_inference.json:target_b_over_model_rungs.f1_all_7.loo_range")
    else:
        notes.append("results/model_level_inference.json missing "
                     "(run pipeline/model_level_inference.py)")

    # --- choice-grounded metric audit -----------------------------
    cg = RESULTS / "choice_grounded_audit.json"
    if cg.exists():
        d = json.loads(cg.read_text(encoding="utf-8"))
        md = d["metric_disagreement"]
        add("f1_credits_wrong_choice_rate", md["f1_credits_a_wrong_choice_rate"],
            f"{md['f1_credits_a_wrong_choice']} of {md['n_scored']} scored answers "
            "score >=0.5 token-F1 while naming a distractor",
            "sec:results", "results/choice_grounded_audit.json:metric_disagreement")
        sc = d.get("scale_slope_choice_grounded")
        if sc:
            add("scale_slope_choice_grounded", sc["slope_pts_per_decade"],
                f"exact permutation p={sc['exact_permutation_p']} on n={sc['n_models']}; "
                "the association does NOT clear significance under this metric",
                "sec:results",
                "results/choice_grounded_audit.json:scale_slope_choice_grounded")
    else:
        notes.append("results/choice_grounded_audit.json missing "
                     "(run pipeline/choice_grounded_audit.py)")

    # --- retrieval gains by regime, quiz-clustered ---------------
    rc = RESULTS / "retrieval_cluster_ci.json"
    if rc.exists():
        d = json.loads(rc.read_text(encoding="utf-8"))
        for m, rec in d["models"].items():
            for reg in ("static", "fresh"):
                r = rec.get(reg)
                if not r:
                    continue
                add(f"retrieval_gain_{m}_{reg}", r["gain_f1"],
                    f"[{r['ci95'][0]}, {r['ci95'][1]}] quiz bootstrap over "
                    f"{r['n_quizzes']} quizzes, p={r['p_plus_one']}",
                    "tab:retrieval-regime",
                    f"results/retrieval_cluster_ci.json:models.{m}.{reg}")
        add("retrieval_significant_harm_contrasts",
            len(d["summary"]["significant_harm"]),
            ", ".join(d["summary"]["significant_harm"]) or "none",
            "tab:retrieval-regime", "results/retrieval_cluster_ci.json:summary")
    else:
        notes.append("results/retrieval_cluster_ci.json missing "
                     "(run pipeline/retrieval_cluster_ci.py)")

    # --- stale-label trend test ----------------------------------
    st = RESULTS / "stale_trend_test.json"
    if st.exists():
        d = json.loads(st.read_text(encoding="utf-8"))
        pp = d["pooled_precision"]
        add("stale_label_precision_pooled", pp["point"],
            f"Wilson {pp['wilson95']}, k={pp['k']}/n={pp['n']} -- SUPPORTED",
            "sec:analysis", "results/stale_trend_test.json:pooled_precision")
        ca = d["cochran_armitage_trend"]
        add("stale_precision_trend_p", ca["p_permutation"],
            f"exact Cochran-Armitage over {ca['n_shuffles']} shuffles; "
            f"capability-grading NOT supported",
            "sec:analysis", "results/stale_trend_test.json:cochran_armitage_trend")
        fe = d["fisher_extreme_pair"]
        add("stale_precision_fisher_p", fe["p_two_sided"],
            f"{fe['comparison']}, two-sided Fisher exact",
            "sec:analysis", "results/stale_trend_test.json:fisher_extreme_pair")
        pw = d["power"]
        add("stale_audit_n_for_80pct_power", pw["n_per_model_for_80pct_power"],
            f"per model; actual {pw['n_per_model_actual']}",
            "sec:analysis", "results/stale_trend_test.json:power")
    else:
        notes.append("results/stale_trend_test.json missing "
                     "(run pipeline/stale_trend_test.py)")

    # --- second model family -------------------------------------
    sf = RESULTS / "second_family.json"
    if sf.exists():
        d = json.loads(sf.read_text(encoding="utf-8"))
        nm = d["new_model"]
        sh = d["shared_boundary_2023_vs_2026"]
        for m, r in sh.items():
            add(f"secondfamily_shared_gap_{m}", r["diff"],
                f"[{r['ci95'][0]}, {r['ci95'][1]}] quiz bootstrap, "
                f"p={r['p_plus_one']}, {r['family']} {r['params_B']}B",
                "sec:results", f"results/second_family.json:"
                               f"shared_boundary_2023_vs_2026.{m}")
        ow = d["llama_boundary_2022_vs_2024"]
        for m in (nm, d["control"]):
            if m in ow:
                r = ow[m]
                add(f"secondfamily_owncutoff_gap_{m}", r["diff"],
                    f"[{r['ci95'][0]}, {r['ci95'][1]}] quiz bootstrap, "
                    f"p={r['p_plus_one']}; 2022 vs 2024 at Llama-3's disclosed "
                    f"March-2023 cutoff",
                    "sec:results",
                    f"results/second_family.json:llama_boundary_2022_vs_2024.{m}")
        raw = d.get("llama_2022_vs_post2024_raw")
        if raw:
            add("secondfamily_llama_pooled_post_gap", raw["diff"],
                f"[{raw['ci95'][0]}, {raw['ci95'][1]}], p={raw['p_plus_one']}; "
                f"2022 vs 2024-2026",
                "sec:results", "results/second_family.json:llama_2022_vs_post2024_raw")
        for tag, key in (("", "did_vs_controls"),
                         ("_pooled", "did_vs_controls_pooled_post")):
            for c, r in d.get(key, {}).items():
                add(f"secondfamily_did{tag}_vs_{c}", r["did"],
                    f"[{r['ci95'][0]}, {r['ci95'][1]}], p={r['p_plus_one']}; "
                    f"isolates the cutoff from year-difficulty drift",
                    "sec:results", f"results/second_family.json:{key}.{c}")
    else:
        notes.append("results/second_family.json missing "
                     "(run pipeline/second_family.py)")

    # --- second temporal source: FreshQA --------------------------
    ss = RESULTS / "second_source.json"
    if ss.exists():
        d = json.loads(ss.read_text(encoding="utf-8"))
        for m, r in d.get("models", {}).items():
            add(f"freshqa_gap_{m}", r["diff"],
                f"[{r['ci95'][0]}, {r['ci95'][1]}], p={r['p_plus_one']}; "
                f"static {r['static_mean']} vs fresh {r['fresh_mean']}, "
                f"n={r['n']}",
                "tab:second-source", f"results/second_source.json:models.{m}")
        sl = d.get("scale_slope")
        if sl:
            pm = sl.get("exact_permutation") or {}
            add("freshqa_scale_slope", sl["slope_f1_per_decade"],
                f"n={sl['n_models']} models"
                + (f", exact permutation p={pm['p_exact']}" if pm else ""),
                "tab:second-source", "results/second_source.json:scale_slope")
    else:
        notes.append("results/second_source.json missing "
                     "(run pipeline/second_source.py)")

    # --- held-out future weeks -----------------------------------
    hf = RESULTS / "heldout_future.json"
    if hf.exists():
        d = json.loads(hf.read_text(encoding="utf-8"))
        add("heldout_future_n_questions", d["n_questions"],
            f"weeks {d['weeks']}, all post-snapshot and all Fresh-side",
            "sec:results", "results/heldout_future.json:n_questions")
        for m, r in d.get("models", {}).items():
            h = r["heldout_f1"]
            add(f"heldout_future_f1_{m}", h["mean"],
                f"[{h['ci95'][0]}, {h['ci95'][1]}] item bootstrap, n={h['n_items']}"
                + (f"; snapshot Fresh {r['snapshot_fresh_f1']}"
                   if r.get("snapshot_fresh_f1") is not None else ""),
                "sec:results", f"results/heldout_future.json:models.{m}")
        sl = d.get("scale_slope_on_heldout_fresh")
        if sl:
            add("heldout_future_fresh_slope", sl["slope_f1_per_decade"],
                f"slope of FRESH accuracy on scale, n={sl['n_models']}",
                "sec:results",
                "results/heldout_future.json:scale_slope_on_heldout_fresh")
    else:
        notes.append("results/heldout_future.json missing "
                     "(run pipeline/heldout_future.py)")

    # --- support-conditioned retrieval mechanisms ------------
    sc = RESULTS / "support_conditioned.json"
    if sc.exists():
        d = json.loads(sc.read_text(encoding="utf-8"))
        for m, rec in d.get("models", {}).items():
            for reg in ("static", "fresh"):
                r = rec.get(reg)
                if not r:
                    continue
                add(f"support_rate_{m}_{reg}", r["support_rate"],
                    f"share of questions whose retrieved context contains the "
                    f"gold answer; n={r['n_questions']}",
                    "sec:mechanism", f"results/support_conditioned.json:models.{m}.{reg}")
                add(f"override_rate_{m}_{reg}", r["override_rate"],
                    "correct closed-book, wrong with retrieval",
                    "sec:mechanism", f"results/support_conditioned.json:models.{m}.{reg}")
                g = r.get("paired_gain_given_support")
                if g:
                    add(f"support_conditioned_gain_{m}_{reg}", g["mean"],
                        f"[{g['ci95'][0]}, {g['ci95'][1]}] quiz bootstrap, "
                        f"p={g['p_plus_one']}; retrieval gain restricted to "
                        f"answer-bearing contexts",
                        "sec:mechanism",
                        f"results/support_conditioned.json:models.{m}.{reg}")
        mech = d.get("mechanism", {})
        for k in ("mean_override_rate_static_strong", "mean_override_rate_fresh_strong"):
            if mech.get(k) is not None:
                add(k, mech[k], "mean over the three strongest generators",
                    "sec:mechanism", f"results/support_conditioned.json:mechanism.{k}")
    else:
        notes.append("results/support_conditioned.json missing "
                     "(run pipeline/support_conditioned.py)")

    # --- cutoff-relative reparameterisation ---------------
    cr = RESULTS / "cutoff_relative.json"
    if cr.exists():
        d = json.loads(cr.read_text(encoding="utf-8"))
        for m, rec in d.get("per_model", {}).items():
            for tag, key in (("own", "own_boundary"), ("shared", "shared_2025_boundary")):
                r = rec.get(key)
                if r:
                    add(f"cutoff_{tag}_gap_{m}", r["gap"],
                        f"[{r['ci95'][0]}, {r['ci95'][1]}], p={r['p_plus_one']}; "
                        f"cutoff {rec['cutoff']} ({rec['cutoff_source']})",
                        "sec:mechanism", f"results/cutoff_relative.json:per_model.{m}.{key}")
            band = rec.get("sensitivity_over_cutoff_interval") or {}
            if band:
                add(f"cutoff_sensitivity_{m}",
                    f"[{min(band.values())}, {max(band.values())}]",
                    "gap under the plausible cutoff interval, not a point date",
                    "sec:mechanism",
                    f"results/cutoff_relative.json:per_model.{m}.sensitivity_over_cutoff_interval")
        mis = d.get("misattribution_slope")
        if mis:
            add("shared_boundary_misattribution_slope",
                mis["slope_gap_difference_per_month_of_cutoff_offset"],
                f"n={mis['n_models']} disclosed-cutoff models; descriptive",
                "sec:mechanism", "results/cutoff_relative.json:misattribution_slope")
    else:
        notes.append("results/cutoff_relative.json missing "
                     "(run pipeline/cutoff_relative.py)")

    # --- semantic (entailment) support relabelling --------------
    ns = RESULTS / "nli_support.json"
    if ns.exists():
        d = json.loads(ns.read_text(encoding="utf-8"))
        ca = d["context_level_agreement"]
        add("nli_vs_string_support_agreement", ca["raw_agreement"],
            f"{d['n_contexts_labelled']} retrieved contexts relabelled with "
            f"{d['nli_model']}", "sec:nli_support",
            "results/nli_support.json:context_level_agreement")
        add("string_supported_not_entailed", ca["string_supported_but_not_entailed"],
            "share of string-supported contexts the NLI model does not judge "
            "entailing", "sec:nli_support",
            "results/nli_support.json:context_level_agreement")
        add("entailed_no_string_match", ca["entailed_but_no_string"],
            "paraphrase/alias support that string matching misses",
            "sec:nli_support", "results/nli_support.json:context_level_agreement")
        add("contradiction_inside_string_support",
            d["contradiction_inside_string_support"],
            "string-supported contexts that CONTRADICT the answer",
            "sec:nli_support", "results/nli_support.json")
        for m, rec in d.get("gain_given_support_by_label", {}).items():
            for k, v in rec.items():
                add(f"gain_given_support_{k}_{m}", v["mean"],
                    f"[{v['ci95'][0]}, {v['ci95'][1]}], p={v['p_plus_one']}; "
                    f"support label = {k.split('_')[0]} (oracle)",
                    "sec:nli_support",
                    f"results/nli_support.json:gain_given_support_by_label.{m}.{k}")
        add("mechanism_survives_semantic_relabel",
            int(d["verdict"]["mechanism_survives_semantic_relabel"]),
            "1 if Static gain given support stays non-negative for the strong "
            "models under the entailment label", "sec:nli_support",
            "results/nli_support.json:verdict")
    else:
        notes.append("results/nli_support.json missing "
                     "(run pipeline/nli_support.py)")

    # --- answer-independent routing baselines ---------------
    rb = RESULTS / "routing_baselines.json"
    if rb.exists():
        d = json.loads(rb.read_text(encoding="utf-8"))
        for m, rec in d.get("models", {}).items():
            for pol, v in rec.get("policies", {}).items():
                gold = pol.startswith("gold")
                add(f"routing_{pol}_{m}", v["mean"],
                    f"[{v['ci95'][0]}, {v['ci95'][1]}] quiz bootstrap, marginal "
                    f"mean over {v['n_questions']} questions"
                    + ("; GOLD-CONDITIONED (uses the answer): a diagnostic, "
                       "not a policy and not a ceiling" if gold else "")
                    + ("; per-question upper bound over all deterministic gates"
                       if pol == "perq_best" else ""),
                    "sec:routing",
                    f"results/routing_baselines.json:models.{m}.policies.{pol}")
            # the estimand the manuscript may quote: PAIRED difference
            for pol, v in rec.get("vs_reference", {}).items():
                add(f"routing_delta_{pol}_{m}", v["delta"],
                    f"[{v['ci95'][0]}, {v['ci95'][1]}] paired quiz-clustered vs "
                    f"'{rec['reference']}'"
                    + (f", p_holm={v['p_holm']}" if "p_holm" in v else
                       f", p={v['p_plus_one']} (uncorrected)"),
                    "sec:routing",
                    f"results/routing_baselines.json:models.{m}.vs_reference.{pol}")
            cf = rec.get("cross_fit", {}).get("overlap")
            if cf:
                add(f"routing_overlap_threshold_range_{m}",
                    f"{cf['threshold_min']}-{cf['threshold_max']}",
                    f"selected overlap thresholds across repeated two-fold "
                    f"cross-fits; retrieve rate {cf['retrieve_rate_mean']}",
                    "sec:routing",
                    f"results/routing_baselines.json:models.{m}.cross_fit.overlap")
        s_ = d.get("overlap_summary", {})
        if s_:
            add("routing_overlap_n_significant", s_["n_significant_after_holm"],
                f"of {s_['n_generators']} generators whose paired overlap gain "
                f"excludes zero after Holm ({', '.join(s_['significant_models'])}); "
                f"point estimate positive for {s_['n_positive_point_estimate']}. "
                f"NOT every generator: worst is {s_['worst_model']} at "
                f"{s_['worst_delta']}",
                "sec:routing", "results/routing_baselines.json:overlap_summary")
        cr = d.get("ceiling_recovery", {})
        if cr.get("median_recovered") is not None:
            add("routing_median_ceiling_recovered", cr["median_recovered"],
                "median fraction of the gap from the better FIXED policy to the "
                "per-question best that the best answer-independent gate "
                "recovers; bounded above by 1 and defined for every generator",
                "sec:routing", "results/routing_baselines.json:ceiling_recovery")
    else:
        notes.append("results/routing_baselines.json missing "
                     "(run pipeline/routing_baselines.py)")

    # --- direct support x retrieval interaction -------------
    si = RESULTS / "support_interaction.json"
    if si.exists():
        d = json.loads(si.read_text(encoding="utf-8"))
        for label, per_model in d.get("labels", {}).items():
            for m, rec in per_model.items():
                for regime, r in rec.items():
                    add(f"support_interaction_{label}_{regime}_{m}",
                        r["interaction"],
                        f"[{r['ci95'][0]}, {r['ci95'][1]}], p={r['p_plus_one']}; "
                        f"gain|supported {r['gain_supported']} vs "
                        f"gain|unsupported {r['gain_unsupported']}; support is "
                        f"NOT randomised and the label uses the gold answer",
                        "sec:mechanism",
                        f"results/support_interaction.json:labels.{label}.{m}.{regime}")
        for label, t in d.get("unsupported_cost_vs_scale", {}).items():
            add(f"unsupported_cost_slope_{label}", t["slope_per_decade"],
                f"Static unsupported-subset retrieval gain on log10(params), "
                f"exact permutation p={t['p_exact']} over {t['n_permutations']} "
                f"assignments, n={t['n_rungs']} rungs; roster-confounded",
                "sec:mechanism",
                f"results/support_interaction.json:unsupported_cost_vs_scale.{label}")
    else:
        notes.append("results/support_interaction.json missing "
                     "(run pipeline/support_interaction.py)")

    # --- human validation ---
    hv = RESULTS / "human_validation.json"
    if hv.exists():
        d = json.loads(hv.read_text(encoding="utf-8"))
        prov = d.get("label_provenance", "unknown")
        tag = ("PROVISIONAL, single unadjudicated annotator; no kappa, the "
               "stop rule never ran" if d.get("provisional") else "adjudicated")
        for scope in ("population_weighted", "hard_subset_only"):
            b = d.get(scope) or {}
            for k in ("f1_agrees_with_human", "em_agrees_with_human",
                      "human_correct_rate", "f1_misses_a_human_right",
                      "f1_credits_a_non_correct",
                      "f1_credits_a_strictly_incorrect"):
                if b.get(k) is not None:
                    add(f"human_{scope}_{k}", b[k],
                        f"{tag}; covers {b.get('population_covered')} of the "
                        f"item population, n={b.get('n_items')}",
                        "sec:limitations", f"results/human_validation.json:{scope}.{k}")
        for m, v in (d.get("unweighted_kappa_vs_human") or {}).items():
            add(f"human_kappa_{m}", v,
                f"Cohen kappa, automatic vs human, unweighted; {tag}",
                "sec:limitations",
                f"results/human_validation.json:unweighted_kappa_vs_human.{m}")
    else:
        notes.append("results/human_validation.json missing "
                     "(run annotation/score_annotations.py --stage final)")

    aj = RESULTS / "adjudication.json"
    if aj.exists():
        d = json.loads(aj.read_text(encoding="utf-8"))
        for field, r in (d.get("fields") or {}).items():
            if not r.get("n_adjudicated"):
                continue
            add(f"adjudication_departure_{field}", r["departure_rate"],
                f"blind third-party adjudicator vs the majority on "
                f"{r['n_adjudicated']} contested items; random-choice baseline "
                f"{r['departure_rate_if_choosing_at_random']}, so the excess is "
                f"{r['departure_vs_chance']:+.3f}. Switches: {r['switches']}",
                "sec:limitations", f"results/adjudication.json:fields.{field}")
            add(f"adjudication_offballot_{field}", r["off_ballot_rate"],
                "fraction where the adjudicator chose a label none of the three "
                "annotators offered", "sec:limitations",
                f"results/adjudication.json:fields.{field}")

    ar = ROOT / "annotation/pack_2026-08-02/agreement_report.json"
    if ar.exists():
        d = json.loads(ar.read_text(encoding="utf-8"))
        for name, u in (d.get("usability") or {}).items():
            add(f"iaa_{name.replace(':', '_')}", u["cohens_kappa"],
                f"Cohen kappa between the two annotators; {u['ruling']}",
                "sec:limitations",
                f"annotation/pack_2026-08-02/agreement_report.json:usability.{name}")

    sv = RESULTS / "support_label_validation.json"
    if sv.exists():
        d = json.loads(sv.read_text(encoding="utf-8"))
        for auto, defs in (d.get("labels") or {}).items():
            for name, r in defs.items():
                add(f"support_label_{auto}_{name}_kappa", r["cohens_kappa"],
                    f"automatic vs human at the passage level, n={r['n']}; "
                    f"precision {r['precision']}, recall {r['recall']}, "
                    f"agreement {r['agreement']['rate']} {r['agreement']['ci95']}",
                    "sec:mechanism",
                    f"results/support_label_validation.json:labels.{auto}.{name}")

    mc = RESULTS / "masked_counterfactual.json"
    if mc.exists():
        d = json.loads(mc.read_text(encoding="utf-8"))
        for m, r in (d.get("per_model") or {}).items():
            for regime in ("static", "fresh"):
                if regime not in r:
                    continue
                for est in ("delta_mask", "delta_rag_masked", "delta_rag_unmasked"):
                    v = r[regime][est]
                    add(f"masked_{est}_{regime}_{m}", v["delta"],
                        f"[{v['ci95'][0]}, {v['ci95'][1]}], p={v['p_plus_one']}; "
                        f"PAIRED within question over {v['n_questions']} "
                        f"questions in {v['n_quizzes']} quizzes",
                        "sec:temporal",
                        f"results/masked_counterfactual.json:per_model.{m}.{regime}.{est}")
        for label, per_m in (d.get("support_interaction_mask_sensitivity") or {}).items():
            for m, v in per_m.items():
                add(f"support_interaction_maskshift_{label}_{m}", v["shift"],
                    f"Static support x retrieval interaction, masked "
                    f"{v['interaction_masked']} vs unmasked "
                    f"{v['interaction_unmasked']}; unsupported arm "
                    f"{v['unsupported_masked']} vs {v['unsupported_unmasked']}",
                    "sec:mechanism",
                    f"results/masked_counterfactual.json:support_interaction_mask_sensitivity.{label}.{m}")
        for label, t in (d.get("unsupported_cost_vs_scale_masked") or {}).items():
            add(f"unsupported_cost_slope_masked_{label}", t["slope_per_decade"],
                f"exact permutation p={t['p_exact']}, {t['n_rungs']} rungs, "
                f"computed on MASKED contexts",
                "sec:mechanism",
                f"results/masked_counterfactual.json:unsupported_cost_vs_scale_masked.{label}")
    else:
        notes.append("results/masked_counterfactual.json missing "
                     "(run scripts/run_masked_panel.sh then "
                     "pipeline/masked_counterfactual.py)")

    # --- the support interaction under HUMAN passage labels ------------------
    hsi = RESULTS / "human_support_interaction.json"
    if hsi.exists():
        d = json.loads(hsi.read_text(encoding="utf-8"))
        for m, rec in (d.get("per_model") or {}).items():
            for label, per_regime in rec.items():
                for regime, v in per_regime.items():
                    add(f"human_support_interaction_{label}_{regime}_{m}",
                        v["interaction"],
                        f"[{v['ci95'][0]}, {v['ci95'][1]}], p={v['p_plus_one']}; "
                        f"human passage labels, quiz-clustered over "
                        f"{v['n_quizzes']} quizzes "
                        f"({v['n_supported']} supported / "
                        f"{v['n_unsupported']} unsupported questions)",
                        "sec:evidencecontent",
                        f"results/human_support_interaction.json:"
                        f"per_model.{m}.{label}.{regime}")
        agr = d.get("question_level_label_agreement")
        if agr:
            add("human_vs_auto_question_support_agreement", agr["rate"],
                f"n={agr['n']} labelled questions", "sec:evidencecontent",
                "results/human_support_interaction.json:"
                "question_level_label_agreement.rate")
    else:
        notes.append("results/human_support_interaction.json missing "
                     "(run pipeline/human_support_interaction.py)")

    hm = RESULTS / "human_metric_audit.json"
    if hm.exists():
        d = json.loads(hm.read_text(encoding="utf-8"))
        c = d["does_metric_error_cancel_across_the_boundary"]
        add("metric_error_slice_differential_pts", c["difference_pts"],
            f"{c['ci95_pts']}, p={c['p_plus_one']}; Static {c['static_false_negative_rate']} "
            f"vs Fresh {c['fresh_false_negative_rate']} miss prevalence among all outputs. "
            f"Interval INCLUDES zero, so this is a caution, not a demonstrated "
            f"bias. Adjudicated labels; common stratified bootstrap.",
            "sec:limitations",
            "results/human_metric_audit.json:does_metric_error_cancel_across_the_boundary")
        for group, metrics in d.get("consistent_summaries", {}).items():
            for metric, rec in metrics.items():
                add("adjudicated_" + group + "_" + metric, rec["estimate"],
                    f"95% CI {rec['ci95']}; 5000 common within-stratum draws, seed 20260802",
                    "app:humanstudy", "results/human_metric_audit.json:consistent_summaries." + group + "." + metric)
        fr = d.get("freshness_construct_reliability") or {}
        for k in ("as_collected_5_labels", "anachronistic_vs_rest",
                  "superseded_vs_rest"):
            if fr.get(k) is not None:
                add(f"freshness_kappa_{k}", fr[k],
                    f"inter-annotator kappa for this coding of freshness; "
                    f"usable at 0.6: {fr.get('usable_at_threshold', {}).get(k)}",
                    "sec:limitations",
                    f"results/human_metric_audit.json:freshness_construct_reliability.{k}")
        a = d["abstention_detector"]
        add("abstention_detector_recall", a["recall"],
            f"our is_abstention() rule finds {a['detected_by_our_rule']} of "
            f"{a['human_labelled_abstentions']} human-labelled refusals; "
            f"reported rates are detector-defined, not bounds",
            "sec:limitations", "results/human_metric_audit.json:abstention_detector")
    else:
        notes.append("results/human_metric_audit.json missing "
                     "(run pipeline/human_metric_audit.py)")

    # --- controlled-cutoff family --------
    cc = RESULTS / "controlled_cutoff.json"
    if cc.exists():
        d = json.loads(cc.read_text(encoding="utf-8"))
        for tag, r in d.get("gap_at_own_cutoff", {}).items():
            add(f"controlled_gap_own_cutoff_{tag}", r["gap"],
                f"[{r['ci95'][0]}, {r['ci95'][1]}], p={r['p_plus_one']}; "
                f"{r['params_B']}B at its disclosed cutoff {r['cutoff']} "
                f"({r['n_static']} Static / {r['n_fresh']} Fresh)",
                "sec:controlled",
                f"results/controlled_cutoff.json:gap_at_own_cutoff.{tag}")
        a = d.get("arm_a_cutoff_varies_size_fixed", {})
        for k in ("before_both_cutoffs", "between_the_cutoffs",
                  "after_both_cutoffs"):
            r = a.get(k)
            if r:
                add(f"arm_a_{k}", r["delta"],
                    f"[{r['ci95'][0]}, {r['ci95'][1]}], p={r['p_plus_one']}; "
                    f"PAIRED, {r['n_questions']} questions in "
                    f"{r['n_quizzes']} quizzes; later-cutoff minus "
                    f"earlier-cutoff at {a.get('size_held_at_B')}B",
                    "sec:controlled",
                    f"results/controlled_cutoff.json:arm_a_cutoff_varies_size_fixed.{k}")
        if a.get("did_window_minus_before") is not None:
            add("arm_a_did_window_minus_before", a["did_window_minus_before"],
                f"cutoff effect at FIXED size: the later model's advantage on "
                f"questions in {a.get('window')} minus its advantage on "
                f"questions before both cutoffs",
                "sec:controlled",
                "results/controlled_cutoff.json:arm_a_cutoff_varies_size_fixed")
        b = d.get("arm_b_size_varies_cutoff_fixed", {})
        for tag, r in b.get("per_model", {}).items():
            add(f"arm_b_gap_{tag}", r["gap"],
                f"[{r['ci95'][0]}, {r['ci95'][1]}], p={r['p_plus_one']}; "
                f"{r['params_B']}B at the SHARED {b.get('shared_cutoff')} "
                f"cutoff", "sec:controlled",
                f"results/controlled_cutoff.json:arm_b_size_varies_cutoff_fixed.per_model.{tag}")
        sb = b.get("slope_bootstrap")
        if sb:
            add("arm_b_scale_slope_at_fixed_cutoff", sb["slope_per_decade"],
                f"[{sb['ci95'][0]}, {sb['ci95'][1]}], p={sb['p_plus_one']}; "
                f"quiz-clustered bootstrap over {len(sb['rungs'])} rungs "
                f"sharing the disclosed cutoff {b.get('shared_cutoff')}. This is "
                f"the scale slope with CUTOFF HELD FIXED",
                "sec:controlled",
                "results/controlled_cutoff.json:arm_b_size_varies_cutoff_fixed.slope_bootstrap")
        sp = b.get("slope_permutation")
        if sp:
            add("arm_b_slope_permutation_p", sp["p_exact"],
                f"exact over {sp['n_permutations']} assignments of "
                f"{sp['n_rungs']} rungs; resolution floor "
                f"{sp['p_resolution_floor']} -- "
                + ("UNDERPOWERED BY CONSTRUCTION, the bootstrap is primary"
                   if sp["underpowered_by_construction"] else "usable"),
                "sec:controlled",
                "results/controlled_cutoff.json:arm_b_size_varies_cutoff_fixed.slope_permutation")
        v = b.get("vs_main_ladder")
        if v:
            add("controlled_vs_shared_boundary_ratio", v["ratio"],
                f"controlled slope {v['controlled_slope']} over shared-boundary "
                f"slope {v['main_ladder_slope']}; below 1 means the "
                f"shared-boundary estimate was partly cutoff offset",
                "sec:controlled",
                "results/controlled_cutoff.json:arm_b_size_varies_cutoff_fixed.vs_main_ladder")
    else:
        notes.append("results/controlled_cutoff.json missing "
                     "(run scripts/run_controlled_cutoff.sh then "
                     "pipeline/controlled_cutoff.py)")

    # --- composition-matched temporal contrasts -------------
    ms = RESULTS / "matched_strata.json"
    if ms.exists():
        d = json.loads(ms.read_text(encoding="utf-8"))
        for tag in ("raw", "residualised", "matched"):
            v = d["scale_slope"].get(tag)
            if v is not None:
                add(f"scale_slope_{tag}", v,
                    f"2023-vs-2026 gap on log10(params), composition control="
                    f"{tag}; n_matched_pairs={d.get('n_matched_pairs')}",
                    "sec:naive", f"results/matched_strata.json:scale_slope.{tag}")
        for m, rec in d.get("per_model", {}).items():
            for tag in ("raw", "residualised", "matched"):
                r = rec.get(tag)
                if r:
                    add(f"contrast_{tag}_{m}", r["gap"],
                        f"[{r['ci95'][0]}, {r['ci95'][1]}], p={r['p_plus_one']}"
                        + (f"; PAIRED over {r['n_pairs']} matched pairs, "
                           f"clustered on {r['n_clusters']} 2023 quizzes"
                           if r.get("paired") else "; unpaired, quiz-clustered"),
                        "sec:naive", f"results/matched_strata.json:per_model.{m}.{tag}")
        for tag, r in (d.get("scale_slope_main_ladder") or {}).items():
            if r:
                add(f"scale_slope_{tag}_modellevel_p", r["p_exact"],
                    f"exact permutation over {r['n_permutations']} assignments "
                    f"of the {r['n_rungs']} main-ladder rungs; slope "
                    f"{r['slope']}, leave-one-out range "
                    f"{r['leave_one_out_range']}",
                    "sec:naive",
                    f"results/matched_strata.json:scale_slope_main_ladder.{tag}")
        v = d.get("verdict", {})
        if v:
            add("matched_controls_covariates",
                ", ".join(v.get("controlled_covariates", [])),
                "the ONLY covariates controlled; uncontrolled: "
                + ", ".join(v.get("uncontrolled", [])),
                "sec:naive", "results/matched_strata.json:verdict")
        c = d["composition"]
        add("composition_drift_answer_len",
            f"{c['2023']['mean_answer_len']}->{c['2026']['mean_answer_len']}",
            "mean gold-answer length, 2023 vs 2026",
            "sec:naive", "results/matched_strata.json:composition")
    else:
        notes.append("results/matched_strata.json missing "
                     "(run pipeline/matched_strata.py)")

    sys.path.insert(0, str(ROOT / "pipeline"))
    from corrected_ledger import reconcile
    rows = reconcile(rows, ROOT)
    # Retrospective aligned-estimand extension. Existing claim IDs remain intact.
    ac = RESULTS / "aligned_contrasts.json"
    if ac.exists():
        d = json.loads(ac.read_text())
        for contrast, c in d["contrasts"].items():
            for weight, wd in c["results"].items():
                for endpoint, ed in wd.items():
                    for panel, value in ed["slopes"].items():
                        add("aligned_" + "_".join([contrast,weight,endpoint,panel]), value["estimate"],
                            f"95% CI {value['ci95']}; model-permutation p={value['p_model_permutation']}",
                            "tab:aligned-slopes", f"results/aligned_contrasts.json:contrasts.{contrast}.results.{weight}.{endpoint}.slopes.{panel}")
                    for model, value in ed["per_model"].items():
                        add("aligned_gap_" + "_".join([contrast,weight,endpoint,model]),value["estimate"],
                            f"95% CI {value['ci95']}", "tab:aligned-endpoints",
                            f"results/aligned_contrasts.json:contrasts.{contrast}.results.{weight}.{endpoint}.per_model.{model}")
        for weight, wd in d["half_year_sensitivity"].items():
            for model,value in wd["mean_across_valid_boundaries"].items():
                add("boundary_mean_"+weight+"_"+model,value["estimate"],f"95% CI {value['ci95']}; retrospective",
                    "tab:boundary-average",f"results/aligned_contrasts.json:half_year_sensitivity.{weight}.mean_across_valid_boundaries.{model}")
    rc = RESULTS / "reporting_checks.json"
    if rc.exists():
        d=json.loads(rc.read_text())
        for regime,value in d["haiku_unmasked_nli"]["per_slice"].items():
            add("haiku_unmasked_visible_nli_"+regime,value["interaction"],f"95% CI {value['ci95']}",
                "app:panels",f"results/reporting_checks.json:haiku_unmasked_nli.per_slice.{regime}")
    pc = RESULTS / "paired_label_comparison.json"
    if pc.exists():
        d = json.loads(pc.read_text())
        for pack in ("task_b", "masked_pack"):
            for contrast, v in d[pack]["paired_differences"].items():
                add("paired_kappa_" + pack + "_" + contrast, v["estimate"],
                    f"95% CI {v['ci95']}, Holm p={v['holm_p_three_comparisons_within_pack']}, unit passage weights, quiz bootstrap",
                    "app:paired-kappa", f"results/paired_label_comparison.json:{pack}.paired_differences.{contrast}")
    if rc.exists():
        for regime, v in json.loads(rc.read_text())["stale"]["query_grace_window_exposure"].items():
            for key, value in v.items():
                if isinstance(value, (int, float)):
                    add("stale_late_" + regime + "_" + key, value,
                        "Document date exceeds query date plus three days", "app:evidencequality",
                        f"results/reporting_checks.json:stale.query_grace_window_exposure.{regime}.{key}")
    return rows, notes


def validate(rows: list[list[str]]) -> list[str]:
    errors = []
    for i, r in enumerate(rows, start=2):
        if len(r) != len(SCHEMA):
            errors.append(f"row {i}: {len(r)} fields, expected {len(SCHEMA)}")
        joined = " ".join(str(x) for x in r)
        for key in SUPERSEDED_KEYS:
            if key in joined:
                errors.append(f"row {i}: contains superseded field '{key}' ({r[0]})")
        if len(r) == len(SCHEMA) and r[4]:
            src = r[4].split(":")[0]
            if src.startswith("results/") and not (ROOT / src).exists():
                errors.append(f"row {i}: provenance file missing: {src}")
    ids = [r[0] for r in rows if r]
    from collections import Counter
    dupes = {i for i, n in Counter(ids).items() if n > 1}
    if dupes:
        errors.append(f"duplicate claim_ids: {sorted(dupes)[:5]}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only; do not write")
    ap.add_argument("--output", type=Path, default=RESULTS / "claim_ledger.csv")
    args = ap.parse_args()

    # Report the state of the shipped ledger before replacing it.
    if args.output.exists():
        old = list(csv.reader(args.output.open(encoding="utf-8")))
        if old:
            bad = sum(1 for r in old[1:] if len(r) != len(old[0]))
            stale = sum(1 for r in old[1:] if any(k in " ".join(r) for k in SUPERSEDED_KEYS))
            print(f"existing ledger: {len(old)-1} data rows, {bad} off-schema, {stale} carrying "
                  f"superseded inferential fields")

    rows, notes = rows_from_json()
    for n in notes:
        print(f"  note: {n}")
    errors = validate(rows)

    print(f"rebuilt ledger: {len(rows)} rows, {len(errors)} validation errors")
    for e in errors[:10]:
        print(f"  ERROR {e}")
    if errors:
        return 1

    # Serialise the rebuild once, then either write it or compare it with what
    # is on disk. The previous --check only validated the rebuilt rows against
    # the schema, so a ledger that had drifted from the result JSONs still
    # passed: the shipped 541-row file and a 567-row rebuild both "checked out".
    # A drift guard has to compare, so --check now diffs byte for byte.
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(SCHEMA)
    w.writerows(rows)
    rebuilt = buf.getvalue()

    if args.check:
        if not args.output.exists():
            print(f"DRIFT: {args.output} does not exist; rebuild has {len(rows)} rows")
            return 1
        # newline="" disables universal-newline translation: the file is written
        # with explicit LF endings. Preserve exact bytes during validation.
        with args.output.open(encoding="utf-8", newline="") as fh:
            stored = fh.read()
        if stored != rebuilt:
            n_stored = max(len(stored.splitlines()) - 1, 0)
            print(f"DRIFT: stored ledger ({n_stored} rows) differs from rebuild "
                  f"({len(rows)} rows). Run without --check to refresh.")
            s_rows = stored.splitlines()[1:]
            r_rows = rebuilt.splitlines()[1:]
            s_ids = {r.split(",", 1)[0] for r in s_rows}
            r_ids = {r.split(",", 1)[0] for r in r_rows}
            only_new = sorted(r_ids - s_ids)
            only_old = sorted(s_ids - r_ids)
            if only_new:
                print(f"  {len(only_new)} claim_ids only in rebuild, e.g. {only_new[:5]}")
            if only_old:
                print(f"  {len(only_old)} claim_ids only in stored ledger, e.g. {only_old[:5]}")
            changed = [a.split(",", 1)[0] for a, b in zip(s_rows, r_rows) if a != b]
            if changed:
                print(f"  {len(changed)} rows changed in place, e.g. {changed[:5]}")
            return 1
        print(f"ledger in sync: {len(rows)} rows match {args.output} byte for byte")
        return 0

    with args.output.open("w", encoding="utf-8", newline="") as fh:
        fh.write(rebuilt)
    print(f"wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
