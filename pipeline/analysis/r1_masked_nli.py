#!/usr/bin/env python3
"""R1: masked NLI support interaction with labels on the MASKED contexts shown.

masked_counterfactual.py labels its masked-context NLI arm with per-question
labels computed on the UNMASKED Hybrid contexts (text[:1500]). This script

  1. validates the GPU-server labels (r1_nli_server.py) against the bundle
     manifest (every tuple once, premise/hypothesis hashes), and reports
     device/tokenizer reproduction checks against the stored MPS labels;
  2. writes results/analysis/r1_nli_masked_600.jsonl: one row per masked
     (qid, doc_id) in retrieval_masked.json['hybrid'][:5] (== the doc_ids of
     every generations_*/hybrid_masked.jsonl, verified by the bundle builder),
     premise = text[:600], undated hypothesis, same model/tokenizer/max_length;
  3. reproduces masked_counterfactual.json support_interaction_masked['string']
     and ['nli'] (and both unsupported-cost slopes) EXACTLY with the script's own
     loaders and interaction_ci (seed 20260803, B 5000, Static only,
     question-pooled arm means, quiz-cluster bootstrap);
  4. recomputes the masked NLI interaction under
        stored   unmasked contexts, text[:1500]   (the published, cross-context)
        e0       unmasked contexts, text[:600]    (window fixed, still cross-context)
        new      MASKED contexts,   text[:600]    (corrected; primary)
     and the masked unsupported-cost slope for each;
  5. lists every consumer of NLI labels in the repository, says whether it used
     labels from a different context set, and recomputes the label-dependent
     outputs where a 600-char label on the analysed contexts applies
     (paper macros/table/figure values from masked_counterfactual.json; the
     routing GOLD_NLI diagnostic).

Outputs results/analysis/r1_masked_nli.json (+ r1_raw/).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import re
import statistics
import subprocess
import time
from pathlib import Path

from _e0util import (OUT, PROC, RES, ROOT, dump, load_json, meta, repro_record,
                     window_fn)

import masked_counterfactual as MC          # noqa: E402
import routing_baselines as RB              # noqa: E402

RAW = OUT / "r1_raw"
SRV = RAW / "server_out"
MANIFEST = RAW / "bundle_manifest.json"
MASKED_LABELS = OUT / "r1_nli_masked_600.jsonl"
E0_NLI600 = OUT / "nli_support_labels_600.jsonl"
STORED_NLI = RES / "nli_support_labels.jsonl"
DIAG_TOK = RAW / "diag_server_tokenizer_tf4.57.1"
SEED_MC, SEED_RB = 20260803, 20260802
FULL, VIS = window_fn(None), window_fn(600)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hyp_undated(q):
    return f"The answer to \"{q['question']}\" is {q['gold_answer']}."


MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def hyp_dated(q):
    y, m, d = (int(x) for x in str(q["date"])[:10].split("-"))
    return (f"As of {MONTHS[m - 1]} {d}, {y}, the answer to \"{q['question']}\" "
            f"is {q['gold_answer']}.")


def load_server_labels(job, qs, manifest):
    """(qid, doc) -> server row, after hash validation against the manifest."""
    rows = read_jsonl(SRV / f"{job}.jsonl")
    by = {}
    dup = 0
    for r in rows:
        k = (r["qid"], r["doc_id"])
        dup += k in by
        by[k] = r
    tup = {(t["qid"], t["doc_id"]): t for t in manifest["tuples"]}
    hyp = hyp_undated if job == "undated600" else hyp_dated
    bad_prem = sum(1 for k, r in by.items() if tup.get(k, {}).get("premise_sha256") != r["premise_sha256"])
    bad_hyp = sum(1 for k, r in by.items() if sha(hyp(qs[k[0]])) != r["hypothesis_sha256"])
    check = {"job": job, "n_rows": len(rows), "n_unique": len(by), "n_duplicates": dup,
             "n_manifest_tuples": len(tup), "all_manifest_tuples_labelled": set(by) == set(tup),
             "premise_sha_mismatches": bad_prem, "hypothesis_sha_mismatches": bad_hyp}
    return by, check


def q_entail_any(label_by_pair, docs_by_q):
    out = {}
    for q, docs in docs_by_q.items():
        labs = [label_by_pair.get((q, d)) for d in docs]
        if any(l is None for l in labs):
            out[q] = None
        else:
            out[q] = any(l.startswith("entail") for l in labs)
    return out


def stored_style_q_labels(path):
    """Exactly masked_counterfactual.py's construction (qid -> entail-any)."""
    nli = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            nli.setdefault(r["qid"], False)
            nli[r["qid"]] |= r["nli_label"].startswith("entail")
    return nli


# ---------------------------------------------------------------------------
# masked_counterfactual.py's support-interaction block, verbatim except that the
# support decision is a parameter (sup_fn(q, masked_row) -> bool | None).
# ---------------------------------------------------------------------------
def masked_interaction(sup_fn, B, seed):
    qs = MC.load_questions()
    regenerated = []
    for m in MC.PANEL:
        ma = MC.load(m, "hybrid_masked")
        if not ma or len(ma) < len(qs):
            continue
        regenerated.append(m)
    inter = {}
    for m in regenerated:
        cb, ma = MC.load(m, "closed_book"), MC.load(m, "hybrid_masked")
        common = sorted(set(cb) & set(ma) & set(qs))
        units = collections.defaultdict(lambda: ([], []))
        for q in common:
            if qs[q]["year"] >= MC.BOUNDARY:
                continue
            sup = sup_fn(q, ma[q])
            if sup is None:
                continue
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
    return inter, slope_masked(inter)


def slope_masked(inter):
    """masked_counterfactual.py's unsupported-cost slope block, verbatim."""
    pts = sorted((MC.LADDER_B[m], v["gain_unsupported"]) for m, v in inter.items()
                 if m in MC.LADDER_B)
    if len(pts) < 4:
        return None
    x = [math.log10(a) for a, _ in pts]
    y = [b for _, b in pts]
    mx = statistics.fmean(x)
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - statistics.fmean(y)) for a, b in zip(x, y)) / sxx
    fit = lambda yy: sum((a - mx) * (b - statistics.fmean(yy)) for a, b in zip(x, yy)) / sxx
    ge = tot = 0
    for pm in itertools.permutations(y):
        tot += 1
        ge += abs(fit(list(pm))) >= abs(slope) - 1e-12
    return {"slope_per_decade": round(slope, 3), "p_exact": round(ge / tot, 5),
            "n_rungs": len(y), "points": [{"params_B": a, "gain_unsupported": b} for a, b in pts]}


def excludes_zero(r):
    return bool(r and "ci95" in r and (r["ci95"][0] > 0 or r["ci95"][1] < 0))


def macro_values(inter, slope):
    """What scripts/make_masked_table.py would print for the 'nli' label."""
    vals = [v["interaction"] for v in inter.values()]
    withci = [v for v in inter.values() if "ci95" in v]
    allsig = all(v["ci95"][0] > 0 for v in withci) and withci
    num = lambda x, dp=1: f"{x:+.{dp}f}"
    out = {"MaskInterNliMin": num(min(vals)), "MaskInterNliMax": num(max(vals)),
           "MaskInterNliN": str(len(vals)), "MaskInterNliAllSig": "all" if allsig else "not all"}
    if slope:
        out.update({"MaskSlopeNli": num(slope["slope_per_decade"]),
                    "MaskSlopeNliP": f"{slope['p_exact']:.3f}",
                    "MaskSlopeNliRungs": str(slope["n_rungs"])})
    return out



# ---------------------------------------------------------------------------
# routing_baselines.py GOLD_NLI diagnostic (unmasked Hybrid; label per question)
# ---------------------------------------------------------------------------
def routing_gold_nli(qlab, B, seed):
    qs = RB.load_questions()
    models = sorted(d.name.replace("generations_", "")
                    for d in PROC.glob("generations_*") if d.is_dir())
    out = {}
    for m in models:
        cb, rt = RB.load_gen(m, "closed_book"), RB.load_gen(m, "hybrid")
        if not cb or not rt:
            continue
        common = sorted(q for q in qs if q in cb and q in rt)
        if not any(qlab.get(q) is not None for q in common):
            continue
        f_cb = {q: RB.token_f1(cb[q]["prediction"] or "", qs[q]["gold_answer"]) * 100 for q in common}
        f_rt = {q: RB.token_f1(rt[q]["prediction"] or "", qs[q]["gold_answer"]) * 100 for q in common}
        score = lambda q, use: (f_rt if use else f_cb)[q]
        pol = {"never": {q: score(q, False) for q in common},
               "always": {q: score(q, True) for q in common},
               "gold_nli": {q: score(q, bool(qlab.get(q))) for q in common}}
        stats = {k: RB.cluster_mean_ci([(RB.quiz_of(q), v[q]) for q in common], B, seed)
                 for k, v in pol.items()}
        ref = max(("always", "never"), key=lambda k: stats[k]["mean"])
        d = RB.paired_delta([(RB.quiz_of(q), pol["gold_nli"][q] - pol[ref][q]) for q in common], B, seed)
        out[m] = {"reference": ref, "gold_nli_policy": stats["gold_nli"],
                  "gold_nli_vs_reference": d,
                  "gold_nli_retrieve_rate": round(sum(1 for q in common if qlab.get(q)) / len(common), 4)}
    return out


def grep_consumers():
    pats = ["nli_support_labels", "support_interaction_masked", "unsupported_cost_vs_scale_masked",
            "MaskSlopeNli", "MaskInterNli", "gold_nli"]
    hits = collections.defaultdict(list)
    for pat in pats:
        try:
            res = subprocess.run(["git", "-C", str(ROOT), "grep", "-n", "--untracked", "-I", pat, "--",
                                  "pipeline", "scripts", "annotation", "sections", "appendix",
                                  "main.tex"], capture_output=True, text=True, timeout=60)
            for line in res.stdout.splitlines():
                f = line.split(":", 1)[0]
                if "/analysis/r1_" in f or "/analysis/r2_" in f:
                    continue
                hits[pat].append(line[:200])
        except Exception as exc:                      # pragma: no cover
            hits[pat].append(f"grep failed: {exc}")
    return dict(hits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    args = ap.parse_args()
    t_start = time.time()
    qs = MC.load_questions()
    manifest = load_json(MANIFEST)
    corpus = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        for line in f.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                corpus.setdefault(d["doc_id"], d.get("text", ""))

    # ---- 1. validation and reproduction checks ------------------------------
    und, chk_und = load_server_labels("undated600", qs, manifest)
    srv_meta = {j: load_json(SRV / f"{j}.meta.json") for j in ("repro1500", "undated600", "dated600")
                if (SRV / f"{j}.meta.json").exists()}
    stored = {(r["qid"], r["doc_id"]): r for r in read_jsonl(STORED_NLI)}
    e0 = {(r["qid"], r["doc_id"]): r for r in read_jsonl(E0_NLI600)}

    def compare(new_by, old_by, keys, new_lab="label", old_lab="nli_label"):
        keys = [k for k in keys if k in new_by and k in old_by]
        agree = sum(1 for k in keys if new_by[k][new_lab] == old_by[k][old_lab])
        diffs = [max(abs(new_by[k]["probs"][c] - old_by[k]["probs"][c]) for c in new_by[k]["probs"])
                 for k in keys]
        flips = [{"qid": k[0], "doc_id": k[1], "old": old_by[k][old_lab], "new": new_by[k][new_lab],
                  "old_probs": old_by[k]["probs"], "new_probs": new_by[k]["probs"]}
                 for k in keys if new_by[k][new_lab] != old_by[k][old_lab]]
        return {"n": len(keys), "label_agreement": agree,
                "label_agreement_rate": round(agree / max(len(keys), 1), 5),
                "max_abs_prob_diff": round(max(diffs), 4) if diffs else None,
                "n_prob_diff_over_0.001": sum(1 for x in diffs if x > 0.001),
                "n_prob_diff_over_0.01": sum(1 for x in diffs if x > 0.01),
                "label_flips": flips[:20]}

    rep = {(r["qid"], r["doc_id"]): r for r in read_jsonl(SRV / "repro1500.jsonl")}
    unmasked_keys = [(t["qid"], t["doc_id"]) for t in manifest["tuples"] if "unmasked" in t["sets"]]
    masked_keys = [(t["qid"], t["doc_id"]) for t in manifest["tuples"] if "masked" in t["sets"]]
    both = [(t["qid"], t["doc_id"]) for t in manifest["tuples"]
            if "masked" in t["sets"] and "unmasked" in t["sets"]]
    checks = {
        "server_label_file_validation": chk_und,
        "server_runs": srv_meta,
        "cuda_premise1500_vs_stored_mps_labels (300 random stored pairs)": compare(rep, stored, list(rep)),
        "cuda_premise600_vs_e0_mps_labels (all 4,390 unmasked pairs)": compare(und, e0, unmasked_keys),
        "tokenizer_note": (
            "Token ids were produced locally (transformers 5.11.0 DebertaV2Tokenizer), the tokenizer "
            "that reproduces the stored labels exactly; the server only ran the forward pass. A first "
            "server run that tokenized with the server's transformers 4.57.1 fast tokenizer is kept "
            "in r1_raw/diag_server_tokenizer_tf4.57.1 as a sensitivity record."),
    }
    if DIAG_TOK.exists():
        diag_rep = {(r["qid"], r["doc_id"]): r for r in read_jsonl(DIAG_TOK / "repro1500.jsonl")}
        diag_und = {(r["qid"], r["doc_id"]): r for r in read_jsonl(DIAG_TOK / "undated600.jsonl")}
        checks["tokenizer_sensitivity_server_tf4.57.1_tokenizer"] = {
            "premise1500_vs_stored": {k: v for k, v in compare(diag_rep, stored, list(diag_rep)).items()
                                      if k != "label_flips"},
            "premise600_vs_local_tokenizer_cuda": {k: v for k, v in compare(diag_und, und, list(und), "label", "label").items()
                                                   if k != "label_flips"},
        }

    # ---- 2. the masked label file -------------------------------------------
    tup = {(t["qid"], t["doc_id"]): t for t in manifest["tuples"]}
    n_written = 0
    with MASKED_LABELS.open("w", encoding="utf-8") as fh:
        ret_m = load_json(PROC / "retrieval_masked.json")["hybrid"]
        for q in sorted(qs):
            for rank, d in enumerate(ret_m[q][:5]):
                r = und[(q, d)]
                text = corpus.get(d) or ""
                assert sha(text[:600]) == r["premise_sha256"] == tup[(q, d)]["premise_sha256"]
                fh.write(json.dumps({
                    "qid": q, "doc_id": d, "masked_rank": rank,
                    "premise_sha256": r["premise_sha256"], "premise_chars": 600,
                    "hypothesis_sha256": r["hypothesis_sha256"],
                    "label": r["label"], "probs": r["probs"],
                    "also_in_unmasked_top5": "unmasked" in tup[(q, d)]["sets"],
                    "string_support_600": bool(VIS(qs[q]["gold_answer"], text)),
                    "labeller": "DeBERTa-v3-large-mnli-fever-anli-ling-wanli @b3546ea, CUDA fp32, "
                                "local tokenizer ids, max_length 384, batch 1"}) + "\n")
                n_written += 1

    masked_rows = read_jsonl(MASKED_LABELS)
    mlab = {(r["qid"], r["doc_id"]): r["label"] for r in masked_rows}
    ret_m = load_json(PROC / "retrieval_masked.json")["hybrid"]
    ret_u = load_json(PROC / "retrieval.json")["hybrid"]
    q_masked600 = q_entail_any(mlab, {q: ret_m[q][:5] for q in qs})
    q_stored = stored_style_q_labels(STORED_NLI)
    q_e0 = stored_style_q_labels(E0_NLI600)
    q_unmasked600_cuda = q_entail_any({k: und[k]["label"] for k in unmasked_keys},
                                      {q: ret_u[q][:5] for q in qs})

    # ---- 3. reproduce the stored masked block -------------------------------
    stored_mc = load_json(RES / "masked_counterfactual.json")
    s_full, sl_full = masked_interaction(
        lambda q, row: any(FULL(qs[q]["gold_answer"], t) for t in
                           [corpus.get(d, "") for d in (row.get("doc_ids") or [])] if t),
        args.B, SEED_MC)
    n_stored, sl_n_stored = masked_interaction(lambda q, row: q_stored.get(q), args.B, SEED_MC)
    repro = {
        "string_fulltext": repro_record(
            {"inter": stored_mc["support_interaction_masked"]["string"],
             "slope": stored_mc["unsupported_cost_vs_scale_masked"]["string"]},
            {"inter": s_full, "slope": sl_full},
            "masked_counterfactual.json support_interaction_masked['string'] + slope"),
        "nli_stored_labels": repro_record(
            {"inter": stored_mc["support_interaction_masked"]["nli"],
             "slope": stored_mc["unsupported_cost_vs_scale_masked"]["nli"]},
            {"inter": n_stored, "slope": sl_n_stored},
            "masked_counterfactual.json support_interaction_masked['nli'] + slope"),
    }
    if not all(v["reproduced_exactly"] for v in repro.values()):
        print("WARNING: stored masked interaction NOT reproduced", json.dumps(repro)[:2000])

    # ---- 4. recompute -------------------------------------------------------
    n_e0, sl_e0 = masked_interaction(lambda q, row: q_e0.get(q), args.B, SEED_MC)
    n_cuda_u, sl_cuda_u = masked_interaction(lambda q, row: q_unmasked600_cuda.get(q), args.B, SEED_MC)

    def sup_masked(q, row):
        docs = row.get("doc_ids") or []
        assert docs == ret_m[q][:5]
        return q_masked600[q]
    n_new, sl_new = masked_interaction(sup_masked, args.B, SEED_MC)
    s_600, sl_600 = masked_interaction(
        lambda q, row: any(VIS(qs[q]["gold_answer"], t) for t in
                           [corpus.get(d, "") for d in (row.get("doc_ids") or [])] if t),
        args.B, SEED_MC)
    e0vs = load_json(OUT / "e0_visible_support.json")["masked_hybrid_support_interaction"]
    s600_matches_e0 = all(
        {k: s_600[m].get(k) for k in ("interaction", "ci95", "n_supported")}
        == {k: e0vs["per_model"]["haiku" if m == "" else m]["new_visible600"].get(k)
            for k in ("interaction", "ci95", "n_supported")} for m in s_600)

    keep = ("interaction", "ci95", "p_plus_one", "gain_supported", "gain_unsupported",
            "n_supported", "n_unsupported", "n_quizzes")
    per_model = {}
    for m in n_stored:
        o, e, n = n_stored[m], n_e0.get(m), n_new.get(m)
        per_model["haiku" if m == "" else m] = {
            "old_stored_unmasked_contexts_premise1500": {k: o[k] for k in keep if k in o},
            "e0_unmasked_contexts_premise600_still_cross_context": {k: e[k] for k in keep if k in e} if e else None,
            "new_masked_contexts_premise600": {k: n[k] for k in keep if k in n} if n else None,
            "change_new_minus_old": round(n["interaction"] - o["interaction"], 3) if n else None,
            "ci_excludes_zero": {"old": excludes_zero(o), "new": excludes_zero(n)},
            "string600_masked_reference": {k: s_600[m][k] for k in ("interaction", "ci95", "n_supported")},
            "cuda_unmasked600_cross_context_device_check": {k: n_cuda_u[m][k] for k in ("interaction", "ci95", "n_supported")},
        }

    # question-level support transitions (all 878 questions, by regime)
    trans = {}
    for reg in ("static", "fresh"):
        c = collections.Counter()
        for q in qs:
            if (qs[q]["year"] >= MC.BOUNDARY) != (reg == "fresh"):
                continue
            c[("E" if q_stored[q] else "N") + "->" + ("E" if q_masked600[q] else "N")] += 1
        trans[reg] = {**dict(c), "n_supported_old": c["E->E"] + c["E->N"],
                      "n_supported_new": c["E->E"] + c["N->E"]}
    both_q = sum(1 for q in qs if q_masked600[q] == q_e0[q])
    ctx_trans_masked_vs_unmasked = {
        "n_questions_masked_label_equals_e0_unmasked600_label": both_q,
        "n_questions": len(qs),
        "n_questions_context_set_differs": sum(1 for q in qs if ret_m[q][:5] != ret_u[q][:5]),
        "n_questions_context_SET_differs_ignoring_order": sum(1 for q in qs if set(ret_m[q][:5]) != set(ret_u[q][:5])),
    }
    label_dist = collections.Counter(r["label"] for r in masked_rows)
    tab = collections.Counter((r["string_support_600"], r["label"].startswith("entail")) for r in masked_rows)
    ctx_agree = {"both_support": tab[(True, True)], "string_only": tab[(True, False)],
                 "nli_only": tab[(False, True)], "neither": tab[(False, False)],
                 "raw_agreement": round((tab[(True, True)] + tab[(False, False)]) / len(masked_rows), 4)}

    # ---- 5. consumers --------------------------------------------------------
    old_macros = load_json(RES / "analysis/masked_reporting_values.json")
    new_macros = macro_values(n_new, sl_new)
    macros = {k: {"current_file_value": old_macros.get(k),
                  "value_reproduced_from_stored_labels": macro_values(n_stored, sl_n_stored).get(k),
                  "value_with_masked600_labels": v} for k, v in new_macros.items()}
    ledger_old = None
    for line in (RES / "claim_ledger.csv").read_text().splitlines():
        if line.startswith("unsupported_cost_slope_masked_nli,"):
            ledger_old = line
    order = ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
             "mistral_large", "", "sonnet", "gpt4omini"]
    table_col = {("haiku" if m == "" else m): {"current_cell": f"${n_stored[m]['interaction']:+.1f}$",
                                               "cell_with_masked600_labels": f"${n_new[m]['interaction']:+.1f}$",
                                               "excludes_zero_new": excludes_zero(n_new[m])}
                 for m in order if m in n_new}
    all_cells_sig_new = all(excludes_zero(v) for v in n_new.values()) and all(
        excludes_zero(v) for v in s_full.values())

    t0 = time.time()
    rb_stored_file = load_json(RES / "routing_baselines.json")
    rb_old = routing_gold_nli(q_stored, args.B, SEED_RB)
    rb_new = routing_gold_nli(q_e0, args.B, SEED_RB)
    rb_repro = repro_record(
        {m: {"reference": v["reference"], "gold_nli": v["policies"].get("gold_nli"),
             "vs": v["vs_reference"].get("gold_nli")} for m, v in rb_stored_file["models"].items()
         if "gold_nli" in v["policies"]},
        {m: {"reference": v["reference"], "gold_nli": v["gold_nli_policy"], "vs": v["gold_nli_vs_reference"]}
         for m, v in rb_old.items()},
        "routing_baselines.json policies.gold_nli, vs_reference.gold_nli, reference")
    rb_rt = round(time.time() - t0, 1)

    consumers = [
        {"consumer": "pipeline/masked_counterfactual.py -> results/masked_counterfactual.json "
                     "support_interaction_masked['nli'], unsupported_cost_vs_scale_masked['nli']",
         "labels_used": "results/nli_support_labels.jsonl (qwen32b UNMASKED Hybrid top-5, text[:1500]) per question",
         "analysed_contexts": "MASKED Hybrid top-5 (retrieval_masked.json)",
         "cross_context": True, "status": "RECOMPUTED here on masked text[:600] (support_interaction_masked_nli)"},
        {"consumer": "appendix/auto_masked_macros.tex MaskInterNli{Min,Max,N,AllSig}, MaskSlopeNli{,P,Rungs} "
                     "(scripts/make_masked_table.py); sections/05_results.tex:244 cites MaskSlopeNli/P",
         "labels_used": "via masked_counterfactual.json", "cross_context": True,
         "status": "values recomputed below (paper_macros); file NOT regenerated (no existing file modified)"},
        {"consumer": "appendix/auto_support_table.tex tab:support 'Masked (auto) Entail.' column; "
                     "figures/real_support_scale.pdf (scripts/make_support_table.py)",
         "labels_used": "via masked_counterfactual.json", "cross_context": True,
         "status": "per-model cells and slope recomputed below; files NOT regenerated"},
        {"consumer": "appendix/auto_fig_supportscale.tex + figures/data/support_scale.dat, support_fit.dat "
                     "(scripts/make_tikz_figures.py emit_support_scale)",
         "labels_used": "via masked_counterfactual.json unsupported_cost_vs_scale_masked['nli'].points",
         "cross_context": True, "status": "points and slope recomputed below; files NOT regenerated"},
        {"consumer": "results/claim_ledger.csv row unsupported_cost_slope_masked_nli (pipeline/rebuild_ledger.py)",
         "labels_used": "via masked_counterfactual.json", "cross_context": True,
         "status": "new value below; ledger NOT regenerated"},
        {"consumer": "pipeline/support_interaction.py labels['nli'] (unmasked Hybrid, Static+Fresh) and "
                     "unsupported_cost_vs_scale['nli']",
         "labels_used": "qwen32b unmasked Hybrid top-5; generator doc_ids identical for all generators (verified)",
         "cross_context": False,
         "status": "window corrected (600) in e0_nli_visible.json; dated sensitivity in r2_dated_nli.json"},
        {"consumer": "pipeline/routing_baselines.py GOLD_NLI diagnostic policy",
         "labels_used": "per-question entail-any on retrieval.json hybrid top-5 == generation doc_ids (verified)",
         "cross_context": False,
         "status": "window NOT corrected in e0; RECOMPUTED here with the e0 600-char labels "
                   "(routing_gold_nli). Not printed in any paper table (make_routing_table.py does not read it)"},
        {"consumer": "pipeline/support_label_validation.py (Task B passage-level NLI vs human)",
         "labels_used": "(qid, doc_id) pair labels; every Task B pair is in the unmasked set", "cross_context": False,
         "status": "window corrected in e0_human_alignment.json; 700-vs-600 restriction in r2_human_alignment.json"},
        {"consumer": "pipeline/support_label_typology.py (appendix app:typology entailment disagreement rates)",
         "labels_used": "(qid, doc_id) pair labels, text[:1500]", "cross_context": False,
         "status": "window NOT corrected in e0; rerun at 600 in r2_human_alignment.json (typology_600)"},
        {"consumer": "pipeline/nli_support.py (results/nli_support.json: context agreement 77.6%, "
                     "contradiction-inside-string 11.8%, gain_given_support_by_label)",
         "labels_used": "its own cache on unmasked contexts, text[:1500]", "cross_context": False,
         "status": "window NOT fully corrected in e0; rerun at 600 (agreement, contradiction-inside-string, "
                   "gain_given_support_by_label) in r1_remaining_string_support.json"},
        {"consumer": "pipeline/human_support_interaction.py; annotation/.../score_masked_support.py; "
                     "support_conditioned.py; k_ablation.py; analysis e2/e3/e5/e6/hidden_answer/stale scripts",
         "labels_used": "none (string or human labels only; checked by grep)", "cross_context": False,
         "status": "not applicable to NLI"},
    ]

    out = {
        "step": "R1 -- masked NLI support interaction on the masked contexts actually shown",
        "meta": meta([MANIFEST, MASKED_LABELS, STORED_NLI, E0_NLI600, SRV / "undated600.jsonl",
                      PROC / "retrieval_masked.json", RES / "masked_counterfactual.json"]),
        "tuple_construction": {
            "masked_contexts": "retrieval_masked.json['hybrid'][qid][:5]",
            "verified_equal_to_generation_doc_ids": manifest["context_set_checks"],
            "rendered_text": "corpus text[:600] (generator DOC_CHARS)",
            "hypothesis": 'The answer to "<question>" is <gold>.  (undated; as e0_nli_visible.py)',
            "question_label": "supported iff any of the 5 masked passages is labelled entailment",
            "n_masked_tuples_labelled": n_written,
            "n_masked_tuples_also_in_unmasked_top5": len(both),
            "masked_label_file": str(MASKED_LABELS.relative_to(ROOT)),
            "bundle_jobs_file_sha256": manifest["jobs_file_sha256"],
            "tokenizer": manifest.get("tokenizer_local"),
        },
        "reproduction_and_device_checks": {**checks, "stored_masked_counterfactual_blocks": repro,
                                           "string600_masked_equals_e0_visible_support": s600_matches_e0,
                                           "routing_gold_nli_stored": rb_repro},
        "support_interaction_masked_nli": {
            "script_estimator": "masked_counterfactual.py support-interaction block (verbatim), Static only",
            "seed": SEED_MC, "B": args.B,
            "weighting": ("QUESTION-POOLED arm means (gain = 100*(F1 masked RAG - F1 closed-book), token F1); "
                          "quiz-cluster bootstrap CI (interaction_ci). Not quiz-weighted."),
            "n_generators": len(n_new), "n_static_questions": sum(1 for q in qs if qs[q]["year"] < MC.BOUNDARY),
            "per_model": per_model,
            "unsupported_cost_vs_scale_masked_nli": {
                "old_stored": sl_n_stored, "e0_unmasked600_cross_context": sl_e0, "new_masked600": sl_new},
            "string_reference": {"fulltext_stored": sl_full, "visible600": sl_600},
            "n_generators_ci_excludes_zero": {"old": sum(excludes_zero(v) for v in n_stored.values()),
                                              "new": sum(excludes_zero(v) for v in n_new.values())},
            "range_interaction": {"old": [min(v["interaction"] for v in n_stored.values()),
                                          max(v["interaction"] for v in n_stored.values())],
                                  "new": [min(v["interaction"] for v in n_new.values()),
                                          max(v["interaction"] for v in n_new.values())]},
        },
        "question_level_support_transitions_stored_to_masked600": trans,
        "masked_vs_unmasked_labels": ctx_trans_masked_vs_unmasked,
        "masked_context_label_distribution": dict(label_dist),
        "masked_context_agreement_string600_vs_nli600": ctx_agree,
        "consumers_of_nli_labels": consumers,
        "consumer_grep_evidence": grep_consumers(),
        "paper_macros_from_masked_counterfactual": macros,
        "paper_table_tab_support_entail_column": {
            "per_model": table_col,
            "caption_every_interval_excludes_zero_still_true": all_cells_sig_new,
            "note": "the string column also changes under the 600-char window (e0_visible_support.json)"},
        "paper_figure_support_scale_nli_points": {"old": sl_n_stored["points"], "new": sl_new["points"],
                                                  "legend_old": f"Entailment ({sl_n_stored['slope_per_decade']:+.1f}/decade)",
                                                  "legend_new": f"Entailment ({sl_new['slope_per_decade']:+.1f}/decade)"},
        "claim_ledger_row": {"current": ledger_old,
                             "value_with_masked600_labels": sl_new["slope_per_decade"],
                             "p_exact": sl_new["p_exact"]},
        "routing_gold_nli": {
            "script": "pipeline/routing_baselines.py GOLD_NLI (gold-conditioned diagnostic, unmasked Hybrid)",
            "seed": SEED_RB, "B": args.B,
            "weighting": "QUIZ-WEIGHTED policy means and paired delta vs better fixed policy; quiz bootstrap",
            "per_model": {m: {"reference": rb_old[m]["reference"],
                              "old_premise1500": {"policy_mean": rb_old[m]["gold_nli_policy"]["mean"],
                                                  "vs_reference": rb_old[m]["gold_nli_vs_reference"],
                                                  "retrieve_rate": rb_old[m]["gold_nli_retrieve_rate"]},
                              "new_premise600": {"policy_mean": rb_new[m]["gold_nli_policy"]["mean"],
                                                 "vs_reference": rb_new[m]["gold_nli_vs_reference"],
                                                 "retrieve_rate": rb_new[m]["gold_nli_retrieve_rate"]}}
                          for m in rb_old},
            "runtime_sec": rb_rt},
        "not_causal": "Support is a gold-conditioned, non-randomised partition; descriptive associations only.",
        "runtime_total_sec": round(time.time() - t_start, 1),
    }
    dump(out, OUT / "r1_masked_nli.json")

    print("repro:", {k: v["reproduced_exactly"] for k, v in repro.items()}, "routing:", rb_repro["reproduced_exactly"])
    print("device checks:", {k: (v.get("label_agreement"), v.get("n"), v.get("max_abs_prob_diff"))
                             for k, v in checks.items() if isinstance(v, dict) and "n" in v})
    print(f"\n{'model':15s}{'old':>9s}{'e0x':>9s}{'new':>9s}  new CI            n_sup old/new")
    for m, v in per_model.items():
        o, e, n = (v["old_stored_unmasked_contexts_premise1500"],
                   v["e0_unmasked_contexts_premise600_still_cross_context"], v["new_masked_contexts_premise600"])
        print(f"{m:15s}{o['interaction']:9.2f}{e['interaction']:9.2f}{n['interaction']:9.2f}  "
              f"{str(n['ci95']):18s}{o['n_supported']}/{n['n_supported']}")
    print("slopes:", sl_n_stored["slope_per_decade"], sl_n_stored["p_exact"], "->",
          sl_new["slope_per_decade"], sl_new["p_exact"])
    print("wrote", OUT / "r1_masked_nli.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
