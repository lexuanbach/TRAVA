#!/usr/bin/env python3
"""E6 (preregistered, results/analysis/PREREG_E2_E3_E6.md): corrected context-budget ablation.

Inputs (all generated under ONE configuration by pipeline/analysis/gen_hf_notrunc.py):
  data/processed/analysis/kabl_out/out/generations_<tag>/hybrid_masked_{k3,k5,k10refused,k10prefix}.jsonl
  data/processed/analysis/kabl_out/out_e3/generations_<tag>/closed_book.jsonl   (same configuration)

If the same-configuration closed-book file is missing, the frozen MPS closed-book cache is
used and every interaction is flagged `closed_book_config: "mixed (sensitivity only)"`.

Outputs results/analysis/e6_kablation.json with, per generator:
  * length audit and truncation flag for every arm (must be zero truncated);
  * reproduction of the frozen MPS k3/k5 caches (exact-match rate, mean F1 difference);
  * mean F1 per arm and paired contrasts k10prefix-k5, k10refused-k5, k3-k5;
  * support interaction gain(S) - gain(U) per arm, visible-text string label
    (gold in the first 600 characters of any shown passage), Static / Fresh / pooled,
    question-pooled (primary, matches k_ablation.py) and quiz-weighted;
  * support transition counts k5 -> k10prefix and k5 -> k10refused;
  * the invalid frozen k10 values, for the record, labelled as invalid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import (PROC, QuizBoot, WEIGHTINGS, is_fresh, load_questions,  # noqa: E402
                             load_rows, summarize, write_json, M)

ARMS = ("k3", "k5", "k10refused", "k10prefix")
TAGS = ("qwen1_5b", "qwen3b", "qwen7b")
WINDOW = 600
SEED = 20260917
KOUT = PROC / "analysis" / "kabl_out" / "out"
E3OUT = PROC / "analysis" / "kabl_out" / "out_e3"


def read(p: Path) -> dict:
    rows = {}
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["qid"]] = r
    return rows


def main() -> int:
    qs = load_questions()
    qids = sorted(qs)
    corpus = {}
    for line in (PROC / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["doc_id"]] = d.get("text", "")
    boot = QuizBoot(qids, seed=SEED)
    fresh = np.array([is_fresh(qs[q]) for q in qids])
    regimes = {"static": ~fresh, "fresh": fresh, "pooled": np.ones(len(qids), bool)}

    def f1(rows):
        return np.array([100.0 * M.token_f1(rows[q].get("prediction") or "", qs[q]["gold_answer"])
                         for q in qids])

    def visible_support(rows):
        return np.array([any(M.gold_in_text(qs[q]["gold_answer"], corpus.get(d, "")[:WINDOW])
                             for d in (rows[q].get("doc_ids") or [])) for q in qids])

    out = {"prereg": "results/analysis/PREREG_E2_E3_E6.md#E6", "seed": SEED, "B": boot.B,
           "support_label": f"gold string in first {WINDOW} chars of any shown passage",
           "primary_weighting": "question_pooled", "models": {}}

    for tag in TAGS:
        arms = {a: read(KOUT / f"generations_{tag}" / f"hybrid_masked_{a}.jsonl") for a in ARMS}
        if any(len(arms[a]) < len(qids) for a in ARMS):
            out["models"][tag] = {"status": "incomplete", "rows": {a: len(arms[a]) for a in ARMS}}
            continue
        cb = read(E3OUT / f"generations_{tag}" / "closed_book.jsonl")
        cb_cfg = "same configuration"
        if len(cb) < len(qids):
            cb, cb_cfg = load_rows(tag, "closed_book"), "mixed (sensitivity only): frozen MPS cache"
        rec = {"closed_book_config": cb_cfg, "rows": {a: len(arms[a]) for a in ARMS}}

        rec["truncation"] = {a: {"n_truncated": sum(bool(r.get("truncated")) for r in arms[a].values()),
                                 "max_input_tokens": max(r.get("n_input_tokens", 0) for r in arms[a].values())}
                             for a in ARMS}
        # reproduction of frozen MPS caches (k3, k5 only; frozen k10 is invalid)
        repro = {}
        for a, frozen_cond in (("k3", "hybrid_masked_k3"), ("k5", "hybrid_masked")):
            fz = load_rows(tag, frozen_cond)
            common = [q for q in qids if q in fz]
            if common:
                same = sum((fz[q].get("prediction") or "").strip() == (arms[a][q].get("prediction") or "").strip()
                           for q in common)
                d = np.mean([100 * (M.token_f1(arms[a][q]["prediction"] or "", qs[q]["gold_answer"])
                                    - M.token_f1(fz[q]["prediction"] or "", qs[q]["gold_answer"])) for q in common])
                same_docs = sum(list(fz[q].get("doc_ids") or []) == list(arms[a][q].get("doc_ids") or [])
                                for q in common)
                repro[a] = {"frozen_condition": frozen_cond, "n": len(common), "same_doc_ids": same_docs,
                            "exact_prediction_match": same, "mean_f1_new_minus_frozen": round(float(d), 3)}
        rec["reproduction_vs_frozen_mps"] = repro

        F = {a: f1(arms[a]) for a in ARMS}
        Fcb = f1(cb)
        S = {a: visible_support(arms[a]) for a in ARMS}

        rec["mean_f1"] = {}
        rec["contrasts"] = {}
        for w in WEIGHTINGS:
            rec["mean_f1"][w] = {a: summarize(*boot.est(w, F[a], regimes["pooled"])) for a in ARMS}
            rec["mean_f1"][w]["closed_book"] = summarize(*boot.est(w, Fcb, regimes["pooled"]))
            rec["contrasts"][w] = {}
            for a in ("k3", "k10prefix", "k10refused"):
                rec["contrasts"][w][f"{a}_minus_k5"] = {
                    r: summarize(*boot.est(w, F[a] - F["k5"], m)) for r, m in regimes.items()}
            rec["contrasts"][w]["k10prefix_minus_k10refused"] = {
                r: summarize(*boot.est(w, F["k10prefix"] - F["k10refused"], m)) for r, m in regimes.items()}

        rec["interaction"] = {}
        # secondary (declared deviation, see PREREG "Deviations"): paired difference of interactions
        rec["interaction_difference_vs_k5"] = {}
        for w in WEIGHTINGS:
            rec["interaction"][w] = {}
            raw = {}
            for a in ARMS:
                g = F[a] - Fcb
                rec["interaction"][w][a] = {}
                for r, m in regimes.items():
                    oS, dS = boot.est(w, g, m & S[a])
                    oU, dU = boot.est(w, g, m & ~S[a])
                    raw[(a, r)] = (oS - oU, np.asarray(dS) - np.asarray(dU))
                    rec["interaction"][w][a][r] = summarize(
                        oS - oU, dS - dU,
                        extra={"gain_supported": round(oS, 2), "gain_unsupported": round(oU, 2),
                               "n_supported": int((m & S[a]).sum()), "n_unsupported": int((m & ~S[a]).sum())})
            rec["interaction_difference_vs_k5"][w] = {
                f"{a}_minus_k5": {r: summarize(raw[(a, r)][0] - raw[("k5", r)][0], raw[(a, r)][1] - raw[("k5", r)][1])
                                  for r in regimes}
                for a in ("k3", "k10prefix", "k10refused")}
        rec["support_transitions"] = {
            f"k5_to_{a}": {"S->S": int((S["k5"] & S[a]).sum()), "S->U": int((S["k5"] & ~S[a]).sum()),
                           "U->S": int((~S["k5"] & S[a]).sum()), "U->U": int((~S["k5"] & ~S[a]).sum())}
            for a in ("k3", "k10prefix", "k10refused")}
        out["models"][tag] = rec

    frozen = json.loads((Path(PROC).parents[1] / "results" / "k_ablation.json").read_text())
    out["frozen_k10_invalid_for_record"] = {
        m: frozen["per_model"][m]["k10"] for m in frozen.get("per_model", {}) if "k10" in frozen["per_model"][m]}
    out["frozen_k10_invalid_reason"] = ("local_generate.py truncated inputs at 1536 tokens from the right; "
                                        "639/878 k10 prompts lost the question (results/analysis/token_length_audit.json)")
    p = write_json(out, "e6_kablation.json")
    print(f"wrote {p}")
    for tag, rec in out["models"].items():
        if "interaction" not in rec:
            print(tag, rec)
            continue
        row = rec["interaction"]["question_pooled"]
        print(tag, rec["closed_book_config"], {a: (row[a]["static"]["estimate"], row[a]["static"]["ci95"]) for a in ARMS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
