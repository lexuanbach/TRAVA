#!/usr/bin/env python3
"""Rebuttal-review checks R3 and R5 (read-only over existing outputs).

R3  k-ablation contrasts that name what changed in each arm:
    * order-preservation audit of every arm against k5 (first five positions);
    * support interaction at k10prefix and k10refused with the support partition FIXED
      at its k5 definition, next to the arm's own partition, so a change in the
      interaction is not confounded by questions moving between support arms;
    * support transitions k5 -> arm.
R5  scope checks behind the stale-index and masking comparisons:
    * how many Static / Fresh context sets differ between unmasked Hybrid and the stale index,
      and how many unmasked Static contexts contained post-2024 documents;
    * masking effect (hybrid_masked - hybrid) and stale effect (hybrid_stale - hybrid) on the SAME
      questions, generators, regime and weighting, so the two magnitudes are comparable.
Writes results/analysis/r3_r5_checks.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import PROC, QuizBoot, WEIGHTINGS, is_fresh, load_questions, load_rows, summarize, write_json, M  # noqa: E402

SEED = 20260918
WINDOW = 600
KOUT = PROC / "analysis" / "kabl_out" / "out"
E3OUT = PROC / "analysis" / "kabl_out" / "out_e3"


def read(p):
    return {json.loads(l)["qid"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}


def main():
    qs = load_questions()
    qids = sorted(qs)
    corpus = {}
    for line in (PROC / "corpus.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["doc_id"]] = d
    boot = QuizBoot(qids, seed=SEED)
    fresh = np.array([is_fresh(qs[q]) for q in qids])
    regimes = {"static": ~fresh, "fresh": fresh, "pooled": np.ones(len(qids), bool)}
    f1 = lambda rows: np.array([100 * M.token_f1(rows[q].get("prediction") or "", qs[q]["gold_answer"]) for q in qids])
    vis = lambda rows: np.array([any(M.gold_in_text(qs[q]["gold_answer"], corpus.get(d, {}).get("text", "")[:WINDOW])
                                     for d in rows[q].get("doc_ids") or []) for q in qids])
    out = {"seed": SEED, "B": boot.B, "R3": {}, "R5": {}}

    rank = json.loads((PROC / "analysis" / "kabl_bundle" / "rankings.json").read_text())
    out["R3"]["order_audit_vs_k5"] = {
        a: {"contains_all_k5_docs": sum(set(rank["k5"][q]) <= set(rank[a][q]) for q in qids),
            "first_five_positions_equal_k5": sum(rank[a][q][:5] == rank["k5"][q] for q in qids),
            "n": len(qids)} for a in ("k10refused", "k10prefix")}
    for tag in ("qwen1_5b", "qwen3b", "qwen7b"):
        arms = {a: read(KOUT / f"generations_{tag}" / f"hybrid_masked_{a}.jsonl") for a in ("k5", "k10prefix", "k10refused")}
        cb = read(E3OUT / f"generations_{tag}" / "closed_book.jsonl")
        F = {a: f1(r) for a, r in arms.items()}
        Fcb = f1(cb)
        S = {a: vis(r) for a, r in arms.items()}
        rec = {"transitions_from_k5": {a: {"U->S": int((~S["k5"] & S[a]).sum()), "S->U": int((S["k5"] & ~S[a]).sum())}
                                       for a in ("k10prefix", "k10refused")}}
        for w in WEIGHTINGS:
            rec[w] = {}
            for r_name, m in regimes.items():
                cell = {}
                raw = {}
                for a in ("k5", "k10prefix", "k10refused"):
                    for part_name, part in (("own_partition", S[a]), ("k5_partition", S["k5"])):
                        g = F[a] - Fcb
                        oS, dS = boot.est(w, g, m & part)
                        oU, dU = boot.est(w, g, m & ~part)
                        raw[(a, part_name)] = (oS - oU, np.asarray(dS) - np.asarray(dU))
                        cell[f"{a}/{part_name}"] = summarize(oS - oU, dS - dU, extra={
                            "gain_supported": round(oS, 2), "gain_unsupported": round(oU, 2),
                            "n_supported": int((m & part).sum())})
                for a in ("k10prefix", "k10refused"):
                    for part_name in ("own_partition", "k5_partition"):
                        o = raw[(a, part_name)][0] - raw[("k5", "own_partition")][0]
                        d = raw[(a, part_name)][1] - raw[("k5", "own_partition")][1]
                        cell[f"diff_{a}_minus_k5/{part_name}"] = summarize(o, d)
                o, d = boot.est(w, F["k10prefix"] - F["k5"], m)
                cell["mean_f1_k10prefix_minus_k5"] = summarize(o, d)
                o, d = boot.est(w, F["k10refused"] - F["k10prefix"], m)
                cell["mean_f1_order_k10refused_minus_k10prefix"] = summarize(o, d)
                rec[w][r_name] = cell
        out["R3"][tag] = rec

    # ---------------- R5 ----------------
    hyb = {m: load_rows(m, "hybrid") for m in ("qwen3b", "qwen32b", "mistral_large")}
    stale = {m: load_rows(m, "hybrid_stale") for m in hyb}
    masked = {m: load_rows(m, "hybrid_masked") for m in hyb}
    ref = hyb["qwen3b"]
    changed = np.array([list(ref[q].get("doc_ids") or []) != list(stale["qwen3b"][q].get("doc_ids") or []) for q in qids])

    def has_post2024(dids):
        return any(str(corpus.get(d, {}).get("publish_date", ""))[:4] >= "2025" for d in dids)
    post = np.array([has_post2024(ref[q].get("doc_ids") or []) for q in qids])
    out["R5"]["stale_vs_unmasked_context_change"] = {
        "static_changed": int((changed & ~fresh).sum()), "static_n": int((~fresh).sum()),
        "fresh_changed": int((changed & fresh).sum()), "fresh_n": int(fresh.sum()),
        "static_unmasked_contexts_with_2025plus_doc": int((post & ~fresh).sum()),
        "doc_ids_identical_across_generators": all(
            list(hyb[m][q].get("doc_ids") or []) == list(ref[q].get("doc_ids") or []) for m in hyb for q in qids)}
    same_scale = {}
    for m in hyb:
        Fh, Fs, Fm = f1(hyb[m]), f1(stale[m]), f1(masked[m])
        same_scale[m] = {w: {r: {"stale_minus_unmasked": summarize(*boot.est(w, Fs - Fh, rm)),
                                 "masked_minus_unmasked": summarize(*boot.est(w, Fm - Fh, rm))}
                             for r, rm in regimes.items()} for w in WEIGHTINGS}
    out["R5"]["same_scale_effects"] = same_scale
    p = write_json(out, "r3_r5_checks.json")
    print("wrote", p)
    print(json.dumps(out["R3"]["order_audit_vs_k5"]), json.dumps(out["R5"]["stale_vs_unmasked_context_change"]))
    for tag in ("qwen1_5b", "qwen3b", "qwen7b"):
        c = out["R3"][tag]["question_pooled"]["static"]
        print(tag, {k: (v["estimate"], v["ci95"]) for k, v in c.items() if "partition" in k or "mean_f1" in k})
    for m, v in same_scale.items():
        print(m, {w: {r: (x["stale_minus_unmasked"]["estimate"], x["masked_minus_unmasked"]["estimate"], x["masked_minus_unmasked"]["ci95"]) for r, x in v[w].items()} for w in v})


if __name__ == "__main__":
    main()
