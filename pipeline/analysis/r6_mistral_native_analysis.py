#!/usr/bin/env python3
"""R6: Mistral-Large-2 native-template sensitivity vs the frozen llama-2 fallback format.

Compares, on the same 878 questions and the same GGUF/decoding, the two prompt wrappers:
  * closed-book: mean F1 by regime, Static - Fresh gap (the paper's L1 anchor), and the
    2023 - 2026 conservative gap;
  * masked Hybrid: mean F1, retrieval gain, and the visible-text (600-char) string support
    interaction (question-pooled, as in the masked analysis), Static / Fresh / pooled;
  * paired native - fallback differences from one joint weekly-quiz bootstrap;
  * exact-prediction agreement between wrappers.
Each arm is complete within one wrapper: fallback rows come only from the frozen cache and
native rows only from the new run, so no paired comparison mixes formats within an arm.
Writes results/analysis/r6_mistral_native.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import PROC, QuizBoot, WEIGHTINGS, is_fresh, load_questions, load_rows, summarize, write_json, M  # noqa: E402

SEED = 20260918
NATIVE = PROC / "analysis" / "generations_mistral_large" / "native"


def read(p):
    return {json.loads(l)["qid"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()} if p.is_file() else {}


def main():
    qs = load_questions()
    corpus = {}
    for line in (PROC / "corpus.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["doc_id"]] = d.get("text", "")
    fb = {c: load_rows("mistral_large", c) for c in ("closed_book", "hybrid_masked")}
    nat = {c: read(NATIVE / f"{c}.jsonl") for c in ("closed_book", "hybrid_masked")}
    out = {"seed": SEED, "status": {c: {"native_rows": len(nat[c]), "fallback_rows": len(fb[c])} for c in nat}}
    qids = sorted(q for q in qs if all(q in fb[c] and q in nat[c] for c in fb))
    out["n_common"] = len(qids)
    if len(qids) < 100:
        write_json(out, "r6_mistral_native.json")
        print("incomplete", out)
        return
    boot = QuizBoot(qids, seed=SEED)
    year = np.array([int(str(qs[q]["date"])[:4]) for q in qids])
    fresh = np.array([is_fresh(qs[q]) for q in qids])
    regimes = {"static": ~fresh, "fresh": fresh, "pooled": np.ones(len(qids), bool)}
    F = {(w, c): np.array([100 * M.token_f1((src[c][q].get("prediction") or ""), qs[q]["gold_answer"]) for q in qids])
         for w, src in (("fallback", fb), ("native", nat)) for c in fb}
    sup = np.array([any(M.gold_in_text(qs[q]["gold_answer"], corpus.get(d, "")[:600])
                        for d in fb["hybrid_masked"][q].get("doc_ids") or []) for q in qids])
    out["exact_prediction_agreement"] = {c: round(float(np.mean([(fb[c][q]["prediction"] or "").strip() ==
                                                                  (nat[c][q]["prediction"] or "").strip() for q in qids])), 4)
                                         for c in fb}
    res = {}
    for wt in WEIGHTINGS:
        cell = {}
        for wrap in ("fallback", "native"):
            cb, rag = F[(wrap, "closed_book")], F[(wrap, "hybrid_masked")]
            d = {}
            for r, m in regimes.items():
                d[f"cb_{r}"] = summarize(*boot.est(wt, cb, m))
                d[f"rag_{r}"] = summarize(*boot.est(wt, rag, m))
            oS, dS = boot.est(wt, cb, ~fresh)
            oF, dF = boot.est(wt, cb, fresh)
            d["cb_gap_static_minus_fresh"] = summarize(oS - oF, np.asarray(dS) - np.asarray(dF))
            o23, d23 = boot.est(wt, cb, year == 2023)
            o26, d26 = boot.est(wt, cb, year == 2026)
            d["cb_gap_2023_minus_2026"] = summarize(o23 - o26, np.asarray(d23) - np.asarray(d26))
            g = rag - cb
            for r, m in regimes.items():
                a, da = boot.est(wt, g, m & sup)
                b, db = boot.est(wt, g, m & ~sup)
                d[f"interaction_visible_{r}"] = summarize(a - b, np.asarray(da) - np.asarray(db),
                                                          extra={"gain_supported": round(a, 2), "gain_unsupported": round(b, 2)})
            cell[wrap] = d
        diffs = {}
        for key in ("cb_gap_static_minus_fresh", "cb_gap_2023_minus_2026"):
            pass
        # paired native - fallback differences
        for c in ("closed_book", "hybrid_masked"):
            for r, m in regimes.items():
                diffs[f"{c}_{r}"] = summarize(*boot.est(wt, F[("native", c)] - F[("fallback", c)], m))
        # difference in the L1 gap and in the interaction, same draws
        def gap_draws(wrap):
            s, ds = boot.est(wt, F[(wrap, "closed_book")], ~fresh)
            f, df = boot.est(wt, F[(wrap, "closed_book")], fresh)
            return s - f, np.asarray(ds) - np.asarray(df)
        gn, dgn = gap_draws("native")
        gf, dgf = gap_draws("fallback")
        diffs["cb_gap_native_minus_fallback"] = summarize(gn - gf, dgn - dgf)
        def inter_draws(wrap, m):
            g = F[(wrap, "hybrid_masked")] - F[(wrap, "closed_book")]
            a, da = boot.est(wt, g, m & sup)
            b, db = boot.est(wt, g, m & ~sup)
            return a - b, np.asarray(da) - np.asarray(db)
        for r, m in regimes.items():
            i_n, di_n = inter_draws("native", m)
            i_f, di_f = inter_draws("fallback", m)
            diffs[f"interaction_{r}_native_minus_fallback"] = summarize(i_n - i_f, di_n - di_f)
        cell["native_minus_fallback"] = diffs
        res[wt] = cell
    out["results"] = res
    p = write_json(out, "r6_mistral_native.json")
    print("wrote", p, "n", len(qids), out["exact_prediction_agreement"])
    for wt in WEIGHTINGS:
        r = res[wt]
        print(wt, "gap fb/nat", r["fallback"]["cb_gap_static_minus_fresh"]["estimate"], r["native"]["cb_gap_static_minus_fresh"]["estimate"],
              "diff", r["native_minus_fallback"]["cb_gap_native_minus_fallback"]["estimate"], r["native_minus_fallback"]["cb_gap_native_minus_fallback"]["ci95"],
              "| inter static fb/nat", r["fallback"]["interaction_visible_static"]["estimate"], r["native"]["interaction_visible_static"]["estimate"],
              "diff", r["native_minus_fallback"]["interaction_static_native_minus_fallback"]["estimate"], r["native_minus_fallback"]["interaction_static_native_minus_fallback"]["ci95"])


if __name__ == "__main__":
    main()
