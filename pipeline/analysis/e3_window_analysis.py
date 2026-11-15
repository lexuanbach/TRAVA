#!/usr/bin/env python3
"""E3 (preregistered, results/analysis/PREREG_E2_E3_E6.md): admissibility-window sensitivity.

Arms per generator: closed_book, hybrid_w0, hybrid_w1, hybrid_w3, hybrid_w7, hybrid_unmasked, all
from one configuration per generator:
  HF rungs   data/processed/analysis/kabl_out/out_e3/generations_<tag>/<arm>.jsonl  (gen_hf_notrunc.py)
  GGUF rungs data/processed/analysis/generations_<tag>/e3/<arm>.jsonl (gen_gguf_notrunc.py; rows whose
             prompt equals a frozen row reuse it, valid because frozen rows reproduced exactly)

Primary (quiz-weighted): F1(w) - F1(unmasked) for w in {0,1,3,7}; F1(0)-F1(3); F1(7)-F1(3).
Equivalence margin +/-2 F1 judged on the 90% CI; "different" if the 95% CI excludes 0.
Holm across generators within each primary contrast. Question-pooled reported alongside.
Also: Static/Fresh breakdown, the changed-context subgroup (ordered top-5 differs from w3),
rescued/harmed counts (answer coverage F1>=0.5 flips), and visible-text support transitions.
Writes results/analysis/e3_window.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import PROC, QuizBoot, WEIGHTINGS, is_fresh, load_questions, write_json, M  # noqa: E402

SEED = 20260917
MARGIN = 2.0
WINDOW = 600
ARMS = ("hybrid_w0", "hybrid_w1", "hybrid_w3", "hybrid_w7", "hybrid_unmasked")
SOURCES = {
    "qwen1_5b": PROC / "analysis" / "kabl_out" / "out_e3" / "generations_qwen1_5b",
    "qwen3b": PROC / "analysis" / "kabl_out" / "out_e3" / "generations_qwen3b",
    "qwen7b": PROC / "analysis" / "kabl_out" / "out_e3" / "generations_qwen7b",
    "qwen32b": PROC / "analysis" / "generations_qwen32b" / "e3",
    "mistral_large": PROC / "analysis" / "generations_mistral_large" / "e3",
}
PRIMARY = {"hybrid_w0_minus_unmasked": ("hybrid_w0", "hybrid_unmasked"),
           "hybrid_w1_minus_unmasked": ("hybrid_w1", "hybrid_unmasked"),
           "hybrid_w3_minus_unmasked": ("hybrid_w3", "hybrid_unmasked"),
           "hybrid_w7_minus_unmasked": ("hybrid_w7", "hybrid_unmasked"),
           "hybrid_w0_minus_w3": ("hybrid_w0", "hybrid_w3"),
           "hybrid_w7_minus_w3": ("hybrid_w7", "hybrid_w3")}


def read(p: Path) -> dict:
    rows = {}
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["qid"]] = r
    return rows


def ci(draws, level):
    d = np.asarray(draws)
    d = d[np.isfinite(d)]
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(d, [a, 100 - a])
    return [round(float(lo), 2), round(float(hi), 2)]


def contrast(boot, w, v, mask):
    obs, draws = boot.est(w, v, mask)
    d = np.asarray(draws)
    d = d[np.isfinite(d)]
    le, ge = int((d <= 0).sum()), int((d >= 0).sum())
    p = min(1.0, 2.0 * min(le + 1, ge + 1) / (len(d) + 1))
    c95, c90 = ci(d, 0.95), ci(d, 0.90)
    if c95[0] > 0 or c95[1] < 0:
        verdict = "different"
    elif -MARGIN < c90[0] and c90[1] < MARGIN:
        verdict = "practically_equivalent"
    else:
        verdict = "not_shown_equivalent"
    return {"estimate": round(obs, 2), "ci95": c95, "ci90": c90, "p_two_sided_boot": round(p, 5),
            "verdict_margin_2": verdict, **boot.counts(mask)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main() -> int:
    qs = load_questions()
    qids = sorted(qs)
    rankings = json.loads((PROC / "analysis" / "e3_bundle" / "rankings.json").read_text())
    corpus = {}
    for line in (PROC / "analysis" / "e3_bundle" / "corpus_subset.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["doc_id"]] = d["text"]
    boot = QuizBoot(qids, seed=SEED)
    fresh = np.array([is_fresh(qs[q]) for q in qids])
    allq = np.ones(len(qids), bool)
    regimes = {"pooled": allq, "static": ~fresh, "fresh": fresh}
    changed = {a: np.array([rankings[a][q] != rankings["hybrid_w3"][q] for q in qids]) for a in ARMS}
    support = {a: np.array([any(M.gold_in_text(qs[q]["gold_answer"], corpus.get(d, "")[:WINDOW])
                                for d in rankings[a][q]) for q in qids]) for a in ARMS}

    out = {"prereg": "results/analysis/PREREG_E2_E3_E6.md#E3", "seed": SEED, "B": boot.B,
           "equivalence_margin_f1": MARGIN, "primary_weighting": "quiz_weighted",
           "n_changed_vs_w3": {a: int(changed[a].sum()) for a in ARMS},
           "visible_support_rate": {a: round(float(support[a].mean()), 4) for a in ARMS},
           "support_transitions_vs_w3": {a: {"gained": int((~support["hybrid_w3"] & support[a]).sum()),
                                             "lost": int((support["hybrid_w3"] & ~support[a]).sum())}
                                         for a in ARMS if a != "hybrid_w3"},
           "models": {}, "holm": {}}

    for tag, src in SOURCES.items():
        rows = {a: read(src / f"{a}.jsonl") for a in ARMS + ("closed_book",)}
        missing = {a: len(qids) - sum(q in rows[a] for q in qids) for a in rows}
        if any(missing.values()):
            out["models"][tag] = {"status": "incomplete", "missing": missing}
            continue
        bad = [a for a in ARMS if any(list(rows[a][q].get("doc_ids") or []) != rankings[a][q] for q in qids)]
        if bad:
            out["models"][tag] = {"status": "doc_ids do not match bundle rankings", "arms": bad}
            continue
        F = {a: np.array([100 * M.token_f1(rows[a][q].get("prediction") or "", qs[q]["gold_answer"]) for q in qids])
             for a in rows}
        rec = {"source": str(src.relative_to(PROC.parents[1])),
               "n_reused_from_frozen": {a: sum(bool(rows[a][q].get("reused_from_frozen")) for q in qids)
                                        for a in rows},
               "mean_f1": {w: {a: round(boot.est(w, F[a], allq)[0], 2) for a in rows} for w in WEIGHTINGS},
               "contrasts": {}, "changed_context_subgroup": {}}
        for w in WEIGHTINGS:
            rec["contrasts"][w] = {name: {r: contrast(boot, w, F[x] - F[y], m) for r, m in regimes.items()}
                                   for name, (x, y) in PRIMARY.items()}
        for a in ("hybrid_w0", "hybrid_w1", "hybrid_w7"):
            m = changed[a]
            cov_a, cov_3 = F[a] >= 50, F["hybrid_w3"] >= 50
            rec["changed_context_subgroup"][a] = {
                "n": int(m.sum()),
                "f1_minus_w3_question_pooled": round(float((F[a] - F["hybrid_w3"])[m].mean()), 2) if m.any() else None,
                "rescued": int((m & cov_a & ~cov_3).sum()), "harmed": int((m & ~cov_a & cov_3).sum())}
        m = changed["hybrid_unmasked"]
        rec["changed_context_subgroup"]["hybrid_unmasked"] = {
            "n": int(m.sum()),
            "w3_minus_unmasked_question_pooled": round(float((F["hybrid_w3"] - F["hybrid_unmasked"])[m].mean()), 2)}
        out["models"][tag] = rec

    done = [t for t, r in out["models"].items() if "contrasts" in r]
    for name in PRIMARY:
        ps = [out["models"][t]["contrasts"]["quiz_weighted"][name]["pooled"]["p_two_sided_boot"] for t in done]
        out["holm"][name] = dict(zip(done, [round(x, 5) for x in holm(ps)])) if ps else {}
    p = write_json(out, "e3_window.json")
    print(f"wrote {p}")
    for t in out["models"]:
        r = out["models"][t]
        if "contrasts" not in r:
            print(t, r)
            continue
        c = r["contrasts"]["quiz_weighted"]
        print(t, {k: (v["pooled"]["estimate"], v["pooled"]["ci90"], v["pooled"]["verdict_margin_2"]) for k, v in c.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
