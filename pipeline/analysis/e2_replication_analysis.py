#!/usr/bin/env python3
"""E2 Phase A (preregistered, results/analysis/PREREG_E2_E3_E6.md): question-disjoint replication.

Samples: `original` = the paper's 878 questions; `new` = 800 unused questions (200 per year,
2022-2025, inclusion_prob stored per item). Both are answered under one configuration per
generator over the frozen 26,611-document corpus_full, with arms closed_book,
hybrid_full_masked (+3 days) and hybrid_full_unmasked.

Estimands, per sample and generator:
  mean F1 per arm; dmask = masked - unmasked; visible-text support rate;
  dsup = gain(S) - gain(U), gain = masked Hybrid F1 - closed-book F1, support = gold string
  in the first 600 characters of any shown masked passage.
Regimes: pooled, static (<2025), fresh (2025 only; the new sample has no 2026 items).
Primary comparison (fixed in the prereg): quiz-weighted, common years 2022-2025,
new - original, from one joint weekly-quiz bootstrap over the union of both samples
(shared quizzes resampled together). Stability margins judged on the 90% CI:
dmask +/-2 F1, dsup +/-5 F1. Secondary: question-pooled; new sample weighted by 1/inclusion_prob
(question-pooled Horvitz-Thompson style); original including 2026; the 43 new-sample questions
from quizzes absent from the original sample, descriptively.
Writes results/analysis/e2_replication.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import PROC, QuizBoot, M, write_json  # noqa: E402

SEED = 20260917
WINDOW = 600
MARGINS = {"dmask": 2.0, "dsup": 5.0}
ARMS = ("closed_book", "hybrid_full_masked", "hybrid_full_unmasked")
SOURCES = {
    "qwen3b": PROC / "analysis" / "kabl_out" / "out_e2" / "generations_qwen3b",
    "qwen7b": PROC / "analysis" / "kabl_out" / "out_e2" / "generations_qwen7b",
    "qwen32b": PROC / "analysis" / "generations_qwen32b" / "e2",
    "mistral_large": PROC / "analysis" / "generations_mistral_large" / "e2",
}
BUNDLE = PROC / "analysis" / "e2_bundle"


def read(p: Path) -> dict:
    rows = {}
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["qid"]] = r
    return rows


class WBoot(QuizBoot):
    """QuizBoot with optional per-question weights (inclusion-probability reweighting)."""

    def wsums(self, v, mask, w):
        m = np.asarray(mask, bool)
        S = np.bincount(self.qi, weights=np.where(m, w * v, 0.0), minlength=self.nq)
        N = np.bincount(self.qi, weights=np.where(m, w, 0.0), minlength=self.nq)
        return S, N

    def mean(self, v, mask, weighting, w=None):
        w = np.ones(len(self.qids)) if w is None else w
        S, N = self.wsums(np.asarray(v, float), mask, w)
        if weighting == "question_pooled":
            obs = S.sum() / N.sum() if N.sum() > 0 else np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                return float(obs), (self.W @ S) / (self.W @ N)
        has = N > 0
        qm = np.zeros(self.nq)
        qm[has] = S[has] / N[has]
        h = has.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(qm[has].mean()), (self.W @ (qm * h)) / (self.W @ h)


def summ(obs, draws, margin=None):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    lo95, hi95 = np.percentile(d, [2.5, 97.5])
    lo90, hi90 = np.percentile(d, [5, 95])
    out = {"estimate": round(float(obs), 2), "ci95": [round(float(lo95), 2), round(float(hi95), 2)],
           "ci90": [round(float(lo90), 2), round(float(hi90), 2)]}
    if margin is not None:
        if lo95 > 0 or hi95 < 0:
            out["verdict"] = "changed"
        elif -margin < lo90 and hi90 < margin:
            out["verdict"] = "stable_within_margin"
        else:
            out["verdict"] = "stability_not_established"
        out["margin"] = margin
    return out


def main() -> int:
    qs = {}
    for line in (BUNDLE / "questions.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            qs[d["qid"]] = d
    qids = sorted(qs)
    rankings = json.loads((BUNDLE / "rankings.json").read_text())
    corpus = {}
    for line in (BUNDLE / "corpus_subset.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            corpus[d["doc_id"]] = d["text"]
    boot = WBoot(qids, seed=SEED)
    year = np.array([int(str(qs[q]["date"])[:4]) for q in qids])
    new = np.array([qs[q].get("e2_sample") == "new" for q in qids])
    orig = ~new
    common = year <= 2025
    ipw = np.array([1.0 / qs[q]["inclusion_prob"] if qs[q].get("e2_sample") == "new" else 1.0 for q in qids])
    orig_quizzes = {q.split("_")[0] for q in qids if qs[q].get("e2_sample") != "new"}
    in_new_quiz = np.array([qs[q].get("e2_sample") == "new" and q.split("_")[0] not in orig_quizzes for q in qids])
    regimes = {"pooled": np.ones(len(qids), bool), "static": year < 2025, "fresh": year >= 2025}
    samples = {"original_common": orig & common, "new": new, "original_all": orig,
               "new_in_unseen_quizzes": new & in_new_quiz}
    support = np.array([any(M.gold_in_text(qs[q]["gold_answer"], corpus.get(d, "")[:WINDOW])
                            for d in rankings["hybrid_full_masked"][q]) for q in qids])

    out = {"prereg": "results/analysis/PREREG_E2_E3_E6.md#E2", "seed": SEED, "B": boot.B,
           "primary": "quiz_weighted, original_common vs new, 90% CI inside margin",
           "margins": MARGINS,
           "counts": {s: {r: int((m & rm).sum()) for r, rm in regimes.items()} for s, m in samples.items()},
           "visible_support_rate_masked": {s: round(float(support[m].mean()), 4) for s, m in samples.items()},
           "models": {}}

    for tag, src in SOURCES.items():
        rows = {a: read(src / f"{a}.jsonl") for a in ARMS}
        missing = {a: sum(q not in rows[a] for q in qids) for a in ARMS}
        if any(missing.values()):
            out["models"][tag] = {"status": "incomplete", "missing": missing}
            continue
        mism = {a: sum(list(rows[a][q].get("doc_ids") or []) != rankings[a][q] for q in qids) for a in ARMS}
        if any(mism.values()):
            out["models"][tag] = {"status": "doc_ids mismatch", "n": mism}
            continue
        F = {a: np.array([100 * M.token_f1(rows[a][q].get("prediction") or "", qs[q]["gold_answer"]) for q in qids])
             for a in ARMS}
        dmask_v = F["hybrid_full_masked"] - F["hybrid_full_unmasked"]
        gain = F["hybrid_full_masked"] - F["closed_book"]
        rec = {"n_reused_from_frozen": {a: sum(bool(rows[a][q].get("reused_from_frozen")) for q in qids) for a in ARMS},
               "estimates": {}, "new_minus_original_common": {}}
        for wt in ("quiz_weighted", "question_pooled"):
            est, draws = {}, {}
            for s, sm in samples.items():
                est[s] = {}
                for r, rm in regimes.items():
                    m = sm & rm
                    if m.sum() < 10:
                        continue
                    cell = {}
                    for a in ARMS:
                        cell[f"f1_{a}"] = round(boot.mean(F[a], m, wt)[0], 2)
                    o, d = boot.mean(dmask_v, m, wt)
                    cell["dmask"] = summ(o, d)
                    draws[(s, r, "dmask")] = (o, d)
                    oS, dS = boot.mean(gain, m & support, wt)
                    oU, dU = boot.mean(gain, m & ~support, wt)
                    cell["dsup"] = summ(oS - oU, dS - dU)
                    cell["dsup"].update({"gain_supported": round(oS, 2), "gain_unsupported": round(oU, 2),
                                         "n_supported": int((m & support).sum()),
                                         "n_unsupported": int((m & ~support).sum())})
                    draws[(s, r, "dsup")] = (oS - oU, dS - dU)
                    est[s][r] = cell
            rec["estimates"][wt] = est
            rec["new_minus_original_common"][wt] = {
                r: {k: summ(draws[("new", r, k)][0] - draws[("original_common", r, k)][0],
                            draws[("new", r, k)][1] - draws[("original_common", r, k)][1], MARGINS[k])
                    for k in ("dmask", "dsup")}
                for r in regimes if ("new", r, "dmask") in draws and ("original_common", r, "dmask") in draws}
        # inclusion-probability-weighted population estimate for the new sample (secondary)
        o, d = boot.mean(dmask_v, new, "question_pooled", ipw)
        oS, dS = boot.mean(gain, new & support, "question_pooled", ipw)
        oU, dU = boot.mean(gain, new & ~support, "question_pooled", ipw)
        rec["new_ipw_question_pooled"] = {"dmask": summ(o, d), "dsup": summ(oS - oU, dS - dU)}
        out["models"][tag] = rec

    p = write_json(out, "e2_replication.json")
    print(f"wrote {p}")
    for tag, rec in out["models"].items():
        if "new_minus_original_common" not in rec:
            print(tag, rec)
            continue
        c = rec["new_minus_original_common"]["quiz_weighted"]["pooled"]
        e = rec["estimates"]["quiz_weighted"]
        print(tag, "dmask orig/new", e["original_common"]["pooled"]["dmask"]["estimate"], e["new"]["pooled"]["dmask"]["estimate"],
              "dsup orig/new", e["original_common"]["pooled"]["dsup"]["estimate"], e["new"]["pooled"]["dsup"]["estimate"],
              "diff", c["dmask"]["verdict"], c["dsup"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
