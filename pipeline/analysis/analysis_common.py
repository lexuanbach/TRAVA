#!/usr/bin/env python3
"""Shared helpers for the analysis cached-output analyses (read-only).

The statistical contract is:

  * Resampling unit: the weekly quiz (`clusters.quiz_id`). One draw resamples
    the FULL quiz set once, with replacement; every quantity (both regimes,
    every condition, every model on the same questions) is recomputed from the
    same multiplicity matrix, so contrasts across regimes/models are joint.
  * Two estimators are always reported and labelled:
      question_pooled : mean over questions (weights = quiz multiplicity x
                        number of eligible questions in the quiz)
      quiz_weighted   : mean over eligible quizzes of the within-quiz mean
                        (a quiz without an eligible question for that arm is
                        left out of that arm's denominator)
  * Boundary: Static = question `date` year < 2025, Fresh = year >= 2025. The
    `split` field is never used (its meaning differs between question files).

No existing pipeline file is modified; this module only imports from them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "pipeline"
if str(PIPE) not in sys.path:
    sys.path.insert(0, str(PIPE))

from clusters import quiz_id  # noqa: E402
import metrics as M  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "analysis"
BOUNDARY_YEAR = 2025
B_DEFAULT = 5000


def year_of(q: dict) -> int:
    return int(str(q["date"])[:4])


def is_fresh(q: dict) -> bool:
    return year_of(q) >= BOUNDARY_YEAR


def load_questions(name: str = "questions.jsonl") -> dict[str, dict]:
    out = {}
    for line in (PROC / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["qid"]] = d
    return out


def gen_path(model: str, cond: str) -> Path:
    # Claude Haiku 4.5 lives in the un-suffixed directory (see aggregate.py).
    d = "generations" if model == "haiku" else f"generations_{model}"
    return PROC / d / f"{cond}.jsonl"


def load_rows(model: str, cond: str) -> dict[str, dict]:
    p = gen_path(model, cond)
    if not p.is_file():
        return {}
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["qid"]] = r          # last occurrence wins (files are deduped)
    return rows


def f1_scores(rows: dict, questions: dict) -> dict[str, float]:
    return {q: 100.0 * M.token_f1(r.get("prediction") or "", questions[q]["gold_answer"])
            for q, r in rows.items() if q in questions}


class QuizBoot:
    """Joint weekly-quiz bootstrap over a fixed, ordered question list."""

    def __init__(self, qids, B: int = B_DEFAULT, seed: int = 0):
        self.qids = list(qids)
        self.quizzes = sorted({quiz_id(q) for q in self.qids})
        pos = {k: i for i, k in enumerate(self.quizzes)}
        self.qi = np.array([pos[quiz_id(q)] for q in self.qids], dtype=np.int64)
        self.nq = len(self.quizzes)
        self.B = B
        self.seed = seed
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.nq, size=(B, self.nq))
        W = np.zeros((B, self.nq), dtype=np.float64)
        rows = np.repeat(np.arange(B), self.nq)
        np.add.at(W, (rows, idx.ravel()), 1.0)
        self.W = W

    # -- core estimators ---------------------------------------------------
    def _sums(self, v, mask):
        v = np.asarray(v, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        S = np.bincount(self.qi, weights=np.where(m, v, 0.0), minlength=self.nq)
        N = np.bincount(self.qi, weights=m.astype(np.float64), minlength=self.nq)
        return S, N

    def pooled(self, v, mask):
        """(observed, draws) for the question-pooled mean of v over mask."""
        S, N = self._sums(v, mask)
        obs = S.sum() / N.sum() if N.sum() > 0 else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = (self.W @ S) / (self.W @ N)
        return float(obs), draws

    def quizw(self, v, mask):
        """(observed, draws) for the mean of within-quiz means over quizzes
        that contain at least one masked question."""
        S, N = self._sums(v, mask)
        has = N > 0
        qm = np.zeros(self.nq)
        qm[has] = S[has] / N[has]
        obs = qm[has].mean() if has.any() else np.nan
        h = has.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = (self.W @ (qm * h)) / (self.W @ h)
        return float(obs), draws

    def est(self, weighting: str, v, mask):
        return self.pooled(v, mask) if weighting == "question_pooled" else self.quizw(v, mask)

    def counts(self, mask):
        m = np.asarray(mask, dtype=bool)
        return {"n_questions": int(m.sum()),
                "n_quizzes": int(len(set(self.qi[m].tolist())))}


WEIGHTINGS = ("question_pooled", "quiz_weighted")


def summarize(obs: float, draws, nd: int = 2, extra: dict | None = None) -> dict:
    d = np.asarray(draws, dtype=np.float64)
    ok = d[np.isfinite(d)]
    out = {"estimate": round(float(obs), nd)}
    if len(ok) >= 100:
        lo, hi = np.percentile(ok, [2.5, 97.5])
        # Algebraic zero can differ by a few ulps across BLAS implementations.
        # Include numerical ties in both tails, without changing estimates or CIs.
        zero_tolerance = 1e-12
        le = int((ok <= zero_tolerance).sum())
        ge = int((ok >= -zero_tolerance).sum())
        p = min(1.0, 2.0 * min(le + 1, ge + 1) / (len(ok) + 1))
        out.update({"ci95": [round(float(lo), nd), round(float(hi), nd)],
                    "se_boot": round(float(ok.std()), nd),
                    "p_two_sided_boot": round(p, 5),
                    "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    else:
        out.update({"ci95": None, "se_boot": None, "p_two_sided_boot": None,
                    "ci_excludes_zero": None})
    n_bad = int(len(d) - len(ok))
    if n_bad:
        out["n_draws_undefined_excluded"] = n_bad
    if extra:
        out.update(extra)
    return out


def write_json(obj, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return p
