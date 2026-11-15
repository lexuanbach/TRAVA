#!/usr/bin/env python3
"""Scoring primitives: SQuAD-style EM/F1, citation precision, decay slope, CIs.

Pure functions over REAL predictions; no numbers are invented anywhere.
"""
import re
import string
from collections import Counter

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def gold_in_text(gold: str, text: str) -> bool:
    return normalize(gold) in normalize(text)


def p_cite(gold: str, doc_texts) -> float:
    """Citation precision proxy: fraction of in-context docs that are
    answer-bearing (contain the gold answer span)."""
    if not doc_texts:
        return 0.0
    return sum(gold_in_text(gold, t) for t in doc_texts) / len(doc_texts)


def bootstrap_ci(values, n=1000, ci=0.95, seed=0):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_pvalue(a, b, n=1000, seed=0):
    """One-sided p-value that mean(a) > mean(b) via paired bootstrap (same items)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float((boot <= 0).mean())


def two_sample_bootstrap(a, b, n=1000, seed=0):
    """UNPAIRED comparison of two disjoint groups (e.g. Static vs Fresh items).

    Returns (diff=mean(a)-mean(b), ci95 of the diff, one_sided_p that mean(a)>mean(b)).
    Resamples each group independently, so it is valid for non-overlapping sets.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ba = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    bb = rng.choice(b, size=(n, len(b)), replace=True).mean(axis=1)
    diff = ba - bb
    lo, hi = np.quantile(diff, [0.025, 0.975])
    p = float((diff <= 0).mean())  # one-sided: P(mean(a) <= mean(b))
    return float(a.mean() - b.mean()), [float(lo), float(hi)], p


_ABSTAIN = re.compile(
    r"do(?:es)?\s+not\s+(?:contain|specify|provide|mention|include|have)"
    r"|don'?t\s+have|do\s+not\s+have|cannot\s+(?:answer|determine|provide|be)"
    r"|no\s+(?:information|evidence|mention|specific)"
    r"|not\s+(?:available|specified|provided|contain|enough|mentioned)"
    r"|unable\s+to|i\s+do\s+not\s+have|not\s+enough\s+(?:information|context)"
    r"|is\s+in\s+the\s+future|knowledge\s+(?:cutoff|was\s+last)",
    re.I)


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN.search(text or ""))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(1, na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def ols_slope(ages, scores):
    """OLS slope of score vs age (years). Returns (slope, intercept, r2)."""
    x, y = np.asarray(ages, float), np.asarray(scores, float)
    if len(x) < 3 or x.std() == 0:
        return 0.0, float(y.mean()) if len(y) else 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    (m, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = m * x + c
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(m), float(c), float(r2)
