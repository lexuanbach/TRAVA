#!/usr/bin/env python3
"""Do the temporal contrasts survive matching on item composition?

RealTimeQA questions are not randomised across dates. Topic, answer form,
answer length and news source drift year to year, so a 2023-vs-2026 contrast
mixes cutoff position with item composition. The paper already shows one
symptom -- every generator, including a 0.5B model with nothing to recall, dips
at 2025 and recovers at 2026 -- but controls for it only by choosing a
non-adjacent year pair.

This controls for it directly, two ways:

  RESIDUALISE  centre each per-question score within its composition cell
               (answer type x answer-length bucket x question-length bucket),
               removing additive composition effects.
  MATCH        build a 1:1 matched sample, pairing each 2023 question with a
               2026 question of the same answer type and nearest answer and
               question length. No additivity assumption, at the cost of n.

Answer type is derived from the gold answer alone -- numeric, date, proper noun,
other -- so it exists for every item and is model-independent.

Two things this script is careful about, both raised in review.

  PAIRED INFERENCE. An earlier version built the 1:1 pairs and then handed the
  two arms to an unpaired bootstrap, which resamples them independently and
  throws away exactly the covariance the matching was built to exploit. The
  matched contrast is now resampled as PAIR DIFFERENCES, clustered on the 2023
  question's quiz, so the inference matches the design. The raw and residualised
  contrasts really are between different question sets and stay unpaired.

  MODEL-LEVEL UNCERTAINTY. A slope that keeps its sign across three estimators
  is weak evidence with seven rungs. Each controlled slope now carries its own
  exact permutation test over all assignments of the observed gaps to parameter
  counts, plus a leave-one-rung-out range.

What survives is a statement about the covariates we measured -- answer type,
answer length, question length. Topic, outlet, event volatility and answer
popularity are not measured here and are not controlled by this script.

Outputs results/matched_strata.json.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import token_f1                                   # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
PARAMS_B = {"qwen": 0.5, "qwen1_5b": 1.5, "qwen3b": 3.0, "qwen7b": 7.0,
            "qwen32b": 32.0, "qwen72b": 72.0, "mistral_large": 123.0,
            "llama3_8b": 8.0}
MONTHS = ("january february march april may june july august september "
          "october november december").split()
# the seven rungs of the main comparison ladder; llama3_8b is a cutoff-relative
# control from a different family and is not on it, so slopes over the eight
# available models are not comparable to the headline 5.9/decade.
MAIN_LADDER = ("qwen", "qwen1_5b", "qwen3b", "qwen7b", "qwen32b", "qwen72b",
               "mistral_large")


def quiz_of(qid):
    return qid.split("_")[0]


def answer_type(a: str) -> str:
    s = (a or "").strip().lower()
    if re.fullmatch(r"[^a-z]*\d[\d,.\s%$-]*", s or "x"):
        return "numeric"
    if any(m in s for m in MONTHS) or re.search(r"\b(19|20)\d{2}\b", s):
        return "date"
    ws = (a or "").split()[:2]
    if ws and all(w[:1].isupper() for w in ws if w[:1].isalpha()):
        return "proper_noun"
    return "other"


def covariates(q):
    return {"ans_len": len((q["gold_answer"] or "").split()),
            "q_len": len((q["question"] or "").split()),
            "ans_type": answer_type(q["gold_answer"])}


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in
             (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}


def load_scores(model, qs):
    f = PROC / f"generations_{model}" / "closed_book.jsonl"
    if not f.is_file():
        return None
    out = {}
    for line in f.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            q = qs.get(r["qid"])
            if q:
                out[r["qid"]] = 100.0 * token_f1(r["prediction"] or "",
                                                 q["gold_answer"])
    return out


def unpaired_boot(a, b, B, seed):
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean([a[rng.randrange(len(a))] for _ in a])
        - statistics.fmean([b[rng.randrange(len(b))] for _ in b])
        for _ in range(B))
    le = sum(1 for x in draws if x <= 0)
    return {"gap": round(statistics.fmean(a) - statistics.fmean(b), 3),
            "ci95": [round(draws[int(0.025 * B)], 3), round(draws[int(0.975 * B)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5)}


def paired_boot(pairs, sc, B, seed):
    """Resample MATCHED PAIRS, clustered on the 2023 question's quiz.

    The estimand is the mean within-pair difference. Resampling the two arms
    independently -- what this script used to do -- discards the matching and
    gives an interval for a design that was not run.
    """
    by = collections.defaultdict(list)
    for a, b in pairs:
        if a in sc and b in sc:
            by[quiz_of(a)].append(sc[a] - sc[b])
    clusters = [v for v in by.values() if v]
    if len(clusters) < 2:
        return None
    flat = [d for c in clusters for d in c]
    obs = statistics.fmean(flat)
    # Resample quizzes and POOL their pairs, so the reported gap is the centre
    # of its own interval. Reporting a pair-weighted point estimate beside a
    # quiz-weighted interval mixes two estimands.
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        pool = []
        for _ in clusters:
            pool.extend(clusters[rng.randrange(len(clusters))])
        if pool:
            draws.append(statistics.fmean(pool))
    if len(draws) < 100:
        return None
    draws.sort()
    n = len(draws)
    le = sum(1 for x in draws if x <= 0)
    return {"gap": round(obs, 3),
            "ci95": [round(draws[int(0.025 * n)], 3), round(draws[int(0.975 * n)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 5),
            "n_pairs": len(flat), "n_clusters": len(clusters), "paired": True}


def perm_slope(gaps):
    """Exact permutation test for the log10-parameter slope over the rungs.

    Enumerates every assignment of the observed gaps to the parameter counts.
    With <=8 rungs this is exhaustive, so there is no sampling error in p.
    """
    ms = sorted(gaps)
    if len(ms) < 3 or len(ms) > 9:
        return None
    x = [math.log10(PARAMS_B[m]) for m in ms]
    y = [gaps[m] for m in ms]
    mx = statistics.fmean(x)
    sxx = sum((a - mx) ** 2 for a in x)
    fit = lambda yy: sum((a - mx) * (b - statistics.fmean(yy))
                         for a, b in zip(x, yy)) / sxx
    obs = fit(y)
    ge = tot = 0
    for p in itertools.permutations(y):
        tot += 1
        if abs(fit(list(p))) >= abs(obs) - 1e-12:
            ge += 1
    loo = []                       # refit x as well as y when a rung is dropped
    for i in range(len(y)):
        xs = [v for j, v in enumerate(x) if j != i]
        ys = [v for j, v in enumerate(y) if j != i]
        m2 = statistics.fmean(xs)
        s2 = sum((a - m2) ** 2 for a in xs)
        loo.append(round(sum((a - m2) * (b - statistics.fmean(ys))
                             for a, b in zip(xs, ys)) / s2, 3))
    return {"slope": round(obs, 3), "p_exact": round(ge / tot, 5),
            "n_permutations": tot, "n_rungs": len(y),
            "leave_one_out_range": [min(loo), max(loo)],
            "rungs": ms}


def by_quiz(sel, sc):
    acc = collections.defaultdict(list)
    for qid in sel:
        if qid in sc:
            acc[quiz_of(qid)].append(sc[qid])
    return [statistics.fmean(v) for v in acc.values()]


def residualise(sc, qs, ids):
    cell = {q: (covariates(qs[q])["ans_type"],
                min(covariates(qs[q])["ans_len"], 6),
                min(covariates(qs[q])["q_len"] // 8, 5)) for q in ids}
    groups = collections.defaultdict(list)
    for q in ids:
        if q in sc:
            groups[cell[q]].append(sc[q])
    mu = {k: statistics.fmean(v) for k, v in groups.items()}
    have = [q for q in ids if q in sc]
    grand = statistics.fmean([sc[q] for q in have])
    return {q: sc[q] - mu[cell[q]] + grand for q in have}


def match_pairs(qs, pre_ids, post_ids, rng):
    pool = collections.defaultdict(list)
    for q in post_ids:
        pool[covariates(qs[q])["ans_type"]].append(q)
    for v in pool.values():
        rng.shuffle(v)
    used, pairs = set(), []
    for q in pre_ids:
        c = covariates(qs[q])
        cands = [x for x in pool.get(c["ans_type"], []) if x not in used]
        if not cands:
            continue
        best = min(cands, key=lambda x: (
            abs(covariates(qs[x])["ans_len"] - c["ans_len"]) * 3
            + abs(covariates(qs[x])["q_len"] - c["q_len"])))
        used.add(best)
        pairs.append((q, best))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--output", type=Path, default=RES / "matched_strata.json")
    args = ap.parse_args()

    qs = load_questions()
    pre = sorted(q for q, d in qs.items() if d["year"] == 2023)
    post = sorted(q for q, d in qs.items() if d["year"] == 2026)

    comp = {}
    for tag, ids in (("2023", pre), ("2026", post)):
        cs = [covariates(qs[q]) for q in ids]
        comp[tag] = {"n": len(ids),
                     "mean_answer_len": round(statistics.fmean(c["ans_len"] for c in cs), 2),
                     "mean_question_len": round(statistics.fmean(c["q_len"] for c in cs), 2),
                     "answer_type_share": {k: round(v / len(cs), 3) for k, v in
                                           collections.Counter(c["ans_type"] for c in cs).items()}}
    out = {"scope": "The conservative 2023-vs-2026 contrast under composition "
                    "controls: residualisation and 1:1 matching.",
           "composition": comp, "per_model": {}}
    print("composition drift 2023 -> 2026:")
    for k in ("mean_answer_len", "mean_question_len"):
        print(f"  {k:20s} {comp['2023'][k]:6.2f} -> {comp['2026'][k]:6.2f}")
    print(f"  types 2023 {comp['2023']['answer_type_share']}")
    print(f"  types 2026 {comp['2026']['answer_type_share']}")

    rng = random.Random(args.seed)
    pairs = match_pairs(qs, pre, post, rng)
    out["n_matched_pairs"] = len(pairs)
    print(f"\nmatched pairs: {len(pairs)}")
    print(f"\n{'model':15s}{'raw':>10s}{'residualised':>15s}{'matched':>12s}")
    raws, resids, matched = {}, {}, {}
    for m in sorted(PARAMS_B):
        sc = load_scores(m, qs)
        if not sc:
            continue
        raw = unpaired_boot(by_quiz(pre, sc), by_quiz(post, sc), args.B, args.seed)
        rs = residualise(sc, qs, pre + post)
        res = unpaired_boot(by_quiz(pre, rs), by_quiz(post, rs), args.B, args.seed)
        mp = [(a, b) for a, b in pairs if a in sc and b in sc]
        mt = paired_boot(mp, sc, args.B, args.seed) if len(mp) > 10 else None
        out["per_model"][m] = {"raw": raw, "residualised": res, "matched": mt,
                               "params_B": PARAMS_B[m]}
        if raw: raws[m] = raw["gap"]
        if res: resids[m] = res["gap"]
        if mt: matched[m] = mt["gap"]
        f = lambda r: f"{r['gap']:+.2f}" if r else "-"
        print(f"{m:15s}{f(raw):>10s}{f(res):>15s}{f(mt):>12s}")

    def slope(d):
        pts = [(math.log10(PARAMS_B[m]), v) for m, v in d.items()]
        if len(pts) < 3:
            return None
        x = [p[0] for p in pts]; y = [p[1] for p in pts]
        mx, my = statistics.fmean(x), statistics.fmean(y)
        sxx = sum((a - mx) ** 2 for a in x)
        return round(sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx, 4)
    out["scale_slope"] = {"raw": slope(raws), "residualised": slope(resids),
                          "matched": slope(matched)}
    # model-level uncertainty for each CONTROLLED slope, not just its sign
    out["scale_slope_inference"] = {
        k: perm_slope(d) for k, d in
        (("raw", raws), ("residualised", resids), ("matched", matched))}
    # restricted to the seven main-ladder rungs, so these are directly
    # comparable to the shared-boundary headline slope
    out["scale_slope_main_ladder"] = {
        k: perm_slope({m: v for m, v in d.items() if m in MAIN_LADDER})
        for k, d in (("raw", raws), ("residualised", resids),
                     ("matched", matched))}
    signs = [v for v in out["scale_slope"].values() if v is not None]
    ok = bool(signs) and all(v > 0 for v in signs)
    # judge significance on the SEVEN main-ladder rungs, the estimand that is
    # comparable to the headline slope. On eight models (adding an off-ladder
    # 8B control) every controlled slope clears 0.05; on the seven comparable
    # ones only the matched slope does, and the other two sit just above it.
    sig = {k: (v["p_exact"] < 0.05)
           for k, v in out["scale_slope_main_ladder"].items() if v}
    out["verdict"] = {
        "sign_stable_under_controls": ok,
        "significant_at_model_level": sig,
        "controlled_covariates": ["answer type", "answer length",
                                  "question length"],
        "uncontrolled": ["topic", "outlet", "event volatility", "ambiguity",
                         "answer popularity"],
        "significance_basis": ("seven main-ladder rungs; the eight-model figures "
                               "add an off-ladder 8B control and are not "
                               "comparable to the headline slope"),
        "reading": ("The measured form covariates do not explain the contrast: "
                    "the per-model gaps and the scale slope keep their sign "
                    "under residualisation and under 1:1 matching. This is NOT "
                    "the claim that item composition does not produce the "
                    "contrast -- topic, outlet, volatility and answer popularity "
                    "are not measured here. Matching costs sample size, and the "
                    "matched interval is a paired one clustered on the 2023 "
                    "quiz. At model level on the seven comparable rungs only "
                    "the matched slope clears p<0.05; raw and residualised sit "
                    "just above it, so the controlled evidence is borderline, "
                    "not confirmatory."),
    }
    print(f"\n{'estimator':>14s}{'n':>3s}{'slope':>9s}{'p_exact':>10s}"
          f"{'LOO range':>18s}   (all available models)")
    for k in ("raw", "residualised", "matched"):
        r = out["scale_slope_inference"].get(k)
        if r:
            print(f"{k:>14s}{r['n_rungs']:3d}{r['slope']:9.2f}{r['p_exact']:10.4f}"
                  f"{str(r['leave_one_out_range']):>18s}")
    print(f"{'estimator':>14s}{'n':>3s}{'slope':>9s}{'p_exact':>10s}"
          f"{'LOO range':>18s}   (main ladder only)")
    for k in ("raw", "residualised", "matched"):
        r = out["scale_slope_main_ladder"].get(k)
        if r:
            print(f"{k:>14s}{r['n_rungs']:3d}{r['slope']:9.2f}{r['p_exact']:10.4f}"
                  f"{str(r['leave_one_out_range']):>18s}")
    print(f"sign stable under controls: {ok}; "
          f"significant at model level: {sig}")
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
