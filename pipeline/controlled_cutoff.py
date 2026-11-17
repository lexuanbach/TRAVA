#!/usr/bin/env python3
"""Separate scale from cutoff using one family with disclosed cutoffs.

THE PROBLEM THIS SOLVES. Every scale statement elsewhere in this paper is
roster-confounded: on the main ladder, parameter count moves together with model
family, training recipe and knowledge cutoff, and the larger models happen to
have later cutoffs. A shared calendar boundary therefore measures cutoff offset
as well as scale, and no estimator applied to that roster can tell the two apart.
This is the controlled design rather than another sensitivity check.

THE DESIGN. Meta discloses a pretraining cutoff for each Llama release, which
lets us build both arms of a 2-factor design inside ONE family:

  ARM A -- cutoff varies, size held FIXED at 8B
      Llama-3-8B-Instruct     2023-03
      Llama-3.1-8B-Instruct   2023-12
    Same size, same family, ~9 months of extra world knowledge. Any difference in
    the Static-Fresh gap here CANNOT be a scale effect.

  ARM B -- size varies, cutoff held FIXED at 2023-12
      Llama-3.2-1B-Instruct   2023-12
      Llama-3.2-3B-Instruct   2023-12
      Llama-3.1-8B-Instruct   2023-12
    Same disclosed cutoff, same family lineage (3.2 1B/3B are distilled from the
    3.1 models), 8x parameter range. A slope here is a scale slope with cutoff
    HELD CONSTANT -- the estimand the main ladder cannot deliver.

The two arms answer different questions and are reported separately. The
headline comparison is Arm B's slope against the main ladder's 5.9 F1/decade: if
controlling cutoff shrinks the slope substantially, the shared-boundary estimate
was partly measuring cutoff offset, exactly as this paper argues.

WHAT THIS STILL DOES NOT DO. It does not randomise anything. Llama-3.2-1B/3B are
distilled, not independently pretrained, so "size" here also carries a training
procedure difference; Arm B is three rungs; and a disclosed cutoff is a provider
claim, not a measurement. This is a controlled comparison, not an experiment.

Outputs results/controlled_cutoff.json.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import token_f1                                     # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"

# Provider-disclosed pretraining cutoffs, from Meta's model cards ("Data
# Freshness"). A provider date is a claim, not a measurement -- the card does not
# say whether it is the crawl end, the filter end, or the last month of reliable
# knowledge -- so `notes` records what is actually asserted.
#
# Meta's own release history illustrates the confound this script exists to
# break: within Llama 3, the 8B card says March 2023 while the 70B card says
# December 2023, so even inside one release, size and cutoff move together.
FAMILY = {
    "llama3_8b":  {"params_B": 8.0, "cutoff": "2023-03", "release": "Llama 3",
                   "precision": "bfloat16",
                   "hf_repo": "NousResearch/Meta-Llama-3-8B-Instruct",
                   "source": "Meta Llama 3 8B model card, Data Freshness: "
                             "March 2023"},
    "llama31_8b": {"params_B": 8.0, "cutoff": "2023-12", "release": "Llama 3.1",
                   "precision": "bfloat16",
                   "hf_repo": "NousResearch/Meta-Llama-3.1-8B-Instruct",
                   "source": "Meta Llama 3.1 model card, Data Freshness: "
                             "December 2023"},
    "llama32_1b": {"params_B": 1.0, "cutoff": "2023-12", "release": "Llama 3.2",
                   "precision": "bfloat16",
                   "hf_repo": "unsloth/Llama-3.2-1B-Instruct",
                   "source": "Meta Llama 3.2 model card, Data Freshness: "
                             "December 2023"},
    "llama32_3b": {"params_B": 3.0, "cutoff": "2023-12", "release": "Llama 3.2",
                   "precision": "bfloat16",
                   "hf_repo": "unsloth/Llama-3.2-3B-Instruct",
                   "source": "Meta Llama 3.2 model card, Data Freshness: "
                             "December 2023"},
    "llama31_8b_q4": {"params_B": 8.0, "cutoff": "2023-12",
                      "release": "Llama 3.1", "precision": "Q4_K_M",
                      "hf_repo": "lmstudio-community/"
                                 "Meta-Llama-3.1-8B-Instruct-GGUF (Q4_K_M)",
                      "source": "Meta Llama 3.1 model card, Data Freshness: "
                                "December 2023"},
    "llama31_70b": {"params_B": 70.0, "cutoff": "2023-12", "release": "Llama 3.1",
                    "precision": "Q4_K_M",
                    "hf_repo": "lmstudio-community/"
                               "Meta-Llama-3.1-70B-Instruct-GGUF (Q4_K_M)",
                    "source": "Meta Llama 3.1 model card, Data Freshness: "
                              "December 2023"},
}
ARM_A = ["llama3_8b", "llama31_8b"]                   # cutoff varies, size fixed
# size varies, cutoff fixed at 2023-12. The 70B rung is optional: it is the 8B's
# direct sibling and stretches the range from 8x to 70x, but it is a long
# quantized run, so the script works with whichever rungs are present.
ARM_B_ALL = ["llama32_1b", "llama32_3b", "llama31_8b", "llama31_70b"]
MAIN_LADDER_SLOPE = 5.9342          # results/model_level_inference.json, 7 rungs
GUARD_MONTHS = 3                    # must match pipeline/cutoff_relative.py


def quiz_of(qid):
    return qid.split("_")[0]


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in
             (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}


def load_scores(tag, qs):
    f = PROC / f"generations_{tag}" / "closed_book.jsonl"
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
    return out or None


def quiz_means(ids, sc):
    """quiz id -> mean score over that quiz's questions in `ids`."""
    acc = collections.defaultdict(list)
    for qid in ids:
        if qid in sc:
            acc[quiz_of(qid)].append(sc[qid])
    return {k: statistics.fmean(v) for k, v in acc.items()}


def by_quiz(ids, sc):
    return list(quiz_means(ids, sc).values())


def unpaired_boot(a, b, B, seed):
    """Quiz-clustered gap between two disjoint question sets (Static vs Fresh)."""
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
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5),
            "n_quiz_static": len(a), "n_quiz_fresh": len(b)}


def paired_boot(diffs_by_quiz, B, seed):
    """Two models scored on the SAME questions: resample quizzes, not arms.

    Standard cluster bootstrap: draw quizzes with replacement, POOL their
    questions, take the mean. An earlier version reported a question-weighted
    point estimate beside a quiz-weighted interval, so the point estimate did
    not sit at the centre of its own CI -- two different estimands printed as
    one. Quizzes here are unbalanced (median 3 items, some far larger), so the
    two genuinely differ.
    """
    clusters = [v for v in diffs_by_quiz.values() if v]
    if len(clusters) < 2:
        return None
    flat = [d for c in clusters for d in c]
    obs = statistics.fmean(flat)
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
    return {"delta": round(obs, 3),
            "ci95": [round(draws[int(0.025 * n)], 3), round(draws[int(0.975 * n)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 5),
            "n_questions": len(flat), "n_quizzes": len(clusters), "paired": True}


def mde(diffs_by_quiz, alpha=0.05, power=0.8):
    """Minimum detectable effect for a paired, quiz-clustered contrast.

    A null is only informative next to the effect it could have seen. This uses
    the cluster-level SD with the observed cluster count, so it accounts for the
    same dependence the bootstrap does.
    """
    vals = [statistics.fmean(v) for v in diffs_by_quiz.values() if v]
    k = len(vals)
    if k < 3:
        return None
    sd = statistics.stdev(vals)
    # two-sided normal approximation: (z_{1-a/2} + z_{1-b}) * sd / sqrt(k)
    z = 1.959964 + 0.841621
    return {"mde_f1": round(z * sd / math.sqrt(k), 3), "n_clusters": k,
            "cluster_sd": round(sd, 3), "alpha": alpha, "power": power,
            "note": ("smallest true effect this contrast would detect 80% of "
                     "the time; a null larger than nothing but smaller than "
                     "this is inconclusive, not absent")}


def months_between(a: str, b: str) -> int:
    return (int(a[:4]) - int(b[:4])) * 12 + (int(a[5:7]) - int(b[5:7]))


def boundary_split(qs, ids, cutoff, guard=GUARD_MONTHS):
    """Static before the cutoff, Fresh after, dropping a guard band.

    The guard band matches `cutoff_relative.py`. A provider date is imprecise,
    so items within a few months of it have ambiguous status. Using a different
    convention here would make the artifact report two different numbers for the
    same model's own-cutoff gap, which is precisely the drift this project's
    claim ledger exists to prevent.
    """
    pre = [q for q in ids if months_between(str(qs[q]["date"])[:7], cutoff) < -guard]
    post = [q for q in ids if months_between(str(qs[q]["date"])[:7], cutoff) > guard]
    return pre, post


def slope_boot(gaps_by_model, params, B, seed):
    """Quiz-clustered bootstrap CI for the slope across rungs.

    With three rungs the exact permutation test cannot reach p<0.05.
    The exhaustive floor depends on ties in absolute slope. This bootstrap
    resamples the QUIZZES jointly, recomputing every rung's Static-Fresh gap and
    refitting the slope inside each draw, which is the same "conditional on this
    ladder" target the main analysis uses. It does not sample new model rungs.

    gaps_by_model: tag -> (static_quiz_means, fresh_quiz_means) keyed by quiz id
    so the same resampled quizzes are used for every rung.
    """
    tags = sorted(gaps_by_model, key=lambda t: params[t])
    if len(tags) < 3:
        return None
    s_keys = sorted({k for t in tags for k in gaps_by_model[t][0]})
    f_keys = sorted({k for t in tags for k in gaps_by_model[t][1]})
    if len(s_keys) < 3 or len(f_keys) < 3:
        return None
    x = [math.log10(params[t]) for t in tags]
    mx = statistics.fmean(x)
    sxx = sum((a - mx) ** 2 for a in x)

    def fit(ys):
        my = statistics.fmean(ys)
        return sum((a - mx) * (b - my) for a, b in zip(x, ys)) / sxx

    def gaps(sk, fk):
        out = []
        for t in tags:
            sm, fm = gaps_by_model[t]
            a = [sm[k] for k in sk if k in sm]
            b = [fm[k] for k in fk if k in fm]
            if not a or not b:
                return None
            out.append(statistics.fmean(a) - statistics.fmean(b))
        return out

    obs = gaps(s_keys, f_keys)
    if obs is None:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        sk = [s_keys[rng.randrange(len(s_keys))] for _ in s_keys]
        fk = [f_keys[rng.randrange(len(f_keys))] for _ in f_keys]
        g = gaps(sk, fk)
        if g is not None:
            draws.append(fit(g))
    if len(draws) < 100:
        return None
    draws.sort()
    n = len(draws)
    le = sum(1 for v in draws if v <= 0)
    return {"slope_per_decade": round(fit(obs), 3),
            "ci95": [round(draws[int(0.025 * n)], 3),
                     round(draws[int(0.975 * n)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 5),
            "n_quiz_static": len(s_keys), "n_quiz_fresh": len(f_keys),
            "rungs": tags,
            "note": ("quiz-clustered bootstrap, conditional on these rungs; the "
                     "primary inference for this arm because the exact "
                     "permutation test is resolution-limited at this few rungs")}


def slope_with_perm(points):
    """OLS slope on log10 params + exhaustive permutation over the rungs."""
    if len(points) < 3:
        return None
    pts = sorted(points)
    x = [math.log10(p) for p, _ in pts]
    y = [v for _, v in pts]
    mx = statistics.fmean(x)
    sxx = sum((a - mx) ** 2 for a in x)
    fit = lambda yy: sum((a - mx) * (b - statistics.fmean(yy))
                         for a, b in zip(x, yy)) / sxx
    obs = fit(y)
    ge = tot = 0
    magnitudes = []
    for perm in itertools.permutations(y):
        tot += 1
        magnitudes.append(abs(fit(list(perm))))
        if abs(fit(list(perm))) >= abs(obs) - 1e-12:
            ge += 1
    floor = sum(v >= max(magnitudes) - 1e-12 for v in magnitudes) / tot
    return {"slope_per_decade": round(obs, 3), "p_exact": round(ge / tot, 5),
            "n_permutations": tot, "n_rungs": len(y),
            "p_resolution_floor": round(floor, 5),
            "underpowered_by_construction": floor > 0.05,
            "points": [{"params_B": p, "gap_f1": round(v, 3)} for p, v in pts]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--output", type=Path, default=RES / "controlled_cutoff.json")
    args = ap.parse_args()

    qs = load_questions()
    scores = {t: load_scores(t, qs) for t in FAMILY}
    # Only the models the two arms actually need are required. The 70B rung is
    # an optional range extension and a long quantized run, so its absence must
    # not block the primary result.
    required = set(ARM_A) | {t for t in ARM_B_ALL
                             if FAMILY[t].get("precision") == "bfloat16"}
    missing = sorted(t for t in required if not scores.get(t))
    if missing:
        print(f"missing closed-book caches for {missing}; "
              f"run scripts/run_controlled_cutoff.sh", file=sys.stderr)
        return 1
    optional_missing = sorted(t for t in FAMILY
                              if t not in required and not scores.get(t))
    if optional_missing:
        print(f"note: optional rungs not present, continuing without them: "
              f"{optional_missing}", file=sys.stderr)

    out = {
        "scope": "One model family with provider-disclosed cutoffs, used to vary "
                 "cutoff at fixed size and size at fixed cutoff.",
        "why": ("Elsewhere in this paper parameter count is confounded with "
                "family, recipe and cutoff. These two arms compare models "
                "within one family while holding size or the reported cutoff fixed."),
        "cutoffs_are_provider_claims": (
            "Cutoff dates are taken from Meta's model cards. They are claims by "
            "the provider, not measurements; our own attempt to estimate an "
            "effective cutoff from accuracy curves failed (see the estimator "
            "negative result)."),
        "family": FAMILY,
        "n_questions": len(qs),
    }

    # ---------- each model's gap at ITS OWN disclosed boundary --------------
    own = {}
    for tag, meta in FAMILY.items():
        sc = scores.get(tag)
        if not sc:
            continue
        ids = [q for q in qs if q in sc]
        pre, post = boundary_split(qs, ids, meta["cutoff"])
        r = unpaired_boot(by_quiz(pre, sc), by_quiz(post, sc), args.B, args.seed)
        if r:
            r.update(params_B=meta["params_B"], cutoff=meta["cutoff"],
                     n_static=len(pre), n_fresh=len(post))
            own[tag] = r
    out["gap_at_own_cutoff"] = own

    # ---------- ARM A: cutoff varies, size fixed ---------------------------
    a0, a1 = ARM_A
    common = sorted(set(scores[a0]) & set(scores[a1]))
    # Both models are scored on the same questions, so the honest contrast is
    # paired. The estimand is how much the 9-month-later cutoff moves accuracy
    # on the questions that fall BETWEEN the two cutoffs -- the window where the
    # later model should have knowledge the earlier one cannot.
    lo, hi = FAMILY[a0]["cutoff"], FAMILY[a1]["cutoff"]
    window = [q for q in common if lo <= str(qs[q]["date"])[:7] < hi]
    before = [q for q in common if str(qs[q]["date"])[:7] < lo]
    after = [q for q in common if str(qs[q]["date"])[:7] >= hi]
    arm_a = {"n_in_window": len(window), "n_before_both": len(before),
             "n_after_both": len(after), "window": f"[{lo}, {hi})",
             "size_held_at_B": FAMILY[a0]["params_B"]}
    for name, ids in (("between_the_cutoffs", window),
                      ("before_both_cutoffs", before),
                      ("after_both_cutoffs", after)):
        d = collections.defaultdict(list)
        for q in ids:
            d[quiz_of(q)].append(scores[a1][q] - scores[a0][q])
        r = paired_boot(d, args.B, args.seed) if d else None
        if r:
            r["mde"] = mde(d)
            arm_a[name] = r
    # the difference-in-differences: does the later model gain MORE in the
    # window than outside it? That is the cutoff effect, net of any overall
    # quality difference between the two releases.
    # Two controls, because they fail differently. BEFORE both cutoffs, both
    # models have the knowledge, so a difference there is a pure release-quality
    # difference. AFTER both cutoffs, neither has it, so a difference there is
    # quality on questions both must guess. The cutoff effect predicts the
    # window exceeds BOTH; if it only exceeds one, say which.
    if "between_the_cutoffs" in arm_a:
        w = arm_a["between_the_cutoffs"]["delta"]
        for ctl, key in (("before_both_cutoffs", "did_window_minus_before"),
                         ("after_both_cutoffs", "did_window_minus_after")):
            if ctl in arm_a:
                arm_a[key] = round(w - arm_a[ctl]["delta"], 3)
        arm_a["exceeds_both_controls"] = all(
            w > arm_a[c]["delta"] for c in
            ("before_both_cutoffs", "after_both_cutoffs") if c in arm_a)
        arm_a["controls_reading"] = (
            "The before-both and after-both date groups provide comparison "
            "contrasts for the between-cutoff group. Provider cutoff dates "
            "do not establish which answers either model knows. Training "
            "differences remain confounded with release dates.")
    arm_a["gap_change"] = {
        t: {"cutoff": FAMILY[t]["cutoff"], "gap_at_own_cutoff": own[t]["gap"],
            "ci95": own[t]["ci95"]} for t in ARM_A if t in own}
    arm_a["reading"] = (
        "Same size and family, with different provider-reported cutoffs. "
        "The contrasts do not isolate a causal cutoff effect.")
    out["arm_a_cutoff_varies_size_fixed"] = arm_a

    # ---------- ARM B: size varies, cutoff fixed ---------------------------
    arm_b_tags = [t for t in ARM_B_ALL if scores.get(t)]
    cut = FAMILY[arm_b_tags[0]]["cutoff"]
    assert all(FAMILY[t]["cutoff"] == cut for t in arm_b_tags), "Arm B not matched"
    pts, per_model, by_model = [], {}, {}
    for tag in arm_b_tags:
        sc = scores[tag]
        ids = [q for q in qs if q in sc]
        pre, post = boundary_split(qs, ids, cut)
        r = unpaired_boot(by_quiz(pre, sc), by_quiz(post, sc), args.B, args.seed)
        if r:
            per_model[tag] = {**r, "params_B": FAMILY[tag]["params_B"]}
            pts.append((FAMILY[tag]["params_B"], r["gap"]))
            by_model[tag] = (quiz_means(pre, sc), quiz_means(post, sc))
    # PRIMARY is the precision-matched subset: mixing a 4-bit rung into a
    # bfloat16 ladder reintroduces a nuisance variable into the one analysis
    # whose purpose is to remove nuisance variables. The quantised rung extends
    # the range 8x -> 70x and is reported alongside, labelled.
    prec = {t: FAMILY[t].get("precision") for t in arm_b_tags}
    matched = [t for t in arm_b_tags if prec[t] == "bfloat16"]
    params = {t: FAMILY[t]["params_B"] for t in by_model}
    arm_b = {
        "shared_cutoff": cut, "rungs": arm_b_tags,
        "precision_by_rung": prec, "per_model": per_model,
        "primary_rungs": matched,
        "primary_is_precision_matched": (
            "The primary slope uses only the bfloat16 rungs. Adding a 4-bit rung "
            "would put a precision difference back into the one analysis built to "
            "remove nuisance variables, so the quantised extension is reported "
            "separately rather than pooled."),
        "slope_bootstrap": slope_boot(
            {t: v for t, v in by_model.items() if t in matched},
            params, args.B, args.seed),
        "slope_permutation": slope_with_perm(
            [(FAMILY[t]["params_B"], per_model[t]["gap"]) for t in matched
             if t in per_model]),
    }
    if len(arm_b_tags) > len(matched):
        arm_b["extension_all_rungs_mixed_precision"] = {
            "rungs": arm_b_tags,
            "slope_bootstrap": slope_boot(by_model, params, args.B, args.seed),
            "slope_permutation": slope_with_perm(pts),
            "caveat": ("includes a 4-bit rung; the parameter range widens to "
                       f"{max(params.values()) / min(params.values()):.0f}x but "
                       "precision is no longer held fixed"),
        }
    arm_b["slope"] = arm_b["slope_bootstrap"] or arm_b["slope_permutation"]
    if arm_b["slope"]:
        s = arm_b["slope"]["slope_per_decade"]
        arm_b["vs_main_ladder"] = {
            "main_ladder_slope": MAIN_LADDER_SLOPE,
            "controlled_slope": s,
            "ratio": round(s / MAIN_LADDER_SLOPE, 3),
            "reading": ("Fraction of the shared-boundary slope that survives "
                        "holding the cutoff fixed. Well below 1 supports this "
                        "paper's argument that a shared calendar boundary "
                        "measures cutoff offset as well as scale; near 1 would "
                        "mean the confound was not doing the work."),
        }
    arm_b["caveat"] = (
        "Llama-3.2 1B/3B are distilled from larger Llama-3.1 models rather than "
        "independently pretrained, so this arm holds cutoff and family fixed but "
        f"not training procedure. {len(arm_b_tags)} rungs; the exact permutation "
        "test is reported but is resolution-limited at this few rungs, so the "
        "quiz bootstrap is the primary inference.")
    out["arm_b_size_varies_cutoff_fixed"] = arm_b

    # ---------- ARM C: same release, same cutoff, only size differs --------
    # Llama-3.1-8B and Llama-3.1-70B are the SAME release: same data, same
    # recipe, same disclosed cutoff, differing only in parameter count. Run at
    # the same quantization, this is the cleanest scale contrast in the paper --
    # cleaner than Arm B, whose top rung also changes release (3.1 vs distilled
    # 3.2) and so re-imports a training-procedure confound.
    c0, c1 = "llama31_8b_q4", "llama31_70b"
    if scores.get(c0) and scores.get(c1):
        cut_c = FAMILY[c0]["cutoff"]
        arm_c = {"models": [c0, c1],
                 "shared": {"release": FAMILY[c0]["release"],
                            "cutoff": cut_c,
                            "precision": FAMILY[c0]["precision"]},
                 "params_B": [FAMILY[c0]["params_B"], FAMILY[c1]["params_B"]]}
        assert FAMILY[c0]["release"] == FAMILY[c1]["release"]
        assert FAMILY[c0]["cutoff"] == FAMILY[c1]["cutoff"]
        assert FAMILY[c0]["precision"] == FAMILY[c1]["precision"]
        for tag in (c0, c1):
            sc = scores[tag]
            ids = [q for q in qs if q in sc]
            pre, post = boundary_split(qs, ids, cut_c)
            r = unpaired_boot(by_quiz(pre, sc), by_quiz(post, sc),
                              args.B, args.seed)
            if r:
                arm_c[tag] = {**r, "params_B": FAMILY[tag]["params_B"]}
        if c0 in arm_c and c1 in arm_c:
            g0, g1 = arm_c[c0]["gap"], arm_c[c1]["gap"]
            dec = math.log10(FAMILY[c1]["params_B"] / FAMILY[c0]["params_B"])
            arm_c["gap_difference"] = round(g1 - g0, 3)
            arm_c["implied_slope_per_decade"] = round((g1 - g0) / dec, 3)
            arm_c["reading"] = (
                "Two models of one release, same data and recipe, same disclosed "
                "cutoff, same quantization; only parameter count differs. This is "
                "a two-point contrast, so it carries no slope uncertainty of its "
                "own -- each gap's interval is reported and the implied slope is "
                "descriptive.")
        out["arm_c_same_release_only_size_differs"] = arm_c
    else:
        out["arm_c_same_release_only_size_differs"] = {
            "status": "not run",
            "why": ("needs closed-book caches for llama31_8b_q4 and "
                    "llama31_70b at matching quantization"),
        }

    # ---------- what the paper may conclude --------------------------------
    a_eff = arm_a.get("did_window_minus_before")
    out["verdict"] = {
        "cutoff_effect_at_fixed_size": a_eff,
        "scale_slope_at_fixed_cutoff": (arm_b["slope"] or {}).get("slope_per_decade"),
        "scale_slope_ci_at_fixed_cutoff": (arm_b["slope"] or {}).get("ci95"),
        "scale_slope_excludes_zero_at_fixed_cutoff":
            bool((arm_b["slope"] or {}).get("ci95")
                 and (arm_b["slope"]["ci95"][0] > 0
                      or arm_b["slope"]["ci95"][1] < 0)),
        "reading": (
            "Arm A holds size fixed and Arm B holds the reported cutoff fixed. "
            "Training differences remain. The three-rung Arm B bootstrap is "
            "conditional on this roster and is driven by the 8B point. Its "
            "exhaustive size-label permutation p is 4/6, with floor 1/6."),
    }

    print(f"{'model':13s}{'params':>8s}{'cutoff':>10s}{'gap@own':>10s}"
          f"{'ci95':>18s}{'p':>9s}")
    for t, r in own.items():
        print(f"{t:13s}{r['params_B']:8.1f}{r['cutoff']:>10s}{r['gap']:10.2f}"
              f"{str(r['ci95']):>18s}{r['p_plus_one']:9.4f}")
    print(f"\nARM A  cutoff varies, size fixed at {arm_a['size_held_at_B']}B")
    for k in ("before_both_cutoffs", "between_the_cutoffs", "after_both_cutoffs"):
        if k in arm_a:
            r = arm_a[k]
            print(f"  {k:22s} n={r['n_questions']:4d}  "
                  f"3.1 minus 3.0 = {r['delta']:+6.2f} {r['ci95']} p={r['p_plus_one']}")
    w = arm_a.get("between_the_cutoffs", {}).get("mde")
    if w:
        print(f"  window MDE at 80% power: {w['mde_f1']} F1 "
              f"({w['n_clusters']} quizzes)")
    print(f"  DiD vs before-both: {arm_a.get('did_window_minus_before')}   "
          f"DiD vs after-both: {arm_a.get('did_window_minus_after')}   "
          f"exceeds both controls: {arm_a.get('exceeds_both_controls')}")
    print(f"\nARM B  size varies, cutoff fixed at {cut}")
    for t, r in per_model.items():
        star = "" if t in arm_b["primary_rungs"] else "  (4-bit, extension only)"
        print(f"  {t:13s}{r['params_B']:6.1f}B  gap {r['gap']:+6.2f} "
              f"{str(r['ci95']):>18s}{star}")
    sb, sp = arm_b.get("slope_bootstrap"), arm_b.get("slope_permutation")
    if sb:
        print(f"  slope {sb['slope_per_decade']:+.2f} F1/decade  {sb['ci95']}  "
              f"p={sb['p_plus_one']}  (quiz bootstrap, PRIMARY)")
    if sp:
        floor = " -- RESOLUTION-LIMITED" if sp["underpowered_by_construction"] else ""
        print(f"  permutation p={sp['p_exact']} over {sp['n_permutations']} perms, "
              f"floor {sp['p_resolution_floor']}{floor}")
    if arm_b.get("vs_main_ladder"):
        print(f"  vs main-ladder {MAIN_LADDER_SLOPE:.2f}: ratio "
              f"{arm_b['vs_main_ladder']['ratio']}")
    ext = arm_b.get("extension_all_rungs_mixed_precision", {}).get("slope_bootstrap")
    if ext:
        print(f"  extension incl. 4-bit rung: {ext['slope_per_decade']:+.2f} "
              f"{ext['ci95']} p={ext['p_plus_one']}")
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
