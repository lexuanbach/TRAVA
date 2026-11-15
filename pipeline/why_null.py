#!/usr/bin/env python3
"""Why is the 32B cliff null? Distinguish 'no effect' from 'cannot see the effect'.

Candidate causes, each tested here:
  H1 POWER      878 of 4,927 available questions were sampled. What is the MDE, and
                what n would we need for the observed effect?
  H2 VARIANCE   The Static/Fresh comparison is unpaired across disjoint question
                sets, so ALL question-level difficulty variance sits in the error
                term. Removing it with a per-question difficulty covariate (a model
                that cannot know the answer) should shrink the SE a lot.
  H3 DILUTION   The shared 2025 boundary puts 2024 in Static. If Qwen2.5's cutoff is
                ~2023-2024, part of "Static" is actually post-cutoff, which drags
                Static down and shrinks the measured cliff.
  H4 CEILING    Maybe the 32B simply has no parametric knowledge of this news, in
                which case there is nothing to lose and the null is the finding.

Writes results/why_null.json.
"""
import json
import math
from pathlib import Path

import numpy as np

from metrics import token_f1

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

TAGS = {"qwen": "generations_qwen32b", }
GEN = {"qwen": "generations_qwen", "qwen1_5b": "generations_qwen1_5b",
       "qwen3b": "generations_qwen3b", "qwen7b": "generations_qwen7b",
       "qwen32b": "generations_qwen32b", "qwen72b": "generations_qwen72b",
       "mistral_large": "generations_mistral_large", "haiku": "generations",
       "sonnet": "generations_sonnet"}


def rng(label):
    import zlib
    return np.random.default_rng(zlib.crc32(label.encode()) & 0xFFFFFFFF)


def load(tag, cond, Q):
    f = PROC / GEN[tag] / f"{cond}.jsonl"
    if not f.exists():
        return None
    return {r["qid"]: token_f1(r["prediction"], Q[r["qid"]]["gold_answer"]) * 100
            for r in (json.loads(l) for l in f.read_text().splitlines())
            if r["qid"] in Q}


def unpaired(a, b, label, n=4000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = rng(label)
    d = a.mean() - b.mean()
    ds = np.array([r.choice(a, len(a)).mean() - r.choice(b, len(b)).mean() for _ in range(n)])
    return {"diff": round(float(d), 2),
            "se": round(float(ds.std()), 2),
            "ci95": [round(float(np.percentile(ds, 2.5)), 2),
                     round(float(np.percentile(ds, 97.5)), 2)],
            "one_sided_p": round(float((ds <= 0).mean()), 4),
            "n_a": len(a), "n_b": len(b)}


def mde(sd_a, sd_b, n_a, n_b, alpha=0.05, power=0.80):
    """Two-sided MDE for a difference of means."""
    from statistics import NormalDist
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    se = math.sqrt(sd_a ** 2 / n_a + sd_b ** 2 / n_b)
    return (z_a + z_b) * se


def main():
    Q = {q["qid"]: q for q in
         (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines())}
    res = {}
    S = [q for q in Q if Q[q]["year"] < 2025]
    F = [q for q in Q if Q[q]["year"] >= 2025]

    cb = {t: load(t, "closed_book", Q) for t in GEN}
    cb = {t: v for t, v in cb.items() if v}

    # ---------------- H1: power ------------------------------------------------
    x = cb["qwen32b"]
    a = np.array([x[q] for q in S]); b = np.array([x[q] for q in F])
    m = mde(a.std(ddof=1), b.std(ddof=1), len(a), len(b))
    obs = a.mean() - b.mean()
    # n per group needed to detect the observed effect at 80% power
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    var = a.var(ddof=1) + b.var(ddof=1)
    n_need = (z ** 2) * var / (obs ** 2) if obs else None
    res["H1_power"] = {
        "observed_gap": round(float(obs), 2),
        "sd_static": round(float(a.std(ddof=1)), 1), "sd_fresh": round(float(b.std(ddof=1)), 1),
        "n_static": len(a), "n_fresh": len(b),
        "mde_80pct": round(float(m), 2),
        "underpowered": bool(m > obs),
        "n_per_group_needed_for_observed": int(n_need) if n_need else None,
        "questions_available_total": 4927,
        "questions_used": len(Q),
        "note": "Token F1 on short spans is near-bimodal, so its SD is huge (~40) "
                "relative to the effect. That, not the effect's absence, is what "
                "the MDE reflects.",
    }

    # ---------------- H2: variance reduction via a difficulty covariate --------
    # A model that cannot know the answer gives a per-question difficulty signal.
    out = {}
    for base in ("qwen", "qwen1_5b", "qwen3b"):
        if base not in cb:
            continue
        common = [q for q in Q if q in cb["qwen32b"] and q in cb[base]]
        cs = [q for q in common if Q[q]["year"] < 2025]
        cf = [q for q in common if Q[q]["year"] >= 2025]
        raw = unpaired([cb["qwen32b"][q] for q in cs], [cb["qwen32b"][q] for q in cf],
                       f"raw:{base}")
        adj = unpaired([cb["qwen32b"][q] - cb[base][q] for q in cs],
                       [cb["qwen32b"][q] - cb[base][q] for q in cf], f"adj:{base}")
        corr = float(np.corrcoef([cb["qwen32b"][q] for q in common],
                                 [cb[base][q] for q in common])[0, 1])
        out[base] = {"corr_with_primary": round(corr, 3),
                     "raw": raw, "difficulty_adjusted": adj,
                     "se_reduction_pct": round((1 - adj["se"] / raw["se"]) * 100, 1)}
    # LOO (same model, no memory of the event, no evidence) as the covariate
    loo = load("qwen32b", "hybrid_loo", Q)
    if loo:
        common = [q for q in Q if q in cb["qwen32b"] and q in loo]
        cs = [q for q in common if Q[q]["year"] < 2025]
        cf = [q for q in common if Q[q]["year"] >= 2025]
        raw = unpaired([cb["qwen32b"][q] for q in cs], [cb["qwen32b"][q] for q in cf], "raw:loo")
        adj = unpaired([cb["qwen32b"][q] - loo[q] for q in cs],
                       [cb["qwen32b"][q] - loo[q] for q in cf], "adj:loo")
        corr = float(np.corrcoef([cb["qwen32b"][q] for q in common],
                                 [loo[q] for q in common])[0, 1])
        out["hybrid_loo_same_model"] = {"corr_with_primary": round(corr, 3),
                                        "raw": raw, "difficulty_adjusted": adj,
                                        "se_reduction_pct": round((1 - adj["se"] / raw["se"]) * 100, 1)}
    res["H2_variance_reduction"] = out

    # ---------------- H3: boundary dilution -----------------------------------
    # If Qwen's cutoff is ~2023/2024, 2024 sits in Static but is post-cutoff.
    by = {}
    for t in ("qwen32b", "qwen72b", "mistral_large"):
        if t not in cb:
            continue
        v = cb[t]
        rows = {}
        for bnd in (2024, 2025):
            s = [v[q] for q in Q if Q[q]["year"] < bnd and q in v]
            f = [v[q] for q in Q if Q[q]["year"] >= bnd and q in v]
            rows[f"boundary_{bnd}"] = unpaired(s, f, f"dilute:{t}:{bnd}")
        # drop the ambiguous 2024 bin entirely
        s = [v[q] for q in Q if Q[q]["year"] <= 2023 and q in v]
        f = [v[q] for q in Q if Q[q]["year"] >= 2025 and q in v]
        rows["exclude_2024"] = unpaired(s, f, f"dilute:{t}:excl")
        by[t] = rows
    res["H3_boundary_dilution"] = by

    # ---------------- H4: is there knowledge to lose? -------------------------
    # Compare each model against a no-knowledge floor, per slice.
    floor = cb.get("qwen")
    know = {}
    for t in ("qwen32b", "qwen72b", "mistral_large", "haiku"):
        if t not in cb or not floor:
            continue
        v = cb[t]
        common = [q for q in Q if q in v and q in floor]
        for sl, sel in (("static", lambda y: y < 2025), ("fresh", lambda y: y >= 2025)):
            qs = [q for q in common if sel(Q[q]["year"])]
            know.setdefault(t, {})[sl] = round(
                float(np.mean([v[q] - floor[q] for q in qs])), 2)
    res["H4_knowledge_over_floor"] = {
        "premium_over_0.5B": know,
        "note": "Static premium is what the model could lose at the cutoff. If it is "
                "large but the cliff is null, the null is a measurement problem.",
    }

    (RES / "why_null.json").write_text(json.dumps(res, indent=2))

    print("=== H1: POWER ===")
    h = res["H1_power"]
    print(f"  observed gap {h['observed_gap']}  MDE(80%) {h['mde_80pct']}  "
          f"-> {'UNDERPOWERED' if h['underpowered'] else 'powered'}")
    print(f"  SD static {h['sd_static']} / fresh {h['sd_fresh']}  (n={h['n_static']}/{h['n_fresh']})")
    print(f"  n per group needed for the observed effect: {h['n_per_group_needed_for_observed']}"
          f"   (we used {h['n_static']}/{h['n_fresh']}; {h['questions_available_total']} exist)")
    print("\n=== H2: VARIANCE REDUCTION (difficulty covariate) ===")
    for k, v in res["H2_variance_reduction"].items():
        r, a = v["raw"], v["difficulty_adjusted"]
        print(f"  {k:24s} corr={v['corr_with_primary']:+.2f}  raw {r['diff']:+5.2f} "
              f"(SE {r['se']:.2f}, p={r['one_sided_p']:.3f})  ->  adj {a['diff']:+5.2f} "
              f"(SE {a['se']:.2f}, p={a['one_sided_p']:.3f})  SE {v['se_reduction_pct']:+.1f}%")
    print("\n=== H3: BOUNDARY DILUTION ===")
    for t, rows in res["H3_boundary_dilution"].items():
        s = "  ".join(f"{k.replace('boundary_','b'):12s}={v['diff']:+6.2f} (p={v['one_sided_p']:.3f})"
                      for k, v in rows.items())
        print(f"  {t:14s} {s}")
    print("\n=== H4: KNOWLEDGE PREMIUM OVER THE 0.5B FLOOR ===")
    for t, v in res["H4_knowledge_over_floor"]["premium_over_0.5B"].items():
        print(f"  {t:14s} Static +{v['static']:.1f}  Fresh +{v['fresh']:.1f}  "
              f"-> differential {v['static']-v['fresh']:+.1f}")


if __name__ == "__main__":
    main()
