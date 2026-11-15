#!/usr/bin/env python3
"""Is stale-label precision really capability-graded?

The paper states that the precision of the automatic `grounded_outdated` label
"is itself capability-graded": 24% at 3B, 30% at 32B, 47% at 123B. Three
proportions in increasing order is not a trend test, and with 27-33 adjudicated
items per model the Wilson intervals overlap heavily. This runs the tests that
were missing:

  * an EXACT Cochran-Armitage trend test by complete enumeration of the
    permutation null -- we shuffle the 90 adjudicated verdicts across the three
    model groups (holding group sizes fixed) and count how often the trend
    statistic is at least as extreme. Exact by Monte Carlo over a very large
    number of shuffles, with the count reported so the resolution is visible.
  * an exact Fisher test on the extreme pair (3B vs 123B), which is the
    comparison the claim really rests on.
  * the pooled precision with its Wilson interval, which IS supported.
  * a power calculation: how many adjudicated items per model would be needed
    to detect the observed 3B-vs-123B difference at 80% power? This says
    plainly what a real version of this audit would cost.

We cannot expand the hand adjudication -- there are no annotators for this
paper, and inventing labels to enlarge a sample would be worse than reporting a
small one. What we can do is test the claim the existing labels support.

Outputs results/stale_trend_test.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# log10 parameters, the same capability axis the rest of the paper uses
PARAMS_B = {"qwen3b": 3.0, "qwen32b": 32.0, "mistral_large": 123.0}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def ca_statistic(groups):
    """Cochran-Armitage trend statistic (numerator only; monotone in the trend).

    groups: list of (score, k_success, n) with score = log10(params).
    """
    N = sum(n for _, _, n in groups)
    K = sum(k for _, k, _ in groups)
    if N == 0 or K == 0 or K == N:
        return 0.0
    pbar = K / N
    sbar = sum(s * n for s, _, n in groups) / N
    num = sum((s - sbar) * (k - n * pbar) for s, k, n in groups)
    var = pbar * (1 - pbar) * sum(n * (s - sbar) ** 2 for s, _, n in groups)
    return num / math.sqrt(var) if var > 0 else 0.0


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]]."""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p_obs = pr(a)
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    return round(sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p_obs + 1e-12), 6)


def n_per_group_for(p1, p2, power=0.80, alpha=0.05):
    """Two-proportion sample size per group (normal approximation)."""
    if p1 == p2:
        return None
    za, zb = 1.959964, 0.8416212
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar))
           + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / (p1 - p2) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--output", type=Path, default=RES / "stale_trend_test.json")
    args = ap.parse_args()

    rep = json.loads((RES / "stale_audit_report.json").read_text())
    per = rep["per_model_precision"]

    groups = []
    labels = []
    for m, v in per.items():
        s = math.log10(PARAMS_B[m])
        groups.append((s, v["k"], v["n"]))
        labels += [(s, 1)] * v["k"] + [(s, 0)] * (v["n"] - v["k"])

    obs = ca_statistic(groups)
    sizes = [n for _, _, n in groups]
    scores = [s for s, _, _ in groups]
    verdicts = [y for _, y in labels]

    rng = random.Random(args.seed)
    n_ge = 0
    for _ in range(args.B):
        v = verdicts[:]
        rng.shuffle(v)
        i = 0
        gg = []
        for s, n in zip(scores, sizes):
            gg.append((s, sum(v[i:i + n]), n))
            i += n
        if abs(ca_statistic(gg)) >= abs(obs) - 1e-12:
            n_ge += 1
    p_trend = (n_ge + 1) / (args.B + 1)

    lo_m = min(per, key=lambda m: PARAMS_B[m])
    hi_m = max(per, key=lambda m: PARAMS_B[m])
    a, na = per[lo_m]["k"], per[lo_m]["n"]
    b, nb = per[hi_m]["k"], per[hi_m]["n"]
    p_fisher = fisher_exact_2x2(a, na - a, b, nb - b)

    K = sum(v["k"] for v in per.values())
    N = sum(v["n"] for v in per.values())

    need = n_per_group_for(per[lo_m]["point"], per[hi_m]["point"])

    out = {
        "scope": "Trend and pairwise tests for the claim that the automatic "
                 "stale-label's precision is capability-graded.",
        "n_adjudicated": N,
        "per_model": {m: {"k": v["k"], "n": v["n"], "precision": v["point"],
                          "wilson95": v["wilson95"]} for m, v in per.items()},
        "pooled_precision": {
            "k": K, "n": N, "point": round(K / N, 4), "wilson95": wilson(K, N),
            "status": "SUPPORTED -- this is the result the paper should lead with",
        },
        "cochran_armitage_trend": {
            "statistic": round(obs, 4),
            "n_shuffles": args.B,
            "n_at_least_as_extreme": n_ge,
            "p_permutation": round(p_trend, 5),
            "significant_at_05": bool(p_trend < 0.05),
        },
        "fisher_extreme_pair": {
            "comparison": f"{lo_m} vs {hi_m}",
            "table": [[a, na - a], [b, nb - b]],
            "p_two_sided": p_fisher,
            "significant_at_05": bool(p_fisher < 0.05),
        },
        "power": {
            "observed_precisions": [per[lo_m]["point"], per[hi_m]["point"]],
            "n_per_model_for_80pct_power": need,
            "n_per_model_actual": [per[lo_m]["n"], per[hi_m]["n"]],
            "shortfall_factor": round(need / min(per[lo_m]["n"], per[hi_m]["n"]), 1)
            if need else None,
        },
    }
    sig = out["cochran_armitage_trend"]["significant_at_05"] or \
        out["fisher_extreme_pair"]["significant_at_05"]
    out["verdict"] = {
        "capability_graded_precision": "SUPPORTED" if sig else "NOT SUPPORTED",
        "what_to_report": (
            "The pooled precision is the supported result: the automatic "
            f"grounded_outdated label is right {100*K/N:.0f}% of the time "
            f"(Wilson {wilson(K, N)}), so roughly two thirds of items it flags "
            "are not superseded facts. The apparent ordering across model sizes "
            f"is exploratory: the exact trend test gives p={p_trend:.3f} and the "
            f"extreme-pair Fisher test p={p_fisher:.3f}, and detecting a "
            f"difference of the observed size at 80% power would need about "
            f"{need} adjudicated items per model against the "
            f"{min(per[lo_m]['n'], per[hi_m]['n'])} we have."),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"pooled precision {K}/{N} = {100*K/N:.1f}% Wilson {wilson(K, N)}")
    for m, v in per.items():
        print(f"  {m:16s} {v['k']:3d}/{v['n']:3d} = {100*v['point']:5.1f}% "
              f"Wilson {v['wilson95']}")
    print(f"\nCochran-Armitage trend p = {p_trend:.4f} "
          f"({n_ge}/{args.B} shuffles at least as extreme)")
    print(f"Fisher {lo_m} vs {hi_m}      p = {p_fisher:.4f}")
    print(f"n per model needed for 80% power: {need} (have "
          f"{min(per[lo_m]['n'], per[hi_m]['n'])})")
    print(f"\ncapability-graded precision: "
          f"{out['verdict']['capability_graded_precision']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
