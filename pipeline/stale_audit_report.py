#!/usr/bin/env python3
"""Score the manual adjudication of the stale-failure taxonomy.

`stale_taxonomy.py` labels a stale-index failure `grounded_outdated` when the
answer's content words overlap the retrieved text, and an earlier version of
this paper read that label as "the model faithfully reported a superseded
fact". The audit in `results/stale_audit_labels.json` checks that reading on a
random sample of 60 such items and finds it wrong for most of them: the rule
also fires on answers that were *never* correct at any time, but happen to be
salient entities of roughly the right type sitting in stale news.

This script combines the audit with the automatic counts to produce the
corrected rate, with a Wilson interval on the audited proportion propagated
through to the failure-level rate.

Outputs `results/stale_audit_report.json`.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/stale_audit_report.json")
    args = ap.parse_args()

    tax = json.loads((ROOT / "results/stale_taxonomy.json").read_text())

    # Two audits: the original Qwen-only sample, and a separate Mistral sample
    # drawn once its stale run finished. They are scored separately because the
    # precision of the automatic label turns out to depend on the generator.
    audits = [("results/stale_audit_labels.json", "results/stale_audit_sample.json"),
              ("results/stale_audit_labels_mistral.json",
               "results/stale_audit_sample_mistral.json")]
    items, by_i = [], {}
    for lab_rel, samp_rel in audits:
        lab_p, samp_p = ROOT / lab_rel, ROOT / samp_rel
        if not (lab_p.exists() and samp_p.exists()):
            continue
        labels = json.loads(lab_p.read_text())["labels"]
        sample = json.loads(samp_p.read_text())["items"]
        assert len(labels) == len(sample), f"{lab_rel} must cover its whole sample"
        assert {int(l["i"]) for l in labels} == set(range(len(sample))), \
            f"{lab_rel}: every sampled item needs a verdict"
        base = len(items)
        for l in labels:
            by_i[base + int(l["i"])] = l["verdict"]
        items.extend(sample)

    counts = Counter(by_i[i] for i in range(len(items)))
    n = len(items)
    k = counts["superseded"]
    lo, hi = wilson(k, n)

    # Per-model precision -- the correction is NOT uniform across generators.
    per_model = {}
    for i, it in enumerate(items):
        d = per_model.setdefault(it["model"], Counter())
        d[by_i[i]] += 1
    per_model_precision = {}
    for m, c in per_model.items():
        nm = sum(c.values()); km = c["superseded"]
        mlo, mhi = wilson(km, nm)
        per_model_precision[m] = {"k": km, "n": nm, "point": round(km / nm, 4),
                                  "wilson95": [round(mlo, 4), round(mhi, 4)]}

    out = {
        "scope": "Precision of the automatic `grounded_outdated` label against "
                 "manual adjudication, and the corrected failure-level rate.",
        "audited": {"n": n, "counts": dict(counts)},
        "precision_of_automatic_label": {
            "definition": "share of automatically-labelled grounded_outdated "
                          "items that are genuinely superseded facts",
            "point": round(k / n, 4),
            "wilson95": [round(lo, 4), round(hi, 4)],
            "k": k, "n": n,
        },
        "per_model_audit": {m: dict(c) for m, c in per_model.items()},
        "per_model_precision": per_model_precision,
        "corrected_rates": {},
    }

    for key, m in tax["models"].items():
        nf = m["n_stale_failures"]
        auto_pct = m["pct"].get("grounded_outdated")
        if not nf or auto_pct is None:
            continue
        auto = auto_pct / 100.0
        # Use this generator's OWN audited precision where we have one; only
        # fall back to the pooled figure otherwise, and say which was used.
        pm = per_model_precision.get(key)
        src = "own audit" if pm else "pooled (no per-generator audit)"
        pk, plo, phi = ((pm["point"], *pm["wilson95"]) if pm else (k / n, lo, hi))
        out["corrected_rates"][key] = {
            "name": m["name"],
            "n_stale_failures": nf,
            "automatic_grounded_outdated_pct": auto_pct,
            "precision_source": src,
            "precision_used": round(pk, 4),
            "corrected_superseded_pct": round(100.0 * auto * pk, 1),
            "corrected_ci95_pct": [round(100.0 * auto * plo, 1),
                                   round(100.0 * auto * phi, 1)],
        }

    out["precision_varies_by_generator"] = (
        "The automatic label's precision is NOT constant across generators: it is "
        + ", ".join(f"{per_model_precision[m]['point']*100:.0f}% on {m}"
                    for m in sorted(per_model_precision))
        + ". The echo failure mode -- an answer lifted from stale news that was "
        "never correct -- is concentrated in the smaller models; when the 123B "
        "produces a grounded-looking wrong answer it is far more often a "
        "genuinely superseded fact. Correcting every generator by one pooled "
        "precision would therefore have been wrong, and we use each generator's "
        "own audited precision where one exists."
    )
    out["finding"] = (
        "The automatic label is right about %d of %d sampled items (%.0f%%, "
        "95%% Wilson [%.0f%%, %.0f%%]). The dominant error is what we call an "
        "ECHO: an answer pulled from the stale context that was never a "
        "correct answer to the question at any time (%d of %d), plus outright "
        "type errors (%d) and non-answers (%d). Correcting the automatic "
        "counts by this precision, faithful reporting of a superseded fact "
        "accounts for roughly %s%% of stale-index failures, not the "
        "%s%% the unaudited rule reports."
        % (k, n, 100.0 * k / n, 100.0 * lo, 100.0 * hi,
           counts["echo"], n, counts["type_error"], counts["non_answer"],
           "-".join(sorted({str(v["corrected_superseded_pct"])
                            for v in out["corrected_rates"].values()},
                           key=float)),
           "-".join(sorted({str(v["automatic_grounded_outdated_pct"])
                            for v in out["corrected_rates"].values()},
                           key=float)))
    )
    out["implication"] = (
        "Both modes are dangerous and both are grounded in the retrieved text, "
        "but they call for different mitigations. A faithfully-read superseded "
        "fact is a freshness problem: the fix is recency-aware retrieval or "
        "date-conditioned generation. An echo is a grounding and calibration "
        "problem: the evidence does not support the answer at all, and the fix "
        "is abstention or answer-verification. Reporting them as one number, "
        "as the unaudited taxonomy did, points at the wrong fix for the "
        "majority of failures."
    )

    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["audited"], indent=2))
    print(f"\npooled precision {k}/{n} = {100.0*k/n:.1f}%  Wilson95 "
          f"[{100*lo:.1f}%, {100*hi:.1f}%]")
    for m, v in sorted(per_model_precision.items()):
        print(f"  precision {m:16s} {v['k']:2d}/{v['n']:2d} = {100*v['point']:5.1f}%  "
              f"Wilson95 [{100*v['wilson95'][0]:.1f}%, {100*v['wilson95'][1]:.1f}%]")
    print()
    for key, v in out["corrected_rates"].items():
        print(f"  {v['name']:18s} auto {v['automatic_grounded_outdated_pct']:5.1f}%"
              f" -> corrected {v['corrected_superseded_pct']:5.1f}% "
              f"{v['corrected_ci95_pct']}  ({v['precision_source']})")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
