#!/usr/bin/env python3
"""Score the freshness re-pilot and decide whether the rewritten rule works.

The pilot passes if Krippendorff's alpha on `freshness` reaches 0.6. If it does
not, this script says WHERE the rule failed rather than just that it did, using
the `claim_type` column:

  * raters agree on claim_type but differ on freshness -> the step-3 mapping is
    wrong, and the fix is small;
  * raters differ on claim_type -> step 2 is still ambiguous, and the rule needs
    another rewrite before anyone labels anything again.

Outputs results/freshness_repilot.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

# these scripts import siblings by path; keep the shipped
# artifact free of __pycache__ directories
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "annotation"))
from score_annotations import cohens_kappa, krippendorff_alpha_nominal  # noqa: E402

PACK = Path(__file__).resolve().parent / "pack"
TARGET = 0.6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, default=PACK)
    args = ap.parse_args()

    files = sorted(args.pack.glob("freshness_repilot__annotator?.csv"))
    if len(files) < 2:
        print(f"found {len(files)} returned files in {args.pack}; need >=2")
        return 1
    per, who = {}, []
    for f in files:
        w = f.name.split("__annotator")[1][0]
        who.append(w)
        for r in csv.DictReader(f.open(encoding="utf-8")):
            per.setdefault((r["item_id"], r["evidence_id"]), {})[w] = r
    ks = sorted(k for k, v in per.items() if len(v) == len(who))

    out = {"scope": "freshness re-pilot under the v2 operationalised rule",
           "n_items": len(ks), "raters": who, "target_alpha": TARGET}
    for col in ("freshness", "claim_type"):
        units = [[per[k][w][col].strip().upper() for w in who] for k in ks]
        a = krippendorff_alpha_nominal(units)
        pk = {f"{x}{y}": round(cohens_kappa([per[k][x][col] for k in ks],
                                            [per[k][y][col] for k in ks]), 4)
              for i, x in enumerate(who) for y in who[i + 1:]}
        out[col] = {"krippendorff_alpha": round(a, 4), "pairwise_kappa": pk,
                    "distribution": dict(collections.Counter(
                        v for u in units for v in u).most_common())}
        print(f"{col:11s} alpha={a:.3f}  pairwise={pk}")

    # An aggregate alpha can hide a 2-vs-1 split: two raters applying the rule
    # identically and one applying a different one average out to a number that
    # looks like moderate agreement from everybody. Check for it explicitly.
    NA = "NOT_ABOUT_THIS_QUESTION"
    pk = out["freshness"]["pairwise_kappa"]
    outlier = None
    if len(who) >= 3 and pk:
        by_rater = {w: [v for pair, v in pk.items() if w in pair] for w in who}
        worst = min(by_rater, key=lambda w: sum(by_rater[w]) / len(by_rater[w]))
        best_pair = max(pk.values())
        if best_pair - max(v for pair, v in pk.items() if worst in pair) > 0.15:
            outlier = worst
    # and how much of any split is just the off-topic escape hatch?
    scoped = [k for k in ks
              if all(per[k][w]["freshness"].strip().upper() != NA for w in who)]
    a_scoped = (krippendorff_alpha_nominal(
        [[per[k][w]["freshness"].strip().upper() for w in who] for k in scoped])
        if len(scoped) > 20 else None)
    out["scope_disagreement"] = {
        "off_topic_calls_per_rater": {
            w: sum(1 for k in ks if per[k][w]["freshness"].strip().upper() == NA)
            for w in who},
        "n_rows_all_agree_on_topic": len(scoped),
        "alpha_excluding_any_off_topic_call": round(a_scoped, 4) if a_scoped else None,
        "outlier_rater": outlier,
        "reading": ("If alpha rises sharply once rows any rater called off-topic "
                    "are excluded, the freshness rule itself is fine and the "
                    "unresolved question is what counts as 'about this "
                    "question'. Those are different defects with different "
                    "fixes."),
    }

    fa = out["freshness"]["krippendorff_alpha"]
    ca = out["claim_type"]["krippendorff_alpha"]
    sd = out["scope_disagreement"]
    conditional = bool(fa >= TARGET and (outlier or (
        a_scoped and a_scoped - fa > 0.1)))
    out["verdict"] = {
        "freshness_alpha": fa, "claim_type_alpha": ca,
        "passes": bool(fa >= TARGET),
        "unconditional_pass": bool(fa >= TARGET and not conditional),
        "conditional_pass": conditional,
        "caution": ((f"Alpha clears the threshold but rater {outlier} is an "
                     f"outlier (pairwise kappa {min(pk.values()):.2f} against "
                     f"{max(pk.values()):.2f} for the closest pair), and alpha "
                     f"rises to {a_scoped:.3f} once rows any rater called "
                     f"off-topic are dropped. The freshness rule is not what is "
                     f"failing; the off-topic boundary is. Fix that before "
                     f"scaling, or the full run inherits the same split.")
                    if conditional else None),
        "diagnosis": (
            "PASS: the rewritten rule clears the threshold; re-run freshness on "
            "the full on-topic set." if fa >= TARGET else
            "FAIL, and step 2 is the problem: raters do not agree on what kind "
            "of claim they are reading, so the freshness mapping cannot help. "
            "Rewrite claim_type before collecting anything else."
            if ca < TARGET else
            "FAIL, but step 2 is sound: raters agree on claim_type and still "
            "differ on freshness, so the step-3 mapping is what is wrong. Fix "
            "the mapping, not the typology."),
    }
    # where exactly does the mapping leak?
    leak = collections.Counter()
    for k in ks:
        cts = {per[k][w]["claim_type"].strip().upper() for w in who}
        frs = {per[k][w]["freshness"].strip().upper() for w in who}
        if len(cts) == 1 and len(frs) > 1:
            leak[(next(iter(cts)), tuple(sorted(frs)))] += 1
    out["mapping_leaks"] = {f"{c} -> {'/'.join(f)}": n
                            for (c, f), n in leak.most_common(10)}
    print(f"\n{out['verdict']['diagnosis']}")
    if leak:
        print("agreed claim_type but split freshness:")
        for (c, f), n in leak.most_common(5):
            print(f"  {c:9s} -> {'/'.join(f)}  ({n})")
    dest = ROOT / "results" / "freshness_repilot.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
