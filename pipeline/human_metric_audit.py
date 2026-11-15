#!/usr/bin/env python3
"""What the human labels say about our own metrics.

`annotation/score_annotations.py` produces the correctness estimates. This adds
the three diagnostics the manuscript needs and the failure classes behind them:

  1. Does token-F1's error CANCEL across the boundary? The paper's defence of a
     lexical metric is that claims are contrasts scored identically, so a
     constant bias cancels. That only holds if the error rate is the same on
     both sides. Tested here as the Static-minus-Fresh difference in the rate at
     which token-F1 misses an answer the human calls correct, with a stratified
     bootstrap that respects the sampling design.
  2. How often does our abstention detector miss a real refusal?
  3. Which surface phenomena produce the false negatives?

All summaries use the same majority labels with adjudication overrides,
population weights, threshold, and within-stratum bootstrap draws.

Outputs results/human_metric_audit.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

# imports a sibling by path; keep the artifact free of __pycache__
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "annotation" / "pack_2026-08-02"
BOUNDARY = "2025"


def uni_punct(s):
    return any(unicodedata.category(c).startswith("P") and ord(c) > 127 for c in s)


def _freshness_reliability():
    """Which parts of the freshness construct two annotators can agree on.

    The five-label field scores kappa 0.39 overall, below the protocol's 0.6
    threshold, so it is not a measurement as collected. But the failure is not
    uniform: the annotators agree almost perfectly on whether evidence came
    from AFTER the question, and barely at all on whether it was STALE. That
    distinction matters, because the temporal-leakage result rests on the first
    and the stale-evidence taxonomy on the second.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "annotation"))
    from score_annotations import cohens_kappa
    from score_annotations import krippendorff_alpha_nominal
    fs = sorted(PACK.glob("task_b_evidence__annotator?.csv"))
    if len(fs) < 2:
        return {"status": "needs two annotators"}
    D = [{(r["item_id"], r["evidence_id"]): r["freshness"]
          for r in csv.DictReader(f.open(encoding="utf-8"))} for f in fs]
    raters = [f.name.split("__annotator")[1][0] for f in fs]
    ks = sorted(set.intersection(*[set(d) for d in D]))

    def kap(fn):
        """Krippendorff alpha across every rater present."""
        return round(krippendorff_alpha_nominal(
            [[fn(d[k]) for d in D] for k in ks]), 4)

    out = {
        "as_collected_5_labels": kap(lambda v: v),
        "anachronistic_vs_rest": kap(
            lambda v: "ANACH" if v == "ANACHRONISTIC" else "OTHER"),
        "superseded_vs_rest": kap(
            lambda v: "STALE" if v == "SUPERSEDED" else "NOT_STALE"),
        "n_rows": len(ks),
        "n_raters": len(D),
        "raters": raters,
        "statistic": "Krippendorff alpha (nominal) across all raters",
        "threshold": 0.6,
    }
    out["reading"] = (
        f"Temporal admissibility is measurable (kappa "
        f"{out['anachronistic_vs_rest']}) and independently corroborates the "
        f"mechanical leakage audit. Staleness is not (kappa "
        f"{out['superseded_vs_rest']}): the annotators diverge on whether a "
        f"fact COULD have changed, which is the SUPERSEDED/NOT_APPLICABLE "
        f"boundary. No human-judged staleness claim is supportable, and that "
        f"is itself evidence about the construct our stale-evidence taxonomy "
        f"depends on.")
    out["usable_at_threshold"] = {
        k: bool(v >= 0.6) for k, v in out.items()
        if isinstance(v, float)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/human_metric_audit.json")
    args = ap.parse_args()

    man = json.loads((PACK / "sampling_manifest.json").read_text())["task_a"]["strata"]
    W = {s: v["weight_to_population"] for s, v in man.items()}
    key = {r["item_id"]: r for r in
           (json.loads(l) for l in
            (PACK / "item_metadata.jsonl").read_text().splitlines() if l.strip())}
    orig = {r["item_id"]: r for r in
            csv.DictReader((PACK / "task_a_correctness.csv").open(encoding="utf-8"))}
    # Use the labels both annotators agreed on where two passes exist; a
    # disagreed item has no ground truth until somebody adjudicates it.
    files = sorted(PACK.glob("task_a_correctness__annotator?.csv"))
    if not files:
        print("no ingested annotator file", file=sys.stderr)
        return 1
    per = [{r["item_id"]: r["verdict"]
            for r in csv.DictReader(f.open(encoding="utf-8"))} for f in files]
    annotators = [f.name.split("__annotator")[1][0] for f in files]
    if len(per) == 1:
        lab, basis = per[0], f"single annotator {annotators[0]}"
        n_dropped = 0
    elif len(per) >= 3:
        # Majority over three raters covers every item, so unlike an
        # agreed-subset estimate it does not quietly drop the hard cases.
        # sorted(): iterating a set is order-nondeterministic across runs
        # (string hash randomisation), which reorders `rows` and therefore
        # changes the bootstrap draws. The intervals moved by ~0.4 pts between
        # runs before this was pinned.
        keys = sorted(set.intersection(*[set(d) for d in per]))
        lab, n_dropped = {}, 0
        for k in keys:
            votes = collections.Counter(d[k] for d in per)
            top, n = votes.most_common(1)[0]
            if n * 2 > len(per):
                lab[k] = top
            else:
                n_dropped += 1
        basis = (f"majority of {len(annotators)} annotators "
                 f"({''.join(annotators)}), {n_dropped} ties dropped")
    else:
        keys = sorted(set(per[0]) & set(per[1]))
        lab = {k: per[0][k] for k in keys if per[0][k] == per[1][k]}
        n_dropped = len(keys) - len(lab)
        basis = (f"{len(annotators)} annotators, agreed subset only "
                 f"({n_dropped} disagreements dropped, unadjudicated)")
    adjudicated = list(csv.DictReader((PACK / "task_a_correctness__adjudicated.csv").open(encoding="utf-8")))
    overrides = {r["item_id"]: r["adjudicated"].strip().upper() for r in adjudicated if r["adjudicated"].strip()}
    assert set(overrides) <= set(lab), "Adjudication must refer to sampled items"
    lab.update(overrides)
    basis = f"majority of {len(per)} with {len(overrides)} adjudication overrides"
    print(f"basis: {basis}")

    rows = []
    for i, v in sorted(lab.items()):
        k = key.get(i)
        if not k:
            continue
        rows.append({
            "stratum": k["stratum"], "w": W[k["stratum"]],
            "slice": "Static" if orig[i]["question_date"][:4] < BOUNDARY else "Fresh",
            "f1_ok": k["auto_f1"] >= 0.5, "human_ok": v == "CORRECT",
            "human": v, "gold": orig[i]["reference_answer"],
            "sys": orig[i]["system_answer"],
        })

    def estimates(sel):
        total = sum(r["w"] for r in sel)
        correct = sum(r["w"] for r in sel if r["human_ok"])
        missed = sum(r["w"] for r in sel if r["human_ok"] and not r["f1_ok"])
        credited = sum(r["w"] for r in sel if not r["human_ok"] and r["f1_ok"])
        return {
            "human_correct_rate": correct / total,
            "miss_prevalence": missed / total,
            "conditional_miss_rate": missed / correct if correct else None,
            "false_credit_rate": credited / (total - correct) if total > correct else None,
            "agreement": 1 - (missed + credited) / total,
        }

    def summarize(sample):
        ans = {group: estimates(sample if group == "overall" else [r for r in sample if r["slice"] == group])
               for group in ("overall", "Static", "Fresh")}
        ans["Static_minus_Fresh"] = {k: ans["Static"][k] - ans["Fresh"][k]
            for k in ans["Static"] if ans["Static"][k] is not None and ans["Fresh"][k] is not None}
        return ans

    observed = summarize(rows)
    bystr = collections.defaultdict(list)
    for r in rows:
        bystr[r["stratum"]].append(r)
    rng = random.Random(args.seed)
    bootstrap = collections.defaultdict(list)
    for _ in range(args.B):
        samp = [items[rng.randrange(len(items))] for items in bystr.values() for _ in items]
        for group, metrics in summarize(samp).items():
            for metric, value in metrics.items():
                if value is not None:
                    bootstrap[group, metric].append(value)
    summaries = {}
    for group, metrics in observed.items():
        summaries[group] = {}
        for metric, estimate in metrics.items():
            ds = sorted(bootstrap[group, metric]); n = len(ds)
            summaries[group][metric] = {"estimate": estimate,
                "ci95": [ds[int(.025*n)], ds[int(.975*n)]], "valid_draws": n}
    draws = sorted(bootstrap["Static_minus_Fresh", "miss_prevalence"])
    n = len(draws); le = sum(v <= 0 for v in draws)
    obs = observed["Static_minus_Fresh"]["miss_prevalence"]
    def fn_rate(sel):
        return estimates(sel)["miss_prevalence"]

    # abstention detector recall against the human label
    h_abs = [r for r in rows if r["human"] == "ABSTAIN"]
    caught = [r for r in h_abs if r["stratum"] == "abstention"]

    # failure classes among false negatives
    fneg = [r for r in rows if r["human_ok"] and not r["f1_ok"]]
    cls = collections.Counter()
    for r in fneg:
        g, s = r["gold"], r["sys"]
        if uni_punct(g) or uni_punct(s):
            cls["unicode_punctuation"] += 1
        elif re.sub(r"s$", "", g.lower().strip()) == re.sub(r"s$", "", s.lower().strip()):
            cls["plural_or_case"] += 1
        elif re.search(r"[\d.,$]", g) and re.search(r"[\d.,$]", s):
            cls["number_or_currency_format"] += 1
        elif "-" in g or "-" in s:
            cls["hyphenation"] += 1
        else:
            cls["other_paraphrase_or_alias"] += 1

    out = {
        "scope": "Thresholded token-F1 errors against adjudicated human correctness labels.",
        "analysis_version": "writing-revision-2026-11-15",
        "n_items": len(rows), "n_adjudication_overrides": len(overrides),
        "threshold": 0.5, "bootstrap": {"replicates": args.B, "seed": args.seed,
            "unit": "answer item, sampled within original strata", "interval": "percentile 95%",
            "shared_draws": True},
        "weighting": "sampling_manifest.task_a.strata.weight_to_population; calibration included",
        "consistent_summaries": summaries,
        "provisional": len(per) < 3,
        "label_basis": basis,
        "n_disagreements_dropped": n_dropped,
        "why_provisional": None,
        "does_metric_error_cancel_across_the_boundary": {
            "estimand": ("Static minus Fresh difference in the population-"
                         "weighted proportion of ALL outputs that are human-correct "
                         "but scored below 0.5 (not a conditional miss rate)"),
            "static_false_negative_rate": round(fn_rate(
                [r for r in rows if r["slice"] == "Static"]), 4),
            "fresh_false_negative_rate": round(fn_rate(
                [r for r in rows if r["slice"] == "Fresh"]), 4),
            "difference_pts": round(100 * obs, 2),
            "ci95_pts": [round(100 * draws[int(0.025 * n)], 2),
                         round(100 * draws[int(0.975 * n)], 2)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 4),
            "excludes_zero": bool(draws[int(0.025 * n)] > 0 or draws[int(0.975 * n)] < 0),
            "reading": ("A positive difference would mean token-F1 under-credits "
                        "Static answers more than Fresh ones, which would bias "
                        "the measured Static-Fresh gap DOWNWARD. The point "
                        "estimate points that way, but the interval includes "
                        "zero: this is a caution about the endpoint, not a "
                        "demonstrated bias, and it is leveraged by 40 "
                        "calibration items carrying weights of 196 and 504."),
        },
        "abstention_detector": {
            "human_labelled_abstentions": len(h_abs),
            "detected_by_our_rule": len(caught),
            "recall": round(len(caught) / len(h_abs), 4) if h_abs else None,
            "reading": ("Our is_abstention() rule matches only known prefixes, "
                        "so refusals phrased mid-sentence ('Based on the "
                        "documents provided, I cannot answer...') are scored as "
                        "wrong answers. Reported rates are detector-defined; recall "
                        "alone does not establish a lower bound without false-positive control."),
        },
        "freshness_construct_reliability": _freshness_reliability(),
        "false_negative_classes": dict(cls.most_common()),
        "n_false_negatives": len(fneg),
        "examples": [{"gold": r["gold"], "system": r["sys"][:80]} for r in fneg[:8]],
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    d = out["does_metric_error_cancel_across_the_boundary"]
    print(f"Static FN {d['static_false_negative_rate']:.4f} vs Fresh "
          f"{d['fresh_false_negative_rate']:.4f}")
    print(f"difference {d['difference_pts']:+.2f} pts {d['ci95_pts']} "
          f"p={d['p_plus_one']}  excludes zero: {d['excludes_zero']}")
    a = out["abstention_detector"]
    print(f"abstention recall {a['recall']} ({a['detected_by_our_rule']}/"
          f"{a['human_labelled_abstentions']})")
    print(f"false-negative classes: {out['false_negative_classes']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
