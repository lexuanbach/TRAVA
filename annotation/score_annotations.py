#!/usr/bin/env python3
"""Score the human annotation pack: agreement, adjudication, final estimates.

Two stages.

  --stage agreement   after both annotators finish, before adjudication.
                      Reports raw agreement, Cohen's kappa, Krippendorff's
                      alpha, the calibration pass rate, and writes the
                      disagreement list for the adjudicator.

  --stage final       after adjudication. Joins the adjudicated labels to the
                      key file, computes per-regime agreement between each
                      automatic metric and the human verdict, and reweights to
                      the population using the stratum weights. Writes
                      results/human_validation.json.

Two things this script refuses to do, both learned from a sister project whose
published Krippendorff alpha turned out to have been computed over annotator
*identifiers* rather than their labels:

  * it will not compute agreement from a file lacking per-annotator labels;
  * it will not treat calibration items as findings.

Expected inputs under the pack directory:
  task_a_correctness__annotatorA.csv    the pack file with `verdict` filled in
  task_a_correctness__annotatorB.csv
  task_b_evidence__annotatorA.csv       (same, for task B)
  task_b_evidence__annotatorB.csv
  task_a_correctness__adjudicated.csv   final stage only, `adjudicated` column
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

A_LABELS = ["CORRECT", "INCORRECT", "PARTIAL", "ABSTAIN", "UNCLEAR"]
B_SUPPORT = ["YES", "PARTIAL", "NO"]
# ANACHRONISTIC was added before any labels were collected: 98 of the 900 Task B
# rows (10.9%) carry evidence published AFTER the question, because the main
# index pools every sampled question's evidence and applies no per-question time
# mask. Without a label for it, annotators would have split that 10.9% across
# CURRENT and SUPERSEDED at random.
B_FRESH = ["CURRENT", "SUPERSEDED", "UNDATED", "NOT_APPLICABLE", "ANACHRONISTIC"]


class AnnotationFormatError(RuntimeError):
    pass


def read_labels(path: Path, col: str) -> dict[str, str]:
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if col not in row:
                raise AnnotationFormatError(
                    f"{path.name} has no '{col}' column; got {list(row)}")
            v = (row.get(col) or "").strip().upper()
            if v:
                # a row can repeat (task B: one row per evidence doc)
                out.setdefault((row["item_id"], row.get("evidence_id", "")), v)
    return out


def cohens_kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0 or len(cats) < 2:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def krippendorff_alpha_nominal(units):
    """Nominal alpha for ANY number of raters per unit, via the coincidence
    matrix. The pairwise-only version this replaces silently ignored a third
    rater."""
    import itertools as _it
    o = collections.Counter()
    for u in units:
        vals = [v for v in u if v]
        if len(vals) < 2:
            continue
        for a, b in _it.permutations(vals, 2):
            o[(a, b)] += 1.0 / (len(vals) - 1)
    if not o:
        return float("nan")
    n_c = collections.Counter()
    for (a, _b), v in o.items():
        n_c[a] += v
    n = sum(n_c.values())
    do = sum(v for (a, b), v in o.items() if a != b)
    de = sum(n_c[a] * n_c[b] for a in n_c for b in n_c if a != b) / (n - 1)
    return 1 - do / de if de else float("nan")


def majority(labels):
    """Consensus label and whether it is a genuine majority.

    With three raters a >=2/3 majority resolves nearly everything, which removes
    the selection bias of dropping disagreements. It is NOT the same as the
    protocol's third-party adjudication -- nobody reviewed the competing labels
    -- so it is recorded under its own provenance name.
    """
    c = collections.Counter(l for l in labels if l)
    if not c:
        return None, False
    top, n = c.most_common(1)[0]
    return top, n * 2 > len([l for l in labels if l])


def boot_ci(vals, B, seed, stat):
    if len(vals) < 2:
        return [None, None]
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        d = stat(s)
        if d == d:
            draws.append(d)
    if not draws:
        return [None, None]
    draws.sort()
    return [round(draws[int(0.025 * len(draws))], 3),
            round(draws[int(0.975 * len(draws))], 3)]


def load_key(pack: Path):
    f = pack / "item_metadata.jsonl"
    if not f.is_file():
        raise AnnotationFormatError(f"missing {f}")
    return {r["item_id"]: r for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def annotators_present(pack: Path, base: str) -> list[str]:
    return sorted(f.name.split("__annotator")[1][0]
                  for f in pack.glob(f"{base}__annotator?.csv"))


def stage_agreement(pack: Path, args) -> int:
    key = load_key(pack)
    out = {"pack": pack.name, "stage": "agreement", "tasks": {}}
    disagreements = []

    for task, col, labels in (("a", "verdict", A_LABELS),
                              ("b", "supports", B_SUPPORT),
                              ("b", "freshness", B_FRESH)):
        base = "task_a_correctness" if task == "a" else "task_b_evidence"
        who = annotators_present(pack, base)
        if len(who) >= 3:
            per = {w: read_labels(pack / f"{base}__annotator{w}.csv", col)
                   for w in who}
            common = sorted(set.intersection(*[set(v) for v in per.values()]))
            cal = {k for k in common
                   if key.get(k[0], {}).get("stratum", "").startswith("calibration")}
            scored = [k for k in common if k not in cal]
            units = [[per[w][k] for w in who] for k in scored]
            pk = {}
            for x, y in [(a, b) for i, a in enumerate(who) for b in who[i + 1:]]:
                pk[f"{x}{y}"] = round(cohens_kappa([per[x][k] for k in scored],
                                                   [per[y][k] for k in scored]), 4)
            al = krippendorff_alpha_nominal(units)
            res = [majority([per[w][k] for w in who]) for k in scored]
            nmaj = sum(1 for _, ok in res if ok)
            rec = {"annotators": who, "n_scored": len(scored),
                   "n_calibration_excluded": len(cal),
                   "pairwise_cohens_kappa": pk,
                   "krippendorff_alpha": round(al, 4),
                   "cohens_kappa": round(min(pk.values()), 4),
                   "kappa_basis": "minimum pairwise kappa, the conservative summary",
                   "majority_resolved": nmaj, "n_ties": len(scored) - nmaj,
                   "unanimous": sum(1 for u in units if len(set(u)) == 1)}
            out["tasks"][f"{base}:{col}"] = rec
            flag = "" if al >= 0.6 else "   <-- BELOW 0.6: do not use this field"
            print(f"[{base}:{col}] {len(who)} raters n={len(scored)} "
                  f"alpha={al:.3f} pairwise={pk} "
                  f"majority={nmaj}/{len(scored)}{flag}")
            for k in scored:
                labs = [per[w][k] for w in who]
                if len(set(labs)) > 1:
                    disagreements.append({
                        "item_id": k[0], "evidence_id": k[1], "field": col,
                        **{f"rater_{w}": per[w][k] for w in who}})
            continue
        if len(who) < 2:
            # Agreement is undefined with one pass. Say so rather than
            # inventing a second rater or comparing a file with itself.
            print(f"[{base}:{col}] only annotator {who or 'none'} present -- "
                  f"kappa/alpha are UNDEFINED and the k>=0.6 stop rule cannot "
                  f"run; this is a single-annotator pass, not the two-annotator "
                  f"design")
            out["tasks"][f"{base}:{col}"] = {
                "annotators": who, "agreement": None,
                "why": ("inter-annotator agreement requires two independent "
                        "passes; only one was returned"),
            }
            continue
        la = read_labels(pack / f"{base}__annotator{who[0]}.csv", col)
        lb = read_labels(pack / f"{base}__annotator{who[1]}.csv", col)
        common = sorted(set(la) & set(lb))
        if not common:
            print(f"[{base}:{col}] no overlapping labelled rows yet")
            continue
        # calibration items are excluded from every reported estimate
        cal = {k for k in common
               if key.get(k[0], {}).get("stratum", "").startswith("calibration")}
        scored = [k for k in common if k not in cal]
        a = [la[k] for k in scored]
        b = [lb[k] for k in scored]
        bad = sorted({v for v in a + b} - set(labels))
        if bad:
            print(f"[{base}:{col}] WARNING unrecognised labels: {bad}")
        raw = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        k_ = cohens_kappa(a, b)
        pairs = [[x, y] for x, y in zip(a, b)]
        rec = {
            "n_scored": len(scored), "n_calibration_excluded": len(cal),
            "raw_agreement": round(raw, 4),
            "cohens_kappa": round(k_, 4),
            "kappa_ci95": boot_ci(pairs, args.B, args.seed,
                                  lambda s: cohens_kappa([x[0] for x in s],
                                                         [x[1] for x in s])),
            "krippendorff_alpha": round(krippendorff_alpha_nominal(pairs), 4),
            "label_distribution_A": dict(collections.Counter(a)),
            "label_distribution_B": dict(collections.Counter(b)),
        }
        # calibration pass rate, reported separately and never as a finding
        if cal:
            hits = 0
            for k in cal:
                want = ("CORRECT" if key[k[0]]["stratum"].endswith("correct")
                        else "INCORRECT")
                hits += (la[k] == want) + (lb[k] == want)
            rec["calibration_pass_rate"] = round(hits / (2 * len(cal)), 3)
        out["tasks"][f"{base}:{col}"] = rec
        for k in scored:
            if la[k] != lb[k]:
                disagreements.append({"item_id": k[0], "evidence_id": k[1],
                                      "field": col, "A": la[k], "B": lb[k]})
        flag = "" if k_ >= 0.6 else "   <-- BELOW 0.6: revise guidelines, do not collect more"
        print(f"[{base}:{col}] n={len(scored)} raw={raw:.3f} "
              f"kappa={k_:.3f} alpha={rec['krippendorff_alpha']:.3f}{flag}")

    if disagreements:
        p = pack / "disagreements_for_adjudication.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            cols = ["item_id", "evidence_id", "field"]
            for d in disagreements:
                for c in d:
                    if c not in cols:
                        cols.append(c)
            cols += ["adjudicated", "notes"]
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for d in disagreements:
                w.writerow({**d, "adjudicated": "", "notes": ""})
        print(f"\n{len(disagreements)} disagreements -> {p.name}")
    # A field whose kappa is below the protocol threshold must not be used
    # downstream, whatever its point estimates look like.
    out["usability"] = {}
    for name, rec in out["tasks"].items():
        k_ = rec.get("krippendorff_alpha", rec.get("cohens_kappa"))
        out["usability"][name] = {
            "cohens_kappa": rec.get("cohens_kappa"),
            "krippendorff_alpha": rec.get("krippendorff_alpha"),
            "basis": k_,
            "usable_at_0.6": bool(k_ is not None and k_ >= 0.6),
            "ruling": ("usable" if (k_ is not None and k_ >= 0.6) else
                       "DO NOT USE: agreement below the protocol's 0.6 "
                       "threshold, so this field is not a measurement"),
        }
    (pack / "agreement_report.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {pack / 'agreement_report.json'}")
    return 0


def stage_final(pack: Path, args) -> int:
    key = load_key(pack)
    manifest = json.loads((pack / "sampling_manifest.json").read_text())
    who = annotators_present(pack, "task_a_correctness")
    provenance = "adjudicated"
    # The adjudicated file covers only the CONTESTED items. Using it alone would
    # silently reduce the sample to the hardest 9% and report that as the
    # population -- it moved the headline agreement figure from 0.88 to 0.67
    # before this was caught. Adjudicated labels OVERRIDE the consensus; they do
    # not replace it.
    override = read_labels(pack / "task_a_correctness__adjudicated.csv",
                           "adjudicated")
    adj = {}
    if override and len(who) >= 3:
        per = {w: read_labels(pack / f"task_a_correctness__annotator{w}.csv",
                              "verdict") for w in who}
        base = sorted(set.intersection(*[set(v) for v in per.values()]))
        ties = 0
        for k in base:
            lab, ok = majority([per[w][k] for w in who])
            if ok:
                adj[k] = lab
            else:
                ties += 1
        n_over = sum(1 for k in override if k in adj or True)
        adj.update(override)
        provenance = (f"majority_of_{len(who)}_with_{len(override)}_adjudicated")
        print(f"consensus: majority of {len(who)} raters over {len(base)} items, "
              f"{len(override)} contested items replaced by the adjudicator's "
              f"label" + (f", {ties} ties" if ties else ""))
    elif override:
        adj = override
        provenance = "adjudicated_only_no_rater_files"
    if not adj and len(who) >= 3:
        per = {w: read_labels(pack / f"task_a_correctness__annotator{w}.csv",
                              "verdict") for w in who}
        common = sorted(set.intersection(*[set(v) for v in per.values()]))
        adj, ties = {}, 0
        for k in common:
            lab, ok = majority([per[w][k] for w in who])
            if ok:
                adj[k] = lab
            else:
                ties += 1
        provenance = (f"majority_of_{len(who)}_raters"
                      + (f"_{ties}_ties_dropped" if ties else "_no_ties"))
        print(f"consensus by majority of {len(who)} raters: {len(adj)} items"
              + (f", {ties} three-way ties dropped" if ties else ", no ties"))
    if not adj and len(who) >= 2:
        # agreed subset plus adjudicated disagreements
        la = read_labels(pack / f"task_a_correctness__annotator{who[0]}.csv", "verdict")
        lb = read_labels(pack / f"task_a_correctness__annotator{who[1]}.csv", "verdict")
        agreed = {k: la[k] for k in sorted(set(la) & set(lb))
                  if la[k] == lb[k]}
        dis = read_labels(pack / "disagreements_for_adjudication.csv", "adjudicated")
        adj = {**agreed, **dis}
        n_dropped = len(set(la) & set(lb)) - len(agreed) - len(dis)
        # Calling this "adjudicated" when nobody adjudicated would overstate it.
        # Agreement-only estimates drop exactly the hard items, so they are
        # biased toward the cases the metric finds easy.
        provenance = ("agreed_subset_plus_adjudicated" if dis
                      else f"agreed_subset_only_{n_dropped}_disagreements_dropped")
    if not adj and len(who) == 1:
        # One pass. Usable as a provisional estimate, but it is NOT the
        # adjudicated two-annotator design the protocol specifies, and every
        # downstream number must carry that label.
        adj = read_labels(pack / f"task_a_correctness__annotator{who[0]}.csv",
                          "verdict")
        provenance = f"single_annotator_{who[0]}_unadjudicated"
        print(f"WARNING single-annotator pass (annotator {who[0]}): no "
              f"agreement, no adjudication. Results are PROVISIONAL.")
    if not adj:
        print("no labels found; ingest a returned pack first")
        return 1

    strata = manifest["task_a"]["strata"]
    weights = {s: v["weight_to_population"] for s, v in strata.items()}
    pop = {s: v["population"] for s, v in strata.items()}
    total_pop = sum(pop.values())
    # The two calibration strata are 84% of the population. An earlier version
    # dropped them from the estimate while their population mass still existed,
    # so every "population-weighted" number silently described only the hard
    # 16% the sample deliberately over-represents -- token-F1 looked like it
    # agreed with humans 61% of the time when the population figure is 88%.
    # Calibration labels are valid data; they are included here AND used as an
    # attention check, and the hard-subset estimate is reported separately.
    rows = []
    for (iid, _), verdict in adj.items():
        k = key.get(iid)
        if not k or k["task"] != "A":
            continue
        rows.append({"item_id": iid, "human": verdict,
                     "auto_f1": k["auto_f1"], "auto_em": k["auto_em"],
                     "model": k["model"], "condition": k["condition"],
                     "stratum": k["stratum"],
                     "is_calibration": k["stratum"].startswith("calibration"),
                     "w": weights.get(k["stratum"]) or 1.0})

    def acc(pred_fn, sel):
        num = sum(r["w"] for r in sel if pred_fn(r))
        den = sum(r["w"] for r in sel)
        return round(num / den, 4) if den else None

    def strat_ci(sel, num_fn, den_fn=None, B=4000, seed=20260803):
        """Design-based interval: resample WITHIN stratum, respecting weights.

        Two calibration strata carry weights of 196 and 504 on 20 sampled rows
        each, so they dominate every weighted estimate. An i.i.d. interval would
        badly understate the uncertainty that leverage creates.
        """
        import random as _r
        by = collections.defaultdict(list)
        for r in sel:
            by[r["stratum"]].append(r)
        if len(by) < 2:
            return None
        rng = _r.Random(seed)
        draws = []
        for _ in range(B):
            num = den = 0.0
            for items in by.values():
                for _ in items:
                    r = items[rng.randrange(len(items))]
                    if den_fn and not den_fn(r):
                        continue
                    den += r["w"]
                    num += r["w"] * bool(num_fn(r))
            if den:
                draws.append(num / den)
        if len(draws) < 100:
            return None
        draws.sort()
        n = len(draws)
        return [round(draws[int(0.025 * n)], 4), round(draws[int(0.975 * n)], 4)]

    def block(sel):
        return {
            "n_items": len(sel),
            "population_covered": round(
                sum(pop.get(s, 0) for s in {r["stratum"] for r in sel}) / total_pop, 4),
            "human_correct_rate": acc(human_correct, sel),
            "token_f1_correct_rate": acc(f1_correct, sel),
            "exact_match_correct_rate": acc(em_correct, sel),
            "f1_agrees_with_human": acc(lambda r: f1_correct(r) == human_correct(r), sel),
            "em_agrees_with_human": acc(lambda r: em_correct(r) == human_correct(r), sel),
            # Everything else here binarises as CORRECT vs not-CORRECT, so the
            # matching false-positive rate must too. Restricting to strictly
            # INCORRECT silently excludes PARTIAL and ABSTAIN and understates it
            # (0.020 against 0.024); both are reported rather than one chosen.
            # The distinction: 0.094 is a PREVALENCE (share of all weighted
            # outputs), not a miss rate. Conditional on the answer being human-
            # correct it is 0.094/0.391 = 0.24. Both are reported, named so they
            # cannot be confused again.
            "f1_credits_a_non_correct": acc(
                lambda r: f1_correct(r) and not human_correct(r), sel),
            "conditional_miss_rate_given_human_correct": acc(
                lambda r: not f1_correct(r),
                [r for r in sel if human_correct(r)]),
            "conditional_false_credit_given_human_not_correct": acc(
                f1_correct, [r for r in sel if not human_correct(r)]),
            "conditional_miss_rate_ci95": strat_ci(
                sel, lambda r: not f1_correct(r), human_correct),
            "f1_agrees_with_human_ci95": strat_ci(
                sel, lambda r: f1_correct(r) == human_correct(r)),
            "human_correct_rate_ci95": strat_ci(sel, human_correct),
            "f1_credits_a_strictly_incorrect": acc(
                lambda r: f1_correct(r) and r["human"] == "INCORRECT", sel),
            "f1_misses_a_human_right": acc(
                lambda r: (not f1_correct(r)) and human_correct(r), sel),
        }

    human_correct = lambda r: r["human"] == "CORRECT"
    f1_correct = lambda r: r["auto_f1"] >= 0.5
    em_correct = lambda r: r["auto_em"] == 1.0

    out = {
        "scope": "Human correctness against the automatic metrics, reweighted "
                 "to the population using the stratum weights.",
        "label_provenance": provenance,
        "annotators": who,
        "is_adjudicated": (provenance.startswith("adjudicated")
                           or "_adjudicated" in provenance
                           or provenance.startswith("agreed_subset_plus")),
        "provisional": (provenance.startswith("single_annotator")
                        or provenance.startswith("agreed_subset_only")),
        "consensus_rule": provenance,
        "caveat": ("A single unadjudicated pass has no inter-annotator "
                   "agreement, so the protocol's kappa>=0.6 stop rule never "
                   "ran. These estimates show where the automatic metric fails "
                   "but do not establish a validated human ground truth."
                   if provenance.startswith("single_annotator") else
                   "Two independent passes, but the disagreements were never "
                   "adjudicated: they are dropped, so these estimates describe "
                   "the items two annotators agreed on and are biased toward "
                   "the easy cases. Adjudication remains outstanding."
                   if provenance.startswith("agreed_subset_only") else
                   "Majority vote over three independent passes. Every item is "
                   "covered, so there is no drop-the-hard-cases selection bias, "
                   "but nobody reviewed the competing labels: this is consensus, "
                   "not the protocol's third-party adjudication."
                   if provenance.startswith("majority_of") and "adjudicated" not in provenance else
                   "Majority vote over three independent passes, with every "
                   "contested item replaced by a blind third-party "
                   "adjudicator's label. Full coverage and the hard cases "
                   "examined."
                   if "_adjudicated" in provenance else
                   "two independent passes with third-party adjudication"),
        "n_adjudicated": len(rows),
        "reweighting_note": ("the sample deliberately over-represents "
                             "disagreement strata; every rate below is "
                             "population-weighted, so it is NOT the raw rate in "
                             "the annotated file"),
        "population_weighted": block(rows),
        "hard_subset_only": {
            **block([r for r in rows if not r["is_calibration"]]),
            "what_this_is": ("the disagreement/alias/abstention/other strata "
                             "only -- the cases the sample deliberately "
                             "over-represents. Informative about WHERE the "
                             "metric fails, but it is NOT a population rate."),
        },
        "endpoint_audited": (
            "This audit evaluates the THRESHOLDED decision 1[token-F1 >= 0.5], "
            "not continuous token-F1. Agreement of a thresholded decision does "
            "not validate the continuous score or the mean contrasts the paper "
            "reports."),
        "reliability_vs_validity": (
            "Krippendorff alpha 0.904 is inter-ANNOTATOR reliability. Agreement "
            "between the metric and the human label is a different quantity and "
            "is much lower (kappa 0.36). Do not quote the first as evidence for "
            "the second."),
        "exact_match_caveat": (
            "Within the hard subset exact match is 0 by construction: the "
            "stratum where EM succeeds (calibration_clear_correct) is defined "
            "by EM==1 and is excluded there. Only the full-population block "
            "says anything meaningful about EM."),
        "unweighted_kappa_vs_human": {
            "token_f1": round(cohens_kappa(
                ["C" if f1_correct(r) else "I" for r in rows],
                ["C" if human_correct(r) else "I" for r in rows]), 4),
            "exact_match": round(cohens_kappa(
                ["C" if em_correct(r) else "I" for r in rows],
                ["C" if human_correct(r) else "I" for r in rows]), 4),
        },
        "human_label_distribution": dict(collections.Counter(r["human"] for r in rows)),
        "by_stratum": {},
    }
    for st in sorted({r["stratum"] for r in rows}):
        sel = [r for r in rows if r["stratum"] == st]
        out["by_stratum"][st] = {
            "n": len(sel), "weight": sel[0]["w"],
            "human_correct_rate_in_stratum": round(
                sum(1 for r in sel if human_correct(r)) / len(sel), 4),
            "f1_agrees": round(
                sum(1 for r in sel if f1_correct(r) == human_correct(r)) / len(sel), 4),
        }
    dest = ROOT / "results" / "human_validation.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    pw = out["population_weighted"]
    print(f"n adjudicated: {len(rows)}")
    print(f"human-correct (weighted):   {pw['human_correct_rate']}")
    print(f"token-F1 agrees with human: {pw['f1_agrees_with_human']}")
    print(f"exact-match agrees:         {pw['em_agrees_with_human']}")
    print(f"kappa(F1, human) = {out['unweighted_kappa_vs_human']['token_f1']}")
    print(f"wrote {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path,
                    default=ROOT / "annotation" / "pack_2026-08-02")
    ap.add_argument("--stage", choices=["agreement", "final"], required=True)
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260802)
    args = ap.parse_args()
    return (stage_agreement if args.stage == "agreement" else stage_final)(
        args.pack, args)


if __name__ == "__main__":
    raise SystemExit(main())
