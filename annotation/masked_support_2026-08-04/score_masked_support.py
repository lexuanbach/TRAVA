#!/usr/bin/env python3
"""Score the masked-context support pack, exactly as pre-registered.

Two stages, deliberately separated so that agreement is inspected before any
adjudication and before any estimate exists:

  --stage agreement   three-way agreement, Krippendorff's alpha, and the
                      contested rows written out for a blind adjudicator.
                      HALTS if alpha < 0.6, per PREREGISTRATION.md.
  --stage final       population-weighted estimates and the joint masked+human
                      Layer-3 interaction.

Nothing here reports an unweighted mean as a population quantity: the pack draws
its four strata equally, so every population number is reweighted by the
inclusion weights in sampling_manifest.json.

Writes results/masked_human_support.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.dont_write_bytecode = True
from metrics import token_f1                                   # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
LABELS = ("YES", "PARTIAL", "NO", "UNCLEAR")
ALPHA_FLOOR = 0.6
EQUIV_MARGIN = 5.0            # F1, fixed in the pre-registration


def quiz_of(qid):
    return qid.split("_")[0]


def load_key():
    return {r["item_id"]: r for r in
            (json.loads(l) for l in
             (HERE / "item_metadata.jsonl").read_text().splitlines()
             if l.strip())}


def load_annotations():
    """item_id -> {annotator: label}, from whatever passes have been returned."""
    got = collections.defaultdict(dict)
    for f in sorted(HERE.glob("task_support_masked__annotator?.csv")):
        who = f.stem[-1]
        for row in csv.DictReader(f.open(encoding="utf-8")):
            lab = (row.get("supports") or "").strip().upper()
            if lab:
                got[row["item_id"]][who] = lab
    return got


def krippendorff_nominal(units):
    """Alpha for nominal data over units with 2+ coders."""
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    Do_num = Do_den = 0
    counts = collections.Counter()
    for u in units:
        m = len(u)
        counts.update(u)
        for a, b in itertools.permutations(u, 2):
            Do_num += (a != b)
        Do_den += m - 1
    if Do_den == 0:
        return None
    Do = Do_num / Do_den
    n = sum(counts.values())
    if n < 2:
        return None
    De = 1 - sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    return None if De == 0 else round(1 - Do / De, 4)


def majority(labels):
    c = collections.Counter(labels)
    top, n = c.most_common(1)[0]
    return top if n > len(labels) / 2 else None


def clustered_interaction(units, B, seed):
    """Quiz-clustered bootstrap of (mean supported gain - mean unsupported gain)."""
    cl = list(units.values())
    sup = [v for a, _ in cl for v in a]
    uns = [v for _, b in cl for v in b]
    if len(sup) < 5 or len(uns) < 5 or len(cl) < 3:
        return None
    obs = statistics.fmean(sup) - statistics.fmean(uns)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        sa, sb = [], []
        for _ in cl:
            a, b = cl[rng.randrange(len(cl))]
            sa.extend(a); sb.extend(b)
        if sa and sb:
            draws.append(statistics.fmean(sa) - statistics.fmean(sb))
    if len(draws) < B // 2:
        return None
    draws.sort()
    n = len(draws)
    le = sum(1 for v in draws if v <= 0)
    lo, hi = draws[int(0.025 * n)], draws[int(0.975 * n)]
    return {
        "interaction": round(obs, 3),
        "ci95": [round(lo, 3), round(hi, 3)],
        "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 5),
        "n_supported": len(sup), "n_unsupported": len(uns), "n_quizzes": len(cl),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "within_equivalence_margin": bool(abs(lo) <= EQUIV_MARGIN
                                          and abs(hi) <= EQUIV_MARGIN),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["agreement", "final"], required=True)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--threshold", choices=["yes", "yes_or_partial"], default="yes")
    ap.add_argument("--unadjudicated", choices=["require", "drop", "unclear"],
                    default="require",
                    help="what to do with contested rows that no adjudicator "
                         "resolved: refuse to score (default), drop them and "
                         "report the exclusion, or treat them as UNCLEAR")
    ap.add_argument("--output", type=Path, default=RES / "masked_human_support.json")
    args = ap.parse_args()

    key = load_key()
    got = load_annotations()
    if not got:
        print("No returned annotator files found in\n  " + str(HERE) +
              "\nExpected task_support_masked__annotatorA.csv (and B, C).")
        return 1

    manifest = json.loads((HERE / "sampling_manifest.json").read_text())
    weights = {s: v["weight"] for s, v in manifest["strata"].items()}

    units = [list(v.values()) for v in got.values() if len(v) >= 2]
    alpha = krippendorff_nominal(units)
    n_full = sum(1 for v in got.values() if len(v) >= 3)
    print(f"{len(got)} rows with labels, {n_full} with all three passes")
    print(f"Krippendorff alpha (nominal, supports) = {alpha}")

    if args.stage == "agreement":
        contested = [iid for iid, v in got.items()
                     if len(v) >= 3 and majority(list(v.values())) is None]
        out = HERE / "disagreements_for_adjudication.csv"
        src = {r["item_id"]: r for r in
               csv.DictReader((HERE / "task_support_masked.csv").open(encoding="utf-8"))}
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["item_id", "question", "question_date", "reference_answer",
                        "evidence_date", "evidence_excerpt", "adjudicated", "reason"])
            for iid in contested:
                r = src.get(iid, {})
                w.writerow([iid, r.get("question", ""), r.get("question_date", ""),
                            r.get("reference_answer", ""), r.get("evidence_date", ""),
                            r.get("evidence_excerpt", ""), "", ""])
        print(f"{len(contested)} contested rows written to {out.name}")
        print("The adjudicator must not see the vote counts.")
        if alpha is not None and alpha < ALPHA_FLOOR:
            print(f"\nSTOP: alpha {alpha} is below the pre-registered floor "
                  f"{ALPHA_FLOOR}. Revise the guidelines and re-pilot. Do not "
                  f"collect more rows and do not report an estimate.")
            return 2
        print("\nAgreement clears the floor. Adjudicate, then run --stage final.")
        return 0

    # ---------------- final ----------------
    adj = {}
    f = HERE / "disagreements_for_adjudication.csv"
    if f.is_file():
        for row in csv.DictReader(f.open(encoding="utf-8")):
            lab = (row.get("adjudicated") or "").strip().upper()
            if lab:
                adj[row["item_id"]] = lab

    # Contested rows that nobody adjudicated must be handled explicitly. Silently
    # letting them fall out of the dict is how an undisclosed exclusion happens.
    passage_label, unresolved = {}, []
    for iid, v in got.items():
        lab = adj.get(iid) or majority(list(v.values()))
        if lab:
            passage_label[iid] = lab
        else:
            unresolved.append(iid)

    if unresolved and args.unadjudicated == "require":
        print(f"\n{len(unresolved)} contested rows have no adjudicated label.")
        print("Adjudicate them (see ADJUDICATOR_GUIDE.md), or re-run with")
        print("  --unadjudicated drop      exclude them and report the exclusion")
        print("  --unadjudicated unclear   count them as UNCLEAR (not supported)")
        return 3
    if unresolved and args.unadjudicated == "unclear":
        for iid in unresolved:
            passage_label[iid] = "UNCLEAR"

    supported_vals = ("YES",) if args.threshold == "yes" else ("YES", "PARTIAL")

    # question-level: supported when ANY shown passage is labelled supported
    q_sup, q_stratum, q_slice = {}, {}, {}
    for iid, lab in passage_label.items():
        k = key.get(iid)
        if not k:
            continue
        qid = k["qid"]
        q_sup[qid] = q_sup.get(qid, False) or (lab in supported_vals)
        q_stratum[qid] = k["stratum"]
        q_slice[qid] = k["slice"]

    qs = {d["qid"]: d for d in (json.loads(l) for l in
          (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}

    def gens(model, cond):
        # "" is Haiku, which lives in the bare generations/ directory.
        p = PROC / (f"generations_{model}" if model else "generations") / f"{cond}.jsonl"
        if not p.is_file():
            return {}
        return {r["qid"]: (r["prediction"] or "") for r in
                (json.loads(l) for l in p.read_text().splitlines() if l.strip())}

    out = {
        "scope": "Human support labels on MASKED contexts. This is the joint "
                 "mask + human estimate the paper previously could not make.",
        "preregistration": "annotation/masked_support_2026-08-04/PREREGISTRATION.md",
        "support_threshold": args.threshold,
        "krippendorff_alpha": alpha,
        "n_passages_labelled": len(passage_label),
        "n_questions": len(q_sup),
        "n_adjudicated": len(adj),
        "unresolved_rows": len(unresolved),
        "unresolved_policy": args.unadjudicated,
        "unresolved_share_of_sample": (round(len(unresolved) / len(got), 5)
                                       if got else None),
        "unresolved_note": (
            "Contested rows with no adjudicated label. Under 'drop' they are "
            "excluded from every estimate and this count is the exclusion the "
            "paper must state. Under 'unclear' they are counted as UNCLEAR, "
            "which does not count as supported."),
        "weighting": "support rate: inverse inclusion probability on 200 stratified questions; interactions: unweighted sample-level on 210 questions",
        "n_population_questions": sum(q_stratum[q] in weights for q in q_sup),
        "n_consistency_questions": sum(q_stratum[q] == "consistency_recheck" for q in q_sup),
        "equivalence_margin_f1": EQUIV_MARGIN,
        "weighted_support_rate": None,
        "per_model": {},
    }

    # population-weighted support rate
    num = den = 0.0
    for qid, sup in q_sup.items():
        if q_stratum.get(qid) not in weights:
            continue  # consistency cases have no population inclusion probability
        w = weights[q_stratum[qid]]
        num += w * bool(sup); den += w
    if den:
        out["weighted_support_rate"] = round(num / den, 4)

    models = ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
              "mistral_large", "", "sonnet", "gpt4omini"]
    for m in models:
        cb, ma = gens(m, "closed_book"), gens(m, "hybrid_masked")
        if not cb or not ma:
            continue
        rec = {}
        for regime in ("static", "fresh"):
            units = collections.defaultdict(lambda: ([], []))
            for qid, sup in q_sup.items():
                if q_slice.get(qid) != regime or qid not in cb or qid not in ma:
                    continue
                gold = qs[qid]["gold_answer"]
                gain = 100.0 * (token_f1(ma[qid], gold) - token_f1(cb[qid], gold))
                units[quiz_of(qid)][0 if sup else 1].append(gain)
            r = clustered_interaction(units, args.B, args.seed)
            if r:
                rec[regime] = r
        if rec:
            out["per_model"][m or "haiku"] = rec

    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    if unresolved:
        print(f"\n{len(unresolved)} contested rows unresolved "
              f"({len(unresolved) / len(got):.2%} of the sample), policy="
              f"{args.unadjudicated}. State this exclusion in the paper.")
    print(f"\nweighted support rate: {out['weighted_support_rate']}")
    print(f"{'model':14s}{'interaction':>13s}{'ci95':>22s}{'p':>9s}  (Static)")
    n_sig = 0
    for m, rec in out["per_model"].items():
        r = rec.get("static")
        if not r:
            continue
        n_sig += r["excludes_zero"]
        print(f"{m:14s}{r['interaction']:13.2f}{str(r['ci95']):>22s}"
              f"{r['p_plus_one']:9.4f}")
    print(f"\n{n_sig} of {len(out['per_model'])} exclude zero under the JOINT "
          f"mask + human estimate")
    print("If this is 0, the joint claim is not supported and the paper says so "
          "in the main text, per the pre-registration.")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
