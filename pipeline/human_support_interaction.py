#!/usr/bin/env python3
"""The support interaction estimated against HUMAN passage labels.

The paper's L3 result is stratified by two automatic support proxies, and the
paper itself reports that neither is reliable: exact-string containment reaches
kappa=0.56 against human passage labels and entailment 0.65. The obvious
follow-up is whether the interaction can be estimated directly against the human
labels, with uncertainty clustered. This does that.

  Delta_sup(human) = Delta_RAG | human-supported  -  Delta_RAG | human-unsupported

A question counts as human-supported when at least one of its labelled passages
was called YES (a stricter YES-or-PARTIAL variant is reported alongside).

LABEL RESOLUTION. Passage labels are the majority of three independent
annotators, with the BLIND ADJUDICATOR SUBSTITUTED on the 168 contested passages
recorded in annotation/adjudication/pack/adjudicate_supports.csv. This matches
pipeline/support_label_validation.py and the appendix description. An earlier
version of this script used the bare majority and so disagreed with both.

DESIGN WEIGHTS. Task B is a stratified probability sample, 75 items from each
regime x auto_supported cell, with population weights 22.8, 25.68, 25.2 and
31.68 recorded in the pack manifest. Sampling depends on the AUTOMATIC support
variable while the analysis conditions on the correlated HUMAN one, so an
unweighted conditional mean is not the population interaction the prose
describes. We therefore report the design-weighted estimate as primary and carry
the unweighted one alongside. Every one of the 254 unique questions falls in
exactly one cell (verified at run time), so the 41 questions sampled more than
once inherit a single unambiguous weight.

THREE LIMITS, none of them removable here:
  1. Task B showed up to three of the top five passages, so a question marked
     unsupported may have been supported by a passage nobody saw. Whether that
     attenuates the interaction depends on the errors being nondifferential
     across the two arms, which we have not established, so we report it as an
     open direction of bias rather than as evidence the estimate is conservative.
  2. 254 questions carry labels, 127 per slice, so intervals are wide. This is a
     validation of the automatic result's direction, not a replacement for it.
  3. Labels were collected on the shipped (unmasked) rankings, so this speaks to
     the unmasked contrast.

Outputs results/human_support_interaction.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.dont_write_bytecode = True
from metrics import token_f1                                   # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
PACK = ROOT / "annotation/pack_2026-08-02"
ADJ = ROOT / "annotation/adjudication/pack/adjudicate_supports.csv"
BOUNDARY = 2025
# The human support label is a property of the QUESTION (did a shown passage
# support the answer?), not of the generator, so it applies to any generator with
# both a closed-book and a Hybrid cache. Llama-3-8B was missing from an earlier
# version of this list for no reason other than oversight.
MODELS = ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
          "mistral_large", "", "sonnet", "gpt4omini"]       # "" is Haiku


def quiz_of(qid):
    return qid.split("_")[0]


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()
             if l.strip())}


def load_gen(model, cond):
    # "" is Haiku, which lives in the bare generations/ directory.
    f = PROC / (f"generations_{model}" if model else "generations") / f"{cond}.jsonl"
    if not f.is_file():
        return {}
    return {r["qid"]: (r["prediction"] or "") for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


CELL_WEIGHT = {}          # (regime, auto_supported) -> weight_to_population


def load_cell_weights():
    """Design weights from the pack manifest, keyed the way the KEY records cells."""
    man = json.loads((PACK / "sampling_manifest.json").read_text())
    cells = (man.get("task_b") or man)["cells"]
    out = {}
    for name, rec in cells.items():
        regime, sup = name.split("|")
        out[(regime, sup.split("=")[1] == "True")] = rec["weight_to_population"]
    return out


def human_question_labels():
    """qid -> label tables, sampling weights and resolution provenance.

    Returns (strict, loose, weight, n_passages, prov) where the label of a
    passage is the majority of three annotators unless it was contested, in
    which case the blind adjudicator's label replaces it.
    """
    key = {}
    for line in (PACK / "item_metadata.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("task") == "B":
                key[d["item_id"]] = d

    # one ballot per (item_id, evidence_id): the same item appears once per passage
    votes = collections.defaultdict(list)
    for letter in "ABC":
        f = PACK / f"task_b_evidence__annotator{letter}.csv"
        if not f.is_file():
            continue
        for row in csv.DictReader(f.open(encoding="utf-8")):
            lab = (row.get("supports") or "").strip().upper()
            if lab:
                votes[(row["item_id"], row["evidence_id"])].append(lab)

    # adjudicated labels replace the majority on contested passages
    adjudicated = {}
    if ADJ.is_file():
        for row in csv.DictReader(ADJ.open(encoding="utf-8")):
            if (row.get("field") or "").strip() != "supports":
                continue
            lab = (row.get("adjudicated") or "").strip().upper()
            if lab:
                adjudicated[(row["item_id"], row["evidence_id"])] = lab

    cell_w = load_cell_weights()
    strict, loose = collections.defaultdict(bool), collections.defaultdict(bool)
    weight, qid_cells = {}, collections.defaultdict(set)
    n_passages = n_adj_applied = n_adj_changed = 0
    for pk, vs in votes.items():
        k = key.get(pk[0])
        if not k:
            continue
        maj = collections.Counter(vs).most_common(1)[0][0]
        lab = adjudicated.get(pk, maj)
        if pk in adjudicated:
            n_adj_applied += 1
            n_adj_changed += (lab != maj)
        n_passages += 1
        qid = k["qid"]
        cell = (k["regime"], bool(k["auto_supported"]))
        qid_cells[qid].add(cell)
        weight[qid] = cell_w[cell]
        strict[qid] |= (lab == "YES")
        loose[qid] |= (lab in ("YES", "PARTIAL"))

    # The weight is only unambiguous because no question straddles two cells.
    multi = sorted(q for q, c in qid_cells.items() if len(c) > 1)
    if multi:
        raise SystemExit(
            f"{len(multi)} questions fall in more than one sampling cell "
            f"(e.g. {multi[:3]}), so a single weight per question is not defined. "
            "Aggregate at the sampled-row level instead.")

    prov = {
        "label_rule": "majority of three annotators, adjudicator substituted on "
                      "contested passages",
        "n_contested_passages_adjudicated": n_adj_applied,
        "n_passages_whose_label_changed": n_adj_changed,
        "n_questions_sampled_more_than_once": sum(
            1 for q in qid_cells if sum(1 for k in key.values() if k["qid"] == q) > 1),
        "questions_straddling_cells": 0,
        "cell_weights": {f"{r}|supported={s}": w for (r, s), w in cell_w.items()},
    }
    return dict(strict), dict(loose), dict(weight), n_passages, prov


def _wmean(pairs):
    """Inverse-inclusion-weighted mean of (value, weight)."""
    w = sum(x[1] for x in pairs)
    return sum(v * x for v, x in pairs) / w if w else None


def interaction_ci(units, B, seed, weighted=True):
    """Quiz-clustered bootstrap of (mean supported - mean unsupported).

    Each observation carries its stratum's inverse-inclusion weight. Setting
    weighted=False recovers the unweighted convenience-sample estimate, which we
    report only as a comparison.
    """
    cl = list(units.values())
    sup = [x for a, _ in cl for x in a]
    uns = [x for _, b in cl for x in b]
    if len(sup) < 8 or len(uns) < 8 or len(cl) < 3:
        return None
    if not weighted:
        sup = [(v, 1.0) for v, _ in sup]
        uns = [(v, 1.0) for v, _ in uns]
    obs = _wmean(sup) - _wmean(uns)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        sa, sb = [], []
        for _ in cl:
            a, b = cl[rng.randrange(len(cl))]
            sa.extend(a if weighted else [(v, 1.0) for v, _ in a])
            sb.extend(b if weighted else [(v, 1.0) for v, _ in b])
        if sa and sb:
            draws.append(_wmean(sa) - _wmean(sb))
    if len(draws) < B // 2:
        return None
    draws.sort()
    n = len(draws)
    le = sum(1 for v in draws if v <= 0)
    return {
        "interaction": round(obs, 3),
        "ci95": [round(draws[int(0.025 * n)], 3), round(draws[int(0.975 * n)], 3)],
        "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 5),
        "gain_supported": round(_wmean(sup), 3),
        "gain_unsupported": round(_wmean(uns), 3),
        "n_supported": len(sup), "n_unsupported": len(uns), "n_quizzes": len(cl),
        "estimator": "design-weighted" if weighted else "unweighted",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--output", type=Path,
                    default=RES / "human_support_interaction.json")
    args = ap.parse_args()

    qs = load_questions()
    strict, loose, qweight, n_passages, prov = human_question_labels()
    labelled = sorted(set(strict) & set(qs))

    out = {
        "scope": "Support interaction stratified by HUMAN passage labels rather "
                 "than an automatic proxy, quiz-clustered.",
        "estimand": "Delta_RAG | supported  -  Delta_RAG | unsupported, "
                    "Hybrid vs closed-book, paired within question",
        "label": "majority of three annotators over the passages each question "
                 "actually showed, adjudicator substituted where contested",
        "label_resolution": prov,
        "primary_estimator": "design-weighted (inverse inclusion probability of "
                             "the regime x auto-support sampling cell)",
        "caveats": {
            "partial_passage_coverage":
                "Task B showed up to three of the top five passages, so an "
                "unsupported label can mean 'no support among the passages shown'. "
                "The direction of the resulting bias is not established, since "
                "attenuation would require the misclassification to be "
                "nondifferential across arms.",
            "power": f"{len(labelled)} labelled questions in total",
            "unmasked": "labels were collected on the shipped (unmasked) rankings",
        },
        "n_labelled_passages": n_passages,
        "n_labelled_questions": len(labelled),
        "slice_counts": {
            "static": sum(1 for q in labelled if qs[q]["year"] < BOUNDARY),
            "fresh": sum(1 for q in labelled if qs[q]["year"] >= BOUNDARY),
        },
        "per_model": {},
    }

    for m in MODELS:
        cb, hy = load_gen(m, "closed_book"), load_gen(m, "hybrid")
        if not cb or not hy:
            continue
        rec = {}
        for label_name, table in (("strict_yes", strict), ("yes_or_partial", loose)):
            for regime in ("static", "fresh", "all"):
                units = collections.defaultdict(lambda: ([], []))
                for q in labelled:
                    if q not in cb or q not in hy:
                        continue
                    if regime != "all" and ((qs[q]["year"] >= BOUNDARY) !=
                                            (regime == "fresh")):
                        continue
                    gain = 100.0 * (token_f1(hy[q], qs[q]["gold_answer"])
                                    - token_f1(cb[q], qs[q]["gold_answer"]))
                    units[quiz_of(q)][0 if table.get(q) else 1].append(
                        (gain, qweight[q]))
                r = interaction_ci(units, args.B, args.seed, weighted=True)
                if r:
                    u = interaction_ci(units, args.B, args.seed, weighted=False)
                    if u:
                        r["unweighted"] = {k: u[k] for k in
                                           ("interaction", "ci95", "p_plus_one")}
                    rec.setdefault(label_name, {})[regime] = r
        if rec:
            out["per_model"][m] = rec

    # agreement between the human question-level variable and the automatic one
    auto = {}
    for line in (PACK / "item_metadata.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("task") == "B" and "auto_supported" in d:
                auto.setdefault(d["qid"], False)
                auto[d["qid"]] |= bool(d["auto_supported"])
    both = [q for q in labelled if q in auto]
    if both:
        agree = sum(1 for q in both if auto[q] == strict.get(q, False))
        out["question_level_label_agreement"] = {
            "n": len(both), "rate": round(agree / len(both), 4),
            "note": "automatic question-level support against the human one, on "
                    "the questions carrying labels",
        }

    n_sig = 0
    print(f"{'model':14s}{'inter':>9s}{'ci95':>20s}{'p':>9s}   (Static, strict YES)")
    for m, rec in out["per_model"].items():
        r = (rec.get("strict_yes") or {}).get("static")
        if not r:
            continue
        n_sig += r["ci95"][0] > 0
        print(f"{m:14s}{r['interaction']:9.2f}"
              f"{str(r['ci95']):>20s}{r['p_plus_one']:9.4f}")
    out["n_models_static_interaction_excludes_zero"] = n_sig
    print(f"\n{n_sig} of {len(out['per_model'])} generators keep a positive "
          f"Static interaction under human labels")
    if "question_level_label_agreement" in out:
        print("question-level auto-vs-human agreement: "
              f"{out['question_level_label_agreement']['rate']:.3f}")

    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
