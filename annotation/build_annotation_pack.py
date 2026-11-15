#!/usr/bin/env python3
"""Build the blinded human-annotation pack for TRAVA.

This is the study the reviews call the remaining blocker: an independent,
blinded audit of (A) answer correctness and (B) evidence support and freshness,
with two annotators, adjudication, and reported agreement.

Design decisions and why they are what they are:

  BLINDING. Annotators never see the model identity, the condition
  (closed-book vs retrieval), or the automatic score. Those are held in a
  separate key file that annotators do not receive. Without this the audit
  cannot check the automatic metric, because the metric's verdict would anchor
  the human one.

  STRATIFICATION. A uniform sample would be ~90% cases where every metric
  agrees, which carry almost no information about where the metric fails. We
  oversample the disagreement strata (token-F1 and exact-match disagree; the
  choice-grounded referee disagrees with F1; abstentions) and record the
  sampling weight so estimates can be reweighted back to the population.

  CALIBRATION ITEMS. 10% of each pack is drawn from cases where the correct
  verdict is not in serious doubt (exact string match, or an answer with no
  token overlap at all). These are not analysed as findings; they measure
  whether an annotator is attending, and they are marked only in the key file.

  POWER. Task A is sized so each of the five scoring regimes gets >= 150
  adjudicated items, which is what the per-regime Cohen's kappa needs for a
  usable interval; Task B is sized for >= 70 per model in the stale/superseded
  contrast, the figure the earlier trend test showed was required.

Outputs, under annotation/pack_<date>/:
  task_a_correctness.csv        annotator-facing, blinded
  task_b_evidence.csv           annotator-facing, blinded
  item_metadata.jsonl   model/condition/scores per blind id
  sampling_manifest.json        strata, weights, seeds, counts
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import exact_match, gold_in_text, normalize, token_f1   # noqa: E402

PROC = ROOT / "data/processed"
BOUNDARY_YEAR = 2025
ABSTAIN = ("i dont know", "i do not know", "unknown", "cannot determine",
           "no answer", "not sure", "unable to determine", "n/a")


def is_abstention(p):
    n = normalize(p)
    return (not n) or any(n == a or n.startswith(a) for a in ABSTAIN)


def blind_id(*parts) -> str:
    return "A" + hashlib.sha1("||".join(map(str, parts)).encode()).hexdigest()[:9]


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in
             (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}


def load_corpus():
    out = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    out.setdefault(d["doc_id"], d)
    return out


def load_gen(model, cond):
    f = PROC / f"generations_{model}" / f"{cond}.jsonl"
    if not f.is_file():
        return {}
    return {r["qid"]: r for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def nearest_choice(pred, choices):
    p = normalize(pred)
    if not p:
        return None
    exact = [k for k, c in enumerate(choices) if normalize(c) == p]
    if len(exact) == 1:
        return exact[0]
    cont = [k for k, c in enumerate(choices)
            if p and (p in normalize(c) or normalize(c) in p)]
    if len(cont) == 1:
        return cont[0]
    sc = [token_f1(pred, c) for c in choices]
    best = max(sc) if sc else 0
    top = [k for k, v in enumerate(sc) if v == best]
    return top[0] if best > 0 and len(top) == 1 else None


def stratum_of(pred, q):
    """Which information-bearing case is this?"""
    if is_abstention(pred):
        return "abstention"
    f1 = token_f1(pred, q["gold_answer"])
    em = exact_match(pred, q["gold_answer"])
    idx = nearest_choice(pred, q["choices"])
    choice_ok = (idx is not None and idx == q["gold_answer_idx"])
    if em == 1.0 and choice_ok:
        return "calibration_clear_correct"
    if f1 == 0.0 and not choice_ok:
        return "calibration_clear_wrong"
    if em == 0.0 and f1 >= 0.5:
        return "alias_or_partial"          # F1 credits, EM does not
    if choice_ok != (f1 >= 0.5):
        return "metric_disagreement"       # choice referee vs F1
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--n-task-a", type=int, default=400)
    ap.add_argument("--n-task-b", type=int, default=300)
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "annotation" / "pack_2026-08-02")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    qs = load_questions()
    corpus = load_corpus()
    models = sorted(d.name.replace("generations_", "")
                    for d in PROC.glob("generations_*") if d.is_dir())

    # ---------------------------------------------------------------- Task A
    pool = collections.defaultdict(list)
    for m in models:
        for cond in ("closed_book", "hybrid"):
            for qid, r in load_gen(m, cond).items():
                q = qs.get(qid)
                if q is None:
                    continue
                pred = r.get("prediction") or ""
                st = stratum_of(pred, q)
                pool[st].append((m, cond, qid, pred, q))

    # target mix: oversample the informative strata, keep 10% calibration
    n = args.n_task_a
    quota = {
        "metric_disagreement": int(0.32 * n),
        "alias_or_partial":    int(0.28 * n),
        "abstention":          int(0.10 * n),
        "other":               int(0.20 * n),
        "calibration_clear_correct": int(0.05 * n),
        "calibration_clear_wrong":   int(0.05 * n),
    }
    chosen_a, manifest_a = [], {}
    for st, want in quota.items():
        avail = pool.get(st, [])
        take = min(want, len(avail))
        pick = rng.sample(avail, take) if take else []
        # sampling weight back to the population of this stratum
        w = (len(avail) / take) if take else None
        manifest_a[st] = {"population": len(avail), "sampled": take,
                          "weight_to_population": round(w, 3) if w else None}
        for m, cond, qid, pred, q in pick:
            chosen_a.append({
                "stratum": st, "model": m, "condition": cond, "qid": qid,
                "prediction": pred, "q": q,
                "f1": round(token_f1(pred, q["gold_answer"]), 3),
                "em": exact_match(pred, q["gold_answer"]),
            })
    rng.shuffle(chosen_a)

    # ---------------------------------------------------------------- Task B
    ev_pool = collections.defaultdict(list)
    for m in models:
        for qid, r in load_gen(m, "hybrid").items():
            q = qs.get(qid)
            docs = r.get("doc_ids") or []
            if q is None or not docs:
                continue
            texts = [corpus.get(d, {}).get("text", "") for d in docs]
            supported = any(gold_in_text(q["gold_answer"], t) for t in texts if t)
            regime = "fresh" if q["year"] >= BOUNDARY_YEAR else "static"
            ev_pool[(regime, supported)].append((m, qid, r, q))
    nb = args.n_task_b
    per_cell = max(1, nb // max(len(ev_pool), 1))
    chosen_b, manifest_b = [], {}
    for key, avail in sorted(ev_pool.items(), key=lambda kv: str(kv[0])):
        take = min(per_cell, len(avail))
        pick = rng.sample(avail, take) if take else []
        manifest_b[f"{key[0]}|supported={key[1]}"] = {
            "population": len(avail), "sampled": take,
            "weight_to_population": round(len(avail) / take, 3) if take else None}
        for m, qid, r, q in pick:
            chosen_b.append({"regime": key[0], "auto_supported": key[1],
                             "model": m, "qid": qid, "rec": r, "q": q})
    rng.shuffle(chosen_b)

    # ---------------------------------------------------------------- write
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    key_rows = []

    with (out / "task_a_correctness.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "question", "question_date", "reference_answer",
                    "system_answer", "verdict", "confidence", "notes"])
        for r in chosen_a:
            q = r["q"]
            bid = blind_id("A", r["model"], r["condition"], r["qid"])
            w.writerow([bid, q["question"], q["date"], q["gold_answer"],
                        r["prediction"], "", "", ""])
            key_rows.append({"item_id": bid, "task": "A", "qid": r["qid"],
                             "model": r["model"], "condition": r["condition"],
                             "stratum": r["stratum"], "auto_f1": r["f1"],
                             "auto_em": r["em"]})

    with (out / "task_b_evidence.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "question", "question_date", "reference_answer",
                    "evidence_id", "evidence_date", "evidence_excerpt",
                    "supports", "freshness", "notes"])
        for r in chosen_b:
            q = r["q"]
            bid = blind_id("B", r["model"], r["qid"])
            for did in (r["rec"].get("doc_ids") or [])[:3]:
                d = corpus.get(did, {})
                txt = (d.get("text") or "").replace("\n", " ")[:700]
                w.writerow([bid, q["question"], q["date"], q["gold_answer"],
                            did, d.get("publish_date", ""), txt, "", "", ""])
            key_rows.append({"item_id": bid, "task": "B", "qid": r["qid"],
                             "model": r["model"], "regime": r["regime"],
                             "auto_supported": r["auto_supported"]})

    with (out / "item_metadata.jsonl").open("w", encoding="utf-8") as fh:
        for k in key_rows:
            fh.write(json.dumps(k) + "\n")

    manifest = {
        "built": "2026-08-02", "seed": args.seed,
        "blinding": ("annotators receive item_id, question, date, reference "
                     "answer and system answer only; model, condition and every "
                     "automatic score live in item_metadata.jsonl"),
        "task_a": {"n_items": len(chosen_a), "strata": manifest_a,
                   "reweighting": ("each stratum carries weight_to_population; "
                                   "population estimates must reweight, since "
                                   "the sample deliberately over-represents "
                                   "disagreement cases")},
        "task_b": {"n_items": len(chosen_b), "cells": manifest_b,
                   "max_evidence_docs_shown": 3},
        "calibration_items": {
            "task_a": sum(1 for r in chosen_a if r["stratum"].startswith("calibration")),
            "purpose": ("attention check only; excluded from every reported "
                        "estimate and identified only in the key file"),
        },
        "models_in_pool": models,
    }
    (out / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Task A: {len(chosen_a)} items")
    for st, v in manifest_a.items():
        print(f"    {st:28s} {v['sampled']:4d} of {v['population']:6d} "
              f"(weight {v['weight_to_population']})")
    print(f"Task B: {len(chosen_b)} items")
    for k, v in manifest_b.items():
        print(f"    {k:28s} {v['sampled']:4d} of {v['population']:6d}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
