#!/usr/bin/env python3
"""Why does retrieval help or hurt? A support-conditioned decomposition (R2, P0).

The paper reports that retrieval helps every generator on Fresh questions and
significantly harms the three strongest on Static ones. It does not say *why*,
because a pooled gain conflates three different failures:

  RETRIEVAL FAILURE   the answer was never in the retrieved context, so the
                      generator had nothing to use.
  SYNTHESIS FAILURE   the answer WAS in the context and the generator still got
                      it wrong.
  OVERRIDE            the generator was right closed-book and wrong with
                      retrieval -- evidence displaced correct parametric
                      knowledge. This is the mechanism the Static-side harm is
                      supposed to be about, and it has never been counted.

Every generation record stores the doc_ids it was shown, so all three are
directly observable: a retrieved context is "answer-bearing" when the gold
answer string appears in one of those documents. That gives a support-conditioned
accuracy -- accuracy restricted to questions where the evidence was actually
present -- which is the quantity R2's third question asks for.

The headline this produces is a 2x2 per model and regime: was the answer
supported, and did the model get it right, closed-book versus with retrieval.

Outputs results/support_conditioned.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import gold_in_text, token_f1                    # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
BOUNDARY_YEAR = 2025
CORRECT = 0.5          # token-F1 above this counts as correct for the 2x2


def quiz_of(qid):
    return qid.split("_")[0]


def load_corpus():
    out = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if not f.is_file():
            continue
        for line in f.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                out.setdefault(d["doc_id"], d.get("text", ""))
    return out


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in
             (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}


def load_gen(model, cond):
    f = PROC / f"generations_{model}" / f"{cond}.jsonl"
    if not f.is_file():
        return None
    out = {}
    for line in f.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["qid"]] = r
    return out


def boot(vals, B, seed):
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean([vals[rng.randrange(len(vals))]
                                     for _ in vals]) for _ in range(B))
    return {"mean": round(statistics.fmean(vals), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "n": len(vals)}


def paired_boot(d, B, seed):
    if len(d) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean([d[rng.randrange(len(d))] for _ in d])
                   for _ in range(B))
    le = sum(1 for x in draws if x <= 0)
    return {"mean": round(statistics.fmean(d), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5),
            "n": len(d)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--condition", default="hybrid")
    ap.add_argument("--output", type=Path, default=RES / "support_conditioned.json")
    args = ap.parse_args()

    qs = load_questions()
    corpus = load_corpus()
    models = sorted(d.name.replace("generations_", "")
                    for d in PROC.glob("generations_*"))

    out = {
        "scope": "Retrieval gains decomposed by whether the retrieved context "
                 "actually contained the answer.",
        "support_definition": "a context is answer-bearing if the gold answer "
                              "string appears in at least one retrieved document",
        "correct_threshold": CORRECT,
        "retrieval_condition": args.condition,
        "models": {},
    }

    print(f"{'model':16s}{'regime':8s}{'support':>9s}{'acc|sup':>9s}"
          f"{'acc|unsup':>11s}{'CB acc':>9s}{'override':>10s}{'rescue':>8s}")
    for m in models:
        cb = load_gen(m, "closed_book")
        rt = load_gen(m, args.condition)
        if not cb or not rt:
            continue
        rec = {}
        for regime in ("static", "fresh"):
            rows = []
            for qid, r in rt.items():
                q = qs.get(qid)
                if q is None or qid not in cb:
                    continue
                is_fresh = q["year"] >= BOUNDARY_YEAR
                if (regime == "fresh") != is_fresh:
                    continue
                gold = q["gold_answer"]
                texts = [corpus.get(d, "") for d in (r.get("doc_ids") or [])]
                supported = any(gold_in_text(gold, t) for t in texts if t)
                a_rt = token_f1(r["prediction"] or "", gold) >= CORRECT
                a_cb = token_f1(cb[qid]["prediction"] or "", gold) >= CORRECT
                rows.append((quiz_of(qid), supported, a_cb, a_rt))
            if len(rows) < 20:
                continue
            n = len(rows)
            sup = [x for x in rows if x[1]]
            uns = [x for x in rows if not x[1]]
            # quiz-clustered means
            def by_quiz(sel, f):
                acc = collections.defaultdict(list)
                for x in sel:
                    acc[x[0]].append(100.0 * f(x))
                return [statistics.fmean(v) for v in acc.values()]
            rec[regime] = {
                "n_questions": n,
                "support_rate": round(len(sup) / n, 4),
                "acc_given_support": boot(by_quiz(sup, lambda x: x[3]), args.B, args.seed) if sup else None,
                "acc_given_no_support": boot(by_quiz(uns, lambda x: x[3]), args.B, args.seed) if uns else None,
                "closed_book_acc": boot(by_quiz(rows, lambda x: x[2]), args.B, args.seed),
                # the two mechanisms, as rates over all questions in the regime
                "override_rate": round(sum(1 for x in rows if x[2] and not x[3]) / n, 4),
                "rescue_rate": round(sum(1 for x in rows if not x[2] and x[3]) / n, 4),
                "override_when_unsupported": round(
                    sum(1 for x in uns if x[2] and not x[3]) / len(uns), 4) if uns else None,
                "override_when_supported": round(
                    sum(1 for x in sup if x[2] and not x[3]) / len(sup), 4) if sup else None,
                # synthesis failure: evidence was there and the model still missed
                "synthesis_failure_rate": round(
                    sum(1 for x in sup if not x[3]) / len(sup), 4) if sup else None,
                # support-conditioned paired gain, the R2 Q3 quantity
                "paired_gain_given_support": paired_boot(
                    [100.0 * (x[3] - x[2]) for x in sup], args.B, args.seed) if sup else None,
            }
            r_ = rec[regime]
            print(f"{m:16s}{regime:8s}{r_['support_rate']:9.3f}"
                  f"{(r_['acc_given_support'] or {}).get('mean', float('nan')):9.2f}"
                  f"{(r_['acc_given_no_support'] or {}).get('mean', float('nan')):11.2f}"
                  f"{r_['closed_book_acc']['mean']:9.2f}"
                  f"{r_['override_rate']:10.3f}{r_['rescue_rate']:8.3f}")
        if rec:
            out["models"][m] = rec

    # ---- what the mechanism decomposition says -----------------------------
    strong = ["mistral_large", "sonnet", "gpt4omini"]
    ov_static, ov_fresh = [], []
    for m, rec in out["models"].items():
        if m in strong:
            if "static" in rec:
                ov_static.append(rec["static"]["override_rate"])
            if "fresh" in rec:
                ov_fresh.append(rec["fresh"]["override_rate"])
    sup_gain_static = [rec["static"]["paired_gain_given_support"]["mean"]
                       for rec in out["models"].values()
                       if rec.get("static", {}).get("paired_gain_given_support")]
    out["mechanism"] = {
        "strong_models": strong,
        "mean_override_rate_static_strong": round(statistics.fmean(ov_static), 4) if ov_static else None,
        "mean_override_rate_fresh_strong": round(statistics.fmean(ov_fresh), 4) if ov_fresh else None,
        "mean_support_conditioned_gain_static": round(statistics.fmean(sup_gain_static), 3) if sup_gain_static else None,
        "reading": (
            "OVERRIDE is the rate at which a model that was right closed-book "
            "becomes wrong once evidence is supplied. If Static-side harm is "
            "evidence displacing correct parametric knowledge, the override "
            "rate should be high on Static and low on Fresh for the strong "
            "models -- and the harm should persist even when the evidence IS "
            "answer-bearing, which is what the support-conditioned gain tests."),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nmean override rate (strong models): static "
          f"{out['mechanism']['mean_override_rate_static_strong']}, fresh "
          f"{out['mechanism']['mean_override_rate_fresh_strong']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
