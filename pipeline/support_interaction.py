#!/usr/bin/env python3
"""The direct support x retrieval interaction, not two subset signs.

`support_conditioned.py` reports the retrieval gain inside the supported subset
and the retrieval gain inside the unsupported subset, and the paper read a
non-negative supported-subset gain as showing that harm is concentrated in the
unsupported subset. The review is right that this does not follow: two separate
within-subset point estimates, each with its own interval, are not a test that
the two differ. The quantity the claim needs is the DIFFERENCE of the two gains.

  gain(S)   = mean over supported questions of   (RAG - closed-book)
  gain(U)   = mean over unsupported questions of (RAG - closed-book)
  INTERACTION = gain(S) - gain(U)

This is a difference-in-differences: retrieval condition crossed with support
status. It is estimated per generator and per temporal regime, under both
support labels (exact string, and the NLI entailment relabel), with a
quiz-clustered bootstrap that resamples whole quizzes and recomputes both subset
means inside each draw, so the two arms move together the way they do in the
data.

Three cautions this script keeps attached to its own output:

  * Support is NOT randomised. Whether the retriever found answer-bearing
    evidence is a property of the question and the index, correlated with
    question popularity, entity salience and recency. A positive interaction
    says the gain differs by support status; it does not say that supplying
    support would produce that gain.
  * Both labels use the gold answer, so neither is available at inference time
    and neither is a routing rule.
  * The NLI label is a model prediction on out-of-domain text, not verified
    semantics.

Outputs results/support_interaction.json.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import gold_in_text, token_f1                       # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
BOUNDARY_YEAR = 2025
STRONG = ("mistral_large", "sonnet", "gpt4omini")
# main comparison ladder, the only models with published parameter counts
LADDER_B = {"qwen": 0.5, "qwen1_5b": 1.5, "qwen3b": 3.0, "qwen7b": 7.0,
            "qwen32b": 32.0, "qwen72b": 72.0, "mistral_large": 123.0}


def quiz_of(qid):
    return qid.split("_")[0]


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
                    out.setdefault(d["doc_id"], d.get("text", ""))
    return out


def load_gen(model, cond):
    f = PROC / f"generations_{model}" / f"{cond}.jsonl"
    if not f.is_file():
        return {}
    return {r["qid"]: r for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def did_boot(units, B, seed):
    """units: quiz -> (list of supported gains, list of unsupported gains).

    Resample quizzes with replacement and recompute BOTH arms inside each draw.
    Draws in which a resampled sample has an empty arm are skipped and counted.
    """
    keys = list(units)
    if len(keys) < 3:
        return None
    sup = [g for k in keys for g in units[k][0]]
    uns = [g for k in keys for g in units[k][1]]
    if len(sup) < 5 or len(uns) < 5:
        return None
    obs = statistics.fmean(sup) - statistics.fmean(uns)
    rng = random.Random(seed)
    draws, skipped = [], 0
    for _ in range(B):
        s, u = [], []
        for _ in keys:
            a, b = units[keys[rng.randrange(len(keys))]]
            s.extend(a)
            u.extend(b)
        if not s or not u:
            skipped += 1
            continue
        draws.append(statistics.fmean(s) - statistics.fmean(u))
    if len(draws) < 100:
        return None
    draws.sort()
    n = len(draws)
    le = sum(1 for x in draws if x <= 0)
    return {"interaction": round(obs, 3),
            "ci95": [round(draws[int(0.025 * n)], 3),
                     round(draws[int(0.975 * n)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 5),
            "gain_supported": round(statistics.fmean(sup), 3),
            "gain_unsupported": round(statistics.fmean(uns), 3),
            "n_supported": len(sup), "n_unsupported": len(uns),
            "n_quizzes": len(keys), "n_draws_skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--output", type=Path, default=RES / "support_interaction.json")
    args = ap.parse_args()

    qs = load_questions()
    corpus = load_corpus()

    nli = {}
    f = RES / "nli_support_labels.jsonl"
    if f.is_file():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                nli.setdefault(r["qid"], False)
                nli[r["qid"]] |= r["nli_label"].startswith("entail")

    models = sorted(d.name.replace("generations_", "")
                    for d in PROC.glob("generations_*") if d.is_dir())

    out = {
        "scope": "Difference-in-differences: the retrieval gain among "
                 "automatically supported questions minus the gain among "
                 "unsupported ones, per generator and temporal regime.",
        "estimand": "gain(supported) - gain(unsupported), token F1 points",
        "not_causal": ("Support is not randomised and both labels use the gold "
                       "answer. A positive interaction says the retrieval gain "
                       "differs by automatically labelled support status; it "
                       "does not say that supplying support would cause that "
                       "gain, and it is not a deployable rule."),
        "nli_caveat": ("NLI labels are DeBERTa predictions on truncated news "
                       "passages and generated hypotheses, not verified "
                       "semantics; they are unvalidated against human judgement."),
        "labels": {},
    }

    for label in ("string", "nli"):
        per_model = {}
        for m in models:
            cb, rt = load_gen(m, "closed_book"), load_gen(m, "hybrid")
            if not cb or not rt:
                continue
            rec = {}
            for regime in ("static", "fresh"):
                units = collections.defaultdict(lambda: ([], []))
                for qid, q in qs.items():
                    if (q["year"] >= BOUNDARY_YEAR) != (regime == "fresh"):
                        continue
                    if qid not in rt or qid not in cb:
                        continue
                    if label == "nli":
                        sup = nli.get(qid)
                        if sup is None:      # not in the relabelled sample
                            continue
                    else:
                        texts = [corpus.get(d, "") for d in
                                 (rt[qid].get("doc_ids") or [])]
                        sup = any(gold_in_text(q["gold_answer"], t)
                                  for t in texts if t)
                    g = 100.0 * (token_f1(rt[qid]["prediction"] or "",
                                          q["gold_answer"])
                                 - token_f1(cb[qid]["prediction"] or "",
                                            q["gold_answer"]))
                    units[quiz_of(qid)][0 if sup else 1].append(g)
                r = did_boot(dict(units), args.B, args.seed)
                if r:
                    rec[regime] = r
            if rec:
                per_model[m] = rec
        out["labels"][label] = per_model

        print(f"\n=== support label: {label} ===")
        print(f"{'model':15s}{'regime':>8s}{'gain|S':>9s}{'gain|U':>9s}"
              f"{'interact':>10s}{'ci95':>20s}{'p':>9s}")
        for m, rec in per_model.items():
            for regime, r in rec.items():
                print(f"{m:15s}{regime:>8s}{r['gain_supported']:9.2f}"
                      f"{r['gain_unsupported']:9.2f}{r['interaction']:+10.2f}"
                      f"{str(r['ci95']):>20s}{r['p_plus_one']:9.4f}")

    # ---- does the COST of unsupported evidence scale with capability? ------
    # This is the sharp version of the capability-conditioned reading: it is not
    # that big models gain less from retrieval, but that unsupported evidence
    # costs them more. Tested only where parameter counts are published, by
    # exhaustive permutation over the ladder.
    trend = {}
    for label, per_model in out["labels"].items():
        pts = [(LADDER_B[m], r["static"]["gain_unsupported"])
               for m, r in per_model.items()
               if m in LADDER_B and "static" in r]
        if len(pts) >= 5:
            pts.sort()
            x = [math.log10(a) for a, _ in pts]
            y = [b for _, b in pts]
            mx = statistics.fmean(x)
            sxx = sum((a - mx) ** 2 for a in x)
            fit = lambda yy: sum((a - mx) * (b - statistics.fmean(yy))
                                 for a, b in zip(x, yy)) / sxx
            obs = fit(y)
            ge = tot = 0
            for perm in itertools.permutations(y):
                tot += 1
                if abs(fit(list(perm))) >= abs(obs) - 1e-12:
                    ge += 1
            trend[label] = {
                "slope_per_decade": round(obs, 3), "p_exact": round(ge / tot, 5),
                "n_permutations": tot, "n_rungs": len(y),
                "points": [{"params_B": a, "gain_unsupported": b} for a, b in pts],
                "reading": ("Static-regime retrieval gain among automatically "
                            "UNSUPPORTED questions, regressed on log10 "
                            "parameters over the published-count ladder. A "
                            "negative slope means unsupported evidence costs "
                            "more the more capable the generator -- the cost is "
                            "borne by models with parametric knowledge to "
                            "overwrite. Roster-confounded like every other "
                            "scale statement in this paper."),
            }
    out["unsupported_cost_vs_scale"] = trend
    print(f"\n{'label':8s}{'slope/decade':>14s}{'p_exact':>10s}{'rungs':>7s}")
    for label, t in trend.items():
        print(f"{label:8s}{t['slope_per_decade']:14.2f}{t['p_exact']:10.4f}"
              f"{t['n_rungs']:7d}")

    # ---- summary the manuscript is allowed to state -------------------------
    summ = {}
    for label, per_model in out["labels"].items():
        rows = [(m, r["static"]) for m, r in per_model.items() if "static" in r]
        pos = [m for m, r in rows if r["ci95"][0] > 0]
        strong_rows = [(m, r) for m, r in rows if m in STRONG]
        summ[label] = {
            "n_models_static": len(rows),
            "n_static_interaction_excludes_zero": len(pos),
            "models_static_interaction_excludes_zero": sorted(pos),
            "strong_models": {m: {"interaction": r["interaction"],
                                  "ci95": r["ci95"],
                                  "gain_unsupported": r["gain_unsupported"],
                                  "gain_supported": r["gain_supported"]}
                              for m, r in strong_rows},
            "strong_unsupported_all_negative": all(
                r["gain_unsupported"] < 0 for _, r in strong_rows),
        }
    out["summary"] = summ
    out["verdict"] = {
        "reading": ("The claim the paper may make is that on Static questions "
                    "the retrieval gain is higher among automatically supported "
                    "items than among unsupported ones, for the generators whose "
                    "interaction interval excludes zero, and that the "
                    "unsupported-subset gain is where the negative sign sits. "
                    "It may not claim that unsupported evidence CAUSES the harm, "
                    "nor that harm VANISHES under support."),
    }
    print("\nsummary:")
    for label, s in summ.items():
        print(f"  {label:6s} static interaction excludes zero for "
              f"{s['n_static_interaction_excludes_zero']}/{s['n_models_static']}"
              f" generators: {', '.join(s['models_static_interaction_excludes_zero'])}")
        print(f"         strong-model unsupported gains all negative: "
              f"{s['strong_unsupported_all_negative']}")
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
