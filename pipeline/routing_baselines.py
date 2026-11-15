#!/usr/bin/env python3
"""Answer-independent routing, with paired inference and a real ceiling.

The paper shows retrieval's Static-side harm is concentrated where the retrieved
context does not support the answer. That subset is identified with a
GOLD-CONDITIONED label -- it uses the gold answer -- so it says where to look but
cannot be computed at inference time. The review's objection is right: a
gold-conditioned diagnostic is not a routing recommendation.

This asks the deployable question. For each question, decide whether to use the
retrieved context or answer closed-book, using only signals available BEFORE the
answer is known. Three things the previous version of this script got wrong, all
fixed here:

  1. PAIRED INFERENCE. Marginal bootstrap intervals on two policy means say
     nothing about their difference when both are scored on the same questions
     from the same cached outputs. Every comparison below is a paired
     quiz-clustered bootstrap of (policy - reference) per question, with Holm
     correction across the nine generators.

  2. A REAL CEILING. The old "fraction of oracle headroom recovered" used the
     gold-string policy as the denominator. That is not a ceiling: it is worse
     than always-retrieve for Qwen2.5-3B (so the ratio was undefined and that
     model was silently dropped) and worse than the overlap gate for two others
     (so the ratio exceeded 1). The per-question upper bound over all
     deterministic gates is PERQ_BEST -- retrieve iff doing so scores higher on
     that question. No gate can beat it, so the recovered fraction is at most 1
     by construction and is defined for every model.

  3. CROSS-FITTING. A single 99/100-quiz split makes the threshold, and hence
     the gain, a draw from a distribution nobody reported. Thresholds are now
     chosen by repeated two-fold cross-fitting: for each of S seeds the quizzes
     are split in half, each half's threshold is tuned on the other half, and
     every question is scored under a threshold that never saw it. We report the
     spread of thresholds, retrieve rates, and paired gains across seeds.

The policies:

  NEVER / ALWAYS   retrieve nothing / everything -- the two fixed baselines. The
                   reference for every paired contrast is the better of the two.
  OVERLAP          question-document lexical overlap over a threshold.
  CONSENSUS        k of the 6 retrievers agree on the top document.
  RECENCY          a retrieved document is dated at or after the question.
  SELF_KNOWLEDGE   the model abstains or hedges closed-book (SKR/RetrievalQA
                   idea, using the only self-knowledge signal our cache keeps).
  CUTOFF           the question post-dates the model's disclosed cutoff;
                   available only for the four generators with a published date.

  GOLD_STRING      retrieve iff the context contains the gold answer string.
  GOLD_NLI         retrieve iff the context entails the answer.
                   Both use the gold answer: diagnostics, not policies, and
                   NEITHER is a ceiling (see 2 above).
  PERQ_BEST        retrieve iff retrieval scores higher on this question. The
                   actual upper bound.

Outputs results/routing_baselines.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import gold_in_text, token_f1                       # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
RETRIEVERS = ("bm25", "dpr", "bge", "splade", "colbert", "hybrid")
CUTOFFS = {"llama3_8b": "2023-03", "gpt4omini": "2023-10",
           "haiku": "2025-07", "sonnet": "2025-07"}
HEDGE = ("i don't know", "i do not know", "unknown", "cannot determine",
         "not sure", "unable to", "no information", "as of my", "my training")
DEPLOYABLE = ("overlap", "consensus", "recency", "self_knowledge", "cutoff")
TUNED = ("overlap", "consensus")


def quiz_of(qid):
    return qid.split("_")[0]


def toks(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


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


def cluster_mean_ci(per_q, B, seed):
    """per_q: (quiz, value). Cluster bootstrap over quizzes."""
    by = collections.defaultdict(list)
    for q, v in per_q:
        by[q].append(v)
    vals = [statistics.fmean(v) for v in by.values()]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean([vals[rng.randrange(len(vals))]
                                     for _ in vals]) for _ in range(B))
    return {"mean": round(statistics.fmean(vals), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "n_quizzes": len(vals), "n_questions": len(per_q)}


def paired_delta(per_q, B, seed):
    """per_q: (quiz, policy_score - reference_score) per question.

    Quiz-clustered bootstrap of the PAIRED difference. Both policies are scored
    on the same question from the same cached generation, so the difference is
    the estimand; two marginal intervals are not.
    """
    by = collections.defaultdict(list)
    for q, v in per_q:
        by[q].append(v)
    vals = [statistics.fmean(v) for v in by.values()]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean([vals[rng.randrange(len(vals))]
                                     for _ in vals]) for _ in range(B))
    le = sum(1 for x in draws if x <= 0)
    return {"delta": round(statistics.fmean(vals), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5),
            "n_quizzes": len(vals), "n_questions": len(per_q)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, run = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, round(run, 5))
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--splits", type=int, default=20,
                    help="cross-fitting repetitions (each is a 2-fold split)")
    ap.add_argument("--output", type=Path, default=RES / "routing_baselines.json")
    args = ap.parse_args()

    qs = load_questions()
    corpus = load_corpus()
    ranked = json.loads((PROC / "retrieval.json").read_text())
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

    # ---- answer-independent per-question signals ---------------------------
    sig = {}
    for qid, q in qs.items():
        docs = ranked.get("hybrid", {}).get(qid, [])[:5]
        texts = [corpus.get(d, {}).get("text", "") for d in docs]
        qt = toks(q["question"])
        ov = max((len(qt & toks(t)) / max(len(qt), 1) for t in texts), default=0.0)
        tops = [ranked.get(r, {}).get(qid, [None])[0] for r in RETRIEVERS]
        tops = [t for t in tops if t]
        cons = (collections.Counter(tops).most_common(1)[0][1] / len(tops)) if tops else 0.0
        qdate = str(q["date"])[:10]
        rec = any((corpus.get(d, {}).get("publish_date") or "")[:10] >= qdate
                  for d in docs)
        sig[qid] = {"overlap": ov, "consensus": cons, "recency": rec,
                    "gold_string": any(gold_in_text(q["gold_answer"], t)
                                       for t in texts if t),
                    "gold_nli": nli.get(qid)}

    quizzes = sorted({quiz_of(q) for q in qs})
    grid = [i / 20 for i in range(21)]

    out = {
        "scope": "Per-question routing policies scored on cached closed-book and "
                 "RAG outputs; no new generation.",
        "inference": ("Every policy is compared against the better fixed policy "
                      "(always- or never-retrieve) by a PAIRED quiz-clustered "
                      "bootstrap of the per-question difference, Holm-corrected "
                      "across the nine generators."),
        "gold_conditioned_are_not_policies": (
            "GOLD_STRING and GOLD_NLI use the gold answer. They are diagnostics, "
            "not deployable rules, and neither is an upper bound: GOLD_STRING is "
            "below always-retrieve for one generator and below the lexical gate "
            "for two."),
        "ceiling": ("PERQ_BEST -- retrieve iff that raises this question's score "
                    "-- is the per-question upper bound over all deterministic "
                    "gates, so recovered fractions are at most 1 and are defined "
                    "for every generator."),
        "cross_fitting": (f"{args.splits} repeated two-fold splits over the "
                          f"{len(quizzes)} quizzes; each fold's threshold is "
                          "tuned on the complementary fold, so every question is "
                          "scored under a threshold that never saw it."),
        "models": {},
    }

    print(f"{'model':15s}{'never':>8s}{'always':>8s}{'overlap':>9s}"
          f"{'consens':>9s}{'recency':>9s}{'selfk':>8s}{'cutoff':>8s}"
          f"{'gold':>8s}{'CEIL':>8s}")
    contrasts = {}
    for m in models:
        cb, rt = load_gen(m, "closed_book"), load_gen(m, "hybrid")
        if not cb or not rt:
            continue
        common = sorted(q for q in qs if q in cb and q in rt)
        f_cb = {q: token_f1(cb[q]["prediction"] or "", qs[q]["gold_answer"]) * 100
                for q in common}
        f_rt = {q: token_f1(rt[q]["prediction"] or "", qs[q]["gold_answer"]) * 100
                for q in common}
        hedged = {q: any(h in (cb[q]["prediction"] or "").lower() for h in HEDGE)
                  for q in common}
        score = lambda q, use: (f_rt if use else f_cb)[q]

        # ---- repeated two-fold cross-fitting for the tuned thresholds ------
        # cf[key][qid] -> the decision this question got under a threshold
        # tuned without it; collected per seed so we can report spread.
        cf_scores = {k: [] for k in TUNED}      # per seed: mean over questions
        cf_thresh = {k: [] for k in TUNED}      # per seed: the two thresholds
        cf_rate = {k: [] for k in TUNED}
        cf_perq = {k: collections.defaultdict(list) for k in TUNED}
        for s in range(args.splits):
            rng = random.Random(args.seed + 1009 * s)
            shuf = quizzes[:]
            rng.shuffle(shuf)
            half = set(shuf[:len(shuf) // 2])
            folds = [half, set(shuf[len(shuf) // 2:])]
            for key in TUNED:
                dec, ths = {}, []
                for fold in folds:
                    other = [q for q in common if quiz_of(q) not in fold]
                    best, best_v = grid[0], -1e9
                    for th in grid:
                        v = statistics.fmean(
                            [score(q, sig[q][key] >= th) for q in other] or [0])
                        if v > best_v:
                            best, best_v = th, v
                    ths.append(best)
                    for q in common:
                        if quiz_of(q) in fold:
                            dec[q] = sig[q][key] >= best
                cf_thresh[key].append(sorted(ths))
                cf_rate[key].append(round(sum(dec.values()) / len(dec), 3))
                cf_scores[key].append(statistics.fmean(
                    [score(q, dec[q]) for q in common]))
                for q in common:
                    cf_perq[key][q].append(score(q, dec[q]))

        # cross-fitted score per question = mean over the repeated splits
        policies = {
            "never":   {q: score(q, False) for q in common},
            "always":  {q: score(q, True) for q in common},
            "recency": {q: score(q, sig[q]["recency"]) for q in common},
            "self_knowledge": {q: score(q, hedged[q]) for q in common},
            "gold_string": {q: score(q, sig[q]["gold_string"]) for q in common},
            "perq_best": {q: max(f_cb[q], f_rt[q]) for q in common},
        }
        for key in TUNED:
            policies[key] = {q: statistics.fmean(cf_perq[key][q]) for q in common}
        if m in CUTOFFS:
            policies["cutoff"] = {q: score(q, str(qs[q]["date"])[:7] > CUTOFFS[m])
                                  for q in common}
        if any(sig[q]["gold_nli"] is not None for q in common):
            policies["gold_nli"] = {q: score(q, bool(sig[q]["gold_nli"]))
                                    for q in common}

        rec = {"n_questions": len(common),
               "n_quizzes": len({quiz_of(q) for q in common}),
               "cross_fit": {k: {"thresholds": cf_thresh[k],
                                 "threshold_min": min(t for p in cf_thresh[k] for t in p),
                                 "threshold_max": max(t for p in cf_thresh[k] for t in p),
                                 "retrieve_rate_mean": round(statistics.fmean(cf_rate[k]), 3),
                                 "retrieve_rate_range": [min(cf_rate[k]), max(cf_rate[k])],
                                 "score_mean": round(statistics.fmean(cf_scores[k]), 3),
                                 "score_range": [round(min(cf_scores[k]), 3),
                                                 round(max(cf_scores[k]), 3)]}
                             for k in TUNED},
               "policies": {}, "vs_reference": {}}

        for name, sc in policies.items():
            r = cluster_mean_ci([(quiz_of(q), sc[q]) for q in common],
                                args.B, args.seed)
            if name in ("never", "always"):
                r["retrieve_rate"] = 0.0 if name == "never" else 1.0
            elif name in TUNED:
                r["retrieve_rate"] = rec["cross_fit"][name]["retrieve_rate_mean"]
            rec["policies"][name] = r

        # reference = the better FIXED policy, chosen per model and reported
        ref = max(("always", "never"), key=lambda k: rec["policies"][k]["mean"])
        rec["reference"] = ref
        for name, sc in policies.items():
            if name == ref:
                continue
            d = paired_delta([(quiz_of(q), sc[q] - policies[ref][q])
                              for q in common], args.B, args.seed)
            if d:
                rec["vs_reference"][name] = d
        contrasts[m] = rec["vs_reference"].get("overlap", {}).get("p_plus_one")
        out["models"][m] = rec
        g = lambda k: rec["policies"].get(k, {}).get("mean")
        fmt = lambda v: f"{v:8.2f}" if v is not None else f"{'-':>8s}"
        print(f"{m:15s}{fmt(g('never'))}{fmt(g('always'))}{fmt(g('overlap'))}"
              f"{fmt(g('consensus'))}{fmt(g('recency'))}{fmt(g('self_knowledge'))}"
              f"{fmt(g('cutoff'))}{fmt(g('gold_string'))}{fmt(g('perq_best'))}")

    # ---- Holm across generators for the headline overlap contrast ----------
    names = [m for m in out["models"] if contrasts.get(m) is not None]
    adj = holm([contrasts[m] for m in names])
    for m, a in zip(names, adj):
        out["models"][m]["vs_reference"]["overlap"]["p_holm"] = a

    # ---- how much of the REAL ceiling does the best deployable gate get? ---
    rows = []
    for m, rec in out["models"].items():
        p, ref = rec["policies"], rec["reference"]
        if "perq_best" not in p:
            continue
        base, ceil = p[ref]["mean"], p["perq_best"]["mean"]
        cand = [(k, p[k]["mean"]) for k in DEPLOYABLE if k in p]
        best = max(cand, key=lambda kv: kv[1]) if cand else None
        rows.append({
            "model": m, "reference": ref, "reference_score": base,
            "ceiling": ceil, "best_deployable": best[0], "best_value": best[1],
            "recovered": round((best[1] - base) / (ceil - base), 3)
            if ceil > base else None,
            "overlap_delta": rec["vs_reference"].get("overlap", {}).get("delta"),
            "overlap_ci95": rec["vs_reference"].get("overlap", {}).get("ci95"),
            "overlap_p_holm": rec["vs_reference"].get("overlap", {}).get("p_holm"),
        })
    sig_pos = [r for r in rows if r["overlap_ci95"] and r["overlap_ci95"][0] > 0
               and (r["overlap_p_holm"] or 1) < 0.05]
    out["ceiling_recovery"] = {
        "per_model": rows,
        "definition": ("(best deployable gate - better fixed policy) / "
                       "(per-question best - better fixed policy); every "
                       "generator is included and the ratio cannot exceed 1"),
        "median_recovered": round(statistics.median(
            [r["recovered"] for r in rows if r["recovered"] is not None]), 3)
        if any(r["recovered"] is not None for r in rows) else None,
    }
    out["overlap_summary"] = {
        "n_generators": len(rows),
        "n_positive_point_estimate": sum(1 for r in rows if (r["overlap_delta"] or 0) > 0),
        "n_significant_after_holm": len(sig_pos),
        "significant_models": sorted(r["model"] for r in sig_pos),
        "worst_model": min(rows, key=lambda r: r["overlap_delta"] or 0)["model"] if rows else None,
        "worst_delta": min((r["overlap_delta"] or 0) for r in rows) if rows else None,
        "statement": ("The lexical-overlap gate does NOT improve every generator: "
                      "report the number with intervals excluding zero after Holm "
                      "correction, and report the generators where it does not."),
    }

    print(f"\n{'model':15s}{'ref':>8s}{'delta':>8s}{'ci95':>18s}{'p_holm':>9s}"
          f"{'ceiling':>9s}{'recov':>8s}")
    for r in rows:
        ci = str(r["overlap_ci95"]) if r["overlap_ci95"] else "-"
        ph = f"{r['overlap_p_holm']:.4f}" if r["overlap_p_holm"] is not None else "-"
        rv = f"{r['recovered']:.3f}" if r["recovered"] is not None else "-"
        print(f"{r['model']:15s}{r['reference']:>8s}"
              f"{(r['overlap_delta'] or 0):+8.2f}{ci:>18s}{ph:>9s}"
              f"{r['ceiling']:9.2f}{rv:>8s}")
    s = out["overlap_summary"]
    print(f"\noverlap > reference (point): {s['n_positive_point_estimate']}/{s['n_generators']}"
          f"; significant after Holm: {s['n_significant_after_holm']} "
          f"({', '.join(s['significant_models'])})")
    print(f"worst generator: {s['worst_model']} {s['worst_delta']:+.3f}")
    print(f"median ceiling recovery: {out['ceiling_recovery']['median_recovered']}")
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
