#!/usr/bin/env python3
"""What kind of error does stale evidence actually cause? (TODO M-1)

The paper reports that swapping the live index for a pre-2025 one drops
Fresh F1 from 39.1 to 17.4 --- below closed-book. That number says the
harm is real but not what it *is*, so this computes the
partition. This script provides it, grounded in what was actually
retrieved: every generation record carries the `doc_ids` it was shown,
so each failure can be attributed to the evidence the model saw rather
than guessed at from the answer string alone.

Categories (mutually exclusive, in priority order):

  abstained            the model declined to answer.
  answer_present       the gold answer string IS in the retrieved stale
                       context and the model still missed it. A reading
                       failure, not a staleness failure --- this is the
                       control that keeps the other counts honest.
  grounded_outdated    the gold answer is ABSENT from the retrieved
                       context, and the model's answer is drawn from
                       that context (content-word overlap with the
                       retrieved text). The model read stale evidence
                       faithfully and reported a superseded fact. This
                       is the dangerous case: the pipeline behaves
                       correctly and is confidently wrong.
  ungrounded           gold absent from context and the answer is not
                       traceable to it either --- the model fell back on
                       parametric memory or fabricated.

Run on every model with a cached `hybrid_stale`, which is Qwen2.5-32B
and Qwen2.5-3B, so the control no longer rests on a single generator.

Outputs `results/stale_taxonomy.json`.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import metrics as M  # noqa: E402

MODELS = [("qwen32b", "Qwen2.5-32B"), ("qwen3b", "Qwen2.5-3B"),
          ("mistral_large", "Mistral-Large-2")]
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "for",
    "is", "was", "were", "are", "be", "been", "it", "its", "that", "this",
    "with", "as", "by", "from", "has", "have", "had", "not", "no", "he",
    "she", "they", "his", "her", "their", "we", "you", "i", "but", "than",
    "who", "what", "when", "which", "will", "would", "there", "said",
}


def content_words(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if t not in STOP and len(t) > 2}


def load_questions() -> dict[str, dict]:
    path = ROOT / "data/processed/questions.jsonl"
    return {json.loads(l)["qid"]: json.loads(l) for l in path.open(encoding="utf-8")}


def load_corpus() -> dict[str, str]:
    out = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        p = ROOT / "data/processed" / name
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            d = json.loads(line)
            out.setdefault(d["doc_id"], d.get("text", ""))
    return out


def load_gen(model: str, cond: str) -> dict[str, dict]:
    p = ROOT / f"data/processed/generations_{model}/{cond}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["qid"]: json.loads(l) for l in p.open(encoding="utf-8")}


def doc_ids_of(rec: dict) -> list[str]:
    raw = rec.get("doc_ids")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            return list(v) if isinstance(v, (list, tuple)) else []
        except Exception:
            return []
    return []


def classify(pred: str, gold: str, context: str, thresh: float = 0.5) -> str:
    """Lexical, not semantic. `grounded_outdated` means the answer's content
    words are largely present in the retrieved text -- which is consistent with
    a faithfully-read superseded fact but does not establish one."""
    if M.is_abstention(pred):
        return "abstained"
    ctx_l, gold_l = context.lower(), (gold or "").strip().lower()
    if gold_l and gold_l in ctx_l:
        return "answer_present"
    pw, cw = content_words(pred), content_words(context)
    if pw and len(pw & cw) / len(pw) >= thresh:
        return "grounded_outdated"
    return "ungrounded"


def classify_strict(pred: str, gold: str, context: str) -> str:
    """A deliberately stricter second rule, reported alongside the first so the
    headline share is not an artifact of one threshold and one notion of
    'grounded'. Here the answer must appear in the context as a contiguous
    span (after normalisation), not merely share a bag of content words."""
    if M.is_abstention(pred):
        return "abstained"
    ctx_l, gold_l = context.lower(), (gold or "").strip().lower()
    if gold_l and gold_l in ctx_l:
        return "answer_present"
    pred_n = " ".join(re.findall(r"[a-z0-9]+", (pred or "").lower()))
    ctx_n = " ".join(re.findall(r"[a-z0-9]+", (context or "").lower()))
    if pred_n and pred_n in ctx_n:
        return "grounded_outdated"
    return "ungrounded"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fail-margin", type=float, default=10.0,
                    help="F1 points by which stale must trail the live "
                         "index for the item to count as a stale-induced "
                         "failure")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/stale_taxonomy.json")
    args = ap.parse_args()

    questions, corpus = load_questions(), load_corpus()
    out = {
        "scope": "Partition of stale-index failures by what the retrieved "
                 "stale context actually contained. A failure is an item "
                 "where the stale index scores at least --fail-margin F1 "
                 "below the live index.",
        "fail_margin_f1": args.fail_margin,
        "categories": {
            "abstained": "model declined to answer",
            "answer_present": "gold string was in the stale context and the "
                              "model still missed it (reading failure, not "
                              "staleness)",
            "grounded_outdated": "gold absent from context; answer traceable "
                                 "to the stale context (faithful reading of "
                                 "superseded evidence)",
            "ungrounded": "gold absent from context; answer not traceable to "
                          "it (parametric fallback or fabrication)",
        },
        "models": {},
    }

    for key, name in MODELS:
        live, stale = load_gen(key, "hybrid"), load_gen(key, "hybrid_stale")
        if not live or not stale:
            print(f"  [skip] {name}: missing hybrid or hybrid_stale")
            continue
        counts, fresh_counts, examples = {}, {}, []
        strict_counts, sweep = {}, {}
        n_fail = n_fresh_fail = 0
        for qid, srec in stale.items():
            q, lrec = questions.get(qid), live.get(qid)
            if not q or not lrec:
                continue
            gold = q["gold_answer"]
            f_live = 100.0 * M.token_f1(lrec["prediction"], gold)
            f_stale = 100.0 * M.token_f1(srec["prediction"], gold)
            if f_live - f_stale < args.fail_margin:
                continue
            n_fail += 1
            ctx = " ".join(corpus.get(d, "") for d in doc_ids_of(srec))
            cat = classify(srec["prediction"], gold, ctx)
            counts[cat] = counts.get(cat, 0) + 1
            cat_s = classify_strict(srec["prediction"], gold, ctx)
            strict_counts[cat_s] = strict_counts.get(cat_s, 0) + 1
            for t in SWEEP:
                if classify(srec["prediction"], gold, ctx, t) == "grounded_outdated":
                    sweep[t] = sweep.get(t, 0) + 1
            is_fresh = str(q["date"])[:4] >= "2025"
            if is_fresh:
                n_fresh_fail += 1
                fresh_counts[cat] = fresh_counts.get(cat, 0) + 1
            if cat == "grounded_outdated" and len(examples) < 5:
                examples.append({
                    "qid": qid, "question": q["question"][:150],
                    "gold": gold,
                    "stale_prediction": srec["prediction"][:150],
                    "live_f1": round(f_live, 1), "stale_f1": round(f_stale, 1),
                })
        out["models"][key] = {
            "name": name,
            "n_stale_failures": n_fail,
            "counts": counts,
            "pct": {k: round(100.0 * v / n_fail, 1) for k, v in counts.items()} if n_fail else {},
            "n_fresh_failures": n_fresh_fail,
            "fresh_counts": fresh_counts,
            "fresh_pct": {k: round(100.0 * v / n_fresh_fail, 1)
                          for k, v in fresh_counts.items()} if n_fresh_fail else {},
            "grounded_outdated_examples": examples,
            "strict_span_rule": {
                "rule": "answer must appear as a contiguous normalised span in "
                        "the retrieved context",
                "counts": strict_counts,
                "pct": {k: round(100.0 * v / n_fail, 1)
                        for k, v in strict_counts.items()} if n_fail else {},
            },
            "threshold_sweep_grounded_pct": {
                str(t): round(100.0 * sweep.get(t, 0) / n_fail, 1) for t in SWEEP
            } if n_fail else {},
        }
        print(f"\n{name}: {n_fail} stale-induced failures "
              f"({n_fresh_fail} on Fresh)")
        for cat in ("grounded_outdated", "ungrounded", "answer_present", "abstained"):
            c = counts.get(cat, 0)
            fc = fresh_counts.get(cat, 0)
            print(f"    {cat:20s} {c:4d} ({100.0*c/max(n_fail,1):5.1f}%)   "
                  f"Fresh {fc:4d} ({100.0*fc/max(n_fresh_fail,1):5.1f}%)")

    out["interpretation"] = (
        "THIS IS A LEXICAL HEURISTIC, NOT AN ADJUDICATED LABEL. 'grounded' is "
        "content-word overlap between the answer and the retrieved text. It "
        "cannot distinguish a faithfully-read superseded fact from a "
        "coincidental lexical match, and it cannot check that the answer is "
        "actually the OUTDATED version of the fact rather than an unrelated "
        "string that happens to appear in a news article. The headline share "
        "should therefore be read as an UPPER BOUND on faithful-stale-reading "
        "and a LOWER BOUND on the complementary hallucination share. We report "
        "two sensitivity analyses in place of the human audit this needs: a "
        "threshold sweep over the overlap cutoff, and a strict contiguous-span "
        "rule that is a different notion of grounding rather than a tighter "
        "version of the same one. The answer_present category is the internal "
        "control: items in it are reading failures rather than staleness "
        "failures, so the staleness categories are not inflated by generic "
        "model error."
    )
    out["validation_status"] = (
        "NOT VALIDATED AGAINST HUMAN JUDGEMENT. A blinded audit of a random "
        "sample is required before any specific percentage is quoted as the "
        "rate of faithful stale reading."
    )
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
