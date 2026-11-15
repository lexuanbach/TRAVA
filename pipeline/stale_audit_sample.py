#!/usr/bin/env python3
"""Draw a random sample of automatically-labelled `grounded_outdated` items
for manual adjudication.

The lexical rule in `stale_taxonomy.py` calls an item a faithful reading of
superseded evidence when the answer's content words overlap the retrieved
text. Inspecting the stored examples showed the rule is not measuring that:
it labels type errors ("TobyKeith" as a dog breed) and salient-entity echoes
("Nick Cannon" for a citizenship threat) the same way it labels a genuinely
outdated fact. The percentage cannot be quoted until it has been checked
against judgement, so this script draws a fixed random sample and writes it
in adjudication order; `stale_audit_labels.json` holds the verdicts and
`stale_audit_report.py` scores them.

Rubric (exactly one per item):

  superseded    The answer is, or was, a correct answer to this question at an
                earlier time -- including a correct answer about an earlier
                instance of a recurring event. This is the paper's claim.
  echo          The answer is drawn from the retrieved text but was never a
                correct answer to this question: a salient entity of roughly
                the right type, grabbed from stale news.
  type_error    The answer is not the right kind of thing at all.
  non_answer    Narrative, refusal-like, or otherwise not an answer attempt.

Only `superseded` supports the claim in the paper.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import metrics as M  # noqa: E402
from stale_taxonomy import (classify, doc_ids_of, load_corpus,  # noqa: E402
                            load_gen, load_questions)

MODELS = [("qwen32b", "Qwen2.5-32B"), ("qwen3b", "Qwen2.5-3B"),
          ("mistral_large", "Mistral-Large-2")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60, help="total sample size")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--fail-margin", type=float, default=10.0)
    ap.add_argument("--only-model", default=None,
                    help="restrict the pool to one generator key")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/stale_audit_sample.json")
    args = ap.parse_args()

    questions, corpus = load_questions(), load_corpus()
    pool = []
    for key, name in MODELS:
        if args.only_model and key != args.only_model:
            continue
        live, stale = load_gen(key, "hybrid"), load_gen(key, "hybrid_stale")
        if not live or not stale:
            continue
        for qid, srec in stale.items():
            q, lrec = questions.get(qid), live.get(qid)
            if not q or not lrec:
                continue
            gold = q["gold_answer"]
            f_live = 100.0 * M.token_f1(lrec["prediction"], gold)
            f_stale = 100.0 * M.token_f1(srec["prediction"], gold)
            if f_live - f_stale < args.fail_margin:
                continue
            ctx = " ".join(corpus.get(d, "") for d in doc_ids_of(srec))
            if classify(srec["prediction"], gold, ctx) != "grounded_outdated":
                continue
            pool.append({
                "model": key, "model_name": name, "qid": qid,
                "date": str(q["date"])[:10],
                "question": q["question"],
                "gold_answer": gold,
                "stale_prediction": srec["prediction"][:400],
                "context_head": ctx[:900],
            })

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sample = pool[: args.n]
    out = {
        "scope": "Random sample of automatically-labelled grounded_outdated "
                 "items, drawn for manual adjudication because the lexical "
                 "rule is not validated against judgement.",
        "rubric": {
            "superseded": "was a correct answer to this question at an earlier "
                          "time (supports the paper's claim)",
            "echo": "drawn from the retrieved text but never a correct answer "
                    "to this question",
            "type_error": "not the right kind of thing at all",
            "non_answer": "narrative or refusal-like, not an answer attempt",
        },
        "seed": args.seed, "pool_size": len(pool), "n_sampled": len(sample),
        "items": sample,
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"pool={len(pool)}  sampled={len(sample)}  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
