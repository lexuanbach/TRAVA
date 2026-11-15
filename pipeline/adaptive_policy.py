#!/usr/bin/env python
"""Adaptive-retrieval policies evaluated from cached generations.

Identification problem this addresses: the paper's central empirical
message is that retrieval helps when parametric recall is weak and can hurt when
it is strong -- which is precisely the setting adaptive retrieval addresses -- yet
no routing baseline is reported. The audit needs at least a simple
confidence/self-knowledge router and a "retrieve only on post-boundary questions"
oracle policy.

Every policy below is computed by *selecting between already-cached closed-book
and retrieval-augmented generations per question*. No new inference is required,
so this costs nothing and introduces no new model version.

Policies:
  never          always answer closed-book (retrieval rate 0)
  always         always answer with retrieval (retrieval rate 1)
  temporal       retrieve iff the question post-dates the shared boundary. This is
                 an *oracle* policy: it uses the question date, which a deployed
                 system would have, but it presumes the boundary is known.
  abstain        retrieve iff the closed-book answer is an abstention. A
                 self-knowledge router in the SKR/Self-RAG family, using the only
                 confidence signal available from cached text output.
  oracle_perq    retrieve iff doing so raises token-F1 on that question. Not
                 attainable; reported as the ceiling for any router.

Outputs results/adaptive_policy.json.

Usage:
    python pipeline/adaptive_policy.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from metrics import is_abstention, token_f1  # noqa: E402

BOUNDARY = date(2025, 1, 1)
MODELS = [
    ("qwen1_5b", "Qwen2.5-1.5B"), ("qwen3b", "Qwen2.5-3B"), ("qwen7b", "Qwen2.5-7B"),
    ("qwen32b", "Qwen2.5-32B"), ("qwen72b", "Qwen2.5-72B"),
    ("mistral_large", "Mistral-Large-2"), ("gpt4omini", "GPT-4o-mini"), ("sonnet", "Claude-Sonnet"),
]


def load_questions() -> dict[str, dict]:
    return {json.loads(l)["qid"]: json.loads(l)
            for l in (ROOT / "data/processed/questions.jsonl").open(encoding="utf-8")}


def load_condition(model: str, condition: str) -> dict[str, str]:
    path = ROOT / f"data/processed/generations_{model}/{condition}.jsonl"
    if not path.exists():
        return {}
    return {json.loads(l)["qid"]: json.loads(l)["prediction"] for l in path.open(encoding="utf-8")}


def is_fresh(q: dict) -> bool:
    return q["date"] >= BOUNDARY.isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrieval-condition", default="hybrid")
    ap.add_argument("--output", type=Path, default=ROOT / "results/adaptive_policy.json")
    args = ap.parse_args()

    questions = load_questions()
    report = {
        "scope": ("Adaptive-retrieval policies computed by selecting between cached closed-book and "
                  "retrieval-augmented generations per question. No new inference; no new model "
                  "version. 'temporal' and 'oracle_perq' are oracle policies and are reported as "
                  "reference points, not as deployable systems."),
        "boundary": BOUNDARY.isoformat(),
        "retrieval_condition": args.retrieval_condition,
        "models": {},
    }

    for key, name in MODELS:
        cb = load_condition(key, "closed_book")
        rag = load_condition(key, args.retrieval_condition)
        if not cb or not rag:
            continue
        qids = [q for q in cb if q in rag and q in questions]
        f_cb = {q: 100.0 * token_f1(cb[q], questions[q]["gold_answer"]) for q in qids}
        f_rag = {q: 100.0 * token_f1(rag[q], questions[q]["gold_answer"]) for q in qids}

        policies = {
            "never": {q: False for q in qids},
            "always": {q: True for q in qids},
            "temporal": {q: is_fresh(questions[q]) for q in qids},
            "abstain": {q: is_abstention(cb[q]) for q in qids},
            "oracle_perq": {q: f_rag[q] > f_cb[q] for q in qids},
        }

        model_out = {}
        for pname, decide in policies.items():
            picked = {q: (f_rag[q] if decide[q] else f_cb[q]) for q in qids}
            static = [v for q, v in picked.items() if not is_fresh(questions[q])]
            fresh = [v for q, v in picked.items() if is_fresh(questions[q])]
            model_out[pname] = {
                "f1_all": round(statistics.fmean(picked.values()), 2),
                "f1_static": round(statistics.fmean(static), 2),
                "f1_fresh": round(statistics.fmean(fresh), 2),
                "retrieval_rate": round(sum(decide.values()) / len(qids), 3),
            }
        # Headroom the best DEPLOYABLE policy leaves on the table. `temporal`
        # is excluded here even though it is often the best-scoring non-perq
        # policy: it reads the question's publication date, which a deployed
        # system does not have. Including it (as an earlier version of this
        # script did) understates the headroom, because it credits the system
        # with information it cannot obtain.
        DEPLOYABLE = ("never", "always", "abstain")
        best_real = max(model_out[p]["f1_all"] for p in DEPLOYABLE)
        model_out["deployable_policies"] = list(DEPLOYABLE)
        model_out["best_deployable"] = max(DEPLOYABLE, key=lambda p: model_out[p]["f1_all"])
        model_out["best_deployable_f1"] = best_real
        model_out["oracle_headroom_f1"] = round(model_out["oracle_perq"]["f1_all"] - best_real, 2)
        # How much of that headroom the date oracle alone would buy.
        model_out["temporal_oracle_gain_f1"] = round(
            model_out["temporal"]["f1_all"] - best_real, 2)
        model_out["n_questions"] = len(qids)
        report["models"][key] = {"name": name, **model_out}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'model':18s} " + " ".join(f"{p:>9s}" for p in
          ("never", "always", "temporal", "abstain", "oracle"))
          + "   best-depl.  headroom  temp.gain")
    for key, m in report["models"].items():
        row = " ".join(f"{m[p]['f1_all']:9.2f}" for p in
                       ("never", "always", "temporal", "abstain", "oracle_perq"))
        print(f"{m['name']:18s} {row}   {m['best_deployable']:>10s}  "
              f"{m['oracle_headroom_f1']:8.2f}  {m['temporal_oracle_gain_f1']:+8.2f}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
