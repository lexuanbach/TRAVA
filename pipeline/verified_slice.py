#!/usr/bin/env python3
"""Per-model cutoff evidence, and the gap on a verified post-training slice.

Addresses three linked review concerns.

(a) The 2025 analysis boundary was read off Claude Haiku's closed-book knee and
     then applied to every model. Selecting a year from an observed benchmark curve
     is not prespecification, and year difficulty is shared across models, so the
     selection can propagate to the open-model ladder. The 2025-boundary result is
     therefore exploratory.

(b) The claim that every question dated on or after 2025-01-01 post-dates every
     generator's pretraining is false for the Claude models: Anthropic documents
     public-Internet training data through July 2025, with reliable-knowledge
     cutoffs in January (Sonnet 4.5) and February (Haiku 4.5) 2025. The sampled
     2025 questions are therefore NOT guaranteed post-training for them.

(c) A benchmark must store, per model, the evidence behind its pretraining-safe
     dates and must distinguish `calendar-post-boundary` from
     `verified-post-training`.

This script (a) records a per-model cutoff-evidence table with an explicit evidence
class, and (b) recomputes every model's boundary gap on the **2026-only** slice,
which is post-training for every generator here under any documented reading, and
which does not depend on where the 2025 boundary was placed.

Outputs `results/verified_slice.json`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from metrics import token_f1  # noqa: E402
from clusters import quiz_id  # canonical resampling unit

# evidence_class:
#   disclosed_exact     - vendor publishes a dated cutoff
#   disclosed_range     - vendor documents a training-data end and a separate
#                         "reliable knowledge" date; the two differ
#   undisclosed         - no cutoff published by the vendor
CUTOFF_EVIDENCE = {
    "qwen":          dict(name="Qwen2.5-0.5B",      evidence_class="undisclosed",
                          note="No cutoff in the Qwen2.5 technical report."),
    "qwen1_5b":      dict(name="Qwen2.5-1.5B",      evidence_class="undisclosed",
                          note="No cutoff in the Qwen2.5 technical report."),
    "qwen3b":        dict(name="Qwen2.5-3B",        evidence_class="undisclosed",
                          note="No cutoff in the Qwen2.5 technical report."),
    "qwen7b":        dict(name="Qwen2.5-7B",        evidence_class="undisclosed",
                          note="No cutoff in the Qwen2.5 technical report."),
    "qwen32b":       dict(name="Qwen2.5-32B",       evidence_class="undisclosed",
                          note="No cutoff in the Qwen2.5 technical report."),
    "qwen72b":       dict(name="Qwen2.5-72B",       evidence_class="undisclosed",
                          note="No cutoff in the Qwen2.5 technical report."),
    "mistral_large": dict(name="Mistral-Large-2",   evidence_class="undisclosed",
                          note="No cutoff published for Mistral-Large-Instruct-2407."),
    "gpt4omini":     dict(name="GPT-4o-mini",       evidence_class="disclosed_exact",
                          cutoff="2023-10-01",
                          note="Day-precise vendor-disclosed cutoff."),
    "sonnet":        dict(name="Claude-Sonnet-4.5", evidence_class="disclosed_range",
                          training_data_through="2025-07", reliable_knowledge="2025-01",
                          note="Training data through July 2025; reliable-knowledge "
                               "cutoff January 2025. 2025 questions are NOT "
                               "guaranteed post-training."),
    "haiku":         dict(name="Claude-Haiku-4.5",  evidence_class="disclosed_range",
                          training_data_through="2025-07", reliable_knowledge="2025-02",
                          note="Training data through July 2025; reliable-knowledge "
                               "cutoff February 2025. 2025 questions are NOT "
                               "guaranteed post-training."),
}

LADDER = [("qwen", 0.5), ("qwen1_5b", 1.5), ("qwen3b", 3.0), ("qwen7b", 7.0),
          ("qwen32b", 32.0), ("qwen72b", 72.0), ("mistral_large", 123.0)]
EXTRA = ["gpt4omini", "sonnet", "haiku"]


def load_questions():
    p = ROOT / "data/processed/questions.jsonl"
    return {json.loads(l)["qid"]: json.loads(l) for l in p.open(encoding="utf-8")}


def load_cb(model, questions):
    for cand in (f"generations_{model}/closed_book.jsonl",
                 f"generations/closed_book.jsonl" if model == "haiku" else None):
        if not cand:
            continue
        p = ROOT / "data/processed" / cand
        if p.exists():
            out = {}
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                q = questions.get(r["qid"])
                if q:
                    out[r["qid"]] = 100.0 * token_f1(r["prediction"], q["gold_answer"])
            return out
    return {}


def block_boot_gap(scores, pre, post, quiz_of, B, seed):
    """Paired-by-quiz bootstrap of mean(pre) - mean(post)."""
    import random
    rng = random.Random(seed)
    quizzes = sorted({quiz_of[q] for q in list(pre) + list(post)})
    by_q = defaultdict(lambda: ([], []))
    for q in pre:
        by_q[quiz_of[q]][0].append(scores[q])
    for q in post:
        by_q[quiz_of[q]][1].append(scores[q])
    draws = []
    for _ in range(B):
        picked = [quizzes[rng.randrange(len(quizzes))] for _ in quizzes]
        a, b = [], []
        for qz in picked:
            a.extend(by_q[qz][0]); b.extend(by_q[qz][1])
        if a and b:
            draws.append(statistics.fmean(a) - statistics.fmean(b))
    draws.sort()
    return (draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))],
            sum(1 for d in draws if d <= 0) / len(draws))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--output", type=Path, default=ROOT / "results/verified_slice.json")
    args = ap.parse_args()

    questions = load_questions()
    quiz_of = {qid: quiz_id(qid) for qid in questions}
    year = {qid: str(q["date"])[:4] for qid, q in questions.items()}

    # 2023 vs 2026: 2026 post-dates every documented training window here
    # (the latest is Claude's July 2025), and 2023 is unambiguously before.
    pre = [q for q in questions if year[q] == "2023"]
    post = [q for q in questions if year[q] == "2026"]

    out = {
        "scope": "Per-model cutoff evidence, and the boundary gap recomputed on a "
                 "slice that is post-training for every generator under any "
                 "documented reading (2026 questions only).",
        "why": "The 2025 analysis boundary was selected from an observed curve and "
               "is not post-training for the Claude models, whose training data "
               "runs through July 2025. Results at that boundary are exploratory; "
               "this slice is the confirmatory one.",
        "verified_slice": {"pre_year": "2023", "post_year": "2026",
                           "n_pre": len(pre), "n_post": len(post)},
        "cutoff_evidence": CUTOFF_EVIDENCE,
        "models": {},
    }

    print(f"verified slice: 2023 (n={len(pre)}) vs 2026 (n={len(post)})\n")
    for key, params in LADDER + [(k, None) for k in EXTRA]:
        sc = load_cb(key, questions)
        if not sc:
            print(f"  [skip] {key}")
            continue
        p_ok = [q for q in pre if q in sc]
        q_ok = [q for q in post if q in sc]
        gap = statistics.fmean(sc[q] for q in p_ok) - statistics.fmean(sc[q] for q in q_ok)
        lo, hi, p_le0 = block_boot_gap(sc, p_ok, q_ok, quiz_of, args.B, args.seed)
        p_two = min(1.0, 2 * min(p_le0, 1 - p_le0))
        p_two = max(p_two, 2.0 / (args.B + 1))
        ev = CUTOFF_EVIDENCE.get(key, {})
        out["models"][key] = {
            "name": ev.get("name", key), "params_B": params,
            "evidence_class": ev.get("evidence_class"),
            "gap_2023_vs_2026": round(gap, 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "two_sided_p_plus_one": round(p_two, 5),
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
        print(f"  {ev.get('name', key):20s} gap {gap:+7.2f}  "
              f"CI [{lo:+7.2f},{hi:+7.2f}]  p={p_two:.4f}"
              f"{'  *' if (lo > 0 or hi < 0) else ''}")

    n_excl = sum(1 for v in out["models"].values() if v["excludes_zero"])
    out["summary"] = (
        f"On the verified 2026 slice, {n_excl} of {len(out['models'])} generators "
        "have a boundary gap whose quiz-block-bootstrap interval excludes zero. "
        "This is the confirmatory contrast; the 2025-boundary numbers elsewhere in "
        "the paper are exploratory because that boundary was selected from an "
        "observed curve and is not post-training for the Claude models."
    )
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{out['summary']}\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
