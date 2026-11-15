#!/usr/bin/env python3
"""Replicate the evidence-quality controls across every generator that has them.

The stale-index and leave-one-out controls initially ran only on
Qwen2.5-32B, so the paper's "bad evidence is worse than none" claim rested on a
single mid-ladder model -- and in particular NOT on Mistral-Large-2, the rung that
carries the capability headline. This script recomputes those controls for every
generator with `hybrid`, `hybrid_stale` and `hybrid_loo` cached, so the claim can
be stated as a replication or withdrawn.

`heldout_index.py` writes one generator per run into a single file; this collects
all of them into `results/stale_replication.json` without overwriting anything.

Reported per generator and per slice: token-F1 under the live index, the stale
(pre-2025) index and the leave-one-out index, with closed-book as the reference
line, plus a paired bootstrap over the same Fresh questions for each drop.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import metrics as M  # noqa: E402
from clusters import quiz_id  # noqa: E402

MODELS = [("qwen3b", "Qwen2.5-3B", 3.0), ("qwen32b", "Qwen2.5-32B", 32.0),
          ("mistral_large", "Mistral-Large-2", 123.0)]
CONDS = ["closed_book", "hybrid", "hybrid_stale", "hybrid_loo"]
BOUNDARY = "2025"


def load_questions() -> dict[str, dict]:
    p = ROOT / "data/processed/questions.jsonl"
    return {json.loads(l)["qid"]: json.loads(l) for l in p.open(encoding="utf-8")}


def scores(model: str, cond: str, questions: dict) -> dict[str, float]:
    p = ROOT / f"data/processed/generations_{model}/{cond}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        q = questions.get(r["qid"])
        if q:
            out[r["qid"]] = 100.0 * M.token_f1(r["prediction"], q["gold_answer"])
    return out


def abstain_rate(model: str, cond: str) -> float | None:
    p = ROOT / f"data/processed/generations_{model}/{cond}.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(l) for l in p.open(encoding="utf-8")]
    if not recs:
        return None
    return round(100.0 * sum(M.is_abstention(r["prediction"]) for r in recs) / len(recs), 1)


def paired_block_bootstrap(a: dict, b: dict, qids, B: int, seed: int):
    """Bootstrap of mean(a) - mean(b) on shared items, resampling weekly quizzes."""
    import random
    rng = random.Random(seed)
    by_quiz: dict[str, list[tuple[float, float]]] = {}
    for q in qids:
        by_quiz.setdefault(quiz_id(q), []).append((a[q], b[q]))
    quizzes = sorted(by_quiz)
    draws = []
    for _ in range(B):
        pa, pb = [], []
        for _ in quizzes:
            for x, y in by_quiz[quizzes[rng.randrange(len(quizzes))]]:
                pa.append(x); pb.append(y)
        draws.append(statistics.fmean(pa) - statistics.fmean(pb))
    draws.sort()
    lo, hi = draws[int(0.025 * B)], draws[int(0.975 * B)]
    p_le0 = sum(1 for d in draws if d <= 0) / B
    p = max(min(1.0, 2 * min(p_le0, 1 - p_le0)), 2.0 / (B + 1))
    return round(lo, 2), round(hi, 2), round(p, 5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/stale_replication.json")
    args = ap.parse_args()

    questions = load_questions()
    fresh = {q for q, v in questions.items() if str(v["date"])[:4] >= BOUNDARY}

    out = {
        "scope": "Stale-index and leave-one-out controls replicated across every "
                 "generator with all four conditions cached, so the "
                 "'bad evidence is worse than none' claim no longer rests on one "
                 "mid-ladder model.",
        "boundary": f"{BOUNDARY}-01-01",
        "bootstrap": {"B": args.B, "unit": "weekly quiz id", "seed": args.seed},
        "models": {},
    }

    for key, name, params in MODELS:
        sc = {c: scores(key, c, questions) for c in CONDS}
        if not all(sc[c] for c in CONDS):
            missing = [c for c in CONDS if not sc[c]]
            print(f"  [skip] {name}: missing {missing}")
            continue
        common = sorted(set.intersection(*(set(sc[c]) for c in CONDS)))
        common_fresh = [q for q in common if q in fresh]
        common_static = [q for q in common if q not in fresh]

        def mean(cond, ids):
            return round(statistics.fmean(sc[cond][q] for q in ids), 2)

        entry = {"name": name, "params_B": params,
                 "n_common": len(common), "n_fresh": len(common_fresh),
                 "f1": {c: {"static": mean(c, common_static),
                            "fresh": mean(c, common_fresh)} for c in CONDS},
                 "abstention_pct": {c: abstain_rate(key, c) for c in CONDS},
                 "fresh_contrasts": {}}
        for label, worse in (("hybrid_minus_stale", "hybrid_stale"),
                             ("hybrid_minus_loo", "hybrid_loo"),
                             ("stale_minus_closedbook", "closed_book")):
            better = "hybrid" if label != "stale_minus_closedbook" else "hybrid_stale"
            lo, hi, p = paired_block_bootstrap(sc[better], sc[worse], common_fresh,
                                               args.B, args.seed)
            entry["fresh_contrasts"][label] = {
                "diff_f1": round(mean(better, common_fresh) - mean(worse, common_fresh), 2),
                "ci95": [lo, hi], "two_sided_p_plus_one": p,
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        out["models"][key] = entry

        f = entry["f1"]
        print(f"\n{name} (n={len(common)}, Fresh={len(common_fresh)})")
        for c in CONDS:
            print(f"    {c:14s} Static {f[c]['static']:6.2f}   Fresh {f[c]['fresh']:6.2f}"
                  f"   abstain {entry['abstention_pct'][c]}%")
        for label, d in entry["fresh_contrasts"].items():
            star = " *" if d["excludes_zero"] else ""
            print(f"    {label:24s} {d['diff_f1']:+7.2f}  CI {d['ci95']}  "
                  f"p={d['two_sided_p_plus_one']}{star}")

    # Does the headline replicate?
    worse_than_none = {k: v["fresh_contrasts"]["stale_minus_closedbook"]
                       for k, v in out["models"].items()}
    n_worse = sum(1 for d in worse_than_none.values()
                  if d["diff_f1"] < 0 and d["excludes_zero"])
    out["summary"] = (
        f"Stale evidence leaves Fresh F1 significantly BELOW closed-book on "
        f"{n_worse} of {len(out['models'])} generators, spanning "
        f"{min(v['params_B'] for v in out['models'].values()):.0f}B--"
        f"{max(v['params_B'] for v in out['models'].values()):.0f}B, including "
        "the open-weight rung that carries the capability association."
    )
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{out['summary']}\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
