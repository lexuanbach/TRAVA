#!/usr/bin/env python3
"""Does the support interaction survive a different context budget?

The paper fixes top-k=5. That choice is not innocent for Layer 3, because the
support label is "at least one of the k shown passages carries the gold span",
so k moves the supported/unsupported partition itself. Shrinking k moves
questions out of the supported arm, growing k moves them in, and the interaction
could in principle be an artefact of where k sits.

We re-answer the SAME questions with the SAME generators under masked rankings
at k=3 and k=10 and recompute

    Delta_sup = Delta_RAG | supported  -  Delta_RAG | unsupported

on the Static slice, quiz-clustered, alongside the k=5 result the paper reports.

One caveat we state rather than hide: Hybrid fuses BM25 and BGE by reciprocal
rank fusion, so a deeper candidate list changes the fusion, not only the number
of passages shown. The k=10 arm therefore varies retrieval depth and context
budget together, which is what a practitioner changes in practice, but it is not
a pure truncation of the k=5 ranking. The k=3 arm IS a pure truncation.

Outputs results/k_ablation.json.
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
sys.dont_write_bytecode = True
from metrics import gold_in_text, token_f1                        # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
BOUNDARY = 2025
# rungs servable through transformers; the GGUF rungs are excluded because a
# two-condition ablation on a 123B GGUF costs about a day for one extra point
PANEL = ["qwen1_5b", "qwen3b", "qwen7b"]
ARMS = [("k3", "retrieval_masked_k3.json", "hybrid_masked_k3"),
        ("k5", "retrieval_masked.json", "hybrid_masked"),
        ("k10", "retrieval_masked_k10.json", "hybrid_masked_k10")]


def quiz_of(qid):
    return qid.split("_")[0]


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()
             if l.strip())}


def load_gen(model, cond):
    f = PROC / (f"generations_{model}" if model else "generations") / f"{cond}.jsonl"
    if not f.is_file():
        return {}
    return {r["qid"]: (r["prediction"] or "") for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def interaction(units, B, seed):
    cl = list(units.values())
    sup = [v for a, _ in cl for v in a]
    uns = [v for _, b in cl for v in b]
    if len(sup) < 8 or len(uns) < 8 or len(cl) < 3:
        return None
    obs = statistics.fmean(sup) - statistics.fmean(uns)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        sa, sb = [], []
        for _ in cl:
            a, b = cl[rng.randrange(len(cl))]
            sa.extend(a); sb.extend(b)
        if sa and sb:
            draws.append(statistics.fmean(sa) - statistics.fmean(sb))
    if len(draws) < B // 2:
        return None
    draws.sort()
    n = len(draws)
    le = sum(1 for v in draws if v <= 0)
    lo, hi = draws[int(0.025 * n)], draws[int(0.975 * n)]
    return {"interaction": round(obs, 3), "ci95": [round(lo, 3), round(hi, 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, n - le + 1) / (n + 1)), 5),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "gain_supported": round(statistics.fmean(sup), 3),
            "gain_unsupported": round(statistics.fmean(uns), 3),
            "n_supported": len(sup), "n_unsupported": len(uns)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--output", type=Path, default=RES / "k_ablation.json")
    args = ap.parse_args()

    qs = load_questions()
    corpus = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    corpus.setdefault(d["doc_id"], d.get("text", ""))

    rankings, support = {}, {}
    for arm, rfile, _ in ARMS:
        f = PROC / rfile
        if not f.is_file():
            continue
        rankings[arm] = json.loads(f.read_text())["hybrid"]
        support[arm] = {
            qid: any(gold_in_text(qs[qid]["gold_answer"], corpus.get(d, ""))
                     for d in docs)
            for qid, docs in rankings[arm].items() if qid in qs
        }

    out = {
        "question": "Does the Layer-3 support interaction depend on the top-k "
                    "context budget the paper fixes at 5?",
        "arms": {a: {"ranking": r, "condition": c} for a, r, c in ARMS
                 if a in rankings},
        "caveat": ("Hybrid is reciprocal rank fusion over BM25 and BGE, so a "
                   "deeper candidate list re-fuses rather than merely extending "
                   "the k=5 list. k=3 is a pure truncation of k=5; k=10 varies "
                   "retrieval depth and context budget together."),
        "partition_shift": {},
        "per_model": {},
    }

    base = support.get("k5", {})
    for arm in rankings:
        cur = support[arm]
        shared = sorted(set(cur) & set(base))
        out["partition_shift"][arm] = {
            "n_questions": len(cur),
            "support_rate": round(sum(cur.values()) / len(cur), 4) if cur else None,
            "moved_arm_vs_k5": sum(1 for q in shared if cur[q] != base[q]),
            "n_shared": len(shared),
        }

    for m in PANEL:
        cb = load_gen(m, "closed_book")
        if not cb:
            continue
        rec = {}
        for arm, _rfile, cond in ARMS:
            if arm not in rankings:
                continue
            ma = load_gen(m, cond)
            if len(ma) < 878:
                rec[arm] = {"incomplete": len(ma)}
                continue
            units = collections.defaultdict(lambda: ([], []))
            for qid in sorted(set(cb) & set(ma) & set(qs)):
                q = qs[qid]
                if q["year"] >= BOUNDARY:
                    continue                       # Static slice, as in the paper
                g = 100.0 * (token_f1(ma[qid], q["gold_answer"])
                             - token_f1(cb[qid], q["gold_answer"]))
                units[quiz_of(qid)][0 if support[arm].get(qid) else 1].append(g)
            r = interaction(units, args.B, args.seed)
            if r:
                rec[arm] = r
        if rec:
            out["per_model"][m] = rec

    wanted = [a for a, _, _ in ARMS if a in rankings]
    complete = {m: r for m, r in out["per_model"].items()
                if all(isinstance(r.get(a), dict) and "interaction" in r[a]
                       for a in wanted)}
    n_all = sum(1 for r in complete.values()
                if all(r[a]["excludes_zero"] for a in wanted))
    if not complete:
        verdict = ("INCOMPLETE: no generator has all of "
                   f"{', '.join(wanted)}, so the ablation is not yet decided")
    elif n_all == len(complete):
        verdict = "The support interaction holds at every context budget tested"
    else:
        verdict = "The support interaction does NOT hold at every context budget"
    out["summary"] = {
        "arms_available": wanted,
        "n_models_with_all_arms": len(complete),
        "n_excluding_zero_in_every_arm": n_all,
        "verdict": verdict,
    }

    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    hdr = "  ".join(f"{a:>18s}" for a, _, _ in ARMS if a in rankings)
    print(f"{'model':10s}{hdr}")
    for m, rec in out["per_model"].items():
        cells = []
        for arm, _, _ in ARMS:
            v = rec.get(arm)
            cells.append(f"{v['interaction']:7.1f} {str(v['ci95']):>10s}"
                         if v and "interaction" in v else f"{'--':>18s}")
        print(f"{m:10s}" + "  ".join(cells))
    print()
    for arm, v in out["partition_shift"].items():
        print(f"{arm:4s} support rate {v['support_rate']}, "
              f"{v['moved_arm_vs_k5']} of {v['n_shared']} questions change arm vs k=5")
    print(out["summary"]["verdict"])
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
