#!/usr/bin/env python3
"""Does the support interaction survive the stricter correctness endpoint?

The paper reports that token-F1 has condition-dependent errors, and that the
capability trend attenuates and loses significance under the choice-grounded
endpoint. The SUPPORT INTERACTION, the one result the audit leaves standing,
therefore has to be checked under that stricter endpoint too. If it were an
artifact of lexical overlap, this is where it would show.

The choice-grounded endpoint maps each free-form answer to the RealTimeQA option
it best matches and scores against the original gold option, so it is a binary
correctness measure that cannot be inflated by surface overlap with the passage.

We recompute

    Delta_sup = Delta_RAG | supported  -  Delta_RAG | unsupported

on masked contexts under BOTH endpoints, quiz-clustered, and report them side by
side. A sign change or a collapse to zero under the strict endpoint would mean
the interaction is a lexical artifact.

Outputs results/support_endpoint_robustness.json.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.dont_write_bytecode = True
from metrics import gold_in_text, token_f1, normalize            # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
BOUNDARY = 2025
PANEL = ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
         "mistral_large", "", "sonnet", "gpt4omini"]
NAMES = {"": "haiku"}


def quiz_of(qid):
    return qid.split("_")[0]


def load_questions():
    out = {}
    for line in (PROC / "questions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        ch = d.get("choices")
        if isinstance(ch, str):
            try:
                ch = ast.literal_eval(ch)
            except Exception:
                ch = None
        d["choices"] = ch if isinstance(ch, list) else None
        out[d["qid"]] = d
    return out


def load(model, cond):
    p = PROC / (f"generations_{model}" if model else "generations") / f"{cond}.jsonl"
    if not p.is_file():
        return {}
    return {r["qid"]: (r["prediction"] or "") for r in
            (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def nearest_choice(pred, choices):
    """Index of the option the prediction is closest to, or None if ambiguous.

    Mirrors pipeline/choice_grounded_audit.py: exact normalised match first,
    then containment, then best token-F1, with ties refused rather than guessed.
    """
    p = normalize(pred)
    if not p:
        return None
    exact = [k for k, c in enumerate(choices) if normalize(c) == p]
    if len(exact) == 1:
        return exact[0]
    contained = [k for k, c in enumerate(choices)
                 if p and (p in normalize(c) or normalize(c) in p)]
    if len(contained) == 1:
        return contained[0]
    scores = [token_f1(pred, c) for c in choices]
    best = max(scores)
    if best <= 0:
        return None
    top = [k for k, s in enumerate(scores) if s == best]
    return top[0] if len(top) == 1 else None


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
            "n_supported": len(sup), "n_unsupported": len(uns)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--output", type=Path,
                    default=RES / "support_endpoint_robustness.json")
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

    ranked = json.loads((PROC / "retrieval_masked.json").read_text())["hybrid"]
    n_with_choices = sum(1 for q in qs.values() if q["choices"])

    out = {
        "scope": "The Layer-3 support interaction recomputed under the strict "
                 "choice-grounded endpoint alongside token-F1, on masked "
                 "contexts, Static slice.",
        "why": ("Token-F1 has condition-dependent errors, so the interaction "
                "could in principle be a lexical-overlap artifact. The "
                "choice-grounded endpoint is binary and cannot be inflated by "
                "surface overlap with the retrieved passage."),
        "endpoints": {
            "token_f1": "SQuAD-normalised token F1, scaled to 0-100",
            "choice_grounded": "free-form answer mapped to its nearest "
                               "RealTimeQA option, scored against the gold "
                               "option, 0 or 100; ambiguous maps score 0",
        },
        "n_questions_with_choices": n_with_choices,
        "per_model": {},
    }

    for m in PANEL:
        cb, ma = load(m, "closed_book"), load(m, "hybrid_masked")
        if not cb or not ma:
            continue
        rec = {}
        for endpoint in ("token_f1", "choice_grounded"):
            units = collections.defaultdict(lambda: ([], []))
            for qid in sorted(set(cb) & set(ma) & set(qs)):
                q = qs[qid]
                if q["year"] >= BOUNDARY:
                    continue                       # Static slice, as in the paper
                if endpoint == "choice_grounded" and not q["choices"]:
                    continue
                texts = [corpus.get(d, "") for d in (ranked.get(qid) or [])]
                sup = any(gold_in_text(q["gold_answer"], t) for t in texts if t)

                if endpoint == "token_f1":
                    g = 100.0 * (token_f1(ma[qid], q["gold_answer"])
                                 - token_f1(cb[qid], q["gold_answer"]))
                else:
                    gi = q.get("gold_answer_idx")
                    if gi is None:
                        continue
                    a = nearest_choice(ma[qid], q["choices"])
                    b = nearest_choice(cb[qid], q["choices"])
                    g = 100.0 * ((a == gi) - (b == gi))
                units[quiz_of(qid)][0 if sup else 1].append(g)
            r = interaction(units, args.B, args.seed)
            if r:
                rec[endpoint] = r
        if len(rec) == 2:
            rec["sign_agrees"] = bool(
                (rec["token_f1"]["interaction"] > 0)
                == (rec["choice_grounded"]["interaction"] > 0))
            out["per_model"][NAMES.get(m, m) or m] = rec

    n = len(out["per_model"])
    n_sig_f1 = sum(1 for r in out["per_model"].values()
                   if r["token_f1"]["excludes_zero"])
    n_sig_cg = sum(1 for r in out["per_model"].values()
                   if r["choice_grounded"]["excludes_zero"])
    n_same = sum(1 for r in out["per_model"].values() if r["sign_agrees"])
    out["summary"] = {
        "n_models": n,
        "n_excluding_zero_token_f1": n_sig_f1,
        "n_excluding_zero_choice_grounded": n_sig_cg,
        "n_sign_agrees": n_same,
        "verdict": ("The interaction survives the strict endpoint"
                    if n_sig_cg == n else
                    "The interaction does NOT fully survive the strict endpoint"),
    }

    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{'model':14s}{'token-F1':>22s}{'choice-grounded':>24s}")
    for m, r in out["per_model"].items():
        a, b = r["token_f1"], r["choice_grounded"]
        print(f"{m:14s}{a['interaction']:8.2f} {str(a['ci95']):>13s}"
              f"{b['interaction']:9.2f} {str(b['ci95']):>13s}")
    print(f"\nexcluding zero: {n_sig_f1}/{n} under token-F1, "
          f"{n_sig_cg}/{n} under choice-grounded; signs agree {n_same}/{n}")
    print(out["summary"]["verdict"])
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
