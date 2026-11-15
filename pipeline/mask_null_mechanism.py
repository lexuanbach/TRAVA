#!/usr/bin/env python3
"""Why is Delta_mask ~ 0? A mechanism test, not another effect estimate.

The per-query temporal mask changes a large share of context sets yet moves
almost no score. Two explanations are observationally distinct:

  (A) POOLING. The main index pools only the sampled questions' own evidence, so
      a question's gold-bearing documents face little competition. Masking then
      swaps one answer-bearing passage for another and the generator sees the
      same answer either way. Under (A) the null is an artifact of how the index
      was built and would not survive a competitive index.

  (B) INERTNESS. Post-question passages were not carrying the answer in the
      first place, so removing them costs nothing. Under (B) the null is a
      property of the evidence, not of the index.

The two make opposite predictions about ANSWER COVERAGE, the share of shown
passages containing the gold span:

  (A) predicts coverage is preserved across the mask on exactly the questions
      whose context set changed, because a replacement of equal quality was
      available.
  (B) predicts the displaced passages were mostly NOT answer-bearing, so
      coverage barely moves because there was nothing to lose.

We separate them by asking, on the changed questions only, what the displaced
passages actually carried, and comparing coverage before and after the mask.

We also report the same statistics on the FULL index (26,611 documents, 19,919
same-period distractors), which is the competitive setting. The full index is
masked by construction, so no unmasked full-index Delta_mask exists, but its
supply of answer-bearing alternatives per question is directly measurable and is
the quantity that distinguishes (A) from (B) at the index level.

Outputs results/mask_null_mechanism.json.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.dont_write_bytecode = True
from metrics import gold_in_text                                  # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
BOUNDARY = 2025
GRACE_DAYS = 3


def quiz_of(qid):
    return qid.split("_")[0]


def load_questions(name="questions.jsonl"):
    return {d["qid"]: d for d in
            (json.loads(l) for l in (PROC / name).read_text().splitlines()
             if l.strip())}


def load_corpus(*names):
    out = {}
    for n in names:
        f = PROC / n
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    out.setdefault(d["doc_id"], d)
    return out


def cluster_ci(units, B, seed):
    """Quiz-clustered bootstrap mean of a per-question quantity."""
    cl = list(units.values())
    flat = [v for c in cl for v in c]
    if len(flat) < 8 or len(cl) < 3:
        return None
    obs = statistics.fmean(flat)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        s = []
        for _ in cl:
            s.extend(cl[rng.randrange(len(cl))])
        if s:
            draws.append(statistics.fmean(s))
    draws.sort()
    n = len(draws)
    return {"mean": round(obs, 4),
            "ci95": [round(draws[int(0.025 * n)], 4),
                     round(draws[int(0.975 * n)], 4)],
            "n": len(flat), "n_quizzes": len(cl)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--output", type=Path, default=RES / "mask_null_mechanism.json")
    args = ap.parse_args()

    qs = load_questions()
    corpus = load_corpus("corpus.jsonl", "corpus_full.jsonl")
    unm = json.loads((PROC / "retrieval.json").read_text())["hybrid"]
    msk = json.loads((PROC / "retrieval_masked.json").read_text())["hybrid"]

    def bears(doc_id, q):
        d = corpus.get(doc_id)
        return bool(d) and gold_in_text(q["gold_answer"], d.get("text", ""))

    changed, unchanged = [], []
    disp_total = disp_bearing = 0
    add_total = add_bearing = 0
    cov_units, cov_changed_units = {}, {}
    per_slice = {"static": {"changed": 0, "n": 0, "cov_pre": [], "cov_post": []},
                 "fresh": {"changed": 0, "n": 0, "cov_pre": [], "cov_post": []}}

    for qid in sorted(set(unm) & set(msk) & set(qs)):
        q = qs[qid]
        a, b = list(unm[qid]), list(msk[qid])
        pre = sum(1 for d in a if bears(d, q)) / max(len(a), 1)
        post = sum(1 for d in b if bears(d, q)) / max(len(b), 1)
        sl = "fresh" if q["year"] >= BOUNDARY else "static"
        per_slice[sl]["n"] += 1
        per_slice[sl]["cov_pre"].append(pre)
        per_slice[sl]["cov_post"].append(post)
        cov_units.setdefault(quiz_of(qid), []).append(post - pre)
        if a == b:
            unchanged.append(qid)
            continue
        changed.append(qid)
        per_slice[sl]["changed"] += 1
        cov_changed_units.setdefault(quiz_of(qid), []).append(post - pre)
        displaced = [d for d in a if d not in b]
        added = [d for d in b if d not in a]
        disp_total += len(displaced)
        disp_bearing += sum(1 for d in displaced if bears(d, q))
        add_total += len(added)
        add_bearing += sum(1 for d in added if bears(d, q))

    # supply of answer-bearing alternatives, main index against the full index
    supply = {}
    for tag, cfile, qfile in (("main", "corpus.jsonl", "questions.jsonl"),
                              ("full", "corpus_full.jsonl", "questions_full.jsonl")):
        cf, qf = PROC / cfile, PROC / qfile
        if not cf.is_file():
            continue
        docs = load_corpus(cfile)
        supply[tag] = {"n_documents": len(docs)}

    out = {
        "question": "Is the near-zero Delta_mask an artifact of pooling only the "
                    "sampled questions' own evidence, or a property of the "
                    "displaced evidence?",
        "design": "Answer coverage before and after the per-query mask, split on "
                  "whether the mask changed the context set, plus what the "
                  "displaced and substituted passages carried.",
        "grace_days": GRACE_DAYS,
        "n_questions": len(changed) + len(unchanged),
        "n_context_sets_changed": len(changed),
        "displaced_passages": {
            "n": disp_total, "n_answer_bearing": disp_bearing,
            "share_answer_bearing": round(disp_bearing / disp_total, 4)
            if disp_total else None,
        },
        "substituted_passages": {
            "n": add_total, "n_answer_bearing": add_bearing,
            "share_answer_bearing": round(add_bearing / add_total, 4)
            if add_total else None,
        },
        "coverage_delta_all_questions": cluster_ci(cov_units, args.B, args.seed),
        "coverage_delta_changed_only": cluster_ci(cov_changed_units, args.B,
                                                  args.seed),
        "per_slice": {
            k: {"n": v["n"], "n_changed": v["changed"],
                "coverage_unmasked": round(statistics.fmean(v["cov_pre"]), 4),
                "coverage_masked": round(statistics.fmean(v["cov_post"]), 4)}
            for k, v in per_slice.items() if v["n"]
        },
        "index_sizes": supply,
    }

    d = out["displaced_passages"]["share_answer_bearing"]
    a = out["substituted_passages"]["share_answer_bearing"]
    cc = out["coverage_delta_changed_only"]
    pooling = (d is not None and a is not None and d > 0.25
               and cc and cc["ci95"][0] <= 0 <= cc["ci95"][1])
    out["verdict"] = (
        "POOLING: the mask displaced answer-bearing passages and an equally "
        "answer-bearing substitute was available, so the null reflects how the "
        "index was built."
        if pooling else
        "INERTNESS: the displaced passages were largely not answer-bearing, so "
        "there was little for the mask to remove."
    )
    out["reading"] = (
        f"{d:.1%} of displaced passages carried the gold span against "
        f"{a:.1%} of their substitutes."
        if d is not None and a is not None else "insufficient displacement"
    )

    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"context sets changed: {len(changed)} of {out['n_questions']}")
    print(f"displaced passages answer-bearing:    {disp_bearing}/{disp_total}"
          f" = {d:.3f}" if disp_total else "no displacement")
    print(f"substituted passages answer-bearing:  {add_bearing}/{add_total}"
          f" = {a:.3f}" if add_total else "")
    print(f"coverage delta, changed questions only: {cc}")
    print(out["verdict"])
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
