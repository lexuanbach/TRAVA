#!/usr/bin/env python3
"""Full-scale closed-book cliff: 4,182 questions instead of the sampled 878.

Reports the boundary split and the clean 2023-vs-2026 contrast with cluster-robust
(block bootstrap over weekly quizzes) two-sided inference, and -- because the point
of the exercise is a NULL -- an equivalence bound: the largest cliff the data can
still hide. Also checks the sampled 878 against the full set on the SAME questions,
so we can say whether the sample was unrepresentative or merely small.

Writes results/fullscale.json.
"""
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import token_f1
from clusters import quiz_id  # canonical resampling unit

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
B = 4000
GEN = {"qwen32b": "generations_qwen32b", "qwen72b": "generations_qwen72b",
       "gpt4omini": "generations_gpt4omini"}


def rng(label):
    return np.random.default_rng(zlib.crc32(label.encode()) & 0xFFFFFFFF)


def batch_of(qid):
    return quiz_id(qid)


def block_boot(a_vals, a_b, b_vals, b_b, label):
    r = rng(label)
    ga, gb = defaultdict(list), defaultdict(list)
    for v, k in zip(a_vals, a_b):
        ga[k].append(v)
    for v, k in zip(b_vals, b_b):
        gb[k].append(v)
    ka, kb = list(ga), list(gb)
    obs = np.mean(a_vals) - np.mean(b_vals)
    d = []
    for _ in range(B):
        sa = r.choice(len(ka), len(ka))
        sb = r.choice(len(kb), len(kb))
        d.append(np.mean([x for i in sa for x in ga[ka[i]]]) -
                 np.mean([x for i in sb for x in gb[kb[i]]]))
    d = np.array(d)
    return {"diff": round(float(obs), 2),
            "se": round(float(d.std()), 2),
            "ci95": [round(float(np.percentile(d, 2.5)), 2),
                     round(float(np.percentile(d, 97.5)), 2)],
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4),
            "n": [len(a_vals), len(b_vals)],
            "n_clusters": [len(ka), len(kb)]}


def main():
    Qf = {q["qid"]: q for q in
          (json.loads(l) for l in (PROC / "questions_full.jsonl").read_text().splitlines())}
    sampled = {json.loads(l)["qid"] for l in
               (PROC / "questions.jsonl").read_text().splitlines()}
    res = {"n_questions_full": len(Qf), "n_questions_sampled": len(sampled)}

    for tag, d in GEN.items():
        f = PROC / d / "closed_book_full.jsonl"
        if not f.exists():
            continue
        sc = {r["qid"]: token_f1(r["prediction"], Qf[r["qid"]]["gold_answer"]) * 100
              for r in (json.loads(l) for l in f.read_text().splitlines())
              if r["qid"] in Qf}
        if len(sc) < 0.95 * len(Qf):
            print(f"skip {tag}: only {len(sc)}/{len(Qf)} complete")
            continue
        row = {"n_scored": len(sc)}

        def contrast(sel_a, sel_b, label):
            A = [q for q in sc if sel_a(Qf[q]["year"])]
            Bq = [q for q in sc if sel_b(Qf[q]["year"])]
            return block_boot([sc[q] for q in A], [batch_of(q) for q in A],
                              [sc[q] for q in Bq], [batch_of(q) for q in Bq],
                              f"{tag}:{label}")

        row["boundary_2025"] = contrast(lambda y: y < 2025, lambda y: y >= 2025, "bnd")
        row["clean_2023_vs_2026"] = contrast(lambda y: y == 2023, lambda y: y == 2026, "cln")
        # the sampled subset, same model, same estimator -> was 878 unrepresentative?
        S = [q for q in sc if q in sampled and Qf[q]["year"] < 2025]
        F = [q for q in sc if q in sampled and Qf[q]["year"] >= 2025]
        row["sampled_subset_only"] = block_boot(
            [sc[q] for q in S], [batch_of(q) for q in S],
            [sc[q] for q in F], [batch_of(q) for q in F], f"{tag}:sub")
        row["by_year"] = {str(y): round(float(np.mean(
            [sc[q] for q in sc if Qf[q]["year"] == y])), 2)
            for y in sorted({Qf[q]["year"] for q in sc})}
        # equivalence bound: largest cliff still compatible with the data
        for k in ("boundary_2025", "clean_2023_vs_2026"):
            row[k]["largest_cliff_not_excluded"] = row[k]["ci95"][1]
        res[tag] = row

    (RES / "fullscale.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    for tag in GEN:
        if tag not in res:
            continue
        r = res[tag]
        print(f"\n=== {tag}: {r['n_scored']} questions (was 878) ===")
        for k in ("sampled_subset_only", "boundary_2025", "clean_2023_vs_2026"):
            v = r[k]
            print(f"  {k:22s} {v['diff']:+6.2f}  CI {v['ci95']}  p={v['p_two_sided']:.4f}  "
                  f"n={v['n'][0]}/{v['n'][1]}")
        print(f"  by year: {r['by_year']}")
        ub = r["clean_2023_vs_2026"]["largest_cliff_not_excluded"]
        print(f"  -> clean contrast excludes any cliff above {ub:+.1f} F1 "
              f"(Mistral-Large-2 is +12.0)")


if __name__ == "__main__":
    main()
