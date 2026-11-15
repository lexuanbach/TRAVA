#!/usr/bin/env python3
"""Cluster-robust re-test of every headline result.

RealTimeQA ships questions in weekly quiz batches (199 of them here, median 3
questions). Questions within a quiz share a news cycle, so they are not
independent -- and because a quiz happens on ONE date, every batch falls entirely
on one side of the Static/Fresh boundary (verified: 0 batches straddle it). Batch
is therefore NESTED IN slice, which is the worst case: batch-level variance loads
directly onto the contrast we care about.

Resampling questions i.i.d. (what the paper does) understates the SE. This
re-tests every headline with a BLOCK bootstrap that resamples whole quizzes.

Writes results/cluster_robust.json.
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
GEN = {"qwen": "generations_qwen", "qwen1_5b": "generations_qwen1_5b",
       "qwen3b": "generations_qwen3b", "qwen7b": "generations_qwen7b",
       "qwen32b": "generations_qwen32b", "qwen72b": "generations_qwen72b",
       "mistral_large": "generations_mistral_large", "haiku": "generations",
       "sonnet": "generations_sonnet"}
NAME = {"qwen": "Qwen2.5-0.5B", "qwen1_5b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B",
        "qwen7b": "Qwen2.5-7B", "qwen32b": "Qwen2.5-32B", "qwen72b": "Qwen2.5-72B",
        "mistral_large": "Mistral-Large-2", "haiku": "Claude Haiku 4.5",
        "sonnet": "Claude Sonnet 4.5"}
B = 4000


def rng(label):
    return np.random.default_rng(zlib.crc32(label.encode()) & 0xFFFFFFFF)


def batch_of(qid):
    return quiz_id(qid)


def load(tag, cond, Q):
    f = PROC / GEN[tag] / f"{cond}.jsonl"
    if not f.exists():
        return None
    return {r["qid"]: token_f1(r["prediction"], Q[r["qid"]]["gold_answer"]) * 100
            for r in (json.loads(l) for l in f.read_text().splitlines())
            if r["qid"] in Q}


def two_sample(vals_a, vals_b, batches_a, batches_b, label):
    """Block bootstrap: resample whole quizzes within each slice.

    Reported p is TWO-SIDED. The manuscript's one-sided p is also given, since
    the direction was not pre-registered.
    """
    r = rng(label)
    ga, gb = defaultdict(list), defaultdict(list)
    for v, b in zip(vals_a, batches_a):
        ga[b].append(v)
    for v, b in zip(vals_b, batches_b):
        gb[b].append(v)
    ka, kb = list(ga), list(gb)
    obs = np.mean(vals_a) - np.mean(vals_b)
    draws = []
    for _ in range(B):
        sa = r.choice(len(ka), len(ka))
        sb = r.choice(len(kb), len(kb))
        a = [x for i in sa for x in ga[ka[i]]]
        b = [x for i in sb for x in gb[kb[i]]]
        draws.append(np.mean(a) - np.mean(b))
    d = np.array(draws)
    return {"diff": round(float(obs), 2),
            "se_cluster": round(float(d.std()), 2),
            "ci95_cluster": [round(float(np.percentile(d, 2.5)), 2),
                             round(float(np.percentile(d, 97.5)), 2)],
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4),
            "p_one_sided": round(float((d <= 0).mean()), 4),
            "n_clusters": [len(ka), len(kb)], "n_items": [len(vals_a), len(vals_b)]}


def iid(vals_a, vals_b, label):
    r = rng(label + ":iid")
    a, b = np.asarray(vals_a), np.asarray(vals_b)
    d = np.array([r.choice(a, len(a)).mean() - r.choice(b, len(b)).mean() for _ in range(B)])
    return {"se_iid": round(float(d.std()), 2),
            "p_two_sided_iid": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}


def main():
    Q = {q["qid"]: q for q in
         (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines())}
    res = {"design": {}}

    b = defaultdict(list)
    for qid, q in Q.items():
        b[batch_of(qid)].append(q)
    res["design"] = {
        "n_quiz_batches": len(b),
        "median_batch_size": int(np.median([len(v) for v in b.values()])),
        "batches_straddling_boundary": sum(
            1 for v in b.values() if len({q["year"] >= 2025 for q in v}) > 1),
        "note": "Batch is nested in slice, so quiz-level variance loads onto the "
                "Static/Fresh contrast. i.i.d. resampling understates the SE.",
    }

    # ---- cliffs at the shared 2025 boundary --------------------------------
    cliffs = {}
    for t in GEN:
        cb = load(t, "closed_book", Q)
        if not cb:
            continue
        S = [q for q in cb if Q[q]["year"] < 2025]
        F = [q for q in cb if Q[q]["year"] >= 2025]
        r = two_sample([cb[q] for q in S], [cb[q] for q in F],
                       [batch_of(q) for q in S], [batch_of(q) for q in F], f"cliff:{t}")
        r.update(iid([cb[q] for q in S], [cb[q] for q in F], f"cliff:{t}"))
        r["se_inflation"] = round(r["se_cluster"] / r["se_iid"], 3)
        cliffs[t] = r
    res["cliff_boundary_2025"] = cliffs

    # ---- clean contrast 2023 vs 2026 ---------------------------------------
    clean = {}
    for t in GEN:
        cb = load(t, "closed_book", Q)
        if not cb:
            continue
        S = [q for q in cb if Q[q]["year"] == 2023]
        F = [q for q in cb if Q[q]["year"] == 2026]
        r = two_sample([cb[q] for q in S], [cb[q] for q in F],
                       [batch_of(q) for q in S], [batch_of(q) for q in F], f"clean:{t}")
        r.update(iid([cb[q] for q in S], [cb[q] for q in F], f"clean:{t}"))
        r["se_inflation"] = round(r["se_cluster"] / r["se_iid"], 3)
        clean[t] = r
    res["clean_contrast"] = clean

    # ---- the retrieval interaction (DiD) -----------------------------------
    did = {}
    for t in ("qwen32b", "qwen72b", "mistral_large", "haiku", "sonnet"):
        cb, hy = load(t, "closed_book", Q), load(t, "hybrid", Q)
        if not cb or not hy:
            continue
        common = sorted(set(cb) & set(hy))
        S = [q for q in common if Q[q]["year"] < 2025]
        F = [q for q in common if Q[q]["year"] >= 2025]
        # per-question retrieval advantage; DiD = Fresh advantage - Static advantage
        r = two_sample([hy[q] - cb[q] for q in F], [hy[q] - cb[q] for q in S],
                       [batch_of(q) for q in F], [batch_of(q) for q in S], f"did:{t}")
        r.update(iid([hy[q] - cb[q] for q in F], [hy[q] - cb[q] for q in S], f"did:{t}"))
        r["se_inflation"] = round(r["se_cluster"] / r["se_iid"], 3)
        did[t] = r
    res["did"] = did

    (RES / "cluster_robust.json").write_text(json.dumps(res, indent=2))

    d = res["design"]
    print(f"DESIGN: {d['n_quiz_batches']} quiz batches, median size "
          f"{d['median_batch_size']}, {d['batches_straddling_boundary']} straddle "
          f"the boundary\n")
    for title, block in (("CLIFF @2025", cliffs), ("CLEAN 2023-vs-2026", clean),
                         ("DiD (retrieval interaction)", did)):
        print(f"=== {title} — cluster-robust ===")
        print(f"{'model':18s}{'diff':>7s}{'SE_iid':>8s}{'SE_clu':>8s}{'infl':>6s}"
              f"{'p_2sided':>10s}{'CI95 (cluster)':>20s}")
        for t, r in block.items():
            ci = f"[{r['ci95_cluster'][0]:.1f},{r['ci95_cluster'][1]:.1f}]"
            sig = "*" if r["p_two_sided"] < 0.05 else " "
            print(f"{NAME.get(t,t):18s}{r['diff']:7.2f}{r['se_iid']:8.2f}"
                  f"{r['se_cluster']:8.2f}{r['se_inflation']:6.2f}"
                  f"{r['p_two_sided']:10.4f}{ci:>20s} {sig}")
        print()


if __name__ == "__main__":
    main()
