#!/usr/bin/env python3
"""Test a DISCLOSED cutoff at quarterly resolution.

Calendar-year bins cannot test a month-level cutoff: they have ~12-month
resolution, so an argmax over year-over-year drops selecting "2024" is equally
consistent with any cutoff from mid-2023 to mid-2024. Our questions are dated to
the day, so we can bin by quarter instead and ask the sharper question directly.

gpt-4o-mini is the only generator here with a day-precise disclosed cutoff
(OpenAI documents "Oct 01, 2023 knowledge cutoff"), so it is the one model where
the protocol can be checked against ground truth rather than assumed. If the
protocol works, its closed-book accuracy should step down between 2023-Q3 and
2024-Q1 and NOT elsewhere.

Controls: two Qwen2.5 models with no relevant disclosed cutoff give the background
year/quarter difficulty trend. Note they carry their own (undisclosed) cutoffs, so
they are contaminated controls -- which biases the difference-in-differences
DOWNWARD, i.e. conservative.

Writes results/quarterly_cutoff.json.
"""
import json
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from metrics import token_f1

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
GEN = {"gpt4omini": "generations_gpt4omini", "qwen32b": "generations_qwen32b",
       "qwen72b": "generations_qwen72b"}
DISCLOSED = {"gpt4omini": "2023-10-01"}
B = 4000


def rng(label):
    return np.random.default_rng(zlib.crc32(label.encode()) & 0xFFFFFFFF)


def quarter(d):
    dt = datetime.strptime(d, "%Y-%m-%d")
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"


def block_boot(a, ab, b, bb, label):
    r = rng(label)
    ga, gb = defaultdict(list), defaultdict(list)
    for v, k in zip(a, ab):
        ga[k].append(v)
    for v, k in zip(b, bb):
        gb[k].append(v)
    ka, kb = list(ga), list(gb)
    obs = np.mean(a) - np.mean(b)
    d = np.array([np.mean([x for i in r.choice(len(ka), len(ka)) for x in ga[ka[i]]]) -
                  np.mean([x for i in r.choice(len(kb), len(kb)) for x in gb[kb[i]]])
                  for _ in range(B)])
    return {"diff": round(float(obs), 2),
            "ci95": [round(float(np.percentile(d, 2.5)), 2),
                     round(float(np.percentile(d, 97.5)), 2)],
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4),
            "n": [len(a), len(b)]}


def main():
    Q = {q["qid"]: q for q in
         (json.loads(l) for l in (PROC / "questions_full.jsonl").read_text().splitlines())}
    res = {"note": "gpt-4o-mini is the only generator with a day-precise disclosed "
                   "cutoff (OpenAI: 'Oct 01, 2023 knowledge cutoff'). Quarterly bins "
                   "test it directly; year bins cannot (~12-month resolution)."}
    sc = {}
    for tag, d in GEN.items():
        f = PROC / d / "closed_book_full.jsonl"
        if not f.exists():
            continue
        sc[tag] = {r["qid"]: token_f1(r["prediction"], Q[r["qid"]]["gold_answer"]) * 100
                   for r in (json.loads(l) for l in f.read_text().splitlines())
                   if r["qid"] in Q}

    # ---- per-quarter means -------------------------------------------------
    byq = {}
    for tag, s in sc.items():
        g = defaultdict(list)
        for qid, v in s.items():
            g[quarter(Q[qid]["date"])].append(v)
        byq[tag] = {k: (round(float(np.mean(v)), 2), len(v)) for k, v in sorted(g.items())}
    res["by_quarter"] = {t: {k: v[0] for k, v in b.items()} for t, b in byq.items()}
    res["n_per_quarter"] = {t: {k: v[1] for k, v in b.items()} for t, b in byq.items()}

    # ---- quarter-over-quarter drops, and where the argmax lands -------------
    steps = {}
    for tag, b in byq.items():
        ks = list(b)
        dd = {ks[i + 1]: round(b[ks[i]][0] - b[ks[i + 1]][0], 2) for i in range(len(ks) - 1)}
        steps[tag] = {"drops": dd, "argmax_drop_at": max(dd, key=dd.get),
                      "max_drop": max(dd.values())}
    res["quarter_steps"] = steps

    # ---- the direct test: pre-cutoff vs post-cutoff quarters ----------------
    # Convention stated in advance: an Oct-1-2023 cutoff makes 2023-Q4 the first
    # partially-affected quarter and 2024-Q1 the first fully-post-cutoff quarter.
    if "gpt4omini" in sc:
        s = sc["gpt4omini"]
        pre = [q for q in s if quarter(Q[q]["date"]) <= "2023-Q3"]
        post = [q for q in s if "2024-Q1" <= quarter(Q[q]["date"]) <= "2024-Q4"]
        res["direct_test_pre_vs_post"] = block_boot(
            [s[q] for q in pre], [q.split("_")[0] for q in pre],
            [s[q] for q in post], [q.split("_")[0] for q in post], "gpt:prepost")
        res["direct_test_pre_vs_post"]["definition"] = (
            "pre = quarters <= 2023-Q3 (fully pre-cutoff); post = 2024 quarters "
            "(fully post-cutoff, and before the 2025/2026 bins where every model "
            "including no-knowledge controls rebounds)")
        # placebo: the same-width contrast one year EARLIER, where no cutoff sits
        pre_p = [q for q in s if quarter(Q[q]["date"]) <= "2022-Q3"]
        post_p = [q for q in s if "2023-Q1" <= quarter(Q[q]["date"]) <= "2023-Q3"]
        if len(pre_p) > 30:
            res["placebo_one_year_earlier"] = block_boot(
                [s[q] for q in pre_p], [q.split("_")[0] for q in pre_p],
                [s[q] for q in post_p], [q.split("_")[0] for q in post_p], "gpt:placebo")

    (RES / "quarterly_cutoff.json").write_text(json.dumps(res, indent=2))

    print("closed-book F1 by quarter (n in parens):\n")
    qs = sorted({k for b in byq.values() for k in b})
    print(f"{'quarter':10s}" + "".join(f"{t:>14s}" for t in byq))
    for k in qs:
        row = f"{k:10s}"
        for t in byq:
            v = byq[t].get(k)
            row += f"{(f'{v[0]:.1f} ({v[1]})' if v else '-'):>14s}"
        mark = "  <== disclosed cutoff 2023-10-01" if k == "2023-Q4" else ""
        print(row + mark)
    print("\nargmax quarter-over-quarter drop:")
    for t, v in steps.items():
        print(f"  {t:10s} at {v['argmax_drop_at']}  (drop {v['max_drop']})")
    if "direct_test_pre_vs_post" in res:
        d = res["direct_test_pre_vs_post"]
        print(f"\ngpt-4o-mini  <=2023-Q3 vs 2024: {d['diff']:+.2f}  CI {d['ci95']}  "
              f"p={d['p_two_sided']}  n={d['n'][0]}/{d['n'][1]}")
        if "placebo_one_year_earlier" in res:
            p = res["placebo_one_year_earlier"]
            print(f"placebo (one year earlier, no cutoff): {p['diff']:+.2f}  "
                  f"CI {p['ci95']}  p={p['p_two_sided']}")


if __name__ == "__main__":
    main()
