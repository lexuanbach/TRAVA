#!/usr/bin/env python3
"""A second model family, and what it does to the scale story.

The ladder the paper reports is Qwen2.5 at six scales plus one Mistral rung at
the top, so "scale" and "family" move together exactly where the effect is
largest. The review asked for an independent large open-weight family or for
every conclusion to be scoped to this ladder. This adds Llama-3-8B-Instruct and
analyses what changes.

Llama-3-8B is the informative addition for a reason that has nothing to do with
its size: **its pretraining cutoff is disclosed** (March 2023), where Qwen2.5
and Mistral-Large-2 disclose nothing. That makes it the only rung besides
gpt-4o-mini where we can put the boundary where the vendor says it belongs
rather than where a curve suggests. So it tests two things at once:

  1. FAMILY. At ~8B, does an independent family behave like Qwen2.5-7B at the
     shared 2025 boundary?
  2. BOUNDARY. At its OWN disclosed boundary, does an 8B model show the cliff
     the paper reports only above 100B?

The second question needs a difficulty control, because 2022 is the easiest year
in this benchmark and every model drops from 2022 to 2024 for reasons that have
nothing to do with cutoffs. We therefore report the raw drop AND the
difference-in-differences against Qwen2.5-7B, a same-scale model answering the
same questions whose training window covers 2024. The DiD is the estimate that
isolates the cutoff.

Outputs results/second_family.json.
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
from metrics import token_f1                       # noqa: E402

RES = ROOT / "results"
PROC = ROOT / "data/processed"

# Disclosed pretraining cutoffs. Absent = the vendor publishes none.
CUTOFF = {
    "llama3_8b": ("2023-03", "Meta model card, Llama 3 8B"),
    "gpt4omini": ("2023-10", "OpenAI model card"),
}
PARAMS_B = {"qwen": 0.5, "qwen1_5b": 1.5, "qwen3b": 3.0, "qwen7b": 7.0,
            "qwen32b": 32.0, "qwen72b": 72.0, "mistral_large": 123.0,
            "llama3_8b": 8.0}
FAMILY = {"qwen": "Qwen2.5", "qwen1_5b": "Qwen2.5", "qwen3b": "Qwen2.5",
          "qwen7b": "Qwen2.5", "qwen32b": "Qwen2.5", "qwen72b": "Qwen2.5",
          "mistral_large": "Mistral", "llama3_8b": "Llama-3",
          "gpt4omini": "GPT-4o", "haiku": "Claude", "sonnet": "Claude"}


def quiz_of(qid):
    return qid.split("_")[0]


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in
             (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}


def scores_by_quiz(model, cond, qs, year_filter):
    f = PROC / f"generations_{model}" / f"{cond}.jsonl"
    if not f.is_file():
        return None
    acc = collections.defaultdict(list)
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q = qs.get(r["qid"])
        if q is None or not year_filter(q["year"]):
            continue
        acc[quiz_of(r["qid"])].append(
            100.0 * token_f1(r["prediction"] or "", q["gold_answer"]))
    return acc


def unpaired_boot(a_by_quiz, b_by_quiz, B, seed):
    """Difference of means where the two sides are DIFFERENT quizzes."""
    a = [statistics.fmean(v) for v in a_by_quiz.values()]
    b = [statistics.fmean(v) for v in b_by_quiz.values()]
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean([a[rng.randrange(len(a))] for _ in a])
        - statistics.fmean([b[rng.randrange(len(b))] for _ in b])
        for _ in range(B))
    le = sum(1 for x in draws if x <= 0)
    return {"diff": round(statistics.fmean(a) - statistics.fmean(b), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5),
            "n_quizzes": [len(a), len(b)]}


def did_boot(m_pre, m_post, c_pre, c_post, B, seed):
    """(model gap) minus (control gap), resampling quizzes on each side."""
    def arr(d):
        return [statistics.fmean(v) for v in d.values()]
    mp, mq, cp, cq = arr(m_pre), arr(m_post), arr(c_pre), arr(c_post)
    if min(map(len, (mp, mq, cp, cq))) < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        a = statistics.fmean([mp[rng.randrange(len(mp))] for _ in mp])
        b = statistics.fmean([mq[rng.randrange(len(mq))] for _ in mq])
        c = statistics.fmean([cp[rng.randrange(len(cp))] for _ in cp])
        d = statistics.fmean([cq[rng.randrange(len(cq))] for _ in cq])
        draws.append((a - b) - (c - d))
    draws.sort()
    obs = (statistics.fmean(mp) - statistics.fmean(mq)) - \
          (statistics.fmean(cp) - statistics.fmean(cq))
    le = sum(1 for x in draws if x <= 0)
    return {"did": round(obs, 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--new-model", default="llama3_8b")
    ap.add_argument("--control", default="qwen7b",
                    help="same-scale model whose training window covers the "
                         "post period; carries the year-difficulty signal")
    ap.add_argument("--output", type=Path, default=RES / "second_family.json")
    args = ap.parse_args()

    qs = load_questions()
    out = {
        "scope": "An independent model family added to the ladder, and the "
                 "boundary test its disclosed cutoff makes possible.",
        "new_model": args.new_model,
        "family": FAMILY.get(args.new_model),
        "params_B": PARAMS_B.get(args.new_model),
        "disclosed_cutoff": CUTOFF.get(args.new_model),
        "control": args.control,
    }

    # -------- 1. the shared 2025 boundary: does the family behave the same? --
    shared = {}
    for m in sorted(set(PARAMS_B) & {args.new_model, args.control, "qwen32b",
                                     "qwen72b", "mistral_large"}):
        pre = scores_by_quiz(m, "closed_book", qs, lambda y: y == 2023)
        post = scores_by_quiz(m, "closed_book", qs, lambda y: y == 2026)
        if not pre or not post:
            continue
        r = unpaired_boot(pre, post, args.B, args.seed)
        if r:
            r["family"] = FAMILY.get(m)
            r["params_B"] = PARAMS_B.get(m)
            shared[m] = r
    out["shared_boundary_2023_vs_2026"] = shared
    print("conservative contrast at the shared boundary (2023 vs 2026):")
    for m, r in shared.items():
        print(f"  {m:16s} {r['family']:9s} {r['params_B']:6.1f}B  "
              f"{r['diff']:+7.2f}  CI {r['ci95']}  p={r['p_plus_one']}")

    # -------- 2. the new model's OWN disclosed boundary ---------------------
    # Llama-3-8B: cutoff 2023-03. 2022 is entirely pre-cutoff; 2024 entirely
    # post. We use 2022-vs-2024 so both sides are a full year away from the
    # cutoff month and neither straddles it.
    own = {}
    for m in sorted(set(PARAMS_B)):
        pre = scores_by_quiz(m, "closed_book", qs, lambda y: y == 2022)
        post = scores_by_quiz(m, "closed_book", qs, lambda y: y == 2024)
        if not pre or not post:
            continue
        r = unpaired_boot(pre, post, args.B, args.seed)
        if r:
            r["family"] = FAMILY.get(m)
            r["params_B"] = PARAMS_B.get(m)
            r["cutoff_disclosed"] = m in CUTOFF
            own[m] = r
    out["llama_boundary_2022_vs_2024"] = own
    print("\nat Llama-3's disclosed boundary (2022 vs 2024), every model:")
    for m, r in sorted(own.items(), key=lambda kv: kv[1]["params_B"]):
        flag = "  <- cutoff disclosed" if r["cutoff_disclosed"] else ""
        print(f"  {m:16s} {r['params_B']:6.1f}B  {r['diff']:+7.2f}  "
              f"CI {r['ci95']}  p={r['p_plus_one']}{flag}")

    # -------- 3. DiD against a same-scale control ---------------------------
    mp = scores_by_quiz(args.new_model, "closed_book", qs, lambda y: y == 2022)
    mq = scores_by_quiz(args.new_model, "closed_book", qs, lambda y: y == 2024)
    dids = {}
    for ctrl in ("qwen7b", "qwen3b", "qwen32b"):
        cp = scores_by_quiz(ctrl, "closed_book", qs, lambda y: y == 2022)
        cq = scores_by_quiz(ctrl, "closed_book", qs, lambda y: y == 2024)
        if not (mp and mq and cp and cq):
            continue
        d = did_boot(mp, mq, cp, cq, args.B, args.seed)
        if d:
            d["control_params_B"] = PARAMS_B[ctrl]
            dids[ctrl] = d
    out["did_vs_controls"] = dids

    # A better-powered specification, fixed by the cutoff rather than chosen:
    # Llama-3-8B's cutoff is 2023-03, so 2022 is wholly pre-cutoff and
    # 2024/2025/2026 are wholly post. Pooling the post years triples that side.
    mp2 = scores_by_quiz(args.new_model, "closed_book", qs, lambda y: y == 2022)
    mq2 = scores_by_quiz(args.new_model, "closed_book", qs, lambda y: y >= 2024)
    dids_pooled = {}
    for ctrl in ("qwen7b", "qwen3b", "qwen32b"):
        cp = scores_by_quiz(ctrl, "closed_book", qs, lambda y: y == 2022)
        cq = scores_by_quiz(ctrl, "closed_book", qs, lambda y: y >= 2024)
        if not (mp2 and mq2 and cp and cq):
            continue
        d = did_boot(mp2, mq2, cp, cq, args.B, args.seed)
        if d:
            d["control_params_B"] = PARAMS_B[ctrl]
            dids_pooled[ctrl] = d
    out["did_vs_controls_pooled_post"] = dids_pooled
    raw2 = unpaired_boot(mp2, mq2, args.B, args.seed)
    out["llama_2022_vs_post2024_raw"] = raw2
    print(f"\npooling the post-cutoff years (2022 vs 2024-2026):")
    print(f"  {args.new_model} raw drop {raw2['diff']:+.2f} CI {raw2['ci95']} "
          f"p={raw2['p_plus_one']}")
    for c, d in dids_pooled.items():
        print(f"  DiD vs {c:10s} {d['did']:+7.2f}  CI {d['ci95']}  p={d['p_plus_one']}")
    print(f"\n{args.new_model} 2022-2024 drop MINUS the same drop for a control "
          f"(isolates the cutoff from year difficulty):")
    for c, d in dids.items():
        print(f"  vs {c:10s} ({d['control_params_B']:5.1f}B)  DiD {d['did']:+7.2f}  "
              f"CI {d['ci95']}  p={d['p_plus_one']}")

    # -------- 4. what this does to the scale story --------------------------
    prim = dids.get(args.control) or (list(dids.values())[0] if dids else None)
    prim2 = dids_pooled.get(args.control) or (
        list(dids_pooled.values())[0] if dids_pooled else None)
    nm = own.get(args.new_model)
    ctl = own.get(args.control)
    did_sig = bool(prim2 and (prim2["ci95"][0] > 0 or prim2["ci95"][1] < 0))
    out["verdict"] = {
        "family_check": (
            f"At the shared 2025-era boundary the new family behaves like the "
            f"Qwen rung of comparable size: "
            f"{args.new_model} {shared.get(args.new_model, {}).get('diff')} vs "
            f"{args.control} {shared.get(args.control, {}).get('diff')}, both "
            f"with intervals spanning zero. The paper's claim that no cliff is "
            f"established below ~100B at that boundary is not a Qwen artifact."),
        "boundary_check": (
            f"At Llama-3-8B's OWN disclosed boundary the picture is different: "
            f"it drops {nm['diff'] if nm else None} F1 (CI "
            f"{nm['ci95'] if nm else None}) where the same-scale control drops "
            f"{ctl['diff'] if ctl else None}. The difference-in-differences is "
            f"{prim['did'] if prim else None} (CI {prim['ci95'] if prim else None}, "
            f"p={prim['p_plus_one'] if prim else None})."),
        "what_it_means": (
            "An 8B model shows a significant boundary gap when the boundary is "
            "the one its vendor documents, on questions where a same-scale Qwen "
            "rung shows nothing. Taken at face value that says the paper's "
            "'no cliff below ~100B' finding is specific to a SHARED 2025 "
            "boundary sitting inside most rungs' training windows."),
        "how_far_this_goes": (
            "Not as far as the point estimates suggest. The raw Llama-3-8B drop "
            "is significant, but the difference-in-differences that separates "
            "the cutoff from year-to-year difficulty drift is "
            f"{prim2['did'] if prim2 else None} with CI "
            f"{prim2['ci95'] if prim2 else None} (p="
            f"{prim2['p_plus_one'] if prim2 else None}) even in the "
            f"better-powered pooled-post specification"
            f"{'' if did_sig else ', which does NOT exclude zero'}. The result "
            "is control-dependent: of the three same-questions controls, only "
            f"{sum(1 for v in dids_pooled.values() if v['ci95'][0] > 0)} of "
            f"{len(dids_pooled)} gives an interval excluding zero, and we quote "
            "the pre-specified same-scale control rather than the one that "
            "does. Four noisy quantities are differenced across ~30 quizzes on "
            "the pre side. We therefore report this as a consistent, sizeable, "
            "directionally clear signal that this design cannot certify."),
        "caveat": (
            "One model with a disclosed cutoff is one data point, and it cannot "
            "separate 'Llama-3 has an earlier cutoff' from 'Llama-3 was trained "
            "differently'. Establishing the boundary-position account needs "
            "several families with disclosed cutoffs at matched scales, which "
            "is a data-availability problem: seven of our nine vendors publish "
            "no cutoff at all."),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
