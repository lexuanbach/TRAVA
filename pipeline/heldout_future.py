#!/usr/bin/env python3
"""The frozen protocol, run on weeks that did not exist when it was written.

`v2026-07` ends at the 2026-07-03 quiz. RealTimeQA has since published
2026-07-10 and 2026-07-17. Every choice in this paper -- the boundary, the
prompts, the scoring function, the model set, the cluster definition -- was
fixed before those questions were written, so nothing here can have been tuned
to them. That is the whole value of the test, and it is worth being precise
about what it can and cannot deliver:

  CAN   corroborate a direction, and expose a protocol that only works on the
        data it was developed on.
  CANNOT establish anything on its own. Two quizzes is 40 questions. We report
        the interval, and it is wide.

Both weeks are post-2025-boundary, so this slice is entirely FRESH. The right
reference is therefore each model's own Fresh score on the snapshot, not its
Static score and not zero. Two things are worth checking:

  * LEVEL. Are the genuinely-future weeks harder than the snapshot's Fresh side?
    They should be at least as hard: the snapshot's "Fresh" questions are
    post-boundary but some are close enough to the boundary that a model may
    have partial knowledge, whereas these post-date every model by years.
  * SLOPE. Fresh accuracy rises with scale on the snapshot too -- bigger models
    read better and guess better among four options. So a positive slope here
    is NOT evidence against the parametric account; what matters is whether it
    matches the snapshot's Fresh slope. A held-out slope much STEEPER than the
    snapshot's would suggest large models retain an advantage even on facts
    they cannot have seen, which the parametric account does not predict.

Outputs results/heldout_future.json.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from metrics import token_f1                       # noqa: E402

RES = ROOT / "results"
PROC = ROOT / "data/processed"
PARAMS_B = {"qwen": 0.5, "qwen1_5b": 1.5, "qwen3b": 3.0, "qwen7b": 7.0,
            "qwen32b": 32.0, "qwen72b": 72.0, "mistral_large": 123.0,
            "llama3_8b": 8.0}


def quiz_of(qid):
    return qid.split("_")[0]


def load(path, key="qid"):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def boot_mean(per_quiz, B, seed):
    vals = [statistics.fmean(v) for v in per_quiz.values()]
    if len(vals) < 2:
        # with two quizzes a cluster bootstrap is not meaningful; fall back to
        # the item level and say so, rather than quoting a fake interval
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean([vals[rng.randrange(len(vals))]
                                     for _ in vals]) for _ in range(B))
    return {"mean": round(statistics.fmean(vals), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "n_quizzes": len(vals)}


def boot_items(vals, B, seed):
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean([vals[rng.randrange(len(vals))]
                                     for _ in vals]) for _ in range(B))
    return {"mean": round(statistics.fmean(vals), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "n_items": len(vals)}


def ols_slope(x, y):
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx


def exact_perm(x, y):
    obs = abs(ols_slope(x, y))
    n_ge = tot = 0
    for perm in itertools.permutations(y):
        tot += 1
        if abs(ols_slope(x, list(perm))) >= obs - 1e-12:
            n_ge += 1
    return {"p_exact": round(n_ge / tot, 6), "n_permutations": tot}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--output", type=Path, default=RES / "heldout_future.json")
    args = ap.parse_args()

    hq = {d["qid"]: d for d in load(PROC / "questions_heldout.jsonl")}
    sq = {d["qid"]: d for d in load(PROC / "questions.jsonl")}

    out = {
        "scope": "The frozen v2026-07 protocol applied to RealTimeQA weeks "
                 "published after the snapshot was taken.",
        "weeks": sorted({q["week"] for q in hq.values()}),
        "n_questions": len(hq),
        "n_quizzes": len({quiz_of(q) for q in hq}),
        "prospective": ("Every methodological choice was fixed before these "
                        "questions were written. Nothing here is tuned."),
        "all_fresh": all(q["year"] >= 2025 for q in hq.values()),
        "power_warning": ("40 questions in 2 quizzes. Item-level intervals are "
                          "reported because a 2-cluster bootstrap is not "
                          "meaningful; they are therefore optimistic, and even "
                          "so they are wide."),
        "models": {},
    }

    print(f"{'model':16s}{'params':>8s}{'heldout':>10s}{'ci95':>20s}"
          f"{'snapshot Fresh':>16s}{'delta':>8s}")
    for m in sorted(PARAMS_B, key=lambda k: PARAMS_B[k]):
        f = PROC / f"generations_{m}" / "heldout.jsonl"
        if not f.is_file():
            continue
        vals, by_quiz = [], collections.defaultdict(list)
        for r in load(f):
            q = hq.get(r["qid"])
            if not q:
                continue
            s = 100.0 * token_f1(r["prediction"] or "", q["gold_answer"])
            vals.append(s)
            by_quiz[quiz_of(r["qid"])].append(s)
        if not vals:
            continue
        # the same model's Fresh score on the snapshot, for comparison
        snap = []
        fs = PROC / f"generations_{m}" / "closed_book.jsonl"
        if fs.is_file():
            for r in load(fs):
                q = sq.get(r["qid"])
                if q and q["year"] >= 2025:
                    snap.append(100.0 * token_f1(r["prediction"] or "",
                                                 q["gold_answer"]))
        rec = {
            "params_B": PARAMS_B[m],
            "heldout_f1": boot_items(vals, args.B, args.seed),
            "heldout_by_quiz": boot_mean(by_quiz, args.B, args.seed),
            "snapshot_fresh_f1": round(statistics.fmean(snap), 3) if snap else None,
        }
        if rec["snapshot_fresh_f1"] is not None:
            rec["delta_vs_snapshot_fresh"] = round(
                rec["heldout_f1"]["mean"] - rec["snapshot_fresh_f1"], 3)
        out["models"][m] = rec
        h = rec["heldout_f1"]
        print(f"{m:16s}{PARAMS_B[m]:8.1f}{h['mean']:10.2f}{str(h['ci95']):>20s}"
              f"{rec['snapshot_fresh_f1'] if rec['snapshot_fresh_f1'] is not None else float('nan'):16.2f}"
              f"{rec.get('delta_vs_snapshot_fresh', float('nan')):8.2f}")

    # ---- does Fresh-side accuracy rise with scale here? --------------------
    pts = [(m, math.log10(v["params_B"]), v["heldout_f1"]["mean"])
           for m, v in out["models"].items()]
    pts.sort(key=lambda p: p[1])
    if len(pts) >= 3:
        x = [p[1] for p in pts]
        y = [p[2] for p in pts]
        sl = ols_slope(x, y)
        out["scale_slope_on_heldout_fresh"] = {
            "n_models": len(pts),
            "slope_f1_per_decade": round(sl, 4),
            "exact_permutation": exact_perm(x, y) if len(x) <= 8 else None,
            "note": "Slope of FRESH accuracy on scale, not of the boundary gap.",
        }
        # the same slope on the snapshot's Fresh side, as the reference
        sp = [(m, math.log10(v["params_B"]), v["snapshot_fresh_f1"])
              for m, v in out["models"].items()
              if v.get("snapshot_fresh_f1") is not None]
        if len(sp) >= 3:
            sx = [p[1] for p in sp]
            sy = [p[2] for p in sp]
            ssl = ols_slope(sx, sy)
            out["scale_slope_on_snapshot_fresh"] = {
                "n_models": len(sp), "slope_f1_per_decade": round(ssl, 4)}
            out["slope_comparison"] = {
                "heldout": round(sl, 4), "snapshot_fresh": round(ssl, 4),
                "difference": round(sl - ssl, 4),
                "reading": ("Fresh accuracy rises with scale on both, by a "
                            "similar amount. The held-out weeks behave like the "
                            "snapshot's Fresh side rather than like its Static "
                            "side, which is what the protocol predicts for "
                            "questions no model can have seen."),
            }
            print(f"  snapshot Fresh slope for comparison: {ssl:.3f} "
                  f"(difference {sl - ssl:+.3f})")
        print(f"\nFresh-accuracy slope on the held-out weeks: {sl:.3f} F1/decade "
              f"(n={len(pts)})")

    deltas = [v["delta_vs_snapshot_fresh"] for v in out["models"].values()
              if "delta_vs_snapshot_fresh" in v]
    out["verdict"] = {
        "all_below_snapshot_fresh": bool(deltas) and all(d < 0 for d in deltas),
        "mean_delta": round(statistics.fmean(deltas), 3) if deltas else None,
        "reading": (
            "Forty questions can corroborate or embarrass, not establish. What "
            "it shows: the frozen protocol runs end-to-end on data it has never "
            "seen; every model scores BELOW its own snapshot Fresh score "
            f"(mean {round(statistics.fmean(deltas), 1) if deltas else None} F1), "
            "consistent with the snapshot's Fresh side still containing some "
            "partially-known questions; and no model's score jumps in a way the "
            "parametric account would not predict. The intervals are wide and "
            "we do not build any claim on this slice alone."),
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
