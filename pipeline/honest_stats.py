#!/usr/bin/env python3
"""Inferential analyses behind the paper, all computed from cached generations.

Every one of these is a number the review round showed was either (a) missing,
(b) reported for the models that pass and omitted for the model that fails, or
(c) contradicted by the artifact.  Writes results/honest_stats.json.

  clean_contrast   2023 (pre-cutoff for every model) vs 2026 (post-cutoff for
                   every model) -- avoids the 2022-easy / 2025-hard bins that
                   drive the boundary-split cliff.
  year_difficulty  per-year difficulty index from conditions that cannot
                   memorise (small models; hybrid_loo with hit@5=0).
  multiplicity     Benjamini-Hochberg + Bonferroni over the cliff family and
                   the DiD family.
  capability_trend regression of cliff on log10(params) within the open ladder,
                   plus the relative cliff -- the paper claims a gradient and
                   never tests it.
  boundary_dependence  DiD at every candidate boundary, on the primary.
"""
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import token_f1

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def rng_for(label):
    """Stable per-analysis RNG.

    A single module-level RNG makes every p-value depend on how many draws were
    taken before it, so adding or reordering an analysis silently changes numbers
    already quoted in the manuscript. Seeding per analysis label makes each result
    reproducible on its own.
    """
    return np.random.default_rng(zlib.crc32(label.encode()) & 0xFFFFFFFF)

TAGS = {
    "qwen": ("Qwen2.5-0.5B", 0.5), "qwen1_5b": ("Qwen2.5-1.5B", 1.5),
    "qwen3b": ("Qwen2.5-3B", 3.0), "qwen7b": ("Qwen2.5-7B", 7.0),
    "qwen32b": ("Qwen2.5-32B", 32.0), "qwen72b": ("Qwen2.5-72B", 72.0),
    "mistral_large": ("Mistral-Large-2", 123.0),
    "haiku": ("Claude Haiku 4.5", None), "sonnet": ("Claude Sonnet 4.5", None),
}
GEN_DIR = {t: (PROC / ("generations" if t == "haiku" else f"generations_{t}")) for t in TAGS}


def questions():
    return {q["qid"]: q for q in
            (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines())}


def load_cond(tag, cond, Q):
    f = GEN_DIR[tag] / f"{cond}.jsonl"
    if not f.exists():
        return None
    out = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        q = Q.get(r["qid"])
        if q:
            out[r["qid"]] = (token_f1(r["prediction"], q["gold_answer"]) * 100, q["year"])
    return out


def boot_unpaired(a, b, n=2000, label="boot"):
    """One-sided p for mean(a) > mean(b), plus CI of the difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    rng = rng_for(label)
    d = a.mean() - b.mean()
    ds = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
                   for _ in range(n)])
    return {"diff": round(float(d), 2),
            "ci95": [round(float(np.percentile(ds, 2.5)), 2),
                     round(float(np.percentile(ds, 97.5)), 2)],
            "one_sided_p": round(float((ds <= 0).mean()), 4),
            "n_a": len(a), "n_b": len(b)}


def bh(pvals):
    """Benjamini-Hochberg at 0.05 -> list of (p, threshold, reject)."""
    idx = np.argsort(pvals)
    m = len(pvals)
    out = [None] * m
    prev = False
    for rank, i in enumerate(reversed(idx), start=1):
        k = m - rank + 1
        thr = 0.05 * k / m
        rej = (pvals[i] <= thr) or prev
        prev = rej
        out[i] = {"p": pvals[i], "bh_threshold": round(thr, 4), "reject": bool(rej)}
    return out


def main():
    Q = questions()
    res = {}

    # ---- 1. clean contrast: 2023 vs 2026 -----------------------------------
    clean = {}
    for tag, (name, par) in TAGS.items():
        cb = load_cond(tag, "closed_book", Q)
        if not cb:
            continue
        a = [f for f, y in cb.values() if y == 2023]
        b = [f for f, y in cb.values() if y == 2026]
        t = boot_unpaired(a, b, label=f"clean:{tag}")
        by = defaultdict(list)
        for f, y in cb.values():
            by[y].append(f)
        clean[tag] = {
            "model": name, "params_B": par,
            "by_year": {str(y): round(float(np.mean(v)), 2) for y, v in sorted(by.items())},
            "f1_2023": round(float(np.mean(a)), 2), "f1_2026": round(float(np.mean(b)), 2),
            "clean_cliff_2023_minus_2026": t["diff"], "ci95": t["ci95"],
            "one_sided_p": t["one_sided_p"],
        }
    res["clean_contrast"] = clean

    # ---- 2. year difficulty from non-memorising conditions ------------------
    diff_idx = {}
    loo = load_cond("qwen32b", "hybrid_loo", Q)
    if loo:
        by = defaultdict(list)
        for f, y in loo.values():
            by[y].append(f)
        diff_idx["qwen32b_hybrid_loo"] = {str(y): round(float(np.mean(v)), 2)
                                          for y, v in sorted(by.items())}
    for tag in ("qwen", "qwen1_5b", "qwen3b"):
        cb = load_cond(tag, "closed_book", Q)
        if cb:
            by = defaultdict(list)
            for f, y in cb.values():
                by[y].append(f)
            diff_idx[f"{tag}_closed_book"] = {str(y): round(float(np.mean(v)), 2)
                                              for y, v in sorted(by.items())}
    res["year_difficulty"] = {
        "index": diff_idx,
        "note": "These conditions cannot answer from memory of the event (sub-3B "
                "models have no such parametric knowledge; hybrid_loo has hit@5=0 "
                "by construction). Their year profile is a difficulty signal, and "
                "it is U-shaped: 2022 and 2026 easy, 2025 hard.",
    }

    # ---- 3. multiplicity over the cliff family ------------------------------
    cm = json.loads((RES / "cross_model.json").read_text())
    fam = [(t, cm[t]["cliff"]["one_sided_p"]) for t in cm
           if isinstance(cm[t], dict) and cm[t].get("cliff")]
    ps = np.array([p for _, p in fam])
    bh_res = bh(ps)
    res["multiplicity_cliff"] = {
        "bonferroni_alpha": round(0.05 / len(fam), 5),
        "tests": {t: {"p": p, "bonferroni_reject": bool(p <= 0.05 / len(fam)),
                      **bh_res[i]} for i, (t, p) in enumerate(fam)},
    }

    rb = json.loads((RES / "robustness.json").read_text())
    did = rb.get("did", {}).get("conditions", {})
    dfam = [(k, v["p"]) for k, v in did.items() if isinstance(v, dict) and "p" in v]
    if dfam:
        dps = np.array([p for _, p in dfam])
        dbh = bh(dps)
        res["multiplicity_did"] = {
            "bonferroni_alpha": round(0.05 / len(dfam), 5),
            "n_tests": len(dfam),
            "tests": {k: {"p": p, "bonferroni_reject": bool(p <= 0.05 / len(dfam)),
                          **dbh[i]} for i, (k, p) in enumerate(dfam)},
            "n_bh_discoveries": int(sum(d["reject"] for d in dbh)),
        }

    # ---- 4. capability trend (the claimed contribution, never tested) -------
    pts = [(math.log10(cm[t]["params_B"]), cm[t]["cliff"]["gap"])
           for t in cm if isinstance(cm[t], dict) and cm[t].get("params_B")
           and cm[t].get("cliff")]
    if len(pts) >= 3:
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        slope, icpt = np.polyfit(x, y, 1)
        pred = slope * x + icpt
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # permutation test on the slope (n is small; no normality assumption)
        prng = rng_for("capability_trend")
        perm = np.array([abs(np.polyfit(x, prng.permutation(y), 1)[0]) for _ in range(10000)])
        res["capability_trend"] = {
            "n_models": len(pts),
            "slope_f1_per_decade_params": round(float(slope), 2),
            "r2": round(float(r2), 3),
            "permutation_p": round(float((perm >= abs(slope)).mean()), 4),
            "note": "Open-weight ladder only (params known). Tests the gradient the "
                    "paper claims but never tested.",
        }
    rel = {}
    for t in cm:
        if isinstance(cm[t], dict) and cm[t].get("cliff"):
            c = cm[t]["cliff"]
            if c["static_f1"]:
                rel[t] = round(c["gap"] / c["static_f1"] * 100, 1)
    res["relative_cliff_pct"] = rel

    # ---- 5. boundary dependence of the interaction, on the primary ---------
    cb = load_cond("qwen32b", "closed_book", Q)
    hy = load_cond("qwen32b", "hybrid", Q)
    bd = {}
    if cb and hy:
        for b in (2023, 2024, 2025, 2026):
            cs = [f for f, y in cb.values() if y < b]; cf = [f for f, y in cb.values() if y >= b]
            hs = [f for f, y in hy.values() if y < b]; hf = [f for f, y in hy.values() if y >= b]
            if min(len(cs), len(cf)) < 20:
                continue
            cg = float(np.mean(cs) - np.mean(cf))
            hg = float(np.mean(hs) - np.mean(hf))
            bd[str(b)] = {"closed_book_gap": round(cg, 2), "hybrid_gap": round(hg, 2),
                          "did": round(cg - hg, 2), "n_static": len(cs), "n_fresh": len(cf)}
    res["boundary_dependence_primary"] = {
        "by_boundary": bd,
        "note": "The slice x condition interaction on the primary exists only at the "
                "2025 boundary; it is ~0 at 2023 and 2024.",
    }

    cb_prim = load_cond("qwen32b", "closed_book", Q)

    # ---- 6. the interaction on a model that HAS a collapse ----------------
    # The primary (32B) has no clean cliff, so "retrieval absorbs the collapse"
    # cannot be tested there. Mistral-Large-2 is open-weight, non-Claude, abstains
    # ~0%, and has a robust cliff -- the right model for this interaction.
    inter = {}
    for tag in ("mistral_large", "qwen72b", "haiku", "sonnet", "qwen32b"):
        cb = load_cond(tag, "closed_book", Q)
        hy = load_cond(tag, "hybrid", Q)
        if not cb or not hy:
            continue
        common = sorted(set(cb) & set(hy))
        for label, (sy, fy) in {"boundary_2025": (None, None),
                                "clean_2023_vs_2026": (2023, 2026)}.items():
            if sy is None:
                s = [q for q in common if Q[q]["year"] < 2025]
                f = [q for q in common if Q[q]["year"] >= 2025]
            else:
                s = [q for q in common if Q[q]["year"] == sy]
                f = [q for q in common if Q[q]["year"] == fy]
            if min(len(s), len(f)) < 20:
                continue
            # per-question retrieval advantage, then unpaired bootstrap across slices
            ds = np.array([hy[q][0] - cb[q][0] for q in s])
            df = np.array([hy[q][0] - cb[q][0] for q in f])
            t = boot_unpaired(df, ds, label=f"inter:{tag}:{label}")  # DiD>0: retrieval helps MORE on Fresh
            inter.setdefault(tag, {})[label] = {
                "cb_gap": round(float(np.mean([cb[q][0] for q in s]) -
                                      np.mean([cb[q][0] for q in f])), 2),
                "hybrid_gap": round(float(np.mean([hy[q][0] for q in s]) -
                                          np.mean([hy[q][0] for q in f])), 2),
                "did_f1": t["diff"], "ci95": t["ci95"], "one_sided_p": t["one_sided_p"],
                "n_static": len(s), "n_fresh": len(f),
            }
    res["interaction_by_model"] = inter

    # ---- 7. cutoff controls for Mistral (paper ran these only on 32B/frontier)
    ctrl = {}
    for tag in ("mistral_large", "qwen72b"):
        cb = load_cond(tag, "closed_book", Q)
        if not cb:
            continue
        # permutation null: is the observed max year-over-year drop beyond chance?
        yrs = sorted({y for _, y in cb.values()})
        vals = list(cb.values())

        def maxdrop(assign):
            m = {}
            for (f, _), y in zip(vals, assign):
                m.setdefault(y, []).append(f)
            means = [np.mean(m[y]) for y in yrs if y in m]
            return max((means[i] - means[i + 1] for i in range(len(means) - 1)), default=0.0)

        obs = maxdrop([y for _, y in vals])
        nrng = rng_for(f"permnull:{tag}")
        null = np.array([maxdrop(nrng.permutation([y for _, y in vals])) for _ in range(2000)])
        # cross-fit: estimate the knee on odd ISO weeks, measure the gap on even
        def wk(qid):
            try:
                return int(qid.split("_")[0][-2:])
            except Exception:
                return 0
        odd = {q: v for q, v in cb.items() if wk(q) % 2 == 1}
        even = {q: v for q, v in cb.items() if wk(q) % 2 == 0}
        knee = None
        if odd:
            m = defaultdict(list)
            for f, y in odd.values():
                m[y].append(f)
            means = {y: float(np.mean(v)) for y, v in m.items()}
            drops = {y: means[y - 1] - means[y] for y in sorted(means) if y - 1 in means}
            if drops:
                best = max(drops, key=drops.get)
                knee = int(best) if drops[best] >= 5.0 else None
        cf = None
        if knee and even:
            a = [f for f, y in even.values() if y < knee]
            b = [f for f, y in even.values() if y >= knee]
            if min(len(a), len(b)) >= 20:
                cf = boot_unpaired(a, b, label=f"crossfit:{tag}")
        ctrl[tag] = {
            "permutation_null": {"observed_max_drop": round(float(obs), 2),
                                 "p": round(float((null >= obs).mean()), 4)},
            "crossfit": {"cutoff_estimated_on_odd_weeks": knee,
                         "gap_measured_on_even_weeks": cf},
        }
    res["cutoff_controls_extra"] = ctrl

    # ---- 7b. LOO / stale vs closed-book, paired, per slice ------------------
    # The manuscript previously asserted p<0.001 for BOTH slices; only Static is.
    ctrl_vs_cb = {}
    for cond in ("hybrid_loo", "hybrid_stale"):
        r = load_cond("qwen32b", cond, Q)
        if not r:
            continue
        row = {}
        for sl in ("static", "fresh"):
            qs = [q for q in r if q in cb_prim
                  and ((Q[q]["year"] < 2025) == (sl == "static"))]
            d = np.array([r[q][0] - cb_prim[q][0] for q in qs])
            rng = rng_for(f"vs_cb:{cond}:{sl}")
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(2000)])
            row[sl] = {"cond_f1": round(float(np.mean([r[q][0] for q in qs])), 2),
                       "closed_book_f1": round(float(np.mean([cb_prim[q][0] for q in qs])), 2),
                       "diff": round(float(d.mean()), 2),
                       "ci95": [round(float(np.percentile(bs, 2.5)), 2),
                                round(float(np.percentile(bs, 97.5)), 2)],
                       "two_sided_p": round(float(2 * min((bs <= 0).mean(),
                                                          (bs >= 0).mean())), 4),
                       "n": len(qs)}
        ctrl_vs_cb[cond] = row
    res["controls_vs_closedbook"] = ctrl_vs_cb

    # ---- 8. multiplicity for the Clean and DiD columns of Table 1 ----------
    # Each column of the cross-model table is its own family of tests; correcting
    # only the boundary-split column would be arbitrary.
    cl = [(t, v["one_sided_p"]) for t, v in res["clean_contrast"].items()]
    cbh = bh(np.array([p for _, p in cl]))
    res["multiplicity_clean"] = {
        "bonferroni_alpha": round(0.05 / len(cl), 5),
        "tests": {t: {"p": p, "bonferroni_reject": bool(p <= 0.05 / len(cl)), **cbh[i]}
                  for i, (t, p) in enumerate(cl)}}
    di = [(t, v["boundary_2025"]["one_sided_p"]) for t, v in res["interaction_by_model"].items()
          if v.get("boundary_2025")]
    dbh2 = bh(np.array([p for _, p in di]))
    res["multiplicity_interaction"] = {
        "bonferroni_alpha": round(0.05 / len(di), 5),
        "tests": {t: {"p": p, "bonferroni_reject": bool(p <= 0.05 / len(di)), **dbh2[i]}
                  for i, (t, p) in enumerate(di)}}

    (RES / "honest_stats.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:4000])


if __name__ == "__main__":
    main()
