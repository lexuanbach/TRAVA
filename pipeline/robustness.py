#!/usr/bin/env python3
"""Robustness and rigor analyses. All computed from
CACHED real predictions (no new API calls). Writes results/robustness.json.

Contents:
  did              difference-in-differences: is the closed-book Static-Fresh drop
                   significantly larger than each retriever's / oracle's drop?
  paired_rcb       paired within-slice retrieval-minus-closedbook gains + p-values
  difficulty       closed-book MINUS oracle (and MINUS hybrid) differential
                   (memorization signal net of intrinsic question difficulty)
  cutoff_robust    cross-fit cutoff (odd weeks -> even weeks), permutation null,
                   threshold sensitivity (5/8/12 F1)
  power            minimum detectable effect for the retriever Static-Fresh nulls
  mc               multiple-choice accuracy (map free-form answer to nearest choice)
  stale_vs_cb      is stale-index Fresh F1 significantly below closed-book Fresh?
"""
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

import metrics as M

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
CUT = date(2025, 1, 1)
RNG = np.random.default_rng(0)

MODELS = {"qwen": PROC / "generations_qwen", "qwen1_5b": PROC / "generations_qwen1_5b",
          "qwen3b": PROC / "generations_qwen3b", "qwen7b": PROC / "generations_qwen7b",
          "qwen32b": PROC / "generations_qwen32b", "qwen72b": PROC / "generations_qwen72b",
          "haiku": PROC / "generations", "sonnet": PROC / "generations_sonnet"}
NAME = {"qwen": "Qwen2.5-0.5B", "qwen1_5b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B",
        "qwen7b": "Qwen2.5-7B", "qwen32b": "Qwen2.5-32B", "qwen72b": "Qwen2.5-72B",
        "haiku": "Claude Haiku 4.5", "sonnet": "Claude Sonnet 4.5"}


def parse(s):
    y, m, d = map(int, s.split("-")); return date(y, m, d)


def load_q():
    return {q["qid"]: q for q in (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines())}


def load_cond(gdir, cond):
    p = gdir / f"{cond}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["qid"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}


def split(q):
    return "fresh" if parse(q["date"]) >= CUT else "static"


def f1_by_q(rows, Q):
    return {qid: M.token_f1(r.get("prediction", ""), Q[qid]["gold_answer"])
            for qid, r in rows.items() if qid in Q}


def two_sample(a, b):
    diff, ci, p = M.two_sample_bootstrap(a, b, 2000, seed=0)
    return {"diff": round(100 * diff, 2), "ci95": [round(100 * ci[0], 2), round(100 * ci[1], 2)],
            "p": round(p, 4), "n_a": len(a), "n_b": len(b)}


def did_test(cb, ret, Q):
    """DiD = (cb_static-cb_fresh) - (ret_static-ret_fresh).
    Per-question paired delta d=cb-ret within slice; DiD = mean(d|static)-mean(d|fresh),
    tested with an unpaired two-sample bootstrap on d across the disjoint slices."""
    common = [q for q in cb if q in ret]
    ds = [cb[q] - ret[q] for q in common if split(Q[q]) == "static"]
    df = [cb[q] - ret[q] for q in common if split(Q[q]) == "fresh"]
    r = two_sample(ds, df)
    r["did_f1"] = r.pop("diff")   # DiD point estimate (F1 points)
    return r


def main():
    Q = load_q()
    out = {}
    MAIN = "qwen32b"   # primary (local) generator for the main-table analyses

    # ---- primary-generator conditions (main table generator) ----
    H = {c: f1_by_q(load_cond(MODELS[MAIN], c), Q)
         for c in ("closed_book", "bm25", "dpr", "splade", "colbert", "bge", "hybrid", "oracle")}
    cb = H["closed_book"]
    out["primary_generator"] = NAME[MAIN]

    # (1) DiD: closed-book drop vs each retriever / oracle drop
    out["did"] = {"note": "DiD>0 means retrieval offsets the cutoff collapse more than closed-book; "
                          "tests the interaction, not two marginal tests.",
                  "conditions": {}}
    for c in ("bm25", "dpr", "splade", "colbert", "bge", "hybrid", "oracle"):
        if c in H:
            out["did"]["conditions"][c] = did_test(cb, H[c], Q)

    # (2) paired within-slice R-CB gains + p (retrieval - closed_book on same questions)
    out["paired_rcb"] = {}
    for c in ("bm25", "hybrid", "oracle"):
        if c not in H:
            continue
        d = {}
        for sp in ("static", "fresh"):
            common = [q for q in H[c] if q in cb and split(Q[q]) == sp]
            a = [H[c][q] for q in common]; b = [cb[q] for q in common]
            p = M.paired_bootstrap_pvalue(a, b, 2000, seed=0)
            d[sp] = {"gain_f1": round(100 * (np.mean(a) - np.mean(b)), 2), "p": round(p, 4), "n": len(common)}
        out["paired_rcb"][c] = d

    # (3) difficulty control: closed-book minus oracle / minus hybrid, by year + Static/Fresh
    diffc = {}
    for ref in ("oracle", "hybrid"):
        if ref not in H:
            continue
        byyr = defaultdict(list); bysp = defaultdict(list)
        for q in cb:
            if q in H[ref]:
                dv = cb[q] - H[ref][q]
                byyr[int(Q[q]["date"][:4])].append(dv); bysp[split(Q[q])].append(dv)
        diffc[f"closedbook_minus_{ref}"] = {
            "by_year": {str(y): round(100 * float(np.mean(v)), 1) for y, v in sorted(byyr.items())},
            "static": round(100 * float(np.mean(bysp["static"])), 1),
            "fresh": round(100 * float(np.mean(bysp["fresh"])), 1),
            "static_minus_fresh": round(100 * float(np.mean(bysp["static"]) - np.mean(bysp["fresh"])), 1),
        }
    out["difficulty"] = {"note": "closed-book F1 minus a reference that has the evidence; the "
                                 "Static-Fresh differential is the memorization signal net of "
                                 "intrinsic question difficulty.", **diffc}

    # (4a) cutoff cross-fit: estimate on odd ISO weeks, measure gap on even weeks (Sonnet=biggest cliff)
    def week_parity(qid):
        # qid like YYYYMMDD_idx
        d = parse(Q[qid]["date"]); return d.isocalendar()[1] % 2

    def yearly(rows, subset):
        by = defaultdict(list)
        for q in subset:
            if q in rows:
                by[int(Q[q]["date"][:4])].append(rows[q])
        return {y: 100 * float(np.mean(v)) for y, v in sorted(by.items()) if v}

    def est_cutoff(ymean, thr=8.0):
        ys = sorted(ymean)
        drops = [(ymean[ys[i-1]] - ymean[ys[i]], ys[i]) for i in range(1, len(ys))]
        drop, yr = max(drops)
        return (yr if drop >= thr else None), round(drop, 1)

    out["cutoff_robust"] = {}
    for mk in (MAIN, "haiku", "sonnet"):
        rows = f1_by_q(load_cond(MODELS[mk], "closed_book"), Q)
        odd = [q for q in rows if week_parity(q) == 1]
        even = [q for q in rows if week_parity(q) == 0]
        cut_odd, drop_odd = est_cutoff(yearly(rows, odd))
        b = date(cut_odd or 2025, 1, 1)
        a_even = [rows[q] for q in even if parse(Q[q]["date"]) < b]
        f_even = [rows[q] for q in even if parse(Q[q]["date"]) >= b]
        gap_even = two_sample(a_even, f_even) if a_even and f_even else None
        # permutation null: shuffle year labels, distribution of max year-to-year drop
        allqs = list(rows); vals = np.array([rows[q] for q in allqs])
        years = np.array([int(Q[q]["date"][:4]) for q in allqs])
        obs_drop = est_cutoff(yearly(rows, allqs))[1]
        null = []
        for _ in range(1000):
            perm = RNG.permutation(vals)
            ym = {y: 100 * perm[years == y].mean() for y in np.unique(years)}
            null.append(est_cutoff(ym)[1])
        perm_p = float(np.mean(np.array(null) >= obs_drop))
        # threshold sensitivity
        thr_sens = {}
        for thr in (5, 8, 12):
            c2, d2 = est_cutoff(yearly(rows, allqs), thr)
            thr_sens[str(thr)] = {"cutoff": c2, "max_drop": d2}
        out["cutoff_robust"][mk] = {
            "crossfit": {"cutoff_estimated_on_odd_weeks": cut_odd,
                         "gap_measured_on_even_weeks": gap_even},
            "permutation_null": {"observed_max_drop": round(obs_drop, 1), "p": round(perm_p, 4)},
            "threshold_sensitivity": thr_sens}

    # (4b) power / MDE for retriever Static-Fresh nulls (Haiku)
    out["power"] = {"note": "MDE at 80% power, alpha=0.05 two-sided, for the retriever "
                            "Static-Fresh comparison at the observed n and pooled SD.",
                    "retrievers": {}}
    for c in ("bm25", "dpr", "splade", "colbert", "bge", "hybrid"):
        if c not in H:
            continue
        s = [H[c][q] for q in H[c] if split(Q[q]) == "static"]
        f = [H[c][q] for q in H[c] if split(Q[q]) == "fresh"]
        sd = np.sqrt((np.var(s, ddof=1) + np.var(f, ddof=1)) / 2)
        n_h = 2 / (1 / len(s) + 1 / len(f))  # harmonic-ish effective n/group
        mde = (1.96 + 0.84) * sd * np.sqrt(2 / n_h)   # two-sample MDE
        out["power"]["retrievers"][c] = {"observed_diff_f1": round(100 * (np.mean(s) - np.mean(f)), 2),
                                         "mde_f1_80pct": round(100 * mde, 1),
                                         "n_static": len(s), "n_fresh": len(f)}

    # (5) MC accuracy via free-form -> nearest choice (all models), Static/Fresh
    def mc_acc(rows):
        by = defaultdict(lambda: [0, 0])
        for q, r in rows.items():
            Qi = Q.get(q)
            if not Qi or "choices" not in Qi or Qi.get("gold_answer_idx") is None:
                continue
            pred = r.get("prediction", "")
            scores = [M.token_f1(pred, ch) for ch in Qi["choices"]]
            pick = int(np.argmax(scores)) if max(scores) > 0 else -1
            correct = int(pick == Qi["gold_answer_idx"])
            sp = split(Qi); by[sp][0] += correct; by[sp][1] += 1
            by["all"][0] += correct; by["all"][1] += 1
        return {sp: round(100 * c / n, 1) for sp, (c, n) in by.items() if n}

    out["mc"] = {"note": "MC-proxy: free-form answer mapped to the choice with highest token-F1; "
                         "correct if it matches gold_answer_idx. Random baseline 25%.",
                 "models": {}}
    for mk, gdir in MODELS.items():
        cbrows = load_cond(gdir, "closed_book"); hyrows = load_cond(gdir, "hybrid")
        out["mc"]["models"][mk] = {"name": NAME[mk], "closed_book": mc_acc(cbrows),
                                   "hybrid": mc_acc(hyrows) if hyrows else None}

    # (6) stale-index Fresh vs closed-book Fresh (Qwen-3B) — is stale < closed-book?
    q3 = MODELS["qwen3b"]
    cb3 = f1_by_q(load_cond(q3, "closed_book"), Q)
    st3 = f1_by_q(load_cond(q3, "hybrid_stale"), Q)
    if st3:
        common = [q for q in st3 if q in cb3 and split(Q[q]) == "fresh"]
        a = [cb3[q] for q in common]; b = [st3[q] for q in common]  # cb - stale
        p = M.paired_bootstrap_pvalue(a, b, 2000, seed=0)
        out["stale_vs_cb"] = {"closed_book_fresh_f1": round(100 * float(np.mean(a)), 2),
                              "stale_hybrid_fresh_f1": round(100 * float(np.mean(b)), 2),
                              "diff_cb_minus_stale": round(100 * (np.mean(a) - np.mean(b)), 2),
                              "one_sided_p_cb_gt_stale": round(p, 4), "n": len(common),
                              "note": "paired over same Fresh questions (Qwen-3B)"}

    (RES / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
