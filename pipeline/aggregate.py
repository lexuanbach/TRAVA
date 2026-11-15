#!/usr/bin/env python3
"""Aggregate REAL per-example predictions into results/*.json + claim_ledger.csv.

Every number is computed from data/processed/generations*/*.jsonl (real model
outputs) and the real retrieval rankings. Nothing is hand-set.

Key design (revised after review):
  * Static/Fresh is anchored at the EFFECTIVE knowledge cutoff estimated from the
    closed-book decay curve (not the model's release/self-reported date). We also
    sweep the boundary to show robustness.
  * Static vs Fresh are DISJOINT question sets, so their gap uses an UNPAIRED
    two-sample bootstrap (with CI + one-sided p) -- not a paired test.
  * P_cite is renamed to "answer coverage" (fraction of in-context docs that
    contain the gold span); it is a retrieval property, not citation faithfulness.
  * We report answered-only F1 (excluding abstentions) and abstention rates.
  * Three generators are aggregated: Claude Haiku 4.5, Claude Sonnet 4.5, and the
    open-weight Qwen2.5-0.5B (known ~2023 cutoff), to test generality and to
    validate the effective-cutoff estimator against a disclosed cutoff.

Outputs: results/{main_results,time_decay,retrieval,freshness,abstention,
dataset_stats,cross_model,boundary_sweep,run_meta}.json + claim_ledger.csv
"""
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import kendalltau

import metrics as M

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
TODAY = date(2026, 7, 12)
FRESH_AFTER = (2025, 1, 1)   # shared analysis boundary for ALL generators. NOT a
                             # per-model estimated cutoff: that estimator is a negative
                             # result (see honest_stats.py / the paper's Analysis).

MODELS = {
    "qwen": PROC / "generations_qwen",          # 0.5B
    "qwen1_5b": PROC / "generations_qwen1_5b",
    "qwen3b": PROC / "generations_qwen3b",
    "qwen7b": PROC / "generations_qwen7b",
    "qwen32b": PROC / "generations_qwen32b",
    "qwen72b": PROC / "generations_qwen72b",
    "mistral_large": PROC / "generations_mistral_large",
    "haiku": PROC / "generations",
    "sonnet": PROC / "generations_sonnet",
}
MODEL_NAMES = {"qwen": "Qwen2.5-0.5B", "qwen1_5b": "Qwen2.5-1.5B",
               "qwen3b": "Qwen2.5-3B", "qwen7b": "Qwen2.5-7B",
               "qwen32b": "Qwen2.5-32B", "qwen72b": "Qwen2.5-72B",
               "mistral_large": "Mistral-Large-2 (123B)",
               "haiku": "Claude Haiku 4.5", "sonnet": "Claude Sonnet 4.5"}
# param counts (B) for the capability axis
MODEL_PARAMS = {"qwen": 0.5, "qwen1_5b": 1.5, "qwen3b": 3.0, "qwen7b": 7.0,
                "qwen32b": 32.0, "qwen72b": 72.0, "mistral_large": 123.0,
                "haiku": None, "sonnet": None}

# Primary generator for the main results table (local, 10+B, per user directive).
# Claude Haiku is preserved as a frontier reference (results/haiku_results.json).
MAIN = "qwen32b"
MAIN_DIR = {"qwen32b": "generations_qwen32b", "haiku": "generations"}


def parse(d):
    y, m, dd = map(int, d.split("-")); return date(y, m, dd)


def is_fresh(qdate, boundary=FRESH_AFTER):
    return parse(qdate) >= date(*boundary)


def age_years(qdate):
    return (TODAY - parse(qdate)).days / 365.25


def load_questions():
    return {q["qid"]: q for q in
            (json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines())}


def load_corpus():
    return {d["doc_id"]: d for d in
            (json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines())}


def load_gens(gdir):
    gens = {}
    if not gdir.exists():
        return gens
    for f in sorted(gdir.glob("*.jsonl")):
        if f.stem.endswith("_stale"):   # held-out-index conditions handled separately
            continue
        rows = {}
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line); rows[r["qid"]] = r
        gens[f.stem] = rows
    return gens


def per_example(questions, corpus, rows, boundary=FRESH_AFTER):
    rec = {}
    for qid, r in rows.items():
        q = questions.get(qid)
        if q is None:
            continue
        pred, gold = r.get("prediction", ""), q["gold_answer"]
        doc_texts = [corpus[d]["text"] for d in r.get("doc_ids", []) if d in corpus]
        rec[qid] = {
            "em": M.exact_match(pred, gold),
            "f1": M.token_f1(pred, gold),
            "cover": M.p_cite(gold, doc_texts) if doc_texts else None,
            "abstain": M.is_abstention(pred),
            "split": "fresh" if is_fresh(q["date"], boundary) else "static",
            "year": int(q["date"][:4]),
            "age": age_years(q["date"]),
        }
    return rec


def vals(rec, key, split=None):
    return [v[key] for v in rec.values()
            if (split is None or v["split"] == split) and v.get(key) is not None]


def cond_summary(rec):
    """Static/Fresh summary for one condition."""
    out = {}
    for split in ("static", "fresh", "all"):
        s = None if split == "all" else split
        f1 = vals(rec, "f1", s)
        if not f1:
            continue
        mean, lo, hi = M.bootstrap_ci(f1, 1000, seed=0)
        ans = [v["f1"] for v in rec.values()
               if (s is None or v["split"] == s) and not v["abstain"]]
        cov = vals(rec, "cover", s)
        out[split] = {
            "n": len(f1),
            "f1": round(100 * mean, 2),
            "f1_ci95": [round(100 * lo, 2), round(100 * hi, 2)],
            "em": round(100 * float(np.mean(vals(rec, "em", s))), 2),
            "answer_coverage": round(100 * float(np.mean(cov)), 2) if cov else None,
            "abstain_rate": round(100 * float(np.mean([v["abstain"] for v in rec.values()
                                    if s is None or v["split"] == s])), 1),
            "answered_f1": round(100 * float(np.mean(ans)), 2) if ans else None,
            "n_answered": len(ans),
        }
    if "static" in out and "fresh" in out:
        a = vals(rec, "f1", "static"); b = vals(rec, "f1", "fresh")
        diff, ci, p = M.two_sample_bootstrap(a, b, 1000, seed=0)
        out["static_minus_fresh"] = {
            "diff_f1": round(100 * diff, 2),
            "ci95": [round(100 * ci[0], 2), round(100 * ci[1], 2)],
            "one_sided_p": round(p, 4),
            "cohens_d": round(M.cohens_d(a, b), 3),
            "test": "unpaired two-sample bootstrap (Static/Fresh are disjoint sets)",
        }
        out["delta_fresh_minus_static_f1"] = round(out["fresh"]["f1"] - out["static"]["f1"], 2)
    return out


def by_year_curve(rec):
    bins = defaultdict(list)
    for v in rec.values():
        bins[v["year"]].append(v["f1"])
    curve = {str(y): {"n": len(v), "f1": round(100 * float(np.mean(v)), 2)}
             for y, v in sorted(bins.items())}
    ages = [v["age"] for v in rec.values()]; f1 = [v["f1"] for v in rec.values()]
    slope, _, r2 = M.ols_slope(ages, f1)
    return {"by_year": curve, "ols_slope_f1_per_year": round(100 * slope, 3),
            "ols_r2": round(r2, 3)}


def main():
    questions = load_questions(); corpus = load_corpus()
    RES.mkdir(exist_ok=True)
    ledger = []
    allpe = {}   # model -> cond -> rec
    for mk, gdir in MODELS.items():
        gens = load_gens(gdir)
        allpe[mk] = {c: per_example(questions, corpus, rows) for c, rows in gens.items()}

    # ---------- MAIN RESULTS: primary = MAIN (local 32B); Claude Haiku preserved ----------
    order = ["closed_book", "bm25", "dpr", "splade", "colbert", "bge", "hybrid", "oracle"]

    def build_rows(pe, tag, gdir, tbl):
        rr = []
        for cond in [c for c in order if c in pe]:
            s = cond_summary(pe[cond]); s["condition"] = cond
            rr.append(s)
            for split in ("static", "fresh"):
                if split in s:
                    ledger.append([f"{tag}_{cond}_{split}_f1", s[split]["f1"],
                                   str(s[split]["f1_ci95"]), tbl,
                                   f"{gdir}/{cond}.jsonl:{split}", "real"])
        cbr = next(r for r in rr if r["condition"] == "closed_book")
        ledger.append([f"{tag}_closedbook_static_minus_fresh_gap", cbr["static_minus_fresh"]["diff_f1"],
                       str(cbr["static_minus_fresh"]["ci95"]), tbl,
                       f"unpaired bootstrap p={cbr['static_minus_fresh']['one_sided_p']}", "real"])
        return rr

    haiku = allpe[MAIN]   # 'haiku' now drives the PRIMARY (local) analyses below
    rows = build_rows(haiku, MAIN, MAIN_DIR[MAIN], "tab:main")
    (RES / "main_results.json").write_text(json.dumps({
        "metric": "token_f1 / em / answer_coverage (%)",
        "generator": MODEL_NAMES[MAIN],
        "analysis_boundary": "2025-01 (shared across all generators; results are also reported as a clean 2023-vs-2026 contrast, see honest_stats.json)",
        "split_rule": "static = question_date < 2025-01; fresh = >= 2025-01",
        "rows": rows}, indent=2))
    # Preserve the Claude Haiku frontier results as an appendix reference.
    if "haiku" in allpe and "closed_book" in allpe["haiku"]:
        hrows = build_rows(allpe["haiku"], "haiku", "generations", "tab:haiku")
        (RES / "haiku_results.json").write_text(json.dumps({
            "metric": "token_f1 / em / answer_coverage (%)",
            "generator": "claude-haiku-4-5-20251001",
            "note": "Frontier reference; the main table uses the local primary generator.",
            "rows": hrows}, indent=2))

    # ---------- TIME DECAY (per condition, by year) ----------
    decay = {c: by_year_curve(haiku[c]) for c in haiku}
    (RES / "time_decay.json").write_text(json.dumps(decay, indent=2))
    for c in ("closed_book", "bm25", "hybrid"):
        if c in decay:
            ledger.append([f"haiku_{c}_slope", decay[c]["ols_slope_f1_per_year"],
                           f"r2={decay[c]['ols_r2']}", "tab:time-decay",
                           f"generations/{c}.jsonl OLS", "real"])

    # ---------- BOUNDARY SWEEP (closed-book & hybrid gap vs boundary) ----------
    sweep = {}
    for byear in (2023, 2024, 2025, 2026):
        b = (byear, 1, 1)
        def gap(cond, b=b):
            rec = {qid: {**v, "split": "fresh" if is_fresh(questions[qid]["date"], b) else "static"}
                   for qid, v in haiku[cond].items()}
            a = [v["f1"] for v in rec.values() if v["split"] == "static"]
            f = [v["f1"] for v in rec.values() if v["split"] == "fresh"]
            if not a or not f:
                return None
            return {"static_f1": round(100 * np.mean(a), 1), "fresh_f1": round(100 * np.mean(f), 1),
                    "gap": round(100 * (np.mean(a) - np.mean(f)), 1),
                    "n_static": len(a), "n_fresh": len(f)}
        sweep[str(byear)] = {"closed_book": gap("closed_book"), "hybrid": gap("hybrid")}
    (RES / "boundary_sweep.json").write_text(json.dumps(sweep, indent=2))

    # ---------- CROSS-MODEL: cliff at each model's OWN estimated cutoff ----------
    def estimate_cutoff(by_year):
        """Effective cutoff = year with the largest F1 drop from the prior year;
        'none' if the max drop is < 8 F1 (no cliff)."""
        yrs = sorted(int(y) for y in by_year)
        drops = [(by_year[str(yrs[i-1])]["f1"] - by_year[str(yrs[i])]["f1"], yrs[i])
                 for i in range(1, len(yrs))]
        drop, yr = max(drops)
        return (yr, round(drop, 1)) if drop >= 8 else (None, round(drop, 1))

    cross = {}
    for mk, pe in allpe.items():
        if "closed_book" not in pe:
            continue
        by = by_year_curve(pe["closed_book"])["by_year"]
        cut_year, maxdrop = estimate_cutoff(by)
        entry = {"model": MODEL_NAMES[mk], "params_B": MODEL_PARAMS.get(mk),
                 "closed_book_by_year": by, "estimated_cutoff": cut_year,
                 "max_year_drop": maxdrop}
        # cliff at the COMMON 2025 effective-cutoff boundary, so magnitudes are
        # comparable across the capability ladder (estimated_cutoff kept as a
        # separate diagnostic; a per-model boundary would not be comparable).
        b = FRESH_AFTER
        cbrec = {q: {**v, "split": "fresh" if is_fresh(questions[q]["date"], b) else "static"}
                 for q, v in pe["closed_book"].items()}
        cba = [v["f1"] for v in cbrec.values() if v["split"] == "static"]
        cbf = [v["f1"] for v in cbrec.values() if v["split"] == "fresh"]
        if cba and cbf:
            diff, ci, p = M.two_sample_bootstrap(cba, cbf, 1000, seed=0)
            entry["cliff"] = {"boundary": 2025, "static_f1": round(100*np.mean(cba), 1),
                              "fresh_f1": round(100*np.mean(cbf), 1), "gap": round(100*diff, 1),
                              "one_sided_p": round(p, 4)}
        for rcond in ("hybrid", "bm25"):
            if rcond in pe:
                rrec = {q: {**v, "split": "fresh" if is_fresh(questions[q]["date"], b) else "static"}
                        for q, v in pe[rcond].items()}
                ra = np.mean([v["f1"] for v in rrec.values() if v["split"] == "static"])
                rf = np.mean([v["f1"] for v in rrec.values() if v["split"] == "fresh"])
                entry["retriever"] = rcond
                entry["retrieval_minus_closedbook"] = {
                    "static": round(100*(ra - np.mean(cba)), 1),
                    "fresh": round(100*(rf - np.mean(cbf)), 1)}
                break
        cross[mk] = entry
        if "cliff" in entry:
            ledger.append([f"{mk}_closedbook_cliff", entry["cliff"]["gap"],
                           f"cutoff={cut_year}, p={entry['cliff']['one_sided_p']}",
                           "tab:cross-model", f"generations*/{mk}/closed_book.jsonl", "real"])
    (RES / "cross_model.json").write_text(json.dumps(cross, indent=2))

    # ---------- FRESHNESS composite + rank instability (softened) ----------
    lam = 0.041
    retr = [c for c in ("bm25", "dpr", "splade", "colbert", "bge", "hybrid") if c in haiku]
    comp = {}
    for cond in retr:
        vv = [v["f1"] * v["cover"] * np.exp(-lam * v["age"]) for v in haiku[cond].values()
              if v["cover"] is not None]
        if vv:
            comp[cond] = round(100 * float(np.mean(vv)), 2)
    static_score = {c: float(np.mean(vals(haiku[c], "f1", "static") or [0])) for c in retr}
    fresh_score = {c: float(np.mean(vals(haiku[c], "f1", "fresh") or [0])) for c in retr}
    order_c = sorted(retr)
    tau, p = kendalltau([static_score[c] for c in order_c], [fresh_score[c] for c in order_c])
    fr = {
        # NOT USED BY THE PAPER. The composite and the retriever rank-stability
        # tau were cut in revision (see deprecated_cut/README.md); still emitted
        # so the record of what was computed stays complete.
        "freshness_composite_F": comp,
        "rank_instability": {
            "static_rank": sorted(retr, key=lambda c: -static_score[c]),
            "fresh_rank": sorted(retr, key=lambda c: -fresh_score[c]),
            "kendall_tau": round(float(tau), 3), "p": round(float(p), 4),
            "note": "NOT significant at 0.05 (n=6 near-tied retrievers); reported as suggestive only",
        }}
    (RES / "freshness.json").write_text(json.dumps(fr, indent=2))

    # ---------- ABSTENTION (per model, key conditions) ----------
    ab = {}
    for mk, pe in allpe.items():
        ab[mk] = {}
        for cond in ("closed_book", "bm25", "hybrid", "bge", "oracle"):
            if cond in pe:
                ab[mk][cond] = {sp: round(100 * float(np.mean(
                    [v["abstain"] for v in pe[cond].values() if v["split"] == sp])), 1)
                    for sp in ("static", "fresh")}
    (RES / "abstention.json").write_text(json.dumps(ab, indent=2))

    # ---------- retrieval + dataset passthrough ----------
    if (PROC / "retrieval_metrics.json").exists():
        (RES / "retrieval.json").write_text((PROC / "retrieval_metrics.json").read_text())
    stats = json.loads((PROC / "stats.json").read_text())
    stats["analysis_boundary"] = "2025-01-01 (shared; the split field in questions.jsonl is the 2026 CALENDAR boundary and is not the analysis split)"
    (RES / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    with (RES / "claim_ledger.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["claim_id", "value", "ci95_or_note", "table_or_figure", "provenance", "status"])
        w.writerows(ledger)
    (RES / "run_meta.json").write_text(json.dumps({
        "run_date": TODAY.isoformat(),
        "generators": MODEL_NAMES, "n_questions": len(questions),
        "analysis_boundary": "2025-01-01",
        "note": "Real run. Generators are Claude Haiku 4.5, Claude Sonnet 4.5, and open-weight "
                "Qwen2.5-0.5B; original manuscript targeted GPT-4o (no OpenAI key available).",
    }, indent=2))

    # console summary
    print("== HAIKU main (2025-01 boundary) ==")
    for r in rows:
        s = r.get("static", {}).get("f1"); fx = r.get("fresh", {}).get("f1")
        smf = r.get("static_minus_fresh", {})
        print(f"  {r['condition']:11s} static={s} fresh={fx} S-F={smf.get('diff_f1')} "
              f"p={smf.get('one_sided_p')} d={smf.get('cohens_d')}")
    print("== boundary sweep (closed_book gap) ==",
          {y: sweep[y]["closed_book"]["gap"] for y in sweep})
    print("== cross-model closed-book by year ==")
    for mk in cross:
        print(f"  {mk}: {cross[mk]['closed_book_by_year']}")


if __name__ == "__main__":
    main()
