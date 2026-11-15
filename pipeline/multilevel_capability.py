#!/usr/bin/env python3
"""Item-level multilevel test of the fresh x capability interaction (TODO H-4).

The paper's capability claim has so far rested on a slope fitted to seven
per-model point estimates. Even with a joint quiz bootstrap
(`joint_slope.py`) that design has two weaknesses:

  1. the predictor is parameter count, which conflates capability with
     model family, training mixture and cutoff; and
  2. it discards item-level structure, so the year-difficulty drift that
     this paper is otherwise about is never modelled.

This script addresses both.

**Capability proxy.** Instead of parameter count we use each model's
mean token-F1 on the *Fresh* slice **under retrieval** (hybrid). Fresh
questions post-date every model's pretraining, so no model can answer
them from memory; performance there measures the ability to read and use
supplied evidence. It is therefore a capability measure that is
independent, by construction, of the parametric-memory quantity whose
scaling we are testing. (Parameter count is retained as a secondary
predictor for comparison.)

**Model.** One row per (question, model) closed-book answer:

    f1 ~ fresh + capability_c + fresh:capability_c + (1 | question)

fitted by REML with a random intercept for the *question*. Note what this
does and does not do: it accounts for repeated measurement of the same
question across generators, but because every question is
deterministically Static or Fresh, a question intercept cannot remove
systematic differences between the two sets. We therefore do NOT claim
difficulty is "absorbed". Sign convention: `fresh` is an
indicator, so the paper's boundary *gap* (Static minus Fresh) is
**minus** the `fresh` coefficient, and a **negative** `fresh:capability`
interaction means the gap *widens* with capability.

**Inference caveat, stated because it is decisive.** Capability varies
only between models, so the effective sample size for the interaction is
the number of models (six open-weight rungs), not the number of rows.
The item-level p-value is therefore anticonservative for a model-level
claim and we do not report it as the headline. What the item-level fit
legitimately buys is difficulty absorption; what it cannot buy is
model-level replication. We accompany it with a leave-one-model-out
refit, which is the direct test of whether a single rung carries the
association.

Outputs `results/multilevel_capability.json`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from metrics import token_f1  # noqa: E402

# Every model with both a closed-book and a hybrid-retrieval run cached,
# so the capability proxy is measured under an identical condition.
MODELS = [
    ("qwen1_5b", "Qwen2.5-1.5B", 1.5, "open"),
    ("qwen3b", "Qwen2.5-3B", 3.0, "open"),
    ("qwen7b", "Qwen2.5-7B", 7.0, "open"),
    ("qwen32b", "Qwen2.5-32B", 32.0, "open"),
    ("qwen72b", "Qwen2.5-72B", 72.0, "open"),
    ("mistral_large", "Mistral-Large-2", 123.0, "open"),
    ("gpt4omini", "GPT-4o-mini", None, "api"),
    ("sonnet", "Claude-Sonnet-4.5", None, "api"),
]
BOUNDARY = date(2025, 1, 1)


def load_questions() -> dict[str, dict]:
    path = ROOT / "data/processed/questions.jsonl"
    return {json.loads(l)["qid"]: json.loads(l) for l in path.open(encoding="utf-8")}


def load_condition(model: str, cond: str, questions: dict) -> dict[str, float]:
    path = ROOT / f"data/processed/generations_{model}/{cond}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        q = questions.get(rec["qid"])
        if q:
            out[rec["qid"]] = 100.0 * token_f1(rec["prediction"], q["gold_answer"])
    return out


def q_year(q: dict) -> int:
    return int(str(q["date"])[:4])


def is_fresh(q: dict) -> bool:
    y, m, d = (int(x) for x in str(q["date"])[:10].split("-"))
    return date(y, m, d) >= BOUNDARY


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/multilevel_capability.json")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    questions = load_questions()

    # --- capability proxy: Fresh-slice retrieval-augmented F1 -------------
    cap, rows = {}, []
    for key, name, params, fam in MODELS:
        cb = load_condition(key, "closed_book", questions)
        hy = load_condition(key, "hybrid", questions)
        if not cb or not hy:
            print(f"  [skip] {name}: missing closed_book or hybrid")
            continue
        fresh_hy = [v for qid, v in hy.items() if is_fresh(questions[qid])]
        cap[key] = statistics.fmean(fresh_hy)
        for qid, f1 in cb.items():
            q = questions[qid]
            rows.append({
                "qid": qid, "model": key, "model_name": name,
                "quiz": q["date"],  # RealTimeQA ships one dated quiz per week
                "year": q_year(q), "fresh": int(is_fresh(q)),
                "f1": f1, "params_B": params, "family": fam,
            })
        print(f"  {name:18s} capability (Fresh RAG F1) = {cap[key]:5.2f}  "
              f"n_cb={len(cb)}")

    df = pd.DataFrame(rows)
    df["capability"] = df["model"].map(cap)
    # centre so `fresh` reads as the gap at mean capability
    cap_mean = float(statistics.fmean(cap.values()))
    df["capability_c"] = df["capability"] - cap_mean
    df["log_params_c"] = np.where(
        df["params_B"].notna(), np.log10(df["params_B"].astype(float)), np.nan)
    df["log_params_c"] = df["log_params_c"] - df["log_params_c"].mean()

    print(f"\nrows={len(df)}  models={df['model'].nunique()}  "
          f"quizzes={df['quiz'].nunique()}  mean capability={cap_mean:.2f}")

    def fit(formula, data, label):
        m = smf.mixedlm(formula, data, groups=data["qid"]).fit(reml=True)
        term = "fresh:capability_c" if "capability_c" in formula else "fresh:log_params_c"
        if term not in m.params.index:
            term = [t for t in m.params.index if t.startswith("fresh:")][0]
        ci = m.conf_int().loc[term]
        return {
            "label": label, "formula": formula, "n_obs": int(len(data)),
            "n_groups_question": int(data["qid"].nunique()),
            "n_models": int(data["model"].nunique()),
            "interaction_term": term,
            "coef": float(m.params[term]),
            "se": float(m.bse[term]),
            "z": float(m.tvalues[term]),
            "p": float(m.pvalues[term]),
            "ci95": [float(ci[0]), float(ci[1])],
            "fresh_main": float(m.params["fresh"]),
            "fresh_main_p": float(m.pvalues["fresh"]),
        }

    out = {
        "scope": "Item-level test of whether the closed-book boundary gap "
                 "widens with capability. The random intercept is on the "
                 "QUESTION (`qid`), not the weekly quiz batch: it accounts "
                 "for repeated measurement of the same question across "
                 "generators. It does NOT absorb systematic differences "
                 "between the Static and Fresh question sets, because every "
                 "question is deterministically in one set or the other, so "
                 "the fixed `fresh` coefficient still rests on the "
                 "random-effects independence assumption.",
        "boundary": BOUNDARY.isoformat(),
        "capability_proxy": "mean token-F1 on the Fresh slice under hybrid "
                            "retrieval; Fresh post-dates every model's "
                            "pretraining, so this measures use of supplied "
                            "evidence and not parametric recall",
        "capability_by_model": {k: round(v, 4) for k, v in cap.items()},
        "capability_mean": round(cap_mean, 4),
        "fits": {},
    }

    open_df = df[df["family"] == "open"].copy()
    out["fits"]["capability_proxy_open"] = fit(
        "f1 ~ fresh * capability_c", open_df,
        "Fresh x capability (RAG-F1 proxy), open-weight rungs")
    out["fits"]["log_params_open"] = fit(
        "f1 ~ fresh * log_params_c", open_df,
        "Fresh x log10(parameters), open-weight rungs")

    # The proxy is invalid where a model abstains heavily: Claude-Sonnet
    # scores lowest on it (below Qwen2.5-3B) because it declines to answer
    # ~56% of Fresh questions, not because it is least capable. We record
    # the all-model fit only to document that failure, not as evidence.
    out["fits"]["capability_proxy_all_models_INVALID"] = fit(
        "f1 ~ fresh * capability_c", df,
        "All 8 models (proxy invalid for abstainers -- documentation only)")
    out["proxy_failure_note"] = (
        "Claude-Sonnet-4.5's Fresh RAG-F1 (%.2f) is the lowest of the eight, "
        "below Qwen2.5-3B (%.2f), because it abstains on roughly half of "
        "Fresh questions. The proxy measures answered-question accuracy and "
        "so mis-ranks heavy abstainers; the interaction is therefore "
        "estimated on the open-weight rungs, where abstention is under 7%%."
        % (cap.get("sonnet", float("nan")), cap.get("qwen3b", float("nan")))
    )

    # Leave-one-model-out: does one rung carry the association?
    lomo = {}
    for key, name, _p, fam in MODELS:
        if fam != "open" or key not in cap:
            continue
        sub = open_df[open_df["model"] != key]
        if sub["model"].nunique() < 3:
            continue
        f = fit("f1 ~ fresh * capability_c", sub, f"drop {name}")
        lomo[key] = {"dropped": name, "coef": f["coef"], "ci95": f["ci95"]}
        print(f"    LOMO drop {name:18s} coef={f['coef']:+.3f} "
              f"CI[{f['ci95'][0]:+.3f},{f['ci95'][1]:+.3f}]")
    coefs = [v["coef"] for v in lomo.values()]
    out["leave_one_model_out"] = {
        "note": "Refit dropping each open-weight rung in turn. Read the SIGN "
                "pattern, not the intervals: capability varies only between "
                "models, so every interval here is pseudoreplicated at the "
                "item level and is far too narrow for a model-level claim. "
                "What the refits establish is that the direction does not "
                "depend on any one rung; they establish nothing about "
                "significance.",
        "intervals_are_pseudoreplicated": True,
        "fits": lomo,
        "coef_range": [min(coefs), max(coefs)] if coefs else None,
        "all_same_sign": bool(coefs) and (all(c < 0 for c in coefs) or all(c > 0 for c in coefs)),
    }

    out["interpretation"] = 'At centered capability c the Static-minus-Fresh contrast is -(beta_fresh + beta_interaction*c). Centering uses the recorded eight-model mean before selecting the six open-weight models. The random intercept accounts for repeated questions across models, not between-slice difficulty. The year-pair analysis is a sensitivity check, not a difficulty control. Model-level information comes from six models; the fit is a descriptive association.'

    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    for k, f in out["fits"].items():
        print(f"  {f['label']:44s} coef={f['coef']:+.4f} "
              f"CI[{f['ci95'][0]:+.3f},{f['ci95'][1]:+.3f}] p={f['p']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
