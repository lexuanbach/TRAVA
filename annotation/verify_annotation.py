#!/usr/bin/env python3
"""Recompute every annotation-derived number in the paper, from the raw returns.

One command, no arguments, no cached JSON: this reads the returned annotator
CSVs and recomputes each figure the manuscript quotes, then prints it beside the
value stored in results/. A user can run it to confirm that nothing in the
pipeline between the CSVs and the paper is doing unexplained work.

    python annotation/verify_annotation.py

Exit code 0 means every recomputed number matches its stored counterpart.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import sys
from pathlib import Path

# these scripts import siblings by path; keep the shipped
# artifact free of __pycache__ directories
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "annotation"))
from score_annotations import (cohens_kappa,                     # noqa: E402
                               krippendorff_alpha_nominal, majority)

PACK = ROOT / "annotation" / "pack_2026-08-02"
REPILOT = ROOT / "annotation" / "freshness_repilot" / "pack"
ADJ = ROOT / "annotation" / "adjudication" / "pack"
TOL = 0.002


def read(stem, keys, col, where=PACK):
    out = {}
    for f in sorted(glob.glob(str(where / f"{stem}__annotator?.csv"))):
        w = Path(f).name.split("__annotator")[1][0]
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            out.setdefault(tuple(r[k] for k in keys), {})[w] = \
                (r.get(col) or "").strip().upper()
    return out


def main() -> int:
    checks, bad = [], 0

    def cmp(label, got, want, tol=TOL):
        nonlocal bad
        ok = want is not None and abs(got - want) <= tol
        checks.append((ok, label, got, want))
        if not ok:
            bad += 1

    # ---- inter-annotator agreement, main pack --------------------------
    key = {r["item_id"]: r for r in
           (json.loads(l) for l in
            (PACK / "item_metadata.jsonl").read_text().splitlines() if l.strip())}
    ar = json.loads((PACK / "agreement_report.json").read_text())
    for stem, keys, col, name in (
            ("task_a_correctness", ("item_id",), "verdict", "task_a_correctness:verdict"),
            ("task_b_evidence", ("item_id", "evidence_id"), "supports", "task_b_evidence:supports"),
            ("task_b_evidence", ("item_id", "evidence_id"), "freshness", "task_b_evidence:freshness")):
        d = read(stem, keys, col)
        who = sorted(next(iter(d.values())))
        ks = [k for k in sorted(d)
              if not key.get(k[0], {}).get("stratum", "").startswith("calibration")]
        a = krippendorff_alpha_nominal([[d[k][w] for w in who] for k in ks])
        cmp(f"alpha {col}", round(a, 4),
            ar["tasks"].get(name, {}).get("krippendorff_alpha"))

    # ---- freshness sub-codings ------------------------------------------
    fr = read("task_b_evidence", ("item_id", "evidence_id"), "freshness")
    who = sorted(next(iter(fr.values())))
    ks = sorted(fr)
    hm = json.loads((ROOT / "results/human_metric_audit.json").read_text())
    f2 = hm["freshness_construct_reliability"]
    for name, fn, want_key in (
            ("ANACHRONISTIC vs rest",
             lambda v: "A" if v == "ANACHRONISTIC" else "O", "anachronistic_vs_rest"),
            ("SUPERSEDED vs rest",
             lambda v: "S" if v == "SUPERSEDED" else "O", "superseded_vs_rest")):
        a = krippendorff_alpha_nominal([[fn(fr[k][w]) for w in who] for k in ks])
        cmp(f"alpha {name}", round(a, 4), f2.get(want_key))

    # ---- consensus + adjudication ---------------------------------------
    hv = json.loads((ROOT / "results/human_validation.json").read_text())
    va = read("task_a_correctness", ("item_id",), "verdict")
    over = {}
    f = PACK / "task_a_correctness__adjudicated.csv"
    if f.is_file():
        over = {(r["item_id"],): r["adjudicated"].strip().upper()
                for r in csv.DictReader(f.open(encoding="utf-8"))}
    who = sorted(next(iter(va.values())))
    lab = {}
    for k in sorted(va):
        m, ok = majority([va[k][w] for w in who])
        if ok:
            lab[k] = m
    lab.update(over)
    cmp("n items scored", float(len(lab)),
        float(hv["population_weighted"]["n_items"]), 0.5)

    man = json.loads((PACK / "sampling_manifest.json").read_text())["task_a"]["strata"]
    W = {s: v["weight_to_population"] for s, v in man.items()}
    num = den = miss = cred = 0.0
    for (iid,), v in lab.items():
        k = key.get(iid)
        if not k:
            continue
        w = W[k["stratum"]]
        f1 = k["auto_f1"] >= 0.5
        h = v == "CORRECT"
        den += w
        num += w * (f1 == h)
        miss += w * (h and not f1)
        cred += w * (f1 and not h)
    pw = hv["population_weighted"]
    cmp("f1 agrees with human", round(num / den, 4), pw["f1_agrees_with_human"])
    cmp("f1 misses a right answer", round(miss / den, 4), pw["f1_misses_a_human_right"])
    cmp("f1 credits a non-correct", round(cred / den, 4),
        pw["f1_credits_a_non_correct"])

    aj = ROOT / "results/adjudication.json"
    if aj.is_file():
        d = json.loads(aj.read_text())
        votes = {}
        for line in (ADJ / "adjudication_metadata.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                votes[(r["field"], r["item_id"], r["evidence_id"])] = r["votes"]
        for field, fn in (("verdict", "adjudicate_verdict.csv"),
                          ("supports", "adjudicate_supports.csv")):
            rows = list(csv.DictReader((ADJ / fn).open(encoding="utf-8-sig")))
            dep = sum(1 for r in rows
                      if r["adjudicated"].strip().upper() !=
                      collections.Counter(votes[(field, r["item_id"],
                                                 r.get("evidence_id", ""))].values()
                                          ).most_common(1)[0][0])
            cmp(f"departure rate {field}", round(dep / len(rows), 4),
                d["fields"][field]["departure_rate"])

    # ---- freshness re-pilot ---------------------------------------------
    rp = ROOT / "results/freshness_repilot.json"
    if rp.is_file() and list(REPILOT.glob("freshness_repilot__annotator?.csv")):
        d = json.loads(rp.read_text())
        pr = read("freshness_repilot", ("item_id", "evidence_id"), "freshness",
                  where=REPILOT)
        who = sorted(next(iter(pr.values())))
        ks = sorted(pr)
        cmp("re-pilot alpha freshness",
            round(krippendorff_alpha_nominal([[pr[k][w] for w in who] for k in ks]), 4),
            d["freshness"]["krippendorff_alpha"])
        NA = "NOT_ABOUT_THIS_QUESTION"
        sub = [k for k in ks if all(pr[k][w] != NA for w in who)]
        cmp("re-pilot alpha, on-topic only",
            round(krippendorff_alpha_nominal([[pr[k][w] for w in who] for k in sub]), 4),
            d["scope_disagreement"]["alpha_excluding_any_off_topic_call"])

    width = max(len(c[1]) for c in checks)
    print(f"{'check':<{width}}  {'recomputed':>11}  {'stored':>9}")
    for ok, label, got, want in checks:
        mark = "ok " if ok else "FAIL"
        print(f"{mark} {label:<{width - 3}}  {got:11.4f}  "
              f"{(want if want is not None else float('nan')):9.4f}")
    print(f"\n{len(checks)} checks recomputed from the returned CSVs, {bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
