#!/usr/bin/env python3
"""Validate the automatic support labels against the human ones.

The paper's strongest surviving claim conditions on whether retrieved evidence
supports the answer, using two automatic labels: exact-string containment and
NLI entailment. It has been saying those labels are unvalidated -- while three
annotators plus a blind adjudicator had already labelled 900 evidence excerpts
for exactly that property (alpha 0.755). This is the
most relevant unused evidence in the artifact.

WHAT IS AND IS NOT COMPARABLE. Task B shows annotators up to three of the top
five passages, so a question labelled unsupported by a human may still have had
a supporting passage the annotator never saw. The comparison is therefore made
at the PASSAGE level, where both labels describe the same text, and only then
aggregated to the question level over the passages the human actually saw. The
automatic label is recomputed on that same restricted passage set so the two
sides describe the same evidence.

Outputs results/support_label_validation.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.dont_write_bytecode = True
from metrics import gold_in_text                                 # noqa: E402
sys.path.insert(0, str(ROOT / "annotation"))
from score_annotations import cohens_kappa, majority             # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
PACK = ROOT / "annotation" / "pack_2026-08-02"


def quiz_of(q):
    return q.split("_")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--output", type=Path,
                    default=RES / "support_label_validation.json")
    args = ap.parse_args()

    qs = {d["qid"]: d for d in
          (json.loads(l) for l in
           (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}
    corpus = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    corpus.setdefault(d["doc_id"], d.get("text", ""))
    nli = {}
    f = RES / "nli_support_labels.jsonl"
    if f.is_file():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                nli[(r["qid"], r["doc_id"])] = r["nli_label"].startswith("entail")

    # human labels: majority of the raters, then the adjudicator where present
    per = {}
    for fp in sorted(glob.glob(str(PACK / "task_b_evidence__annotator?.csv"))):
        w = Path(fp).name.split("__annotator")[1][0]
        for r in csv.DictReader(open(fp, encoding="utf-8-sig")):
            per.setdefault((r["item_id"], r["evidence_id"]), {})[w] = \
                r["supports"].strip().upper()
    who = sorted({w for v in per.values() for w in v})
    human = {}
    for k, v in per.items():
        lab, ok = majority([v[w] for w in who if w in v])
        if ok:
            human[k] = lab
    adj = ROOT / "annotation/adjudication/pack/adjudicate_supports.csv"
    n_adj = 0
    if adj.is_file():
        for r in csv.DictReader(adj.open(encoding="utf-8-sig")):
            lab = (r.get("adjudicated") or "").strip().upper()
            if lab in ("YES", "PARTIAL", "NO"):
                human[(r["item_id"], r["evidence_id"])] = lab
                n_adj += 1

    # map blinded item ids back to qids via the key
    key = {r["item_id"]: r for r in
           (json.loads(l) for l in
            (PACK / "item_metadata.jsonl").read_text().splitlines() if l.strip())}

    rows = []
    for (iid, did), h in human.items():
        k = key.get(iid)
        if not k or not k.get("qid"):
            continue
        qid = k["qid"]
        q = qs.get(qid)
        if not q or did not in corpus:
            continue
        rows.append({
            "qid": qid, "doc_id": did, "quiz": quiz_of(qid),
            "human": h,
            "human_bin": h in ("YES",),
            "human_bin_loose": h in ("YES", "PARTIAL"),
            "string": bool(gold_in_text(q["gold_answer"], corpus[did])),
            "nli": nli.get((qid, did)),
        })
    if not rows:
        print("no overlapping passages; is the KEY missing qid?", file=sys.stderr)

    out = {
        "scope": "Automatic support labels against human ones, at the passage "
                 "level where both describe the same text.",
        "n_passages": len(rows),
        "n_questions": len({r["qid"] for r in rows}),
        "human_label_source": (f"majority of {len(who)} annotators, "
                               f"{n_adj} contested passages replaced by the "
                               f"blind adjudicator"),
        "caveat": ("Task B showed up to three of the top five passages, so this "
                   "validates the labels on the passages a human saw, not the "
                   "question-level support variable used in the analysis."),
        "labels": {},
    }

    def boot_rate(sel, fn, B, seed):
        by = collections.defaultdict(list)
        for r in sel:
            by[r["quiz"]].append(1.0 if fn(r) else 0.0)
        cl = list(by.values())
        if len(cl) < 3:
            return None
        flat = [v for c in cl for v in c]
        rng = random.Random(seed)
        draws = []
        for _ in range(B):
            pool = []
            for _ in cl:
                pool.extend(cl[rng.randrange(len(cl))])
            draws.append(statistics.fmean(pool))
        draws.sort()
        n = len(draws)
        return {"rate": round(statistics.fmean(flat), 4),
                "ci95": [round(draws[int(0.025 * n)], 4),
                         round(draws[int(0.975 * n)], 4)]}

    for auto in ("string", "nli"):
        sel = [r for r in rows if r[auto] is not None]
        if len(sel) < 20:
            continue
        for strict, name in ((True, "human_YES_only"), (False, "human_YES_or_PARTIAL")):
            hb = "human_bin" if strict else "human_bin_loose"
            a = [("Y" if r[auto] else "N") for r in sel]
            h = [("Y" if r[hb] else "N") for r in sel]
            tab = collections.Counter(zip(a, h))
            tp, fp = tab[("Y", "Y")], tab[("Y", "N")]
            fn_, tn = tab[("N", "Y")], tab[("N", "N")]
            out["labels"].setdefault(auto, {})[name] = {
                "n": len(sel),
                "agreement": boot_rate(sel, lambda r, x=auto, y=hb: r[x] == r[y],
                                       args.B, args.seed),
                "cohens_kappa": round(cohens_kappa(a, h), 4),
                "precision": round(tp / max(tp + fp, 1), 4),
                "recall": round(tp / max(tp + fn_, 1), 4),
                "confusion": {"auto_yes_human_yes": tp, "auto_yes_human_no": fp,
                              "auto_no_human_yes": fn_, "auto_no_human_no": tn},
            }

    print(f"{len(rows)} passages, {out['n_questions']} questions "
          f"({out['human_label_source']})")
    print(f"{'auto':7s}{'human def':22s}{'n':>6s}{'agree':>8s}{'kappa':>8s}"
          f"{'prec':>7s}{'rec':>7s}")
    for auto, d in out["labels"].items():
        for name, r in d.items():
            print(f"{auto:7s}{name:22s}{r['n']:6d}"
                  f"{r['agreement']['rate']:8.3f}{r['cohens_kappa']:8.3f}"
                  f"{r['precision']:7.3f}{r['recall']:7.3f}")
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
