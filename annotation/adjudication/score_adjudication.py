#!/usr/bin/env python3
"""Score the returned adjudication and measure how far it departed from majority.

The number this exists to produce is the DEPARTURE RATE: how often the
adjudicator, who could not see the vote split, chose against the majority. If it
is near zero, majority vote was an adequate substitute and the paper can say so
with evidence instead of a hedge. If it is high, every estimate built on
majority labels moves and must be recomputed.

Also reports the off-ballot rate (all three annotators wrong) and the
unjudgeable rate, both of which are statements about the items rather than the
annotators.

Outputs results/adjudication.json and, when adjudicated labels exist, an
authoritative label file the final scoring stage prefers over majority vote.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = Path(__file__).resolve().parent / "pack"
DEST_PACK = ROOT / "annotation" / "pack_2026-08-02"
UNJUDGEABLE = {"UNJUDGEABLE", "UNCLEAR", ""}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, default=PACK)
    args = ap.parse_args()

    votes = {}
    kf = args.pack / "adjudication_metadata.jsonl"
    if not kf.is_file():
        print(f"missing {kf}; run build_adjudication_pack.py")
        return 1
    for line in kf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            votes[(r["field"], r["item_id"], r["evidence_id"])] = r["votes"]

    out = {
        "scope": "Third-party adjudication of the items three annotators "
                 "disagreed on, for the fields that passed reliability.",
        "blind_to_vote_counts": True,
        "excluded": {"freshness": "alpha 0.47; needs a re-pilot, not a referee"},
        "fields": {},
    }
    any_labels = False
    for field in ("verdict", "supports"):
        f = args.pack / f"adjudicate_{field}.csv"
        if not f.is_file():
            continue
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        done = [r for r in rows if (r.get("adjudicated") or "").strip()]
        if not done:
            print(f"{field:9s} 0/{len(rows)} adjudicated yet")
            out["fields"][field] = {"n_contested": len(rows), "n_adjudicated": 0}
            continue
        any_labels = True
        agree = off = unj = 0
        no_reason = []
        for r in done:
            lab = r["adjudicated"].strip().upper()
            k = (field, r["item_id"], r.get("evidence_id", ""))
            v = votes.get(k, {})
            cnt = collections.Counter(v.values())
            maj = cnt.most_common(1)[0][0] if cnt else None
            cands = {c.strip().upper() for c in r["candidates"].split("|")}
            if lab in UNJUDGEABLE:
                unj += 1
            elif lab not in cands:
                off += 1
            elif lab == maj:
                agree += 1
            if not (r.get("reason") or "").strip():
                no_reason.append(r["item_id"])
        n = len(done)
        # A departure rate means nothing without the rate you would see if the
        # adjudicator picked among the candidates at random. On 2-1 splits with
        # two candidates that baseline is 0.5, so a "53% departure" is NOT the
        # adjudicator overturning half the majority -- it is no measurable
        # agreement with the majority either way.
        chance = sum(1 - 1 / max(len({c.strip().upper()
                                      for c in r["candidates"].split("|")}), 1)
                     for r in done) / n
        # direction is a separate question from magnitude: switches concentrated
        # on one side of a boundary mean the adjudicator used a different
        # threshold, which IS systematic even when the rate looks like chance
        switches = collections.Counter()
        for r in done:
            lab = r["adjudicated"].strip().upper()
            v = votes.get((field, r["item_id"], r.get("evidence_id", "")), {})
            c = collections.Counter(v.values())
            m = c.most_common(1)[0][0] if c else None
            if m and lab != m and lab not in UNJUDGEABLE:
                switches[f"{m}->{lab}"] += 1
        rec = {
            "n_contested": len(rows), "n_adjudicated": n,
            "agreed_with_majority": agree,
            "departed_from_majority": n - agree - off - unj,
            "departure_rate": round(1 - agree / n, 4) if n else None,
            "off_ballot": off,
            "off_ballot_rate": round(off / n, 4) if n else None,
            "unjudgeable": unj,
            "unjudgeable_rate": round(unj / n, 4) if n else None,
            "rows_missing_a_reason": no_reason[:20],
            "departure_rate_if_choosing_at_random": round(chance, 4),
            "departure_vs_chance": round((1 - agree / n) - chance, 4),
            "switches": dict(switches.most_common(8)),
        }
        d = rec["departure_rate"] - chance
        rec["reading"] = (
            f"Departure {rec['departure_rate']:.3f} against a random-choice "
            f"baseline of {chance:.3f}. "
            + ("Below the baseline: the adjudicator agreed with the majority "
               "more often than chance, so majority vote carried real signal "
               "on the contested items." if d < -0.05 else
               "Indistinguishable from the baseline: on these items the "
               "adjudicator's verdict is uncorrelated with the majority, so "
               "majority vote was effectively arbitrary here -- which does not "
               "by itself mean the population estimates move, because contested "
               "items are a small and low-weight slice." if abs(d) <= 0.05 else
               "Above the baseline: the adjudicator systematically disagreed "
               "with the majority, which needs explaining before the labels "
               "are used.")
            + (f" The switches are directional ({max(switches, key=switches.get)} "
               f"x{max(switches.values())}), so the adjudicator applied a "
               f"different threshold rather than choosing noisily."
               if switches and max(switches.values()) >= 3 * min(
                   [v for v in switches.values()] or [1]) else ""))
        out["fields"][field] = rec
        print(f"{field:9s} {n}/{len(rows)} adjudicated | departure "
              f"{rec['departure_rate']} | off-ballot {rec['off_ballot']} | "
              f"unjudgeable {rec['unjudgeable']}")
        if no_reason:
            print(f"          WARNING {len(no_reason)} rows have no reason given")

        # authoritative labels: adjudicated where present, majority elsewhere
        if field == "verdict":
            dest = DEST_PACK / "task_a_correctness__adjudicated.csv"
            adj = {r["item_id"]: r["adjudicated"].strip().upper() for r in done
                   if r["adjudicated"].strip().upper() not in UNJUDGEABLE}
            with dest.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["item_id", "adjudicated"])
                w.writeheader()
                for k, v in sorted(adj.items()):
                    w.writerow({"item_id": k, "adjudicated": v})
            print(f"          wrote {dest.name} ({len(adj)} rows); "
                  f"score_annotations.py --stage final will now prefer it")

    dest = ROOT / "results" / "adjudication.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    if not any_labels:
        print("\nnothing adjudicated yet; send the pack out first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
