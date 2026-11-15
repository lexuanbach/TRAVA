#!/usr/bin/env python3
"""R1 coverage sweep: support-label analyses that E0 did not rerun on visible text.

While building the correction manifest, three paper analyses were found whose
support label still reads the FULL stored document (string) or the 1,500-char
NLI cache, and which e0_visible_support.py / e0_nli_visible.py did not rerun:

  pipeline/support_endpoint_robustness.py  support interaction under the
      choice-grounded endpoint (masked Hybrid, Static; paper macros CgInter*,
      sections/05_results.tex:233)
  pipeline/mask_null_mechanism.py          answer coverage of displaced vs
      substituted passages (appendix f_additional_tables.tex:510-520)
  pipeline/nli_support.py                  context-level string/NLI agreement
      (77.6%), contradiction inside string support (11.8%), gain given support

Each is first rerun unchanged and compared with its stored JSON, then rerun with
the label restricted to the 600-character prefix the generators saw
(gold_in_text patched in memory; for nli_support.py the label cache is swapped
for results/analysis/nli_support_labels_600.jsonl, whose string_support field is
the 600-char string label). No model is run: nli_support.py only labels contexts
missing from its cache, the cache is checked to be complete beforehand, and the
model name is patched to an invalid id so an accidental load fails immediately.

Outputs results/analysis/r1_remaining_string_support.json (+ r1_raw/).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from _e0util import OUT, RES, dump, load_json, meta, repro_record, run_main, window_fn  # noqa: E402

import mask_null_mechanism as MNM            # noqa: E402
import nli_support as NS                     # noqa: E402
import support_endpoint_robustness as SER    # noqa: E402

RAW = OUT / "r1_raw"
FULL, VIS = window_fn(None), window_fn(600)
E0_NLI600 = OUT / "nli_support_labels_600.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    args = ap.parse_args()
    t0 = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    rep, rts, out = {}, {}, {}

    # ---- support_endpoint_robustness.py --------------------------------------
    f_old, f_new = RAW / "support_endpoint_robustness__fulltext.json", RAW / "support_endpoint_robustness__vis600.json"
    _, rts["ser_full"] = run_main(SER, ["-B", args.B, "--output", f_old], RAW / "support_endpoint_robustness__fulltext.log")
    _, rts["ser_600"] = run_main(SER, ["-B", args.B, "--output", f_new], RAW / "support_endpoint_robustness__vis600.log",
                                 gold_in_text=VIS)
    rep["support_endpoint_robustness.py"] = repro_record(load_json(RES / "support_endpoint_robustness.json"),
                                                         load_json(f_old), "entire output JSON")
    o, n = load_json(f_old), load_json(f_new)
    per = {}
    for m in o["per_model"]:
        per[m] = {}
        for ep in ("token_f1", "choice_grounded"):
            a, b = o["per_model"][m].get(ep), n["per_model"][m].get(ep)
            per[m][ep] = {"old_fulltext": a, "new_visible600": b,
                          "change": round(b["interaction"] - a["interaction"], 3) if a and b else None}

    def cg_macros(d):
        cg = [r["choice_grounded"]["interaction"] for r in d["per_model"].values() if "choice_grounded" in r]
        su = d.get("summary") or {}
        return {"CgInterMin": f"+{min(cg):.1f}", "CgInterMax": f"+{max(cg):.1f}",
                "CgInterSig": str(su.get("n_excluding_zero_choice_grounded")), "CgInterN": str(su.get("n_models"))}
    out["support_endpoint_robustness"] = {
        "script": "pipeline/support_endpoint_robustness.py", "condition": "hybrid_masked top-5, Static",
        "seed": 20260804, "B": args.B,
        "weighting": "QUESTION-POOLED arm means; quiz-cluster bootstrap CI",
        "per_model": per, "summary": {"old_fulltext": o.get("summary"), "new_visible600": n.get("summary")},
        "paper_macros": {"old_fulltext": cg_macros(o), "new_visible600": cg_macros(n)}}

    # ---- mask_null_mechanism.py ------------------------------------------------
    f_old, f_new = RAW / "mask_null_mechanism__fulltext.json", RAW / "mask_null_mechanism__vis600.json"
    _, rts["mnm_full"] = run_main(MNM, ["-B", args.B, "--output", f_old], RAW / "mask_null_mechanism__fulltext.log")
    _, rts["mnm_600"] = run_main(MNM, ["-B", args.B, "--output", f_new], RAW / "mask_null_mechanism__vis600.log",
                                 gold_in_text=VIS)
    rep["mask_null_mechanism.py"] = repro_record(load_json(RES / "mask_null_mechanism.json"), load_json(f_old),
                                                 "entire output JSON")
    out["mask_null_mechanism"] = {"script": "pipeline/mask_null_mechanism.py",
                                  "label": "answer-bearing passage = gold string in passage",
                                  "old_fulltext": load_json(f_old), "new_visible600": load_json(f_new)}

    # ---- nli_support.py ---------------------------------------------------------
    qs = NS.load_questions()
    gen = NS.load_gen("qwen32b", "hybrid")
    qids = sorted(q for q in gen if q in qs)
    if len(qids) > 900:
        qids = sorted(random.Random(20260802).sample(qids, 900))
    expected = {(q, d) for q in qids for d in (gen[q].get("doc_ids") or [])}
    cache_ok = {}
    for tag, path in (("stored", NS.CACHE), ("vis600", E0_NLI600)):
        have = {(json.loads(l)["qid"], json.loads(l)["doc_id"]) for l in open(path, encoding="utf-8") if l.strip()}
        cache_ok[tag] = expected <= have
    if not all(cache_ok.values()):
        raise SystemExit(f"label cache incomplete {cache_ok}; refusing to run (would start NLI inference)")
    f_old, f_new = RAW / "nli_support__stored_cache.json", RAW / "nli_support__cache600.json"
    _, rts["ns_stored"] = run_main(NS, ["-B", args.B, "--device", "cpu", "--output", f_old],
                                   RAW / "nli_support__stored_cache.log", NLI_NAME="__disabled_no_model_run__")
    _, rts["ns_600"] = run_main(NS, ["-B", args.B, "--device", "cpu", "--output", f_new],
                                RAW / "nli_support__cache600.log", NLI_NAME="__disabled_no_model_run__",
                                CACHE=E0_NLI600, gold_in_text=VIS)
    o, n = load_json(f_old), load_json(f_new)
    rep["nli_support.py"] = repro_record(
        load_json(RES / "nli_support.json"), {**o, "nli_model": NS.NLI_NAME},
        "entire output JSON except the 'nli_model' string, which echoes the in-memory NLI_NAME "
        "patched to an invalid id so that no model could be loaded")
    out["nli_support"] = {
        "script": "pipeline/nli_support.py", "condition": "hybrid (unmasked) top-5 contexts, 878 questions",
        "label_windows": {"old": "string full text; NLI text[:1500]", "new": "string text[:600]; NLI text[:600]"},
        "gain_given_support_weighting": "QUESTION-POOLED binary-correct (F1>=0.5) difference; i.i.d. question bootstrap",
        "cache_complete_no_inference": cache_ok,
        "old": {k: o[k] for k in ("context_level_agreement", "nli_label_distribution",
                                  "contradiction_inside_string_support", "verdict")},
        "new": {k: n[k] for k in ("context_level_agreement", "nli_label_distribution",
                                  "contradiction_inside_string_support", "verdict")},
        "gain_given_support_by_label": {"old": o["gain_given_support_by_label"],
                                        "new": n["gain_given_support_by_label"]}}

    res = {"step": "R1 coverage sweep -- remaining full-text / 1500-char support labels rerun at 600 chars",
           "meta": meta([E0_NLI600]), "reproduction_of_stored_results": rep, "runtimes_sec": rts, **out,
           "runtime_total_sec": round(time.time() - t0, 1)}
    dump(res, OUT / "r1_remaining_string_support.json")
    print("repro:", {k: v["reproduced_exactly"] for k, v in rep.items()})
    for m, v in per.items():
        print(f"{m:15s} tokF1 {v['token_f1']['old_fulltext']['interaction']:7.2f} -> "
              f"{v['token_f1']['new_visible600']['interaction']:7.2f}   CG "
              f"{v['choice_grounded']['old_fulltext']['interaction']:7.2f} -> "
              f"{v['choice_grounded']['new_visible600']['interaction']:7.2f} "
              f"{v['choice_grounded']['new_visible600']['ci95']}")
    print(out["support_endpoint_robustness"]["paper_macros"])
    for k in ("displaced_passages", "substituted_passages"):
        print(k, out["mask_null_mechanism"]["old_fulltext"].get(k), out["mask_null_mechanism"]["new_visible600"].get(k))
    print("nli_support ctx:", out["nli_support"]["old"]["context_level_agreement"]["raw_agreement"],
          out["nli_support"]["new"]["context_level_agreement"]["raw_agreement"],
          out["nli_support"]["old"]["contradiction_inside_string_support"],
          out["nli_support"]["new"]["contradiction_inside_string_support"])
    print("wrote", OUT / "r1_remaining_string_support.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
