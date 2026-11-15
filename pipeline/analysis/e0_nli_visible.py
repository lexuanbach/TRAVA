#!/usr/bin/env python3
"""E0 task 2: NLI support labels with the premise the generator actually saw.

pipeline/nli_support.py labelled every (qid, doc_id) in Qwen2.5-32B's Hybrid
contexts (878 questions x 5 = 4,390 contexts) with
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, premise =
text[:1500] and then tokenizer truncation at max_length=384, hypothesis =
'The answer to "<question>" is <gold>.'. Generators see text[:600].

This script
  1. checks that the stored label file covers exactly the sample nli_support.py
     defines, and reproduces stored labels on a random subsample with the
     ORIGINAL premise (text[:1500], same tokenizer call);
  2. times 50 contexts with premise = text[:600] and extrapolates; if the full
     run would exceed --max-hours it relabels a documented random subsample of
     QUESTIONS (all contexts of each sampled question), otherwise all of them;
  3. writes results/analysis/nli_support_labels_600.jsonl (resumable);
  4. reruns pipeline/support_interaction.py with its label file swapped for the
     600-char one (module RES patched in memory) and reports the 'nli'
     interaction old vs new, after first reproducing the stored old values.

Outputs results/analysis/e0_nli_visible.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse                                                   # noqa: E402
import collections                                                # noqa: E402
import json                                                       # noqa: E402
import random                                                     # noqa: E402
import time                                                       # noqa: E402

from _e0util import (GEN_DOC_CHARS, OUT, RAW, RES, dump, load_json, meta,  # noqa: E402
                     repro_record, run_main, window_fn)

import nli_support as NS                                          # noqa: E402
import support_interaction as SI                                  # noqa: E402

OLD_LABELS = RES / "nli_support_labels.jsonl"
NEW_LABELS = OUT / "nli_support_labels_600.jsonl"
OLD_PREMISE_CHARS = 1500      # nli_support.py
MAX_LENGTH = 384              # nli_support.py tokenizer call
FULL, VIS = window_fn(None), window_fn(GEN_DOC_CHARS)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


class Labeller:
    """The exact model/tokenizer call of nli_support.py, premise window as arg."""

    def __init__(self, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = device if (device != "mps" or torch.backends.mps.is_available()) else "cpu"
        self.tok = AutoTokenizer.from_pretrained(NS.NLI_NAME)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(NS.NLI_NAME).to(self.dev).eval()
        self.id2 = {int(k): str(v).lower() for k, v in (self.mdl.config.id2label or {}).items()}

    def __call__(self, premise, q):
        hyp = f"The answer to \"{q['question']}\" is {q['gold_answer']}."
        with self.torch.no_grad():
            enc = self.tok(premise, hyp, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH).to(self.dev)
            probs = self.torch.softmax(self.mdl(**enc).logits[0], -1).tolist()
        lab = {self.id2.get(i, str(i)): round(p, 4) for i, p in enumerate(probs)}
        return max(lab, key=lab.get), lab

    def untruncated_len(self, premise, q):
        hyp = f"The answer to \"{q['question']}\" is {q['gold_answer']}."
        return len(self.tok(premise, hyp, truncation=False)["input_ids"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-repro", type=int, default=300,
                    help="stored contexts relabelled with the ORIGINAL premise")
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    t_start = time.time()
    RAW.mkdir(parents=True, exist_ok=True)

    qs = NS.load_questions()
    corpus = NS.load_corpus()
    old = read_jsonl(OLD_LABELS)
    old_by = {(r["qid"], r["doc_id"]): r for r in old}

    # ---- 1a. the sample nli_support.py defines -----------------------------
    gen = NS.load_gen("qwen32b", "hybrid")
    qids = sorted(q for q in gen if q in qs)
    if len(qids) > 900:
        qids = sorted(random.Random(20260802).sample(qids, 900))
    expected = [(q, d) for q in qids for d in (gen[q].get("doc_ids") or [])]
    sample_check = {
        "n_stored_contexts": len(old), "n_unique_stored": len(old_by),
        "n_expected_from_nli_support_py": len(expected),
        "stored_equals_expected_set": set(old_by) == set(expected),
        "n_questions": len({q for q, _ in old_by}),
    }

    lab = Labeller(args.device)
    info = {"device": lab.dev, "model": NS.NLI_NAME, "id2label": lab.id2,
            "max_length": MAX_LENGTH}

    # ---- 1b. old-premise truncation audit + reproduction --------------------
    t0 = time.time()
    lens_old = [lab.untruncated_len((corpus.get(d) or "")[:OLD_PREMISE_CHARS], qs[q])
                for q, d in expected]
    lens_new = [lab.untruncated_len((corpus.get(d) or "")[:GEN_DOC_CHARS], qs[q])
                for q, d in expected]
    trunc = {
        "old_premise_1500_pairs_over_384_tokens": sum(1 for x in lens_old if x > MAX_LENGTH),
        "new_premise_600_pairs_over_384_tokens": sum(1 for x in lens_new if x > MAX_LENGTH),
        "n_contexts": len(expected),
        "old_median_pair_tokens": sorted(lens_old)[len(lens_old) // 2],
        "new_max_pair_tokens": max(lens_new),
        "note": ("pairs over 384 tokens had the premise truncated longest-first by "
                 "the tokenizer; every other old pair used the full text[:1500]. "
                 "No 600-char pair reaches the cap."),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(trunc), flush=True)

    rng = random.Random(args.seed)
    rep_keys = rng.sample(sorted(old_by), min(args.n_repro, len(old_by)))
    t0 = time.time()
    agree, maxdiff, diffs = 0, 0.0, []
    for q, d in rep_keys:
        prem = (corpus.get(d) or "")[:OLD_PREMISE_CHARS]
        best, probs = lab(prem, qs[q])
        s = old_by[(q, d)]
        agree += best == s["nli_label"]
        md = max(abs(probs[k] - s["probs"][k]) for k in probs)
        maxdiff = max(maxdiff, md)
        diffs.append(md)
    repro_labels = {
        "n": len(rep_keys), "label_agreement": agree,
        "label_agreement_rate": round(agree / len(rep_keys), 4),
        "max_abs_prob_diff": round(maxdiff, 4),
        "n_prob_diff_over_0.01": sum(1 for x in diffs if x > 0.01),
        "seconds_per_context_old_premise": round((time.time() - t0) / len(rep_keys), 4),
        "seed": args.seed,
    }
    print("repro:", repro_labels, flush=True)

    # stored context-level agreement vs results/nli_support.json
    tab = collections.Counter((r["string_support"], r["nli_label"].startswith("entail"))
                              for r in old)
    stored_ns = load_json(RES / "nli_support.json")["context_level_agreement"]
    rec_ctx = {"both_support": tab[(True, True)], "string_only": tab[(True, False)],
               "nli_only": tab[(False, True)], "neither": tab[(False, False)]}
    ctx_repro = repro_record({k: stored_ns[k] for k in rec_ctx}, rec_ctx,
                             "nli_support.json context_level_agreement counts from the stored label file")
    str_field_ok = sum(1 for r in old if r["string_support"] == FULL(qs[r["qid"]]["gold_answer"],
                                                                     corpus.get(r["doc_id"], "")))

    # ---- 2. timing on 50 contexts, then decide full run vs subsample --------
    done = {}
    if NEW_LABELS.exists():
        for r in read_jsonl(NEW_LABELS):
            done[(r["qid"], r["doc_id"])] = r
    todo = [k for k in expected if k not in done]
    plan = {"resumed_from_existing": len(done)}
    NEW_LABELS.parent.mkdir(parents=True, exist_ok=True)

    def label_and_write(keys, fh):
        for i, (q, d) in enumerate(keys):
            text = corpus.get(d) or ""
            prem = text[:GEN_DOC_CHARS]
            if not prem:
                continue
            best, probs = lab(prem, qs[q])
            rec = {"qid": q, "doc_id": d, "nli_label": best, "probs": probs,
                   "premise_chars": GEN_DOC_CHARS,
                   "string_support": bool(VIS(qs[q]["gold_answer"], text)),
                   "string_support_fulltext": bool(FULL(qs[q]["gold_answer"], text))}
            done[(q, d)] = rec
            fh.write(json.dumps(rec) + "\n")
            if i % 200 == 0:
                fh.flush()
                print(f"  {i}/{len(keys)}", flush=True)

    with NEW_LABELS.open("a", encoding="utf-8") as fh:
        t0 = time.time()
        pilot = todo[:50]
        label_and_write(pilot, fh)
        per = (time.time() - t0) / max(len(pilot), 1)
        rest = [k for k in expected if k not in done]
        est_h = per * len(rest) / 3600
        plan.update({"pilot_contexts": len(pilot), "seconds_per_context": round(per, 4),
                     "estimated_hours_for_remaining": round(est_h, 3),
                     "max_hours": args.max_hours})
        if est_h <= args.max_hours:
            plan["mode"] = "full: every context of the stored sample relabelled"
            label_and_write(rest, fh)
        else:
            budget = int(args.max_hours * 3600 / per / 5)
            sub_q = set(random.Random(args.seed).sample(qids, min(budget, len(qids))))
            plan["mode"] = (f"SUBSAMPLE: {len(sub_q)} of {len(qids)} questions drawn at "
                            f"random (seed {args.seed}), all 5 contexts each")
            label_and_write([k for k in rest if k[0] in sub_q], fh)
        plan["labelling_seconds"] = round(time.time() - t0, 1)
    new = read_jsonl(NEW_LABELS)
    new_by = {(r["qid"], r["doc_id"]): r for r in new}
    plan["n_contexts_labelled_600"] = len(new_by)
    plan_file = RAW / "nli600_run_plan.json"
    if plan.get("pilot_contexts"):
        dump(plan, plan_file)                 # keep the timing of the run that labelled
    elif plan_file.exists():
        plan = {**load_json(plan_file), "this_invocation": plan}

    # ---- 3. interaction: old reproduced, then new --------------------------
    f_old = RAW / "support_interaction__nli_stored_labels.json"
    f_new = RAW / "support_interaction__nli600_labels.json"
    resdir = RAW / "nli600_resdir"
    resdir.mkdir(parents=True, exist_ok=True)
    link = resdir / "nli_support_labels.jsonl"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(NEW_LABELS, resdir))
    _, rt_old = run_main(SI, ["-B", args.B, "--output", f_old], RAW / "support_interaction__nli_stored_labels.log")
    _, rt_new = run_main(SI, ["-B", args.B, "--output", f_new], RAW / "support_interaction__nli600_labels.log",
                         RES=resdir)
    stored_si = load_json(RES / "support_interaction.json")
    o, n = load_json(f_old), load_json(f_new)
    si_repro = repro_record({"nli": stored_si["labels"]["nli"],
                             "slope": stored_si["unsupported_cost_vs_scale"].get("nli"),
                             "summary": stored_si["summary"]["nli"]},
                            {"nli": o["labels"]["nli"],
                             "slope": o["unsupported_cost_vs_scale"].get("nli"),
                             "summary": o["summary"]["nli"]},
                            "support_interaction.json labels['nli'], nli slope and summary")
    keep = ("interaction", "ci95", "p_plus_one", "gain_supported", "gain_unsupported",
            "n_supported", "n_unsupported", "n_quizzes")
    per_model = {}
    for m in o["labels"]["nli"]:
        per_model[m] = {}
        for reg in ("static", "fresh"):
            a, b = o["labels"]["nli"][m].get(reg), n["labels"]["nli"][m].get(reg)
            rec = {"old_nli_premise1500": {k: a[k] for k in keep} if a else None,
                   "new_nli_premise600": {k: b[k] for k in keep} if b else None}
            if a and b:
                rec["interaction_change"] = round(b["interaction"] - a["interaction"], 3)
            per_model[m][reg] = rec
    # string-600 alongside, for reference (from the same new run's string arm is
    # full text, so recompute nothing: point to e0_visible_support.json)

    # ---- label transitions --------------------------------------------------
    common = sorted(set(old_by) & set(new_by))
    ctx_trans = collections.Counter((old_by[k]["nli_label"], new_by[k]["nli_label"]) for k in common)
    q_old, q_new = collections.defaultdict(bool), collections.defaultdict(bool)
    for k in common:
        q_old[k[0]] |= old_by[k]["nli_label"].startswith("entail")
        q_new[k[0]] |= new_by[k]["nli_label"].startswith("entail")
    q_trans = {"static": collections.Counter(), "fresh": collections.Counter()}
    for q in q_old:
        reg = "fresh" if qs[q]["year"] >= SI.BOUNDARY_YEAR else "static"
        q_trans[reg][("E" if q_old[q] else "N") + "->" + ("E" if q_new[q] else "N")] += 1
    tab_new = collections.Counter((new_by[k]["string_support"],
                                   new_by[k]["nli_label"].startswith("entail")) for k in common)

    def agreement(t):
        nn = sum(t.values())
        return {"both_support": t[(True, True)], "string_only": t[(True, False)],
                "nli_only": t[(False, True)], "neither": t[(False, False)],
                "raw_agreement": round((t[(True, True)] + t[(False, False)]) / nn, 4),
                "string_supported_but_not_entailed":
                    round(t[(True, False)] / max(t[(True, True)] + t[(True, False)], 1), 4),
                "entailed_but_no_string":
                    round(t[(False, True)] / max(t[(False, True)] + t[(False, False)], 1), 4)}

    out = {
        "step": "E0 task 2 -- NLI support label on generator-visible text",
        "meta": meta([OLD_LABELS, NEW_LABELS]),
        "labeller": info,
        "label_definitions": {
            "old_nli_premise1500": "nli_support.py: premise text[:1500], tokenizer truncation at 384 tokens",
            "new_nli_premise600": "premise text[:600] (what generators saw); same model, hypothesis, "
                                  "tokenizer call; question supported iff any context entails",
        },
        "sample_check": sample_check,
        "stored_string_support_field_equals_fulltext_label": f"{str_field_ok}/{len(old)}",
        "reproduction": {
            "nli_labels_original_premise_on_random_subsample": repro_labels,
            "nli_support_json_context_counts": ctx_repro,
            "support_interaction_json_nli_arm": si_repro,
        },
        "old_premise_token_truncation": trunc,
        "run_plan": plan,
        "support_interaction_nli": {
            "script": "pipeline/support_interaction.py (label 'nli')",
            "condition": "hybrid (unmasked), k=5", "seed": 20260802, "B": args.B,
            "weighting": ("QUESTION-POOLED arm means; quiz-cluster bootstrap CI "
                          "(same estimator as the string label)"),
            "per_model": per_model,
            "unsupported_cost_vs_scale": {"old_nli_premise1500": o["unsupported_cost_vs_scale"].get("nli"),
                                          "new_nli_premise600": n["unsupported_cost_vs_scale"].get("nli")},
            "summary": {"old_nli_premise1500": o["summary"]["nli"],
                        "new_nli_premise600": n["summary"]["nli"]},
            "runtimes_sec": {"old": rt_old, "new": rt_new},
        },
        "question_level_transitions_entail_any": {r: dict(c) for r, c in q_trans.items()},
        "question_level_supported_counts": {
            "old": {r: q_trans[r]["E->E"] + q_trans[r]["E->N"] for r in q_trans},
            "new": {r: q_trans[r]["E->E"] + q_trans[r]["N->E"] for r in q_trans}},
        "context_level_label_transitions_old_to_new": {f"{a}->{b}": c for (a, b), c in sorted(ctx_trans.items())},
        "context_level_agreement_with_string": {
            "old_nli1500_vs_string_fulltext": agreement(tab),
            "new_nli600_vs_string600": agreement(tab_new)},
        "known_issue_not_fixed_here": (
            "masked_counterfactual.py reuses these per-QUESTION NLI labels, computed "
            "on the UNMASKED Hybrid contexts of Qwen2.5-32B, as the support label for "
            "the MASKED-context interaction, so its 'nli' masked interaction labels "
            "different passages than the generator saw. Not recomputed here."),
        "runtime_total_sec": round(time.time() - t_start, 1),
    }
    dump(out, OUT / "e0_nli_visible.json")
    print(f"\n{'model':15s}{'reg':>7s}{'old':>9s}{'new':>9s}")
    for m, r in per_model.items():
        for reg, v in r.items():
            print(f"{m:15s}{reg:>7s}{v['old_nli_premise1500']['interaction']:9.2f}"
                  f"{v['new_nli_premise600']['interaction']:9.2f}")
    print("repro:", si_repro["reproduced_exactly"], ctx_repro["reproduced_exactly"], repro_labels)
    print(f"wrote {OUT / 'e0_nli_visible.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
