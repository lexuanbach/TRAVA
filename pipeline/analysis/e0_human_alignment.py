#!/usr/bin/env python3
"""E0 task 3: automatic support labels against human labels, window-matched.

Three different text windows are in play:
  generator prompt      text[:600] per passage          (DOC_CHARS=600)
  human, Task B pack    text.replace("\\n"," ")[:700]    (build_annotation_pack.py)
  human, masked pack    " ".join(text.split())[:700]    (build_masked_pack.py)
  string label          full stored text                (support_label_validation.py etc.)
  NLI label             text[:1500] + 384-token cap     (nli_support.py)

(a) support_label_validation.py rerun with the string label at windows
    {full, 1500, 700nl, 600} (and NLI at the stored 1500 vs the new 600 labels),
    after reproducing the stored JSON; the same metrics on the masked pack
    (annotation/masked_support_2026-08-04), for which no validation script exists.
(b) human_support_interaction.py and annotation/.../score_masked_support.py
    rerun and compared with the stored JSON; their automatic-label pieces
    (question-level auto/human agreement; sampling strata) recomputed under
    the 600-char window, plus the same interaction estimator applied to the
    automatic labels on the SAME human-labelled questions for comparison;
    passages whose gold string is visible to humans (700) but not to the
    generator (600) are counted.

Outputs results/analysis/e0_human_alignment.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import importlib.util
import json
import random
import statistics
import time
from pathlib import Path

from _e0util import (ANNOT, GEN_DOC_CHARS, OUT, PROC, RAW, RES, ROOT, dump,
                     load_json, meta, repro_record, run_main, window_fn)

import human_support_interaction as HSI                           # noqa: E402
import support_label_validation as SLV                            # noqa: E402
from metrics import token_f1                                      # noqa: E402
from score_annotations import cohens_kappa, majority              # noqa: E402

PACK_B = ANNOT / "pack_2026-08-02"
PACK_M = ANNOT / "masked_support_2026-08-04"
NLI600_RESDIR = RAW / "nli600_resdir"      # created by e0_nli_visible.py
NLI600 = OUT / "nli_support_labels_600.jsonl"
WINDOWS_B = {"full": None, "w1500": 1500, "w700nl": "700nl", "w600": GEN_DOC_CHARS}
WINDOWS_M = {"full": None, "w1500": 1500, "w700ws": "700ws", "w600": GEN_DOC_CHARS}


def quiz_of(q):
    return q.split("_")[0]


def load_mhs():
    spec = importlib.util.spec_from_file_location(
        "score_masked_support", PACK_M / "score_masked_support.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_corpus():
    corpus = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    corpus.setdefault(d["doc_id"], d.get("text", ""))
    return corpus


def load_questions():
    return {d["qid"]: d for d in (json.loads(l) for l in
            (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}


def nli_map(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["qid"], r["doc_id"])] = r["nli_label"].startswith("entail")
    return out


# --------------------------------------------------------------------------
# Task B rows, built exactly as support_label_validation.py builds them (same
# order, so the clustered bootstrap reproduces), with window labels attached.
# --------------------------------------------------------------------------
def packb_rows(qs, corpus):
    per = {}
    for fp in sorted(glob.glob(str(PACK_B / "task_b_evidence__annotator?.csv"))):
        w = Path(fp).name.split("__annotator")[1][0]
        for r in csv.DictReader(open(fp, encoding="utf-8-sig")):
            per.setdefault((r["item_id"], r["evidence_id"]), {})[w] = r["supports"].strip().upper()
    who = sorted({w for v in per.values() for w in v})
    human = {}
    for k, v in per.items():
        lab, ok = majority([v[w] for w in who if w in v])
        if ok:
            human[k] = lab
    adj = ANNOT / "adjudication/pack/adjudicate_supports.csv"
    if adj.is_file():
        for r in csv.DictReader(adj.open(encoding="utf-8-sig")):
            lab = (r.get("adjudicated") or "").strip().upper()
            if lab in ("YES", "PARTIAL", "NO"):
                human[(r["item_id"], r["evidence_id"])] = lab
    key = {r["item_id"]: r for r in (json.loads(l) for l in
           (PACK_B / "item_metadata.jsonl").read_text().splitlines() if l.strip())}
    excerpt = {(r["item_id"], r["evidence_id"]): r["evidence_excerpt"] for r in
               csv.DictReader((PACK_B / "task_b_evidence.csv").open(encoding="utf-8"))}
    rows = []
    for (iid, did), h in human.items():
        k = key.get(iid)
        if not k or not k.get("qid"):
            continue
        q = qs.get(k["qid"])
        if not q or did not in corpus:
            continue
        text = corpus[did]
        r = {"pack": "task_b_2026-08-02", "item_id": iid, "evidence_id": did,
             "qid": k["qid"], "doc_id": did, "quiz": quiz_of(k["qid"]), "human": h,
             "human_bin": h in ("YES",), "human_bin_loose": h in ("YES", "PARTIAL"),
             "excerpt_matches_700nl": excerpt.get((iid, did)) == text.replace("\n", " ")[:700]}
        for tag, spec in WINDOWS_B.items():
            r[f"string_{tag}"] = bool(window_fn(spec)(q["gold_answer"], text))
        r["string_excerpt_as_shown"] = bool(window_fn(None)(q["gold_answer"], excerpt.get((iid, did), "")))
        rows.append(r)
    return rows


def masked_rows(qs, corpus, mhs):
    key = mhs.load_key()
    got = mhs.load_annotations()
    adj = {}
    f = PACK_M / "disagreements_for_adjudication.csv"
    if f.is_file():
        for row in csv.DictReader(f.open(encoding="utf-8")):
            lab = (row.get("adjudicated") or "").strip().upper()
            if lab:
                adj[row["item_id"]] = lab
    excerpt = {r["item_id"]: r["evidence_excerpt"] for r in
               csv.DictReader((PACK_M / "task_support_masked.csv").open(encoding="utf-8"))}
    rows = []
    for iid, v in got.items():                      # same order as the script
        lab = adj.get(iid) or mhs.majority(list(v.values()))
        k = key.get(iid)
        if not lab or not k:
            continue
        q = qs[k["qid"]]
        text = corpus[k["doc_id"]]
        r = {"pack": "masked_support_2026-08-04", "item_id": iid, "evidence_id": k["doc_id"],
             "qid": k["qid"], "doc_id": k["doc_id"], "quiz": quiz_of(k["qid"]),
             "slice": k["slice"], "stratum": k["stratum"], "human": lab,
             "human_bin": lab in ("YES",), "human_bin_loose": lab in ("YES", "PARTIAL"),
             "key_auto_string": bool(k["auto_string"]),
             "excerpt_matches_700ws": excerpt.get(iid) == " ".join(text.split())[:700]}
        for tag, spec in WINDOWS_M.items():
            r[f"string_{tag}"] = bool(window_fn(spec)(q["gold_answer"], text))
        rows.append(r)
    return rows


def boot_rate(sel, fn, B, seed):
    """support_label_validation.py's quiz-clustered agreement bootstrap, verbatim."""
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
            "ci95": [round(draws[int(0.025 * n)], 4), round(draws[int(0.975 * n)], 4)]}


def label_metrics(rows, auto, B, seed):
    """support_label_validation.py's metric block for one automatic label."""
    sel = [r for r in rows if r.get(auto) is not None]
    out = {}
    for strict, name in ((True, "human_YES_only"), (False, "human_YES_or_PARTIAL")):
        hb = "human_bin" if strict else "human_bin_loose"
        a = [("Y" if r[auto] else "N") for r in sel]
        h = [("Y" if r[hb] else "N") for r in sel]
        tab = collections.Counter(zip(a, h))
        tp, fp = tab[("Y", "Y")], tab[("Y", "N")]
        fn_, tn = tab[("N", "Y")], tab[("N", "N")]
        out[name] = {
            "n": len(sel),
            "agreement": boot_rate(sel, lambda r, x=auto, y=hb: r[x] == r[y], B, seed),
            "cohens_kappa": round(cohens_kappa(a, h), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn_, 1), 4),
            "confusion": {"auto_yes_human_yes": tp, "auto_yes_human_no": fp,
                          "auto_no_human_yes": fn_, "auto_no_human_no": tn},
        }
    return out


def gold_only_between(rows, lo_key, hi_key):
    sel = [r for r in rows if r[hi_key] and not r[lo_key]]
    uniq = {(r["qid"], r["doc_id"]) for r in sel}
    return {"n_passage_rows": len(sel), "n_unique_qid_doc": len(uniq),
            "human_label_distribution": dict(collections.Counter(r["human"] for r in sel)),
            "n_questions": len({r["qid"] for r in sel})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    args = ap.parse_args()
    t_start = time.time()
    if not NLI600.exists():
        raise SystemExit("run e0_nli_visible.py first (needs nli_support_labels_600.jsonl)")
    qs, corpus = load_questions(), load_corpus()
    mhs = load_mhs()
    repro, runtimes = {}, {}
    out = {"step": "E0 task 3 -- human alignment of automatic support labels",
           "meta": meta([NLI600]),
           "windows": {
               "generator": "text[:600] per passage",
               "human_task_b": "text.replace('\\n',' ')[:700] (verified against every excerpt shown)",
               "human_masked_pack": "' '.join(text.split())[:700] (verified against every excerpt shown)",
               "string_full": "full stored text", "string_w1500": "text[:1500]",
               "nli_stored": "text[:1500], 384-token cap", "nli_600": "text[:600]"}}

    # ============ (a) Task B pack: support_label_validation.py ===============
    slv_out = {}
    for tag, spec in WINDOWS_B.items():
        f = RAW / f"support_label_validation__{tag}.json"
        patches = {"gold_in_text": window_fn(spec)}
        if tag == "w600":
            patches["RES"] = NLI600_RESDIR          # NLI arm uses the 600-char labels
        _, runtimes[f"slv_{tag}"] = run_main(SLV, ["-B", args.B, "--output", f],
                                             RAW / f"support_label_validation__{tag}.log", **patches)
        slv_out[tag] = load_json(f)
    repro["support_label_validation.py"] = repro_record(
        load_json(RES / "support_label_validation.json"), slv_out["full"], "entire output JSON")

    rows_b = packb_rows(qs, corpus)
    nli_old, nli_new = nli_map(RES / "nli_support_labels.jsonl"), nli_map(NLI600)
    for r in rows_b:
        r["nli_1500"] = nli_old.get((r["qid"], r["doc_id"]))
        r["nli_600"] = nli_new.get((r["qid"], r["doc_id"]))
    # my row builder must reproduce the script before it is trusted on the masked pack
    mine = {"string": label_metrics(rows_b, "string_full", args.B, 20260803),
            "nli": label_metrics(rows_b, "nli_1500", args.B, 20260803)}
    repro["support_label_validation.py__row_builder_used_for_masked_pack"] = repro_record(
        load_json(RES / "support_label_validation.json")["labels"], mine,
        "labels{} recomputed by this script's own row builder + metric code")

    task_b = {"script": "pipeline/support_label_validation.py", "seed": 20260803, "B": args.B,
              "unit": "passage rows (900 = 300 items x up to 3 passages; 254 questions)",
              "weighting": "unweighted passage-level confusion; agreement CI quiz-clustered",
              "n_passages": slv_out["full"]["n_passages"],
              "excerpt_reconstruction_check": f"{sum(r['excerpt_matches_700nl'] for r in rows_b)}/{len(rows_b)} "
                                              "excerpts equal text.replace('\\n',' ')[:700]",
              "string_label_by_window": {tag: slv_out[tag]["labels"]["string"] for tag in WINDOWS_B},
              "string_label_on_excerpt_as_shown": label_metrics(rows_b, "string_excerpt_as_shown", args.B, 20260803),
              "nli_label": {"stored_premise1500": slv_out["full"]["labels"]["nli"],
                            "premise600": slv_out["w600"]["labels"]["nli"]}}
    task_b["gold_string_visible_to_humans_not_generator"] = {
        "definition": "gold in text[:700] (human excerpt) but not in text[:600] (generator)",
        **gold_only_between(rows_b, "string_w600", "string_w700nl")}
    task_b["gold_string_beyond_human_excerpt"] = {
        "definition": "gold in full text but not in the 700-char excerpt humans saw",
        **gold_only_between(rows_b, "string_w700nl", "string_full")}
    out["a_task_b_pack"] = task_b

    # ============ (a) masked pack: no validation script exists ===============
    rows_m = masked_rows(qs, corpus, mhs)
    masked = {"source": "annotation/masked_support_2026-08-04 (3 annotators, no contested rows)",
              "note": "no validation script exists for this pack; metrics computed with "
                      "support_label_validation.py's metric code (verified above), seed 20260804 "
                      "(the pack's seed)",
              "n_passages": len(rows_m), "n_questions": len({r["qid"] for r in rows_m}),
              "key_auto_string_equals_fulltext_label": f"{sum(r['key_auto_string'] == r['string_full'] for r in rows_m)}/{len(rows_m)}",
              "excerpt_reconstruction_check": f"{sum(r['excerpt_matches_700ws'] for r in rows_m)}/{len(rows_m)} "
                                              "excerpts equal ' '.join(text.split())[:700]",
              "weighting": "unweighted passage-level (strata sampled equally; not population-weighted)",
              "string_label_by_window": {tag: label_metrics(rows_m, f"string_{tag}", args.B, 20260804)
                                         for tag in WINDOWS_M}}
    masked["gold_string_visible_to_humans_not_generator"] = {
        "definition": "gold in whitespace-collapsed text[:700] (human excerpt) but not in raw text[:600]",
        **gold_only_between(rows_m, "string_w600", "string_w700ws")}
    masked["gold_string_beyond_human_excerpt"] = {
        "definition": "gold in full text but not in the 700-char excerpt humans saw",
        **gold_only_between(rows_m, "string_w700ws", "string_full")}
    out["a_masked_pack"] = masked

    # ============ (b) human_support_interaction.py ============================
    f = RAW / "human_support_interaction__rerun.json"
    _, runtimes["hsi_rerun"] = run_main(HSI, ["-B", args.B, "--output", f], RAW / "human_support_interaction__rerun.log")
    hsi_stored, hsi_rerun = load_json(RES / "human_support_interaction.json"), load_json(f)
    repro["human_support_interaction.py"] = repro_record(hsi_stored, hsi_rerun, "entire output JSON")

    strict, loose, qweight, _, _ = HSI.human_question_labels()
    labelled = sorted(set(strict) & set(qs))
    keyb = [json.loads(l) for l in (PACK_B / "item_metadata.jsonl").read_text().splitlines() if l.strip()]
    key_auto = {}
    for d in keyb:
        if d.get("task") == "B" and "auto_supported" in d:
            key_auto.setdefault(d["qid"], False)
            key_auto[d["qid"]] |= bool(d["auto_supported"])
    hyb = HSI.load_gen("qwen32b", "hybrid")          # predictions only; need doc_ids:
    hyb_docs = {}
    for line in (PROC / "generations_qwen32b" / "hybrid.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            hyb_docs[r["qid"]] = r.get("doc_ids") or []
    del hyb
    shown = collections.defaultdict(set)             # docs humans actually saw per qid
    for r in rows_b:
        shown[r["qid"]].add(r["doc_id"])

    def q_label(qid, docs, spec):
        g = qs[qid]["gold_answer"]
        return any(window_fn(spec)(g, corpus.get(d, "")) for d in docs if corpus.get(d))

    auto_tables = {
        "auto_key_stored (full text, 5 docs)": {q: key_auto[q] for q in labelled if q in key_auto},
        "auto_full_5docs": {q: q_label(q, hyb_docs[q], None) for q in labelled},
        "auto_600_5docs": {q: q_label(q, hyb_docs[q], GEN_DOC_CHARS) for q in labelled},
        "auto_full_shown3": {q: q_label(q, sorted(shown[q]), None) for q in labelled},
        "auto_700nl_shown3": {q: q_label(q, sorted(shown[q]), "700nl") for q in labelled},
        "auto_600_shown3": {q: q_label(q, sorted(shown[q]), GEN_DOC_CHARS) for q in labelled},
    }
    agreement = {name: {"n": len(t), "rate_vs_human_strict": round(
        sum(1 for q in t if t[q] == strict.get(q, False)) / len(t), 4),
        "n_supported": sum(t.values())} for name, t in auto_tables.items()}
    tables = {"human_strict_yes": strict, "human_yes_or_partial": loose,
              **{k: v for k, v in auto_tables.items() if not k.startswith("auto_key")}}
    inter = {}
    for m in HSI.MODELS:
        cb, hy = HSI.load_gen(m, "closed_book"), HSI.load_gen(m, "hybrid")
        if not cb or not hy:
            continue
        mk = m or "haiku"
        for name, table in tables.items():
            for regime in ("static", "fresh"):
                units = collections.defaultdict(lambda: ([], []))
                for q in labelled:
                    if q not in cb or q not in hy:
                        continue
                    if (qs[q]["year"] >= HSI.BOUNDARY) != (regime == "fresh"):
                        continue
                    gain = 100.0 * (token_f1(hy[q], qs[q]["gold_answer"]) - token_f1(cb[q], qs[q]["gold_answer"]))
                    units[quiz_of(q)][0 if table.get(q) else 1].append((gain, qweight[q]))
                r = HSI.interaction_ci(units, args.B, 20260803, weighted=True)
                if r:
                    u = HSI.interaction_ci(units, args.B, 20260803, weighted=False)
                    if u:
                        r["unweighted"] = {k: u[k] for k in ("interaction", "ci95", "p_plus_one")}
                inter.setdefault(mk, {}).setdefault(name, {})[regime] = r
    # reproduction of the human arms through this code path
    repro["human_support_interaction.py__interaction_code_path"] = repro_record(
        {m or "haiku": {"strict": {g: v["strict_yes"][g] for g in ("static", "fresh")},
                        "loose": {g: v["yes_or_partial"][g] for g in ("static", "fresh")}}
         for m, v in hsi_stored["per_model"].items()},
        {m: {"strict": v["human_strict_yes"], "loose": v["human_yes_or_partial"]} for m, v in inter.items()},
        "per_model strict_yes/yes_or_partial static+fresh via this script's loop")
    out["b_human_support_interaction"] = {
        "script": "pipeline/human_support_interaction.py", "seed": 20260803, "B": args.B,
        "weighting": ("DESIGN-WEIGHTED question-pooled arm means (inverse inclusion "
                      "weight of the regime x FULL-TEXT auto-support sampling cell; "
                      "these remain the correct design weights because the sample was "
                      "drawn on the full-text label); unweighted alongside; quiz-cluster CI"),
        "automatic_label_pieces_in_the_script": (
            "only question_level_label_agreement (KEY auto_supported = full text over "
            "the model's 5 Hybrid docs) and the sampling cells/weights; the script "
            "has no automatic-label interaction arm"),
        "question_level_agreement_with_human_strict": agreement,
        "stored_question_level_label_agreement": hsi_stored.get("question_level_label_agreement"),
        "sampling_cells_that_would_change_at_600": {
            reg: sum(1 for q in labelled if key_auto.get(q) and not auto_tables["auto_600_5docs"][q]
                     and (qs[q]["year"] >= HSI.BOUNDARY) == (reg == "fresh"))
            for reg in ("static", "fresh")},
        "interaction_same_questions_by_label": inter,
    }

    # ============ (b) score_masked_support.py =================================
    f = RAW / "masked_human_support__rerun.json"
    _, runtimes["mhs_rerun"] = run_main(mhs, ["--stage", "final", "-B", args.B, "--output", f],
                                        RAW / "masked_human_support__rerun.log")
    mhs_stored, mhs_rerun = load_json(RES / "masked_human_support.json"), load_json(f)
    repro["score_masked_support.py"] = repro_record(mhs_stored, mhs_rerun, "entire output JSON")

    # same q_sup order as the script
    q_sup, q_slice, q_stratum, q_docs = {}, {}, {}, collections.defaultdict(list)
    key_m = mhs.load_key()
    for r in rows_m:
        q_sup[r["qid"]] = q_sup.get(r["qid"], False) or r["human_bin"]
        q_slice[r["qid"]], q_stratum[r["qid"]] = r["slice"], r["stratum"]
    for iid, k in key_m.items():
        q_docs[k["qid"]].append(k["doc_id"])
    auto_m = {
        "auto_full_shown3": {q: q_label(q, q_docs[q], None) for q in q_sup},
        "auto_700ws_shown3": {q: q_label(q, q_docs[q], "700ws") for q in q_sup},
        "auto_600_shown3": {q: q_label(q, q_docs[q], GEN_DOC_CHARS) for q in q_sup},
    }
    strat_check = sum(1 for q in q_sup if q_stratum[q] != "consistency_recheck"
                      and (q_stratum[q].endswith("_supported")) == auto_m["auto_full_shown3"][q])
    n_strat = sum(1 for q in q_sup if q_stratum[q] != "consistency_recheck")
    switches = collections.Counter()
    for q in q_sup:
        if q_stratum[q] == "consistency_recheck":
            continue
        if q_stratum[q].endswith("_supported") and not auto_m["auto_600_shown3"][q]:
            switches[f"{q_slice[q]}_supported->unsupported"] += 1
        if q_stratum[q].endswith("_unsupported") and auto_m["auto_600_shown3"][q]:
            switches[f"{q_slice[q]}_unsupported->supported"] += 1

    def gens(model, cond):
        p = PROC / (f"generations_{model}" if model else "generations") / f"{cond}.jsonl"
        if not p.is_file():
            return {}
        return {r["qid"]: (r["prediction"] or "") for r in
                (json.loads(l) for l in p.read_text().splitlines() if l.strip())}

    inter_m = {}
    for m in ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
              "mistral_large", "", "sonnet", "gpt4omini"]:
        cb, ma = gens(m, "closed_book"), gens(m, "hybrid_masked")
        if not cb or not ma:
            continue
        for name, table in (("human_strict_yes", q_sup), *auto_m.items()):
            for regime in ("static", "fresh"):
                units = collections.defaultdict(lambda: ([], []))
                for qid in q_sup:
                    if q_slice.get(qid) != regime or qid not in cb or qid not in ma:
                        continue
                    gold = qs[qid]["gold_answer"]
                    gain = 100.0 * (token_f1(ma[qid], gold) - token_f1(cb[qid], gold))
                    units[quiz_of(qid)][0 if table[qid] else 1].append(gain)
                r = mhs.clustered_interaction(units, args.B, 20260804)
                if r:
                    inter_m.setdefault(m or "haiku", {}).setdefault(name, {})[regime] = r
    repro["score_masked_support.py__interaction_code_path"] = repro_record(
        mhs_stored["per_model"], {m: v["human_strict_yes"] for m, v in inter_m.items()},
        "per_model static/fresh human interaction via this script's loop")
    out["b_masked_human_support"] = {
        "script": "annotation/masked_support_2026-08-04/score_masked_support.py --stage final",
        "seed": 20260804, "B": args.B,
        "weighting": ("interaction: UNWEIGHTED question-pooled arm means, quiz-cluster CI "
                      "(the stratum weights are used only for weighted_support_rate)"),
        "automatic_label_pieces_in_the_script": (
            "none in the estimator; the automatic full-text string label defined the "
            "four sampling strata (first 3 masked Hybrid passages) and hence the weights"),
        "stratum_label_reproduces_full_text_label": f"{strat_check}/{n_strat}",
        "strata_that_would_change_at_600": dict(switches),
        "question_level_agreement_with_human_strict": {
            name: {"n": len(t), "rate": round(sum(1 for q in t if t[q] == q_sup[q]) / len(t), 4),
                   "n_supported": sum(t.values())} for name, t in auto_m.items()},
        "interaction_same_questions_by_label": inter_m,
    }

    out["reproduction_of_stored_results"] = repro
    out["runtimes_sec"] = {**runtimes, "total": round(time.time() - t_start, 1)}
    out["window_mismatch_note"] = (
        "Human annotators saw 700 characters per passage while generators saw 600. "
        "Passages whose gold string first appears in characters 600-700 are "
        "counted under gold_string_visible_to_humans_not_generator; for those, a "
        "human YES describes text the generator never received.")
    dump(out, OUT / "e0_human_alignment.json")

    print("reproduction:", {k: v["reproduced_exactly"] for k, v in repro.items()})
    for tag, d in task_b["string_label_by_window"].items():
        s = d["human_YES_only"]
        print(f"taskB string {tag:6s} prec {s['precision']:.3f} rec {s['recall']:.3f} "
              f"kappa {s['cohens_kappa']:.3f} {s['confusion']}")
    for tag, d in task_b["nli_label"].items():
        s = d["human_YES_only"]
        print(f"taskB nli {tag:18s} prec {s['precision']:.3f} rec {s['recall']:.3f} kappa {s['cohens_kappa']:.3f}")
    for tag, d in masked["string_label_by_window"].items():
        s = d["human_YES_only"]
        print(f"masked string {tag:6s} prec {s['precision']:.3f} rec {s['recall']:.3f} "
              f"kappa {s['cohens_kappa']:.3f} {s['confusion']}")
    print("600-700 only:", task_b["gold_string_visible_to_humans_not_generator"],
          masked["gold_string_visible_to_humans_not_generator"])
    print(f"wrote {OUT / 'e0_human_alignment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
