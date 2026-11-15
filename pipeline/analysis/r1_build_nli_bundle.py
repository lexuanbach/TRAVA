#!/usr/bin/env python3
"""R1/R2(c): build the exact NLI input tuples the generators were shown.

No model is run here. The script writes a transfer bundle (JSONL) for the GPU
server labeller (r1_nli_server.py) and a manifest that fixes every tuple by
hash, so labels coming back can be checked against what was sent.

Context sets (both verified, not assumed):
  unmasked  data/processed/retrieval.json['hybrid'][q][:5]
            == generations_qwen32b/hybrid.jsonl doc_ids (all 878 questions)
  masked    data/processed/retrieval_masked.json['hybrid'][q][:5]
            == generations_<m>/hybrid_masked.jsonl doc_ids for every generator
               in masked_counterfactual.PANEL (all 878 questions)
Rendered passage text = corpus text[:600] (DOC_CHARS=600 in every generator).
Human-pack passages (Task B, masked pack) are checked for coverage; any not in
these sets are added as pack_only tuples.

Hypotheses (question quoted exactly as nli_support.py does):
  undated  The answer to "<question>" is <gold>.
  dated    As of <Month D, YYYY>, the answer to "<question>" is <gold>.
           date = questions.jsonl 'date' (the date the temporal mask uses)

TOKENIZATION IS DONE HERE, LOCALLY (tokenizer only, no model forward pass).
The stored labels (results/nli_support_labels.jsonl) and the e0 600-char labels
reproduce exactly with the local transformers 5.x DeBERTa-v2 tokenizer. The GPU
server's transformers 4.57.1 fast tokenizer normalises some characters
differently (e.g. U+2026 '...' -> three '.' tokens instead of [UNK]); on a
300-pair check it changed 1 label and moved probabilities by up to 0.14. To use
exactly the tokenizer call of nli_support.py / e0_nli_visible.py,
  tok(premise, hypothesis, truncation=True, max_length=384)
is run here and the ids are shipped; the server only runs the forward pass.

Jobs in the bundle:
  repro1500    300 stored (qid, doc) pairs, premise text[:1500], undated: checks
               checkpoint/tokenizer/device equivalence with the stored MPS labels
  undated600   every unique (qid, doc) in unmasked U masked U packs
  dated600     same tuples, dated hypothesis

Writes  <bundle dir>/nli_jobs.jsonl  and  results/analysis/r1_raw/bundle_manifest.json
"""
from __future__ import annotations

import argparse
import collections
import os
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
RES = ROOT / "results"
OUT = RES / "analysis"
RAW = OUT / "r1_raw"
PACK_B = ROOT / "annotation/pack_2026-08-02"
PACK_M = ROOT / "annotation/masked_support_2026-08-04"
GEN_DOC_CHARS = 600
OLD_PREMISE_CHARS = 1500
NLI_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MAX_LENGTH = 384
PANEL = ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
         "mistral_large", "", "sonnet", "gpt4omini"]          # "" is Haiku
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_questions():
    return {d["qid"]: d for d in read_jsonl(PROC / "questions.jsonl")}


def load_corpus():
    out = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    out.setdefault(d["doc_id"], d.get("text", ""))
    return out


def hyp_undated(q):
    return f"The answer to \"{q['question']}\" is {q['gold_answer']}."


def human_date(iso):
    y, m, d = (int(x) for x in str(iso)[:10].split("-"))
    return f"{MONTHS[m - 1]} {d}, {y}"


def hyp_dated(q):
    return (f"As of {human_date(q['date'])}, the answer to \"{q['question']}\" "
            f"is {q['gold_answer']}.")


def gen_doc_ids(model, cond):
    f = PROC / (f"generations_{model}" if model else "generations") / f"{cond}.jsonl"
    if not f.is_file():
        return None
    return {r["qid"]: list(r.get("doc_ids") or []) for r in read_jsonl(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--n-repro", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import importlib.metadata as md
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLI_NAME)
    tok_info = {"class": type(tok).__name__,
                "transformers": md.version("transformers"),
                "tokenizers": md.version("tokenizers"),
                "call": "tok(premise, hypothesis, truncation=True, max_length=384)"}

    def encode(prem, hyp):
        enc = tok(prem, hyp, truncation=True, max_length=MAX_LENGTH)
        full = len(tok(prem, hyp, truncation=False)["input_ids"])
        return {"input_ids": enc["input_ids"],
                "token_type_ids": enc.get("token_type_ids"),
                "attention_mask": enc["attention_mask"],
                "n_pair_tokens_untruncated": full}

    qs = load_questions()
    corpus = load_corpus()
    ret_u = json.loads((PROC / "retrieval.json").read_text())["hybrid"]
    ret_m = json.loads((PROC / "retrieval_masked.json").read_text())["hybrid"]

    # ---- verify the context sets against the frozen generations ----------
    checks = {}
    g32 = gen_doc_ids("qwen32b", "hybrid")
    checks["unmasked_retrieval_top5_equals_qwen32b_hybrid_doc_ids"] = (
        f"{sum(1 for q in qs if g32.get(q) == ret_u[q][:5])}/{len(qs)}")
    per_gen_u = {}
    for m in PANEL:
        g = gen_doc_ids(m, "hybrid")
        if g is not None:
            per_gen_u[m or "haiku"] = f"{sum(1 for q in qs if g.get(q) == ret_u[q][:5])}/{len(qs)}"
    checks["unmasked_hybrid_doc_ids_equal_retrieval_top5_per_generator"] = per_gen_u
    per_gen_m = {}
    for m in PANEL:
        g = gen_doc_ids(m, "hybrid_masked")
        if g is None:
            per_gen_m[m or "haiku"] = "missing"
            continue
        per_gen_m[m or "haiku"] = (f"{sum(1 for q in qs if g.get(q) == ret_m[q][:5])}/"
                                   f"{len(qs)} (rows {len(g)})")
    checks["masked_hybrid_doc_ids_equal_retrieval_masked_top5_per_generator"] = per_gen_m

    sets = collections.defaultdict(set)      # (qid, doc) -> {"unmasked", "masked", ...}
    rank = collections.defaultdict(dict)
    for q in sorted(qs):
        for i, d in enumerate(ret_u[q][:5]):
            sets[(q, d)].add("unmasked")
            rank[(q, d)]["unmasked"] = i
        for i, d in enumerate(ret_m[q][:5]):
            sets[(q, d)].add("masked")
            rank[(q, d)]["masked"] = i

    # ---- human pack passages ------------------------------------------------
    keyb = {r["item_id"]: r for r in read_jsonl(PACK_B / "item_metadata.jsonl")}
    b_pairs = set()
    for r in csv.DictReader((PACK_B / "task_b_evidence.csv").open(encoding="utf-8")):
        k = keyb.get(r["item_id"])
        if k and k.get("qid"):
            b_pairs.add((k["qid"], r["evidence_id"]))
    m_pairs = {(r["qid"], r["doc_id"]) for r in read_jsonl(PACK_M / "item_metadata.jsonl")}
    pack_cov = {
        "task_b_pairs": len(b_pairs),
        "task_b_pairs_in_unmasked_set": sum(1 for p in b_pairs if "unmasked" in sets.get(p, ())),
        "masked_pack_pairs": len(m_pairs),
        "masked_pack_pairs_in_masked_set": sum(1 for p in m_pairs if "masked" in sets.get(p, ())),
    }
    for p in b_pairs:
        sets[p].add("task_b")
    for p in m_pairs:
        sets[p].add("masked_pack")

    # ---- jobs -----------------------------------------------------------------
    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    jobs_f = args.bundle_dir / "nli_jobs.jsonl"
    n_jobs = collections.Counter()
    empty = []
    tuples = []
    with jobs_f.open("w", encoding="utf-8") as fh:
        # repro on the ORIGINAL premise
        old = read_jsonl(RES / "nli_support_labels.jsonl")
        old_keys = sorted({(r["qid"], r["doc_id"]) for r in old})
        rep = random.Random(args.seed).sample(old_keys, min(args.n_repro, len(old_keys)))
        for q, d in rep:
            prem = (corpus.get(d) or "")[:OLD_PREMISE_CHARS]
            hyp = hyp_undated(qs[q])
            fh.write(json.dumps({"job": "repro1500", "qid": q, "doc_id": d,
                                 "premise": prem, "hypothesis": hyp,
                                 "premise_sha256": sha(prem),
                                 "hypothesis_sha256": sha(hyp),
                                 **encode(prem, hyp)}) + "\n")
            n_jobs["repro1500"] += 1
        for (q, d) in sorted(sets):
            prem = (corpus.get(d) or "")[:GEN_DOC_CHARS]
            if not prem:
                empty.append([q, d])
                continue
            tuples.append({"qid": q, "doc_id": d, "sets": sorted(sets[(q, d)]),
                           "rank": rank.get((q, d), {}), "premise_sha256": sha(prem),
                           "premise_chars": len(prem)})
            for job, hyp in (("undated600", hyp_undated(qs[q])), ("dated600", hyp_dated(qs[q]))):
                fh.write(json.dumps({"job": job, "qid": q, "doc_id": d,
                                     "premise": prem, "hypothesis": hyp,
                                     "premise_sha256": sha(prem),
                                     "hypothesis_sha256": sha(hyp),
                                     **encode(prem, hyp)}) + "\n")
                n_jobs[job] += 1

    set_counts = collections.Counter()
    for t in tuples:
        for s in t["sets"]:
            set_counts[s] += 1
    manifest = {
        "built_by": "pipeline/analysis/r1_build_nli_bundle.py",
        "seed_repro_subsample": args.seed,
        "tokenizer_local": tok_info,
        "rendered_text": f"corpus text[:{GEN_DOC_CHARS}] (generator DOC_CHARS)",
        "hypotheses": {"undated": 'The answer to "<question>" is <gold>.',
                       "dated": 'As of <Month D, YYYY>, the answer to "<question>" is <gold>. '
                                "(date = questions.jsonl 'date', the date the temporal mask uses)"},
        "context_set_checks": checks,
        "pack_coverage": pack_cov,
        "n_unique_tuples": len(tuples),
        "tuples_per_set": dict(set_counts),
        "n_tuples_in_both_unmasked_and_masked": sum(1 for t in tuples
                                                    if {"unmasked", "masked"} <= set(t["sets"])),
        "empty_premise_skipped": empty,
        "n_jobs": dict(n_jobs),
        "jobs_file": str(jobs_f),
        "jobs_file_sha256": file_sha(jobs_f),
        "input_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in (
            PROC / "questions.jsonl", PROC / "corpus.jsonl", PROC / "retrieval.json",
            PROC / "retrieval_masked.json", RES / "nli_support_labels.jsonl")},
        "tuples": tuples,
    }
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "bundle_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "tuples"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
