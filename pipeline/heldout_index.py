#!/usr/bin/env python3
"""Held-out / time-consistent index condition.

The main results pool every question's own evidence into the index, so a gold
document exists for almost every question (hit@5 ~= 0.8 on both slices). That
measures retrieval rescue in the BEST case (relevant evidence is retrievable).
This script measures the realistic DEPLOYMENT lower bound: an index that has not
yet caught up to fresh events. We build a STALE index containing only documents
published before the effective cutoff, re-retrieve, and re-generate with the
same Claude generator and the exact main-pipeline prompt.

Expectation: Static questions (evidence predates the cutoff) are ~unchanged;
Fresh questions lose their contemporaneous evidence, so hybrid Fresh F1 should
collapse toward the closed-book Fresh level -- confirming the main numbers are
an upper bound and that retrieval's rescue is conditional on a fresh index.

Phases (default all):
  retrieve : build stale corpus, run bm25+bge+hybrid, write retrieval_stale.json
  generate : Claude answers for *_stale conditions (cached, temp 0)
  aggregate: Static/Fresh F1 with CIs + unpaired significance -> results/heldout_index.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import threading
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml

import metrics as M

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
GEN = PROC / "generations"
RES = ROOT / "results"
DOC_CHARS = 600
_TOK = re.compile(r"[A-Za-z0-9]+")
CUTOFF = (2025, 1, 1)          # effective-cutoff anchor (matches main analysis)


def tok(s):
    return _TOK.findall(s.lower())


def parse(s):
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)


def is_stale(pubdate, cutoff=CUTOFF):
    try:
        return parse(pubdate) < date(*cutoff)
    except Exception:
        return False


def load():
    questions = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()]
    corpus = [json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines()]
    qrels = json.loads((PROC / "qrels.json").read_text())
    return questions, corpus, qrels


# ------------------------------ retrieval ---------------------------------
def topk_ids(scores, ids, k):
    k = min(k, len(ids))
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [ids[i] for i in idx]


def retrieve_phase(questions, corpus, qrels, cfg, k):
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    stale = [d for d in corpus if is_stale(d.get("publish_date", ""))]
    ids = [d["doc_id"] for d in stale]
    print(f"stale index: {len(stale)}/{len(corpus)} docs (published < {CUTOFF[0]})", flush=True)

    # BM25
    c = cfg["bm25"]
    bm = BM25Okapi([tok(d["text"]) for d in stale], k1=c["k1"], b=c["b"])
    bm25_ranked, bm25_scores = {}, {}
    for q in questions:
        s = np.asarray(bm.get_scores(tok(q["question"])))
        bm25_ranked[q["qid"]] = topk_ids(s, ids, k)
        bm25_scores[q["qid"]] = s

    # BGE
    bc = cfg["bge"]
    m = SentenceTransformer(bc["model"])
    demb = m.encode([d["text"] for d in stale], batch_size=64,
                    normalize_embeddings=True, show_progress_bar=False)
    qemb = m.encode([bc["query_prefix"] + q["question"] for q in questions], batch_size=64,
                    normalize_embeddings=True, show_progress_bar=False)
    sims = qemb @ demb.T
    bge_ranked = {}
    for i, q in enumerate(questions):
        bge_ranked[q["qid"]] = topk_ids(sims[i], ids, k)

    # Hybrid RRF(bm25, bge)
    rrf_k = cfg["hybrid"]["rrf_k"]
    hybrid_ranked = {}
    for q in questions:
        qid = q["qid"]
        fused = defaultdict(float)
        bm_order = np.argsort(bm25_scores[qid])[::-1]
        for rank, di in enumerate(bm_order[:100]):
            fused[ids[di]] += 1.0 / (rrf_k + rank + 1)
        for rank, did in enumerate(bge_ranked[qid][:100]):
            fused[did] += 1.0 / (rrf_k + rank + 1)
        hybrid_ranked[qid] = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]

    ranked = {"bm25_stale": bm25_ranked, "bge_stale": bge_ranked, "hybrid_stale": hybrid_ranked}
    (PROC / "retrieval_stale.json").write_text(json.dumps(ranked))

    # retrieval hit@5 per split (shows fresh evidence is gone)
    def hit(rk):
        by = defaultdict(list)
        for q in questions:
            gold = set(qrels[q["qid"]]); got = set(rk[q["qid"]][:k])
            by[q["split"]].append(1.0 if got & gold else 0.0)
            by["all"].append(1.0 if got & gold else 0.0)
        return {s: round(float(np.mean(v)), 3) for s, v in by.items()}
    hits = {r: hit(ranked[r]) for r in ranked}
    print("stale-index hit@5:", json.dumps(hits), flush=True)
    return ranked, hits


# ------------------------------ generation --------------------------------
SYS = ("You are a precise question-answering system. Answer the question with the "
       "shortest possible span (a name, number, date, or short phrase). Do not "
       "explain. If evidence documents are provided, answer using them.")


def build_prompt(q, doc_ids, corpus_by_id):
    if not doc_ids:
        return f"Question ({q['date']}): {q['question']}\nAnswer with the shortest exact span."
    ctx = []
    for i, did in enumerate(doc_ids, 1):
        d = corpus_by_id.get(did)
        if d:
            ctx.append(f"[{i}] ({d.get('publish_date','')}) {d['text'][:DOC_CHARS]}")
    return ("Use the dated evidence documents to answer the question.\n\n" + "\n\n".join(ctx)
            + f"\n\nQuestion ({q['date']}): {q['question']}\n"
              "Answer with the shortest exact span, using the documents.")


def generate_local(questions, corpus_by_id, conds, model_id, gen_tag, batch=16):
    """Local transformers generation (no API), same prompt as the main pipeline.
    Writes generations_<gen_tag>/<cond>.jsonl so the held-out condition uses the
    SAME open-weight generator as its full-index counterpart."""
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    print(f"loaded {model_id}", flush=True)
    ranked = json.loads((PROC / "retrieval_stale.json").read_text())
    outdir = PROC / f"generations_{gen_tag}"; outdir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def gen_batch(prompts):
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1536)
        out = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(g, skip_special_tokens=True).strip() for g in out[:, enc.input_ids.shape[1]:]]

    for cond in conds:
        out_path = outdir / f"{cond}.jsonl"
        cache = set()
        if out_path.exists():
            cache = {json.loads(l)["qid"] for l in out_path.read_text().splitlines() if l.strip()}
        todo = [q for q in questions if q["qid"] not in cache]
        print(f"[{cond}] cached={len(cache)} todo={len(todo)}", flush=True)
        with out_path.open("a") as fout:
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                dl = [ranked[cond][q["qid"]] for q in chunk]
                preds = gen_batch([build_prompt(q, d, corpus_by_id) for q, d in zip(chunk, dl)])
                for q, d, pred in zip(chunk, dl, preds):
                    fout.write(json.dumps({"qid": q["qid"], "condition": cond,
                                           "model": model_id, "prediction": pred, "doc_ids": d}) + "\n")
                fout.flush()
                if (i // batch) % 5 == 0:
                    print(f"  {cond}: {i+len(chunk)}/{len(todo)}", flush=True)
    print("local generation done", flush=True)


# ------------------------------ aggregate ---------------------------------
def eff_split(q):
    return "fresh" if parse(q["date"]) >= date(*CUTOFF) else "static"


def aggregate_phase(questions, corpus_by_id, qrels, gen_tag):
    """Compare full-index vs stale-index for the SAME local generator (gen_tag),
    partitioned at the 2025 effective cutoff (matching the main analysis)."""
    qmap = {q["qid"]: q for q in questions}
    gdir = PROC / f"generations_{gen_tag}"

    def load_cond(name):
        p = gdir / f"{name}.jsonl"
        rows = {}
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); rows[r["qid"]] = r
        return rows

    def per_example(rows):
        rec = {}
        for qid, r in rows.items():
            q = qmap.get(qid)
            if q is None or not r.get("prediction", "").strip():
                if q is not None:  # empty prediction still counts as F1 0 / not abstain
                    rec[qid] = {"f1": 0.0, "abstain": False, "split": eff_split(q)}
                continue
            rec[qid] = {"f1": M.token_f1(r["prediction"], q["gold_answer"]),
                        "abstain": M.is_abstention(r["prediction"]),
                        "split": eff_split(q)}
        return rec

    def split_f1(rec, split):
        return [v["f1"] for v in rec.values() if v["split"] == split]

    ranked = json.loads((PROC / "retrieval_stale.json").read_text())
    k = 5

    def hit(rk):
        by = defaultdict(list)
        for q in questions:
            gold = set(qrels[q["qid"]]); got = set(rk[q["qid"]][:k])
            by[eff_split(q)].append(1.0 if got & gold else 0.0)
        return {s: round(float(np.mean(v)), 3) for s, v in by.items()}

    cb = per_example(load_cond("closed_book"))
    out = {"cutoff": f"{CUTOFF[0]}-01-01", "generator": gen_tag,
           "note": "Stale index = docs published before the cutoff; Fresh questions "
                   "lose contemporaneous evidence. Same local generator on both.",
           "closed_book_fresh_f1": round(100*float(np.mean(split_f1(cb, "fresh"))), 2)
           if cb else None,
           "conditions": {}}
    for full, stale in [("hybrid", "hybrid_stale"), ("bm25", "bm25_stale")]:
        full_rec = per_example(load_cond(full))
        stale_rec = per_example(load_cond(stale))
        if not full_rec or not stale_rec:
            continue
        entry = {"stale_index_hit@5": hit(ranked[stale])}
        for label, rec in (("full_index", full_rec), ("stale_index", stale_rec)):
            e = {}
            for sp in ("static", "fresh"):
                v = split_f1(rec, sp)
                if v:
                    mean, lo, hi = M.bootstrap_ci(v, 1000, seed=0)
                    e[sp] = {"n": len(v), "f1": round(100*mean, 2),
                             "f1_ci95": [round(100*lo, 2), round(100*hi, 2)]}
            entry[label] = e
        common = [qid for qid in stale_rec if qid in full_rec and eff_split(qmap[qid]) == "fresh"]
        a = [full_rec[q]["f1"] for q in common]; b = [stale_rec[q]["f1"] for q in common]
        p = M.paired_bootstrap_pvalue(a, b, 1000, seed=0)
        entry["fresh_drop_full_minus_stale"] = {
            "diff_f1": round(100*(np.mean(a) - np.mean(b)), 2),
            "one_sided_p": round(p, 4), "n": len(common),
            "test": "paired bootstrap over the same Fresh questions",
        }
        out["conditions"][full] = entry
    (RES / "heldout_index.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["retrieve", "generate", "aggregate", "all"])
    ap.add_argument("--conditions", default="hybrid_stale")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--gen-tag", default="qwen3b")
    args = ap.parse_args()

    questions, corpus, qrels = load()
    corpus_by_id = {d["doc_id"]: d for d in corpus}
    rcfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text())
    cfg, k = rcfg["retrievers"], rcfg["top_k"]

    if args.phase in ("retrieve", "all"):
        retrieve_phase(questions, corpus, qrels, cfg, k)
    if args.phase in ("generate", "all"):
        generate_local(questions, corpus_by_id, args.conditions.split(","), args.model, args.gen_tag)
    if args.phase in ("aggregate", "all"):
        aggregate_phase(questions, corpus_by_id, qrels, args.gen_tag)


if __name__ == "__main__":
    main()
