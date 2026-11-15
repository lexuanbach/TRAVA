#!/usr/bin/env python3
"""Leave-one-out-gold-docs control: test retrieval OUTSIDE the
pooled-gold best case. For each question we remove its OWN gold documents from the
candidate pool, retrieve the top-5 from the REST of the corpus, and re-generate
(Qwen-3B, local). If retrieval still rescues, it is finding genuine alternative
evidence; if F1 collapses toward closed-book, the main numbers depended on the
planted gold docs.

Phases: retrieve (bm25+bge+hybrid, k over full corpus, LOO-mask gold) -> generate
(Qwen-3B) -> aggregate (Static/Fresh F1 vs full-index, 2025 boundary).
"""
import argparse
import json
import os
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml

import metrics as M

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
CUT = date(2025, 1, 1)
DOC_CHARS = 600
_TOK = re.compile(r"[A-Za-z0-9]+")
SYS = ("You are a precise question-answering system. Answer the question with the "
       "shortest possible span (a name, number, date, or short phrase). Do not "
       "explain. If evidence documents are provided, answer using them.")


def tok(s): return _TOK.findall(s.lower())
def parse(s): y, m, d = map(int, s.split("-")); return date(y, m, d)
def split(q): return "fresh" if parse(q["date"]) >= CUT else "static"


def load():
    Q = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()]
    corpus = [json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines()]
    qrels = json.loads((PROC / "qrels.json").read_text())
    return Q, corpus, qrels


def retrieve(Q, corpus, qrels, cfg, k=5):
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    ids = [d["doc_id"] for d in corpus]
    pos = {d: i for i, d in enumerate(ids)}
    bm = BM25Okapi([tok(d["text"]) for d in corpus], k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])
    m = SentenceTransformer(cfg["bge"]["model"])
    demb = m.encode([d["text"] for d in corpus], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    qemb = m.encode([cfg["bge"]["query_prefix"] + q["question"] for q in Q], batch_size=64,
                    normalize_embeddings=True, show_progress_bar=False)
    rrf_k = cfg["hybrid"]["rrf_k"]
    ranked = {}
    for i, q in enumerate(Q):
        gold = set(qrels[q["qid"]])
        mask = np.array([1.0 if d not in gold else -1e9 for d in ids])   # LOO: drop own gold
        bs = np.asarray(bm.get_scores(tok(q["question"]))) + (mask < 0) * -1e9
        gs = (qemb[i] @ demb.T) + (mask < 0) * -1e9
        bm_order = np.argsort(bs)[::-1]
        bge_order = np.argsort(gs)[::-1]
        fused = defaultdict(float)
        for r, di in enumerate(bm_order[:100]):
            if ids[di] not in gold:
                fused[ids[di]] += 1.0 / (rrf_k + r + 1)
        for r, di in enumerate(bge_order[:100]):
            if ids[di] not in gold:
                fused[ids[di]] += 1.0 / (rrf_k + r + 1)
        ranked[q["qid"]] = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]
    (PROC / "retrieval_loo.json").write_text(json.dumps(ranked))
    print("LOO retrieval written", flush=True)
    return ranked


def build(q, dids, cby):
    if not dids:
        return f"Question ({q['date']}): {q['question']}\nAnswer with the shortest exact span."
    ctx = [f"[{i}] ({cby[d].get('publish_date','')}) {cby[d]['text'][:DOC_CHARS]}"
           for i, d in enumerate(dids, 1) if d in cby]
    return ("Use the dated evidence documents to answer the question.\n\n" + "\n\n".join(ctx)
            + f"\n\nQuestion ({q['date']}): {q['question']}\nAnswer with the shortest exact span, using the documents.")


def generate(Q, cby, model_id, batch=16):
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    tokz = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokz.pad_token is None:
        tokz.pad_token = tokz.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    ranked = json.loads((PROC / "retrieval_loo.json").read_text())
    outdir = PROC / "generations_qwen3b"; out_path = outdir / "hybrid_loo.jsonl"
    cache = set()
    if out_path.exists():
        cache = {json.loads(l)["qid"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [q for q in Q if q["qid"] not in cache]
    print(f"[hybrid_loo] todo={len(todo)}", flush=True)

    @torch.no_grad()
    def gen(prompts):
        texts = [tokz.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": p}],
                 tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tokz(texts, return_tensors="pt", padding=True, truncation=True, max_length=1536)
        o = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tokz.pad_token_id)
        return [tokz.decode(g, skip_special_tokens=True).strip() for g in o[:, enc.input_ids.shape[1]:]]

    with out_path.open("a") as f:
        for i in range(0, len(todo), batch):
            ch = todo[i:i + batch]
            preds = gen([build(q, ranked[q["qid"]], cby) for q in ch])
            for q, p in zip(ch, preds):
                f.write(json.dumps({"qid": q["qid"], "prediction": p, "doc_ids": ranked[q["qid"]]}) + "\n")
            f.flush()
            if (i // batch) % 5 == 0:
                print(f"  {i+len(ch)}/{len(todo)}", flush=True)
    print("LOO generation done", flush=True)


def aggregate(Q, qrels):
    qm = {q["qid"]: q for q in Q}
    def load_c(name):
        p = PROC / "generations_qwen3b" / f"{name}.jsonl"
        return {json.loads(l)["qid"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()} if p.exists() else {}
    def f1s(rows, sp):
        return [M.token_f1(rows[q].get("prediction", ""), qm[q]["gold_answer"])
                for q in rows if q in qm and split(qm[q]) == sp]
    full = load_c("hybrid"); loo = load_c("hybrid_loo"); cb = load_c("closed_book")
    ranked = json.loads((PROC / "retrieval_loo.json").read_text())
    def hit(sp):
        v = []
        for q in Q:
            if split(q) != sp: continue
            v.append(1.0 if set(ranked[q["qid"]][:5]) & set(qrels[q["qid"]]) else 0.0)
        return round(float(np.mean(v)), 3)
    out = {"note": "Leave-one-out of each question's own gold docs (Qwen-3B, hybrid). "
                   "hit@5 by construction excludes planted gold; tests rescue from the rest of the corpus.",
           "hit@5_loo": {"static": hit("static"), "fresh": hit("fresh")}}
    for label, rows in (("full_index", full), ("loo_index", loo), ("closed_book", cb)):
        e = {}
        for sp in ("static", "fresh"):
            v = f1s(rows, sp)
            if v:
                mean, lo, hi = M.bootstrap_ci(v, 2000, seed=0)
                e[sp] = {"n": len(v), "f1": round(100 * mean, 2), "f1_ci95": [round(100 * lo, 2), round(100 * hi, 2)]}
        out[label] = e
    # paired full-vs-loo on each slice
    for sp in ("static", "fresh"):
        common = [q for q in loo if q in full and split(qm[q]) == sp]
        a = [M.token_f1(full[q].get("prediction", ""), qm[q]["gold_answer"]) for q in common]
        b = [M.token_f1(loo[q].get("prediction", ""), qm[q]["gold_answer"]) for q in common]
        p = M.paired_bootstrap_pvalue(a, b, 2000, seed=0)
        out.setdefault("full_minus_loo", {})[sp] = {"diff_f1": round(100 * (np.mean(a) - np.mean(b)), 2),
                                                     "p": round(p, 4), "n": len(common)}
    (RES / "loo_control.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["retrieve", "generate", "aggregate", "all"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    a = ap.parse_args()
    Q, corpus, qrels = load()
    cby = {d["doc_id"]: d for d in corpus}
    cfg = yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text())["retrievers"]
    if a.phase in ("retrieve", "all"):
        retrieve(Q, corpus, qrels, cfg)
    if a.phase in ("generate", "all"):
        generate(Q, cby, a.model)
    if a.phase in ("aggregate", "all"):
        aggregate(Q, qrels)


if __name__ == "__main__":
    main()
