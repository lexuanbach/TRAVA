#!/usr/bin/env python3
"""Generate answers with a larger LOCAL model via llama.cpp (quantized GGUF).

Lets us run a 7B open-weight model on CPU fast enough to complete, providing a
capable LOCAL generator (disclosed cutoff, fully reproducible) that can exhibit
the parametric cliff, closing the gap that both strong models are otherwise
Claude-family. Reuses the exact prompts and retrieval rankings.
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
DOC_CHARS = 600
SYS = ("You are a precise question-answering system. Answer the question with the "
       "shortest possible span (a name, number, date, or short phrase). Do not "
       "explain. If evidence documents are provided, answer using them.")


RETRIEVAL_FILE = ["retrieval.json"]


def load_data():
    questions = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()]
    corpus = {d["doc_id"]: d for d in
              (json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines())}
    ranked = json.loads((PROC / RETRIEVAL_FILE[0]).read_text())
    return questions, corpus, ranked


def build_user(q, doc_ids, corpus):
    if not doc_ids:
        return f"Question ({q['date']}): {q['question']}\nAnswer with the shortest exact span."
    ctx = []
    for i, did in enumerate(doc_ids, 1):
        d = corpus.get(did)
        if d:
            ctx.append(f"[{i}] ({d.get('publish_date','')}) {d['text'][:DOC_CHARS]}")
    return ("Use the dated evidence documents to answer the question.\n\n" + "\n\n".join(ctx)
            + f"\n\nQuestion ({q['date']}): {q['question']}\n"
              "Answer with the shortest exact span, using the documents.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-7B-Instruct-GGUF")
    ap.add_argument("--files", default="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf,"
                    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
                    help="comma list of GGUF shards; llama.cpp loads from the first")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--retrieval", default="retrieval.json",
                    help="ranking file under data/processed; use "
                         "retrieval_masked.json for the per-query "
                         "temporal mask")
    ap.add_argument("--suffix", default="",
                    help="appended to the output condition name, e.g. _masked")
    ap.add_argument("--conditions", default="closed_book,hybrid")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--n_gpu_layers", type=int, default=-1, help="-1 = offload all layers to Metal GPU")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    RETRIEVAL_FILE[0] = args.retrieval

    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    paths = []
    for fn in args.files.split(","):
        print(f"downloading {args.repo}/{fn} ...", flush=True)
        paths.append(hf_hub_download(repo_id=args.repo, filename=fn))
    path = paths[0]
    print("model at", path, flush=True)

    from llama_cpp import Llama
    llm = Llama(model_path=path, n_ctx=2048, n_threads=args.threads,
                n_gpu_layers=args.n_gpu_layers, verbose=False)
    print(f"loaded gguf (n_gpu_layers={args.n_gpu_layers})", flush=True)

    questions, corpus, ranked = load_data()
    if args.limit:
        questions = questions[:args.limit]
    outdir = PROC / f"generations_{args.tag}"; outdir.mkdir(parents=True, exist_ok=True)

    def ask(user):
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            max_tokens=32, temperature=0.0)
        return r["choices"][0]["message"]["content"].strip()

    for cond in args.conditions.split(","):
        out_path = outdir / f"{cond}{args.suffix}.jsonl"
        cache = set()
        if out_path.exists():
            cache = {json.loads(l)["qid"] for l in out_path.read_text().splitlines() if l.strip()}
        todo = [q for q in questions if q["qid"] not in cache]
        print(f"[{cond}] cached={len(cache)} todo={len(todo)}", flush=True)
        with out_path.open("a") as fout:
            for i, q in enumerate(todo):
                dids = [] if cond == "closed_book" else ranked[cond][q["qid"]]
                pred = ask(build_user(q, dids, corpus))
                fout.write(json.dumps({"qid": q["qid"], "condition": cond + args.suffix,
                                       "model": args.repo, "prediction": pred,
                                       "doc_ids": dids}) + "\n")
                if i % 50 == 0:
                    fout.flush(); print(f"  {cond}: {i}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
