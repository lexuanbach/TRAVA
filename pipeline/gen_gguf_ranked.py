#!/usr/bin/env python3
"""Generate with a GGUF model (llama.cpp, Metal GPU) over arbitrary precomputed
rankings, so the held-out-index and leave-one-out controls can run on the SAME
model as the main table (the local 32B primary) rather than only on Qwen-3B.

--jobs is a comma list of  cond_name:ranked_file[:key]  entries. If :key is given
the file is {key: {qid:[docs]}}; otherwise it is a flat {qid:[docs]}.
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


def build(q, dids, corpus):
    if not dids:
        return f"Question ({q['date']}): {q['question']}\nAnswer with the shortest exact span."
    ctx = [f"[{i}] ({corpus[d].get('publish_date','')}) {corpus[d]['text'][:DOC_CHARS]}"
           for i, d in enumerate(dids, 1) if d in corpus]
    return ("Use the dated evidence documents to answer the question.\n\n" + "\n\n".join(ctx)
            + f"\n\nQuestion ({q['date']}): {q['question']}\nAnswer with the shortest exact span, using the documents.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", required=True, help="cond:ranked_file[:key],...")
    ap.add_argument("--corpus", default="corpus.jsonl",
                    help="corpus file the doc_ids in --jobs refer to "
                         "(use corpus_full.jsonl for distractor-corpus conditions)")
    ap.add_argument("--n_gpu_layers", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    paths = [hf_hub_download(repo_id=args.repo, filename=fn) for fn in args.files.split(",")]
    from llama_cpp import Llama
    llm = Llama(model_path=paths[0], n_ctx=2048, n_threads=args.threads,
                n_gpu_layers=args.n_gpu_layers, verbose=False)
    print(f"loaded {args.repo}", flush=True)

    questions = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()]
    corpus = {d["doc_id"]: d for d in (json.loads(l) for l in (PROC / args.corpus).read_text().splitlines())}
    print(f"corpus={args.corpus} docs={len(corpus)}", flush=True)
    outdir = PROC / f"generations_{args.tag}"; outdir.mkdir(parents=True, exist_ok=True)

    def ask(user):
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            max_tokens=32, temperature=0.0)
        return r["choices"][0]["message"]["content"].strip()

    for job in args.jobs.split(","):
        parts = job.split(":")
        cond, rfile = parts[0], parts[1]
        key = parts[2] if len(parts) > 2 else None
        ranked = json.loads((PROC / rfile).read_text())
        if key:
            ranked = ranked[key]
        out_path = outdir / f"{cond}.jsonl"
        cache = set()
        if out_path.exists():
            cache = {json.loads(l)["qid"] for l in out_path.read_text().splitlines() if l.strip()}
        todo = [q for q in questions if q["qid"] not in cache]
        print(f"[{cond}] todo={len(todo)}", flush=True)
        with out_path.open("a") as f:
            for i, q in enumerate(todo):
                dids = ranked[q["qid"]]
                f.write(json.dumps({"qid": q["qid"], "condition": cond, "model": args.repo,
                                    "prediction": ask(build(q, dids, corpus)), "doc_ids": dids}) + "\n")
                if i % 50 == 0:
                    f.flush(); print(f"  {cond}: {i}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
