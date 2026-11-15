#!/usr/bin/env python3
"""Closed-book generation over the FULL question set (no sampling cap).

Closed-book needs no corpus, no retrieval and no API -- just the question -- so the
one legitimate fix for our underpowered null is cheap: run every usable question
instead of the 878 we sampled. Resumable: re-running skips qids already present.

  python3 pipeline/gen_closedbook_full.py --repo Qwen/Qwen2.5-32B-Instruct-GGUF \
      --files "...gguf,..." --tag qwen32b
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
SYS = ("You are a precise question-answering system. Answer the question with the "
       "shortest possible span (a name, number, date, or short phrase). Do not "
       "explain. If evidence documents are provided, answer using them.")


def prompt(q):
    # byte-identical to gen_gguf_ranked.py's no-document branch, so the full-scale
    # scores are directly comparable with the sampled run's closed_book scores.
    return (f"Question ({q['date']}): {q['question']}\n"
            "Answer with the shortest exact span.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_gpu_layers", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    paths = [hf_hub_download(repo_id=args.repo, filename=fn) for fn in args.files.split(",")]
    from llama_cpp import Llama
    llm = Llama(model_path=paths[0], n_ctx=2048, n_threads=args.threads,
                n_gpu_layers=args.n_gpu_layers, verbose=False)
    print(f"loaded {args.repo}", flush=True)

    questions = [json.loads(l) for l in
                 (PROC / "questions_full.jsonl").read_text().splitlines()]
    outdir = PROC / f"generations_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "closed_book_full.jsonl"

    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in out.read_text().splitlines() if l.strip()}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[{args.tag}] total={len(questions)} done={len(done)} todo={len(todo)}", flush=True)

    with out.open("a") as f:
        for i, q in enumerate(todo):
            r = llm.create_chat_completion(
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": prompt(q)}],
                max_tokens=32, temperature=0.0)
            f.write(json.dumps({"qid": q["qid"], "condition": "closed_book_full",
                                "model": args.repo,
                                "prediction": r["choices"][0]["message"]["content"].strip(),
                                "doc_ids": []}) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"  {args.tag}: {i}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
