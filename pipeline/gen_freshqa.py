#!/usr/bin/env python3
"""Closed-book answers on the FreshQA replication set.

Deliberately reuses the SAME system prompt and decoding settings as
`local_generate.py`, because the point of a replication is that only the data
changes. FreshQA ships no evidence corpus, so there is no retrieval condition
here; this tests the closed-book boundary claim, which is the one the paper's
headline rests on.

Two backends, matching the main pipeline: `--backend hf` for models that load
under transformers, `--backend gguf` for the quantised 32B/72B/123B rungs that
run through llama.cpp exactly as stage 02 runs them.

Writes data/processed/generations_<tag>/freshqa.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROC = ROOT / "data" / "processed"

# identical to local_generate.py
SYS = ("You are a precise question-answering system. Answer the question with the "
       "shortest possible span (a name, number, date, or short phrase). Do not "
       "explain. If evidence documents are provided, answer using them.")


def build_user(q):
    return (f"Question: {q['question']}\n"
            "Answer with the shortest exact span.")


def run_hf(questions, out_path, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).eval()
    if args.device != "cpu":
        model = model.to(args.device)
    print(f"loaded {args.model} on {args.device}/{args.dtype}", flush=True)

    @torch.no_grad()
    def gen(prompts):
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1536).to(args.device)
        out = model.generate(**enc, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return [tok.decode(g, skip_special_tokens=True).strip()
                for g in out[:, enc.input_ids.shape[1]:]]

    with out_path.open("a") as fh:
        for i in range(0, len(questions), args.batch):
            chunk = questions[i:i + args.batch]
            preds = gen([build_user(q) for q in chunk])
            for q, p in zip(chunk, preds):
                fh.write(json.dumps({"qid": q["qid"], "condition": "freshqa",
                                     "model": args.model, "prediction": p,
                                     "doc_ids": [], "device": args.device,
                                     "dtype": args.dtype}) + "\n")
            fh.flush()
            if (i // args.batch) % 5 == 0:
                print(f"  {i + len(chunk)}/{len(questions)}", flush=True)


def run_gguf(questions, out_path, args):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    paths = [hf_hub_download(repo_id=args.repo, filename=f)
             for f in args.files.split(",")]
    llm = Llama(model_path=paths[0], n_ctx=2048, n_threads=12,
                n_gpu_layers=-1, verbose=False)
    print(f"loaded {args.repo}", flush=True)
    with out_path.open("a") as fh:
        for k, q in enumerate(questions):
            r = llm.create_chat_completion(
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": build_user(q)}],
                max_tokens=32, temperature=0.0)
            pred = r["choices"][0]["message"]["content"].strip()
            fh.write(json.dumps({"qid": q["qid"], "condition": "freshqa",
                                 "model": args.repo, "prediction": pred,
                                 "doc_ids": []}) + "\n")
            if k % 25 == 0:
                fh.flush()
                print(f"  {k}/{len(questions)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="hf", choices=["hf", "gguf"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--repo", default="")
    ap.add_argument("--files", default="")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hf-home", default=None)
    args = ap.parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    questions = [json.loads(l) for l in
                 (PROC / "freshqa.jsonl").read_text().splitlines() if l.strip()]
    outdir = PROC / f"generations_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "freshqa.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["qid"] for l in
                out_path.read_text().splitlines() if l.strip()}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[{args.tag}] cached={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    (run_gguf if args.backend == "gguf" else run_hf)(todo, out_path, args)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
