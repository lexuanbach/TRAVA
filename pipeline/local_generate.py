#!/usr/bin/env python3
"""Generate answers with a LOCAL open-weight model (known knowledge cutoff).

Adds a third generator from a different model family whose training cutoff is
DISCLOSED, so we can (a) test whether the parametric-cliff / retrieval-rescue
pattern is Claude-specific, and (b) validate the empirical effective-cutoff
estimator against a model with a known cutoff. Small open models also tend to
hallucinate rather than abstain, a useful contrast to Claude.

Reuses the exact prompts and retrieval rankings from the main pipeline. CPU,
batched greedy decoding, on-disk caching. Writes data/processed/generations_<tag>/.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "data" / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

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
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--retrieval", default="retrieval.json",
                    help="ranking file under data/processed; use "
                         "retrieval_masked.json for the per-query "
                         "temporal mask")
    ap.add_argument("--suffix", default="",
                    help="appended to the output condition name, e.g. _masked")
    ap.add_argument("--conditions", default="closed_book,hybrid")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    # Backward-compatible: the defaults reproduce the original CPU/fp32 runs
    # that produced the shipped Qwen caches. The new rungs use --device mps
    # --dtype bfloat16, which is recorded per row in the cache.
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--hf-home", default=None,
                    help="override HF_HOME (e.g. the user cache holding a model "
                         "this project has not downloaded itself)")
    args = ap.parse_args()
    RETRIEVAL_FILE[0] = args.retrieval
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).eval()
    if args.device != "cpu":
        model = model.to(args.device)
    print(f"loaded {args.model} on {args.device}/{args.dtype}", flush=True)

    questions, corpus, ranked = load_data()
    if args.limit:
        questions = questions[:args.limit]
    outdir = PROC / f"generations_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def gen_batch(prompts):
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1536).to(args.device)
        out = model.generate(**enc, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip() for g in gen]

    for cond in args.conditions.split(","):
        out_path = outdir / f"{cond}{args.suffix}.jsonl"
        cache = set()
        if out_path.exists():
            cache = {json.loads(l)["qid"] for l in out_path.read_text().splitlines() if l.strip()}
        todo = [q for q in questions if q["qid"] not in cache]
        print(f"[{cond}] cached={len(cache)} todo={len(todo)}", flush=True)
        with out_path.open("a") as fout:
            for i in range(0, len(todo), args.batch):
                chunk = todo[i:i + args.batch]
                prompts, doclists = [], []
                for q in chunk:
                    dids = [] if cond == "closed_book" else ranked[cond][q["qid"]]
                    doclists.append(dids)
                    prompts.append(build_user(q, dids, corpus))
                preds = gen_batch(prompts)
                for q, dids, pred in zip(chunk, doclists, preds):
                    fout.write(json.dumps({"qid": q["qid"], "condition": cond + args.suffix,
                                           "model": args.model, "prediction": pred,
                                           "doc_ids": dids,
                                           "device": args.device,
                                           "dtype": args.dtype}) + "\n")
                fout.flush()
                if (i // args.batch) % 5 == 0:
                    print(f"  {cond}: {i+len(chunk)}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
