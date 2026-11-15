#!/usr/bin/env python3
"""Re-answer ranked-context conditions with a HF model and NO silent truncation.

Why this exists: `pipeline/local_generate.py` tokenizes with
`truncation=True, max_length=1536`, and the Qwen tokenizer truncates on the
right. At k=10, 639 of 878 prompts exceed 1,536 tokens, so the question and the
assistant turn were cut off and the cached k=10 answers are passage
continuations. This script keeps the prompt construction of
`local_generate.py` byte-for-byte (same SYS, same `build_user`, same 600-char
passage prefix) but

  * never truncates: every prompt is length-checked first and the run aborts
    if any prompt exceeds --max-length;
  * logs, per row, the input token count, a SHA-256 of the rendered chat text,
    device, dtype and library versions, so "what did the model see" is
    answerable from the cache alone.

It is self-contained (no repo imports) so it can be copied to a GPU server
together with a bundle built by `make_kabl_bundle.py`.

Usage:
  python3 gen_hf_notrunc.py --bundle BUNDLE_DIR --model Qwen/Qwen2.5-3B-Instruct \
      --tag qwen3b --arms k3,k5,k10refused,k10prefix --out OUT_DIR
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

DOC_CHARS = 600
SYS = ("You are a precise question-answering system. Answer the question with the "
       "shortest possible span (a name, number, date, or short phrase). Do not "
       "explain. If evidence documents are provided, answer using them.")


def build_user(q, doc_ids, corpus):
    # identical to pipeline/local_generate.py::build_user
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
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", required=True,
                    help="comma list of ranking names in bundle/rankings.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cond-prefix", default="hybrid_masked_",
                    help="output condition = prefix + arm; use '' when arm names are full "
                         "condition names (an arm whose lists are empty is closed-book)")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions = [json.loads(l) for l in (args.bundle / "questions.jsonl").read_text().splitlines() if l.strip()]
    corpus = {d["doc_id"]: d for d in
              (json.loads(l) for l in (args.bundle / "corpus_subset.jsonl").read_text().splitlines() if l.strip())}
    rankings = json.loads((args.bundle / "rankings.json").read_text())
    if args.limit:
        questions = questions[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    arms = args.arms.split(",")

    def render(q, dids):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": build_user(q, dids, corpus)}],
            tokenize=False, add_generation_prompt=True)

    # ---- length audit BEFORE loading the model: fail fast, never truncate ----
    audit = {}
    for arm in arms:
        lens = []
        for q in questions:
            dids = rankings[arm][q["qid"]]
            missing = [d for d in dids if d not in corpus]
            if missing:
                sys.exit(f"[{arm}] {q['qid']}: doc ids missing from corpus subset: {missing[:3]}")
            lens.append(len(tok(render(q, dids)).input_ids))
        audit[arm] = {"n": len(lens), "max": max(lens), "median": sorted(lens)[len(lens) // 2],
                      "n_over_limit": sum(l > args.max_length for l in lens)}
        print(f"[audit] {arm}: {audit[arm]}", flush=True)
        if audit[arm]["n_over_limit"]:
            sys.exit(f"[{arm}] {audit[arm]['n_over_limit']} prompts exceed --max-length "
                     f"{args.max_length}; refusing to truncate")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"length_audit_{args.tag}.json").write_text(json.dumps(
        {"model": args.model, "max_length": args.max_length, "arms": audit}, indent=2))

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    env = {"device": args.device, "dtype": args.dtype, "torch": torch.__version__,
           "transformers": transformers.__version__, "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
           "batch": args.batch, "max_new_tokens": 32, "do_sample": False}
    print(f"loaded {args.model} {env}", flush=True)

    @torch.no_grad()
    def gen_batch(texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=False).to(args.device)
        out = model.generate(**enc, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        n_in = enc.attention_mask.sum(dim=1).tolist()
        return [tok.decode(g, skip_special_tokens=True).strip() for g in gen], n_in

    for arm in arms:
        cond = f"{args.cond_prefix}{arm}"
        out_path = args.out / f"generations_{args.tag}" / f"{cond}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out_path.exists():
            done = {json.loads(l)["qid"] for l in out_path.read_text().splitlines() if l.strip()}
        todo = [q for q in questions if q["qid"] not in done]
        print(f"[{arm}] cached={len(done)} todo={len(todo)}", flush=True)
        t0 = time.time()
        with out_path.open("a") as f:
            for i in range(0, len(todo), args.batch):
                chunk = todo[i:i + args.batch]
                dls = [rankings[arm][q["qid"]] for q in chunk]
                texts = [render(q, d) for q, d in zip(chunk, dls)]
                preds, n_in = gen_batch(texts)
                for q, d, t, p, n in zip(chunk, dls, texts, preds, n_in):
                    f.write(json.dumps({
                        "qid": q["qid"], "condition": cond,
                        "model": args.model, "prediction": p, "doc_ids": d,
                        "n_input_tokens": int(n), "truncated": False,
                        "prompt_sha256": hashlib.sha256(t.encode()).hexdigest(),
                        "env": env}) + "\n")
                f.flush()
                if (i // args.batch) % 20 == 0:
                    print(f"  {arm}: {i + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
