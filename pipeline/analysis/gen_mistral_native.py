#!/usr/bin/env python3
"""Bounded native-template sensitivity for Mistral-Large-2407 (analysis review R6).

The frozen Mistral runs used llama-cpp-python's "llama-2" fallback chat format
(`[INST] <<SYS>>...<</SYS>> ... [/INST]`), because the GGUF carries no chat template.
This script re-answers the SAME questions with the SAME GGUF, n_ctx, decoding and
prompt text, changing only the wrapper to Mistral's own v3 instruct format, as
implemented by mistral_common (InstructTokenizerV2/V3, single turn):

    [BOS] [INST] encode(SYSTEM + "\n\n" + USER) [/INST]

Tokens are built directly (no chat formatter) and verified against the official v3
SentencePiece model shipped with mistral_common before generation. A new arm is run
for every question; the old fallback outputs are never mixed into it.

Outputs data/processed/analysis/generations_mistral_large/native/<cond>.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_gguf_notrunc import SYS, build, N_CTX  # noqa: E402  identical prompt text

ML = ["Mistral-Large-Instruct-2407-Q4_K_M-00001-of-00002.gguf",
      "Mistral-Large-Instruct-2407-Q4_K_M-00002-of-00002.gguf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="closed_book,hybrid_masked")
    ap.add_argument("--spm", required=True, help="mistral_instruct_tokenizer_240323.model.v3")
    ap.add_argument("--hf-home", default=os.path.expanduser("~/.cache/huggingface"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    import llama_cpp
    import sentencepiece as spm

    path = [hf_hub_download("lmstudio-community/Mistral-Large-Instruct-2407-GGUF", f) for f in ML][0]
    questions = {json.loads(l)["qid"]: json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines() if l.strip()}
    corpus = {d["doc_id"]: d for d in (json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines() if l.strip())}
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    inst, einst = sp.piece_to_id("[INST]"), sp.piece_to_id("[/INST]")

    def ref_tokens(user):
        # mistral_common V2/V3 single-turn: [BOS] [INST] sp.encode(system + "\n\n" + content) [/INST]
        return [sp.bos_id(), inst] + sp.encode(SYS + "\n\n" + user) + [einst]

    vocab = Llama(model_path=path, vocab_only=True, verbose=False)
    l_inst = vocab.tokenize(b"[INST]", add_bos=False, special=True)
    l_einst = vocab.tokenize(b"[/INST]", add_bos=False, special=True)
    assert len(l_inst) == 1 and len(l_einst) == 1, (l_inst, l_einst)

    def llama_tokens(user):
        # this GGUF's llama.cpp tokenizer adds no SentencePiece dummy-space prefix, so prepend the
        # space explicitly; identical ids then confirm the GGUF vocabulary matches the official v3 model
        return [vocab.token_bos()] + l_inst + vocab.tokenize((" " + SYS + "\n\n" + user).encode(), add_bos=False,
                                                             special=False) + l_einst

    frozen = {c: {json.loads(l)["qid"]: json.loads(l) for l in
                  (PROC / "generations_mistral_large" / f"{c}.jsonl").read_text().splitlines() if l.strip()}
              for c in args.conds.split(",")}
    # verify token construction against the official SentencePiece model on every prompt
    mism, n_checked = 0, 0
    for c in frozen:
        for qid, r in frozen[c].items():
            user = build(questions[qid], r.get("doc_ids") or [], corpus)
            n_checked += 1
            mism += llama_tokens(user) != ref_tokens(user)
    print(f"token check: {n_checked - mism}/{n_checked} prompts identical to mistral_common v3 encoding", flush=True)
    outdir = PROC / "analysis" / "generations_mistral_large" / "native"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "token_check.json").write_text(json.dumps({"n": n_checked, "identical": n_checked - mism}, indent=2))
    if mism:
        sys.exit("token construction differs from the official v3 encoding; refusing to run")
    del vocab

    llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=12, n_gpu_layers=-1, verbose=False)
    env = {"llama_cpp": llama_cpp.__version__, "n_ctx": N_CTX, "backend": "metal", "max_tokens": 32,
           "temperature": 0.0, "template": "mistral v3 instruct (mistral_common), system prepended to user turn"}
    eos = llm.token_eos()
    for c, rows in frozen.items():
        out_path = outdir / f"{c}.jsonl"
        done = set()
        if out_path.exists():
            done = {json.loads(l)["qid"] for l in out_path.read_text().splitlines() if l.strip()}
        todo = [q for q in rows if q not in done]
        if args.limit:
            todo = todo[:args.limit]
        print(f"[{c}] todo={len(todo)}", flush=True)
        t0 = time.time()
        with out_path.open("a") as f:
            for i, qid in enumerate(todo):
                dids = rows[qid].get("doc_ids") or []
                user = build(questions[qid], dids, corpus)
                toks = ref_tokens(user)
                assert len(toks) + 32 <= N_CTX
                r = llm.create_completion(prompt=toks, max_tokens=32, temperature=0.0)
                pred = r["choices"][0]["text"].strip()
                f.write(json.dumps({"qid": qid, "condition": c, "prediction": pred, "doc_ids": dids,
                                    "n_input_tokens": len(toks),
                                    "prompt_sha256": hashlib.sha256(json.dumps(toks).encode()).hexdigest(),
                                    "fallback_prediction": rows[qid]["prediction"], "env": env}) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(f"  {c}: {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
