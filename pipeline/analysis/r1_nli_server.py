#!/usr/bin/env python3
"""R1/R2(c) NLI labeller for the GPU server (CUDA). Runs no analysis.

The model call is exactly the one in pipeline/nli_support.py and
pipeline/analysis/e0_nli_visible.py:
  model      MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
             (revision pinned to the snapshot used for the stored labels)
  tokenizer  tok(premise, hypothesis, return_tensors="pt", truncation=True,  [run LOCALLY, ids shipped]
                 max_length=384)
  batching   one (premise, hypothesis) pair per forward pass, no padding
  labels     softmax(logits[0]); id2label lower-cased; probs rounded to 4 dp;
             label = argmax of the rounded dict (max(lab, key=lab.get))
Premise and hypothesis strings arrive pre-built in the jobs file (built by
r1_build_nli_bundle.py) TOGETHER WITH THEIR TOKEN IDS, produced locally by the
same tokenizer call with the transformers version that reproduces the stored
labels exactly. This script checks the string hashes, loads the tokenizer only
to confirm vocabulary size, and runs the forward pass on the shipped ids
(the server's transformers 4.57.1 tokenizer normalises some characters
differently, which changes labels; see r1_build_nli_bundle.py).

Usage (server):
  python3 r1_nli_server.py --jobs nli_jobs.jsonl --job undated600 --out out/undated600.jsonl
Resumable: rows already in --out are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

NLI_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
MAX_LENGTH = 384


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    todo = [r for r in (json.loads(l) for l in args.jobs.open(encoding="utf-8") if l.strip())
            if r["job"] == args.job]
    done = set()
    if args.out.exists():
        for l in args.out.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["qid"], r["doc_id"]))
    todo = [r for r in todo if (r["qid"], r["doc_id"]) not in done]
    for r in todo:
        assert sha(r["premise"]) == r["premise_sha256"], (r["qid"], r["doc_id"])
        assert sha(r["hypothesis"]) == r["hypothesis_sha256"], (r["qid"], r["doc_id"])

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(NLI_NAME, revision=REVISION)
    mdl = AutoModelForSequenceClassification.from_pretrained(NLI_NAME, revision=REVISION)
    mdl = mdl.to(args.device).eval()
    id2 = {int(k): str(v).lower() for k, v in (mdl.config.id2label or {}).items()}
    load_s = time.time() - t0
    info = {"job": args.job, "n_todo": len(todo), "n_already_done": len(done),
            "model": NLI_NAME, "revision": REVISION, "max_length": MAX_LENGTH,
            "id2label": id2, "device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "dtype": str(next(mdl.parameters()).dtype),
            "server_tokenizer_class_not_used_for_ids": type(tok).__name__,
            "ids_source": "shipped token ids (local tokenizer)",
            "max_shipped_id": max(max(r["input_ids"]) for r in todo) if todo else None,
            "model_vocab_size": int(mdl.config.vocab_size),
            "model_load_seconds": round(load_s, 1)}
    print(json.dumps(info), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    with args.out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(todo):
            assert len(r["input_ids"]) <= MAX_LENGTH
            enc = {"input_ids": torch.tensor([r["input_ids"]], dtype=torch.long),
                   "attention_mask": torch.tensor([r["attention_mask"]], dtype=torch.long)}
            if r.get("token_type_ids") is not None:
                enc["token_type_ids"] = torch.tensor([r["token_type_ids"]], dtype=torch.long)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = mdl(**enc).logits[0]
                probs = torch.softmax(logits, -1).tolist()
            lab = {id2.get(j, str(j)): round(p, 4) for j, p in enumerate(probs)}
            best = max(lab, key=lab.get)
            rec = {"job": args.job, "qid": r["qid"], "doc_id": r["doc_id"],
                   "premise_sha256": r["premise_sha256"],
                   "hypothesis_sha256": r["hypothesis_sha256"],
                   "label": best, "probs": lab,
                   "probs_unrounded": {id2.get(j, str(j)): p for j, p in enumerate(probs)},
                   "n_pair_tokens": int(enc["input_ids"].shape[1])}
            fh.write(json.dumps(rec) + "\n")
            if i % 250 == 0:
                fh.flush()
                print(f"  {args.job} {i}/{len(todo)} {time.time() - t1:.0f}s", flush=True)
    info["label_seconds"] = round(time.time() - t1, 1)
    info["seconds_per_pair"] = round((time.time() - t1) / max(len(todo), 1), 4)
    meta = args.out.with_suffix(".meta.json")
    prev = json.loads(meta.read_text()) if meta.exists() else []
    prev = prev if isinstance(prev, list) else [prev]
    meta.write_text(json.dumps(prev + [info], indent=1) + "\n")
    print("DONE", json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
