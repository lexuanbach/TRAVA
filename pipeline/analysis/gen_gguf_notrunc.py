#!/usr/bin/env python3
"""GGUF (llama.cpp, Metal) generation over a analysis bundle, mirroring
`pipeline/gen_gguf_ranked.py` exactly (same SYS, same prompt builder, n_ctx=2048,
max_tokens=32, temperature 0) but

  * reading a bundle (questions.jsonl, corpus_subset.jsonl, rankings.json) built
    by the pipeline/analysis/*_bundle scripts;
  * writing ONLY under data/processed/analysis/generations_<tag>/, never into the
    frozen data/processed/generations_* caches;
  * counting prompt tokens with the model's own tokenizer and aborting if any
    prompt + 32 new tokens would exceed n_ctx;
  * `--check-cached COND` regenerates a sample of an existing frozen condition to
    measure exact-match reproduction before cached rows are reused.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
DOC_CHARS = 600
N_CTX = 2048
SYS = ("You are a precise question-answering system. Answer the question with the "
       "shortest possible span (a name, number, date, or short phrase). Do not "
       "explain. If evidence documents are provided, answer using them.")


def build(q, dids, corpus):
    # identical to pipeline/gen_gguf_ranked.py::build
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
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--arms", default="")
    ap.add_argument("--qids", type=Path, help="optional file with one qid per line to restrict to")
    ap.add_argument("--check-cached", default="", help="frozen condition name to reproduce")
    ap.add_argument("--check-n", type=int, default=40)
    ap.add_argument("--hf-home", default=str(ROOT / "data" / "hf_cache"))
    ap.add_argument("--reuse-frozen", default="",
                    help="comma list arm=frozen_condition; a frozen row is copied (flagged "
                         "reused_from_frozen) instead of generated when its doc_ids equal the arm's "
                         "doc_ids. Valid only after --check-cached showed exact reproduction.")
    ap.add_argument("--out-subdir", default="", help="optional subdirectory under generations_<tag>/")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    import llama_cpp
    paths = [hf_hub_download(repo_id=args.repo, filename=fn) for fn in args.files.split(",")]
    llm = Llama(model_path=paths[0], n_ctx=N_CTX, n_threads=12, n_gpu_layers=-1, verbose=False)
    env = {"llama_cpp": llama_cpp.__version__, "n_ctx": N_CTX, "backend": "metal", "max_tokens": 32,
           "temperature": 0.0}

    def ask(user):
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            max_tokens=32, temperature=0.0)
        return r["choices"][0]["message"]["content"].strip()

    def n_tokens(user):
        # conservative count: system + user text plus template overhead
        return len(llm.tokenize((SYS + "\n" + user).encode(), add_bos=True)) + 32 + 40  # 32 new tokens + chat-template margin

    outdir = PROC / "analysis" / f"generations_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.check_cached:
        questions = {json.loads(l)["qid"]: json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines() if l.strip()}
        corpus = {d["doc_id"]: d for d in (json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines() if l.strip())}
        frozen = [json.loads(l) for l in (PROC / f"generations_{args.tag}" / f"{args.check_cached}.jsonl").read_text().splitlines() if l.strip()]
        random.Random(20260917).shuffle(frozen)
        rows, t0 = [], time.time()
        for r in frozen[:args.check_n]:
            p = ask(build(questions[r["qid"]], r.get("doc_ids") or [], corpus))
            rows.append({"qid": r["qid"], "frozen": r["prediction"], "now": p, "match": p == r["prediction"]})
        rep = {"condition": args.check_cached, "n": len(rows), "exact_match": sum(x["match"] for x in rows),
               "seconds": round(time.time() - t0, 1), "env": env, "rows": rows}
        (outdir / f"repro_check_{args.check_cached}.json").write_text(json.dumps(rep, indent=2))
        print(f"repro {args.tag}/{args.check_cached}: {rep['exact_match']}/{rep['n']} exact, {rep['seconds']}s", flush=True)
        return

    questions = [json.loads(l) for l in (args.bundle / "questions.jsonl").read_text().splitlines() if l.strip()]
    corpus = {d["doc_id"]: d for d in (json.loads(l) for l in (args.bundle / "corpus_subset.jsonl").read_text().splitlines() if l.strip())}
    rankings = json.loads((args.bundle / "rankings.json").read_text())
    if args.qids:
        keep = set(args.qids.read_text().split())
        questions = [q for q in questions if q["qid"] in keep]

    for arm in args.arms.split(","):
        lens = [n_tokens(build(q, rankings[arm][q["qid"]], corpus)) for q in questions]
        over = sum(l > N_CTX for l in lens)
        print(f"[audit] {arm}: n={len(lens)} max={max(lens)} over={over}", flush=True)
        if over:
            sys.exit(f"[{arm}] {over} prompts would exceed n_ctx={N_CTX}; refusing")

    reuse = dict(x.split("=", 1) for x in args.reuse_frozen.split(",") if x)
    if args.out_subdir:
        outdir = outdir / args.out_subdir
        outdir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms.split(","):
        out_path = outdir / f"{arm}.jsonl"
        done = set()
        if out_path.exists():
            done = {json.loads(l)["qid"] for l in out_path.read_text().splitlines() if l.strip()}
        frozen = {}
        if arm in reuse:
            # "arm:<name>" reuses rows this same run already wrote for another arm (same
            # configuration, identical prompt when doc_ids match); otherwise a frozen condition
            src = reuse[arm]
            fp = (outdir / f"{src[4:]}.jsonl") if src.startswith("arm:") else \
                 (PROC / f"generations_{args.tag}" / f"{src}.jsonl")
            for l in fp.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    frozen[r["qid"]] = r
        todo = [q for q in questions if q["qid"] not in done]
        n_reused = sum(1 for q in todo if q["qid"] in frozen
                       and list(frozen[q["qid"]].get("doc_ids") or []) == list(rankings[arm][q["qid"]]))
        print(f"[{arm}] cached={len(done)} todo={len(todo)} reusable_from_frozen={n_reused} "
              f"to_generate={len(todo) - n_reused}", flush=True)
        t0 = time.time()
        with out_path.open("a") as f:
            for i, q in enumerate(todo):
                dids = rankings[arm][q["qid"]]
                user = build(q, dids, corpus)
                fr = frozen.get(q["qid"])
                if fr is not None and list(fr.get("doc_ids") or []) == list(dids):
                    pred, reused = fr["prediction"], reuse[arm]
                else:
                    pred, reused = ask(user), None
                f.write(json.dumps({"qid": q["qid"], "condition": arm, "model": args.repo,
                                    "prediction": pred, "doc_ids": dids,
                                    "prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
                                    "reused_from_frozen": reused, "env": env}) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(f"  {arm}: {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
