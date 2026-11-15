#!/usr/bin/env python3
"""Closed-book / RAG generation with an OpenAI model (default gpt-4o-mini).

Why this model is worth adding:
  * A THIRD model family. Our capability claim currently rests on one open-weight
    model (Mistral-Large-2) plus two Claude models; a third vendor tests whether
    the effect is a family artifact.
  * A DISCLOSED cutoff. Qwen2.5 and Mistral-Large-2 disclose none, which is why we
    report our cutoff estimator as a negative result. OpenAI states 2023-10 for
    gpt-4o-mini, so this is the one model where a per-model split is defensible
    rather than guessed.
  * It scales. The Anthropic budget is exhausted, so Haiku/Sonnet are stuck at
    n=878; gpt-4o-mini can run the full 4,182 for a few cents.

Prompts are byte-identical to gen_gguf_ranked.py / gen_closedbook_full.py so the
scores are directly comparable with the local ladder. Resumable; concurrent.

  export OPENAI_API_KEY=$(tr -d ' \\n' < ~/Desktop/keys/openai.md)
  python3 pipeline/gen_openai.py --tag gpt4omini --questions questions_full.jsonl
"""
import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
URL = "https://api.openai.com/v1/chat/completions"
SYS = ("You are a precise question-answering system. Answer the question with the "
       "shortest possible span (a name, number, date, or short phrase). Do not "
       "explain. If evidence documents are provided, answer using them.")
DOC_CHARS = 600
_lock = threading.Lock()


def build(q, dids, corpus):
    if not dids:
        return (f"Question ({q['date']}): {q['question']}\n"
                "Answer with the shortest exact span.")
    ctx = [f"[{i}] ({corpus[d].get('publish_date','')}) {corpus[d]['text'][:DOC_CHARS]}"
           for i, d in enumerate(dids, 1) if d in corpus]
    return ("Use the dated evidence documents to answer the question.\n\n"
            + "\n\n".join(ctx)
            + f"\n\nQuestion ({q['date']}): {q['question']}\n"
              "Answer with the shortest exact span, using the documents.")


def ask(key, model, user, retries=5):
    body = json.dumps({"model": model,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": user}],
                       "max_tokens": 32, "temperature": 0}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=60))
            return r["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and a < retries - 1:
                import time
                time.sleep(2 ** a)
                continue
            raise
        except Exception:
            if a < retries - 1:
                import time
                time.sleep(2 ** a)
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--tag", default="gpt4omini")
    ap.add_argument("--questions", default="questions_full.jsonl")
    ap.add_argument("--cond", default="closed_book_full")
    ap.add_argument("--ranked", default=None, help="retrieval json, for a RAG condition")
    ap.add_argument("--key", default=None, help="ranked-file key, e.g. hybrid")
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        sys.exit("OPENAI_API_KEY not set")

    questions = [json.loads(l) for l in
                 (PROC / args.questions).read_text().splitlines()]
    ranked, corpus = {}, {}
    if args.ranked:
        ranked = json.loads((PROC / args.ranked).read_text())
        if args.key:
            ranked = ranked[args.key]
        corpus = {d["doc_id"]: d for d in
                  (json.loads(l) for l in (PROC / args.corpus).read_text().splitlines())}

    outdir = PROC / f"generations_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.cond}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in out.read_text().splitlines() if l.strip()}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[{args.tag}/{args.cond}] total={len(questions)} done={len(done)} "
          f"todo={len(todo)}", flush=True)

    f = out.open("a")
    n = [0]

    def work(q):
        dids = ranked.get(q["qid"], []) if ranked else []
        try:
            pred = ask(key, args.model, build(q, dids, corpus))
        except Exception as e:
            print(f"  FAIL {q['qid']}: {type(e).__name__} {str(e)[:90]}", flush=True)
            return
        with _lock:
            f.write(json.dumps({"qid": q["qid"], "condition": args.cond,
                                "model": args.model, "prediction": pred,
                                "doc_ids": dids}) + "\n")
            n[0] += 1
            if n[0] % 200 == 0:
                f.flush()
                print(f"  {args.cond}: {n[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    f.close()
    print(f"DONE wrote={n[0]}", flush=True)


if __name__ == "__main__":
    main()
