#!/usr/bin/env python3
"""Generate answers with a real LLM (Anthropic Claude) under each condition.

Conditions:
  closed_book                : question only (parametric / memorized answer)
  bm25/dpr/bge/splade/colbert/hybrid : question + that retriever's top-k docs
  oracle                     : question + its gold evidence docs

Responses are cached per (condition, qid) in data/processed/generations/ so
metrics never require regeneration and re-runs cost nothing. Deterministic
(temperature 0). Honest substitution: the manuscript names GPT-4o; no OpenAI
key is available, so the model actually used is the Anthropic model in
configs/models.yaml — recorded in every output row.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
GEN = PROC / "generations"
DOC_CHARS = 600   # per-doc context truncation


def load_cfg():
    return yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["generator"]


RETRIEVAL_FILE = ["retrieval.json"]


def load_data():
    questions = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines()]
    corpus = {d["doc_id"]: d for d in
              (json.loads(l) for l in (PROC / "corpus.jsonl").read_text().splitlines())}
    rpath = PROC / RETRIEVAL_FILE[0]
    ranked = json.loads(rpath.read_text()) if rpath.exists() else {}
    return questions, corpus, ranked


def build_prompt(q, doc_ids, corpus):
    if not doc_ids:   # closed book
        return (f"Question ({q['date']}): {q['question']}\n"
                f"Answer with the shortest exact span.")
    ctx = []
    for i, did in enumerate(doc_ids, 1):
        d = corpus.get(did)
        if not d:
            continue
        dt = d.get("publish_date", "")
        ctx.append(f"[{i}] ({dt}) {d['text'][:DOC_CHARS]}")
    return ("Use the dated evidence documents to answer the question.\n\n"
            + "\n\n".join(ctx)
            + f"\n\nQuestion ({q['date']}): {q['question']}\n"
              f"Answer with the shortest exact span, using the documents.")


def conditions(ranked):
    conds = ["closed_book"] + [r for r in ranked if r != "oracle"] + ["oracle"]
    # de-dup preserve order
    seen, out = set(), []
    for c in conds:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on N questions")
    ap.add_argument("--conditions", default="", help="comma list; default=all")
    ap.add_argument("--model", default="", help="override generator model id")
    ap.add_argument("--tag", default="", help="write to generations_<tag>/ (2nd model)")
    ap.add_argument("--retrieval", default="retrieval.json",
                    help="ranked-results file under data/processed/ "
                         "(retrieval_masked.json for the per-query temporal mask)")
    ap.add_argument("--suffix", default="",
                    help="appended to the output condition name, e.g. _masked")
    args = ap.parse_args()
    RETRIEVAL_FILE[0] = args.retrieval

    cfg = load_cfg()
    if args.model:
        cfg["model"] = args.model
    global GEN
    if args.tag:
        GEN = PROC / f"generations_{args.tag}"
    questions, corpus, ranked = load_data()
    if args.limit:
        questions = questions[:args.limit]
    conds = args.conditions.split(",") if args.conditions else conditions(ranked)
    GEN.mkdir(parents=True, exist_ok=True)

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = cfg["model"]
    lock = threading.Lock()
    counters = {"calls": 0, "cache": 0, "err": 0, "in_tok": 0, "out_tok": 0}

    def ask(prompt):
        for attempt in range(cfg["max_retries"]):
            try:
                r = client.messages.create(
                    model=model, max_tokens=cfg["max_tokens"],
                    temperature=cfg["temperature"], system=cfg["system_prompt"],
                    messages=[{"role": "user", "content": prompt}])
                txt = "".join(b.text for b in r.content if b.type == "text").strip()
                return txt, r.usage.input_tokens, r.usage.output_tokens
            except Exception as e:  # noqa: BLE001
                if attempt == cfg["max_retries"] - 1:
                    raise
                time.sleep(2 ** attempt)

    for cond in conds:
        out_path = GEN / f"{cond}{args.suffix}.jsonl"
        cache = {}
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    cache[row["qid"]] = row
        todo = [q for q in questions if q["qid"] not in cache]
        print(f"[{cond}] cached={len(cache)} todo={len(todo)}", flush=True)

        def work(q):
            doc_ids = [] if cond == "closed_book" else ranked[cond][q["qid"]]
            prompt = build_prompt(q, doc_ids, corpus)
            try:
                ans, it, ot = ask(prompt)
            except Exception as e:  # noqa: BLE001
                with lock:
                    counters["err"] += 1
                return {"qid": q["qid"], "condition": cond + args.suffix, "model": model,
                        "prediction": "", "error": str(e), "doc_ids": doc_ids}
            with lock:
                counters["calls"] += 1
                counters["in_tok"] += it; counters["out_tok"] += ot
                if counters["calls"] % 100 == 0:
                    print(f"  {cond}: {counters['calls']} calls "
                          f"(in {counters['in_tok']}, out {counters['out_tok']} tok)", flush=True)
            return {"qid": q["qid"], "condition": cond + args.suffix, "model": model,
                    "prediction": ans, "doc_ids": doc_ids}

        if todo:
            with out_path.open("a") as fout, \
                 cf.ThreadPoolExecutor(max_workers=cfg["concurrency"]) as ex:
                for row in ex.map(work, todo):
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
    print(f"DONE calls={counters['calls']} err={counters['err']} "
          f"in_tok={counters['in_tok']} out_tok={counters['out_tok']}")


if __name__ == "__main__":
    main()
