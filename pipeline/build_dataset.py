#!/usr/bin/env python3
"""Turn raw RealTimeQA files into a processed FreshBench-RAG benchmark.

Outputs (data/processed/):
  questions.jsonl : {qid, date, year, split, question, gold_answer,
                     gold_answer_idx, choices, source, url}
  corpus.jsonl    : {doc_id, text, publish_date, source_qid, url}  (pooled, deduped)
  qrels.json      : {qid: [relevant doc_ids]}  (a question's own evidence docs)
  stats.json      : summary counts (for the paper's dataset table T1)

Everything is derived from downloaded data; no numbers are invented.
"""
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def load_cfg():
    return yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())


def parse_date(s):
    s = (s or "").strip().replace("-", "/")
    for fmt in ("%Y/%m/%d", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:10], "%Y/%m/%d")
        except ValueError:
            continue
    return None


def norm_key(text):
    return hashlib.sha1(" ".join(text.lower().split())[:300].encode()).hexdigest()[:16]


def load_questions(cfg):
    qs = []
    for f in sorted(RAW.glob("*_qa.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ans = d.get("answer", [])
            if cfg["sample"]["require_single_answer"] and len(ans) != 1:
                continue
            choices = d.get("choices", [])
            if len(choices) < cfg["sample"]["min_choices"]:
                continue
            try:
                idx = int(ans[0])
                gold = choices[idx]
            except (ValueError, IndexError):
                continue
            qd = parse_date(d.get("question_date"))
            if qd is None:
                continue
            qs.append({
                "qid": d["question_id"],
                "date": qd.strftime("%Y-%m-%d"),
                "_dt": qd,
                "year": qd.year,
                "question": d["question_sentence"].strip(),
                "gold_answer": gold.strip(),
                "gold_answer_idx": idx,
                "choices": choices,
                "source": d.get("question_source", ""),
                "url": d.get("question_url", ""),
            })
    return qs


def sample_questions(qs, cfg):
    rng = random.Random(cfg["sample_seed"])
    cutoff = parse_date(cfg["fresh_after"])
    fresh, static_by_year = [], defaultdict(list)
    for q in qs:
        if q["_dt"] >= cutoff:
            q["split"] = "fresh"
            fresh.append(q)
        else:
            q["split"] = "static"
            static_by_year[q["year"]].append(q)
    chosen = list(fresh) if cfg["sample"]["fresh_take_all"] else []
    n = cfg["sample"]["static_per_year"]
    for yr, lst in sorted(static_by_year.items()):
        rng.shuffle(lst)
        chosen.extend(lst[:n])
    rng.shuffle(chosen)
    return chosen


def load_gcs_index():
    """qid -> list of search_result docs."""
    idx = {}
    for f in sorted(RAW.glob("*_gcs.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            idx[d["question_id"]] = d.get("search_result", [])
    return idx


def build_corpus(chosen, gcs, cfg):
    ccfg = cfg["corpus"]
    lo, hi = ccfg["min_doc_chars"], ccfg["max_doc_chars"]
    max_lag = timedelta(days=ccfg["drop_docs_after_question_date_days"])
    corpus, key2id, qrels = {}, {}, defaultdict(list)
    for q in chosen:
        for r in gcs.get(q["qid"], []):
            text = (r.get("text") or "").strip()
            if len(text) < lo:
                continue
            text = text[:hi]
            pub = parse_date(r.get("publish_date"))
            # timestamp invariant: evidence must not post-date the question (+lag)
            if pub is not None and pub > q["_dt"] + max_lag:
                continue
            k = norm_key(text)
            if k not in key2id:
                did = f"d{len(key2id):06d}"
                key2id[k] = did
                corpus[did] = {
                    "doc_id": did,
                    "text": text,
                    "publish_date": pub.strftime("%Y-%m-%d") if pub else "",
                    "source_qid": q["qid"],
                    "url": r.get("url", ""),
                }
            did = key2id[k]
            if did not in qrels[q["qid"]]:
                qrels[q["qid"]].append(did)
    return corpus, qrels


def main():
    cfg = load_cfg()
    OUT.mkdir(parents=True, exist_ok=True)
    qs = load_questions(cfg)
    print(f"parsed {len(qs)} usable questions")
    chosen = sample_questions(qs, cfg)
    gcs = load_gcs_index()
    corpus, qrels = build_corpus(chosen, gcs, cfg)

    # keep only questions that have >=1 evidence doc (needed for oracle/qrels)
    chosen = [q for q in chosen if qrels.get(q["qid"])]

    with (OUT / "questions.jsonl").open("w") as f:
        for q in chosen:
            q = {k: v for k, v in q.items() if k != "_dt"}
            f.write(json.dumps(q) + "\n")
    with (OUT / "corpus.jsonl").open("w") as f:
        for d in corpus.values():
            f.write(json.dumps(d) + "\n")
    (OUT / "qrels.json").write_text(json.dumps({q["qid"]: qrels[q["qid"]] for q in chosen}))

    from collections import Counter
    by_split = Counter(q["split"] for q in chosen)
    by_year = Counter(q["year"] for q in chosen)
    ndocs = [len(qrels[q["qid"]]) for q in chosen]
    stats = {
        "n_questions": len(chosen),
        "by_split": dict(by_split),
        "by_year": dict(sorted(by_year.items())),
        "corpus_docs": len(corpus),
        "avg_gold_docs_per_q": round(sum(ndocs) / len(ndocs), 2),
        "knowledge_cutoff": cfg["knowledge_cutoff"],
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
