#!/usr/bin/env python3
"""Build the upload bundle for the corrected context-budget ablation.

Arms (all under the +3-day per-query mask, Hybrid):
  k3          retrieval_masked_k3.json    (pure truncation of the k=5 list)
  k5          retrieval_masked.json       (the paper's main masked condition)
  k10refused  retrieval_masked_k10.json   (the original k=10 arm: RRF re-fused
                                           over a deeper BGE list)
  k10prefix   the k=5 list, then the next distinct documents of the re-fused
              k=10 list, so its first five passages equal k5 exactly

Writes data/processed/analysis/kabl_bundle/{questions.jsonl, corpus_subset.jsonl,
rankings.json, manifest.json}. No existing file is modified.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = PROC / "analysis" / "kabl_bundle"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qs = [json.loads(l) for l in (PROC / "questions.jsonl").read_text().splitlines() if l.strip()]
    k3 = json.loads((PROC / "retrieval_masked_k3.json").read_text())["hybrid"]
    k5 = json.loads((PROC / "retrieval_masked.json").read_text())["hybrid"]
    k10 = json.loads((PROC / "retrieval_masked_k10.json").read_text())["hybrid"]

    prefix, n_short, n_prefix_mismatch_k3 = {}, 0, 0
    for q in qs:
        qid = q["qid"]
        base = list(k5[qid])
        ext = base + [d for d in k10[qid] if d not in base]
        prefix[qid] = ext[:10]
        n_short += len(prefix[qid]) < 10
        n_prefix_mismatch_k3 += k3[qid] != base[:3]
    rankings = {"k3": {q["qid"]: k3[q["qid"]] for q in qs},
                "k5": {q["qid"]: k5[q["qid"]] for q in qs},
                "k10refused": {q["qid"]: k10[q["qid"]] for q in qs},
                "k10prefix": prefix}

    need = {d for r in rankings.values() for v in r.values() for d in v}
    n_written = 0
    with (OUT / "corpus_subset.jsonl").open("w") as f:
        for line in (PROC / "corpus.jsonl").open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                if d["doc_id"] in need:
                    f.write(json.dumps({"doc_id": d["doc_id"], "publish_date": d.get("publish_date", ""),
                                        "text": d["text"][:600]}) + "\n")
                    n_written += 1
    (OUT / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    (OUT / "rankings.json").write_text(json.dumps(rankings))

    k10_top5_eq_k5 = sum(k10[q["qid"]][:5] == k5[q["qid"]] for q in qs)
    manifest = {
        "n_questions": len(qs),
        "n_docs_needed": len(need), "n_docs_written": n_written,
        "k3_is_prefix_of_k5": len(qs) - n_prefix_mismatch_k3,
        "k10refused_top5_equals_k5": k10_top5_eq_k5,
        "k10prefix_len_lt_10": n_short,
        "sources": {p: sha(PROC / p) for p in ("questions.jsonl", "corpus.jsonl", "retrieval_masked_k3.json",
                                               "retrieval_masked.json", "retrieval_masked_k10.json")},
        "note": "corpus_subset stores only the first 600 characters, exactly the text the prompt shows",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    assert n_written == len(need), "some ranked doc ids are missing from corpus.jsonl"


if __name__ == "__main__":
    main()
