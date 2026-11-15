#!/usr/bin/env python3
"""Download RealTimeQA (weekly time-stamped news QA + retrieved evidence).

Source: https://github.com/realtimeqa/realtimeqa_public (see DATA_LICENSE.md)
We pull, for every weekly stem that has BOTH a `*_qa.jsonl` (questions + gold
answers + dates) and a `*_gcs.jsonl` (Google-Custom-Search evidence docs with
publish dates), the two raw files into data/raw/.

These two file types give us everything a rolling RAG benchmark needs:
  - time-stamped questions with gold answers            (the QA task)
  - time-stamped candidate evidence documents           (the retrieval corpus)

No fabrication anywhere: every byte here is downloaded from the public repo.
"""
import concurrent.futures as cf
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = "realtimeqa/realtimeqa_public"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/main?recursive=1"
RAW = "https://raw.githubusercontent.com/" + REPO + "/main/{path}"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"


def get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "freshbench-rag/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_stems():
    tree = json.loads(get(TREE_URL))
    paths = [t["path"] for t in tree.get("tree", [])]
    qa, gcs = {}, {}
    for p in paths:
        m = re.match(r"past/(\d{4})/(\d{8})_qa\.jsonl$", p)
        if m:
            qa[(m.group(1), m.group(2))] = p
        m = re.match(r"past/(\d{4})/(\d{8})_gcs\.jsonl$", p)
        if m:
            gcs[(m.group(1), m.group(2))] = p
    both = sorted(set(qa) & set(gcs))
    return [(y, d, qa[(y, d)], gcs[(y, d)]) for (y, d) in both]


def fetch_one(args):
    y, d, qa_path, gcs_path = args
    ok = 0
    for kind, path in (("qa", qa_path), ("gcs", gcs_path)):
        dest = OUT / f"{y}_{d}_{kind}.jsonl"
        if dest.exists() and dest.stat().st_size > 0:
            ok += 1
            continue
        try:
            data = get(RAW.format(path=path))
            dest.write_bytes(data)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ! {path}: {e}", file=sys.stderr)
    return ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stems = list_stems()
    print(f"Found {len(stems)} weekly stems with both qa+gcs "
          f"(span {stems[0][1]} .. {stems[-1][1]})")
    done = 0
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for n in ex.map(fetch_one, stems):
            done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(stems)} stems")
    files = sorted(OUT.glob("*.jsonl"))
    print(f"Downloaded {len(files)} raw files into {OUT}")


if __name__ == "__main__":
    main()
