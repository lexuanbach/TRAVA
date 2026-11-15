#!/usr/bin/env python3
"""Shared helpers for the E2/E3 ranking builders (read-only w.r.t. the repo).

* `bge_encode_cached`  -- the exact `SentenceTransformer.encode` call of
  pipeline/retrieval.py::r_bge and pipeline/retrieval_full.py (batch_size=64,
  normalize_embeddings=True, default device unless overridden), cached as .npy
  under data/processed/analysis/emb/ keyed by a SHA-256 of the input texts, the
  model, the device and the batch size. Batch composition matters on MPS
  (sentence-transformers length-sorts within the list it is given), so callers
  must pass exactly the list the original run encoded.
* `token_audit`        -- the length audit of pipeline/analysis/gen_hf_notrunc.py
  (same SYS, same build_user, same chat rendering), run locally before a bundle
  is written.
* `sha`, `date_ord`, small exposure helpers.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "pipeline"
REB = PIPE / "analysis"
for p in (PIPE, REB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PROC = ROOT / "data" / "processed"
OUTD = PROC / "analysis"
EMB = OUTD / "emb"
RES = ROOT / "results" / "analysis"
BGE_MODEL = "BAAI/bge-base-en-v1.5"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def date_ord(s: str) -> int:
    return date.fromisoformat(str(s)[:10]).toordinal()


_MODELS: dict = {}


def _model(device):
    from sentence_transformers import SentenceTransformer
    key = device or "auto"
    if key not in _MODELS:
        _MODELS[key] = (SentenceTransformer(BGE_MODEL) if device is None
                        else SentenceTransformer(BGE_MODEL, device=device))
    return _MODELS[key]


def bge_encode_cached(texts: list[str], name: str, device=None, batch_size: int = 64):
    """Return (float32 embeddings, info). Cached under EMB/<name>__<device>__<hash>.npy."""
    EMB.mkdir(parents=True, exist_ok=True)
    th = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    dev_tag = device or "auto"
    stem = f"{name}__{dev_tag}__bs{batch_size}__{th[:16]}"
    npy, meta = EMB / f"{stem}.npy", EMB / f"{stem}.json"
    if npy.is_file() and meta.is_file():
        info = json.loads(meta.read_text())
        info["cached"] = True
        return np.load(npy), info
    m = _model(device)
    t0 = time.time()
    arr = m.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    arr = np.asarray(arr)
    info = {"name": name, "n": len(texts), "texts_sha256": th, "model": BGE_MODEL,
            "requested_device": dev_tag, "actual_device": str(m.device),
            "batch_size": batch_size, "dtype": str(arr.dtype), "seconds": round(time.time() - t0, 1)}
    np.save(npy, arr)
    meta.write_text(json.dumps(info, indent=2))
    info["cached"] = False
    return arr, info


def token_audit(questions: list[dict], rankings: dict, corpus_subset: dict,
                model: str = "Qwen/Qwen2.5-3B-Instruct", max_length: int = 4096) -> dict:
    """Exactly the length audit of gen_hf_notrunc.py, per arm."""
    from transformers import AutoTokenizer
    from gen_hf_notrunc import SYS, build_user
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    out = {}
    for arm, rk in rankings.items():
        lens = []
        for q in questions:
            dids = rk[q["qid"]]
            missing = [d for d in dids if d not in corpus_subset]
            if missing:
                raise SystemExit(f"[{arm}] {q['qid']}: doc ids missing from corpus subset: {missing[:3]}")
            text = tok.apply_chat_template(
                [{"role": "system", "content": SYS},
                 {"role": "user", "content": build_user(q, dids, corpus_subset)}],
                tokenize=False, add_generation_prompt=True)
            lens.append(len(tok(text).input_ids))
        s = sorted(lens)
        out[arm] = {"n": len(lens), "max": s[-1], "median": s[len(s) // 2], "min": s[0],
                    "n_over_limit": sum(l > max_length for l in lens)}
        print(f"[token audit] {arm}: {out[arm]}", flush=True)
    return {"tokenizer": model, "max_length": max_length, "arms": out,
            "renderer": "gen_hf_notrunc.SYS + gen_hf_notrunc.build_user + apply_chat_template"}


# ---------------------------------------------------------------- exposure ---
BOUNDARY_YEAR = 2025


def regime(q: dict) -> str:
    return "fresh" if int(str(q["date"])[:4]) >= BOUNDARY_YEAR else "static"


def exposure_block(qs: list[dict], rk: dict, doc_date: dict, lags=(0, 3), ref=None) -> dict:
    """Context- and query-level exposure of one arm over the questions `qs`.

    A shown context (qid, doc) is "after q_date + L" if publish_date > q_date + L.
    `ref` maps a label to another ranking; ordered and set-level differences of
    the top-k lists against it are counted.
    """
    n_ctx = 0
    ctx_after = {L: 0 for L in lags}
    q_after = {L: 0 for L in lags}
    n_short = 0
    diff = {lab: {"ordered": 0, "set": 0} for lab in (ref or {})}
    for q in qs:
        qid, qd = q["qid"], date_ord(q["date"])
        dl = rk[qid]
        n_ctx += len(dl)
        n_short += len(dl) < 5
        for L in lags:
            k = sum(date_ord(doc_date[d]) > qd + L for d in dl)
            ctx_after[L] += k
            q_after[L] += k > 0
        for lab, other in (ref or {}).items():
            diff[lab]["ordered"] += list(dl) != list(other[qid])
            diff[lab]["set"] += set(dl) != set(other[qid])
    nq = len(qs)
    out = {"n_questions": nq, "n_contexts": n_ctx, "n_questions_lt5_contexts": n_short}
    for L in lags:
        tag = "q_date" if L == 0 else f"q_date_plus{L}"
        out[f"ctx_after_{tag}"] = {"n": ctx_after[L],
                                   "share": round(ctx_after[L] / n_ctx, 6) if n_ctx else None}
        out[f"questions_any_after_{tag}"] = {"n": q_after[L],
                                             "share": round(q_after[L] / nq, 6) if nq else None}
    for lab, v in diff.items():
        out[f"topk_differs_vs_{lab}"] = {"ordered_n": v["ordered"], "set_n": v["set"],
                                         "ordered_share": round(v["ordered"] / nq, 6) if nq else None,
                                         "set_share": round(v["set"] / nq, 6) if nq else None}
    return out


def by_regime(qs: list[dict], fn) -> dict:
    return {"all": fn(qs),
            "static": fn([q for q in qs if regime(q) == "static"]),
            "fresh": fn([q for q in qs if regime(q) == "fresh"])}
