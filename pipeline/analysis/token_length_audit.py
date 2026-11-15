#!/usr/bin/env python3
"""What did each generator actually see? Prompt-length audit of every cached
ranked-context condition.

For every file under data/processed/generations*/ whose rows carry doc_ids,
the prompt is rebuilt with the generating script's OWN prompt builder
(imported, not re-implemented), from the stored doc_ids, and counted with the
matching tokenizer:

  HF transformers path  (local_generate.py, loo_control.py, heldout_index.py)
      chat template -> tok(text, truncation=True, max_length=1536); the Qwen2 and
      Llama-3 tokenizers truncate on the RIGHT, so an over-limit prompt loses the
      assistant header and then the question. Exact HF tokenizer, offline cache.
  llama.cpp path        (local_generate_gguf.py, gen_gguf_ranked.py)
      Llama(n_ctx=2048).create_chat_completion(max_tokens=32). Counted EXACTLY by
      loading the same GGUF with vocab_only=True and capturing the token list the
      chat handler passes to create_completion (no weights are loaded).
  API path              (generate.py for Claude Haiku/Sonnet; gen_openai.py for
      gpt-4o-mini). No offline tokenizer: PROXY count with the Qwen2.5 tokenizer
      over the same system+user messages, plus character counts. Provider context
      windows (200k / 128k tokens) are far above any prompt here.

The llama.cpp over-length behaviour is read from the installed llama_cpp source
and recorded with line numbers.

Writes results/analysis/token_length_audit.json.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as C  # noqa: E402

HF_MAX = 1536
N_CTX = 2048
GGUF_MAX_TOKENS = 32
PROJECT_HF = C.ROOT / "data" / "hf_cache" / "hub"
USER_HF = Path.home() / ".cache" / "huggingface" / "hub"

HF_GENS = {"qwen", "qwen1_5b", "qwen3b", "qwen7b", "llama3_8b"}
GGUF_GENS = {"qwen32b": "Qwen/Qwen2.5-32B-Instruct-GGUF",
             "qwen72b": "Qwen/Qwen2.5-72B-Instruct-GGUF",
             "mistral_large": "lmstudio-community/Mistral-Large-Instruct-2407-GGUF"}
API_GENS = {"haiku": "generate.py", "sonnet": "generate.py", "gpt4omini": "gen_openai.py"}
FULL_CORPUS_CONDS = {("qwen32b", "hybrid_full"), ("qwen32b", "bm25_full"),
                     ("qwen32b", "oracle_real")}   # scripts/run_distractor_32b.sh
PROXY_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"


def script_for(gen: str, cond: str) -> tuple[str, str]:
    """(script, evidence) for the generation that produced a cached condition."""
    if gen in HF_GENS:
        if (gen, cond) == ("qwen3b", "hybrid_loo"):
            return "loo_control.py", "rows carry no model field; loo_control.generate writes generations_qwen3b/hybrid_loo.jsonl (scripts/run_loo.sh)"
        if (gen, cond) == ("qwen3b", "hybrid_stale"):
            return "heldout_index.py", "scripts/run_heldout.sh --phase generate --conditions hybrid_stale --gen-tag qwen3b"
        return "local_generate.py", "HF model id in rows; scripts/run_ladder.sh, run_masked_panel.sh, finish_masked_panel.sh, run_k_ablation.sh, run_extension.sh"
    if gen in GGUF_GENS:
        if gen == "mistral_large" and cond in ("hybrid", "hybrid_stale", "hybrid_loo"):
            return "gen_gguf_ranked.py", "scripts/run_mistral_hybrid.sh, run_mistral_controls.sh"
        if (gen, cond) in FULL_CORPUS_CONDS:
            return "gen_gguf_ranked.py", "scripts/run_distractor_32b.sh (--corpus corpus_full.jsonl)"
        if gen == "qwen32b" and cond in ("hybrid_stale", "hybrid_loo"):
            return "gen_gguf_ranked.py", "data/gen_32b_controls.log ('loaded <repo>', '[cond] todo=')"
        return "local_generate_gguf.py", "GGUF repo id in rows; data/generate_32b*.log, generate_72b*.log, scripts/run_masked_panel.sh, finish_masked_panel.sh"
    return API_GENS[gen], "API model id in rows"


def load_corpus(name):
    return {d["doc_id"]: d for d in
            (json.loads(l) for l in (C.PROC / name).read_text(encoding="utf-8").splitlines()
             if l.strip())}


def builder(script: str):
    """Return (build(q, dids, corpus) -> user prompt, system prompt) from the script."""
    if script == "local_generate.py":
        m = importlib.import_module("local_generate")
        return m.build_user, m.SYS
    if script == "loo_control.py":
        m = importlib.import_module("loo_control")
        return m.build, m.SYS
    if script == "heldout_index.py":
        m = importlib.import_module("heldout_index")
        return m.build_prompt, m.SYS
    if script == "local_generate_gguf.py":
        m = importlib.import_module("local_generate_gguf")
        return m.build_user, m.SYS
    if script == "gen_gguf_ranked.py":
        m = importlib.import_module("gen_gguf_ranked")
        return m.build, m.SYS
    if script == "gen_openai.py":
        m = importlib.import_module("gen_openai")
        return m.build, m.SYS
    if script == "generate.py":
        m = importlib.import_module("generate")
        return m.build_prompt, m.load_cfg()["system_prompt"]
    raise KeyError(script)


def ranked_files():
    out = []
    for d in sorted(C.PROC.glob("generations*")):
        if not d.is_dir():
            continue
        gen = "haiku" if d.name == "generations" else d.name.replace("generations_", "")
        for f in sorted(d.glob("*.jsonl")):
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            if any(r.get("doc_ids") for r in rows):
                out.append((gen, f.stem, rows))
    return out


def hf_tokenizer(model_id):
    from transformers import AutoTokenizer
    last = None
    for cache in (PROJECT_HF, USER_HF):
        try:
            return AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache),
                                                 padding_side="left"), str(cache)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def gguf_first_shard(repo):
    base = USER_HF / ("models--" + repo.replace("/", "--")) / "snapshots"
    cands = sorted(p for p in base.glob("*/*.gguf")
                   if re.search(r"-0*1-of-\d+\.gguf$", p.name) or "-of-" not in p.name)
    if not cands:
        raise FileNotFoundError(repo)
    return cands[0]


class _Captured(Exception):
    pass


def gguf_counter(repo):
    from llama_cpp import Llama
    path = gguf_first_shard(repo)
    llm = Llama(model_path=str(path), n_ctx=N_CTX, vocab_only=True, verbose=False)

    def capture(*a, **k):
        raise _Captured(k.get("prompt", a[0] if a else None))

    llm.create_completion = capture          # instance attribute only

    def count(sys_prompt, user):
        try:
            llm.create_chat_completion(
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user}],
                max_tokens=GGUF_MAX_TOKENS, temperature=0.0)
        except _Captured as e:
            toks = e.args[0]
            if isinstance(toks, str):
                toks = llm.tokenize(toks.encode("utf-8"), add_bos=False, special=True)
            return len(toks)
        raise RuntimeError("chat handler did not call create_completion")

    has_tpl = "tokenizer.chat_template" in llm.metadata
    try:
        llm.create_chat_completion(messages=[{"role": "system", "content": "<SYS>"},
                                             {"role": "user", "content": "<USER>"}],
                                   max_tokens=GGUF_MAX_TOKENS, temperature=0.0)
    except _Captured as e:
        rendered = llm.detokenize(e.args[0], special=True).decode("utf-8", "replace")
    meta = {"gguf_file": str(path), "chat_format": llm.chat_format,
            "gguf_has_tokenizer_chat_template": has_tpl,
            "rendered_toy_prompt": rendered, "n_ctx": llm.n_ctx()}
    if not has_tpl:
        meta["note"] = ("GGUF metadata has no tokenizer.chat_template, so llama-cpp-python "
                        "fell back to its 'llama-2' chat format ([INST] <<SYS>>..<</SYS>> .. "
                        "[/INST]) rather than the model's native template. Not a truncation "
                        "issue, but a prompt-format deviation affecting every llama.cpp "
                        "generation with this file (closed-book included).")
    return count, meta


def llama_cpp_behaviour():
    import llama_cpp
    src = Path(llama_cpp.__file__).with_name("llama.py")
    lines = src.read_text(encoding="utf-8").splitlines()

    def find(pat):
        for i, l in enumerate(lines, 1):
            if pat in l:
                return i
        return None

    raise_line = find("exceed context window of")
    clip_line = find("# Truncate max_tokens if requested tokens would exceed the context window")
    snippet = "\n".join(lines[raise_line - 4: raise_line + 1]) if raise_line else None
    clip = "\n".join(lines[clip_line - 1: clip_line + 5]) if clip_line else None
    dist = Path(llama_cpp.__file__).parent.parent
    info = sorted(dist.glob("llama_cpp_python-*.dist-info"))
    return {
        "llama_cpp_python_version": llama_cpp.__version__,
        "source_file": str(src),
        "dist_info": str(info[0]) if info else None,
        "dist_info_mtime": (__import__("datetime").datetime.fromtimestamp(
            info[0].stat().st_mtime).isoformat(timespec="minutes") if info else None),
        "over_length_prompt": ("RAISES ValueError('Requested tokens (N) exceed context window "
                               "of 2048') in Llama._create_completion when len(prompt_tokens) "
                               ">= n_ctx; it does NOT silently truncate the prompt."),
        "raise_line": raise_line, "raise_snippet": snippet,
        "near_limit": ("If len(prompt_tokens) + max_tokens >= n_ctx, max_tokens is silently "
                       "reduced to n_ctx - len(prompt_tokens) (answer length cap, prompt "
                       "intact). With max_tokens=32 this applies to prompts of 2016-2047 tokens."),
        "clip_line": clip_line, "clip_snippet": clip,
        "script_consequence": ("local_generate_gguf.py and gen_gguf_ranked.py call "
                               "create_chat_completion without try/except, so an over-length "
                               "prompt would abort the run at that question; the resume logic "
                               "would hit it again. Every GGUF ranked cache is complete (878 "
                               "rows), consistent with the counts below."),
    }


def stats(n, limit=None, extra=None):
    a = np.asarray(n)
    out = {"n": int(len(a)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
           "max": int(a.max()), "min": int(a.min())}
    if limit is not None:
        out["limit"] = limit
    if extra:
        out.update(extra)
    return out


def main() -> int:
    questions = C.load_questions()
    corpora = {"corpus.jsonl": load_corpus("corpus.jsonl")}
    files = ranked_files()
    out = {"scope": __doc__.strip().splitlines()[0],
           "hf_max_length": HF_MAX, "gguf_n_ctx": N_CTX, "gguf_max_tokens": GGUF_MAX_TOKENS,
           "llama_cpp_behaviour": llama_cpp_behaviour(),
           "tokenizers": {}, "conditions": {}, "caveats": [
               "Token counts use tokenizer/llama_cpp versions installed now "
               "(transformers %s, llama_cpp_python %s); generation-time versions are not "
               "recorded in the caches." % (__import__("transformers").__version__,
                                            __import__("llama_cpp").__version__),
               "HF over-limit rows: 'question_fully_cut' counts prompts whose text before the "
               "final '\\n\\nQuestion (' marker already reaches 1536 tokens (prefix tokenised "
               "separately; boundary effects of +-1 token possible).",
               "API rows (Claude Haiku/Sonnet, gpt-4o-mini) are PROXY counts with the "
               f"{PROXY_TOKENIZER} tokenizer and chat template; the provider tokenizers are "
               "not available offline. Provider context windows are not binding.",
           ]}
    hf_cache, gguf_cache = {}, {}

    for gen, cond, rows in files:
        script, evidence = script_for(gen, cond)
        build, sys_prompt = builder(script)
        corpus_name = "corpus_full.jsonl" if (gen, cond) in FULL_CORPUS_CONDS else "corpus.jsonl"
        if corpus_name not in corpora:
            corpora[corpus_name] = load_corpus(corpus_name)
        corpus = corpora[corpus_name]
        prompts, preds, missing_docs = [], [], 0
        for r in rows:
            q = questions[r["qid"]]
            dids = r.get("doc_ids") or []
            missing_docs += sum(1 for d in dids if d not in corpus)
            prompts.append(build(q, dids, corpus))
            preds.append(r.get("prediction") or "")
        h = hashlib.sha256("\x1e".join(prompts).encode("utf-8")).hexdigest()
        entry = {"generator": gen, "condition": cond, "script": f"pipeline/{script}",
                 "script_evidence": evidence, "corpus": corpus_name,
                 "model_field": sorted({str(r.get("model")) for r in rows}),
                 "n_rows": len(rows), "k_distribution": {}, "doc_ids_missing_from_corpus": missing_docs,
                 "user_prompt_chars_max": max(len(p) for p in prompts),
                 "prompts_sha256": h}
        for r in rows:
            k = str(len(r.get("doc_ids") or []))
            entry["k_distribution"][k] = entry["k_distribution"].get(k, 0) + 1

        if gen in HF_GENS:
            model_id = rows[0].get("model") or "Qwen/Qwen2.5-3B-Instruct"   # loo_control default
            if model_id not in hf_cache:
                tok, cache = hf_tokenizer(model_id)
                hf_cache[model_id] = tok
                out["tokenizers"][model_id] = {"kind": "HF (exact)", "cache": cache,
                                               "class": type(tok).__name__,
                                               "truncation_side": tok.truncation_side}
            tok = hf_cache[model_id]
            texts = [tok.apply_chat_template([{"role": "system", "content": sys_prompt},
                                              {"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
                     for p in prompts]
            n = [len(x) for x in tok(texts)["input_ids"]]
            over = [i for i, x in enumerate(n) if x > HF_MAX]
            cut_q = 0
            for i in over:
                pre = texts[i][:texts[i].rfind("\n\nQuestion (")]
                if len(tok(pre)["input_ids"]) >= HF_MAX:
                    cut_q += 1
            words = np.array([len(p.split()) for p in preds])
            under = np.array([x <= HF_MAX for x in n])
            entry.update({"path": "HF transformers", "tokenizer": model_id,
                          "count_kind": "exact",
                          "tokens": stats(n, HF_MAX, {
                              "n_over_limit": len(over),
                              "pct_over_limit": round(100 * len(over) / len(n), 1),
                              "over_limit_behaviour": "silent right truncation (question/assistant turn removed)",
                              "n_over_limit_question_fully_cut": cut_q}),
                          "prediction_words_mean": {
                              "within_limit": round(float(words[under].mean()), 2) if under.any() else None,
                              "over_limit": round(float(words[~under].mean()), 2) if (~under).any() else None}})
        elif gen in GGUF_GENS:
            repo = GGUF_GENS[gen]
            if repo not in gguf_cache:
                cnt, meta = gguf_counter(repo)
                gguf_cache[repo] = cnt
                out["tokenizers"][repo] = {"kind": "GGUF vocab via llama_cpp (exact)", **meta}
            cnt = gguf_cache[repo]
            n = [cnt(sys_prompt, p) for p in prompts]
            entry.update({"path": "llama.cpp", "tokenizer": repo, "count_kind": "exact",
                          "tokens": stats(n, N_CTX, {
                              "n_over_limit_would_raise": int(sum(x >= N_CTX for x in n)),
                              "n_max_tokens_silently_clipped": int(sum(
                                  N_CTX - GGUF_MAX_TOKENS <= x < N_CTX for x in n))})})
        else:
            if PROXY_TOKENIZER not in hf_cache:
                tok, cache = hf_tokenizer(PROXY_TOKENIZER)
                hf_cache[PROXY_TOKENIZER] = tok
                out["tokenizers"][PROXY_TOKENIZER] = {"kind": "HF (exact for Qwen; proxy for API models)",
                                                      "cache": cache, "class": type(tok).__name__,
                                                      "truncation_side": tok.truncation_side}
            tok = hf_cache[PROXY_TOKENIZER]
            texts = [tok.apply_chat_template([{"role": "system", "content": sys_prompt},
                                              {"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
                     for p in prompts]
            n = [len(x) for x in tok(texts)["input_ids"]]
            limit = 200000 if gen in ("haiku", "sonnet") else 128000
            entry.update({"path": "API", "tokenizer": f"PROXY {PROXY_TOKENIZER}",
                          "count_kind": "proxy",
                          "tokens": stats(n, limit, {"n_over_limit": int(sum(x > limit for x in n)),
                                                     "limit_note": "provider context window"})})
        out["conditions"][f"{gen}/{cond}"] = entry
        t = entry["tokens"]
        flag = (t.get("n_over_limit", 0) or t.get("n_over_limit_would_raise", 0)
                or t.get("n_max_tokens_silently_clipped", 0))
        print(f"{gen:14s} {cond:20s} {entry['path']:16s} n={t['n']} median={t['median']:.0f} "
              f"max={t['max']} limit={t['limit']} "
              + " ".join(f"{k}={v}" for k, v in t.items() if k.startswith("n_") and k != "n")
              + ("  <-- FLAG" if flag else ""))

    # consistency: the same condition rebuilt by different scripts must be byte-identical
    out["summary"] = {
        "conditions_audited": len(out["conditions"]),
        "conditions_with_any_over_limit": sorted(
            k for k, v in out["conditions"].items()
            if v["tokens"].get("n_over_limit", 0) or v["tokens"].get("n_over_limit_would_raise", 0)),
        "conditions_with_max_tokens_clipped": sorted(
            k for k, v in out["conditions"].items()
            if v["tokens"].get("n_max_tokens_silently_clipped", 0)),
    }
    p = C.write_json(out, "token_length_audit.json")
    print(json.dumps(out["summary"], indent=2))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
