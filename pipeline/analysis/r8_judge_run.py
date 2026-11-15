#!/usr/bin/env python3
"""R8: label the EXACT generator-visible passage text with an independent judge.

WHY. Generators render every retrieved passage as `text[:600]` (DOC_CHARS=600 in
gen_gguf_ranked.py / gen_gguf_notrunc.py / local_generate.py / gen_openai.py).
Human annotators judged 700-character excerpts (Task B pack:
`text.replace("\\n"," ")[:700]`; masked pack: `" ".join(text.split())[:700]`).
So no existing statistic labels relation support on the exact model input, and
humans cannot be re-run. This script labels the exact 600-character inputs with
an independent open-weight judge (Qwen2.5-14B-Instruct, greedy, one label), and
labels the same annotated passages at the 700-character human window as well, so
the judge can be calibrated against the existing human labels on their own
window and the 600-vs-700 difference can be measured with the label held fixed.

THIS IS AN AUTOMATIC JUDGE, NOT A HUMAN STUDY. It establishes what an
independent LLM judge, calibrated against the legacy human labels, says about
the exact shown text. It does not establish human agreement on the exact input.

THREE STAGES (one file, so the same prompt builder ships to the GPU server):

  --build DIR   local, no model. Writes DIR/r8_jobs.jsonl (one row per
                (qid, doc_id, window) with the full prompt text) and
                results/analysis/r8_judge_bundle_manifest.json.
  --label       GPU server. Reads the bundle, scores the three label
                continuations at the first generated token, writes
                <out>/r8_judge_labels.jsonl.
  --ingest F    local. Verifies F against the manifest (every job labelled,
                prompt hashes match) and installs it as
                results/analysis/r8_judge_labels.jsonl.

PAIRS LABELLED (deduplicated on (qid, doc_id, window); the rendered text of a
passage does not depend on whether the context set was masked, so one label
serves both conditions and the membership is recorded in `condition`):

  window w600    every unique (qid, doc_id) in
                   unmasked Hybrid top-5   retrieval.json['hybrid'][q][:5]
                   masked   Hybrid top-5   retrieval_masked.json['hybrid'][q][:5]
                 for all 878 questions, plus every human-annotated passage of
                 either pack (they are subsets of those top-5 lists; verified).
  window w700nl  every human-annotated Task B passage, in the Task B pack's own
                 700-character normalisation.
  window w700ws  every human-annotated masked-pack passage, in that pack's own
                 700-character normalisation.

PROMPT. The passage block is rendered byte-identically to the generator's own
context entry, `f"[{i}] ({publish_date}) {text[:DOC_CHARS]}"` with i=1 (see
gen_gguf_notrunc.build), from the same corpus files. The instruction is the
Task B `supports` rubric of annotation/GUIDELINES.md, quoted, with the
truncation length of the window being judged.

No existing file is modified. Outputs go only to results/analysis/r8_*.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "analysis"
PACK_B = ROOT / "annotation" / "pack_2026-08-02"
PACK_M = ROOT / "annotation" / "masked_support_2026-08-04"

# The generator prompt builder (pipeline/analysis/gen_gguf_notrunc.py::build and
# pipeline/gen_gguf_ranked.py::build) renders each passage with DOC_CHARS=600.
DOC_CHARS = 600
JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LABELS = ("YES", "PARTIAL", "NO")

SYS = ("You are an expert annotator labelling retrieved evidence for a news "
       "question-answering audit. You follow the annotation guidelines exactly "
       "and reply with one label and nothing else.")

# The rubric below is quoted from annotation/GUIDELINES.md, Task B, `supports`.
USER_TMPL = """Task B - evidence support.

Each row is one retrieved document for a question. Decide whether this document contains the answer.

`supports` - does this document contain the answer?

YES      The excerpt states the reference answer, or states it plainly enough that a careful reader would extract it.
PARTIAL  The excerpt is about the right event and narrows it down, but does not state the answer.
NO       The excerpt does not contain the answer.

Judge the excerpt shown, not the document it came from. Excerpts are truncated at {ncut} characters; if the answer is plausibly just past the cut, that is still NO - we are measuring what the model was shown.

Do not look anything up. Judge only from what is on this row.

Question ({qdate}): {question}
Reference answer: {gold}

Evidence excerpt:
{passage}

Reply with exactly one of: YES, PARTIAL, NO."""

WINDOWS = {
    "w600":   {"chars": 600, "render": lambda t: t[:600],
               "note": "raw text[:600] - exactly what the generator was shown"},
    "w700nl": {"chars": 700, "render": lambda t: t.replace("\n", " ")[:700],
               "note": "Task B pack human excerpt (build_annotation_pack.py)"},
    "w700ws": {"chars": 700, "render": lambda t: " ".join(t.split())[:700],
               "note": "masked pack human excerpt (build_masked_pack.py)"},
}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(rows, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_questions():
    return {d["qid"]: d for d in read_jsonl(PROC / "questions.jsonl")}


def load_corpus():
    """doc_id -> record, corpus.jsonl winning over corpus_full.jsonl, exactly as
    e0_human_alignment.load_corpus / support_interaction.load_corpus do."""
    out = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    out.setdefault(d["doc_id"], d)
    return out


def render_passage(doc, window):
    """Byte-identical to the generator's context entry for i=1 when window=w600."""
    return "[1] ({pd}) {txt}".format(pd=doc.get("publish_date", ""),
                                     txt=WINDOWS[window]["render"](doc.get("text", "")))


def build_prompt(q, doc, window):
    passage = render_passage(doc, window)
    user = USER_TMPL.format(ncut=WINDOWS[window]["chars"], qdate=q["date"],
                            question=q["question"], gold=q["gold_answer"],
                            passage=passage)
    return user, passage


# ---------------------------------------------------------------------------
# annotated pairs
# ---------------------------------------------------------------------------
def annotated_pairs():
    """(qid, doc_id) pairs that carry a human `supports` label in each pack.

    Built from the packs' own KEY/CSV files; the validation script rebuilds the
    labels themselves through e0_human_alignment (the same rows)."""
    keyb = {r["item_id"]: r for r in read_jsonl(PACK_B / "item_metadata.jsonl")
            if r.get("task") == "B"}
    b_rows, b_excerpt = [], {}
    with (PACK_B / "task_b_evidence.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = keyb.get(r["item_id"])
            if not k:
                continue
            b_rows.append((k["qid"], r["evidence_id"]))
            b_excerpt[(k["qid"], r["evidence_id"])] = r["evidence_excerpt"]
    keym = {r["item_id"]: r for r in read_jsonl(PACK_M / "item_metadata.jsonl")}
    m_rows, m_excerpt = [], {}
    with (PACK_M / "task_support_masked.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = keym.get(r["item_id"])
            if not k:
                continue
            m_rows.append((k["qid"], k["doc_id"]))
            m_excerpt[(k["qid"], k["doc_id"])] = r["evidence_excerpt"]
    return b_rows, b_excerpt, m_rows, m_excerpt


# ---------------------------------------------------------------------------
# --build
# ---------------------------------------------------------------------------
def cmd_build(args) -> int:
    t0 = time.time()
    qs, corpus = load_questions(), load_corpus()
    ret_u = json.loads((PROC / "retrieval.json").read_text())["hybrid"]
    ret_m = json.loads((PROC / "retrieval_masked.json").read_text())["hybrid"]

    # -- verification: the top-5 lists are the doc_ids the generators recorded --
    checks = {"unmasked_top5_equals_generation_doc_ids": {},
              "masked_top5_equals_generation_doc_ids": {}}
    for model_dir in sorted(PROC.glob("generations*")):
        if not model_dir.is_dir():
            continue
        tag = model_dir.name.replace("generations_", "")
        tag = "haiku" if model_dir.name == "generations" else tag
        for cond, ret, key in (("hybrid", ret_u, "unmasked_top5_equals_generation_doc_ids"),
                               ("hybrid_masked", ret_m, "masked_top5_equals_generation_doc_ids")):
            f = model_dir / f"{cond}.jsonl"
            if not f.is_file():
                continue
            rows = {r["qid"]: list(r.get("doc_ids") or []) for r in read_jsonl(f)}
            ok = sum(1 for q, d in rows.items() if d == list(ret.get(q, []))[:5])
            checks[key][tag] = f"{ok}/{len(rows)}"

    memb = collections.defaultdict(set)          # (qid, doc_id) -> {context names}
    for q in qs:
        for d in ret_u.get(q, [])[:5]:
            memb[(q, d)].add("unmasked_top5")
        for d in ret_m.get(q, [])[:5]:
            memb[(q, d)].add("masked_top5")
    b_rows, b_excerpt, m_rows, m_excerpt = annotated_pairs()
    for k in b_rows:
        memb[k].add("taskb_pack")
    for k in m_rows:
        memb[k].add("masked_pack")

    # -- the packs' excerpts must equal this script's 700-char renderings ------
    exc_check = {}
    for tag, rows, exc, win in (("task_b", b_rows, b_excerpt, "w700nl"),
                                ("masked_pack", m_rows, m_excerpt, "w700ws")):
        ok = sum(1 for k in set(rows)
                 if k[1] in corpus
                 and WINDOWS[win]["render"](corpus[k[1]]["text"]) == exc[k])
        exc_check[tag] = {"window": win, "n_unique_pairs": len(set(rows)),
                          "n_excerpt_matches_rendering": ok}

    jobs = []
    for (qid, did), ctx in sorted(memb.items()):
        if qid not in qs or did not in corpus:
            continue
        wins = ["w600"]
        if "taskb_pack" in ctx:
            wins.append("w700nl")
        if "masked_pack" in ctx:
            wins.append("w700ws")
        for w in wins:
            user, passage = build_prompt(qs[qid], corpus[did], w)
            jobs.append({
                "job_id": sha(f"{qid}|{did}|{w}")[:16],
                "qid": qid, "doc_id": did, "window": w,
                "condition": "|".join(sorted(ctx)),
                "in_unmasked_top5": "unmasked_top5" in ctx,
                "in_masked_top5": "masked_top5" in ctx,
                "in_taskb_pack": "taskb_pack" in ctx,
                "in_masked_pack": "masked_pack" in ctx,
                "premise_sha256": sha(passage),
                "prompt_sha256": sha(SYS + "\n" + user),
                "system": SYS, "user": user,
            })

    bundle = Path(args.build)
    bundle.mkdir(parents=True, exist_ok=True)
    write_jsonl(jobs, bundle / "r8_jobs.jsonl")

    by_win = collections.Counter(j["window"] for j in jobs)
    manifest = {
        "step": "R8 build -- judge prompts on the exact generator-visible text",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "judge_model_requested": JUDGE_MODEL, "fallback_model": FALLBACK_MODEL,
        "doc_chars": DOC_CHARS,
        "prompt": {"system": SYS, "user_template": USER_TMPL,
                   "passage_render": "f'[1] ({publish_date}) {window(text)}'  "
                                     "(i=1 form of gen_gguf_notrunc.build)",
                   "rubric_source": "annotation/GUIDELINES.md, Task B, `supports`"},
        "windows": {k: {"chars": v["chars"], "note": v["note"]} for k, v in WINDOWS.items()},
        "n_jobs": len(jobs), "n_by_window": dict(by_win),
        "n_unique_qid_doc": len({(j["qid"], j["doc_id"]) for j in jobs}),
        "n_questions": len({j["qid"] for j in jobs}),
        "context_membership_counts": dict(collections.Counter(
            j["condition"] for j in jobs if j["window"] == "w600")),
        "dedupe_note": ("the rendered passage text does not depend on the masked/unmasked "
                        "condition, so one (qid, doc_id, window) label serves every context "
                        "set that contains the pair; `condition` lists the memberships"),
        "pairs_per_context": {
            "unmasked_top5": sum(1 for j in jobs if j["window"] == "w600" and j["in_unmasked_top5"]),
            "masked_top5": sum(1 for j in jobs if j["window"] == "w600" and j["in_masked_top5"]),
            "taskb_pack": sum(1 for j in jobs if j["window"] == "w600" and j["in_taskb_pack"]),
            "masked_pack": sum(1 for j in jobs if j["window"] == "w600" and j["in_masked_pack"]),
            "unmasked_and_masked_top5": sum(1 for j in jobs if j["window"] == "w600"
                                            and j["in_unmasked_top5"] and j["in_masked_top5"]),
        },
        "annotated_pairs_inside_shown_top5": {
            "taskb_in_unmasked_top5": sum(1 for j in jobs if j["window"] == "w600"
                                          and j["in_taskb_pack"] and j["in_unmasked_top5"]),
            "taskb_total": sum(1 for j in jobs if j["window"] == "w600" and j["in_taskb_pack"]),
            "masked_pack_in_masked_top5": sum(1 for j in jobs if j["window"] == "w600"
                                              and j["in_masked_pack"] and j["in_masked_top5"]),
            "masked_pack_total": sum(1 for j in jobs if j["window"] == "w600" and j["in_masked_pack"]),
        },
        "human_pack_excerpt_reproduction": exc_check,
        "doc_ids_verification": checks,
        "input_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in (
            PROC / "questions.jsonl", PROC / "corpus.jsonl", PROC / "retrieval.json",
            PROC / "retrieval_masked.json",
            PACK_B / "item_metadata.jsonl", PACK_M / "item_metadata.jsonl",
            PACK_B / "task_b_evidence.csv", PACK_M / "task_support_masked.csv")},
        "bundle_sha256": file_sha(bundle / "r8_jobs.jsonl"),
        "job_ids_sha256": sha("".join(j["job_id"] for j in jobs)),
        "build_seconds": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r8_judge_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("prompt", "input_sha256")}, indent=2)[:4000])
    print("wrote", bundle / "r8_jobs.jsonl", "and", OUT / "r8_judge_bundle_manifest.json")
    return 0


# ---------------------------------------------------------------------------
# --label  (GPU server)
# ---------------------------------------------------------------------------
def cmd_label(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    jobs = read_jsonl(Path(args.bundle) / "r8_jobs.jsonl")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists() and not args.overwrite:
        done = {r["job_id"] for r in read_jsonl(out_path)}
    todo = [j for j in jobs if j["job_id"] not in done]
    print(f"jobs={len(jobs)} done={len(done)} todo={len(todo)}", flush=True)

    name = args.model
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.eval()

    first_ids = {}
    for lab in LABELS:
        ids = tok.encode(lab, add_special_tokens=False)
        first_ids[lab] = ids[0]
    assert len({*first_ids.values()}) == len(LABELS), first_ids
    id2lab = {v: k for k, v in first_ids.items()}
    cand = torch.tensor([first_ids[l] for l in LABELS], device="cuda")
    print("label first-token ids:", first_ids, flush=True)

    def chat(j):
        return tok.apply_chat_template(
            [{"role": "system", "content": j["system"]},
             {"role": "user", "content": j["user"]}],
            tokenize=False, add_generation_prompt=True)

    # deterministic batching: sort by token length, fixed chunks
    enc_len = [(len(tok(chat(j), add_special_tokens=False)["input_ids"]), i)
               for i, j in enumerate(todo)]
    enc_len.sort()
    order = [i for _, i in enc_len]
    print(f"prompt tokens: min={enc_len[0][0]} max={enc_len[-1][0]}", flush=True)

    n_written = 0
    fh = out_path.open("a" if done and not args.overwrite else "w", encoding="utf-8")
    with torch.inference_mode():
        for s in range(0, len(order), args.batch):
            chunk = [todo[i] for i in order[s:s + args.batch]]
            enc = tok([chat(j) for j in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            try:
                o = model(**enc, logits_to_keep=1)
            except TypeError:
                o = model(**enc)
            logits = o.logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            top = logits.argmax(dim=-1)
            csel = probs[:, cand]
            norm = csel / csel.sum(dim=-1, keepdim=True)
            pick = norm.argmax(dim=-1)
            for k, j in enumerate(chunk):
                lab = LABELS[int(pick[k])]
                top_id = int(top[k])
                fh.write(json.dumps({
                    "job_id": j["job_id"], "qid": j["qid"], "doc_id": j["doc_id"],
                    "condition": j["condition"], "window": j["window"],
                    "label": lab,
                    "prob": round(float(norm[k, int(pick[k])]), 6),
                    "probs": {l: round(float(norm[k, m]), 6) for m, l in enumerate(LABELS)},
                    "prob_unnormalised": round(float(csel[k, int(pick[k])]), 6),
                    "label_mass": round(float(csel[k].sum()), 6),
                    "argmax_token_id": top_id,
                    "argmax_token": tok.decode([top_id]),
                    "argmax_in_label_set": bool(top_id in id2lab),
                    "argmax_label": id2lab.get(top_id),
                    "premise_sha256": j["premise_sha256"],
                    "prompt_sha256": j["prompt_sha256"],
                    "model": name,
                }, ensure_ascii=False) + "\n")
                n_written += 1
            fh.flush()
            if (s // args.batch) % 25 == 0:
                print(f"  {s + len(chunk)}/{len(order)}  {time.time() - t0:.0f}s", flush=True)
    fh.close()

    # -- free-generation sanity check on a fixed subsample --------------------
    import random
    rng = random.Random(20260920)
    sub = rng.sample(todo, min(args.gen_check, len(todo)))
    gen_rows = []
    with torch.inference_mode():
        for s in range(0, len(sub), args.batch):
            chunk = sub[s:s + args.batch]
            enc = tok([chat(j) for j in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            g = model.generate(**enc, max_new_tokens=6, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            for k, j in enumerate(chunk):
                txt = tok.decode(g[k][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                gen_rows.append({"job_id": j["job_id"], "text": txt,
                                 "parsed": txt.upper().strip(" .:*").split()[0]
                                 if txt.strip() else ""})
    write_jsonl(gen_rows, out_path.parent / "r8_judge_gencheck.jsonl")

    rep = {"model": name, "n_jobs": len(jobs), "n_labelled_now": n_written,
           "seconds": round(time.time() - t0, 1), "batch": args.batch,
           "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0),
           "label_first_token_ids": first_ids,
           "gen_check_n": len(gen_rows)}
    (out_path.parent / "r8_judge_run_report.json").write_text(
        json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)
    return 0


# ---------------------------------------------------------------------------
# --ingest
# ---------------------------------------------------------------------------
def cmd_ingest(args) -> int:
    man = json.loads((OUT / "r8_judge_bundle_manifest.json").read_text())
    jobs = {j["job_id"]: j for j in read_jsonl(Path(args.bundle) / "r8_jobs.jsonl")}
    rows = read_jsonl(Path(args.ingest))
    seen, bad_hash, dup = set(), 0, 0
    keep = []
    for r in rows:
        if r["job_id"] in seen:
            dup += 1
            continue
        seen.add(r["job_id"])
        j = jobs.get(r["job_id"])
        if j is None or j["prompt_sha256"] != r.get("prompt_sha256") \
                or j["premise_sha256"] != r.get("premise_sha256"):
            bad_hash += 1
            continue
        keep.append({k: r[k] for k in (
            "qid", "doc_id", "condition", "window", "label", "prob", "probs",
            "prob_unnormalised", "label_mass", "argmax_token", "argmax_in_label_set",
            "argmax_label", "premise_sha256", "prompt_sha256", "model") if k in r})
    missing = sorted(set(jobs) - seen)
    keep.sort(key=lambda r: (r["qid"], r["doc_id"], r["window"]))
    write_jsonl(keep, OUT / "r8_judge_labels.jsonl")
    rep = {"n_manifest_jobs": man["n_jobs"], "n_rows_in": len(rows),
           "n_duplicates_dropped": dup, "n_hash_mismatch_dropped": bad_hash,
           "n_written": len(keep), "n_missing": len(missing),
           "missing_job_ids_first20": missing[:20],
           "label_distribution": dict(collections.Counter(r["label"] for r in keep)),
           "label_distribution_by_window": {
               w: dict(collections.Counter(r["label"] for r in keep if r["window"] == w))
               for w in sorted({r["window"] for r in keep})},
           "n_argmax_outside_label_set": sum(1 for r in keep if not r["argmax_in_label_set"]),
           "n_argmax_disagrees_with_constrained": sum(
               1 for r in keep if r["argmax_in_label_set"] and r["argmax_label"] != r["label"]),
           "models": sorted({r["model"] for r in keep}),
           "output_sha256": file_sha(OUT / "r8_judge_labels.jsonl")}
    (OUT / "r8_judge_ingest_report.json").write_text(json.dumps(rep, indent=2) + "\n",
                                                     encoding="utf-8")
    print(json.dumps(rep, indent=2))
    assert not missing, f"{len(missing)} jobs unlabelled"
    assert bad_hash == 0, f"{bad_hash} rows failed the prompt/premise hash check"
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", help="bundle directory to write")
    ap.add_argument("--label", action="store_true", help="run the judge (GPU server)")
    ap.add_argument("--ingest", help="server label file to verify and install")
    ap.add_argument("--bundle", default=str(OUT / "r8_bundle"))
    ap.add_argument("--out", default=str(OUT / "r8_raw" / "r8_judge_labels_raw.jsonl"))
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gen-check", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.build:
        return cmd_build(args)
    if args.label:
        return cmd_label(args)
    if args.ingest:
        return cmd_ingest(args)
    ap.error("one of --build / --label / --ingest is required")


if __name__ == "__main__":
    raise SystemExit(main())
