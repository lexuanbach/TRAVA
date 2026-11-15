#!/usr/bin/env python3
"""Does the evidence ENTAIL the answer, or merely contain the string?

`support_conditioned.py` labels a retrieved context "supported" when the gold
answer string appears in it. The review is right that this is an incomplete
label in both directions:

  MISSES   paraphrase, alias, and entailment. "An ion with more electrons than
           protons carries a net negative charge" supports "negatively charged"
           without containing it.
  FIRES    inside contradictory or irrelevant passages. A document saying
           "Ron Klain, who preceded Jeff Zients" contains the string "Jeff
           Zients" while asserting he is the successor, not the incumbent.

This relabels the same contexts with an NLI model (DeBERTa-v3-large-MNLI),
asking whether the evidence entails the proposition "the answer to <question>
is <gold>". That gives three labels per context -- entails, neutral,
contradicts -- against the binary string label, so we can measure exactly how
often the two disagree and in which direction, and re-run the mechanism
analysis under the semantic label.

Both labels remain ORACLE diagnostics: each uses the gold answer. Neither is a
deployable gate, and this script makes no routing recommendation. Its purpose is
to test how much the paper's mechanism claim depends on the definition of
support.

Outputs results/nli_support.json (and caches per-context labels).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from metrics import gold_in_text, token_f1                    # noqa: E402

PROC = ROOT / "data/processed"
RES = ROOT / "results"
CACHE = RES / "nli_support_labels.jsonl"
BOUNDARY_YEAR = 2025
CORRECT = 0.5
NLI_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def quiz_of(qid):
    return qid.split("_")[0]


def load_questions():
    return {d["qid"]: d for d in
            (json.loads(l) for l in
             (PROC / "questions.jsonl").read_text().splitlines() if l.strip())}


def load_corpus():
    out = {}
    for name in ("corpus.jsonl", "corpus_full.jsonl"):
        f = PROC / name
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    out.setdefault(d["doc_id"], d.get("text", ""))
    return out


def load_gen(model, cond):
    f = PROC / f"generations_{model}" / f"{cond}.jsonl"
    if not f.is_file():
        return {}
    return {r["qid"]: r for r in
            (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def paired_boot(d, B, seed):
    if len(d) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(statistics.fmean([d[rng.randrange(len(d))] for _ in d])
                   for _ in range(B))
    le = sum(1 for x in draws if x <= 0)
    return {"mean": round(statistics.fmean(d), 3),
            "ci95": [round(draws[int(0.025 * B)], 3),
                     round(draws[int(0.975 * B)], 3)],
            "p_plus_one": round(min(1.0, 2.0 * min(le + 1, B - le + 1) / (B + 1)), 5),
            "n": len(d)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-questions", type=int, default=900,
                    help="questions to relabel (each has up to 5 contexts)")
    ap.add_argument("--output", type=Path, default=RES / "nli_support.json")
    args = ap.parse_args()

    qs = load_questions()
    corpus = load_corpus()
    # one context set per question is enough: retrieval is model-independent
    # for a fixed condition, so we relabel per (qid, doc) and reuse across models
    ref_model = "qwen32b"
    gen = load_gen(ref_model, "hybrid")
    qids = sorted(q for q in gen if q in qs)
    rng = random.Random(args.seed)
    if len(qids) > args.max_questions:
        qids = sorted(rng.sample(qids, args.max_questions))

    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["qid"], r["doc_id"])] = r

    todo = [(q, d) for q in qids for d in (gen[q].get("doc_ids") or [])
            if (q, d) not in done]
    print(f"{len(qids)} questions, {len(todo)} contexts to label", flush=True)

    if todo:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        dev = args.device if (args.device != "mps" or
                              torch.backends.mps.is_available()) else "cpu"
        tok = AutoTokenizer.from_pretrained(NLI_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(NLI_NAME).to(dev).eval()
        id2 = {int(k): str(v).lower() for k, v in (mdl.config.id2label or {}).items()}
        print(f"NLI loaded on {dev}: {id2}", flush=True)

        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a") as fh:
            for n, (qid, did) in enumerate(todo):
                q = qs[qid]
                prem = (corpus.get(did) or "")[:1500]
                if not prem:
                    continue
                hyp = f"The answer to \"{q['question']}\" is {q['gold_answer']}."
                with torch.no_grad():
                    enc = tok(prem, hyp, return_tensors="pt", truncation=True,
                              max_length=384).to(dev)
                    probs = torch.softmax(mdl(**enc).logits[0], -1).tolist()
                lab = {id2.get(i, str(i)): round(p, 4)
                       for i, p in enumerate(probs)}
                best = max(lab, key=lab.get)
                rec = {"qid": qid, "doc_id": did, "nli_label": best,
                       "probs": lab,
                       "string_support": bool(gold_in_text(q["gold_answer"],
                                                           corpus.get(did, "")))}
                done[(qid, did)] = rec
                fh.write(json.dumps(rec) + "\n")
                if n % 200 == 0:
                    fh.flush()
                    print(f"  {n}/{len(todo)}", flush=True)

    # ---------------- agreement between the two support labels --------------
    ctx = [r for r in done.values() if r["qid"] in set(qids)]
    tab = collections.Counter((r["string_support"],
                               r["nli_label"].startswith("entail")) for r in ctx)
    n = sum(tab.values())
    out = {
        "scope": "Semantic (entailment) support labels for retrieved contexts, "
                 "against the exact-string label used in support_conditioned.py.",
        "both_are_oracles": ("both labels use the gold answer; neither is a "
                             "deployable gate and no routing recommendation "
                             "follows from either"),
        "nli_model": NLI_NAME,
        "n_contexts_labelled": n,
        "n_questions": len(qids),
        "context_level_agreement": {
            "both_support": tab[(True, True)],
            "string_only": tab[(True, False)],
            "nli_only": tab[(False, True)],
            "neither": tab[(False, False)],
            "raw_agreement": round((tab[(True, True)] + tab[(False, False)]) / n, 4) if n else None,
            "string_supported_but_not_entailed": round(
                tab[(True, False)] / max(tab[(True, True)] + tab[(True, False)], 1), 4),
            "entailed_but_no_string": round(
                tab[(False, True)] / max(tab[(False, True)] + tab[(False, False)], 1), 4),
        },
        "nli_label_distribution": dict(collections.Counter(r["nli_label"] for r in ctx)),
        "contradiction_inside_string_support": round(
            sum(1 for r in ctx if r["string_support"]
                and r["nli_label"].startswith("contradict"))
            / max(sum(1 for r in ctx if r["string_support"]), 1), 4),
    }
    ca = out["context_level_agreement"]
    print(f"\ncontexts {n}: both {ca['both_support']}, string-only "
          f"{ca['string_only']}, nli-only {ca['nli_only']}, neither {ca['neither']}")
    print(f"raw agreement {ca['raw_agreement']}")
    print(f"string-supported but NOT entailed: {ca['string_supported_but_not_entailed']}")
    print(f"entailed with NO string match:     {ca['entailed_but_no_string']}")
    print(f"contradiction inside string-support: {out['contradiction_inside_string_support']}")

    # ---------------- does the mechanism claim survive the relabel? ---------
    q_nli = collections.defaultdict(bool)
    q_str = collections.defaultdict(bool)
    for r in ctx:
        q_nli[r["qid"]] |= r["nli_label"].startswith("entail")
        q_str[r["qid"]] |= r["string_support"]

    models = sorted(d.name.replace("generations_", "")
                    for d in PROC.glob("generations_*") if d.is_dir())
    per_model = {}
    for m in models:
        cb, rt = load_gen(m, "closed_book"), load_gen(m, "hybrid")
        if not cb or not rt:
            continue
        rec = {}
        for label, sup in (("string", q_str), ("nli", q_nli)):
            for regime in ("static", "fresh"):
                d = []
                for qid in qids:
                    q = qs[qid]
                    if (q["year"] >= BOUNDARY_YEAR) != (regime == "fresh"):
                        continue
                    if qid not in rt or qid not in cb or not sup[qid]:
                        continue
                    a_rt = token_f1(rt[qid]["prediction"] or "", q["gold_answer"]) >= CORRECT
                    a_cb = token_f1(cb[qid]["prediction"] or "", q["gold_answer"]) >= CORRECT
                    d.append(100.0 * (a_rt - a_cb))
                r = paired_boot(d, args.B, args.seed)
                if r:
                    rec[f"{label}_{regime}"] = r
        if rec:
            per_model[m] = rec
    out["gain_given_support_by_label"] = per_model

    strong = ["mistral_large", "sonnet", "gpt4omini"]
    both_nonneg = all(
        per_model.get(m, {}).get("nli_static", {}).get("mean", 0) >= -1e-9
        for m in strong if m in per_model)
    out["verdict"] = {
        "mechanism_survives_semantic_relabel": bool(both_nonneg),
        "reading": (
            "If the Static gain given support stays non-negative for the strong "
            "models under the entailment label as well as the string label, the "
            "paper's descriptive claim -- harm is concentrated in the "
            "unsupported subset -- does not depend on which support definition "
            "is used. If it flips, the claim was an artifact of string "
            "matching."),
    }
    print(f"\n{'model':16s}{'string static':>22s}{'nli static':>22s}")
    for m in strong:
        r = per_model.get(m, {})
        f = lambda k: (f"{r[k]['mean']:+7.2f} {str(r[k]['ci95'])}" if k in r else "-")
        print(f"{m:16s}{f('string_static'):>22s}{f('nli_static'):>22s}")
    print(f"\nmechanism survives semantic relabel: {both_nonneg}")
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
