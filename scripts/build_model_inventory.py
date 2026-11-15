#!/usr/bin/env python3
"""Build machine-readable model and analysis-panel inventory from cached inputs."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed'
sys.path.insert(0,str(ROOT/'pipeline'))
from controlled_cutoff import FAMILY
PRETTY = {
    "qwen": "0.5B", "qwen1_5b": "1.5B", "qwen3b": "3B", "qwen7b": "7B",
    "llama3_8b": "L3-8B", "qwen32b": "32B", "qwen72b": "72B",
    "mistral_large": "123B", "": "Haiku", "sonnet": "Sonnet",
    "gpt4omini": "4o-mini",
}
ORDER = list(PRETTY)

# the controlled family varies one factor at a time and is reported separately
CONTROLLED = {"llama32_1b": "L3.2-1B", "llama32_3b": "L3.2-3B",
              "llama31_8b": "L3.1-8B", "llama31_8b_q4": "L3.1-8B-q4",
              "llama31_70b": "L3.1-70B", "llama3_8b": "L3-8B"}

LADDER = ["qwen", "qwen1_5b", "qwen3b", "qwen7b", "qwen32b", "qwen72b",
          "mistral_large", "", "sonnet"]          # the nine of tab:cross-model
# The Hybrid panel is NOT the ladder plus GPT-4o-mini: Qwen2.5-0.5B was never
# run under retrieval, and Llama-3-8B was, so declaring it as LADDER+4o-mini
# both omitted a member and invented one. Declared explicitly and verified.
HYBRID = ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
          "mistral_large", "", "sonnet", "gpt4omini"]

# (label, estimand, declared members, required conditions, expected questions).
# Membership is DECLARED and then verified against disk, rather than inferred
# from whatever caches happen to exist: "every directory with a closed_book
# file" silently pulled the controlled-family checkpoints and GPT-4o-mini into
# the cutoff ladder.
def _n_unique(name):
    f = PROC / name
    if not f.is_file():
        return 0
    return len({json.loads(l)["qid"] for l in f.read_text().splitlines() if l.strip()})


N_MAIN = _n_unique("questions.jsonl") or 878
N_FULL = _n_unique("questions_full.jsonl") or 4176

ANALYSES = [
    ("Cutoff panel", r"$\Delta_{\text{cut}}$ at 2025", LADDER,
     ["closed_book"], N_MAIN),
    ("Full-scale closed-book", r"$\Delta_{\text{cut}}$, no per-year cap",
     ["qwen32b", "qwen72b", "gpt4omini"], ["closed_book_full"], N_FULL),
    ("Unmasked Hybrid", r"$\Delta_{\text{RAG}}$", HYBRID,
     ["closed_book", "hybrid"], N_MAIN),
    ("Masked Hybrid (paired)", r"$\Delta_{\text{mask}}$",
     ["qwen1_5b", "qwen3b", "qwen7b", "llama3_8b", "qwen32b", "qwen72b",
      "mistral_large", "", "sonnet", "gpt4omini"],
     ["closed_book", "hybrid", "hybrid_masked"], N_MAIN),
    ("Support (string/NLI)", r"$\Delta_{\text{sup}}$, masked", HYBRID,
     ["closed_book", "hybrid_masked"], N_MAIN),
    ("Unmasked support", r"$\Delta_{\text{sup}}$, string/NLI", [m for m in HYBRID if m],
     ["closed_book", "hybrid"], N_MAIN),
    ("Decision", "held-out policy gain", [m for m in HYBRID if m],
     ["closed_book", "hybrid", "hybrid_masked"], N_MAIN),
    ("Cutoff--scale", "joint slope", ["qwen", "qwen1_5b", "qwen3b", "qwen7b", "qwen32b", "qwen72b", "mistral_large"], ["closed_book"], N_MAIN),
    ("Support--scale", "unsupported gain slope", ["qwen1_5b", "qwen3b", "qwen7b", "qwen32b", "qwen72b", "mistral_large"], ["closed_book", "hybrid_masked"], N_MAIN),
    ("Item-level fit", "capability interaction", ["qwen1_5b", "qwen3b", "qwen7b", "qwen32b", "qwen72b", "mistral_large"], ["closed_book", "hybrid"], N_MAIN)

]


def rows_in(model: str, cond: str) -> int:
    """Unique question ids in a cache, not raw lines.

    A cache can hold duplicate rows for one question; counting lines then
    reports 4182 where the deduplicated set has 4176, which conflates cache
    size with coverage. Coverage is what this table is about.
    """
    f = PROC / (f"generations_{model}" if model else "generations") / f"{cond}.jsonl"
    if not f.is_file():
        return 0
    ids = set()
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ids.add(json.loads(line)["qid"])
    return len(ids)


def main():
 panels=[]
 for label,estimand,members,conditions,n in ANALYSES:
  panels.append({'name':label,'estimand':estimand,'expected_questions':n,'conditions':conditions,'members':[{'model':m or 'haiku','coverage':{c:rows_in(m,c) for c in conditions}} for m in members]})
 models={}
 for m in PRETTY:
  q='Qwen/Qwen2.5-'+PRETTY[m]+'-Instruct'
  if m in ['qwen32b','qwen72b']:q+='-GGUF'
  models[m or 'haiku']={'checkpoint':q if m.startswith('qwen') else None,'temperature':0,'runtime':'local','revision':None,'tokenizer_revision':None,'precision':'Q4_K_M' if m in ['qwen32b','qwen72b','mistral_large'] else 'not retained for original small-Qwen runs'}
 models['mistral_large'].update(checkpoint='lmstudio-community/Mistral-Large-Instruct-2407-GGUF')
 for m,checkpoint,provider,tokens in [('haiku','claude-haiku-4-5-20251001','Anthropic',64),('sonnet','claude-sonnet-4-5-20250929','Anthropic',64),('gpt4omini','gpt-4o-mini','OpenAI',32)]:
  models[m].update(checkpoint=checkpoint,runtime='API',provider=provider,max_tokens=tokens,precision=None,served_snapshot=None)
 for m,rec in FAMILY.items():models[m]={'runtime':'local','temperature':0,'checkpoint':rec['hf_repo'],'precision':rec['precision'],'cutoff':rec['cutoff'],'revision':None,'tokenizer_revision':None}
 out={'scope':'Recorded identifiers and cache coverage, not a reconstruction of missing historical metadata','missing_value':'null means unavailable, not a default or inferred revision','sources':['pipeline/controlled_cutoff.py','pipeline/gen_openai.py','configs/models.yaml'],'models':models,'panels':panels}
 (ROOT/'docs/model_inventory.json').write_text(json.dumps(out,indent=2)+'\n')
 print('Wrote docs/model_inventory.json')
if __name__=='__main__':main()
