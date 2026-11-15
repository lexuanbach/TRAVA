# Optional retrieval and model generation

The cached replay is the supported reproduction entry point. These original pipeline scripts are supplied for extending the study; they require external resources and were not rerun during package preparation. Run them in a separate copy, because their default paths can overwrite caches.

## Data and retrieval

1. Inspect `configs/datasets.yaml` and `DATA_LICENSE.md`.
2. `python pipeline/download_data.py` fetches weekly question/evidence files from the current upstream RealTimeQA repository into `data/raw/`.
3. `python pipeline/build_dataset.py` constructs sampled questions and the main corpus. `build_questions_full.py` and `build_corpus_full.py` construct the uncapped panels.
4. `python pipeline/retrieval.py --only bm25` builds BM25 rankings under the default mask. `--no-mask` requests the unmasked condition; use `--out-suffix` to distinguish outputs. See `--help` and `configs/retrieval.yaml` before using the other retrievers.

Downloading from upstream today does not reconstruct the frozen historical snapshot automatically. Use the distributed question IDs, corpus IDs and hashes when exact comparability matters. FreshQA and model downloads have separate upstream sources and terms.

## Open-weight generation

Install `requirements-generation.txt` in a separate environment. GPU runtimes and llama.cpp may need platform-specific installation; the recorded original platform was Apple silicon with Metal offload for the large GGUF models.

A small Transformers example, in a disposable copy:

```bash
python pipeline/local_generate.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --tag example_qwen --conditions closed_book,hybrid --limit 10
```

This is a ten-question smoke run, not the full experiment. `--tag` selects a response directory. The larger GGUF path is `local_generate_gguf.py`; `gen_gguf_ranked.py` accepts explicit `--repo`, `--files`, `--tag` and `--jobs` for named ranking conditions. Actual model identifiers and available cache panels are in `results/environment_manifest.json` and `data/processed/`.

The principal 32B, 72B and Mistral 123B conditions used Q4_K_M GGUF quantization, a 2,048-token context and 32 output tokens. Corrected context-budget conditions use their separately recorded untruncated configuration. The original Mistral fallback template differs from the native-template sensitivity run. Do not silently mix these settings.

## API models

`pipeline/generate.py` reads `ANTHROPIC_API_KEY` from the environment. `pipeline/gen_openai.py` handles the separate GPT-4o-mini panel. Credentials are not included. API snapshots, access and costs can change; cached responses are the evidence for the reported run. A tag does not pin the live provider backend.

## Support labels and human data

NLI uses the DeBERTa checkpoint recorded in `results/environment_manifest.json`; the visible/masked/dated variants are under `pipeline/analysis/`. Judge inference also needs external weights. These outputs are already cached for the CPU replay.

No script can substitute for collecting the prepared 600-character human annotation packs. Their status remains uncollected. Existing labels and sampling weights are documented in `docs/DATA.md`.
