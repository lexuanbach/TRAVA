# Reproducing the cached analyses

## Requirements

The tested environment is Python 3.14 with the versions in `requirements.txt`, on macOS arm64. The CPU analysis code uses NumPy, SciPy, pandas and PyYAML. Allow about 1 GB of free disk space for the repository and one isolated replay; the largest corpus file is about 82 MB. Model weights and GPU memory are unnecessary for cached replay. Install dependencies while online; the commands below then use local inputs only.

## Three levels of reproduction

| Level | Command | What it establishes |
|---|---|---|
| Snapshot integrity | `python scripts/verify_snapshot.py` | All distributed files match the recorded bytes |
| Computational checks | `python -m pytest tests -q` | The tested counting, weighting, alignment and clustering invariants hold |
| Cached analysis | `python scripts/reproduce.py --suite full --output outputs/full` | Listed analysis stages can recompute their outputs from this repository alone |

The eight-stage core suite is a shorter entry point. The full suite uses the native default bootstrap budgets, including 5,000 draws for most sensitivity analyses and 1,000 outer bootstrap draws with five selection refits for the policy analysis. These defaults are in the scripts and output metadata. Do not reduce them when comparing manuscript intervals.

## Run isolation and outputs

The runner checks snapshot integrity first, then copies `pipeline/`, `annotation/`, `configs/`, `data/`, `results/` into `<output>/work/`. Analysis scripts run there. Reporting constants are stored as JSON in `results/analysis/masked_reporting_values.json`; no LaTeX installation or manuscript source is needed. They can overwrite reference-result paths only inside that copy. A nonempty output directory is rejected.

The runner records each command, exit code, elapsed time and status in `summary.json`, with stdout/stderr under `logs/`. A failed stage makes the runner exit nonzero. Skipped external-resource stages are recorded separately; they are never counted as completed analysis. Successful execution is distinct from equality with reference numbers: compare the regenerated output with the mapped reference record in `docs/RESULTS.md`, and use `scripts/compare_results.py` to report numerical changes for the files produced by the replay.

```bash
python scripts/compare_results.py outputs/full
```

The comparison ignores timing/provenance metadata, including stored-result comparison summaries and correction-history annotations and compares numeric leaves with an absolute tolerance of 1e-8. It reports missing, added and changed numerical fields. Different library versions can affect resampling or linear algebra. Inspect differences rather than assuming a successful script reproduces every stored number.

## What is deliberately not rerun

- Model answer generation, NLI relabeling and model-based judging: cached outputs are supplied.
- Human annotation: the exact-input 600-character packs are blank; no completion is claimed.
- Fresh tokenizer/GGUF prompt-length analysis: the external tokenizer assets are not supplied. The frozen historical result is available; tests independently reconstruct cached prompts and their hashes.
- Corpus retrieval from new upstream data: it changes the snapshot and requires additional dependencies.

## Troubleshooting

- **Missing or modified snapshot file:** restore the distributed file before replay. The runner never fills missing scientific inputs from another local project.
- **Optional model import fails:** use the cached runner, not a generation or retrieval script; model dependencies belong to the optional environment.
- **Output directory exists:** choose a new `--output` path. Keep previous results for comparison.
- **Memory pressure:** run one replay at a time. `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are set to 1 for consistent CPU use.
- **Reference differences:** check the stage's defaults, library versions and the distinction between legacy full-text and active visible-text results in the result guide.

## Validation of this package

On 17 November 2026, all 24 full-suite cached-analysis stages completed on macOS arm64 with the pinned CPU environment. Three external-resource activities were explicitly skipped: fresh tokenization, model generation and human collection. The 35 computational tests passed. The numerical comparison covered the 23 JSON outputs named by the stage manifest; the mention-example stage produces JSONL rather than a numerical summary. This validation does not claim that fresh retrieval or model generation was repeated.

The deterministic ledger check is `python pipeline/rebuild_ledger.py --check`. It validates the released result paths and compares the stored CSV with a fresh byte-for-byte rebuild. The November replay comparison covers 47,029 numeric fields with zero differences.

The shared quiz-bootstrap helper treats residuals within 1e-12 of zero as ties in both tails. This stabilizes p-values across floating-point backends without changing the reported point estimates or intervals. It is an arithmetic convention, not a significance threshold. CI runs the full cached suite and prints any remaining numerical differences.
