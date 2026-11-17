# TRAVA

**A Temporal RAG Validity Audit of Knowledge Cutoffs and Evidence Support**  
Research artifact for *Findings of EACL 2027* · Benchmark snapshot `v2026-07`

TRAVA asks three questions that an aggregate retrieval-augmented generation score leaves open:

1. **Question time:** how does the date boundary change the closed-book comparison?
2. **Evidence time:** does each retrieved passage meet the question's publication window?
3. **Evidence content:** does the text actually shown to the generator support the gold answer?

This repository contains the analysis code, frozen model responses, retrieval rankings, evidence corpora, and existing human labels needed to inspect these comparisons. The default workflow runs on a CPU without model downloads or API calls. It **recomputes analyses from cached answers and labels**; it does not generate new answers or human judgments.

## Start here

Use Python 3.14, then run these commands from the repository root:

```bash
git clone https://github.com/lexuanbach/TRAVA.git
cd TRAVA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_snapshot.py
python -m pytest tests -q
python scripts/reproduce.py --suite core --output outputs/core
```

`verify_snapshot.py` checks every distributed input against SHA-256 checksums. The tests check quiz clustering, visible-context alignment, complete prompt reconstruction, adjudicated metric denominators, annotation counting units, and the capability-centering convention.

The **core** replay covers metric errors, masked human support, stale-index effects, hidden-answer effects, boundary contrasts, context budgets, and publication windows. The **full** replay additionally recomputes support-label, within-source, policy-selection, and native-template analyses:

```bash
python scripts/reproduce.py --suite full --output outputs/full
```

Each run creates an isolated working copy inside its output directory. Frozen inputs and results in this repository remain unchanged. Inspect `outputs/full/summary.json` for stage outcomes and `outputs/full/logs/` for command output. Reruns require a new output directory.

[Reproduction guide](docs/REPRODUCING.md) · [Data and schemas](docs/DATA.md) · [Result guide](docs/RESULTS.md) · [Methods and limits](docs/METHODS.md)

## What to look for

- **Boundary sensitivity:** the 32B closed-book gap changes from about 4.9 F1 under the selected split to 0.2 under the conservative year pair; both comparisons retain uncertainty.
- **Index interventions:** restricting the index to pre-2025 documents costs about 21 Fresh F1 across three generators. Removing each question's linked evidence also reduces performance. These are compound interventions, not isolated effects of evidence recency.
- **Visible support:** retrieval gains concentrate where the shown passage supports the answer. The support labels use the gold answer, so this is an offline diagnostic, not a deployable routing signal.
- **Policy utility:** selection-refit results do not establish a general utility gain. Prediction quality alone is not evidence that a retrieval policy improves deployment outcomes.

The result guide identifies the JSON records and reproduction stages behind each finding, including uncertainty and analysis conditions.

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | Dataset, generator, retriever and evaluation settings |
| `pipeline/` | Metrics, retrieval, generation and statistical analysis code |
| `pipeline/analysis/` | Visible-context, temporal and policy sensitivity analyses |
| `data/processed/` | Frozen questions, corpora, rankings and model responses |
| `annotation/` | Existing labels, blinded-study metadata and scoring code |
| `results/` | Reference results; see the result guide for the active analysis basis |
| `scripts/` | Snapshot verification and isolated reproduction entry points |
| `tests/` | Computational regression checks |
| `docs/` | Reproduction, data, methods and result documentation |
| `snapshot.json` | File hashes and snapshot identity |

The manuscript, LaTeX sources, reviewer correspondence and internal revision records are not included.

## Scope

The main panel contains **878 questions**. The uncapped question file contains **4,176 unique questions**, with a smaller model panel. Exact 600-character human annotation packs are present but **uncollected**; existing human-support results use legacy excerpts. Fresh prompt tokenization requires external tokenizers/GGUF files and is not part of the default replay. Cached prompt-hash and context-alignment checks are included.

[Optional retrieval and generation](docs/GENERATION.md) explains the supplied entry points and the additional hardware, weights, data and credentials they require. Fresh generation was not revalidated as part of this release preparation.

## Citation and contact

Authors: **Xuan-Bach Le\*, Phat T. Tran-Truong\*, Minh-Hieu Le, Duy Nguyen, Son Ha Xuan**. Xuan-Bach Le and Phat T. Tran-Truong contributed equally. Xuan-Bach Le is the corresponding author.

Please cite the TRAVA paper and the upstream RealTimeQA dataset when using this artifact. Machine-readable citation metadata is in [CITATION.cff](CITATION.cff); the paper is accepted and does not yet have a proceedings DOI in this package.

Contact: **Xuan-Bach Le**, [lexuanbach@hcmut.edu.vn](mailto:lexuanbach@hcmut.edu.vn) · [Project author's website](https://lexuanbach.github.io/).

## License

TRAVA code is available under the [MIT License](LICENSE). Questions, news excerpts, model outputs and external checkpoints are **not relicensed by that code license**. Read [DATA_LICENSE.md](DATA_LICENSE.md) for attribution, upstream terms and the status of article-text redistribution.

### Numerical provenance

Run `python pipeline/rebuild_ledger.py --check` to verify the stored claim ledger against the current result files. Run without `--check` to rebuild it after an intentional result update. `python pipeline/reporting_summary.py` recomputes the common-sample choice scores, matched scale contrast and frozen-panel equivalence check from cached answers.

`docs/model_inventory.json` lists analysis panels, checkpoint identifiers and unavailable historical settings. `docs/protocols/` and `docs/corrections.md` document scientific provenance. They are retrospective publication of dated local records, not independently authenticated preregistrations.
