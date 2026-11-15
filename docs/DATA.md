# Data and annotation guide

## Snapshot and units

`v2026-07` fixes the sampled questions, corpora, retrieval rankings and cached responses. The latest main-panel quiz is dated 2026-07-03. The separate future-held-out file contains 40 questions from July 9, 10 and 17. Counts are not interchangeable:

| File or study | Counting unit |
|---|---|
| `data/processed/questions.jsonl` | 878 unique sampled questions |
| `data/processed/questions_full.jsonl` | 4,176 unique uncapped questions |
| Task A correctness | 395 answer items, with 34 adjudication records |
| Task B support | 900 passage judgments; 300 question–context instances; 254 unique questions |
| Masked support | 630 passage rows; 210 questions = 200 stratified + 10 consistency cases |

The corrected masked population support estimate uses the 200 stratified questions and their design weights. The ten consistency cases have no population inclusion weights. The cached support interactions use an unweighted 210-question sample and must not be described as population estimates.

## Core formats

- **Questions (`questions*.jsonl`):** one object per question, including `qid`, `date`, `question` and `gold_answer`. `qid` joins response and ranking records. The stored `split` belongs to the sampling convention; derive the analysis regime from the date.
- **Corpora (`corpus*.jsonl`):** `doc_id`, text, source metadata and publication date. The generator normally sees only the first 600 characters of each selected passage. A retrieved document can contain an answer outside that visible prefix.
- **Rankings (`retrieval*.json`):** ordered document IDs by retriever and question. Preserve ordering when reconstructing prompts.
- **Responses (`generations*/*.jsonl`):** `qid`, `condition`, model identity, `prediction`, and usually `doc_ids`. Corrected context-budget records also include prompt hashes, token counts and truncation flags.
- **Automatic support labels (`results/analysis/*labels*.jsonl` and NLI caches):** question/passage identities, classifier outputs and, where recorded, input-prefix hashes. These are frozen model outputs, not human judgments.
- **Human CSVs (`annotation/`):** anonymous item IDs, labels and study fields. `sampling_manifest.json` records strata and population weights. `item_metadata.jsonl` maps blinded IDs to question/model/condition and automatic scores; it is needed for analysis and must be hidden during any new blinded annotation exercise.

## Annotation limits

Task A uses majority labels with the existing adjudication overrides. The shared metric-error summaries use the same weights, F1 threshold 0.5 and 5,000 within-stratum bootstrap draws (seed 20260802). A conditional miss rate divides by human-correct outputs; missed-correct prevalence divides by all outputs.

Task B and masked support use legacy excerpts, generally 700 characters under the historical whitespace conventions. Human-unsupported means unsupported within the annotated passage subset. It does not establish that no answer-bearing evidence exists anywhere in the corpus.

`annotation/exact_input_600_2026-09-20/` contains prepared, blank 900-row and 630-row packs. They are **not completed annotations**. Guidelines describe the intended task; their presence is not evidence that a new study occurred.

## Provenance

The upstream dataset is [RealTimeQA](https://github.com/realtimeqa/realtimeqa_public). `configs/datasets.yaml` records the sampling configuration, and `snapshot.json` hashes this artifact's actual files. Recorded model and package identifiers are in `results/environment_manifest.json`. Some historical result objects retain source hashes from the experiment; those are historical provenance, while `snapshot.json` is authoritative for the files distributed here.
