# Analysis conventions and limits

## Three layers, four stages

The three layers concern question time, evidence time and evidence content. The operational stages construct questions/corpora, retrieve passages, generate answers and then score/analyze the frozen outputs. They describe different aspects of the same workflow.

**Question time:** Static means a question date before 2025-01-01; Fresh means on or after 2025-01-01. Membership is descriptive, not evidence of pretraining inclusion. The stored sampling split uses a different boundary. Closed-book contrasts remain uncertain and sensitive to year composition and boundary selection.

**Evidence time:** the principal admissibility rule is publication date no later than question date plus three days. This grace window permits some post-question publication; it does not prove availability at question time. Zero-, one- and seven-day windows are examined separately.

**Evidence content:** visible support applies a string/entailment diagnostic to the actual shown prefix. It is conditioned on the gold answer and therefore is an offline diagnostic. Positive support interactions do not identify the generator's reasoning mechanism.

## Inference

Question-level contrasts resample weekly quiz blocks, keeping paired conditions together. `question_pooled` weights eligible questions; `quiz_weighted` gives each eligible quiz equal weight. These estimands can differ. Interval exclusion of zero, Holm-adjusted significance and practical equivalence answer different questions.

The capability regression centers capability at the recorded eight-model mean (36.4664), then fits the primary six-open-model panel. At centered capability c, the Static-minus-Fresh contrast is -(beta_Fresh + beta_interaction * c). It equals -beta_Fresh only at c = 0.

## Interpretation

Stale-index and leave-one-out interventions simultaneously change multiple properties of the retrieved evidence. The stale-index effect cannot be assigned exclusively to recency. Hidden-answer versus visible-answer groups also differ observationally in rank and question composition.

Abstention detector recall is not an abstention-rate bound. The fraction of questions without string matches is not a semantic coverage ceiling. Existing human labels concern selected passages and legacy windows. Selection-refit policy evaluation leaves utility gains unresolved.

## Active and historical records

Several root-level JSON files retain the original full-document support analysis because paired reruns use them as comparators. The active shown-text results are under `results/analysis/`, with `new_visible600` fields where both conditions are stored. Use `docs/RESULTS.md` to select the intended basis. Do not mix original full-document support figures with current visible-prefix interpretations.
