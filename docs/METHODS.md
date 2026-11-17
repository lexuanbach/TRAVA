# Analysis conventions and limits

## Three layers, four stages

The three layers define what we audit: question time, evidence time and evidence content. We organize the evaluation into four stages. Stages 1 and 2 construct dated question groups and choose the calendar split. Stage 3 compares closed-book performance, and stage 4 evaluates evidence timing and support. The released replay recomputes these analyses from frozen answers and labels.

**Question time:** Static means a question date before 2025-01-01. Fresh means on or after that date. Membership describes question dates and does not establish pretraining inclusion. The stored sampling split uses a different boundary. Several models have resolved closed-book gaps. Weekly-quiz bootstrap tests retain Mistral and both Claude contrasts after Benjamini and Hochberg adjustment across nine models, conditional on the outcome-selected 2025 boundary. Historical max-drop and odd/even-day diagnostics test different statistics. Mistral selects 2024 in the latter, while 32B falls back to the full-data 2025 boundary. Gap magnitudes depend on question cohorts, weighting, scoring metric and boundary selection. These performance differences do not identify knowledge cutoffs. The [result guide](RESULTS.md) identifies the corresponding records and analysis conditions.

**Evidence time:** the principal admissibility rule is publication date no later than question date plus three days. This grace window permits some post-question publication; it does not prove availability at question time. Zero-, one- and seven-day windows are examined separately.

**Evidence content:** visible support applies a string/entailment diagnostic to the actual shown prefix. It is conditioned on the gold answer and therefore is an offline diagnostic. Positive support interactions do not identify the generator's reasoning mechanism. Gold-conditioned groups can favor answers copied from context.

## Inference

Primary paired contrasts resample weekly quiz blocks. Historical max-drop, odd/even-day, distractor and some human summaries use item resampling and ignore weekly dependence. Each record identifies its procedure. `question_pooled` weights eligible questions; `quiz_weighted` gives each eligible quiz equal weight. These estimands can differ. Interval exclusion of zero, Holm-adjusted significance and practical equivalence answer different questions.

The capability regression centers capability at the recorded eight-model mean (36.4664), then fits the primary six-open-model panel. At centered capability c, the Static-minus-Fresh contrast is -(beta_Fresh + beta_interaction * c). It equals -beta_Fresh only at c = 0.

## Interpretation

Stale-index and leave-one-out interventions simultaneously change multiple properties of the retrieved evidence. The stale-index effect cannot be assigned exclusively to recency. Hidden-answer versus visible-answer groups also differ observationally in rank and question composition.

Abstention detector recall is not an abstention-rate bound. The fraction of questions without string matches is not a semantic coverage ceiling. Existing human labels concern selected passages and legacy windows. Selection-refit policy evaluation leaves utility gains unresolved.

## Active and historical records

Several root-level JSON files retain the original full-document support analysis because paired reruns use them as comparators. The active shown-text results are under `results/analysis/`, with `new_visible600` fields where both conditions are stored. Use `docs/RESULTS.md` to select the intended basis. Do not mix original full-document support figures with current visible-prefix interpretations.

## Retrospective statistical clarifications

The largest slice-specific masking point estimate is 1.9 F1. Across 20 cells, 95% interval endpoints span -2.6 to 4.1. Neither resolved cell survives Holm adjustment across the 20 masking contrasts. Equivalence within two F1 points applies only to the specified overall window-study contrasts.

The stale index admits 216 late contexts among 2,020 Static contexts, affecting 141 of 404 questions. Its publication-date rule implies query admissibility for Fresh questions, but not Static questions.

Paired quiz resampling on the same 900 Task B passage rows favors cached NLI and judge labels over string matching, with Holm p = 0.006 for each. This is agreement with legacy 700-character human excerpts, not exact-input human validation. The 630-row masked pack is analyzed separately. All three comparisons within a pack share bootstrap draws and form one Holm family.

The three-rung Llama size comparison has exhaustive permutation p = 4/6 and minimum attainable p = 1/6. Its bootstrap interval holds the roster fixed. A positive roster-conditional slope does not establish a causal size effect.

The eight-boundary mean uses 3,136 valid draws out of 5,000, conditional on resampling the earliest quiz. It remains a retrospective sensitivity analysis and does not correct boundary selection.
