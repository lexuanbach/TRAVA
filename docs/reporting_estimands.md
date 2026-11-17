# Comparing temporal contrasts on aligned terms

The 17 November 2026 reporting extension reuses frozen model answers and adjudicated labels. It adds no model generation or human judgments. These analyses were added retrospectively and do not replace the original protocol chronology.

## Endpoint, population and weighting

Run `python pipeline/aligned_contrasts.py` to reproduce `results/aligned_contrasts.json`. The panel is six Qwen2.5 sizes, Mistral 123B, Haiku 4.5, Sonnet 4.5 and GPT-4o-mini. The common 878-question sample has 404 pre-2025 and 474 later questions. The year-pair comparison uses 131 questions from 2023 and 346 from 2026.

Both token-F1 and nearest-option accuracy use identical questions. Exact option matches precede unique containment and token-overlap matching. Ties and unmatched answers receive zero. This is an automatic endpoint, not a human judgment. Question pooling weights items equally. Equal-quiz estimates first average within each quiz and temporal group, then give quizzes equal weight.

The seven-rung parameter slope excludes API models, whose sizes are undisclosed. A separate six-Qwen analysis removes the cross-family Mistral endpoint. Quantization still varies with size. Quiz-bootstrap intervals use 5,000 shared multinomial quiz draws with NumPy seed 20260731. Exact model-permutation p-values answer a different question and are reported separately. The historical `joint_slope.json` uses Python-random draws and remains unchanged.

The half-year summary averages the eight candidate boundaries from the original policy protocol, with shared quiz draws. All boundaries have nonempty observed sides. July 2022 has only seven pre-boundary questions in one quiz, so bootstrap draws missing that side are omitted and valid draw counts are reported. The average uses draws with valid contrasts at all eight boundaries. This sensitivity is not prospective validation or a correction for original boundary selection.

## Human metric agreement

`pipeline/human_metric_audit.py` now adds `cohens_kappa` to each `consistent_summaries` group and reports its interval using the existing adjudicated labels, population weights and shared within-stratum bootstrap. The overall weighted estimate is 0.745, compared with 0.356 without weights. `unweighted_summaries` explicitly records the latter basis. Existing fields and adjudication-count compatibility aliases remain available. No prior scalar estimates changed.

## Index and coverage checks

Run `python pipeline/reporting_checks.py`. The stale index restricts document publication dates to before 2025. Its Fresh linked-document hit rate is 288/474. Of 639 retrieved linked passages, 79 contain the answer string in the first 600 characters. Link recovery, string occurrence and semantic answer support are different measurements.

The second within-source sample's three-day exposure is 659/4000 contexts (16.475%). The matching original 2022–2025 sample has 453/2660 (17.030%). Both use the 26,611-document corpus. The older 17.5%/12.0% comparison mixed after-question exposure with an original population including 2026.

For 32B, July 2022 wins 80 of 100 policy fits but produces the same held-out retrieval actions as RAG. It is not globally equivalent: seven June questions exist in one quiz. The date rule is selected only when that quiz is in training. RAG wins all 20 fits holding it out. Original primary successes for Mistral and Sonnet remain documented separately from the retrospective selection-refit analysis, which resolves no adjusted gain.

Historical unmasked support/subset summaries omitted the bare `generations/` directory used by Haiku. Its answers and 600-character NLI labels already existed. The extension reports its unmasked NLI interactions and admissible-subset gains in `reporting_checks.json`. It verifies that cached labels match each retrieved passage. This completes coverage without creating labels or overwriting the historical nine-model summary.

Third-party excerpt rights, fresh tokenizer validation and the exact-input human annotation commitment are unchanged. See `DATA_LICENSE.md` and the annotation documentation.
