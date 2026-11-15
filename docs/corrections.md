# Measurement corrections and protocol chronology

The dated local protocols in `protocols/` record planned analyses. They were published with this release and were not independently registered before analysis. Their dated additions remain part of the record.

The support correction inventory is `results/analysis/CORRECTION_MANIFEST.json`. Of 36 analyses, 22 were recomputed on the visible 600-character prefix, seven retain comparison with legacy 700-character human annotations, six are unaffected, and one truncated context-budget condition is withdrawn. The corrected masked result is `results/analysis/masked_counterfactual__vis600.json`.

The window study's 1.5B, 3B and 7B arms use historical untruncated HF/bfloat16 generations. The 32B and Mistral arms reuse frozen responses where rankings match. Equivalence of the 20 overall window-study contrasts does not establish equivalence for the frozen main panel. Its Qwen2.5-1.5B three-day 90% interval is [-0.61, 2.35], outside the prespecified ±2 margin. `pipeline/reporting_summary.py` recomputes this check from stored answers.

The replication contains 43 questions in ten weekly quizzes absent from the original sample, correcting the local protocol's preliminary 173-question count. Follow-up visible-text, masked-entailment, prompt-format, selection-refit and judge analyses were retrospective. They do not inherit the primary protocol's prespecification status.

Metric-error summaries use the same adjudicated labels, weights, threshold and bootstrap draws. There are 34 adjudicated correctness items and 18 changed majority labels. `n_adjudication_overrides` remains a compatibility alias for `n_adjudicated_items`, not the number changed. Use `n_labels_changed_by_adjudication` for the latter.

Three annotators and a separate adjudicator volunteered without monetary compensation, as confirmed by the authors on 15 November 2026. Exact-input human packs remain uncollected. This release generates neither new annotations nor model outputs.

The ledger command rebuilds result-derived numerical rows and checks the stored CSV byte for byte. Historical support-dependent entries retain their superseded status. Public result paths replace earlier local paths without changing the underlying scientific evidence.

## Provider cutoff correction (15 November 2026)

The Claude Haiku 4.5 system card states a February 2025 training-data cutoff. The earlier July entry conflated Haiku with Sonnet 4.5. The verified-slice and cutoff-relative analysis entry points now use February for Haiku. Of the 474 Fresh questions, 23 predate this cutoff, compared with 95 before Sonnet's July cutoff. The uncertainty band around a cutoff is an analyst sensitivity range, not a provider-certified range. Generator outputs remain unchanged.

Source: [Haiku 4.5 system card, page 5](https://assets.anthropic.com/m/99128ddd009bdcb/original/Claude-Haiku-4-5-System-Card.pdf). Sonnet's reliable-knowledge cutoff is January 2025 and its training-data cutoff is July 2025, as distinguished by the [Anthropic Transparency Hub](https://www.anthropic.com/transparency).

The historical choice-audit summary juxtaposed a 2025-boundary token-F1 slope with a 2023/2026 choice-accuracy slope. That record now names the estimands and directs readers to the aligned comparison in `results/reporting_summary.json`. The choice result is positive but does not establish metric independence.

The nearest-option entry point now fixes the 878-question, eleven-generator closed-book panel explicitly, including the unsuffixed Haiku cache. This replaces an open-ended directory glob. The cutoff-relative loader likewise recognizes the unsuffixed Haiku cache. These repairs change analysis coverage, not stored answers.
