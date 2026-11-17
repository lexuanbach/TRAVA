# Measurement corrections and protocol chronology

The dated local protocols in `protocols/` record planned analyses. They were published with this release and were not independently registered before analysis. Their dated additions remain part of the record.

The support correction inventory is `results/analysis/CORRECTION_MANIFEST.json`. Of 36 analyses, 22 were recomputed on the visible 600-character prefix, seven retain comparison with legacy 700-character human annotations, six are unaffected, and one truncated context-budget condition is withdrawn. The corrected masked result is `results/analysis/masked_counterfactual__vis600.json`.

The window study's 1.5B, 3B and 7B arms use historical untruncated HF/bfloat16 generations. The 32B and Mistral arms reuse frozen responses where rankings match. Equivalence of the 20 overall window-study contrasts does not establish equivalence for the frozen main panel. Its Qwen2.5-1.5B three-day 90% interval is [-0.61, 2.35], outside the prespecified ±2 margin. `pipeline/reporting_summary.py` recomputes this check from stored answers.

The replication contains 43 questions in ten weekly quizzes absent from the original sample, correcting the local protocol's preliminary 173-question count. Follow-up visible-text, masked-entailment, prompt-format, selection-refit and judge analyses were retrospective. They do not inherit the primary protocol's prespecification status.

Metric-error summaries use the same adjudicated labels, weights, threshold and bootstrap draws. There are 34 adjudicated correctness items and 18 changed majority labels. `n_adjudication_overrides` remains a compatibility alias for `n_adjudicated_items`, not the number changed. Use `n_labels_changed_by_adjudication` for the latter.

Three annotators and a separate adjudicator volunteered without monetary compensation, as confirmed by the authors on 15 November 2026. Exact-input human packs remain uncollected. This release generates neither new annotations nor model outputs.

The ledger command rebuilds result-derived numerical rows and checks the stored CSV byte for byte. Historical support-dependent entries retain their superseded status. Public result paths replace earlier local paths without changing the underlying scientific evidence.

## Provider cutoff records (15 November 2026, reconciled)

An earlier correction used February 2025 from page 5 of the original Haiku 4.5 system card. That passage explicitly concerns public-Internet data. The current model specifications separately list July 2025 for the overall training-data cutoff and February for reliable knowledge. These records differ in scope, so February must not replace the overall training-data date. Both are retained in `docs/provider_cutoffs.md`.

The primary cutoff-relative analysis uses July 2025. Its analyst-chosen sensitivity band is April–October, and an explicit February alternative preserves the original system-card interpretation. Of 474 Fresh question dates, 95 fall through July and 23 through February. These counts do not establish training membership. The 2023/2026 slice, stored model answers and human labels are unchanged. Derived cutoff-relative summaries and the ledger are rebuilt from those caches.

The historical choice-audit summary juxtaposed a 2025-boundary token-F1 slope with a 2023/2026 choice-accuracy slope. That record now names the estimands and directs readers to the aligned comparison in `results/reporting_summary.json`. The choice result is positive but does not establish metric independence.

The nearest-option entry point now fixes the 878-question, eleven-generator closed-book panel explicitly, including the unsuffixed Haiku cache. This replaces an open-ended directory glob. The cutoff-relative loader likewise recognizes the unsuffixed Haiku cache. These repairs change analysis coverage, not stored answers.

## 17 November 2026: aligned estimands and reporting

The retrospective extension in [reporting_estimands.md](reporting_estimands.md) adds consistently weighted kappa, aligned endpoint/weighting contrasts, a within-Qwen slope, a boundary-average sensitivity and explicit index/fold coverage checks. It corrects the mixed-population exposure comparison and completes previously omitted Haiku summaries from existing caches. Historical protocols and model/annotation inputs are unchanged.
