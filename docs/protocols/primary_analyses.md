> Published 15 November 2026 from a dated local protocol. This is not an independently authenticated preregistration. The historical text below preserves its planned settings and dated amendments. Final implementation differences are described in `../corrections.md` and the result records. Original SHA-256: `431b7a15359f8c761a7692ca9878c0bc072858b4b1a4766efd74205c3eacc58b`.

# Dated local protocol: E2 (sample replication), E3 (window sensitivity), E6 (context budget)

Written 2026-09-17 ~11:00 ICT, before any E2/E3 generation and before any output of the corrected E6 run was inspected. Changes after this point are listed at the end with a date and reason.

## Shared configuration

- **Generation.** HF transformers 4.57.1, bfloat16, greedy, `max_new_tokens=32`, batch size 1 (no padding), H100 MIG 2g.20gb slices. Prompts are byte-identical to `pipeline/local_generate.py` (600-character passage prefix, same system prompt). No truncation: `pipeline/analysis/gen_hf_notrunc.py` aborts if any prompt exceeds 4,096 tokens. Every compared arm, closed-book included, is regenerated under this one configuration. Cached MPS outputs are used only for a reproducibility-agreement check, never as an arm in a paired contrast.
- **Generators.** Primary panel in this window: Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. The planned 3B/32B/Mistral panel required additional GGUF hardware. Qwen2.5-32B and Mistral-Large-2 will be added under the same prompts when hardware allows, reported as a separate panel, and not pooled with the HF rows.
- **Endpoint.** Primary is question-level token F1 (`pipeline/metrics.py`), comparable with the paper. Secondary are EM and the answer-coverage indicator (F1 ≥ 0.5). No other endpoint will be introduced to rescue a result.
- **Inference.** Paired bootstrap resampling whole weekly quizzes (`qid.split('_')[0]`), B=5,000, seed 20260917. Both quiz-weighted (mean of within-quiz means) and question-pooled estimates are reported. **Quiz-weighted is primary for E3 and E2** (matching the paper's mask table); question-pooled is primary for E6 (matching `k_ablation.py`). Holm correction across generators within each primary family.
- **Support label.** Gold string (`metrics.gold_in_text`) in the first 600 characters of any shown passage, which is the visible-text label.
- **Boundary.** Static = year < 2025, Fresh = year ≥ 2025, derived from `date`.

## E6: context budget (878 questions, +3-day mask, Hybrid)

Arms: k3 (prefix of k5), k5, k10refused (original re-fused k=10 list), k10prefix (k5 plus the next five distinct re-fused candidates).

- Primary family: the support interaction (gain_S − gain_U against closed-book under the same configuration) at each arm, per generator. The closed-book arm for E6 is generated in the E3/E2 closed-book run under the same configuration.
- Primary contrasts: mean F1 of k10prefix − k5 and k10refused − k5, paired.
- Reported regardless of outcome: the support transition matrix k5→k10prefix and k5→k10refused.
- Interpretation: if the interaction at k10prefix stays within the k5 CI, report no evidence of attenuation from added context. If it falls with a CI excluding the k5 point, report attenuation from added context. Differences between k10prefix and k10refused are attributed to re-fusion.

## E3: admissibility window (878 questions, Hybrid top-5)

Arms: w ∈ {0, 1, 3, 7} days and unmasked, where a document is admissible if `publish_date ≤ q_date + w`, undated documents are admissible (as in `pipeline/retrieval.py`), and the mask is applied before BM25 top-100/BGE top-5 RRF fusion. w=3 must reproduce `retrieval_masked.json` and unmasked must reproduce `retrieval.json`. Otherwise the arm is rebuilt and the discrepancy reported.

- Exposure (no generation): share of shown contexts dated after `q_date` and after `q_date + w`, query-level exposure, and context-set change against w=3.
- Primary contrasts: F1(w) − F1(unmasked) for each w, and F1(0) − F1(3), F1(7) − F1(3), all paired and quiz-weighted.
- **Equivalence margin: ±2 F1.** A window is "practically equivalent" to another only if the 90% CI of the paired difference lies inside ±2. Otherwise the result is "not shown equivalent" or "different" (95% CI excludes 0).
- Scope note fixed in advance: the corpus was source-filtered at +3 days, so w=7 tests sensitivity within the retained document universe only.

## E2: question-disjoint replication (Phase A)

- **Sample.** 800 question IDs from the 3,298 IDs in `questions_full.jsonl` not in `questions.jsonl`, stratified by year with 200 per year for 2022–2025 (no unused 2026 items exist). Seed 20260917. Inclusion probability = 200 / N_year, stored per item.
- **Corpus.** The frozen 26,611-document `corpus_full.jsonl` for both the new 800 and the original 878. The original 878 on `corpus_full` masked must reproduce `retrieval_full.json` Hybrid.
- **Arms.** Closed-book, Hybrid top-5 masked +3, Hybrid top-5 unmasked, for new and original questions under the same configuration.
- **Estimands.** For each sample: Δmask = masked − unmasked (quiz-weighted); exposure; visible-text support prevalence; Δsup = gain_S − gain_U (masked Hybrid vs closed-book), for Static, Fresh (2025 only) and pooled. Common-year estimates use 2022–2025 only, since the original sample's 2026 items have no new counterpart. The new sample is weighted by inverse inclusion probability for population estimates, and raw estimates are shown as well.
- **Comparison.** New − original differences from joint resampling of shared quizzes. Stability margins, fixed now: ±2 F1 for Δmask and ±5 F1 for Δsup, judged by whether the 90% CI of the difference lies inside the margin. Otherwise the result is "stability not established" or "changed". The 173 questions in the 10 quizzes absent from the original sample are reported separately and descriptively.

## E5: does the audit change a retrieval-policy decision? (cached outputs only)

Added 2026-09-17 ~11:30 ICT, before any E5 computation.

- **Generators.** The nine with cached `closed_book`, `hybrid` and `hybrid_masked` over 878 questions: qwen1_5b, qwen3b, qwen7b, qwen32b, qwen72b, mistral_large, llama3_8b, gpt4omini, sonnet.
- **Candidate policies**, using only information available before answering (question text and date):
  - `CB`: always closed-book.
  - `RAG`: always masked Hybrid.
  - `DATE_b`: closed-book if question date < b, masked Hybrid otherwise, for b ∈ {2022-07-01, 2023-01-01, 2023-07-01, 2024-01-01, 2024-07-01, 2025-01-01, 2025-07-01, 2026-01-01}.
- **Naive procedure.** Choose between CB and RAG by dev-fold aggregate mean F1, which is what an aggregate RAG-vs-closed-book benchmark licenses.
- **Audit-informed procedure.** Choose among CB, RAG and all DATE_b by dev-fold mean F1. This is the decision space that exists only once question time is treated as a validity layer.
- **Leakage-audit variant (secondary).** The naive procedure scored on unmasked `hybrid` versus the same procedure scored on `hybrid_masked`. Does the admissibility correction change the CB-vs-RAG choice?
- **Validation.** 20 repeats of 5-fold cross-fitting grouped by weekly quiz (seed 20260917). Each question's test score comes from a policy chosen without its quiz. This is internal validation, not a prospective holdout.
- **Endpoint.** Cross-fitted per-question token F1 of the audit-informed procedure minus the naive one, averaged over repeats. Primary estimate is quiz-weighted with a 95% paired quiz bootstrap CI (B=5,000); question-pooled is reported alongside. Holm correction across the nine generators. Also reported: selection frequencies of each policy, the per-question oracle only as a descriptive ceiling, Static/Fresh subgroup results (harm check: Static F1 of the chosen policy versus CB), and retrieval call rate as the cost proxy. Latency is not claimed.
- **Interpretation, fixed now.** A positive difference with a Holm-adjusted CI excluding 0 means the audit changes the decision and improves held-out utility for that generator. Identical selections, or a CI including 0, mean "decision unchanged or not resolved." A negative difference is reported as a failure. No rule is changed after seeing results.

## Deviations

1. **2026-09-17 ~12:00 ICT, E6, added secondary analysis.** A preliminary E6 run used the frozen MPS closed-book cache, because the same-configuration closed-book arm was not yet generated. It was inspected and is labeled a sensitivity analysis only. Two things were added afterwards:
   - The CI-overlap rule for attenuation is weak; §5 of the plan says CI overlap is not a test. A paired bootstrap of the interaction difference (arm − k5), from the same quiz draws, is therefore added as a secondary analysis. The preregistered rule is still reported unchanged.
   - Every re-fused k=10 list contains all five k=5 documents (878/878), so `k10prefix` and `k10refused` hold the same ten documents in a different order (identical order for 154/878). Their comparison is reinterpreted as a passage-order contrast, not a membership contrast. No arm was changed.
2. **E3 generation, change of panel.** Qwen2.5-1.5B is run in E3 because its closed-book arm is needed for E6. Qwen2.5-32B and Mistral-Large-2 E3 arms run locally on Metal, the original configuration. Their frozen rows are reused where the rendered prompt's document list is identical, which is valid because frozen outputs reproduced exactly (32B 40/40, Mistral 20/20). These are reported as a separate GGUF panel, as preregistered.
