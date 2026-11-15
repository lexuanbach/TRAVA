> Published 15 November 2026 from the dated local protocol below. The prior public timestamp is unverified. Historical criteria are retained. Current interpretation distinguishes support in three annotated passages from support in all five retrieved passages.

# Pre-registration — human support labels on masked contexts

Fixed on 2026-08-04, **before any label was collected**. The pack was built by
`annotation/build_masked_pack.py` with seed 20260804. Nothing below may be
changed once labels arrive. Anything added afterwards is exploratory and must be
reported as such.

## The one confirmatory analysis

**Estimand.** On the Static slice, under the per-query temporal mask, with human
support labels:

```
Delta_sup(human, masked) = Delta_RAG | human-supported  -  Delta_RAG | human-unsupported
```

where `Delta_RAG = F1(masked hybrid) - F1(closed-book)`, paired within question.

**Question-level label.** A question counts as human-supported when at least one
of its three shown passages carries a majority human label of `YES`. This mirrors
the automatic question-level rule exactly, so the two are comparable.

**Weighting.** Every population estimate is weighted by the inverse inclusion
probability in `sampling_manifest.json`. Unweighted means over this pack are not
population quantities and will not be reported as such.

**Uncertainty.** Quiz-clustered bootstrap, 5000 draws, resampling whole weekly
quizzes so the supported and unsupported arms move together within a draw.
Reported as a 95% interval and a plus-one two-sided p-value.

**What would falsify the paper's claim.** The paper says the support interaction
survives the mask under automatic labels and, separately, survives human labels
on unmasked rankings. If the joint estimate's interval **includes zero**, the
joint claim is not supported and the paper will say the interaction is
established only under the two separate checks. We commit to reporting that
outcome in the main text, not the appendix.

**Practical-equivalence margin.** Fixed here at 5 F1. An interval lying wholly
within +/- 5 F1 is reported as no material interaction regardless of its p-value.

## Secondary, also fixed now

1. The same estimand on the Fresh slice.
2. `YES` versus `YES-or-PARTIAL` as the support threshold, both reported.
3. Agreement of the human label with exact-string containment and with
   entailment on these masked passages, as kappa with a bootstrap interval.
4. Cross-pack consistency on the 10 questions re-drawn from the previous pack.

## Stop rule

Krippendorff's alpha below **0.6** on `supports` halts the analysis. We revise
the guidelines and re-pilot rather than collecting more rows or reporting a
number from unreliable labels. Precedent: the freshness field of the previous
round reached 0.47 and no human staleness number was ever reported.

## Adjudication

Rows where the three annotators disagree go to a fourth person who does not see
the vote counts. The adjudicated label is final. We report the departure rate
from majority vote, because in the previous round the adjudicator departed from
the majority on 18 of 34 contested items, which is close to chance and is
evidence about how contestable those items were.

## What we will not do

- We will not drop annotators, items or strata after seeing the labels.
- We will not switch the support threshold to whichever gives a cleaner result.
  Both are pre-specified and both get reported.
- We will not report an unweighted mean as a population estimate.
- We will not rebuild the pack with a new seed after seeing an unwelcome result.
