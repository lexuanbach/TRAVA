# Existing human annotations

See [the data guide](../docs/DATA.md) for sampling units, weights, adjudication and limits.

- `pack_2026-08-02/`: 395 correctness items and 900 passage rows, three rater files and existing adjudication.
- `masked_support_2026-08-04/`: 630 passage rows across 210 questions; use the 200 stratified questions for population weighting.
- `adjudication/`: recorded adjudication inputs and scorer.
- `freshness_repilot/`: historical freshness-label pilot.
- `exact_input_600_2026-09-20/`: blank, uncollected exact-input packs.

The `item_metadata.jsonl` and `adjudication_metadata.jsonl` files are blinding keys, not credentials or named rater records. They are required for reproduction and must be withheld from annotators during a new blinded exercise. No new annotation collection is performed by the reproduction runner.

The guidelines preserve the study instructions. Run the supported scoring stages through `scripts/reproduce.py` so reference results remain untouched.
