# Freshness, version 2 — operationalised decision rule

This replaces the `freshness` instructions in `../GUIDELINES.md`. Nothing else
about Task B changes.

**Why there is a version 2.** Three annotators labelled freshness and agreed at
Krippendorff's α = 0.47, below the 0.6 we require before treating a column as a
measurement. We diagnosed the failure rather than re-running it, and found two
causes. Both are fixed below; neither was the annotators' fault.

---

## Fix 1 — you will only be asked about evidence that is on topic

In the first pass, **64% of rows** asked whether a document was "current for
this question's date" when the document had nothing to do with the question — an
article about Nichelle Nichols filed against a question about Cindy Williams.
There is no right answer to that, and agreement there was α = 0.29 against 0.57
on rows where the evidence was actually about the question.

This pack contains on-topic rows only. If one still slips through, that is what
`NOT_ABOUT_THIS_QUESTION` is for. **Use it freely** — it is a real answer, not a
failure to decide, and it tells us the filter missed something.

## Fix 2 — the rule no longer asks you what the world is like

Version 1 asked whether a fact "could change". That is a question about the
world, and reasonable people answered it differently: is "the Steelers have won
six Super Bowls" a fact that can change (yes — they could win a seventh) or a
settled historical count (also defensible)? Nothing in the guidelines decided it.

Version 2 asks about **the sentence in front of you**, which you can check.

---

## Step 1 — find the relevant sentence

Read the excerpt and find **the sentence that bears on the question**. If more
than one does, use the one that comes closest to answering it. You are labelling
that sentence, not the article, not the topic, not the event.

**Every excerpt in this pack ends at a complete sentence.** In the first pass it
did not — 98% were cut at a fixed character count and 77% ended mid-word — and
you cannot judge the tense of half a sentence. If you were given one of those,
"…the plant opened in 2019 and is" would look like an ongoing state when the
claim is a completed event. That is fixed; the excerpts here are trimmed back to
the last full sentence, so what you see is always readable and is always a
subset of what the system itself was shown.

If, despite that, **no sentence in the excerpt bears on the question**, label
`NOT_ABOUT_THIS_QUESTION` and move on. Do not reconstruct what a missing
sentence probably said.

### What "bears on the question" means — apply this test, do not judge it

The first pilot of this rule agreed at α = 0.85 on rows where everyone accepted
the excerpt was in scope, and fell apart on this decision: two annotators called
37 and 42 of 150 rows off topic, the third called one. Both readings were
defensible under "use it freely", which is why it now has a test.

Mark `NOT_ABOUT_THIS_QUESTION` when **both** of these hold:

1. **No sentence names the thing the question asks about** — the entity, event,
   number or role in the question does not appear, by name or by an obvious
   alias, anywhere in the excerpt; and
2. **you could not tell what the question was** if you were shown only the
   excerpt.

If the excerpt is about the right entity or event but simply does not answer the
question, that is **on topic** — proceed to step 2 and label freshness. "Does
not contain the answer" is the `supports` column's job, not this one.

| Question | Excerpt | Call |
|---|---|---|
| Which actress died this week? (Cindy Williams) | article about Nichelle Nichols | `NOT_ABOUT_THIS_QUESTION` — different person, and you could not guess the question |
| Which team won the Super Bowl? | article on career Super Bowl totals | **on topic** — same event, wrong detail |
| How much has the war cost? | article about the war, no figure | **on topic** — right subject, no answer |
| In which year did women first compete? | 2014 article on ski jumping at Sochi | **on topic** — same subject, wrong year |

When in doubt, **call it on topic**. A wrong freshness label is recoverable; a
row silently dropped from the analysis is not.

Write nothing down for this step; it just fixes what "the claim" means below.

## Step 2 — classify the claim (`claim_type` column)

| `claim_type` | The relevant sentence asserts… | Examples |
|---|---|---|
| `STATE` | an **ongoing situation** — something that holds "as of now" and could later stop holding | "Ron Klain **has served** as chief of staff since 2021", "the fund **is** worth $2 trillion", "X **leads** the polls" |
| `EVENT` | a **completed occurrence tied to a date** — once true, true forever | "the Steelers **won** the 2009 Super Bowl", "the rocket **rolled back** on Sept 2", "she **died** on Jan 25" |
| `TIMELESS` | a definition, a physical or mathematical fact, or a categorical property with no time index | "an ion with more electrons than protons carries a net negative charge" |

**The grammatical cue is reliable and you should use it.** Present tense, present
perfect, "currently", "since", "as of" → `STATE`. Simple past with a date or a
recoverable date → `EVENT`.

**The trap:** an article *published* recently about a *completed* event is still
`EVENT`. Publication date is not claim type.

**Counts and records are `STATE`**, not `EVENT`: "the Steelers have won six
Super Bowls" is an ongoing tally that a seventh win changes. This is the case
that split the first pass; it is now decided.

## Step 3 — assign `freshness`

Your `claim_type` almost determines the answer.

| If `claim_type` is… | then `freshness` is… |
|---|---|
| `TIMELESS` | `NOT_APPLICABLE`. Always. Stop. |
| `EVENT` | `NOT_APPLICABLE` **unless** the event is dated after `question_date`, in which case `ANACHRONISTIC`. A completed past event cannot be superseded. |
| `STATE` | one of the three below |

For a `STATE` claim, compare when the claim was true against `question_date`:

| `freshness` | Use when |
|---|---|
| `CURRENT` | the state it describes still held at `question_date` |
| `SUPERSEDED` | the state had already been overtaken by `question_date` — the classic stale-evidence case |
| `ANACHRONISTIC` | the state it describes belongs to a time **after** `question_date` |
| `UNDATED` | you cannot place the state in time at all, and the date column does not settle it |

**`SUPERSEDED` now requires a `STATE` claim.** If you are reaching for
`SUPERSEDED` on an `EVENT`, re-read step 2 — that is the error the rewrite
exists to prevent.

---

## Worked examples

| Question (date) | Relevant sentence | `claim_type` | `freshness` | Why |
|---|---|---|---|---|
| Who is chief of staff? (2023-02-24) | "Ron Klain has served as chief of staff since 2021" | `STATE` | `SUPERSEDED` | ongoing state, over by the question date |
| Who is chief of staff? (2023-02-24) | "Biden named Jeff Zients as his next chief of staff" | `EVENT` | `NOT_APPLICABLE` | a completed appointment; it cannot go stale |
| Which team won the Super Bowl? (2026-02-12) | "the Steelers and Patriots have each won six" | `STATE` | `SUPERSEDED` | a tally, and it had moved on |
| Where was the rocket moved? (2022-09-02) | "NASA is moving the SLS back to the VAB" (2026-02-25) | `STATE` | `ANACHRONISTIC` | describes 2026, after the question |
| Which actress died this week? (2023-02-02) | "Nichelle Nichols, best known for…" | — | `NOT_ABOUT_THIS_QUESTION` | different person; not our question |
| Charge of the ion? (2022-06-17) | "more electrons than protons → net negative" | `TIMELESS` | `NOT_APPLICABLE` | a definition |
| In which year did women first compete? (2024-07-26) | "women first competed in 1900" | `EVENT` | `NOT_APPLICABLE` | a dated historical first |

---

## Before you start: calibration

Label these five, then check yourself against the answers. If you get two or
more wrong, re-read steps 2 and 3 before touching the pack.

1. Q (2025-03-01). Sentence: "The company **is** the largest employer in the state."
2. Q (2025-03-01). Sentence: "The company **opened** its third plant in June 2019."
3. Q (2023-08-14). Sentence: "She **has held** the seat since 2019." (she lost it in 2022)
4. Q (2022-11-01). Sentence: "As of April 2025 the toll **stands** at 12."
5. Q (2024-01-05). Sentence: "Water **boils** at 100 °C at sea level."

<details><summary>Answers</summary>

1. `STATE` → `CURRENT` (unless the excerpt shows it stopped holding)
2. `EVENT` → `NOT_APPLICABLE`
3. `STATE` → `SUPERSEDED`
4. `STATE` → `ANACHRONISTIC` (the state belongs to 2025, after the question)
5. `TIMELESS` → `NOT_APPLICABLE`

</details>

---

## Notes on filling the file

- Label values are typed in **English, uppercase**; the scorer matches those exact strings.
- `notes` takes any language. Use it whenever the rule felt forced — those notes
  are how version 3 gets written.
- Do not discuss items with the other annotators until scoring is done.
- **Avoid opening the CSV in a spreadsheet.** The last returned pack came back
  with every ISO date rewritten to US format. Use a text editor, or import every
  column as Text.

## What happens next

Three annotators label the same 150 rows independently. If Krippendorff's α on
`freshness` reaches **0.6**, the field becomes usable and we re-run it on the
full set. If it does not, we do **not** collect more data under this rule — we
read the `claim_type` column to find out whether the labels or the rule failed:

- raters agree on `claim_type` but differ on `freshness` → the step-3 table is wrong;
- raters differ on `claim_type` → step 2 is still ambiguous and needs version 3.
