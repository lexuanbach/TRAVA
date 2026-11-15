# Annotation guidelines — TRAVA human validation

Read this once before starting. It should take about 15 minutes. The worked
examples matter more than the prose; if you only read one thing, read them.

**Time budget.** Task A is ~400 items at roughly 20–30 seconds each: about
2.5–3 hours. Task B is ~300 items with up to 3 evidence excerpts each: about
3–4 hours. Do them in sittings of at most 90 minutes. Fatigue shows up in this
kind of data as a drift toward the middle option.

**Do not look anything up.** Judge only from what is on the row. If you find
yourself wanting to search the web, that is a signal to use `UNCLEAR`, which is
a real and useful answer here.

**Do not discuss items with the other annotator** until adjudication. The whole
point of two independent passes is that they are independent.

---

## What this study is for

We score model answers automatically — with token overlap (F1), exact match,
and a multiple-choice referee. Those metrics disagree with each other on a
meaningful fraction of answers, and we do not know which of them is right. Your
labels are the ground truth against which we measure the automatic metrics.

You are **not** evaluating the models. You are evaluating our measuring
instrument. An answer that is obviously right but which our metric scored zero
is the most valuable row in the dataset, not a mistake to smooth over.

You will not be told which model produced an answer, or whether it had access to
retrieved documents. That is deliberate.

---

## Task A — answer correctness

**File:** `task_a_correctness.csv`. **Fill in:** `verdict`, `confidence`, `notes`.

You see a question, its date, a reference answer, and a system answer. Decide
whether the system answer conveys the reference answer.

### The verdict scale

| Verdict | Use when |
|---|---|
| `CORRECT` | The system answer conveys the reference answer, whatever its wording, formatting, or verbosity. |
| `INCORRECT` | The system answer conveys something other than the reference answer, or nothing. |
| `PARTIAL` | The answer is right in part and wrong or incomplete in a way that matters — e.g. it names one of two required entities, or gets the entity right and the number wrong. |
| `ABSTAIN` | The system declined: "I don't know", "I cannot determine", empty. Refusing is not being wrong; we count it separately. |
| `UNCLEAR` | You cannot tell without outside knowledge, or the reference answer itself looks wrong or ambiguous. Use this freely. |

### The rules that decide most cases

1. **Formatting is never a reason to mark incorrect.** These systems were asked
   for different output formats. `B`, `(B)`, `B. positively charged`,
   `{"answer": "B"}`, and `The answer is B.` are all `CORRECT` if B is right.

2. **Aliases and surface variants are correct.** `Squid Game` for `"Squid Game"`.
   `two trillion` for `$2 trillion`. `JFK` for `John F. Kennedy`. `US` for
   `United States`. `2.5 million` for `2,500,000`.

3. **Extra correct material does not hurt.** If the answer states the reference
   answer and then adds accurate context, it is `CORRECT`. If the added material
   is *wrong* and contradicts the answer, use `PARTIAL`.

4. **A more specific answer is correct if it entails the reference.**
   `March 2023` for reference `2023` is `CORRECT`. A *less* specific answer that
   does not pin the reference down is `PARTIAL`: `2023` for reference
   `March 2023`.

5. **Judge against the question's date.** These are news questions. The
   reference answer is what was true when the question was written, which is in
   the `question_date` column. An answer that would be right *today* but was
   wrong then is `INCORRECT`. This is the single most common way to go wrong on
   this task.

6. **If the reference answer looks wrong to you, use `UNCLEAR`** and say so in
   `notes`. Do not silently correct it. We need to know how often this happens.

### Confidence

`HIGH` if you would defend the verdict; `LOW` if you are guessing. Low-confidence
items get priority in adjudication. Roughly 10–20% `LOW` is normal; if you are
above 40%, stop and re-read this document.

### Worked examples

| Question (date) | Reference | System answer | Verdict | Why |
|---|---|---|---|---|
| Which show was green-lit for a new season? (2022-06-16) | `"Squid Game"` | `Squid Game` | `CORRECT` | Quoting is formatting |
| What did the fund reach? (2026-05-15) | `$2 trillion` | `two trillion dollars` | `CORRECT` | Alias |
| Who is White House chief of staff? (2023-02-24) | `Jeff Zients` | `Ron Klain` | `INCORRECT` | True earlier, not at the question's date — rule 5 |
| Which two countries signed? (2024-10-04) | `France and Germany` | `France` | `PARTIAL` | Half the required answer |
| Charge of the atom? (2022-06-17) | `B` | `{"answer": "B"}` | `CORRECT` | Formatting |
| Who won the election? (2025-11-07) | `Zohran Mamdani` | `I don't have information after my training cutoff` | `ABSTAIN` | Declined |
| What was the vote count? (2024-03-08) | `219–212` | `219 to 212` | `CORRECT` | Alias |
| When was it founded? (2023-09-22) | `1995` | `the mid-1990s` | `PARTIAL` | Does not pin down the year |
| Which economist died at 100? (2023-12-22) | `Henry Kissinger` | `Jimmy Carter` | `INCORRECT` | Salient name, wrong answer |

---

## Task B — evidence support and freshness

**File:** `task_b_evidence.csv`. **Fill in:** `supports`, `freshness`, `notes`.

Each row is **one retrieved document** for a question. A question may span up to
three consecutive rows sharing an `item_id`. Judge each row on its own.

### `supports` — does this document contain the answer?

| Value | Use when |
|---|---|
| `YES` | The excerpt states the reference answer, or states it plainly enough that a careful reader would extract it. |
| `PARTIAL` | The excerpt is about the right event and narrows it down, but does not state the answer. |
| `NO` | The excerpt does not contain the answer. |

Judge the **excerpt shown**, not the document it came from. Excerpts are
truncated at 700 characters; if the answer is plausibly just past the cut, that
is still `NO` — we are measuring what the model was shown.

### `freshness` — is this document current *for this question's date*?

**Judge the time the content refers to, not the `evidence_date` column.** The
column is a publication date and it is only a hint: a 2026 retrospective can
describe the state of affairs in 2022 perfectly well, and a document published
before the question can already be out of date.

Ask two questions in order.

1. **Can this fact change over time at all?** If it cannot — a definition, a
   physical constant, a settled historical fact — the answer is
   `NOT_APPLICABLE` and you are done.
2. **What moment does the content describe, compared with `question_date`?**

| Value | Use when |
|---|---|
| `CURRENT` | The content describes the state of affairs as of `question_date`. |
| `SUPERSEDED` | The content describes an *earlier* state that had already been overtaken by `question_date` — the classic stale-evidence case, e.g. naming the previous officeholder. |
| `ANACHRONISTIC` | The content describes a *later* state than `question_date` — information that did not yet exist when the question was asked. |
| `UNDATED` | You cannot place the content in time at all, and the date column does not settle it either. |
| `NOT_APPLICABLE` | Timeless content (step 1). |

`SUPERSEDED` and `ANACHRONISTIC` point in opposite directions and are both
common here. Roughly one row in nine carries evidence published after its
question, because the retrieval index pools every question's evidence without a
per-question time filter. That is a real property of the system we are auditing,
not a mistake in the data, and labelling it correctly is part of the point.

`SUPERSEDED` versus `NOT_APPLICABLE` remains the distinction annotators most
often diverge on. A fact that *cannot* change is `NOT_APPLICABLE`. A fact that
*could* change and has is `SUPERSEDED`. A fact that could change and has not is
`CURRENT`.

`freshness` is judged **independently of `supports`**. All combinations occur: a
document can be `CURRENT` without containing the answer, and `SUPERSEDED` while
containing the answer string.

### Worked examples

| Question (date) | Reference | Excerpt | supports | freshness |
|---|---|---|---|---|
| Who is chief of staff? (2023-02-24) | `Jeff Zients` | "President Biden named Jeff Zients as his next chief of staff…" | `YES` | `CURRENT` |
| Who is chief of staff? (2023-02-24) | `Jeff Zients` | "Ron Klain has served as White House chief of staff since 2021…" | `NO` | `SUPERSEDED` |
| Which country hosted the summit? (2026-07-10) | `Turkey` | "NATO leaders gathered in Ankara this week…" | `YES` | `CURRENT` |
| Which country hosted the summit? (2026-07-10) | `Turkey` | "NATO's 2026 summit will be held in a member state to be confirmed…" | `PARTIAL` | `CURRENT` |
| Charge of an atom with 2 protons, 3 electrons? (2022-06-17) | `negatively charged` | "An ion with more electrons than protons carries a net negative charge." | `YES` | `NOT_APPLICABLE` |
| Where was the Artemis I rocket moved? (2022-09-02) | `Vehicle Assembly Building` | "Feb. 25, 2026 — NASA's SLS rocket is being moved from the launch pad back to the Vehicle Assembly Building for repairs." | `YES` | `ANACHRONISTIC` (2026 content, 2022 question — it happens to match, but it is not evidence available then) |
| Which show was renewed? (2022-06-16) | `"Squid Game"` | "A 2026 retrospective: in June 2022 Netflix confirmed a second season of Squid Game…" | `YES` | `CURRENT` (published 2026, but the content describes June 2022) |

---

## Adjudication

Two annotators (A and B) label every item independently. Then:

1. Run `python annotation/score_annotations.py --stage agreement`. It reports
   raw agreement, Cohen's κ per task, and Krippendorff's α, and writes the
   disagreement list.
2. A third person adjudicates **only** the disagreements, seeing both labels but
   not who gave which. Record the final call in the `adjudicated` column.
3. Re-run with `--stage final` to produce `results/human_validation.json`.

**Do not adjudicate by averaging or by seniority.** If neither label is
defensible, the adjudicated verdict is `UNCLEAR` — that is a legitimate outcome
and we would rather record it than manufacture agreement.

**If agreement is poor** (κ below about 0.6 on Task A), stop. Do not annotate
more items. Poor agreement means the guidelines are ambiguous, not that the
annotators are careless, and more data collected under ambiguous guidelines is
not worth having. Revise the rules, note what changed, and re-annotate a small
pilot before continuing.

---

## What we will report

Per-regime Cohen's κ against the adjudicated labels, with intervals;
Krippendorff's α between annotators; the stratum weights applied so estimates
refer to the population and not to our deliberately skewed sample; and the
calibration-item pass rate. All of it, including the parts that make the
automatic metrics look bad.

If you notice a systematic problem with the questions, the reference answers, or
these instructions, write it in `notes` and tell us. That feedback is part of the
result.
