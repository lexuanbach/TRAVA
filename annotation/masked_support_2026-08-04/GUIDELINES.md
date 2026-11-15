# Annotation guidelines — evidence support

Read this once before starting. It takes about ten minutes. The worked examples
matter more than the prose. If you read only one section, read the examples.

**Your file:** `task_support_masked.csv`
**You fill in:** `supports` and `notes`. Nothing else.
**Size:** 630 rows, grouped as 210 questions with 3 evidence excerpts each.
**Time:** roughly 15 seconds a row, so about 2.5 hours. Work in sittings of at
most 90 minutes. Fatigue shows up in this kind of data as drift toward the
middle option.

---

## Three rules

1. **Do not look anything up.** Judge only from what is on the row. If you want
   to search the web, that is exactly when to use `UNCLEAR`. It is a real and
   useful answer, not a failure.
2. **Do not discuss items with the other annotators** until adjudication is
   finished. Independence is the whole point of running three passes.
3. **Judge each row on its own.** A question spans three consecutive rows that
   share nothing but the question text. Do not let one excerpt change how you
   read another, and do not try to be consistent across the three.

---

## What you are deciding

For each row you see a **question**, the **date it was asked**, a **reference
answer**, and **one retrieved document excerpt** with its publication date.

The only question you answer is:

> Does this excerpt give a careful reader the reference answer?

You are **not** judging whether the document is interesting, well written,
recent, on-topic, or trustworthy. You are **not** judging any model. No model
output appears anywhere in your file.

You will not be told which system retrieved the document or under what
conditions. That is deliberate, and it is why your labels are usable as ground
truth.

---

## The `supports` scale

| Value | Use when |
|---|---|
| `YES` | The excerpt states the reference answer, or states it plainly enough that a careful reader would extract it. Different wording is fine. |
| `PARTIAL` | The excerpt is clearly about the right event and narrows things down, but does not actually state the answer. |
| `NO` | The excerpt does not contain the answer. This includes excerpts on the right topic that simply never say it. |
| `UNCLEAR` | You cannot tell without outside knowledge, or the reference answer itself looks wrong or ambiguous. Use freely. |

**`NO` is the most common correct label.** Most retrieved documents are about
the right topic and still do not state the answer. That is the finding this
study exists to measure, so do not feel that a page of `NO` means you are doing
it wrong.

---

## Worked examples

**Question** (2026-05-14): *Where will the 2026 World Cup championship match take
place?* **Reference answer:** New Jersey

| Excerpt | Label | Why |
|---|---|---|
| "The final will be played at MetLife Stadium in East Rutherford, New Jersey." | `YES` | States it. |
| "The final will be held at MetLife Stadium." | `YES` | A careful reader gets there: MetLife is named as the venue and the answer is its location. Wording need not match. |
| "Tickets for the final go on sale Monday; organisers expect record demand." | `PARTIAL` | Right event, narrows nothing about location. |
| "The 2026 tournament will be hosted across the United States, Canada and Mexico." | `PARTIAL` | Right event, and it constrains the answer without stating it. |
| "Spencer Pratt lost ground in the Los Angeles mayoral race." | `NO` | Unrelated. |
| "The championship match returns to the region for the first time since 1994." | `UNCLEAR` | You would need to know which region to judge this. |

Note the second row. **Paraphrase counts as `YES`.** An automatic string match
would score it zero, and rows like it are the most valuable in the dataset.

Note the fourth row. `PARTIAL` is for genuine narrowing, not for "vaguely
related". If the excerpt would not move a careful reader closer to the specific
answer, it is `NO`.

---

## `notes`

Optional, and worth the seconds when:

- you nearly chose a different label, so write which and why,
- the reference answer looks wrong or ambiguous to you,
- the excerpt is truncated at exactly the wrong point.

One short clause is plenty. `notes` is never scored, but it is read during
adjudication and it is where disagreements get resolved quickly.

---

## What happens to your labels

Three people label the same 630 rows independently. Where all three agree, that
is the label. Where they disagree, a fourth person who has not seen the vote
counts adjudicates. Only then is anything computed.

Your labels become the ground truth for one specific measurement: whether the
gain a retrieval system shows depends on the retrieved evidence actually
supporting the answer. If your labels disagree with our automatic ones, the
automatic ones are what get corrected.

---

## Do not

- Do not open `item_metadata.jsonl` if it reached you. Tell the
  coordinator. It maps rows to systems and to our automatic labels, and seeing
  it invalidates your pass.
- Do not fill `supports` with anything outside the four values above. The scorer
  matches those exact strings, uppercase.
- Do not leave a row blank. Use `UNCLEAR` instead.
- Do not reorder, sort or delete rows. Return the file with the same rows in the
  same order, with only your two columns filled.
