#!/usr/bin/env python3
"""The canonical resampling unit for this benchmark.

Every interval and every p-value in this paper is a block bootstrap or a
cluster-robust test, and all of them must resample the SAME unit. An earlier
version of `joint_slope.py` did not: it clustered on the question's publication
DATE while every other script clustered on the weekly QUIZ id. The two are not
the same -- 86 dates carry items from more than one quiz -- so the date version
split 199 true clusters into 282 pieces and reported an interval narrower than
the design supports. The manuscript then contradicted itself, quoting 199
clusters in the setup and 282 in the headline analysis.

This module is the single definition. Import `quiz_id` rather than writing
`qid.split("_")[0]` again, and `assert_canonical_clusters` in any script that
resamples. `tests/test_clusters.py` enforces both.

Why the quiz and not the date: RealTimeQA ships questions in weekly quiz
batches. The batch is the unit of collection -- questions in one batch share a
compiler, a news window and an editorial style -- so it is the level at which
outcomes are correlated. The date is a property of the batch, not a unit of
sampling, and it is neither finer nor coarser than the batch in a well-behaved
way: most dates carry one quiz, but a substantial minority carry several.
"""
from __future__ import annotations

CANONICAL_UNIT = "weekly_quiz_id"
EXPECTED_N_CLUSTERS = 199  # over data/processed/questions.jsonl


def quiz_id(qid: str) -> str:
    """The weekly quiz batch a question belongs to.

    Question ids are `<quizid>_<index>`, so the batch is the prefix.
    """
    if "_" not in qid:
        raise ValueError(f"question id {qid!r} has no quiz prefix")
    return qid.split("_")[0]


def quiz_map(qids) -> dict[str, str]:
    return {q: quiz_id(q) for q in qids}


def assert_canonical_clusters(qids, expected: int | None = None) -> int:
    """Fail loudly if a script is about to resample the wrong unit."""
    n = len({quiz_id(q) for q in qids})
    exp = EXPECTED_N_CLUSTERS if expected is None else expected
    if expected is not None or n != exp:
        if n != exp:
            raise AssertionError(
                f"expected {exp} {CANONICAL_UNIT} clusters, found {n}. "
                "If the question set changed, update EXPECTED_N_CLUSTERS and "
                "the cluster count quoted in the experimental setup together."
            )
    return n
