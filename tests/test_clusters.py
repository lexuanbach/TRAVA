"""Guard the weekly-quiz resampling unit using the frozen questions."""
import json
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from clusters import quiz_id, assert_canonical_clusters, EXPECTED_N_CLUSTERS

def questions():
    return [json.loads(x) for x in (ROOT / "data/processed/questions.jsonl").read_text().splitlines()]

def test_quiz_identity_and_count():
    qs = questions()
    assert quiz_id("20260130_3") == "20260130"
    with pytest.raises(ValueError): quiz_id("noseparator")
    assert len({quiz_id(q["qid"]) for q in qs}) == EXPECTED_N_CLUSTERS
    assert_canonical_clusters([q["qid"] for q in qs])

def test_date_does_not_replace_quiz_block():
    qs = questions(); groups = {}
    for q in qs: groups.setdefault(quiz_id(q["qid"]), set()).add(q["date"][:10])
    assert sum(len(ds) > 1 for ds in groups.values()) == 86
    assert len({q["date"][:10] for q in qs}) != len(groups)

@pytest.mark.parametrize("name,key", [("questions.jsonl", "qid"), ("questions_full.jsonl", "qid"), ("corpus.jsonl", "doc_id"), ("corpus_full.jsonl", "doc_id")])
def test_snapshot_ids_are_unique(name, key):
    ids = [json.loads(x)[key] for x in (ROOT / "data/processed" / name).read_text().splitlines()]
    assert len(ids) == len(set(ids))
