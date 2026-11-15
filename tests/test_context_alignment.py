"""Guard the three measurement defects using actual cached contexts and prompts."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
sys.path.insert(0,str(ROOT/'pipeline/analysis'))
from _e0util import window_fn
from gen_hf_notrunc import build_user, SYS

def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def test_visible_window_excludes_hidden_answer():
    visible=window_fn(600); full=window_fn(None)
    text='x '*300+' plutonium'
    assert full('plutonium',text)
    assert not visible('plutonium',text)
    assert visible('plutonium','plutonium '+text)

def test_masked_labels_match_actual_passages_and_hashes():
    proc=ROOT/'data/processed'
    corpus={r['doc_id']:r['text'] for r in rows(proc/'corpus.jsonl')}
    ranked=json.loads((proc/'retrieval_masked.json').read_text())['hybrid']
    labels=rows(ROOT/'results/analysis/r1_nli_masked_600.jsonl')
    expected={(q,d) for q,ds in ranked.items() for d in ds[:5]}
    assert len(labels)==len(expected)
    assert {(r['qid'],r['doc_id']) for r in labels}==expected
    for r in labels:
        text=corpus[r['doc_id']][:600]
        assert r['premise_chars']==600  # cache field denotes configured maximum
        assert r['premise_sha256']==hashlib.sha256(text.encode()).hexdigest()
        assert ranked[r['qid']][r['masked_rank']]==r['doc_id']

def test_corrected_k10_cache_preserves_full_prompt():
    proc=ROOT/'data/processed'
    corpus={r['doc_id']:r for r in rows(proc/'corpus.jsonl')}
    qs={r['qid']:r for r in rows(proc/'questions.jsonl')}
    caches=sorted((proc/'analysis/kabl_out/out').glob('generations_*/hybrid_masked_k10*.jsonl'))
    assert len(caches)==6
    for p in caches:
        rr=rows(p); assert len(rr)==878
        for r in rr:
            assert r['truncated'] is False
            assert 0<r['n_input_tokens']<=4096
            assert len(r['prompt_sha256'])==64
            prompt=build_user(qs[r['qid']],r['doc_ids'],corpus)
            rendered='<|im_start|>system\n'+SYS+'<|im_end|>\n<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n'
            assert hashlib.sha256(rendered.encode()).hexdigest()==r['prompt_sha256']
            assert qs[r['qid']]['question'] in prompt
            assert prompt.endswith('Answer with the shortest exact span, using the documents.')

def test_full_question_stats_match_rows_and_ids():
    proc=ROOT/'data/processed'; rr=rows(proc/'questions_full.jsonl')
    stats=json.loads((proc/'questions_full_stats.json').read_text())
    assert stats['n_question_rows']==len(rr)
    assert stats['n_questions_full']==stats['n_unique_questions']==len({r['qid'] for r in rr})
