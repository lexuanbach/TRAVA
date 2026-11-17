"""Guard paired inference, permutation resolution and date-filter scope."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
from paired_label_comparison import compare
from controlled_cutoff import slope_with_perm

def test_identical_labelers_have_zero_paired_difference():
    rows=[];nli={};judge={}
    for quiz in range(6):
        for item, (pred, human) in enumerate([(True,True),(True,False),(False,True),(False,False)]):
            qid=f'q{quiz}_{item}';key=(qid,'doc')
            rows.append(dict(qid=qid,doc_id='doc',quiz=str(quiz),string_w600=pred,human_bin=human))
            nli[key]=pred;judge[(*key,'w600')]={'label':'YES' if pred else 'NO'}
    out=compare(rows,nli,judge,B=200)
    assert out['n_rows']==24 and out['n_quizzes']==6
    for v in out['paired_differences'].values():
        assert v['estimate']==0 and v['ci95']==[0,0]
        assert v['two_sided_p_plus_one']==v['holm_p_three_comparisons_within_pack']==1

def test_absolute_slope_floor_depends_on_size_spacing():
    asymmetric=slope_with_perm([(1,1.16),(3,-.5),(8,11.61)])
    assert asymmetric['p_resolution_floor']==pytest.approx(1/6,abs=1e-5)
    assert asymmetric['p_exact']==pytest.approx(4/6,abs=1e-5)
    symmetric=slope_with_perm([(1,1.16),(10,-.5),(100,11.61)])
    assert symmetric['p_resolution_floor']==pytest.approx(2/6,abs=1e-5)

def test_paired_comparison_uses_frozen_inputs_and_declared_family():
    d=json.loads((ROOT/'results/paired_label_comparison.json').read_text())
    for path,digest in d['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    for pack,n in [('task_b',900),('masked_pack',630)]:
        assert d[pack]['n_rows']==n
        values=list(d[pack]['paired_differences'].values())
        assert len(values)==3
        for v in values:
            assert v['valid_draws']==5000
            assert 0<v['two_sided_p_plus_one']<=v['holm_p_three_comparisons_within_pack']<=1
    assert d['task_b']['paired_differences']['nli600_minus_string600']['ci95'][0]>0
    assert d['task_b']['paired_differences']['nli600_minus_judge600']['ci95'][0]<0

def test_stale_filter_only_implies_query_mask_on_fresh():
    d=json.loads((ROOT/'results/reporting_checks.json').read_text())['stale']['query_grace_window_exposure']
    assert d['static']['contexts']==5*d['static']['questions']==2020
    assert d['static']['late_contexts']==216
    assert d['static']['questions_with_late_context']==141
    assert d['static']['late_context_fraction']==pytest.approx(216/2020)
    assert d['fresh']['late_contexts']==d['fresh']['questions_with_late_context']==0
