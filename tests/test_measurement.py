"""Checks for metric denominators, annotation units, and capability centering."""
import collections
import csv
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'annotation/pack_2026-08-02'
def read_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def audit():
    return json.loads((ROOT/'results/human_metric_audit.json').read_text())

def test_adjudicated_metric_denominators():
    key = {r['item_id']:r for r in read_jsonl(PACK/'item_metadata.jsonl') if r.get('task')=='A'}
    orig = {r['item_id']:r for r in csv.DictReader((PACK/'task_a_correctness.csv').open())}
    raters = [{r['item_id']:r['verdict'] for r in csv.DictReader(p.open())} for p in sorted(PACK.glob('task_a_correctness__annotator?.csv'))]
    labels = {k:collections.Counter(r[k] for r in raters).most_common(1)[0][0] for k in key}
    overrides = {r['item_id']:r['adjudicated'] for r in csv.DictReader((PACK/'task_a_correctness__adjudicated.csv').open()) if r['adjudicated']}
    assert len(key)==395 and len(overrides)==34
    labels.update(overrides)
    weights = json.loads((PACK/'sampling_manifest.json').read_text())['task_a']['strata']
    for group in ['overall','Static','Fresh']:
        ids=[k for k in key if group=='overall' or ('Static' if orig[k]['question_date']<'2025-01-01' else 'Fresh')==group]
        total=sum(weights[key[k]['stratum']]['weight_to_population'] for k in ids)
        correct=sum(weights[key[k]['stratum']]['weight_to_population'] for k in ids if labels[k]=='CORRECT')
        missed=sum(weights[key[k]['stratum']]['weight_to_population'] for k in ids if labels[k]=='CORRECT' and key[k]['auto_f1']<.5)
        s=audit()['consistent_summaries'][group]
        assert s['miss_prevalence']['estimate']==pytest.approx(missed/total)
        assert s['conditional_miss_rate']['estimate']==pytest.approx(missed/correct)
        assert s['human_correct_rate']['estimate']==pytest.approx(correct/total)

def test_common_resampling_and_intervals():
    d=audit();assert d['bootstrap']==dict(replicates=5000,seed=20260802,unit='answer item, sampled within original strata',interval='percentile 95%',shared_draws=True)
    for metrics in d['consistent_summaries'].values():
        for rec in metrics.values():
            assert rec['valid_draws']==5000
            assert rec['ci95'][0]<=rec['estimate']<=rec['ci95'][1]
    a=d['consistent_summaries'];assert a['Fresh']['conditional_miss_rate']['estimate'] < a['overall']['conditional_miss_rate']['estimate'] < a['Static']['conditional_miss_rate']['estimate']

def test_annotation_counting_units():
    b=[r for r in read_jsonl(PACK/'item_metadata.jsonl') if r.get('task')=='B']
    assert len(b)==300 and len({r['qid'] for r in b})==254
    rows=list(csv.DictReader((PACK/'task_b_evidence.csv').open()))
    assert len(rows)==900 and set(collections.Counter(r['item_id'] for r in rows).values())=={3}
    masked=read_jsonl(ROOT/'annotation/masked_support_2026-08-04/item_metadata.jsonl')
    assert len(masked)==630 and len({r['qid'] for r in masked})==210
    assert len({r['qid'] for r in masked if r['stratum']=='consistency_recheck'})==10
    assert len({r['qid'] for r in masked if r['stratum']!='consistency_recheck'})==200

def test_masked_population_excludes_consistency_cases():
    d=json.loads((ROOT/'results/masked_human_support.json').read_text())
    assert d['n_population_questions']==200 and d['n_consistency_questions']==10
    strata=json.loads((ROOT/'annotation/masked_support_2026-08-04/sampling_manifest.json').read_text())['strata']
    assert sum(x['population_questions'] for x in strata.values())==878
    assert sum(x['weight']*x['sampled_questions'] for x in strata.values())==pytest.approx(878)
    assert d['weighted_support_rate']==pytest.approx(.3821)
    assert d['per_model']['qwen32b']['static']['n_supported']+d['per_model']['qwen32b']['static']['n_unsupported']==103

def test_capability_centering_matches_record():
    d=json.loads((ROOT/'results/multilevel_capability.json').read_text())
    assert len(d['capability_by_model'])==8
    assert sum(d['capability_by_model'].values())/8==pytest.approx(d['capability_mean'],abs=.0001)
    assert d['capability_mean']==pytest.approx(36.4664)
