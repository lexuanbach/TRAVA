"""Guard provider date semantics and analysis-panel membership."""
import importlib.util
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((ROOT/name).read_text())
def test_provider_dates_do_not_collapse_distinct_claims():
    d=read('results/verified_slice.json')['cutoff_evidence']
    h=d['haiku']
    assert h['training_data_through']=='2025-07'
    assert h['reliable_knowledge']==h['public_internet_through_system_card']=='2025-02'
    qs=[json.loads(x) for x in (ROOT/'data/processed/questions.jsonl').read_text().splitlines()]
    fresh=[q for q in qs if q['date']>='2025-01-01']
    assert len(fresh)==474
    assert sum(q['date'][:7]<='2025-07' for q in fresh)==95
    assert sum(q['date'][:7]<='2025-02' for q in fresh)==23

def test_cutoff_relative_keeps_internet_date_sensitivity():
    d=read('results/cutoff_relative.json');h=d['per_model']['haiku']
    assert h['cutoff']=='2025-07'
    assert h['public_internet_date_sensitivity']['cutoff']=='2025-02'
    assert h['own_boundary']['n_quizzes'] != h['public_internet_date_sensitivity']['own_boundary']['n_quizzes']
    assert d['scale_within_matched_cutoff']['matched_groups']['2025-07']==['haiku','sonnet']

def test_inventory_separates_support_and_decision_panels():
    d=read('docs/model_inventory.json')['panels']
    assert len({p['name'] for p in d})==len(d)
    panels={p['name']:p for p in d}
    for name in ['Decision']:
        p=panels[name]
        assert len(p['members'])==9
        assert 'haiku' not in {m['model'] for m in p['members']}
        assert all(n==878 for m in p['members'] for n in m['coverage'].values())
    assert len(panels['Support (string/NLI)']['members'])==10
    assert len(panels['Unmasked support']['members'])==10
    assert 'haiku' in {m['model'] for m in panels['Unmasked support']['members']}

def test_legacy_haiku_key_has_explicit_identity():
    d=read('results/analysis/masked_counterfactual__vis600.json')
    assert d['model_key_aliases']['']=='haiku'
    assert '' in d['per_model']
    normalize=lambda entries: {d['model_key_aliases'].get(k,k) for k in entries}
    assert normalize(d['per_model'])==normalize(d['support_interaction_masked']['string'])
    assert len(normalize(d['per_model']))==10
    assert 'haiku' in normalize(d['per_model'])
