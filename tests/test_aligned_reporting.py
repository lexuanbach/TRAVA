import json,math
from pathlib import Path
import pytest
R=Path(__file__).resolve().parents[1]
def load(n):return json.loads((R/'results'/n).read_text())
def test_weighted_kappa_uses_same_confusion_table():
 d=load('human_metric_audit.json')
 for group in ['overall','Static','Fresh']:
  x={k:v['estimate'] for k,v in d['consistent_summaries'][group].items()};h=x['human_correct_rate'];fn=x['miss_prevalence'];fp=(1-h)*x['false_credit_rate'];pred=h-fn+fp;pe=h*pred+(1-h)*(1-pred)
  assert x['cohens_kappa']==pytest.approx((x['agreement']-pe)/(1-pe))
  assert d['consistent_summaries'][group]['cohens_kappa']['valid_draws']==5000
 assert d['consistent_summaries']['overall']['cohens_kappa']['estimate']>.7
 assert d['unweighted_summaries']['overall']['cohens_kappa']==pytest.approx(.35642,abs=1e-5)
def test_slopes_share_panels_and_reproduce_independent_ols():
 d=load('aligned_contrasts.json');params=[.5,1.5,3,7,32,72,123];xs=[math.log10(p) for p in params]
 for c in d['contrasts'].values():
  for wd in c['results'].values():
   assert set(wd['token_f1']['per_model'])==set(wd['choice_accuracy']['per_model'])==set(d['models'])
   for ed in wd.values():
    for panel,n in [('seven_rungs',7),('qwen_only',6)]:
     rec=ed['slopes'][panel];assert rec['models']==d['models'][:n];ys=[ed['per_model'][m]['estimate'] for m in rec['models']];mx=sum(xs[:n])/n;my=sum(ys)/n;expected=sum((x-mx)*(y-my) for x,y in zip(xs[:n],ys))/sum((x-mx)**2 for x in xs[:n]);assert rec['estimate']==pytest.approx(expected)
 assert d['contrasts']['years_2023_2026']['n_pre']==131
 assert d['contrasts']['years_2023_2026']['n_post']==346
 assert d['contrasts']['boundary_2025']['results']['question_pooled']['token_f1']['slopes']['seven_rungs']['estimate']==pytest.approx(load('joint_slope.json')['slope_f1_per_decade']['observed'],abs=.0001)
def test_half_year_average_is_not_selected_maximum():
 d=load('aligned_contrasts.json')
 for w,x in d['half_year_sensitivity'].items():
  assert len(x['boundaries'])==8 and not x['excluded']
  for m,v in x['mean_across_valid_boundaries'].items():assert v['estimate']==pytest.approx(sum(b['per_model'][m]['estimate'] for b in x['boundaries'].values())/8)
  assert x['boundaries']['2022-07-01']['n_pre']==7
  assert 0<x['mean_across_valid_boundaries']['qwen']['valid_draws']<5000

def test_exposure_and_fold_conditions_are_explicit():
 d=load('reporting_checks.json');e=d['exposure'];a=e['original_common_years'];b=e['new_sample'];assert a['n_contexts']==2660 and b['n_contexts']==4000;assert a['ctx_after_q_date_plus3']['n']==453 and b['ctx_after_q_date_plus3']['n']==659
 f=d['early_policy'];assert len(f['pre_boundary_questions'])==7;assert f['date_policy_selected_fits']==80;assert f['policy_when_early_quiz_held_out']=={'RAG':20}
 assert d['stale']['latest_retrieved_publish_date']<'2025-01-01';assert d['stale']['fresh_linked_hit_questions']==288

def test_masked_harm_and_haiku_completion():
 d=load('reporting_checks.json');bad={m for m,x in d['masked_static_gains'].items() if x['delta_rag_masked']['ci95'][1]<0};assert bad=={'mistral_large','sonnet','gpt4omini'}
 for s,x in d['haiku_unmasked_nli']['per_slice'].items():assert x['n_supported']+x['n_unsupported']==(404 if s=='static' else 474);assert x['ci95'][0]>0
 assert d['haiku_admissible_subset']['results']['admissible']['n_questions']==179
 assert d['stale_wilson']['ci95']==pytest.approx([.2444,.4358],abs=.0003)


def test_bootstrap_zero_ties_are_platform_stable():
 import sys
 import numpy as np
 sys.path.insert(0, str(R / 'pipeline' / 'analysis'))
 from analysis_common import summarize
 # Exact zeros and small signed summation residuals represent the same ties.
 base = np.array([-1.0] * 60 + [0.0] * 20 + [1.0] * 120)
 perturbed = base.copy()
 perturbed[60:80] = np.tile([-1e-15, 1e-15], 10)
 expected = 2 * (80 + 1) / 201
 assert summarize(0.3, base)['p_two_sided_boot'] == round(expected, 5)
 assert summarize(0.3, perturbed)['p_two_sided_boot'] == round(expected, 5)
 # A genuine small nonzero draw must retain its direction.
 perturbed[60:80] = 1e-9
 assert summarize(0.3, perturbed)['p_two_sided_boot'] == round(2 * 61 / 201, 5)
