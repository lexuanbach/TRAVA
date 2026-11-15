"""Reporting contracts checked independently against cached evidence."""
import csv,json,subprocess,sys,collections
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from choice_grounded_audit import nearest_choice
from masked_counterfactual import load,load_questions

def test_choice_counts_and_common_population():
 d=json.loads((ROOT/'results/reporting_summary.json').read_text());qs=load_questions()
 cg=json.loads((ROOT/'results/choice_grounded_audit.json').read_text())
 assert cg['n_questions']==878 and set(cg['panel'])==set(d['models'])
 assert all(x['n_predictions']==878 for x in cg['per_model'].values())
 for model,conditions in d['models'].items():
  for cond,slices in conditions.items():
   rows=load('' if model=='haiku' else model,cond)
   correct=sum(nearest_choice(rows[q]['prediction'] or '',qs[q]['choices'])[0]==qs[q]['gold_answer_idx'] for q in qs)
   assert slices['all']['correct']==correct
   assert slices['static']['n']==404 and slices['fresh']['n']==474
   assert slices['static']['correct']+slices['fresh']['correct']==correct
   assert slices['all']['accuracy']==pytest.approx(100*correct/878)

def test_matched_scale_estimands():
 d=json.loads((ROOT/'results/reporting_summary.json').read_text())['matched_scale_contrast']
 m=json.loads((ROOT/'results/matched_strata.json').read_text())
 assert d['years']==[2023,2026]
 assert set(d['models'])==set(d['choice_accuracy_per_decade']['per_model'])
 for key,gap in d['token_f1_gaps'].items():assert gap==pytest.approx(m['per_model'][key]['raw']['gap'],abs=.00051)
 assert d['token_f1_per_decade']==pytest.approx(3.4575,abs=.0001)

def test_frozen_equivalence_is_not_claimed():
 d=json.loads((ROOT/'results/reporting_summary.json').read_text())['frozen_qwen1_5b_mask']
 m=json.loads((ROOT/'results/analysis/masked_counterfactual__vis600.json').read_text())['per_model']['qwen1_5b']['all']['delta_mask']
 assert d['estimate']==pytest.approx(m['delta'],abs=.00051)
 assert d['ci95']==pytest.approx(m['ci95'],abs=.00051)
 assert d['ci90'][1]>2 and not d['equivalent_within_2']

def test_adjudicated_items_distinct_from_changed_labels():
 p=ROOT/'annotation/pack_2026-08-02'
 raters=[{r['item_id']:r['verdict'].strip().upper() for r in csv.DictReader(f.open())} for f in sorted(p.glob('task_a_correctness__annotator?.csv'))]
 adj={r['item_id']:r['adjudicated'].strip().upper() for r in csv.DictReader((p/'task_a_correctness__adjudicated.csv').open()) if r['adjudicated'].strip()}
 changes=sum(collections.Counter(d[k] for d in raters).most_common(1)[0][0]!=v for k,v in adj.items())
 d=json.loads((ROOT/'results/human_metric_audit.json').read_text())
 assert d['n_adjudicated_items']==d['n_adjudication_overrides']==len(adj)==34
 assert d['n_labels_changed_by_adjudication']==changes==18

def test_panel_coverage_and_dates():
 d=json.loads((ROOT/'docs/model_inventory.json').read_text())
 p={x['name']:x for x in d['panels']}
 assert len(p['Cutoff panel']['members'])==9
 assert len(p['Unmasked Hybrid']['members'])==10
 assert len(p['Decision']['members'])==9
 for name in ['Cutoff panel','Unmasked Hybrid','Decision']:
  assert all(n==878 for m in p[name]['members'] for n in m['coverage'].values())
 qs=[json.loads(x) for x in (ROOT/'data/processed/questions_heldout.jsonl').read_text().splitlines()]
 assert len({q['qid'].split('_')[0] for q in qs})==2
 assert len({q['date'] for q in qs})==3

def test_corrected_ledger_sources_and_values():
 rows=list(csv.DictReader((ROOT/'results/claim_ledger.csv').open(newline='')))
 cache={}
 for row in rows:
  path,_,ptr=row['provenance'].partition(':')
  assert (ROOT/path).is_file(),path
  if row['status']!='corrected-evidence':continue
  if path not in cache:cache[path]=json.loads((ROOT/path).read_text())
  obj=cache[path]
  for part in ptr.split('/')[1:]:obj=obj[part.replace('~1','/').replace('~0','~')]
  assert json.loads(row['value'])==obj

def test_ledger_drift_detection(tmp_path):
 script=ROOT/'pipeline/rebuild_ledger.py';path=tmp_path/'ledger.csv'
 subprocess.run([sys.executable,str(script),'--output',str(path)],check=True,capture_output=True)
 assert subprocess.run([sys.executable,str(script),'--output',str(path),'--check'],capture_output=True).returncode==0
 path.write_bytes(path.read_bytes().replace(b'5.9252',b'9.9252',1))
 assert subprocess.run([sys.executable,str(script),'--output',str(path),'--check'],capture_output=True).returncode!=0

def test_ordered_rank_changes_distinct_from_set_changes():
 proc=ROOT/'data/processed'
 a=json.loads((proc/'retrieval.json').read_text())['hybrid']
 b=json.loads((proc/'retrieval_masked.json').read_text())['hybrid']
 ids=set(load_questions());ordered=sum(a[q][:5]!=b[q][:5] for q in ids)
 sets=sum(set(a[q][:5])!=set(b[q][:5]) for q in ids)
 displaced=sum(len(set(a[q][:5])-set(b[q][:5])) for q in ids)
 d=json.loads((ROOT/'results/analysis/e3_exposure.json').read_text())['arms']['w3']['all']['topk_differs_vs_unmasked']
 assert ordered==d['ordered_n']==334
 assert sets==d['set_n']==328
 assert displaced==569

def test_routing_interval_and_holm_distinction():
 d=json.loads((ROOT/'results/routing_baselines.json').read_text())
 rows={m:v['vs_reference']['overlap'] for m,v in d['models'].items()}
 assert sum(v['ci95'][0]*v['ci95'][1]>0 for v in rows.values())==4
 significant={m for m,v in rows.items() if v['p_holm']<.05}
 assert significant==set(d['overlap_summary']['significant_models'])
 assert len(significant)==3
 # Llama's unadjusted interval excludes zero, but its adjusted test does not.
 assert rows['llama3_8b']['ci95'][0]>0 and rows['llama3_8b']['p_holm']>.05

def test_provider_cutoff_metadata_distinguishes_haiku_and_sonnet():
 d=json.loads((ROOT/'results/verified_slice.json').read_text())['cutoff_evidence']
 assert d['haiku']['training_data_through']=='2025-02'
 assert d['sonnet']['training_data_through']=='2025-07'
 assert d['sonnet']['reliable_knowledge']=='2025-01'
