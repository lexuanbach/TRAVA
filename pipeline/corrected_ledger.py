"""Version historical support claims and enumerate corrected numeric evidence.

No historical result JSON is overwritten. Corrected rows use exact JSON pointers;
old support-dependent rows remain explicitly superseded for auditability.
"""
import json

AFFECTED = {'support_interaction.json','support_conditioned.json','nli_support.json','routing_baselines.json','masked_counterfactual.json','support_label_validation.json','adaptive_policy.json','matched_strata.json','human_validation.json'}
SOURCES = {
 '../human_metric_audit.json':'app:endpointbias',
 '../reporting_summary.json':'tab:mc',
 '../masked_human_support.json':'app:jointpack',
 'e0_raw/support_interaction__vis600.json':'tab:support',
 'e0_raw/support_interaction__nli600_labels.json':'tab:support',
 'e0_raw/support_conditioned__vis600.json':'sec:support',
 'manuscript/masked_counterfactual__vis600.json':'tab:masked',
 'r1_masked_nli.json':'sec:support', 'r1_remaining_string_support.json':'sec:support',
 'r2_human_alignment.json':'tab:support-validation','r2_dated_nli.json':'sec:nli_support',
 'stale_placebo.json':'tab:interventions','hidden_answer_contrast.json':'app:hidden',
 'l1_gap_differences.json':'sec:questiontime', 'e3_window.json':'sec:window',
 'e2_replication.json':'app:replication','e2_sample_stats.json':'app:replication',
 'e6_kablation.json':'app:kablation','r3_r5_checks.json':'app:kablation',
 'e5_decision.json':'tab:decision-main','r4_e5_outer.json':'tab:decision-main',
 'r6_mistral_native.json':'app:repro','r8_judge_validation.json':'app:judge',
 'r8_judge_interaction.json':'app:judge','token_length_audit.json':'app:repro',
}
EXCLUDE={'old_fulltext','old','old_ranges_for_record','reproduction','repro','label_flips','per_question','rows','invalid_frozen_k10','frozen','predictions','p_one_sided_perm','estimated_cutoff','per_question_records','by_question','bootstrap_draws','draws','timing','runtime'}
def reconcile(rows, root):
 for r in rows:
  if r[4].split(':')[0].split('/')[-1] in AFFECTED:
   r[5]='historical-superseded';r[2]='HISTORICAL ONLY: pre-correction label or context; use revised/ rows. '+r[2]
 def walk(v,path=''):
  if isinstance(v,bool):return
  if any(len(part)>=8 and part[:8].isdigit() for part in path.split('/')):return
  if isinstance(v,(int,float)):
   yield path,str(v)
  elif isinstance(v,list) and 0<len(v)<=4 and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in v):
   yield path,json.dumps(v)
  elif isinstance(v,dict):
   for k,x in sorted(v.items()):
    if k not in EXCLUDE and not k.startswith('old_'): yield from walk(x,path+'/'+k.replace('~','~0').replace('/','~1'))
 for name,loc in SOURCES.items():
  public_name = name.replace('manuscript/', '')
  p=(root/'results/analysis'/public_name).resolve()
  if not p.exists(): raise FileNotFoundError(p)
  for ptr,val in walk(json.loads(p.read_text())):
   rows.append(['revised/'+name+'#'+ptr,val,'Exact JSON pointer; units, estimand and CI level are defined by parent result object.',loc,str(p.relative_to(root.resolve()))+':'+ptr,'corrected-evidence'])
 return rows
