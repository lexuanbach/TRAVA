"""Check frozen index dates, exposure populations, fold actions and omitted panels."""
from pathlib import Path
import collections,json,random,math
from metrics import token_f1,gold_in_text
from support_interaction import did_boot
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'data/processed';R=ROOT/'results'
def read(p):return json.loads(p.read_text())
def rows(p,key):return {x[key]:x for x in map(json.loads,p.read_text().splitlines())}
def main():
 qs=rows(P/'questions.jsonl','qid');ds=rows(P/'corpus.jsonl','doc_id');ret=read(P/'retrieval_stale.json')['hybrid_stale'];qrels=read(P/'qrels.json');fresh=[q for q in qs if qs[q]['date']>='2025-01-01'];hits=[q for q in fresh if set(ret[q])&set(qrels[q])];linked=[(q,d) for q in hits for d in ret[q] if d in qrels[q]]
 out={'stale':{'definition':'document publish_date before 2025-01-01','latest_retrieved_publish_date':max(ds[d]['publish_date'] for v in ret.values() for d in v),'fresh_linked_hit_questions':len(hits),'fresh_questions':len(fresh),'fresh_hit_at_5':len(hits)/len(fresh),'linked_passages':len(linked),'linked_passages_with_shown_gold_string':sum(gold_in_text(qs[q]['gold_answer'],ds[d]['text'][:600]) for q,d in linked),'note':'Link membership and gold-string occurrence are not adjudicated answer support.'}}
 from datetime import date,timedelta
 stale_exposure={}
 for regime,is_fresh in [('static',False),('fresh',True)]:
  ids=[q for q in qs if (qs[q]['date']>='2025-01-01')==is_fresh]
  late={q:sum(bool(ds[d].get('publish_date')) and ds[d]['publish_date']>str(date.fromisoformat(qs[q]['date'][:10])+timedelta(days=3)) for d in ret[q]) for q in ids}
  n=sum(len(ret[q]) for q in ids)
  stale_exposure[regime]={'questions':len(ids),'contexts':n,'late_contexts':sum(late.values()),'questions_with_late_context':sum(x>0 for x in late.values()),'late_context_fraction':sum(late.values())/n}
 out['stale']['query_grace_window_exposure']=stale_exposure
 out['undated_documents']={name:sum(not d.get('publish_date') for d in rows(P/name,'doc_id').values()) for name in ['corpus.jsonl','corpus_full.jsonl']}
 e=read(R/'analysis/e2_sample_stats.json');out['exposure']={'corpus':'corpus_full.jsonl, 26611 documents','original_common_years':e['exposure']['original']['hybrid_full_unmasked']['common_years_2022_2025'],'new_sample':e['exposure']['new']['hybrid_full_unmasked']['all'],'original_all_years':e['exposure']['original']['hybrid_full_unmasked']['all']}
 early=[q for q in qs if qs[q]['date']<'2022-07-01'];equiz={q.split('_')[0] for q in early};quiz=sorted({q.split('_')[0] for q in qs});fits=read(R/'analysis/e5_decision.json')['generators']['qwen32b']['fits'];count=0;heldout=[]
 for f in fits:
  sh=quiz.copy();random.Random(20260917+f['repeat']).shuffle(sh);sizes=[len(sh)//5+(i<len(sh)%5) for i in range(5)];lo=sum(sizes[:f['fold']]);test=set(sh[lo:lo+sizes[f['fold']]])
  if f['audit']=='DATE_2022-07-01':count+=1;assert not(test&equiz)
  if test&equiz:heldout.append(f['audit'])
 out['early_policy']={'pre_boundary_questions':early,'pre_boundary_quizzes':sorted(equiz),'date_policy_selected_fits':count,'total_fits':len(fits),'policy_when_early_quiz_held_out':dict(collections.Counter(heldout)),'explanation':'Not globally degenerate. DATE is chosen only when the one early quiz is in training, so test actions match RAG. RAG is chosen when that quiz is held out.'}
 masked=read(R/'analysis/masked_counterfactual__vis600.json');out['masked_static_gains']={m or 'haiku':v['static'] for m,v in masked['per_model'].items()}
 lab={(r['qid'],r['doc_id']):r['nli_label'].startswith('entail') for r in map(json.loads,(R/'analysis/nli_support_labels_600.jsonl').read_text().splitlines())};cb=rows(P/'generations/closed_book.jsonl','qid');hy=rows(P/'generations/hybrid.jsonl','qid');pan={}
 for reg in ['static','fresh']:
  units=collections.defaultdict(lambda:([],[]))
  for q in sorted(qs):
   if (qs[q]['date']>='2025-01-01')!=(reg=='fresh'):continue
   docs=hy[q]['doc_ids'];assert all((q,d) in lab for d in docs)
   sup=any(lab[q,d] for d in docs);g=100*(token_f1(hy[q]['prediction'],qs[q]['gold_answer'])-token_f1(cb[q]['prediction'],qs[q]['gold_answer']));units[q.split('_')[0]][0 if sup else 1].append(g)
  pan[reg]=did_boot(units,5000,20260802)
 out['haiku_unmasked_nli']={'basis':'Existing nli_support_labels_600.jsonl and bare generations directory. Historical generations_* glob omitted this directory, not missing answers or labels.','model':'haiku','B':5000,'seed':20260802,'per_slice':pan}
 n=90;p=30/n;z=1.959963984540054;mid=(p+z*z/(2*n))/(1+z*z/n);rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);out['stale_wilson']={'successes':30,'n':90,'ci95':[mid-rad,mid+rad]}
 # Same-question retrieval gain on all Static questions and the originally admissible subset.
 from datetime import date,timedelta
 import statistics
 subset={}
 for scope in ['all','admissible']:
  ids=[q for q in qs if qs[q]['date']<'2025-01-01' and (scope=='all' or all(not ds[d].get('publish_date') or ds[d]['publish_date']<=str(date.fromisoformat(qs[q]['date'][:10])+timedelta(days=3)) for d in hy[q]['doc_ids']))]
  units=collections.defaultdict(list)
  for q in ids:units[q.split('_')[0]].append(100*(token_f1(hy[q]['prediction'],qs[q]['gold_answer'])-token_f1(cb[q]['prediction'],qs[q]['gold_answer'])))
  keys=sorted(units);rng=random.Random(20260802);draws=[]
  for _ in range(5000):draws.append(statistics.fmean(g for _ in keys for g in units[keys[rng.randrange(len(keys))]]))
  draws.sort();subset[scope]={'n_questions':len(ids),'estimate':statistics.fmean(g for u in units.values() for g in u),'ci95':[draws[125],draws[4875]]}
 out['haiku_admissible_subset']={'bootstrap':'5000 weekly-quiz draws, seed 20260802, question-pooled means','results':subset}
 (R/'reporting_checks.json').write_text(json.dumps(out,indent=2)+'\n');print('Wrote reporting_checks.json')
if __name__=='__main__':main()
