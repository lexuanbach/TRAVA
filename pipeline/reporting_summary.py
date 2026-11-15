#!/usr/bin/env python3
"""Recompute reporting summaries from cached answers, without model calls."""
from pathlib import Path
import collections,hashlib,json,math,random,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from choice_grounded_audit import nearest_choice,is_abstention
from metrics import token_f1
from masked_counterfactual import load,load_questions,quiz_of
MODELS=['qwen','qwen1_5b','qwen3b','qwen7b','qwen32b','qwen72b','mistral_large','','sonnet','llama3_8b','gpt4omini']
PARAMS=dict(zip(MODELS[:7],[.5,1.5,3,7,32,72,123]))
def mean(x):return statistics.fmean(x)
def slope(xs,ys):
 xm,ym=mean(xs),mean(ys)
 return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
def main():
 qs=load_questions();out={'scope':'Cached-answer summaries on the fixed 878-question sample','n_questions':len(qs),'choice_weighting':'question-pooled within each temporal slice','choice_rule':'nearest_choice from pipeline/choice_grounded_audit.py; unmatched responses score zero; detected abstentions counted separately','models':{},'inputs':{}}
 for model in MODELS:
  rec={}
  for cond in ['closed_book','hybrid']:
   rows=load(model,cond)
   if not rows:continue
   ids=sorted(set(rows)&set(qs));assert len(ids)==len(qs),(model,cond,len(ids))
   f=ROOT/'data/processed'/('generations_'+model if model else 'generations')/(cond+'.jsonl')
   out['inputs'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
   slices={}
   for label in ['all','static','fresh']:
    selected=[q for q in ids if label=='all' or (qs[q]['year']<2025)==(label=='static')]
    correct=abstain=unmatched=0
    for q in selected:
     pred=rows[q]['prediction'] or '';idx,_=nearest_choice(pred,qs[q]['choices'])
     correct+=int(idx is not None and idx==qs[q]['gold_answer_idx']);unmatched+=idx is None;abstain+=is_abstention(pred)
    slices[label]={'n':len(selected),'correct':correct,'accuracy':100*correct/len(selected),'detected_abstentions':abstain,'unmatched':unmatched,'token_f1':mean([100*token_f1(rows[q]['prediction'] or '',qs[q]['gold_answer']) for q in selected])}
   rec[cond]=slices
  out['models'][model or 'haiku']=rec
 gaps={}
 for model in PARAMS:
  rows=load(model,'closed_book');by={year:collections.defaultdict(list) for year in [2023,2026]}
  for q in sorted(qs):
   if qs[q]['year'] in by:by[qs[q]['year']][quiz_of(q)].append(100*token_f1(rows[q]['prediction'] or '',qs[q]['gold_answer']))
  gaps[model]=mean([mean(v) for v in by[2023].values()])-mean([mean(v) for v in by[2026].values()])
 cg=json.loads((ROOT/'results/choice_grounded_audit.json').read_text())
 out['matched_scale_contrast']={'years':[2023,2026],'weighting':'equal quiz means within each year','models':list(PARAMS),'token_f1_gaps':gaps,'token_f1_per_decade':slope([math.log10(PARAMS[m]) for m in PARAMS],list(gaps.values())),'choice_accuracy_per_decade':cg['scale_slope_choice_grounded'],'note':'The matching concerns the estimand, model panel and question sample. No new model answers.'}
 # Match masked_counterfactual.py ordering, random generator, seed and percentile indexing.
 un,ma=load('qwen1_5b','hybrid'),load('qwen1_5b','hybrid_masked');by=collections.defaultdict(list)
 for q in sorted(qs):by[quiz_of(q)].append(100*(token_f1(ma[q]['prediction'] or '',qs[q]['gold_answer'])-token_f1(un[q]['prediction'] or '',qs[q]['gold_answer'])))
 values=[mean(v) for v in by.values()];rng=random.Random(20260803);B=5000
 draws=sorted(mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(B))
 out['frozen_qwen1_5b_mask']={'estimate':mean(values),'ci90':[draws[int(.05*B)],draws[int(.95*B)]],'ci95':[draws[int(.025*B)],draws[int(.975*B)]],'B':B,'seed':20260803,'weighting':'quiz means','n_quizzes':len(values),'n_questions':len(qs),'equivalent_within_2':draws[int(.05*B)]>-2 and draws[int(.95*B)]<2}
 p=ROOT/'results/reporting_summary.json';p.write_text(json.dumps(out,indent=2)+'\n');print(p)
if __name__=='__main__':main()
