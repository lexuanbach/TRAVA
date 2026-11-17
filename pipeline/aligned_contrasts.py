#!/usr/bin/env python3
"""Matched endpoint/weighting sensitivity from frozen closed-book answers only.

Joint weekly-quiz bootstrap, fixed roster. Half-year summaries are retrospective
sensitivity analyses, not independent validation of a data-selected boundary.
"""
import argparse, itertools, json, math
from pathlib import Path
import numpy as np
from choice_grounded_audit import load_questions, load_predictions, nearest_choice
from metrics import token_f1
from joint_slope import LADDER
from clusters import quiz_id
ROOT=Path(__file__).resolve().parents[1]
BOUNDARIES=['2022-07-01','2023-01-01','2023-07-01','2024-01-01','2024-07-01','2025-01-01','2025-07-01','2026-01-01']
def slope(x,y):
 x=np.asarray(x); z=x-x.mean();return np.asarray(y)@z/(z@z)
def summary(value,draws):
 ds=np.sort(draws[np.isfinite(draws)]);n=len(ds)
 return {'estimate':float(value),'ci95':[float(ds[int(.025*n)]),float(ds[int(.975*n)])], 'p_bootstrap_two_sided':min(1.,2*min(1+np.sum(ds<=0),1+np.sum(ds>=0))/(n+1)), 'valid_draws':n}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--B',type=int,default=5000);ap.add_argument('--seed',type=int,default=20260731);ap.add_argument('--output',type=Path,default=ROOT/'results/aligned_contrasts.json');a=ap.parse_args()
 qs=load_questions();pred=load_predictions();models=[m for m,_,_ in LADDER]+['haiku','sonnet','gpt4omini'];ids=sorted(qs)
 assert all(set(ids)<=set(pred[m]) for m in models)
 quizzes=sorted({quiz_id(q) for q in ids});qi={q:i for i,q in enumerate(quizzes)};ix=np.array([qi[quiz_id(q)] for q in ids]);dates=np.array([qs[q]['date'][:10] for q in ids])
 scores={name:np.array([[100*(token_f1(pred[m][q],qs[q]['gold_answer']) if name=='token_f1' else nearest_choice(pred[m][q],qs[q]['choices'])[0]==qs[q]['gold_answer_idx']) for m in models] for q in ids]) for name in ['token_f1','choice_accuracy']}
 rng=np.random.default_rng(a.seed);draw=rng.multinomial(len(quizzes),np.full(len(quizzes),1/len(quizzes)),size=a.B)
 def gap(mask1,mask2,score,weight):
  def side(mask):
   counts=np.bincount(ix[mask],minlength=len(quizzes)); sums=np.zeros((len(quizzes),len(models)));np.add.at(sums,ix[mask],score[mask]);valid=counts>0
   if not np.any(valid):return None,None
   vals=sums[valid];den=counts[valid]
   if weight=='equal_quiz':vals=vals/den[:,None];den=np.ones(len(vals))
   observed=vals.sum(axis=0)/den.sum()
   with np.errstate(invalid="ignore", divide="ignore"):
    bs=draw[:,valid]@vals/(draw[:,valid]@den)[:,None]
   return observed,bs
  p,pb=side(mask1);q,qb=side(mask2)
  return (None,None) if p is None or q is None else (p-q,pb-qb)
 out={'scope':'Retrospective cached-answer sensitivity, not selection-free confirmation','models':models,'n_questions':len(ids),'n_quizzes':len(quizzes),'bootstrap':{'B':a.B,'seed':a.seed,'unit':'weekly quiz','shared_draws':True},'question_ids':ids,'contrasts':{},'half_year_sensitivity':{}}
 contrasts={'boundary_2025':(dates<'2025-01-01',dates>='2025-01-01'),'years_2023_2026':(np.char.startswith(dates,'2023'),np.char.startswith(dates,'2026'))}
 x=np.log10([p for _,_,p in LADDER]);panels={'seven_rungs':7,'qwen_only':6}
 for key,(pre,post) in contrasts.items():
  c={'n_pre':int(pre.sum()),'n_post':int(post.sum()),'results':{}}
  for weight in ['question_pooled','equal_quiz']:
   c['results'][weight]={}
   for endpoint,score in scores.items():
    g,bs=gap(pre,post,score,weight);entry={'per_model':{m:summary(g[i],bs[:,i]) for i,m in enumerate(models)},'slopes':{}}
    for panel,n in panels.items():
     sl=slope(x[:n],g[:n]);v=summary(sl,slope(x[:n],bs[:,:n]));perms=np.array(list(itertools.permutations(g[:n])));v['p_model_permutation']=float(np.mean(np.abs(slope(x[:n],perms))>=abs(sl)-1e-12));v['models']=models[:n];entry['slopes'][panel]=v
    c['results'][weight][endpoint]=entry
  out['contrasts'][key]=c
 for weight in ['question_pooled','equal_quiz']:
  rec={};gs=[];draws=[];excluded=[]
  for bound in BOUNDARIES:
   pre=dates<bound;post=~pre;g,bs=gap(pre,post,scores['token_f1'],weight)
   if g is None:excluded.append({'boundary':bound,'reason':'empty pre or post slice','n_pre':int(pre.sum()),'n_post':int(post.sum())});continue
   rec[bound]={'n_pre':int(pre.sum()),'n_post':int(post.sum()),'per_model':{m:summary(g[i],bs[:,i]) for i,m in enumerate(models)}};gs.append(g);draws.append(bs)
  avg=np.mean(gs,axis=0);avgbs=np.mean(draws,axis=0)
  out['half_year_sensitivity'][weight]={'boundaries':rec,'excluded':excluded,'mean_across_valid_boundaries':{m:summary(avg[i],avgbs[:,i]) for i,m in enumerate(models)},'note':'Equal weights over valid prespecified candidate boundaries; common bootstrap draws preserve dependence. Excludes empty-slice boundaries. Not a prospective or selection-corrected estimate.'}
 # numpy scalar conversion for portability
 a.output.write_text(json.dumps(out,indent=2,default=lambda x:x.item())+'\n');print('Wrote',a.output)
if __name__=='__main__':main()
