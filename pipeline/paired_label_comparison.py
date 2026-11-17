"""Compare cached support labels on shared human rows with paired quiz resampling.

No model inference or annotation is performed. Each contrast uses the same
quiz draws for both labelers and strict YES human labels on legacy excerpts.
"""
from pathlib import Path
import sys,json,random,hashlib,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline/analysis'))
import e0_human_alignment as E
import r2_human_alignment as R
from r8_judge_validation import judge_map

def kappa(conf):
    tp,fp,fn,tn=np.moveaxis(np.asarray(conf,dtype=float),-1,0)
    n=tp+fp+fn+tn
    po=(tp+tn)/n
    pe=((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/(n*n)
    return (po-pe)/(1-pe)

def compare(rows,nli,judge,B=5000,seed=20260803):
    names=['string600','nli600','judge600'];clusters={}
    for r in rows:
        key=(r['qid'],r['doc_id']);assert key in nli and (*key,'w600') in judge
        pred=[r['string_w600'],nli[key],judge[*key,'w600']['label']=='YES']
        conf=clusters.setdefault(r['quiz'],np.zeros((3,4),dtype=int))
        for i,a in enumerate(pred):
            h=r['human_bin'];conf[i,0 if a and h else 1 if a else 2 if h else 3]+=1
    matrix=np.stack(list(clusters.values()));obs=kappa(matrix.sum(axis=0));rng=random.Random(seed)
    draws=np.array([kappa(matrix[[rng.randrange(len(matrix)) for _ in matrix]].sum(axis=0)) for _ in range(B)])
    def summary(x,point):
        a=np.sort(x[np.isfinite(x)]);n=len(a)
        return {'estimate':float(point),'ci95':[float(a[int(.025*n)]),float(a[int(.975*n)])],'valid_draws':n}
    out={'n_rows':len(rows),'n_questions':len({r['qid'] for r in rows}),'n_quizzes':len(matrix),'seed':seed,'B':B,'labels':{name:summary(draws[:,i],obs[i]) for i,name in enumerate(names)},'paired_differences':{}}
    for i,j in [(1,0),(2,0),(1,2)]:
        key=names[i]+'_minus_'+names[j];d=draws[:,i]-draws[:,j];v=summary(d,obs[i]-obs[j]);v['two_sided_p_plus_one']=min(1.,2*min((1+int(np.sum(d<=1e-12)))/(B+1),(1+int(np.sum(d>=-1e-12)))/(B+1)));out['paired_differences'][key]=v
    ordered=sorted(out['paired_differences'],key=lambda x:out['paired_differences'][x]['two_sided_p_plus_one']);running=0.
    for i,key in enumerate(ordered):
        v=out['paired_differences'][key];running=max(running,min(1.,(len(ordered)-i)*v['two_sided_p_plus_one']));v['holm_p_three_comparisons_within_pack']=running
    return out

def main():
    qs=E.load_questions();corpus=E.load_corpus();J=judge_map();mhs=E.load_mhs()
    out={'basis':'Strict YES, unit passage-row weights, legacy 700-character human excerpts versus cached 600-character automatic labels. Three paired contrasts form a Holm family within each pack. Quiz-cluster bootstrap. Retrospective sensitivity, not exact-input validation.','task_b':compare(E.packb_rows(qs,corpus),E.nli_map(R.E0_NLI600),J,seed=R.SEED_B),'masked_pack':compare(E.masked_rows(qs,corpus,mhs),R.server_labels('undated600'),J,seed=R.SEED_M)}
    paths=[ROOT/'data/processed/questions.jsonl',ROOT/'data/processed/corpus.jsonl',R.E0_NLI600,R.SRV/'undated600.jsonl',ROOT/'results/analysis/r8_judge_labels.jsonl']
    for directory in [E.PACK_B,E.PACK_M,ROOT/'annotation/adjudication/pack']:
        paths.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in {'.csv','.jsonl'})
    out['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    p=ROOT/'results/paired_label_comparison.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v['paired_differences'] for k,v in out.items() if k in ('task_b','masked_pack')},indent=2))
if __name__=='__main__':main()
