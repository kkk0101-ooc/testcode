#!/usr/bin/env python3
"""CPU-only image/class-level v5 inference and strictly held-out ridge evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import rel_entr
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[1];ROOT=R/'reports/nns_vit_revision_v5_r1_20260921';M=R/'manifests/nns_vit_revision_v5';O=ROOT/'analysis';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'src'))
from vitnns.endpoint_m1_statistics import bootstrap_median
SOURCES={};B=[0,3,7,11]
protocol=json.loads((M/'protocol.json').read_text());models={m['model_id']:m for m in protocol['models']};items={r['sample_id']:r for r in json.loads((M/'image_bindings.json').read_text())['rows']}
def seed(*parts):return int.from_bytes(hashlib.sha256(('NNS_VIT_REVISION_V5_20260921_R1|'+'|'.join(map(str,parts))).encode()).digest()[:8],'big')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources(kind):
 rows=[]
 for p in sorted((ROOT/kind).glob('*.json')):
  r=json.loads(p.read_text());assert r['status']=='VALID';q=R/r['output'];assert sha(q)==r['output_sha256'];assert r['image_sha256']==items[r['image']]['image_sha256'];SOURCES[str(p.relative_to(R))]=sha(p);SOURCES[str(q.relative_to(R))]=sha(q);rows.append(r)
 return rows
def load(r):
 with np.load(R/r['output'],allow_pickle=False) as z:return {k:z[k] for k in z.files}
def dump(name,rows):
 assert rows,name
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def finite_median(v):
 a=np.asarray(v);a=a[np.isfinite(a)];return float(np.median(a)) if len(a) else np.nan
def summarize(v,key):
 a=np.asarray(v,float);n=len(a);a=a[np.isfinite(a)]
 if not len(a):return dict(n_total=n,n_valid=0,n_missing=n,median=np.nan,ci95_low=np.nan,ci95_high=np.nan,p=1.,positive=0,negative=0,ties=0)
 ci=bootstrap_median(a,'V5|'+str(key));pos=int((a>1e-12).sum());neg=int((a< -1e-12).sum());N=pos+neg;p=min(1.,2*math.fsum(math.comb(N,j) for j in range(min(pos,neg)+1))/2**N) if N else 1.
 return dict(n_total=n,n_valid=len(a),n_missing=n-len(a),median=float(np.median(a)),ci95_low=ci[0],ci95_high=ci[1],p=p,positive=pos,negative=neg,ties=len(a)-N)
def holm(rows,key):
 groups=defaultdict(list)
 for r in rows:groups[key(r)].append(r)
 for rs in groups.values():
  previous=0
  for i,r in enumerate(sorted(rs,key=lambda r:r['p'])):
   previous=max(previous,min(1.,(len(rs)-i)*r['p']));r['holm_p']=previous;r['decision']='SUPPORTED_DIRECTION' if previous<=.05 and r['median']>0 else 'OPPOSITE_DIRECTION' if previous<=.05 and r['median']<0 else 'NOT_SUPPORTED'
def rho(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if len(a)<4 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(a)==0 or np.ptp(b)==0:return np.nan
 return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def normalized(a):
 a=np.asarray(a,float);den=a.sum(-1,keepdims=True)
 return np.divide(a,den,out=np.full_like(a,np.nan),where=den!=0)
def js(p,q):
 m=(p+q)/2
 with np.errstate(invalid='ignore',divide='ignore'):
  return .5*np.sum(rel_entr(p,m)+rel_entr(q,m),axis=-1)/math.log(2)
def grouped_summary(rows,keys,value='value'):
 groups=defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in keys)].append(r[value])
 return [dict(zip(keys,k),**summarize(v,k)) for k,v in sorted(groups.items())]

def e1():
 records=sources('e1');assert len(records)==1000;rows=[];activation=[];pairs=[]
 for model in ('deit_s16','vit_b16'):
  rs=[r for r in records if r['model_id']==model];data=[load(r) for r in rs];classes=np.array([r['class_id'] for r in rs]);ids=[r['image'] for r in rs]
  # Recover raw RGB means from model-normalized descriptor. Matching is independent of neuron outputs.
  mean=np.repeat(models[model]['mean'],64);std=np.repeat(models[model]['std'],64);desc=np.asarray([z['rgb_descriptor'] for z in data])*std+mean
  distances=((desc[:,None]-desc[None,:])**2).mean(-1);distances[classes[:,None]==classes[None,:]]=np.inf
  match=distances.argmin(1);random=[min(np.where(classes!=c)[0],key=lambda j:seed('random_pair',ids[i],ids[j])) for i,c in enumerate(classes)]
  for i in range(500):pairs.append(dict(model=model,image=ids[i],matched_image=ids[match[i]],random_image=ids[random[i]],descriptor_mse=distances[i,match[i]]))
  for b in B:
   for kind in ('post','mlp'):
    prefix=f'b{b:02d}_{kind}_';p=normalized(np.asarray([z[prefix+'patch_mass'] for z in data]));act=np.asarray([z[prefix+'activation'] for z in data]);overall=act.mean(0);den=((act-overall)**2).sum(0);between=sum(5*(act[classes==c].mean(0)-overall)**2 for c in set(classes));eta=np.divide(between,den,out=np.full(32,np.nan),where=den!=0)
    for u in range(32):activation.append(dict(model=model,block=b,kind=kind,unit=u,activation_eta_squared=eta[u],mean=float(overall[u]),sd=float(act[:,u].std()),token=int(data[0][prefix+'units'][u,0]),channel=int(data[0][prefix+'units'][u,1])))
    for i in range(500):
     same=np.where((classes==classes[i])&(np.arange(500)!=i))[0]
     s=float(np.nanmean(1-js(p[same],p[i])));m=float(np.nanmean(1-js(p[match[i]],p[i])));r=float(np.nanmean(1-js(p[random[i]],p[i])))
     rows.append(dict(model=model,image=ids[i],class_id=int(classes[i]),block=b,kind=kind,same_similarity=s,matched_similarity=m,random_similarity=r,same_minus_matched=s-m,same_minus_random=s-r,n_units_valid=int(np.isfinite(p[i]).all(1).sum()),mass=finite_median(data[i][prefix+'absolute_mass']),signed_sum=finite_median(data[i][prefix+'signed_sum']),entropy=finite_median(data[i][prefix+'entropy'])))
 dump('e1_pairs.csv',pairs);dump('e1_image_values.csv',rows);dump('e1_activation_selectivity.csv',activation)
 classes_out=[];by=defaultdict(list)
 for r in rows:by[r['model'],r['class_id'],r['block'],r['kind']].append(r)
 for k,rr in sorted(by.items()):
  for metric in ('same_similarity','matched_similarity','random_similarity','same_minus_matched','same_minus_random','mass','signed_sum','entropy'):
   classes_out.append(dict(model=k[0],class_id=k[1],block=k[2],kind=k[3],metric=metric,value=float(np.mean([r[metric] for r in rr]))))
 dump('e1_class_values.csv',classes_out)
 summary=grouped_summary(classes_out,['model','block','kind','metric']);holm(summary,lambda r:(r['model'],r['metric']));dump('e1_summary.csv',summary)

def e2():
 records=sources('e2');assert len(records)==200;rows=[]
 for r in records:
  z=load(r)
  for b in range(12):
   for kind in ('mlp_own','mlp_scalar','mlp_family','heads'):
    pre=f'b{b:02d}_{kind}_';mass=z[pre+'absolute_mass'];p=normalized(z[pre+'patch_mass']);i,j=np.triu_indices(len(p),1)
    vals={m:finite_median(z[pre+m]) for m in ('absolute_mass','signed_sum','entropy','sparsity','cancellation')}
    vals['redundancy_JS_similarity']=finite_median(1-js(p[i],p[j]));vals['top8_sampled_mass_share']=float(np.sort(mass)[-8:].sum()/mass.sum()) if mass.sum()!=0 else np.nan
    for metric,value in vals.items():rows.append(dict(model=r['model_id'],image=r['image'],block=b,kind=kind,metric=metric,value=value))
 dump('e2_image_values.csv',rows);dump('e2_summary.csv',grouped_summary(rows,['model','block','kind','metric']))


def e3():
 records=sources('e3');assert len(records)==200;single=[];top=[];pairs=[];outnames=['gt','high_non_target','random_non_target','dynamic_margin'];methods=['endpoint','activation','gradient','activation_gradient','displacement','taylor']
 for r in records:
  z=load(r)
  for b in B:
   for kind in ('post','mlp','family'):
    for ref in ('normalized_zero','blur_rgb'):
     prefix=f'b{b:02d}_{kind}_{ref}_';scores=z[prefix+'scores'][0];effect=z[prefix+'single_effect'];idx=z[prefix+'topk_indices'];tk=z[prefix+'topk_effect'];tks=np.concatenate([tk[:6],tk[6:26].mean(0,keepdims=True),tk[26:]],0);tka=np.concatenate([abs(tk[:6]),abs(tk[6:26]).mean(0,keepdims=True),abs(tk[26:])],0);names=methods+['random','activation_matched','high_non_target_endpoint','random_non_target_endpoint']
     for ai,alpha in enumerate((.1,1.)):
      for oi,outcome in enumerate(outnames):
       y=abs(effect[ai,:,oi]);base_metrics={}
       for mi,name in enumerate(methods):
        chosen=int(np.argsort(-scores[mi],kind='stable')[0]);v={'spearman':rho(scores[mi],y),'capture_at_1':float(y[chosen]/y.max()) if y.max()!=0 else np.nan,'regret_at_1':float(y.max()-y[chosen]),'selected_signed_effect':float(effect[ai,chosen,oi])};base_metrics[name]=v
        for metric,value in v.items():single.append(dict(model=r['model_id'],image=r['image'],block=b,kind=kind,reference=ref,alpha=alpha,outcome=outcome,method=name,metric=metric,value=value))
       # Uniform ranking capture/regret expectations and mean random rank correlation (zero) are analytic.
       for metric,value in dict(spearman=0. if np.ptp(y)>0 else np.nan,capture_at_1=float(y.mean()/y.max()) if y.max()!=0 else np.nan,regret_at_1=float(y.max()-y.mean()),selected_signed_effect=float(effect[ai,:,oi].mean())).items():single.append(dict(model=r['model_id'],image=r['image'],block=b,kind=kind,reference=ref,alpha=alpha,outcome=outcome,method='random',metric=metric,value=value))
       for metric in ('spearman','capture_at_1','regret_at_1'):pairs.append(dict(model=r['model_id'],image=r['image'],block=b,kind=kind,reference=ref,alpha=alpha,budget=1,outcome=outcome,metric=metric,baseline='activation_gradient',value=base_metrics['endpoint'][metric]-base_metrics['activation_gradient'][metric]))
     for oi,outcome in enumerate(outnames):
      for mi,name in enumerate(names):
       for bi,budget in enumerate((1,2,4,8)):
        for metric,value in [('signed_effect',tks[mi,bi,oi]),('absolute_effect',tka[mi,bi,oi])]:top.append(dict(model=r['model_id'],image=r['image'],block=b,kind=kind,reference=ref,budget=budget,outcome=outcome,method=name,metric=metric,value=float(value)))
      for baseline in ('activation_gradient','random','activation_matched','high_non_target_endpoint','random_non_target_endpoint'):
       mi=names.index(baseline)
       for bi,budget in enumerate((1,2,4,8)):
        for metric,arr in [('topk_signed',tks),('topk_absolute',tka)]:pairs.append(dict(model=r['model_id'],image=r['image'],block=b,kind=kind,reference=ref,alpha=1.,budget=budget,outcome=outcome,metric=metric,baseline=baseline,value=float(arr[0,bi,oi]-arr[mi,bi,oi])))
 dump('e3_single_image_values.csv',single);dump('e3_topk_image_values.csv',top);dump('e3_paired_image_values.csv',pairs)
 dump('e3_single_summary.csv',grouped_summary(single,['model','block','kind','reference','alpha','outcome','method','metric']))
 dump('e3_topk_summary.csv',grouped_summary(top,['model','block','kind','reference','budget','outcome','method','metric']))
 summary=grouped_summary(pairs,['model','block','kind','reference','alpha','budget','outcome','metric','baseline']);holm(summary,lambda r:(r['model'],r['kind'],r['outcome'],r['metric'],r['baseline']));dump('e3_paired_summary.csv',summary)


def e5():
 records=sources('e5');assert len(records)==1600;rows=[];classification=[]
 for r in records:
  z=load(r);vals=defaultdict(dict)
  for b in B:
   get=lambda k:float(z[f'b{b:02d}_{k}_absolute_mass'][0])
   vals['entropy'][b]=finite_median(z[f'b{b:02d}_post_entropy']);a=get('routing');v=get('V');vals['routing'][b]=a/(a+v) if a+v else np.nan
   a=get('carry');v=get('mlp_branch');vals['mlp_identity'][b]=a/(a+v) if a+v else np.nan
   a=get('identity');v=get('attention');vals['attention_identity'][b]=a/(a+v) if a+v else np.nan
   y=abs(z[f'b{b:02d}_normalized_zero_head_effect']);vals['head_increment'][b]=rho(z[f'b{b:02d}_heads_absolute_mass'],y)-rho(z[f'b{b:02d}_head_activation_gradient'],y)
  values={'H1':vals['entropy'][11]-vals['entropy'][0],'H2':vals['routing'][11]-vals['routing'][0],'H3':(vals['mlp_identity'][11]-vals['mlp_identity'][0])-(vals['attention_identity'][11]-vals['attention_identity'][0]),'H4':float(np.mean(list(vals['head_increment'].values())))}
  for key,value in values.items():rows.append(dict(model=r['model_id'],image=r['image'],hypothesis=key,value=value))
  for metric,d in vals.items():
   for b,value in d.items():classification.append(dict(model=r['model_id'],image=r['image'],class_id=r['class_id'],correct=int(z['logits'].argmax()==r['class_id']),block=b,metric=metric,value=value))
 dump('e5_image_values.csv',rows);dump('e5_profiles.csv',classification);summary=grouped_summary(rows,['model','hypothesis']);holm(summary,lambda r:r['model']);dump('e5_summary.csv',summary)
 e4(records)

def e4(records):
 predictions=[];summaries=[];paired=[]
 for model in models:
  rs=[r for r in records if r['model_id']==model];zs=[load(r) for r in rs];total=np.asarray([normalized(z['total_patch_mass'])[0] for z in zs]);edges=['Q','K','V','identity','attention','carry','mlp_branch']
  parts=np.asarray([[np.nan_to_num(normalized(z[f'b{b:02d}_{k}_patch_mass'])[0]) for b in B for k in edges] for z in zs]).reshape(400,4,7,196)
  shuffled=parts.copy()
  for i,r in enumerate(rs):
   for bi,b in enumerate(B):shuffled[i,bi]=parts[i,bi,np.random.default_rng(seed('e4_labels',r['image'],b)).permutation(7)]
  xs={'total':np.nan_to_num(total),'structure':np.concatenate((np.nan_to_num(total),parts.reshape(400,-1)),1),'shuffled':np.concatenate((np.nan_to_num(total),shuffled.reshape(400,-1)),1)}
  y=np.asarray([np.mean([np.mean(abs(z[f'b{b:02d}_normalized_zero_head_effect'])) for b in B]) for z in zs]);split=np.asarray([items[r['image']]['prediction_split'] for r in rs]);train=split=='train';valid=split=='validation';test=split=='test';estimates={}
  for name,x in xs.items():
   mu=x[train].mean(0);sd=x[train].std(0);sd[sd==0]=1;x=(x-mu)/sd;center=y[train].mean();best=None
   for penalty in (.1,1.,10.,100.):
    dual=np.linalg.solve(x[train]@x[train].T+penalty*np.eye(train.sum()),y[train]-center);coeff=x[train].T@dual;v=x@coeff+center;loss=float(abs(v[valid]-y[valid]).mean())
    if best is None or loss<best[0]:best=(loss,penalty,v,coeff)
   loss,penalty,v,coeff=best;estimates[name]=v
   np.savez_compressed(O/f'e4_fit_{model}_{name}.npz',mean=mu,sd=sd,coeff=coeff,intercept=center,penalty=penalty)
   summaries.append(dict(model=model,features=name,n_train=int(train.sum()),n_validation=int(valid.sum()),n_test=int(test.sum()),penalty=penalty,validation_mae=loss,test_mae=float(abs(v[test]-y[test]).mean()),test_spearman=rho(v[test],y[test])))
   for i in np.where(test)[0]:predictions.append(dict(model=model,image=rs[i]['image'],method=name,target=y[i],prediction=v[i],absolute_error=abs(y[i]-v[i])))
  for baseline in ('total','shuffled'):
   value=abs(y[test]-estimates[baseline][test])-abs(y[test]-estimates['structure'][test]);row=dict(model=model,baseline=baseline,**summarize(value,('e4',model,baseline)));paired.append(row)
  indices=np.where(test)[0];i,j=np.triu_indices(len(indices),1);i,j=indices[i],indices[j];d=js(total[i],total[j]);threshold=np.quantile(d[np.isfinite(d)],.1);keep=np.isfinite(d)&(d<=threshold);pairs=[]
  for a,b,distance in zip(i[keep],j[keep],d[keep]):pairs.append(dict(model=model,image_a=rs[a]['image'],image_b=rs[b]['image'],total_js=distance,selection_threshold=threshold,structural_js=float(np.mean(js(parts[a],parts[b]))),effect_difference=float(abs(y[a]-y[b]))))
  dump(f'e4_similar_total_pairs_{model}.csv',pairs)
 dump('e4_predictions.csv',predictions);dump('e4_models.csv',summaries);holm(paired,lambda r:r['model']);dump('e4_paired.csv',paired)

def main():
 a=argparse.ArgumentParser();a.add_argument('kind',choices=['e1','e2','e3','e5']);args=a.parse_args();globals()[args.kind]()
 receipt=dict(status='ANALYZED',sources=SOURCES,script_sha256=sha(Path(__file__)),estimands_sha256=sha(M/'analysis_estimands.json'),outputs={p.name:sha(p) for p in O.glob(args.kind+'*.csv')})
 (O/f'{args.kind}_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if args.kind=='e5':
  e4_outputs={p.name:sha(p) for p in O.glob('e4*') if p.name!='e4_receipt.json'}
  (O/'e4_receipt.json').write_text(json.dumps(dict(status='ANALYZED_HELD_OUT',sources=SOURCES,script_sha256=sha(Path(__file__)),estimands_sha256=sha(M/'analysis_estimands.json'),outputs=e4_outputs),indent=2)+'\n')
 print(args.kind,'ANALYZED',len(SOURCES))
if __name__=='__main__':main()
