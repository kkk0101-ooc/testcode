#!/usr/bin/env python3
"""CPU-only image-level supplemental analysis, including crossed seed bootstrap."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,gzip,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'reports/nns_vit_endpoint_supplement_v1_20260920'
SOURCES={};BOOT=2000

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p,expected=None):
 raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
 if expected:assert h==expected,str(p)
 SOURCES[str(p.relative_to(REPO))]=h
 return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
def write(name,d):(ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(name,rows):
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def med(v):
 a=np.asarray([x for x in v if x is not None],float);a=a[np.isfinite(a)]
 return float(np.median(a)) if a.size else None
def rng_for(*parts):return np.random.default_rng(int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big'))
def estimate(v,key):
 a=np.asarray([np.nan if x is None else x for x in v],float);a=a[np.isfinite(a)]
 if not len(a):return dict(n=0,na=len(v),mean=None,median=None,ci95=None)
 rng=rng_for('shared_image_bootstrap',len(a));draw=a[rng.integers(len(a),size=(BOOT,len(a)))].mean(1)
 return dict(n=len(a),na=len(v)-len(a),mean=float(a.mean()),median=float(np.median(a)),ci95=np.quantile(draw,[.025,.975]).tolist())
def corr(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);m=np.isfinite(a)&np.isfinite(b)
 if m.sum()<2:return None
 a=rankdata(a[m]);b=rankdata(b[m]);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
 return None if den==0 else float(a@b/den)
def phase(values,blocks):return med([values[b] for b in blocks])
def trend(values):
 a=phase(values,range(4));b=phase(values,range(8,12))
 return None if a is None or b is None else b-a

def m3_blocks(path,channels):
 d=np.load(path);SOURCES[str(path.relative_to(REPO))]=sha(path);out=[]
 for b in range(12):
  mask=(d['block']==b)&np.isin(d['channel'],channels[str(b)]);tokens=d['token'][mask];mass=d['patch_absolute_mass_196'][mask].astype(float);total=mass.sum(1);p=np.divide(mass,total[:,None],out=np.full_like(mass,np.nan),where=total[:,None]>0)
  with np.errstate(divide='ignore',invalid='ignore'):entropy=-np.where(p>0,p*np.log(p),0).sum(1)/np.log(196)
  entropy[total==0]=np.nan
  patch=tokens>0;coords=np.indices((14,14)).reshape(2,196).T;tc=np.stack(((tokens[patch]-1)//14,(tokens[patch]-1)%14),axis=1);dist=np.abs(tc[:,None,:]-coords[None,:,:]);local=(p[patch]*(dist.max(2)<=1)).sum(1);distance=(p[patch]*np.sqrt((dist**2).sum(2))/(13*np.sqrt(2))).sum(1)
  out.append(dict(block=b,entropy=med(entropy[patch]),locality=med(local),distance=med(distance),cancellation=med(d['cancellation'][mask][patch]),cls_minus_patch_entropy=med(entropy[~patch])-med(entropy[patch])))
 return out

def m4_blocks(d,jvp=None):
 att=d['attention'];res=d['residual'];jmap={(r['block'],r['head']):r for r in (jvp or att)};out=[]
 for b in sorted({r['block'] for r in att}):
  ar=[r for r in att if r['block']==b]
  row=dict(block=b,routing_scale=med([r['routing_share'] for r in ar]),value_error=med([r['frozen_attention_endpoint_error'] for r in ar]),jvp_discrepancy=med([jmap[b,r['head']]['frozen_attention_jvp_error'] for r in ar]),attention_path_cancellation=med([r['routing_value_path_cancellation'] for r in ar]),saturation_routing=corr([r['mean_query_saturation'] for r in ar],[r['routing_share'] for r in ar]),saturation_error=corr([r['mean_query_saturation'] for r in ar],[r['frozen_attention_endpoint_error'] for r in ar]),target_contracted_jvp=med([r.get('target_contracted_jvp_discrepancy') for r in ar]))
  for kind in ('attention','mlp'):
   rr=next(r for r in res if r['block']==b and r['residual_type']==kind)
   row.update({kind+'_'+name:rr[field] for name,field in [('identity_scale','identity_share'),('output_identity_tv','output_identity_patch_tv'),('residual_path_cancellation','path_cancellation')]})
  out.append(row)
 return out

def compare_training(blockrows):
 grouped=defaultdict(dict)
 for row in blockrows:
  ident=(row['model'],row['study'],row['state'],row['sample_id']);grouped[ident][row['block']]=row
 trends=[]
 for (model,study,state,sample),blocks in grouped.items():
  assert set(blocks)==set(range(12))
  for metric in next(iter(blocks.values())):
   if metric in ('model','study','state','sample_id','block','target_contracted_jvp'):continue
   values=[blocks[b][metric] for b in range(12)]
   trends.append(dict(model=model,study=study,state=state,sample_id=sample,metric=metric,early=phase(values,range(4)),late=phase(values,range(8,12)),late_minus_early=trend(values)))
 table('training_image_trends.csv',trends)
 comparisons=[];seeds=['202609201','202609202','202609203']
 for model,study,metric in sorted({(r['model'],r['study'],r['metric']) for r in trends}):
  subset=[r for r in trends if (r['model'],r['study'],r['metric'])==(model,study,metric)];lookup={(r['state'],r['sample_id']):r for r in subset};samples=sorted({r['sample_id'] for r in subset});assert len(samples)==32
  for quantity in ('early','late','late_minus_early'):
   trained=np.array([lookup['trained',s][quantity] for s in samples],float);random=np.array([[lookup[k,s][quantity] for s in samples] for k in seeds],float)
   valid=np.isfinite(trained)&np.isfinite(random).all(0);t=trained[valid];r=random[:,valid]
   if not len(t):
    comparisons.append(dict(model=model,study=study,metric=metric,quantity=quantity,n_images=0,n_seeds=3,trained_mean=None,random_mean=None,random_seed_means=[None]*3,trained_minus_random=None,ci95=None,state_summary={}))
    continue
   rng=rng_for(model,study,metric,quantity);boot=[]
   for _ in range(BOOT):
    ii=rng.integers(len(t),size=len(t));ss=rng.integers(3,size=3);boot.append(float((t[ii]-r[ss][:,ii].mean(0)).mean()))
   comparisons.append(dict(model=model,study=study,metric=metric,quantity=quantity,n_images=len(t),n_seeds=3,trained_mean=float(t.mean()),random_mean=float(r.mean()),random_seed_means=r.mean(1).tolist(),trained_minus_random=float((t-r.mean(0)).mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),state_summary={state:estimate([lookup[state,s][quantity] for s in samples],f'{model}|{study}|{metric}|{quantity}|{state}') for state in ['trained',*seeds]}))
 return comparisons

def feature_metrics(row):
 p=row['p'];out=[]
 if p is None:return []
 p=np.array(p)
 for control,key in [('paired','q'),('wrong_channel','q_wrong_channel'),('cross_layer','q_cross_layer'),('permuted','q_permuted'),('random_channel','q_random_channel')]:
  if row[key] is None:continue
  q=np.array(row[key]);m=(p+q)/2
  def kl(a):
   keep=a>0;return float(np.sum(a[keep]*np.log(a[keep]/m[keep])))
  out.append(dict(control=control,spearman=corr(p,q),cosine=float(p@q/(np.linalg.norm(p)*np.linalg.norm(q))),js_distance=float(np.sqrt(max(0.,.5*(kl(p)+kl(q))))),top20_overlap=len(set(np.lexsort((np.arange(196),-p))[:20])&set(np.lexsort((np.arange(196),-q))[:20]))/20))
 return out

def main():
 inherited=read(ROOT/'inherited.json');tasks=read(ROOT/'task_manifest.json')['tasks'];bindings=read(ROOT/'source_bindings.json');blockrows=[];feature_rows=[];target_rows=[];validations=[];random_bindings=defaultdict(set)
 for row in inherited:
  t=row['payload'];model=t['model_id'];identity=dict(model=model,sample_id=t['sample_id'],state='trained')
  if t['cohort_order']<32:
   d=read(REPO/row['m3_result'],bindings[row['m3_result']])['result'];p=REPO/d['metrics_path'];assert sha(p)==d['metrics_sha256']
   blockrows.extend(dict(**identity,study='m3',**r) for r in m3_blocks(p,t['channels']))
   d=read(REPO/row['m4_result'],bindings[row['m4_result']])['result'];blockrows.extend(dict(**identity,study='m4',**r) for r in m4_blocks(d))
 for task in tasks:
  p=ROOT/'results/attempts'/task['task_id']/'attempt-01.json';env=read(p);unsigned=dict(env);expected=unsigned.pop('canonical_digest');assert hashlib.sha256(json.dumps(unsigned,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==expected
  d=env['result'];t=task['payload'];assert d['image_sha256']==t['image_sha256'];identity=dict(model=d['model_id'],sample_id=d['sample_id'])
  for f in d['payload_files']:assert sha(REPO/f['path'])==f['sha256']
  if d['study']=='random':
   random_bindings[d['model_id'],d['init_seed']].add(d['binding']['tensor_digest'])
   blockrows.extend(dict(**identity,state=str(d['init_seed']),study='m3',**r) for r in m3_blocks(ROOT/'payloads'/task['task_id']/'m3.npz',t['channels']))
   blockrows.extend(dict(**identity,state=str(d['init_seed']),study='m4',**r) for r in m4_blocks(d['m4'],d['jvp']))
   validations.append(dict(task_id=task['task_id'],kind='random',scans=[d['m4']],jvp=d['jvp'],m3_checks=d['m3_validation']['checks']))
  elif d['study']=='feature':
   rows=read(ROOT/'payloads'/task['task_id']/'feature.json.gz');assert len(rows)==192
   for row in rows:
    for r in feature_metrics(row):feature_rows.append(dict(**identity,block=row['block'],channel=row['channel'],**r))
  else:
   oldpath=f"reports/nns_vit_endpoint_m4_r1_20260919/results/attempts/{d['model_id']}__{d['split_order']:03d}/attempt-01.json";old=read(REPO/oldpath,bindings[oldpath])['result']
   # New GT must agree with original trained GT at the frozen numerical tolerance.
   previous={(r['block'],r['head']):r for r in old['attention']}
   for r in d['targets']['gt']['attention']:
    for k in ('routing_share','frozen_attention_endpoint_error'):
     assert abs(r[k]-previous[r['block'],r['head']][k])<=2e-5+2e-4*abs(previous[r['block'],r['head']][k])
   if d['correct']:assert d['targets']['gt']==d['targets']['top1']
   for target,scan in d['targets'].items():
    target_rows.extend(dict(**identity,target=target,correct=d['correct'],gt=d['gt'],top1=d['top1'],runnerup=d['runnerup'],**r) for r in m4_blocks(scan,old['attention']))
   validations.append(dict(task_id=task['task_id'],kind='target',scans=list(d['targets'].values()),jvp=[],m3_checks=[]))
 for key,ds in random_bindings.items():assert len(ds)==1,key
 assert len(random_bindings)==6 and len({next(iter(ds)) for ds in random_bindings.values()})==6
 max_closure=0.;max_jvp=0.;count=0
 for item in validations:
  for scan in item['scans']:
   for r in scan['attention']:
    for key in ('qkv_native_closure','frozen_attention_identity'):
     assert r[key]['pass'];max_closure=max(max_closure,r[key]['max_abs']);count+=1
   for r in scan['residual']:assert r['closure']['pass'];count+=1
   for b in scan['block_validation']:
    for key in ('head_sum_attention_closure','qk_routing_closure','routing_value_attention_closure'):assert b[key]['pass'];count+=1
  for r in item['jvp']:
   assert r['explicit_jvp_relative_error'] is None or r['explicit_jvp_relative_error']<=1e-4
   max_jvp=max(max_jvp,r['explicit_jvp_relative_error'] or 0)
  for c in item['m3_checks']:assert c['batched_vjp_scalar_vjp_max_abs']<=2e-5+2e-4*c['scalar_endpoint_max_abs']
 # M3 and M4 have distinct columns; keep tables separate.
 for study in ('m3','m4'):table(f'training_{study}_block_values.csv',[r for r in blockrows if r['study']==study])
 training=compare_training(blockrows);write('training_analysis.json',dict(comparisons=training,numerical_checks=count,max_closure_abs=max_closure,max_explicit_jvp_relative=max_jvp,random_state_digests=[dict(model=k[0],seed=k[1],digest=next(iter(v))) for k,v in sorted(random_bindings.items())]))
 table('target_block_values.csv',target_rows);target_summary=[];target_contrasts=[]
 metrics=[k for k in target_rows[0] if k not in ('model','sample_id','target','correct','gt','top1','runnerup','block')]
 for model in ('vit_b16','deit_s16'):
  for group in ('all','correct','incorrect'):
   subset=[r for r in target_rows if r['model']==model and (group=='all' or r['correct']==(group=='correct'))];samples=sorted({r['sample_id'] for r in subset});lookup={(r['sample_id'],r['target'],r['block']):r for r in subset}
   for metric in metrics:
    for quantity in ('b0','b5','b11','late_minus_early'):
     vectors={}
     for target in ('gt','top1','margin'):
      vals=[]
      for sample in samples:
       if quantity.startswith('b'):v=lookup[sample,target,int(quantity[1:])][metric]
       else:
        a=lookup[sample,target,0][metric];b=lookup[sample,target,11][metric];v=None if a is None or b is None else b-a
       vals.append(v)
      vectors[target]=vals;target_summary.append(dict(model=model,subset=group,target=target,metric=metric,quantity=quantity,**estimate(vals,f'{model}|{group}|{target}|{metric}|{quantity}')))
     for target in ('top1','margin'):
      delta=[None if a is None or b is None else a-b for a,b in zip(vectors[target],vectors['gt'])]
      target_contrasts.append(dict(model=model,subset=group,comparison=target+'-gt',metric=metric,quantity=quantity,**estimate(delta,f'delta|{model}|{group}|{target}|{metric}|{quantity}')))
 write('target_analysis.json',dict(summary=target_summary,contrasts=target_contrasts,cohort=[dict(model=m,n=100,correct=sum(r['model']==m and r['target']=='gt' and r['block']==0 and r['correct'] for r in target_rows)) for m in ('vit_b16','deit_s16')],native_jvp_scope='Target-independent AV JVP inherited; target_contracted_jvp is target dependent'))
 table('feature_unit_values.csv',feature_rows);feature_images=[];grouped=defaultdict(list)
 for r in feature_rows:grouped[r['model'],r['sample_id'],r['control'],r['block']].append(r)
 for (model,sample,control,b),rs in sorted(grouped.items()):feature_images.append(dict(model=model,sample_id=sample,control=control,block=b,**{k:med([r[k] for r in rs]) for k in ('spearman','cosine','js_distance','top20_overlap')}))
 table('feature_block_image_values.csv',feature_images);feature_summary=[];feature_contrasts=[]
 for model in ('vit_b16','deit_s16'):
  for phase_name,bs in [('all',range(12)),('early',range(4)),('late',range(8,12))]:
   subset=[r for r in feature_images if r['model']==model and r['block'] in bs];samples=sorted({r['sample_id'] for r in subset});lookup={(r['sample_id'],r['control'],r['block']):r for r in subset}
   for metric in ('spearman','cosine','js_distance','top20_overlap'):
    vectors={}
    for control in ('paired','wrong_channel','cross_layer','permuted','random_channel'):
     values=[med([lookup[sample,control,b][metric] for b in bs]) for sample in samples];vectors[control]=values
     feature_summary.append(dict(model=model,phase=phase_name,control=control,metric=metric,**estimate(values,f'{model}|{phase_name}|{control}|{metric}')))
    for control in ('wrong_channel','cross_layer','permuted','random_channel'):
     values=[None if a is None or b is None else (b-a if metric=='js_distance' else a-b) for a,b in zip(vectors['paired'],vectors[control])]
     feature_contrasts.append(dict(model=model,phase=phase_name,control=control,metric=metric,positive_means='paired more similar',**estimate(values,f'contrast|{model}|{phase_name}|{control}|{metric}')))
 write('feature_analysis.json',dict(summary=feature_summary,contrasts=feature_contrasts,unit_count=len(feature_rows)//5,controls=5,zero_undefined_count=sum(r['spearman'] is None for r in feature_rows)))
 write('analysis_sources.json',SOURCES);print(json.dumps(dict(status='COMPLETE',training=len(training),target_rows=len(target_rows),feature_units=len(feature_rows)//5,checks=count)))
if __name__=='__main__':main()
