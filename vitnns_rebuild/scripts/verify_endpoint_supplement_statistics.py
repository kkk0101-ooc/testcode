#!/usr/bin/env python3
"""Independent checks of reported estimates, source M2 inference and paper rows."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,rankdata
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'reports/nns_vit_endpoint_supplement_v1_20260920';PAPER=REPO/'papers/nns_vit_endpoint/publication_r2_20260920';sys.path.insert(0,str(REPO/'src'))
from vitnns.endpoint_m2_regions import METHODS,BLOCKS
checks=[]
def ck(name,value):
 checks.append(dict(name=name,passed=bool(value)))
 if not value:raise ValueError(name)
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):return list(csv.DictReader((ROOT/name).open()))
# Recompute all 672 original M2 sign tests, preserving global Holm order.
root=REPO/'reports/nns_vit_endpoint_m2_r1_20260919';manifest={r['path']:r['sha256'] for r in load(root/'artifact_manifest.json')['files']}
for name in ('image_values.json.gz','auc_tests.json'):ck('M2 source '+name,sha(root/name)==manifest[name])
images=json.loads(gzip.decompress((root/'image_values.json.gz').read_bytes()));tests=load(root/'auc_tests.json')['tests'];lookup={tuple(r['key']):r['images'] for r in images};recomputed=[]
for r in tests:
 key=(r['model'],r['block'],r['geometry'],r['reference'],r['outcome'],r['metric']);values=np.array(list(lookup[key].values()));delta=values[:,0]-values[:,METHODS.index(r['baseline'])]
 if r['direction']=='baseline_greater':delta=-delta
 positive=int((delta>1e-12).sum());negative=int((delta< -1e-12).sum());p=1. if positive+negative==0 else float(binomtest(positive,positive+negative,.5,alternative='greater').pvalue)
 ck('M2 signs '+r['test_id'],positive==r['positive'] and negative==r['negative'] and abs(p-r['p'])<1e-12);recomputed.append((p,r['test_id']))
corrected={};previous=0.
for rank,(p,tid) in enumerate(sorted(recomputed)):
 previous=max(previous,min(1.,p*(len(tests)-rank)));corrected[tid]=previous
for r in tests:ck('M2 Holm '+r['test_id'],abs(corrected[r['test_id']]-r['holm_p'])<1e-12 and (corrected[r['test_id']]<.05)==r['reject'])
# Point estimates must reconstruct from exported image rows, independently of summary routines.
training=load(ROOT/'training_analysis.json')['comparisons'];tv=rows('training_image_trends.csv')
for r in training:
 subset=[v for v in tv if (v['model'],v['study'],v['metric'])==(r['model'],r['study'],r['metric'])];means={state:np.mean([float(v[r['quantity']]) for v in subset if v['state']==state]) for state in ('trained','202609201','202609202','202609203')}
 ck('training point '+str((r['model'],r['metric'],r['quantity'])),np.isclose(means['trained'],r['trained_mean'],atol=1e-12) and np.isclose(means['trained']-np.mean([means[s] for s in ('202609201','202609202','202609203')]),r['trained_minus_random'],atol=1e-12))
# Source p/q -> independent rank coefficient -> exported unit statistic for every feature target.
unit={(r['model'],r['sample_id'],int(r['block']),int(r['channel']),r['control']):r for r in rows('feature_unit_values.csv')};inherited={r['task_id']:r['payload'] for r in load(ROOT/'inherited.json')}
for p in sorted((ROOT/'payloads').glob('feature__*/feature.json.gz')):
 tid=p.parent.name.removeprefix('feature__');task=inherited[tid];records=json.loads(gzip.decompress(p.read_bytes()))
 for r in records:
  for control,key in [('paired','q'),('wrong_channel','q_wrong_channel'),('cross_layer','q_cross_layer'),('permuted','q_permuted'),('random_channel','q_random_channel')]:
   a=np.array(r['p']);b=np.array(r[key]);rho=float(np.corrcoef(rankdata(a),rankdata(b))[0,1]);cos=float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)));stored=unit[task['model_id'],task['sample_id'],r['block'],r['channel'],control]
   ck('feature '+str((tid,r['block'],r['channel'],control)),np.isclose(rho,float(stored['spearman']),atol=1e-12) and np.isclose(cos,float(stored['cosine']),atol=1e-12))
# Current main table columns must be corrected, and supplement rows bind all summaries.
base=load(ROOT/'baseline_analysis.json')['summary'];labels={'vit_b16':'ViT-B/16','deit_s16':'DeiT-S/16'};refs={'normalized_zero':'normalized-zero','blur_rgb':'blur'}
for lang in ('zh','en'):
 text=(PAPER/lang/'manuscript.md').read_text();sections={}
 for n in (5,6,7):sections[n]=text.split(f'**Table {n}.')[1].split('**Table ')[0]
 for r in base:
  prefix=f"| {labels[r['model']]} | {refs[r['reference']]} |"
  for n,column,key in [(5,5,'endpoint_minus_random'),(6,4,'random_corrected'),(7,4,'random_regret')]:
   line=next(l for l in sections[n].splitlines() if l.startswith(prefix));cells=[x.strip() for x in line.strip('|').split('|')];ck(f'paper {lang} {n} {prefix}',cells[column]==f"{r[key]['median']:.4f}")
 sup=(PAPER/lang/'supplement_experiments_v1.md').read_text()
 for r in training:
  if r['quantity']=='late_minus_early':ck('training table '+lang+str((r['model'],r['metric'])),f"| {labels[r['model']]} | {r['study']} | {r['metric']} | {r['trained_mean']:.4f} | {r['random_mean']:.4f} |" in sup)
 for r in load(ROOT/'target_analysis.json')['summary']:
  if r['quantity']=='late_minus_early':ck('target table '+lang+str((r['model'],r['subset'],r['target'],r['metric'])),f"| {labels[r['model']]} | {r['subset']} | {r['target']} | {r['metric']} | {r['n']} | {r['mean']:.4f} |" in sup)
result=dict(status='PASS',check_count=len(checks),M2_tests_recomputed=672,feature_pairs_recomputed=61440,checks=checks)
(ROOT/'independent_statistics_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
