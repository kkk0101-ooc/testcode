#!/usr/bin/env python3
"""Freeze supplemental scope and source bindings before any new inference."""
import hashlib,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from vitnns.endpoint_ep1_queue import canonical,digest,create_queue
ROOT=REPO/'reports/nns_vit_endpoint_supplement_v1_20260920'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ROOT.mkdir(exist_ok=True)
 if (ROOT/'freeze.json').exists():raise RuntimeError('ALREADY_FROZEN')
 sources={}; manifests={}
 for stage in ('e5_spatial_r1_20260918','m2_r1_20260919','m3_r1_20260919','m4_r1_20260919'):
  root=REPO/('reports/nns_vit_endpoint_'+stage)
  manifest=json.loads((root/'artifact_manifest.json').read_text());terminal=json.loads((root/'terminal_receipt.json').read_text())
  assert 'PASS' in terminal['status']
  if 'artifact_manifest_sha256' in terminal:assert sha(root/'artifact_manifest.json')==terminal['artifact_manifest_sha256']
  for name in ('artifact_manifest.json','terminal_receipt.json'):
   p=root/name;sources[str(p.relative_to(REPO))]=sha(p)
  manifests[stage]={r['path']:r['sha256'] for r in manifest['files']}
 def bind(stage,relative):
  p=REPO/('reports/nns_vit_endpoint_'+stage)/relative
  h=sha(p);assert h==manifests[stage][relative],str(p)
  sources[str(p.relative_to(REPO))]=h
  return json.loads(p.read_text()) if p.suffix=='.json' else None
 m3=bind('m3_r1_20260919','task_manifest.json')['tasks']
 seeds=[202609201,202609202,202609203];tasks=[];inherited=[]
 for row in m3:
  p=dict(row['payload']); model=p['model_id'];order=p['cohort_order'];base=row['task_id']
  p['channels']={b:sorted(sorted(cs,key=lambda c:hashlib.sha256(f'ENDPOINT_SUP_V1_CHANNEL|{model}|{b}|{c}'.encode()).hexdigest())[:16]) for b,cs in p['channels'].items()}
  p['gradient_chunk_size']=16
  result=bind('m3_r1_20260919',f'results/attempts/{base}/attempt-01.json')['result']
  if order<32:bind('m3_r1_20260919',str(Path(result['metrics_path']).relative_to('reports/nns_vit_endpoint_m3_r1_20260919')))
  m4id=f"{model}__{p['split_order']:03d}"
  bind('m4_r1_20260919',f'results/attempts/{m4id}/attempt-01.json')
  inherited.append(dict(task_id=base,payload=p,m3_result=f'reports/nns_vit_endpoint_m3_r1_20260919/results/attempts/{base}/attempt-01.json',m4_result=f'reports/nns_vit_endpoint_m4_r1_20260919/results/attempts/{m4id}/attempt-01.json'))
  tasks.append(dict(task_id=f'target__{base}',payload={**p,'study':'target','model_state':'trained','init_seed':None}))
  if order<32:
   tasks.append(dict(task_id=f'feature__{base}',payload={**p,'study':'feature','model_state':'trained','init_seed':None}))
   for seed in seeds:tasks.append(dict(task_id=f'random{seed}__{base}',payload={**p,'study':'random','model_state':'random','init_seed':seed}))
 for stage in ('e5_spatial_r1_20260918','m2_r1_20260919'):
  selected=[r for r in manifests[stage] if (stage.startswith('e5') and r.startswith('formal_results/attempts/') and r.endswith('.json')) or (stage.startswith('m2') and r in ('auc_tests.json','test_results.json','report_data.json','summary.csv','paired.csv','image_values.json.gz'))]
  for rel in selected:bind(stage,rel)
 for p in [REPO/'补充实验/补充实验v1.md',REPO/'补充实验/执行协议v1.md',REPO/'src/vitnns/endpoint_supplement_v1.py',REPO/'scripts/run_endpoint_supplement_v1_worker.py']:
  sources[str(p.relative_to(REPO))]=sha(p)
 tasks.sort(key=lambda r:r['task_id'])
 def write(name,d):(ROOT/name).write_bytes(canonical(d)+b'\n')
 write('source_bindings.json',sources);write('inherited.json',inherited);write('task_manifest.json',dict(tasks=tasks))
 write('freeze.json',dict(status='FROZEN_BEFORE_NEW_MODEL_RUNS',time_ns=time.time_ns(),protocol_sha256=sha(REPO/'补充实验/执行协议v1.md'),task_count=len(tasks),task_manifest_digest=digest(tasks),source_bindings_sha256=sha(ROOT/'source_bindings.json'),seeds=seeds,bootstrap_repetitions=2000,optional_P2_P3='NOT_EXECUTED_MINIMUM_PACKAGE'))
 print(create_queue(ROOT/'queue.sqlite3',tasks))
if __name__=='__main__':main()
