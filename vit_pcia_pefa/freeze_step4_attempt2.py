#!/usr/bin/env python3
"""Freeze the execution-coherent Step-4 attempt2 recovery."""
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;EP0=REPO/'vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918'
FILES=['执行规划.md','method_contract.md','step3_decision.md','quadrature_registry.json','step4_attempt0_receipt.json','step4_attempt1_receipt.json','step4_attempt2_amendment.md','vit_port_registry.json','step4_runtime/discovery_receipt.json','src/vit_adapter.py','tests/test_step4_vit_adapter.py','run_step4_attempt2_worker.py','freeze_step4_attempt2.py']
EXTERNAL=['vitnns_rebuild/src/vitnns/endpoint_vit.py','vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json','vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def main():
 out=ROOT/'step4_attempt2_source_binding.json'
 if out.exists():raise RuntimeError('ATTEMPT2_BINDING_EXISTS')
 files=[]
 for rel in FILES:
  p=ROOT/rel;files.append({'path':p.relative_to(REPO).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 for rel in EXTERNAL:
  p=REPO/rel;files.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
 models=json.loads((EP0/'model_registry.json').read_text())['models']
 checkpoints=[]
 for model in models:
  p=REPO/'vitnns_rebuild'/model['checkpoint']['path'];actual={'model_id':model['model_id'],'path':p.relative_to(REPO).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
  if (actual['bytes'],actual['sha256'])!=(model['checkpoint']['bytes'],model['checkpoint']['sha256']):raise RuntimeError('CHECKPOINT_IDENTITY:'+model['model_id'])
  checkpoints.append(actual)
 nodes,weights=np.polynomial.legendre.leggauss(512);nodes=np.ascontiguousarray((nodes+1)/2,dtype='<f8');weights=np.ascontiguousarray(weights/2,dtype='<f8')
 v={'schema_version':1,'stage':'ENDPOINT_PCIA_PEFA_STEP4_ATTEMPT2','status':'FROZEN_PROSPECTIVE','producer_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'files':files,'checkpoints':checkpoints,'sample':{'split':'discovery','split_order':0,'models_share_row':True},'models':['vit_b16','deit_s16'],'target':'frozen EP0 ground-truth raw logit','source_coordinate_resolution':'one exact all-coordinate contraction','T_e':'Q_e(x)-Q_e(0)','R_e':'T_e-A_e','path_ig_reference':'the __full__ field in the same 512-node PCIA execution','independent_ig_role':'cost baseline and batch-sensitivity diagnostic only','nodes_sha256':hashlib.sha256(nodes.tobytes()).hexdigest(),'weights_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),'numeric':{'atol':1e-5,'rtol':1e-4,'near_zero':1e-6,'forward_dtype':'float32','reduction_dtype':'float64'},'scope':{'primary_block':0,'all_blocks':list(range(12)),'cut_families':9,'omega_direction_seeds':[20260921,20260922]},'cost_gate':{'h20_hours':16,'wall_hours':24,'retained_gib':32,'gpu_memory_gib_per_worker':24,'cpu_rss_gib_per_worker':32,'task_seconds':1800,'projection_images':64,'projection_models':2,'projection_blocks':12},'step5_authorized':False};v['canonical_digest']=canon(v)
 fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'w') as f:json.dump(v,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'PASS_ATTEMPT2_FREEZE','canonical_digest':v['canonical_digest']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
