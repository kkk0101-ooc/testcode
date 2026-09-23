#!/usr/bin/env python3
"""Freeze v6 protocol, schemas, fixed cohorts and source identities before outcomes."""
import datetime,hashlib,json,os,platform,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from vitnns.endpoint_ep1_queue import create_queue
ROOT=R/'reports/nns_vit_endpoint_revision_v6_r1_20260923'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write(chr(10))
def closed(props):return dict(type='object',properties=props,required=list(props),additionalProperties=False)
def main():
 assert not ROOT.exists(),'FRESH_ROOT_REQUIRED'
 assert json.loads((R/'papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v6/stage1_validation.json').read_text())['status']=='PASS'
 tasks=[];source=[]
 required=['cache_path','checkpoint','checkpoint_bytes','checkpoint_sha256','class_id','embedding_dim','image_bytes','image_sha256','mean','std','model_id','sample_id','tensor_digest','timm_name']
 for kind,root in [('units','nns_vit_endpoint_m3_r1_20260919'),('structure','nns_vit_endpoint_m4_r1_20260919')]:
  path=R/'reports'/root/'task_manifest.json';source.append(dict(path=str(path.relative_to(R)),sha256=sha(path),bytes=path.stat().st_size))
  old=json.loads(path.read_text())['tasks']
  for t in old:
   payload={k:t['payload'][k] for k in required};payload.update(kind=kind,tokens=t['payload'].get('tokens',[]),channels=t['payload'].get('channels',{}))
   tasks.append(dict(task_id=kind+'__'+t['task_id'],payload=payload))
 assert len(tasks)==1000 and sum(t['payload']['kind']=='units' for t in tasks)==200
 pilots=[next(t for t in tasks if t['payload']['kind']==kind and t['payload']['model_id']==model) for kind in ['structure','units'] for model in ['vit_b16','deit_s16']]
 scripts=['scripts/prepare_revision_v6.py','scripts/run_revision_v6_worker.py','scripts/run_revision_v6_analytic.py','src/vitnns/revision_v6_coordinate.py','src/vitnns/revision_v6_analytic.py','src/vitnns/endpoint_vit.py','src/vitnns/endpoint_m4.py','src/vitnns/endpoint_nns.py','src/vitnns/endpoint_ep1_queue.py','scripts/run_endpoint_e4_intervention_worker.py']
 protocol=dict(schema_version=1,namespace='ENDPOINT_REVISION_V6_R1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),authorization='user authorized v6 all stages; no pre-experiment commit required',coordinates=['normalized_endpoint','raw_gradient','common_origin'],coordinate_formulas=['x * grad_x','grad_x / sigma','(x + mu / sigma) * grad_x'],common_origin_rgb=[0,0,0],rgb_range=[0,1],units=dict(images_per_model=100,models=['vit_b16','deit_s16'],blocks=list(range(12)),tokens=16,channels_per_block=32,targets_per_image=6144,metrics=['entropy','locality','distance'],contrast='median(all late block units) minus median(all early block units); CLS excluded for locality/distance'),structure=dict(images_per_model=400,models=['vit_b16','deit_s16'],blocks=list(range(12)),objects=['Q','K','V','attention_output','attention_identity','mlp_output','mlp_identity'],metrics=['entropy','output_identity_tv'],contrast='late blocks 8-11 median minus early 0-3 median; additionally block11 minus block0 for exact legacy QKV comparison'),statistics=dict(unit='image',bootstrap=10000,seed=2026092302,interval='percentile 95%',paired_coordinates=True,missing='zero mass is NA; keep source image in expected denominator; report valid n and NA; no replacement',identity='prospective robustness',decision='directions consistent with all intervals excluding zero across three representations permits only three-coordinate robustness; otherwise normalized-input-only claims; no sign-based execution failure'),analytic=dict(dtype='float64',cpu_only=True,fixtures=['single_head_attention','double_head_attention','preln_residual_attention_mlp','shared_producer_two_slots'],seed=2026092301,component_atol=1e-11,component_rtol=1e-9,fd_epsilon=1e-5,fd_atol=1e-8,fd_rtol=1e-6,components=['query','key','value','routing','heads','identity','attention','branch_reconstruction','full','with_bypass'],mutations=['wrong_slot','merged_fanout','merged_fanout_duplicated','attention_wrong_qk_slot']),thresholds=dict(correlation=[.05,.10,.15],regional_gain=[0,.025,.05,.10],identity='descriptive sensitivity; original families and decisions unchanged'),resources=dict(tasks=1000,pilot_tasks=4,estimated_gpu_hours=[4,16],estimated_wall_hours_4_gpus=[1,4],estimated_retained_gib=6,max_gpu_hours=32,max_wall_hours=12,min_free_disk_gib=50,max_gpus=4,min_free_mib=32768,max_utilization=10,task_timeout_seconds=1800,dynamic_atomic_queue=True),failure='definition/source/schema/nonfinite/runtime conflict stops branch; immutable failure receipt then new versioned root; no cohort shrink',outcome_schema=dict(npz_keys=['block','object','coordinate','token','channel','patch_mass'],units_rows=18432,structure_rows=252,patch_shape=196,dtype='float32 patch mass; int8 block; int16 token/channel; U20 object; U24 coordinate',raw_pilot='first unit in each frozen chunk, every structural map; NPZ keys equal retained row indexes',image_csv=['model','image','coordinate','object','metric','contrast','early','late','value','valid_units','expected_units'],summary_csv=['model','coordinate','object','metric','contrast','n','missing','median','ci_low','ci_high','direction'],threshold_csv=['family','metric','threshold','n','median','ci_low','ci_high','point_exceeds','ci_exceeds','original_composite']))
 ROOT.mkdir()
 dump(ROOT/'protocol.json',protocol);dump(ROOT/'source_bindings.json',dict(schema_version=1,inputs=source,scripts=[dict(path=p,sha256=sha(R/p),bytes=(R/p).stat().st_size) for p in scripts]))
 dump(ROOT/'cohort_bindings.json',dict(schema_version=1,tasks=tasks,pilot_task_ids=[t['task_id'] for t in pilots]))
 dump(ROOT/'environment.json',dict(schema_version=1,python_realpath=str(Path(sys.executable).resolve()),python=platform.python_version(),platform=platform.platform(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES',''),free_disk_bytes=shutil.disk_usage(R).free,git_commit=protocol['git_commit']))
 # Closed schemas are defined prospectively from the declared data contracts.
 num=dict(type='number');s=dict(type='string');b=dict(type='boolean');i=dict(type='integer')
 row=closed(dict(fixture=s,component=s,max_abs_error=num,relative_l2=num,atol=num,rtol=num,passed=b))
 schema=closed(dict(schema_version=dict(const=1),status=dict(enum=['PASS','FAIL']),cpu_only=dict(const=True),seconds=num,rows=dict(type='array',items=row),array_sha256=s))
 dump(ROOT/'analytic_reference.schema.json',schema)
 mutationrow=closed(dict(mutation=s,rejected=b,reason=s,correct_pass=b))
 toyprops={"('producer', 'consumer', 0, 0)":dict(type='array',items=num,minItems=3,maxItems=3),"('producer', 'consumer', 1, 0)":dict(type='array',items=num,minItems=3,maxItems=3)}
 dump(ROOT/'mutation.schema.json',closed(dict(schema_version=dict(const=1),status=dict(enum=['PASS','FAIL']),rows=dict(type='array',items=mutationrow),fanout_maps=closed(toyprops))))
 create_queue(ROOT/'pilot_queue.sqlite3',pilots);create_queue(ROOT/'queue.sqlite3',tasks)
 dump(ROOT/'protocol_freeze.json',dict(schema_version=1,created_utc=protocol['created_utc'],sha256={p.name:sha(p) for p in ROOT.iterdir() if p.suffix=='.json'},outcomes_accessed=False,stage1_gate_sha256=sha(R/'papers/nns_vit_endpoint/publication_r2_20260920/qc/revision_v6/stage1_validation.json')))
 print('FROZEN',ROOT,len(tasks),len(pilots))
if __name__=='__main__':main()
