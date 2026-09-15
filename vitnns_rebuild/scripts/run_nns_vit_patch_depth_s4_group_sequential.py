#!/usr/bin/env python3
"""Post-seal cache binding and look-scoped dynamic S4 execution."""
from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'evidence/nns_vit_patch_depth/s4_r9_group_sequential_20260915'
PAYLOAD=REPO/'artifacts/nns_vit_patch_depth/s4_r9_group_sequential_20260915'
R4_ROOT=REPO/'evidence/nns_vit_patch_depth/s4_r4_20260915'
R4_PAYLOAD=REPO/'artifacts/nns_vit_patch_depth/s4_r4_20260915'
PYTHON=REPO.parent/'.conda/envs/vitnns/bin/python'
RUNTIME=REPO.parent/'.runtime/nns_vit_patch_depth_s4_group_sequential'
sys.path.insert(0,str(REPO/'src'))
from vitnns import nns_patch_depth_s4 as s4  # noqa:E402
from vitnns import nns_patch_depth_s4_group_sequential as gs  # noqa:E402
from vitnns import nns_patch_depth_s4_group_queue as queue  # noqa:E402

RUNTIME_KEYS=("TMPDIR","TMP","TEMP","XDG_CACHE_HOME","PIP_CACHE_DIR","CONDA_PKGS_DIRS","HF_HOME","HF_DATASETS_CACHE","TRANSFORMERS_CACHE","TORCH_HOME","TORCH_EXTENSIONS_DIR","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","CUDA_CACHE_PATH","MPLCONFIGDIR","NUMBA_CACHE_DIR")

def require(c:bool,m:str)->None:
    if not c: raise RuntimeError(m)

def git(*args:str)->str: return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()

def remote_oid()->str:
    branch=git('symbolic-ref','--short','HEAD'); line=subprocess.check_output(['git','-C',str(REPO),'ls-remote','--heads','origin',f'refs/heads/{branch}'],text=True).strip(); return line.split()[0] if line else ''

def environment()->dict[str,str]:
    value=os.environ.copy(); mapping={"TMPDIR":"tmp","TMP":"tmp","TEMP":"tmp","XDG_CACHE_HOME":"cache","PIP_CACHE_DIR":"pip","CONDA_PKGS_DIRS":"conda_pkgs","HF_HOME":"hf","HF_DATASETS_CACHE":"hf_datasets","TRANSFORMERS_CACHE":"hf/transformers","TORCH_HOME":"torch","TORCH_EXTENSIONS_DIR":"torch_extensions","TORCHINDUCTOR_CACHE_DIR":"torchinductor","TRITON_CACHE_DIR":"triton","CUDA_CACHE_PATH":"cuda","MPLCONFIGDIR":"matplotlib","NUMBA_CACHE_DIR":"numba"}
    for key,suffix in mapping.items(): path=RUNTIME/suffix; path.mkdir(parents=True,exist_ok=True); value[key]=str(path)
    value.update({"CUBLAS_WORKSPACE_CONFIG":":4096:8","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","NUMEXPR_NUM_THREADS":"2","PYTHONPATH":str(REPO/'src')}); require(all(os.path.realpath(value[k]).startswith('/data01/') for k in RUNTIME_KEYS),'runtime path drift'); return value

def verify_remote_seal(allow_runtime:bool=False)->tuple[str,dict[str,Any]]:
    seal_path=ROOT/'implementation_seal.json'; require(seal_path.is_file(),'implementation seal absent'); seal=json.loads(seal_path.read_text()); require(gs.verify_signed(seal) and seal['status']=='PASS','implementation seal invalid')
    head=git('rev-parse','HEAD'); require(remote_oid()==head,'remote branch does not equal local seal HEAD'); implementation=seal['payload']['implementation_commit']; subprocess.check_call(['git','-C',str(REPO),'merge-base','--is-ancestor',implementation,head])
    tracked=git('ls-tree','-r','--name-only',head).splitlines(); expected='evidence/nns_vit_patch_depth/s4_r9_group_sequential_20260915/implementation_seal.json'; require(expected in tracked or 'vitnns_rebuild/'+expected in tracked,'seal absent from Git tree')
    porcelain=git('status','--porcelain=v2','--untracked-files=all').splitlines(); allowed=['? ../.root_cleanup_quarantine/','? artifacts/nns_vit_patch_depth/s4_r9_group_sequential_20260915/']
    if allow_runtime: allowed += ['? evidence/nns_vit_patch_depth/s4_r9_group_sequential_20260915/']
    unexpected=[line for line in porcelain if not any(line.startswith(prefix) for prefix in allowed)]; require(not unexpected,'unexpected worktree changes: '+repr(unexpected[:10])); return head,seal

def gpu_rows()->list[dict[str,Any]]:
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,compute_mode','--format=csv,noheader,nounits'],text=True); processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    active={}
    for line in processes.splitlines():
        if line.strip(): uuid,pid,used=[x.strip() for x in line.split(',')]; active.setdefault(uuid,[]).append({'pid':int(pid),'used_memory_mib':int(used)})
    rows=[]
    for line in query.splitlines():
        index,uuid,name,total,used,free,util,mode=[x.strip() for x in line.split(',')]; eligible=name=='NVIDIA H20' and int(free)>=21504 and int(util)<=10
        rows.append({'physical_index':int(index),'gpu_uuid':uuid,'name':name,'memory_total_mib':int(total),'memory_used_mib':int(used),'memory_free_mib':int(free),'utilization_percent':int(util),'compute_mode':mode,'existing_compute_processes':active.get(uuid,[]),'existing_compute_PID_disqualifying':False,'eligible':eligible,'eligibility_reason':'PASS' if eligible else 'name_memory_or_utilization_threshold'})
    return sorted(rows,key=lambda r:(r['physical_index'],r['gpu_uuid']))

def cache_rebind()->None:
    head,seal=verify_remote_seal(False); destination=ROOT/'selected_cache_rebind.json'; require(not destination.exists(),'cache rebind already exists')
    schedule=json.loads((REPO/gs.SCHEDULE_REL).read_text())['payload']['records']; r4_doc=json.loads((R4_ROOT/'selected_member_materialization.json').read_text()); require(s4.verify_signed(r4_doc) and r4_doc['status']=='PASS','R4 materialization evidence invalid'); by_index={int(row['schedule_index']):row for row in r4_doc['payload']['records']}; require(len(by_index)==1744,'R4 cache record count drift')
    jpeg=REPO/s4.JPEG_ARCHIVE_REL; mask=REPO/s4.MASK_ARCHIVE_REL; require(jpeg.stat().st_size==s4.JPEG_ARCHIVE_BYTES and gs.sha_file(jpeg)==s4.JPEG_ARCHIVE_SHA256,'JPEG archive drift'); require(gs.sha_file(mask)==s4.MASK_ARCHIVE_SHA256,'mask archive drift')
    from PIL import Image
    records=[]
    for row in schedule:
        index=int(row['schedule_index']); old=by_index[index]; image_bytes=(REPO/old['image_cache_path']).read_bytes(); mask_bytes=(REPO/old['mask_cache_path']).read_bytes(); require(len(image_bytes)==int(row['image_bytes']) and hashlib_sha(image_bytes)==row['image_sha256'],'cached JPEG drift'); require(hashlib_sha(mask_bytes)==row['annotation_sha256'],'cached mask drift')
        with Image.open(io.BytesIO(image_bytes)) as image: image.load(); image_size=list(image.size)
        with Image.open(io.BytesIO(mask_bytes)) as image: image.load(); mask_size=list(image.size)
        require(image_size==[row['image_width'],row['image_height']] and mask_size==[row['mask_width'],row['mask_height']],'cached geometry drift')
        image_path=PAYLOAD/'selected_member_cache/images'/f'{index:04d}.jpeg'; mask_path=PAYLOAD/'selected_member_cache/masks'/f'{index:04d}.png'; s4.write_bytes_atomic(image_path,image_bytes); s4.write_bytes_atomic(mask_path,mask_bytes)
        records.append({'schedule_index':index,'source_id':row['source_image_id'],'image_cache_path':image_path.relative_to(REPO).as_posix(),'image_sha256':row['image_sha256'],'image_bytes':len(image_bytes),'image_size_wh':image_size,'mask_cache_path':mask_path.relative_to(REPO).as_posix(),'mask_sha256':row['annotation_sha256'],'mask_bytes':len(mask_bytes),'mask_size_wh':mask_size})
    gs.write_json_new(destination,gs.signed('s4_group_sequential_selected_cache_rebind','PASS',{'producer_commit':head,'implementation_seal_sha256':gs.sha_file(ROOT/'implementation_seal.json'),'source_materialization':{'path':(R4_ROOT/'selected_member_materialization.json').relative_to(REPO).as_posix(),'sha256':gs.sha_file(R4_ROOT/'selected_member_materialization.json'),'canonical_digest':r4_doc['canonical_digest']},'authorized_source_count':1744,'incremental_archive_JPEG_member_reads':0,'incremental_archive_mask_member_reads':0,'revalidated_cache_JPEG_reads':1744,'revalidated_cache_mask_reads':1744,'D2_only_member_reads':0,'records_digest':gs.canonical_digest(records),'records':records},producer_commit=head)); print(json.dumps({'status':'PASS','cache_records':len(records),'incremental_archive_member_reads':0},sort_keys=True))

def hashlib_sha(value:bytes)->str:
    import hashlib; return hashlib.sha256(value).hexdigest()

def worker_command(launch_id:str,look:int,worker_id:str,device:Mapping[str,Any],producer_commit:str)->list[str]:
    code="from vitnns.nns_patch_depth_s4_worker import execute_group_sequential; import json; print(json.dumps(execute_group_sequential(%r,%r,%r,%r,%d,%r,%d,%r,%r),sort_keys=True))"%(str(REPO),str(ROOT),str(PAYLOAD),launch_id,look,worker_id,int(device['physical_index']),str(device['gpu_uuid']),producer_commit); return [str(PYTHON),'-c',code]

def launch(look:int,launch_id:str)->None:
    head,seal=verify_remote_seal(True); require((ROOT/'selected_cache_rebind.json').is_file(),'cache rebind absent'); require((ROOT/'s4_r4_completed_terminal_rebind_inventory.json').is_file(),'terminal rebind absent'); schedule=json.loads((REPO/gs.SCHEDULE_REL).read_text())['payload']['records']; require(1<=look<=5,'look out of range')
    if look>1:
        prior=json.loads((ROOT/f'interim/look-{look-1}/decision.json').read_text()); require(prior['status']=='CONTINUE_TO_NEXT_LOOK','prior interim did not authorize next look')
    for earlier in range(1,look):
        prior=json.loads((ROOT/f'interim/look-{earlier}/decision.json').read_text()); require(prior['status']!='STOP_SUCCESS','later tranche forbidden after STOP_SUCCESS')
    missing=queue.missing_units(ROOT,schedule,look); launch_root=ROOT/'launches'/launch_id; require(not launch_root.exists(),'fresh launch root required'); rows=gpu_rows(); eligible=[row for row in rows if row['eligible']]; selected=eligible[:8]; launch_root.mkdir(parents=True)
    if not missing:
        counts=queue.terminal_counts(ROOT,schedule,look); gs.write_json_new(launch_root/'launch_manifest.json',gs.signed('s4_group_sequential_launch_manifest','FROZEN',{'launch_id':launch_id,'look':look,'GPU_snapshot':rows,'selected_GPU_count':0,'missing_units_at_launch':0,'no_worker_required':True},producer_commit=head)); gs.write_json_new(launch_root/'launch_terminal.json',gs.signed('s4_group_sequential_launch_terminal','PASS',{'launch_id':launch_id,'look':look,'remaining_units':[],'terminal_count':counts['terminal'],'planned_unit_count':counts['planned'],'no_worker_started':True},producer_commit=head)); publish_look_completion(look,schedule,head); print(json.dumps({'status':'PASS','look':look,'missing':0},sort_keys=True)); return
    if not selected:
        gs.write_json_new(launch_root/'launch_terminal.json',gs.signed('s4_group_sequential_launch_terminal','BLOCKED_INFRASTRUCTURE',{'launch_id':launch_id,'look':look,'GPU_snapshot':rows,'remaining_unit_count':len(missing),'scientific_terminal_created':False},producer_commit=head)); print(json.dumps({'status':'BLOCKED_INFRASTRUCTURE','look':look},sort_keys=True)); return
    if len(eligible)>=8: require(len(selected)==8,'eight eligible require eight workers')
    gs.write_json_new(launch_root/'launch_manifest.json',gs.signed('s4_group_sequential_launch_manifest','FROZEN',{'launch_id':launch_id,'look':look,'created_unix_ns':time.time_ns(),'producer_commit':head,'GPU_snapshot':rows,'eligible_GPU_count':len(eligible),'selected_GPU_count':len(selected),'selected_GPUs':selected,'worker_count':len(selected),'missing_units_at_launch':len(missing),'already_resolved_units':3*gs.LOOK_IMAGE_COUNTS[look-1]-len(missing),'future_tranche_claim_count':0,'dynamic_queue':True,'static_shards':False,'GPU_set_fixed':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8'},producer_commit=head))
    locks=[]; processes=[]; failures=[]; started=time.monotonic(); started_ns=time.time_ns()
    try:
        for device in selected:
            lock_path=RUNTIME/'gpu_locks'/(device['gpu_uuid']+'.lock'); lock_path.parent.mkdir(parents=True,exist_ok=True); handle=lock_path.open('a+'); fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(handle)
        for slot,device in enumerate(selected):
            worker_id=f'worker-{slot:02d}'; child=environment(); child['CUDA_VISIBLE_DEVICES']=str(device['physical_index']); log=launch_root/'workers'/worker_id/'worker.log'; log.parent.mkdir(parents=True,exist_ok=True); handle=log.open('xb'); processes.append((worker_id,subprocess.Popen(worker_command(launch_id,look,worker_id,device,head),cwd=REPO,env=child,stdout=handle,stderr=subprocess.STDOUT),handle))
        for worker_id,process,handle in processes:
            code=process.wait(); handle.flush(); os.fsync(handle.fileno()); handle.close();
            if code!=0: failures.append({'worker_id':worker_id,'exit_code':code})
    finally:
        for handle in locks: fcntl.flock(handle.fileno(),fcntl.LOCK_UN); handle.close()
    remaining=queue.missing_units(ROOT,schedule,look); status='PASS' if not remaining and not failures else 'BLOCKED_INFRASTRUCTURE' if failures else 'FAILED_IMPLEMENTATION'; counts=queue.terminal_counts(ROOT,schedule,look)
    gs.write_json_new(launch_root/'launch_terminal.json',gs.signed('s4_group_sequential_launch_terminal',status,{'launch_id':launch_id,'look':look,'started_unix_ns':started_ns,'completed_unix_ns':time.time_ns(),'launch_elapsed_seconds':time.monotonic()-started,'selected_GPU_count':len(selected),'selected_GPUs':selected,'worker_process_failures':failures,'planned_unit_count':counts['planned'],'terminal_count_after_launch':counts['terminal'],'remaining_units':[{'unit_type':x,'schedule_index':i} for x,i in remaining],'future_tranche_claim_count':0,'cost_or_wall_time_termination_used':False},producer_commit=head))
    if status=='PASS': publish_look_completion(look,schedule,head)
    print(json.dumps({'status':status,'look':look,'remaining_units':len(remaining),'worker_failures':failures},sort_keys=True))

def publish_look_completion(look:int,schedule:list[dict[str,Any]],producer_commit:str)->None:
    destination=ROOT/f'look_{look}_completion_receipt.json'
    if destination.exists(): return
    counts=queue.terminal_counts(ROOT,schedule,look); require(counts['planned']==counts['terminal'],'look topology incomplete'); state_counts={}
    for row in counts['records']: state_counts[row['status']]=state_counts.get(row['status'],0)+1
    future_claims=[]
    for path in ROOT.glob('launches/*/claims/*/*.json'):
        d=json.loads(path.read_text()); p=d['payload'];
        if int(p.get('first_authorized_look',look))>look: future_claims.append(path.relative_to(REPO).as_posix())
    require(not future_claims,'future tranche claimed')
    gs.write_json_new(destination,gs.signed('s4_group_sequential_look_completion','PASS',{'look':look,'class_count':gs.LOOK_CLASS_COUNTS[look-1],'image_count':gs.LOOK_IMAGE_COUNTS[look-1],'planned_unit_count':counts['planned'],'terminal_count':counts['terminal'],'state_counts':state_counts,'source_completion_count':sum((ROOT/'source_completions'/f"source-{int(row['schedule_index']):04d}.json").is_file() for row in schedule if int(row['first_authorized_look'])<=look),'future_tranche_claim_count':0,'scientific_aggregate_access_count':0},producer_commit=producer_commit))

def main()->int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True); sub.add_parser('cache-rebind'); launch_parser=sub.add_parser('launch'); launch_parser.add_argument('--look',type=int,required=True); launch_parser.add_argument('--launch-id',required=True); args=parser.parse_args(); cache_rebind() if args.command=='cache-rebind' else launch(args.look,args.launch_id); return 0

if __name__=='__main__': raise SystemExit(main())
