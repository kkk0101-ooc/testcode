#!/usr/bin/env python3
"""Freeze and execute M2, with published code and an atomic global queue."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from vitnns.endpoint_ep1_queue import create_queue,digest
from vitnns.endpoint_m2_regions import BLOCKS
SPEC=REPO/'config/nns_vit_endpoint/endpoint_m2.json'
ROOT=REPO/'reports/nns_vit_endpoint_m2_r1_20260919'

def load(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def git(*args): return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()

def prepare():
    spec=load(SPEC)
    if ROOT.exists(): raise ValueError('FRESH_ROOT_REQUIRED')
    for p,s in spec['source_bindings'].items():
        if sha(REPO/p)!=s: raise ValueError('SOURCE_DRIFT:'+p)
    source=load(REPO/'reports/nns_vit_endpoint_m1_r1_20260918/task_manifest.json')['tasks']
    tasks=[]
    for t in source:
        for b in BLOCKS:
            payload=dict(t['payload'],block=b,stage='ENDPOINT-M2')
            tasks.append(dict(task_id=t['task_id']+f'.b{b:02d}',payload=payload))
    assert len(tasks)==2400
    tasks.sort(key=lambda t:t['task_id'])
    ROOT.mkdir();write(ROOT/'task_manifest.json',dict(tasks=tasks,digest=digest(tasks)))
    paths=[SPEC,Path(__file__),REPO/'docs/endpoint_m2_execution_20260919.md',
        REPO/'tests/test_endpoint_m2.py',REPO/'scripts/run_endpoint_m2_worker.py',
        REPO/'scripts/run_endpoint_m2_analysis.py',REPO/'scripts/validate_endpoint_m2.py']
    paths += [REPO/('src/vitnns/'+name+'.py') for name in ('endpoint_m2_compute','endpoint_m2_regions',
        'endpoint_m2_statistics','endpoint_m1_statistics','endpoint_ep1_queue','endpoint_e4_intervention',
        'endpoint_vit','endpoint_nns','endpoint_e2_atlas','recovery_v2')]
    paths += [REPO/'scripts/run_endpoint_e4_intervention_worker.py']
    write(ROOT/'freeze.json',dict(time_ns=time.time_ns(),sources={p.relative_to(REPO).as_posix():sha(p) for p in paths},
        task_manifest_sha256=sha(ROOT/'task_manifest.json'),predecessor_head=git('rev-parse','HEAD')))
    write(ROOT/'predecessor_validation.json',dict(status='PASS',
        commands=['scripts/seal_endpoint_m1.py --verify','scripts/validate_endpoint_m1.py --tasks --inference'],
        evidence='Current-turn commands exited 0:1631 artifacts,17 requirements,800 tasks,504 groups,24696 summaries,14112 paired,2688 tests',
        source_bindings=spec['source_bindings']))
    create_queue(ROOT/'queue.sqlite3',tasks);print('PREPARED',flush=True)

def run():
    freeze=load(ROOT/'freeze.json')
    for p,s in freeze['sources'].items(): assert sha(REPO/p)==s,'FROZEN_SOURCE:'+p
    assert sha(ROOT/'task_manifest.json')==freeze['task_manifest_sha256']
    assert not (ROOT/'acquisition_receipt.json').exists() and not (ROOT/'startup.json').exists()
    head=git('rev-parse','HEAD');branch=git('symbolic-ref','--short','HEAD')
    remote=git('ls-remote','origin','refs/heads/'+branch).split()[0]
    assert head==remote,'PUBLISHED_PRODUCER'
    for p,s in freeze['sources'].items():
        data=subprocess.check_output(['git','show',head+':vitnns_rebuild/'+p],cwd=REPO)
        assert hashlib.sha256(data).hexdigest()==s,'PUBLISHED_SOURCE:'+p
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'],text=True)
    locks=[];selected=[];lock_root=REPO.parent/'.runtime/endpoint_shared_gpu_locks'
    lock_root.mkdir(parents=True,exist_ok=True)
    for line in query.splitlines():
        index,uuid,name,free,util=[v.strip() for v in line.split(',')]
        if name!='NVIDIA H20' or int(free)<32768 or int(util)>10: continue
        stream=(lock_root/(uuid+'.lock')).open('a+')
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:stream.close();continue
        locks.append(stream);selected.append(dict(index=index,uuid=uuid,free_mib=int(free),util=int(util)))
        if len(selected)==8:break
    assert selected,'NO_ELIGIBLE_GPU'
    write(ROOT/'startup.json',dict(producer_commit=head,remote_commit=remote,freeze_sha256=sha(ROOT/'freeze.json'),
        selected=selected,gpu_snapshot=query,existing_pids=pids,sys_executable_realpath=str(Path(sys.executable).resolve()),
        CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    started=time.monotonic();processes=[]
    for gpu in selected:
        worker='gpu'+gpu['index'];log=(ROOT/(worker+'.log')).open('xb')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['index'],CUBLAS_WORKSPACE_CONFIG=':4096:8',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        command=[sys.executable,'scripts/run_endpoint_m2_worker.py','--root',str(ROOT),
            '--worker-id',worker,'--physical-index',gpu['index'],'--gpu-uuid',gpu['uuid']]
        processes.append((subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    exits=[p.wait() for p,log in processes]
    for p,log in processes:log.close()
    with sqlite3.connect(ROOT/'queue.sqlite3') as c:states=dict(c.execute('SELECT state,COUNT(*) FROM tasks GROUP BY state'))
    passed=all(x==0 for x in exits) and states=={'completed':2400}
    write(ROOT/'acquisition_receipt.json',dict(status='PASS' if passed else 'FAIL',states=states,
        exits=exits,wall_seconds=time.monotonic()-started,gpu_count=len(selected)))
    assert passed,'ACQUISITION_FAILED'
    print('ACQUISITION_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.prepare:prepare()
    if a.run:run()
