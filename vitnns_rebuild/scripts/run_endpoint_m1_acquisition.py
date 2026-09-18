#!/usr/bin/env python3
"""Freeze source identities then acquire only missing M1 components."""
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from vitnns.endpoint_ep1_queue import create_queue, digest

def load(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    spec_path = REPO / 'config/nns_vit_endpoint/endpoint_m1.json'
    spec = load(spec_path); root = REPO / spec['output_root']
    if args.prepare:
        assert not root.exists(), 'FRESH_ROOT_REQUIRED'
        for relative, expected in spec['source_bindings'].items():
            assert sha(REPO / relative) == expected, relative
        e4 = {r['task_id']: r for r in load(REPO / 'reports/nns_vit_endpoint_e4_intervention_r3_20260918/combined_result_index.json')['records']}
        tasks = load(REPO / 'reports/nns_vit_endpoint_e4_intervention_r2_20260918/task_manifest.json')['tasks']
        for task in tasks:
            tid = task['task_id']; payload = task['payload']
            payload['e3_path'] = f'reports/nns_vit_endpoint_e3_closure_r1_20260918/formal_results/attempts/{tid}/attempt-01.json'
            payload['e4_path'] = e4[tid]['source_result_path']
            for name in ('e3', 'e4'):
                p = REPO / payload[name + '_path']
                value = load(p); expected = value.pop('canonical_digest')
                assert digest(value) == expected
                result = value['result']
                assert (result['sample_id'], result['class_id'], result['image_sha256']) == (payload['sample_id'], payload['class_id'], payload['image_sha256'])
                payload[name + '_sha256'] = sha(p)
        root.mkdir()
        write(root / 'task_manifest.json', {'tasks': tasks, 'digest': digest(tasks)})
        sources = [spec_path, REPO / 'src/vitnns/endpoint_m1_compute.py',
                   REPO / 'scripts/run_endpoint_m1_worker.py', Path(__file__),
                   REPO / 'docs/endpoint_m1_execution_20260918.md',
                   REPO / 'tests/test_endpoint_m1.py']
        write(root / 'freeze.json', {'time_ns': time.time_ns(), 'status': spec['status'],
              'sources': {p.relative_to(REPO).as_posix(): sha(p) for p in sources},
              'task_manifest_sha256': sha(root / 'task_manifest.json'),
              'predecessor_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()})
        create_queue(root / 'queue.sqlite3', tasks)
        print('PREPARED', root, flush=True)
    if args.run:
        freeze = load(root / 'freeze.json')
        for path, expected in freeze['sources'].items():
            assert sha(REPO / path) == expected, 'FROZEN_SOURCE_DRIFT:' + path
        assert not (root / 'acquisition_receipt.json').exists()
        head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        branch = subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=REPO,text=True).strip()
        remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True).split()[0]
        assert head == remote, 'PUBLISHED_PRODUCER_REQUIRED'
        for path in freeze['sources']:
            data = subprocess.check_output(['git','show',head+':vitnns_rebuild/'+path],cwd=REPO)
            assert hashlib.sha256(data).hexdigest() == freeze['sources'][path]
        query = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
        locks = []; selected = []
        lock_root = REPO.parent / '.runtime/endpoint_shared_gpu_locks'; lock_root.mkdir(parents=True, exist_ok=True)
        for line in query.splitlines():
            index, uuid, name, free, util = [x.strip() for x in line.split(',')]
            if name != 'NVIDIA H20' or int(free) < 32768 or int(util) > 10:
                continue
            stream = (lock_root / (uuid + '.lock')).open('a+')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close(); continue
            locks.append(stream); selected.append({'index':index,'uuid':uuid,'free_mib':int(free),'util':int(util)})
            if len(selected) == spec['supplemental_compute']['gpus_max']:
                break
        assert selected, 'NO_ELIGIBLE_GPU'
        write(root / 'startup.json', dict(producer_commit=head, remote_commit=remote,
              freeze_sha256=sha(root/'freeze.json'), gpu_snapshot=query, existing_pids=pids,
              selected=selected, sys_executable_realpath=str(Path(sys.executable).resolve()),
              CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        started = time.monotonic(); processes=[]
        for gpu in selected:
            worker='gpu'+gpu['index']
            log=(root/(worker+'.log')).open('xb')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['index'],CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            command=[sys.executable,'scripts/run_endpoint_m1_worker.py','--root',str(root),'--worker-id',worker,'--physical-index',gpu['index'],'--gpu-uuid',gpu['uuid']]
            processes.append((subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT), log))
        exits=[p.wait() for p,log in processes]
        for p,log in processes:
            log.close()
        with sqlite3.connect(root/'queue.sqlite3') as c:
            states=dict(c.execute('SELECT state,COUNT(*) FROM tasks GROUP BY state').fetchall())
        write(root/'acquisition_receipt.json', dict(exits=exits,states=states,wall_seconds=time.monotonic()-started,gpu_count=len(selected),status='PASS' if all(x==0 for x in exits) and states=={'completed':800} else 'FAIL'))
        assert all(x==0 for x in exits) and states=={'completed':800}, 'ACQUISITION_FAILED'
        print('ACQUISITION_PASS',flush=True)

if __name__ == '__main__':
    main()
