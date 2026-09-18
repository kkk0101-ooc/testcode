#!/usr/bin/env python3
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    raise RuntimeError('CUBLAS_PREIMPORT')
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from run_endpoint_e4_intervention_worker import load_model, load_inputs, worker_identity, abandon_task
from vitnns.endpoint_m1_compute import acquire
from vitnns.endpoint_ep1_queue import claim_task, complete_task, digest
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker-id', required=True)
    p.add_argument('--physical-index', type=int, required=True)
    p.add_argument('--gpu-uuid', required=True)
    args = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(args.physical_index)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    identity = worker_identity(device, args)
    models = {}
    while True:
        claim = claim_task(args.root / 'queue.sqlite3', worker_id=args.worker_id,
                           device_uuid=args.gpu_uuid, lease_seconds=1800, max_attempts=1)
        if claim is None:
            return
        task = claim['payload']
        started = time.monotonic()
        try:
            source = {}
            for name in ('e3', 'e4'):
                data = (REPO / task[name + '_path']).read_bytes()
                assert __import__('hashlib').sha256(data).hexdigest() == task[name + '_sha256']
                value = json.loads(data)
                expected = value.pop('canonical_digest')
                assert digest(value) == expected
                source[name] = value['result']
            if task['model_id'] not in models:
                models[task['model_id']] = load_model(task, device)
            model, binding = models[task['model_id']]
            image, references, _ = load_inputs(task, device)
            torch.cuda.reset_peak_memory_stats()
            value = acquire(model, image, references, task, source['e3'], source['e4'])
            torch.cuda.synchronize()
            value.update(model_id=task['model_id'], sample_id=task['sample_id'],
                         split_order=task['split_order'], class_id=task['class_id'],
                         image_sha256=task['image_sha256'], model_binding=binding,
                         worker_identity=identity, seconds=time.monotonic()-started,
                         peak_gpu_bytes=torch.cuda.max_memory_reserved())
            complete_task(args.root / 'queue.sqlite3', args.root / 'results', claim, value)
            print(json.dumps({'task': claim['task_id'], 'seconds': value['seconds']}), flush=True)
        except Exception as error:
            error_record = dict(task_id=claim['task_id'], error_type=type(error).__name__, error=str(error))
            (args.root / ('failure_' + claim['task_id'] + '.json')).write_text(json.dumps(error_record))
            raise

if __name__ == '__main__':
    main()
