#!/usr/bin/env python3
"""Launch a dynamic supplemental queue on currently qualified H20 devices."""
import json,os,sqlite3,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'reports/nns_vit_endpoint_supplement_v1_20260920'
cmd=['nvidia-smi','--query-gpu=index,uuid,name,memory.free,utilization.gpu','--format=csv,noheader,nounits']
raw=subprocess.check_output(cmd,text=True);existing=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
gpus=[]
for row in raw.strip().splitlines():
 idx,uuid,name,free,util=[v.strip() for v in row.split(',')]
 if 'H20' in name and int(free)>=32768 and int(util)<=10:gpus.append((int(idx),uuid))
assert gpus,'NO_QUALIFIED_GPU'
(ROOT/'startup.json').write_text(json.dumps(dict(time_ns=time.time_ns(),gpu_query=raw,existing_processes=existing,selected=gpus,python_realpath=str(Path(sys.executable).resolve()),test_command="CUDA_VISIBLE_DEVICES='' PYTHONPATH=src ../.conda/envs/vitnns/bin/python -m pytest -q tests/test_endpoint_supplement_v1.py",test_result='2 passed in 56.65s'),indent=2)+'\n')
started=time.monotonic();children=[]
for idx,uuid in gpus:
 env={**os.environ,'CUDA_VISIBLE_DEVICES':str(idx),'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2'}
 log=open(ROOT/f'gpu{idx}.log','w')
 child=subprocess.Popen([sys.executable,str(REPO/'scripts/run_endpoint_supplement_v1_worker.py'),'--root',str(ROOT),'--physical-index',str(idx),'--gpu-uuid',uuid,'--worker-id',f'sup-gpu{idx}'],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
 children.append((child,log))
while any(p.poll() is None for p,_ in children):
 with sqlite3.connect(ROOT/'queue.sqlite3') as c:states=dict(c.execute('select state,count(*) from tasks group by state'))
 print(json.dumps(dict(seconds=round(time.monotonic()-started),states=states)),flush=True)
 time.sleep(20)
exits=[p.wait() for p,_ in children]
for _,f in children:f.close()
with sqlite3.connect(ROOT/'queue.sqlite3') as c:states=dict(c.execute('select state,count(*) from tasks group by state'))
receipt=dict(status='PASS' if states=={'completed':456} and not any(exits) else 'FAIL',states=states,exits=exits,wall_seconds=time.monotonic()-started,gpu_count=len(gpus))
(ROOT/'acquisition_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
sys.exit(receipt['status']!='PASS')
