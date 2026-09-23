#!/usr/bin/env python3
"""Acquire idle H20 locks, launch the frozen atomic queue, and preserve failures."""
import argparse,fcntl,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();root=a.root.resolve();mode='pilot' if a.pilot else 'production'
 if not a.pilot:assert json.loads((root/'pilot_validation.json').read_text())['status']=='PASS'
 snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
 selected=[];locks=[];lockroot=R.parent/'.runtime/endpoint_shared_gpu_locks';lockroot.mkdir(parents=True,exist_ok=True)
 for line in snapshot.splitlines():
  index,uuid,name,free,util=[x.strip() for x in line.split(',')]
  if name!='NVIDIA H20' or int(free)<32768 or int(util)>10:continue
  lock=(lockroot/(uuid+'.lock')).open('a+')
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:lock.close();continue
  locks.append(lock);selected.append(dict(index=index,uuid=uuid,free_mib=int(free),util=int(util)))
  if len(selected)==(2 if a.pilot else 4):break
 assert selected,'NO_ELIGIBLE_GPU'
 receipt=dict(schema_version=1,mode=mode,selected=selected,gpu_snapshot=snapshot,existing_pids=pids,python_realpath=str(Path(sys.executable).resolve()),started_ns=time.time_ns(),CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with (root/(mode+'_startup.json')).open('x') as f:json.dump(receipt,f,indent=2)
 children=[];started=time.monotonic()
 for gpu in selected:
  log=(root/(mode+'_gpu'+gpu['index']+'.log')).open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['index'],CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  cmd=[sys.executable,'scripts/run_revision_v6_worker.py','--root',str(root),'--worker-id',mode+'_gpu'+gpu['index'],'--physical-index',gpu['index'],'--gpu-uuid',gpu['uuid']]+(['--pilot'] if a.pilot else [])
  children.append((subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT),log))
 exits=[p.wait() for p,log in children]
 for p,log in children:log.close()
 with sqlite3.connect(root/('pilot_queue.sqlite3' if a.pilot else 'queue.sqlite3')) as c:states=dict(c.execute('SELECT state,COUNT(*) FROM tasks GROUP BY state'))
 expected=4 if a.pilot else 1000;passed=all(x==0 for x in exits) and states=={'completed':expected}
 result=dict(schema_version=1,status='PASS' if passed else 'FAIL',mode=mode,exits=exits,states=states,wall_seconds=time.monotonic()-started,gpu_hours_reserved=(time.monotonic()-started)*len(selected)/3600)
 with (root/(mode+'_acquisition.json')).open('x') as f:json.dump(result,f,indent=2)
 if not passed:
  with (root/'terminal_receipt.json').open('x') as f:json.dump(dict(schema_version=1,status='FAIL_IMPLEMENTATION_OR_CONTRACT',source=mode+'_acquisition.json'),f,indent=2)
 print(json.dumps(result));return not passed
if __name__=='__main__':raise SystemExit(main())
