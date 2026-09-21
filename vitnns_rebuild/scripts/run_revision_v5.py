#!/usr/bin/env python3
"""Atomic global task queue, four idle GPU workers, and immutable v5 result records."""
import argparse,hashlib,json,os,sqlite3,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
M=Path(os.environ.get('REVISION_V5_MANIFEST_ROOT',R/'manifests/nns_vit_revision_v5')).resolve()
ROOT=Path(os.environ.get('REVISION_V5_RUN_ROOT',R/'reports/nns_vit_revision_v5_r1_20260921')).resolve()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def prepare(kind):
 p=read(M/'protocol.json');bindings=read(M/'image_bindings.json')
 assert bindings['protocol_sha256']==sha(M/'protocol.json')
 db=ROOT/(kind+'.sqlite3');ROOT.mkdir(parents=True,exist_ok=True)
 assert not db.exists(),'QUEUE_EXISTS'
 with sqlite3.connect(db) as c:
  c.execute('CREATE TABLE tasks (id TEXT PRIMARY KEY, model TEXT, payload TEXT, state TEXT, worker TEXT, start REAL, end REAL, error TEXT)')
  for m in p['models']:
   if kind in ('e1','e2','e3') and m['role']!='original':continue
   for row in bindings['rows']:
    if kind=='e1' and row['cohort']!='e1':continue
    if kind in ('e2','e3') and not row['deep_study']:continue
    if kind=='e5' and row['cohort']!='e5':continue
    tid=m['model_id']+'__'+row['sample_id'].split('.')[0]
    c.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?)',(tid,m['model_id'],json.dumps(row),'pending',None,None,None,None))
  print(kind,c.execute('SELECT COUNT(*) FROM tasks').fetchone()[0])
def worker(args):
 assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
 import fcntl
 lockdir=R.parent/'.runtime/endpoint_shared_gpu_locks';lockdir.mkdir(parents=True,exist_ok=True)
 stream=(lockdir/(args.uuid+'.lock')).open('a+')
 fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import numpy as np
 import torch,timm
 from timm.layers import set_fused_attn
 sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'scripts'))
 from run_endpoint_e4_intervention_worker import load_inputs
 from vitnns.revision_v5_scan import neuron_scan
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;set_fused_attn(False)
 device=torch.device('cuda:0');probe=torch.randn(4,4,device=device,requires_grad=True);(probe@probe.T).square().sum().backward()
 protocol=read(M/'protocol.json');targets=read(M/'targets.json')
 assert sha(M/'targets.json')==protocol['targets_sha256']
 source_files=[Path(__file__),R/'src/vitnns/revision_v5_scan.py',R/'src/vitnns/endpoint_m3_atlas.py',R/'scripts/run_endpoint_e4_intervention_worker.py']
 if args.kind in ('e2','e3','e5'):source_files.append(R/'src/vitnns/revision_v5_extended.py')
 identity=dict(kind=args.kind,pid=os.getpid(),gpu_index=args.gpu,gpu_uuid=args.uuid,python=str(Path(sys.executable).resolve()),torch=torch.__version__,timm=timm.__version__,cublas=os.environ['CUBLAS_WORKSPACE_CONFIG'],deterministic=True,tf32=False,probe='PASS',protocol_sha256=sha(M/'protocol.json'),sources={str(p.relative_to(R)):sha(p) for p in source_files})
 write(ROOT/'startup'/f'{args.kind}_{args.gpu}_{os.getpid()}.json',identity)
 models={m['model_id']:m for m in protocol['models']};model=None;mid=None
 db=ROOT/(args.kind+'.sqlite3');who=str(os.getpid());completed=0
 while True:
  with sqlite3.connect(db,timeout=60,isolation_level=None) as c:
   c.execute('BEGIN IMMEDIATE')
   row=c.execute("SELECT id,model,payload FROM tasks WHERE state='pending' ORDER BY (model=?) DESC,id LIMIT 1",(mid or '',)).fetchone()
   if row is None:c.commit();break
   tid,newmid,payload=row;c.execute("UPDATE tasks SET state='running',worker=?,start=? WHERE id=?",(who,time.time(),tid));c.commit()
  start=time.monotonic()
  try:
   if mid!=newmid:
    if model is not None:del model;torch.cuda.empty_cache()
    spec=models[newmid];path=R/spec['checkpoint'];assert sha(path)==spec['checkpoint_sha256']
    model=timm.create_model(spec['timm_name'],pretrained=False).eval()
    if spec['format']=='npz':model.load_pretrained(str(path))
    else:
     state=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(state.get('model',state),strict=True)
    for b in model.blocks:b.attn.fused_attn=False
    for param in model.parameters():param.requires_grad_(False)
    model.to(device);mid=newmid
   item=json.loads(payload);item.update(mean=spec['mean'],std=spec['std'])
   image,refs,size=load_inputs(item,device);torch.cuda.reset_peak_memory_stats()
   if args.kind=='e1':data,checks=neuron_scan(model,image,targets[mid])
   else:
    from vitnns.revision_v5_extended import extended_scan
    data,checks=extended_scan(args.kind,model,image,refs,targets[mid],item)
   torch.cuda.synchronize()
   out=ROOT/args.kind;out.mkdir(exist_ok=True)
   dest=out/(tid+'.npz');assert not dest.exists()
   with dest.with_suffix('.npz.tmp').open('xb') as f:np.savez_compressed(f,**data)
   dest.with_suffix('.npz.tmp').rename(dest)
   result=dict(task_id=tid,model_id=mid,image=item['sample_id'],class_id=item['class_id'],image_sha256=item['image_sha256'],output=str(dest.relative_to(R)),output_sha256=sha(dest),seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_reserved(),status='VALID',checks=checks,source_identity=identity)
   write(out/(tid+'.json'),result)
   with sqlite3.connect(db,timeout=60) as c:c.execute("UPDATE tasks SET state='done',end=? WHERE id=? AND worker=?",(time.time(),tid,who))
   completed+=1
   print(json.dumps(dict(task=tid,seconds=result['seconds'],completed=completed)),flush=True)
   if args.limit and completed>=args.limit:break
  except Exception:
   import traceback
   error=traceback.format_exc()
   write(ROOT/'failures'/f'{args.kind}_{tid}_{os.getpid()}.json',dict(error=error,identity=identity))
   with sqlite3.connect(db,timeout=60) as c:c.execute("UPDATE tasks SET state='failed',end=?,error=? WHERE id=? AND worker=?",(time.time(),error,tid,who))
   print(error,flush=True);raise

def run(args):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
 available=[]
 for line in raw.splitlines():
  idx,uuid,mem,util=[v.strip() for v in line.split(',')]
  if float(mem)>=80000 and float(util)<=10:available.append((idx,uuid))
 assert available,'NO_IDLE_GPU'
 write(ROOT/'startup'/f'{args.kind}_launch_{int(time.time())}.json',dict(gpus=raw,existing_processes=pids))
 children=[]
 for idx,uuid in available[:args.gpus]:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=idx,CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
  log=(ROOT/f'{args.kind}_gpu{idx}_{int(time.time())}.log').open('w')
  children.append(subprocess.Popen([sys.executable,__file__,'worker','--kind',args.kind,'--gpu',idx,'--uuid',uuid,'--limit',str(args.limit)],env=env,stdout=log,stderr=subprocess.STDOUT))
 codes=[p.wait() for p in children];print('WORKER_EXIT_CODES',codes);return int(any(codes))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','worker','run','status']);a.add_argument('--kind',choices=['e1','e2','e3','e5'],default='e1');a.add_argument('--gpu',default='');a.add_argument('--uuid',default='');a.add_argument('--gpus',type=int,default=4);a.add_argument('--limit',type=int,default=0);args=a.parse_args()
 if args.mode=='prepare':prepare(args.kind)
 elif args.mode=='worker':worker(args)
 elif args.mode=='run':raise SystemExit(run(args))
 else:
  with sqlite3.connect(ROOT/(args.kind+'.sqlite3')) as c:print(c.execute('SELECT state,COUNT(*) FROM tasks GROUP BY state').fetchall())
