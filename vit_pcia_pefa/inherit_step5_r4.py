#!/usr/bin/env python3
"""Verify and inherit only completed r4 results into the fresh r5 queue."""
from __future__ import annotations
import hashlib,json,os,shutil,sqlite3,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
sys.path[:0]=[str(REPO),str(REPO/'vitnns_rebuild/src')]
from vitnns.endpoint_ep1_queue import create_queue,digest
R4=ROOT/'step5_pilot_r4_20260921';R5=ROOT/'step5_pilot_r5_20260921'
def canon(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def signed(path:Path):
 x=json.loads(path.read_text());d=x.pop('canonical_digest');assert canon(x)==d;x['canonical_digest']=d;return x
def write(path:Path,v:Any):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'w') as f:json.dump(v,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 manifest=signed(R5/'task_manifest.json');receipt=signed(R5/'freeze_receipt.json');assert not receipt['pilot_outcome_access_before_freeze']
 failure=json.loads((R4/'failure_receipt.json').read_text());assert failure['status']=='TERMINAL_INFRASTRUCTURE_FAIL'
 q=R5/'queue.sqlite3';info=create_queue(q,manifest['tasks']);out=[]
 source=sqlite3.connect(R4/'queue.sqlite3');target=sqlite3.connect(q)
 for row in source.execute("select task_id,attempt,result_path,result_digest from tasks where state='completed' order by task_id"):
  task,attempt,relative,result_digest=row;assert task.startswith('deit_s16__') and attempt==1
  src=R4/'results'/relative;doc=signed(src);assert doc['canonical_digest']==result_digest and doc['task_id']==task and doc['result']['status']=='PASS_TASK'
  destination=R5/'results'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,destination);assert signed(destination)['canonical_digest']==result_digest
  target.execute("update tasks set state='completed',attempt=?,result_path=?,result_digest=? where task_id=?",(attempt,relative,result_digest,task))
  target.execute("insert into attempts(task_id,attempt,fencing_token,worker_id,device_uuid,claimed_ns,lease_until_ns,status,completed_ns) values(?,?,?,?,?,?,?,?,?)",(task,attempt,attempt,'INHERITED_R4','INHERITED_R4',0,0,'COMPLETED',0))
  out.append({'task_id':task,'attempt':attempt,'source_path':src.relative_to(REPO).as_posix(),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'canonical_digest':result_digest})
 target.commit();target.close();source.close();assert len(out)==8
 value={'schema_version':1,'run_id':manifest['run_id'],'status':'PASS_INHERITANCE','source_run_id':'step5_pilot_r4_20260921','source_status':'TERMINAL_INFRASTRUCTURE_FAIL','source_failure_receipt_sha256':hashlib.sha256((R4/'failure_receipt.json').read_bytes()).hexdigest(),'inherited_task_ids':[r['task_id'] for r in out],'records':out,'queue_manifest_digest':info['manifest_digest'],'scientific_values_changed':False,'missing_tasks':['vit_b16__%02d'%i for i in range(8)]};value['canonical_digest']=digest(value);write(R5/'inheritance_receipt.json',value)
 print(json.dumps({'status':'PASS_INHERITANCE','inherited':len(out),'missing':8},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
