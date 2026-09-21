#!/usr/bin/env python3
"""Bind and copy r4's inherited A0 and C1 evidence into r5."""
from __future__ import annotations
import hashlib,json,os,shutil,time
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT/"step5_1_r4_20260921";TARGET=ROOT/"step5_1_r5_20260921"
NAMES=("a0_fixture_result.json","a0_startup_snapshot.json","a0_launch_terminal.json","a0/deit_s16.json","a0/vit_b16.json","a0_decision.json","c1_result.json")
def canonical(value:Any)->bytes:return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def verify(path:Path)->dict[str,Any]:
 value=json.loads(path.read_text());claimed=value.pop("canonical_digest")
 if hashlib.sha256(canonical(value)).hexdigest()!=claimed:raise RuntimeError("CANONICAL_DIGEST:"+str(path))
 value["canonical_digest"]=claimed;return value
def sha256(path:Path)->str:
 digest=hashlib.sha256()
 with path.open("rb") as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
 return digest.hexdigest()
def write_new(path:Path,value:dict[str,Any])->None:
 body=dict(value);body["canonical_digest"]=hashlib.sha256(canonical(body)).hexdigest();descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(descriptor,"wb") as stream:stream.write(json.dumps(body,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n");stream.flush();os.fsync(stream.fileno())
def main()->int:
 protocol=verify(TARGET/"protocol.json");failure=verify(SOURCE/"a1_failure_receipt.json")
 if protocol["run_id"]!=TARGET.name or failure["accepted_result_count"]!=0 or failure["a1_scientific_outcome_available"]:raise RuntimeError("R4_NOT_INHERITABLE")
 records=[]
 for name in NAMES:
  source=SOURCE/name;verify(source);destination=TARGET/name;destination.parent.mkdir(parents=True,exist_ok=True)
  if destination.exists():raise RuntimeError("INHERIT_DESTINATION_EXISTS:"+name)
  shutil.copyfile(source,destination)
  if sha256(source)!=sha256(destination):raise RuntimeError("INHERIT_COPY_DIGEST:"+name)
  records.append({"path":name,"bytes":destination.stat().st_size,"sha256":sha256(destination)})
 write_new(TARGET/"a0_inheritance_receipt.json",{"schema_version":1,"run_id":TARGET.name,"status":"PASS_A0_C1_INHERITANCE","source_root":SOURCE.name,"r4_failure_digest":failure["canonical_digest"],"artifacts":records,"a0_recomputed":False,"c1_recomputed":False,"completed_unix_ns":time.time_ns(),"evaluation_access":False,"step6_started":False})
 print(json.dumps({"status":"PASS_A0_C1_INHERITANCE","artifact_count":len(records)},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
