#!/usr/bin/env python3
"""Seal the interrupted r4 infrastructure failure for prospective recovery."""
from __future__ import annotations
import hashlib,json,os,sqlite3,time
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parent; RUN=ROOT/"step5_1_r4_20260921"
def canonical(value:Any)->bytes:return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def write_new(path:Path,value:dict[str,Any])->None:
 body=dict(value);body["canonical_digest"]=hashlib.sha256(canonical(body)).hexdigest();payload=json.dumps(body,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
 descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(descriptor,"wb") as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
def main()->int:
 snapshot=json.load(open(RUN/"a1_startup_snapshot.json"));start=int(snapshot["created_unix_ns"])
 with sqlite3.connect(RUN/"a1_queue.sqlite3") as connection:
  states=dict(connection.execute("select state,count(*) from tasks group by state").fetchall())
  attempts=connection.execute("select task_id,attempt,status,completed_ns from attempts order by task_id,attempt").fetchall()
 last=max([int(row[3]) for row in attempts if row[3] is not None]+[time.time_ns()])
 # The interactive stop killed all workers before sealing. Counting until seal is conservative.
 elapsed=(last-start)/1e9;selected=len(snapshot["selected_gpus"]);hours=elapsed*selected/3600.0
 write_new(RUN/"a1_failure_receipt.json",{"schema_version":1,"run_id":RUN.name,"status":"TERMINAL_INFRASTRUCTURE_FAIL","reason":"A1 worker read task_id from normalized payload instead of claim envelope","exception":"KeyError: task_id","task_states_at_seal":states,"accepted_result_count":0,"a1_scientific_outcome_available":False,"selected_gpu_count":selected,"conservative_elapsed_seconds_to_seal":elapsed,"conservative_h20_hours":hours,"workers_stopped":True,"evaluation_access":False,"step6_started":False,"completed_unix_ns":time.time_ns()})
 print(json.dumps({"status":"TERMINAL_INFRASTRUCTURE_FAIL","conservative_h20_hours":hours},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
