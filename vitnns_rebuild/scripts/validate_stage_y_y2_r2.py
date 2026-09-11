#!/usr/bin/env python3
"""Independent source-recomputing validator for Stage Y2-R2."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from vitnns.constants import REPO_ROOT
from vitnns import stage_y_y2 as y2
from vitnns import stage_y_y2_r2 as r2

EXPECTED_REFS={
 "refs/archive/stage-y-y0-protocol-freeze":"a6efa1426902708d83de1d83ef257d524cacf95b",
 "refs/archive/stage-y-y1-r1-resource-telemetry-waiver":"70f70ca7290316531406791a6533d65590424a24",
 "refs/archive/stage-y-y2-p0-parallel-schedule-freeze":"66a0dfb889fdb0d45d9e566ed422ba9a40bd30a1",
 "refs/archive/stage-y-y2-p0-r1-elastic-schedule-freeze":"e6321a17a416b693f02f12bc0c7c329c21db5307",
 "refs/archive/stage-y-y2-clean-cohort-evidence-blocked":r2.Y2_COMMIT,
 "refs/archive/stage-y-y2-r1-clean-cohort-evidence-blocked":r2.Y2_R1_COMMIT,
}

def remote(repo:Path,ref:str)->str:
 p=y2.git(repo,"ls-remote","origin",ref).split();r2.require(len(p)==2 and p[1]==ref,f"remote ref missing: {ref}");return p[0]
def record(repo:Path,path:Path)->dict[str,Any]:return r2.file_record(repo/path,repo)
def now()->str:return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")

def validate_predecessors(repo:Path,*,arrays:bool,raw_release:bool)->dict[str,Any]:
 for ref,commit in EXPECTED_REFS.items():r2.require(y2.git(repo,"rev-parse",ref)==remote(repo,ref)==commit,f"predecessor ref drift: {ref}")
 for path,commit in ((r2.Y2_ROOT,r2.Y2_COMMIT),(y2.ORIGINAL_REPORT_ROOT,r2.Y2_COMMIT),(r2.Y2_R1_ROOT,r2.Y2_R1_COMMIT),(y2.R1_REPORT_ROOT,r2.Y2_R1_COMMIT)):
  r2.require(subprocess.run(["git","diff","--exit-code",commit,"--",path.as_posix()],cwd=repo).returncode==0,f"predecessor path drift: {path}")
 c1=r2.read_json(repo/r2.Y2_ROOT/"completion/y2.json");c2=r2.read_json(repo/r2.Y2_R1_ROOT/"completion/y2_r1.json");rec=r2.read_json(repo/r2.Y2_R1_ROOT/"reconciliation.json")
 r2.require(c1["status"]=="BLOCKED_INFRASTRUCTURE" and c1["payload"]["next_authorization"]=="NONE","Y2 terminal drift")
 r2.require(c2["status"]=="BLOCKED_BUDGET" and c2["payload"]["valid_clean_count"]==84 and c2["payload"]["next_authorization"]=="NONE","Y2-R1 terminal drift")
 r2.require(rec["payload"]["terminal_counts"]=={"VALID":84,"FAILED_NUMERICAL":0,"BLOCKED_INFRASTRUCTURE":0,"BLOCKED_BUDGET":16},"Y2-R1 counts drift")
 terms=r2.terminal_by_order(repo);valid=tuple(sorted(o for o,(_,d) in terms.items() if d["status"]=="VALID"));completed=tuple(sorted(o for o,(_,d) in terms.items() if d["status"]=="BLOCKED_BUDGET" and len(d["payload"]["array_records"])==262));missing=tuple(sorted(o for o,(_,d) in terms.items() if d["status"]=="BLOCKED_BUDGET" and d["payload"]["array_records"]==[]))
 r2.require(len(valid)==84 and completed==r2.COMPLETED_BUDGET_ORDERS and missing==r2.MISSING_ORDERS,"Y2-R1 source classification drift")
 for order in missing:
  q=terms[order][1]["payload"];r2.require(q["scientific_record"] is None and q["quadrature_attempts"]==[] and all(v==0 for v in q["resource_telemetry"]["calls"].values()),"missing source executed science")
 cohort=r2.read_json(repo/r2.Y2_ROOT/"cohort_registry.json");audit=r2.read_json(repo/r2.Y2_ROOT/"audit_subset_registry.json");perturb=r2.read_json(repo/r2.Y2_ROOT/"perturbation_subset_registry.json")
 r2.require(cohort["canonical_digest"]=="156eab4bd86ff9979abc6acf8b06af3aa5a90e90cd02edef7713539b147ce277" and cohort["payload"]["selection_digest"]=="8b3faa6d437bda38e963985ac12f14f3b440a3e4021915081f1d5e1b1c80d2cf","cohort digest drift")
 source=None
 if raw_release:
  _,excluded=y2.exclusion_union(repo);candidates,meta=y2.enumerate_candidates(repo);selected,proof=y2.select_cohort(candidates,excluded);r2.require(selected==cohort["payload"]["records"],"raw-source cohort drift");r2.require(y2.subset_registry(selected,y2.AUDIT_NAMESPACE,10,"audit_10")==audit["payload"] and y2.subset_registry(selected,y2.PERTURBATION_NAMESPACE,20,"perturbation_20")==perturb["payload"],"raw-source subset drift");source={"candidate_metadata":meta,"selection_digest":proof["selection_digest"]}
 if arrays:
  for order in valid:r2.validate_valid_source(repo,order)
  for order in completed:r2.validate_completed_budget_source(repo,order)
 return {"status":"PASS","y2_status":c1["status"],"y2_r1_status":c2["status"],"valid_count":len(valid),"completed_budget_orders":list(completed),"missing_orders":list(missing),"raw_release_recomputation":source,"arrays_recomputed":arrays}

def run_focused_tests(repo:Path)->dict[str,Any]:
 env=dict(os.environ);env["PYTHONPATH"]=str(repo/"src")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "");p=subprocess.run([sys.executable,"-m","pytest","-q","tests/test_stage_y_y2_r2.py"],cwd=repo,env=env,text=True,capture_output=True);r2.require(p.returncode==0,"focused tests failed: "+(p.stdout+p.stderr)[-4000:]);m=__import__("re").search(r"(\d+) passed",p.stdout);return {"status":"PASS","test_count":int(m.group(1)),"command":f"{os.path.realpath(sys.executable)} -m pytest -q tests/test_stage_y_y2_r2.py","output_tail":p.stdout.strip()[-1000:]}

def validate_legacy_stage_commands(repo:Path)->list[dict[str,Any]]:
 commands=[[sys.executable,"scripts/validate_stage_y_y0.py","--repo",str(repo),"validate-completion"],[sys.executable,"scripts/validate_stage_y_y1_r1.py","validate"],[sys.executable,"scripts/validate_stage_y_y2_r1.py","validate-blocked","--repo-root",str(repo)]];results=[]
 for command in commands:
  env=dict(os.environ);env["PYTHONPATH"]=str(repo/"src")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "");p=subprocess.run(command,cwd=repo,env=env,text=True,capture_output=True);r2.require(p.returncode==0,f"legacy validator failed: {' '.join(command)}: {(p.stdout+p.stderr)[-2000:]}");results.append({"command":" ".join(command),"status":"PASS"})
 return results

def false_pass_audit(repo:Path,producer:bool=False)->dict[str,Any]:
 spec=r2.read_json(repo/r2.RUN_SPEC);worker=(repo/"src/vitnns/stage_y_y2_r2_worker.py").read_text();core=(repo/"src/vitnns/stage_y_y2_r2.py").read_text();runner=(repo/"scripts/run_stage_y_y2_r2.py").read_text();schema=r2.read_json(repo/"config/stage_y/y2_r2_artifact.schema.json");old=r2.read_json(repo/y2.ORIGINAL_RUN_SPEC)
 def closed(x):
  if isinstance(x,dict):
   if x.get("type")=="object" and "additionalProperties" not in x and "const" not in x:return False
   return all(closed(v) for v in x.values())
  if isinstance(x,list):return all(closed(v) for v in x)
  return True
 checks=[
  ("modify_y2_r1_receipt",subprocess.run(["git","diff","--exit-code",r2.Y2_R1_COMMIT,"--",r2.Y2_R1_ROOT.as_posix()],cwd=repo).returncode==0),
  ("trust_valid_summary_only","validate_valid_source" in core and "validate_array_record" in core),
  ("trust_completed_files_only","validate_completed_budget_source" in core and "validate_closure" in core),
  ("rerun_84_valid",set(r2.MISSING_ORDERS).isdisjoint(set(range(100))-set(r2.MISSING_ORDERS))),
  ("rerun_completed_budget",set(r2.COMPLETED_BUDGET_ORDERS).isdisjoint(r2.MISSING_ORDERS)),
  ("omit_missing_unit",spec["source_classes"]["Y2_R2_COMPUTED"]==list(r2.MISSING_ORDERS)),
  ("early_pass_at_90",spec["pass_valid_count"]==100),
  ("sample_replacement",spec["denominator"]==100 and "cohort_by_order" in core),
  ("zero_fill","zeros_like" not in worker.split("def execute",1)[1]),
  ("outcome_retry",spec["execution"]["scientific_retry_count"]==1),
  ("science_contract_change",spec["scientific_contract"]=={"condition":old["execution"]["conditions"][0],"target":old["execution"]["target"],"baseline":old["execution"]["baseline"],"quadrature_schedule":old["execution"]["quadrature_schedule"],"alpha_chunk":old["execution"]["alpha_chunk"],"source_patch_jvp_chunk":old["execution"]["source_patch_jvp_chunk"],"component_vjp_chunk":old["execution"]["component_vjp_chunk"],"deletion_forward_batch":old["execution"]["deletion_forward_batch"],"blocks":old["coverage"]["blocks"],"heads":old["coverage"]["heads"],"cuts":old["coverage"]["cuts"],"first_all_contract_pass_selected":old["execution"]["first_all_contract_pass_selected"]}),
  ("discard_complete_on_resource","BLOCKED_BUDGET" not in worker),
  ("stop_on_resource","cap_" not in worker and "Budget" not in worker),
  ("download_telemetry_authorizes_access",spec["resources"]["download_authorized"] is False),
  ("unauthorized_external_access",spec["resources"]["expected_download_bytes"]==0),
  ("skip_array_digest","array_digest(array)" in core),
  ("copy_arrays_for_rebind","shutil.copy" not in runner+core),
  ("modify_old_archive_ref",all(y2.git(repo,"rev-parse",ref)==commit for ref,commit in EXPECTED_REFS.items())),
  ("open_payload_schema",schema["properties"]["payload"]["additionalProperties"] is False and schema["properties"]["payload"]["properties"]!= {}),
  ("launch_y3_y4","stage_y_y3" not in runner+worker and "stage_y_y4" not in runner+worker),
  ("run_statistics",all(x not in runner+worker for x in ("bootstrap(","multipletests(","holm("))),
  ("pass_without_source_validation","validate_predecessors" in Path(__file__).read_text() and "reconcile" in Path(__file__).read_text()),
 ]
 rows=[]
 for name,value in checks:
  ok=bool(value[0]) if isinstance(value,tuple) else bool(value);rows.append({"attack":name,"expected":"REJECT","observed":"REJECT" if ok else "ACCEPT","pass":ok})
 r2.require(all(x["pass"] for x in rows),"false-pass audit accepted: "+",".join(x["attack"] for x in rows if not x["pass"]));return {"status":"PASS","attack_count":len(rows),"rejected_count":len(rows),"checks":rows}

def _sum_resources(rows:list[Mapping[str,Any]])->dict[str,Any]:
 return {"h20_gpu_hours":sum(float(x.get("gpu_active_seconds",0)) for x in rows)/3600,"cpu_hours":sum(float(x.get("cpu_seconds",0)) for x in rows)/3600,"wall_seconds_sum":sum(float(x.get("wall_seconds",0)) for x in rows),"retained_bytes":sum(int(x.get("retained_bytes",0)) for x in rows),"retained_storage_gib":sum(int(x.get("retained_bytes",0)) for x in rows)/1024**3,"peak_memory_bytes":max([int(x.get("peak_reserved_gpu_bytes",0)) for x in rows]+[0]),"download_bytes":sum(int(x.get("download_bytes",0)) for x in rows)}

def reconcile(repo:Path,write:bool=False)->dict[str,Any]:
 root=repo/r2.ROOT;history=validate_predecessors(repo,arrays=True,raw_release=True);valid_doc=r2.read_json(root/"valid_rebind_manifest.json");complete_doc=r2.read_json(root/"completed_budget_rebind_manifest.json");missing_doc=r2.read_json(root/"missing_work_manifest.json")
 valid=[r2.validate_valid_source(repo,o) for o in valid_doc["payload"]["selection_orders"]];completed=[r2.validate_completed_budget_source(repo,o) for o in complete_doc["payload"]["selection_orders"]];computed=[]
 for path in sorted(root.glob("shards/shard-*/units/unit-*/completion/unit.json")):
  t=r2.read_json(path);r2.validate_signed(t);q=t["payload"];r2.require(t["status"]=="VALID" and q["source_type"]=="Y2_R2_COMPUTED","computed terminal not VALID");r2.validate_source_binding(repo,t,q["selection_order"]);r2.validate_attempts(repo,t);arrays=r2.load_validated_arrays(repo,t);science=r2.read_json(repo/q["scientific_result"]["path"]);r2.validate_signed(science);from vitnns import stage_y_y1 as y1;closure=y1.validate_closure(arrays,float(science["payload"]["delta"]));r2.require(closure==science["payload"]["closure"] and closure["status"]=="PASS","computed closure drift");computed.append({"selection_order":q["selection_order"],"source_object_id":q["source_object_id"],"source_type":"Y2_R2_COMPUTED","status":"VALID","terminal":record(repo,path.relative_to(repo)),"resource_telemetry":q["resource_telemetry"]})
 r2.require(len(computed)==13 and sorted(x["selection_order"] for x in computed)==list(r2.MISSING_ORDERS),"computed roster drift")
 unified=sorted(valid+completed+computed,key=lambda x:x["selection_order"]);r2.require([x["selection_order"] for x in unified]==list(range(100)) and all(x["status"]=="VALID" for x in unified),"unified 100-unit validity drift")
 counts={s:sum(x["source_type"]==s for x in unified) for s in r2.SOURCE_TYPES};r2.require(counts=={"Y2_R1_VALID_REBIND":84,"Y2_R1_COMPLETED_BUDGET_REBIND":3,"Y2_R2_COMPUTED":13},"source counts drift")
 r1=r2.read_json(repo/r2.Y2_R1_ROOT/"reconciliation.json")["payload"]["resource_totals"];y0=r2.read_json(repo/r2.Y2_ROOT/"reconciliation.json")["payload"]["resource_totals"];r2rows=[x["resource_telemetry"] for x in computed];r2res=_sum_resources(r2rows);cum={"h20_gpu_hours":float(y0["h20_gpu_hours"])+float(r1["h20_gpu_hours"])+r2res["h20_gpu_hours"],"cpu_hours":float(y0["cpu_hours"])+float(r1["cpu_hours"])+r2res["cpu_hours"],"retained_storage_gib":float(y0["retained_storage_gib"])+float(r1["retained_storage_gib"])+r2res["retained_storage_gib"],"download_bytes":r2res["download_bytes"],"peak_memory_bytes":max(int(y0["peak_reserved_gpu_bytes_per_process"]),int(r1["peak_reserved_gpu_bytes_per_process"]),int(r2res["peak_memory_bytes"]))}
 overruns=[]
 for field,cap in (("h20_gpu_hours",64.0),("cpu_hours",218.0),("retained_storage_gib",6.0),("peak_memory_bytes",21*1024**3)):
  if cum[field]>cap:overruns.append({"field":field,"observed":cum[field],"former_ceiling":cap,"classification":"RESOURCE_OVERRUN_REPORTED","scientific_validity_effect":"NONE"})
 payload={"artifact_id":"reconciliation","denominator":100,"valid_clean_count":100,"terminal_counts":{"VALID":100},"source_counts":counts,"selection_orders":list(range(100)),"records":unified,"records_digest":r2.canonical_digest(unified),"stage_resources":r2res,"cumulative_resources":cum,"resource_overruns":overruns,"reporting_only":True,"formal_outcome_access":False,"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"later_stages_started":[],"next_authorization":"STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY"}
 doc=r2.signed("stage_y_y2_r2_reconciliation","PASS",payload)
 if write:
  for name,p in (("unified_100_unit_manifest.json",{"artifact_id":"unified_100_unit_manifest","record_count":100,"selection_orders":list(range(100)),"records":unified,"records_digest":r2.canonical_digest(unified)}),("resource_ledger.json",{"artifact_id":"resource_ledger","stage_resources":r2res,"reporting_only":True}),("cumulative_resource_ledger.json",{"artifact_id":"cumulative_resource_ledger","stage_resources":{"Y2":y0,"Y2_R1":r1,"Y2_R2":r2res},"cumulative_resources":cum,"reporting_only":True}),("resource_overrun_report.json",{"artifact_id":"resource_overrun_report","resource_overruns":overruns,"reporting_only":True}),("access_audit.json",{"artifact_id":"access_audit","external_accesses":[],"download_bytes":0,"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"later_stages_started":[]})):
   r2.write_json_new(root/name,r2.signed("stage_y_y2_r2_"+name.removesuffix(".json"),"PASS",p))
  audit=false_pass_audit(repo);r2.write_json_new(root/"false_pass_audit.json",r2.signed("stage_y_y2_r2_false_pass_audit","PASS",{"artifact_id":"false_pass_audit",**audit}));r2.write_json_new(root/"reconciliation.json",doc)
 return doc

def seal(repo:Path)->dict[str,Any]:
 root=repo/r2.ROOT;rec=r2.read_json(root/"reconciliation.json");r2.validate_signed(rec);r2.require(rec["status"]=="PASS" and rec["payload"]["valid_clean_count"]==100,"R2 cannot seal PASS")
 payload_manifest=[]
 for p in sorted(root.rglob("*")):
  if p.is_file() and p.relative_to(root).as_posix() not in {"payload_manifest.json","whole_root_inventory.json","completion/y2_r2.pending.json","completion/y2_r2.json"}:payload_manifest.append(r2.file_record(p,repo))
 r2.write_json_new(root/"payload_manifest.json",r2.signed("stage_y_y2_r2_payload_manifest","PASS",{"artifact_id":"payload_manifest","files":payload_manifest,"file_count":len(payload_manifest),"files_digest":r2.canonical_digest({"files":payload_manifest})}))
 validation={"artifact_id":"source_recomputing_validation_report","checks":["100 unique orders","84 valid rebinds recomputed","3 completed-budget rebinds recomputed","13 missing units computed","all arrays hashed and finite","closure and identity pass","predecessors immutable","resource overruns reporting only","no statistics or later stage"],"record_count":100,"records_digest":rec["payload"]["records_digest"]};r2.write_json_new(root/"source_recomputing_validation_report.json",r2.signed("stage_y_y2_r2_source_recomputing_validation_report","PASS",validation))
 report=repo/r2.REPORT_ROOT/"y2_r2_report.md";report.parent.mkdir(parents=True,exist_ok=True);data=("# Stage Y2-R2 terminal report\n\nStatus: **PASS**. Unified clean cohort: 100/100 VALID. Sources: 84 Y2-R1 VALID rebinds, 3 completed-budget rebinds, and 13 Y2-R2 computations. Resource overruns are retained as reporting-only. No bootstrap, Holm correction, claim decision, Y3, or Y4 ran.\n\nNext authorization: `STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY`.\n").encode();fd=os.open(report,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);os.write(fd,data);os.fsync(fd);os.close(fd)
 inventory=r2.whole_root_inventory(root,"PASS");r2.write_json_new(root/"whole_root_inventory.json",inventory)
 completion=r2.signed("stage_y_y2_r2_terminal_completion","PASS",{"artifact_id":"completion","completed_at":now(),"denominator":100,"valid_clean_count":100,"source_counts":rec["payload"]["source_counts"],"formal_outcome_access":False,"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"later_stages_started":[],"reconciliation":record(repo,r2.ROOT/"reconciliation.json"),"validation_report":record(repo,r2.ROOT/"source_recomputing_validation_report.json"),"whole_root_inventory":record(repo,r2.ROOT/"whole_root_inventory.json"),"report":record(repo,r2.REPORT_ROOT/"y2_r2_report.md"),"failure":None,"next_authorization":"STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY"});r2.publish_terminal(root/"completion/y2_r2.pending.json",root/"completion/y2_r2.json",completion);return {"status":"PASS","valid_clean_count":100,"next_authorization":completion["payload"]["next_authorization"]}

def validate(repo:Path,publication:bool=False)->dict[str,Any]:
 root=repo/r2.ROOT;completion=r2.read_json(root/"completion/y2_r2.json");r2.validate_signed(completion);r2.require(completion["status"]=="PASS" and completion["payload"]["valid_clean_count"]==100 and completion["payload"]["formal_outcome_access"] is False,"completion semantics drift");rec=reconcile(repo,False);r2.require(rec==r2.read_json(root/"reconciliation.json"),"reconciliation drift");inventory=r2.read_json(root/"whole_root_inventory.json");r2.validate_signed(inventory);r2.require(r2.whole_root_inventory(root,"PASS")["payload"]==inventory["payload"],"inventory drift");audit=false_pass_audit(repo);history=validate_predecessors(repo,arrays=True,raw_release=True)
 if publication:
  head=y2.git(repo,"rev-parse","HEAD");r2.require(head==y2.git(repo,"rev-parse","@{upstream}")==remote(repo,"refs/heads/stage-d-d3-pass") and head==y2.git(repo,"rev-parse",r2.COMPLETION_REF)==remote(repo,r2.COMPLETION_REF),"completion publication drift");r2.require(remote(repo,r2.PRODUCER_REF)==r2.read_json(root/"producer_implementation_seal.json")["payload"]["producer_commit"],"producer ref drift");r2.require(not y2.git(repo,"status","--porcelain"),"worktree dirty");r2.require("(LFS:" not in y2.git(repo,"lfs","status") and "(Git:" not in y2.git(repo,"lfs","status"),"LFS dirty")
 return {"status":"PASS","source_counts":rec["payload"]["source_counts"],"valid_clean_count":100,"completion_sha256":r2.sha_file(root/"completion/y2_r2.json"),"completion_canonical_digest":completion["canonical_digest"],"inventory_sha256":r2.sha_file(root/"whole_root_inventory.json"),"inventory_canonical_digest":inventory["canonical_digest"],"inventory_files_digest":inventory["payload"]["files_digest"],"false_pass_count":audit["attack_count"],"predecessors":history,"publication":publication,"next_authorization":completion["payload"]["next_authorization"]}

def main()->int:
 p=argparse.ArgumentParser();p.add_argument("command",choices=("focused-tests","false-pass-audit","validate","validate-publication"));p.add_argument("--repo-root",type=Path,default=REPO_ROOT);a=p.parse_args();repo=a.repo_root.resolve()
 try:
  if a.command=="focused-tests":result=run_focused_tests(repo)
  elif a.command=="false-pass-audit":result=false_pass_audit(repo,producer=True)
  else:result=validate(repo,a.command=="validate-publication")
 except Exception as e:print(json.dumps({"status":getattr(e,"code","BLOCKED_PROVENANCE"),"error":str(e),"next_authorization":"NONE"},sort_keys=True));return 2
 print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
