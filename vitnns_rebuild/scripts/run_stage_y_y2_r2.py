#!/usr/bin/env python3
"""Orchestrate producer freeze, rebinding, missing-only compute, and Y2-R2 seal."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from vitnns.constants import REPO_ROOT
from vitnns import stage_y_y2 as y2
from vitnns import stage_y_y2_r2 as r2

SOURCE_FILES=(Path("docs/stage_y_y2_r2_resource_reporting_recovery_amendment_20260912.md"),Path("config/stage_y/y2_r2_run_spec.json"),Path("config/stage_y/y2_r2_run_spec.schema.json"),Path("config/stage_y/y2_r2_artifact.schema.json"),Path("src/vitnns/stage_y_y2_r2.py"),Path("src/vitnns/stage_y_y2_r2_worker.py"),Path("scripts/run_stage_y_y2_r2.py"),Path("scripts/validate_stage_y_y2_r2.py"),Path("tests/test_stage_y_y2_r2.py"))
IMPLEMENTATION_DEPENDENCIES=(Path("src/vitnns/stage_y_y2.py"),Path("src/vitnns/stage_y_y2_worker.py"),Path("src/vitnns/stage_y_y1.py"),Path("src/vitnns/stage_y_y1_worker.py"),Path("src/vitnns/stage_x_nns_v2.py"),Path("src/vitnns/modeling.py"),Path("scripts/run_stage_y_y2.py"),Path("config/stage_y/y2_run_spec.json"))
EXPECTED_REFS={
 "refs/archive/stage-y-y0-protocol-freeze":"a6efa1426902708d83de1d83ef257d524cacf95b",
 "refs/archive/stage-y-y1-r1-resource-telemetry-waiver":"70f70ca7290316531406791a6533d65590424a24",
 "refs/archive/stage-y-y2-p0-parallel-schedule-freeze":"66a0dfb889fdb0d45d9e566ed422ba9a40bd30a1",
 "refs/archive/stage-y-y2-p0-r1-elastic-schedule-freeze":"e6321a17a416b693f02f12bc0c7c329c21db5307",
 "refs/archive/stage-y-y2-clean-cohort-evidence-blocked":r2.Y2_COMMIT,
 "refs/archive/stage-y-y2-r1-clean-cohort-evidence-blocked":r2.Y2_R1_COMMIT,
}

def now()->str:return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
def remote(repo:Path,ref:str)->str:
 p=y2.git(repo,"ls-remote","origin",ref).split();r2.require(len(p)==2 and p[1]==ref,f"remote ref missing: {ref}",r2.StartupError);return p[0]
def record(repo:Path,path:Path)->dict[str,Any]:return r2.file_record(repo/path,repo)
def validator(repo:Path):
 s=importlib.util.spec_from_file_location("validate_y2_r2",repo/"scripts/validate_stage_y_y2_r2.py");r2.require(s is not None and s.loader is not None,"validator unavailable");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def startup(repo:Path,allow_sources:bool=False)->dict[str,Any]:
 head=y2.git(repo,"rev-parse","HEAD");tracking=y2.git(repo,"rev-parse","@{upstream}");branch=y2.git(repo,"symbolic-ref","--short","HEAD");remote_branch=remote(repo,"refs/heads/stage-d-d3-pass")
 r2.require(branch=="stage-d-d3-pass" and head==tracking==remote_branch==r2.Y2_R1_COMMIT,"startup branch state drift",r2.StartupError)
 for ref,commit in EXPECTED_REFS.items():r2.require(y2.git(repo,"rev-parse",ref)==remote(repo,ref)==commit,f"archive ref drift: {ref}",r2.StartupError)
 status=y2.git(repo,"status","--porcelain");prefix=repo.name+"/";paths=[]
 for line in status.splitlines():
  path=line[2:].lstrip().split(" -> ")[-1];paths.append(path[len(prefix):] if path.startswith(prefix) else path)
 if allow_sources:r2.require(all(Path(path) in SOURCE_FILES for path in paths),f"unexplained worktree paths: {paths}",r2.StartupError)
 else:r2.require(not status,"worktree/index not clean",r2.StartupError)
 lfs=y2.git(repo,"lfs","status");
 if not allow_sources:r2.require("(LFS:" not in lfs and "(Git:" not in lfs,"Git LFS not clean",r2.StartupError)
 for ref in (r2.PRODUCER_REF,r2.COMPLETION_REF):r2.require(subprocess.run(["git","show-ref","--verify","--quiet",ref],cwd=repo).returncode!=0,f"R2 ref exists: {ref}",r2.StartupError)
 for path,commit in ((r2.Y2_ROOT,r2.Y2_COMMIT),(y2.ORIGINAL_REPORT_ROOT,r2.Y2_COMMIT),(r2.Y2_R1_ROOT,r2.Y2_R1_COMMIT),(y2.R1_REPORT_ROOT,r2.Y2_R1_COMMIT)):
  r2.require(subprocess.run(["git","diff","--exit-code",commit,"--",path.as_posix()],cwd=repo).returncode==0,f"immutable path drift: {path}",r2.StartupError)
 r2.require(not (repo/r2.ROOT).exists() and not (repo/r2.REPORT_ROOT).exists(),"R2 root exists",r2.StartupError)
 ps=subprocess.run(["ps","-eo","pid=,args="],text=True,capture_output=True,check=True);stale=[line.strip() for line in ps.stdout.splitlines() if re.search(r"stage_y_y2(?:_r[12])?_worker|run_stage_y_y2(?:_r2)?\.py\s+(run-shard|run-all)",line) and str(os.getpid()) not in line]
 r2.require(not stale,"residual Y2 worker exists",r2.StartupError)
 return {"artifact_id":"startup_validation","branch":branch,"head":head,"tracking":tracking,"remote_branch":remote_branch,"refs":EXPECTED_REFS,"worktree_clean":not bool(status),"lfs_clean":"(LFS:" not in lfs and "(Git:" not in lfs,"r2_refs_absent":True,"r2_roots_absent":True,"residual_workers":[]}

def producer_preflight(repo:Path)->dict[str,Any]:
 gate=startup(repo,True);v=validator(repo);tests=v.run_focused_tests(repo);audit=v.false_pass_audit(repo,producer=True);legacy=v.validate_legacy_stage_commands(repo);history=v.validate_predecessors(repo,arrays=False,raw_release=True)
 return {"status":"PASS","startup":gate,"focused_tests":tests,"false_pass_audit":audit,"legacy_validators":legacy,"predecessor_validation":history,"producer_files":[record(repo,p) for p in SOURCE_FILES]}

def write_producer_seal(repo:Path,commit:str)->dict[str,Any]:
 root=repo/r2.ROOT;r2.require(not root.exists(),"R2 root already exists");root.joinpath("completion").mkdir(parents=True);(repo/r2.REPORT_ROOT).mkdir(parents=True)
 r2.write_json_new(root/"completion/y2_r2.pending.json",r2.signed("stage_y_y2_r2_pending_completion","PENDING",{"artifact_id":"y2_r2_pending","started_at":now(),"terminal_published":False,"next_authorization":"NONE_WHILE_PENDING"}))
 files=[record(repo,p) for p in SOURCE_FILES+IMPLEMENTATION_DEPENDENCIES];payload={"artifact_id":"producer_implementation_seal","producer_commit":commit,"producer_ref":r2.PRODUCER_REF,"source_files":files,"files_digest":r2.canonical_digest({"files":files})}
 r2.write_json_new(root/"producer_implementation_seal.json",r2.signed("stage_y_y2_r2_producer_implementation_seal","PASS",payload))
 r2.write_json_new(root/"prospective_amendment_binding.json",r2.signed("stage_y_y2_r2_prospective_amendment_binding","PASS",{"artifact_id":"prospective_amendment_binding","source_files":[record(repo,Path("docs/stage_y_y2_r2_resource_reporting_recovery_amendment_20260912.md")),record(repo,r2.RUN_SPEC)],"files_digest":r2.canonical_digest([record(repo,Path("docs/stage_y_y2_r2_resource_reporting_recovery_amendment_20260912.md")),record(repo,r2.RUN_SPEC)])}))
 r2.write_json_new(root/"original_y2_immutable_binding.json",r2.signed("stage_y_y2_r2_original_y2_immutable_binding","PASS",{"artifact_id":"original_y2_immutable_binding","producer_commit":r2.Y2_COMMIT,"producer_ref":"refs/archive/stage-y-y2-clean-cohort-evidence-blocked","source_files":[record(repo,r2.Y2_ROOT/"completion/y2.json"),record(repo,r2.Y2_ROOT/"whole_root_inventory.json")],"files_digest":r2.canonical_digest([record(repo,r2.Y2_ROOT/"completion/y2.json"),record(repo,r2.Y2_ROOT/"whole_root_inventory.json")])}))
 r2.write_json_new(root/"y2_r1_immutable_binding.json",r2.signed("stage_y_y2_r2_y2_r1_immutable_binding","PASS",{"artifact_id":"y2_r1_immutable_binding","producer_commit":r2.Y2_R1_COMMIT,"producer_ref":"refs/archive/stage-y-y2-r1-clean-cohort-evidence-blocked","source_files":[record(repo,r2.Y2_R1_ROOT/"completion/y2_r1.json"),record(repo,r2.Y2_R1_ROOT/"whole_root_inventory.json")],"files_digest":r2.canonical_digest([record(repo,r2.Y2_R1_ROOT/"completion/y2_r1.json"),record(repo,r2.Y2_R1_ROOT/"whole_root_inventory.json")])}))
 return payload

def build_rebinds(repo:Path)->dict[str,Any]:
 root=repo/r2.ROOT;r2.require((root/"producer_implementation_seal.json").is_file(),"producer seal absent")
 old=r2.terminal_by_order(repo);valid_orders=tuple(sorted(o for o,(_,d) in old.items() if d["status"]=="VALID"));r2.require(len(valid_orders)==84,"Y2-R1 VALID count drift")
 valid=[r2.validate_valid_source(repo,o) for o in valid_orders]
 completed=[r2.validate_completed_budget_source(repo,o) for o in r2.COMPLETED_BUDGET_ORDERS]
 missing=r2.validate_missing_sources(repo)
 for name,source_type,records in (("valid_rebind_manifest.json","Y2_R1_VALID_REBIND",valid),("completed_budget_rebind_manifest.json","Y2_R1_COMPLETED_BUDGET_REBIND",completed),("missing_work_manifest.json","Y2_R2_COMPUTED",missing)):
  payload={"artifact_id":name.removesuffix(".json"),"source_type":source_type,"record_count":len(records),"selection_orders":[x["selection_order"] for x in records],"records":records,"records_digest":r2.canonical_digest(records)};r2.write_json_new(root/name,r2.signed("stage_y_y2_r2_"+name.removesuffix(".json"),"PASS",payload))
 return {"status":"PASS","valid_rebind":84,"completed_budget_rebind":3,"missing":13}

def _gpu_rows(repo:Path)->list[dict[str,Any]]:
 out=subprocess.run(["nvidia-smi","--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,driver_version","--format=csv,noheader,nounits"],cwd=repo,text=True,capture_output=True,check=True).stdout;rows=[]
 for line in out.splitlines():
  x=[v.strip() for v in line.split(",")];rows.append({"physical_index":int(x[0]),"device_name":x[1],"gpu_uuid":x[2],"pci_bus_id":x[3],"total_memory_mib":int(x[4]),"free_memory_mib":int(x[5]),"driver":x[6]})
 busy_out=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],cwd=repo,text=True,capture_output=True,check=True).stdout;busy={x.split(",",1)[0].strip() for x in busy_out.splitlines() if x.strip() and not x.startswith("No running")};return y2.select_idle_h20_devices(rows,busy)

def validate_startup(repo:Path)->dict[str,Any]:
 root=repo/r2.ROOT;r2.require((root/"missing_work_manifest.json").is_file(),"rebind/missing validation absent");rows=_gpu_rows(repo);checkpoint=y2.sha_file(repo/"assets/models/vit_base_patch16_224_weights.semantic_candidate.pth");probe=[]
 launch=importlib.util.spec_from_file_location("y2_probe",repo/"scripts/run_stage_y_y2.py");m=importlib.util.module_from_spec(launch);launch.loader.exec_module(m);m.y2.activate_recovery_contract();probed=m._probe_devices(repo,rows,checkpoint)
 for x in probed: probe.append({**x,"environment_digest":x["environment_digest"]})
 gate=r2.signed("stage_y_y2_r2_startup_gate","PASS",{"artifact_id":"startup_gate","locked_worker_count":len(probe),"devices":probe,"python_executable_realpath":os.path.realpath(sys.executable),"environment_digest":probe[0]["environment_digest"],"numeric_probe_digest":probe[0]["numeric_probe_digest"],"dynamic_rebalancing":False})
 r2.write_json_new(root/"startup_gate.json",gate);return {"status":"PASS","locked_worker_count":len(probe),"physical_indices":[x["physical_index"] for x in probe]}

def build_shards(repo:Path)->dict[str,Any]:
 root=repo/r2.ROOT;gate=r2.read_json(root/"startup_gate.json");k=gate["payload"]["locked_worker_count"];cohort=r2.cohort_by_order(repo)
 for shard in range(k):
  orders=[o for i,o in enumerate(r2.MISSING_ORDERS) if i%k==shard];payload={"artifact_id":f"shard-{shard:02d}-manifest","shard_id":shard,"unit_count":len(orders),"selection_orders":orders,"source_type":"Y2_R2_COMPUTED","mapping_rule":"missing_roster_zero_based_index mod locked_worker_count","dynamic_rebalancing":False,"records":[{"selection_order":o,"source_object_id":cohort[o]["source_object_id"],"source_type":"Y2_R2_COMPUTED","status":"PENDING"} for o in orders]};r2.write_json_new(root/"shards"/f"shard-{shard:02d}"/"shard_manifest.json",r2.signed("stage_y_y2_r2_shard_manifest","PASS",payload))
 return {"status":"PASS","shard_counts":[len([o for i,o in enumerate(r2.MISSING_ORDERS) if i%k==s]) for s in range(k)]}

def run_all(repo:Path)->dict[str,Any]:
 root=repo/r2.ROOT;gate=r2.read_json(root/"startup_gate.json");k=gate["payload"]["locked_worker_count"];processes=[]
 for shard in range(k):
  sr=root/"shards"/f"shard-{shard:02d}";r2.write_json_new(sr/"completion/shard.pending.json",r2.signed("stage_y_y2_r2_shard_pending","PENDING",{"artifact_id":f"shard-{shard:02d}","shard_id":shard,"terminal_published":False,"next_authorization":"NONE_WHILE_PENDING"}));env=dict(os.environ);env["CUDA_VISIBLE_DEVICES"]=str(gate["payload"]["devices"][shard]["physical_index"]);env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8";env["STAGE_Y2_R2_LAUNCH_AUTHORIZATION"]=gate["canonical_digest"];env["PYTHONDONTWRITEBYTECODE"]="1";env["PYTHONPATH"]=str(repo/"src")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "");p=subprocess.Popen([sys.executable,"-m","vitnns.stage_y_y2_r2_worker","--repo-root",str(repo),"--shard-id",str(shard)],cwd=repo,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);processes.append((shard,p))
 results=[]
 for shard,p in processes:
  out,err=p.communicate();results.append({"shard_id":shard,"pid":p.pid,"returncode":p.returncode,"stdout":out.strip(),"stderr":err.strip()})
 r2.require(all(x["returncode"]==0 for x in results),f"R2 worker failure: {results}",r2.InfrastructureError);return {"status":"PASS","results":results}

def reconcile(repo:Path)->dict[str,Any]:return validator(repo).reconcile(repo,write=True)
def seal(repo:Path)->dict[str,Any]:return validator(repo).seal(repo)
def validate(repo:Path,publication:bool=False)->dict[str,Any]:return validator(repo).validate(repo,publication=publication)

def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument("command",choices=("producer-preflight","write-producer-seal","build-rebinds","validate-startup","build-shards","run-all","reconcile","seal","validate","validate-publication"));p.add_argument("--repo-root",type=Path,default=REPO_ROOT);p.add_argument("--producer-commit");a=p.parse_args(argv);repo=a.repo_root.resolve()
 try:
  if a.command=="producer-preflight":result=producer_preflight(repo)
  elif a.command=="write-producer-seal":r2.require(a.producer_commit is not None,"producer commit required");result=write_producer_seal(repo,a.producer_commit)
  elif a.command=="build-rebinds":result=build_rebinds(repo)
  elif a.command=="validate-startup":result=validate_startup(repo)
  elif a.command=="build-shards":result=build_shards(repo)
  elif a.command=="run-all":result=run_all(repo)
  elif a.command=="reconcile":result=reconcile(repo)
  elif a.command=="seal":result=seal(repo)
  else:result=validate(repo,a.command=="validate-publication")
 except Exception as error:print(json.dumps({"status":getattr(error,"code","BLOCKED_PROVENANCE"),"error":str(error),"next_authorization":"NONE"},sort_keys=True));return 2
 print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
