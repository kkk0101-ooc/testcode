#!/usr/bin/env python3
"""Publish, validate, and execute the prospective missing-only recovery."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from vitnns.constants import REPO_ROOT
from vitnns import stage_y_y2 as y2
from vitnns import stage_y_y2_r2 as r2


def now(): return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
def remote(repo,ref):
    rows=y2.git(repo,"ls-remote","origin",ref).split()
    r2.require(len(rows)==2 and rows[1]==ref,"remote ref absent: "+ref,r2.StartupError)
    return rows[0]

def validator(repo):
    spec=importlib.util.spec_from_file_location("validate_y2_r2",repo/"scripts/validate_stage_y_y2_r2.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def startup(repo,allow_sources=False):
    r2.require(os.path.realpath(sys.executable)==os.path.realpath(r2.PYTHON),"launcher Python drift",r2.StartupError)
    r2.require(y2.git(repo,"symbolic-ref","--short","HEAD")=="stage-d-d3-pass","branch drift",r2.StartupError)
    r2.require(y2.git(repo,"rev-parse","HEAD")==y2.git(repo,"rev-parse","@{upstream}")==remote(repo,"refs/heads/stage-d-d3-pass")==r2.STARTUP_COMMIT,"startup HEAD drift",r2.StartupError)
    validator(repo).validate_refs(repo,remote_check=True)
    status=y2.git(repo,"status","--porcelain")
    paths=[line[2:].lstrip().removeprefix(repo.name+"/") for line in status.splitlines()]
    r2.require((not paths) or (allow_sources and all(Path(p) in r2.SOURCE_FILES for p in paths)),"unexpected worktree/index change: "+str(paths),r2.StartupError)
    for path,commit in r2.PROTECTED:
        r2.require(subprocess.run(["git","diff","--exit-code",commit,"--",str(path)],cwd=repo).returncode==0,"immutable predecessor drift",r2.StartupError)
    for ref in (r2.PRODUCER_REF,r2.COMPLETION_REF,r2.BLOCKED_REF):
        r2.require(subprocess.run(["git","show-ref","--verify","--quiet",ref],cwd=repo).returncode!=0 and not y2.git(repo,"ls-remote","origin",ref),"new recovery ref exists",r2.StartupError)
    r2.require(not (repo/r2.ROOT).exists() and not (repo/r2.REPORT_ROOT).exists(),"new recovery root exists",r2.StartupError)
    if not allow_sources:
        lfs=y2.git(repo,"lfs","status");r2.require("(LFS:" not in lfs and "(Git:" not in lfs,"LFS pending",r2.StartupError)
    validator(repo).validate_no_later_stages(repo)
    import re
    processes=subprocess.check_output(["ps","-eo","comm=,args="],text=True)
    stale=[line for line in processes.splitlines() if line.lstrip().startswith(("python ","python3 ","python3.10 ")) and re.search(r"(?:-m vitnns.stage_y_y2.*worker|run_stage_y_y2[^ ]*\.py (?:run-all|run-shard))",line)]
    r2.require(not stale,"residual worker exists",r2.StartupError)


def producer_preflight(repo):
    started=time.monotonic();cpu=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN);child_cpu=children.ru_utime+children.ru_stime
    startup(repo,True);v=validator(repo)
    v.validate_schemas(repo);v.validate_predecessors(repo,raw_release=True,arrays=False)
    legacy=v.validate_legacy_stage_commands(repo)
    tests=v.run_focused_tests(repo);audit=v.false_pass_audit(repo)
    import py_compile,tempfile
    with tempfile.TemporaryDirectory(prefix="y2-r2-r1-compile-") as tmp:
        for i,path in enumerate(r2.SOURCE_FILES):
            if path.suffix==".py":py_compile.compile(str(repo/path),cfile=f"{tmp}/{i}.pyc",doraise=True)
    return r2.signed("preflight","PASS",{"checks":["startup Git/LFS/refs","closed schemas","raw-source cohort/subsets","predecessor immutable metadata",*legacy,"focused synthetic pipeline","false-pass mutations","py_compile"],"focused_test_count":tests,"false_pass_count":audit["attack_count"],"cohort_digest":r2.COHORT_DIGEST,"selection_digest":r2.SELECTION_DIGEST,"startup_commit":r2.STARTUP_COMMIT,"resource_telemetry":{"cpu_seconds":time.process_time()-cpu+resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime-child_cpu,"wall_seconds":time.monotonic()-started,"gpu_active_seconds":0,"peak_reserved_gpu_bytes":0,"retained_bytes":0,"download_bytes":0}})


def write_producer_seal(repo,commit,preflight_file):
    r2.require(not (repo/r2.ROOT).exists(),"new root exists")
    r2.require(y2.git(repo,"rev-parse","HEAD")==y2.git(repo,"rev-parse","@{upstream}")==remote(repo,"refs/heads/stage-d-d3-pass")==remote(repo,r2.PRODUCER_REF)==y2.git(repo,"rev-parse",r2.PRODUCER_REF)==commit,"producer publication not closed",r2.StartupError)
    r2.require(not y2.git(repo,"status","--porcelain"),"producer worktree dirty",r2.StartupError)
    preflight=r2.read_json(preflight_file);r2.validate_signed(preflight);r2.require(preflight["status"]=="PASS","preflight not PASS")
    v=validator(repo);v.validate_schemas(repo);v.validate_refs(repo,True)
    root=repo/r2.ROOT
    r2.write_json_new(root/"completion/y2_r2_r1.pending.json",r2.signed("pending","PENDING",{"started_at":now(),"started_unix_ns":time.time_ns(),"next_authorization":"NONE_WHILE_PENDING"}))
    files=[r2.file_record(repo/p,repo) for p in r2.SOURCE_FILES+r2.DEPENDENCIES]
    r2.write_json_new(root/"producer_implementation_seal.json",r2.signed("binding","PASS",{"producer_commit":commit,"producer_ref":r2.PRODUCER_REF,"source_files":files,"files_digest":r2.canonical_digest({"files":files})}))
    r2.write_json_new(root/"producer_preflight.json",preflight)
    for name,revision,ref,paths in (
        ("prospective_amendment_binding",commit,r2.PRODUCER_REF,(r2.AMENDMENT,r2.RUN_SPEC)),
        ("original_y2_immutable_binding",r2.Y2_COMMIT,"refs/archive/stage-y-y2-clean-cohort-evidence-blocked",(r2.Y2_ROOT/"completion/y2.json",r2.Y2_ROOT/"whole_root_inventory.json")),
        ("y2_r1_immutable_binding",r2.Y2_R1_COMMIT,"refs/archive/stage-y-y2-r1-clean-cohort-evidence-blocked",(r2.Y2_R1_ROOT/"completion/y2_r1.json",r2.Y2_R1_ROOT/"whole_root_inventory.json")),
        ("y2_r2_immutable_binding",r2.STARTUP_COMMIT,"refs/archive/stage-y-y2-r2-resource-reporting-recovery-blocked",(r2.Y2_R2_ROOT/"completion/y2_r2.json",r2.Y2_R2_ROOT/"whole_root_inventory.json"))):
        records=[r2.file_record(repo/p,repo) for p in paths]
        r2.write_json_new(root/(name+".json"),r2.signed("binding","PASS",{"producer_commit":revision,"producer_ref":ref,"source_files":records,"files_digest":r2.canonical_digest({"files":records})}))
    r2.validate_implementation(repo)
    return {"status":"PASS","producer_commit":commit}


def build_rebinds(repo):
    r2.validate_implementation(repo);start=time.monotonic();cpu=time.process_time();root=repo/r2.ROOT
    validator(repo).validate_predecessors(repo,raw_release=True,arrays=False)
    terms=r2.terminal_by_order(repo);orders=sorted(o for o,(_,d) in terms.items() if d["status"]=="VALID")
    r2.require(len(orders)==84,"original VALID roster drift")
    valid=[];completed=[]
    for roster,output,fn in ((orders,valid,r2.validate_valid_source),(r2.COMPLETED_BUDGET_ORDERS,completed,r2.validate_completed_budget_source)):
        for order in roster:
            record=fn(repo,order);output.append(record)
            pending=root/f"rebind_units/unit-{order:03d}/completion/unit.pending.json"
            r2.write_json_new(pending,r2.signed("unit_pending","PENDING",{"selection_order":order,"source_object_id":record["source_object_id"],"attempt_id":"source-validation-001","retry_index":1}))
            r2.publish_terminal(pending,pending.with_name("unit.json"),r2.signed("rebind_record","VALID",record))
            print(json.dumps({"event":"rebind_validated","selection_order":order,"source_type":record["source_type"]}),flush=True)
    for name,kind,records in (("valid_rebind_manifest",r2.SOURCE_TYPES[0],valid),("completed_budget_rebind_manifest",r2.SOURCE_TYPES[1],completed)):
        r2.write_json_new(root/(name+".json"),r2.signed("rebind_manifest","PASS",{"source_type":kind,"record_count":len(records),"selection_orders":[r["selection_order"] for r in records],"records":records,"records_digest":r2.canonical_digest(records)}))
    missing=r2.validate_missing_sources(repo)
    r2.write_json_new(root/"missing_work_manifest.json",r2.signed("missing_manifest","PASS",{"record_count":13,"selection_orders":list(r2.MISSING_ORDERS),"records":missing,"records_digest":r2.canonical_digest(missing)}))
    r2.write_json_new(root/"rebind_validation_resource.json",r2.signed("verification_resource","PASS",{"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.monotonic()-start,"gpu_active_seconds":0,"peak_reserved_gpu_bytes":0,"retained_bytes":0,"download_bytes":0}))
    return {"status":"PASS","source_counts":[84,3],"missing_count":13}


def _gpu_rows(repo):
    spec=importlib.util.spec_from_file_location("frozen_y2_launcher",repo/"scripts/run_stage_y_y2.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,module._eligible_gpu_rows(repo)


def validate_startup(repo):
    r2.validate_implementation(repo);root=repo/r2.ROOT
    r2.require((root/"missing_work_manifest.json").is_file(),"rebind validation absent")
    r2.require(not (root/"startup_gate.json").exists(),"GPU roster already frozen")
    module,rows=_gpu_rows(repo);module.y2.activate_recovery_contract()
    r2.require(r2.sha_file(repo/"assets/models/vit_base_patch16_224_weights.semantic_candidate.pth")==r2.CHECKPOINT_SHA,"checkpoint drift")
    rows=module._probe_devices(repo,rows,r2.CHECKPOINT_SHA)
    gate=r2.signed("startup_gate","PASS",{"locked_worker_count":len(rows),"devices":rows,"python_executable_realpath":os.path.realpath(sys.executable),"environment_digest":rows[0]["environment_digest"],"numeric_probe_digest":rows[0]["numeric_probe_digest"],"dynamic_rebalancing":False})
    r2.write_json_new(root/"startup_gate.json",gate)
    return {"status":"PASS","physical_indices":[d["physical_index"] for d in rows]}


def build_shards(repo):
    r2.validate_implementation(repo);root=repo/r2.ROOT;gate=r2.read_json(root/"startup_gate.json");r2.validate_signed(gate)
    k=gate["payload"]["locked_worker_count"]
    for shard in range(k):
        orders=r2.shard_orders(shard,k)
        r2.write_json_new(root/f"shards/shard-{shard:02d}/shard_manifest.json",r2.signed("shard_manifest","PASS",{"shard_id":shard,"unit_count":len(orders),"selection_orders":orders,"mapping_rule":"missing_roster_zero_based_index mod locked_worker_count","dynamic_rebalancing":False,"source_type":"Y2_R2_COMPUTED"}))
    return {"status":"PASS","shard_counts":[len(r2.shard_orders(s,k)) for s in range(k)]}


def worker_env(repo,gate,shard):
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=str(gate["payload"]["devices"][shard]["physical_index"]),CUBLAS_WORKSPACE_CONFIG=":4096:8",STAGE_Y2_R2_LAUNCH_AUTHORIZATION=gate["canonical_digest"],PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(repo/"src"),PYTHONUNBUFFERED="1")
    return env


def run_all(repo):
    r2.validate_implementation(repo);root=repo/r2.ROOT;gate=r2.read_json(root/"startup_gate.json");r2.validate_signed(gate)
    validator(repo).validate_shard_manifests(repo)
    _,available=_gpu_rows(repo)
    r2.require({d["gpu_uuid"] for d in gate["payload"]["devices"]}<={d["gpu_uuid"] for d in available},"locked GPU became busy",r2.StartupError)
    r2.write_json_new(root/"execution_start.json",r2.signed("pending","PASS",r2.read_json(root/"completion/y2_r2_r1.pending.json")["payload"]))
    processes=[];logs=[]
    for shard in range(gate["payload"]["locked_worker_count"]):
        sr=root/f"shards/shard-{shard:02d}"
        r2.write_json_new(sr/"completion/shard.pending.json",r2.signed("shard_pending","PENDING",{"shard_id":shard,"started_unix_ns":time.time_ns()}))
        log=(sr/"worker.log").open("x");logs.append(log)
        process=subprocess.Popen([str(r2.PYTHON),"-m","vitnns.stage_y_y2_r2_worker","--repo-root",str(repo),"--shard-id",str(shard)],cwd=repo,env=worker_env(repo,gate,shard),stdout=log,stderr=subprocess.STDOUT)
        processes.append((shard,process));print(json.dumps({"event":"worker_started","shard_id":shard,"pid":process.pid}),flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(processes)) as pool:
            futures={pool.submit(p.wait):(s,p) for s,p in processes}
            for future in concurrent.futures.as_completed(futures):
                shard,process=futures[future];rc=future.result()
                if rc:
                    for _,other in processes:
                        if other.poll() is None: other.terminate()
                    failed=[r2.read_json(path) for path in (root/f"shards/shard-{shard:02d}").glob("units/unit-*/completion/unit.json") if r2.read_json(path)["status"]!="VALID"]
                    code=failed[-1]["status"] if failed else "BLOCKED_INFRASTRUCTURE"
                    error_type={"FAILED_NUMERICAL":r2.NumericalError,"BLOCKED_SOURCE_DRIFT":r2.SourceDriftError,"BLOCKED_ACCESS":r2.AccessError,"BLOCKED_PROVENANCE":r2.R2Error}.get(code,r2.InfrastructureError)
                    raise error_type(f"worker {shard} exited {rc}; see immutable worker log")
    finally:
        for log in logs: log.flush();os.fsync(log.fileno());log.close()
    return {"status":"PASS","workers":len(processes)}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("command",choices=("startup","producer-preflight","write-producer-seal","build-rebinds","validate-startup","build-shards","run-all","reconcile","seal","validate","validate-publication"));p.add_argument("--repo-root",type=Path,default=REPO_ROOT);p.add_argument("--producer-commit");p.add_argument("--preflight-file",type=Path);p.add_argument("--output",type=Path);a=p.parse_args(argv);repo=a.repo_root.resolve()
    try:
        if a.command=="startup": startup(repo);result={"status":"PASS"}
        elif a.command=="producer-preflight":result=producer_preflight(repo)
        elif a.command=="write-producer-seal":result=write_producer_seal(repo,a.producer_commit,a.preflight_file)
        elif a.command=="build-rebinds":result=build_rebinds(repo)
        elif a.command=="validate-startup":result=validate_startup(repo)
        elif a.command=="build-shards":result=build_shards(repo)
        elif a.command=="run-all":result=run_all(repo)
        elif a.command=="reconcile":result=validator(repo).reconcile(repo,write=True)
        elif a.command=="seal":result=validator(repo).seal(repo)
        else:result=validator(repo).validate(repo,publication=a.command=="validate-publication")
        if a.output:r2.write_json_new(a.output,result)
    except Exception as error:
        result={"status":getattr(error,"code","BLOCKED_PROVENANCE"),"error":str(error),"next_authorization":"NONE"}
        if (repo/r2.ROOT/"completion/y2_r2_r1.pending.json").is_file():validator(repo).seal_failure(repo,a.command,error)
        print(json.dumps(result),flush=True);return 2
    print(json.dumps(result,indent=2,sort_keys=True),flush=True);return 0

if __name__=="__main__":raise SystemExit(main())
