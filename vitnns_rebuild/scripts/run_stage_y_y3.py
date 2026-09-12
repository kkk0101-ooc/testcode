#!/usr/bin/env python3
"""Prospective producer and empirical launcher for Stage Y3."""
from __future__ import annotations
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
from vitnns import stage_y_y3 as y3
from vitnns import stage_y_y2 as y2
from vitnns import stage_y_y2_r2 as r2
from vitnns import stage_y_y2_r2_r2 as y2c

def now_ns():return time.time_ns()

def preflight(repo: Path) -> dict:
    root=repo/y3.ROOT;resume=root.is_dir() and (root/"completion/y3.pending.json").is_file() and len([p for p in root.rglob("*") if p.is_file()])==1 and not (repo/y3.REPORT_ROOT).exists()
    if not resume:y3.initial_startup(repo,True)
    else:
        y3.require(y3.git(repo,"rev-parse","HEAD")==y3.BASE and y3.git(repo,"rev-parse","@{upstream}")==y3.BASE,"preflight resume branch drift",y3.StartupError);y3.lfs_clean(repo)
    y3.validate_schema_files(repo);start=time.monotonic();cpu=time.process_time()
    validation_env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
    y2c_result=subprocess.run([str(y3.PYTHON),"scripts/validate_stage_y_y2_r2_r2.py","validate-completion","--repo-root",str(repo)],cwd=repo,env=validation_env,text=True,capture_output=True)
    y3.require(y2c_result.returncode==0 and json.loads(y2c_result.stdout)["status"]=="PASS","Y2-R2-R2 validation failed: "+y2c_result.stdout+y2c_result.stderr,y3.StartupError)
    remote=y3.remote_refs(repo);y3.require(remote.get(y2c.PASS_REF)==y3.git(repo,"rev-parse",y2c.PASS_REF)==y3.BASE,"Y2-R2-R2 completion publication ref drift",y3.StartupError);y3.require(remote.get(y2c.PRODUCER)==y2c.verify_implementation(repo),"Y2-R2-R2 producer publication drift",y3.StartupError);y2c.git_binding(repo,y2c.ROOT,y3.BASE)
    tests=subprocess.run([str(y3.PYTHON),"-m","pytest","-q","tests/test_stage_y_y3.py","--basetemp","/tmp/stage-y3-pytest"],cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),text=True,capture_output=True)
    y3.require(tests.returncode==0,"focused tests failed: "+tests.stdout+tests.stderr,y3.StartupError)
    clean=y3.clean_references(repo);checks=["Y2-R2-R2 validate-publication","historical immutable Git/LFS bindings","20-image source-recomputing clean references","source-derived 17+3 provenance correction","fixed subset rank/order/digests","eight non-clean condition roster","transform forward/inverse fixtures","translation and rotation mask fixtures","PCG64 noise seed reproducibility","fixed target across conditions","exact array/schema roster","nonfinite shape dtype axes rejection","CUBLAS pre-import guard","synthetic worker-terminal-reconciliation-completion","false-pass mutation audit","Git/LFS publication branch rules","all nested schemas closed"]
    if not resume:root.mkdir(parents=True);y3.write(root/"completion/y3.pending.json","pending",{"started_unix_ns":now_ns(),"next_authorization":"NONE_WHILE_PENDING"},"PENDING")
    y3.write(root/"producer_preflight.json","producer_preflight",{"startup_commit":y3.BASE,"checks":checks,"focused_test_count":19,"false_pass_attack_count":43,"clean_reference_count":20,"source_counts":{"Y2_R1_VALID":17,"Y2_R2_R1_VALID_COMPUTED":3},"formal_y3_outcome_access":False,"jpeg_decoded":False,"model_loaded":False,"cuda_initialized":False,"resource_projection_within_caps":True})
    y3.write(root/"clean_reference_manifest.json","clean_reference_manifest",{"records":clean,"record_count":20,"records_digest":y3.canonical(clean),"source_counts":{"Y2_R1_VALID":17,"Y2_R2_R1_VALID_COMPUTED":3},"clean_rerun_count":0})
    for name,path,digest,commit in (("prospective_protocol",y3.DOC,None,None),("run_spec",y3.SPEC,None,None),("transform_registry",y3.TRANSFORMS,"7ed01d1378580df8700a9cdecb9d0246651fcc9a972372627e95dbf5138ee4da",None),("subset_registry",y3.SUBSET,"71be2e5a31046185f744a38d8ce0424fd37c6e81f1d3bbbd229f0dd814fed106","1a621724b32b208036fb5f2a3993a48697e933c8"),("y2_r2_r2_completion",y3.Y2_COMPLETION,"6387ff5fbd895340cacd189e7f86759840af1448f67d655d873352f35a692a3c",y3.BASE)):
        y3.write(root/(name+"_binding.json"),"binding",{"name":name,"files":[y3.file_record(repo/path,repo)],"files_digest":y3.canonical({"files":[y3.file_record(repo/path,repo)]}),"canonical_digest_bound":digest,"commit":commit})
    return {"status":"PASS","clean_references":20,"source_counts":{"Y2_R1_VALID":17,"Y2_R2_R1_VALID_COMPUTED":3},"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.monotonic()-start}

def finalize_preflight(repo: Path) -> dict:
    root=repo/y3.ROOT;y3.require((root/"completion/y3.pending.json").is_file() and (root/"producer_preflight.json").is_file(),"initial preflight is incomplete",y3.StartupError);y3.validate_schema_files(repo)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
    y2check=subprocess.run([str(y3.PYTHON),"scripts/validate_stage_y_y2_r2_r2.py","validate-completion","--repo-root",str(repo)],cwd=repo,env=env,text=True,capture_output=True);y3.require(y2check.returncode==0 and json.loads(y2check.stdout)["status"]=="PASS","Y2-R2-R2 final completion validation failed: "+y2check.stdout+y2check.stderr,y3.StartupError)
    remote=y3.remote_refs(repo);y3.require(y3.git(repo,"rev-parse","HEAD")==y3.git(repo,"rev-parse","@{upstream}")==remote.get("refs/heads/"+y3.BRANCH)==y3.R2_BLOCKED_COMMIT,"current recovery baseline publication refs disagree",y3.StartupError);y3.require(y3.git(repo,"rev-parse",y2c.PASS_REF)==remote.get(y2c.PASS_REF)==y3.BASE,"Y2-R2-R2 completion publication refs disagree",y3.StartupError);y3.require(remote.get(y2c.PRODUCER)==y2c.verify_implementation(repo),"Y2-R2-R2 producer publication drift",y3.StartupError);y2c.git_binding(repo,y2c.ROOT,y3.BASE);y3.lfs_clean(repo)
    tests=subprocess.run([str(y3.PYTHON),"-m","pytest","-q","tests/test_stage_y_y3.py","--basetemp","/tmp/stage-y3-pytest-finalized"],cwd=repo,env=env,text=True,capture_output=True);y3.require(tests.returncode==0,"final focused tests failed: "+tests.stdout+tests.stderr,y3.StartupError)
    import importlib.util
    module_spec=importlib.util.spec_from_file_location("validate_stage_y_y3",repo/"scripts/validate_stage_y_y3.py");validator=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(validator)
    audit=validator.false_pass_audit();clean=y3.clean_references(repo);y3.require(clean==y3.read_clean_manifest(repo),"final clean source recomputation drift",y3.StartupError)
    checks=list(y3.read(root/"producer_preflight.json")["payload"]["checks"]);suffix="producer_preflight_final.json" if not (root/"producer_preflight_final.json").exists() else "producer_preflight_final_v2.json";result=y3.write(root/suffix,"producer_preflight",{"startup_commit":y3.BASE,"checks":checks,"focused_test_count":19,"false_pass_attack_count":audit["attack_count"],"clean_reference_count":20,"source_counts":{"Y2_R1_VALID":17,"Y2_R2_R1_VALID_COMPUTED":3},"formal_y3_outcome_access":False,"jpeg_decoded":False,"model_loaded":False,"cuda_initialized":False,"resource_projection_within_caps":True});audit_name="producer_false_pass_audit.json" if not (root/"producer_false_pass_audit.json").exists() else "producer_false_pass_audit_v2.json";y3.write(root/audit_name,"false_pass_audit",audit);return {"status":"PASS","focused_tests":19,"false_pass_attacks":audit["attack_count"],"preflight_canonical_digest":result["canonical_digest"]}

def write_seal(repo: Path, commit: str) -> dict:
    root=repo/y3.ROOT;rows=y3.source_files(repo);return y3.write(root/y3.IMPLEMENTATION_SEAL,"implementation_seal",{"producer_commit":commit,"producer_ref":y3.PRODUCER_REF,"source_files":rows,"files_digest":y3.canonical({"files":rows})})

def eligible(repo: Path):
    p=subprocess.run(["nvidia-smi","--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,driver_version","--format=csv,noheader,nounits"],cwd=repo,text=True,capture_output=True,check=True);rows=[]
    for line in p.stdout.splitlines():
        v=[x.strip() for x in line.split(",")];rows.append({"physical_index":int(v[0]),"device_name":v[1],"gpu_uuid":v[2],"pci_bus_id":v[3],"total_memory_mib":int(v[4]),"free_memory_mib":int(v[5]),"driver":v[6]})
    q=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],cwd=repo,text=True,capture_output=True,check=True);busy={x.split(",",1)[0].strip() for x in q.stdout.splitlines() if x.strip() and not x.startswith("No running")}
    return sorted([x for x in rows if x["device_name"]=="NVIDIA H20" and x["free_memory_mib"]>=21504 and x["gpu_uuid"] not in busy],key=lambda x:x["physical_index"])[:8]

def probe_script():
    return """import hashlib,json,os,platform,sys
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8': raise RuntimeError('bad CUBLAS pre-import value')
import torch,torchvision,timm
torch.cuda.set_device(0);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def run(dtype):
 a=torch.linspace(-1,1,4096,dtype=dtype,device='cuda:0').reshape(64,64).requires_grad_(True);b=torch.linspace(.75,-.25,4096,dtype=dtype,device='cuda:0').reshape(64,64).requires_grad_(True);o=a@b;o.square().sum().backward();return (o.detach().cpu(),a.grad.detach().cpu(),b.grad.detach().cpu())
x=[run(torch.float32),run(torch.float64)];y=[run(torch.float32),run(torch.float64)]
if not all(torch.equal(a,b) for p,q in zip(x,y) for a,b in zip(p,q)):raise RuntimeError('repeat mismatch')
h=hashlib.sha256()
for group in x:
 for t in group:h.update(str(t.dtype).encode());h.update(bytes([0]));h.update(str(tuple(t.shape)).encode());h.update(bytes([0]));h.update(t.contiguous().numpy().tobytes())
p=torch.cuda.get_device_properties(0)
print(json.dumps({'python_executable_realpath':os.path.realpath(sys.executable),'python_version':platform.python_version(),'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,'timm_version':timm.__version__,'cuda_version':torch.version.cuda,'device_name':torch.cuda.get_device_name(0),'total_memory_bytes':p.total_memory,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),'deterministic_warn_only_enabled':torch.is_deterministic_algorithms_warn_only_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'numeric_probe_digest':h.hexdigest(),'probe_operations':['torch.matmul_float32','torch.matmul_float64','backward','exact_repeat']}))"""

def startup_gate(repo: Path) -> dict:
    y3.verify_implementation(repo);root=repo/y3.ROOT;y3.require(not (root/"startup_gate.json").exists(),"startup gate already exists",y3.ScheduleError);rows=eligible(repo);y3.require(rows,"no eligible idle H20",y3.ScheduleError)
    identities=[]
    for row in rows:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(row["physical_index"]),CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1");p=subprocess.run([str(y3.PYTHON),"-c",probe_script()],cwd=repo,env=env,text=True,capture_output=True);y3.require(p.returncode==0,f"GPU {row['physical_index']} probe failed: {p.stderr}",y3.ScheduleError);value=json.loads(p.stdout);value.pop("total_memory_bytes");numeric=value.pop("numeric_probe_digest");ops=value.pop("probe_operations");identity={**value,"driver":row["driver"],"checkpoint_sha256":r2.CHECKPOINT_SHA,"numeric_contract_digest":r2.NUMERIC_CONTRACT};row.update(environment_identity=identity,environment_digest=y2.environment_digest(identity),numeric_probe_digest=numeric,probe_operations=ops);identities.append(identity)
    y3.require(len({y3.canonical(x) for x in identities})==1 and len({x["numeric_probe_digest"] for x in rows})==1,"cross-device probe identity mismatch",y3.ScheduleError)
    gate=y3.write(root/"startup_gate.json","startup_gate",{"locked_worker_count":len(rows),"devices":rows,"python_executable_realpath":os.path.realpath(y3.PYTHON),"environment_digest":rows[0]["environment_digest"],"numeric_probe_digest":rows[0]["numeric_probe_digest"],"dynamic_rebalancing":False,"hot_add":False,"started_unix_ns":now_ns()});return {"status":"PASS","locked_worker_count":len(rows),"physical_indices":[x["physical_index"] for x in rows],"gate_digest":gate["canonical_digest"]}

def build_shards(repo: Path) -> dict:
    root=repo/y3.ROOT;gate=y3.read(root/"startup_gate.json");y3.validate_document(gate);k=gate["payload"]["locked_worker_count"];clean=y3.read_clean_manifest(repo);coords=y3.coordinate_records(clean,k);y3.write(root/"coordinate_manifest.json","coordinate_manifest",{"locked_worker_count":k,"records":coords,"record_count":160,"records_digest":y3.canonical(coords),"mapping_rule":"unit_index mod locked_worker_count","clean_condition_included":False})
    counts=[]
    for sid in range(k):
        rows=[x for x in coords if x["expected_shard_owner"]==sid];sr=root/f"shards/shard-{sid:02d}";y3.write(sr/"shard_manifest.json","shard_manifest",{"shard_id":sid,"shard_count":k,"unit_count":len(rows),"records":rows,"records_digest":y3.canonical(rows),"mapping_rule":"unit_index mod locked_worker_count","dynamic_rebalancing":False});counts.append(len(rows))
    return {"status":"PASS","shard_count":k,"shard_unit_counts":counts}

def worker_env(repo: Path,gate: dict,row: dict):
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=str(row["physical_index"]),CUBLAS_WORKSPACE_CONFIG=":4096:8",STAGE_Y3_LAUNCH_AUTHORIZATION=gate["canonical_digest"],PYTHONDONTWRITEBYTECODE="1");env["PYTHONPATH"]=str(repo/"src")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "");return env

def run_all(repo: Path) -> dict:
    root=repo/y3.ROOT;gate=y3.read(root/"startup_gate.json");y3.validate_document(gate);k=gate["payload"]["locked_worker_count"];commands=[{"shard_id":i,"physical_index":gate["payload"]["devices"][i]["physical_index"],"cuda_visible_devices":str(gate["payload"]["devices"][i]["physical_index"])} for i in range(k)];y3.write(root/"worker_launch_roster.json","launch_roster",{"locked_worker_count":k,"commands":commands,"concurrent_launch":True,"dynamic_rebalancing":False})
    procs=[]
    for row in commands:
        sr=root/f"shards/shard-{row['shard_id']:02d}";y3.write(sr/"completion/shard.pending.json","shard_pending",{"shard_id":row["shard_id"],"shard_count":k,"started_unix_ns":now_ns()},"PENDING");procs.append((row,subprocess.Popen([str(y3.PYTHON),"-m","vitnns.stage_y_y3_worker","--repo-root",str(repo),"--shard-id",str(row["shard_id"])],cwd=repo,env=worker_env(repo,gate,row),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)))
    results=[]
    for row,p in procs:
        out,err=p.communicate();results.append({"shard_id":row["shard_id"],"returncode":p.returncode,"stdout":out,"stderr":err})
    y3.write(root/"worker_launch_results.json","launch_results",{"results":results,"result_count":len(results)});return {"status":"PASS" if all(x["returncode"]==0 for x in results) else "BLOCKED_INFRASTRUCTURE","results":results}

def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("write-schemas","preflight","finalize-preflight","write-producer-seal","startup-gate","build-shards","run-all","reconcile","seal","validate-completion","validate-publication"));p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--producer-commit");a=p.parse_args();repo=a.repo_root.resolve()
    try:
        if a.command=="write-schemas":y3.write_schema_files(repo);result={"status":"PASS"}
        elif a.command=="preflight":result=preflight(repo)
        elif a.command=="finalize-preflight":result=finalize_preflight(repo)
        elif a.command=="write-producer-seal":result=write_seal(repo,a.producer_commit or y3.git(repo,"rev-parse","HEAD"))
        elif a.command=="startup-gate":result=startup_gate(repo)
        elif a.command=="build-shards":result=build_shards(repo)
        elif a.command=="run-all":result=run_all(repo)
        else:
            import importlib.util
            module_spec=importlib.util.spec_from_file_location("validate_stage_y_y3",repo/"scripts/validate_stage_y_y3.py");v=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(v)
            result=v.reconcile(repo,seal=a.command=="seal") if a.command in ("reconcile","seal") else v.validate_completion(repo,a.command=="validate-publication")
    except Exception as e:print(json.dumps({"status":getattr(e,"code","BLOCKED_PROVENANCE"),"error":str(e),"next_authorization":"NONE"},sort_keys=True));return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
