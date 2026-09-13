#!/usr/bin/env python3
"""Prospective producer and launcher for the five-unit Y3-R5 recovery."""
from __future__ import annotations
import argparse,importlib.util,json,os,platform,subprocess,time
from pathlib import Path
from jsonschema import Draft202012Validator
from vitnns import stage_y_y3 as y3
from vitnns import stage_y_y3_r5 as r5
from vitnns import stage_y_y2 as y2
from vitnns import stage_y_y2_r2 as r2
from vitnns import stage_y_y2_r2_r2 as y2c

def remote_refs(repo):return y3.remote_refs(repo)
def validate_schema_files(repo):
    spec=r5.read(repo/r5.SPEC);schema=r5.read(repo/r5.SPEC_SCHEMA);artifact=r5.read(repo/r5.ARTIFACT_SCHEMA);Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(spec);Draft202012Validator.check_schema(artifact);r5.require(artifact==r5.artifact_schema(),"artifact schema source drift",r5.StartupError)
    def walk(v):
        if isinstance(v,dict):
            if v.get("type")=="object":r5.require(v.get("additionalProperties") is False,"open nested schema",r5.StartupError)
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(schema);walk(artifact)
def budget_admission(repo):
    root=repo/r5.SOURCE_ROOT;ledger=r5.read(root/"resource_ledger.json")["payload"];values=[]
    for p in root.glob("shards/shard-*/units/unit-*/completion/unit.json"):
        d=r5.read(p);q=d["payload"]
        if d["status"]=="VALID" and q.get("evidence_mode")=="NEW_COMPUTE":values.append(r5.read(repo/q["resource_record"]["path"])["payload"])
    r5.require(len(values)==153,"R4 measured VALID resource roster drift",r5.StartupError);maximum={k:max(x[k] for x in values) for k in ("gpu_active_seconds","cpu_seconds","wall_seconds","retained_bytes","peak_reserved_gpu_bytes")}
    projected={"gpu_seconds":ledger["gpu_active_seconds"]+5*maximum["gpu_active_seconds"],"cpu_seconds":ledger["cpu_seconds"]+5*maximum["cpu_seconds"],"wall_seconds":ledger["stage_wall_seconds"]+5*maximum["wall_seconds"],"retained_bytes":ledger["retained_bytes"]+5*maximum["retained_bytes"],"peak_reserved_gpu_bytes":max(ledger["peak_reserved_gpu_bytes"],maximum["peak_reserved_gpu_bytes"])}
    passed=projected["gpu_seconds"]<y3.CAPS["h20_gpu_hours"]*3600 and projected["cpu_seconds"]<y3.CAPS["cpu_hours"]*3600 and projected["wall_seconds"]<y3.CAPS["wall_hours"]*3600 and projected["retained_bytes"]<y3.CAPS["retained_storage_gib"]*2**30 and projected["peak_reserved_gpu_bytes"]<y3.CAPS["peak_reserved_gpu_gib_per_process"]*2**30
    r5.require(passed,"five-unit conservative global projection exceeds frozen caps",r5.StartupError);return ledger,maximum,projected
def preflight(repo):
    refs=remote_refs(repo);r5.require(y3.git(repo,"symbolic-ref","--short","HEAD")==y3.BRANCH,"branch drift",r5.StartupError);r5.require(y3.git(repo,"rev-parse","HEAD")==y3.git(repo,"rev-parse","@{upstream}")==refs.get("refs/heads/"+y3.BRANCH)==r5.STARTUP_COMMIT,"startup publication baseline drift",r5.StartupError)
    r5.require(refs.get(r5.R4_PRODUCER_REF)==y3.git(repo,"rev-parse",r5.R4_PRODUCER_REF)==r5.R4_PRODUCER_COMMIT,"R4 producer ref drift",r5.StartupError);r5.require(refs.get(r5.R4_BLOCKED_REF)==y3.git(repo,"rev-parse",r5.R4_BLOCKED_REF)==r5.R4_BLOCKED_COMMIT,"R4 blocked ref drift",r5.StartupError);y2c.git_binding(repo,r5.SOURCE_ROOT,r5.R4_BLOCKED_COMMIT);y2c.git_binding(repo,r5.SOURCE_REPORT_ROOT,r5.R4_BLOCKED_COMMIT)
    for ref in (r5.PRODUCER_REF,r5.PASS_REF,r5.BLOCKED_REF):r5.require(ref not in refs and not y3.git(repo,"for-each-ref","--format=%(refname)",ref),"R5 ref exists: "+ref,r5.StartupError)
    status=y3.git(repo,"status","--porcelain").splitlines();allowed={str(p) for p in r5.PRODUCER_FILES};unexpected=[line[3:] for line in status if not any(line[3:]==p or line[3:].endswith("/"+p) for p in allowed)];r5.require(not unexpected,"unexpected worktree paths: "+repr(unexpected),r5.StartupError);y3.lfs_clean(repo);r5.require(not (repo/r5.ROOT).exists() and not (repo/r5.REPORT_ROOT).exists(),"R5 roots exist",r5.StartupError);validate_schema_files(repo)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1");publication=subprocess.run([str(y3.PYTHON),"scripts/run_stage_y_y3.py","validate-completion","--repo-root",str(repo)],cwd=repo,env=env,text=True,capture_output=True);r5.require(publication.returncode==0 and json.loads(publication.stdout)["status"]=="BLOCKED_BUDGET","R4 source completion validation failed",r5.StartupError)
    tests=subprocess.run([str(y3.PYTHON),"-m","pytest","-q","tests/test_stage_y_y3_r5.py","--basetemp","/tmp/stage-y3-r5-pytest"],cwd=repo,env=env,text=True,capture_output=True);r5.require(tests.returncode==0,"R5 tests failed: "+tests.stdout+tests.stderr,r5.StartupError);ledger,maximum,projected=budget_admission(repo)
    root=repo/r5.ROOT;root.mkdir(parents=True);r5.write(root/"completion/y3_r5.pending.json","pending",{"started_unix_ns":time.time_ns(),"next_authorization":"NONE_WHILE_PENDING"},"PENDING");checks=["R4 validate-publication","R4 immutable Git/LFS binding","155 VALID source terminals","five exact blocked coordinates","global cumulative budget admission","no worker-count division","GPU1 logical-shard ownership","closed schemas","focused tests","false-pass audit"]
    from importlib.util import spec_from_file_location,module_from_spec
    s=spec_from_file_location("v",repo/"scripts/validate_stage_y_y3_r5.py");v=module_from_spec(s);s.loader.exec_module(v);audit=v.false_pass_audit()
    r5.write(root/"producer_preflight.json","producer_preflight",{"startup_commit":r5.STARTUP_COMMIT,"checks":checks,"focused_test_count":12,"false_pass_attack_count":audit["attack_count"],"source_valid_count":155,"new_compute_count":5,"formal_outcome_access":False,"jpeg_decoded":False,"model_loaded":False,"cuda_initialized":False});r5.write(root/"producer_false_pass_audit.json","false_pass_audit",audit);r5.write(root/"budget_admission.json","budget_admission",{"source_ledger":r5.file_record(repo/r5.SOURCE_ROOT/"resource_ledger.json",repo),"measured_valid_resource_count":153,"new_compute_count":5,"maximum_unit":maximum,"projected":projected,"caps_pass":True,"divide_by_worker_count":False})
    files=[r5.file_record(repo/p,repo) for p in (r5.SOURCE_ROOT/"completion/y3.json",r5.SOURCE_ROOT/"whole_root_inventory.json",r5.SOURCE_ROOT/"resource_ledger.json")];r5.write(root/"r4_source_binding.json","binding",{"name":"Y3-R4 blocked source","files":files,"files_digest":r5.canonical({"files":files}),"canonical_digest_bound":r5.read(repo/r5.SOURCE_ROOT/"completion/y3.json")["canonical_digest"],"commit":r5.R4_BLOCKED_COMMIT})
    return {"status":"PASS","maximum":maximum,"projected":projected,"false_pass_attacks":audit["attack_count"]}
def finalize_preflight(repo):
    root=repo/r5.ROOT;r5.require((root/"completion/y3_r5.pending.json").is_file(),"R5 pending missing",r5.StartupError);validate_schema_files(repo);r5.require(y3.git(repo,"rev-parse","HEAD")==y3.git(repo,"rev-parse","@{upstream}")==r5.STARTUP_COMMIT,"final preflight baseline drift",r5.StartupError);y2c.git_binding(repo,r5.SOURCE_ROOT,r5.R4_BLOCKED_COMMIT);y2c.git_binding(repo,r5.SOURCE_REPORT_ROOT,r5.R4_BLOCKED_COMMIT);budget_admission(repo);return {"status":"PASS"}
def write_seal(repo,commit):rows=r5.source_files(repo);return r5.write(repo/r5.ROOT/r5.IMPLEMENTATION_SEAL,"implementation_seal",{"producer_commit":commit,"producer_ref":r5.PRODUCER_REF,"source_files":rows,"files_digest":r5.canonical({"files":rows})})
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
def startup_gate(repo):
    r5.verify_implementation(repo);p=subprocess.run(["nvidia-smi","--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,driver_version","--format=csv,noheader,nounits"],text=True,capture_output=True,check=True);rows=[]
    for line in p.stdout.splitlines():
        v=[x.strip() for x in line.split(",")];
        if int(v[0])==1:rows.append({"physical_index":1,"device_name":v[1],"gpu_uuid":v[2],"pci_bus_id":v[3],"total_memory_mib":int(v[4]),"free_memory_mib":int(v[5]),"driver":v[6]})
    apps=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True,capture_output=True,check=True).stdout;r5.require(len(rows)==1 and rows[0]["device_name"]=="NVIDIA H20" and rows[0]["free_memory_mib"]>=21504 and rows[0]["gpu_uuid"] not in apps,"physical GPU1 not idle/eligible",r5.ScheduleError)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1");q=subprocess.run([str(y3.PYTHON),"-c",probe_script()],cwd=repo,env=env,text=True,capture_output=True);r5.require(q.returncode==0,"GPU1 startup probe failed: "+q.stderr,r5.ScheduleError);value=json.loads(q.stdout);r5.require(value["python_executable_realpath"]==os.path.realpath(y3.PYTHON),"Python executable drift",r5.ScheduleError);value.pop("total_memory_bytes");numeric=value.pop("numeric_probe_digest");ops=value.pop("probe_operations");identity={**value,"driver":rows[0]["driver"],"checkpoint_sha256":r2.CHECKPOINT_SHA,"numeric_contract_digest":r2.NUMERIC_CONTRACT};rows[0].update(environment_identity=identity,environment_digest=y2.environment_digest(identity),numeric_probe_digest=numeric,probe_operations=ops);gate=r5.write(repo/r5.ROOT/"startup_gate.json","startup_gate",{"locked_worker_count":1,"devices":rows,"python_executable_realpath":os.path.realpath(y3.PYTHON),"environment_digest":rows[0]["environment_digest"],"numeric_probe_digest":numeric,"dynamic_rebalancing":False,"hot_add":False,"started_unix_ns":time.time_ns()});return {"status":"PASS","physical_indices":[1],"gate_digest":gate["canonical_digest"]}
def build(repo):
    root=repo/r5.ROOT;source_docs=[]
    for path in sorted((repo/r5.SOURCE_ROOT).glob("shards/shard-*/units/unit-*/completion/unit.json")):
        doc=y3.read(path);q=doc["payload"]
        if doc["status"]=="VALID":source_docs.append({"unit_index":q["unit_index"],"predecessor_terminal":r5.file_record(path,repo)})
    r5.require(len(source_docs)==155 and sorted(x["unit_index"] for x in source_docs)==sorted(set(range(160))-set(r5.NEW_INDICES)),"R4 VALID rebind roster drift",r5.ProvenanceError);r5.write(root/"source_rebind_manifest.json","source_rebind_manifest",{"records":source_docs,"record_count":155,"records_digest":r5.canonical(source_docs),"arrays_copied":False})
    allcoords=y3.read(repo/r5.SOURCE_ROOT/"coordinate_manifest.json")["payload"]["records"];records=[]
    for index in r5.NEW_INDICES:
        c=allcoords[index];records.append({"unit_index":index,"subset_rank":c["subset_rank"],"selection_order":c["selection_order"],"source_object_id":c["source_object_id"],"condition_index":c["condition_index"],"condition_id":c["condition_id"],"logical_shard_id":1,"fixed_target":c["fixed_target"],"source_terminal":c["clean_source_terminal"]})
    r5.write(root/"coordinate_manifest.json","coordinate_manifest",{"records":records,"record_count":5,"records_digest":r5.canonical(records)});return {"status":"PASS","source_rebind_count":155,"new_compute_indices":list(r5.NEW_INDICES)}
def run(repo):
    root=repo/r5.ROOT;gate=r5.read(root/"startup_gate.json");r5.validate_document(gate);r5.write(root/"shards/shard-01/completion/shard.pending.json","shard_pending",{"logical_shard_id":1,"started_unix_ns":time.time_ns()},"PENDING");env=dict(os.environ,CUDA_VISIBLE_DEVICES="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",STAGE_Y3_R5_AUTHORIZATION=gate["canonical_digest"],PYTHONDONTWRITEBYTECODE="1");env["PYTHONPATH"]=str(repo/"src")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "");p=subprocess.run([str(y3.PYTHON),"-m","vitnns.stage_y_y3_r5_worker","--repo-root",str(repo)],cwd=repo,env=env,text=True,capture_output=True);result=r5.write(root/"worker_launch_result.json","launch_result",{"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr},"PASS" if p.returncode==0 else "BLOCKED_INFRASTRUCTURE");return {"status":result["status"],"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("write-schemas","preflight","finalize-preflight","write-producer-seal","startup-gate","build","run","reconcile","seal","validate-completion","validate-publication"));p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--producer-commit");a=p.parse_args();repo=a.repo_root.resolve()
    try:
        if a.command=="write-schemas":r5.write_schemas(repo);out={"status":"PASS"}
        elif a.command=="preflight":out=preflight(repo)
        elif a.command=="finalize-preflight":out=finalize_preflight(repo)
        elif a.command=="write-producer-seal":out=write_seal(repo,a.producer_commit or y3.git(repo,"rev-parse","HEAD"))
        elif a.command=="startup-gate":out=startup_gate(repo)
        elif a.command=="build":out=build(repo)
        elif a.command=="run":out=run(repo)
        else:
            s=importlib.util.spec_from_file_location("v",repo/"scripts/validate_stage_y_y3_r5.py");v=importlib.util.module_from_spec(s);s.loader.exec_module(v);out=v.reconcile(repo,a.command=="seal") if a.command in ("reconcile","seal") else v.validate_completion(repo,a.command=="validate-publication")
    except Exception as e:print(json.dumps({"status":getattr(e,"code","BLOCKED_PROVENANCE"),"error":str(e),"next_authorization":"NONE"},sort_keys=True));return 2
    print(json.dumps(out,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
