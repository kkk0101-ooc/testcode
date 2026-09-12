from __future__ import annotations

import ast
import copy
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import ValidationError
from vitnns import stage_y_y2_r2_r2 as r

REPO=Path(__file__).resolve().parents[1]


def sample(schema):
    if "const" in schema:return copy.deepcopy(schema["const"])
    if "enum" in schema:return schema["enum"][0]
    if "anyOf" in schema:return sample(schema["anyOf"][0])
    kind=schema["type"]
    if kind=="object":return {k:sample(v) for k,v in schema["properties"].items() if k in schema["required"]}
    if kind=="array":return [sample(schema["items"]) for _ in range(schema.get("minItems",0))]
    if kind=="string":return "a"*64 if "{64}" in schema.get("pattern","") else "a"*40 if "{40}" in schema.get("pattern","") else "PASS"
    if kind=="boolean":return True
    if kind=="null":return None
    return schema.get("minimum",0)


def synthetic_arrays():
    arrays={name:np.zeros(r.old.expected_shape(name),dtype="<f4") for name in r.old.EXPECTED_ARRAY_NAMES}
    arrays["input_attribution"][0,0,0]=1
    for name in arrays:
        if name in ("input_cell_map","post_embedding_cell","final_norm_cell","classifier_input_cell") or any(name==f"b{b:02d}_{s}_cell" for b in range(12) for s in ("input","output","first_identity","second_residual")):arrays[name][0,0]=1
        if name in ("post_embedding_cls","final_norm_cls","classifier_input_cls") or any(name==f"b{b:02d}_{s}_cls" for b in range(12) for s in ("input","output","first_identity","second_residual")):arrays[name][0]=1
    return arrays


def test_numpy_closure_matches_frozen_function_on_synthetic_cases():
    source=ast.parse((REPO/"src/vitnns/stage_y_y1.py").read_text())
    node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="validate_closure")
    node.returns=None
    for arg in node.args.args:arg.annotation=None
    namespace={k:getattr(r,k) for k in ("np","BLOCKS","INPUT_ATOL","INPUT_RTOL","CUT_PIXEL_ATOL","CUT_PIXEL_RTOL","CUT_TOTAL_ATOL","CUT_TOTAL_RTOL","validate_scientific_array")}
    exec(compile(ast.Module(body=[node],type_ignores=[]),"frozen_numpy_closure","exec"),namespace)
    for perturb in (False,True):
        arrays=synthetic_arrays()
        if perturb:arrays["b03_first_attention_cell"][1,1]=10
        assert namespace["validate_closure"](arrays,1.0)==r.validate_closure(arrays,1.0)
    assert "torch" not in sys.modules


def test_schema_is_closed_and_all_document_variants_roundtrip():
    def check(value):
        if isinstance(value,dict):
            if value.get("type")=="object":assert value["additionalProperties"] is False
            for child in value.values():check(child)
        if isinstance(value,list):
            for child in value:check(child)
    check(r.artifact_schema());r.validate_schema_sources(REPO)
    for kind,schema in r.artifact_schema()["$defs"].items():r.validate_document(r.signed(kind,sample(schema)))


def raw_rows():
    values=(1e16,1.,1.,3.125,0.03125,65536.125)
    rows=[r.observation("Y2-R1","unit",f"unit-{i:03d}","cpu_seconds",value,[],[]) for i,value in enumerate(values)]
    rows.append(r.observation("Y2-R2","unrecorded","cpu","cpu_seconds",None,[],[],status="NOT_RECORDED"))
    return rows


def test_resource_permutation_canonical_bytes_and_old_failure_reproduction():
    rows=raw_rows();expected=r.canonical_bytes(r.canonical_ledger(rows))
    rng=random.Random(901)
    for _ in range(12):
        rng.shuffle(rows);assert r.canonical_bytes(r.canonical_ledger(rows))==expected
    assert sum([1e16,1.,1.])!=sum([1.,1.,1e16])
    assert math.fsum([1e16,1.,1.])==math.fsum([1.,1.,1e16])
    assert r.canonical_ledger(rows)["stage_totals"][2]["cpu_seconds"] is None


def git(repo,*args):
    return subprocess.check_output(["git",*args],cwd=repo,text=True,stderr=subprocess.STDOUT).strip()


def old_doc(kind,payload,status="PASS",stage="Y2-R1"):
    value={"$schema":"synthetic.json","schema_version":1,"protocol_id":r.old.PROTOCOL_ID,"document_type":kind,"stage_id":stage,"status":status,"payload":payload}
    value["canonical_digest"]=r.canonical(value);return value


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.canonical_bytes(value)+b"\n")


@pytest.fixture(scope="module")
def synthetic_repo(tmp_path_factory):
    repo=tmp_path_factory.mktemp("ledger-synthetic")
    for path in r.FILES+r.DEPENDENCIES:
        destination=repo/path;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/path,destination)
    (repo/"assets").symlink_to(REPO/"assets",target_is_directory=True)
    arrays=synthetic_arrays();pool=repo/"fixture_pool";pool.mkdir()
    array_templates={}
    for name,array in arrays.items():
        path=pool/(name+".npy");np.save(path,array,allow_pickle=False)
        array_templates[name]={"bytes":path.stat().st_size,"sha256":r.sha(path),"canonical_digest":r.old.array_digest(array),"dtype":"float32","shape":list(array.shape),"axes":r.old.expected_axes(name,array.shape)}
    cohort=r.old.cohort_by_order(REPO)
    for base in r.ROOTS:
        (repo/base/"completion").mkdir(parents=True)
    for name in ("cohort_registry.json","audit_subset_registry.json","perturbation_subset_registry.json"):
        shutil.copyfile(REPO/r.old.Y2_ROOT/name,repo/r.old.Y2_ROOT/name)
    for base in (r.old.Y2_ROOT,r.old.Y2_R1_ROOT,r.old.ROOT):
        stage="Y2-R2-R1" if base==r.old.ROOT else "Y2-R1" if base==r.old.Y2_R1_ROOT else "Y2"
        gate_name="startup_gate.json" if base==r.old.ROOT else "multi_gpu_startup_gate.json"
        gate=r.read(REPO/base/gate_name)
        gate["payload"]["started_unix_ns"]=1000000000
        gate["canonical_digest"]=r.canonical({k:v for k,v in gate.items() if k!="canonical_digest"})
        put(repo/base/gate_name,gate)
        if base==r.old.ROOT:put(repo/base/"execution_start.json",old_doc("execution",{"started_unix_ns":1000000000},stage=stage))
        for shard in range(4):
            dest=repo/base/f"shards/shard-{shard:02d}"
            put(dest/"worker_identity.json",r.read(REPO/base/f"shards/shard-{shard:02d}/worker_identity.json"))
            fixed={"cpu_seconds":1.25+shard,"gpu_active_seconds":0.5+shard/8,"wall_seconds":0.5+shard/4,"peak_reserved_gpu_bytes":400000,"retained_bytes":100,"download_bytes":0}
            q={"shard_id":shard,"completed_unix_ns":9000000000+shard,"worker_wall_seconds":7.+shard/10,"shard_elapsed_seconds":7.+shard/10,"fixed_initialization":fixed,"fixed_initialization_resource":fixed}
            put(dest/"completion/shard.json",old_doc("shard",q,stage=stage))
    for order in range(100):
        for base in (r.old.Y2_ROOT,r.old.Y2_R1_ROOT,r.old.ROOT):
            if base==r.old.ROOT and order not in r.old.MISSING_ORDERS:continue
            stage="Y2-R2-R1" if base==r.old.ROOT else "Y2-R1" if base==r.old.Y2_R1_ROOT else "Y2"
            shard=r.old.MISSING_ORDERS.index(order)%4 if base==r.old.ROOT else order%4
            unit=repo/base/f"shards/shard-{shard:02d}/units/unit-{order:03d}"
            missing=base==r.old.Y2_ROOT or base==r.old.Y2_R1_ROOT and order in r.old.MISSING_ORDERS
            completed=base==r.old.Y2_R1_ROOT and order in r.old.COMPLETED_BUDGET_ORDERS
            resource={"cpu_seconds":(order+1)*0.125,"gpu_active_seconds":(order+1)*0.03125,"wall_seconds":(order+1)*0.0625,"peak_reserved_gpu_bytes":10000000,"peak_allocated_gpu_bytes":9000000,"retained_bytes":200,"download_bytes":0,"calls":dict.fromkeys(r.old.CALL_KEYS,0)}
            status="BLOCKED_INFRASTRUCTURE" if base==r.old.Y2_ROOT else "BLOCKED_BUDGET" if missing or completed else "VALID"
            failure="CUBLAS_WORKSPACE_CONFIG" if base==r.old.Y2_ROOT else "per-shard resource cap already exhausted" if missing else "per-shard resource cap exceeded by completed unit" if completed else None
            q={"selection_order":order,"source_object_id":cohort[order]["source_object_id"],"shard_id":shard,"condition_id":"clean","failure_code":failure,"target_identity":{"class_id":7,"rule":"clean predicted-class raw logit"},"source_binding":{"cohort_digest":r.old.COHORT_DIGEST,"file_sha256":cohort[order]["file_sha256"]},"checkpoint_sha256":r.old.CHECKPOINT_SHA,"numeric_contract_digest":r.old.NUMERIC_CONTRACT,"environment_digest":r.read(repo/base/f"shards/shard-{shard:02d}/worker_identity.json")["payload"]["environment_digest"],"array_records":[],"attempt_records":[],"quadrature_attempts":[],"scientific_record":None,"resource_telemetry":resource,"attempt_id":"failure-000"}
            if not missing:
                (unit/"arrays").mkdir(parents=True)
                for name,template in sorted(array_templates.items()):
                    path=unit/"arrays"/(name+".npy");os.link(pool/(name+".npy"),path);q["array_records"].append({"path":path.relative_to(repo).as_posix(),**template})
                steps=512 if order==86 and completed else 1024 if completed else 128
                for step in r.old.LADDER[:r.old.LADDER.index(steps)+1]:
                    passed=step==steps
                    q["quadrature_attempts"].append({"steps":step,"all_contracts_pass":passed,"input_sum":1. if passed else 2.,"input_delta":1.,"input_abs_error":0. if passed else 1.,"threshold":0.0051,"closure_status":"PASS" if passed else None,"identity_max_abs_error":0. if passed else None,"wall_seconds":0.1,"array_digest":array_templates["input_attribution"]["canonical_digest"]})
                if stage=="Y2-R2-R1":
                    q["attempt_id"]="scientific-001";path=unit/"attempts/attempt-001.json"
                    put(path,r.old.signed("attempt","PASS",{"attempt_id":q["attempt_id"],"selection_order":order,"retry_index":1,"quadrature_attempts":q["quadrature_attempts"]}));q["attempt_records"].append(r.record(path,repo))
                else:
                    for i,attempt in enumerate(q["quadrature_attempts"]):
                        path=unit/f"attempts/attempt-{i:03d}.json";q["attempt_id"]=f"q-{i:03d}"
                        put(path,old_doc("attempt",{"attempt_id":q["attempt_id"],"attempt_index":i,**attempt}));q["attempt_records"].append(r.record(path,repo))
                if completed:
                    q["resource_telemetry"]["calls"]=dict(zip(r.old.CALL_KEYS,(7359,352,6272,25088,128,3584) if steps==512 else (14015,736,12544,50176,256,7168)))
                else:
                    science=sample(r.science_schema());science.update(target_identity=q["target_identity"],delta=1.,clean_logit=1.,baseline_logit=0.,selected_steps=steps,closure=r.validate_closure(arrays,1.),calls=resource["calls"])
                    for endpoint in (science["ablations"],science["deletion_operator"]):
                        endpoint["target"]=7
                        for b,block in enumerate(endpoint["blocks"]):block["block"]=b
                    put(unit/"scientific_result.json",old_doc("science",science,"VALID",stage));q["scientific_result" if stage=="Y2-R2-R1" else "scientific_record"]=r.record(unit/"scientific_result.json",repo)
                if stage=="Y2-R2-R1":
                    q.pop("scientific_record");q["predecessor_terminal"]=r.record(repo/r.old.Y2_R1_ROOT/f"shards/shard-{order%4:02d}/units/unit-{order:03d}/completion/unit.json",repo)
                    put(unit/"array_inventory.json",old_doc("arrays",{"array_records":q["array_records"],"records_digest":r.canonical(q["array_records"])},stage=stage));q["array_inventory"]=r.record(unit/"array_inventory.json",repo)
                    put(unit/"resource_record.json",old_doc("resource",resource,stage=stage));q["resource_record"]=r.record(unit/"resource_record.json",repo)
            put(unit/"completion/unit.json",old_doc("terminal",q,status,stage))
    for base,name,status in ((r.old.Y2_ROOT,"y2","BLOCKED_INFRASTRUCTURE"),(r.old.Y2_R1_ROOT,"y2_r1","BLOCKED_BUDGET"),(r.old.Y2_R2_ROOT,"y2_r2","BLOCKED_STARTUP_STATE"),(r.old.ROOT,"y2_r2_r1","BLOCKED_SOURCE_DRIFT")):
        put(repo/base/f"completion/{name}.json",old_doc("completion",{"next_authorization":"NONE"},status))
    put(repo/r.old.Y2_R2_ROOT/"access_audit.json",old_doc("access",{"gpu_execution":False,"external_accesses":[]}))
    for name in ("rebind_validation_resource.json","reconciliation_resource.json","producer_preflight.json"):
        q={"cpu_seconds":3.125,"gpu_active_seconds":0,"wall_seconds":4.25,"retained_bytes":0,"peak_reserved_gpu_bytes":0,"download_bytes":0}
        put(repo/r.old.ROOT/name,old_doc("resource",{"resource_telemetry":q} if name=="producer_preflight.json" else q))
    oldrows=[r.checked(p)["payload"]["resource_telemetry"] for p in sorted((repo/r.old.ROOT).glob("shards/shard-*/units/unit-*/completion/unit.json"))]
    oldrows += [r.checked(p)["payload"]["fixed_initialization"] for p in sorted((repo/r.old.ROOT).glob("shards/shard-*/completion/shard.json"))]
    oldrows += [r.checked(repo/r.old.ROOT/p)["payload"] for p in ("rebind_validation_resource.json","reconciliation_resource.json")]+[r.checked(repo/r.old.ROOT/"producer_preflight.json")["payload"]["resource_telemetry"]]
    put(repo/r.old.ROOT/"resource_ledger.json",old_doc("ledger",{"stage_resources":{"Y2_R2_R1":{"cpu_hours":sum(x["cpu_seconds"] for x in oldrows)/3600,"h20_gpu_hours":sum(x["gpu_active_seconds"] for x in oldrows)/3600}}}))
    put(repo/r.old.ROOT/"reconciliation.json",old_doc("rec",{"valid_clean_count":100}))
    for base in r.ROOTS:
        inventory=r.payload_inventory(repo/base,{"whole_root_inventory.json",*(p.relative_to(repo/base).as_posix() for p in (repo/base/"completion").glob("*.json"))})
        simple=[{k:v for k,v in row.items() if k!="canonical_digest"} for row in inventory["files"]];inventory["files"]=simple;inventory["files_digest"]=r.canonical({"files":simple})
        put(repo/base/"whole_root_inventory.json",old_doc("inventory",inventory))
    for path,_ in r.PROTECTED:
        if str(path).startswith("reports/"):(repo/path).mkdir(parents=True,exist_ok=True)
    remote=tmp_path_factory.mktemp("ledger-remote")
    git(remote,"init","--bare");git(repo,"init");git(repo,"symbolic-ref","HEAD","refs/heads/stage-d-d3-pass")
    (repo/".gitattributes").write_text("*.npy filter=lfs diff=lfs merge=lfs -text\n")
    git(repo,"add",".")
    env=dict(os.environ,GIT_AUTHOR_NAME="Synthetic Test",GIT_AUTHOR_EMAIL="synthetic@example.invalid",GIT_COMMITTER_NAME="Synthetic Test",GIT_COMMITTER_EMAIL="synthetic@example.invalid")
    subprocess.check_output(["git","commit","-qm","Synthetic predecessor and producer fixture"],cwd=repo,env=env)
    commit=git(repo,"rev-parse","HEAD");git(repo,"remote","add","origin",str(remote));git(repo,"push","-u","origin","stage-d-d3-pass")
    git(repo,"update-ref",r.PRODUCER,commit);git(repo,"push","origin",r.PRODUCER)
    return repo,commit


def install_fixture_boundaries(monkeypatch,repo,commit):
    def history(target):
        rows=[r.git_binding(target,path,commit) for path,_ in r.PROTECTED]
        return {"bindings":rows,"bindings_digest":r.canonical(rows)}
    def snapshots(target):
        rows=[r.record(target/p,target) for p in r.DEPENDENCIES]
        return {"source_files":rows,"files_digest":r.canonical({"files":rows})}
    monkeypatch.setattr(r,"validate_history",history)
    monkeypatch.setattr(r,"source_snapshots",snapshots)
    monkeypatch.setattr(r,"check_refs",lambda *args,**kwargs:{})
    original_validate_cohort=r.validate_cohort
    monkeypatch.setattr(r,"validate_cohort",lambda target,raw_release=True:original_validate_cohort(target,False))
    return history(repo),snapshots(repo)


def test_full_synthetic_source_ledger_reconcile_seal_readonly_and_git_lfs_publication(synthetic_repo,monkeypatch):
    repo,commit=synthetic_repo;history,snapshot=install_fixture_boundaries(monkeypatch,repo,commit);root=repo/r.ROOT
    files=[r.record(repo/p,repo) for p in r.FILES+r.DEPENDENCIES]
    r.write(root/"completion/y2_r2_r2.pending.json","pending",{"started_unix_ns":1,"next_authorization":"NONE_WHILE_PENDING"},"PENDING")
    r.write(root/"producer_implementation_seal.json","implementation",{"producer_commit":commit,"producer_ref":r.PRODUCER,"source_files":files,"files_digest":r.canonical({"files":files})})
    checks=[{"name":"false_pass_synthetic_"+str(i),"status":"PASS"} for i in range(24)]
    r.write(root/"producer_preflight.json","preflight",{"tests":checks,"cpu_seconds":13.125,"wall_seconds":15.25,"measurement_window":"synthetic nonzero preflight","startup_commit":r.BASE,"source_files":files,"files_digest":r.canonical({"files":files})})
    r.write(root/"historical_immutable_bindings.json","bindings",history)
    r.write(root/"historical_source_snapshot.json","source_snapshot",snapshot)
    r.write(root/"prospective_amendment_binding.json","source_snapshot",{"source_files":[],"files_digest":r.canonical({"files":[]})})
    result=r.collect(repo);assert result["valid_clean_count"]==100
    rows=r.resource_observations(repo);shuffled=list(reversed(rows));assert r.canonical_bytes(r.canonical_ledger(rows))==r.canonical_bytes(r.validate_observations(repo,shuffled))
    r.seal(repo,{"checks":checks,"rejected_count":len(checks)})
    first=r.validate_completion(repo);assert r.validate_completion(repo)==first
    assert not list(root.rglob("*.npy")) and first["new_scientific_attempt_count"]==0
    git(repo,"add",str(r.ROOT),str(r.REPORT))
    env=dict(os.environ,GIT_AUTHOR_NAME="Synthetic Test",GIT_AUTHOR_EMAIL="synthetic@example.invalid",GIT_COMMITTER_NAME="Synthetic Test",GIT_COMMITTER_EMAIL="synthetic@example.invalid")
    subprocess.check_output(["git","commit","-qm","Synthetic completion"],cwd=repo,env=env)
    git(repo,"push","origin","stage-d-d3-pass");git(repo,"update-ref",r.PASS_REF,"HEAD");git(repo,"push","origin",r.PASS_REF)
    assert r.validate_completion(repo,publication=True)==first


@pytest.mark.parametrize("attack",["resource_value","resource_hash","duplicate_observation","omit_observation","count_rebind_twice","zero_impute_missing"])
def test_false_pass_resource_mutations(attack,synthetic_repo):
    repo,_=synthetic_repo;rows=r.resource_observations(repo)
    if attack=="resource_value":rows[0]["raw_value"]+=0.1
    elif attack=="resource_hash":rows[0]["source_records"][0]["sha256"]="f"*64
    elif attack=="duplicate_observation":rows.append(rows[0])
    elif attack=="omit_observation":rows.pop()
    elif attack=="count_rebind_twice":
        row=next(x for x in rows if not x["included"]);row["included"]=True;row["duplicate_of"]=None
    else:next(x for x in rows if x["raw_value"] is None)["raw_value"]=0
    with pytest.raises(r.RecoveryError):r.validate_observations(repo,rows)


@pytest.mark.parametrize("attack",["early90","duplicate_order","source_type","producing_stage","budget_relabel","gpu_rerun","array_copy","missing_array_validation","wrong_order","failed_unit"])
def test_false_pass_manifest_mutations(attack,synthetic_repo):
    repo,_=synthetic_repo;rows=copy.deepcopy(r.read_records(repo/r.ROOT))
    if attack=="early90":rows=rows[:90]
    elif attack=="duplicate_order":rows[-1]["selection_order"]=0
    elif attack=="source_type":rows[0]["source_type"]=r.old.SOURCE_TYPES[2]
    elif attack=="producing_stage":rows[0]["original_producing_stage"]="Y2-R2-R1"
    elif attack=="budget_relabel":rows[0]["status"]="BLOCKED_BUDGET"
    elif attack=="gpu_rerun":rows[0]["gpu_reexecuted"]=True
    elif attack=="array_copy":rows[0]["arrays_copied"]=True
    elif attack=="missing_array_validation":rows[0]["validated_array_count"]=0
    elif attack=="wrong_order":rows[-1]["selection_order"]=100
    else:rows[0]["status"]="FAILED_NUMERICAL"
    with pytest.raises(r.RecoveryError):r.manifest_counts(rows)


@pytest.mark.parametrize("attack",["array_sha","array_digest","dtype","shape","axes","finite","name"])
def test_false_pass_array_mutations(attack,tmp_path):
    array=np.ones((3,224,224),dtype="<f4");name="input_attribution"
    if attack=="dtype":array=array.astype(np.float64)
    if attack=="shape":array=array.reshape(3,112,448)
    if attack=="finite":array[0,0,0]=np.nan
    if attack=="name":name="not_frozen"
    path=tmp_path/(name+".npy");np.save(path,array,allow_pickle=False)
    row={**r.record(path,tmp_path),"canonical_digest":r.old.array_digest(array),"dtype":str(array.dtype),"shape":list(array.shape),"axes":["rgb","pixel_y","pixel_x"]}
    if attack=="array_sha":row["sha256"]="a"*64
    if attack=="array_digest":row["canonical_digest"]="b"*64
    if attack=="axes":row["axes"]=["wrong"]
    with pytest.raises(r.old.R2Error):r.old.validate_array_record(tmp_path,row)


@pytest.mark.parametrize("attack",["open_nested_payload","git_sha256","missing_next_authorization","new_attempt","gpu_execution","false_zero_count"])
def test_false_pass_schema_mutations(attack,synthetic_repo):
    repo,_=synthetic_repo;doc=copy.deepcopy(r.read(repo/r.ROOT/"completion/y2_r2_r2.json"))
    if attack=="open_nested_payload":doc["payload"]["report"]["extra"]={}
    elif attack=="git_sha256":
        doc=r.signed("implementation",{"producer_commit":"a"*40,"producer_ref":r.PRODUCER,"source_files":[],"files_digest":"b"*64});doc["payload"]["producer_commit"]="a"*64
    elif attack=="missing_next_authorization":doc["payload"].pop("next_authorization")
    elif attack=="new_attempt":doc["payload"]["new_scientific_attempt_count"]=1
    elif attack=="gpu_execution":doc["payload"]["gpu_execution"]=True
    else:
        doc["payload"]["valid_clean_count"]=0;doc["canonical_digest"]=r.canonical({k:v for k,v in doc.items() if k!="canonical_digest"})
        with pytest.raises(r.RecoveryError):r.validate_package(repo,doc)
        return
    doc["canonical_digest"]=r.canonical({k:v for k,v in doc.items() if k!="canonical_digest"})
    with pytest.raises((ValidationError,r.RecoveryError)):r.validate_document(doc)


@pytest.mark.parametrize("name",["torch","torchvision","timm","PIL","cv2"])
def test_false_pass_forbidden_import(name):
    r.runtime_guard()
    with pytest.raises(r.AccessError):importlib.import_module(name)


def test_false_pass_external_access_and_empirical_command():
    import socket
    r.runtime_guard()
    with pytest.raises(r.AccessError):socket.getaddrinfo("example.invalid",443)
    with pytest.raises(r.AccessError):subprocess.run([sys.executable,"-c","print(1)"])


def test_false_pass_history_git_bytes_and_archive_ref(synthetic_repo):
    repo,commit=synthetic_repo;path=repo/r.old.Y2_R2_ROOT/"access_audit.json";data=path.read_bytes()
    try:
        path.write_bytes(data+b" ")
        with pytest.raises(r.RecoveryError):r.git_binding(repo,r.old.Y2_R2_ROOT,commit)
    finally:path.write_bytes(data)
    git(repo,"update-ref",r.PASS_REF,commit)
    try:
        with pytest.raises(r.RecoveryError):r.validate_publication(repo)
    finally:git(repo,"update-ref",r.PASS_REF,"HEAD")


def test_false_pass_pending_and_inventory(synthetic_repo):
    repo,_=synthetic_repo;root=repo/r.ROOT;path=root/"completion/extra.pending.json"
    path.write_text("{}")
    try:
        with pytest.raises(r.RecoveryError):r.validate_package(repo,r.read(root/"completion/y2_r2_r2.json"))
    finally:path.unlink()


def test_false_pass_frozen_implementation_drift(synthetic_repo):
    repo,_=synthetic_repo;path=repo/r.SPEC;data=path.read_bytes()
    try:
        path.write_bytes(data+b" ")
        with pytest.raises(r.RecoveryError):r.verify_implementation(repo)
    finally:path.write_bytes(data)


@pytest.mark.parametrize("attack",["target","cohort","checkpoint","shard_owner","quadrature_ladder","not_first_pass"])
def test_false_pass_source_contract_mutations(attack,synthetic_repo,monkeypatch):
    repo,commit=synthetic_repo;install_fixture_boundaries(monkeypatch,repo,commit)
    path,_,_=r.source_terminal(repo,0);data=path.read_bytes();doc=r.read(path);q=doc["payload"]
    if attack=="target":q["target_identity"]["rule"]="post-hoc target"
    elif attack=="cohort":q["source_binding"]["cohort_digest"]="f"*64
    elif attack=="checkpoint":q["checkpoint_sha256"]="f"*64
    elif attack=="shard_owner":q["shard_id"]=1
    elif attack=="quadrature_ladder":q["quadrature_attempts"][0]["steps"]=256
    else:q["quadrature_attempts"][0]["all_contracts_pass"]=False
    doc["canonical_digest"]=r.canonical({k:v for k,v in doc.items() if k!="canonical_digest"})
    try:
        put(path,doc)
        with pytest.raises((r.RecoveryError,r.old.R2Error)):r.validate_source(repo,0)
    finally:path.write_bytes(data)


def test_false_pass_failure_completion_preserves_verified_count(tmp_path):
    root=tmp_path/r.ROOT
    r.write(root/"completion/y2_r2_r2.pending.json","pending",{"started_unix_ns":1,"next_authorization":"NONE_WHILE_PENDING"},"PENDING")
    rows=[]
    for order in (0,73,77):
        source_type=r.old.SOURCE_TYPES[2] if order==77 else r.old.SOURCE_TYPES[1] if order==73 else r.old.SOURCE_TYPES[0]
        stage="Y2-R2-R1" if order==77 else "Y2-R1"
        row={"selection_order":order,"source_object_id":f"source-{order}","class_id":order,"source_type":source_type,"original_producing_stage":stage,"validation_stage":r.STAGE,"evidence_mode":"REBIND_ONLY","status":"VALID","source_terminal":{"path":"source.json","bytes":1,"sha256":"a"*64},"source_terminal_status":"VALID","scientific_result":None,"array_records_digest":"b"*64,"attempt_records_digest":"c"*64,"validated_array_count":262,"validation_digest":"d"*64,"gpu_reexecuted":False,"arrays_copied":False,"scientific_scope_changed":False,"intervention_payload_retained":source_type!=r.old.SOURCE_TYPES[1]}
        r.write(root/f"units/unit-{order:03d}/validation.json","rebind",row,"VALID");rows.append(row)
    r.seal_failure(tmp_path,"synthetic",r.RecoveryError("forced failure"))
    completion=r.read(root/"completion/y2_r2_r2.json")
    assert completion["payload"]["valid_clean_count"]==3
    assert completion["payload"]["source_counts"]==dict(zip(r.old.SOURCE_TYPES,(1,1,1)))
