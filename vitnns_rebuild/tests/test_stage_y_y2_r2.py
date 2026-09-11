from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError
from vitnns import stage_y_y2_r2 as r
from vitnns import stage_y_y2_r2_worker as worker
from vitnns.stage_y_y2_r2_schema import artifact_schema, science_schema

REPO=Path(__file__).resolve().parents[1]


def module(name):
    s=importlib.util.spec_from_file_location(name,REPO/f"scripts/{name}.py")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def sample(schema):
    if "const" in schema:return copy.deepcopy(schema["const"])
    if "anyOf" in schema:return sample(schema["anyOf"][0])
    if "enum" in schema:return schema["enum"][0]
    kind=schema["type"]
    if kind=="object":return {k:sample(v) for k,v in schema["properties"].items() if k in schema["required"]}
    if kind=="array":return [sample(schema["items"]) for _ in range(schema.get("minItems",0))]
    if kind=="string":return "a"*64 if "{64}" in schema.get("pattern","") else "a"*40 if "{40}" in schema.get("pattern","") else "PASS"
    if kind=="boolean":return True
    if kind=="null":return None
    return schema.get("minimum",0)


def synthetic_science():
    from vitnns import stage_y_y1 as y1
    arrays={name:np.zeros(r.expected_shape(name),dtype="<f4") for name in r.EXPECTED_ARRAY_NAMES}
    science=sample(science_schema());science["closure"]=y1.validate_closure(arrays,0.0);science["selected_steps"]=128
    science["target_identity"]={"class_id":0,"rule":"clean predicted-class raw logit"}
    for endpoint in (science["ablations"],science["deletion_operator"]):
        for b,block in enumerate(endpoint["blocks"]):block["block"]=b
    q=[{"steps":128,"all_contracts_pass":True,"input_sum":0.0,"input_delta":0.0,"input_abs_error":0.0,"threshold":1e-4,"closure_status":"PASS","identity_max_abs_error":0.0,"wall_seconds":1.0,"array_digest":r.array_digest(arrays["input_attribution"])}]
    return science,q,arrays


def test_schema_serialization_and_science_contract():
    module("validate_stage_y_y2_r2").validate_schemas(REPO)


def test_all_schema_objects_are_recursively_closed():
    def walk(value):
        if isinstance(value,dict):
            kind=value.get("type")
            if kind=="object" or isinstance(kind,list) and "object" in kind:assert value.get("additionalProperties") is False
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(artifact_schema())


def test_synthetic_all_artifact_variants_roundtrip():
    for kind,schema in artifact_schema()["$defs"].items():
        payload=sample(schema)
        r.validate_signed(r.signed(kind,"PASS",payload))


def test_producer_40_character_git_commit_is_accepted():
    doc=r.signed("binding","PASS",{"producer_commit":"a"*40,"producer_ref":r.PRODUCER_REF,"source_files":[],"files_digest":r.canonical_digest({"files":[]})})
    r.validate_signed(doc)


def test_original_84_3_13_metadata_only():
    terms=r.terminal_by_order(REPO)
    assert sum(t["status"]=="VALID" for _,t in terms.values())==84
    assert sorted(o for o,(_,t) in terms.items() if t["status"]=="BLOCKED_BUDGET" and t["payload"]["array_records"])==list(r.COMPLETED_BUDGET_ORDERS)
    assert [row["selection_order"] for row in r.validate_missing_sources(REPO)]==list(r.MISSING_ORDERS)


def test_modulo_shards_all_locked_counts():
    for count in range(1,9):
        assert sorted(o for s in range(count) for o in r.shard_orders(s,count))==list(r.MISSING_ORDERS)


def test_launcher_injects_before_child_import():
    runner=module("run_stage_y_y2_r2")
    gate={"canonical_digest":"a"*64,"payload":{"devices":[{"physical_index":3}]}}
    env=runner.worker_env(REPO,gate,0)
    out=subprocess.check_output([str(r.PYTHON),"-c","import os,sys; assert 'torch' not in sys.modules; assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'; from vitnns.stage_y_y2_r2_worker import check_environment; check_environment(); assert 'torch' not in sys.modules; print('PASS')"],env=env,text=True)
    assert out.strip()=="PASS"


def test_synthetic_worker_output_to_independent_unit_validator(tmp_path,monkeypatch):
    science,quadrature,arrays=synthetic_science();order=77;shard=0;unit=tmp_path/r.ROOT/f"shards/shard-{shard:02d}/units/unit-{order:03d}"
    cohort={order:{"selection_order":order,"source_object_id":"synthetic-source","class_id":0,"file_sha256":"b"*64}}
    monkeypatch.setattr(r,"cohort_by_order",lambda repo:cohort)
    old=tmp_path/"original_terminal.json";old.write_text("{}")
    monkeypatch.setattr(r,"terminal_by_order",lambda repo:{order:(old,{})})
    r.write_json_new(unit/"completion/unit.pending.json",r.signed("unit_pending","PENDING",{"selection_order":order,"source_object_id":"synthetic-source","attempt_id":"unit-077-scientific-001","retry_index":1}))
    resource={"cpu_seconds":999999999.,"gpu_active_seconds":999999999.,"wall_seconds":999999999.,"peak_reserved_gpu_bytes":50*1024**3,"retained_bytes":0}
    terminal=worker.publish_success(tmp_path,unit,order,shard,"e"*64,science,quadrature,arrays,resource)
    assert terminal["status"]=="VALID" and not (unit/"completion/unit.pending.json").exists()
    assert terminal["payload"]["attempt_id"]=="unit-077-scientific-001"
    monkeypatch.setattr(r,"validate_source_binding",lambda *args:None)
    identity=unit.parents[1]/"worker_identity.json";identity.write_text(json.dumps({"payload":{"physical_index":0,"gpu_uuid":"synthetic","pci_bus_id":"synthetic","environment_digest":"e"*64}}))
    gate={"payload":{"locked_worker_count":1,"devices":[{"physical_index":0,"gpu_uuid":"synthetic","pci_bus_id":"synthetic","environment_digest":"e"*64}]}}
    record=module("validate_stage_y_y2_r2").validate_computed(tmp_path,unit/"completion/unit.json",gate)
    assert record["status"]=="VALID" and record["selection_order"]==order
    with pytest.raises(Exception):worker.publish_success(tmp_path,unit,order,shard,"e"*64,science,quadrature,arrays,resource)


def unified_records():
    return [{"selection_order":o,"status":"VALID","source_type":r.SOURCE_TYPES[2] if o in r.MISSING_ORDERS else r.SOURCE_TYPES[1] if o in r.COMPLETED_BUDGET_ORDERS else r.SOURCE_TYPES[0]} for o in range(100)]


@pytest.mark.parametrize("attack",["omit_13th_missing","pass_at_90","duplicate_order","replace_sample","blocked_as_valid","rerun_existing_valid","rerun_completed_budget","wrong_completed_candidate"])
def test_false_pass_mutation_unified(attack):
    rows=unified_records()
    if attack=="omit_13th_missing":rows=[row for row in rows if row["selection_order"]!=98]
    elif attack=="pass_at_90":rows=rows[:90]
    elif attack=="duplicate_order":rows[99]["selection_order"]=98
    elif attack=="replace_sample":rows[99]["selection_order"]=100
    elif attack=="blocked_as_valid":rows[77]["status"]="BLOCKED_INFRASTRUCTURE"
    elif attack=="rerun_existing_valid":rows[0]["source_type"]=r.SOURCE_TYPES[2]
    elif attack=="rerun_completed_budget":rows[73]["source_type"]=r.SOURCE_TYPES[2]
    else:rows[73]["source_type"],rows[72]["source_type"]=rows[72]["source_type"],rows[73]["source_type"]
    with pytest.raises(r.R2Error):module("validate_stage_y_y2_r2").validate_unified(rows)


@pytest.mark.parametrize("attack",["commit_as_sha256","unknown_nested_field","wrong_stage","wrong_digest","missing_attempt_id","late_variable","wrong_variable"])
def test_false_pass_mutation_schema_environment(attack,monkeypatch):
    if attack in ("late_variable","wrong_variable"):
        if attack=="late_variable":monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG",raising=False)
        else:monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":16:8")
        with pytest.raises(r.StartupError):worker.check_environment()
        return
    doc=r.signed("binding","PASS",{"producer_commit":"a"*40,"producer_ref":r.PRODUCER_REF,"source_files":[],"files_digest":"b"*64})
    if attack=="commit_as_sha256":doc["payload"]["producer_commit"]="a"*64
    elif attack=="unknown_nested_field":doc["payload"]["source_files"]=[{"path":"x","bytes":1,"sha256":"a"*64,"open_payload":{}}]
    elif attack=="wrong_stage":doc["stage_id"]="Y2-R2"
    elif attack=="missing_attempt_id":
        doc=r.signed("attempt","PASS",{"attempt_id":"x","selection_order":77,"retry_index":1,"quadrature_attempts":[]});doc["payload"].pop("attempt_id")
    doc["canonical_digest"]=r.canonical_digest({k:v for k,v in doc.items() if k!="canonical_digest"})
    if attack=="wrong_digest":doc["canonical_digest"]="f"*64
    with pytest.raises((r.R2Error,ValidationError)):r.validate_signed(doc)


@pytest.mark.parametrize("attack",["file_sha","canonical_digest","dtype","shape","axes","nonfinite","unexpected_array"])
def test_false_pass_mutation_array(attack,tmp_path):
    name="input_attribution";array=np.zeros((3,224,224),dtype="<f4")
    if attack=="dtype":array=array.astype(np.float64)
    if attack=="nonfinite":array[0,0,0]=np.nan
    if attack=="shape":array=array.reshape(3,112,448)
    if attack=="unexpected_array":name="made_up"
    path=tmp_path/(name+".npy");np.save(path,array,allow_pickle=False)
    row={**r.file_record(path,tmp_path),"canonical_digest":r.array_digest(array),"dtype":str(array.dtype),"shape":list(array.shape),"axes":["rgb","pixel_y","pixel_x"]}
    if attack=="file_sha":row["sha256"]="a"*64
    elif attack=="canonical_digest":row["canonical_digest"]="a"*64
    elif attack=="axes":row["axes"]=["wrong"]
    with pytest.raises(r.R2Error):r.validate_array_record(tmp_path,row)


@pytest.mark.parametrize("attack",["changed_threshold","skipped_quadrature","extra_retry","wrong_attempt_reference"])
def test_false_pass_mutation_attempt(attack,tmp_path):
    unit=tmp_path/"unit";science,q,_=synthetic_science()
    if attack=="changed_threshold":q[0]["threshold"]=1.
    if attack=="skipped_quadrature":q[0]["steps"]=256
    doc=r.signed("attempt","PASS",{"attempt_id":"x","selection_order":77,"retry_index":1,"quadrature_attempts":q})
    r.write_json_new(unit/"attempts/attempt-001.json",doc)
    records=[r.file_record(unit/"attempts/attempt-001.json",tmp_path)]
    if attack=="extra_retry":
        r.write_json_new(unit/"attempts/attempt-002.json",doc);records.append(r.file_record(unit/"attempts/attempt-002.json",tmp_path))
    t={"stage_id":r.STAGE_ID,"payload":{"attempt_id":"y" if attack=="wrong_attempt_reference" else "x","attempt_records":records,"quadrature_attempts":q}}
    with pytest.raises(r.R2Error):r.validate_attempts(tmp_path,t,unit)


def test_resource_sum_includes_initialization_and_prior_attempts():
    rows=[{"cpu_seconds":3600,"gpu_active_seconds":1800,"wall_seconds":100,"retained_bytes":10,"peak_reserved_gpu_bytes":10},{"cpu_seconds":7200,"gpu_active_seconds":3600,"wall_seconds":200,"retained_bytes":20,"peak_reserved_gpu_bytes":20}]
    total=r.resource_sum(rows);assert total["cpu_hours"]==3 and total["h20_gpu_hours"]==1.5 and total["retained_bytes"]==30


def test_predecessor_science_json_schemas_without_array_access():
    validator=Draft202012Validator(science_schema())
    for path in (REPO/r.Y2_R1_ROOT).glob("shards/shard-*/units/unit-*/scientific_result.json"):
        doc=r.read_json(path);r.validate_canonical(doc);validator.validate(doc["payload"])
    for root in (r.Y2_ROOT,r.Y2_R1_ROOT):
        module("validate_stage_y_y2_r2")._historical_resources(REPO,root)


def test_synthetic_reconcile_to_completion_and_publication_validation(tmp_path,monkeypatch):
    validator=module("validate_stage_y_y2_r2");root=tmp_path/r.ROOT
    payload=sample(artifact_schema()["$defs"]["reconciliation"])
    for i,row in enumerate(payload["records"]):row["selection_order"]=i
    rec=r.signed("reconciliation","PASS",payload)
    r.write_json_new(root/"reconciliation.json",rec)
    r.write_json_new(root/"completion/y2_r2_r1.pending.json",r.signed("pending","PENDING",{"started_at":"synthetic","started_unix_ns":1,"next_authorization":"NONE_WHILE_PENDING"}))
    ledger=sample(artifact_schema()["$defs"]["resource_ledger"])
    for name in ("resource_ledger","cumulative_resource_ledger"):
        r.write_json_new(root/(name+".json"),r.signed("resource_ledger","PASS",ledger))
    monkeypatch.setattr(validator,"reconcile",lambda *args:rec)
    monkeypatch.setattr(validator,"resource_ledger",lambda *args:ledger)
    monkeypatch.setattr(validator,"false_pass_audit",lambda *args:{})
    assert validator.seal(tmp_path)["status"]=="PASS"
    assert validator.validate(tmp_path)["status"]=="PASS"


def test_completion_and_inventory_schema_roundtrip(tmp_path):
    root=tmp_path/r.ROOT
    r.write_json_new(root/"completion/y2_r2_r1.pending.json",r.signed("pending","PENDING",{"started_at":"synthetic","started_unix_ns":1,"next_authorization":"NONE_WHILE_PENDING"}))
    inv=r.whole_root_inventory(root,"PASS");r.write_json_new(root/"whole_root_inventory.json",inv)
    record=r.file_record(root/"whole_root_inventory.json",tmp_path)
    doc=r.signed("completion","PASS",{"completed_at":"synthetic","denominator":100,"valid_clean_count":100,"source_counts":dict(zip(r.SOURCE_TYPES,(84,3,13))),"formal_outcome_access":False,"next_authorization":r.NEXT,"reconciliation":record,"validation_report":record,"whole_root_inventory":record,"report":record,"failure":None})
    r.publish_terminal(root/"completion/y2_r2_r1.pending.json",root/"completion/y2_r2_r1.json",doc)
    assert r.whole_root_inventory(root,"PASS")==inv
