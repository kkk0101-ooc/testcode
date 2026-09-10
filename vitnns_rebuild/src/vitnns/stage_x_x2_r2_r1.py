"""Prospective X2-R2-R1 control-initialization recovery control plane."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_x_x2_r1 as r1
from . import stage_x_x2_r1_worker as r1_worker
from . import stage_x_x2_r2 as r2

RUN_ID="STAGE_X_X2_R2_R1_V1_20260909"
ROOT=Path("evidence/stage_x/x2_r2_r1_nns_v2_development_20260909/formal")
REPORT_ROOT=Path("reports/stage_x_x2_r2_r1_nns_v2_development_20260909")
CONFIG=Path("config/stage_x_x2_r2_r1.json")
CONFIG_SCHEMA=Path("config/stage_x_x2_r2_r1.schema.json")
ARTIFACT_SCHEMA=Path("config/stage_x_x2_r2_r1_artifact.schema.json")
AMENDMENT=Path("docs/stage_x_x2_r2_r1_control_initialization_recovery_amendment_20260909.md")
R1_ROOT=r1.ROOT
R1_STATUS=R1_ROOT/"stage_failure.json"
R2_ROOT=r2.ROOT
R2_STATUS=R2_ROOT/"stage_failure.json"
R2_IMPLEMENTATION_SEAL=R2_ROOT/"implementation_seal.json"
R2_MUTATION_RECEIPT=R2_ROOT/"pre_outcome/mutation_suite_v2.json"
MUTATION_RECEIPT=ROOT/"pre_outcome/mutation_suite.json"
RUN_SPEC=ROOT/"x2_r2_r1_run_spec.json"
PREFLIGHT_RECEIPT=ROOT/"preflight_worker_clean.json"
MODEL_BINDING=ROOT/"model_materialization_binding.json"
UNIT_CSV=ROOT/"manifests/development_units.csv"
SAMPLE_CSV=ROOT/"manifests/development_samples.csv"
REUSE_CSV=ROOT/"manifests/reuse_clean_routes.csv"
NEW_CSV=ROOT/"manifests/new_clean_routes.csv"
REQUIRED_FILES_JSON=ROOT/"worker_required_files_clean.json"
UNIT_FIELDS=r1.UNIT_FIELDS
SAMPLE_FIELDS=r1.SAMPLE_FIELDS
REUSE_FIELDS=("unit_digest","r1_unit_digest","model_id","sample_id","sample_sha256","r1_terminal_path","r1_terminal_sha256","r1_terminal_canonical_digest","r1_tree_digest","r1_selected_steps","checkpoint_path","checkpoint_sha256","array_count","array_binding_digest","cut_pass","ig_identity_pass","reuse_status")
NEW_FIELDS=("unit_digest","r1_unit_digest","model_id","sample_id","sample_sha256","development_rank","shard","r1_state","r1_source_path")
ZERO_RESOURCES=r1.ZERO_RESOURCES
MODEL_LOCAL_PATHS=r1.MODEL_LOCAL_PATHS
OLD_CONFIG=r1.OLD_CONFIG
_load_state=r1._load_state
IMPLEMENTATION_FILES=(
    "src/vitnns/stage_x_x2_r2_r1.py",
    "src/vitnns/stage_x_x2_r2_r1_worker.py",
    "src/vitnns/stage_x_x2_r2_r1_reconcile.py",
    "src/vitnns/stage_x_x2_metrics.py",
    "src/vitnns/stage_x_nns_v2.py",
    "src/vitnns/stage_x_comparators.py",
    "config/stage_x_x2_r2_r1.json",
    "config/stage_x_x2_r2_r1.schema.json",
    "config/stage_x_x2_r2_r1_artifact.schema.json",
    "docs/stage_x_x2_r2_r1_control_initialization_recovery_amendment_20260909.md",
    "tests/test_stage_x_x2_r2_r1.py",
)


class X2R2R1Error(ValueError):pass


def strict_read(path:Path)->dict[str,Any]:return r1.strict_read(path)
def sha_file(path:Path)->str:return r1.sha_file(path)
def scalar_rows(value:Mapping[str,Any])->list[dict[str,Any]]:return r1.scalar_rows(value)
def terminal_state(**kwargs:Any)->dict[str,Any]:return r1.terminal_state(**kwargs)
def payload(**kwargs:Any)->dict[str,Any]:return r1.payload(**kwargs)
def file_record(repo:Path,relative:Path,*,canonical_digest:str|None=None)->dict[str,Any]:return r1.file_record(repo,relative,canonical_digest=canonical_digest)
def write_csv_new(path:Path,fields:Sequence[str],rows:Sequence[Mapping[str,Any]])->None:return r1.write_csv_new(path,fields,rows)
def write_bytes_new(path:Path,data:bytes)->None:return r1.write_bytes_new(path,data)
def load_samples(repo:Path)->list[dict[str,Any]]:return _load_samples(repo)


def _load_samples(repo:Path)->list[dict[str,Any]]:
    manifest=strict_read(repo/ROOT/"development_sample_manifest.json");validate_artifact(manifest,repo,verify_files=True);rows=r1.read_csv(repo/SAMPLE_CSV,SAMPLE_FIELDS)
    for row in rows:row["development_rank"]=int(row["development_rank"]);row["file_bytes"]=int(row["file_bytes"])
    if len(rows)!=40 or len({row["source_object_id"] for row in rows})!=40:raise X2R2R1Error("R2-R1 sample roster drifted")
    return rows


def load_units(repo:Path)->list[dict[str,Any]]:
    manifest=strict_read(repo/ROOT/"development_unit_manifest.json");validate_artifact(manifest,repo,verify_files=True);samples={row["source_object_id"]:row for row in _load_samples(repo)};models={row["model_id"] for row in strict_read(repo/OLD_CONFIG)["models"]};rows=r1.read_csv(repo/UNIT_CSV,UNIT_FIELDS)
    for row in rows:
        row["development_rank"]=int(row["development_rank"]);row["shard"]=int(row["shard"]);r1.validate_unit_row(row,samples,models) if row["run_id"]==r1.RUN_ID else _validate_r2_unit(row,samples,models)
    if len(rows)!=200 or len({row["unit_digest"] for row in rows})!=200:raise X2R2R1Error("R2-R1 unit roster drifted")
    return rows


def _validate_r2_unit(row:Mapping[str,Any],samples:Mapping[str,Mapping[str,Any]],models:set[str])->None:
    if row["run_id"]!=RUN_ID or row["model_id"] not in models or row["sample_id"] not in samples:raise X2R2R1Error("R2-R1 unit binding drifted")
    sample=samples[row["sample_id"]];identity={"run_id":RUN_ID,"model_id":row["model_id"],"sample_id":row["sample_id"],"sample_sha256":row["sample_sha256"],"development_rank":int(row["development_rank"])};digest=canonical_sha256(identity)
    if row["sample_sha256"]!=sample["file_sha256"] or int(row["development_rank"])!=int(sample["development_rank"]) or row["unit_digest"]!=digest or int(row["shard"])!=int.from_bytes(bytes.fromhex(digest)[:8],"big")%4:raise X2R2R1Error("R2-R1 unit digest/sample/shard drifted")


def publish_atomic(pending:Path,terminal:Path,value:Mapping[str,Any],repo:Path)->None:
    if pending.exists() or terminal.exists():raise X2R2R1Error("pending or terminal exists")
    pending_body=copy.deepcopy(value["payload"]);pending_body["terminal"]=terminal_state();write_new(pending,signed(value["document_type"],"PENDING",pending_body),repo);temporary=terminal.with_name(terminal.name+".terminal.pending");write_new(temporary,value,repo);os.replace(temporary,terminal);pending.unlink()


def signed(document_type:str,status:str,body:Mapping[str,Any])->dict[str,Any]:
    if document_type.startswith("stage_x_x2_r2_r1"):
        normalized=document_type
    elif document_type.startswith("stage_x_x2_r2"):
        normalized=document_type.replace("stage_x_x2_r2","stage_x_x2_r2_r1",1)
    elif document_type.startswith("stage_x_x2_r1"):
        normalized=document_type.replace("stage_x_x2_r1","stage_x_x2_r2_r1",1)
    else:
        normalized=document_type
    value={"$schema":ARTIFACT_SCHEMA.name,"schema_version":1,"document_type":normalized,"stage_id":"X2-R2-R1","run_id":RUN_ID,"status":status,"payload":dict(body)};value["canonical_digest"]=canonical_sha256(value);return value


def load_config(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();value=strict_read(repo/CONFIG);schema=strict_read(repo/CONFIG_SCHEMA);Draft202012Validator.check_schema(schema);errors=sorted(Draft202012Validator(schema).iter_errors(value),key=lambda error:tuple(str(part) for part in error.absolute_path))
    if errors:raise X2R2R1Error(errors[0].message)
    plain=dict(value);observed=plain.pop("canonical_digest");
    if observed!=canonical_sha256(plain):raise X2R2R1Error("X2-R2-R1 config digest drifted")
    return value


def validate_coverage_states(rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    if len(rows)!=200 or len({row.get("unit_digest") for row in rows})!=200:raise X2R2R1Error("coverage requires exactly 200 unique routes")
    allowed={"VALID","FAILED_ROUTE_LOCAL_NUMERICAL","FAILED_ROUTE_LOCAL_EXECUTION","NA_ROUTE_LOCAL_INFRASTRUCTURE"}
    if any(row.get("status") not in allowed for row in rows):raise X2R2R1Error("coverage contains untyped, interrupted, or not-started state")
    totals=Counter(row["model_id"] for row in rows);valid=Counter(row["model_id"] for row in rows if row["status"]=="VALID")
    if len(totals)!=5 or set(totals.values())!={40}:raise X2R2R1Error("coverage model denominator drifted")
    return {"expected":200,"valid":sum(valid.values()),"per_model":dict(sorted(valid.items())),"pass":sum(valid.values())>=190 and all(valid[model]>=38 for model in totals)}


def validate_checkpoint_record(repo: Path, record: Mapping[str, Any]) -> None:
    path=repo/str(record["path"])
    if not path.is_file() or sha_file(path)!=record["sha256"]:
        raise X2R2R1Error("checkpoint digest drift")


def _validate_file_record(repo: Path, record: Mapping[str, Any]) -> None:
    path=repo/str(record["path"])
    if not path.is_file() or path.stat().st_size!=record["bytes"] or sha_file(path)!=record["sha256"]:
        raise X2R2R1Error(f"file record drift: {path}")
    if record["kind"]=="array":
        if r1.array_record(repo,Path(str(record["path"])))!=record:raise X2R2R1Error(f"array record drift: {path}")
    elif record["canonical_digest"] is not None:
        if path.suffix==".json":
            value=strict_read(path)
            if value.get("$schema")==ARTIFACT_SCHEMA.name:validate_artifact(value,repo)
            elif value.get("$schema")==r2.ARTIFACT_SCHEMA.name:r2.validate_artifact(value,repo)
            elif value.get("$schema")==r1.ARTIFACT_SCHEMA.name:r1.validate_artifact(value,repo)
            else:
                plain=dict(value);observed=plain.pop("canonical_digest",None)
                if observed!=canonical_sha256(plain):raise X2R2R1Error(f"document digest drift: {path}")
            observed=value["canonical_digest"]
        elif path.name in {"development_samples.csv"}:
            rows=r1.read_csv(path,SAMPLE_FIELDS)
            for row in rows:row["development_rank"]=int(row["development_rank"]);row["file_bytes"]=int(row["file_bytes"])
            observed=canonical_sha256(rows)
        elif path.name in {"development_units.csv"}:
            rows=r1.read_csv(path,UNIT_FIELDS)
            for row in rows:row["development_rank"]=int(row["development_rank"]);row["shard"]=int(row["shard"])
            observed=canonical_sha256(rows)
        elif path.suffix==".csv":
            with path.open("r",encoding="utf-8",newline="") as stream:observed=canonical_sha256([dict(row) for row in csv.DictReader(stream)])
        else:raise X2R2R1Error(f"unsupported canonical file record: {path}")
        if observed!=record["canonical_digest"]:raise X2R2R1Error(f"document digest drift: {path}")


def validate_file_record(repo: Path, record: Mapping[str, Any]) -> None:
    _validate_file_record(repo,record)


def validate_artifact(value:Mapping[str,Any],repo_root:Path|str=REPO_ROOT,*,verify_files:bool=False)->None:
    repo=Path(repo_root).resolve();schema=strict_read(repo/ARTIFACT_SCHEMA);errors=sorted(Draft202012Validator(schema).iter_errors(value),key=lambda error:tuple(str(part) for part in error.absolute_path))
    if errors:raise X2R2R1Error(errors[0].message)
    plain=dict(value);observed=plain.pop("canonical_digest",None)
    if observed!=canonical_sha256(plain):raise X2R2R1Error("X2-R2-R1 artifact digest drifted")
    body=value["payload"]
    for field,key in (("identity","name"),("counts","name"),("state","name"),("checks","check_id"),("metrics","metric_id"),("bindings","path"),("files","path")):
        rows=body[field];keys=[row[key] for row in rows]
        if len(keys)!=len(set(keys)):raise X2R2R1Error(f"duplicate {field}")
    if value["document_type"] in {"stage_x_x2_r2_r1_run_spec","stage_x_x2_r2_r1_gamma_selection","stage_x_x2_r2_r1_completion"}:
        state={row["name"]:row["value"] for row in body["state"]}
        if state.get("holm_family_count")!=5 or state.get("full_model_ig_in_holm") is not False:raise X2R2R1Error("Holm family drifted")
    if verify_files:
        for record in [*body["bindings"],*body["files"]]:_validate_file_record(repo,record)


def write_new(path:Path,value:Mapping[str,Any],repo_root:Path|str=REPO_ROOT)->None:
    validate_artifact(value,repo_root);path.parent.mkdir(parents=True,exist_ok=True);descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(descriptor,"wb") as stream:stream.write(canonical_json_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())


def artifact_record(repo:Path,relative:Path)->dict[str,Any]:
    value=strict_read(repo/relative);validate_artifact(value,repo);return file_record(repo,relative,canonical_digest=value["canonical_digest"])


def _tree_digest(path:Path)->str:
    rows=[(item.relative_to(path).as_posix(),item.stat().st_size,sha_file(item)) for item in sorted(path.rglob("*")) if item.is_file()];return canonical_sha256(rows)


def validate_r1_clean(repo:Path,r2_unit:Mapping[str,Any],r1_unit:Mapping[str,Any])->dict[str,Any]:
    base=repo/R1_ROOT/"units"/r1_unit["unit_digest"]/"attempts/attempt-000001/clean";terminal=strict_read(base/"terminal.json");r1.validate_artifact(terminal,repo,verify_files=True);summary=r1_worker._read_one_csv(base/"summary.csv")
    if terminal["status"]!="PASS" or summary["unit_digest"]!=r1_unit["unit_digest"] or summary["sample_sha256"]!=r2_unit["sample_sha256"] or summary["model_id"]!=r2_unit["model_id"] or int(summary["complete_cut_count"])!=135 or summary["complete_cut_all_pass"]!="True" or float(summary["identity_max_abs_error"])>1e-6:raise X2R2R1Error("R1 clean route cannot be reused")
    checkpoint=MODEL_LOCAL_PATHS[next(row["id"] for row in strict_read(repo/OLD_CONFIG)["models"] if row["model_id"]==r2_unit["model_id"])]
    arrays=[row for row in terminal["payload"]["files"] if row["kind"]=="array"]
    for record in arrays:_validate_file_record(repo,record)
    return {"unit_digest":r2_unit["unit_digest"],"r1_unit_digest":r1_unit["unit_digest"],"model_id":r2_unit["model_id"],"sample_id":r2_unit["sample_id"],"sample_sha256":r2_unit["sample_sha256"],"r1_terminal_path":(base/"terminal.json").relative_to(repo).as_posix(),"r1_terminal_sha256":sha_file(base/"terminal.json"),"r1_terminal_canonical_digest":terminal["canonical_digest"],"r1_tree_digest":_tree_digest(base),"r1_selected_steps":int(summary["selected_steps"]),"reuse_status":"SOURCE_VALIDATED_VALID","checkpoint_path":checkpoint.as_posix(),"checkpoint_sha256":sha_file(repo/checkpoint),"array_count":len(arrays),"array_binding_digest":canonical_sha256(arrays),"cut_pass":True,"ig_identity_pass":True}


def run_mutation_suite(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=load_config(repo)
    if (repo/MUTATION_RECEIPT).exists():value=strict_read(repo/MUTATION_RECEIPT);validate_artifact(value,repo,verify_files=True);return value
    executed=list((repo/ROOT/"units").glob("*/attempts/attempt-000001/journal/*.json")) if (repo/ROOT/"units").exists() else []
    actual_terminals=list((repo/ROOT/"units").glob("*/terminal.json")) if (repo/ROOT/"units").exists() else []
    if executed or actual_terminals:raise X2R2R1Error("cannot replace mutation suite after new R2-R1 outcome access")
    checks=[];base=signed("stage_x_x2_r2_r1_phase_manifest","PASS",payload(state={"holm_family_count":5,"full_model_ig_in_holm":False}))
    def reject(name:str,fn:Any)->None:
        try:fn()
        except Exception as error:checks.append({"check_id":name,"status":"PASS","detail":type(error).__name__})
        else:raise X2R2R1Error(f"mutation accepted: {name}")
    def mutate(fn:Any)->None:
        value=copy.deepcopy(base);fn(value);plain=dict(value);plain.pop("canonical_digest",None);value["canonical_digest"]=canonical_sha256(plain);validate_artifact(value,repo)
    reject("root_unknown_field",lambda:mutate(lambda value:value.__setitem__("UNKNOWN",True)));reject("payload_unknown_field",lambda:mutate(lambda value:value["payload"].__setitem__("UNKNOWN",True)));reject("nested_unknown_field",lambda:mutate(lambda value:value["payload"]["terminal"].__setitem__("UNKNOWN",True)));reject("missing_required_field",lambda:mutate(lambda value:value["payload"].pop("resources")));reject("duplicate_json_key",lambda:r1.strict_loads('{"a":1,"a":2}'));reject("nan",lambda:r1.strict_loads('{"a":NaN}'));reject("infinity",lambda:r1.strict_loads('{"a":Infinity}'));bad=copy.deepcopy(base);bad["canonical_digest"]="0"*64;reject("bad_canonical_digest",lambda:validate_artifact(bad,repo));reject("bool_as_int",lambda:mutate(lambda value:value["payload"]["metrics"].append({"metric_id":"bad","value":True,"unit":"count"})))
    models=[f"m{index}" for index in range(5)];good=[{"unit_digest":hashlib.sha256(f"{model}-{index}".encode()).hexdigest(),"model_id":model,"status":"VALID"} for model in models for index in range(40)]
    reject("expected_route_count_not_200",lambda:validate_coverage_states(good[:-1]));bad_overall=copy.deepcopy(good)
    for index in range(11):bad_overall[index]["status"]="FAILED_ROUTE_LOCAL_NUMERICAL"
    result=validate_coverage_states(bad_overall)
    if result["pass"]:raise X2R2R1Error("overall threshold below 190 accepted")
    checks.append({"check_id":"overall_threshold_not_190","status":"PASS","detail":"coverage_rejected"});bad_model=copy.deepcopy(good)
    for index in range(3):bad_model[index]["status"]="FAILED_ROUTE_LOCAL_NUMERICAL"
    result=validate_coverage_states(bad_model)
    if result["pass"]:raise X2R2R1Error("per-model threshold below 38 accepted")
    checks.append({"check_id":"per_model_threshold_not_38","status":"PASS","detail":"coverage_rejected"})
    bad_denominator=copy.deepcopy(good);bad_denominator=bad_denominator[:-1]
    reject("model_denominator_not_40",lambda:validate_coverage_states(bad_denominator))
    for name,status in (("not_started_as_terminal","NOT_STARTED"),("interrupted_as_terminal","INTERRUPTED")):
        mutated=copy.deepcopy(good);mutated[0]["status"]=status;reject(name,lambda mutated=mutated:validate_coverage_states(mutated))
    sample={"development_rank":0,"file_sha256":"a"*64};identity={"run_id":RUN_ID,"model_id":"m","sample_id":"s","sample_sha256":"a"*64,"development_rank":0};digest=canonical_sha256(identity);unit={**identity,"unit_digest":digest,"shard":int.from_bytes(bytes.fromhex(digest)[:8],"big")%4}
    reject("sample_replacement",lambda:_validate_r2_unit({**unit,"sample_id":"other"},{"s":sample},{"m"}))
    reject("wrong_r1_r2_unit_mapping",lambda:_validate_r2_unit({**unit,"run_id":r1.RUN_ID},{"s":sample},{"m"}))
    reject("wrong_unit_digest",lambda:_validate_r2_unit({**unit,"unit_digest":"b"*64},{"s":sample},{"m"}))
    reject("wrong_sample_digest",lambda:_validate_r2_unit({**unit,"sample_sha256":"b"*64},{"s":sample},{"m"}))
    checkpoint=MODEL_LOCAL_PATHS[next(row["id"] for row in strict_read(repo/OLD_CONFIG)["models"] if row["model_id"]=="vit_base_patch32_224.augreg_in21k_ft_in1k")]
    checkpoint_record={"path":checkpoint.as_posix(),"sha256":sha_file(repo/checkpoint)}
    reject("wrong_checkpoint_digest",lambda:validate_checkpoint_record(repo,{**checkpoint_record,"sha256":"0"*64}))
    failed_old=next(item for item in r1.load_units(repo) if (repo/R1_ROOT/"units"/item["unit_digest"]/"terminal.json").is_file())
    failed_identity={"run_id":RUN_ID,"model_id":failed_old["model_id"],"sample_id":failed_old["sample_id"],"sample_sha256":failed_old["sample_sha256"],"development_rank":int(failed_old["development_rank"])}
    failed_r2={**failed_identity,"unit_digest":canonical_sha256(failed_identity),"shard":0}
    reject("failed_route_relabelled_valid",lambda:validate_r1_clean(repo,failed_r2,failed_old))
    reject("full_model_ig_in_holm",lambda:validate_artifact(signed("stage_x_x2_r2_r1_run_spec","PASS",payload(state={"holm_family_count":5,"full_model_ig_in_holm":True})),repo));reject("holm_family_not_five",lambda:validate_artifact(signed("stage_x_x2_r2_r1_run_spec","PASS",payload(state={"holm_family_count":6,"full_model_ig_in_holm":False})),repo))
    with tempfile.TemporaryDirectory(prefix="x2-r2-mut-") as name:
        temp=Path(name);one=temp/"one.json";write_new(one,base,repo);reject("terminal_overwrite",lambda:write_new(one,base,repo));pending=temp/"pending.json";terminal=temp/"terminal.json";publish_atomic(pending,terminal,base,repo);reject("pending_overwrite",lambda:publish_atomic(pending,terminal,base,repo))
        array=temp/"value.npy";np.save(array,np.zeros((2,3),dtype=np.float32),allow_pickle=False);record={"bytes":array.stat().st_size,"sha256":sha_file(array),"dtype":"float32","shape":[2,3]}
        def check_array(changed:Mapping[str,Any])->None:
            value=np.load(array,allow_pickle=False);actual={"bytes":array.stat().st_size,"sha256":sha_file(array),"dtype":str(value.dtype),"shape":list(value.shape)}
            if dict(changed)!=actual:raise X2R2R1Error("array drift")
        reject("array_dtype_shape_hash_digest_drift",lambda:check_array({**record,"shape":[3,2]}))
    body=payload(identity={"suite":"X2_R2_R1_PRE_OUTCOME_MUTATION_SUITE"},bindings=[file_record(repo,ARTIFACT_SCHEMA),file_record(repo,CONFIG,canonical_digest=config["canonical_digest"]),file_record(repo,AMENDMENT),file_record(repo,R2_MUTATION_RECEIPT,canonical_digest=strict_read(repo/R2_MUTATION_RECEIPT)["canonical_digest"])],checks=checks,counts={"attack_count":len(checks),"rejected_count":len(checks)},state={"outcome_access_before_suite":False,"all_mutations_rejected":True,"real_validators_used":True},terminal=terminal_state(terminal=True,immutable=True,complete=True));receipt=signed("stage_x_x2_r2_r1_pre_outcome_mutation_receipt","PASS",body);write_new(repo/MUTATION_RECEIPT,receipt,repo);return receipt


def _fresh_root_precheck(repo: Path) -> None:
    if (repo/ROOT).exists() or (repo/REPORT_ROOT).exists():
        raise X2R2R1Error("fresh X2-R2-R1 roots must not exist before initialization")
    x3_to_x5=[path for path in (repo/"evidence/stage_x").glob("x[3-5]*") if path.exists()]
    if x3_to_x5:raise X2R2R1Error("X3-X5 root exists before authorization")


def _recompute_r2_blocked(repo: Path) -> dict[str, Any]:
    stage=r2.strict_read(repo/R2_STATUS);r2.validate_artifact(stage,repo,verify_files=True)
    seal=r2.strict_read(repo/R2_IMPLEMENTATION_SEAL);r2.validate_artifact(seal,repo,verify_files=True)
    terminals=list((repo/R2_ROOT/"units").glob("*/terminal.json"))
    reuse=list((repo/R2_ROOT/"units").glob("*/attempts/attempt-000001/clean/terminal.json"))
    journals=list((repo/R2_ROOT/"units").glob("*/attempts/attempt-000001/journal/*.json"))
    gamma=list((repo/R2_ROOT/"units").glob("*/attempts/attempt-000001/gamma/terminal.json"))
    pending=list((repo/R2_ROOT).rglob("*pending*"))
    terminal_states=[]
    for path in terminals:
        value=r2.strict_read(path);r2.validate_artifact(value,repo,verify_files=True)
        terminal_states.append({**{row["name"]:row["value"] for row in value["payload"]["state"]},"unit_digest":path.parent.name,"status":value["status"],"path":path.relative_to(repo).as_posix(),"canonical_digest":value["canonical_digest"]})
    if stage["status"]!="BLOCKED_INFRASTRUCTURE" or seal["canonical_digest"]!="514df49130f4e7dd903468d20ad8e269717dd48061bdee12c73c568884184b7c" or len(reuse)!=197 or len(terminals)!=2 or len(journals)!=0 or len(gamma)!=0 or pending or (repo/R2_ROOT/"development_sample_manifest.json").exists():
        raise X2R2R1Error("X2-R2 blocked source state differs from authorization")
    if any(row["status"]!="NA" or row.get("failure_code")!="BLOCKED_INFRASTRUCTURE" for row in terminal_states):
        raise X2R2R1Error("X2-R2 NA diagnostic terminal drifted")
    return {"stage":stage,"seal":seal,"terminals":terminal_states,"reuse_count":len(reuse)}


def _write_required_files(repo: Path) -> dict[str, Any]:
    files=[
        "startup_gate.json",
        "amendment_binding.json",
        "x2_r1_failure_binding.json",
        "x2_r2_blocked_binding.json",
        "development_sample_manifest.json",
        "development_unit_manifest.json",
        "manifests/reuse_clean_routes.csv",
        "manifests/new_clean_routes.csv",
        "model_materialization_binding.json",
        "pre_outcome/mutation_suite.json",
        "x2_r2_r1_run_spec.json",
        "implementation_seal.json",
    ]
    missing=[name for name in files if not (repo/ROOT/name).is_file()]
    if missing:raise X2R2R1Error(f"required file missing before list publication: {missing}")
    required_files_json=json.dumps(files,sort_keys=True,separators=(",",":"))
    value=signed("stage_x_x2_r2_r1_preflight_worker","PASS",payload(identity={"phase":"clean","contract":"worker_required_files","required_files_json":required_files_json},files=[file_record(repo,ROOT/file) for file in files],counts={"required_file_count":len(files)},state={"explicit_required_files_list":True,"cpu_only":True,"model_free":True,"jpeg_free":True,"creates_unit_journal":False,"creates_unit_terminal":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)))
    write_new(repo/REQUIRED_FILES_JSON,value,repo)
    return value


def preflight_worker(repo_root:Path|str=REPO_ROOT,phase:str="clean")->dict[str,Any]:
    repo=Path(repo_root).resolve()
    if phase!="clean":raise X2R2R1Error("only clean preflight is authorized")
    imported=set(__import__("sys").modules)
    if "torch" in imported or "timm" in imported:raise X2R2R1Error("preflight must not import torch/timm")
    required_doc=strict_read(repo/REQUIRED_FILES_JSON);validate_artifact(required_doc,repo,verify_files=True)
    identity={row["name"]:row["value"] for row in required_doc["payload"]["identity"]}
    required=json.loads(str(identity["required_files_json"]))
    checks=[]
    # The identity scalar contract cannot carry arrays, so recover the file set
    # from the bound file records and expected manifest names.
    expected=[
        ROOT/"startup_gate.json",ROOT/"amendment_binding.json",ROOT/"x2_r1_failure_binding.json",ROOT/"x2_r2_blocked_binding.json",
        ROOT/"development_sample_manifest.json",ROOT/"development_unit_manifest.json",REUSE_CSV,NEW_CSV,MODEL_BINDING,MUTATION_RECEIPT,RUN_SPEC,ROOT/"implementation_seal.json",
    ]
    for relative in expected:
        value_path=repo/relative
        if not value_path.is_file():raise X2R2R1Error(f"preflight required file missing: {relative}")
        if value_path.suffix==".json":validate_artifact(strict_read(value_path),repo,verify_files=True)
        checks.append({"check_id":"read_"+relative.name,"status":"PASS","detail":relative.as_posix()})
    if [path.relative_to(ROOT).as_posix() for path in expected]!=required:raise X2R2R1Error("explicit required-files list drifted")
    samples=load_samples(repo);units=load_units(repo);reuse=r1.read_csv(repo/REUSE_CSV,REUSE_FIELDS);new=r1.read_csv(repo/NEW_CSV,NEW_FIELDS)
    for row in new:
        row["development_rank"]=int(row["development_rank"]);row["shard"]=int(row["shard"])
    if len(new)!=3 or len(reuse)!=197:raise X2R2R1Error("preflight route split drifted")
    sample_by_id={row["source_object_id"]:row for row in samples};unit_by_id={row["unit_digest"]:row for row in units}
    for row in new:
        unit=unit_by_id[row["unit_digest"]];sample=sample_by_id[row["sample_id"]];_validate_r2_unit(unit,sample_by_id,{item["model_id"] for item in strict_read(repo/OLD_CONFIG)["models"]})
        if row["sample_sha256"]!=sample["file_sha256"] or row["sample_sha256"]!=unit["sample_sha256"] or int(row["shard"])!=int(unit["shard"]):raise X2R2R1Error("preflight new route binding drifted")
    for row in reuse:
        unit=unit_by_id[row["unit_digest"]];sample=sample_by_id[row["sample_id"]]
        if row["sample_sha256"]!=sample["file_sha256"] or row["sample_sha256"]!=unit["sample_sha256"] or row["reuse_status"]!="SOURCE_VALIDATED_VALID":raise X2R2R1Error("preflight reuse route binding drifted")
        _validate_file_record(repo,{"kind":"file","path":row["r1_terminal_path"],"bytes":(repo/row["r1_terminal_path"]).stat().st_size,"sha256":row["r1_terminal_sha256"],"canonical_digest":row["r1_terminal_canonical_digest"]})
    stale=list((repo/ROOT).rglob("*pending*"))
    if stale:raise X2R2R1Error("stale pending file exists before worker launch")
    after=set(__import__("sys").modules)
    if "torch" in after or "timm" in after:raise X2R2R1Error("preflight imported torch/timm")
    receipt=signed("stage_x_x2_r2_r1_preflight_worker","PASS",payload(bindings=[artifact_record(repo,REQUIRED_FILES_JSON),artifact_record(repo,RUN_SPEC),artifact_record(repo,ROOT/"implementation_seal.json"),artifact_record(repo,MODEL_BINDING),artifact_record(repo,ROOT/"development_sample_manifest.json"),artifact_record(repo,ROOT/"development_unit_manifest.json")],checks=checks,counts={"new_routes":len(new),"reuse_routes":len(reuse),"required_files":len(expected),"unit_journals_created":0,"unit_terminals_created":0},state={"phase":phase,"cpu_only":True,"model_free":True,"jpeg_free":True,"torch_imported":False,"timm_imported":False,"jpeg_decoded":False,"model_loaded":False,"scientific_kernel_started":False,"stale_pending":False},terminal=terminal_state(terminal=True,immutable=True,complete=True,next_authorization="X2_R2_R1_CONTROL_INITIALIZATION_RECOVERY_ONLY")))
    if (repo/PREFLIGHT_RECEIPT).exists():
        value=strict_read(repo/PREFLIGHT_RECEIPT);validate_artifact(value,repo,verify_files=True);return value
    write_new(repo/PREFLIGHT_RECEIPT,receipt,repo);return receipt


def initialize(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();_fresh_root_precheck(repo);config=load_config(repo);r1_status=r1.strict_read(repo/R1_STATUS);r1.validate_artifact(r1_status,repo,verify_files=True)
    if r1_status["status"]!="NO_GO_CLASS_LOGIT_COMPLETENESS":raise X2R2R1Error("X2-R1 immutable NO_GO missing")
    startup_state=r1._startup(repo)
    r2_blocked=_recompute_r2_blocked(repo)
    ROOT_DIR=repo/ROOT;ROOT_DIR.mkdir(parents=True,exist_ok=False);(repo/ROOT/"pre_outcome").mkdir(parents=True,exist_ok=False);(repo/ROOT/"manifests").mkdir(parents=True,exist_ok=False);(repo/REPORT_ROOT).mkdir(parents=True,exist_ok=False)
    mutation=run_mutation_suite(repo)
    r1_units=r1.load_units(repo);r1_samples=r1.load_samples(repo);valid=[];new=[];r2_units=[]
    for old in r1_units:
        identity={"run_id":RUN_ID,"model_id":old["model_id"],"sample_id":old["sample_id"],"sample_sha256":old["sample_sha256"],"development_rank":int(old["development_rank"])};digest=canonical_sha256(identity);unit={"unit_digest":digest,**identity,"shard":int.from_bytes(bytes.fromhex(digest)[:8],"big")%4};r2_units.append(unit)
        clean=repo/R1_ROOT/"units"/old["unit_digest"]/"attempts/attempt-000001/clean/terminal.json"
        if clean.is_file():valid.append(validate_r1_clean(repo,unit,old))
        else:new.append({"unit_digest":unit["unit_digest"],"r1_unit_digest":old["unit_digest"],"model_id":unit["model_id"],"sample_id":unit["sample_id"],"sample_sha256":unit["sample_sha256"],"development_rank":unit["development_rank"],"shard":unit["shard"],"r1_state":"FAILED" if (repo/R1_ROOT/"units"/old["unit_digest"]/"terminal.json").is_file() else "NOT_STARTED","r1_source_path":(R1_ROOT/"units"/old["unit_digest"]).as_posix()})
    if len(valid)!=197 or Counter(row["r1_state"] for row in new)!=Counter({"NOT_STARTED":2,"FAILED":1}):raise X2R2R1Error("X2-R1 source union differs from authorization")
    r1.write_csv_new(repo/SAMPLE_CSV,SAMPLE_FIELDS,r1_samples);r1.write_csv_new(repo/UNIT_CSV,UNIT_FIELDS,r2_units);r1.write_csv_new(repo/REUSE_CSV,REUSE_FIELDS,valid);r1.write_csv_new(repo/NEW_CSV,NEW_FIELDS,new)
    common={"holm_family_count":5,"full_model_ig_in_holm":False,"x3_started":False,"x4_started":False,"x5_started":False}
    startup=signed("stage_x_x2_r2_r1_startup_gate","PASS",payload(identity=startup_state,bindings=[file_record(repo,CONFIG,canonical_digest=config["canonical_digest"]),file_record(repo,AMENDMENT),file_record(repo,R1_STATUS,canonical_digest=r1_status["canonical_digest"]),file_record(repo,R2_STATUS,canonical_digest=r2_blocked["stage"]["canonical_digest"]),file_record(repo,R2_IMPLEMENTATION_SEAL,canonical_digest=r2_blocked["seal"]["canonical_digest"]),file_record(repo,MUTATION_RECEIPT,canonical_digest=mutation["canonical_digest"])],counts={"r1_valid_reuse":197,"new_execution":3,"expected_routes":200,"r2_control_only_na_terminals":2},state={**common,"fresh_root":True,"control_preflight_only_no_scientific_route_execution":True}));write_new(repo/ROOT/"startup_gate.json",startup,repo)
    amendment=signed("stage_x_x2_r2_r1_amendment_binding","PASS",payload(bindings=[file_record(repo,AMENDMENT),file_record(repo,CONFIG,canonical_digest=config["canonical_digest"])],state={**common,"prospective_before_x2_r2_r1_outcome":True,"same_root_retry_allowed":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(repo/ROOT/"amendment_binding.json",amendment,repo)
    r1_binding=signed("stage_x_x2_r2_r1_x2_r1_failure_binding","PASS",payload(bindings=[file_record(repo,R1_STATUS,canonical_digest=r1_status["canonical_digest"])],counts={"source_valid_clean_routes":197,"failed_routes":1,"never_started_routes":2},state={**common,"failed_route_relabelled_valid":False,"source_validated_reuse_only":True},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(repo/ROOT/"x2_r1_failure_binding.json",r1_binding,repo)
    r2_terminal_bindings=[file_record(repo,Path(row["path"]),canonical_digest=row["canonical_digest"]) for row in r2_blocked["terminals"]]
    r2_binding=signed("stage_x_x2_r2_r1_x2_r2_blocked_binding","PASS",payload(bindings=[file_record(repo,R2_STATUS,canonical_digest=r2_blocked["stage"]["canonical_digest"]),file_record(repo,R2_IMPLEMENTATION_SEAL,canonical_digest=r2_blocked["seal"]["canonical_digest"]),*r2_terminal_bindings],counts={"source_reuse_clean_terminals":197,"control_only_na_terminals":2,"new_scientific_journals":0,"gamma_terminals":0},state={**common,"blocked_cause":"missing development_sample_manifest.json","control_only_classification":"CONTROL_PREFLIGHT_ONLY_NO_SCIENTIFIC_ROUTE_EXECUTION","actual_execution_attempted_in_old_na_is_inaccurate":True,"model_loaded_maybe":True,"jpeg_decoded":False,"scientific_kernel_started":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(repo/ROOT/"x2_r2_blocked_binding.json",r2_binding,repo)
    sample_manifest=signed("stage_x_x2_r2_r1_sample_manifest","PASS",payload(bindings=[file_record(repo,R1_ROOT/"development_sample_manifest.json",canonical_digest=r1.strict_read(repo/R1_ROOT/"development_sample_manifest.json")["canonical_digest"])],files=[file_record(repo,SAMPLE_CSV,canonical_digest=canonical_sha256(r1_samples))],counts={"sample_count":40,"unique_sample_count":40,"model_count":5},state={**common,"shared_ordered_list_across_models":True,"reselection_performed":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(repo/ROOT/"development_sample_manifest.json",sample_manifest,repo)
    manifest=signed("stage_x_x2_r2_r1_unit_manifest","PASS",payload(files=[file_record(repo,SAMPLE_CSV,canonical_digest=canonical_sha256(r1_samples)),file_record(repo,UNIT_CSV,canonical_digest=canonical_sha256(r2_units)),file_record(repo,REUSE_CSV,canonical_digest=canonical_sha256(r1.read_csv(repo/REUSE_CSV,REUSE_FIELDS))),file_record(repo,NEW_CSV,canonical_digest=canonical_sha256(r1.read_csv(repo/NEW_CSV,NEW_FIELDS)))],counts={"unit_count":200,"unique_unit_count":200,"sample_count":40,"model_count":5,"reuse_count":197,"new_count":3},state={**common,"overall_valid_min":190,"per_model_valid_min":38,"all_expected_require_terminal":True},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(repo/ROOT/"development_unit_manifest.json",manifest,repo)
    r1_model=r1.strict_read(repo/R1_ROOT/"model_materialization.json");r1.validate_artifact(r1_model,repo,verify_files=True)
    model_doc=signed("stage_x_x2_r2_r1_model_materialization","PASS",payload(bindings=[file_record(repo,R1_ROOT/"model_materialization.json",canonical_digest=r1_model["canonical_digest"]),file_record(repo,R1_ROOT/"gpu_execution_diagnostic.json",canonical_digest=r1.strict_read(repo/R1_ROOT/"gpu_execution_diagnostic.json")["canonical_digest"])],counts={"model_count":5,"downloaded_bytes":875799674},state={**common,"cache_reused":True,"strict_load_required_at_worker":True,"checkpoint_redownloaded":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(repo/MODEL_BINDING,model_doc,repo)
    run=signed("stage_x_x2_r2_r1_run_spec","PASS",payload(identity={"authorization":config["authorization"],"run_id":RUN_ID},bindings=[artifact_record(repo,ROOT/"startup_gate.json"),artifact_record(repo,ROOT/"amendment_binding.json"),artifact_record(repo,ROOT/"x2_r1_failure_binding.json"),artifact_record(repo,ROOT/"x2_r2_blocked_binding.json"),artifact_record(repo,ROOT/"development_sample_manifest.json"),artifact_record(repo,ROOT/"development_unit_manifest.json"),artifact_record(repo,MODEL_BINDING),artifact_record(repo,MUTATION_RECEIPT)],counts={"expected_routes":200,"overall_valid_min":190,"per_model_valid_min":38,"gamma_candidates":15},state={**common,"original_schedule_unchanged":True,"tolerances_unchanged":True,"pre_outcome_receipt_sha256":sha_file(repo/MUTATION_RECEIPT),"same_root_retry_allowed":False}));write_new(repo/RUN_SPEC,run,repo)
    seal_implementation(repo)
    _write_required_files(repo)
    for row in valid:
        r2_unit=next(unit for unit in r2_units if unit["unit_digest"]==row["unit_digest"]);r1_terminal=Path(row["r1_terminal_path"]);source=r1.strict_read(repo/r1_terminal);files=source["payload"]["files"];body=payload(identity={"attempt_id":"reuse-r1-attempt-000001","phase":"clean","unit_digest":r2_unit["unit_digest"]},bindings=[file_record(repo,r1_terminal,canonical_digest=source["canonical_digest"])],files=files,counts={"complete_cut_count":135,"source_validated_reuse":1},state={"status":"VALID","reuse_status":"SOURCE_VALIDATED_VALID","r1_unit_digest":row["r1_unit_digest"],"r1_tree_digest":row["r1_tree_digest"],"identity_control_pass":True,"complete_cuts_pass":True,"native_pixel_provenance":True},resources=source["payload"]["resources"],terminal=terminal_state(terminal=True,immutable=True,complete=True));terminal=signed("stage_x_x2_r2_phase_manifest","PASS",body);path=repo/ROOT/"units"/r2_unit["unit_digest"]/"attempts/attempt-000001/clean/terminal.json";write_new(path,terminal,repo)
    return {"status":"PASS","reuse":197,"new":3,"preflight":"READY"}


def finalize_control(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve()
    value=strict_read(repo/RUN_SPEC);validate_artifact(value,repo,verify_files=True);return value


def seal_implementation(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();path=repo/ROOT/"implementation_seal.json"
    if path.exists():
        value=strict_read(path);validate_artifact(value,repo,verify_files=True);return value
    for relative in IMPLEMENTATION_FILES:
        if not (repo/relative).is_file():raise X2R2R1Error(f"implementation file missing: {relative}")
    bindings=[file_record(repo,Path(name)) for name in IMPLEMENTATION_FILES]+[artifact_record(repo,RUN_SPEC),artifact_record(repo,MUTATION_RECEIPT),artifact_record(repo,MODEL_BINDING)]
    value=signed("stage_x_x2_r2_r1_implementation_seal","PASS",payload(bindings=bindings,counts={"implementation_files":len(IMPLEMENTATION_FILES)},state={"sealed_before_x2_r2_r1_outcome_access":True,"source_changes_after_seal_allowed":False,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(path,value,repo);return value


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("mutate","initialize","finalize-control","seal","preflight-worker"));parser.add_argument("--repo-root",type=Path,default=REPO_ROOT);parser.add_argument("--phase",default="clean");args=parser.parse_args(argv)
    try:
        if args.command=="preflight-worker":value=preflight_worker(args.repo_root,args.phase)
        else:value={"mutate":run_mutation_suite,"initialize":initialize,"finalize-control":finalize_control,"seal":seal_implementation}[args.command](args.repo_root)
    except Exception as error:print(json.dumps({"status":"BLOCKED","error":f"{type(error).__name__}: {error}"},sort_keys=True));return 2
    print(json.dumps(value,sort_keys=True,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
