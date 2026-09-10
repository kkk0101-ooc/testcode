"""Prospective X2-R2 coverage-gated recovery control plane."""

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

RUN_ID="STAGE_X_X2_R2_V1_20260909"
ROOT=Path("evidence/stage_x/x2_r2_nns_v2_development_20260909/formal")
REPORT_ROOT=Path("reports/stage_x_x2_r2_nns_v2_development_20260909")
CONFIG=Path("config/stage_x_x2_r2.json")
CONFIG_SCHEMA=Path("config/stage_x_x2_r2.schema.json")
ARTIFACT_SCHEMA=Path("config/stage_x_x2_r2_artifact.schema.json")
AMENDMENT=Path("docs/stage_x_x2_r2_coverage_recovery_amendment_20260909.md")
R1_ROOT=r1.ROOT
R1_STATUS=R1_ROOT/"stage_failure.json"
MUTATION_RECEIPT_V1=ROOT/"pre_outcome/mutation_suite.json"
MUTATION_RECEIPT=ROOT/"pre_outcome/mutation_suite_v2.json"
RUN_SPEC_V1=ROOT/"x2_r2_run_spec.json"
RUN_SPEC=ROOT/"x2_r2_run_spec_v2.json"
UNIT_CSV=ROOT/"manifests/development_units.csv"
SAMPLE_CSV=ROOT/"manifests/development_samples.csv"
REUSE_CSV=ROOT/"manifests/reuse_clean_routes.csv"
NEW_CSV=ROOT/"manifests/new_clean_routes.csv"
UNIT_FIELDS=r1.UNIT_FIELDS
SAMPLE_FIELDS=r1.SAMPLE_FIELDS
REUSE_FIELDS=("unit_digest","r1_unit_digest","model_id","sample_id","sample_sha256","r1_terminal_path","r1_terminal_sha256","r1_terminal_canonical_digest","r1_tree_digest","r1_selected_steps","reuse_status")
NEW_FIELDS=("unit_digest","r1_unit_digest","model_id","sample_id","sample_sha256","development_rank","shard","r1_state","r1_source_path")
ZERO_RESOURCES=r1.ZERO_RESOURCES
MODEL_LOCAL_PATHS=r1.MODEL_LOCAL_PATHS
OLD_CONFIG=r1.OLD_CONFIG
_load_state=r1._load_state


class X2R2Error(ValueError):pass


def strict_read(path:Path)->dict[str,Any]:return r1.strict_read(path)
def sha_file(path:Path)->str:return r1.sha_file(path)
def scalar_rows(value:Mapping[str,Any])->list[dict[str,Any]]:return r1.scalar_rows(value)
def terminal_state(**kwargs:Any)->dict[str,Any]:return r1.terminal_state(**kwargs)
def payload(**kwargs:Any)->dict[str,Any]:return r1.payload(**kwargs)
def file_record(repo:Path,relative:Path,*,canonical_digest:str|None=None)->dict[str,Any]:return r1.file_record(repo,relative,canonical_digest=canonical_digest)
def load_samples(repo:Path)->list[dict[str,Any]]:return r1.load_samples(repo) if (repo/ROOT)==(repo/R1_ROOT) else _load_samples(repo)


def _load_samples(repo:Path)->list[dict[str,Any]]:
    manifest=strict_read(repo/ROOT/"development_unit_manifest.json");validate_artifact(manifest,repo,verify_files=True);rows=r1.read_csv(repo/SAMPLE_CSV,SAMPLE_FIELDS)
    for row in rows:row["development_rank"]=int(row["development_rank"]);row["file_bytes"]=int(row["file_bytes"])
    if len(rows)!=40 or len({row["source_object_id"] for row in rows})!=40:raise X2R2Error("R2 sample roster drifted")
    return rows


def load_units(repo:Path)->list[dict[str,Any]]:
    manifest=strict_read(repo/ROOT/"development_unit_manifest.json");validate_artifact(manifest,repo,verify_files=True);samples={row["source_object_id"]:row for row in _load_samples(repo)};models={row["model_id"] for row in strict_read(repo/OLD_CONFIG)["models"]};rows=r1.read_csv(repo/UNIT_CSV,UNIT_FIELDS)
    for row in rows:
        row["development_rank"]=int(row["development_rank"]);row["shard"]=int(row["shard"]);r1.validate_unit_row(row,samples,models) if row["run_id"]==r1.RUN_ID else _validate_r2_unit(row,samples,models)
    if len(rows)!=200 or len({row["unit_digest"] for row in rows})!=200:raise X2R2Error("R2 unit roster drifted")
    return rows


def _validate_r2_unit(row:Mapping[str,Any],samples:Mapping[str,Mapping[str,Any]],models:set[str])->None:
    if row["run_id"]!=RUN_ID or row["model_id"] not in models or row["sample_id"] not in samples:raise X2R2Error("R2 unit binding drifted")
    sample=samples[row["sample_id"]];identity={"run_id":RUN_ID,"model_id":row["model_id"],"sample_id":row["sample_id"],"sample_sha256":row["sample_sha256"],"development_rank":int(row["development_rank"])};digest=canonical_sha256(identity)
    if row["sample_sha256"]!=sample["file_sha256"] or int(row["development_rank"])!=int(sample["development_rank"]) or row["unit_digest"]!=digest or int(row["shard"])!=int.from_bytes(bytes.fromhex(digest)[:8],"big")%4:raise X2R2Error("R2 unit digest/sample/shard drifted")


def publish_atomic(pending:Path,terminal:Path,value:Mapping[str,Any],repo:Path)->None:
    if pending.exists() or terminal.exists():raise X2R2Error("pending or terminal exists")
    pending_body=copy.deepcopy(value["payload"]);pending_body["terminal"]=terminal_state();write_new(pending,signed(value["document_type"],"PENDING",pending_body),repo);temporary=terminal.with_name(terminal.name+".terminal.pending");write_new(temporary,value,repo);os.replace(temporary,terminal);pending.unlink()


def signed(document_type:str,status:str,body:Mapping[str,Any])->dict[str,Any]:
    value={"$schema":ARTIFACT_SCHEMA.name,"schema_version":1,"document_type":document_type.replace("stage_x_x2_r1","stage_x_x2_r2"),"stage_id":"X2-R2","run_id":RUN_ID,"status":status,"payload":dict(body)};value["canonical_digest"]=canonical_sha256(value);return value


def load_config(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();value=strict_read(repo/CONFIG);schema=strict_read(repo/CONFIG_SCHEMA);Draft202012Validator.check_schema(schema);errors=sorted(Draft202012Validator(schema).iter_errors(value),key=lambda error:tuple(str(part) for part in error.absolute_path))
    if errors:raise X2R2Error(errors[0].message)
    plain=dict(value);observed=plain.pop("canonical_digest");
    if observed!=canonical_sha256(plain):raise X2R2Error("X2-R2 config digest drifted")
    return value


def validate_coverage_states(rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    if len(rows)!=200 or len({row.get("unit_digest") for row in rows})!=200:raise X2R2Error("coverage requires exactly 200 unique routes")
    allowed={"VALID","FAILED_ROUTE_LOCAL_NUMERICAL","FAILED_ROUTE_LOCAL_EXECUTION","NA_ROUTE_LOCAL_INFRASTRUCTURE"}
    if any(row.get("status") not in allowed for row in rows):raise X2R2Error("coverage contains untyped, interrupted, or not-started state")
    totals=Counter(row["model_id"] for row in rows);valid=Counter(row["model_id"] for row in rows if row["status"]=="VALID")
    if len(totals)!=5 or set(totals.values())!={40}:raise X2R2Error("coverage model denominator drifted")
    return {"expected":200,"valid":sum(valid.values()),"per_model":dict(sorted(valid.items())),"pass":sum(valid.values())>=190 and all(valid[model]>=38 for model in totals)}


def validate_artifact(value:Mapping[str,Any],repo_root:Path|str=REPO_ROOT,*,verify_files:bool=False)->None:
    repo=Path(repo_root).resolve();schema=strict_read(repo/ARTIFACT_SCHEMA);errors=sorted(Draft202012Validator(schema).iter_errors(value),key=lambda error:tuple(str(part) for part in error.absolute_path))
    if errors:raise X2R2Error(errors[0].message)
    plain=dict(value);observed=plain.pop("canonical_digest",None)
    if observed!=canonical_sha256(plain):raise X2R2Error("X2-R2 artifact digest drifted")
    body=value["payload"]
    for field,key in (("identity","name"),("counts","name"),("state","name"),("checks","check_id"),("metrics","metric_id"),("bindings","path"),("files","path")):
        rows=body[field];keys=[row[key] for row in rows]
        if len(keys)!=len(set(keys)):raise X2R2Error(f"duplicate {field}")
    if value["document_type"] in {"stage_x_x2_r2_run_spec","stage_x_x2_r2_gamma_selection","stage_x_x2_r2_completion"}:
        state={row["name"]:row["value"] for row in body["state"]}
        if state.get("holm_family_count")!=5 or state.get("full_model_ig_in_holm") is not False:raise X2R2Error("Holm family drifted")
    if verify_files:
        for record in [*body["bindings"],*body["files"]]:r1.validate_file_record(repo,record)


def write_new(path:Path,value:Mapping[str,Any],repo_root:Path|str=REPO_ROOT)->None:
    validate_artifact(value,repo_root);path.parent.mkdir(parents=True,exist_ok=True);descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(descriptor,"wb") as stream:stream.write(canonical_json_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())


def artifact_record(repo:Path,relative:Path)->dict[str,Any]:
    value=strict_read(repo/relative);validate_artifact(value,repo);return file_record(repo,relative,canonical_digest=value["canonical_digest"])


def _tree_digest(path:Path)->str:
    rows=[(item.relative_to(path).as_posix(),item.stat().st_size,sha_file(item)) for item in sorted(path.rglob("*")) if item.is_file()];return canonical_sha256(rows)


def validate_r1_clean(repo:Path,r2_unit:Mapping[str,Any],r1_unit:Mapping[str,Any])->dict[str,Any]:
    base=repo/R1_ROOT/"units"/r1_unit["unit_digest"]/"attempts/attempt-000001/clean";terminal=strict_read(base/"terminal.json");r1.validate_artifact(terminal,repo,verify_files=True);summary=r1_worker._read_one_csv(base/"summary.csv")
    if terminal["status"]!="PASS" or summary["unit_digest"]!=r1_unit["unit_digest"] or summary["sample_sha256"]!=r2_unit["sample_sha256"] or summary["model_id"]!=r2_unit["model_id"] or int(summary["complete_cut_count"])!=135 or summary["complete_cut_all_pass"]!="True" or float(summary["identity_max_abs_error"])>1e-6:raise X2R2Error("R1 clean route cannot be reused")
    return {"unit_digest":r2_unit["unit_digest"],"r1_unit_digest":r1_unit["unit_digest"],"model_id":r2_unit["model_id"],"sample_id":r2_unit["sample_id"],"sample_sha256":r2_unit["sample_sha256"],"r1_terminal_path":(base/"terminal.json").relative_to(repo).as_posix(),"r1_terminal_sha256":sha_file(base/"terminal.json"),"r1_terminal_canonical_digest":terminal["canonical_digest"],"r1_tree_digest":_tree_digest(base),"r1_selected_steps":int(summary["selected_steps"]),"reuse_status":"SOURCE_VALIDATED_VALID"}


def run_mutation_suite(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=load_config(repo)
    if (repo/MUTATION_RECEIPT).exists():value=strict_read(repo/MUTATION_RECEIPT);validate_artifact(value,repo,verify_files=True);return value
    executed=list((repo/ROOT/"units").glob("*/attempts/attempt-000001/journal/*.json")) if (repo/ROOT/"units").exists() else []
    actual_terminals=list((repo/ROOT/"units").glob("*/terminal.json")) if (repo/ROOT/"units").exists() else []
    if executed or actual_terminals:raise X2R2Error("cannot replace mutation suite after new R2 outcome access")
    checks=[];base=signed("stage_x_x2_r2_phase_manifest","PASS",payload(state={"holm_family_count":5,"full_model_ig_in_holm":False}))
    def reject(name:str,fn:Any)->None:
        try:fn()
        except Exception as error:checks.append({"check_id":name,"status":"PASS","detail":type(error).__name__})
        else:raise X2R2Error(f"mutation accepted: {name}")
    def mutate(fn:Any)->None:
        value=copy.deepcopy(base);fn(value);plain=dict(value);plain.pop("canonical_digest",None);value["canonical_digest"]=canonical_sha256(plain);validate_artifact(value,repo)
    reject("root_unknown_field",lambda:mutate(lambda value:value.__setitem__("UNKNOWN",True)));reject("payload_unknown_field",lambda:mutate(lambda value:value["payload"].__setitem__("UNKNOWN",True)));reject("nested_unknown_field",lambda:mutate(lambda value:value["payload"]["terminal"].__setitem__("UNKNOWN",True)));reject("missing_required_field",lambda:mutate(lambda value:value["payload"].pop("resources")));reject("duplicate_json_key",lambda:r1.strict_loads('{"a":1,"a":2}'));reject("nan",lambda:r1.strict_loads('{"a":NaN}'));reject("infinity",lambda:r1.strict_loads('{"a":Infinity}'));bad=copy.deepcopy(base);bad["canonical_digest"]="0"*64;reject("bad_canonical_digest",lambda:validate_artifact(bad,repo));reject("bool_as_int",lambda:mutate(lambda value:value["payload"]["metrics"].append({"metric_id":"bad","value":True,"unit":"count"})))
    models=[f"m{index}" for index in range(5)];good=[{"unit_digest":hashlib.sha256(f"{model}-{index}".encode()).hexdigest(),"model_id":model,"status":"VALID"} for model in models for index in range(40)]
    reject("wrong_expected_count",lambda:validate_coverage_states(good[:-1]));bad_overall=copy.deepcopy(good)
    for index in range(11):bad_overall[index]["status"]="FAILED_ROUTE_LOCAL_NUMERICAL"
    result=validate_coverage_states(bad_overall)
    if result["pass"]:raise X2R2Error("wrong overall threshold accepted")
    checks.append({"check_id":"wrong_overall_threshold","status":"PASS","detail":"coverage_rejected"});bad_model=copy.deepcopy(good)
    for index in range(3):bad_model[index]["status"]="FAILED_ROUTE_LOCAL_NUMERICAL"
    result=validate_coverage_states(bad_model)
    if result["pass"]:raise X2R2Error("wrong per-model threshold accepted")
    checks.append({"check_id":"wrong_per_model_threshold","status":"PASS","detail":"coverage_rejected"})
    for name,status in (("not_started_as_terminal","NOT_STARTED"),("sample_replacement","REPLACED"),("failed_route_reused_valid","DIAGNOSTIC_ONLY")):
        mutated=copy.deepcopy(good);mutated[0]["status"]=status;reject(name,lambda mutated=mutated:validate_coverage_states(mutated))
    reject("full_model_ig_in_holm",lambda:validate_artifact(signed("stage_x_x2_r2_run_spec","PASS",payload(state={"holm_family_count":5,"full_model_ig_in_holm":True})),repo));reject("holm_not_five",lambda:validate_artifact(signed("stage_x_x2_r2_run_spec","PASS",payload(state={"holm_family_count":6,"full_model_ig_in_holm":False})),repo))
    with tempfile.TemporaryDirectory(prefix="x2-r2-mut-") as name:
        temp=Path(name);one=temp/"one.json";write_new(one,base,repo);reject("terminal_overwrite",lambda:write_new(one,base,repo));pending=temp/"pending.json";terminal=temp/"terminal.json";publish_atomic(pending,terminal,base,repo);reject("pending_overwrite",lambda:publish_atomic(pending,terminal,base,repo))
        array=temp/"value.npy";np.save(array,np.zeros((2,3),dtype=np.float32),allow_pickle=False);record={"bytes":array.stat().st_size,"sha256":sha_file(array),"dtype":"float32","shape":[2,3]}
        def check_array(changed:Mapping[str,Any])->None:
            value=np.load(array,allow_pickle=False);actual={"bytes":array.stat().st_size,"sha256":sha_file(array),"dtype":str(value.dtype),"shape":list(value.shape)}
            if dict(changed)!=actual:raise X2R2Error("array drift")
        reject("array_dtype_shape_hash_digest_drift",lambda:check_array({**record,"shape":[3,2]}))
    body=payload(identity={"suite":"X2_R2_PRE_OUTCOME_MUTATION_SUITE_V2"},bindings=[file_record(repo,ARTIFACT_SCHEMA),file_record(repo,CONFIG,canonical_digest=config["canonical_digest"]),file_record(repo,AMENDMENT),file_record(repo,MUTATION_RECEIPT_V1,canonical_digest=strict_read(repo/MUTATION_RECEIPT_V1)["canonical_digest"])],checks=checks,counts={"attack_count":len(checks),"rejected_count":len(checks)},state={"outcome_access_before_suite":False,"all_mutations_rejected":True,"supersedes":"mutation_suite.json"},terminal=terminal_state(terminal=True,immutable=True,complete=True));receipt=signed("stage_x_x2_r2_pre_outcome_mutation_receipt","PASS",body);write_new(repo/MUTATION_RECEIPT,receipt,repo);return receipt


def initialize(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=load_config(repo);mutation=strict_read(repo/MUTATION_RECEIPT);validate_artifact(mutation,repo,verify_files=True);r1_status=r1.strict_read(repo/R1_STATUS);r1.validate_artifact(r1_status,repo,verify_files=True)
    if r1_status["status"]!="NO_GO_CLASS_LOGIT_COMPLETENESS":raise X2R2Error("X2-R1 immutable NO_GO missing")
    r1_units=r1.load_units(repo);r1_samples=r1.load_samples(repo);valid=[];new=[];r2_units=[]
    for old in r1_units:
        identity={"run_id":RUN_ID,"model_id":old["model_id"],"sample_id":old["sample_id"],"sample_sha256":old["sample_sha256"],"development_rank":int(old["development_rank"])};digest=canonical_sha256(identity);unit={"unit_digest":digest,**identity,"shard":int.from_bytes(bytes.fromhex(digest)[:8],"big")%4};r2_units.append(unit)
        clean=repo/R1_ROOT/"units"/old["unit_digest"]/"attempts/attempt-000001/clean/terminal.json"
        if clean.is_file():valid.append(validate_r1_clean(repo,unit,old))
        else:new.append({"unit_digest":unit["unit_digest"],"r1_unit_digest":old["unit_digest"],"model_id":unit["model_id"],"sample_id":unit["sample_id"],"sample_sha256":unit["sample_sha256"],"development_rank":unit["development_rank"],"shard":unit["shard"],"r1_state":"FAILED" if (repo/R1_ROOT/"units"/old["unit_digest"]/"terminal.json").is_file() else "NOT_STARTED","r1_source_path":(R1_ROOT/"units"/old["unit_digest"]).as_posix()})
    if len(valid)!=197 or Counter(row["r1_state"] for row in new)!=Counter({"NOT_STARTED":2,"FAILED":1}):raise X2R2Error("X2-R1 source union differs from authorization")
    ROOT_DIR=repo/ROOT;ROOT_DIR.mkdir(parents=True,exist_ok=True);(repo/REPORT_ROOT).mkdir(parents=True,exist_ok=False);r1.write_csv_new(repo/SAMPLE_CSV,SAMPLE_FIELDS,r1_samples);r1.write_csv_new(repo/UNIT_CSV,UNIT_FIELDS,r2_units);r1.write_csv_new(repo/REUSE_CSV,REUSE_FIELDS,valid);r1.write_csv_new(repo/NEW_CSV,NEW_FIELDS,new)
    common={"holm_family_count":5,"full_model_ig_in_holm":False,"x3_started":False}
    startup=signed("stage_x_x2_r2_startup_gate","PASS",payload(bindings=[file_record(repo,CONFIG,canonical_digest=config["canonical_digest"]),file_record(repo,AMENDMENT),file_record(repo,R1_STATUS,canonical_digest=r1_status["canonical_digest"]),file_record(repo,MUTATION_RECEIPT,canonical_digest=mutation["canonical_digest"])],counts={"r1_valid_reuse":197,"new_execution":3,"expected_routes":200},state=common));write_new(repo/ROOT/"startup_gate.json",startup,repo)
    manifest=signed("stage_x_x2_r2_unit_manifest","PASS",payload(files=[file_record(repo,SAMPLE_CSV,canonical_digest=canonical_sha256(r1_samples)),file_record(repo,UNIT_CSV,canonical_digest=canonical_sha256(r2_units)),file_record(repo,REUSE_CSV,canonical_digest=canonical_sha256(r1.read_csv(repo/REUSE_CSV,REUSE_FIELDS))),file_record(repo,NEW_CSV,canonical_digest=canonical_sha256(r1.read_csv(repo/NEW_CSV,NEW_FIELDS)))],counts={"unit_count":200,"unique_unit_count":200,"sample_count":40,"model_count":5,"reuse_count":197,"new_count":3},state={**common,"overall_valid_min":190,"per_model_valid_min":38,"all_expected_require_terminal":True}));write_new(repo/ROOT/"development_unit_manifest.json",manifest,repo)
    run=signed("stage_x_x2_r2_run_spec","PASS",payload(identity={"authorization":config["authorization"],"run_id":RUN_ID},bindings=[artifact_record(repo,ROOT/"startup_gate.json"),artifact_record(repo,ROOT/"development_unit_manifest.json")],counts={"expected_routes":200,"overall_valid_min":190,"per_model_valid_min":38,"gamma_candidates":15},state={**common,"original_schedule_unchanged":True,"tolerances_unchanged":True,"pre_outcome_receipt_sha256":sha_file(repo/MUTATION_RECEIPT)}));write_new(repo/RUN_SPEC,run,repo)
    r1_model=r1.strict_read(repo/R1_ROOT/"model_materialization.json");r1.validate_artifact(r1_model,repo,verify_files=True);model_doc=signed("stage_x_x2_r2_model_materialization","PASS",payload(bindings=[file_record(repo,R1_ROOT/"model_materialization.json",canonical_digest=r1_model["canonical_digest"]),file_record(repo,R1_ROOT/"gpu_execution_diagnostic.json",canonical_digest=r1.strict_read(repo/R1_ROOT/"gpu_execution_diagnostic.json")["canonical_digest"])],counts={"model_count":5,"downloaded_bytes":875799674},state={**common,"cache_reused":True,"strict_load_required_at_worker":True,"checkpoint_redownloaded":False}));write_new(repo/ROOT/"model_materialization.json",model_doc,repo)
    for row in valid:
        r2_unit=next(unit for unit in r2_units if unit["unit_digest"]==row["unit_digest"]);r1_terminal=Path(row["r1_terminal_path"]);source=r1.strict_read(repo/r1_terminal);files=source["payload"]["files"];body=payload(identity={"attempt_id":"reuse-r1-attempt-000001","phase":"clean","unit_digest":r2_unit["unit_digest"]},bindings=[file_record(repo,r1_terminal,canonical_digest=source["canonical_digest"])],files=files,counts={"complete_cut_count":135,"source_validated_reuse":1},state={"status":"VALID","reuse_status":"SOURCE_VALIDATED_VALID","r1_unit_digest":row["r1_unit_digest"],"r1_tree_digest":row["r1_tree_digest"],"identity_control_pass":True,"complete_cuts_pass":True,"native_pixel_provenance":True},resources=source["payload"]["resources"],terminal=terminal_state(terminal=True,immutable=True,complete=True));terminal=signed("stage_x_x2_r2_phase_manifest","PASS",body);path=repo/ROOT/"units"/r2_unit["unit_digest"]/"attempts/attempt-000001/clean/terminal.json";write_new(path,terminal,repo)
    return {"status":"PASS","reuse":197,"new":3}


def finalize_control(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=load_config(repo);mutation=strict_read(repo/MUTATION_RECEIPT);validate_artifact(mutation,repo,verify_files=True);manifest=strict_read(repo/ROOT/"development_unit_manifest.json");validate_artifact(manifest,repo,verify_files=True)
    if (repo/RUN_SPEC).exists():value=strict_read(repo/RUN_SPEC);validate_artifact(value,repo,verify_files=True);return value
    original=strict_read(repo/RUN_SPEC_V1);validate_artifact(original,repo,verify_files=True)
    run=signed("stage_x_x2_r2_run_spec","PASS",payload(identity={"authorization":config["authorization"],"run_id":RUN_ID},bindings=[artifact_record(repo,ROOT/"startup_gate.json"),artifact_record(repo,ROOT/"development_unit_manifest.json"),artifact_record(repo,MUTATION_RECEIPT)],counts={"expected_routes":200,"overall_valid_min":190,"per_model_valid_min":38,"gamma_candidates":15},state={"holm_family_count":5,"full_model_ig_in_holm":False,"x3_started":False,"original_schedule_unchanged":True,"tolerances_unchanged":True,"pre_outcome_receipt_sha256":sha_file(repo/MUTATION_RECEIPT),"supersedes_run_spec":RUN_SPEC_V1.as_posix()}));write_new(repo/RUN_SPEC,run,repo);return run


def seal_implementation(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();path=repo/ROOT/"implementation_seal.json";files=("src/vitnns/stage_x_x2_r2.py","src/vitnns/stage_x_x2_r2_worker.py","src/vitnns/stage_x_x2_r2_reconcile.py","config/stage_x_x2_r2.json","config/stage_x_x2_r2.schema.json","config/stage_x_x2_r2_artifact.schema.json","docs/stage_x_x2_r2_coverage_recovery_amendment_20260909.md","tests/test_stage_x_x2_r2.py")
    bindings=[file_record(repo,Path(name)) for name in files]+[artifact_record(repo,RUN_SPEC),artifact_record(repo,MUTATION_RECEIPT)]
    value=signed("stage_x_x2_r2_implementation_seal","PASS",payload(bindings=bindings,counts={"implementation_files":len(files)},state={"sealed_before_x2_r2_outcome_access":True,"source_changes_after_seal_allowed":False,"holm_family_count":5,"full_model_ig_in_holm":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(path,value,repo);return value


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("mutate","initialize","finalize-control","seal"));parser.add_argument("--repo-root",type=Path,default=REPO_ROOT);args=parser.parse_args(argv)
    try:value={"mutate":run_mutation_suite,"initialize":initialize,"finalize-control":finalize_control,"seal":seal_implementation}[args.command](args.repo_root)
    except Exception as error:print(json.dumps({"status":"BLOCKED","error":f"{type(error).__name__}: {error}"},sort_keys=True));return 2
    print(json.dumps(value,sort_keys=True,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
