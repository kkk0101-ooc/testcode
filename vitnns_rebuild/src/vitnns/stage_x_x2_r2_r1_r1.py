"""Prospective X2-R2-R1-R1 formal-stability inverse-map recovery."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_x_x2_r1 as r1
from . import stage_x_x2_r2 as r2
from . import stage_x_x2_r2_r1 as source

RUN_ID="STAGE_X_X2_R2_R1_R1_V1_20260910"
ROOT=Path("evidence/stage_x/x2_r2_r1_r1_nns_v2_development_20260910/formal")
REPORT_ROOT=Path("reports/stage_x_x2_r2_r1_r1_nns_v2_development_20260910")
CONFIG=Path("config/stage_x_x2_r2_r1_r1.json")
CONFIG_SCHEMA=Path("config/stage_x_x2_r2_r1_r1.schema.json")
ARTIFACT_SCHEMA=Path("config/stage_x_x2_r2_r1_r1_artifact.schema.json")
AMENDMENT=Path("docs/stage_x_x2_r2_r1_r1_formal_stability_inverse_map_recovery_amendment_20260910.md")
SOURCE_ROOT=source.ROOT
SOURCE_STAGE_FAILURE=SOURCE_ROOT/"stage_failure.json"
SOURCE_CLEAN_COVERAGE=SOURCE_ROOT/"clean_coverage.json"
SOURCE_GAMMA_SELECTION=SOURCE_ROOT/"gamma_selection.json"
SOURCE_RUN_SPEC=SOURCE_ROOT/"x2_r2_r1_run_spec.json"
SOURCE_IMPLEMENTATION_SEAL=SOURCE_ROOT/"implementation_seal.json"
MUTATION_RECEIPT=ROOT/"pre_outcome/mutation_suite.json"
SUPERSEDED_RUN_SPECS=(ROOT/"x2_r2_r1_r1_run_spec.json",ROOT/"x2_r2_r1_r1_run_spec_v2.json")
SUPERSEDED_IMPLEMENTATION_SEALS=(ROOT/"implementation_seal.json",ROOT/"implementation_seal_v2.json")
RUN_SPEC=ROOT/"x2_r2_r1_r1_run_spec_v3.json"
PREFLIGHT_RECEIPT=ROOT/"preflight_worker_formal_v3.json"
MODEL_BINDING=ROOT/"model_materialization_binding.json"
UNIT_CSV=ROOT/"manifests/development_units.csv"
SAMPLE_CSV=ROOT/"manifests/development_samples.csv"
REUSE_CLEAN_CSV=ROOT/"manifests/reuse_clean_routes.csv"
REUSE_GAMMA_CSV=ROOT/"manifests/reuse_gamma_routes.csv"
REQUIRED_FILES_JSON=ROOT/"worker_required_files_formal_v3.json"
IMPLEMENTATION_SEAL=ROOT/"implementation_seal_v3.json"
ATTEMPT_ID="attempt-000001"
REUSE_ATTEMPT_ID="reuse-x2-r2-r1-attempt-000001"
UNIT_FIELDS=source.UNIT_FIELDS
SAMPLE_FIELDS=source.SAMPLE_FIELDS
OLD_CONFIG=source.OLD_CONFIG
MODEL_LOCAL_PATHS=source.MODEL_LOCAL_PATHS
ZERO_RESOURCES=source.ZERO_RESOURCES
REUSE_PHASE_FIELDS=("unit_digest","phase","source_terminal_path","source_terminal_sha256","source_terminal_canonical_digest","source_tree_digest","status","reuse_status")
IMPLEMENTATION_FILES=(
    "src/vitnns/stage_x_x2_r2_r1_r1.py",
    "src/vitnns/stage_x_x2_r2_r1_r1_worker.py",
    "src/vitnns/stage_x_x2_r2_r1_r1_reconcile.py",
    "src/vitnns/stage_x_x2_r2_r1.py",
    "src/vitnns/stage_x_x2_r2_r1_worker.py",
    "src/vitnns/stage_x_x2_metrics.py",
    "src/vitnns/stage_x_nns_v2.py",
    "src/vitnns/stage_x_comparators.py",
    "config/stage_x_x2_r2_r1_r1.json",
    "config/stage_x_x2_r2_r1_r1.schema.json",
    "config/stage_x_x2_r2_r1_r1_artifact.schema.json",
    "docs/stage_x_x2_r2_r1_r1_formal_stability_inverse_map_recovery_amendment_20260910.md",
    "tests/test_stage_x_x2_r2_r1.py",
)


class X2R2R1R1Error(ValueError): pass


def strict_read(path:Path)->dict[str,Any]: return source.strict_read(path)
def sha_file(path:Path)->str: return source.sha_file(path)
def scalar_rows(value:Mapping[str,Any])->list[dict[str,Any]]: return r1.scalar_rows(value)
def terminal_state(**kwargs:Any)->dict[str,Any]:
    value=r1.terminal_state(**kwargs)
    if value["next_authorization"]=="X2_R2_R1_CONTROL_INITIALIZATION_RECOVERY_ONLY":
        value["next_authorization"]="X2_R2_R1_R1_FORMAL_STABILITY_INVERSE_MAP_SHAPE_RECOVERY_ONLY"
    return value
def payload(**kwargs:Any)->dict[str,Any]: return r1.payload(**kwargs)
def write_csv_new(path:Path,fields:Sequence[str],rows:Sequence[Mapping[str,Any]])->None: return r1.write_csv_new(path,fields,rows)
def write_bytes_new(path:Path,data:bytes)->None: return r1.write_bytes_new(path,data)


def signed(document_type:str,status:str,body:Mapping[str,Any])->dict[str,Any]:
    normalized=document_type.replace("stage_x_x2_r2_r1_r1","stage_x_x2_r2_r1_r1")
    if not normalized.startswith("stage_x_x2_r2_r1_r1"):
        normalized=normalized.replace("stage_x_x2_r2_r1","stage_x_x2_r2_r1_r1",1)
    value={"$schema":ARTIFACT_SCHEMA.name,"schema_version":1,"document_type":normalized,"stage_id":"X2-R2-R1-R1","run_id":RUN_ID,"status":status,"payload":dict(body)}
    value["canonical_digest"]=canonical_sha256(value)
    return value


def load_config(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();value=strict_read(repo/CONFIG);schema=strict_read(repo/CONFIG_SCHEMA)
    Draft202012Validator.check_schema(schema);errors=sorted(Draft202012Validator(schema).iter_errors(value),key=lambda error:tuple(str(part) for part in error.absolute_path))
    if errors:raise X2R2R1R1Error(errors[0].message)
    plain=dict(value);observed=plain.pop("canonical_digest")
    if observed!=canonical_sha256(plain):raise X2R2R1R1Error("X2-R2-R1-R1 config digest drifted")
    return value


def _self_validate(value:Mapping[str,Any],context:str)->None:
    plain=dict(value);observed=plain.pop("canonical_digest",None)
    if observed!=canonical_sha256(plain):raise X2R2R1R1Error(f"canonical digest mismatch: {context}")


def file_record(repo:Path,relative:Path,*,canonical_digest:str|None=None)->dict[str,Any]:
    path=repo/relative
    return {"kind":"file","path":relative.as_posix(),"bytes":path.stat().st_size,"sha256":sha_file(path),"canonical_digest":canonical_digest}


def validate_file_record(repo:Path,record:Mapping[str,Any])->None:
    path=repo/str(record["path"])
    if not path.is_file() or path.stat().st_size!=record["bytes"] or sha_file(path)!=record["sha256"]:
        raise X2R2R1R1Error(f"file record drift: {path}")
    if record["kind"]=="array":
        if r1.array_record(repo,Path(str(record["path"])))!=record:raise X2R2R1R1Error(f"array record drift: {path}")
    elif record["canonical_digest"] is not None:
        if path.suffix==".json":
            value=strict_read(path);_self_validate(value,path.as_posix());observed=value["canonical_digest"]
        elif path.suffix==".csv":
            with path.open("r",encoding="utf-8",newline="") as stream:observed=canonical_sha256([dict(row) for row in csv.DictReader(stream)])
        else:
            raise X2R2R1R1Error(f"unsupported canonical file record: {path}")
        if observed!=record["canonical_digest"]:raise X2R2R1R1Error(f"document digest drift: {path}")


def validate_artifact(value:Mapping[str,Any],repo_root:Path|str=REPO_ROOT,*,verify_files:bool=False)->None:
    repo=Path(repo_root).resolve();schema=strict_read(repo/ARTIFACT_SCHEMA)
    errors=sorted(Draft202012Validator(schema).iter_errors(value),key=lambda error:tuple(str(part) for part in error.absolute_path))
    if errors:raise X2R2R1R1Error(errors[0].message)
    _self_validate(value,str(value.get("document_type")))
    body=value["payload"]
    for field,key in (("identity","name"),("counts","name"),("state","name"),("checks","check_id"),("metrics","metric_id"),("bindings","path"),("files","path")):
        keys=[row[key] for row in body[field]]
        if len(keys)!=len(set(keys)):raise X2R2R1R1Error(f"duplicate {field}")
    if value["document_type"] in {"stage_x_x2_r2_r1_r1_run_spec","stage_x_x2_r2_r1_r1_gamma_selection","stage_x_x2_r2_r1_r1_completion"}:
        state={row["name"]:row["value"] for row in body["state"]}
        if state.get("holm_family_count")!=5 or state.get("full_model_ig_in_holm") is not False:raise X2R2R1R1Error("Holm family drifted")
    if verify_files:
        for record in [*body["bindings"],*body["files"]]:validate_file_record(repo,record)


def write_new(path:Path,value:Mapping[str,Any],repo_root:Path|str=REPO_ROOT)->None:
    validate_artifact(value,repo_root);path.parent.mkdir(parents=True,exist_ok=True);descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(descriptor,"wb") as stream:stream.write(canonical_json_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())


def publish_atomic(pending:Path,terminal:Path,value:Mapping[str,Any],repo:Path)->None:
    if pending.exists() or terminal.exists():raise X2R2R1R1Error("pending or terminal exists")
    pending_body=copy.deepcopy(value["payload"]);pending_body["terminal"]=terminal_state();write_new(pending,signed(value["document_type"],"PENDING",pending_body),repo);temporary=terminal.with_name(terminal.name+".terminal.pending");write_new(temporary,value,repo);os.replace(temporary,terminal);pending.unlink()


def artifact_record(repo:Path,relative:Path)->dict[str,Any]:
    value=strict_read(repo/relative);validate_artifact(value,repo);return file_record(repo,relative,canonical_digest=value["canonical_digest"])


def source_artifact_record(repo:Path,relative:Path)->dict[str,Any]:
    value=strict_read(repo/relative);_self_validate(value,relative.as_posix());return file_record(repo,relative,canonical_digest=value["canonical_digest"])


def _tree_digest(path:Path)->str:
    rows=[(item.relative_to(path).as_posix(),item.stat().st_size,sha_file(item)) for item in sorted(path.rglob("*")) if item.is_file()]
    return canonical_sha256(rows)


def read_csv(path:Path,fields:Sequence[str])->list[dict[str,str]]:
    with path.open("r",encoding="utf-8",newline="") as stream:
        reader=csv.DictReader(stream)
        if tuple(reader.fieldnames or ())!=tuple(fields):raise X2R2R1R1Error(f"CSV header mismatch: {path}")
        return [dict(row) for row in reader]


def load_samples(repo:Path)->list[dict[str,Any]]:
    rows=read_csv(repo/SAMPLE_CSV,SAMPLE_FIELDS)
    for row in rows:row["development_rank"]=int(row["development_rank"]);row["file_bytes"]=int(row["file_bytes"])
    source_rows=source.load_samples(repo)
    if rows!=source_rows:raise X2R2R1R1Error("sample roster differs from X2-R2-R1 source")
    return rows


def load_units(repo:Path)->list[dict[str,Any]]:
    rows=read_csv(repo/UNIT_CSV,UNIT_FIELDS)
    for row in rows:row["development_rank"]=int(row["development_rank"]);row["shard"]=int(row["shard"])
    source_rows=source.load_units(repo)
    if rows!=source_rows:raise X2R2R1R1Error("unit roster differs from X2-R2-R1 source")
    return rows


def validate_coverage_states(rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    return source.validate_coverage_states(rows)


def phase_file(repo:Path,path:Path,terminal:Mapping[str,Any],name:str)->Path:
    local=path/name
    if local.is_file():return local
    matches=[repo/row["path"] for row in terminal["payload"]["files"] if Path(row["path"]).name==name]
    if len(matches)!=1:raise X2R2R1R1Error(f"phase file not uniquely bound: {name}")
    return matches[0]


def _fresh_root_precheck(repo:Path)->None:
    if (repo/ROOT).exists() or (repo/REPORT_ROOT).exists():raise X2R2R1R1Error("fresh X2-R2-R1-R1 roots must not exist before initialization")
    if [path for path in (repo/"evidence/stage_x").glob("x[3-5]*") if path.exists()]:raise X2R2R1R1Error("X3-X5 root exists before authorization")


def _validate_source_state(repo:Path)->dict[str,Any]:
    config=load_config(repo)
    failure=strict_read(repo/SOURCE_STAGE_FAILURE);source.validate_artifact(failure,repo,verify_files=True)
    clean=source.strict_read(repo/SOURCE_CLEAN_COVERAGE);source.validate_artifact(clean,repo,verify_files=True)
    gamma=source.strict_read(repo/SOURCE_GAMMA_SELECTION);source.validate_artifact(gamma,repo,verify_files=True)
    if failure["status"]!="BLOCKED_INFRASTRUCTURE" or failure["canonical_digest"]!=config["source_reuse"]["source_stage_failure_canonical_digest"]:
        raise X2R2R1R1Error("X2-R2-R1 stage failure binding drifted")
    if gamma["status"]!="PASS" or gamma["canonical_digest"]!=config["source_reuse"]["gamma_selection_canonical_digest"]:
        raise X2R2R1R1Error("X2-R2-R1 gamma selection binding drifted")
    gamma_state={row["name"]:row["value"] for row in gamma["payload"]["state"]}
    if float(gamma_state["selected_conv_gamma"])!=100.0 or float(gamma_state["selected_linear_gamma"])!=0.05:raise X2R2R1R1Error("selected gamma drifted")
    units=source.load_units(repo);samples=source.load_samples(repo);clean_rows=[];gamma_rows=[]
    for unit in units:
        clean_path=repo/SOURCE_ROOT/"units"/unit["unit_digest"]/"attempts/attempt-000001/clean/terminal.json"
        gamma_path=repo/SOURCE_ROOT/"units"/unit["unit_digest"]/"attempts/attempt-000001/gamma/terminal.json"
        clean_term=source.strict_read(clean_path);source.validate_artifact(clean_term,repo,verify_files=True)
        gamma_term=source.strict_read(gamma_path);source.validate_artifact(gamma_term,repo,verify_files=True)
        clean_state={row["name"]:row["value"] for row in clean_term["payload"]["state"]}
        status="VALID" if clean_term["status"]=="PASS" and clean_state.get("status")=="VALID" else clean_state.get("status")
        clean_rows.append({"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"status":status})
        gamma_rows.append(gamma_term)
    coverage=validate_coverage_states(clean_rows)
    if not coverage["pass"] or coverage["valid"]!=199 or len(gamma_rows)!=200:raise X2R2R1R1Error("source clean/gamma counts differ from authorization")
    if list((repo/SOURCE_ROOT).rglob("*pending*")):raise X2R2R1R1Error("source X2-R2-R1 pending files exist")
    if list((repo/SOURCE_ROOT/"units").glob("*/attempts/attempt-000001/formal/terminal.json")):raise X2R2R1R1Error("source X2-R2-R1 formal terminal unexpectedly exists")
    return {"config":config,"failure":failure,"clean":clean,"gamma":gamma,"units":units,"samples":samples,"coverage":coverage}


def run_mutation_suite(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=load_config(repo)
    if (repo/MUTATION_RECEIPT).exists():value=strict_read(repo/MUTATION_RECEIPT);validate_artifact(value,repo,verify_files=True);return value
    if list((repo/ROOT/"units").glob("*/attempts/attempt-000001/formal/terminal.json")):raise X2R2R1R1Error("cannot replace mutation suite after R1-R1 formal outcome access")
    checks=[];base=signed("stage_x_x2_r2_r1_r1_phase_manifest","PASS",payload(state={"holm_family_count":5,"full_model_ig_in_holm":False}))
    def reject(name:str,fn:Any)->None:
        try:fn()
        except Exception as error:checks.append({"check_id":name,"status":"PASS","detail":type(error).__name__})
        else:raise X2R2R1R1Error(f"mutation accepted: {name}")
    def mutate(fn:Any)->None:
        value=copy.deepcopy(base);fn(value);plain=dict(value);plain.pop("canonical_digest",None);value["canonical_digest"]=canonical_sha256(plain);validate_artifact(value,repo)
    reject("root_unknown_field",lambda:mutate(lambda value:value.__setitem__("UNKNOWN",True)))
    reject("payload_unknown_field",lambda:mutate(lambda value:value["payload"].__setitem__("UNKNOWN",True)))
    reject("bad_canonical_digest",lambda:validate_artifact({**base,"canonical_digest":"0"*64},repo))
    reject("bool_as_int_metric",lambda:mutate(lambda value:value["payload"]["metrics"].append({"metric_id":"bad","value":True,"unit":"count"})))
    models=[f"m{index}" for index in range(5)];good=[{"unit_digest":hashlib.sha256(f"{model}-{index}".encode()).hexdigest(),"model_id":model,"status":"VALID"} for model in models for index in range(40)]
    reject("expected_route_count_not_200",lambda:validate_coverage_states(good[:-1]))
    bad=copy.deepcopy(good)
    for index in range(11):bad[index]["status"]="FAILED_ROUTE_LOCAL_NUMERICAL"
    if validate_coverage_states(bad)["pass"]:raise X2R2R1R1Error("overall threshold below 190 accepted")
    checks.append({"check_id":"overall_threshold_not_190","status":"PASS","detail":"coverage_rejected"})
    with tempfile.TemporaryDirectory(prefix="x2-r2-r1-r1-mut-") as name:
        temp=Path(name);one=temp/"one.json";write_new(one,base,repo);reject("terminal_overwrite",lambda:write_new(one,base,repo))
    body=payload(identity={"suite":"X2_R2_R1_R1_PRE_OUTCOME_MUTATION_SUITE"},bindings=[file_record(repo,ARTIFACT_SCHEMA),file_record(repo,CONFIG,canonical_digest=config["canonical_digest"]),file_record(repo,AMENDMENT),source_artifact_record(repo,SOURCE_STAGE_FAILURE),source_artifact_record(repo,SOURCE_GAMMA_SELECTION)],checks=checks,counts={"attack_count":len(checks),"rejected_count":len(checks)},state={"outcome_access_before_suite":False,"all_mutations_rejected":True,"real_validators_used":True},terminal=terminal_state(terminal=True,immutable=True,complete=True))
    receipt=signed("stage_x_x2_r2_r1_r1_pre_outcome_mutation_receipt","PASS",body);write_new(repo/MUTATION_RECEIPT,receipt,repo);return receipt


def _write_required_files(repo:Path)->dict[str,Any]:
    files=["startup_gate.json","amendment_binding.json","x2_r2_r1_failure_binding.json","clean_gamma_reuse_binding.json","development_sample_manifest.json","development_unit_manifest.json","manifests/reuse_clean_routes.csv","manifests/reuse_gamma_routes.csv","model_materialization_binding.json","pre_outcome/mutation_suite.json","x2_r2_r1_r1_run_spec_v3.json","implementation_seal_v3.json"]
    missing=[name for name in files if not (repo/ROOT/name).is_file()]
    if missing:raise X2R2R1R1Error(f"required file missing before list publication: {missing}")
    value=signed("stage_x_x2_r2_r1_r1_preflight_worker","PASS",payload(identity={"phase":"formal","contract":"worker_required_files","required_files_json":json.dumps(files,sort_keys=True,separators=(",",":"))},files=[file_record(repo,ROOT/file) for file in files],counts={"required_file_count":len(files)},state={"explicit_required_files_list":True,"cpu_only":True,"model_free":True,"jpeg_free":True,"creates_unit_journal":False,"creates_unit_terminal":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)))
    write_new(repo/REQUIRED_FILES_JSON,value,repo);return value


def preflight_worker(repo_root:Path|str=REPO_ROOT,phase:str="formal")->dict[str,Any]:
    repo=Path(repo_root).resolve()
    if phase!="formal":raise X2R2R1R1Error("only formal preflight is authorized")
    imported=set(__import__("sys").modules)
    if "torch" in imported or "timm" in imported:raise X2R2R1R1Error("preflight must not import torch/timm")
    required_doc=strict_read(repo/REQUIRED_FILES_JSON);validate_artifact(required_doc,repo,verify_files=True)
    identity={row["name"]:row["value"] for row in required_doc["payload"]["identity"]};required=json.loads(str(identity["required_files_json"]))
    expected=[ROOT/"startup_gate.json",ROOT/"amendment_binding.json",ROOT/"x2_r2_r1_failure_binding.json",ROOT/"clean_gamma_reuse_binding.json",ROOT/"development_sample_manifest.json",ROOT/"development_unit_manifest.json",REUSE_CLEAN_CSV,REUSE_GAMMA_CSV,MODEL_BINDING,MUTATION_RECEIPT,RUN_SPEC,IMPLEMENTATION_SEAL]
    if [path.relative_to(ROOT).as_posix() for path in expected]!=required:raise X2R2R1R1Error("explicit required-files list drifted")
    checks=[]
    for relative in expected:
        value_path=repo/relative
        if not value_path.is_file():raise X2R2R1R1Error(f"preflight required file missing: {relative}")
        if value_path.suffix==".json":validate_artifact(strict_read(value_path),repo,verify_files=True)
        checks.append({"check_id":"read_"+relative.name,"status":"PASS","detail":relative.as_posix()})
    if list((repo/ROOT).rglob("*pending*")):raise X2R2R1R1Error("stale pending file exists before worker launch")
    after=set(__import__("sys").modules)
    if "torch" in after or "timm" in after:raise X2R2R1R1Error("preflight imported torch/timm")
    receipt=signed("stage_x_x2_r2_r1_r1_preflight_worker","PASS",payload(bindings=[artifact_record(repo,REQUIRED_FILES_JSON),artifact_record(repo,RUN_SPEC),artifact_record(repo,ROOT/"implementation_seal.json"),artifact_record(repo,MODEL_BINDING),artifact_record(repo,ROOT/"development_sample_manifest.json"),artifact_record(repo,ROOT/"development_unit_manifest.json")],checks=checks,counts={"required_files":len(expected),"unit_journals_created":0,"unit_terminals_created":0},state={"phase":phase,"cpu_only":True,"model_free":True,"jpeg_free":True,"torch_imported":False,"timm_imported":False,"jpeg_decoded":False,"model_loaded":False,"scientific_kernel_started":False,"stale_pending":False},terminal=terminal_state(terminal=True,immutable=True,complete=True,next_authorization="X2_R2_R1_R1_FORMAL_STABILITY_INVERSE_MAP_SHAPE_RECOVERY_ONLY")))
    if (repo/PREFLIGHT_RECEIPT).exists():value=strict_read(repo/PREFLIGHT_RECEIPT);validate_artifact(value,repo,verify_files=True);return value
    write_new(repo/PREFLIGHT_RECEIPT,receipt,repo);return receipt


def seal_implementation(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();path=repo/IMPLEMENTATION_SEAL
    if path.exists():value=strict_read(path);validate_artifact(value,repo,verify_files=True);return value
    for relative in IMPLEMENTATION_FILES:
        if not (repo/relative).is_file():raise X2R2R1R1Error(f"implementation file missing: {relative}")
    bindings=[file_record(repo,Path(name)) for name in IMPLEMENTATION_FILES]+[artifact_record(repo,RUN_SPEC),artifact_record(repo,MUTATION_RECEIPT),artifact_record(repo,MODEL_BINDING)]
    superseded=[source_artifact_record(repo,path) for path in (*SUPERSEDED_RUN_SPECS,*SUPERSEDED_IMPLEMENTATION_SEALS) if (repo/path).is_file()]
    value=signed("stage_x_x2_r2_r1_r1_implementation_seal","PASS",payload(bindings=[*bindings,*superseded],counts={"implementation_files":len(IMPLEMENTATION_FILES)},state={"sealed_before_x2_r2_r1_r1_outcome_access":True,"source_changes_after_seal_allowed":False,"supersedes_pre_outcome_control_seal":bool(superseded),"holm_family_count":5,"full_model_ig_in_holm":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)))
    write_new(path,value,repo);return value


def _phase_reuse_row(repo:Path,unit:Mapping[str,Any],phase:str)->dict[str,Any]:
    path=repo/SOURCE_ROOT/"units"/unit["unit_digest"]/"attempts/attempt-000001"/phase
    terminal=source.strict_read(path/"terminal.json");source.validate_artifact(terminal,repo,verify_files=True)
    return {"unit_digest":unit["unit_digest"],"phase":phase,"source_terminal_path":(path/"terminal.json").relative_to(repo).as_posix(),"source_terminal_sha256":sha_file(path/"terminal.json"),"source_terminal_canonical_digest":terminal["canonical_digest"],"source_tree_digest":_tree_digest(path),"status":terminal["status"],"reuse_status":"SOURCE_VALIDATED_REUSE"}


def _publish_phase_reuse(repo:Path,unit:Mapping[str,Any],phase:str)->dict[str,Any]:
    source_path=repo/SOURCE_ROOT/"units"/unit["unit_digest"]/"attempts/attempt-000001"/phase
    source_terminal=source.strict_read(source_path/"terminal.json");source.validate_artifact(source_terminal,repo,verify_files=True)
    status=source_terminal["status"];state={row["name"]:row["value"] for row in source_terminal["payload"]["state"]}
    state.update({"source_stage":"X2-R2-R1","source_validated_reuse":True,"large_arrays_copied":False})
    counts={row["name"]:row["value"] for row in source_terminal["payload"]["counts"]};counts["source_validated_reuse"]=1
    body=payload(identity={"attempt_id":REUSE_ATTEMPT_ID,"phase":phase,"unit_digest":unit["unit_digest"]},bindings=[source_artifact_record(repo,(source_path/"terminal.json").relative_to(repo))],files=source_terminal["payload"]["files"],counts=counts,state=state,resources=source_terminal["payload"]["resources"],terminal=terminal_state(terminal=True,immutable=True,complete=True))
    terminal=signed("stage_x_x2_r2_r1_r1_phase_manifest",status,body)
    path=repo/ROOT/"units"/unit["unit_digest"]/"attempts"/ATTEMPT_ID/phase/"terminal.json";write_new(path,terminal,repo);return terminal


def _publish_failed_unit_reuse(repo:Path,unit:Mapping[str,Any])->None:
    source_terminal_path=SOURCE_ROOT/"units"/unit["unit_digest"]/"terminal.json"
    source_terminal=source.strict_read(repo/source_terminal_path);source.validate_artifact(source_terminal,repo,verify_files=True)
    state={row["name"]:row["value"] for row in source_terminal["payload"]["state"]};state.update({"source_stage":"X2-R2-R1","source_validated_reuse":True})
    counts={row["name"]:row["value"] for row in source_terminal["payload"]["counts"]}
    terminal=signed("stage_x_x2_r2_r1_r1_unit_terminal","FAILED",payload(identity={"attempt_id":ATTEMPT_ID,"unit_digest":unit["unit_digest"],"model_id":unit["model_id"],"sample_id":unit["sample_id"],"sample_sha256":unit["sample_sha256"]},bindings=[source_artifact_record(repo,source_terminal_path),artifact_record(repo,ROOT/"units"/unit["unit_digest"]/"attempts"/ATTEMPT_ID/"clean/terminal.json"),artifact_record(repo,ROOT/"units"/unit["unit_digest"]/"attempts"/ATTEMPT_ID/"gamma/terminal.json")],counts=counts,state=state,terminal=terminal_state(terminal=True,immutable=True,complete=True)))
    publish_atomic(repo/ROOT/"units"/unit["unit_digest"]/"terminal.pending.json",repo/ROOT/"units"/unit["unit_digest"]/"terminal.json",terminal,repo)


def initialize(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();_fresh_root_precheck(repo);state=_validate_source_state(repo);config=state["config"];units=state["units"];samples=state["samples"]
    root=repo/ROOT;root.mkdir(parents=True,exist_ok=False);(root/"pre_outcome").mkdir(parents=True,exist_ok=False);(root/"manifests").mkdir(parents=True,exist_ok=False)
    mutation=run_mutation_suite(repo)
    write_csv_new(repo/SAMPLE_CSV,SAMPLE_FIELDS,samples);write_csv_new(repo/UNIT_CSV,UNIT_FIELDS,units)
    clean_rows=[_phase_reuse_row(repo,unit,"clean") for unit in units];gamma_rows=[_phase_reuse_row(repo,unit,"gamma") for unit in units]
    write_csv_new(repo/REUSE_CLEAN_CSV,REUSE_PHASE_FIELDS,clean_rows);write_csv_new(repo/REUSE_GAMMA_CSV,REUSE_PHASE_FIELDS,gamma_rows)
    common={"holm_family_count":5,"full_model_ig_in_holm":False,"x3_started":False,"x4_started":False,"x5_started":False}
    startup=signed("stage_x_x2_r2_r1_r1_startup_gate","PASS",payload(bindings=[file_record(repo,CONFIG,canonical_digest=config["canonical_digest"]),file_record(repo,AMENDMENT),source_artifact_record(repo,SOURCE_STAGE_FAILURE),source_artifact_record(repo,SOURCE_CLEAN_COVERAGE),source_artifact_record(repo,SOURCE_GAMMA_SELECTION),artifact_record(repo,MUTATION_RECEIPT)],counts={"expected_routes":200,"source_clean_valid":199,"source_clean_failed_route_local_numerical":1,"source_gamma_terminals":200,"formal_source_terminals":0},state={**common,"fresh_root":True,"source_stage_failure_immutable":True,"formal_only_rerun":True,"large_arrays_copied":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(root/"startup_gate.json",startup,repo)
    amendment=signed("stage_x_x2_r2_r1_r1_amendment_binding","PASS",payload(bindings=[file_record(repo,AMENDMENT),file_record(repo,CONFIG,canonical_digest=config["canonical_digest"])],state={**common,"prospective_before_x2_r2_r1_r1_outcome":True,"same_root_retry_allowed":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(root/"amendment_binding.json",amendment,repo)
    failure_binding=signed("stage_x_x2_r2_r1_r1_x2_r2_r1_failure_binding","PASS",payload(bindings=[source_artifact_record(repo,SOURCE_STAGE_FAILURE)],counts={"formal_terminals":0,"completed_units_before_stop":0},state={**common,"blocked_cause":"inverse_map_2d_attribution_shape_handling","failure_type":"IndexError","source_stage_failure_immutable":True},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(root/"x2_r2_r1_failure_binding.json",failure_binding,repo)
    reuse_binding=signed("stage_x_x2_r2_r1_r1_clean_gamma_reuse_binding","PASS",payload(bindings=[source_artifact_record(repo,SOURCE_CLEAN_COVERAGE),source_artifact_record(repo,SOURCE_GAMMA_SELECTION)],files=[file_record(repo,REUSE_CLEAN_CSV,canonical_digest=canonical_sha256(read_csv(repo/REUSE_CLEAN_CSV,REUSE_PHASE_FIELDS))),file_record(repo,REUSE_GAMMA_CSV,canonical_digest=canonical_sha256(read_csv(repo/REUSE_GAMMA_CSV,REUSE_PHASE_FIELDS)))],counts={"clean_terminals_source_validated":200,"clean_valid":199,"clean_failed_route_local_numerical":1,"gamma_terminals_source_validated":200,"gamma_arrays_copied":0},state={**common,"source_validated_reuse_only":True,"large_arrays_copied":False,"selected_conv_gamma":100.0,"selected_linear_gamma":0.05},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(root/"clean_gamma_reuse_binding.json",reuse_binding,repo)
    sample_manifest=signed("stage_x_x2_r2_r1_r1_sample_manifest","PASS",payload(bindings=[source_artifact_record(repo,SOURCE_ROOT/"development_sample_manifest.json")],files=[file_record(repo,SAMPLE_CSV,canonical_digest=canonical_sha256(read_csv(repo/SAMPLE_CSV,SAMPLE_FIELDS)))],counts={"sample_count":40,"unique_sample_count":40,"model_count":5},state={**common,"source_roster_reused":True,"reselection_performed":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(root/"development_sample_manifest.json",sample_manifest,repo)
    unit_manifest=signed("stage_x_x2_r2_r1_r1_unit_manifest","PASS",payload(bindings=[source_artifact_record(repo,SOURCE_ROOT/"development_unit_manifest.json")],files=[file_record(repo,SAMPLE_CSV,canonical_digest=canonical_sha256(read_csv(repo/SAMPLE_CSV,SAMPLE_FIELDS))),file_record(repo,UNIT_CSV,canonical_digest=canonical_sha256(read_csv(repo/UNIT_CSV,UNIT_FIELDS))),file_record(repo,REUSE_CLEAN_CSV,canonical_digest=canonical_sha256(read_csv(repo/REUSE_CLEAN_CSV,REUSE_PHASE_FIELDS))),file_record(repo,REUSE_GAMMA_CSV,canonical_digest=canonical_sha256(read_csv(repo/REUSE_GAMMA_CSV,REUSE_PHASE_FIELDS)))],counts={"unit_count":200,"unique_unit_count":200,"sample_count":40,"model_count":5,"clean_valid":199,"clean_failed_route_local_numerical":1,"gamma_reuse_count":200},state={**common,"route_roster_unchanged":True,"expected_denominator":200,"paired_effect_roster_clean_valid_only":True},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(root/"development_unit_manifest.json",unit_manifest,repo)
    source_model=source.strict_read(repo/source.MODEL_BINDING);source.validate_artifact(source_model,repo,verify_files=True)
    model_doc=signed("stage_x_x2_r2_r1_r1_model_materialization","PASS",payload(bindings=[source_artifact_record(repo,source.MODEL_BINDING)],counts={"model_count":5,"checkpoint_redownloaded":0},state={**common,"cache_reused":True,"strict_load_required_at_worker":True},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(repo/MODEL_BINDING,model_doc,repo)
    run=signed("stage_x_x2_r2_r1_r1_run_spec","PASS",payload(identity={"authorization":config["authorization"],"run_id":RUN_ID},bindings=[artifact_record(repo,root.relative_to(repo)/"startup_gate.json"),artifact_record(repo,root.relative_to(repo)/"amendment_binding.json"),artifact_record(repo,root.relative_to(repo)/"x2_r2_r1_failure_binding.json"),artifact_record(repo,root.relative_to(repo)/"clean_gamma_reuse_binding.json"),artifact_record(repo,root.relative_to(repo)/"development_sample_manifest.json"),artifact_record(repo,root.relative_to(repo)/"development_unit_manifest.json"),artifact_record(repo,MODEL_BINDING),artifact_record(repo,MUTATION_RECEIPT),source_artifact_record(repo,SOURCE_RUN_SPEC),source_artifact_record(repo,SOURCE_IMPLEMENTATION_SEAL)],counts={"expected_routes":200,"formal_candidate_routes":199,"gamma_candidates":15},state={**common,"formal_only_rerun":True,"original_schedule_unchanged":True,"tolerances_unchanged":True,"transformations_unchanged":True,"pre_outcome_receipt_sha256":sha_file(repo/MUTATION_RECEIPT),"same_root_retry_allowed":False},terminal=terminal_state(terminal=True,immutable=True,complete=True)));write_new(repo/RUN_SPEC,run,repo)
    seal_implementation(repo);_write_required_files(repo);preflight_worker(repo,"formal")
    for unit in units:
        clean=_publish_phase_reuse(repo,unit,"clean");gamma=_publish_phase_reuse(repo,unit,"gamma")
        state_clean={row["name"]:row["value"] for row in clean["payload"]["state"]}
        if clean["status"]=="FAILED" and state_clean.get("status")=="FAILED_ROUTE_LOCAL_NUMERICAL":_publish_failed_unit_reuse(repo,unit)
    return {"status":"PASS","source_clean_reuse":200,"source_gamma_reuse":200,"formal_ready":199}


def reseal_pre_outcome_control(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve()
    if not (repo/ROOT).is_dir():raise X2R2R1R1Error("R1-R1 root is not initialized")
    if list((repo/ROOT/"units").glob("*/terminal.json")) or list((repo/ROOT/"units").glob("*/attempts/attempt-000001/formal/terminal.json")) or (repo/ROOT/"stage_failure.json").exists():
        raise X2R2R1R1Error("cannot reseal after R1-R1 outcome access")
    state=_validate_source_state(repo);config=state["config"]
    superseded=[source_artifact_record(repo,path) for path in (*SUPERSEDED_RUN_SPECS,*SUPERSEDED_IMPLEMENTATION_SEALS) if (repo/path).is_file()]
    run=signed("stage_x_x2_r2_r1_r1_run_spec","PASS",payload(identity={"authorization":config["authorization"],"run_id":RUN_ID},bindings=[artifact_record(repo,ROOT/"startup_gate.json"),artifact_record(repo,ROOT/"amendment_binding.json"),artifact_record(repo,ROOT/"x2_r2_r1_failure_binding.json"),artifact_record(repo,ROOT/"clean_gamma_reuse_binding.json"),artifact_record(repo,ROOT/"development_sample_manifest.json"),artifact_record(repo,ROOT/"development_unit_manifest.json"),artifact_record(repo,MODEL_BINDING),artifact_record(repo,MUTATION_RECEIPT),source_artifact_record(repo,SOURCE_RUN_SPEC),source_artifact_record(repo,SOURCE_IMPLEMENTATION_SEAL),*superseded],counts={"expected_routes":200,"formal_candidate_routes":199,"gamma_candidates":15},state={"holm_family_count":5,"full_model_ig_in_holm":False,"x3_started":False,"x4_started":False,"x5_started":False,"formal_only_rerun":True,"original_schedule_unchanged":True,"tolerances_unchanged":True,"transformations_unchanged":True,"pre_outcome_receipt_sha256":sha_file(repo/MUTATION_RECEIPT),"same_root_retry_allowed":False,"supersedes_pre_outcome_control_seal":True},terminal=terminal_state(terminal=True,immutable=True,complete=True)))
    if not (repo/RUN_SPEC).exists():write_new(repo/RUN_SPEC,run,repo)
    seal_implementation(repo)
    if not (repo/REQUIRED_FILES_JSON).exists():_write_required_files(repo)
    return {"status":"PASS","run_spec":RUN_SPEC.as_posix(),"implementation_seal":IMPLEMENTATION_SEAL.as_posix(),"preflight_required":REQUIRED_FILES_JSON.as_posix()}


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("mutate","initialize","seal","reseal-pre-outcome-control","preflight-worker"));parser.add_argument("--repo-root",type=Path,default=REPO_ROOT);parser.add_argument("--phase",default="formal");args=parser.parse_args(argv)
    try:
        if args.command=="preflight-worker":value=preflight_worker(args.repo_root,args.phase)
        else:value={"mutate":run_mutation_suite,"initialize":initialize,"seal":seal_implementation,"reseal-pre-outcome-control":reseal_pre_outcome_control}[args.command](args.repo_root)
    except Exception as error:
        print(json.dumps({"status":"BLOCKED","error":f"{type(error).__name__}: {error}"},sort_keys=True));return 2
    if isinstance(value,dict) and "document_type" in value:
        output={"status":value["status"],"document_type":value["document_type"],"canonical_digest":value["canonical_digest"]}
    else:
        output=value
    print(json.dumps(output,sort_keys=True,indent=2));return 0


if __name__=="__main__": raise SystemExit(main())
