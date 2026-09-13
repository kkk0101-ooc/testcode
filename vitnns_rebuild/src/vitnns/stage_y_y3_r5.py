"""Prospective Stage Y3-R5 recovery contracts (Torch-free at import)."""
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
from typing import Any,Mapping,Sequence
from jsonschema import Draft202012Validator
from . import stage_y_y3 as y3

TOKEN="STAGE_Y3_R5_GLOBAL_BUDGET_GATE_RECOVERY_ONLY"
STARTUP_COMMIT="fb431aa1792a671d790baf224773e327ec017f21"
ROOT=Path("evidence/stage_y/y3_r5_perturbation_formal_20260913/formal")
REPORT_ROOT=Path("reports/stage_y_y3_r5_20260913")
SOURCE_ROOT=Path("evidence/stage_y/y3_r4_perturbation_formal_20260912/formal")
SOURCE_REPORT_ROOT=Path("reports/stage_y_y3_r4_20260912")
SPEC=Path("config/stage_y/y3_r5_run_spec.json")
SPEC_SCHEMA=Path("config/stage_y/y3_r5_run_spec.schema.json")
ARTIFACT_SCHEMA=Path("config/stage_y/y3_r5_artifact.schema.json")
DOC=Path("docs/stage_y_y3_r5_budget_gate_recovery_20260913.md")
NEW_INDICES=(141,145,149,153,157)
PHYSICAL_INDICES=(1,)
LOGICAL_SHARD=1
SOURCE_VALID_COUNT=155
PRODUCER_REF="refs/archive/stage-y-y3-r5-global-budget-gate-recovery-producer"
PASS_REF="refs/archive/stage-y-y3-r5-perturbation-mechanism-evidence"
BLOCKED_REF="refs/archive/stage-y-y3-r5-perturbation-mechanism-evidence-blocked"
R4_PRODUCER_REF=y3.PRODUCER_REF
R4_PRODUCER_COMMIT="e46e7500e1c1b3ef00f96f0bdd907070b70ec11e"
R4_BLOCKED_REF=y3.BLOCKED_REF
R4_BLOCKED_COMMIT=STARTUP_COMMIT
IMPLEMENTATION_SEAL="producer_implementation_seal.json"
PRODUCER_FILES=(DOC,SPEC,SPEC_SCHEMA,ARTIFACT_SCHEMA,Path("src/vitnns/stage_y_y3_r5.py"),Path("src/vitnns/stage_y_y3_r5_worker.py"),Path("scripts/run_stage_y_y3_r5.py"),Path("scripts/validate_stage_y_y3_r5.py"),Path("tests/test_stage_y_y3_r5.py"))
DEPENDENCIES=tuple(dict.fromkeys(y3.PRODUCER_FILES+y3.DEPENDENCIES))

class R5Error(RuntimeError):code="BLOCKED_PROVENANCE"
class StartupError(R5Error):code="BLOCKED_STARTUP_STATE"
class BudgetError(R5Error):code="BLOCKED_BUDGET"
class ScheduleError(R5Error):code="BLOCKED_RESOURCE_SCHEDULE"
class ProvenanceError(R5Error):code="BLOCKED_PROVENANCE"
class InfrastructureError(R5Error):code="BLOCKED_INFRASTRUCTURE"

def require(flag:Any,message:str,error:type[R5Error]=ProvenanceError)->None:
    if not flag:raise error(message)

def obj(properties:Mapping[str,Any],optional:Sequence[str]=())->dict[str,Any]:
    return {"type":"object","additionalProperties":False,"properties":dict(properties),"required":[k for k in properties if k not in optional]}
def arr(item:Mapping[str,Any],count:int|None=None)->dict[str,Any]:
    value={"type":"array","items":dict(item)}
    if count is not None:value.update(minItems=count,maxItems=count)
    return value
def nullable(item:Mapping[str,Any])->dict[str,Any]:return {"anyOf":[dict(item),{"type":"null"}]}
S={"type":"string"};I={"type":"integer","minimum":0};N={"type":"number"};NN={"type":"number","minimum":0};B={"type":"boolean"};H={"type":"string","pattern":"^[0-9a-f]{64}$"};G={"type":"string","pattern":"^[0-9a-f]{40}$"}
FILE=y3.FILE;RESOURCE=y3.RESOURCE;TARGET=y3.TARGET

def artifact_schema()->dict[str,Any]:
    terminal=y3.artifact_schema()["$defs"]["unit_terminal"]
    variants={
      "pending":obj({"started_unix_ns":I,"next_authorization":{"const":"NONE_WHILE_PENDING"}}),
      "producer_preflight":obj({"startup_commit":{"const":STARTUP_COMMIT},"checks":arr(S),"focused_test_count":I,"false_pass_attack_count":I,"source_valid_count":{"const":155},"new_compute_count":{"const":5},"formal_outcome_access":{"const":False},"jpeg_decoded":{"const":False},"model_loaded":{"const":False},"cuda_initialized":{"const":False}}),
      "binding":obj({"name":S,"files":arr(FILE),"files_digest":H,"canonical_digest_bound":nullable(H),"commit":nullable(G)}),
      "implementation_seal":obj({"producer_commit":G,"producer_ref":{"const":PRODUCER_REF},"source_files":arr(FILE),"files_digest":H}),
      "startup_gate":obj({"locked_worker_count":{"const":1},"devices":arr(y3.artifact_schema()["$defs"]["startup_gate"]["properties"]["devices"]["items"],1),"python_executable_realpath":S,"environment_digest":H,"numeric_probe_digest":H,"dynamic_rebalancing":{"const":False},"hot_add":{"const":False},"started_unix_ns":I}),
      "coordinate_manifest":obj({"records":arr(obj({"unit_index":{"enum":list(NEW_INDICES)},"subset_rank":I,"selection_order":I,"source_object_id":S,"condition_index":I,"condition_id":{"enum":list(y3.CONDITIONS)},"logical_shard_id":{"const":1},"fixed_target":TARGET,"source_terminal":FILE}),5),"record_count":{"const":5},"records_digest":H}),
      "source_rebind_manifest":obj({"records":arr(obj({"unit_index":I,"predecessor_terminal":FILE}),155),"record_count":{"const":155},"records_digest":H,"arrays_copied":{"const":False}}),
      "budget_admission":obj({"source_ledger":FILE,"measured_valid_resource_count":{"const":153},"new_compute_count":{"const":5},"maximum_unit":obj({"gpu_active_seconds":NN,"cpu_seconds":NN,"wall_seconds":NN,"retained_bytes":I,"peak_reserved_gpu_bytes":I}),"projected":obj({"gpu_seconds":NN,"cpu_seconds":NN,"wall_seconds":NN,"retained_bytes":I,"peak_reserved_gpu_bytes":I}),"caps_pass":{"const":True},"divide_by_worker_count":{"const":False}}),
      "unit_pending":y3.artifact_schema()["$defs"]["unit_pending"],
      "attempt":y3.artifact_schema()["$defs"]["attempt"],
      "transform":y3.artifact_schema()["$defs"]["transform"],
      "prediction":y3.artifact_schema()["$defs"]["prediction"],
      "array_inventory":y3.artifact_schema()["$defs"]["array_inventory"],
      "scientific_result":y3.artifact_schema()["$defs"]["scientific_result"],
      "unit_terminal":terminal,
      "resource":RESOURCE,
      "shard_pending":obj({"logical_shard_id":{"const":1},"started_unix_ns":I}),
      "shard_terminal":obj({"logical_shard_id":{"const":1},"expected_unit_count":{"const":5},"terminal_count":{"const":5},"terminal_counts":obj({k:I for k in y3.TERMINAL_STATES}),"fixed_initialization_resource":RESOURCE,"worker_wall_seconds":NN,"completed_unix_ns":I}),
      "worker_identity":y3.artifact_schema()["$defs"]["worker_identity"],
      "launch_result":obj({"returncode":I,"stdout":S,"stderr":S}),
      "resource_ledger":obj({"r4_ledger":FILE,"r4_accounted_gpu_seconds":NN,"r4_accounted_cpu_seconds":NN,"r4_stage_wall_seconds":NN,"r4_retained_bytes":I,"r5_gpu_seconds":NN,"r5_cpu_seconds":NN,"r5_stage_wall_seconds":NN,"r5_retained_bytes":I,"combined_gpu_seconds":NN,"combined_cpu_seconds":NN,"combined_stage_wall_seconds":NN,"combined_retained_bytes":I,"peak_reserved_gpu_bytes":I,"caps":obj({k:NN for k in y3.CAPS}),"caps_pass":B}),
      "reconciliation":obj({"denominator_images":{"const":20},"expected_nonclean_units":{"const":160},"source_rebind_count":{"const":155},"new_compute_count":{"const":5},"terminal_count":{"const":160},"valid_count":I,"complete_image_count":I,"prediction_flip_count":I,"condition_counts":obj({c:obj({k:I for k in y3.TERMINAL_STATES}) for c in y3.CONDITIONS}),"blocking_states":arr(S),"bootstrap_executed":{"const":False},"holm_executed":{"const":False},"claim_decision_executed":{"const":False},"next_authorization":{"enum":[y3.NEXT,"NONE"]}}),
      "validation_report":obj({"checks":arr(S),"source_recomputed":{"const":True},"terminal_count":{"const":160},"valid_count":I,"complete_image_count":I,"prediction_flip_count":I,"records_digest":H}),
      "false_pass_audit":y3.artifact_schema()["$defs"]["false_pass_audit"],
      "access_audit":y3.artifact_schema()["$defs"]["access_audit"],
      "inventory":y3.artifact_schema()["$defs"]["inventory"],
      "failure":y3.artifact_schema()["$defs"]["failure"],
      "completion":obj({"workflow_status":{"enum":["PASS",*y3.TERMINAL_STATES[1:]]},"denominator_images":{"const":20},"expected_nonclean_units":{"const":160},"source_rebind_count":{"const":155},"new_compute_count":{"const":5},"terminal_count":{"const":160},"complete_image_count":I,"formal_outcome_access":{"const":False},"bootstrap_executed":{"const":False},"holm_executed":{"const":False},"claim_decision_executed":{"const":False},"next_authorization":{"enum":[y3.NEXT,"NONE"]},"reconciliation":FILE,"validation_report":FILE,"whole_root_inventory":FILE,"payload_manifest":FILE,"report":FILE,"failure":nullable(FILE)}),
    }
    envelope=obj({"$schema":{"const":ARTIFACT_SCHEMA.name},"schema_version":{"const":1},"protocol_id":{"const":y3.PROTOCOL_ID},"stage_id":{"const":"Y3-R5"},"document_type":{"enum":list(variants)},"status":{"enum":["PENDING","PASS",*y3.TERMINAL_STATES]},"payload":{},"canonical_digest":H})
    envelope.update({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":ARTIFACT_SCHEMA.name,"$defs":variants,"allOf":[{"if":{"properties":{"document_type":{"const":k}}},"then":{"properties":{"payload":{"$ref":"#/$defs/"+k}}}} for k in variants]})
    return envelope

canonical=y3.canonical;read=y3.read;sha=y3.sha;file_record=y3.file_record;canonical_bytes=y3.canonical_bytes
def validate_document(value:Mapping[str,Any])->None:
    unsigned=dict(value);claimed=unsigned.pop("canonical_digest",None);require(claimed==canonical(unsigned),"canonical digest mismatch");Draft202012Validator(artifact_schema()).validate(value)
def signed(kind:str,payload:Mapping[str,Any],status:str="PASS")->dict[str,Any]:
    value={"$schema":ARTIFACT_SCHEMA.name,"schema_version":1,"protocol_id":y3.PROTOCOL_ID,"stage_id":"Y3-R5","document_type":kind,"status":status,"payload":dict(payload)};value["canonical_digest"]=canonical(value);validate_document(value);return value
def write(path:Path,kind:str,payload:Mapping[str,Any],status:str="PASS")->dict[str,Any]:
    value=signed(kind,payload,status);path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream:stream.write(canonical_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY);os.fsync(directory);os.close(directory);return value
def publish_terminal(pending:Path,path:Path,value:Mapping[str,Any])->None:
    validate_document(value);require(pending.is_file() and not path.exists(),"pending/terminal transition invalid");temp=path.with_name(path.name+".tmp");fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream:stream.write(canonical_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path);pending.unlink();directory=os.open(path.parent,os.O_RDONLY);os.fsync(directory);os.close(directory)
def source_files(repo:Path)->list[dict[str,Any]]:return [file_record(repo/p,repo) for p in PRODUCER_FILES+DEPENDENCIES]
def verify_implementation(repo:Path)->str:
    seal=read(repo/ROOT/IMPLEMENTATION_SEAL);validate_document(seal);q=seal["payload"];commit=q["producer_commit"];require(y3.git(repo,"rev-parse",PRODUCER_REF)==commit,"producer ref drift");rows=source_files(repo);require(rows==q["source_files"] and canonical({"files":rows})==q["files_digest"],"producer source drift");top=Path(y3.git(repo,"rev-parse","--show-toplevel"))
    for row in rows:
        rel=(repo/row["path"]).relative_to(top);blob=subprocess.check_output(["git","show",f"{commit}:{rel}"],cwd=repo)
        if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            lines=blob.decode().splitlines();require(lines[1].endswith(row["sha256"]) and int(lines[2].split()[1])==row["bytes"],"producer LFS pointer drift")
        else:require(hashlib.sha256(blob).hexdigest()==row["sha256"],"producer Git blob drift: "+row["path"])
    return commit
def write_schemas(repo:Path)->None:
    spec=read(repo/SPEC);pairs=((repo/SPEC_SCHEMA,{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":SPEC_SCHEMA.name,"const":spec}),(repo/ARTIFACT_SCHEMA,artifact_schema()))
    for path,value in pairs:
        require(not path.exists(),"schema exists",StartupError);path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(fd,"wb") as stream:stream.write(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()+b"\n");stream.flush();os.fsync(stream.fileno())
        directory=os.open(path.parent,os.O_RDONLY);os.fsync(directory);os.close(directory)
