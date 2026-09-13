"""Torch-free Y3-R6 validator-only recovery contracts."""
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
from typing import Any,Mapping,Sequence
from jsonschema import Draft202012Validator
from . import stage_y_y3 as y3
from . import stage_y_y3_r5 as r5
TOKEN="STAGE_Y3_R6_VALIDATOR_ONLY_RECONCILIATION_RECOVERY"
STARTUP_COMMIT="b027612e1963bb8636e575d8f6dc44d71ee0c302"
ROOT=Path("evidence/stage_y/y3_r6_reconciliation_formal_20260913/formal")
REPORT_ROOT=Path("reports/stage_y_y3_r6_20260913")
SPEC=Path("config/stage_y/y3_r6_run_spec.json");SPEC_SCHEMA=Path("config/stage_y/y3_r6_run_spec.schema.json");ARTIFACT_SCHEMA=Path("config/stage_y/y3_r6_artifact.schema.json");DOC=Path("docs/stage_y_y3_r6_validator_recovery_20260913.md")
PRODUCER_REF="refs/archive/stage-y-y3-r6-validator-recovery-producer";PASS_REF="refs/archive/stage-y-y3-r6-perturbation-mechanism-evidence";BLOCKED_REF="refs/archive/stage-y-y3-r6-perturbation-mechanism-evidence-blocked"
R4_COMMIT=r5.R4_BLOCKED_COMMIT;R4_REF=r5.R4_BLOCKED_REF;R5_COMMIT=STARTUP_COMMIT;R5_REF=r5.BLOCKED_REF;IMPLEMENTATION_SEAL="producer_implementation_seal.json"
PRODUCER_FILES=(DOC,SPEC,SPEC_SCHEMA,ARTIFACT_SCHEMA,Path("src/vitnns/stage_y_y3_r6.py"),Path("scripts/run_stage_y_y3_r6.py"),Path("scripts/validate_stage_y_y3_r6.py"),Path("tests/test_stage_y_y3_r6.py"))
DEPENDENCIES=tuple(dict.fromkeys(r5.PRODUCER_FILES+r5.DEPENDENCIES))
class R6Error(RuntimeError):code="BLOCKED_PROVENANCE"
class StartupError(R6Error):code="BLOCKED_STARTUP_STATE"
class ProvenanceError(R6Error):code="BLOCKED_PROVENANCE"
def require(flag:Any,message:str,error:type[R6Error]=ProvenanceError):
    if not flag:raise error(message)
def obj(props:Mapping[str,Any],optional:Sequence[str]=()):return {"type":"object","additionalProperties":False,"properties":dict(props),"required":[k for k in props if k not in optional]}
def arr(item:Mapping[str,Any],count:int|None=None):
    value={"type":"array","items":dict(item)}
    if count is not None:value.update(minItems=count,maxItems=count)
    return value
def nullable(item):return {"anyOf":[dict(item),{"type":"null"}]}
S={"type":"string"};I={"type":"integer","minimum":0};NN={"type":"number","minimum":0};B={"type":"boolean"};H={"type":"string","pattern":"^[0-9a-f]{64}$"};G={"type":"string","pattern":"^[0-9a-f]{40}$"};FILE=y3.FILE
def artifact_schema():
    source=obj({"unit_index":I,"source_stage":{"enum":["Y3-R4","Y3-R5"]},"source_terminal":FILE})
    variants={
      "pending":obj({"started_unix_ns":I,"next_authorization":{"const":"NONE_WHILE_PENDING"}}),
      "producer_preflight":obj({"startup_commit":{"const":STARTUP_COMMIT},"checks":arr(S),"focused_test_count":I,"false_pass_attack_count":I,"source_count":{"const":160},"new_empirical_attempt_count":{"const":0},"gpu_execution":{"const":False},"jpeg_decoded":{"const":False},"model_loaded":{"const":False},"torch_imported":{"const":False}}),
      "binding":obj({"name":S,"files":arr(FILE),"files_digest":H,"canonical_digest_bound":nullable(H),"commit":nullable(G)}),
      "implementation_seal":obj({"producer_commit":G,"producer_ref":{"const":PRODUCER_REF},"source_files":arr(FILE),"files_digest":H}),
      "source_manifest":obj({"records":arr(source,160),"record_count":{"const":160},"records_digest":H,"comparison_key":{"const":"unit_index"},"arrays_copied":{"const":False}}),
      "resource_ledger":r5.artifact_schema()["$defs"]["resource_ledger"],
      "false_pass_audit":r5.artifact_schema()["$defs"]["false_pass_audit"],
      "access_audit":r5.artifact_schema()["$defs"]["access_audit"],
      "inventory":r5.artifact_schema()["$defs"]["inventory"],
      "failure":r5.artifact_schema()["$defs"]["failure"],
      "reconciliation":r5.artifact_schema()["$defs"]["reconciliation"],
      "validation_report":r5.artifact_schema()["$defs"]["validation_report"],
      "completion":r5.artifact_schema()["$defs"]["completion"],
    }
    envelope=obj({"$schema":{"const":ARTIFACT_SCHEMA.name},"schema_version":{"const":1},"protocol_id":{"const":y3.PROTOCOL_ID},"stage_id":{"const":"Y3-R6"},"document_type":{"enum":list(variants)},"status":{"enum":["PENDING","PASS",*y3.TERMINAL_STATES]},"payload":{},"canonical_digest":H});envelope.update({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":ARTIFACT_SCHEMA.name,"$defs":variants,"allOf":[{"if":{"properties":{"document_type":{"const":k}}},"then":{"properties":{"payload":{"$ref":"#/$defs/"+k}}}} for k in variants]});return envelope
canonical=y3.canonical;read=y3.read;sha=y3.sha;file_record=y3.file_record;canonical_bytes=y3.canonical_bytes
def validate_document(value):unsigned=dict(value);claimed=unsigned.pop("canonical_digest",None);require(claimed==canonical(unsigned),"canonical digest mismatch");Draft202012Validator(artifact_schema()).validate(value)
def signed(kind,payload,status="PASS"):
    value={"$schema":ARTIFACT_SCHEMA.name,"schema_version":1,"protocol_id":y3.PROTOCOL_ID,"stage_id":"Y3-R6","document_type":kind,"status":status,"payload":dict(payload)};value["canonical_digest"]=canonical(value);validate_document(value);return value
def write(path,kind,payload,status="PASS"):
    value=signed(kind,payload,status);path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream:stream.write(canonical_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())
    d=os.open(path.parent,os.O_RDONLY);os.fsync(d);os.close(d);return value
def publish_terminal(pending,path,value):
    validate_document(value);require(pending.is_file() and not path.exists(),"pending transition drift");tmp=path.with_name(path.name+".tmp");fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream:stream.write(canonical_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path);pending.unlink();d=os.open(path.parent,os.O_RDONLY);os.fsync(d);os.close(d)
def source_files(repo):return [file_record(repo/p,repo) for p in PRODUCER_FILES+DEPENDENCIES]
def verify_implementation(repo):
    seal=read(repo/ROOT/IMPLEMENTATION_SEAL);validate_document(seal);q=seal["payload"];commit=q["producer_commit"];require(y3.git(repo,"rev-parse",PRODUCER_REF)==commit,"producer ref drift");rows=source_files(repo);require(rows==q["source_files"] and canonical({"files":rows})==q["files_digest"],"producer source drift");top=Path(y3.git(repo,"rev-parse","--show-toplevel"))
    for row in rows:
        rel=(repo/row["path"]).relative_to(top);blob=subprocess.check_output(["git","show",f"{commit}:{rel}"],cwd=repo)
        if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            lines=blob.decode().splitlines();require(lines[1].endswith(row["sha256"]) and int(lines[2].split()[1])==row["bytes"],"producer LFS pointer drift")
        else:require(hashlib.sha256(blob).hexdigest()==row["sha256"],"producer Git blob drift: "+row["path"])
    return commit
def write_schemas(repo):
    values=((repo/SPEC_SCHEMA,{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":SPEC_SCHEMA.name,"const":read(repo/SPEC)}),(repo/ARTIFACT_SCHEMA,artifact_schema()))
    for path,value in values:
        require(not path.exists(),"schema exists",StartupError);path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(fd,"wb") as stream:stream.write(json.dumps(value,sort_keys=True,separators=(",",":")).encode()+b"\n");stream.flush();os.fsync(stream.fileno())
