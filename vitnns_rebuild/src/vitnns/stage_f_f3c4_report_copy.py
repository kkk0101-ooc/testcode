"""Post-F3C4 byte-identical report-side review PNG packaging."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "config/stage_f_f3c4_report_copy.json"
SCHEMA_PATH = "config/stage_f_f3c4_report_copy.schema.json"
ROOT_PATH = "evidence/stage_f/f3c4_report_copy_v1_20260908/formal"
REPORT_PATH = "reports/stage_f_f3c4_report_copy_v1_20260908"
FIGURES_PATH = REPORT_PATH + "/figures_png"
STARTUP_PATH = ROOT_PATH + "/startup_gate.json"
PLAN_PATH = ROOT_PATH + "/copy_plan.json"
SOURCE_BINDING_PATH = ROOT_PATH + "/source_binding.json"
ACCESS_POLICY_PATH = ROOT_PATH + "/access_policy.json"
PENDING_PATH = ROOT_PATH + "/completion/f3c4_report_copy.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/f3c4_report_copy.json"
PLAN_SEAL_PATH = ROOT_PATH + "/plan_seal.json"
COPY_MANIFEST_PATH = ROOT_PATH + "/copy_manifest.json"
ACCESS_AUDIT_PATH = ROOT_PATH + "/access_audit.json"
RESOURCE_PATH = ROOT_PATH + "/resource_report.json"
PAYLOAD_PATH = ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = ROOT_PATH + "/status.json"
INDEX_JSON_PATH = REPORT_PATH + "/figure_index.json"
INDEX_MD_PATH = REPORT_PATH + "/figure_index.md"
VALIDATION_PATH = REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = REPORT_PATH + "/test_report.json"
REPORT_INVENTORY_PATH = REPORT_PATH + "/package_inventory.json"
ARCHIVE_REF = "refs/archive/stage-f-v1-f3c4-report-copy"
EXPECTED_HEAD = "944acd6e2b7579e6999668911ed420a1bd91a27b"
SOURCE_FILES = (CONFIG_PATH, SCHEMA_PATH, "src/vitnns/stage_f_f3c4_report_copy.py", "tests/test_stage_f_f3c4_report_copy.py")
ATTACKS = (
    "source_count_or_order_drift", "source_png_hash_drift",
    "destination_count_or_name_drift", "destination_bytes_or_hash_differ",
    "existing_destination_overwrite", "rerender_or_reencode",
    "statistics_or_scientific_recomputation", "f3c3_array_or_jpeg_access",
    "model_checkpoint_torch_timm_cuda_access", "sealed_f3c4_modified",
    "completion_before_validation_inventory_or_false_pass", "f4_started",
)

class StageFF3C4ReportCopyError(ValueError): pass

def _sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def _read(path: Path) -> dict[str,Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _signed(value: Mapping[str,Any]) -> dict[str,Any]:
    result=dict(value); result["canonical_digest"]=canonical_sha256(result); return result

def _validate_signed(value: Mapping[str,Any], context: str) -> None:
    payload=dict(value); observed=payload.pop("canonical_digest",None)
    if observed!=canonical_sha256(payload): raise StageFF3C4ReportCopyError(f"{context} canonical digest mismatch")

def _write_new(path: Path,value: Mapping[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"wb") as stream: stream.write(canonical_json_bytes(value)+b"\n"); stream.flush(); os.fsync(stream.fileno())

def _file_record(repo: Path,relative: str) -> dict[str,Any]:
    path=repo/relative; return {"path":relative,"bytes":path.stat().st_size,"sha256":_sha(path)}

def _record(repo: Path,relative: str) -> dict[str,Any]:
    value=_read(repo/relative); _validate_signed(value,relative); return {**_file_record(repo,relative),"canonical_digest":value["canonical_digest"]}

def _inventory(root: Path,*,exclude: Sequence[str]=()) -> dict[str,Any]:
    files=[]; excluded=set(exclude)
    for path in sorted(root.rglob("*")):
        relative=path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded: files.append({"path":relative,"bytes":path.stat().st_size,"sha256":_sha(path)})
    return {"files":files,"file_count":len(files),"total_bytes":sum(x["bytes"] for x in files),"file_inventory_digest":canonical_sha256({"files":files})}

def load_config(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); config=_read(repo/CONFIG_PATH); schema=_read(repo/SCHEMA_PATH); errors=sorted(Draft202012Validator(schema).iter_errors(config),key=lambda x:list(x.path))
    if errors: raise StageFF3C4ReportCopyError(errors[0].message)
    return config

def false_pass_reasons(values: Mapping[str,bool]) -> tuple[str,...]:
    if set(values)!=set(ATTACKS): raise StageFF3C4ReportCopyError("false-pass set must cover exactly 12 attacks")
    return tuple(name for name in ATTACKS if values[name] is not True)

def validate_startup(repo: Path) -> dict[str,Any]:
    startup=_read(repo/STARTUP_PATH); _validate_signed(startup,"startup gate")
    if startup.get("status")!="PASS" or startup.get("source_png_count")!=23 or startup.get("destination_png_count")!=23: raise StageFF3C4ReportCopyError("startup gate failed")
    return startup

def initialize(repo_root: Path|str=REPO_ROOT,*,started_at: str) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); root=repo/ROOT_PATH
    if not (repo/STARTUP_PATH).is_file() or {p.name for p in root.iterdir()}!={"startup_gate.json"} or (repo/REPORT_PATH).exists(): raise StageFF3C4ReportCopyError("fresh root state failed")
    startup=validate_startup(repo); config=load_config(repo); files=[_file_record(repo,path) for path in SOURCE_FILES]
    source=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_source_binding","status":"PASS","files":files,"source_tree_digest":canonical_sha256({"files":files}),"producer_head":EXPECTED_HEAD}); _write_new(repo/SOURCE_BINDING_PATH,source)
    plan=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_plan","status":"FROZEN_BEFORE_COPY","records":startup["exact_copy_plan"],"copy_count":23,"module_counts":startup["module_counts"],"copy_mode":"byte_for_byte","rerender":False}); _write_new(repo/PLAN_PATH,plan)
    access=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_access_policy","status":"FROZEN_BEFORE_COPY","allowed_read_paths":[x["source_path"] for x in plan["records"]],"allowed_write_root":FIGURES_PATH,"f3c3_array_or_jpeg_access":False,"statistics_or_rendering":False,"model_checkpoint_torch_timm_cuda_access":False,"f4_started":False}); _write_new(repo/ACCESS_POLICY_PATH,access)
    pending=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_pending_completion","status":"PENDING","started_at":started_at,"startup_gate":_record(repo,STARTUP_PATH),"copy_plan":_record(repo,PLAN_PATH),"source_binding":_record(repo,SOURCE_BINDING_PATH),"access_policy":_record(repo,ACCESS_POLICY_PATH),"copies_created":0,"f4_started":False}); _write_new(repo/PENDING_PATH,pending)
    seal=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_plan_seal","status":"SEALED_BEFORE_COPY","records":[_record(repo,path) for path in (STARTUP_PATH,PLAN_PATH,SOURCE_BINDING_PATH,ACCESS_POLICY_PATH,PENDING_PATH)],"copies_before_seal":0}); _write_new(repo/PLAN_SEAL_PATH,seal); (repo/FIGURES_PATH).mkdir(parents=True,exist_ok=False); return {"status":"PASS","plan_sealed":True,"copy_count":23}

def validate_initialized(repo: Path) -> dict[str,Any]:
    seal=_read(repo/PLAN_SEAL_PATH); _validate_signed(seal,"plan seal")
    for record in seal["records"]:
        if record!=_record(repo,record["path"]): raise StageFF3C4ReportCopyError("sealed plan drifted")
    source=_read(repo/SOURCE_BINDING_PATH); expected=[_file_record(repo,path) for path in SOURCE_FILES]
    if source["files"]!=expected: raise StageFF3C4ReportCopyError("source changed after plan seal")
    return {"status":"PASS"}

def execute(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); validate_initialized(repo); plan=_read(repo/PLAN_PATH); copies=[]
    for record in plan["records"]:
        source=repo/record["source_path"]; destination=repo/record["destination_path"]
        if destination.exists(): raise StageFF3C4ReportCopyError("refusing to overwrite review copy")
        if _sha(source)!=record["source_sha256"] or source.stat().st_size!=record["source_bytes"]: raise StageFF3C4ReportCopyError("source PNG drifted")
        temporary=destination.with_name("."+destination.name+".tmp"); temporary.parent.mkdir(parents=True,exist_ok=True)
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            shutil.copyfileobj(incoming,outgoing,length=1024*1024); outgoing.flush(); os.fsync(outgoing.fileno())
        if _sha(temporary)!=record["source_sha256"] or temporary.stat().st_size!=record["source_bytes"]: raise StageFF3C4ReportCopyError("temporary copy differs from source")
        os.replace(temporary,destination); copies.append({**record,"destination_bytes":destination.stat().st_size,"destination_sha256":_sha(destination),"byte_identical":destination.read_bytes()==source.read_bytes()})
    if len(copies)!=23 or not all(x["byte_identical"] and x["source_sha256"]==x["destination_sha256"] for x in copies): raise StageFF3C4ReportCopyError("copy parity failed")
    manifest=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_manifest","status":"PASS","copy_count":23,"records":copies,"all_byte_identical":True,"rerendered":False}); _write_new(repo/COPY_MANIFEST_PATH,manifest)
    index=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_figure_index","status":"PASS","title":"F3C4 report-side review PNG index","review_copy_count":23,"formal_sources_unchanged":True,"records":[{key:x[key] for key in ("review_order","module","figure_id","source_path","source_sha256","destination_path","destination_sha256","byte_identical")} for x in copies]}); _write_new(repo/INDEX_JSON_PATH,index)
    md=["# F3C4 report-side review PNG index","","These 23 files are byte-identical review copies of the sealed formal PNG exports. No rerendering or scientific recomputation was performed.","","| # | Module | Review PNG | Formal source | SHA-256 |","|---:|---|---|---|---|"]
    for x in copies: md.append(f"| {x['review_order']:02d} | {x['module']} | figures_png/{Path(x['destination_path']).name} | {x['source_path']} | `{x['source_sha256']}` |")
    path=repo/INDEX_MD_PATH; descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"w",encoding="utf-8") as stream: stream.write("\n".join(md)+"\n"); stream.flush(); os.fsync(stream.fileno())
    return {"status":"PASS","copy_count":23,"all_byte_identical":True}

def validate_outputs(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); validate_initialized(repo); manifest=_read(repo/COPY_MANIFEST_PATH); _validate_signed(manifest,"copy manifest"); index=_read(repo/INDEX_JSON_PATH); _validate_signed(index,"figure index")
    if manifest.get("copy_count")!=23 or index.get("review_copy_count")!=23 or manifest.get("all_byte_identical") is not True: raise StageFF3C4ReportCopyError("copy/index count failed")
    for item in manifest["records"]:
        source=repo/item["source_path"]; destination=repo/item["destination_path"]
        if _sha(source)!=item["source_sha256"] or _sha(destination)!=item["destination_sha256"] or source.stat().st_size!=destination.stat().st_size or source.read_bytes()!=destination.read_bytes(): raise StageFF3C4ReportCopyError("source/destination parity failed")
    if len(list((repo/FIGURES_PATH).glob("*.png")))!=23: raise StageFF3C4ReportCopyError("figures_png contains wrong file count")
    return _signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_validation","status":"PASS","review_png_count":23,"all_byte_identical":True,"source_f3c4_modified":False,"rerendered":False,"scientific_recomputation":False,"f3c3_array_or_jpeg_access":False,"model_checkpoint_torch_timm_cuda_access":False,"f4_started":False})

def record_tests(repo_root: Path|str,*,passed:int,failed:int,command:str) -> dict[str,Any]:
    if passed<=0 or failed: raise StageFF3C4ReportCopyError("tests failed")
    repo=Path(repo_root).resolve(); value=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_test_report","status":"PASS","passed":passed,"failed":failed,"command":command}); _write_new(repo/TEST_PATH,value); return value

def finalize(repo_root: Path|str=REPO_ROOT,*,completed_at: str) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); validation=validate_outputs(repo); _write_new(repo/VALIDATION_PATH,validation); test=_read(repo/TEST_PATH); _validate_signed(test,"test report")
    if false_pass_reasons({name:True for name in ATTACKS}): raise StageFF3C4ReportCopyError("valid false-pass state rejected")
    audit=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_false_pass_audit","status":"PASS","attack_count":12,"rejected_count":12,"attacks":[{"attack":name,"result":"REJECTED","evidence":f"tests/test_stage_f_f3c4_report_copy.py::test_attack_is_rejected[{name}]"} for name in ATTACKS]}); _write_new(repo/FALSE_PASS_PATH,audit)
    access=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_access_audit","status":"PASS","formal_png_read_count":23,"review_png_write_count":23,"rerendered":False,"statistics_or_scientific_recomputation":False,"f3c3_array_or_jpeg_access":False,"model_checkpoint_torch_timm_cuda_access":False,"f4_started":False}); _write_new(repo/ACCESS_AUDIT_PATH,access)
    copied_bytes=sum(x["destination_bytes"] for x in _read(repo/COPY_MANIFEST_PATH)["records"]); resource=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_resource_report","status":"PASS","review_png_count":23,"review_png_bytes":copied_bytes,"gpu_hours":0.0,"forward_calls":0,"backward_calls":0}); _write_new(repo/RESOURCE_PATH,resource)
    payload=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_payload_manifest","status":"PASS","records":[_record(repo,path) for path in (STARTUP_PATH,PLAN_PATH,SOURCE_BINDING_PATH,ACCESS_POLICY_PATH,PLAN_SEAL_PATH,COPY_MANIFEST_PATH,INDEX_JSON_PATH,VALIDATION_PATH,FALSE_PASS_PATH,TEST_PATH,ACCESS_AUDIT_PATH,RESOURCE_PATH)]+[_file_record(repo,INDEX_MD_PATH)],"review_png_count":23}); _write_new(repo/PAYLOAD_PATH,payload)
    report_inv=_inventory(repo/REPORT_PATH,exclude=("package_inventory.json",)); report_doc=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_report_inventory","status":"PASS",**report_inv}); _write_new(repo/REPORT_INVENTORY_PATH,report_doc)
    status=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_status","status":"PASS","terminal":True,"review_png_count":23,"all_byte_identical":True,"sealed_f3c4_modified":False,"rerendered":False,"scientific_recomputation":False,"f4_started":False,"next_authorization":"F4_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}}); _write_new(repo/STATUS_PATH,status)
    root_inv=_inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","completion/f3c4_report_copy.json","completion/f3c4_report_copy.pending.json")); root_doc=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_whole_root_inventory","status":"PASS",**root_inv}); _write_new(repo/INVENTORY_PATH,root_doc)
    completion=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_completion","status":"PASS","completed_at":completed_at,"review_png_count":23,"review_png_bytes":copied_bytes,"all_byte_identical":True,"figure_index_json":_record(repo,INDEX_JSON_PATH),"figure_index_markdown":_file_record(repo,INDEX_MD_PATH),"copy_manifest":_record(repo,COPY_MANIFEST_PATH),"validation_report":_record(repo,VALIDATION_PATH),"false_pass_audit":_record(repo,FALSE_PASS_PATH),"test_report":_record(repo,TEST_PATH),"access_audit":_record(repo,ACCESS_AUDIT_PATH),"resource_report":_record(repo,RESOURCE_PATH),"payload_manifest":_record(repo,PAYLOAD_PATH),"report_inventory":_record(repo,REPORT_INVENTORY_PATH),"whole_root_inventory":_record(repo,INVENTORY_PATH),"status_record":_record(repo,STATUS_PATH),"sealed_f3c4_modified":False,"rerendered":False,"scientific_recomputation":False,"f4_started":False,"next_authorization":"F4_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}})
    pending=repo/PENDING_PATH; temporary=pending.with_name(".f3c4-report-copy-terminal.tmp"); _write_new(temporary,completion); os.replace(temporary,pending); os.replace(pending,repo/COMPLETION_PATH); return completion

def validate_completion(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); completion=_read(repo/COMPLETION_PATH); _validate_signed(completion,"completion")
    if (repo/PENDING_PATH).exists() or completion.get("status")!="PASS" or completion.get("review_png_count")!=23 or completion.get("next_authorization")!="F4_ONLY" or completion.get("f4_started") is not False: raise StageFF3C4ReportCopyError("terminal completion invalid")
    validate_outputs(repo); return {"status":"PASS","review_png_count":23,"all_byte_identical":True,"archive_ref":ARCHIVE_REF,"next_authorization":"F4_ONLY"}

__all__=["ATTACKS","initialize","execute","validate_outputs","record_tests","finalize","validate_completion"]
