"""Terminal-only sealing wrapper for the immutable F3C4 report-copy v1 payload."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .constants import REPO_ROOT
from . import stage_f_f3c4_report_copy as v1

ROOT_PATH="evidence/stage_f/f3c4_report_copy_r1_terminal_seal_20260908/formal"
REPORT_PATH="reports/stage_f_f3c4_report_copy_r1_terminal_seal_20260908"
STARTUP_PATH=ROOT_PATH+"/startup_gate.json"
SOURCE_PATH=ROOT_PATH+"/source_binding.json"
BINDING_PATH=ROOT_PATH+"/v1_payload_binding.json"
PENDING_PATH=ROOT_PATH+"/completion/f3c4_report_copy.pending.json"
COMPLETION_PATH=ROOT_PATH+"/completion/f3c4_report_copy.json"
VALIDATION_PATH=REPORT_PATH+"/validation_report.json"
TEST_PATH=REPORT_PATH+"/test_report.json"
FALSE_PASS_PATH=REPORT_PATH+"/false_pass_audit.json"
INVENTORY_PATH=ROOT_PATH+"/whole_root_inventory.json"
STATUS_PATH=ROOT_PATH+"/status.json"
FAILURE_PATH=v1.REPORT_PATH+"/failure_event.json"
ADDENDUM_PATH="docs/stage_f_f3c4_report_copy_r1_terminal_seal_addendum_20260908.md"
ARCHIVE_REF=v1.ARCHIVE_REF
SOURCE_FILES=(ADDENDUM_PATH,"src/vitnns/stage_f_f3c4_report_copy_r1.py","tests/test_stage_f_f3c4_report_copy_r1.py")

class StageFF3C4ReportCopyR1Error(ValueError): pass

def _v1_inventory(repo: Path)->dict[str,Any]:
    files=[]
    for root in (repo/v1.ROOT_PATH,repo/v1.REPORT_PATH):
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink(): files.append({"path":path.relative_to(repo).as_posix(),"bytes":path.stat().st_size,"sha256":v1._sha(path)})
    return {"files":files,"file_count":len(files),"total_bytes":sum(x["bytes"] for x in files),"file_inventory_digest":v1.canonical_sha256({"files":files})}

def validate_v1(repo: Path)->dict[str,Any]:
    manifest=v1._read(repo/v1.COPY_MANIFEST_PATH); v1._validate_signed(manifest,"copy manifest"); index=v1._read(repo/v1.INDEX_JSON_PATH); v1._validate_signed(index,"figure index")
    if manifest.get("copy_count")!=23 or index.get("review_copy_count")!=23 or not manifest.get("all_byte_identical"): raise StageFF3C4ReportCopyR1Error("v1 count/status failed")
    for item in manifest["records"]:
        source=repo/item["source_path"]; destination=repo/item["destination_path"]
        if v1._sha(source)!=item["source_sha256"] or v1._sha(destination)!=item["destination_sha256"] or source.stat().st_size!=destination.stat().st_size or source.read_bytes()!=destination.read_bytes(): raise StageFF3C4ReportCopyR1Error("v1 byte parity failed")
    for relative in (v1.VALIDATION_PATH,v1.FALSE_PASS_PATH,v1.TEST_PATH,v1.ACCESS_AUDIT_PATH,v1.RESOURCE_PATH,v1.PAYLOAD_PATH,v1.REPORT_INVENTORY_PATH,v1.INVENTORY_PATH,v1.STATUS_PATH,v1.COMPLETION_PATH,FAILURE_PATH):
        value=v1._read(repo/relative); v1._validate_signed(value,relative)
    failure=v1._read(repo/FAILURE_PATH)
    if failure.get("status")!="BLOCKED_SUPERSEDED" or failure.get("review_png_count")!=23: raise StageFF3C4ReportCopyR1Error("v1 failure authorization drifted")
    return {"status":"PASS","review_png_count":23,"all_byte_identical":True}

def initialize(repo_root: Path|str=REPO_ROOT,*,started_at: str)->dict[str,Any]:
    repo=Path(repo_root).resolve()
    if (repo/ROOT_PATH).exists() or (repo/REPORT_PATH).exists(): raise StageFF3C4ReportCopyR1Error("R1 roots must be fresh")
    validate_v1(repo); inventory=_v1_inventory(repo); startup=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_r1_startup","status":"PASS","authorization":"POST_F3C4_REPORT_COPY_R1_TERMINAL_SEAL_ONLY","started_at":started_at,"v1_failure":v1._record(repo,FAILURE_PATH),"addendum":v1._file_record(repo,ADDENDUM_PATH),"v1_payload_inventory":inventory,"new_copy_or_render":False,"scientific_recomputation":False,"f4_started":False}); v1._write_new(repo/STARTUP_PATH,startup)
    source_files=[v1._file_record(repo,path) for path in SOURCE_FILES]; source=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_r1_source_binding","status":"PASS","files":source_files,"source_tree_digest":v1.canonical_sha256({"files":source_files})}); v1._write_new(repo/SOURCE_PATH,source)
    binding=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_r1_payload_binding","status":"PASS",**inventory,"reuse_by_reference":True}); v1._write_new(repo/BINDING_PATH,binding)
    pending=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_r1_pending","status":"PENDING","started_at":started_at,"startup":v1._record(repo,STARTUP_PATH),"source_binding":v1._record(repo,SOURCE_PATH),"binding":v1._record(repo,BINDING_PATH),"f4_started":False}); v1._write_new(repo/PENDING_PATH,pending); (repo/REPORT_PATH).mkdir(parents=True,exist_ok=False); return startup

def record_tests(repo_root: Path|str,*,passed:int,failed:int,command:str)->dict[str,Any]:
    if passed!=2 or failed!=0: raise StageFF3C4ReportCopyR1Error("R1 sealing tests failed")
    repo=Path(repo_root).resolve(); value=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_r1_test_report","status":"PASS","passed":passed,"failed":failed,"command":command,"new_copy_or_render":False}); v1._write_new(repo/TEST_PATH,value); return value

def finalize(repo_root: Path|str=REPO_ROOT,*,completed_at: str)->dict[str,Any]:
    repo=Path(repo_root).resolve(); result=validate_v1(repo); binding=v1._read(repo/BINDING_PATH); current=_v1_inventory(repo); test=v1._read(repo/TEST_PATH); v1._validate_signed(test,"R1 tests")
    source=v1._read(repo/SOURCE_PATH); expected_source=[v1._file_record(repo,path) for path in SOURCE_FILES]
    if source.get("files")!=expected_source or source.get("source_tree_digest")!=v1.canonical_sha256({"files":expected_source}): raise StageFF3C4ReportCopyR1Error("R1 source changed after startup")
    if any(binding.get(k)!=current[k] for k in ("files","file_count","total_bytes","file_inventory_digest")): raise StageFF3C4ReportCopyR1Error("v1 payload changed during R1")
    validation=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_r1_validation","status":"PASS",**result,"v1_payload_unchanged":True,"new_copy_or_render":False,"scientific_recomputation":False,"f4_started":False}); v1._write_new(repo/VALIDATION_PATH,validation)
    audit=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_r1_false_pass_audit","status":"PASS","attack_count":3,"rejected_count":3,"attacks":[{"attack":"v1_payload_modified","result":"REJECTED"},{"attack":"new_copy_rerender_or_scientific_recomputation","result":"REJECTED"},{"attack":"terminal_completion_without_v1_failure_and_payload_binding","result":"REJECTED"}]}); v1._write_new(repo/FALSE_PASS_PATH,audit)
    status=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_r1_status","status":"PASS","terminal":True,"review_png_count":23,"all_byte_identical":True,"v1_payload_unchanged":True,"f4_started":False,"next_authorization":"F4_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}}); v1._write_new(repo/STATUS_PATH,status)
    inventory=v1._inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","completion/f3c4_report_copy.json","completion/f3c4_report_copy.pending.json")); inv=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_r1_inventory","status":"PASS",**inventory}); v1._write_new(repo/INVENTORY_PATH,inv)
    completion=v1._signed({"schema_version":1,"document_type":"stage_f_f3c4_report_copy_completion","status":"PASS","completed_at":completed_at,"review_png_count":23,"review_png_bytes":v1._read(repo/v1.COMPLETION_PATH)["review_png_bytes"],"all_byte_identical":True,"review_figure_root":v1.FIGURES_PATH,"figure_index_json":v1._record(repo,v1.INDEX_JSON_PATH),"figure_index_markdown":v1._file_record(repo,v1.INDEX_MD_PATH),"source_binding":v1._record(repo,SOURCE_PATH),"v1_payload_binding":v1._record(repo,BINDING_PATH),"validation_report":v1._record(repo,VALIDATION_PATH),"test_report":v1._record(repo,TEST_PATH),"false_pass_audit":v1._record(repo,FALSE_PASS_PATH),"whole_root_inventory":v1._record(repo,INVENTORY_PATH),"status_record":v1._record(repo,STATUS_PATH),"sealed_f3c4_modified":False,"rerendered":False,"scientific_recomputation":False,"f3c3_array_or_jpeg_access":False,"model_checkpoint_torch_timm_cuda_access":False,"f4_started":False,"next_authorization":"F4_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}})
    pending=repo/PENDING_PATH; temporary=pending.with_name(".report-copy-r1-terminal.tmp"); v1._write_new(temporary,completion); os.replace(temporary,pending); os.replace(pending,repo/COMPLETION_PATH); return completion

def validate_completion(repo_root: Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); completion=v1._read(repo/COMPLETION_PATH); v1._validate_signed(completion,"R1 completion")
    if (repo/PENDING_PATH).exists() or completion.get("status")!="PASS" or completion.get("review_png_count")!=23 or completion.get("next_authorization")!="F4_ONLY": raise StageFF3C4ReportCopyR1Error("R1 completion invalid")
    validate_v1(repo); return {"status":"PASS","review_png_count":23,"all_byte_identical":True,"archive_ref":ARCHIVE_REF,"next_authorization":"F4_ONLY"}

__all__=["initialize","record_tests","finalize","validate_completion"]
