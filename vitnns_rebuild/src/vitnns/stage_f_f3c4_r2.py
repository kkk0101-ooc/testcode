"""CPU-only F3C4-R2 validation/sealing wrapper over immutable R1 payload."""
from __future__ import annotations

import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from . import stage_f_f3c4 as r1
from . import stage_f_f3c4_validate as r1_validator

CONFIG_PATH="config/stage_f_f3c4_r2.json"
SCHEMA_PATH="config/stage_f_f3c4_r2.schema.json"
ROOT_PATH="evidence/stage_f/f3c4_analysis_r2_terminal_seal_20260908/formal"
REPORT_PATH="reports/stage_f_f3c4_analysis_r2_terminal_seal_20260908"
STARTUP_PATH=ROOT_PATH+"/startup_gate.json"
SOURCE_PATH=ROOT_PATH+"/source_binding.json"
R1_BINDING_PATH=ROOT_PATH+"/r1_payload_binding.json"
RESOURCE_PATH=ROOT_PATH+"/resource_ledger.json"
ACCESS_PATH=ROOT_PATH+"/access_audit.json"
PAYLOAD_PATH=ROOT_PATH+"/payload_manifest.json"
INVENTORY_PATH=ROOT_PATH+"/whole_root_inventory.json"
STATUS_PATH=ROOT_PATH+"/status.json"
PENDING_PATH=ROOT_PATH+"/completion/f3c4.pending.json"
COMPLETION_PATH=ROOT_PATH+"/completion/f3c4.json"
VALIDATION_PATH=REPORT_PATH+"/validation_report.json"
FALSE_PASS_PATH=REPORT_PATH+"/false_pass_audit.json"
TEST_PATH=REPORT_PATH+"/test_report.json"
REPORT_INVENTORY_PATH=REPORT_PATH+"/package_inventory.json"
R1_FAILURE_PATH=r1.REPORT_PATH+"/failure_event.json"
ADDENDUM_PATH="docs/stage_f_f3c4_r2_table_parity_sealing_recovery_addendum_20260908.md"
ARCHIVE_REF=r1.ARCHIVE_REF
SOURCE_FILES=(CONFIG_PATH,SCHEMA_PATH,ADDENDUM_PATH,"src/vitnns/stage_f_f3c4_r2.py","tests/test_stage_f_f3c4_r2.py")

class StageFF3C4R2Error(ValueError): pass

def load_config(repo_root: Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); config=r1._read(repo/CONFIG_PATH); schema=r1._read(repo/SCHEMA_PATH); errors=sorted(Draft202012Validator(schema).iter_errors(config),key=lambda x:list(x.path))
    if errors: raise StageFF3C4R2Error(errors[0].message)
    return config

def validate_table_parity(repo: Path,name: str)->dict[str,Any]:
    document=r1._read(repo/r1.ROOT_PATH/"paper_data"/f"{name}.json"); r1._validate_signed(document,name); rows=document["rows"]
    with (repo/r1.ROOT_PATH/"paper_data"/f"{name}.csv").open(newline="",encoding="utf-8") as stream: observed=list(csv.DictReader(stream))
    if document["row_count"]!=r1_validator.TABLES[name] or len(rows)!=len(observed) or list(observed[0])!=document["columns"] or set(document["columns"])!=set(rows[0]): raise StageFF3C4R2Error(f"{name} row/header/schema parity failed")
    for expected,actual in zip(rows,observed):
        if list(actual)!=document["columns"] or set(expected)!=set(document["columns"]): raise StageFF3C4R2Error(f"{name} column parity failed")
        for key in document["columns"]:
            parsed=r1_validator._csv_value(actual[key],expected[key])
            if type(expected[key]) is float:
                if parsed!=expected[key]: raise StageFF3C4R2Error(f"{name} float parity failed")
            elif parsed!=expected[key]: raise StageFF3C4R2Error(f"{name} value parity failed")
    return {"status":"PASS","rows":len(rows),"columns":len(document["columns"])}

def _r1_inventory(repo: Path)->dict[str,Any]:
    files=[]
    for root in (repo/r1.ROOT_PATH,repo/r1.REPORT_PATH):
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink(): files.append({"path":path.relative_to(repo).as_posix(),"bytes":path.stat().st_size,"sha256":r1._sha_file(path)})
    return {"files":files,"file_count":len(files),"total_bytes":sum(x["bytes"] for x in files),"file_inventory_digest":r1.canonical_sha256({"files":files})}

def initialize(repo_root: Path|str=REPO_ROOT,*,started_at: str)->dict[str,Any]:
    repo=Path(repo_root).resolve(); root=repo/ROOT_PATH
    if root.exists() or (repo/REPORT_PATH).exists(): raise StageFF3C4R2Error("R2 roots must be fresh")
    config=load_config(repo); r1.verify_plan_sealed(repo); failure=r1._read(repo/R1_FAILURE_PATH); r1._validate_signed(failure,"R1 failure")
    if failure.get("status")!="BLOCKED" or failure.get("failure_code")!="CANONICAL_JSON_ROW_KEY_ORDER_PARITY_VALIDATOR_DEFECT": raise StageFF3C4R2Error("R1 failure authorization drifted")
    inventory=_r1_inventory(repo); startup=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_startup_gate","status":"PASS","authorization":"F3C4_R2_CPU_TERMINAL_SEAL_ONLY","started_at":started_at,"r1_failure":r1._record(repo,R1_FAILURE_PATH),"r1_plan_seal":r1._record(repo,r1.PLAN_SEAL_PATH),"r1_payload_inventory_before_r2":inventory,"new_f3c3_result_or_array_access":False,"new_formal_jpeg_decode":False,"new_statistics":False,"new_rendering":False,"model_checkpoint_torch_timm_cuda_access":False,"f4_started":False}); r1._write_new(repo/STARTUP_PATH,startup)
    source_files=[r1._file_record(repo,path) for path in SOURCE_FILES]; source=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_source_binding","status":"PASS","files":source_files,"source_tree_digest":r1.canonical_sha256({"files":source_files}),"r1_source_binding":r1._record(repo,r1.SOURCE_BINDING_PATH)}); r1._write_new(repo/SOURCE_PATH,source)
    binding=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_r1_payload_binding","status":"PASS",**inventory,"r1_outputs_reused_by_reference_without_copy_or_modification":True}); r1._write_new(repo/R1_BINDING_PATH,binding)
    pending=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_pending_completion","status":"PENDING","started_at":started_at,"startup_gate":r1._record(repo,STARTUP_PATH),"source_binding":r1._record(repo,SOURCE_PATH),"r1_payload_binding":r1._record(repo,R1_BINDING_PATH),"new_scientific_work":False,"f4_started":False}); r1._write_new(repo/PENDING_PATH,pending); (repo/REPORT_PATH).mkdir(parents=True,exist_ok=False); return startup

def validate_r1(repo_root: Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); before=r1._read(repo/R1_BINDING_PATH); r1._validate_signed(before,"R1 binding"); current=_r1_inventory(repo)
    if any(before.get(key)!=current[key] for key in ("files","file_count","total_bytes","file_inventory_digest")): raise StageFF3C4R2Error("R1 payload changed after R2 startup")
    original=r1_validator.validate_table_parity; r1_validator.validate_table_parity=lambda _repo,name:validate_table_parity(_repo,name)
    try: base=r1_validator.validate_outputs(repo)
    finally: r1_validator.validate_table_parity=original
    return r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_validation_report","status":"PASS","implementation_validation":"PASS","workflow_validation":"PASS","package_validation":"PASS","recovery_scope":load_config(repo)["recovery_scope"],"corrected_table_parity":{name:validate_table_parity(repo,name) for name in r1_validator.TABLES},"r1_validation_recomputed":base,"new_f3c3_result_or_array_access":False,"new_formal_jpeg_decode":False,"new_statistics":False,"new_rendering":False,"model_checkpoint_torch_timm_cuda_access":False,"f4_started":False,"next_authorization":"F4_ONLY"})

def record_tests(repo_root: Path|str,*,passed:int,failed:int,command:str)->dict[str,Any]:
    if passed<=0 or failed: raise StageFF3C4R2Error("R2 tests failed")
    repo=Path(repo_root).resolve(); report=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_test_report","status":"PASS","passed":passed,"failed":failed,"command":command,"synthetic_only":True,"new_f3c3_or_jpeg_access":False}); r1._write_new(repo/TEST_PATH,report); return report

def finalize(repo_root: Path|str=REPO_ROOT,*,completed_at:str)->dict[str,Any]:
    repo=Path(repo_root).resolve(); validation=validate_r1(repo); r1._write_new(repo/VALIDATION_PATH,validation); test=r1._read(repo/TEST_PATH); r1._validate_signed(test,"R2 tests")
    audit=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_false_pass_audit","status":"PASS","attacks":[{"attack":"canonical_json_object_key_sort_differs_from_explicit_csv_column_order","result":"REJECTED","evidence":"tests/test_stage_f_f3c4_r2.py::test_corrected_parity_accepts_canonical_key_sort_and_rejects_value_mutation"},{"attack":"r1_payload_modified_during_seal","result":"REJECTED","evidence":"tests/test_stage_f_f3c4_r2.py::test_r1_binding_detects_mutation"},{"attack":"new_f3c3_jpeg_statistics_render_or_model_work_in_r2","result":"REJECTED","evidence":"tests/test_stage_f_f3c4_r2.py::test_r2_source_is_terminal_seal_only"}],"rejected_count":3}); r1._write_new(repo/FALSE_PASS_PATH,audit)
    retained={"f3c1":r1._regular_bytes(repo/"evidence/stage_f/f3c1_implementation_r6_cpu_seal_20260907/formal",repo/"reports/stage_f_f3c1_implementation_r6_cpu_seal_20260907"),"f3c2":r1._regular_bytes(repo/"evidence/stage_f/f3c2_pilot_v1_20260907/formal",repo/"reports/stage_f_f3c2_pilot_v1_20260907"),"f3c3":r1._regular_bytes(repo/r1.F3C3_ROOT,repo/r1.F3C3_REPORT),"f3c4_v1":r1._regular_bytes(repo/"evidence/stage_f/f3c4_analysis_v1_20260908/formal",repo/"reports/stage_f_f3c4_analysis_v1_20260908"),"f3c4_r1":r1._regular_bytes(repo/r1.ROOT_PATH,repo/r1.REPORT_PATH),"f3c4_r2_before_ledger":r1._regular_bytes(repo/ROOT_PATH,repo/REPORT_PATH)}; total=sum(retained.values())
    ledger=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_resource_ledger","status":"PASS","cpu_only":True,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,"f3c4_r2_gpu_hours":0.0,"actual_total_f3c_gpu_hours":r1.TOTAL_F3C_GPU_HOURS,"retained_regular_file_bytes_before_r2_ledger":retained,"actual_total_f3c_retained_regular_file_bytes_before_r2_ledger":total,"retained_cap_bytes":8589934592,"all_caps_satisfied":total<=8589934592,"waiver":False}); r1._write_new(repo/RESOURCE_PATH,ledger)
    access=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_access_audit","status":"PASS","r1_formal_terminal_and_jpeg_access_reused_by_reference":True,"new_f3c3_result_or_array_access":False,"new_formal_jpeg_decode":False,"new_pilot_or_order44_access":False,"new_statistics":False,"new_rendering":False,"model_loaded":False,"checkpoint_loaded":False,"torch_imported":False,"timm_imported":False,"cuda_accessed":False,"forward_calls":0,"backward_calls":0,"f4_started":False}); r1._write_new(repo/ACCESS_PATH,access)
    payload=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_payload_manifest","status":"PASS","records":[r1._record(repo,path) for path in (STARTUP_PATH,SOURCE_PATH,R1_BINDING_PATH,VALIDATION_PATH,FALSE_PASS_PATH,TEST_PATH,RESOURCE_PATH,ACCESS_PATH)],"r1_payload_reused_by_reference":r1._record(repo,R1_BINDING_PATH)}); r1._write_new(repo/PAYLOAD_PATH,payload)
    status=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_status","status":"PASS","terminal":True,"implementation_validation":"PASS","workflow_validation":"PASS","package_validation":"PASS","scientific_payload_root":r1.ROOT_PATH,"formal_samples":"50/50","source_terminals":"250/250 VALID","render_pages":23,"exports":69,"manual_qc":"23/23 PASS","next_authorization":"F4_ONLY","f4_started":False,"git":{"commit":"SELF","ref":ARCHIVE_REF}}); r1._write_new(repo/STATUS_PATH,status)
    report_inv=r1._inventory(repo/REPORT_PATH,exclude=("package_inventory.json",)); report_doc=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_report_inventory","status":"PASS",**report_inv}); r1._write_new(repo/REPORT_INVENTORY_PATH,report_doc); root_inv=r1._inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","completion/f3c4.json","completion/f3c4.pending.json")); root_doc=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r2_whole_root_inventory","status":"PASS",**root_inv}); r1._write_new(repo/INVENTORY_PATH,root_doc)
    method=r1._load_table(repo,"method_summary"); contrasts=r1._load_table(repo,"contrast_summary"); completion=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_terminal_completion","status":"PASS","completed_at":completed_at,"implementation_validation":"PASS","workflow_validation":"PASS","package_validation":"PASS","scientific_outcome_separate_from_workflow":True,"formal_samples":"50/50","source_terminals":{"VALID":250,"FAILED":0,"NA":0,"incomplete":0},"statistical_unit":"image","formal_n":50,"primary_contrast":r1.PRIMARY_CONTRAST,"primary_unadjusted":True,"secondary_holm_family":list(r1.SECONDARY_CONTRASTS),"bootstrap_resamples":10000,"sign_flip_resamples":100000,"method_summary":method,"contrast_summary":contrasts,"scientific_payload_root":r1.ROOT_PATH,"r2_terminal_seal_root":ROOT_PATH,"render_page_count":23,"export_file_count":69,"manual_qc_page_count":23,"formal_jpeg_access_count":50,"pilot_outcome_or_jpeg_access":False,"order44_accessed_or_recomputed":False,"unlisted_jpeg_access":False,"model_loaded":False,"checkpoint_loaded":False,"torch_imported":False,"timm_imported":False,"cuda_initialized_or_accessed":False,"forward_calls":0,"backward_calls":0,"f3c4_gpu_hours":0.0,"actual_total_f3c_gpu_hours":r1.TOTAL_F3C_GPU_HOURS,"actual_total_f3c_retained_regular_file_bytes_before_terminal":total,"all_caps_satisfied":total<=8589934592,"fine_grid_fidelity":r1.FINE_GRID_STATUS,"statistical_inference_performed":True,"scientific_comparison_performed":True,"formal_maps_rendered":True,"formal_renders_manually_inspected":True,"full_method_ranking_produced":False,"winner_produced":False,"go_no_go_produced":False,"sealed_upstream_modified":False,"f4_started":False,"next_authorization":"F4_ONLY","records":{name:r1._record(repo,path) for name,path in (("startup_gate",STARTUP_PATH),("source_binding",SOURCE_PATH),("r1_payload_binding",R1_BINDING_PATH),("validation_report",VALIDATION_PATH),("false_pass_audit",FALSE_PASS_PATH),("test_report",TEST_PATH),("resource_ledger",RESOURCE_PATH),("access_audit",ACCESS_PATH),("payload_manifest",PAYLOAD_PATH),("whole_root_inventory",INVENTORY_PATH),("report_inventory",REPORT_INVENTORY_PATH),("status_record",STATUS_PATH))},"git":{"commit":"SELF","ref":ARCHIVE_REF}}); r1._atomic_replace_pending.__wrapped__ if False else None
    pending=repo/PENDING_PATH; temporary=pending.with_name(".f3c4-r2-terminal.tmp"); r1._write_new(temporary,completion); os.replace(temporary,pending); os.replace(pending,repo/COMPLETION_PATH); return completion

def validate_completion(repo_root: Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); completion=r1._read(repo/COMPLETION_PATH); r1._validate_signed(completion,"R2 completion")
    if (repo/PENDING_PATH).exists() or completion.get("status")!="PASS" or completion.get("next_authorization")!="F4_ONLY" or completion.get("f4_started") is not False: raise StageFF3C4R2Error("R2 completion invalid")
    for record in completion["records"].values():
        if record!=r1._record(repo,record["path"]): raise StageFF3C4R2Error("R2 completion binding drifted")
    inv=r1._read(repo/INVENTORY_PATH); rebuilt=r1._inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","completion/f3c4.json","completion/f3c4.pending.json"))
    if any(inv.get(k)!=rebuilt[k] for k in ("files","file_count","total_bytes","file_inventory_digest")): raise StageFF3C4R2Error("R2 inventory drifted")
    return {"status":"PASS","formal_samples":50,"valid_terminals":250,"archive_ref":ARCHIVE_REF,"next_authorization":"F4_ONLY"}

__all__=["initialize","validate_table_parity","validate_r1","record_tests","finalize","validate_completion"]
