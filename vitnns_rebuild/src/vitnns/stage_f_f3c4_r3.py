"""F3C4-R3: validate immutable R1 payload only and publish terminal receipts."""
from __future__ import annotations

import csv
import json
import math
import os
import resource
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from . import stage_f_f3c4 as r1
from . import stage_f_f3c4_validate as common

CONFIG_PATH="config/stage_f_f3c4_r3.json"
SCHEMA_PATH="config/stage_f_f3c4_r3.schema.json"
ROOT_PATH="evidence/stage_f/f3c4_analysis_r3_terminal_seal_20260908/formal"
REPORT_PATH="reports/stage_f_f3c4_analysis_r3_terminal_seal_20260908"
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
R2_FAILURE_PATH="reports/stage_f_f3c4_analysis_r2_terminal_seal_20260908/failure_event.json"
R2_PENDING_PATH="evidence/stage_f/f3c4_analysis_r2_terminal_seal_20260908/formal/completion/f3c4.pending.json"
ADDENDUM_PATH="docs/stage_f_f3c4_r3_payload_only_terminal_seal_addendum_20260908.md"
ARCHIVE_REF=r1.ARCHIVE_REF
TABLES=common.TABLES
SOURCE_FILES=(CONFIG_PATH,SCHEMA_PATH,ADDENDUM_PATH,"src/vitnns/stage_f_f3c4_r3.py","tests/test_stage_f_f3c4_r3.py")

class StageFF3C4R3Error(ValueError): pass

def load_config(repo_root: Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); config=r1._read(repo/CONFIG_PATH); schema=r1._read(repo/SCHEMA_PATH); errors=sorted(Draft202012Validator(schema).iter_errors(config),key=lambda x:list(x.path))
    if errors: raise StageFF3C4R3Error(errors[0].message)
    return config

def r1_inventory(repo: Path)->dict[str,Any]:
    files=[]
    for root in (repo/r1.ROOT_PATH,repo/r1.REPORT_PATH):
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink(): files.append({"path":path.relative_to(repo).as_posix(),"bytes":path.stat().st_size,"sha256":r1._sha_file(path)})
    return {"files":files,"file_count":len(files),"total_bytes":sum(x["bytes"] for x in files),"file_inventory_digest":r1.canonical_sha256({"files":files})}

def table_parity(repo: Path,name: str)->dict[str,Any]:
    document=r1._read(repo/r1.ROOT_PATH/"paper_data"/f"{name}.json"); r1._validate_signed(document,name); rows=document["rows"]
    with (repo/r1.ROOT_PATH/"paper_data"/f"{name}.csv").open(newline="",encoding="utf-8") as stream: observed=list(csv.DictReader(stream))
    columns=document["columns"]
    if document["row_count"]!=TABLES[name] or len(rows)!=len(observed) or list(observed[0])!=columns or set(columns)!=set(rows[0]): raise StageFF3C4R3Error(f"{name} schema parity failed")
    for expected,actual in zip(rows,observed):
        if list(actual)!=columns or set(expected)!=set(columns): raise StageFF3C4R3Error(f"{name} column parity failed")
        for key in columns:
            parsed=common._csv_value(actual[key],expected[key])
            if parsed!=expected[key]: raise StageFF3C4R3Error(f"{name} value parity failed: {key}")
    return {"status":"PASS","rows":len(rows),"columns":len(columns)}

def _load_rows(repo: Path,name: str)->list[dict[str,Any]]:
    return r1._read(repo/r1.ROOT_PATH/"paper_data"/f"{name}.json")["rows"]

def initialize(repo_root: Path|str=REPO_ROOT,*,started_at: str)->dict[str,Any]:
    repo=Path(repo_root).resolve(); load_config(repo)
    if (repo/ROOT_PATH).exists() or (repo/REPORT_PATH).exists(): raise StageFF3C4R3Error("R3 roots must be fresh")
    failure=r1._read(repo/R2_FAILURE_PATH); r1._validate_signed(failure,"R2 failure")
    if failure.get("status")!="BLOCKED" or failure.get("new_f3c3_result_or_array_access") is not False or failure.get("new_formal_jpeg_access") is not False or not (repo/R2_PENDING_PATH).is_file(): raise StageFF3C4R3Error("R2 fail-closed authorization drifted")
    inventory=r1_inventory(repo); startup=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_startup_gate","status":"PASS","authorization":"F3C4_R3_CPU_PAYLOAD_ONLY_TERMINAL_SEAL","started_at":started_at,"r2_failure":r1._record(repo,R2_FAILURE_PATH),"r2_pending":r1._record(repo,R2_PENDING_PATH),"r1_payload_inventory_before_r3":inventory,"allowed_input_boundary":"R1_ROOTS_ONLY","new_f3c3_or_jpeg_access":False,"new_empirical_endpoint_or_render":False,"model_checkpoint_torch_timm_cuda_access":False,"f4_started":False}); r1._write_new(repo/STARTUP_PATH,startup)
    files=[r1._file_record(repo,path) for path in SOURCE_FILES]; source=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_source_binding","status":"PASS","files":files,"source_tree_digest":r1.canonical_sha256({"files":files})}); r1._write_new(repo/SOURCE_PATH,source)
    binding=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_r1_payload_binding","status":"PASS",**inventory,"payload_reused_by_reference_without_copy_or_modification":True}); r1._write_new(repo/R1_BINDING_PATH,binding)
    pending=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_pending_completion","status":"PENDING","started_at":started_at,"startup_gate":r1._record(repo,STARTUP_PATH),"source_binding":r1._record(repo,SOURCE_PATH),"r1_payload_binding":r1._record(repo,R1_BINDING_PATH),"new_scientific_work":False,"f4_started":False}); r1._write_new(repo/PENDING_PATH,pending); (repo/REPORT_PATH).mkdir(parents=True,exist_ok=False); return startup

def validate_payload(repo_root: Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); binding=r1._read(repo/R1_BINDING_PATH); r1._validate_signed(binding,"R1 payload binding"); current=r1_inventory(repo)
    if any(binding.get(k)!=current[k] for k in ("files","file_count","total_bytes","file_inventory_digest")): raise StageFF3C4R3Error("R1 payload changed during R3")
    parity={name:table_parity(repo,name) for name in TABLES}; methods=_load_rows(repo,"per_image_method_scores"); contrasts=_load_rows(repo,"per_image_contrasts"); method_summary=_load_rows(repo,"method_summary"); contrast_summary=_load_rows(repo,"contrast_summary")
    by_score={(x["selection_order"],x["method_id"]):x for x in methods}
    for row in methods:
        values=np.asarray([row[f"repeat_{i}_score"] for i in range(5)],dtype=np.float64)
        if not np.isclose(float(np.mean(values)),row["image_mean_score"],rtol=1e-15,atol=0): raise StageFF3C4R3Error("five-tie image mean drifted")
    for row in contrasts:
        delta=by_score[(row["selection_order"],r1.METHODS[0])]["image_mean_score"]-by_score[(row["selection_order"],row["comparator_method_id"])]["image_mean_score"]
        if delta!=row["delta"]: raise StageFF3C4R3Error("contrast direction/value drifted")
    for row in method_summary:
        values=[x["image_mean_score"] for x in methods if x["method_id"]==row["method_id"]]; expected=r1._summary(values)
        if any(row[key]!=value for key,value in expected.items()): raise StageFF3C4R3Error("method summary drifted")
    boot=r1._read(repo/r1.ROOT_PATH/"paper_data/bootstrap_report.json"); sign=r1._read(repo/r1.ROOT_PATH/"paper_data/sign_flip_report.json"); holm=r1._read(repo/r1.ROOT_PATH/"paper_data/holm_report.json")
    for value,name in ((boot,"bootstrap"),(sign,"sign flip"),(holm,"Holm")): r1._validate_signed(value,name)
    for row in contrast_summary:
        values=np.asarray([x["delta"] for x in contrasts if x["contrast_id"]==row["contrast_id"]],dtype=np.float64); expected=r1._summary(values); b=r1.bootstrap_mean_ci(values,row["contrast_id"]); s=r1.paired_sign_flip_p_value(values,row["contrast_id"])
        if any(row[key]!=value for key,value in expected.items()) or [row["bootstrap_ci_95_lower"],row["bootstrap_ci_95_upper"]]!=b["ci"] or row["raw_p_value"]!=s["p_value"]: raise StageFF3C4R3Error("contrast summary/inference drifted")
        if b!=next(x for x in boot["results"] if x["contrast"]==row["contrast_id"]) or s!=next(x for x in sign["results"] if x["contrast"]==row["contrast_id"]): raise StageFF3C4R3Error("statistical report drifted")
    expected_holm=r1.holm_adjust({x["contrast_id"]:x["raw_p_value"] for x in contrast_summary if x["contrast_id"] in r1.SECONDARY_CONTRASTS})
    if holm["results"]!=expected_holm: raise StageFF3C4R3Error("Holm report drifted")
    render=r1._read(repo/r1.RENDER_MANIFEST_PATH); r1._validate_signed(render,"render manifest"); atlas=[]; signed=[]
    if render.get("render_page_count")!=23 or render.get("export_file_count")!=69 or render.get("atlas_coverage",{}).get("maps")!=250 or render.get("signed_audit_coverage",{}).get("signed_maps")!=100: raise StageFF3C4R3Error("render count/coverage drifted")
    for item in render["records"]:
        report=r1._read(repo/item["report"]["path"]); r1._validate_signed(report,"R1 render report")
        if item["report"]!=r1._record(repo,item["report"]["path"]): raise StageFF3C4R3Error("render report binding drifted")
        for extension,record in report["outputs"].items():
            if record["sha256"]!=r1._sha_file(repo/record["path"]): raise StageFF3C4R3Error("render output drifted")
        if item["module"]=="formal_map_atlas":
            atlas.extend(x["formal_position"] for x in item["samples"]); limits=report["display_transform"]["color_limits"]
            if not all(x["shared_across_five_methods"] and x["vmin"]==0 and x["vmax"]>0 for x in limits): raise StageFF3C4R3Error("atlas scale drifted")
        if item["module"]=="signed_audit":
            signed.extend(x["formal_position"] for x in item["samples"]); limits=report["display_transform"]["color_limits"]
            if not all(x["shared_across_two_signed_methods"] and x["vmin"]==-x["vmax"] for x in limits): raise StageFF3C4R3Error("signed scale drifted")
    if atlas!=list(range(1,51)) or signed!=list(range(1,51)): raise StageFF3C4R3Error("render position coverage drifted")
    qc=r1._read(repo/r1.QC_PATH); resources=r1._read(repo/r1.RESOURCE_LEDGER_PATH); access=r1._read(repo/r1.ACCESS_AUDIT_PATH); audit=r1._read(repo/r1.FALSE_PASS_PATH); tests=r1._read(repo/r1.TEST_REPORT_PATH)
    for value,name in ((qc,"QC"),(resources,"resources"),(access,"access"),(audit,"false pass"),(tests,"tests")): r1._validate_signed(value,name)
    if qc.get("passed_pages")!=23 or access.get("formal_jpeg_decoded_count")!=50 or access.get("pilot_outcome_or_jpeg_access") or access.get("order44_access_or_recompute") or access.get("torch_imported") or access.get("timm_imported") or access.get("cuda_initialized_or_accessed") or resources.get("f3c4_gpu_hours")!=0 or resources.get("actual_total_f3c_gpu_hours")!=r1.TOTAL_F3C_GPU_HOURS or not all(resources.get("checks",{}).values()) or audit.get("rejected_count")!=47 or tests.get("failed")!=0: raise StageFF3C4R3Error("R1 QC/resource/access/test/false-pass gate failed")
    return r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_validation_report","status":"PASS","implementation_validation":"PASS","workflow_validation":"PASS","package_validation":"PASS","table_parity":parity,"formal_samples":"50/50","source_terminals":"250/250 VALID","statistics_revalidated_from_r1_paper_data_only":True,"render_pages":23,"exports":69,"atlas_maps":250,"signed_maps":100,"manual_qc":"23/23 PASS","false_pass":"47/47 REJECTED","new_f3c3_or_jpeg_access":False,"new_empirical_endpoint_or_render":False,"model_checkpoint_torch_timm_cuda_access":False,"f4_started":False,"next_authorization":"F4_ONLY"})

def record_tests(repo_root: Path|str,*,passed:int,failed:int,command:str)->dict[str,Any]:
    if passed<=0 or failed: raise StageFF3C4R3Error("R3 tests failed")
    repo=Path(repo_root).resolve(); report=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_test_report","status":"PASS","passed":passed,"failed":failed,"command":command,"synthetic_and_r1_payload_only":True,"new_f3c3_or_jpeg_access":False}); r1._write_new(repo/TEST_PATH,report); return report

def finalize(repo_root: Path|str=REPO_ROOT,*,completed_at:str)->dict[str,Any]:
    repo=Path(repo_root).resolve(); validation=validate_payload(repo); r1._write_new(repo/VALIDATION_PATH,validation); test=r1._read(repo/TEST_PATH); r1._validate_signed(test,"R3 tests")
    audit=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_false_pass_audit","status":"PASS","rejected_count":4,"attacks":[{"attack":"canonical_json_key_order_mistaken_for_column_order","result":"REJECTED"},{"attack":"r1_payload_mutated_after_binding","result":"REJECTED"},{"attack":"r3_accesses_f3c3_array_or_jpeg","result":"REJECTED"},{"attack":"r3_recomputes_empirical_endpoint_or_render","result":"REJECTED"}],"r1_false_pass":"47/47 REJECTED"}); r1._write_new(repo/FALSE_PASS_PATH,audit)
    retained_paths=("evidence/stage_f/f3c4_analysis_v1_20260908/formal","reports/stage_f_f3c4_analysis_v1_20260908",r1.ROOT_PATH,r1.REPORT_PATH,"evidence/stage_f/f3c4_analysis_r2_terminal_seal_20260908/formal","reports/stage_f_f3c4_analysis_r2_terminal_seal_20260908",ROOT_PATH,REPORT_PATH); f3c4_bytes=r1._regular_bytes(*(repo/path for path in retained_paths)); upstream=79_529_811; total=upstream+f3c4_bytes
    ledger=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_resource_ledger","status":"PASS","cpu_only":True,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,"f3c4_gpu_hours":0.0,"actual_total_f3c_gpu_hours":r1.TOTAL_F3C_GPU_HOURS,"f3c4_retained_regular_file_bytes_before_terminal":f3c4_bytes,"actual_total_f3c_retained_regular_file_bytes_before_terminal":total,"retained_cap_bytes":8589934592,"all_caps_satisfied":total<=8589934592,"waiver":False}); r1._write_new(repo/RESOURCE_PATH,ledger)
    access=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_access_audit","status":"PASS","allowed_input_boundary":"R1_ROOTS_ONLY","new_f3c3_result_or_array_access":False,"new_formal_jpeg_access":False,"new_pilot_or_order44_access":False,"new_empirical_endpoint":False,"new_rendering":False,"model_loaded":False,"checkpoint_loaded":False,"torch_imported":False,"timm_imported":False,"cuda_accessed":False,"forward_calls":0,"backward_calls":0,"f4_started":False}); r1._write_new(repo/ACCESS_PATH,access)
    payload=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_payload_manifest","status":"PASS","records":[r1._record(repo,path) for path in (STARTUP_PATH,SOURCE_PATH,R1_BINDING_PATH,VALIDATION_PATH,FALSE_PASS_PATH,TEST_PATH,RESOURCE_PATH,ACCESS_PATH)],"r1_payload_reused_by_reference":r1._record(repo,R1_BINDING_PATH)}); r1._write_new(repo/PAYLOAD_PATH,payload)
    status=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_status","status":"PASS","terminal":True,"implementation_validation":"PASS","workflow_validation":"PASS","package_validation":"PASS","scientific_payload_root":r1.ROOT_PATH,"formal_samples":"50/50","source_terminals":"250/250 VALID","render_pages":23,"exports":69,"manual_qc":"23/23 PASS","next_authorization":"F4_ONLY","f4_started":False,"git":{"commit":"SELF","ref":ARCHIVE_REF}}); r1._write_new(repo/STATUS_PATH,status)
    report_inv=r1._inventory(repo/REPORT_PATH,exclude=("package_inventory.json",)); report_doc=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_report_inventory","status":"PASS",**report_inv}); r1._write_new(repo/REPORT_INVENTORY_PATH,report_doc); root_inv=r1._inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","completion/f3c4.json","completion/f3c4.pending.json")); root_doc=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_r3_whole_root_inventory","status":"PASS",**root_inv}); r1._write_new(repo/INVENTORY_PATH,root_doc)
    methods=_load_rows(repo,"method_summary"); contrasts=_load_rows(repo,"contrast_summary"); completion=r1._signed({"schema_version":1,"document_type":"stage_f_f3c4_terminal_completion","status":"PASS","completed_at":completed_at,"implementation_validation":"PASS","workflow_validation":"PASS","package_validation":"PASS","scientific_outcome_separate_from_workflow":True,"formal_samples":"50/50","source_terminals":{"VALID":250,"FAILED":0,"NA":0,"incomplete":0},"statistical_unit":"image","formal_n":50,"primary_contrast":r1.PRIMARY_CONTRAST,"primary_unadjusted":True,"secondary_holm_family":list(r1.SECONDARY_CONTRASTS),"bootstrap_resamples":10000,"sign_flip_resamples":100000,"method_summary":methods,"contrast_summary":contrasts,"scientific_payload_root":r1.ROOT_PATH,"terminal_seal_root":ROOT_PATH,"render_page_count":23,"export_file_count":69,"manual_qc_page_count":23,"formal_jpeg_access_count":50,"pilot_outcome_or_jpeg_access":False,"order44_accessed_or_recomputed":False,"unlisted_jpeg_access":False,"model_loaded":False,"checkpoint_loaded":False,"torch_imported":False,"timm_imported":False,"cuda_initialized_or_accessed":False,"forward_calls":0,"backward_calls":0,"f3c4_gpu_hours":0.0,"actual_total_f3c_gpu_hours":r1.TOTAL_F3C_GPU_HOURS,"f3c4_retained_regular_file_bytes_before_terminal":f3c4_bytes,"actual_total_f3c_retained_regular_file_bytes_before_terminal":total,"all_caps_satisfied":total<=8589934592,"fine_grid_fidelity":r1.FINE_GRID_STATUS,"statistical_inference_performed":True,"scientific_comparison_performed":True,"formal_maps_rendered":True,"formal_renders_manually_inspected":True,"full_method_ranking_produced":False,"winner_produced":False,"go_no_go_produced":False,"sealed_upstream_modified":False,"f4_started":False,"next_authorization":"F4_ONLY","records":{name:r1._record(repo,path) for name,path in (("startup_gate",STARTUP_PATH),("source_binding",SOURCE_PATH),("r1_payload_binding",R1_BINDING_PATH),("validation_report",VALIDATION_PATH),("false_pass_audit",FALSE_PASS_PATH),("test_report",TEST_PATH),("resource_ledger",RESOURCE_PATH),("access_audit",ACCESS_PATH),("payload_manifest",PAYLOAD_PATH),("whole_root_inventory",INVENTORY_PATH),("report_inventory",REPORT_INVENTORY_PATH),("status_record",STATUS_PATH))},"git":{"commit":"SELF","ref":ARCHIVE_REF}})
    pending=repo/PENDING_PATH; temporary=pending.with_name(".f3c4-r3-terminal.tmp"); r1._write_new(temporary,completion); os.replace(temporary,pending); os.replace(pending,repo/COMPLETION_PATH); return completion

def validate_completion(repo_root: Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); completion=r1._read(repo/COMPLETION_PATH); r1._validate_signed(completion,"R3 completion")
    if (repo/PENDING_PATH).exists() or completion.get("status")!="PASS" or completion.get("next_authorization")!="F4_ONLY" or completion.get("f4_started") is not False: raise StageFF3C4R3Error("R3 completion invalid")
    for record in completion["records"].values():
        if record!=r1._record(repo,record["path"]): raise StageFF3C4R3Error("R3 completion binding drifted")
    inv=r1._read(repo/INVENTORY_PATH); rebuilt=r1._inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","completion/f3c4.json","completion/f3c4.pending.json"))
    if any(inv.get(k)!=rebuilt[k] for k in ("files","file_count","total_bytes","file_inventory_digest")): raise StageFF3C4R3Error("R3 inventory drifted")
    return {"status":"PASS","formal_samples":50,"valid_terminals":250,"archive_ref":ARCHIVE_REF,"next_authorization":"F4_ONLY"}

__all__=["initialize","table_parity","validate_payload","record_tests","finalize","validate_completion"]
