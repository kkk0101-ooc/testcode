"""Independent CPU-only validator for the sealed F3C4 package."""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .constants import REPO_ROOT
from . import stage_f_f3c4 as f3c4

TABLES={
    "per_image_method_scores":250, "per_image_contrasts":200,
    "method_summary":5, "contrast_summary":4, "curve_summary":210,
}

def _csv_value(raw: str, exemplar: Any) -> Any:
    if exemplar is None: return None if raw=="" else raw
    if type(exemplar) is bool: return {"true":True,"false":False}.get(raw,raw)
    if type(exemplar) is int: return int(raw)
    if type(exemplar) is float: return float(raw)
    return raw

def validate_table_parity(repo: Path, name: str) -> dict[str,Any]:
    document=f3c4._read(repo/f3c4.ROOT_PATH/"paper_data"/f"{name}.json"); f3c4._validate_signed(document,name); rows=document["rows"]
    with (repo/f3c4.ROOT_PATH/"paper_data"/f"{name}.csv").open(newline="",encoding="utf-8") as stream: csv_rows=list(csv.DictReader(stream))
    if document["row_count"]!=TABLES[name] or len(rows)!=TABLES[name] or len(csv_rows)!=len(rows) or document["columns"]!=list(rows[0]): raise f3c4.StageFF3C4Error(f"{name} row or column count drifted")
    for expected,observed in zip(rows,csv_rows):
        if list(observed)!=list(expected): raise f3c4.StageFF3C4Error(f"{name} CSV/JSON columns differ")
        for key,value in expected.items():
            parsed=_csv_value(observed[key],value)
            if type(value) is float:
                if not math.isclose(parsed,value,rel_tol=0,abs_tol=0): raise f3c4.StageFF3C4Error(f"{name} CSV/JSON float parity failed")
            elif parsed!=value: raise f3c4.StageFF3C4Error(f"{name} CSV/JSON parity failed")
    return {"status":"PASS","rows":len(rows),"columns":len(document["columns"])}

def validate_render_manifest(repo: Path) -> dict[str,Any]:
    manifest=f3c4._read(repo/f3c4.RENDER_MANIFEST_PATH); f3c4._validate_signed(manifest,"render manifest")
    if manifest.get("render_page_count")!=23 or manifest.get("export_file_count")!=69 or manifest.get("module_counts")!={"fidelity_summary":1,"frozen_contrast_summary":1,"insertion_deletion_curves":1,"formal_map_atlas":10,"signed_audit":10}: raise f3c4.StageFF3C4Error("render module/page counts drifted")
    if manifest.get("atlas_coverage")!={"samples":50,"methods_per_sample":5,"maps":250,"originals":50} or manifest.get("signed_audit_coverage")!={"samples":50,"signed_methods_per_sample":2,"signed_maps":100}: raise f3c4.StageFF3C4Error("render coverage is not exhaustive")
    positions=[]; signed_positions=[]
    for item in manifest["records"]:
        report=f3c4._read(repo/item["report"]["path"]); f3c4._validate_signed(report,"render report")
        if item["report"]!=f3c4._record(repo,item["report"]["path"]): raise f3c4.StageFF3C4Error("render report binding drifted")
        if report.get("torch_imported") or report.get("timm_imported") or report.get("cuda_accessed") or not report.get("display_only") or report.get("metric_modified_by_display"): raise f3c4.StageFF3C4Error("render boundary failed")
        for record in report["source_files"]:
            if record!=f3c4._file_record(repo,record["path"]): raise f3c4.StageFF3C4Error("render source hash drifted")
        for extension,record in report["outputs"].items():
            if record["path"]!=item["outputs"][extension]["path"] or record["sha256"]!=f3c4._sha_file(repo/record["path"]): raise f3c4.StageFF3C4Error("render output hash drifted")
        with Image.open(repo/report["outputs"]["png"]["path"]) as image:
            if image.width<1000 or image.height<1000: raise f3c4.StageFF3C4Error("formal render resolution too small")
        if item["module"]=="formal_map_atlas":
            positions.extend(x["formal_position"] for x in item["samples"]); limits=report["display_transform"]["color_limits"]
            if len(limits)!=5 or not all(x["shared_across_five_methods"] and x["vmin"]==0 and x["vmax"]>0 for x in limits): raise f3c4.StageFF3C4Error("atlas scale contract failed")
        if item["module"]=="signed_audit":
            signed_positions.extend(x["formal_position"] for x in item["samples"]); limits=report["display_transform"]["color_limits"]
            if len(limits)!=5 or not all(x["shared_across_two_signed_methods"] and x["vmin"]==-x["vmax"] for x in limits): raise f3c4.StageFF3C4Error("signed scale contract failed")
    if positions!=list(range(1,51)) or signed_positions!=list(range(1,51)): raise f3c4.StageFF3C4Error("render formal order drifted")
    jpeg=manifest["formal_jpeg_access"]
    if len(jpeg)!=50 or [x["formal_position"] for x in jpeg]!=list(range(1,51)) or len({x["path"] for x in jpeg})!=50 or not all(x["hash_verified_before_decode"] for x in jpeg): raise f3c4.StageFF3C4Error("formal JPEG access set drifted")
    return {"status":"PASS","pages":23,"exports":69,"atlas_maps":250,"signed_maps":100}

def validate_outputs(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); f3c4.verify_plan_sealed(repo)
    if "torch" in sys.modules or "timm" in sys.modules: raise f3c4.StageFF3C4Error("forbidden runtime imported during validation")
    for relative in (f3c4.RECONCILIATION_PATH,f3c4.STAT_BINDING_PATH,f3c4.RENDER_MANIFEST_PATH,f3c4.QC_PATH,f3c4.RESOURCE_LEDGER_PATH,f3c4.ACCESS_AUDIT_PATH,f3c4.TEST_REPORT_PATH,f3c4.FALSE_PASS_PATH):
        value=f3c4._read(repo/relative); f3c4._validate_signed(value,relative)
    tables={name:validate_table_parity(repo,name) for name in TABLES}; method_rows=f3c4._load_table(repo,"per_image_method_scores"); contrast_rows=f3c4._load_table(repo,"per_image_contrasts"); method_summary=f3c4._load_table(repo,"method_summary"); contrast_summary=f3c4._load_table(repo,"contrast_summary")
    if [x["method_id"] for x in method_summary]!=list(f3c4.METHODS) or len({(x["formal_position"],x["method_id"]) for x in method_rows})!=250: raise f3c4.StageFF3C4Error("method table roster drifted")
    if [x["contrast_id"] for x in contrast_summary]!=list(f3c4.CONTRASTS) or len({(x["formal_position"],x["contrast_id"]) for x in contrast_rows})!=200: raise f3c4.StageFF3C4Error("contrast table roster drifted")
    by_score={(x["selection_order"],x["method_id"]):x for x in method_rows}
    for row in method_rows:
        repeats=np.asarray([row[f"repeat_{i}_score"] for i in range(5)],dtype=np.float64)
        if not np.isclose(np.mean(repeats),row["image_mean_score"],atol=0,rtol=1e-15): raise f3c4.StageFF3C4Error("image score is not mean of five repeats")
    for row in contrast_rows:
        observed=by_score[(row["selection_order"],f3c4.METHODS[0])]["image_mean_score"]-by_score[(row["selection_order"],row["comparator_method_id"])]["image_mean_score"]
        if not math.isclose(observed,row["delta"],rel_tol=0,abs_tol=0): raise f3c4.StageFF3C4Error("contrast direction or value drifted")
    freeze=f3c4._read(repo/f3c4.F3C1_STAT_PATH); f3c4._validate_signed(freeze,"F3C1 statistical freeze")
    if freeze.get("status")!="FROZEN_BEFORE_PILOT_OUTCOME_ACCESS" or freeze["primary_contrast"]["id"]!=f3c4.PRIMARY_CONTRAST or freeze["secondary_holm_family"]["members_in_frozen_order"]!=list(f3c4.SECONDARY_CONTRASTS): raise f3c4.StageFF3C4Error("statistical freeze binding drifted")
    boot=f3c4._read(repo/f3c4.ROOT_PATH/"paper_data/bootstrap_report.json"); sign=f3c4._read(repo/f3c4.ROOT_PATH/"paper_data/sign_flip_report.json"); holm=f3c4._read(repo/f3c4.ROOT_PATH/"paper_data/holm_report.json")
    for value,name in ((boot,"bootstrap"),(sign,"sign flip"),(holm,"Holm")): f3c4._validate_signed(value,name)
    delta_by={c:np.asarray([x["delta"] for x in contrast_rows if x["contrast_id"]==c],dtype=np.float64) for c in f3c4.CONTRASTS}
    for contrast in f3c4.CONTRASTS:
        expected_boot=f3c4.bootstrap_mean_ci(delta_by[contrast],contrast); expected_sign=f3c4.paired_sign_flip_p_value(delta_by[contrast],contrast)
        if expected_boot!=next(x for x in boot["results"] if x["contrast"]==contrast) or expected_sign!=next(x for x in sign["results"] if x["contrast"]==contrast): raise f3c4.StageFF3C4Error("bootstrap or sign-flip reproducibility drifted")
    expected_holm=f3c4.holm_adjust({c:next(x["p_value"] for x in sign["results"] if x["contrast"]==c) for c in f3c4.SECONDARY_CONTRASTS})
    if holm["results"]!=expected_holm: raise f3c4.StageFF3C4Error("Holm result drifted")
    render=validate_render_manifest(repo); qc=f3c4._read(repo/f3c4.QC_PATH)
    if qc.get("status")!="PASS" or qc.get("reviewed_pages")!=23 or qc.get("passed_pages")!=23 or any(x.get("result")!="PASS" for x in qc.get("records",[])): raise f3c4.StageFF3C4Error("manual QC incomplete")
    access=f3c4._read(repo/f3c4.ACCESS_AUDIT_PATH); ledger=f3c4._read(repo/f3c4.RESOURCE_LEDGER_PATH); audit=f3c4._read(repo/f3c4.FALSE_PASS_PATH); tests=f3c4._read(repo/f3c4.TEST_REPORT_PATH)
    forbidden=(access.get("pilot_outcome_or_jpeg_access"),access.get("order44_access_or_recompute"),access.get("unlisted_jpeg_access"),access.get("model_loaded"),access.get("checkpoint_loaded"),access.get("torch_imported"),access.get("timm_imported"),access.get("cuda_initialized_or_accessed"),access.get("f4_started"))
    if any(forbidden) or access.get("formal_jpeg_decoded_count")!=50 or ledger.get("f3c4_gpu_hours")!=0 or ledger.get("actual_total_f3c_gpu_hours")!=f3c4.TOTAL_F3C_GPU_HOURS or not all(ledger.get("checks",{}).values()): raise f3c4.StageFF3C4Error("resource/access boundary failed")
    if audit.get("attack_count")!=47 or audit.get("rejected_count")!=47 or any(x.get("result")!="REJECTED" for x in audit.get("attacks",[])) or tests.get("failed")!=0: raise f3c4.StageFF3C4Error("false-pass or test gate failed")
    source=(repo/"src/vitnns/stage_f_f3c4.py").read_text()+"\n"+(repo/"src/vitnns/stage_f_f3c4_validate.py").read_text()
    if re.search(r"^\s*(?:from|import)\s+(torch|timm)(?:\.|\s|$)",source,re.MULTILINE): raise f3c4.StageFF3C4Error("F3C4 source imports forbidden model runtime")
    if subprocess.run(["git","diff","--check"],cwd=repo).returncode: raise f3c4.StageFF3C4Error("git diff --check failed")
    return f3c4._signed({"schema_version":1,"document_type":"stage_f_f3c4_validation_report","status":"PASS","implementation_validation":"PASS","workflow_validation":"PASS","package_validation":"PASS","formal_samples":"50/50","source_terminals":"250/250 VALID","formal_n":50,"tables":tables,"rendering":render,"manual_qc":"23/23 PASS","false_pass_attacks":"47/47 REJECTED","cpu_only":True,"f3c4_gpu_hours":0.0,"actual_total_f3c_gpu_hours":f3c4.TOTAL_F3C_GPU_HOURS,"formal_jpeg_access_count":50,"pilot_access":False,"order44_access":False,"sealed_upstream_modified":False,"full_method_ranking_winner_or_go_no_go":False,"f4_started":False,"next_authorization":"F4_ONLY"})

__all__=["TABLES","validate_outputs","validate_render_manifest","validate_table_parity"]
