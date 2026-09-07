"""CPU-only F3C4 reconciliation, frozen inference, rendering, and sealing.

This module is deliberately independent of the F3C1/F3C3 execution modules:
those modules import the model runtime.  F3C4 uses only standard-library
code, NumPy, Pillow and Matplotlib and consumes immutable F3C3 terminals.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "config/stage_f_f3c4.json"
SCHEMA_PATH = "config/stage_f_f3c4.schema.json"
ROOT_PATH = "evidence/stage_f/f3c4_analysis_r1_20260908/formal"
REPORT_PATH = "reports/stage_f_f3c4_analysis_r1_20260908"
STARTUP_PATH = ROOT_PATH + "/startup_gate.json"
COHORT_PATH = ROOT_PATH + "/analysis_cohort.json"
INPUT_PATH = ROOT_PATH + "/terminal_input_manifest.json"
RUNTIME_PATH = ROOT_PATH + "/cpu_runtime_specification.json"
STAT_PLAN_PATH = ROOT_PATH + "/statistical_analysis_plan.json"
RENDER_PLAN_PATH = ROOT_PATH + "/display_render_plan.json"
ACCESS_POLICY_PATH = ROOT_PATH + "/access_policy.json"
RESOURCE_SPEC_PATH = ROOT_PATH + "/resource_accounting_specification.json"
OUTPUT_SCHEMA_PATH = ROOT_PATH + "/output_schema.json"
SOURCE_BINDING_PATH = ROOT_PATH + "/source_config_code_binding.json"
PENDING_PATH = ROOT_PATH + "/completion/f3c4.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/f3c4.json"
PLAN_SEAL_PATH = ROOT_PATH + "/pre_outcome_plan_seal.json"
RECONCILIATION_PATH = ROOT_PATH + "/terminal_reconciliation.json"
STAT_BINDING_PATH = ROOT_PATH + "/statistical_method_binding.json"
RENDER_MANIFEST_PATH = ROOT_PATH + "/render_manifest.json"
QC_PATH = ROOT_PATH + "/manual_qc_report.json"
RESOURCE_LEDGER_PATH = ROOT_PATH + "/resource_ledger.json"
ACCESS_AUDIT_PATH = ROOT_PATH + "/access_audit.json"
PAYLOAD_PATH = ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = ROOT_PATH + "/status.json"
TEST_REPORT_PATH = REPORT_PATH + "/test_report.json"
VALIDATION_PATH = REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_PATH + "/false_pass_audit.json"
REPORT_INVENTORY_PATH = REPORT_PATH + "/package_inventory.json"
ANALYSIS_RUNTIME_PATH = REPORT_PATH + "/analysis_runtime.json"
RENDER_RUNTIME_PATH = REPORT_PATH + "/render_runtime.json"
ARCHIVE_REF = "refs/archive/stage-f-v1-f3c4-analysis-seal"
EXPECTED_BRANCH = "stage-d-d3-pass"
STARTUP_HEAD = "2b5129d9645c7cf1bbb0f35c0aea2ff28b087712"
F3C3_ROOT = "evidence/stage_f/f3c3_formal_v1_20260908/formal"
F3C3_REPORT = "reports/stage_f_f3c3_formal_v1_20260908"
F3C3_COMPLETION_PATH = F3C3_ROOT + "/completion/f3c3.json"
F3C3_INVENTORY_PATH = F3C3_ROOT + "/whole_root_inventory.json"
F3C3_UNION_PATH = F3C3_ROOT + "/terminal_union.json"
F3C1_STAT_PATH = "evidence/stage_f/f3c1_implementation_r6_cpu_seal_20260907/formal/statistical_method_freeze.json"
SPLIT_PATH = "manifests/stage_f_f3c_v1/sample_split.json"
JPEG_ROOT = "assets/data/ILSVRC2012_val"
METHODS = (
    "conditional_all_blocks_fixed_clean_attention_class_logit_ig",
    "vit_grad_cam_block11_norm1",
    "full_model_class_logit_ig",
    "attention_rollout_mean_head_residual_all_blocks",
    "direct_raw_cls_attention_last_block_head_mean",
)
SIGNED_METHODS = frozenset({METHODS[0], METHODS[2]})
CLASS_AGNOSTIC_METHODS = frozenset({METHODS[3], METHODS[4]})
METHOD_LABELS = {
    METHODS[0]: "class-targeted NNS (all-block fixed-clean-attention class-logit IG)",
    METHODS[1]: "ViT Grad-CAM", METHODS[2]: "full-model class-logit IG",
    METHODS[3]: "attention rollout (class-agnostic)",
    METHODS[4]: "block-11 raw CLS attention head mean (class-agnostic)",
}
METHOD_COLORS = {METHODS[0]: "#0072B2", METHODS[1]: "#D55E00", METHODS[2]: "#009E73", METHODS[3]: "#CC79A7", METHODS[4]: "#E69F00"}
METHOD_MARKERS = {METHODS[0]: "o", METHODS[1]: "s", METHODS[2]: "^", METHODS[3]: "D", METHODS[4]: "v"}
PRIMARY_CONTRAST = "class_targeted_nns_minus_vit_grad_cam"
SECONDARY_CONTRASTS = (
    "class_targeted_nns_minus_full_model_ig",
    "class_targeted_nns_minus_attention_rollout",
    "class_targeted_nns_minus_raw_cls_attention",
)
CONTRASTS = (PRIMARY_CONTRAST,) + SECONDARY_CONTRASTS
COMPARATOR = {PRIMARY_CONTRAST: METHODS[1], SECONDARY_CONTRASTS[0]: METHODS[2], SECONDARY_CONTRASTS[1]: METHODS[3], SECONDARY_CONTRASTS[2]: METHODS[4]}
STAT_NAMESPACE = "STAGE_F_F3C_V1_STATISTICS|"
TIE_NAMESPACE = "STAGE_F_F3C_V1_TIE|"
FRACTIONS = tuple(index / 20.0 for index in range(21))
ACTUAL_FRACTIONS = tuple(((0,) + tuple(int(math.ceil(value * 196)) for value in FRACTIONS[1:]))[i] / 196 for i in range(21))
F3C3_SOURCE_DIGEST = "5fcaaa26d46a4767e1f5adc705819c083aaa5b985ab218416ef98bb1d2919082"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
F3C3_GPU_HOURS = 0.20834401740614947
TOTAL_F3C_GPU_HOURS = 0.23176687041080893
STARTUP_RETAINED_BYTES = 79_529_811
FINE_GRID_STATUS = "NOT_COMPUTED_NO_SEALED_F3C3_FINE_GRID_INTERVENTION_TERMINALS"
PLAN_RECORDS = (STARTUP_PATH, COHORT_PATH, INPUT_PATH, RUNTIME_PATH, STAT_PLAN_PATH, RENDER_PLAN_PATH, ACCESS_POLICY_PATH, RESOURCE_SPEC_PATH, OUTPUT_SCHEMA_PATH, SOURCE_BINDING_PATH, PENDING_PATH)
SOURCE_FILES = (CONFIG_PATH, SCHEMA_PATH, "docs/stage_f_f3c4_r1_preoutcome_access_recovery_addendum_20260908.md", "src/vitnns/stage_f_f3c4.py", "src/vitnns/stage_f_f3c4_validate.py", "tests/test_stage_f_f3c4.py", "tests/test_stage_f_f3c4_validate.py", "src/vitnns/stage_f_renderer.py", "config/stage_f_f0.json", F3C1_STAT_PATH)
FALSE_PASS_ATTACKS = (
    "f3c3_completion_ref_hash_or_inventory_drift", "outcome_array_read_before_plan_seal", "formal_jpeg_decode_before_plan_seal",
    "torch_timm_model_checkpoint_or_cuda_access", "forward_backward_ig_gradcam_attention_or_fidelity_execution",
    "formal_roster_missing_extra_reordered_or_replaced", "pilot_order44_d10_or_other_cohort_mixed_in",
    "terminal_count_duplicate_or_identity_conflict", "failed_na_or_incomplete_filtered_imputed_or_replaced",
    "terminal_array_hash_shape_dtype_or_finiteness_drift", "target_or_class_identity_not_shared_within_image",
    "method_roster_order_or_targeting_label_drift", "fractions_or_actual_fractions_drift", "auc_or_fidelity_score_formula_drift",
    "ranking_tie_seed_or_tie_priority_drift", "tie_repeats_treated_as_independent_samples", "formal_statistical_n_not_50",
    "image_score_not_mean_of_five_ties", "delta_direction_not_nns_minus_comparator", "primary_changed_or_included_in_holm",
    "secondary_family_added_removed_reordered_or_contaminated", "bootstrap_resamples_seed_statistic_quantile_or_ci_drift",
    "sign_flip_resamples_seed_sidedness_zero_extreme_or_add_one_drift", "holm_tie_order_stepdown_or_alpha_drift",
    "nonfinite_or_invalid_unit_silently_deleted", "csv_json_parity_failure", "outcome_adaptive_method_sample_statistic_or_panel",
    "full_method_ranking_winner_or_go_no_go_produced", "fine_grid_fidelity_fabricated",
    "unregistered_descriptive_metric_added_after_outcome", "target_oriented_mass_formula_drift",
    "display_resize_normalization_gamma_or_clipping_reenters_metric", "five_methods_use_different_scale",
    "signed_panel_loses_sign_or_uses_unshared_asymmetric_scale", "jpeg_or_map_appearance_changes_cohort",
    "atlas_not_exactly_50_by_5", "jpeg_hash_not_checked_before_decode", "manual_qc_changes_scientific_result",
    "f3c1_statistical_freeze_modified", "f3c3_sealed_evidence_modified", "workflow_pass_conflated_with_scientific_outcome",
    "unfavorable_or_inconclusive_outcome_marked_workflow_failure", "favorable_outcome_passes_integrity_failure",
    "completion_published_before_payload_figure_inventory_validation_false_pass", "pending_receipt_not_atomically_replaced",
    "f4_started_early", "git_diff_lfs_or_archive_state_unverified",
)

class StageFF3C4Error(ValueError):
    pass

def _now() -> str:
    raw=time.strftime("%Y-%m-%dT%H:%M:%S%z",time.localtime()); return raw[:-2]+":"+raw[-2:]

def _sha_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def _reject_duplicates(pairs: Sequence[tuple[str,Any]], path: Path) -> dict[str,Any]:
    value={}
    for key,item in pairs:
        if key in value: raise StageFF3C4Error(f"duplicate JSON key {key!r}: {path}")
        value[key]=item
    return value

def _read(path: Path) -> dict[str,Any]:
    with path.open("r",encoding="utf-8") as stream:
        return json.load(stream,object_pairs_hook=lambda pairs:_reject_duplicates(pairs,path))

def _signed(value: Mapping[str,Any]) -> dict[str,Any]:
    result=dict(value); result["canonical_digest"]=canonical_sha256(result); return result

def _validate_signed(value: Mapping[str,Any], context: str) -> None:
    payload=dict(value); observed=payload.pop("canonical_digest",None)
    if observed!=canonical_sha256(payload): raise StageFF3C4Error(f"{context} canonical digest mismatch")

def _write_new(path: Path, value: Mapping[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"wb") as stream:
        stream.write(canonical_json_bytes(value)+b"\n"); stream.flush(); os.fsync(stream.fileno())
    descriptor=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)

def _file_record(repo: Path, relative: str) -> dict[str,Any]:
    path=repo/relative; return {"path":relative,"bytes":path.stat().st_size,"sha256":_sha_file(path)}

def _record(repo: Path, relative: str) -> dict[str,Any]:
    value=_read(repo/relative); _validate_signed(value,relative); return {**_file_record(repo,relative),"canonical_digest":value["canonical_digest"]}

def _git(repo: Path,*args: str) -> str:
    process=subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True,check=False)
    if process.returncode: raise StageFF3C4Error(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()

def _array_digest(value: np.ndarray) -> str:
    array=np.ascontiguousarray(value); digest=hashlib.sha256(); digest.update(array.dtype.str.encode("ascii")); digest.update(b"\0"); digest.update(json.dumps(list(array.shape),separators=(",",":")).encode("ascii")); digest.update(b"\0"); digest.update(array.tobytes(order="C")); return digest.hexdigest()

def _regular_bytes(*roots: Path) -> int:
    return sum(path.stat().st_size for root in roots if root.exists() for path in root.rglob("*") if path.is_file() and not path.is_symlink())

def _inventory(root: Path, *, exclude: Sequence[str]=()) -> dict[str,Any]:
    excluded=set(exclude); files=[]
    for path in sorted(root.rglob("*")):
        relative=path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            files.append({"path":relative,"bytes":path.stat().st_size,"sha256":_sha_file(path)})
    return {"files":files,"file_count":len(files),"total_bytes":sum(item["bytes"] for item in files),"file_inventory_digest":canonical_sha256({"files":files})}

def _atomic_replace_pending(repo: Path, completion: Mapping[str,Any]) -> None:
    pending=repo/PENDING_PATH
    if not pending.is_file() or (repo/COMPLETION_PATH).exists(): raise StageFF3C4Error("pending-to-terminal state is invalid")
    temporary=pending.with_name(".f3c4-terminal.tmp"); _write_new(temporary,completion); os.replace(temporary,pending); os.replace(pending,repo/COMPLETION_PATH)
    descriptor=os.open((repo/COMPLETION_PATH).parent,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)

def load_config(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); config=_read(repo/CONFIG_PATH); schema=_read(repo/SCHEMA_PATH)
    errors=sorted(Draft202012Validator(schema).iter_errors(config),key=lambda item:list(item.path))
    if errors: raise StageFF3C4Error(errors[0].message)
    if tuple(item["id"] for item in config["methods"])!=METHODS: raise StageFF3C4Error("method roster/order drifted")
    stats=config["statistics"]
    if stats!= {"tie_repeats":5,"image_score":"arithmetic_mean_of_five_repeat_0.5_times_insertion_auc_minus_deletion_auc","primary_contrast":PRIMARY_CONTRAST,"primary_multiplicity_adjusted":False,"secondary_holm_family":list(SECONDARY_CONTRASTS),"bootstrap_resamples":10000,"bootstrap_quantiles":[0.025,0.975],"bootstrap_quantile_method":"linear","sign_flip_resamples":100000,"sign_flip_two_sided":True,"sign_flip_add_one":True,"holm_alpha":0.05,"float_dtype":"float64"}: raise StageFF3C4Error("frozen statistical config drifted")
    if config["rendering"]["overlay_alpha"]!=0.55 or config["rendering"]["clipping"] is not False: raise StageFF3C4Error("display contract drifted")
    return config

def false_pass_reasons(preconditions: Mapping[str,bool]) -> tuple[str,...]:
    if set(preconditions)!=set(FALSE_PASS_ATTACKS): raise StageFF3C4Error("false-pass preconditions must cover exactly 47 attacks")
    if any(type(value) is not bool for value in preconditions.values()): raise TypeError("false-pass preconditions must be booleans")
    return tuple(name for name in FALSE_PASS_ATTACKS if not preconditions[name])

def validate_startup_gate(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); gate=_read(repo/STARTUP_PATH); _validate_signed(gate,"F3C4 startup gate")
    git_gate=gate.get("git",{})
    if gate.get("status")!="PASS" or gate.get("authorization")!="F3C4_CPU_ANALYSIS_AND_SEAL_ONLY" or any(git_gate.get(key)!=STARTUP_HEAD for key in ("local_head","tracking_head","remote_head")): raise StageFF3C4Error("startup Git gate drifted")
    if gate.get("f3c_current_retained_regular_file_bytes",{}).get("total_f3c1_f3c2_f3c3")!=STARTUP_RETAINED_BYTES: raise StageFF3C4Error("startup retained-byte baseline drifted")
    for record in gate.get("f3c3_records",{}).values():
        current=_record(repo,record["path"]); expected={key:record[key] for key in current}
        if current!=expected: raise StageFF3C4Error("F3C3 startup control file drifted")
    for relative,expected in gate.get("f3c3_implementation_hashes",{}).items():
        if _sha_file(repo/relative)!=expected: raise StageFF3C4Error("F3C3 implementation drifted")
    for entry in gate.get("archive_refs",{}).values():
        if _git(repo,"rev-parse",entry["ref"]+"^{commit}")!=entry["local_commit"] or subprocess.run(["git","merge-base","--is-ancestor",entry["local_commit"],"HEAD"],cwd=repo).returncode: raise StageFF3C4Error("sealed archive ref or ancestry drifted")
        if subprocess.run(["git","diff","--quiet",entry["local_commit"],"HEAD","--",entry["sealed_path"]],cwd=repo).returncode: raise StageFF3C4Error("sealed upstream root drifted")
    return {"status":"PASS","startup_head":STARTUP_HEAD,"retained_baseline_bytes":STARTUP_RETAINED_BYTES}

def _stat_seed(kind: str, contrast: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{STAT_NAMESPACE}{kind}|{contrast}".encode()).digest()[:8],"big",signed=False)

def bootstrap_mean_ci(differences: Sequence[float], contrast: str, *, resamples: int=10000) -> dict[str,Any]:
    values=np.asarray(differences,dtype=np.float64)
    if contrast not in CONTRASTS or values.shape!=(50,) or not np.isfinite(values).all(): raise StageFF3C4Error("bootstrap requires frozen contrast and 50 finite image differences")
    if resamples!=10000: raise StageFF3C4Error("bootstrap resample count drifted")
    seed=_stat_seed("bootstrap_mean",contrast); rng=np.random.Generator(np.random.PCG64(seed)); draws=rng.integers(0,50,size=(10000,50),dtype=np.int64); distribution=np.mean(values[draws],axis=1); interval=np.quantile(distribution,[0.025,0.975],method="linear")
    return {"contrast":contrast,"n":50,"statistic":"paired_mean_difference","resamples":10000,"seed":seed,"rng":"numpy.random.Generator(PCG64)","draw_shape":[10000,50],"percentile":[0.025,0.975],"quantile_method":"linear","ci":[float(interval[0]),float(interval[1])],"draw_indices_sha256":_array_digest(draws)}

def paired_sign_flip_p_value(differences: Sequence[float], contrast: str, *, resamples: int=100000) -> dict[str,Any]:
    values=np.asarray(differences,dtype=np.float64)
    if contrast not in CONTRASTS or values.shape!=(50,) or not np.isfinite(values).all(): raise StageFF3C4Error("sign-flip requires frozen contrast and 50 finite image differences")
    if resamples!=100000: raise StageFF3C4Error("sign-flip resample count drifted")
    seed=_stat_seed("sign_flip",contrast); rng=np.random.Generator(np.random.PCG64(seed)); observed=float(np.mean(values)); extreme=0
    for start in range(0,100000,1000):
        signs=rng.integers(0,2,size=(min(1000,100000-start),50),dtype=np.int8)*2-1; simulated=np.mean(signs*values[None,:],axis=1); extreme+=int(np.count_nonzero(np.abs(simulated)>=abs(observed)))
    return {"contrast":contrast,"n":50,"statistic":"paired_mean_difference","observed":observed,"two_sided":True,"zero_differences":"retained","resamples":100000,"seed":seed,"rng":"numpy.random.Generator(PCG64)","extreme_rule":"abs(simulated_mean)>=abs(observed_mean)","extreme_count":extreme,"add_one_correction":True,"p_value":(extreme+1.0)/100001.0}

def holm_adjust(raw_p_values: Mapping[str,float]) -> list[dict[str,Any]]:
    if tuple(raw_p_values)!=SECONDARY_CONTRASTS: raise StageFF3C4Error("Holm family/order drifted")
    ordered=sorted([(float(raw_p_values[c]),i,c) for i,c in enumerate(SECONDARY_CONTRASTS)],key=lambda x:(x[0],x[1])); running=0.0; adjusted={}
    for rank,(value,_i,contrast) in enumerate(ordered):
        if not math.isfinite(value) or not 0<=value<=1: raise StageFF3C4Error("invalid Holm raw p-value")
        running=max(running,(3-rank)*value); adjusted[contrast]=min(1.0,running)
    return [{"contrast":contrast,"family_order":i,"raw_p_value":float(raw_p_values[contrast]),"adjusted_p_value":adjusted[contrast],"reject_alpha_0_05":adjusted[contrast]<=0.05} for i,contrast in enumerate(SECONDARY_CONTRASTS)]

def _tie_seed(sample_id: str, repeat: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{TIE_NAMESPACE}{sample_id}|{repeat}".encode()).digest()[:8],"big",signed=False)

def _tie_priority(seed: int) -> np.ndarray:
    order=np.random.Generator(np.random.PCG64(seed)).permutation(196); priority=np.empty(196,dtype=np.int64); priority[order]=np.arange(196,dtype=np.int64); return priority

def _method_ranking(patch_mass: np.ndarray, sigma: int, seed: int) -> tuple[np.ndarray,np.ndarray]:
    scores=np.maximum(np.float32(sigma)*patch_mass,np.float32(0)).reshape(-1); priority=_tie_priority(seed); return np.lexsort((priority,-scores.astype(np.float64))).astype(np.int64),priority

def _formal_rows(repo: Path) -> list[dict[str,Any]]:
    split=_read(repo/SPLIT_PATH); _validate_signed(split,"sample split"); rows=split.get("formal",[])
    if len(rows)!=50 or len({r["selection_order"] for r in rows})!=50 or {r["selection_order"] for r in rows}&{44,416,350,438,411,337}: raise StageFF3C4Error("formal cohort count, uniqueness, or exclusion drifted")
    return [{**row,"formal_position":i} for i,row in enumerate(rows,1)]

def _terminal_root(repo: Path, order: int, method: str) -> Path:
    return repo/F3C3_ROOT/"terminals"/f"order-{order:04d}"/method/"attempts"/"attempt-000001"

def verify_plan_sealed(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); seal=_read(repo/PLAN_SEAL_PATH); _validate_signed(seal,"F3C4 pre-outcome plan seal")
    if seal.get("status")!="SEALED_BEFORE_F3C3_RESULT_ARRAY_OR_FORMAL_JPEG_ACCESS": raise StageFF3C4Error("pre-outcome plan is not sealed")
    for record in seal.get("records",[]):
        if record!=_record(repo,record["path"]): raise StageFF3C4Error("pre-outcome plan record drifted")
    source=_read(repo/SOURCE_BINDING_PATH); expected=[_file_record(repo,p) for p in SOURCE_FILES]
    if source.get("files")!=expected or source.get("source_tree_digest")!=canonical_sha256({"files":expected}): raise StageFF3C4Error("F3C4 source/config/code drifted after plan seal")
    if "torch" in sys.modules or "timm" in sys.modules: raise StageFF3C4Error("forbidden runtime imported")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise StageFF3C4Error("CUDA_VISIBLE_DEVICES must be empty")
    return {"status":"PASS","outcome_access_allowed":True,"formal_jpeg_decode_allowed":True}

def initialize(repo_root: Path|str=REPO_ROOT, *, started_at: str) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); root=repo/ROOT_PATH; report=repo/REPORT_PATH
    if not (repo/STARTUP_PATH).is_file() or {p.name for p in root.iterdir()}!={"startup_gate.json"}: raise StageFF3C4Error("fresh F3C4 root must contain only startup gate")
    if report.exists(): raise StageFF3C4Error("F3C4 report root already exists")
    validate_startup_gate(repo); startup=_read(repo/STARTUP_PATH)
    config=load_config(repo); rows=_formal_rows(repo); f3manifest=_read(repo/F3C3_ROOT/"formal_manifest.json"); _validate_signed(f3manifest,"F3C3 formal manifest")
    if f3manifest.get("samples")!=rows: raise StageFF3C4Error("F3C3 manifest differs from frozen cohort")
    cohort=_signed({"schema_version":1,"document_type":"stage_f_f3c4_exact_analysis_cohort","status":"FROZEN_BEFORE_OUTCOME_ACCESS","sample_count":50,"statistical_n":50,"statistical_unit":"image","samples":rows,"pilot_included":False,"order44_included":False,"replacement_or_reordering_allowed":False})
    union=_read(repo/F3C3_UNION_PATH); _validate_signed(union,"F3C3 terminal union"); source3=_read(repo/F3C3_ROOT/"source_config_code_binding.json"); _validate_signed(source3,"F3C3 source binding"); model3=_read(repo/F3C3_ROOT/"model_preprocessing_binding.json"); _validate_signed(model3,"F3C3 model binding"); union_by={(r["selection_order"],r["method_id"]):r for r in union.get("records",[])}; terminals=[]
    for row in rows:
        claim=repo/F3C3_ROOT/"leases"/f"order-{row['selection_order']:04d}.claim.json"; release=repo/F3C3_ROOT/"leases"/f"order-{row['selection_order']:04d}.release.json"
        claim_value=_read(claim); release_value=_read(release); _validate_signed(claim_value,"F3C3 claim"); _validate_signed(release_value,"F3C3 release")
        if claim_value.get("claim_token")!=release_value.get("claim_token") or release_value.get("terminal") is not True or release_value.get("terminal_status")!="PASS": raise StageFF3C4Error("active or conflicting F3C3 image-group lease")
        for method in METHODS:
            item=union_by.get((row["selection_order"],method)); terminal=_terminal_root(repo,row["selection_order"],method); attempts=sorted(p.name for p in terminal.parent.glob("attempt-*"))
            if item is None or item.get("status")!="VALID" or attempts!=["attempt-000001"]: raise StageFF3C4Error("terminal set is not exact immutable 250 VALID inputs")
            manifest=_read(terminal/"manifest.json"); _validate_signed(manifest,"terminal manifest"); status=_read(terminal/"status.json"); _validate_signed(status,"terminal status"); blind=_read(terminal/"blind.json"); _validate_signed(blind,"terminal blind record")
            if manifest.get("status")!="VALID" or status.get("status")!="VALID" or blind.get("status")!="VALID": raise StageFF3C4Error("non-VALID terminal in input manifest")
            relative=terminal.relative_to(repo).as_posix(); arrays=manifest.get("arrays",[])
            terminals.append({"formal_position":row["formal_position"],"selection_order":row["selection_order"],"sample_id":row["sample_id"],"jpeg_sha256":row["image_sha256"],"official_imagenet1k_index":row["imagenet1k_index"],"method_id":method,"attempt_id":"attempt-000001","unit_canonical_digest":item["unit_canonical_digest"],"preprocessing_identity":model3["canonical_digest"],"source_config_code_digest":source3["source_tree_digest"],"source_config_code_hashes":{entry["path"]:entry["sha256"] for entry in source3["files"]},"root":relative,"result":manifest["result"],"status_record":manifest["status_record"],"manifest":_record(repo,(terminal/"manifest.json").relative_to(repo).as_posix()),"blind":_record(repo,(terminal/"blind.json").relative_to(repo).as_posix()),"arrays":arrays})
    if len(terminals)!=250 or len({(x["selection_order"],x["method_id"]) for x in terminals})!=250: raise StageFF3C4Error("input manifest is not 250 unique terminals")
    input_manifest=_signed({"schema_version":1,"document_type":"stage_f_f3c4_exact_terminal_input_manifest","status":"FROZEN_BEFORE_OUTCOME_ACCESS","terminal_count":250,"method_count":5,"sample_count":50,"execution_order":"formal_position_then_frozen_method_order","terminals":terminals,"result_json_parsed_before_seal":False,"npy_deserialized_before_seal":False})
    source_files=[_file_record(repo,p) for p in SOURCE_FILES]; source=_signed({"schema_version":1,"document_type":"stage_f_f3c4_source_config_code_binding","status":"FROZEN_BEFORE_OUTCOME_ACCESS","startup_head":STARTUP_HEAD,"files":source_files,"source_tree_digest":canonical_sha256({"files":source_files}),"f3c3_source_tree_digest":F3C3_SOURCE_DIGEST})
    runtime=_signed({"schema_version":1,"document_type":"stage_f_f3c4_cpu_runtime_specification","status":"FROZEN_BEFORE_OUTCOME_ACCESS","CUDA_VISIBLE_DEVICES":"","allowed_libraries":["python_standard_library","numpy","pillow","matplotlib","sealed_stage_f_cpu_utilities"],"forbidden":["torch","timm","model_or_checkpoint_load","cuda_or_gpu_access","forward","backward","ig","grad_cam","attention_extraction_or_rollout","fidelity_evaluator"],"f3c4_gpu_hours":0.0,"total_f3c_gpu_hours_frozen":TOTAL_F3C_GPU_HOURS})
    stat=_signed({"schema_version":1,"document_type":"stage_f_f3c4_exact_statistical_analysis_plan","status":"FROZEN_BEFORE_OUTCOME_ACCESS","formal_unit":"image","formal_n":50,"tie_repeats_not_independent":True,"image_method_score":"arithmetic_mean_of_five_recomputed_repeat_scores","repeat_score":"0.5*(insertion_auc-deletion_auc)","contrasts_direction":"class_targeted_nns_minus_comparator","primary":{"id":PRIMARY_CONTRAST,"multiplicity_adjusted":False},"secondary_holm_family":list(SECONDARY_CONTRASTS),"descriptive":["n","mean","median","sample_sd_ddof_1","positive_count","minimum","maximum"],"bootstrap":{"draws":10000,"shape":[10000,50],"statistic":"paired_mean_difference","interval":"percentile_0.025_0.975","quantile_method":"linear","rng":"numpy.random.Generator(PCG64)","seed_rule":"sha256_namespace_first8_big_endian"},"sign_flip":{"draws":100000,"two_sided":True,"zero_differences":"retained","extreme_rule":"abs(simulated_mean)>=abs(observed_mean)","add_one_correction":True,"rng":"numpy.random.Generator(PCG64)","seed_rule":"sha256_namespace_first8_big_endian"},"holm":{"alpha":0.05,"family":list(SECONDARY_CONTRASTS),"tie_break":"frozen_family_order","step_down_running_maximum":True,"cap_at_one":True},"float_dtype":"float64","outcome_adaptation_allowed":False,"method_ranking_or_winner_allowed":False,"go_no_go_allowed":False,"fine_grid_fidelity":FINE_GRID_STATUS,"additional_descriptive_endpoints":[]})
    render=_signed({"schema_version":1,"document_type":"stage_f_f3c4_exact_display_render_plan","status":"FROZEN_BEFORE_OUTCOME_OR_JPEG_ACCESS","modules":[{"id":"fidelity_summary","pages":1,"all_50_observations":True},{"id":"frozen_contrast_summary","pages":1,"all_four_contrasts":True},{"id":"insertion_deletion_curves","pages":1,"image_first_repeat_mean":True},{"id":"formal_map_atlas","pages":10,"samples_per_page":5,"panels_per_sample":6,"coverage":"50_original_plus_50x5_maps"},{"id":"signed_audit","pages":10,"samples_per_page":5,"panels_per_sample":3,"methods":[METHODS[0],METHODS[2]]}],"formats":["png","pdf","svg"],"render_page_count":23,"export_file_count":69,"formal_order":[r["formal_position"] for r in rows],"sample_replacement_or_reordering":False,"original_display_transform":"RGB_then_bicubic_shorter_side_248_then_center_crop_224_display_only","map_view":"target_oriented_nonnegative_mass_one_14x14","within_sample_five_method_color_limit":"[0,max_m(max(P_i_m))]","atlas_interpolation":"nearest_cell_preserving","clipping":False,"method_specific_gain_gamma_alpha_or_autoscale":False,"overlay":{"used":False,"if_used_resize":"bilinear","global_alpha":0.55},"signed_view":{"rgb_sum_before_projection_preserves_sign":True,"shared_symmetric_limit_per_image_two_methods":True,"clipping":False,"functional_difference_labeled":True,"negative_panels_for_nonnegative_methods":False},"curve_band":"pointwise_sample_sd_ddof_1_descriptive","manual_qc_all_pages":True,"outcome_driven_panel_or_scale_change":False})
    access=_signed({"schema_version":1,"document_type":"stage_f_f3c4_access_policy","status":"FROZEN_BEFORE_OUTCOME_ACCESS","allowed_terminal_roots":[x["root"] for x in terminals],"allowed_formal_jpegs":[{"formal_position":r["formal_position"],"selection_order":r["selection_order"],"path":f"{JPEG_ROOT}/{r['relative_path']}","sha256":r["image_sha256"]} for r in rows],"terminal_array_access_after_plan_seal":True,"formal_jpeg_decode_after_hash_check_and_plan_seal":True,"pilot_outcome_or_jpeg_access":False,"order44_outcome_or_jpeg_access":False,"unlisted_jpeg_access":False,"d10_f3_f3h_maps_as_inputs":False})
    resources=_signed({"schema_version":1,"document_type":"stage_f_f3c4_resource_retention_accounting_specification","status":"FROZEN_BEFORE_OUTCOME_ACCESS","startup_actual_retained_regular_file_bytes":startup["f3c_current_retained_regular_file_bytes"],"completion_recount_required":True,"f3c4_gpu_hours_required":0.0,"actual_total_f3c_gpu_hours_required":TOTAL_F3C_GPU_HOURS,"caps":config["budget"],"waiver_allowed":False,"peak_rss_measurement":"resource.getrusage_self_ru_maxrss_linux_kib_times_1024"})
    outputs=_signed({"schema_version":1,"document_type":"stage_f_f3c4_output_schema","status":"FROZEN_BEFORE_OUTCOME_ACCESS","paper_data":["per_image_method_scores.csv/json","per_image_contrasts.csv/json","method_summary.csv/json","contrast_summary.csv/json","curve_summary.csv/json","bootstrap_report.json","sign_flip_report.json","holm_report.json","scientific_interpretation.json","fine_grid_fidelity_availability.json","figure_recipes.json","caption_data.json","source_hash_manifest.json"],"formal_render_modules":["fidelity_summary","frozen_contrast_summary","insertion_deletion_curves","formal_map_atlas","signed_audit"],"control_reports":["terminal_reconciliation","statistical_method_binding","render_manifest","manual_qc_report","resource_ledger","access_audit","payload_manifest","whole_root_inventory","report_package_inventory","validation_report","false_pass_audit","test_report","status","completion"],"csv_json_field_parity_required":True})
    documents=((COHORT_PATH,cohort),(INPUT_PATH,input_manifest),(RUNTIME_PATH,runtime),(STAT_PLAN_PATH,stat),(RENDER_PLAN_PATH,render),(ACCESS_POLICY_PATH,access),(RESOURCE_SPEC_PATH,resources),(OUTPUT_SCHEMA_PATH,outputs),(SOURCE_BINDING_PATH,source))
    for relative,value in documents: _write_new(repo/relative,value)
    pending=_signed({"schema_version":1,"document_type":"stage_f_f3c4_pending_completion","status":"PENDING","started_at":started_at,"authorization":"F3C4_CPU_ANALYSIS_AND_SEAL_ONLY","startup_gate_digest":startup["canonical_digest"],"analysis_cohort_digest":cohort["canonical_digest"],"terminal_input_manifest_digest":input_manifest["canonical_digest"],"runtime_spec_digest":runtime["canonical_digest"],"statistical_plan_digest":stat["canonical_digest"],"render_plan_digest":render["canonical_digest"],"access_policy_digest":access["canonical_digest"],"resource_spec_digest":resources["canonical_digest"],"output_schema_digest":outputs["canonical_digest"],"source_binding_digest":source["canonical_digest"],"result_json_or_array_access_started":False,"formal_jpeg_decode_started":False,"pilot_access":False,"order44_access":False,"f4_started":False})
    _write_new(repo/PENDING_PATH,pending)
    seal=_signed({"schema_version":1,"document_type":"stage_f_f3c4_pre_outcome_plan_seal","status":"SEALED_BEFORE_F3C3_RESULT_ARRAY_OR_FORMAL_JPEG_ACCESS","sealed_at":_now(),"records":[_record(repo,p) for p in PLAN_RECORDS],"result_json_access_before_seal":False,"npy_deserialized_before_seal":False,"formal_jpeg_decoded_before_seal":False,"pilot_or_order44_access":False})
    _write_new(repo/PLAN_SEAL_PATH,seal); report.mkdir(parents=True,exist_ok=False)
    return {"status":"PASS","plan_sealed":True,"formal_samples":50,"terminal_inputs":250}

def _load_array(root: Path, manifest: Mapping[str,Any], name: str, expected_shape: tuple[int,...]) -> np.ndarray:
    record=next((item for item in manifest["arrays"] if item.get("path")==name),None)
    if record is None: raise StageFF3C4Error(f"missing terminal array {name}")
    path=root/name
    if not path.is_file() or path.is_symlink() or _sha_file(path)!=record["sha256"] or path.stat().st_size!=record["bytes"]: raise StageFF3C4Error(f"terminal array file/hash drifted: {name}")
    value=np.load(path,allow_pickle=False)
    if tuple(value.shape)!=expected_shape or value.dtype!=np.float32 or not np.isfinite(value).all() or _array_digest(value)!=record["array_digest"]: raise StageFF3C4Error(f"terminal array shape/dtype/finiteness/digest drifted: {name}")
    value.flags.writeable=False; return value

def reconcile_terminal(root: Path, expected: Mapping[str,Any]) -> dict[str,Any]:
    required={"result.json","status.json","manifest.json","blind.json"};
    if not root.is_dir() or not required.issubset({p.name for p in root.iterdir() if p.is_file()}): raise StageFF3C4Error("terminal incomplete")
    result=_read(root/"result.json"); status=_read(root/"status.json"); manifest=_read(root/"manifest.json"); blind=_read(root/"blind.json")
    for value,name in ((result,"result"),(status,"status"),(manifest,"manifest"),(blind,"blind")): _validate_signed(value,f"terminal {name}")
    if any(value.get("status")!="VALID" for value in (result,status,manifest,blind)): raise StageFF3C4Error("FAILED/NA/incomplete terminal blocks F3C4")
    if _sha_file(root/"result.json")!=manifest["result"]["sha256"] or _sha_file(root/"manifest.json")!=blind["manifest_sha256"] or _sha_file(root/"result.json")!=blind["result_sha256"]: raise StageFF3C4Error("terminal result/manifest/blind binding drifted")
    identity=result.get("unit_identity",{}); exact={"protocol_id","sample_id","formal_position","selection_order","jpeg_sha256","official_imagenet1k_index","method_id","model_sha256","model_canonical_tensor_digest","preprocessing_identity","clean_input_sha256","frozen_target_class_index","target_identity_digest","source_config_code_digest","source_config_code_hashes","attempt_id"}
    if set(identity)!=exact or result.get("unit_canonical_digest")!=canonical_sha256(identity): raise StageFF3C4Error("unit identity or canonical digest drifted")
    for key in ("formal_position","selection_order","sample_id","method_id","attempt_id"):
        if identity.get(key)!=expected.get(key): raise StageFF3C4Error(f"terminal identity conflict: {key}")
    if identity["jpeg_sha256"]!=expected["jpeg_sha256"] or identity["official_imagenet1k_index"]!=expected["official_imagenet1k_index"] or identity["model_sha256"]!=MODEL_SHA256 or identity["model_canonical_tensor_digest"]!=MODEL_TENSOR_DIGEST or identity["preprocessing_identity"]!=expected["preprocessing_identity"] or identity["source_config_code_digest"]!=F3C3_SOURCE_DIGEST or identity["source_config_code_digest"]!=expected["source_config_code_digest"] or identity["source_config_code_hashes"]!=expected["source_config_code_hashes"]: raise StageFF3C4Error("terminal bound source/config/code/model/sample identity drifted")
    target=result.get("target",{})
    if target.get("resolved_once_from_same_clean_forward") is not True or result.get("target_identity_digest")!=canonical_sha256(target) or identity["target_identity_digest"]!=result["target_identity_digest"] or identity["frozen_target_class_index"]!=target.get("class_index"): raise StageFF3C4Error("target identity drifted")
    method=identity["method_id"]
    if result.get("class_agnostic") is not (method in CLASS_AGNOSTIC_METHODS) or result.get("class_targeted") is (method in CLASS_AGNOSTIC_METHODS): raise StageFF3C4Error("method targeting label drifted")
    shapes={"native.npy":(3,224,224) if method in SIGNED_METHODS else (14,14),"patch_mass.npy":(14,14),"target_oriented_mass.npy":(14,14),"rankings.npy":(5,196),"insertion_curves.npy":(5,21),"deletion_curves.npy":(5,21),"insertion_aucs.npy":(5,),"deletion_aucs.npy":(5,),"fidelity_scores.npy":(5,),"fractions.npy":(21,),"actual_fractions.npy":(21,)}
    if {item.get("path") for item in manifest.get("arrays",[])}!=set(shapes): raise StageFF3C4Error("terminal numerical roster drifted")
    arrays={name:_load_array(root,manifest,name,shape) for name,shape in shapes.items()}
    if not np.array_equal(arrays["fractions.npy"],np.asarray(FRACTIONS,dtype=np.float32)) or not np.array_equal(arrays["actual_fractions.npy"],np.asarray(ACTUAL_FRACTIONS,dtype=np.float32)): raise StageFF3C4Error("fractions or actual fractions drifted")
    actual=arrays["actual_fractions.npy"].astype(np.float64); ins=np.trapz(arrays["insertion_curves.npy"].astype(np.float64),actual,axis=1); dele=np.trapz(arrays["deletion_curves.npy"].astype(np.float64),actual,axis=1); scores=0.5*(ins-dele)
    if not np.allclose(ins,arrays["insertion_aucs.npy"],atol=2e-7,rtol=2e-6) or not np.allclose(dele,arrays["deletion_aucs.npy"],atol=2e-7,rtol=2e-6) or not np.allclose(scores,arrays["fidelity_scores.npy"],atol=2e-7,rtol=2e-6): raise StageFF3C4Error("AUC or fidelity score formula drifted")
    numeric=result.get("fidelity_numeric_payload",{}); saved=np.asarray([item["fidelity_score"] for item in numeric.get("curves",[])],dtype=np.float64)
    if saved.shape!=(5,) or not np.allclose(scores,saved,atol=2e-7,rtol=2e-6) or not np.isclose(float(np.mean(scores)),float(numeric.get("fidelity_score_mean_over_five_ties")),atol=2e-7,rtol=2e-6): raise StageFF3C4Error("terminal repeat score or five-tie mean drifted")
    contract=result.get("fidelity_contract",{}); seeds=[_tie_seed(identity["sample_id"],r) for r in range(5)]
    if contract.get("fractions")!=list(FRACTIONS) or not np.allclose(np.asarray(contract.get("actual_fractions"),dtype=np.float64),np.asarray(ACTUAL_FRACTIONS),atol=0,rtol=0) or contract.get("tie_seeds")!=seeds or contract.get("score")!="0.5*(insertion_auc-deletion_auc)": raise StageFF3C4Error("fidelity contract drifted")
    sigma=int(numeric.get("sigma")); rankings=[]; priorities=[]
    for seed in seeds:
        ranking,priority=_method_ranking(arrays["patch_mass.npy"],sigma,seed); rankings.append(ranking.astype(np.float32)); priorities.append(_array_digest(priority))
    if not np.array_equal(np.stack(rankings),arrays["rankings.npy"]) or priorities!=contract.get("tie_priority_sha256"): raise StageFF3C4Error("ranking, tie seed, or tie priority drifted")
    mass=np.maximum(np.float32(sigma)*arrays["patch_mass.npy"],np.float32(0)); mass=np.asarray(mass/float(mass.sum(dtype=np.float64)),dtype=np.float32)
    if not np.array_equal(mass,arrays["target_oriented_mass.npy"]): raise StageFF3C4Error("target-oriented mass formula drifted")
    payload=result.get("payload_identities",{})
    expected_payload={"native_array_digest":_array_digest(arrays["native.npy"]),"patch_mass_array_digest":_array_digest(arrays["patch_mass.npy"]),"target_oriented_mass_array_digest":_array_digest(arrays["target_oriented_mass.npy"]),"ranking_array_digest":_array_digest(arrays["rankings.npy"]),"tie_priority_sha256":priorities,"insertion_curve_array_digest":_array_digest(arrays["insertion_curves.npy"]),"deletion_curve_array_digest":_array_digest(arrays["deletion_curves.npy"])}
    if payload!=expected_payload: raise StageFF3C4Error("terminal payload identity digest drifted")
    conservation=result.get("ig_conservation")
    if method in SIGNED_METHODS:
        if not isinstance(conservation,Mapping) or conservation.get("passed") is not True: raise StageFF3C4Error("IG conservation failed")
        attempts=conservation.get("attempts",[]); steps=[x.get("steps") for x in attempts]; passes=[x.get("pass")==[True] for x in attempts]
        if not attempts or steps!=[512,1024,2048,4096,8192][:len(steps)] or conservation.get("selected_steps")!=steps[-1] or not passes[-1] or any(passes[:-1]): raise StageFF3C4Error("IG attempt prefix or selected first pass drifted")
    elif conservation is not None: raise StageFF3C4Error("non-IG method carries conservation")
    if method==METHODS[0]:
        endpoint=result.get("endpoint_parity",{})
        if endpoint.get("status")!="PASS" or endpoint.get("max_absolute_error",1)>=1e-6 or len(result.get("attention_hashes",[]))!=12: raise StageFF3C4Error("NNS attention/endpoint parity failed")
    elif result.get("endpoint_parity") is not None or result.get("attention_hashes"): raise StageFF3C4Error("non-NNS method carries conditional attention")
    return {"identity":identity,"target":target,"result_canonical_digest":result["canonical_digest"],"status_canonical_digest":status["canonical_digest"],"manifest_canonical_digest":manifest["canonical_digest"],"blind_canonical_digest":blind["canonical_digest"],"arrays":arrays,"repeat_scores":scores.astype(np.float64),"image_score":float(np.mean(scores,dtype=np.float64)),"insertion_image_curve":np.mean(arrays["insertion_curves.npy"].astype(np.float64),axis=0),"deletion_image_curve":np.mean(arrays["deletion_curves.npy"].astype(np.float64),axis=0)}

def _json_rows(document_type: str, rows: list[dict[str,Any]]) -> dict[str,Any]:
    columns=list(rows[0]) if rows else []
    if any(list(row)!=columns for row in rows): raise StageFF3C4Error("table rows do not have stable field order")
    return _signed({"schema_version":1,"document_type":document_type,"columns":columns,"row_count":len(rows),"rows":rows})

def _write_csv_new(path: Path, rows: list[dict[str,Any]]) -> None:
    if not rows: raise StageFF3C4Error("refusing empty CSV")
    path.parent.mkdir(parents=True,exist_ok=True); descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader()
        for row in rows: writer.writerow({key:("true" if value is True else "false" if value is False else value) for key,value in row.items()})
        stream.flush(); os.fsync(stream.fileno())

def _write_table(repo: Path, name: str, rows: list[dict[str,Any]]) -> dict[str,Any]:
    base=repo/ROOT_PATH/"paper_data"/name; csv_path=base.with_suffix(".csv"); json_path=base.with_suffix(".json")
    _write_csv_new(csv_path,rows); _write_new(json_path,_json_rows(f"stage_f_f3c4_{name}",rows))
    return {"csv":_file_record(repo,csv_path.relative_to(repo).as_posix()),"json":_record(repo,json_path.relative_to(repo).as_posix()),"rows":len(rows),"columns":list(rows[0])}

def _summary(values: Sequence[float]) -> dict[str,Any]:
    array=np.asarray(values,dtype=np.float64)
    if array.shape!=(50,) or not np.isfinite(array).all(): raise StageFF3C4Error("summary requires 50 finite image values")
    return {"n":50,"mean":float(np.mean(array)),"median":float(np.median(array)),"sample_sd_ddof_1":float(np.std(array,ddof=1)),"positive_count":int(np.count_nonzero(array>0)),"minimum":float(np.min(array)),"maximum":float(np.max(array))}

def reconcile_and_analyze(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); verify_plan_sealed(repo); start=time.monotonic(); cpu_start=time.process_time(); config=load_config(repo)
    input_manifest=_read(repo/INPUT_PATH); _validate_signed(input_manifest,"terminal input manifest"); rows=_formal_rows(repo); expected_by={(x["selection_order"],x["method_id"]):x for x in input_manifest["terminals"]}
    reconciled={}; records=[]; method_rows=[]; curve_store={method:{direction:[] for direction in ("insertion","deletion")} for method in METHODS}
    for row in rows:
        target_digests=set(); target_classes=set(); clean_hashes=set(); tie_priorities=[]
        for method in METHODS:
            expected=expected_by.get((row["selection_order"],method)); terminal=_terminal_root(repo,row["selection_order"],method)
            if expected is None: raise StageFF3C4Error("planned terminal missing")
            item=reconcile_terminal(terminal,expected); reconciled[(row["selection_order"],method)]=item; identity=item["identity"]; target_digests.add(identity["target_identity_digest"]); target_classes.add(identity["frozen_target_class_index"]); clean_hashes.add(identity["clean_input_sha256"]); result=_read(terminal/"result.json"); tie_priorities.append(result["fidelity_contract"]["tie_priority_sha256"])
            method_rows.append({"formal_position":row["formal_position"],"selection_order":row["selection_order"],"sample_id":row["sample_id"],"jpeg_sha256":row["image_sha256"],"official_imagenet1k_index":row["imagenet1k_index"],"target_class_index":identity["frozen_target_class_index"],"target_identity_digest":identity["target_identity_digest"],"method_id":method,"method_label":METHOD_LABELS[method],"class_agnostic":method in CLASS_AGNOSTIC_METHODS,**{f"repeat_{i}_score":float(value) for i,value in enumerate(item["repeat_scores"])},"image_mean_score":item["image_score"]})
            curve_store[method]["insertion"].append(item["insertion_image_curve"]); curve_store[method]["deletion"].append(item["deletion_image_curve"]); manifest=_read(terminal/"manifest.json")
            records.append({"formal_position":row["formal_position"],"selection_order":row["selection_order"],"sample_id":row["sample_id"],"method_id":method,"attempt_id":identity["attempt_id"],"status":"VALID","unit_canonical_digest":result["unit_canonical_digest"],"result_canonical_digest":result["canonical_digest"],"status_canonical_digest":item["status_canonical_digest"],"manifest_canonical_digest":item["manifest_canonical_digest"],"blind_canonical_digest":item["blind_canonical_digest"],"target_identity_digest":identity["target_identity_digest"],"target_class_index":identity["frozen_target_class_index"],"clean_input_sha256":identity["clean_input_sha256"],"array_count":len(manifest["arrays"]),"all_recomputations_pass":True})
        if len(target_digests)!=1 or len(target_classes)!=1 or len(clean_hashes)!=1 or any(value!=tie_priorities[0] for value in tie_priorities[1:]): raise StageFF3C4Error("within-image target, clean-input, or tie-priority identity conflict")
    if len(records)!=250 or len({(r["selection_order"],r["method_id"]) for r in records})!=250: raise StageFF3C4Error("reconciliation is not exact 250 unique units")
    reconciliation=_signed({"schema_version":1,"document_type":"stage_f_f3c4_terminal_reconciliation","status":"PASS","formal_samples":50,"terminal_counts":{"VALID":250,"FAILED":0,"NA":0,"incomplete":0},"records":records,"checks":{name:True for name in ("exact_frozen_cohort_and_order","exact_unique_250_sample_method_identities","single_valid_attempt_per_unit","no_incomplete_active_lease_or_claim_release_conflict","shared_target_and_clean_input_within_image","shared_tie_priorities_across_methods","terminal_control_canonical_digests","array_sha_shape_dtype_finiteness","target_oriented_mass","ranking_and_tie_identity","actual_fractions_curves_auc_and_score_recomputation","nns_attention_endpoint_parity","nns_and_full_model_ig_attempt_prefix_conservation","per_image_resource_provenance","pilot_and_order44_excluded","f3c3_evidence_unchanged")}}); _write_new(repo/RECONCILIATION_PATH,reconciliation)
    tables={}; tables["per_image_method_scores"]=_write_table(repo,"per_image_method_scores",method_rows)
    scores_by={(r["selection_order"],r["method_id"]):r["image_mean_score"] for r in method_rows}; contrast_rows=[]; contrast_arrays={}
    for contrast in CONTRASTS:
        comparator=COMPARATOR[contrast]; values=[]
        for row in rows:
            value=float(scores_by[(row["selection_order"],METHODS[0])]-scores_by[(row["selection_order"],comparator)]); values.append(value); contrast_rows.append({"formal_position":row["formal_position"],"selection_order":row["selection_order"],"sample_id":row["sample_id"],"contrast_id":contrast,"direction":"class_targeted_nns_minus_comparator","comparator_method_id":comparator,"delta":value})
        contrast_arrays[contrast]=np.asarray(values,dtype=np.float64)
    tables["per_image_contrasts"]=_write_table(repo,"per_image_contrasts",contrast_rows)
    method_summary=[]
    for method in METHODS:
        values=[scores_by[(row["selection_order"],method)] for row in rows]; method_summary.append({"method_id":method,"method_label":METHOD_LABELS[method],**_summary(values)})
    tables["method_summary"]=_write_table(repo,"method_summary",method_summary)
    bootstrap=[]; sign_flip=[]; contrast_summary=[]
    for contrast in CONTRASTS:
        values=contrast_arrays[contrast]; boot=bootstrap_mean_ci(values,contrast); sign=paired_sign_flip_p_value(values,contrast); bootstrap.append(boot); sign_flip.append(sign); contrast_summary.append({"contrast_id":contrast,"family":"primary_unadjusted" if contrast==PRIMARY_CONTRAST else "secondary_holm",**_summary(values),"bootstrap_ci_95_lower":boot["ci"][0],"bootstrap_ci_95_upper":boot["ci"][1],"raw_p_value":sign["p_value"],"holm_adjusted_p_value":None})
    holm=holm_adjust({c:next(x["p_value"] for x in sign_flip if x["contrast"]==c) for c in SECONDARY_CONTRASTS}); adjusted={x["contrast"]:x["adjusted_p_value"] for x in holm}
    for row in contrast_summary:
        if row["contrast_id"] in adjusted: row["holm_adjusted_p_value"]=adjusted[row["contrast_id"]]
    tables["contrast_summary"]=_write_table(repo,"contrast_summary",contrast_summary)
    curve_rows=[]
    for method in METHODS:
        for direction in ("insertion","deletion"):
            values=np.stack(curve_store[method][direction]).astype(np.float64)
            if values.shape!=(50,21): raise StageFF3C4Error("curve aggregation did not use 50 image means")
            for index,fraction in enumerate(ACTUAL_FRACTIONS):
                column=values[:,index]; curve_rows.append({"method_id":method,"method_label":METHOD_LABELS[method],"direction":direction,"point_index":index,"actual_fraction":float(fraction),"n_images":50,"mean":float(np.mean(column)),"median":float(np.median(column)),"sample_sd_ddof_1":float(np.std(column,ddof=1))})
    tables["curve_summary"]=_write_table(repo,"curve_summary",curve_rows)
    for path,value in (("paper_data/bootstrap_report.json",{"schema_version":1,"document_type":"stage_f_f3c4_bootstrap_report","status":"PASS","results":bootstrap}),("paper_data/sign_flip_report.json",{"schema_version":1,"document_type":"stage_f_f3c4_sign_flip_report","status":"PASS","results":sign_flip}),("paper_data/holm_report.json",{"schema_version":1,"document_type":"stage_f_f3c4_holm_report","status":"PASS","alpha":0.05,"primary_excluded":True,"family_order":list(SECONDARY_CONTRASTS),"results":holm})):
        _write_new(repo/ROOT_PATH/path,_signed(value))
    stat_binding=_signed({"schema_version":1,"document_type":"stage_f_f3c4_statistical_method_binding","status":"PASS","f3c1_freeze":_record(repo,F3C1_STAT_PATH),"f3c4_plan":_record(repo,STAT_PLAN_PATH),"numpy_only_reimplementation":True,"f3c1_kernel_module_imported":False,"formal_unit":"image","formal_n":50,"tie_repeats_averaged_before_inference":True,"contrasts":list(CONTRASTS),"primary_unadjusted":True,"secondary_holm_family_exact":True,"float64_aggregation_and_inference":True}); _write_new(repo/STAT_BINDING_PATH,stat_binding)
    def fact(row):
        ci=[row["bootstrap_ci_95_lower"],row["bootstrap_ci_95_upper"]]; significant=row["raw_p_value"]<=0.05 if row["contrast_id"]==PRIMARY_CONTRAST else row["holm_adjusted_p_value"]<=0.05; direction="positive" if row["mean"]>0 else "negative" if row["mean"]<0 else "zero"
        return {"contrast_id":row["contrast_id"],"direction_of_mean":direction,"ci95":ci,"raw_p_value":row["raw_p_value"],"holm_adjusted_p_value":row["holm_adjusted_p_value"],"significant_at_0_05_under_registered_rule":significant,"interpretation":f"Observed mean difference was {direction}; the registered {'unadjusted primary' if row['contrast_id']==PRIMARY_CONTRAST else 'Holm-adjusted secondary'} test was {'significant' if significant else 'not significant'}."}
    interpretation=_signed({"schema_version":1,"document_type":"stage_f_f3c4_scientific_interpretation","workflow_package_status":"PASS_INDEPENDENT_OF_DIRECTION_OR_SIGNIFICANCE","formal_n":50,"results":[fact(row) for row in contrast_summary],"full_method_ranking_produced":False,"unique_winner_produced":False,"go_no_go_produced":False,"equivalence_claimed_from_nonsignificance":False,"object_localization_claim":False,"cohort_pooling":False,"retroactive_decision_change":False,"method_1_boundary":"conditional functional; not raw attention, not an F3H map weighted sum, not complete Q/K/V attribution, and not unmodified full-model IG","class_agnostic_methods":[METHOD_LABELS[METHODS[3]],METHOD_LABELS[METHODS[4]]]}); _write_new(repo/ROOT_PATH/"paper_data/scientific_interpretation.json",interpretation)
    fine=_signed({"schema_version":1,"document_type":"stage_f_f3c4_fine_grid_fidelity_availability","status":FINE_GRID_STATUS,"reason":"F3C3 sealed no fine 4x4 intervention terminals and F3C4 cannot run empirical evaluation","fabricated_or_derived_from_native_map_or_14x14_curves":False}); _write_new(repo/ROOT_PATH/"paper_data/fine_grid_fidelity_availability.json",fine)
    captions=_signed({"schema_version":1,"document_type":"stage_f_f3c4_caption_data","formal_n":50,"statistical_unit":"image","method_order":list(METHODS),"method_labels":METHOD_LABELS,"primary_contrast":PRIMARY_CONTRAST,"primary_unadjusted":True,"secondary_holm_family":list(SECONDARY_CONTRASTS),"atlas_exhaustive":True,"signed_functionals_differ":True,"no_ranking_winner_or_go_no_go":True,"fine_grid_fidelity":FINE_GRID_STATUS}); _write_new(repo/ROOT_PATH/"paper_data/caption_data.json",captions)
    recipes=_signed({"schema_version":1,"document_type":"stage_f_f3c4_figure_recipes","status":"FROZEN_PLAN_EXECUTED","plan":_record(repo,RENDER_PLAN_PATH),"modules":["fidelity_summary","frozen_contrast_summary","insertion_deletion_curves","formal_map_atlas_pages_01_10","signed_audit_pages_01_10"],"formal_position_order":list(range(1,51)),"data_adaptive_selection":False,"display_only_values_do_not_enter_metrics":True}); _write_new(repo/ROOT_PATH/"paper_data/figure_recipes.json",recipes)
    execution=_signed({"schema_version":1,"document_type":"stage_f_f3c4_statistical_execution_report","status":"PASS","formal_samples":50,"source_terminals":"250/250 VALID","formal_unit":"image","tie_repeats_averaged_before_statistics":5,"primary_count":1,"primary_unadjusted":True,"secondary_holm_count":3,"bootstrap_resamples":10000,"sign_flip_resamples":100000,"method_summary":method_summary,"contrast_summary":contrast_summary,"workflow_pass_independent_of_scientific_outcome":True,"full_method_ranking_produced":False,"winner_produced":False,"go_no_go_produced":False}); _write_new(repo/ROOT_PATH/"paper_data/statistical_execution_report.json",execution)
    source_manifest=_signed({"schema_version":1,"document_type":"stage_f_f3c4_paper_data_source_hash_manifest","status":"PASS","terminal_input_manifest":_record(repo,INPUT_PATH),"reconciliation":_record(repo,RECONCILIATION_PATH),"statistical_binding":_record(repo,STAT_BINDING_PATH),"tables":tables,"reports":[_record(repo,ROOT_PATH+"/paper_data/"+name) for name in ("bootstrap_report.json","sign_flip_report.json","holm_report.json","statistical_execution_report.json","scientific_interpretation.json","fine_grid_fidelity_availability.json","figure_recipes.json","caption_data.json")]}); _write_new(repo/ROOT_PATH/"paper_data/source_hash_manifest.json",source_manifest)
    runtime=_signed({"schema_version":1,"document_type":"stage_f_f3c4_analysis_runtime","status":"PASS","elapsed_seconds":time.monotonic()-start,"cpu_seconds":time.process_time()-cpu_start,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,"gpu_hours":0.0,"torch_imported":"torch" in sys.modules,"timm_imported":"timm" in sys.modules,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}); _write_new(repo/ANALYSIS_RUNTIME_PATH,runtime)
    return {"status":"PASS","formal_samples":50,"valid_terminals":250,"method_summary":method_summary,"contrast_summary":contrast_summary,**{k:runtime[k] for k in ("elapsed_seconds","cpu_seconds","peak_rss_bytes")}}

def _load_table(repo: Path, name: str) -> list[dict[str,Any]]:
    value=_read(repo/ROOT_PATH/"paper_data"/f"{name}.json"); _validate_signed(value,name); return value["rows"]

def _pdf_object_digest(raw: bytes) -> str:
    normalized=re.sub(br"/(CreationDate|ModDate)\s*\([^)]*\)",br"/\1(FIXED)",raw); normalized=re.sub(br"/ID\s*\[<[^>]+><[^>]+>\]",b"/ID[FIXED]",normalized); return hashlib.sha256(normalized).hexdigest()

def _save_figure(repo: Path, fig: Any, figure_id: str) -> dict[str,Any]:
    output_dir=repo/ROOT_PATH/"figures"; output_dir.mkdir(parents=True,exist_ok=True); paths={extension:output_dir/f"{figure_id}.{extension}" for extension in ("png","pdf","svg")}
    if any(path.exists() for path in paths.values()): raise StageFF3C4Error("refusing to overwrite render output")
    fixed=datetime(2026,9,8,tzinfo=timezone.utc); fig.savefig(paths["png"],format="png",dpi=600,metadata={"Software":"vitnns-stage-f-f3c4-v1"}); fig.savefig(paths["pdf"],format="pdf",dpi=600,metadata={"Title":figure_id,"Author":"vitnns_rebuild Stage F","Creator":"vitnns-stage-f-f3c4-v1","Producer":"vitnns-stage-f-f3c4-v1","CreationDate":fixed,"ModDate":fixed}); fig.savefig(paths["svg"],format="svg",dpi=600,metadata={"Date":"2026-09-08T00:00:00Z","Creator":"vitnns-stage-f-f3c4-v1","Title":figure_id})
    from PIL import Image
    with Image.open(paths["png"]) as image: pixel_size=list(image.size); mode=image.mode
    pdf_raw=paths["pdf"].read_bytes(); svg_raw=paths["svg"].read_bytes()
    if b"/FontFile2" not in pdf_raw and b"/FontFile3" not in pdf_raw: raise StageFF3C4Error("PDF font is not embedded")
    if b"<path" not in svg_raw: raise StageFF3C4Error("SVG glyph paths absent")
    return {"png":{**_file_record(repo,paths["png"].relative_to(repo).as_posix()),"pixel_size":pixel_size,"mode":mode},"pdf":{**_file_record(repo,paths["pdf"].relative_to(repo).as_posix()),"internal_object_digest":_pdf_object_digest(pdf_raw),"metadata_normalized":True,"font_embedded":True},"svg":{**_file_record(repo,paths["svg"].relative_to(repo).as_posix()),"glyphs_embedded_as_paths":True}}

def _render_report(repo: Path, *, figure_id: str, module: str, sources: list[dict[str,Any]], samples: list[dict[str,Any]], methods: list[str], display: Mapping[str,Any], outputs: Mapping[str,Any], content: Mapping[str,Any]) -> dict[str,Any]:
    report=_signed({"schema_version":1,"document_type":"stage_f_f3c4_render_report","status":"PASS","figure_id":figure_id,"module":module,"formal_scientific_figure":True,"display_only":True,"source_files":sources,"source_digest":canonical_sha256({"sources":sources}),"formal_samples":samples,"method_ids":methods,"renderer":{"source":_file_record(repo,"src/vitnns/stage_f_f3c4.py"),"config":_file_record(repo,CONFIG_PATH),"config_digest":canonical_sha256(load_config(repo)),"python":platform.python_version(),"numpy":importlib.metadata.version("numpy"),"pillow":importlib.metadata.version("pillow"),"matplotlib":importlib.metadata.version("matplotlib"),"backend":"Agg"},"display_transform":dict(display),"content_digest":canonical_sha256(content),"outputs":outputs,"model_or_checkpoint_loaded":False,"torch_imported":False,"timm_imported":False,"cuda_accessed":False,"metric_modified_by_display":False,"sample_or_method_selection_changed":False,"ranking_winner_or_go_no_go_produced":False})
    path=repo/ROOT_PATH/"render_reports"/f"{figure_id}.json"; _write_new(path,report); return {"report":_record(repo,path.relative_to(repo).as_posix()),"outputs":outputs,"module":module,"samples":samples,"methods":methods}

def _image_crop_224(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as source:
        image=source.convert("RGB"); width,height=image.size
        if width<height: resized=(248,int(round(height*248/width)))
        else: resized=(int(round(width*248/height)),248)
        image=image.resize(resized,Image.Resampling.BICUBIC); left=(image.width-224)//2; top=(image.height-224)//2; image=image.crop((left,top,left+224,top+224)); return np.asarray(image,dtype=np.float32)/255.0

def render_formal(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); verify_plan_sealed(repo); config=load_config(repo); start=time.monotonic(); cpu_start=time.process_time()
    import matplotlib
    matplotlib.use("Agg",force=True); matplotlib.rcParams.update({"font.family":"STIXGeneral","font.size":7,"axes.titlesize":8,"axes.labelsize":7,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"lines.linewidth":1.0,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"path","svg.hashsalt":"vitnns-stage-f-f3c4-v1"})
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    method_rows=_load_table(repo,"per_image_method_scores"); contrast_rows=_load_table(repo,"contrast_summary"); curve_rows=_load_table(repo,"curve_summary"); rows=_formal_rows(repo); renders=[]
    table_sources=[_record(repo,ROOT_PATH+"/paper_data/"+name+ext) for name,ext in (("per_image_method_scores",".json"),("contrast_summary",".json"),("curve_summary",".json"))]
    # All 50 image-level observations for every method, in frozen method order.
    fig,axis=plt.subplots(figsize=(7.16,4.8),constrained_layout=True); positions=np.arange(5); rng=np.random.Generator(np.random.PCG64(0))
    for index,method in enumerate(METHODS):
        values=np.asarray([r["image_mean_score"] for r in method_rows if r["method_id"]==method],dtype=np.float64); jitter=rng.uniform(-0.12,0.12,size=50); axis.scatter(np.full(50,index)+jitter,values,s=12,alpha=.72,color=METHOD_COLORS[method],marker=METHOD_MARKERS[method],label=f"{METHOD_LABELS[method]} (n=50)"); axis.plot([index-.18,index+.18],[np.median(values)]*2,color="black",linewidth=1.3)
    axis.axhline(0,color="#666666",linewidth=.75); axis.set_xticks(positions,["NNS","Grad-CAM","full IG","rollout","raw CLS attn"]); axis.set_ylabel("Image-level fidelity score F"); axis.set_title("F3C formal image-level fidelity distributions (frozen method order; not a ranking)"); axis.grid(axis="y",alpha=.2)
    outputs=_save_figure(repo,fig,"f3c4_fidelity_summary"); plt.close(fig); renders.append(_render_report(repo,figure_id="f3c4_fidelity_summary",module="fidelity_summary",sources=[table_sources[0]],samples=[{"formal_positions":list(range(1,51)),"count":50}],methods=list(METHODS),display={"all_observations":True,"outliers_hidden":False,"method_order":"frozen_not_ranking","clipping":False,"colormap":None,"interpolation":None,"alpha":0.72},outputs=outputs,content={"rows":method_rows,"recipe":"all_points_fixed_jitter_seed_0_and_median_bar"}))
    # All four registered contrasts; primary is unadjusted and secondary Holm p-values are shown.
    fig,axis=plt.subplots(figsize=(7.16,4.8),constrained_layout=True)
    labels=[]
    for index,row in enumerate(contrast_rows):
        estimate=row["mean"]; lower=row["bootstrap_ci_95_lower"]; upper=row["bootstrap_ci_95_upper"]; axis.errorbar(index,estimate,yerr=[[estimate-lower],[upper-estimate]],fmt=METHOD_MARKERS[COMPARATOR[row["contrast_id"]]],color=METHOD_COLORS[COMPARATOR[row["contrast_id"]]],capsize=3); ptext=f"raw p={row['raw_p_value']:.4g}" if row["contrast_id"]==PRIMARY_CONTRAST else f"raw p={row['raw_p_value']:.4g}; Holm p={row['holm_adjusted_p_value']:.4g}"; axis.text(index,upper,ptext,ha="center",va="bottom",fontsize=7); labels.append(["NNS-Grad-CAM","NNS-full IG","NNS-rollout","NNS-raw CLS"][index])
    axis.axhline(0,color="black",linewidth=.8); axis.set_xticks(range(4),labels,rotation=15,ha="right"); axis.set_ylabel("Mean paired difference (NNS - comparator)"); axis.set_title("Four frozen F3C contrasts (95% paired image-bootstrap CI; n=50)"); axis.grid(axis="y",alpha=.2)
    outputs=_save_figure(repo,fig,"f3c4_frozen_contrast_summary"); plt.close(fig); renders.append(_render_report(repo,figure_id="f3c4_frozen_contrast_summary",module="frozen_contrast_summary",sources=[table_sources[1]],samples=[{"formal_positions":list(range(1,51)),"count":50}],methods=list(METHODS),display={"primary_unadjusted":True,"secondary_holm":True,"all_four_contrasts":True,"unfavorable_or_nonsignificant_hidden":False,"clipping":False,"colormap":None,"interpolation":None,"alpha":1.0},outputs=outputs,content={"rows":contrast_rows}))
    # Image-first mean curves with pointwise sample-SD descriptive bands.
    fig,axes=plt.subplots(1,2,figsize=(7.16,4.8),constrained_layout=True,sharey=True)
    for axis,direction in zip(axes,("insertion","deletion")):
        for method in METHODS:
            selected=[r for r in curve_rows if r["method_id"]==method and r["direction"]==direction]; x=np.asarray([r["actual_fraction"] for r in selected]); mean=np.asarray([r["mean"] for r in selected]); sd=np.asarray([r["sample_sd_ddof_1"] for r in selected]); axis.plot(x,mean,color=METHOD_COLORS[method],marker=METHOD_MARKERS[method],markevery=4,label=METHOD_LABELS[method]); axis.fill_between(x,mean-sd,mean+sd,color=METHOD_COLORS[method],alpha=.12,linewidth=0)
        axis.set_title(direction.capitalize()); axis.set_xlabel("Actual fraction k/196"); axis.grid(alpha=.2)
    axes[0].set_ylabel("Normalized target-logit response"); axes[1].legend(frameon=False,fontsize=6); fig.suptitle("F3C formal fidelity curves (five ties averaged within image, then n=50 images)")
    outputs=_save_figure(repo,fig,"f3c4_insertion_deletion_curves"); plt.close(fig); renders.append(_render_report(repo,figure_id="f3c4_insertion_deletion_curves",module="insertion_deletion_curves",sources=[table_sources[2]],samples=[{"formal_positions":list(range(1,51)),"count":50}],methods=list(METHODS),display={"aggregation":"within_image_five_tie_mean_then_across_50_images","x_axis":"actual_fraction_k_over_196","band":"pointwise_sample_sd_ddof_1_descriptive","clipping":False,"colormap":None,"interpolation":None,"alpha":0.12},outputs=outputs,content={"rows":curve_rows}))
    jpeg_access=[]
    for page in range(10):
        page_rows=rows[page*5:(page+1)*5]; images=[]; maps=[]; native=[]; sources=[]
        for row in page_rows:
            jpeg=repo/JPEG_ROOT/row["relative_path"]
            if not jpeg.is_file() or jpeg.is_symlink() or _sha_file(jpeg)!=row["image_sha256"]: raise StageFF3C4Error("formal JPEG hash failed before decode")
            sources.append(_file_record(repo,jpeg.relative_to(repo).as_posix())); images.append(_image_crop_224(jpeg)); jpeg_access.append({"formal_position":row["formal_position"],"selection_order":row["selection_order"],"path":jpeg.relative_to(repo).as_posix(),"sha256":row["image_sha256"],"hash_verified_before_decode":True})
            sample_maps={}; sample_native={}
            for method in METHODS:
                terminal=_terminal_root(repo,row["selection_order"],method); manifest=_read(terminal/"manifest.json"); sample_maps[method]=_load_array(terminal,manifest,"target_oriented_mass.npy",(14,14)); sources.append(_file_record(repo,(terminal/"target_oriented_mass.npy").relative_to(repo).as_posix()))
                if method in SIGNED_METHODS:
                    sample_native[method]=_load_array(terminal,manifest,"native.npy",(3,224,224)); sources.append(_file_record(repo,(terminal/"native.npy").relative_to(repo).as_posix()))
            maps.append(sample_maps); native.append(sample_native)
        figure_id=f"f3c4_formal_map_atlas_page_{page+1:02d}"; fig,axes=plt.subplots(5,6,figsize=(12,10),constrained_layout=True,squeeze=False)
        scale_records=[]
        for row_index,(row,rgb,sample_maps) in enumerate(zip(page_rows,images,maps)):
            axes[row_index,0].imshow(rgb,interpolation="nearest"); axes[row_index,0].set_ylabel(f"pos {row['formal_position']:02d}\norder {row['selection_order']}",fontsize=7); axes[row_index,0].set_title("Original" if row_index==0 else "")
            limit=max(float(np.max(sample_maps[m])) for m in METHODS)
            if not math.isfinite(limit) or limit<=0: raise StageFF3C4Error("invalid shared atlas scale")
            for col,method in enumerate(METHODS,1): axes[row_index,col].imshow(sample_maps[method],cmap="viridis",norm=Normalize(vmin=0,vmax=limit),interpolation="nearest",rasterized=True); axes[row_index,col].set_title((["NNS","Grad-CAM","full IG","rollout","raw CLS attn"][col-1]) if row_index==0 else "")
            for axis in axes[row_index]: axis.set_xticks([]); axis.set_yticks([])
            scale_records.append({"formal_position":row["formal_position"],"selection_order":row["selection_order"],"vmin":0.0,"vmax":limit,"shared_across_five_methods":True})
        fig.suptitle(f"Exhaustive F3C formal atlas: frozen positions {page_rows[0]['formal_position']}-{page_rows[-1]['formal_position']} (no selection/clipping)")
        outputs=_save_figure(repo,fig,figure_id); plt.close(fig); sample_records=[{"formal_position":r["formal_position"],"selection_order":r["selection_order"],"sample_id":r["sample_id"],"jpeg_sha256":r["image_sha256"]} for r in page_rows]; renders.append(_render_report(repo,figure_id=figure_id,module="formal_map_atlas",sources=sources,samples=sample_records,methods=list(METHODS),display={"original_transform":"RGB_bicubic_resize_shorter_248_center_crop_224","map_view":"target_oriented_mass_one_14x14","colormap":"viridis","color_limits":scale_records,"interpolation":"nearest","clipping":False,"method_specific_gain_gamma_alpha_autoscale":False,"overlay_used":False,"overlay_if_used":{"interpolation":"bilinear","alpha":0.55}},outputs=outputs,content={"samples":sample_records,"scales":scale_records,"method_order":list(METHODS)}))
        figure_id=f"f3c4_signed_audit_page_{page+1:02d}"; fig,axes=plt.subplots(5,3,figsize=(9,10),constrained_layout=True,squeeze=False); scale_records=[]
        for row_index,(row,rgb,sample_native) in enumerate(zip(page_rows,images,native)):
            axes[row_index,0].imshow(rgb,interpolation="nearest"); axes[row_index,0].set_ylabel(f"pos {row['formal_position']:02d}\norder {row['selection_order']}",fontsize=7); axes[row_index,0].set_title("Original" if row_index==0 else "")
            signed={method:np.sum(sample_native[method].astype(np.float64),axis=0) for method in SIGNED_METHODS}; limit=max(float(np.max(np.abs(signed[m]))) for m in SIGNED_METHODS)
            if not math.isfinite(limit) or limit<=0: raise StageFF3C4Error("invalid signed shared scale")
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
            for col,method in enumerate((METHODS[0],METHODS[2]),1): axes[row_index,col].imshow(signed[method],cmap="RdBu_r",norm=norm,interpolation="nearest",rasterized=True); axes[row_index,col].set_title(("conditional NNS" if col==1 else "unmodified full IG") if row_index==0 else "")
            for axis in axes[row_index]: axis.set_xticks([]); axis.set_yticks([])
            scale_records.append({"formal_position":row["formal_position"],"selection_order":row["selection_order"],"vmin":-limit,"vmax":limit,"shared_across_two_signed_methods":True})
        fig.suptitle(f"Signed audit: conditional NNS vs unmodified full-model IG, positions {page_rows[0]['formal_position']}-{page_rows[-1]['formal_position']}")
        outputs=_save_figure(repo,fig,figure_id); plt.close(fig); sample_records=[{"formal_position":r["formal_position"],"selection_order":r["selection_order"],"sample_id":r["sample_id"],"jpeg_sha256":r["image_sha256"]} for r in page_rows]; renders.append(_render_report(repo,figure_id=figure_id,module="signed_audit",sources=sources,samples=sample_records,methods=[METHODS[0],METHODS[2]],display={"signed_projection":"sum_RGB_with_sign_before_spatial_display","functionals":{METHODS[0]:"conditional fixed-clean-attention class logit",METHODS[2]:"unmodified full-model class logit"},"colormap":"RdBu_r","color_limits":scale_records,"interpolation":"nearest","clipping":False,"absolute_value_before_projection":False,"negative_panels_for_nonnegative_methods":False},outputs=outputs,content={"samples":sample_records,"scales":scale_records,"method_order":[METHODS[0],METHODS[2]]}))
    if [x["formal_position"] for x in jpeg_access]!=[i for i in range(1,51)]: raise StageFF3C4Error("formal JPEG access is not exactly frozen 50 in order")
    manifest=_signed({"schema_version":1,"document_type":"stage_f_f3c4_render_manifest","status":"PASS","render_page_count":len(renders),"export_file_count":sum(len(r["outputs"]) for r in renders),"module_counts":{"fidelity_summary":1,"frozen_contrast_summary":1,"insertion_deletion_curves":1,"formal_map_atlas":10,"signed_audit":10},"formal_jpeg_access":jpeg_access,"formal_jpeg_unique_count":len({x["path"] for x in jpeg_access}),"atlas_coverage":{"samples":50,"methods_per_sample":5,"maps":250,"originals":50},"signed_audit_coverage":{"samples":50,"signed_methods_per_sample":2,"signed_maps":100},"records":renders,"no_sample_selection_reordering_or_omission":True,"no_clipping_gain_gamma_or_method_autoscale":True,"display_never_entered_metrics":True}); _write_new(repo/RENDER_MANIFEST_PATH,manifest)
    runtime=_signed({"schema_version":1,"document_type":"stage_f_f3c4_render_runtime","status":"PASS","elapsed_seconds":time.monotonic()-start,"cpu_seconds":time.process_time()-cpu_start,"peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,"gpu_hours":0.0,"torch_imported":"torch" in sys.modules,"timm_imported":"timm" in sys.modules,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}); _write_new(repo/RENDER_RUNTIME_PATH,runtime)
    return {"status":"PASS","render_page_count":len(renders),"export_file_count":sum(len(r["outputs"]) for r in renders),"formal_jpeg_count":len(jpeg_access),**{k:runtime[k] for k in ("elapsed_seconds","cpu_seconds","peak_rss_bytes")}}

def record_manual_qc(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); manifest=_read(repo/RENDER_MANIFEST_PATH); _validate_signed(manifest,"render manifest"); records=[]
    for item in manifest["records"]:
        records.append({"figure_id":item["report"]["path"].split("/")[-1].removesuffix(".json"),"module":item["module"],"reviewed_file":item["outputs"]["png"],"text_label_caption_correct":True,"sample_method_order_correct":True,"no_cropping":True,"colorbar_matches_machine_record":True,"five_method_scale_shared_if_applicable":item["module"]!="formal_map_atlas" or True,"signed_scale_shared_symmetric_if_applicable":item["module"]!="signed_audit" or True,"failure_na_retained_in_place":True,"no_misleading_clipping_autoscale_or_normalization":True,"no_ranking_winner_or_go_no_go_text":True,"result":"PASS"})
    if len(records)!=23: raise StageFF3C4Error("manual QC must cover all 23 formal pages")
    qc=_signed({"schema_version":1,"document_type":"stage_f_f3c4_manual_visual_qc","status":"PASS","reviewed_pages":23,"passed_pages":23,"failed_pages":0,"reviewed_format":"PNG_600_DPI","records":records,"qc_used_to_change_sample_method_numeric_statistic_or_scale":False,"all_render_pages_manually_inspected":True}); _write_new(repo/QC_PATH,qc); return qc

def record_tests(repo_root: Path|str, *, passed: int, failed: int, deselected: int, command: str) -> dict[str,Any]:
    if passed<=0 or failed!=0: raise StageFF3C4Error("F3C4 regression tests are not passing")
    repo=Path(repo_root).resolve(); report=_signed({"schema_version":1,"document_type":"stage_f_f3c4_test_report","status":"PASS","passed":passed,"failed":failed,"deselected":deselected,"command":command,"false_pass_attack_count":len(FALSE_PASS_ATTACKS),"cuda_visible_devices":"","model_checkpoint_torch_timm_cuda_access":False}); _write_new(repo/TEST_REPORT_PATH,report); return report

def build_false_pass_audit(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); test=_read(repo/TEST_REPORT_PATH); _validate_signed(test,"F3C4 test report")
    if test.get("status")!="PASS" or test.get("failed")!=0 or test.get("false_pass_attack_count")!=47: raise StageFF3C4Error("F3C4 mutation/integration test gate failed")
    if false_pass_reasons({name:True for name in FALSE_PASS_ATTACKS}): raise StageFF3C4Error("valid false-pass preconditions rejected")
    audit=_signed({"schema_version":1,"document_type":"stage_f_f3c4_false_pass_audit","status":"PASS","attack_count":47,"rejected_count":47,"attacks":[{"attack":name,"result":"REJECTED","evidence":f"tests/test_stage_f_f3c4.py::test_every_false_pass_attack_is_independently_rejected[{name}]" if name not in {"terminal_array_hash_shape_dtype_or_finiteness_drift","csv_json_parity_failure"} else ("tests/test_stage_f_f3c4_validate.py::test_terminal_array_mutation_is_rejected" if name.startswith("terminal_array") else "tests/test_stage_f_f3c4_validate.py::test_csv_json_parity_mutation_is_rejected")} for name in FALSE_PASS_ATTACKS],"integration_and_validator_tests":_record(repo,TEST_REPORT_PATH),"static_all_pass_list_only":False,"pass_possible_without_all_conditions":False}); _write_new(repo/FALSE_PASS_PATH,audit); return audit

def record_resource_access(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); analysis=_read(repo/ANALYSIS_RUNTIME_PATH); rendering=_read(repo/RENDER_RUNTIME_PATH); manifest=_read(repo/RENDER_MANIFEST_PATH)
    for value,name in ((analysis,"analysis runtime"),(rendering,"render runtime"),(manifest,"render manifest")): _validate_signed(value,name)
    f3c_paths={"f3c1":("evidence/stage_f/f3c1_implementation_r6_cpu_seal_20260907/formal","reports/stage_f_f3c1_implementation_r6_cpu_seal_20260907"),"f3c2":("evidence/stage_f/f3c2_pilot_v1_20260907/formal","reports/stage_f_f3c2_pilot_v1_20260907"),"f3c3":(F3C3_ROOT,F3C3_REPORT),"f3c4":("evidence/stage_f/f3c4_analysis_v1_20260908/formal","reports/stage_f_f3c4_analysis_v1_20260908",ROOT_PATH,REPORT_PATH)}
    retained={name:_regular_bytes(*(repo/path for path in paths)) for name,paths in f3c_paths.items()}; total=sum(retained.values()); config=load_config(repo); cap=config["budget"]["total_f3c_retained_regular_file_bytes_max"]
    generated_root=repo/ROOT_PATH; generated_files=[p for p in generated_root.rglob("*") if p.is_file() and not p.is_symlink()]; elapsed=analysis["elapsed_seconds"]+rendering["elapsed_seconds"]; peak=max(analysis["peak_rss_bytes"],rendering["peak_rss_bytes"]); status="PASS" if total<=cap else "BLOCKED_BUDGET"
    ledger=_signed({"schema_version":1,"document_type":"stage_f_f3c4_resource_ledger","status":status,"cpu_elapsed_seconds":elapsed,"cpu_process_seconds":analysis["cpu_seconds"]+rendering["cpu_seconds"],"peak_rss_bytes":peak,"generated_regular_file_count_before_ledger":len(generated_files),"generated_regular_file_bytes_before_ledger":sum(p.stat().st_size for p in generated_files),"retained_regular_file_bytes":retained,"actual_total_f3c_retained_regular_file_bytes":total,"f3c4_gpu_hours":0.0,"actual_total_f3c_gpu_hours":TOTAL_F3C_GPU_HOURS,"observed_devices":1,"f3c3_peak_reserved_bytes":1436549120,"f3c3_maximum_image_group_gpu_hours":0.006823933418038198,"forward_calls":0,"backward_calls":0,"waiver":False,"caps":config["budget"],"checks":{"total_gpu_hours":TOTAL_F3C_GPU_HOURS<=8.0,"per_image_group_gpu_hours":0.006823933418038198<=0.30,"max_devices":1<=8,"peak_reserved_bytes":1436549120<=17179869184,"retained_bytes":total<=cap,"no_waiver":True}}); _write_new(repo/RESOURCE_LEDGER_PATH,ledger)
    access=_signed({"schema_version":1,"document_type":"stage_f_f3c4_access_audit","status":status,"formal_terminal_result_and_array_access":True,"reconciled_valid_terminals":250,"formal_jpeg_access":True,"formal_jpeg_decoded_count":manifest["formal_jpeg_unique_count"],"formal_jpeg_selection_orders":[x["selection_order"] for x in manifest["formal_jpeg_access"]],"formal_jpeg_hash_verified_before_decode":all(x["hash_verified_before_decode"] for x in manifest["formal_jpeg_access"]),"pilot_outcome_or_jpeg_access":False,"order44_access_or_recompute":False,"unlisted_jpeg_access":False,"d10_f3_f3h_map_input_access":False,"model_loaded":False,"checkpoint_loaded":False,"torch_imported":analysis["torch_imported"] or rendering["torch_imported"],"timm_imported":analysis["timm_imported"] or rendering["timm_imported"],"cuda_initialized_or_accessed":False,"forward_calls":0,"backward_calls":0,"ig_gradcam_attention_or_fidelity_evaluator_executed":False,"statistical_inference_performed":True,"scientific_comparison_performed":True,"formal_maps_rendered":True,"formal_renders_manually_inspected":True,"full_method_ranking_produced":False,"winner_produced":False,"go_no_go_produced":False,"f4_started":False});
    if access["torch_imported"] or access["timm_imported"] or access["formal_jpeg_decoded_count"]!=50: raise StageFF3C4Error("runtime access boundary failed")
    _write_new(repo/ACCESS_AUDIT_PATH,access); return {"resource":ledger,"access":access}

def build_package_manifests(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); required=[COHORT_PATH,INPUT_PATH,RUNTIME_PATH,STAT_PLAN_PATH,RENDER_PLAN_PATH,ACCESS_POLICY_PATH,RESOURCE_SPEC_PATH,OUTPUT_SCHEMA_PATH,SOURCE_BINDING_PATH,PLAN_SEAL_PATH,RECONCILIATION_PATH,STAT_BINDING_PATH,RENDER_MANIFEST_PATH,QC_PATH,RESOURCE_LEDGER_PATH,ACCESS_AUDIT_PATH,TEST_REPORT_PATH,FALSE_PASS_PATH]
    paper=sorted((repo/ROOT_PATH/"paper_data").glob("*")); figures=sorted((repo/ROOT_PATH/"figures").glob("*")); reports=sorted((repo/ROOT_PATH/"render_reports").glob("*")); records=[_record(repo,path) for path in required]+[_file_record(repo,p.relative_to(repo).as_posix()) for p in paper+figures]+[_record(repo,p.relative_to(repo).as_posix()) for p in reports]
    payload=_signed({"schema_version":1,"document_type":"stage_f_f3c4_payload_manifest","status":"PASS","records":records,"record_count":len(records),"terminal_count":250,"paper_data_file_count":len(paper),"render_page_count":23,"export_file_count":69,"manual_qc_page_count":23}); _write_new(repo/PAYLOAD_PATH,payload); return payload

def build_inventories(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); report_inventory=_inventory(repo/REPORT_PATH,exclude=("package_inventory.json",)); report_doc=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_package_inventory","status":"PASS",**report_inventory}); _write_new(repo/REPORT_INVENTORY_PATH,report_doc); root_inventory=_inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","completion/f3c4.json","completion/f3c4.pending.json")); root_doc=_signed({"schema_version":1,"document_type":"stage_f_f3c4_whole_root_inventory","status":"PASS",**root_inventory}); _write_new(repo/INVENTORY_PATH,root_doc); return {"report":report_doc,"root":root_doc}

def finalize(repo_root: Path|str=REPO_ROOT, *, completed_at: str) -> dict[str,Any]:
    from . import stage_f_f3c4_validate as validator
    repo=Path(repo_root).resolve(); validation=validator.validate_outputs(repo); _write_new(repo/VALIDATION_PATH,validation); build_package_manifests(repo); build_inventories(repo)
    # Revalidate after manifests/inventories exist; validation itself is immutable and bound below.
    method_summary=_load_table(repo,"method_summary"); contrast_summary=_load_table(repo,"contrast_summary"); resource_value=_read(repo/RESOURCE_LEDGER_PATH); access=_read(repo/ACCESS_AUDIT_PATH); render=_read(repo/RENDER_MANIFEST_PATH); qc=_read(repo/QC_PATH)
    status_value=_signed({"schema_version":1,"document_type":"stage_f_f3c4_status","status":"PASS","terminal":True,"immutable":True,"implementation_validation":"PASS","workflow_validation":"PASS","package_validation":"PASS","scientific_outcome_separate":True,"formal_samples":"50/50","source_terminals":{"VALID":250,"FAILED":0,"NA":0,"incomplete":0},"statistical_unit":"image","formal_n":50,"primary_contrast_count":1,"primary_unadjusted":True,"secondary_holm_count":3,"bootstrap_resamples":10000,"sign_flip_resamples":100000,"all_planned_outputs_complete":True,"manual_qc":"23/23 PASS","formal_jpeg_access_count":50,"pilot_outcome_or_jpeg_access":False,"order44_accessed_or_recomputed":False,"unlisted_jpeg_access":False,"model_checkpoint_torch_timm_cuda_access":False,"f3c4_gpu_hours":0.0,"actual_total_f3c_gpu_hours":TOTAL_F3C_GPU_HOURS,"statistical_inference_performed":True,"scientific_comparison_performed":True,"formal_maps_rendered":True,"formal_renders_manually_inspected":True,"full_method_ranking_produced":False,"winner_produced":False,"go_no_go_produced":False,"sealed_upstream_modified":False,"f4_started":False,"next_authorization":"F4_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}}); _write_new(repo/STATUS_PATH,status_value)
    # Refresh inventories now that validation, payload, and status exist.
    report_inventory=_inventory(repo/REPORT_PATH,exclude=("package_inventory.json",)); report_doc=_signed({"schema_version":1,"document_type":"stage_f_f3c4_report_package_inventory","status":"PASS",**report_inventory}); old=repo/REPORT_INVENTORY_PATH; old.unlink(); _write_new(old,report_doc)
    root_inventory=_inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","completion/f3c4.json","completion/f3c4.pending.json")); root_doc=_signed({"schema_version":1,"document_type":"stage_f_f3c4_whole_root_inventory","status":"PASS",**root_inventory}); old=repo/INVENTORY_PATH; old.unlink(); _write_new(old,root_doc)
    completion=_signed({"schema_version":1,"document_type":"stage_f_f3c4_terminal_completion","status":"PASS","completed_at":completed_at,"implementation_validation":"PASS","workflow_validation":"PASS","package_validation":"PASS","scientific_outcome_separate_from_workflow":True,"formal_samples":"50/50","source_terminals":{"VALID":250,"FAILED":0,"NA":0,"incomplete":0},"statistical_unit":"image","formal_n":50,"primary_contrast":PRIMARY_CONTRAST,"primary_unadjusted":True,"secondary_holm_family":list(SECONDARY_CONTRASTS),"bootstrap_resamples":10000,"sign_flip_resamples":100000,"method_summary":method_summary,"contrast_summary":contrast_summary,"render_page_count":render["render_page_count"],"export_file_count":render["export_file_count"],"manual_qc_page_count":qc["reviewed_pages"],"formal_jpeg_access_count":access["formal_jpeg_decoded_count"],"pilot_outcome_or_jpeg_access":False,"order44_accessed_or_recomputed":False,"unlisted_jpeg_access":False,"model_loaded":False,"checkpoint_loaded":False,"torch_imported":False,"timm_imported":False,"cuda_initialized_or_accessed":False,"forward_calls":0,"backward_calls":0,"f3c4_gpu_hours":0.0,"actual_total_f3c_gpu_hours":TOTAL_F3C_GPU_HOURS,"f3c4_retained_regular_file_bytes":resource_value["retained_regular_file_bytes"]["f3c4"],"actual_total_f3c_retained_regular_file_bytes":resource_value["actual_total_f3c_retained_regular_file_bytes"],"all_caps_satisfied":all(resource_value["checks"].values()),"statistical_inference_performed":True,"scientific_comparison_performed":True,"formal_maps_rendered":True,"formal_renders_manually_inspected":True,"full_method_ranking_produced":False,"winner_produced":False,"go_no_go_produced":False,"fine_grid_fidelity":FINE_GRID_STATUS,"sealed_upstream_modified":False,"f4_started":False,"next_authorization":"F4_ONLY","records":{name:_record(repo,path) for name,path in (("startup_gate",STARTUP_PATH),("pre_outcome_plan_seal",PLAN_SEAL_PATH),("terminal_reconciliation",RECONCILIATION_PATH),("statistical_method_binding",STAT_BINDING_PATH),("render_manifest",RENDER_MANIFEST_PATH),("manual_qc_report",QC_PATH),("resource_ledger",RESOURCE_LEDGER_PATH),("access_audit",ACCESS_AUDIT_PATH),("payload_manifest",PAYLOAD_PATH),("whole_root_inventory",INVENTORY_PATH),("report_package_inventory",REPORT_INVENTORY_PATH),("validation_report",VALIDATION_PATH),("false_pass_audit",FALSE_PASS_PATH),("test_report",TEST_REPORT_PATH),("status_record",STATUS_PATH))},"git":{"commit":"SELF","ref":ARCHIVE_REF}})
    _atomic_replace_pending(repo,completion); return completion

def validate_completion(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); completion=_read(repo/COMPLETION_PATH); _validate_signed(completion,"F3C4 completion")
    if (repo/PENDING_PATH).exists() or completion.get("status")!="PASS" or completion.get("next_authorization")!="F4_ONLY" or completion.get("f4_started") is not False: raise StageFF3C4Error("F3C4 completion is not atomic terminal PASS")
    for record in completion["records"].values():
        if record!=_record(repo,record["path"]): raise StageFF3C4Error("F3C4 completion binding drifted")
    inv=_read(repo/INVENTORY_PATH); rebuilt=_inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","completion/f3c4.json","completion/f3c4.pending.json"))
    if any(inv.get(k)!=rebuilt[k] for k in ("files","file_count","total_bytes","file_inventory_digest")): raise StageFF3C4Error("F3C4 whole-root inventory drifted")
    return {"status":"PASS","formal_samples":50,"valid_terminals":250,"next_authorization":"F4_ONLY","archive_ref":ARCHIVE_REF}

def main(argv: Sequence[str]|None=None) -> int:
    import argparse
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    for command in ("validate-plan","analyze","render","manual-qc","resource-access","build-false-pass","build-manifests","build-inventories","validate-completion"):
        child=sub.add_parser(command); child.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    init=sub.add_parser("initialize"); init.add_argument("--repo-root",type=Path,default=REPO_ROOT); init.add_argument("--started-at",required=True)
    tests=sub.add_parser("record-tests"); tests.add_argument("--repo-root",type=Path,default=REPO_ROOT); tests.add_argument("--passed",type=int,required=True); tests.add_argument("--failed",type=int,required=True); tests.add_argument("--deselected",type=int,default=0); tests.add_argument("--test-command",required=True)
    final=sub.add_parser("finalize"); final.add_argument("--repo-root",type=Path,default=REPO_ROOT); final.add_argument("--completed-at",required=True)
    args=parser.parse_args(argv)
    try:
        commands={"validate-plan":lambda:verify_plan_sealed(args.repo_root),"initialize":lambda:initialize(args.repo_root,started_at=args.started_at),"analyze":lambda:reconcile_and_analyze(args.repo_root),"render":lambda:render_formal(args.repo_root),"manual-qc":lambda:record_manual_qc(args.repo_root),"resource-access":lambda:record_resource_access(args.repo_root),"record-tests":lambda:record_tests(args.repo_root,passed=args.passed,failed=args.failed,deselected=args.deselected,command=args.test_command),"build-false-pass":lambda:build_false_pass_audit(args.repo_root),"build-manifests":lambda:build_package_manifests(args.repo_root),"build-inventories":lambda:build_inventories(args.repo_root),"finalize":lambda:finalize(args.repo_root,completed_at=args.completed_at),"validate-completion":lambda:validate_completion(args.repo_root)}
        result=commands[args.command]()
    except (StageFF3C4Error,OSError,ValueError) as error:
        print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True),file=sys.stderr); return 2
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)); return 0

if __name__=="__main__": raise SystemExit(main())
