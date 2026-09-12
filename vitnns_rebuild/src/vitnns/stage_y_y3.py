"""Frozen Stage Y3 perturbation production contracts and helpers.

This module is deliberately Torch-free at import time.  GPU/model code lives in
``stage_y_y3_worker`` and is imported only after the launcher has sealed a
deterministic single-device environment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from . import stage_y_y2_r2 as r2
from . import stage_y_y2_r2_r2 as y2c
from .stage_y_y2_r2_schema import science_schema as y2_science_schema

PROTOCOL_ID = "STAGE_Y_VIT_BLOCK_LEVEL_NNS_MECHANISM_V1_20260910"
STAGE_ID = "Y3"
TOKEN = "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY"
NEXT = "STAGE_Y4_CPU_ONLY_MECHANISM_ANALYSIS_AND_DECISION"
BASE = "f679987a517412dc54fcde4a0d5820241ca7c6c8"
BRANCH = "stage-d-d3-pass"
PYTHON = Path("/data01/home/yeziqi.831/nxtrec_wksp/testcode/.conda/envs/vitnns/bin/python")
ROOT = Path("evidence/stage_y/y3_r4_perturbation_formal_20260912/formal")
REPORT_ROOT = Path("reports/stage_y_y3_r4_20260912")
SPEC = Path("config/stage_y/y3_run_spec.json")
SPEC_SCHEMA = Path("config/stage_y/y3_run_spec.schema.json")
ARTIFACT_SCHEMA = Path("config/stage_y/y3_artifact.schema.json")
DOC = Path("docs/stage_y_y3_perturbation_production_20260912.md")
SUBSET = Path("evidence/stage_y/y2_clean_formal_20260910/formal/perturbation_subset_registry.json")
TRANSFORMS = Path("manifests/stage_y/transform_registry.json")
Y2_COMPLETION = y2c.ROOT / "completion/y2_r2_r2.json"
ORIGINAL_PRODUCER_REF = "refs/archive/stage-y-y3-perturbation-mechanism-producer"
ORIGINAL_PRODUCER_COMMIT = "d5e96c0e0c55e685d38f182e63f3227dd8c1cbb9"
R1_PRODUCER_REF = "refs/archive/stage-y-y3-r1-perturbation-mechanism-producer"
R1_PRODUCER_COMMIT = "63fc7a4fb38a7ead26c23e5de9bfd36f7d512fea"
R2_PRODUCER_REF = "refs/archive/stage-y-y3-r2-perturbation-mechanism-producer"
R2_PRODUCER_COMMIT = "10420ab5df70a2b7bfdec0c0208c11c1f9b8774c"
R2_BLOCKED_REF = "refs/archive/stage-y-y3-r2-perturbation-mechanism-evidence-blocked"
R2_BLOCKED_COMMIT = "b8f0e6e285845c5e3b755b4d59cc1bc67adf2976"
R3_PRODUCER_REF = "refs/archive/stage-y-y3-r3-perturbation-mechanism-producer"
R3_PRODUCER_COMMIT = "2518c848db05d0eedc73993b3a4f789dc554261c"
R3_BLOCKED_REF = "refs/archive/stage-y-y3-r3-perturbation-mechanism-evidence-blocked"
R3_BLOCKED_COMMIT = "b75a326168974420c056819c713f422dc2f37d7a"
STARTUP_COMMIT = R3_BLOCKED_COMMIT
R3_ROOT = Path("evidence/stage_y/y3_r3_perturbation_formal_20260912/formal")
R3_REPORT_ROOT = Path("reports/stage_y_y3_r3_20260912")
INHERITED_INDICES = (0,2)
INTERRUPTED_INDICES = (1,3,5)
R3_INTERRUPTED_ELAPSED_UPPER_BOUND_SECONDS = 1882.0
R3_INTERRUPTED_GPU_RESERVE_SECONDS = 3 * R3_INTERRUPTED_ELAPSED_UPPER_BOUND_SECONDS
R3_INTERRUPTED_CPU_RESERVE_SECONDS = 4 * R3_INTERRUPTED_GPU_RESERVE_SECONDS
PRODUCER_REF = "refs/archive/stage-y-y3-r4-perturbation-mechanism-producer"
PASS_REF = "refs/archive/stage-y-y3-r4-perturbation-mechanism-evidence"
BLOCKED_REF = "refs/archive/stage-y-y3-r4-perturbation-mechanism-evidence-blocked"
IMPLEMENTATION_SEAL = "producer_implementation_seal.json"
CONDITIONS = ("horizontal_flip","translate_x_-8","translate_x_8","rotate_-10","rotate_10","brightness_0.9","brightness_1.1","gaussian_noise_0.01_seed_0")
SELECTION_ORDER = (17,13,48,75,26,24,74,34,93,69,68,71,84,77,35,72,43,20,15,41)
COMPUTED_ORDERS = (77,84,93)
AUDIT_OVERLAP = (17,)
EXPECTED_UNITS = 160
CORE_ARRAY_NAMES = r2.EXPECTED_ARRAY_NAMES
EXTRA_ARRAY_NAMES = ("valid_mask","fully_valid_patch_mask","patch_inverse_operator")
AUDIT_SIGNED_NAMES = ("input_cell_map",) + tuple(name for name in CORE_ARRAY_NAMES if name.endswith("_cell"))
AUDIT_ARRAY_NAMES = ("positive_input_map","negative_input_map") + tuple(prefix+name for name in AUDIT_SIGNED_NAMES for prefix in ("positive_","negative_")) + tuple(f"raw_attention_b{b:02d}" for b in range(12))
TERMINAL_STATES = ("VALID","FAILED_NUMERICAL","NA","BLOCKED_STARTUP_STATE","BLOCKED_RESOURCE_SCHEDULE","BLOCKED_BUDGET","BLOCKED_INFRASTRUCTURE","BLOCKED_PROVENANCE","BLOCKED_ACCESS","BLOCKED_SCIENTIFIC_VALIDITY")
BLOCKING_STATES = TERMINAL_STATES[3:]
CAPS = {"h20_gpu_hours":102.4,"cpu_hours":349.0,"wall_hours":103.0,"peak_reserved_gpu_gib_per_process":21.0,"retained_storage_gib":9.0,"download_bytes":0}
EXPECTED_REFS = {
    "refs/archive/stage-y-y0-protocol-freeze":"a6efa1426902708d83de1d83ef257d524cacf95b",
    "refs/archive/stage-y-y1-r1-resource-telemetry-waiver":"70f70ca7290316531406791a6533d65590424a24",
    "refs/archive/stage-y-y2-p0-parallel-schedule-freeze":"66a0dfb889fdb0d45d9e566ed422ba9a40bd30a1",
    "refs/archive/stage-y-y2-p0-r1-elastic-schedule-freeze":"e6321a17a416b693f02f12bc0c7c329c21db5307",
    "refs/archive/stage-y-y2-clean-cohort-evidence-blocked":"1a621724b32b208036fb5f2a3993a48697e933c8",
    "refs/archive/stage-y-y2-r1-clean-cohort-evidence-blocked":"7cf9b1f9a25695821b19e98cff4e2edbc369891a",
    "refs/archive/stage-y-y2-r2-resource-reporting-recovery-producer":"d0023fa67d494e73422fd0e7df767f768a3c942a",
    "refs/archive/stage-y-y2-r2-resource-reporting-recovery-blocked":"1fd71d1b3e4d4c10fd3c65374169c734b64b8ff3",
    "refs/archive/stage-y-y2-r2-r1-resource-reporting-recovery-producer":"ce56ee5bdb92910d3ed4b0e8a0bbf6a1093c9e8b",
    "refs/archive/stage-y-y2-r2-r1-resource-reporting-recovery-blocked":"dbb63a2d2c96243f05af5abdeffa45045590f58a",
    "refs/archive/stage-y-y2-r2-r2-resource-ledger-recovery-producer":"795f2bcc3a08f872b0e865b544db285b1983c9a5",
    "refs/archive/stage-y-y2-r2-r2-clean-cohort-evidence-completion":BASE,
}
OLD_PASS = "refs/archive/stage-y-y2-r2-r1-clean-cohort-evidence-completion"
PRODUCER_FILES = (DOC,Path("docs/stage_y_y3_r3_worker_import_recovery_20260912.md"),Path("docs/stage_y_y3_r4_four_gpu_recovery_20260912.md"),SPEC,SPEC_SCHEMA,ARTIFACT_SCHEMA,Path("src/vitnns/stage_y_y3.py"),Path("src/vitnns/stage_y_y3_worker.py"),Path("scripts/run_stage_y_y3.py"),Path("scripts/validate_stage_y_y3.py"),Path("tests/test_stage_y_y3.py"))
DEPENDENCIES = tuple(dict.fromkeys(y2c.DEPENDENCIES + (Path("src/vitnns/stage_y_y2_r2_r2.py"),Path("manifests/stage_y/transform_registry.json"),Path("manifests/stage_y/retention_plan.json"),Path("manifests/stage_y/schema_registry.json"),Path("manifests/stage_y/component_cut_registry.json"),Path("manifests/stage_y/metric_registry.json"))))

class Y3Error(RuntimeError): code="BLOCKED_PROVENANCE"
class StartupError(Y3Error): code="BLOCKED_STARTUP_STATE"
class ScheduleError(Y3Error): code="BLOCKED_RESOURCE_SCHEDULE"
class BudgetError(Y3Error): code="BLOCKED_BUDGET"
class InfrastructureError(Y3Error): code="BLOCKED_INFRASTRUCTURE"
class ProvenanceError(Y3Error): code="BLOCKED_PROVENANCE"
class AccessError(Y3Error): code="BLOCKED_ACCESS"
class ScientificValidityError(Y3Error): code="BLOCKED_SCIENTIFIC_VALIDITY"
class NumericalError(Y3Error): code="FAILED_NUMERICAL"

def require(flag: Any, message: str, error: type[Y3Error]=ProvenanceError) -> None:
    if not flag: raise error(message)

canonical = r2.canonical_digest
sha = r2.sha_file
read = r2.read_json

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def git(repo: Path, *args: str) -> str:
    return r2.y2.git(repo,*args)

def file_record(path: Path, repo: Path) -> dict[str,Any]:
    value=r2.file_record(path,repo)
    return value

def checked_file(repo: Path, value: Mapping[str,Any], parent: Path|None=None) -> Path:
    return r2.checked_record(repo,value,parent)

def obj(properties: Mapping[str,Any], optional: Sequence[str]=()) -> dict[str,Any]:
    return {"type":"object","additionalProperties":False,"properties":dict(properties),"required":[k for k in properties if k not in optional]}
def arr(item: Mapping[str,Any], count: int|None=None) -> dict[str,Any]:
    out={"type":"array","items":dict(item)}
    if count is not None: out.update(minItems=count,maxItems=count)
    return out
def nullable(item: Mapping[str,Any]) -> dict[str,Any]: return {"anyOf":[dict(item),{"type":"null"}]}
S={"type":"string"}; I={"type":"integer","minimum":0}; N={"type":"number"}; NN={"type":"number","minimum":0}; B={"type":"boolean"}; H={"type":"string","pattern":"^[0-9a-f]{64}$"}; G={"type":"string","pattern":"^[0-9a-f]{40}$"}
FILE=obj({"path":S,"bytes":I,"sha256":H,"canonical_digest":H},("canonical_digest",))
ARRAY=obj({"path":S,"bytes":I,"sha256":H,"canonical_digest":H,"dtype":{"enum":["float32","bool"]},"shape":arr(I),"axes":arr(S)})
CALLS=obj({k:I for k in r2.CALL_KEYS})
RESOURCE=obj({"gpu_active_seconds":NN,"cpu_seconds":NN,"wall_seconds":NN,"peak_allocated_gpu_bytes":I,"peak_reserved_gpu_bytes":I,"retained_bytes":I,"download_bytes":{"const":0},"calls":CALLS})
TARGET=obj({"class_id":{"type":"integer","minimum":0,"maximum":999},"rule":{"const":"clean predicted-class raw logit"}})
CHECK=obj({"attack":S,"expected":{"const":"REJECT"},"observed":{"const":"REJECT"},"pass":{"const":True}})

@lru_cache(None)
def artifact_schema() -> dict[str,Any]:
    clean=obj({"subset_rank":I,"selection_order":I,"source_object_id":S,"source_stage":{"enum":["Y2-R1","Y2-R2-R1"]},"target_identity":TARGET,"clean_target_logit":N,"clean_baseline_logit":N,"source_terminal":FILE,"source_scientific_result":FILE,"source_validation_digest":H,"validated_array_count":{"const":262},"status":{"const":"VALID"}})
    coordinate=obj({"unit_index":I,"subset_rank":I,"selection_order":I,"source_object_id":S,"condition_index":I,"condition_id":{"enum":list(CONDITIONS)},"expected_shard_owner":I,"clean_source_terminal":FILE,"fixed_target":TARGET})
    environment=r2.y2_schema.ENV if hasattr(r2,"y2_schema") else obj({"python_executable_realpath":S,"python_version":S,"torch_version":S,"torchvision_version":S,"timm_version":S,"cuda_version":S,"driver":S,"device_name":{"const":"NVIDIA H20"},"cublas_workspace_config":{"const":":4096:8"},"deterministic_algorithms_enabled":{"const":True},"deterministic_warn_only_enabled":{"const":False},"cudnn_benchmark":{"const":False},"cudnn_deterministic":{"const":True},"cuda_matmul_allow_tf32":{"const":False},"cudnn_allow_tf32":{"const":False},"checkpoint_sha256":H,"numeric_contract_digest":S})
    device=obj({"physical_index":I,"device_name":{"const":"NVIDIA H20"},"gpu_uuid":S,"pci_bus_id":S,"total_memory_mib":I,"free_memory_mib":I,"driver":S,"environment_identity":environment,"environment_digest":H,"numeric_probe_digest":H,"probe_operations":{"const":["torch.matmul_float32","torch.matmul_float64","backward","exact_repeat"]}})
    worker=obj({"physical_index":I,"cuda_visible_devices":S,"gpu_uuid":S,"pci_bus_id":S,"device_name":{"const":"NVIDIA H20"},"hostname":S,"pid":I,"environment_identity":environment,"environment_digest":H})
    attempt_record=obj({"steps":nullable(I),"all_contracts_pass":B,"input_sum":nullable(N),"input_delta":nullable(N),"input_abs_error":nullable(NN),"threshold":nullable(NN),"closure_status":nullable(S),"identity_max_abs_error":nullable(NN),"wall_seconds":NN,"array_digest":nullable(H),"failure_code":nullable(S),"failure_phase":nullable(S)})
    transform=obj({"condition_id":{"enum":list(CONDITIONS)},"registry_record":obj({"id":S,"kind":S,"forward_map":S,"inverse_map":S,"interpolation":S,"padding_fill":S,"full_domain":B,"control_role":S},("control_role",)),"adapter_metadata":obj({"condition":S,"inverse":S,"full_domain":B,"amount":N,"angle":N,"seed":I},("amount","angle","seed")),"source_coordinate_convention":S,"transformed_coordinate_convention":S,"valid_threshold":N,"noise_seed":nullable(I),"valid_domain_coverage":NN,"inverse_operator_contract":S})
    prediction=obj({"fixed_target":TARGET,"clean_target_logit":N,"clean_baseline_logit":N,"transformed_predicted_class":I,"transformed_fixed_target_logit":N,"baseline_target_logit":N,"transformed_target_delta":N,"prediction_changed":B,"target_reselected":{"const":False},"clean_forward_executed":{"const":False}})
    science=obj({"y2_compatible":y2_science_schema(),"transformed_prediction":prediction,"endpoint_parity_max_abs_error":NN,"endpoint_parity_pass":B,"transform":transform,"inverse_materialization":obj({"stored_duplicate_inverse_arrays":{"const":False},"operator_name":{"const":"patch_inverse_operator"},"matrix_formula":{"const":"W @ T_transformed @ W.T"},"cls_formula":{"const":"W @ C_transformed"},"recompute_dtype":{"const":"float64_then_float32_little_endian_c_contiguous"},"fully_valid_pairs_only":{"const":True}}),"raw_attention_role":{"const":"descriptive structural reference only"},"audit_overlap":B})
    variants={
      "pending":obj({"started_unix_ns":I,"next_authorization":{"const":"NONE_WHILE_PENDING"}}),
      "producer_preflight":obj({"startup_commit":{"const":STARTUP_COMMIT},"checks":arr(S),"focused_test_count":I,"false_pass_attack_count":I,"clean_reference_count":{"const":20},"source_counts":obj({"Y2_R1_VALID":{"const":17},"Y2_R2_R1_VALID_COMPUTED":{"const":3}}),"formal_y3_outcome_access":{"const":False},"jpeg_decoded":{"const":False},"model_loaded":{"const":False},"cuda_initialized":{"const":False},"resource_projection_within_caps":{"const":True}}),
      "binding":obj({"name":S,"files":arr(FILE),"files_digest":H,"canonical_digest_bound":nullable(H),"commit":nullable(G)}),
      "clean_reference":clean,"clean_reference_manifest":obj({"records":arr(clean,20),"record_count":{"const":20},"records_digest":H,"source_counts":obj({"Y2_R1_VALID":{"const":17},"Y2_R2_R1_VALID_COMPUTED":{"const":3}}),"clean_rerun_count":{"const":0}}),
      "coordinate_manifest":obj({"locked_worker_count":I,"records":arr(coordinate,160),"record_count":{"const":160},"records_digest":H,"mapping_rule":{"const":"unit_index mod locked_worker_count"},"clean_condition_included":{"const":False}}),
      "false_pass_audit":obj({"checks":arr(CHECK),"attack_count":I,"rejected_count":I}),
      "implementation_seal":obj({"producer_commit":G,"producer_ref":{"const":PRODUCER_REF},"source_files":arr(FILE),"files_digest":H}),
      "startup_gate":obj({"locked_worker_count":{"const":4},"devices":arr(device,4),"python_executable_realpath":S,"environment_digest":H,"numeric_probe_digest":H,"dynamic_rebalancing":{"const":False},"hot_add":{"const":False},"started_unix_ns":I}),
      "shard_manifest":obj({"shard_id":I,"shard_count":I,"unit_count":I,"records":arr(coordinate),"records_digest":H,"mapping_rule":{"const":"unit_index mod locked_worker_count"},"dynamic_rebalancing":{"const":False}}),
      "shard_pending":obj({"shard_id":I,"shard_count":I,"started_unix_ns":I}),
      "worker_identity":worker,
      "unit_pending":obj({"unit_index":I,"subset_rank":I,"selection_order":I,"condition_index":I,"condition_id":S,"source_object_id":S,"attempt_id":S}),
      "attempt":obj({"unit_index":I,"attempt_id":S,"attempt_index":I,"record":attempt_record}),
      "transform":transform,"prediction":prediction,"resource":RESOURCE,"array_inventory":obj({"unit_index":I,"records":arr(ARRAY),"record_count":I,"records_digest":H}),"scientific_result":science,
      "unit_terminal":obj({"unit_index":I,"subset_rank":I,"selection_order":I,"condition_index":I,"condition_id":S,"source_object_id":S,"shard_id":I,"attempt_id":S,"failure_code":nullable(S),"fixed_target":TARGET,"source_terminal":FILE,"source_jpeg_sha256":H,"checkpoint_sha256":H,"environment_digest":H,"transform_record":nullable(FILE),"prediction_record":nullable(FILE),"scientific_result":nullable(FILE),"resource_record":FILE,"array_inventory":nullable(FILE),"array_records":arr(ARRAY),"attempt_records":arr(FILE),"quadrature_attempts":arr(attempt_record),"clean_rerun":{"const":False},"retry_count":{"type":"integer","minimum":0,"maximum":1},"evidence_mode":{"enum":["NEW_COMPUTE","SOURCE_REBIND"]},"predecessor_terminal":nullable(FILE)}),
      "shard_terminal":obj({"shard_id":I,"shard_count":I,"expected_unit_count":I,"terminal_count":I,"terminal_counts":obj({k:I for k in TERMINAL_STATES}),"fixed_initialization_resource":RESOURCE,"worker_wall_seconds":NN,"completed_unix_ns":I}),
      "launch_roster":obj({"locked_worker_count":I,"commands":arr(obj({"shard_id":I,"physical_index":I,"cuda_visible_devices":S})),"concurrent_launch":{"const":True},"dynamic_rebalancing":{"const":False}}),
      "launch_results":obj({"results":arr(obj({"shard_id":I,"returncode":I,"stdout":S,"stderr":S})),"result_count":I}),
      "transform_manifest":obj({"records":arr(obj({"unit_index":I,"condition_id":S,"transform_record":FILE})),"record_count":I,"records_digest":H}),
      "mask_inventory":obj({"records":arr(obj({"unit_index":I,"condition_id":S,"valid_mask":ARRAY,"fully_valid_patch_mask":ARRAY,"patch_inverse_operator":ARRAY})),"record_count":I,"records_digest":H}),
      "scientific_array_manifest":obj({"records":arr(obj({"unit_index":I,"selection_order":I,"condition_id":S,"array_inventory":nullable(FILE),"array_count":I,"records_digest":H,"status":{"enum":list(TERMINAL_STATES)}})),"record_count":I,"records_digest":H}),
      "prediction_manifest":obj({"records":arr(obj({"unit_index":I,"subset_rank":I,"condition_id":S,"status":{"enum":list(TERMINAL_STATES)},"prediction_changed":B,"prediction_record":nullable(FILE)})),"record_count":I,"records_digest":H}),
      "condition_summaries":obj({"records":arr(obj({"condition_id":{"enum":list(CONDITIONS)},"terminal_counts":obj({k:I for k in TERMINAL_STATES})})),"record_count":{"const":8},"records_digest":H}),
      "unified_manifest":obj({"records":arr(obj({"subset_rank":I,"condition_id":{"enum":["clean",*CONDITIONS]},"status":{"enum":list(TERMINAL_STATES)}}),180),"record_count":{"const":180},"records_digest":H}),
      "cumulative_resource":obj({"records":arr(obj({"stage":S,"measured_gpu_active_seconds":NN,"measured_cpu_seconds":NN,"accounted_gpu_active_seconds":NN,"accounted_cpu_seconds":NN,"conservative_gpu_reserve_seconds":NN,"conservative_cpu_reserve_seconds":NN,"critical_path_wall_seconds":nullable(NN),"retained_bytes":I,"missing_observation_ids":arr(S)}),2),"record_count":{"const":2},"records_digest":H}),
      "resource_ledger":obj({"stage":{"const":"Y3-R3+Y3-R4"},"unit_count":{"const":160},"source_rebind_unit_count":{"const":2},"new_compute_unit_count":{"const":158},"r3_measured_valid_gpu_active_seconds":NN,"r3_measured_valid_cpu_seconds":NN,"r3_interrupted_elapsed_upper_bound_seconds":NN,"r3_interrupted_gpu_reserve_seconds":NN,"r3_interrupted_cpu_reserve_seconds":NN,"r4_measured_gpu_active_seconds":NN,"r4_measured_cpu_seconds":NN,"measured_gpu_active_seconds":NN,"measured_cpu_seconds":NN,"gpu_active_seconds":NN,"h20_gpu_hours":NN,"cpu_seconds":NN,"cpu_hours":NN,"worker_wall_seconds":arr(NN,4),"critical_path_wall_seconds":NN,"stage_wall_seconds":NN,"retained_bytes":I,"retained_storage_gib":NN,"download_bytes":I,"peak_reserved_gpu_bytes":I,"caps":obj({k:NN for k in CAPS}),"caps_pass":B,"missing_fields":arr(S)}),
      "access_audit":obj({"download_bytes":{"const":0},"external_accesses":{"const":[]},"clean_rerun_count":{"const":0},"display_files":{"const":[]},"later_stages_started":{"const":[]},"bootstrap_executed":{"const":False},"holm_executed":{"const":False},"claim_decision_executed":{"const":False}}),
      "validation_report":obj({"checks":arr(S),"source_recomputed":{"const":True},"terminal_count":I,"valid_count":I,"complete_image_count":I,"prediction_flip_count":I,"records_digest":H}),
      "reconciliation":obj({"denominator_images":{"const":20},"expected_nonclean_units":{"const":160},"clean_reference_count":{"const":20},"terminal_count":I,"valid_count":I,"complete_image_count":I,"prediction_flip_count":I,"condition_counts":obj({c:obj({k:I for k in TERMINAL_STATES}) for c in CONDITIONS}),"shard_counts":arr(obj({"shard_id":I,"terminal_counts":obj({k:I for k in TERMINAL_STATES})})),"unified_condition_count":I,"blocking_states":arr(S),"bootstrap_executed":{"const":False},"holm_executed":{"const":False},"claim_decision_executed":{"const":False},"next_authorization":{"enum":[NEXT,"NONE"]}}),
      "inventory":obj({"root":S,"excluded":arr(S),"files":arr(FILE),"file_count":I,"total_bytes":I,"files_digest":H}),
      "completion":obj({"workflow_status":{"enum":["PASS",*TERMINAL_STATES[1:]]},"denominator_images":{"const":20},"expected_nonclean_units":{"const":160},"clean_reference_count":{"const":20},"terminal_count":I,"complete_image_count":I,"formal_outcome_access":{"const":False},"bootstrap_executed":{"const":False},"holm_executed":{"const":False},"claim_decision_executed":{"const":False},"next_authorization":{"enum":[NEXT,"NONE"]},"reconciliation":FILE,"validation_report":FILE,"whole_root_inventory":FILE,"payload_manifest":FILE,"report":FILE,"failure":nullable(FILE)}),
      "failure":obj({"phase":S,"failure_type":S,"message":S,"next_authorization":{"const":"NONE"}}),
    }
    envelope=obj({"$schema":{"const":ARTIFACT_SCHEMA.name},"schema_version":{"const":1},"protocol_id":{"const":PROTOCOL_ID},"stage_id":{"const":STAGE_ID},"document_type":{"enum":list(variants)},"status":{"enum":["PENDING","PASS",*TERMINAL_STATES]},"payload":{},"canonical_digest":H})
    envelope.update({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":ARTIFACT_SCHEMA.name,"$defs":variants,"allOf":[{"if":{"properties":{"document_type":{"const":k}}},"then":{"properties":{"payload":{"$ref":"#/$defs/"+k}}}} for k in variants]})
    return envelope

def validate_canonical(value: Mapping[str,Any]) -> None:
    unsigned=dict(value); claimed=unsigned.pop("canonical_digest",None)
    require(claimed==canonical(unsigned),"canonical digest mismatch")

def validate_document(value: Mapping[str,Any]) -> None:
    validate_canonical(value); Draft202012Validator(artifact_schema()).validate(value)

def signed(kind: str, payload: Mapping[str,Any], status: str="PASS") -> dict[str,Any]:
    value={"$schema":ARTIFACT_SCHEMA.name,"schema_version":1,"protocol_id":PROTOCOL_ID,"stage_id":STAGE_ID,"document_type":kind,"status":status,"payload":dict(payload)}
    value["canonical_digest"]=canonical(value);validate_document(value);return value

def write(path: Path, kind: str, payload: Mapping[str,Any], status: str="PASS") -> dict[str,Any]:
    value=signed(kind,payload,status);path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream:stream.write(canonical_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return value

def publish_terminal(pending: Path, path: Path, value: Mapping[str,Any]) -> None:
    validate_document(value);require(pending.is_file() and not path.exists(),"pending/terminal transition invalid");temporary=path.with_name(path.name+".tmp");require(not temporary.exists(),"terminal temp already exists")
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream:stream.write(canonical_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);pending.unlink();directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)

def expected_shape(name: str, audit: bool=False) -> tuple[int,...]:
    if name in CORE_ARRAY_NAMES:return r2.expected_shape(name)
    if name=="valid_mask":return (224,224)
    if name=="fully_valid_patch_mask":return (196,)
    if name=="patch_inverse_operator":return (196,196)
    if name in ("positive_input_map","negative_input_map"):return (224,224)
    if name.startswith(("positive_","negative_")):
        base=name.split("_",1)[1]
        if base in AUDIT_SIGNED_NAMES:return r2.expected_shape(base)
    if name.startswith("raw_attention_b") and audit:return (12,197,197)
    raise ProvenanceError("unexpected Y3 array name: "+name)

def expected_axes(name: str, shape: Sequence[int]) -> list[str]:
    special={"valid_mask":["pixel_y","pixel_x"],"fully_valid_patch_mask":["clean_patch"],"patch_inverse_operator":["clean_patch","transformed_patch"],"positive_input_map":["pixel_y","pixel_x"],"negative_input_map":["pixel_y","pixel_x"]}
    if name.startswith("raw_attention_b"):return ["head","query_token","key_token"]
    if name.startswith(("positive_","negative_")):
        base=name.split("_",1)[1]
        if base in AUDIT_SIGNED_NAMES:return r2.expected_axes(base,shape)
    return special[name] if name in special else r2.expected_axes(name,shape)

def array_digest(value: np.ndarray) -> str:
    return r2.array_digest(np.asarray(value))

def write_array(path: Path, value: np.ndarray, repo: Path) -> dict[str,Any]:
    array=np.ascontiguousarray(value,dtype=np.bool_ if np.asarray(value).dtype==np.bool_ else "<f4")
    path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream:np.save(stream,array,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return {**file_record(path,repo),"canonical_digest":array_digest(array),"dtype":"bool" if array.dtype==np.bool_ else "float32","shape":list(array.shape),"axes":expected_axes(path.stem,array.shape)}

def validate_array(repo: Path, row: Mapping[str,Any], audit: bool=False) -> np.ndarray:
    path=repo/str(row["path"]);require(path.is_file() and not path.is_symlink() and path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],"array file record drift: "+str(path));name=path.stem;array=np.load(path,allow_pickle=False)
    require(tuple(row["shape"])==array.shape==expected_shape(name,audit),"array shape drift: "+str(path),NumericalError)
    expected_dtype="bool" if name in ("valid_mask","fully_valid_patch_mask") else "float32"
    dtype="bool" if array.dtype==np.bool_ else "float32" if array.dtype==np.dtype("<f4") else str(array.dtype)
    require(dtype==expected_dtype and row["dtype"]==expected_dtype and row["axes"]==expected_axes(name,array.shape),"array dtype/axes drift: "+str(path),NumericalError)
    require(array.flags.c_contiguous and (array.dtype==np.bool_ or np.isfinite(array).all()),"array finite/layout drift: "+str(path),NumericalError)
    require(row["canonical_digest"]==array_digest(array),"array digest drift: "+str(path),NumericalError);return array

def expected_array_names(selection_order: int) -> tuple[str,...]:
    names=CORE_ARRAY_NAMES+EXTRA_ARRAY_NAMES+(AUDIT_ARRAY_NAMES if selection_order in AUDIT_OVERLAP else ())
    return tuple(sorted(names))

def noise_seed(source_object_id: str, condition: str) -> int:
    return int.from_bytes(hashlib.sha256(("STAGE_Y_NOISE_V1_20260910\0"+source_object_id+"\0"+condition).encode()).digest()[:8],"big")

def inverse_matrix(operator: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    w=np.asarray(operator,dtype=np.float64);t=np.asarray(matrix,dtype=np.float64)
    require(w.shape==t.shape==(196,196),"inverse matrix shape drift",ScientificValidityError)
    return np.ascontiguousarray(w@t@w.T,dtype="<f4")

def inverse_cls(operator: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w=np.asarray(operator,dtype=np.float64);c=np.asarray(vector,dtype=np.float64)
    require(w.shape==(196,196) and c.shape==(196,),"inverse CLS shape drift",ScientificValidityError)
    return np.ascontiguousarray(w@c,dtype="<f4")

def unit_index(subset_rank: int, condition_index: int) -> int:
    require(0<=subset_rank<20 and 0<=condition_index<8,"invalid Y3 coordinate");return subset_rank*8+condition_index

def subset_records(repo: Path) -> list[dict[str,Any]]:
    doc=y2c.checked(repo/SUBSET);q=doc["payload"]
    require(doc["canonical_digest"]=="71be2e5a31046185f744a38d8ce0424fd37c6e81f1d3bbbd229f0dd814fed106" and q["subset_digest"]=="91880fb763eccbe7f67f409c3c0284a74d8a48bb06538e9cf1ca3345427fe000","subset identity drift")
    require(q["selected_before_outcome_access"] is True and q["count"]==20,"subset selection contract drift")
    require(tuple(x["selection_order"] for x in q["records"])==SELECTION_ORDER and [x["subset_rank"] for x in q["records"]]==list(range(20)),"subset rank/order drift")
    return q["records"]

def transform_records(repo: Path) -> list[dict[str,Any]]:
    doc=y2c.checked(repo/TRANSFORMS);require(doc["canonical_digest"]=="7ed01d1378580df8700a9cdecb9d0246651fcc9a972372627e95dbf5138ee4da","transform registry drift")
    rows=doc["conditions"];require(tuple(x["id"] for x in rows)==("clean",)+CONDITIONS,"condition roster drift");return rows[1:]

def clean_references(repo: Path) -> list[dict[str,Any]]:
    result=[]
    for subset in subset_records(repo):
        order=subset["selection_order"];validated=y2c.validate_source(repo,order)
        source_stage=validated["original_producing_stage"]
        require(source_stage==("Y2-R2-R1" if order in COMPUTED_ORDERS else "Y2-R1"),"clean source classification drift")
        science_path=checked_file(repo,validated["scientific_result"]);science=y2c.checked(science_path)["payload"]
        source_terminal=y2c.checked(checked_file(repo,validated["source_terminal"]))
        target=science["target_identity"];require(target["class_id"]==source_terminal["payload"]["target_identity"]["class_id"],"clean target source drift")
        result.append({"subset_rank":subset["subset_rank"],"selection_order":order,"source_object_id":subset["source_object_id"],"source_stage":source_stage,"target_identity":target,"clean_target_logit":science["clean_logit"],"clean_baseline_logit":science["baseline_logit"],"source_terminal":validated["source_terminal"],"source_scientific_result":validated["scientific_result"],"source_validation_digest":validated["validation_digest"],"validated_array_count":262,"status":"VALID"})
    require(sum(x["source_stage"]=="Y2-R1" for x in result)==17 and sum(x["source_stage"]=="Y2-R2-R1" for x in result)==3,"source-derived 17+3 classification drift")
    return result

def coordinate_records(clean: Sequence[Mapping[str,Any]], worker_count: int) -> list[dict[str,Any]]:
    result=[]
    for row in clean:
        for ci,condition in enumerate(CONDITIONS):
            index=unit_index(row["subset_rank"],ci)
            result.append({"unit_index":index,"subset_rank":row["subset_rank"],"selection_order":row["selection_order"],"source_object_id":row["source_object_id"],"condition_index":ci,"condition_id":condition,"expected_shard_owner":index%worker_count,"clean_source_terminal":row["source_terminal"],"fixed_target":row["target_identity"]})
    require([x["unit_index"] for x in result]==list(range(160)),"coordinate coverage drift");return result

def remote_refs(repo: Path) -> dict[str,str]:
    return {line.split()[1]:line.split()[0] for line in git(repo,"ls-remote","origin").splitlines()}

def lfs_clean(repo: Path) -> None:
    value=git(repo,"lfs","status");require("(LFS:" not in value,"Git LFS object pending state",StartupError)

def initial_startup(repo: Path, allow_sources: bool=False) -> dict[str,Any]:
    remote=remote_refs(repo);require(git(repo,"symbolic-ref","--short","HEAD")==BRANCH,"branch drift",StartupError)
    require(git(repo,"rev-parse","HEAD")==git(repo,"rev-parse","@{upstream}")==remote.get("refs/heads/"+BRANCH)==STARTUP_COMMIT,"startup branch identity drift",StartupError)
    for ref,commit in EXPECTED_REFS.items():require(git(repo,"rev-parse",ref)==commit and remote.get(ref)==commit,"archive ref drift: "+ref,StartupError)
    require(OLD_PASS not in remote and not git(repo,"for-each-ref","--format=%(refname)",OLD_PASS),"forbidden old PASS ref exists",StartupError)
    require(remote.get(ORIGINAL_PRODUCER_REF)==git(repo,"rev-parse",ORIGINAL_PRODUCER_REF)==ORIGINAL_PRODUCER_COMMIT,"original Y3 producer ref drift",StartupError)
    require(remote.get(R1_PRODUCER_REF)==git(repo,"rev-parse",R1_PRODUCER_REF)==R1_PRODUCER_COMMIT,"Y3-R1 producer ref drift",StartupError)
    require(remote.get(R2_PRODUCER_REF)==git(repo,"rev-parse",R2_PRODUCER_REF)==R2_PRODUCER_COMMIT,"Y3-R2 producer ref drift",StartupError)
    require(remote.get(R2_BLOCKED_REF)==git(repo,"rev-parse",R2_BLOCKED_REF)==R2_BLOCKED_COMMIT,"Y3-R2 blocked ref drift",StartupError)
    require(remote.get(R3_PRODUCER_REF)==git(repo,"rev-parse",R3_PRODUCER_REF)==R3_PRODUCER_COMMIT,"Y3-R3 producer ref drift",StartupError)
    require(remote.get(R3_BLOCKED_REF)==git(repo,"rev-parse",R3_BLOCKED_REF)==R3_BLOCKED_COMMIT,"Y3-R3 blocked ref drift",StartupError)
    y2c.git_binding(repo,Path("evidence/stage_y/y3_perturbation_formal_20260910/formal"),R2_BLOCKED_COMMIT);y2c.git_binding(repo,Path("reports/stage_y_y3_20260912"),R2_BLOCKED_COMMIT);y2c.git_binding(repo,R3_ROOT,R3_BLOCKED_COMMIT);y2c.git_binding(repo,R3_REPORT_ROOT,R3_BLOCKED_COMMIT)
    for ref in (PRODUCER_REF,PASS_REF,BLOCKED_REF):require(ref not in remote and not git(repo,"for-each-ref","--format=%(refname)",ref),"Y3-R4 ref already exists: "+ref,StartupError)
    status=git(repo,"status","--porcelain").splitlines();paths=[line[3:].split(" -> ")[-1] for line in status]
    allowed={str(p) for p in PRODUCER_FILES} if allow_sources else set()
    unexpected=[path for path in paths if not any(path==candidate or path.endswith("/"+candidate) for candidate in allowed)]
    require(not unexpected,"unexpected worktree paths: "+repr(sorted(unexpected)),StartupError);lfs_clean(repo)
    require(not (repo/ROOT).exists() and not (repo/REPORT_ROOT).exists(),"Y3 formal/report root already exists",StartupError)
    for parent in (repo/"evidence/stage_y",repo/"reports"):
        require(not any(p.name.startswith(("y4","stage_y_y4")) for p in parent.iterdir()),"Y4 root exists",StartupError)
    ps=subprocess.run(["ps","-eo","pid=,args="],text=True,capture_output=True,check=True).stdout
    stale=[x for x in ps.splitlines() if str(os.getpid()) not in x and ("stage_y_y3_worker" in x or "run_stage_y_y3.py run-all" in x)]
    require(not stale,"residual Stage Y worker/PID",StartupError)
    y2c.check_refs(repo,True);y2c.validate_history(repo)
    completion=y2c.checked(repo/Y2_COMPLETION)
    require(sha(repo/Y2_COMPLETION)=="f09205218a1c24d6b4e3be457f3bb22c60dd099c7cf73a0db7cb6a8e545e54d5" and completion["canonical_digest"]=="6387ff5fbd895340cacd189e7f86759840af1448f67d655d873352f35a692a3c","Y2-R2-R2 completion identity drift",StartupError)
    p=completion["payload"];require(completion["status"]=="PASS" and p["denominator"]==p["valid_clean_count"]==100 and p["source_counts"]=={"Y2_R1_COMPLETED_BUDGET_REBIND":3,"Y2_R1_VALID_REBIND":84,"Y2_R2_COMPUTED":13} and p["gpu_execution"] is False and p["new_scientific_attempt_count"]==0 and p["next_authorization"]==TOKEN,"Y2-R2-R2 completion semantics drift",StartupError)
    subset_records(repo);transform_records(repo);return {"head":STARTUP_COMMIT,"remote_refs":len(remote)}

@lru_cache(None)
def r3_artifact_schema(repo: Path) -> dict[str,Any]:
    top=Path(git(repo,"rev-parse","--show-toplevel"))
    rel=(repo/ARTIFACT_SCHEMA).relative_to(top)
    return json.loads(subprocess.check_output(["git","show",f"{R3_PRODUCER_COMMIT}:{rel.as_posix()}"],cwd=repo))

def validate_r3_document(repo: Path, value: Mapping[str,Any]) -> None:
    validate_canonical(value)
    Draft202012Validator(r3_artifact_schema(repo)).validate(value)

def source_files(repo: Path) -> list[dict[str,Any]]:
    return [file_record(repo/p,repo) for p in PRODUCER_FILES+DEPENDENCIES]

def verify_implementation(repo: Path) -> str:
    seal=read(repo/ROOT/IMPLEMENTATION_SEAL);validate_document(seal);q=seal["payload"];commit=q["producer_commit"]
    require(git(repo,"rev-parse",PRODUCER_REF)==commit,"producer ref drift")
    rows=source_files(repo);require(rows==q["source_files"] and canonical({"files":rows})==q["files_digest"],"producer source bytes drift")
    top=Path(git(repo,"rev-parse","--show-toplevel"))
    for row in rows:
        rel=(repo/row["path"]).relative_to(top);blob=subprocess.check_output(["git","show",f"{commit}:{rel}"],cwd=repo)
        if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            lines=blob.decode().splitlines();require(lines[1].endswith(row["sha256"]) and int(lines[2].split()[1])==row["bytes"],"producer LFS pointer drift")
        else:require(hashlib.sha256(blob).hexdigest()==row["sha256"],"producer Git blob drift: "+row["path"])
    return commit

def payload_inventory(root: Path, excluded: set[str], repo: Path) -> dict[str,Any]:
    rows=[file_record(p,repo) for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in excluded]
    return {"root":root.relative_to(repo).as_posix(),"excluded":sorted(excluded),"files":rows,"file_count":len(rows),"total_bytes":sum(x["bytes"] for x in rows),"files_digest":canonical({"files":rows})}

def validate_schema_files(repo: Path) -> None:
    spec=read(repo/SPEC);schema=read(repo/SPEC_SCHEMA);artifact=read(repo/ARTIFACT_SCHEMA)
    Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(spec)
    Draft202012Validator.check_schema(artifact);require(artifact==artifact_schema(),"artifact schema source drift")
    def walk(value: Any) -> None:
        if isinstance(value,dict):
            if value.get("type")=="object":require(value.get("additionalProperties") is False,"open nested schema")
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(schema);walk(artifact)

def write_schema_files(repo: Path) -> None:
    spec=read(repo/SPEC)
    values=((repo/SPEC_SCHEMA,{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":SPEC_SCHEMA.name,"const":spec}),(repo/ARTIFACT_SCHEMA,artifact_schema()))
    for path,value in values:
        require(not path.exists(),"schema file already exists: "+str(path),StartupError)
        path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(fd,"wb") as stream:stream.write(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()+b"\n");stream.flush();os.fsync(stream.fileno())

def manifest_payload(name: str, records: Sequence[Any]) -> dict[str,Any]:
    rows=list(records);return {"name":name,"records":rows,"record_count":len(rows),"records_digest":canonical(rows)}

def read_clean_manifest(repo: Path) -> list[dict[str,Any]]:
    doc=read(repo/ROOT/"clean_reference_manifest.json");validate_document(doc);q=doc["payload"]
    require(q["records_digest"]==canonical(q["records"]),"clean reference manifest digest drift");return q["records"]

def read_coordinates(repo: Path) -> list[dict[str,Any]]:
    doc=read(repo/ROOT/"coordinate_manifest.json");validate_document(doc);q=doc["payload"]
    require(q["records_digest"]==canonical(q["records"]),"coordinate manifest digest drift");return q["records"]
