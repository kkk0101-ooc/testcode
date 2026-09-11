"""Closed artifact contracts shared by writer validation and independent audit."""
from functools import lru_cache

S={"type":"string"}
I={"type":"integer","minimum":0}
N={"type":"number"}
B={"type":"boolean"}
SHA={"type":"string","pattern":"^[0-9a-f]{64}$"}
COMMIT={"type":"string","pattern":"^[0-9a-f]{40}$"}

def obj(properties, optional=()):
    return {"type":"object","additionalProperties":False,"properties":properties,"required":[k for k in properties if k not in optional]}

def arr(item, count=None):
    value={"type":"array","items":item}
    if count is not None: value.update(minItems=count,maxItems=count)
    return value

def nullable(value): return {"anyOf":[value,{"type":"null"}]}
def const(value): return {"const":value}

FILE=obj({"path":S,"bytes":I,"sha256":SHA,"canonical_digest":SHA},("canonical_digest",))
ARRAY=obj({"path":S,"bytes":I,"sha256":SHA,"canonical_digest":SHA,"dtype":const("float32"),"shape":arr(I),"axes":arr(S)})
CALLS=obj({k:I for k in ("forward_calls","backward_calls","jvp_calls","jvp_direction_lanes","quadrature_alpha_batches","vjp_calls")})
TARGET=obj({"class_id":{"type":"integer","minimum":0,"maximum":999},"rule":const("clean predicted-class raw logit")})
RESOURCE=obj({"gpu_active_seconds":N,"cpu_seconds":N,"wall_seconds":N,"peak_allocated_gpu_bytes":I,"peak_reserved_gpu_bytes":I,"retained_bytes":I,"download_bytes":I,"calls":CALLS},("peak_allocated_gpu_bytes","download_bytes","calls"))
TOTAL=obj({"h20_gpu_hours":N,"cpu_hours":N,"wall_seconds_sum":N,"retained_bytes":I,"retained_storage_gib":N,"peak_memory_bytes":I,"download_bytes":I})
ATTEMPT=obj({"steps":I,"all_contracts_pass":B,"input_sum":N,"input_delta":N,"input_abs_error":N,"threshold":N,"closure_status":nullable(S),"identity_max_abs_error":nullable(N),"wall_seconds":N,"array_digest":SHA,"failure_code":S,"failure_phase":S},("input_sum","input_delta","input_abs_error","threshold","closure_status","identity_max_abs_error","wall_seconds","array_digest","failure_code","failure_phase"))
ENV=obj({"python_executable_realpath":S,"python_version":S,"torch_version":S,"torchvision_version":S,"timm_version":S,"cuda_version":S,"driver":S,"device_name":const("NVIDIA H20"),"cublas_workspace_config":const(":4096:8"),"deterministic_algorithms_enabled":const(True),"deterministic_warn_only_enabled":const(False),"cudnn_benchmark":const(False),"cudnn_deterministic":const(True),"cuda_matmul_allow_tf32":const(False),"cudnn_allow_tf32":const(False),"checkpoint_sha256":SHA,"numeric_contract_digest":S})
DEVICE=obj({"physical_index":I,"device_name":S,"gpu_uuid":S,"pci_bus_id":S,"total_memory_mib":I,"free_memory_mib":I,"driver":S,"environment_identity":ENV,"environment_digest":SHA,"numeric_probe_digest":SHA,"probe_operations":const(["torch.matmul_float32","torch.matmul_float64","backward","exact_repeat"])})
WORKER=obj({"physical_index":I,"cuda_visible_devices":S,"gpu_uuid":S,"pci_bus_id":S,"device_name":S,"hostname":S,"pid":I,"environment_identity":ENV,"environment_digest":SHA})
CHECK=obj({"attack":S,"expected":const("REJECT"),"observed":const("REJECT"),"pass":const(True)})

@lru_cache(None)
def science_schema():
    cut=obj({"patch_row_max_abs_error":N,"cell_max_abs_error":N,"total_abs_error":N,"scalar_map_matrix_abs_error":N,"pass":B})
    cuts=["post_patch_embedding_tokens"]+[f"b{b:02d}.{s}" for b in range(12) for s in ("input","first_residual","second_residual","output")]+["final_norm","classifier_input"]
    closure=obj({"status":S,"failures":arr(S),"input_abs_error":N,"input_threshold":N,"pixel_base_threshold":N,"total_threshold":N,"head_sum_max_abs_error":N,"branch_cell_max_abs_error":N,"scalar_map_matrix_max_abs_error":N,"per_cut":obj({k:cut for k in cuts})})
    head=obj({"head":I,"raw_logit_loss":N,"normalized_loss":N})
    curve=obj({"k":I,"heads":arr(I),"raw_logit_loss":N,"normalized_loss":N})
    ablock=obj({"block":I,"native_logit":N,"allocation_absolute_mass":arr(N,12),"rank_order":arr(I,12),"single_head":arr(head,12),"curves":obj({"top":arr(curve,8),"bottom":arr(curve,8)})})
    ablation=obj({"status":S,"target":I,"normalization_denominator":N,"intervention_point":const("native per-head value-mixed output before concat/output projection"),"replacement":const("zero same shape dtype"),"blocks":arr(ablock,12),"single_head_coordinates":const(144),"curve_points":const(192)})
    control=obj({"seed_index":I,"losses":arr(N,21),"auc":N})
    dblock=obj({"block":I,"percentiles":const(list(range(0,101,5))),"selected_counts":arr(I,21),"nns_losses":arr(N,21),"nns_auc":N,"controls":arr(control,32),"control_median_auc":N,"auc_gain":N})
    deletion=obj({"status":S,"scientific_decision":const(None),"blocks":arr(dblock,12),"control_permutations_per_block":const(32),"same_native_patch_order":const(True),"masking_baseline":const("normalized_zero"),"target":I})
    vjp=obj({"alpha_begin":I,"alpha_end":I,"component_start":I,"component_stop":I,"component_labels":arr(S),"wall_seconds":N})
    jvp=obj({"alpha_begin":I,"alpha_end":I,"source_patch_groups":arr(arr(I)),"cut_ids":arr(S),"wall_seconds":N})
    phase=obj({"forward_backward_wall_seconds":N,"vjp_wall_seconds":N,"jvp_wall_seconds":N,"vjp_groups":arr(vjp),"jvp_groups":arr(jvp)})
    return obj({"target_identity":TARGET,"clean_logit":N,"baseline_logit":N,"delta":N,"selected_steps":I,"closure":closure,"ablations":ablation,"deletion_operator":deletion,"calls":CALLS,"phase_telemetry":phase})

REBIND=obj({"selection_order":I,"source_object_id":S,"class_id":I,"source_type":{"enum":["Y2_R1_VALID_REBIND","Y2_R1_COMPLETED_BUDGET_REBIND"]},"status":const("VALID"),"source_terminal":FILE,"source_terminal_status":S,"source_scientific_result":nullable(FILE),"array_count":const(262),"array_records_digest":SHA,"attempt_records_digest":SHA,"validation_digest":SHA,"resource_telemetry":RESOURCE,"arrays_copied":const(False),"gpu_reexecuted":const(False),"intervention_payload_retained":B})
MISSING=obj({"selection_order":I,"source_object_id":S,"source_terminal":FILE})
COMPUTED=obj({"selection_order":I,"source_object_id":S,"class_id":I,"source_type":const("Y2_R2_COMPUTED"),"status":const("VALID"),"terminal":FILE,"resource_telemetry":RESOURCE,"validation_digest":SHA})
COUNTS=obj({"Y2_R1_VALID_REBIND":const(84),"Y2_R1_COMPLETED_BUDGET_REBIND":const(3),"Y2_R2_COMPUTED":const(13)})
OVERRUN=obj({"stage":S,"field":S,"observed":N,"former_ceiling":N,"classification":const("RESOURCE_OVERRUN_REPORTED"),"scientific_validity_effect":const("NONE")})
LEDGER=obj({"stage_resources":obj({k:TOTAL for k in ("Y2","Y2_R1","Y2_R2_BLOCKED","Y2_R2_R1")}),"cumulative_resources":TOTAL,"worker_wall_seconds":arr(N),"critical_path_wall_seconds":N,"overall_stage_wall_seconds":N,"prior_stage_wall_seconds":obj({"Y2":N,"Y2_R1":N,"Y2_R2_BLOCKED":nullable(N)}),"reporting_only":const(True),"retained_bytes_basis":S,"historical_blocked_cpu_measurement":S})

@lru_cache(None)
def artifact_schema():
    from . import stage_y_y2_r2 as r
    binding=obj({"producer_commit":COMMIT,"producer_ref":S,"source_files":arr(FILE),"files_digest":SHA})
    pending=obj({"started_at":S,"started_unix_ns":I,"next_authorization":const("NONE_WHILE_PENDING")})
    unit_pending=obj({"selection_order":I,"source_object_id":S,"attempt_id":S,"retry_index":const(1)})
    unit=obj({"selection_order":I,"source_object_id":S,"source_type":const("Y2_R2_COMPUTED"),"shard_id":I,"attempt_id":S,"condition_id":const("clean"),"failure_code":nullable(S),"target_identity":{"anyOf":[TARGET,const({})]},"source_binding":obj({"cohort_digest":SHA,"file_sha256":SHA}),"checkpoint_sha256":SHA,"environment_digest":SHA,"numeric_contract_digest":S,"quadrature_attempts":arr(ATTEMPT),"resource_telemetry":RESOURCE,"scientific_result":nullable(FILE),"array_inventory":nullable(FILE),"resource_record":FILE,"array_records":arr(ARRAY),"attempt_records":arr(FILE),"predecessor_terminal":FILE,"retry_index":const(1),"scientific_scope_changed":const(False),"target_metric_tolerance_changed":const(False)})
    shard=obj({"shard_id":I,"unit_count":I,"terminal_counts":obj({"VALID":I,"FAILED_NUMERICAL":I,"BLOCKED_PROVENANCE":I,"BLOCKED_SOURCE_DRIFT":I,"BLOCKED_ACCESS":I,"BLOCKED_INFRASTRUCTURE":I}),"fixed_initialization":RESOURCE,"worker_wall_seconds":N,"completed_unix_ns":I})
    reconciliation=obj({"denominator":const(100),"valid_clean_count":const(100),"source_counts":COUNTS,"selection_orders":const(list(range(100))),"records":arr({"anyOf":[REBIND,COMPUTED]},100),"records_digest":SHA,"formal_outcome_access":const(False),"bootstrap_executed":const(False),"holm_executed":const(False),"claim_decision_executed":const(False),"later_stages_started":const([]),"next_authorization":const(r.NEXT)})
    completion=obj({"completed_at":S,"denominator":const(100),"valid_clean_count":I,"source_counts":obj({k:I for k in r.SOURCE_TYPES}),"formal_outcome_access":const(False),"next_authorization":{"enum":[r.NEXT,"NONE"]},"reconciliation":nullable(FILE),"validation_report":nullable(FILE),"whole_root_inventory":FILE,"report":FILE,"failure":nullable(FILE)})
    variants={
        "pending":pending,"binding":binding,"unit_pending":unit_pending,"unit_terminal":unit,
        "scientific_result":science_schema(),"attempt":obj({"attempt_id":S,"selection_order":I,"retry_index":const(1),"quadrature_attempts":arr(ATTEMPT)}),
        "array_inventory":obj({"selection_order":I,"array_records":arr(ARRAY,262),"records_digest":SHA}),
        "resource_record":RESOURCE,"worker_identity":WORKER,"shard_terminal":shard,
        "shard_pending":obj({"shard_id":I,"started_unix_ns":I}),
        "startup_gate":obj({"locked_worker_count":{"type":"integer","minimum":1,"maximum":8},"devices":arr(DEVICE),"python_executable_realpath":S,"environment_digest":SHA,"numeric_probe_digest":SHA,"dynamic_rebalancing":const(False)}),
        "shard_manifest":obj({"shard_id":I,"unit_count":I,"selection_orders":arr(I),"mapping_rule":const("missing_roster_zero_based_index mod locked_worker_count"),"dynamic_rebalancing":const(False),"source_type":const("Y2_R2_COMPUTED")}),
        "rebind_record":REBIND,"rebind_manifest":obj({"source_type":S,"record_count":I,"selection_orders":arr(I),"records":arr(REBIND),"records_digest":SHA}),
        "missing_manifest":obj({"record_count":const(13),"selection_orders":const(list(r.MISSING_ORDERS)),"records":arr(MISSING,13),"records_digest":SHA}),
        "preflight":obj({"checks":arr(S),"focused_test_count":I,"false_pass_count":I,"cohort_digest":const(r.COHORT_DIGEST),"selection_digest":const(r.SELECTION_DIGEST),"startup_commit":COMMIT,"resource_telemetry":RESOURCE}),
        "verification_resource":RESOURCE,"resource_ledger":LEDGER,
        "overruns":obj({"resource_overruns":arr(OVERRUN),"reporting_only":const(True)}),
        "access_audit":obj({"external_accesses":const([]),"download_bytes":const(0),"clean_array_validation_performed":B,"bootstrap_executed":const(False),"holm_executed":const(False),"claim_decision_executed":const(False),"later_stages_started":const([])}),
        "false_pass_audit":obj({"attack_count":I,"rejected_count":I,"checks":arr(CHECK)}),
        "reconciliation":reconciliation,"unified_manifest":reconciliation,
        "validation_report":obj({"checks":arr(S),"record_count":const(100),"records_digest":SHA}),
        "payload_manifest":obj({"files":arr(FILE),"file_count":I,"files_digest":SHA}),
        "inventory":obj({"root":S,"excluded":arr(S),"files":arr(FILE),"file_count":I,"total_bytes":I,"files_digest":SHA}),
        "completion":completion,"failure":obj({"phase":S,"failure_type":S,"message":S,"next_authorization":const("NONE")}),
    }
    envelope=obj({"$schema":const(r.ARTIFACT_SCHEMA.name),"schema_version":const(1),"document_type":{"enum":[r.PREFIX+k for k in variants]},"protocol_id":const(r.PROTOCOL_ID),"stage_id":const(r.STAGE_ID),"status":{"enum":["PASS","VALID","PENDING","FAILED_NUMERICAL","BLOCKED_STARTUP_STATE","BLOCKED_SOURCE_DRIFT","BLOCKED_ACCESS","BLOCKED_PROVENANCE","BLOCKED_INFRASTRUCTURE","BLOCKED_SCIENTIFIC_VALIDITY"]},"payload":{"oneOf":[{"$ref":"#/$defs/"+k} for k in variants]},"canonical_digest":SHA})
    envelope["properties"]["payload"]={"anyOf":[{"$ref":"#/$defs/"+k} for k in variants]}
    envelope.update({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":r.ARTIFACT_SCHEMA.name,"$defs":variants,
        "allOf":[{"if":{"properties":{"document_type":const(r.PREFIX+k)}},"then":{"properties":{"payload":{"$ref":"#/$defs/"+k}}}} for k in variants]})
    return envelope

if __name__=="__main__":
    import json
    from pathlib import Path
    from . import stage_y_y2_r2 as r
    from .constants import REPO_ROOT
    repo=Path(REPO_ROOT)
    for path,value in ((r.ARTIFACT_SCHEMA,artifact_schema()),(r.RUN_SCHEMA,{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":r.RUN_SCHEMA.name,"const":r.read_json(repo/r.RUN_SPEC)})):
        with (repo/path).open("x") as stream: json.dump(value,stream,indent=2);stream.write("\n")
