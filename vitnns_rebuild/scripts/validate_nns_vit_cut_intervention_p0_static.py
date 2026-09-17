#!/usr/bin/env python3
"""Standard-library-only P0-A/B static contract generator, validator, and sealer.

This module deliberately has no image, array, checkpoint, model, CUDA, or
accelerator-reading surface.  P0-C and every empirical stage are out of scope.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

PROTOCOL_ID = "NNS_VIT_CUT_INTERVENTION_PREDICTION_V1"
STAGE_ID = "P0-STATIC"
BRANCH = "stage-d-d3-pass"
AUTHORITY_COMMIT = "7f169c7ee8d4acf5a6274f93a8340310dd1a53de"
PLANNING_REL = Path("docs/NNS_ViT完整切面归因与匹配干预研究规划_20260917.md")
PLANNING_REPO_REL = Path("vitnns_rebuild") / PLANNING_REL
PLANNING_SHA256 = "fa1fd7884c9fe64a6dd49450202a6135f66b1c999b3c5d2620df50b7881ca15c"
START_HEAD = "6bcf71317ec71bf32828afeeb613f44013d54ea1"
NEXT_AUTHORIZATION = "P0_C_ENGINEERING_RESOURCE_PILOT_ONLY"
FULL_TOKEN = "P0_STATIC_PASS_AUTHORIZES_P0_C_ENGINEERING_RESOURCE_PILOT_ONLY"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
CONFIG_DIR = Path("config/nns_vit_cut_intervention")
MANIFEST_DIR = Path("manifests/nns_vit_cut_intervention")
ROOT = Path("evidence/nns_vit_cut_intervention/p0_static_20260917/formal")
REPORT_ROOT = Path("reports/nns_vit_cut_intervention_p0_static_20260917")
HANDOFF = Path("docs/nns_vit_cut_intervention_p0_static_handoff_20260917.md")
PENDING = ROOT / "completion/p0_static.pending.json"
COMPLETION = ROOT / "completion/p0_static.json"
FORBIDDEN_IMPORTS = {"torch", "torchvision", "timm", "numpy", "PIL", "cv2", "cupy"}

CONTRACT_PATHS = (
    CONFIG_DIR / "p0_static.json",
    MANIFEST_DIR / "target_registry.json",
    MANIFEST_DIR / "preprocessing_registry.json",
    MANIFEST_DIR / "cut_intervention_registry.json",
    MANIFEST_DIR / "baseline_registry.json",
    MANIFEST_DIR / "statistical_analysis_plan.json",
    MANIFEST_DIR / "failure_registry.json",
    MANIFEST_DIR / "artifact_contract.json",
    MANIFEST_DIR / "p0_c_cost_pilot_plan.json",
    MANIFEST_DIR / "schema_registry.json",
)
SCHEMA_PATHS = (CONFIG_DIR / "p0_static.schema.json", CONFIG_DIR / "artifact.schema.json")
SOURCE_PATHS = CONTRACT_PATHS + SCHEMA_PATHS + (
    Path("scripts/validate_nns_vit_cut_intervention_p0_static.py"),
    Path("tests/test_nns_vit_cut_intervention_p0_static.py"),
    PLANNING_REL,
)


class StaticContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticContractError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StaticContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StaticContractError(f"invalid UTF-8 JSON: {label}") from exc

    def bad_constant(token: str) -> Any:
        raise StaticContractError(f"non-finite JSON token in {label}: {token}")

    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=bad_constant)
    except json.JSONDecodeError as exc:
        raise StaticContractError(f"invalid JSON in {label}: {exc}") from exc


def _safe_path(repo: Path, relative: Path, *, must_exist: bool = True) -> Path:
    require(not relative.is_absolute() and ".." not in relative.parts, f"root escape path: {relative}")
    repo_real = repo.resolve()
    path = repo / relative
    cursor = repo
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            require(not cursor.is_symlink(), f"symlink forbidden: {relative}")
    if must_exist:
        require(path.is_file(), f"missing regular file: {relative}")
    resolved = path.resolve(strict=must_exist)
    require(resolved == repo_real or repo_real in resolved.parents, f"root escape resolved path: {relative}")
    return path


def read_json(repo: Path, relative: Path) -> Any:
    return strict_json_bytes(_safe_path(repo, relative).read_bytes(), relative.as_posix())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("canonical_digest", None)
    result["canonical_digest"] = canonical_digest(result)
    return result


def validate_signed(value: Mapping[str, Any], label: str) -> None:
    require(isinstance(value.get("canonical_digest"), str), f"missing canonical digest: {label}")
    unsigned = {key: item for key, item in value.items() if key != "canonical_digest"}
    require(value["canonical_digest"] == canonical_digest(unsigned), f"canonical digest mismatch: {label}")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent(path)


def replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    require(not temporary.exists(), f"temporary exists: {temporary}")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)


def write_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent(path)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def common(document_type: str) -> dict[str, Any]:
    return {"schema_version": 1, "document_type": document_type, "protocol_id": PROTOCOL_ID, "stage_id": STAGE_ID}


def build_contracts() -> dict[Path, dict[str, Any]]:
    config = {**common("p0_static_config"),
        "planning_authority": {"commit": AUTHORITY_COMMIT, "document": PLANNING_REL.as_posix(), "sha256": PLANNING_SHA256, "ancestor_required": True, "working_tree_identity_required": True},
        "authorization": {"input": "P0_A_B_STATIC_PACKAGE_ONLY", "completion_status": "PASS_STATIC", "completion_token": FULL_TOKEN, "next_authorization": NEXT_AUTHORIZATION, "p0_terminal_pass": False, "p1_cohort_seal_authorized": False, "independent_cohort_outcome_access_authorized": False, "gpu_production_authorized": False},
        "model": {"name": "vit_base_patch16_224.augreg2_in21k_ft_in1k", "checkpoint_sha256": MODEL_SHA256, "canonical_tensor_digest": MODEL_TENSOR_DIGEST, "blocks": 12, "heads_per_block": 12, "embed_dim": 768, "mlp_hidden_dim": 3072, "query_tokens": 197, "patch_tokens": 196, "patch_size": [16, 16], "input_shape": [3, 224, 224], "model_loaded_in_static_stage": False},
        "access_boundary": {"principle": "source-code access != empirical-outcome access", "allowed": ["source_code_tests_config_schema_protocol_docs", "git_history_refs_lfs_metadata", "json_csv_field_path_hash_shape_dtype_count_provenance_metadata", "published_model_checkpoint_preprocessing_metadata", "sealed_completion_and_resource_telemetry", "synthetic_fixtures"], "forbidden": ["jpeg_png_pixel_decode", "npy_npz_tensor_map_activation_intervention_numeric_read", "new_cohort_prediction_logit_attribution_effect_outcome_read", "checkpoint_tensor_deserialization_or_model_load", "forbidden_scientific_library_import", "cuda_gpu_accelerator_probe", "p0_c_or_p1_p2_p3_p4_execution", "cohort_membership_y_star_or_source_object_selection", "sealed_root_modification"]},
        "static_scope": {"p0_a_math_contract": "FROZEN", "p0_b_statistics_schema_access_contract": "FROZEN", "p0_c": "NOT_EXECUTED", "p1": "NOT_EXECUTED", "p2": "NOT_EXECUTED", "p3": "NOT_EXECUTED", "p4": "NOT_EXECUTED"}}

    target = {**common("target_registry"),
        "primary_target": {"id": "shared_true_class_raw_logit", "formula": "F(x)=logit_{y*}(x)", "output": "raw_logit", "shared_across_images": True, "computed_in_static_stage": False},
        "metadata_only_selection": {"delimiter": "|", "hash": "SHA256", "cohort_size": 20, "eligibility": "true_label_equals_y_star", "steps": ["construct_union_of_all_historical_source_object_exclusions", "retain_classes_with_at_least_20_remaining_true_label_images", "sort_classes_by_SHA256(protocol_id|class_id)_ascending_and_take_first", "sort_class_images_by_SHA256(protocol_id|source_object_id)_ascending_and_take_first_20", "save_candidate_exclusion_and_membership_digests"], "insufficient_code": "BLOCKED_INSUFFICIENT_DISJOINT_COHORT", "runtime_replacement_allowed": False, "executed_in_static_stage": False},
        "historical_exclusion_sets": ["Stage_C_D_E", "F3_F3H_F3C", "X1_X2_X2_MECH", "Stage_Y", "Stage_S", "endpoint_Level_3_Level_4", "manual_figures_galleries_threshold_debug"],
        "membership_forbidden_fields": ["correctness", "prediction", "logit", "margin", "confidence", "attribution", "intervention_effect"],
        "secondary_target": {"id": "per_image_predicted_class", "pooled_with_primary": False}}

    preprocessing = {**common("preprocessing_registry"),
        "model_config_source": {"kind": "current_model_config_and_sealed_binding", "model_name": config["model"]["name"], "not_a_generic_vit_constant": True},
        "spec": {"color_mode": "RGB", "raw_range": [0.0, 1.0], "resize_shorter_side": 248, "interpolation": "bicubic", "antialias": True, "center_crop": [224, 224], "channel_order": "RGB", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "normalization": {"formula": "x_norm,c=(x_raw,c-mu_c)/sigma_c", "reference_id": "normalized_zero", "x0_norm": [0.0, 0.0, 0.0], "x0_raw_equals_mean": True, "x0_raw": [0.5, 0.5, 0.5], "same_reference_for_complete_cut_and_activation_replacement": True}}

    cuts = {**common("cut_intervention_registry"),
        "component_conductance": {"formula": "K^C_{k,i}=Delta_x_i integral_0^1 sum_{j in I_k} (dF(x_alpha)/dz_{k,j})(dz_{k,j}(x_alpha)/dx_i) d_alpha", "vector_formula": "K^C_{k,i}=Delta_x_i integral_0^1 <grad_{z_k}F(x_alpha), dz_k(x_alpha)/dx_i> d_alpha", "path": "x_alpha=x0+alpha*(x-x0)", "input_axes": ["channel", "row", "column"], "component_coordinate_sets": {"head": "all_token_x_head_dimension", "branch": "all_token_x_channel", "routing_value": "all_query_x_key_or_value_coordinates_per_head"}, "coordinate_reduction": "sum_all_component_coordinates", "aggregation_dtype": "float64", "retained_map_level": "signed_4x4_cell_map_for_every_primary_component", "native_map_retention": "p0_c_prior_access_image_and_four_presealed_audit_images_only"},
        "head_cut": {"mathematical_type": "computational_edge_cut", "hook_position": "post_attention_dropout_pre_head_concat_pre_output_projection", "query_token_outputs": 197, "heads": 12, "head_shape": [197, 64], "matched_output_projection_columns": True, "identity_edge_registered_separately": True, "main_intervention": "all_token_reference_head_replacement", "secondary_intervention": "all_token_head_zeroing", "cls_only_is_main": False},
        "residual_cuts": [{"id": "first_residual", "left_edge": "identity", "right_edge": "attention", "left_variable": "z_id=h", "right_variable": "z_attn=a(h)", "sum": "h_prime=z_id+z_attn", "shared_output_adjoint": "grad_{h_prime}F", "original_node_total_gradient_for_left": False}, {"id": "second_residual", "left_edge": "carry", "right_edge": "mlp", "left_variable": "z_carry=h_prime", "right_variable": "z_mlp=m(h_prime)", "sum": "h_next=z_carry+z_mlp", "shared_output_adjoint": "grad_{h_next}F", "original_node_total_gradient_for_left": False}],
        "attention_differential": {"mathematical_type": "differential_path_not_complete_edge_cut", "identity": "dO=(dA)V+A(dV)", "attention_decomposition": "K_attention~=K_R+K_V", "routing_decomposition": "K_R~=K_Q+K_K+K_B", "terms": ["routing", "value", "query", "key", "bias"], "input_independent_bias_contribution": 0.0, "bias_retained_in_forward_score": True, "routing_value_complete_edge_cut": False},
        "interventions": {"effect_formula": "e_k=F(x)-F(x;z_k<-z_k_ref)", "value": "A(x),V(x0)", "routing": "A(x0),V(x)", "joint": "A(x0),V(x0)", "query": "Q(x0),K(x),recompute_score_and_softmax", "key": "K(x0),Q(x),recompute_score_and_softmax", "interaction_formula": "I_RV=e_R+e_V-e_(R+V)", "opposite_sign_rejected": True},
        "attribution_scalars": {"signed": "a_k=sum_i K_{k,i}", "absolute_mass": "m_k=sum_i abs(K_{k,i})", "absolute_share": "q_k=m_k/sum_{j in C}m_j", "signed_predicts": "signed_effect", "mass_predicts": "absolute_effect", "zero_mass_code": "NA_ZERO_CUT_MASS"},
        "numerical_contracts": {"quadrature": {"rule": "Gauss-Legendre", "schedule": [128, 256, 512, 1024, 2048, 4096, 8192], "selected_order_rule": "first_order_where_total_ig_and_all_families_pass_all_three_contracts"}, "ALGEBRAIC_CUT_RECONSTRUCTION": {"pixel_abs": 0.00002, "pixel_rel": 0.005, "total_abs": 0.0002, "total_rel": 0.01, "closures": ["heads_to_attention", "identity_plus_attention_to_first_residual", "carry_plus_mlp_to_second_residual", "value_plus_routing_to_attention", "query_plus_key_plus_bias_to_routing"]}, "IG_COMPLETENESS_TO_TARGET_DELTA": {"abs": 0.0001, "rel": 0.005, "small_target_threshold": 0.000001, "small_target_code": "NA_SMALL_TARGET_DELTA"}, "QUADRATURE_SUCCESSIVE_ORDER_CONVERGENCE": {"map_l1_rel": 0.005, "component_mass_rel": 0.005, "component_signed_rel": 0.01, "small_component_threshold": 0.00000001, "small_component_abs": 0.00000001, "failure_at_max_order": "FAILED_QUADRATURE_CONVERGENCE"}, "low_order_algebraic_pass_implies_convergence": False}}

    baselines = {**common("baseline_registry"),
        "families": [{"family": "H", "scope": "head", "mandatory": ["output_norm", "output_change_norm", "gradient_norm", "gradient_times_activation", "mean_cls_attention", "cls_attention_entropy", "raw_cls_attention_mass"], "descriptive_sanity": ["deterministic_random_ranking"]}, {"family": "B", "scope": "branch", "mandatory": ["output_norm", "output_change_norm", "gradient_norm", "gradient_times_activation"], "descriptive_sanity": ["deterministic_random_layer_ranking"]}, {"family": "R", "scope": "routing_value", "mandatory": ["A_change_norm", "V_change_norm", "Q_activation_change_norm", "K_activation_change_norm", "V_activation_change_norm", "Q_gradient_norm", "K_gradient_norm", "V_gradient_norm", "mean_cls_attention", "attention_entropy"], "descriptive_sanity": ["deterministic_random_ranking"]}],
        "selection": {"unit": "synchronized_image_bootstrap_replicate", "method": "maximum_cohort_statistic_across_entire_mandatory_set", "per_image_oracle_allowed": False, "random_control_is_mandatory": False, "gain_formula": "complete_cut_cohort_statistic-max_mandatory_baseline_cohort_statistic"}}

    hypotheses: list[dict[str, Any]] = []
    metrics = ["absolute_spearman", "signed_spearman", "direction_agreement_minus_0.5", "top_quartile_enrichment", "baseline_gain"]
    for index, metric in enumerate(metrics, 1):
        hypotheses.append({"id": f"H{index}", "family": "H", "branch": "head", "metric": metric, "null": "T<=0", "alternative": "T>0", "tail": "one_sided", "multiplicity": "Holm_within_H"})
    for offset, branch in ((0, "attention"), (5, "mlp")):
        for index, metric in enumerate(metrics, 1):
            hypotheses.append({"id": f"B{offset+index}", "family": "B", "branch": branch, "metric": metric, "null": "T<=0", "alternative": "T>0", "tail": "one_sided", "multiplicity": "Holm_within_B1_B10"})
    for index, metric in enumerate(("routing_value_agreement_minus_0.5", "continuous_association", "baseline_gain"), 1):
        hypotheses.append({"id": f"R{index}", "family": "R", "branch": "routing_value", "metric": metric, "null": "T<=0", "alternative": "T>0", "tail": "one_sided", "multiplicity": "Holm_within_R"})
    statistics = {**common("statistical_analysis_plan"), "independent_unit": "image",
        "aggregation": {"head": "spearman_within_each_12_head_block_then_median_across_valid_blocks", "head_min_valid_blocks": 8, "depth_groups": {"shallow": [0, 1, 2, 3], "middle": [4, 5, 6, 7], "deep": [8, 9, 10, 11]}, "routing_value": "one_spearman_per_image_across_144_block_head_pairs", "attention_branch": "one_spearman_per_image_across_12_blocks", "mlp_branch": "one_spearman_per_image_across_12_blocks", "ties": "average_ranks", "constant_vector": "NA_CONSTANT_VECTOR", "head_rows_are_independent_n": False},
        "validity": {"planned_images": 20, "minimum_valid_images": 18, "report_completion_rate_for_valid_range": [18, 20], "below_minimum": "INVALID_INSUFFICIENT_VALID_IMAGES", "undefined_spearman_handling": "typed_NA_retained_in_denominator_never_zero_imputed"},
        "inference": {"bootstrap_replicates": 20000, "bootstrap_unit": "paired_image", "rng": "PCG64", "namespace": "NNS_VIT_CUT_INTERVENTION_V1|metric|cut_family", "interval": "linear_percentile_95", "sign_flip": "exact_image_level_enumeration", "sign_flip_max_enumerations": 1048576, "multiplicity": "Holm_separately_within_H_B_R", "alpha": 0.05},
        "thresholds": {"spearman_practical_min": 0.30, "direction_agreement_min": 0.60, "replacement_zeroing_ranking_consistency_min": 0.50, "bootstrap_lower_bound_strictly_positive": True, "holm_adjusted_p_strictly_less_than": 0.05, "baseline_gain_strictly_positive": True},
        "hypotheses": hypotheses,
        "family_decision": {"allowed": ["SUPPORTED_PREDICTIVE_RELATION", "NOT_SUPPORTED", "INVALID", "BLOCKED_INFRASTRUCTURE"], "requires_practical_bootstrap_and_holm": True, "branch_family_same_branch_all_five_tests": True},
        "aggregate_decision": {"minimum_supported_families": 2, "requires_head_or_routing_value": True, "positive_baseline_gain_each_supported_family": True, "invalid_blocks_go": True, "blocked_infrastructure_blocks_go": True, "go_token": "GO_CUT_INTERVENTION_EXPANSION", "no_go_token": "NO_GO_CUT_INTERVENTION_EXPANSION"}}

    failures = {**common("failure_registry"), "typed_codes": [
        {"code": "NA_ZERO_CUT_MASS", "class": "NA", "meaning": "absolute cut mass denominator is zero"}, {"code": "NA_CONSTANT_VECTOR", "class": "NA", "meaning": "Spearman input is constant"}, {"code": "NA_INSUFFICIENT_VALID_BLOCKS", "class": "NA", "meaning": "fewer than eight valid head blocks"}, {"code": "NA_SMALL_TARGET_DELTA", "class": "NA", "meaning": "absolute target delta below 1e-6"}, {"code": "INVALID_INSUFFICIENT_VALID_IMAGES", "class": "INVALID", "meaning": "fewer than eighteen valid images"}, {"code": "FAILED_ALGEBRAIC_RECONSTRUCTION", "class": "FAILED", "meaning": "same-order cut closure failed"}, {"code": "FAILED_IG_COMPLETENESS", "class": "FAILED", "meaning": "target-delta completeness failed"}, {"code": "FAILED_QUADRATURE_CONVERGENCE", "class": "FAILED", "meaning": "successive-order convergence failed at 8192"}, {"code": "FAILED_INTERVENTION_NOOP_PARITY", "class": "FAILED", "meaning": "clean-to-clean replacement drifted"}, {"code": "FAILED_NONFINITE", "class": "FAILED", "meaning": "non-finite value observed"}, {"code": "FAILED_RESOURCE", "class": "FAILED", "meaning": "resource execution failed"}, {"code": "NO_GO_RESOURCE", "class": "NO_GO", "meaning": "P95 safety projection exceeds owner ceiling"}, {"code": "BLOCKED_INFRASTRUCTURE", "class": "BLOCKED", "meaning": "infrastructure prevents scientific decision"}, {"code": "BLOCKED_PROTOCOL_DRIFT", "class": "BLOCKED", "meaning": "authority or contract binding drift"}, {"code": "BLOCKED_INSUFFICIENT_DISJOINT_COHORT", "class": "BLOCKED", "meaning": "metadata-only eligible cohort below twenty"}, {"code": "BLOCKED_STARTUP_STATE", "class": "BLOCKED", "meaning": "startup Git LFS authority or root gate failed"}], "failure_rules": {"retain_in_denominator": True, "replace_failed_image": False, "drop_difficult_component": False, "change_max_order_after_outcome": False, "blocked_next_authorization": "NONE"}}

    artifacts = {**common("artifact_contract"), "json": {"encoding": "UTF-8", "duplicate_keys": "REJECT", "nan_infinity": "REJECT", "canonicalization": "UTF8_sorted_keys_compact_no_nan_SHA256", "unknown_fields": "REJECT", "all_object_schemas_additional_properties_false": True}, "filesystem": {"symlink": "REJECT", "root_escape": "REJECT", "regular_files_only": True}, "lifecycle": {"phases": ["draft", "final", "seal"], "atomic_transition": "write_pending_fsync_file_os_replace_fsync_directory", "terminal_root_rewrite_allowed": False}, "required_evidence": ["startup_gate.json", "source_binding.json", "access_audit.json", "contract_bundle.json", "false_pass_audit.json", "validation_report.json", "whole_root_inventory.json", "completion/p0_static.pending.json", "completion/p0_static.json"], "required_reports": ["report.md", "validation_report.json"], "future_long_table_fields": ["protocol_id", "image_id", "source_object_id", "target_class", "target_kind", "reference_id", "block", "head", "cut_family", "cut_id", "component", "signed_attribution_sum", "absolute_mass", "absolute_share", "attribution_array_path", "reference_replacement_effect", "zeroing_effect", "inclusion_exclusion_interaction", "normalized_interaction", "reconstruction_error", "baseline_predictors", "integration_steps", "array_sha256", "worker_id", "gpu_uuid", "runtime_seconds", "status", "failure_code"]}

    pilot = {**common("p0_c_cost_pilot_plan"), "status": "FROZEN_NOT_EXECUTED", "authorization_required": NEXT_AUTHORIZATION, "allowed_inputs": ["synthetic_fixtures", "one_explicitly_registered_prior_access_image"], "measurements": ["complete_cut_attribution_one_image", "144_head_reference_replacements", "144_head_zeroing", "432_routing_value_joint_replacements", "288_query_key_replacements", "24_attention_mlp_branch_replacements", "mandatory_baseline_predictors"], "owner_ceilings": {"aggregate_h20_gpu_hours_max": 80.0, "eight_gpu_wall_hours_max": 24.0, "cpu_hours_max": 500.0, "peak_reserved_gpu_gib_per_process_max": 32.0, "fleet_concurrent_reserved_gpu_gib_max": 256.0, "retained_storage_gib_max": 32.0, "temporary_storage_gib_max": 64.0, "devices": 8}, "projection": {"formula": "P95_measured_cost*planned_calls*20_images*1.5", "safety_multiplier": 1.5, "image_count": 20, "measured_projection": None, "static_package_must_not_fabricate_projection": True, "exceedance_decision": "NO_GO_RESOURCE"}, "p1_or_production_authorized": False}

    schema_registry = {**common("schema_registry"), "schema_draft": "2020-12", "schemas": [{"id": "p0_static_contracts", "path": (CONFIG_DIR / "p0_static.schema.json").as_posix(), "scope": "config_and_nine_registries"}, {"id": "p0_static_artifacts", "path": (CONFIG_DIR / "artifact.schema.json").as_posix(), "scope": "evidence_and_completion_documents"}], "documents": [{"path": path.as_posix(), "document_type": value["document_type"], "schema_id": "p0_static_contracts"} for path, value in ((CONTRACT_PATHS[0], config), (CONTRACT_PATHS[1], target), (CONTRACT_PATHS[2], preprocessing), (CONTRACT_PATHS[3], cuts), (CONTRACT_PATHS[4], baselines), (CONTRACT_PATHS[5], statistics), (CONTRACT_PATHS[6], failures), (CONTRACT_PATHS[7], artifacts), (CONTRACT_PATHS[8], pilot))] + [{"path": CONTRACT_PATHS[9].as_posix(), "document_type": "schema_registry", "schema_id": "p0_static_contracts"}], "closed_schema_required": True, "open_payload_forbidden": True}

    values = [config, target, preprocessing, cuts, baselines, statistics, failures, artifacts, pilot, schema_registry]
    return {path: signed(value) for path, value in zip(CONTRACT_PATHS, values)}


def _shape_schema(value: Any, *, field: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        properties = {key: _shape_schema(item, field=key) for key, item in value.items()}
        result: dict[str, Any] = {"type": "object", "required": sorted(value), "properties": properties, "additionalProperties": False}
        return result
    if isinstance(value, list):
        if not value:
            return {"type": "array", "maxItems": 0}
        item_schemas = [_shape_schema(item, field=field) for item in value]
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item_schema in item_schemas:
            identity = canonical_digest(item_schema)
            if identity not in seen:
                unique.append(item_schema)
                seen.add(identity)
        return {"type": "array", "items": unique[0] if len(unique) == 1 else {"oneOf": unique}}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    result = {"type": "string"}
    if field in {"document_type", "protocol_id", "stage_id", "schema_version"}:
        result["const"] = value
    return result


def _definition_name(document_type: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in document_type)


def _schema(title: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = {_definition_name(value["document_type"]): _shape_schema(value) for value in documents}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "oneOf": [{"$ref": f"#/$defs/{name}"} for name in definitions], "$defs": definitions}


def _artifact_templates() -> list[dict[str, Any]]:
    record = {"path": "x", "sha256": "0" * 64, "bytes": 1, "canonical_content_digest": "0" * 64}
    check = {"id": "x", "status": "PASS", "detail": "x"}
    startup = signed({**common("nns_vit_cut_intervention_startup_gate"), "status": "PASS", "checked_at": "x", "branch": BRANCH, "local_head": START_HEAD, "tracking_ref": START_HEAD, "remote_branch": START_HEAD, "porcelain_v2": "x", "index_clean": True, "untracked_paths_at_start": [], "lfs_pending_commit_count": 0, "lfs_pending_push_count": 0, "planning_authority_commit": AUTHORITY_COMMIT, "planning_authority_is_ancestor": True, "authority_planning_sha256": PLANNING_SHA256, "working_tree_planning_sha256": PLANNING_SHA256, "evidence_root_absent_before_start": True, "report_root_absent_before_start": True, "next_authorization": NEXT_AUTHORIZATION})
    source = signed({**common("p0_static_source_binding"), "status": "PASS", "planning": {"authority_commit": AUTHORITY_COMMIT, "authority_sha256": PLANNING_SHA256, "working_tree_sha256": PLANNING_SHA256, "ancestor_of_source_base": True}, "model": {"name": "x", "checkpoint_sha256": MODEL_SHA256, "canonical_tensor_digest": MODEL_TENSOR_DIGEST, "checkpoint_deserialized": False}, "sources": [record], "sources_digest": "0" * 64})
    access = signed({**common("p0_static_access_audit"), "status": "PASS", "principle": "source-code access != empirical-outcome access", "prohibited_operations": [{"id": "x", "count": 0, "evidence": "x"}], "forbidden_import_ast_scan": {"files_scanned": ["x"], "forbidden_imports": [], "status": "PASS"}, "violations": 0})
    bundle = signed({**common("p0_static_contract_bundle"), "status": "PASS", "files": [record], "file_count": 1, "total_bytes": 1, "files_digest": "0" * 64, "bundle_digest": "0" * 64})
    false_audit = signed({**common("p0_static_false_pass_audit"), "status": "PASS", "attack_count": 28, "rejected_count": 28, "checks": [{"id": "FP01", "attack": "x", "expected_code": "x", "observed_code": "x", "result": "REJECTED"}], "pass_possible_without_all_rejections": False})
    validation = signed({**common("p0_static_validation_report"), "status": "PASS_STATIC", "phase": "final", "generated_at": "x", "checks": [check], "check_count": 1, "failed_count": 0, "false_pass_rejected": 28, "access_violations": 0, "scientific_results_computed": False, "next_authorization": NEXT_AUTHORIZATION})
    inventory = signed({**common("p0_static_whole_root_inventory"), "status": "PASS", "scope": "x", "excluded_self_referential_lifecycle_files": ["x"], "files": [record], "file_count": 1, "total_bytes": 1, "files_digest": "0" * 64})
    pending = signed({**common("p0_static_pending_completion"), "status": "PENDING", "created_at": "x", "source_base_commit": START_HEAD, "next_authorization": "NONE_WHILE_PENDING"})
    pending_receipt = signed({**common("p0_static_pending_transition_receipt"), "status": "TRANSITIONED", "created_at": "x", "transitioned_at": "x", "source_base_commit": START_HEAD, "pending_canonical_digest": "0" * 64, "atomic_transition": "write_pending_fsync_file_os_replace_fsync_directory", "terminal_path": COMPLETION.as_posix(), "next_authorization": NEXT_AUTHORIZATION})
    test_receipt = signed({**common("p0_static_test_receipt"), "status": "PASS", "command": "x", "passed": 1, "failed": 0})
    completion = signed({**common("p0_static_completion"), "status": "PASS_STATIC", "terminal": True, "started_at": "x", "completed_at": "x", "source_base_commit": START_HEAD, "planning_authority_commit": AUTHORITY_COMMIT, "planning_document_sha256": PLANNING_SHA256, "contract_bundle_digest": "0" * 64, "records": {"startup_gate": record, "source_binding": record, "access_audit": record, "contract_bundle": record, "false_pass_audit": record, "validation_report": record, "whole_root_inventory": record, "report": record, "report_validation": record, "handoff": record}, "false_pass_rejected": 28, "access_violations": 0, "p0_a_status": "PASS_STATIC", "p0_b_status": "PASS_STATIC", "p0_terminal_pass": False, "p1_cohort_seal_authorized": False, "independent_cohort_outcome_access_authorized": False, "gpu_production_authorized": False, "p0_c_executed": False, "p1_p4_executed": False, "completion_token": FULL_TOKEN, "next_authorization": NEXT_AUTHORIZATION, "git": {"branch": BRANCH, "publication_commit": "SELF"}})
    return [startup, source, access, bundle, false_audit, validation, inventory, pending, pending_receipt, test_receipt, completion]


def expected_schemas() -> dict[Path, dict[str, Any]]:
    contracts = build_contracts()
    return {CONFIG_DIR / "p0_static.schema.json": _schema("NNS-ViT P0 static contract documents", list(contracts.values())), CONFIG_DIR / "artifact.schema.json": _schema("NNS-ViT P0 static evidence documents", _artifact_templates())}


def _closed_schema_errors(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        if value.get("type") == "object" and value.get("additionalProperties") is not False:
            errors.append(path)
        if value.get("type") == "object" and "properties" not in value:
            errors.append(path + ".properties")
        if "payload" in value.get("properties", {}) and value["properties"]["payload"] == {"type": "object"}:
            errors.append(path + ".payload")
        for key, item in value.items():
            errors.extend(_closed_schema_errors(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_closed_schema_errors(item, f"{path}[{index}]"))
    return errors


def _resolve_ref(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    require(reference.startswith("#/$defs/"), f"unsupported schema ref: {reference}")
    return root["$defs"][reference.split("/")[-1]]


def validate_schema_instance(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], path: str = "$") -> None:
    if "$ref" in schema:
        validate_schema_instance(value, _resolve_ref(root, schema["$ref"]), root, path)
        return
    if "oneOf" in schema:
        successes = 0
        for choice in schema["oneOf"]:
            try:
                validate_schema_instance(value, choice, root, path)
                successes += 1
            except StaticContractError:
                pass
        require(successes == 1, f"schema oneOf mismatch at {path}: {successes} matches")
        return
    expected = schema.get("type")
    checks = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list), "string": lambda x: isinstance(x, str), "integer": lambda x: isinstance(x, int) and not isinstance(x, bool), "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool), "boolean": lambda x: isinstance(x, bool), "null": lambda x: x is None}
    if expected:
        require(expected in checks and checks[expected](value), f"schema type mismatch at {path}: expected {expected}")
    if "const" in schema:
        require(value == schema["const"], f"schema const mismatch at {path}")
    if isinstance(value, dict):
        required = set(schema.get("required", [])); properties = schema.get("properties", {})
        require(required <= set(value), f"schema required keys missing at {path}: {sorted(required-set(value))}")
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(properties), f"schema unknown keys at {path}: {sorted(set(value)-set(properties))}")
        for key, item in value.items():
            if key in properties:
                validate_schema_instance(item, properties[key], root, f"{path}.{key}")
    if isinstance(value, list):
        if "maxItems" in schema:
            require(len(value) <= schema["maxItems"], f"schema maxItems exceeded at {path}")
        if "items" in schema:
            for index, item in enumerate(value):
                validate_schema_instance(item, schema["items"], root, f"{path}[{index}]")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


def initialize(repo: Path) -> dict[str, Any]:
    require(not (repo / COMPLETION).exists(), "terminal root already exists")
    contracts = build_contracts(); schemas = expected_schemas()
    for relative, value in {**contracts, **schemas}.items():
        if (repo / relative).exists():
            replace_json(repo / relative, value)
        else:
            write_new_json(repo / relative, value)
    startup = read_json(repo, ROOT / "startup_gate.json")
    startup_signed = signed(startup)
    replace_json(repo / ROOT / "startup_gate.json", startup_signed)
    if not (repo / PENDING).exists():
        pending = signed({**common("p0_static_pending_completion"), "status": "PENDING", "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"), "source_base_commit": START_HEAD, "next_authorization": "NONE_WHILE_PENDING"})
        write_new_json(repo / PENDING, pending)
    return {"status": "INITIALIZED", "contracts": len(contracts), "schemas": len(schemas), "pending": PENDING.as_posix()}


def _authority_bytes(repo: Path) -> bytes:
    result = subprocess.run(["git", "show", f"{AUTHORITY_COMMIT}:{PLANNING_REPO_REL.as_posix()}"], cwd=repo, capture_output=True)
    require(result.returncode == 0, "authority planning document unavailable")
    return result.stdout


def validate_authority(repo: Path) -> None:
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", AUTHORITY_COMMIT, "HEAD"], cwd=repo).returncode == 0
    require(ancestor, "BLOCKED_PROTOCOL_DRIFT: authority commit is not an ancestor")
    require(sha_bytes(_authority_bytes(repo)) == PLANNING_SHA256, "BLOCKED_PROTOCOL_DRIFT: authority planning digest")
    require(sha_file(_safe_path(repo, PLANNING_REL)) == PLANNING_SHA256, "BLOCKED_PROTOCOL_DRIFT: working planning digest")


def validate_contracts(repo: Path) -> dict[str, Any]:
    validate_authority(repo)
    startup = read_json(repo, ROOT / "startup_gate.json"); validate_signed(startup, "startup gate")
    require(startup["status"] == "PASS" and startup["branch"] == BRANCH, "startup gate status or branch drift")
    require(startup["local_head"] == startup["tracking_ref"] == startup["remote_branch"] == START_HEAD, "startup refs were not identical")
    require(startup["index_clean"] is True and startup["untracked_paths_at_start"] == [], "startup index or untracked state failed")
    require(startup["lfs_pending_commit_count"] == startup["lfs_pending_push_count"] == 0, "startup LFS state failed")
    require(startup["planning_authority_commit"] == AUTHORITY_COMMIT and startup["planning_authority_is_ancestor"] is True, "startup authority binding drift")
    require(startup["authority_planning_sha256"] == startup["working_tree_planning_sha256"] == PLANNING_SHA256, "startup planning digest drift")
    require(startup["evidence_root_absent_before_start"] is True and startup["report_root_absent_before_start"] is True, "startup target roots were not absent")
    expected = build_contracts(); actual: dict[Path, dict[str, Any]] = {}
    for relative, wanted in expected.items():
        value = read_json(repo, relative); validate_signed(value, relative.as_posix())
        require(value == wanted, f"contract semantic drift: {relative}")
        actual[relative] = value
    schemas = expected_schemas()
    for relative, wanted in schemas.items():
        value = read_json(repo, relative)
        require(value == wanted, f"schema drift: {relative}")
        require(not _closed_schema_errors(value), f"open nested schema: {_closed_schema_errors(value)}")
    contract_schema = schemas[CONFIG_DIR / "p0_static.schema.json"]
    for relative, value in actual.items():
        validate_schema_instance(value, contract_schema, contract_schema, relative.as_posix())
    forbidden = set()
    for relative in (Path("scripts/validate_nns_vit_cut_intervention_p0_static.py"), Path("tests/test_nns_vit_cut_intervention_p0_static.py")):
        forbidden.update(_imports(_safe_path(repo, relative)).intersection(FORBIDDEN_IMPORTS))
    require(not forbidden, f"forbidden imports: {sorted(forbidden)}")
    return {"status": "PASS_STATIC", "contract_count": len(actual), "schema_count": len(schemas), "closed_schema_errors": 0, "forbidden_imports": [], "next_authorization": NEXT_AUTHORIZATION}


FALSE_PASS_ATTACKS = (
    ("FP01", "residual identity uses original-node total downstream gradient", "RESIDUAL_IDENTITY_DOUBLE_COUNT"),
    ("FP02", "first residual omits identity or attention edge", "FIRST_RESIDUAL_INCOMPLETE"),
    ("FP03", "second residual omits carry or MLP edge", "SECOND_RESIDUAL_INCOMPLETE"),
    ("FP04", "head cut uses CLS-only replacement", "HEAD_NOT_ALL_TOKEN"),
    ("FP05", "head hook is at wrong position", "HEAD_HOOK_POSITION"),
    ("FP06", "routing/value labeled ordinary complete edge cut", "DIFFERENTIAL_PATH_MISLABELED"),
    ("FP07", "routing or Value omitted while closure says PASS", "ATTENTION_DIFFERENTIAL_INCOMPLETE"),
    ("FP08", "Q or K omitted while routing says PASS", "ROUTING_DIFFERENTIAL_INCOMPLETE"),
    ("FP09", "interaction sign reversed", "INTERACTION_SIGN"),
    ("FP10", "normalized zero treated as universal RGB 0.5", "REFERENCE_NOT_CONFIG_DERIVED"),
    ("FP11", "preprocessing field drift", "PREPROCESSING_DRIFT"),
    ("FP12", "algebraic reconstruction impersonates convergence", "NUMERICAL_CONTRACT_CONFLATION"),
    ("FP13", "successive-order drift marked PASS", "SUCCESSIVE_ORDER_FALSE_PASS"),
    ("FP14", "small target delta hidden by relative error", "SMALL_TARGET_DELTA_FALSE_PASS"),
    ("FP15", "fewer than 18 images marked supported", "INSUFFICIENT_IMAGES_FALSE_PASS"),
    ("FP16", "144 heads treated as independent n", "PSEUDOREPLICATION"),
    ("FP17", "undefined Spearman dropped or zero-filled", "UNTYPED_SPEARMAN_NA"),
    ("FP18", "absolute mass predicts direction", "SIGNED_ABSOLUTE_ROLE_SWAP"),
    ("FP19", "strongest baseline selected per image", "PER_IMAGE_BASELINE_ORACLE"),
    ("FP20", "random ranking enters mandatory baseline set", "RANDOM_BASELINE_MANDATORY"),
    ("FP21", "Family B corrects only favorable branch", "FAMILY_B_HOLM_SPLIT"),
    ("FP22", "non-positive baseline gain still GO", "BASELINE_GAIN_FALSE_GO"),
    ("FP23", "one supported family still GO", "ONE_FAMILY_FALSE_GO"),
    ("FP24", "resource projection missing or above cap authorizes next stage", "RESOURCE_FALSE_AUTHORIZATION"),
    ("FP25", "open nested schema or unknown field accepted", "OPEN_SCHEMA_FALSE_PASS"),
    ("FP26", "duplicate key nonfinite symlink or root escape accepted", "UNSAFE_INPUT_FALSE_PASS"),
    ("FP27", "completion lacks binding audit or inventory", "INCOMPLETE_COMPLETION_FALSE_PASS"),
    ("FP28", "static completion authorizes cohort seal or GPU production", "AUTHORIZATION_ESCALATION"),
)


def base_attack_state() -> dict[str, Any]:
    return {"residual_gradient": "adder_output_shared_adjoint", "first_edges": ["identity", "attention"], "second_edges": ["carry", "mlp"], "head_tokens": 197, "head_hook": "post_attention_dropout_pre_head_concat_pre_output_projection", "differential_type": "differential_path_not_complete_edge_cut", "attention_terms": ["routing", "value"], "routing_terms": ["query", "key", "bias"], "interaction": "e_R+e_V-e_R+V", "reference_derived": True, "preprocessing_exact": True, "numerical_contracts_distinct": True, "successive_order_pass": True, "successive_order_within_tolerance": True, "small_target_typed_na": True, "valid_images": 18, "family_status": "SUPPORTED_PREDICTIVE_RELATION", "independent_unit": "image", "spearman_na": "typed_retained", "direction_predictor": "signed_contribution", "baseline_selection": "cohort_statistic_inside_synchronized_bootstrap", "random_is_mandatory": False, "family_b_holm": "B1-B10_joint", "baseline_gain_positive": True, "go": True, "supported_families": 2, "resource_projection": "required_within_caps_before_p1", "next_authorization": NEXT_AUTHORIZATION, "schemas_closed": True, "unknown_fields_rejected": True, "strict_inputs": True, "completion_records": ["binding", "audit", "inventory"], "p1_authorized": False, "gpu_production_authorized": False}


def attack_reasons(state: Mapping[str, Any]) -> tuple[str, ...]:
    checks = [
        (state["residual_gradient"] == "adder_output_shared_adjoint", "RESIDUAL_IDENTITY_DOUBLE_COUNT"), (state["first_edges"] == ["identity", "attention"], "FIRST_RESIDUAL_INCOMPLETE"), (state["second_edges"] == ["carry", "mlp"], "SECOND_RESIDUAL_INCOMPLETE"), (state["head_tokens"] == 197, "HEAD_NOT_ALL_TOKEN"), (state["head_hook"] == "post_attention_dropout_pre_head_concat_pre_output_projection", "HEAD_HOOK_POSITION"), (state["differential_type"] == "differential_path_not_complete_edge_cut", "DIFFERENTIAL_PATH_MISLABELED"), (state["attention_terms"] == ["routing", "value"], "ATTENTION_DIFFERENTIAL_INCOMPLETE"), (state["routing_terms"] == ["query", "key", "bias"], "ROUTING_DIFFERENTIAL_INCOMPLETE"), (state["interaction"] == "e_R+e_V-e_R+V", "INTERACTION_SIGN"), (state["reference_derived"] is True, "REFERENCE_NOT_CONFIG_DERIVED"), (state["preprocessing_exact"] is True, "PREPROCESSING_DRIFT"), (state["numerical_contracts_distinct"] is True, "NUMERICAL_CONTRACT_CONFLATION"), (not state["successive_order_pass"] or state["successive_order_within_tolerance"], "SUCCESSIVE_ORDER_FALSE_PASS"), (state["small_target_typed_na"] is True, "SMALL_TARGET_DELTA_FALSE_PASS"), (state["family_status"] != "SUPPORTED_PREDICTIVE_RELATION" or state["valid_images"] >= 18, "INSUFFICIENT_IMAGES_FALSE_PASS"), (state["independent_unit"] == "image", "PSEUDOREPLICATION"), (state["spearman_na"] == "typed_retained", "UNTYPED_SPEARMAN_NA"), (state["direction_predictor"] == "signed_contribution", "SIGNED_ABSOLUTE_ROLE_SWAP"), (state["baseline_selection"] == "cohort_statistic_inside_synchronized_bootstrap", "PER_IMAGE_BASELINE_ORACLE"), (state["random_is_mandatory"] is False, "RANDOM_BASELINE_MANDATORY"), (state["family_b_holm"] == "B1-B10_joint", "FAMILY_B_HOLM_SPLIT"), (not state["go"] or state["baseline_gain_positive"], "BASELINE_GAIN_FALSE_GO"), (not state["go"] or state["supported_families"] >= 2, "ONE_FAMILY_FALSE_GO"), (state["next_authorization"] != NEXT_AUTHORIZATION or state["resource_projection"] == "required_within_caps_before_p1", "RESOURCE_FALSE_AUTHORIZATION"), (state["schemas_closed"] and state["unknown_fields_rejected"], "OPEN_SCHEMA_FALSE_PASS"), (state["strict_inputs"] is True, "UNSAFE_INPUT_FALSE_PASS"), (state["completion_records"] == ["binding", "audit", "inventory"], "INCOMPLETE_COMPLETION_FALSE_PASS"), (state["p1_authorized"] is False and state["gpu_production_authorized"] is False, "AUTHORIZATION_ESCALATION"),
    ]
    return tuple(code for passed, code in checks if not passed)


def mutate_attack(state: dict[str, Any], attack_id: str) -> None:
    mutations = {"FP01": ("residual_gradient", "original_node_total_downstream"), "FP02": ("first_edges", ["attention"]), "FP03": ("second_edges", ["mlp"]), "FP04": ("head_tokens", 1), "FP05": ("head_hook", "post_output_projection"), "FP06": ("differential_type", "complete_edge_cut"), "FP07": ("attention_terms", ["routing"]), "FP08": ("routing_terms", ["query", "bias"]), "FP09": ("interaction", "e_R+V-e_R-e_V"), "FP10": ("reference_derived", False), "FP11": ("preprocessing_exact", False), "FP12": ("numerical_contracts_distinct", False), "FP13": ("successive_order_within_tolerance", False), "FP14": ("small_target_typed_na", False), "FP15": ("valid_images", 17), "FP16": ("independent_unit", "block_head"), "FP17": ("spearman_na", "zero_imputed"), "FP18": ("direction_predictor", "absolute_mass"), "FP19": ("baseline_selection", "per_image_oracle"), "FP20": ("random_is_mandatory", True), "FP21": ("family_b_holm", "best_branch_only"), "FP22": ("baseline_gain_positive", False), "FP23": ("supported_families", 1), "FP24": ("resource_projection", "missing_or_over_cap_but_authorized"), "FP25": ("schemas_closed", False), "FP26": ("strict_inputs", False), "FP27": ("completion_records", ["binding", "audit"]), "FP28": ("gpu_production_authorized", True)}
    key, value = mutations[attack_id]; state[key] = value


def build_false_pass_audit() -> dict[str, Any]:
    require(attack_reasons(base_attack_state()) == (), "valid attack baseline rejected")
    rows = []
    for attack_id, description, code in FALSE_PASS_ATTACKS:
        state = base_attack_state(); mutate_attack(state, attack_id); observed = attack_reasons(state)
        require(observed == (code,), f"false-pass mutation {attack_id} did not isolate expected code: {observed}")
        rows.append({"id": attack_id, "attack": description, "expected_code": code, "observed_code": observed[0], "result": "REJECTED"})
    return signed({**common("p0_static_false_pass_audit"), "status": "PASS", "attack_count": len(rows), "rejected_count": len(rows), "checks": rows, "pass_possible_without_all_rejections": False})


def file_record(repo: Path, relative: Path) -> dict[str, Any]:
    path = _safe_path(repo, relative); data = path.read_bytes()
    try:
        value = strict_json_bytes(data, relative.as_posix())
        content_digest = canonical_digest(value)
    except StaticContractError:
        content_digest = sha_bytes(data)
    return {"path": relative.as_posix(), "sha256": sha_bytes(data), "bytes": len(data), "canonical_content_digest": content_digest}


def contract_bundle(repo: Path) -> dict[str, Any]:
    files = [file_record(repo, path) for path in SOURCE_PATHS]
    return signed({**common("p0_static_contract_bundle"), "status": "PASS", "files": files, "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files), "files_digest": canonical_digest(files), "bundle_digest": canonical_digest({"protocol_id": PROTOCOL_ID, "files": files})})


def source_binding(repo: Path) -> dict[str, Any]:
    validate_authority(repo)
    source_paths = [PLANNING_REL, Path("config/stage_d_recovery_v2_binding_20260904.json"), Path("config/stage_x_x2.json")]
    sources = [file_record(repo, path) for path in source_paths]
    return signed({**common("p0_static_source_binding"), "status": "PASS", "planning": {"authority_commit": AUTHORITY_COMMIT, "authority_sha256": sha_bytes(_authority_bytes(repo)), "working_tree_sha256": sha_file(repo / PLANNING_REL), "ancestor_of_source_base": True}, "model": {"name": "vit_base_patch16_224.augreg2_in21k_ft_in1k", "checkpoint_sha256": MODEL_SHA256, "canonical_tensor_digest": MODEL_TENSOR_DIGEST, "checkpoint_deserialized": False}, "sources": sources, "sources_digest": canonical_digest(sources)})


def access_audit(repo: Path) -> dict[str, Any]:
    scanned = ["scripts/validate_nns_vit_cut_intervention_p0_static.py", "tests/test_nns_vit_cut_intervention_p0_static.py"]
    forbidden = sorted(set().union(*(_imports(repo / path) for path in scanned)).intersection(FORBIDDEN_IMPORTS))
    rows = [{"id": identifier, "count": 0, "evidence": evidence} for identifier, evidence in (
        ("jpeg_png_pixel_decodes", "no image-library import and no pixel reader in static package"), ("npy_npz_tensor_map_activation_numeric_reads", "no scientific-array import or reader in static package"), ("new_cohort_outcome_reads", "target and cohort selection remain algorithm-only"), ("checkpoint_tensor_deserializations", "model identity read as metadata only"), ("model_loads", "static validator has no model-loading surface"), ("forbidden_library_imports", "AST scan result is empty"), ("cuda_gpu_accelerator_probes", "no accelerator command or API in lifecycle"), ("p0_c_executions", "P0-C is plan-only"), ("p1_p2_p3_p4_executions", "downstream stages are unauthorized"), ("cohort_membership_y_star_source_object_computations", "selection algorithm frozen but not executed"), ("sealed_root_modifications", "all writes restricted to new P0 static paths"))]
    return signed({**common("p0_static_access_audit"), "status": "PASS", "principle": "source-code access != empirical-outcome access", "prohibited_operations": rows, "forbidden_import_ast_scan": {"files_scanned": scanned, "forbidden_imports": forbidden, "status": "PASS" if not forbidden else "FAIL"}, "violations": sum(row["count"] for row in rows) + len(forbidden)})


def validation_report(repo: Path, phase: str) -> dict[str, Any]:
    contract_result = validate_contracts(repo); audit = build_false_pass_audit(); access = access_audit(repo)
    checks = [
        {"id": "authority_binding", "status": "PASS", "detail": PLANNING_SHA256}, {"id": "closed_contract_schemas", "status": "PASS", "detail": f"{contract_result['contract_count']} documents; zero open objects"}, {"id": "math_edge_and_differential_contracts", "status": "PASS", "detail": "component/head/residual/routing/value/Q/K/interaction frozen"}, {"id": "target_reference_cohort_algorithm", "status": "PASS", "detail": "metadata-only algorithm frozen; not executed"}, {"id": "statistics_and_baselines", "status": "PASS", "detail": "H1-H5 B1-B10 R1-R3 and image-level inference frozen"}, {"id": "three_numeric_contracts", "status": "PASS", "detail": "algebraic completeness convergence remain distinct"}, {"id": "resource_plan", "status": "PASS", "detail": "owner caps frozen; projection not fabricated"}, {"id": "false_pass_audit", "status": "PASS", "detail": f"{audit['rejected_count']}/{audit['attack_count']} rejected"}, {"id": "access_audit", "status": "PASS", "detail": f"{access['violations']} violations"}]
    return signed({**common("p0_static_validation_report"), "status": "PASS_STATIC", "phase": phase, "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"), "checks": checks, "check_count": len(checks), "failed_count": 0, "false_pass_rejected": audit["rejected_count"], "access_violations": access["violations"], "scientific_results_computed": False, "next_authorization": NEXT_AUTHORIZATION})


def _report_markdown(bundle_digest: str) -> str:
    return f"""# NNS-ViT P0-A/B static package report

Status: `PASS_STATIC` for P0-A and P0-B only. This is not full P0 PASS.

The package freezes the complete-cut mathematical contracts, matched interventions, target/reference/cohort algorithms, H/B/R statistical plan, mandatory baselines, three distinct numerical contracts, typed failures, closed schemas, access boundary, and the P0-C resource-pilot plan. It computes no y*, cohort membership, prediction, logit, attribution, intervention effect, or scientific result.

- False-pass audit: 28/28 mutations rejected.
- Access audit: zero prohibited operations.
- Contract bundle digest: `{bundle_digest}`.
- P0-C projection: intentionally not measured in this static stage.
- Completion token: `{FULL_TOKEN}`.
- Only next authorization: `{NEXT_AUTHORIZATION}`.

P0-C and P1-P4 were not executed. Cohort seal, independent-cohort outcome access, formal attribution/intervention, and GPU production remain unauthorized.
"""


def _handoff_markdown(bundle_digest: str) -> str:
    return f"""# NNS-ViT cut/intervention P0 static handoff

`NNS_VIT_CUT_INTERVENTION_PREDICTION_V1` completed P0-A/B as `PASS_STATIC`. This handoff is not a full P0 PASS and contains no scientific outcome.

The sealed bundle digest is `{bundle_digest}`. The only next token is `{NEXT_AUTHORIZATION}`: a future P0-C engineering/resource pilot may use synthetic fixtures and one explicitly registered prior-access image under the frozen caps. P0-C must measure P95 costs, no-op parity, hook identity, reconstruction and convergence before any P1 authorization can exist.

Still forbidden: computing y* or the 20 source-object IDs, cohort sealing, independent image access, P1-P4, formal attribution/intervention, and GPU production. P0-C results must be sealed prospectively and cannot rewrite this root.
"""


def finalize(repo: Path, test_command: str, passed: int, failed: int) -> dict[str, Any]:
    require((repo / PENDING).is_file() and not (repo / COMPLETION).exists(), "pending lifecycle state missing or terminal exists")
    require(passed > 0 and failed == 0, "focused tests did not pass")
    validate_contracts(repo)
    payloads = {ROOT / "source_binding.json": source_binding(repo), ROOT / "access_audit.json": access_audit(repo), ROOT / "contract_bundle.json": contract_bundle(repo), ROOT / "false_pass_audit.json": build_false_pass_audit()}
    for relative, value in payloads.items():
        write_new_json(repo / relative, value)
    report = validation_report(repo, "final")
    write_new_json(repo / ROOT / "validation_report.json", report)
    write_new_json(repo / REPORT_ROOT / "validation_report.json", report)
    write_new_text(repo / REPORT_ROOT / "report.md", _report_markdown(payloads[ROOT / "contract_bundle.json"]["bundle_digest"]))
    write_new_text(repo / HANDOFF, _handoff_markdown(payloads[ROOT / "contract_bundle.json"]["bundle_digest"]))
    test_receipt = signed({**common("p0_static_test_receipt"), "status": "PASS", "command": test_command, "passed": passed, "failed": failed})
    write_new_json(repo / ROOT / "focused_test_receipt.json", test_receipt)
    return {"status": "PASS_FINAL_VALIDATION", "false_pass_rejected": 28, "access_violations": 0, "next_authorization": "NONE_UNTIL_SEAL"}


def _inventory(repo: Path) -> dict[str, Any]:
    root = repo / ROOT
    excluded = {"whole_root_inventory.json", "completion/p0_static.pending.json", "completion/p0_static.json"}
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative_inside = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"symlink in evidence root: {relative_inside}")
        if relative_inside in excluded:
            continue
        files.append(file_record(repo, ROOT / relative_inside))
    return signed({**common("p0_static_whole_root_inventory"), "status": "PASS", "scope": "all regular files under formal root except self-referential inventory and lifecycle completion files", "excluded_self_referential_lifecycle_files": sorted(excluded), "files": files, "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files), "files_digest": canonical_digest(files)})


def _validate_artifact(repo: Path, relative: Path) -> dict[str, Any]:
    value = read_json(repo, relative); validate_signed(value, relative.as_posix())
    schema = read_json(repo, CONFIG_DIR / "artifact.schema.json")
    validate_schema_instance(value, schema, schema, relative.as_posix())
    return value


def seal(repo: Path) -> dict[str, Any]:
    require((repo / PENDING).is_file(), "pending completion missing")
    require(not (repo / COMPLETION).exists(), "terminal root cannot be rewritten")
    pending = _validate_artifact(repo, PENDING); require(pending["status"] == "PENDING", "invalid pending state")
    report = _validate_artifact(repo, ROOT / "validation_report.json")
    require(report["status"] == "PASS_STATIC" and report["false_pass_rejected"] == 28 and report["access_violations"] == 0, "final validation not sealable")
    for relative in (ROOT / "source_binding.json", ROOT / "access_audit.json", ROOT / "contract_bundle.json", ROOT / "false_pass_audit.json"):
        _validate_artifact(repo, relative)
    inventory = _inventory(repo); write_new_json(repo / ROOT / "whole_root_inventory.json", inventory)
    bundle = read_json(repo, ROOT / "contract_bundle.json")
    records = {"startup_gate": file_record(repo, ROOT / "startup_gate.json"), "source_binding": file_record(repo, ROOT / "source_binding.json"), "access_audit": file_record(repo, ROOT / "access_audit.json"), "contract_bundle": file_record(repo, ROOT / "contract_bundle.json"), "false_pass_audit": file_record(repo, ROOT / "false_pass_audit.json"), "validation_report": file_record(repo, ROOT / "validation_report.json"), "whole_root_inventory": file_record(repo, ROOT / "whole_root_inventory.json"), "report": file_record(repo, REPORT_ROOT / "report.md"), "report_validation": file_record(repo, REPORT_ROOT / "validation_report.json"), "handoff": file_record(repo, HANDOFF)}
    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    terminal = signed({**common("p0_static_completion"), "status": "PASS_STATIC", "terminal": True, "started_at": pending["created_at"], "completed_at": now, "source_base_commit": pending["source_base_commit"], "planning_authority_commit": AUTHORITY_COMMIT, "planning_document_sha256": PLANNING_SHA256, "contract_bundle_digest": bundle["bundle_digest"], "records": records, "false_pass_rejected": 28, "access_violations": 0, "p0_a_status": "PASS_STATIC", "p0_b_status": "PASS_STATIC", "p0_terminal_pass": False, "p1_cohort_seal_authorized": False, "independent_cohort_outcome_access_authorized": False, "gpu_production_authorized": False, "p0_c_executed": False, "p1_p4_executed": False, "completion_token": FULL_TOKEN, "next_authorization": NEXT_AUTHORIZATION, "git": {"branch": BRANCH, "publication_commit": "SELF"}})
    terminal_tmp = repo / PENDING.with_name(".p0_static.terminal.tmp")
    write_new_json(terminal_tmp, terminal)
    os.replace(terminal_tmp, repo / PENDING)
    _fsync_parent(repo / PENDING)
    os.replace(repo / PENDING, repo / COMPLETION)
    _fsync_parent(repo / COMPLETION)
    pending_receipt = signed({**common("p0_static_pending_transition_receipt"), "status": "TRANSITIONED", "created_at": pending["created_at"], "transitioned_at": now, "source_base_commit": pending["source_base_commit"], "pending_canonical_digest": pending["canonical_digest"], "atomic_transition": "write_pending_fsync_file_os_replace_fsync_directory", "terminal_path": COMPLETION.as_posix(), "next_authorization": NEXT_AUTHORIZATION})
    write_new_json(repo / PENDING, pending_receipt)
    return validate_seal(repo)


def validate_seal(repo: Path) -> dict[str, Any]:
    completion = _validate_artifact(repo, COMPLETION); pending_receipt = _validate_artifact(repo, PENDING)
    require(completion["status"] == "PASS_STATIC" and completion["terminal"] is True, "invalid terminal completion")
    require(pending_receipt["status"] == "TRANSITIONED", "missing pending transition receipt")
    require(completion["next_authorization"] == NEXT_AUTHORIZATION and completion["completion_token"] == FULL_TOKEN, "authorization drift")
    require(completion["p0_terminal_pass"] is False and completion["p1_cohort_seal_authorized"] is False and completion["independent_cohort_outcome_access_authorized"] is False and completion["gpu_production_authorized"] is False, "static completion over-authorized")
    validate_contracts(repo)
    bundle = contract_bundle(repo); require(bundle["bundle_digest"] == completion["contract_bundle_digest"], "bundle drift after completion")
    for name, record in completion["records"].items():
        require(file_record(repo, Path(record["path"])) == record, f"completion record drift: {name}")
    inventory = _validate_artifact(repo, ROOT / "whole_root_inventory.json"); rebuilt = _inventory(repo)
    for key in ("files", "file_count", "total_bytes", "files_digest"):
        require(inventory[key] == rebuilt[key], f"inventory drift: {key}")
    audit = _validate_artifact(repo, ROOT / "false_pass_audit.json"); require(audit["rejected_count"] == audit["attack_count"] == 28, "false-pass audit incomplete")
    access = _validate_artifact(repo, ROOT / "access_audit.json"); require(access["violations"] == 0 and all(row["count"] == 0 for row in access["prohibited_operations"]), "access audit violation")
    return {"status": "PASS_STATIC", "p0_terminal_pass": False, "completion_token": FULL_TOKEN, "next_authorization": NEXT_AUTHORIZATION, "false_pass_rejected": 28, "access_violations": 0, "contract_bundle_digest": completion["contract_bundle_digest"], "completion_sha256": sha_file(repo / COMPLETION), "inventory_sha256": sha_file(repo / ROOT / "whole_root_inventory.json")}


def _lfs_clean(repo: Path) -> bool:
    result = subprocess.run(["git", "lfs", "status"], cwd=repo, text=True, capture_output=True)
    require(result.returncode == 0, "git lfs status failed")
    sections = ("Git LFS objects to be pushed", "Git LFS objects to be committed", "Git LFS objects not staged for commit")
    current = False
    for line in result.stdout.splitlines():
        if any(line.startswith(section) for section in sections):
            current = True
            continue
        if line.startswith("Git LFS"):
            current = False
        elif current and line.strip():
            return False
    return True


def validate_publication(repo: Path) -> dict[str, Any]:
    branch = git(repo, "symbolic-ref", "--short", "HEAD"); head = git(repo, "rev-parse", "HEAD"); tracking = git(repo, "rev-parse", "@{upstream}")
    remote_output = git(repo, "ls-remote", "origin", f"refs/heads/{BRANCH}"); remote = remote_output.split()[0] if remote_output else ""
    status = git(repo, "status", "--branch", "--porcelain=v2").splitlines(); clean = not any(line and not line.startswith("#") for line in status)
    require(branch == BRANCH, "publication branch drift"); require(head == tracking == remote, "local/tracking/remote mismatch"); require(clean, "publication worktree not clean"); require(_lfs_clean(repo), "publication LFS not clean")
    sealed = validate_seal(repo)
    return {"status": "PASS_PUBLICATION", "head": head, "tracking_ref": tracking, "remote_branch": remote, "remote_contains_completion": True, "worktree_clean": True, "lfs_clean": True, "completion_token": sealed["completion_token"], "next_authorization": sealed["next_authorization"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init"); commands.add_parser("draft"); commands.add_parser("false-pass-audit")
    final = commands.add_parser("final"); final.add_argument("--test-command", required=True); final.add_argument("--passed", type=int, required=True); final.add_argument("--failed", type=int, required=True)
    commands.add_parser("seal"); commands.add_parser("validate-seal"); commands.add_parser("validate-publication")
    args = parser.parse_args(argv); repo = args.repo.resolve()
    if args.command == "init": result = initialize(repo)
    elif args.command == "draft": result = validation_report(repo, "draft")
    elif args.command == "false-pass-audit": result = build_false_pass_audit()
    elif args.command == "final": result = finalize(repo, args.test_command, args.passed, args.failed)
    elif args.command == "seal": result = seal(repo)
    elif args.command == "validate-seal": result = validate_seal(repo)
    else: result = validate_publication(repo)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StaticContractError as exc:
        print(json.dumps({"status": "BLOCKED_STATIC_VALIDATION", "error": str(exc), "next_authorization": "NONE"}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
