"""CPU-only C0 validator for the frozen Stage C data and protocol contracts.

This is intentionally not the production Stage C parser promised by C1.  It
checks the checked-in C0 instance and its input assets without loading a model
or computing an attribution.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from vitnns.constants import (
    IMAGENET_SYNSET_ORDER_SHA256,
    OFFICIAL_DEVKIT_SHA256,
)
from vitnns.dataset import build_manifest
from vitnns.io_utils import sha256_file


if not __debug__:
    raise RuntimeError(
        "The C0 validator must not run with Python -O because its asset assertions "
        "are an acceptance boundary"
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
CONFIG_PATH = REPO_ROOT / "config/stage_c.json"
CONFIG_SCHEMA_PATH = REPO_ROOT / "config/stage_c.schema.json"
REGISTRY_SCHEMA_PATH = REPO_ROOT / "config/stage_c_registry.schema.json"
SELECTION_SCHEMA_PATH = REPO_ROOT / "config/stage_c_selection.schema.json"
REGISTRY_PATH = REPO_ROOT / "manifests/stage_c_sample_registry.csv"
REGISTRY_METADATA_PATH = REPO_ROOT / "manifests/stage_c_sample_registry_metadata.json"
SOURCE_MANIFEST_PATH = REPO_ROOT / "manifests/stage_c_flat_val_500/manifest.csv"
SOURCE_METADATA_PATH = REPO_ROOT / "manifests/stage_c_flat_val_500/manifest_metadata.json"
DEVKIT_PATH = REPO_ROOT / "artifacts/sources/ILSVRC2012_devkit_t12.tar.gz"
DATA_ROOT = WORKSPACE_ROOT / "49_vit /datas/ILSVRC2012_val"
STAGE_B_ROOT = REPO_ROOT / "outputs/stage_b_formal_30732c9"

FROZEN_IMAGE_IDS = [
    "ILSVRC2012_val_00000079",
    "ILSVRC2012_val_00002607",
    "ILSVRC2012_val_00005279",
    "ILSVRC2012_val_00007489",
    "ILSVRC2012_val_00009888",
    "ILSVRC2012_val_00012573",
    "ILSVRC2012_val_00014787",
    "ILSVRC2012_val_00018380",
    "ILSVRC2012_val_00020970",
    "ILSVRC2012_val_00023229",
    "ILSVRC2012_val_00025927",
    "ILSVRC2012_val_00028908",
    "ILSVRC2012_val_00031099",
    "ILSVRC2012_val_00033512",
    "ILSVRC2012_val_00035680",
    "ILSVRC2012_val_00038565",
    "ILSVRC2012_val_00041511",
    "ILSVRC2012_val_00044187",
    "ILSVRC2012_val_00047079",
    "ILSVRC2012_val_00049970",
]
CALIBRATION_IDS = {
    "ILSVRC2012_val_00000079",
    "ILSVRC2012_val_00009888",
    "ILSVRC2012_val_00023229",
    "ILSVRC2012_val_00035680",
    "ILSVRC2012_val_00049970",
}
AUDIT_IDS = {
    "ILSVRC2012_val_00000079",
    "ILSVRC2012_val_00049970",
}
SOURCE_INDICES = [
    0, 26, 52, 78, 105, 131, 157, 183, 210, 236,
    262, 288, 315, 341, 367, 393, 420, 446, 472, 499,
]
REGISTRY_COLUMNS = [
    "schema_version", "sample_id", "image_id", "relative_path",
    "image_sha256", "size_bytes", "width", "height", "official_wnid",
    "imagenet1k_index", "label_source", "role", "audit",
    "selection_order", "source_selection_index", "selection_rule",
]
SOURCE_COLUMNS = [
    "sample_id", "relative_path", "image_sha256", "size_bytes",
    "width", "height", "wnid", "imagenet1k_index", "label_source",
]
METHOD_IDS = {
    "direct_raw_cls_attention_per_head",
    "conditional_fixed_clean_attention_head_direction_value_path_ig",
    "matched_fixed_clean_attention_patch_token_contribution",
    "direct_raw_cls_attention_last_block_head_mean",
    "attention_rollout_mean_head_residual_all_blocks",
    "vit_grad_cam_block11_norm1",
    "full_model_class_logit_ig",
}
PERTURBATION_IDS = {
    "translate_right_down_4",
    "center_crop_216_resize_224",
    "gaussian_rgb_std_0_01",
}
DECISIONS = {
    "GO", "NO_GO_VALIDITY", "NO_GO_NOVELTY",
    "NO_GO_COST", "INCONCLUSIVE",
}
SELECTION_RULE = (
    "legacy_sorted_500_linspace20_then_fixed_role_positions_0_4_9_14_19_"
    "audit_0_19"
)
CALIBRATION_SAMPLE_IDS = [
    "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00009888.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00023229.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00035680.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00049970.JPEG",
]


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_constant,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_relative_path(base: Path, raw: str, context: str) -> Path:
    pure = PurePosixPath(raw)
    _require(not pure.is_absolute(), f"{context} must be relative")
    _require(".." not in pure.parts, f"{context} must not contain '..'")
    target = (base / Path(*pure.parts)).resolve()
    _require(target.is_relative_to(base.resolve()), f"{context} escapes its base")
    return target


def _contract_paths(config: dict[str, Any]) -> dict[str, Path]:
    registry_cfg = config["registry"]
    registry_metadata_path = _safe_relative_path(
        REPO_ROOT, registry_cfg["metadata_path"], "registry.metadata_path"
    )
    metadata = strict_json_load(registry_metadata_path)
    return {
        "registry": _safe_relative_path(REPO_ROOT, registry_cfg["path"], "registry.path"),
        "registry_metadata": registry_metadata_path,
        "source_manifest": _safe_relative_path(
            REPO_ROOT, registry_cfg["source_manifest_path"], "registry.source_manifest_path"
        ),
        "source_metadata": _safe_relative_path(
            REPO_ROOT, registry_cfg["source_manifest_metadata_path"],
            "registry.source_manifest_metadata_path",
        ),
        "devkit": _safe_relative_path(
            REPO_ROOT, metadata["official_labels"]["devkit_relative_path"],
            "registry_metadata.official_labels.devkit_relative_path",
        ),
        "data_root": _safe_relative_path(
            WORKSPACE_ROOT, metadata["dataset"]["runtime_root_relative_to_workspace"],
            "registry_metadata.dataset.runtime_root_relative_to_workspace",
        ),
    }


def _exact_keys(value: Any, expected: set[str], context: str) -> None:
    assert isinstance(value, dict), f"{context} must be an object"
    actual = set(value)
    assert actual == expected, (
        f"{context} key mismatch: missing={sorted(expected-actual)}, "
        f"unknown={sorted(actual-expected)}"
    )


def _finite_numbers(value: Any, context: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        assert value == value and value not in (float("inf"), float("-inf")), context
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _finite_numbers(child, f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_numbers(child, f"{context}.{key}")
        return
    raise AssertionError(f"Unsupported JSON value at {context}: {type(value)}")


def _positive_number(value: Any, context: str) -> None:
    assert not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0, context


def _validate_config(config: dict[str, Any]) -> None:
    top_keys = {
        "$schema", "schema_version", "document_type", "stage", "protocol",
        "registry", "roles", "methods", "targets", "layers_heads",
        "projection", "metrics", "fidelity", "perturbations", "retention",
        "budget", "selection", "statistics", "decision_gates",
    }
    _exact_keys(config, top_keys, "config")
    assert config["$schema"] == "stage_c.schema.json"
    assert config["schema_version"] == 1 and config["stage"] == "C"
    assert config["document_type"] == "stage_c_protocol_config"
    _finite_numbers(config)

    _exact_keys(config["protocol"], {"path", "version", "amendment_policy"}, "protocol")
    registry_keys = {
        "path", "sha256", "metadata_path", "metadata_sha256",
        "source_manifest_path", "source_manifest_sha256",
        "source_manifest_metadata_path", "source_manifest_metadata_sha256",
        "official_devkit_sha256", "expected_counts",
    }
    _exact_keys(config["registry"], registry_keys, "registry")
    _exact_keys(
        config["registry"]["expected_counts"],
        {"total", "calibration", "holdout", "audit"},
        "registry.expected_counts",
    )
    assert config["registry"]["expected_counts"] == {
        "total": 20, "calibration": 5, "holdout": 15, "audit": 2,
    }
    _exact_keys(config["roles"], {
        "analysis_role_field", "allowed_analysis_roles", "audit_field",
        "audit_is_calibration_subset", "calibration_confirmatory_statistics",
        "holdout_decides_go_no_go", "all_20_summary",
        "retain_misclassified", "retain_failed_and_degenerate_rows",
    }, "roles")
    assert config["roles"]["allowed_analysis_roles"] == ["calibration", "holdout"]
    assert config["roles"]["audit_is_calibration_subset"] is True
    assert config["roles"]["calibration_confirmatory_statistics"] is False

    method_keys = {
        "id", "exact_name", "comparison_line", "explanation_target_ref",
        "fidelity_target_ref", "class_agnostic", "fidelity_eligible",
        "structural_reference_only", "native_map", "projection_ref",
    }
    methods = config["methods"]
    assert isinstance(methods, list) and len(methods) == 7
    for index, method in enumerate(methods):
        _exact_keys(method, method_keys, f"methods[{index}]")
    by_method = {item["id"]: item for item in methods}
    assert len(by_method) == len(methods) and set(by_method) == METHOD_IDS
    nns = by_method["conditional_fixed_clean_attention_head_direction_value_path_ig"]
    matched = by_method["matched_fixed_clean_attention_patch_token_contribution"]
    per_head_raw = by_method["direct_raw_cls_attention_per_head"]
    assert nns["explanation_target_ref"] == matched["explanation_target_ref"] == (
        "fixed_clean_attention_cls_head_direction"
    )
    assert nns["fidelity_target_ref"] == matched["fidelity_target_ref"] == (
        "fixed_clean_attention_cls_head_direction"
    )
    assert per_head_raw["structural_reference_only"] is True
    assert per_head_raw["fidelity_eligible"] is False
    assert per_head_raw["explanation_target_ref"] is None
    for method_id in (
        "direct_raw_cls_attention_last_block_head_mean",
        "attention_rollout_mean_head_residual_all_blocks",
    ):
        assert by_method[method_id]["class_agnostic"] is True
        assert by_method[method_id]["explanation_target_ref"] is None
        assert by_method[method_id]["fidelity_target_ref"] == "clean_predicted_class_raw_logit"
    assert "attention attribution" not in nns["exact_name"].lower()

    targets = config["targets"]
    _exact_keys(targets, {"baseline", "ig", "fixed_clean_attention_cls_head_direction", "clean_predicted_class_raw_logit", "repeatability"}, "targets")
    _exact_keys(targets["baseline"], {"id", "normalized_value", "canonical_rgb_value", "channels"}, "targets.baseline")
    _exact_keys(targets["ig"], {"quadrature", "steps_schedule", "step_chunk_size", "atol", "rtol"}, "targets.ig")
    _exact_keys(targets["fixed_clean_attention_cls_head_direction"], {
        "query_token", "head_dim", "attention_source",
        "baseline_uses_clean_fixed_attention", "direction_formula_id",
        "detach_and_freeze_attention", "detach_and_freeze_direction",
        "direction_norm_relative_threshold", "direction_norm_scale_floor",
        "degenerate_direction_reason", "token_parity_atol", "token_parity_rtol",
        "patch_support_minimum", "positive_patch_evidence_minimum_fraction_of_token_l1",
        "cls_contribution_retained_separately",
    }, "targets.fixed_clean_attention_cls_head_direction")
    _exact_keys(targets["clean_predicted_class_raw_logit"], {
        "target_kind", "class_index_source", "freeze_class_index_for_all_interventions",
        "exclude_misclassified",
    }, "targets.clean_predicted_class_raw_logit")
    _exact_keys(targets["repeatability"], {"audit_only", "atol", "rtol"}, "targets.repeatability")
    assert targets["baseline"] == {"id": "zero_normalized", "normalized_value": 0.0, "canonical_rgb_value": 0.5, "channels": 3}
    assert targets["ig"]["steps_schedule"] == [512, 1024, 2048, 4096, 8192]
    assert targets["fixed_clean_attention_cls_head_direction"]["baseline_uses_clean_fixed_attention"] is True

    layers = config["layers_heads"]
    _exact_keys(layers, {
        "model_blocks", "heads_per_block", "tokens", "patch_tokens",
        "cls_token_index", "calibration_blocks", "calibration_heads",
        "depth_bands", "holdout_value_path_configs_source",
        "confirmatory_value_path_config_count", "exploratory_value_path_config_count",
        "direct_class_logit_block", "direct_class_logit_head_fusion",
        "rollout_blocks", "rollout_head_fusion", "rollout_add_identity",
        "rollout_row_normalize_after_identity", "rollout_multiplication_order",
        "grad_cam_target_layer", "grad_cam_channel_weight_reduction",
        "grad_cam_activation", "grad_cam_minmax_normalization",
    }, "layers_heads")
    assert (layers["model_blocks"], layers["heads_per_block"], layers["tokens"], layers["patch_tokens"]) == (12, 12, 197, 196)
    assert layers["calibration_blocks"] == layers["calibration_heads"] == list(range(12))
    assert layers["rollout_blocks"] == list(range(12))
    assert layers["depth_bands"] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

    projection = config["projection"]
    _exact_keys(projection, {
        "input_height", "input_width", "patch_height", "patch_width",
        "metric_grid_height", "metric_grid_width", "metric_ready_dtype",
        "metric_accumulation_dtype", "rgb_reduction", "pixel_to_patch_operator",
        "patch_to_pixel_operator", "fine_cell_operator",
        "matched_patch_to_fine_cell_operator", "projection_mass_atol",
        "projection_mass_rtol", "raw_cls_attention_to_patch", "signed_views",
        "absolute_value_default_forbidden", "display_artifact_metric_input_forbidden",
    }, "projection")
    _exact_keys(projection["raw_cls_attention_to_patch"], {"query", "exclude_cls_key", "renormalize_patch_mass_to_one"}, "projection.raw_cls_attention_to_patch")
    assert projection["input_height"] == projection["patch_height"] * projection["metric_grid_height"] == 224
    assert projection["input_width"] == projection["patch_width"] * projection["metric_grid_width"] == 224
    assert projection["display_artifact_metric_input_forbidden"] is True

    metrics = config["metrics"]
    _exact_keys(metrics, {
        "required", "correlation_views", "spearman_tie_method",
        "constant_relative_l2_threshold", "distribution_views",
        "distribution_zero_mass_relative_l1_threshold", "js_log_base",
        "js_epsilon", "js_zero_log_zero_convention", "kl_log_base",
        "kl_epsilon", "kl_directions", "hotspot_fractions",
        "hotspot_count_rounding", "hotspot_sort", "constant_hotspot_is_na",
        "entropy_log_base", "entropy_denominator", "na_reasons",
        "invalidating_na_reasons", "raw_attention_vs_value_path_label",
    }, "metrics")
    assert metrics["js_log_base"] == metrics["kl_log_base"] == 2
    assert metrics["js_epsilon"] == 0.0 and metrics["kl_epsilon"] == 1e-12
    assert metrics["hotspot_fractions"] == [0.1, 0.2]
    assert len(metrics["required"]) == len(set(metrics["required"]))

    fidelity = config["fidelity"]
    _exact_keys(fidelity, {
        "replacement_baseline_ref", "rgb_channels_replaced_together",
        "ranking_evidence", "target_response_normalization",
        "target_delta_relative_threshold", "curve_fraction_count",
        "curve_fractions", "unit_count_rounding", "auc_integration",
        "clip_target_response", "patch_level", "fine_level",
        "within_patch_tie_replicates", "base_seed", "seed_namespace",
        "seed_hash_extraction", "matched_baseline_auc_aggregation",
        "sanity_rankings", "sufficiency_fraction",
        "comprehensiveness_fraction", "value_path_pair", "class_logit_methods",
    }, "fidelity")
    _exact_keys(fidelity["patch_level"], {"unit", "unit_count", "claim_boundary"}, "fidelity.patch_level")
    _exact_keys(fidelity["fine_level"], {"unit", "unit_count", "cells_per_patch", "claim_boundary", "confirmatory_delta_fid_scale"}, "fidelity.fine_level")
    fractions = fidelity["curve_fractions"]
    assert len(fractions) == fidelity["curve_fraction_count"] == 21
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert all(left < right for left, right in zip(fractions, fractions[1:]))
    assert fidelity["patch_level"]["unit_count"] == 196
    assert fidelity["fine_level"]["unit_count"] == 3136
    assert fidelity["value_path_pair"] == [
        "conditional_fixed_clean_attention_head_direction_value_path_ig",
        "matched_fixed_clean_attention_patch_token_contribution",
    ]
    assert set(fidelity["class_logit_methods"]) <= METHOD_IDS

    perturbations = config["perturbations"]
    _exact_keys(perturbations, {
        "application_domain", "share_exact_tensor_across_methods",
        "freeze_clean_class_target", "freeze_clean_value_attention_and_direction",
        "items", "report_all_samples_fixed_target",
        "report_prediction_unchanged_subset_descriptive", "report_class_flip_rate",
        "stage_b_sensitivity_sanity_is_separate",
    }, "perturbations")
    by_perturbation = {item["id"]: item for item in perturbations["items"]}
    assert len(by_perturbation) == 3 and set(by_perturbation) == PERTURBATION_IDS
    _exact_keys(by_perturbation["translate_right_down_4"], {
        "id", "kind", "dx_pixels", "dy_pixels", "padding",
        "interpolation", "inverse_map", "valid_pixel_box_half_open",
        "fully_valid_patch_rows", "fully_valid_patch_cols",
    }, "perturbation.translation")
    _exact_keys(by_perturbation["center_crop_216_resize_224"], {
        "id", "kind", "crop_box_half_open", "crop_height", "crop_width",
        "resize_height", "resize_width", "image_interpolation",
        "image_antialias", "inverse_map_interpolation",
        "inverse_map_antialias", "inverse_mass_jacobian_multiplier",
        "valid_pixel_box_half_open", "fully_valid_patch_rows",
        "fully_valid_patch_cols",
    }, "perturbation.crop_scale")
    _exact_keys(by_perturbation["gaussian_rgb_std_0_01"], {
        "id", "kind", "mean", "std_rgb_0_to_1", "generator_device",
        "generator_dtype", "base_seed", "sample_seed_rule", "clamp_min",
        "clamp_max", "inverse_map", "valid_pixel_box_half_open",
        "fully_valid_patch_rows", "fully_valid_patch_cols",
    }, "perturbation.noise")
    assert by_perturbation["translate_right_down_4"]["dx_pixels"] == 4
    assert by_perturbation["translate_right_down_4"]["dy_pixels"] == 4
    assert by_perturbation["gaussian_rgb_std_0_01"]["std_rgb_0_to_1"] == 0.01

    retention = config["retention"]
    _exact_keys(retention, {
        "metric_compute_dtype", "l1_storage_dtype", "levels",
        "audit_sample_ids", "display_is_never_metric_input",
        "forbidden_artifacts", "execution_unit_immutable",
        "overwrite_existing_unit",
        "resume_requires_complete_status_manifest_hash_and_all_artifact_hashes",
        "failed_unit_restarts_in_new_empty_output",
        "failure_table_retains_every_expected_unit",
        "batch_complete_requires_every_expected_unit_terminal",
    }, "retention")
    _exact_keys(retention["levels"], {"L0", "L1", "L2", "L3"}, "retention.levels")
    assert set(retention["audit_sample_ids"]) == {f"ILSVRC2012_val/{item}.JPEG" for item in AUDIT_IDS}
    assert retention["display_is_never_metric_input"] is True
    assert retention["overwrite_existing_unit"] is False

    budget = config["budget"]
    _exact_keys(budget, {
        "bytes_per_l0_l1_execution_unit_max", "bytes_per_audit_sample_l2_max",
        "stage_c_c8_c9_total_artifact_bytes_max",
        "stage_d_500_total_artifact_bytes_max",
        "forward_calls_per_execution_unit_max",
        "backward_calls_per_execution_unit_max",
        "cuda_peak_reserved_bytes_per_execution_unit_max",
        "stage_c_c8_c9_h20_gpu_hours_max", "stage_d_500_h20_gpu_hours_max",
        "stage_d_h20_gpu_hours_per_image_max", "projection_safety_factor",
        "storage_control_overhead_factor", "preflight_includes",
        "c7_may_loosen_caps", "over_cap_action",
    }, "budget")
    for key, value in budget.items():
        if key.endswith(("_max", "_factor")):
            _positive_number(value, f"budget.{key}")
    assert budget["c7_may_loosen_caps"] is False

    selection = config["selection"]
    _exact_keys(selection, {
        "stage", "eligible_role", "forbidden_role", "candidate_space",
        "require_all_five_calibration_valid", "validity_filter",
        "ranking_order_descending", "na_stability_rank_value",
        "na_residual_ratio_rank_value", "tie_break_ascending",
        "winner_per_depth_band", "primary_from_band_winners",
        "empty_band_action", "holdout_accessed_must_be_false",
        "output_path", "manual_edit_forbidden",
    }, "selection")
    assert selection["eligible_role"] == "calibration"
    assert selection["forbidden_role"] == "holdout"
    assert selection["tie_break_ascending"] == ["block_index", "head_index"]
    assert selection["winner_per_depth_band"] == selection["primary_from_band_winners"] == 1

    statistics = config["statistics"]
    _exact_keys(statistics, {
        "unit", "confirmatory_role", "confirmatory_n", "calibration_n",
        "calibration_in_confirmatory_statistics", "all_20_is_descriptive_only",
        "bootstrap_resamples", "bootstrap_cluster", "bootstrap_statistic",
        "bootstrap_base_seed", "bootstrap_seed_namespace",
        "bootstrap_seed_hash_extraction", "confidence_level",
        "confidence_interval", "exploratory_band_winner_multiple_testing",
        "required_summaries",
    }, "statistics")
    assert statistics["unit"] == statistics["bootstrap_cluster"] == "image"
    assert statistics["confirmatory_n"] == 15 and statistics["calibration_n"] == 5
    assert statistics["calibration_in_confirmatory_statistics"] is False
    assert statistics["bootstrap_resamples"] == 10_000

    gates = config["decision_gates"]
    _exact_keys(gates, {
        "allowed_decisions", "endpoint_parity_max_abs_exclusive",
        "conservation_atol", "conservation_rtol", "token_parity_atol",
        "token_parity_rtol", "primary_holdout_required_complete",
        "delta_fid_definition", "delta_fid_scale", "effect_median_minimum",
        "effect_positive_count_minimum", "effect_bootstrap_ci_lower_exclusive",
        "effect_fail_if_bootstrap_ci_upper_exclusive_below",
        "patch_support_minimum",
        "positive_patch_evidence_minimum_fraction_of_token_l1",
        "patch_residual_l1_ratio_minimum",
        "spatial_joint_support_positive_evidence_and_residual_count_minimum",
        "stability_perturbations",
        "stability_spearman_positive_count_minimum_per_perturbation",
        "stability_median_spearman_minimum",
        "stability_bootstrap_ci_lower_exclusive",
        "stability_fail_if_bootstrap_ci_upper_exclusive_below",
        "cost_caps_ref", "decision_priority", "equality_at_ci_boundary",
    }, "decision_gates")
    assert set(gates["allowed_decisions"]) == DECISIONS
    assert gates["primary_holdout_required_complete"] == 15
    assert gates["effect_median_minimum"] == 0.02
    assert gates["effect_positive_count_minimum"] == 12
    assert gates["positive_patch_evidence_minimum_fraction_of_token_l1"] == 0.01
    assert gates["patch_residual_l1_ratio_minimum"] == 0.05
    assert gates["spatial_joint_support_positive_evidence_and_residual_count_minimum"] == 12
    assert gates["stability_median_spearman_minimum"] == 0.5
    assert set(gates["stability_perturbations"]) == PERTURBATION_IDS
    assert gates["decision_priority"] == [
        "validity_failure_to_NO_GO_VALIDITY",
        "effect_or_spatial_failure_to_NO_GO_NOVELTY",
        "effect_or_spatial_uncertain_to_INCONCLUSIVE",
        "cost_failure_to_NO_GO_COST",
        "all_pass_to_GO",
    ]


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == columns, f"Unexpected CSV header in {path}"
        rows = list(reader)
    assert all(None not in row for row in rows), f"Extra CSV fields in {path}"
    assert all(set(row) == set(columns) for row in rows), f"Missing CSV fields in {path}"
    return rows


def _validate_source_manifest(config: dict[str, Any]) -> list[dict[str, str]]:
    registry_cfg = config["registry"]
    paths = _contract_paths(config)
    assert paths["source_manifest"] == SOURCE_MANIFEST_PATH.resolve()
    assert paths["source_metadata"] == SOURCE_METADATA_PATH.resolve()
    assert paths["devkit"] == DEVKIT_PATH.resolve()
    assert paths["data_root"] == DATA_ROOT.resolve()
    assert sha256_file(paths["source_manifest"]) == registry_cfg["source_manifest_sha256"]
    assert sha256_file(paths["source_metadata"]) == registry_cfg["source_manifest_metadata_sha256"]
    assert sha256_file(paths["devkit"]) == registry_cfg["official_devkit_sha256"] == OFFICIAL_DEVKIT_SHA256
    metadata = strict_json_load(paths["source_metadata"])
    assert metadata["sample_count"] == metadata["labeled_count"] == 500
    assert metadata["official_devkit"]["sha256"] == OFFICIAL_DEVKIT_SHA256
    assert metadata["mapping"]["synset_order_sha256"] == IMAGENET_SYNSET_ORDER_SHA256
    rows = _read_csv(paths["source_manifest"], SOURCE_COLUMNS)
    assert len(rows) == 500
    assert len({row["sample_id"] for row in rows}) == 500
    assert len({row["relative_path"] for row in rows}) == 500
    assert all(row["label_source"] == "official_devkit_val_gt" for row in rows)

    rebuilt = build_manifest(paths["data_root"], devkit_path=paths["devkit"], require_labels=True)
    assert len(rebuilt) == 500
    rebuilt_strings = [
        {key: str(value) for key, value in asdict(row).items()} for row in rebuilt
    ]
    assert rows == rebuilt_strings, "Tracked 500-row manifest differs from Stage A rebuild"
    return rows


def _validate_registry(config: dict[str, Any], source_rows: list[dict[str, str]]) -> None:
    registry_cfg = config["registry"]
    paths = _contract_paths(config)
    assert paths["registry"] == REGISTRY_PATH.resolve()
    assert paths["registry_metadata"] == REGISTRY_METADATA_PATH.resolve()
    assert sha256_file(paths["registry"]) == registry_cfg["sha256"]
    assert sha256_file(paths["registry_metadata"]) == registry_cfg["metadata_sha256"]
    metadata = strict_json_load(paths["registry_metadata"])
    _exact_keys(
        metadata,
        {
            "schema_version", "document_type", "dataset", "official_labels",
            "registry", "selection", "generation", "asset_findings",
        },
        "registry_metadata",
    )
    _exact_keys(metadata["dataset"], {
        "logical_root", "runtime_root_relative_to_workspace",
        "source_manifest_path", "source_manifest_sha256",
        "source_manifest_metadata_path", "source_manifest_metadata_sha256",
    }, "registry_metadata.dataset")
    _exact_keys(metadata["official_labels"], {
        "devkit_relative_path", "devkit_sha256", "label_source",
        "synset_order_sha256",
    }, "registry_metadata.official_labels")
    _exact_keys(metadata["registry"], {
        "path", "sha256", "sample_count", "calibration_count",
        "holdout_count", "audit_count",
    }, "registry_metadata.registry")
    _exact_keys(metadata["selection"], {
        "legacy_source_read_only", "legacy_list_path", "legacy_list_sha256",
        "legacy_script_path", "legacy_script_sha256", "rule_id",
        "source_selection_indices_zero_based",
        "calibration_selection_orders_zero_based",
        "audit_selection_orders_zero_based", "selection_information_used",
        "forbidden_selection_information",
    }, "registry_metadata.selection")
    _exact_keys(metadata["generation"], {
        "stage_a_command", "base_git_commit", "tree_dirty_during_generation",
        "dirty_reason", "dependency_lock_sha256",
    }, "registry_metadata.generation")
    assert metadata["schema_version"] == 1
    assert metadata["document_type"] == "stage_c_sample_registry_metadata"
    assert metadata["registry"]["sha256"] == registry_cfg["sha256"]
    assert metadata["dataset"]["source_manifest_sha256"] == registry_cfg["source_manifest_sha256"]
    assert metadata["official_labels"]["devkit_sha256"] == OFFICIAL_DEVKIT_SHA256
    assert metadata["official_labels"]["synset_order_sha256"] == IMAGENET_SYNSET_ORDER_SHA256
    assert metadata["generation"]["dependency_lock_sha256"] == sha256_file(
        REPO_ROOT / "requirements.lock.txt"
    )
    assert metadata["selection"]["source_selection_indices_zero_based"] == SOURCE_INDICES
    assert metadata["selection"]["legacy_source_read_only"] is True
    assert sha256_file(WORKSPACE_ROOT / metadata["selection"]["legacy_list_path"]) == (
        metadata["selection"]["legacy_list_sha256"]
    )
    assert sha256_file(WORKSPACE_ROOT / metadata["selection"]["legacy_script_path"]) == (
        metadata["selection"]["legacy_script_sha256"]
    )

    rows = _read_csv(paths["registry"], REGISTRY_COLUMNS)
    row_schema = strict_json_load(REGISTRY_SCHEMA_PATH)["properties"]["row_contract"]
    row_validator = Draft202012Validator(row_schema)
    for row in rows:
        typed_row = dict(row)
        for key in (
            "schema_version", "size_bytes", "width", "height",
            "imagenet1k_index", "selection_order", "source_selection_index",
        ):
            typed_row[key] = int(typed_row[key])
        typed_row["audit"] = {"true": True, "false": False}[typed_row["audit"]]
        row_validator.validate(typed_row)
    assert len(rows) == 20
    assert [row["image_id"] for row in rows] == FROZEN_IMAGE_IDS
    assert [int(row["selection_order"]) for row in rows] == list(range(20))
    assert [int(row["source_selection_index"]) for row in rows] == SOURCE_INDICES
    for unique_field in ("sample_id", "image_id", "relative_path", "image_sha256"):
        assert len({row[unique_field] for row in rows}) == 20, unique_field
    assert Counter(row["role"] for row in rows) == {"calibration": 5, "holdout": 15}
    calibration = {row["image_id"] for row in rows if row["role"] == "calibration"}
    audit = {row["image_id"] for row in rows if row["audit"] == "true"}
    assert calibration == CALIBRATION_IDS and audit == AUDIT_IDS and audit <= calibration
    assert all(row["audit"] in {"true", "false"} for row in rows)
    assert all(row["schema_version"] == "1" for row in rows)
    assert all(row["selection_rule"] == SELECTION_RULE for row in rows)
    assert all(row["label_source"] == "official_devkit_val_gt" for row in rows)

    source_by_path = {row["relative_path"]: row for row in source_rows}
    for row, source_index in zip(rows, SOURCE_INDICES):
        relative = PurePosixPath(row["relative_path"])
        assert not relative.is_absolute() and ".." not in relative.parts and len(relative.parts) == 1
        source = source_by_path[row["relative_path"]]
        assert source_rows[source_index] == source
        assert row["sample_id"] == source["sample_id"]
        assert row["image_id"] + ".JPEG" == row["relative_path"]
        for registry_key, source_key in (
            ("image_sha256", "image_sha256"),
            ("size_bytes", "size_bytes"),
            ("width", "width"),
            ("height", "height"),
            ("official_wnid", "wnid"),
            ("imagenet1k_index", "imagenet1k_index"),
            ("label_source", "label_source"),
        ):
            assert row[registry_key] == source[source_key], (registry_key, row["image_id"])
        image_path = paths["data_root"] / row["relative_path"]
        assert image_path.is_file() and image_path.resolve().is_relative_to(paths["data_root"].resolve())
        assert sha256_file(image_path) == row["image_sha256"]
        assert image_path.stat().st_size == int(row["size_bytes"])
        with Image.open(image_path) as image:
            assert image.size == (int(row["width"]), int(row["height"]))
            image.verify()


def _validate_schemas(config: dict[str, Any]) -> None:
    config_schema = strict_json_load(CONFIG_SCHEMA_PATH)
    registry_schema = strict_json_load(REGISTRY_SCHEMA_PATH)
    selection_schema = strict_json_load(SELECTION_SCHEMA_PATH)
    for schema in (config_schema, registry_schema, selection_schema):
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object" and schema["additionalProperties"] is False
    Draft202012Validator(config_schema).validate(config)
    assert set(config_schema["required"]) == set(config_schema["properties"])
    assert config_schema["properties"]["schema_version"]["const"] == 1
    assert registry_schema["properties"]["required_columns"]["const"] == REGISTRY_COLUMNS
    row_contract = registry_schema["properties"]["row_contract"]
    assert row_contract["additionalProperties"] is False
    assert set(row_contract["required"]) == set(row_contract["properties"]) == set(REGISTRY_COLUMNS)
    assert selection_schema["properties"]["holdout_accessed"]["const"] is False
    assert selection_schema["properties"]["candidate_records"]["minItems"] == 144
    assert selection_schema["properties"]["depth_band_winners"]["minItems"] == 4
    assert selection_schema["$defs"]["input_hashes"]["additionalProperties"] is False
    assert selection_schema["$defs"]["candidate"]["additionalProperties"] is False
    assert selection_schema["$defs"]["candidate_ref"]["additionalProperties"] is False
    assert selection_schema["$defs"]["failure"]["additionalProperties"] is False


def _sample_selection_document() -> dict[str, Any]:
    zero_hash = "0" * 64
    candidates: list[dict[str, Any]] = []
    for block in range(12):
        for head in range(12):
            key = f"block_{block:02d}_head_{head:02d}"
            candidates.append({
                "candidate_key": key,
                "block_index": block,
                "head_index": head,
                "depth_band": block // 3,
                "calibration_sample_count": 5,
                "eligible": True,
                "validity_failures": [],
                "count_delta_fid_fine4_gt_zero": 5,
                "median_delta_fid_fine4": 0.1,
                "minimum_perturbation_median_residual_spearman": 0.6,
                "median_patch_residual_l1_ratio": 0.1,
                "rounded_rank_tuple": [5, 0.1, 0.6, 0.1, -block, -head],
                "resource_within_hard_budget": True,
                "artifact_manifest_sha256": zero_hash,
                "rank_within_depth_band": 1,
                "selected_depth_band_winner": False,
                "selected_primary": False,
            })
    for band in range(4):
        band_candidates = [item for item in candidates if item["depth_band"] == band]
        ordered = sorted(band_candidates, key=lambda item: tuple(item["rounded_rank_tuple"]), reverse=True)
        for rank, item in enumerate(ordered, 1):
            item["rank_within_depth_band"] = rank
        ordered[0]["selected_depth_band_winner"] = True
    candidates[0]["selected_primary"] = True
    refs = [
        {
            "candidate_key": f"block_{block:02d}_head_00",
            "block_index": block,
            "head_index": 0,
            "depth_band": block // 3,
        }
        for block in (0, 3, 6, 9)
    ]
    return {
        "schema_version": 1,
        "document_type": "stage_c_selection",
        "generated_by": "test_fixture",
        "holdout_accessed": False,
        "input_hashes": {
            "code_commit": "0" * 40,
            "code_tree_clean": True,
            "model_sha256": zero_hash,
            "dependency_lock_sha256": zero_hash,
            "registry_sha256": zero_hash,
            "registry_metadata_sha256": zero_hash,
            "protocol_sha256": zero_hash,
            "acceptance_sha256": zero_hash,
            "config_sha256": zero_hash,
            "config_schema_sha256": zero_hash,
            "calibration_artifact_manifest_sha256": zero_hash,
        },
        "calibration_sample_ids": CALIBRATION_SAMPLE_IDS,
        "candidate_records": candidates,
        "depth_band_winners": refs,
        "primary": refs[0],
        "failure_records": [],
    }


def _validate_selection_semantics(document: dict[str, Any]) -> None:
    candidates = document["candidate_records"]
    expected = {(block, head) for block in range(12) for head in range(12)}
    actual = {(item["block_index"], item["head_index"]) for item in candidates}
    _require(len(candidates) == 144 and actual == expected, "selection must cover 12x12 exactly once")
    by_key = {item["candidate_key"]: item for item in candidates}
    _require(len(by_key) == 144, "selection candidate keys must be unique")
    for item in candidates:
        expected_key = f"block_{item['block_index']:02d}_head_{item['head_index']:02d}"
        _require(item["candidate_key"] == expected_key, "candidate key/index mismatch")
        _require(item["depth_band"] == item["block_index"] // 3, "candidate depth-band mismatch")
        if item["eligible"]:
            _require(not item["validity_failures"], "eligible candidate has validity failures")
            _require(item["resource_within_hard_budget"], "eligible candidate is over budget")
            _require(item["count_delta_fid_fine4_gt_zero"] is not None, "eligible candidate lacks n_pos")
            _require(item["median_delta_fid_fine4"] is not None, "eligible candidate lacks fidelity")
            m_stab = (
                -1.0 if item["minimum_perturbation_median_residual_spearman"] is None
                else round(item["minimum_perturbation_median_residual_spearman"], 6)
            )
            m_res = (
                0.0 if item["median_patch_residual_l1_ratio"] is None
                else round(item["median_patch_residual_l1_ratio"], 6)
            )
            expected_tuple = [
                item["count_delta_fid_fine4_gt_zero"],
                round(item["median_delta_fid_fine4"], 6),
                m_stab,
                m_res,
                -item["block_index"],
                -item["head_index"],
            ]
            _require(item["rounded_rank_tuple"] == expected_tuple, "candidate rank tuple mismatch")
            _require(item["rank_within_depth_band"] is not None, "eligible candidate lacks rank")
        else:
            _require(bool(item["validity_failures"]), "ineligible candidate lacks failure reason")
            _require(item["rounded_rank_tuple"] is None, "ineligible candidate has a rank tuple")
            _require(item["rank_within_depth_band"] is None, "ineligible candidate has a rank")
    for band in range(4):
        eligible = [item for item in candidates if item["depth_band"] == band and item["eligible"]]
        _require(bool(eligible), f"depth band {band} has no eligible candidate")
        ordered = sorted(eligible, key=lambda item: tuple(item["rounded_rank_tuple"]), reverse=True)
        _require(
            [item["rank_within_depth_band"] for item in ordered] == list(range(1, len(ordered) + 1)),
            f"depth band {band} ranks do not follow frozen tuple",
        )
    winners = document["depth_band_winners"]
    _require({item["depth_band"] for item in winners} == {0, 1, 2, 3}, "one winner per band required")
    winner_keys = {item["candidate_key"] for item in winners}
    for reference in [*winners, document["primary"]]:
        _require(reference["candidate_key"] in by_key, "selection reference is missing")
        candidate = by_key[reference["candidate_key"]]
        _require(
            (reference["block_index"], reference["head_index"], reference["depth_band"]) ==
            (candidate["block_index"], candidate["head_index"], candidate["depth_band"]),
            "selection reference fields disagree with candidate",
        )
    _require(document["primary"]["candidate_key"] in winner_keys, "primary must be a band winner")
    _require(
        {item["candidate_key"] for item in candidates if item["selected_depth_band_winner"]} == winner_keys,
        "winner flags disagree with winner references",
    )
    _require(
        [item["candidate_key"] for item in candidates if item["selected_primary"]] ==
        [document["primary"]["candidate_key"]],
        "exactly one primary flag must match primary reference",
    )
    for failure in document["failure_records"]:
        _require(failure["candidate_key"] in by_key, "failure references unknown candidate")
        if failure["sample_id"] is not None:
            _require(failure["sample_id"] in CALIBRATION_SAMPLE_IDS, "failure references non-calibration sample")


def _validate_stage_b_chain() -> None:
    status_path = STAGE_B_ROOT / "status.json"
    manifest_path = STAGE_B_ROOT / "manifest.json"
    provenance_path = STAGE_B_ROOT / "metric_ready/provenance.json"
    status = strict_json_load(status_path)
    manifest = strict_json_load(manifest_path)
    provenance = strict_json_load(provenance_path)
    assert status["complete"] is True and status["status"] == "complete"
    assert status["profile"] == "paper"
    assert sha256_file(manifest_path) == status["manifest_sha256"]
    assert manifest["complete"] is True and manifest["status"] == "complete"
    for artifact in manifest["artifacts"]:
        path = STAGE_B_ROOT / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["actual_bytes"] == artifact["bytes"]
        assert sha256_file(path) == artifact["sha256"]
    assert provenance["code"] == {
        "commit": "30732c9f1413c58a36dd6a8d85ba8ea5788e0c8a",
        "dirty": False,
    }
    assert provenance["configuration"]["formal"] is True
    assert provenance["method"]["name"] == "conditional_fixed_attention_value_path_ig"
    historical_lock = subprocess.run(
        [
            "git", "show",
            "30732c9f1413c58a36dd6a8d85ba8ea5788e0c8a:vitnns_rebuild/requirements.lock.txt",
        ],
        cwd=REPO_ROOT, check=True, capture_output=True,
    ).stdout
    assert hashlib.sha256(historical_lock).hexdigest() == provenance["dependencies"]["lock_sha256"]


def _validate_docs() -> None:
    required = {
        "docs/阶段C_protocol.md": [
            "conditional_fixed_clean_attention_head_direction_value_path_ig",
            "spatial structure similarity",
            "NO_GO_VALIDITY",
            "NO_GO_NOVELTY",
            "NO_GO_COST",
            "INCONCLUSIVE",
            "4×4-cell/subpatch fidelity",
        ],
        "docs/stage_c_acceptance.md": [
            "fail-closed", "C0: protocol and sample freeze", "C1: strict config",
        ],
        "docs/stage_c_config_contract.md": [
            "C0 deliberately supplies data contracts only", "duplicate-key",
        ],
    }
    for relative, needles in required.items():
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"Missing {needle!r} from {relative}"
        for unresolved in ("TBD", "TODO", "待 C8", "待 C9", "结果后决定"):
            assert unresolved not in text, f"Unresolved marker {unresolved!r} in {relative}"
    step_text = (REPO_ROOT / "docs/阶段C-step-by-step.md").read_text(encoding="utf-8")
    assert (
        "当前状态：阶段 A、阶段 B、阶段 C/C0–C5 已完成；下一任务是 C6。"
        in step_text
    )
    assert "| C0 | PASS |" in step_text
    assert "| C1 | PASS |" in step_text
    assert "| C2 | PASS |" in step_text
    assert step_text.count("| C3 | PASS |") == 1
    assert step_text.count("| C4 | PASS |") == 1
    assert step_text.count("| C5 | PASS |") == 1
    assert step_text.count("| C6 | NOT_STARTED（NEXT） |") == 1
    assert "### 2026-08-21 16:05 +0800 — C0 completion record" in step_text
    assert "C1 completion record" in step_text
    assert "C2 completion record" in step_text
    assert step_text.count("C3 completion record") == 2  # heading + C4 prompt
    assert step_text.count("C4 completion record") == 3  # heading + C5 prompt/rule
    assert step_text.count("C5 completion record") == 2  # heading + C6 prompt
    assert "你现在只执行 ViT-NNS 重建工程“阶段 C”的 C1" in step_text
    assert "你现在只执行 ViT-NNS 重建工程“阶段 C”的 C2" in step_text
    assert "你现在只执行 ViT-NNS 重建工程“阶段 C”的 C3" in step_text
    assert step_text.count("你现在只执行 ViT-NNS 重建工程“阶段 C”的 C4") == 1
    assert step_text.count("你现在只执行 ViT-NNS 重建工程“阶段 C”的 C5") == 1
    assert step_text.count("你现在只执行 ViT-NNS 重建工程“阶段 C”的 C6") == 1
    c3_record = step_text.split(
        "### 2026-08-21 22:15 +0800 — C3 completion record", 1
    )[1].split("— C4 completion record", 1)[0]
    assert c3_record.count("- status: PASS") == 1
    assert c3_record.count("- next_task: C4") == 1
    c4_record = step_text.split("— C4 completion record", 1)[1].split(
        "— C5 completion record", 1
    )[0]
    assert c4_record.count("- status: PASS") == 1
    assert c4_record.count("- next_task: C5") == 1
    c5_record = step_text.split("— C5 completion record", 1)[1]
    assert c5_record.count("- status: PASS") == 1
    assert c5_record.count("- next_task: C6") == 1
    for token in (
        "C0_COMMIT_SHA",
        "STAGE_C_CONFIG_SHA256", "STAGE_C_PROTOCOL_SHA256",
        "STAGE_C_ACCEPTANCE_SHA256", "STAGE_C_CONFIG_SCHEMA_SHA256",
        "STAGE_C_REGISTRY_SCHEMA_SHA256", "STAGE_C_SELECTION_SCHEMA_SHA256",
    ):
        assert token not in step_text


def validate_all(*, require_tracked: bool = False, require_clean: bool = False) -> None:
    config = strict_json_load(CONFIG_PATH)
    _validate_config(config)
    _validate_schemas(config)
    source_rows = _validate_source_manifest(config)
    _validate_registry(config, source_rows)
    _validate_stage_b_chain()
    _validate_docs()
    if require_tracked:
        tracked = [
            ".gitignore",
            "config/stage_c.json",
            "config/stage_c.schema.json",
            "config/stage_c_registry.schema.json",
            "config/stage_c_selection.schema.json",
            "docs/阶段C-step-by-step.md",
            "docs/阶段C_protocol.md",
            "docs/stage_c_acceptance.md",
            "docs/stage_c_config_contract.md",
            "manifests/stage_c_flat_val_500/manifest.csv",
            "manifests/stage_c_flat_val_500/manifest_metadata.json",
            "manifests/stage_c_sample_registry.csv",
            "manifests/stage_c_sample_registry_metadata.json",
            "src/vitnns/stage_c_config.py",
            "src/vitnns/stage_c_maps.py",
            "src/vitnns/stage_c_metrics.py",
            "src/vitnns/stage_c_attention.py",
            "src/vitnns/stage_c_class_logit.py",
            "src/vitnns/stage_c_value_path.py",
            "src/vitnns/stage_c_fidelity.py",
            "src/vitnns/stage_c_perturbations.py",
            "src/vitnns/stage_c_stability.py",
            "src/vitnns/stage_c_resources.py",
            "tests/test_stage_c_config.py",
            "tests/test_stage_c_maps.py",
            "tests/test_stage_c_metrics.py",
            "tests/test_stage_c_attention.py",
            "tests/test_stage_c_class_logit.py",
            "tests/test_stage_c_value_path.py",
            "tests/test_stage_c_fidelity.py",
            "tests/test_stage_c_perturbations.py",
            "tests/test_stage_c_stability.py",
            "tests/test_stage_c_resources.py",
            "tests/test_stage_c_contract.py",
        ]
        for relative in tracked:
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=REPO_ROOT, check=True, capture_output=True, text=True,
            )
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
        assert not status.strip(), f"Git tree is not clean:\n{status}"


def test_frozen_stage_c_contract() -> None:
    validate_all()


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"key": 1, "key": 2}\n', encoding="utf-8")
    try:
        strict_json_load(duplicate)
    except ValueError as error:
        assert "Duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate JSON key was accepted")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n', encoding="utf-8")
    try:
        strict_json_load(nonfinite)
    except ValueError as error:
        assert "Non-finite JSON number" in str(error)
    else:
        raise AssertionError("non-finite JSON number was accepted")


def test_config_validator_rejects_missing_and_unknown_fields() -> None:
    config = strict_json_load(CONFIG_PATH)
    missing = copy.deepcopy(config)
    del missing["decision_gates"]
    try:
        _validate_config(missing)
    except AssertionError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing required config field was accepted")
    unknown = copy.deepcopy(config)
    unknown["selection"]["choose_after_holdout"] = True
    try:
        _validate_config(unknown)
    except AssertionError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown nested config field was accepted")


def test_draft_2020_12_config_schema_rejects_nested_drift() -> None:
    config = strict_json_load(CONFIG_PATH)
    validator = Draft202012Validator(strict_json_load(CONFIG_SCHEMA_PATH))
    validator.validate(config)
    for mutant in (
        lambda value: value["roles"].update({"silent_fallback": True}),
        lambda value: value["targets"]["ig"].pop("rtol"),
        lambda value: value["decision_gates"].update({"effect_median_minimum": 0.0}),
        lambda value: value["methods"].append(copy.deepcopy(value["methods"][0])),
    ):
        changed = copy.deepcopy(config)
        mutant(changed)
        try:
            validator.validate(changed)
        except ValidationError:
            pass
        else:
            raise AssertionError("nested config drift passed the exact-instance schema")


def test_selection_schema_and_semantics_fail_closed() -> None:
    schema = strict_json_load(SELECTION_SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    document = _sample_selection_document()
    validator.validate(document)
    _validate_selection_semantics(document)

    schema_mutations = (
        lambda value: value["input_hashes"].pop("protocol_sha256"),
        lambda value: value["candidate_records"][0].update({"unknown": 1}),
        lambda value: value["candidate_records"][0].pop("block_index"),
        lambda value: value.update({"holdout_accessed": True}),
    )
    for mutate in schema_mutations:
        changed = copy.deepcopy(document)
        mutate(changed)
        try:
            validator.validate(changed)
        except ValidationError:
            pass
        else:
            raise AssertionError("invalid C8 selection passed JSON Schema")

    semantic_mutations = (
        lambda value: value["candidate_records"].__setitem__(
            1, copy.deepcopy(value["candidate_records"][0])
        ),
        lambda value: value["depth_band_winners"].__setitem__(
            1, copy.deepcopy(value["depth_band_winners"][0])
        ),
        lambda value: value["primary"].update({"candidate_key": "block_11_head_11", "block_index": 11, "head_index": 11, "depth_band": 3}),
        lambda value: value["candidate_records"][0].update({"rank_within_depth_band": 2}),
    )
    for mutate in semantic_mutations:
        changed = copy.deepcopy(document)
        mutate(changed)
        validator.validate(changed)
        try:
            _validate_selection_semantics(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("semantically invalid C8 selection passed validation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tracked", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    validate_all(require_tracked=args.require_tracked, require_clean=args.require_clean)
    print("Stage C C0 static contract validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
