"""Immutable, metadata-only Stage D planning and budget contracts.

The module deliberately has no model, image, CUDA, runner, or artifact-writer
dependencies.  It expands the frozen D2 contract into content-addressed image
groups and units while representing the 211,010 work coordinates through lazy
sequences.  The compact serialized plan stores the expansion dimensions and a
streaming digest of every coordinate instead of a 210,000-row JSON array.
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = 1
PLAN_DOCUMENT_TYPE = "stage_d_execution_plan"
ESTIMATE_DOCUMENT_TYPE = "stage_d_preflight_estimate"
PREFLIGHT_DOCUMENT_TYPE = "stage_d_metadata_preflight"
PROFILE_DOCUMENT_TYPE = "stage_d_resource_profile"
PROFILE_UNIT_KINDS = ("primary_execution_unit", "audit_repeat_map")

NNS_METHOD_ID = (
    "conditional_fixed_clean_attention_head_direction_value_path_ig"
)
MATCHED_METHOD_ID = "matched_fixed_clean_attention_patch_token_contribution"
METHOD_IDS = (NNS_METHOD_ID, MATCHED_METHOD_ID)
BLOCK_INDEX = 0
HEAD_INDEX = 9
TARGET_ID = "fixed_clean_attention_cls_head_direction"
ENDPOINT_ID = "fine_4x4_cell_delta_fid"
FIDELITY_SCALE = "fine_4x4_cell"
FINE_CELL_COUNT = 3136
DIRECTIONS = ("insertion", "deletion")
CURVE_FRACTIONS: tuple[int | float, ...] = (
    0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1,
)
CONFIG_CURVE_FRACTIONS: tuple[float, ...] = tuple(
    float(value) for value in CURVE_FRACTIONS
)
TIE_REPEAT_INDICES = (0, 1, 2, 3, 4)
IMAGE_GROUP_COUNT = 500
EXECUTION_UNIT_COUNT = 1000
PRIMARY_CLEAN_MAP_COUNT = 1000
AUDIT_IMAGE_COUNT = 5
AUDIT_REPEAT_CLEAN_MAP_COUNT = 10
MAP_WORK_ITEM_COUNT = 1010
FIDELITY_CURVE_COUNT = 10000
FIDELITY_CURVE_POINT_COUNT = 210000
LOGICAL_WORK_ITEM_COUNT = 211010
STATISTICAL_N = 500

SHARD_COUNT_STATUS = "DEFERRED_TO_D7_BEFORE_EXECUTION_AUTHORIZATION"
SHARD_ASSIGNMENT_FORMULA = "selection_order_modulo_frozen_shard_count"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COORDINATE_STREAM_DOMAIN = b"vitnns-stage-d-coordinate-stream-v1\0"
_PLAN_ISSUER = object()
FROZEN_CONFIG_CANONICAL_DIGEST = (
    "21f9dcf1d615f2531f7d7d9a36319b958e06d0606c325890dd31ad89b7c2caae"
)
FROZEN_CONFIG_FILE_SHA256 = (
    "88fdc9cb2b4a5bd74ff7d8e68c79463cafac6437e29b5564c2115a13f1f1136d"
)
FROZEN_RUN_SPEC_CANONICAL_DIGEST = (
    "56a1d946c96ce5f86699e52a359557c1cdfd6ebbfd5aa43646d5e3a2d6428c85"
)
FROZEN_REGISTRY_CANONICAL_DIGEST = (
    "514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028"
)
FROZEN_REGISTRY_FILE_SHA256 = (
    "9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b"
)
FROZEN_ORDERED_SAMPLE_IDS_DIGEST = (
    "b187ff5cbacc1b595f06d12e309602b09965e01e7ee1461fd8a694354d4b8f50"
)
FROZEN_PLAN_CONTEXT_DIGEST = (
    "3620c32e07610c22051b91b724227c4440563420b49b92867789cb7179628f14"
)
FROZEN_MODEL_CONTEXT_DIGEST = (
    "cbd00ccb5284aab8fd1cd797cbd87a7f1ae1d78fae3304316115f9cf62e15412"
)
FROZEN_TARGET_CONTEXT_DIGEST = (
    "da9f15eeaec4cc04b0eac3e895eeda2fdf551296b31d45aaec1ec4f30648e2a1"
)
FROZEN_FIDELITY_CONTEXT_DIGEST = (
    "e87a0c0145b3519665aeb1c41d0d5482ba8dcf87bf828b406a149e105c39143f"
)
FROZEN_GROUP_PROJECTION_DIGEST = (
    "7fa5758956feb8997471a661c170973d0999be30fbe57d6f1621fe7126b069b0"
)
FROZEN_AUDITS_BY_ORDER = (
    (14, "ILSVRC2012_val/ILSVRC2012_val_00028787.JPEG", "STAGE_D_FIDELITY_V1_AUDIT_01"),
    (145, "ILSVRC2012_val/ILSVRC2012_val_00024160.JPEG", "STAGE_D_FIDELITY_V1_AUDIT_02"),
    (231, "ILSVRC2012_val/ILSVRC2012_val_00004877.JPEG", "STAGE_D_FIDELITY_V1_AUDIT_00"),
    (326, "ILSVRC2012_val/ILSVRC2012_val_00016533.JPEG", "STAGE_D_FIDELITY_V1_AUDIT_03"),
    (380, "ILSVRC2012_val/ILSVRC2012_val_00014223.JPEG", "STAGE_D_FIDELITY_V1_AUDIT_04"),
)


class StageDPlanError(ValueError):
    """Raised when a Stage D plan or planning input is not exact."""


class StageDBudgetError(StageDPlanError):
    """Raised when a profiled Stage D launch would violate a frozen gate."""


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(child) for child in value]
    if isinstance(value, list):
        return [_deep_thaw(child) for child in value]
    return copy.deepcopy(value)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _deep_freeze(child) for key, child in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    return copy.deepcopy(value)


def _validate_json_safe(value: Any, context: str = "value") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDPlanError(f"{context} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDPlanError(
                    f"{context} keys must be non-empty strings"
                )
            _validate_json_safe(child, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_safe(child, f"{context}[{index}]")
        return
    raise StageDPlanError(
        f"{context} contains non-JSON-safe {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the frozen compact JSON encoding used by every digest."""

    thawed = _deep_thaw(value)
    _validate_json_safe(thawed)
    return json.dumps(
        thawed, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _semantic_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: child for key, child in value.items()
        if key != "canonical_digest"
    })


def _coerce_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        converter = getattr(value, "to_dict", None)
        if not callable(converter):
            raise TypeError(f"{context} must be a mapping or expose to_dict()")
        value = converter()
    if not isinstance(value, Mapping):
        raise TypeError(f"{context}.to_dict() must return a mapping")
    result = _deep_thaw(value)
    _validate_json_safe(result, context)
    return result


def _exact_keys(value: Any, expected: Iterable[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageDPlanError(f"{context} must be an object")
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise StageDPlanError(
            f"{context} fields differ: missing={sorted(wanted - actual)}, "
            f"unknown={sorted(actual - wanted)}"
        )
    return value


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageDPlanError(f"{context} must be an object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise StageDPlanError(f"{context} must be an array")
    return value


def _require_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise StageDPlanError(f"{context} must be a boolean")
    return value


def _require_int(
    value: Any, context: str, *, minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise StageDPlanError(f"{context} must be an integer (not a boolean)")
    if minimum is not None and value < minimum:
        raise StageDPlanError(f"{context} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise StageDPlanError(f"{context} must be <= {maximum}")
    return value


def _require_number(
    value: Any, context: str, *, positive: bool = False,
) -> int | float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise StageDPlanError(f"{context} must be a finite number")
    if not math.isfinite(float(value)):
        raise StageDPlanError(f"{context} must be a finite number")
    if positive and value <= 0:
        raise StageDPlanError(f"{context} must be positive")
    if not positive and value < 0:
        raise StageDPlanError(f"{context} must be non-negative")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise StageDPlanError(f"{context} must be a non-empty string")
    return value


def _require_sha256(value: Any, context: str) -> str:
    text = _require_string(value, context)
    if _SHA256_RE.fullmatch(text) is None:
        raise StageDPlanError(f"{context} must be a lowercase SHA-256")
    return text


def _expect(actual: Any, expected: Any, context: str) -> None:
    """Compare frozen JSON values without accepting bool as an integer."""

    if isinstance(expected, Mapping):
        _exact_keys(actual, expected, context)
        for key, child in expected.items():
            _expect(actual[key], child, f"{context}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise StageDPlanError(f"{context} does not match the frozen array")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _expect(left, right, f"{context}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise StageDPlanError(
            f"{context} must equal frozen value {expected!r}"
        )


def _verify_self_digest(document: Mapping[str, Any], context: str) -> str:
    digest = _require_sha256(document.get("canonical_digest"), f"{context}.canonical_digest")
    if _semantic_digest(document) != digest:
        raise StageDPlanError(f"{context}.canonical_digest mismatch")
    return digest


_WORKLOAD = {
    "image_groups": IMAGE_GROUP_COUNT,
    "methods_per_image_group": 2,
    "execution_units": EXECUTION_UNIT_COUNT,
    "primary_clean_maps": PRIMARY_CLEAN_MAP_COUNT,
    "audit_repeat_clean_maps": AUDIT_REPEAT_CLEAN_MAP_COUNT,
    "map_work_items": MAP_WORK_ITEM_COUNT,
    "fidelity_curves": FIDELITY_CURVE_COUNT,
    "fidelity_curve_points": FIDELITY_CURVE_POINT_COUNT,
    "logical_work_items": LOGICAL_WORK_ITEM_COUNT,
    "logical_work_items_are_forward_or_backward_calls": False,
}

_RUN_FORBIDDEN = {
    "run_stability": False,
    "run_random_sanity": False,
    "run_center_sanity": False,
    "run_patch_scale_fidelity": False,
    "run_spatial_or_stability_endpoints": False,
    "run_secondary_inference": False,
    "run_display_pipeline": False,
    "other_method_ids": [],
    "other_block_head_coordinates": [],
    "class_logit_method_ids": [],
}

_CONFIG_FORBIDDEN = {
    "random_sanity": False,
    "center_sanity": False,
    "patch_scale_fidelity": False,
    "spatial_or_stability_endpoints": False,
    "other_methods": [],
    "other_blocks_or_heads": [],
    "class_logit_methods": [],
    "display_pipeline": False,
    "secondary_inference": False,
}

_SHARDING = {
    "shard_count_status": SHARD_COUNT_STATUS,
    "group_assignment_formula": SHARD_ASSIGNMENT_FORMULA,
    "image_group_not_split_across_shards": True,
    "single_coordinator_serial_shard_handoff": True,
    "d2_freezes_assignment_algorithm_not_shard_count": True,
}

LAUNCH_SAFETY_CAPS = MappingProxyType({
    "l0_l1_bytes_per_execution_unit_max": 67_108_864,
    "l2_bytes_per_audit_image_max": 1_073_741_824,
    "forward_calls_per_execution_unit_max": 30_000,
    "backward_calls_per_execution_unit_max": 30_000,
    "cuda_peak_reserved_bytes_per_execution_unit_max": 96_636_764_160,
    "projected_completed_campaign_gate_must_pass_before_launch": True,
    "preflight_failure_action": "BLOCKED_NO_LAUNCH",
    "runtime_breach_action": (
        "STOP_NEW_CLAIMS_RETAIN_EVIDENCE_AND_BLOCK_WORKFLOW"
    ),
    "runtime_breach_is_no_go_cost": False,
})

COMPLETED_CAMPAIGN_GATES = MappingProxyType({
    "stage_d_total_retained_regular_file_bytes_max": 68_719_476_736,
    "stage_d_total_h20_gpu_hours_max": 2000.0,
    "stage_d_h20_gpu_hours_per_image_group_max": 4.0,
    "actual_cost_includes_all_attempts_and_audit_repeats": True,
    "evaluated_only_after_complete_campaign": True,
    "aggregate_overrun_alone_stops_in_progress_campaign": False,
    "failure_status": "FAIL_COST",
    "decision_if_validity_and_fidelity_pass": "NO_GO_COST",
})


def _counts() -> dict[str, int]:
    return {
        "image_groups": IMAGE_GROUP_COUNT,
        "execution_units": EXECUTION_UNIT_COUNT,
        "primary_clean_maps": PRIMARY_CLEAN_MAP_COUNT,
        "audit_repeat_clean_maps": AUDIT_REPEAT_CLEAN_MAP_COUNT,
        "map_work_items": MAP_WORK_ITEM_COUNT,
        "fidelity_curves": FIDELITY_CURVE_COUNT,
        "fidelity_curve_points": FIDELITY_CURVE_POINT_COUNT,
        "logical_work_items": LOGICAL_WORK_ITEM_COUNT,
        "statistical_n": STATISTICAL_N,
        "statistical_n_increment": 0,
    }


def _validate_config(document: Mapping[str, Any]) -> str:
    if document.get("schema_version") != 1 or document.get("document_type") != "stage_d_fidelity_protocol_config":
        raise StageDPlanError("Stage D config identity is invalid")
    if document.get("stage") != "D" or document.get("milestone") != "D2":
        raise StageDPlanError("Stage D config stage/milestone is invalid")
    digest = _verify_self_digest(document, "Stage D config")
    _expect(
        digest, FROZEN_CONFIG_CANONICAL_DIGEST,
        "Stage D config frozen canonical digest",
    )
    cohort = _require_mapping(document.get("cohort"), "config.cohort")
    for key, expected in (
        ("registry_row_count", 500), ("statistical_n", 500),
        ("selection_order_min", 0), ("selection_order_max", 499),
        ("selection_order_contiguous_and_unique", True),
        ("c10_and_stage_d_pooling_allowed", False),
        ("misclassified_samples_retained", True),
        ("sample_replacement_allowed", False),
        ("complete_case_analysis_allowed", False),
    ):
        _expect(cohort.get(key), expected, f"config.cohort.{key}")
    primary = _require_mapping(document.get("primary_scope"), "config.primary_scope")
    for key, expected in (
        ("block_index", BLOCK_INDEX), ("head_index", HEAD_INDEX),
        ("target_id", TARGET_ID),
        ("methods_in_image_group_order", list(METHOD_IDS)),
        ("endpoint_id", ENDPOINT_ID), ("statistical_unit", "image"),
    ):
        _expect(primary.get(key), expected, f"config.primary_scope.{key}")
    target = _require_mapping(document.get("target_and_map"), "config.target_and_map")
    target_expected = {
        "query_token_index": 0, "head_dimension": 64,
        "replacement_baseline_id": "normalized_zero_canonical_rgb_0_5",
        "replacement_baseline_normalized_value": 0.0,
        "replacement_baseline_canonical_rgb_value": 0.5,
        "rgb_channels_replaced_together": True,
        "clean_attention_source": "clean_real_forward_attention_used",
        "baseline_uses_clean_fixed_attention": True,
        "attention_direction_detached_and_frozen": True,
        "direction_formula_id": "fixed_A_endpoint_difference_l2_unit",
        "direction_norm_relative_threshold": 0.000001,
        "direction_norm_scale_floor": 1.0,
        "ig_quadrature": "gauss_legendre",
        "ig_steps_schedule": [512, 1024, 2048, 4096, 8192],
        "ig_step_chunk_size": 8,
        "map_compute_and_storage_dtype": "float32",
        "metric_accumulation_dtype": "float64",
        "nns_native_map": "signed_float32_224x224_rgb_channel_sum",
        "matched_native_map": "signed_float32_14x14_patch_mass_plus_separate_cls_scalar",
        "fine_cell_operator": "nonoverlapping_4x4_sum",
        "matched_patch_to_fine_cell_operator": "uniform_mass_divide_by_16",
        "fine_cell_count": FINE_CELL_COUNT,
        "projection_mass_atol": 0.000001,
        "projection_mass_rtol": 0.000001,
        "cls_contribution_retained_separately": True,
        "native_single_pixel_claim_allowed": False,
    }
    for key, expected in target_expected.items():
        _expect(target.get(key), expected, f"config.target_and_map.{key}")
    fidelity = _require_mapping(document.get("fidelity"), "config.fidelity")
    fidelity_expected = {
        "directions": list(DIRECTIONS),
        "curve_fraction_count": len(CURVE_FRACTIONS),
        "curve_fractions": list(CONFIG_CURVE_FRACTIONS),
        "unit_count_rounding": "k0_zero_else_ceil_fraction_times_3136",
        "auc_abscissa": "actual_k_divided_by_3136",
        "auc_integration": "trapezoidal",
        "target_response_normalization": "signed_endpoint_delta_to_baseline_zero_clean_one",
        "target_delta_relative_threshold": 0.000001,
        "response_clipping_allowed": False,
        "ranking_evidence": "positive_part_after_target_orientation",
        "tie_repeat_count": len(TIE_REPEAT_INDICES),
        "tie_repeat_indices": list(TIE_REPEAT_INDICES),
        "tie_seed_base": 20260821,
        "tie_seed_namespace": "stage-d-fidelity|20260821|",
        "tie_seed_material": "namespace + sample_id + '|' + block_index + '|' + head_index + '|' + repeat_index",
        "tie_seed_hash_extraction": "sha256_first_8_bytes_big_endian_unsigned",
        "tie_seed_golden_first_sample_repeat_0_decimal": "6554433287525610283",
        "tie_priority_algorithm": "PCG64_PERMUTATION_THEN_LEXSORT_SCORE_DESC_PRIORITY_ASC",
        "tie_priority_paired_across_methods_and_directions": True,
        "within_image_auc_aggregation": "arithmetic_mean_over_complete_five_repeats",
        "within_image_auc_repeat_sd": "population_ddof_0_over_complete_five_repeats",
    }
    for key, expected in fidelity_expected.items():
        _expect(fidelity.get(key), expected, f"config.fidelity.{key}")
    _validate_config_audit(document.get("audit"), "config.audit")
    workload = _require_mapping(document.get("workload"), "config.workload")
    for key, expected in _WORKLOAD.items():
        _expect(workload.get(key), expected, f"config.workload.{key}")
    _expect(document.get("forbidden_workload"), _CONFIG_FORBIDDEN, "config.forbidden_workload")
    _expect(document.get("sharding"), _SHARDING, "config.sharding")
    cost = _require_mapping(document.get("cost"), "config.cost")
    _expect(cost.get("old_c10_pass_cost_authorizes_stage_d"), False, "config.cost.old_c10_pass_cost_authorizes_stage_d")
    _expect(cost.get("d7_independent_budget_required"), True, "config.cost.d7_independent_budget_required")
    _expect(cost.get("d7_may_loosen_caps"), False, "config.cost.d7_may_loosen_caps")
    _expect(cost.get("time_projection_safety_factor"), 1.25, "config.cost.time_projection_safety_factor")
    _expect(cost.get("storage_control_provenance_overhead_factor"), 1.25, "config.cost.storage_control_provenance_overhead_factor")
    _expect(cost.get("launch_safety_caps"), LAUNCH_SAFETY_CAPS, "config.cost.launch_safety_caps")
    _expect(cost.get("completed_campaign_cost_gate"), COMPLETED_CAMPAIGN_GATES, "config.cost.completed_campaign_cost_gate")
    _expect(cost.get("budget_status_at_d2"), "NOT_EVALUATED", "config.cost.budget_status_at_d2")
    projection = _require_mapping(cost.get("d7_projection_formula"), "config.cost.d7_projection_formula")
    _expect(projection.get("logical_work_items_used_as_call_count"), False, "config.cost.d7_projection_formula.logical_work_items_used_as_call_count")
    return digest


def _validate_config_audit(value: Any, context: str) -> None:
    audit = _require_mapping(value, context)
    fixed = {
        "audit_image_count": 5,
        "audit_repeatability_runs": 2,
        "primary_clean_run_included": True,
        "additional_repeat_clean_runs_per_method": 1,
        "repeatability_scope": "clean_metric_ready_map_only",
        "statistical_n_increment": 0,
        "l2_only_for_audit_ids": True,
        "audit_or_l2_enters_primary_statistics_or_decision": False,
    }
    for key, expected in fixed.items():
        _expect(audit.get(key), expected, f"{context}.{key}")
    for key in ("audit_overlay_ids", "audit_sample_ids", "audit_selection_orders"):
        values = _require_list(audit.get(key), f"{context}.{key}")
        if len(values) != 5 or len(set(values)) != 5:
            raise StageDPlanError(f"{context}.{key} must contain five unique values")
    for value in audit["audit_selection_orders"]:
        _require_int(value, f"{context}.audit_selection_orders[]", minimum=0, maximum=499)


def _registry_document(value: Any) -> dict[str, Any]:
    direct = getattr(value, "registry_document", None)
    if isinstance(direct, Mapping):
        return _deep_thaw(direct)
    document = _coerce_mapping(value, "registry")
    if "rows" in document:
        return document
    for key in ("registry", "registry_document", "document"):
        nested = document.get(key)
        if isinstance(nested, Mapping) and "rows" in nested:
            return _deep_thaw(nested)
    raise StageDPlanError("registry mapping does not contain a rows array")


def _validate_registry(document: Mapping[str, Any]) -> tuple[str, str, tuple[dict[str, Any], ...]]:
    if document.get("schema_version") != 1 or document.get("document_type") != "stage_d_fidelity_v1_registry":
        raise StageDPlanError("Stage D registry identity is invalid")
    if document.get("package_id") != "STAGE_D_FIDELITY_V1" or document.get("status") != "FROZEN_METADATA_ONLY":
        raise StageDPlanError("Stage D registry package/status is invalid")
    digest = _verify_self_digest(document, "Stage D registry")
    _expect(
        digest, FROZEN_REGISTRY_CANONICAL_DIGEST,
        "Stage D registry frozen canonical digest",
    )
    rows = _require_list(document.get("rows"), "registry.rows")
    if len(rows) != IMAGE_GROUP_COUNT:
        raise StageDPlanError("registry must contain exactly 500 rows")
    validated: list[dict[str, Any]] = []
    identities: dict[str, set[str]] = {
        "sample_id": set(), "relative_path": set(), "image_sha256": set(),
    }
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"registry.rows[{index}]")
        row_digest = _verify_self_digest(row, f"registry.rows[{index}]")
        _expect(row.get("schema_version"), 1, f"registry.rows[{index}].schema_version")
        _expect(row.get("selection_order"), index, f"registry.rows[{index}].selection_order")
        _expect(row.get("role"), "STAGE_D_FIDELITY_PRIMARY_COHORT", f"registry.rows[{index}].role")
        sample_id = _require_string(row.get("sample_id"), f"registry.rows[{index}].sample_id")
        relative_path = _require_string(row.get("relative_path"), f"registry.rows[{index}].relative_path")
        image_sha256 = _require_sha256(row.get("image_sha256"), f"registry.rows[{index}].image_sha256")
        _require_sha256(row.get("candidate_row_canonical_digest"), f"registry.rows[{index}].candidate_row_canonical_digest")
        audit = _require_bool(row.get("audit_overlay"), f"registry.rows[{index}].audit_overlay")
        overlay = row.get("audit_overlay_id")
        if audit:
            _require_string(overlay, f"registry.rows[{index}].audit_overlay_id")
        elif overlay is not None:
            raise StageDPlanError(
                f"registry.rows[{index}] non-audit row has an audit overlay id"
            )
        for name, item in (
            ("sample_id", sample_id), ("relative_path", relative_path),
            ("image_sha256", image_sha256),
        ):
            if item in identities[name]:
                raise StageDPlanError(f"registry contains duplicate {name}")
            identities[name].add(item)
        copied = _deep_thaw(row)
        copied["canonical_digest"] = row_digest
        validated.append(copied)
    ordered_digest = canonical_sha256([row["sample_id"] for row in validated])
    _expect(document.get("ordered_sample_ids_sha256"), ordered_digest, "registry.ordered_sample_ids_sha256")
    _expect(
        ordered_digest, FROZEN_ORDERED_SAMPLE_IDS_DIGEST,
        "registry frozen ordered sample IDs digest",
    )
    expected_rows_digest = canonical_sha256({"rows": validated})
    _expect(document.get("rows_canonical_digest"), expected_rows_digest, "registry.rows_canonical_digest")
    cohort = _require_mapping(document.get("cohort"), "registry.cohort")
    for key, expected in (("sample_count", 500), ("statistical_n", 500), ("selection_order_min", 0), ("selection_order_max", 499), ("candidate_order_preserved", True), ("rows_reselected", False), ("rows_replaced_or_supplemented", False)):
        _expect(cohort.get(key), expected, f"registry.cohort.{key}")
    return digest, ordered_digest, tuple(validated)


def _validate_run_spec(
    document: Mapping[str, Any], *, config_digest: str, registry_digest: str,
    ordered_sample_ids_digest: str, rows: Sequence[Mapping[str, Any]],
) -> str:
    if document.get("schema_version") != 1 or document.get("document_type") != "stage_d_fidelity_run_spec":
        raise StageDPlanError("Stage D run-spec identity is invalid")
    if document.get("stage") != "D" or document.get("milestone") != "D2":
        raise StageDPlanError("Stage D run-spec stage/milestone is invalid")
    digest = _verify_self_digest(document, "Stage D run spec")
    _expect(
        digest, FROZEN_RUN_SPEC_CANONICAL_DIGEST,
        "Stage D run-spec frozen canonical digest",
    )
    config_binding = _require_mapping(document.get("config_binding"), "run_spec.config_binding")
    _expect(config_binding.get("config_canonical_digest"), config_digest, "run_spec.config_binding.config_canonical_digest")
    _expect(config_binding.get("config_file_sha256"), FROZEN_CONFIG_FILE_SHA256, "run_spec.config_binding.config_file_sha256")
    registry_binding = _require_mapping(document.get("registry_binding"), "run_spec.registry_binding")
    _expect(registry_binding.get("package_id"), "STAGE_D_FIDELITY_V1", "run_spec.registry_binding.package_id")
    _expect(registry_binding.get("registry_canonical_digest"), registry_digest, "run_spec.registry_binding.registry_canonical_digest")
    _expect(registry_binding.get("ordered_sample_ids_sha256"), ordered_sample_ids_digest, "run_spec.registry_binding.ordered_sample_ids_sha256")
    _expect(registry_binding.get("registry_file_sha256"), FROZEN_REGISTRY_FILE_SHA256, "run_spec.registry_binding.registry_file_sha256")
    selection = _require_mapping(document.get("selection"), "run_spec.selection")
    _expect(selection.get("selection_mode"), "ALL_REGISTRY_ROWS_IN_EXACT_SELECTION_ORDER", "run_spec.selection.selection_mode")
    _expect(selection.get("row_count"), 500, "run_spec.selection.row_count")
    _expect(selection.get("selection_order_min"), 0, "run_spec.selection.selection_order_min")
    _expect(selection.get("selection_order_max"), 499, "run_spec.selection.selection_order_max")
    _expect(selection.get("sample_ids"), [row["sample_id"] for row in rows], "run_spec.selection.sample_ids")
    execution = _require_mapping(document.get("execution_coordinates"), "run_spec.execution_coordinates")
    execution_expected = {
        "block_index": 0, "head_index": 9, "target_id": TARGET_ID,
        "method_ids": list(METHOD_IDS),
        "method_order_within_image_group": "AS_LISTED",
        "statistical_unit": "image",
    }
    _expect(execution, execution_expected, "run_spec.execution_coordinates")
    fidelity = _require_mapping(document.get("fidelity"), "run_spec.fidelity")
    fidelity_expected = {
        "endpoint_id": ENDPOINT_ID,
        "spatial_scope": "FINE_4X4_CELLS_ONLY",
        "fine_cell_count": FINE_CELL_COUNT,
        "directions": list(DIRECTIONS),
        "curve_fractions": list(CURVE_FRACTIONS),
        "tie_repeat_indices": list(TIE_REPEAT_INDICES),
        "tie_priority_paired_across_methods_and_directions": True,
        "registered_derived_fraction": 0.2,
        "registered_derived_metrics_additional_curve_points_or_work_items": 0,
    }
    _expect(fidelity, fidelity_expected, "run_spec.fidelity")
    audit = _require_mapping(document.get("audit"), "run_spec.audit")
    audit_expected = {
        "audit_image_count": 5,
        "audit_overlay_ids": [], "audit_sample_ids": [],
        "audit_selection_orders": [],
        "audit_repeatability_runs": 2,
        "primary_clean_run_included": True,
        "additional_repeat_clean_runs_per_method": 1,
        "repeatability_scope": "clean_metric_ready_map_only",
        "statistical_n_increment": 0,
    }
    for key, expected in audit_expected.items():
        if expected == []:
            values = _require_list(audit.get(key), f"run_spec.audit.{key}")
            if len(values) != 5 or len(set(values)) != 5:
                raise StageDPlanError(f"run_spec.audit.{key} must contain five unique values")
        else:
            _expect(audit.get(key), expected, f"run_spec.audit.{key}")
    _expect(document.get("workload"), _WORKLOAD, "run_spec.workload")
    _expect(document.get("forbidden_workload"), _RUN_FORBIDDEN, "run_spec.forbidden_workload")
    _expect(document.get("sharding"), _SHARDING, "run_spec.sharding")
    return digest


def _identity_payload(document_type: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": document_type,
        **fields,
    }


def _identity_digest(document_type: str, fields: Mapping[str, Any]) -> str:
    return canonical_sha256(_identity_payload(document_type, **dict(fields)))


def _unchecked_frozen(cls: type[Any], fields: Mapping[str, Any]) -> Any:
    """Construct a frozen record whose fields were derived internally."""

    result = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    return result


@dataclass(frozen=True, slots=True)
class StageDPairBinding:
    """Scientific context shared by the two units in one image group."""

    sample_id: str
    selection_order: int
    image_sha256: str
    model_context_digest: str
    target_context_digest: str
    fidelity_context_digest: str
    plan_context_digest: str
    canonical_digest: str

    @classmethod
    def create(
        cls, *, sample_id: str, selection_order: int, image_sha256: str,
        model_context_digest: str, target_context_digest: str,
        fidelity_context_digest: str, plan_context_digest: str,
    ) -> "StageDPairBinding":
        fields = {
            "sample_id": sample_id, "selection_order": selection_order,
            "image_sha256": image_sha256,
            "model_context_digest": model_context_digest,
            "target_context_digest": target_context_digest,
            "fidelity_context_digest": fidelity_context_digest,
            "plan_context_digest": plan_context_digest,
        }
        return cls(
            **fields,
            canonical_digest=_identity_digest("stage_d_pair_binding", fields),
        )

    def __post_init__(self) -> None:
        _require_string(self.sample_id, "pair_binding.sample_id")
        _require_int(self.selection_order, "pair_binding.selection_order", minimum=0, maximum=499)
        for name in ("image_sha256", "model_context_digest", "target_context_digest", "fidelity_context_digest", "plan_context_digest", "canonical_digest"):
            _require_sha256(getattr(self, name), f"pair_binding.{name}")
        fields = {
            "sample_id": self.sample_id,
            "selection_order": self.selection_order,
            "image_sha256": self.image_sha256,
            "model_context_digest": self.model_context_digest,
            "target_context_digest": self.target_context_digest,
            "fidelity_context_digest": self.fidelity_context_digest,
            "plan_context_digest": self.plan_context_digest,
        }
        expected = _identity_digest("stage_d_pair_binding", fields)
        if self.canonical_digest != expected:
            raise StageDPlanError("pair binding canonical_digest mismatch")

    @property
    def digest(self) -> str:
        return self.canonical_digest

    def to_dict(self) -> dict[str, Any]:
        return _identity_payload(
            "stage_d_pair_binding", sample_id=self.sample_id,
            selection_order=self.selection_order, image_sha256=self.image_sha256,
            model_context_digest=self.model_context_digest,
            target_context_digest=self.target_context_digest,
            fidelity_context_digest=self.fidelity_context_digest,
            plan_context_digest=self.plan_context_digest,
            canonical_digest=self.canonical_digest,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDPairBinding":
        keys = {
            "schema_version", "document_type", "sample_id",
            "selection_order", "image_sha256", "model_context_digest",
            "target_context_digest", "fidelity_context_digest",
            "plan_context_digest",
            "canonical_digest",
        }
        _exact_keys(value, keys, "pair binding")
        _expect(value["schema_version"], 1, "pair binding.schema_version")
        _expect(value["document_type"], "stage_d_pair_binding", "pair binding.document_type")
        return cls(
            sample_id=value["sample_id"],
            selection_order=value["selection_order"],
            image_sha256=value["image_sha256"],
            model_context_digest=value["model_context_digest"],
            target_context_digest=value["target_context_digest"],
            fidelity_context_digest=value["fidelity_context_digest"],
            plan_context_digest=value["plan_context_digest"],
            canonical_digest=value["canonical_digest"],
        )


@dataclass(frozen=True, slots=True)
class StageDUnit:
    """One content-addressed method unit in an image group."""

    sample_id: str
    selection_order: int
    method_index: int
    method_id: str
    block_index: int
    head_index: int
    target_id: str
    pair_binding_digest: str
    audit: bool
    audit_overlay_id: str | None
    canonical_digest: str

    @classmethod
    def create(
        cls, *, sample_id: str, selection_order: int, method_index: int,
        method_id: str, pair_binding_digest: str, audit: bool,
        audit_overlay_id: str | None,
    ) -> "StageDUnit":
        fields = {
            "sample_id": sample_id, "selection_order": selection_order,
            "method_index": method_index, "method_id": method_id,
            "block_index": BLOCK_INDEX, "head_index": HEAD_INDEX,
            "target_id": TARGET_ID,
            "pair_binding_digest": pair_binding_digest, "audit": audit,
            "audit_overlay_id": audit_overlay_id,
        }
        return cls(
            **fields,
            canonical_digest=_identity_digest("stage_d_execution_unit", fields),
        )

    def __post_init__(self) -> None:
        _require_string(self.sample_id, "unit.sample_id")
        _require_int(self.selection_order, "unit.selection_order", minimum=0, maximum=499)
        _require_int(self.method_index, "unit.method_index", minimum=0, maximum=1)
        _require_int(self.block_index, "unit.block_index", minimum=0, maximum=0)
        _require_int(self.head_index, "unit.head_index", minimum=9, maximum=9)
        if self.method_id != METHOD_IDS[self.method_index]:
            raise StageDPlanError("unit method index/order is not NNS then matched")
        if self.block_index != BLOCK_INDEX or self.head_index != HEAD_INDEX or self.target_id != TARGET_ID:
            raise StageDPlanError("unit primary coordinate drift")
        _require_sha256(self.pair_binding_digest, "unit.pair_binding_digest")
        _require_bool(self.audit, "unit.audit")
        if self.audit:
            _require_string(self.audit_overlay_id, "unit.audit_overlay_id")
        elif self.audit_overlay_id is not None:
            raise StageDPlanError("non-audit unit has an audit overlay")
        _require_sha256(self.canonical_digest, "unit.canonical_digest")
        fields = {
            "sample_id": self.sample_id,
            "selection_order": self.selection_order,
            "method_index": self.method_index, "method_id": self.method_id,
            "block_index": self.block_index, "head_index": self.head_index,
            "target_id": self.target_id,
            "pair_binding_digest": self.pair_binding_digest,
            "audit": self.audit, "audit_overlay_id": self.audit_overlay_id,
        }
        expected = _identity_digest("stage_d_execution_unit", fields)
        if expected != self.canonical_digest:
            raise StageDPlanError("unit canonical_digest mismatch")

    @property
    def digest(self) -> str:
        return self.canonical_digest

    def to_dict(self) -> dict[str, Any]:
        return _identity_payload(
            "stage_d_execution_unit", sample_id=self.sample_id,
            selection_order=self.selection_order, method_index=self.method_index,
            method_id=self.method_id, block_index=self.block_index,
            head_index=self.head_index, target_id=self.target_id,
            pair_binding_digest=self.pair_binding_digest, audit=self.audit,
            audit_overlay_id=self.audit_overlay_id,
            canonical_digest=self.canonical_digest,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDUnit":
        keys = {
            "schema_version", "document_type", "sample_id",
            "selection_order", "method_index", "method_id",
            "block_index", "head_index", "target_id",
            "pair_binding_digest", "audit", "audit_overlay_id",
            "canonical_digest",
        }
        _exact_keys(value, keys, "execution unit")
        _expect(value["schema_version"], 1, "execution unit.schema_version")
        _expect(value["document_type"], "stage_d_execution_unit", "execution unit.document_type")
        return cls(**{key: value[key] for key in keys - {"schema_version", "document_type"}})


@dataclass(frozen=True, slots=True)
class StageDMapCoordinate:
    """A primary or audit-repeat metric-ready map coordinate."""

    unit_digest: str
    pair_binding_digest: str
    sample_id: str
    selection_order: int
    method_id: str
    block_index: int
    head_index: int
    target_id: str
    map_kind: str
    clean_run_index: int
    audit: bool
    audit_overlay_id: str | None
    statistical_n_increment: int
    canonical_digest: str

    @classmethod
    def create(cls, unit: StageDUnit, *, repeat: bool) -> "StageDMapCoordinate":
        if not isinstance(unit, StageDUnit):
            raise TypeError("map coordinate requires a StageDUnit")
        _require_bool(repeat, "map coordinate repeat")
        fields = {
            "unit_digest": unit.canonical_digest,
            "pair_binding_digest": unit.pair_binding_digest,
            "sample_id": unit.sample_id,
            "selection_order": unit.selection_order,
            "method_id": unit.method_id, "block_index": BLOCK_INDEX,
            "head_index": HEAD_INDEX, "target_id": TARGET_ID,
            "map_kind": "audit_repeat_clean_map" if repeat else "primary_clean_map",
            "clean_run_index": 1 if repeat else 0,
            "audit": unit.audit, "audit_overlay_id": unit.audit_overlay_id,
            "statistical_n_increment": 0,
        }
        return _unchecked_frozen(cls, {
            **fields,
            "canonical_digest": _identity_digest(
                "stage_d_map_coordinate", fields
            ),
        })

    def __post_init__(self) -> None:
        for name in ("unit_digest", "pair_binding_digest", "canonical_digest"):
            _require_sha256(getattr(self, name), f"map coordinate.{name}")
        _require_string(self.sample_id, "map coordinate.sample_id")
        _require_int(self.selection_order, "map coordinate.selection_order", minimum=0, maximum=499)
        _require_bool(self.audit, "map coordinate.audit")
        _require_int(self.block_index, "map coordinate.block_index", minimum=0, maximum=0)
        _require_int(self.head_index, "map coordinate.head_index", minimum=9, maximum=9)
        if self.audit:
            _require_string(
                self.audit_overlay_id, "map coordinate.audit_overlay_id"
            )
        elif self.audit_overlay_id is not None:
            raise StageDPlanError(
                "non-audit map coordinate has an audit overlay id"
            )
        if self.method_id not in METHOD_IDS or self.block_index != 0 or self.head_index != 9 or self.target_id != TARGET_ID:
            raise StageDPlanError("map coordinate scientific context drift")
        if self.map_kind == "primary_clean_map":
            if self.clean_run_index != 0:
                raise StageDPlanError("primary clean map must be run zero")
        elif self.map_kind == "audit_repeat_clean_map":
            if self.clean_run_index != 1 or not self.audit or self.audit_overlay_id is None:
                raise StageDPlanError("repeat clean map must be the one audit-only repeat")
        else:
            raise StageDPlanError("map coordinate kind is forbidden")
        if self.statistical_n_increment != 0:
            raise StageDPlanError("map coordinate may not increment statistical n")
        unit_fields = {
            "sample_id": self.sample_id,
            "selection_order": self.selection_order,
            "method_index": METHOD_IDS.index(self.method_id),
            "method_id": self.method_id, "block_index": self.block_index,
            "head_index": self.head_index, "target_id": self.target_id,
            "pair_binding_digest": self.pair_binding_digest,
            "audit": self.audit, "audit_overlay_id": self.audit_overlay_id,
        }
        if _identity_digest("stage_d_execution_unit", unit_fields) != self.unit_digest:
            raise StageDPlanError("map coordinate unit_digest mismatch")
        fields = {name: getattr(self, name) for name in (
            "unit_digest", "pair_binding_digest", "sample_id",
            "selection_order", "method_id", "block_index",
            "head_index", "target_id", "map_kind",
            "clean_run_index", "audit", "audit_overlay_id",
            "statistical_n_increment",
        )}
        expected = _identity_digest("stage_d_map_coordinate", fields)
        if expected != self.canonical_digest:
            raise StageDPlanError("map coordinate canonical_digest mismatch")

    @property
    def kind(self) -> str:
        return self.map_kind

    @property
    def digest(self) -> str:
        return self.canonical_digest

    def to_dict(self) -> dict[str, Any]:
        return _identity_payload(
            "stage_d_map_coordinate",
            **{name: getattr(self, name) for name in (
                "unit_digest", "pair_binding_digest", "sample_id",
                "selection_order", "method_id", "block_index",
                "head_index", "target_id", "map_kind",
                "clean_run_index", "audit", "audit_overlay_id",
                "statistical_n_increment", "canonical_digest",
            )},
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDMapCoordinate":
        fields = (
            "unit_digest", "pair_binding_digest", "sample_id",
            "selection_order", "method_id", "block_index",
            "head_index", "target_id", "map_kind",
            "clean_run_index", "audit", "audit_overlay_id",
            "statistical_n_increment", "canonical_digest",
        )
        _exact_keys(value, {"schema_version", "document_type", *fields}, "map coordinate")
        _expect(value["schema_version"], 1, "map coordinate.schema_version")
        _expect(value["document_type"], "stage_d_map_coordinate", "map coordinate.document_type")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True, slots=True)
class StageDCurvePoint:
    """One fine-scale intervention response coordinate."""

    unit_digest: str
    pair_binding_digest: str
    sample_id: str
    selection_order: int
    method_id: str
    block_index: int
    head_index: int
    target_id: str
    endpoint_id: str
    fidelity_scale: str
    fine_cell_count: int
    direction: str
    repeat_index: int
    fraction_index: int
    fraction: int | float
    intervention_count: int
    canonical_digest: str

    @classmethod
    def create(
        cls, unit: StageDUnit, *, direction: str, repeat_index: int,
        fraction_index: int,
    ) -> "StageDCurvePoint":
        if not isinstance(unit, StageDUnit):
            raise TypeError("curve point requires a StageDUnit")
        if direction not in DIRECTIONS:
            raise StageDPlanError("curve point direction is forbidden")
        if repeat_index not in TIE_REPEAT_INDICES or type(repeat_index) is not int:
            raise StageDPlanError("curve point repeat index is forbidden")
        _require_int(
            fraction_index, "curve point.fraction_index", minimum=0, maximum=20
        )
        fraction = CURVE_FRACTIONS[fraction_index]
        fields = {
            "unit_digest": unit.canonical_digest,
            "pair_binding_digest": unit.pair_binding_digest,
            "sample_id": unit.sample_id,
            "selection_order": unit.selection_order,
            "method_id": unit.method_id, "block_index": BLOCK_INDEX,
            "head_index": HEAD_INDEX, "target_id": TARGET_ID,
            "endpoint_id": ENDPOINT_ID, "fidelity_scale": FIDELITY_SCALE,
            "fine_cell_count": FINE_CELL_COUNT, "direction": direction,
            "repeat_index": repeat_index, "fraction_index": fraction_index,
            "fraction": fraction,
            "intervention_count": (
                0 if fraction_index == 0 else math.ceil(float(fraction) * FINE_CELL_COUNT)
            ),
        }
        return _unchecked_frozen(cls, {
            **fields,
            "canonical_digest": _identity_digest(
                "stage_d_fidelity_curve_point", fields
            ),
        })

    def __post_init__(self) -> None:
        for name in ("unit_digest", "pair_binding_digest", "canonical_digest"):
            _require_sha256(getattr(self, name), f"curve point.{name}")
        _require_string(self.sample_id, "curve point.sample_id")
        _require_int(self.selection_order, "curve point.selection_order", minimum=0, maximum=499)
        _require_int(self.block_index, "curve point.block_index", minimum=0, maximum=0)
        _require_int(self.head_index, "curve point.head_index", minimum=9, maximum=9)
        _require_int(self.fine_cell_count, "curve point.fine_cell_count", minimum=3136, maximum=3136)
        _require_int(self.repeat_index, "curve point.repeat_index", minimum=0, maximum=4)
        if self.method_id not in METHOD_IDS or self.block_index != 0 or self.head_index != 9 or self.target_id != TARGET_ID:
            raise StageDPlanError("curve point scientific context drift")
        if self.endpoint_id != ENDPOINT_ID or self.fidelity_scale != FIDELITY_SCALE or self.fine_cell_count != FINE_CELL_COUNT:
            raise StageDPlanError("only the frozen fine-4x4-cell endpoint is legal")
        if self.direction not in DIRECTIONS:
            raise StageDPlanError("curve point direction is forbidden")
        if self.repeat_index not in TIE_REPEAT_INDICES:
            raise StageDPlanError("curve point repeat index is forbidden")
        _require_int(self.fraction_index, "curve point.fraction_index", minimum=0, maximum=20)
        if type(self.fraction) is not type(CURVE_FRACTIONS[self.fraction_index]) or self.fraction != CURVE_FRACTIONS[self.fraction_index]:
            raise StageDPlanError("curve point fraction/index mismatch")
        expected_count = 0 if self.fraction_index == 0 else math.ceil(float(self.fraction) * 3136)
        _require_int(self.intervention_count, "curve point.intervention_count", minimum=0, maximum=3136)
        if self.intervention_count != expected_count:
            raise StageDPlanError("curve point intervention count mismatch")
        # Unit audit fields are not curve semantics.  Recompute the point digest
        # directly so audit and non-audit units keep the same coordinate schema.
        expected = canonical_sha256(_identity_payload(
            "stage_d_fidelity_curve_point",
            **{name: getattr(self, name) for name in (
                "unit_digest", "pair_binding_digest", "sample_id",
                "selection_order", "method_id", "block_index",
                "head_index", "target_id", "endpoint_id",
                "fidelity_scale", "fine_cell_count", "direction",
                "repeat_index", "fraction_index", "fraction",
                "intervention_count",
            )},
        ))
        if expected != self.canonical_digest:
            raise StageDPlanError("curve point canonical_digest mismatch")

    @property
    def kind(self) -> str:
        return "fidelity_curve_point"

    @property
    def digest(self) -> str:
        return self.canonical_digest

    @property
    def curve_key(self) -> tuple[str, str, str, int]:
        return (self.unit_digest, self.method_id, self.direction, self.repeat_index)

    def to_dict(self) -> dict[str, Any]:
        fields = (
            "unit_digest", "pair_binding_digest", "sample_id",
            "selection_order", "method_id", "block_index",
            "head_index", "target_id", "endpoint_id",
            "fidelity_scale", "fine_cell_count", "direction",
            "repeat_index", "fraction_index", "fraction",
            "intervention_count", "canonical_digest",
        )
        return _identity_payload(
            "stage_d_fidelity_curve_point",
            **{name: getattr(self, name) for name in fields},
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDCurvePoint":
        fields = (
            "unit_digest", "pair_binding_digest", "sample_id",
            "selection_order", "method_id", "block_index",
            "head_index", "target_id", "endpoint_id",
            "fidelity_scale", "fine_cell_count", "direction",
            "repeat_index", "fraction_index", "fraction",
            "intervention_count", "canonical_digest",
        )
        _exact_keys(value, {"schema_version", "document_type", *fields}, "curve point")
        _expect(value["schema_version"], 1, "curve point.schema_version")
        _expect(value["document_type"], "stage_d_fidelity_curve_point", "curve point.document_type")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True, slots=True)
class StageDImageGroup:
    """One ordered image plus its NNS/matched pair and shared binding."""

    selection_order: int
    sample_id: str
    relative_path: str
    image_sha256: str
    registry_row_digest: str
    audit: bool
    audit_overlay_id: str | None
    pair_binding: StageDPairBinding
    units: tuple[StageDUnit, StageDUnit]
    canonical_digest: str

    @classmethod
    def create(
        cls, *, selection_order: int, sample_id: str, relative_path: str,
        image_sha256: str, registry_row_digest: str, audit: bool,
        audit_overlay_id: str | None, pair_binding: StageDPairBinding,
        units: Sequence[StageDUnit],
    ) -> "StageDImageGroup":
        unit_tuple = tuple(units)
        if len(unit_tuple) != 2:
            raise StageDPlanError("image group must contain exactly two units")
        fields = {
            "selection_order": selection_order, "sample_id": sample_id,
            "relative_path": relative_path, "image_sha256": image_sha256,
            "registry_row_digest": registry_row_digest, "audit": audit,
            "audit_overlay_id": audit_overlay_id,
            "pair_binding": pair_binding.to_dict(),
            "units": [unit.to_dict() for unit in unit_tuple],
        }
        return cls(
            selection_order, sample_id, relative_path, image_sha256,
            registry_row_digest, audit, audit_overlay_id, pair_binding,
            unit_tuple, canonical_sha256(_identity_payload(
                "stage_d_image_group", **fields
            )),
        )

    def __post_init__(self) -> None:
        _require_int(self.selection_order, "image group.selection_order", minimum=0, maximum=499)
        _require_string(self.sample_id, "image group.sample_id")
        _require_string(self.relative_path, "image group.relative_path")
        _require_sha256(self.image_sha256, "image group.image_sha256")
        _require_sha256(self.registry_row_digest, "image group.registry_row_digest")
        _require_bool(self.audit, "image group.audit")
        if self.audit:
            _require_string(self.audit_overlay_id, "image group.audit_overlay_id")
        elif self.audit_overlay_id is not None:
            raise StageDPlanError("non-audit image group has an overlay id")
        if not isinstance(self.pair_binding, StageDPairBinding):
            raise StageDPlanError("image group pair binding is not typed")
        if not isinstance(self.units, tuple) or len(self.units) != 2:
            raise StageDPlanError("image group must contain an immutable unit pair")
        if tuple(unit.method_id for unit in self.units) != METHOD_IDS:
            raise StageDPlanError("image group units must be ordered NNS then matched")
        for unit in self.units:
            if not isinstance(unit, StageDUnit):
                raise StageDPlanError("image group unit is not typed")
            if (
                unit.sample_id != self.sample_id
                or unit.selection_order != self.selection_order
                or unit.pair_binding_digest != self.pair_binding.canonical_digest
                or unit.audit is not self.audit
                or unit.audit_overlay_id != self.audit_overlay_id
            ):
                raise StageDPlanError("image group pair/unit binding mismatch")
        if (
            self.pair_binding.sample_id != self.sample_id
            or self.pair_binding.selection_order != self.selection_order
            or self.pair_binding.image_sha256 != self.image_sha256
        ):
            raise StageDPlanError("image group pair binding does not bind the image")
        _require_sha256(self.canonical_digest, "image group.canonical_digest")
        fields = {
            "selection_order": self.selection_order,
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "image_sha256": self.image_sha256,
            "registry_row_digest": self.registry_row_digest,
            "audit": self.audit,
            "audit_overlay_id": self.audit_overlay_id,
            "pair_binding": self.pair_binding.to_dict(),
            "units": [unit.to_dict() for unit in self.units],
        }
        expected = _identity_digest("stage_d_image_group", fields)
        if expected != self.canonical_digest:
            raise StageDPlanError("image group canonical_digest mismatch")

    @property
    def digest(self) -> str:
        return self.canonical_digest

    @property
    def map_coordinates(self) -> tuple[StageDMapCoordinate, ...]:
        result: list[StageDMapCoordinate] = []
        for unit in self.units:
            result.append(StageDMapCoordinate.create(unit, repeat=False))
            if self.audit:
                result.append(StageDMapCoordinate.create(unit, repeat=True))
        return tuple(result)

    @property
    def work_items(self) -> Sequence[StageDMapCoordinate | StageDCurvePoint]:
        group = self

        class _GroupItems(Sequence[StageDMapCoordinate | StageDCurvePoint]):
            def __len__(self) -> int:
                return len(group.map_coordinates) + 420

            def __getitem__(self, index: int | slice) -> Any:
                if isinstance(index, slice):
                    return tuple(self[item] for item in range(*index.indices(len(self))))
                if type(index) is not int:
                    raise TypeError("group work-item index must be integer or slice")
                if index < 0:
                    index += len(self)
                maps = group.map_coordinates
                if index < 0 or index >= len(self):
                    raise IndexError(index)
                if index < len(maps):
                    return maps[index]
                return group.curve_point_at(index - len(maps))

        return _GroupItems()

    def curve_point_at(self, local_index: int) -> StageDCurvePoint:
        _require_int(local_index, "curve local index", minimum=0, maximum=419)
        per_unit = len(DIRECTIONS) * len(TIE_REPEAT_INDICES) * len(CURVE_FRACTIONS)
        method_index, offset = divmod(local_index, per_unit)
        direction_index, offset = divmod(
            offset, len(TIE_REPEAT_INDICES) * len(CURVE_FRACTIONS)
        )
        repeat_index, fraction_index = divmod(offset, len(CURVE_FRACTIONS))
        return StageDCurvePoint.create(
            self.units[method_index], direction=DIRECTIONS[direction_index],
            repeat_index=repeat_index, fraction_index=fraction_index,
        )

    def iter_curve_points(self) -> Iterator[StageDCurvePoint]:
        for local_index in range(420):
            yield self.curve_point_at(local_index)

    def to_dict(self) -> dict[str, Any]:
        return _identity_payload(
            "stage_d_image_group", selection_order=self.selection_order,
            sample_id=self.sample_id, relative_path=self.relative_path,
            image_sha256=self.image_sha256,
            registry_row_digest=self.registry_row_digest, audit=self.audit,
            audit_overlay_id=self.audit_overlay_id,
            pair_binding=self.pair_binding.to_dict(),
            units=[unit.to_dict() for unit in self.units],
            canonical_digest=self.canonical_digest,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDImageGroup":
        keys = {
            "schema_version", "document_type", "selection_order",
            "sample_id", "relative_path", "image_sha256",
            "registry_row_digest", "audit", "audit_overlay_id",
            "pair_binding", "units", "canonical_digest",
        }
        _exact_keys(value, keys, "image group")
        _expect(value["schema_version"], 1, "image group.schema_version")
        _expect(value["document_type"], "stage_d_image_group", "image group.document_type")
        units = _require_list(value["units"], "image group.units")
        return cls(
            selection_order=value["selection_order"], sample_id=value["sample_id"],
            relative_path=value["relative_path"], image_sha256=value["image_sha256"],
            registry_row_digest=value["registry_row_digest"], audit=value["audit"],
            audit_overlay_id=value["audit_overlay_id"],
            pair_binding=StageDPairBinding.from_dict(value["pair_binding"]),
            units=tuple(StageDUnit.from_dict(item) for item in units),
            canonical_digest=value["canonical_digest"],
        )


class StageDCoordinateSequence(Sequence[StageDMapCoordinate | StageDCurvePoint]):
    """Lazy, random-access view of all 211,010 ordered coordinates."""

    __slots__ = ("_groups", "_map_offsets")

    def __init__(self, groups: Sequence[StageDImageGroup]) -> None:
        self._groups = tuple(groups)
        if len(self._groups) != 500 or any(
            not isinstance(group, StageDImageGroup) for group in self._groups
        ):
            raise StageDPlanError(
                "coordinate sequence requires the 500 typed image groups"
            )
        offsets = [0]
        for group in self._groups:
            offsets.append(offsets[-1] + 2 + (2 if group.audit else 0))
        self._map_offsets = tuple(offsets)

    def __len__(self) -> int:
        return LOGICAL_WORK_ITEM_COUNT

    def __iter__(self) -> Iterator[StageDMapCoordinate | StageDCurvePoint]:
        # Maps are first to make their 1,010-count segment independently auditable.
        for group in self._groups:
            yield from group.map_coordinates
        for group in self._groups:
            yield from group.iter_curve_points()

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        if type(index) is not int:
            raise TypeError("coordinate index must be an integer or slice")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < MAP_WORK_ITEM_COUNT:
            group_index = bisect.bisect_right(self._map_offsets, index) - 1
            local = index - self._map_offsets[group_index]
            return self._groups[group_index].map_coordinates[local]
        curve_offset = index - MAP_WORK_ITEM_COUNT
        group_index, local = divmod(curve_offset, 420)
        return self._groups[group_index].curve_point_at(local)


def _stream_coordinate_digest(
    coordinates: Iterable[StageDMapCoordinate | StageDCurvePoint],
) -> str:
    digest = hashlib.sha256()
    digest.update(_COORDINATE_STREAM_DOMAIN)
    count = 0
    for count, coordinate in enumerate(coordinates, 1):
        payload = canonical_json_bytes(coordinate.to_dict())
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


def _dimensions_payload() -> dict[str, Any]:
    return {
        "block_index": BLOCK_INDEX, "head_index": HEAD_INDEX,
        "target_id": TARGET_ID, "endpoint_id": ENDPOINT_ID,
        "fidelity_scale": FIDELITY_SCALE, "fine_cell_count": FINE_CELL_COUNT,
        "methods_in_image_group_order": list(METHOD_IDS),
        "directions": list(DIRECTIONS),
        "tie_repeat_indices": list(TIE_REPEAT_INDICES),
        "curve_fractions": list(CURVE_FRACTIONS),
        "coordinate_order": (
            "all_maps_by_group_method_run_then_all_curve_points_by_"
            "group_method_direction_repeat_fraction"
        ),
        "shard_count_status": SHARD_COUNT_STATUS,
        "shard_assignment_formula": SHARD_ASSIGNMENT_FORMULA,
    }


@dataclass(frozen=True, slots=True)
class StageDExecutionPlan:
    schema_version: int
    document_type: str
    config_digest: str
    run_spec_digest: str
    registry_digest: str
    ordered_sample_ids_digest: str
    context_digest: str
    coordinate_digest: str
    counts: Mapping[str, int]
    dimensions: Mapping[str, Any]
    groups: tuple[StageDImageGroup, ...]
    canonical_digest: str
    _issuer: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_int(self.schema_version, "plan.schema_version", minimum=1, maximum=1)
        if self.document_type != PLAN_DOCUMENT_TYPE:
            raise StageDPlanError("Stage D execution-plan identity is invalid")
        for name in ("config_digest", "run_spec_digest", "registry_digest", "ordered_sample_ids_digest", "context_digest", "coordinate_digest", "canonical_digest"):
            _require_sha256(getattr(self, name), f"plan.{name}")
        object.__setattr__(self, "counts", _deep_freeze(self.counts))
        object.__setattr__(self, "dimensions", _deep_freeze(self.dimensions))
        if not isinstance(self.groups, tuple) or len(self.groups) != 500:
            raise StageDPlanError("plan must contain 500 immutable image groups")
        if any(not isinstance(group, StageDImageGroup) for group in self.groups):
            raise StageDPlanError("plan groups must be typed StageDImageGroup values")

    @property
    def digest(self) -> str:
        return self.canonical_digest

    @property
    def image_groups(self) -> tuple[StageDImageGroup, ...]:
        return self.groups

    @property
    def units(self) -> tuple[StageDUnit, ...]:
        return tuple(unit for group in self.groups for unit in group.units)

    @property
    def map_coordinates(self) -> tuple[StageDMapCoordinate, ...]:
        return tuple(item for group in self.groups for item in group.map_coordinates)

    @property
    def curve_points(self) -> Sequence[StageDCurvePoint]:
        plan = self

        class _CurvePoints(Sequence[StageDCurvePoint]):
            def __len__(self) -> int:
                return FIDELITY_CURVE_POINT_COUNT

            def __getitem__(self, index: int | slice) -> Any:
                if isinstance(index, slice):
                    return tuple(self[item] for item in range(*index.indices(len(self))))
                if type(index) is not int:
                    raise TypeError("curve-point index must be an integer or slice")
                if index < 0:
                    index += len(self)
                if index < 0 or index >= len(self):
                    raise IndexError(index)
                group_index, local = divmod(index, 420)
                return plan.groups[group_index].curve_point_at(local)

        return _CurvePoints()

    @property
    def coordinates(self) -> StageDCoordinateSequence:
        return StageDCoordinateSequence(self.groups)

    @property
    def work_items(self) -> StageDCoordinateSequence:
        return self.coordinates

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "config_digest": self.config_digest,
            "run_spec_digest": self.run_spec_digest,
            "registry_digest": self.registry_digest,
            "ordered_sample_ids_digest": self.ordered_sample_ids_digest,
            "context_digest": self.context_digest,
            "coordinate_digest": self.coordinate_digest,
            "counts": _deep_thaw(self.counts),
            "dimensions": _deep_thaw(self.dimensions),
            "groups": [group.to_dict() for group in self.groups],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["canonical_digest"] = self.canonical_digest
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDExecutionPlan":
        keys = {
            "schema_version", "document_type", "config_digest",
            "run_spec_digest", "registry_digest",
            "ordered_sample_ids_digest", "context_digest",
            "coordinate_digest", "counts", "dimensions", "groups",
            "canonical_digest",
        }
        _exact_keys(value, keys, "StageDExecutionPlan")
        groups = _require_list(value["groups"], "StageDExecutionPlan.groups")
        result = cls(
            schema_version=value["schema_version"],
            document_type=value["document_type"],
            config_digest=value["config_digest"],
            run_spec_digest=value["run_spec_digest"],
            registry_digest=value["registry_digest"],
            ordered_sample_ids_digest=value["ordered_sample_ids_digest"],
            context_digest=value["context_digest"],
            coordinate_digest=value["coordinate_digest"],
            counts=value["counts"], dimensions=value["dimensions"],
            groups=tuple(StageDImageGroup.from_dict(item) for item in groups),
            canonical_digest=value["canonical_digest"], _issuer=None,
        )
        validate_stage_d_plan(result)
        return result


def _plan_context_payload(
    config: Mapping[str, Any], run_spec: Mapping[str, Any],
    *, config_digest: str, run_spec_digest: str, registry_digest: str,
    ordered_sample_ids_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "document_type": "stage_d_plan_context",
        "config_digest": config_digest,
        "run_spec_digest": run_spec_digest,
        "registry_digest": registry_digest,
        "ordered_sample_ids_digest": ordered_sample_ids_digest,
        "model_and_preprocessing": _deep_thaw(config["model_and_preprocessing"]),
        "primary_scope": _deep_thaw(config["primary_scope"]),
        "target_and_map": _deep_thaw(config["target_and_map"]),
        "fidelity": _deep_thaw(config["fidelity"]),
        "audit": _deep_thaw(config["audit"]),
        "run_execution_coordinates": _deep_thaw(run_spec["execution_coordinates"]),
        "run_fidelity": _deep_thaw(run_spec["fidelity"]),
        "run_audit": _deep_thaw(run_spec["audit"]),
        "forbidden_workload": {
            "config": _deep_thaw(config["forbidden_workload"]),
            "run_spec": _deep_thaw(run_spec["forbidden_workload"]),
        },
        "sharding": _deep_thaw(run_spec["sharding"]),
    }


def build_stage_d_plan(
    config: Any, run_spec: Any, registry: Any,
) -> StageDExecutionPlan:
    """Build the exact metadata-only Stage D plan.

    Inputs may be mappings or typed loader objects exposing ``to_dict()``.
    A manifest object exposing ``registry_document`` is accepted directly.
    """

    config_document = _coerce_mapping(config, "config")
    run_document = _coerce_mapping(run_spec, "run_spec")
    registry_document = _registry_document(registry)
    config_digest = _validate_config(config_document)
    registry_digest, ordered_digest, rows = _validate_registry(registry_document)
    run_digest = _validate_run_spec(
        run_document, config_digest=config_digest,
        registry_digest=registry_digest,
        ordered_sample_ids_digest=ordered_digest, rows=rows,
    )
    config_audit = config_document["audit"]
    run_audit = run_document["audit"]
    for key in ("audit_overlay_ids", "audit_sample_ids", "audit_selection_orders"):
        _expect(run_audit[key], config_audit[key], f"config/run_spec audit binding {key}")
    audit_rows = [row for row in rows if row["audit_overlay"]]
    expected_audits_by_order = sorted(zip(
        config_audit["audit_selection_orders"],
        config_audit["audit_sample_ids"],
        config_audit["audit_overlay_ids"],
    ))
    _expect(
        [
            (row["selection_order"], row["sample_id"], row["audit_overlay_id"])
            for row in audit_rows
        ],
        expected_audits_by_order, "registry audit overlay/order binding",
    )
    for sample_id, selection_order, overlay_id in zip(
        config_audit["audit_sample_ids"], config_audit["audit_selection_orders"],
        config_audit["audit_overlay_ids"],
    ):
        row = rows[selection_order]
        if row["sample_id"] != sample_id or row["audit_overlay_id"] != overlay_id:
            raise StageDPlanError("registry audit sample/order/overlay binding mismatch")
    context_payload = _plan_context_payload(
        config_document, run_document, config_digest=config_digest,
        run_spec_digest=run_digest, registry_digest=registry_digest,
        ordered_sample_ids_digest=ordered_digest,
    )
    context_digest = canonical_sha256(context_payload)
    model_context_digest = canonical_sha256({
        "config_digest": config_digest,
        "model_and_preprocessing": config_document["model_and_preprocessing"],
    })
    target_context_digest = canonical_sha256({
        "config_digest": config_digest,
        "primary_scope": config_document["primary_scope"],
        "target_and_map": config_document["target_and_map"],
    })
    fidelity_context_digest = canonical_sha256({
        "config_digest": config_digest,
        "fidelity": config_document["fidelity"],
        "run_fidelity": run_document["fidelity"],
    })
    groups: list[StageDImageGroup] = []
    for row in rows:
        pair = StageDPairBinding.create(
            sample_id=row["sample_id"], selection_order=row["selection_order"],
            image_sha256=row["image_sha256"],
            model_context_digest=model_context_digest,
            target_context_digest=target_context_digest,
            fidelity_context_digest=fidelity_context_digest,
            plan_context_digest=context_digest,
        )
        units = tuple(
            StageDUnit.create(
                sample_id=row["sample_id"],
                selection_order=row["selection_order"],
                method_index=method_index, method_id=method_id,
                pair_binding_digest=pair.canonical_digest,
                audit=row["audit_overlay"],
                audit_overlay_id=row["audit_overlay_id"],
            )
            for method_index, method_id in enumerate(METHOD_IDS)
        )
        groups.append(StageDImageGroup.create(
            selection_order=row["selection_order"], sample_id=row["sample_id"],
            relative_path=row["relative_path"],
            image_sha256=row["image_sha256"],
            registry_row_digest=row["canonical_digest"],
            audit=row["audit_overlay"],
            audit_overlay_id=row["audit_overlay_id"],
            pair_binding=pair, units=units,
        ))
    partial = StageDExecutionPlan(
        1, PLAN_DOCUMENT_TYPE, config_digest, run_digest, registry_digest,
        ordered_digest, context_digest, "0" * 64, _counts(),
        _dimensions_payload(), tuple(groups), "0" * 64, _PLAN_ISSUER,
    )
    coordinate_digest = _stream_coordinate_digest(partial.coordinates)
    result = StageDExecutionPlan(
        1, PLAN_DOCUMENT_TYPE, config_digest, run_digest, registry_digest,
        ordered_digest, context_digest, coordinate_digest, _counts(),
        _dimensions_payload(), tuple(groups), "0" * 64, _PLAN_ISSUER,
    )
    object.__setattr__(result, "canonical_digest", canonical_sha256(result._payload()))
    # All descendants and the complete coordinate stream were created from
    # already validated frozen sources above.  Detached plans are subjected to
    # the deliberately more expensive source-recomputing validator.
    return result


def validate_stage_d_plan(plan: Any) -> StageDExecutionPlan:
    """Validate a typed plan, recomputing every group and coordinate digest."""

    if not isinstance(plan, StageDExecutionPlan):
        return StageDExecutionPlan.from_dict(_coerce_mapping(plan, "plan"))
    _expect(_deep_thaw(plan.counts), _counts(), "plan.counts")
    _expect(_deep_thaw(plan.dimensions), _dimensions_payload(), "plan.dimensions")
    if [group.selection_order for group in plan.groups] != list(range(500)):
        raise StageDPlanError("plan image groups are missing, duplicate, or reordered")
    sample_ids = [group.sample_id for group in plan.groups]
    if len(sample_ids) != len(set(sample_ids)):
        raise StageDPlanError("plan image groups contain duplicate sample IDs")
    if canonical_sha256(sample_ids) != plan.ordered_sample_ids_digest:
        raise StageDPlanError("plan ordered_sample_ids_digest mismatch")
    if plan.ordered_sample_ids_digest != FROZEN_ORDERED_SAMPLE_IDS_DIGEST:
        raise StageDPlanError("plan ordered sample IDs are not the frozen D1 cohort")
    if plan.config_digest != FROZEN_CONFIG_CANONICAL_DIGEST or plan.run_spec_digest != FROZEN_RUN_SPEC_CANONICAL_DIGEST or plan.registry_digest != FROZEN_REGISTRY_CANONICAL_DIGEST:
        raise StageDPlanError("plan source contract digest drift")
    if plan.context_digest != FROZEN_PLAN_CONTEXT_DIGEST:
        raise StageDPlanError("plan context_digest is not the frozen D2 context")
    group_projection = [{
        "selection_order": group.selection_order,
        "sample_id": group.sample_id,
        "relative_path": group.relative_path,
        "image_sha256": group.image_sha256,
        "registry_row_digest": group.registry_row_digest,
        "audit": group.audit,
        "audit_overlay_id": group.audit_overlay_id,
    } for group in plan.groups]
    if canonical_sha256({"groups": group_projection}) != FROZEN_GROUP_PROJECTION_DIGEST:
        raise StageDPlanError("plan group projection does not match frozen D1 rows")
    for group in plan.groups:
        pair = group.pair_binding
        if (
            pair.plan_context_digest != plan.context_digest
            or pair.model_context_digest != FROZEN_MODEL_CONTEXT_DIGEST
            or pair.target_context_digest != FROZEN_TARGET_CONTEXT_DIGEST
            or pair.fidelity_context_digest != FROZEN_FIDELITY_CONTEXT_DIGEST
        ):
            raise StageDPlanError("plan pair binding context drift")
    unit_digests = [unit.canonical_digest for unit in plan.units]
    if len(unit_digests) != 1000 or len(set(unit_digests)) != 1000:
        raise StageDPlanError("plan unit coverage/digests are invalid")
    audits = [group for group in plan.groups if group.audit]
    if len(audits) != 5 or len(plan.map_coordinates) != 1010:
        raise StageDPlanError("plan audit/map coordinate coverage is invalid")
    if tuple(
        (group.selection_order, group.sample_id, group.audit_overlay_id)
        for group in audits
    ) != FROZEN_AUDITS_BY_ORDER:
        raise StageDPlanError("plan audit overlays are not the frozen D1 set")
    if any(item.statistical_n_increment != 0 for item in plan.map_coordinates):
        raise StageDPlanError("audit work may not increment statistical n")
    computed_coordinate_digest = _stream_coordinate_digest(plan.coordinates)
    if computed_coordinate_digest != plan.coordinate_digest:
        raise StageDPlanError("plan coordinate_digest mismatch")
    if canonical_sha256(plan._payload()) != plan.canonical_digest:
        raise StageDPlanError("plan canonical_digest mismatch")
    return plan


validate_stage_d_execution_plan = validate_stage_d_plan
ExecutionPlan = StageDExecutionPlan
ImageGroup = StageDImageGroup
ExecutionUnit = StageDUnit
PairBinding = StageDPairBinding
MapCoordinate = StageDMapCoordinate
FidelityCurvePointCoordinate = StageDCurvePoint


def _coerce_validated_plan(plan: Any) -> StageDExecutionPlan:
    if isinstance(plan, StageDExecutionPlan) and plan._issuer is _PLAN_ISSUER:
        return plan
    return validate_stage_d_plan(plan)


def validate_stage_d_shard_count(shard_count: Any) -> int:
    """Validate a future D7 shard count without storing it in the plan."""

    return _require_int(shard_count, "shard_count", minimum=1)


def stage_d_shard_id(selection_order: Any, shard_count: Any) -> int:
    order = _require_int(selection_order, "selection_order", minimum=0, maximum=499)
    count = validate_stage_d_shard_count(shard_count)
    return order % count


def validate_stage_d_shard_mapping(
    plan: Any, shard_count: Any, mapping: Mapping[Any, Any] | None = None,
) -> Mapping[int, int]:
    """Validate/derive the future modulo mapping without mutating the plan."""

    typed = _coerce_validated_plan(plan)
    count = validate_stage_d_shard_count(shard_count)
    expected = {
        group.selection_order: group.selection_order % count
        for group in typed.groups
    }
    if mapping is not None:
        if not isinstance(mapping, Mapping):
            raise StageDPlanError("shard mapping must be an object")
        normalized: dict[int, int] = {}
        for raw_key, value in mapping.items():
            if type(raw_key) is int:
                key = raw_key
            elif isinstance(raw_key, str) and raw_key.isascii() and raw_key.isdigit() and str(int(raw_key)) == raw_key:
                key = int(raw_key)
            else:
                raise StageDPlanError("shard mapping keys must be canonical selection orders")
            normalized[key] = _require_int(value, f"shard mapping[{raw_key!r}]", minimum=0, maximum=count - 1)
        if normalized != expected:
            raise StageDPlanError("shard mapping is not exact selection_order modulo shard_count")
    return MappingProxyType(expected)


_PROFILE_FIELDS = (
    "seconds", "forward_calls", "backward_calls",
    "l0_l1_payload_bytes", "l2_payload_bytes",
    "control_and_provenance_bytes", "cuda_peak_reserved_bytes",
)
_PROFILE_UNIT_KINDS = PROFILE_UNIT_KINDS
_PROFILE_KEYS = tuple(
    (method_id, unit_kind)
    for method_id in METHOD_IDS for unit_kind in _PROFILE_UNIT_KINDS
)
PROFILE_SOURCE_STATUS = "SYNTHETIC_OR_PRE_D7_METADATA_PROFILE_NOT_AUTHORITY"
UNVALIDATED_PROFILE_STATUS = "UNVALIDATED_PROJECTED_FOR_D7"


@dataclass(frozen=True, slots=True)
class StageDResourceProfile:
    """One D7-supplied conservative profile for a typed method/work unit."""

    method_id: str
    unit_kind: str
    seconds: int | float
    forward_calls: int
    backward_calls: int
    l0_l1_payload_bytes: int
    l2_payload_bytes: int
    control_and_provenance_bytes: int
    cuda_peak_reserved_bytes: int
    source_status: str
    canonical_digest: str = ""

    def __post_init__(self) -> None:
        if self.method_id not in METHOD_IDS:
            raise StageDPlanError("resource profile method is outside Stage D")
        if self.unit_kind not in _PROFILE_UNIT_KINDS:
            raise StageDPlanError("resource profile unit_kind is invalid")
        _require_number(self.seconds, "profile.seconds", positive=True)
        for name in _PROFILE_FIELDS[1:]:
            _require_int(getattr(self, name), f"profile.{name}", minimum=0)
        if self.forward_calls == 0 and self.backward_calls == 0:
            raise StageDPlanError("resource profile must account for method calls")
        if self.unit_kind == "primary_execution_unit" and self.l2_payload_bytes != 0:
            raise StageDPlanError(
                "primary profile L2 must be zero; audit profile owns five-image L2"
            )
        if self.source_status != PROFILE_SOURCE_STATUS:
            raise StageDPlanError(
                "D3 accepts only non-authoritative synthetic/pre-D7 profiles"
            )
        payload = self._payload()
        expected = canonical_sha256(payload)
        if not self.canonical_digest:
            object.__setattr__(self, "canonical_digest", expected)
        elif _require_sha256(self.canonical_digest, "profile.canonical_digest") != expected:
            raise StageDPlanError("resource profile canonical_digest mismatch")

    @property
    def digest(self) -> str:
        return self.canonical_digest

    @property
    def key(self) -> tuple[str, str]:
        return self.method_id, self.unit_kind

    def _payload(self) -> dict[str, Any]:
        return _identity_payload(
            PROFILE_DOCUMENT_TYPE, method_id=self.method_id,
            unit_kind=self.unit_kind, seconds=self.seconds,
            forward_calls=self.forward_calls, backward_calls=self.backward_calls,
            l0_l1_payload_bytes=self.l0_l1_payload_bytes,
            l2_payload_bytes=self.l2_payload_bytes,
            control_and_provenance_bytes=self.control_and_provenance_bytes,
            cuda_peak_reserved_bytes=self.cuda_peak_reserved_bytes,
            source_status=self.source_status,
        )

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["canonical_digest"] = self.canonical_digest
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDResourceProfile":
        keys = {
            "schema_version", "document_type", "method_id",
            "unit_kind", *_PROFILE_FIELDS, "source_status",
            "canonical_digest",
        }
        _exact_keys(value, keys, "StageDResourceProfile")
        _expect(value["schema_version"], 1, "profile.schema_version")
        _expect(value["document_type"], PROFILE_DOCUMENT_TYPE, "profile.document_type")
        return cls(**{key: value[key] for key in keys - {"schema_version", "document_type"}})


def _coerce_profiles(value: Any) -> tuple[StageDResourceProfile, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        items: list[Any] = []
        for method_id, by_kind in value.items():
            if method_id not in METHOD_IDS or not isinstance(by_kind, Mapping):
                raise StageDPlanError("profiles must be keyed by frozen method then unit kind")
            for unit_kind, profile in by_kind.items():
                if isinstance(profile, StageDResourceProfile):
                    items.append(profile)
                    continue
                document = _coerce_mapping(profile, "resource profile")
                document.setdefault("schema_version", 1)
                document.setdefault("document_type", PROFILE_DOCUMENT_TYPE)
                document.setdefault("method_id", method_id)
                document.setdefault("unit_kind", unit_kind)
                if "canonical_digest" not in document:
                    items.append(StageDResourceProfile(
                        **{key: document[key] for key in (
                            "method_id", "unit_kind", *_PROFILE_FIELDS,
                            "source_status",
                        )}
                    ))
                else:
                    items.append(StageDResourceProfile.from_dict(document))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
    else:
        raise TypeError("profiles must be a mapping, sequence, or None")
    result: list[StageDResourceProfile] = []
    for item in items:
        if isinstance(item, StageDResourceProfile):
            result.append(item)
        else:
            result.append(StageDResourceProfile.from_dict(
                _coerce_mapping(item, "resource profile")
            ))
    keys = [item.key for item in result]
    if len(keys) != len(set(keys)):
        raise StageDPlanError("resource profiles contain a duplicate method/unit kind")
    if result and set(keys) != set(_PROFILE_KEYS):
        raise StageDPlanError(
            "profiles must provide all four method/unit-kind combinations"
        )
    return tuple(sorted(result, key=lambda item: _PROFILE_KEYS.index(item.key)))


@dataclass(frozen=True, slots=True)
class StageDEstimate:
    schema_version: int
    document_type: str
    plan_digest: str
    profile_status: str
    budget_status: str
    counts: Mapping[str, int]
    profiles: tuple[StageDResourceProfile, ...]
    projections: Mapping[str, Any]
    launch_safety_caps: Mapping[str, Any]
    completed_campaign_gates: Mapping[str, Any]
    launch_violations: tuple[str, ...]
    completed_gate_violations: tuple[str, ...]
    canonical_digest: str
    _issuer: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_int(self.schema_version, "estimate.schema_version", minimum=1, maximum=1)
        if self.document_type != ESTIMATE_DOCUMENT_TYPE:
            raise StageDPlanError("Stage D estimate identity is invalid")
        _require_sha256(self.plan_digest, "estimate.plan_digest")
        if self.profile_status not in (
            "D7_PROFILE_REQUIRED", UNVALIDATED_PROFILE_STATUS,
        ):
            raise StageDPlanError("estimate profile_status is invalid")
        if self.budget_status not in ("NOT_EVALUATED", "PASS", "BLOCKED_NO_LAUNCH"):
            raise StageDPlanError("estimate budget_status is invalid")
        object.__setattr__(self, "counts", _deep_freeze(self.counts))
        object.__setattr__(self, "projections", _deep_freeze(self.projections))
        object.__setattr__(self, "launch_safety_caps", _deep_freeze(self.launch_safety_caps))
        object.__setattr__(self, "completed_campaign_gates", _deep_freeze(self.completed_campaign_gates))
        if not isinstance(self.profiles, tuple) or any(not isinstance(item, StageDResourceProfile) for item in self.profiles):
            raise StageDPlanError("estimate profiles must be an immutable typed tuple")
        if not isinstance(self.launch_violations, tuple) or not isinstance(self.completed_gate_violations, tuple):
            raise StageDPlanError("estimate violations must be immutable tuples")
        _require_sha256(self.canonical_digest, "estimate.canonical_digest")

    @property
    def digest(self) -> str:
        return self.canonical_digest

    @property
    def within_budget(self) -> bool:
        return self.budget_status == "PASS"

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "plan_digest": self.plan_digest,
            "profile_status": self.profile_status,
            "budget_status": self.budget_status,
            "counts": _deep_thaw(self.counts),
            "profiles": [item.to_dict() for item in self.profiles],
            "projections": _deep_thaw(self.projections),
            "launch_safety_caps": _deep_thaw(self.launch_safety_caps),
            "completed_campaign_gates": _deep_thaw(self.completed_campaign_gates),
            "launch_violations": list(self.launch_violations),
            "completed_gate_violations": list(self.completed_gate_violations),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["canonical_digest"] = self.canonical_digest
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDEstimate":
        keys = {
            "schema_version", "document_type", "plan_digest",
            "profile_status", "budget_status", "counts", "profiles",
            "projections", "launch_safety_caps",
            "completed_campaign_gates", "launch_violations",
            "completed_gate_violations", "canonical_digest",
        }
        _exact_keys(value, keys, "StageDEstimate")
        profiles = _require_list(value["profiles"], "StageDEstimate.profiles")
        launch = _require_list(value["launch_violations"], "StageDEstimate.launch_violations")
        campaign = _require_list(value["completed_gate_violations"], "StageDEstimate.completed_gate_violations")
        if any(not isinstance(item, str) for item in (*launch, *campaign)):
            raise StageDPlanError("estimate violations must be strings")
        result = cls(
            schema_version=value["schema_version"],
            document_type=value["document_type"],
            plan_digest=value["plan_digest"],
            profile_status=value["profile_status"],
            budget_status=value["budget_status"], counts=value["counts"],
            profiles=tuple(StageDResourceProfile.from_dict(item) for item in profiles),
            projections=value["projections"],
            launch_safety_caps=value["launch_safety_caps"],
            completed_campaign_gates=value["completed_campaign_gates"],
            launch_violations=tuple(launch),
            completed_gate_violations=tuple(campaign),
            canonical_digest=value["canonical_digest"], _issuer=None,
        )
        validate_stage_d_estimate(result)
        return result


def _unit_counts(plan: StageDExecutionPlan) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for unit in plan.units:
        counts[(unit.method_id, "primary_execution_unit")] += 1
        if unit.audit:
            counts[(unit.method_id, "audit_repeat_map")] += 1
    return counts


def _estimate_projection(
    plan: StageDExecutionPlan, profiles: tuple[StageDResourceProfile, ...],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...], str, str]:
    if not profiles:
        return ({
            "accounting_unit": "typed_method_unit_and_audit_profile",
            "profile_authority": "UNVALIDATED_PROJECTED_FOR_D7_NOT_FINAL",
            "attempt_scope": "ONE_PLANNED_ATTEMPT_ONLY_D7_MUST_ADD_CONSERVATIVE_RETRY_ALLOWANCE",
            "logical_work_items_used_as_call_count": False,
            "forward_calls_total": None, "backward_calls_total": None,
            "projected_h20_gpu_hours": None,
            "projected_retained_regular_file_bytes": None,
            "projected_h20_gpu_hours_per_image_group_max": None,
            "maximum_l0_l1_bytes_per_execution_unit": None,
            "maximum_l2_bytes_per_audit_image": None,
            "maximum_forward_calls_per_execution_unit": None,
            "maximum_backward_calls_per_execution_unit": None,
            "maximum_cuda_peak_reserved_bytes_per_execution_unit": None,
            "by_profile": {},
        }, (), (), "D7_PROFILE_REQUIRED", "NOT_EVALUATED")
    by_key = {item.key: item for item in profiles}
    multiplicities = _unit_counts(plan)
    by_profile: dict[str, Any] = {}
    total_seconds = total_forward = total_backward = 0
    total_l0_l1 = total_l2 = total_control = 0
    for key in _PROFILE_KEYS:
        profile = by_key[key]
        count = multiplicities[key]
        label = f"{key[0]}::{key[1]}"
        values = {
            "count": count,
            "seconds": float(profile.seconds) * count,
            "forward_calls": profile.forward_calls * count,
            "backward_calls": profile.backward_calls * count,
            "l0_l1_payload_bytes": profile.l0_l1_payload_bytes * count,
            "l2_payload_bytes": profile.l2_payload_bytes * count,
            "control_and_provenance_bytes": profile.control_and_provenance_bytes * count,
        }
        by_profile[label] = values
        total_seconds += values["seconds"]
        total_forward += values["forward_calls"]
        total_backward += values["backward_calls"]
        total_l0_l1 += values["l0_l1_payload_bytes"]
        total_l2 += values["l2_payload_bytes"]
        total_control += values["control_and_provenance_bytes"]
    # Profile taxonomy is explicit: the 500-count primary execution-unit
    # profile owns primary L0/L1 and calls.  The five-count audit-repeat-map
    # profile owns all L2 bytes retained for an audit image (primary plus
    # repeat raw evidence) and its incremental map/call/L0/L1 work.
    primary_profiles = [by_key[(method, "primary_execution_unit")] for method in METHOD_IDS]
    audit_profiles = [by_key[(method, "audit_repeat_map")] for method in METHOD_IDS]
    max_group_seconds = sum(float(item.seconds) for item in primary_profiles) + sum(float(item.seconds) for item in audit_profiles)
    projected_gpu_hours = 1.25 * total_seconds / 3600.0
    projected_storage = math.ceil(1.25 * (total_l0_l1 + total_l2 + total_control))
    per_image_hours = 1.25 * max_group_seconds / 3600.0
    # The primary unit profile owns its primary outputs and the audit-repeat
    # profile owns only the incremental second-clean-run output.  Per-unit
    # safety therefore includes both for audit units.
    maximum_l0_l1 = max(
        by_key[(method, "primary_execution_unit")].l0_l1_payload_bytes
        + by_key[(method, "audit_repeat_map")].l0_l1_payload_bytes
        for method in METHOD_IDS
    )
    maximum_l2 = sum(item.l2_payload_bytes for item in audit_profiles)
    maximum_forward = max(
        by_key[(method, "primary_execution_unit")].forward_calls
        + by_key[(method, "audit_repeat_map")].forward_calls
        for method in METHOD_IDS
    )
    maximum_backward = max(
        by_key[(method, "primary_execution_unit")].backward_calls
        + by_key[(method, "audit_repeat_map")].backward_calls
        for method in METHOD_IDS
    )
    maximum_cuda = max(item.cuda_peak_reserved_bytes for item in (*primary_profiles, *audit_profiles))
    projection = {
        "accounting_unit": "typed_method_unit_and_audit_profile",
        "profile_authority": "UNVALIDATED_PROJECTED_FOR_D7_NOT_FINAL",
        "attempt_scope": "ONE_PLANNED_ATTEMPT_ONLY_D7_MUST_ADD_CONSERVATIVE_RETRY_ALLOWANCE",
        "logical_work_items_used_as_call_count": False,
        "forward_calls_total": total_forward,
        "backward_calls_total": total_backward,
        "projected_h20_gpu_hours": projected_gpu_hours,
        "projected_retained_regular_file_bytes": projected_storage,
        "projected_h20_gpu_hours_per_image_group_max": per_image_hours,
        "maximum_l0_l1_bytes_per_execution_unit": maximum_l0_l1,
        "maximum_l2_bytes_per_audit_image": maximum_l2,
        "maximum_forward_calls_per_execution_unit": maximum_forward,
        "maximum_backward_calls_per_execution_unit": maximum_backward,
        "maximum_cuda_peak_reserved_bytes_per_execution_unit": maximum_cuda,
        "by_profile": by_profile,
    }
    launch_comparisons = (
        (maximum_l0_l1, "l0_l1_bytes_per_execution_unit_max"),
        (maximum_l2, "l2_bytes_per_audit_image_max"),
        (maximum_forward, "forward_calls_per_execution_unit_max"),
        (maximum_backward, "backward_calls_per_execution_unit_max"),
        (maximum_cuda, "cuda_peak_reserved_bytes_per_execution_unit_max"),
    )
    campaign_comparisons = (
        (projected_storage, "stage_d_total_retained_regular_file_bytes_max"),
        (projected_gpu_hours, "stage_d_total_h20_gpu_hours_max"),
        (per_image_hours, "stage_d_h20_gpu_hours_per_image_group_max"),
    )
    launch_violations = tuple(
        name for observed, name in launch_comparisons
        if observed > LAUNCH_SAFETY_CAPS[name]
    )
    campaign_violations = tuple(
        name for observed, name in campaign_comparisons
        if observed > COMPLETED_CAMPAIGN_GATES[name]
    )
    budget_status = (
        "PASS" if not launch_violations and not campaign_violations
        else "BLOCKED_NO_LAUNCH"
    )
    # These remain non-authoritative D3 projections.  D7 must replace them with
    # validated conservative smoke profiles before any launch authorization.
    return (
        projection, launch_violations, campaign_violations,
        UNVALIDATED_PROFILE_STATUS,
        "NOT_EVALUATED" if budget_status == "PASS" else budget_status,
    )


def estimate_stage_d_run(
    config: Any, plan: Any, profiles: Any = None,
) -> StageDEstimate:
    """Return a metadata-only estimate; no profile means D7-not-evaluated."""

    config_document = _coerce_mapping(config, "config")
    config_digest = _validate_config(config_document)
    typed_plan = _coerce_validated_plan(plan)
    if typed_plan.config_digest != config_digest:
        raise StageDPlanError("estimate config does not match the plan")
    typed_profiles = _coerce_profiles(profiles)
    projection, launch, campaign, profile_status, budget_status = (
        _estimate_projection(typed_plan, typed_profiles)
    )
    partial = StageDEstimate(
        1, ESTIMATE_DOCUMENT_TYPE, typed_plan.canonical_digest, profile_status,
        budget_status, _counts(), typed_profiles, projection,
        LAUNCH_SAFETY_CAPS, COMPLETED_CAMPAIGN_GATES, launch, campaign,
        "0" * 64, _PLAN_ISSUER,
    )
    object.__setattr__(partial, "canonical_digest", canonical_sha256(partial._payload()))
    validate_stage_d_estimate(partial)
    return partial


estimate_stage_d_plan = estimate_stage_d_run
estimate_stage_d = estimate_stage_d_run


def validate_stage_d_estimate(estimate: Any) -> StageDEstimate:
    if not isinstance(estimate, StageDEstimate):
        return StageDEstimate.from_dict(_coerce_mapping(estimate, "estimate"))
    _expect(_deep_thaw(estimate.counts), _counts(), "estimate.counts")
    _expect(_deep_thaw(estimate.launch_safety_caps), LAUNCH_SAFETY_CAPS, "estimate.launch_safety_caps")
    _expect(_deep_thaw(estimate.completed_campaign_gates), COMPLETED_CAMPAIGN_GATES, "estimate.completed_campaign_gates")
    profiles = _coerce_profiles(estimate.profiles)
    expected = _estimate_projection(
        # Projection needs only unit multiplicities; reconstruct from counts.
        # Use the exact 500/5 distribution encoded in the typed estimate below.
        _SyntheticPlanForEstimate(), profiles,  # type: ignore[arg-type]
    )
    projection, launch, campaign, profile_status, budget_status = expected
    _expect(_deep_thaw(estimate.projections), projection, "estimate.projections")
    if estimate.launch_violations != launch or estimate.completed_gate_violations != campaign:
        raise StageDPlanError("estimate violation lists disagree with profiles")
    if estimate.profile_status != profile_status or estimate.budget_status != budget_status:
        raise StageDPlanError("estimate status disagrees with profiles and gates")
    if canonical_sha256(estimate._payload()) != estimate.canonical_digest:
        raise StageDPlanError("estimate canonical_digest mismatch")
    return estimate


class _SyntheticPlanForEstimate:
    """Minimal exact unit stream used to validate detached estimates."""

    @property
    def units(self) -> Iterator[Any]:
        for method_id in METHOD_IDS:
            for index in range(500):
                yield _ProfileUnit(method_id, index < 5)


@dataclass(frozen=True, slots=True)
class _ProfileUnit:
    method_id: str
    audit: bool


@dataclass(frozen=True, slots=True)
class StageDPreflight:
    schema_version: int
    document_type: str
    plan_digest: str
    estimate_digest: str
    status: str
    execution_authorization: str
    analysis_authorization: str
    d7_final: bool
    violations: tuple[str, ...]
    canonical_digest: str
    _issuer: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_int(self.schema_version, "preflight.schema_version", minimum=1, maximum=1)
        if self.document_type != PREFLIGHT_DOCUMENT_TYPE:
            raise StageDPlanError("Stage D preflight identity is invalid")
        _require_sha256(self.plan_digest, "preflight.plan_digest")
        _require_sha256(self.estimate_digest, "preflight.estimate_digest")
        if self.status not in ("PROFILE_REQUIRED", "READY_FOR_D7", "BLOCKED_NO_LAUNCH"):
            raise StageDPlanError("preflight status is invalid")
        if self.execution_authorization != "NOT_ISSUED" or self.analysis_authorization != "NOT_EVALUATED" or self.d7_final is not False:
            raise StageDPlanError("D3 preflight cannot issue execution/analysis authority or be D7-final")
        if not isinstance(self.violations, tuple) or any(not isinstance(item, str) for item in self.violations):
            raise StageDPlanError("preflight violations must be an immutable string tuple")
        _require_sha256(self.canonical_digest, "preflight.canonical_digest")

    @property
    def digest(self) -> str:
        return self.canonical_digest

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type, "plan_digest": self.plan_digest,
            "estimate_digest": self.estimate_digest, "status": self.status,
            "execution_authorization": self.execution_authorization,
            "analysis_authorization": self.analysis_authorization,
            "d7_final": self.d7_final, "violations": list(self.violations),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["canonical_digest"] = self.canonical_digest
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDPreflight":
        keys = {
            "schema_version", "document_type", "plan_digest",
            "estimate_digest", "status", "execution_authorization",
            "analysis_authorization", "d7_final", "violations",
            "canonical_digest",
        }
        _exact_keys(value, keys, "StageDPreflight")
        violations = _require_list(value["violations"], "StageDPreflight.violations")
        result = cls(
            **{key: value[key] for key in keys - {"violations"}},
            violations=tuple(violations), _issuer=None,
        )
        validate_stage_d_preflight(result)
        return result


def preflight_stage_d(
    config_or_estimate: Any, plan: Any = None, profiles: Any = None,
) -> StageDPreflight:
    """Classify an estimate, or build one from ``config, plan, profiles``."""

    if plan is None:
        typed = validate_stage_d_estimate(config_or_estimate)
    else:
        typed = estimate_stage_d_run(config_or_estimate, plan, profiles)
    status = {
        "NOT_EVALUATED": "PROFILE_REQUIRED",
        "PASS": "READY_FOR_D7",
        "BLOCKED_NO_LAUNCH": "BLOCKED_NO_LAUNCH",
    }[typed.budget_status]
    violations = typed.launch_violations + typed.completed_gate_violations
    result = StageDPreflight(
        1, PREFLIGHT_DOCUMENT_TYPE, typed.plan_digest, typed.canonical_digest,
        status, "NOT_ISSUED", "NOT_EVALUATED", False, violations,
        "0" * 64, _PLAN_ISSUER,
    )
    object.__setattr__(result, "canonical_digest", canonical_sha256(result._payload()))
    validate_stage_d_preflight(result)
    return result


def validate_stage_d_preflight(preflight: Any) -> StageDPreflight:
    if not isinstance(preflight, StageDPreflight):
        return StageDPreflight.from_dict(_coerce_mapping(preflight, "preflight"))
    if preflight.status == "PROFILE_REQUIRED" and preflight.violations:
        raise StageDPlanError("profile-required preflight cannot invent violations")
    if preflight.status in ("PROFILE_REQUIRED", "READY_FOR_D7") and preflight.violations:
        raise StageDPlanError("non-blocked preflight cannot contain violations")
    if preflight.status == "BLOCKED_NO_LAUNCH" and not preflight.violations:
        raise StageDPlanError("blocked preflight requires a violation")
    if canonical_sha256(preflight._payload()) != preflight.canonical_digest:
        raise StageDPlanError("preflight canonical_digest mismatch")
    return preflight


def enforce_stage_d_preflight(preflight: StageDPreflight) -> None:
    """Reject D3 preflight as execution authority, even when projected PASS."""

    typed = validate_stage_d_preflight(preflight)
    if typed._issuer is not _PLAN_ISSUER:
        raise StageDPlanError("only a freshly computed preflight can be checked")
    if typed.status == "PROFILE_REQUIRED":
        raise StageDBudgetError("D7 measured resource profiles are required")
    if typed.status == "BLOCKED_NO_LAUNCH":
        raise StageDBudgetError(
            "Stage D projection blocks launch: " + ", ".join(typed.violations)
        )
    raise StageDPlanError(
        "D3 preflight is ready for D7 review but cannot authorize execution"
    )


__all__ = [
    "AUDIT_IMAGE_COUNT", "AUDIT_REPEAT_CLEAN_MAP_COUNT",
    "BLOCK_INDEX", "COMPLETED_CAMPAIGN_GATES", "CURVE_FRACTIONS",
    "DIRECTIONS", "ENDPOINT_ID", "EXECUTION_UNIT_COUNT",
    "FIDELITY_CURVE_COUNT", "FIDELITY_CURVE_POINT_COUNT",
    "FIDELITY_SCALE", "FINE_CELL_COUNT", "HEAD_INDEX",
    "IMAGE_GROUP_COUNT", "LAUNCH_SAFETY_CAPS",
    "LOGICAL_WORK_ITEM_COUNT", "MAP_WORK_ITEM_COUNT",
    "MATCHED_METHOD_ID", "METHOD_IDS", "NNS_METHOD_ID",
    "PRIMARY_CLEAN_MAP_COUNT", "PROFILE_SOURCE_STATUS",
    "PROFILE_UNIT_KINDS",
    "SHARD_ASSIGNMENT_FORMULA",
    "SHARD_COUNT_STATUS", "STATISTICAL_N", "StageDBudgetError",
    "StageDCoordinateSequence", "StageDCurvePoint", "StageDEstimate",
    "StageDExecutionPlan", "StageDImageGroup", "StageDMapCoordinate",
    "StageDPairBinding", "StageDPlanError", "StageDPreflight",
    "StageDResourceProfile", "StageDUnit", "TARGET_ID",
    "TIE_REPEAT_INDICES", "UNVALIDATED_PROFILE_STATUS",
    "build_stage_d_plan", "canonical_json_bytes",
    "canonical_sha256", "enforce_stage_d_preflight",
    "estimate_stage_d", "estimate_stage_d_plan", "estimate_stage_d_run",
    "preflight_stage_d", "stage_d_shard_id",
    "validate_stage_d_estimate", "validate_stage_d_execution_plan",
    "validate_stage_d_plan", "validate_stage_d_preflight",
    "validate_stage_d_shard_count", "validate_stage_d_shard_mapping",
    "ExecutionPlan", "ExecutionUnit", "FidelityCurvePointCoordinate",
    "ImageGroup", "MapCoordinate", "PairBinding",
]
