from __future__ import annotations

import copy
import json
from types import MappingProxyType

import pytest

from vitnns.constants import MODEL_NAME, MODEL_SHA256
from vitnns.stage_c_fidelity import fidelity_seed
from vitnns.stage_c_provenance import (
    CANONICAL_RGB_HASH_DOMAIN,
    NORMALIZED_INPUT_HASH_DOMAIN,
    StageCProvenanceError,
    build_stage_c_provenance,
    build_unit_identity,
    canonical_sha256,
    validate_stage_c_provenance,
)


SAMPLE = "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG"
METHOD = "conditional_fixed_clean_attention_head_direction_value_path_ig"
TARGET = "fixed_clean_attention_cls_head_direction"
SHA = "a" * 64
NORMALIZED_SHA = "b" * 64
CANONICAL_SHA = "c" * 64


def _candidate(block: int) -> dict[str, object]:
    return {
        "candidate_key": f"block_{block:02d}_head_01",
        "block_index": block,
        "head_index": 1,
        "depth_band": block // 3,
    }


def valid_document(*, selection_present: bool = False):
    unit = build_unit_identity(
        sample_id=SAMPLE,
        method_id=METHOD,
        config_id="block_02_head_01",
        block_index=2,
        head_index=1,
        target_ref=TARGET,
    )
    target_identity = {
        "comparison_line": "value_path",
        "target_ref": TARGET,
        "clean_input_sha256": NORMALIZED_SHA,
        "frozen_target": {
            "target_kind": TARGET,
            "clean_input_sha256": NORMALIZED_SHA,
            "candidate": {"block_index": 2, "head_index": 1},
        },
    }
    selection = {
        "schema_version": 1,
        "status": "not_applicable_pre_c8",
        "path": "selection/stage_c_selection.json",
        "reason": "C8 selection has not yet been produced",
        "sha256": None,
        "primary": None,
        "depth_band_winners": [],
    }
    if selection_present:
        winners = [_candidate(block) for block in (2, 5, 8, 11)]
        selection.update(
            status="present", reason=None, sha256="d" * 64,
            primary=winners[0], depth_band_winners=winners,
        )
    perturbations = []
    parameters_by_id = {
        "translate_right_down_4": {
            "kind": "integer_translation", "dx_pixels": 4, "dy_pixels": 4,
            "padding": "reflection", "interpolation": "none",
            "inverse_map": (
                "copy_perturbed_rows_4_223_cols_4_223_to_clean_rows_0_219_cols_0_219"
            ),
            "valid_pixel_box_half_open": [0, 220, 0, 220],
            "fully_valid_patch_rows": list(range(13)),
            "fully_valid_patch_cols": list(range(13)),
        },
        "center_crop_216_resize_224": {
            "kind": "center_crop_scale", "crop_box_half_open": [4, 220, 4, 220],
            "crop_height": 216, "crop_width": 216, "resize_height": 224,
            "resize_width": 224, "image_interpolation": "bicubic",
            "image_antialias": True, "inverse_map_interpolation": "bilinear",
            "inverse_map_antialias": True,
            "inverse_mass_jacobian_multiplier": 1.075445816186557,
            "valid_pixel_box_half_open": [4, 220, 4, 220],
            "fully_valid_patch_rows": list(range(1, 13)),
            "fully_valid_patch_cols": list(range(1, 13)),
        },
        "gaussian_rgb_std_0_01": {
            "kind": "gaussian_noise", "mean": 0.0, "std_rgb_0_to_1": 0.01,
            "generator_device": "cpu", "generator_dtype": "float32",
            "base_seed": 20260821,
            "sample_seed_rule": "base_seed_plus_registry_selection_order",
            "seed": 20260821, "clamp_min": 0.0, "clamp_max": 1.0,
            "inverse_map": "identity", "valid_pixel_box_half_open": [0, 224, 0, 224],
            "fully_valid_patch_rows": list(range(14)),
            "fully_valid_patch_cols": list(range(14)),
        },
    }
    for index, perturbation_id in enumerate((
        "translate_right_down_4",
        "center_crop_216_resize_224",
        "gaussian_rgb_std_0_01",
    )):
        parameters = parameters_by_id[perturbation_id]
        perturbations.append({
            "id": perturbation_id,
            "canonical_rgb": {
                "domain": CANONICAL_RGB_HASH_DOMAIN,
                "sha256": str(index + 1) * 64,
            },
            "normalized_input": {
                "domain": NORMALIZED_INPUT_HASH_DOMAIN,
                "sha256": str(index + 4) * 64,
            },
            "parameters": parameters,
            "shared_exact_tensor_across_methods": True,
        })
    preflight = {
        "schema_version": 1,
        "document_type": "stage_c_preflight_estimate",
        "plan_digest": SHA,
        "counts": {}, "storage": {}, "compute": {}, "caps": {},
        "violations": [], "within_budget": True,
    }
    preflight["canonical_digest"] = canonical_sha256(preflight)
    record = build_stage_c_provenance(
        unit_identity=unit,
        code={"commit": "d" * 40, "dirty": True},
        model={
            "name": MODEL_NAME,
            "state_dict_path": "models/vit.pth",
            "state_dict_sha256": MODEL_SHA256,
        },
        input_record={
            "schema_version": 1,
            "sample_id": SAMPLE,
            "image_id": "ILSVRC2012_val_00000079",
            "relative_path": "ILSVRC2012_val_00000079.JPEG",
            "image_sha256": SHA,
            "size_bytes": 661635,
            "width": 900,
            "height": 1000,
            "official_wnid": "n02091635",
            "imagenet1k_index": 175,
            "label_source": "official_devkit_val_gt",
            "role": "calibration",
            "audit": True,
            "selection_order": 0,
            "source_selection_index": 0,
            "selection_rule": (
                "legacy_sorted_500_linspace20_then_fixed_role_positions_0_4_9_"
                "14_19_audit_0_19"
            ),
            "canonical_rgb_sha256": CANONICAL_SHA,
            "normalized_tensor_sha256": NORMALIZED_SHA,
        },
        dependencies={"lock_path": "requirements.lock.txt", "lock_sha256": SHA},
        registry={
            "path": "manifests/stage_c_sample_registry.csv", "sha256": SHA,
            "metadata_path": "manifests/stage_c_sample_registry_metadata.json",
            "metadata_sha256": SHA,
            "source_manifest_path": "manifests/source/manifest.csv",
            "source_manifest_sha256": SHA,
            "source_manifest_metadata_path": "manifests/source/metadata.json",
            "source_manifest_metadata_sha256": SHA,
        },
        protocol={"path": "docs/protocol.md", "sha256": SHA, "version": "1.0.0"},
        acceptance={"path": "docs/acceptance.md", "sha256": SHA},
        configuration={
            "path": "config/stage_c.json", "sha256": SHA,
            "schema_path": "config/stage_c.schema.json", "schema_sha256": SHA,
            "registry_schema_path": "config/stage_c_registry.schema.json",
            "registry_schema_sha256": SHA,
            "selection_schema_path": "config/stage_c_selection.schema.json",
            "selection_schema_sha256": SHA,
            "run_spec_sha256": SHA, "execution_plan_sha256": SHA,
        },
        selection=selection,
        method={
            "id": METHOD,
            "exact_name": "NNS fixed-clean-attention head-direction Value-path IG",
            "version": "1.0.0", "comparison_line": "value_path",
            "explanation_target_ref": TARGET, "fidelity_target_ref": TARGET,
            "class_agnostic": True, "fidelity_eligible": True,
            "structural_reference_only": False,
            "native_map": "signed_float32_224x224_rgb_channel_sum",
        },
        target={
            "status": "resolved", "reason": None,
            "comparison_line": "value_path", "target_ref": TARGET,
            "clean_input_normalized_sha256": NORMALIZED_SHA,
            "identity_sha256": canonical_sha256(target_identity),
            "identity": target_identity,
        },
        map_record={
            "projection": "pixel_mass_to_patch_sum", "sign_semantics": "signed",
            "normalization": "none_signed_mass_preserving",
            "metric_ready_dtype": "float32", "metric_ready_shape": [14, 14],
        },
        baseline={
            "id": "zero_normalized", "normalized_value": 0.0,
            "canonical_rgb_value": 0.5, "channels": 3,
            "normalized_tensor_sha256": "e" * 64,
        },
        ig_schedule={
            "quadrature": "gauss_legendre",
            "steps_schedule": [512, 1024, 2048, 4096, 8192],
            "step_chunk_size": 8, "atol": 0.0001, "rtol": 0.01,
        },
        seeds={
            "base_seed": 20260821, "noise_seed": 20260821,
            "noise_seed_rule": "base_seed_plus_registry_selection_order",
            "fidelity_enabled": True,
            "fidelity_seed_namespace": (
                "stage-c-fidelity|20260821|sample_id|block|head|replicate"
            ),
            "fidelity_pairing_context": {"block_index": 2, "head_index": 1},
            "fidelity_seeds": [fidelity_seed(SAMPLE, 2, 1, i) for i in range(5)],
        },
        perturbations={
            "enabled": True,
            "application_domain": CANONICAL_RGB_HASH_DOMAIN,
            "normalized_domain": NORMALIZED_INPUT_HASH_DOMAIN,
            "clean": {
                "id": "clean",
                "canonical_rgb": {
                    "domain": CANONICAL_RGB_HASH_DOMAIN,
                    "sha256": CANONICAL_SHA,
                },
                "normalized_input": {
                    "domain": NORMALIZED_INPUT_HASH_DOMAIN,
                    "sha256": NORMALIZED_SHA,
                },
            },
            "items": perturbations,
        },
        status={"state": "completed", "terminal": True},
        failure=None,
        resources={
            "preflight": preflight,
            "actual": {
                "status": "completed", "wall_time_seconds": 1.25,
                "cuda": {
                    "status": "not_applicable_cpu", "device": None,
                    "peak_allocated_bytes": None,
                    "peak_reserved_bytes": None,
                },
                "calls": {
                    "accounting_unit": "batched_model_invocation",
                    "forward_calls": 3, "backward_calls": 2,
                },
                "artifact_bytes": 7,
                "artifact_bytes_scope": "explicit_payload_file_list",
                "artifact_root": None,
                "files": [{"path": "L1/map.npy", "bytes": 7}],
                "exception": None, "instrumentation_errors": [],
            },
        },
        retention={
            "audit": True, "display_is_never_metric_input": True,
            "l1_storage_dtype": "float32", "l2_audit_only": True,
            "l3_display_only": True,
            "forbidden_granularities": [
                "per_quadrature_step_files", "per_pixel_files",
                "per_dimension_files", "display_png_as_metric_input",
            ],
        },
        artifact_inventory=[{
            "level": "L1", "path": "L1/map.npy", "sha256": SHA,
            "bytes": 7, "kind": "metric_ready", "dtype": "float32",
            "shape": [14, 14], "format": "npy",
        }],
    )
    return record


def mutate(document, path, value):
    result = copy.deepcopy(document)
    cursor = result
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return result


def test_complete_builder_is_json_safe_immutable_and_round_trips():
    provenance = valid_document()
    payload = provenance.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["unit_identity_sha256"] == payload["unit_identity"]["canonical_digest"]
    assert payload["bindings"]["code"]["dirty"] is True
    assert "provenance_sha256" not in json.dumps(payload)
    with pytest.raises(TypeError):
        provenance.bindings["code"]["dirty"] = False
    payload["bindings"]["code"]["dirty"] = False
    assert provenance.bindings["code"]["dirty"] is True


@pytest.mark.parametrize(
    "path",
    [
        ("bindings", "model"),
        ("bindings", "registry"),
        ("bindings", "configuration"),
        ("bindings", "target"),
        ("bindings", "resources"),
        ("bindings", "retention"),
    ],
)
def test_missing_required_section_or_field_fails_closed(path):
    value = valid_document().to_dict()
    if len(path) == 2:
        del value[path[0]][path[1]]
    with pytest.raises(StageCProvenanceError, match="missing"):
        validate_stage_c_provenance(value)


def test_tampered_unit_and_target_hashes_are_rejected():
    value = valid_document().to_dict()
    value["unit_identity"]["head_index"] = 3
    with pytest.raises(StageCProvenanceError, match="config_id|canonical_digest"):
        validate_stage_c_provenance(value)

    value = valid_document().to_dict()
    value["bindings"]["target"]["identity"]["frozen_target"]["candidate"]["head_index"] = 3
    with pytest.raises(StageCProvenanceError, match="identity_sha256"):
        validate_stage_c_provenance(value)


def test_hash_domain_swap_and_cross_line_target_are_rejected():
    value = valid_document().to_dict()
    clean = value["bindings"]["perturbations"]["clean"]
    clean["canonical_rgb"]["domain"], clean["normalized_input"]["domain"] = (
        clean["normalized_input"]["domain"], clean["canonical_rgb"]["domain"]
    )
    with pytest.raises(StageCProvenanceError, match="domain swap"):
        validate_stage_c_provenance(value)

    value = valid_document().to_dict()
    identity = value["bindings"]["target"]["identity"]
    identity["comparison_line"] = "class_logit"
    value["bindings"]["target"]["identity_sha256"] = canonical_sha256(identity)
    with pytest.raises(StageCProvenanceError, match="crosses comparison line"):
        validate_stage_c_provenance(value)


def test_selection_states_never_use_a_placeholder_hash():
    validate_stage_c_provenance(valid_document(selection_present=False))
    validate_stage_c_provenance(valid_document(selection_present=True))

    value = valid_document().to_dict()
    value["bindings"]["selection"]["sha256"] = "0" * 64
    with pytest.raises(StageCProvenanceError, match="cannot contain a hash"):
        validate_stage_c_provenance(value)


def test_disabled_perturbations_are_explicit_and_have_no_fabricated_hashes():
    value = valid_document().to_dict()
    value["bindings"]["perturbations"].update(enabled=False, items=[])
    validate_stage_c_provenance(value)


def test_fidelity_pairing_and_derived_seeds_cannot_drift():
    value = valid_document().to_dict()
    value["bindings"]["seeds"]["fidelity_pairing_context"]["head_index"] = 2
    with pytest.raises(StageCProvenanceError, match="pairing"):
        validate_stage_c_provenance(value)

    value = valid_document().to_dict()
    value["bindings"]["seeds"]["fidelity_seeds"][0] += 1
    with pytest.raises(StageCProvenanceError, match="registered namespace"):
        validate_stage_c_provenance(value)


def test_execution_plan_and_run_spec_are_both_bound():
    value = valid_document().to_dict()
    del value["bindings"]["configuration"]["run_spec_sha256"]
    with pytest.raises(StageCProvenanceError, match="run_spec_sha256"):
        validate_stage_c_provenance(value)

    value = valid_document().to_dict()
    value["bindings"]["resources"]["preflight"]["plan_digest"] = "f" * 64
    with pytest.raises(StageCProvenanceError, match="plan_digest"):
        validate_stage_c_provenance(value)

    value = valid_document(selection_present=True).to_dict()
    value["bindings"]["selection"]["sha256"] = None
    with pytest.raises(StageCProvenanceError, match="SHA-256"):
        validate_stage_c_provenance(value)


def test_dirty_state_is_preserved_and_cannot_be_reported_clean_to_observer():
    value = valid_document().to_dict()
    validate_stage_c_provenance(
        value, observed_git_state={"commit": "d" * 40, "dirty": True}
    )
    value["bindings"]["code"]["dirty"] = False
    with pytest.raises(StageCProvenanceError, match="observed Git state"):
        validate_stage_c_provenance(
            value, observed_git_state={"commit": "d" * 40, "dirty": True}
        )


def test_status_failure_resources_inventory_and_no_self_hash_are_consistent():
    value = valid_document().to_dict()
    value["bindings"]["failure"] = {
        "reason": "toy_failure", "phase": "compute",
        "type": "builtins.RuntimeError", "message": "boom",
    }
    with pytest.raises(StageCProvenanceError, match="failure=null"):
        validate_stage_c_provenance(value)


def test_pre_target_failure_uses_structured_unavailable_not_fake_hash():
    value = valid_document().to_dict()
    value["bindings"]["status"] = {"state": "failed", "terminal": True}
    value["bindings"]["failure"] = {
        "reason": "model_load_failed", "phase": "model_loading",
        "type": "builtins.RuntimeError", "message": "boom",
    }
    value["bindings"]["target"] = {
        "status": "unavailable", "reason": "target was not resolved",
        "comparison_line": "value_path", "target_ref": TARGET,
        "clean_input_normalized_sha256": None,
        "identity_sha256": None, "identity": None,
    }
    value["bindings"]["seeds"].update(
        fidelity_enabled=False, fidelity_pairing_context=None,
        fidelity_seeds=[],
    )
    value["bindings"]["perturbations"].update(enabled=False, items=[])
    value["bindings"]["resources"]["actual"].update(
        status="failed",
        exception={"type": "builtins.RuntimeError", "message": "boom"},
    )
    value["bindings"]["artifact_inventory"].append({
        "level": "L0", "path": "L0/failure.json", "sha256": "f" * 64,
        "bytes": 0, "kind": "failure", "dtype": "json", "shape": None,
        "format": "json",
    })
    value["bindings"]["resources"]["actual"]["files"].append({
        "path": "L0/failure.json", "bytes": 0,
    })
    validate_stage_c_provenance(value)

    value["bindings"]["target"]["identity_sha256"] = "0" * 64
    with pytest.raises(StageCProvenanceError, match="fabricated"):
        validate_stage_c_provenance(value)


def test_pre_target_failure_may_explicitly_lack_input_tensor_hashes():
    value = valid_document().to_dict()
    value["bindings"]["status"] = {"state": "failed", "terminal": True}
    value["bindings"]["failure"] = {
        "reason": "model_load_failed", "phase": "model_loading",
        "type": "builtins.RuntimeError", "message": "boom",
    }
    value["bindings"]["input"]["canonical_rgb_sha256"] = None
    value["bindings"]["input"]["normalized_tensor_sha256"] = None
    value["bindings"]["target"] = {
        "status": "unavailable", "reason": "target was not resolved",
        "comparison_line": "value_path", "target_ref": TARGET,
        "clean_input_normalized_sha256": None,
        "identity_sha256": None, "identity": None,
    }
    value["bindings"]["seeds"].update(
        fidelity_enabled=False, fidelity_pairing_context=None,
        fidelity_seeds=[],
    )
    value["bindings"]["perturbations"].update(enabled=False, items=[])
    value["bindings"]["perturbations"]["clean"] = {
        "id": "clean",
        "canonical_rgb": {
            "domain": CANONICAL_RGB_HASH_DOMAIN, "sha256": None,
        },
        "normalized_input": {
            "domain": NORMALIZED_INPUT_HASH_DOMAIN, "sha256": None,
        },
    }
    value["bindings"]["resources"]["actual"].update(
        status="failed",
        exception={"type": "builtins.RuntimeError", "message": "boom"},
    )
    failure_record = {
        "level": "L0", "path": "L0/failure.json", "sha256": "f" * 64,
        "bytes": 0, "kind": "failure", "dtype": "json", "shape": None,
        "format": "json",
    }
    value["bindings"]["artifact_inventory"].append(failure_record)
    value["bindings"]["resources"]["actual"]["files"].append({
        "path": "L0/failure.json", "bytes": 0,
    })
    validate_stage_c_provenance(value)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("one_input_hash", "both be null or both present"),
        ("target_resolved", "canonical_rgb_sha256"),
        ("perturbations_enabled", "disabled perturbations"),
        ("late_failure", "pre-target"),
        ("clean_fake_hash", "disagrees with input canonical hash"),
    ],
)
def test_unavailable_input_tensor_state_fails_closed(mutation, match):
    value = valid_document().to_dict()
    value["bindings"]["status"] = {"state": "failed", "terminal": True}
    value["bindings"]["failure"] = {
        "reason": "setup_failed", "phase": "setup",
        "type": "builtins.RuntimeError", "message": "boom",
    }
    value["bindings"]["input"]["canonical_rgb_sha256"] = None
    value["bindings"]["input"]["normalized_tensor_sha256"] = None
    value["bindings"]["target"] = {
        "status": "unavailable", "reason": "target was not resolved",
        "comparison_line": "value_path", "target_ref": TARGET,
        "clean_input_normalized_sha256": None,
        "identity_sha256": None, "identity": None,
    }
    value["bindings"]["seeds"].update(
        fidelity_enabled=False, fidelity_pairing_context=None,
        fidelity_seeds=[],
    )
    value["bindings"]["perturbations"].update(enabled=False, items=[])
    value["bindings"]["perturbations"]["clean"]["canonical_rgb"]["sha256"] = None
    value["bindings"]["perturbations"]["clean"]["normalized_input"]["sha256"] = None
    value["bindings"]["resources"]["actual"].update(
        status="failed",
        exception={"type": "builtins.RuntimeError", "message": "boom"},
    )
    value["bindings"]["artifact_inventory"].append({
        "level": "L0", "path": "L0/failure.json", "sha256": "f" * 64,
        "bytes": 0, "kind": "failure", "dtype": "json", "shape": None,
        "format": "json",
    })
    value["bindings"]["resources"]["actual"]["files"].append({
        "path": "L0/failure.json", "bytes": 0,
    })
    if mutation == "one_input_hash":
        value["bindings"]["input"]["canonical_rgb_sha256"] = CANONICAL_SHA
    elif mutation == "target_resolved":
        value["bindings"]["target"].update(status="resolved", reason=None)
    elif mutation == "perturbations_enabled":
        value["bindings"]["perturbations"]["enabled"] = True
    elif mutation == "late_failure":
        value["bindings"]["failure"]["phase"] = "metrics"
    elif mutation == "clean_fake_hash":
        value["bindings"]["perturbations"]["clean"]["canonical_rgb"]["sha256"] = "0" * 64
    with pytest.raises(StageCProvenanceError, match=match):
        validate_stage_c_provenance(value)

    value = valid_document().to_dict()
    value["bindings"]["resources"]["actual"]["cuda"]["peak_allocated_bytes"] = 0
    with pytest.raises(StageCProvenanceError, match="fabricated zeros"):
        validate_stage_c_provenance(value)

    value = valid_document().to_dict()
    value["bindings"]["artifact_inventory"][0]["path"] = "L0/provenance.json"
    with pytest.raises(StageCProvenanceError, match="exclude itself"):
        validate_stage_c_provenance(value)


def test_unknown_nonfinite_and_non_json_safe_data_are_rejected():
    value = valid_document().to_dict()
    value["extra"] = True
    with pytest.raises(StageCProvenanceError, match="unknown"):
        validate_stage_c_provenance(value)


def test_schema_versions_reject_boolean_and_selection_winners_are_ordered():
    for path in (
        ("schema_version",),
        ("bindings", "input", "schema_version"),
        ("bindings", "selection", "schema_version"),
        ("bindings", "resources", "preflight", "schema_version"),
    ):
        value = valid_document().to_dict()
        cursor = value
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = True
        with pytest.raises(StageCProvenanceError):
            validate_stage_c_provenance(value)

    value = valid_document(selection_present=True).to_dict()
    value["bindings"]["selection"]["depth_band_winners"].reverse()
    with pytest.raises(StageCProvenanceError, match="ordered"):
        validate_stage_c_provenance(value)


def test_failed_resource_exception_must_match_failure_and_inventory():
    value = valid_document().to_dict()
    failure = {
        "reason": "boom", "phase": "compute",
        "type": "builtins.RuntimeError", "message": "boom",
    }
    value["bindings"]["status"] = {"state": "failed", "terminal": True}
    value["bindings"]["failure"] = failure
    value["bindings"]["resources"]["actual"].update(
        status="failed",
        exception={"type": "builtins.ValueError", "message": "different"},
    )
    value["bindings"]["artifact_inventory"].append({
        "level": "L0", "path": "L0/failure.json", "sha256": "f" * 64,
        "bytes": 0, "kind": "failure", "dtype": "json", "shape": None,
        "format": "json",
    })
    value["bindings"]["resources"]["actual"]["files"].append({
        "path": "L0/failure.json", "bytes": 0,
    })
    with pytest.raises(StageCProvenanceError, match="disagrees"):
        validate_stage_c_provenance(value)

    value = valid_document().to_dict()
    value["bindings"]["resources"]["actual"]["wall_time_seconds"] = float("nan")
    with pytest.raises(StageCProvenanceError, match="non-finite"):
        validate_stage_c_provenance(value)

    value = valid_document().to_dict()
    value["bindings"]["perturbations"]["items"][0]["parameters"] = object()
    with pytest.raises(TypeError, match="non-JSON-safe"):
        validate_stage_c_provenance(value)


def test_global_unit_rejects_fake_block_head_and_config_id_is_hashed():
    with pytest.raises(StageCProvenanceError, match="fake block/head"):
        build_unit_identity(
            sample_id=SAMPLE,
            method_id="vit_grad_cam_block11_norm1",
            config_id="block_00_head_00",
            block_index=0,
            head_index=0,
            target_ref="clean_predicted_class_raw_logit",
        )
    left = build_unit_identity(
        sample_id=SAMPLE, method_id=METHOD, config_id="block_02_head_01",
        block_index=2, head_index=1, target_ref=TARGET,
    )
    right = dict(left)
    right["config_id"] = "block_03_head_01"
    assert canonical_sha256({key: right[key] for key in right if key != "canonical_digest"}) != left["canonical_digest"]


def test_expected_unit_and_circular_hash_claims_are_checked():
    value = valid_document().to_dict()
    other = build_unit_identity(
        sample_id=SAMPLE, method_id=METHOD, config_id="block_03_head_01",
        block_index=3, head_index=1, target_ref=TARGET,
    )
    with pytest.raises(StageCProvenanceError, match="expected unit"):
        validate_stage_c_provenance(value, expected_unit=other)

    value = valid_document().to_dict()
    value["bindings"]["resources"]["preflight"]["manifest_sha256"] = SHA
    with pytest.raises(StageCProvenanceError, match="circular"):
        validate_stage_c_provenance(value)


def test_caller_mapping_is_copied_before_freeze():
    mutable = valid_document().to_dict()
    record = build_stage_c_provenance(
        unit_identity=mutable["unit_identity"],
        input_record=mutable["bindings"].pop("input"),
        map_record=mutable["bindings"].pop("map"),
        artifact_inventory=mutable["bindings"].pop("artifact_inventory"),
        **mutable["bindings"],
    )
    mutable["unit_identity"]["sample_id"] = "changed"
    assert record.unit_identity["sample_id"] == SAMPLE
    assert isinstance(record.bindings, MappingProxyType)
