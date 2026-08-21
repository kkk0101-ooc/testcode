from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from vitnns.display import DisplayBoundaryError, DisplayTransform
from vitnns.stage_c_maps import (
    ComparisonLine, MethodResult, NAReason, SignSemantics, StructuredNA, raw_array,
    stage_c_metric_ready,
)
from vitnns.stage_c_metrics import MetricValue
from vitnns.stage_c_perturbations import NOISE_ID, SCALE_ID, perturbation_masks
from vitnns.stage_c_perturbations import generate_stage_c_perturbations
from vitnns.stage_c_stability import (
    AlignedMapResult, PredictionObservation, StageCStabilityError, aligned_map_result,
    evaluate_map_stability, inverse_aligned_map_result, patch_inner_residual,
    residual_signed_spearman,
)

HASH = "a" * 64
CLEAN_HASH = "c" * 64
VALUE_FROZEN = {
    "target_kind": "fixed_clean_attention_cls_head_direction",
    "clean_input_sha256": CLEAN_HASH, "candidate": {"block_index": 2, "head_index": 3},
    "runtime_model_binding": {"sha256": "d" * 64},
}
TARGET = {"comparison_line": "value_path",
          "target_ref": "fixed_clean_attention_cls_head_direction",
          "clean_input_sha256": CLEAN_HASH, "frozen_target": VALUE_FROZEN}
CLASS_FROZEN = {"target_kind": "raw_logit", "clean_input_sha256": CLEAN_HASH,
                "class_index": 7, "class_count": 1000,
                "runtime_model_binding": {"sha256": "d" * 64}}
CLASS_TARGET = {"comparison_line": "class_logit",
                "target_ref": "clean_predicted_class_raw_logit",
                "clean_input_sha256": CLEAN_HASH, "frozen_target": CLASS_FROZEN}


def perturbation(perturbation_id=NOISE_ID):
    bundle = generate_stage_c_perturbations(
        torch.full((1, 3, 224, 224), 0.5, dtype=torch.float32), selection_order=0,
    )
    return bundle[perturbation_id]


def context(perturbation_id=NOISE_ID):
    return dict(method_id="conditional_fixed_clean_attention_head_direction_value_path_ig",
                method_version="1.0.0", comparison_line=ComparisonLine.VALUE_PATH,
                target_identity=TARGET, perturbation=perturbation(perturbation_id))


def bind_perturbed(clean, item):
    return dataclasses.replace(
        clean, provenance={**dict(clean.provenance),
                           "intervention_input_sha256": item.normalized_tensor_sha256},
    )


def aligned(clean, identity=TARGET, perturbation_id=NOISE_ID, perturbed=None):
    item = perturbation(perturbation_id)
    perturbed = bind_perturbed(clean, item) if perturbed is None else perturbed
    return aligned_map_result(
        clean, target_identity=identity, perturbation=item,
        perturbed_method_result=perturbed,
    )


def observation(index, identity, intervention_hash):
    runtime = identity["frozen_target"]["runtime_model_binding"]["sha256"]
    return PredictionObservation(index, 1000, intervention_hash, runtime)


def residual(pixels, perturbation_id=NOISE_ID):
    return patch_inner_residual(raw_array(np.asarray(pixels, np.float32)), **context(perturbation_id))


def method(values, *, semantics=SignSemantics.SIGNED):
    pixels = np.repeat(np.repeat(np.asarray(values, np.float32), 16, 0), 16, 1) / np.float32(256)
    native = np.stack((pixels, np.zeros_like(pixels), np.zeros_like(pixels)))
    return MethodResult(
        method_id="conditional_fixed_clean_attention_head_direction_value_path_ig",
        method_version="1.0.0", comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target="fixed_clean_attention_cls_head_direction",
        fidelity_target="fixed_clean_attention_cls_head_direction", class_agnostic=True,
        native_tensor=raw_array(native), metric_map=stage_c_metric_ready(np.asarray(values, np.float32)),
        projection="pixel_mass_to_patch_sum", sign_semantics=semantics,
        valid_mask=np.ones((14, 14), bool), provenance={"target": VALUE_FROZEN},
    )


def class_attention(values, identity=CLASS_TARGET, valid_mask=None):
    values = np.asarray(values, np.float32)
    return MethodResult(
        method_id="direct_raw_cls_attention_last_block_head_mean", method_version="1.0.0",
        comparison_line=ComparisonLine.CLASS_LOGIT, explanation_target=None,
        fidelity_target="clean_predicted_class_raw_logit", class_agnostic=True,
        native_tensor=raw_array(values), metric_map=stage_c_metric_ready(values),
        projection="raw_cls_attention_to_patch", sign_semantics=SignSemantics.NONNEGATIVE,
        valid_mask=np.ones((14, 14), bool) if valid_mask is None else valid_mask,
        provenance={"caller_context": {"fidelity_target_identity": identity}},
    )


def class_grad_cam(values, frozen=CLASS_FROZEN):
    values = np.asarray(values, np.float32)
    return MethodResult(
        method_id="vit_grad_cam_block11_norm1", method_version="1.0.0",
        comparison_line=ComparisonLine.CLASS_LOGIT,
        explanation_target="clean_predicted_class_raw_logit",
        fidelity_target="clean_predicted_class_raw_logit", class_agnostic=False,
        native_tensor=raw_array(values), metric_map=stage_c_metric_ready(values),
        projection="identity_14x14", sign_semantics=SignSemantics.NONNEGATIVE,
        valid_mask=np.ones((14, 14), bool), provenance={"target": frozen},
    )


def test_uniform_and_zero_residual_are_explicit_spatial_failures():
    patches = np.arange(1, 197, dtype=np.float32).reshape(14, 14)
    uniform = residual(np.repeat(np.repeat(patches, 16, 0), 16, 1))
    assert uniform.rho.value == 0.0
    assert uniform.status == "SPATIAL_FAILURE" and uniform.failure_reason == "constant_residual"
    zero = residual(np.zeros((224, 224), np.float32))
    assert isinstance(zero.rho, StructuredNA) and zero.rho.reason is NAReason.ZERO_MASS_LEFT
    assert zero.spatial_failure and zero.failure_reason == "zero_map_l1"
    json.dumps(zero.to_dict(), allow_nan=False)


def test_signed_cancellation_and_hand_computable_rho():
    pixels = np.zeros((224, 224), np.float32); pixels[0, 0] = 1; pixels[0, 1] = -1
    result = residual(pixels)
    assert result.residual.data[0, 0] == 1 and result.residual.data[0, 1] == -1
    assert result.rho.value == pytest.approx(1) and result.status == "OK"
    single = np.zeros_like(pixels); single[0, 0] = 4
    result = residual(single)
    expected = (4 - 4 / 256) + 255 * 4 / 256
    assert result.rho.value == pytest.approx(expected / 4)


def test_partial_boundary_uses_registered_mask_and_valid_pixel_divisor():
    pixels = np.full((224, 224), 1000, np.float32)
    valid, _ = perturbation_masks(SCALE_ID); pixels[valid] = 3
    result = residual(pixels, SCALE_ID)
    assert result.spatial_failure and result.rho.value == 0
    assert np.count_nonzero(result.residual.data[~valid]) == 0
    assert result.details["partial_patch_mass_divisor"] == "valid_pixel_count"


def test_residual_requires_exact_registered_masks_and_pair_identity():
    pixels = np.zeros((224, 224), np.float32); pixels[0, :16] = np.arange(16)
    bad = np.ones((224, 224), bool); bad[0, 0] = False
    with pytest.raises(StageCStabilityError, match="registered"):
        from vitnns.stage_c_stability import _registered_masks
        _registered_masks(NOISE_ID, bad, np.ones((14, 14), bool))
    left = residual(pixels)
    other = generate_stage_c_perturbations(
        torch.full((1, 3, 224, 224), 0.4, dtype=torch.float32),
        selection_order=1
    )[NOISE_ID]
    right = patch_inner_residual(
        raw_array(pixels), **{**context(), "perturbation": other}
    )
    with pytest.raises(StageCStabilityError, match="identity mismatch"):
        residual_signed_spearman(left, right)


def test_residual_spearman_success_and_constant_failure_na():
    pixels = np.zeros((224, 224), np.float32); pixels.ravel()[:100] = np.arange(100)
    first = residual(pixels); second = residual(pixels * 3)
    result = residual_signed_spearman(first, second)
    assert result.value == pytest.approx(1) and result.details["common_valid_pixel_count"] == 224 ** 2
    constant = residual(np.ones((224, 224), np.float32))
    assert residual_signed_spearman(constant, first).reason is NAReason.CONSTANT_LEFT


def test_aligned_result_binds_method_target_hash_and_registered_masks():
    clean = method(np.arange(196).reshape(14, 14))
    issued = aligned(clean)
    assert isinstance(issued, AlignedMapResult)
    assert issued.method_id == clean.method_id
    assert issued.target_identity["frozen_target"]["candidate"]["head_index"] == 3
    assert json.loads(json.dumps(issued.to_dict()))["fully_valid_patch_count"] == 196
    with pytest.raises(TypeError):
        issued.target_identity["x"] = 1
    with pytest.raises(TypeError):
        aligned_map_result(
            clean, target_identity=TARGET, perturbation_id=NOISE_ID,
            perturbation_tensor_sha256=HASH, metric_map=clean.metric_map,
            valid_pixel_mask=np.ones((224, 224), bool),
            fully_valid_patch_mask=np.ones((14, 14), bool),
        )


def test_inverse_aligned_result_derives_map_from_bound_perturbed_native_map():
    clean = method(np.arange(196).reshape(14, 14))
    bundle = generate_stage_c_perturbations(
        torch.full((1, 3, 224, 224), 0.5, dtype=torch.float32),
        selection_order=0,
    )
    perturbation = bundle[NOISE_ID]
    perturbed = method(np.arange(196).reshape(14, 14))
    provenance = dict(perturbed.provenance)
    provenance["intervention_input_sha256"] = perturbation.normalized_tensor_sha256
    perturbed = dataclasses.replace(perturbed, provenance=provenance)
    aligned = inverse_aligned_map_result(
        clean, target_identity=TARGET, perturbation=perturbation,
        perturbed_method_result=perturbed,
    )
    np.testing.assert_allclose(
        aligned.metric_map.data, perturbed.metric_map.data, atol=1e-6, rtol=1e-6
    )
    bad = dataclasses.replace(
        perturbed, provenance={**provenance, "intervention_input_sha256": "f" * 64}
    )
    with pytest.raises(StageCStabilityError, match="intervention hash"):
        inverse_aligned_map_result(
            clean, target_identity=TARGET, perturbation=perturbation,
            perturbed_method_result=bad,
        )


def test_general_stability_uses_c1_metrics_mask_and_separates_prediction_fields():
    values = np.arange(196, dtype=np.float32).reshape(14, 14) - 80
    clean = method(values)
    item = perturbation(SCALE_ID)
    issued = aligned_map_result(
        clean, target_identity=TARGET, perturbation=item,
        perturbed_method_result=bind_perturbed(clean, item),
    )
    result = evaluate_map_stability(
        clean, issued, target_identity=TARGET,
        clean_prediction=observation(4, TARGET, CLEAN_HASH),
        perturbed_natural_prediction=observation(9, TARGET, item.normalized_tensor_sha256),
    )
    assert set(result.metrics) == {"pearson", "spearman", "jensen_shannon", "top10_jaccard", "top20_jaccard"}
    assert set(result.metrics["pearson"]) == {"signed", "positive", "negative"}
    assert result.metrics["pearson"]["signed"].value == pytest.approx(1, abs=5e-6)
    assert result.metrics["pearson"]["positive"].value > 0.9999
    assert result.metrics["pearson"]["negative"].value > 0.9999
    assert result.metrics["jensen_shannon"]["positive"].value < 2e-4
    assert result.metrics["jensen_shannon"]["negative"].value < 5e-4
    assert result.metrics["top10_jaccard"]["positive"].details["n_valid"] == 144
    payload = result.to_dict(); json.dumps(payload, allow_nan=False)
    assert payload["fixed_target_primary"]["clean_target_fixed"] is True
    descriptive = payload["natural_prediction_descriptive"]
    assert descriptive["clean_prediction"]["class_index"] == 4
    assert descriptive["perturbed_prediction"]["class_index"] == 9
    assert not descriptive["prediction_unchanged"] and descriptive["prediction_flipped"]


def test_class_line_uses_raw_logit_payload_but_class_target_ref_and_c2_binding():
    values = np.arange(1, 197, dtype=np.float32).reshape(14, 14)
    clean = class_attention(values)
    item = perturbation(NOISE_ID)
    issued = aligned_map_result(
        clean, target_identity=CLASS_TARGET, perturbation=item,
        perturbed_method_result=bind_perturbed(clean, item),
    )
    result = evaluate_map_stability(
        clean, issued, target_identity=CLASS_TARGET,
        clean_prediction=observation(7, CLASS_TARGET, CLEAN_HASH),
        perturbed_natural_prediction=observation(7, CLASS_TARGET, item.normalized_tensor_sha256),
    )
    assert result.comparison_line is ComparisonLine.CLASS_LOGIT
    assert result.target_identity["target_ref"] == "clean_predicted_class_raw_logit"
    assert result.target_identity["frozen_target"]["target_kind"] == "raw_logit"
    assert result.prediction_unchanged and not result.prediction_flipped
    for metric in ("pearson", "spearman"):
        assert result.metrics[metric]["signed"].reason is NAReason.UNSUPPORTED_SIGN_SEMANTICS
        assert result.metrics[metric]["negative"].reason is NAReason.UNSUPPORTED_SIGN_SEMANTICS
        assert result.metrics[metric]["positive"].value == pytest.approx(1)
    for metric in ("jensen_shannon", "top10_jaccard", "top20_jaccard"):
        assert result.metrics[metric]["negative"].reason is NAReason.UNSUPPORTED_SIGN_SEMANTICS
    payload = result.to_dict()
    assert payload["fixed_target_primary"]["metrics"]["pearson"]["signed"] == {
        "status": "NA", "value": None, "reason": "unsupported_sign_semantics"
    }


def test_c3_class_method_binds_provenance_target_payload():
    values = np.arange(1, 197, dtype=np.float32).reshape(14, 14)
    clean = class_grad_cam(values)
    item = perturbation(NOISE_ID)
    issued = aligned_map_result(
        clean, target_identity=CLASS_TARGET, perturbation=item,
        perturbed_method_result=bind_perturbed(clean, item),
    )
    result = evaluate_map_stability(
        clean, issued, target_identity=CLASS_TARGET,
        clean_prediction=observation(7, CLASS_TARGET, CLEAN_HASH),
        perturbed_natural_prediction=observation(8, CLASS_TARGET, item.normalized_tensor_sha256),
    )
    assert result.prediction_flipped
    wrong_class = {**CLASS_TARGET, "frozen_target": {**CLASS_FROZEN, "class_index": 8}}
    with pytest.raises(StageCStabilityError, match="target identity mismatch"):
        aligned(clean, wrong_class)


def test_stability_rejects_target_or_method_drift_and_nonfinite_metadata():
    values = np.arange(196, dtype=np.float32).reshape(14, 14) + 1
    clean = method(values)
    with pytest.raises(StageCStabilityError, match="non-finite"):
        aligned(clean, {**TARGET, "frozen_target": {**VALUE_FROZEN, "direction_norm": float("nan")}})
    issued = aligned(clean)
    item = perturbation(NOISE_ID)
    with pytest.raises(StageCStabilityError, match="target identity mismatch"):
        evaluate_map_stability(
            clean, issued, target_identity={**TARGET, "clean_input_sha256": "e" * 64, "frozen_target": {**VALUE_FROZEN, "clean_input_sha256": "e" * 64}},
            clean_prediction=observation(1, TARGET, CLEAN_HASH),
            perturbed_natural_prediction=observation(1, TARGET, item.normalized_tensor_sha256),
        )
    with pytest.raises(StageCStabilityError, match="comparison_line/target_ref"):
        aligned(clean, {**TARGET, "target_ref": "clean_predicted_class_raw_logit"})
    perturbed_clean = dataclasses.replace(
        clean, provenance={**dict(clean.provenance),
                           "intervention_input_sha256": "f" * 64},
    )
    with pytest.raises(StageCStabilityError, match="not evaluated.*clean"):
        evaluate_map_stability(
            perturbed_clean, issued, target_identity=TARGET,
            clean_prediction=observation(1, TARGET, CLEAN_HASH),
            perturbed_natural_prediction=observation(1, TARGET, item.normalized_tensor_sha256),
        )


@pytest.mark.parametrize(
    "changed",
    [
        {**TARGET, "clean_input_sha256": "e" * 64, "frozen_target": {**VALUE_FROZEN, "clean_input_sha256": "e" * 64}},
        {**TARGET, "frozen_target": {**VALUE_FROZEN, "candidate": {"block_index": 2, "head_index": 4}}},
        {**TARGET, "frozen_target": {**VALUE_FROZEN, "runtime_model_binding": {"sha256": "e" * 64}}},
    ],
)
def test_aligned_rejects_wrong_sample_head_or_model_identity(changed):
    clean = method(np.arange(196, dtype=np.float32).reshape(14, 14))
    with pytest.raises(StageCStabilityError, match="target identity mismatch"):
        aligned(clean, changed)


def test_class_binding_rejects_wrong_class_and_clean_mask_must_be_all_true():
    values = np.arange(1, 197, dtype=np.float32).reshape(14, 14)
    wrong = {**CLASS_TARGET, "frozen_target": {**CLASS_FROZEN, "class_index": 8}}
    clean = class_attention(values)
    with pytest.raises(StageCStabilityError, match="target identity mismatch"):
        aligned(clean, wrong)
    invalid = np.ones((14, 14), bool); invalid[0, 0] = False
    partial_clean = class_attention(values, valid_mask=invalid)
    item = perturbation(NOISE_ID)
    issued = aligned(partial_clean, CLASS_TARGET)
    with pytest.raises(StageCStabilityError, match="all true"):
        evaluate_map_stability(
            partial_clean, issued, target_identity=CLASS_TARGET,
            clean_prediction=observation(7, CLASS_TARGET, CLEAN_HASH),
            perturbed_natural_prediction=observation(7, CLASS_TARGET, item.normalized_tensor_sha256),
        )


def test_stability_primitives_reject_quantitative_boundary_violations():
    bare = np.zeros((224, 224), np.float32)
    with pytest.raises(DisplayBoundaryError):
        patch_inner_residual(bare, **context())
    display = DisplayTransform()(stage_c_metric_ready(np.ones((14, 14), np.float32)))
    with pytest.raises(DisplayBoundaryError):
        patch_inner_residual(display, **context())
