from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vitnns.display import DisplayBoundaryError, DisplayTransform
from vitnns.retention import ArtifactArray, ArtifactKind
from vitnns.stage_c_maps import (
    ComparisonLine,
    INVALIDATING_NA_REASONS,
    MethodResult,
    NAReason,
    SignSemantics,
    SignView,
    StructuredNA,
    patch_to_fine_cell_mass,
    patch_to_pixel_mass,
    pixel_to_fine_cell_mass,
    pixel_to_patch_mass,
    raw_array,
    signed_rgb_channel_sum,
    split_sign_views,
    stage_c_metric_ready,
)


@pytest.mark.parametrize(
    "pixels",
    [
        np.ones((224, 224), dtype=np.float32),
        -np.ones((224, 224), dtype=np.float32),
        np.zeros((224, 224), dtype=np.float32),
        np.random.default_rng(20260821).normal(size=(224, 224)).astype(np.float32),
    ],
)
def test_pixel_patch_projection_preserves_signed_mass_and_patch_roundtrip(pixels):
    patches = pixel_to_patch_mass(raw_array(pixels))
    reconstructed = patch_to_pixel_mass(patches)
    roundtrip = pixel_to_patch_mass(reconstructed)

    assert patches.kind is ArtifactKind.METRIC_READY
    assert patches.dtype == np.float32
    assert patches.shape == (14, 14)
    assert reconstructed.kind is ArtifactKind.RAW
    assert reconstructed.shape == (224, 224)
    np.testing.assert_allclose(
        np.sum(pixels, dtype=np.float64),
        np.sum(patches.data, dtype=np.float64),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(roundtrip.data, patches.data, atol=1e-6, rtol=1e-6)
    assert not np.shares_memory(patches.data, pixels)
    with pytest.raises(ValueError):
        patches.data.setflags(write=True)


def test_rgb_sum_is_signed_and_projection_precedes_sign_split():
    rgb = np.zeros((3, 224, 224), dtype=np.float32)
    rgb[0, 0, 0] = 3.0
    rgb[1, 0, 0] = -2.0
    rgb[2, 0, 0] = -0.5
    reduced = signed_rgb_channel_sum(raw_array(rgb))
    assert reduced.data[0, 0] == pytest.approx(0.5)
    np.testing.assert_allclose(
        reduced.data.sum(dtype=np.float64), rgb.sum(dtype=np.float64), atol=1e-6, rtol=1e-6
    )

    cancel = np.zeros((224, 224), dtype=np.float32)
    cancel[0, 0] = 1.0
    cancel[0, 1] = -1.0
    projected = pixel_to_patch_mass(raw_array(cancel))
    views = split_sign_views(projected, SignSemantics.SIGNED)
    assert views[SignView.SIGNED].data[0, 0] == 0.0
    assert views[SignView.POSITIVE].data[0, 0] == 0.0
    assert views[SignView.NEGATIVE_MAGNITUDE].data[0, 0] == 0.0


def test_four_by_four_cell_operators_preserve_mass_and_patch_baseline_is_uniform():
    pixels = np.arange(224 * 224, dtype=np.float32).reshape(224, 224) / 1000.0
    cells = pixel_to_fine_cell_mass(raw_array(pixels))
    assert cells.shape == (56, 56)
    assert cells.data[0, 0] == pytest.approx(float(pixels[:4, :4].sum(dtype=np.float64)))
    np.testing.assert_allclose(cells.data.sum(dtype=np.float64), pixels.sum(dtype=np.float64), rtol=1e-6, atol=1e-6)

    patches = stage_c_metric_ready(np.arange(196, dtype=np.float32).reshape(14, 14))
    uniform = patch_to_fine_cell_mass(patches)
    assert uniform.shape == (56, 56)
    np.testing.assert_array_equal(uniform.data[:4, :4], np.full((4, 4), patches.data[0, 0] / 16, dtype=np.float32))
    np.testing.assert_allclose(uniform.data.sum(dtype=np.float64), patches.data.sum(dtype=np.float64), rtol=1e-6, atol=1e-6)


def test_nonnegative_negative_view_is_structured_na_not_fake_zero_map():
    source = stage_c_metric_ready(np.ones((14, 14), dtype=np.float32))
    views = split_sign_views(source, SignSemantics.NONNEGATIVE)
    negative = views[SignView.NEGATIVE_MAGNITUDE]
    assert isinstance(negative, StructuredNA)
    assert negative.to_dict() == {
        "status": "NA", "value": None,
        "reason": "unsupported_sign_semantics",
    }
    with pytest.raises(ValueError, match="contains negative"):
        split_sign_views(
            stage_c_metric_ready(-np.ones((14, 14), dtype=np.float32)),
            SignSemantics.NONNEGATIVE,
        )


def test_all_frozen_na_reasons_are_serializable_and_classified():
    assert {reason.value for reason in NAReason} == {
        "constant_left", "constant_right", "zero_mass_left",
        "zero_mass_right", "unsupported_sign_semantics",
        "empty_valid_region", "shape_mismatch", "non_finite",
        "degenerate_direction", "degenerate_target_delta",
        "missing_artifact",
    }
    for reason in NAReason:
        result = StructuredNA(reason)
        assert result.to_dict()["reason"] == reason.value
        assert result.invalidating is (reason in INVALIDATING_NA_REASONS)


def make_method_result(**updates) -> MethodResult:
    native = np.zeros((3, 224, 224), dtype=np.float32)
    values = {
        "method_id": "conditional_fixed_clean_attention_head_direction_value_path_ig",
        "method_version": "1.0.0",
        "comparison_line": ComparisonLine.VALUE_PATH,
        "explanation_target": "fixed_clean_attention_cls_head_direction",
        "fidelity_target": "fixed_clean_attention_cls_head_direction",
        "class_agnostic": True,
        "native_tensor": raw_array(native),
        "metric_map": pixel_to_patch_mass(
            signed_rgb_channel_sum(raw_array(native))
        ),
        "projection": "pixel_mass_to_patch_sum",
        "sign_semantics": SignSemantics.SIGNED,
        "valid_mask": np.ones((14, 14), dtype=bool),
        "provenance": {"config_sha256": "a" * 64, "projection_order": ["signed", "project", "split"]},
    }
    values.update(updates)
    return MethodResult(**values)


def test_method_result_is_immutable_and_records_complete_identity_contract():
    result = make_method_result()
    assert result.metric_map.kind is ArtifactKind.METRIC_READY
    assert result.sign_views[SignView.NEGATIVE_MAGNITUDE].kind is ArtifactKind.METRIC_READY
    assert result.provenance["projection_order"] == ("signed", "project", "split")
    with pytest.raises(TypeError):
        result.provenance["tamper"] = True
    with pytest.raises(ValueError):
        result.valid_mask.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.method_version = "2.0.0"


@pytest.mark.parametrize(
    "updates",
    [
        {"method_version": "latest"},
        {"comparison_line": ComparisonLine.CLASS_LOGIT},
        {"explanation_target": "clean_predicted_class_raw_logit"},
        {"fidelity_target": None},
        {"projection": "identity_14x14"},
        {"sign_semantics": SignSemantics.NONNEGATIVE},
        {"valid_mask": np.ones((13, 14), dtype=bool)},
        {"valid_mask": np.ones((14, 14), dtype=np.uint8)},
        {"provenance": {}},
    ],
)
def test_method_result_rejects_identity_target_projection_sign_mask_and_provenance_drift(updates):
    with pytest.raises((TypeError, ValueError)):
        make_method_result(**updates)


def test_empty_mask_requires_matching_structured_na():
    empty = np.zeros((14, 14), dtype=bool)
    with pytest.raises(ValueError, match="empty valid_mask"):
        make_method_result(valid_mask=empty)
    result = make_method_result(
        valid_mask=empty, na=StructuredNA(NAReason.EMPTY_VALID_REGION)
    )
    assert result.na.invalidating


def test_method_result_validates_metric_map_against_native_projection():
    bad_map = np.zeros((14, 14), dtype=np.float32)
    bad_map[0, 0] = 1.0
    with pytest.raises(ValueError, match="registered projection"):
        make_method_result(metric_map=stage_c_metric_ready(bad_map))


def test_failed_method_result_may_retain_na_without_fabricated_arrays():
    result = make_method_result(
        native_tensor=None,
        metric_map=None,
        valid_mask=np.zeros((14, 14), dtype=bool),
        na=StructuredNA(NAReason.MISSING_ARTIFACT),
    )
    assert result.na.reason is NAReason.MISSING_ARTIFACT
    assert all(value is result.na for value in result.sign_views.values())


def test_patch_native_method_result_uses_native_patch_mass_as_metric_map():
    patches = np.arange(196, dtype=np.float32).reshape(14, 14)
    result = MethodResult(
        method_id="direct_raw_cls_attention_per_head",
        method_version="1.0.0",
        comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target=None,
        fidelity_target=None,
        class_agnostic=True,
        native_tensor=raw_array(patches),
        metric_map=stage_c_metric_ready(patches),
        projection="raw_cls_attention_to_patch",
        sign_semantics=SignSemantics.NONNEGATIVE,
        valid_mask=np.ones((14, 14), dtype=bool),
        provenance={"source": "synthetic_contract_test"},
    )
    np.testing.assert_array_equal(result.metric_map.data, patches)


@pytest.mark.parametrize(
    "call",
    [
        lambda: stage_c_metric_ready(np.zeros((14, 14), dtype=np.float64)),
        lambda: stage_c_metric_ready(np.zeros((13, 14), dtype=np.float32)),
        lambda: stage_c_metric_ready(np.full((14, 14), np.nan, dtype=np.float32)),
        lambda: raw_array(np.array([], dtype=np.float32)),
        lambda: signed_rgb_channel_sum(np.zeros((224, 224, 3), dtype=np.float32)),
        lambda: pixel_to_patch_mass(np.zeros((223, 224), dtype=np.float32)),
    ],
)
def test_map_shape_dtype_finite_and_empty_checks_fail_closed(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_display_arrays_cannot_reenter_projection_or_metric_ready_boundary():
    metric = stage_c_metric_ready(np.ones((14, 14), dtype=np.float32))
    display = DisplayTransform(resize=(224, 224))(metric)
    with pytest.raises(DisplayBoundaryError):
        pixel_to_patch_mass(display)
    with pytest.raises(DisplayBoundaryError):
        raw_array(display)
    with pytest.raises(DisplayBoundaryError):
        stage_c_metric_ready(display)


def test_numeric_projection_requires_explicit_raw_input_tag():
    bare = np.ones((224, 224), dtype=np.float32)
    with pytest.raises(DisplayBoundaryError, match="explicitly tagged raw"):
        pixel_to_patch_mass(bare)
    metric_tag = ArtifactArray(bare, ArtifactKind.METRIC_READY)
    with pytest.raises(DisplayBoundaryError, match="requires raw"):
        pixel_to_fine_cell_mass(metric_tag)


def test_patch_expansion_requires_explicit_metric_ready_tag():
    bare = np.ones((14, 14), dtype=np.float32)
    with pytest.raises(DisplayBoundaryError, match="explicitly tagged"):
        patch_to_pixel_mass(bare)
    raw = ArtifactArray(bare, ArtifactKind.RAW)
    with pytest.raises(DisplayBoundaryError, match="requires metric_ready"):
        patch_to_fine_cell_mass(raw)
    with pytest.raises(DisplayBoundaryError, match="cannot relabel"):
        raw_array(stage_c_metric_ready(bare))
    with pytest.raises(DisplayBoundaryError, match="cannot relabel"):
        stage_c_metric_ready(raw)
