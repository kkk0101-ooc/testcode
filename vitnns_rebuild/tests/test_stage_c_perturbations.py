from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vitnns.display import DisplayBoundaryError, DisplayTransform
from vitnns.stage_c_maps import raw_array, stage_c_metric_ready
from vitnns.stage_c_perturbations import (
    NOISE_BASE_SEED,
    NOISE_ID,
    SCALE_ID,
    SCALE_INVERSE_MASS_JACOBIAN,
    TRANSLATION_ID,
    StageCPerturbationError,
    center_crop_216_resize_224,
    gaussian_rgb_std_0_01,
    generate_stage_c_perturbations,
    inverse_align_pixel_map,
    perturbation_masks,
    translate_right_down_4,
    tensor_sha256,
)
from vitnns.stage_c_class_logit import _tensor_sha256 as scalar_tensor_sha256


def clean_rgb() -> torch.Tensor:
    return torch.arange(3 * 224 * 224, dtype=torch.float32).reshape(1, 3, 224, 224) / (3 * 224 * 224)


def test_translation_moves_pixels_right_down_four_and_uses_reflection_padding():
    clean = clean_rgb()
    actual = translate_right_down_4(clean)
    expected = F.pad(clean, (4, 0, 4, 0), mode="reflect")[:, :, :224, :224]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[:, :, 4:, 4:], clean[:, :, :220, :220])
    assert actual.data_ptr() != clean.data_ptr()


def test_crop_scale_is_exact_frozen_bicubic_antialias_operator():
    clean = clean_rgb()
    expected = F.interpolate(clean[:, :, 4:220, 4:220], size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    actual = center_crop_216_resize_224(clean)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_noise_uses_cpu_float32_registered_seed_is_repeatable_and_clamped():
    clean = torch.full((1, 3, 224, 224), 0.5, dtype=torch.float32)
    first = gaussian_rgb_std_0_01(clean, selection_order=7)
    second = gaussian_rgb_std_0_01(clean, selection_order=7)
    other = gaussian_rgb_std_0_01(clean, selection_order=8)
    generator = torch.Generator(device="cpu").manual_seed(NOISE_BASE_SEED + 7)
    expected = torch.clamp(clean + 0.01 * torch.randn(clean.shape, generator=generator, dtype=torch.float32), 0, 1)
    torch.testing.assert_close(first, expected, atol=0, rtol=0)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.equal(first, other)
    assert first.device.type == "cpu" and first.dtype is torch.float32
    bounded = gaussian_rgb_std_0_01(torch.zeros_like(clean), selection_order=0)
    assert float(bounded.min()) == 0.0 and float(bounded.max()) <= 1.0


@pytest.mark.parametrize(
    "bad,exception,match",
    [
        (np.zeros((1, 3, 224, 224), np.float32), TypeError, "Tensor"),
        (torch.zeros((3, 224, 224)), StageCPerturbationError, "shape"),
        (torch.zeros((1, 3, 224, 224), dtype=torch.float64), TypeError, "float32"),
        (torch.full((1, 3, 224, 224), 1.1), StageCPerturbationError, r"\[0,1\]"),
        (torch.full((1, 3, 224, 224), float("nan")), StageCPerturbationError, "finite"),
    ],
)
def test_perturbations_fail_closed_on_noncanonical_input(bad, exception, match):
    with pytest.raises(exception, match=match):
        translate_right_down_4(bad)


@pytest.mark.parametrize("order", [True, -1, 1.5])
def test_noise_rejects_invalid_selection_order(order):
    with pytest.raises(TypeError, match="selection_order"):
        gaussian_rgb_std_0_01(torch.zeros((1, 3, 224, 224)), selection_order=order)


def test_frozen_masks_match_exact_pixel_boxes_and_complete_patch_counts():
    translation_pixels, translation_patches = perturbation_masks(TRANSLATION_ID)
    scale_pixels, scale_patches = perturbation_masks(SCALE_ID)
    noise_pixels, noise_patches = perturbation_masks(NOISE_ID)
    assert translation_pixels.sum() == 220 * 220
    assert translation_patches.sum() == 13 * 13
    assert translation_patches[:13, :13].all() and not translation_patches[13].any()
    assert scale_pixels.sum() == 216 * 216
    assert scale_patches.sum() == 12 * 12
    assert scale_patches[1:13, 1:13].all()
    assert not scale_patches[0].any() and not scale_patches[13].any()
    assert noise_pixels.all() and noise_patches.all()
    with pytest.raises(ValueError):
        translation_pixels.setflags(write=True)


def test_generated_bundle_is_immutable_json_safe_and_does_not_alias_clean():
    clean = torch.full((1, 3, 224, 224), 0.5, dtype=torch.float32)
    bundle = generate_stage_c_perturbations(clean, selection_order=3)
    assert tuple(bundle) == (TRANSLATION_ID, SCALE_ID, NOISE_ID)
    before = bundle[NOISE_ID].tensor.clone()
    clean.zero_()
    torch.testing.assert_close(bundle[NOISE_ID].tensor, before)
    exposed = bundle[NOISE_ID].tensor
    exposed.zero_()
    torch.testing.assert_close(bundle[NOISE_ID].tensor, before)
    identities = []
    record = bundle[NOISE_ID]
    record.consume(lambda shared: identities.append(id(shared)))
    record.consume(lambda shared: identities.append(id(shared)))
    assert identities[0] == identities[1]
    assert record.provenance["seed"] == NOISE_BASE_SEED + 3
    assert json.loads(json.dumps(record.to_dict()))["fully_valid_patch_count"] == 196
    with pytest.raises(TypeError):
        bundle["extra"] = record
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.tensor_sha256 = "0" * 64
    assert record.tensor_sha256 == scalar_tensor_sha256(record.tensor)
    assert record.tensor_sha256 == tensor_sha256(record.tensor)
    torch.testing.assert_close(record.normalized_tensor, (record.tensor - 0.5) / 0.5)
    assert record.normalized_tensor_sha256 == scalar_tensor_sha256(
        record.normalized_tensor
    )
    with pytest.raises(StageCPerturbationError, match="must not return"):
        record.consume(lambda shared: shared)
    with pytest.raises(StageCPerturbationError, match="must not return"):
        record.consume(lambda shared: shared.view_as(shared))


def test_shared_tensor_consumer_detects_mutation_restores_canonical_bytes():
    record = generate_stage_c_perturbations(
        torch.full((1, 3, 224, 224), 0.5), selection_order=1
    )[NOISE_ID]
    with pytest.raises(StageCPerturbationError, match="mutated.*restored"):
        record.consume(lambda shared: shared.zero_())
    record.consume(lambda shared: None)
    assert record.tensor_sha256 == tensor_sha256(record.tensor)


@pytest.mark.parametrize("metadata", [{"x": float("nan")}, {"x": object()}, {1: "bad"}])
def test_perturbation_result_rejects_non_json_safe_metadata(metadata):
    record = generate_stage_c_perturbations(torch.zeros((1, 3, 224, 224)), selection_order=0)[NOISE_ID]
    from vitnns.stage_c_perturbations import PerturbationResult
    with pytest.raises((TypeError, StageCPerturbationError)):
        PerturbationResult(record.perturbation_id, record.tensor, record.tensor_sha256,
                           record.valid_pixel_mask, record.fully_valid_patch_mask, metadata)


def test_translation_inverse_and_noise_identity_are_exact_on_valid_pixels():
    clean_map = np.arange(224 * 224, dtype=np.float32).reshape(224, 224)
    perturbed = np.zeros_like(clean_map)
    perturbed[4:, 4:] = clean_map[:220, :220]
    translation = inverse_align_pixel_map(raw_array(perturbed), TRANSLATION_ID)
    np.testing.assert_array_equal(translation.data[:220, :220], clean_map[:220, :220])
    assert np.count_nonzero(translation.data[220:, :]) == 0
    identity = inverse_align_pixel_map(raw_array(clean_map), NOISE_ID)
    np.testing.assert_array_equal(identity.data, clean_map)


def test_scale_inverse_uses_bilinear_antialias_and_multiplies_mass_jacobian():
    source = np.ones((224, 224), dtype=np.float32)
    aligned = inverse_align_pixel_map(raw_array(source), SCALE_ID)
    np.testing.assert_allclose(aligned.data[4:220, 4:220], SCALE_INVERSE_MASS_JACOBIAN, atol=3e-7, rtol=3e-7)
    assert np.count_nonzero(aligned.data[:4]) == 0
    expected_mass = 216 * 216 * SCALE_INVERSE_MASS_JACOBIAN
    assert float(aligned.data.sum(dtype=np.float64)) == pytest.approx(expected_mass, rel=1e-7)


def test_inverse_geometry_rejects_bare_metric_display_bad_shape_and_unknown_id():
    source = np.ones((224, 224), dtype=np.float32)
    with pytest.raises(DisplayBoundaryError):
        inverse_align_pixel_map(source, NOISE_ID)
    with pytest.raises(DisplayBoundaryError):
        inverse_align_pixel_map(stage_c_metric_ready(np.ones((14, 14), np.float32)), NOISE_ID)
    display = DisplayTransform()(stage_c_metric_ready(np.ones((14, 14), np.float32)))
    with pytest.raises(DisplayBoundaryError):
        inverse_align_pixel_map(display, NOISE_ID)
    with pytest.raises(StageCPerturbationError, match="shape"):
        inverse_align_pixel_map(raw_array(np.ones((223, 224), np.float32)), NOISE_ID)
    with pytest.raises(StageCPerturbationError, match="unknown"):
        inverse_align_pixel_map(raw_array(source), "other")
