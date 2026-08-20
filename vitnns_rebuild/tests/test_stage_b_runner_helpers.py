from dataclasses import dataclass

import pytest
import torch

from vitnns.dataset import ManifestRow
from vitnns.modeling import PreprocessSpec
from vitnns.stage_b_runner import (
    _summarize_ig_result,
    ClassTargetConfig,
    SanityConfig,
    build_baseline,
    compare_attribution_maps,
    deterministic_input_perturbation,
    map_change_metrics,
    native_signed_maps,
    resolve_class_target,
    estimate_stage_b_run,
    StageBConfig,
)


@pytest.fixture
def preprocess() -> PreprocessSpec:
    return PreprocessSpec(
        input_size=(3, 224, 224),
        resize_shorter_side=248,
        center_crop=(224, 224),
        interpolation="bicubic",
        antialias=True,
        mean=(0.5, 0.25, 0.75),
        std=(0.25, 0.5, 0.25),
        crop_pct=0.9,
        crop_mode="center",
    )


def manifest_row(label: int | None = 7) -> ManifestRow:
    return ManifestRow(
        sample_id="fixture/image.JPEG",
        relative_path="n00000001/image.JPEG",
        image_sha256="0" * 64,
        size_bytes=1,
        width=224,
        height=224,
        wnid=None if label is None else "n00000001",
        imagenet1k_index=label,
        label_source="fixture",
    )


def test_resolve_class_target_modes_and_unlabeled_failure():
    logits = torch.tensor([[0.1, -2.0, 3.0, 1.0, 2.0, 0.0, -1.0, 0.5]])
    assert resolve_class_target(ClassTargetConfig("predicted"), logits, manifest_row()) == 2
    assert resolve_class_target(ClassTargetConfig("ground_truth"), logits, manifest_row()) == 7
    assert resolve_class_target(ClassTargetConfig("index", 4), logits, manifest_row()) == 4
    with pytest.raises(ValueError, match="requires a labeled"):
        resolve_class_target(
            ClassTargetConfig("ground_truth"), logits, manifest_row(label=None)
        )
    with pytest.raises(ValueError, match="outside"):
        resolve_class_target(ClassTargetConfig("index", 10), logits, manifest_row())


def test_baseline_semantics_are_unambiguous(preprocess):
    inputs = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    zero = build_baseline(inputs, "zero_normalized", preprocess)
    assert zero.shape == inputs.shape
    assert zero.dtype == inputs.dtype
    assert torch.count_nonzero(zero) == 0

    black = build_baseline(inputs, "black_rgb", preprocess)
    expected = torch.tensor([-2.0, -0.5, -3.0], dtype=torch.float64).view(1, 3, 1, 1)
    torch.testing.assert_close(black, expected.expand_as(inputs), rtol=0.0, atol=0.0)
    assert black.data_ptr() != inputs.data_ptr()


def test_seeded_perturbation_is_deterministic_and_clamped(preprocess):
    sanity = SanityConfig(
        enabled=True,
        perturbation="seeded_gaussian_noise",
        seed=42,
        noise_std=10.0,
        clamp_to_valid_range=True,
        min_relative_l1_change=0.1,
        epsilon=1e-12,
    )
    inputs = torch.zeros(1, 3, 8, 8)
    first = deterministic_input_perturbation(inputs, sanity, preprocess)
    second = deterministic_input_perturbation(inputs, sanity, preprocess)
    assert torch.equal(first, second)
    assert not torch.equal(first, inputs)
    lows = torch.tensor([-2.0, -0.5, -3.0]).view(1, 3, 1, 1)
    highs = torch.tensor([2.0, 1.5, 1.0]).view(1, 3, 1, 1)
    assert torch.all(first >= lows)
    assert torch.all(first <= highs)


def test_map_change_and_repeatability_helpers():
    reference = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
    changed = reference + 0.5
    metrics = map_change_metrics(reference, changed, epsilon=1e-12)
    assert metrics["reference_l1"] == 6.0
    assert metrics["difference_l1"] == 2.0
    assert metrics["relative_l1_change"] == pytest.approx(1.0 / 3.0)
    assert metrics["max_abs_change"] == 0.5

    exact = compare_attribution_maps(reference, reference.clone(), atol=0.0, rtol=0.0)
    assert exact == {"pass": True, "max_abs_error": 0.0, "atol": 0.0, "rtol": 0.0}
    mismatch = compare_attribution_maps(reference, changed, atol=1e-6, rtol=1e-6)
    assert mismatch["pass"] is False
    assert mismatch["max_abs_error"] == 0.5


@dataclass
class FakeIGResult:
    attribution: torch.Tensor


def test_native_signed_rgb_and_signed_sum_are_not_resized_or_absolutized():
    inputs = torch.zeros(1, 3, 224, 224)
    attribution = torch.zeros_like(inputs)
    attribution[0, 0, 0, 0] = 3.0
    attribution[0, 1, 0, 0] = -2.0
    attribution[0, 2, 0, 0] = -0.5
    signed_rgb, rgb_sum = native_signed_maps(FakeIGResult(attribution), inputs)
    assert signed_rgb.shape == (3, 224, 224)
    assert rgb_sum.shape == (224, 224)
    assert signed_rgb.dtype == torch.float32
    assert rgb_sum[0, 0].item() == 0.5
    assert torch.count_nonzero(rgb_sum) == 1


def test_native_signed_map_rejects_wrong_shape_and_nonfinite_values():
    inputs = torch.zeros(1, 3, 224, 224)
    with pytest.raises(ValueError, match="native input shape"):
        native_signed_maps(FakeIGResult(torch.zeros(1, 3, 14, 14)), inputs)
    invalid = torch.zeros_like(inputs)
    invalid[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        native_signed_maps(FakeIGResult(invalid), inputs)


def test_conservation_summary_accepts_core_length_one_tensors():
    result = type("IGResultFixture", (), {
        "conservation": {
            "delta": torch.tensor([0.5]),
            "sum": torch.tensor([0.5]),
            "abs": torch.tensor([0.0]),
            "rel": torch.tensor([0.0]),
            "pass": torch.tensor([True]),
        },
        "steps": 32,
        "converged": True,
        "attempts": ({"steps": 32, "pass": [True]},),
        "metadata": {"scope": "fixed_attention_value_path"},
    })()
    signed_rgb = torch.zeros(3, 2, 2)
    signed_rgb[0, 0, 0] = 0.5
    summary = _summarize_ig_result(result, signed_rgb, atol=1e-6, rtol=1e-6)
    assert summary["delta"] == 0.5
    assert summary["native_signed_rgb_sum"] == 0.5
    assert summary["native_pass"] is True
    assert summary["steps"] == 32


def test_compute_budget_preflight_fails_closed():
    from test_stage_b_config import valid_config_dict

    document = valid_config_dict()
    document["compute_budget"] = {
        "max_forward_calls": 1,
        "max_backward_calls": 1,
    }
    estimate = estimate_stage_b_run(StageBConfig.from_dict(document))
    assert estimate["storage_within_budget"] is True
    assert estimate["compute"]["within_budget"] is False
    assert estimate["within_budget"] is False


def test_compute_estimate_counts_restarted_schedule_and_batched_nodes():
    from test_stage_b_config import valid_config_dict

    estimate = estimate_stage_b_run(StageBConfig.from_dict(valid_config_dict()))
    # schedule 16/32/64 with chunk 8: 2+4+8 backward batches, plus two
    # endpoint forwards once per attribution. Five blocks run main/repeat/sanity,
    # with one direct and perturbed-direct forward and two parities per block.
    assert estimate["compute"]["worst_case_ig_steps_per_attribution"] == 112
    assert estimate["compute"]["backward_calls_per_attribution"] == 14
    assert estimate["compute"]["forward_calls_per_attribution"] == 16
    assert estimate["compute"]["estimated_forward_calls"] == 252
    assert estimate["compute"]["estimated_backward_calls"] == 210
