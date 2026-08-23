from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import vitnns.stage_c_class_logit as class_logit
from vitnns.attribution import IGResult
from vitnns.modeling import load_canonical_model
from vitnns.retention import ArtifactKind
from vitnns.stage_c_class_logit import (
    EMBED_DIM,
    IG_ATOL,
    IG_RTOL,
    IG_STEPS,
    IG_STEP_CHUNK_SIZE,
    METHOD_VERSION,
    FrozenClassLogitTarget,
    FrozenScalarEvaluation,
    StageCClassLogitError,
    StageCClassLogitIGConvergenceError,
    evaluate_frozen_class_logit_target,
    full_model_class_logit_ig,
    full_model_class_logit_ig_at_intervention,
    resolve_clean_class_logit_target,
    resolve_natural_prediction_observation,
    vit_grad_cam_block11_norm1,
    vit_grad_cam_block11_norm1_at_intervention,
    vit_grad_cam_patch_map,
)
from vitnns.stage_c_fidelity import (
    ClassLogitScalarAdapter, resolve_target_response,
)
from vitnns.stage_c_integrity import (
    ModelIntegrityBoundary,
    ModelIntegrityError,
    StrongModelSnapshot,
)
from vitnns.stage_c_maps import ComparisonLine, MethodResult, SignSemantics
from vitnns.targets import TargetSpec


PATCH_SHAPE = (14, 14)
TARGET_REF = "clean_predicted_class_raw_logit"


class _ToyNorm(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens


class _RegistrationFailNorm(_ToyNorm):
    def register_forward_hook(self, *args: Any, **kwargs: Any):
        raise RuntimeError("synthetic target-layer hook registration failure")


class _ToyBlock(nn.Module):
    def __init__(self, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.norm1 = _ToyNorm() if norm is None else norm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm1(tokens)


class _ToyClassModel(nn.Module):
    """Cheap 12-block/197-token model with target-sensitive class maps."""

    def __init__(self, *, registration_failure: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.unused = nn.Parameter(
            torch.tensor(2.0, dtype=torch.float32), requires_grad=False
        )
        blocks = [_ToyBlock() for _ in range(11)]
        final_norm: nn.Module = (
            _RegistrationFailNorm() if registration_failure else _ToyNorm()
        )
        blocks.append(_ToyBlock(final_norm))
        self.blocks = nn.ModuleList(blocks)
        self.forward_calls = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        patch_rgb = F.avg_pool2d(image[:, :2], kernel_size=16, stride=16)
        patch_channels = patch_rgb.flatten(2).transpose(1, 2) * self.scale
        padding = torch.zeros(
            image.shape[0],
            196,
            EMBED_DIM - 2,
            dtype=image.dtype,
            device=image.device,
        )
        patch_tokens = torch.cat((patch_channels, padding), dim=-1)
        cls = torch.zeros(
            image.shape[0], 1, EMBED_DIM, dtype=image.dtype, device=image.device
        )
        tokens = torch.cat((cls, patch_tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        patches = tokens[:, 1:, :]
        class_zero = (2.0 * patches[:, :, 0] - 0.5 * patches[:, :, 1]).mean(
            dim=1
        ) + 0.5
        class_one = (-0.5 * patches[:, :, 0] + 2.0 * patches[:, :, 1]).mean(
            dim=1
        )
        class_two = -(patches[:, :, 0] + patches[:, :, 1]).mean(dim=1) - 0.5
        return torch.stack((class_zero, class_one, class_two), dim=1)


def _clean_image() -> torch.Tensor:
    patch_zero = torch.linspace(0.4, 1.2, 196, dtype=torch.float32).reshape(14, 14)
    patch_one = torch.linspace(0.7, 0.1, 196, dtype=torch.float32).reshape(14, 14)
    channel_zero = patch_zero.repeat_interleave(16, 0).repeat_interleave(16, 1)
    channel_one = patch_one.repeat_interleave(16, 0).repeat_interleave(16, 1)
    channel_two = torch.zeros((224, 224), dtype=torch.float32)
    return torch.stack((channel_zero, channel_one, channel_two)).unsqueeze(0)


def _hook_inventory(model: nn.Module) -> tuple[tuple[tuple[str, int, int], ...], ...]:
    inventories = []
    for module in model.modules():
        entries: list[tuple[str, int, int]] = []
        for name in (
            "_forward_pre_hooks",
            "_forward_hooks",
            "_backward_pre_hooks",
            "_backward_hooks",
        ):
            hooks = getattr(module, name)
            entries.extend((name, int(key), id(value)) for key, value in hooks.items())
        inventories.append(tuple(entries))
    return tuple(inventories)


def _mode_flags(model: nn.Module) -> tuple[bool, ...]:
    return tuple(module.training for module in model.modules())


def _requires_grad_flags(model: nn.Module) -> tuple[bool, ...]:
    return tuple(parameter.requires_grad for parameter in model.parameters())


def _grad_snapshot(
    model: nn.Module,
) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], ...]:
    return tuple(
        (parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone())
        for parameter in model.parameters()
    )


def _assert_grad_snapshot(
    model: nn.Module,
    snapshot: tuple[tuple[torch.Tensor | None, torch.Tensor | None], ...],
) -> None:
    for parameter, (original, values) in zip(model.parameters(), snapshot):
        assert parameter.grad is original
        if original is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, values, atol=0.0, rtol=0.0)


def _mixed_caller_state(model: _ToyClassModel) -> None:
    model.eval()
    model.blocks[0].train()
    model.scale.requires_grad_(True)
    model.unused.requires_grad_(False)
    model.scale.grad = torch.tensor(7.25, dtype=torch.float32)
    model.unused.grad = None


def _json_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_thaw(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _assert_common_method_result(
    result: MethodResult,
    *,
    method_id: str,
    projection: str,
    sign_semantics: SignSemantics,
) -> None:
    assert isinstance(result, MethodResult)
    assert result.method_id == method_id
    assert result.method_version == METHOD_VERSION == "1.0.0"
    assert result.comparison_line is ComparisonLine.CLASS_LOGIT
    assert result.explanation_target == TARGET_REF
    assert result.fidelity_target == TARGET_REF
    assert result.class_agnostic is False
    assert result.projection == projection
    assert result.sign_semantics is sign_semantics
    assert result.na is None
    assert result.native_tensor is not None
    assert result.native_tensor.kind is ArtifactKind.RAW
    assert result.metric_map is not None
    assert result.metric_map.kind is ArtifactKind.METRIC_READY
    assert result.metric_map.shape == PATCH_SHAPE
    assert result.metric_map.dtype == np.dtype(np.float32)
    assert np.isfinite(result.metric_map.data).all()
    assert np.count_nonzero(result.metric_map.data) > 0
    assert result.valid_mask.dtype == np.bool_
    assert result.valid_mask.shape == PATCH_SHAPE
    assert bool(result.valid_mask.all())
    assert result.provenance["stage"] == "C3"
    assert result.provenance["method_implementation_version"] == METHOD_VERSION
    assert result.provenance["comparison_line"] == "class_logit"
    assert result.provenance["class_agnostic"] is False
    assert result.provenance["target"]["class_index_source"] == (
        "single_clean_input_argmax"
    )
    assert result.provenance["target"]["exclude_misclassified"] is False
    json.dumps(_json_thaw(result.provenance), allow_nan=False)
    with pytest.raises(TypeError):
        result.provenance["tamper"] = True


def test_resolve_clean_target_argmax_once_records_wrong_truth_without_exclusion():
    model = _ToyClassModel()
    image = _clean_image()
    _mixed_caller_state(model)
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)

    target = resolve_clean_class_logit_target(
        model, image, official_truth_index=1
    )

    assert model.forward_calls == 1
    assert target.class_index == 0
    assert target.target_spec == TargetSpec.class_logit(0)
    assert target.class_count == 3
    assert target.clean_raw_logit == pytest.approx(1.9, abs=2e-6)
    assert target.official_truth_index == 1
    assert target.clean_top1_correct is False
    assert target.to_dict()["exclude_misclassified"] is False
    assert target.to_dict()["freeze_class_index_for_all_interventions"] is True
    assert len(target.clean_input_sha256) == 64
    assert _hook_inventory(model) == hooks_before
    assert _mode_flags(model) == modes_before
    assert _requires_grad_flags(model) == requires_before
    _assert_grad_snapshot(model, grads_before)
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.class_index = 1


def test_frozen_class_scalar_uses_fixed_index_and_restores_caller_lifecycle():
    model = _ToyClassModel()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    intervention = torch.zeros_like(image)
    intervention[:, 1] = 2.0
    assert torch.argmax(model(intervention)[0]).item() == 1
    _mixed_caller_state(model)
    persistent = model.blocks[2].register_forward_hook(
        lambda _module, _arguments, _output: None
    )
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)
    try:
        result = evaluate_frozen_class_logit_target(
            model, image, intervention, target
        )
        assert isinstance(result, FrozenScalarEvaluation)
        assert result.value == pytest.approx(-0.5)
        assert result.class_index == target.class_index == 0
        assert result.target_kind == TARGET_REF
        assert result.clean_input_sha256 == target.clean_input_sha256
        assert result.intervention_input_sha256 == class_logit._tensor_sha256(
            intervention
        )
        assert result.runtime_model_binding_sha256 == (
            target.runtime_model_binding_sha256
        )
        json.dumps(result.to_dict(), allow_nan=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.value = 0.0
        with pytest.raises(TypeError, match="only be issued"):
            FrozenScalarEvaluation()
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
    finally:
        persistent.remove()


def test_frozen_class_scalar_prehashed_path_skips_tensor_content_hashes(monkeypatch):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    intervention = image * 0.5
    intervention_digest = class_logit._tensor_sha256(intervention)

    def unexpected_hash(_value):
        raise AssertionError("prehashed scalar path must not hash tensors")

    monkeypatch.setattr(class_logit, "_tensor_sha256", unexpected_hash)
    result = evaluate_frozen_class_logit_target(
        model, image, intervention, target,
        clean_input_sha256=target.clean_input_sha256,
        intervention_input_sha256=intervention_digest,
    )

    assert result.intervention_input_sha256 == intervention_digest


@pytest.mark.parametrize(
    "name,value",
    [
        ("clean_input_sha256", "A" * 64),
        ("intervention_input_sha256", "not-a-digest"),
    ],
)
def test_frozen_class_scalar_rejects_invalid_prebound_digest(name, value):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    kwargs = {
        "clean_input_sha256": target.clean_input_sha256,
        "intervention_input_sha256": target.clean_input_sha256,
    }
    kwargs[name] = value
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        evaluate_frozen_class_logit_target(
            model, image, image, target, **kwargs
        )


def test_frozen_target_requires_resolver_clean_tensor_identity_and_live_state():
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)

    with pytest.raises(StageCClassLogitError, match="clean input object"):
        evaluate_frozen_class_logit_target(
            model, image.clone(), image, target,
            clean_input_sha256=target.clean_input_sha256,
            intervention_input_sha256=target.clean_input_sha256,
        )

    image.add_(0.0)
    with pytest.raises(StageCClassLogitError, match="clean input live state"):
        evaluate_frozen_class_logit_target(
            model, image, image, target,
            clean_input_sha256=target.clean_input_sha256,
            intervention_input_sha256=target.clean_input_sha256,
        )


def test_natural_prediction_observation_binds_model_and_intervention_hash():
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    intervention = torch.zeros_like(image)
    intervention[:, 1] = 2.0
    observed = resolve_natural_prediction_observation(
        model, image, intervention, target
    )
    assert observed["class_index"] == 1
    assert observed["class_count"] == target.class_count
    assert observed["intervention_input_sha256"] == class_logit._tensor_sha256(
        intervention
    )
    assert observed["runtime_model_binding_sha256"] == (
        target.runtime_model_binding_sha256
    )


def test_c5_class_adapter_resolves_endpoints_without_reselecting_class():
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    adapter = ClassLogitScalarAdapter(model, image, target)
    response = resolve_target_response(adapter, image, torch.zeros_like(image))
    assert response.comparison_line is ComparisonLine.CLASS_LOGIT
    assert response.target_identity["clean_input_sha256"] == (
        target.clean_input_sha256
    )
    assert response.target_identity["frozen_target"]["class_index"] == (
        target.class_index
    )
    assert response.clean_value == pytest.approx(target.clean_raw_logit)
    assert response.normalize(response.clean_value) == pytest.approx(1.0)
    assert response.normalize(response.baseline_value) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("make_bad", "exception", "message"),
    [
        (lambda image: image.to(torch.float64), TypeError, "float32"),
        (lambda image: image[:, :, :-1], StageCClassLogitError, "shape"),
        (
            lambda image: image.clone().masked_fill(
                torch.zeros_like(image, dtype=torch.bool).index_fill(3, torch.tensor([0]), True),
                float("nan"),
            ),
            StageCClassLogitError,
            "finite",
        ),
    ],
)
def test_frozen_class_scalar_rejects_invalid_intervention(
    make_bad, exception, message
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    with pytest.raises(exception, match=message):
        evaluate_frozen_class_logit_target(
            model, image, make_bad(image), target
        )


def test_frozen_class_scalar_exception_restores_state_and_rejects_context():
    model = _ToyClassModel()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    changed_clean = image.clone()
    changed_clean[0, 0, 0, 0] += 1.0
    with pytest.raises(StageCClassLogitError, match="object"):
        evaluate_frozen_class_logit_target(model, changed_clean, image, target)
    with pytest.raises(StageCClassLogitError, match="model instance"):
        evaluate_frozen_class_logit_target(
            _ToyClassModel(), image, image, target
        )

    _mixed_caller_state(model)
    failing = model.register_forward_pre_hook(
        lambda _module, _arguments: (_ for _ in ()).throw(
            RuntimeError("synthetic scalar failure")
        )
    )
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)
    try:
        with pytest.raises(RuntimeError, match="synthetic scalar failure"):
            evaluate_frozen_class_logit_target(model, image, image, target)
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
    finally:
        failing.remove()


@pytest.mark.parametrize(
    ("truth", "exception", "message"),
    [
        (True, TypeError, "integer.*not bool"),
        (-1, StageCClassLogitError, "non-negative"),
        (3, StageCClassLogitError, "outside 3 classes"),
    ],
)
def test_target_resolution_rejects_invalid_official_truth(truth, exception, message):
    with pytest.raises(exception, match=message):
        resolve_clean_class_logit_target(
            _ToyClassModel().eval(), _clean_image(), official_truth_index=truth
        )


def test_frozen_target_rejects_direct_construction_other_model_and_weight_drift():
    model = _ToyClassModel().eval()
    other = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)

    with pytest.raises(TypeError, match="cannot be constructed directly"):
        FrozenClassLogitTarget(
            class_index=target.class_index,
            clean_raw_logit=target.clean_raw_logit,
            class_count=target.class_count,
            clean_input_sha256=target.clean_input_sha256,
            clean_input_shape=target.clean_input_shape,
        )
    with pytest.raises(TypeError, match="only be issued"):
        FrozenClassLogitTarget._create(
            model=model,
            clean_inputs=image,
            class_index=1,
            clean_raw_logit=-999.0,
            class_count=target.class_count,
            clean_input_sha256=target.clean_input_sha256,
            clean_input_shape=target.clean_input_shape,
            official_truth_index=None,
            clean_top1_correct=None,
            issuer_secret=b"forged",
        )
    with pytest.raises(StageCClassLogitError, match="model instance"):
        vit_grad_cam_block11_norm1(other, image, target)
    with torch.no_grad():
        model.scale.add_(0.125)
    with pytest.raises(StageCClassLogitError, match="state changed"):
        vit_grad_cam_block11_norm1(model, image, target)

    data_mutated = _ToyClassModel().eval()
    load_snapshot = StrongModelSnapshot.capture(data_mutated)
    with pytest.raises(ModelIntegrityError, match="integrity verification failed"):
        with ModelIntegrityBoundary(
            data_mutated, reference_snapshot=load_snapshot
        ):
            data_mutated.scale.data.add_(0.25)


def test_multiple_resolver_issued_targets_remain_independently_valid():
    model = _ToyClassModel().eval()
    image = _clean_image()
    first = resolve_clean_class_logit_target(model, image)
    second = resolve_clean_class_logit_target(model, image)

    assert first is not second
    assert first != second
    first_result = vit_grad_cam_block11_norm1(model, image, first)
    second_result = vit_grad_cam_block11_norm1(model, image, second)
    np.testing.assert_array_equal(
        first_result.metric_map.data, second_result.metric_map.data
    )


@pytest.mark.parametrize(
    ("field_name", "mutated_value"),
    [
        ("class_index", 1),
        ("clean_raw_logit", -123.0),
        ("class_count", 4),
        ("official_truth_index", 1),
        ("clean_top1_correct", False),
    ],
)
def test_frozen_target_rejects_low_level_field_tampering(
    field_name, mutated_value
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    object.__setattr__(target, field_name, mutated_value)

    with pytest.raises(StageCClassLogitError, match="mutated after issue"):
        vit_grad_cam_block11_norm1(model, image, target)


def test_target_resolver_rejects_model_mutation_during_clean_forward():
    model = _ToyClassModel().eval()
    image = _clean_image()

    def mutate_parameter(_module, _args, _output):
        model.scale.add_(0.5)

    handle = model.register_forward_hook(mutate_parameter)
    try:
        with pytest.raises(StageCClassLogitError, match="changed during.*resolution"):
            resolve_clean_class_logit_target(model, image)
        assert handle.id in model._forward_hooks
    finally:
        handle.remove()


@pytest.mark.parametrize(
    ("transform", "exception", "message"),
    [
        (lambda value: value.repeat(2, 1, 1, 1), StageCClassLogitError, "shape"),
        (lambda value: value[:, :, :-1, :], StageCClassLogitError, "shape"),
        (lambda value: value.double(), TypeError, "float32"),
        (lambda value: value.long(), TypeError, "float32"),
        (
            lambda value: value.clone().index_put_(
                (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                torch.tensor(float("nan")),
            ),
            StageCClassLogitError,
            "finite",
        ),
    ],
)
def test_target_resolution_rejects_bad_input_geometry_dtype_and_values(
    transform, exception, message
):
    with pytest.raises(exception, match=message):
        resolve_clean_class_logit_target(
            _ToyClassModel().eval(), transform(_clean_image())
        )


def test_grad_cam_core_hand_computes_cls_exclusion_196_mean_relu_and_no_minmax():
    activation = torch.zeros((1, 197, 3), dtype=torch.float32)
    gradient = torch.zeros_like(activation)
    patch_index = torch.arange(1, 197, dtype=torch.float32)
    activation[0, 1:, 0] = patch_index
    activation[0, 1:, 1] = 2.0
    activation[0, 1:, 2] = -3.0
    gradient[0, 1:, 0] = torch.linspace(-1.0, 3.0, 196)
    gradient[0, 1:, 1] = -4.0
    gradient[0, 1:, 2] = 1.0
    # These enormous CLS values would dominate either a 197-token gradient
    # mean or a weighted sum that failed to remove CLS.
    activation[0, 0] = torch.tensor([1e8, -1e8, 1e8])
    gradient[0, 0] = torch.tensor([1e8, 1e8, -1e8])

    actual = vit_grad_cam_patch_map(activation, gradient)

    weights = gradient[:, 1:, :].mean(dim=1)
    expected = torch.relu(
        (activation[:, 1:, :] * weights[:, None, :]).sum(dim=-1)
    ).reshape(14, 14)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    wrong_with_cls = torch.relu(
        (activation[:, 1:, :] * gradient.mean(dim=1)[:, None, :]).sum(dim=-1)
    ).reshape(14, 14)
    assert not torch.equal(actual, wrong_with_cls)
    assert actual.dtype is torch.float32
    assert actual.shape == PATCH_SHAPE
    assert bool((actual >= 0).all())
    assert bool((actual == 0).any())  # ReLU is observable.
    assert float(actual.max().item()) > 100.0  # No min-max normalization.
    assert float(actual.max().item()) != pytest.approx(1.0)


@pytest.mark.parametrize(
    ("activation", "gradient", "exception", "message"),
    [
        (torch.zeros((197, 2)), torch.zeros((197, 2)), StageCClassLogitError, "shape"),
        (
            torch.zeros((2, 197, 2)),
            torch.zeros((2, 197, 2)),
            StageCClassLogitError,
            "shape",
        ),
        (
            torch.zeros((1, 196, 2)),
            torch.zeros((1, 196, 2)),
            StageCClassLogitError,
            "shape",
        ),
        (
            torch.zeros((1, 197, 2), dtype=torch.float64),
            torch.zeros((1, 197, 2), dtype=torch.float64),
            TypeError,
            "float32",
        ),
        (
            torch.zeros((1, 197, 2)),
            torch.zeros((1, 197, 3)),
            StageCClassLogitError,
            "identical",
        ),
        (
            torch.full((1, 197, 2), float("nan")),
            torch.zeros((1, 197, 2)),
            StageCClassLogitError,
            "non-finite",
        ),
        (
            torch.ones((1, 197, 2)),
            torch.zeros((1, 197, 2)),
            StageCClassLogitError,
            "all zero",
        ),
    ],
)
def test_grad_cam_core_fails_closed_on_invalid_or_zero_output(
    activation, gradient, exception, message
):
    with pytest.raises(exception, match=message):
        vit_grad_cam_patch_map(activation, gradient)


def test_grad_cam_wrapper_target_sensitivity_repeatability_and_method_contract():
    model = _ToyClassModel().eval()
    image = _clean_image()
    predicted = resolve_clean_class_logit_target(model, image)
    caller_context = {"sample_id": "toy", "nested": {"values": [1, 2]}}

    first = vit_grad_cam_block11_norm1(
        model, image, predicted, provenance=caller_context
    )
    repeated = vit_grad_cam_block11_norm1(model, image, predicted)

    _assert_common_method_result(
        first,
        method_id="vit_grad_cam_block11_norm1",
        projection="identity_14x14",
        sign_semantics=SignSemantics.NONNEGATIVE,
    )
    assert first.native_tensor.shape == PATCH_SHAPE
    np.testing.assert_array_equal(first.native_tensor.data, first.metric_map.data)
    np.testing.assert_array_equal(first.metric_map.data, repeated.metric_map.data)
    assert np.all(first.metric_map.data >= 0)
    algorithm = first.provenance["algorithm"]
    assert algorithm["target_layer"] == "blocks[11].norm1"
    assert algorithm["exclude_cls_token"] is True
    assert algorithm["patch_tokens"] == 196
    assert algorithm["channel_weight"] == (
        "arithmetic_mean_target_gradient_over_196_patch_tokens"
    )
    assert algorithm["activation"] == "relu"
    assert algorithm["minmax_normalization"] is False
    assert algorithm["smoothing"] is False
    assert algorithm["percentile_enhancement"] is False
    assert algorithm["display_resize"] is False
    assert first.provenance["target"]["class_index"] == 0
    assert first.provenance["intervention_input_sha256"] == (
        class_logit._tensor_sha256(image)
    )

    caller_context["nested"]["values"].append(3)
    assert first.provenance["caller_context"]["nested"]["values"] == (1, 2)


def test_grad_cam_at_intervention_uses_fixed_class_and_maps_z():
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    intervention = image.clone()
    intervention[:, 0] *= 0.5
    intervention[:, 1] += 1.0
    assert torch.argmax(model(intervention)[0]).item() == 1

    clean_result = vit_grad_cam_block11_norm1(model, image, target)
    result = vit_grad_cam_block11_norm1_at_intervention(
        model, image, intervention, target
    )

    assert result.provenance["target"]["class_index"] == target.class_index == 0
    assert result.provenance["intervention_input_sha256"] == (
        class_logit._tensor_sha256(intervention)
    )
    assert not np.array_equal(result.metric_map.data, clean_result.metric_map.data)


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros((1, 3, 223, 224), dtype=torch.float32),
        torch.zeros((1, 3, 224, 224), dtype=torch.float64),
        torch.full((1, 3, 224, 224), float("nan"), dtype=torch.float32),
    ],
)
def test_grad_cam_at_intervention_rejects_invalid_z(bad):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    with pytest.raises((TypeError, StageCClassLogitError)):
        vit_grad_cam_block11_norm1_at_intervention(model, image, bad, target)


@pytest.mark.parametrize("should_raise", [False, True], ids=["success", "failure"])
def test_grad_cam_at_intervention_restores_lifecycle(should_raise):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    intervention = image * 0.75
    _mixed_caller_state(model)

    def caller_hook(_module, _arguments, _output):
        model.blocks[4].train()
        model.scale.requires_grad_(True)
        model.scale.grad.add_(-50.0)
        if should_raise:
            raise RuntimeError("synthetic intervention Grad-CAM failure")

    handle = model.register_forward_hook(caller_hook)
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)
    try:
        if should_raise:
            with pytest.raises(RuntimeError, match="intervention Grad-CAM failure"):
                vit_grad_cam_block11_norm1_at_intervention(
                    model, image, intervention, target
                )
        else:
            result = vit_grad_cam_block11_norm1_at_intervention(
                model, image, intervention, target
            )
            assert np.count_nonzero(result.metric_map.data) > 0
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
    finally:
        handle.remove()


def test_grad_cam_exception_detects_caller_intervention_mutation():
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    intervention = image.clone()

    def hostile(_module, _arguments, _output):
        intervention.add_(1.0)
        raise RuntimeError("hostile hook")

    handle = model.register_forward_hook(hostile)
    try:
        with pytest.raises(StageCClassLogitError, match="mutated caller"):
            vit_grad_cam_block11_norm1_at_intervention(
                model, image, intervention, target
            )
    finally:
        handle.remove()


def test_grad_cam_is_target_sensitive_on_the_same_activation():
    model = _ToyClassModel().eval()
    image = _clean_image()
    with class_logit._preserve_model_attribution_state(
        model, freeze_parameters=True
    ), class_logit._ActivationCapture(
        model.blocks[11].norm1
    ) as capture, torch.enable_grad():
        probe = image.detach().clone().requires_grad_(True)
        logits = model(probe)
        activation = capture.activation
        assert activation is not None and capture.call_count == 1
        first_gradient = torch.autograd.grad(
            logits[:, 0].sum(), activation, retain_graph=True
        )[0]
        second_gradient = torch.autograd.grad(
            logits[:, 1].sum(), activation
        )[0]
        first = vit_grad_cam_patch_map(activation, first_gradient)
        second = vit_grad_cam_patch_map(activation, second_gradient)

    assert not torch.allclose(first, second, atol=1e-6, rtol=1e-5)


def test_grad_cam_rejects_input_or_target_drift_and_inference_mode():
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    changed = image.clone()
    changed[0, 0, 0, 0] += 1.0
    with pytest.raises(StageCClassLogitError, match="object.*frozen target"):
        evaluate_frozen_class_logit_target(model, changed, image, target)
    with pytest.raises(TypeError, match="frozen_target"):
        vit_grad_cam_block11_norm1(model, image, target.target_spec)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="inference_mode"):
            vit_grad_cam_block11_norm1(model, image, target)


def test_grad_cam_preserves_existing_hooks_mixed_modes_requires_grad_and_grads():
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    _mixed_caller_state(model)
    caller_hook = model.blocks[11].norm1.register_forward_hook(
        lambda _module, _args, _output: None
    )
    try:
        hooks_before = _hook_inventory(model)
        modes_before = _mode_flags(model)
        requires_before = _requires_grad_flags(model)
        grads_before = _grad_snapshot(model)
        grad_version_before = model.scale.grad._version

        result = vit_grad_cam_block11_norm1(model, image, target)

        assert np.count_nonzero(result.metric_map.data) > 0
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert model.scale.grad._version == grad_version_before
        assert caller_hook.id in model.blocks[11].norm1._forward_hooks
    finally:
        caller_hook.remove()


def test_grad_cam_restores_lifecycle_when_existing_forward_hook_raises():
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    _mixed_caller_state(model)

    def mutate_and_fail(_module, _args, _output):
        model.blocks[2].train(False)
        model.scale.requires_grad_(True)
        model.scale.grad.add_(-100.0)
        raise RuntimeError("synthetic caller forward failure")

    caller_hook = model.register_forward_hook(mutate_and_fail)
    try:
        hooks_before = _hook_inventory(model)
        modes_before = _mode_flags(model)
        requires_before = _requires_grad_flags(model)
        grads_before = _grad_snapshot(model)
        with pytest.raises(RuntimeError, match="caller forward failure"):
            vit_grad_cam_block11_norm1(model, image, target)
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert caller_hook.id in model._forward_hooks
    finally:
        caller_hook.remove()


def test_grad_cam_restores_lifecycle_when_hook_registration_raises():
    model = _ToyClassModel(registration_failure=True).eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    _mixed_caller_state(model)
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)

    with pytest.raises(RuntimeError, match="hook registration failure"):
        vit_grad_cam_block11_norm1(model, image, target)

    assert _hook_inventory(model) == hooks_before
    assert _mode_flags(model) == modes_before
    assert _requires_grad_flags(model) == requires_before
    _assert_grad_snapshot(model, grads_before)


def test_resolver_and_grad_cam_preserve_caller_inference_grad_without_writes():
    model = _ToyClassModel().eval()
    image = _clean_image()
    with torch.inference_mode():
        inference_grad = torch.tensor(4.5)
    model.scale.grad = inference_grad

    target = resolve_clean_class_logit_target(model, image)
    result = vit_grad_cam_block11_norm1(model, image, target)

    assert np.count_nonzero(result.metric_map.data) > 0
    assert model.scale.grad is inference_grad
    assert model.scale.grad.is_inference()
    assert float(model.scale.grad.item()) == pytest.approx(4.5)


def _signed_native() -> torch.Tensor:
    native = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    native[0, 0, :16, :16] = 1.0
    native[0, 1, :16, :16] = -0.25
    native[0, 2, 16:32, :16] = -2.0
    return native


def _ig_result(
    target: FrozenClassLogitTarget,
    *,
    attribution: torch.Tensor | None = None,
    steps: int = 512,
    converged: bool = True,
    pass_value: bool = True,
    metadata_updates: Mapping[str, Any] | None = None,
    attempt_steps: tuple[int, ...] | None = None,
) -> IGResult:
    native = _signed_native() if attribution is None else attribution
    finite_sum = (
        float(native.detach().to(torch.float32).sum().item())
        if native.numel() and bool(torch.isfinite(native.detach()).all().item())
        else 0.0
    )
    conservation = {
        "delta": torch.tensor([finite_sum], dtype=torch.float32),
        "sum": torch.tensor([finite_sum], dtype=torch.float32),
        "abs": torch.tensor([0.0], dtype=torch.float32),
        "rel": torch.tensor([0.0], dtype=torch.float32),
        "pass": torch.tensor([pass_value], dtype=torch.bool),
    }
    if attempt_steps is None:
        attempt_steps = IG_STEPS[: IG_STEPS.index(steps) + 1]
    attempts = tuple(
        {
            "steps": step,
            "delta": [finite_sum],
            "sum": [finite_sum],
            "abs": [0.0],
            "rel": [0.0],
            "pass": [pass_value if step == steps else False],
        }
        for step in attempt_steps
    )
    metadata: dict[str, Any] = {
        "method": "integrated_gradients",
        "quadrature": "gauss_legendre",
        "integration_dtype": "float32",
        "input_shape": [1, 3, 224, 224],
        "step_schedule": list(IG_STEPS),
        "step_chunk_size": IG_STEP_CHUNK_SIZE,
        "atol": IG_ATOL,
        "rtol": IG_RTOL,
        "scope": "full_model",
        "target": target.target_spec.to_dict(),
        "includes_residual_paths": True,
    }
    metadata.update(dict(metadata_updates or {}))
    return IGResult(
        attribution=native,
        conservation=conservation,
        steps=steps,
        converged=converged,
        attempts=attempts,
        metadata=metadata,
    )


def _nonconverged_ig_result(
    target: FrozenClassLogitTarget,
    *,
    attribution: torch.Tensor | None = None,
    report_delta: float = 0.0,
) -> IGResult:
    native = _signed_native() if attribution is None else attribution
    report_sum = native.to(torch.float32).flatten(1).sum(dim=1)
    delta = torch.tensor([report_delta], dtype=torch.float32)
    absolute = (delta - report_sum).abs()
    relative = absolute / delta.abs().clamp_min(torch.finfo(torch.float32).eps)
    passes = absolute <= (IG_ATOL + IG_RTOL * delta.abs())
    assert not bool(passes.item())
    conservation = {
        "delta": delta,
        "sum": report_sum,
        "abs": absolute,
        "rel": relative,
        "pass": passes,
    }
    attempts = tuple(
        {
            "steps": step,
            "delta": delta.tolist(),
            "sum": report_sum.tolist(),
            "abs": absolute.tolist(),
            "rel": relative.tolist(),
            "pass": passes.tolist(),
        }
        for step in IG_STEPS
    )
    return IGResult(
        attribution=native,
        conservation=conservation,
        steps=IG_STEPS[-1],
        converged=False,
        attempts=attempts,
        metadata=dict(_ig_result(target).metadata),
    )


def test_full_model_ig_dispatches_exact_frozen_contract_and_projects_signed_rgb(
    monkeypatch,
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image, official_truth_index=0)
    calls: dict[str, Any] = {}

    def fake_full_model_ig(
        received_model, received_inputs, received_baseline, received_target, **kwargs
    ):
        calls.update(
            {
                "model": received_model,
                "inputs": received_inputs,
                "baseline": received_baseline.detach().clone(),
                "target": received_target,
                "kwargs": kwargs,
                "modes": _mode_flags(received_model),
                "requires_grad": _requires_grad_flags(received_model),
            }
        )
        return _ig_result(target)

    monkeypatch.setattr(class_logit.attribution, "full_model_ig", fake_full_model_ig)
    result = full_model_class_logit_ig(
        model, image, target, provenance={"sample_id": "ig-toy"}
    )

    assert calls["model"] is model
    assert calls["inputs"] is image
    torch.testing.assert_close(calls["baseline"], torch.zeros_like(image))
    assert calls["target"] == TargetSpec.class_logit(target.class_index)
    assert calls["kwargs"] == {
        "steps": IG_STEPS,
        "step_chunk_size": 8,
        "atol": 1e-4,
        "rtol": 0.01,
    }
    assert not any(calls["modes"])
    assert not any(calls["requires_grad"])
    _assert_common_method_result(
        result,
        method_id="full_model_class_logit_ig",
        projection="pixel_mass_to_patch_sum",
        sign_semantics=SignSemantics.SIGNED,
    )
    expected_native = _signed_native()[0].numpy()
    np.testing.assert_array_equal(result.native_tensor.data, expected_native)
    expected_pixels = expected_native.sum(axis=0, dtype=np.float64).astype(np.float32)
    expected_patches = expected_pixels.reshape(14, 16, 14, 16).sum(
        axis=(1, 3), dtype=np.float64
    ).astype(np.float32)
    np.testing.assert_array_equal(result.metric_map.data, expected_patches)
    assert result.metric_map.data[0, 0] == pytest.approx(192.0)
    assert result.metric_map.data[1, 0] == pytest.approx(-512.0)
    assert np.any(result.metric_map.data < 0)
    algorithm = result.provenance["algorithm"]
    assert result.provenance["exact_method_name"] == "full-model class-logit IG"
    assert algorithm["model_scope"] == "real_unmodified_full_model"
    assert algorithm["baseline"] == {
        "id": "zero_normalized",
        "normalized_value": 0.0,
        "canonical_rgb_value": 0.5,
        "is_black_image": False,
    }
    assert algorithm["steps_schedule"] == IG_STEPS
    assert algorithm["step_chunk_size"] == 8
    assert algorithm["absolute_value_before_projection"] is False
    assert algorithm["native_attribution"] == "signed_rgb_float32_3x224x224"
    assert result.provenance["ig_result"]["selected_steps"] == 512
    assert result.provenance.get("intervention_input_sha256") is None


def test_full_model_ig_at_intervention_uses_z_delta_fixed_class_and_hash(
    monkeypatch,
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    intervention = image.clone()
    intervention[:, 0] *= 0.25
    intervention[:, 1] += 2.0
    assert torch.argmax(model(intervention)[0]).item() == 1
    expected_delta = float(
        (
            model(intervention)[0, target.class_index]
            - model(torch.zeros_like(intervention))[0, target.class_index]
        ).item()
    )
    calls: dict[str, Any] = {}

    def fake(received_model, received_inputs, baseline, received_target, **kwargs):
        calls["inputs"] = received_inputs
        calls["target"] = received_target
        native = torch.zeros_like(received_inputs)
        native[0, 0, 0, 0] = expected_delta
        result = _ig_result(target, attribution=native)
        result.conservation["delta"][0] = expected_delta
        result.conservation["sum"][0] = expected_delta
        for attempt in result.attempts:
            attempt["delta"][0] = expected_delta
            attempt["sum"][0] = expected_delta
        return result

    monkeypatch.setattr(class_logit.attribution, "full_model_ig", fake)
    result = full_model_class_logit_ig_at_intervention(
        model, image, intervention, target
    )
    assert calls["inputs"] is not intervention
    torch.testing.assert_close(calls["inputs"], intervention)
    assert calls["target"] == TargetSpec.class_logit(target.class_index)
    assert result.provenance["target"]["class_index"] == target.class_index
    assert result.provenance["intervention_input_sha256"] == (
        class_logit._tensor_sha256(intervention)
    )
    assert result.provenance["ig_result"]["conservation"]["delta"] == (
        pytest.approx(expected_delta),
    )


def test_full_model_ig_at_intervention_rejects_wrong_delta_and_restores(
    monkeypatch,
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    intervention = image * 0.5
    _mixed_caller_state(model)
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)
    monkeypatch.setattr(
        class_logit.attribution, "full_model_ig",
        lambda *_args, **_kwargs: _ig_result(target),
    )
    with pytest.raises(StageCClassLogitError, match=r"t\(z\)-t\(x0\)"):
        full_model_class_logit_ig_at_intervention(
            model, image, intervention, target
        )
    assert _hook_inventory(model) == hooks_before
    assert _mode_flags(model) == modes_before
    assert _requires_grad_flags(model) == requires_before
    _assert_grad_snapshot(model, grads_before)


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros((1, 3, 223, 224), dtype=torch.float32),
        torch.zeros((1, 3, 224, 224), dtype=torch.float64),
        torch.full((1, 3, 224, 224), float("inf"), dtype=torch.float32),
    ],
)
def test_full_model_ig_at_intervention_rejects_invalid_z(bad):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    with pytest.raises((TypeError, StageCClassLogitError)):
        full_model_class_logit_ig_at_intervention(model, image, bad, target)


def test_full_model_ig_nonconvergence_retains_validated_complete_evidence(
    monkeypatch,
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    nonconverged = _nonconverged_ig_result(target)
    monkeypatch.setattr(
        class_logit.attribution,
        "full_model_ig",
        lambda *_args, **_kwargs: nonconverged,
    )
    calls_before = model.forward_calls

    with pytest.raises(StageCClassLogitIGConvergenceError) as captured:
        full_model_class_logit_ig(model, image, target)

    assert captured.value.ig_result is nonconverged
    assert captured.value.result is nonconverged
    assert captured.value.attempts is nonconverged.attempts
    assert tuple(attempt["steps"] for attempt in captured.value.attempts) == (
        IG_STEPS
    )
    assert all(attempt["pass"] == [False] for attempt in captured.value.attempts)
    assert model.forward_calls == calls_before


def test_full_model_ig_at_intervention_nonconvergence_retains_evidence(
    monkeypatch,
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    intervention = image * 0.5
    with torch.no_grad():
        expected_delta = float(
            (
                model(intervention)[0, target.class_index]
                - model(torch.zeros_like(intervention))[0, target.class_index]
            ).item()
        )
    nonconverged = _nonconverged_ig_result(
        target, report_delta=expected_delta
    )
    monkeypatch.setattr(
        class_logit.attribution,
        "full_model_ig",
        lambda *_args, **_kwargs: nonconverged,
    )
    calls_before = model.forward_calls

    with pytest.raises(StageCClassLogitIGConvergenceError) as captured:
        full_model_class_logit_ig_at_intervention(
            model, image, intervention, target
        )

    assert captured.value.ig_result is nonconverged
    assert tuple(attempt["steps"] for attempt in captured.value.attempts) == (
        IG_STEPS
    )
    assert model.forward_calls == calls_before + 2


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("contradictory_converged", "all five failed attempts"),
        ("truncated_schedule", "all five failed attempts"),
        ("attempt_arithmetic", "attempt absolute error"),
        ("attempt_pass", "attempt pass flag"),
        ("conservation_arithmetic", "conservation sum"),
        ("metadata", "metadata 'step_chunk_size'"),
    ],
)
def test_full_model_ig_malformed_nonconvergence_fails_closed_without_evidence(
    monkeypatch, defect, message,
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    malformed = _nonconverged_ig_result(target)
    if defect == "contradictory_converged":
        malformed = _ig_result(target, converged=False)
    elif defect == "truncated_schedule":
        malformed = dataclasses.replace(
            malformed, steps=IG_STEPS[-2], attempts=malformed.attempts[:-1]
        )
    elif defect == "attempt_arithmetic":
        malformed.attempts[0]["sum"][0] += 1.0
    elif defect == "attempt_pass":
        malformed.attempts[0]["pass"][0] = True
    elif defect == "conservation_arithmetic":
        malformed.conservation["sum"][0] += 1.0
    elif defect == "metadata":
        malformed.metadata["step_chunk_size"] = 16
    monkeypatch.setattr(
        class_logit.attribution,
        "full_model_ig",
        lambda *_args, **_kwargs: malformed,
    )

    with pytest.raises(StageCClassLogitError, match=message) as captured:
        full_model_class_logit_ig(model, image, target)

    assert type(captured.value) is StageCClassLogitError
    assert not hasattr(captured.value, "ig_result")


@pytest.mark.parametrize(
    ("factory", "exception", "message"),
    [
        (
            lambda target: _ig_result(target, pass_value=False),
            StageCClassLogitError,
            "attempt pass flag",
        ),
        (
            lambda target: _ig_result(
                target, attribution=torch.zeros((1, 3, 223, 224), dtype=torch.float32)
            ),
            StageCClassLogitError,
            "shape",
        ),
        (
            lambda target: _ig_result(
                target, attribution=_signed_native().to(torch.float64)
            ),
            TypeError,
            "float32",
        ),
        (
            lambda target: _ig_result(
                target,
                attribution=_signed_native().index_put(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                    torch.tensor(float("nan")),
                ),
            ),
            StageCClassLogitError,
            "non-finite",
        ),
        (
            lambda target: _ig_result(
                target, attribution=torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            ),
            StageCClassLogitError,
            "all zero",
        ),
        (
            lambda target: _ig_result(
                target, metadata_updates={"step_chunk_size": 16}
            ),
            StageCClassLogitError,
            "metadata 'step_chunk_size'",
        ),
        (
            lambda target: _ig_result(target, attempt_steps=(1024,), steps=1024),
            StageCClassLogitError,
            "frozen prefix",
        ),
    ],
)
def test_full_model_ig_fails_closed_on_invalid_result(
    monkeypatch, factory, exception, message
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    monkeypatch.setattr(
        class_logit.attribution,
        "full_model_ig",
        lambda *_args, **_kwargs: factory(target),
    )
    with pytest.raises(exception, match=message):
        full_model_class_logit_ig(model, image, target)


def test_full_model_ig_rejects_internally_inconsistent_conservation(
    monkeypatch,
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    malformed = _ig_result(target)
    malformed.conservation["sum"][0] += 1.0
    malformed.attempts[0]["sum"][0] += 1.0
    monkeypatch.setattr(
        class_logit.attribution,
        "full_model_ig",
        lambda *_args, **_kwargs: malformed,
    )
    with pytest.raises(StageCClassLogitError, match="attempt absolute error"):
        full_model_class_logit_ig(model, image, target)

    bad_attempt = _ig_result(target)
    bad_attempt.attempts[0]["sum"][0] += 1.0
    monkeypatch.setattr(
        class_logit.attribution,
        "full_model_ig",
        lambda *_args, **_kwargs: bad_attempt,
    )
    with pytest.raises(StageCClassLogitError, match="attempt absolute error"):
        full_model_class_logit_ig(model, image, target)


def test_full_model_ig_rejects_signed_rgb_cancellation_to_all_zero_metric(
    monkeypatch,
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    cancellation = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    cancellation[:, 0] = 1.0
    cancellation[:, 1] = -1.0
    monkeypatch.setattr(
        class_logit.attribution,
        "full_model_ig",
        lambda *_args, **_kwargs: _ig_result(target, attribution=cancellation),
    )
    with pytest.raises(StageCClassLogitError, match="patch map is all zero"):
        full_model_class_logit_ig(model, image, target)


@pytest.mark.parametrize("should_raise", [False, True], ids=["success", "failure"])
def test_full_model_ig_restores_hooks_modes_requires_grad_and_existing_grads(
    monkeypatch, should_raise
):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    _mixed_caller_state(model)
    caller_hook = model.blocks[3].register_forward_pre_hook(
        lambda _module, _args: None
    )
    hooks_before = _hook_inventory(model)
    modes_before = _mode_flags(model)
    requires_before = _requires_grad_flags(model)
    grads_before = _grad_snapshot(model)

    def fake(received_model, *_args, **_kwargs):
        assert not any(_mode_flags(received_model))
        assert not any(_requires_grad_flags(received_model))
        received_model.blocks[4].train()
        received_model.scale.requires_grad_(True)
        received_model.scale.grad.add_(-91.0)
        if should_raise:
            raise RuntimeError("synthetic full-model IG failure")
        return _ig_result(target)

    monkeypatch.setattr(class_logit.attribution, "full_model_ig", fake)
    try:
        if should_raise:
            with pytest.raises(RuntimeError, match="full-model IG failure"):
                full_model_class_logit_ig(model, image, target)
        else:
            result = full_model_class_logit_ig(model, image, target)
            assert np.count_nonzero(result.metric_map.data) > 0
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert caller_hook.id in model.blocks[3]._forward_pre_hooks
    finally:
        caller_hook.remove()


def test_full_model_ig_repeatability_and_target_input_guards(monkeypatch):
    model = _ToyClassModel().eval()
    image = _clean_image()
    target = resolve_clean_class_logit_target(model, image)
    monkeypatch.setattr(
        class_logit.attribution,
        "full_model_ig",
        lambda *_args, **_kwargs: _ig_result(target),
    )
    first = full_model_class_logit_ig(model, image, target)
    second = full_model_class_logit_ig(model, image, target)
    np.testing.assert_array_equal(first.native_tensor.data, second.native_tensor.data)
    np.testing.assert_array_equal(first.metric_map.data, second.metric_map.data)

    changed = image.clone()
    changed[0, 0, 0, 0] += 1.0
    with pytest.raises(StageCClassLogitError, match="object.*frozen target"):
        evaluate_frozen_class_logit_target(model, changed, image, target)
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="inference_mode"):
            full_model_class_logit_ig(model, image, target)


@pytest.mark.integration
def test_real_cpu_grad_cam_smoke_uses_block11_norm1_and_restores_lifecycle(
    state_dict_path,
):
    torch.manual_seed(20260821)
    model = load_canonical_model(state_dict_path, device="cpu")
    assert next(model.parameters()).device.type == "cpu"
    image = torch.randn(
        (1, 3, 224, 224), dtype=torch.float32, device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(20260821),
    )
    target = resolve_clean_class_logit_target(model, image)
    # Exercise non-default caller state on the real model as well as in toys.
    model.blocks[0].train()
    model.cls_token.requires_grad_(False)
    caller_grad = torch.linspace(
        -1.0, 1.0, model.norm.weight.numel(), dtype=torch.float32
    ).reshape_as(model.norm.weight)
    model.norm.weight.grad = caller_grad
    caller_calls = {"count": 0}

    def caller_hook(_module, _args, _output):
        caller_calls["count"] += 1

    persistent = model.blocks[11].norm1.register_forward_hook(caller_hook)
    try:
        hooks_before = _hook_inventory(model)
        modes_before = _mode_flags(model)
        requires_before = _requires_grad_flags(model)
        grads_before = _grad_snapshot(model)

        result = vit_grad_cam_block11_norm1(model, image, target)

        # One additional low-cost real forward computes gradients for the
        # frozen target and nearby alternate class targets at the same real
        # norm1 activation.  This is a sensitivity/parity smoke, not an IG run.
        with class_logit._preserve_model_attribution_state(
            model, freeze_parameters=True
        ), class_logit._ActivationCapture(
            model.blocks[11].norm1
        ) as capture, torch.enable_grad():
            probe = image.detach().clone().requires_grad_(True)
            logits = model(probe)
            activation = capture.activation
            assert activation is not None and capture.call_count == 1
            predicted_gradient = torch.autograd.grad(
                logits[:, target.class_index].sum(),
                activation,
                retain_graph=True,
            )[0]
            predicted_core = vit_grad_cam_patch_map(
                activation, predicted_gradient
            )
            alternate_core = None
            for candidate in torch.topk(logits[0], k=10).indices.tolist():
                if candidate == target.class_index:
                    continue
                candidate_gradient = torch.autograd.grad(
                    logits[:, candidate].sum(),
                    activation,
                    retain_graph=True,
                )[0]
                try:
                    candidate_map = vit_grad_cam_patch_map(
                        activation, candidate_gradient
                    )
                except StageCClassLogitError as exc:
                    if "all zero" in str(exc):
                        continue
                    raise
                if not torch.allclose(
                    predicted_core, candidate_map, atol=1e-6, rtol=1e-5
                ):
                    alternate_core = candidate_map
                    break
            assert alternate_core is not None

        _assert_common_method_result(
            result,
            method_id="vit_grad_cam_block11_norm1",
            projection="identity_14x14",
            sign_semantics=SignSemantics.NONNEGATIVE,
        )
        assert caller_calls["count"] == 2
        assert result.metric_map.shape == PATCH_SHAPE
        assert np.isfinite(result.metric_map.data).all()
        assert np.all(result.metric_map.data >= 0)
        assert np.count_nonzero(result.metric_map.data) > 0
        np.testing.assert_allclose(
            result.metric_map.data,
            predicted_core.detach().cpu().numpy(),
            atol=1e-6,
            rtol=1e-5,
        )
        assert not torch.allclose(
            predicted_core, alternate_core, atol=1e-6, rtol=1e-5
        )
        assert result.provenance["algorithm"]["target_layer"] == (
            "blocks[11].norm1"
        )
        assert _hook_inventory(model) == hooks_before
        assert _mode_flags(model) == modes_before
        assert _requires_grad_flags(model) == requires_before
        _assert_grad_snapshot(model, grads_before)
        assert model.norm.weight.grad is caller_grad
        assert persistent.id in model.blocks[11].norm1._forward_hooks
    finally:
        persistent.remove()
