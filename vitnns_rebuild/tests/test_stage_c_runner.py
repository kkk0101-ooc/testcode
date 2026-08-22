from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vitnns.stage_c_runner import (
    ProductionStageCBackend,
    StageCBackendError,
    StageCPreflightError,
    UnitPayload,
    aggregate_stage_c_batch,
    run_execution_unit,
    run_stage_c_batch,
    validate_stage_c_batch,
)
from vitnns.constants import MODEL_NAME, MODEL_SHA256
from vitnns.stage_c_artifacts import validate_execution_unit
from vitnns.stage_c_fidelity import fidelity_seed
from vitnns.stage_c_maps import NAReason, StructuredNA
from vitnns.stage_c_plan import WorkItem
from vitnns.stage_c_resources import CallCounter
from vitnns.stage_c_provenance import (
    CANONICAL_RGB_HASH_DOMAIN,
    NORMALIZED_INPUT_HASH_DOMAIN,
    build_stage_c_provenance,
    build_unit_identity,
    canonical_sha256,
)


SAMPLE_A = "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG"
SAMPLE_B = "ILSVRC2012_val/ILSVRC2012_val_00009888.JPEG"
METHOD = "vit_grad_cam_block11_norm1"
TARGET = "clean_predicted_class_raw_logit"
CONFIG_ID = f"global_{METHOD}"
ROLLOUT_METHOD = "attention_rollout_mean_head_residual_all_blocks"
SHA = "a" * 64
PASSING_PREFLIGHT = {
    "schema_version": 1,
    "document_type": "stage_c_preflight_estimate",
    "plan_digest": "f" * 64,
    "counts": {},
    "storage": {},
    "compute": {},
    "caps": {},
    "violations": [],
    "within_budget": True,
}
PASSING_PREFLIGHT["canonical_digest"] = canonical_sha256(PASSING_PREFLIGHT)
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


@dataclasses.dataclass(frozen=True)
class ToyKey:
    sample_id: str
    method_id: str = METHOD
    block_index: int | None = None
    head_index: int | None = None
    target_ref: str | None = TARGET
    config_id: str | None = None

    @property
    def canonical_digest(self) -> str:
        config_id = self.config_id or f"global_{self.method_id}"
        return build_unit_identity(
            sample_id=self.sample_id, method_id=self.method_id,
            config_id=config_id, block_index=self.block_index,
            head_index=self.head_index, target_ref=self.target_ref,
        )["canonical_digest"]

    def to_dict(self) -> dict[str, Any]:
        config_id = self.config_id or f"global_{self.method_id}"
        return build_unit_identity(
            sample_id=self.sample_id, method_id=self.method_id,
            config_id=config_id, block_index=self.block_index,
            head_index=self.head_index, target_ref=self.target_ref,
        )


@dataclasses.dataclass(frozen=True)
class ToySample:
    sample_id: str
    role: str = "calibration"
    audit: bool = False
    selection_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ToyUnit:
    sample: ToySample
    method_id: str = METHOD
    comparison_line: str = "class_logit"
    work_items: tuple[Any, ...] = ()

    @property
    def key(self) -> ToyKey:
        return ToyKey(self.sample.sample_id, method_id=self.method_id)

    @property
    def audit(self) -> bool:
        return self.sample.audit

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "sample": self.sample.to_dict(),
            "comparison_line": self.comparison_line,
            "audit": self.audit,
            "work_items": [
                item.to_dict() if callable(getattr(item, "to_dict", None)) else item
                for item in self.work_items
            ],
        }


@dataclasses.dataclass(frozen=True)
class ToyPlan:
    units: tuple[ToyUnit, ...]
    variant: str = "A"
    run_spec: dict[str, Any] | None = None

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            (
                self.variant + "|"
                + json.dumps(self.run_spec, sort_keys=True)
                + "|"
                + "|".join(unit.key.canonical_digest for unit in self.units)
            ).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "document_type": "stage_c_execution_plan",
            "variant": self.variant,
            "run_spec": self.run_spec,
            "digest": self.digest,
            "units": [unit.to_dict() for unit in self.units],
        }


class ToyBackend:
    def __init__(
        self,
        *,
        fail_sample_ids: tuple[str, ...] = (),
        comparison_line: str = "class_logit",
        include_l3: bool = False,
        l2_elements: int = 0,
        plan: ToyPlan | None = None,
    ) -> None:
        self.calls: list[tuple[Any, str, Any]] = []
        self.fail_sample_ids = frozenset(fail_sample_ids)
        self.comparison_line = comparison_line
        self.include_l3 = include_l3
        self.l2_elements = l2_elements
        self.plan = plan

    def run(self, unit, *, device: str, call_counter) -> UnitPayload:
        self.calls.append((unit, device, call_counter))
        if unit.key.sample_id in self.fail_sample_ids:
            raise RuntimeError(f"toy failure for {unit.key.sample_id}")
        return UnitPayload(
            l0={
                "sample_id": unit.key.sample_id,
                "method_id": unit.key.method_id,
                "comparison_line": self.comparison_line,
                "target_ref": unit.key.target_ref,
                "prediction": 17,
                "metrics": {"toy_score": 0.25},
            },
            l1={"metric_map": np.arange(196, dtype=np.float32).reshape(14, 14)},
            l2=(
                {"audit_raw": np.zeros(self.l2_elements, dtype=np.float32)}
                if self.l2_elements else {}
            ),
            l3={"display.png": PNG_1X1} if self.include_l3 else {},
            provenance={
                "comparison_line": self.comparison_line,
                "target_ref": unit.key.target_ref,
                "toy_backend": True,
            },
            resources={
                "wall_seconds": 0.001,
                "forward_calls": 1,
                "backward_calls": 1,
                "cuda_peak_allocated_bytes": None,
                "cuda_peak_reserved_bytes": None,
            },
        )

    def build_provenance(
        self, *, unit, payload, status, failure, resources, preflight,
        artifact_inventory,
    ):
        normalized = "b" * 64
        canonical = "c" * 64
        identity = {
            "comparison_line": "class_logit",
            "target_ref": TARGET,
            "clean_input_sha256": normalized,
            "frozen_target": {
                "target_kind": "raw_logit",
                "clean_input_sha256": normalized,
                "class_index": 17,
            },
        }
        parameters_by_id = {
            "translate_right_down_4": {
                "kind": "integer_translation", "dx_pixels": 4,
                "dy_pixels": 4, "padding": "reflection",
                "interpolation": "none",
                "inverse_map": (
                    "copy_perturbed_rows_4_223_cols_4_223_to_clean_rows_"
                    "0_219_cols_0_219"
                ),
                "valid_pixel_box_half_open": [0, 220, 0, 220],
                "fully_valid_patch_rows": list(range(13)),
                "fully_valid_patch_cols": list(range(13)),
            },
            "center_crop_216_resize_224": {
                "kind": "center_crop_scale",
                "crop_box_half_open": [4, 220, 4, 220],
                "crop_height": 216, "crop_width": 216,
                "resize_height": 224, "resize_width": 224,
                "image_interpolation": "bicubic", "image_antialias": True,
                "inverse_map_interpolation": "bilinear",
                "inverse_map_antialias": True,
                "inverse_mass_jacobian_multiplier": 1.075445816186557,
                "valid_pixel_box_half_open": [4, 220, 4, 220],
                "fully_valid_patch_rows": list(range(1, 13)),
                "fully_valid_patch_cols": list(range(1, 13)),
            },
            "gaussian_rgb_std_0_01": {
                "kind": "gaussian_noise", "mean": 0.0,
                "std_rgb_0_to_1": 0.01, "generator_device": "cpu",
                "generator_dtype": "float32", "base_seed": 20260821,
                "sample_seed_rule": "base_seed_plus_registry_selection_order",
                "seed": 20260821 + unit.sample.selection_order,
                "clamp_min": 0.0, "clamp_max": 1.0,
                "inverse_map": "identity",
                "valid_pixel_box_half_open": [0, 224, 0, 224],
                "fully_valid_patch_rows": list(range(14)),
                "fully_valid_patch_cols": list(range(14)),
            },
        }
        perturbations = []
        for index, perturbation_id in enumerate((
            "translate_right_down_4",
            "center_crop_216_resize_224",
            "gaussian_rgb_std_0_01",
        )):
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
                "parameters": parameters_by_id[perturbation_id],
                "shared_exact_tensor_across_methods": True,
            })
        method_contracts = {
            METHOD: {
                "exact_name": (
                    "ViT Grad-CAM (blocks[11].norm1, clean predicted-class raw logit)"
                ),
                "explanation_target_ref": TARGET,
                "class_agnostic": False,
                "native_map": "nonnegative_float32_14x14",
                "projection": "identity_14x14",
                "normalization": "none",
            },
            ROLLOUT_METHOD: {
                "exact_name": "attention rollout (mean-head residual, blocks 0-11)",
                "explanation_target_ref": None,
                "class_agnostic": True,
                "native_map": "nonnegative_14x14_patch_mass",
                "projection": "rollout_cls_to_patch",
                "normalization": "patch_mass_renormalized_to_one_after_cls_exclusion",
            },
        }
        method = method_contracts[unit.key.method_id]
        return build_stage_c_provenance(
            unit_identity=unit.key.to_dict(),
            code={"commit": "d" * 40, "dirty": True},
            model={
                "name": MODEL_NAME, "state_dict_path": "model/toy.pth",
                "state_dict_sha256": MODEL_SHA256,
            },
            input_record={
                "schema_version": 1, "sample_id": unit.sample.sample_id,
                "image_id": unit.sample.sample_id.split("/")[-1][:-5],
                "relative_path": unit.sample.sample_id.split("/")[-1],
                "image_sha256": SHA, "size_bytes": 1,
                "width": 224, "height": 224, "official_wnid": "n00000001",
                "imagenet1k_index": 17,
                "label_source": "official_devkit_val_gt",
                "role": unit.sample.role, "audit": unit.sample.audit,
                "selection_order": unit.sample.selection_order,
                "source_selection_index": unit.sample.selection_order,
                "selection_rule": (
                    "legacy_sorted_500_linspace20_then_fixed_role_positions_0_4_9_"
                    "14_19_audit_0_19"
                ),
                "canonical_rgb_sha256": canonical,
                "normalized_tensor_sha256": normalized,
            },
            dependencies={"lock_path": "requirements.lock.txt", "lock_sha256": SHA},
            registry={
                "path": "manifests/registry.csv", "sha256": SHA,
                "metadata_path": "manifests/metadata.json",
                "metadata_sha256": SHA,
                "source_manifest_path": "manifests/source.csv",
                "source_manifest_sha256": SHA,
                "source_manifest_metadata_path": "manifests/source_metadata.json",
                "source_manifest_metadata_sha256": SHA,
            },
            protocol={"path": "docs/protocol.md", "sha256": SHA, "version": "1.0.0"},
            acceptance={"path": "docs/acceptance.md", "sha256": SHA},
            configuration={
                "path": "config/stage_c.json", "sha256": SHA,
                "schema_path": "config/stage_c.schema.json",
                "schema_sha256": SHA,
                "registry_schema_path": "config/registry.schema.json",
                "registry_schema_sha256": SHA,
                "selection_schema_path": "config/selection.schema.json",
                "selection_schema_sha256": SHA,
                "run_spec_sha256": (
                    SHA if self.plan is None or self.plan.run_spec is None
                    else canonical_sha256(self.plan.run_spec)
                ),
                "execution_plan_sha256": preflight["plan_digest"],
            },
            selection={
                "schema_version": 1, "status": "not_applicable_pre_c8",
                "path": "selection/stage_c_selection.json", "sha256": None,
                "primary": None, "depth_band_winners": [],
                "reason": "toy calibration run before C8",
            },
            method={
                "id": unit.key.method_id, "exact_name": method["exact_name"],
                "version": "1.0.0", "comparison_line": "class_logit",
                "explanation_target_ref": method["explanation_target_ref"],
                "fidelity_target_ref": TARGET,
                "class_agnostic": method["class_agnostic"],
                "fidelity_eligible": True,
                "structural_reference_only": False,
                "native_map": method["native_map"],
            },
            target={
                "status": "resolved", "reason": None,
                "comparison_line": "class_logit", "target_ref": TARGET,
                "clean_input_normalized_sha256": normalized,
                "identity_sha256": canonical_sha256(identity), "identity": identity,
            },
            map_record={
                "projection": method["projection"],
                "sign_semantics": "nonnegative",
                "normalization": method["normalization"],
                "metric_ready_dtype": "float32",
                "metric_ready_shape": [14, 14],
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
                "base_seed": 20260821,
                "noise_seed": 20260821 + unit.sample.selection_order,
                "noise_seed_rule": "base_seed_plus_registry_selection_order",
                "fidelity_enabled": True,
                "fidelity_seed_namespace": (
                    "stage-c-fidelity|20260821|sample_id|block|head|replicate"
                ),
                "fidelity_pairing_context": {"block_index": 0, "head_index": 0},
                "fidelity_seeds": [
                    fidelity_seed(unit.sample.sample_id, 0, 0, repeat)
                    for repeat in range(5)
                ],
            },
            perturbations={
                "enabled": True,
                "application_domain": CANONICAL_RGB_HASH_DOMAIN,
                "normalized_domain": NORMALIZED_INPUT_HASH_DOMAIN,
                "clean": {
                    "id": "clean",
                    "canonical_rgb": {"domain": CANONICAL_RGB_HASH_DOMAIN, "sha256": canonical},
                    "normalized_input": {"domain": NORMALIZED_INPUT_HASH_DOMAIN, "sha256": normalized},
                },
                "items": perturbations,
            },
            status=status, failure=failure,
            resources={"preflight": preflight, "actual": resources},
            retention={
                "audit": unit.sample.audit, "display_is_never_metric_input": True,
                "l1_storage_dtype": "float32", "l2_audit_only": True,
                "l3_display_only": True,
                "forbidden_granularities": [
                    "per_quadrature_step_files", "per_pixel_files",
                    "per_dimension_files", "display_png_as_metric_input",
                ],
            },
            artifact_inventory=artifact_inventory,
        ).to_dict()


def toy_unit(
    sample_id: str = SAMPLE_A, *, audit: bool = False,
    method_id: str = METHOD, work_items: tuple[Any, ...] = (),
) -> ToyUnit:
    return ToyUnit(
        ToySample(sample_id=sample_id, audit=audit),
        method_id=method_id, work_items=work_items,
    )


def toy_plan(*sample_ids: str) -> ToyPlan:
    if not sample_ids:
        sample_ids = (SAMPLE_A,)
    return ToyPlan(tuple(toy_unit(sample_id) for sample_id in sample_ids))


def passing_estimate(plan: ToyPlan) -> dict[str, Any]:
    effective = {
        "bytes_per_l0_l1_execution_unit_max": 64 * 1024 * 1024,
        "bytes_per_audit_sample_l2_max": 1024 * 1024 * 1024,
        "stage_c_c8_c9_total_artifact_bytes_max": 32 * 1024**3,
        "forward_calls_per_execution_unit_max": 30_000,
        "backward_calls_per_execution_unit_max": 30_000,
        "cuda_peak_reserved_bytes_per_execution_unit_max": 90 * 1024**3,
        "stage_c_c8_c9_h20_gpu_hours_max": 1000.0,
    }
    document = {
        "schema_version": 1,
        "document_type": "stage_c_preflight_estimate",
        "plan_digest": plan.digest, "counts": {},
        "storage": {
            "maximum_l0_l1_bytes_per_execution_unit": 1,
            "maximum_l2_bytes_per_audit_sample": 0,
            "estimated_all_regular_file_bytes": 1,
        },
        "compute": {
            "maximum_forward_calls_per_execution_unit": 1,
            "maximum_backward_calls_per_execution_unit": 1,
            "measured_peak_reserved_bytes_per_unit": None,
            "projected_h20_gpu_hours": None,
        },
        "caps": {"effective": effective}, "violations": [],
        "within_budget": True,
    }
    document["canonical_digest"] = canonical_sha256(document)
    return document


def run_toy_unit(
    unit: ToyUnit, *, output_root: Path, backend: Any, plan: ToyPlan | None = None,
    preflight: dict[str, Any] | None = None, **kwargs: Any,
):
    plan = ToyPlan((unit,)) if plan is None else plan
    return run_execution_unit(
        unit, output_root=output_root, backend=backend, plan=plan,
        preflight=passing_estimate(plan) if preflight is None else preflight,
        **kwargs,
    )


def _attempts(output_root: Path) -> list[Path]:
    return sorted(path for path in output_root.rglob("attempt-*") if path.is_dir())


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _path(value: str | Path | None) -> Path:
    assert value is not None
    return Path(value)


def _unit_sample_id(value: Any) -> str:
    if isinstance(value, dict):
        return value["sample_id"]
    return value.sample_id


def _map_work(input_variant: str, repeat_index: int | None) -> WorkItem:
    return WorkItem(
        kind="map", input_variant=input_variant, repeat_index=repeat_index,
        fidelity_scale=None, curve_direction=None, ranking_kind=None,
        replicate_index=None, fraction_index=None, fraction=None,
        pairing_block_index=None, pairing_head_index=None,
    )


def _curve_work(
    fraction_index: int, *, direction: str = "insertion",
    replicate: int = 0, block: int = 0, head: int = 0,
) -> WorkItem:
    return WorkItem(
        kind="fidelity_curve_point", input_variant="clean",
        repeat_index=None, fidelity_scale="patch_16x16",
        curve_direction=direction,
        ranking_kind="target_oriented_positive_tie_permuted",
        replicate_index=replicate, fraction_index=fraction_index,
        fraction=fraction_index / 20.0, pairing_block_index=block,
        pairing_head_index=head,
    )


def _method_result(
    method_id: str, value: float = 0.0, *, na: StructuredNA | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        method_id=method_id, method_version="1.0.0",
        comparison_line=SimpleNamespace(value="class_logit"),
        fidelity_target=TARGET, na=na, provenance={},
        projection=(
            "identity_14x14" if method_id == METHOD else "rollout_cls_to_patch"
        ),
        sign_semantics=SimpleNamespace(value="nonnegative"),
        metric_map=(
            None
            if na is not None
            else SimpleNamespace(
                data=np.full((14, 14), value, dtype=np.float32)
            )
        ),
        native_tensor=None,
    )


def _production_backend_harness(
    monkeypatch, unit: ToyUnit, results: list[SimpleNamespace], *,
    provenance_builder=None,
) -> tuple[ProductionStageCBackend, list[dict[str, Any]]]:
    # Production execution counts real root-module invocations through a
    # forward pre-hook.  Keep the orchestration seam lightweight while still
    # satisfying that production model contract.
    model = torch.nn.Identity()
    tensors = {
        "canonical_rgb": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        "normalized_input": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        "registry_row": SimpleNamespace(
            imagenet1k_index=17, selection_order=unit.sample.selection_order
        ),
    }
    backend = ProductionStageCBackend(
        sample_loader=lambda _sample, _model: tensors,
        model_loader=lambda **_kwargs: model,
        provenance_builder=provenance_builder,
    )
    # Keep this a pure orchestration test: bypass canonical model loading and
    # preprocessing validation while still exercising ProductionStageCBackend.run.
    backend._models["cpu"] = model

    from vitnns import extractor, stage_c_class_logit, stage_c_fidelity

    class FakeExtractor:
        def __init__(self, received_model):
            assert received_model is model

        def extract(self, inputs):
            assert tuple(inputs.shape) == (1, 3, 224, 224)
            return object()

    target = SimpleNamespace(
        class_index=17, official_truth_index=17, clean_top1_correct=True
    )
    response = SimpleNamespace(target_identity={
        "comparison_line": "class_logit", "target_ref": TARGET,
        "clean_input_sha256": "b" * 64,
        "frozen_target": {"target_kind": "raw_logit", "class_index": 17},
    })
    monkeypatch.setattr(extractor, "NativeViTExtractor", FakeExtractor)
    monkeypatch.setattr(
        stage_c_class_logit, "resolve_clean_class_logit_target",
        lambda *_args, **_kwargs: target,
    )
    monkeypatch.setattr(
        stage_c_class_logit, "resolve_natural_prediction_observation",
        lambda *_args, **_kwargs: {
            "class_index": 17, "class_count": 1000,
            "intervention_input_sha256": "b" * 64,
            "runtime_model_binding_sha256": "c" * 64,
            "source": "natural_raw_logits_argmax",
        },
    )
    monkeypatch.setattr(
        stage_c_fidelity, "ClassLogitScalarAdapter",
        lambda *_args, **_kwargs: SimpleNamespace(
            target_identity=response.target_identity
        ),
    )
    monkeypatch.setattr(
        stage_c_fidelity, "resolve_target_response",
        lambda *_args, **_kwargs: response,
    )
    calls: list[dict[str, Any]] = []
    iterator = iter(results)

    def fake_run_method(received_unit, **kwargs):
        assert received_unit is unit
        calls.append(kwargs)
        return next(iterator)

    monkeypatch.setattr(backend, "_run_method", fake_run_method)
    return backend, calls


def test_normal_unit_run_is_complete_and_hash_valid(tmp_path):
    backend = ToyBackend()
    unit = toy_unit()

    result = run_toy_unit(
        unit, output_root=tmp_path / "run", backend=backend, device="cpu",
    )

    assert result.state == "succeeded"
    assert result.resumed is False
    assert result.failure is None
    assert len(backend.calls) == 1
    attempt_path = _path(result.attempt_path)
    assert attempt_path.is_dir()
    status = _json(attempt_path / "status.json")
    assert status["complete"] is True
    assert status["status"] == "complete"
    assert (attempt_path / "L0").is_dir()
    assert (attempt_path / "L1").is_dir()
    validation = validate_execution_unit(
        attempt_path, expected_unit_identity=unit.key.to_dict()
    )
    assert validation.valid is True
    assert validation.resumable is True
    assert validation.errors == ()
    telemetry = status["runner_telemetry"]
    timings = telemetry["phase_timings"]
    assert set(timings["phases"]) == {
        "attribution_or_map_compute", "fidelity_compute",
        "integrity_checking", "artifact_finalize",
    }
    assert timings["synchronized_total_wall_time_seconds"] >= max(
        item["wall_time_seconds"] for item in timings["phases"].values()
    )
    assert telemetry["measurement_environment"]["device_type"] == "cpu"
    assert telemetry["lifecycle"]["integrity"]["algorithm"] == (
        "unavailable_non_model_backend"
    )
    assert validate_execution_unit(
        attempt_path, expected_unit_identity=unit.key.to_dict(),
        require_runner_telemetry=True,
    ).resumable


def test_valid_complete_resume_skips_backend_and_reuses_attempt(tmp_path):
    output_root = tmp_path / "run"
    unit = toy_unit()
    plan = ToyPlan((unit,))
    first_backend = ToyBackend()
    first = run_toy_unit(
        unit, output_root=output_root, backend=first_backend, resume=True, plan=plan,
    )
    second_backend = ToyBackend()

    second = run_toy_unit(
        unit, output_root=output_root, backend=second_backend, resume=True, plan=plan,
    )

    assert first.state == "succeeded"
    assert second.state == "skipped_valid_complete"
    assert second.resumed is True
    assert second.attempt_path == first.attempt_path
    assert second_backend.calls == []
    assert _attempts(output_root) == [_path(first.attempt_path)]


def test_same_unit_key_with_changed_plan_and_preflight_never_resumes(tmp_path):
    unit = toy_unit()
    first_plan = ToyPlan((unit,), variant="first")
    second_plan = ToyPlan((unit,), variant="second")
    output_root = tmp_path / "run"

    first = run_stage_c_batch(
        first_plan, output_root=output_root, backend=ToyBackend(),
        estimate=passing_estimate(first_plan), _allow_unissued_estimate=True,
    )
    second_backend = ToyBackend()
    second = run_stage_c_batch(
        second_plan, output_root=output_root, backend=second_backend,
        estimate=passing_estimate(second_plan), _allow_unissued_estimate=True,
    )

    assert first.successful is True
    assert len(second_backend.calls) == 1
    assert second.unit_results[0].state == "succeeded"
    assert second.unit_results[0].resumed is False
    assert second.unit_results[0].attempt_path != first.unit_results[0].attempt_path
    assert len(_attempts(output_root)) == 2
    validation = validate_stage_c_batch(output_root, second_plan)
    assert validation.valid is True
    assert validation.successful is True


@pytest.mark.parametrize(
    "prior_kind", ["incomplete", "artifact_tampered", "manifest_tampered"]
)
def test_invalid_prior_attempt_is_preserved_and_rerun_uses_fresh_directory(
    tmp_path, prior_kind
):
    output_root = tmp_path / "run"
    unit = toy_unit()
    plan = ToyPlan((unit,))
    old_bytes: bytes
    if prior_kind == "incomplete":
        first = run_toy_unit(
            unit, output_root=output_root, backend=ToyBackend(), resume=True,
            plan=plan,
            fault_injector="before_complete_status",
        )
        assert first.state == "incomplete"
        status_path = _path(first.attempt_path) / "status.json"
        old_bytes = status_path.read_bytes()
    elif prior_kind == "artifact_tampered":
        first = run_toy_unit(
            unit, output_root=output_root, backend=ToyBackend(), resume=True,
            plan=plan,
        )
        artifact = _path(first.attempt_path) / "L1" / "metric_map.npy"
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
        status_path = artifact
        old_bytes = artifact.read_bytes()
    else:
        first = run_toy_unit(
            unit, output_root=output_root, backend=ToyBackend(), resume=True,
            plan=plan,
        )
        status_path = _path(first.attempt_path) / "manifest.json"
        status_path.write_bytes(status_path.read_bytes() + b" ")
        old_bytes = status_path.read_bytes()

    backend = ToyBackend()
    second = run_toy_unit(
        unit, output_root=output_root, backend=backend, resume=True, plan=plan,
    )

    assert second.state == "succeeded"
    assert second.resumed is False
    assert second.attempt_path != first.attempt_path
    assert status_path.read_bytes() == old_bytes
    assert len(backend.calls) == 1
    assert _attempts(output_root) == [
        _path(first.attempt_path), _path(second.attempt_path)
    ]


def test_failed_attempt_is_preserved_and_resume_uses_fresh_attempt(tmp_path):
    output_root = tmp_path / "run"
    unit = toy_unit()
    plan = ToyPlan((unit,))
    failed = run_toy_unit(
        unit, output_root=output_root,
        backend=ToyBackend(fail_sample_ids=(SAMPLE_A,)), resume=True, plan=plan,
    )
    failed_path = _path(failed.attempt_path)
    failed_status = (failed_path / "status.json").read_bytes()

    successful = run_toy_unit(
        unit, output_root=output_root, backend=ToyBackend(), resume=True, plan=plan,
    )

    assert failed.state == "failed"
    assert successful.state == "succeeded"
    assert _path(successful.attempt_path) != failed_path
    assert (failed_path / "status.json").read_bytes() == failed_status
    failed_telemetry = _json(failed_path / "status.json")["runner_telemetry"]
    assert failed_telemetry["phase_timings"][
        "synchronized_total_wall_time_seconds"
    ] >= failed_telemetry["phase_timings"]["phases"][
        "artifact_finalize"
    ]["wall_time_seconds"]
    assert failed_telemetry["lifecycle"]["integrity"][
        "body_exception_type"
    ] == "builtins.RuntimeError"


def test_resume_rejects_complete_attempt_without_runner_telemetry(tmp_path):
    output_root = tmp_path / "run"
    unit = toy_unit()
    plan = ToyPlan((unit,))
    first = run_toy_unit(
        unit, output_root=output_root, backend=ToyBackend(), resume=True, plan=plan,
    )
    attempt = _path(first.attempt_path)
    status = _json(attempt / "status.json")
    status.pop("runner_telemetry")
    manifest = _json(attempt / "manifest.json")
    manifest.pop("runner_telemetry_sha256")
    (attempt / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status["manifest_sha256"] = hashlib.sha256(
        (attempt / "manifest.json").read_bytes()
    ).hexdigest()
    (attempt / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    backend = ToyBackend()
    second = run_toy_unit(
        unit, output_root=output_root, backend=backend, resume=True, plan=plan,
    )

    assert second.state == "succeeded"
    assert second.attempt_path != first.attempt_path
    assert len(backend.calls) == 1


def test_total_timer_stops_after_first_finalize_validation(
    monkeypatch, tmp_path
):
    import vitnns.stage_c_artifacts as artifacts
    import vitnns.stage_c_runner as runner
    from vitnns.stage_c_resources import PhaseTimer as RealPhaseTimer

    timers = []

    def make_timer(*, device):
        timer = RealPhaseTimer(device=device)
        timers.append(timer)
        return timer

    original_validate = artifacts.validate_execution_unit
    validation_states = []

    def observed_validate(*args, **kwargs):
        validation_states.append((
            timers[-1].stopped, kwargs.get("require_runner_telemetry", False)
        ))
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(runner, "PhaseTimer", make_timer)
    monkeypatch.setattr(artifacts, "validate_execution_unit", observed_validate)

    result = run_toy_unit(
        toy_unit(), output_root=tmp_path / "run", backend=ToyBackend(),
        resume=False,
    )

    assert result.state == "succeeded"
    assert validation_states == [(False, False), (True, True)]


def test_phase_timing_does_not_change_batched_call_accounting(tmp_path):
    class CountingBackend(ToyBackend):
        def run(self, unit, *, device, call_counter):
            call_counter.record_forward(2)
            call_counter.record_backward(1)
            return super().run(
                unit, device=device, call_counter=call_counter
            )

    result = run_toy_unit(
        toy_unit(), output_root=tmp_path / "run",
        backend=CountingBackend(), resume=False,
    )
    attempt = _path(result.attempt_path)
    l0 = _json(attempt / "L0" / "result.json")

    assert l0["resources"]["calls"] == {
        "accounting_unit": "batched_model_invocation",
        "forward_calls": 2,
        "backward_calls": 1,
    }
    assert _json(attempt / "status.json")["runner_telemetry"][
        "phase_timings"
    ]["phases"]["attribution_or_map_compute"]["invocations"] == 1


def test_production_pre_target_failure_is_terminal_and_retry_uses_fresh_attempt(
    tmp_path,
):
    from vitnns.stage_c_config import load_stage_c_config
    from vitnns.stage_c_plan import (
        build_execution_plan, estimate_stage_c_run, load_stage_c_registry,
        make_run_spec,
    )

    config = load_stage_c_config()
    registry = load_stage_c_registry(config)
    run_spec = make_run_spec(
        sample_ids=[SAMPLE_A], method_ids=[METHOD],
        value_config_source="none", value_candidates=[],
        fidelity_pairing_context=None, audit_repeatability_runs=1,
        audit_repeatability_scope="clean_map_only",
        run_registered_fidelity=False, run_sanity=False,
        run_stability=False,
    )
    plan = build_execution_plan(config, registry, run_spec)
    unit = plan.units[0]
    estimate = estimate_stage_c_run(config, plan).to_dict()
    attempts = 0

    def fail_before_sample(_sample, _model):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("sample preprocessing failed")

    backend = ProductionStageCBackend(
        sample_loader=fail_before_sample,
        model_loader=lambda **_kwargs: torch.nn.Identity(),
        config=config, registry=registry, plan=plan,
    )
    # Skip canonical-model metadata validation so this regression stays a
    # lightweight test of the runner's pre-target persistence path.
    backend._models["cpu"] = torch.nn.Identity()
    output_root = tmp_path / "production-pre-target-failure"

    first = run_execution_unit(
        unit, output_root=output_root, backend=backend, plan=plan,
        preflight=estimate, device="cpu", resume=True,
    )

    assert first.state == "failed"
    assert attempts == 1
    first_attempt = _path(first.attempt_path)
    status = _json(first_attempt / "status.json")
    provenance = _json(first_attempt / "L0" / "provenance.json")
    assert status["status"] == "failed"
    assert status["terminal"] is True
    assert status["complete"] is False
    assert provenance["bindings"]["target"] == {
        "status": "unavailable",
        "reason": "sample preprocessing failed",
        "comparison_line": "class_logit",
        "target_ref": TARGET,
        "clean_input_normalized_sha256": None,
        "identity_sha256": None,
        "identity": None,
    }
    assert provenance["bindings"]["input"]["canonical_rgb_sha256"] is None
    assert provenance["bindings"]["input"]["normalized_tensor_sha256"] is None
    assert provenance["bindings"]["perturbations"]["enabled"] is False
    assert provenance["bindings"]["perturbations"]["items"] == []
    validation = validate_execution_unit(
        first_attempt, expected_unit_identity=unit.key.to_dict(),
        observed_git_state=backend.code_state,
    )
    assert validation.valid is True
    assert validation.resumable is False

    second = run_execution_unit(
        unit, output_root=output_root, backend=backend, plan=plan,
        preflight=estimate, device="cpu", resume=True,
    )

    assert second.state == "failed"
    assert attempts == 2
    assert second.attempt_path != first.attempt_path
    assert _json(first_attempt / "status.json") == status
    assert len(_attempts(output_root)) == 2


def test_resume_false_never_overwrites_an_existing_complete_attempt(tmp_path):
    output_root = tmp_path / "run"
    unit = toy_unit()
    plan = ToyPlan((unit,))
    first = run_toy_unit(
        unit, output_root=output_root, backend=ToyBackend(), resume=True, plan=plan,
    )
    sentinel = _path(first.attempt_path) / "sentinel.keep"
    sentinel.write_bytes(b"preserve me")

    second = run_toy_unit(
        unit, output_root=output_root, backend=ToyBackend(), resume=False, plan=plan,
    )

    assert second.state == "succeeded"
    assert second.attempt_path != first.attempt_path
    assert sentinel.read_bytes() == b"preserve me"


def test_partial_failure_is_terminal_and_every_expected_unit_gets_a_row(tmp_path):
    plan = toy_plan(SAMPLE_A, SAMPLE_B)
    backend = ToyBackend(fail_sample_ids=(SAMPLE_B,))

    result = run_stage_c_batch(
        plan, output_root=tmp_path / "batch", backend=backend, device="cpu",
        estimate=passing_estimate(plan),
        _allow_unissued_estimate=True,
    )

    assert result.orchestration_complete is True
    assert result.successful is False
    assert len(result.unit_results) == len(plan.units) == 2
    states = {
        _unit_sample_id(row.unit_key): row.state for row in result.unit_results
    }
    assert states == {SAMPLE_A: "succeeded", SAMPLE_B: "failed"}
    validation = validate_stage_c_batch(tmp_path / "batch", plan)
    assert validation.orchestration_complete is True
    assert validation.successful is False
    assert len(validation.rows) == 2


def test_missing_failure_table_row_invalidates_batch(tmp_path):
    plan = toy_plan(SAMPLE_A, SAMPLE_B)
    output_root = tmp_path / "batch"
    run_stage_c_batch(
        plan, output_root=output_root, backend=ToyBackend(),
        estimate=passing_estimate(plan),
        _allow_unissued_estimate=True,
    )
    failure_table = output_root / "failure_table.json"
    document = _json(failure_table)
    rows_key = "rows" if isinstance(document, dict) else None
    if rows_key is None:
        document = document[:-1]
    else:
        document[rows_key] = document[rows_key][:-1]
    failure_table.write_text(
        json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation = validate_stage_c_batch(output_root, plan)

    assert validation.valid is False
    assert validation.orchestration_complete is False
    assert any("missing" in error.lower() for error in validation.errors)


def test_preflight_rejection_precedes_output_root_and_backend(tmp_path):
    output_root = tmp_path / "must-not-exist"
    backend = ToyBackend()
    estimate = type("RejectedEstimate", (), {
        "within_budget": False,
        "violations": ("forward_calls_per_execution_unit_max",),
        "to_dict": lambda self: {
            "within_budget": False,
            "violations": ["forward_calls_per_execution_unit_max"],
        },
    })()

    with pytest.raises(StageCPreflightError, match="budget|preflight|forward"):
        run_stage_c_batch(
            toy_plan(), output_root=output_root, backend=backend, estimate=estimate
        )

    assert backend.calls == []
    assert not output_root.exists()


def test_backend_payload_cannot_cross_comparison_lines(tmp_path):
    unit = toy_unit()
    backend = ToyBackend(comparison_line="value_path")

    result = run_toy_unit(
        unit, output_root=tmp_path / "run", backend=backend,
    )

    assert result.state == "failed"
    assert result.failure is not None
    assert "comparison" in json.dumps(result.failure).lower()


def test_validate_and_aggregate_do_not_read_l3_display(monkeypatch, tmp_path):
    plan = ToyPlan((toy_unit(audit=True),))
    output_root = tmp_path / "batch"
    run_stage_c_batch(
        plan, output_root=output_root, backend=ToyBackend(include_l3=True),
        estimate=passing_estimate(plan),
        _allow_unissued_estimate=True,
    )
    loaded_npy: list[Path] = []
    original_numpy_load = np.load

    def guarded_numpy_load(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            path = Path(file)
            assert "L3" not in path.parts
            assert path.suffix.lower() != ".png"
            loaded_npy.append(path)
        elif isinstance(file, io.BytesIO):
            position = file.tell()
            signature = file.read(8)
            file.seek(position)
            assert signature != b"\x89PNG\r\n\x1a\n"
        return original_numpy_load(file, *args, **kwargs)

    # Integrity validation may hash every retained byte, including L3.  The
    # quantitative boundary is that neither validator nor aggregate may decode
    # a display file as a numeric array.
    monkeypatch.setattr(np, "load", guarded_numpy_load)
    validation = validate_stage_c_batch(output_root, plan)
    aggregate = aggregate_stage_c_batch(output_root, plan)

    assert validation.valid is True
    assert all("L3" not in path.parts for path in loaded_npy)
    assert aggregate["source_levels"] == ["L0"]
    assert aggregate["row_count"] == 1


def test_production_backend_reexecutes_each_clean_repeat(monkeypatch):
    unit = toy_unit(
        audit=True, method_id=ROLLOUT_METHOD,
        work_items=(
            _map_work("clean", 0),
            _map_work("repeat_clean", 1),
            _map_work("repeat_clean", 2),
        ),
    )
    backend, method_calls = _production_backend_harness(
        monkeypatch, unit,
        [_method_result(ROLLOUT_METHOD) for _index in range(3)],
    )

    payload = backend.run(unit, device="cpu", call_counter=CallCounter())

    assert len(method_calls) == 3
    assert [item["repeat_index"] for item in payload.l0["repeatability"]] == [1, 2]
    assert all(item["pass"] is True for item in payload.l0["repeatability"])
    assert [
        item["input_variant"]
        for item in payload.l0["map_execution_records"]
    ] == ["clean", "repeat_clean", "repeat_clean"]
    assert all(
        item["actual_calls"]["accounting_unit"]
        == "batched_model_invocation"
        for item in payload.l0["map_execution_records"]
    )


def test_production_backend_strong_signatures_scale_by_unit_not_maps(
    monkeypatch
):
    from vitnns import stage_c_integrity

    unit = toy_unit(
        audit=True, method_id=ROLLOUT_METHOD,
        work_items=(
            _map_work("clean", 0),
            _map_work("repeat_clean", 1),
            _map_work("repeat_clean", 2),
        ),
    )
    backend, _method_calls = _production_backend_harness(
        monkeypatch, unit,
        [_method_result(ROLLOUT_METHOD) for _index in range(6)],
    )
    original = stage_c_integrity.strong_model_signature
    calls = 0

    def counted(model):
        nonlocal calls
        calls += 1
        return original(model)

    monkeypatch.setattr(stage_c_integrity, "strong_model_signature", counted)

    first = backend.run(unit, device="cpu", call_counter=CallCounter())
    second = backend.run(unit, device="cpu", call_counter=CallCounter())

    assert calls == 1 + 2 * 2
    for payload in (first, second):
        assert payload.lifecycle["integrity"]["signature_calls"] == 2
        assert payload.lifecycle["integrity"]["matched"] is True
        assert payload.lifecycle["restoration"]["restored"] is True


def test_production_backend_fails_on_nonrepeatable_clean_map(monkeypatch):
    unit = toy_unit(
        audit=True, method_id=ROLLOUT_METHOD,
        work_items=(
            _map_work("clean", 0), _map_work("repeat_clean", 1),
        ),
    )
    backend, method_calls = _production_backend_harness(
        monkeypatch, unit,
        [_method_result(ROLLOUT_METHOD, 0.0), _method_result(ROLLOUT_METHOD, 0.01)],
    )

    with pytest.raises(StageCBackendError, match="repeatability failed"):
        backend.run(unit, device="cpu", call_counter=CallCounter())

    assert len(method_calls) == 2


def test_production_backend_without_stability_never_builds_perturbations(
    monkeypatch
):
    unit = toy_unit(
        audit=True, method_id=ROLLOUT_METHOD,
        work_items=(
            _map_work("clean", 0), _map_work("repeat_clean", 1),
        ),
    )
    backend, _method_calls = _production_backend_harness(
        monkeypatch, unit,
        [_method_result(ROLLOUT_METHOD), _method_result(ROLLOUT_METHOD)],
    )
    from vitnns import stage_c_perturbations

    monkeypatch.setattr(
        stage_c_perturbations, "generate_stage_c_perturbations",
        lambda *_args, **_kwargs: pytest.fail(
            "stability-disabled unit generated perturbations"
        ),
    )

    payload = backend.run(unit, device="cpu", call_counter=CallCounter())

    assert payload.l0["stability_enabled"] is False
    assert payload.l0["perturbations"] == []
    assert backend._perturbations == {}


def test_frozen_perturbation_ids_are_not_copied_into_provenance_parameters():
    from vitnns.stage_c_config import load_stage_c_config
    from vitnns.stage_c_runner import _provenance_perturbation_parameters

    items = load_stage_c_config().perturbations["items"]
    parameters = _provenance_perturbation_parameters(items)

    assert tuple(parameters) == tuple(item["id"] for item in items)
    assert all("id" not in value for value in parameters.values())
    assert all("id" in item for item in items)


def test_structural_raw_attention_executes_and_retains_all_perturbation_maps(
    monkeypatch
):
    method_id = "direct_raw_cls_attention_per_head"
    unit = SimpleNamespace(
        key=SimpleNamespace(to_dict=lambda: build_unit_identity(
            sample_id=SAMPLE_A, method_id=method_id,
            config_id="block_00_head_00", block_index=0, head_index=0,
            target_ref=None,
        )),
        sample=ToySample(SAMPLE_A, audit=True), comparison_line="value_path",
        audit=True,
        work_items=(
            _map_work("clean", 0),
            _map_work("translate_right_down_4", None),
            _map_work("center_crop_216_resize_224", None),
            _map_work("gaussian_rgb_std_0_01", None),
        ),
    )
    results = [
        _method_result(ROLLOUT_METHOD, float(index)) for index in range(4)
    ]
    for result in results:
        result.method_id = method_id
        result.fidelity_target = None
        result.comparison_line = SimpleNamespace(value="value_path")
        result.projection = "raw_cls_attention_to_patch"
        result.metric_map = __import__(
            "vitnns.stage_c_maps", fromlist=["stage_c_metric_ready"]
        ).stage_c_metric_ready(result.metric_map.data)
    backend, method_calls = _production_backend_harness(
        monkeypatch, unit, results
    )

    payload = backend.run(unit, device="cpu", call_counter=CallCounter())

    assert len(method_calls) == 4
    assert set(payload.l1) == {
        "clean", "translate_right_down_4",
        "center_crop_216_resize_224", "gaussian_rgb_std_0_01",
    }
    assert payload.l0["stability"] == []
    assert [
        item["perturbation_id"]
        for item in payload.l0["structural_perturbations"]
    ] == [
        "translate_right_down_4", "center_crop_216_resize_224",
        "gaussian_rgb_std_0_01",
    ]
    assert all(
        item["comparison_label"]
        == "spatial_structure_similarity_not_same_target_fidelity"
        and item["target_ref"] is None
        for item in payload.l0["structural_perturbations"]
    )
    assert all(
        set(item["metrics"]) == {
            "pearson", "spearman", "jensen_shannon_divergence", "kl",
            "hotspot_jaccard_top_10_percent",
            "hotspot_jaccard_top_20_percent",
        }
        for item in payload.l0["structural_perturbations"]
    )
    assert payload.l0["fidelity_enabled"] is False
    assert [
        (item["input_variant"], item["repeat_index"])
        for item in payload.l0["map_execution_records"]
    ] == [
        ("clean", 0),
        ("translate_right_down_4", None),
        ("center_crop_216_resize_224", None),
        ("gaussian_rgb_std_0_01", None),
    ]
    expected_digests = {
        name: item.normalized_tensor_sha256
        for name, item in backend._perturbations[(SAMPLE_A, "cpu")].items()
    }
    assert method_calls[0].get("intervention_input_sha256") is None
    assert {
        name: call["intervention_input_sha256"]
        for name, call in zip(
            (
                "translate_right_down_4", "center_crop_216_resize_224",
                "gaussian_rgb_std_0_01",
            ),
            method_calls[1:],
        )
    } == expected_digests


@pytest.mark.parametrize(
    "indices", [tuple(range(20)), tuple(range(20)) + (19,)]
)
def test_production_backend_rejects_incomplete_or_duplicate_fidelity_fractions(
    monkeypatch, indices
):
    unit = toy_unit(
        method_id=ROLLOUT_METHOD,
        work_items=tuple(_curve_work(index) for index in indices),
    )
    backend, _method_calls = _production_backend_harness(
        monkeypatch, unit, [_method_result(ROLLOUT_METHOD)]
    )
    from vitnns import stage_c_fidelity

    monkeypatch.setattr(
        stage_c_fidelity, "evaluate_fidelity_curve",
        lambda *_args, **_kwargs: pytest.fail(
            "malformed curve reached the fidelity evaluator"
        ),
    )

    with pytest.raises(StageCBackendError, match="complete 21-point curves"):
        backend.run(unit, device="cpu", call_counter=CallCounter())


def test_structured_na_is_terminal_failure_without_fabricated_l1(
    monkeypatch, tmp_path
):
    unit = toy_unit(method_id=ROLLOUT_METHOD, work_items=(_map_work("clean", 0),))
    na_result = _method_result(
        ROLLOUT_METHOD, na=StructuredNA(NAReason.MISSING_ARTIFACT, "toy NA")
    )
    provenance = ToyBackend()
    backend, _method_calls = _production_backend_harness(
        monkeypatch, unit, [na_result],
        provenance_builder=provenance.build_provenance,
    )

    result = run_toy_unit(
        unit, output_root=tmp_path / "run", backend=backend,
    )

    assert result.state == "failed"
    assert "structured NA" in result.failure["message"]
    attempt = _path(result.attempt_path)
    status = _json(attempt / "status.json")
    assert status["terminal"] is True
    assert status["complete"] is False
    assert not list((attempt / "L1").glob("*")) if (attempt / "L1").exists() else True


@pytest.mark.parametrize(
    ("cap_name", "cap_value", "l2_elements", "error_text"),
    [
        ("bytes_per_audit_sample_l2_max", 5_000, 1_000, "audit sample"),
        ("stage_c_c8_c9_total_artifact_bytes_max", 1, 0, "regular-file bytes"),
    ],
)
def test_batch_actual_byte_caps_fail_closed_after_preserving_rows(
    tmp_path, cap_name, cap_value, l2_elements, error_text
):
    units = (
        toy_unit(SAMPLE_A, audit=True, method_id=METHOD),
        toy_unit(SAMPLE_A, audit=True, method_id=ROLLOUT_METHOD),
    )
    plan = ToyPlan(units)
    estimate = passing_estimate(plan)
    estimate["caps"]["effective"][cap_name] = cap_value
    estimate["canonical_digest"] = canonical_sha256({
        key: value for key, value in estimate.items() if key != "canonical_digest"
    })
    caps = dict(estimate["caps"]["effective"])

    result = run_stage_c_batch(
        plan, output_root=tmp_path / "batch",
        backend=ToyBackend(l2_elements=l2_elements), estimate=estimate, caps=caps,
        _allow_unissued_estimate=True,
    )
    validation = validate_stage_c_batch(tmp_path / "batch", plan)

    assert len(result.unit_results) == len(units)
    assert result.orchestration_complete is False
    assert result.successful is False
    assert validation.valid is False
    assert any(error_text in message for message in validation.errors)


def _run_spec_fixture(*, run_stability: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "document_type": "stage_c_run_spec",
        "sample_ids": [SAMPLE_A],
        "method_ids": [METHOD],
        "value_config_source": "none",
        "value_candidates": [],
        "fidelity_pairing_context": None,
        "audit_repeatability_runs": 1,
        "audit_repeatability_scope": "clean_map_only",
        "run_registered_fidelity": False,
        "run_sanity": False,
        "run_stability": run_stability,
    }


def _rewrite_batch_controls_for_plan(
    output_root: Path, plan: ToyPlan, preflight: dict[str, Any]
) -> None:
    (output_root / "execution_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failure_path = output_root / "failure_table.json"
    failure_table = _json(failure_path)
    failure_table["plan_digest"] = plan.digest
    failure_path.write_text(
        json.dumps(failure_table, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path = output_root / "batch_status.json"
    status = _json(status_path)
    status["plan_digest"] = plan.digest
    status["failure_table_sha256"] = hashlib.sha256(
        failure_path.read_bytes()
    ).hexdigest()
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_changed_plan_and_preflight_never_resume_skip_same_unit(tmp_path):
    unit = toy_unit()
    plan_a = ToyPlan(
        (unit,), variant="A", run_spec=_run_spec_fixture(run_stability=False)
    )
    plan_b = ToyPlan(
        (unit,), variant="B", run_spec=_run_spec_fixture(run_stability=True)
    )
    output_root = tmp_path / "batch"
    backend_a = ToyBackend(plan=plan_a)
    first = run_stage_c_batch(
        plan_a, output_root=output_root, backend=backend_a,
        estimate=passing_estimate(plan_a), _allow_unissued_estimate=True,
    )
    old_attempt = _path(first.unit_results[0].attempt_path)
    old_provenance = (old_attempt / "L0" / "provenance.json").read_bytes()
    backend_b = ToyBackend(plan=plan_b)

    second = run_stage_c_batch(
        plan_b, output_root=output_root, backend=backend_b,
        estimate=passing_estimate(plan_b), _allow_unissued_estimate=True,
    )

    assert plan_a.digest != plan_b.digest
    assert second.unit_results[0].state == "succeeded"
    assert second.unit_results[0].resumed is False
    assert len(backend_b.calls) == 1
    assert _path(second.unit_results[0].attempt_path) != old_attempt
    assert (old_attempt / "L0" / "provenance.json").read_bytes() == old_provenance
    assert len(_attempts(output_root)) == 2


def test_rewritten_batch_controls_cannot_authorize_old_plan_provenance(tmp_path):
    unit = toy_unit()
    plan_a = ToyPlan(
        (unit,), variant="A", run_spec=_run_spec_fixture(run_stability=False)
    )
    plan_b = ToyPlan(
        (unit,), variant="B", run_spec=_run_spec_fixture(run_stability=True)
    )
    output_root = tmp_path / "batch"
    result = run_stage_c_batch(
        plan_a, output_root=output_root, backend=ToyBackend(plan=plan_a),
        estimate=passing_estimate(plan_a), _allow_unissued_estimate=True,
    )
    attempt = _path(result.unit_results[0].attempt_path)
    provenance_bytes = (attempt / "L0" / "provenance.json").read_bytes()

    preflight_b = passing_estimate(plan_b)
    _rewrite_batch_controls_for_plan(output_root, plan_b, preflight_b)
    validation = validate_stage_c_batch(output_root, plan_b)

    assert validation.valid is False
    assert validation.successful is False
    assert validation.rows[0]["state"] == "tampered"
    assert any(
        "provenance binding mismatch" in message
        and ("execution_plan_sha256" in message or "run_spec_sha256" in message)
        for message in validation.errors
    )
    assert (attempt / "L0" / "provenance.json").read_bytes() == provenance_bytes


def test_execution_plan_control_must_equal_current_plan(tmp_path):
    plan = toy_plan()
    output_root = tmp_path / "batch"
    result = run_stage_c_batch(
        plan, output_root=output_root, backend=ToyBackend(),
        estimate=passing_estimate(plan), _allow_unissued_estimate=True,
    )
    assert result.successful is True
    document = _json(output_root / "execution_plan.json")
    document["variant"] = "tampered"
    (output_root / "execution_plan.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation = validate_stage_c_batch(output_root, plan)

    assert validation.valid is False
    assert any(
        "execution_plan.json does not match" in message
        for message in validation.errors
    )


def test_top20_never_cross_pairs_different_block_head_contexts(monkeypatch):
    work_items = tuple(
        _curve_work(
            fraction, direction=direction, replicate=replicate,
            block=block, head=head,
        )
        for block, head in ((0, 0), (1, 1))
        for direction in ("insertion", "deletion")
        for replicate in range(5)
        for fraction in range(21)
    )
    unit = toy_unit(method_id=ROLLOUT_METHOD, work_items=work_items)
    backend, _method_calls = _production_backend_harness(
        monkeypatch, unit, [_method_result(ROLLOUT_METHOD)]
    )
    from vitnns import stage_c_fidelity

    monkeypatch.setattr(
        stage_c_fidelity, "method_unit_masses",
        lambda _result, _response, scale: SimpleNamespace(scale=scale.value),
    )

    def fake_schedule(
        _clean, _baseline, unit_map, _response, *, sample_id, block, head,
        replicate, direction,
    ):
        return SimpleNamespace(
            scale=unit_map.scale, direction=direction.value, replicate=replicate,
            block=block, head=head, sample_id=sample_id,
        )

    monkeypatch.setattr(
        stage_c_fidelity, "build_method_intervention_schedule", fake_schedule
    )

    def fake_curve(_evaluator, _response, schedule, *, method_id):
        context = (
            schedule.scale, schedule.direction, schedule.replicate,
            schedule.block, schedule.head,
        )
        return SimpleNamespace(
            context=context,
            to_dict=lambda: {
                "method_id": method_id, "context": list(context),
            },
        )

    monkeypatch.setattr(stage_c_fidelity, "evaluate_fidelity_curve", fake_curve)

    def fake_aggregate(curves):
        curves = tuple(curves)
        contexts = {
            (curve.context[0], curve.context[1], curve.context[3], curve.context[4])
            for curve in curves
        }
        assert len(contexts) == 1
        return SimpleNamespace(to_dict=lambda: {"context": list(next(iter(contexts)))})

    monkeypatch.setattr(stage_c_fidelity, "aggregate_five_curves", fake_aggregate)
    top20_pairs: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []

    def fake_top20(insertion, deletion):
        top20_pairs.append((insertion.context, deletion.context))
        return SimpleNamespace(to_dict=lambda: {
            "block": insertion.context[3], "head": insertion.context[4],
            "replicate": insertion.context[2],
        })

    monkeypatch.setattr(stage_c_fidelity, "top20_from_curves", fake_top20)

    payload = backend.run(unit, device="cpu", call_counter=CallCounter())

    assert len(top20_pairs) == 10
    assert {
        (left[3], left[4]) for left, _right in top20_pairs
    } == {(0, 0), (1, 1)}
    assert all(
        left[0] == right[0]
        and left[2:] == right[2:]
        and left[1] == "insertion"
        and right[1] == "deletion"
        for left, right in top20_pairs
    )
    assert len(payload.l0["top20"]) == 10


def test_batch_total_cap_counts_every_retained_attempt_and_extra_file(tmp_path):
    plan = toy_plan()
    output_root = tmp_path / "batch"
    estimate = passing_estimate(plan)
    first = run_stage_c_batch(
        plan, output_root=output_root, backend=ToyBackend(),
        estimate=estimate, _allow_unissued_estimate=True,
    )
    assert first.successful is True
    # A deliberately unselected historical attempt-shaped directory and an
    # unrelated ordinary file both consume the frozen batch storage budget.
    stale = (
        output_root / "units" / plan.units[0].key.canonical_digest
        / "attempts" / "attempt-999999"
    )
    stale.mkdir()
    (stale / "retained.bin").write_bytes(b"x" * 4096)
    (output_root / "extra.bin").write_bytes(b"y" * 4096)
    preflight = _json(output_root / "preflight.json")
    preflight["caps"]["effective"][
        "stage_c_c8_c9_total_artifact_bytes_max"
    ] = 1
    preflight["canonical_digest"] = canonical_sha256({
        key: value for key, value in preflight.items()
        if key != "canonical_digest"
    })
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation = validate_stage_c_batch(output_root, plan)

    assert validation.valid is False
    assert any(
        "regular-file bytes" in message or "artifact tree" in message
        for message in validation.errors
    )


@pytest.mark.integration
def test_real_cpu_production_backend_minimal_direct_attention_smoke(
    state_dict_path, tmp_path
):
    """Exercise the complete C6 runner and real ViT without fidelity or IG."""

    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform_functional

    from vitnns.constants import DEFAULT_VAL_ROOT
    from vitnns.modeling import load_canonical_model
    from vitnns.stage_c_config import load_stage_c_config
    from vitnns.stage_c_plan import (
        ValueCandidate, build_execution_plan, estimate_stage_c_run,
        load_stage_c_registry, make_run_spec,
    )

    image_path = DEFAULT_VAL_ROOT / "ILSVRC2012_val_00000079.JPEG"
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        canonical = transform_functional.pil_to_tensor(
            transform_functional.center_crop(
                transform_functional.resize(
                    rgb, 248, interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                [224, 224],
            )
        ).to(dtype=torch.float32).div(255.0).unsqueeze(0)
    normalized = (canonical - 0.5) / 0.5
    method_id = "direct_raw_cls_attention_per_head"
    config = load_stage_c_config()
    registry = load_stage_c_registry(config)
    run_spec = make_run_spec(
        sample_ids=[SAMPLE_A], method_ids=[method_id],
        value_config_source="explicit",
        value_candidates=[ValueCandidate(0, 0)],
        fidelity_pairing_context=None, audit_repeatability_runs=1,
        audit_repeatability_scope="clean_map_only",
        run_registered_fidelity=False, run_sanity=False,
        run_stability=False,
    )
    plan = build_execution_plan(config, registry, run_spec)
    estimate = estimate_stage_c_run(config, plan)
    assert estimate.within_budget and len(plan.units) == 1
    row = registry.by_sample_id()[SAMPLE_A]
    model = load_canonical_model(
        state_dict_path, device="cpu", unfused_attention=True
    )
    backend = ProductionStageCBackend(
        sample_loader=lambda _sample, received_model: {
            "canonical_rgb": canonical,
            "normalized_input": normalized,
            "registry_row": row,
        },
        model_loader=lambda **_kwargs: model,
        state_dict_path=state_dict_path,
        config=config, registry=registry, plan=plan,
    )
    output_root = tmp_path / "real-cpu-c6-smoke"
    run = run_stage_c_batch(
        plan, output_root=output_root, backend=backend, device="cpu",
        estimate=estimate, caps=config.budget.to_dict(),
    )

    assert run.successful is True and run.orchestration_complete is True
    assert len(run.unit_results) == 1
    attempt = _path(run.unit_results[0].attempt_path)
    validation = validate_stage_c_batch(output_root, plan)
    assert validation.valid and validation.successful
    result_record = _json(attempt / "L0" / "result.json")
    assert result_record["scalars"]["method_id"] == method_id
    assert result_record["scalars"]["target_ref"] is None
    assert result_record["scalars"]["fidelity_enabled"] is False
    assert result_record["scalars"]["stability_enabled"] is False
    assert result_record["resources"]["calls"] == {
        "accounting_unit": "batched_model_invocation",
        "forward_calls": 1, "backward_calls": 0,
    }
    l1 = np.load(attempt / "L1" / "clean.npy", allow_pickle=False)
    assert l1.shape == (14, 14) and l1.dtype == np.float32
