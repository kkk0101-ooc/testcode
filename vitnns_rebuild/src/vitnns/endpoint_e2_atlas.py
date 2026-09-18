"""Closed primitives for the ENDPOINT-E2 registered-target atlas.

The scanner evaluates one native model forward and one input VJP for every
registered scalar.  It deliberately contains no path integration, input
perturbation, stability, fidelity, or intervention operation.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .endpoint_nns import _frozen_eval
from .endpoint_vit import EndpointTarget, _select


METHOD_ID = "canonical_endpoint_x_times_gradient"
CLAIM_DECISION = "C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES"
AUTHORIZATION = "ENDPOINT_E2_ATLAS_REQUIRES_EXPLICIT_USER_REQUEST"
PASS_STATUS = "ENDPOINT_E2_PASS_NEXT_ENDPOINT_E3_REQUIRES_EXPLICIT_REQUEST"


class EndpointE2Error(ValueError):
    """An ENDPOINT-E2 contract was violated."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(str(array.dtype).encode())
    result.update(str(array.shape).encode())
    result.update(array.tobytes(order="C"))
    return result.hexdigest()


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False,
        ).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_npy_new(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def validate_run_spec(value: Mapping[str, Any]) -> None:
    """Validate the closed run-spec semantics without importing model assets."""
    expected_top = {
        "schema_version", "stage", "authorization_token", "claim_decision",
        "predecessors", "cohort", "models", "targets", "method_scope",
        "retention", "runtime", "outputs", "failure",
    }
    if set(value) != expected_top:
        raise EndpointE2Error("RUN_SPEC_KEYS")
    if value["schema_version"] != 1 or value["stage"] != "ENDPOINT-E2":
        raise EndpointE2Error("RUN_SPEC_IDENTITY")
    if value["authorization_token"] != AUTHORIZATION or value["claim_decision"] != CLAIM_DECISION:
        raise EndpointE2Error("AUTHORIZATION_OR_CLAIM")
    cohort = value["cohort"]
    if set(cohort) != {"split", "count", "models_share_rows", "replacement_allowed"}:
        raise EndpointE2Error("COHORT_KEYS")
    if cohort != {"split": "evaluation", "count": 400, "models_share_rows": True, "replacement_allowed": False}:
        raise EndpointE2Error("COHORT_SCOPE")
    if value["models"] != ["vit_b16", "deit_s16"]:
        raise EndpointE2Error("MODEL_SCOPE")
    targets = value["targets"]
    if set(targets) != {"registry", "registry_sha256", "count_by_model", "cross_model_channels"}:
        raise EndpointE2Error("TARGET_KEYS")
    if targets["count_by_model"] != {"vit_b16": 457, "deit_s16": 385}:
        raise EndpointE2Error("TARGET_COUNTS")
    scope = value["method_scope"]
    if scope != {
        "allowed": [METHOD_ID], "path_ig_allowed": False,
        "input_fidelity_allowed": False, "input_stability_allowed": False,
        "finite_intervention_allowed": False, "utility_claim_allowed": False,
    }:
        raise EndpointE2Error("METHOD_SCOPE")
    retention = value["retention"]
    if retention != {
        "target_summaries": "all_400_each_model",
        "raw_signed_rgb_class_maps": "all_400_each_model",
        "full_raw_atlas_evaluation_orders": list(range(16)),
        "gallery_evaluation_orders": list(range(8)),
        "near_zero_max_abs": 1e-12,
        "raw_dtype": "float32",
        "display_only_not_statistics": True,
    }:
        raise EndpointE2Error("RETENTION_SCOPE")
    runtime = value["runtime"]
    required_runtime = {
        "gpu_hours_cap": 64, "wall_hours_cap": 72, "incremental_retained_gib_cap": 128,
        "max_gpus": 8, "min_gpus": 1, "gpu_type": "NVIDIA H20",
        "min_free_mib": 32768, "max_utilization_percent": 10,
        "gpu_memory_cap_gib": 24, "cpu_rss_cap_gib": 32,
        "minimum_free_disk_gib": 50, "task_timeout_seconds": 1800,
        "claim_lease_seconds": 180, "heartbeat_seconds": 30, "max_attempts": 2,
        "dynamic_global_queue": True, "static_modulo_sharding": False,
        "single_multi_same_task_manifest": True, "single_multi_require_exact_map_digest": True,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "production_dtype": "float32",
        "tf32": False, "autocast": False, "deterministic_algorithms": True,
    }
    if runtime != required_runtime:
        raise EndpointE2Error("RUNTIME_SCOPE")
    if value["failure"] != {
        "missing_duplicate_nonfinite_source_drift_or_cap": "FAIL_CLOSED_PRESERVE_ROOT",
        "resource_cap": "PAUSED_RESOURCE_CAP",
        "next_on_pass": "ENDPOINT_E3_REQUIRES_EXPLICIT_USER_REQUEST",
    }:
        raise EndpointE2Error("FAILURE_SCOPE")


def registered_targets(
    rows: Sequence[Mapping[str, Any]], *, ground_truth_class_id: int,
) -> tuple[EndpointTarget, ...]:
    targets = []
    for row in rows:
        if set(row) != {"target_id", "module", "kind", "location", "channel", "token", "head"}:
            raise EndpointE2Error("TARGET_ROW_SCHEMA")
        kind = str(row["kind"])
        class_id = ground_truth_class_id if kind == "ground_truth_logit" else None
        targets.append(EndpointTarget(
            name=str(row["target_id"]), module=str(row["module"]), kind=kind,
            location=str(row["location"]), channel=int(row["channel"]),
            token=int(row["token"]), head=None if int(row["head"]) < 0 else int(row["head"]),
            class_id=class_id,
        ))
    if len({target.name for target in targets}) != len(targets):
        raise EndpointE2Error("TARGET_ID_DUPLICATE")
    return tuple(targets)


@dataclass(frozen=True)
class RegisteredScan:
    summaries: list[dict[str, Any]]
    class_maps: np.ndarray
    atlas_maps: np.ndarray | None
    logits_sha256: str
    clean_top1_class_id: int
    numeric_digest: str
    forward_count: int


def scan_registered_targets(
    model: nn.Module, inputs: Tensor, targets: Sequence[EndpointTarget], *,
    retain_full_atlas: bool, near_zero_max_abs: float = 1e-12,
    progress: Callable[[], None] | None = None,
) -> RegisteredScan:
    """Scan all registered targets while retaining only contracted raw maps."""
    if inputs.shape != (1, 3, 224, 224) or inputs.dtype is not torch.float32:
        raise EndpointE2Error("INPUT_SHAPE_OR_DTYPE")
    if not bool(torch.isfinite(inputs).all()) or near_zero_max_abs != 1e-12:
        raise EndpointE2Error("INPUT_FINITE_OR_NEAR_ZERO_CONTRACT")
    targets = tuple(targets)
    if not targets or len({target.name for target in targets}) != len(targets):
        raise EndpointE2Error("TARGET_SET")
    class_indices = [
        index for index, target in enumerate(targets)
        if target.kind in {"ground_truth_logit", "frozen_clean_top1_logit"}
    ]
    if len(class_indices) != 2:
        raise EndpointE2Error("CLASS_TARGET_COUNT")
    modules: dict[tuple[str, str], nn.Module] = {}
    for target in targets:
        if target.location not in {"input", "output"}:
            raise EndpointE2Error("TARGET_LOCATION")
        if target.module:
            modules[(target.module, target.location)] = model.get_submodule(target.module)
        elif target.kind not in {"ground_truth_logit", "frozen_clean_top1_logit"}:
            raise EndpointE2Error("TARGET_MODULE")
    captured: dict[tuple[str, str], Tensor] = {}
    clean = inputs.detach().clone().requires_grad_(True)
    handles = []

    def save(key: tuple[str, str], value: Tensor) -> None:
        if key in captured or not isinstance(value, Tensor):
            raise EndpointE2Error("HOOK_CAPTURE")
        captured[key] = value

    try:
        for key, module in modules.items():
            if key[1] == "input":
                def pre_hook(_module, args, key=key):
                    save(key, args[0])
                handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                def post_hook(_module, _args, output, key=key):
                    save(key, output)
                handles.append(module.register_forward_hook(post_hook))
        with ExitStack() as stack:
            stack.enter_context(_frozen_eval(model))
            stack.enter_context(torch.enable_grad())
            logits = model(clean)
            if logits.shape != (1, 1000) or not bool(torch.isfinite(logits).all()):
                raise EndpointE2Error("LOGITS")
            selected = []
            for target in targets:
                value = captured[(target.module, target.location)] if target.module else logits
                selected.append(_select(model, target, value))
            atlas = np.empty((len(targets), 3, 224, 224), dtype=np.float32) if retain_full_atlas else None
            class_maps = np.empty((2, 3, 224, 224), dtype=np.float32)
            class_position = {target_index: output_index for output_index, target_index in enumerate(class_indices)}
            summaries: list[dict[str, Any]] = []
            numeric = hashlib.sha256()
            for index, (resolved, score, _token_values) in enumerate(selected):
                if not bool(torch.isfinite(score).all()):
                    raise EndpointE2Error("NONFINITE_TARGET:" + resolved.name)
                gradient = torch.autograd.grad(
                    score.sum(), clean, retain_graph=index + 1 < len(selected), create_graph=False,
                )[0]
                attribution = (clean.detach() * gradient).detach()
                if not bool(torch.isfinite(attribution).all()):
                    raise EndpointE2Error("NONFINITE_MAP:" + resolved.name)
                array = np.ascontiguousarray(attribution[0].cpu().numpy(), dtype=np.float32)
                map_digest = array_sha256(array)
                flat = array.reshape(-1)
                signed_sum = math.fsum(map(float, flat))
                absolute_mass = math.fsum(map(float, np.abs(flat)))
                max_abs = float(np.max(np.abs(flat)))
                value_scalar = float(score.detach().cpu().item())
                values = (value_scalar, signed_sum, absolute_mass, max_abs)
                if not all(math.isfinite(item) for item in values):
                    raise EndpointE2Error("NONFINITE_SUMMARY:" + resolved.name)
                summary = {
                    "target_id": resolved.name, "kind": resolved.kind,
                    "module": resolved.module, "location": resolved.location,
                    "channel": int(resolved.channel), "token": int(resolved.token),
                    "head": -1 if resolved.head is None else int(resolved.head),
                    "resolved_class_id": -1 if resolved.class_id is None else int(resolved.class_id),
                    "value": value_scalar, "signed_sum": signed_sum,
                    "absolute_mass": absolute_mass, "max_abs": max_abs,
                    "near_zero": max_abs <= near_zero_max_abs, "finite": True,
                    "map_sha256": map_digest,
                }
                summaries.append(summary)
                numeric.update(canonical(summary))
                if atlas is not None:
                    atlas[index] = array
                if index in class_position:
                    class_maps[class_position[index]] = array
                del gradient, attribution, array
                if progress is not None:
                    progress()
            logits_array = np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float32)
            return RegisteredScan(
                summaries=summaries, class_maps=class_maps, atlas_maps=atlas,
                logits_sha256=array_sha256(logits_array),
                clean_top1_class_id=int(logits_array[0].argmax()),
                numeric_digest=numeric.hexdigest(), forward_count=1,
            )
    finally:
        for handle in handles:
            handle.remove()


def display_rgb_signed_sum(value: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Convert raw signed RGB attribution to an unsmoothed diverging RGB display."""
    if value.shape != (3, 224, 224) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise EndpointE2Error("DISPLAY_INPUT")
    signed = value.astype(np.float64).sum(axis=0)
    scale = float(np.max(np.abs(signed)))
    near_zero = scale <= 1e-12
    normalized = np.zeros_like(signed) if near_zero else signed / scale
    image = np.empty((224, 224, 3), dtype=np.uint8)
    positive = normalized >= 0
    magnitude = np.abs(normalized)
    image[..., 0] = np.where(positive, 255, 255 * (1 - magnitude)).round().astype(np.uint8)
    image[..., 1] = (255 * (1 - magnitude)).round().astype(np.uint8)
    image[..., 2] = np.where(positive, 255 * (1 - magnitude), 255).round().astype(np.uint8)
    return image, scale, near_zero


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1 or not all(math.isfinite(float(v)) for v in values):
        raise EndpointE2Error("PERCENTILE_INPUT")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
