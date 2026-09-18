"""Closed scientific primitives for ENDPOINT-E3 structural closure.

This module evaluates only first-order Endpoint edge-cut correctness.  It does
not implement path integration, input perturbation, activation replacement, or
any utility analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .endpoint_e2_atlas import array_sha256, canonical, digest
from .endpoint_vit import (
    EndpointTarget, decompose_endpoint_vit, numeric_parity,
    validate_decomposition_cut,
)


AUTHORIZATION = "ENDPOINT_E3_FORMAL_CLOSURE_EXPLICIT_USER_REQUEST"
PASS_STATUS = "ENDPOINT_E3_PASS_CORRECTNESS_ONLY_NEXT_ENDPOINT_E4_REQUIRES_EXPLICIT_REQUEST"
CLAIM_DECISION = "RQ2_STRUCTURAL_EDGE_CUT_CLOSURE_CORRECTNESS"
CUT_ALIASES = {
    "heads_branch": "heads_branch",
    "heads_total": "heads_with_bypass",
    "qkv_branch": "qkv_branch",
    "routing_value_branch": "routing_value_branch",
    "qkv_total": "qkv_with_bypass",
    "routing_value_total": "routing_value_with_bypass",
    "residual1": "residual1",
    "residual2": "residual2",
}
EXPECTED_COMPONENT_PATTERNS = {
    "heads_branch": "head_0..H-1",
    "heads_total": "identity + head_0..H-1",
    "qkv_branch": "Q+K+V",
    "routing_value_branch": "routing+V",
    "qkv_total": "identity+Q+K+V",
    "routing_value_total": "identity+routing+V",
    "residual1": "identity+attention",
    "residual2": "carry+MLP",
}


class EndpointE3Error(ValueError):
    """An ENDPOINT-E3 contract was violated."""


def validate_run_spec(value: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version", "stage", "authorization_token", "claim_decision",
        "predecessors", "cohort", "models", "target", "decomposition",
        "numeric", "negative_controls", "retention", "method_scope",
        "runtime", "outputs", "failure",
    }
    if set(value) != expected_top:
        raise EndpointE3Error("RUN_SPEC_KEYS")
    if value["schema_version"] != 1 or value["stage"] != "ENDPOINT-E3":
        raise EndpointE3Error("RUN_SPEC_IDENTITY")
    if value["authorization_token"] != AUTHORIZATION or value["claim_decision"] != CLAIM_DECISION:
        raise EndpointE3Error("AUTHORIZATION_OR_CLAIM")
    if value["cohort"] != {
        "split": "evaluation", "count": 400, "models_share_rows": True,
        "replacement_allowed": False, "source": "ENDPOINT-E2_authoritative_materialization",
    }:
        raise EndpointE3Error("COHORT_SCOPE")
    if value["models"] != ["vit_b16", "deit_s16"]:
        raise EndpointE3Error("MODEL_SCOPE")
    if value["target"] != {
        "kind": "ground_truth_logit", "class_source": "frozen_EP0_class_id",
        "margin_allowed": False,
    }:
        raise EndpointE3Error("TARGET_SCOPE")
    decomposition = value["decomposition"]
    if set(decomposition) != {"blocks", "cuts", "backend", "reconstruction_ids"}:
        raise EndpointE3Error("DECOMPOSITION_KEYS")
    if decomposition["blocks"] != list(range(12)):
        raise EndpointE3Error("BLOCK_SCOPE")
    expected_cuts = [
        {"cut_id": name, "components": EXPECTED_COMPONENT_PATTERNS[name],
         "reconstruction": "A_F" if name.endswith("branch") else "L_F"}
        for name in CUT_ALIASES
    ]
    if decomposition["cuts"] != expected_cuts:
        raise EndpointE3Error("CUT_SCOPE")
    if decomposition["backend"] != "timm_0.9.16_standard_Block_Attention_explicit_unfused":
        raise EndpointE3Error("BACKEND_SCOPE")
    if decomposition["reconstruction_ids"] != ["A_F", "L_F"]:
        raise EndpointE3Error("RECONSTRUCTION_SCOPE")
    if value["numeric"] != {
        "dtype": "float32", "atol": 1e-5, "rtol": 1e-4,
        "near_zero_norm": 1e-6, "all_raw_values_finite": True,
        "acceptance": "elementwise_and_absolute_or_relative_branch",
    }:
        raise EndpointE3Error("NUMERIC_SCOPE")
    if value["negative_controls"] != [
        "omit_largest_nonzero_head_from_A_F",
        "omit_nonzero_routing_from_A_F",
        "duplicate_identity_in_residual1",
    ]:
        raise EndpointE3Error("NEGATIVE_CONTROL_SCOPE")
    if value["retention"] != {
        "all_task_block_cut_and_component_summaries": True,
        "raw_component_maps_evaluation_orders": list(range(16)),
        "raw_dtype": "float32", "display_outputs": False,
    }:
        raise EndpointE3Error("RETENTION_SCOPE")
    if value["method_scope"] != {
        "endpoint_structure_closure_allowed": True, "path_ig_allowed": False,
        "input_fidelity_allowed": False, "input_stability_allowed": False,
        "activation_replacement_allowed": False, "utility_claim_allowed": False,
    }:
        raise EndpointE3Error("METHOD_SCOPE")
    if value["runtime"] != {
        "gpu_hours_cap": 64, "wall_hours_cap": 72, "incremental_retained_gib_cap": 128,
        "max_gpus": 8, "min_gpus": 1, "gpu_type": "NVIDIA H20",
        "min_free_mib": 32768, "max_utilization_percent": 10,
        "gpu_memory_cap_gib": 24, "cpu_rss_cap_gib": 32,
        "minimum_free_disk_gib": 50, "task_timeout_seconds": 1800,
        "claim_lease_seconds": 180, "heartbeat_seconds": 30, "max_attempts": 2,
        "dynamic_global_queue": True, "static_modulo_sharding": False,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "production_dtype": "float32",
        "tf32": False, "autocast": False, "deterministic_algorithms": True,
    }:
        raise EndpointE3Error("RUNTIME_SCOPE")
    if value["failure"] != {
        "missing_duplicate_nonfinite_source_drift_or_cap": "FAIL_CLOSED_PRESERVE_ROOT",
        "resource_cap": "PAUSED_RESOURCE_CAP",
        "next_on_pass": "ENDPOINT_E4_REQUIRES_EXPLICIT_USER_REQUEST",
    }:
        raise EndpointE3Error("FAILURE_SCOPE")


def _summary(value: Tensor) -> dict[str, Any]:
    array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = np.ascontiguousarray(array[0])
    flat = array.reshape(-1)
    result = {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "signed_sum": math.fsum(map(float, flat)),
        "absolute_mass": math.fsum(map(float, np.abs(flat))),
        "max_abs": float(np.max(np.abs(flat))), "l2_norm": float(np.linalg.norm(flat)),
        "array_sha256": array_sha256(array), "finite": bool(np.isfinite(array).all()),
    }
    if not result["finite"] or not all(math.isfinite(float(result[key])) for key in ("signed_sum", "absolute_mass", "max_abs", "l2_norm")):
        raise EndpointE3Error("NONFINITE_COMPONENT_SUMMARY")
    return result


def _negative_controls(result, *, atol: float, rtol: float) -> list[dict[str, Any]]:
    heads = sorted(
        (name for name in result.components if name.startswith("head_")),
        key=lambda name: int(name.split("_")[1]),
    )
    omitted = max(heads, key=lambda name: float(result.components[name].abs().max()))
    controls = []
    cases = [
        ("omit_largest_nonzero_head_from_A_F",
         tuple(name for name in heads if name != omitted), result.reconstructions["A_F"], "A_F",
         omitted, float(result.components[omitted].abs().max()), "CUT_CLOSURE_FAILED:A_F"),
        ("omit_nonzero_routing_from_A_F", ("value",), result.reconstructions["A_F"], "A_F",
         "routing", float(result.components["routing"].abs().max()), "CUT_CLOSURE_FAILED:A_F"),
        ("duplicate_identity_in_residual1", ("identity", "attention", "identity"),
         result.reconstructions["L_F"], "L_F", "identity",
         float(result.components["identity"].abs().max()), "DUPLICATE_OR_EMPTY_CUT"),
    ]
    for control_id, keys, reference, reconstruction_id, affected, magnitude, expected in cases:
        if magnitude <= 0:
            raise EndpointE3Error("NEGATIVE_CONTROL_COMPONENT_NOT_NONZERO:" + control_id)
        try:
            validate_decomposition_cut(
                components=result.components, keys=keys, reference=reference,
                reconstruction_id=reconstruction_id, atol=0.0, rtol=0.0,
            )
        except ValueError as error:
            observed = str(error)
            if not observed.startswith(expected):
                raise EndpointE3Error("NEGATIVE_CONTROL_WRONG_FAILURE:" + control_id) from error
        else:
            raise EndpointE3Error("NEGATIVE_CONTROL_FALSE_PASS:" + control_id)
        controls.append({
            "control_id": control_id, "affected_component": affected,
            "affected_component_max_abs": magnitude, "expected_failure": expected,
            "observed_failure": observed, "passed": True,
            "checker_fixture_atol": 0.0, "checker_fixture_rtol": 0.0,
            "formal_positive_atol": atol, "formal_positive_rtol": rtol,
        })
    return controls


@dataclass(frozen=True)
class ClosureScan:
    blocks: list[dict[str, Any]]
    raw_blocks: list[np.ndarray] | None
    raw_component_orders: list[list[str]] | None
    numeric_digest: str


def scan_structural_closure(
    model: nn.Module, inputs: Tensor, *, ground_truth_class_id: int,
    blocks: Sequence[int] = tuple(range(12)), atol: float = 1e-5, rtol: float = 1e-4,
    near_zero_norm: float = 1e-6, retain_raw: bool = False, progress=None,
) -> ClosureScan:
    if inputs.shape != (1, 3, 224, 224) or inputs.dtype is not torch.float32:
        raise EndpointE3Error("INPUT_SHAPE_OR_DTYPE")
    if not bool(torch.isfinite(inputs).all()):
        raise EndpointE3Error("INPUT_NONFINITE")
    if tuple(blocks) != tuple(range(12)) or len(model.blocks) != 12:
        raise EndpointE3Error("ALL_BLOCK_SCOPE_REQUIRED")
    if not 0 <= ground_truth_class_id < 1000:
        raise EndpointE3Error("GROUND_TRUTH_CLASS_ID")
    if (atol, rtol, near_zero_norm) != (1e-5, 1e-4, 1e-6):
        raise EndpointE3Error("FLOAT32_TOLERANCE_DRIFT")
    target = EndpointTarget(
        "ground_truth_logit", "", kind="ground_truth_logit",
        class_id=int(ground_truth_class_id),
    )
    block_rows: list[dict[str, Any]] = []
    raw_blocks: list[np.ndarray] | None = [] if retain_raw else None
    raw_orders: list[list[str]] | None = [] if retain_raw else None
    for block in blocks:
        decomposition = decompose_endpoint_vit(
            model, inputs, block=int(block), target=target, atol=atol, rtol=rtol,
        )
        component_rows = {name: _summary(value) for name, value in sorted(decomposition.components.items())}
        reconstruction_rows = {name: _summary(value) for name, value in sorted(decomposition.reconstructions.items())}
        cut_rows = []
        for cut_id, internal_id in CUT_ALIASES.items():
            keys = decomposition.cuts[internal_id]
            reconstruction_id = decomposition.cut_reconstruction[internal_id]
            reference = decomposition.reconstructions[reconstruction_id]
            summed = sum((decomposition.components[key] for key in keys), torch.zeros_like(reference))
            parity = numeric_parity(
                summed, reference, atol=atol, rtol=rtol, near_zero_norm=near_zero_norm,
            )
            if not parity["pass"]:
                raise EndpointE3Error(f"CUT_CLOSURE_FAILED:{block}:{cut_id}")
            cut_rows.append({
                "cut_id": cut_id, "component_ids": list(keys),
                "reconstruction_id": reconstruction_id, "pass": True, **parity,
                "summed_sha256": _summary(summed)["array_sha256"],
                "reference_sha256": reconstruction_rows[reconstruction_id]["array_sha256"],
            })
        negatives = _negative_controls(decomposition, atol=atol, rtol=rtol)
        block_row = {
            "block": int(block), "target_kind": decomposition.target.kind,
            "resolved_class_id": int(decomposition.target.class_id),
            "target_value": float(decomposition.value.item()),
            "captured_edges": list(decomposition.captured_edges),
            "components": component_rows, "reconstructions": reconstruction_rows,
            "cuts": cut_rows, "negative_controls": negatives,
        }
        if not math.isfinite(block_row["target_value"]):
            raise EndpointE3Error("NONFINITE_TARGET_VALUE")
        block_row["canonical_digest"] = digest(block_row)
        block_rows.append(block_row)
        if retain_raw:
            assert raw_blocks is not None and raw_orders is not None
            order = sorted(decomposition.components) + ["A_F", "L_F"]
            values = [decomposition.components[name] for name in sorted(decomposition.components)]
            values += [decomposition.reconstructions["A_F"], decomposition.reconstructions["L_F"]]
            raw_blocks.append(np.stack([value.numpy()[0] for value in values]).astype(np.float32, copy=False))
            raw_orders.append(order)
        del decomposition
        if progress is not None:
            progress()
    return ClosureScan(
        blocks=block_rows, raw_blocks=raw_blocks, raw_component_orders=raw_orders,
        numeric_digest=digest(block_rows),
    )
