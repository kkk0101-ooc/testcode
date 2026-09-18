"""Frozen ENDPOINT-E5 spatial ranking and isolated-head intervention logic.

The scientific unit returned by :func:`scan_e5_task` is one image-model task.
It deliberately stops at per-image C4b inputs: no bootstrap, hypothesis test,
Holm adjustment, or claim decision is implemented in this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .endpoint_e4_intervention import (
    EndpointE4Error, _check_model, capture_registered_edges, component_tensor,
    intervention_scores,
)
from .endpoint_nns import _frozen_eval
from .endpoint_vit import EndpointTarget, decompose_endpoint_vit, scan_endpoint_vit

BLOCK = 5
BUDGET_COUNTS = (0, 1, 2, 4)
AREAS = (0.0, 1.0 / 16.0, 2.0 / 16.0, 4.0 / 16.0)
REFERENCES = ("normalized_zero", "blur_rgb")
DETERMINISTIC_METHODS = ("endpoint_component", "total_gxi", "raw_gradient", "regional_taylor")
BASELINES = ("total_gxi", "raw_gradient", "regional_taylor", "random_regions")
HEADS_BY_MODEL = {"vit_b16": (0, 6, 11), "deit_s16": (0, 3, 5)}
RANDOM_REPEATS = 20
RANDOM_NAMESPACE = "NNS_VIT_ENDPOINT_E5_RANDOM_REGIONS_V1"


class EndpointE5Error(ValueError):
    """The E5 contract or a scientific numeric invariant was violated."""


def candidate_slices() -> tuple[tuple[slice, slice], ...]:
    return tuple(
        (slice(row * 56, (row + 1) * 56), slice(column * 56, (column + 1) * 56))
        for row in range(4) for column in range(4)
    )


def candidate_mask_digests() -> list[str]:
    output = []
    for ys, xs in candidate_slices():
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[ys, xs] = 1
        output.append(hashlib.sha256(mask.tobytes()).hexdigest())
    return output


def region_scores(value: Tensor) -> list[float]:
    """Sum absolute RGB-coordinate mass in the frozen 4x4 regions."""
    if tuple(value.shape) != (1, 3, 224, 224) or not bool(torch.isfinite(value).all()):
        raise EndpointE5Error("REGION_MAP_CONTRACT")
    absolute = value.detach().double().abs()
    return [float(absolute[:, :, ys, xs].sum()) for ys, xs in candidate_slices()]


def ranked_regions(scores: Sequence[float]) -> list[int]:
    if len(scores) != 16 or not all(math.isfinite(float(value)) for value in scores):
        raise EndpointE5Error("RANK_SCORE_CONTRACT")
    return sorted(range(16), key=lambda index: (-float(scores[index]), index))


def random_region_order(task_seed: int, model_id: str, reference_id: str, head: int, repeat: int) -> tuple[int, list[int]]:
    if model_id not in HEADS_BY_MODEL or reference_id not in REFERENCES or head not in HEADS_BY_MODEL[model_id]:
        raise EndpointE5Error("RANDOM_IDENTITY")
    if not 0 <= repeat < RANDOM_REPEATS:
        raise EndpointE5Error("RANDOM_REPEAT")
    payload = f"{RANDOM_NAMESPACE}|{task_seed}|{model_id}|{reference_id}|{head}|{repeat}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(16).tolist()
    return seed, [int(value) for value in order]


def normalized_auc(absolute_effects: Sequence[float]) -> float:
    if len(absolute_effects) != len(AREAS) or not all(math.isfinite(float(value)) and float(value) >= 0 for value in absolute_effects):
        raise EndpointE5Error("AUC_INPUT")
    total = 0.0
    for left in range(len(AREAS) - 1):
        total += (AREAS[left + 1] - AREAS[left]) * (float(absolute_effects[left]) + float(absolute_effects[left + 1])) / 2.0
    return total / 0.25


def replace_regions(clean: Tensor, reference: Tensor, selections: Sequence[Sequence[int]]) -> Tensor:
    if tuple(clean.shape) != (1, 3, 224, 224) or reference.shape != clean.shape:
        raise EndpointE5Error("PERTURBATION_INPUT")
    output = clean.detach().repeat(len(selections), 1, 1, 1)
    for batch_index, selected in enumerate(selections):
        if len(set(selected)) != len(selected) or any(not 0 <= int(index) < 16 for index in selected):
            raise EndpointE5Error("PERTURBATION_SELECTION")
        for index in selected:
            ys, xs = candidate_slices()[int(index)]
            output[batch_index, :, ys, xs] = reference[0, :, ys, xs]
    return output


@contextmanager
def _capture_block5_heads(model: nn.Module) -> Iterator[dict[str, Tensor]]:
    observed: dict[str, Tensor] = {}
    def hook(_module, args):
        if "heads" in observed:
            raise EndpointE5Error("BLOCK5_HEAD_CAPTURE_DUPLICATE")
        observed["heads"] = args[0]
    handle = model.blocks[BLOCK].attn.proj.register_forward_pre_hook(hook)
    try:
        yield observed
    finally:
        handle.remove()


def _perturbed_outputs(
    model: nn.Module, clean: Tensor, reference: Tensor, selections: Sequence[Sequence[int]],
    *, class_id: int, batch_size: int, progress=None,
) -> tuple[list[float], list[Tensor]]:
    if batch_size <= 0:
        raise EndpointE5Error("PERTURBED_BATCH_SIZE")
    scores: list[float] = []
    heads: list[Tensor] = []
    for start in range(0, len(selections), batch_size):
        batch = replace_regions(clean, reference, selections[start:start + batch_size])
        with _frozen_eval(model), torch.no_grad(), _capture_block5_heads(model) as observed:
            logits = model(batch)
        if "heads" not in observed or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(observed["heads"]).all()):
            raise EndpointE5Error("PERTURBED_NONFINITE_OR_CAPTURE")
        scores.extend(float(value) for value in logits[:, int(class_id)])
        heads.extend(value.detach().cpu() for value in observed["heads"][:, None])
        if progress is not None:
            progress()
    if len(scores) != len(selections) or len(heads) != len(selections):
        raise EndpointE5Error("PERTURBED_COVERAGE")
    return scores, heads


def _curve_specs(
    model_id: str, reference_id: str, task_seed: int, rankings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    specs = []
    for head in HEADS_BY_MODEL[model_id]:
        for method in DETERMINISTIC_METHODS:
            key = f"head_{head}" if method in {"endpoint_component", "regional_taylor"} else "shared"
            specs.append({"curve_id": f"{method}.head_{head}", "method": method, "head": head,
                          "repeat": None, "random_seed": None, "ranking": list(rankings[method][key])})
        for repeat in range(RANDOM_REPEATS):
            seed, order = random_region_order(task_seed, model_id, reference_id, head, repeat)
            specs.append({"curve_id": f"random_regions.head_{head}.repeat_{repeat:02d}",
                          "method": "random_regions", "head": head, "repeat": repeat,
                          "random_seed": seed, "ranking": order})
    if len(specs) != 72 or len({row["curve_id"] for row in specs}) != 72:
        raise EndpointE5Error("CURVE_SPEC_COVERAGE")
    return specs


def _isolate_head_scores(
    model: nn.Module, clean: Tensor, *, class_id: int, conditions: Sequence[Mapping[str, Any]],
    perturbed_heads: Sequence[Tensor], batch_size: int, progress=None,
) -> tuple[float, list[float]]:
    outputs: list[float] = []
    observed_clean: float | None = None
    num_heads = int(model.blocks[BLOCK].attn.num_heads)
    for start in range(0, len(conditions), batch_size):
        part = conditions[start:start + batch_size]
        rows = []
        replacements = {}
        for offset, condition in enumerate(part, start):
            component_id = f"spatial_{offset:03d}"
            head = int(condition["head"]); edge = perturbed_heads[offset]
            width = edge.shape[-1] // num_heads
            rows.append({"component_id": component_id, "operation": "head", "block": BLOCK,
                         "head": head, "num_heads": num_heads})
            replacements[component_id] = edge[:, :, head * width:(head + 1) * width]
        clean_score, scores, _ = intervention_scores(
            model, clean, class_id=class_id, candidates=rows, replacements=replacements,
        )
        if observed_clean is None:
            observed_clean = clean_score
        elif abs(observed_clean - clean_score) > 1e-5 + 1e-4 * abs(observed_clean):
            raise EndpointE5Error("ISOLATED_CLEAN_DRIFT")
        outputs.extend(scores)
        if progress is not None:
            progress()
    if observed_clean is None or len(outputs) != len(conditions):
        raise EndpointE5Error("ISOLATED_COVERAGE")
    return observed_clean, outputs


def scan_e5_task(
    model: nn.Module, inputs: Tensor, references: Mapping[str, Tensor], *, class_id: int,
    task_seed: int, model_id: str, perturbed_batch_size: int = 16, isolation_batch_size: int = 8,
    retain_raw_maps: bool = False, progress=None,
) -> dict[str, Any]:
    """Produce all frozen spatial curves and image-level C4b inputs."""
    _check_model(model)
    if model_id not in HEADS_BY_MODEL or set(references) != set(REFERENCES):
        raise EndpointE5Error("MODEL_OR_REFERENCE_CONTRACT")
    if tuple(inputs.shape) != (1, 3, 224, 224) or inputs.dtype is not torch.float32:
        raise EndpointE5Error("INPUT_CONTRACT")
    heads = HEADS_BY_MODEL[model_id]
    if int(model.blocks[BLOCK].attn.num_heads) <= max(heads):
        raise EndpointE5Error("HEAD_REGISTRY_DRIFT")

    target = EndpointTarget("e5_ground_truth_logit", "", kind="ground_truth_logit", class_id=int(class_id))
    decomposition = decompose_endpoint_vit(model, inputs, block=BLOCK, target=target, atol=1e-5, rtol=1e-4)
    endpoint = scan_endpoint_vit(model, inputs, (target,))
    unit = endpoint.units[target.name]
    clean_capture = capture_registered_edges(model, inputs, class_id=class_id, enable_grad=True)
    clean_score = float(clean_capture.score.detach())
    clean_predicted_class = int(clean_capture.logits.detach().argmax())
    clean_edge = clean_capture.edges[BLOCK]["heads"].detach()
    edge_gradient = torch.autograd.grad(clean_capture.score.sum(), clean_capture.edges[BLOCK]["heads"])[0].detach()
    del clean_capture
    for observed in (float(decomposition.value), float(unit.value), clean_score):
        if abs(observed - clean_score) > 1e-5 + 1e-4 * abs(clean_score):
            raise EndpointE5Error("CLEAN_SCORE_PARITY")

    component_maps = {head: decomposition.components[f"head_{head}"].to(torch.float32) for head in heads}
    total_map = decomposition.full.to(torch.float32)
    raw_gradient = unit.gradient.to(torch.float32)
    endpoint_scores = {f"head_{head}": region_scores(component_maps[head]) for head in heads}
    total_scores = region_scores(total_map); gradient_scores = region_scores(raw_gradient)
    all_references: dict[str, Any] = {}

    for reference_id in REFERENCES:
        reference = references[reference_id]
        singleton = [[index] for index in range(16)]
        candidate_full_scores, candidate_heads = _perturbed_outputs(
            model, inputs, reference, singleton, class_id=class_id, batch_size=perturbed_batch_size, progress=progress,
        )
        regional_scores: dict[str, list[float]] = {}
        width = clean_edge.shape[-1] // int(model.blocks[BLOCK].attn.num_heads)
        for head in heads:
            lo, hi = head * width, (head + 1) * width
            gradient = edge_gradient[:, :, lo:hi].detach().cpu().double()
            clean_head = clean_edge[:, :, lo:hi].detach().cpu().double()
            values = []
            for perturbed in candidate_heads:
                displacement = clean_head - perturbed[:, :, lo:hi].double()
                values.append(abs(float((gradient * displacement).sum())))
            regional_scores[f"head_{head}"] = values
        rankings = {
            "endpoint_component": {key: ranked_regions(value) for key, value in endpoint_scores.items()},
            "total_gxi": {"shared": ranked_regions(total_scores)},
            "raw_gradient": {"shared": ranked_regions(gradient_scores)},
            "regional_taylor": {key: ranked_regions(value) for key, value in regional_scores.items()},
        }
        specs = _curve_specs(model_id, reference_id, task_seed, rankings)
        unique_specs = []
        seen_selections = set()
        for spec in specs:
            for count in BUDGET_COUNTS[1:]:
                selected_regions = tuple(spec["ranking"][:count])
                if selected_regions not in seen_selections:
                    seen_selections.add(selected_regions); unique_specs.append(selected_regions)
        multi_specs = [selected for selected in unique_specs if len(selected) > 1]
        multi_full_scores, multi_heads = _perturbed_outputs(
            model, inputs, reference, multi_specs,
            class_id=class_id, batch_size=perturbed_batch_size, progress=progress,
        )
        cached = {(index,): (candidate_full_scores[index], candidate_heads[index]) for index in range(16)}
        cached.update({selected: (score, head) for selected, score, head in zip(multi_specs, multi_full_scores, multi_heads)})
        conditions = []
        perturbed_scores = []
        perturbed_heads = []
        for spec in specs:
            for count in BUDGET_COUNTS[1:]:
                selected_regions = tuple(spec["ranking"][:count]); full_score, perturbed_head = cached[selected_regions]
                conditions.append({"curve_id": spec["curve_id"], "method": spec["method"],
                                   "head": spec["head"], "repeat": spec["repeat"],
                                   "budget_count": count, "selected_regions": list(selected_regions)})
                perturbed_scores.append(full_score); perturbed_heads.append(perturbed_head)
        isolated_clean, isolated_scores = _isolate_head_scores(
            model, inputs, class_id=class_id, conditions=conditions, perturbed_heads=perturbed_heads,
            batch_size=isolation_batch_size, progress=progress,
        )
        if abs(isolated_clean - clean_score) > 1e-5 + 1e-4 * abs(clean_score):
            raise EndpointE5Error("ISOLATED_CLEAN_SCORE_PARITY")
        by_curve: dict[str, dict[int, tuple[float, float]]] = {}
        for condition, full_score, isolated_score in zip(conditions, perturbed_scores, isolated_scores):
            by_curve.setdefault(condition["curve_id"], {})[int(condition["budget_count"])] = (full_score, isolated_score)
        curves = []
        for spec in specs:
            points = [{"budget_count": 0, "actual_area_fraction": 0.0, "selected_regions": [],
                       "isolated_score": clean_score, "isolated_effect": 0.0,
                       "full_forward_score": clean_score, "full_forward_effect": 0.0}]
            for count, area in zip(BUDGET_COUNTS[1:], AREAS[1:]):
                full_score, isolated_score = by_curve[spec["curve_id"]][count]
                points.append({"budget_count": count, "actual_area_fraction": area,
                               "selected_regions": spec["ranking"][:count],
                               "isolated_score": isolated_score, "isolated_effect": clean_score - isolated_score,
                               "full_forward_score": full_score, "full_forward_effect": clean_score - full_score})
            curves.append({**spec, "points": points,
                           "absolute_isolated_effect_auc_div_0_25": normalized_auc([abs(row["isolated_effect"]) for row in points])})
        method_scores = {}
        for method in DETERMINISTIC_METHODS:
            selected = [row["absolute_isolated_effect_auc_div_0_25"] for row in curves if row["method"] == method]
            if len(selected) != 3:
                raise EndpointE5Error("DETERMINISTIC_CURVE_COVERAGE")
            method_scores[method] = float(np.mean(selected))
        random_head_scores = []
        random_aggregates = []
        for head in heads:
            selected = [row for row in curves if row["method"] == "random_regions" and row["head"] == head]
            if len(selected) != RANDOM_REPEATS:
                raise EndpointE5Error("RANDOM_CURVE_COVERAGE")
            mean_signed = [float(np.mean([row["points"][index]["isolated_effect"] for row in selected])) for index in range(4)]
            score = normalized_auc([abs(value) for value in mean_signed]); random_head_scores.append(score)
            random_aggregates.append({"head": head, "repeat_count": RANDOM_REPEATS,
                                      "mean_signed_isolated_effect_curve": mean_signed,
                                      "absolute_of_mean_isolated_effect_curve": [abs(value) for value in mean_signed],
                                      "absolute_isolated_effect_auc_div_0_25": score})
        method_scores["random_regions"] = float(np.mean(random_head_scores))
        all_references[reference_id] = {
            "ranking_scores": {"endpoint_component": endpoint_scores, "total_gxi": total_scores,
                               "raw_gradient": gradient_scores, "regional_taylor": regional_scores},
            "candidate_single_region_full_forward_scores": candidate_full_scores,
            "candidate_single_region_full_forward_effects": [clean_score - value for value in candidate_full_scores],
            "unique_multi_region_full_forward_input_count": len(multi_specs),
            "curves": curves, "random_aggregates": random_aggregates,
            "image_method_scores": method_scores,
            "paired_endpoint_minus_baseline": {baseline: method_scores["endpoint_component"] - method_scores[baseline] for baseline in BASELINES},
        }

    raw_maps = None
    if retain_raw_maps:
        raw_maps = {"total_gxi": total_map.numpy(), "raw_gradient": raw_gradient.numpy(),
                    **{f"head_{head}": component_maps[head].numpy() for head in heads}}
    return {
        "block": BLOCK, "heads": list(heads), "clean_score": clean_score,
        "clean_predicted_class": clean_predicted_class, "candidate_mask_sha256": candidate_mask_digests(),
        "budgets": list(BUDGET_COUNTS), "actual_area_fractions": list(AREAS),
        "references": all_references, "raw_maps": raw_maps,
        "counts": {"candidate_region_count": 16, "reference_count": 2,
                   "selected_head_count": 3, "curve_count": 144,
                   "random_curve_count": 120, "candidate_taylor_forward_evaluations": 32,
                   "finite_union_full_forward_curve_points": 432,
                   "finite_union_isolated_head_evaluations": 432},
    }


def validate_run_spec(value: Mapping[str, Any]) -> None:
    required = {"schema_version", "stage", "authorization_token", "predecessors", "cohort",
                "models", "target", "spatial_intervention", "method_scope", "runtime",
                "outputs", "failure"}
    if set(value) != required or value["schema_version"] != 1 or value["stage"] != "ENDPOINT-E5":
        raise EndpointE5Error("RUN_SPEC_SCHEMA")
    if value["authorization_token"] != "ENDPOINT_E5_EXPLICIT_USER_REQUEST_20260918":
        raise EndpointE5Error("AUTHORIZATION")
    if value["cohort"] != {"split": "evaluation", "count": 400, "models_share_rows": True,
                              "source": "ENDPOINT-E2_authoritative_materialization", "replacement_allowed": False}:
        raise EndpointE5Error("COHORT")
    if value["models"] != ["vit_b16", "deit_s16"] or value["target"] != {"kind": "ground_truth_logit", "class_source": "frozen_EP0_class_id"}:
        raise EndpointE5Error("MODEL_OR_TARGET")
    expected_spatial = {
        "block": BLOCK, "heads_by_model": {key: list(item) for key, item in HEADS_BY_MODEL.items()},
        "candidate_regions": "4x4_nonoverlapping_56x56_row_major_full_224_crop",
        "budget_counts": list(BUDGET_COUNTS), "actual_area_fractions": list(AREAS),
        "references": list(REFERENCES), "primary_ranker": "component_map_RGB_coordinate_absolute_mass",
        "mandatory_baselines": list(BASELINES), "random_repeats": RANDOM_REPEATS,
        "random_seed_namespace": RANDOM_NAMESPACE,
        "ranking_tie_break": "descending_score_then_ascending_row_major_candidate_index",
        "regional_taylor": "abs_dot_clean_head_cotangent_clean_minus_single_region_head_activation",
        "isolated_effect": "clean_logit_minus_selected_head_replacement_logit_other_heads_and_identity_clean",
        "full_forward_effect": "descriptive_only_clean_logit_minus_region_replaced_input_logit",
        "primary_score": "mean_three_heads_trapezoid_absolute_isolated_effect_over_actual_area_div_0.25",
        "random_aggregation": "mean_signed_effect_curve_over_20_permutations_within_head_then_absolute_auc_then_mean_three_heads",
        "effect_based_region_selection_allowed": False,
    }
    if value["spatial_intervention"] != expected_spatial:
        raise EndpointE5Error("SPATIAL_CONTRACT")
    if value["method_scope"] != {"c4b_per_image_inputs_allowed": True, "descriptive_aggregate_allowed": True,
                                     "bootstrap_allowed": False, "holm_allowed": False,
                                     "final_claim_decision_allowed": False, "path_ig_allowed": False,
                                     "input_fidelity_or_stability_allowed": False, "endpoint_e6_allowed": False}:
        raise EndpointE5Error("METHOD_SCOPE")
    runtime = value["runtime"]
    expected_runtime = {"combined_e4_e5_gpu_hours_cap": 136, "combined_e4_e5_wall_hours_cap": 132,
                        "combined_e4_e5_incremental_retained_gib_cap": 56, "max_gpus": 8, "min_gpus": 1,
                        "gpu_type": "NVIDIA H20", "min_free_mib": 32768, "max_utilization_percent": 10,
                        "gpu_memory_cap_gib": 24, "cpu_rss_cap_gib": 32, "minimum_free_disk_gib": 50,
                        "task_timeout_seconds": 1800, "claim_lease_seconds": 180, "heartbeat_seconds": 30,
                        "max_attempts": 2, "dynamic_global_queue": True, "static_modulo_sharding": False,
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "production_dtype": "float32",
                        "tf32": False, "autocast": False, "deterministic_algorithms": True}
    if runtime != expected_runtime:
        raise EndpointE5Error("RUNTIME")
    if value["outputs"] != {"formal_root": "reports/nns_vit_endpoint_e5_spatial_r1_20260918",
                               "source_materialization": "reports/nns_vit_endpoint_e2_atlas_r1_20260918/materialization_receipt.json",
                               "raw_maps": "first16_evaluation_orders_per_model_npz"}:
        raise EndpointE5Error("OUTPUTS")
    if value["failure"] != {"missing_duplicate_nonfinite_source_drift_protocol_or_cap": "FAIL_CLOSED_PRESERVE_ROOT",
                               "resource_cap": "PAUSED_RESOURCE_CAP",
                               "next_on_valid_execution": "ENDPOINT_E6_REQUIRES_EXPLICIT_USER_REQUEST"}:
        raise EndpointE5Error("FAILURE")
    expected_predecessors = {"cohort_registry_sha256", "endpoint_e4_r3_terminal_sha256",
                             "endpoint_e4_r3_validation_sha256", "intervention_registry_sha256",
                             "materialization_receipt_sha256", "model_registry_sha256",
                             "statistical_analysis_plan_sha256", "target_registry_sha256",
                             "unified_plan_sha256"}
    if set(value["predecessors"]) != expected_predecessors or not all(isinstance(item, str) and len(item) == 64 for item in value["predecessors"].values()):
        raise EndpointE5Error("PREDECESSORS")
