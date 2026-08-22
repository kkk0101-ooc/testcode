from __future__ import annotations

import dataclasses
import json
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
import torch

import vitnns.stage_c_fidelity as fidelity
from vitnns.stage_c_fidelity import (
    CLASS_TARGET_REF,
    CURVE_FRACTIONS,
    MATCHED_METHOD_ID,
    NNS_METHOD_ID,
    VALUE_TARGET_REF,
    CurveDirection,
    FidelityContractError,
    FidelityScale,
    RankingKind,
    TIE_REPLICATES,
    TargetResponse,
    UnitMassMap,
    ValuePathDeltaFidelity,
    aggregate_five_curves,
    actual_curve_fractions,
    build_intervention_schedule,
    build_method_intervention_schedule,
    center_distance_ranking,
    curve_unit_counts,
    evaluate_fidelity_curve,
    fidelity_seed,
    fidelity_target_provenance_binding,
    global_random_ranking,
    method_unit_masses,
    paired_rankings_for_method,
    random_rankings,
    resolve_target_response,
    target_oriented_scores,
    target_response_from_values,
    tensor_sha256,
    tie_permuted_ranking,
    tie_priority_sha256,
    top20_from_curves,
    trapezoid_auc,
    value_path_delta_fidelity,
)
from vitnns.stage_c_maps import (
    ComparisonLine,
    MethodResult,
    NAReason,
    SignSemantics,
    StructuredNA,
    patch_to_fine_cell_mass,
    pixel_to_patch_mass,
    pixel_to_fine_cell_mass,
    raw_array,
    signed_rgb_channel_sum,
    stage_c_metric_ready,
)


def _baseline_inputs() -> torch.Tensor:
    return torch.zeros((1, 3, 224, 224), dtype=torch.float32)


def _unit_tensor(
    scale: FidelityScale, active_units: list[int], value: float = 1.0
) -> torch.Tensor:
    tensor = _baseline_inputs()
    cell = scale.unit_size
    for unit in active_units:
        row, col = divmod(unit, scale.grid_size)
        r0 = row * cell
        c0 = col * cell
        tensor[:, :, r0 : r0 + cell, c0 : c0 + cell] = value
    return tensor


def _frozen_target_payload(
    line: ComparisonLine,
    clean_inputs: torch.Tensor,
    *,
    label: str,
) -> dict[str, Any]:
    clean_sha256 = tensor_sha256(clean_inputs)
    if line is ComparisonLine.VALUE_PATH:
        return {
            "target_kind": VALUE_TARGET_REF,
            "clean_input_sha256": clean_sha256,
            "runtime_model_binding_sha256": "1" * 64,
            "block_index": 3,
            "head_index": 5,
            "query_token": 0,
            "label": label,
        }
    return {
        "target_kind": CLASS_TARGET_REF,
        "clean_input_sha256": clean_sha256,
        "runtime_model_binding_sha256": "2" * 64,
        "class_index": 7,
        "label": label,
    }


def _canonical_target_identity(
    line: ComparisonLine,
    clean_inputs: torch.Tensor,
    *,
    label: str,
) -> dict[str, Any]:
    target_ref = VALUE_TARGET_REF if line is ComparisonLine.VALUE_PATH else CLASS_TARGET_REF
    return {
        "comparison_line": line.value,
        "target_ref": target_ref,
        "clean_input_sha256": tensor_sha256(clean_inputs),
        "frozen_target": _frozen_target_payload(line, clean_inputs, label=label),
    }


class ToyEvaluator:
    def __init__(
        self,
        *,
        comparison_line: ComparisonLine,
        target_ref: str | None = None,
        target_identity: dict[str, Any],
        mutate_inputs: bool = False,
        mutate_identity_after_first_call: bool = False,
    ) -> None:
        self._comparison_line = comparison_line
        self._target_ref = target_ref or (
            VALUE_TARGET_REF if comparison_line is ComparisonLine.VALUE_PATH else CLASS_TARGET_REF
        )
        self._target_identity = dict(target_identity)
        self._mutate_inputs = mutate_inputs
        self._mutate_identity_after_first_call = mutate_identity_after_first_call
        self._calls = 0

    @property
    def comparison_line(self) -> ComparisonLine:
        return self._comparison_line

    @property
    def target_ref(self) -> str:
        return self._target_ref

    @property
    def target_identity(self) -> dict[str, Any]:
        return self._target_identity

    def evaluate(self, inputs: torch.Tensor) -> float:
        self._calls += 1
        if self._mutate_inputs:
            inputs[:, :, 0, 0].add_(1.0)
        value = float(inputs.sum().item())
        if self._mutate_identity_after_first_call and self._calls == 1:
            self._target_identity = {**self._target_identity, "mutated_after_call": 1}
        return value


def _build_schedule_compat(
    clean_inputs: torch.Tensor,
    baseline_inputs: torch.Tensor,
    ranking: np.ndarray,
    *,
    scale: FidelityScale,
    direction: CurveDirection,
    ranking_kind: RankingKind,
    sample_id: str = "toy-sample",
    block: int = 3,
    head: int = 5,
    seed: int | None = None,
    replicate: int | None = None,
    scores: np.ndarray | None = None,
    unit_map: UnitMassMap | None = None,
    response: TargetResponse | None = None,
):
    if ranking_kind is RankingKind.TARGET_ORIENTED_POSITIVE:
        if unit_map is None or response is None:
            # Strictly test the low-level schedule only through a temporary
            # synthetic UnitMassMap bound to this exact target/method.
            if response is None:
                raise TypeError("target-oriented test schedule requires response")
            masses = np.asarray(scores, dtype=np.float32).reshape(
                scale.grid_size, scale.grid_size
            )
            unit_map = UnitMassMap(
                method_id=NNS_METHOD_ID, comparison_line=response.comparison_line,
                target_ref=response.target_ref,
                target_identity_sha256=response.target_identity_sha256,
                source_clean_input_sha256=tensor_sha256(clean_inputs),
                scale=scale, sign_semantics=SignSemantics.SIGNED, masses=masses,
                sha256=__import__("vitnns.stage_c_fidelity", fromlist=["_array_sha256"])._array_sha256(masses),
            )
        return build_method_intervention_schedule(
            clean_inputs, baseline_inputs, unit_map, response,
            sample_id=sample_id, block=block, head=head, replicate=replicate,
            direction=direction,
        )
    return build_intervention_schedule(
        clean_inputs,
        baseline_inputs,
        ranking,
        scores=scores,
        scale=scale,
        direction=direction,
        ranking_kind=ranking_kind,
        sample_id=sample_id,
        block=block,
        head=head,
        seed=seed,
        replicate=replicate,
    )


def _response_from_clean(
    clean_inputs: torch.Tensor,
    *,
    line: ComparisonLine = ComparisonLine.VALUE_PATH,
    label: str = "toy-response",
    target_identity: dict[str, Any] | None = None,
) -> tuple[ToyEvaluator, TargetResponse]:
    evaluator = ToyEvaluator(
        comparison_line=line,
        target_identity=target_identity or _canonical_target_identity(line, clean_inputs, label=label),
    )
    response = resolve_target_response(evaluator, clean_inputs, _baseline_inputs())
    return evaluator, response


def _build_center_deletion_schedule(clean_inputs: torch.Tensor):
    return _build_schedule_compat(
        clean_inputs,
        _baseline_inputs(),
        center_distance_ranking(FidelityScale.PATCH_16),
        scale=FidelityScale.PATCH_16,
        direction=CurveDirection.DELETION,
        ranking_kind=RankingKind.CENTER_DISTANCE_ROW_MAJOR,
    )


def _record_tensor_sha256_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[torch.Tensor]:
    calls: list[torch.Tensor] = []
    original = fidelity.tensor_sha256

    def recording_tensor_sha256(value: torch.Tensor) -> str:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(fidelity, "tensor_sha256", recording_tensor_sha256)
    return calls


def _provenance_target_exact(response: TargetResponse) -> MappingProxyType[str, Any]:
    return MappingProxyType(
        {
            "source": "toy-stage-c-fidelity",
            "target": dict(response.target_identity["frozen_target"]),
        }
    )


def _provenance_caller_context(response: TargetResponse) -> MappingProxyType[str, Any]:
    return MappingProxyType(
        {
            "source": "toy-stage-c-fidelity",
            "caller_context": fidelity_target_provenance_binding(response),
        }
    )


def _make_nns_method_result(
    clean_inputs: torch.Tensor, response: TargetResponse
) -> MethodResult:
    native = np.zeros((3, 224, 224), dtype=np.float32)
    native[0, 0:4, 0:4] = 2.0
    native[1, 0:4, 0:4] = -1.0
    native[2, 0:4, 0:4] = 0.5
    native[:, 0:4, 4:8] = np.asarray(
        (-0.25, 0.0, 0.0), dtype=np.float32
    )[:, None, None]
    raw_native = raw_array(native)
    metric = pixel_to_patch_mass(signed_rgb_channel_sum(raw_native))
    return MethodResult(
        method_id=NNS_METHOD_ID,
        method_version="1.0.0",
        comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target=VALUE_TARGET_REF,
        fidelity_target=VALUE_TARGET_REF,
        class_agnostic=True,
        native_tensor=raw_native,
        metric_map=metric,
        projection="pixel_mass_to_patch_sum",
        sign_semantics=SignSemantics.SIGNED,
        valid_mask=np.ones((14, 14), dtype=bool),
        provenance=_provenance_target_exact(response),
    )


def _make_matched_method_result(
    clean_inputs: torch.Tensor, response: TargetResponse
) -> MethodResult:
    patches = np.zeros((14, 14), dtype=np.float32)
    patches[0, 0] = 32.0
    patches[0, 1] = -16.0
    metric = stage_c_metric_ready(patches)
    return MethodResult(
        method_id=MATCHED_METHOD_ID,
        method_version="1.0.0",
        comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target=VALUE_TARGET_REF,
        fidelity_target=VALUE_TARGET_REF,
        class_agnostic=True,
        native_tensor=raw_array(patches),
        metric_map=metric,
        projection="patch_mass_to_pixel_uniform",
        sign_semantics=SignSemantics.SIGNED,
        valid_mask=np.ones((14, 14), dtype=bool),
        provenance=_provenance_caller_context(response),
    )


def _make_value_aggregate(
    *,
    method_id: str,
    direction: CurveDirection,
    scale: FidelityScale = FidelityScale.FINE_4,
    ranking_kind: RankingKind = RankingKind.TARGET_ORIENTED_POSITIVE,
    sample_id: str = "toy-sample",
    block: int = 3,
    head: int = 5,
    clean_inputs: torch.Tensor,
    target_identity: dict[str, Any],
) -> tuple[Any, tuple[int, ...]]:
    evaluator, response = _response_from_clean(
        clean_inputs,
        label="shared-value-target",
        target_identity=target_identity,
    )
    if method_id == NNS_METHOD_ID:
        result = _make_nns_method_result(clean_inputs, response)
    elif method_id == MATCHED_METHOD_ID:
        result = _make_matched_method_result(clean_inputs, response)
    else:
        raise AssertionError(method_id)
    unit_map = method_unit_masses(result, response, scale)
    scores = target_oriented_scores(unit_map, response).reshape(-1)
    rankings = paired_rankings_for_method(scores, sample_id=sample_id, block=block, head=head)
    curves = []
    for replicate, (seed, ranking) in enumerate(rankings):
        schedule = _build_schedule_compat(
            clean_inputs,
            _baseline_inputs(),
            ranking,
            scale=scale,
            direction=direction,
            ranking_kind=ranking_kind,
            sample_id=sample_id,
            block=block,
            head=head,
            seed=seed,
            replicate=replicate,
            scores=scores,
            unit_map=unit_map, response=response,
        )
        curves.append(
            evaluate_fidelity_curve(evaluator, response, schedule, method_id=method_id)
        )
    return aggregate_five_curves(curves), tuple(seed for seed, _ in rankings)


def test_target_response_normalization_preserves_sign_no_clip_and_exact_degeneracy():
    with pytest.raises(TypeError, match="only be issued"):
        TargetResponse()
    value_clean = _unit_tensor(FidelityScale.PATCH_16, [0], value=1.0)
    value_identity = _canonical_target_identity(
        ComparisonLine.VALUE_PATH, value_clean, label="value"
    )
    positive = target_response_from_values(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_ref=VALUE_TARGET_REF,
        target_identity=value_identity,
        clean_value=3.0,
        baseline_value=1.0,
    )
    assert positive.sigma == 1
    assert positive.status == "OK"
    assert positive.normalize(5.0) == pytest.approx(2.0)
    assert positive.normalize(0.5) == pytest.approx(-0.25)

    class_clean = _unit_tensor(FidelityScale.PATCH_16, [0], value=1.0)
    class_identity = _canonical_target_identity(
        ComparisonLine.CLASS_LOGIT, class_clean, label="class"
    )
    negative = target_response_from_values(
        comparison_line=ComparisonLine.CLASS_LOGIT,
        target_ref=CLASS_TARGET_REF,
        target_identity=class_identity,
        clean_value=1.0,
        baseline_value=3.0,
    )
    assert negative.sigma == -1
    assert negative.normalize(0.0) == pytest.approx(1.5)
    assert negative.normalize(4.0) == pytest.approx(-0.5)

    degenerate_value = target_response_from_values(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_ref=VALUE_TARGET_REF,
        target_identity=value_identity,
        clean_value=1e-6,
        baseline_value=0.0,
    )
    assert degenerate_value.sigma is None
    assert degenerate_value.na == StructuredNA(
        NAReason.DEGENERATE_TARGET_DELTA,
        "Value-path target-delta degeneracy aligns with direction validity failure",
    )
    assert degenerate_value.normalize(1.2) == degenerate_value.na
    edge_non_degenerate = target_response_from_values(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_ref=VALUE_TARGET_REF,
        target_identity=value_identity,
        clean_value=float(np.nextafter(1e-6, 1.0)),
        baseline_value=0.0,
    )
    assert edge_non_degenerate.sigma == 1

    degenerate_class = target_response_from_values(
        comparison_line=ComparisonLine.CLASS_LOGIT,
        target_ref=CLASS_TARGET_REF,
        target_identity=class_identity,
        clean_value=1e-6,
        baseline_value=0.0,
    )
    assert degenerate_class.sigma is None
    assert degenerate_class.na == StructuredNA(NAReason.DEGENERATE_TARGET_DELTA)


def test_resolve_target_response_requires_typed_evaluator_and_stable_inputs_identity():
    clean_inputs = _unit_tensor(FidelityScale.PATCH_16, [0, 1, 2], value=1.0)
    baseline = _baseline_inputs()

    with pytest.raises(TypeError, match="bare callable is forbidden"):
        resolve_target_response(lambda _: 0.0, clean_inputs, baseline)

    wrong_ref = ToyEvaluator(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_ref=CLASS_TARGET_REF,
        target_identity=_canonical_target_identity(
            ComparisonLine.VALUE_PATH, clean_inputs, label="wrong-ref"
        ),
    )
    with pytest.raises(FidelityContractError, match="does not match"):
        resolve_target_response(wrong_ref, clean_inputs, baseline)

    mutating_inputs = ToyEvaluator(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_identity=_canonical_target_identity(
            ComparisonLine.VALUE_PATH, clean_inputs, label="mutating-inputs"
        ),
        mutate_inputs=True,
    )
    with pytest.raises(FidelityContractError, match="mutated an intervention tensor"):
        resolve_target_response(mutating_inputs, clean_inputs, baseline)

    mutating_identity = ToyEvaluator(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_identity=_canonical_target_identity(
            ComparisonLine.VALUE_PATH, clean_inputs, label="mutating-identity"
        ),
        mutate_identity_after_first_call=True,
    )
    with pytest.raises(FidelityContractError, match="identity changed"):
        resolve_target_response(mutating_identity, clean_inputs, baseline)


def test_method_unit_masses_projects_patch_and_fine_value_methods_with_registered_mass_rules():
    clean_inputs = _unit_tensor(FidelityScale.FINE_4, list(range(20)), value=1.0)
    _, response = _response_from_clean(clean_inputs)
    nns = _make_nns_method_result(clean_inputs, response)
    matched = _make_matched_method_result(clean_inputs, response)

    nns_patch = method_unit_masses(nns, response, FidelityScale.PATCH_16)
    nns_fine = method_unit_masses(nns, response, FidelityScale.FINE_4)
    matched_fine = method_unit_masses(matched, response, FidelityScale.FINE_4)

    expected_patch = pixel_to_patch_mass(signed_rgb_channel_sum(nns.native_tensor)).data
    expected_fine = pixel_to_fine_cell_mass(signed_rgb_channel_sum(nns.native_tensor)).data
    expected_matched = patch_to_fine_cell_mass(matched.metric_map).data

    np.testing.assert_array_equal(nns_patch.masses, expected_patch)
    np.testing.assert_array_equal(nns_fine.masses, expected_fine)
    np.testing.assert_array_equal(matched_fine.masses, expected_matched)
    assert nns_fine.masses[0, 0] == pytest.approx(24.0)
    assert nns_fine.masses[0, 1] == pytest.approx(-4.0)
    np.testing.assert_array_equal(
        matched_fine.masses[:4, :4],
        np.full((4, 4), 2.0, dtype=np.float32),
    )
    with pytest.raises(ValueError):
        nns_fine.masses.setflags(write=True)

    schedule = build_method_intervention_schedule(
        clean_inputs, _baseline_inputs(), nns_patch, response,
        sample_id="toy-sample", block=3, head=5, replicate=0,
        direction=CurveDirection.DELETION,
    )
    assert schedule.source_method_id == NNS_METHOD_ID
    assert schedule.source_unit_map_sha256 == nns_patch.sha256
    assert schedule.source_target_identity_sha256 == response.target_identity_sha256
    with pytest.raises(FidelityContractError, match="UnitMassMap"):
        build_intervention_schedule(
            clean_inputs, _baseline_inputs(), schedule.ranking,
            scores=target_oriented_scores(nns_patch, response).reshape(-1),
            scale=FidelityScale.PATCH_16, direction=CurveDirection.DELETION,
            ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE,
            sample_id="toy-sample", block=3, head=5, seed=schedule.seed,
            replicate=0,
        )


def test_target_oriented_scores_take_positive_part_after_target_sign_orientation():
    masses = np.zeros((14, 14), dtype=np.float32)
    masses[0, 0] = 3.0
    masses[0, 1] = -2.0
    masses[0, 2] = 0.0
    masses[0, 3] = 4.0
    positive_clean = _unit_tensor(FidelityScale.PATCH_16, [0], value=1.0)
    positive_response = target_response_from_values(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_ref=VALUE_TARGET_REF,
        target_identity=_canonical_target_identity(
            ComparisonLine.VALUE_PATH, positive_clean, label="oriented-pos"
        ),
        clean_value=2.0,
        baseline_value=0.0,
    )
    positive_result = MethodResult(
        method_id=MATCHED_METHOD_ID,
        method_version="1.0.0",
        comparison_line=ComparisonLine.VALUE_PATH,
        explanation_target=VALUE_TARGET_REF,
        fidelity_target=VALUE_TARGET_REF,
        class_agnostic=True,
        native_tensor=raw_array(masses),
        metric_map=stage_c_metric_ready(masses),
        projection="patch_mass_to_pixel_uniform",
        sign_semantics=SignSemantics.SIGNED,
        valid_mask=np.ones((14, 14), dtype=bool),
        provenance=_provenance_caller_context(positive_response),
    )
    positive_map = method_unit_masses(positive_result, positive_response, FidelityScale.FINE_4)
    positive_scores = target_oriented_scores(positive_map, positive_response)
    assert positive_scores[0, 0] == pytest.approx(3.0 / 16.0)
    assert positive_scores[0, 4] == 0.0
    assert positive_scores[0, 12] == pytest.approx(4.0 / 16.0)

    negative_response = target_response_from_values(
        comparison_line=ComparisonLine.CLASS_LOGIT,
        target_ref=CLASS_TARGET_REF,
        target_identity=_canonical_target_identity(
            ComparisonLine.CLASS_LOGIT, positive_clean, label="oriented-neg"
        ),
        clean_value=1.0,
        baseline_value=3.0,
    )
    negative_map = UnitMassMap(
        method_id="full_model_class_logit_ig",
        comparison_line=ComparisonLine.CLASS_LOGIT,
        target_ref=CLASS_TARGET_REF,
        target_identity_sha256=negative_response.target_identity_sha256,
        source_clean_input_sha256=negative_response.target_identity["clean_input_sha256"],
        scale=FidelityScale.FINE_4,
        sign_semantics=SignSemantics.SIGNED,
        masses=np.array(positive_map.masses, copy=True),
        sha256=positive_map.sha256,
    )
    negative_scores = target_oriented_scores(negative_map, negative_response)
    assert negative_scores[0, 0] == 0.0
    assert negative_scores[0, 4] == pytest.approx(2.0 / 16.0)
    assert negative_scores[0, 12] == 0.0
    with pytest.raises(FidelityContractError, match="do not pair"):
        target_oriented_scores(positive_map, negative_response)


def test_seed_and_rankings_are_golden_repeatable_and_preserve_tie_groups():
    assert fidelity_seed("toy-sample", 3, 5, 2) == 10453737017400795055

    scores = np.asarray([3.0, 3.0, 1.0, 1.0, 0.0], dtype=np.float32)
    seed0 = fidelity_seed("toy-sample", 3, 5, 0)
    seed1 = fidelity_seed("toy-sample", 3, 5, 1)
    ranking0 = tie_permuted_ranking(scores, seed0)
    ranking0_repeat = tie_permuted_ranking(scores, seed0)
    ranking1 = tie_permuted_ranking(scores, seed1)

    np.testing.assert_array_equal(ranking0, ranking0_repeat)
    assert set(ranking0[:2].tolist()) == {0, 1}
    assert set(ranking0[2:4].tolist()) == {2, 3}
    assert ranking0[4] == 4
    assert set(ranking1[:2].tolist()) == {0, 1}
    assert set(ranking1[2:4].tolist()) == {2, 3}

    paired = paired_rankings_for_method(scores, sample_id="toy-sample", block=3, head=5)
    assert len(paired) == TIE_REPLICATES
    assert tuple(seed for seed, _ in paired) == tuple(
        fidelity_seed("toy-sample", 3, 5, replicate) for replicate in range(TIE_REPLICATES)
    )


def test_center_and_random_rankings_use_registered_orders_and_five_replicates():
    center_patch = center_distance_ranking(FidelityScale.PATCH_16)
    np.testing.assert_array_equal(center_patch[:4], np.asarray([90, 91, 104, 105], dtype=np.int64))

    random_a = global_random_ranking(8, seed=7)
    random_b = global_random_ranking(8, seed=7)
    np.testing.assert_array_equal(random_a, random_b)
    np.testing.assert_array_equal(np.sort(random_a), np.arange(8, dtype=np.int64))

    random_sets = random_rankings(8, sample_id="toy-sample", block=3, head=5)
    assert len(random_sets) == 5
    assert len({seed for seed, _ in random_sets}) == 5
    for seed, ranking in random_sets:
        assert seed in {
            fidelity_seed("toy-sample", 3, 5, replicate) for replicate in range(TIE_REPLICATES)
        }
        np.testing.assert_array_equal(np.sort(ranking), np.arange(8, dtype=np.int64))

    clean = _unit_tensor(FidelityScale.PATCH_16, [0, 1])
    center_schedule = build_intervention_schedule(
        clean, _baseline_inputs(), center_patch,
        scale=FidelityScale.PATCH_16, direction=CurveDirection.DELETION,
        ranking_kind=RankingKind.CENTER_DISTANCE_ROW_MAJOR,
        sample_id="toy-sample", block=3, head=5,
    )
    assert center_schedule.seed is None and center_schedule.replicate is None
    random_seed = fidelity_seed("toy-sample", 3, 5, 0)
    random_order = global_random_ranking(196, random_seed)
    random_schedule = build_intervention_schedule(
        clean, _baseline_inputs(), random_order,
        scale=FidelityScale.PATCH_16, direction=CurveDirection.INSERTION,
        ranking_kind=RankingKind.GLOBAL_RANDOM, sample_id="toy-sample",
        block=3, head=5, seed=random_seed, replicate=0,
    )
    assert random_schedule.ranking_kind is RankingKind.GLOBAL_RANDOM


def test_curve_schedule_uses_21_point_ceil_grid_actual_x_and_rgb_coupled_zero_corruption():
    clean_inputs = _unit_tensor(FidelityScale.PATCH_16, list(range(49)), value=1.0)
    evaluator, response = _response_from_clean(clean_inputs, label="curve-grid")
    scores = np.arange(FidelityScale.PATCH_16.unit_count, 0, -1, dtype=np.float32)
    seed = fidelity_seed("toy-sample", 3, 5, 0)
    ranking = tie_permuted_ranking(scores, seed)
    deletion = _build_schedule_compat(
        clean_inputs,
        _baseline_inputs(),
        ranking,
        scale=FidelityScale.PATCH_16,
        direction=CurveDirection.DELETION,
        ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE,
        seed=seed,
        replicate=0,
        scores=scores,
        response=response,
    )
    insertion = _build_schedule_compat(
        clean_inputs,
        _baseline_inputs(),
        ranking,
        scale=FidelityScale.PATCH_16,
        direction=CurveDirection.INSERTION,
        ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE,
        seed=seed,
        replicate=0,
        scores=scores,
        response=response,
    )
    deletion_curve = evaluate_fidelity_curve(evaluator, response, deletion, method_id=NNS_METHOD_ID)
    insertion_curve = evaluate_fidelity_curve(evaluator, response, insertion, method_id=NNS_METHOD_ID)

    assert deletion.unit_counts == curve_unit_counts(196)
    assert deletion.actual_fractions == actual_curve_fractions(196)
    assert deletion.unit_counts == (0, 10, 20, 30, 40, 49, 59, 69, 79, 89, 98, 108, 118, 128, 138, 147, 157, 167, 177, 187, 196)
    assert deletion.actual_fractions[4] == pytest.approx(40 / 196)
    torch.testing.assert_close(deletion.tensor_at(0), clean_inputs)
    torch.testing.assert_close(insertion.tensor_at(0), _baseline_inputs())
    assert float(deletion.tensor_at(1)[:, :, :16, :16].abs().sum().item()) == 0.0
    assert float(insertion.tensor_at(1)[:, :, :16, :16].min().item()) == 1.0
    assert float(deletion.tensor_at(1)[:, :, :16, :16].max().item()) == 0.0

    assert deletion_curve.responses[0] == pytest.approx(1.0)
    assert deletion_curve.responses[-1] == pytest.approx(0.0)
    assert insertion_curve.responses[0] == pytest.approx(0.0)
    assert insertion_curve.responses[-1] == pytest.approx(1.0)
    assert deletion_curve.auc == pytest.approx(trapezoid_auc(deletion.actual_fractions, deletion_curve.responses))
    assert insertion_curve.auc == pytest.approx(trapezoid_auc(insertion.actual_fractions, insertion_curve.responses))


def test_evaluate_curve_and_method_projection_fail_closed_on_wrong_target_method_line_and_identity():
    clean_inputs = _unit_tensor(FidelityScale.PATCH_16, list(range(10)), value=1.0)
    evaluator, response = _response_from_clean(clean_inputs, label="failures")
    nns = _make_nns_method_result(clean_inputs, response)
    seed = fidelity_seed("toy-sample", 3, 5, 0)
    schedule = _build_schedule_compat(
        clean_inputs,
        _baseline_inputs(),
        tie_permuted_ranking(
            np.arange(FidelityScale.PATCH_16.unit_count, 0, -1, dtype=np.float32),
            seed,
        ),
        scale=FidelityScale.PATCH_16,
        direction=CurveDirection.DELETION,
        ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE,
        seed=seed,
        replicate=0,
        scores=np.arange(FidelityScale.PATCH_16.unit_count, 0, -1, dtype=np.float32),
        response=response,
    )

    wrong_identity = ToyEvaluator(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_identity=_canonical_target_identity(
            ComparisonLine.VALUE_PATH, clean_inputs, label="other"
        ),
    )
    with pytest.raises(FidelityContractError, match="does not match target response"):
        evaluate_fidelity_curve(wrong_identity, response, schedule, method_id=NNS_METHOD_ID)

    with pytest.raises(FidelityContractError, match="eligible for this target line"):
        evaluate_fidelity_curve(evaluator, response, schedule, method_id="full_model_class_logit_ig")

    class_response = target_response_from_values(
        comparison_line=ComparisonLine.CLASS_LOGIT,
        target_ref=CLASS_TARGET_REF,
        target_identity=_canonical_target_identity(
            ComparisonLine.CLASS_LOGIT, clean_inputs, label="cross-line"
        ),
        clean_value=2.0,
        baseline_value=0.0,
    )
    with pytest.raises(FidelityContractError, match="cross comparison lines"):
        method_unit_masses(nns, class_response, FidelityScale.PATCH_16)
    perturbed_source = dataclasses.replace(
        nns, provenance={**dict(nns.provenance),
                         "intervention_input_sha256": "f" * 64},
    )
    with pytest.raises(FidelityContractError, match="not the frozen clean"):
        method_unit_masses(perturbed_source, response, FidelityScale.PATCH_16)

def test_shared_schedule_binds_hash_identity_detects_mutation_and_serializes_json_safely():
    clean_inputs = _unit_tensor(FidelityScale.PATCH_16, list(range(20)), value=1.0)
    evaluator, response = _response_from_clean(clean_inputs, label="shared-schedule")
    scores = np.arange(FidelityScale.PATCH_16.unit_count, 0, -1, dtype=np.float32)
    seed = fidelity_seed("toy-sample", 3, 5, 0)
    ranking = tie_permuted_ranking(scores, seed)
    schedule = _build_schedule_compat(
        clean_inputs,
        _baseline_inputs(),
        ranking,
        scale=FidelityScale.PATCH_16,
        direction=CurveDirection.DELETION,
        ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE,
        seed=seed,
        replicate=0,
        scores=scores,
        response=response,
    )

    nns_curve = evaluate_fidelity_curve(evaluator, response, schedule, method_id=NNS_METHOD_ID)
    with pytest.raises(FidelityContractError, match="does not come from this method"):
        evaluate_fidelity_curve(
            evaluator, response, schedule, method_id=MATCHED_METHOD_ID
        )
    json.loads(json.dumps(schedule.to_dict()))
    json.loads(json.dumps(nns_curve.to_dict()))

    exposed = schedule.tensor_at(0)
    exposed[:, :, 0, 0].add_(1.0)
    schedule.verify_integrity()
    with pytest.raises(AttributeError, match="immutable"):
        schedule.sample_id = "other"


def test_schedule_construction_strong_hashes_each_owned_tensor_once(
    monkeypatch: pytest.MonkeyPatch,
):
    clean_inputs = _unit_tensor(
        FidelityScale.PATCH_16, list(range(20)), value=1.0
    )
    hash_calls = _record_tensor_sha256_calls(monkeypatch)

    schedule = _build_center_deletion_schedule(clean_inputs)

    assert len(hash_calls) == 2 + len(CURVE_FRACTIONS)
    assert len({id(value) for value in hash_calls}) == len(hash_calls)
    assert [id(value) for value in hash_calls[2:]] == [
        id(value) for value in schedule._interventions
    ]
    torch.testing.assert_close(hash_calls[0], clean_inputs)
    torch.testing.assert_close(hash_calls[1], _baseline_inputs())


def test_curve_strong_hashes_only_the_group_at_entry_and_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    clean_inputs = _unit_tensor(
        FidelityScale.PATCH_16, list(range(20)), value=1.0
    )
    evaluator, response = _response_from_clean(clean_inputs, label="hash-boundaries")
    schedule = _build_center_deletion_schedule(clean_inputs)
    point_hash_counts: list[int] = []
    original_evaluate = evaluator.evaluate
    hash_calls = _record_tensor_sha256_calls(monkeypatch)

    def observe_hash_count(inputs: torch.Tensor) -> float:
        point_hash_counts.append(len(hash_calls))
        return original_evaluate(inputs)

    monkeypatch.setattr(evaluator, "evaluate", observe_hash_count)
    calls_before_curve = evaluator._calls

    curve = evaluate_fidelity_curve(
        evaluator, response, schedule, method_id=NNS_METHOD_ID
    )

    intervention_ids = [id(value) for value in schedule._interventions]
    assert curve.responses is not None and len(curve.responses) == len(CURVE_FRACTIONS)
    assert evaluator._calls - calls_before_curve == len(CURVE_FRACTIONS)
    assert point_hash_counts == [len(CURVE_FRACTIONS)] * len(CURVE_FRACTIONS)
    assert [id(value) for value in hash_calls] == intervention_ids * 2


def test_evaluate_at_ordinary_in_place_mutation_uses_the_cheap_point_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    schedule = _build_center_deletion_schedule(
        _unit_tensor(FidelityScale.PATCH_16, list(range(20)), value=1.0)
    )

    def forbid_strong_hash(_value: torch.Tensor) -> str:
        raise AssertionError("evaluate_at must not strongly hash a curve point")

    monkeypatch.setattr(fidelity, "tensor_sha256", forbid_strong_hash)

    def mutate(inputs: torch.Tensor) -> float:
        inputs[:, :, 0, 0].add_(1.0)
        return float(inputs.sum().item())

    with pytest.raises(
        FidelityContractError, match=r"^intervention tensor 0 was mutated$"
    ):
        schedule.evaluate_at(0, mutate)


def test_curve_exit_strong_hash_catches_data_mutation_after_point_guards(
    monkeypatch: pytest.MonkeyPatch,
):
    clean_inputs = _unit_tensor(
        FidelityScale.PATCH_16, list(range(20)), value=1.0
    )
    evaluator, response = _response_from_clean(clean_inputs, label="data-mutation")
    schedule = _build_center_deletion_schedule(clean_inputs)
    original_evaluate = evaluator.evaluate
    point_calls = 0

    def mutate_data_once(inputs: torch.Tensor) -> float:
        nonlocal point_calls
        if point_calls == 0:
            inputs.data[:, :, 0, 0].add_(1.0)
        point_calls += 1
        return original_evaluate(inputs)

    monkeypatch.setattr(evaluator, "evaluate", mutate_data_once)
    hash_calls = _record_tensor_sha256_calls(monkeypatch)

    with pytest.raises(
        FidelityContractError, match=r"^an intervention tensor was mutated$"
    ):
        evaluate_fidelity_curve(
            evaluator, response, schedule, method_id=NNS_METHOD_ID
        )

    intervention_ids = [id(value) for value in schedule._interventions]
    assert point_calls == len(CURVE_FRACTIONS)
    assert [id(value) for value in hash_calls] == intervention_ids * 2


def test_curve_consumer_exception_still_runs_exit_strong_verification(
    monkeypatch: pytest.MonkeyPatch,
):
    clean_inputs = _unit_tensor(
        FidelityScale.PATCH_16, list(range(20)), value=1.0
    )
    evaluator, response = _response_from_clean(
        clean_inputs, label="consumer-exception"
    )
    schedule = _build_center_deletion_schedule(clean_inputs)
    point_calls = 0

    def raise_from_consumer(_inputs: torch.Tensor) -> float:
        nonlocal point_calls
        point_calls += 1
        raise RuntimeError("consumer failed")

    monkeypatch.setattr(evaluator, "evaluate", raise_from_consumer)
    hash_calls = _record_tensor_sha256_calls(monkeypatch)

    with pytest.raises(RuntimeError, match=r"^consumer failed$"):
        evaluate_fidelity_curve(
            evaluator, response, schedule, method_id=NNS_METHOD_ID
        )

    intervention_ids = [id(value) for value in schedule._interventions]
    assert point_calls == 1
    assert [id(value) for value in hash_calls] == intervention_ids * 2


def test_aggregate_uses_five_repeat_mean_population_std_and_preserves_paired_na():
    clean_inputs = _unit_tensor(FidelityScale.PATCH_16, list(range(49)), value=1.0)
    evaluator, response = _response_from_clean(clean_inputs, label="aggregate")
    scores = np.arange(FidelityScale.PATCH_16.unit_count, 0, -1, dtype=np.float32)
    ranking = tie_permuted_ranking(scores, fidelity_seed("toy-sample", 3, 5, 0))
    base_curve = evaluate_fidelity_curve(
        evaluator,
        response,
        _build_schedule_compat(
            clean_inputs,
            _baseline_inputs(),
            ranking,
            scale=FidelityScale.PATCH_16,
            direction=CurveDirection.DELETION,
            ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE,
            seed=fidelity_seed("toy-sample", 3, 5, 0),
            replicate=0,
            scores=scores,
            response=response,
        ),
        method_id=NNS_METHOD_ID,
    )
    offsets = (0.0, 0.02, 0.04, 0.06, 0.08)
    curves = []
    for replicate, offset in enumerate(offsets):
        assert base_curve.responses is not None
        responses = tuple(
            1.0 if index == 0 else 0.0 if index == 20 else max(0.0, min(1.0, value + offset))
            for index, value in enumerate(base_curve.responses)
        )
        curves.append(
            dataclasses.replace(
                base_curve,
                replicate=replicate,
                seed=fidelity_seed("toy-sample", 3, 5, replicate),
                tie_priority_sha256=tie_priority_sha256(
                    base_curve.scale.unit_count,
                    fidelity_seed("toy-sample", 3, 5, replicate),
                ),
                responses=responses,
                auc=trapezoid_auc(base_curve.actual_fractions, responses),
            )
        )
    aggregate = aggregate_five_curves(curves)
    matrix = np.asarray([curve.responses for curve in curves], dtype=np.float64)
    aucs = np.asarray([curve.auc for curve in curves], dtype=np.float64)
    assert aggregate.response_mean == pytest.approx(tuple(matrix.mean(axis=0).tolist()))
    assert aggregate.response_std_population == pytest.approx(tuple(matrix.std(axis=0, ddof=0).tolist()))
    assert aggregate.auc_mean == pytest.approx(float(aucs.mean()))
    assert aggregate.auc_std_population == pytest.approx(float(aucs.std(ddof=0)))
    json.loads(json.dumps(aggregate.to_dict()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        aggregate.auc_mean = 0.0

    degenerate = target_response_from_values(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_ref=VALUE_TARGET_REF,
        target_identity=dict(response.target_identity),
        clean_value=1e-6,
        baseline_value=0.0,
    )
    na_curves = tuple(
        dataclasses.replace(
            base_curve,
            replicate=replicate,
            seed=fidelity_seed("toy-sample", 3, 5, replicate),
            tie_priority_sha256=tie_priority_sha256(
                base_curve.scale.unit_count,
                fidelity_seed("toy-sample", 3, 5, replicate),
            ),
            responses=None, auc=None, na=degenerate.na,
        )
        for replicate in range(TIE_REPLICATES)
    )
    na_aggregate = aggregate_five_curves(na_curves)
    assert na_aggregate.na == StructuredNA(
        NAReason.DEGENERATE_TARGET_DELTA,
        "Value-path target-delta degeneracy aligns with direction validity failure",
    )


def test_top20_uses_registered_ceil_count_and_returns_paired_na_when_curves_are_na():
    clean_inputs = _unit_tensor(FidelityScale.PATCH_16, list(range(49)), value=1.0)
    evaluator, response = _response_from_clean(clean_inputs, label="top20")
    scores = np.arange(FidelityScale.PATCH_16.unit_count, 0, -1, dtype=np.float32)
    seed = fidelity_seed("toy-sample", 3, 5, 0)
    ranking = tie_permuted_ranking(scores, seed)
    insertion = evaluate_fidelity_curve(
        evaluator,
        response,
        _build_schedule_compat(
            clean_inputs,
            _baseline_inputs(),
            ranking,
            scale=FidelityScale.PATCH_16,
            direction=CurveDirection.INSERTION,
            ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE,
            seed=seed,
            replicate=0,
            scores=scores,
            response=response,
        ),
        method_id=NNS_METHOD_ID,
    )
    deletion = evaluate_fidelity_curve(
        evaluator,
        response,
        _build_schedule_compat(
            clean_inputs,
            _baseline_inputs(),
            ranking,
            scale=FidelityScale.PATCH_16,
            direction=CurveDirection.DELETION,
            ranking_kind=RankingKind.TARGET_ORIENTED_POSITIVE,
            seed=seed,
            replicate=0,
            scores=scores,
            response=response,
        ),
        method_id=NNS_METHOD_ID,
    )
    top20 = top20_from_curves(insertion, deletion)
    assert top20.k == 40
    assert top20.actual_fraction == pytest.approx(40 / 196)
    assert top20.sufficiency == pytest.approx(40 / 49)
    assert top20.sufficiency_gap == pytest.approx(9 / 49)
    assert top20.comprehensiveness == pytest.approx(40 / 49)
    json.loads(json.dumps(top20.to_dict()))

    degenerate = target_response_from_values(
        comparison_line=ComparisonLine.VALUE_PATH,
        target_ref=VALUE_TARGET_REF,
        target_identity=dict(response.target_identity),
        clean_value=1e-6,
        baseline_value=0.0,
    )
    insertion_na = dataclasses.replace(
        insertion, responses=None, auc=None, na=degenerate.na
    )
    deletion_na = dataclasses.replace(
        deletion, responses=None, auc=None, na=degenerate.na
    )
    assert top20_from_curves(insertion_na, deletion_na) == degenerate.na


def test_value_path_delta_fidelity_is_fine_only_same_target_and_uses_paired_seeds():
    shared_clean = _unit_tensor(FidelityScale.FINE_4, list(range(320)))
    shared_identity = _canonical_target_identity(
        ComparisonLine.VALUE_PATH, shared_clean, label="shared-value-target"
    )
    nns_insertion, nns_ins_seeds = _make_value_aggregate(
        method_id=NNS_METHOD_ID,
        direction=CurveDirection.INSERTION,
        clean_inputs=shared_clean,
        target_identity=shared_identity,
    )
    nns_deletion, nns_del_seeds = _make_value_aggregate(
        method_id=NNS_METHOD_ID,
        direction=CurveDirection.DELETION,
        clean_inputs=shared_clean,
        target_identity=shared_identity,
    )
    matched_insertion, matched_ins_seeds = _make_value_aggregate(
        method_id=MATCHED_METHOD_ID,
        direction=CurveDirection.INSERTION,
        clean_inputs=shared_clean,
        target_identity=shared_identity,
    )
    matched_deletion, matched_del_seeds = _make_value_aggregate(
        method_id=MATCHED_METHOD_ID,
        direction=CurveDirection.DELETION,
        clean_inputs=shared_clean,
        target_identity=shared_identity,
    )

    assert nns_ins_seeds == matched_ins_seeds
    assert nns_del_seeds == matched_del_seeds
    assert {curve.tie_priority_sha256 for curve in nns_insertion.curves} == {
        curve.tie_priority_sha256 for curve in matched_insertion.curves
    }
    assert {curve.tie_priority_sha256 for curve in nns_deletion.curves} == {
        curve.tie_priority_sha256 for curve in matched_deletion.curves
    }

    delta = value_path_delta_fidelity(
        nns_insertion=nns_insertion,
        nns_deletion=nns_deletion,
        matched_insertion=matched_insertion,
        matched_deletion=matched_deletion,
    )
    assert isinstance(delta, ValuePathDeltaFidelity)
    expected = 0.5 * (
        (nns_insertion.auc_mean - matched_insertion.auc_mean)
        + (matched_deletion.auc_mean - nns_deletion.auc_mean)
    )
    assert delta.value == pytest.approx(expected)
    json.loads(json.dumps(delta.to_dict()))

    wrong_identity_curves = tuple(
        dataclasses.replace(curve, target_identity_sha256="f" * 64)
        for curve in matched_insertion.curves
    )
    wrong_identity = aggregate_five_curves(wrong_identity_curves)
    with pytest.raises(FidelityContractError, match="mix frozen target identities"):
        value_path_delta_fidelity(
            nns_insertion=nns_insertion,
            nns_deletion=nns_deletion,
            matched_insertion=wrong_identity,
            matched_deletion=matched_deletion,
        )

    wrong_direction_curves = tuple(
        dataclasses.replace(curve, direction=CurveDirection.INSERTION)
        for curve in matched_deletion.curves
    )
    wrong_direction = aggregate_five_curves(wrong_direction_curves)
    with pytest.raises(FidelityContractError, match="same-target fine-4 Value-path pair"):
        value_path_delta_fidelity(
            nns_insertion=nns_insertion,
            nns_deletion=nns_deletion,
            matched_insertion=matched_insertion,
            matched_deletion=wrong_direction,
        )


def test_curve_fraction_grid_is_frozen_and_auc_requires_exact_21_point_endpoints():
    assert CURVE_FRACTIONS[0] == 0.0
    assert CURVE_FRACTIONS[-1] == 1.0
    assert curve_unit_counts(10) == (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10)
    with pytest.raises(FidelityContractError, match="frozen 21-point grid"):
        curve_unit_counts(10, fractions=(0.0, 1.0))
    with pytest.raises(FidelityContractError, match="exactly 21 x/y values"):
        trapezoid_auc((0.0, 1.0), (0.0, 1.0))
    with pytest.raises(FidelityContractError, match="increase from 0 to 1"):
        trapezoid_auc(tuple([0.0] + [0.1] * 20), tuple(float(index) for index in range(21)))
