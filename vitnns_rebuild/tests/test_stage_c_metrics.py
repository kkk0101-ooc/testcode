from __future__ import annotations

import math

import numpy as np
import pytest

from vitnns.display import DisplayBoundaryError, DisplayTransform
from vitnns.retention import ArtifactArray, ArtifactKind
from vitnns.stage_c_maps import (
    NAReason,
    SignSemantics,
    SignView,
    StructuredNA,
    stage_c_metric_ready,
)
from vitnns.stage_c_metrics import (
    KL_EPSILON,
    BidirectionalKL,
    MetricValue,
    bidirectional_kl,
    hotspot_jaccard,
    hotspot_jaccard_top_10_percent,
    hotspot_jaccard_top_20_percent,
    jensen_shannon_divergence,
    normalized_entropy,
    pearson,
    spearman,
)


FULL_MASK = np.ones((14, 14), dtype=bool)


def metric(values) -> ArtifactArray:
    return stage_c_metric_ready(np.asarray(values, dtype=np.float32).reshape(14, 14))


def sparse(values: list[float]) -> tuple[ArtifactArray, np.ndarray]:
    array = np.zeros((14, 14), dtype=np.float32)
    mask = np.zeros((14, 14), dtype=bool)
    flat = array.ravel()
    mask_flat = mask.ravel()
    flat[: len(values)] = values
    mask_flat[: len(values)] = True
    return stage_c_metric_ready(array), mask


def correlation_kwargs(mask=FULL_MASK, view=SignView.SIGNED):
    return {
        "mask": mask,
        "view": view,
        "left_semantics": SignSemantics.SIGNED,
        "right_semantics": SignSemantics.SIGNED,
    }


def test_pearson_and_spearman_match_hand_computable_values_with_average_ties():
    left, mask = sparse([1.0, 2.0, 2.0, 4.0])
    right, _ = sparse([4.0, 1.0, 1.0, 3.0])

    pearson_result = pearson(left, right, **correlation_kwargs(mask))
    assert isinstance(pearson_result, MetricValue)
    expected_pearson = float(np.corrcoef([1, 2, 2, 4], [4, 1, 1, 3])[0, 1])
    assert pearson_result.value == pytest.approx(expected_pearson, abs=1e-15)

    spearman_result = spearman(left, right, **correlation_kwargs(mask))
    assert isinstance(spearman_result, MetricValue)
    left_ranks = np.array([1.0, 2.5, 2.5, 4.0])
    right_ranks = np.array([4.0, 1.5, 1.5, 3.0])
    expected_spearman = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
    assert spearman_result.value == pytest.approx(expected_spearman, abs=1e-15)
    assert spearman_result.details["tie_method"] == "average_rank"


def test_correlations_support_registered_views_and_return_constant_na():
    signed = np.arange(-98, 98, dtype=np.float32).reshape(14, 14)
    first = stage_c_metric_ready(signed)
    second = stage_c_metric_ready(2 * signed)
    assert pearson(first, second, **correlation_kwargs()).value == pytest.approx(1.0)
    assert spearman(first, second, **correlation_kwargs()).value == pytest.approx(1.0)

    constant = stage_c_metric_ready(np.ones((14, 14), dtype=np.float32))
    result = pearson(constant, second, **correlation_kwargs())
    assert isinstance(result, StructuredNA)
    assert result.reason is NAReason.CONSTANT_LEFT
    reverse = spearman(second, constant, **correlation_kwargs())
    assert reverse.reason is NAReason.CONSTANT_RIGHT

    unsupported = pearson(
        first,
        stage_c_metric_ready(np.abs(signed)),
        mask=FULL_MASK,
        view=SignView.SIGNED,
        left_semantics=SignSemantics.SIGNED,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert unsupported.reason is NAReason.UNSUPPORTED_SIGN_SEMANTICS


def test_relative_l2_constant_threshold_marks_nearly_constant_float32_map_na():
    # The max(1, ||v||) floor makes a sub-1e-12 near-zero deviation constant.
    nearly_constant = np.zeros((14, 14), dtype=np.float32)
    nearly_constant[0, 0] = np.float32(1e-14)
    varying = metric(np.arange(196))
    result = pearson(
        stage_c_metric_ready(nearly_constant), varying, **correlation_kwargs()
    )
    assert result.reason is NAReason.CONSTANT_LEFT


def test_base2_js_has_no_epsilon_uses_zero_log_zero_and_is_divergence():
    left, mask = sparse([1.0, 0.0])
    right, _ = sparse([0.0, 1.0])
    result = jensen_shannon_divergence(
        left,
        right,
        mask=mask,
        view=SignView.POSITIVE,
        left_semantics=SignSemantics.NONNEGATIVE,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert isinstance(result, MetricValue)
    assert result.value == pytest.approx(1.0, abs=1e-15)
    assert result.details["log_base"] == 2
    assert result.details["epsilon"] == 0.0

    identical = jensen_shannon_divergence(
        left,
        left,
        mask=mask,
        view=SignView.POSITIVE,
        left_semantics=SignSemantics.NONNEGATIVE,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert identical.value == pytest.approx(0.0, abs=1e-15)


def test_bidirectional_base2_kl_uses_registered_epsilon_formula():
    left, mask = sparse([3.0, 1.0])
    right, _ = sparse([1.0, 3.0])
    result = bidirectional_kl(
        left,
        right,
        mask=mask,
        view=SignView.POSITIVE,
        left_semantics=SignSemantics.NONNEGATIVE,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert isinstance(result, BidirectionalKL)
    p = np.array([0.75, 0.25], dtype=np.float64)
    q = np.array([0.25, 0.75], dtype=np.float64)
    p_eps = (p + KL_EPSILON) / (1 + 2 * KL_EPSILON)
    q_eps = (q + KL_EPSILON) / (1 + 2 * KL_EPSILON)
    expected_pq = float(np.sum(p_eps * np.log2(p_eps / q_eps)))
    expected_qp = float(np.sum(q_eps * np.log2(q_eps / p_eps)))
    assert result.left_to_right.value == pytest.approx(expected_pq, abs=1e-15)
    assert result.right_to_left.value == pytest.approx(expected_qp, abs=1e-15)


def test_distribution_metrics_reject_signed_view_and_report_zero_mass_per_side():
    signed = stage_c_metric_ready(
        np.resize(np.array([-1.0, 1.0], dtype=np.float32), (14, 14))
    )
    unsupported = jensen_shannon_divergence(
        signed,
        signed,
        mask=FULL_MASK,
        view=SignView.SIGNED,
        left_semantics=SignSemantics.SIGNED,
        right_semantics=SignSemantics.SIGNED,
    )
    assert unsupported.reason is NAReason.UNSUPPORTED_SIGN_SEMANTICS

    zero = stage_c_metric_ready(np.zeros((14, 14), dtype=np.float32))
    positive = stage_c_metric_ready(np.ones((14, 14), dtype=np.float32))
    left_zero = jensen_shannon_divergence(
        zero,
        positive,
        mask=FULL_MASK,
        view=SignView.POSITIVE,
        left_semantics=SignSemantics.NONNEGATIVE,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert left_zero.reason is NAReason.ZERO_MASS_LEFT
    right_zero = bidirectional_kl(
        positive,
        zero,
        mask=FULL_MASK,
        view=SignView.POSITIVE,
        left_semantics=SignSemantics.NONNEGATIVE,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert right_zero.reason is NAReason.ZERO_MASS_RIGHT


def test_sparse_branch_uses_relative_signed_l1_zero_mass_threshold():
    left = np.zeros((14, 14), dtype=np.float32)
    left.ravel()[0] = 1e-14
    left.ravel()[1] = -1.0
    right = np.ones((14, 14), dtype=np.float32)
    result = jensen_shannon_divergence(
        stage_c_metric_ready(left),
        stage_c_metric_ready(right),
        mask=FULL_MASK,
        view=SignView.POSITIVE,
        left_semantics=SignSemantics.SIGNED,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert result.reason is NAReason.ZERO_MASS_LEFT


def test_hotspot_uses_ceil_counts_row_major_ties_and_records_cutoff_counts():
    left = np.zeros(196, dtype=np.float32)
    right = np.zeros(196, dtype=np.float32)
    left[:19] = np.arange(100, 81, -1, dtype=np.float32)
    right[:19] = left[:19]
    left[[19, 20]] = 1.0
    right[[20, 21]] = 1.0
    left_map = stage_c_metric_ready(left.reshape(14, 14))
    right_map = stage_c_metric_ready(right.reshape(14, 14))

    result = hotspot_jaccard_top_10_percent(
        left_map,
        right_map,
        mask=FULL_MASK,
        view=SignView.POSITIVE,
        left_semantics=SignSemantics.NONNEGATIVE,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert isinstance(result, MetricValue)
    assert result.details["k"] == math.ceil(0.1 * 196) == 20
    assert result.details["left_cutoff_tie_count"] == 2
    assert result.details["right_cutoff_tie_count"] == 2
    assert result.value == pytest.approx(19 / 21)

    unique_left = np.arange(196, dtype=np.float32).reshape(14, 14)
    unique = stage_c_metric_ready(unique_left)
    top20 = hotspot_jaccard_top_20_percent(
        unique,
        unique,
        mask=FULL_MASK,
        view=SignView.POSITIVE,
        left_semantics=SignSemantics.NONNEGATIVE,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert top20.details["k"] == math.ceil(0.2 * 196) == 40
    assert top20.value == 1.0


def test_hotspot_rejects_unregistered_fraction_and_constant_maps():
    varying = metric(np.arange(196))
    with pytest.raises(ValueError, match="frozen 0.1 or 0.2"):
        hotspot_jaccard(
            varying,
            varying,
            mask=FULL_MASK,
            view=SignView.POSITIVE,
            left_semantics=SignSemantics.NONNEGATIVE,
            right_semantics=SignSemantics.NONNEGATIVE,
            fraction=0.15,
        )
    constant = metric(np.ones(196))
    result = hotspot_jaccard_top_10_percent(
        constant,
        varying,
        mask=FULL_MASK,
        view=SignView.POSITIVE,
        left_semantics=SignSemantics.NONNEGATIVE,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert result.reason is NAReason.CONSTANT_LEFT


def test_normalized_entropy_uniform_delta_zero_mass_and_single_valid_cell():
    uniform = metric(np.ones(196))
    uniform_result = normalized_entropy(
        uniform, mask=FULL_MASK, view=SignView.POSITIVE,
        semantics=SignSemantics.NONNEGATIVE,
    )
    assert uniform_result.value == pytest.approx(1.0, abs=1e-15)

    delta = np.zeros((14, 14), dtype=np.float32)
    delta[0, 0] = 1.0
    delta_result = normalized_entropy(
        stage_c_metric_ready(delta), mask=FULL_MASK, view=SignView.POSITIVE,
        semantics=SignSemantics.NONNEGATIVE,
    )
    assert delta_result.value == pytest.approx(0.0, abs=1e-15)

    zero = normalized_entropy(
        metric(np.zeros(196)), mask=FULL_MASK, view=SignView.POSITIVE,
        semantics=SignSemantics.NONNEGATIVE,
    )
    assert zero.reason is NAReason.ZERO_MASS_LEFT

    one_mask = np.zeros((14, 14), dtype=bool)
    one_mask[0, 0] = True
    single = normalized_entropy(
        uniform, mask=one_mask, view=SignView.POSITIVE,
        semantics=SignSemantics.NONNEGATIVE,
    )
    assert single.reason is NAReason.CONSTANT_LEFT


def test_shape_mismatch_nonfinite_and_empty_mask_are_invalidating_structured_na():
    valid = metric(np.arange(196))
    wrong_shape = ArtifactArray(
        np.ones((13, 14), dtype=np.float32), ArtifactKind.METRIC_READY
    )
    shape = pearson(wrong_shape, valid, **correlation_kwargs())
    assert shape.reason is NAReason.SHAPE_MISMATCH and shape.invalidating

    nonfinite = np.ones((14, 14), dtype=np.float32)
    nonfinite[0, 0] = np.nan
    nonfinite_tagged = ArtifactArray(nonfinite, ArtifactKind.METRIC_READY)
    finite = pearson(nonfinite_tagged, valid, **correlation_kwargs())
    assert finite.reason is NAReason.NON_FINITE and finite.invalidating

    empty = pearson(
        valid, valid, **correlation_kwargs(mask=np.zeros((14, 14), dtype=bool))
    )
    assert empty.to_dict() == {
        "status": "NA", "value": None, "reason": "empty_valid_region"
    }
    assert empty.invalidating


def test_metrics_only_accept_explicit_metric_ready_arrays_and_explicit_bool_masks():
    tagged = metric(np.arange(196))
    bare = np.arange(196, dtype=np.float32).reshape(14, 14)
    with pytest.raises(DisplayBoundaryError, match="bare arrays"):
        pearson(bare, tagged, **correlation_kwargs())
    raw = ArtifactArray(bare, ArtifactKind.RAW)
    with pytest.raises(DisplayBoundaryError, match="requires metric_ready"):
        pearson(raw, tagged, **correlation_kwargs())
    display = DisplayTransform(resize=(28, 28))(tagged)
    with pytest.raises(DisplayBoundaryError, match="display"):
        pearson(display, tagged, **correlation_kwargs())
    with pytest.raises(TypeError, match="boolean dtype"):
        pearson(
            tagged,
            tagged,
            **correlation_kwargs(mask=np.ones((14, 14), dtype=np.uint8)),
        )


def test_nonnegative_semantics_cannot_hide_negative_values_or_negative_view():
    negative = metric(-np.arange(196, dtype=np.float32))
    positive = metric(np.arange(196, dtype=np.float32))
    with pytest.raises(ValueError, match="violates.*nonnegative"):
        pearson(
            negative,
            positive,
            mask=FULL_MASK,
            view=SignView.POSITIVE,
            left_semantics=SignSemantics.NONNEGATIVE,
            right_semantics=SignSemantics.NONNEGATIVE,
        )
    unsupported = pearson(
        positive,
        positive,
        mask=FULL_MASK,
        view=SignView.NEGATIVE_MAGNITUDE,
        left_semantics=SignSemantics.NONNEGATIVE,
        right_semantics=SignSemantics.NONNEGATIVE,
    )
    assert unsupported.reason is NAReason.UNSUPPORTED_SIGN_SEMANTICS
