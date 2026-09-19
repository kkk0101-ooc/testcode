import pytest

from vitnns.endpoint_m5_analysis import (
    METHOD_MATRIX, average_ranks, bootstrap_median, head_block_metrics, spearman,
)


def test_average_ranks_and_spearman_ties():
    assert average_ranks([3, 1, 1, 2]) == [3, 0.5, 0.5, 2]
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert spearman([1, 2, 3], [1, 2, 3]) is None


def test_bootstrap_is_deterministic_and_constant_safe():
    assert bootstrap_median([3] * 400, "constant") == [3, 3]
    assert bootstrap_median(range(400), "fixture", 1000) == bootstrap_median(range(400), "fixture", 1000)


def test_head_block_metrics_separate_reference_invariant_and_sensitive_fields():
    zero = []
    blur = []
    for block in range(12):
        for head in range(4):
            common = {"component_id": f"b{block:02d}.head_{head:02d}", "operation": "head", "block": block, "m": head + 1.0, "a": head - 2.0}
            zero.append({**common, "p_ref": head + 1.0, "absolute_effect": head + 1.0})
            blur.append({**common, "p_ref": 4.0 - head, "absolute_effect": 4.0 - head})
    values, exact = head_block_metrics(zero, blur)
    assert values["taylor_reference_concordance"] == pytest.approx(-1)
    assert values["finite_effect_reference_concordance"] == pytest.approx(-1)
    assert values["endpoint_behavior_normalized_zero"] == pytest.approx(1)
    assert values["endpoint_behavior_blur_rgb"] == pytest.approx(-1)
    assert exact == {"taylor": 0, "finite_effect": 0}


def test_method_matrix_has_closed_unique_schema():
    required = {"method", "primary_question", "internal_scalar_target", "input_space_map", "external_intervention_reference", "local_jacobian_structural_closure", "finite_replacement_prediction"}
    assert len(METHOD_MATRIX) == 7
    assert len({row["method"] for row in METHOD_MATRIX}) == 7
    assert all(set(row) == required for row in METHOD_MATRIX)
