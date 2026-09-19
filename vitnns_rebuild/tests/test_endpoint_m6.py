import math

import numpy as np

from vitnns.endpoint_m6 import patch_metrics, select_nearest_median, select_tail, spearman


def test_inclusive_tail_and_id_tie_break():
    rows = [
        {"sample_id": "b", "model_id": "m", "block": 0, "value": 9.0},
        {"sample_id": "a", "model_id": "m", "block": 1, "value": 9.0},
        {"sample_id": "c", "model_id": "m", "block": 2, "value": 1.0},
    ]
    result = select_tail(rows, "value", "highest", fraction=1 / 3)
    assert result["threshold"] == 9.0
    assert result["inclusive_candidate_count"] == 2
    assert result["selected"]["sample_id"] == "a"


def test_nearest_median_and_spearman_with_ties():
    rows = [{"sample_id": key, "x": value} for key, value in (("d", 1), ("b", 2), ("a", 2), ("c", 8))]
    result = select_nearest_median(rows, "x")
    assert result["median"] == 2.0
    assert result["selected"]["sample_id"] == "a"
    assert math.isclose(spearman([1, 2, 2, 4], [4, 3, 3, 1]), -1.0)


def test_patch_locality_uses_registered_token_neighborhood():
    mass = np.zeros(196); mass[15] = 1.0
    metrics = patch_metrics(mass, token=16)
    assert metrics["own_token_locality"] == 1.0
    assert metrics["input_dependency_distance"] == 0.0
    assert patch_metrics(mass, token=0)["own_token_locality"] is None
