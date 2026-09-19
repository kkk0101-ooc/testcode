import math

import numpy as np

from vitnns.endpoint_m4_review import (
    bootstrap_median, entropy, exact_sign, holm, partial_spearman,
    patch_distribution, quantiles, spearman, total_variation,
)


def test_distribution_and_inference_helpers_are_closed_and_deterministic():
    assert quantiles(range(1, 10))["median"] == 5
    assert bootstrap_median([1, 2, 3], "fixed", 100) == bootstrap_median([1, 2, 3], "fixed", 100)
    assert exact_sign([1, 2, -1, 0])["positive"] == 2
    adjusted = holm([{"test_id": "a", "p": .01}, {"test_id": "b", "p": .04}])
    assert adjusted[0]["holm_p"] == .02 and adjusted[1]["holm_p"] == .04


def test_rank_and_partial_rank_relationships():
    x = [1, 2, 3, 4, 5, 6]
    assert spearman(x, x) == 1
    assert spearman(x, x[::-1]) == -1
    control = [2, 1, 4, 3, 6, 5]
    observed = partial_spearman(x, x[::-1], [control])
    assert math.isclose(observed, -1, abs_tol=1e-12)


def test_patch_entropy_and_tv_keep_nonadditive_spatial_objects_separate():
    left = np.zeros((3, 224, 224), dtype=np.float32); right = left.copy()
    left[:, :16, :16] = 1; right[:, -16:, -16:] = 1
    p, q = patch_distribution(left), patch_distribution(right)
    assert entropy(p) == 0 and entropy(q) == 0
    assert total_variation(p, q) == 1
    assert patch_distribution(np.zeros_like(left)) is None
