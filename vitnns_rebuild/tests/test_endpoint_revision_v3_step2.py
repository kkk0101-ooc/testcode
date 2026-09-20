import numpy as np
import pytest
import torch

from vitnns.endpoint_revision_v3_step2 import (
    amplitude_errors, bootstrap_median, independently_sampled_channels,
    independently_sampled_tokens, interpolate, qkv_map_metrics, qkv_metrics, spearman,
)


def test_five_independent_token_and_channel_samples_are_frozen_and_distinct():
    token_groups = [independently_sampled_tokens(f"sample_{index}") for index in range(5)]
    assert all(len(group) == 16 and group[0] == 0 and len(set(group)) == 16 for group in token_groups)
    assert len(set(token_groups)) == 5
    channel_groups = [independently_sampled_channels(f"sample_{index}", "vit_b16", 5, 768) for index in range(5)]
    assert all(len(group) == len(set(group)) == 32 for group in channel_groups)
    assert len(set(channel_groups)) == 5


def test_interpolation_and_taylor_error_contract():
    clean = torch.tensor([1.0, 3.0]); reference = torch.tensor([5.0, -1.0])
    torch.testing.assert_close(interpolate(clean, reference, 0.25), torch.tensor([2.0, 2.0]))
    row = amplitude_errors(effect=0.4, p_ref=1.0, alpha=0.25)
    assert row == pytest.approx({"taylor_prediction": 0.25, "absolute_error": 0.15, "relative_error": 0.375})
    assert amplitude_errors(effect=0.0, p_ref=1.0, alpha=0.5)["relative_error"] is None


def test_qkv_spatial_metrics_and_relative_scale():
    point = np.zeros((3, 224, 224), dtype=np.float32); point[0, 8, 8] = 2.0
    spread = np.ones((3, 224, 224), dtype=np.float32)
    point_metrics = qkv_map_metrics(point); spread_metrics = qkv_map_metrics(spread)
    assert point_metrics["normalized_spatial_entropy"] == pytest.approx(0.0)
    assert point_metrics["normalized_pairwise_distance"] == pytest.approx(0.0)
    assert spread_metrics["normalized_spatial_entropy"] == pytest.approx(1.0)
    assert spread_metrics["normalized_pairwise_distance"] > 0.3
    rows = qkv_metrics({"Q": point, "K": 2 * point, "V": 3 * point})
    assert sum(row["relative_scale"] for row in rows.values()) == pytest.approx(1.0)
    assert rows["V"]["relative_scale"] == pytest.approx(0.5)


def test_rank_and_bootstrap_are_deterministic():
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert bootstrap_median(range(9), "fixture", 100) == bootstrap_median(range(9), "fixture", 100)
