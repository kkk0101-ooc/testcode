import numpy as np
import pytest

from vitnns.display import (
    DisplayBoundaryError,
    DisplayTransform,
    metric_ready,
    raw,
    validate_display_input,
    validate_metric_input,
    validate_raw_input,
)
from vitnns.retention import ArtifactArray, ArtifactKind, array_sha256


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_default_display_transform_is_identity_without_mutation_or_hash_change(dtype):
    source = np.arange(20, dtype=dtype).reshape(4, 5)
    before = source.copy()
    before_hash = array_sha256(source)
    tagged = metric_ready(source)

    transform = DisplayTransform()
    displayed = transform.apply(tagged)

    assert transform.is_identity
    assert transform.legacy_fusion_weight is None
    assert transform.percentile_clip is None
    assert transform.max_pool_size is None
    assert transform.gaussian_sigma is None
    assert transform.resize is None
    assert displayed.kind is ArtifactKind.DISPLAY
    assert displayed.transforms == ("identity",)
    assert displayed.metadata["transformed"] is False
    assert displayed.metadata["native_shape"] == (4, 5)
    assert displayed.metadata["display_shape"] == (4, 5)
    assert displayed.dtype == source.dtype
    np.testing.assert_array_equal(displayed.data, before)
    np.testing.assert_array_equal(source, before)
    assert array_sha256(source) == before_hash
    assert displayed.sha256 == before_hash
    assert not np.shares_memory(displayed.data, source)


def test_all_legacy_display_operations_are_explicit_and_marked():
    attention = metric_ready(np.arange(16, dtype=np.float32).reshape(4, 4))
    class_feature = metric_ready(np.flipud(np.arange(16, dtype=np.float32).reshape(4, 4)))
    transform = DisplayTransform(
        legacy_fusion_weight=0.7,
        percentile_clip=(1, 99),
        max_pool_size=2,
        gaussian_sigma=1.0,
        resize=(8, 6),
        resize_order=1,
    )

    displayed = transform(attention, class_feature=class_feature)

    assert displayed.kind is ArtifactKind.DISPLAY
    assert displayed.transforms == (
        "legacy_fusion",
        "percentile_clip",
        "max_pool",
        "gaussian_smoothing",
        "resize",
    )
    assert displayed.shape == (8, 6)
    assert displayed.metadata["native_shape"] == (4, 4)
    assert displayed.metadata["display_shape"] == (8, 6)
    fusion = displayed.metadata["transforms"][0]
    assert fusion["class_feature_weight"] == pytest.approx(0.7)
    assert fusion["attention_weight"] == pytest.approx(0.3)


def test_metric_validation_rejects_display_input():
    metric = metric_ready(np.ones((3, 4), dtype=np.float32))
    display = DisplayTransform(resize=(6, 8))(metric)

    with pytest.raises(DisplayBoundaryError, match="rejects display"):
        validate_metric_input(display)
    with pytest.raises(DisplayBoundaryError, match="explicitly tagged"):
        validate_metric_input(display.data)
    with pytest.raises(DisplayBoundaryError, match="explicitly tagged"):
        validate_metric_input(np.asarray(display))


def test_tagged_arrays_and_nested_metadata_are_immutable():
    source = np.ones((3, 4), dtype=np.float32)
    tagged = metric_ready(source)
    display = DisplayTransform()(tagged)

    with pytest.raises(ValueError):
        tagged.data.setflags(write=True)
    with pytest.raises(ValueError):
        display.data.setflags(write=True)
    with pytest.raises(TypeError):
        display.metadata["transforms"][0]["name"] = "tampered"


def test_display_transform_rejects_raw_and_already_display_data():
    raw = ArtifactArray(np.ones((3, 4), dtype=np.float32), ArtifactKind.RAW)
    with pytest.raises(DisplayBoundaryError, match="metric_ready"):
        DisplayTransform()(raw)

    display = DisplayTransform()(metric_ready(np.ones((3, 4), dtype=np.float32)))
    with pytest.raises(DisplayBoundaryError, match="display data"):
        DisplayTransform()(display)


def test_disabled_fusion_rejects_unused_class_feature():
    metric = metric_ready(np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="disabled"):
        DisplayTransform()(metric, class_feature=metric)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"legacy_fusion_weight": -0.01},
        {"legacy_fusion_weight": 1.01},
        {"percentile_clip": (99, 1)},
        {"max_pool_size": 0},
        {"gaussian_sigma": -1},
        {"resize": (0, 8)},
        {"resize_order": 6},
    ],
)
def test_invalid_display_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        DisplayTransform(**kwargs)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 3, 4), dtype=np.float32),
        np.array([[0.0, np.nan]], dtype=np.float32),
        np.array([[0.0, np.inf]], dtype=np.float32),
    ],
)
def test_metric_ready_rejects_non_2d_or_non_finite_data(bad):
    with pytest.raises((TypeError, ValueError)):
        metric_ready(bad)


def test_raw_metric_ready_and_display_identities_cannot_be_retagged():
    raw_rgb = raw(np.zeros((8, 9, 3), dtype=np.float32))
    metric = metric_ready(np.ones((3, 4), dtype=np.float32))
    display = DisplayTransform()(metric)

    assert validate_raw_input(raw_rgb).shape == (8, 9, 3)
    assert validate_display_input(display).shape == (3, 4)
    with pytest.raises(DisplayBoundaryError, match="cannot relabel display"):
        metric_ready(display)
    with pytest.raises(DisplayBoundaryError, match="cannot relabel raw"):
        metric_ready(raw_rgb)
    with pytest.raises(DisplayBoundaryError, match="requires display"):
        validate_display_input(metric)
    with pytest.raises(DisplayBoundaryError, match="requires raw"):
        validate_raw_input(display)
