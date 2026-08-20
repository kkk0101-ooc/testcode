import hashlib
import json
from dataclasses import FrozenInstanceError
from dataclasses import replace

import numpy as np
import pytest

from vitnns.retention import (
    ArtifactKind,
    ArtifactWriter,
    BudgetExceededError,
    BudgetTracker,
    IncompleteArtifactError,
    OutputDirectoryNotEmptyError,
    PolicyViolationError,
    RetentionPolicy,
    audit_replay_report,
    estimate_output_bytes,
    get_retention_policy,
    paper_float16_roundtrip_error,
)
from vitnns.display import DisplayTransform, metric_ready


def test_profiles_are_immutable_and_have_expected_boundaries():
    paper = get_retention_policy("paper")
    audit = get_retention_policy("audit")
    debug = get_retention_policy("debug")

    assert isinstance(paper, RetentionPolicy)
    assert paper.allowed_kinds == frozenset({ArtifactKind.METRIC_READY})
    assert paper.metric_storage_dtype == "float16"
    assert paper.metric_ndim == 2
    assert not paper.allow_png
    assert not paper.allow_per_pixel_files
    assert not paper.allow_per_dimension_files
    assert audit.allowed_kinds == frozenset(
        {ArtifactKind.RAW, ArtifactKind.METRIC_READY}
    )
    assert audit.raw_storage_dtype == "float32"
    assert audit.metric_storage_dtype == "float32"
    assert audit.raw_rgb_required
    assert debug.allowed_kinds == frozenset(ArtifactKind)
    assert debug.allow_png

    with pytest.raises(FrozenInstanceError):
        paper.allow_png = True
    with pytest.raises(PolicyViolationError, match="custom policy overrides"):
        get_retention_policy(replace(paper, allow_png=True))
    with pytest.raises(ValueError, match="Unknown retention profile"):
        get_retention_policy("production-ish")


def test_static_estimate_accounts_for_profile_shape_and_multiplicity():
    paper_one = estimate_output_bytes(1, (10, 20), "paper")
    paper_many = estimate_output_bytes(2, (10, 20), "paper", head_count=3, target_count=2)
    audit_one = estimate_output_bytes(1, (10, 20), "audit")
    audit_rgb_shape = estimate_output_bytes(1, (3, 10, 20), "audit")

    assert paper_one > 10 * 20 * np.dtype(np.float16).itemsize
    assert paper_many > paper_one
    assert audit_one > paper_one
    assert audit_rgb_shape == audit_one
    with pytest.raises(ValueError, match="shape must be"):
        estimate_output_bytes(1, (2, 10, 20), "paper")


def test_paper_writes_only_float16_metric_ready_2d_and_no_png(tmp_path):
    source = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    writer = ArtifactWriter(tmp_path / "paper", "paper", budget_bytes=100_000)

    record = writer.write_npz(ArtifactKind.METRIC_READY, "maps", {"map": source})
    metrics_record = writer.write_json(
        ArtifactKind.METRIC_READY, "metrics", {"entropy": 0.5}
    )
    completed = writer.finalize()

    assert record["kind"] == "metric_ready"
    assert record["dtype"] == "float16"
    assert record["shape"] == [3, 4]
    assert metrics_record["kind"] == "metric_ready"
    with np.load(tmp_path / "paper" / record["path"]) as saved:
        assert saved["map"].dtype == np.float16
        np.testing.assert_array_equal(saved["map"], source.astype(np.float16))
    assert completed["status"] == "complete"
    assert json.loads(writer.status_path.read_text())["status"] == "complete"
    assert not list((tmp_path / "paper").rglob("*.png"))

    blocked = ArtifactWriter(tmp_path / "blocked", "paper", budget_bytes=100_000)
    with pytest.raises(PolicyViolationError, match="does not allow"):
        blocked.write_npz(ArtifactKind.RAW, "raw", {"map": source})
    assert blocked.status == "incomplete"


def test_paper_rejects_finite_float32_value_that_overflows_float16(tmp_path):
    writer = ArtifactWriter(tmp_path / "overflow", "paper", 100_000)
    metric = np.full((2, 2), np.finfo(np.float32).max, dtype=np.float32)
    with pytest.raises(PolicyViolationError, match="representable as finite"):
        writer.write_npz(ArtifactKind.METRIC_READY, "map", {"map": metric})
    assert not (writer.output_dir / "metric_ready" / "map.npz").exists()
    assert writer.status == "incomplete"


@pytest.mark.parametrize(
    ("name", "shape", "granularity"),
    [
        ("map.png", (3, 4), "aggregate"),
        ("pixel_12", (3, 4), "per_pixel"),
        ("dimension_9", (3, 4), "per_dimension"),
        ("high_dim", (2, 3, 4), "aggregate"),
    ],
)
def test_paper_rejects_png_granular_and_high_dimensional_outputs(
    tmp_path, name, shape, granularity
):
    writer = ArtifactWriter(tmp_path / name.replace(".", "_"), "paper", 100_000)
    array = np.zeros(shape, dtype=np.float32)
    with pytest.raises(PolicyViolationError):
        writer.write_npz(
            ArtifactKind.METRIC_READY, name, {"map": array}, granularity=granularity
        )
    assert writer.status == "incomplete"


def test_audit_persists_float32_rgb_and_metric_and_reports_roundtrip(tmp_path):
    raw = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5) / 17
    metric = raw.mean(axis=0)
    writer = ArtifactWriter(tmp_path / "audit", "audit", budget_bytes=100_000)

    raw_record = writer.write_npz(ArtifactKind.RAW, "raw", {"rgb": raw})
    metric_record = writer.write_npz(
        ArtifactKind.METRIC_READY, "metric", {"attribution": metric}
    )
    completed = writer.finalize()

    assert raw_record["dtype"] == "float32"
    assert raw_record["shape"] == [3, 4, 5]
    assert metric_record["dtype"] == "float32"
    expected_error = paper_float16_roundtrip_error(metric_ready(metric.astype(np.float32)))
    assert metric_record["paper_float16_roundtrip_error"]["attribution"] == expected_error
    with np.load(tmp_path / "audit" / raw_record["path"]) as saved:
        assert saved["rgb"].dtype == np.float32
    assert completed["status"] == "complete"


def test_audit_replay_report_passes_and_checks_paper_roundtrip():
    raw = (np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) - 20) / 31
    metric = raw.sum(axis=0, dtype=np.float32)

    report = audit_replay_report(raw, metric)

    assert report["raw_shape"] == [3, 4, 5]
    assert report["metric_shape"] == [4, 5]
    assert report["max_abs_error"] == 0.0
    assert report["rmse"] == 0.0
    assert report["raw_replay_pass"] is True
    assert report["paper_replay_pass"] is True
    assert report["pass"] is True
    assert report["paper_float16_roundtrip_error"]["storage_dtype"] == "float16"


def test_audit_replay_report_detects_tampered_metric():
    raw = np.linspace(-1, 1, 3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    metric = raw.sum(axis=0, dtype=np.float32)
    tampered = metric.copy()
    tampered[1, 2] += 0.1

    report = audit_replay_report(raw, tampered)

    assert report["max_abs_error"] == pytest.approx(0.1, abs=1e-7)
    assert report["rmse"] > 0.0
    assert report["raw_replay_pass"] is False
    assert report["pass"] is False


@pytest.mark.parametrize(
    ("raw", "metric", "match"),
    [
        (np.zeros((3, 2, 2), np.float64), np.zeros((2, 2), np.float32), "raw RGB must be float32"),
        (np.zeros((3, 2, 2), np.float32), np.zeros((2, 2), np.float64), "metric map must be float32"),
        (np.zeros((2, 2, 3), np.float32), np.zeros((2, 2), np.float32), r"shape \[3,H,W\]"),
        (np.zeros((3, 2, 2), np.float32), np.zeros((3, 2), np.float32), "spatial shapes differ"),
    ],
)
def test_audit_replay_report_validates_dtypes_and_shapes(raw, metric, match):
    with pytest.raises(PolicyViolationError, match=match):
        audit_replay_report(raw, metric)


def test_audit_requires_both_raw_and_metric_npz_to_complete(tmp_path):
    writer = ArtifactWriter(tmp_path / "audit", "audit", 100_000)
    writer.write_npz(
        ArtifactKind.METRIC_READY, "metric", {"map": np.zeros((4, 5), np.float32)}
    )
    with pytest.raises(IncompleteArtifactError, match="raw"):
        writer.finalize()
    assert writer.status == "incomplete"
    assert json.loads(writer.status_path.read_text())["status"] == "incomplete"


def test_audit_writer_requires_channel_first_raw_rgb(tmp_path):
    writer = ArtifactWriter(tmp_path / "audit-hwc", "audit", 100_000)
    with pytest.raises(PolicyViolationError, match=r"RGB \[3,H,W\]"):
        writer.write_npz(
            ArtifactKind.RAW, "raw", {"map": np.zeros((4, 5, 3), np.float32)}
        )


def test_budget_exhaustion_is_checked_before_artifact_write_and_cannot_complete(tmp_path):
    output = tmp_path / "small-budget"
    budget = BudgetTracker(65_536)
    writer = ArtifactWriter(output, "paper", budget)

    oversized = np.linspace(0, 1, 300_000, dtype=np.float32).reshape(500, 600)
    with pytest.raises(BudgetExceededError):
        writer.write_npz(
            ArtifactKind.METRIC_READY,
            "map",
            {"map": oversized},
            compressed=False,
        )
    assert not (output / "metric_ready" / "map.npz").exists()
    assert writer.status == "incomplete"
    assert budget.actual_bytes == 0
    assert budget.failed
    with pytest.raises(IncompleteArtifactError):
        writer.finalize()
    assert json.loads(writer.status_path.read_text())["status"] == "incomplete"


def test_manifest_records_exact_bytes_file_hash_dtype_shape_and_kind(tmp_path):
    writer = ArtifactWriter(tmp_path / "output", "debug", 100_000)
    record = writer.write_npz(
        ArtifactKind.RAW, "tensor", {"x": np.ones((2, 3), dtype=np.float32)}
    )
    path = writer.output_dir / record["path"]

    assert record["actual_bytes"] == path.stat().st_size
    assert record["bytes"] == path.stat().st_size
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["dtype"] == "float32"
    assert record["shape"] == [2, 3]
    assert record["kind"] == "raw"
    assert writer.budget.actual_bytes == path.stat().st_size
    writer.finalize()
    assert writer.actual_bytes == sum(
        candidate.stat().st_size
        for candidate in writer.output_dir.rglob("*")
        if candidate.is_file()
    )
    assert writer.actual_bytes <= writer.budget.max_bytes
    assert writer.actual_bytes > writer.budget.actual_bytes
    status = json.loads(writer.status_path.read_text())
    assert status["manifest_sha256"] == hashlib.sha256(
        writer.manifest_path.read_bytes()
    ).hexdigest()


def test_json_writer_rejects_nested_display_artifact(tmp_path):
    display = DisplayTransform()(metric_ready(np.ones((2, 3), dtype=np.float32)))
    writer = ArtifactWriter(tmp_path / "json-boundary", "debug", 100_000)
    with pytest.raises(PolicyViolationError, match="tagged numerical artifacts"):
        writer.write_json(ArtifactKind.METRIC_READY, "metrics", {"map": display})
    assert not (writer.output_dir / "metric_ready" / "metrics.json").exists()
    assert writer.status == "incomplete"


def test_paper_roundtrip_metric_rejects_untagged_and_display_arrays():
    array = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(PolicyViolationError, match="explicitly tagged"):
        paper_float16_roundtrip_error(array)
    display = DisplayTransform()(metric_ready(array))
    with pytest.raises(PolicyViolationError, match="got display"):
        paper_float16_roundtrip_error(display)


def test_non_empty_output_directory_is_refused(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("owned by caller")

    with pytest.raises(OutputDirectoryNotEmptyError):
        ArtifactWriter(output, "debug", 100_000)
    assert sentinel.read_text() == "owned by caller"


def test_empty_writer_never_marks_complete(tmp_path):
    writer = ArtifactWriter(tmp_path / "empty", "debug", 100_000)
    with pytest.raises(IncompleteArtifactError, match="empty"):
        writer.finalize()
    assert writer.status == "incomplete"
    assert json.loads(writer.manifest_path.read_text())["complete"] is False
