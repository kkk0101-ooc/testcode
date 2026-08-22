from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vitnns.stage_c_artifacts import (
    ArtifactBudgetError,
    ArtifactPolicyError,
    ArtifactWriterStateError,
    AttemptNotEmptyError,
    InjectedArtifactCrash,
    StageCArtifactWriter,
    attempt_directories,
    next_attempt_directory,
    unit_identity_sha256,
    validate_execution_unit,
)


def identity() -> dict[str, object]:
    return {"sample_id": "sample-a", "method_id": "toy"}


def unit_key_identity() -> dict[str, object]:
    semantic: dict[str, object] = {
        "schema_version": 1,
        "sample_id": "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG",
        "method_id": "vit_grad_cam_block11_norm1",
        "config_id": "global_vit_grad_cam_block11_norm1",
        "block_index": None,
        "head_index": None,
        "target_ref": "clean_predicted_class_raw_logit",
    }
    semantic["canonical_digest"] = hashlib.sha256(
        json.dumps(
            semantic, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return semantic


def provenance() -> dict[str, object]:
    return {
        "code": {"commit": "a" * 40, "dirty": True},
        "resources": {"wall_seconds": 0.1},
    }


def metric_map() -> np.ndarray:
    return np.arange(196, dtype=np.float32).reshape(14, 14)


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def complete_unit(
    root: Path, *, audit: bool = False, l2: bool = False, l3: bool = False
) -> tuple[Path, StageCArtifactWriter]:
    attempt = next_attempt_directory(root)
    writer = StageCArtifactWriter(
        attempt, unit_identity=identity(), audit=audit
    )
    writer.write_l0("result", {"status": "OK", "score": 0.5})
    writer.write_l1("metric_map", metric_map())
    if l2:
        writer.write_l2(
            "audit_bundle",
            {
                "attention": np.zeros((2, 197, 197), np.float32),
                "endpoint": np.ones((64,), np.float32),
            },
            kind="intermediate",
        )
    if l3:
        writer.write_l3("display", png_bytes())
    writer.finalize(provenance=provenance())
    return attempt, writer


def rewrite_json(path: Path, transform) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    transform(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_complete_inventory_and_identity_provenance_bindings(tmp_path):
    attempt, writer = complete_unit(
        tmp_path / "unit", audit=True, l2=True, l3=True
    )
    report = validate_execution_unit(
        attempt,
        expected_unit_identity=identity(),
        expected_unit_digest=unit_identity_sha256(identity()),
        expected_provenance_bindings={"code": {"dirty": True}},
    )
    assert report.valid and report.complete and report.terminal
    assert report.resumable and report.status == "complete"
    manifest = json.loads(writer.manifest_path.read_text())
    status = json.loads(writer.status_path.read_text())
    assert status["manifest_sha256"] == hashlib.sha256(
        writer.manifest_path.read_bytes()
    ).hexdigest()
    assert set(manifest["artifacts"][0]) == {
        "level", "path", "sha256", "bytes", "kind", "dtype", "shape", "format"
    }
    assert {item["level"] for item in manifest["artifacts"]} == {
        "L0", "L1", "L2", "L3"
    }
    for item in manifest["artifacts"]:
        payload = attempt / item["path"]
        assert payload.stat().st_size == item["bytes"]
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == item["sha256"]
    display = next(
        item for item in manifest["artifacts"] if item["level"] == "L3"
    )
    assert display["display_only"] is True
    assert display["metric_input"] is False


def test_unit_key_digest_is_shared_with_provenance_contract():
    value = unit_key_identity()
    assert unit_identity_sha256(value) == value["canonical_digest"]
    tampered = dict(value)
    tampered["sample_id"] = "ILSVRC2012_val/ILSVRC2012_val_00049970.JPEG"
    with pytest.raises(ValueError, match="canonical_digest"):
        unit_identity_sha256(tampered)


def test_production_unit_key_requires_full_stage_c_provenance(tmp_path):
    writer = StageCArtifactWriter(
        next_attempt_directory(tmp_path / "strict"),
        unit_identity=unit_key_identity(), audit=False,
    )
    writer.write_l0("result", {"status": "OK"})
    writer.write_l1("map", metric_map())
    with pytest.raises(ArtifactPolicyError, match="full Stage C provenance"):
        writer.finalize(provenance={"code": {"commit": "a" * 40}})


def test_attempts_are_fresh_sorted_and_never_reopened(tmp_path):
    root = tmp_path / "attempts"
    first = next_attempt_directory(root)
    StageCArtifactWriter(first, unit_identity=identity(), audit=False)
    second = next_attempt_directory(root)
    assert first.name == "attempt-000001"
    assert second.name == "attempt-000002"
    assert attempt_directories(root) == (first, second)
    with pytest.raises(AttemptNotEmptyError):
        StageCArtifactWriter(first, unit_identity=identity(), audit=False)


@pytest.mark.parametrize(
    "fault_point",
    ["after_running", "after_payload", "after_manifest", "before_complete_status"],
)
def test_crash_faults_never_form_a_resumable_attempt(tmp_path, fault_point):
    attempt = next_attempt_directory(tmp_path / fault_point)
    if fault_point == "after_running":
        with pytest.raises(InjectedArtifactCrash):
            StageCArtifactWriter(
                attempt, unit_identity=identity(), audit=False,
                fault_injector=fault_point,
            )
    else:
        writer = StageCArtifactWriter(
            attempt, unit_identity=identity(), audit=False,
            fault_injector=fault_point,
        )
        if fault_point == "after_payload":
            with pytest.raises(InjectedArtifactCrash):
                writer.write_l0("result", {"status": "OK"})
        else:
            writer.write_l0("result", {"status": "OK"})
            writer.write_l1("map", metric_map())
            with pytest.raises(InjectedArtifactCrash):
                writer.finalize(provenance=provenance())
    report = validate_execution_unit(attempt)
    assert not report.resumable
    if fault_point == "after_running":
        assert report.valid and report.status == "running"
    else:
        assert not report.valid
    fresh = next_attempt_directory(attempt.parent)
    assert fresh != attempt and not any(fresh.iterdir())


def test_explicit_failure_is_valid_terminal_but_never_resumable(tmp_path):
    attempt = next_attempt_directory(tmp_path / "unit")
    writer = StageCArtifactWriter(attempt, unit_identity=identity(), audit=False)
    failure = {"exception_type": "RuntimeError", "message": "boom"}
    writer.prepare_failure("backend_failed", failure)
    writer.fail(
        "backend_failed",
        failure,
        provenance={"artifact_inventory": list(writer.artifacts)},
    )
    report = validate_execution_unit(attempt)
    assert report.valid and report.terminal and report.status == "failed"
    assert not report.complete and not report.resumable
    with pytest.raises(ArtifactWriterStateError):
        writer.write_l0("late", {})


def test_failure_after_manifest_crash_is_not_terminal_or_resumable(tmp_path):
    attempt = next_attempt_directory(tmp_path / "failure-crash")
    writer = StageCArtifactWriter(
        attempt,
        unit_identity=identity(),
        audit=False,
        fault_injector="after_manifest",
    )
    writer.prepare_failure("backend_failed", {"message": "boom"})
    with pytest.raises(InjectedArtifactCrash):
        writer.fail(
            "backend_failed",
            {"message": "boom"},
            provenance={"artifact_inventory": list(writer.artifacts)},
        )
    report = validate_execution_unit(attempt)
    assert not report.valid and not report.resumable


def test_failure_terminal_checks_exact_tree(tmp_path):
    attempt = next_attempt_directory(tmp_path / "failure-extra")
    writer = StageCArtifactWriter(attempt, unit_identity=identity(), audit=False)
    writer.prepare_failure("backend_failed")
    (attempt / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ArtifactPolicyError, match="file set mismatch"):
        writer.fail("backend_failed", provenance={})
    assert json.loads(writer.status_path.read_text())["status"] == "running"


@pytest.mark.parametrize("target", ["manifest", "status", "artifact"])
def test_tamper_breaks_resume_chain(tmp_path, target):
    attempt, writer = complete_unit(tmp_path / target)
    if target == "manifest":
        rewrite_json(writer.manifest_path, lambda value: value.update(audit=True))
    elif target == "status":
        rewrite_json(writer.status_path, lambda value: value.update(status="running"))
    else:
        (attempt / "L1/metric_map.npy").write_bytes(b"tampered")
    report = validate_execution_unit(attempt)
    assert not report.valid and not report.resumable
    assert report.errors


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_or_extra_file_breaks_exact_inventory(tmp_path, mutation):
    attempt, _writer = complete_unit(tmp_path / mutation)
    if mutation == "missing":
        (attempt / "L1/metric_map.npy").unlink()
    else:
        (attempt / "extra.bin").write_bytes(b"extra")
    report = validate_execution_unit(attempt)
    assert not report.valid and not report.resumable
    assert any(mutation in error for error in report.errors)


def test_l1_l2_l3_boundaries(tmp_path):
    writer = StageCArtifactWriter(
        next_attempt_directory(tmp_path / "l1"),
        unit_identity=identity(), audit=False,
    )
    with pytest.raises(ArtifactPolicyError, match="float32"):
        writer.write_l1("bad", np.zeros((14, 14), np.float64))

    non_audit = StageCArtifactWriter(
        next_attempt_directory(tmp_path / "l2"),
        unit_identity=identity(), audit=False,
    )
    with pytest.raises(ArtifactPolicyError, match="audit"):
        non_audit.write_l2("raw", np.zeros((2, 3), np.float32))

    display = StageCArtifactWriter(
        next_attempt_directory(tmp_path / "l3"),
        unit_identity=identity(), audit=False,
    )
    with pytest.raises(ArtifactPolicyError, match="audit unit or failure"):
        display.write_l3("view", png_bytes())

    failed_display = StageCArtifactWriter(
        next_attempt_directory(tmp_path / "l3-failure"),
        unit_identity=identity(), audit=False,
    )
    failed_display.write_l3("view", png_bytes(), failure=True)
    failed_display.write_l0("result", {"status": "OK"})
    failed_display.write_l1("map", metric_map())
    with pytest.raises(ArtifactPolicyError, match="non-audit success"):
        failed_display.finalize(provenance=provenance())


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "pixel_003", "step-512", "dimension.9"]
)
def test_path_escape_and_granular_filenames_are_rejected(tmp_path, name):
    writer = StageCArtifactWriter(
        next_attempt_directory(tmp_path / name.replace("/", "_")),
        unit_identity=identity(), audit=False,
    )
    with pytest.raises(ArtifactPolicyError):
        writer.write_l1(name, metric_map())


def test_symlink_is_rejected_by_writer_and_validator(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactPolicyError, match="symlink"):
        next_attempt_directory(link / "units")

    attempt, _writer = complete_unit(tmp_path / "normal")
    (attempt / "L1/metric_map.npy").unlink()
    (attempt / "L1/metric_map.npy").symlink_to(tmp_path / "target")
    report = validate_execution_unit(attempt)
    assert not report.valid
    assert any("symlink" in error for error in report.errors)


def test_reserved_l0_names_and_kinds_are_rejected(tmp_path):
    for index, (name, kind) in enumerate(
        (("provenance", "record"), ("failure", "record"), ("other", "provenance"))
    ):
        writer = StageCArtifactWriter(
            next_attempt_directory(tmp_path / f"reserved-{index}"),
            unit_identity=identity(), audit=False,
        )
        with pytest.raises(ArtifactPolicyError, match="reserved"):
            writer.write_l0(name, {}, kind=kind)


def test_caps_count_controls_provenance_and_all_regular_files(tmp_path):
    attempt = next_attempt_directory(tmp_path / "cap")
    writer = StageCArtifactWriter(
        attempt, unit_identity=identity(), audit=False,
        l0_l1_cap_bytes=2_500,
    )
    writer.write_l0("result", {"payload": "x" * 10})
    writer.write_l1("map", metric_map())
    with pytest.raises(ArtifactBudgetError, match="exceed cap"):
        writer.finalize(provenance={"large": "x" * 1_000})
    status = json.loads(writer.status_path.read_text())
    assert status["status"] == "running" and status["complete"] is False
    report = validate_execution_unit(attempt)
    assert not report.valid and not report.resumable
    assert not (attempt / "L0/failure.json").exists()


def test_l2_cap_is_enforced_before_payload_publication(tmp_path):
    attempt = next_attempt_directory(tmp_path / "l2-cap")
    writer = StageCArtifactWriter(
        attempt, unit_identity=identity(), audit=True, l2_cap_bytes=16,
    )
    with pytest.raises(ArtifactBudgetError, match="L2 bytes"):
        writer.write_l2("raw", np.zeros((10,), np.float32))
    assert not (attempt / "L2/raw.npy").exists()
    assert json.loads(writer.status_path.read_text())["status"] == "failed"


def test_validator_detects_rehashed_manifest_policy_forgery(tmp_path):
    attempt, writer = complete_unit(tmp_path / "forged")
    manifest = json.loads(writer.manifest_path.read_text())
    l1 = next(item for item in manifest["artifacts"] if item["level"] == "L1")
    l1["dtype"] = "float64"
    writer.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = json.loads(writer.status_path.read_text())
    status["manifest_sha256"] = hashlib.sha256(
        writer.manifest_path.read_bytes()
    ).hexdigest()
    writer.status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = validate_execution_unit(attempt)
    assert not report.valid and not report.resumable
    assert any("dtype/shape" in error for error in report.errors)


def test_validator_rejects_non_audit_l2_even_if_chain_is_rehashed(tmp_path):
    attempt, writer = complete_unit(
        tmp_path / "l2-forged", audit=True, l2=True
    )
    manifest = json.loads(writer.manifest_path.read_text())
    manifest["audit"] = False
    writer.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = json.loads(writer.status_path.read_text())
    status["manifest_sha256"] = hashlib.sha256(
        writer.manifest_path.read_bytes()
    ).hexdigest()
    writer.status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = validate_execution_unit(attempt)
    assert not report.resumable
    assert any(
        "non-audit manifest contains L2" in item for item in report.errors
    )


def test_validator_reports_exact_l2_bytes_for_batch_sample_cap(tmp_path):
    attempt, writer = complete_unit(
        tmp_path / "l2-count", audit=True, l2=True
    )
    expected = sum(
        item["bytes"] for item in writer.artifacts if item["level"] == "L2"
    )
    report = validate_execution_unit(attempt)
    assert report.valid and report.l2_bytes == expected
    assert report.total_bytes == report.l0_l1_bytes + report.l2_bytes


def test_validator_rejects_rehashed_truncated_png(tmp_path):
    attempt, writer = complete_unit(
        tmp_path / "png-truncated", audit=True, l3=True
    )
    display_path = attempt / "L3/display.png"
    truncated = display_path.read_bytes()[:33]
    display_path.write_bytes(truncated)
    manifest = json.loads(writer.manifest_path.read_text())
    record = next(
        item for item in manifest["artifacts"] if item["level"] == "L3"
    )
    record["bytes"] = len(truncated)
    record["sha256"] = hashlib.sha256(truncated).hexdigest()
    writer.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = json.loads(writer.status_path.read_text())
    status["manifest_sha256"] = hashlib.sha256(
        writer.manifest_path.read_bytes()
    ).hexdigest()
    writer.status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = validate_execution_unit(attempt)
    assert not report.resumable
    assert any("truncated PNG" in item for item in report.errors)
