"""Focused model-free tests for the Stage D source manifest verifier."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import vitnns.stage_d_manifest as manifest_module
from vitnns.stage_d_manifest import (
    D1_COMPLETION_COMMIT,
    D1_COMPLETION_REF,
    D2_COMPLETION_COMMIT,
    D2_COMPLETION_REF,
    StageDManifest,
    StageDManifestError,
    verify_stage_d_manifest,
)
from vitnns.stage_d_registry import (
    D0_COMPLETION_COMMIT,
    D0_COMPLETION_REF,
    PACKAGE_DIR,
    PACKAGE_FILES,
    canonical_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config/stage_d.json"


def _config() -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _fake_c10_result(binding: dict) -> dict:
    file_fields = {
        "status.json": "status_file_sha256",
        "stage_c_report.json": "report_file_sha256",
        "decision.json": "decision_file_sha256",
        "holm.json": "holm_file_sha256",
        "provenance.json": "provenance_file_sha256",
        "manifest.json": "manifest_file_sha256",
    }
    canonical_fields = {
        "status.json": "status_canonical_digest",
        "stage_c_report.json": "report_canonical_digest",
        "decision.json": "decision_canonical_digest",
        "holm.json": "holm_canonical_digest",
        "provenance.json": "provenance_canonical_digest",
        "manifest.json": "manifest_canonical_digest",
    }
    return {
        "status": "PASS",
        "root": binding["formal_root"],
        "file_count": binding["whole_root_file_count"],
        "total_bytes": binding["whole_root_total_bytes"],
        "whole_root_inventory_digest": binding["whole_root_inventory_digest"],
        "file_sha256": {
            filename: binding[field] for filename, field in file_fields.items()
        },
        "canonical_digests": {
            filename: binding[field]
            for filename, field in canonical_fields.items()
        },
        "archive_ref": binding["producer_ref"],
        "producer_commit": binding["producer_commit"],
        "producer_tree": binding["producer_tree"],
        "source_tree": binding["producer_source_tree"],
        "scientific_results_recomputed": False,
    }


@pytest.fixture
def fake_c10(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, object]]:
    calls: list[tuple[object, object]] = []

    def verify(root, *, repo_root):
        calls.append((root, repo_root))
        return _fake_c10_result(_config()["upstream"]["c10"])

    monkeypatch.setattr(manifest_module, "verify_c10_sealed_evidence", verify)
    return calls


def test_checked_in_sources_produce_immutable_content_addressed_snapshot(
    fake_c10: list[tuple[object, object]],
) -> None:
    class LoadedConfig:
        def __init__(self, value: dict) -> None:
            self.value = value

        def to_dict(self) -> dict:
            return copy.deepcopy(self.value)

    result = verify_stage_d_manifest(LoadedConfig(_config()), repo_root=REPO_ROOT)

    assert result.status == "PASS"
    assert result.d0["completion_commit"] == D0_COMPLETION_COMMIT
    assert result.d1["completion_commit"] == D1_COMPLETION_COMMIT
    assert result.d2["completion_commit"] == D2_COMPLETION_COMMIT
    assert result.d1["registry_row_count"] == 500
    assert result.d1["package_file_count"] == 6
    assert result.c10["verification_status"] == "PASS"
    assert result.c10["primary_decision"] == "NO_GO_NOVELTY"
    assert result.c10["stage_c_acceptance"] == "BLOCKED"
    assert result.c10["workflow_blocker"] == "HOLM_BLOCKED_UNDERSPECIFIED"
    assert result.c10["spatial_status"] == "FAIL_SPATIAL"
    assert result.c10["downstream_authorized"] is False
    assert result.c10["stage_d_started"] is False
    assert result.c10["scientific_results_recomputed"] is False
    assert len(result.registry_rows) == 500
    assert result.registry is result.registry_document
    assert result.registry_snapshot is result.registry_document
    assert result.registry_document["rows"] is result.registry_rows
    assert result.registry_rows[0]["selection_order"] == 0
    assert result.registry_rows[-1]["selection_order"] == 499
    assert len(fake_c10) == 1

    exported = result.to_dict()
    digest = exported.pop("canonical_digest")
    assert digest == result.canonical_digest == canonical_sha256(exported)
    assert set(result) == {
        "schema_version", "document_type", "status", "bindings",
        "registry_document", "canonical_digest",
    }
    with pytest.raises(TypeError):
        result.bindings["d0"]["completion_commit"] = "0" * 40
    with pytest.raises(TypeError):
        result.registry_rows[0]["sample_id"] = "changed"
    exported["bindings"]["d0"]["completion_commit"] = "0" * 40
    exported["registry_document"]["rows"][0]["sample_id"] = "changed"
    assert result.d0["completion_commit"] == D0_COMPLETION_COMMIT
    assert result.registry_rows[0]["sample_id"] != "changed"


def test_package_snapshot_validation_is_internal_only_and_does_not_rebuild_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    calls: list[tuple[object, object, bool, bool]] = []
    original = manifest_module.validate_stage_d_registry_package_bytes

    def validate(payloads, *, repo_root, verify_sources, verify_rebuild):
        calls.append((set(payloads), repo_root, verify_sources, verify_rebuild))
        return original(
            payloads, repo_root=repo_root,
            verify_sources=verify_sources, verify_rebuild=verify_rebuild,
        )

    monkeypatch.setattr(
        manifest_module, "validate_stage_d_registry_package_bytes", validate,
    )
    monkeypatch.setattr(
        manifest_module, "verify_c10_sealed_evidence",
        lambda root, *, repo_root: _fake_c10_result(config["upstream"]["c10"]),
    )
    result = verify_stage_d_manifest(config, repo_root=REPO_ROOT)
    assert result.status == "PASS"
    assert calls == [(
        set(PACKAGE_FILES),
        REPO_ROOT,
        False,
        False,
    )]


def test_optional_c10_mode_still_preserves_exact_frozen_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("sealed C10 verifier must not run")

    monkeypatch.setattr(manifest_module, "verify_c10_sealed_evidence", forbidden)
    result = verify_stage_d_manifest(_config(), repo_root=REPO_ROOT, verify_c10=False)
    assert result.c10["verification_status"] == "CONFIG_BINDINGS_ONLY"
    assert result.c10["primary_decision"] == "NO_GO_NOVELTY"
    assert result.c10["stage_c_acceptance"] == "BLOCKED"
    assert result.c10["workflow_blocker"] == "HOLM_BLOCKED_UNDERSPECIFIED"
    assert result.c10["spatial_status"] == "FAIL_SPATIAL"
    assert result.c10["downstream_authorized"] is False
    assert result.c10["stage_d_started"] is False
    assert result.c10["scientific_results_recomputed"] is False


def test_manifest_cannot_be_constructed_outside_the_verifier() -> None:
    with pytest.raises(TypeError, match="only be issued"):
        StageDManifest(
            schema_version=True,
            document_type="stage_d_verified_source_manifest",
            status="PASS",
            bindings={},
            registry_document={"rows": [{}] * 500},
            canonical_digest="0" * 64,
        )


def test_loaded_config_source_drift_is_rejected_before_upstream_use() -> None:
    changed = _config()
    changed["upstream"]["d0"]["completion_commit"] = "0" * 40
    with pytest.raises(
        StageDManifestError, match="differs from the safely bound D2 config bytes",
    ):
        verify_stage_d_manifest(changed, repo_root=REPO_ROOT, verify_c10=False)


def test_d0_source_file_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative in (
        manifest_module.D0_MACHINE_PATH,
        manifest_module.D0_SCHEMA_PATH,
        manifest_module.D0_DOCUMENT_PATH,
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    document = tmp_path / manifest_module.D0_DOCUMENT_PATH
    document.write_bytes(document.read_bytes() + b"drift")
    monkeypatch.setattr(
        manifest_module, "_verify_completion_ref",
        lambda root, *, stage, reference, expected: expected,
    )
    with pytest.raises(StageDManifestError, match="D0 document file SHA-256 drifted"):
        manifest_module._verify_d0(_config(), tmp_path)


def test_d2_config_file_symlink_is_rejected(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    outside = tmp_path / "stage_d.outside.json"
    shutil.copyfile(CONFIG_PATH, outside)
    (config_dir / "stage_d.json").symlink_to(outside)
    shutil.copyfile(
        REPO_ROOT / manifest_module.D2_CONFIG_SCHEMA_PATH,
        config_dir / "stage_d.schema.json",
    )
    with pytest.raises(
        StageDManifestError, match="single-link regular file|cannot safely open",
    ):
        manifest_module._verify_d2_config(_config(), tmp_path)


@pytest.mark.parametrize(
    "stage, target_ref",
    [("D0", D0_COMPLETION_REF), ("D1", D1_COMPLETION_REF),
     ("D2", D2_COMPLETION_REF)],
)
def test_each_completion_ref_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch, stage: str, target_ref: str,
) -> None:
    expected = {
        D0_COMPLETION_REF: D0_COMPLETION_COMMIT,
        D1_COMPLETION_REF: D1_COMPLETION_COMMIT,
        D2_COMPLETION_REF: D2_COMPLETION_COMMIT,
    }

    def resolve(repo_root, revision):
        reference = revision.removesuffix("^{commit}")
        return "0" * 40 if reference == target_ref else expected[reference]

    monkeypatch.setattr(manifest_module, "_git_rev_parse", resolve)
    with pytest.raises(StageDManifestError, match=f"{stage} completion ref drifted"):
        verify_stage_d_manifest(_config(), repo_root=REPO_ROOT, verify_c10=False)


def test_git_queries_ignore_ambient_repository_redirection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        monkeypatch.setenv(name, "/definitely/not/the/stage-d-repository")
    assert manifest_module._git_rev_parse(
        REPO_ROOT, f"{D2_COMPLETION_REF}^{{commit}}",
    ) == D2_COMPLETION_COMMIT


def _copy_package(repo_root: Path) -> Path:
    target = repo_root / "manifests/stage_d_fidelity_v1"
    target.mkdir(parents=True)
    for name in PACKAGE_FILES:
        shutil.copyfile(PACKAGE_DIR / name, target / name)
    return target


def _isolate_d1(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path,
) -> None:
    monkeypatch.setattr(
        manifest_module, "_verify_repository_identity", lambda root: None,
    )
    monkeypatch.setattr(
        manifest_module, "_verify_d2_config",
        lambda config, root: {"verification_status": "PASS"},
    )
    monkeypatch.setattr(
        manifest_module, "_verify_d0",
        lambda config, root: {"verification_status": "PASS"},
    )
    monkeypatch.setattr(
        manifest_module, "_verify_c10_binding",
        lambda config, root, verify_full: {
            "verification_status": "CONFIG_BINDINGS_ONLY",
            "scientific_results_recomputed": False,
        },
    )
    monkeypatch.setattr(
        manifest_module, "_verify_completion_ref",
        lambda root, *, stage, reference, expected: expected,
    )


def test_d1_package_byte_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _copy_package(tmp_path)
    _isolate_d1(monkeypatch, tmp_path)
    victim = package / "registry_metadata.json"
    victim.write_bytes(victim.read_bytes() + b" ")
    with pytest.raises(
        StageDManifestError, match="D1 registry package validation failed",
    ):
        verify_stage_d_manifest(_config(), repo_root=tmp_path, verify_c10=False)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_d1_snapshot_rejects_symlink_and_hardlink_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link_kind: str,
) -> None:
    package = _copy_package(tmp_path)
    _isolate_d1(monkeypatch, tmp_path)
    victim = package / "registry_metadata.json"
    outside = tmp_path / "registry_metadata.outside.json"
    shutil.copyfile(victim, outside)
    victim.unlink()
    if link_kind == "symlink":
        victim.symlink_to(outside)
    else:
        os.link(outside, victim)
    with pytest.raises(
        StageDManifestError,
        match="single-link regular file|cannot safely open metadata file",
    ):
        verify_stage_d_manifest(_config(), repo_root=tmp_path, verify_c10=False)


def test_single_stream_reader_detects_toctou_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"{}\n")
    root_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_fstat = manifest_module.os.fstat
    calls = 0

    def changed_after_read(fd):
        nonlocal calls
        value = original_fstat(fd)
        calls += 1
        if calls != 2:
            return value
        return SimpleNamespace(
            st_dev=value.st_dev, st_ino=value.st_ino, st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns + 1, st_ctime_ns=value.st_ctime_ns,
            st_nlink=value.st_nlink, st_mode=value.st_mode,
        )

    monkeypatch.setattr(manifest_module.os, "fstat", changed_after_read)
    try:
        with pytest.raises(StageDManifestError, match="changed while being read"):
            manifest_module._read_relative(root_fd, "source.json")
    finally:
        os.close(root_fd)


def test_single_stream_reader_rejects_oversized_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"{}\n")
    monkeypatch.setattr(manifest_module, "_MAX_METADATA_FILE_BYTES", 1)
    root_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(StageDManifestError, match="exceeds safe size limit"):
            manifest_module._read_relative(root_fd, "source.json")
    finally:
        os.close(root_fd)


def test_package_snapshot_detects_inventory_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = manifest_module.os.listdir
    calls = 0

    def changing_inventory(path):
        nonlocal calls
        calls += 1
        names = original(path)
        return names if calls == 1 else [*names, "unexpected.json"]

    monkeypatch.setattr(manifest_module.os, "listdir", changing_inventory)
    with pytest.raises(
        StageDManifestError, match="inventory changed while taking snapshot",
    ):
        manifest_module._read_package_snapshot(REPO_ROOT)


def test_full_c10_verifier_summary_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    bad = _fake_c10_result(config["upstream"]["c10"])
    bad["whole_root_inventory_digest"] = "0" * 64
    monkeypatch.setattr(
        manifest_module, "verify_c10_sealed_evidence",
        lambda root, *, repo_root: bad,
    )
    with pytest.raises(
        StageDManifestError, match="sealed C10 verifier binding drifted",
    ):
        verify_stage_d_manifest(config, repo_root=REPO_ROOT)
