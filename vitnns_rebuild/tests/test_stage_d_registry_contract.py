"""Metadata-only Stage D D1 registry package tests."""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest
from jsonschema import Draft202012Validator

from vitnns.stage_d_registry import (
    AUDIT_ALGORITHM_ID,
    AUDIT_NAMESPACE,
    D0_COMPLETION_COMMIT,
    D0_COMPLETION_REF,
    D0_SCOPE_CANONICAL_DIGEST,
    PACKAGE_DIR,
    PACKAGE_FILES,
    PACKAGE_SCHEMA_NAME,
    SOURCE_FILE_SPECS,
    StageDRegistryError,
    build_stage_d_registry_package,
    canonical_file_bytes,
    canonical_sha256,
    load_and_validate_sources,
    strict_json_bytes,
    validate_stage_d_registry_package,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP = REPO_ROOT / "docs/阶段D-step-by-step.md"
FORBIDDEN_MODULE_PREFIXES = (
    "PIL", "cv2", "timm", "torch", "torchvision",
    "vitnns.attribution", "vitnns.dataset", "vitnns.extractor",
    "vitnns.modeling", "vitnns.runner", "vitnns.stage_c_runner",
)


def _load(name: str) -> dict:
    payload = (PACKAGE_DIR / name).read_bytes()
    document = strict_json_bytes(payload, name)
    assert isinstance(document, dict)
    return document


def _copy_package(destination: Path) -> Path:
    target = destination / "stage_d_fidelity_v1"
    target.mkdir(parents=True)
    for name in PACKAGE_FILES:
        shutil.copyfile(PACKAGE_DIR / name, target / name)
    return target


def test_d1_guard_blocks_image_model_outcome_and_nongit_subprocess() -> None:
    assert os.environ.get("VITNNS_D1_STRICT") == "1"
    assert os.environ.get("VITNNS_D1_GUARD_ACTIVE") == "1"
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    loaded = set(sys.modules)
    assert not sorted(
        name for name in loaded
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MODULE_PREFIXES)
    )
    with pytest.raises(ImportError, match="D1 strict guard blocked import"):
        importlib.import_module("torch")
    with pytest.raises(PermissionError, match="blocked image/model/binary path"):
        Path("d1_guard_probe.JPEG").stat()
    with pytest.raises(PermissionError, match="blocked Stage D outcome path"):
        Path("stage_d_outcomes/probe.json").stat()


def test_schema_is_closed_draft_2020_12() -> None:
    schema = _load(PACKAGE_SCHEMA_NAME)
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    for name in ("source_binding", "registry", "pre_authorization_access_attestation", "registry_metadata", "manifest"):
        definition = schema["$defs"][name]
        assert definition["type"] == "object"
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == set(definition["properties"])
    for definition in schema["$defs"].values():
        if isinstance(definition, dict) and definition.get("type") == "object":
            assert definition.get("additionalProperties") is False
            assert set(definition.get("required", ())) == set(definition.get("properties", ()))


def test_checked_in_package_validates_hash_chain_and_rebuilds_exactly() -> None:
    result = validate_stage_d_registry_package()
    assert result["status"] == "PASS"
    assert result["file_count"] == 6
    assert result["registry_rows"] == 500
    assert result["audit_overlay_count"] == 5
    assert result["reserve_image_bytes_accessed"] is False
    assert result["reserve_result_accessed"] is False
    assert result["model_or_gpu_used"] is False
    assert result["execution_authorization"] == "NOT_ISSUED"
    assert set(result["file_sha256"]) == set(PACKAGE_FILES)
    assert set(result["canonical_digests"]) == set(PACKAGE_FILES)


def test_source_bindings_are_exact_and_same_bytes_were_parsed() -> None:
    sources = load_and_validate_sources()
    source_binding = _load("source_binding.json")
    expected = {path: (sha, canonical, mode) for path, sha, canonical, mode, _ in SOURCE_FILE_SPECS}
    assert len(source_binding["source_files"]) == 11
    for record in source_binding["source_files"]:
        assert record["path"] in expected
        sha, canonical, mode = expected[record["path"]]
        assert record["file_sha256"] == sha == sources.bindings[record["path"]].sha256
        assert record["canonical_digest"] == canonical
        assert record["canonical_digest_mode"] == mode
        assert record["bytes"] == sources.bindings[record["path"]].size_bytes
        assert sources.bindings[record["path"]].mtime_ns > 0
        assert record["mtime_stability_checked_before_parse"] is True
        assert record["parsed_from_same_bound_bytes"] is True
        assert record["live_image_or_outcome_bytes_read"] is False
    assert source_binding["d0_scope"]["completion_ref"] == D0_COMPLETION_REF
    assert source_binding["d0_scope"]["completion_commit"] == D0_COMPLETION_COMMIT
    assert source_binding["d0_scope"]["machine_record_canonical_digest"] == D0_SCOPE_CANONICAL_DIGEST


def test_registry_rows_order_uniqueness_labels_disjointness_and_audit() -> None:
    registry = _load("registry.json")
    metadata = _load("registry_metadata.json")
    rows = registry["rows"]
    assert len(rows) == 500
    assert [row["selection_order"] for row in rows] == list(range(500))
    for field in ("sample_id", "relative_path", "image_sha256"):
        assert len({row[field] for row in rows}) == 500
    assert all(row["label_source"] == "official_devkit_val_gt" for row in rows)
    assert all(0 <= row["imagenet1k_index"] <= 999 for row in rows)
    assert all(1 <= row["official_ilsvrc2012_class_id"] <= 1000 for row in rows)
    assert metadata["counts"]["unique_official_wnids"] == 399
    assert metadata["disjointness"] == {
        "stage_c_registry_file_sha256": "9a6667bec6a27c715d272382a494f06924db881dca991dc4fac4751dbe7f37f0",
        "sample_id_overlap_count": 0,
        "relative_path_overlap_count": 0,
        "image_sha256_overlap_count": 0,
        "all_three_identity_dimensions_disjoint": True,
    }
    audit = registry["audit_selection"]
    assert audit["namespace"] == AUDIT_NAMESPACE
    assert audit["algorithm_id"] == AUDIT_ALGORITHM_ID
    assert audit["audit_image_count"] == 5
    assert audit["statistical_n_increment"] == 0
    assert len(set(audit["selected_sample_ids"])) == 5
    assert [entry["audit_rank"] for entry in audit["entries"]] == list(range(5))
    assert sum(row["audit_overlay"] for row in rows) == 5


def test_attestation_is_bounded_and_never_claims_absolute_non_access() -> None:
    attestation = _load("pre_authorization_access_attestation.json")
    facts = attestation["current_task_access_facts"]
    assert facts["reserve_metadata_contents_accessed"] is True
    for key, value in facts.items():
        if key != "reserve_metadata_contents_accessed":
            assert value is False, key
    boundary = attestation["epistemic_boundary"]
    assert boundary["absolute_historical_non_access_claim_made"] is False
    assert boundary["ledger_absence_is_scope_bounded"] is True
    assert boundary["schema_and_digests_fix_declared_content_not_independent_behavior_proof"] is True
    ledger = attestation["prior_access_ledger"]
    assert ledger["reserve_rows_checked"] == ledger["reserve_rows_marked_unseen"] == 500
    assert ledger["reserve_rows_with_prior_access_record"] == 0
    assert ledger["historical_directory_byte_audit_performed"] is False
    assert ledger["historical_directory_exhaustiveness_claimed"] is False
    serialized = canonical_file_bytes(attestation).decode("utf-8")
    assert "RESERVE_ABSOLUTELY_NEVER_VIEWED" not in serialized


def test_strict_parser_rejects_duplicate_nonfinite_and_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(StageDRegistryError, match="duplicate JSON key"):
        strict_json_bytes(b'{"a":1,"a":2}', "duplicate")
    with pytest.raises(StageDRegistryError, match="non-finite JSON number"):
        strict_json_bytes(b'{"a":NaN}', "nonfinite")
    target = _copy_package(tmp_path)
    document = json.loads((target / "registry_metadata.json").read_text())
    document["unknown"] = True
    (target / "registry_metadata.json").write_bytes(canonical_file_bytes(document))
    with pytest.raises(StageDRegistryError, match="schema validation failed"):
        validate_stage_d_registry_package(target, verify_sources=False, verify_rebuild=False)


def test_tamper_rehash_and_symlink_fail_closed(tmp_path: Path) -> None:
    target = _copy_package(tmp_path)
    registry = json.loads((target / "registry.json").read_text())
    registry["rows"][0]["official_wnid"] = "n00000000"
    registry["rows"][0]["canonical_digest"] = canonical_sha256(
        {key: value for key, value in registry["rows"][0].items() if key != "canonical_digest"}
    )
    registry["rows_canonical_digest"] = canonical_sha256({"rows": registry["rows"]})
    registry["canonical_digest"] = canonical_sha256(
        {key: value for key, value in registry.items() if key != "canonical_digest"}
    )
    (target / "registry.json").write_bytes(canonical_file_bytes(registry))
    with pytest.raises(StageDRegistryError, match="manifest binding drifted|deterministically rebuildable"):
        validate_stage_d_registry_package(target)

    second = _copy_package(tmp_path / "second")
    victim = second / "registry_metadata.json"
    backup = tmp_path / "registry_metadata.target"
    shutil.copyfile(victim, backup)
    victim.unlink()
    victim.symlink_to(backup)
    with pytest.raises(StageDRegistryError, match="cannot safely open metadata file|single-link regular file"):
        validate_stage_d_registry_package(second, verify_sources=False, verify_rebuild=False)


def test_rebuild_detects_source_metadata_drift_without_images(tmp_path: Path) -> None:
    schema_bytes = (PACKAGE_DIR / PACKAGE_SCHEMA_NAME).read_bytes()
    expected = build_stage_d_registry_package(schema_bytes=schema_bytes)
    assert expected == {name: (PACKAGE_DIR / name).read_bytes() for name in PACKAGE_FILES}
    # A fully self-resigned output still cannot replace the frozen candidate source.
    target = _copy_package(tmp_path)
    registry = json.loads((target / "registry.json").read_text())
    registry["rows"][0]["candidate_selection_sha256"] = "0" * 64
    registry["rows"][0]["canonical_digest"] = canonical_sha256(
        {key: value for key, value in registry["rows"][0].items() if key != "canonical_digest"}
    )
    registry["rows_canonical_digest"] = canonical_sha256({"rows": registry["rows"]})
    registry["canonical_digest"] = canonical_sha256(
        {key: value for key, value in registry.items() if key != "canonical_digest"}
    )
    (target / "registry.json").write_bytes(canonical_file_bytes(registry))
    with pytest.raises(StageDRegistryError):
        validate_stage_d_registry_package(target)


def test_static_route_keeps_d1_metadata_only_as_d2_hands_off_to_d3() -> None:
    roadmap = ROADMAP.read_text(encoding="utf-8")
    assert "当前 next task：`D3`" in roadmap
    assert "| D1 | PASS |" in roadmap
    assert "| D2 | PASS |" in roadmap
    assert "| D3 | NOT_STARTED |" in roadmap
    assert "reserve_image_bytes_accessed=false" in roadmap
    assert "reserve_result_accessed=false" in roadmap
    assert "model_or_gpu_used=false" in roadmap
    assert "execution_authorization=NOT_ISSUED" in roadmap
    assert "audit overlay `5/5`" in roadmap
    assert "audit overlay 恰好 5 个" in roadmap
