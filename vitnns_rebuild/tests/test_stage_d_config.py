"""Metadata/synthetic tests for the strict Stage D config loaders."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from vitnns.stage_d_config import (
    DEFAULT_STAGE_D_CONFIG,
    DEFAULT_STAGE_D_RUN_SPEC,
    DEFAULT_STAGE_D_RUN_SPEC_SCHEMA,
    DEFAULT_STAGE_D_SCHEMA,
    STAGE_D_CONFIG_CANONICAL_DIGEST,
    STAGE_D_CONFIG_FILE_SHA256,
    STAGE_D_CONFIG_PATH,
    STAGE_D_CONFIG_SCHEMA_FILE_SHA256,
    STAGE_D_CONFIG_SCHEMA_PATH,
    STAGE_D_RUN_SPEC_CANONICAL_DIGEST,
    STAGE_D_RUN_SPEC_FILE_SHA256,
    STAGE_D_RUN_SPEC_PATH,
    STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256,
    STAGE_D_RUN_SPEC_SCHEMA_PATH,
    StageDConfig,
    StageDConfigError,
    StageDContracts,
    StageDRunSpec,
    load_stage_d_config,
    load_stage_d_contracts,
    load_stage_d_run_spec,
    strict_json_bytes,
)


FORBIDDEN_MODULE_PREFIXES = (
    "PIL", "cv2", "timm", "torch", "torchvision",
    "vitnns.attribution", "vitnns.dataset", "vitnns.extractor",
    "vitnns.modeling", "vitnns.runner", "vitnns.stage_c_runner",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(document: Mapping[str, object]) -> str:
    semantic = {
        key: value for key, value in document.items() if key != "canonical_digest"
    }
    payload = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_file(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_contract_root(
    tmp_path: Path,
    *,
    config: Mapping[str, object] | None = None,
    config_schema: Mapping[str, object] | None = None,
    run_spec: Mapping[str, object] | None = None,
    run_spec_schema: Mapping[str, object] | None = None,
) -> Path:
    root = tmp_path / "repo"
    target = root / "config"
    target.mkdir(parents=True)
    values = (
        ("stage_d.json", config, STAGE_D_CONFIG_PATH),
        ("stage_d.schema.json", config_schema, STAGE_D_CONFIG_SCHEMA_PATH),
        ("stage_d_run_spec.json", run_spec, STAGE_D_RUN_SPEC_PATH),
        ("stage_d_run_spec.schema.json", run_spec_schema, STAGE_D_RUN_SPEC_SCHEMA_PATH),
    )
    for name, value, source in values:
        (target / name).write_bytes(
            source.read_bytes() if value is None else _canonical_file(value)
        )
    return root


def _resign(document: dict) -> dict:
    document["canonical_digest"] = _canonical_digest(document)
    return document


def _mutate(document: dict, path: tuple[object, ...], value: object) -> dict:
    changed = copy.deepcopy(document)
    cursor: object = changed
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    return changed


def test_default_paths_hashes_and_model_free_import_contract() -> None:
    assert os.environ.get("VITNNS_D3_STRICT") == "1"
    assert os.environ.get("VITNNS_D3_GUARD_ACTIVE") == "1"
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import vitnns.stage_d_config as module

    assert DEFAULT_STAGE_D_CONFIG == STAGE_D_CONFIG_PATH
    assert DEFAULT_STAGE_D_SCHEMA == STAGE_D_CONFIG_SCHEMA_PATH
    assert DEFAULT_STAGE_D_RUN_SPEC == STAGE_D_RUN_SPEC_PATH
    assert DEFAULT_STAGE_D_RUN_SPEC_SCHEMA == STAGE_D_RUN_SPEC_SCHEMA_PATH
    assert hashlib.sha256(STAGE_D_CONFIG_PATH.read_bytes()).hexdigest() == STAGE_D_CONFIG_FILE_SHA256
    assert hashlib.sha256(STAGE_D_CONFIG_SCHEMA_PATH.read_bytes()).hexdigest() == STAGE_D_CONFIG_SCHEMA_FILE_SHA256
    assert hashlib.sha256(STAGE_D_RUN_SPEC_PATH.read_bytes()).hexdigest() == STAGE_D_RUN_SPEC_FILE_SHA256
    assert hashlib.sha256(STAGE_D_RUN_SPEC_SCHEMA_PATH.read_bytes()).hexdigest() == STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256
    assert _load_json(STAGE_D_CONFIG_PATH)["canonical_digest"] == STAGE_D_CONFIG_CANONICAL_DIGEST
    assert _load_json(STAGE_D_RUN_SPEC_PATH)["canonical_digest"] == STAGE_D_RUN_SPEC_CANONICAL_DIGEST
    source = module.__loader__.get_source(module.__name__)
    assert source is not None
    assert all(f"import {name}" not in source for name in FORBIDDEN_MODULE_PREFIXES)


def test_frozen_contract_bundle_roundtrips_and_is_deeply_immutable() -> None:
    expected_config = _load_json(STAGE_D_CONFIG_PATH)
    expected_run_spec = _load_json(STAGE_D_RUN_SPEC_PATH)
    contracts = load_stage_d_contracts()

    assert isinstance(contracts, StageDContracts)
    assert isinstance(contracts.config, StageDConfig)
    assert isinstance(contracts.run_spec, StageDRunSpec)
    assert isinstance(contracts.config, Mapping)
    assert contracts["config"] is contracts.config
    assert contracts.config.to_dict() == expected_config
    assert contracts.run_spec.to_dict() == expected_run_spec
    assert contracts.to_dict() == {
        "config": expected_config, "run_spec": expected_run_spec,
    }
    assert contracts.config.canonical_digest == STAGE_D_CONFIG_CANONICAL_DIGEST
    assert contracts.config.file_sha256 == STAGE_D_CONFIG_FILE_SHA256
    assert contracts.config.schema_file_sha256 == STAGE_D_CONFIG_SCHEMA_FILE_SHA256
    assert contracts.run_spec.canonical_digest == STAGE_D_RUN_SPEC_CANONICAL_DIGEST
    assert contracts.run_spec.file_sha256 == STAGE_D_RUN_SPEC_FILE_SHA256
    assert contracts.run_spec.schema_file_sha256 == STAGE_D_RUN_SPEC_SCHEMA_FILE_SHA256
    assert contracts.config.authority["execution_authorization"] == "NOT_ISSUED"
    assert contracts.run_spec.selection["sample_ids"][0].endswith(".JPEG")

    with pytest.raises(TypeError):
        contracts.config.authority["execution_authorization"] = "AUTHORIZED"
    with pytest.raises(TypeError):
        contracts.run_spec.selection["sample_ids"][0] = "changed"
    with pytest.raises((AttributeError, TypeError)):
        contracts.config.file_sha256 = "0" * 64
    detached = contracts.to_dict()
    detached["config"]["authority"]["execution_authorization"] = "changed"
    assert contracts.config.authority["execution_authorization"] == "NOT_ISSUED"


@pytest.mark.parametrize(
    "payload,match",
    [
        (b'{"outer":{"x":1,"x":2}}', "duplicate JSON key: x"),
        (b'{"x":NaN}', "non-finite JSON number"),
        (b'{"x":Infinity}', "non-finite JSON number"),
        (b'{"x":-Infinity}', "non-finite JSON number"),
        (b'"\xff"', "must be UTF-8"),
        (b'{', "invalid JSON"),
    ],
)
def test_strict_json_bytes_rejects_non_json_extensions(payload: bytes, match: str) -> None:
    with pytest.raises(StageDConfigError, match=match):
        strict_json_bytes(payload, "synthetic input")


def test_loader_rejects_duplicate_nonfinite_and_non_utf8_contract_files(
    tmp_path: Path,
) -> None:
    cases = (
        (
            STAGE_D_CONFIG_PATH.read_text(encoding="utf-8").replace(
                '"schema_version": 1,',
                '"schema_version": 1, "schema_version": 1,', 1,
            ).encode("utf-8"),
            "duplicate JSON key",
        ),
        (
            STAGE_D_CONFIG_PATH.read_text(encoding="utf-8").replace(
                "0.000001", "NaN", 1,
            ).encode("utf-8"),
            "non-finite JSON number",
        ),
        (b"{\xff}", "must be UTF-8"),
    )
    for index, (payload, match) in enumerate(cases):
        root = tmp_path / str(index)
        (root / "config").mkdir(parents=True)
        (root / "config/stage_d.json").write_bytes(payload)
        (root / "config/stage_d.schema.json").write_bytes(
            STAGE_D_CONFIG_SCHEMA_PATH.read_bytes()
        )
        with pytest.raises(StageDConfigError, match=match):
            load_stage_d_config(
                "config/stage_d.json",
                schema_path="config/stage_d.schema.json",
                repo_root=root,
            )


def test_descriptor_safe_reader_rejects_symlink_hardlink_and_path_escape(
    tmp_path: Path,
) -> None:
    root = _write_contract_root(tmp_path)
    config = root / "config/stage_d.json"
    original = root / "config/stage_d.real.json"
    config.rename(original)
    config.symlink_to(original.name)
    with pytest.raises(StageDConfigError, match="safely open|symlink"):
        load_stage_d_config(
            "config/stage_d.json",
            schema_path="config/stage_d.schema.json",
            repo_root=root,
        )

    config.unlink()
    os.link(original, config)
    with pytest.raises(StageDConfigError, match="single-link"):
        load_stage_d_config(
            "config/stage_d.json",
            schema_path="config/stage_d.schema.json",
            repo_root=root,
        )

    with pytest.raises(StageDConfigError, match="unsafe path component"):
        load_stage_d_config(
            "../stage_d.json",
            schema_path="config/stage_d.schema.json",
            repo_root=root,
        )
    with pytest.raises(StageDConfigError, match="escapes repository root"):
        load_stage_d_config(
            Path("/etc/passwd"),
            schema_path="config/stage_d.schema.json",
            repo_root=root,
        )


def test_descriptor_reader_detects_change_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vitnns.stage_d_config as module

    root = _write_contract_root(tmp_path)
    real_fstat = module.os.fstat
    target_inode = (root / "config/stage_d.json").stat().st_ino
    target_calls = 0

    class ChangedStat:
        def __init__(self, value: os.stat_result) -> None:
            for name in (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid",
                "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns",
            ):
                setattr(self, name, getattr(value, name))
            self.st_mtime_ns += 1

    def changing_fstat(fd: int):
        nonlocal target_calls
        value = real_fstat(fd)
        if stat_is_regular(value.st_mode) and value.st_ino == target_inode:
            target_calls += 1
            if target_calls >= 2:
                return ChangedStat(value)
        return value

    def stat_is_regular(mode: int) -> bool:
        import stat
        return stat.S_ISREG(mode)

    monkeypatch.setattr(module.os, "fstat", changing_fstat)
    with pytest.raises(StageDConfigError, match="changed while being read"):
        load_stage_d_config(
            "config/stage_d.json",
            schema_path="config/stage_d.schema.json",
            repo_root=root,
        )


def test_schema_meta_validation_and_frozen_schema_hash_fail_closed(tmp_path: Path) -> None:
    invalid_meta = _load_json(STAGE_D_CONFIG_SCHEMA_PATH)
    invalid_meta["type"] = "not-a-json-schema-type"
    root = _write_contract_root(tmp_path / "meta", config_schema=invalid_meta)
    with pytest.raises(StageDConfigError, match="schema file SHA-256 drifted"):
        load_stage_d_config(
            "config/stage_d.json", schema_path="config/stage_d.schema.json",
            repo_root=root,
        )

    open_schema = _load_json(STAGE_D_RUN_SPEC_SCHEMA_PATH)
    open_schema["properties"]["audit"]["additionalProperties"] = True
    root = _write_contract_root(tmp_path / "open", run_spec_schema=open_schema)
    with pytest.raises(StageDConfigError, match="schema file SHA-256 drifted"):
        load_stage_d_run_spec(
            "config/stage_d_run_spec.json",
            schema_path="config/stage_d_run_spec.schema.json",
            repo_root=root,
        )

    harmless = _load_json(STAGE_D_CONFIG_SCHEMA_PATH)
    harmless["description"] += " tampered"
    root = _write_contract_root(tmp_path / "hash", config_schema=harmless)
    with pytest.raises(StageDConfigError, match="schema file SHA-256 drifted"):
        load_stage_d_config(
            "config/stage_d.json", schema_path="config/stage_d.schema.json",
            repo_root=root,
        )


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("schema_version",), True, "wrong JSON type|booleans"),
        (("cohort", "registry_row_count"), True, "wrong JSON type|booleans"),
        (("cohort", "registry_row_count"), 499, "frozen schema const"),
        (("authority", "execution_authorization"), "AUTHORIZED_STAGE_D_FIDELITY_SCOPE", "frozen schema const"),
        (("access_boundary", "reserve_image_bytes_accessed"), True, "frozen schema const"),
        (("secondary_and_spatial", "run_stability"), True, "frozen schema const"),
        (("forbidden_workload", "other_methods"), ["extra"], "length drifted|frozen schema const"),
        (("workload", "logical_work_items"), 211009, "frozen schema const"),
    ],
)
def test_config_rejects_type_authority_forbidden_and_count_drift(
    tmp_path: Path, path: tuple[object, ...], value: object, match: str,
) -> None:
    document = _resign(_mutate(_load_json(STAGE_D_CONFIG_PATH), path, value))
    root = _write_contract_root(tmp_path, config=document)
    with pytest.raises(StageDConfigError, match=match):
        load_stage_d_config(
            "config/stage_d.json", schema_path="config/stage_d.schema.json",
            repo_root=root,
        )


def test_config_unknown_missing_unsigned_and_self_resigned_drift_fail_closed(
    tmp_path: Path,
) -> None:
    unknown = _load_json(STAGE_D_CONFIG_PATH)
    unknown["silent_default"] = True
    root = _write_contract_root(tmp_path / "unknown", config=_resign(unknown))
    with pytest.raises(StageDConfigError, match="unknown fields|Additional properties"):
        load_stage_d_config(
            "config/stage_d.json", schema_path="config/stage_d.schema.json",
            repo_root=root,
        )

    missing = _load_json(STAGE_D_CONFIG_PATH)
    del missing["audit"]
    root = _write_contract_root(tmp_path / "missing", config=_resign(missing))
    with pytest.raises(StageDConfigError, match="missing|required"):
        load_stage_d_config(
            "config/stage_d.json", schema_path="config/stage_d.schema.json",
            repo_root=root,
        )

    unsigned = _mutate(
        _load_json(STAGE_D_CONFIG_PATH),
        ("authority", "execution_authorization"), "tampered",
    )
    root = _write_contract_root(tmp_path / "unsigned", config=unsigned)
    with pytest.raises(StageDConfigError):
        load_stage_d_config(
            "config/stage_d.json", schema_path="config/stage_d.schema.json",
            repo_root=root,
        )

    resigned = _mutate(
        _load_json(STAGE_D_CONFIG_PATH),
        ("canonical_digest",), "0" * 64,
    )
    root = _write_contract_root(tmp_path / "resigned", config=resigned)
    with pytest.raises(StageDConfigError, match="canonical self-digest mismatch"):
        load_stage_d_config(
            "config/stage_d.json", schema_path="config/stage_d.schema.json",
            repo_root=root,
        )


def test_valid_self_resigned_contract_still_fails_frozen_authority(
    tmp_path: Path,
) -> None:
    changed = _mutate(
        _load_json(STAGE_D_CONFIG_PATH),
        ("canonical_digest",), STAGE_D_RUN_SPEC_CANONICAL_DIGEST,
    )
    changed["authority"]["execution_authorization"] = "SELF_RESIGNED_TAMPER"
    changed["canonical_digest"] = _canonical_digest(changed)
    root = _write_contract_root(tmp_path, config=changed)
    with pytest.raises(StageDConfigError, match="frozen schema const"):
        load_stage_d_config(
            "config/stage_d.json", schema_path="config/stage_d.schema.json",
            repo_root=root,
        )


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("selection", "row_count"), True),
        (("execution_coordinates", "head_index"), 8),
        (("fidelity", "spatial_scope"), "PATCH_ONLY"),
        (("forbidden_workload", "run_stability"), True),
        (("forbidden_workload", "other_method_ids"), ["extra"]),
        (("workload", "logical_work_items"), 211009),
        (("sharding", "shard_count_status"), "FROZEN_AT_D2"),
    ],
)
def test_run_spec_rejects_bool_scope_forbidden_count_and_sharding_drift(
    tmp_path: Path, path: tuple[object, ...], value: object,
) -> None:
    document = _resign(_mutate(_load_json(STAGE_D_RUN_SPEC_PATH), path, value))
    root = _write_contract_root(tmp_path, run_spec=document)
    with pytest.raises(StageDConfigError):
        load_stage_d_run_spec(
            "config/stage_d_run_spec.json",
            schema_path="config/stage_d_run_spec.schema.json",
            repo_root=root,
        )


def test_cross_binding_validator_rejects_a_mismatched_validated_pair() -> None:
    import vitnns.stage_d_config as module

    config = load_stage_d_config()
    run_spec = load_stage_d_run_spec()
    replacement = run_spec.to_dict()
    replacement["config_binding"]["config_file_sha256"] = "0" * 64
    forged = StageDRunSpec(
        _data=replacement, source_path=run_spec.source_path,
        file_sha256=run_spec.file_sha256, size_bytes=run_spec.size_bytes,
        schema_path=run_spec.schema_path,
        schema_file_sha256=run_spec.schema_file_sha256,
        _issuer=module._DOCUMENT_ISSUER,
    )
    with pytest.raises(StageDConfigError, match="config file SHA-256"):
        module._validate_cross_bindings(config, forged)


def test_change_to_exact_bytes_fails_frozen_file_hash_even_if_json_is_equivalent(
    tmp_path: Path,
) -> None:
    root = _write_contract_root(tmp_path)
    path = root / "config/stage_d.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(StageDConfigError, match="file SHA-256 drifted"):
        load_stage_d_config(
            "config/stage_d.json", schema_path="config/stage_d.schema.json",
            repo_root=root,
        )
