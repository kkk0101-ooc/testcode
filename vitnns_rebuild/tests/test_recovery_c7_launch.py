from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from jsonschema import Draft202012Validator

from vitnns import recovery_c7_launch as launch
from vitnns.recovery_v2 import load_recovery_v2_binding


STAGE_A_ROOT = "evidence/stage_a/recovery_v2_d8c2f84"
STAGE_A_COMMIT = "d8c2f84e94b354bc07c23ecc1dcaee1f6d96bcd8"
STAGE_A_TREE = "4a59af4c9bd5500b09bfc6517ab8a64431e7b417"
PRODUCER_COMMIT = "a" * 40
PRODUCER_TREE = "b" * 40
SOURCE_TREE = "c" * 40
LAUNCH_COMMIT = "d" * 40


def _resign(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["canonical_digest"] = launch.launch_record_digest(result)
    return result


def _write_document(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(launch._json_file_bytes(document))


def _model_record() -> dict[str, Any]:
    return {
        "name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "identity": "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718",
        "state_dict_path": (
            "assets/models/"
            "vit_base_patch16_224_weights.semantic_candidate.pth"
        ),
        "state_dict_sha256": (
            "489bfa2f1aa3b6781e268a87eb629816"
            "b89eebba439420978a794b8e8151565d"
        ),
        "state_dict_bytes": 346_329_934,
        "lfs_oid_sha256": (
            "489bfa2f1aa3b6781e268a87eb629816"
            "b89eebba439420978a794b8e8151565d"
        ),
        "canonical_tensor_digest": (
            "150234d8117a50057f62aa14a97f46bb"
            "f1da92b00b386c10c71cc7000c5d51dd"
        ),
        "canonical_tensor_digest_algorithm": (
            "sha256_sorted_key_utf8_plus_torch_dtype_utf8_plus_"
            "python_shape_tuple_utf8_plus_cpu_contiguous_native_"
            "little_endian_numpy_c_order_bytes_no_delimiters"
        ),
        "expected_key_count": 152,
        "expected_parameter_count": 86_567_656,
        "historical_artifact_recovered": False,
        "historical_state_dict_sha256": (
            "2e718037429f18ec77aa629471beb2e999"
            "93d7383375188bab777aba04d5e4d0"
        ),
    }


def _binding_record() -> dict[str, Any]:
    return {
        "binding_id": "RECOVERY_V2_STAGE_A_20260901",
        "status": "BOUND_FOR_STAGE_A_AND_C7_PREPARATION_ONLY",
        "path": "config/recovery_v2_binding.json",
        "file_sha256": (
            "ce334e40969a7b9e06ecbe3cfea64c6"
            "b684bdcb02e7b7eea077ecd2af94485d6"
        ),
        "schema_path": "config/recovery_v2_binding.schema.json",
        "schema_file_sha256": (
            "48381d6e8a9c7bfb844e3d4e544c8b58"
            "02dbf3abbb984e32561d8ef8805a3b80"
        ),
        "canonical_digest": (
            "e49786a0648df60081419e997c0e5e2e"
            "c2fd61e38a30e979532852bb6e6c853e"
        ),
    }


def _stage_a_record() -> dict[str, Any]:
    return {
        "root": STAGE_A_ROOT,
        "receipt_path": f"{STAGE_A_ROOT}/stage_a_verification.json",
        "receipt_file_sha256": (
            "fc0a6e673be2e6292444074ee0c5daf2"
            "d8c15d08ca87b259cd8f357dbb903de5"
        ),
        "receipt_canonical_digest": (
            "178c6e9332a674f3536be3b4cf1888d0"
            "d2d5193727ab9016c9fbc7599719cbaa"
        ),
        "status": "PASS",
        "producer_commit": STAGE_A_COMMIT,
        "producer_tree": STAGE_A_TREE,
        "artifacts": {
            "extraction.pt": {
                "path": f"{STAGE_A_ROOT}/extraction.pt",
                "bytes": 58_690_363,
                "file_sha256": (
                    "8a32fe50ec2174e9162e2aead432e665"
                    "3ac44e47c0080ff02628337535c10eac"
                ),
            },
            "provenance.json": {
                "path": f"{STAGE_A_ROOT}/provenance.json",
                "bytes": 9_073,
                "file_sha256": (
                    "a76fb2f388199232a7a2890f719f86c6"
                    "aa681342d525873aaf2e17058d68dd9d"
                ),
            },
            "validation.json": {
                "path": f"{STAGE_A_ROOT}/validation.json",
                "bytes": 1_026,
                "file_sha256": (
                    "7cf4dad6a20a3aeb16a019fec5417f39"
                    "7a26c00f1d2ef5ef8a99fb2aa1e12fa1"
                ),
            },
        },
    }


def _contract_records() -> dict[str, Any]:
    return {
        "stage_c_config": {
            "path": "config/stage_c.json",
            "file_sha256": (
                "0ee57c32b0fcb3116f91fddce5d918c8"
                "eb1ae05606c3616b6a2ec9afa9f0e669"
            ),
        },
        "protocol": {
            "path": "docs/阶段C_protocol.md",
            "file_sha256": (
                "ff958a9455ae15307827a9e40fc6d7d0"
                "3c9baa10bf1fb8b83650a10bd85ff7b8"
            ),
        },
        "acceptance": {
            "path": "docs/stage_c_acceptance.md",
            "file_sha256": (
                "f4629105d411b32d4f7700caa22d7d6b"
                "6c1bf4b8e01f37e4e2ac6ac4851dfafd"
            ),
        },
        "diagnostic_run_spec": {
            "path": "config/stage_c_c7_diagnostic_run_spec.json",
            "file_sha256": (
                "7966d846ceec1cb372cc1c177e412f87"
                "dc67e097e61f5f3f81a1822091b3f7ec"
            ),
            "canonical_sha256": (
                "5cdaa0ba34b303c5d4ae1071613ff440"
                "c2b084fa239a4ddf4168fd6e7898dfec"
            ),
        },
        "formal_run_spec": {
            "path": "config/stage_c_c7_run_spec.json",
            "file_sha256": (
                "76f7e1ffd6e15b8e79ed856eba90bc93"
                "0622399681bf32df8dc564abbe323404"
            ),
            "canonical_sha256": (
                "3078b09f8e956c68a430836443a982eb"
                "2894e724426f4e0b798b94c3b817e5e2"
            ),
        },
    }


def _valid_document(schema_sha256: str) -> dict[str, Any]:
    common = "evidence/stage_c/recovery_c7_unit"
    document: dict[str, Any] = {
        "$schema": "recovery_c7_launch_record.schema.json",
        "schema_version": 1,
        "document_type": launch.DOCUMENT_TYPE,
        "record_id": launch.RECORD_ID,
        "status": launch.STATUS,
        "launch_record_commit": "SELF",
        "launch_schema": {
            "path": "config/recovery_c7_launch_record.schema.json",
            "file_sha256": schema_sha256,
        },
        "amendment": {
            "id": "RECOVERY-C7-RERUN-20260901",
            "path": "docs/recovery_protocol_amendment_20260901.md",
            "commit": "4a87e4d9cb71fe9c394bb2fdd5d4e71bbfe8aaed",
            "file_sha256": (
                "07e00c12e689af9180d03338738394374"
                "c7c1afbd15a52a9493a23b689b4787a"
            ),
        },
        "recovery_binding": _binding_record(),
        "stage_a": _stage_a_record(),
        "model": _model_record(),
        "dependencies": {
            "requirements_lock_path": "requirements.lock.txt",
            "requirements_lock_sha256": (
                "81beabd6aba9ede777e973e2ee5f0c60"
                "ce2aa88ce55b29d025e95d67ab175984"
            ),
            "requirements_lock_bytes": 1_788,
        },
        "c7_contracts": _contract_records(),
        "producer_git": {
            "commit": PRODUCER_COMMIT,
            "tree": PRODUCER_TREE,
            "source_subtree_path": launch.SOURCE_SUBTREE_PATH,
            "source_subtree_tree": SOURCE_TREE,
            "dirty": False,
        },
        "outputs": {
            "common_root": common,
            "launch_record_path": (
                f"{common}/{launch.LAUNCH_RECORD_FILENAME}"
            ),
            "diagnostic_root": f"{common}/diagnostic",
            "formal_root": f"{common}/formal",
            "reports_root": f"{common}/reports",
        },
        "scope": {
            "execution_authorization": (
                "RECOVERY_C7_DIAGNOSTIC_AND_FORMAL_ONLY"
            ),
            "rerun_interpretation": (
                "recovery_replication_not_new_confirmatory"
            ),
            "c7_diagnostic_authorized": True,
            "c7_formal_authorized": True,
            "stage_c_holdout_access_authorized": False,
            "c8_authorized": False,
            "c9_authorized": False,
            "c10_authorized": False,
            "stage_d_d7_authorized": False,
            "stage_d_d8_authorized": False,
            "stage_d_reserve_access_authorized": False,
        },
    }
    return _resign(document)


def _fake_binding(repo: Path, document: dict[str, Any]) -> SimpleNamespace:
    binding_record = document["recovery_binding"]
    model = copy.deepcopy(document["model"])
    return SimpleNamespace(
        path=repo / binding_record["path"],
        schema_path=repo / binding_record["schema_path"],
        document={
            "binding_id": binding_record["binding_id"],
            "status": binding_record["status"],
            "amendment": {
                "id": document["amendment"]["id"],
                "path": document["amendment"]["path"],
                "commit": document["amendment"]["commit"],
                "sha256": document["amendment"]["file_sha256"],
            },
            "environment": {
                "requirements_lock_path": "requirements.lock.txt",
                "requirements_lock_sha256": document["dependencies"][
                    "requirements_lock_sha256"
                ],
            },
            "c7_contracts": {},
        },
        model=model,
        model_path=repo / model["state_dict_path"],
        data_root=repo / "assets/data/ILSVRC2012_val",
        devkit_path=repo / "assets/imagenet/devkit.tar.gz",
        canonical_digest=binding_record["canonical_digest"],
    )


@pytest.fixture
def launch_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    repo = tmp_path / "vitnns_rebuild"
    schema = repo / "config/recovery_c7_launch_record.schema.json"
    schema.parent.mkdir(parents=True)
    schema.write_bytes(launch.DEFAULT_SCHEMA_PATH.read_bytes())
    document = _valid_document(launch.sha256_file(schema))
    path = repo / document["outputs"]["launch_record_path"]
    _write_document(path, document)
    receipt = repo / document["stage_a"]["receipt_path"]
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    binding = _fake_binding(repo, document)

    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(
        launch, "load_recovery_v2_binding", lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        launch, "verify_recovery_v2_lfs_identity",
        lambda *_args, **_kwargs: {
            "oid_sha256": document["model"]["lfs_oid_sha256"],
            "hydrated": True,
        },
    )
    monkeypatch.setattr(
        launch, "_binding_record",
        lambda *_args, **_kwargs: copy.deepcopy(_binding_record()),
    )
    monkeypatch.setattr(
        launch, "_model_record",
        lambda *_args, **_kwargs: copy.deepcopy(_model_record()),
    )
    monkeypatch.setattr(
        launch, "_dependency_record",
        lambda *_args, **_kwargs: copy.deepcopy(document["dependencies"]),
    )
    monkeypatch.setattr(
        launch, "_contract_records",
        lambda *_args, **_kwargs: copy.deepcopy(_contract_records()),
    )

    def validate_stage_a(
        _repo: Path, _top: Path, _receipt: Path, _binding: Any,
        expected: Any = None,
    ) -> dict[str, Any]:
        actual = _stage_a_record()
        if expected is not None and launch.canonical_json_bytes(expected) != (
            launch.canonical_json_bytes(actual)
        ):
            raise launch.RecoveryC7LaunchError(
                "Stage A launch binding drifted"
            )
        return actual

    def validate_git(
        _repo: Path, _top: Path, _launch: Path, candidate: Any,
        _critical: Any, *, require_tracked: bool,
    ) -> str:
        if candidate["producer_git"] != document["producer_git"]:
            raise launch.RecoveryC7LaunchError("Producer Git identity drifted")
        return LAUNCH_COMMIT if require_tracked else "SELF"

    monkeypatch.setattr(launch, "_validate_stage_a_receipt", validate_stage_a)
    monkeypatch.setattr(launch, "_validate_git_authority", validate_git)

    def write(candidate: dict[str, Any]) -> None:
        _write_document(path, candidate)

    return SimpleNamespace(
        repo=repo, schema=schema, path=path, document=document,
        binding=binding, receipt=receipt, write=write,
    )


def _load(case: SimpleNamespace, **kwargs: Any) -> launch.RecoveryC7LaunchRecord:
    return launch.load_recovery_c7_launch_record(
        case.path, repo_root=case.repo, schema_path=case.schema,
        require_fresh_roots=kwargs.pop("require_fresh_roots", True),
        require_tracked=kwargs.pop("require_tracked", False), **kwargs,
    )


def test_schema_is_draft_2020_12_and_recursively_closed() -> None:
    schema = json.loads(launch.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            properties = node.get("properties", {})
            assert set(node.get("required", ())) == set(properties)
        for child in node.values():
            if isinstance(child, dict):
                visit(child)
            elif isinstance(child, list):
                for item in child:
                    visit(item)

    visit(schema)


def test_digest_helpers_are_order_stable_and_exclude_only_self_digest() -> None:
    left = {"b": [2, 3], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2, 3]}
    assert launch.canonical_sha256(left) == launch.canonical_sha256(right)

    document = {**left, "canonical_digest": "0" * 64}
    assert launch.launch_record_digest(document) == launch.canonical_sha256(left)
    document["canonical_digest"] = "f" * 64
    assert launch.launch_record_digest(document) == launch.canonical_sha256(left)


def test_load_and_validate_return_frozen_metadata_only_record(launch_case) -> None:
    record = _load(launch_case)
    assert isinstance(record, launch.RecoveryC7LaunchRecord)
    assert record.path == launch_case.path
    assert record.schema_path == launch_case.schema
    assert record.canonical_digest == launch_case.document["canonical_digest"]
    assert record.launch_commit == "SELF"
    assert record.model_path == launch_case.binding.model_path
    assert record.diagnostic_root == (
        launch_case.repo / launch_case.document["outputs"]["diagnostic_root"]
    )
    with pytest.raises(TypeError):
        record.document["scope"]["c8_authorized"] = True
    with pytest.raises(TypeError):
        record.model_identity["identity"] = "forged"

    validated = launch.validate_recovery_c7_launch_record(
        launch_case.path, repo_root=launch_case.repo,
        schema_path=launch_case.schema, require_fresh_roots=True,
        require_tracked=False,
    )
    assert validated.canonical_digest == record.canonical_digest


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"schema_version":1,"schema_version":1}', "Duplicate JSON key"),
        (b'{"value":NaN}', "Non-finite JSON value"),
        (b'{"value":Infinity}', "Non-finite JSON value"),
    ],
)
def test_loader_rejects_duplicate_keys_and_nonfinite_json(
    launch_case, payload: bytes, message: str,
) -> None:
    launch_case.path.write_bytes(payload)
    with pytest.raises(launch.RecoveryC7LaunchError, match=message):
        _load(launch_case)


@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
def test_loader_rejects_unknown_schema_fields(launch_case, nested: bool) -> None:
    document = copy.deepcopy(launch_case.document)
    target = document["outputs"] if nested else document
    target["unexpected"] = True
    launch_case.write(_resign(document))
    with pytest.raises(launch.RecoveryC7LaunchError, match="schema failure"):
        _load(launch_case)


def test_loader_rejects_stale_canonical_digest(launch_case) -> None:
    document = copy.deepcopy(launch_case.document)
    document["stage_a"]["receipt_file_sha256"] = "f" * 64
    launch_case.write(document)
    with pytest.raises(launch.RecoveryC7LaunchError, match="canonical digest"):
        _load(launch_case)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["stage_a"].__setitem__(
                "receipt_file_sha256", "f" * 64
            ),
            "Stage A launch binding drifted",
        ),
        (
            lambda value: value["c7_contracts"]["formal_run_spec"].__setitem__(
                "canonical_sha256", "f" * 64
            ),
            "C7 contracts drifted",
        ),
        (
            lambda value: value["producer_git"].__setitem__(
                "source_subtree_tree", "f" * 40
            ),
            "Producer Git identity drifted",
        ),
    ],
    ids=["stage-a", "run-spec", "producer"],
)
def test_self_resigned_semantic_tamper_is_rejected(
    launch_case, mutate: Callable[[dict[str, Any]], None], message: str,
) -> None:
    document = copy.deepcopy(launch_case.document)
    mutate(document)
    launch_case.write(_resign(document))
    with pytest.raises(launch.RecoveryC7LaunchError, match=message):
        _load(launch_case)


@pytest.mark.parametrize("field", ["state_dict_sha256", "canonical_tensor_digest"])
def test_loader_rejects_live_model_identity_drift(
    launch_case, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    live = _model_record()
    live[field] = "f" * 64
    monkeypatch.setattr(launch, "_model_record", lambda _binding: live)
    with pytest.raises(launch.RecoveryC7LaunchError, match="model launch identity"):
        _load(launch_case)


@pytest.mark.parametrize(
    ("contract", "field"),
    [
        ("stage_c_config", "file_sha256"),
        ("diagnostic_run_spec", "file_sha256"),
        ("formal_run_spec", "canonical_sha256"),
    ],
)
def test_loader_rejects_live_contract_and_run_spec_drift(
    launch_case, monkeypatch: pytest.MonkeyPatch, contract: str, field: str,
) -> None:
    live = _contract_records()
    live[contract][field] = "f" * 64
    monkeypatch.setattr(launch, "_contract_records", lambda *_args: live)
    with pytest.raises(launch.RecoveryC7LaunchError, match="C7 contracts drifted"):
        _load(launch_case)


def test_generate_builds_exact_schema_valid_record_without_creating_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    schema = repo / "config/recovery_c7_launch_record.schema.json"
    schema.parent.mkdir(parents=True)
    schema.write_bytes(launch.DEFAULT_SCHEMA_PATH.read_bytes())
    template = _valid_document(launch.sha256_file(schema))
    binding_path = repo / template["recovery_binding"]["path"]
    binding_path.write_text("{}\n", encoding="utf-8")
    receipt = repo / template["stage_a"]["receipt_path"]
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    binding = _fake_binding(repo, template)

    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(
        launch, "load_recovery_v2_binding", lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        launch, "verify_recovery_v2_lfs_identity",
        lambda *_args, **_kwargs: {
            "oid_sha256": template["model"]["lfs_oid_sha256"],
            "hydrated": True,
        },
    )
    monkeypatch.setattr(launch, "_binding_record", lambda *_: _binding_record())
    monkeypatch.setattr(launch, "_model_record", lambda *_: _model_record())
    monkeypatch.setattr(
        launch, "_dependency_record", lambda *_: template["dependencies"],
    )
    monkeypatch.setattr(launch, "_contract_records", lambda *_: _contract_records())
    monkeypatch.setattr(
        launch, "_validate_stage_a_receipt", lambda *_: _stage_a_record(),
    )
    monkeypatch.setattr(
        launch, "_capture_clean_producer",
        lambda *_: copy.deepcopy(template["producer_git"]),
    )
    monkeypatch.setattr(launch, "_is_ancestor", lambda *_: None)
    monkeypatch.setattr(
        launch, "_run_git",
        lambda _repo, *args: f"{args[-1]}: filter: lfs",
    )

    common = repo / template["outputs"]["common_root"]
    document = launch.generate_recovery_c7_launch_record(
        repo_root=repo,
        launch_record_path=repo / template["outputs"]["launch_record_path"],
        diagnostic_root=repo / template["outputs"]["diagnostic_root"],
        formal_root=repo / template["outputs"]["formal_root"],
        reports_root=repo / template["outputs"]["reports_root"],
        binding_path=binding_path, stage_a_receipt_path=receipt,
        schema_path=schema,
    )

    Draft202012Validator(json.loads(schema.read_text(encoding="utf-8"))).validate(
        document
    )
    assert document["canonical_digest"] == launch.launch_record_digest(document)
    assert document["launch_record_commit"] == "SELF"
    assert document["model"] == _model_record()
    assert document["stage_a"]["producer_commit"] == STAGE_A_COMMIT
    assert document["producer_git"]["commit"] == PRODUCER_COMMIT
    assert document["scope"]["c7_formal_authorized"] is True
    assert document["scope"]["c8_authorized"] is False
    assert not common.exists()


def test_write_is_exclusive_and_preserves_existing_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    repo.mkdir()
    document = _valid_document("1" * 64)
    path = repo / document["outputs"]["launch_record_path"]
    binding = _fake_binding(repo, document)
    monkeypatch.setattr(
        launch, "generate_recovery_c7_launch_record",
        lambda **_kwargs: copy.deepcopy(document),
    )
    monkeypatch.setattr(
        launch, "load_recovery_v2_binding", lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(
        launch, "_capture_clean_producer",
        lambda *_args: copy.deepcopy(document["producer_git"]),
    )

    record = launch.write_recovery_c7_launch_record(
        repo_root=repo, launch_record_path=path,
        diagnostic_root=repo / document["outputs"]["diagnostic_root"],
        formal_root=repo / document["outputs"]["formal_root"],
        reports_root=repo / document["outputs"]["reports_root"],
    )
    original = path.read_bytes()
    assert record.path == path
    assert record.file_sha256 == hashlib.sha256(original).hexdigest()

    with pytest.raises(launch.RecoveryC7LaunchError, match="[Ff]resh|publish"):
        launch.write_recovery_c7_launch_record(
            repo_root=repo, launch_record_path=path,
            diagnostic_root=repo / document["outputs"]["diagnostic_root"],
            formal_root=repo / document["outputs"]["formal_root"],
            reports_root=repo / document["outputs"]["reports_root"],
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("launch_path", "diagnostic", "formal", "reports", "message"),
    [
        ("../outside/recovery_c7_launch_record.json", "d", "f", "r", "normalized"),
        ("outputs/run/recovery_c7_launch_record.json", "d", "f", "r", "evidence/stage_c"),
        (
            "evidence/stage_c/run/recovery_c7_launch_record.json",
            "evidence/stage_c/run/nested/diagnostic",
            "evidence/stage_c/run/formal",
            "evidence/stage_c/run/reports",
            "direct child",
        ),
        (
            "evidence/stage_c/run/recovery_c7_launch_record.json",
            "evidence/stage_c/run/same",
            "evidence/stage_c/run/same",
            "evidence/stage_c/run/reports",
            "mutually distinct",
        ),
    ],
)
def test_output_path_policy_rejects_unsafe_or_overlapping_roots(
    tmp_path: Path, launch_path: str, diagnostic: str, formal: str,
    reports: str, message: str,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    repo.mkdir()
    if diagnostic in {"d", "f", "r"}:
        common = Path(launch_path).parent
        diagnostic = (common / diagnostic).as_posix()
        formal = (common / formal).as_posix()
        reports = (common / reports).as_posix()
    with pytest.raises(launch.RecoveryC7LaunchError, match=message):
        launch._validate_output_paths(
            repo, launch_path, diagnostic, formal, reports,
            require_fresh_roots=True, launch_must_exist=False,
        )


def test_output_root_freshness_can_be_relaxed_only_for_existing_directories(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    common = repo / "evidence/stage_c/run"
    common.mkdir(parents=True)
    record = common / launch.LAUNCH_RECORD_FILENAME
    record.write_text("{}\n", encoding="utf-8")
    diagnostic, formal, reports = (common / name for name in (
        "diagnostic", "formal", "reports"
    ))

    launch._validate_output_paths(
        repo, record, diagnostic, formal, reports,
        require_fresh_roots=True, launch_must_exist=True,
    )
    diagnostic.mkdir()
    with pytest.raises(launch.RecoveryC7LaunchError, match="not fresh"):
        launch._validate_output_paths(
            repo, record, diagnostic, formal, reports,
            require_fresh_roots=True, launch_must_exist=True,
        )
    checked = launch._validate_output_paths(
        repo, record, diagnostic, formal, reports,
        require_fresh_roots=False, launch_must_exist=True,
    )
    assert checked[2] == diagnostic

    formal.write_text("not a directory", encoding="utf-8")
    with pytest.raises(launch.RecoveryC7LaunchError, match="must be a directory"):
        launch._validate_output_paths(
            repo, record, diagnostic, formal, reports,
            require_fresh_roots=False, launch_must_exist=True,
        )


def test_loader_freshness_flag_distinguishes_prelaunch_and_later_phases(
    launch_case,
) -> None:
    diagnostic = launch_case.repo / launch_case.document["outputs"][
        "diagnostic_root"
    ]
    diagnostic.mkdir()
    with pytest.raises(launch.RecoveryC7LaunchError, match="not fresh"):
        _load(launch_case, require_fresh_roots=True)
    assert _load(launch_case, require_fresh_roots=False).diagnostic_root == diagnostic


def _phase_record(repo: Path) -> launch.RecoveryC7LaunchRecord:
    common = repo / "evidence/stage_c/run"
    common.mkdir(parents=True)
    path = common / launch.LAUNCH_RECORD_FILENAME
    path.write_text("{}\n", encoding="utf-8")
    return launch.RecoveryC7LaunchRecord(
        path=path, schema_path=path, document={"canonical_digest": "0" * 64},
        binding=SimpleNamespace(), model_path=repo / "model.pth",
        data_root=repo / "data", devkit_path=repo / "devkit.tar.gz",
        diagnostic_root=common / "diagnostic", formal_root=common / "formal",
        reports_root=common / "reports", model_identity={},
        file_sha256=launch.sha256_file(path), launch_commit=LAUNCH_COMMIT,
        repo_root=repo,
    )


@pytest.mark.parametrize(
    ("phase", "relative_paths"),
    [
        ("pre_diagnostic", []),
        ("diagnostic_gate", ["diagnostic", "diagnostic/unit/status.json"]),
        ("formal", ["diagnostic/gate.json", "reports/gate.json"]),
        (
            "report",
            ["diagnostic/gate.json", "formal/unit.json", "reports/report.json"],
        ),
    ],
)
def test_phase_scoped_dirty_roots_are_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
    relative_paths: list[str],
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    repo.mkdir()
    record = _phase_record(repo)
    common = "evidence/stage_c/run"
    entries = [{"status": "??", "path": f"{common}/{path}"}
               for path in relative_paths]
    verified: list[tuple[Path, str | None]] = []
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(
        launch, "_run_git", lambda _repo, *args: LAUNCH_COMMIT,
    )
    monkeypatch.setattr(launch, "_git_status_entries", lambda _repo: entries)
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head",
        lambda _repo, _top, path, *, expected_sha256=None: (
            verified.append((path, expected_sha256)) or ("0" * 64, 3)
        ),
    )

    assert launch.capture_authorized_c7_source_state(record, phase) == {
        "commit": LAUNCH_COMMIT, "dirty": False,
    }
    assert verified == [(record.path, record.file_sha256)]


@pytest.mark.parametrize(
    ("phase", "dirty_path"),
    [
        ("pre_diagnostic", "evidence/stage_c/run/diagnostic/unit.json"),
        ("diagnostic_gate", "evidence/stage_c/run/formal/unit.json"),
        ("formal", "evidence/stage_c/run/formal/unit.json"),
        ("report", "src/vitnns/stage_c_runner.py"),
    ],
)
def test_phase_scoped_dirty_roots_reject_out_of_phase_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, dirty_path: str,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    repo.mkdir()
    record = _phase_record(repo)
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(launch, "_run_git", lambda _repo, *args: LAUNCH_COMMIT)
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head", lambda *_args, **_kwargs: ("0" * 64, 3),
    )
    monkeypatch.setattr(
        launch, "_git_status_entries",
        lambda _repo: [{"status": "??", "path": dirty_path}],
    )
    with pytest.raises(launch.RecoveryC7LaunchError, match="outside roots"):
        launch.capture_authorized_c7_source_state(record, phase)


def test_phase_capture_rejects_unknown_phase_dirty_launch_and_current_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    repo.mkdir()
    record = _phase_record(repo)
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head", lambda *_args, **_kwargs: ("0" * 64, 3),
    )
    with pytest.raises(launch.RecoveryC7LaunchError, match="Unknown recovery C7 phase"):
        launch.capture_authorized_c7_source_state(record, "c8")

    monkeypatch.setattr(launch, "_run_git", lambda _repo, *args: "f" * 40)
    with pytest.raises(launch.RecoveryC7LaunchError, match="current HEAD"):
        launch.capture_authorized_c7_source_state(record, "report")

    monkeypatch.setattr(launch, "_run_git", lambda _repo, *args: LAUNCH_COMMIT)
    launch_top = record.path.relative_to(repo).as_posix()
    monkeypatch.setattr(
        launch, "_git_status_entries",
        lambda _repo: [{"status": " M", "path": launch_top}],
    )
    with pytest.raises(launch.RecoveryC7LaunchError, match="itself is dirty"):
        launch.capture_authorized_c7_source_state(record, "report")


@pytest.mark.parametrize("changed", ["head", "status"])
def test_phase_capture_rejects_git_state_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    repo.mkdir()
    record = _phase_record(repo)
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    heads = iter(
        [LAUNCH_COMMIT, "f" * 40]
        if changed == "head" else [LAUNCH_COMMIT, LAUNCH_COMMIT]
    )
    monkeypatch.setattr(launch, "_run_git", lambda _repo, *args: next(heads))
    clean: list[dict[str, str]] = []
    statuses = iter(
        [clean, clean]
        if changed == "head" else [clean, [{"status": "??", "path": "x"}]]
    )
    monkeypatch.setattr(launch, "_git_status_entries", lambda _repo: next(statuses))
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head",
        lambda *_args, **_kwargs: ("0" * 64, 3),
    )
    with pytest.raises(launch.RecoveryC7LaunchError, match="changed during"):
        launch.capture_authorized_c7_source_state(record, "pre_diagnostic")


@pytest.mark.parametrize("status", [" M", "D ", "A "])
def test_phase_capture_rejects_tracked_mutation_inside_allowed_evidence_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    repo.mkdir()
    record = _phase_record(repo)
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(launch, "_run_git", lambda _repo, *args: LAUNCH_COMMIT)
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head", lambda *_args, **_kwargs: ("0" * 64, 3),
    )
    monkeypatch.setattr(
        launch, "_git_status_entries",
        lambda _repo: [{
            "status": status,
            "path": "evidence/stage_c/run/diagnostic/previous.json",
        }],
    )
    with pytest.raises(launch.RecoveryC7LaunchError, match="newly created"):
        launch.capture_authorized_c7_source_state(record, "report")


def test_tracked_lfs_pointer_binds_head_oid_size_and_live_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    path = repo / "evidence/item.json"
    path.parent.mkdir(parents=True)
    payload = b"tiny hydrated payload"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{digest}\nsize {len(payload)}\n"
    ).encode("ascii")
    tracked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        launch, "_require_tracked",
        lambda _repo, top_path, revision="HEAD": tracked.append(
            (top_path, revision)
        ),
    )
    monkeypatch.setattr(
        launch, "_git_process",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=pointer, stderr=b""
        ),
    )
    monkeypatch.setattr(launch, "_run_git", lambda *_args: "")

    assert launch._verify_lfs_pointer_at_head(
        repo, repo, path, expected_sha256=digest
    ) == (digest, len(payload))
    assert tracked == [("evidence/item.json", "HEAD")]


@pytest.mark.parametrize(
    ("pointer_factory", "dirty", "message"),
    [
        (lambda digest, size: b"ordinary Git blob", "", "strict Git LFS pointer"),
        (
            lambda digest, size: (
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{'f' * 64}\nsize {size}\n"
            ).encode("ascii"),
            "",
            "content drifted",
        ),
        (
            lambda digest, size: (
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{digest}\nsize {size}\n"
            ).encode("ascii"),
            " M evidence/item.json",
            "is dirty",
        ),
    ],
    ids=["not-pointer", "oid-drift", "dirty-live-file"],
)
def test_tracked_lfs_pointer_rejects_invalid_or_dirty_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    pointer_factory: Callable[[str, int], bytes], dirty: str, message: str,
) -> None:
    repo = tmp_path / "repo"
    path = repo / "evidence/item.json"
    path.parent.mkdir(parents=True)
    payload = b"tiny payload"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    pointer = pointer_factory(digest, len(payload))
    monkeypatch.setattr(launch, "_require_tracked", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        launch, "_git_process",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=pointer, stderr=b""
        ),
    )
    monkeypatch.setattr(launch, "_run_git", lambda *_args: dirty)
    with pytest.raises(launch.RecoveryC7LaunchError, match=message):
        launch._verify_lfs_pointer_at_head(
            repo, repo, path, expected_sha256=digest
        )


def _install_git_authority_stubs(
    monkeypatch: pytest.MonkeyPatch, repo: Path, record_path: Path,
    *, head: str = LAUNCH_COMMIT, head_source: str = SOURCE_TREE,
    launch_parent: str = PRODUCER_COMMIT, changed: list[str] | None = None,
    launch_blob: str = "1" * 40, head_launch_blob: str | None = None,
) -> None:
    launch_top = record_path.relative_to(repo).as_posix()

    def run_git(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--verify", f"{PRODUCER_COMMIT}^{{tree}}"):
            return PRODUCER_TREE
        if arguments == ("rev-parse", f"{PRODUCER_COMMIT}:src/vitnns"):
            return SOURCE_TREE
        if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
            return head
        if arguments == ("rev-parse", "HEAD:src/vitnns"):
            return head_source
        if arguments == ("rev-parse", f"{LAUNCH_COMMIT}:{launch_top}"):
            return launch_blob
        if arguments == ("rev-parse", f"HEAD:{launch_top}"):
            return head_launch_blob or launch_blob
        if arguments == (
            "log", "-n", "1", "--format=%H", "HEAD", "--",
            f":(top){launch_top}",
        ):
            return LAUNCH_COMMIT
        if arguments == ("rev-list", "--parents", "-n", "1", LAUNCH_COMMIT):
            return f"{LAUNCH_COMMIT} {launch_parent}"
        if arguments == (
            "diff", "--name-only", PRODUCER_COMMIT, LAUNCH_COMMIT, "--",
        ):
            return "\n".join(changed if changed is not None else [launch_top])
        raise AssertionError(f"unexpected git command: {arguments}")

    monkeypatch.setattr(launch, "_run_git", run_git)
    monkeypatch.setattr(launch, "_is_ancestor", lambda *_args: None)
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_revision",
        lambda *_args, **_kwargs: ("0" * 64, 1),
    )


def test_git_authority_binds_producer_parent_current_head_and_only_launch_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    (repo / "src/vitnns").mkdir(parents=True)
    record_path = repo / (
        f"evidence/stage_c/run/{launch.LAUNCH_RECORD_FILENAME}"
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_text("{}\n", encoding="utf-8")
    document = _valid_document("1" * 64)
    document["launch_record_commit"] = LAUNCH_COMMIT
    _install_git_authority_stubs(monkeypatch, repo, record_path)

    assert launch._validate_git_authority(
        repo, repo, record_path, document, (), require_tracked=True
    ) == LAUNCH_COMMIT


def test_self_launch_load_allows_descendant_but_execution_capture_rejects_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    (repo / "src/vitnns").mkdir(parents=True)
    record_path = repo / (
        f"evidence/stage_c/run/{launch.LAUNCH_RECORD_FILENAME}"
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_text("{}\n", encoding="utf-8")
    document = _valid_document("1" * 64)
    document["launch_record_commit"] = "SELF"
    descendant = "e" * 40
    _install_git_authority_stubs(
        monkeypatch, repo, record_path, head=descendant,
    )

    assert launch._validate_git_authority(
        repo, repo, record_path, document, (), require_tracked=True,
    ) == LAUNCH_COMMIT

    record = launch.RecoveryC7LaunchRecord(
        path=record_path, schema_path=record_path,
        document={"canonical_digest": "0" * 64}, binding=SimpleNamespace(),
        model_path=repo / "model.pth", data_root=repo / "data",
        devkit_path=repo / "devkit.tar.gz",
        diagnostic_root=record_path.parent / "diagnostic",
        formal_root=record_path.parent / "formal",
        reports_root=record_path.parent / "reports", model_identity={},
        file_sha256=launch.sha256_file(record_path),
        launch_commit=LAUNCH_COMMIT, repo_root=repo,
    )
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head",
        lambda *_args, **_kwargs: ("0" * 64, 3),
    )
    monkeypatch.setattr(launch, "_git_status_entries", lambda _repo: [])
    with pytest.raises(launch.RecoveryC7LaunchError, match="current HEAD"):
        launch.capture_authorized_c7_source_state(record, "pre_diagnostic")


def test_git_authority_rejects_current_head_launch_blob_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    (repo / "src/vitnns").mkdir(parents=True)
    record_path = repo / (
        f"evidence/stage_c/run/{launch.LAUNCH_RECORD_FILENAME}"
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_text("{}\n", encoding="utf-8")
    document = _valid_document("1" * 64)
    document["launch_record_commit"] = "SELF"
    _install_git_authority_stubs(
        monkeypatch, repo, record_path, head="e" * 40,
        head_launch_blob="2" * 40,
    )

    with pytest.raises(launch.RecoveryC7LaunchError, match="blob differs"):
        launch._validate_git_authority(
            repo, repo, record_path, document, (), require_tracked=True,
        )


def test_git_authority_resolves_self_with_top_anchored_pathspec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    (repo / "src/vitnns").mkdir(parents=True)
    record_path = repo / (
        f"evidence/stage_c/run/{launch.LAUNCH_RECORD_FILENAME}"
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_text("{}\n", encoding="utf-8")
    document = _valid_document("1" * 64)
    document["launch_record_commit"] = "SELF"
    calls: list[tuple[str, ...]] = []
    _install_git_authority_stubs(monkeypatch, repo, record_path)
    original = launch._run_git

    def observe(_repo: Path, *arguments: str) -> str:
        calls.append(arguments)
        return original(_repo, *arguments)

    monkeypatch.setattr(launch, "_run_git", observe)
    assert launch._validate_git_authority(
        repo, repo, record_path, document, (), require_tracked=True,
    ) == LAUNCH_COMMIT
    assert (
        "log", "-n", "1", "--format=%H", "HEAD", "--",
        f":(top)evidence/stage_c/run/{launch.LAUNCH_RECORD_FILENAME}",
    ) in calls


@pytest.mark.parametrize(
    (
        "mutation", "head", "head_source", "launch_parent",
        "changed", "message",
    ),
    [
        (
            lambda document: document["producer_git"].__setitem__(
                "tree", "f" * 40
            ),
            LAUNCH_COMMIT, SOURCE_TREE, PRODUCER_COMMIT, None,
            "Producer tree drifted",
        ),
        (
            lambda document: None, LAUNCH_COMMIT, "f" * 40, PRODUCER_COMMIT,
            None,
            "source drifted",
        ),
        (
            lambda document: None, LAUNCH_COMMIT, SOURCE_TREE, "e" * 40,
            None, "one commit directly after",
        ),
        (
            lambda document: None, LAUNCH_COMMIT, SOURCE_TREE, PRODUCER_COMMIT,
            [
                f"evidence/stage_c/run/{launch.LAUNCH_RECORD_FILENAME}",
                "src/vitnns/changed.py",
            ],
            "drift exceeds",
        ),
    ],
    ids=[
        "producer-tree", "current-source",
        "producer-parent", "extra-diff",
    ],
)
def test_git_authority_rejects_producer_or_current_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None], head: str, head_source: str,
    launch_parent: str, changed: list[str] | None, message: str,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    (repo / "src/vitnns").mkdir(parents=True)
    record_path = repo / (
        f"evidence/stage_c/run/{launch.LAUNCH_RECORD_FILENAME}"
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_text("{}\n", encoding="utf-8")
    document = _valid_document("1" * 64)
    document["launch_record_commit"] = LAUNCH_COMMIT
    mutation(document)
    _install_git_authority_stubs(
        monkeypatch, repo, record_path, head=head, head_source=head_source,
        launch_parent=launch_parent, changed=changed,
    )
    with pytest.raises(launch.RecoveryC7LaunchError, match=message):
        launch._validate_git_authority(
            repo, repo, record_path, document, (), require_tracked=True
        )


def test_clean_producer_capture_rejects_dirty_tree_before_git_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    repo.mkdir()
    monkeypatch.setattr(
        launch, "_git_status_entries",
        lambda _repo: [{"status": " M", "path": "src/vitnns/x.py"}],
    )
    monkeypatch.setattr(
        launch, "_run_git",
        lambda *_args: pytest.fail("Git identity queried after dirty-tree failure"),
    )
    with pytest.raises(launch.RecoveryC7LaunchError, match="clean Git worktree"):
        launch._capture_clean_producer(repo, repo)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda receipt: receipt["model"].__setitem__(
                "canonical_tensor_digest", "f" * 64
            ),
            "model identity drifted",
        ),
        (
            lambda receipt: receipt["artifacts"]["validation.json"].__setitem__(
                "sha256", "f" * 64
            ),
            "artifact drifted",
        ),
        (
            lambda receipt: receipt["source_git_state"].__setitem__(
                "tree", "f" * 40
            ),
            "producer tree drifted",
        ),
    ],
    ids=["model", "artifact", "producer"],
)
def test_stage_a_receipt_rejects_self_resigned_drift_without_reading_model(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None], message: str,
) -> None:
    repo = launch.REPO_ROOT
    receipt_path = launch.DEFAULT_STAGE_A_RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation_source = copy.deepcopy(receipt)
    mutate(mutation_source)
    mutation_source["canonical_digest"] = launch.launch_record_digest(
        mutation_source
    )
    provenance_path = receipt_path.parent / "provenance.json"
    validation_path = receipt_path.parent / "validation.json"
    documents = {
        receipt_path.resolve(): mutation_source,
        provenance_path.resolve(): json.loads(
            provenance_path.read_text(encoding="utf-8")
        ),
        validation_path.resolve(): json.loads(
            validation_path.read_text(encoding="utf-8")
        ),
    }
    monkeypatch.setattr(
        launch, "_read_json",
        lambda path, _context: copy.deepcopy(documents[Path(path).resolve()]),
    )
    monkeypatch.setattr(launch, "_regular_single_link_file", lambda *_: None)
    original_sha256 = launch.sha256_file
    artifact_hashes = {
        name: receipt["artifacts"][name]["sha256"]
        for name in ("extraction.pt", "provenance.json", "validation.json")
    }

    def cheap_sha256(path: Path) -> str:
        candidate = Path(path)
        if candidate.parent == receipt_path.parent and candidate.name in artifact_hashes:
            return artifact_hashes[candidate.name]
        return original_sha256(candidate)

    monkeypatch.setattr(launch, "sha256_file", cheap_sha256)
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head", lambda *_args, **_kwargs: ("0" * 64, 1),
    )
    def stage_a_git(_repo: Path, *arguments: str) -> str:
        if arguments == (
            "rev-parse", "--verify",
            f"{receipt['source_git_state']['remote_tracking_commit']}^{{commit}}",
        ):
            return receipt["source_git_state"]["remote_tracking_commit"]
        return STAGE_A_TREE

    monkeypatch.setattr(launch, "_run_git", stage_a_git)
    monkeypatch.setattr(launch, "_is_ancestor", lambda *_args: None)
    binding = load_recovery_v2_binding(verify_assets=False)

    with pytest.raises(launch.RecoveryC7LaunchError, match=message):
        launch._validate_stage_a_receipt(
            repo, repo.parent, receipt_path, binding,
        )


def test_stage_a_receipt_accepts_remote_tracking_fast_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = launch.REPO_ROOT
    receipt_path = launch.DEFAULT_STAGE_A_RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    provenance_path = receipt_path.parent / "provenance.json"
    validation_path = receipt_path.parent / "validation.json"
    documents = {
        receipt_path.resolve(): receipt,
        provenance_path.resolve(): json.loads(
            provenance_path.read_text(encoding="utf-8")
        ),
        validation_path.resolve(): json.loads(
            validation_path.read_text(encoding="utf-8")
        ),
    }
    monkeypatch.setattr(
        launch, "_read_json",
        lambda path, _context: copy.deepcopy(documents[Path(path).resolve()]),
    )
    monkeypatch.setattr(launch, "_regular_single_link_file", lambda *_: None)
    original_sha256 = launch.sha256_file
    artifact_hashes = {
        name: receipt["artifacts"][name]["sha256"]
        for name in ("extraction.pt", "provenance.json", "validation.json")
    }

    def cheap_sha256(path: Path) -> str:
        candidate = Path(path)
        if candidate.parent == receipt_path.parent and candidate.name in artifact_hashes:
            return artifact_hashes[candidate.name]
        return original_sha256(candidate)

    monkeypatch.setattr(launch, "sha256_file", cheap_sha256)
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head", lambda *_args, **_kwargs: ("0" * 64, 1),
    )

    def stage_a_git(_repo: Path, *arguments: str) -> str:
        if arguments == (
            "rev-parse", "--verify",
            f"{receipt['source_git_state']['remote_tracking_commit']}^{{commit}}",
        ):
            return receipt["source_git_state"]["remote_tracking_commit"]
        if arguments == (
            "rev-parse", "--verify",
            f"{receipt['source_git_state']['remote_tracking_ref']}^{{commit}}",
        ):
            return "e" * 40
        if arguments == (
            "rev-parse", "--verify",
            f"{receipt['source_git_state']['commit']}^{{tree}}",
        ):
            return receipt["source_git_state"]["tree"]
        return STAGE_A_TREE

    ancestor_checks: list[tuple[str, str, str]] = []

    def track_ancestor(
        _repo: Path, older: str, newer: str, context: str,
    ) -> None:
        ancestor_checks.append((older, newer, context))

    monkeypatch.setattr(launch, "_run_git", stage_a_git)
    monkeypatch.setattr(launch, "_is_ancestor", track_ancestor)
    binding = load_recovery_v2_binding(verify_assets=False)

    actual = launch._validate_stage_a_receipt(
        repo, repo.parent, receipt_path, binding,
    )

    assert actual["producer_commit"] == receipt["source_git_state"]["commit"]
    assert (
        receipt["source_git_state"]["remote_tracking_commit"],
        receipt["source_git_state"]["commit"],
        "Stage A remote-tracking commit",
    ) in ancestor_checks


def test_stage_a_receipt_rejects_missing_sealed_remote_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = launch.REPO_ROOT
    receipt_path = launch.DEFAULT_STAGE_A_RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(launch, "_regular_single_link_file", lambda *_: None)

    def stage_a_git(_repo: Path, *arguments: str) -> str:
        if arguments == (
            "rev-parse", "--verify",
            f"{receipt['source_git_state']['remote_tracking_commit']}^{{commit}}",
        ):
            return "e" * 40
        if arguments == (
            "rev-parse", "--verify",
            f"{receipt['source_git_state']['commit']}^{{tree}}",
        ):
            return receipt["source_git_state"]["tree"]
        return STAGE_A_TREE

    monkeypatch.setattr(launch, "_run_git", stage_a_git)
    binding = load_recovery_v2_binding(verify_assets=False)

    with pytest.raises(
        launch.RecoveryC7LaunchError,
        match="Stage A remote-tracking identity drifted",
    ):
        launch._validate_stage_a_receipt(
            repo, repo.parent, receipt_path, binding,
        )


def test_stage_a_receipt_rejects_sealed_remote_not_source_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = launch.REPO_ROOT
    receipt_path = launch.DEFAULT_STAGE_A_RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(launch, "_regular_single_link_file", lambda *_: None)

    def stage_a_git(_repo: Path, *arguments: str) -> str:
        if arguments == (
            "rev-parse", "--verify",
            f"{receipt['source_git_state']['remote_tracking_commit']}^{{commit}}",
        ):
            return receipt["source_git_state"]["remote_tracking_commit"]
        if arguments == (
            "rev-parse", "--verify",
            f"{receipt['source_git_state']['commit']}^{{tree}}",
        ):
            return receipt["source_git_state"]["tree"]
        return STAGE_A_TREE

    def reject_ancestor(
        _repo: Path, _older: str, newer: str, context: str,
    ) -> None:
        raise launch.RecoveryC7LaunchError(f"{context} is not an ancestor of {newer}")

    monkeypatch.setattr(launch, "_run_git", stage_a_git)
    monkeypatch.setattr(launch, "_is_ancestor", reject_ancestor)
    binding = load_recovery_v2_binding(verify_assets=False)

    with pytest.raises(
        launch.RecoveryC7LaunchError,
        match="Stage A remote-tracking commit is not an ancestor",
    ):
        launch._validate_stage_a_receipt(
            repo, repo.parent, receipt_path, binding,
        )
