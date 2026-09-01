from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import pytest
from jsonschema import Draft202012Validator

from vitnns import recovery_c7_launch as c7_launch
from vitnns import recovery_c8_launch as launch


PRODUCER_COMMIT = "a" * 40
PRODUCER_TREE = "b" * 40
SOURCE_TREE = "c" * 40
LAUNCH_COMMIT = "d" * 40


def _source_c7_document() -> dict[str, Any]:
    return json.loads(
        launch.DEFAULT_C7_LAUNCH_RECORD_PATH.read_text(encoding="utf-8")
    )


def _fake_c7_record(repo: Path) -> c7_launch.RecoveryC7LaunchRecord:
    document = _source_c7_document()
    binding_record = document["recovery_binding"]
    model = document["model"]
    binding = SimpleNamespace(
        path=repo / binding_record["path"],
        schema_path=repo / binding_record["schema_path"],
        document={"binding_id": binding_record["binding_id"]},
        model=model,
        model_path=repo / model["state_dict_path"],
        data_root=repo / "assets/data/ILSVRC2012_val",
        devkit_path=repo / "assets/imagenet/devkit.tar.gz",
    )
    outputs = document["outputs"]
    return c7_launch.RecoveryC7LaunchRecord(
        path=repo / launch.C7_LAUNCH_RELATIVE_PATH,
        schema_path=repo / "config/recovery_c7_launch_record.schema.json",
        document=document, binding=binding, model_path=binding.model_path,
        data_root=binding.data_root, devkit_path=binding.devkit_path,
        diagnostic_root=repo / outputs["diagnostic_root"],
        formal_root=repo / outputs["formal_root"],
        reports_root=repo / outputs["reports_root"],
        model_identity=document["model"],
        file_sha256=launch.C7_LAUNCH_FILE_SHA256,
        launch_commit=launch.C7_LAUNCH_COMMIT, repo_root=repo,
    )


def _c7_report_record() -> dict[str, Any]:
    return {
        "path": launch.C7_REPORT_RELATIVE_PATH,
        "file_sha256": launch.C7_REPORT_FILE_SHA256,
        "canonical_digest": launch.C7_REPORT_CANONICAL_DIGEST,
        "report_commit": launch.C7_REPORT_COMMIT,
        "status": "PASS",
        "source_commit": launch.C7_LAUNCH_COMMIT,
    }


def _contracts() -> dict[str, Any]:
    return {
        "stage_c_config": {
            "path": "config/stage_c.json",
            "file_sha256": "1" * 64,
            "canonical_sha256": "2" * 64,
        },
        "run_spec": {
            "path": "config/stage_c_c8_run_spec.json",
            "file_sha256": "3" * 64,
            "canonical_sha256": "4" * 64,
        },
    }


def _producer() -> dict[str, Any]:
    return {
        "commit": PRODUCER_COMMIT, "tree": PRODUCER_TREE,
        "source_subtree_path": launch.SOURCE_SUBTREE_PATH,
        "source_subtree_tree": SOURCE_TREE, "dirty": False,
    }


def _resign(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["canonical_digest"] = launch.launch_record_digest(result)
    return result


def _valid_document(repo: Path, schema_sha256: str) -> dict[str, Any]:
    c7 = _fake_c7_record(repo)
    source = c7.document
    common = "evidence/stage_c/recovery_c8_unit"
    document: dict[str, Any] = {
        "$schema": "recovery_c8_launch_record.schema.json",
        "schema_version": 1,
        "document_type": launch.DOCUMENT_TYPE,
        "record_id": launch.RECORD_ID,
        "status": launch.STATUS,
        "launch_record_commit": "SELF",
        "launch_schema": {
            "path": launch.SCHEMA_RELATIVE_PATH,
            "file_sha256": schema_sha256,
        },
        "amendment": copy.deepcopy(source["amendment"]),
        "recovery_binding": copy.deepcopy(source["recovery_binding"]),
        "stage_a": copy.deepcopy(source["stage_a"]),
        "c7_launch": launch._c7_launch_record(repo, c7),
        "c7_report": _c7_report_record(),
        "model": copy.deepcopy(source["model"]),
        "c8_contracts": _contracts(),
        "producer_git": _producer(),
        "outputs": {
            "common_root": common,
            "launch_record_path": f"{common}/{launch.LAUNCH_RECORD_FILENAME}",
            "formal_root": f"{common}/formal",
            "reports_root": f"{common}/reports",
            "selection_root": f"{common}/selection",
        },
        "scope": copy.deepcopy(launch._SCOPE),
    }
    return _resign(document)


def _write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(launch._json_file_bytes(document))


@pytest.fixture
def launch_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    repo = tmp_path / "vitnns_rebuild"
    schema = repo / launch.SCHEMA_RELATIVE_PATH
    schema.parent.mkdir(parents=True)
    schema.write_bytes(launch.DEFAULT_SCHEMA_PATH.read_bytes())
    document = _valid_document(repo, launch.sha256_file(schema))
    path = repo / document["outputs"]["launch_record_path"]
    _write(path, document)
    c7 = _fake_c7_record(repo)
    c7.path.parent.mkdir(parents=True, exist_ok=True)
    c7.path.write_text("{}\n", encoding="utf-8")
    report_path = repo / launch.C7_REPORT_RELATIVE_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(launch, "_load_c7_authority", lambda *_args, **_kwargs: c7)

    def validate_report(
        _repo: Path, _top: Path, _path: Path, _c7: Any,
        expected: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        actual = _c7_report_record()
        if expected is not None and launch.canonical_json_bytes(expected) != (
            launch.canonical_json_bytes(actual)
        ):
            raise launch.RecoveryC8LaunchError(
                "C7 PASS report launch binding drifted"
            )
        return actual

    monkeypatch.setattr(launch, "_validate_c7_report", validate_report)
    monkeypatch.setattr(
        launch, "_contract_records", lambda *_args: copy.deepcopy(_contracts()),
    )
    monkeypatch.setattr(launch, "_validate_lfs_coverage", lambda *_args: None)

    def validate_git(
        _repo: Path, _top: Path, _launch: Path, candidate: Mapping[str, Any],
        _critical: Sequence[Path], *, require_tracked: bool,
    ) -> str:
        if candidate["producer_git"] != _producer():
            raise launch.RecoveryC8LaunchError("Producer Git identity drifted")
        return LAUNCH_COMMIT if require_tracked else "SELF"

    monkeypatch.setattr(launch, "_validate_git_authority", validate_git)
    return SimpleNamespace(
        repo=repo, schema=schema, path=path, document=document, c7=c7,
    )


def _load(case: SimpleNamespace, **kwargs: Any) -> launch.RecoveryC8LaunchRecord:
    return launch.load_recovery_c8_launch_record(
        case.path, repo_root=case.repo, schema_path=case.schema,
        require_fresh_roots=kwargs.pop("require_fresh_roots", True),
        require_tracked=kwargs.pop("require_tracked", True), **kwargs,
    )


def test_schema_is_draft_2020_12_and_recursively_closed() -> None:
    schema = json.loads(launch.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node.get("required", ())) == set(node.get("properties", {}))
        for child in node.values():
            if isinstance(child, dict):
                visit(child)
            elif isinstance(child, list):
                for item in child:
                    visit(item)

    visit(schema)


def test_digest_is_order_stable_and_excludes_only_self_digest() -> None:
    left = {"b": [2, 3], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2, 3]}
    assert launch.canonical_sha256(left) == launch.canonical_sha256(right)
    value = {**left, "canonical_digest": "0" * 64}
    assert launch.launch_record_digest(value) == launch.canonical_sha256(left)


def test_load_returns_frozen_authority_and_exact_output_files(launch_case) -> None:
    record = _load(launch_case)
    assert isinstance(record, launch.RecoveryC8LaunchRecord)
    assert record.c7_launch_record is launch_case.c7
    assert record.report_path == (
        record.reports_root / launch.C8_REPORT_FILENAME
    )
    assert record.selection_path == (
        record.selection_root / launch.C8_SELECTION_FILENAME
    )
    assert record.launch_commit == LAUNCH_COMMIT
    assert record.model_identity["canonical_tensor_digest"] == (
        launch_case.document["model"]["canonical_tensor_digest"]
    )
    with pytest.raises(TypeError):
        record.document["scope"]["c9_authorized"] = True

    validated = launch.validate_recovery_c8_launch_record(
        launch_case.path, repo_root=launch_case.repo,
        schema_path=launch_case.schema, require_fresh_roots=True,
        require_tracked=True,
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
    with pytest.raises(launch.RecoveryC8LaunchError, match=message):
        _load(launch_case)


def test_loader_rejects_unknown_fields_and_stale_digest(launch_case) -> None:
    document = copy.deepcopy(launch_case.document)
    document["outputs"]["unexpected"] = True
    _write(launch_case.path, _resign(document))
    with pytest.raises(launch.RecoveryC8LaunchError, match="schema failure"):
        _load(launch_case)

    document = copy.deepcopy(launch_case.document)
    document["c7_report"]["file_sha256"] = "f" * 64
    _write(launch_case.path, document)
    with pytest.raises(launch.RecoveryC8LaunchError, match="canonical digest"):
        _load(launch_case)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["c7_launch"].__setitem__(
                "canonical_digest", "f" * 64
            ),
            "C7 launch handoff drifted",
        ),
        (
            lambda value: value["c7_report"].__setitem__(
                "canonical_digest", "f" * 64
            ),
            "C7 PASS report launch binding drifted",
        ),
        (
            lambda value: value["model"].__setitem__(
                "canonical_tensor_digest", "f" * 64
            ),
            "model handoff drifted",
        ),
        (
            lambda value: value["c8_contracts"]["run_spec"].__setitem__(
                "canonical_sha256", "f" * 64
            ),
            "C8 contracts drifted",
        ),
        (
            lambda value: value["producer_git"].__setitem__(
                "source_subtree_tree", "f" * 40
            ),
            "Producer Git identity drifted",
        ),
    ],
)
def test_self_resigned_semantic_tamper_is_rejected(
    launch_case, mutate: Callable[[dict[str, Any]], None], message: str,
) -> None:
    document = copy.deepcopy(launch_case.document)
    mutate(document)
    _write(launch_case.path, _resign(document))
    with pytest.raises(launch.RecoveryC8LaunchError, match=message):
        _load(launch_case)


def test_scope_is_c8_only_and_provenance_context_returns_exact_c7_pair(
    launch_case,
) -> None:
    record = _load(launch_case)
    assert record.document["scope"] == {
        "execution_authorization": "RECOVERY_C8_CALIBRATION_AND_SELECTION_ONLY",
        "rerun_interpretation": "recovery_replication_not_new_confirmatory",
        "c7_authorized": False,
        "c8_authorized": True,
        "c8_selection_authorized": True,
        "stage_c_holdout_access_authorized": False,
        "c9_authorized": False,
        "c10_authorized": False,
        "stage_d_authorized": False,
    }
    context = launch.recovery_provenance_context(record)
    assert context == {
        "recovery_binding": record.binding,
        "recovery_launch_record": record.c7_launch_record,
    }
    with pytest.raises(launch.RecoveryC8LaunchError, match="validated"):
        launch.recovery_provenance_context(SimpleNamespace())


def test_generate_is_schema_valid_clean_and_does_not_create_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    schema = repo / launch.SCHEMA_RELATIVE_PATH
    schema.parent.mkdir(parents=True)
    schema.write_bytes(launch.DEFAULT_SCHEMA_PATH.read_bytes())
    c7 = _fake_c7_record(repo)
    c7.path.parent.mkdir(parents=True, exist_ok=True)
    c7.path.write_text("{}\n", encoding="utf-8")
    report = repo / launch.C7_REPORT_RELATIVE_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(launch, "_load_c7_authority", lambda *_a, **_k: c7)
    monkeypatch.setattr(launch, "_validate_c7_report", lambda *_a, **_k: _c7_report_record())
    monkeypatch.setattr(launch, "_contract_records", lambda *_a: _contracts())
    monkeypatch.setattr(launch, "_validate_lfs_coverage", lambda *_a: None)
    captures: list[dict[str, Any]] = []

    def capture(_repo: Path, _top: Path) -> dict[str, Any]:
        captures.append(_producer())
        return copy.deepcopy(_producer())

    monkeypatch.setattr(launch, "_capture_clean_producer", capture)
    monkeypatch.setattr(launch, "_is_ancestor", lambda *_a: None)
    monkeypatch.setattr(
        launch, "_run_git",
        lambda _repo, *args: f"{args[-1]}: filter: lfs",
    )
    common = repo / "evidence/stage_c/recovery_c8_unit"
    document = launch.generate_recovery_c8_launch_record(
        repo_root=repo,
        launch_record_path=common / launch.LAUNCH_RECORD_FILENAME,
        formal_root=common / "formal", reports_root=common / "reports",
        selection_root=common / "selection", c7_launch_record_path=c7.path,
        c7_report_path=report, schema_path=schema,
    )
    Draft202012Validator(json.loads(schema.read_text())).validate(document)
    assert document["canonical_digest"] == launch.launch_record_digest(document)
    assert document["launch_record_commit"] == "SELF"
    assert document["c7_report"]["report_commit"] == launch.C7_REPORT_COMMIT
    assert document["scope"]["c8_authorized"] is True
    assert document["scope"]["stage_c_holdout_access_authorized"] is False
    assert len(captures) == 2
    assert not common.exists()


@pytest.mark.parametrize(
    ("launch_path", "formal", "reports", "selection", "message"),
    [
        ("../x/recovery_c8_launch_record.json", "formal", "reports", "selection", "normalized"),
        ("out/run/recovery_c8_launch_record.json", "formal", "reports", "selection", "evidence/stage_c"),
        ("evidence/stage_c/run/recovery_c8_launch_record.json", "nested/formal", "reports", "selection", "direct child"),
        ("evidence/stage_c/run/recovery_c8_launch_record.json", "wrong", "reports", "selection", "named formal"),
    ],
)
def test_output_paths_are_repo_local_direct_named_and_fresh(
    tmp_path: Path, launch_path: str, formal: str, reports: str,
    selection: str, message: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    common = Path(launch_path).parent
    paths = [
        value if "/" in value else (common / value).as_posix()
        for value in (formal, reports, selection)
    ]
    with pytest.raises(launch.RecoveryC8LaunchError, match=message):
        launch._validate_output_paths(
            repo, launch_path, *paths, require_fresh_roots=True,
            launch_must_exist=False,
        )

def test_lfs_coverage_is_checked_for_all_c8_output_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    common = repo / "evidence/stage_c/run"
    observed: list[str] = []

    def run_git(_repo: Path, *args: str) -> str:
        relative = args[-1]
        observed.append(relative)
        return f"{relative}: filter: lfs"

    monkeypatch.setattr(launch, "_run_git", run_git)
    launch._validate_lfs_coverage(
        repo, common / launch.LAUNCH_RECORD_FILENAME, common / "formal",
        common / "reports", common / "selection",
    )
    assert observed == [
        f"evidence/stage_c/run/{launch.LAUNCH_RECORD_FILENAME}",
        "evidence/stage_c/run/formal/.probe",
        "evidence/stage_c/run/reports/.probe",
        "evidence/stage_c/run/selection/.probe",
    ]


def _phase_record(repo: Path) -> launch.RecoveryC8LaunchRecord:
    common = repo / "evidence/stage_c/run"
    common.mkdir(parents=True)
    path = common / launch.LAUNCH_RECORD_FILENAME
    path.write_text("{}\n", encoding="utf-8")
    c7 = _fake_c7_record(repo)
    return launch.RecoveryC8LaunchRecord(
        path=path, schema_path=path, document={
            "canonical_digest": "0" * 64, "producer_git": _producer(),
        }, binding=c7.binding, c7_launch_record=c7,
        model_path=c7.model_path, data_root=c7.data_root,
        devkit_path=c7.devkit_path, formal_root=common / "formal",
        reports_root=common / "reports", selection_root=common / "selection",
        report_path=common / "reports" / launch.C8_REPORT_FILENAME,
        selection_path=common / "selection" / launch.C8_SELECTION_FILENAME,
        model_identity={}, file_sha256=launch.sha256_file(path),
        launch_commit=LAUNCH_COMMIT, repo_root=repo,
    )


@pytest.mark.parametrize(
    ("phase", "relative_paths"),
    [
        ("pre_formal", []),
        ("formal", ["formal/unit/status.json"]),
        ("finalize", ["formal/batch_status.json"]),
        ("report", ["formal/unit.json", "reports/report.json"]),
        ("selection", ["formal/unit.json", "reports/report.json", "selection/stage_c_selection.json"]),
    ],
)
def test_phase_capture_allows_only_declared_fresh_evidence_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
    relative_paths: list[str],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    record = _phase_record(repo)
    common = "evidence/stage_c/run"
    entries = [
        {"status": "??", "path": f"{common}/{path}"}
        for path in relative_paths
    ]
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))

    def run_git(_repo: Path, *args: str) -> str:
        if args == ("rev-parse", "--verify", "HEAD^{commit}"):
            return LAUNCH_COMMIT
        if args == ("rev-parse", "HEAD:src/vitnns"):
            return SOURCE_TREE
        raise AssertionError(args)

    monkeypatch.setattr(launch, "_run_git", run_git)
    monkeypatch.setattr(launch, "_git_status_entries", lambda _repo: entries)
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head",
        lambda *_args, **_kwargs: ("0" * 64, 3),
    )
    assert launch.capture_authorized_c8_source_state(record, phase) == {
        "commit": LAUNCH_COMMIT, "dirty": False,
    }


def test_phase_capture_rejects_out_of_scope_or_tracked_dirt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    record = _phase_record(repo)
    monkeypatch.setattr(launch, "_repository_identity", lambda root: (repo, ""))
    monkeypatch.setattr(
        launch, "_run_git",
        lambda _repo, *args: LAUNCH_COMMIT
        if args[-1] == "HEAD^{commit}" else SOURCE_TREE,
    )
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_head",
        lambda *_args, **_kwargs: ("0" * 64, 3),
    )
    monkeypatch.setattr(
        launch, "_git_status_entries",
        lambda _repo: [{"status": "??", "path": "selection/stage_c_selection.json"}],
    )
    with pytest.raises(launch.RecoveryC8LaunchError, match="outside roots"):
        launch.capture_authorized_c8_source_state(record, "formal")
    monkeypatch.setattr(
        launch, "_git_status_entries",
        lambda _repo: [{
            "status": " M",
            "path": "evidence/stage_c/run/formal/status.json",
        }],
    )
    with pytest.raises(launch.RecoveryC8LaunchError, match="newly created"):
        launch.capture_authorized_c8_source_state(record, "formal")


def test_current_recovery_c7_launch_and_pass_report_revalidate_without_gpu() -> None:
    repo = launch.REPO_ROOT
    c7 = launch._load_c7_authority(
        repo, launch.DEFAULT_C7_LAUNCH_RECORD_PATH, require_tracked=True,
    )
    report = launch._validate_c7_report(
        repo, repo.parent, launch.DEFAULT_C7_REPORT_PATH, c7,
    )
    assert c7.launch_commit == launch.C7_LAUNCH_COMMIT
    assert c7.document["stage_a"]["status"] == "PASS"
    assert report == _c7_report_record()


def test_c7_report_rejects_self_resigned_non_pass_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    report_path = repo / launch.C7_REPORT_RELATIVE_PATH
    report_path.parent.mkdir(parents=True)
    report = json.loads(
        launch.DEFAULT_C7_REPORT_PATH.read_text(encoding="utf-8")
    )
    report["projections"]["hard_cap_checks"][0]["status"] = "BLOCKED"
    report["canonical_digest"] = launch.launch_record_digest(report)
    _write(report_path, report)
    c7 = _fake_c7_record(repo)
    monkeypatch.setattr(launch, "_require_tracked", lambda *_a, **_k: None)
    monkeypatch.setattr(launch, "_is_ancestor", lambda *_a, **_k: None)
    monkeypatch.setattr(launch, "_run_git", lambda *_a: launch.C7_REPORT_COMMIT)
    monkeypatch.setattr(
        launch, "_verify_lfs_pointer_at_revision",
        lambda *_a, **_k: ("0" * 64, 1),
    )
    with pytest.raises(launch.RecoveryC8LaunchError, match="ten measured"):
        launch._validate_c7_report(repo, repo, report_path, c7)


def test_c7_report_rejects_malformed_projection_container_with_public_error(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "vitnns_rebuild"
    report_path = repo / launch.C7_REPORT_RELATIVE_PATH
    report_path.parent.mkdir(parents=True)
    report = json.loads(
        launch.DEFAULT_C7_REPORT_PATH.read_text(encoding="utf-8")
    )
    report["projections"] = None
    report["canonical_digest"] = launch.launch_record_digest(report)
    _write(report_path, report)
    with pytest.raises(launch.RecoveryC8LaunchError, match="projections"):
        launch._validate_c7_report(
            repo, repo, report_path, _fake_c7_record(repo),
        )
