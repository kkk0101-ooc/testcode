from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from vitnns import recovery_c8_salvage_repair as repair
from vitnns import stage_c_c8 as c8
from vitnns.recovery_v2 import load_recovery_v2_binding


def test_repair_schema_is_recursively_closed() -> None:
    schema = json.loads(repair.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" and "properties" in value:
                assert value.get("additionalProperties") is False
                assert set(value.get("required", ())) == set(value["properties"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)


def test_repair_schema_fixes_custody_and_denies_downstream_scope() -> None:
    schema = json.loads(repair.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    prior = schema["properties"]["prior_partial_root"]["properties"]
    assert prior["evidence_commit"] == {
        "const": repair.PRIOR_PARTIAL_CUSTODY_COMMIT
    }
    assert prior["common_tree_object"] == {
        "const": repair.PRIOR_PARTIAL_COMMON_TREE
    }
    assert prior["gap_tree_object"] == {
        "const": repair.PRIOR_PARTIAL_GAP_TREE
    }
    scope = schema["properties"]["scope"]["properties"]
    for key in (
        "holdout_access_authorized", "c9_authorized",
        "c10_authorized", "stage_d_authorized",
    ):
        assert scope[key] == {"const": False}


def test_repair_policy_declares_three_sources_and_advisory_counts() -> None:
    schema = json.loads(repair.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    origins = schema["properties"]["origins"]["properties"]
    assert {key: value["const"] for key, value in origins.items()} == repair.ORIGINS
    policy = schema["properties"]["policy"]["properties"]
    assert policy["three_source_union"] == {"const": True}
    assert policy["source_recomputation_required"] == {"const": True}
    assert policy["valid_terminal_failure_inherited"] == {"const": True}
    assert policy["reported_counts_trusted"] == {"const": False}
    counts = schema["properties"]["prior_partial_root"]["properties"][
        "reported_counts"
    ]["properties"]
    assert {key: value["const"] for key, value in counts.items()} == (
        repair.REPORTED_COUNTS
    )


def test_real_partial_custody_tree_controls_and_inventory_are_exact() -> None:
    repo = repair.REPO_ROOT.resolve()
    if not repair.DEFAULT_PRIOR_PARTIAL_ROOT.exists():
        pytest.skip("requires local full C8 salvage evidence root")
    try:
        _git(repo, "cat-file", "-e", repair.PRIOR_PARTIAL_CUSTODY_COMMIT + "^{commit}")
    except subprocess.CalledProcessError:
        pytest.skip("requires local full C8 salvage custody commit")
    binding = repair._prior_binding(repo, repair.DEFAULT_PRIOR_PARTIAL_ROOT)
    assert binding["evidence_commit"] == repair.PRIOR_PARTIAL_CUSTODY_COMMIT
    assert binding["common_tree_object"] == repair.PRIOR_PARTIAL_COMMON_TREE
    assert binding["gap_tree_object"] == repair.PRIOR_PARTIAL_GAP_TREE
    assert binding["control_sha256"] == repair.PRIOR_PARTIAL_CONTROL_SHA256
    assert binding["inventory"] == repair.PRIOR_PARTIAL_GAP_INVENTORY
    assert binding["reported_counts"] == repair.REPORTED_COUNTS
    assert binding["reported_counts_are_advisory"] is True


@pytest.mark.parametrize(
    "relative",
    (
        "unexpected.json",
        "units/" + "a" * 64 + "/attempts/attempt-999999/status.json",
    ),
)
def test_historical_source_custody_rejects_untracked_entries(
    tmp_path: Path, relative: str,
) -> None:
    repo = tmp_path / "repo"
    root = repo / "evidence/stage_c/original/formal"
    root.mkdir(parents=True)
    (root / "tracked.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "repair-test@example.invalid")
    _git(repo, "config", "user.name", "Repair Test")
    custody = _commit(repo, "original custody")
    extra = root / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        repair.RecoveryC8SalvageRepairError,
        match="outside its custody commit",
    ):
        repair._require_custody_clean(repo, root, custody)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD^{commit}")


def _runtime_repository(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    source = repo / repair.SOURCE_SUBTREE_PATH
    source.mkdir(parents=True)
    (source / "repair.py").write_text("VERSION = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "repair-test@example.invalid")
    _git(repo, "config", "user.name", "Repair Test")
    producer_commit = _commit(repo, "producer")
    producer = {
        "commit": producer_commit,
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "source_subtree_path": repair.SOURCE_SUBTREE_PATH,
        "source_subtree_tree": _git(
            repo, "rev-parse", f"HEAD:{repair.SOURCE_SUBTREE_PATH}",
        ),
        "dirty": False,
    }
    common = repo / "evidence/stage_c/repair-v1"
    launch = common / repair.LAUNCH_FILENAME
    launch.parent.mkdir(parents=True)
    launch.write_text("{}\n", encoding="utf-8")
    launch_commit = _commit(repo, "launch only")
    roots = tuple(
        f"evidence/stage_c/repair-v1/{name}"
        for name in ("repair", "reports", "selection", "logs")
    )
    return {
        "repo": repo, "launch": launch, "producer": producer,
        "launch_commit": launch_commit, "roots": roots,
    }


def test_repair_launch_is_exact_direct_parent_and_launch_only(tmp_path: Path) -> None:
    case = _runtime_repository(tmp_path)
    assert repair._resolve_launch(
        case["repo"], case["launch"], "SELF",
        case["producer"]["commit"],
    ) == case["launch_commit"]
    with pytest.raises(repair.RecoveryC8SalvageRepairError, match="directly"):
        repair._resolve_launch(
            case["repo"], case["launch"], "SELF", "0" * 40,
        )


def test_repair_runtime_accepts_evidence_only_descendant_with_real_head(
    tmp_path: Path,
) -> None:
    case = _runtime_repository(tmp_path)
    report = case["repo"] / case["roots"][1] / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}\n", encoding="utf-8")
    execution_commit = _commit(case["repo"], "evidence custody")
    pending = case["repo"] / case["roots"][0] / "pending.json"
    pending.parent.mkdir(parents=True)
    pending.write_text("{}\n", encoding="utf-8")

    identity = repair._validate_runtime(
        case["repo"], launch_commit=case["launch_commit"],
        producer=case["producer"], evidence_roots=case["roots"],
        check_worktree=True,
    )
    assert identity == {
        "commit": execution_commit,
        "tree": _git(case["repo"], "rev-parse", "HEAD^{tree}"),
        "source_subtree_path": repair.SOURCE_SUBTREE_PATH,
        "source_subtree_tree": case["producer"]["source_subtree_tree"],
        "dirty": False, "evidence_worktree_dirty": True,
        "authorization_launch_commit": case["launch_commit"],
    }


def test_repair_runtime_rejects_any_source_subtree_change(tmp_path: Path) -> None:
    case = _runtime_repository(tmp_path)
    (case["repo"] / repair.SOURCE_SUBTREE_PATH / "repair.py").write_text(
        "VERSION = 2\n", encoding="utf-8",
    )
    _commit(case["repo"], "source change")
    with pytest.raises(
        repair.RecoveryC8SalvageRepairError, match="source subtree differs",
    ):
        repair._validate_runtime(
            case["repo"], launch_commit=case["launch_commit"],
            producer=case["producer"], evidence_roots=case["roots"],
            check_worktree=True,
        )


def test_hot_runtime_revalidation_does_not_rehash_historical_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    launch = repo / "evidence/stage_c/repair-v1" / repair.LAUNCH_FILENAME
    schema = repo / "config/repair-schema.json"
    launch.parent.mkdir(parents=True)
    schema.parent.mkdir(parents=True)
    schema.write_text("{}\n", encoding="utf-8")
    producer = {
        "commit": "1" * 40, "tree": "2" * 40,
        "source_subtree_path": repair.SOURCE_SUBTREE_PATH,
        "source_subtree_tree": "3" * 40, "dirty": False,
    }
    outputs = {
        "common_root": "evidence/stage_c/repair-v1",
        "launch_record_path": (
            "evidence/stage_c/repair-v1/" + repair.LAUNCH_FILENAME
        ),
        "repair_root": "evidence/stage_c/repair-v1/repair",
        "reports_root": "evidence/stage_c/repair-v1/reports",
        "selection_root": "evidence/stage_c/repair-v1/selection",
        "logs_root": "evidence/stage_c/repair-v1/logs",
        "combined_manifest_path": (
            "evidence/stage_c/repair-v1/reports/"
            "c8_salvage_repair_combined_manifest.json"
        ),
        "combined_report_path": (
            "evidence/stage_c/repair-v1/reports/"
            "c8_salvage_repair_combined_report.json"
        ),
        "selection_path": (
            "evidence/stage_c/repair-v1/selection/"
            "stage_c_salvage_repair_selection.json"
        ),
        "selection_report_path": (
            "evidence/stage_c/repair-v1/reports/"
            "stage_c_c8_salvage_repair_selection_report.json"
        ),
    }
    document = {
        "canonical_digest": "f" * 64,
        "launch_record_commit": "SELF",
        "launch_schema": {
            "path": "config/repair-schema.json", "file_sha256": "e" * 64,
        },
        "producer_git": producer, "outputs": outputs,
    }
    launch.write_text(json.dumps(document) + "\n", encoding="utf-8")
    paths = tuple(repo / outputs[key] for key in (
        "repair_root", "reports_root", "selection_root", "logs_root",
        "combined_manifest_path", "combined_report_path", "selection_path",
        "selection_report_path",
    ))
    record = repair.RecoveryC8SalvageRepairLaunchRecord(
        path=launch, schema_path=schema, document=document,
        binding=object(), c7_launch_record=object(), source_c8_record=object(),
        original_salvage_record=object(), source_root=repo / "original",
        prior_gap_root=repo / "prior", repair_root=paths[0],
        reports_root=paths[1], selection_root=paths[2], logs_root=paths[3],
        combined_manifest_path=paths[4], combined_report_path=paths[5],
        selection_path=paths[6], selection_report_path=paths[7],
        model_path=repo / "model.pth", data_root=repo / "data",
        launch_commit="4" * 40, file_sha256="d" * 64, repo_root=repo,
    )
    monkeypatch.setattr(
        repair, "_read_json",
        lambda path, _context: document if path == launch else {},
    )
    monkeypatch.setattr(repair, "_validate_schema", lambda *_args: None)
    monkeypatch.setattr(repair, "canonical_sha256", lambda *_args: "f" * 64)
    monkeypatch.setattr(
        repair, "sha256_file",
        lambda path: "d" * 64 if path == launch else "e" * 64,
    )
    monkeypatch.setattr(
        repair, "_validate_outputs",
        lambda *_args: (*paths[:4], tuple(outputs[key] for key in (
            "repair_root", "reports_root", "selection_root", "logs_root",
        ))),
    )

    def git(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "1" * 40 + "^{tree}"):
            return "2" * 40
        if arguments[:2] == ("rev-parse", "1" * 40 + ":src/vitnns"):
            return "3" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(repair, "_git", git)
    monkeypatch.setattr(repair, "_top_relative", lambda *_args: "src/vitnns")
    monkeypatch.setattr(repair, "_resolve_launch", lambda *_args: "4" * 40)
    monkeypatch.setattr(repair, "_validate_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        repair, "_tree_inventory",
        lambda *_args, **_kwargs: pytest.fail("hot path rehashed history"),
    )

    assert repair.revalidate_recovery_c8_salvage_repair_runtime_record(
        record
    ) is record


def test_authorized_source_state_is_strict_two_field_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair, "capture_c8_salvage_repair_verifier_identity",
        lambda record, phase: {
            "commit": "a" * 40, "tree": "b" * 40,
            "source_subtree_path": repair.SOURCE_SUBTREE_PATH,
            "source_subtree_tree": "c" * 40, "dirty": False,
            "evidence_worktree_dirty": True,
            "authorization_launch_commit": "d" * 40,
        },
    )
    assert repair.capture_authorized_c8_salvage_repair_source_state(
        object()  # type: ignore[arg-type]
    ) == {"commit": "a" * 40, "dirty": False}


def test_c8_read_only_boundary_reloads_typed_repair_authority_but_launch_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = load_recovery_v2_binding(verify_assets=False)
    record = repair.RecoveryC8SalvageRepairLaunchRecord(
        path=tmp_path / "repair-authority.json",
        schema_path=tmp_path / "repair-schema.json",
        document={}, binding=binding, c7_launch_record=object(),
        source_c8_record=object(), original_salvage_record=object(),
        source_root=tmp_path / "original",
        prior_gap_root=tmp_path / "prior", repair_root=tmp_path / "repair",
        reports_root=tmp_path / "reports",
        selection_root=tmp_path / "selection", logs_root=tmp_path / "logs",
        combined_manifest_path=tmp_path / "reports/manifest.json",
        combined_report_path=tmp_path / "reports/report.json",
        selection_path=tmp_path / "selection/selection.json",
        selection_report_path=tmp_path / "reports/selection-report.json",
        model_path=tmp_path / "model.pth", data_root=tmp_path / "data",
        launch_commit="a" * 40, file_sha256="b" * 64, repo_root=tmp_path,
    )
    calls = []
    monkeypatch.setattr(
        repair, "load_recovery_c8_salvage_repair_launch_record",
        lambda *args, **kwargs: calls.append((args, kwargs)) or record,
    )

    assert c8._trusted_recovery_c8_evidence_record(record) is record
    assert calls == [((record.path,), {
        "repo_root": tmp_path, "schema_path": record.schema_path,
        "require_fresh_outputs": False, "require_tracked": True,
    })]
    with pytest.raises(
        c8.StageCBenchmarkIntegrityError,
        match="trusted recovery C8 context",
    ):
        c8._trusted_recovery_c8_record(record)


def test_amendment_explicitly_requires_reaudit_and_valid_failure_inheritance() -> None:
    text = repair.DEFAULT_AMENDMENT_PATH.read_text(encoding="utf-8")
    assert "exactly three evidence origins" in text
    assert "568 reported" in text
    assert "one reported failed attempt" in text
    assert "151 units" in text
    assert "not trusted classifications" in text
    assert "must not be rerun" in text


def test_generate_and_uncommitted_load_bind_closed_repair_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    schema = repo / "config/recovery_c8_salvage_repair_launch_record.schema.json"
    amendment = repo / "docs/recovery_c8_salvage_repair_amendment_20260903.md"
    original_path = repo / (
        "evidence/stage_c/c8_salvage_recovery_v2_ac77b47/"
        "recovery_c8_salvage_launch_record.json"
    )
    prior = repo / "evidence/stage_c/c8_salvage_recovery_v2_ac77b47"
    schema.parent.mkdir(parents=True)
    schema.write_bytes(repair.DEFAULT_SCHEMA_PATH.read_bytes())
    amendment.parent.mkdir(parents=True)
    amendment.write_bytes(repair.DEFAULT_AMENDMENT_PATH.read_bytes())
    original_path.parent.mkdir(parents=True)
    original_path.write_text("{}\n", encoding="utf-8")
    producer = {
        "commit": "e" * 40, "tree": "1" * 40,
        "source_subtree_path": repair.SOURCE_SUBTREE_PATH,
        "source_subtree_tree": "2" * 40, "dirty": False,
    }
    original = SimpleNamespace(
        path=original_path, file_sha256=repair.ORIGINAL_SALVAGE_FILE_SHA256,
        canonical_digest=repair.ORIGINAL_SALVAGE_CANONICAL_DIGEST,
        launch_commit=repair.ORIGINAL_SALVAGE_LAUNCH_COMMIT,
        document={
            "producer_git": {"commit": repair.ORIGINAL_SALVAGE_PRODUCER_COMMIT},
            "source_interrupted_root": {
                "formal_root": "evidence/stage_c/original/formal",
                "evidence_commit": "3" * 40, "tree_object": "4" * 40,
                "execution_commit": "5" * 40, "plan_digest": "6" * 64,
                "control_sha256": {
                    "execution_plan.json": "7" * 64,
                    "preflight.json": "8" * 64,
                    "c8_launch_authorization.json": "9" * 64,
                    "failure_table.json": "a" * 64,
                    "batch_status.json": "b" * 64,
                },
            },
            "contracts": {
                "config": {
                    "path": "config/stage_c.json", "file_sha256": "c" * 64,
                    "canonical_sha256": "d" * 64,
                },
                "run_spec": {
                    "path": "config/run.json", "file_sha256": "e" * 64,
                    "canonical_sha256": "f" * 64,
                },
            },
            "model": {
                "identity": "model", "state_dict_path": "model.pth",
                "state_dict_sha256": "1" * 64,
                "canonical_tensor_digest": "2" * 64,
                "historical_artifact_recovered": False,
            },
        },
        binding=object(), c7_launch_record=object(), source_c8_record=object(),
        source_root=repo / "evidence/stage_c/original/formal",
        model_path=repo / "model.pth", data_root=repo / "data",
    )
    prior_binding = {
        "common_root": "evidence/stage_c/c8_salvage_recovery_v2_ac77b47",
        "gap_root": "evidence/stage_c/c8_salvage_recovery_v2_ac77b47/gap",
        "evidence_commit": repair.PRIOR_PARTIAL_CUSTODY_COMMIT,
        "common_tree_object": repair.PRIOR_PARTIAL_COMMON_TREE,
        "gap_tree_object": repair.PRIOR_PARTIAL_GAP_TREE,
        "control_sha256": dict(repair.PRIOR_PARTIAL_CONTROL_SHA256),
        "inventory": dict(repair.PRIOR_PARTIAL_GAP_INVENTORY),
        "reported_counts": dict(repair.REPORTED_COUNTS),
        "reported_counts_are_advisory": True,
    }
    monkeypatch.setattr(repair, "_historical_original_salvage", lambda *_: original)
    monkeypatch.setattr(repair, "_prior_binding", lambda *_: prior_binding)
    monkeypatch.setattr(repair, "_capture_producer", lambda *_: producer)
    amendment_commit = "f" * 40

    launch = repo / "evidence/stage_c/repair-unit" / repair.LAUNCH_FILENAME
    kwargs = {
        "repo_root": repo, "launch_record_path": launch,
        "repair_root": launch.parent / "repair",
        "reports_root": launch.parent / "reports",
        "selection_root": launch.parent / "selection",
        "prior_partial_root": prior,
        "original_salvage_authority_path": original_path,
        "amendment_path": amendment, "schema_path": schema,
    }
    monkeypatch.setattr(repair, "_top", lambda _repo: repo)
    def generate_git(_repo: Path, *arguments: str) -> str:
        if arguments[:5] == ("log", "-n", "1", "--format=%H", "HEAD"):
            return amendment_commit
        if arguments[:3] == ("merge-base", "--is-ancestor", amendment_commit):
            return ""
        if arguments[:3] == (
            "merge-base", "--is-ancestor", repair.PRIOR_PARTIAL_CUSTODY_COMMIT,
        ):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(repair, "_git", generate_git)
    document = repair.generate_recovery_c8_salvage_repair_launch_record(**kwargs)
    assert document["origins"] == repair.ORIGINS
    assert document["prior_partial_root"] == prior_binding
    assert document["policy"]["reported_counts_trusted"] is False
    assert document["canonical_digest"] == repair.canonical_sha256({
        key: value for key, value in document.items() if key != "canonical_digest"
    })

    launch.parent.mkdir(parents=True)
    launch.write_text(
        json.dumps(document, sort_keys=True) + "\n", encoding="utf-8",
    )

    def fake_git(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", f"{amendment_commit}^{{commit}}"):
            return amendment_commit
        if arguments == ("rev-parse", f"{producer['commit']}^{{tree}}"):
            return producer["tree"]
        if arguments == (
            "rev-parse", f"{producer['commit']}:{repair.SOURCE_SUBTREE_PATH}",
        ):
            return producer["source_subtree_tree"]
        if arguments == ("merge-base", "--is-ancestor", amendment_commit, producer["commit"]):
            return ""
        if arguments == (
            "merge-base", "--is-ancestor",
            repair.PRIOR_PARTIAL_CUSTODY_COMMIT, producer["commit"],
        ):
            return ""
        if arguments == ("rev-parse", "HEAD^{commit}"):
            return producer["commit"]
        raise AssertionError(arguments)

    monkeypatch.setattr(repair, "_git", fake_git)
    monkeypatch.setattr(
        repair, "_git_bytes",
        lambda _repo, *arguments: amendment.read_bytes()
        if arguments[0] == "show" else b"",
    )
    monkeypatch.setattr(
        repair, "_worktree_dirty_paths",
        lambda _repo: [launch.relative_to(repo).as_posix()],
    )
    loaded = repair.load_recovery_c8_salvage_repair_launch_record(
        launch, repo_root=repo, schema_path=schema, require_tracked=False,
    )
    assert isinstance(loaded, repair.RecoveryC8SalvageRepairLaunchRecord)
    assert loaded.repair_root == launch.parent / "repair"
    assert loaded.prior_gap_root == prior / "gap"
    assert loaded.prior_authority_digest == repair.ORIGINAL_SALVAGE_CANONICAL_DIGEST
