from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from vitnns import recovery_c8_salvage as salvage
from vitnns.stage_c_provenance import canonical_sha256


def test_salvage_schema_is_closed_and_valid():
    schema = json.loads(salvage.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["schema_version"] == {"const": 2}
    assert "logs_root" in schema["properties"]["outputs"]["required"]
    assert (
        schema["properties"]["outputs"]["properties"]
        ["post_launch_handoff_paths"]["prefixItems"]
        == [
            {"const": "docs/recovery_and_rerun.md"},
            {"const": "docs/阶段C-step-by-step.md"},
        ]
    )

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False
                assert set(node.get("required", ())) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)


def test_salvage_policy_cannot_authorize_downstream():
    schema = json.loads(salvage.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    scope = schema["properties"]["scope"]["properties"]
    assert scope["holdout_access_authorized"] == {"const": False}
    assert scope["c9_authorized"] == {"const": False}
    assert scope["c10_authorized"] == {"const": False}
    assert scope["stage_d_authorized"] == {"const": False}


def test_historical_loader_is_explicit_read_only_api(monkeypatch, tmp_path):
    from vitnns import recovery_c8_launch as base

    assert callable(base.load_historical_recovery_c8_launch_record)
    with pytest.raises(TypeError):
        # custody_commit is deliberately mandatory: callers cannot silently
        # reinterpret a current live authority as historical source evidence.
        base.load_historical_recovery_c8_launch_record(tmp_path / "missing.json")


def test_salvage_canonical_digest_detects_edit():
    document = {"x": 1}
    digest = canonical_sha256(document)
    changed = copy.deepcopy(document)
    changed["x"] = 2
    assert canonical_sha256(changed) != digest


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
    source = repo / salvage.SOURCE_SUBTREE_PATH
    source.mkdir(parents=True)
    (source / "worker.py").write_text("VERSION = 1\n", encoding="utf-8")
    # This exact path was previously granted a post-launch source exception.
    (source / "recovery_c8_salvage.py").write_text(
        "AUTHORITY_VERSION = 2\n", encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "salvage-test@example.invalid")
    _git(repo, "config", "user.name", "Salvage Test")
    producer_commit = _commit(repo, "producer")
    producer = {
        "commit": producer_commit,
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "source_subtree_path": salvage.SOURCE_SUBTREE_PATH,
        "source_subtree_tree": _git(
            repo, "rev-parse", f"HEAD:{salvage.SOURCE_SUBTREE_PATH}",
        ),
        "dirty": False,
    }
    common = repo / "evidence/stage_c/salvage-v2"
    launch = common / salvage.LAUNCH_FILENAME
    launch.parent.mkdir(parents=True)
    launch.write_text('{"schema_version":2}\n', encoding="utf-8")
    launch_commit = _commit(repo, "launch authority")
    roots = tuple(
        f"evidence/stage_c/salvage-v2/{name}"
        for name in ("gap", "reports", "selection", "logs")
    )
    return {
        "repo": repo, "launch": launch, "producer": producer,
        "launch_commit": launch_commit, "roots": roots,
    }


def test_v2_launch_commit_is_direct_child_of_producer(tmp_path: Path) -> None:
    case = _runtime_repository(tmp_path)
    assert salvage._resolve_tracked_launch_commit(
        case["repo"], case["launch"], "SELF",
        case["producer"]["commit"],
    ) == case["launch_commit"]

    with pytest.raises(salvage.RecoveryC8SalvageError, match="direct parent"):
        salvage._resolve_tracked_launch_commit(
            case["repo"], case["launch"], "SELF", "0" * 40,
        )


def test_v2_rejects_post_launch_source_change(tmp_path: Path) -> None:
    case = _runtime_repository(tmp_path)
    (case["repo"] / salvage.SOURCE_SUBTREE_PATH / "worker.py").write_text(
        "VERSION = 2\n", encoding="utf-8",
    )
    _commit(case["repo"], "change runtime source")

    with pytest.raises(
        salvage.RecoveryC8SalvageError, match="source subtree differs",
    ):
        salvage._validate_runtime_git_identity(
            case["repo"], launch_commit=case["launch_commit"],
            producer=case["producer"],
            declared_evidence_roots=case["roots"], check_worktree=True,
        )


def test_v2_accepts_evidence_only_descendant_with_real_execution_identity(
    tmp_path: Path,
) -> None:
    case = _runtime_repository(tmp_path)
    report = case["repo"] / case["roots"][1] / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}\n", encoding="utf-8")
    execution_commit = _commit(case["repo"], "seal evidence")
    dirty = case["repo"] / case["roots"][0] / "runtime.json"
    dirty.parent.mkdir(parents=True)
    dirty.write_text("{}\n", encoding="utf-8")

    identity = salvage._validate_runtime_git_identity(
        case["repo"], launch_commit=case["launch_commit"],
        producer=case["producer"],
        declared_evidence_roots=case["roots"], check_worktree=True,
    )

    assert identity == {
        "commit": execution_commit,
        "tree": _git(case["repo"], "rev-parse", "HEAD^{tree}"),
        "source_subtree_path": salvage.SOURCE_SUBTREE_PATH,
        "source_subtree_tree": case["producer"]["source_subtree_tree"],
        "dirty": False,
        "evidence_worktree_dirty": True,
        "authorization_launch_commit": case["launch_commit"],
    }
    assert identity["commit"] != identity["authorization_launch_commit"]


def test_v2_rejects_future_change_to_formerly_allowlisted_source_path(
    tmp_path: Path,
) -> None:
    case = _runtime_repository(tmp_path)
    authority_source = (
        case["repo"] / salvage.SOURCE_SUBTREE_PATH
        / "recovery_c8_salvage.py"
    )
    authority_source.write_text("AUTHORITY_VERSION = 3\n", encoding="utf-8")
    _commit(case["repo"], "future authority source change")

    with pytest.raises(
        salvage.RecoveryC8SalvageError, match="source subtree differs",
    ):
        salvage._validate_runtime_git_identity(
            case["repo"], launch_commit=case["launch_commit"],
            producer=case["producer"],
            declared_evidence_roots=case["roots"], check_worktree=True,
        )


def test_source_state_projection_is_exact_code_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = object()
    monkeypatch.setattr(
        salvage, "capture_c8_salvage_verifier_identity",
        lambda candidate, phase: {
            "commit": "a" * 40, "tree": "b" * 40,
            "source_subtree_path": salvage.SOURCE_SUBTREE_PATH,
            "source_subtree_tree": "c" * 40, "dirty": False,
            "evidence_worktree_dirty": True,
            "authorization_launch_commit": "d" * 40,
        },
    )
    assert salvage.capture_authorized_c8_salvage_source_state(record) == {
        "commit": "a" * 40, "dirty": False,
    }


def test_amendment_containing_commit_must_bind_current_bytes(tmp_path):
    case = _runtime_repository(tmp_path)
    amendment = case["repo"] / "docs/amendment.md"
    amendment.parent.mkdir(parents=True)
    amendment.write_text("approved\n", encoding="utf-8")
    amendment_commit = _commit(case["repo"], "seal amendment")
    assert salvage._git_bytes(
        case["repo"], "show", f"{amendment_commit}:docs/amendment.md",
    ) == amendment.read_bytes()
    amendment.write_text("silently changed\n", encoding="utf-8")
    assert salvage._git_bytes(
        case["repo"], "show", f"{amendment_commit}:docs/amendment.md",
    ) != amendment.read_bytes()


def test_v2_allows_only_declared_committed_handoff_paths(tmp_path: Path) -> None:
    case = _runtime_repository(tmp_path)
    handoff = case["repo"] / "docs/handoff.md"
    handoff.parent.mkdir(parents=True)
    handoff.write_text("done\n", encoding="utf-8")
    completion = _commit(case["repo"], "document handoff")
    identity = salvage._validate_runtime_git_identity(
        case["repo"], launch_commit=case["launch_commit"],
        producer=case["producer"], declared_evidence_roots=case["roots"],
        declared_committed_paths=("docs/handoff.md",), check_worktree=True,
    )
    assert identity["commit"] == completion
    unrelated = case["repo"] / "README.md"
    unrelated.write_text("drift\n", encoding="utf-8")
    _commit(case["repo"], "unrelated change")
    with pytest.raises(
        salvage.RecoveryC8SalvageError, match="escape declared evidence roots",
    ):
        salvage._validate_runtime_git_identity(
            case["repo"], launch_commit=case["launch_commit"],
            producer=case["producer"], declared_evidence_roots=case["roots"],
            declared_committed_paths=("docs/handoff.md",),
            check_worktree=True,
        )
