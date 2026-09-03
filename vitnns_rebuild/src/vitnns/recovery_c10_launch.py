"""Launch authority for the recovery-v2 C10 aggregation only."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .io_utils import sha256_file
from .stage_c_provenance import canonical_sha256, capture_git_state

SCHEMA_VERSION = 1
DOCUMENT_TYPE = "vitnns_recovery_c10_launch_record"
RECORD_ID = "RECOVERY_C10_SALVAGE_C8_NEW_C9_20260903"
STATUS = "AUTHORIZED_FOR_RECOVERY_C10_ONLY"
AMENDMENT_ID = "RECOVERY-C10-SALVAGE-C8-NEW-C9-20260903"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "config/recovery_c10_launch_record.schema.json"
DEFAULT_AMENDMENT_PATH = REPO_ROOT / "docs/recovery_c10_amendment_20260903.md"
DEFAULT_C9_AUTHORITY_PATH = (
    REPO_ROOT / "config/recovery_c9_launch_record_retry_20260903.json"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RecoveryC10LaunchError(ValueError):
    """Raised when the recovery C10 authority is invalid."""


def _plain(value: Any) -> Any:
    method = getattr(value, "to_dict", None)
    value = method() if callable(method) else value
    return json.loads(json.dumps(value, allow_nan=False))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryC10LaunchError(
            f"recovery C10 git {' '.join(args)} failed: {error}"
        ) from error


def _read(path: Path, context: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RecoveryC10LaunchError(f"duplicate key in {context}: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite {token}")
            ),
        )
    except Exception as error:
        raise RecoveryC10LaunchError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryC10LaunchError(f"{context} must be an object")
    return value


def _path(repo: Path, raw: Path | str, context: str, *, exists: bool = True) -> Path:
    candidate = Path(raw)
    result = Path(os.path.abspath(os.fspath(candidate if candidate.is_absolute() else repo / candidate)))
    try:
        result.relative_to(repo)
    except ValueError as error:
        raise RecoveryC10LaunchError(f"{context} escapes repository") from error
    current = Path(result.anchor)
    for part in result.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise RecoveryC10LaunchError(f"{context} contains a symlink")
    if exists and not result.exists():
        raise RecoveryC10LaunchError(f"{context} is missing")
    return result


def _relative(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _self_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != "canonical_digest"})


def _capture_producer(repo: Path) -> dict[str, Any]:
    state = capture_git_state(repo)
    if state.get("dirty") is not False:
        raise RecoveryC10LaunchError("recovery C10 producer worktree must be clean")
    commit = str(state["commit"])
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    source = (repo / "src/vitnns").relative_to(top).as_posix()
    return {
        "commit": commit, "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "source_subtree_path": "src/vitnns",
        "source_subtree_tree": _git(repo, "rev-parse", f"HEAD:{source}"),
        "dirty": False,
    }


def _c8_binding() -> dict[str, Any]:
    return {
        "root": "evidence/stage_c/c8_salvage_repair_recovery_v2_de71d55",
        "custody_commit": "82825131a598d4fffc22ca761e4d9a30b5d35666",
        "completion_file_sha256": "974e9366f240018d6118edc47dfbd623d9690cb407961fcf73437cee07c7898b",
        "completion_canonical_digest": "4e31223f9f4e8113b035d9d8ddeb221bec916278864873180593d106684d4612",
        "combined_manifest_canonical_digest": "52ea4e8c959d73f5c0b8e3a8b46cd4a7232042e25053a46a5d86dd9166f83811",
        "combined_manifest_file_sha256": "c6bd5f4f4b79e1b6dd67c6d53fc43e5283aca2db51b791f7c462ac5a30009101",
        "combined_artifact_manifest_digest": "4a7f1bd35aadfb704196625c709f21c1fd51159dbae8d5b1fb9dae8f95043257",
        "source_reaudit_digest": "86d946ecb096104978cf5c9b2a7bd5a6c33af91ad222c1fb49afbf1f6f5cd925",
        "selection_file_sha256": "5d26c78bb94278c7f12d7bd88b1a37cfb97352f3d0bde5e1c4e52f8e811c7982",
        "selection_canonical_digest": "6915f7c4383c91ed5fcc465a52d3582eebdb6cd325c7bc2107b03c7996c87620",
        "selection_report_canonical_digest": "b0a1c557fa734673f0acc126f83a826bf493e2861788003c95476ef784ab44ce",
        "selection_report_file_sha256": "dcb875911c0285a28fa87dc4ec8226672e15ed39d36d4e437e89c47aa9123aae",
        "origin_counts": {
            "inherited_from_original_interrupted_root": 1460,
            "inherited_from_prior_partial_salvage_root": 569,
            "newly_run_in_salvage_repair_root": 151,
        },
        "terminal_state_counts": {"failed": 1, "succeeded": 2179},
        "producer_commits": {
            "inherited_from_original_interrupted_root": "10633700a268170bd7c6f0ee1c7a52f8d4d97610",
            "inherited_from_prior_partial_salvage_root": "6a6655b604f7310e298490837cac7e66ebbf2744",
            "newly_run_in_salvage_repair_root": "6b98c17c4108ce2d0a0891c66399e4c1c89bf424",
        },
        "root_inventory_digests": {
            "inherited_from_original_interrupted_root": "25c4321a2067022ff21bb40616cd29a21aa5d5889956c1925d905b618bcd734d",
            "inherited_from_prior_partial_salvage_root": "56b3eda4a23ea15506252aae38bc7e93be53f0942f5dde306140e92d44cdae2e",
            "newly_run_in_salvage_repair_root": "c00af1b2993d482c99d8393199c34ce66bce99c7e88209e0befcf7233b59886f",
        },
    }


def _c9_binding() -> dict[str, Any]:
    return {
        "root": "evidence/stage_c/c9_recovery_v2_0d88fbb/formal",
        "producer_commit": "0d88fbb8b71c604d3dab94db19725e01b7519ae5",
        "authority_execution_commit": "c66e5224e87924253f10a9f5fead006fcc5b5ac7",
        "authority_path": "config/recovery_c9_launch_record_retry_20260903.json",
        "authority_file_sha256": "8e466718d2efa1205bcce2906442fa2ff1efac6b957f6f8693f269c66771a9cc",
        "authority_canonical_digest": "d2b24e8566e63ead545d81c1574b608e20aae41b1b151657af858706f78b12f3",
        "report_file_sha256": "ad164a87aa6dedba461e1b19d639d22d955a675d75e0c6e257af4d5846682763",
        "report_canonical_digest": "959bc874b04b007f487d05734cf5b85b1eaef75bb8c2ff2ded222c7cf9083248",
        "execution_plan_canonical_digest": "d1c0a75b22f82f580f6a2ddbc73f399159d3ca95fc6ce42518dc8382922d1eb1",
        "preflight_canonical_digest": "b1c258eceae20e0c9023df607ee28e4234a5e327660329bc92c0ea6b9714f8ec",
        "internal_authorization_canonical_digest": "f6aa3e3af953277945a522a9bddec4d3b795309274ff83a0c85d39b4506ca55a",
        "whole_root_regular_file_count": 2705,
        "whole_root_regular_file_bytes": 225985972,
        "whole_root_inventory_canonical_digest": "6ab8ee024ce7d5bfa3c6a408ec118a9eeac7960d046d7001d8019fb8ab6650c8",
        "expected_units": 320, "succeeded_units": 320,
        "primary_holdout_valid": 15,
    }


@dataclass(frozen=True)
class RecoveryC10LaunchRecord:
    path: Path
    document: Mapping[str, Any]
    c9_record: Any
    output_root: Path
    producer_commit: str
    launch_commit: str
    file_sha256: str
    repo_root: Path = REPO_ROOT

    @property
    def canonical_digest(self) -> str:
        return str(self.document["canonical_digest"])


def generate_recovery_c10_launch_record(
    *, repo_root: Path | str, launch_record_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    launch = _path(repo, launch_record_path, "C10 authority", exists=False)
    output = _path(repo, output_root, "C10 output root", exists=False)
    if os.path.lexists(launch) or os.path.lexists(output):
        raise RecoveryC10LaunchError("C10 authority and formal root must be fresh")
    producer = _capture_producer(repo)
    from .recovery_c9_launch import load_recovery_c9_launch_record
    c9_record = load_recovery_c9_launch_record(
        DEFAULT_C9_AUTHORITY_PATH, repo_root=repo, require_tracked=True,
        full_c8_validation=False, validate_runtime=False,
    )
    model = repo / "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"
    if sha256_file(model) != "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d":
        raise RecoveryC10LaunchError("recovery-v2 model file identity drifted")
    contract_paths = {
        "recovery_binding": "config/recovery_v2_binding.json",
        "stage_c_config": "config/stage_c.json",
        "c9_run_spec": "config/stage_c_c9_run_spec.json",
        "protocol": "docs/阶段C_protocol.md",
        "acceptance": "docs/stage_c_acceptance.md",
        "statistics_implementation": "src/vitnns/stage_c_c10_stats.py",
        "authority_schema": "config/recovery_c10_launch_record.schema.json",
    }
    contracts = {
        key: {"path": relative, "file_sha256": sha256_file(repo / relative)}
        for key, relative in contract_paths.items()
    }
    amendment = _path(repo, DEFAULT_AMENDMENT_PATH, "C10 amendment")
    document = {
        "$schema": DEFAULT_SCHEMA_PATH.name, "schema_version": 1,
        "document_type": DOCUMENT_TYPE, "record_id": RECORD_ID,
        "status": STATUS,
        "amendment": {
            "id": AMENDMENT_ID, "path": _relative(repo, amendment),
            "file_sha256": sha256_file(amendment),
            "containing_commit": producer["commit"],
        },
        "producer_git": producer, "launch_record_commit": "SELF",
        "upstream": {"c8_salvage": _c8_binding(), "c9": _c9_binding()},
        "model": {
            "identity": "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718",
            "state_dict_path": _relative(repo, model),
            "state_dict_sha256": "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d",
            "canonical_tensor_digest": "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd",
            "historical_artifact_recovered": False,
        },
        "contracts": contracts,
        "outputs": {"formal_root": _relative(repo, output)},
        "scope": {
            "c10_aggregation_authorized": True,
            "source_recompute_required": True,
            "bootstrap_authorized": True,
            "decision_authorized": True,
            "holm_authorized": False,
            "holm_status_required": "HOLM_BLOCKED_UNDERSPECIFIED",
            "stage_d_execution_authorized": False,
            "stage_d_analysis_authorized": False,
            "stage_d_reserve_access_authorized": False,
        },
    }
    document["canonical_digest"] = _self_digest(document)
    _validate_document(document)
    Draft202012Validator(_read(DEFAULT_SCHEMA_PATH, "C10 authority schema")).validate(document)
    return document


def write_recovery_c10_launch_record(**kwargs: Any) -> Path:
    repo = Path(kwargs["repo_root"]).resolve(strict=True)
    path = _path(repo, kwargs["launch_record_path"], "C10 authority", exists=False)
    document = generate_recovery_c10_launch_record(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _validate_document(document: Mapping[str, Any]) -> None:
    required = {
        "$schema", "schema_version", "document_type", "record_id",
        "status", "amendment", "producer_git", "launch_record_commit",
        "upstream", "model", "contracts", "outputs", "scope",
        "canonical_digest",
    }
    if set(document) != required or document.get("canonical_digest") != _self_digest(document):
        raise RecoveryC10LaunchError("C10 authority fields or digest drifted")
    if (
        document.get("schema_version") != 1
        or document.get("document_type") != DOCUMENT_TYPE
        or document.get("record_id") != RECORD_ID
        or document.get("status") != STATUS
        or document.get("upstream") != {"c8_salvage": _c8_binding(), "c9": _c9_binding()}
        or document.get("scope") != {
            "c10_aggregation_authorized": True,
            "source_recompute_required": True,
            "bootstrap_authorized": True,
            "decision_authorized": True,
            "holm_authorized": False,
            "holm_status_required": "HOLM_BLOCKED_UNDERSPECIFIED",
            "stage_d_execution_authorized": False,
            "stage_d_analysis_authorized": False,
            "stage_d_reserve_access_authorized": False,
        }
    ):
        raise RecoveryC10LaunchError("C10 authority identity, upstream, or scope drifted")
    model = document.get("model", {})
    if (
        model.get("identity") != "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718"
        or model.get("state_dict_sha256") != "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
        or model.get("canonical_tensor_digest") != "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
        or model.get("historical_artifact_recovered") is not False
    ):
        raise RecoveryC10LaunchError("C10 recovery-v2 model binding drifted")
    expected_contracts = {
        "recovery_binding", "stage_c_config", "c9_run_spec",
        "protocol", "acceptance", "statistics_implementation",
        "authority_schema",
    }
    contracts = document.get("contracts")
    if not isinstance(contracts, Mapping) or set(contracts) != expected_contracts:
        raise RecoveryC10LaunchError("C10 frozen contract set drifted")
    for name, item in contracts.items():
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "file_sha256"}
            or not isinstance(item.get("path"), str)
            or _SHA256.fullmatch(str(item.get("file_sha256"))) is None
        ):
            raise RecoveryC10LaunchError(
                f"C10 frozen contract binding is invalid: {name}"
            )
    outputs = document.get("outputs")
    if (
        not isinstance(outputs, Mapping) or set(outputs) != {"formal_root"}
        or not isinstance(outputs.get("formal_root"), str)
        or not outputs["formal_root"].startswith("evidence/stage_c/c10_recovery_v2_")
        or not outputs["formal_root"].endswith("/formal")
    ):
        raise RecoveryC10LaunchError("C10 formal output root is invalid")


def load_recovery_c10_launch_record(
    path: Path | str, *, repo_root: Path | str = REPO_ROOT,
    require_tracked: bool = True, validate_runtime: bool = True,
) -> RecoveryC10LaunchRecord:
    repo = Path(repo_root).resolve(strict=True)
    authority_path = _path(repo, path, "C10 authority")
    document = _read(authority_path, "C10 authority")
    _validate_document(document)
    Draft202012Validator(_read(DEFAULT_SCHEMA_PATH, "C10 authority schema")).validate(document)
    producer = document["producer_git"]
    if (
        producer.get("dirty") is not False
        or any(_COMMIT.fullmatch(str(producer.get(key))) is None for key in ("commit", "tree", "source_subtree_tree"))
        or producer.get("source_subtree_path") != "src/vitnns"
    ):
        raise RecoveryC10LaunchError("C10 producer binding is invalid")
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    relative = authority_path.relative_to(top).as_posix()
    launch = document["launch_record_commit"]
    if require_tracked:
        if launch == "SELF":
            launch = _git(repo, "log", "-n", "1", "--format=%H", "HEAD", "--", f":(top){relative}")
        if _git(repo, "rev-list", "--parents", "-n", "1", launch).split() != [launch, producer["commit"]]:
            raise RecoveryC10LaunchError("C10 authority must directly follow producer")
        changed = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", producer["commit"], launch).splitlines()
        if changed != [relative]:
            raise RecoveryC10LaunchError("C10 authority commit must be launch-only")
    elif launch != "SELF":
        raise RecoveryC10LaunchError("untracked C10 authority must use SELF")
    source_top = (repo / "src/vitnns").relative_to(top).as_posix()
    if (
        _git(repo, "rev-parse", f"{producer['commit']}^{{tree}}") != producer["tree"]
        or _git(repo, "rev-parse", f"{producer['commit']}:{source_top}") != producer["source_subtree_tree"]
    ):
        raise RecoveryC10LaunchError("C10 producer commit/tree drifted")
    amendment = document["amendment"]
    amendment_path = _path(repo, amendment["path"], "C10 amendment")
    if (
        amendment.get("id") != AMENDMENT_ID
        or amendment.get("containing_commit") != producer["commit"]
        or sha256_file(amendment_path) != amendment.get("file_sha256")
    ):
        raise RecoveryC10LaunchError("C10 amendment binding drifted")
    for item in document["contracts"].values():
        target = _path(repo, item["path"], "C10 frozen contract")
        if sha256_file(target) != item["file_sha256"]:
            raise RecoveryC10LaunchError("C10 frozen contract hash drifted")
    model_path = _path(repo, document["model"]["state_dict_path"], "C10 model")
    if sha256_file(model_path) != document["model"]["state_dict_sha256"]:
        raise RecoveryC10LaunchError("C10 model file hash drifted")
    from .recovery_c9_launch import load_recovery_c9_launch_record
    c9_record = load_recovery_c9_launch_record(
        DEFAULT_C9_AUTHORITY_PATH, repo_root=repo, require_tracked=True,
        full_c8_validation=False, validate_runtime=False,
    )
    output = _path(repo, document["outputs"]["formal_root"], "C10 formal root", exists=False)
    result = RecoveryC10LaunchRecord(
        authority_path, _freeze(document), c9_record, output,
        producer["commit"], launch, sha256_file(authority_path), repo,
    )
    if validate_runtime:
        state = capture_git_state(repo)
        if state.get("dirty") is not False:
            raise RecoveryC10LaunchError("C10 runtime worktree must be clean")
        _git(repo, "merge-base", "--is-ancestor", launch, str(state["commit"]))
        if _git(repo, "rev-parse", f"HEAD:{source_top}") != producer["source_subtree_tree"]:
            raise RecoveryC10LaunchError("C10 runtime source differs from producer")
    return result


__all__ = [
    "RecoveryC10LaunchError", "RecoveryC10LaunchRecord",
    "generate_recovery_c10_launch_record",
    "write_recovery_c10_launch_record", "load_recovery_c10_launch_record",
]
