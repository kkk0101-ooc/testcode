"""Recovery-v2 D7 smoke, budget, producer freeze and execution gate.

The module never resolves a Stage D registry image path.  Its integration
smoke uses deterministic arrays, while its final budget is recomputed from the
sealed recovery C9 H20 telemetry.  The only formal capability it can issue is
an execution capability for a future D8.0; analysis remains unissued.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_config import load_stage_d_contracts, strict_json_bytes
from .stage_d_plan import (
    D7_PROFILE_SOURCE_STATUS, D7_VALIDATED_PROFILE_STATUS,
    MATCHED_METHOD_ID, METHOD_IDS, NNS_METHOD_ID, StageDEstimate,
    StageDExecutionPlan, StageDPreflight, StageDResourceProfile,
    build_stage_d_plan, canonical_json_bytes, canonical_sha256,
    estimate_stage_d_run, preflight_stage_d, validate_stage_d_estimate,
    validate_stage_d_plan, validate_stage_d_preflight,
    validate_stage_d_shard_mapping,
)
from .stage_d_recovery import load_recovery_binding


class StageDD7Error(ValueError):
    """D7 cannot establish a closed, recovery-v2 execution gate."""


D7_ID = "STAGE_D_RECOVERY_V2_D7_R4_20260904"
LEGACY_D7_ID = "STAGE_D_RECOVERY_V2_D7_20260904"
SMOKE_DOCUMENT_TYPE = "stage_d_d7_synthetic_smoke_receipt"
BUDGET_DOCUMENT_TYPE = "stage_d_d7_final_budget"
AUTHORIZATION_DOCUMENT_TYPE = "stage_d_d7_execution_authorization"
COMPLETION_DOCUMENT_TYPE = "stage_d_d7_completion"
EXECUTION_AUTHORIZATION = "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"
ANALYSIS_AUTHORIZATION = "NOT_EVALUATED"
PRODUCER_REF = "refs/archive/stage-d-fidelity-v1-d7-r4-producer"
AUTHORITY_REF = "refs/archive/stage-d-fidelity-v1-d7-r4-authority"
COMPLETION_REF = "refs/archive/stage-d-fidelity-v1-d7-r4-completion"
SMOKE_ROOT = "evidence/stage_d/d7_recovery_v2_20260904/synthetic_smoke"
SMOKE_RECEIPT_PATH = "reports/stage_d_d7_recovery_v2_20260904/smoke_receipt.json"
BUDGET_PATH = "config/stage_d_d7_budget_20260904_r4.json"
HISTORICAL_AUTHORIZATION_PATH = (
    "config/stage_d_d7_execution_authorization_20260904.json"
)
AUTHORIZATION_PATH = "config/stage_d_d7_execution_authorization_20260904_r4.json"
COMPLETION_ROOT = "reports/stage_d_d7_recovery_v2_20260904_r4"
VALIDATION_PATH = f"{COMPLETION_ROOT}/validation.json"
STATUS_PATH = f"{COMPLETION_ROOT}/status.json"
HANDOFF_PATH = f"{COMPLETION_ROOT}/d7_to_future_d8_handoff.json"
FULL_CHAIN_ROOT = (
    "evidence/stage_d/d7_recovery_v2_20260904/full_chain_synthetic_r4"
)
FULL_CHAIN_RECEIPT_PATH = f"{COMPLETION_ROOT}/full_chain_smoke_receipt.json"
ACCESS_SMOKE_ROOT = (
    "evidence/stage_d/d7_recovery_v2_20260904/d8_initializer_synthetic_r4"
)
ACCESS_SMOKE_RECEIPT_PATH = f"{COMPLETION_ROOT}/d8_initializer_smoke_receipt.json"
FORMAL_ROOT = "evidence/stage_d/d8_recovery_v2_20260904/formal"
D8_0_REBIND_ID = "STAGE_D_RECOVERY_V2_D8_0_REBIND_R1_20260904"
D8_0_REBIND_AUTHORIZATION_DOCUMENT_TYPE = (
    "stage_d_d8_0_fresh_root_authorization"
)
D8_0_REBIND_AUTHORIZATION_PATH = (
    "config/stage_d_d8_0_fresh_root_authorization_20260904_r1.json"
)
D8_0_REBIND_AUTHORITY_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-0-rebind-r1-authority"
)
D8_0_REBIND_PRODUCER_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-0-rebind-r1-producer"
)
D8_0_REBIND_FORMAL_ROOT = (
    "evidence/stage_d/d8_recovery_v2_20260904/formal_r2"
)
D8_CONTINUATION_ID = "STAGE_D_RECOVERY_V2_D8_CONTINUATION_R1_20260904"
D8_CONTINUATION_AUTHORIZATION_DOCUMENT_TYPE = (
    "stage_d_d8_continuation_authorization"
)
D8_CONTINUATION_AUTHORIZATION_PATH = (
    "config/stage_d_d8_continuation_authorization_20260904_r1.json"
)
D8_CONTINUATION_AUTHORITY_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r1-authority"
)
D8_CONTINUATION_PRODUCER_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r1-producer"
)
D8_CONTINUATION_R2_ID = "STAGE_D_RECOVERY_V2_D8_CONTINUATION_R2_20260905"
D8_CONTINUATION_R2_AUTHORIZATION_PATH = (
    "config/stage_d_d8_continuation_authorization_20260905_r2.json"
)
D8_CONTINUATION_R2_AUTHORITY_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r2-authority"
)
D8_CONTINUATION_R2_PRODUCER_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r2-producer"
)
D8_CONTINUATION_R3_ID = "STAGE_D_RECOVERY_V2_D8_CONTINUATION_R3_20260905"
D8_CONTINUATION_R3_AUTHORIZATION_PATH = (
    "config/stage_d_d8_continuation_authorization_20260905_r3.json"
)
D8_CONTINUATION_R3_AUTHORITY_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r3-authority"
)
D8_CONTINUATION_R3_PRODUCER_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r3-producer"
)
D8_CONTINUATION_R4_ID = "STAGE_D_RECOVERY_V2_D8_CONTINUATION_R4_20260905"
D8_CONTINUATION_R4_AUTHORIZATION_PATH = (
    "config/stage_d_d8_continuation_authorization_20260905_r4.json"
)
D8_CONTINUATION_R4_AUTHORITY_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r4-authority"
)
D8_CONTINUATION_R4_PRODUCER_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r4-producer"
)
D8_CONTINUATION_R5_ID = "STAGE_D_RECOVERY_V2_D8_CONTINUATION_R5_20260905"
D8_CONTINUATION_R5_AUTHORIZATION_PATH = (
    "config/stage_d_d8_continuation_authorization_20260905_r5.json"
)
D8_CONTINUATION_R5_AUTHORITY_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r5-authority"
)
D8_CONTINUATION_R5_PRODUCER_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r5-producer"
)
D8_CONTINUATION_R6_ID = "STAGE_D_RECOVERY_V2_D8_CONTINUATION_R6_20260905"
D8_CONTINUATION_R6_AUTHORIZATION_PATH = (
    "config/stage_d_d8_continuation_authorization_20260905_r6.json"
)
D8_CONTINUATION_R6_AUTHORITY_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r6-authority"
)
D8_CONTINUATION_R6_PRODUCER_REF = (
    "refs/archive/stage-d-fidelity-v1-d8-continuation-r6-producer"
)
D8_S00_RECEIPT_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s00_shard_receipt.json"
)
D8_S00_COMPLETION_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s00_completion.json"
)
D8_S01_RECEIPT_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s01_shard_receipt.json"
)
D8_S01_COMPLETION_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s01_completion.json"
)
D8_S02_RECEIPT_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s02_shard_receipt.json"
)
D8_S02_COMPLETION_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s02_completion.json"
)
D8_S03_RECEIPT_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s03_shard_receipt.json"
)
D8_S03_COMPLETION_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s03_completion.json"
)
D8_S04_RECEIPT_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s04_shard_receipt.json"
)
D8_S04_COMPLETION_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s04_completion.json"
)
D8_S05_RECEIPT_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s05_shard_receipt.json"
)
D8_S05_COMPLETION_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/d8_s05_completion.json"
)
D8_0_INITIAL_BLOCKED_RECORD_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/"
    "d8_0_initial_blocked_root.json"
)
D8_0_MATERIALIZATION_RECORD_PATH = (
    "reports/stage_d_d8_recovery_v2_20260904_r4/"
    "asset_materialization_verification.json"
)
C9_REPORT_PATH = "evidence/stage_c/c9_recovery_v2_0d88fbb/formal/c9_evidence_report.json"
C9_REPORT_FILE_SHA256 = "ad164a87aa6dedba461e1b19d639d22d955a675d75e0c6e257af4d5846682763"
C9_REPORT_CANONICAL_DIGEST = "959bc874b04b007f487d05734cf5b85b1eaef75bb8c2ff2ded222c7cf9083248"
D6_COMPLETION_COMMIT = "586634ee9919d857cd3409e08b2bf6edfdaa7867"
D6_TERMINAL_DIGEST = "fc9e9f5bd2be749947881615fb6944181a6d3e5b64378aaca8e5592935626637"
RECOVERY_MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
RECOVERY_MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
DEFAULT_SHARD_COUNT = 8

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_SMOKE_SCHEMA = "config/stage_d_d7_smoke.schema.json"
_FULL_CHAIN_SCHEMA = "config/stage_d_d7_full_chain_smoke_r4.schema.json"
_ACCESS_SMOKE_SCHEMA = "config/stage_d_d7_access_smoke_r4.schema.json"
_BUDGET_SCHEMA = "config/stage_d_d7_budget_r4.schema.json"
_AUTH_SCHEMA = "config/stage_d_d7_execution_authorization_r4.schema.json"
_D8_0_REBIND_AUTH_SCHEMA = (
    "config/stage_d_d8_0_fresh_root_authorization.schema.json"
)
_D8_CONTINUATION_AUTH_SCHEMA = (
    "config/stage_d_d8_continuation_authorization.schema.json"
)
_D8_CONTINUATION_R2_AUTH_SCHEMA = (
    "config/stage_d_d8_continuation_authorization_r2.schema.json"
)
_D8_CONTINUATION_R3_AUTH_SCHEMA = (
    "config/stage_d_d8_continuation_authorization_r3.schema.json"
)
_D8_CONTINUATION_R4_AUTH_SCHEMA = (
    "config/stage_d_d8_continuation_authorization_r4.schema.json"
)
_D8_CONTINUATION_R5_AUTH_SCHEMA = (
    "config/stage_d_d8_continuation_authorization_r5.schema.json"
)
_D8_CONTINUATION_R6_AUTH_SCHEMA = (
    "config/stage_d_d8_continuation_authorization_r6.schema.json"
)
_AUTHORITY_ONLY_PATHS = {f"vitnns_rebuild/{AUTHORIZATION_PATH}"}
_D8_0_REBIND_AUTHORITY_ONLY_PATHS = {
    f"vitnns_rebuild/{D8_0_REBIND_AUTHORIZATION_PATH}",
}
_D8_CONTINUATION_AUTHORITY_ONLY_PATHS = {
    f"vitnns_rebuild/{D8_CONTINUATION_AUTHORIZATION_PATH}",
}
_D8_CONTINUATION_R2_AUTHORITY_ONLY_PATHS = {
    f"vitnns_rebuild/{D8_CONTINUATION_R2_AUTHORIZATION_PATH}",
}
_D8_CONTINUATION_R3_AUTHORITY_ONLY_PATHS = {
    f"vitnns_rebuild/{D8_CONTINUATION_R3_AUTHORIZATION_PATH}",
}
_D8_CONTINUATION_R4_AUTHORITY_ONLY_PATHS = {
    f"vitnns_rebuild/{D8_CONTINUATION_R4_AUTHORIZATION_PATH}",
}
_D8_CONTINUATION_R5_AUTHORITY_ONLY_PATHS = {
    f"vitnns_rebuild/{D8_CONTINUATION_R5_AUTHORIZATION_PATH}",
}
_D8_CONTINUATION_R6_AUTHORITY_ONLY_PATHS = {
    f"vitnns_rebuild/{D8_CONTINUATION_R6_AUTHORIZATION_PATH}",
}


def _plain(value: Any, context: str = "value") -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter) and not isinstance(value, Mapping):
        value = converter()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageDD7Error(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise StageDD7Error(f"{context} has an invalid key")
            result[key] = _plain(child, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain(child, f"{context}[{index}]") for index, child in enumerate(value)]
    raise StageDD7Error(f"{context} is not JSON-compatible")


def _semantic_digest(document: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: child for key, child in document.items() if key != "canonical_digest"
    })


def _file_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain(document), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"


def _read_regular_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise StageDD7Error("safe no-follow reads are unavailable")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size > _MAX_JSON_BYTES
        ):
            raise StageDD7Error(f"{context} must be a bounded single-link regular file")
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
        )
        if len(raw) != before.st_size or identity(before) != identity(after):
            raise StageDD7Error(f"{context} changed while being read")
    except StageDD7Error:
        raise
    except OSError as error:
        raise StageDD7Error(f"cannot read {context}: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        value = strict_json_bytes(raw, context)
    except Exception as error:
        raise StageDD7Error(f"invalid {context}") from error
    if not isinstance(value, dict):
        raise StageDD7Error(f"{context} must be an object")
    return value, raw


def _write_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _file_bytes(document)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _git(repo: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment.update({
        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
    })
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=repo, check=True, capture_output=True,
            text=True, timeout=60, env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise StageDD7Error(f"Git check failed: {' '.join(arguments)}") from error
    return result.stdout.strip()


def _git_success(repo: Path, *arguments: str) -> bool:
    environment = os.environ.copy()
    environment.update({
        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
    })
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=repo, check=False, capture_output=True,
            timeout=60, env=environment,
        )
    except OSError as error:
        raise StageDD7Error(f"Git check failed: {' '.join(arguments)}") from error
    return result.returncode == 0


def _schema_validate(document: Mapping[str, Any], path: Path, context: str) -> None:
    schema, _ = _read_regular_json(path, f"{context} schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "root"
        raise StageDD7Error(f"{context} schema failure at {location}: {errors[0].message}")


def validate_archived_recovery_chain(
    *, repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    """Recompute D0--D6 from each immutable completion snapshot."""

    repo = Path(repo_root).resolve()
    prefix = _git(repo, "rev-parse", "--show-prefix")
    records = []
    previous: str | None = None
    expected_scopes = {
        "D0": "FORMAL_PREPARATION_PASS",
        "D1": "FORMAL_PREPARATION_PASS",
        "D2": "FORMAL_PREPARATION_PASS",
        "D3": "CONTROL_PLANE_READINESS_PASS",
        "D4": "PRODUCER_IMPLEMENTATION_READINESS_PASS",
        "D5": "VALIDATION_IMPLEMENTATION_READINESS_PASS",
        "D6": "ANALYSIS_IMPLEMENTATION_READINESS_PASS",
    }
    for stage in expected_scopes:
        lower = stage.lower()
        ref = f"refs/archive/stage-d-recovery-v2-r3-{lower}-completion"
        commit = _git(repo, "rev-parse", f"{ref}^{{commit}}")
        if previous is not None and not _git_success(
            repo, "merge-base", "--is-ancestor", previous, commit
        ):
            raise StageDD7Error(f"{stage} does not descend from direct upstream")
        base = f"reports/stage_d_recovery_v2_20260904_r3/{lower}"

        def blob(relative: str) -> bytes:
            try:
                return subprocess.run(
                    ["git", "show", f"{commit}:{prefix}{relative}"],
                    cwd=repo, check=True, capture_output=True, timeout=30,
                    env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_LFS_SKIP_SMUDGE": "1"},
                ).stdout
            except (OSError, subprocess.SubprocessError) as error:
                raise StageDD7Error(f"cannot read archived {stage} source: {relative}") from error

        completion_raw = blob(f"{base}/completion.json")
        validation_raw = blob(f"{base}/validation.json")
        status_raw = blob(f"{base}/status.json")
        completion = strict_json_bytes(completion_raw, f"archived {stage} completion")
        validation = strict_json_bytes(validation_raw, f"archived {stage} validation")
        status = strict_json_bytes(status_raw, f"archived {stage} status")
        if any(not isinstance(item, Mapping) for item in (completion, validation, status)):
            raise StageDD7Error(f"archived {stage} record is not an object")
        for document, label in ((completion, "completion"), (validation, "validation"), (status, "status")):
            if document.get("canonical_digest") != _semantic_digest(document):
                raise StageDD7Error(f"archived {stage} {label} digest mismatch")
        if (
            status.get("status") != "PASS"
            or status.get("pass_scope") != expected_scopes[stage]
            or status.get("completion_file_sha256") != hashlib.sha256(completion_raw).hexdigest()
            or status.get("validation_file_sha256") != hashlib.sha256(validation_raw).hexdigest()
            or status.get("validation_complete_before_terminal") is not True
            or set(status.get("authorization", {}).values()) != {False}
            or set(status.get("access_attestation", {}).values()) != {False}
            or validation.get("source_recomputed") is not True
            or validation.get("errors") != []
            or validation.get("blocking_reasons") != []
        ):
            raise StageDD7Error(f"archived {stage} terminal semantics drifted")
        producer = completion["git"]["producer_commit"]
        for source in completion["source_files"]:
            try:
                source_raw = subprocess.run(
                    ["git", "show", f"{producer}:{prefix}{source['path']}"],
                    cwd=repo, check=True, capture_output=True, timeout=30,
                    env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_LFS_SKIP_SMUDGE": "1"},
                ).stdout
            except (OSError, subprocess.SubprocessError) as error:
                raise StageDD7Error(
                    f"cannot read archived {stage} producer source: {source['path']}"
                ) from error
            if hashlib.sha256(source_raw).hexdigest() != source["file_sha256"]:
                raise StageDD7Error(f"archived {stage} producer source hash drifted")
        records.append({
            "stage": stage, "commit": commit,
            "terminal_file_sha256": hashlib.sha256(status_raw).hexdigest(),
            "terminal_canonical_digest": status["canonical_digest"],
            "producer_source_count": len(completion["source_files"]),
        })
        previous = commit
    return {
        "status": "PASS", "source_recomputed_from_archive_blobs": True,
        "stages": records, "canonical_digest": canonical_sha256(records),
    }


@dataclass(frozen=True, slots=True)
class RecoveryStageDPipeline:
    contracts: Any
    manifest: Any
    plan: StageDExecutionPlan


def load_recovery_pipeline(*, repo_root: Path | str = REPO_ROOT) -> RecoveryStageDPipeline:
    """Build the frozen plan through the R3 overlay, never the stale D3 source loader."""

    repo = Path(repo_root).resolve()
    archived = validate_archived_recovery_chain(repo_root=repo)
    d6_ref = "refs/archive/stage-d-recovery-v2-r3-d6-completion"
    if _git(repo, "rev-parse", f"{d6_ref}^{{commit}}") != D6_COMPLETION_COMMIT:
        raise StageDD7Error("D6 recovery-v2 direct upstream is not current")
    prefix = _git(repo, "rev-parse", "--show-prefix")
    d6_relative = "reports/stage_d_recovery_v2_20260904_r3/d6/status.json"
    try:
        d6_raw = subprocess.run(
            ["git", "show", f"{D6_COMPLETION_COMMIT}:{prefix}{d6_relative}"],
            cwd=repo, check=True, capture_output=True, timeout=30,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_LFS_SKIP_SMUDGE": "1"},
        ).stdout
        d6 = strict_json_bytes(d6_raw, "archived D6 terminal record")
    except (OSError, subprocess.SubprocessError) as error:
        raise StageDD7Error("cannot read archived D6 terminal record") from error
    if (
        not isinstance(d6, Mapping) or d6.get("status") != "PASS"
        or d6.get("pass_scope") != "ANALYSIS_IMPLEMENTATION_READINESS_PASS"
        or d6.get("canonical_digest") != D6_TERMINAL_DIGEST
        or _semantic_digest(d6) != D6_TERMINAL_DIGEST
        or d6.get("authorization", {}).get("execution_authorization") is not False
        or d6.get("access_attestation", {}).get("reserve_image_opened") is not False
    ):
        raise StageDD7Error("archived D6 terminal record is invalid")
    if not _git_success(repo, "merge-base", "--is-ancestor", D6_COMPLETION_COMMIT, "HEAD"):
        raise StageDD7Error("current HEAD does not descend from D6 completion")
    recovery, _file_sha, _digest = load_recovery_binding(repo)
    if recovery["model"]["file_sha256"] != RECOVERY_MODEL_SHA256:
        raise StageDD7Error("recovery-v2 model overlay drifted")
    contracts = load_stage_d_contracts(repo_root=repo)
    registry, _ = _read_regular_json(
        repo / "manifests/stage_d_fidelity_v1/registry.json",
        "tracked Stage D registry metadata",
    )
    plan = validate_stage_d_plan(build_stage_d_plan(
        contracts.config, contracts.run_spec, registry,
    ))
    manifest = SimpleNamespace(
        registry_document=registry,
        bindings=MappingProxyType({
            "d6": MappingProxyType({
                "verification_status": "PASS",
                "completion_commit": D6_COMPLETION_COMMIT,
                "terminal_canonical_digest": D6_TERMINAL_DIGEST,
                "archive_chain_digest": archived["canonical_digest"],
            }),
            "recovery_v2": MappingProxyType({
                "verification_status": "PASS",
                "canonical_digest": recovery["canonical_digest"],
            }),
        }),
        canonical_digest=canonical_sha256({
            "d6": D6_TERMINAL_DIGEST,
            "recovery_v2": recovery["canonical_digest"],
            "registry": registry["canonical_digest"],
        }),
    )
    return RecoveryStageDPipeline(contracts, manifest, plan)


class _SmokeBackend:
    def __init__(self) -> None:
        self.active: str | None = None
        self.release_count = 0
        self.method_order: list[str] = []

    @contextmanager
    def image_scope(self, sample_id: str, *, device: str):
        if device != "cpu" or self.active is not None:
            raise StageDD7Error("D7 smoke backend scope is invalid")
        self.active = sample_id
        try:
            yield self
        finally:
            self.active = None
            self.release_count += 1

    def run_stage_d_unit(self, group: Any, unit: Any, *, device: str, call_counter: Any) -> Any:
        del group, device
        from .stage_d_artifacts import StageDUnitPayload

        self.method_order.append(unit.method_id)
        call_counter.record_forward(1)
        if unit.method_id == NNS_METHOD_ID:
            call_counter.record_backward(1)
        base = 1.0 if unit.method_id == NNS_METHOD_ID else 0.5
        metric = np.full((56, 56), base, dtype=np.float32)
        curves = np.empty((2, 5, 21), dtype=np.float64)
        for direction in range(2):
            for repeat in range(5):
                curves[direction, repeat] = np.linspace(
                    base + direction * 0.01 + repeat * 0.001,
                    base + 1.0 + direction * 0.01 + repeat * 0.001, 21,
                )
        audit = (
            {
                "raw_attribution": np.array([base], dtype=np.float32),
                "repeat_clean_metric_ready_map": metric.copy(),
            }
            if unit.audit else None
        )
        return StageDUnitPayload(
            l0_scalars={"endpoint_parity": 0.0},
            metric_ready_map=metric, fidelity_curves=curves,
            audit_evidence=audit,
        )


def _inventory(root: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    total = 0
    for directory, names, files in os.walk(root, followlinks=False):
        names.sort()
        files.sort()
        current = Path(directory)
        for name in files:
            path = current / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageDD7Error("smoke root contains a non-ordinary file")
            relative = path.relative_to(root).as_posix()
            raw = path.read_bytes()
            records.append({
                "path": relative, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
            total += len(raw)
    return records, total


def _validate_smoke_pair(
    root: Path, group: Any, *, plan_digest: str,
) -> tuple[list[dict[str, Any]], str]:
    """Source-recompute the two synthetic attempts and their numeric shape."""

    from .stage_d_artifacts import validate_stage_d_attempt

    rows: list[dict[str, Any]] = []
    for unit in group.units:
        attempt = root / "units" / unit.canonical_digest / "attempts/attempt-000001"
        validation = validate_stage_d_attempt(
            attempt, expected_unit=unit, expected_group=group,
            expected_plan_digest=plan_digest,
            expected_mode="SYNTHETIC_D4_TEST_ONLY",
        )
        if not validation.valid or not validation.terminal or not validation.complete:
            raise StageDD7Error("synthetic smoke attempt source validation failed")
        map_value = np.load(attempt / "L1/metric_ready_map.npy", allow_pickle=False)
        with np.load(attempt / "L1/fidelity_curves.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"curves"}:
                raise StageDD7Error("synthetic smoke curve archive is not closed")
            curves = archive["curves"]
        if (
            map_value.dtype != np.float32 or map_value.shape != (56, 56)
            or curves.dtype != np.float64 or curves.shape != (2, 5, 21)
            or not np.isfinite(map_value).all() or not np.isfinite(curves).all()
        ):
            raise StageDD7Error("synthetic smoke numeric artifact shape drifted")
        rows.append({
            "unit_digest": unit.canonical_digest, "method_id": unit.method_id,
            "attempt_digest": validation.digest,
            "manifest_sha256": validation.manifest_sha256,
            "artifact_count": validation.artifact_count,
            "map_sha256": hashlib.sha256(map_value.tobytes(order="C")).hexdigest(),
            "curve_sha256": hashlib.sha256(curves.tobytes(order="C")).hexdigest(),
        })
    if [row["method_id"] for row in rows] != list(METHOD_IDS):
        raise StageDD7Error("synthetic smoke pair order drifted")
    return rows, canonical_sha256(rows)


def run_synthetic_smoke(
    *, repo_root: Path | str = REPO_ROOT, output_root: Path | str | None = None,
    receipt_path: Path | str | None = None,
) -> Mapping[str, Any]:
    """Run a fresh two-method, 422-coordinate diagnostic round trip."""

    from .stage_d_artifacts import validate_stage_d_attempt
    from .stage_d_producer import run_stage_d_execution_unit, run_stage_d_image_group
    from .stage_d_runtime import _issue_synthetic_d4_test_context

    repo = Path(repo_root).resolve()
    root = repo / SMOKE_ROOT if output_root is None else Path(output_root).absolute()
    receipt_file = (
        repo / SMOKE_RECEIPT_PATH if receipt_path is None else Path(receipt_path).absolute()
    )
    if os.path.lexists(root):
        raise StageDD7Error("D7 smoke root must be fresh")
    pipeline = load_recovery_pipeline(repo_root=repo)
    plan = pipeline.plan
    estimate = estimate_stage_d_run(pipeline.contracts.config, plan)
    preflight = preflight_stage_d(estimate)
    context, gate = _issue_synthetic_d4_test_context(
        plan, estimate, preflight, shard_count=DEFAULT_SHARD_COUNT,
    )
    # Exercise the gate before creating the diagnostic root.
    checked = gate.validate_execution_context(
        context, operation="smoke", output_root=root, plan=plan,
        estimate=estimate, preflight=preflight,
    )
    if checked is not context:
        raise StageDD7Error("synthetic smoke gate changed its opaque context")
    root.mkdir(parents=True, mode=0o700)
    group = next(item for item in plan.groups if not item.audit)
    backend = _SmokeBackend()
    owner = "stage-d-d7-synthetic-smoke"
    owner_digest = hashlib.sha256(owner.encode("utf-8")).hexdigest()

    def run_one(unit: Any, *, backend: Any, device: str) -> Mapping[str, Any]:
        claim = {
            "owner_token": owner, "owner_token_digest": owner_digest,
            "group_owner_token_digest": owner_digest,
            "unit_digest": unit.canonical_digest,
            "group_digest": group.canonical_digest,
            "attempt_id": "attempt-000001", "attempt_number": 1,
            "attempt_path": str(
                root / "units" / unit.canonical_digest / "attempts"
                / "attempt-000001"
            ),
        }
        result = run_stage_d_execution_unit(
            unit, group=group, plan=plan, output_root=root, backend=backend,
            context=context, claim=claim, device=device,
        )
        return {
            **result,
            "state": {
                "succeeded": "TERMINAL_SUCCESS",
                "scientific_failed": "TERMINAL_SCIENTIFIC_FAILURE",
                "infrastructure_failed": "RETRY_PENDING",
                "blocked": "BLOCKED",
            }[result["state"]],
        }

    result = run_stage_d_image_group(
        group, backend=backend, run_unit=run_one, device="cpu",
    )
    if (
        result.state != "TERMINAL_SUCCESS" or not result.cache_released
        or backend.release_count != 1
        or backend.method_order != list(METHOD_IDS)
    ):
        raise StageDD7Error("synthetic smoke producer/cache gate failed")
    attempts = []
    for unit in group.units:
        attempt = root / "units" / unit.canonical_digest / "attempts/attempt-000001"
        validation = validate_stage_d_attempt(
            attempt, expected_unit=unit, expected_group=group,
            expected_plan_digest=plan.canonical_digest,
            expected_mode="SYNTHETIC_D4_TEST_ONLY",
        )
        if not validation.valid or not validation.terminal or not validation.complete:
            raise StageDD7Error("synthetic smoke attempt source validation failed")
        attempts.append({
            "unit_digest": unit.canonical_digest,
            "method_id": unit.method_id, "attempt_digest": validation.digest,
            "manifest_sha256": validation.manifest_sha256,
            "artifact_count": validation.artifact_count,
        })
    # Run a second, independent source pass that decodes only the synthetic
    # NPY/NPZ products.  This is the D7 mini-finalize/aggregate boundary.
    aggregate_rows, aggregate_digest = _validate_smoke_pair(
        root, group, plan_digest=plan.canonical_digest,
    )
    spec = {
        "schema_version": 1, "document_type": "stage_d_d7_smoke_spec",
        "formal": False, "input_kind": "DETERMINISTIC_SYNTHETIC_ARRAYS",
        "sample_count": 1, "unit_count": 2, "map_count": 2,
        "curve_count": 20, "curve_point_count": 420,
        "logical_work_item_count": 422,
        "audit_repeat_clean_included": False,
        "methods": list(METHOD_IDS), "fine_only": True,
        "reserve_image_accessed": False, "model_loaded": False,
        "gpu_or_cuda_used": False, "formal_evidence": False,
    }
    spec["canonical_digest"] = _semantic_digest(spec)
    finalization = {
        "schema_version": 1, "document_type": "stage_d_d7_smoke_finalization",
        "status": "PASS", "source_recomputed": True,
        "terminal_units": 2, "succeeded_units": 2,
        "attempts": attempts, "formal_evidence": False,
        "independent_revalidation": True,
    }
    finalization["canonical_digest"] = _semantic_digest(finalization)
    aggregate = {
        "schema_version": 1, "document_type": "stage_d_d7_smoke_aggregation",
        "status": "PASS", "diagnostic_only": True,
        "methods_present": list(METHOD_IDS), "paired_sample_count": 1,
        "source_rows_digest": aggregate_digest,
        "source_rows": aggregate_rows,
        "formal_effect_result": False, "scientific_decision": "NOT_EVALUATED",
    }
    aggregate["canonical_digest"] = _semantic_digest(aggregate)
    paper = {
        "schema_version": 1, "document_type": "stage_d_d7_smoke_paper_package",
        "status": "PASS", "diagnostic_only": True,
        "source_aggregation_digest": aggregate["canonical_digest"],
        "formal_paper_data_generated": False, "formal_figure_generated": False,
    }
    paper["canonical_digest"] = _semantic_digest(paper)
    for name, document in (
        ("smoke_spec.json", spec), ("finalization.json", finalization),
        ("aggregation.json", aggregate), ("paper_package.json", paper),
    ):
        _write_atomic(root / name, document)
    records, total = _inventory(root)
    receipt = {
        "$schema": "../../config/stage_d_d7_smoke.schema.json",
        "schema_version": 1, "document_type": SMOKE_DOCUMENT_TYPE,
        "d7_id": LEGACY_D7_ID, "status": "PASS", "formal": False,
        "input_kind": "DETERMINISTIC_SYNTHETIC_ARRAYS",
        "root": (
            str(root.relative_to(repo)) if root.is_relative_to(repo) else str(root)
        ),
        "plan_digest": plan.canonical_digest,
        "counts": {
            "sample_count": 1, "unit_count": 2, "map_count": 2,
            "curve_count": 20, "curve_point_count": 420,
            "logical_work_item_count": 422,
        },
        "chain": {
            "producer": "PASS", "artifact": "PASS",
            "finalize": "PASS", "source_validate": "PASS",
            "aggregate": "PASS", "paper_export": "PASS",
            "cache_release": "PASS",
        },
        "root_inventory": {
            "file_count": len(records), "total_bytes": total,
            "canonical_digest": canonical_sha256(records),
        },
        "access_attestation": {
            "reserve_path_exists_checked": False,
            "reserve_image_statted": False, "reserve_image_opened": False,
            "reserve_image_hashed": False, "reserve_image_decoded": False,
            "reserve_image_preprocessed": False,
            "reserve_image_used_for_inference": False,
            "model_loaded": False, "gpu_or_cuda_used": False,
            "stage_d_outcome_accessed": False,
        },
        "authority": {
            "execution_authorization": "NOT_ISSUED",
            "analysis_authorization": ANALYSIS_AUTHORIZATION,
            "stage_d_execution_started": False,
        },
    }
    receipt["canonical_digest"] = _semantic_digest(receipt)
    _schema_validate(receipt, repo / _SMOKE_SCHEMA, "D7 smoke receipt")
    _write_atomic(receipt_file, receipt)
    return receipt


def validate_smoke(
    *, repo_root: Path | str = REPO_ROOT, receipt_path: Path | str | None = None,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    path = repo / SMOKE_RECEIPT_PATH if receipt_path is None else Path(receipt_path)
    receipt, _ = _read_regular_json(path, "D7 smoke receipt")
    _schema_validate(receipt, repo / _SMOKE_SCHEMA, "D7 smoke receipt")
    if receipt.get("canonical_digest") != _semantic_digest(receipt):
        raise StageDD7Error("D7 smoke receipt canonical digest mismatch")
    if receipt.get("status") != "PASS" or receipt.get("formal") is not False:
        raise StageDD7Error("D7 smoke did not pass as a non-formal diagnostic")
    root_value = Path(receipt["root"])
    root = root_value if root_value.is_absolute() else repo / root_value
    records, total = _inventory(root)
    expected = {
        "file_count": len(records), "total_bytes": total,
        "canonical_digest": canonical_sha256(records),
    }
    if receipt.get("root_inventory") != expected:
        raise StageDD7Error("D7 smoke root inventory drifted")
    if set(receipt["access_attestation"].values()) != {False}:
        raise StageDD7Error("D7 smoke access boundary was crossed")
    spec, _ = _read_regular_json(root / "smoke_spec.json", "D7 smoke spec")
    finalization, _ = _read_regular_json(
        root / "finalization.json", "D7 smoke finalization",
    )
    aggregate, _ = _read_regular_json(
        root / "aggregation.json", "D7 smoke aggregation",
    )
    paper, _ = _read_regular_json(
        root / "paper_package.json", "D7 smoke paper package",
    )
    for document, label in (
        (spec, "spec"), (finalization, "finalization"),
        (aggregate, "aggregation"), (paper, "paper package"),
    ):
        if document.get("canonical_digest") != _semantic_digest(document):
            raise StageDD7Error(f"D7 smoke {label} digest mismatch")
    if (
        spec.get("logical_work_item_count") != 422
        or finalization.get("independent_revalidation") is not True
        or aggregate.get("scientific_decision") != "NOT_EVALUATED"
        or paper.get("formal_paper_data_generated") is not False
        or paper.get("source_aggregation_digest") != aggregate["canonical_digest"]
    ):
        raise StageDD7Error("D7 smoke chain semantics drifted")
    pipeline = load_recovery_pipeline(repo_root=repo)
    group = next(item for item in pipeline.plan.groups if not item.audit)
    rows, digest = _validate_smoke_pair(
        root, group, plan_digest=pipeline.plan.canonical_digest,
    )
    if (
        aggregate.get("source_rows") != rows
        or aggregate.get("source_rows_digest") != digest
    ):
        raise StageDD7Error("D7 smoke independent source rows drifted")
    return receipt


def record_failed_smoke(
    failed_root: Path | str, *, reason: str, repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    """Write a non-PASS index for a retained D7 development smoke failure."""

    repo = Path(repo_root).resolve()
    root = Path(failed_root).absolute()
    records, total = _inventory(root)
    document = {
        "schema_version": 1,
        "document_type": "stage_d_d7_failed_smoke_record",
        "d7_id": LEGACY_D7_ID, "status": "FAILED_RETAINED",
        "root": str(root.relative_to(repo)), "reason": reason,
        "root_inventory": {
            "file_count": len(records), "total_bytes": total,
            "canonical_digest": canonical_sha256(records),
        },
        "terminal_pass_generated": False, "formal_evidence": False,
        "reserve_image_accessed": False, "stage_d_execution_started": False,
    }
    document["canonical_digest"] = _semantic_digest(document)
    _write_atomic(root / "failure_record.json", document)
    return document


def _run_full_chain_shard(
    root: Path, plan: StageDExecutionPlan, estimate: StageDEstimate,
    preflight: StageDPreflight, context: Any, gate: Any, backend: _SmokeBackend,
) -> Any:
    """Run the real D4 shard with only checkpoint durability batching changed."""

    from . import stage_d_runtime as runtime

    original_validate = runtime._validate_checkpoint
    original_read = runtime._read_valid_checkpoint_locked
    original_replace = runtime._replace_checkpoint
    initial, _ = _read_regular_json(
        root / runtime.RUNTIME_CHECKPOINT_FILENAME, "synthetic runtime checkpoint",
    )
    binding = initial["execution_context"]
    current = original_read(root, plan, estimate, preflight, binding)
    published = False

    def fast_read(
        checked_root: Path, checked_plan: Any, checked_estimate: Any,
        checked_preflight: Any, checked_binding: Any,
        *, transitioning_unit_digest: str | None = None,
    ) -> Any:
        if checked_root != root:
            return original_read(
                checked_root, checked_plan, checked_estimate, checked_preflight,
                checked_binding,
                transitioning_unit_digest=transitioning_unit_digest,
            )
        if checked_binding != binding:
            raise StageDD7Error("synthetic checkpoint binding drifted")
        return current

    def validate_once(document: Any, **kwargs: Any) -> Any:
        if kwargs.get("root") != root:
            return original_validate(document, **kwargs)
        return document

    def fast_replace(checked_root: Path, previous: Any, changed: Any, **kwargs: Any) -> Any:
        nonlocal current, published
        if checked_root != root:
            return original_replace(checked_root, previous, changed, **kwargs)
        if previous is not current or changed is not current:
            raise StageDD7Error("synthetic checkpoint ownership drifted")
        prior = previous["canonical_digest"]
        changed["generation"] = previous["generation"] + 1
        changed["previous_checkpoint_digest"] = prior
        changed["canonical_digest"] = canonical_sha256({
            "generation": changed["generation"],
            "previous_checkpoint_digest": prior,
        })
        current = changed
        released = (
            all(row["state"] != "leased" for row in changed["shards"])
            and all(row["state"] != "claimed" for row in changed["groups"])
            and all(row["state"] != "claimed" for row in changed["units"])
        )
        if released:
            changed["failure_rows"] = runtime._failure_rows(changed["units"])
            changed["resources"] = runtime._sum_resources(changed["units"])
            lifecycle = changed["lifecycle"]
            changed["lifecycle"] = runtime._lifecycle_projection(
                lifecycle["state"], changed["shards"], changed["groups"],
                changed["units"], lifecycle["blocked_reasons"],
            )
            changed.pop("canonical_digest", None)
            changed["canonical_digest"] = canonical_sha256(changed)
            runtime._atomic_replace_checkpoint(
                root / runtime.RUNTIME_CHECKPOINT_FILENAME, changed,
            )
            published = True
        return current

    runtime._validate_checkpoint = validate_once
    runtime._read_valid_checkpoint_locked = fast_read
    runtime._replace_checkpoint = fast_replace
    try:
        summary = runtime.run_stage_d_shard(
            plan, output_root=root, estimate=estimate, preflight=preflight,
            shard_id=0, execution_context=context, execution_gate=gate,
            backend_factory=lambda **_kwargs: backend,
        )
    finally:
        runtime._validate_checkpoint = original_validate
        runtime._read_valid_checkpoint_locked = original_read
        runtime._replace_checkpoint = original_replace
    if not published:
        raise StageDD7Error("synthetic full-chain checkpoint was not published")
    original_read(root, plan, estimate, preflight, binding)
    return summary


def run_full_chain_synthetic_smoke(
    *, repo_root: Path | str = REPO_ROOT, output_root: Path | str | None = None,
    receipt_path: Path | str | None = None,
) -> Mapping[str, Any]:
    """Run the real D4/D5/D6 chain over a complete synthetic cohort."""

    from .stage_d_analysis import aggregate_stage_d_analysis, closed_analysis_summary
    from .stage_d_analysis_input import validate_stage_d_analysis_input
    from .stage_d_finalize import finalize_stage_d_root, validate_stage_d_root
    from .stage_d_paper import (
        build_stage_d_analysis_package, build_stage_d_paper_package,
        validate_stage_d_analysis_package, validate_stage_d_paper_package,
    )
    from .stage_d_runtime import (
        _issue_synthetic_d4_test_context, initialize_stage_d_output_root,
    )

    repo = Path(repo_root).resolve()
    base = repo / FULL_CHAIN_ROOT if output_root is None else Path(output_root).absolute()
    receipt_file = (
        repo / FULL_CHAIN_RECEIPT_PATH
        if receipt_path is None else Path(receipt_path).absolute()
    )
    if os.path.lexists(base):
        raise StageDD7Error("D7 full-chain smoke root must be fresh")
    base.mkdir(parents=True, mode=0o700)
    execution = base / "execution"
    analysis = base / "analysis"
    paper = base / "paper"
    pipeline = load_recovery_pipeline(repo_root=repo)
    plan = pipeline.plan
    estimate = estimate_stage_d_run(pipeline.contracts.config, plan)
    preflight = preflight_stage_d(estimate)
    context, gate = _issue_synthetic_d4_test_context(
        plan, estimate, preflight, shard_count=1,
    )
    initialize_stage_d_output_root(
        execution, plan, estimate, preflight, execution_context=context,
        execution_gate=gate,
    )
    backend = _SmokeBackend()
    summary = _run_full_chain_shard(
        execution, plan, estimate, preflight, context, gate, backend,
    )
    if (
        summary.state != "completed"
        or summary.counts["terminal_groups"] != 500
        or summary.counts["terminal_units"] != 1000
        or backend.release_count != 500
        or backend.method_order != list(METHOD_IDS) * 500
    ):
        raise StageDD7Error("D7 full-chain D4 execution did not complete")
    finalized = finalize_stage_d_root(
        execution, plan, estimate, preflight, allow_formal=False,
    )
    validated = validate_stage_d_root(
        execution, plan, estimate, preflight, allow_formal=False,
    )
    receipt = validate_stage_d_analysis_input(execution, plan, estimate, preflight)
    result = aggregate_stage_d_analysis(receipt, plan)
    analysis_summary = build_stage_d_analysis_package(
        result, analysis, repo_root=repo,
    )
    analysis_validation = validate_stage_d_analysis_package(analysis, repo_root=repo)
    paper_summary = build_stage_d_paper_package(analysis, paper, repo_root=repo)
    paper_validation = validate_stage_d_paper_package(paper, repo_root=repo)
    closed = closed_analysis_summary(result)
    inventory, total = _inventory(base)
    document = {
        "$schema": "../../config/stage_d_d7_full_chain_smoke_r4.schema.json",
        "schema_version": 1, "document_type": "stage_d_d7_full_chain_smoke_receipt",
        "d7_id": D7_ID, "status": "PASS", "formal": False,
        "input_kind": "COMPLETE_500_GROUP_DETERMINISTIC_SYNTHETIC_FIXTURE",
        "root": str(base.relative_to(repo)) if base.is_relative_to(repo) else str(base),
        "plan_digest": plan.canonical_digest,
        "counts": {
            "image_groups": 500, "execution_units": 1000,
            "terminal_groups": summary.counts["terminal_groups"],
            "terminal_units": summary.counts["terminal_units"],
            "cache_releases": backend.release_count,
            "analysis_coordinates": closed["coordinate_count"],
        },
        "real_api_chain": {
            "initialize_stage_d_output_root": "PASS",
            "run_stage_d_shard": "PASS",
            "finalize_stage_d_root": finalized.workflow_status,
            "validate_stage_d_root": validated.workflow_status,
            "validate_stage_d_analysis_input": "PASS",
            "aggregate_stage_d_analysis": closed["status"],
            "build_stage_d_analysis_package": "PASS" if analysis_summary["complete"] else "FAIL",
            "validate_stage_d_analysis_package": "PASS" if analysis_validation["complete"] else "FAIL",
            "build_stage_d_paper_package": "PASS" if paper_summary["complete"] else "FAIL",
            "validate_stage_d_paper_package": "PASS" if paper_validation["complete"] else "FAIL",
        },
        "diagnostic_semantics": {
            "checkpoint_durability_batching_used": True,
            "attempt_and_final_package_source_recomputed": True,
            "synthetic_scientific_decision_is_not_a_formal_result": True,
            "excluded_from_d8_d9_d10": True,
        },
        "root_inventory": {
            "file_count": len(inventory), "total_bytes": total,
            "canonical_digest": canonical_sha256(inventory),
        },
        "access_attestation": {
            "reserve_image_exists_checked": False,
            "reserve_image_statted": False, "reserve_image_opened": False,
            "reserve_image_hashed": False, "reserve_image_decoded": False,
            "reserve_image_preprocessed": False,
            "reserve_image_used_for_inference": False,
            "model_loaded": False, "gpu_or_cuda_used": False,
            "stage_d_outcome_accessed": False,
        },
        "authority": {
            "execution_authorization": "NOT_ISSUED",
            "analysis_authorization": ANALYSIS_AUTHORIZATION,
            "stage_d_execution_started": False,
        },
    }
    if set(document["real_api_chain"].values()) != {"PASS"}:
        raise StageDD7Error("D7 full-chain synthetic API did not pass")
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(document, repo / _FULL_CHAIN_SCHEMA, "D7 full-chain smoke")
    _write_atomic(receipt_file, document)
    return document


def _synthetic_asset_contracts(pipeline: RecoveryStageDPipeline) -> list[dict[str, Any]]:
    return [_plain(row) for row in pipeline.manifest.registry_document["rows"]]


def _synthetic_access_observation(
    group: Any, contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload_digest = hashlib.sha256(
        ("stage-d-d8-initializer-synthetic|" + group.sample_id).encode("utf-8")
    ).hexdigest()
    return {
        "source_kind": "SYNTHETIC_D7_R4_FILESYSTEM_FIXTURE",
        "source_payload_sha256": payload_digest,
        "image_sha256": payload_digest,
        "size_bytes": contract["size_bytes"],
        "width": contract["width"], "height": contract["height"],
        "mode": contract["mode"], "image_format": contract["image_format"],
        "official_ilsvrc2012_class_id": contract["official_ilsvrc2012_class_id"],
        "official_wnid": contract["official_wnid"],
        "imagenet1k_index": contract["imagenet1k_index"],
        "label_source": contract["label_source"],
    }


def run_d8_initializer_synthetic_smoke(
    *, repo_root: Path | str = REPO_ROOT, output_root: Path | str | None = None,
    receipt_path: Path | str | None = None,
) -> Mapping[str, Any]:
    """Exercise the frozen D8.0 implementation without resolving live assets."""

    from . import stage_d_runtime as runtime

    repo = Path(repo_root).resolve()
    root = repo / ACCESS_SMOKE_ROOT if output_root is None else Path(output_root).absolute()
    receipt_file = (
        repo / ACCESS_SMOKE_RECEIPT_PATH
        if receipt_path is None else Path(receipt_path).absolute()
    )
    if os.path.lexists(root):
        raise StageDD7Error("D7-R4 initializer smoke root must be fresh")
    pipeline = load_recovery_pipeline(repo_root=repo)
    estimate = estimate_stage_d_run(pipeline.contracts.config, pipeline.plan)
    preflight = preflight_stage_d(estimate)
    context, gate = runtime._issue_synthetic_d4_test_context(
        pipeline.plan, estimate, preflight, shard_count=DEFAULT_SHARD_COUNT,
    )
    contracts = _synthetic_asset_contracts(pipeline)
    calls: list[int] = []

    def validator(group: Any, contract: Mapping[str, Any]) -> Mapping[str, Any]:
        access_root = root / "access"
        lifecycle, _ = _read_regular_json(
            access_root / runtime.ACCESS_LIFECYCLE_FILENAME,
            "synthetic access lifecycle before asset callback",
        )
        journal = sorted((access_root / runtime.ACCESS_JOURNAL_DIRECTORY).iterdir())
        if (
            lifecycle["state"] != "ACCESS_STARTED"
            or lifecycle["journal_entry_count"] != len(journal)
            or lifecycle["journal_head_digest"]
            != _read_regular_json(journal[-1], "synthetic access intent")[0]["canonical_digest"]
            or _read_regular_json(journal[-1], "synthetic access intent")[0]["event"]
            not in {"ASSET_ACCESS_INTENT", "ASSET_ACCESS_RESUMED"}
        ):
            raise StageDD7Error("asset callback ran before durable access intent")
        calls.append(group.selection_order)
        return _synthetic_access_observation(group, contract)

    injected = False
    def stop_after_prefix(point: str, selection_order: int | None) -> None:
        nonlocal injected
        if point == "after_asset_validated" and selection_order == 126 and not injected:
            injected = True
            raise KeyboardInterrupt("D7-R4 synthetic interruption")

    try:
        runtime.initialize_stage_d_access_root(
            root, pipeline.plan, estimate, preflight, execution_context=context,
            execution_gate=gate, asset_contracts=contracts, asset_validator=validator,
            fault_injector=stop_after_prefix,
        )
        raise StageDD7Error("D7-R4 interruption injection did not fire")
    except KeyboardInterrupt as error:
        if str(error) != "D7-R4 synthetic interruption":
            raise
    interrupted = runtime.validate_stage_d_access_root(
        root, pipeline.plan, estimate, preflight, execution_context=context,
        execution_gate=gate, asset_contracts=contracts, require_runnable=False,
    )
    if (
        interrupted["state"] != "ACCESS_STARTED"
        or interrupted["validated_asset_count"] != 127
        or list((root / "units").glob("*/attempts/*"))
    ):
        raise StageDD7Error("D7-R4 interruption evidence is not resumable/no-unit")
    resumed = runtime.initialize_stage_d_access_root(
        root, pipeline.plan, estimate, preflight, execution_context=context,
        execution_gate=gate, asset_contracts=contracts, asset_validator=validator,
    )
    independently_validated = runtime.validate_stage_d_access_root(
        root, pipeline.plan, estimate, preflight, execution_context=context,
        execution_gate=gate, asset_contracts=contracts, require_runnable=True,
    )
    if (
        resumed["state"] != "RUNNABLE"
        or independently_validated["validated_asset_count"] != 500
        or calls != list(range(500))
        or list((root / "units").glob("*/attempts/*"))
    ):
        raise StageDD7Error("D7-R4 initializer smoke did not reach no-unit RUNNABLE")
    inventory, total = _inventory(root)
    document = {
        "$schema": "../../config/stage_d_d7_access_smoke_r4.schema.json",
        "schema_version": 1,
        "document_type": "stage_d_d7_d8_initializer_smoke_receipt",
        "d7_id": D7_ID, "status": "PASS", "formal": False,
        "input_kind": "COMPLETE_500_GROUP_SYNTHETIC_FILESYSTEM_FIXTURE",
        "root": str(root.relative_to(repo)) if root.is_relative_to(repo) else str(root),
        "plan_digest": pipeline.plan.canonical_digest,
        "state_sequence": ["ACCESS_STARTED", "INTERRUPTED", "RESUMED", "RUNNABLE"],
        "counts": {
            "expected_assets": 500, "validated_assets": 500,
            "interrupted_after_assets": 127,
            "journal_entries": independently_validated["journal_entry_count"],
            "unit_attempt_directories": 0,
        },
        "digests": {
            "access_lifecycle": independently_validated["canonical_digest"],
            "journal_head": independently_validated["journal_head_digest"],
            "live_assets": independently_validated["live_assets_digest"],
        },
        "properties": {
            "access_started_durable_before_callback": True,
            "per_asset_intent_durable_before_callback": True,
            "append_only_hash_chain_source_recomputed": True,
            "interruption_resumed_without_duplicate_validation": True,
            "all_assets_required_before_runnable": True,
            "unit_execution_started": False,
            "synthetic_only": True,
        },
        "root_inventory": {
            "file_count": len(inventory), "total_bytes": total,
            "canonical_digest": canonical_sha256(inventory),
        },
        "access_attestation": {
            "reserve_image_exists_checked": False, "reserve_image_statted": False,
            "reserve_image_opened": False, "reserve_image_hashed": False,
            "reserve_image_decoded": False, "reserve_image_preprocessed": False,
            "reserve_image_used_for_inference": False,
            "stage_d_outcome_accessed": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(document, repo / _ACCESS_SMOKE_SCHEMA, "D7-R4 access smoke")
    _write_atomic(receipt_file, document)
    return document


def validate_d8_initializer_synthetic_smoke(
    *, repo_root: Path | str = REPO_ROOT, receipt_path: Path | str | None = None,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    path = repo / ACCESS_SMOKE_RECEIPT_PATH if receipt_path is None else Path(receipt_path)
    receipt, _ = _read_regular_json(path, "D7-R4 access smoke receipt")
    _schema_validate(receipt, repo / _ACCESS_SMOKE_SCHEMA, "D7-R4 access smoke")
    if receipt.get("canonical_digest") != _semantic_digest(receipt):
        raise StageDD7Error("D7-R4 access smoke receipt digest mismatch")
    root_value = Path(receipt["root"])
    root = root_value if root_value.is_absolute() else repo / root_value
    inventory, total = _inventory(root)
    if receipt["root_inventory"] != {
        "file_count": len(inventory), "total_bytes": total,
        "canonical_digest": canonical_sha256(inventory),
    }:
        raise StageDD7Error("D7-R4 access smoke inventory drifted")
    if set(receipt["access_attestation"].values()) != {False}:
        raise StageDD7Error("D7-R4 access smoke crossed the reserve boundary")
    return receipt


def validate_full_chain_synthetic_smoke(
    *, repo_root: Path | str = REPO_ROOT, receipt_path: Path | str | None = None,
) -> Mapping[str, Any]:
    from .stage_d_analysis import aggregate_stage_d_analysis, closed_analysis_summary
    from .stage_d_analysis_input import validate_stage_d_analysis_input
    from .stage_d_finalize import validate_stage_d_root
    from .stage_d_paper import (
        validate_stage_d_analysis_package, validate_stage_d_paper_package,
    )

    repo = Path(repo_root).resolve()
    path = repo / FULL_CHAIN_RECEIPT_PATH if receipt_path is None else Path(receipt_path)
    document, _ = _read_regular_json(path, "D7 full-chain smoke receipt")
    _schema_validate(document, repo / _FULL_CHAIN_SCHEMA, "D7 full-chain smoke")
    if document.get("canonical_digest") != _semantic_digest(document):
        raise StageDD7Error("D7 full-chain smoke digest mismatch")
    root_value = Path(document["root"])
    root = root_value if root_value.is_absolute() else repo / root_value
    pipeline = load_recovery_pipeline(repo_root=repo)
    estimate = estimate_stage_d_run(pipeline.contracts.config, pipeline.plan)
    preflight = preflight_stage_d(estimate)
    validated = validate_stage_d_root(
        root / "execution", pipeline.plan, estimate, preflight, allow_formal=False,
    )
    receipt = validate_stage_d_analysis_input(
        root / "execution", pipeline.plan, estimate, preflight,
    )
    recomputed = aggregate_stage_d_analysis(receipt, pipeline.plan)
    closed = closed_analysis_summary(recomputed)
    analysis = validate_stage_d_analysis_package(root / "analysis", repo_root=repo)
    paper = validate_stage_d_paper_package(root / "paper", repo_root=repo)
    inventory, total = _inventory(root)
    if (
        validated.workflow_status != "PASS"
        or closed["status"] != "PASS"
        or analysis["analysis_canonical_digest"] != closed["analysis_canonical_digest"]
        or paper["analysis_canonical_digest"] != closed["analysis_canonical_digest"]
        or document["root_inventory"] != {
            "file_count": len(inventory), "total_bytes": total,
            "canonical_digest": canonical_sha256(inventory),
        }
        or set(document["access_attestation"].values()) != {False}
    ):
        raise StageDD7Error("D7 full-chain smoke source recomputation failed")
    return document


def _load_c9_budget_source(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report, raw = _read_regular_json(repo / C9_REPORT_PATH, "sealed C9 report")
    if hashlib.sha256(raw).hexdigest() != C9_REPORT_FILE_SHA256:
        raise StageDD7Error("sealed C9 report file digest drifted")
    if report.get("canonical_digest") != C9_REPORT_CANONICAL_DIGEST or (
        _semantic_digest(report) != C9_REPORT_CANONICAL_DIGEST
    ):
        raise StageDD7Error("sealed C9 report canonical digest drifted")
    if not (
        report.get("status") == "PASS"
        and report.get("integrity_valid") is True
        and report.get("terminal_unit_count") == 320
        and report.get("state_counts", {}).get("succeeded") == 320
    ):
        raise StageDD7Error("sealed C9 report is not a complete source")
    projection = report["post_c9_budget"]["projection"]
    return report, projection


def build_d7_budget(*, repo_root: Path | str = REPO_ROOT) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    pipeline = load_recovery_pipeline(repo_root=repo)
    _report, source = _load_c9_budget_source(repo)
    measured = source["measured_maxima"]
    storage = source["storage"]
    seconds = float(source["time"]["slowest_valid_measured_unit_seconds"])
    profiles = []
    for method in METHOD_IDS:
        l2 = int(storage["method_profiles"][method]["conservative_l2_bytes_per_audit_unit"])
        for kind in ("primary_execution_unit", "audit_repeat_map"):
            profiles.append(StageDResourceProfile(
                method_id=method, unit_kind=kind, seconds=seconds,
                forward_calls=int(measured["forward_calls_per_unit"]),
                backward_calls=int(measured["backward_calls_per_unit"]),
                l0_l1_payload_bytes=int(
                    storage["method_profiles"][method]["conservative_non_l2_bytes_per_unit"]
                ),
                l2_payload_bytes=l2 if kind == "audit_repeat_map" else 0,
                control_and_provenance_bytes=int(
                    storage["conservative_batch_control_bytes_per_unit"]
                ),
                cuda_peak_reserved_bytes=int(
                    measured["cuda_peak_reserved_bytes_per_unit"]
                ),
                source_status=D7_PROFILE_SOURCE_STATUS,
            ))
    estimate = validate_stage_d_estimate(estimate_stage_d_run(
        pipeline.contracts.config, pipeline.plan, profiles,
    ))
    preflight = validate_stage_d_preflight(preflight_stage_d(estimate))
    if (
        estimate.profile_status != D7_VALIDATED_PROFILE_STATUS
        or estimate.budget_status != "PASS"
        or preflight.status != "READY_FOR_D7_AUTHORIZATION"
        or preflight.d7_final is not True
        or estimate.launch_violations or estimate.completed_gate_violations
    ):
        raise StageDD7Error("D7 conservative budget blocks launch")
    document = {
        "$schema": "stage_d_d7_budget_r4.schema.json",
        "schema_version": 1, "document_type": BUDGET_DOCUMENT_TYPE,
        "d7_id": D7_ID, "status": "PASS",
        "source": {
            "path": C9_REPORT_PATH,
            "file_sha256": C9_REPORT_FILE_SHA256,
            "canonical_digest": C9_REPORT_CANONICAL_DIGEST,
            "role": "SEALED_RECOVERY_C9_H20_TELEMETRY_NOT_C10_AUTHORITY",
            "slowest_valid_measured_unit_seconds": seconds,
        },
        "plan": {
            "canonical_digest": pipeline.plan.canonical_digest,
            "coordinate_digest": pipeline.plan.coordinate_digest,
            "image_groups": 500, "execution_units": 1000,
            "audit_repeat_maps": 10, "logical_work_items": 211010,
            "historical_8000_unit_projection_used": False,
        },
        "safety_factors": {"time": 1.25, "storage_and_control": 1.25},
        "estimate": estimate.to_dict(), "preflight": preflight.to_dict(),
        "authority": {
            "execution_authorization": "NOT_ISSUED",
            "analysis_authorization": ANALYSIS_AUTHORIZATION,
            "stage_d_execution_started": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(document, repo / _BUDGET_SCHEMA, "D7 budget")
    return document


def write_d7_budget(*, repo_root: Path | str = REPO_ROOT) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d7_budget(repo_root=repo)
    _write_atomic(repo / BUDGET_PATH, document)
    return document


def load_d7_budget(*, repo_root: Path | str = REPO_ROOT) -> tuple[Mapping[str, Any], StageDEstimate, StageDPreflight]:
    repo = Path(repo_root).resolve()
    document, _ = _read_regular_json(repo / BUDGET_PATH, "D7 budget")
    _schema_validate(document, repo / _BUDGET_SCHEMA, "D7 budget")
    if document.get("canonical_digest") != _semantic_digest(document):
        raise StageDD7Error("D7 budget canonical digest mismatch")
    expected = build_d7_budget(repo_root=repo)
    if document != expected:
        raise StageDD7Error("D7 budget is not source-recomputable")
    estimate = validate_stage_d_estimate(document["estimate"])
    preflight = validate_stage_d_preflight(document["preflight"])
    return document, estimate, preflight


def verify_recovery_model(*, repo_root: Path | str = REPO_ROOT) -> Mapping[str, Any]:
    import torch

    from .recovery_v2 import canonical_state_dict_tensor_digest

    repo = Path(repo_root).resolve()
    recovery, _file_sha, _canonical = load_recovery_binding(repo)
    model = recovery["model"]
    path = repo / model["path"]
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise StageDD7Error("recovery-v2 model must be a single-link regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            handle.seek(0)
            state = torch.load(handle, map_location="cpu", weights_only=True)
        after = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
        )
        if identity(before) != identity(after):
            raise StageDD7Error("recovery-v2 model changed during verification")
    finally:
        os.close(descriptor)
    tensor_digest = canonical_state_dict_tensor_digest(state)
    parameter_count = sum(int(value.numel()) for value in state.values())
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    relative = path.relative_to(top).as_posix()
    matches = [
        line.strip() for line in _git(repo, "lfs", "ls-files", "-l").splitlines()
        if line.strip().endswith(" " + relative)
    ]
    if len(matches) != 1:
        raise StageDD7Error("recovery-v2 model has no unique LFS identity")
    fields = matches[0].split(maxsplit=2)
    if len(fields) != 3 or fields[2] != relative:
        raise StageDD7Error("recovery-v2 model LFS path binding drifted")
    # Hydration is established by the stable 346 MB no-follow read and the
    # exact content digest above.  Git LFS marker spelling differs by client
    # version and is therefore not used as a second hydration oracle.
    tensors = {
        "state_dict_sha256": digest.hexdigest(),
        "canonical_tensor_digest": tensor_digest, "key_count": len(state),
        "parameter_count": parameter_count,
    }
    if (
        tensors["state_dict_sha256"] != RECOVERY_MODEL_SHA256
        or tensors["canonical_tensor_digest"] != RECOVERY_MODEL_TENSOR_DIGEST
        or fields[0] != RECOVERY_MODEL_SHA256
        or tensors["key_count"] != 152
        or tensors["parameter_count"] != 86_567_656
    ):
        raise StageDD7Error("recovery-v2 model identity verification failed")
    return {
        "identity": model["identity"], "path": model["path"],
        **tensors, "lfs_oid_sha256": fields[0],
    }


def production_backend_factory(*, context: Any, device: str) -> Any:
    """Create the D8 backend lazily after runtime validates D7 authority."""

    del device
    from .modeling import canonical_preprocess_spec
    from .recovery_v2 import load_recovery_v2_binding
    from .stage_d_producer import ProductionStageDBackend

    binding = load_recovery_v2_binding(
        REPO_ROOT / "config/recovery_v2_binding.json",
        schema_path=REPO_ROOT / "config/recovery_v2_binding.schema.json",
        repo_root=REPO_ROOT, verify_assets=False,
    )
    pipeline = load_recovery_pipeline(repo_root=REPO_ROOT)
    rows_by_sample = {
        row["sample_id"]: row
        for row in pipeline.manifest.registry_document["rows"]
    }

    def sample_field(sample: Any, name: str) -> Any:
        if isinstance(sample, Mapping):
            return sample.get(name)
        return getattr(sample, name, None)

    def load_sample(sample: Any, model: Any) -> dict[str, Any]:
        import torch
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transform_functional
        from .io_utils import sha256_file

        sample_id = sample_field(sample, "sample_id")
        relative_path = sample_field(sample, "relative_path")
        expected_sha256 = sample_field(sample, "image_sha256")
        if (
            not isinstance(sample_id, str) or not isinstance(relative_path, str)
            or not sample_id.endswith("/" + relative_path)
            or not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise StageDD7Error("authorized Stage D sample identity is invalid")
        row = rows_by_sample.get(sample_id)
        if (
            not isinstance(row, Mapping)
            or row.get("relative_path") != relative_path
            or row.get("image_sha256") != expected_sha256
        ):
            raise StageDD7Error("authorized sample differs from frozen registry")
        # This closure is invoked only by a future D8 run.  Merely constructing
        # the D7 producer never resolves or stats a registry-referenced image.
        image_path = Path(binding.data_root) / relative_path
        if sha256_file(image_path) != expected_sha256:
            raise StageDD7Error("authorized Stage D image digest mismatch")
        specification = canonical_preprocess_spec(model)
        if (
            specification.input_size != (3, 224, 224)
            or specification.resize_shorter_side != 248
            or specification.center_crop != (224, 224)
            or specification.interpolation != "bicubic"
            or specification.mean != (0.5, 0.5, 0.5)
            or specification.std != (0.5, 0.5, 0.5)
        ):
            raise StageDD7Error("canonical preprocessing contract drifted")
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            resized = transform_functional.resize(
                rgb, specification.resize_shorter_side,
                interpolation=InterpolationMode.BICUBIC, antialias=True,
            )
            cropped = transform_functional.center_crop(
                resized, list(specification.center_crop),
            )
            canonical = transform_functional.pil_to_tensor(cropped).to(
                dtype=torch.float32
            ).div(255.0).unsqueeze(0)
        mean = torch.tensor(specification.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(specification.std, dtype=torch.float32).view(1, 3, 1, 1)
        return {
            "canonical_rgb": canonical,
            "normalized_input": (canonical - mean) / std,
            "registry_row": row, "image_path": image_path,
        }

    return ProductionStageDBackend(
        sample_loader=load_sample, state_dict_path=binding.model_path,
        # The Stage C base class historically requires a paired C7 authority.
        # D7 itself is the stronger Stage D execution gate, so model loading is
        # bound directly by recovery-v2 and the opaque runtime context.
        model_loader=lambda **kwargs: __import__(
            "vitnns.modeling", fromlist=["load_canonical_model"]
        ).load_canonical_model(
            **{key: value for key, value in kwargs.items() if key != "recovery_binding"},
            recovery_binding=binding,
        ),
        stage_d_model_binding=dict(context.model_binding),
        config=pipeline.contracts.config,
        registry=pipeline.manifest.registry_document, plan=pipeline.plan,
        config_path=REPO_ROOT / "config/stage_d.json",
        code_state=dict(context.code_binding),
    )


def validate_live_asset(
    group: Any, contract: Mapping[str, Any], *, data_root: Path | str,
) -> Mapping[str, Any]:
    """Hash and decode exactly one authorized asset after durable intent.

    The caller owns ordering and journaling.  This function accepts only a flat,
    no-follow, single-link regular file and returns no pixels or model inputs.
    """

    relative_path = getattr(group, "relative_path", None)
    sample_id = getattr(group, "sample_id", None)
    expected_sha256 = getattr(group, "image_sha256", None)
    pure = PurePosixPath(relative_path) if isinstance(relative_path, str) else None
    if (
        pure is None or pure.is_absolute() or len(pure.parts) != 1
        or pure.as_posix() != relative_path or sample_id != f"ILSVRC2012_val/{relative_path}"
        or not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise StageDD7Error("authorized live-asset path is unsafe")
    if (
        contract.get("sample_id") != sample_id
        or contract.get("relative_path") != relative_path
        or contract.get("image_sha256") != expected_sha256
    ):
        raise StageDD7Error("live-asset contract differs from the frozen plan")
    root = Path(os.path.abspath(os.fspath(data_root)))
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    root_fd = os.open(root, directory_flags)
    try:
        descriptor = os.open(relative_path, file_flags, dir_fd=root_fd)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size != contract.get("size_bytes")
            ):
                raise StageDD7Error("live asset is not the expected single-link file")
            payload = bytearray()
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                payload.extend(block)
                digest.update(block)
            after = os.fstat(descriptor)
            identity = lambda item: (
                item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
                item.st_size, item.st_mtime_ns, item.st_ctime_ns,
            )
            if identity(before) != identity(after) or len(payload) != before.st_size:
                raise StageDD7Error("live asset changed while being read")
        finally:
            os.close(descriptor)
    finally:
        os.close(root_fd)
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != expected_sha256:
        raise StageDD7Error("live asset SHA-256 differs from frozen registry")
    from PIL import Image
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
            mode, image_format = image.mode, image.format
    except Exception as error:
        raise StageDD7Error("live asset is not a decodable JPEG") from error
    expected_metadata = {
        "width": contract.get("width"), "height": contract.get("height"),
        "mode": contract.get("mode"), "image_format": contract.get("image_format"),
    }
    if {"width": width, "height": height, "mode": mode, "image_format": image_format} != expected_metadata:
        raise StageDD7Error("live asset decoded metadata differs from frozen registry")
    return {
        "source_kind": "LIVE_RESERVE_JPEG",
        "source_payload_sha256": observed_sha256,
        "image_sha256": observed_sha256, "size_bytes": len(payload),
        "width": width, "height": height, "mode": mode,
        "image_format": image_format,
        "official_ilsvrc2012_class_id": contract["official_ilsvrc2012_class_id"],
        "official_wnid": contract["official_wnid"],
        "imagenet1k_index": contract["imagenet1k_index"],
        "label_source": contract["label_source"],
    }


def production_access_validator(
    *, repo_root: Path | str = REPO_ROOT,
) -> Any:
    """Return a lazy validator so no data path is resolved before journaling."""

    repo = Path(repo_root).resolve()
    data_root: Path | None = None

    def validate(group: Any, contract: Mapping[str, Any]) -> Mapping[str, Any]:
        nonlocal data_root
        if data_root is None:
            binding, _ = _read_regular_json(
                repo / "config/recovery_v2_binding.json", "recovery-v2 binding"
            )
            raw = binding.get("stage_c_input", {}).get("data_root")
            relative = PurePosixPath(raw) if isinstance(raw, str) else None
            if (
                relative is None or relative.is_absolute() or relative.as_posix() != raw
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise StageDD7Error("recovery-v2 data root binding is unsafe")
            data_root = repo.joinpath(*relative.parts)
        return validate_live_asset(group, contract, data_root=data_root)

    return validate


def _fresh_formal_root(repo: Path, raw: str) -> Path:
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or not raw.startswith("evidence/stage_d/"):
        raise StageDD7Error("formal root must be a safe project-relative Stage D evidence path")
    root = repo.joinpath(*relative.parts)
    current = repo
    for part in relative.parts[:-1]:
        current /= part
        if os.path.lexists(current):
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise StageDD7Error("formal root parent is unsafe")
    if os.path.lexists(root):
        raise StageDD7Error("formal D8 root is not fresh")
    return root


def _producer_state(
    repo: Path, *, producer_ref: str = PRODUCER_REF,
) -> Mapping[str, Any]:
    producer = _git(repo, "rev-parse", f"{producer_ref}^{{commit}}")
    tree = _git(repo, "rev-parse", f"{producer}^{{tree}}")
    if _COMMIT_RE.fullmatch(producer) is None or _COMMIT_RE.fullmatch(tree) is None:
        raise StageDD7Error("D7 producer ref is invalid")
    return {"commit": producer, "tree": tree, "clean": True, "ref": producer_ref}


def _source_bindings(repo: Path) -> Mapping[str, Any]:
    recovery, recovery_sha, recovery_digest = load_recovery_binding(repo)
    d6, d6_raw = _read_regular_json(
        repo / "reports/stage_d_recovery_v2_20260904_r3/d6/status.json",
        "D6 terminal record",
    )
    return {
        "d0": {"status": "RECOVERY_V2_R3_PASS", "canonical_digest": "124b404297c99e5a1913511a51caedec241ce05804949b390a14719814eb6c11"},
        "d1": {"status": "RECOVERY_V2_R3_METADATA_PASS", "canonical_digest": "00ad4968256606e95022f22d7502990223274738ca8066526134cae1945cbf40"},
        "d2": {"status": "RECOVERY_V2_R3_PASS", "canonical_digest": "5b3c8497dccda450b1b2682364c4f07bcf017e657d999ccb1d765aae9eec224c"},
        "c10": {
            "status": "BLOCKED_NO_GO_NOVELTY_HISTORICAL_SOURCE",
            "canonical_digest": recovery["recovery_chain"]["c10"]["report_canonical_digest"],
            "downstream_authorized": False,
        },
        "d6": {
            "status": "ANALYSIS_IMPLEMENTATION_READINESS_PASS",
            "canonical_digest": d6["canonical_digest"],
            "file_sha256": hashlib.sha256(d6_raw).hexdigest(),
        },
        "recovery_binding": {
            "status": "CURRENT_RECOVERY_V2_OVERLAY",
            "canonical_digest": recovery_digest, "file_sha256": recovery_sha,
        },
    }


def build_execution_authorization(
    *, repo_root: Path | str = REPO_ROOT, shard_count: int = DEFAULT_SHARD_COUNT,
    formal_root: str = FORMAL_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageDD7Error("D7 authorization requires a clean producer worktree")
    producer = _producer_state(repo)
    if _git(repo, "rev-parse", "HEAD") != producer["commit"]:
        raise StageDD7Error("authorization must be generated at the frozen producer commit")
    smoke = validate_full_chain_synthetic_smoke(repo_root=repo)
    initializer_smoke = validate_d8_initializer_synthetic_smoke(repo_root=repo)
    budget, estimate, preflight = load_d7_budget(repo_root=repo)
    pipeline = load_recovery_pipeline(repo_root=repo)
    _fresh_formal_root(repo, formal_root)
    model = verify_recovery_model(repo_root=repo)
    mapping = dict(validate_stage_d_shard_mapping(pipeline.plan, shard_count))
    group_counts = [sum(value == index for value in mapping.values()) for index in range(shard_count)]
    lock_raw = (repo / "requirements.lock.txt").read_bytes()
    attestation, attestation_raw = _read_regular_json(
        repo / "manifests/stage_d_fidelity_v1/pre_authorization_access_attestation.json",
        "pre-authorization access attestation",
    )
    if (
        attestation["current_task_access_facts"]["reserve_image_bytes_accessed"] is not False
        or attestation["authorization"]["execution_authorization"] != "NOT_ISSUED"
    ):
        raise StageDD7Error("pre-authorization access attestation is not closed")
    document = {
        "$schema": "stage_d_d7_execution_authorization_r4.schema.json",
        "schema_version": 1, "document_type": AUTHORIZATION_DOCUMENT_TYPE,
        "d7_id": D7_ID, "status": "PASS",
        "authorization_scope": "D7_RECOVERY_V2_FIDELITY_ONLY",
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "reserve_access_boundary": "DEFERRED_TO_D8_0_AFTER_REVALIDATION",
        "stage_d_execution_started": False,
        "authority_commit": "SELF", "authority_ref": AUTHORITY_REF,
        "producer": producer,
        "formal_root": formal_root, "formal_root_fresh_at_issuance": True,
        "plan": {
            "canonical_digest": pipeline.plan.canonical_digest,
            "coordinate_digest": pipeline.plan.coordinate_digest,
            "counts": _plain(pipeline.plan.counts),
        },
        "budget": {
            "path": BUDGET_PATH, "canonical_digest": budget["canonical_digest"],
            "estimate": estimate.to_dict(), "preflight": preflight.to_dict(),
        },
        "sharding": {
            "shard_count": shard_count,
            "assignment_formula": "selection_order_modulo_frozen_shard_count",
            "mapping_digest": canonical_sha256([mapping[index] for index in range(500)]),
            "image_group_counts": group_counts,
            "serial_shard_handoff": True, "image_group_split_allowed": False,
        },
        "smoke": {
            "path": FULL_CHAIN_RECEIPT_PATH,
            "canonical_digest": smoke["canonical_digest"],
            "formal": False, "image_groups": 500,
            "execution_units": 1000,
            "real_d4_d5_d6_api_chain": True,
        },
        "d8_initializer": {
            "path": ACCESS_SMOKE_RECEIPT_PATH,
            "canonical_digest": initializer_smoke["canonical_digest"],
            "synthetic_only": True, "durable_access_started": True,
            "append_only_journal": True, "interruption_recovery": True,
            "failure_blocks": True, "all_500_required_before_runnable": True,
            "unit_execution_started": False,
        },
        "code_binding": {
            "commit": producer["commit"], "tree": producer["tree"],
            "clean": True,
        },
        "dependency_binding": {
            "path": "requirements.lock.txt",
            "lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
        },
        "model_binding": model, "source_bindings": _source_bindings(repo),
        "pre_authorization_attestation": {
            "path": "manifests/stage_d_fidelity_v1/pre_authorization_access_attestation.json",
            "file_sha256": hashlib.sha256(attestation_raw).hexdigest(),
            "canonical_digest": attestation["canonical_digest"],
            "forbidden_access_detected": False,
        },
        "access_attestation": {
            "reserve_image_bytes_accessed": False,
            "model_loaded_for_identity_verification": True,
            "model_used_for_inference": False,
            "cuda_accessed": False, "outcome_accessed": False,
        },
        "model_verification": {
            "file_and_tensor_identity_verified": True,
            "used_for_inference": False, "cuda_used": False,
        },
        "c10_status_preserved": {
            "acceptance": "BLOCKED", "primary_decision": "NO_GO_NOVELTY",
            "translation_stability": "FAIL_STABILITY",
            "center_crop_resize_stability": "FAIL_STABILITY",
            "overall_spatial": "FAIL_SPATIAL",
            "holm": "HOLM_BLOCKED_UNDERSPECIFIED",
            "downstream_authorized": False,
        },
        "prohibitions": {
            "d8_initialized": False, "d9_started": False, "d10_started": False,
            "formal_aggregation_authorized": False,
            "c10_stage_d_pooling_allowed": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(document, repo / _AUTH_SCHEMA, "D7 execution authorization")
    return document


def _load_r4_execution_authorization(
    repo: Path,
) -> tuple[Mapping[str, Any], bytes]:
    source, raw = _read_regular_json(
        repo / AUTHORIZATION_PATH, "D7-R4 execution authorization",
    )
    _schema_validate(source, repo / _AUTH_SCHEMA, "D7-R4 execution authorization")
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D7-R4 execution authorization digest mismatch")
    return source, raw


def _blocked_root_record(path: Path) -> Mapping[str, Any]:
    source, _ = _read_regular_json(path, "D8.0 blocked-root record")
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D8.0 blocked-root record digest mismatch")
    if (
        source.get("document_type") != "stage_d_d8_0_initial_blocked_root"
        or source.get("subtask_status") != "SUBTASK_BLOCKED"
        or source.get("formal_root") != FORMAL_ROOT
        or source.get("prohibitions_preserved", {}).get("unit_execution_started")
        is not False
    ):
        raise StageDD7Error("D8.0 blocked-root record is not authoritative")
    observed = source.get("asset_validation_counts", {})
    if observed.get("validated") != 0 or observed.get("missing") != 1:
        raise StageDD7Error("D8.0 blocked-root count evidence drifted")
    return source


def _materialization_record(path: Path) -> Mapping[str, Any]:
    source, _ = _read_regular_json(path, "D8.0 asset materialization record")
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D8.0 asset materialization record digest mismatch")
    if (
        source.get("document_type") != "stage_d_d8_0_asset_materialization"
        or source.get("status") != "PASS"
    ):
        raise StageDD7Error("D8.0 materialization record is not a PASS")
    counts = source.get("asset_validation_counts", {})
    if (
        counts.get("expected") != 500
        or counts.get("validated") != 500
        or counts.get("missing") != 0
        or counts.get("tampered") != 0
        or counts.get("unsafe") != 0
        or counts.get("decode_failure") != 0
        or counts.get("metadata_mismatch") != 0
    ):
        raise StageDD7Error("D8.0 materialization counts are not complete")
    return source


def build_d8_0_fresh_root_authorization(
    *, repo_root: Path | str = REPO_ROOT,
    formal_root: str = D8_0_REBIND_FORMAL_ROOT,
    require_current_head: bool = True,
    require_fresh_root: bool = True,
    require_clean_worktree: bool = True,
) -> Mapping[str, Any]:
    """Bind a replacement fresh root after the first D8.0 root blocked.

    This does not reissue the scientific execution authority.  It records the
    outcome-blind diagnosis, successful asset materialization, and new root
    path so the existing R4 authority can be consumed without reusing a blocked
    root.
    """

    repo = Path(repo_root).resolve()
    if (
        require_clean_worktree
        and _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise StageDD7Error("D8.0 rebind authorization requires a clean worktree")
    producer = _producer_state(repo, producer_ref=D8_0_REBIND_PRODUCER_REF)
    if require_current_head and _git(repo, "rev-parse", "HEAD") != producer["commit"]:
        raise StageDD7Error(
            "D8.0 rebind authorization must be generated at the frozen producer commit"
        )
    r4, r4_raw = _load_r4_execution_authorization(repo)
    blocked = _blocked_root_record(repo / D8_0_INITIAL_BLOCKED_RECORD_PATH)
    materialized = _materialization_record(repo / D8_0_MATERIALIZATION_RECORD_PATH)
    if require_fresh_root:
        _fresh_formal_root(repo, formal_root)
    if formal_root == r4["formal_root"]:
        raise StageDD7Error("D8.0 rebind authorization must select a new fresh root")
    if (
        materialized.get("source_blocked_root", {}).get("path") != FORMAL_ROOT
        or materialized.get("source_blocked_root", {}).get("lifecycle_digest")
        != blocked["digests"]["access_lifecycle"]
        or materialized.get("source_blocked_root", {}).get("journal_head_digest")
        != blocked["digests"]["journal_head"]
        or materialized.get("live_assets_digest") is None
    ):
        raise StageDD7Error("D8.0 materialization is not bound to the blocked root")
    authority = _git(repo, "rev-parse", f"{AUTHORITY_REF}^{{commit}}")
    document = {
        "$schema": "stage_d_d8_0_fresh_root_authorization.schema.json",
        "schema_version": 1,
        "document_type": D8_0_REBIND_AUTHORIZATION_DOCUMENT_TYPE,
        "authorization_id": D8_0_REBIND_ID,
        "status": "PASS",
        "producer": producer,
        "code_binding": {
            "commit": producer["commit"], "tree": producer["tree"],
            "clean": True,
        },
        "base_authorization": {
            "path": AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(r4_raw).hexdigest(),
            "canonical_digest": r4["canonical_digest"],
            "authority_ref": AUTHORITY_REF,
            "authority_commit": authority,
        },
        "diagnosis": {
            "blocked_record_path": D8_0_INITIAL_BLOCKED_RECORD_PATH,
            "blocked_formal_root": FORMAL_ROOT,
            "blocked_lifecycle_digest": blocked["digests"]["access_lifecycle"],
            "blocked_journal_head_digest": blocked["digests"]["journal_head"],
            "blocking_reason": "LIVE_ASSET_VALIDATION_FAILED",
            "validated_asset_count_before_block": 0,
            "root_retained": True,
        },
        "asset_materialization": {
            "record_path": D8_0_MATERIALIZATION_RECORD_PATH,
            "canonical_digest": materialized["canonical_digest"],
            "live_assets_digest": materialized["live_assets_digest"],
            "validated_asset_count": 500,
            "missing": 0,
            "tampered": 0,
            "unsafe": 0,
            "decode_failure": 0,
            "metadata_mismatch": 0,
        },
        "replacement_formal_root": formal_root,
        "replacement_formal_root_fresh_at_issuance": True,
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "stage_d_execution_started": False,
        "d8_shards_started": False,
        "d9_started": False,
        "d10_started": False,
        "prohibitions": {
            "run_shard_before_reinitialized_runnable": False,
            "formal_aggregation_authorized": False,
            "analysis_authorization_issued": False,
            "c10_status_rewrite_allowed": False,
            "reuse_blocked_root_allowed": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(
        document, repo / _D8_0_REBIND_AUTH_SCHEMA,
        "D8.0 fresh-root authorization",
    )
    return document


def _validate_d8_0_fresh_root_authorization(
    source: Mapping[str, Any], raw: bytes, *, repo: Path,
    require_fresh_root: bool, require_authority_ref: bool,
) -> tuple[Mapping[str, Any], str]:
    _schema_validate(
        source, repo / _D8_0_REBIND_AUTH_SCHEMA,
        "D8.0 fresh-root authorization",
    )
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D8.0 fresh-root authorization digest mismatch")
    expected = build_d8_0_fresh_root_authorization(
        repo_root=repo, formal_root=source["replacement_formal_root"],
        require_current_head=False, require_fresh_root=False,
        require_clean_worktree=False,
    )
    if source != expected:
        raise StageDD7Error("D8.0 fresh-root authorization is not source-recomputable")
    if require_fresh_root:
        _fresh_formal_root(repo, source["replacement_formal_root"])
    if require_authority_ref:
        authority = _git(repo, "rev-parse", f"{D8_0_REBIND_AUTHORITY_REF}^{{commit}}")
        parent = _git(repo, "rev-parse", f"{authority}^")
        producer = _git(repo, "rev-parse", f"{D8_0_REBIND_PRODUCER_REF}^{{commit}}")
        changed = set(_git(repo, "diff", "--name-only", parent, authority).splitlines())
        if parent != producer or changed != _D8_0_REBIND_AUTHORITY_ONLY_PATHS:
            raise StageDD7Error(
                "D8.0 rebind authority commit is not an authority-only direct child"
            )
        prefix = _git(repo, "rev-parse", "--show-prefix")
        blob = _git(
            repo, "rev-parse",
            f"{authority}:{prefix}{D8_0_REBIND_AUTHORIZATION_PATH}",
        )
        expected_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != expected_blob:
            raise StageDD7Error(
                "D8.0 rebind authority ref does not contain the authorization bytes"
            )
    base = source["base_authorization"]
    producer = _producer_state(repo, producer_ref=D8_0_REBIND_PRODUCER_REF)
    if source["producer"] != producer or source["code_binding"] != {
        "commit": producer["commit"], "tree": producer["tree"], "clean": True,
    }:
        raise StageDD7Error("D8.0 rebind producer binding drifted")
    return base, source["replacement_formal_root"]


def write_d8_0_fresh_root_authorization(
    *, repo_root: Path | str = REPO_ROOT,
    formal_root: str = D8_0_REBIND_FORMAL_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d8_0_fresh_root_authorization(
        repo_root=repo, formal_root=formal_root,
    )
    _write_atomic(repo / D8_0_REBIND_AUTHORIZATION_PATH, document)
    return document


def build_d8_continuation_authorization(
    *, repo_root: Path | str = REPO_ROOT, require_current_head: bool = True,
    require_clean_worktree: bool = True,
) -> Mapping[str, Any]:
    """Authorize the checkpoint-binding correction for post-S00 shards."""

    repo = Path(repo_root).resolve()
    if (
        require_clean_worktree
        and _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise StageDD7Error(
            "D8 continuation authorization requires a clean worktree"
        )
    producer = _producer_state(
        repo, producer_ref=D8_CONTINUATION_PRODUCER_REF,
    )
    if require_current_head and _git(repo, "rev-parse", "HEAD") != producer["commit"]:
        raise StageDD7Error(
            "D8 continuation authorization must be generated at the frozen producer"
        )
    rebind, rebind_raw = _read_regular_json(
        repo / D8_0_REBIND_AUTHORIZATION_PATH,
        "D8.0 fresh-root authorization",
    )
    _validate_d8_0_fresh_root_authorization(
        rebind, rebind_raw, repo=repo, require_fresh_root=False,
        require_authority_ref=True,
    )
    receipt, receipt_raw = _read_regular_json(
        repo / D8_S00_RECEIPT_PATH, "D8-S00 shard receipt",
    )
    completion, completion_raw = _read_regular_json(
        repo / D8_S00_COMPLETION_PATH, "D8-S00 completion record",
    )
    if (
        receipt.get("document_type") != "stage_d_d8_shard_receipt"
        or receipt.get("status") != "PASS"
        or receipt.get("subtask") != "D8-S00"
        or receipt.get("canonical_digest") != _semantic_digest(receipt)
        or receipt.get("formal_root") != D8_0_REBIND_FORMAL_ROOT
        or receipt.get("ledger_transition", {}).get("after_shard_state")
        != "completed"
        or receipt.get("ledger_transition", {}).get("later_shard_states")
        != ["pending"] * 7
    ):
        raise StageDD7Error("D8-S00 receipt is not a completed continuation anchor")
    if (
        completion.get("document_type") != "stage_d_d8_s00_completion"
        or completion.get("subtask_status") != "SUBTASK_PASS"
        or completion.get("canonical_digest") != _semantic_digest(completion)
        or completion.get("receipt", {}).get("receipt_canonical_digest")
        != receipt["canonical_digest"]
        or completion.get("continuation", {}).get("shard_continuation_authorized")
        is not False
    ):
        raise StageDD7Error("D8-S00 completion does not require continuation repair")
    document = {
        "$schema": "stage_d_d8_continuation_authorization.schema.json",
        "schema_version": 1,
        "document_type": D8_CONTINUATION_AUTHORIZATION_DOCUMENT_TYPE,
        "authorization_id": D8_CONTINUATION_ID,
        "status": "PASS",
        "producer": producer,
        "control_plane_code_binding": {
            "commit": producer["commit"], "tree": producer["tree"],
            "clean": True,
        },
        "persisted_execution_context": {
            "authorization_path": D8_0_REBIND_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(rebind_raw).hexdigest(),
            "canonical_digest": rebind["canonical_digest"],
            "authority_ref": D8_0_REBIND_AUTHORITY_REF,
            "authority_commit": _git(
                repo, "rev-parse", f"{D8_0_REBIND_AUTHORITY_REF}^{{commit}}",
            ),
            "code_binding": rebind["code_binding"],
        },
        "formal_root": D8_0_REBIND_FORMAL_ROOT,
        "s00_anchor": {
            "receipt_path": D8_S00_RECEIPT_PATH,
            "receipt_file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_canonical_digest": receipt["canonical_digest"],
            "completion_path": D8_S00_COMPLETION_PATH,
            "completion_file_sha256": hashlib.sha256(completion_raw).hexdigest(),
            "completion_canonical_digest": completion["canonical_digest"],
            "ledger_after_digest": receipt["ledger_transition"]["after_canonical_digest"],
            "ledger_after_generation": receipt["ledger_transition"]["after_generation"],
        },
        "correction": {
            "scope": "ACCESS_START_CHECKPOINT_BINDING_ONLY",
            "d8_0_initial_checkpoint_remains_immutable": True,
            "current_checkpoint_still_source_validated": True,
            "access_journal_live_assets_and_authority_checks_unchanged": True,
            "scientific_scope_changed": False,
            "existing_s00_evidence_rewritten": False,
        },
        "authorized_next_shard": 1,
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "d9_started": False,
        "d10_started": False,
        "prohibitions": {
            "rerun_s00_allowed": False,
            "run_s02_or_later_allowed": False,
            "formal_aggregation_authorized": False,
            "analysis_authorization_issued": False,
            "c10_status_rewrite_allowed": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(
        document, repo / _D8_CONTINUATION_AUTH_SCHEMA,
        "D8 continuation authorization",
    )
    return document


def _validate_d8_continuation_authorization(
    source: Mapping[str, Any], raw: bytes, *, repo: Path,
    require_authority_ref: bool,
) -> Mapping[str, Any]:
    _schema_validate(
        source, repo / _D8_CONTINUATION_AUTH_SCHEMA,
        "D8 continuation authorization",
    )
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D8 continuation authorization digest mismatch")
    expected = build_d8_continuation_authorization(
        repo_root=repo, require_current_head=False,
        require_clean_worktree=False,
    )
    if source != expected:
        raise StageDD7Error("D8 continuation authorization is not source-recomputable")
    if require_authority_ref:
        authority = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_AUTHORITY_REF}^{{commit}}",
        )
        parent = _git(repo, "rev-parse", f"{authority}^")
        producer = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_PRODUCER_REF}^{{commit}}",
        )
        changed = set(_git(repo, "diff", "--name-only", parent, authority).splitlines())
        if parent != producer or changed != _D8_CONTINUATION_AUTHORITY_ONLY_PATHS:
            raise StageDD7Error(
                "D8 continuation authority is not an authority-only direct child"
            )
        prefix = _git(repo, "rev-parse", "--show-prefix")
        blob = _git(
            repo, "rev-parse",
            f"{authority}:{prefix}{D8_CONTINUATION_AUTHORIZATION_PATH}",
        )
        expected_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != expected_blob:
            raise StageDD7Error(
                "D8 continuation authority does not contain the authorization bytes"
            )
    return source["persisted_execution_context"]


def write_d8_continuation_authorization(
    *, repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d8_continuation_authorization(repo_root=repo)
    _write_atomic(repo / D8_CONTINUATION_AUTHORIZATION_PATH, document)
    return document


def _validate_d8_continuation_r2_checkpoint_anchor(
    source: Mapping[str, Any], *, repo: Path,
) -> None:
    """Require the live root to remain at the exact pre-S02 handoff."""

    checkpoint, _raw = _read_regular_json(
        repo / D8_0_REBIND_FORMAL_ROOT / "runtime_checkpoint.json",
        "D8 generation-760 runtime checkpoint",
    )
    anchor = source["checkpoint_anchor"]
    shards = checkpoint.get("shards")
    units = checkpoint.get("units")
    groups = checkpoint.get("groups")
    lifecycle = checkpoint.get("lifecycle")
    if (
        checkpoint.get("canonical_digest") != _semantic_digest(checkpoint)
        or checkpoint.get("canonical_digest") != anchor["canonical_digest"]
        or checkpoint.get("generation") != anchor["generation"]
        or not isinstance(shards, list)
        or [row.get("state") for row in shards] != anchor["shard_states"]
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("active_shard_leases")
        != anchor["active_shard_leases"]
        or lifecycle.get("active_group_claims")
        != anchor["active_group_claims"]
        or not isinstance(groups, list)
        or not isinstance(units, list)
    ):
        raise StageDD7Error(
            "D8-S02 checkpoint drifted from the authorized generation-760 handoff"
        )
    shard_groups = [
        row for row in groups if row.get("selection_order", -1) % DEFAULT_SHARD_COUNT == 2
    ]
    shard_units = [
        row for row in units
        if row.get("group_selection_order", -1) % DEFAULT_SHARD_COUNT == 2
    ]
    if (
        len(shard_groups) != 63
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_groups
        )
        or len(shard_units) != 126
        or sum(len(row.get("attempts", [])) for row in shard_units)
        != anchor["authorized_shard_attempt_count"]
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_units
        )
    ):
        raise StageDD7Error(
            "D8-S02 coordinates are not pristine at the authorized checkpoint"
        )


def build_d8_continuation_r2_authorization(
    *, repo_root: Path | str = REPO_ROOT, require_current_head: bool = True,
    require_clean_worktree: bool = True, require_current_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Roll the immutable S01 receipt into an exact shard-2 authority."""

    repo = Path(repo_root).resolve()
    if (
        require_clean_worktree
        and _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise StageDD7Error(
            "D8 continuation R2 authorization requires a clean worktree"
        )
    producer = _producer_state(
        repo, producer_ref=D8_CONTINUATION_R2_PRODUCER_REF,
    )
    if require_current_head and _git(repo, "rev-parse", "HEAD") != producer["commit"]:
        raise StageDD7Error(
            "D8 continuation R2 authorization must be generated at the frozen producer"
        )

    rebind, rebind_raw = _read_regular_json(
        repo / D8_0_REBIND_AUTHORIZATION_PATH,
        "D8.0 fresh-root authorization",
    )
    _validate_d8_0_fresh_root_authorization(
        rebind, rebind_raw, repo=repo, require_fresh_root=False,
        require_authority_ref=True,
    )
    prior_authority, prior_authority_raw = _read_regular_json(
        repo / D8_CONTINUATION_AUTHORIZATION_PATH,
        "D8 continuation R1 authorization",
    )
    _validate_d8_continuation_authorization(
        prior_authority, prior_authority_raw, repo=repo,
        require_authority_ref=True,
    )
    receipt, receipt_raw = _read_regular_json(
        repo / D8_S01_RECEIPT_PATH, "D8-S01 shard receipt",
    )
    completion, completion_raw = _read_regular_json(
        repo / D8_S01_COMPLETION_PATH, "D8-S01 completion record",
    )
    if (
        receipt.get("document_type") != "stage_d_d8_shard_receipt"
        or receipt.get("status") != "PASS"
        or receipt.get("subtask") != "D8-S01"
        or receipt.get("canonical_digest") != _semantic_digest(receipt)
        or receipt.get("formal_root") != D8_0_REBIND_FORMAL_ROOT
        or receipt.get("ledger_transition", {}).get("after_shard_state")
        != "completed"
        or receipt.get("ledger_transition", {}).get("prior_shard_states")
        != ["completed"]
        or receipt.get("ledger_transition", {}).get("later_shard_states")
        != ["pending"] * 6
        or receipt.get("authority", {}).get("continuation_authority_commit")
        != _git(repo, "rev-parse", f"{D8_CONTINUATION_AUTHORITY_REF}^{{commit}}")
        or receipt.get("authority", {}).get(
            "continuation_authorization_canonical_digest"
        ) != prior_authority["canonical_digest"]
    ):
        raise StageDD7Error(
            "D8-S01 receipt is not a completed generation-760 continuation anchor"
        )
    if (
        completion.get("document_type") != "stage_d_d8_s01_completion"
        or completion.get("subtask_status") != "SUBTASK_PASS"
        or completion.get("canonical_digest") != _semantic_digest(completion)
        or completion.get("receipt", {}).get("receipt_canonical_digest")
        != receipt["canonical_digest"]
        or completion.get("continuation", {}).get("next_task") != "D8-S02"
        or completion.get("continuation", {}).get(
            "shard_continuation_authorized"
        ) is not False
    ):
        raise StageDD7Error(
            "D8-S01 completion does not require a new shard-2 authority"
        )

    ledger = receipt["ledger_transition"]
    authority_commit = _git(
        repo, "rev-parse", f"{D8_CONTINUATION_AUTHORITY_REF}^{{commit}}",
    )
    document = {
        "$schema": "stage_d_d8_continuation_authorization_r2.schema.json",
        "schema_version": 1,
        "document_type": D8_CONTINUATION_AUTHORIZATION_DOCUMENT_TYPE,
        "authorization_id": D8_CONTINUATION_R2_ID,
        "status": "PASS",
        "producer": producer,
        "control_plane_code_binding": {
            "commit": producer["commit"], "tree": producer["tree"],
            "clean": True,
        },
        "persisted_execution_context": {
            "authorization_path": D8_0_REBIND_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(rebind_raw).hexdigest(),
            "canonical_digest": rebind["canonical_digest"],
            "authority_ref": D8_0_REBIND_AUTHORITY_REF,
            "authority_commit": _git(
                repo, "rev-parse", f"{D8_0_REBIND_AUTHORITY_REF}^{{commit}}",
            ),
            "code_binding": rebind["code_binding"],
        },
        "prior_authority": {
            "authorization_path": D8_CONTINUATION_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(prior_authority_raw).hexdigest(),
            "canonical_digest": prior_authority["canonical_digest"],
            "producer_ref": D8_CONTINUATION_PRODUCER_REF,
            "producer_commit": _git(
                repo, "rev-parse", f"{D8_CONTINUATION_PRODUCER_REF}^{{commit}}",
            ),
            "authority_ref": D8_CONTINUATION_AUTHORITY_REF,
            "authority_commit": authority_commit,
            "authorized_shard": 1,
        },
        "formal_root": D8_0_REBIND_FORMAL_ROOT,
        "prior_shard_anchor": {
            "shard_index": 1,
            "receipt_path": D8_S01_RECEIPT_PATH,
            "receipt_file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_canonical_digest": receipt["canonical_digest"],
            "completion_path": D8_S01_COMPLETION_PATH,
            "completion_file_sha256": hashlib.sha256(completion_raw).hexdigest(),
            "completion_canonical_digest": completion["canonical_digest"],
            "ledger_after_digest": ledger["after_canonical_digest"],
            "ledger_after_generation": ledger["after_generation"],
        },
        "checkpoint_anchor": {
            "canonical_digest": ledger["after_canonical_digest"],
            "generation": ledger["after_generation"],
            "shard_states": ["completed", "completed"] + ["pending"] * 6,
            "active_shard_leases": 0,
            "active_group_claims": 0,
            "authorized_shard_attempt_count": 0,
        },
        "scope_preservation": {
            "persisted_execution_context_unchanged": True,
            "access_journal_live_assets_and_authority_checks_unchanged": True,
            "scientific_scope_changed": False,
            "existing_s00_s01_evidence_rewritten": False,
        },
        "authorized_next_shard": 2,
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "d9_started": False,
        "d10_started": False,
        "prohibitions": {
            "rerun_prior_shards_allowed": False,
            "run_s03_or_later_allowed": False,
            "formal_aggregation_authorized": False,
            "analysis_authorization_issued": False,
            "c10_status_rewrite_allowed": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(
        document, repo / _D8_CONTINUATION_R2_AUTH_SCHEMA,
        "D8 continuation R2 authorization",
    )
    if require_current_checkpoint:
        _validate_d8_continuation_r2_checkpoint_anchor(document, repo=repo)
    return document


def _validate_d8_continuation_r2_authorization(
    source: Mapping[str, Any], raw: bytes, *, repo: Path,
    require_authority_ref: bool,
) -> Mapping[str, Any]:
    _schema_validate(
        source, repo / _D8_CONTINUATION_R2_AUTH_SCHEMA,
        "D8 continuation R2 authorization",
    )
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D8 continuation R2 authorization digest mismatch")
    expected = build_d8_continuation_r2_authorization(
        repo_root=repo, require_current_head=False,
        require_clean_worktree=False, require_current_checkpoint=False,
    )
    if source != expected:
        raise StageDD7Error(
            "D8 continuation R2 authorization is not source-recomputable"
        )
    if require_authority_ref:
        authority = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R2_AUTHORITY_REF}^{{commit}}",
        )
        parent = _git(repo, "rev-parse", f"{authority}^")
        producer = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R2_PRODUCER_REF}^{{commit}}",
        )
        changed = set(_git(repo, "diff", "--name-only", parent, authority).splitlines())
        if parent != producer or changed != _D8_CONTINUATION_R2_AUTHORITY_ONLY_PATHS:
            raise StageDD7Error(
                "D8 continuation R2 authority is not an authority-only direct child"
            )
        prefix = _git(repo, "rev-parse", "--show-prefix")
        blob = _git(
            repo, "rev-parse",
            f"{authority}:{prefix}{D8_CONTINUATION_R2_AUTHORIZATION_PATH}",
        )
        expected_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != expected_blob:
            raise StageDD7Error(
                "D8 continuation R2 authority does not contain the authorization bytes"
            )
    return source["persisted_execution_context"]


def write_d8_continuation_r2_authorization(
    *, repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d8_continuation_r2_authorization(repo_root=repo)
    _write_atomic(repo / D8_CONTINUATION_R2_AUTHORIZATION_PATH, document)
    return document


def _validate_d8_continuation_r3_checkpoint_anchor(
    source: Mapping[str, Any], *, repo: Path,
) -> None:
    """Require the live root to remain at the exact pre-S03 handoff."""

    checkpoint, _raw = _read_regular_json(
        repo / D8_0_REBIND_FORMAL_ROOT / "runtime_checkpoint.json",
        "D8 generation-1140 runtime checkpoint",
    )
    anchor = source["checkpoint_anchor"]
    shards = checkpoint.get("shards")
    units = checkpoint.get("units")
    groups = checkpoint.get("groups")
    lifecycle = checkpoint.get("lifecycle")
    if (
        checkpoint.get("canonical_digest") != _semantic_digest(checkpoint)
        or checkpoint.get("canonical_digest") != anchor["canonical_digest"]
        or checkpoint.get("generation") != anchor["generation"]
        or not isinstance(shards, list)
        or [row.get("state") for row in shards] != anchor["shard_states"]
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("active_shard_leases")
        != anchor["active_shard_leases"]
        or lifecycle.get("active_group_claims")
        != anchor["active_group_claims"]
        or not isinstance(groups, list)
        or not isinstance(units, list)
    ):
        raise StageDD7Error(
            "D8-S03 checkpoint drifted from the authorized generation-1140 handoff"
        )
    shard_groups = [
        row for row in groups if row.get("selection_order", -1) % DEFAULT_SHARD_COUNT == 3
    ]
    shard_units = [
        row for row in units
        if row.get("group_selection_order", -1) % DEFAULT_SHARD_COUNT == 3
    ]
    if (
        len(shard_groups) != 63
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_groups
        )
        or len(shard_units) != 126
        or sum(len(row.get("attempts", [])) for row in shard_units)
        != anchor["authorized_shard_attempt_count"]
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_units
        )
    ):
        raise StageDD7Error(
            "D8-S03 coordinates are not pristine at the authorized checkpoint"
        )


def build_d8_continuation_r3_authorization(
    *, repo_root: Path | str = REPO_ROOT, require_current_head: bool = True,
    require_clean_worktree: bool = True, require_current_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Roll the immutable S02 receipt into an exact shard-3 authority."""

    repo = Path(repo_root).resolve()
    if (
        require_clean_worktree
        and _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise StageDD7Error(
            "D8 continuation R3 authorization requires a clean worktree"
        )
    producer = _producer_state(
        repo, producer_ref=D8_CONTINUATION_R3_PRODUCER_REF,
    )
    if require_current_head and _git(repo, "rev-parse", "HEAD") != producer["commit"]:
        raise StageDD7Error(
            "D8 continuation R3 authorization must be generated at the frozen producer"
        )

    rebind, rebind_raw = _read_regular_json(
        repo / D8_0_REBIND_AUTHORIZATION_PATH,
        "D8.0 fresh-root authorization",
    )
    _validate_d8_0_fresh_root_authorization(
        rebind, rebind_raw, repo=repo, require_fresh_root=False,
        require_authority_ref=True,
    )
    prior_authority, prior_authority_raw = _read_regular_json(
        repo / D8_CONTINUATION_R2_AUTHORIZATION_PATH,
        "D8 continuation R2 authorization",
    )
    _validate_d8_continuation_r2_authorization(
        prior_authority, prior_authority_raw, repo=repo,
        require_authority_ref=True,
    )
    receipt, receipt_raw = _read_regular_json(
        repo / D8_S02_RECEIPT_PATH, "D8-S02 shard receipt",
    )
    completion, completion_raw = _read_regular_json(
        repo / D8_S02_COMPLETION_PATH, "D8-S02 completion record",
    )
    if (
        receipt.get("document_type") != "stage_d_d8_shard_receipt"
        or receipt.get("status") != "PASS"
        or receipt.get("subtask") != "D8-S02"
        or receipt.get("canonical_digest") != _semantic_digest(receipt)
        or receipt.get("formal_root") != D8_0_REBIND_FORMAL_ROOT
        or receipt.get("ledger_transition", {}).get("after_shard_state")
        != "completed"
        or receipt.get("ledger_transition", {}).get("prior_shard_states")
        != ["completed", "completed"]
        or receipt.get("ledger_transition", {}).get("later_shard_states")
        != ["pending"] * 5
        or receipt.get("authority", {}).get("continuation_authority_commit")
        != _git(repo, "rev-parse", f"{D8_CONTINUATION_R2_AUTHORITY_REF}^{{commit}}")
        or receipt.get("authority", {}).get(
            "continuation_authorization_canonical_digest"
        ) != prior_authority["canonical_digest"]
    ):
        raise StageDD7Error(
            "D8-S02 receipt is not a completed generation-1140 continuation anchor"
        )
    if (
        completion.get("document_type") != "stage_d_d8_s02_completion"
        or completion.get("subtask_status") != "SUBTASK_PASS"
        or completion.get("canonical_digest") != _semantic_digest(completion)
        or completion.get("receipt", {}).get("receipt_canonical_digest")
        != receipt["canonical_digest"]
        or completion.get("continuation", {}).get("next_task") != "D8-S03"
        or completion.get("continuation", {}).get(
            "shard_continuation_authorized"
        ) is not False
    ):
        raise StageDD7Error(
            "D8-S02 completion does not require a new shard-3 authority"
        )

    ledger = receipt["ledger_transition"]
    authority_commit = _git(
        repo, "rev-parse", f"{D8_CONTINUATION_R2_AUTHORITY_REF}^{{commit}}",
    )
    document = {
        "$schema": "stage_d_d8_continuation_authorization_r3.schema.json",
        "schema_version": 1,
        "document_type": D8_CONTINUATION_AUTHORIZATION_DOCUMENT_TYPE,
        "authorization_id": D8_CONTINUATION_R3_ID,
        "status": "PASS",
        "producer": producer,
        "control_plane_code_binding": {
            "commit": producer["commit"], "tree": producer["tree"],
            "clean": True,
        },
        "persisted_execution_context": {
            "authorization_path": D8_0_REBIND_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(rebind_raw).hexdigest(),
            "canonical_digest": rebind["canonical_digest"],
            "authority_ref": D8_0_REBIND_AUTHORITY_REF,
            "authority_commit": _git(
                repo, "rev-parse", f"{D8_0_REBIND_AUTHORITY_REF}^{{commit}}",
            ),
            "code_binding": rebind["code_binding"],
        },
        "prior_authority": {
            "authorization_path": D8_CONTINUATION_R2_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(prior_authority_raw).hexdigest(),
            "canonical_digest": prior_authority["canonical_digest"],
            "producer_ref": D8_CONTINUATION_R2_PRODUCER_REF,
            "producer_commit": _git(
                repo, "rev-parse", f"{D8_CONTINUATION_R2_PRODUCER_REF}^{{commit}}",
            ),
            "authority_ref": D8_CONTINUATION_R2_AUTHORITY_REF,
            "authority_commit": authority_commit,
            "authorized_shard": 2,
        },
        "formal_root": D8_0_REBIND_FORMAL_ROOT,
        "prior_shard_anchor": {
            "shard_index": 2,
            "receipt_path": D8_S02_RECEIPT_PATH,
            "receipt_file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_canonical_digest": receipt["canonical_digest"],
            "completion_path": D8_S02_COMPLETION_PATH,
            "completion_file_sha256": hashlib.sha256(completion_raw).hexdigest(),
            "completion_canonical_digest": completion["canonical_digest"],
            "ledger_after_digest": ledger["after_canonical_digest"],
            "ledger_after_generation": ledger["after_generation"],
        },
        "checkpoint_anchor": {
            "canonical_digest": ledger["after_canonical_digest"],
            "generation": ledger["after_generation"],
            "shard_states": ["completed"] * 3 + ["pending"] * 5,
            "active_shard_leases": 0,
            "active_group_claims": 0,
            "authorized_shard_attempt_count": 0,
        },
        "scope_preservation": {
            "persisted_execution_context_unchanged": True,
            "access_journal_live_assets_and_authority_checks_unchanged": True,
            "scientific_scope_changed": False,
            "existing_s00_s02_evidence_rewritten": False,
        },
        "authorized_next_shard": 3,
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "d9_started": False,
        "d10_started": False,
        "prohibitions": {
            "rerun_prior_shards_allowed": False,
            "run_s04_or_later_allowed": False,
            "formal_aggregation_authorized": False,
            "analysis_authorization_issued": False,
            "c10_status_rewrite_allowed": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(
        document, repo / _D8_CONTINUATION_R3_AUTH_SCHEMA,
        "D8 continuation R3 authorization",
    )
    if require_current_checkpoint:
        _validate_d8_continuation_r3_checkpoint_anchor(document, repo=repo)
    return document


def _validate_d8_continuation_r3_authorization(
    source: Mapping[str, Any], raw: bytes, *, repo: Path,
    require_authority_ref: bool,
) -> Mapping[str, Any]:
    _schema_validate(
        source, repo / _D8_CONTINUATION_R3_AUTH_SCHEMA,
        "D8 continuation R3 authorization",
    )
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D8 continuation R3 authorization digest mismatch")
    expected = build_d8_continuation_r3_authorization(
        repo_root=repo, require_current_head=False,
        require_clean_worktree=False, require_current_checkpoint=False,
    )
    if source != expected:
        raise StageDD7Error(
            "D8 continuation R3 authorization is not source-recomputable"
        )
    if require_authority_ref:
        authority = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R3_AUTHORITY_REF}^{{commit}}",
        )
        parent = _git(repo, "rev-parse", f"{authority}^")
        producer = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R3_PRODUCER_REF}^{{commit}}",
        )
        changed = set(_git(repo, "diff", "--name-only", parent, authority).splitlines())
        if parent != producer or changed != _D8_CONTINUATION_R3_AUTHORITY_ONLY_PATHS:
            raise StageDD7Error(
                "D8 continuation R3 authority is not an authority-only direct child"
            )
        prefix = _git(repo, "rev-parse", "--show-prefix")
        blob = _git(
            repo, "rev-parse",
            f"{authority}:{prefix}{D8_CONTINUATION_R3_AUTHORIZATION_PATH}",
        )
        expected_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != expected_blob:
            raise StageDD7Error(
                "D8 continuation R3 authority does not contain the authorization bytes"
            )
    return source["persisted_execution_context"]


def write_d8_continuation_r3_authorization(
    *, repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d8_continuation_r3_authorization(repo_root=repo)
    _write_atomic(repo / D8_CONTINUATION_R3_AUTHORIZATION_PATH, document)
    return document


def _validate_d8_continuation_r4_checkpoint_anchor(
    source: Mapping[str, Any], *, repo: Path,
) -> None:
    """Require the live root to remain at the exact pre-S04 handoff."""

    checkpoint, _raw = _read_regular_json(
        repo / D8_0_REBIND_FORMAL_ROOT / "runtime_checkpoint.json",
        "D8 generation-1520 runtime checkpoint",
    )
    anchor = source["checkpoint_anchor"]
    shards = checkpoint.get("shards")
    units = checkpoint.get("units")
    groups = checkpoint.get("groups")
    lifecycle = checkpoint.get("lifecycle")
    if (
        checkpoint.get("canonical_digest") != _semantic_digest(checkpoint)
        or checkpoint.get("canonical_digest") != anchor["canonical_digest"]
        or checkpoint.get("generation") != anchor["generation"]
        or not isinstance(shards, list)
        or [row.get("state") for row in shards] != anchor["shard_states"]
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("active_shard_leases")
        != anchor["active_shard_leases"]
        or lifecycle.get("active_group_claims")
        != anchor["active_group_claims"]
        or not isinstance(groups, list)
        or not isinstance(units, list)
    ):
        raise StageDD7Error(
            "D8-S04 checkpoint drifted from the authorized generation-1520 handoff"
        )
    shard_groups = [
        row for row in groups if row.get("selection_order", -1) % DEFAULT_SHARD_COUNT == 4
    ]
    shard_units = [
        row for row in units
        if row.get("group_selection_order", -1) % DEFAULT_SHARD_COUNT == 4
    ]
    if (
        len(shard_groups) != 62
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_groups
        )
        or len(shard_units) != 124
        or sum(len(row.get("attempts", [])) for row in shard_units)
        != anchor["authorized_shard_attempt_count"]
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_units
        )
    ):
        raise StageDD7Error(
            "D8-S04 coordinates are not pristine at the authorized checkpoint"
        )


def build_d8_continuation_r4_authorization(
    *, repo_root: Path | str = REPO_ROOT, require_current_head: bool = True,
    require_clean_worktree: bool = True, require_current_checkpoint: bool = True,
    ) -> Mapping[str, Any]:
    """Roll the immutable S03 receipt into an exact shard-4 authority."""

    repo = Path(repo_root).resolve()
    if (
        require_clean_worktree
        and _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise StageDD7Error(
            "D8 continuation R4 authorization requires a clean worktree"
        )
    producer = _producer_state(
        repo, producer_ref=D8_CONTINUATION_R4_PRODUCER_REF,
    )
    if require_current_head and _git(repo, "rev-parse", "HEAD") != producer["commit"]:
        raise StageDD7Error(
            "D8 continuation R4 authorization must be generated at the frozen producer"
        )

    rebind, rebind_raw = _read_regular_json(
        repo / D8_0_REBIND_AUTHORIZATION_PATH,
        "D8.0 fresh-root authorization",
    )
    _validate_d8_0_fresh_root_authorization(
        rebind, rebind_raw, repo=repo, require_fresh_root=False,
        require_authority_ref=True,
    )
    prior_authority, prior_authority_raw = _read_regular_json(
        repo / D8_CONTINUATION_R3_AUTHORIZATION_PATH,
        "D8 continuation R3 authorization",
    )
    _validate_d8_continuation_r3_authorization(
        prior_authority, prior_authority_raw, repo=repo,
        require_authority_ref=True,
    )
    receipt, receipt_raw = _read_regular_json(
        repo / D8_S03_RECEIPT_PATH, "D8-S03 shard receipt",
    )
    completion, completion_raw = _read_regular_json(
        repo / D8_S03_COMPLETION_PATH, "D8-S03 completion record",
    )
    if (
        receipt.get("document_type") != "stage_d_d8_shard_receipt"
        or receipt.get("status") != "PASS"
        or receipt.get("subtask") != "D8-S03"
        or receipt.get("canonical_digest") != _semantic_digest(receipt)
        or receipt.get("formal_root") != D8_0_REBIND_FORMAL_ROOT
        or receipt.get("ledger_transition", {}).get("after_shard_state")
        != "completed"
        or receipt.get("ledger_transition", {}).get("prior_shard_states")
        != ["completed", "completed", "completed"]
        or receipt.get("ledger_transition", {}).get("later_shard_states")
        != ["pending"] * 4
        or receipt.get("authority", {}).get("continuation_authority_commit")
        != _git(repo, "rev-parse", f"{D8_CONTINUATION_R3_AUTHORITY_REF}^{{commit}}")
        or receipt.get("authority", {}).get(
            "continuation_authorization_canonical_digest"
        ) != prior_authority["canonical_digest"]
    ):
        raise StageDD7Error(
            "D8-S03 receipt is not a completed generation-1520 continuation anchor"
        )
    if (
        completion.get("document_type") != "stage_d_d8_s03_completion"
        or completion.get("subtask_status") != "SUBTASK_PASS"
        or completion.get("canonical_digest") != _semantic_digest(completion)
        or completion.get("receipt", {}).get("receipt_canonical_digest")
        != receipt["canonical_digest"]
        or completion.get("continuation", {}).get("next_task") != "D8-S04"
        or completion.get("continuation", {}).get(
            "shard_continuation_authorized"
        ) is not False
    ):
        raise StageDD7Error(
            "D8-S03 completion does not require a new shard-4 authority"
        )

    ledger = receipt["ledger_transition"]
    authority_commit = _git(
        repo, "rev-parse", f"{D8_CONTINUATION_R3_AUTHORITY_REF}^{{commit}}",
    )
    document = {
        "$schema": "stage_d_d8_continuation_authorization_r4.schema.json",
        "schema_version": 1,
        "document_type": D8_CONTINUATION_AUTHORIZATION_DOCUMENT_TYPE,
        "authorization_id": D8_CONTINUATION_R4_ID,
        "status": "PASS",
        "producer": producer,
        "control_plane_code_binding": {
            "commit": producer["commit"], "tree": producer["tree"],
            "clean": True,
        },
        "persisted_execution_context": {
            "authorization_path": D8_0_REBIND_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(rebind_raw).hexdigest(),
            "canonical_digest": rebind["canonical_digest"],
            "authority_ref": D8_0_REBIND_AUTHORITY_REF,
            "authority_commit": _git(
                repo, "rev-parse", f"{D8_0_REBIND_AUTHORITY_REF}^{{commit}}",
            ),
            "code_binding": rebind["code_binding"],
        },
        "prior_authority": {
            "authorization_path": D8_CONTINUATION_R3_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(prior_authority_raw).hexdigest(),
            "canonical_digest": prior_authority["canonical_digest"],
            "producer_ref": D8_CONTINUATION_R3_PRODUCER_REF,
            "producer_commit": _git(
                repo, "rev-parse", f"{D8_CONTINUATION_R3_PRODUCER_REF}^{{commit}}",
            ),
            "authority_ref": D8_CONTINUATION_R3_AUTHORITY_REF,
            "authority_commit": authority_commit,
            "authorized_shard": 3,
        },
        "formal_root": D8_0_REBIND_FORMAL_ROOT,
        "prior_shard_anchor": {
            "shard_index": 3,
            "receipt_path": D8_S03_RECEIPT_PATH,
            "receipt_file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_canonical_digest": receipt["canonical_digest"],
            "completion_path": D8_S03_COMPLETION_PATH,
            "completion_file_sha256": hashlib.sha256(completion_raw).hexdigest(),
            "completion_canonical_digest": completion["canonical_digest"],
            "ledger_after_digest": ledger["after_canonical_digest"],
            "ledger_after_generation": ledger["after_generation"],
        },
        "checkpoint_anchor": {
            "canonical_digest": ledger["after_canonical_digest"],
            "generation": ledger["after_generation"],
            "shard_states": ["completed"] * 4 + ["pending"] * 4,
            "active_shard_leases": 0,
            "active_group_claims": 0,
            "authorized_shard_attempt_count": 0,
        },
        "scope_preservation": {
            "persisted_execution_context_unchanged": True,
            "access_journal_live_assets_and_authority_checks_unchanged": True,
            "scientific_scope_changed": False,
            "existing_s00_s03_evidence_rewritten": False,
        },
        "authorized_next_shard": 4,
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "d9_started": False,
        "d10_started": False,
        "prohibitions": {
            "rerun_prior_shards_allowed": False,
            "run_s05_or_later_allowed": False,
            "formal_aggregation_authorized": False,
            "analysis_authorization_issued": False,
            "c10_status_rewrite_allowed": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(
        document, repo / _D8_CONTINUATION_R4_AUTH_SCHEMA,
        "D8 continuation R4 authorization",
    )
    if require_current_checkpoint:
        _validate_d8_continuation_r4_checkpoint_anchor(document, repo=repo)
    return document


def _validate_d8_continuation_r4_authorization(
    source: Mapping[str, Any], raw: bytes, *, repo: Path,
    require_authority_ref: bool,
) -> Mapping[str, Any]:
    _schema_validate(
        source, repo / _D8_CONTINUATION_R4_AUTH_SCHEMA,
        "D8 continuation R4 authorization",
    )
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D8 continuation R4 authorization digest mismatch")
    expected = build_d8_continuation_r4_authorization(
        repo_root=repo, require_current_head=False,
        require_clean_worktree=False, require_current_checkpoint=False,
    )
    if source != expected:
        raise StageDD7Error(
            "D8 continuation R4 authorization is not source-recomputable"
        )
    if require_authority_ref:
        authority = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R4_AUTHORITY_REF}^{{commit}}",
        )
        parent = _git(repo, "rev-parse", f"{authority}^")
        producer = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R4_PRODUCER_REF}^{{commit}}",
        )
        changed = set(_git(repo, "diff", "--name-only", parent, authority).splitlines())
        if parent != producer or changed != _D8_CONTINUATION_R4_AUTHORITY_ONLY_PATHS:
            raise StageDD7Error(
                "D8 continuation R4 authority is not an authority-only direct child"
            )
        prefix = _git(repo, "rev-parse", "--show-prefix")
        blob = _git(
            repo, "rev-parse",
            f"{authority}:{prefix}{D8_CONTINUATION_R4_AUTHORIZATION_PATH}",
        )
        expected_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != expected_blob:
            raise StageDD7Error(
                "D8 continuation R4 authority does not contain the authorization bytes"
            )
    return source["persisted_execution_context"]


def write_d8_continuation_r4_authorization(
    *, repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d8_continuation_r4_authorization(repo_root=repo)
    _write_atomic(repo / D8_CONTINUATION_R4_AUTHORIZATION_PATH, document)
    return document


def _validate_d8_continuation_r5_checkpoint_anchor(
    source: Mapping[str, Any], *, repo: Path,
) -> None:
    """Require the live root to remain at the exact pre-S05 handoff."""

    checkpoint, _raw = _read_regular_json(
        repo / D8_0_REBIND_FORMAL_ROOT / "runtime_checkpoint.json",
        "D8 generation-1894 runtime checkpoint",
    )
    anchor = source["checkpoint_anchor"]
    shards = checkpoint.get("shards")
    units = checkpoint.get("units")
    groups = checkpoint.get("groups")
    lifecycle = checkpoint.get("lifecycle")
    if (
        checkpoint.get("canonical_digest") != _semantic_digest(checkpoint)
        or checkpoint.get("canonical_digest") != anchor["canonical_digest"]
        or checkpoint.get("generation") != anchor["generation"]
        or not isinstance(shards, list)
        or [row.get("state") for row in shards] != anchor["shard_states"]
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("active_shard_leases")
        != anchor["active_shard_leases"]
        or lifecycle.get("active_group_claims")
        != anchor["active_group_claims"]
        or not isinstance(groups, list)
        or not isinstance(units, list)
    ):
        raise StageDD7Error(
            "D8-S05 checkpoint drifted from the authorized generation-1894 handoff"
        )
    shard_groups = [
        row for row in groups if row.get("selection_order", -1) % DEFAULT_SHARD_COUNT == 5
    ]
    shard_units = [
        row for row in units
        if row.get("group_selection_order", -1) % DEFAULT_SHARD_COUNT == 5
    ]
    if (
        len(shard_groups) != 62
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_groups
        )
        or len(shard_units) != 124
        or sum(len(row.get("attempts", [])) for row in shard_units)
        != anchor["authorized_shard_attempt_count"]
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_units
        )
    ):
        raise StageDD7Error(
            "D8-S05 coordinates are not pristine at the authorized checkpoint"
        )


def build_d8_continuation_r5_authorization(
    *, repo_root: Path | str = REPO_ROOT, require_current_head: bool = True,
    require_clean_worktree: bool = True, require_current_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Roll the immutable S04 receipt into an exact shard-5 authority."""

    repo = Path(repo_root).resolve()
    if (
        require_clean_worktree
        and _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise StageDD7Error(
            "D8 continuation R5 authorization requires a clean worktree"
        )
    producer = _producer_state(
        repo, producer_ref=D8_CONTINUATION_R5_PRODUCER_REF,
    )
    if require_current_head and _git(repo, "rev-parse", "HEAD") != producer["commit"]:
        raise StageDD7Error(
            "D8 continuation R5 authorization must be generated at the frozen producer"
        )

    rebind, rebind_raw = _read_regular_json(
        repo / D8_0_REBIND_AUTHORIZATION_PATH,
        "D8.0 fresh-root authorization",
    )
    _validate_d8_0_fresh_root_authorization(
        rebind, rebind_raw, repo=repo, require_fresh_root=False,
        require_authority_ref=True,
    )
    prior_authority, prior_authority_raw = _read_regular_json(
        repo / D8_CONTINUATION_R4_AUTHORIZATION_PATH,
        "D8 continuation R4 authorization",
    )
    _validate_d8_continuation_r4_authorization(
        prior_authority, prior_authority_raw, repo=repo,
        require_authority_ref=True,
    )
    receipt, receipt_raw = _read_regular_json(
        repo / D8_S04_RECEIPT_PATH, "D8-S04 shard receipt",
    )
    completion, completion_raw = _read_regular_json(
        repo / D8_S04_COMPLETION_PATH, "D8-S04 completion record",
    )
    if (
        receipt.get("document_type") != "stage_d_d8_shard_receipt"
        or receipt.get("status") != "PASS"
        or receipt.get("subtask") != "D8-S04"
        or receipt.get("canonical_digest") != _semantic_digest(receipt)
        or receipt.get("formal_root") != D8_0_REBIND_FORMAL_ROOT
        or receipt.get("ledger_transition", {}).get("after_shard_state")
        != "completed"
        or receipt.get("ledger_transition", {}).get("prior_shard_states")
        != ["completed", "completed", "completed", "completed"]
        or receipt.get("ledger_transition", {}).get("later_shard_states")
        != ["pending"] * 3
        or receipt.get("authority", {}).get("continuation_authority_commit")
        != _git(repo, "rev-parse", f"{D8_CONTINUATION_R4_AUTHORITY_REF}^{{commit}}")
        or receipt.get("authority", {}).get(
            "continuation_authorization_canonical_digest"
        ) != prior_authority["canonical_digest"]
    ):
        raise StageDD7Error(
            "D8-S04 receipt is not a completed generation-1894 continuation anchor"
        )
    if (
        completion.get("document_type") != "stage_d_d8_s04_completion"
        or completion.get("subtask_status") != "SUBTASK_PASS"
        or completion.get("canonical_digest") != _semantic_digest(completion)
        or completion.get("receipt", {}).get("receipt_canonical_digest")
        != receipt["canonical_digest"]
        or completion.get("continuation", {}).get("next_task") != "D8-S05"
        or completion.get("continuation", {}).get(
            "shard_continuation_authorized"
        ) is not False
    ):
        raise StageDD7Error(
            "D8-S04 completion does not require a new shard-5 authority"
        )

    ledger = receipt["ledger_transition"]
    authority_commit = _git(
        repo, "rev-parse", f"{D8_CONTINUATION_R4_AUTHORITY_REF}^{{commit}}",
    )
    document = {
        "$schema": "stage_d_d8_continuation_authorization_r5.schema.json",
        "schema_version": 1,
        "document_type": D8_CONTINUATION_AUTHORIZATION_DOCUMENT_TYPE,
        "authorization_id": D8_CONTINUATION_R5_ID,
        "status": "PASS",
        "producer": producer,
        "control_plane_code_binding": {
            "commit": producer["commit"], "tree": producer["tree"],
            "clean": True,
        },
        "persisted_execution_context": {
            "authorization_path": D8_0_REBIND_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(rebind_raw).hexdigest(),
            "canonical_digest": rebind["canonical_digest"],
            "authority_ref": D8_0_REBIND_AUTHORITY_REF,
            "authority_commit": _git(
                repo, "rev-parse", f"{D8_0_REBIND_AUTHORITY_REF}^{{commit}}",
            ),
            "code_binding": rebind["code_binding"],
        },
        "prior_authority": {
            "authorization_path": D8_CONTINUATION_R4_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(prior_authority_raw).hexdigest(),
            "canonical_digest": prior_authority["canonical_digest"],
            "producer_ref": D8_CONTINUATION_R4_PRODUCER_REF,
            "producer_commit": _git(
                repo, "rev-parse", f"{D8_CONTINUATION_R4_PRODUCER_REF}^{{commit}}",
            ),
            "authority_ref": D8_CONTINUATION_R4_AUTHORITY_REF,
            "authority_commit": authority_commit,
            "authorized_shard": 4,
        },
        "formal_root": D8_0_REBIND_FORMAL_ROOT,
        "prior_shard_anchor": {
            "shard_index": 4,
            "receipt_path": D8_S04_RECEIPT_PATH,
            "receipt_file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_canonical_digest": receipt["canonical_digest"],
            "completion_path": D8_S04_COMPLETION_PATH,
            "completion_file_sha256": hashlib.sha256(completion_raw).hexdigest(),
            "completion_canonical_digest": completion["canonical_digest"],
            "ledger_after_digest": ledger["after_canonical_digest"],
            "ledger_after_generation": ledger["after_generation"],
        },
        "checkpoint_anchor": {
            "canonical_digest": ledger["after_canonical_digest"],
            "generation": ledger["after_generation"],
            "shard_states": ["completed"] * 5 + ["pending"] * 3,
            "active_shard_leases": 0,
            "active_group_claims": 0,
            "authorized_shard_attempt_count": 0,
        },
        "scope_preservation": {
            "persisted_execution_context_unchanged": True,
            "access_journal_live_assets_and_authority_checks_unchanged": True,
            "scientific_scope_changed": False,
            "existing_s00_s04_evidence_rewritten": False,
        },
        "authorized_next_shard": 5,
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "d9_started": False,
        "d10_started": False,
        "prohibitions": {
            "rerun_prior_shards_allowed": False,
            "run_s06_or_later_allowed": False,
            "formal_aggregation_authorized": False,
            "analysis_authorization_issued": False,
            "c10_status_rewrite_allowed": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(
        document, repo / _D8_CONTINUATION_R5_AUTH_SCHEMA,
        "D8 continuation R5 authorization",
    )
    if require_current_checkpoint:
        _validate_d8_continuation_r5_checkpoint_anchor(document, repo=repo)
    return document


def _validate_d8_continuation_r5_authorization(
    source: Mapping[str, Any], raw: bytes, *, repo: Path,
    require_authority_ref: bool,
) -> Mapping[str, Any]:
    _schema_validate(
        source, repo / _D8_CONTINUATION_R5_AUTH_SCHEMA,
        "D8 continuation R5 authorization",
    )
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D8 continuation R5 authorization digest mismatch")
    expected = build_d8_continuation_r5_authorization(
        repo_root=repo, require_current_head=False,
        require_clean_worktree=False, require_current_checkpoint=False,
    )
    if source != expected:
        raise StageDD7Error(
            "D8 continuation R5 authorization is not source-recomputable"
        )
    if require_authority_ref:
        authority = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R5_AUTHORITY_REF}^{{commit}}",
        )
        parent = _git(repo, "rev-parse", f"{authority}^")
        producer = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R5_PRODUCER_REF}^{{commit}}",
        )
        changed = set(_git(repo, "diff", "--name-only", parent, authority).splitlines())
        if parent != producer or changed != _D8_CONTINUATION_R5_AUTHORITY_ONLY_PATHS:
            raise StageDD7Error(
                "D8 continuation R5 authority is not an authority-only direct child"
            )
        prefix = _git(repo, "rev-parse", "--show-prefix")
        blob = _git(
            repo, "rev-parse",
            f"{authority}:{prefix}{D8_CONTINUATION_R5_AUTHORIZATION_PATH}",
        )
        expected_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != expected_blob:
            raise StageDD7Error(
                "D8 continuation R5 authority does not contain the authorization bytes"
            )
    return source["persisted_execution_context"]


def write_d8_continuation_r5_authorization(
    *, repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d8_continuation_r5_authorization(repo_root=repo)
    _write_atomic(repo / D8_CONTINUATION_R5_AUTHORIZATION_PATH, document)
    return document


def _validate_d8_continuation_r6_checkpoint_anchor(
    source: Mapping[str, Any], *, repo: Path,
) -> None:
    """Require the live root to remain at the exact pre-S06 handoff."""

    checkpoint, _raw = _read_regular_json(
        repo / D8_0_REBIND_FORMAL_ROOT / "runtime_checkpoint.json",
        "D8 generation-2268 runtime checkpoint",
    )
    anchor = source["checkpoint_anchor"]
    shards = checkpoint.get("shards")
    units = checkpoint.get("units")
    groups = checkpoint.get("groups")
    lifecycle = checkpoint.get("lifecycle")
    if (
        checkpoint.get("canonical_digest") != _semantic_digest(checkpoint)
        or checkpoint.get("canonical_digest") != anchor["canonical_digest"]
        or checkpoint.get("generation") != anchor["generation"]
        or not isinstance(shards, list)
        or [row.get("state") for row in shards] != anchor["shard_states"]
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("active_shard_leases")
        != anchor["active_shard_leases"]
        or lifecycle.get("active_group_claims")
        != anchor["active_group_claims"]
        or not isinstance(groups, list)
        or not isinstance(units, list)
    ):
        raise StageDD7Error(
            "D8-S06 checkpoint drifted from the authorized generation-2268 handoff"
        )
    shard_groups = [
        row for row in groups if row.get("selection_order", -1) % DEFAULT_SHARD_COUNT == 6
    ]
    shard_units = [
        row for row in units
        if row.get("group_selection_order", -1) % DEFAULT_SHARD_COUNT == 6
    ]
    if (
        len(shard_groups) != 62
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_groups
        )
        or len(shard_units) != 124
        or sum(len(row.get("attempts", [])) for row in shard_units)
        != anchor["authorized_shard_attempt_count"]
        or any(
            row.get("state") != "pending"
            or row.get("terminal") is not False
            or row.get("owner_token_digest") is not None
            for row in shard_units
        )
    ):
        raise StageDD7Error(
            "D8-S06 coordinates are not pristine at the authorized checkpoint"
        )


def build_d8_continuation_r6_authorization(
    *, repo_root: Path | str = REPO_ROOT, require_current_head: bool = True,
    require_clean_worktree: bool = True, require_current_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Roll the immutable S05 receipt into an exact shard-6 authority."""

    repo = Path(repo_root).resolve()
    if (
        require_clean_worktree
        and _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise StageDD7Error(
            "D8 continuation R6 authorization requires a clean worktree"
        )
    producer = _producer_state(
        repo, producer_ref=D8_CONTINUATION_R6_PRODUCER_REF,
    )
    if require_current_head and _git(repo, "rev-parse", "HEAD") != producer["commit"]:
        raise StageDD7Error(
            "D8 continuation R6 authorization must be generated at the frozen producer"
        )

    rebind, rebind_raw = _read_regular_json(
        repo / D8_0_REBIND_AUTHORIZATION_PATH,
        "D8.0 fresh-root authorization",
    )
    _validate_d8_0_fresh_root_authorization(
        rebind, rebind_raw, repo=repo, require_fresh_root=False,
        require_authority_ref=True,
    )
    prior_authority, prior_authority_raw = _read_regular_json(
        repo / D8_CONTINUATION_R5_AUTHORIZATION_PATH,
        "D8 continuation R5 authorization",
    )
    _validate_d8_continuation_r5_authorization(
        prior_authority, prior_authority_raw, repo=repo,
        require_authority_ref=True,
    )
    receipt, receipt_raw = _read_regular_json(
        repo / D8_S05_RECEIPT_PATH, "D8-S05 shard receipt",
    )
    completion, completion_raw = _read_regular_json(
        repo / D8_S05_COMPLETION_PATH, "D8-S05 completion record",
    )
    if (
        receipt.get("document_type") != "stage_d_d8_shard_receipt"
        or receipt.get("status") != "PASS"
        or receipt.get("subtask") != "D8-S05"
        or receipt.get("canonical_digest") != _semantic_digest(receipt)
        or receipt.get("formal_root") != D8_0_REBIND_FORMAL_ROOT
        or receipt.get("ledger_transition", {}).get("after_shard_state")
        != "completed"
        or receipt.get("ledger_transition", {}).get("prior_shard_states")
        != ["completed", "completed", "completed", "completed", "completed"]
        or receipt.get("ledger_transition", {}).get("later_shard_states")
        != ["pending"] * 2
        or receipt.get("authority", {}).get("continuation_authority_commit")
        != _git(repo, "rev-parse", f"{D8_CONTINUATION_R5_AUTHORITY_REF}^{{commit}}")
        or receipt.get("authority", {}).get(
            "continuation_authorization_canonical_digest"
        ) != prior_authority["canonical_digest"]
    ):
        raise StageDD7Error(
            "D8-S05 receipt is not a completed generation-2268 continuation anchor"
        )
    if (
        completion.get("document_type") != "stage_d_d8_s05_completion"
        or completion.get("subtask_status") != "SUBTASK_PASS"
        or completion.get("canonical_digest") != _semantic_digest(completion)
        or completion.get("receipt", {}).get("receipt_canonical_digest")
        != receipt["canonical_digest"]
        or completion.get("continuation", {}).get("next_task") != "D8-S06"
        or completion.get("continuation", {}).get(
            "shard_continuation_authorized"
        ) is not False
    ):
        raise StageDD7Error(
            "D8-S05 completion does not require a new shard-6 authority"
        )

    ledger = receipt["ledger_transition"]
    authority_commit = _git(
        repo, "rev-parse", f"{D8_CONTINUATION_R5_AUTHORITY_REF}^{{commit}}",
    )
    document = {
        "$schema": "stage_d_d8_continuation_authorization_r6.schema.json",
        "schema_version": 1,
        "document_type": D8_CONTINUATION_AUTHORIZATION_DOCUMENT_TYPE,
        "authorization_id": D8_CONTINUATION_R6_ID,
        "status": "PASS",
        "producer": producer,
        "control_plane_code_binding": {
            "commit": producer["commit"], "tree": producer["tree"],
            "clean": True,
        },
        "persisted_execution_context": {
            "authorization_path": D8_0_REBIND_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(rebind_raw).hexdigest(),
            "canonical_digest": rebind["canonical_digest"],
            "authority_ref": D8_0_REBIND_AUTHORITY_REF,
            "authority_commit": _git(
                repo, "rev-parse", f"{D8_0_REBIND_AUTHORITY_REF}^{{commit}}",
            ),
            "code_binding": rebind["code_binding"],
        },
        "prior_authority": {
            "authorization_path": D8_CONTINUATION_R5_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(prior_authority_raw).hexdigest(),
            "canonical_digest": prior_authority["canonical_digest"],
            "producer_ref": D8_CONTINUATION_R5_PRODUCER_REF,
            "producer_commit": _git(
                repo, "rev-parse", f"{D8_CONTINUATION_R5_PRODUCER_REF}^{{commit}}",
            ),
            "authority_ref": D8_CONTINUATION_R5_AUTHORITY_REF,
            "authority_commit": authority_commit,
            "authorized_shard": 5,
        },
        "formal_root": D8_0_REBIND_FORMAL_ROOT,
        "prior_shard_anchor": {
            "shard_index": 5,
            "receipt_path": D8_S05_RECEIPT_PATH,
            "receipt_file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_canonical_digest": receipt["canonical_digest"],
            "completion_path": D8_S05_COMPLETION_PATH,
            "completion_file_sha256": hashlib.sha256(completion_raw).hexdigest(),
            "completion_canonical_digest": completion["canonical_digest"],
            "ledger_after_digest": ledger["after_canonical_digest"],
            "ledger_after_generation": ledger["after_generation"],
        },
        "checkpoint_anchor": {
            "canonical_digest": ledger["after_canonical_digest"],
            "generation": ledger["after_generation"],
            "shard_states": ["completed"] * 6 + ["pending"] * 2,
            "active_shard_leases": 0,
            "active_group_claims": 0,
            "authorized_shard_attempt_count": 0,
        },
        "scope_preservation": {
            "persisted_execution_context_unchanged": True,
            "access_journal_live_assets_and_authority_checks_unchanged": True,
            "scientific_scope_changed": False,
            "existing_s00_s05_evidence_rewritten": False,
        },
        "authorized_next_shard": 6,
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "d9_started": False,
        "d10_started": False,
        "prohibitions": {
            "rerun_prior_shards_allowed": False,
            "run_s07_or_later_allowed": False,
            "formal_aggregation_authorized": False,
            "analysis_authorization_issued": False,
            "c10_status_rewrite_allowed": False,
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    _schema_validate(
        document, repo / _D8_CONTINUATION_R6_AUTH_SCHEMA,
        "D8 continuation R6 authorization",
    )
    if require_current_checkpoint:
        _validate_d8_continuation_r6_checkpoint_anchor(document, repo=repo)
    return document


def _validate_d8_continuation_r6_authorization(
    source: Mapping[str, Any], raw: bytes, *, repo: Path,
    require_authority_ref: bool,
) -> Mapping[str, Any]:
    _schema_validate(
        source, repo / _D8_CONTINUATION_R6_AUTH_SCHEMA,
        "D8 continuation R6 authorization",
    )
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D8 continuation R6 authorization digest mismatch")
    expected = build_d8_continuation_r6_authorization(
        repo_root=repo, require_current_head=False,
        require_clean_worktree=False, require_current_checkpoint=False,
    )
    if source != expected:
        raise StageDD7Error(
            "D8 continuation R6 authorization is not source-recomputable"
        )
    if require_authority_ref:
        authority = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R6_AUTHORITY_REF}^{{commit}}",
        )
        parent = _git(repo, "rev-parse", f"{authority}^")
        producer = _git(
            repo, "rev-parse", f"{D8_CONTINUATION_R6_PRODUCER_REF}^{{commit}}",
        )
        changed = set(_git(repo, "diff", "--name-only", parent, authority).splitlines())
        if parent != producer or changed != _D8_CONTINUATION_R6_AUTHORITY_ONLY_PATHS:
            raise StageDD7Error(
                "D8 continuation R6 authority is not an authority-only direct child"
            )
        prefix = _git(repo, "rev-parse", "--show-prefix")
        blob = _git(
            repo, "rev-parse",
            f"{authority}:{prefix}{D8_CONTINUATION_R6_AUTHORIZATION_PATH}",
        )
        expected_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != expected_blob:
            raise StageDD7Error(
                "D8 continuation R6 authority does not contain the authorization bytes"
            )
    return source["persisted_execution_context"]


def write_d8_continuation_r6_authorization(
    *, repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document = build_d8_continuation_r6_authorization(repo_root=repo)
    _write_atomic(repo / D8_CONTINUATION_R6_AUTHORIZATION_PATH, document)
    return document


@dataclass(frozen=True, slots=True)
class _D7ValidatedExecutionContext:
    formal: bool
    formal_root: str
    authorization_scope: str
    execution_authorization: str
    authorization_slot: Mapping[str, Any]
    shard_count: int
    shard_mapping: Mapping[int, int]
    plan_digest: str
    budget_digest: str
    preflight_digest: str
    code_binding: Mapping[str, Any]
    dependency_binding: Mapping[str, Any]
    model_binding: Mapping[str, Any]
    source_bindings: Mapping[str, Any]
    access_attestation: Mapping[str, bool]
    canonical_digest: str
    _issuer: object = field(repr=False, compare=False)


_D7_ISSUER = object()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return copy.deepcopy(value)


def validate_execution_authorization(
    value: Mapping[str, Any] | None = None, *, repo_root: Path | str = REPO_ROOT,
    require_authority_ref: bool = True, require_fresh_root: bool = True,
) -> _D7ValidatedExecutionContext:
    repo = Path(repo_root).resolve()
    supplied = None if value is None else _plain(value)
    rebind: Mapping[str, Any] | None = None
    continuation: Mapping[str, Any] | None = None
    if (
        isinstance(supplied, Mapping)
        and supplied.get("document_type")
        == D8_CONTINUATION_AUTHORIZATION_DOCUMENT_TYPE
    ):
        authorization_id = supplied.get("authorization_id")
        is_r6 = authorization_id == D8_CONTINUATION_R6_ID
        is_r5 = authorization_id == D8_CONTINUATION_R5_ID
        is_r4 = authorization_id == D8_CONTINUATION_R4_ID
        is_r3 = authorization_id == D8_CONTINUATION_R3_ID
        is_r2 = authorization_id == D8_CONTINUATION_R2_ID
        continuation_path = (
            D8_CONTINUATION_R6_AUTHORIZATION_PATH
            if is_r6 else (
                D8_CONTINUATION_R5_AUTHORIZATION_PATH
                if is_r5 else (
                    D8_CONTINUATION_R4_AUTHORIZATION_PATH
                    if is_r4 else (
                        D8_CONTINUATION_R3_AUTHORIZATION_PATH
                        if is_r3 else (
                            D8_CONTINUATION_R2_AUTHORIZATION_PATH
                            if is_r2 else D8_CONTINUATION_AUTHORIZATION_PATH
                        )
                    )
                )
            )
        )
        continuation, continuation_raw = _read_regular_json(
            repo / continuation_path,
            ("D8 continuation R6 authorization" if is_r6 else
             "D8 continuation R5 authorization" if is_r5 else
             "D8 continuation R4 authorization" if is_r4 else
             "D8 continuation R3 authorization" if is_r3 else
             "D8 continuation R2 authorization" if is_r2 else
             "D8 continuation authorization"),
        )
        if supplied != continuation:
            raise StageDD7Error(
                "supplied D8 continuation authorization differs from tracked source"
            )
        if is_r6:
            persisted = _validate_d8_continuation_r6_authorization(
                continuation, continuation_raw, repo=repo,
                require_authority_ref=require_authority_ref,
            )
            _validate_d8_continuation_r6_checkpoint_anchor(
                continuation, repo=repo,
            )
        elif is_r5:
            persisted = _validate_d8_continuation_r5_authorization(
                continuation, continuation_raw, repo=repo,
                require_authority_ref=require_authority_ref,
            )
            _validate_d8_continuation_r5_checkpoint_anchor(
                continuation, repo=repo,
            )
        elif is_r4:
            persisted = _validate_d8_continuation_r4_authorization(
                continuation, continuation_raw, repo=repo,
                require_authority_ref=require_authority_ref,
            )
            _validate_d8_continuation_r4_checkpoint_anchor(
                continuation, repo=repo,
            )
        elif is_r3:
            persisted = _validate_d8_continuation_r3_authorization(
                continuation, continuation_raw, repo=repo,
                require_authority_ref=require_authority_ref,
            )
            _validate_d8_continuation_r3_checkpoint_anchor(
                continuation, repo=repo,
            )
        elif is_r2:
            persisted = _validate_d8_continuation_r2_authorization(
                continuation, continuation_raw, repo=repo,
                require_authority_ref=require_authority_ref,
            )
            _validate_d8_continuation_r2_checkpoint_anchor(
                continuation, repo=repo,
            )
        else:
            persisted = _validate_d8_continuation_authorization(
                continuation, continuation_raw, repo=repo,
                require_authority_ref=require_authority_ref,
            )
        rebind, rebind_raw = _read_regular_json(
            repo / D8_0_REBIND_AUTHORIZATION_PATH,
            "D8.0 fresh-root authorization",
        )
        base, formal_root = _validate_d8_0_fresh_root_authorization(
            rebind, rebind_raw, repo=repo, require_fresh_root=False,
            require_authority_ref=require_authority_ref,
        )
        if persisted != {
            "authorization_path": D8_0_REBIND_AUTHORIZATION_PATH,
            "file_sha256": hashlib.sha256(rebind_raw).hexdigest(),
            "canonical_digest": rebind["canonical_digest"],
            "authority_ref": D8_0_REBIND_AUTHORITY_REF,
            "authority_commit": _git(
                repo, "rev-parse", f"{D8_0_REBIND_AUTHORITY_REF}^{{commit}}",
            ),
            "code_binding": rebind["code_binding"],
        }:
            raise StageDD7Error(
                "D8 continuation persisted execution context drifted"
            )
    elif (
        isinstance(supplied, Mapping)
        and supplied.get("document_type")
        == D8_0_REBIND_AUTHORIZATION_DOCUMENT_TYPE
    ):
        rebind, rebind_raw = _read_regular_json(
            repo / D8_0_REBIND_AUTHORIZATION_PATH,
            "D8.0 fresh-root authorization",
        )
        if supplied != rebind:
            raise StageDD7Error(
                "supplied D8.0 fresh-root authorization differs from tracked source"
            )
        base, formal_root = _validate_d8_0_fresh_root_authorization(
            rebind, rebind_raw, repo=repo,
            require_fresh_root=require_fresh_root,
            require_authority_ref=require_authority_ref,
        )
    else:
        formal_root = None
    source, raw = _load_r4_execution_authorization(repo)
    if rebind is not None and base != {
        "path": AUTHORIZATION_PATH,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_digest": source["canonical_digest"],
        "authority_ref": AUTHORITY_REF,
        "authority_commit": _git(repo, "rev-parse", f"{AUTHORITY_REF}^{{commit}}"),
    }:
        raise StageDD7Error(
            "D8.0 fresh-root authorization base authority binding drifted"
        )
    if rebind is None:
        if supplied is not None and supplied != source:
            raise StageDD7Error("supplied D7 authorization differs from its tracked source")
        formal_root = source["formal_root"]
    _schema_validate(source, repo / _AUTH_SCHEMA, "D7 execution authorization")
    if source.get("canonical_digest") != _semantic_digest(source):
        raise StageDD7Error("D7 execution authorization digest mismatch")
    pipeline = load_recovery_pipeline(repo_root=repo)
    budget, estimate, preflight = load_d7_budget(repo_root=repo)
    producer = _producer_state(repo)
    if source["producer"] != producer or source["code_binding"] != {
        "commit": producer["commit"], "tree": producer["tree"], "clean": True,
    }:
        raise StageDD7Error("D7 authorization producer binding drifted")
    if source["plan"]["canonical_digest"] != pipeline.plan.canonical_digest:
        raise StageDD7Error("D7 authorization plan binding drifted")
    if (
        source["budget"]["canonical_digest"] != budget["canonical_digest"]
        or source["budget"]["estimate"] != estimate.to_dict()
        or source["budget"]["preflight"] != preflight.to_dict()
    ):
        raise StageDD7Error("D7 authorization budget binding drifted")
    observed_smoke = validate_full_chain_synthetic_smoke(repo_root=repo)
    if (
        source["smoke"]["canonical_digest"]
        != observed_smoke["canonical_digest"]
        or source["smoke"]["real_d4_d5_d6_api_chain"] is not True
    ):
        raise StageDD7Error("D7 authorization full-chain smoke binding drifted")
    observed_initializer = validate_d8_initializer_synthetic_smoke(repo_root=repo)
    if source["d8_initializer"] != {
        "path": ACCESS_SMOKE_RECEIPT_PATH,
        "canonical_digest": observed_initializer["canonical_digest"],
        "synthetic_only": True,
        "durable_access_started": True,
        "append_only_journal": True,
        "interruption_recovery": True,
        "failure_blocks": True,
        "all_500_required_before_runnable": True,
        "unit_execution_started": False,
    }:
        raise StageDD7Error("D7 authorization D8.0 initializer binding drifted")
    observed_model = verify_recovery_model(repo_root=repo)
    if source["model_binding"] != observed_model:
        raise StageDD7Error("D7 authorization model binding drifted")
    lock_raw = (repo / "requirements.lock.txt").read_bytes()
    if source["dependency_binding"] != {
        "path": "requirements.lock.txt",
        "lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
    }:
        raise StageDD7Error("D7 authorization dependency binding drifted")
    if source["source_bindings"] != _source_bindings(repo):
        raise StageDD7Error("D7 authorization recovery source binding drifted")
    count = source["sharding"]["shard_count"]
    mapping = dict(validate_stage_d_shard_mapping(pipeline.plan, count))
    mapping_list = [mapping[index] for index in range(500)]
    if (
        source["sharding"]["mapping_digest"] != canonical_sha256(mapping_list)
        or source["sharding"]["image_group_counts"]
        != [sum(value == index for value in mapping.values()) for index in range(count)]
    ):
        raise StageDD7Error("D7 authorization shard mapping drifted")
    if require_fresh_root and rebind is None:
        _fresh_formal_root(repo, source["formal_root"])
    if require_authority_ref:
        authority = _git(repo, "rev-parse", f"{AUTHORITY_REF}^{{commit}}")
        parent = _git(repo, "rev-parse", f"{authority}^")
        changed = set(_git(repo, "diff", "--name-only", parent, authority).splitlines())
        if parent != producer["commit"] or changed != _AUTHORITY_ONLY_PATHS:
            raise StageDD7Error("D7 authority commit is not an authority-only direct child")
        prefix = _git(repo, "rev-parse", "--show-prefix")
        blob = _git(repo, "rev-parse", f"{authority}:{prefix}{AUTHORIZATION_PATH}")
        expected_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != expected_blob:
            raise StageDD7Error("D7 authority ref does not contain the authorization bytes")
    # A continuation wrapper authorizes corrected control code but must not
    # mutate the execution-context identity persisted in the existing root.
    active_authorization_digest = (
        source["canonical_digest"] if rebind is None else rebind["canonical_digest"]
    )
    active_authority_ref = AUTHORITY_REF if rebind is None else D8_0_REBIND_AUTHORITY_REF
    slot = {
        "status": EXECUTION_AUTHORIZATION,
        "authorization_canonical_digest": active_authorization_digest,
        "authority_ref": active_authority_ref, "authority_commit": "SELF",
    }
    code_binding = source["code_binding"] if rebind is None else rebind["code_binding"]
    values = {
        "formal": True,
        "authorization_scope": source["authorization_scope"],
        "execution_authorization": EXECUTION_AUTHORIZATION,
        "authorization_slot": slot, "shard_count": count,
        "shard_mapping": mapping_list, "plan_digest": pipeline.plan.canonical_digest,
        "budget_digest": estimate.canonical_digest,
        "preflight_digest": preflight.canonical_digest,
        "code_binding": code_binding,
        "dependency_binding": source["dependency_binding"],
        "model_binding": source["model_binding"],
        "source_bindings": {
            key: source["source_bindings"][key] for key in ("d0", "d1", "d2", "c10")
        },
        "access_attestation": source["access_attestation"],
    }
    digest = canonical_sha256(values)
    return _D7ValidatedExecutionContext(
        **{**values, "formal_root": formal_root, "shard_mapping": MappingProxyType(mapping)},
        canonical_digest=digest, _issuer=_D7_ISSUER,
    )


class D7ExecutionGate:
    def __init__(self, *, repo_root: Path | str = REPO_ROOT) -> None:
        self.repo_root = Path(repo_root).resolve()

    def validate_execution_context(
        self, context: object | None, *, operation: str, output_root: Path | str,
        plan: StageDExecutionPlan, estimate: StageDEstimate, preflight: StageDPreflight,
    ) -> _D7ValidatedExecutionContext:
        if not isinstance(context, Mapping):
            raise StageDD7Error("D7 gate requires the tracked authorization document")
        if operation not in {"initialize-access", "validate-access", "run-shard"}:
            raise StageDD7Error("D7 authorization operation is outside D8 execution")
        checked = validate_execution_authorization(
            context, repo_root=self.repo_root,
            require_fresh_root=False,
        )
        expected_root = self.repo_root / checked.formal_root
        if Path(output_root).absolute() != expected_root.absolute():
            raise StageDD7Error("D7 authorization output root binding drifted")
        if (
            plan.canonical_digest != checked.plan_digest
            or estimate.canonical_digest != checked.budget_digest
            or preflight.canonical_digest != checked.preflight_digest
        ):
            raise StageDD7Error("D7 runtime controls drifted from authorization")
        return checked


def controls_from_authorization(
    authorization: Mapping[str, Any], *, repo_root: Path | str = REPO_ROOT,
) -> tuple[StageDEstimate, StageDPreflight]:
    source, _estimate, _preflight = load_d7_budget(repo_root=repo_root)
    budget = authorization.get("budget")
    if budget is None and (
        authorization.get("document_type") in {
            D8_0_REBIND_AUTHORIZATION_DOCUMENT_TYPE,
            D8_CONTINUATION_AUTHORIZATION_DOCUMENT_TYPE,
        }
    ):
        base, _ = _load_r4_execution_authorization(Path(repo_root).resolve())
        budget = base.get("budget")
    if budget is None or budget.get("canonical_digest") != source["canonical_digest"]:
        raise StageDD7Error("authorization does not bind the current D7 budget")
    return (
        validate_stage_d_estimate(budget["estimate"]),
        validate_stage_d_preflight(budget["preflight"]),
    )


def build_d7_validation(
    *, repo_root: Path | str = REPO_ROOT,
    test_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build a nonterminal independent-validation record after authority."""

    repo = Path(repo_root).resolve()
    context = validate_execution_authorization(repo_root=repo)
    source, raw = _read_regular_json(
        repo / AUTHORIZATION_PATH, "D7 execution authorization",
    )
    if not test_results or any(
        not isinstance(item, Mapping) or item.get("actual_exit_status") != 0
        for item in test_results
    ):
        raise StageDD7Error("D7 completion requires all declared validations to pass")
    authority = _git(repo, "rev-parse", f"{AUTHORITY_REF}^{{commit}}")
    document = {
        "schema_version": 1, "document_type": "stage_d_d7_validation",
        "d7_id": D7_ID, "status": "PASS_NONTERMINAL",
        "independent_validation": True, "source_recomputed": True,
        "authorization_path": AUTHORIZATION_PATH,
        "authorization_file_sha256": hashlib.sha256(raw).hexdigest(),
        "authorization_canonical_digest": source["canonical_digest"],
        "runtime_context_digest": context.canonical_digest,
        "producer_commit": source["producer"]["commit"],
        "authority_commit": authority, "authority_ref": AUTHORITY_REF,
        "tests": [_plain(item) for item in test_results],
        "errors": [], "blocking_reasons": [],
        "access_attestation": {
            "reserve_image_exists_checked": False,
            "reserve_image_statted": False, "reserve_image_opened": False,
            "reserve_image_hashed": False, "reserve_image_decoded": False,
            "reserve_image_preprocessed": False,
            "reserve_image_used_for_inference": False,
            "stage_d_outcome_accessed": False,
        },
        "authorization": {
            "execution_authorization": EXECUTION_AUTHORIZATION,
            "analysis_authorization": ANALYSIS_AUTHORIZATION,
            "reserve_access_authorization": "DEFERRED_TO_D8_0",
        },
    }
    document["canonical_digest"] = _semantic_digest(document)
    return document


def write_d7_terminal_records(
    *, repo_root: Path | str = REPO_ROOT,
    test_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Publish validation first, then atomically publish terminal D7 PASS."""

    repo = Path(repo_root).resolve()
    validation_path = repo / VALIDATION_PATH
    status_path = repo / STATUS_PATH
    handoff_path = repo / HANDOFF_PATH
    for path in (validation_path, status_path, handoff_path):
        if os.path.lexists(path):
            raise StageDD7Error(f"D7 terminal path is not fresh: {path}")
    validation = build_d7_validation(repo_root=repo, test_results=test_results)
    _write_atomic(validation_path, validation)
    validation_raw = validation_path.read_bytes()
    authorization, authorization_raw = _read_regular_json(
        repo / AUTHORIZATION_PATH, "D7 execution authorization",
    )
    status = {
        "schema_version": 1, "document_type": COMPLETION_DOCUMENT_TYPE,
        "d7_id": D7_ID, "status": "PASS",
        "pass_scope": "EXECUTION_AUTHORIZATION_GATE_PASS_NOT_EXECUTION_OR_ANALYSIS",
        "validation_complete_before_terminal": True,
        "validation_path": VALIDATION_PATH,
        "validation_file_sha256": hashlib.sha256(validation_raw).hexdigest(),
        "validation_canonical_digest": validation["canonical_digest"],
        "authorization_path": AUTHORIZATION_PATH,
        "authorization_file_sha256": hashlib.sha256(authorization_raw).hexdigest(),
        "authorization_canonical_digest": authorization["canonical_digest"],
        "producer_commit": authorization["producer"]["commit"],
        "authority_commit": validation["authority_commit"],
        "completion_commit": "SELF", "completion_ref": COMPLETION_REF,
        "d8_initialized": False, "d9_started": False, "d10_started": False,
        "scientific_result": "NOT_EVALUATED",
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "reserve_access_authorization": "DEFERRED_TO_D8_0",
        "access_attestation": validation["access_attestation"],
    }
    status["canonical_digest"] = _semantic_digest(status)
    pending = status_path.with_name("status.pending.json")
    _write_atomic(pending, status)
    # Validate exact bytes before the terminal name exists.
    staged, staged_raw = _read_regular_json(pending, "pending D7 terminal status")
    if staged != status or hashlib.sha256(staged_raw).hexdigest() != hashlib.sha256(_file_bytes(status)).hexdigest():
        raise StageDD7Error("pending D7 terminal status failed byte validation")
    os.replace(pending, status_path)
    handoff = {
        "schema_version": 1, "document_type": "stage_d_d7_to_future_d8_handoff",
        "d7_id": D7_ID, "status": "PASS",
        "d7_scope": "EXECUTION_AUTHORIZATION_GATE_PASS_NOT_SCIENTIFIC_RESULT",
        "producer_commit": authorization["producer"]["commit"],
        "authority_commit": validation["authority_commit"],
        "completion_commit": "SELF",
        "authorization_canonical_digest": authorization["canonical_digest"],
        "budget_canonical_digest": authorization["budget"]["canonical_digest"],
        "smoke_canonical_digest": authorization["smoke"]["canonical_digest"],
        "next_task": "FUTURE_D8_0_REVALIDATE_AND_INITIALIZE_ONLY",
        "d8_initialized": False, "d8_shards_started": False,
        "d9_started": False, "d10_started": False,
        "analysis_authorization": ANALYSIS_AUTHORIZATION,
        "reserve_access_this_d7": False,
        "d8_0_must_revalidate": [
            "producer_ref", "authority_ref", "authorization_bytes",
            "recovery_model", "plan", "budget", "shard_mapping",
            "fresh_formal_root", "pre_authorization_access_attestation",
        ],
        "c10_status_preserved": authorization["c10_status_preserved"],
    }
    handoff["canonical_digest"] = _semantic_digest(handoff)
    _write_atomic(handoff_path, handoff)
    return status


__all__ = [
    "ANALYSIS_AUTHORIZATION", "AUTHORIZATION_PATH", "AUTHORITY_REF",
    "HISTORICAL_AUTHORIZATION_PATH",
    "BUDGET_PATH", "COMPLETION_REF", "D7ExecutionGate", "D7_ID",
    "DEFAULT_SHARD_COUNT", "EXECUTION_AUTHORIZATION", "FORMAL_ROOT",
    "D8_0_REBIND_AUTHORIZATION_PATH", "D8_0_REBIND_AUTHORITY_REF",
    "D8_0_REBIND_FORMAL_ROOT", "D8_0_REBIND_PRODUCER_REF",
    "PRODUCER_REF", "SMOKE_RECEIPT_PATH", "SMOKE_ROOT",
    "StageDD7Error", "build_d7_budget", "build_execution_authorization",
    "build_d8_0_fresh_root_authorization",
    "controls_from_authorization", "load_d7_budget",
    "load_recovery_pipeline", "run_synthetic_smoke",
    "record_failed_smoke",
    "production_backend_factory",
    "build_d7_validation", "write_d7_terminal_records",
    "validate_archived_recovery_chain",
    "run_full_chain_synthetic_smoke",
    "validate_full_chain_synthetic_smoke",
    "validate_execution_authorization", "validate_smoke",
    "verify_recovery_model", "write_d7_budget",
    "write_d8_0_fresh_root_authorization",
    "write_d8_continuation_r6_authorization",
]
