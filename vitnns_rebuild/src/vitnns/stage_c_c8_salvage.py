"""Amendment-authorized virtual-union recovery for interrupted C8."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .recovery_c8_salvage import (
    RecoveryC8SalvageError, RecoveryC8SalvageLaunchRecord,
    _tree_inventory, load_recovery_c8_salvage_launch_record,
)
from .stage_c_artifacts import attempt_directories, validate_execution_unit
from .stage_c_c8 import (
    _artifact_activity_lock, _atomic_create_json, _atomic_replace_json,
    _checkpoint_lock, _lifecycle_lock, build_c8_measured_preflight,
    validate_c8_execution_plan, validate_fixed_c7_report,
)
from .stage_c_provenance import canonical_sha256
from .stage_c_resources import measure_artifact_tree_bytes
from .stage_c_runner import run_execution_unit

SOURCE_ORIGIN = "inherited_from_interrupted_root"
GAP_ORIGIN = "newly_run_in_salvage_root"
REJECTED_ORIGIN = "rejected_from_interrupted_root"
STATUS_FILE = "salvage_status.json"
SCOPE_FILE = "salvage_scope.json"
PLAN_FILE = "execution_plan.json"
PREFLIGHT_FILE = "preflight.json"
EXPECTED_UNIT_COUNT = 2180
SALVAGE_RULES_VERSION = "recovery-c8-salvage-v2"
_SCOPE_STATES = frozenset({
    "pending", "claimed", "incomplete", "succeeded", "failed",
})


def _plain(value: Any) -> Any:
    method = getattr(value, "to_dict", None)
    if callable(method):
        value = method()
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    raise TypeError(f"not JSON-compatible: {type(value).__name__}")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryC8SalvageError(f"{path} is not an object")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_or_accept_exact_json(
    path: Path, value: Mapping[str, Any], *, context: str,
) -> None:
    payload = (
        json.dumps(
            _plain(value), indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RecoveryC8SalvageError(
                f"existing {context} differs from source recomputation"
            )
        return
    try:
        _atomic_create_json(path, value)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RecoveryC8SalvageError(
                f"raced {context} differs from source recomputation"
            )


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryC8SalvageError(
            f"cannot capture salvage verifier Git identity: {error}"
        ) from error


def _source_subtree_git_path(repo: Path) -> str:
    top = Path(_git(repo, "rev-parse", "--show-toplevel"))
    try:
        return (repo / "src/vitnns").resolve().relative_to(top.resolve()).as_posix()
    except ValueError as error:
        raise RecoveryC8SalvageError(
            "salvage verifier source subtree escapes the Git worktree"
        ) from error


def _git_identity_for_commit(authority: Any, commit: str) -> dict[str, Any]:
    if (
        not isinstance(commit, str) or len(commit) != 40
        or set(commit) - set("0123456789abcdef")
    ):
        raise RecoveryC8SalvageError("execution commit is not lowercase 40-hex")
    repo = Path(authority.repo_root).resolve()
    subtree = _source_subtree_git_path(repo)
    if _git(repo, "rev-parse", f"{commit}^{{commit}}") != commit:
        raise RecoveryC8SalvageError("execution commit does not resolve exactly")
    return {
        "commit": commit,
        "tree": _git(repo, "rev-parse", f"{commit}^{{tree}}"),
        "source_subtree_path": "src/vitnns",
        "source_subtree_tree": _git(repo, "rev-parse", f"{commit}:{subtree}"),
        "dirty": False,
    }


def _capture_verifier_identity(authority: Any) -> dict[str, Any]:
    repo = Path(authority.repo_root).resolve()
    subtree = _source_subtree_git_path(repo)
    if _git(repo, "status", "--porcelain=v1", "--", f":(top){subtree}"):
        raise RecoveryC8SalvageError(
            "salvage verifier source subtree must be clean"
        )
    commit = _git(repo, "rev-parse", "HEAD^{commit}")
    return {
        "commit": commit,
        "tree": _git(repo, "rev-parse", f"{commit}^{{tree}}"),
        "source_subtree_path": "src/vitnns",
        "source_subtree_tree": _git(repo, "rev-parse", f"{commit}:{subtree}"),
        "dirty": False,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "rules_version": SALVAGE_RULES_VERSION,
    }


def _validate_verifier_identity(authority: Any, raw: Any) -> dict[str, Any]:
    verifier = _plain(raw)
    required = {
        "commit", "tree", "source_subtree_path", "source_subtree_tree",
        "dirty", "verified_at_utc", "rules_version",
    }
    if not isinstance(verifier, dict) or set(verifier) != required:
        raise RecoveryC8SalvageError("salvage verifier identity is malformed")
    if (
        verifier["source_subtree_path"] != "src/vitnns"
        or verifier["dirty"] is not False
        or verifier["rules_version"] != SALVAGE_RULES_VERSION
    ):
        raise RecoveryC8SalvageError("salvage verifier identity drifted")
    try:
        timestamp = str(verifier["verified_at_utc"])
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if not timestamp.endswith("Z") or parsed.tzinfo != timezone.utc:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise RecoveryC8SalvageError(
            "salvage verifier time is not canonical UTC"
        ) from error
    repo = Path(authority.repo_root).resolve()
    subtree = _source_subtree_git_path(repo)
    commit = str(verifier["commit"])
    if (
        _git(repo, "rev-parse", f"{commit}^{{commit}}") != commit
        or _git(repo, "rev-parse", f"{commit}^{{tree}}") != verifier["tree"]
        or _git(repo, "rev-parse", f"{commit}:{subtree}")
        != verifier["source_subtree_tree"]
    ):
        raise RecoveryC8SalvageError("salvage verifier Git objects drifted")
    try:
        _git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    except RecoveryC8SalvageError as error:
        raise RecoveryC8SalvageError(
            "current HEAD does not descend from the salvage verifier"
        ) from error
    if _git(repo, "rev-parse", f"HEAD:{subtree}") != verifier["source_subtree_tree"]:
        raise RecoveryC8SalvageError(
            "salvage verifier source subtree changed after verification"
        )
    return verifier


def _authority(record: RecoveryC8SalvageLaunchRecord) -> RecoveryC8SalvageLaunchRecord:
    if type(record) is not RecoveryC8SalvageLaunchRecord:
        raise RecoveryC8SalvageError("validated salvage authority required")
    checked = load_recovery_c8_salvage_launch_record(
        record.path, repo_root=record.repo_root, schema_path=record.schema_path,
        require_tracked=True, require_fresh_outputs=False,
    )
    if checked != record:
        raise RecoveryC8SalvageError("salvage authority differs from its bytes")
    return checked


def _expected_bindings(plan: Any, estimate: Any) -> dict[str, Any]:
    return {
        "configuration": {
            "execution_plan_sha256": plan.canonical_digest,
            "run_spec_sha256": canonical_sha256(plan.run_spec.to_dict()),
        },
        "resources": {
            "preflight": {
                "plan_digest": plan.canonical_digest,
                "canonical_digest": estimate.canonical_digest,
            }
        },
    }


def _level_bytes(inventory: Sequence[Mapping[str, Any]], attempt: Path) -> dict[str, int]:
    values = {name: 0 for name in ("L0", "L1", "L2", "L3", "control")}
    for item in inventory:
        if isinstance(item, Mapping) and item.get("level") in values:
            values[str(item["level"])] += int(item.get("bytes", 0))
    values["control"] = sum((attempt / name).stat().st_size for name in ("manifest.json", "status.json"))
    return values


def _validate_one(
    attempt: Path, unit: Any, *, plan: Any, estimate: Any, authority: Any,
    observed_git_state: Mapping[str, Any], origin: str, plan_index: int,
    verifier_identity: Mapping[str, Any], execution_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    checked = validate_execution_unit(
        attempt, expected_unit_identity=unit.key.to_dict(),
        expected_provenance_bindings=_expected_bindings(plan, estimate),
        observed_git_state=observed_git_state, caps=dict(estimate.caps["effective"]),
        require_runner_telemetry=True, trusted_recovery_context=authority,
    )
    errors = list(checked.errors)
    terminal_failed = (
        checked.valid and checked.status == "failed"
        and checked.terminal and not checked.complete
    )
    if not checked.resumable and not terminal_failed:
        errors.append("attempt is not a valid terminal unit")
    manifest = _read(attempt / "manifest.json") if (attempt / "manifest.json").is_file() else {}
    status = _read(attempt / "status.json") if (attempt / "status.json").is_file() else {}
    provenance = _read(attempt / "L0/provenance.json") if (attempt / "L0/provenance.json").is_file() else {}
    result = _read(attempt / "L0/result.json") if (attempt / "L0/result.json").is_file() else {}
    inventory = manifest.get("artifacts", []) if isinstance(manifest, Mapping) else []
    code = provenance.get("bindings", {}).get("code", {}) if provenance else {}
    producer = (
        authority.source_c8_record.document["producer_git"]
        if origin == SOURCE_ORIGIN else authority.document["producer_git"]
    )
    execution_commit = code.get("commit")
    if code != {"commit": execution_identity["commit"], "dirty": False}:
        errors.append("attempt code binding differs from its execution identity")
    return {
        "plan_index": plan_index, "unit_key": unit.key.to_dict(), "origin": origin,
        "state": (
            "succeeded" if checked.resumable
            else "failed" if terminal_failed else "rejected"
        ),
        "source_root": str(attempt.parents[3]), "attempt_path": str(attempt),
        "execution_code_commit": execution_commit,
        "original_producer_git": _plain(producer),
        "execution_git_identity": _plain(execution_identity),
        "salvage_verifier": _plain(verifier_identity),
        "hashes": {"manifest_sha256": checked.manifest_sha256, "status_sha256": _sha(attempt / "status.json") if status else None, "provenance_sha256": _sha(attempt / "L0/provenance.json") if provenance else None, "result_sha256": _sha(attempt / "L0/result.json") if result else None, "unit_identity_sha256": checked.unit_identity_sha256, "artifact_inventory_sha256": canonical_sha256({"artifacts": inventory})},
        "artifact_inventory": _plain(inventory), "scalars": _plain(result.get("scalars", {})),
        "provenance": _plain(provenance), "resources": _plain(result.get("resources", {})),
        "runner_telemetry": _plain(status.get("runner_telemetry", {})),
        "work_item_counts": _plain(unit.work_item_counts), "audit": bool(unit.audit),
        "l0_l1_cap_accounting_bytes": checked.l0_l1_bytes, "level_bytes": _level_bytes(inventory, attempt),
        "total_regular_file_bytes": checked.total_bytes, "validation_errors": errors,
    }, errors


def audit_c8_salvage_source(
    authority: RecoveryC8SalvageLaunchRecord, plan: Any, config: Any, estimate: Any,
    *, verifier_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = _authority(authority)
    verifier = (
        _capture_verifier_identity(authority) if verifier_identity is None
        else _validate_verifier_identity(authority, verifier_identity)
    )
    scope = validate_c8_execution_plan(plan, config, trusted_recovery_context=authority)
    if not scope["valid"]:
        raise RecoveryC8SalvageError("salvage source plan is not exact C8")
    auth = _read(authority.source_root / "c8_launch_authorization.json")
    old_code = auth["source"]["execution_git_state"]
    old_execution_identity = _git_identity_for_commit(
        authority, old_code.get("commit"),
    )
    checkpoint = _read(authority.source_root / "failure_table.json")
    claimed = {row["unit_key"]["canonical_digest"]: row for row in checkpoint["rows"]}
    before = _tree_inventory(authority.repo_root, authority.source_root, revision=authority.document["source_interrupted_root"]["evidence_commit"])
    inherited: list[dict[str, Any]] = []; gaps: list[dict[str, Any]] = []; rejected: list[dict[str, Any]] = []
    for index, unit in enumerate(plan.units):
        digest = unit.key.canonical_digest
        attempts = attempt_directories(authority.source_root / "units" / digest / "attempts")
        if claimed[digest].get("state") == "succeeded" and len(attempts) == 1:
            record, errors = _validate_one(attempts[0], unit, plan=plan, estimate=estimate, authority=authority, observed_git_state=old_code, origin=SOURCE_ORIGIN, plan_index=index, verifier_identity=verifier, execution_identity=old_execution_identity)
            if not errors and record["state"] == "succeeded":
                inherited.append(record); continue
            rejected.append({"plan_index": index, "unit_key": unit.key.to_dict(), "origin": REJECTED_ORIGIN, "errors": errors})
        elif attempts:
            rejected.append({"plan_index": index, "unit_key": unit.key.to_dict(), "origin": REJECTED_ORIGIN, "errors": [f"source state {claimed[digest].get('state')} is not inheritable"]})
        gaps.append({"plan_index": index, "unit_key": unit.key.to_dict()})
    after = _tree_inventory(authority.repo_root, authority.source_root, revision=authority.document["source_interrupted_root"]["evidence_commit"])
    if before != after:
        raise RecoveryC8SalvageError("interrupted source changed during audit")
    document = {
        "schema_version": 1, "document_type": "stage_c_c8_salvage_source_audit",
        "status": "PASS", "holdout_accessed": False, "plan_digest": plan.canonical_digest,
        "source_root": str(authority.source_root), "source_inventory": before,
        "source_checkpoint_state_counts": dict(sorted(Counter(row["state"] for row in checkpoint["rows"]).items())),
        "inherited_count": len(inherited), "gap_count": len(gaps), "rejected_count": len(rejected),
        "inherited_unit_digests": [item["unit_key"]["canonical_digest"] for item in inherited],
        "gap_unit_digests": [item["unit_key"]["canonical_digest"] for item in gaps],
        "rejected_units": rejected, "inherited_records": inherited,
        "verifier": verifier,
    }
    if len(inherited) + len(gaps) != EXPECTED_UNIT_COUNT:
        raise RecoveryC8SalvageError(
            "salvage audit does not partition the exact 2,180-unit plan"
        )
    document["canonical_digest"] = canonical_sha256(document)
    return document


def initialize_c8_salvage_gap_root(
    authority: RecoveryC8SalvageLaunchRecord, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    if authority.gap_root.exists():
        raise RecoveryC8SalvageError("fresh salvage gap root already exists")
    report = validate_fixed_c7_report(
        authority.repo_root / authority.source_c8_record.document["c7_report"]["path"],
        repo_root=authority.repo_root, trusted_recovery_context=authority,
    )
    estimate = build_c8_measured_preflight(
        plan, config, report, trusted_recovery_context=authority,
    )
    audit = audit_c8_salvage_source(authority, plan, config, estimate)
    audit_path = authority.reports_root / "c8_salvage_source_audit.json"
    authority.reports_root.mkdir(parents=True, exist_ok=False)
    compact_audit = {
        key: value for key, value in audit.items()
        if key != "inherited_records"
    }
    compact_audit["inherited_record_index"] = [
        {
            "plan_index": item["plan_index"],
            "unit_digest": item["unit_key"]["canonical_digest"],
            "attempt_path": item["attempt_path"],
            "execution_code_commit": item["execution_code_commit"],
            "hashes": item["hashes"],
        }
        for item in audit["inherited_records"]
    ]
    compact_audit["canonical_digest"] = canonical_sha256({
        key: value for key, value in compact_audit.items()
        if key != "canonical_digest"
    })
    _atomic_create_json(audit_path, compact_audit)
    authority.gap_root.mkdir(parents=True, exist_ok=False)
    scope = {
        "schema_version": 1, "document_type": "stage_c_c8_salvage_scope",
        "plan_digest": plan.canonical_digest,
        "source_audit_path": str(audit_path),
        "source_audit_digest": compact_audit["canonical_digest"],
        "gap_unit_digests": audit["gap_unit_digests"],
        "inherited_unit_digests": audit["inherited_unit_digests"],
        "states": {digest: "pending" for digest in audit["gap_unit_digests"]},
        "claim_tokens": {},
        "canonical_digest": None,
    }
    scope["canonical_digest"] = canonical_sha256({k: v for k, v in scope.items() if k != "canonical_digest"})
    _atomic_create_json(authority.gap_root / PLAN_FILE, plan.to_dict())
    _atomic_create_json(authority.gap_root / PREFLIGHT_FILE, estimate.to_dict())
    _atomic_create_json(authority.gap_root / SCOPE_FILE, scope)
    _atomic_create_json(
        authority.gap_root / STATUS_FILE,
        _runtime_status(scope, state="running"),
    )
    return {"status": "PASS", "inherited_count": audit["inherited_count"], "gap_count": audit["gap_count"], "rejected_count": audit["rejected_count"], "source_audit_digest": compact_audit["canonical_digest"]}


def _scope(authority: Any, plan: Any) -> dict[str, Any]:
    value = _read(authority.gap_root / SCOPE_FILE)
    digest = value.pop("canonical_digest", None)
    if digest != canonical_sha256(value):
        raise RecoveryC8SalvageError("salvage scope digest drifted")
    value["canonical_digest"] = digest
    if value["plan_digest"] != plan.canonical_digest:
        raise RecoveryC8SalvageError("salvage scope plan drifted")
    return value


def _source_audit_binding(authority: Any, scope: Mapping[str, Any]) -> dict[str, Any]:
    expected = authority.reports_root / "c8_salvage_source_audit.json"
    if scope.get("source_audit_path") != str(expected):
        raise RecoveryC8SalvageError("salvage source-audit path drifted")
    audit = _read(expected)
    claimed_digest = audit.get("canonical_digest")
    if claimed_digest != canonical_sha256({
        key: value for key, value in audit.items() if key != "canonical_digest"
    }):
        raise RecoveryC8SalvageError("frozen salvage source audit digest drifted")
    if scope.get("source_audit_digest") != claimed_digest:
        raise RecoveryC8SalvageError("salvage scope/source-audit binding drifted")
    return {
        "path": str(expected),
        "file_sha256": _sha(expected),
        "canonical_digest": claimed_digest,
        "document": audit,
    }


def _validate_scope_partition(
    authority: Any, plan: Any, scope: Mapping[str, Any],
    *, source_reaudit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan_digests = [unit.key.canonical_digest for unit in plan.units]
    if (
        len(plan_digests) != EXPECTED_UNIT_COUNT
        or len(set(plan_digests)) != EXPECTED_UNIT_COUNT
    ):
        raise RecoveryC8SalvageError(
            "salvage plan is not the exact 2,180-unit unique scope"
        )
    gap = scope.get("gap_unit_digests")
    inherited = scope.get("inherited_unit_digests")
    if not isinstance(gap, list) or not isinstance(inherited, list):
        raise RecoveryC8SalvageError("salvage scope partition is malformed")
    if (
        len(gap) + len(inherited) != EXPECTED_UNIT_COUNT
        or len(set(gap)) != len(gap)
        or len(set(inherited)) != len(inherited)
        or set(gap) & set(inherited)
        or set(gap) | set(inherited) != set(plan_digests)
        or gap != [digest for digest in plan_digests if digest in set(gap)]
        or inherited
        != [digest for digest in plan_digests if digest in set(inherited)]
    ):
        raise RecoveryC8SalvageError(
            "salvage scope is not the frozen ordered 2,180-unit partition"
        )
    binding = _source_audit_binding(authority, scope)
    frozen = binding["document"]
    if (
        frozen.get("status") != "PASS"
        or frozen.get("plan_digest") != plan.canonical_digest
        or frozen.get("inherited_unit_digests") != inherited
        or frozen.get("gap_unit_digests") != gap
        or frozen.get("inherited_count") != len(inherited)
        or frozen.get("gap_count") != len(gap)
    ):
        raise RecoveryC8SalvageError(
            "salvage scope differs from its frozen source audit"
        )
    if source_reaudit is not None and (
        source_reaudit.get("status") != "PASS"
        or source_reaudit.get("plan_digest") != plan.canonical_digest
        or source_reaudit.get("inherited_unit_digests") != inherited
        or source_reaudit.get("gap_unit_digests") != gap
        or source_reaudit.get("inherited_count") != len(inherited)
        or source_reaudit.get("gap_count") != len(gap)
    ):
        raise RecoveryC8SalvageError(
            "salvage scope partition differs from the current source re-audit"
        )
    if source_reaudit is not None:
        current_index = [
            {
                "plan_index": item["plan_index"],
                "unit_digest": item["unit_key"]["canonical_digest"],
                "attempt_path": item["attempt_path"],
                "execution_code_commit": item["execution_code_commit"],
                "hashes": item["hashes"],
            }
            for item in source_reaudit.get("inherited_records", ())
        ]
        if (
            frozen.get("source_inventory")
            != source_reaudit.get("source_inventory")
            or frozen.get("rejected_count")
            != source_reaudit.get("rejected_count")
            or frozen.get("rejected_units")
            != source_reaudit.get("rejected_units")
            or frozen.get("inherited_record_index") != current_index
        ):
            raise RecoveryC8SalvageError(
                "frozen source audit differs from bytewise current re-audit"
            )
    states = scope.get("states")
    claims = scope.get("claim_tokens")
    if not isinstance(states, Mapping) or set(states) != set(gap):
        raise RecoveryC8SalvageError(
            "salvage state table does not exactly cover the frozen gap"
        )
    if any(state not in _SCOPE_STATES for state in states.values()):
        raise RecoveryC8SalvageError("salvage state table contains an invalid state")
    if not isinstance(claims, Mapping):
        raise RecoveryC8SalvageError("salvage claim table is malformed")
    claimed = {digest for digest, state in states.items() if state == "claimed"}
    if set(claims) != claimed or any(
        not isinstance(token, str) or len(token) != 64
        or set(token) - set("0123456789abcdef")
        for token in claims.values()
    ):
        raise RecoveryC8SalvageError(
            "salvage claims do not exactly match claimed gap states"
        )
    return {key: value for key, value in binding.items() if key != "document"}


def _require_finalized_snapshot(
    authority: Any, plan: Any, scope: Mapping[str, Any],
    *, source_reaudit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = _validate_scope_partition(
        authority, plan, scope, source_reaudit=source_reaudit,
    )
    if scope.get("claim_tokens"):
        raise RecoveryC8SalvageError("finalized salvage scope retains claims")
    bad = {
        digest: state for digest, state in scope["states"].items()
        if state != "succeeded"
    }
    if bad:
        raise RecoveryC8SalvageError(
            f"finalized salvage scope is not all succeeded: {Counter(bad.values())}"
        )
    _require_runtime_status(authority, scope, state="finalized")
    return binding


def _runtime_status(scope: Mapping[str, Any], *, state: str) -> dict[str, Any]:
    counts = Counter(scope["states"].values())
    return {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_status",
        "state": state,
        "active_claim_count": int(counts["claimed"]),
        "gap_count": len(scope["gap_unit_digests"]),
        "succeeded_count": int(counts["succeeded"]),
        "failed_count": int(counts["failed"]),
        "incomplete_count": int(counts["incomplete"]),
        "pending_count": int(counts["pending"]),
    }


def _require_runtime_status(
    authority: Any, scope: Mapping[str, Any], *, state: str,
) -> None:
    if _read(authority.gap_root / STATUS_FILE) != _runtime_status(
        scope, state=state,
    ):
        raise RecoveryC8SalvageError(
            f"salvage status is not the exact {state} snapshot"
        )


def _publish_scope(authority: Any, scope: dict[str, Any], *, state: str) -> None:
    scope.pop("canonical_digest", None)
    scope["canonical_digest"] = canonical_sha256(scope)
    _atomic_replace_json(authority.gap_root / SCOPE_FILE, scope)
    _atomic_replace_json(
        authority.gap_root / STATUS_FILE, _runtime_status(scope, state=state),
    )


def _runtime_git_state(authority: Any, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {"commit", "dirty"}:
        raise RecoveryC8SalvageError(
            "salvage runtime Git state must contain exactly commit and dirty"
        )
    state = _plain(raw)
    if state != {"commit": authority.launch_commit, "dirty": False}:
        raise RecoveryC8SalvageError(
            "salvage runtime Git state is not the real clean launch commit"
        )
    return state


def run_c8_salvage_gap_shard(
    authority: RecoveryC8SalvageLaunchRecord, plan: Any, backend: Any, *,
    shard_index: int, shard_count: int, device: str = "cuda:0",
    runtime_git_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    authority = _authority(authority); scope = _scope(authority, plan)
    _validate_scope_partition(authority, plan, scope)
    if not (0 <= shard_index < shard_count):
        raise ValueError("invalid salvage shard")
    estimate = __import__("vitnns.stage_c_plan", fromlist=["StageCEstimate"]).StageCEstimate.from_dict(_read(authority.gap_root / PREFLIGHT_FILE))
    code = _runtime_git_state(authority, runtime_git_state)
    backend_context = getattr(backend, "trusted_recovery_context", None)
    if backend_context != authority:
        raise RecoveryC8SalvageError(
            "salvage gap backend lacks the exact launch authority"
        )
    by_digest = {unit.key.canonical_digest: unit for unit in plan.units}
    requested = [digest for index, digest in enumerate(scope["gap_unit_digests"]) if index % shard_count == shard_index]
    results = []
    with _lifecycle_lock(authority.gap_root, exclusive=False):
        for digest in requested:
            claim = secrets.token_hex(32)
            with _checkpoint_lock(authority.gap_root):
                fresh_scope = _scope(authority, plan)
                _validate_scope_partition(authority, plan, fresh_scope)
                _require_runtime_status(authority, fresh_scope, state="running")
                prior_state = fresh_scope["states"].get(digest)
                if prior_state in {"succeeded", "failed"}:
                    results.append({
                        "unit_key": by_digest[digest].key.to_dict(),
                        "state": prior_state,
                        "attempt_path": None,
                        "resumed": True,
                        "failure": None,
                    })
                    continue
                if prior_state != "pending":
                    raise RecoveryC8SalvageError(
                        f"gap unit is not pending: {digest}"
                    )
                fresh_scope["states"][digest] = "claimed"
                fresh_scope.setdefault("claim_tokens", {})[digest] = claim
                _publish_scope(authority, fresh_scope, state="running")
            try:
                with _artifact_activity_lock(
                    authority.gap_root, exclusive=False
                ):
                    result = run_execution_unit(
                        by_digest[digest], output_root=authority.gap_root,
                        backend=backend, device=device, resume=False,
                        caps=dict(estimate.caps["effective"]),
                        preflight=estimate.to_dict(), plan=plan,
                        observed_git_state=code,
                    )
            except BaseException:
                with _checkpoint_lock(authority.gap_root):
                    fresh_scope = _scope(authority, plan)
                    _validate_scope_partition(authority, plan, fresh_scope)
                    if fresh_scope.get("claim_tokens", {}).get(digest) == claim:
                        fresh_scope["states"][digest] = "incomplete"
                        fresh_scope["claim_tokens"].pop(digest, None)
                        _publish_scope(authority, fresh_scope, state="running")
                raise
            state = str(result.state)
            with _checkpoint_lock(authority.gap_root):
                fresh_scope = _scope(authority, plan)
                _validate_scope_partition(authority, plan, fresh_scope)
                if (
                    fresh_scope["states"].get(digest) != "claimed"
                    or fresh_scope.get("claim_tokens", {}).get(digest) != claim
                ):
                    raise RecoveryC8SalvageError(
                        f"gap claim changed during execution: {digest}"
                    )
                fresh_scope["states"][digest] = state
                fresh_scope["claim_tokens"].pop(digest, None)
                _publish_scope(authority, fresh_scope, state="running")
            results.append(_plain(result))
    return results


def build_c8_salvage_combined_manifest(
    authority: RecoveryC8SalvageLaunchRecord, plan: Any, config: Any,
    *, verifier_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = _authority(authority); scope = _scope(authority, plan)
    verifier = (
        _capture_verifier_identity(authority) if verifier_identity is None
        else _validate_verifier_identity(authority, verifier_identity)
    )
    estimate = __import__("vitnns.stage_c_plan", fromlist=["StageCEstimate"]).StageCEstimate.from_dict(_read(authority.gap_root / PREFLIGHT_FILE))
    # Revalidate old source on every combine; never trust cached inherited records.
    source_audit = audit_c8_salvage_source(
        authority, plan, config, estimate, verifier_identity=verifier,
    )
    frozen_audit = _validate_scope_partition(
        authority, plan, scope, source_reaudit=source_audit,
    )
    inherited = {item["unit_key"]["canonical_digest"]: item for item in source_audit["inherited_records"]}
    gap_set = set(scope["gap_unit_digests"]); rows = []; errors = []
    code = {"commit": authority.launch_commit, "dirty": False}
    gap_execution_identity = _git_identity_for_commit(
        authority, authority.launch_commit,
    )
    for index, unit in enumerate(plan.units):
        digest = unit.key.canonical_digest
        if digest in inherited:
            rows.append(inherited[digest]); continue
        attempts = attempt_directories(authority.gap_root / "units" / digest / "attempts")
        if digest not in gap_set or len(attempts) != 1:
            errors.append(f"{digest}: gap attempt count is not one"); continue
        record, unit_errors = _validate_one(attempts[0], unit, plan=plan, estimate=estimate, authority=authority, observed_git_state=code, origin=GAP_ORIGIN, plan_index=index, verifier_identity=verifier, execution_identity=gap_execution_identity)
        if record["execution_code_commit"] != authority.launch_commit:
            unit_errors.append("new gap unit does not bind the real salvage launch commit")
        if scope["states"].get(digest) != record["state"]:
            unit_errors.append("gap record state differs from salvage scope")
        if record["state"] != "succeeded":
            unit_errors.append("combined record state is not succeeded")
        errors.extend(f"{digest}: {item}" for item in unit_errors); rows.append(record)
    unexpected = []
    units_dir = authority.gap_root / "units"
    if units_dir.exists():
        unexpected = sorted(path.name for path in units_dir.iterdir() if path.name not in gap_set)
    if unexpected:
        errors.append(f"gap root contains inherited/unexpected unit directories: {unexpected}")
    expected_order = [unit.key.canonical_digest for unit in plan.units]
    observed_order = [item["unit_key"]["canonical_digest"] for item in rows]
    if observed_order != expected_order:
        errors.append("combined record order/coverage differs from the frozen plan")
    if any(
        item.get("state") != "succeeded"
        or item.get("validation_errors") != []
        for item in rows
    ):
        errors.append("every combined record must be validated and succeeded")
    if len(rows) != EXPECTED_UNIT_COUNT or errors:
        raise RecoveryC8SalvageError("salvage union is incomplete or invalid: " + "; ".join(errors[:20]))
    origin_counts = Counter(item["origin"] for item in rows)
    manifest = {
        "schema_version": 1, "document_type": "stage_c_c8_salvage_combined_manifest",
        "status": "PASS", "holdout_accessed": False,
        "plan_digest": plan.canonical_digest,
        "salvage_authority": {
            "path": str(authority.path), "file_sha256": authority.file_sha256,
            "canonical_digest": authority.canonical_digest,
            "launch_commit": authority.launch_commit,
        },
        "frozen_source_audit": frozen_audit,
        "source_reaudit_digest": source_audit["canonical_digest"],
        "source_partition_counts": {
            "inherited": source_audit["inherited_count"],
            "gap": source_audit["gap_count"],
            "rejected_source_attempts": source_audit["rejected_count"],
        },
        "rejected_source_attempts": _plain(source_audit["rejected_units"]),
        "source_root": str(authority.source_root), "gap_root": str(authority.gap_root),
        "source_root_inventory": source_audit["source_inventory"],
        "gap_root_inventory": _tree_inventory(authority.repo_root, authority.gap_root),
        "origin_counts": dict(sorted(origin_counts.items())), "records": rows,
        "terminal_state_counts": dict(sorted(Counter(item["state"] for item in rows).items())),
        "producer_identities": {
            "interrupted_root_producer": _plain(
                authority.source_c8_record.document["producer_git"]
            ),
            "salvage_authority_producer": _plain(
                authority.document["producer_git"]
            ),
            "interrupted_execution": _git_identity_for_commit(
                authority, authority.document["source_interrupted_root"]["execution_commit"],
            ),
            "gap_execution": _git_identity_for_commit(
                authority, authority.launch_commit,
            ),
        },
        "verifier": verifier,
        "combined_artifact_manifest_digest": canonical_sha256({"domain": "vitnns-c8-salvage-union-v1", "records": rows}),
    }
    manifest["canonical_digest"] = canonical_sha256(manifest)
    return manifest


def finalize_c8_salvage(
    authority: RecoveryC8SalvageLaunchRecord, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    with (
        _lifecycle_lock(authority.gap_root, exclusive=True),
        _checkpoint_lock(authority.gap_root),
        _artifact_activity_lock(authority.gap_root, exclusive=True),
    ):
        scope = _scope(authority, plan)
        _validate_scope_partition(authority, plan, scope)
        not_succeeded = {
            digest: state for digest, state in scope["states"].items()
            if state != "succeeded"
        }
        if (
            len(scope["states"]) != len(scope["gap_unit_digests"])
            or not_succeeded or scope.get("claim_tokens")
        ):
            return {
                "status": "BLOCKED", "successful": False,
                "c8_pass": False, "manifest_written": False,
                "gap_count": len(scope["states"]),
                "gap_state_counts": dict(sorted(
                    Counter(scope["states"].values()).items()
                )),
                "active_claim_count": len(scope.get("claim_tokens", {})),
                "errors": [
                    "all frozen gap units must be succeeded with zero claims"
                ],
            }
        verifier = _capture_verifier_identity(authority)
        _publish_scope(authority, scope, state="finalized")
        try:
            manifest = build_c8_salvage_combined_manifest(
                authority, plan, config, verifier_identity=verifier,
            )
            if any(
                record.get("state") != "succeeded"
                for record in manifest.get("records", ())
            ):
                raise RecoveryC8SalvageError(
                    "combined manifest contains a non-succeeded record"
                )
            authority.reports_root.mkdir(parents=True, exist_ok=True)
            _atomic_create_json(authority.combined_manifest_path, manifest)
        except BaseException:
            _publish_scope(authority, scope, state="blocked")
            raise
    return manifest


def validate_c8_salvage_from_source(
    authority: Any, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    with (
        _lifecycle_lock(authority.gap_root, exclusive=True),
        _checkpoint_lock(authority.gap_root),
        _artifact_activity_lock(authority.gap_root, exclusive=True),
    ):
        scope = _scope(authority, plan)
        _require_finalized_snapshot(authority, plan, scope)
        persisted = _read(authority.combined_manifest_path)
        verifier = _validate_verifier_identity(
            authority, persisted.get("verifier"),
        )
        rebuilt = build_c8_salvage_combined_manifest(
            authority, plan, config, verifier_identity=verifier,
        )
        if persisted != rebuilt:
            raise RecoveryC8SalvageError(
                "persisted combined manifest differs from source recomputation"
            )
        if any(
            record.get("state") != "succeeded"
            for record in persisted.get("records", ())
        ):
            raise RecoveryC8SalvageError(
                "persisted combined manifest contains a non-succeeded record"
            )
        return persisted


def _selection_generator_state(
    authority: Any, selection: Mapping[str, Any],
) -> dict[str, Any]:
    prefix = "vitnns.stage_c_c8_salvage_selection@"
    generated_by = selection.get("generated_by")
    if not isinstance(generated_by, str) or not generated_by.startswith(prefix):
        raise RecoveryC8SalvageError("salvage selection generator is invalid")
    commit = generated_by[len(prefix):]
    if (
        len(commit) != 40 or set(commit) - set("0123456789abcdef")
        or _git(
            Path(authority.repo_root), "rev-parse", f"{commit}^{{commit}}"
        ) != commit
    ):
        raise RecoveryC8SalvageError("salvage selection commit is invalid")
    return {"commit": commit, "dirty": False}


def _validated_selection_bindings(
    authority: Any, plan: Any, config: Any,
) -> dict[str, Any]:
    selection = _read(authority.selection_path)
    report = _read(authority.selection_report_path)
    selection_hash = _sha(authority.selection_path)
    report_hash = _sha(authority.selection_report_path)
    generator = _selection_generator_state(authority, selection)
    from .stage_c_c8_salvage_selection import (
        validate_c8_salvage_selection_report,
    )

    validate_c8_salvage_selection_report(
        authority, plan, config, report, selection,
        generator_git_state=generator,
    )
    if (
        _sha(authority.selection_path) != selection_hash
        or _sha(authority.selection_report_path) != report_hash
    ):
        raise RecoveryC8SalvageError(
            "salvage selection changed during completion verification"
        )
    return {
        "selection": {
            "path": str(authority.selection_path),
            "file_sha256": selection_hash,
            "canonical_digest": selection["canonical_digest"],
        },
        "selection_report": {
            "path": str(authority.selection_report_path),
            "file_sha256": report_hash,
            "canonical_digest": report["canonical_digest"],
        },
    }


def build_c8_salvage_completion_report(
    authority: Any, plan: Any, config: Any, *,
    verifier_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = _authority(authority)
    manifest = validate_c8_salvage_from_source(authority, plan, config)
    selection = _validated_selection_bindings(authority, plan, config)
    verifier = (
        _capture_verifier_identity(authority) if verifier_identity is None
        else _validate_verifier_identity(authority, verifier_identity)
    )
    frozen_audit = manifest.get("frozen_source_audit")
    partition = manifest.get("source_partition_counts", {})
    inherited_count = int(partition.get("inherited", -1))
    gap_count = int(partition.get("gap", -1))
    rejected_count = int(partition.get("rejected_source_attempts", -1))
    if (
        inherited_count < 0 or gap_count < 0 or rejected_count < 0
        or inherited_count + gap_count != EXPECTED_UNIT_COUNT
        or manifest.get("origin_counts") != {
            GAP_ORIGIN: gap_count, SOURCE_ORIGIN: inherited_count,
        }
        or not isinstance(frozen_audit, Mapping)
        or set(frozen_audit) != {"path", "file_sha256", "canonical_digest"}
    ):
        raise RecoveryC8SalvageError(
            "completion counts differ from the validated source partition"
        )
    report = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_completion_report",
        "status": "PASS", "c8_pass": True,
        "salvage_authority": {
            "path": str(authority.path), "file_sha256": authority.file_sha256,
            "canonical_digest": authority.canonical_digest,
            "launch_commit": authority.launch_commit,
        },
        "amendment": _plain(authority.document["amendment"]),
        "base_c8_authority": _plain(authority.document["base_c8_authority"]),
        "contracts": _plain(authority.document["contracts"]),
        "model": _plain(authority.document["model"]),
        "combined_manifest": {
            "path": str(authority.combined_manifest_path),
            "file_sha256": _sha(authority.combined_manifest_path),
            "canonical_digest": manifest["canonical_digest"],
            "combined_artifact_manifest_digest":
                manifest["combined_artifact_manifest_digest"],
        },
        "source_audit": {
            **_plain(frozen_audit),
            "source_reaudit_digest": manifest["source_reaudit_digest"],
        },
        "selection": selection["selection"],
        "selection_report": selection["selection_report"],
        "roots": {
            "interrupted_source_root": str(authority.source_root),
            "fresh_gap_root": str(authority.gap_root),
            "virtual_combined_manifest": str(
                authority.combined_manifest_path
            ),
        },
        "source_root_inventory": manifest["source_root_inventory"],
        "gap_root_inventory": manifest["gap_root_inventory"],
        "producer_identities": manifest["producer_identities"],
        "verifier": verifier,
        "counts": {
            "expected_units": EXPECTED_UNIT_COUNT,
            "inherited_units": inherited_count,
            "gap_units": gap_count,
            "newly_run_units": gap_count,
            "rejected_source_attempts": rejected_count,
            "failed_or_blocked_units": 0,
            "succeeded_units": EXPECTED_UNIT_COUNT,
            "origin_counts": manifest["origin_counts"],
        },
        "flags": {
            "holdout_accessed": False, "selection_performed": True,
            "bootstrap_performed": False, "holm_performed": False,
            "decision_performed": False, "c9_authorized": False,
            "c10_authorized": False, "stage_d_authorized": False,
        },
    }
    report["canonical_digest"] = canonical_sha256(report)
    _validate_completion_contract(authority, report)
    return report


def _validate_completion_contract(authority: Any, raw: Any) -> dict[str, Any]:
    report = _plain(raw)
    top_keys = {
        "schema_version", "document_type", "status", "c8_pass",
        "salvage_authority", "amendment", "base_c8_authority",
        "contracts", "model", "combined_manifest", "source_audit",
        "selection", "selection_report", "roots",
        "source_root_inventory", "gap_root_inventory",
        "producer_identities", "verifier", "counts", "flags",
        "canonical_digest",
    }
    if not isinstance(report, dict) or set(report) != top_keys:
        raise RecoveryC8SalvageError(
            "salvage completion report fields are not closed"
        )
    if (
        report["schema_version"] != 1
        or report["document_type"]
        != "stage_c_c8_salvage_completion_report"
        or report["status"] != "PASS"
        or report["c8_pass"] is not True
        or report["canonical_digest"]
        != canonical_sha256({
            key: value for key, value in report.items()
            if key != "canonical_digest"
        })
    ):
        raise RecoveryC8SalvageError(
            "salvage completion identity or digest is invalid"
        )
    exact_nested = {
        "salvage_authority": {
            "path", "file_sha256", "canonical_digest", "launch_commit",
        },
        "combined_manifest": {
            "path", "file_sha256", "canonical_digest",
            "combined_artifact_manifest_digest",
        },
        "source_audit": {
            "path", "file_sha256", "canonical_digest",
            "source_reaudit_digest",
        },
        "selection": {"path", "file_sha256", "canonical_digest"},
        "selection_report": {
            "path", "file_sha256", "canonical_digest",
        },
        "roots": {
            "interrupted_source_root", "fresh_gap_root",
            "virtual_combined_manifest",
        },
        "counts": {
            "expected_units", "inherited_units", "gap_units",
            "newly_run_units", "rejected_source_attempts",
            "failed_or_blocked_units", "succeeded_units",
            "origin_counts",
        },
        "flags": {
            "holdout_accessed", "selection_performed",
            "bootstrap_performed", "holm_performed",
            "decision_performed", "c9_authorized", "c10_authorized",
            "stage_d_authorized",
        },
    }
    for name, keys in exact_nested.items():
        if not isinstance(report[name], dict) or set(report[name]) != keys:
            raise RecoveryC8SalvageError(
                f"salvage completion {name} fields are not closed"
            )
    hashes = []
    for name in (
        "combined_manifest", "source_audit", "selection",
        "selection_report",
    ):
        hashes.extend(
            report[name][key] for key in report[name]
            if key.endswith("sha256") or key == "canonical_digest"
        )
    if any(
        not isinstance(value, str) or len(value) != 64
        or set(value) - set("0123456789abcdef")
        for value in hashes
    ):
        raise RecoveryC8SalvageError(
            "salvage completion contains an invalid SHA-256 binding"
        )
    counts = report["counts"]
    if (
        type(counts["expected_units"]) is not int
        or counts["expected_units"] != EXPECTED_UNIT_COUNT
        or type(counts["inherited_units"]) is not int
        or type(counts["gap_units"]) is not int
        or counts["inherited_units"] + counts["gap_units"]
        != EXPECTED_UNIT_COUNT
        or counts["newly_run_units"] != counts["gap_units"]
        or counts["succeeded_units"] != EXPECTED_UNIT_COUNT
        or counts["failed_or_blocked_units"] != 0
        or type(counts["rejected_source_attempts"]) is not int
        or counts["rejected_source_attempts"] < 0
        or counts["origin_counts"] != {
            GAP_ORIGIN: counts["gap_units"],
            SOURCE_ORIGIN: counts["inherited_units"],
        }
    ):
        raise RecoveryC8SalvageError(
            "salvage completion counts are invalid"
        )
    expected_flags = {
        "holdout_accessed": False, "selection_performed": True,
        "bootstrap_performed": False, "holm_performed": False,
        "decision_performed": False, "c9_authorized": False,
        "c10_authorized": False, "stage_d_authorized": False,
    }
    if report["flags"] != expected_flags:
        raise RecoveryC8SalvageError(
            "salvage completion access or downstream flags are invalid"
        )
    expected_paths = {
        "salvage_authority": str(authority.path),
        "combined_manifest": str(authority.combined_manifest_path),
        "selection": str(authority.selection_path),
        "selection_report": str(authority.selection_report_path),
    }
    for name, path in expected_paths.items():
        if report[name]["path"] != path:
            raise RecoveryC8SalvageError(
                f"salvage completion {name} path drifted"
            )
    if report["roots"] != {
        "interrupted_source_root": str(authority.source_root),
        "fresh_gap_root": str(authority.gap_root),
        "virtual_combined_manifest": str(authority.combined_manifest_path),
    }:
        raise RecoveryC8SalvageError(
            "salvage completion root bindings drifted"
        )
    return report


def write_c8_salvage_completion_report(
    authority: Any, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    if os.path.lexists(authority.combined_report_path):
        if (
            authority.combined_report_path.is_symlink()
            or not authority.combined_report_path.is_file()
        ):
            raise RecoveryC8SalvageError(
                "existing salvage completion report is unsafe"
            )
        return validate_c8_salvage_completion_report(
            authority, plan, config,
        )
    report = build_c8_salvage_completion_report(authority, plan, config)
    try:
        _atomic_create_json(authority.combined_report_path, report)
    except FileExistsError:
        return validate_c8_salvage_completion_report(
            authority, plan, config,
        )
    return report


def validate_c8_salvage_completion_report(
    authority: Any, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    persisted = _read(authority.combined_report_path)
    _validate_completion_contract(authority, persisted)
    verifier = _validate_verifier_identity(authority, persisted.get("verifier"))
    rebuilt = build_c8_salvage_completion_report(
        authority, plan, config, verifier_identity=verifier,
    )
    if persisted != rebuilt:
        raise RecoveryC8SalvageError(
            "persisted salvage completion report differs from source recomputation"
        )
    return persisted


__all__ = [
    "GAP_ORIGIN", "REJECTED_ORIGIN", "SOURCE_ORIGIN",
    "audit_c8_salvage_source", "build_c8_salvage_combined_manifest",
    "build_c8_salvage_completion_report",
    "finalize_c8_salvage", "initialize_c8_salvage_gap_root",
    "run_c8_salvage_gap_shard", "validate_c8_salvage_completion_report",
    "validate_c8_salvage_from_source", "write_c8_salvage_completion_report",
]
