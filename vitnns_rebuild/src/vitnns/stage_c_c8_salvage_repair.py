"""Three-origin, authority-bound repair runtime for interrupted C8 salvage."""

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

from .recovery_c8_salvage_repair import (
    ORIGINS as AUTHORITY_ORIGINS,
    RecoveryC8SalvageRepairError,
    RecoveryC8SalvageRepairLaunchRecord,
    _require_custody_clean,
    _tree_inventory,
    capture_authorized_c8_salvage_repair_source_state,
    capture_c8_salvage_repair_verifier_identity,
    load_recovery_c8_salvage_repair_launch_record,
)
from .stage_c_artifacts import attempt_directories, validate_execution_unit
from .stage_c_c8 import (
    _artifact_activity_lock, _atomic_create_json, _atomic_replace_json,
    _checkpoint_lock, _lifecycle_lock, build_c8_measured_preflight,
    validate_c8_execution_plan, validate_fixed_c7_report,
)
from .stage_c_provenance import canonical_sha256
from .stage_c_runner import run_execution_unit

ORIGINAL_SOURCE_ORIGIN = AUTHORITY_ORIGINS["original_interrupted"]
PRIOR_GAP_ORIGIN = AUTHORITY_ORIGINS["prior_partial_salvage"]
REPAIR_ORIGIN = AUTHORITY_ORIGINS["fresh_repair"]
ORIGINS = (ORIGINAL_SOURCE_ORIGIN, PRIOR_GAP_ORIGIN, REPAIR_ORIGIN)

EXPECTED_UNIT_COUNT = 2180
EXPECTED_ORIGINAL_COUNT = 1460
EXPECTED_PRIOR_SUCCEEDED_COUNT = 568
EXPECTED_PRIOR_FAILED_COUNT = 1
EXPECTED_PRIOR_COUNT = 569
EXPECTED_REPAIR_COUNT = 151
RULES_VERSION = "recovery-c8-salvage-repair-v1"

PLAN_FILE = "execution_plan.json"
PREFLIGHT_FILE = "preflight.json"
SCOPE_FILE = "salvage_repair_scope.json"
STATUS_FILE = "salvage_repair_status.json"
AUDIT_FILE = "c8_salvage_repair_source_audit.json"
_SCOPE_STATES = frozenset({
    "pending", "claimed", "incomplete", "succeeded", "failed",
})
_TERMINAL_STATES = frozenset({"succeeded", "failed"})
_SCIENTIFIC_FAILURE_TYPES = frozenset({
    "vitnns.stage_c_runner.StageCScientificInvalidity",
    "vitnns.stage_c_value_path.StageCValuePathConvergenceError",
    "vitnns.stage_c_value_path.StageCValuePathEndpointParityError",
    "vitnns.stage_c_value_path.StageCValuePathMatchedConservationError",
    "vitnns.stage_c_class_logit.StageCClassLogitIGConvergenceError",
})


def _plain(value: Any) -> Any:
    method = getattr(value, "to_dict", None)
    if callable(method):
        value = method()
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    raise TypeError(f"repair value is not JSON-compatible: {type(value).__name__}")


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RecoveryC8SalvageRepairError(f"invalid repair JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryC8SalvageRepairError(f"repair JSON is not an object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        _plain(value), indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _create_or_accept(path: Path, value: Mapping[str, Any], context: str) -> None:
    payload = _json_bytes(value)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RecoveryC8SalvageRepairError(
                f"existing {context} differs from source recomputation"
            )
        return
    try:
        _atomic_create_json(path, value)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RecoveryC8SalvageRepairError(
                f"raced {context} differs from source recomputation"
            )


def _authority(record: Any) -> RecoveryC8SalvageRepairLaunchRecord:
    if type(record) is not RecoveryC8SalvageRepairLaunchRecord:
        raise RecoveryC8SalvageRepairError(
            "validated C8 salvage repair authority required"
        )
    checked = load_recovery_c8_salvage_repair_launch_record(
        record.path, repo_root=record.repo_root, schema_path=record.schema_path,
        require_tracked=True, require_fresh_outputs=False,
    )
    if checked != record:
        raise RecoveryC8SalvageRepairError(
            "repair authority differs from its validated source bytes"
        )
    return checked


def _git(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryC8SalvageRepairError(
            f"repair git {' '.join(arguments)} failed: {error}"
        ) from error


def _subtree_path(authority: Any) -> str:
    top = Path(_git(Path(authority.repo_root), "rev-parse", "--show-toplevel"))
    return (Path(authority.repo_root) / "src/vitnns").resolve().relative_to(
        top.resolve()
    ).as_posix()


def _execution_identity(authority: Any, commit: Any) -> dict[str, Any]:
    if (
        not isinstance(commit, str) or len(commit) != 40
        or set(commit) - set("0123456789abcdef")
    ):
        raise RecoveryC8SalvageRepairError(
            "repair execution commit is not lowercase 40-hex"
        )
    repo = Path(authority.repo_root)
    subtree = _subtree_path(authority)
    if _git(repo, "rev-parse", f"{commit}^{{commit}}") != commit:
        raise RecoveryC8SalvageRepairError("repair execution commit is unavailable")
    identity = {
        "commit": commit,
        "tree": _git(repo, "rev-parse", f"{commit}^{{tree}}"),
        "source_subtree_path": "src/vitnns",
        "source_subtree_tree": _git(repo, "rev-parse", f"{commit}:{subtree}"),
        "dirty": False,
    }
    if identity["source_subtree_tree"] != authority.document["producer_git"]["source_subtree_tree"] and commit not in {
        authority.document["original_interrupted_root"]["execution_commit"],
        authority.document["original_salvage_authority"]["launch_commit"],
    }:
        raise RecoveryC8SalvageRepairError(
            "repair execution source subtree differs from the authorized producer"
        )
    return identity


def _capture_verifier(authority: Any, phase: str) -> dict[str, Any]:
    identity = _plain(capture_c8_salvage_repair_verifier_identity(authority, phase))
    identity.update({
        "verified_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "rules_version": RULES_VERSION,
    })
    return identity


def _validate_verifier(authority: Any, raw: Any) -> dict[str, Any]:
    value = _plain(raw)
    required = {
        "commit", "tree", "source_subtree_path", "source_subtree_tree",
        "dirty", "authorization_launch_commit", "evidence_worktree_dirty",
        "verified_at_utc", "rules_version",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RecoveryC8SalvageRepairError("repair verifier identity is malformed")
    if (
        value["dirty"] is not False
        or type(value["evidence_worktree_dirty"]) is not bool
        or value["source_subtree_path"] != "src/vitnns"
        or value["authorization_launch_commit"] != authority.launch_commit
        or value["rules_version"] != RULES_VERSION
    ):
        raise RecoveryC8SalvageRepairError("repair verifier identity drifted")
    try:
        timestamp = str(value["verified_at_utc"])
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if not timestamp.endswith("Z") or parsed.tzinfo != timezone.utc:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise RecoveryC8SalvageRepairError(
            "repair verifier time is not canonical UTC"
        ) from error
    repo = Path(authority.repo_root)
    subtree = _subtree_path(authority)
    commit = value["commit"]
    if (
        _git(repo, "rev-parse", f"{commit}^{{commit}}") != commit
        or _git(repo, "rev-parse", f"{commit}^{{tree}}") != value["tree"]
        or _git(repo, "rev-parse", f"{commit}:{subtree}")
        != value["source_subtree_tree"]
        or _git(repo, "rev-parse", f"HEAD:{subtree}")
        != value["source_subtree_tree"]
    ):
        raise RecoveryC8SalvageRepairError("repair verifier Git identity drifted")
    try:
        _git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    except RecoveryC8SalvageRepairError as error:
        raise RecoveryC8SalvageRepairError(
            "current HEAD does not descend from the repair verifier"
        ) from error
    return value


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
            },
        },
    }


def _level_bytes(inventory: Sequence[Mapping[str, Any]], attempt: Path) -> dict[str, int]:
    result = {name: 0 for name in ("L0", "L1", "L2", "L3", "control")}
    for item in inventory:
        level = item.get("level") if isinstance(item, Mapping) else None
        if level in result:
            result[str(level)] += int(item.get("bytes", 0))
    result["control"] = sum(
        (attempt / name).stat().st_size for name in ("manifest.json", "status.json")
    )
    return result


def _scientific_failure(attempt: Path) -> bool:
    status = _read(attempt / "status.json")
    failure = status.get("failure")
    details = failure.get("details") if isinstance(failure, Mapping) else None
    return (
        status.get("status") == "failed"
        and status.get("terminal") is True
        and status.get("complete") is False
        and isinstance(details, Mapping)
        and details.get("type") in _SCIENTIFIC_FAILURE_TYPES
    )


def _validate_attempt(
    attempt: Path, unit: Any, *, plan: Any, estimate: Any, authority: Any,
    observed_git_state: Mapping[str, Any], execution_identity: Mapping[str, Any],
    verifier: Mapping[str, Any], origin: str, plan_index: int,
) -> tuple[dict[str, Any], list[str]]:
    checked = validate_execution_unit(
        attempt, expected_unit_identity=unit.key.to_dict(),
        expected_provenance_bindings=_expected_bindings(plan, estimate),
        observed_git_state=observed_git_state,
        caps=dict(estimate.caps["effective"]), require_runner_telemetry=True,
        trusted_recovery_context=authority,
    )
    errors = list(checked.errors)
    terminal_failed = (
        checked.valid and checked.status == "failed" and checked.terminal
        and not checked.complete and _scientific_failure(attempt)
    )
    if not checked.resumable and not terminal_failed:
        errors.append("attempt is not a valid terminal scientific unit")
    manifest = _read(attempt / "manifest.json") if (attempt / "manifest.json").is_file() else {}
    status = _read(attempt / "status.json") if (attempt / "status.json").is_file() else {}
    provenance = _read(attempt / "L0/provenance.json") if (attempt / "L0/provenance.json").is_file() else {}
    result = _read(attempt / "L0/result.json") if (attempt / "L0/result.json").is_file() else {}
    inventory = manifest.get("artifacts", []) if isinstance(manifest, Mapping) else []
    code = provenance.get("bindings", {}).get("code", {}) if provenance else {}
    if code != {"commit": execution_identity["commit"], "dirty": False}:
        errors.append("attempt code binding differs from its origin execution identity")
    state = "succeeded" if checked.resumable else "failed" if terminal_failed else "rejected"
    return {
        "plan_index": plan_index, "unit_key": unit.key.to_dict(),
        "origin": origin, "state": state,
        "source_root": str(attempt.parents[3]), "attempt_path": str(attempt),
        "execution_code_commit": execution_identity["commit"],
        "execution_git_identity": _plain(execution_identity),
        "verifier": _plain(verifier),
        "hashes": {
            "manifest_sha256": checked.manifest_sha256,
            "status_sha256": _sha(attempt / "status.json") if status else None,
            "provenance_sha256": _sha(attempt / "L0/provenance.json") if provenance else None,
            "result_sha256": _sha(attempt / "L0/result.json") if result else None,
            "unit_identity_sha256": checked.unit_identity_sha256,
            "artifact_inventory_sha256": canonical_sha256({"artifacts": inventory}),
        },
        "artifact_inventory": _plain(inventory),
        "scalars": _plain(result.get("scalars", {})),
        "provenance": _plain(provenance),
        "resources": _plain(result.get("resources", {})),
        "runner_telemetry": _plain(status.get("runner_telemetry", {})),
        "work_item_counts": _plain(unit.work_item_counts),
        "audit": bool(unit.audit),
        "l0_l1_cap_accounting_bytes": checked.l0_l1_bytes,
        "level_bytes": _level_bytes(inventory, attempt),
        "total_regular_file_bytes": checked.total_bytes,
        "validation_errors": errors,
    }, errors


def _plan_digests(plan: Any) -> list[str]:
    digests = [unit.key.canonical_digest for unit in plan.units]
    if len(digests) != EXPECTED_UNIT_COUNT or len(set(digests)) != EXPECTED_UNIT_COUNT:
        raise RecoveryC8SalvageRepairError(
            "repair requires the exact unique 2,180-unit C8 plan"
        )
    return digests


def _estimate(authority: Any, plan: Any, config: Any) -> Any:
    report_path = (
        Path(authority.repo_root)
        / authority.source_c8_record.document["c7_report"]["path"]
    )
    report = validate_fixed_c7_report(
        report_path, repo_root=authority.repo_root,
        trusted_recovery_context=authority,
    )
    measured = build_c8_measured_preflight(
        plan, config, report,
        trusted_recovery_context=authority,
    )
    from .stage_c_plan import StageCEstimate

    prior = StageCEstimate.from_dict(_read(authority.prior_gap_root / PREFLIGHT_FILE))
    if prior.to_dict() != measured.to_dict():
        raise RecoveryC8SalvageRepairError(
            "prior partial preflight differs from fresh frozen recomputation"
        )
    return measured


def _record_index(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "plan_index": item["plan_index"],
            "unit_digest": item["unit_key"]["canonical_digest"],
            "origin": item["origin"], "state": item["state"],
            "attempt_path": item["attempt_path"],
            "execution_git_identity": item["execution_git_identity"],
            "hashes": item["hashes"],
        }
        for item in records
    ]


def _prior_scope(authority: Any, plan: Any) -> dict[str, Any]:
    value = _read(authority.prior_gap_root / "salvage_scope.json")
    digest = value.pop("canonical_digest", None)
    if digest != canonical_sha256(value):
        raise RecoveryC8SalvageRepairError("prior partial scope digest drifted")
    value["canonical_digest"] = digest
    if value.get("plan_digest") != plan.canonical_digest:
        raise RecoveryC8SalvageRepairError("prior partial scope plan drifted")
    states = value.get("states")
    claims = value.get("claim_tokens")
    gap = value.get("gap_unit_digests")
    inherited = value.get("inherited_unit_digests")
    if (
        not isinstance(gap, list) or not isinstance(inherited, list)
        or not isinstance(states, Mapping) or set(states) != set(gap)
        or len(gap) + len(inherited) != EXPECTED_UNIT_COUNT
        or set(gap) & set(inherited) or claims != {}
        or any(state not in _SCOPE_STATES for state in states.values())
    ):
        raise RecoveryC8SalvageRepairError(
            "prior partial scope is not a closed zero-claim partition"
        )
    expected_status = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_status",
        "state": "running", "active_claim_count": 0,
        "gap_count": len(gap),
        "succeeded_count": sum(state == "succeeded" for state in states.values()),
        "failed_count": sum(state == "failed" for state in states.values()),
        "incomplete_count": sum(state == "incomplete" for state in states.values()),
        "pending_count": sum(state == "pending" for state in states.values()),
    }
    if _read(authority.prior_gap_root / "salvage_status.json") != expected_status:
        raise RecoveryC8SalvageRepairError(
            "prior partial status differs from its frozen scope"
        )
    return value


def audit_c8_salvage_repair_sources(
    authority: RecoveryC8SalvageRepairLaunchRecord, plan: Any, config: Any,
    estimate: Any, *, verifier_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = _authority(authority)
    verifier = (
        _capture_verifier(authority, "source-audit")
        if verifier_identity is None else _validate_verifier(authority, verifier_identity)
    )
    scope_check = validate_c8_execution_plan(plan, config)
    if not scope_check["valid"]:
        raise RecoveryC8SalvageRepairError(
            "repair plan is not exact C8: " + "; ".join(scope_check["violations"])
        )
    digests = _plan_digests(plan)
    by_digest = {unit.key.canonical_digest: unit for unit in plan.units}
    original_revision = authority.document["original_interrupted_root"]["evidence_commit"]
    prior_revision = authority.document["prior_partial_root"]["evidence_commit"]
    # Both inherited trees are immutable historical inputs.  A revision-scoped
    # ``git diff`` does not report untracked files, so explicitly require the
    # materialized original root to be byte-for-byte within its custody commit
    # before inventorying or validating any unit from it.  The prior partial
    # root receives the same check while its authority binding is loaded.
    _require_custody_clean(
        authority.repo_root, authority.source_root, original_revision,
    )
    original_before = _tree_inventory(
        authority.repo_root, authority.source_root, revision=original_revision,
    )
    prior_before = _tree_inventory(
        authority.repo_root, authority.prior_gap_root, revision=prior_revision,
    )
    original_auth = _read(authority.source_root / "c8_launch_authorization.json")
    original_code = original_auth["source"]["execution_git_state"]
    original_identity = _execution_identity(authority, original_code["commit"])
    original_checkpoint = _read(authority.source_root / "failure_table.json")
    rows = original_checkpoint.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_UNIT_COUNT:
        raise RecoveryC8SalvageRepairError("original checkpoint is not 2,180 rows")
    claimed = {row["unit_key"]["canonical_digest"]: row for row in rows}
    if set(claimed) != set(digests):
        raise RecoveryC8SalvageRepairError("original checkpoint coverage drifted")

    original_records: list[dict[str, Any]] = []
    original_digests: list[str] = []
    for index, digest in enumerate(digests):
        if claimed[digest].get("state") != "succeeded":
            continue
        attempts = attempt_directories(
            authority.source_root / "units" / digest / "attempts"
        )
        if len(attempts) != 1:
            raise RecoveryC8SalvageRepairError(
                f"original succeeded unit attempt count is not one: {digest}"
            )
        record, errors = _validate_attempt(
            attempts[0], by_digest[digest], plan=plan, estimate=estimate,
            authority=authority, observed_git_state=original_code,
            execution_identity=original_identity, verifier=verifier,
            origin=ORIGINAL_SOURCE_ORIGIN, plan_index=index,
        )
        if errors or record["state"] != "succeeded":
            raise RecoveryC8SalvageRepairError(
                f"original inherited unit is invalid: {digest}: "
                + "; ".join(errors[:10])
            )
        original_digests.append(digest)
        original_records.append(record)
    if len(original_records) != EXPECTED_ORIGINAL_COUNT:
        raise RecoveryC8SalvageRepairError(
            f"original source re-audit yielded {len(original_records)}, expected 1460"
        )

    prior_scope = _prior_scope(authority, plan)
    if (
        prior_scope["inherited_unit_digests"] != original_digests
        or prior_scope["gap_unit_digests"]
        != [digest for digest in digests if digest not in set(original_digests)]
    ):
        raise RecoveryC8SalvageRepairError(
            "prior partial partition differs from original source re-audit"
        )
    prior_code = {
        "commit": authority.original_salvage_record.launch_commit,
        "dirty": False,
    }
    prior_identity = _execution_identity(authority, prior_code["commit"])
    prior_records: list[dict[str, Any]] = []
    prior_digests: list[str] = []
    prior_failed: list[str] = []
    repair_digests: list[str] = []
    rejected_prior: list[dict[str, Any]] = []
    states = prior_scope["states"]
    for index, digest in enumerate(digests):
        if digest in set(original_digests):
            continue
        state = states[digest]
        attempts = attempt_directories(
            authority.prior_gap_root / "units" / digest / "attempts"
        )
        if state in _TERMINAL_STATES:
            if len(attempts) != 1:
                raise RecoveryC8SalvageRepairError(
                    f"prior terminal unit attempt count is not one: {digest}"
                )
            record, errors = _validate_attempt(
                attempts[0], by_digest[digest], plan=plan, estimate=estimate,
                authority=authority, observed_git_state=prior_code,
                execution_identity=prior_identity, verifier=verifier,
                origin=PRIOR_GAP_ORIGIN, plan_index=index,
            )
            if not errors and record["state"] == state:
                prior_digests.append(digest)
                prior_records.append(record)
                if state == "failed":
                    prior_failed.append(digest)
                continue
            # A checkpoint label is advisory.  A failed/succeeded candidate that
            # does not survive bytewise validation belongs to the repair
            # complement and its rejected attempt remains explicit evidence.
            rejected_prior.append({
                "plan_index": index, "unit_key": by_digest[digest].key.to_dict(),
                "attempt_path": str(attempts[0]),
                "reported_state": state,
                "errors": errors or [
                    f"validated state {record['state']} differs from reported {state}"
                ],
            })
            repair_digests.append(digest)
            continue
        if state not in {"pending", "incomplete"}:
            raise RecoveryC8SalvageRepairError(
                f"prior nonterminal unit has forbidden state: {digest}: {state}"
            )
        if state == "pending" and attempts:
            raise RecoveryC8SalvageRepairError(
                f"prior pending unit unexpectedly has an attempt: {digest}"
            )
        if state == "incomplete" and len(attempts) != 1:
            raise RecoveryC8SalvageRepairError(
                f"prior incomplete unit attempt count is not one: {digest}"
            )
        if attempts:
            rejected, rejection_errors = _validate_attempt(
                attempts[0], by_digest[digest], plan=plan, estimate=estimate,
                authority=authority, observed_git_state=prior_code,
                execution_identity=prior_identity, verifier=verifier,
                origin=PRIOR_GAP_ORIGIN, plan_index=index,
            )
            if not rejection_errors and rejected["state"] in _TERMINAL_STATES:
                raise RecoveryC8SalvageRepairError(
                    f"prior scope hides a valid terminal attempt: {digest}"
                )
            rejected_prior.append({
                "plan_index": index, "unit_key": by_digest[digest].key.to_dict(),
                "attempt_path": str(attempts[0]), "errors": rejection_errors,
            })
        repair_digests.append(digest)

    if (
        len(prior_records) != EXPECTED_PRIOR_COUNT
        or sum(item["state"] == "succeeded" for item in prior_records)
        != EXPECTED_PRIOR_SUCCEEDED_COUNT
        or len(prior_failed) != EXPECTED_PRIOR_FAILED_COUNT
        or len(repair_digests) != EXPECTED_REPAIR_COUNT
    ):
        raise RecoveryC8SalvageRepairError(
            "source re-audit is not exactly 1460 original / 568 prior succeeded "
            "/ 1 prior scientific failed / 151 repair"
        )
    groups = [set(original_digests), set(prior_digests), set(repair_digests)]
    if (
        any(groups[left] & groups[right] for left in range(3) for right in range(left))
        or set().union(*groups) != set(digests)
    ):
        raise RecoveryC8SalvageRepairError(
            "repair three-origin partition does not exactly cover the plan"
        )
    original_after = _tree_inventory(
        authority.repo_root, authority.source_root, revision=original_revision,
    )
    prior_after = _tree_inventory(
        authority.repo_root, authority.prior_gap_root, revision=prior_revision,
    )
    if original_before != original_after or prior_before != prior_after:
        raise RecoveryC8SalvageRepairError(
            "an inherited repair source changed during re-audit"
        )
    document = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_repair_source_audit",
        "status": "PASS", "holdout_accessed": False,
        "plan_digest": plan.canonical_digest,
        "original_unit_digests": original_digests,
        "prior_unit_digests": prior_digests,
        "prior_failed_unit_digests": prior_failed,
        "repair_unit_digests": repair_digests,
        "counts": {
            "original_succeeded": len(original_records),
            "prior_succeeded": EXPECTED_PRIOR_SUCCEEDED_COUNT,
            "prior_terminal_failed": len(prior_failed),
            "prior_terminal": len(prior_records),
            "repair": len(repair_digests),
            "expected": EXPECTED_UNIT_COUNT,
        },
        "original_root_inventory": original_before,
        "prior_gap_root_inventory": prior_before,
        "original_records": original_records,
        "prior_records": prior_records,
        "rejected_prior_attempts": rejected_prior,
        "verifier": verifier,
    }
    document["canonical_digest"] = canonical_sha256(document)
    return document


def _compact_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: _plain(value) for key, value in audit.items()
        if key not in {"original_records", "prior_records", "canonical_digest"}
    }
    result["original_record_index"] = _record_index(audit["original_records"])
    result["prior_record_index"] = _record_index(audit["prior_records"])
    result["canonical_digest"] = canonical_sha256(result)
    return result


def initialize_c8_salvage_repair_root(
    authority: RecoveryC8SalvageRepairLaunchRecord, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    if any(os.path.lexists(path) for path in (authority.repair_root, authority.reports_root)):
        raise RecoveryC8SalvageRepairError(
            "fresh repair and report roots must not already exist"
        )
    estimate = _estimate(authority, plan, config)
    execution_state = capture_authorized_c8_salvage_repair_source_state(
        authority, "initialize",
    )
    execution_identity = _execution_identity(authority, execution_state["commit"])
    audit = audit_c8_salvage_repair_sources(
        authority, plan, config, estimate,
        verifier_identity=_capture_verifier(authority, "initialize-audit"),
    )
    authority.reports_root.mkdir(parents=True, exist_ok=False)
    compact = _compact_audit(audit)
    audit_path = authority.reports_root / AUDIT_FILE
    _atomic_create_json(audit_path, compact)
    authority.repair_root.mkdir(parents=True, exist_ok=False)
    scope = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_repair_scope",
        "plan_digest": plan.canonical_digest,
        "source_audit_path": str(audit_path),
        "source_audit_digest": compact["canonical_digest"],
        "original_unit_digests": audit["original_unit_digests"],
        "prior_unit_digests": audit["prior_unit_digests"],
        "prior_failed_unit_digests": audit["prior_failed_unit_digests"],
        "repair_unit_digests": audit["repair_unit_digests"],
        "repair_execution_identity": execution_identity,
        "states": {digest: "pending" for digest in audit["repair_unit_digests"]},
        "claim_tokens": {}, "canonical_digest": None,
    }
    scope["canonical_digest"] = canonical_sha256({
        key: value for key, value in scope.items() if key != "canonical_digest"
    })
    _atomic_create_json(authority.repair_root / PLAN_FILE, plan.to_dict())
    _atomic_create_json(authority.repair_root / PREFLIGHT_FILE, estimate.to_dict())
    _atomic_create_json(authority.repair_root / SCOPE_FILE, scope)
    _atomic_create_json(
        authority.repair_root / STATUS_FILE, _runtime_status(scope, state="running"),
    )
    return {
        "status": "PASS", "original_count": EXPECTED_ORIGINAL_COUNT,
        "prior_terminal_count": EXPECTED_PRIOR_COUNT,
        "prior_terminal_failed_count": EXPECTED_PRIOR_FAILED_COUNT,
        "repair_count": EXPECTED_REPAIR_COUNT,
        "source_audit_digest": compact["canonical_digest"],
    }


def _read_preflight(authority: Any) -> Any:
    from .stage_c_plan import StageCEstimate

    return StageCEstimate.from_dict(_read(authority.repair_root / PREFLIGHT_FILE))


def _scope(authority: Any, plan: Any) -> dict[str, Any]:
    value = _read(authority.repair_root / SCOPE_FILE)
    claimed = value.pop("canonical_digest", None)
    if claimed != canonical_sha256(value):
        raise RecoveryC8SalvageRepairError("repair scope digest drifted")
    value["canonical_digest"] = claimed
    if value.get("plan_digest") != plan.canonical_digest:
        raise RecoveryC8SalvageRepairError("repair scope plan digest drifted")
    return value


def _frozen_audit(authority: Any, scope: Mapping[str, Any]) -> dict[str, Any]:
    expected = authority.reports_root / AUDIT_FILE
    if scope.get("source_audit_path") != str(expected):
        raise RecoveryC8SalvageRepairError("repair source audit path drifted")
    audit = _read(expected)
    claimed = audit.get("canonical_digest")
    if claimed != canonical_sha256({
        key: value for key, value in audit.items() if key != "canonical_digest"
    }):
        raise RecoveryC8SalvageRepairError("frozen repair source audit drifted")
    if claimed != scope.get("source_audit_digest"):
        raise RecoveryC8SalvageRepairError("repair scope/audit binding drifted")
    return audit


def _validate_partition(
    authority: Any, plan: Any, scope: Mapping[str, Any], *,
    source_reaudit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan_digests = _plan_digests(plan)
    original = scope.get("original_unit_digests")
    prior = scope.get("prior_unit_digests")
    failed = scope.get("prior_failed_unit_digests")
    repair = scope.get("repair_unit_digests")
    if not all(isinstance(value, list) for value in (original, prior, failed, repair)):
        raise RecoveryC8SalvageRepairError("repair scope partition is malformed")
    sets = [set(original), set(prior), set(repair)]
    if (
        [len(value) for value in (original, prior, repair)]
        != [EXPECTED_ORIGINAL_COUNT, EXPECTED_PRIOR_COUNT, EXPECTED_REPAIR_COUNT]
        or any(len(group) != len(value) for group, value in zip(sets, (original, prior, repair)))
        or any(sets[left] & sets[right] for left in range(3) for right in range(left))
        or set().union(*sets) != set(plan_digests)
        or original != [digest for digest in plan_digests if digest in sets[0]]
        or prior != [digest for digest in plan_digests if digest in sets[1]]
        or repair != [digest for digest in plan_digests if digest in sets[2]]
        or len(failed) != EXPECTED_PRIOR_FAILED_COUNT
        or not set(failed) <= sets[1]
    ):
        raise RecoveryC8SalvageRepairError(
            "repair scope is not the exact ordered 1460/569/151 partition"
        )
    states = scope.get("states")
    claims = scope.get("claim_tokens")
    if not isinstance(states, Mapping) or set(states) != sets[2]:
        raise RecoveryC8SalvageRepairError(
            "repair state table does not exactly cover the 151-unit repair set"
        )
    if any(state not in _SCOPE_STATES for state in states.values()):
        raise RecoveryC8SalvageRepairError("repair state table has an invalid state")
    if not isinstance(claims, Mapping):
        raise RecoveryC8SalvageRepairError("repair claim table is malformed")
    claimed_set = {digest for digest, state in states.items() if state == "claimed"}
    if set(claims) != claimed_set or any(
        not isinstance(token, str) or len(token) != 64
        or set(token) - set("0123456789abcdef")
        for token in claims.values()
    ):
        raise RecoveryC8SalvageRepairError(
            "repair claim table differs from claimed unit states"
        )
    identity = scope.get("repair_execution_identity")
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {
            "commit", "tree", "source_subtree_path",
            "source_subtree_tree", "dirty",
        }
        or identity.get("dirty") is not False
        or identity.get("source_subtree_path") != "src/vitnns"
    ):
        raise RecoveryC8SalvageRepairError(
            "repair execution identity is malformed"
        )
    audit = _frozen_audit(authority, scope)
    expected = {
        "original_unit_digests": original, "prior_unit_digests": prior,
        "prior_failed_unit_digests": failed, "repair_unit_digests": repair,
    }
    if audit.get("status") != "PASS" or audit.get("plan_digest") != plan.canonical_digest:
        raise RecoveryC8SalvageRepairError("frozen repair audit is not PASS")
    for name, value in expected.items():
        if audit.get(name) != value:
            raise RecoveryC8SalvageRepairError(
                f"repair scope differs from frozen audit: {name}"
            )
    if source_reaudit is not None:
        for name, value in expected.items():
            if source_reaudit.get(name) != value:
                raise RecoveryC8SalvageRepairError(
                    f"repair partition differs from source re-audit: {name}"
                )
        if (
            audit.get("original_root_inventory")
            != source_reaudit.get("original_root_inventory")
            or audit.get("prior_gap_root_inventory")
            != source_reaudit.get("prior_gap_root_inventory")
            or audit.get("original_record_index")
            != _record_index(source_reaudit.get("original_records", ()))
            or audit.get("prior_record_index")
            != _record_index(source_reaudit.get("prior_records", ()))
            or audit.get("rejected_prior_attempts")
            != source_reaudit.get("rejected_prior_attempts")
        ):
            raise RecoveryC8SalvageRepairError(
                "frozen repair audit differs from bytewise source re-audit"
            )
    return {
        "path": str(authority.reports_root / AUDIT_FILE),
        "file_sha256": _sha(authority.reports_root / AUDIT_FILE),
        "canonical_digest": audit["canonical_digest"],
    }


def _runtime_status(scope: Mapping[str, Any], *, state: str) -> dict[str, Any]:
    counts = Counter(scope["states"].values())
    return {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_repair_status",
        "state": state, "active_claim_count": int(counts["claimed"]),
        "repair_count": len(scope["repair_unit_digests"]),
        "succeeded_count": int(counts["succeeded"]),
        "failed_count": int(counts["failed"]),
        "incomplete_count": int(counts["incomplete"]),
        "pending_count": int(counts["pending"]),
    }


def _require_status(authority: Any, scope: Mapping[str, Any], *, state: str) -> None:
    if _read(authority.repair_root / STATUS_FILE) != _runtime_status(scope, state=state):
        raise RecoveryC8SalvageRepairError(
            f"repair status is not the exact {state} snapshot"
        )


def _publish_scope(authority: Any, scope: dict[str, Any], *, state: str) -> None:
    scope.pop("canonical_digest", None)
    scope["canonical_digest"] = canonical_sha256(scope)
    _atomic_replace_json(authority.repair_root / SCOPE_FILE, scope)
    _atomic_replace_json(
        authority.repair_root / STATUS_FILE, _runtime_status(scope, state=state),
    )


def _runtime_git_state(
    authority: Any, scope: Mapping[str, Any], raw: Any,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {"commit", "dirty"}:
        raise RecoveryC8SalvageRepairError(
            "repair runtime Git state must contain exactly commit and dirty"
        )
    value = _plain(raw)
    identity = scope["repair_execution_identity"]
    if value != {"commit": identity["commit"], "dirty": False}:
        raise RecoveryC8SalvageRepairError(
            "repair runtime Git state differs from initialized execution identity"
        )
    current = capture_authorized_c8_salvage_repair_source_state(
        authority, "repair-execution",
    )
    if current != value:
        raise RecoveryC8SalvageRepairError(
            "repair runtime Git state differs from live authority capture"
        )
    return value


def run_c8_salvage_repair_shard(
    authority: RecoveryC8SalvageRepairLaunchRecord, plan: Any, backend: Any, *,
    shard_index: int, shard_count: int, device: str = "cuda:0",
    runtime_git_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    authority = _authority(authority)
    if not (type(shard_index) is type(shard_count) is int and shard_count > 0
            and 0 <= shard_index < shard_count):
        raise ValueError("invalid repair shard coordinates")
    initial = _scope(authority, plan)
    _validate_partition(authority, plan, initial)
    code = _runtime_git_state(authority, initial, runtime_git_state)
    if getattr(backend, "trusted_recovery_context", None) != authority:
        raise RecoveryC8SalvageRepairError(
            "repair backend lacks the exact repair launch authority"
        )
    estimate = _read_preflight(authority)
    by_digest = {unit.key.canonical_digest: unit for unit in plan.units}
    requested = [
        digest for index, digest in enumerate(initial["repair_unit_digests"])
        if index % shard_count == shard_index
    ]
    results: list[dict[str, Any]] = []
    with _lifecycle_lock(authority.repair_root, exclusive=False):
        for digest in requested:
            token = secrets.token_hex(32)
            with _checkpoint_lock(authority.repair_root):
                scope = _scope(authority, plan)
                _validate_partition(authority, plan, scope)
                _require_status(authority, scope, state="running")
                state = scope["states"].get(digest)
                if state in _TERMINAL_STATES:
                    results.append({
                        "unit_key": by_digest[digest].key.to_dict(),
                        "state": state, "attempt_path": None,
                        "resumed": True, "failure": None,
                    })
                    continue
                if state != "pending":
                    raise RecoveryC8SalvageRepairError(
                        f"repair unit is not pending: {digest}"
                    )
                scope["states"][digest] = "claimed"
                scope["claim_tokens"][digest] = token
                _publish_scope(authority, scope, state="running")
            try:
                with _artifact_activity_lock(authority.repair_root, exclusive=False):
                    result = run_execution_unit(
                        by_digest[digest], output_root=authority.repair_root,
                        backend=backend, device=device, resume=False,
                        caps=dict(estimate.caps["effective"]),
                        preflight=estimate.to_dict(), plan=plan,
                        observed_git_state=code,
                    )
                result_document = _plain(result)
                state = result_document.get("state")
                attempts = attempt_directories(
                    authority.repair_root / "units" / digest / "attempts"
                )
                record = None
                errors: list[str] = []
                if state in _TERMINAL_STATES and len(attempts) == 1:
                    record, errors = _validate_attempt(
                        attempts[0], by_digest[digest], plan=plan, estimate=estimate,
                        authority=authority, observed_git_state=code,
                        execution_identity=scope["repair_execution_identity"],
                        verifier=_capture_verifier(authority, "post-unit"),
                        origin=REPAIR_ORIGIN,
                        plan_index=_plan_digests(plan).index(digest),
                    )
                if (
                    state not in _TERMINAL_STATES or len(attempts) != 1
                    or errors or record is None or record["state"] != state
                ):
                    state = "incomplete"
                    result_document = {
                        **result_document, "state": state,
                        "failure": {
                            "reason": "repair_post_run_validation_failed",
                            "errors": errors or [
                                "repair run did not produce one valid terminal attempt"
                            ],
                        },
                    }
            except BaseException:
                with _checkpoint_lock(authority.repair_root):
                    scope = _scope(authority, plan)
                    _validate_partition(authority, plan, scope)
                    if scope.get("claim_tokens", {}).get(digest) == token:
                        scope["states"][digest] = "incomplete"
                        scope["claim_tokens"].pop(digest, None)
                        _publish_scope(authority, scope, state="running")
                raise
            with _checkpoint_lock(authority.repair_root):
                scope = _scope(authority, plan)
                _validate_partition(authority, plan, scope)
                if (
                    scope["states"].get(digest) != "claimed"
                    or scope["claim_tokens"].get(digest) != token
                ):
                    raise RecoveryC8SalvageRepairError(
                        f"repair claim changed during execution: {digest}"
                    )
                scope["states"][digest] = state
                scope["claim_tokens"].pop(digest, None)
                _publish_scope(authority, scope, state="running")
            results.append(result_document)
    return results


def _require_finalized_snapshot(
    authority: Any, plan: Any, scope: Mapping[str, Any], *,
    source_reaudit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = _validate_partition(
        authority, plan, scope, source_reaudit=source_reaudit,
    )
    if scope.get("claim_tokens"):
        raise RecoveryC8SalvageRepairError("finalized repair retains claims")
    nonterminal = {
        digest: state for digest, state in scope["states"].items()
        if state not in _TERMINAL_STATES
    }
    if nonterminal:
        raise RecoveryC8SalvageRepairError(
            f"finalized repair contains nonterminal states: {Counter(nonterminal.values())}"
        )
    _require_status(authority, scope, state="finalized")
    return binding


def _c7_report_binding(authority: Any) -> dict[str, Any]:
    report_path = (
        Path(authority.repo_root)
        / authority.source_c8_record.document["c7_report"]["path"]
    )
    report = validate_fixed_c7_report(
        report_path, repo_root=authority.repo_root,
        trusted_recovery_context=authority,
    )
    source = authority.source_c8_record.document["c7_report"]
    recomputed = report.get("bindings", {}).get("recomputed_estimate")
    control_bytes = report.get("measurement_boundary", {}).get(
        "source_batch_control_bytes"
    )
    if not isinstance(recomputed, Mapping) or type(control_bytes) is not int:
        raise RecoveryC8SalvageRepairError(
            "validated C7 report lacks frozen storage operands"
        )
    return {
        "c7_report_path": str(report_path),
        "c7_report_file_sha256": _sha(report_path),
        "c7_report_canonical_digest": report["canonical_digest"],
        "c7_code_commit": source["source_commit"],
        "recomputed_estimate": _plain(recomputed),
        "source_batch_control_bytes": control_bytes,
    }


def _control_hashes(authority: Any) -> dict[str, str]:
    result = {
        f"original/{name}": str(digest)
        for name, digest in authority.document["original_interrupted_root"][
            "control_sha256"
        ].items()
    }
    result.update({
        f"prior/{name}": str(digest)
        for name, digest in authority.document["prior_partial_root"][
            "control_sha256"
        ].items()
    })
    result.update({
        f"repair/{name}": _sha(authority.repair_root / name)
        for name in (PLAN_FILE, PREFLIGHT_FILE, SCOPE_FILE, STATUS_FILE)
    })
    result[f"reports/{AUDIT_FILE}"] = _sha(authority.reports_root / AUDIT_FILE)
    return dict(sorted(result.items()))


def build_c8_salvage_repair_combined_manifest(
    authority: RecoveryC8SalvageRepairLaunchRecord, plan: Any, config: Any, *,
    verifier_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = _authority(authority)
    scope = _scope(authority, plan)
    verifier = (
        _capture_verifier(authority, "combine")
        if verifier_identity is None else _validate_verifier(authority, verifier_identity)
    )
    estimate = _read_preflight(authority)
    measured = _estimate(authority, plan, config)
    if estimate.to_dict() != measured.to_dict():
        raise RecoveryC8SalvageRepairError(
            "repair preflight differs from source recomputation"
        )
    audit = audit_c8_salvage_repair_sources(
        authority, plan, config, estimate, verifier_identity=verifier,
    )
    frozen_binding = _validate_partition(
        authority, plan, scope, source_reaudit=audit,
    )
    original = {
        item["unit_key"]["canonical_digest"]: item
        for item in audit["original_records"]
    }
    prior = {
        item["unit_key"]["canonical_digest"]: item
        for item in audit["prior_records"]
    }
    repair_set = set(scope["repair_unit_digests"])
    repair_identity = _plain(scope["repair_execution_identity"])
    code = {"commit": repair_identity["commit"], "dirty": False}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, unit in enumerate(plan.units):
        digest = unit.key.canonical_digest
        if digest in original:
            rows.append(original[digest])
            continue
        if digest in prior:
            rows.append(prior[digest])
            continue
        attempts = attempt_directories(
            authority.repair_root / "units" / digest / "attempts"
        )
        if digest not in repair_set or len(attempts) != 1:
            errors.append(f"{digest}: repair attempt count is not one")
            continue
        record, unit_errors = _validate_attempt(
            attempts[0], unit, plan=plan, estimate=estimate, authority=authority,
            observed_git_state=code, execution_identity=repair_identity,
            verifier=verifier, origin=REPAIR_ORIGIN, plan_index=index,
        )
        if scope["states"].get(digest) != record["state"]:
            unit_errors.append("repair record state differs from repair scope")
        if record["state"] not in _TERMINAL_STATES:
            unit_errors.append("repair record is not a valid terminal unit")
        errors.extend(f"{digest}: {error}" for error in unit_errors)
        rows.append(record)
    units_dir = authority.repair_root / "units"
    unexpected = [] if not units_dir.exists() else sorted(
        path.name for path in units_dir.iterdir() if path.name not in repair_set
    )
    if unexpected:
        errors.append(f"repair root contains unexpected unit directories: {unexpected}")
    expected_order = _plan_digests(plan)
    observed_order = [item["unit_key"]["canonical_digest"] for item in rows]
    if observed_order != expected_order:
        errors.append("combined repair records differ from frozen plan order/coverage")
    if any(
        item.get("state") not in _TERMINAL_STATES
        or item.get("validation_errors") != []
        for item in rows
    ):
        errors.append("combined repair records are not all validated terminal units")
    if len(rows) != EXPECTED_UNIT_COUNT or errors:
        raise RecoveryC8SalvageRepairError(
            "repair union is incomplete or invalid: " + "; ".join(errors[:20])
        )
    origin_counts = dict(sorted(Counter(item["origin"] for item in rows).items()))
    expected_origins = {
        ORIGINAL_SOURCE_ORIGIN: EXPECTED_ORIGINAL_COUNT,
        PRIOR_GAP_ORIGIN: EXPECTED_PRIOR_COUNT, REPAIR_ORIGIN: EXPECTED_REPAIR_COUNT,
    }
    if origin_counts != dict(sorted(expected_origins.items())):
        raise RecoveryC8SalvageRepairError("repair origin counts drifted")
    source_inventory = audit["original_root_inventory"]
    prior_inventory = audit["prior_gap_root_inventory"]
    repair_inventory = _tree_inventory(authority.repo_root, authority.repair_root)
    root_inventories = {
        ORIGINAL_SOURCE_ORIGIN: source_inventory,
        PRIOR_GAP_ORIGIN: prior_inventory,
        REPAIR_ORIGIN: repair_inventory,
    }
    identities = {
        ORIGINAL_SOURCE_ORIGIN: rows[0]["execution_git_identity"],
        PRIOR_GAP_ORIGIN: prior[next(iter(prior))]["execution_git_identity"],
        REPAIR_ORIGIN: repair_identity,
    }
    if len({item["commit"] for item in identities.values()}) != 3:
        raise RecoveryC8SalvageRepairError(
            "repair union does not preserve three distinct execution identities"
        )
    control_hashes = _control_hashes(authority)
    c7_binding = _c7_report_binding(authority)
    manifest = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_repair_combined_manifest",
        "status": "PASS", "holdout_accessed": False,
        "plan_digest": plan.canonical_digest,
        "repair_authority": {
            "path": str(authority.path), "file_sha256": authority.file_sha256,
            "canonical_digest": authority.canonical_digest,
            "launch_commit": authority.launch_commit,
        },
        "prior_salvage_authority": _plain(
            authority.document["original_salvage_authority"]
        ),
        "source_audit": frozen_binding,
        "source_reaudit_digest": audit["canonical_digest"],
        "origin_counts": origin_counts,
        "terminal_state_counts": dict(sorted(
            Counter(item["state"] for item in rows).items()
        )),
        "root_inventories": root_inventories,
        "producer_identities": identities,
        "verifier": verifier, "records": rows,
        "budget_evidence": {
            "bindings": {
                "source_tree_total_bytes": sum(
                    inventory["total_bytes"] for inventory in root_inventories.values()
                ),
                "control_file_sha256": control_hashes,
                "c7_report_binding": c7_binding,
            },
            "c7_report_binding": c7_binding,
        },
        "combined_artifact_manifest_digest": canonical_sha256({
            "domain": "vitnns-c8-salvage-repair-union-v1", "records": rows,
        }),
    }
    manifest["canonical_digest"] = canonical_sha256(manifest)
    return manifest


def finalize_c8_salvage_repair(
    authority: RecoveryC8SalvageRepairLaunchRecord, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    with (
        _lifecycle_lock(authority.repair_root, exclusive=True),
        _checkpoint_lock(authority.repair_root),
        _artifact_activity_lock(authority.repair_root, exclusive=True),
    ):
        scope = _scope(authority, plan)
        _validate_partition(authority, plan, scope)
        nonterminal = {
            digest: state for digest, state in scope["states"].items()
            if state not in _TERMINAL_STATES
        }
        if (
            len(scope["states"]) != EXPECTED_REPAIR_COUNT
            or nonterminal or scope.get("claim_tokens")
        ):
            return {
                "status": "BLOCKED", "successful": False,
                "orchestration_complete": False, "manifest_written": False,
                "repair_count": len(scope["states"]),
                "repair_state_counts": dict(sorted(
                    Counter(scope["states"].values()).items()
                )),
                "active_claim_count": len(scope.get("claim_tokens", {})),
                "errors": [
                    "all 151 repair units must be valid terminal units with zero claims"
                ],
            }
        verifier = _capture_verifier(authority, "finalize")
        _publish_scope(authority, scope, state="finalized")
        try:
            manifest = build_c8_salvage_repair_combined_manifest(
                authority, plan, config, verifier_identity=verifier,
            )
            if any(
                record.get("state") not in _TERMINAL_STATES
                or record.get("validation_errors") != []
                for record in manifest.get("records", ())
            ):
                raise RecoveryC8SalvageRepairError(
                    "combined repair manifest contains invalid/nonterminal records"
                )
            authority.reports_root.mkdir(parents=True, exist_ok=True)
            _create_or_accept(
                authority.combined_manifest_path, manifest,
                "repair combined manifest",
            )
        except BaseException:
            _publish_scope(authority, scope, state="blocked")
            raise
    return manifest


def validate_c8_salvage_repair_from_source(
    authority: RecoveryC8SalvageRepairLaunchRecord, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    with (
        _lifecycle_lock(authority.repair_root, exclusive=True),
        _checkpoint_lock(authority.repair_root),
        _artifact_activity_lock(authority.repair_root, exclusive=True),
    ):
        scope = _scope(authority, plan)
        _require_finalized_snapshot(authority, plan, scope)
        persisted = _read(authority.combined_manifest_path)
        verifier = _validate_verifier(authority, persisted.get("verifier"))
        rebuilt = build_c8_salvage_repair_combined_manifest(
            authority, plan, config, verifier_identity=verifier,
        )
        if persisted != rebuilt:
            raise RecoveryC8SalvageRepairError(
                "persisted repair manifest differs from source recomputation"
            )
        if any(
            record.get("state") not in _TERMINAL_STATES
            or record.get("validation_errors") != []
            for record in persisted.get("records", ())
        ):
            raise RecoveryC8SalvageRepairError(
                "persisted repair manifest contains invalid terminal evidence"
            )
        return persisted


def _selection_generator_state(authority: Any, selection: Any, report: Any) -> dict[str, Any]:
    generated = selection.get("generated_by") if isinstance(selection, Mapping) else None
    if generated is None and isinstance(report, Mapping):
        generated = report.get("generated_by")
        if generated is None:
            generated = report.get("assembler_identity", {}).get("commit")
    prefix = "vitnns.stage_c_c8_salvage_repair_selection@"
    commit = generated[len(prefix):] if isinstance(generated, str) and generated.startswith(prefix) else generated
    if (
        not isinstance(commit, str) or len(commit) != 40
        or set(commit) - set("0123456789abcdef")
        or _git(Path(authority.repo_root), "rev-parse", f"{commit}^{{commit}}") != commit
    ):
        raise RecoveryC8SalvageRepairError(
            "repair selection generator identity is invalid"
        )
    return {"commit": commit, "dirty": False}


def _validated_selection_bindings(
    authority: Any, plan: Any, config: Any,
) -> dict[str, Any]:
    selection = _read(authority.selection_path)
    report = _read(authority.selection_report_path)
    selection_hash = _sha(authority.selection_path)
    report_hash = _sha(authority.selection_report_path)
    generator = _selection_generator_state(authority, selection, report)
    from .stage_c_c8_salvage_repair_selection import (
        validate_c8_salvage_repair_selection_report,
    )

    validate_c8_salvage_repair_selection_report(
        authority, plan, config, report, selection,
        generator_git_state=generator,
    )
    if report.get("status") != "PASS" or selection.get("canonical_digest") is None:
        raise RecoveryC8SalvageRepairError(
            "repair completion requires a PASS selection and selection report"
        )
    if (
        _sha(authority.selection_path) != selection_hash
        or _sha(authority.selection_report_path) != report_hash
    ):
        raise RecoveryC8SalvageRepairError(
            "repair selection changed during completion validation"
        )
    return {
        "selection": {
            "path": str(authority.selection_path), "file_sha256": selection_hash,
            "canonical_digest": selection["canonical_digest"],
        },
        "selection_report": {
            "path": str(authority.selection_report_path),
            "file_sha256": report_hash,
            "canonical_digest": report["canonical_digest"],
        },
    }


def build_c8_salvage_repair_completion_report(
    authority: RecoveryC8SalvageRepairLaunchRecord, plan: Any, config: Any, *,
    verifier_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authority = _authority(authority)
    manifest = validate_c8_salvage_repair_from_source(authority, plan, config)
    selection = _validated_selection_bindings(authority, plan, config)
    verifier = (
        _capture_verifier(authority, "completion")
        if verifier_identity is None else _validate_verifier(authority, verifier_identity)
    )
    counts = Counter(item["state"] for item in manifest["records"])
    if (
        counts["succeeded"] + counts["failed"] != EXPECTED_UNIT_COUNT
        or set(counts) - _TERMINAL_STATES
        or counts["failed"] < EXPECTED_PRIOR_FAILED_COUNT
    ):
        raise RecoveryC8SalvageRepairError(
            "repair completion requires exactly 2,180 validated terminal units"
        )
    report = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_repair_completion_report",
        "status": "PASS", "c8_pass": True,
        "repair_authority": {
            "path": str(authority.path), "file_sha256": authority.file_sha256,
            "canonical_digest": authority.canonical_digest,
            "launch_commit": authority.launch_commit,
        },
        "prior_salvage_authority": _plain(
            authority.document["original_salvage_authority"]
        ),
        "amendment": _plain(authority.document["amendment"]),
        "contracts": _plain(authority.document["contracts"]),
        "model": _plain(authority.document["model"]),
        "combined_manifest": {
            "path": str(authority.combined_manifest_path),
            "file_sha256": _sha(authority.combined_manifest_path),
            "canonical_digest": manifest["canonical_digest"],
            "combined_artifact_manifest_digest":
                manifest["combined_artifact_manifest_digest"],
        },
        "selection": selection["selection"],
        "selection_report": selection["selection_report"],
        "source_audit": _plain(manifest["source_audit"]),
        "source_reaudit_digest": manifest["source_reaudit_digest"],
        "origin_counts": manifest["origin_counts"],
        "terminal_state_counts": dict(sorted(counts.items())),
        "root_inventories": manifest["root_inventories"],
        "producer_identities": manifest["producer_identities"],
        "budget_evidence": manifest["budget_evidence"],
        "verifier": verifier,
        "flags": {
            "holdout_accessed": False, "selection_performed": True,
            "bootstrap_performed": False, "holm_performed": False,
            "decision_performed": False, "c9_authorized": False,
            "c10_authorized": False, "stage_d_authorized": False,
        },
    }
    report["canonical_digest"] = canonical_sha256(report)
    return report


def write_c8_salvage_repair_completion_report(
    authority: RecoveryC8SalvageRepairLaunchRecord, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    if os.path.lexists(authority.combined_report_path):
        if authority.combined_report_path.is_symlink() or not authority.combined_report_path.is_file():
            raise RecoveryC8SalvageRepairError(
                "existing repair completion report is unsafe"
            )
        return validate_c8_salvage_repair_completion_report(authority, plan, config)
    report = build_c8_salvage_repair_completion_report(authority, plan, config)
    _create_or_accept(
        authority.combined_report_path, report, "repair completion report",
    )
    return report


def validate_c8_salvage_repair_completion_report(
    authority: RecoveryC8SalvageRepairLaunchRecord, plan: Any, config: Any,
) -> dict[str, Any]:
    authority = _authority(authority)
    persisted = _read(authority.combined_report_path)
    if (
        persisted.get("canonical_digest")
        != canonical_sha256({
            key: value for key, value in persisted.items()
            if key != "canonical_digest"
        })
        or persisted.get("status") != "PASS"
        or persisted.get("c8_pass") is not True
    ):
        raise RecoveryC8SalvageRepairError(
            "repair completion report identity or digest is invalid"
        )
    verifier = _validate_verifier(authority, persisted.get("verifier"))
    rebuilt = build_c8_salvage_repair_completion_report(
        authority, plan, config, verifier_identity=verifier,
    )
    if persisted != rebuilt:
        raise RecoveryC8SalvageRepairError(
            "persisted repair completion differs from source recomputation"
        )
    return persisted


# Compatibility aliases retained for integration code drafted against the
# first repair design.  The canonical names above are used by the CLI.
initialize_c8_salvage_repair_gap_root = initialize_c8_salvage_repair_root
run_c8_salvage_repair_gap_shard = run_c8_salvage_repair_shard
build_c8_salvage_repair_manifest = build_c8_salvage_repair_combined_manifest


__all__ = [
    "ORIGINAL_SOURCE_ORIGIN", "PRIOR_GAP_ORIGIN", "REPAIR_ORIGIN",
    "audit_c8_salvage_repair_sources",
    "initialize_c8_salvage_repair_root",
    "initialize_c8_salvage_repair_gap_root",
    "run_c8_salvage_repair_shard", "run_c8_salvage_repair_gap_shard",
    "build_c8_salvage_repair_combined_manifest",
    "build_c8_salvage_repair_manifest", "finalize_c8_salvage_repair",
    "validate_c8_salvage_repair_from_source",
    "build_c8_salvage_repair_completion_report",
    "write_c8_salvage_repair_completion_report",
    "validate_c8_salvage_repair_completion_report",
]
