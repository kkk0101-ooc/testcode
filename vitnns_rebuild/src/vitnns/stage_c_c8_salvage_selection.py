"""Selection over a source-recomputed virtual C8 salvage union."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .recovery_c8_salvage import (
    RecoveryC8SalvageError, RecoveryC8SalvageLaunchRecord,
    load_recovery_c8_salvage_launch_record,
)
from .stage_c_c8_salvage import (
    GAP_ORIGIN, SOURCE_ORIGIN, _capture_verifier_identity,
    _validate_verifier_identity,
    validate_c8_salvage_from_source,
)
from .stage_c_c8 import validate_fixed_c7_report
from .stage_c_provenance import canonical_sha256
from . import stage_c_selection as base


_RULES_VERSION = "stage-c-c8-salvage-selection-v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryC8SalvageError(f"{path} is not a JSON object")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> str:
    payload = _pretty_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _create_or_accept_exact(
    path: Path, value: Mapping[str, Any], *, context: str,
) -> str:
    payload = _pretty_json_bytes(value)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file():
            raise RecoveryC8SalvageError(
                f"existing {context} is not a safe regular file"
            )
        observed = path.read_bytes()
        if observed != payload:
            raise RecoveryC8SalvageError(
                f"existing {context} differs from source recomputation"
            )
        return hashlib.sha256(observed).hexdigest()
    try:
        return _write(path, value)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RecoveryC8SalvageError(
                f"raced {context} differs from source recomputation"
            )
        return hashlib.sha256(payload).hexdigest()


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _plain(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    raise RecoveryC8SalvageError(
        f"salvage selection value is not JSON-compatible: {type(value).__name__}"
    )


def _authority(record: Any) -> RecoveryC8SalvageLaunchRecord:
    if type(record) is not RecoveryC8SalvageLaunchRecord:
        raise RecoveryC8SalvageError("validated salvage authority required")
    checked = load_recovery_c8_salvage_launch_record(
        record.path, repo_root=record.repo_root, schema_path=record.schema_path,
        require_tracked=True, require_fresh_outputs=False,
    )
    if checked != record:
        raise RecoveryC8SalvageError("salvage authority differs from source bytes")
    return checked


def _scientific_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 2180:
        raise RecoveryC8SalvageError("combined manifest must contain 2180 records")
    result = []
    seen: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise RecoveryC8SalvageError(f"combined record {index} is invalid")
        if raw.get("origin") not in {SOURCE_ORIGIN, GAP_ORIGIN}:
            raise RecoveryC8SalvageError(f"combined record {index} origin is invalid")
        if raw.get("state") != "succeeded" or raw.get("validation_errors") != []:
            raise RecoveryC8SalvageError(f"combined record {index} is not usable")
        try:
            if raw["plan_index"] != index:
                raise RecoveryC8SalvageError(
                    f"combined record {index} plan index drifted"
                )
            unit_digest = raw["unit_key"]["canonical_digest"]
            if not isinstance(unit_digest, str) or unit_digest in seen:
                raise RecoveryC8SalvageError(
                    f"combined record {index} unit identity is invalid or duplicate"
                )
            seen.add(unit_digest)
            result.append({
                "plan_index": raw["plan_index"], "unit_key": raw["unit_key"],
                "state": raw["state"], "trusted_by_batch_validator": True,
                "hashes": {
                    "manifest_sha256": raw["hashes"]["manifest_sha256"],
                    "status_sha256": raw["hashes"]["status_sha256"],
                    "selected_status_sha256": raw["hashes"]["status_sha256"],
                    "provenance_sha256": raw["hashes"]["provenance_sha256"],
                    "result_sha256": raw["hashes"]["result_sha256"],
                    "unit_identity_sha256": raw["hashes"]["unit_identity_sha256"],
                },
                "scalars": raw["scalars"], "provenance": raw["provenance"],
                "resources": raw["resources"], "runner_telemetry": raw["runner_telemetry"],
                "artifact_inventory": raw["artifact_inventory"],
                "work_item_counts": raw["work_item_counts"], "audit": raw["audit"],
                "validation_errors": [], "attempt_path": raw["attempt_path"],
                "l0_l1_cap_accounting_bytes": raw["l0_l1_cap_accounting_bytes"],
                "level_bytes": raw["level_bytes"],
                "total_regular_file_bytes": raw["total_regular_file_bytes"],
            })
        except (KeyError, TypeError) as error:
            raise RecoveryC8SalvageError(
                f"combined record {index} lacks required scientific evidence"
            ) from error
    return result


def _common_inputs(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "model_sha256": ("model", "state_dict_sha256"),
        "dependency_lock_sha256": ("dependencies", "lock_sha256"),
        "registry_sha256": ("registry", "sha256"),
        "registry_metadata_sha256": ("registry", "metadata_sha256"),
        "protocol_sha256": ("protocol", "sha256"),
        "acceptance_sha256": ("acceptance", "sha256"),
        "config_sha256": ("configuration", "sha256"),
        "config_schema_sha256": ("configuration", "schema_sha256"),
    }
    values: dict[str, Any] = {}
    for name, path in fields.items():
        try:
            observed = {
                record["provenance"]["bindings"][path[0]][path[1]]
                for record in records
            }
        except (KeyError, TypeError) as error:
            raise RecoveryC8SalvageError(
                f"combined records lack common input binding: {name}"
            ) from error
        if len(observed) != 1:
            raise RecoveryC8SalvageError(f"mixed-source common input drift: {name}")
        values[name] = observed.pop()
    return values


def _git_state(value: Any, context: str) -> dict[str, Any]:
    try:
        return base._selection_generator_code(value)
    except Exception as error:
        raise RecoveryC8SalvageError(f"invalid {context}: {error}") from error


def _generator_identity(
    authority: RecoveryC8SalvageLaunchRecord, value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryC8SalvageError(
            "selection generator identity must be a mapping"
        )
    if set(value) == {"commit", "dirty"}:
        expected = _git_state(value, "selection generator Git state")
        verifier = _capture_verifier_identity(authority)
        observed = _git_state(
            {"commit": verifier["commit"], "dirty": verifier["dirty"]},
            "captured selection generator Git state",
        )
        if observed != expected:
            raise RecoveryC8SalvageError(
                "captured selection generator differs from expected Git state"
            )
    else:
        verifier = _validate_verifier_identity(authority, value)
    generator = _git_state(
        {"commit": verifier["commit"], "dirty": verifier["dirty"]},
        "selection generator Git state",
    )
    return verifier


def _origin_execution_identities(
    manifest: Mapping[str, Any], authority: RecoveryC8SalvageLaunchRecord,
) -> dict[str, dict[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise RecoveryC8SalvageError("combined manifest records are invalid")
    identities: dict[str, dict[str, Any] | None] = {
        SOURCE_ORIGIN: None, GAP_ORIGIN: None,
    }
    observed_counts = {SOURCE_ORIGIN: 0, GAP_ORIGIN: 0}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RecoveryC8SalvageError(f"combined record {index} is invalid")
        origin = record.get("origin")
        if origin not in identities:
            raise RecoveryC8SalvageError(f"combined record {index} origin is invalid")
        commit = record.get("execution_code_commit")
        try:
            code = record["provenance"]["bindings"]["code"]
            identity = _plain(record["execution_git_identity"])
        except (KeyError, TypeError) as error:
            raise RecoveryC8SalvageError(
                f"combined record {index} lacks execution code provenance"
            ) from error
        if (
            not isinstance(commit, str) or len(commit) != 40
            or set(commit) - set("0123456789abcdef")
            or not isinstance(code, Mapping)
            or code.get("commit") != commit
            or code.get("dirty") is not False
            or not isinstance(identity, dict)
            or identity.get("commit") != commit
            or identity.get("dirty") is not False
            or identity.get("source_subtree_path") != "src/vitnns"
            or not isinstance(identity.get("tree"), str)
            or len(identity["tree"]) != 40
            or set(identity["tree"]) - set("0123456789abcdef")
            or not isinstance(identity.get("source_subtree_tree"), str)
            or len(identity["source_subtree_tree"]) != 40
            or set(identity["source_subtree_tree"]) - set("0123456789abcdef")
        ):
            raise RecoveryC8SalvageError(
                f"combined record {index} execution identity is invalid"
            )
        prior = identities[origin]
        if prior is not None and identity != prior:
            raise RecoveryC8SalvageError(
                f"{origin} records bind multiple execution identities"
            )
        identities[origin] = identity
        observed_counts[origin] += 1
    expected_counts = manifest.get("origin_counts")
    if expected_counts != {
        origin: observed_counts[origin] for origin in sorted(observed_counts)
    }:
        raise RecoveryC8SalvageError("combined manifest origin counts drifted")
    for origin, identity in identities.items():
        if identity is None:
            raise RecoveryC8SalvageError(
                f"{origin} records do not bind exactly one real execution commit"
            )
    try:
        retained = authority.document["source_interrupted_root"]["execution_commit"]
    except (KeyError, TypeError, AttributeError) as error:
        raise RecoveryC8SalvageError(
            "salvage authority lacks execution identity bindings"
        ) from error
    if identities[SOURCE_ORIGIN]["commit"] != retained:
        raise RecoveryC8SalvageError(
            "inherited records differ from the authorized source execution commit"
        )
    declared = manifest.get("producer_identities")
    if not isinstance(declared, Mapping) or (
        declared.get("interrupted_execution") != identities[SOURCE_ORIGIN]
        or declared.get("gap_execution") != identities[GAP_ORIGIN]
    ):
        raise RecoveryC8SalvageError(
            "combined manifest producer identities differ from unit origins"
        )
    if identities[SOURCE_ORIGIN]["commit"] == identities[GAP_ORIGIN]["commit"]:
        raise RecoveryC8SalvageError(
            "inherited and gap records must retain distinct execution commits"
        )
    return {origin: identity for origin, identity in identities.items()}


def _budget_evidence(
    authority: RecoveryC8SalvageLaunchRecord, manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        c7_binding = authority.source_c8_record.document["c7_report"]
        c7_path = authority.repo_root / c7_binding["path"]
        c7_report = validate_fixed_c7_report(
            c7_path, repo_root=authority.repo_root,
            trusted_recovery_context=authority,
        )
        report_bindings = c7_report["bindings"]
        measurement = c7_report["measurement_boundary"]
        recomputed = report_bindings["recomputed_estimate"]
        control_bytes = measurement["source_batch_control_bytes"]
        source_bytes = manifest["source_root_inventory"]["total_bytes"]
        gap_bytes = manifest["gap_root_inventory"]["total_bytes"]
        controls = authority.document["source_interrupted_root"]["control_sha256"]
    except Exception as error:
        raise RecoveryC8SalvageError(
            "salvage source lacks frozen post-C8 budget evidence"
        ) from error
    if (
        type(control_bytes) is not int or control_bytes < 0
        or type(source_bytes) is not int or source_bytes < 0
        or type(gap_bytes) is not int or gap_bytes < 0
        or not isinstance(recomputed, Mapping)
        or not isinstance(controls, Mapping)
    ):
        raise RecoveryC8SalvageError(
            "salvage post-C8 budget operands are invalid"
        )
    return {
        "records": list(records),
        "bindings": {
            "source_tree_total_bytes": source_bytes + gap_bytes,
            "control_file_sha256": _plain(controls),
        },
        "c7_report_binding": {
            "recomputed_estimate": _plain(recomputed),
            "source_batch_control_bytes": control_bytes,
        },
    }


def _post_c8_budget_from_source(
    authority: RecoveryC8SalvageLaunchRecord, manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence = _budget_evidence(authority, manifest, records)
    try:
        return base._post_c8_budget(evidence)
    except Exception as error:
        raise RecoveryC8SalvageError(
            f"salvage post-C8 budget recomputation failed: {error}"
        ) from error


def _global_class_failures(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    class_records = 0
    for record in records:
        try:
            key = record["unit_key"]
            method_id = key["method_id"]
            sample_id = key["sample_id"]
        except (KeyError, TypeError) as error:
            raise RecoveryC8SalvageError(
                "salvage record lacks class-unit identity"
            ) from error
        if method_id not in base.CLASS_METHOD_IDS:
            continue
        class_records += 1
        try:
            errors = base._record_validity(record, f"{sample_id}:{method_id}")
        except Exception as error:
            raise RecoveryC8SalvageError(
                f"class-unit validity recomputation failed: {sample_id}:{method_id}: {error}"
            ) from error
        failures.extend({
            "candidate_key": None, "sample_id": sample_id,
            "reason": f"global_unit:{method_id}:{reason}",
            "unit_key": _plain(key),
        } for reason in errors)
    expected = len(base.CALIBRATION_SAMPLE_IDS) * len(base.CLASS_METHOD_IDS)
    if class_records != expected:
        raise RecoveryC8SalvageError(
            f"salvage records require exactly {expected} global class units"
        )
    return failures


def _derive_c8_salvage_selection(
    authority: RecoveryC8SalvageLaunchRecord, manifest: Mapping[str, Any], *,
    generator_git_state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive selection outputs only from a source-recomputed union."""

    generator_identity = _generator_identity(authority, generator_git_state)
    generator = _git_state(
        {
            "commit": generator_identity["commit"],
            "dirty": generator_identity["dirty"],
        },
        "selection generator Git state",
    )
    records = _scientific_records(manifest)
    producer_identities = _origin_execution_identities(manifest, authority)
    candidates: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for block in range(12):
            for head in range(12):
                candidate, audit, local = base._build_candidate(
                    records, block, head,
                )
                candidates.append(candidate)
                audits.append(audit)
                failures.extend(local)
    except Exception as error:
        raise RecoveryC8SalvageError(
            f"salvage candidate recomputation failed: {error}"
        ) from error

    global_failures = _global_class_failures(records)
    post_c8_budget = _post_c8_budget_from_source(authority, manifest, records)
    if global_failures:
        raise RecoveryC8SalvageError(
            "salvage selection blocked by global class-unit validation failure"
        )
    if post_c8_budget.get("status") != base.STATUS_PASS:
        raise RecoveryC8SalvageError(
            "salvage selection blocked by post-C8 frozen budget: "
            + str(post_c8_budget.get("reason"))
        )

    winners: list[dict[str, Any]] = []
    for band in range(4):
        eligible = [
            item for item in candidates
            if item["depth_band"] == band and item["eligible"]
        ]
        ordered = sorted(
            eligible, key=lambda item: tuple(item["rounded_rank_tuple"]),
            reverse=True,
        )
        if not ordered:
            raise RecoveryC8SalvageError(
                f"depth band {band} has no eligible candidate"
            )
        for rank, item in enumerate(ordered, 1):
            item["rank_within_depth_band"] = rank
        ordered[0]["selected_depth_band_winner"] = True
        winners.append(base._candidate_ref(
            ordered[0]["block_index"], ordered[0]["head_index"],
        ))
    by_key = {item["candidate_key"]: item for item in candidates}
    primary_record = max(
        (by_key[item["candidate_key"]] for item in winners),
        key=lambda item: tuple(item["rounded_rank_tuple"]),
    )
    primary_record["selected_primary"] = True
    primary = base._candidate_ref(
        primary_record["block_index"], primary_record["head_index"],
    )

    retained_identity = producer_identities[SOURCE_ORIGIN]
    gap_identity = producer_identities[GAP_ORIGIN]
    verifier_identity = _plain(generator_identity)
    inputs = {
        **_common_inputs(records),
        "calibration_artifact_manifest_sha256": manifest["combined_artifact_manifest_digest"],
        "retained_execution_commit": retained_identity["commit"],
        "retained_execution_identity": retained_identity,
        "gap_execution_commit": gap_identity["commit"],
        "gap_execution_identity": gap_identity,
        "assembler_commit": generator["commit"],
        "assembler_identity": _plain(generator_identity),
        "verifier_commit": generator["commit"],
        "verifier_identity": verifier_identity,
        "selection_rules_version": _RULES_VERSION,
        "salvage_authority_digest": authority.canonical_digest,
        "combined_manifest_digest": manifest["canonical_digest"],
        "source_root_inventory_digest": manifest["source_root_inventory"]["digest"],
        "gap_root_inventory_digest": manifest["gap_root_inventory"]["digest"],
    }
    selection = {
        "schema_version": 1, "document_type": "stage_c_c8_salvage_selection",
        "generated_by": f"vitnns.stage_c_c8_salvage_selection@{generator['commit']}",
        "holdout_accessed": False, "input_bindings": inputs,
        "calibration_sample_ids": list(base.CALIBRATION_SAMPLE_IDS),
        "candidate_records": candidates, "depth_band_winners": winners,
        "primary": primary, "failure_records": failures,
    }
    selection["canonical_digest"] = canonical_sha256(selection)
    selection_file_sha256 = hashlib.sha256(
        _pretty_json_bytes(selection)
    ).hexdigest()
    report = {
        "schema_version": 1, "document_type": "stage_c_c8_salvage_selection_report",
        "status": "PASS", "integrity_valid": True,
        "holdout_accessed": False, "selection_performed": True,
        "bootstrap_performed": False, "holm_performed": False,
        "decision_performed": False,
        "salvage_authority_digest": authority.canonical_digest,
        "combined_manifest_digest": manifest["canonical_digest"],
        "combined_artifact_manifest_digest": manifest["combined_artifact_manifest_digest"],
        "source_root_inventory": _plain(manifest["source_root_inventory"]),
        "gap_root_inventory": _plain(manifest["gap_root_inventory"]),
        "origin_counts": _plain(manifest["origin_counts"]),
        "producer_identities": _plain(manifest["producer_identities"]),
        "unit_origin_execution_identities": _plain(producer_identities),
        "selection_generator_identity": _plain(generator_identity),
        "assembler_identity": _plain(generator_identity),
        "verifier_identity": verifier_identity,
        "selection_rules_version": _RULES_VERSION,
        "input_bindings": inputs,
        "evidence_summary": {
            "expected_unit_count": 2180,
            "observed_unit_count": len(records),
            "succeeded_unit_count": sum(
                record["state"] == "succeeded" for record in records
            ),
            "trusted_unit_count": sum(
                record["trusted_by_batch_validator"] is True
                for record in records
            ),
            "candidate_count": len(candidates),
            "eligible_candidate_count": sum(
                item["eligible"] for item in candidates
            ),
            "candidate_manifest_unit_count": 15,
            "global_class_failure_count": 0,
        },
        "post_c8_budget": post_c8_budget,
        "selection_digest": selection["canonical_digest"],
        "selection_file_sha256": selection_file_sha256,
        "selection": selection, "candidate_audit_records": audits,
    }
    report["canonical_digest"] = canonical_sha256(report)
    return selection, report


def build_c8_salvage_selection(
    authority: RecoveryC8SalvageLaunchRecord, plan: Any, config: Any,
    *, generator_git_state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = _authority(authority)
    expected_generator = _git_state(
        generator_git_state, "selection generator Git state",
    )
    persisted_identity: Any = None
    if os.path.lexists(authority.selection_report_path):
        if (
            authority.selection_report_path.is_symlink()
            or not authority.selection_report_path.is_file()
        ):
            raise RecoveryC8SalvageError(
                "existing selection report is not a safe regular file"
            )
        persisted_identity = _read(
            authority.selection_report_path
        ).get("verifier_identity")
    elif os.path.lexists(authority.selection_path):
        if authority.selection_path.is_symlink() or not authority.selection_path.is_file():
            raise RecoveryC8SalvageError(
                "existing selection document is not a safe regular file"
            )
        persisted_identity = _read(authority.selection_path).get(
            "input_bindings", {}
        ).get("verifier_identity")
    if persisted_identity is not None:
        checked = _generator_identity(authority, persisted_identity)
        observed_generator = _git_state(
            {"commit": checked["commit"], "dirty": checked["dirty"]},
            "persisted selection generator Git state",
        )
        if observed_generator != expected_generator:
            raise RecoveryC8SalvageError(
                "persisted selection generator differs from current source"
            )
        generator_git_state = checked
    manifest = validate_c8_salvage_from_source(authority, plan, config)
    return _derive_c8_salvage_selection(
        authority, manifest, generator_git_state=generator_git_state,
    )


def validate_c8_salvage_selection_report(
    authority: Any, plan: Any, config: Any, report: Mapping[str, Any],
    selection: Mapping[str, Any], *, generator_git_state: Mapping[str, Any],
) -> None:
    if report.get("canonical_digest") != canonical_sha256({k: v for k, v in report.items() if k != "canonical_digest"}):
        raise RecoveryC8SalvageError("salvage selection report digest drifted")
    if selection.get("canonical_digest") != canonical_sha256({k: v for k, v in selection.items() if k != "canonical_digest"}):
        raise RecoveryC8SalvageError("salvage selection digest drifted")
    if report.get("selection") != dict(selection) or report.get("selection_digest") != selection.get("canonical_digest"):
        raise RecoveryC8SalvageError("salvage report/selection binding drifted")
    if report.get("status") != "PASS" or report.get("integrity_valid") is not True:
        raise RecoveryC8SalvageError("salvage selection report is not PASS")
    for flag in ("holdout_accessed", "bootstrap_performed", "holm_performed", "decision_performed"):
        if report.get(flag) is not False:
            raise RecoveryC8SalvageError(f"salvage selection report {flag} must be false")
    if selection.get("holdout_accessed") is not False:
        raise RecoveryC8SalvageError("salvage selection holdout flag must be false")
    if report.get("selection_performed") is not True:
        raise RecoveryC8SalvageError("salvage selection was not performed")
    expected_generator = _git_state(
        generator_git_state, "selection generator Git state",
    )
    generator_identity = _generator_identity(
        authority, report.get("verifier_identity"),
    )
    generator = _git_state(
        {"commit": generator_identity["commit"],
         "dirty": generator_identity["dirty"]},
        "selection generator Git state",
    )
    if generator != expected_generator:
        raise RecoveryC8SalvageError(
            "salvage selection verifier differs from expected generator Git state"
        )
    if selection.get("generated_by") != f"vitnns.stage_c_c8_salvage_selection@{generator['commit']}":
        raise RecoveryC8SalvageError("salvage selection generator drifted")
    candidates = selection.get("candidate_records")
    if not isinstance(candidates, list) or len(candidates) != 144:
        raise RecoveryC8SalvageError("salvage selection requires 144 candidates")
    observed = [(item.get("block_index"), item.get("head_index")) for item in candidates]
    if observed != [(block, head) for block in range(12) for head in range(12)]:
        raise RecoveryC8SalvageError("salvage candidate order drifted")
    winners = selection.get("depth_band_winners")
    primary = selection.get("primary")
    if not isinstance(winners, list) or len(winners) != 4 or primary not in winners:
        raise RecoveryC8SalvageError("salvage winner/primary contract drifted")

    authority = _authority(authority)
    manifest = validate_c8_salvage_from_source(authority, plan, config)
    expected_selection, expected_report = _derive_c8_salvage_selection(
        authority, manifest, generator_git_state=generator_identity,
    )
    bindings = (
        "salvage_authority_digest", "combined_manifest_digest",
        "combined_artifact_manifest_digest", "source_root_inventory",
        "gap_root_inventory", "origin_counts", "producer_identities",
        "unit_origin_execution_identities",
        "selection_generator_identity", "assembler_identity",
        "verifier_identity", "selection_rules_version", "input_bindings",
        "evidence_summary", "post_c8_budget",
    )
    for name in bindings:
        if report.get(name) != expected_report[name]:
            raise RecoveryC8SalvageError(
                f"salvage selection report {name} differs from source recomputation"
            )
    if selection.get("input_bindings") != expected_selection["input_bindings"]:
        raise RecoveryC8SalvageError(
            "salvage selection input bindings differ from source recomputation"
        )
    for context, actual, expected in (
        ("candidate", selection.get("candidate_records"),
         expected_selection["candidate_records"]),
        ("candidate audit", report.get("candidate_audit_records"),
         expected_report["candidate_audit_records"]),
    ):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RecoveryC8SalvageError(
                f"salvage {context} records differ from source recomputation"
            )
        for index, (observed_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            if observed_item != expected_item:
                raise RecoveryC8SalvageError(
                    f"salvage {context} {index} differs from source recomputation"
                )
    for name in ("failure_records", "depth_band_winners", "primary"):
        if selection.get(name) != expected_selection[name]:
            raise RecoveryC8SalvageError(
                f"salvage selection {name} differs from source recomputation"
            )
    if _plain(selection) != expected_selection:
        raise RecoveryC8SalvageError(
            "salvage selection differs from source recomputation"
        )
    if _plain(report) != expected_report:
        raise RecoveryC8SalvageError(
            "salvage selection report differs from source recomputation"
        )


def write_c8_salvage_selection_outputs(
    authority: RecoveryC8SalvageLaunchRecord, selection: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _authority(authority)
    expected_report = _pretty_json_bytes(report)
    expected_selection = _pretty_json_bytes(selection)
    for path, payload, context in (
        (authority.selection_report_path, expected_report, "selection report"),
        (authority.selection_path, expected_selection, "selection document"),
    ):
        if os.path.lexists(path) and (
            path.is_symlink() or not path.is_file()
            or path.read_bytes() != payload
        ):
            raise RecoveryC8SalvageError(
                f"existing {context} differs from source recomputation"
            )
    report_hash = _create_or_accept_exact(
        authority.selection_report_path, report, context="selection report",
    )
    selection_hash = _create_or_accept_exact(
        authority.selection_path, selection, context="selection document",
    )
    return {"status": "PASS", "report_sha256": report_hash, "selection_sha256": selection_hash}


__all__ = ["build_c8_salvage_selection", "validate_c8_salvage_selection_report", "write_c8_salvage_selection_outputs"]
