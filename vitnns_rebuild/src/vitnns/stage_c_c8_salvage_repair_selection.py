"""Selection over a source-recomputed, three-origin C8 repair union."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .recovery_c8_salvage_repair import (
    ORIGINS as _AUTHORITY_ORIGINS,
    RecoveryC8SalvageRepairError,
    capture_c8_salvage_repair_verifier_identity,
)
from .stage_c_provenance import canonical_sha256
from . import stage_c_selection as base


ORIGINAL_SOURCE_ORIGIN = _AUTHORITY_ORIGINS["original_interrupted"]
PRIOR_GAP_ORIGIN = _AUTHORITY_ORIGINS["prior_partial_salvage"]
REPAIR_ORIGIN = _AUTHORITY_ORIGINS["fresh_repair"]
ORIGINS = (
    ORIGINAL_SOURCE_ORIGIN,
    PRIOR_GAP_ORIGIN,
    REPAIR_ORIGIN,
)

SELECTION_DOCUMENT_TYPE = "stage_c_c8_salvage_repair_selection"
REPORT_DOCUMENT_TYPE = "stage_c_c8_salvage_repair_selection_report"
RULES_VERSION = "stage-c-c8-salvage-repair-selection-v1"
EXPECTED_UNIT_COUNT = 2180
EXPECTED_ORIGIN_COUNTS = {
    ORIGINAL_SOURCE_ORIGIN: 1460,
    PRIOR_GAP_ORIGIN: 569,
    REPAIR_ORIGIN: 151,
}


def validate_c8_salvage_repair_from_source(
    authority: Any, plan: Any, config: Any,
) -> Mapping[str, Any]:
    """Lazy bridge to the independent repair runtime."""

    from .stage_c_c8_salvage_repair import (
        validate_c8_salvage_repair_from_source as validate,
    )

    return validate(authority, plan, config)


def _plain(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    raise RecoveryC8SalvageRepairError(
        f"repair selection value is not JSON-compatible: {type(value).__name__}"
    )


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        raise RecoveryC8SalvageRepairError(f"{context} is not lowercase SHA-256")
    return value


def _git_object(value: Any, context: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 40
        or set(value) - set("0123456789abcdef")
    ):
        raise RecoveryC8SalvageRepairError(
            f"{context} is not a lowercase Git object"
        )
    return value


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _plain(value), indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _self_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: item for key, item in value.items()
        if key != "canonical_digest"
    })


def _identity(value: Any, context: str, *, require_tree: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryC8SalvageRepairError(f"{context} is not an object")
    result = _plain(value)
    _git_object(result.get("commit"), f"{context}.commit")
    if result.get("dirty") is not False:
        raise RecoveryC8SalvageRepairError(f"{context} must bind clean source")
    if require_tree:
        _git_object(result.get("tree"), f"{context}.tree")
        _git_object(
            result.get("source_subtree_tree"),
            f"{context}.source_subtree_tree",
        )
        if result.get("source_subtree_path") != "src/vitnns":
            raise RecoveryC8SalvageRepairError(
                f"{context} source subtree path drifted"
            )
    return result


def _generator_state(value: Any) -> dict[str, Any]:
    identity = _identity(value, "repair selection generator", require_tree=False)
    return {"commit": identity["commit"], "dirty": False}


def _stable_runtime_identity(value: Any, context: str) -> dict[str, Any]:
    identity = _identity(value, context, require_tree=True)
    launch = _git_object(
        identity.get("authorization_launch_commit"),
        f"{context}.authorization_launch_commit",
    )
    return {
        "commit": identity["commit"],
        "tree": identity["tree"],
        "source_subtree_path": identity["source_subtree_path"],
        "source_subtree_tree": identity["source_subtree_tree"],
        "dirty": False,
        "authorization_launch_commit": launch,
    }


def _capture_assembler_identity(
    authority: Any, expected_git_state: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _generator_state(expected_git_state)
    captured = capture_c8_salvage_repair_verifier_identity(
        authority, "selection_assembler",
    )
    identity = _stable_runtime_identity(
        captured, "captured repair selection assembler",
    )
    if _generator_state(identity) != expected:
        raise RecoveryC8SalvageRepairError(
            "captured repair selection assembler differs from expected Git state"
        )
    return identity


def _authority_digest(authority: Any) -> str:
    return _sha256(
        getattr(authority, "canonical_digest", None),
        "repair authority canonical digest",
    )


def _prior_authority_digest(authority: Any) -> str:
    value = getattr(authority, "prior_authority_digest", None)
    if value is None:
        try:
            value = authority.document["original_salvage_authority"][
                "canonical_digest"
            ]
        except (KeyError, TypeError, AttributeError) as error:
            raise RecoveryC8SalvageRepairError(
                "repair authority lacks the prior salvage authority digest"
            ) from error
    return _sha256(value, "prior salvage authority digest")


def _root_inventories(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("root_inventories")
    if not isinstance(raw, Mapping) or set(raw) != set(ORIGINS):
        raise RecoveryC8SalvageRepairError(
            "repair manifest must bind exactly three root inventories"
        )
    result: dict[str, Any] = {}
    for origin in ORIGINS:
        if not isinstance(raw[origin], Mapping):
            raise RecoveryC8SalvageRepairError(
                f"repair root inventory is invalid: {origin}"
            )
        inventory = _plain(raw[origin])
        _sha256(inventory.get("digest"), f"root inventory {origin}.digest")
        if (
            type(inventory.get("total_bytes")) is not int
            or inventory["total_bytes"] < 0
        ):
            raise RecoveryC8SalvageRepairError(
                f"root inventory {origin}.total_bytes is invalid"
            )
        result[origin] = inventory
    return result


def _origin_bindings(
    manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    raw = manifest.get("producer_identities")
    if not isinstance(raw, Mapping) or set(raw) != set(ORIGINS):
        raise RecoveryC8SalvageRepairError(
            "repair manifest must bind exactly three producer identities"
        )
    identities = {
        origin: _identity(
            raw[origin], f"repair producer identity {origin}",
            require_tree=True,
        )
        for origin in ORIGINS
    }
    if len({item["commit"] for item in identities.values()}) != 3:
        raise RecoveryC8SalvageRepairError(
            "repair origins must retain three distinct execution commits"
        )
    counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} is not an object"
            )
        origin = record.get("origin")
        if origin not in identities:
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} origin is invalid"
            )
        observed = _identity(
            record.get("execution_git_identity"),
            f"repair combined record {index} execution identity",
            require_tree=True,
        )
        if observed != identities[origin]:
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} identity differs from its origin"
            )
        try:
            code = record["provenance"]["bindings"]["code"]
        except (KeyError, TypeError) as error:
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} lacks execution provenance"
            ) from error
        if not isinstance(code, Mapping) or dict(code) != {
            "commit": observed["commit"], "dirty": False,
        }:
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} provenance differs from its origin"
            )
        counts[origin] += 1
    actual_counts = {origin: counts[origin] for origin in ORIGINS}
    expected_counts = manifest.get("origin_counts")
    if (
        not isinstance(expected_counts, Mapping)
        or dict(expected_counts) != actual_counts
        or actual_counts != EXPECTED_ORIGIN_COUNTS
    ):
        raise RecoveryC8SalvageRepairError("repair manifest origin counts drifted")
    return identities, actual_counts


def _adapt_records(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != EXPECTED_UNIT_COUNT:
        raise RecoveryC8SalvageRepairError(
            "repair combined manifest must contain exactly 2,180 records"
        )
    identities, origin_counts = _origin_bindings(manifest, raw_records)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} is not an object"
            )
        state = raw.get("state")
        if state not in {"succeeded", "failed"}:
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} is not terminal"
            )
        if raw.get("validation_errors") != []:
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} failed integrity validation"
            )
        try:
            if raw["plan_index"] != index:
                raise RecoveryC8SalvageRepairError(
                    f"repair combined record {index} plan index drifted"
                )
            unit_key = _plain(raw["unit_key"])
            unit_digest = _sha256(
                unit_key["canonical_digest"],
                f"repair combined record {index} unit digest",
            )
            if unit_digest in seen:
                raise RecoveryC8SalvageRepairError(
                    f"repair combined record {index} duplicates a unit"
                )
            seen.add(unit_digest)
            hashes = raw["hashes"]
            if not isinstance(hashes, Mapping):
                raise TypeError("hashes")
            status_sha256 = hashes["status_sha256"]
            record = {
                "plan_index": index,
                "unit_key": unit_key,
                "state": state,
                # Source validation has authenticated both successes and valid
                # terminal scientific failures.  Keeping state=failed lets the
                # frozen candidate logic make only that head ineligible.
                "trusted_by_batch_validator": True,
                "hashes": {
                    "manifest_sha256": hashes.get("manifest_sha256"),
                    "status_sha256": status_sha256,
                    "selected_status_sha256": status_sha256,
                    "provenance_sha256": hashes.get("provenance_sha256"),
                    "result_sha256": hashes.get("result_sha256"),
                    "unit_identity_sha256": hashes.get("unit_identity_sha256"),
                },
                "scalars": _plain(raw["scalars"]),
                "provenance": _plain(raw["provenance"]),
                "resources": _plain(raw["resources"]),
                "runner_telemetry": _plain(raw["runner_telemetry"]),
                "artifact_inventory": _plain(raw["artifact_inventory"]),
                "work_item_counts": _plain(raw["work_item_counts"]),
                "audit": raw["audit"],
                "validation_errors": [],
                "attempt_path": raw["attempt_path"],
                "l0_l1_cap_accounting_bytes": raw[
                    "l0_l1_cap_accounting_bytes"
                ],
                "level_bytes": _plain(raw["level_bytes"]),
                "total_regular_file_bytes": raw["total_regular_file_bytes"],
            }
        except (KeyError, TypeError) as error:
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} lacks selection evidence"
            ) from error
        if type(record["audit"]) is not bool:
            raise RecoveryC8SalvageRepairError(
                f"repair combined record {index} audit flag is invalid"
            )
        records.append(record)
    return records, identities, origin_counts


def _common_inputs(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    result: dict[str, Any] = {}
    for name, path in fields.items():
        try:
            values = {
                record["provenance"]["bindings"][path[0]][path[1]]
                for record in records
            }
        except (KeyError, TypeError) as error:
            raise RecoveryC8SalvageRepairError(
                f"repair records lack common input binding: {name}"
            ) from error
        if len(values) != 1:
            raise RecoveryC8SalvageRepairError(
                f"repair records contain mixed input binding: {name}"
            )
        result[name] = values.pop()
    return result


def _budget(
    manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    inventories: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw = manifest.get("budget_evidence")
    if not isinstance(raw, Mapping):
        raise RecoveryC8SalvageRepairError(
            "repair manifest lacks frozen budget evidence"
        )
    evidence = _plain(raw)
    bindings = evidence.get("bindings")
    if not isinstance(bindings, dict):
        raise RecoveryC8SalvageRepairError("repair budget bindings are invalid")
    expected_bytes = sum(
        inventory["total_bytes"] for inventory in inventories.values()
    )
    if bindings.get("source_tree_total_bytes") != expected_bytes:
        raise RecoveryC8SalvageRepairError(
            "repair budget source bytes differ from three root inventories"
        )
    if not isinstance(bindings.get("control_file_sha256"), Mapping):
        raise RecoveryC8SalvageRepairError(
            "repair budget control-file binding is invalid"
        )
    c7_binding = evidence.get("c7_report_binding")
    if c7_binding is None:
        c7_binding = bindings.get("c7_report_binding")
    if not isinstance(c7_binding, Mapping):
        raise RecoveryC8SalvageRepairError(
            "repair budget lacks the frozen C7 report binding"
        )
    evidence["c7_report_binding"] = _plain(c7_binding)
    evidence["records"] = list(records)
    try:
        return base._post_c8_budget(evidence)
    except Exception as error:
        raise RecoveryC8SalvageRepairError(
            f"repair post-C8 budget recomputation failed: {error}"
        ) from error


def _global_class_failures(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    count = 0
    for record in records:
        key = record["unit_key"]
        method_id = key["method_id"]
        if method_id not in base.CLASS_METHOD_IDS:
            continue
        count += 1
        sample_id = key["sample_id"]
        try:
            errors = base._record_validity(
                record, f"{sample_id}:{method_id}",
            )
        except Exception as error:
            raise RecoveryC8SalvageRepairError(
                f"repair class-unit validation failed: {error}"
            ) from error
        failures.extend({
            "candidate_key": None,
            "sample_id": sample_id,
            "reason": f"global_unit:{method_id}:{reason}",
            "unit_key": _plain(key),
        } for reason in errors)
    expected = len(base.CALIBRATION_SAMPLE_IDS) * len(base.CLASS_METHOD_IDS)
    if count != expected:
        raise RecoveryC8SalvageRepairError(
            f"repair records require exactly {expected} global class units"
        )
    return failures


def _derive(
    authority: Any, manifest: Mapping[str, Any], *,
    generator_git_state: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if manifest.get("status") not in {None, "PASS"}:
        raise RecoveryC8SalvageRepairError("repair combined manifest is not PASS")
    manifest_digest = _sha256(
        manifest.get("canonical_digest"), "repair combined manifest digest",
    )
    artifact_digest = _sha256(
        manifest.get("combined_artifact_manifest_digest"),
        "repair combined artifact manifest digest",
    )
    plan_digest = _sha256(
        manifest.get("plan_digest"), "repair combined plan digest",
    )
    authority_digest = _authority_digest(authority)
    prior_digest = _prior_authority_digest(authority)
    generator_identity = _identity(
        generator_git_state, "repair selection assembler", require_tree=False,
    )
    generator = _generator_state(generator_identity)
    verifier = _plain(manifest.get("verifier"))
    _stable_runtime_identity(verifier, "repair combined verifier")
    inventories = _root_inventories(manifest)
    records, execution_identities, origin_counts = _adapt_records(manifest)

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
        raise RecoveryC8SalvageRepairError(
            f"repair candidate recomputation failed: {error}"
        ) from error

    global_failures = _global_class_failures(records)
    failures.extend(global_failures)
    post_c8_budget = _budget(manifest, records, inventories)
    blocked_reasons: list[str] = []
    winners: list[dict[str, Any]] = []
    for band in range(4):
        eligible = [
            item for item in candidates
            if item["depth_band"] == band and item["eligible"]
        ]
        ordered = sorted(
            eligible,
            key=lambda item: tuple(item["rounded_rank_tuple"]),
            reverse=True,
        )
        for rank, item in enumerate(ordered, 1):
            item["rank_within_depth_band"] = rank
        if not ordered:
            blocked_reasons.append(
                f"depth_band_{band}_has_no_eligible_candidate"
            )
            continue
        ordered[0]["selected_depth_band_winner"] = True
        winners.append(base._candidate_ref(
            ordered[0]["block_index"], ordered[0]["head_index"],
        ))
    if global_failures:
        blocked_reasons.append("global_class_unit_failure")
    if post_c8_budget.get("status") != base.STATUS_PASS:
        blocked_reasons.append(str(
            post_c8_budget.get("reason") or "post_c8_frozen_cap_failure"
        ))

    primary: dict[str, Any] | None = None
    if len(winners) == 4:
        by_key = {item["candidate_key"]: item for item in candidates}
        primary_record = max(
            (by_key[item["candidate_key"]] for item in winners),
            key=lambda item: tuple(item["rounded_rank_tuple"]),
        )
        primary_record["selected_primary"] = True
        primary = base._candidate_ref(
            primary_record["block_index"], primary_record["head_index"],
        )

    status = base.STATUS_PASS if not blocked_reasons else base.STATUS_BLOCKED
    if status == base.STATUS_BLOCKED:
        for candidate in candidates:
            candidate["selected_depth_band_winner"] = False
            candidate["selected_primary"] = False

    root_digests = {
        origin: inventories[origin]["digest"] for origin in ORIGINS
    }
    source_bindings = {
        "repair_authority_digest": authority_digest,
        "prior_salvage_authority_digest": prior_digest,
        "plan_digest": plan_digest,
        "combined_manifest_digest": manifest_digest,
        "combined_artifact_manifest_digest": artifact_digest,
        "origin_counts": origin_counts,
        "origin_execution_identities": execution_identities,
        "root_inventory_digests": root_digests,
        "combined_verifier_identity": verifier,
        "assembler_identity": generator_identity,
    }
    input_bindings = {**_common_inputs(records), **source_bindings}
    selection: dict[str, Any] | None = None
    if status == base.STATUS_PASS:
        if primary is None:
            raise RecoveryC8SalvageRepairError(
                "repair PASS selection has no unique primary"
            )
        selection = {
            "schema_version": 1,
            "document_type": SELECTION_DOCUMENT_TYPE,
            "generated_by": (
                "vitnns.stage_c_c8_salvage_repair_selection@"
                + generator["commit"]
            ),
            "holdout_accessed": False,
            "input_bindings": input_bindings,
            "calibration_sample_ids": list(base.CALIBRATION_SAMPLE_IDS),
            "candidate_records": candidates,
            "depth_band_winners": winners,
            "primary": primary,
            "failure_records": [
                item for item in failures
                if item.get("candidate_key") is not None
            ],
        }
        selection["canonical_digest"] = canonical_sha256(selection)

    succeeded = sum(record["state"] == "succeeded" for record in records)
    terminal_failed = sum(record["state"] == "failed" for record in records)
    report: dict[str, Any] = {
        "schema_version": 1,
        "document_type": REPORT_DOCUMENT_TYPE,
        "status": status,
        "integrity_valid": True,
        "blocked_reasons": list(dict.fromkeys(blocked_reasons)),
        "holdout_accessed": False,
        "selection_performed": status == base.STATUS_PASS,
        "bootstrap_performed": False,
        "holm_performed": False,
        "decision_performed": False,
        "repair_authority_digest": authority_digest,
        "prior_salvage_authority_digest": prior_digest,
        "combined_manifest_digest": manifest_digest,
        "combined_artifact_manifest_digest": artifact_digest,
        "plan_digest": plan_digest,
        "origin_counts": origin_counts,
        "producer_identities": _plain(manifest["producer_identities"]),
        "root_inventories": inventories,
        "combined_verifier_identity": verifier,
        "assembler_identity": generator_identity,
        "selection_rules_version": RULES_VERSION,
        "input_bindings": input_bindings,
        "evidence_summary": {
            "expected_unit_count": EXPECTED_UNIT_COUNT,
            "observed_unit_count": len(records),
            "succeeded_unit_count": succeeded,
            "terminal_failed_unit_count": terminal_failed,
            "trusted_terminal_unit_count": len(records),
            "candidate_count": len(candidates),
            "eligible_candidate_count": sum(
                candidate["eligible"] for candidate in candidates
            ),
            "candidate_manifest_unit_count": 15,
            "global_class_failure_count": len(global_failures),
        },
        "post_c8_budget": post_c8_budget,
        "candidate_records": candidates,
        "candidate_audit_records": audits,
        "depth_band_winners": winners if status == base.STATUS_PASS else [],
        "primary": primary if status == base.STATUS_PASS else None,
        "failure_records": failures,
        "selection": selection,
        "selection_digest": (
            None if selection is None else selection["canonical_digest"]
        ),
        "selection_file_sha256": (
            None if selection is None
            else hashlib.sha256(_pretty_json_bytes(selection)).hexdigest()
        ),
    }
    report["canonical_digest"] = canonical_sha256(report)
    return selection, report


def build_c8_salvage_repair_selection(
    authority: Any, plan: Any, config: Any, *,
    generator_git_state: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    assembler = _capture_assembler_identity(authority, generator_git_state)
    manifest = validate_c8_salvage_repair_from_source(authority, plan, config)
    if not isinstance(manifest, Mapping):
        raise RecoveryC8SalvageRepairError(
            "repair source validator did not return a manifest"
        )
    return _derive(
        authority, manifest, generator_git_state=assembler,
    )


def validate_c8_salvage_repair_selection_report(
    authority: Any, plan: Any, config: Any, report: Mapping[str, Any],
    selection: Mapping[str, Any] | None, *,
    generator_git_state: Mapping[str, Any],
) -> None:
    if not isinstance(report, Mapping):
        raise RecoveryC8SalvageRepairError("repair selection report must be an object")
    if report.get("canonical_digest") != _self_digest(report):
        raise RecoveryC8SalvageRepairError("repair selection report digest drifted")
    if report.get("document_type") != REPORT_DOCUMENT_TYPE:
        raise RecoveryC8SalvageRepairError("repair selection report type drifted")
    if report.get("integrity_valid") is not True:
        raise RecoveryC8SalvageRepairError("repair selection report is not integrity-valid")
    for flag in (
        "holdout_accessed", "bootstrap_performed", "holm_performed",
        "decision_performed",
    ):
        if report.get(flag) is not False:
            raise RecoveryC8SalvageRepairError(
                f"repair selection report {flag} must be false"
            )
    current_assembler = _capture_assembler_identity(
        authority, generator_git_state,
    )
    persisted_assembler = _stable_runtime_identity(
        report.get("assembler_identity"),
        "persisted repair selection assembler",
    )
    if persisted_assembler != current_assembler:
        raise RecoveryC8SalvageRepairError(
            "repair selection assembler differs from expected Git state"
        )

    manifest = validate_c8_salvage_repair_from_source(authority, plan, config)
    if not isinstance(manifest, Mapping):
        raise RecoveryC8SalvageRepairError(
            "repair source validator did not return a manifest"
        )
    expected_selection, expected_report = _derive(
        authority, manifest,
        generator_git_state=persisted_assembler,
    )
    if _plain(report) != expected_report:
        raise RecoveryC8SalvageRepairError(
            "repair selection report differs from source recomputation"
        )
    if expected_selection is None:
        if selection is not None:
            raise RecoveryC8SalvageRepairError(
                "BLOCKED repair report must not have a selection"
            )
        return
    if not isinstance(selection, Mapping):
        raise RecoveryC8SalvageRepairError("PASS repair report requires a selection")
    if selection.get("canonical_digest") != _self_digest(selection):
        raise RecoveryC8SalvageRepairError("repair selection digest drifted")
    if _plain(selection) != expected_selection:
        raise RecoveryC8SalvageRepairError(
            "repair selection differs from source recomputation"
        )


def _path(authority: Any, name: str) -> Path:
    value = getattr(authority, name, None)
    if not isinstance(value, (str, os.PathLike)):
        raise RecoveryC8SalvageRepairError(f"repair authority lacks {name}")
    return Path(value).absolute()


def _validate_existing(path: Path, payload: bytes, context: str) -> str | None:
    if not os.path.lexists(path):
        return None
    if path.is_symlink() or not path.is_file():
        raise RecoveryC8SalvageRepairError(
            f"existing repair {context} is not a safe regular file"
        )
    observed = path.read_bytes()
    if observed != payload:
        raise RecoveryC8SalvageRepairError(
            f"existing repair {context} differs from source recomputation"
        )
    return hashlib.sha256(observed).hexdigest()


def _create_or_accept(path: Path, value: Mapping[str, Any], context: str) -> str:
    payload = _pretty_json_bytes(value)
    existing = _validate_existing(path, payload, context)
    if existing is not None:
        return existing
    if path.parent.exists() and path.parent.is_symlink():
        raise RecoveryC8SalvageRepairError(
            f"repair {context} parent cannot be a symlink"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raced = _validate_existing(path, payload, context)
        if raced is None:
            raise
        return raced
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def write_c8_salvage_repair_selection_outputs(
    authority: Any, plan: Any, config: Any,
    selection: Mapping[str, Any] | None, report: Mapping[str, Any], *,
    generator_git_state: Mapping[str, Any],
) -> dict[str, Any]:
    validate_c8_salvage_repair_selection_report(
        authority, plan, config, report, selection,
        generator_git_state=generator_git_state,
    )
    report_path = _path(authority, "selection_report_path")
    selection_path = _path(authority, "selection_path")
    if report_path.resolve(strict=False) == selection_path.resolve(strict=False):
        raise RecoveryC8SalvageRepairError(
            "repair selection and report paths must be distinct"
        )
    report_payload = _pretty_json_bytes(report)
    selection_payload = (
        None if selection is None else _pretty_json_bytes(selection)
    )
    _validate_existing(report_path, report_payload, "selection report")
    if selection_payload is None:
        if os.path.lexists(selection_path):
            raise RecoveryC8SalvageRepairError(
                "BLOCKED repair selection must not have a selection file"
            )
    else:
        _validate_existing(selection_path, selection_payload, "selection document")

    report_hash = _create_or_accept(report_path, report, "selection report")
    if selection is None:
        return {
            "status": base.STATUS_BLOCKED,
            "report_sha256": report_hash,
            "selection_sha256": None,
        }
    selection_hash = _create_or_accept(
        selection_path, selection, "selection document",
    )
    return {
        "status": base.STATUS_PASS,
        "report_sha256": report_hash,
        "selection_sha256": selection_hash,
    }


__all__ = [
    "ORIGINAL_SOURCE_ORIGIN", "PRIOR_GAP_ORIGIN", "REPAIR_ORIGIN",
    "build_c8_salvage_repair_selection",
    "validate_c8_salvage_repair_selection_report",
    "write_c8_salvage_repair_selection_outputs",
]
