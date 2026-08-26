"""Closed D6 input derived from one independent D5 source snapshot."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import stage_d_finalize as _d5
from .stage_d_artifacts import CURVES_PATH, RESULT_PATH
from .stage_d_scientific import PRODUCTION_L0_KEYS


_ISSUER = object()


def _owned_mapping(value: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    return _d5._freeze(_d5._plain(value, context))


@dataclass(frozen=True, slots=True, eq=False)
class StageDValidatedAnalysisUnit:
    """One reconciled unit with optionally exposed numeric evidence."""

    selection_order: int
    sample_id: str
    group_digest: str
    pair_digest: str
    unit_digest: str
    method_index: int
    method_id: str
    audit: bool
    terminal_state: str
    failure_code: str | None
    scientific_errors: tuple[str, ...]
    na_reason: str | None
    resources: Mapping[str, Any]
    numeric_payload_status: str
    l0: Mapping[str, Any] | None = field(repr=False)
    curves: np.ndarray | None = field(repr=False)
    l0_sha256: str | None
    curves_sha256: str | None
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _ISSUER:
            raise TypeError(
                "StageDValidatedAnalysisUnit is issued only by the validator"
            )
        object.__setattr__(
            self, "resources", _owned_mapping(self.resources, "unit resources")
        )
        if (
            not isinstance(self.scientific_errors, tuple)
            or any(not isinstance(item, str) or not item for item in self.scientific_errors)
        ):
            raise TypeError("scientific_errors must be a tuple of non-empty strings")
        if self.na_reason is not None and (
            not isinstance(self.na_reason, str) or not self.na_reason
        ):
            raise TypeError("na_reason must be a non-empty string or None")
        if self.l0 is not None:
            object.__setattr__(self, "l0", _owned_mapping(self.l0, "validated L0"))
        if self.curves is not None:
            owned = np.array(self.curves, dtype=np.float64, copy=True, order="C")
            owned.setflags(write=False)
            object.__setattr__(self, "curves", owned)


@dataclass(frozen=True, slots=True, eq=False)
class StageDValidatedAnalysisInput:
    """Unforgeable-in-process receipt for the closed D6 input boundary."""

    root_validation: _d5.StageDRootValidation
    unit_rows: tuple[StageDValidatedAnalysisUnit, ...] = field(repr=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _ISSUER:
            raise TypeError(
                "StageDValidatedAnalysisInput is issued only by the validator"
            )
        if not isinstance(self.root_validation, _d5.StageDRootValidation):
            raise TypeError("root_validation must be a D5 validation result")
        if (
            not isinstance(self.unit_rows, tuple)
            or len(self.unit_rows) != 1000
            or any(not isinstance(row, StageDValidatedAnalysisUnit)
                   for row in self.unit_rows)
        ):
            raise TypeError("unit_rows must contain exactly 1000 issued units")

    @property
    def package_complete(self) -> bool:
        return self.root_validation.package_complete

    @property
    def scientific_validity(self) -> str:
        return self.root_validation.scientific_validity

    @property
    def workflow_status(self) -> str:
        return self.root_validation.workflow_status

    @property
    def plan_digest(self) -> str:
        return self.root_validation.plan_digest

    @property
    def final_status_sha256(self) -> str:
        return self.root_validation.final_status_sha256

    @property
    def root_inventory(self) -> Mapping[str, Any]:
        return self.root_validation.root_inventory

    def to_dict(self) -> dict[str, Any]:
        """Return control data only; never serialize numeric unit payloads."""
        return {
            "package_complete": self.package_complete,
            "scientific_validity": self.scientific_validity,
            "workflow_status": self.workflow_status,
            "plan_digest": self.plan_digest,
            "final_status_sha256": self.final_status_sha256,
            "root_inventory": _d5._plain(self.root_inventory),
            "unit_count": len(self.unit_rows),
        }


def _artifact_sha(attempt: Mapping[str, Any], path: str) -> str:
    matches = [
        item.get("sha256") for item in attempt["artifacts"]
        if isinstance(item, Mapping) and item.get("path") == path
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise _d5.StageDFinalizationIntegrityError(
            f"validated attempt lacks exactly one {path} binding"
        )
    return matches[0]


def _analysis_rows(
    documents: Mapping[str, Any], captured_bytes: Mapping[str, bytes],
) -> tuple[StageDValidatedAnalysisUnit, ...]:
    validity = documents["evidence_report"]["scientific_validity"]
    expose_numeric = validity == "PASS"
    issued: list[StageDValidatedAnalysisUnit] = []
    for source_row in documents["reconciliation"]["rows"]:
        attempt = source_row["attempts"][-1]
        l0 = curves = None
        l0_sha256 = curves_sha256 = None
        numeric_status = "WITHHELD_DUE_TO_SCIENTIFIC_INVALIDITY"
        if expose_numeric:
            if (
                source_row["terminal_state"] != "succeeded"
                or source_row["failure_code"] is not None
                or source_row["scientific_errors"]
            ):
                raise _d5.StageDFinalizationIntegrityError(
                    "valid analysis input contains a non-successful unit"
                )
            base = attempt["attempt_path"]
            l0_key, curves_key = f"{base}/{RESULT_PATH}", f"{base}/{CURVES_PATH}"
            try:
                l0_raw, curves_raw = captured_bytes[l0_key], captured_bytes[curves_key]
            except KeyError as error:
                raise _d5.StageDFinalizationIntegrityError(
                    "validated analysis payload is absent from the D5 snapshot"
                ) from error
            parsed_l0 = _d5._strict_json(l0_raw, "validated analysis L0")
            if set(parsed_l0) != PRODUCTION_L0_KEYS:
                # D5 deliberately accepts its smaller synthetic L0 envelope.
                # Such a sealed root is a valid D5 fixture but cannot create a
                # D6 numerical coordinate; retain it as explicit invalidity
                # instead of silently turning it into a complete-case input.
                issued.append(StageDValidatedAnalysisUnit(
                    selection_order=source_row["selection_order"],
                    sample_id=source_row["sample_id"],
                    group_digest=source_row["group_digest"],
                    pair_digest=source_row["pair_digest"],
                    unit_digest=source_row["unit_digest"],
                    method_index=source_row["method_index"],
                    method_id=source_row["method_id"], audit=source_row["audit"],
                    terminal_state=source_row["terminal_state"],
                    failure_code=source_row["failure_code"],
                    scientific_errors=tuple(source_row["scientific_errors"]),
                    na_reason=source_row["failure_code"], resources=source_row["resources"],
                    numeric_payload_status="ANALYSIS_INCOMPLETE_SYNTHETIC_L0",
                    l0=None, curves=None, l0_sha256=None, curves_sha256=None,
                    _issuer=_ISSUER,
                ))
                continue
            archive = _d5._load_npz(curves_raw, "validated analysis curves")
            parsed_curves = archive.get("curves") if set(archive) == {"curves"} else None
            if (
                not isinstance(parsed_curves, np.ndarray)
                or parsed_curves.dtype != np.float64
                or parsed_curves.shape != (2, 5, 21)
                or not np.isfinite(parsed_curves).all()
            ):
                raise _d5.StageDFinalizationIntegrityError(
                    "validated analysis curves violate the frozen L1 contract"
                )
            l0_sha256 = hashlib.sha256(l0_raw).hexdigest()
            curves_sha256 = hashlib.sha256(curves_raw).hexdigest()
            if (
                l0_sha256 != _artifact_sha(attempt, RESULT_PATH)
                or curves_sha256 != _artifact_sha(attempt, CURVES_PATH)
            ):
                raise _d5.StageDFinalizationIntegrityError(
                    "analysis payload hashes differ from validated artifacts"
                )
            l0, curves, numeric_status = parsed_l0, parsed_curves, "VALIDATED"

        issued.append(StageDValidatedAnalysisUnit(
            selection_order=source_row["selection_order"],
            sample_id=source_row["sample_id"],
            group_digest=source_row["group_digest"],
            pair_digest=source_row["pair_digest"],
            unit_digest=source_row["unit_digest"],
            method_index=source_row["method_index"],
            method_id=source_row["method_id"],
            audit=source_row["audit"],
            terminal_state=source_row["terminal_state"],
            failure_code=source_row["failure_code"],
            scientific_errors=tuple(source_row["scientific_errors"]),
            na_reason=source_row["failure_code"],
            resources=source_row["resources"],
            numeric_payload_status=numeric_status,
            l0=l0, curves=curves, l0_sha256=l0_sha256,
            curves_sha256=curves_sha256, _issuer=_ISSUER,
        ))
    if len(issued) != 1000:
        raise _d5.StageDFinalizationIntegrityError(
            "validated analysis receipt does not contain 1000 units"
        )
    return tuple(issued)


def validate_stage_d_analysis_input(
    output_root: Path | str, plan: Any, estimate: Any, preflight: Any,
) -> StageDValidatedAnalysisInput:
    """Validate a sealed synthetic root and issue same-snapshot D6 inputs."""

    typed_plan, typed_estimate, typed_preflight = _d5._validate_controls(
        plan, estimate, preflight
    )
    named_root = _d5._safe_root(output_root)
    with _d5._retained_root(named_root) as (
        root, root_descriptor, root_identity,
    ), _d5._root_locks(root):
        _d5._verify_root_binding(named_root, root_descriptor, root_identity)
        prechecked = _d5._checkpoint_control(
            root, typed_plan, typed_estimate, typed_preflight
        )
        _d5._reject_active_checkpoint_control(prechecked)
        (
            source_records, captured_whole, captured_bytes,
            captured_directories,
        ) = _d5._source_snapshot(
            root, sealed=True, root_descriptor=root_descriptor
        )
        prechecked = _d5._checkpoint_control(
            root, typed_plan, typed_estimate, typed_preflight,
            raw_bytes=captured_bytes[_d5._runtime.RUNTIME_CHECKPOINT_FILENAME],
        )
        _d5._validate_source_directories(
            source_records, captured_directories, typed_plan, prechecked
        )
        snapshot_owner, snapshot_root = _d5._materialize_source_snapshot(
            source_records, captured_bytes, captured_directories
        )
        try:
            expected, documents = _d5._build_source_documents(
                snapshot_root, typed_plan, typed_estimate, typed_preflight,
                allow_formal=False, sealed=False, source_records=source_records,
                prechecked_checkpoint=prechecked, captured_bytes=captured_bytes,
                logical_root=named_root,
            )
            unit_rows = _analysis_rows(documents, captured_bytes)
        finally:
            snapshot_owner.cleanup()

        for name in _d5.FINAL_FILES:
            relative = f"{_d5.FINAL_DIRECTORY_NAME}/{name}"
            try:
                actual = captured_bytes[relative]
            except KeyError as error:
                raise _d5.StageDFinalizationIntegrityError(
                    f"final control is absent from captured root: {name}"
                ) from error
            parsed = _d5._strict_json(actual, f"final control {name}")
            _d5._validate_final_document(name, parsed)
            if actual != expected[name]:
                raise _d5.StageDFinalizationIntegrityError(
                    f"final control differs from source recomputation: {name}"
                )

        _d5._verify_root_binding(named_root, root_descriptor, root_identity)
        _d5._root_layout(root, typed_plan, sealed=True)
        current_whole, current_directories = _d5._inventory_and_directories(
            root, exclude_final=False, root_descriptor=root_descriptor
        )
        if (
            current_whole != captured_whole
            or current_directories != captured_directories
        ):
            raise _d5.StageDFinalizationIntegrityError(
                "sealed root changed during independent analysis validation"
            )
        status_raw = captured_bytes[
            f"{_d5.FINAL_DIRECTORY_NAME}/{_d5.FINAL_STATUS_FILENAME}"
        ]
        root_validation = _d5._result(documents, captured_whole, status_raw)
        _d5._validate_root_validation_document(root_validation.to_dict())
        if (
            root_validation.package_complete is not True
            or root_validation.workflow_status != "PASS"
        ):
            raise _d5.StageDFinalizationIntegrityError(
                "D6 input requires a complete PASS-workflow D5 package"
            )
        if root_validation.evidence_report.get("analysis_boundary") != {
            "status": "NOT_PERFORMED", "analysis_performed": False,
            "bootstrap_performed": False, "decision_performed": False,
            "delta_fid_computed": False, "aggregate_effects_computed": False,
            "d6_eligibility": "SYNTHETIC_ONLY",
        }:
            raise _d5.StageDFinalizationIntegrityError(
                "D6 input is not a synthetic-only D5 analysis boundary"
            )
        return StageDValidatedAnalysisInput(
            root_validation=root_validation, unit_rows=unit_rows, _issuer=_ISSUER
        )


__all__ = [
    "StageDValidatedAnalysisInput", "StageDValidatedAnalysisUnit",
    "validate_stage_d_analysis_input",
]
