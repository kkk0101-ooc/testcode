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
from .stage_d_config import strict_json_bytes
from .stage_d_d9_completion import (
    C10_STATUS_PRESERVED,
    REPORT_PATH as D9_REPORT_PATH,
)
from .stage_d_provenance import canonical_sha256
from .stage_d_scientific import PRODUCTION_L0_KEYS


_ISSUER = object()
_TRACELESS_PRODUCTION_L0_KEYS = (
    PRODUCTION_L0_KEYS - frozenset({"provenance_fragment", "map_execution_records"})
) | {"trace_mode"}
_FULL_NNS_PRODUCTION_L0_KEYS = (
    PRODUCTION_L0_KEYS | frozenset({"trace_mode", "nns_contract", "trace"})
)
_D9_REPORT_SHA256 = (
    "bbfb6b5ad879116f6849e6f0a058780fdfd5e54b656e62f0674a3c3184c810d5"
)
_D9_REPORT_CANONICAL_DIGEST = (
    "50e2c1455534e5b10aeacb79fac42d128317e0e943c817addf47671f599ea206"
)
_D9_FINAL_STATUS_SHA256 = (
    "9c2a7f6bbe89f393b17d39128c908b764c5481cb2ab640eea557ea87032e9f4c"
)
_D9_FINAL_STATUS_CANONICAL_DIGEST = (
    "ba5c15aec4d0f8f374e376e7b1ddb28ebbdb9d8ec8ab7ceda1d34438caf9425b"
)
_D9_ROOT_INVENTORY_DIGEST = (
    "76d7d532049a5c90a09c74ba56b019042d7f2f0174ccb439bd8f67713514cc5a"
)


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
    formal: bool = False
    d9_report: Mapping[str, Any] | None = None
    d9_report_sha256: str | None = None
    d9_report_canonical_digest: str | None = None
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
        if type(self.formal) is not bool:
            raise TypeError("formal must be a boolean")
        if self.formal:
            if (
                not isinstance(self.d9_report, Mapping)
                or self.d9_report_sha256 != _D9_REPORT_SHA256
                or self.d9_report_canonical_digest != _D9_REPORT_CANONICAL_DIGEST
            ):
                raise TypeError("formal analysis input requires the frozen D9 report")
            object.__setattr__(
                self, "d9_report", _owned_mapping(self.d9_report, "D9 report")
            )
        elif (
            self.d9_report is not None
            or self.d9_report_sha256 is not None
            or self.d9_report_canonical_digest is not None
        ):
            raise TypeError("synthetic analysis input cannot carry a D9 report")

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
        document = {
            "package_complete": self.package_complete,
            "scientific_validity": self.scientific_validity,
            "workflow_status": self.workflow_status,
            "plan_digest": self.plan_digest,
            "final_status_sha256": self.final_status_sha256,
            "root_inventory": _d5._plain(self.root_inventory),
            "formal": self.formal,
            "unit_count": len(self.unit_rows),
        }
        if self.formal:
            document["d9_report_sha256"] = self.d9_report_sha256
            document["d9_report_canonical_digest"] = self.d9_report_canonical_digest
        return document


def _read_d9_report(path: Path | str) -> tuple[Mapping[str, Any], str, str]:
    report_path = Path(path)
    try:
        raw = report_path.read_bytes()
    except OSError as error:
        raise _d5.StageDFinalizationIntegrityError(
            f"cannot read D9 validation report: {report_path}"
        ) from error
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != _D9_REPORT_SHA256:
        raise _d5.StageDFinalizationIntegrityError(
            "D9 validation report file SHA-256 drifted"
        )
    try:
        report = strict_json_bytes(raw, "D9 validation report")
    except Exception as error:
        raise _d5.StageDFinalizationIntegrityError(
            "D9 validation report is not strict JSON"
        ) from error
    if not isinstance(report, Mapping):
        raise _d5.StageDFinalizationIntegrityError(
            "D9 validation report must be a JSON object"
        )
    canonical = canonical_sha256({
        key: value for key, value in report.items()
        if key != "canonical_digest"
    })
    if (
        canonical != _D9_REPORT_CANONICAL_DIGEST
        or report.get("canonical_digest") != _D9_REPORT_CANONICAL_DIGEST
    ):
        raise _d5.StageDFinalizationIntegrityError(
            "D9 validation report canonical digest drifted"
        )
    validation = report.get("validation")
    scope = report.get("scope_boundaries")
    final_files = report.get("final_files")
    status_file = (
        final_files.get("status.json")
        if isinstance(final_files, Mapping) else None
    )
    root_inventory = (
        validation.get("root_inventory")
        if isinstance(validation, Mapping) else None
    )
    if (
        report.get("status") != "PASS"
        or report.get("analysis_authorized") is not True
        or not isinstance(validation, Mapping)
        or validation.get("package_complete") is not True
        or validation.get("workflow_status") != "PASS"
        or validation.get("scientific_validity") != "PASS"
        or validation.get("analysis_authorized") is not True
        or validation.get("analysis_authorization") != "D9_AUTHORIZED_D10_FULL_ANALYSIS"
        or validation.get("d10_allowed_path") != "D10_FULL_ANALYSIS_AND_DECISION"
        or validation.get("final_status_sha256") != _D9_FINAL_STATUS_SHA256
        or not isinstance(root_inventory, Mapping)
        or root_inventory.get("canonical_digest") != _D9_ROOT_INVENTORY_DIGEST
        or not isinstance(status_file, Mapping)
        or status_file.get("file_sha256") != _D9_FINAL_STATUS_SHA256
        or status_file.get("canonical_digest") != _D9_FINAL_STATUS_CANONICAL_DIGEST
        or report.get("c10_status_preserved") != C10_STATUS_PRESERVED
        or not isinstance(scope, Mapping)
        or scope.get("d10_aggregate_statistics_decision_run") is not False
        or scope.get("paper_data_figures_overlays_or_contact_sheets_generated") is not False
    ):
        raise _d5.StageDFinalizationIntegrityError(
            "D9 validation report does not authorize full D10 analysis"
        )
    return _owned_mapping(report, "D9 report"), file_sha, canonical


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
            if not (
                set(parsed_l0) == PRODUCTION_L0_KEYS
                or set(parsed_l0) == (set(PRODUCTION_L0_KEYS) | {"trace_mode"})
                or set(parsed_l0) == set(_TRACELESS_PRODUCTION_L0_KEYS)
                or set(parsed_l0) == set(_FULL_NNS_PRODUCTION_L0_KEYS)
            ):
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
    *, allow_formal: bool = False, d9_report_path: Path | str | None = None,
) -> StageDValidatedAnalysisInput:
    """Validate one sealed root and issue same-snapshot D10 inputs."""

    report = report_sha = report_digest = None
    if allow_formal:
        report, report_sha, report_digest = _read_d9_report(
            D9_REPORT_PATH if d9_report_path is None else d9_report_path
        )

    typed_plan, typed_estimate, typed_preflight = _d5._validate_controls(
        plan, estimate, preflight
    )
    named_root = _d5._safe_root(output_root)
    with _d5._retained_root(named_root) as (
        root, root_descriptor, root_identity,
    ), _d5._root_locks(root):
        _d5._verify_root_binding(named_root, root_descriptor, root_identity)
        prechecked = _d5._checkpoint_control(
            root, typed_plan, typed_estimate, typed_preflight,
            allow_formal=allow_formal,
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
            allow_formal=allow_formal,
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
                allow_formal=allow_formal, sealed=False, source_records=source_records,
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
        expected_boundary = {
            "status": "NOT_PERFORMED", "analysis_performed": False,
            "bootstrap_performed": False, "decision_performed": False,
            "delta_fid_computed": False, "aggregate_effects_computed": False,
            "d6_eligibility": (
                "D10_FULL_ANALYSIS_ALLOWED" if allow_formal
                else "SYNTHETIC_ONLY"
            ),
        }
        if root_validation.evidence_report.get("analysis_boundary") != expected_boundary:
            raise _d5.StageDFinalizationIntegrityError(
                "D6/D10 input does not match its analysis boundary"
            )
        if allow_formal and (
            root_validation.scientific_validity != "PASS"
            or root_validation.analysis_authorized is not True
            or root_validation.final_status_sha256 != _D9_FINAL_STATUS_SHA256
            or root_validation.root_inventory.get("canonical_digest") != _D9_ROOT_INVENTORY_DIGEST
        ):
            raise _d5.StageDFinalizationIntegrityError(
                "formal D10 input does not match the D9-authorized root"
            )
        return StageDValidatedAnalysisInput(
            root_validation=root_validation, unit_rows=unit_rows,
            formal=allow_formal, d9_report=report,
            d9_report_sha256=report_sha,
            d9_report_canonical_digest=report_digest, _issuer=_ISSUER,
        )


__all__ = [
    "StageDValidatedAnalysisInput", "StageDValidatedAnalysisUnit",
    "validate_stage_d_analysis_input",
]
