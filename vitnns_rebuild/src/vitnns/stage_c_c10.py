"""Stage C10 read-only aggregation, statistics, and final decision.

The C10 boundary consumes only sealed C9 L0/L1 evidence.  It never opens a
source image, invokes a model backend, or writes into the C9 evidence root.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .constants import REPO_ROOT
from .io_utils import sha256_file
from .stage_c_maps import SignSemantics, SignView, stage_c_metric_ready
from .stage_c_metrics import (
    bidirectional_kl, hotspot_jaccard_top_10_percent,
    hotspot_jaccard_top_20_percent, jensen_shannon_divergence,
    normalized_entropy, pearson, spearman,
)
from .stage_c_provenance import canonical_sha256, capture_git_state
from .stage_c_selection import _sample_candidate_metrics
from .stage_c_c10_stats import (
    STABILITY_PERTURBATIONS, audit_holm_contract, decide, evaluate_effect,
    evaluate_spatial, ordered_sample_ids_sha256, summarize_values,
)
from .stage_c_c10_table import (
    NNS_METHOD_ID, DIRECT_METHOD_ID, MATCHED_METHOD_ID, base_row,
    build_c10_long_table, build_row_manifest,
    expected_base_row_coordinates, expected_base_row_count, make_row,
    metric_id_from_selectors, read_csv_bytes, rows_to_csv_bytes,
    schema_document, source_binding, validate_rows,
)


class StageC10Error(RuntimeError):
    """Base error for C10 post-processing."""


class StageC10IntegrityError(StageC10Error):
    """Raised when sealed source or C10 output evidence is inconsistent."""


class StageC10BlockedError(StageC10Error):
    """Raised for a complete, machine-verifiable C10 acceptance blocker."""


SCHEMA_VERSION = 1
PRIMARY = (0, 9)
WINNERS = ((0, 9), (4, 2), (6, 10), (11, 2))
EXPLORATORY = WINNERS[1:]
OUTPUT_FILES = (
    "per_sample.csv", "per_sample.schema.json",
    "per_sample_manifest.json", "summaries.json", "bootstrap.json",
    "holm.json", "reconciliation.json", "decision.json",
    "stage_c_report.json", "provenance.json",
)
MANIFEST_FILENAME = "manifest.json"
STATUS_FILENAME = "status.json"
C10_PRODUCER_ARCHIVE_REF = "archive/c10-recovery-v2-producer"
C10_CODE_SURFACE = "src/vitnns"
_GIT_OID = re.compile(r"[0-9a-f]{40}\Z")
_STAGE_D_CAP_KEYS = (
    "stage_d_500_total_artifact_bytes_max",
    "stage_d_500_h20_gpu_hours_max",
    "stage_d_h20_gpu_hours_per_image_max",
)
_INVALIDATING_REASONS = frozenset({
    "empty_valid_region", "shape_mismatch", "non_finite",
    "degenerate_direction", "missing_artifact",
})
_REQUIRED_METRIC_COVERAGE = {
    "pearson": ["clean_spatial_structure_similarity/pearson"],
    "spearman": ["clean_spatial_structure_similarity/spearman"],
    "jensen_shannon_divergence": ["clean_spatial_structure_similarity/jensen_shannon_divergence"],
    "kl_left_to_right": ["clean_spatial_structure_similarity/kl:left_to_right"],
    "kl_right_to_left": ["clean_spatial_structure_similarity/kl:right_to_left"],
    "hotspot_jaccard_top_10_percent": ["clean_spatial_structure_similarity/hotspot_jaccard_top_10_percent"],
    "hotspot_jaccard_top_20_percent": ["clean_spatial_structure_similarity/hotspot_jaccard_top_20_percent"],
    "normalized_entropy": ["normalized_entropy/normalized_entropy"],
    "conservation_absolute_error": ["ig_conservation/ig_conservation_abs", "matched_contribution/conservation_abs_error"],
    "conservation_relative_error": ["ig_conservation/ig_conservation_rel"],
    "endpoint_intervention_parity": ["endpoint_parity/*"],
    "deletion_auc": ["fidelity_aggregate/fidelity_auc:deletion"],
    "insertion_auc": ["fidelity_aggregate/fidelity_auc:insertion"],
    "sufficiency_20_percent": ["top20_image/sufficiency"],
    "comprehensiveness_20_percent": ["top20_image/comprehensiveness"],
    "stability_pearson": ["stability/pearson", "raw_structural_stability/pearson"],
    "stability_spearman": ["stability/spearman", "nns_residual/residual_signed_spearman"],
    "stability_jensen_shannon_divergence": ["stability/jensen_shannon"],
    "stability_hotspot_jaccard": ["stability/top10_jaccard", "stability/top20_jaccard"],
    "wall_time_seconds": ["resources/wall_time_seconds"],
    "cuda_peak_allocated_bytes": ["resources/cuda_peak_allocated_bytes"],
    "cuda_peak_reserved_bytes": ["resources/cuda_peak_reserved_bytes"],
    "artifact_bytes": ["resources/artifact_bytes"],
}


def validate_metric_source_path(path: str) -> str:
    """Accept only relative L0 JSON or L1 numeric evidence paths.

    C10 deliberately excludes raw/audit L2, display L3, images, and any path
    that could escape its already-validated execution-unit attempt directory.
    """

    if not isinstance(path, str):
        raise TypeError("metric source path must be a string")
    if not path or "\\" in path:
        raise ValueError("metric source path must be a non-empty POSIX path")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("metric source path must be relative and traversal-free")
    if len(candidate.parts) < 2 or candidate.parts[0] not in {"L0", "L1"}:
        raise ValueError("metric source must be sealed L0 or L1 evidence")
    lowered = candidate.as_posix().lower()
    if "display" in lowered or "heatmap" in lowered:
        raise ValueError("display artifacts are forbidden metric inputs")
    if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        raise ValueError("image/display files are forbidden metric inputs")
    if candidate.parts[0] == "L0" and candidate.suffix.lower() != ".json":
        raise ValueError("L0 metric sources must be JSON")
    if candidate.parts[0] == "L1" and candidate.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError("L1 metric sources must be numeric NPY/NPZ artifacts")
    return candidate.as_posix()


def reconcile_expected_rows(
    expected_keys: Sequence[str], observed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Perform an expected-key left join without dropping absent evidence."""

    if isinstance(expected_keys, (str, bytes)) or isinstance(
        observed_rows, (str, bytes, Mapping)
    ):
        raise TypeError("expected_keys and observed_rows must be sequences")
    expected = list(expected_keys)
    if any(not isinstance(key, str) or not key for key in expected):
        raise ValueError("expected row keys must be non-empty strings")
    if len(expected) != len(set(expected)):
        raise ValueError("expected row keys must be unique")
    by_key: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(observed_rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"observed_rows[{index}] must be a mapping")
        row = dict(raw)
        key = row.get("row_key", row.get("row_id"))
        if not isinstance(key, str) or not key:
            raise ValueError(f"observed_rows[{index}] lacks a row key")
        if key in by_key:
            raise ValueError(f"duplicate observed row key: {key}")
        by_key[key] = row
    extras = set(by_key) - set(expected)
    if extras:
        raise ValueError(f"unexpected observed row keys: {sorted(extras)}")
    result: list[dict[str, Any]] = []
    for key in expected:
        if key in by_key:
            result.append(by_key[key])
        else:
            result.append({
                "row_key": key, "status": "SOURCE_MISSING",
                "value": None, "na_reason": "missing_artifact",
                "invalidating": True,
            })
    return result


def partition_rows_by_scope(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return calibration, holdout, and descriptive all-20 row views."""

    if isinstance(rows, (str, bytes, Mapping)):
        raise TypeError("rows must be a sequence of mappings")
    calibration: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping")
        row = dict(raw)
        role = row.get("role")
        if role == "calibration":
            calibration.append(row)
        elif role == "holdout":
            holdout.append(row)
        else:
            raise ValueError(f"rows[{index}] has invalid role {role!r}")
        all_rows.append(row)
    return {
        "calibration": calibration,
        "holdout": holdout,
        "all_20": all_rows,
    }


def _plain(value: Any) -> Any:
    method = getattr(value, "to_dict", None)
    value = method() if callable(method) else value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise StageC10IntegrityError(
        f"value contains non-JSON-safe {type(value).__name__}"
    )


def _selection_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _plain(value)
        for key, value in record.items()
        if key not in {"l1_clean_array", "l1_clean_binding"}
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        _plain(value), indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _strict_json_bytes(data: bytes, context: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StageC10IntegrityError(f"{context} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except StageC10IntegrityError:
        raise
    except Exception as error:
        raise StageC10IntegrityError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise StageC10IntegrityError(f"{context} must be a JSON object")
    if data != _json_bytes(value):
        raise StageC10IntegrityError(f"{context} is not canonical Stage C JSON")
    return value


def _document(document_type: str, **fields: Any) -> dict[str, Any]:
    result = {
        "schema_version": SCHEMA_VERSION,
        "document_type": document_type,
        **_plain(fields),
    }
    result["canonical_digest"] = canonical_sha256(result)
    return result


def _metric_result(value: Any) -> tuple[Any, str, str | None, bool, dict[str, Any]]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        if isinstance(value, bool) or (
            isinstance(value, (int, float)) and math.isfinite(float(value))
        ):
            return value, "OK", None, False, {}
        raise StageC10IntegrityError("derived metric value is invalid")
    status = value.get("status")
    if status == "OK":
        return value.get("value"), "OK", None, False, dict(value.get("details", {}))
    if status == "NA":
        reason = value.get("reason")
        if not isinstance(reason, str):
            raise StageC10IntegrityError("derived structured NA lacks reason")
        return None, ("INVALID" if reason in _INVALIDATING_REASONS else "NA"), reason, reason in _INVALIDATING_REASONS, {
            key: item for key, item in value.items()
            if key not in {"status", "value", "reason"}
        }
    raise StageC10IntegrityError("derived metric status is invalid")


def _append_derived(
    rows: list[dict[str, Any]], base: Mapping[str, Any], *, family: str,
    metric: str, result: Any, source_path: str, artifact_sha256: str,
    selectors: Mapping[str, Any], reference_record: Mapping[str, Any] | None = None,
) -> None:
    value, status, reason, invalidating, details = _metric_result(result)
    secondary: list[dict[str, Any]] = []
    reference_method = None
    if reference_record is not None:
        reference_method = reference_record["unit_key"]["method_id"]
        secondary.append(source_binding(
            reference_record, source_kind="c9_l1_metric_ready",
            source_field_path="/L1/clean.npy",
            source_artifact_sha256=reference_record["l1_clean_binding"]["sha256"],
        ))
    rows.append(make_row(
        base, metric_family=family,
        metric_id=metric_id_from_selectors(metric, **dict(selectors)),
        source_field_path=source_path, source_kind="c9_l1_metric_ready",
        source_artifact_sha256=artifact_sha256, value=value, status=status,
        na_reason=reason, invalidating=invalidating, details=details,
        reference_method_id=reference_method,
        secondary_sources_json=secondary,
        view=selectors.get("view"),
        component=selectors.get("component"),
    ))


def _l1_derived_rows(
    records: Sequence[Mapping[str, Any]], plan_units: Sequence[Any],
) -> list[dict[str, Any]]:
    """Compute registered clean-map metrics from sealed metric-ready L1."""

    units = [unit.to_dict() if hasattr(unit, "to_dict") else dict(unit) for unit in plan_units]
    by_digest = {record["unit_key"]["canonical_digest"]: record for record in records}
    by_identity = {
        (record["unit_key"]["sample_id"], record["unit_key"]["config_id"],
         record["unit_key"]["method_id"]): record
        for record in records
    }
    result_rows: list[dict[str, Any]] = []
    mask = np.ones((14, 14), dtype=np.bool_)
    for index, unit in enumerate(units):
        record = by_digest[unit["key"]["canonical_digest"]]
        array = record["l1_clean_array"]
        tagged = stage_c_metric_ready(array)
        semantics = SignSemantics(record["scalars"]["sign_semantics"])
        base = base_row(record, unit, plan_index=index)
        artifact_hash = record["l1_clean_binding"]["sha256"]
        views = (SignView.POSITIVE,) + (
            (SignView.NEGATIVE_MAGNITUDE,)
            if semantics is SignSemantics.SIGNED else ()
        )
        for view in views:
            _append_derived(
                result_rows, base, family="normalized_entropy",
                metric="normalized_entropy",
                result=normalized_entropy(
                    tagged, mask=mask, view=view, semantics=semantics,
                ),
                source_path="/L1/clean.npy",
                artifact_sha256=artifact_hash, selectors={"view": view.value},
            )
        key = record["unit_key"]
        if key["method_id"] != NNS_METHOD_ID:
            continue
        reference = by_identity[(key["sample_id"], key["config_id"], DIRECT_METHOD_ID)]
        reference_tagged = stage_c_metric_ready(reference["l1_clean_array"] )
        left_semantics = semantics
        right_semantics = SignSemantics(reference["scalars"]["sign_semantics"])
        # Raw attention is nonnegative: signed comparisons are undefined; the
        # registered cross-method structural comparison uses positive mass.
        kwargs = {
            "mask": mask, "view": SignView.POSITIVE,
            "left_semantics": left_semantics,
            "right_semantics": right_semantics,
        }
        pair = {"view": "positive", "comparison": "nns_vs_raw_attention"}
        metrics = (
            ("pearson", pearson(tagged, reference_tagged, **kwargs)),
            ("spearman", spearman(tagged, reference_tagged, **kwargs)),
            ("jensen_shannon_divergence", jensen_shannon_divergence(tagged, reference_tagged, **kwargs)),
            ("hotspot_jaccard_top_10_percent", hotspot_jaccard_top_10_percent(tagged, reference_tagged, **kwargs)),
            ("hotspot_jaccard_top_20_percent", hotspot_jaccard_top_20_percent(tagged, reference_tagged, **kwargs)),
        )
        kl = bidirectional_kl(tagged, reference_tagged, **kwargs)
        for metric, outcome in metrics:
            _append_derived(
                result_rows, base, family="clean_spatial_structure_similarity",
                metric=metric, result=outcome, source_path="/L1/clean.npy",
                artifact_sha256=artifact_hash, selectors=pair,
                reference_record=reference,
            )
        if getattr(kl, "status", None) == "OK":
            for component, outcome in (("left_to_right", kl.left_to_right), ("right_to_left", kl.right_to_left)):
                _append_derived(
                    result_rows, base, family="clean_spatial_structure_similarity",
                    metric="kl", result=outcome, source_path="/L1/clean.npy",
                    artifact_sha256=artifact_hash, selectors={**pair, "component": component},
                    reference_record=reference,
                )
        else:
            for component in ("left_to_right", "right_to_left"):
                _append_derived(
                    result_rows, base, family="clean_spatial_structure_similarity",
                    metric="kl", result=kl, source_path="/L1/clean.npy",
                    artifact_sha256=artifact_hash, selectors={**pair, "component": component},
                    reference_record=reference,
                )
        # The registered signed and negative comparisons are inapplicable
        # when one side is nonnegative raw attention.  They remain explicit
        # structured-NA rows instead of silently disappearing or becoming
        # fabricated all-zero negative maps.
        for unsupported_view in (SignView.SIGNED, SignView.NEGATIVE_MAGNITUDE):
            unsupported_kwargs = {**kwargs, "view": unsupported_view}
            unsupported_pair = {
                "view": unsupported_view.value,
                "comparison": "nns_vs_raw_attention",
            }
            unsupported_metrics = [
                ("pearson", pearson(tagged, reference_tagged, **unsupported_kwargs)),
                ("spearman", spearman(tagged, reference_tagged, **unsupported_kwargs)),
            ]
            if unsupported_view is SignView.NEGATIVE_MAGNITUDE:
                unsupported_metrics.extend([
                    ("jensen_shannon_divergence", jensen_shannon_divergence(
                        tagged, reference_tagged, **unsupported_kwargs
                    )),
                    ("hotspot_jaccard_top_10_percent",
                     hotspot_jaccard_top_10_percent(
                         tagged, reference_tagged, **unsupported_kwargs
                     )),
                    ("hotspot_jaccard_top_20_percent",
                     hotspot_jaccard_top_20_percent(
                         tagged, reference_tagged, **unsupported_kwargs
                     )),
                ])
                unsupported_kl = bidirectional_kl(
                    tagged, reference_tagged, **unsupported_kwargs
                )
                unsupported_metrics.extend([
                    ("kl", unsupported_kl, "left_to_right"),
                    ("kl", unsupported_kl, "right_to_left"),
                ])
            for item in unsupported_metrics:
                metric, outcome, *component = item
                selectors = dict(unsupported_pair)
                if component:
                    selectors["component"] = component[0]
                _append_derived(
                    result_rows, base,
                    family="clean_spatial_structure_similarity",
                    metric=metric, result=outcome,
                    source_path="/L1/clean.npy",
                    artifact_sha256=artifact_hash, selectors=selectors,
                    reference_record=reference,
                )
    return result_rows


def _expected_l1_derived_row_count(
    records: Sequence[Mapping[str, Any]],
) -> int:
    count = 0
    for record in records:
        semantics = SignSemantics(record["scalars"]["sign_semantics"])
        count += 1 + int(semantics is SignSemantics.SIGNED)
        if record["unit_key"]["method_id"] == NNS_METHOD_ID:
            # positive: 7; signed: 2 explicit NA; negative: 7 explicit NA.
            count += 16
    return count


def _expected_scientific_derived_row_count(plan_units: Sequence[Any]) -> int:
    units = [
        unit.to_dict() if hasattr(unit, "to_dict") else dict(unit)
        for unit in plan_units
    ]
    sample_ids = {unit["sample"]["sample_id"] for unit in units}
    winner_delta_rows = len(sample_ids) * len(WINNERS) * 2
    top20_image_rows = sum(
        6 for unit in units if unit["fidelity_eligible"]
    )
    return winner_delta_rows + top20_image_rows


_APPLICABILITY_FIELDS = (
    "plan_index", "sample_id", "method_id", "reference_method_id",
    "config_id", "source_unit_key_sha256", "metric_family",
    "metric_id", "source_kind", "source_field_path",
)


def _applicability_coordinate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: row.get(name) for name in _APPLICABILITY_FIELDS}


def _expected_l1_row_coordinates(
    records: Sequence[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    for record in records:
        key = record["unit_key"]
        base = {
            "plan_index": record["plan_index"],
            "sample_id": key["sample_id"],
            "method_id": key["method_id"],
            "reference_method_id": None,
            "config_id": key["config_id"],
            "source_unit_key_sha256": key["canonical_digest"],
            "source_kind": "c9_l1_metric_ready",
            "source_field_path": "/L1/clean.npy",
        }
        semantics = SignSemantics(record["scalars"]["sign_semantics"])
        views = ["positive"]
        if semantics is SignSemantics.SIGNED:
            views.append("negative_magnitude")
        for view in views:
            yield {
                **base, "metric_family": "normalized_entropy",
                "metric_id": metric_id_from_selectors(
                    "normalized_entropy", view=view
                ),
            }
        if key["method_id"] != NNS_METHOD_ID:
            continue
        comparison_base = {**base, "reference_method_id": DIRECT_METHOD_ID}
        for view, metrics in (
            ("positive", (
                "pearson", "spearman", "jensen_shannon_divergence",
                "hotspot_jaccard_top_10_percent",
                "hotspot_jaccard_top_20_percent",
            )),
            ("signed", ("pearson", "spearman")),
            ("negative_magnitude", (
                "pearson", "spearman", "jensen_shannon_divergence",
                "hotspot_jaccard_top_10_percent",
                "hotspot_jaccard_top_20_percent",
            )),
        ):
            selectors = {
                "view": view, "comparison": "nns_vs_raw_attention"
            }
            for metric in metrics:
                yield {
                    **comparison_base,
                    "metric_family": "clean_spatial_structure_similarity",
                    "metric_id": metric_id_from_selectors(
                        metric, **selectors
                    ),
                }
            if view != "signed":
                for component in ("left_to_right", "right_to_left"):
                    yield {
                        **comparison_base,
                        "metric_family": "clean_spatial_structure_similarity",
                        "metric_id": metric_id_from_selectors(
                            "kl", **selectors, component=component
                        ),
                    }


def _expected_scientific_row_coordinates(
    plan_units: Sequence[Any],
) -> Iterator[dict[str, Any]]:
    units = [
        unit.to_dict() if hasattr(unit, "to_dict") else dict(unit)
        for unit in plan_units
    ]
    by_identity = {
        (unit["key"]["sample_id"], unit["key"]["config_id"],
         unit["key"]["method_id"]): (index, unit)
        for index, unit in enumerate(units)
    }
    sample_order = list(dict.fromkeys(
        unit["key"]["sample_id"] for unit in units
    ))
    for block, head in WINNERS:
        config_id = f"block_{block:02d}_head_{head:02d}"
        for sample_id in sample_order:
            index, unit = by_identity[(sample_id, config_id, NNS_METHOD_ID)]
            key = unit["key"]
            for scale in ("patch_16x16", "fine_4x4_cell"):
                yield {
                    "plan_index": index, "sample_id": sample_id,
                    "method_id": NNS_METHOD_ID,
                    "reference_method_id": MATCHED_METHOD_ID,
                    "config_id": config_id,
                    "source_unit_key_sha256": key["canonical_digest"],
                    "metric_family": "value_path_delta_fidelity",
                    "metric_id": metric_id_from_selectors(
                        "delta_fid", pair="nns_vs_matched", scale=scale
                    ),
                    "source_kind": "c10_derived_l0_pair",
                    "source_field_path": "/derived/value_path_delta_fidelity",
                }
    for index, unit in enumerate(units):
        if not unit["fidelity_eligible"]:
            continue
        key = unit["key"]
        for scale in ("patch_16x16", "fine_4x4_cell"):
            for metric in (
                "sufficiency", "sufficiency_gap", "comprehensiveness"
            ):
                yield {
                    "plan_index": index, "sample_id": key["sample_id"],
                    "method_id": key["method_id"],
                    "reference_method_id": None,
                    "config_id": key["config_id"],
                    "source_unit_key_sha256": key["canonical_digest"],
                    "metric_family": "top20_image",
                    "metric_id": metric_id_from_selectors(
                        metric,
                        ranking="target_oriented_positive_tie_permuted",
                        scale=scale, statistic="mean_over_5_tie_repeats",
                    ),
                    "source_kind": "c10_derived_l0_nuisance_mean",
                    "source_field_path": "/derived/top20_mean_over_5_tie_repeats",
                }


def _coverage_match(row: Mapping[str, Any], specification: str) -> bool:
    family, metric_spec = specification.split("/", 1)
    if row["metric_family"] != family:
        return False
    metric_base, separator, qualifier = metric_spec.partition(":")
    observed_base = row["metric_id"].split("|", 1)[0]
    if metric_base != "*" and observed_base != metric_base:
        return False
    if not separator:
        return True
    return qualifier in {row.get("curve_direction"), row.get("component")}


def _metric_coverage_audit(
    rows: Sequence[Mapping[str, Any]], config: Any,
    expected_sample_ids: Mapping[str, Sequence[str]],
    expected_coordinates: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    required = list(config.metrics["required"])
    if set(required) != set(_REQUIRED_METRIC_COVERAGE) or len(required) != len(set(required)):
        raise StageC10IntegrityError(
            "C10 metric applicability registry does not exactly cover metrics.required"
        )
    exact_applicability = None
    if expected_coordinates is not None:
        observed_counter = Counter(
            tuple(_applicability_coordinate(item).values()) for item in rows
        )
        ordered_digest = hashlib.sha256()
        expected_count = 0
        for item in expected_coordinates:
            coordinate = _applicability_coordinate(item)
            key = tuple(coordinate.values())
            if observed_counter[key] <= 0:
                raise StageC10IntegrityError(
                    "C10 rows omit an exact applicability coordinate"
                )
            observed_counter[key] -= 1
            if observed_counter[key] == 0:
                del observed_counter[key]
            ordered_digest.update(json.dumps(
                coordinate, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8"))
            ordered_digest.update(b"\n")
            expected_count += 1
        if observed_counter:
            raise StageC10IntegrityError(
                "C10 rows differ from the exact applicability coordinate multiset"
            )
        exact_applicability = {
            "status": "PASS", "coordinate_count": expected_count,
            "ordered_coordinates_jsonl_sha256": ordered_digest.hexdigest(),
        }
    records = []
    all_ids = set(expected_sample_ids["all_20"])
    for metric_name in required:
        patterns = _REQUIRED_METRIC_COVERAGE[metric_name]
        pattern_records = []
        for pattern in patterns:
            matches = [row for row in rows if _coverage_match(row, pattern)]
            observed_ids = {row["sample_id"] for row in matches}
            if not matches or observed_ids != all_ids:
                raise StageC10IntegrityError(
                    f"required metric coverage is incomplete for {metric_name}:{pattern}"
                )
            pattern_records.append({
                "pattern": pattern, "row_count": len(matches),
                "sample_count": len(observed_ids),
                "status_counts": dict(sorted(
                    Counter(row["status"] for row in matches).items()
                )),
                "ordered_row_keys_sha256": canonical_sha256({
                    "row_keys": [row["row_key_sha256"] for row in matches]
                }),
            })
        records.append({"required_metric": metric_name, "patterns": pattern_records})
    return {
        "status": "PASS", "required_metric_count": len(required),
        "required_metrics": required, "records": records,
        "exact_applicability": exact_applicability,
    }


def _aggregate_by_image(
    rows: Sequence[Mapping[str, Any]], *, method_id: str, config_id: str,
    metric_family: str, metric_base: str, selectors: Mapping[str, Any],
) -> dict[str, float]:
    metric_id = metric_id_from_selectors(metric_base, **dict(selectors))
    matches = [
        row for row in rows
        if row["method_id"] == method_id and row["config_id"] == config_id
        and row["metric_family"] == metric_family and row["metric_id"] == metric_id
    ]
    result: dict[str, float] = {}
    for row in matches:
        if row["status"] != "OK" or isinstance(row["value"], bool) or not isinstance(row["value"], (int, float)):
            raise StageC10IntegrityError(f"required image metric is nonnumeric: {metric_id}")
        if row["sample_id"] in result:
            raise StageC10IntegrityError(f"duplicate image metric: {metric_id}")
        result[row["sample_id"]] = float(row["value"])
    return result


def _aggregate_auc(scalars: Mapping[str, Any], scale: str, direction: str) -> tuple[float, str]:
    found: list[tuple[float, str]] = []
    for item in scalars.get("fidelity_aggregates", []):
        curves = item.get("curves") if isinstance(item, Mapping) else None
        if (
            not isinstance(curves, Sequence)
            or isinstance(curves, (str, bytes))
            or len(curves) != 5
        ):
            raise StageC10IntegrityError("fidelity aggregate lacks five repeats")
        selectors = {
            (curve.get("scale"), curve.get("direction"), curve.get("ranking_kind"))
            for curve in curves if isinstance(curve, Mapping)
        }
        if len(selectors) != 1:
            raise StageC10IntegrityError("fidelity aggregate mixes selectors")
        if selectors == {(scale, direction, "target_oriented_positive_tie_permuted")}:
            value = item.get("auc_mean")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise StageC10IntegrityError("required aggregate AUC is not finite")
            path = (
                "/scalars/fidelity_aggregates"
                f"{{direction={direction},ranking_kind=target_oriented_positive_tie_permuted,scale={scale}}}/auc_mean"
            )
            found.append((float(value), path))
    if len(found) != 1:
        raise StageC10IntegrityError(
            f"expected one aggregate for scale={scale}, direction={direction}"
        )
    return found[0]


def _derived_scientific_rows(
    records: Sequence[Mapping[str, Any]], plan_units: Sequence[Any],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], list[dict[str, Any]]]]:
    units = [unit.to_dict() if hasattr(unit, "to_dict") else dict(unit) for unit in plan_units]
    by_digest = {record["unit_key"]["canonical_digest"]: record for record in records}
    by_identity = {
        (record["unit_key"]["sample_id"], record["unit_key"]["config_id"],
         record["unit_key"]["method_id"]): record
        for record in records
    }
    unit_by_digest = {unit["key"]["canonical_digest"]: (index, unit) for index, unit in enumerate(units)}
    rows: list[dict[str, Any]] = []
    winner_samples: dict[tuple[int, int], list[dict[str, Any]]] = {winner: [] for winner in WINNERS}
    sample_order: list[str] = []
    roles: dict[str, str] = {}
    for unit in units:
        sample_id = unit["sample"]["sample_id"]
        if sample_id not in roles:
            sample_order.append(sample_id)
            roles[sample_id] = unit["sample"]["role"]

    for block, head in WINNERS:
        config_id = f"block_{block:02d}_head_{head:02d}"
        for sample_id in sample_order:
            try:
                direct = by_identity[(sample_id, config_id, DIRECT_METHOD_ID)]
                nns = by_identity[(sample_id, config_id, NNS_METHOD_ID)]
                matched = by_identity[(sample_id, config_id, MATCHED_METHOD_ID)]
            except KeyError as error:
                raise StageC10IntegrityError(
                    f"winner sample lacks an exact Value-method triple: {sample_id}:{config_id}"
                ) from error
            try:
                metrics, failures = _sample_candidate_metrics(
                    {
                        DIRECT_METHOD_ID: _selection_record(direct),
                        NNS_METHOD_ID: _selection_record(nns),
                        MATCHED_METHOD_ID: _selection_record(matched),
                    },
                    sample_id, block, head,
                )
            except Exception as error:
                raise StageC10IntegrityError(
                    f"winner sample validity parsing failed: {sample_id}:{config_id}: {error}"
                ) from error
            if failures or metrics.get("valid") is not True:
                raise StageC10IntegrityError(
                    f"validated C9 winner evidence became invalid: {sample_id}:{config_id}:{failures}"
                )
            nns_scalars = nns["scalars"]
            matched_scalars = matched["scalars"]
            deltas: dict[str, float] = {}
            delta_sources: dict[str, list[dict[str, Any]]] = {}
            nns_index, nns_unit = unit_by_digest[nns["unit_key"]["canonical_digest"]]
            base = base_row(nns, nns_unit, plan_index=nns_index)
            matched_source = source_binding(
                matched, source_kind="c9_l0_scalar",
                source_field_path="/scalars/fidelity_aggregates",
            )
            for scale in ("patch_16x16", "fine_4x4_cell"):
                ni, ni_path = _aggregate_auc(nns_scalars, scale, "insertion")
                nd, nd_path = _aggregate_auc(nns_scalars, scale, "deletion")
                mi, mi_path = _aggregate_auc(matched_scalars, scale, "insertion")
                md, md_path = _aggregate_auc(matched_scalars, scale, "deletion")
                delta = 0.5 * ((ni - mi) + (md - nd))
                if not math.isfinite(delta):
                    raise StageC10IntegrityError("delta_fid is non-finite")
                selectors = {"pair": "nns_vs_matched", "scale": scale}
                details = {
                    "formula": "0.5*((InsertionAUC_NNS-InsertionAUC_matched)+(DeletionAUC_matched-DeletionAUC_NNS))",
                    "nns_insertion_auc": ni, "nns_deletion_auc": nd,
                    "matched_insertion_auc": mi, "matched_deletion_auc": md,
                    "source_field_paths": [ni_path, nd_path, mi_path, md_path],
                    "tie_repeats_are_within_image_nuisance": True,
                }
                rows.append(make_row(
                    base, metric_family="value_path_delta_fidelity",
                    metric_id=metric_id_from_selectors("delta_fid", **selectors),
                    source_field_path="/derived/value_path_delta_fidelity",
                    source_kind="c10_derived_l0_pair", value=delta,
                    details=details, reference_method_id=MATCHED_METHOD_ID,
                    secondary_sources_json=[matched_source], fidelity_scale=scale,
                ))
                deltas[scale] = delta
                delta_sources[scale] = [
                    {"method_id": NNS_METHOD_ID, "paths": [ni_path, nd_path]},
                    {"method_id": MATCHED_METHOD_ID, "paths": [mi_path, md_path]},
                ]
            contribution = matched_scalars["provenance_fragment"]["matched_contribution"]
            residual_records = {
                item["perturbation_id"]: item
                for item in nns_scalars["residual_stability"]
            }
            gaussian = residual_records["gaussian_rgb_std_0_01"]
            rho_obj = gaussian["clean"]["rho"]
            rho = rho_obj.get("value") if rho_obj.get("status") == "OK" else None
            residuals = {
                perturbation: (
                    residual_records[perturbation]["signed_spearman"].get("value")
                    if residual_records[perturbation]["signed_spearman"].get("status") == "OK"
                    else None
                )
                for perturbation in STABILITY_PERTURBATIONS
            }
            winner_samples[(block, head)].append({
                "sample_id": sample_id, "role": roles[sample_id],
                "config_id": config_id, "valid": True,
                "clean_top1_correct": nns_scalars["prediction"]["clean_top1_correct"],
                "delta_fid_fine4": deltas["fine_4x4_cell"],
                "delta_fid_patch": deltas["patch_16x16"],
                "s_patch": float(contribution["patch_support"]),
                "positive_patch_evidence_fraction_of_token_l1": float(
                    contribution["positive_patch_evidence_fraction"]
                ),
                "rho": None if rho is None else float(rho),
                "residual_spearman": {
                    key: None if value is None else float(value)
                    for key, value in residuals.items()
                },
                "delta_source_bindings": delta_sources,
            })

    # Top-20 tie repeats are retained atomically in the base table and reduced
    # within each image here; they never become 5 independent clusters.
    for record in records:
        if not record["scalars"].get("fidelity_enabled"):
            continue
        index, unit = unit_by_digest[record["unit_key"]["canonical_digest"]]
        base = base_row(record, unit, plan_index=index)
        items = record["scalars"].get("top20", [])
        for scale in ("patch_16x16", "fine_4x4_cell"):
            selected = sorted(
                [item for item in items if item.get("scale") == scale],
                key=lambda item: item.get("replicate"),
            )
            if [item.get("replicate") for item in selected] != list(range(5)):
                raise StageC10IntegrityError("top20 nuisance repeats are incomplete")
            for field in ("sufficiency", "sufficiency_gap", "comprehensiveness"):
                values = [float(item[field]) for item in selected]
                rows.append(make_row(
                    base, metric_family="top20_image",
                    metric_id=metric_id_from_selectors(
                        field, ranking="target_oriented_positive_tie_permuted",
                        scale=scale, statistic="mean_over_5_tie_repeats",
                    ),
                    source_field_path="/derived/top20_mean_over_5_tie_repeats",
                    source_kind="c10_derived_l0_nuisance_mean",
                    value=float(np.mean(np.asarray(values, dtype=np.float64))),
                    fidelity_scale=scale, statistic="mean_over_5_tie_repeats",
                    ranking_kind="target_oriented_positive_tie_permuted",
                    details={"replicate_count": 5, "values": values},
                ))
    return rows, winner_samples


def _summary_document(
    winner_samples: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    expected_sample_ids: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    bootstraps: list[dict[str, Any]] = []
    by_winner: dict[str, Any] = {}
    for winner in WINNERS:
        samples = list(winner_samples[winner])
        config_id = f"block_{winner[0]:02d}_head_{winner[1]:02d}"
        expected_ids = {"calibration": [], "holdout": [], "all_20": []}
        for item in samples:
            expected_ids[item["role"]].append(item["sample_id"])
            expected_ids["all_20"].append(item["sample_id"])
        if len(expected_ids["calibration"]) != 5 or len(expected_ids["holdout"]) != 15 or len(samples) != 20:
            raise StageC10IntegrityError("winner role split is not exact 5/15/20")
        if expected_sample_ids is not None and any(
            list(expected_ids[scope]) != list(expected_sample_ids[scope])
            for scope in ("calibration", "holdout", "all_20")
        ):
            raise StageC10IntegrityError(
                "winner summaries do not use the exact frozen sample order"
            )
        winner_doc: dict[str, Any] = {
            "config_id": config_id,
            "role": "primary" if winner == PRIMARY else "exploratory",
            "affects_primary_decision": winner == PRIMARY,
            "per_sample": _plain(samples), "metrics": {},
        }
        metric_getters = {
            "delta_fid_fine4": lambda item: item["delta_fid_fine4"],
            "delta_fid_patch": lambda item: item["delta_fid_patch"],
            "patch_support": lambda item: item["s_patch"],
            "positive_patch_evidence_fraction": lambda item: item["positive_patch_evidence_fraction_of_token_l1"],
            "clean_residual_rho": lambda item: item["rho"],
            **{
                f"residual_signed_spearman_{perturbation}": (
                    lambda item, perturbation=perturbation: item["residual_spearman"][perturbation]
                )
                for perturbation in STABILITY_PERTURBATIONS
            },
        }
        for metric_name, getter in metric_getters.items():
            metric_scopes: dict[str, Any] = {}
            for scope in ("calibration", "holdout", "all_20"):
                selected = [item for item in samples if scope == "all_20" or item["role"] == scope]
                values = [getter(item) for item in selected]
                if any(value is None for value in values):
                    metric_scopes[scope] = {
                        "metric_id": metric_id_from_selectors(metric_name, config=config_id),
                        "scope": scope, "unit": "image",
                        "confirmatory": scope == "holdout",
                        "n": len(selected), "sample_ids": [item["sample_id"] for item in selected],
                        "per_sample": [
                            {"sample_index": index, "sample_id": item["sample_id"], "value": value}
                            for index, (item, value) in enumerate(zip(selected, values))
                        ],
                        "values": values, "mean": None, "median": None,
                        "standard_deviation_sample_ddof_1": None,
                        "positive_count": sum(
                            value is not None and value > 0 for value in values
                        ),
                        "bootstrap_95_percent_ci": None, "bootstrap": None,
                        "status": "INCOMPLETE_VECTOR",
                        "na_count": sum(value is None for value in values),
                    }
                else:
                    summary = summarize_values(
                        metric_id_from_selectors(metric_name, config=config_id),
                        values, scope=scope,
                        sample_ids=[item["sample_id"] for item in selected],
                    )
                    summary["status"] = "COMPLETE"
                    summary["na_count"] = 0
                    metric_scopes[scope] = summary
                    bootstrap = summary.pop("bootstrap")
                    summary["bootstrap_record_id"] = (
                        None if bootstrap is None else summary["metric_id"]
                    )
                    if bootstrap is not None:
                        bootstraps.append(_compact_bootstrap_record(bootstrap))
                summaries.append(metric_scopes[scope])
            winner_doc["metrics"][metric_name] = metric_scopes
        by_winner[config_id] = winner_doc
    sample_registry = (
        None
        if expected_sample_ids is None
        else {key: list(value) for key, value in expected_sample_ids.items()}
    )
    return (
        _document(
            "stage_c_c10_summaries", scopes=["calibration", "holdout", "all_20"],
            calibration_n=5, holdout_n=15, all_20_n=20,
            statistical_unit="image", audit_is_calibration_overlay=True,
            all_20_descriptive_only=True, sample_registry=sample_registry,
            winners=by_winner, core_summary_record_count=len(summaries),
            core_summary_records_digest=canonical_sha256({"records": summaries}),
        ),
        _document(
            "stage_c_c10_bootstrap", resamples=10_000, cluster="image",
            cluster_size=15, statistic="paired_median", base_seed=20260821,
            quantile_method="linear", sample_registry=sample_registry,
            records=bootstraps,
        ),
    )


def _evaluate_validity(
    snapshot: Any, winner_samples: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    failures: list[str] = []
    report = snapshot.report
    if not (
        report.get("status") == "PASS"
        and report.get("primary_holdout_valid_count") == 15
        and isinstance(report.get("primary_holdout_failures"), Sequence)
        and len(report["primary_holdout_failures"]) == 0
        and isinstance(report.get("errors"), Sequence)
        and len(report["errors"]) == 0
    ):
        failures.append("sealed_c9_primary_validity_not_15_of_15")
    primary = [item for item in winner_samples[PRIMARY] if item["role"] == "holdout"]
    if len(primary) != 15 or any(item.get("valid") is not True for item in primary):
        failures.append("primary_holdout_value_evidence_not_15_of_15")
    audit_records = [record for record in snapshot.records if record["audit"]]
    if len(audit_records) != 32:
        failures.append("audit_unit_coverage_not_32")
    audit_checks = []
    for record in audit_records:
        repeats = record["scalars"].get("repeatability")
        ok = (
            isinstance(repeats, Sequence)
            and not isinstance(repeats, (str, bytes)) and len(repeats) == 1
            and repeats[0].get("atol") == 1e-6
            and repeats[0].get("rtol") == 1e-5
            and repeats[0].get("pass") is True
        )
        audit_checks.append({
            "unit_key_sha256": record["unit_key"]["canonical_digest"],
            "pass": ok, "repeatability": _plain(repeats),
        })
        if not ok:
            failures.append("audit_repeatability_failed")
    holdout_valid = (
        report.get("primary_holdout_valid_count", 0)
        if "sealed_c9_primary_validity_not_15_of_15" in failures
        else sum(item.get("valid") is True for item in primary)
    )
    return {
        "status": "PASS_VALIDITY" if not failures else "FAIL_VALIDITY",
        "primary_config": [0, 9], "holdout_expected": 15,
        "holdout_valid": holdout_valid,
        "sealed_hash_and_provenance_valid": not failures,
        "endpoint_parity_operator": "strictly_less_than_1e-6",
        "ig_conservation_operator": "abs_error <= 1e-4 + 0.01*abs(delta)",
        "token_parity_operator": "abs_error <= 1e-6 + 1e-5*abs(r)",
        "misclassified_excluded": False, "display_used": False,
        "audit_repeatability": audit_checks,
        "failures": sorted(set(failures)),
    }


def _evaluate_cost(report: Mapping[str, Any]) -> dict[str, Any]:
    budget = _plain(report["post_c9_budget"])
    checks = budget["projection"]["hard_cap_checks"]
    if len(checks) != 10 or len({item["cap_key"] for item in checks}) != 10:
        raise StageC10IntegrityError("C9 cost evidence does not retain all ten caps")
    stage_d = [item for item in checks if item["cap_key"] in _STAGE_D_CAP_KEYS]
    if len(stage_d) != 3:
        raise StageC10IntegrityError("Stage D cost evidence is incomplete")
    passed = all(item.get("status") == "PASS" for item in stage_d)
    return {
        "status": "PASS_COST" if passed else "FAIL_COST",
        "stage_d_gate_cap_keys": list(_STAGE_D_CAP_KEYS),
        "stage_d_gate_checks": stage_d, "all_ten_frozen_cap_checks": checks,
        "post_c9_budget_canonical_digest": budget["canonical_digest"],
        "reused_c9_evidence_only": True, "stage_d_executed": False,
    }


_SUMMARY_FAMILIES = frozenset({
    "fidelity_curve", "fidelity_aggregate", "top20_image", "stability",
    "raw_structural_stability", "nns_residual", "endpoint_parity",
    "ig_conservation", "matched_contribution", "resources",
    "normalized_entropy", "clean_spatial_structure_similarity",
    "value_path_delta_fidelity", "prediction",
})


def _summary_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["comparison_line"], row["method_id"],
        row["reference_method_id"], row["config_id"], row["target_ref"],
        row["metric_family"], row["metric_id"], row["input_variant"],
        row["perturbation_id"], row["fidelity_scale"],
        row["curve_direction"], row["ranking_kind"],
        row["fraction_index"], row["fraction"], row["statistic"],
        row["view"], row["component"],
    )


def _summary_metric_id(key: tuple[Any, ...]) -> str:
    return "stage-c/v1/" + hashlib.sha256(
        json.dumps(key, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _scope_sample_ids(plan: Any) -> dict[str, list[str]]:
    result = {"calibration": [], "holdout": [], "all_20": []}
    roles: dict[str, str] = {}
    for raw_unit in plan.units:
        unit = raw_unit.to_dict() if hasattr(raw_unit, "to_dict") else dict(raw_unit)
        sample = unit["sample"]
        sample_id, role = sample["sample_id"], sample["role"]
        previous = roles.setdefault(sample_id, role)
        if previous != role:
            raise StageC10IntegrityError("sample role changed within the execution plan")
        if sample_id not in result["all_20"]:
            result[role].append(sample_id)
            result["all_20"].append(sample_id)
    if tuple(map(len, (result["calibration"], result["holdout"], result["all_20"]))) != (5, 15, 20):
        raise StageC10IntegrityError("execution plan sample split is not exact 5/15/20")
    return result


def _core_owns_summary_coordinate(key: tuple[Any, ...]) -> bool:
    method_id, config_id = key[1], key[3]
    family, table_metric_id = key[5], key[6]
    perturbation_id, component = key[8], key[16]
    if config_id not in {f"block_{block:02d}_head_{head:02d}" for block, head in WINNERS}:
        return False
    metric_base = table_metric_id.split("|", 1)[0]
    if family == "value_path_delta_fidelity" and metric_base == "delta_fid":
        return True
    if (
        family == "matched_contribution"
        and method_id == MATCHED_METHOD_ID
        and metric_base in {"patch_support", "positive_patch_evidence_fraction"}
    ):
        return True
    if family == "nns_residual" and method_id == NNS_METHOD_ID:
        if metric_base == "residual_signed_spearman":
            return perturbation_id in STABILITY_PERTURBATIONS
        return (
            metric_base == "residual_rho"
            and perturbation_id == "gaussian_rgb_std_0_01"
            and component == "clean"
        )
    return False


def _compact_bootstrap_record(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _plain(value)
    sample_ids = result.pop("sample_ids", None)
    if sample_ids is None:
        raise StageC10IntegrityError("production bootstrap lacks ordered sample IDs")
    observed = ordered_sample_ids_sha256(sample_ids)
    if result.get("ordered_sample_ids_sha256") != observed:
        raise StageC10IntegrityError("bootstrap sample-order digest is inconsistent")
    result["sample_registry_ref"] = "sample_registry.holdout"
    return result


def _compact_summary_record(
    summary: Mapping[str, Any], selected: Sequence[Mapping[str, Any]],
    dimensions: Mapping[str, Any], statuses: Counter[str],
    na_reasons: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    full = _plain(summary)
    sample_ids = [row["sample_id"] for row in selected]
    observations = [
        {
            "sample_id": row["sample_id"],
            "row_key_sha256": row["row_key_sha256"],
            "status": row["status"], "value": row["value"],
            "na_reason": row["na_reason"],
        }
        for row in selected
    ]
    bootstrap = full.pop("bootstrap", None)
    full.pop("sample_ids", None)
    full.pop("per_sample", None)
    full.pop("values", None)
    full.update({
        "dimensions": _plain(dimensions),
        "n_expected": len(selected),
        "sample_registry_ref": f"sample_registry.{full['scope']}",
        "ordered_sample_ids_sha256": ordered_sample_ids_sha256(sample_ids),
        "ordered_row_keys_sha256": canonical_sha256({
            "row_keys": [row["row_key_sha256"] for row in selected]
        }),
        "ordered_observations_sha256": canonical_sha256({
            "observations": observations
        }),
        "status_counts": dict(sorted(statuses.items())),
        "na_reason_counts": dict(sorted(na_reasons.items())),
        "per_sample_values_ref": "per_sample.csv",
        "bootstrap_record_id": (
            full["metric_id"] if bootstrap is not None else None
        ),
    })
    return full, (
        None if bootstrap is None else _compact_bootstrap_record(bootstrap)
    )


def _summarize_long_table(
    rows: Sequence[Mapping[str, Any]],
    expected_sample_ids: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric_family"] not in _SUMMARY_FAMILIES:
            continue
        if row["metric_family"] == "ig_conservation" and isinstance(row["value"], bool):
            continue
        groups[_summary_key(row)].append(row)
    records: list[dict[str, Any]] = []
    bootstraps: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple("" if value is None else str(value) for value in item)):
        if _core_owns_summary_coordinate(key):
            continue
        group = sorted(groups[key], key=lambda row: (row["plan_index"], row["sample_id"]))
        sample_ids = [row["sample_id"] for row in group]
        if sample_ids != list(expected_sample_ids["all_20"]):
            raise StageC10IntegrityError(
                f"summary coordinate does not match the exact 20-image order: {key}"
            )
        summary_id = _summary_metric_id(key)
        dimensions = {
            name: value for name, value in zip((
                "comparison_line", "method_id", "reference_method_id",
                "config_id", "target_ref", "metric_family", "table_metric_id",
                "input_variant", "perturbation_id", "fidelity_scale",
                "curve_direction", "ranking_kind", "fraction_index",
                "fraction", "statistic", "view", "component",
            ), key)
        }
        for scope in ("calibration", "holdout", "all_20"):
            selected = [
                row for row in group
                if scope == "all_20" or row["sample_role"] == scope
            ]
            expected_n = {"calibration": 5, "holdout": 15, "all_20": 20}[scope]
            if len(selected) != expected_n:
                raise StageC10IntegrityError(
                    f"summary scope {scope} does not contain {expected_n} image rows"
                )
            if [row["sample_id"] for row in selected] != list(expected_sample_ids[scope]):
                raise StageC10IntegrityError(
                    f"summary scope {scope} sample identity/order drifted"
                )
            statuses = Counter(row["status"] for row in selected)
            na_reasons = Counter(
                row["na_reason"] for row in selected if row["na_reason"] is not None
            )
            numeric = all(
                row["status"] == "OK"
                and isinstance(row["value"], (bool, int, float))
                for row in selected
            )
            if numeric:
                full_summary = summarize_values(
                    summary_id, [float(row["value"]) for row in selected],
                    scope=scope, sample_ids=[row["sample_id"] for row in selected],
                )
                full_summary["status"] = "COMPLETE"
            else:
                full_summary = {
                    "metric_id": summary_id, "scope": scope,
                    "unit": "image", "confirmatory": scope == "holdout",
                    "n": expected_n,
                    "sample_ids": [row["sample_id"] for row in selected],
                    "per_sample": [
                        {"sample_index": index, "sample_id": row["sample_id"],
                         "value": row["value"], "status": row["status"],
                         "na_reason": row["na_reason"]}
                        for index, row in enumerate(selected)
                    ],
                    "values": [row["value"] for row in selected],
                    "mean": None, "median": None,
                    "standard_deviation_sample_ddof_1": None,
                    "positive_count": sum(
                        row["status"] == "OK"
                        and isinstance(row["value"], (int, float))
                        and not isinstance(row["value"], bool)
                        and row["value"] > 0
                        for row in selected
                    ),
                    "bootstrap_95_percent_ci": None, "bootstrap": None,
                    "status": "INCOMPLETE_VECTOR",
                }
            summary, bootstrap = _compact_summary_record(
                full_summary, selected, dimensions, statuses, na_reasons
            )
            records.append(summary)
            if bootstrap is not None:
                bootstraps.append(bootstrap)
    return records, bootstraps


def _unit_reconciliation(
    snapshot: Any, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter(row["source_unit_key_sha256"] for row in rows)
    units = []
    for record in snapshot.records:
        digest = record["unit_key"]["canonical_digest"]
        units.append({
            "plan_index": record["plan_index"],
            "unit_key": _plain(record["unit_key"]),
            "state": record["state"], "attempt_path": record["attempt_path"],
            "hashes": _plain(record["hashes"]),
            "emitted_primary_source_row_count": counts[digest],
            "failure": None,
        })
    return _document(
        "stage_c_c10_unit_reconciliation", expected_unit_count=320,
        observed_unit_count=len(units), state_counts={"succeeded": len(units)},
        failure_count=0, units=units,
        row_status_counts=dict(sorted(Counter(row["status"] for row in rows).items())),
        na_reason_counts=dict(sorted(Counter(
            row["na_reason"] for row in rows if row["na_reason"] is not None
        ).items())),
        misclassified_sample_ids=sorted({
            row["sample_id"] for row in rows if row["misclassified"] is True
        }),
        no_rows_dropped=True, zero_imputation_performed=False,
    )


def _build_documents(
    snapshot: Any, config: Any, plan: Any, rows: Sequence[Mapping[str, Any]],
    winner_samples: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    *, code: Mapping[str, Any], c9_root: Path, output_root: Path,
    recovery_authority: Any | None = None,
) -> dict[str, Any]:
    expected_sample_ids = _scope_sample_ids(plan)
    summary_document, core_bootstrap = _summary_document(
        winner_samples, expected_sample_ids
    )
    long_summaries, long_bootstraps = _summarize_long_table(
        rows, expected_sample_ids
    )
    summary_document["long_table_metric_summaries"] = long_summaries
    summary_document["canonical_digest"] = canonical_sha256({
        key: value for key, value in summary_document.items()
        if key != "canonical_digest"
    })
    core_bootstrap["records"].extend(long_bootstraps)
    core_bootstrap["canonical_digest"] = canonical_sha256({
        key: value for key, value in core_bootstrap.items()
        if key != "canonical_digest"
    })

    primary = [item for item in winner_samples[PRIMARY] if item["role"] == "holdout"]
    fine_summary = summary_document["winners"]["block_00_head_09"]["metrics"]["delta_fid_fine4"]["holdout"]
    effect = evaluate_effect(fine_summary)
    stability_summaries = {
        perturbation: summary_document["winners"]["block_00_head_09"]["metrics"][
            f"residual_signed_spearman_{perturbation}"
        ]["holdout"]
        for perturbation in STABILITY_PERTURBATIONS
    }
    spatial = evaluate_spatial(primary, stability_summaries)
    validity = _evaluate_validity(snapshot, winner_samples)
    cost = _evaluate_cost(snapshot.report)
    primary_decision = decide(
        validity["status"], effect["status"], spatial["status"], cost["status"]
    )
    protocol_path = REPO_ROOT / config.protocol.path
    holm = audit_holm_contract(
        config.to_dict(), protocol_path.read_text(encoding="utf-8")
    )
    if holm["status"] != "HOLM_BLOCKED_UNDERSPECIFIED":
        raise StageC10IntegrityError(
            "frozen C10 inputs unexpectedly claim a complete Holm contract; "
            "this implementation will not mark acceptance PASS without a "
            "pre-registered raw-p-value computation path"
        )
    holm = _document("stage_c_c10_holm_report", **holm)
    blockers = ["HOLM_BLOCKED_UNDERSPECIFIED"]
    acceptance = "BLOCKED"
    decision = _document(
        "stage_c_c10_decision", decision=primary_decision,
        primary_config=[0, 9], confirmatory_role="holdout",
        confirmatory_n=15, decision_performed=True, flags={
            "validity": validity, "effect": effect, "spatial": spatial,
            "stability": spatial["stability"], "cost": cost,
        },
        exploratory_configs=[list(item) for item in EXPLORATORY],
        exploratory_affects_primary_decision=False,
        holm_status=holm["status"], stage_c_acceptance_status=acceptance,
        blocking_reasons=blockers, downstream_authorized=False,
        stage_d_started=False, equivalence_test_performed=False,
    )
    reconciliation = _unit_reconciliation(snapshot, rows)
    provenance = _document(
        "stage_c_c10_provenance", code=_plain(code),
        c10_output_root=".", c9_root_binding={
            "kind": "content_addressed_frozen_c9_root",
            "report_file_sha256": snapshot.bindings["report_file_sha256"],
            "root_inventory_digest": snapshot.bindings["root_inventory_digest"],
        },
        c9_bindings=_plain(snapshot.bindings), c9_hash_audit=_plain(snapshot.hash_audit),
        c9_recovery_authority_attestation={
            "producer_commit": snapshot.bindings["producer_commit"],
            "authority_execution_commit": snapshot.bindings[
                "authority_execution_commit"
            ],
            "authority_path": snapshot.bindings["recovery_authority_path"],
            "authority_file_sha256": snapshot.bindings[
                "recovery_authority_file_sha256"
            ],
            "authority_canonical_digest": snapshot.bindings[
                "recovery_authority_canonical_digest"
            ],
            "source_recomputed_for_c10": True,
        },
        c10_launch_authority=(
            None if recovery_authority is None else {
                "path": recovery_authority.path.relative_to(
                    recovery_authority.repo_root
                ).as_posix(),
                "file_sha256": recovery_authority.file_sha256,
                "canonical_digest": recovery_authority.canonical_digest,
                "launch_commit": recovery_authority.launch_commit,
                "producer_commit": recovery_authority.producer_commit,
            }
        ),
        execution_plan_canonical_digest=plan.canonical_digest,
        frozen_input_hashes={
            "config_sha256": sha256_file(REPO_ROOT / "config/stage_c.json"),
            "protocol_sha256": sha256_file(protocol_path),
            "acceptance_sha256": sha256_file(REPO_ROOT / "docs/stage_c_acceptance.md"),
            "selection_sha256": sha256_file(
                REPO_ROOT / "evidence/stage_c/c8_salvage_repair_recovery_v2_de71d55"
                / "selection/stage_c_salvage_repair_selection.json"
            ),
            "run_spec_sha256": sha256_file(REPO_ROOT / "config/stage_c_c9_run_spec.json"),
        },
        source_levels_used=["L0", "L1"], l2_used=False, l3_used=False,
        source_jpeg_read=False, model_inference_performed=False,
        gpu_workload_performed=False, stage_d_started=False,
        bootstrap_implementation={
            "generator": "numpy.random.Generator", "bit_generator": "PCG64",
            "numpy_version": np.__version__, "resamples": 10_000,
            "cluster": "image", "cluster_size": 15,
            "quantile_method": "linear",
        },
    )
    report = _document(
        "stage_c_c10_acceptance_report", status=acceptance, complete=True,
        build_complete=True, integrity_valid=True,
        acceptance_status=acceptance,
        primary_decision=primary_decision, blocking_reasons=blockers,
        c9_status="PASS", c9_execution_units=320,
        c9_primary_holdout_valid=15, per_sample_row_count=len(rows),
        role_counts={"calibration_images": 5, "holdout_images": 15, "all_20_images": 20},
        primary_config=[0, 9], exploratory_configs=[list(item) for item in EXPLORATORY],
        bootstrap_performed=True, holm_performed=False,
        decision_performed=True, downstream_authorized=False, stage_d_started=False,
        scientific_components={
            "validity": validity["status"],
            "effect_fidelity": effect["status"],
            "clean_joint_spatial": spatial["joint_spatial_status"],
            "translation_stability": spatial["stability"][
                "translate_right_down_4"
            ]["status"],
            "center_crop_resize_stability": spatial["stability"][
                "center_crop_216_resize_224"
            ]["status"],
            "gaussian_stability": spatial["stability"][
                "gaussian_rgb_std_0_01"
            ]["status"],
            "overall_spatial": spatial["status"],
            "cost": cost["status"],
            "holm": holm["status"],
        },
        claims={
            "equivalence_claimed": False, "universal_imagenet_superiority_claimed": False,
            "upsampled_map_claimed_as_native_pixel_evidence": False,
            "value_path_claimed_as_class_or_complete_attention_explanation": False,
            "raw_attention_correlation_used_as_novelty_gate": False,
        },
    )
    return {
        "summaries.json": summary_document, "bootstrap.json": core_bootstrap,
        "holm.json": holm, "reconciliation.json": reconciliation,
        "decision.json": decision, "stage_c_report.json": report,
        "provenance.json": provenance,
    }


def _safe_fresh_root(value: Path | str) -> Path:
    root = Path(os.path.abspath(os.fspath(value)))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise StageC10IntegrityError(f"output path contains symlink: {current}")
    if os.path.lexists(root):
        raise StageC10BlockedError("C10 output root must not already exist")
    if not root.parent.is_dir():
        raise StageC10BlockedError("C10 output parent must already exist")
    return root


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_read_regular_bytes(path: Path, context: str) -> bytes:
    """Read one stable, regular, single-link output file via one fd."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        path_before = path.lstat()
        descriptor = os.open(path, flags)
        fd_before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(path_before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)
            or not stat.S_ISREG(fd_before.st_mode)
            or path_before.st_nlink != 1
            or fd_before.st_nlink != 1
            or (path_before.st_dev, path_before.st_ino)
            != (fd_before.st_dev, fd_before.st_ino)
        ):
            raise ValueError("not the same regular single-link file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            payload = handle.read()
            fd_after = os.fstat(handle.fileno())
        path_after = path.lstat()
        identity_before = (
            fd_before.st_dev, fd_before.st_ino, fd_before.st_size,
            fd_before.st_mtime_ns, fd_before.st_ctime_ns,
        )
        identity_after = (
            fd_after.st_dev, fd_after.st_ino, fd_after.st_size,
            fd_after.st_mtime_ns, fd_after.st_ctime_ns,
        )
        path_identity_after = (
            path_after.st_dev, path_after.st_ino, path_after.st_size,
            path_after.st_mtime_ns, path_after.st_ctime_ns,
        )
        if (
            identity_before != identity_after
            or identity_before != path_identity_after
            or fd_after.st_nlink != 1
            or path_after.st_nlink != 1
            or not stat.S_ISREG(path_after.st_mode)
            or len(payload) != fd_before.st_size
        ):
            raise ValueError("file changed while being read")
        return payload
    except (OSError, ValueError) as error:
        raise StageC10IntegrityError(
            f"{context} is missing, unsafe, or unstable: {path}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_directory_chain(path: Path) -> int:
    """Open every component of one absolute directory without symlinks."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _safe_read_output_directory(
    root: Path, names: set[str],
) -> dict[str, bytes]:
    """Read a fixed child set through one stable no-follow directory fd."""

    root_fd: int | None = None
    verification_fd: int | None = None
    try:
        root_fd = _open_directory_chain(root)
        before = os.fstat(root_fd)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("output root is not a directory")
        observed_names = set(os.listdir(root_fd))
        if observed_names != names:
            raise ValueError(
                f"file set drifted; missing={sorted(names-observed_names)}, "
                f"extra={sorted(observed_names-names)}"
            )
        payloads: dict[str, bytes] = {}
        child_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for name in sorted(names):
            if not name or name in {".", ".."} or "/" in name:
                raise ValueError("invalid output child name")
            child_fd = os.open(name, child_flags, dir_fd=root_fd)
            try:
                child_before = os.fstat(child_fd)
                if (
                    not stat.S_ISREG(child_before.st_mode)
                    or child_before.st_nlink != 1
                ):
                    raise ValueError(f"unsafe output child: {name}")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(child_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                child_after = os.fstat(child_fd)
                identity_before = (
                    child_before.st_dev, child_before.st_ino,
                    child_before.st_size, child_before.st_mtime_ns,
                    child_before.st_ctime_ns,
                )
                identity_after = (
                    child_after.st_dev, child_after.st_ino,
                    child_after.st_size, child_after.st_mtime_ns,
                    child_after.st_ctime_ns,
                )
                payload = b"".join(chunks)
                if (
                    identity_before != identity_after
                    or child_after.st_nlink != 1
                    or len(payload) != child_before.st_size
                ):
                    raise ValueError(f"unstable output child: {name}")
                payloads[name] = payload
            finally:
                os.close(child_fd)
        after = os.fstat(root_fd)
        verification_fd = _open_directory_chain(root)
        verification = os.fstat(verification_fd)
        if (
            (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns)
            or (before.st_dev, before.st_ino)
            != (verification.st_dev, verification.st_ino)
            or set(os.listdir(root_fd)) != observed_names
        ):
            raise ValueError("output directory changed while being read")
        return payloads
    except (OSError, ValueError) as error:
        raise StageC10IntegrityError(
            f"C10 output directory is missing, unsafe, or unstable: {root}"
        ) from error
    finally:
        if root_fd is not None:
            os.close(root_fd)
        if verification_fd is not None:
            os.close(verification_fd)


def _resolve_c10_producer_attestation(
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Resolve the external, local archive ref that anchors C10 code."""

    try:
        ref_commit = subprocess.run(
            ["git", "rev-parse", "--verify", f"{C10_PRODUCER_ARCHIVE_REF}^{{commit}}"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit_tree = subprocess.run(
            ["git", "rev-parse", f"{ref_commit}^{{tree}}"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        prefix = subprocess.run(
            ["git", "rev-parse", "--show-prefix"], cwd=REPO_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        source_path = f"{prefix}{C10_CODE_SURFACE}"
        source_tree = subprocess.run(
            ["git", "rev-parse", f"{ref_commit}:{source_path}"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        live_source_subtree = subprocess.run(
            ["git", "rev-parse", f"HEAD:{source_path}"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        source_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", C10_CODE_SURFACE],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout
        ref_commit_after = subprocess.run(
            ["git", "rev-parse", "--verify", f"{C10_PRODUCER_ARCHIVE_REF}^{{commit}}"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise StageC10IntegrityError(
            "C10 producer archive attestation cannot be resolved"
        ) from error
    if not all(_GIT_OID.fullmatch(value) for value in (
        ref_commit, ref_commit_after, commit_tree, source_tree, live_source_subtree,
    )):
        raise StageC10IntegrityError("C10 producer archive returned malformed Git IDs")
    if ref_commit_after != ref_commit:
        raise StageC10IntegrityError("C10 producer archive ref changed during attestation")
    if expected_commit is not None and ref_commit != expected_commit:
        raise StageC10IntegrityError(
            "C10 build HEAD differs from the frozen producer archive ref"
        )
    if source_status.strip() or live_source_subtree != source_tree:
        raise StageC10IntegrityError(
            "live C10 source surface differs from the frozen producer archive"
        )
    return {
        "commit": ref_commit, "dirty": False, "tree": commit_tree,
        "archive_ref": C10_PRODUCER_ARCHIVE_REF,
        "source_surface": C10_CODE_SURFACE,
        "source_tree": source_tree,
    }


def _output_manifest(root: Path, names: Sequence[str]) -> dict[str, Any]:
    payloads = _safe_read_output_directory(root, set(names))
    return _output_manifest_from_payloads(payloads, names)


def _output_manifest_from_payloads(
    payloads: Mapping[str, bytes], names: Sequence[str],
) -> dict[str, Any]:
    if set(payloads) != set(names):
        raise StageC10IntegrityError("C10 manifest payload set drifted")
    artifacts = []
    for name in names:
        payload = payloads[name]
        if not isinstance(payload, bytes):
            raise StageC10IntegrityError("C10 manifest payload must be bytes")
        artifacts.append({
            "path": name, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    return _document(
        "stage_c_c10_output_manifest", artifacts=artifacts,
        artifact_count=len(artifacts), immutable=True,
    )


def _status_document(report: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return _document(
        "stage_c_c10_status", complete=True, terminal=True,
        status=report["status"], stage_c_acceptance_status=report["status"],
        primary_decision=report["primary_decision"],
        manifest_sha256=manifest_sha256, payload_count=len(OUTPUT_FILES),
        downstream_authorized=False, stage_d_started=False,
    )


def _assemble_c10(
    c9_root: Path | str, output_root: Path | str, plan: Any, config: Any,
    *, code: Mapping[str, Any], recovery_authority: Any | None = None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    from .stage_c_c10_source import load_c9_snapshot
    from .stage_c_c9 import validate_c9_execution_plan

    c9_path = Path(os.path.abspath(os.fspath(c9_root)))
    output_path = Path(os.path.abspath(os.fspath(output_root)))
    _assert_frozen_c10_config(config)
    scope = validate_c9_execution_plan(plan, config)
    if scope.get("valid") is not True:
        raise StageC10IntegrityError(
            "C10 requires the exact C9 plan: "
            + "; ".join(scope.get("violations", ()))
        )
    snapshot = load_c9_snapshot(c9_path, plan, config)
    base_rows = build_c10_long_table(snapshot.records, plan.units)
    l1_rows = _l1_derived_rows(snapshot.records, plan.units)
    derived_rows, winner_samples = _derived_scientific_rows(
        snapshot.records, plan.units
    )
    rows = validate_rows([*base_rows, *l1_rows, *derived_rows], sort=True)
    csv_bytes = rows_to_csv_bytes(rows)
    row_manifest = build_row_manifest(rows, csv_bytes=csv_bytes)
    row_manifest["base_l0_row_count"] = len(base_rows)
    row_manifest["derived_l1_row_count"] = len(l1_rows)
    row_manifest["derived_scientific_row_count"] = len(derived_rows)
    expected_counts = {
        "base_l0": expected_base_row_count(plan.units),
        "derived_l1": _expected_l1_derived_row_count(snapshot.records),
        "derived_scientific": _expected_scientific_derived_row_count(plan.units),
    }
    expected_coordinates = chain(
        expected_base_row_coordinates(plan.units),
        _expected_l1_row_coordinates(snapshot.records),
        _expected_scientific_row_coordinates(plan.units),
    )
    row_manifest["expected_row_counts_from_applicability"] = expected_counts
    expected_sample_ids = _scope_sample_ids(plan)
    row_manifest["metric_coverage_audit"] = _metric_coverage_audit(
        rows, config, expected_sample_ids, expected_coordinates
    )
    if (
        row_manifest["metric_coverage_audit"]["exact_applicability"]
        ["coordinate_count"] != sum(expected_counts.values())
    ):
        raise StageC10IntegrityError(
            "C10 applicability coordinate and row-count registries disagree"
        )
    row_manifest["canonical_digest"] = canonical_sha256({
        key: value for key, value in row_manifest.items()
        if key != "canonical_digest"
    })
    if (
        row_manifest["base_l0_row_count"] != expected_counts["base_l0"]
        or row_manifest["derived_l1_row_count"] != expected_counts["derived_l1"]
        or row_manifest["derived_scientific_row_count"]
        != expected_counts["derived_scientific"]
        or row_manifest["row_count"] != sum(expected_counts.values())
    ):
        raise StageC10IntegrityError("C10 deterministic row-space reconciliation failed")
    documents = _build_documents(
        snapshot, config, plan, rows, winner_samples, code=code,
        c9_root=c9_path, output_root=output_path,
        recovery_authority=recovery_authority,
    )
    payloads: dict[str, bytes] = {
        "per_sample.csv": csv_bytes,
        "per_sample.schema.json": _json_bytes(schema_document()),
        "per_sample_manifest.json": _json_bytes(row_manifest),
        **{name: _json_bytes(document) for name, document in documents.items()},
    }
    return payloads, documents["stage_c_report.json"]


def build_c10_output_root(
    c9_root: Path | str, output_root: Path | str, plan: Any, config: Any, *,
    recovery_authority: Any | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and seal a fresh C10 root from immutable C9 evidence."""

    live = capture_git_state(REPO_ROOT)
    if observed_git_state is not None and dict(observed_git_state) != live:
        raise StageC10IntegrityError("supplied C10 Git state differs from live state")
    if live.get("dirty") is not False:
        raise StageC10BlockedError("formal C10 build requires a clean Git worktree")
    from .recovery_c10_launch import RecoveryC10LaunchRecord
    if type(recovery_authority) is not RecoveryC10LaunchRecord:
        raise StageC10BlockedError("validated recovery C10 authority is required")
    if Path(c9_root).resolve() != recovery_authority.c9_record.output_root:
        raise StageC10IntegrityError("C10 authority C9 root binding drifted")
    if Path(output_root).resolve() != recovery_authority.output_root:
        raise StageC10IntegrityError("C10 authority output-root binding drifted")
    root = _safe_fresh_root(output_root)
    code = _resolve_c10_producer_attestation(recovery_authority.producer_commit)
    payloads, report = _assemble_c10(
        c9_root, root, plan, config, code=code,
        recovery_authority=recovery_authority,
    )
    if capture_git_state(REPO_ROOT) != live:
        raise StageC10IntegrityError(
            "C10 Git state changed during assembly; no output root was created"
        )
    root.mkdir(mode=0o755, parents=False, exist_ok=False)
    for name in OUTPUT_FILES:
        _write_new(root / name, payloads[name])
    manifest = _output_manifest_from_payloads(payloads, OUTPUT_FILES)
    _write_new(root / MANIFEST_FILENAME, _json_bytes(manifest))
    status = _status_document(
        report, sha256_file(root / MANIFEST_FILENAME),
    )
    _write_new(root / STATUS_FILENAME, _json_bytes(status))
    del payloads
    if capture_git_state(REPO_ROOT) != live:
        raise StageC10IntegrityError(
            "C10 Git state changed while the output root was being sealed"
        )
    validated = validate_c10_output_root(
        c9_root, root, plan, config, recovery_authority=recovery_authority
    )
    if capture_git_state(REPO_ROOT) != live:
        raise StageC10IntegrityError(
            "C10 Git state changed during post-build validation"
        )
    return validated


def _safe_existing_output_root(value: Path | str) -> Path:
    root = Path(os.path.abspath(os.fspath(value)))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as error:
            raise StageC10IntegrityError(f"C10 output component missing: {current}") from error
        if stat.S_ISLNK(info.st_mode):
            raise StageC10IntegrityError(f"C10 output contains symlink: {current}")
    if not root.is_dir():
        raise StageC10IntegrityError("C10 output root is not a directory")
    return root


def validate_c10_output_root(
    c9_root: Path | str, output_root: Path | str, plan: Any, config: Any,
    *, recovery_authority: Any | None = None,
) -> dict[str, Any]:
    """Read-only, source-recomputing validation of a sealed C10 root."""

    root = _safe_existing_output_root(output_root)
    from .recovery_c10_launch import RecoveryC10LaunchRecord
    if type(recovery_authority) is not RecoveryC10LaunchRecord:
        raise StageC10IntegrityError("validated recovery C10 authority is required")
    if Path(c9_root).resolve() != recovery_authority.c9_record.output_root:
        raise StageC10IntegrityError("C10 authority C9 root binding drifted")
    if root != recovery_authority.output_root:
        raise StageC10IntegrityError("C10 authority output-root binding drifted")
    expected_names = {*OUTPUT_FILES, MANIFEST_FILENAME, STATUS_FILENAME}
    actual_payloads = _safe_read_output_directory(root, expected_names)
    manifest = _strict_json_bytes(actual_payloads[MANIFEST_FILENAME], MANIFEST_FILENAME)
    status = _strict_json_bytes(actual_payloads[STATUS_FILENAME], STATUS_FILENAME)
    expected_inventory = [
        {"path": name, "bytes": len(actual_payloads[name]),
         "sha256": hashlib.sha256(actual_payloads[name]).hexdigest()}
        for name in OUTPUT_FILES
    ]
    expected_manifest = _document(
        "stage_c_c10_output_manifest", artifacts=expected_inventory,
        artifact_count=len(expected_inventory), immutable=True,
    )
    if manifest != expected_manifest:
        raise StageC10IntegrityError("C10 payload inventory/hash drifted")
    actual_report = _strict_json_bytes(
        actual_payloads["stage_c_report.json"], "stage_c_report.json"
    )
    expected_status = _status_document(
        actual_report, hashlib.sha256(actual_payloads[MANIFEST_FILENAME]).hexdigest(),
    )
    if status != expected_status:
        raise StageC10IntegrityError("C10 status/manifest binding drifted")
    read_csv_bytes(actual_payloads["per_sample.csv"], require_sorted=True)
    for name in OUTPUT_FILES:
        if name.endswith(".json"):
            _strict_json_bytes(actual_payloads[name], name)

    code = _strict_json_bytes(
        actual_payloads["provenance.json"], "provenance.json"
    )["code"]
    if not isinstance(code, Mapping):
        raise StageC10IntegrityError("C10 producer code binding is invalid")
    trusted_code = _resolve_c10_producer_attestation()
    if dict(code) != trusted_code:
        raise StageC10IntegrityError(
            "C10 producer code does not match the independent archive attestation"
        )
    recomputed, report = _assemble_c10(
        c9_root, root, plan, config, code=trusted_code,
        recovery_authority=recovery_authority,
    )
    if recomputed != {name: actual_payloads[name] for name in OUTPUT_FILES}:
        raise StageC10IntegrityError("C10 output differs from source recomputation")
    if status.get("status") != report["status"] or status.get("primary_decision") != report["primary_decision"]:
        raise StageC10IntegrityError("C10 status differs from acceptance report")
    return _strict_json_bytes(actual_payloads["stage_c_report.json"], "stage_c_report.json")


def _assert_frozen_c10_config(config: Any) -> None:
    from .stage_c_config import DEFAULT_STAGE_C_CONFIG, DEFAULT_STAGE_C_SCHEMA, strict_json_load
    from .stage_c_c9 import FROZEN_INPUT_SHA256
    from .stage_c_plan import (
        DEFAULT_ACCEPTANCE, DEFAULT_DEPENDENCY_LOCK, DEFAULT_REGISTRY_SCHEMA,
        DEFAULT_SELECTION_SCHEMA,
    )

    if config.to_dict() != strict_json_load(DEFAULT_STAGE_C_CONFIG):
        raise StageC10IntegrityError("C10 config differs from frozen Stage C config")
    if sha256_file(Path(DEFAULT_STAGE_C_CONFIG)) != FROZEN_INPUT_SHA256["config_sha256"]:
        raise StageC10IntegrityError("frozen Stage C config byte hash drifted")
    frozen_paths = {
        "protocol_sha256": REPO_ROOT / config.protocol.path,
        "acceptance_sha256": Path(DEFAULT_ACCEPTANCE),
        "config_schema_sha256": Path(DEFAULT_STAGE_C_SCHEMA),
        "registry_schema_sha256": Path(DEFAULT_REGISTRY_SCHEMA),
        "selection_schema_sha256": Path(DEFAULT_SELECTION_SCHEMA),
        "dependency_lock_sha256": Path(DEFAULT_DEPENDENCY_LOCK),
        "registry_sha256": REPO_ROOT / config.registry.path,
        "registry_metadata_sha256": REPO_ROOT / config.registry.metadata_path,
        "stage_a_manifest_sha256": REPO_ROOT / config.registry.source_manifest_path,
        "selection_file_sha256": (
            REPO_ROOT / "evidence/stage_c/c8_salvage_repair_recovery_v2_de71d55"
            / "selection/stage_c_salvage_repair_selection.json"
        ),
    }
    expected = {
        **FROZEN_INPUT_SHA256,
        "selection_file_sha256": "5d26c78bb94278c7f12d7bd88b1a37cfb97352f3d0bde5e1c4e52f8e811c7982",
    }
    drifted = [
        name for name, path in frozen_paths.items()
        if sha256_file(path) != expected[name]
    ]
    if drifted:
        raise StageC10IntegrityError(
            "C10 frozen input hash drifted: " + ", ".join(sorted(drifted))
        )
    required = list(config.metrics["required"])
    if set(required) != set(_REQUIRED_METRIC_COVERAGE) or len(required) != len(set(required)):
        raise StageC10IntegrityError(
            "C10 metric applicability registry does not exactly cover metrics.required"
        )


__all__ = [
    "build_c10_output_root", "validate_c10_output_root",
    "StageC10BlockedError", "StageC10Error", "StageC10IntegrityError",
    "partition_rows_by_scope", "reconcile_expected_rows",
    "validate_metric_source_path",
]
