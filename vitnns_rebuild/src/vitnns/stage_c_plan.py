"""Strict Stage C registry, selection, run-plan, and preflight contracts.

This module deliberately stops before model execution and artifact writing.  It
turns the frozen Stage C configuration into immutable, auditable execution
units and a conservative resource estimate.  In particular, a missing C8
selection is represented explicitly; it is never replaced by an invented
head.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .constants import MODEL_SHA256, REPO_ROOT, WORKSPACE_ROOT
from .io_utils import sha256_file
from .recovery_v2 import RecoveryV2Binding, load_recovery_v2_binding
from .stage_c_config import (
    DEFAULT_STAGE_C_CONFIG,
    DEFAULT_STAGE_C_SCHEMA,
    StageCConfig,
    strict_json_load,
)


DEFAULT_REGISTRY_SCHEMA = REPO_ROOT / "config/stage_c_registry.schema.json"
DEFAULT_SELECTION_SCHEMA = REPO_ROOT / "config/stage_c_selection.schema.json"
DEFAULT_ACCEPTANCE = REPO_ROOT / "docs/stage_c_acceptance.md"
DEFAULT_DEPENDENCY_LOCK = REPO_ROOT / "requirements.lock.txt"
FROZEN_REGISTRY_SCHEMA_SHA256 = (
    "4ce0ac8cd5bff93ae2b323c3e0447a87fc825c3243ad46c54c9b48e05e9b2d6d"
)
FROZEN_SELECTION_SCHEMA_SHA256 = (
    "867a3cf22519b2a045d81751a20c5cd52570ddeadede26a2da2bb79973f0b59d"
)

REGISTRY_COLUMNS = (
    "schema_version", "sample_id", "image_id", "relative_path",
    "image_sha256", "size_bytes", "width", "height",
    "official_wnid", "imagenet1k_index", "label_source", "role",
    "audit", "selection_order", "source_selection_index",
    "selection_rule",
)
SOURCE_MANIFEST_COLUMNS = (
    "sample_id", "relative_path", "image_sha256", "size_bytes",
    "width", "height", "wnid", "imagenet1k_index", "label_source",
)
SELECTION_RULE = (
    "legacy_sorted_500_linspace20_then_fixed_role_positions_0_4_9_14_19_"
    "audit_0_19"
)
CALIBRATION_SAMPLE_IDS = (
    "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00009888.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00023229.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00035680.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00049970.JPEG",
)
AUDIT_SAMPLE_IDS = (
    "ILSVRC2012_val/ILSVRC2012_val_00000079.JPEG",
    "ILSVRC2012_val/ILSVRC2012_val_00049970.JPEG",
)

VALUE_CONFIG_SOURCES = (
    "explicit",
    "selection_primary",
    "selection_band_winners",
    "calibration_all_144",
    "none",
)
AUDIT_REPEATABILITY_SCOPES = ("clean_map_only",)
SELECTION_STATUSES = ("not_applicable_pre_c8", "present")
MAP_KIND = "map"
CURVE_POINT_KIND = "fidelity_curve_point"
TARGET_RANKING = "target_oriented_positive_tie_permuted"
RANDOM_RANKING = "global_random"
CENTER_RANKING = "center_distance_then_row_major"
FIDELITY_SCALES = ("patch_16x16", "fine_4x4_cell")
CURVE_DIRECTIONS = ("deletion", "insertion")
VALUE_METHOD_IDS = (
    "direct_raw_cls_attention_per_head",
    "conditional_fixed_clean_attention_head_direction_value_path_ig",
    "matched_fixed_clean_attention_patch_token_contribution",
)
CLASS_METHOD_IDS = (
    "direct_raw_cls_attention_last_block_head_mean",
    "attention_rollout_mean_head_residual_all_blocks",
    "vit_grad_cam_block11_norm1",
    "full_model_class_logit_ig",
)
_METHOD_CONTRACTS = {
    VALUE_METHOD_IDS[0]: ("value_path", None, False, True),
    VALUE_METHOD_IDS[1]: (
        "value_path", "fixed_clean_attention_cls_head_direction", True, False
    ),
    VALUE_METHOD_IDS[2]: (
        "value_path", "fixed_clean_attention_cls_head_direction", True, False
    ),
    CLASS_METHOD_IDS[0]: (
        "class_logit", "clean_predicted_class_raw_logit", True, False
    ),
    CLASS_METHOD_IDS[1]: (
        "class_logit", "clean_predicted_class_raw_logit", True, False
    ),
    CLASS_METHOD_IDS[2]: (
        "class_logit", "clean_predicted_class_raw_logit", True, False
    ),
    CLASS_METHOD_IDS[3]: (
        "class_logit", "clean_predicted_class_raw_logit", True, False
    ),
}
_MAP_VARIANTS = (
    "clean", "translate_right_down_4", "center_crop_216_resize_224",
    "gaussian_rgb_std_0_01", "repeat_clean",
)
_OVERRIDABLE_HARD_CAPS = frozenset({
    "bytes_per_l0_l1_execution_unit_max",
    "bytes_per_audit_sample_l2_max",
    "stage_c_c8_c9_total_artifact_bytes_max",
    "stage_d_500_total_artifact_bytes_max",
    "forward_calls_per_execution_unit_max",
    "backward_calls_per_execution_unit_max",
    "cuda_peak_reserved_bytes_per_execution_unit_max",
    "stage_c_c8_c9_h20_gpu_hours_max",
    "stage_d_500_h20_gpu_hours_max",
    "stage_d_h20_gpu_hours_per_image_max",
})
_ESTIMATE_ISSUER = object()

_SAMPLE_RE = re.compile(r"^ILSVRC2012_val/ILSVRC2012_val_[0-9]{8}\.JPEG$")
_IMAGE_RE = re.compile(r"^ILSVRC2012_val_[0-9]{8}$")
_RELATIVE_IMAGE_RE = re.compile(r"^ILSVRC2012_val_[0-9]{8}\.JPEG$")
_WNID_RE = re.compile(r"^n[0-9]{8}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CANDIDATE_KEY_RE = re.compile(r"^block_(0[0-9]|1[01])_head_(0[0-9]|1[01])$")


class StageCPlanError(ValueError):
    """Base error for Stage C planning contracts."""


class RegistryContractError(StageCPlanError):
    """Raised when the registry or one of its bound assets is invalid."""


class SelectionContractError(StageCPlanError):
    """Raised when a C8 selection is absent where required or is invalid."""


class RunSpecError(StageCPlanError):
    """Raised when an explicit run request is incomplete or contradictory."""


class BudgetExceededError(StageCPlanError):
    """Raised by :func:`enforce_stage_c_budget` before execution/writes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact_keys(value: Any, expected: Iterable[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageCPlanError(f"{context} must be an object")
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise StageCPlanError(
            f"{context} fields differ: missing={sorted(expected_set-actual)}, "
            f"unknown={sorted(actual-expected_set)}"
        )
    return value


def _require_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise StageCPlanError(f"{context} must be a boolean")
    return value


def _require_int(
    value: Any, context: str, *, minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise StageCPlanError(f"{context} must be an integer (not a boolean)")
    if minimum is not None and value < minimum:
        raise StageCPlanError(f"{context} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise StageCPlanError(f"{context} must be <= {maximum}")
    return value


def _require_string(value: Any, context: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise StageCPlanError(f"{context} must be a non-empty string")
    return value


def _require_sha256(value: Any, context: str) -> str:
    text = _require_string(value, context)
    if _SHA256_RE.fullmatch(text) is None:
        raise StageCPlanError(f"{context} must be a lowercase SHA-256")
    return text


def _safe_relative(raw: str, context: str) -> PurePosixPath:
    text = _require_string(raw, context)
    if "\\" in text:
        raise StageCPlanError(f"{context} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != text:
        raise StageCPlanError(f"{context} must be normalized and relative")
    return path


def _resolved_under(base: Path, raw: str, context: str) -> Path:
    relative = _safe_relative(raw, context)
    base_resolved = Path(base).resolve()
    target = (base_resolved / Path(*relative.parts)).resolve()
    if not target.is_relative_to(base_resolved):
        raise StageCPlanError(f"{context} escapes its base directory")
    return target


def _strict_csv_int(
    raw: Any, context: str, *, minimum: int, maximum: int | None = None
) -> int:
    if not isinstance(raw, str) or re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
        raise RegistryContractError(f"{context} must be a canonical unsigned integer")
    value = int(raw)
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise RegistryContractError(f"{context} must be in {minimum}{suffix}")
    return value


def _strict_csv_bool(raw: Any, context: str) -> bool:
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise RegistryContractError(f"{context} must be exactly 'true' or 'false'")


def _read_csv_exact(path: Path, columns: Sequence[str], context: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} not found: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(columns):
                raise RegistryContractError(
                    f"{context} header must equal the frozen ordered columns"
                )
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise RegistryContractError(f"{context} must be UTF-8") from exc
    for index, row in enumerate(rows, 2):
        if None in row or set(row) != set(columns) or any(value is None for value in row.values()):
            raise RegistryContractError(f"{context} row {index} has missing or extra fields")
    return rows


@dataclass(frozen=True, slots=True)
class RegistryRow:
    schema_version: int
    sample_id: str
    image_id: str
    relative_path: str
    image_sha256: str
    size_bytes: int
    width: int
    height: int
    official_wnid: str
    imagenet1k_index: int
    label_source: str
    role: str
    audit: bool
    selection_order: int
    source_selection_index: int
    selection_rule: str

    def __post_init__(self) -> None:
        self._validate("registry row")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in REGISTRY_COLUMNS}

    @classmethod
    def from_csv(cls, row: Mapping[str, str], *, row_number: int) -> "RegistryRow":
        context = f"registry row {row_number}"
        if set(row) != set(REGISTRY_COLUMNS):
            raise RegistryContractError(f"{context} has the wrong fields")
        typed = cls(
            schema_version=_strict_csv_int(row["schema_version"], f"{context}.schema_version", minimum=1, maximum=1),
            sample_id=row["sample_id"], image_id=row["image_id"],
            relative_path=row["relative_path"], image_sha256=row["image_sha256"],
            size_bytes=_strict_csv_int(row["size_bytes"], f"{context}.size_bytes", minimum=1),
            width=_strict_csv_int(row["width"], f"{context}.width", minimum=1),
            height=_strict_csv_int(row["height"], f"{context}.height", minimum=1),
            official_wnid=row["official_wnid"],
            imagenet1k_index=_strict_csv_int(row["imagenet1k_index"], f"{context}.imagenet1k_index", minimum=0, maximum=999),
            label_source=row["label_source"], role=row["role"],
            audit=_strict_csv_bool(row["audit"], f"{context}.audit"),
            selection_order=_strict_csv_int(row["selection_order"], f"{context}.selection_order", minimum=0, maximum=19),
            source_selection_index=_strict_csv_int(row["source_selection_index"], f"{context}.source_selection_index", minimum=0, maximum=499),
            selection_rule=row["selection_rule"],
        )
        typed._validate(context)
        return typed

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryRow":
        _exact_keys(value, REGISTRY_COLUMNS, "registry row")
        row = cls(**dict(value))
        row._validate("registry row")
        return row

    def _validate(self, context: str) -> None:
        _require_int(self.schema_version, f"{context}.schema_version", minimum=1, maximum=1)
        for value, pattern, name in (
            (self.sample_id, _SAMPLE_RE, "sample_id"),
            (self.image_id, _IMAGE_RE, "image_id"),
            (self.relative_path, _RELATIVE_IMAGE_RE, "relative_path"),
            (self.image_sha256, _SHA256_RE, "image_sha256"),
            (self.official_wnid, _WNID_RE, "official_wnid"),
        ):
            if not isinstance(value, str) or pattern.fullmatch(value) is None:
                raise RegistryContractError(f"{context}.{name} has invalid syntax")
        for value, name, low, high in (
            (self.size_bytes, "size_bytes", 1, None),
            (self.width, "width", 1, None), (self.height, "height", 1, None),
            (self.imagenet1k_index, "imagenet1k_index", 0, 999),
            (self.selection_order, "selection_order", 0, 19),
            (self.source_selection_index, "source_selection_index", 0, 499),
        ):
            _require_int(value, f"{context}.{name}", minimum=low, maximum=high)
        _require_bool(self.audit, f"{context}.audit")
        if self.label_source != "official_devkit_val_gt":
            raise RegistryContractError(f"{context}.label_source is not frozen")
        if self.role not in ("calibration", "holdout"):
            raise RegistryContractError(f"{context}.role is invalid")
        if self.selection_rule != SELECTION_RULE:
            raise RegistryContractError(f"{context}.selection_rule is not frozen")
        if self.relative_path != self.image_id + ".JPEG":
            raise RegistryContractError(f"{context} image_id/relative_path mismatch")
        if self.sample_id != "ILSVRC2012_val/" + self.relative_path:
            raise RegistryContractError(f"{context} sample_id/relative_path mismatch")


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    schema_version: int
    document_type: str
    path: str
    sha256: str
    metadata_path: str
    metadata_sha256: str
    registry_schema_sha256: str
    source_manifest_sha256: str
    source_manifest_metadata_sha256: str
    data_root: str
    live_assets_verified: bool
    rows: tuple[RegistryRow, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.document_type != "stage_c_registry_snapshot":
            raise RegistryContractError("registry snapshot identity is invalid")
        _safe_relative(self.path, "registry snapshot.path")
        _safe_relative(self.metadata_path, "registry snapshot.metadata_path")
        for value, name in (
            (self.sha256, "sha256"),
            (self.metadata_sha256, "metadata_sha256"),
            (self.registry_schema_sha256, "registry_schema_sha256"),
            (self.source_manifest_sha256, "source_manifest_sha256"),
            (self.source_manifest_metadata_sha256, "source_manifest_metadata_sha256"),
        ):
            _require_sha256(value, f"registry snapshot.{name}")
        _require_string(self.data_root, "registry snapshot.data_root")
        _require_bool(self.live_assets_verified, "registry snapshot.live_assets_verified")
        if not isinstance(self.rows, tuple):
            raise RegistryContractError(
                "registry snapshot.rows must be an immutable tuple"
            )
        if len(self.rows) != 20:
            raise RegistryContractError("registry snapshot must retain all 20 rows")
        if any(not isinstance(row, RegistryRow) for row in self.rows):
            raise RegistryContractError("registry snapshot rows must be RegistryRow values")
        if [row.selection_order for row in self.rows] != list(range(20)):
            raise RegistryContractError(
                "registry snapshot rows must be ordered by selection_order"
            )
        for name in ("sample_id", "image_id", "relative_path", "image_sha256"):
            values = [getattr(row, name) for row in self.rows]
            if len(values) != len(set(values)):
                raise RegistryContractError(
                    f"registry snapshot rows have duplicate {name}"
                )
        if Counter(row.role for row in self.rows) != Counter(
            {"calibration": 5, "holdout": 15}
        ):
            raise RegistryContractError("registry snapshot role counts are invalid")
        if tuple(row.sample_id for row in self.rows if row.audit) != AUDIT_SAMPLE_IDS:
            raise RegistryContractError("registry snapshot audit rows are invalid")
        if tuple(
            row.sample_id for row in self.rows if row.role == "calibration"
        ) != CALIBRATION_SAMPLE_IDS:
            raise RegistryContractError("registry snapshot calibration rows are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "path": self.path, "sha256": self.sha256,
            "metadata_path": self.metadata_path,
            "metadata_sha256": self.metadata_sha256,
            "registry_schema_sha256": self.registry_schema_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_manifest_metadata_sha256": self.source_manifest_metadata_sha256,
            "data_root": self.data_root,
            "live_assets_verified": self.live_assets_verified,
            "rows": [row.to_dict() for row in self.rows],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistrySnapshot":
        keys = {
            "schema_version", "document_type", "path", "sha256",
            "metadata_path", "metadata_sha256", "registry_schema_sha256",
            "source_manifest_sha256", "source_manifest_metadata_sha256",
            "data_root", "live_assets_verified", "rows",
        }
        _exact_keys(value, keys, "registry snapshot")
        if not isinstance(value["rows"], list):
            raise RegistryContractError("registry snapshot.rows must be an array")
        result = cls(
            schema_version=_require_int(value["schema_version"], "registry snapshot.schema_version", minimum=1, maximum=1),
            document_type=_require_string(value["document_type"], "registry snapshot.document_type"),
            path=_require_string(value["path"], "registry snapshot.path"),
            sha256=_require_sha256(value["sha256"], "registry snapshot.sha256"),
            metadata_path=_require_string(value["metadata_path"], "registry snapshot.metadata_path"),
            metadata_sha256=_require_sha256(value["metadata_sha256"], "registry snapshot.metadata_sha256"),
            registry_schema_sha256=_require_sha256(value["registry_schema_sha256"], "registry snapshot.registry_schema_sha256"),
            source_manifest_sha256=_require_sha256(value["source_manifest_sha256"], "registry snapshot.source_manifest_sha256"),
            source_manifest_metadata_sha256=_require_sha256(value["source_manifest_metadata_sha256"], "registry snapshot.source_manifest_metadata_sha256"),
            data_root=_require_string(value["data_root"], "registry snapshot.data_root"),
            live_assets_verified=_require_bool(value["live_assets_verified"], "registry snapshot.live_assets_verified"),
            rows=tuple(RegistryRow.from_dict(row) for row in value["rows"]),
        )
        if result.document_type != "stage_c_registry_snapshot":
            raise RegistryContractError("registry snapshot document_type is invalid")
        if result.schema_version != 1:
            raise RegistryContractError("registry snapshot schema_version is invalid")
        _safe_relative(result.path, "registry snapshot.path")
        _safe_relative(result.metadata_path, "registry snapshot.metadata_path")
        if not result.rows:
            raise RegistryContractError("registry snapshot rows must not be empty")
        orders = [row.selection_order for row in result.rows]
        if orders != list(range(len(result.rows))):
            raise RegistryContractError(
                "registry snapshot rows must be ordered by selection_order"
            )
        for name in ("sample_id", "image_id", "relative_path", "image_sha256"):
            items = [getattr(row, name) for row in result.rows]
            if len(items) != len(set(items)):
                raise RegistryContractError(
                    f"registry snapshot rows have duplicate {name}"
                )
        if len(result.rows) != 20:
            raise RegistryContractError("registry snapshot must retain all 20 rows")
        if Counter(row.role for row in result.rows) != Counter(
            {"calibration": 5, "holdout": 15}
        ):
            raise RegistryContractError("registry snapshot role counts are invalid")
        if tuple(row.sample_id for row in result.rows if row.audit) != AUDIT_SAMPLE_IDS:
            raise RegistryContractError("registry snapshot audit rows are invalid")
        if tuple(
            row.sample_id for row in result.rows if row.role == "calibration"
        ) != CALIBRATION_SAMPLE_IDS:
            raise RegistryContractError("registry snapshot calibration rows are invalid")
        return result

    def by_sample_id(self) -> dict[str, RegistryRow]:
        return {row.sample_id: row for row in self.rows}


def _validate_registry_metadata(metadata: Any, config: StageCConfig) -> str:
    top = {
        "schema_version", "document_type", "dataset", "official_labels",
        "registry", "selection", "generation", "asset_findings",
    }
    _exact_keys(metadata, top, "registry metadata")
    if metadata["schema_version"] != 1 or metadata["document_type"] != "stage_c_sample_registry_metadata":
        raise RegistryContractError("registry metadata identity is invalid")
    _exact_keys(metadata["dataset"], {
        "logical_root", "runtime_root_relative_to_workspace",
        "source_manifest_path", "source_manifest_sha256",
        "source_manifest_metadata_path", "source_manifest_metadata_sha256",
    }, "registry metadata.dataset")
    _exact_keys(metadata["official_labels"], {
        "devkit_relative_path", "devkit_sha256", "label_source",
        "synset_order_sha256",
    }, "registry metadata.official_labels")
    _exact_keys(metadata["registry"], {
        "path", "sha256", "sample_count", "calibration_count",
        "holdout_count", "audit_count",
    }, "registry metadata.registry")
    _exact_keys(metadata["selection"], {
        "legacy_source_read_only", "legacy_list_path", "legacy_list_sha256",
        "legacy_script_path", "legacy_script_sha256", "rule_id",
        "source_selection_indices_zero_based",
        "calibration_selection_orders_zero_based",
        "audit_selection_orders_zero_based", "selection_information_used",
        "forbidden_selection_information",
    }, "registry metadata.selection")
    _exact_keys(metadata["generation"], {
        "stage_a_command", "base_git_commit", "tree_dirty_during_generation",
        "dirty_reason", "dependency_lock_sha256",
    }, "registry metadata.generation")
    if not isinstance(metadata["asset_findings"], list):
        raise RegistryContractError("registry metadata.asset_findings must be an array")
    for index, finding in enumerate(metadata["asset_findings"]):
        if not isinstance(finding, Mapping):
            raise RegistryContractError(
                f"registry metadata.asset_findings[{index}] must be an object"
            )
        allowed = {"sample_id", "scope", "finding", "disposition"}
        if not set(finding) <= allowed or not {"finding", "disposition"} <= set(finding):
            raise RegistryContractError(
                f"registry metadata.asset_findings[{index}] fields are invalid"
            )
        if ("sample_id" in finding) == ("scope" in finding):
            raise RegistryContractError(
                f"registry metadata.asset_findings[{index}] requires one subject"
            )
        for key, value in finding.items():
            _require_string(value, f"registry metadata.asset_findings[{index}].{key}")
    expected = config.registry
    checks = {
        "registry.path": (metadata["registry"]["path"], expected.path),
        "registry.sha256": (metadata["registry"]["sha256"], expected.sha256),
        "dataset.source_manifest_path": (metadata["dataset"]["source_manifest_path"], expected.source_manifest_path),
        "dataset.source_manifest_sha256": (metadata["dataset"]["source_manifest_sha256"], expected.source_manifest_sha256),
        "dataset.source_manifest_metadata_path": (metadata["dataset"]["source_manifest_metadata_path"], expected.source_manifest_metadata_path),
        "dataset.source_manifest_metadata_sha256": (metadata["dataset"]["source_manifest_metadata_sha256"], expected.source_manifest_metadata_sha256),
        "official_labels.devkit_sha256": (metadata["official_labels"]["devkit_sha256"], expected.official_devkit_sha256),
    }
    for name, (actual, frozen) in checks.items():
        if actual != frozen:
            raise RegistryContractError(f"registry metadata {name} disagrees with config")
    counts = expected.expected_counts
    for metadata_name, config_name in (
        ("sample_count", "total"), ("calibration_count", "calibration"),
        ("holdout_count", "holdout"), ("audit_count", "audit"),
    ):
        if metadata["registry"][metadata_name] != counts[config_name]:
            raise RegistryContractError(f"registry metadata {metadata_name} disagrees with config")
    if metadata["official_labels"]["label_source"] != "official_devkit_val_gt":
        raise RegistryContractError("registry metadata label source is invalid")
    if metadata["dataset"]["logical_root"] != "ILSVRC2012_val_flat_500":
        raise RegistryContractError("registry metadata logical_root is not frozen")
    for path_key in (
        "runtime_root_relative_to_workspace", "source_manifest_path",
        "source_manifest_metadata_path",
    ):
        _safe_relative(metadata["dataset"][path_key], f"registry metadata.dataset.{path_key}")
    _safe_relative(
        metadata["official_labels"]["devkit_relative_path"],
        "registry metadata.official_labels.devkit_relative_path",
    )
    _require_sha256(
        metadata["official_labels"]["synset_order_sha256"],
        "registry metadata.official_labels.synset_order_sha256",
    )
    selection = metadata["selection"]
    if selection["legacy_source_read_only"] is not True or selection["rule_id"] != SELECTION_RULE:
        raise RegistryContractError("registry metadata legacy selection rule is invalid")
    for path_key in ("legacy_list_path", "legacy_script_path"):
        _safe_relative(selection[path_key], f"registry metadata.selection.{path_key}")
    for hash_key in ("legacy_list_sha256", "legacy_script_sha256"):
        _require_sha256(selection[hash_key], f"registry metadata.selection.{hash_key}")
    if selection["source_selection_indices_zero_based"] != [
        0, 26, 52, 78, 105, 131, 157, 183, 210, 236,
        262, 288, 315, 341, 367, 393, 420, 446, 472, 499,
    ]:
        raise RegistryContractError("registry metadata source selection indices are invalid")
    if selection["calibration_selection_orders_zero_based"] != [0, 4, 9, 14, 19]:
        raise RegistryContractError("registry metadata calibration orders are invalid")
    if selection["audit_selection_orders_zero_based"] != [0, 19]:
        raise RegistryContractError("registry metadata audit orders are invalid")
    if selection["selection_information_used"] != (
        "legacy filename order and fixed positions only"
    ):
        raise RegistryContractError("registry metadata selection information is invalid")
    if selection["forbidden_selection_information"] != [
        "image_content", "model_prediction", "prediction_correctness",
        "attribution_map", "candidate_metric",
    ]:
        raise RegistryContractError("registry metadata forbidden selection information is invalid")
    generation = metadata["generation"]
    _require_string(generation["stage_a_command"], "registry metadata.generation.stage_a_command")
    if not isinstance(generation["base_git_commit"], str) or _COMMIT_RE.fullmatch(generation["base_git_commit"]) is None:
        raise RegistryContractError("registry metadata generation commit is invalid")
    _require_bool(generation["tree_dirty_during_generation"], "registry metadata.generation.tree_dirty_during_generation")
    _require_string(generation["dirty_reason"], "registry metadata.generation.dirty_reason")
    if generation["dependency_lock_sha256"] != sha256_file(DEFAULT_DEPENDENCY_LOCK):
        raise RegistryContractError("registry metadata dependency lock hash is stale")
    data_root = metadata["dataset"]["runtime_root_relative_to_workspace"]
    _safe_relative(data_root, "registry metadata runtime data root")
    return data_root


def _validate_registry_rows(
    rows: Sequence[RegistryRow], config: StageCConfig,
    source_rows: Sequence[Mapping[str, str]],
) -> None:
    counts = config.registry.expected_counts
    if len(rows) != counts["total"]:
        raise RegistryContractError("registry row count disagrees with config")
    if [row.selection_order for row in rows] != list(range(counts["total"])):
        raise RegistryContractError("registry selection_order must be exact file order 0..19")
    for name in ("sample_id", "image_id", "relative_path", "image_sha256"):
        values = [getattr(row, name) for row in rows]
        if len(values) != len(set(values)):
            raise RegistryContractError(f"registry {name} values must be unique")
    role_counts = Counter(row.role for row in rows)
    if role_counts != Counter({"calibration": counts["calibration"], "holdout": counts["holdout"]}):
        raise RegistryContractError("registry role counts disagree with config")
    audit_rows = [row for row in rows if row.audit]
    if len(audit_rows) != counts["audit"] or any(row.role != "calibration" for row in audit_rows):
        raise RegistryContractError("audit rows must be the configured calibration subset")
    configured_audits = tuple(config.retention["audit_sample_ids"])
    if tuple(row.sample_id for row in audit_rows) != configured_audits:
        raise RegistryContractError("registry audit IDs disagree with retention config")
    if tuple(row.sample_id for row in rows if row.role == "calibration") != CALIBRATION_SAMPLE_IDS:
        raise RegistryContractError("registry calibration IDs disagree with frozen selection contract")
    if len(source_rows) != 500:
        raise RegistryContractError("source manifest must contain exactly 500 rows")
    by_path = {row["relative_path"]: row for row in source_rows}
    if len(by_path) != 500:
        raise RegistryContractError("source manifest paths must be unique")
    for row in rows:
        source = by_path.get(row.relative_path)
        if source is None or source_rows[row.source_selection_index] is not source:
            raise RegistryContractError(
                f"registry {row.sample_id} does not match source_selection_index"
            )
        expected = {
            "sample_id": row.sample_id, "relative_path": row.relative_path,
            "image_sha256": row.image_sha256, "size_bytes": str(row.size_bytes),
            "width": str(row.width), "height": str(row.height),
            "wnid": row.official_wnid,
            "imagenet1k_index": str(row.imagenet1k_index),
            "label_source": row.label_source,
        }
        if source != expected:
            raise RegistryContractError(f"registry {row.sample_id} differs from source manifest")


def validate_live_registry_assets(
    registry: RegistrySnapshot, *, data_root: Path | str | None = None
) -> None:
    """Validate every selected JPEG by path, bytes, digest, dimensions and decode."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a project dependency
        raise RuntimeError("Pillow is required for live Stage C asset validation") from exc
    root = Path(data_root if data_root is not None else registry.data_root).resolve()
    if not root.is_dir():
        raise RegistryContractError(f"registry live data root is not a directory: {root}")
    for row in registry.rows:
        path = (root / row.relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RegistryContractError(f"live image is missing or escapes data root: {row.sample_id}")
        if path.stat().st_size != row.size_bytes:
            raise RegistryContractError(f"live image byte count mismatch: {row.sample_id}")
        if sha256_file(path) != row.image_sha256:
            raise RegistryContractError(f"live image SHA-256 mismatch: {row.sample_id}")
        try:
            with Image.open(path) as image:
                if image.size != (row.width, row.height):
                    raise RegistryContractError(f"live image dimensions mismatch: {row.sample_id}")
                image.verify()
        except RegistryContractError:
            raise
        except Exception as exc:
            raise RegistryContractError(f"live image decode failed: {row.sample_id}") from exc


def load_stage_c_registry(
    config: StageCConfig, *,
    registry_path: Path | str | None = None,
    registry_schema_path: Path | str = DEFAULT_REGISTRY_SCHEMA,
    metadata_path: Path | str | None = None,
    repo_root: Path | str = REPO_ROOT,
    workspace_root: Path | str = WORKSPACE_ROOT,
    data_root: Path | str | None = None,
    verify_live_assets: bool = False,
    recovery_binding: RecoveryV2Binding | None = None,
) -> RegistrySnapshot:
    """Load the frozen CSV and all config-bound non-image assets strictly.

    ``verify_live_assets`` is the only optional validation tier.  Registry,
    metadata, source-manifest, source-metadata and devkit hashes are always
    checked against the frozen config.  A validated ``recovery_binding``
    relocates only the devkit and live-image root into the active recovery
    project; generation-only legacy selection inputs are not reopened.
    """

    if not isinstance(config, StageCConfig):
        raise TypeError("config must be a StageCConfig")
    if recovery_binding is not None and not isinstance(
        recovery_binding, RecoveryV2Binding
    ):
        raise TypeError("recovery_binding must be a RecoveryV2Binding")
    _require_bool(verify_live_assets, "verify_live_assets")
    repo = Path(repo_root).resolve()
    workspace = Path(workspace_root).resolve()
    if recovery_binding is not None:
        reloaded_binding = load_recovery_v2_binding(
            recovery_binding.path, schema_path=recovery_binding.schema_path,
            repo_root=repo, verify_assets=False,
        )
        mismatches = []
        if recovery_binding.document != reloaded_binding.document:
            mismatches.append("document")
        for name in (
            "path", "schema_path", "model_path", "data_root",
            "devkit_path", "source_manifest_path",
        ):
            if getattr(recovery_binding, name) != getattr(reloaded_binding, name):
                mismatches.append(name)
        if mismatches:
            raise RegistryContractError(
                "recovery binding disagrees with the binding reloaded from disk: "
                + ", ".join(mismatches)
            )
        recovery_binding = reloaded_binding
    reg_path = Path(registry_path).resolve() if registry_path is not None else _resolved_under(repo, config.registry.path, "registry.path")
    meta_path = Path(metadata_path).resolve() if metadata_path is not None else _resolved_under(repo, config.registry.metadata_path, "registry.metadata_path")
    schema_path = Path(registry_schema_path).resolve()
    if sha256_file(schema_path) != FROZEN_REGISTRY_SCHEMA_SHA256:
        raise RegistryContractError("registry schema SHA-256 is not the frozen schema")
    schema = strict_json_load(schema_path)
    try:
        schema_columns = tuple(schema["properties"]["required_columns"]["const"])
        row_properties = set(schema["properties"]["row_contract"]["properties"])
    except (KeyError, TypeError) as exc:
        raise RegistryContractError("registry schema has an invalid structure") from exc
    if schema_columns != REGISTRY_COLUMNS or row_properties != set(REGISTRY_COLUMNS):
        raise RegistryContractError("registry schema does not describe the frozen CSV columns")
    raw_rows = _read_csv_exact(reg_path, REGISTRY_COLUMNS, "Stage C registry")
    rows = tuple(RegistryRow.from_csv(row, row_number=index) for index, row in enumerate(raw_rows, 2))
    if sha256_file(reg_path) != config.registry.sha256:
        raise RegistryContractError("registry SHA-256 disagrees with config")
    if not meta_path.is_file() or sha256_file(meta_path) != config.registry.metadata_sha256:
        raise RegistryContractError("registry metadata SHA-256 disagrees with config")
    metadata = strict_json_load(meta_path)
    configured_data_root = _validate_registry_metadata(metadata, config)
    if recovery_binding is None:
        for path_key, hash_key in (
            ("legacy_list_path", "legacy_list_sha256"),
            ("legacy_script_path", "legacy_script_sha256"),
        ):
            legacy_asset = _resolved_under(
                workspace, metadata["selection"][path_key],
                f"registry metadata.selection.{path_key}",
            )
            if sha256_file(legacy_asset) != metadata["selection"][hash_key]:
                raise RegistryContractError(
                    f"registry metadata selection asset hash is stale: {path_key}"
                )
    source_path = _resolved_under(repo, config.registry.source_manifest_path, "registry.source_manifest_path")
    source_metadata_path = _resolved_under(repo, config.registry.source_manifest_metadata_path, "registry.source_manifest_metadata_path")
    if sha256_file(source_path) != config.registry.source_manifest_sha256:
        raise RegistryContractError("source manifest SHA-256 disagrees with config")
    if sha256_file(source_metadata_path) != config.registry.source_manifest_metadata_sha256:
        raise RegistryContractError("source manifest metadata SHA-256 disagrees with config")
    source_metadata = strict_json_load(source_metadata_path)
    _exact_keys(source_metadata, {
        "code", "dataset_root", "dependencies", "labeled_count",
        "mapping", "official_devkit", "sample_count", "schema_version",
    }, "source manifest metadata")
    _exact_keys(source_metadata["code"], {"commit", "dirty"}, "source manifest metadata.code")
    _exact_keys(source_metadata["dependencies"], {"lock_path", "lock_sha256"}, "source manifest metadata.dependencies")
    _exact_keys(source_metadata["mapping"], {"name", "synset_order_sha256"}, "source manifest metadata.mapping")
    _exact_keys(source_metadata["official_devkit"], {"path", "sha256"}, "source manifest metadata.official_devkit")
    if source_metadata["schema_version"] != 1 or source_metadata.get("sample_count") != 500 or source_metadata.get("labeled_count") != 500:
        raise RegistryContractError("source manifest metadata must bind 500 labeled samples")
    if source_metadata["dependencies"]["lock_sha256"] != sha256_file(DEFAULT_DEPENDENCY_LOCK):
        raise RegistryContractError("source manifest metadata dependency lock hash is stale")
    if source_metadata["official_devkit"]["sha256"] != config.registry.official_devkit_sha256:
        raise RegistryContractError("source manifest metadata devkit hash is stale")
    if source_metadata["mapping"]["synset_order_sha256"] != metadata["official_labels"]["synset_order_sha256"]:
        raise RegistryContractError("source manifest synset-order hash is inconsistent")
    source_rows = _read_csv_exact(source_path, SOURCE_MANIFEST_COLUMNS, "Stage C source manifest")
    _validate_registry_rows(rows, config, source_rows)
    devkit_path = (
        recovery_binding.devkit_path
        if recovery_binding is not None
        else _resolved_under(
            repo, metadata["official_labels"]["devkit_relative_path"],
            "registry metadata devkit_relative_path",
        )
    )
    if sha256_file(devkit_path) != config.registry.official_devkit_sha256:
        raise RegistryContractError("official devkit SHA-256 disagrees with config")
    if recovery_binding is not None:
        bound_data_root = recovery_binding.data_root.resolve()
        if data_root is not None and Path(data_root).resolve() != bound_data_root:
            raise RegistryContractError(
                "data_root disagrees with recovery binding"
            )
        live_root = bound_data_root
    else:
        live_root = (
            Path(data_root).resolve() if data_root is not None
            else _resolved_under(
                workspace, configured_data_root,
                "registry metadata runtime data root",
            )
        )
    snapshot = RegistrySnapshot(
        schema_version=1, document_type="stage_c_registry_snapshot",
        path=config.registry.path, sha256=config.registry.sha256,
        metadata_path=config.registry.metadata_path,
        metadata_sha256=config.registry.metadata_sha256,
        registry_schema_sha256=sha256_file(schema_path),
        source_manifest_sha256=config.registry.source_manifest_sha256,
        source_manifest_metadata_sha256=config.registry.source_manifest_metadata_sha256,
        data_root=str(live_root), live_assets_verified=verify_live_assets, rows=rows,
    )
    if verify_live_assets:
        validate_live_registry_assets(snapshot, data_root=live_root)
    return snapshot


@dataclass(frozen=True, order=True, slots=True)
class ValueCandidate:
    block_index: int
    head_index: int

    def __post_init__(self) -> None:
        _require_int(self.block_index, "value candidate.block_index", minimum=0, maximum=11)
        _require_int(self.head_index, "value candidate.head_index", minimum=0, maximum=11)

    @property
    def candidate_key(self) -> str:
        return f"block_{self.block_index:02d}_head_{self.head_index:02d}"

    @property
    def depth_band(self) -> int:
        return self.block_index // 3

    def to_dict(self) -> dict[str, int]:
        return {"block_index": self.block_index, "head_index": self.head_index}

    def to_selection_ref(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key, "block_index": self.block_index,
            "head_index": self.head_index, "depth_band": self.depth_band,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValueCandidate":
        _exact_keys(value, {"block_index", "head_index"}, "value candidate")
        return cls(
            _require_int(value["block_index"], "value candidate.block_index", minimum=0, maximum=11),
            _require_int(value["head_index"], "value candidate.head_index", minimum=0, maximum=11),
        )

    @classmethod
    def from_selection_ref(cls, value: Mapping[str, Any], context: str = "selection reference") -> "ValueCandidate":
        _exact_keys(value, {"candidate_key", "block_index", "head_index", "depth_band"}, context)
        depth_band = _require_int(
            value["depth_band"], f"{context}.depth_band", minimum=0, maximum=3
        )
        candidate = cls(
            _require_int(value["block_index"], f"{context}.block_index", minimum=0, maximum=11),
            _require_int(value["head_index"], f"{context}.head_index", minimum=0, maximum=11),
        )
        if value["candidate_key"] != candidate.candidate_key or depth_band != candidate.depth_band:
            raise SelectionContractError(f"{context} fields are inconsistent")
        return candidate


@dataclass(frozen=True, slots=True)
class SelectionState:
    schema_version: int
    status: str
    path: str
    reason: str | None
    sha256: str | None
    primary: ValueCandidate | None
    depth_band_winners: tuple[ValueCandidate, ...]

    def __post_init__(self) -> None:
        _require_int(self.schema_version, "selection state.schema_version", minimum=1, maximum=1)
        if self.status not in SELECTION_STATUSES:
            raise SelectionContractError(f"unknown selection status: {self.status!r}")
        _safe_relative(self.path, "selection state.path")
        if self.status == "not_applicable_pre_c8":
            if not isinstance(self.reason, str) or not self.reason:
                raise SelectionContractError("pre-C8 selection state requires a non-empty reason")
            if self.sha256 is not None or self.primary is not None or self.depth_band_winners:
                raise SelectionContractError("pre-C8 selection state cannot contain fabricated selection data")
        else:
            if self.reason is not None:
                raise SelectionContractError("present selection state reason must be null")
            _require_sha256(self.sha256, "selection state.sha256")
            if self.primary is None or len(self.depth_band_winners) != 4:
                raise SelectionContractError("present selection requires primary and four band winners")
            if tuple(item.depth_band for item in self.depth_band_winners) != (0, 1, 2, 3):
                raise SelectionContractError("selection winners must be ordered by depth band 0..3")
            if self.primary not in self.depth_band_winners:
                raise SelectionContractError("selection primary must be a depth-band winner")

    @classmethod
    def missing(cls, path: str) -> "SelectionState":
        return cls(
            1, "not_applicable_pre_c8", path,
            "C8 selection has not been produced; only calibration explicit/all-144 scope is legal",
            None, None, (),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "status": self.status,
            "path": self.path, "reason": self.reason, "sha256": self.sha256,
            "primary": None if self.primary is None else self.primary.to_selection_ref(),
            "depth_band_winners": [item.to_selection_ref() for item in self.depth_band_winners],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionState":
        keys = {"schema_version", "status", "path", "reason", "sha256", "primary", "depth_band_winners"}
        _exact_keys(value, keys, "selection state")
        winners_raw = value["depth_band_winners"]
        if not isinstance(winners_raw, list):
            raise SelectionContractError("selection state.depth_band_winners must be an array")
        primary_raw = value["primary"]
        return cls(
            schema_version=_require_int(value["schema_version"], "selection state.schema_version", minimum=1, maximum=1),
            status=_require_string(value["status"], "selection state.status"),
            path=_require_string(value["path"], "selection state.path"),
            reason=value["reason"],
            sha256=None if value["sha256"] is None else _require_sha256(value["sha256"], "selection state.sha256"),
            primary=None if primary_raw is None else ValueCandidate.from_selection_ref(primary_raw, "selection state.primary"),
            depth_band_winners=tuple(
                ValueCandidate.from_selection_ref(item, f"selection state.depth_band_winners[{index}]")
                for index, item in enumerate(winners_raw)
            ),
        )


_SELECTION_TOP_KEYS = {
    "schema_version", "document_type", "generated_by", "holdout_accessed",
    "input_hashes", "calibration_sample_ids", "candidate_records",
    "depth_band_winners", "primary", "failure_records",
}
_SELECTION_HASH_KEYS = {
    "code_commit", "code_tree_clean", "model_sha256",
    "dependency_lock_sha256", "registry_sha256",
    "registry_metadata_sha256", "protocol_sha256", "acceptance_sha256",
    "config_sha256", "config_schema_sha256",
    "calibration_artifact_manifest_sha256",
}
_CANDIDATE_KEYS = {
    "candidate_key", "block_index", "head_index", "depth_band",
    "calibration_sample_count", "eligible", "validity_failures",
    "count_delta_fid_fine4_gt_zero", "median_delta_fid_fine4",
    "minimum_perturbation_median_residual_spearman",
    "median_patch_residual_l1_ratio", "rounded_rank_tuple",
    "resource_within_hard_budget", "artifact_manifest_sha256",
    "rank_within_depth_band", "selected_depth_band_winner",
    "selected_primary",
}
_REF_KEYS = {"candidate_key", "block_index", "head_index", "depth_band"}
_FAILURE_KEYS = {"candidate_key", "sample_id", "reason"}


def _finite_number_or_none(value: Any, context: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SelectionContractError(f"{context} must be a finite number or null")
    return value


def _validate_selection_document(document: Any) -> tuple[ValueCandidate, tuple[ValueCandidate, ...]]:
    _exact_keys(document, _SELECTION_TOP_KEYS, "selection")
    if document["schema_version"] != 1 or document["document_type"] != "stage_c_selection":
        raise SelectionContractError("selection identity fields are invalid")
    _require_string(document["generated_by"], "selection.generated_by")
    if document["holdout_accessed"] is not False:
        raise SelectionContractError("selection holdout_accessed must be false")
    hashes = _exact_keys(document["input_hashes"], _SELECTION_HASH_KEYS, "selection.input_hashes")
    if not isinstance(hashes["code_commit"], str) or _COMMIT_RE.fullmatch(hashes["code_commit"]) is None:
        raise SelectionContractError("selection.input_hashes.code_commit is invalid")
    if hashes["code_tree_clean"] is not True:
        raise SelectionContractError("selection must bind a clean code tree")
    for key in _SELECTION_HASH_KEYS - {"code_commit", "code_tree_clean"}:
        _require_sha256(hashes[key], f"selection.input_hashes.{key}")
    calibration = document["calibration_sample_ids"]
    if not isinstance(calibration, list) or tuple(calibration) != CALIBRATION_SAMPLE_IDS:
        raise SelectionContractError("selection calibration_sample_ids are not the frozen five")
    candidates = document["candidate_records"]
    if not isinstance(candidates, list) or len(candidates) != 144:
        raise SelectionContractError("selection must contain exactly 144 candidate records")
    by_key: dict[str, Mapping[str, Any]] = {}
    by_pair: dict[tuple[int, int], Mapping[str, Any]] = {}
    for index, item in enumerate(candidates):
        context = f"selection.candidate_records[{index}]"
        _exact_keys(item, _CANDIDATE_KEYS, context)
        block = _require_int(item["block_index"], f"{context}.block_index", minimum=0, maximum=11)
        head = _require_int(item["head_index"], f"{context}.head_index", minimum=0, maximum=11)
        candidate = ValueCandidate(block, head)
        depth_band = _require_int(
            item["depth_band"], f"{context}.depth_band", minimum=0, maximum=3
        )
        if item["candidate_key"] != candidate.candidate_key or depth_band != candidate.depth_band:
            raise SelectionContractError(f"{context} key/index/depth-band mismatch")
        if candidate.candidate_key in by_key or (block, head) in by_pair:
            raise SelectionContractError("selection candidates must cover 12x12 exactly once")
        by_key[candidate.candidate_key] = item
        by_pair[(block, head)] = item
        if item["calibration_sample_count"] != 5:
            raise SelectionContractError(f"{context}.calibration_sample_count must be five")
        eligible = _require_bool(item["eligible"], f"{context}.eligible")
        failures = item["validity_failures"]
        if not isinstance(failures, list) or any(not isinstance(reason, str) or not reason for reason in failures) or len(failures) != len(set(failures)):
            raise SelectionContractError(f"{context}.validity_failures is invalid")
        n_positive = item["count_delta_fid_fine4_gt_zero"]
        if n_positive is not None:
            _require_int(n_positive, f"{context}.count_delta_fid_fine4_gt_zero", minimum=0, maximum=5)
        median_fid = _finite_number_or_none(item["median_delta_fid_fine4"], f"{context}.median_delta_fid_fine4")
        median_stability = _finite_number_or_none(item["minimum_perturbation_median_residual_spearman"], f"{context}.minimum_perturbation_median_residual_spearman")
        median_residual = _finite_number_or_none(item["median_patch_residual_l1_ratio"], f"{context}.median_patch_residual_l1_ratio")
        _require_bool(item["resource_within_hard_budget"], f"{context}.resource_within_hard_budget")
        _require_sha256(item["artifact_manifest_sha256"], f"{context}.artifact_manifest_sha256")
        _require_bool(item["selected_depth_band_winner"], f"{context}.selected_depth_band_winner")
        _require_bool(item["selected_primary"], f"{context}.selected_primary")
        rank = item["rank_within_depth_band"]
        if rank is not None:
            _require_int(rank, f"{context}.rank_within_depth_band", minimum=1, maximum=36)
        rounded = item["rounded_rank_tuple"]
        if eligible:
            if failures or item["resource_within_hard_budget"] is not True:
                raise SelectionContractError(f"{context} eligible flag contradicts failures/resource budget")
            if n_positive is None or median_fid is None or rank is None:
                raise SelectionContractError(f"{context} eligible candidate lacks ranking values")
            expected_tuple = [
                n_positive, round(median_fid, 6),
                -1.0 if median_stability is None else round(median_stability, 6),
                0.0 if median_residual is None else round(median_residual, 6),
                -block, -head,
            ]
            if not isinstance(rounded, list) or len(rounded) != 6 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in rounded
            ) or rounded != expected_tuple:
                raise SelectionContractError(f"{context}.rounded_rank_tuple is inconsistent")
        else:
            if not failures or rounded is not None or rank is not None:
                raise SelectionContractError(f"{context} ineligible candidate has invalid ranking state")
            if item["selected_depth_band_winner"] or item["selected_primary"]:
                raise SelectionContractError(f"{context} ineligible candidate cannot be selected")
    if set(by_pair) != {(block, head) for block in range(12) for head in range(12)}:
        raise SelectionContractError("selection must cover every block/head pair exactly once")
    top_by_band: dict[int, Mapping[str, Any]] = {}
    for band in range(4):
        eligible = [item for item in candidates if item["depth_band"] == band and item["eligible"]]
        if not eligible:
            raise SelectionContractError(f"selection depth band {band} has no eligible candidate")
        ordered = sorted(eligible, key=lambda item: tuple(item["rounded_rank_tuple"]), reverse=True)
        if [item["rank_within_depth_band"] for item in ordered] != list(range(1, len(ordered) + 1)):
            raise SelectionContractError(f"selection depth band {band} ranks are inconsistent")
        top_by_band[band] = ordered[0]
    winner_refs = document["depth_band_winners"]
    if not isinstance(winner_refs, list) or len(winner_refs) != 4:
        raise SelectionContractError("selection requires exactly four depth-band winners")
    winners = tuple(
        ValueCandidate.from_selection_ref(reference, f"selection.depth_band_winners[{index}]")
        for index, reference in enumerate(winner_refs)
    )
    if tuple(item.depth_band for item in winners) != (0, 1, 2, 3):
        raise SelectionContractError("selection winners must be ordered one per band")
    for winner in winners:
        if top_by_band[winner.depth_band]["candidate_key"] != winner.candidate_key:
            raise SelectionContractError("selection winner is not rank one in its depth band")
    primary = ValueCandidate.from_selection_ref(document["primary"], "selection.primary")
    ordered_winners = sorted(
        (by_key[winner.candidate_key] for winner in winners),
        key=lambda item: tuple(item["rounded_rank_tuple"]), reverse=True,
    )
    if primary.candidate_key != ordered_winners[0]["candidate_key"]:
        raise SelectionContractError("selection primary is not the frozen top band winner")
    winner_keys = {winner.candidate_key for winner in winners}
    flagged_winners = {item["candidate_key"] for item in candidates if item["selected_depth_band_winner"]}
    flagged_primary = [item["candidate_key"] for item in candidates if item["selected_primary"]]
    if flagged_winners != winner_keys or flagged_primary != [primary.candidate_key]:
        raise SelectionContractError("selection flags disagree with winner/primary references")
    failures = document["failure_records"]
    if not isinstance(failures, list):
        raise SelectionContractError("selection.failure_records must be an array")
    for index, failure in enumerate(failures):
        context = f"selection.failure_records[{index}]"
        _exact_keys(failure, _FAILURE_KEYS, context)
        if failure["candidate_key"] not in by_key:
            raise SelectionContractError(f"{context} references an unknown candidate")
        sample_id = failure["sample_id"]
        if sample_id is not None and sample_id not in CALIBRATION_SAMPLE_IDS:
            raise SelectionContractError(f"{context} references a non-calibration sample")
        _require_string(failure["reason"], f"{context}.reason")
    return primary, winners


def _static_selection_input_hashes(
    config: StageCConfig, *, config_path: Path, config_schema_path: Path
) -> dict[str, Any]:
    protocol_path = _resolved_under(REPO_ROOT, config.protocol.path, "protocol.path")
    return {
        "model_sha256": MODEL_SHA256,
        "dependency_lock_sha256": sha256_file(DEFAULT_DEPENDENCY_LOCK),
        "registry_sha256": config.registry.sha256,
        "registry_metadata_sha256": config.registry.metadata_sha256,
        "protocol_sha256": sha256_file(protocol_path),
        "acceptance_sha256": sha256_file(DEFAULT_ACCEPTANCE),
        "config_sha256": sha256_file(config_path),
        "config_schema_sha256": sha256_file(config_schema_path),
    }


def load_stage_c_selection(
    config: StageCConfig, *, selection_path: Path | str | None = None,
    schema_path: Path | str = DEFAULT_SELECTION_SCHEMA, required: bool = False,
    expected_input_hashes: Mapping[str, Any] | None = None,
    config_path: Path | str = DEFAULT_STAGE_C_CONFIG,
    config_schema_path: Path | str = DEFAULT_STAGE_C_SCHEMA,
    repo_root: Path | str = REPO_ROOT,
) -> SelectionState:
    """Load future C8 output or return an explicit pre-C8 absence state."""

    if not isinstance(config, StageCConfig):
        raise TypeError("config must be a StageCConfig")
    _require_bool(required, "required")
    configured_path = config.selection["output_path"]
    path = Path(selection_path).resolve() if selection_path is not None else _resolved_under(Path(repo_root), configured_path, "selection.output_path")
    if not path.exists():
        if required:
            raise SelectionContractError(
                f"Stage C selection is required for this scope but is missing: {path}"
            )
        return SelectionState.missing(configured_path)
    if not path.is_file():
        raise SelectionContractError(f"Stage C selection is not a regular file: {path}")
    if sha256_file(Path(schema_path).resolve()) != FROZEN_SELECTION_SCHEMA_SHA256:
        raise SelectionContractError("selection schema SHA-256 is not the frozen schema")
    schema = strict_json_load(schema_path)
    try:
        if set(schema["required"]) != _SELECTION_TOP_KEYS or schema["properties"]["candidate_records"]["minItems"] != 144:
            raise SelectionContractError("selection schema is not the frozen closed schema")
    except (KeyError, TypeError) as exc:
        raise SelectionContractError("selection schema structure is invalid") from exc
    document = strict_json_load(path)
    primary, winners = _validate_selection_document(document)
    expected = _static_selection_input_hashes(
        config, config_path=Path(config_path).resolve(),
        config_schema_path=Path(config_schema_path).resolve(),
    )
    if expected_input_hashes is not None:
        if not isinstance(expected_input_hashes, Mapping):
            raise TypeError("expected_input_hashes must be a mapping or None")
        unknown = set(expected_input_hashes) - _SELECTION_HASH_KEYS
        if unknown:
            raise SelectionContractError(f"unknown expected selection hash fields: {sorted(unknown)}")
        conflicting = {
            key for key in expected_input_hashes
            if key in expected and expected_input_hashes[key] != expected[key]
        }
        if conflicting:
            raise SelectionContractError(
                "caller cannot override static selection hash bindings: "
                f"{sorted(conflicting)}"
            )
        expected.update(dict(expected_input_hashes))
    for key, expected_value in expected.items():
        if document["input_hashes"].get(key) != expected_value:
            raise SelectionContractError(f"selection input hash binding mismatch: {key}")
    return SelectionState(
        1, "present", configured_path, None, sha256_file(path), primary, winners
    )


def _coerce_candidate(value: Any, context: str) -> ValueCandidate:
    if isinstance(value, ValueCandidate):
        return value
    if isinstance(value, Mapping):
        return ValueCandidate.from_dict(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return ValueCandidate(
            _require_int(value[0], f"{context}.block_index", minimum=0, maximum=11),
            _require_int(value[1], f"{context}.head_index", minimum=0, maximum=11),
        )
    raise RunSpecError(f"{context} must be a ValueCandidate, object, or pair")


@dataclass(frozen=True, slots=True)
class RunSpec:
    schema_version: int
    document_type: str
    sample_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    value_config_source: str
    value_candidates: tuple[ValueCandidate, ...]
    fidelity_pairing_context: ValueCandidate | None
    audit_repeatability_runs: int
    audit_repeatability_scope: str
    run_registered_fidelity: bool
    run_sanity: bool
    run_stability: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.document_type != "stage_c_run_spec":
            raise RunSpecError("RunSpec identity fields are invalid")
        if not self.sample_ids or len(self.sample_ids) != len(set(self.sample_ids)):
            raise RunSpecError("RunSpec sample_ids must be non-empty and unique")
        if any(not isinstance(item, str) or _SAMPLE_RE.fullmatch(item) is None for item in self.sample_ids):
            raise RunSpecError("RunSpec contains an invalid sample_id")
        if not self.method_ids or len(self.method_ids) != len(set(self.method_ids)):
            raise RunSpecError("RunSpec method_ids must be non-empty and unique")
        unknown_methods = set(self.method_ids) - set(_METHOD_CONTRACTS)
        if unknown_methods:
            raise RunSpecError(
                f"RunSpec contains unregistered method IDs: {sorted(unknown_methods)}"
            )
        if self.value_config_source not in VALUE_CONFIG_SOURCES:
            raise RunSpecError(f"unknown value_config_source: {self.value_config_source!r}")
        if len(self.value_candidates) != len(set(self.value_candidates)):
            raise RunSpecError("RunSpec value_candidates must be unique")
        if self.value_config_source == "explicit" and not self.value_candidates:
            raise RunSpecError("explicit value_config_source requires value_candidates")
        if self.value_config_source != "explicit" and self.value_candidates:
            raise RunSpecError("only explicit value_config_source may contain value_candidates")
        has_value = bool(set(self.method_ids) & set(VALUE_METHOD_IDS))
        has_class = bool(set(self.method_ids) & set(CLASS_METHOD_IDS))
        if has_value and self.value_config_source == "none":
            raise RunSpecError(
                "value-path methods require an explicit or selection candidate source"
            )
        if not has_value and self.value_config_source != "none":
            raise RunSpecError(
                "class-only RunSpec must set value_config_source='none'"
            )
        _require_int(self.audit_repeatability_runs, "RunSpec.audit_repeatability_runs", minimum=1)
        if self.audit_repeatability_scope not in AUDIT_REPEATABILITY_SCOPES:
            raise RunSpecError("audit_repeatability_scope must be 'clean_map_only'")
        for name in ("run_registered_fidelity", "run_sanity", "run_stability"):
            _require_bool(getattr(self, name), f"RunSpec.{name}")
        fidelity_methods = {
            method_id for method_id in self.method_ids
            if _METHOD_CONTRACTS[method_id][2]
        }
        if (self.run_registered_fidelity or self.run_sanity) and not fidelity_methods:
            raise RunSpecError(
                "fidelity/sanity was requested but no selected method is fidelity eligible"
            )
        if self.run_registered_fidelity or self.run_sanity:
            if self.fidelity_pairing_context is None:
                raise RunSpecError("fidelity/sanity requires an explicit block/head pairing context")
        elif self.fidelity_pairing_context is not None:
            raise RunSpecError("fidelity_pairing_context must be null when fidelity and sanity are disabled")
        if has_class and (self.run_registered_fidelity or self.run_sanity) and self.fidelity_pairing_context is None:
            raise RunSpecError(
                "global class fidelity requires an explicit pairing context"
            )

    @property
    def selection_mode(self) -> str:
        return self.value_config_source

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "sample_ids": list(self.sample_ids), "method_ids": list(self.method_ids),
            "value_config_source": self.value_config_source,
            "value_candidates": [candidate.to_dict() for candidate in self.value_candidates],
            "fidelity_pairing_context": (
                None if self.fidelity_pairing_context is None
                else self.fidelity_pairing_context.to_dict()
            ),
            "audit_repeatability_runs": self.audit_repeatability_runs,
            "audit_repeatability_scope": self.audit_repeatability_scope,
            "run_registered_fidelity": self.run_registered_fidelity,
            "run_sanity": self.run_sanity, "run_stability": self.run_stability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunSpec":
        keys = {
            "schema_version", "document_type", "sample_ids", "method_ids",
            "value_config_source", "value_candidates",
            "fidelity_pairing_context", "audit_repeatability_runs",
            "audit_repeatability_scope", "run_registered_fidelity",
            "run_sanity", "run_stability",
        }
        _exact_keys(value, keys, "RunSpec")
        if not isinstance(value["sample_ids"], list) or not isinstance(value["method_ids"], list) or not isinstance(value["value_candidates"], list):
            raise RunSpecError("RunSpec IDs and value_candidates must be arrays")
        pairing = value["fidelity_pairing_context"]
        return cls(
            schema_version=_require_int(value["schema_version"], "RunSpec.schema_version", minimum=1, maximum=1),
            document_type=_require_string(value["document_type"], "RunSpec.document_type"),
            sample_ids=tuple(value["sample_ids"]), method_ids=tuple(value["method_ids"]),
            value_config_source=_require_string(value["value_config_source"], "RunSpec.value_config_source"),
            value_candidates=tuple(ValueCandidate.from_dict(item) for item in value["value_candidates"]),
            fidelity_pairing_context=None if pairing is None else ValueCandidate.from_dict(pairing),
            audit_repeatability_runs=_require_int(value["audit_repeatability_runs"], "RunSpec.audit_repeatability_runs", minimum=1),
            audit_repeatability_scope=_require_string(value["audit_repeatability_scope"], "RunSpec.audit_repeatability_scope"),
            run_registered_fidelity=_require_bool(value["run_registered_fidelity"], "RunSpec.run_registered_fidelity"),
            run_sanity=_require_bool(value["run_sanity"], "RunSpec.run_sanity"),
            run_stability=_require_bool(value["run_stability"], "RunSpec.run_stability"),
        )


def make_run_spec(
    *, sample_ids: Sequence[str], method_ids: Sequence[str],
    value_candidates: Sequence[ValueCandidate | Mapping[str, Any] | Sequence[int]],
    value_config_source: str | None = None, selection_mode: str | None = None,
    fidelity_pairing_context: ValueCandidate | Mapping[str, Any] | Sequence[int] | None,
    audit_repeatability_runs: int, audit_repeatability_scope: str,
    run_registered_fidelity: bool, run_sanity: bool, run_stability: bool,
) -> RunSpec:
    """Construct a RunSpec without defaults for any execution policy."""

    if (value_config_source is None) == (selection_mode is None):
        raise RunSpecError("provide exactly one of value_config_source or selection_mode")
    source = value_config_source if value_config_source is not None else selection_mode
    assert source is not None
    candidates = tuple(
        _coerce_candidate(item, f"value_candidates[{index}]")
        for index, item in enumerate(value_candidates)
    )
    pairing = None if fidelity_pairing_context is None else _coerce_candidate(
        fidelity_pairing_context, "fidelity_pairing_context"
    )
    return RunSpec(
        1, "stage_c_run_spec", tuple(sample_ids), tuple(method_ids), source,
        candidates, pairing, audit_repeatability_runs, audit_repeatability_scope,
        run_registered_fidelity, run_sanity, run_stability,
    )


def load_run_spec(path: Path | str) -> RunSpec:
    try:
        value = strict_json_load(path)
        return RunSpec.from_dict(value)
    except RunSpecError:
        raise
    except Exception as exc:
        raise RunSpecError(f"invalid Stage C RunSpec: {exc}") from exc


def unit_key_digest(
    *, sample_id: str, method_id: str, config_id: str, block_index: int | None,
    head_index: int | None, target_ref: str | None, schema_version: int = 1,
) -> str:
    """Return the stable SHA-256 of the canonical UnitKey identity fields."""

    payload = {
        "schema_version": schema_version, "sample_id": sample_id,
        "method_id": method_id, "config_id": config_id,
        "block_index": block_index,
        "head_index": head_index, "target_ref": target_ref,
    }
    return _canonical_digest(payload)


@dataclass(frozen=True, slots=True)
class UnitKey:
    schema_version: int
    sample_id: str
    method_id: str
    config_id: str
    block_index: int | None
    head_index: int | None
    target_ref: str | None
    canonical_digest: str

    @classmethod
    def create(
        cls, *, sample_id: str, method_id: str, config_id: str,
        block_index: int | None,
        head_index: int | None, target_ref: str | None,
    ) -> "UnitKey":
        digest = unit_key_digest(
            sample_id=sample_id, method_id=method_id, config_id=config_id,
            block_index=block_index,
            head_index=head_index, target_ref=target_ref,
        )
        return cls(
            1, sample_id, method_id, config_id, block_index, head_index,
            target_ref, digest,
        )

    def __post_init__(self) -> None:
        if self.schema_version != 1 or _SAMPLE_RE.fullmatch(self.sample_id) is None:
            raise RunSpecError("UnitKey identity is invalid")
        if self.method_id not in _METHOD_CONTRACTS:
            raise RunSpecError("UnitKey method_id is unregistered")
        if not isinstance(self.config_id, str) or not self.config_id:
            raise RunSpecError("UnitKey config_id is invalid")
        if (self.block_index is None) != (self.head_index is None):
            raise RunSpecError("UnitKey block_index/head_index must both be integers or both null")
        if self.block_index is not None:
            _require_int(self.block_index, "UnitKey.block_index", minimum=0, maximum=11)
            _require_int(self.head_index, "UnitKey.head_index", minimum=0, maximum=11)
            expected_config_id = (
                f"block_{self.block_index:02d}_head_{self.head_index:02d}"
            )
        else:
            expected_config_id = f"global_{self.method_id}"
        expected_line, expected_target, _eligible, _structural = (
            _METHOD_CONTRACTS[self.method_id]
        )
        if expected_line == "value_path" and self.block_index is None:
            raise RunSpecError("value-path UnitKey requires a real block/head")
        if expected_line == "class_logit" and self.block_index is not None:
            raise RunSpecError("global class UnitKey must keep block/head null")
        if self.config_id != expected_config_id:
            raise RunSpecError("UnitKey config_id disagrees with its config scope")
        if self.target_ref != expected_target:
            raise RunSpecError("UnitKey target_ref disagrees with method contract")
        expected = unit_key_digest(
            sample_id=self.sample_id, method_id=self.method_id,
            config_id=self.config_id,
            block_index=self.block_index, head_index=self.head_index,
            target_ref=self.target_ref, schema_version=self.schema_version,
        )
        if self.canonical_digest != expected:
            raise RunSpecError("UnitKey canonical_digest mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "sample_id": self.sample_id,
            "method_id": self.method_id, "config_id": self.config_id,
            "block_index": self.block_index,
            "head_index": self.head_index, "target_ref": self.target_ref,
            "canonical_digest": self.canonical_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UnitKey":
        keys = {"schema_version", "sample_id", "method_id", "config_id", "block_index", "head_index", "target_ref", "canonical_digest"}
        _exact_keys(value, keys, "UnitKey")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class UnitSample:
    sample_id: str
    role: str
    audit: bool
    selection_order: int
    image_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or _SAMPLE_RE.fullmatch(self.sample_id) is None:
            raise RunSpecError("unit sample.sample_id is invalid")
        if self.role not in ("calibration", "holdout"):
            raise RunSpecError("unit sample.role is invalid")
        _require_bool(self.audit, "unit sample.audit")
        if self.audit and self.role != "calibration":
            raise RunSpecError("unit sample.audit may only be true for calibration")
        _require_int(self.selection_order, "unit sample.selection_order", minimum=0, maximum=19)
        _require_sha256(self.image_sha256, "unit sample.image_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id, "role": self.role, "audit": self.audit,
            "selection_order": self.selection_order, "image_sha256": self.image_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UnitSample":
        _exact_keys(value, {"sample_id", "role", "audit", "selection_order", "image_sha256"}, "unit sample")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class WorkItem:
    kind: str
    input_variant: str
    repeat_index: int | None
    fidelity_scale: str | None
    curve_direction: str | None
    ranking_kind: str | None
    replicate_index: int | None
    fraction_index: int | None
    fraction: float | None
    pairing_block_index: int | None
    pairing_head_index: int | None

    def __post_init__(self) -> None:
        if self.kind not in (MAP_KIND, CURVE_POINT_KIND):
            raise RunSpecError(f"unknown WorkItem kind: {self.kind!r}")
        if self.kind == MAP_KIND:
            if self.input_variant not in _MAP_VARIANTS:
                raise RunSpecError("map WorkItem input_variant is unregistered")
            if any(value is not None for value in (
                self.fidelity_scale, self.curve_direction, self.ranking_kind,
                self.replicate_index, self.fraction_index, self.fraction,
                self.pairing_block_index, self.pairing_head_index,
            )):
                raise RunSpecError("map WorkItem contains curve fields")
            if self.input_variant == "clean":
                if self.repeat_index != 0:
                    raise RunSpecError("clean map WorkItem repeat_index must be zero")
            elif self.input_variant == "repeat_clean":
                _require_int(self.repeat_index, "WorkItem.repeat_index", minimum=1)
            elif self.repeat_index is not None:
                raise RunSpecError("perturbation map WorkItem repeat_index must be null")
        else:
            if self.input_variant != "clean" or self.repeat_index is not None:
                raise RunSpecError("curve WorkItem must use the clean input and no repeat index")
            if self.fidelity_scale not in FIDELITY_SCALES or self.curve_direction not in CURVE_DIRECTIONS:
                raise RunSpecError("curve WorkItem scale/direction is invalid")
            if self.ranking_kind not in (TARGET_RANKING, RANDOM_RANKING, CENTER_RANKING):
                raise RunSpecError("curve WorkItem ranking_kind is invalid")
            expected_replicates = None if self.ranking_kind == CENTER_RANKING else range(5)
            if expected_replicates is None:
                if self.replicate_index is not None:
                    raise RunSpecError("center ranking cannot have a replicate index")
            elif self.replicate_index not in expected_replicates:
                raise RunSpecError("tie/random replicate index must be 0..4")
            _require_int(self.fraction_index, "WorkItem.fraction_index", minimum=0, maximum=20)
            if type(self.fraction) is not float or self.fraction != self.fraction_index / 20.0:
                raise RunSpecError("WorkItem fraction must equal fraction_index/20")
            _require_int(self.pairing_block_index, "WorkItem.pairing_block_index", minimum=0, maximum=11)
            _require_int(self.pairing_head_index, "WorkItem.pairing_head_index", minimum=0, maximum=11)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "input_variant": self.input_variant,
            "repeat_index": self.repeat_index, "fidelity_scale": self.fidelity_scale,
            "curve_direction": self.curve_direction, "ranking_kind": self.ranking_kind,
            "replicate_index": self.replicate_index, "fraction_index": self.fraction_index,
            "fraction": self.fraction, "pairing_block_index": self.pairing_block_index,
            "pairing_head_index": self.pairing_head_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkItem":
        keys = {
            "kind", "input_variant", "repeat_index", "fidelity_scale",
            "curve_direction", "ranking_kind", "replicate_index",
            "fraction_index", "fraction", "pairing_block_index",
            "pairing_head_index",
        }
        _exact_keys(value, keys, "WorkItem")
        return cls(**dict(value))


def _count_work_items(items: Sequence[WorkItem]) -> dict[str, int]:
    map_items = [item for item in items if item.kind == MAP_KIND]
    curve_items = [item for item in items if item.kind == CURVE_POINT_KIND]
    curve_groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for item in curve_items:
        identity = (
            item.fidelity_scale, item.curve_direction, item.ranking_kind,
            item.replicate_index, item.pairing_block_index,
            item.pairing_head_index,
        )
        assert item.fraction_index is not None
        if item.fraction_index in curve_groups[identity]:
            raise RunSpecError("WorkItem list contains a duplicate curve point")
        curve_groups[identity].add(item.fraction_index)
    expected_fractions = set(range(21))
    if any(indices != expected_fractions for indices in curve_groups.values()):
        raise RunSpecError("WorkItem list contains an incomplete fidelity curve")
    clean_maps = [item for item in map_items if item.input_variant == "clean"]
    if len(clean_maps) != 1:
        raise RunSpecError("each execution unit requires exactly one clean map")
    perturbations = [
        item.input_variant for item in map_items
        if item.input_variant not in ("clean", "repeat_clean")
    ]
    if len(perturbations) != len(set(perturbations)):
        raise RunSpecError("execution unit contains duplicate perturbation maps")
    repeats = sorted(
        item.repeat_index for item in map_items
        if item.input_variant == "repeat_clean"
    )
    if repeats != list(range(1, len(repeats) + 1)):
        raise RunSpecError("repeat-clean indices must be contiguous from one")
    return {
        "work_items": len(items), "map_runs": len(map_items),
        "clean_map_runs": sum(item.input_variant == "clean" for item in map_items),
        "perturbation_map_runs": sum(item.input_variant not in ("clean", "repeat_clean") for item in map_items),
        "repeatability_map_runs": sum(item.input_variant == "repeat_clean" for item in map_items),
        "fidelity_curves": len(curve_groups),
        "fidelity_curve_points": len(curve_items),
        "target_tie_curve_points": sum(item.ranking_kind == TARGET_RANKING for item in curve_items),
        "random_sanity_curve_points": sum(item.ranking_kind == RANDOM_RANKING for item in curve_items),
        "center_sanity_curve_points": sum(item.ranking_kind == CENTER_RANKING for item in curve_items),
        "logical_scalar_evaluations": len(curve_items),
    }


@dataclass(frozen=True, slots=True)
class ExecutionUnit:
    schema_version: int
    key: UnitKey
    sample: UnitSample
    comparison_line: str
    fidelity_eligible: bool
    structural_reference_only: bool
    work_items: tuple[WorkItem, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.sample.sample_id != self.key.sample_id:
            raise RunSpecError("ExecutionUnit identity/sample binding is invalid")
        if self.comparison_line not in ("value_path", "class_logit"):
            raise RunSpecError("ExecutionUnit comparison_line is invalid")
        _require_bool(self.fidelity_eligible, "ExecutionUnit.fidelity_eligible")
        _require_bool(self.structural_reference_only, "ExecutionUnit.structural_reference_only")
        expected_line, expected_target, expected_fidelity, expected_structural = (
            _METHOD_CONTRACTS.get(self.key.method_id, (None, None, None, None))
        )
        if (
            self.comparison_line != expected_line
            or self.key.target_ref != expected_target
            or self.fidelity_eligible is not expected_fidelity
            or self.structural_reference_only is not expected_structural
        ):
            raise RunSpecError("ExecutionUnit method metadata disagrees with frozen contract")
        if self.structural_reference_only and any(item.kind == CURVE_POINT_KIND for item in self.work_items):
            raise RunSpecError("structural-only ExecutionUnit cannot contain fidelity work")
        curve_pairings = {
            (item.pairing_block_index, item.pairing_head_index)
            for item in self.work_items
            if item.kind == CURVE_POINT_KIND
        }
        if len(curve_pairings) > 1:
            raise RunSpecError(
                "all fidelity curve points in an ExecutionUnit must share one "
                "pairing context"
            )
        if curve_pairings and self.comparison_line == "value_path":
            expected_pairing = (self.key.block_index, self.key.head_index)
            if next(iter(curve_pairings)) != expected_pairing:
                raise RunSpecError(
                    "value-path fidelity pairing context must equal the unit "
                    "block/head"
                )
        counts = _count_work_items(self.work_items)
        if not self.fidelity_eligible and counts["fidelity_curve_points"]:
            raise RunSpecError("fidelity-ineligible ExecutionUnit contains curve work")

    @property
    def audit(self) -> bool:
        return self.sample.audit

    @property
    def unit_root_name(self) -> str:
        return self.key.canonical_digest

    @property
    def work_item_counts(self) -> dict[str, int]:
        return _count_work_items(self.work_items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "key": self.key.to_dict(),
            "sample": self.sample.to_dict(), "comparison_line": self.comparison_line,
            "fidelity_eligible": self.fidelity_eligible,
            "structural_reference_only": self.structural_reference_only,
            "work_item_counts": self.work_item_counts,
            "work_items": [item.to_dict() for item in self.work_items],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionUnit":
        keys = {
            "schema_version", "key", "sample", "comparison_line",
            "fidelity_eligible", "structural_reference_only",
            "work_item_counts", "work_items",
        }
        _exact_keys(value, keys, "ExecutionUnit")
        if not isinstance(value["work_items"], list) or not isinstance(value["work_item_counts"], Mapping):
            raise RunSpecError("ExecutionUnit work item fields are invalid")
        unit = cls(
            schema_version=value["schema_version"], key=UnitKey.from_dict(value["key"]),
            sample=UnitSample.from_dict(value["sample"]),
            comparison_line=value["comparison_line"],
            fidelity_eligible=value["fidelity_eligible"],
            structural_reference_only=value["structural_reference_only"],
            work_items=tuple(WorkItem.from_dict(item) for item in value["work_items"]),
        )
        if unit.work_item_counts != dict(value["work_item_counts"]):
            raise RunSpecError("ExecutionUnit work_item_counts mismatch")
        return unit


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    schema_version: int
    document_type: str
    run_spec: RunSpec
    registry: Mapping[str, Any]
    selection: SelectionState
    units: tuple[ExecutionUnit, ...]
    canonical_digest: str

    @property
    def digest(self) -> str:
        return self.canonical_digest

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "document_type": self.document_type,
            "run_spec": self.run_spec.to_dict(), "registry": copy.deepcopy(dict(self.registry)),
            "selection": self.selection.to_dict(),
            "units": [unit.to_dict() for unit in self.units],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["canonical_digest"] = self.canonical_digest
        return result

    @classmethod
    def create(
        cls, *, run_spec: RunSpec, registry: Mapping[str, Any],
        selection: SelectionState, units: Sequence[ExecutionUnit],
    ) -> "ExecutionPlan":
        payload = {
            "schema_version": 1, "document_type": "stage_c_execution_plan",
            "run_spec": run_spec.to_dict(), "registry": copy.deepcopy(dict(registry)),
            "selection": selection.to_dict(),
            "units": [unit.to_dict() for unit in units],
        }
        return cls(
            1, "stage_c_execution_plan", run_spec,
            MappingProxyType(copy.deepcopy(dict(registry))), selection, tuple(units),
            _canonical_digest(payload),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionPlan":
        keys = {"schema_version", "document_type", "run_spec", "registry", "selection", "units", "canonical_digest"}
        _exact_keys(value, keys, "ExecutionPlan")
        if value["schema_version"] != 1 or value["document_type"] != "stage_c_execution_plan":
            raise RunSpecError("ExecutionPlan identity is invalid")
        if not isinstance(value["registry"], Mapping) or not isinstance(value["units"], list):
            raise RunSpecError("ExecutionPlan registry/units fields are invalid")
        plan = cls(
            1, "stage_c_execution_plan", RunSpec.from_dict(value["run_spec"]),
            MappingProxyType(copy.deepcopy(dict(value["registry"]))),
            SelectionState.from_dict(value["selection"]),
            tuple(ExecutionUnit.from_dict(item) for item in value["units"]),
            _require_sha256(value["canonical_digest"], "ExecutionPlan.canonical_digest"),
        )
        if _canonical_digest(plan._payload()) != plan.canonical_digest:
            raise RunSpecError("ExecutionPlan canonical_digest mismatch")
        if not plan.units:
            raise RunSpecError("ExecutionPlan units must not be empty")
        digests = [unit.key.canonical_digest for unit in plan.units]
        if len(digests) != len(set(digests)):
            raise RunSpecError("ExecutionPlan contains duplicate UnitKeys")
        expected_samples = set(plan.run_spec.sample_ids)
        if {unit.sample.sample_id for unit in plan.units} != expected_samples:
            raise RunSpecError("ExecutionPlan units do not cover RunSpec samples")
        if any(unit.key.method_id not in plan.run_spec.method_ids for unit in plan.units):
            raise RunSpecError("ExecutionPlan contains a method outside RunSpec")
        _exact_keys(plan.registry, {
            "schema_version", "path", "sha256", "metadata_path",
            "metadata_sha256", "registry_schema_sha256",
            "live_assets_verified",
        }, "ExecutionPlan.registry")
        return plan


def _resolve_candidates(
    spec: RunSpec, selection: SelectionState, *, has_value_methods: bool,
) -> tuple[ValueCandidate, ...]:
    source = spec.value_config_source
    if not has_value_methods:
        if source != "none":
            raise RunSpecError("class-only runs must set value_config_source='none'")
        return ()
    if source == "none":
        raise RunSpecError("value-path methods require an explicit candidate source")
    if source == "explicit":
        return spec.value_candidates
    if source == "calibration_all_144":
        return tuple(ValueCandidate(block, head) for block in range(12) for head in range(12))
    if selection.status != "present":
        raise SelectionContractError("selected value-path scope requires the C8 selection file")
    if source == "selection_primary":
        assert selection.primary is not None
        return (selection.primary,)
    if source == "selection_band_winners":
        return selection.depth_band_winners
    raise RunSpecError(f"unsupported value_config_source: {source}")


def _map_items(
    row: RegistryRow, spec: RunSpec, perturbation_ids: Sequence[str], *,
    structural_reference_only: bool,
) -> list[WorkItem]:
    result = [WorkItem(MAP_KIND, "clean", 0, None, None, None, None, None, None, None, None)]
    # Stability perturbation tensors are shared across every registered method
    # and config.  A structural raw-attention unit therefore recomputes its map
    # on all three perturbations, while remaining target-less and ineligible for
    # every fidelity curve below.
    if spec.run_stability:
        result.extend(
            WorkItem(MAP_KIND, perturbation_id, None, None, None, None, None, None, None, None, None)
            for perturbation_id in perturbation_ids
        )
    if row.audit:
        result.extend(
            WorkItem(MAP_KIND, "repeat_clean", repeat, None, None, None, None, None, None, None, None)
            for repeat in range(1, spec.audit_repeatability_runs)
        )
    return result


def _curve_items(
    *, spec: RunSpec, pairing: ValueCandidate, fidelity_eligible: bool,
) -> list[WorkItem]:
    if not fidelity_eligible:
        return []
    rankings: list[tuple[str, int | None]] = []
    if spec.run_registered_fidelity:
        rankings.extend((TARGET_RANKING, replicate) for replicate in range(5))
    if spec.run_sanity:
        rankings.extend((RANDOM_RANKING, replicate) for replicate in range(5))
        rankings.append((CENTER_RANKING, None))
    result: list[WorkItem] = []
    for scale in FIDELITY_SCALES:
        for direction in CURVE_DIRECTIONS:
            for ranking, replicate in rankings:
                for fraction_index in range(21):
                    result.append(WorkItem(
                        CURVE_POINT_KIND, "clean", None, scale, direction, ranking,
                        replicate, fraction_index, fraction_index / 20.0,
                        pairing.block_index, pairing.head_index,
                    ))
    return result


def build_execution_plan(
    config: StageCConfig, registry: RegistrySnapshot, run_spec: RunSpec, *,
    selection: SelectionState | None = None,
) -> ExecutionPlan:
    """Resolve legal method/config joins and expand every logical WorkItem."""

    if not isinstance(config, StageCConfig) or not isinstance(registry, RegistrySnapshot) or not isinstance(run_spec, RunSpec):
        raise TypeError("build_execution_plan requires StageCConfig, RegistrySnapshot and RunSpec")
    selection_state = selection or SelectionState.missing(config.selection["output_path"])
    rows_by_id = registry.by_sample_id()
    unknown_samples = set(run_spec.sample_ids) - set(rows_by_id)
    if unknown_samples:
        raise RunSpecError(f"RunSpec references unknown samples: {sorted(unknown_samples)}")
    methods_by_id = {method.id: method for method in config.methods}
    unknown_methods = set(run_spec.method_ids) - set(methods_by_id)
    if unknown_methods:
        raise RunSpecError(f"RunSpec references unknown methods: {sorted(unknown_methods)}")
    selected_rows = tuple(rows_by_id[sample_id] for sample_id in run_spec.sample_ids)
    if any(row.role == "holdout" for row in selected_rows) and selection_state.status != "present":
        raise SelectionContractError("holdout scope requires a present, validated C8 selection")
    value_methods = [methods_by_id[method_id] for method_id in run_spec.method_ids if methods_by_id[method_id].comparison_line == "value_path"]
    candidates = _resolve_candidates(run_spec, selection_state, has_value_methods=bool(value_methods))
    if run_spec.value_config_source == "calibration_all_144" and any(row.role != "calibration" for row in selected_rows):
        raise SelectionContractError("pre-C8 all-144 candidate mode is calibration-only")
    perturbation_ids = tuple(item["id"] for item in config.perturbations["items"])
    if len(perturbation_ids) != 3:
        raise RunSpecError("frozen stability plan requires exactly three perturbations")
    units: list[ExecutionUnit] = []
    for row in selected_rows:
        sample = UnitSample(row.sample_id, row.role, row.audit, row.selection_order, row.image_sha256)
        for method_id in run_spec.method_ids:
            method = methods_by_id[method_id]
            configs: tuple[ValueCandidate | None, ...] = tuple(candidates) if method.comparison_line == "value_path" else (None,)
            for candidate in configs:
                block = None if candidate is None else candidate.block_index
                head = None if candidate is None else candidate.head_index
                key = UnitKey.create(
                    sample_id=row.sample_id, method_id=method.id,
                    config_id=(
                        f"global_{method.id}" if candidate is None
                        else candidate.candidate_key
                    ),
                    block_index=block,
                    head_index=head, target_ref=method.fidelity_target_ref,
                )
                work = _map_items(
                    row, run_spec, perturbation_ids,
                    structural_reference_only=method.structural_reference_only,
                )
                if method.fidelity_eligible and (run_spec.run_registered_fidelity or run_spec.run_sanity):
                    pairing = candidate if candidate is not None else run_spec.fidelity_pairing_context
                    if pairing is None:
                        raise RunSpecError("fidelity unit lacks explicit pairing context")
                    work.extend(_curve_items(
                        spec=run_spec, pairing=pairing, fidelity_eligible=True
                    ))
                units.append(ExecutionUnit(
                    1, key, sample, method.comparison_line, method.fidelity_eligible,
                    method.structural_reference_only, tuple(work),
                ))
    digests = [unit.key.canonical_digest for unit in units]
    if len(digests) != len(set(digests)):
        raise RunSpecError("execution plan produced duplicate UnitKeys")
    registry_binding = {
        "schema_version": 1, "path": registry.path, "sha256": registry.sha256,
        "metadata_path": registry.metadata_path,
        "metadata_sha256": registry.metadata_sha256,
        "registry_schema_sha256": registry.registry_schema_sha256,
        "live_assets_verified": registry.live_assets_verified,
    }
    return ExecutionPlan.create(
        run_spec=run_spec, registry=registry_binding, selection=selection_state, units=units
    )


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return copy.deepcopy(value)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class StageCEstimate:
    schema_version: int
    document_type: str
    plan_digest: str
    counts: Mapping[str, Any]
    storage: Mapping[str, Any]
    compute: Mapping[str, Any]
    caps: Mapping[str, Any]
    violations: tuple[str, ...]
    within_budget: bool
    canonical_digest: str = field(default="", compare=True)
    _issuer: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.document_type != "stage_c_preflight_estimate":
            raise RunSpecError("StageCEstimate identity is invalid")
        _require_sha256(self.plan_digest, "StageCEstimate.plan_digest")
        _require_bool(self.within_budget, "StageCEstimate.within_budget")
        if not isinstance(self.violations, tuple) or any(
            not isinstance(item, str) or not item for item in self.violations
        ):
            raise RunSpecError("StageCEstimate violations must be non-empty strings")
        if len(self.violations) != len(set(self.violations)):
            raise RunSpecError("StageCEstimate violations must be unique")
        # A direct in-memory constructor remains useful for runner test doubles.
        # Serialized estimates go through from_dict() below, which is strict and
        # never receives the private issuer capability used for execution.
        if self._issuer is not None and self.within_budget is not (not self.violations):
            raise RunSpecError("StageCEstimate within_budget must equal not violations")
        if not all(isinstance(value, Mapping) for value in (
            self.counts, self.storage, self.compute, self.caps
        )):
            raise RunSpecError("StageCEstimate sections must be mappings")
        if self._issuer is not None or any((self.counts, self.storage, self.compute, self.caps)):
            self._validate_shape()
        object.__setattr__(self, "counts", _deep_freeze(dict(self.counts)))
        object.__setattr__(self, "storage", _deep_freeze(dict(self.storage)))
        object.__setattr__(self, "compute", _deep_freeze(dict(self.compute)))
        object.__setattr__(self, "caps", _deep_freeze(dict(self.caps)))
        payload = self._payload()
        expected_digest = _canonical_digest(payload)
        if not self.canonical_digest:
            object.__setattr__(self, "canonical_digest", expected_digest)
        elif self.canonical_digest != expected_digest:
            raise RunSpecError("StageCEstimate canonical_digest mismatch")

    def _validate_shape(self) -> None:
        count_keys = {
            "samples", "audit_samples", "execution_units",
            "fidelity_eligible_units", "structural_reference_units",
            "work_items", "map_runs", "clean_map_runs",
            "perturbation_map_runs", "repeatability_map_runs",
            "fidelity_curves", "fidelity_curve_points",
            "target_tie_curve_points", "random_sanity_curve_points",
            "center_sanity_curve_points", "logical_scalar_evaluations",
        }
        storage_keys = {
            "ordinary_files_included", "estimate_model", "L0", "L1",
            "L2", "L3", "control",
            "payload_bytes_before_control_overhead",
            "estimated_all_regular_file_bytes",
            "maximum_l0_l1_bytes_per_execution_unit",
            "maximum_l2_bytes_per_audit_sample",
        }
        compute_keys = {
            "accounting_unit", "ig_schedule_restarts", "ig_steps_schedule",
            "ig_step_chunk_size", "ig_backward_calls_per_map_worst_case",
            "ig_forward_calls_per_map_worst_case",
            "estimated_forward_calls_total", "estimated_backward_calls_total",
            "maximum_forward_calls_per_execution_unit",
            "maximum_backward_calls_per_execution_unit", "by_method",
            "measured_slowest_unit_seconds", "projected_h20_gpu_hours",
            "measured_peak_reserved_bytes_per_unit",
        }
        cap_keys = {
            "effective", "test_overrides", "config_mutated",
            "unmeasured_caps",
        }
        for value, expected, context in (
            (self.counts, count_keys, "StageCEstimate.counts"),
            (self.storage, storage_keys, "StageCEstimate.storage"),
            (self.compute, compute_keys, "StageCEstimate.compute"),
            (self.caps, cap_keys, "StageCEstimate.caps"),
        ):
            _exact_keys(value, expected, context)
        for key in count_keys:
            _require_int(self.counts[key], f"StageCEstimate.counts.{key}", minimum=0)
        if self.storage["ordinary_files_included"] is not True:
            raise RunSpecError("StageCEstimate must include every ordinary file")
        for key in (
            "payload_bytes_before_control_overhead",
            "estimated_all_regular_file_bytes",
            "maximum_l0_l1_bytes_per_execution_unit",
            "maximum_l2_bytes_per_audit_sample",
        ):
            _require_int(self.storage[key], f"StageCEstimate.storage.{key}", minimum=0)
        for level in ("L0", "L1", "L2", "L3", "control"):
            record = self.storage[level]
            if not isinstance(record, Mapping):
                raise RunSpecError(f"StageCEstimate.storage.{level} must be an object")
            _require_int(record.get("file_count"), f"StageCEstimate.storage.{level}.file_count", minimum=0)
            _require_int(record.get("estimated_bytes"), f"StageCEstimate.storage.{level}.estimated_bytes", minimum=0)
        if self.compute["accounting_unit"] != "batched_model_invocation":
            raise RunSpecError("StageCEstimate compute accounting unit is invalid")
        if self.compute["ig_schedule_restarts"] is not True:
            raise RunSpecError("StageCEstimate must count restarted IG schedules")
        for key in (
            "ig_step_chunk_size", "ig_backward_calls_per_map_worst_case",
            "ig_forward_calls_per_map_worst_case",
            "estimated_forward_calls_total", "estimated_backward_calls_total",
            "maximum_forward_calls_per_execution_unit",
            "maximum_backward_calls_per_execution_unit",
        ):
            _require_int(self.compute[key], f"StageCEstimate.compute.{key}", minimum=0)
        if not isinstance(self.compute["by_method"], Mapping):
            raise RunSpecError("StageCEstimate.compute.by_method must be an object")
        if not isinstance(self.caps["effective"], Mapping) or not isinstance(
            self.caps["test_overrides"], Mapping
        ):
            raise RunSpecError("StageCEstimate cap mappings are invalid")
        if self.caps["config_mutated"] is not False:
            raise RunSpecError("StageCEstimate may not report config mutation")
        if not isinstance(self.caps["unmeasured_caps"], (list, tuple)):
            raise RunSpecError("StageCEstimate unmeasured_caps must be an array")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "plan_digest": self.plan_digest,
            "counts": _deep_thaw(self.counts),
            "storage": _deep_thaw(self.storage),
            "compute": _deep_thaw(self.compute),
            "caps": _deep_thaw(self.caps),
            "violations": list(self.violations),
            "within_budget": self.within_budget,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["canonical_digest"] = self.canonical_digest
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageCEstimate":
        keys = {"schema_version", "document_type", "plan_digest", "counts", "storage", "compute", "caps", "violations", "within_budget", "canonical_digest"}
        _exact_keys(value, keys, "StageCEstimate")
        if not isinstance(value["violations"], list) or any(not isinstance(item, str) for item in value["violations"]):
            raise RunSpecError("StageCEstimate.violations must be an array of strings")
        if value["within_budget"] is not (not value["violations"]):
            raise RunSpecError("StageCEstimate within_budget must equal not violations")
        result = cls(
            value["schema_version"], value["document_type"], value["plan_digest"],
            value["counts"], value["storage"], value["compute"], value["caps"],
            tuple(value["violations"]), value["within_budget"],
            _require_sha256(value["canonical_digest"], "StageCEstimate.canonical_digest"),
        )
        result._validate_shape()
        return result


_L0_BASE_BYTES_PER_UNIT = 32 * 1024
_L0_BYTES_PER_WORK_ITEM = 192
_L1_CONTAINER_BYTES_PER_UNIT = 8 * 1024
_L1_FLOAT32_PATCH_MAP_BYTES = 14 * 14 * 4
_L2_UNIT_METADATA_BYTES = 64 * 1024
_L2_SHARED_ATTENTION_BYTES_PER_AUDIT_SAMPLE = 12 * 12 * 197 * 197 * 4
_RAW_RGB_FLOAT32_BYTES = 3 * 224 * 224 * 4
_L3_WORST_CASE_BYTES_PER_UNIT = 224 * 224 * 4 + 4096


def _unit_compute(
    unit: ExecutionUnit, *, ig_backward_per_map: int, ig_forward_per_map: int,
) -> tuple[int, int, dict[str, int]]:
    counts = unit.work_item_counts
    map_runs = counts["map_runs"]
    curve_points = counts["fidelity_curve_points"]
    method = unit.key.method_id
    if method in {
        "conditional_fixed_clean_attention_head_direction_value_path_ig",
        "full_model_class_logit_ig",
    }:
        map_forward, map_backward = ig_forward_per_map, ig_backward_per_map
    elif method == "vit_grad_cam_block11_norm1":
        map_forward, map_backward = 1, 1
    elif method == "matched_fixed_clean_attention_patch_token_contribution":
        map_forward, map_backward = 2, 0
    else:
        map_forward, map_backward = 1, 0
    # The production backend performs shared extraction and class-target
    # resolution for every unit.  Value-path target resolution conservatively
    # requires natural clean/baseline endpoints plus clean/baseline fixed-A
    # replays.  Fidelity target normalization adds two endpoint evaluations.
    # Stability records one clean natural prediction and one per perturbation.
    # Shared extraction is one model invocation.  Targeted units also resolve
    # the clean class once; Value-path scalar units additionally resolve their
    # frozen clean-A/direction with four conservative endpoint/replay calls.
    # The structural raw-attention reference owns no scalar target.
    setup_forward = 1
    if unit.key.target_ref is not None:
        setup_forward += 1
    if unit.comparison_line == "value_path" and unit.key.target_ref is not None:
        setup_forward += 4
    fidelity_endpoint_forward = 2 if curve_points else 0
    stability_prediction_forward = (
        1 + counts["perturbation_map_runs"]
        if counts["perturbation_map_runs"] else 0
    )
    forward = (
        setup_forward + fidelity_endpoint_forward
        + stability_prediction_forward + map_runs * map_forward + curve_points
    )
    backward = map_runs * map_backward
    return forward, backward, {
        "units": 1, "map_runs": map_runs, "curve_points": curve_points,
        "setup_forward_calls": setup_forward,
        "fidelity_endpoint_forward_calls": fidelity_endpoint_forward,
        "stability_prediction_forward_calls": stability_prediction_forward,
        "batched_forward_calls": forward, "batched_backward_calls": backward,
    }


def _effective_caps(config: StageCConfig, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    frozen = config.budget.to_dict()
    if overrides is None:
        return frozen
    if not isinstance(overrides, Mapping):
        raise TypeError("cap_overrides must be a mapping or None")
    unknown = set(overrides) - _OVERRIDABLE_HARD_CAPS
    if unknown:
        raise RunSpecError(
            "cap_overrides may contain only hard-cap fields; "
            f"unknown or non-cap fields: {sorted(unknown)}"
        )
    result = dict(frozen)
    for key, value in overrides.items():
        original = frozen[key]
        if isinstance(original, bool) or not isinstance(original, (int, float)):
            raise RunSpecError(f"{key} is not a numeric cap and cannot be overridden")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise RunSpecError(f"test cap override {key} must be a positive finite number")
        if value > original:
            raise RunSpecError(f"test cap override {key} cannot loosen the frozen cap")
        if type(original) is int and type(value) is not int:
            raise RunSpecError(f"test cap override {key} must remain an integer")
        result[key] = value
    return result


def estimate_stage_c_run(
    config: StageCConfig, plan: ExecutionPlan, *,
    cap_overrides: Mapping[str, Any] | None = None,
    measured_slowest_unit_seconds: float | None = None,
    measured_peak_reserved_bytes_per_unit: int | None = None,
) -> StageCEstimate:
    """Estimate logical work, batched calls, every ordinary-file class and caps.

    ``cap_overrides`` exists solely for boundary tests.  Values may only tighten
    numeric caps; the immutable config is never modified.
    """

    if not isinstance(config, StageCConfig) or not isinstance(plan, ExecutionPlan):
        raise TypeError("estimate_stage_c_run requires StageCConfig and ExecutionPlan")
    caps = _effective_caps(config, cap_overrides)
    if measured_slowest_unit_seconds is not None and (
        isinstance(measured_slowest_unit_seconds, bool)
        or not isinstance(measured_slowest_unit_seconds, (int, float))
        or not math.isfinite(float(measured_slowest_unit_seconds))
        or measured_slowest_unit_seconds <= 0
    ):
        raise RunSpecError("measured_slowest_unit_seconds must be positive and finite")
    if measured_peak_reserved_bytes_per_unit is not None:
        _require_int(measured_peak_reserved_bytes_per_unit, "measured_peak_reserved_bytes_per_unit", minimum=0)
    aggregate_counts: Counter[str] = Counter()
    for unit in plan.units:
        aggregate_counts.update(unit.work_item_counts)
    sample_ids = {unit.sample.sample_id for unit in plan.units}
    audit_sample_ids = {unit.sample.sample_id for unit in plan.units if unit.audit}
    fidelity_units = sum(unit.fidelity_eligible for unit in plan.units)
    structural_units = sum(unit.structural_reference_only for unit in plan.units)
    counts: dict[str, Any] = {
        "samples": len(sample_ids), "audit_samples": len(audit_sample_ids),
        "execution_units": len(plan.units), "fidelity_eligible_units": fidelity_units,
        "structural_reference_units": structural_units,
        **{key: int(value) for key, value in aggregate_counts.items()},
    }
    schedule = tuple(config.targets["ig"]["steps_schedule"])
    chunk = config.targets["ig"]["step_chunk_size"]
    ig_backward = sum(math.ceil(steps / chunk) for steps in schedule)
    ig_forward = ig_backward + 4  # two IG endpoints plus wrapper endpoint binding
    per_method: dict[str, Counter[str]] = defaultdict(Counter)
    per_unit_calls: dict[str, tuple[int, int]] = {}
    total_forward = total_backward = 0
    for unit in plan.units:
        forward, backward, detail = _unit_compute(
            unit, ig_backward_per_map=ig_backward, ig_forward_per_map=ig_forward
        )
        total_forward += forward
        total_backward += backward
        per_unit_calls[unit.key.canonical_digest] = (forward, backward)
        per_method[unit.key.method_id].update(detail)
    max_forward = max((value[0] for value in per_unit_calls.values()), default=0)
    max_backward = max((value[1] for value in per_unit_calls.values()), default=0)
    gpu_hours = None if measured_slowest_unit_seconds is None else (
        float(measured_slowest_unit_seconds) * len(plan.units)
        * float(caps["projection_safety_factor"]) / 3600.0
    )
    compute = {
        "accounting_unit": "batched_model_invocation",
        "ig_schedule_restarts": True, "ig_steps_schedule": list(schedule),
        "ig_step_chunk_size": chunk,
        "ig_backward_calls_per_map_worst_case": ig_backward,
        "ig_forward_calls_per_map_worst_case": ig_forward,
        "estimated_forward_calls_total": total_forward,
        "estimated_backward_calls_total": total_backward,
        "maximum_forward_calls_per_execution_unit": max_forward,
        "maximum_backward_calls_per_execution_unit": max_backward,
        "by_method": {method: dict(values) for method, values in sorted(per_method.items())},
        "measured_slowest_unit_seconds": measured_slowest_unit_seconds,
        "projected_h20_gpu_hours": gpu_hours,
        "measured_peak_reserved_bytes_per_unit": measured_peak_reserved_bytes_per_unit,
    }
    l0_by_unit: dict[str, int] = {}
    l1_by_unit: dict[str, int] = {}
    l2_by_sample: Counter[str] = Counter()
    for unit in plan.units:
        work_counts = unit.work_item_counts
        digest = unit.key.canonical_digest
        l0_by_unit[digest] = _L0_BASE_BYTES_PER_UNIT + work_counts["work_items"] * _L0_BYTES_PER_WORK_ITEM
        l1_by_unit[digest] = _L1_CONTAINER_BYTES_PER_UNIT + work_counts["map_runs"] * _L1_FLOAT32_PATCH_MAP_BYTES
        if unit.audit:
            raw_per_map = _RAW_RGB_FLOAT32_BYTES if unit.key.method_id in {
                "conditional_fixed_clean_attention_head_direction_value_path_ig",
                "full_model_class_logit_ig",
            } else _L1_FLOAT32_PATCH_MAP_BYTES
            l2_by_sample[unit.sample.sample_id] += (
                _L2_UNIT_METADATA_BYTES + work_counts["map_runs"] * raw_per_map
            )
    for sample_id in audit_sample_ids:
        l2_by_sample[sample_id] += _L2_SHARED_ATTENTION_BYTES_PER_AUDIT_SAMPLE
    l0_bytes = sum(l0_by_unit.values())
    l1_bytes = sum(l1_by_unit.values())
    l2_bytes = sum(l2_by_sample.values())
    l3_bytes = len(plan.units) * _L3_WORST_CASE_BYTES_PER_UNIT
    payload_bytes = l0_bytes + l1_bytes + l2_bytes + l3_bytes
    total_bytes = math.ceil(payload_bytes * float(caps["storage_control_overhead_factor"]))
    control_bytes = total_bytes - payload_bytes
    max_l0_l1 = max(
        (l0_by_unit[digest] + l1_by_unit[digest] for digest in l0_by_unit),
        default=0,
    )
    max_l2 = max(l2_by_sample.values(), default=0)
    map_file_count = sum(unit.work_item_counts["map_runs"] for unit in plan.units)
    # One result row and one provenance document are retained per successful
    # unit.  L3 uses an all-units failure upper bound; L2 is conservatively one
    # heterogeneous NPZ per audit unit plus one shared-attention file per audit
    # sample.  Manifest and status are ordinary control files too.
    l0_file_count = 2 * len(plan.units)
    l2_file_count = sum(unit.audit for unit in plan.units) + len(audit_sample_ids)
    unit_control_file_count = 2 * len(plan.units)
    batch_control_file_count = 4
    storage = {
        "ordinary_files_included": True,
        "estimate_model": {
            "l0_base_bytes_per_unit": _L0_BASE_BYTES_PER_UNIT,
            "l0_bytes_per_work_item": _L0_BYTES_PER_WORK_ITEM,
            "l1_float32_patch_map_bytes": _L1_FLOAT32_PATCH_MAP_BYTES,
            "l2_shared_attention_bytes_per_audit_sample": _L2_SHARED_ATTENTION_BYTES_PER_AUDIT_SAMPLE,
            "l3_policy": "one_uncompressed_rgba_equivalent_for_every_unit_failure_worst_case",
            "control_overhead_factor": caps["storage_control_overhead_factor"],
        },
        "L0": {"file_count": l0_file_count, "estimated_bytes": l0_bytes},
        "L1": {"file_count": map_file_count, "estimated_bytes": l1_bytes},
        "L2": {
            "file_count": l2_file_count,
            "estimated_bytes": l2_bytes,
            "by_audit_sample": dict(sorted(l2_by_sample.items())),
        },
        "L3": {
            "file_count": len(plan.units), "estimated_bytes": l3_bytes,
            "scope": "audit_or_explicit_failure_with_all_units_as_failure_upper_bound",
        },
        "control": {
            "file_count": unit_control_file_count + batch_control_file_count,
            "estimated_bytes": control_bytes,
            "unit_manifest_files": len(plan.units),
            "unit_status_files": len(plan.units),
            "batch_control_files": batch_control_file_count,
        },
        "payload_bytes_before_control_overhead": payload_bytes,
        "estimated_all_regular_file_bytes": total_bytes,
        "maximum_l0_l1_bytes_per_execution_unit": max_l0_l1,
        "maximum_l2_bytes_per_audit_sample": max_l2,
    }
    violations: list[str] = []
    if max_l0_l1 > caps["bytes_per_l0_l1_execution_unit_max"]:
        violations.append("bytes_per_l0_l1_execution_unit_max")
    if max_l2 > caps["bytes_per_audit_sample_l2_max"]:
        violations.append("bytes_per_audit_sample_l2_max")
    if total_bytes > caps["stage_c_c8_c9_total_artifact_bytes_max"]:
        violations.append("stage_c_c8_c9_total_artifact_bytes_max")
    if max_forward > caps["forward_calls_per_execution_unit_max"]:
        violations.append("forward_calls_per_execution_unit_max")
    if max_backward > caps["backward_calls_per_execution_unit_max"]:
        violations.append("backward_calls_per_execution_unit_max")
    if measured_peak_reserved_bytes_per_unit is not None and measured_peak_reserved_bytes_per_unit > caps["cuda_peak_reserved_bytes_per_execution_unit_max"]:
        violations.append("cuda_peak_reserved_bytes_per_execution_unit_max")
    if gpu_hours is not None and gpu_hours > caps["stage_c_c8_c9_h20_gpu_hours_max"]:
        violations.append("stage_c_c8_c9_h20_gpu_hours_max")
    cap_report = {
        "effective": {key: value for key, value in caps.items() if isinstance(value, (int, float)) and not isinstance(value, bool)},
        "test_overrides": {} if cap_overrides is None else dict(cap_overrides),
        "config_mutated": False,
        "unmeasured_caps": [
            key for key, measured in (
                ("cuda_peak_reserved_bytes_per_execution_unit_max", measured_peak_reserved_bytes_per_unit),
                ("stage_c_c8_c9_h20_gpu_hours_max", gpu_hours),
            ) if measured is None
        ],
    }
    return StageCEstimate(
        1, "stage_c_preflight_estimate", plan.canonical_digest, counts, storage,
        compute, cap_report, tuple(violations), not violations, "",
        _ESTIMATE_ISSUER,
    )


def enforce_stage_c_budget(estimate: StageCEstimate) -> None:
    """Fail before model/GPU initialization or artifact-writer construction."""

    if not isinstance(estimate, StageCEstimate):
        raise TypeError("estimate must be a StageCEstimate")
    if estimate._issuer is not _ESTIMATE_ISSUER:
        raise RunSpecError(
            "only an estimate freshly computed by estimate_stage_c_run may "
            "authorize execution"
        )
    if _canonical_digest(estimate._payload()) != estimate.canonical_digest:
        raise RunSpecError("StageCEstimate canonical digest is invalid")
    computed_violations: list[str] = []
    effective = estimate.caps.get("effective")
    if not isinstance(effective, Mapping):
        raise RunSpecError("StageCEstimate caps.effective is missing")
    comparisons = (
        (
            estimate.storage.get("maximum_l0_l1_bytes_per_execution_unit"),
            "bytes_per_l0_l1_execution_unit_max",
        ),
        (
            estimate.storage.get("maximum_l2_bytes_per_audit_sample"),
            "bytes_per_audit_sample_l2_max",
        ),
        (
            estimate.storage.get("estimated_all_regular_file_bytes"),
            "stage_c_c8_c9_total_artifact_bytes_max",
        ),
        (
            estimate.compute.get("maximum_forward_calls_per_execution_unit"),
            "forward_calls_per_execution_unit_max",
        ),
        (
            estimate.compute.get("maximum_backward_calls_per_execution_unit"),
            "backward_calls_per_execution_unit_max",
        ),
        (
            estimate.compute.get("measured_peak_reserved_bytes_per_unit"),
            "cuda_peak_reserved_bytes_per_execution_unit_max",
        ),
        (
            estimate.compute.get("projected_h20_gpu_hours"),
            "stage_c_c8_c9_h20_gpu_hours_max",
        ),
    )
    for observed, cap_name in comparisons:
        if observed is None:
            continue
        cap = effective.get(cap_name)
        if (
            isinstance(observed, bool) or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed)) or observed < 0
            or isinstance(cap, bool) or not isinstance(cap, (int, float))
            or not math.isfinite(float(cap)) or cap <= 0
        ):
            raise RunSpecError(
                f"StageCEstimate has invalid observed/cap values for {cap_name}"
            )
        if observed > cap:
            computed_violations.append(cap_name)
    if tuple(computed_violations) != estimate.violations:
        raise RunSpecError(
            "StageCEstimate violations do not match its cap comparisons"
        )
    if not estimate.within_budget:
        raise BudgetExceededError(
            "Stage C preflight exceeds frozen caps: " + ", ".join(estimate.violations)
        )


__all__ = [
    "AUDIT_REPEATABILITY_SCOPES", "BudgetExceededError",
    "CALIBRATION_SAMPLE_IDS", "DEFAULT_REGISTRY_SCHEMA",
    "DEFAULT_SELECTION_SCHEMA", "ExecutionPlan", "ExecutionUnit",
    "RegistryContractError", "RegistryRow", "RegistrySnapshot",
    "RunSpec", "RunSpecError", "SelectionContractError",
    "SelectionState", "StageCEstimate", "StageCPlanError", "UnitKey",
    "UnitSample", "VALUE_CONFIG_SOURCES", "ValueCandidate", "WorkItem",
    "build_execution_plan", "enforce_stage_c_budget",
    "estimate_stage_c_run", "load_run_spec", "load_stage_c_registry",
    "load_stage_c_selection", "make_run_spec", "unit_key_digest",
    "validate_live_registry_assets",
]
