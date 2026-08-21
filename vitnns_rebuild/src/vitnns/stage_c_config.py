"""Strict, immutable loader for the frozen Stage C protocol config.

Stage C version 1 deliberately uses an exact-instance JSON Schema.  This
loader enforces that closed instance with type-aware comparison (so ``True``
cannot pass for integer ``1``), then checks the relationships that JSON Schema
cannot express.  It never supplies a default and its ``to_dict`` output is the
exact document that was loaded.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from .constants import REPO_ROOT


DEFAULT_STAGE_C_CONFIG = REPO_ROOT / "config/stage_c.json"
DEFAULT_STAGE_C_SCHEMA = REPO_ROOT / "config/stage_c.schema.json"
TOP_LEVEL_KEYS = (
    "$schema", "schema_version", "document_type", "stage",
    "protocol", "registry", "roles", "methods", "targets",
    "layers_heads", "projection", "metrics", "fidelity",
    "perturbations", "retention", "budget", "selection",
    "statistics", "decision_gates",
)


class StageCConfigError(ValueError):
    """Raised when a Stage C config fails its closed production contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StageCConfigError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_token(token: str) -> None:
    raise StageCConfigError(f"Non-finite JSON number is not allowed: {token}")


def strict_json_load(path: Path | str) -> Any:
    """Parse UTF-8 JSON while rejecting extensions and duplicate keys."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    try:
        return json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_token,
        )
    except UnicodeDecodeError as exc:
        raise StageCConfigError(f"Stage C JSON must be UTF-8: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise StageCConfigError(f"Invalid JSON in {resolved}: {exc}") from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class FrozenSection(Mapping[str, Any]):
    """Read-only typed wrapper for one closed config object."""

    _data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self._data, Mapping):
            raise TypeError("FrozenSection requires a mapping")
        object.__setattr__(self, "_data", _freeze(dict(self._data)))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._data)


@dataclass(frozen=True)
class ProtocolConfig:
    path: str
    version: str
    amendment_policy: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtocolConfig":
        return cls(
            path=value["path"], version=value["version"],
            amendment_policy=value["amendment_policy"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "version": self.version,
            "amendment_policy": self.amendment_policy,
        }


@dataclass(frozen=True)
class RegistryConfig:
    path: str
    sha256: str
    metadata_path: str
    metadata_sha256: str
    source_manifest_path: str
    source_manifest_sha256: str
    source_manifest_metadata_path: str
    source_manifest_metadata_sha256: str
    official_devkit_sha256: str
    expected_counts: FrozenSection

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistryConfig":
        return cls(
            path=value["path"], sha256=value["sha256"],
            metadata_path=value["metadata_path"],
            metadata_sha256=value["metadata_sha256"],
            source_manifest_path=value["source_manifest_path"],
            source_manifest_sha256=value["source_manifest_sha256"],
            source_manifest_metadata_path=value["source_manifest_metadata_path"],
            source_manifest_metadata_sha256=value["source_manifest_metadata_sha256"],
            official_devkit_sha256=value["official_devkit_sha256"],
            expected_counts=FrozenSection(value["expected_counts"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "sha256": self.sha256,
            "metadata_path": self.metadata_path,
            "metadata_sha256": self.metadata_sha256,
            "source_manifest_path": self.source_manifest_path,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_manifest_metadata_path": self.source_manifest_metadata_path,
            "source_manifest_metadata_sha256": self.source_manifest_metadata_sha256,
            "official_devkit_sha256": self.official_devkit_sha256,
            "expected_counts": self.expected_counts.to_dict(),
        }


@dataclass(frozen=True)
class MethodConfig:
    id: str
    exact_name: str
    comparison_line: str
    explanation_target_ref: str | None
    fidelity_target_ref: str | None
    class_agnostic: bool
    fidelity_eligible: bool
    structural_reference_only: bool
    native_map: str
    projection_ref: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MethodConfig":
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "exact_name": self.exact_name,
            "comparison_line": self.comparison_line,
            "explanation_target_ref": self.explanation_target_ref,
            "fidelity_target_ref": self.fidelity_target_ref,
            "class_agnostic": self.class_agnostic,
            "fidelity_eligible": self.fidelity_eligible,
            "structural_reference_only": self.structural_reference_only,
            "native_map": self.native_map, "projection_ref": self.projection_ref,
        }


@dataclass(frozen=True)
class StageCConfig:
    schema_ref: str
    schema_version: int
    document_type: str
    stage: str
    protocol: ProtocolConfig
    registry: RegistryConfig
    roles: FrozenSection
    methods: tuple[MethodConfig, ...]
    targets: FrozenSection
    layers_heads: FrozenSection
    projection: FrozenSection
    metrics: FrozenSection
    fidelity: FrozenSection
    perturbations: FrozenSection
    retention: FrozenSection
    budget: FrozenSection
    selection: FrozenSection
    statistics: FrozenSection
    decision_gates: FrozenSection

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        schema_path: Path | str = DEFAULT_STAGE_C_SCHEMA,
    ) -> "StageCConfig":
        if not isinstance(value, Mapping):
            raise StageCConfigError("Stage C config must be a JSON object")
        plain = copy.deepcopy(dict(value))
        _validate_against_exact_schema(
            plain, schema_path=Path(schema_path), reference_path=DEFAULT_STAGE_C_CONFIG
        )
        _validate_semantics(plain)
        return cls(
            schema_ref=plain["$schema"],
            schema_version=plain["schema_version"],
            document_type=plain["document_type"], stage=plain["stage"],
            protocol=ProtocolConfig.from_dict(plain["protocol"]),
            registry=RegistryConfig.from_dict(plain["registry"]),
            roles=FrozenSection(plain["roles"]),
            methods=tuple(MethodConfig.from_dict(item) for item in plain["methods"]),
            targets=FrozenSection(plain["targets"]),
            layers_heads=FrozenSection(plain["layers_heads"]),
            projection=FrozenSection(plain["projection"]),
            metrics=FrozenSection(plain["metrics"]),
            fidelity=FrozenSection(plain["fidelity"]),
            perturbations=FrozenSection(plain["perturbations"]),
            retention=FrozenSection(plain["retention"]),
            budget=FrozenSection(plain["budget"]),
            selection=FrozenSection(plain["selection"]),
            statistics=FrozenSection(plain["statistics"]),
            decision_gates=FrozenSection(plain["decision_gates"]),
        )

    @property
    def method_by_id(self) -> Mapping[str, MethodConfig]:
        return MappingProxyType({method.id: method for method in self.methods})

    def to_dict(self) -> dict[str, Any]:
        return {
            "$schema": self.schema_ref, "schema_version": self.schema_version,
            "document_type": self.document_type, "stage": self.stage,
            "protocol": self.protocol.to_dict(),
            "registry": self.registry.to_dict(), "roles": self.roles.to_dict(),
            "methods": [method.to_dict() for method in self.methods],
            "targets": self.targets.to_dict(),
            "layers_heads": self.layers_heads.to_dict(),
            "projection": self.projection.to_dict(),
            "metrics": self.metrics.to_dict(), "fidelity": self.fidelity.to_dict(),
            "perturbations": self.perturbations.to_dict(),
            "retention": self.retention.to_dict(), "budget": self.budget.to_dict(),
            "selection": self.selection.to_dict(),
            "statistics": self.statistics.to_dict(),
            "decision_gates": self.decision_gates.to_dict(),
        }


def _canonical_from_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise StageCConfigError("Stage C schema must declare Draft 2020-12")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise StageCConfigError("Stage C schema root must be a closed object")
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list) or not isinstance(properties, Mapping):
        raise StageCConfigError("Stage C schema is missing required/properties")
    if set(required) != set(TOP_LEVEL_KEYS) or set(properties) != set(TOP_LEVEL_KEYS):
        raise StageCConfigError("Stage C schema top-level fields do not match the contract")
    canonical: dict[str, Any] = {}
    for key in TOP_LEVEL_KEYS:
        definition = properties[key]
        if not isinstance(definition, Mapping) or set(definition) != {"const"}:
            raise StageCConfigError(f"Stage C schema property {key!r} must be exact const")
        canonical[key] = copy.deepcopy(definition["const"])
    return canonical


def _same_scalar_type(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if type(expected) is bool:
        return type(actual) is bool
    if type(expected) is int:
        return type(actual) is int
    if type(expected) is float:
        return type(actual) is float
    return type(actual) is type(expected)


def _exact_match(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise StageCConfigError(f"{context} must be an object")
        missing = set(expected) - set(actual)
        unknown = set(actual) - set(expected)
        if missing:
            raise StageCConfigError(f"{context} has missing fields: {sorted(missing)}")
        if unknown:
            raise StageCConfigError(f"{context} has unknown fields: {sorted(unknown)}")
        if list(actual) != list(expected):
            # Object order is not scientifically meaningful, so preserve and
            # accept it; array order remains strict below.
            pass
        for key in expected:
            _exact_match(actual[key], expected[key], f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise StageCConfigError(f"{context} must be an array")
        if len(actual) != len(expected):
            raise StageCConfigError(
                f"{context} length must be {len(expected)}, got {len(actual)}"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _exact_match(actual_item, expected_item, f"{context}[{index}]")
        return
    if not _same_scalar_type(actual, expected):
        raise StageCConfigError(
            f"{context} has wrong type: expected {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )
    if isinstance(actual, float) and not math.isfinite(actual):
        raise StageCConfigError(f"{context} must be finite")
    if actual != expected:
        raise StageCConfigError(
            f"{context} violates the frozen enum/range/value/order contract: "
            f"expected {expected!r}, got {actual!r}"
        )


def _validate_against_exact_schema(
    value: Mapping[str, Any], *, schema_path: Path, reference_path: Path
) -> None:
    schema = strict_json_load(schema_path)
    if not isinstance(schema, Mapping):
        raise StageCConfigError("Stage C schema must be a JSON object")
    canonical = _canonical_from_schema(schema)
    # JSON Schema numeric equality intentionally considers ``0`` and ``0.0``
    # equal.  The checked-in JSON document is additionally the lexical/type
    # authority for this frozen exact instance (for example it distinguishes
    # integer counts from floating tolerances).  Verify that authority against
    # the schema, then use it for the strict type-aware production check.
    frozen_reference = strict_json_load(reference_path)
    _schema_const_match(frozen_reference, canonical, "frozen config/schema")
    _exact_match(value, frozen_reference, "config")


def _schema_const_match(actual: Any, expected: Any, context: str) -> None:
    """Implement JSON-Schema const equality without Python's bool==int leak."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise StageCConfigError(f"{context} object differs from its schema const")
        for key in expected:
            _schema_const_match(actual[key], expected[key], f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise StageCConfigError(f"{context} array differs from its schema const")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _schema_const_match(actual_item, expected_item, f"{context}[{index}]")
        return
    if type(actual) is bool or type(expected) is bool:
        equal = type(actual) is type(expected) and actual == expected
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        equal = math.isfinite(float(actual)) and math.isfinite(float(expected)) and actual == expected
    else:
        equal = type(actual) is type(expected) and actual == expected
    if not equal:
        raise StageCConfigError(
            f"{context} differs from schema const: {actual!r} != {expected!r}"
        )


def _safe_relative(raw: str, context: str) -> None:
    if not isinstance(raw, str) or not raw:
        raise StageCConfigError(f"{context} must be a non-empty POSIX path")
    if "\\" in raw:
        raise StageCConfigError(f"{context} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise StageCConfigError(f"{context} must be normalized and repository-relative")
    if str(path) != raw:
        raise StageCConfigError(f"{context} must be a normalized POSIX path")


def _unique(values: Sequence[Any], context: str) -> None:
    if len(values) != len(set(values)):
        raise StageCConfigError(f"{context} must contain unique items")


def _strictly_increasing_ints(values: Sequence[Any], context: str) -> None:
    if any(type(item) is not int for item in values):
        raise StageCConfigError(f"{context} must contain integers (not booleans)")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise StageCConfigError(f"{context} must be unique and strictly increasing")


def _validate_semantics(config: Mapping[str, Any]) -> None:
    # Path/reference safety.  Paths are references; existence is intentionally
    # checked by C0/C6 asset validators rather than silently resolved here.
    _safe_relative(config["protocol"]["path"], "protocol.path")
    registry = config["registry"]
    for key in (
        "path", "metadata_path", "source_manifest_path",
        "source_manifest_metadata_path",
    ):
        _safe_relative(registry[key], f"registry.{key}")
    _safe_relative(config["layers_heads"]["holdout_value_path_configs_source"], "layers_heads.holdout_value_path_configs_source")
    _safe_relative(config["selection"]["output_path"], "selection.output_path")

    methods = config["methods"]
    method_ids = [method["id"] for method in methods]
    _unique(method_ids, "methods.id")
    by_method = {method["id"]: method for method in methods}
    target_ids = {
        "fixed_clean_attention_cls_head_direction",
        "clean_predicted_class_raw_logit",
    }
    projection_ids = {
        "raw_cls_attention_to_patch", "pixel_mass_to_patch_sum",
        "patch_mass_to_pixel_uniform", "rollout_cls_to_patch",
        "identity_14x14",
    }
    for index, method in enumerate(methods):
        for field in ("class_agnostic", "fidelity_eligible", "structural_reference_only"):
            if type(method[field]) is not bool:
                raise StageCConfigError(f"methods[{index}].{field} must be a boolean")
        for field in ("explanation_target_ref", "fidelity_target_ref"):
            reference = method[field]
            if reference is not None and reference not in target_ids:
                raise StageCConfigError(f"methods[{index}].{field} is unresolved: {reference!r}")
        if method["projection_ref"] not in projection_ids:
            raise StageCConfigError(
                f"methods[{index}].projection_ref is unresolved: {method['projection_ref']!r}"
            )
        if method["class_agnostic"] and method["explanation_target_ref"] == "clean_predicted_class_raw_logit":
            raise StageCConfigError("class-agnostic methods cannot claim class explanations")
        if not method["fidelity_eligible"] and method["fidelity_target_ref"] is not None:
            raise StageCConfigError("fidelity-ineligible method cannot name a fidelity target")
        if "complete attention attribution" in method["exact_name"].lower():
            raise StageCConfigError("no Stage C method may claim complete attention attribution")

    fidelity = config["fidelity"]
    if fidelity["replacement_baseline_ref"] != config["targets"]["baseline"]["id"]:
        raise StageCConfigError("fidelity.replacement_baseline_ref is unresolved")
    for field in ("value_path_pair", "class_logit_methods"):
        _unique(fidelity[field], f"fidelity.{field}")
        unknown = set(fidelity[field]) - set(method_ids)
        if unknown:
            raise StageCConfigError(f"fidelity.{field} has unresolved methods: {sorted(unknown)}")
    for method_id in fidelity["class_logit_methods"]:
        if by_method[method_id]["comparison_line"] != "class_logit":
            raise StageCConfigError("fidelity.class_logit_methods crosses comparison lines")
    for method_id in fidelity["value_path_pair"]:
        method = by_method[method_id]
        if method["comparison_line"] != "value_path" or method["fidelity_target_ref"] != "fixed_clean_attention_cls_head_direction":
            raise StageCConfigError("fidelity.value_path_pair must share the Value-path target")

    layers = config["layers_heads"]
    for field in ("calibration_blocks", "calibration_heads", "rollout_blocks"):
        _strictly_increasing_ints(layers[field], f"layers_heads.{field}")
    flattened_bands = [item for band in layers["depth_bands"] for item in band]
    if flattened_bands != layers["calibration_blocks"]:
        raise StageCConfigError("depth_bands must be ordered, disjoint, and cover all calibration blocks")
    if layers["tokens"] != layers["patch_tokens"] + 1:
        raise StageCConfigError("tokens must equal patch_tokens plus CLS")

    projection = config["projection"]
    if projection["input_height"] != projection["patch_height"] * projection["metric_grid_height"] or projection["input_width"] != projection["patch_width"] * projection["metric_grid_width"]:
        raise StageCConfigError("projection grid geometry is inconsistent")
    if projection["metric_ready_dtype"] != config["retention"]["l1_storage_dtype"]:
        raise StageCConfigError("metric-ready and L1 storage dtype must agree")

    metrics = config["metrics"]
    for field in ("required", "correlation_views", "distribution_views", "na_reasons", "invalidating_na_reasons"):
        _unique(metrics[field], f"metrics.{field}")
    if not set(metrics["invalidating_na_reasons"]) <= set(metrics["na_reasons"]):
        raise StageCConfigError("invalidating NA reasons must be declared NA reasons")
    if metrics["js_epsilon"] != 0 or metrics["kl_epsilon"] <= 0:
        raise StageCConfigError("JS must use no epsilon and KL must use positive epsilon")

    fractions = fidelity["curve_fractions"]
    if len(fractions) != fidelity["curve_fraction_count"]:
        raise StageCConfigError("curve_fraction_count disagrees with curve_fractions")
    if fractions[0] != 0 or fractions[-1] != 1 or any(right <= left for left, right in zip(fractions, fractions[1:])):
        raise StageCConfigError("curve_fractions must be strictly increasing from 0 to 1")
    if fidelity["patch_level"]["unit_count"] != projection["metric_grid_height"] * projection["metric_grid_width"]:
        raise StageCConfigError("patch fidelity unit count disagrees with projection grid")
    if fidelity["fine_level"]["unit_count"] != fidelity["patch_level"]["unit_count"] * fidelity["fine_level"]["cells_per_patch"]:
        raise StageCConfigError("fine fidelity unit count is inconsistent")

    perturbations = config["perturbations"]["items"]
    perturbation_ids = [item["id"] for item in perturbations]
    _unique(perturbation_ids, "perturbations.items.id")
    if set(config["decision_gates"]["stability_perturbations"]) != set(perturbation_ids):
        raise StageCConfigError("decision stability perturbation references are unresolved")
    for item in perturbations:
        _strictly_increasing_ints(item["fully_valid_patch_rows"], f"perturbation {item['id']} rows")
        _strictly_increasing_ints(item["fully_valid_patch_cols"], f"perturbation {item['id']} cols")
        if any(index < 0 or index >= layers["patch_tokens"] ** 0.5 for index in item["fully_valid_patch_rows"] + item["fully_valid_patch_cols"]):
            raise StageCConfigError(f"perturbation {item['id']} has an out-of-range patch index")

    counts = registry["expected_counts"]
    roles = config["roles"]
    statistics = config["statistics"]
    gates = config["decision_gates"]
    if counts["total"] != counts["calibration"] + counts["holdout"]:
        raise StageCConfigError("registry role counts do not sum to total")
    if statistics["confirmatory_n"] != counts["holdout"] or statistics["calibration_n"] != counts["calibration"]:
        raise StageCConfigError("statistics n values disagree with registry counts")
    if gates["primary_holdout_required_complete"] != statistics["confirmatory_n"]:
        raise StageCConfigError("validity denominator disagrees with confirmatory n")
    if roles["calibration_confirmatory_statistics"] or statistics["calibration_in_confirmatory_statistics"]:
        raise StageCConfigError("calibration must not enter confirmatory statistics")
    if gates["cost_caps_ref"] != "budget":
        raise StageCConfigError("decision_gates.cost_caps_ref is unresolved")
    if gates["patch_support_minimum"] != config["targets"]["fixed_clean_attention_cls_head_direction"]["patch_support_minimum"]:
        raise StageCConfigError("patch support thresholds disagree across sections")
    if gates["positive_patch_evidence_minimum_fraction_of_token_l1"] != config["targets"]["fixed_clean_attention_cls_head_direction"]["positive_patch_evidence_minimum_fraction_of_token_l1"]:
        raise StageCConfigError("positive patch evidence thresholds disagree")


def load_stage_c_config(
    path: Path | str = DEFAULT_STAGE_C_CONFIG,
    *,
    schema_path: Path | str = DEFAULT_STAGE_C_SCHEMA,
) -> StageCConfig:
    value = strict_json_load(path)
    return StageCConfig.from_dict(value, schema_path=schema_path)


__all__ = [
    "DEFAULT_STAGE_C_CONFIG", "DEFAULT_STAGE_C_SCHEMA", "FrozenSection",
    "MethodConfig", "ProtocolConfig", "RegistryConfig", "StageCConfig",
    "StageCConfigError", "load_stage_c_config", "strict_json_load",
]
