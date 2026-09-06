"""Fail-closed Stage E E0 contract and completion validation.

This control-plane module validates sealed Stage D evidence and outcome-free
synthetic fixtures. It has no E1-E3 execution entry point and never loads an
image or model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as functional
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_d10_completion import validate_d10_formal_decision_report
from .stage_d_d9_completion import validate_d9_formal_validation_report
from .stage_d_paper import (
    ALL_FILES as D10_PACKAGE_FILES,
    _inventory as d10_package_inventory,
    _safe_read_package as safe_read_d10_package,
    validate_stage_d_analysis_package,
    validate_stage_d_paper_package,
)
from .stage_d_provenance import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "config/stage_e_e0.json"
CONFIG_SCHEMA_PATH = "config/stage_e_e0.schema.json"
RUN_SPEC_PATH = "config/stage_e_run_spec.json"
RUN_SPEC_SCHEMA_PATH = "config/stage_e_run_spec.schema.json"
REUSE_PATH = "evidence/stage_e/e0_recovery_v2_20260906/reuse_inventory.json"
REUSE_SCHEMA_PATH = "config/stage_e_reuse_inventory.schema.json"
COMPLETION_SCHEMA_PATH = "config/stage_e_e0_completion.schema.json"
PENDING_PATH = (
    "evidence/stage_e/e0_recovery_v2_20260906/completion/"
    "e0_completion.pending.json"
)
COMPLETION_PATH = (
    "evidence/stage_e/e0_recovery_v2_20260906/completion/e0_completion.json"
)
UPSTREAM_REPORT_PATH = "reports/stage_e_e0_20260906/upstream_validation.json"
TEST_REPORT_PATH = "reports/stage_e_e0_20260906/test_results.json"
FALSE_PASS_PATH = "reports/stage_e_e0_20260906/false_pass_audit.json"
ACCESS_LOG_PATH = "reports/stage_e_e0_20260906/outcome_access_log.json"
EXPECTED_COMPLETION_COMMIT = "7520689bb031304e4789567f99d093062aa50f72"
EXPECTED_D9_INVENTORY = "76d7d532049a5c90a09c74ba56b019042d7f2f0174ccb439bd8f67713514cc5a"
EXPECTED_D10_INVENTORY = "b72d36430d149598ccd9cfd8dd586297e7231be29e61b44309e9971087e46535"
EXPECTED_COORDINATES = ((0, 9), (4, 2), (6, 10), (11, 2))
EXPECTED_STRATA = ("overall", "correct", "misclassified")
EXPECTED_E2_VARIANTS = (
    "uniform_mass_divide_by_16",
    "mass_preserving_nearest",
    "bilinear",
    "bicubic",
)
EXPECTED_REUSE_VARIANTS = frozenset({
    "d10_fine_4x4_primary_anchor", "nns_native_14x14_patch",
    "matched_native_14x14_patch", "uniform_mass_divide_by_16",
    "mass_preserving_nearest", "bilinear",
    "bicubic", "layer_head_0_9_primary_anchor",
    "layer_head_4_2", "layer_head_6_10", "layer_head_11_2",
    "overall_correct_misclassified_primary_anchor", "clean_margin",
})
UPSTREAM_PROTECTED_PATHS = (
    "reports/stage_d_d9_recovery_v2_20260906",
    "reports/stage_d_d10_recovery_v2_20260906",
    "evidence/stage_d/d8_recovery_v2_20260904/formal_r2",
    "evidence/stage_d/d10_recovery_v2_20260906/formal_r2",
)
FALSE_PASS_PRECONDITIONS = (
    "nonempty_contract_files", "all_required_schemas_valid",
    "all_required_tests_passed", "all_upstream_hashes_verified",
    "typed_plan_complete", "reuse_inventory_complete",
    "budget_within_cap", "authorization_chain_valid",
    "outcome_firewall_clean", "protected_upstream_unchanged",
)


class StageEE0Error(ValueError):
    """E0 cannot pass because a frozen contract or source failed."""


@dataclass(frozen=True, slots=True)
class StageEVariant:
    analysis_family: str
    method_id: str
    variant_id: str
    endpoint: str
    status: str
    new_compute: str
    deterministically_derivable: bool


@dataclass(frozen=True, slots=True)
class StageEPlan:
    config_canonical_digest: str
    run_spec_canonical_digest: str
    ordered_sample_ids_sha256: str
    sample_count: int
    strata: tuple[str, ...]
    coordinates: tuple[tuple[int, int], ...]
    e2_variants: tuple[str, ...]
    variants: tuple[StageEVariant, ...]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_file_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _self_digest(value: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: item for key, item in value.items() if key != "canonical_digest"
    })


def _read_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except Exception as error:
        raise StageEE0Error(
            f"cannot read strict {context}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise StageEE0Error(f"{context} is not a JSON object")
    return value, raw


def _validate_schema(
    document: Mapping[str, Any], schema_path: Path, context: str,
) -> None:
    schema, _ = _read_json(schema_path, f"{context} schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda item: [str(part) for part in item.path],
    )
    if errors:
        where = ".".join(str(part) for part in errors[0].path) or "root"
        raise StageEE0Error(
            f"{context} schema failure at {where}: {errors[0].message}"
        )


def _validate_digest(document: Mapping[str, Any], context: str) -> None:
    if document.get("canonical_digest") != _self_digest(document):
        raise StageEE0Error(f"{context} canonical digest mismatch")


def _exact_keys(
    value: Any, expected: set[str], context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise StageEE0Error(
            f"{context} fields differ: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return value


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ("git", *args), cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise StageEE0Error(
            f"git {' '.join(args)} failed: {error.stderr.strip()}"
        ) from error


def load_stage_e_contract(
    repo_root: Path | str = REPO_ROOT,
) -> tuple[Mapping[str, Any], bytes]:
    repo = Path(repo_root).resolve()
    config, raw = _read_json(repo / CONFIG_PATH, "Stage E config")
    _validate_schema(config, repo / CONFIG_SCHEMA_PATH, "Stage E config")
    _validate_digest(config, "Stage E config")
    _exact_keys(config, {
        "$schema", "schema_version", "document_type", "stage", "milestone",
        "canonical_digest", "protocol", "upstream", "firewall", "cohort",
        "scientific_contract", "analyses", "outputs", "budget", "stage_gates",
    }, "Stage E config")
    protocol = _exact_keys(config["protocol"], {
        "authority", "frozen_protocol", "timing", "descriptive_only",
        "p_values_emitted", "rejection_decisions_emitted",
        "winner_selection_allowed", "new_go_no_go_allowed",
        "d10_primary_replacement_allowed", "renders_png", "renders_pdf",
        "renders_svg", "manual_case_selection_allowed", "numeric_source",
    }, "protocol")
    false_fields = (
        "p_values_emitted", "rejection_decisions_emitted",
        "winner_selection_allowed", "new_go_no_go_allowed",
        "d10_primary_replacement_allowed", "renders_png", "renders_pdf",
        "renders_svg", "manual_case_selection_allowed",
    )
    if (
        protocol["numeric_source"] != "paper_data_only"
        or protocol["descriptive_only"] is not True
        or any(protocol[key] is not False for key in false_fields)
        or "NOT_PRE_D_OUTCOME" not in protocol["timing"]
    ):
        raise StageEE0Error("descriptive-only/render/timing contract drifted")
    firewall = _exact_keys(config["firewall"], {
        "d10_outcome_known", "stage_e_new_outcome_accessed",
        "allowed_classes", "forbidden_classes",
    }, "firewall")
    if (
        firewall["d10_outcome_known"] is not True
        or firewall["stage_e_new_outcome_accessed"] is not False
    ):
        raise StageEE0Error("outcome firewall is not clean")
    cohort = config["cohort"]
    if (
        cohort["row_count"] != 500
        or cohort["stratum_counts"]
        != {"overall": 500, "correct": 418, "misclassified": 82}
        or cohort["strata"] != list(EXPECTED_STRATA)
        or cohort["complete_case_row_drop_allowed"] is not False
    ):
        raise StageEE0Error("cohort or prediction-strata contract drifted")
    science = config["scientific_contract"]
    if (
        science["statistical_unit"] != "image"
        or science["repeats_are_independent_samples"] is not False
        or len(science["fractions"]) != 21
        or science["interval"]["role"] != "DESCRIPTIVE_ONLY"
    ):
        raise StageEE0Error("scientific/statistical contract drifted")
    analyses = config["analyses"]
    if (
        tuple(analyses["e2"]["variants"]) != EXPECTED_E2_VARIANTS
        or analyses["e2"]["uniform_nearest_reporting_status"]
        != "EQUIVALENT_RANKING"
    ):
        raise StageEE0Error("E2 variant/equivalence contract drifted")
    coordinates = tuple(tuple(item) for item in analyses["e3"]["coordinates"])
    if (
        coordinates != EXPECTED_COORDINATES
        or tuple(analyses["e3"]["primary_anchor"]) != (0, 9)
        or analyses["e3"]["all_coordinates_required"] is not True
    ):
        raise StageEE0Error("E3 coordinate contract drifted")
    required_columns = {
        "variant_id", "sample_id", "selection_order", "cohort", "stratum",
        "validity_status", "source_hashes", "method", "target", "scale",
        "layer", "head",
    }
    outputs = config["outputs"]
    if (
        set(outputs["required_common_columns"]) != required_columns
        or outputs["numeric_source"] != "paper_data_only"
        or outputs["renders_png"] is not False
    ):
        raise StageEE0Error("output table or figure contract drifted")
    if (
        config["budget"]["e0"]["gpu_hours_max"] != 0
        or config["budget"]["e4"]["gpu_hours_max"] != 0
    ):
        raise StageEE0Error("E0/E4 CPU-only budget drifted")
    expected_gates = {
        "e1": ("E0", "E2"), "e2": ("E1", "E3"),
        "e3": ("E2", "E4"), "e4": ("E3", "F0"),
    }
    for key, (prior, following) in expected_gates.items():
        if config["stage_gates"][key] != {
            "requires_terminal_pass": prior, "authorizes_on_pass": following,
        }:
            raise StageEE0Error(f"{key} authorization gate drifted")
    return MappingProxyType(config), raw


def load_reuse_inventory(
    repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    document, _ = _read_json(repo / REUSE_PATH, "reuse inventory")
    _validate_schema(document, repo / REUSE_SCHEMA_PATH, "reuse inventory")
    _validate_digest(document, "reuse inventory")
    variants = [item["variant_id"] for item in document["items"]]
    if (
        len(variants) != len(set(variants))
        or set(variants) != EXPECTED_REUSE_VARIANTS
    ):
        raise StageEE0Error(
            "reuse inventory variants are missing, duplicated, or extra"
        )
    allowed = set(document["status_enum"])
    for item in document["items"]:
        if item["status"] not in allowed:
            raise StageEE0Error(
                f"untyped reuse state for {item['variant_id']}"
            )
        if (
            item["status"] == "NOT_RUN_WITH_REASON"
            and not item["reason"].strip()
        ):
            raise StageEE0Error(
                f"missing NOT_RUN_WITH_REASON rationale for {item['variant_id']}"
            )
        for source in item["input_sources"]:
            source_path = repo / source["path"]
            if (
                not source_path.is_file()
                or _sha(source_path.read_bytes()) != source["file_sha256"]
            ):
                raise StageEE0Error(
                    f"reuse source binding failed: {source['path']}"
                )
            if source["canonical_digest"] is not None:
                value, _ = _read_json(
                    source_path, f"reuse source {source['path']}"
                )
                if value.get("canonical_digest") != source["canonical_digest"]:
                    raise StageEE0Error(
                        f"reuse canonical binding failed: {source['path']}"
                    )
    return MappingProxyType(document)


def validate_reuse_inventory_document(
    document: Mapping[str, Any], repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    """Validate an in-memory reuse inventory, including all bound files."""

    repo = Path(repo_root).resolve()
    _validate_schema(
        document, repo / REUSE_SCHEMA_PATH, "reuse inventory"
    )
    _validate_digest(document, "reuse inventory")
    variants = [item["variant_id"] for item in document["items"]]
    if (
        len(variants) != len(set(variants))
        or set(variants) != EXPECTED_REUSE_VARIANTS
    ):
        raise StageEE0Error(
            "reuse inventory variants are missing, duplicated, or extra"
        )
    allowed = set(document["status_enum"])
    for item in document["items"]:
        if item["status"] not in allowed:
            raise StageEE0Error(
                f"untyped reuse state for {item['variant_id']}"
            )
        if (
            item["status"] == "NOT_RUN_WITH_REASON"
            and not item["reason"].strip()
        ):
            raise StageEE0Error(
                f"missing NOT_RUN_WITH_REASON rationale for {item['variant_id']}"
            )
        for source in item["input_sources"]:
            source_path = repo / source["path"]
            if (
                not source_path.is_file()
                or _sha(source_path.read_bytes()) != source["file_sha256"]
            ):
                raise StageEE0Error(
                    f"reuse source binding failed: {source['path']}"
                )
            if source["canonical_digest"] is not None:
                value, _ = _read_json(
                    source_path, f"reuse source {source['path']}"
                )
                if value.get("canonical_digest") != source["canonical_digest"]:
                    raise StageEE0Error(
                        f"reuse canonical binding failed: {source['path']}"
                    )
    return MappingProxyType(dict(document))


def load_stage_e_run_spec(
    repo_root: Path | str = REPO_ROOT,
) -> tuple[Mapping[str, Any], bytes]:
    repo = Path(repo_root).resolve()
    config, config_raw = load_stage_e_contract(repo)
    spec, raw = _read_json(repo / RUN_SPEC_PATH, "Stage E run spec")
    _validate_schema(spec, repo / RUN_SPEC_SCHEMA_PATH, "Stage E run spec")
    _validate_digest(spec, "Stage E run spec")
    if spec["config_binding"] != {
        "path": CONFIG_PATH,
        "file_sha256": _sha(config_raw),
        "canonical_digest": config["canonical_digest"],
    }:
        raise StageEE0Error("run-spec config binding mismatch")
    stages = spec["stages"]
    if [item["stage_id"] for item in stages] != ["E1", "E2", "E3", "E4"]:
        raise StageEE0Error("run-spec stages must be exact ordered E1-E4")
    if [item["next_authorization_on_pass"] for item in stages] != [
        "E2_ONLY", "E3_ONLY", "E4_ONLY", "F0_ONLY",
    ]:
        raise StageEE0Error("run-spec authorization chain drifted")
    expected_authority = {
        "current_stage": "E0", "e0_terminal_pass_authorizes": "E1_ONLY",
        "e0_pass_is_ablation_conclusion": False,
        "e1_started": False, "e2_started": False, "e3_started": False,
        "e4_started": False, "stage_f_started": False,
        "e4_may_issue_new_stage_d_decision": False,
    }
    if spec["authorization"] != expected_authority:
        raise StageEE0Error("run-spec current authorization boundary drifted")
    return MappingProxyType(spec), raw


def build_typed_plan(repo_root: Path | str = REPO_ROOT) -> StageEPlan:
    config, _ = load_stage_e_contract(repo_root)
    spec, _ = load_stage_e_run_spec(repo_root)
    reuse = load_reuse_inventory(repo_root)
    variants = tuple(StageEVariant(
        analysis_family=item["analysis_family"],
        method_id=item["method_id"], variant_id=item["variant_id"],
        endpoint=item["endpoint"], status=item["status"],
        new_compute=item["new_compute"],
        deterministically_derivable=item["deterministically_derivable"],
    ) for item in reuse["items"])
    return StageEPlan(
        config_canonical_digest=config["canonical_digest"],
        run_spec_canonical_digest=spec["canonical_digest"],
        ordered_sample_ids_sha256=config["cohort"]["ordered_sample_ids_sha256"],
        sample_count=config["cohort"]["row_count"],
        strata=tuple(config["cohort"]["strata"]),
        coordinates=tuple(
            tuple(item) for item in config["analyses"]["e3"]["coordinates"]
        ),
        e2_variants=tuple(config["analyses"]["e2"]["variants"]),
        variants=variants,
    )


def validate_registry_and_strata(
    repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    config, _ = load_stage_e_contract(repo)
    registry, registry_raw = _read_json(
        repo / config["cohort"]["registry_path"], "D1 registry"
    )
    if (
        _sha(registry_raw) != config["cohort"]["registry_file_sha256"]
        or registry.get("canonical_digest")
        != config["cohort"]["registry_canonical_digest"]
    ):
        raise StageEE0Error("D1 registry binding mismatch")
    rows = registry.get("rows")
    if (
        not isinstance(rows, list) or len(rows) != 500
        or [row.get("selection_order") for row in rows] != list(range(500))
    ):
        raise StageEE0Error(
            "D1 registry is not the exact ordered 500-image cohort"
        )
    sample_ids = [row.get("sample_id") for row in rows]
    if (
        len(set(sample_ids)) != 500
        or canonical_sha256(sample_ids)
        != config["cohort"]["ordered_sample_ids_sha256"]
    ):
        raise StageEE0Error("D1 ordered sample identity drifted")
    d10_path = repo / (
        "evidence/stage_d/d10_recovery_v2_20260906/formal_r2/paper/"
        "paper_data/per_sample_delta.json"
    )
    d10, _ = _read_json(d10_path, "D10 per-sample table")
    d10_rows = d10.get("rows")
    if (
        not isinstance(d10_rows, list)
        or [row.get("sample_id") for row in d10_rows] != sample_ids
        or [row.get("selection_order") for row in d10_rows] != list(range(500))
    ):
        raise StageEE0Error(
            "D10 per-sample rows differ from the frozen registry order"
        )
    misclassified = [
        row["sample_id"] for row in d10_rows
        if row.get("misclassified") is True
    ]
    correct = [
        row["sample_id"] for row in d10_rows
        if row.get("misclassified") is False
    ]
    if (
        len(misclassified) != 82 or len(correct) != 418
        or set(misclassified).intersection(correct)
        or set(misclassified).union(correct) != set(sample_ids)
    ):
        raise StageEE0Error(
            "prediction strata do not partition the full cohort as 418/82"
        )
    return MappingProxyType({
        "sample_count": 500,
        "ordered_sample_ids_sha256": canonical_sha256(sample_ids),
        "overall": 500, "correct": 418, "misclassified": 82,
    })


def validate_stage_c_coordinates(
    repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    config, _ = load_stage_e_contract(repo)
    source = config["analyses"]["e3"]
    document, raw = _read_json(
        repo / source["selection_path"], "Stage C selection"
    )
    if (
        _sha(raw) != source["selection_file_sha256"]
        or document.get("canonical_digest")
        != source["selection_canonical_digest"]
        or document.get("holdout_accessed") is not False
    ):
        raise StageEE0Error("Stage C pre-holdout selection binding failed")
    coordinates = tuple(
        (item["block_index"], item["head_index"])
        for item in document["depth_band_winners"]
    )
    primary = (
        document["primary"]["block_index"],
        document["primary"]["head_index"],
    )
    if coordinates != EXPECTED_COORDINATES or primary != (0, 9):
        raise StageEE0Error("Stage C depth-band coordinate set drifted")
    return MappingProxyType({
        "coordinates": coordinates, "primary": primary,
        "holdout_accessed": False,
    })


def upsampling_operator(variant_id: str) -> np.ndarray:
    if variant_id not in EXPECTED_E2_VARIANTS:
        raise StageEE0Error(f"unknown E2 variant: {variant_id}")
    basis = torch.eye(196, dtype=torch.float64).reshape(196, 1, 14, 14)
    if variant_id == "uniform_mass_divide_by_16":
        expanded = (
            basis.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
            / 16.0
        )
    else:
        mode = variant_id.removeprefix("mass_preserving_")
        kwargs = (
            {} if mode == "nearest"
            else {"align_corners": False, "antialias": False}
        )
        raw = functional.interpolate(
            basis, size=(56, 56), mode=mode, **kwargs
        )
        columns = raw.reshape(196, -1).sum(dim=1)
        if (
            not torch.isfinite(columns).all()
            or bool(torch.any(columns == 0))
        ):
            raise StageEE0Error(
                f"{variant_id} has invalid interpolation column mass"
            )
        expanded = raw / columns[:, None, None, None]
    return (
        expanded[:, 0].reshape(196, 3136).transpose(0, 1)
        .contiguous().numpy()
    )


def apply_upsampling(values: np.ndarray, variant_id: str) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.shape != (14, 14) or not np.isfinite(source).all():
        raise StageEE0Error(
            "upsampling input must be finite float64-compatible 14x14"
        )
    return (
        upsampling_operator(variant_id) @ source.reshape(-1)
    ).reshape(56, 56)


def ranking(values: np.ndarray, seed: int) -> np.ndarray:
    scores = np.maximum(
        np.asarray(values, dtype=np.float64).reshape(-1), 0.0
    )
    order = np.random.default_rng(seed).permutation(scores.size)
    priority = np.empty(scores.size, dtype=np.int64)
    priority[order] = np.arange(scores.size, dtype=np.int64)
    return np.lexsort((priority, -scores))


def validate_e2_golden() -> Mapping[str, Any]:
    operators = {
        name: upsampling_operator(name) for name in EXPECTED_E2_VARIANTS
    }
    tolerance = 1e-12
    errors = {
        name: float(np.max(np.abs(value.sum(axis=0) - 1.0)))
        for name, value in operators.items()
    }
    if any(error > tolerance for error in errors.values()):
        raise StageEE0Error(f"E2 basis conservation failed: {errors}")
    if not np.array_equal(
        operators[EXPECTED_E2_VARIANTS[0]],
        operators[EXPECTED_E2_VARIANTS[1]],
    ):
        raise StageEE0Error(
            "uniform and mass-preserving nearest operators differ"
        )
    fixtures = (
        np.arange(196, dtype=np.float64).reshape(14, 14) - 97.5,
        np.zeros((14, 14), dtype=np.float64),
        np.random.default_rng(20260906).normal(size=(14, 14)),
    )
    for index, source in enumerate(fixtures):
        transformed = {
            name: apply_upsampling(source, name)
            for name in EXPECTED_E2_VARIANTS
        }
        scale = max(1.0, abs(float(source.sum())))
        if any(
            abs(float(value.sum() - source.sum()))
            > tolerance + tolerance * scale
            for value in transformed.values()
        ):
            raise StageEE0Error(
                f"E2 signed-map conservation failed for fixture {index}"
            )
        left = transformed[EXPECTED_E2_VARIANTS[0]]
        right = transformed[EXPECTED_E2_VARIANTS[1]]
        if (
            not np.array_equal(left, right)
            or not np.array_equal(
                ranking(left, 20260906 + index),
                ranking(right, 20260906 + index),
            )
        ):
            raise StageEE0Error(
                f"E2 uniform/nearest ranking equivalence failed for fixture {index}"
            )
    return MappingProxyType({
        "status": "PASS",
        "basis_conservation_max_abs_errors": errors,
        "uniform_nearest_operator_equal": True,
        "uniform_nearest_ranking_equal": True,
        "fixture_count": len(fixtures),
    })


def false_pass_reasons(
    preconditions: Mapping[str, bool],
) -> tuple[str, ...]:
    if set(preconditions) != set(FALSE_PASS_PRECONDITIONS):
        missing = set(FALSE_PASS_PRECONDITIONS) - set(preconditions)
        extra = set(preconditions) - set(FALSE_PASS_PRECONDITIONS)
        raise StageEE0Error(
            f"false-PASS inputs differ: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return tuple(
        name for name in FALSE_PASS_PRECONDITIONS
        if preconditions[name] is not True
    )


def _d10_inventory(repo: Path) -> dict[str, Any]:
    root = repo / "evidence/stage_d/d10_recovery_v2_20260906/formal_r2"
    rows: list[dict[str, Any]] = []
    for subdirectory in ("analysis", "paper"):
        payloads = safe_read_d10_package(root / subdirectory)
        for row in d10_package_inventory(payloads, D10_PACKAGE_FILES):
            rows.append({
                "path": f"{subdirectory}/{row['path']}",
                "bytes": row["bytes"], "sha256": row["sha256"],
            })
    rows.sort(key=lambda item: item["path"])
    return {
        "file_count": len(rows),
        "total_bytes": sum(item["bytes"] for item in rows),
        "canonical_digest": canonical_sha256({"files": rows}),
    }


def verify_upstream(
    repo_root: Path | str = REPO_ROOT,
    *, d9_source_control: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    repo = Path(repo_root).resolve()
    config, _ = load_stage_e_contract(repo)
    upstream = config["upstream"]
    observed_commit = _git(
        repo, "rev-parse",
        "refs/archive/stage-d-fidelity-v1-d10-completion^{commit}",
    )
    if (
        observed_commit != EXPECTED_COMPLETION_COMMIT
        or observed_commit != upstream["completion_commit"]
    ):
        raise StageEE0Error(
            "D10 completion ref mismatch: "
            f"expected={EXPECTED_COMPLETION_COMMIT} observed={observed_commit}"
        )
    d9 = validate_d9_formal_validation_report(repo_root=repo)
    d10 = validate_d10_formal_decision_report(repo_root=repo)
    analysis = validate_stage_d_analysis_package(
        repo / upstream["d10_report"]["output_root"] / "analysis",
        repo_root=repo,
    )
    paper = validate_stage_d_paper_package(
        repo / upstream["d10_report"]["paper_root"], repo_root=repo,
    )
    if d9_source_control is None:
        command = (
            os.fspath(Path(sys.executable)), "-m", "vitnns.stage_d_cli",
            "validate", "--allow-formal-root", "--d9-formal-budget",
            os.fspath(repo / "config/stage_d_d7_budget_20260904_r4.json"),
            "--output-root",
            os.fspath(repo / upstream["d9_report"]["sealed_root"]),
        )
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES="",
            PYTHONPATH=os.fspath(repo / "src"),
        )
        try:
            completed = subprocess.run(
                command, cwd=repo, env=environment, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            d9_source_control = json.loads(completed.stdout)
        except Exception as error:
            raise StageEE0Error(
                f"D9 source validation failed: {error}"
            ) from error
    if (
        d9_source_control.get("status") != "PASS"
        or d9_source_control.get("analysis_authorized") is not True
        or d9_source_control.get("root_inventory", {}).get("canonical_digest")
        != EXPECTED_D9_INVENTORY
    ):
        raise StageEE0Error(
            "D9 source validation did not reproduce frozen PASS/inventory"
        )
    d10_inventory = _d10_inventory(repo)
    if d10_inventory["canonical_digest"] != EXPECTED_D10_INVENTORY:
        raise StageEE0Error("D10 whole-root inventory drifted")
    expected_stats = upstream["d10_statistics_read_only"]
    observed_stats = d10["statistics"]
    observed_projection = {
        "n": observed_stats["n"],
        "positive_count": observed_stats["positive_count"],
        "mean_delta_fid": observed_stats["mean"],
        "median_delta_fid": observed_stats["median"],
        "sample_sd_ddof_1": observed_stats["sample_sd_ddof_1"],
        "paired_image_bootstrap_ci_95": list(
            observed_stats["bootstrap_ci_95"]
        ),
    }
    expected_c10 = {
        key: value
        for key, value in upstream["c10_history_read_only"].items()
        if key not in {"n", "pool_with_stage_d_allowed"}
    }
    if (
        observed_projection != expected_stats
        or dict(d10["c10_status_preserved"]) != expected_c10
    ):
        raise StageEE0Error("D10 statistics or C10 historical state drifted")
    changed = _git(
        repo, "diff", "--name-only", EXPECTED_COMPLETION_COMMIT, "--",
        *UPSTREAM_PROTECTED_PATHS,
    ).splitlines()
    if changed:
        raise StageEE0Error(
            f"protected D9/D10 tracked paths changed: {changed}"
        )
    return MappingProxyType({
        "status": "PASS", "completion_ref_commit": observed_commit,
        "d9_report_status": d9["status"],
        "d9_source_validation": dict(d9_source_control),
        "d10_report_status": d10["status"],
        "analysis_package_valid": analysis["valid"],
        "paper_package_valid": paper["valid"],
        "analysis_paper_digest_equal": (
            analysis["analysis_canonical_digest"]
            == paper["analysis_canonical_digest"]
        ),
        "d10_output_root_inventory": d10_inventory,
        "protected_tracked_diff": changed,
        "d10_statistics_read_only": observed_projection,
        "d10_decision_read_only": dict(upstream["d10_decision_read_only"]),
        "c10_history_read_only": dict(upstream["c10_history_read_only"]),
    })


def validate_e0_inputs(repo_root: Path | str = REPO_ROOT) -> StageEPlan:
    plan = build_typed_plan(repo_root)
    if (
        plan.sample_count != 500
        or plan.strata != EXPECTED_STRATA
        or plan.coordinates != EXPECTED_COORDINATES
        or plan.e2_variants != EXPECTED_E2_VARIANTS
    ):
        raise StageEE0Error("typed plan differs from the frozen scope")
    validate_registry_and_strata(repo_root)
    validate_stage_c_coordinates(repo_root)
    validate_e2_golden()
    return plan


def file_record(
    repo_root: Path | str, relative_path: str,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    document, raw = _read_json(repo / relative_path, relative_path)
    _validate_digest(document, relative_path)
    return {
        "path": relative_path, "file_sha256": _sha(raw),
        "canonical_digest": document["canonical_digest"],
    }


def validate_completion_document(
    document: Mapping[str, Any], repo_root: Path | str = REPO_ROOT,
) -> Mapping[str, Any]:
    """Validate a terminal completion record without trusting self-reports."""

    repo = Path(repo_root).resolve()
    _validate_schema(
        document, repo / COMPLETION_SCHEMA_PATH, "E0 completion"
    )
    _validate_digest(document, "E0 completion")
    if document["status"] != "PASS":
        raise StageEE0Error("terminal E0 record is not PASS")
    timing = document["planning_timing"]
    if (
        timing.get("d10_outcome_known") is not True
        or timing.get("e0_was_pre_d_outcome_preregistration") is not False
    ):
        raise StageEE0Error("completion omits honest post-D10 timing")
    firewall = document["outcome_firewall"]
    if (
        firewall.get("stage_e_new_outcome_accessed") is not False
        or firewall.get("violations_detected") is not False
        or firewall.get("e1_e2_e3_executed") is not False
    ):
        raise StageEE0Error("completion outcome firewall is not clean")
    for field in (
        "access_log", "upstream_validation", "test_results",
        "false_pass_audit",
    ):
        if dict(document[field]) != file_record(repo, document[field]["path"]):
            raise StageEE0Error(f"completion {field} binding mismatch")
    access, _ = _read_json(
        repo / document["access_log"]["path"], "E0 access log"
    )
    if (
        access.get("stage_e_new_outcome_accessed") is not False
        or access.get("e1_e2_e3_executed") is not False
        or access.get("violations_detected") is not False
    ):
        raise StageEE0Error("bound outcome-access log is not clean")
    upstream, _ = _read_json(
        repo / document["upstream_validation"]["path"],
        "E0 upstream validation",
    )
    if (
        upstream.get("status") != "PASS" or not upstream.get("checks")
        or any(item.get("status") != "PASS" for item in upstream["checks"])
    ):
        raise StageEE0Error("bound upstream validation is incomplete or failed")
    contract_paths = {item["path"] for item in document["contracts"]}
    required_contracts = {
        "docs/stage_e_protocol.md", CONFIG_PATH, CONFIG_SCHEMA_PATH,
        RUN_SPEC_PATH, RUN_SPEC_SCHEMA_PATH, REUSE_PATH, REUSE_SCHEMA_PATH,
        "config/stage_e_output_table.schema.json",
        "config/stage_e_figure_recipes.schema.json",
        "config/stage_e_package.schema.json",
        "config/stage_e_stage_completion.schema.json",
        COMPLETION_SCHEMA_PATH, "src/vitnns/stage_e_e0.py",
        "tests/test_stage_e_e0.py",
    }
    if contract_paths != required_contracts:
        raise StageEE0Error("completion contract inventory is incomplete or extra")
    for record in document["contracts"]:
        path = repo / record["path"]
        if (
            not path.is_file() or path.stat().st_size == 0
            or _sha(path.read_bytes()) != record["file_sha256"]
        ):
            raise StageEE0Error(f"completion contract hash mismatch: {record['path']}")
        if record["canonical_digest"] is not None:
            source, _ = _read_json(path, record["path"])
            if source.get("canonical_digest") != record["canonical_digest"]:
                raise StageEE0Error(
                    f"completion contract canonical mismatch: {record['path']}"
                )
    plan = validate_e0_inputs(repo)
    typed = document["typed_plan"]
    if (
        typed.get("validated") is not True
        or typed.get("variant_count") != len(plan.variants)
        or typed.get("coordinate_count") != len(plan.coordinates)
    ):
        raise StageEE0Error("completion typed plan is incomplete")
    reuse = load_reuse_inventory(repo)
    counts = {
        name: sum(item["status"] == name for item in reuse["items"])
        for name in reuse["status_enum"]
    }
    if (
        document["reuse_inventory"].get("validated") is not True
        or document["reuse_inventory"].get("variant_count")
        != len(reuse["items"])
        or document["reuse_inventory"].get("status_counts") != counts
    ):
        raise StageEE0Error("completion reuse inventory is incomplete")
    budget = document["budget"]
    if (
        budget.get("status") != "PASS"
        or budget.get("gpu_hours_actual") != 0.0
        or budget.get("cpu_hours_actual", math.inf)
        > budget.get("cpu_hours_max", -1)
        or budget.get("retained_bytes_actual", math.inf)
        > budget.get("retained_bytes_max", -1)
        or budget.get("future_stage_caps_frozen") is not True
    ):
        raise StageEE0Error("completion budget gate failed")
    tests, _ = _read_json(
        repo / document["test_results"]["path"], "E0 test report"
    )
    required_test_gates = {
        "contracts", "registry", "coordinates", "strata", "e2_golden",
        "output_columns", "render_boundary", "upstream_fail_closed",
        "reuse_fail_closed", "firewall_fail_closed", "variant_fail_closed",
        "budget_fail_closed", "authorization_fail_closed",
        "protected_upstream_unchanged",
    }
    if (
        tests.get("status") != "PASS" or tests.get("failed") != 0
        or tests.get("required_gates") != {
            key: "PASS" for key in sorted(required_test_gates)
        }
    ):
        raise StageEE0Error("completion test gate is incomplete or failed")
    false_pass, _ = _read_json(
        repo / document["false_pass_audit"]["path"],
        "E0 false-PASS audit",
    )
    if (
        false_pass.get("status") != "PASS"
        or not false_pass.get("checks")
        or any(item.get("result") != "REJECTED" for item in false_pass["checks"])
    ):
        raise StageEE0Error("false-PASS audit is incomplete")
    unchanged = document["upstream_unchanged"]
    if (
        unchanged.get("d9_inventory_before") != EXPECTED_D9_INVENTORY
        or unchanged.get("d9_inventory_after") != EXPECTED_D9_INVENTORY
        or unchanged.get("d10_inventory_before") != EXPECTED_D10_INVENTORY
        or unchanged.get("d10_inventory_after") != EXPECTED_D10_INVENTORY
        or unchanged.get("protected_tracked_diff") != []
    ):
        raise StageEE0Error("completion does not prove protected upstream unchanged")
    authorization = document["authorization"]
    if authorization != {
        "e1_authorized": True, "e2_authorized": False,
        "e3_authorized": False, "e4_authorized": False,
        "stage_f_authorized": False,
        "e0_pass_contains_ablation_conclusion": False,
    }:
        raise StageEE0Error("completion over-authorizes downstream work")
    return MappingProxyType(dict(document))


def publish_completion(
    document: Mapping[str, Any], repo_root: Path | str = REPO_ROOT,
) -> Path:
    """Validate PASS in memory, write it under pending, then rename atomically."""

    repo = Path(repo_root).resolve()
    validated = validate_completion_document(document, repo)
    pending = repo / PENDING_PATH
    target = repo / COMPLETION_PATH
    if target.exists() or target.is_symlink():
        raise StageEE0Error(f"refusing to overwrite completion: {target}")
    pending.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_file_bytes(validated)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".e0-completion-", suffix=".pending", dir=pending.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pending)
        # Re-read and validate the exact pending bytes before terminal rename.
        observed, observed_raw = _read_json(pending, "E0 pending completion")
        if observed_raw != raw or observed != dict(validated):
            raise StageEE0Error("pending completion changed before publication")
        os.replace(pending, target)
        directory_fd = os.open(pending.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
    return target


def _main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vitnns.stage_e_e0")
    parser.add_argument(
        "command", choices=("validate-contracts", "validate-upstream")
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    if args.command == "validate-contracts":
        plan = validate_e0_inputs(args.repo_root)
        value = {
            "status": "PASS", "sample_count": plan.sample_count,
            "variant_count": len(plan.variants),
            "coordinates": [list(item) for item in plan.coordinates],
            "e2_variants": list(plan.e2_variants),
        }
    else:
        value = dict(verify_upstream(args.repo_root))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
