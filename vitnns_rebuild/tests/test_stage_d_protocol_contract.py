"""Static/metadata-only contract tests for the Stage D D2 freeze."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config/stage_d.json"
CONFIG_SCHEMA_PATH = REPO_ROOT / "config/stage_d.schema.json"
RUN_SPEC_PATH = REPO_ROOT / "config/stage_d_run_spec.json"
RUN_SPEC_SCHEMA_PATH = REPO_ROOT / "config/stage_d_run_spec.schema.json"
D0_PATH = REPO_ROOT / "config/stage_d_scope_decision.json"
D1_ROOT = REPO_ROOT / "manifests/stage_d_fidelity_v1"
D1_MANIFEST_PATH = D1_ROOT / "manifest.json"
D1_REGISTRY_PATH = D1_ROOT / "registry.json"
DOCUMENT_PATHS = (
    REPO_ROOT / "docs/阶段D_protocol.md",
    REPO_ROOT / "docs/stage_d_acceptance.md",
    REPO_ROOT / "docs/stage_d_config_contract.md",
)
FORBIDDEN_MODULE_PREFIXES = (
    "PIL", "cv2", "timm", "torch", "torchvision",
    "vitnns.attribution", "vitnns.dataset", "vitnns.extractor",
    "vitnns.modeling", "vitnns.runner", "vitnns.stage_c_runner",
)
class ContractError(ValueError):
    """Raised by the deliberately test-local strict contract checks."""


def _strict_json_bytes(payload: bytes, context: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"{context}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ContractError(f"{context}: non-finite JSON number {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise ContractError(f"{context}: JSON must be UTF-8") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{context}: document root must be an object")
    return value


def _load(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(path.read_bytes(), path.relative_to(REPO_ROOT).as_posix())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_digest(document: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in document.items() if key != "canonical_digest"}
    )


def _assert_self_digest(document: Mapping[str, Any], context: str) -> None:
    if document.get("canonical_digest") != _self_digest(document):
        raise ContractError(f"{context}: canonical digest mismatch")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_run_config_binding(
    run_spec: Mapping[str, Any], config: Mapping[str, Any], config_bytes: bytes,
) -> None:
    binding = run_spec["config_binding"]
    expected = {
        "config_path": "config/stage_d.json",
        "config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_canonical_digest": config["canonical_digest"],
    }
    if binding != expected:
        raise ContractError("run spec config source binding drifted")


def _assert_effectively_closed_schema(schema: Mapping[str, Any]) -> None:
    """Require every object schema to be closed or to freeze an exact object."""

    def visit(rule: Any, location: str) -> None:
        if not isinstance(rule, dict):
            return
        if isinstance(rule.get("const"), dict):
            return  # An exact object const cannot admit an undeclared instance key.
        if rule.get("type") == "object":
            properties = rule.get("properties")
            assert isinstance(properties, dict), location
            assert rule.get("additionalProperties") is False, location
            assert set(rule.get("required", ())) == set(properties), location

        for keyword in ("properties", "patternProperties", "$defs", "dependentSchemas"):
            children = rule.get(keyword, {})
            if isinstance(children, dict):
                for name, child in children.items():
                    visit(child, f"{location}/{keyword}/{name}")
        for keyword in ("items", "contains", "not", "if", "then", "else"):
            visit(rule.get(keyword), f"{location}/{keyword}")
        for keyword in ("prefixItems", "allOf", "anyOf", "oneOf"):
            for index, child in enumerate(rule.get(keyword, ())):
                visit(child, f"{location}/{keyword}/{index}")

    visit(schema, "$")


def _component_status(positive_count: int, median: float, lower: float, upper: float) -> str:
    if positive_count < 400 or upper < 0.02:
        return "FAIL_FIDELITY"
    if median >= 0.02 and positive_count >= 400 and lower > 0.0:
        return "PASS_FIDELITY"
    return "INCONCLUSIVE_FIDELITY"


def test_d2_guard_is_active_while_d3_control_plane_is_model_free() -> None:
    assert os.environ.get("VITNNS_D3_STRICT") == "1"
    assert os.environ.get("VITNNS_D3_GUARD_ACTIVE") == "1"
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    loaded = set(sys.modules)
    assert not sorted(
        name for name in loaded
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MODULE_PREFIXES)
    )
    with pytest.raises(ImportError, match="D3 strict guard blocked import"):
        importlib.import_module("torch")
    with pytest.raises(PermissionError, match="blocked image/model/binary path"):
        Path("d2_guard_probe.JPEG").stat()
    with pytest.raises(PermissionError, match="blocked Stage D outcome path"):
        Path("stage_d_outcomes/probe.json").stat()
    with pytest.raises(PermissionError, match="blocked GPU device access"):
        Path("/dev/nvidia0").stat()
    with pytest.raises(PermissionError, match="blocked subprocess"):
        subprocess.Popen(["true"])
    for module in (
        "vitnns.stage_d_config",
        "vitnns.stage_d_manifest",
        "vitnns.stage_d_plan",
        "vitnns.stage_d_authorization",
        "vitnns.stage_d_cli",
    ):
        importlib.import_module(module)
    loaded = set(sys.modules)
    assert not sorted(
        name for name in loaded
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MODULE_PREFIXES)
    )


@pytest.mark.parametrize("document_path", [CONFIG_PATH, RUN_SPEC_PATH])
def test_both_d2_documents_use_strict_duplicate_and_nonfinite_parsing(
    document_path: Path,
) -> None:
    assert _load(document_path)
    with pytest.raises(ContractError, match="duplicate JSON key"):
        _strict_json_bytes(b'{"a":1,"a":2}', document_path.name)
    for token in (b"NaN", b"Infinity", b"-Infinity"):
        with pytest.raises(ContractError, match="non-finite JSON number"):
            _strict_json_bytes(b'{"a":' + token + b"}", document_path.name)


def test_schemas_are_draft_2020_12_effectively_closed_and_validate_instances() -> None:
    for instance_path, schema_path in (
        (CONFIG_PATH, CONFIG_SCHEMA_PATH),
        (RUN_SPEC_PATH, RUN_SPEC_SCHEMA_PATH),
    ):
        schema = _load(schema_path)
        instance = _load(instance_path)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance)
        _assert_effectively_closed_schema(schema)


def test_self_digests_and_dynamic_config_binding_match_exact_bytes() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    run_bytes = RUN_SPEC_PATH.read_bytes()
    assert config_bytes.endswith(b"\n") and run_bytes.endswith(b"\n")
    config = _strict_json_bytes(config_bytes, "stage_d.json")
    run_spec = _strict_json_bytes(run_bytes, "stage_d_run_spec.json")
    _assert_self_digest(config, "stage_d.json")
    _assert_self_digest(run_spec, "stage_d_run_spec.json")
    _assert_run_config_binding(run_spec, config, config_bytes)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update({"unknown": True}),
        lambda d: d.pop("audit"),
        lambda d: d["execution_coordinates"].update({"head_index": 8}),
        lambda d: d["execution_coordinates"]["method_ids"].reverse(),
        lambda d: d["selection"]["sample_ids"].pop(),
        lambda d: d["fidelity"].update({"spatial_scope": "PATCH_ONLY"}),
        lambda d: d["forbidden_workload"].update({"run_stability": True}),
        lambda d: d["workload"].update({"logical_work_items": 211009}),
        lambda d: d["sharding"].update({"shard_count": 8}),
    ],
)
def test_run_spec_schema_rejects_scope_count_and_unknown_field_drift(mutate) -> None:
    run_spec = copy.deepcopy(_load(RUN_SPEC_PATH))
    mutate(run_spec)
    run_spec["canonical_digest"] = _self_digest(run_spec)
    with pytest.raises(ValidationError):
        Draft202012Validator(_load(RUN_SPEC_SCHEMA_PATH)).validate(run_spec)


def test_config_schema_rejects_self_resigned_scientific_and_authority_drift() -> None:
    schema = _load(CONFIG_SCHEMA_PATH)
    for section, field, value in (
        ("authority", "execution_authorization", "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"),
        ("cohort", "statistical_n", 499),
        ("decision_gates", "fidelity_fail_if_positive_count_below", 399),
        ("secondary_and_spatial", "run_stability", True),
        ("cost", "d7_may_loosen_caps", True),
    ):
        changed = copy.deepcopy(_load(CONFIG_PATH))
        changed[section][field] = value
        changed["canonical_digest"] = _self_digest(changed)
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(changed)


def test_upstream_d0_d1_and_c10_bindings_recompute_from_metadata() -> None:
    config = _load(CONFIG_PATH)
    d0 = _load(D0_PATH)
    _assert_self_digest(d0, "D0 scope")
    assert config["upstream"]["d0"] == {
        "completion_ref": "refs/archive/stage-d-d0-completion",
        "completion_commit": "4c76d468e1368858c52ff8ef73e640463e15ac49",
        "machine_record_path": "config/stage_d_scope_decision.json",
        "machine_record_file_sha256": _file_sha256(D0_PATH),
        "machine_record_canonical_digest": d0["canonical_digest"],
        "schema_file_sha256": _file_sha256(
            REPO_ROOT / "config/stage_d_scope_decision.schema.json"
        ),
        "markdown_file_sha256": _file_sha256(
            REPO_ROOT / "docs/stage_d_scope_decision.md"
        ),
    }

    manifest = _load(D1_MANIFEST_PATH)
    _assert_self_digest(manifest, "D1 manifest")
    payloads = {record["path"]: record for record in manifest["payloads"]}
    assert manifest["payload_count"] == len(payloads) == 5
    for name, record in payloads.items():
        path = D1_ROOT / name
        document = _load(path)
        observed_canonical = (
            _canonical_sha256(document)
            if record["digest_mode"] == "FULL_DOCUMENT"
            else _self_digest(document)
        )
        assert record["bytes"] == len(path.read_bytes())
        assert record["sha256"] == _file_sha256(path)
        assert record["canonical_digest"] == observed_canonical

    upstream_d1 = config["upstream"]["d1"]
    assert upstream_d1["package_manifest_file_sha256"] == _file_sha256(D1_MANIFEST_PATH)
    assert upstream_d1["package_manifest_canonical_digest"] == manifest["canonical_digest"]
    for prefix, name in (
        ("registry", "registry.json"),
        ("registry_schema", "stage_d_registry_package.schema.json"),
        ("source_binding", "source_binding.json"),
        ("registry_metadata", "registry_metadata.json"),
        ("attestation", "pre_authorization_access_attestation.json"),
    ):
        assert upstream_d1[f"{prefix}_file_sha256"] == payloads[name]["sha256"]
        assert upstream_d1[f"{prefix}_canonical_digest"] == payloads[name]["canonical_digest"]

    source_c10 = d0["source_c10"]
    historical = source_c10["historical_status"]
    files = source_c10["files"]
    c10 = config["upstream"]["c10"]
    assert c10 == {
        "formal_root": source_c10["formal_root"],
        "completion_commit": source_c10["completion_commit"],
        "producer_ref": source_c10["aggregation_producer"]["archive_ref"],
        "producer_commit": source_c10["aggregation_producer"]["commit"],
        "producer_tree": source_c10["aggregation_producer"]["tree"],
        "producer_source_tree": source_c10["aggregation_producer"]["source_tree"],
        "status_file_sha256": files["status.json"]["sha256"],
        "status_canonical_digest": files["status.json"]["canonical_digest"],
        "report_file_sha256": files["stage_c_report.json"]["sha256"],
        "report_canonical_digest": files["stage_c_report.json"]["canonical_digest"],
        "decision_file_sha256": files["decision.json"]["sha256"],
        "decision_canonical_digest": files["decision.json"]["canonical_digest"],
        "holm_file_sha256": files["holm.json"]["sha256"],
        "holm_canonical_digest": files["holm.json"]["canonical_digest"],
        "provenance_file_sha256": files["provenance.json"]["sha256"],
        "provenance_canonical_digest": files["provenance.json"]["canonical_digest"],
        "manifest_file_sha256": files["manifest.json"]["sha256"],
        "manifest_canonical_digest": files["manifest.json"]["canonical_digest"],
        "whole_root_file_count": source_c10["whole_root_inventory"]["file_count"],
        "whole_root_total_bytes": source_c10["whole_root_inventory"]["total_bytes"],
        "whole_root_inventory_digest": source_c10["whole_root_inventory"]["canonical_digest"],
        "primary_decision": historical["primary_decision"],
        "stage_c_acceptance": historical["stage_c_acceptance"],
        "workflow_blocker": historical["workflow_blocker"],
        "spatial_status": historical["spatial_status"],
        "downstream_authorized": historical["downstream_authorized"],
        "stage_d_started": historical["stage_d_started"],
        "cost_evidence_role": "HISTORICAL_8000_UNIT_FEASIBILITY_ONLY",
    }
    assert d0["scientific_basis"]["old_cost_evidence_role"] == (
        "HISTORICAL_FEASIBILITY_ONLY_REQUIRES_NEW_D7_BUDGET"
    )


def test_run_spec_matches_config_and_exact_500_row_registry_order() -> None:
    config = _load(CONFIG_PATH)
    run_spec = _load(RUN_SPEC_PATH)
    registry = _load(D1_REGISTRY_PATH)
    rows = registry["rows"]
    assert len(rows) == 500
    assert [row["selection_order"] for row in rows] == list(range(500))
    assert all(row["canonical_digest"] == _self_digest(row) for row in rows)
    sample_ids = [row["sample_id"] for row in rows]
    assert len(set(sample_ids)) == 500
    assert run_spec["selection"]["sample_ids"] == sample_ids
    assert registry["ordered_sample_ids_sha256"] == _canonical_sha256(sample_ids)
    assert run_spec["registry_binding"] == {
        "d1_completion_ref": config["upstream"]["d1"]["completion_ref"],
        "d1_completion_commit": config["upstream"]["d1"]["completion_commit"],
        "package_id": config["upstream"]["d1"]["package_id"],
        "registry_path": config["upstream"]["d1"]["registry_path"],
        "registry_file_sha256": _file_sha256(D1_REGISTRY_PATH),
        "registry_canonical_digest": registry["canonical_digest"],
        "ordered_sample_ids_sha256": registry["ordered_sample_ids_sha256"],
    }
    assert run_spec["execution_coordinates"] == {
        "block_index": config["primary_scope"]["block_index"],
        "head_index": config["primary_scope"]["head_index"],
        "target_id": config["primary_scope"]["target_id"],
        "method_ids": config["primary_scope"]["methods_in_image_group_order"],
        "method_order_within_image_group": "AS_LISTED",
        "statistical_unit": config["primary_scope"]["statistical_unit"],
    }

    audit = registry["audit_selection"]
    ranked = []
    for row in rows:
        material = (
            audit["namespace"] + registry["ordered_sample_ids_sha256"] + "|"
            + f"{row['selection_order']:03d}|{row['sample_id']}"
        )
        ranked.append((hashlib.sha256(material.encode("utf-8")).hexdigest(), row))
    selected = sorted(ranked, key=lambda item: (bytes.fromhex(item[0]), item[1]["selection_order"]))[:5]
    assert [digest for digest, _ in selected] == [entry["selection_digest"] for entry in audit["entries"]]
    assert [row["sample_id"] for _, row in selected] == audit["selected_sample_ids"]
    assert [row["selection_order"] for _, row in selected] == config["audit"]["audit_selection_orders"]
    assert sum(row["audit_overlay"] for row in rows) == 5
    for entry in audit["entries"]:
        row = rows[entry["selection_order"]]
        assert (row["sample_id"], row["audit_overlay_id"]) == (
            entry["sample_id"], entry["audit_overlay_id"],
        )
    for field in ("audit_overlay_ids", "audit_sample_ids", "audit_selection_orders"):
        assert run_spec["audit"][field] == config["audit"][field]


def test_workload_counts_are_independently_recomputed_and_scope_is_closed() -> None:
    config = _load(CONFIG_PATH)
    run_spec = _load(RUN_SPEC_PATH)
    image_groups = len(run_spec["selection"]["sample_ids"])
    methods = len(run_spec["execution_coordinates"]["method_ids"])
    execution_units = image_groups * methods
    audit_repeat_maps = (
        len(run_spec["audit"]["audit_overlay_ids"]) * methods
        * (run_spec["audit"]["audit_repeatability_runs"] - 1)
    )
    map_items = execution_units + audit_repeat_maps
    curves = (
        execution_units * len(run_spec["fidelity"]["directions"])
        * len(run_spec["fidelity"]["tie_repeat_indices"])
    )
    curve_points = curves * len(run_spec["fidelity"]["curve_fractions"])
    expected = {
        "image_groups": image_groups,
        "methods_per_image_group": methods,
        "execution_units": execution_units,
        "primary_clean_maps": execution_units,
        "audit_repeat_clean_maps": audit_repeat_maps,
        "map_work_items": map_items,
        "fidelity_curves": curves,
        "fidelity_curve_points": curve_points,
        "logical_work_items": map_items + curve_points,
        "logical_work_items_are_forward_or_backward_calls": False,
    }
    assert expected == {
        "image_groups": 500, "methods_per_image_group": 2,
        "execution_units": 1000, "primary_clean_maps": 1000,
        "audit_repeat_clean_maps": 10, "map_work_items": 1010,
        "fidelity_curves": 10000, "fidelity_curve_points": 210000,
        "logical_work_items": 211010,
        "logical_work_items_are_forward_or_backward_calls": False,
    }
    assert run_spec["workload"] == expected
    assert {key: config["workload"][key] for key in expected} == expected

    assert all(value is False or value == [] for value in config["forbidden_workload"].values())
    assert all(value is False or value == [] for value in run_spec["forbidden_workload"].values())
    assert config["secondary_and_spatial"]["run_stability"] is False
    assert config["secondary_and_spatial"]["secondary_inference_enabled"] is False
    assert config["registered_derived_metrics"]["additional_curve_points_or_work_items"] == 0
    assert config["audit"]["statistical_n_increment"] == 0
    assert config["cohort"]["c10_and_stage_d_pooling_allowed"] is False
    assert config["cohort"]["sample_replacement_allowed"] is False
    assert config["cohort"]["complete_case_analysis_allowed"] is False


def test_statistics_rationale_golden_seeds_and_decision_equalities() -> None:
    config = _load(CONFIG_PATH)
    registry = _load(D1_REGISTRY_PATH)
    rationale = config["sample_size_rationale"]
    assert rationale["directional_consistency_fraction"] == 400 / 500 == 12 / 15
    assert rationale["positive_count_minimum"] == 400
    assert rationale["formal_power_claim_made"] is False
    assert rationale["binomial_worst_case_standard_error_at_n500"] == pytest.approx(math.sqrt(0.25 / 500))
    assert rationale["binomial_worst_case_normal_95_half_width"] == pytest.approx(1.96 * math.sqrt(0.25 / 500))
    assert rationale["binomial_standard_error_at_p0_8_n500"] == pytest.approx(math.sqrt(0.8 * 0.2 / 500))
    assert rationale["binomial_normal_95_half_width_at_p0_8"] == pytest.approx(1.96 * math.sqrt(0.8 * 0.2 / 500))

    fidelity = config["fidelity"]
    tie_material = (
        fidelity["tie_seed_namespace"] + registry["rows"][0]["sample_id"] + "|"
        + str(config["primary_scope"]["block_index"]) + "|"
        + str(config["primary_scope"]["head_index"]) + "|0"
    )
    tie_seed = int.from_bytes(hashlib.sha256(tie_material.encode()).digest()[:8], "big")
    assert str(tie_seed) == fidelity["tie_seed_golden_first_sample_repeat_0_decimal"]
    statistics = config["statistics"]
    bootstrap_material = statistics["bootstrap_seed_namespace"] + statistics["primary_metric_id"]
    bootstrap_seed = int.from_bytes(hashlib.sha256(bootstrap_material.encode()).digest()[:8], "big")
    assert str(bootstrap_seed) == statistics["bootstrap_seed_golden_decimal"]
    assert statistics["confirmatory_n"] == statistics["bootstrap_cluster_size"] == 500
    assert statistics["bootstrap_resamples"] == 10000
    assert statistics["bootstrap_draw_shape"] == [10000, 500]
    assert statistics["equivalence_test_performed"] is False
    assert statistics["descriptive_curve_band"]["enters_primary_decision"] is False

    gates = config["decision_gates"]
    assert gates["fidelity_evaluation_order"] == [
        "FAIL_FIDELITY", "PASS_FIDELITY", "INCONCLUSIVE_FIDELITY",
    ]
    assert _component_status(399, 1.0, 0.5, 1.5) == "FAIL_FIDELITY"
    assert _component_status(400, 0.02, 1e-12, 0.02) == "PASS_FIDELITY"
    assert _component_status(400, 0.02, 0.0, 0.02) == "INCONCLUSIVE_FIDELITY"
    assert _component_status(400, 0.019, 0.001, 0.02) == "INCONCLUSIVE_FIDELITY"
    assert sum(value > 0 for value in (-1.0, 0.0, 1.0)) == 1
    assert gates["delta_equal_zero_is_positive"] is False
    assert gates["ci_lower_equal_zero"] == "DOES_NOT_PASS"
    assert gates["ci_upper_equal_effect_threshold"] == "DOES_NOT_FAIL"
    assert gates["median_equal_effect_threshold"] == "SATISFIES_MEDIAN_PASS_COMPONENT"
    assert gates["allowed_primary_decisions"] == [
        "NO_GO_VALIDITY", "NO_GO_FIDELITY", "INCONCLUSIVE_FIDELITY",
        "NO_GO_COST", "GO_FIDELITY_SCOPE",
    ]
    assert gates["workflow_blocked_is_scientific_decision"] is False


def test_retry_and_fresh_root_semantics_are_complete_and_outcome_blind() -> None:
    retry = _load(CONFIG_PATH)["retry_and_recovery"]
    assert retry["retryable_infrastructure_failure_codes"] == [
        "SCHEDULER_PREEMPTION", "EXTERNAL_PROCESS_TERMINATION", "HOST_LOSS",
        "GPU_DEVICE_LOSS_OR_RESET", "TRANSIENT_STORAGE_IO_BEFORE_RESULT_PUBLISH",
        "WORKER_PROCESS_CRASH_BEFORE_RESULT_PUBLISH", "STALE_CLAIM_OWNER_CONFIRMED_DEAD",
    ]
    assert retry["nonretryable_failure_classes"] == [
        "SCIENTIFIC_INVALIDITY", "OUT_OF_MEMORY_FOR_FROZEN_WORKLOAD",
        "LAUNCH_SAFETY_CAP_BREACH", "AUTHORIZATION_OR_CONTRACT_DRIFT",
        "TAMPER_OR_UNSAFE_ROOT", "AMBIGUOUS_ATTEMPT_OR_OWNERSHIP",
        "MANUAL_CANCELLATION_AFTER_SCIENTIFIC_RESULT_ACCESS",
    ]
    assert retry["maximum_authorized_infrastructure_retries_per_unit"] == 1
    assert retry["maximum_attempts_per_unit"] == 2
    assert retry["retry_classification_occurs_before_scientific_payload_access"] is True
    assert retry["retry_authorization_binds_failure_code_prior_attempt_digest_and_same_coordinate"] is True
    assert retry["retry_uses_new_immutable_attempt_id"] is True
    assert retry["terminal_unit_overwrite_or_append_allowed"] is False
    assert retry["paired_unit_after_other_scientific_failure_must_run"] is True
    assert retry["fresh_root_required_for_full_campaign_rerun"] is True
    assert retry["fresh_root_requires_documented_diagnosis"] is True
    assert retry["fresh_root_requires_new_d7_authorization"] is True
    assert retry["fresh_root_reruns_all_500_images_and_1000_units"] is True
    assert retry["cross_root_splicing_allowed"] is False
    assert retry["superseded_roots_retained_and_disclosed"] is True


def test_retention_provenance_and_cost_categories_are_not_interchangeable() -> None:
    config = _load(CONFIG_PATH)
    retention = config["retention"]
    assert retention["l0_scope"].startswith("ALL_1000_UNITS")
    assert retention["l1_scope"].startswith("ALL_1000_PRIMARY")
    assert retention["l2_scope"].startswith("ONLY_FIVE_D1_AUDIT_IDS")
    assert retention["l3_scope"] == "NONE"
    assert retention["display_png_saved"] is False
    assert retention["execution_attempts_immutable"] is True
    assert retention["failure_table_retains_every_expected_unit"] is True
    assert retention["campaign_complete_requires_every_expected_coordinate_terminal"] is True
    assert retention["d10_numeric_source"] == "VALIDATED_SEALED_L0_AND_L1_ONLY_L2_EXCLUDED"

    provenance = config["provenance"]
    assert len(provenance["required_bindings"]) == len(set(provenance["required_bindings"])) == 9
    assert provenance["hash_algorithm"] == "sha256"
    assert provenance["ordinary_files_only_no_symlink_or_hardlink"] is True
    assert provenance["source_recomputing_validation_required"] is True

    cost = config["cost"]
    launch = cost["launch_safety_caps"]
    completed = cost["completed_campaign_cost_gate"]
    assert cost["old_c10_pass_cost_authorizes_stage_d"] is False
    assert cost["d7_independent_budget_required"] is True
    assert cost["d7_may_loosen_caps"] is False
    assert {
        "l0_l1_bytes_per_execution_unit_max", "l2_bytes_per_audit_image_max",
        "forward_calls_per_execution_unit_max", "backward_calls_per_execution_unit_max",
        "cuda_peak_reserved_bytes_per_execution_unit_max",
    } < set(launch)
    assert {
        "stage_d_total_retained_regular_file_bytes_max",
        "stage_d_total_h20_gpu_hours_max",
        "stage_d_h20_gpu_hours_per_image_group_max",
    } < set(completed)
    assert launch["preflight_failure_action"] == "BLOCKED_NO_LAUNCH"
    assert launch["runtime_breach_is_no_go_cost"] is False
    assert completed["evaluated_only_after_complete_campaign"] is True
    assert completed["aggregate_overrun_alone_stops_in_progress_campaign"] is False
    assert completed["failure_status"] == "FAIL_COST"
    assert completed["decision_if_validity_and_fidelity_pass"] == "NO_GO_COST"
    assert cost["budget_status_at_d2"] == "NOT_EVALUATED"


def test_protocol_acceptance_contract_and_roadmap_are_consistent() -> None:
    documents = {path.name: path.read_text(encoding="utf-8") for path in DOCUMENT_PATHS}
    protocol = documents["阶段D_protocol.md"]
    acceptance = documents["stage_d_acceptance.md"]
    contract = documents["stage_d_config_contract.md"]
    for text in documents.values():
        assert "500" in text
        assert "400" in text
        assert "211,010" in text
        assert "D3" in text
    assert "execution_authorization=NOT_ISSUED" in protocol
    assert "run_stability=false" in protocol
    assert "NO_GO_NOVELTY" in protocol and "n=515" in protocol
    assert "D2 只冻结合同" in protocol
    assert "D2 does not inspect" in acceptance
    assert "D3 begins strict loading/planning" in contract
    roadmap = (REPO_ROOT / "docs/阶段D-step-by-step.md").read_text(encoding="utf-8")
    assert "| D2 | PASS |" in roadmap
    assert "metadata/static/synthetic fixtures" in roadmap
    assert "不要开始 D3，不要实现 loader/planner/runner/finalize/aggregate" in roadmap


def test_unsigned_tamper_and_self_resigned_source_drift_fail_closed() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    config = _strict_json_bytes(config_bytes, "stage_d.json")
    unsigned = copy.deepcopy(config)
    unsigned["authority"]["execution_authorization"] = "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"
    with pytest.raises(ContractError, match="canonical digest mismatch"):
        _assert_self_digest(unsigned, "unsigned config")

    run_spec = _load(RUN_SPEC_PATH)
    run_schema = _load(RUN_SPEC_SCHEMA_PATH)
    resigned = copy.deepcopy(run_spec)
    resigned["config_binding"]["config_file_sha256"] = "0" * 64
    resigned["canonical_digest"] = _self_digest(resigned)
    Draft202012Validator(run_schema).validate(resigned)
    _assert_self_digest(resigned, "self-resigned run spec")
    with pytest.raises(ContractError, match="source binding drifted"):
        _assert_run_config_binding(resigned, config, config_bytes)
