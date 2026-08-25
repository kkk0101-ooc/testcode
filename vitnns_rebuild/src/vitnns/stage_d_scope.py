"""Fail-closed contract helpers for the Stage D D0 scope decision.

D0 is a governance boundary, not an execution boundary.  This module validates
the checked-in scope record and can independently audit the already sealed C10
package without rebuilding C10 statistics, opening Stage D reserve images, or
constructing a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import errno
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import REPO_ROOT
class StageDScopeError(ValueError):
    """Raised when the D0 scope record or its sealed source is invalid."""


DEFAULT_SCOPE_DECISION = REPO_ROOT / "config/stage_d_scope_decision.json"
DEFAULT_SCOPE_SCHEMA = REPO_ROOT / "config/stage_d_scope_decision.schema.json"
DEFAULT_SCOPE_DOCUMENT = REPO_ROOT / "docs/stage_d_scope_decision.md"

C10_FORMAL_ROOT = Path(
    "/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824"
)
C10_COMPLETION_COMMIT = "69e1fa887862e00da4441842c6de149fe2424225"
C10_PRODUCER_COMMIT = "92ad81342ff22e2710df495a9961a3ab8c015a3a"
C10_PRODUCER_TREE = "3551096e358391fe04952e8ced19ea60884fb32f"
C10_SOURCE_TREE = "6874e71aec0e592e110a540a1638ea2cfc697c8d"
C10_ARCHIVE_REF = "refs/archive/c10-aggregation-producer"
C10_ROOT_FILE_COUNT = 12
C10_ROOT_TOTAL_BYTES = 524_796_892
C10_ROOT_INVENTORY_DIGEST = (
    "382992d94f071ce46164b92ab9eed6a60743a30fdab0bf73253bab95156986a4"
)
D0_SCOPE_CANONICAL_DIGEST = (
    "7508dcb7c063eaeecfc41f2f63f18e70dd5d660bec06c50e7b4b893d9a48ac5f"
)

C10_OUTPUT_FILES = (
    "bootstrap.json",
    "decision.json",
    "holm.json",
    "manifest.json",
    "per_sample.csv",
    "per_sample.schema.json",
    "per_sample_manifest.json",
    "provenance.json",
    "reconciliation.json",
    "stage_c_report.json",
    "status.json",
    "summaries.json",
)

C10_KEY_FILE_SHA256 = {
    "status.json": "192d383d5c4788f612018532074c06d48511513edd7565dbb99934706cd96db2",
    "stage_c_report.json": "9a16034d29bc87eaba852aa0e618928ac0e4ecee9bf745d765f42311dd4563b4",
    "decision.json": "07fc5875e46018adf5dc0cd09967e273df3b443c1deeb1653d672d05103e513e",
    "holm.json": "be37ffbfd2ba60b5549ba791076383d9ba53a3b437178273a608fbf527b91781",
    "provenance.json": "4dcca54fdc83ba7401070ef843636b5e67b68a8ce720fa70f729e407fc5717f0",
    "manifest.json": "507ca3c4ac210882a5918cc602cf7a4656aa398af76ff9f3638a77f3af42a0d8",
}

C10_KEY_CANONICAL_DIGESTS = {
    "status.json": "d08bcb62a0fa95fef81c75d955b88082a8e6788a76b1a0bfffe1aa4e0181eb87",
    "stage_c_report.json": "90232df2010f6a8fc771baeac3aaac44fee93d436597cf0cbdebe3cf8d2142e2",
    "decision.json": "60e112d50f22229474a634039ba71c7eed1adba081d8e1097f850a8a4b4d8dc0",
    "holm.json": "271ba3625de6a91912728a32880cf1e1861aecb9f12903cf759e83e3332ab2fd",
    "provenance.json": "4b26452041685cbb6c12d93139736e65fc4330fd44a64bf854848e7cff66c742",
    "manifest.json": "c71ae7c19941eab556a0ed9fdde553cb1a3fae914e149652df9b31cf3538720a",
}


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode metadata deterministically without importing the model stack."""

    if not isinstance(value, Mapping):
        raise StageDScopeError("canonical JSON input must be a mapping")

    def validate(item: Any, context: str) -> None:
        if item is None or type(item) in {str, bool, int}:
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise StageDScopeError(f"{context} contains a non-finite number")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise StageDScopeError(f"{context} keys must be non-empty strings")
                validate(child, f"{context}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                validate(child, f"{context}[{index}]")
            return
        raise StageDScopeError(
            f"{context} contains non-JSON-safe {type(item).__name__}"
        )

    validate(value, "canonical JSON input")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

ALLOWED_CLAIM_IDS = (
    "C10_PRIMARY_DECISION_NO_GO_NOVELTY",
    "C10_ACCEPTANCE_BLOCKED",
    "C10_HOLM_BLOCKER_UNDERSPECIFIED",
    "C10_SPATIAL_OVERALL_FAIL",
    "C10_SPATIAL_COMPONENTS_MIXED",
    "C10_DOWNSTREAM_NOT_AUTHORIZED",
    "C10_PRIMARY_FIDELITY_COMPONENT_VALID",
    "C10_PRIMARY_FIDELITY_COMPONENT_EFFECT_PASS",
    "STAGE_D_IS_PROSPECTIVE_REPLICATION_EXTENSION",
    "NO_INTERMEDIATE_C_PLUS_REQUIRED",
    "STAGE_D_PRIMARY_SCOPE_DEFINED",
    "FINE_ENDPOINT_IS_4X4_CELL",
    "STAGE_D_SPATIAL_NOT_IN_SCOPE",
    "STAGE_D_HOLM_NOT_APPLICABLE_CONDITIONALLY",
    "APPROVED_TO_PREPARE_STAGE_D",
)

DENIED_CLAIM_IDS = (
    "C10_GO",
    "C10_ACCEPTANCE_PASS",
    "C10_DOWNSTREAM_AUTHORIZED",
    "C10_STAGE_D_STARTED",
    "C10_HOLM_BLOCKER_RESOLVED_OR_CANCELLED",
    "C10_REANALYZED_AS_FIDELITY_ONLY",
    "STAGE_D_EXECUTION_AUTHORIZED_AT_D0",
    "STAGE_D_ANALYSIS_AUTHORIZED_AT_D0",
    "STAGE_D_EFFECT_RESULT_AVAILABLE_AT_D0",
    "STAGE_D_VALIDITY_RESULT_AVAILABLE_AT_D0",
    "STAGE_D_COST_RESULT_AVAILABLE_AT_D0",
    "RESERVE_ABSOLUTELY_NEVER_VIEWED",
    "SPATIAL_ROBUSTNESS_PROVEN",
    "TRANSLATION_ROBUSTNESS_PROVEN",
    "CROP_SCALE_ROBUSTNESS_PROVEN",
    "ALL_SPATIAL_ROBUSTNESS_FAILED",
    "NATIVE_SINGLE_PIXEL_ATTRIBUTION",
    "COMPLETE_QKV_ATTRIBUTION",
    "COMPLETE_ATTENTION_EXPLANATION",
    "CLASS_LOGIT_EXPLANATION",
    "STATISTICAL_EQUIVALENCE",
    "UNIVERSAL_IMAGENET_SUPERIORITY",
    "C10_STAGE_D_POOLED_N515",
    "MISCLASSIFIED_SAMPLES_EXCLUDED",
    "COMPLETE_CASE_EFFECT_ANALYSIS",
    "FAILED_SAMPLES_IMPUTED_OR_REPLACED",
    "POSTHOC_PRIMARY_HEAD_SELECTION",
    "SECONDARY_INFERENCE_WITHOUT_MULTIPLICITY_CONTRACT",
    "DISPLAY_ARTIFACT_USED_AS_QUANTITATIVE_INPUT",
    "C10_PASS_COST_AUTHORIZES_STAGE_D_COST",
    "STAGE_D_SPATIAL_RESULT",
    "PATCH_SCALE_OR_OTHER_METHOD_AS_STAGE_D_PRIMARY",
)

FUTURE_CONDITIONAL_CLAIM_IDS = (
    "STAGE_D_FIDELITY_REPLICATION_PASS",
    "STAGE_D_FIDELITY_REPLICATION_FAIL",
    "STAGE_D_FIDELITY_INCONCLUSIVE",
    "STAGE_D_VALIDITY_FAIL",
    "STAGE_D_COST_FAIL",
)

MANDATORY_DISCLOSURE_IDS = (
    "DISCLOSE_POST_C10_SCOPE_CHANGE",
    "DISCLOSE_C10_COMPOSITE_STATUS",
    "DISCLOSE_C10_FIDELITY_BOUNDARY",
    "DISCLOSE_C10_SPATIAL_MIXED_RESULT",
    "DISCLOSE_NO_GENERAL_SPATIAL_CLAIM",
    "DISCLOSE_4X4_NOT_NATIVE_PIXEL",
    "DISCLOSE_VALUE_PATH_TARGET_LIMIT",
    "DISCLOSE_COHORT_SEPARATION",
    "DISCLOSE_RESERVE_LEDGER_SCOPE",
    "DISCLOSE_STAGE_D_NO_SPATIAL_ENDPOINT",
    "DISCLOSE_HOLM_SEPARATION",
    "DISCLOSE_MISCLASSIFICATION_POLICY",
    "DISCLOSE_FAILURE_POLICY",
    "DISCLOSE_COST_EVIDENCE_BOUNDARY",
)

def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StageDScopeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise StageDScopeError(f"non-finite JSON number is forbidden: {token}")


def _strict_json_bytes(payload: bytes, context: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        raise StageDScopeError(f"{context} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise StageDScopeError(f"{context} is invalid JSON: {error}") from error


def strict_json_load(path: Path | str) -> Any:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise StageDScopeError(f"JSON file is missing: {resolved}")
    return _strict_json_bytes(resolved.read_bytes(), str(resolved))


def scope_decision_digest(document: Mapping[str, Any]) -> str:
    if not isinstance(document, Mapping):
        raise StageDScopeError("scope decision must be a mapping")
    semantic = {key: value for key, value in document.items() if key != "canonical_digest"}
    return canonical_sha256(semantic)


def _require_exact_keys(value: Any, keys: Sequence[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageDScopeError(f"{context} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise StageDScopeError(
            f"{context} fields drifted: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected == "boolean":
        return isinstance(value, bool)
    return False


def _validate_schema_value(value: Any, rule: Any, context: str) -> None:
    if not isinstance(rule, Mapping):
        raise StageDScopeError(f"{context} schema rule must be an object")
    allowed_rule_keys = {
        "$schema", "$id", "title", "description",
        "type", "const", "enum", "pattern", "minimum", "maximum",
        "minItems", "maxItems", "uniqueItems", "items", "required",
        "properties", "additionalProperties",
    }
    if not set(rule) <= allowed_rule_keys:
        raise StageDScopeError(
            f"{context} schema uses unsupported keys: {sorted(set(rule) - allowed_rule_keys)}"
        )
    if "const" in rule and value != rule["const"]:
        raise StageDScopeError(f"{context} differs from its frozen value")
    if "enum" in rule and value not in rule["enum"]:
        raise StageDScopeError(f"{context} is outside its enum")
    expected_type = rule.get("type")
    if expected_type is not None and not _schema_type_matches(value, expected_type):
        raise StageDScopeError(f"{context} must have schema type {expected_type}")
    if isinstance(value, str) and "pattern" in rule:
        if re.fullmatch(rule["pattern"], value) is None:
            raise StageDScopeError(f"{context} does not match its pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in rule and value < rule["minimum"]:
            raise StageDScopeError(f"{context} is below its minimum")
        if "maximum" in rule and value > rule["maximum"]:
            raise StageDScopeError(f"{context} is above its maximum")
    if isinstance(value, list):
        if "minItems" in rule and len(value) < rule["minItems"]:
            raise StageDScopeError(f"{context} has too few items")
        if "maxItems" in rule and len(value) > rule["maxItems"]:
            raise StageDScopeError(f"{context} has too many items")
        if rule.get("uniqueItems") is True:
            rendered = [
                json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                for item in value
            ]
            if len(rendered) != len(set(rendered)):
                raise StageDScopeError(f"{context} items must be unique")
        if "items" in rule:
            for index, item in enumerate(value):
                _validate_schema_value(item, rule["items"], f"{context}[{index}]")
    if isinstance(value, Mapping):
        properties = rule.get("properties", {})
        required = set(rule.get("required", ()))
        if not isinstance(properties, Mapping):
            raise StageDScopeError(f"{context} schema properties must be an object")
        missing = required - set(value)
        if missing:
            raise StageDScopeError(f"{context} is missing fields: {sorted(missing)}")
        if rule.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise StageDScopeError(f"{context} has unknown fields: {sorted(unknown)}")
        for key, item in value.items():
            if key in properties:
                _validate_schema_value(item, properties[key], f"{context}.{key}")


def _validate_exact_instance_schema(document: Mapping[str, Any], schema: Any) -> None:
    schema = _require_exact_keys(
        schema,
        ("$schema", "$id", "title", "description", "type",
         "additionalProperties", "required", "properties"),
        "D0 schema",
    )
    if schema["$schema"] != "https://json-schema.org/draft/2020-12/schema":
        raise StageDScopeError("D0 schema draft drifted")
    if schema["type"] != "object" or schema["additionalProperties"] is not False:
        raise StageDScopeError("D0 schema must be a closed object")
    properties = schema["properties"]
    if not isinstance(properties, Mapping):
        raise StageDScopeError("D0 schema properties must be an object")
    if set(schema["required"]) != set(properties):
        raise StageDScopeError("D0 schema required/properties fields drifted")
    if set(document) != set(properties):
        raise StageDScopeError(
            "D0 schema/document top-level fields drifted: "
            f"missing={sorted(set(properties) - set(document))}, "
            f"unknown={sorted(set(document) - set(properties))}"
        )
    _validate_schema_value(document, schema, "D0 scope decision")


def _validate_semantics(document: Mapping[str, Any]) -> None:
    if document["schema_version"] != 1:
        raise StageDScopeError("D0 schema version drifted")
    if document["document_type"] != "stage_d_scope_decision":
        raise StageDScopeError("D0 document type drifted")
    if (document["stage"], document["milestone"], document["task_status"]) != (
        "D", "D0", "PASS"
    ):
        raise StageDScopeError("D0 identity/status drifted")
    if document["contract"] != {
        "human_document_path": "docs/stage_d_scope_decision.md",
        "machine_record_is_authoritative": True,
        "machine_record_path": "config/stage_d_scope_decision.json",
        "schema_path": "config/stage_d_scope_decision.schema.json",
    }:
        raise StageDScopeError("D0 single-authority contract drifted")

    authority = document["authority"]
    if authority["preparation_authorization"] != "APPROVED_TO_PREPARE_STAGE_D":
        raise StageDScopeError("D0 preparation authorization drifted")
    if authority["execution_authorization"] != "NOT_ISSUED":
        raise StageDScopeError("D0 must not issue execution authorization")
    if authority["analysis_authorization"] != "NOT_EVALUATED":
        raise StageDScopeError("D0 must not issue analysis authorization")
    if authority["authorized_next_task"] != "D1":
        raise StageDScopeError("D0 may only hand off to D1")
    if authority["does_not_override_c10"] is not True:
        raise StageDScopeError("D0 must not override C10")
    if authority["preparation_scope"] != [
        "D1_METADATA_ONLY_REGISTRY_PREPARATION",
        "D2_PROTOCOL_AND_STATISTICS_FREEZE",
        "D3_TO_D6_IMPLEMENTATION_WITH_SYNTHETIC_OR_METADATA_INPUTS",
        "D7_NON_RESERVE_SMOKE_BUDGET_AND_EXECUTION_AUTHORIZATION_GATE",
    ]:
        raise StageDScopeError("D0 preparation scope drifted")
    if document["blocking_reasons"] != []:
        raise StageDScopeError("passing D0 must not retain current blocking reasons")

    attempt_history = document["attempt_history"]
    expected_attempt_history = {
        "current_attempt_id": "D0_MODEL_FREE_RERUN_20260825",
        "records": [
            {
                "attempt_id": "D0_INITIAL_ATTEMPT_20260825",
                "status": "BLOCKED",
                "authoritative_for_current_authorization": False,
                "blocking_reasons": [
                    "D0_NO_MODEL_OR_INFERENCE_GATE_VIOLATED",
                    "D0_NO_JPEG_DECODE_GATE_VIOLATED",
                ],
                "access_facts": {
                    "gpu_used": False,
                    "jpeg_decode_performed": True,
                    "model_inference_performed": True,
                    "model_loaded": True,
                    "model_or_gpu_used": True,
                    "non_reserve_jpeg_bytes_accessed": True,
                    "reserve_image_bytes_accessed": False,
                    "reserve_metadata_contents_accessed": False,
                    "reserve_result_accessed": False,
                },
            },
            {
                "attempt_id": "D0_MODEL_FREE_RERUN_20260825",
                "status": "PASS",
                "authoritative_for_current_authorization": True,
                "blocking_reasons": [],
                "access_facts": {
                    "gpu_used": False,
                    "jpeg_decode_performed": False,
                    "model_inference_performed": False,
                    "model_loaded": False,
                    "model_or_gpu_used": False,
                    "non_reserve_jpeg_bytes_accessed": False,
                    "reserve_image_bytes_accessed": False,
                    "reserve_metadata_contents_accessed": False,
                    "reserve_result_accessed": False,
                },
            },
        ],
    }
    if attempt_history != expected_attempt_history:
        raise StageDScopeError("D0 attempt history drifted")

    history = document["source_c10"]["historical_status"]
    expected_history = {
        "primary_decision": "NO_GO_NOVELTY",
        "stage_c_acceptance": "BLOCKED",
        "workflow_blocker": "HOLM_BLOCKED_UNDERSPECIFIED",
        "validity_status": "PASS_VALIDITY",
        "effect_status": "PASS_EFFECT",
        "spatial_status": "FAIL_SPATIAL",
        "cost_status": "PASS_COST",
        "downstream_authorized": False,
        "stage_d_started": False,
        "holm_performed": False,
        "equivalence_test_performed": False,
    }
    if history != expected_history:
        raise StageDScopeError("sealed C10 historical status was rewritten")

    source = document["source_c10"]
    if source["formal_root"] != str(C10_FORMAL_ROOT):
        raise StageDScopeError("C10 formal root drifted")
    if source["completion_commit"] != C10_COMPLETION_COMMIT:
        raise StageDScopeError("C10 completion commit drifted")
    producer = source["aggregation_producer"]
    if producer != {
        "commit": C10_PRODUCER_COMMIT,
        "archive_ref": C10_ARCHIVE_REF,
        "tree": C10_PRODUCER_TREE,
        "source_tree": C10_SOURCE_TREE,
    }:
        raise StageDScopeError("C10 producer binding drifted")
    inventory = source["whole_root_inventory"]
    if inventory != {
        "file_count": C10_ROOT_FILE_COUNT,
        "total_bytes": C10_ROOT_TOTAL_BYTES,
        "canonical_digest": C10_ROOT_INVENTORY_DIGEST,
    }:
        raise StageDScopeError("C10 whole-root inventory binding drifted")
    expected_file_bindings = {
        name: {
            "sha256": C10_KEY_FILE_SHA256[name],
            "canonical_digest": C10_KEY_CANONICAL_DIGESTS[name],
        }
        for name in C10_KEY_FILE_SHA256
    }
    if source["files"] != expected_file_bindings:
        raise StageDScopeError("C10 key-file bindings drifted")

    basis = document["scientific_basis"]
    if basis["c10_component_status"] != {
        "validity": "PASS_VALIDITY", "effect": "PASS_EFFECT"
    }:
        raise StageDScopeError("C10 component-level scientific basis drifted")
    if basis["c10_effect"] != {
        "sample_count": 15,
        "median_delta_fid": 1.7534972022995547,
        "positive_count": 15,
        "bootstrap_ci_95": [1.3317595184568727, 3.3813212483296478],
        "delta_fid_definition": (
            "0.5*((insertion_auc_nns-insertion_auc_matched)+"
            "(deletion_auc_matched-deletion_auc_nns))"
        ),
    }:
        raise StageDScopeError("C10 effect evidence binding drifted")
    if basis["intermediate_c_plus_required"] is not False:
        raise StageDScopeError("D0 must not add an intermediate C+ gate")

    scope = document["stage_d_scope"]
    if scope["primary"] != {"block_index": 0, "head_index": 9}:
        raise StageDScopeError("Stage D primary must remain (0,9)")
    if scope["methods"] != [
        "conditional_fixed_clean_attention_head_direction_value_path_ig",
        "matched_fixed_clean_attention_patch_token_contribution",
    ]:
        raise StageDScopeError("Stage D method pair drifted")
    if scope["target_id"] != "fixed_clean_attention_cls_head_direction":
        raise StageDScopeError("Stage D target drifted")
    if scope["endpoint"]["id"] != "fine_4x4_cell_delta_fid":
        raise StageDScopeError("Stage D primary endpoint drifted")
    if scope["fidelity_role"] != "PRIMARY_HARD_SCIENTIFIC_GATE":
        raise StageDScopeError("Stage D fidelity must remain the hard scientific gate")
    if scope["run_stability"] is not False or scope["spatial_in_primary_decision"] is not False:
        raise StageDScopeError("Stage D D0 scope must exclude spatial execution/inference")
    if scope["spatial_role"] != "MANDATORY_DISCLOSURE_ONLY_NO_NEW_ENDPOINT":
        raise StageDScopeError("Stage D spatial role drifted")
    if scope["secondary_inference_enabled"] is not False:
        raise StageDScopeError("D0 scope does not permit secondary inference")
    if scope["holm_status"] != "NOT_APPLICABLE_NO_SECONDARY_INFERENCE":
        raise StageDScopeError("Stage D Holm scope drifted")
    if scope["prospective_cohort"]["planned_image_count"] != 500:
        raise StageDScopeError("Stage D planned cohort size drifted")
    if scope["prospective_cohort"]["registry_status"] != "NOT_FORMALIZED_D0":
        raise StageDScopeError("D0 must not formalize the reserve registry")
    if scope["result_status"] != "NOT_EVALUATED":
        raise StageDScopeError("D0 must not report a Stage D result")
    if scope["cohort_pooling_allowed"] is not False:
        raise StageDScopeError("C10 and Stage D cohorts must remain separate")

    claims = document["claim_boundaries"]
    if tuple(claims["allowed_claim_ids"]) != ALLOWED_CLAIM_IDS:
        raise StageDScopeError("D0 claim allowlist drifted")
    if tuple(claims["denied_claim_ids"]) != DENIED_CLAIM_IDS:
        raise StageDScopeError("D0 claim denylist drifted")
    if tuple(claims["future_conditional_claim_ids"]) != FUTURE_CONDITIONAL_CLAIM_IDS:
        raise StageDScopeError("D0 future claim list drifted")
    if tuple(claims["emitted_claim_ids"]) != ALLOWED_CLAIM_IDS:
        raise StageDScopeError("D0 emitted claims exceed or omit the allowlist")
    if set(ALLOWED_CLAIM_IDS) & set(DENIED_CLAIM_IDS):
        raise StageDScopeError("D0 claim allowlist and denylist overlap")
    if set(FUTURE_CONDITIONAL_CLAIM_IDS) & set(claims["emitted_claim_ids"]):
        raise StageDScopeError("future Stage D result claims were emitted at D0")

    disclosure_ids = tuple(item["id"] for item in document["mandatory_disclosures"])
    if disclosure_ids != MANDATORY_DISCLOSURE_IDS:
        raise StageDScopeError("D0 mandatory disclosures drifted")

    access = document["access_attestation"]
    required_false = (
        "reserve_metadata_contents_accessed",
        "reserve_image_bytes_accessed",
        "reserve_result_accessed",
        "stage_d_outcome_available",
        "gpu_used",
        "jpeg_decode_performed",
        "model_loaded",
        "model_inference_performed",
        "model_or_gpu_used",
        "non_reserve_jpeg_bytes_accessed",
        "c9_evidence_modified",
        "c10_evidence_modified",
        "paper_data_or_figures_generated",
    )
    if access["attestation_scope"] != "CURRENT_D0_MODEL_FREE_RERUN_ONLY":
        raise StageDScopeError("D0 current-attempt attestation scope drifted")
    if access["reserve_access_state"] != "METADATA_EXISTENCE_ONLY":
        raise StageDScopeError("D0 rerun reserve access state drifted")
    if any(access[key] is not False for key in required_false):
        raise StageDScopeError("D0 access attestation contains a forbidden access/use")

    verification_policy = document["verification_policy"]
    if verification_policy != {
        "profile_id": "D0_MODEL_FREE_NO_JPEG_V1",
        "allowlist_closed": True,
        "allowed_pytest_node_ids": [
            "tests/test_stage_d_scope_contract.py::test_scope_schema_is_closed_draft_2020_12",
            "tests/test_stage_d_scope_contract.py::test_checked_in_scope_decision_matches_schema_and_semantics",
            "tests/test_stage_d_scope_contract.py::test_scope_import_does_not_load_model_or_torch_stack",
            "tests/test_stage_d_scope_contract.py::test_scope_document_and_machine_record_agree",
            "tests/test_stage_d_scope_contract.py::test_scope_markdown_projection_tamper_is_rejected",
            "tests/test_stage_d_scope_contract.py::test_scope_rejects_schema_and_authorization_drift",
            "tests/test_stage_d_scope_contract.py::test_scope_rejects_attempt_history_or_allowlist_tamper",
            "tests/test_stage_d_scope_contract.py::test_scope_rejects_unsigned_tamper",
            "tests/test_stage_d_scope_contract.py::test_scope_rejects_resigned_semantic_tamper",
            "tests/test_stage_d_scope_contract.py::test_strict_loader_rejects_duplicate_and_nonfinite_json",
            "tests/test_stage_d_scope_contract.py::test_claim_sets_and_access_attestation_are_exact",
            "tests/test_stage_d_scope_contract.py::test_d0_static_files_keep_execution_and_reserve_closed",
        ],
        "full_regression_suite_allowed": False,
        "pytest_plugin_autoload": False,
        "runner_command_template": (
            "VITNNS_D0_STRICT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
            "CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests/d0_guard:src "
            "/data00/home/yeziqi.831/miniforge3/envs/vitnns/bin/python "
            "-m pytest -ra <allowed_pytest_node_ids>"
        ),
        "safety_guard_path": "tests/d0_guard/sitecustomize.py",
        "cuda_visible_devices": "",
        "model_and_image_stack_imports_forbidden": [
            "PIL", "cv2", "timm", "torch", "torchvision",
            "vitnns.modeling",
        ],
        "sealed_c10_validation_mode": (
            "BYTE_HASH_CANONICAL_MANIFEST_STATUS_INVENTORY_AND_GIT_REF_ONLY"
        ),
        "source_scientific_recomputation_allowed": False,
    }:
        raise StageDScopeError("D0 verification policy drifted")


def validate_stage_d_scope_decision(
    document: Mapping[str, Any], *, schema_path: Path | str = DEFAULT_SCOPE_SCHEMA,
) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise StageDScopeError("scope decision must be a JSON object")
    observed = document.get("canonical_digest")
    expected = scope_decision_digest(document)
    if observed != expected:
        raise StageDScopeError(
            f"D0 canonical digest mismatch: expected {expected}, observed {observed}"
        )
    schema = strict_json_load(schema_path)
    _validate_exact_instance_schema(document, schema)
    _validate_semantics(document)
    if observed != D0_SCOPE_CANONICAL_DIGEST:
        raise StageDScopeError(
            "D0 scope decision differs from the trusted D0 semantic digest"
        )
    return dict(document)


def scope_document_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    """Build the structured Markdown projection of every scope section."""

    validate_stage_d_scope_decision(document)
    return {
        "schema_version": 1,
        "document_type": "stage_d_scope_markdown_projection",
        "machine_record_path": document["contract"]["machine_record_path"],
        "machine_record_canonical_digest": document["canonical_digest"],
        "task_status": document["task_status"],
        "blocking_reasons": document["blocking_reasons"],
        "authority": document["authority"],
        "attempt_history": document["attempt_history"],
        "source_c10_historical_status": document["source_c10"]["historical_status"],
        "source_c10_binding_digest": canonical_sha256(document["source_c10"]),
        "scientific_basis_digest": canonical_sha256(document["scientific_basis"]),
        "stage_d_scope_digest": canonical_sha256(document["stage_d_scope"]),
        "allowed_claim_ids_digest": canonical_sha256({
            "ids": document["claim_boundaries"]["allowed_claim_ids"]
        }),
        "denied_claim_ids_digest": canonical_sha256({
            "ids": document["claim_boundaries"]["denied_claim_ids"]
        }),
        "future_conditional_claim_ids_digest": canonical_sha256({
            "ids": document["claim_boundaries"]["future_conditional_claim_ids"]
        }),
        "mandatory_disclosures_digest": canonical_sha256({
            "records": document["mandatory_disclosures"]
        }),
        "access_attestation": document["access_attestation"],
        "verification_policy": document["verification_policy"],
    }


_PROJECTION_BEGIN = "<!-- STAGE_D_SCOPE_PROJECTION_BEGIN -->"
_PROJECTION_END = "<!-- STAGE_D_SCOPE_PROJECTION_END -->"


def extract_scope_document_projection(markdown: str) -> dict[str, Any]:
    """Parse the one controlled JSON projection embedded in the D0 document."""

    if not isinstance(markdown, str):
        raise StageDScopeError("scope Markdown must be text")
    if markdown.count(_PROJECTION_BEGIN) != 1 or markdown.count(_PROJECTION_END) != 1:
        raise StageDScopeError("scope Markdown must contain one structured projection")
    body = markdown.split(_PROJECTION_BEGIN, 1)[1].split(_PROJECTION_END, 1)[0].strip()
    if not body.startswith("```json\n") or not body.endswith("\n```"):
        raise StageDScopeError("scope Markdown projection fence is malformed")
    payload = body[len("```json\n"):-len("\n```")].encode("utf-8")
    parsed = _strict_json_bytes(payload, "scope Markdown projection")
    if not isinstance(parsed, dict):
        raise StageDScopeError("scope Markdown projection must be an object")
    return parsed


def load_stage_d_scope_decision(
    path: Path | str = DEFAULT_SCOPE_DECISION, *,
    schema_path: Path | str = DEFAULT_SCOPE_SCHEMA,
) -> dict[str, Any]:
    document = strict_json_load(path)
    return validate_stage_d_scope_decision(document, schema_path=schema_path)


def _open_sealed_root(root: Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(root)))
    if not absolute.is_absolute():
        raise StageDScopeError("sealed C10 root must resolve to an absolute path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        current_fd = os.open(absolute.anchor, flags)
    except OSError as error:
        raise StageDScopeError("cannot open sealed C10 filesystem anchor") from error
    try:
        for part in absolute.parts[1:]:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as error:
                raise StageDScopeError(
                    f"sealed C10 path contains a missing, symlink, or unsafe component: {part}"
                ) from error
            os.close(current_fd)
            current_fd = next_fd
        info = os.fstat(current_fd)
        if not stat.S_ISDIR(info.st_mode):
            raise StageDScopeError("sealed C10 root must be a directory")
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _safe_open_regular(root_fd: int, name: str) -> tuple[int, os.stat_result]:
    try:
        before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except OSError as error:
        raise StageDScopeError(f"cannot inspect sealed C10 entry: {name}") from error
    if stat.S_ISLNK(before.st_mode):
        raise StageDScopeError(f"sealed C10 entry is a symlink: {name}")
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise StageDScopeError(f"sealed C10 entry is not a single-link regular file: {name}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(name, flags, dir_fd=root_fd)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise StageDScopeError(f"sealed C10 entry is a symlink: {name}") from error
        raise StageDScopeError(f"cannot safely open sealed C10 entry: {name}") from error
    try:
        info = os.fstat(fd)
    except BaseException:
        os.close(fd)
        raise
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise StageDScopeError(f"sealed C10 entry is not a single-link regular file: {name}")
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_nlink)
    opened_identity = (info.st_dev, info.st_ino, info.st_size, info.st_nlink)
    if before_identity != opened_identity:
        os.close(fd)
        raise StageDScopeError(f"sealed C10 entry changed before open: {name}")
    return fd, info


def _safe_read_sealed_root(root: Path) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    root_fd = _open_sealed_root(root)
    try:
        actual_names = set(os.listdir(root_fd))
        if actual_names != set(C10_OUTPUT_FILES):
            raise StageDScopeError(
                "sealed C10 file set drifted: "
                f"missing={sorted(set(C10_OUTPUT_FILES) - actual_names)}, "
                f"extra={sorted(actual_names - set(C10_OUTPUT_FILES))}"
            )
        payloads: dict[str, bytes] = {}
        records: list[dict[str, Any]] = []
        for name in sorted(C10_OUTPUT_FILES):
            fd, before = _safe_open_regular(root_fd, name)
            try:
                with os.fdopen(fd, "rb", closefd=False) as handle:
                    payload = handle.read()
                after = os.fstat(fd)
            finally:
                os.close(fd)
            identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_nlink)
            identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_nlink)
            if identity_before != identity_after or len(payload) != before.st_size:
                raise StageDScopeError(f"sealed C10 entry changed while being read: {name}")
            payloads[name] = payload
            records.append({
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        if set(os.listdir(root_fd)) != actual_names:
            raise StageDScopeError("sealed C10 root changed while being read")
        return payloads, records
    finally:
        os.close(root_fd)


def _git_output(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def verify_c10_sealed_evidence(
    root: Path | str = C10_FORMAL_ROOT, *, repo_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify the sealed C10 package without recomputing scientific results."""

    sealed_root = Path(os.path.abspath(os.fspath(root)))
    payloads, records = _safe_read_sealed_root(sealed_root)
    file_hashes = {item["path"]: item["sha256"] for item in records}
    inventory_digest = canonical_sha256({"records": records})
    if (len(records), sum(item["bytes"] for item in records), inventory_digest) != (
        C10_ROOT_FILE_COUNT, C10_ROOT_TOTAL_BYTES, C10_ROOT_INVENTORY_DIGEST
    ):
        raise StageDScopeError("sealed C10 whole-root inventory drifted")
    for name, expected in C10_KEY_FILE_SHA256.items():
        if file_hashes[name] != expected:
            raise StageDScopeError(f"sealed C10 file hash drifted: {name}")

    json_documents: dict[str, Mapping[str, Any]] = {}
    canonical_digests: dict[str, str] = {}
    for name in sorted(payloads):
        if not name.endswith(".json"):
            continue
        value = _strict_json_bytes(payloads[name], name)
        if not isinstance(value, Mapping):
            raise StageDScopeError(f"sealed C10 JSON is not an object: {name}")
        json_documents[name] = value
        if "canonical_digest" in value:
            semantic = {key: item for key, item in value.items() if key != "canonical_digest"}
            actual_digest = canonical_sha256(semantic)
            if value["canonical_digest"] != actual_digest:
                raise StageDScopeError(f"sealed C10 canonical digest drifted: {name}")
            canonical_digests[name] = actual_digest
    for name, expected in C10_KEY_CANONICAL_DIGESTS.items():
        if canonical_digests.get(name) != expected:
            raise StageDScopeError(f"sealed C10 frozen canonical digest drifted: {name}")

    manifest = json_documents["manifest.json"]
    expected_payloads = {name for name in C10_OUTPUT_FILES if name not in {"manifest.json", "status.json"}}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or manifest.get("artifact_count") != 10:
        raise StageDScopeError("sealed C10 manifest count is invalid")
    if {item.get("path") for item in artifacts if isinstance(item, Mapping)} != expected_payloads:
        raise StageDScopeError("sealed C10 manifest payload set drifted")
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise StageDScopeError("sealed C10 manifest artifact is invalid")
        name = item["path"]
        record = next(record for record in records if record["path"] == name)
        if item.get("bytes") != record["bytes"] or item.get("sha256") != record["sha256"]:
            raise StageDScopeError(f"sealed C10 manifest binding drifted: {name}")

    status = json_documents["status.json"]
    report = json_documents["stage_c_report.json"]
    decision = json_documents["decision.json"]
    holm = json_documents["holm.json"]
    provenance = json_documents["provenance.json"]
    expected_status = {
        "complete": True,
        "terminal": True,
        "status": "BLOCKED",
        "stage_c_acceptance_status": "BLOCKED",
        "primary_decision": "NO_GO_NOVELTY",
        "downstream_authorized": False,
        "stage_d_started": False,
        "payload_count": 10,
        "manifest_sha256": C10_KEY_FILE_SHA256["manifest.json"],
    }
    if any(status.get(key) != value for key, value in expected_status.items()):
        raise StageDScopeError("sealed C10 status semantics drifted")
    if report.get("status") != "BLOCKED" or report.get("blocking_reasons") != [
        "HOLM_BLOCKED_UNDERSPECIFIED"
    ]:
        raise StageDScopeError("sealed C10 acceptance report drifted")
    if report.get("primary_decision") != "NO_GO_NOVELTY":
        raise StageDScopeError("sealed C10 report decision drifted")
    if decision.get("decision") != "NO_GO_NOVELTY" or decision.get("stage_c_acceptance_status") != "BLOCKED":
        raise StageDScopeError("sealed C10 decision status drifted")
    if decision.get("blocking_reasons") != ["HOLM_BLOCKED_UNDERSPECIFIED"]:
        raise StageDScopeError("sealed C10 decision blocker drifted")
    if decision.get("primary_config") != [0, 9] or decision.get("confirmatory_n") != 15:
        raise StageDScopeError("sealed C10 primary/cohort boundary drifted")
    if decision.get("confirmatory_role") != "holdout":
        raise StageDScopeError("sealed C10 confirmatory role drifted")
    flags = decision.get("flags", {})
    expected_flags = {
        "validity": "PASS_VALIDITY",
        "effect": "PASS_EFFECT",
        "spatial": "FAIL_SPATIAL",
        "cost": "PASS_COST",
    }
    if any(not isinstance(flags.get(key), Mapping) or flags[key].get("status") != value for key, value in expected_flags.items()):
        raise StageDScopeError("sealed C10 component status drifted")
    effect = flags["effect"]
    if (
        effect.get("median") != 1.7534972022995547
        or effect.get("positive_count") != 15
        or effect.get("bootstrap_ci_lower") != 1.3317595184568727
        or effect.get("bootstrap_ci_upper") != 3.3813212483296478
    ):
        raise StageDScopeError("sealed C10 effect evidence drifted")
    spatial = flags["spatial"]
    if (
        spatial.get("joint_spatial_status") != "PASS_JOINT_SPATIAL"
        or spatial.get("joint_pass_count") != 15
        or spatial.get("sample_count") != 15
    ):
        raise StageDScopeError("sealed C10 joint spatial evidence drifted")
    perturbations = spatial.get("perturbations", {})
    if (
        perturbations.get("translate_right_down_4", {}).get("status")
        != "FAIL_STABILITY"
        or perturbations.get("center_crop_216_resize_224", {}).get("status")
        != "FAIL_STABILITY"
        or perturbations.get("gaussian_rgb_std_0_01", {}).get("status")
        != "PASS_STABILITY"
    ):
        raise StageDScopeError("sealed C10 perturbation status drifted")
    validity = flags["validity"]
    if (
        validity.get("holdout_expected") != 15
        or validity.get("holdout_valid") != 15
        or validity.get("sealed_hash_and_provenance_valid") is not True
    ):
        raise StageDScopeError("sealed C10 validity evidence drifted")
    if holm.get("status") != "HOLM_BLOCKED_UNDERSPECIFIED" or holm.get("holm_performed") is not False:
        raise StageDScopeError("sealed C10 Holm blocker drifted")
    if (
        decision.get("downstream_authorized") is not False
        or decision.get("stage_d_started") is not False
        or decision.get("equivalence_test_performed") is not False
    ):
        raise StageDScopeError("sealed C10 downstream boundary drifted")

    repo = Path(repo_root).resolve()
    completion = _git_output(repo, "rev-parse", f"{C10_COMPLETION_COMMIT}^{{commit}}")
    if completion != C10_COMPLETION_COMMIT:
        raise StageDScopeError("C10 completion commit is unavailable")
    commit = _git_output(repo, "rev-parse", f"{C10_ARCHIVE_REF}^{{commit}}")
    tree = _git_output(repo, "rev-parse", f"{C10_ARCHIVE_REF}^{{tree}}")
    source_tree = _git_output(
        repo, "rev-parse", f"{C10_ARCHIVE_REF}:vitnns_rebuild/src/vitnns"
    )
    if (commit, tree, source_tree) != (C10_PRODUCER_COMMIT, C10_PRODUCER_TREE, C10_SOURCE_TREE):
        raise StageDScopeError("C10 archive producer ref drifted")
    code = provenance.get("code")
    if not isinstance(code, Mapping) or code.get("commit") != commit:
        raise StageDScopeError("C10 provenance producer commit drifted")
    if code.get("archive_ref") != "archive/c10-aggregation-producer":
        raise StageDScopeError("C10 provenance archive ref drifted")
    if code.get("tree") != tree or code.get("source_tree") != source_tree or code.get("dirty") is not False:
        raise StageDScopeError("C10 provenance Git attestation drifted")

    return {
        "status": "PASS",
        "root": str(sealed_root),
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "whole_root_inventory_digest": inventory_digest,
        "file_sha256": file_hashes,
        "canonical_digests": canonical_digests,
        "archive_ref": C10_ARCHIVE_REF,
        "producer_commit": commit,
        "producer_tree": tree,
        "source_tree": source_tree,
        "scientific_results_recomputed": False,
    }


def required_scope_document_fragments(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable facts that the human-readable scope document must expose."""

    validate_stage_d_scope_decision(document)
    fragments = [
        "config/stage_d_scope_decision.json",
        "APPROVED_TO_PREPARE_STAGE_D",
        "D0_MODEL_FREE_RERUN_20260825",
        "D0_INITIAL_ATTEMPT_20260825",
        "D0_NO_MODEL_OR_INFERENCE_GATE_VIOLATED",
        "D0_NO_JPEG_DECODE_GATE_VIOLATED",
        "execution_authorization=NOT_ISSUED",
        "analysis_authorization=NOT_EVALUATED",
        "NO_GO_NOVELTY",
        "BLOCKED",
        "HOLM_BLOCKED_UNDERSPECIFIED",
        "FAIL_SPATIAL",
        "downstream_authorized=false",
        "stage_d_started=false",
        "PASS_VALIDITY",
        "PASS_EFFECT",
        "(block=0, head=9)",
        "fine-4×4-cell",
        "NOT_APPLICABLE_NO_SECONDARY_INFERENCE",
        C10_ROOT_INVENTORY_DIGEST,
    ]
    fragments.extend(ALLOWED_CLAIM_IDS)
    fragments.extend(DENIED_CLAIM_IDS)
    fragments.extend(MANDATORY_DISCLOSURE_IDS)
    return tuple(fragments)


__all__ = [
    "ALLOWED_CLAIM_IDS",
    "C10_ARCHIVE_REF",
    "C10_FORMAL_ROOT",
    "C10_ROOT_INVENTORY_DIGEST",
    "D0_SCOPE_CANONICAL_DIGEST",
    "DEFAULT_SCOPE_DECISION",
    "DEFAULT_SCOPE_DOCUMENT",
    "DEFAULT_SCOPE_SCHEMA",
    "DENIED_CLAIM_IDS",
    "FUTURE_CONDITIONAL_CLAIM_IDS",
    "MANDATORY_DISCLOSURE_IDS",
    "StageDScopeError",
    "load_stage_d_scope_decision",
    "extract_scope_document_projection",
    "required_scope_document_fragments",
    "scope_document_projection",
    "scope_decision_digest",
    "strict_json_load",
    "validate_stage_d_scope_decision",
    "verify_c10_sealed_evidence",
]
