#!/usr/bin/env python3
"""Independent standard-library validator for NNS-CST N1-R2."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence/n1_r2_20260922"
sys.path.insert(0, str(HERE / "src"))

from nns_cst.core import (  # noqa: E402
    CSTError, PORT_IDENTITY_FIELDS, add, estimate_subspace, matvec, norm,
    project, projector_distance, projector_similarity,
    semantic_complement_attribution, validate_port_registry,
)


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_keys(value: Dict[str, Any], expected: Iterable[str], where: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    require(actual == expected_set,
            "%s keys differ missing=%r extra=%r" %
            (where, sorted(expected_set - actual), sorted(actual - expected_set)))


def load(path: Path) -> Dict[str, Any]:
    require(path.is_file(), "missing artifact: " + str(path.relative_to(HERE)))
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "root is not object: " + str(path))
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_schema_definition(schema: Any, where: str = "root") -> int:
    """Require every object-typed instance schema to reject unknown keys."""
    if not isinstance(schema, dict):
        return 0
    count = 0
    if schema.get("type") == "object":
        require(schema.get("additionalProperties") is False,
                where + " object schema is open")
        require(isinstance(schema.get("properties"), dict),
                where + " missing properties")
        require(isinstance(schema.get("required"), list),
                where + " missing required")
        require(set(schema["required"]) <= set(schema["properties"]),
                where + " required field not declared")
        count += 1
    for key in ("properties",):
        if isinstance(schema.get(key), dict):
            for name, child in schema[key].items():
                count += validate_schema_definition(child, where + "." + name)
    if isinstance(schema.get("items"), dict):
        count += validate_schema_definition(schema["items"], where + "[]")
    return count


def validate_instance(value: Any, schema: Dict[str, Any], where: str = "root") -> None:
    if "const" in schema:
        require(value == schema["const"], where + " const mismatch")
    if "enum" in schema:
        require(value in schema["enum"], where + " enum mismatch")
    expected_type = schema.get("type")
    if expected_type == "object":
        require(isinstance(value, dict), where + " expected object")
        properties = schema["properties"]
        require(set(schema["required"]) <= set(value), where + " missing required property")
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(properties), where + " has additional property")
        for name, item in value.items():
            if name in properties:
                validate_instance(item, properties[name], where + "." + name)
    elif expected_type == "array":
        require(isinstance(value, list), where + " expected array")
        if "minItems" in schema:
            require(len(value) >= schema["minItems"], where + " too short")
        if "maxItems" in schema:
            require(len(value) <= schema["maxItems"], where + " too long")
        if schema.get("uniqueItems"):
            serialized = [json.dumps(item, sort_keys=True) for item in value]
            require(len(serialized) == len(set(serialized)), where + " items not unique")
        if "items" in schema:
            for index, item in enumerate(value):
                validate_instance(item, schema["items"], "%s[%d]" % (where, index))
    elif expected_type == "string":
        require(isinstance(value, str), where + " expected string")
        if "minLength" in schema:
            require(len(value) >= schema["minLength"], where + " too short")
        if "pattern" in schema:
            require(re.search(schema["pattern"], value) is not None, where + " pattern mismatch")
    elif expected_type == "integer":
        require(isinstance(value, int) and not isinstance(value, bool), where + " expected integer")
    elif expected_type == "number":
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
                where + " expected finite number")
    elif expected_type == "boolean":
        require(isinstance(value, bool), where + " expected boolean")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema:
            require(value >= schema["minimum"], where + " below minimum")
        if "exclusiveMinimum" in schema:
            require(value > schema["exclusiveMinimum"], where + " below exclusive minimum")


def validate_schemas() -> Dict[str, int]:
    targets = {
        "schemas/n1_port_registry.schema.json": "manifests/n1_r2/model_port_registry.json",
        "schemas/n1_subspace_contract.schema.json": "config/n1_r2/subspace_contract.json",
        "schemas/n1_fixture_registry.schema.json": "manifests/n1_r2/n1_fixture_registry.json",
        "schemas/n1_analytic_results.schema.json": "evidence/n1_r2_20260922/analytic_results.json",
    }
    counts = {}
    for schema_path, instance_path in targets.items():
        schema = load(HERE / schema_path)
        count = validate_schema_definition(schema, schema_path)
        require(count >= 1, schema_path + " has no object schema")
        validate_instance(load(HERE / instance_path), schema, instance_path)
        counts[schema_path] = count
    terminal_schema = load(HERE / "schemas/n1_terminal_receipt.schema.json")
    counts["schemas/n1_terminal_receipt.schema.json"] = validate_schema_definition(
        terminal_schema, "schemas/n1_terminal_receipt.schema.json")
    return counts


def validate_freeze() -> Dict[str, Any]:
    receipt = load(EVIDENCE / "freeze_receipt.json")
    require_keys(receipt, [
        "schema_version", "document_type", "stage_id", "status", "authorization",
        "predecessor", "authority", "frozen_artifacts", "repository", "execution_boundary",
    ], "freeze receipt")
    require(receipt["status"] == "FROZEN_BEFORE_N1_TEST_EXECUTION", "wrong freeze status")
    require(receipt["authorization"]["authorized_stage"] == "N1", "N1 not authorized")
    require(receipt["authorization"]["n2_authorized"] is False, "N2 boundary open")
    require(receipt["predecessor"]["n0_decision"] == "GO_REQUEST_N1", "wrong N0 decision")
    require(receipt["predecessor"]["n1_r1_status"] == "FAIL_N1_R1_SCHEMA_FREEZE_INCOMPLETE", "R1 not preserved")
    require(receipt["predecessor"]["n1_r1_terminal_sha256"] == sha256(HERE / "evidence/n1_r1_20260922/terminal_receipt.json"), "R1 binding drift")
    require(receipt["authority"]["execution_plan_sha256"] == sha256(HERE / "执行规划.md"), "plan drift")
    require(receipt["authority"]["research_goal_sha256"] == sha256(HERE / "研究目标.md"), "goal drift")
    require(receipt["authority"]["mathematical_contract_sha256"] == sha256(HERE / "N1数学合同.md"), "contract drift")
    frozen = {row["path"]: row["sha256"] for row in receipt["frozen_artifacts"]}
    require(len(frozen) == len(receipt["frozen_artifacts"]) == 16, "frozen artifact count/uniqueness")
    for relative, expected in frozen.items():
        require(sha256(HERE / relative) == expected, "frozen artifact drift: " + relative)
    boundary = receipt["execution_boundary"]
    require(all(boundary[key] == 0 for key in (
        "formal_image_decodes", "model_loads", "checkpoint_deserializations",
        "gpu_or_cuda_uses", "training_runs", "inference_runs", "formal_outcome_reads")),
        "freeze execution boundary violated")
    require(boundary["n2_started"] is False, "N2 started")
    return {"frozen_artifact_count": len(frozen), "status": receipt["status"]}


def validate_ports() -> Dict[str, Any]:
    registry = load(HERE / "manifests/n1_r2/model_port_registry.json")
    require(registry["identity_fields"] == list(PORT_IDENTITY_FIELDS), "identity field drift")
    records = registry["records"]
    expected_ids = [row["port_id"] for row in records]
    validate_port_registry(records, expected_ids)
    require(registry["record_count"] == len(records) == 1330, "port count mismatch")
    models = {row["model_id"]: row for row in registry["models"]}
    require(set(models) == {"vit_b16", "deit_s16"}, "model registry mismatch")
    expected_counts = {"vit_b16": 845, "deit_s16": 485}
    roles = {}
    for model_id, expected in expected_counts.items():
        model_records = [row for row in records if row["model_id"] == model_id]
        require(len(model_records) == expected, "port coverage count: " + model_id)
        roles[model_id] = sorted(set(row["tensor_role"] for row in model_records))
    required_roles = {
        "patch_embedding_residual_stream", "pre_layernorm1_residual", "post_layernorm1",
        "attention_head_query_input", "attention_head_key_input", "attention_head_value_input",
        "attention_head_value", "attention_head_pre_projection_output", "attention_merged_output",
        "attention_residual_use_edge", "pre_layernorm2_residual", "post_layernorm2",
        "mlp_expand", "mlp_activation", "mlp_contract_output", "mlp_residual_use_edge",
        "cls_token", "final_norm_input", "final_norm_output", "classifier_margin",
    }
    for model_id, found in roles.items():
        require(set(found) == required_roles, "tensor role coverage: " + model_id)
    return {"model_counts": expected_counts, "record_count": len(records), "tensor_role_count": len(required_roles)}


def validate_contract_and_registry() -> Dict[str, Any]:
    contract = load(HERE / "config/n1_r2/subspace_contract.json")
    require(contract["activation_layout"]["default"] == "BATCH_TOKEN_CHANNEL", "layout drift")
    require(contract["paired_difference"]["pairing_unit"] == "source_group_id", "pairing drift")
    require(contract["estimator"]["rank_candidates"] == [1, 2, 4, 8, 16], "rank candidates drift")
    require(contract["estimator"]["ridge_candidates"] == [0.0, 1e-8, 1e-6], "ridge drift")
    require(contract["basis_invariants"]["individual_basis_vector_claims_allowed"] is False,
            "basis-vector claim boundary open")
    require(contract["frozen_readout"]["confirmation_refit_allowed"] is False, "readout refit open")
    require(contract["dimension_and_topology"]["implicit_resize_or_token_drop_allowed"] is False,
            "topology boundary open")
    required_failures = {
        "RANK_EXCEEDS_DIMENSION", "RANK_DEFICIENT", "ZERO_PAIR_DIFFERENCE",
        "NO_SEMANTIC_AFTER_CONTROL", "NEAR_SINGULAR_CONTROL_COVARIANCE",
        "NONFINITE_VALUE", "MISSING_TOKEN", "TOKEN_ALIGNMENT_ERROR",
        "PORT_IDENTITY_MISMATCH", "DUPLICATE_PORT", "CUT_MISSING_EDGE",
        "OFF_MANIFOLD_INTERVENTION",
    }
    require(set(contract["typed_failures"]) == required_failures, "typed failure set drift")

    fixtures = load(HERE / "manifests/n1_r2/n1_fixture_registry.json")
    require(len(fixtures["fixtures"]) == 7, "fixture count")
    require(len(fixtures["mutations"]) == 4, "mutation count")
    require(fixtures["microbenchmark"]["iterations"] == 200, "benchmark iteration drift")
    require(fixtures["determinism_probe"]["pythonhashseed_values"] == ["0", "99173"], "seed drift")
    return {"typed_failure_count": len(required_failures), "fixture_count": 7, "mutation_count": 4}


def validate_results() -> Dict[str, Any]:
    registry = load(HERE / "manifests/n1_r2/n1_fixture_registry.json")
    results = load(EVIDENCE / "analytic_results.json")
    require(results["status"] == "PASS_N1_ANALYTIC_FIXTURES", "analytic status")
    require(results["environment"]["device"] == "CPU", "non-CPU result")
    require(results["environment"]["cuda_visible_devices"] == "", "CUDA not hidden")
    require(results["environment"]["third_party_runtime_dependencies"] == [], "third party runtime")
    expected = {row["fixture_id"]: set(row["required_assertions"]) for row in registry["fixtures"]}
    actual = {}
    for row in results["fixtures"]:
        require(all(item["passed"] is True and math.isfinite(item["max_abs_error"])
                    for item in row["assertions"]), "failed/nonfinite assertion: " + row["fixture_id"])
        actual[row["fixture_id"]] = {item["assertion"] for item in row["assertions"]}
    require(actual == expected, "fixture/assertion coverage mismatch")
    mutations_expected = {row["mutation_id"]: row["expected_failure"] for row in registry["mutations"]}
    mutations_actual = {row["mutation_id"]: row["observed_failure"] for row in results["mutations"]}
    require(mutations_actual == mutations_expected, "mutation coverage/failure mismatch")
    require(all(row["passed"] is True for row in results["mutations"]), "mutation escaped")

    # Independently recompute central algebra rather than trusting producer PASS.
    root = math.sqrt(0.5)
    first = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    rotated = [[root, root, 0.0], [-root, root, 0.0]]
    require(projector_distance(first, rotated) <= 1e-9, "recomputed basis distance")
    require(abs(projector_similarity(first, rotated) - 1.0) <= 1e-9, "recomputed similarity")
    control = estimate_subspace([[10, 2, 0], [9, 2, 0], [11, 2, 0]], 1,
                                control_differences=[[1, 0, 0]])
    require(norm(project([1, 0, 0], control["basis"])) <= 1e-9, "recomputed nuisance removal")
    attribution = semantic_complement_attribution([2, -3, 4], [0.5, 2, -1], first)
    require(max(abs(a - b) for a, b in zip(attribution["reconstructed"], attribution["complete_cut"])) <= 1e-9,
            "recomputed coordinate closure")
    branch = [[0.5, -0.25], [0.25, 0.5]]
    require(matvec(branch, [2, -1]) == [1.25, 0.0], "recomputed transport")

    probe_runs = []
    for seed in ("0", "99173"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        environment["CUDA_VISIBLE_DEVICES"] = ""
        process = subprocess.run(
            [sys.executable, str(HERE / "n1_determinism_probe.py")], cwd=str(HERE), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        probe_runs.append(json.loads(process.stdout)["digest"])
    require(len(set(probe_runs)) == 1, "independent determinism mismatch")
    require(probe_runs[0] == results["scientific_payload_digest"], "stored determinism digest mismatch")
    benchmark = results["microbenchmark"]
    require(benchmark["iterations"] == 200 and benchmark["wall_seconds"] > 0.0
            and benchmark["cpu_seconds"] >= 0.0 and benchmark["iterations_per_second"] > 0.0
            and benchmark["peak_rss_kib"] > 0, "invalid benchmark")
    boundary = results["execution_boundary"]
    require(all(boundary[key] == 0 for key in (
        "formal_image_decodes", "model_loads", "checkpoint_deserializations",
        "gpu_or_cuda_uses", "training_runs", "inference_runs", "formal_outcome_reads")),
        "execution boundary violated")
    require(boundary["n2_started"] is False, "N2 started")
    return {"fixture_count": len(actual), "assertion_count": sum(len(value) for value in actual.values()),
            "mutation_count": len(mutations_actual), "determinism_digest": probe_runs[0]}


def validate_source_boundary() -> Dict[str, Any]:
    files = [
        HERE / "freeze_n1.py", HERE / "run_n1.py", HERE / "n1_determinism_probe.py",
        HERE / "src/nns_cst/core.py", HERE / "src/nns_cst/interfaces.py",
        HERE / "validate_n1.py", HERE / "tests/test_n1.py",
    ]
    forbidden_imports = {"torch", "numpy", "PIL", "cv2", "timm", "tensorflow", "jax"}
    observed_imports = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                observed_imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                observed_imports.add(node.module.split(".")[0])
    require(not (observed_imports & forbidden_imports), "forbidden runtime dependency: %r" % (observed_imports & forbidden_imports))
    return {"source_files_scanned": len(files), "forbidden_imports_found": []}


def validate_tests() -> Dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_n1.py", "-v"],
        cwd=str(HERE), env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    require(process.returncode == 0, "unit tests failed:\n" + process.stdout + process.stderr)
    match = re.search(r"Ran (\d+) tests", process.stderr + process.stdout)
    require(match is not None and int(match.group(1)) >= 11, "unit test coverage count")
    return {"tests_run": int(match.group(1)), "status": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"), "CUDA_VISIBLE_DEVICES must be empty or -1")
    summary = {
        "schema_version": 1,
        "document_type": "nns_cst_n1_independent_validation",
        "stage_id": "N1",
        "status": "PASS_N1_INDEPENDENT_VALIDATION",
        "freeze": validate_freeze(),
        "schemas": validate_schemas(),
        "ports": validate_ports(),
        "contract": validate_contract_and_registry(),
        "results": validate_results(),
        "source_boundary": validate_source_boundary(),
        "tests": validate_tests(),
        "n2_started": False,
    }
    raw = json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw, encoding="utf-8")
    print(raw, end="")


if __name__ == "__main__":
    main()
