#!/usr/bin/env python3
"""Independent standard-library validator for fresh NNS-CST N1-R3."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence/n1_r3_20260922"
sys.path.insert(0, str(HERE / "src"))

import validate_n1 as v1  # noqa: E402
from nns_cst.core import (  # noqa: E402
    PORT_IDENTITY_FIELDS, estimate_subspace, matvec, norm, project,
    projector_distance, projector_similarity, readout_score,
    semantic_complement_attribution, validate_port_registry,
)
from nns_cst_r3.readout import fit_frozen_readout  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(relative):
    path = HERE / relative
    require(path.is_file(), "missing artifact: " + relative)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "non-object artifact: " + relative)
    return value


def sha256(relative):
    return hashlib.sha256((HERE / relative).read_bytes()).hexdigest()


def validate_schema(relative_schema, value, where):
    schema = load(relative_schema)
    object_count = v1.validate_schema_definition(schema, relative_schema)
    require(object_count >= 1, "no object schema: " + relative_schema)
    v1.validate_instance(value, schema, where)
    return object_count


def validate_freeze():
    receipt = load("evidence/n1_r3_20260922/freeze_receipt.json")
    validate_schema("schemas/n1_r3_freeze_receipt.schema.json", receipt, "freeze_receipt")
    require(receipt["predecessors"]["n1_r1_terminal_sha256"] == sha256("evidence/n1_r1_20260922/terminal_receipt.json"), "R1 binding")
    require(receipt["predecessors"]["n1_r2_terminal_sha256"] == sha256("evidence/n1_r2_20260922/terminal_receipt.json"), "R2 binding")
    require(receipt["authority"]["execution_plan_sha256"] == sha256("执行规划.md"), "plan drift")
    require(receipt["authority"]["research_goal_sha256"] == sha256("研究目标.md"), "goal drift")
    require(receipt["authority"]["mathematical_contract_sha256"] == sha256("N1数学合同.md"), "contract drift")
    require(receipt["authority"]["recovery_amendment_sha256"] == sha256("N1-R3恢复修订.md"), "amendment drift")
    frozen = {row["path"]: row["sha256"] for row in receipt["frozen_artifacts"]}
    require(len(frozen) == len(receipt["frozen_artifacts"]) == 30, "frozen artifact count/duplicates")
    for relative, expected in frozen.items():
        require(sha256(relative) == expected, "frozen artifact drift: " + relative)
    return receipt, frozen


def validate_registries():
    ports = load("manifests/n1_r3/model_port_registry.json")
    contract = load("config/n1_r3/subspace_contract.json")
    fixtures = load("manifests/n1_r3/n1_fixture_registry.json")
    schema_count = 0
    schema_count += validate_schema("schemas/n1_port_registry.schema.json", ports, "ports")
    schema_count += validate_schema("schemas/n1_subspace_contract.schema.json", contract, "contract")
    schema_count += validate_schema("schemas/n1_r3_fixture_registry.schema.json", fixtures, "fixtures")
    require(ports["identity_fields"] == list(PORT_IDENTITY_FIELDS), "port identity field drift")
    validate_port_registry(ports["records"], [row["port_id"] for row in ports["records"]])
    require(ports["record_count"] == len(ports["records"]) == 1330, "port count")
    expected_counts = {"vit_b16": 845, "deit_s16": 485}
    required_roles = {
        "patch_embedding_residual_stream", "pre_layernorm1_residual", "post_layernorm1",
        "attention_head_query_input", "attention_head_key_input", "attention_head_value_input",
        "attention_head_value", "attention_head_pre_projection_output", "attention_merged_output",
        "attention_residual_use_edge", "pre_layernorm2_residual", "post_layernorm2",
        "mlp_expand", "mlp_activation", "mlp_contract_output", "mlp_residual_use_edge",
        "cls_token", "final_norm_input", "final_norm_output", "classifier_margin",
    }
    for model_id, count in expected_counts.items():
        rows = [row for row in ports["records"] if row["model_id"] == model_id]
        require(len(rows) == count, "model port count: " + model_id)
        require({row["tensor_role"] for row in rows} == required_roles, "role coverage: " + model_id)
    required_failures = {
        "RANK_EXCEEDS_DIMENSION", "RANK_DEFICIENT", "ZERO_PAIR_DIFFERENCE",
        "NO_SEMANTIC_AFTER_CONTROL", "NEAR_SINGULAR_CONTROL_COVARIANCE",
        "NONFINITE_VALUE", "MISSING_TOKEN", "TOKEN_ALIGNMENT_ERROR",
        "PORT_IDENTITY_MISMATCH", "DUPLICATE_PORT", "CUT_MISSING_EDGE",
        "OFF_MANIFOLD_INTERVENTION",
    }
    require(set(contract["typed_failures"]) == required_failures, "typed failure coverage")
    require(contract["estimator"]["rank_candidates"] == [1, 2, 4, 8, 16], "rank candidates")
    require(contract["estimator"]["ridge_candidates"] == [0.0, 1e-8, 1e-6], "ridge candidates")
    require(len(fixtures["fixtures"]) == 8 and len(fixtures["mutations"]) == 4, "fixture/mutation count")
    return ports, contract, fixtures, schema_count, len(required_roles), len(required_failures)


def validate_results(fixtures):
    results = load("evidence/n1_r3_20260922/analytic_results.json")
    schema_count = validate_schema("schemas/n1_r3_analytic_results.schema.json", results, "results")
    schema_count += validate_schema("schemas/n1_r3_frozen_readout.schema.json", results["frozen_readout"], "readout")
    expected = {row["fixture_id"]: set(row["required_assertions"]) for row in fixtures["fixtures"]}
    actual = {}
    for row in results["fixtures"]:
        require(all(item["passed"] is True and math.isfinite(float(item["max_abs_error"]))
                    for item in row["assertions"]), "failed/nonfinite assertion")
        actual[row["fixture_id"]] = {item["assertion"] for item in row["assertions"]}
    require(actual == expected, "fixture assertion coverage")
    assertion_count = sum(len(value) for value in actual.values())
    require(assertion_count == 32, "assertion count")
    expected_mutations = {row["mutation_id"]: row["expected_failure"] for row in fixtures["mutations"]}
    actual_mutations = {row["mutation_id"]: row["observed_failure"] for row in results["mutations"]}
    require(actual_mutations == expected_mutations and all(row["passed"] is True for row in results["mutations"]), "mutation coverage")

    # Independent mathematical recomputation from the frozen raw fixture definitions.
    root = math.sqrt(0.5)
    reference = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    rotated = [[root, root, 0.0], [-root, root, 0.0]]
    require(projector_distance(reference, rotated) <= 1e-9, "basis distance")
    require(abs(projector_similarity(reference, rotated) - 1.0) <= 1e-9, "basis similarity")
    controlled = estimate_subspace([[10, 2, 0], [9, 2, 0], [11, 2, 0]], 1,
                                   control_differences=[[1, 0, 0]])
    require(norm(project([1, 0, 0], controlled["basis"])) <= 1e-9, "nuisance removal")
    attribution = semantic_complement_attribution([2, -3, 4], [0.5, 2, -1], reference)
    require(max(abs(a-b) for a, b in zip(attribution["reconstructed"], attribution["complete_cut"])) <= 1e-9, "complete cut")
    require(matvec([[0.5, -0.25], [0.25, 0.5]], [2, -1]) == [1.25, 0.0], "transport")

    contract_sha = sha256("N1数学合同.md")
    recomputed = fit_frozen_readout(
        [[0, 2, 0], [1, 3, 0]], [[0, -2, 0], [-1, -3, 0]], [[0, 1, 0]],
        "engineering_fixture", contract_sha)
    require(recomputed == results["frozen_readout"], "readout independent recomputation")
    require(readout_score(recomputed, [0, 4, 0]) > 0 and readout_score(recomputed, [0, -4, 0]) < 0, "readout score direction")

    digests = []
    for seed in ("0", "99173"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        environment["CUDA_VISIBLE_DEVICES"] = ""
        process = subprocess.run([sys.executable, str(HERE / "n1_r3_determinism_probe.py")],
                                 cwd=str(HERE), env=environment, check=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        digests.append(json.loads(process.stdout)["digest"])
    require(len(set(digests)) == 1 and digests[0] == results["scientific_payload_digest"], "determinism")
    benchmark = results["microbenchmark"]
    require(benchmark["iterations"] == 200 and benchmark["wall_seconds"] > 0
            and benchmark["iterations_per_second"] > 0 and benchmark["peak_rss_kib"] > 0, "microbenchmark")
    require(results["environment"]["device"] == "CPU" and results["environment"]["cuda_visible_devices"] == "", "CPU boundary")
    require(results["environment"]["third_party_runtime_dependencies"] == [], "third party dependencies")
    require(all(results["execution_boundary"][key] == 0 for key in (
        "formal_image_decodes", "model_loads", "checkpoint_deserializations", "gpu_or_cuda_uses",
        "training_runs", "inference_runs", "formal_outcome_reads")), "execution boundary")
    require(results["execution_boundary"]["n2_started"] is False, "N2 started")
    return results, assertion_count, schema_count, digests[0]


def validate_sources_and_tests():
    sources = [
        "freeze_n1.py", "run_n1.py", "n1_determinism_probe.py", "validate_n1.py",
        "freeze_n1_r3.py", "run_n1_r3.py", "n1_r3_determinism_probe.py",
        "validate_n1_r3.py", "finalize_n1_r3.py", "audit_n1_r3.py",
        "src/nns_cst/core.py", "src/nns_cst/interfaces.py",
        "src/nns_cst_r3/__init__.py", "src/nns_cst_r3/readout.py",
        "tests/test_n1.py", "tests/test_n1_r3.py",
    ]
    forbidden = {"torch", "numpy", "PIL", "cv2", "timm", "tensorflow", "jax"}
    imports = set()
    for relative in sources:
        tree = ast.parse((HERE / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    require(not (imports & forbidden), "forbidden imports: %r" % (imports & forbidden))
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_n1*.py", "-v"],
                             cwd=str(HERE), env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)
    require(process.returncode == 0, "unit tests failed:\n" + process.stdout + process.stderr)
    match = re.search(r"Ran (\d+) tests", process.stdout + process.stderr)
    require(match is not None and int(match.group(1)) >= 15, "unit test count")
    return int(match.group(1)), len(sources)


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"), "CUDA must be hidden")
    receipt, frozen = validate_freeze()
    ports, contract, fixtures, registry_schema_count, role_count, failure_count = validate_registries()
    results, assertion_count, result_schema_count, digest = validate_results(fixtures)
    test_count, source_count = validate_sources_and_tests()
    extra_schemas = [
        "schemas/n1_r3_freeze_receipt.schema.json",
        "schemas/n1_r3_validation.schema.json",
        "schemas/n1_r3_completion_audit.schema.json",
        "schemas/n1_r3_terminal_receipt.schema.json",
    ]
    extra_count = sum(v1.validate_schema_definition(load(path), path) for path in extra_schemas)
    summary = {
        "schema_version": 1,
        "document_type": "nns_cst_n1_r3_independent_validation",
        "stage_id": "N1",
        "status": "PASS_N1_R3_INDEPENDENT_VALIDATION",
        "artifact_hashes": {
            "freeze_receipt_sha256": sha256("evidence/n1_r3_20260922/freeze_receipt.json"),
            "analytic_results_sha256": sha256("evidence/n1_r3_20260922/analytic_results.json"),
        },
        "coverage": {
            "frozen_artifact_count": len(frozen), "port_count": ports["record_count"],
            "tensor_role_count": role_count, "typed_failure_count": failure_count,
            "fixture_count": len(fixtures["fixtures"]), "assertion_count": assertion_count,
            "mutation_count": len(fixtures["mutations"]), "readout_schema_valid": True,
            "closed_object_schema_count": registry_schema_count + result_schema_count + extra_count + 5,
            "unit_test_count": test_count,
        },
        "determinism_digest": digest,
        "n2_started": False,
    }
    validate_schema("schemas/n1_r3_validation.schema.json", summary, "validation")
    raw = json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = EVIDENCE / "validation.json"
    if path.exists() and path.read_text(encoding="utf-8") != raw:
        raise RuntimeError("validation evidence drift")
    if not path.exists():
        path.write_text(raw, encoding="utf-8")
    print(raw, end="")


if __name__ == "__main__":
    main()
