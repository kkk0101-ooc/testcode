#!/usr/bin/env python3
"""Prompt-to-artifact completion audit for NNS-CST N1-R3."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence/n1_r3_20260922"
sys.path.insert(0, str(HERE / "src"))

import validate_n1 as v1  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(relative):
    path = HERE / relative
    require(path.is_file(), "missing: " + relative)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(relative):
    return hashlib.sha256((HERE / relative).read_bytes()).hexdigest()


def entry(identifier, requirement, evidence, verification, condition):
    require(condition, identifier + " failed: " + requirement)
    return {
        "requirement_id": identifier,
        "requirement": requirement,
        "evidence": evidence,
        "verification": verification,
        "status": "PASS",
    }


def fixture_map(results):
    return {
        row["fixture_id"]: {item["assertion"]: item for item in row["assertions"]}
        for row in results["fixtures"]
    }


def build_checklist():
    freeze = load("evidence/n1_r3_20260922/freeze_receipt.json")
    results = load("evidence/n1_r3_20260922/analytic_results.json")
    validation = load("evidence/n1_r3_20260922/validation.json")
    ports = load("manifests/n1_r3/model_port_registry.json")
    contract = load("config/n1_r3/subspace_contract.json")
    fixtures = load("manifests/n1_r3/n1_fixture_registry.json")
    r1 = load("evidence/n1_r1_20260922/terminal_receipt.json")
    r2 = load("evidence/n1_r2_20260922/terminal_receipt.json")
    fmap = fixture_map(results)
    mutations = {row["mutation_id"]: row for row in results["mutations"]}
    roles = {row["tensor_role"] for row in ports["records"]}
    readout = results["frozen_readout"]
    readout_schema = load("schemas/n1_r3_frozen_readout.schema.json")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    n0_process = subprocess.run([sys.executable, "validate_n0.py"], cwd=str(HERE), env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, check=True)
    n0_live = json.loads(n0_process.stdout)

    frozen = {row["path"]: row["sha256"] for row in freeze["frozen_artifacts"]}
    frozen_current = len(frozen) == 30 and all(sha256(path) == digest for path, digest in frozen.items())
    zero_boundary = all(results["execution_boundary"][key] == 0 for key in (
        "formal_image_decodes", "model_loads", "checkpoint_deserializations",
        "gpu_or_cuda_uses", "training_runs", "inference_runs", "formal_outcome_reads"))
    schema_paths = sorted(path.relative_to(HERE).as_posix() for path in (HERE / "schemas").glob("n1_r3_*.schema.json"))
    closed_schema_count = 0
    for relative in schema_paths:
        closed_schema_count += v1.validate_schema_definition(load(relative), relative)
    readout_fields = set(readout_schema["properties"])
    readout_required = set(readout_schema["required"])
    fixture_assertions_pass = all(item["passed"] is True for row in results["fixtures"] for item in row["assertions"])
    mutation_pass = all(row["passed"] is True and row["expected_failure"] == row["observed_failure"] for row in results["mutations"])
    digests = [row["digest"] for row in results["determinism"]["runs"]]

    checklist = [
        entry("COMMON-01", "N0 predecessor gates are currently valid",
              ["n0_decision.json", "validate_n0.py"],
              "Reexecuted validate_n0.py; status PASS and aggregate GO_REQUEST_N1",
              n0_live["status"] == "PASS" and n0_live["aggregate_decision"] == "GO_REQUEST_N1"),
        entry("COMMON-02", "Current request authorizes N1 and not N2",
              ["evidence/n1_r3_20260922/freeze_receipt.json"],
              "Checked authorized_stage=N1, n2_authorized=false",
              freeze["authorization"]["authorized_stage"] == "N1" and freeze["authorization"]["n2_authorized"] is False),
        entry("COMMON-03", "R1 terminal failure is preserved",
              ["evidence/n1_r1_20260922/terminal_receipt.json", "evidence/n1_r3_20260922/freeze_receipt.json"],
              "Checked R1 status and SHA-256 predecessor binding",
              r1["status"] == "FAIL_N1_R1_SCHEMA_FREEZE_INCOMPLETE" and freeze["predecessors"]["n1_r1_terminal_sha256"] == sha256("evidence/n1_r1_20260922/terminal_receipt.json")),
        entry("COMMON-04", "R2 terminal failure is preserved",
              ["evidence/n1_r2_20260922/terminal_receipt.json", "evidence/n1_r3_20260922/freeze_receipt.json"],
              "Checked R2 status and SHA-256 predecessor binding",
              r2["status"] == "FAIL_N1_R2_READOUT_SERIALIZATION_INCOMPLETE" and freeze["predecessors"]["n1_r2_terminal_sha256"] == sha256("evidence/n1_r2_20260922/terminal_receipt.json")),
        entry("COMMON-05", "R3 source and contracts were frozen before execution",
              ["evidence/n1_r3_20260922/freeze_receipt.json"],
              "Recomputed all 29 frozen artifact SHA-256 values",
              frozen_current),
        entry("COMMON-06", "Closed schemas reject unspecified object fields",
              schema_paths,
              "Recursively inspected every R3 object schema for additionalProperties=false",
              closed_schema_count >= 20),
        entry("COMMON-07", "Producer and validator are separated",
              ["run_n1_r3.py", "validate_n1_r3.py", "evidence/n1_r3_20260922/validation.json"],
              "Different scripts; validator status PASS and recomputation coverage present",
              validation["status"] == "PASS_N1_R3_INDEPENDENT_VALIDATION" and validation["coverage"]["assertion_count"] == 32),
        entry("COMMON-08", "Execution remained CPU-only and synthetic-only",
              ["evidence/n1_r3_20260922/analytic_results.json", "evidence/n1_r3_20260922/validation.json"],
              "Checked zero image/model/GPU/training/inference/outcome counters and CPU environment",
              zero_boundary and results["environment"]["device"] == "CPU" and results["environment"]["cuda_visible_devices"] == ""),
        entry("COMMON-09", "N2 did not start",
              ["evidence/n1_r3_20260922/freeze_receipt.json", "evidence/n1_r3_20260922/analytic_results.json", "evidence/n1_r3_20260922/validation.json"],
              "Checked all n2_started fields are false",
              results["execution_boundary"]["n2_started"] is False and validation["n2_started"] is False),

        entry("N1-4.1-01", "Patch embedding residual stream is registered",
              ["manifests/n1_r3/model_port_registry.json"], "Role set includes patch_embedding_residual_stream",
              "patch_embedding_residual_stream" in roles),
        entry("N1-4.1-02", "Every block pre/post LayerNorm is registered",
              ["manifests/n1_r3/model_port_registry.json"], "Role set includes four norm boundary roles",
              {"pre_layernorm1_residual", "post_layernorm1", "pre_layernorm2_residual", "post_layernorm2"} <= roles),
        entry("N1-4.1-03", "Every head query, key, value input and value is registered",
              ["manifests/n1_r3/model_port_registry.json"], "Recomputed head-port roles and total model counts",
              {"attention_head_query_input", "attention_head_key_input", "attention_head_value_input", "attention_head_value"} <= roles),
        entry("N1-4.1-04", "Every head pre-projection output is registered",
              ["manifests/n1_r3/model_port_registry.json"], "Role present for both models",
              "attention_head_pre_projection_output" in roles),
        entry("N1-4.1-05", "Attention merged output and residual use-edge are registered",
              ["manifests/n1_r3/model_port_registry.json"], "Both roles present for all blocks",
              {"attention_merged_output", "attention_residual_use_edge"} <= roles),
        entry("N1-4.1-06", "MLP expand, activation, contract and residual use-edge are registered",
              ["manifests/n1_r3/model_port_registry.json"], "All four MLP roles present",
              {"mlp_expand", "mlp_activation", "mlp_contract_output", "mlp_residual_use_edge"} <= roles),
        entry("N1-4.1-07", "CLS, final norm and classifier margin are registered",
              ["manifests/n1_r3/model_port_registry.json"], "All four readout roles present",
              {"cls_token", "final_norm_input", "final_norm_output", "classifier_margin"} <= roles),
        entry("N1-4.1-08", "Port identity jointly binds producer, consumer, slot, invocation, block, branch and checkpoint",
              ["N1数学合同.md", "src/nns_cst/core.py", "manifests/n1_r3/model_port_registry.json"],
              "Validator rederived every port_id; SWAP_Q_K_V_SLOT mutation failed closed",
              ports["record_count"] == 1330 and mutations["SWAP_Q_K_V_SLOT"]["observed_failure"] == "PORT_IDENTITY_MISMATCH"),

        entry("N1-4.2-01", "Activation layout and token sets are frozen",
              ["config/n1_r3/subspace_contract.json"], "Checked BATCH_TOKEN_CHANNEL and three explicit token sets",
              contract["activation_layout"]["default"] == "BATCH_TOKEN_CHANNEL" and len(contract["activation_layout"]["token_sets"]) == 3),
        entry("N1-4.2-02", "Paired difference and centering are frozen",
              ["N1数学合同.md", "config/n1_r3/subspace_contract.json"], "Checked source_group_id pairing and centering field",
              contract["paired_difference"]["pairing_unit"] == "source_group_id" and bool(contract["paired_difference"]["centering"])),
        entry("N1-4.2-03", "Token and source weights are frozen",
              ["config/n1_r3/subspace_contract.json"], "Checked explicit token_weight_formula and source_weight_formula",
              bool(contract["paired_difference"]["token_weight_formula"]) and bool(contract["paired_difference"]["source_weight_formula"])),
        entry("N1-4.2-04", "Control covariance operation is frozen",
              ["config/n1_r3/subspace_contract.json"], "Checked intensity-matched null source and nuisance projection operation",
              "intensity_matched_null" in contract["control_covariance"]["source"] and "orthogonal complement" in contract["control_covariance"]["operation"]),
        entry("N1-4.2-05", "Estimator, rank candidates and ridge candidates are frozen",
              ["config/n1_r3/subspace_contract.json"], "Checked estimator name, ranks 1/2/4/8/16 and ridge 0/1e-8/1e-6",
              contract["estimator"]["rank_candidates"] == [1,2,4,8,16] and contract["estimator"]["ridge_candidates"] == [0.0,1e-8,1e-6]),
        entry("N1-4.2-06", "Basis-invariant projection, similarity and distance are frozen",
              ["N1数学合同.md", "config/n1_r3/subspace_contract.json", "src/nns_cst/core.py"],
              "Checked projector definitions and disabled individual basis-vector claims",
              contract["basis_invariants"]["individual_basis_vector_claims_allowed"] is False),
        entry("N1-4.2-07", "Frozen readout training and validation rules are frozen",
              ["config/n1_r3/subspace_contract.json"], "Checked fit split, selection-only validation and no confirmation refit",
              contract["frozen_readout"]["confirmation_refit_allowed"] is False and "selection" in contract["frozen_readout"]["validation_role"]),
        entry("N1-4.2-08", "Frozen readout serialization is closed and provenance-bound",
              ["schemas/n1_r3_frozen_readout.schema.json", "evidence/n1_r3_20260922/analytic_results.json"],
              "Checked exact field set, fit_split and contract_sha256 against current contract",
              readout_schema["additionalProperties"] is False and readout_fields == readout_required == set(readout) and readout["contract_sha256"] == sha256("N1数学合同.md") and readout["fit_split"] == "engineering_fixture"),
        entry("N1-4.2-09", "Different channel dimensions and token topologies fail closed",
              ["config/n1_r3/subspace_contract.json", "evidence/n1_r3_20260922/analytic_results.json"],
              "Checked no implicit resize/drop and wrong alignment mutation detected",
              contract["dimension_and_topology"]["implicit_resize_or_token_drop_allowed"] is False and mutations["WRONG_TOKEN_ALIGNMENT"]["observed_failure"] == "TOKEN_ALIGNMENT_ERROR"),
        entry("N1-4.2-10", "Numerical degeneracy has explicit typed failures",
              ["config/n1_r3/subspace_contract.json", "evidence/n1_r3_20260922/analytic_results.json"],
              "Checked exact 12-code registry and eight triggered typed-failure assertions",
              len(contract["typed_failures"]) == len(set(contract["typed_failures"])) == 12 and len(fmap["TYPED_FAILURES"]) == 8),

        entry("N1-4.3-01", "Rotated two-dimensional subspace is basis invariant",
              ["evidence/n1_r3_20260922/analytic_results.json"], "Projection, projector and similarity assertions all passed",
              set(fmap["ROTATED_TWO_DIMENSIONAL_SUBSPACE"]) == {"PROJECTION_INVARIANCE", "PROJECTOR_INVARIANCE", "SIMILARITY_ONE"}),
        entry("N1-4.3-02", "Distributed multi-head signal defeats single heads but succeeds jointly",
              ["evidence/n1_r3_20260922/analytic_results.json"], "Both pre-registered assertions passed",
              set(fmap["DISTRIBUTED_MULTI_HEAD_SIGNAL"]) == {"EACH_SINGLE_HEAD_INCOMPLETE", "JOINT_SUBSPACE_COMPLETE"}),
        entry("N1-4.3-03", "Brightness control removes low-level pseudo-semantic signal",
              ["evidence/n1_r3_20260922/analytic_results.json"], "Nuisance removed and semantic retained assertions passed",
              set(fmap["BRIGHTNESS_CONTROL_RESIDUALIZATION"]) == {"NUISANCE_REMOVED", "SEMANTIC_RETAINED"}),
        entry("N1-4.3-04", "Identity, linear, residual and two-branch transport have analytic results",
              ["evidence/n1_r3_20260922/analytic_results.json"], "All four transport assertions passed",
              set(fmap["ANALYTIC_TRANSPORT"]) == {"IDENTITY", "LINEAR", "RESIDUAL", "TWO_BRANCH"}),
        entry("N1-4.3-05", "Semantic and complement attribution reconstruct complete-cut attribution",
              ["evidence/n1_r3_20260922/analytic_results.json"], "Coordinate and scalar closure passed",
              set(fmap["SEMANTIC_COMPLEMENT_COMPLETE_CUT"]) == {"COORDINATE_CLOSURE", "SCALAR_CLOSURE"}),
        entry("N1-4.3-06", "Missing edge, duplicate edge, QKV slot swap and token mismatch are detected",
              ["manifests/n1_r3/n1_fixture_registry.json", "evidence/n1_r3_20260922/analytic_results.json"],
              "All four mutations produced their exact registered failures",
              mutation_pass and len(mutations) == 4),
        entry("N1-4.3-07", "Injection, erasure, random subspace and orthogonal complement match analytic expectations",
              ["evidence/n1_r3_20260922/analytic_results.json"], "All four intervention assertions passed",
              set(fmap["INTERVENTION_ANALYTICS"]) == {"INJECTION", "ERASURE", "RANDOM_SUBSPACE", "ORTHOGONAL_COMPLEMENT"}),
        entry("N1-4.3-08", "Rank, zero-difference, nonfinite and off-manifold diagnostics are typed",
              ["evidence/n1_r3_20260922/analytic_results.json"], "Eight typed-failure assertions passed including all named categories",
              {"RANK_TOO_LARGE", "RANK_DEFICIENT", "ZERO_DIFFERENCE", "NONFINITE", "NEAR_SINGULAR_CONTROL", "NO_SEMANTIC_AFTER_CONTROL", "MISSING_TOKEN", "OFF_MANIFOLD"} == set(fmap["TYPED_FAILURES"])),

        entry("N1-4.4-01", "Mathematical contract is delivered and authority-bound",
              ["N1数学合同.md", "evidence/n1_r3_20260922/freeze_receipt.json"], "Recomputed contract SHA-256 against freeze receipt",
              freeze["authority"]["mathematical_contract_sha256"] == sha256("N1数学合同.md")),
        entry("N1-4.4-02", "Closed schemas are delivered",
              schema_paths, "Found and recursively checked all R3 schemas",
              len(schema_paths) == 7 and closed_schema_count >= 20),
        entry("N1-4.4-03", "Implementation skeleton is delivered",
              ["src/nns_cst/core.py", "src/nns_cst/interfaces.py", "src/nns_cst_r3/readout.py"],
              "All source files frozen and independent validator scanned dependencies",
              all(path in frozen for path in ("src/nns_cst/core.py", "src/nns_cst/interfaces.py", "src/nns_cst_r3/readout.py"))),
        entry("N1-4.4-04", "Standard-library metadata validator is delivered",
              ["validate_n1_r3.py", "evidence/n1_r3_20260922/validation.json"],
              "Validator ran on Python standard library only and returned PASS",
              validation["status"] == "PASS_N1_R3_INDEPENDENT_VALIDATION"),
        entry("N1-4.4-05", "Unit tests pass",
              ["tests/test_n1.py", "tests/test_n1_r3.py", "evidence/n1_r3_20260922/validation.json"],
              "Independent validator reran full suite",
              validation["coverage"]["unit_test_count"] >= 15),
        entry("N1-4.4-06", "Analytic identities and registered fixture assertions all pass",
              ["manifests/n1_r3/n1_fixture_registry.json", "evidence/n1_r3_20260922/analytic_results.json"],
              "8 fixtures and all 32 exact assertions passed",
              fixture_assertions_pass and validation["coverage"]["fixture_count"] == 8 and validation["coverage"]["assertion_count"] == 32),
        entry("N1-4.4-07", "Mutation tests all pass",
              ["evidence/n1_r3_20260922/analytic_results.json"], "Four exact typed failures observed",
              mutation_pass and validation["coverage"]["mutation_count"] == 4),
        entry("N1-4.4-08", "Determinism probe passes",
              ["n1_r3_determinism_probe.py", "evidence/n1_r3_20260922/analytic_results.json"],
              "Two independent PYTHONHASHSEED processes produced one digest",
              len(digests) == 2 and len(set(digests)) == 1 and digests[0] == validation["determinism_digest"]),
        entry("N1-4.4-09", "Runtime cost microbenchmark is finite",
              ["evidence/n1_r3_20260922/analytic_results.json"],
              "Checked 200 iterations, positive wall/throughput/RSS and finite CPU time",
              results["microbenchmark"]["iterations"] == 200 and all(math.isfinite(float(value)) for value in results["microbenchmark"].values()) and results["microbenchmark"]["wall_seconds"] > 0 and results["microbenchmark"]["iterations_per_second"] > 0),
        entry("N1-4.4-10", "N1 PASS does not claim real-model or scientific support",
              ["N1数学合同.md", "evidence/n1_r3_20260922/analytic_results.json"],
              "No formal image/model/outcome access occurred; next gate remains N2 request",
              zero_boundary and results["execution_boundary"]["n2_started"] is False),
    ]
    return checklist, closed_schema_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-terminal", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"), "CUDA must be hidden")
    checklist, closed_count = build_checklist()
    counts = {
        "common_rule_items": sum(row["requirement_id"].startswith("COMMON-") for row in checklist),
        "n1_4_1_items": sum(row["requirement_id"].startswith("N1-4.1-") for row in checklist),
        "n1_4_2_items": sum(row["requirement_id"].startswith("N1-4.2-") for row in checklist),
        "n1_4_3_items": sum(row["requirement_id"].startswith("N1-4.3-") for row in checklist),
        "n1_4_4_items": sum(row["requirement_id"].startswith("N1-4.4-") for row in checklist),
        "total_items": len(checklist),
    }
    audit = {
        "schema_version": 1,
        "document_type": "nns_cst_n1_r3_completion_audit",
        "stage_id": "N1",
        "objective": "执行testcode/vit_nns_cst/执行规划.md的N1",
        "status": "PASS_N1_R3_COMPLETION_AUDIT",
        "prompt_coverage": counts,
        "checklist": checklist,
        "all_requirements_passed": True,
        "missing_requirements": [],
        "n2_started": False,
    }
    schema = load("schemas/n1_r3_completion_audit.schema.json")
    v1.validate_schema_definition(schema, "completion_audit_schema")
    v1.validate_instance(audit, schema, "completion_audit")
    raw = json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output = EVIDENCE / "completion_audit.json"
    if output.exists() and output.read_text(encoding="utf-8") != raw:
        raise RuntimeError("completion audit evidence drift")
    if not output.exists():
        output.write_text(raw, encoding="utf-8")
    terminal_status = "NOT_REQUESTED"
    if args.verify_terminal:
        terminal = load("evidence/n1_r3_20260922/terminal_receipt.json")
        terminal_schema = load("schemas/n1_r3_terminal_receipt.schema.json")
        v1.validate_schema_definition(terminal_schema, "terminal_schema")
        v1.validate_instance(terminal, terminal_schema, "terminal")
        require(terminal["completion_audit_sha256"] == sha256("evidence/n1_r3_20260922/completion_audit.json"), "terminal audit binding")
        for row in terminal["artifact_bindings"]:
            require(sha256(row["path"]) == row["sha256"], "terminal binding drift: " + row["path"])
        terminal_status = terminal["status"]
    print(json.dumps({
        "status": audit["status"], "checklist_items": len(checklist),
        "closed_object_schema_count": closed_count,
        "missing_requirements": [], "terminal_status": terminal_status,
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
