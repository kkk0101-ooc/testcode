#!/usr/bin/env python3
"""Fail-closed artifact and coverage audit for Endpoint-PCIA-PEFA Step 2."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "vit_pcia_pefa"
EXPECTED_CONTRACT_SHA256 = "9bd1f8718e37d44b17d6017a7c13c6d9fdbd6f8dc3da98121001afbf7b5ef601"
EXPECTED_FIXTURE_REGISTRY_SHA256 = "8555d13e4047163f9f20d46616b58b8f254c7fa0990dc6e25a459ffac2852459"
EXPECTED_QUADRATURE_REGISTRY_SHA256 = "b26418d81d45ed8f17383f78d30b71288b4bac90adc34d3f6e374f38492ba268"
EXPECTED_FIXTURES = (
    "LINEAR3", "BILINEAR2", "SATURATING3",
    "FANOUT2", "RESIDUAL2", "MULTIINPUT3",
)
EXPECTED_STRUCTURES = (
    "linear", "bilinear", "smooth_saturating_nonlinearity",
    "fan_out", "residual", "multi_input_operator",
)
EXPECTED_IDENTITIES = {
    "port_endpoint_closure", "pcia_to_ig", "pefa_to_endpoint",
    "t_equals_delta_a_plus_r", "curvature_cut_zero_sum",
}
EXPECTED_CONTROLS = {
    "OMIT_RESIDUAL_IDENTITY": "FAIL_CUT_NOT_LEGAL",
    "DUPLICATE_RESIDUAL_IDENTITY": "FAIL_CUT_NOT_LEGAL",
    "MERGE_FANOUT_COTANGENTS": "FAIL_PORT_ID_AMBIGUOUS",
    "SWAP_MULTIINPUT_SLOTS": "FAIL_PORT_ID_AMBIGUOUS",
    "DUPLICATE_SERIAL_CUTS": "FAIL_CUT_NOT_LEGAL",
    "DETACH_SECOND_ORDER_D": "FAIL_SECOND_ORDER_GRAPH_TRUNCATED",
    "HESSIAN_DIAGONAL_ONLY": "FAIL_QUADRATURE_CLOSURE",
    "RELU_SAFE_CROSSES_KINK_BETWEEN_NODES": "FAIL_QUADRATURE_CLOSURE",
    "RELU_HIT_FROZEN_NODE": "FAIL_NONSMOOTH_NODE_HIT",
}
PATH_FIELDS = {
    "output_delta", "gradient_x", "endpoint", "port_d_x", "port_q_x",
    "ig", "a", "t", "r", "omega_x",
}
IDENTITY_FIELDS = {
    "port_gradient_closure", "port_endpoint_closure", "pcia_to_ig",
    "pcia_to_output_delta", "pefa_per_port_to_q", "pefa_to_endpoint",
    "t_equals_delta_a_plus_r", "curvature_cut_zero_sum",
    "r_to_endpoint_minus_ig",
}
POINT_FIELDS = {"value", "gradient", "port_d", "port_jac_d", "port_q", "port_jac_q", "omega"}
FD_FIELDS = {
    "gradient_vs_analytic", "port_jac_d_vs_analytic",
    "gradient_vs_autodiff", "port_jac_d_vs_autodiff",
    "port_jac_q_vs_analytic", "port_jac_q_vs_autodiff",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_metric(metric: dict[str, Any], label: str) -> None:
    require(metric.get("passed") is True, f"failed metric: {label}")
    require(metric.get("finite") is True, f"nonfinite metric: {label}")
    require(metric.get("coordinates_checked", 0) >= 1, f"empty metric: {label}")
    require(np.isfinite(metric.get("max_abs")), f"invalid max_abs: {label}")
    require(np.isfinite(metric.get("relative_l2")), f"invalid relative_l2: {label}")


def main() -> None:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"), "CPU verifier requires CUDA hidden")
    fixture_registry = load(HERE / "fixture_registry.json")
    quadrature = load(HERE / "quadrature_registry.json")
    validation = load(HERE / "mvp_validation.json")
    require(sha256(HERE / "method_contract.md") == EXPECTED_CONTRACT_SHA256, "frozen method contract drift")
    require(sha256(HERE / "fixture_registry.json") == EXPECTED_FIXTURE_REGISTRY_SHA256, "frozen fixture registry drift")
    require(sha256(HERE / "quadrature_registry.json") == EXPECTED_QUADRATURE_REGISTRY_SHA256, "frozen quadrature registry drift")

    require(fixture_registry["status"] == "FROZEN_BEFORE_OUTCOME", "fixture registry not frozen")
    require(tuple(fixture_registry["fixture_order"]) == EXPECTED_FIXTURES, "fixture set/order drift")
    require(tuple(row["structure"] for row in fixture_registry["fixtures"]) == EXPECTED_STRUCTURES, "structure coverage drift")
    require(fixture_registry["dtype"] == "float64", "fixture dtype drift")
    require(fixture_registry["tolerance"] == {"atol": 1e-9, "rtol": 1e-7, "near_zero": 1e-12}, "tolerance drift")
    require(fixture_registry["finite_difference"] == {"epsilon": 1e-6, "evaluation_t": [0.0, 0.37, 1.0]}, "finite difference drift")
    for row in fixture_registry["fixtures"]:
        require(row["b"] != row["x"], f"degenerate path: {row['fixture_id']}")
        require(row["ports"], f"no ports: {row['fixture_id']}")
        for port in row["ports"]:
            require(set(port["identity"]) == {
                "model_id", "graph_revision", "block_index", "producer_tensor_id",
                "consumer_operator_id", "input_slot", "invocation_index", "slice_spec",
            }, f"port identity schema drift: {row['fixture_id']}")

    require(quadrature["status"] == "FROZEN_BEFORE_OUTCOME", "quadrature not frozen")
    require([row["count"] for row in quadrature["schedules"]] == [64, 128], "quadrature counts drift")
    for row in quadrature["schedules"]:
        count = row["count"]
        source_nodes, source_weights = np.polynomial.legendre.leggauss(count)
        source_nodes = ((source_nodes + 1.0) / 2.0).astype("<f8")
        source_weights = (source_weights / 2.0).astype("<f8")
        require(np.array_equal(source_nodes, np.asarray(row["nodes"])), f"node bytes drift: {count}")
        require(np.array_equal(source_weights, np.asarray(row["weights"])), f"weight bytes drift: {count}")
        digest = hashlib.sha256(source_nodes.tobytes() + source_weights.tobytes()).hexdigest()
        require(digest == row["nodes_then_weights_float64_sha256"], f"quadrature digest drift: {count}")

    require(validation["status"] == "PASS_STEP2_CPU_FLOAT64_REFERENCE", "Step 2 is not PASS")
    require(validation["scope"] == "CPU_REFERENCE_ONLY_STEP3_TO_STEP6_NOT_AUTHORIZED", "scope gate drift")
    require(validation["all_main_fixtures_passed"] is True, "main fixtures not all passed")
    require(validation["all_negative_controls_detected"] is True, "negative controls not all detected")
    require(validation["environment"]["device"] == "cpu", "non-CPU result")
    require(validation["environment"]["torch_cuda_available"] is False, "CUDA was visible")
    require(validation["environment"]["torch_default_dtype"] == "torch.float64", "Torch dtype drift")
    require(validation["contract"]["fixture_registry_sha256"] == sha256(HERE / "fixture_registry.json"), "fixture binding drift")
    require(validation["contract"]["quadrature_registry_sha256"] == sha256(HERE / "quadrature_registry.json"), "quadrature binding drift")
    require(validation["contract"]["method_contract_sha256"] == sha256(HERE / "method_contract.md"), "contract binding drift")
    require(set(validation["coverage"]["identity_families"]) == EXPECTED_IDENTITIES, "five identity coverage drift")
    require(validation["coverage"]["comparison_methods"] == ["closed_form_analytic", "torch_autodiff", "central_finite_difference"], "three-way comparison missing")

    require(set(validation["fixtures"]) == set(EXPECTED_FIXTURES), "validation fixture coverage drift")
    for fixture_id in EXPECTED_FIXTURES:
        row = validation["fixtures"][fixture_id]
        require(row["passed"] is True, f"fixture failed: {fixture_id}")
        require(row["cut_legality"]["passed"] is True, f"cut illegal: {fixture_id}")
        require(all(value == 1 for value in row["cut_legality"]["path_intersection_counts"].values()), f"path count not one: {fixture_id}")
        require(row["port_identity_passed"] is True, f"port identity failed: {fixture_id}")
        require(len(row["point_comparisons"]) == 3, f"point grid incomplete: {fixture_id}")
        for point in row["point_comparisons"]:
            require(set(point["analytic_vs_autodiff"]) == POINT_FIELDS, f"point fields incomplete: {fixture_id}")
            require(set(point["finite_difference"]) == FD_FIELDS, f"FD fields incomplete: {fixture_id}")
            for name, metric in {**point["analytic_vs_autodiff"], **point["finite_difference"]}.items():
                check_metric(metric, f"{fixture_id}/point/{name}")
        require(set(row["quadrature"]) == {"64", "128"}, f"quadrature coverage incomplete: {fixture_id}")
        for count, result in row["quadrature"].items():
            require(result["passed"] is True, f"quadrature failed: {fixture_id}/{count}")
            require(set(result["identities"]) == IDENTITY_FIELDS, f"identity coverage incomplete: {fixture_id}/{count}")
            require(set(result["closed_form_comparison"]) == PATH_FIELDS, f"closed form coverage incomplete: {fixture_id}/{count}")
            for name, metric in {**result["identities"], **result["closed_form_comparison"]}.items():
                check_metric(metric, f"{fixture_id}/{count}/{name}")
        require(set(row["quadrature_64_vs_128"]) == PATH_FIELDS, f"convergence fields incomplete: {fixture_id}")
        for name, metric in row["quadrature_64_vs_128"].items():
            check_metric(metric, f"{fixture_id}/convergence/{name}")

    summary_controls = {row["control_id"]: row for row in validation["negative_controls"]}
    require(set(summary_controls) == set(EXPECTED_CONTROLS), "negative-control summary coverage drift")
    receipt_paths = sorted((HERE / "negative_control_receipts").glob("*.json"))
    require(len(receipt_paths) == len(EXPECTED_CONTROLS), "negative-control receipt count drift")
    seen = set()
    for path in receipt_paths:
        row = load(path)
        control_id = row["control_id"]
        require(control_id in EXPECTED_CONTROLS, f"unknown negative control: {control_id}")
        require(control_id not in seen, f"duplicate negative control: {control_id}")
        seen.add(control_id)
        require(row["expected_failure"] == EXPECTED_CONTROLS[control_id], f"wrong expected failure: {control_id}")
        require(row["observed_failure"] == EXPECTED_CONTROLS[control_id], f"wrong observed failure: {control_id}")
        require(row["passed"] is True, f"negative control escaped: {control_id}")
        require(summary_controls[control_id]["passed"] is True, f"summary negative failed: {control_id}")
        relative = str(path.relative_to(ROOT))
        require(validation["negative_receipt_sha256"][relative] == sha256(path), f"receipt binding drift: {control_id}")
    require(seen == set(EXPECTED_CONTROLS), "missing negative receipt")

    for relative, expected in validation["source_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"source drift after execution: {relative}")
    source = (HERE / "src/reference.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {"hessian", "jacobian", "jacrev", "jacfwd"}
    used = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    require(not (used & forbidden_calls), f"explicit full derivative API used: {used & forbidden_calls}")

    decision = (HERE / "step2_decision.md").read_text(encoding="utf-8")
    for token in (
        "STEP2_STATUS=PASS_CPU_FLOAT64_REFERENCE",
        "STEP3_REQUIRES_EXPLICIT_USER_REQUEST",
        "STEP3_TO_STEP6_NOT_STARTED=true",
        "RELU_SAFE", "FAIL_QUADRATURE_CLOSURE",
        "RELU_HIT", "FAIL_NONSMOOTH_NODE_HIT",
    ):
        require(token in decision, f"decision token missing: {token}")
    audit_block = decision.split("## Prompt-to-artifact 审计", 1)[1].split("## 复现与验证", 1)[0]
    audit_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in audit_block.splitlines()
        if line.startswith("| ") and not line.startswith("|---")
    ]
    require(len(audit_rows) >= 40, f"prompt audit row count too small: {len(audit_rows) - 1}")
    require(all(len(row) == 3 for row in audit_rows), "prompt audit is not three columns")
    require(all(row[2] in {"状态", "PASS"} for row in audit_rows), "prompt audit contains non-PASS row")

    print(json.dumps({
        "status": "PASS_STEP2_ARTIFACT_AND_COVERAGE_AUDIT",
        "fixtures_verified": len(EXPECTED_FIXTURES),
        "identity_families_verified": len(EXPECTED_IDENTITIES),
        "negative_controls_verified": len(EXPECTED_CONTROLS),
        "point_comparisons_verified": len(EXPECTED_FIXTURES) * 3,
        "quadrature_results_verified": len(EXPECTED_FIXTURES) * 2,
        "prompt_audit_requirements_verified": len(audit_rows) - 1,
        "mvp_validation_sha256": sha256(HERE / "mvp_validation.json"),
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
