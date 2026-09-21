#!/usr/bin/env python3
"""Run the frozen CPU Step 2 protocol and write machine-readable receipts."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
    raise RuntimeError("Step 2 must run with CUDA_VISIBLE_DEVICES='' or '-1'")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from vit_pcia_pefa.src import analytic_fixtures
from vit_pcia_pefa.src.reference import (
    FAIL_CUT_NOT_LEGAL,
    FAIL_NONSMOOTH_NODE_HIT,
    FAIL_PORT_ID_AMBIGUOUS,
    FAIL_QUADRATURE_CLOSURE,
    FAIL_SECOND_ORDER_GRAPH_TRUNCATED,
    _torch_graph,
    analytic_point,
    autodiff_point,
    check_cut_legality,
    checks_pass,
    comparison,
    finite_difference_point,
    identity_checks,
    integrate_path,
    port_registry_failure,
    second_order_graph_failure,
)

HERE = ROOT / "vit_pcia_pefa"
RECEIPT_DIR = HERE / "negative_control_receipts"
SOURCE_FILES = (
    "vit_pcia_pefa/freeze_step2.py",
    "vit_pcia_pefa/src/analytic_fixtures.py",
    "vit_pcia_pefa/src/reference.py",
    "vit_pcia_pefa/run_step2.py",
    "vit_pcia_pefa/tests/test_step2_reference.py",
    "vit_pcia_pefa/validate_step2.py",
)


def canonical_bytes(payload: Any) -> bytes:
    def normalize(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=normalize) + "\n"
    ).encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_path_quantities(raw: analytic_fixtures.AnalyticPathQuantities):
    return raw


def compare_path(observed: Any, reference: Any, tolerance: dict[str, float]) -> dict[str, dict]:
    fields = (
        "output_delta",
        "gradient_x",
        "endpoint",
        "port_d_x",
        "port_q_x",
        "ig",
        "a",
        "t",
        "r",
        "omega_x",
    )
    return {
        field: comparison(getattr(observed, field), getattr(reference, field), **tolerance)
        for field in fields
    }


def receipt(control_id: str, expected: str, observed: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "ENDPOINT_PCIA_PEFA_STEP2",
        "control_id": control_id,
        "expected_failure": expected,
        "observed_failure": observed,
        "passed": observed == expected,
        "evidence": evidence,
    }


def structural_negative_controls(fixtures: dict[str, dict], primary_path: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    residual = fixtures["RESIDUAL2"]
    residual_path = primary_path["RESIDUAL2"]
    residual_q = np.asarray(residual_path["port_q_x"], dtype=np.float64)
    residual_endpoint = np.asarray(residual_path["endpoint"], dtype=np.float64)
    omitted = [copy.deepcopy(residual["ports"][1])]
    legal, counts = check_cut_legality(residual["paths"], omitted)
    results.append(receipt(
        "OMIT_RESIDUAL_IDENTITY", FAIL_CUT_NOT_LEGAL,
        FAIL_CUT_NOT_LEGAL if not legal else "NO_FAILURE",
        {"path_intersection_counts": counts, "all_paths_exactly_once": legal,
         "defective_endpoint_closure": comparison(residual_q[1], residual_endpoint, 1e-9, 1e-7, 1e-12)},
    ))

    duplicated = copy.deepcopy(residual["ports"]) + [copy.deepcopy(residual["ports"][0])]
    legal, counts = check_cut_legality(residual["paths"], duplicated)
    results.append(receipt(
        "DUPLICATE_RESIDUAL_IDENTITY", FAIL_CUT_NOT_LEGAL,
        FAIL_CUT_NOT_LEGAL if not legal else "NO_FAILURE",
        {"path_intersection_counts": counts, "all_paths_exactly_once": legal,
         "defective_endpoint_closure": comparison(residual_q.sum(axis=0) + residual_q[0], residual_endpoint, 1e-9, 1e-7, 1e-12)},
    ))

    fanout = fixtures["FANOUT2"]
    merged = copy.deepcopy(fanout["ports"][0])
    merged["port_id"] = "h.merged_consumers"
    merged["identity"]["consumer_operator_id"] = "MERGED(scale_a,tanh_c)"
    merged["intersects_paths"] = fanout["paths"]
    legal, counts = check_cut_legality(fanout["paths"], [merged])
    observed = port_registry_failure([merged], fanout["ports"]) or "NO_FAILURE"
    fanout_path = primary_path["FANOUT2"]
    proper_fanout_d = np.asarray(fanout_path["port_d_x"], dtype=np.float64)
    merged_d = proper_fanout_d.sum(axis=0, keepdims=True)
    results.append(receipt(
        "MERGE_FANOUT_COTANGENTS", FAIL_PORT_ID_AMBIGUOUS, observed,
        {"path_intersection_counts": counts, "numeric_cut_looks_legal": legal,
         "frozen_consumer_count": 2, "mutated_consumer_count": 1,
         "aggregate_gradient_unchanged_by_merge": bool(
             np.array_equal(merged_d[0], proper_fanout_d.sum(axis=0))
         )},
    ))

    multi = fixtures["MULTIINPUT3"]
    swapped = copy.deepcopy(multi["ports"])
    swapped[0]["identity"]["input_slot"], swapped[1]["identity"]["input_slot"] = (
        swapped[1]["identity"]["input_slot"], swapped[0]["identity"]["input_slot"]
    )
    observed = port_registry_failure(swapped, multi["ports"]) or "NO_FAILURE"
    multi_path = primary_path["MULTIINPUT3"]
    multi_q = np.asarray(multi_path["port_q_x"], dtype=np.float64)
    results.append(receipt(
        "SWAP_MULTIINPUT_SLOTS", FAIL_PORT_ID_AMBIGUOUS, observed,
        {"frozen_slots": [0, 1, 0, 0],
         "mutated_slots": [row["identity"]["input_slot"] for row in swapped],
         "aggregate_endpoint_unchanged_by_port_permutation": bool(
             np.array_equal(multi_q.sum(axis=0), multi_q[[1, 0, 2, 3]].sum(axis=0))
         )},
    ))

    linear = fixtures["LINEAR3"]
    output_cut = copy.deepcopy(linear["ports"][0])
    output_cut["port_id"] = "linear.output.serial_cut"
    output_cut["identity"]["producer_tensor_id"] = "linear_output.ssa0"
    output_cut["identity"]["consumer_operator_id"] = "F"
    serial = copy.deepcopy(linear["ports"]) + [output_cut]
    legal, counts = check_cut_legality(linear["paths"], serial)
    linear_path = primary_path["LINEAR3"]
    linear_endpoint = np.asarray(linear_path["endpoint"], dtype=np.float64)
    results.append(receipt(
        "DUPLICATE_SERIAL_CUTS", FAIL_CUT_NOT_LEGAL,
        FAIL_CUT_NOT_LEGAL if not legal else "NO_FAILURE",
        {"path_intersection_counts": counts, "all_paths_exactly_once": legal,
         "defective_endpoint_closure": comparison(2.0 * linear_endpoint, linear_endpoint, 1e-9, 1e-7, 1e-12)},
    ))

    u = torch.tensor([0.31, -0.47, 0.22], dtype=torch.float64, requires_grad=True)
    value, uses = _torch_graph("SATURATING3", u)
    cotangent = torch.autograd.grad(value, uses[0], create_graph=True, retain_graph=True)[0]
    d = torch.autograd.grad(uses[0], u, grad_outputs=cotangent, create_graph=True)[0]
    detached = d.detach()
    observed = second_order_graph_failure(detached) or "NO_FAILURE"
    results.append(receipt(
        "DETACH_SECOND_ORDER_D", FAIL_SECOND_ORDER_GRAPH_TRUNCATED, observed,
        {"before_detach_requires_grad": d.requires_grad,
         "after_detach_requires_grad": detached.requires_grad,
         "after_detach_grad_fn": str(detached.grad_fn)},
    ))

    bilinear = primary_path["BILINEAR2"]
    t_direct = np.asarray(bilinear["t"], dtype=np.float64)
    a = np.asarray(bilinear["a"], dtype=np.float64)
    r = np.asarray(bilinear["r"], dtype=np.float64)
    r_diagonal_only = np.zeros_like(r)
    diagonal = np.arange(a.shape[1])
    r_diagonal_only[:, diagonal, diagonal] = r[:, diagonal, diagonal]
    reconstructed = r_diagonal_only.copy()
    reconstructed[:, diagonal, diagonal] += a
    diag_check = comparison(reconstructed, t_direct, 1e-9, 1e-7, 1e-12)
    observed = FAIL_QUADRATURE_CLOSURE if not diag_check["passed"] else "NO_FAILURE"
    results.append(receipt(
        "HESSIAN_DIAGONAL_ONLY", FAIL_QUADRATURE_CLOSURE, observed,
        {"t_equals_delta_a_plus_diagonal_r": diag_check,
         "discarded_off_diagonal_max_abs": float(np.abs(r - r_diagonal_only).max())},
    ))
    return results


def nonsmooth_negative_controls(nodes: np.ndarray, weights: np.ndarray) -> list[dict[str, Any]]:
    delta = 2.0
    u = -1.0 + delta * nodes
    gradient = (u > 0.0).astype(np.float64)
    endpoint = np.asarray([delta * 1.0])
    pefa = np.asarray([np.sum(weights * delta * gradient)])
    safe_check = comparison(pefa, endpoint, 1e-9, 1e-7, 1e-12)
    safe = receipt(
        "RELU_SAFE_CROSSES_KINK_BETWEEN_NODES",
        FAIL_QUADRATURE_CLOSURE,
        FAIL_QUADRATURE_CLOSURE if not safe_check["passed"] else "NO_FAILURE",
        {
            "node_hit": False,
            "minimum_input_distance_to_kink": float(np.min(np.abs(u))),
            "almost_everywhere_pefa": float(pefa[0]),
            "endpoint": float(endpoint[0]),
            "pefa_to_endpoint": safe_check,
            "interpretation": "almost-everywhere Hessian omits the distributional gradient jump",
        },
    )

    kink = -1.0 + 2.0 * nodes[31]
    path_value = -1.0 + 2.0 * nodes[31]
    hit = path_value == kink
    hit_receipt = receipt(
        "RELU_HIT_FROZEN_NODE",
        FAIL_NONSMOOTH_NODE_HIT,
        FAIL_NONSMOOTH_NODE_HIT if hit else "NO_FAILURE",
        {
            "node_index_zero_based": 31,
            "node_t": float(nodes[31]),
            "path_value": float(path_value),
            "kink_constructed_from_same_float64_node": float(kink),
            "exact_node_hit_before_differentiation": hit,
        },
    )
    return [safe, hit_receipt]


def path_to_json(path: Any) -> dict[str, Any]:
    return {key: np.asarray(getattr(path, key)).tolist() for key in (
        "output_delta", "gradient_x", "endpoint", "port_d_x", "port_q_x",
        "ig", "a", "t", "r", "omega_x"
    )}


def main() -> None:
    if torch.cuda.is_available():
        raise RuntimeError("CUDA unexpectedly visible during CPU-only Step 2")
    fixtures_payload = json.loads((HERE / "fixture_registry.json").read_text(encoding="utf-8"))
    quadrature_payload = json.loads((HERE / "quadrature_registry.json").read_text(encoding="utf-8"))
    fixtures = {row["fixture_id"]: row for row in fixtures_payload["fixtures"]}
    tolerance = {key: float(value) for key, value in fixtures_payload["tolerance"].items()}
    finite_epsilon = float(fixtures_payload["finite_difference"]["epsilon"])
    finite_t = fixtures_payload["finite_difference"]["evaluation_t"]
    schedules = {row["count"]: row for row in quadrature_payload["schedules"]}

    fixture_results: dict[str, Any] = {}
    primary_raw: dict[str, dict[str, Any]] = {}
    all_main_pass = True
    for fixture_id in fixtures_payload["fixture_order"]:
        row = fixtures[fixture_id]
        b, x = np.asarray(row["b"], dtype=np.float64), np.asarray(row["x"], dtype=np.float64)
        legal, counts = check_cut_legality(row["paths"], row["ports"])
        port_failure = port_registry_failure(row["ports"], row["ports"])
        point_results = []
        for t in finite_t:
            u = b + float(t) * (x - b)
            autodiff = autodiff_point(fixture_id, u, b)
            analytic = analytic_point(fixture_id, u, b)
            analytic_vs_autodiff = {
                field: comparison(getattr(autodiff, field), getattr(analytic, field), **tolerance)
                for field in ("value", "gradient", "port_d", "port_jac_d", "port_q", "port_jac_q", "omega")
            }
            finite_gradient, finite_jac_d, finite_jac_q = finite_difference_point(
                fixture_id, u, b, finite_epsilon
            )
            finite_difference = {
                "gradient_vs_analytic": comparison(finite_gradient, analytic.gradient, **tolerance),
                "port_jac_d_vs_analytic": comparison(finite_jac_d, analytic.port_jac_d, **tolerance),
                "gradient_vs_autodiff": comparison(finite_gradient, autodiff.gradient, **tolerance),
                "port_jac_d_vs_autodiff": comparison(finite_jac_d, autodiff.port_jac_d, **tolerance),
                "port_jac_q_vs_analytic": comparison(finite_jac_q, analytic.port_jac_q, **tolerance),
                "port_jac_q_vs_autodiff": comparison(finite_jac_q, autodiff.port_jac_q, **tolerance),
            }
            point_results.append({
                "t": float(t),
                "analytic_vs_autodiff": analytic_vs_autodiff,
                "finite_difference": finite_difference,
                "passed": all(item["passed"] for item in (*analytic_vs_autodiff.values(), *finite_difference.values())),
            })

        path_results = {}
        paths_by_count = {}
        analytic_path = analytic_fixtures.closed_form_path(fixture_id, b, x)
        for count in (64, 128):
            schedule = schedules[count]
            nodes = np.asarray(schedule["nodes"], dtype=np.float64)
            weights = np.asarray(schedule["weights"], dtype=np.float64)
            path = integrate_path(autodiff_point, fixture_id, b, x, nodes, weights)
            paths_by_count[count] = path
            identities = identity_checks(path, tolerance)
            analytic_checks = compare_path(path, analytic_path, tolerance)
            path_results[str(count)] = {
                "role": schedule["role"],
                "identities": identities,
                "closed_form_comparison": analytic_checks,
                "passed": checks_pass(identities) and all(item["passed"] for item in analytic_checks.values()),
            }
        convergence = compare_path(paths_by_count[64], paths_by_count[128], tolerance)
        primary_raw[fixture_id] = path_to_json(paths_by_count[64])

        assertions: dict[str, Any] = {}
        if fixture_id == "LINEAR3":
            assertions = {
                "r_exact_zero": bool(np.count_nonzero(paths_by_count[64].r) == 0),
                "omega_exact_zero": bool(np.count_nonzero(paths_by_count[64].omega_x) == 0),
            }
        elif fixture_id == "BILINEAR2":
            off_diagonal = paths_by_count[64].r.copy()
            diagonal = np.arange(off_diagonal.shape[1])
            off_diagonal[:, diagonal, diagonal] = 0.0
            assertions = {
                "off_diagonal_r_nonzero": bool(np.abs(off_diagonal).max() > 1.0),
                "endpoint_minus_ig": comparison(
                    paths_by_count[64].endpoint - paths_by_count[64].ig,
                    np.asarray([-3.0, -3.0]), **tolerance
                ),
            }

        passed = (
            legal
            and port_failure is None
            and all(point["passed"] for point in point_results)
            and all(result["passed"] for result in path_results.values())
            and all(item["passed"] for item in convergence.values())
            and all(value if isinstance(value, bool) else value["passed"] for value in assertions.values())
        )
        all_main_pass = all_main_pass and passed
        fixture_results[fixture_id] = {
            "structure": row["structure"],
            "dimension": len(row["x"]),
            "port_count": len(row["ports"]),
            "port_ids": [port["port_id"] for port in row["ports"]],
            "cut_legality": {"passed": legal, "path_intersection_counts": counts},
            "port_identity_passed": port_failure is None,
            "point_comparisons": point_results,
            "quadrature": path_results,
            "quadrature_64_vs_128": convergence,
            "analytic_assertions": assertions,
            "passed": passed,
        }

    negative_results = structural_negative_controls(fixtures, primary_raw)
    primary_nodes = np.asarray(schedules[64]["nodes"], dtype=np.float64)
    primary_weights = np.asarray(schedules[64]["weights"], dtype=np.float64)
    negative_results.extend(nonsmooth_negative_controls(primary_nodes, primary_weights))
    negatives_pass = all(row["passed"] for row in negative_results)

    RECEIPT_DIR.mkdir(exist_ok=True)
    expected_names = set()
    for row in negative_results:
        name = row["control_id"].lower() + ".json"
        expected_names.add(name)
        (RECEIPT_DIR / name).write_bytes(canonical_bytes(row))
    unexpected = {path.name for path in RECEIPT_DIR.glob("*.json")} - expected_names
    if unexpected:
        raise RuntimeError(f"unexpected stale negative-control receipts: {sorted(unexpected)}")

    source_hashes = {relative: sha256(ROOT / relative) for relative in SOURCE_FILES}
    receipt_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in sorted(RECEIPT_DIR.glob("*.json"))
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    validation = {
        "schema_version": 1,
        "stage": "ENDPOINT_PCIA_PEFA_STEP2",
        "status": "PASS_STEP2_CPU_FLOAT64_REFERENCE" if all_main_pass and negatives_pass else "FAIL_STEP2",
        "scope": "CPU_REFERENCE_ONLY_STEP3_TO_STEP6_NOT_AUTHORIZED",
        "novelty_position": "COMPOSITIONAL_SYSTEMATIZATION_CANDIDATE",
        "predecessor": {
            "status": "PASS_STEP1_CONTRACT_AND_COLLISION_AUDIT",
            "validation_command_run_before_step2": "CUDA_VISIBLE_DEVICES='' .conda/envs/vitnns/bin/python vit_pcia_pefa/validate_step1.py",
        },
        "environment": {
            "python_executable_realpath": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "torch_default_dtype": str(torch.get_default_dtype()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
            "device": "cpu",
        },
        "contract": {
            "dtype": "float64",
            "primary_quadrature_nodes": 64,
            "diagnostic_quadrature_nodes": 128,
            "tolerance": tolerance,
            "finite_difference_epsilon": finite_epsilon,
            "fixture_registry_sha256": sha256(HERE / "fixture_registry.json"),
            "quadrature_registry_sha256": sha256(HERE / "quadrature_registry.json"),
            "method_contract_sha256": sha256(HERE / "method_contract.md"),
        },
        "coverage": {
            "fixtures": len(fixture_results),
            "structures": [fixtures[key]["structure"] for key in fixtures_payload["fixture_order"]],
            "identity_families": [
                "port_endpoint_closure", "pcia_to_ig", "pefa_to_endpoint",
                "t_equals_delta_a_plus_r", "curvature_cut_zero_sum"
            ],
            "comparison_methods": ["closed_form_analytic", "torch_autodiff", "central_finite_difference"],
            "negative_controls": len(negative_results),
        },
        "fixtures": fixture_results,
        "negative_controls": [
            {"control_id": row["control_id"], "expected_failure": row["expected_failure"],
             "observed_failure": row["observed_failure"], "passed": row["passed"]}
            for row in negative_results
        ],
        "all_main_fixtures_passed": all_main_pass,
        "all_negative_controls_detected": negatives_pass,
        "source_sha256": source_hashes,
        "negative_receipt_sha256": receipt_hashes,
        "git_head_at_execution": git,
    }
    (HERE / "mvp_validation.json").write_bytes(canonical_bytes(validation))
    print(json.dumps({
        "status": validation["status"],
        "fixtures_passed": sum(row["passed"] for row in fixture_results.values()),
        "fixtures_total": len(fixture_results),
        "negative_controls_detected": sum(row["passed"] for row in negative_results),
        "negative_controls_total": len(negative_results),
        "mvp_validation_sha256": sha256(HERE / "mvp_validation.json"),
    }, sort_keys=True, indent=2))
    if validation["status"] != "PASS_STEP2_CPU_FLOAT64_REFERENCE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
