#!/usr/bin/env python3
"""Materialize the Step 2 registries fixed by method_contract.md.

This script performs no model evaluation and reads no outcomes. Existing registry
files must match byte-for-byte after canonical JSON serialization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def port(
    fixture_id: str,
    short_id: str,
    producer: str,
    consumer: str,
    slot: int,
    paths: list[str],
) -> dict[str, Any]:
    return {
        "port_id": short_id,
        "identity": {
            "model_id": fixture_id,
            "graph_revision": "step2_frozen_v1",
            "block_index": -1,
            "producer_tensor_id": producer,
            "consumer_operator_id": consumer,
            "input_slot": slot,
            "invocation_index": 0,
            "slice_spec": "ALL",
        },
        "intersects_paths": paths,
    }


def fixture_registry() -> dict[str, Any]:
    fixtures = [
        {
            "fixture_id": "LINEAR3",
            "structure": "linear",
            "b": [0.0, 0.0, 0.0],
            "x": [1.25, -0.5, 2.0],
            "expression": "F=1.5*u0-2*u1+0.25*u2",
            "parameters": {"coefficient": [1.5, -2.0, 0.25]},
            "paths": ["u0-linear-F", "u1-linear-F", "u2-linear-F"],
            "ports": [
                port("LINEAR3", "linear.input", "u.ssa0", "linear", 0,
                     ["u0-linear-F", "u1-linear-F", "u2-linear-F"])
            ],
            "analytic_assertions": ["R_ZERO", "OMEGA_ZERO"],
        },
        {
            "fixture_id": "BILINEAR2",
            "structure": "bilinear",
            "b": [0.0, 0.0],
            "x": [2.0, -3.0],
            "expression": "F=u0*u1+0.2*u0",
            "parameters": {"linear_coefficient": 0.2},
            "paths": ["u0-mul0-F", "u1-mul1-F", "u0-linear-F"],
            "ports": [
                port("BILINEAR2", "mul.slot0", "u0.ssa0", "mul", 0, ["u0-mul0-F"]),
                port("BILINEAR2", "mul.slot1", "u1.ssa0", "mul", 1, ["u1-mul1-F"]),
                port("BILINEAR2", "linear.slot0", "u0.ssa0", "scale_add", 0, ["u0-linear-F"]),
            ],
            "analytic_assertions": ["OFF_DIAGONAL_R_NONZERO", "ENDPOINT_MINUS_IG=[-3,-3]"],
        },
        {
            "fixture_id": "SATURATING3",
            "structure": "smooth_saturating_nonlinearity",
            "b": [0.1, -0.2, 0.3],
            "x": [1.2, 0.7, -0.4],
            "expression": "F=tanh(0.7*u0-1.1*u1+0.5*u2+0.2)",
            "parameters": {"coefficient": [0.7, -1.1, 0.5], "bias": 0.2},
            "paths": ["u0-s-tanh-F", "u1-s-tanh-F", "u2-s-tanh-F"],
            "ports": [
                port("SATURATING3", "preactivation.tanh.slot0", "s.ssa0", "tanh", 0,
                     ["u0-s-tanh-F", "u1-s-tanh-F", "u2-s-tanh-F"])
            ],
            "analytic_assertions": ["SMOOTH_SECOND_ORDER", "QUADRATURE_CONVERGENCE"],
        },
        {
            "fixture_id": "FANOUT2",
            "structure": "fan_out",
            "b": [0.0, 0.0],
            "x": [1.1, -0.8],
            "expression": "h=0.6*u0-0.4*u1; a=2*h; c=tanh(h); F=a+0.7*c",
            "parameters": {"h_coefficient": [0.6, -0.4], "a_scale": 2.0, "c_scale": 0.7},
            "paths": ["u0-h-a-F", "u1-h-a-F", "u0-h-c-F", "u1-h-c-F"],
            "ports": [
                port("FANOUT2", "h.linear.slot0", "h.ssa0", "scale_a", 0, ["u0-h-a-F", "u1-h-a-F"]),
                port("FANOUT2", "h.tanh.slot0", "h.ssa0", "tanh_c", 0, ["u0-h-c-F", "u1-h-c-F"]),
            ],
            "analytic_assertions": ["DISTINCT_CONSUMER_USES"],
        },
        {
            "fixture_id": "RESIDUAL2",
            "structure": "residual",
            "b": [0.0, 0.0],
            "x": [0.9, -1.3],
            "expression": "h=tanh(W@u+q); r=u+B@h; F=v@r",
            "parameters": {
                "W": [[0.8, -0.3], [0.2, 0.9]],
                "q": [0.1, -0.2],
                "B": [[0.7, 0.1], [-0.4, 0.6]],
                "v": [1.2, -0.5],
            },
            "paths": ["u0-id-add-F", "u1-id-add-F", "u0-branch-add-F", "u1-branch-add-F"],
            "ports": [
                port("RESIDUAL2", "u.add.slot0", "u.ssa0", "residual_add", 0, ["u0-id-add-F", "u1-id-add-F"]),
                port("RESIDUAL2", "branch.add.slot1", "branch.ssa0", "residual_add", 1, ["u0-branch-add-F", "u1-branch-add-F"]),
            ],
            "analytic_assertions": ["IDENTITY_AND_BRANCH_COMPLETE_CUT"],
        },
        {
            "fixture_id": "MULTIINPUT3",
            "structure": "multi_input_operator",
            "b": [0.0, 0.0, 0.0],
            "x": [0.8, -1.1, 0.6],
            "expression": "a=u0+2*u1; c=0.5*u1-u2; m=a*c; F=m+0.3*a-0.2*c",
            "parameters": {"a_coefficient": [1.0, 2.0, 0.0], "c_coefficient": [0.0, 0.5, -1.0]},
            "paths": ["u0-a-mul0-F", "u1-a-mul0-F", "u1-c-mul1-F", "u2-c-mul1-F",
                      "u0-a-linear-F", "u1-a-linear-F", "u1-c-linear-F", "u2-c-linear-F"],
            "ports": [
                port("MULTIINPUT3", "a.mul.slot0", "a.ssa0", "mul", 0, ["u0-a-mul0-F", "u1-a-mul0-F"]),
                port("MULTIINPUT3", "c.mul.slot1", "c.ssa0", "mul", 1, ["u1-c-mul1-F", "u2-c-mul1-F"]),
                port("MULTIINPUT3", "a.linear.slot0", "a.ssa0", "scale_a", 0, ["u0-a-linear-F", "u1-a-linear-F"]),
                port("MULTIINPUT3", "c.linear.slot0", "c.ssa0", "scale_c", 0, ["u1-c-linear-F", "u2-c-linear-F"]),
            ],
            "analytic_assertions": ["INPUT_SLOTS_DISTINCT"],
        },
    ]
    return {
        "schema_version": 1,
        "stage": "ENDPOINT_PCIA_PEFA_STEP2",
        "status": "FROZEN_BEFORE_OUTCOME",
        "dtype": "float64",
        "path": "gamma(t)=b+t*(x-b)",
        "fixture_order": [row["fixture_id"] for row in fixtures],
        "fixtures": fixtures,
        "nonsmooth_negative_controls": [
            {"fixture_id": "RELU_SAFE", "b": [-1.0], "x": [1.0], "expression": "F=relu(u0)", "kink": 0.0},
            {"fixture_id": "RELU_HIT", "b": [-1.0], "x": [1.0], "expression": "F=relu(u0-u_star)",
             "kink_construction": "u_star=-1.0+2.0*t64[31]"},
        ],
        "finite_difference": {"epsilon": 1e-6, "evaluation_t": [0.0, 0.37, 1.0]},
        "tolerance": {"atol": 1e-9, "rtol": 1e-7, "near_zero": 1e-12},
    }


def quadrature_registry() -> dict[str, Any]:
    schedules = []
    for count, role in ((64, "primary_gate"), (128, "convergence_diagnostic")):
        nodes, weights = np.polynomial.legendre.leggauss(count)
        nodes = ((nodes + 1.0) / 2.0).astype("<f8")
        weights = (weights / 2.0).astype("<f8")
        digest = hashlib.sha256(nodes.tobytes(order="C") + weights.tobytes(order="C")).hexdigest()
        schedules.append({
            "count": count,
            "role": role,
            "generation": "numpy.polynomial.legendre.leggauss; t=(s+1)/2; w=w/2",
            "dtype": "float64_little_endian",
            "order": "ascending_t",
            "nodes": nodes.tolist(),
            "weights": weights.tolist(),
            "nodes_then_weights_float64_sha256": digest,
        })
    return {
        "schema_version": 1,
        "stage": "ENDPOINT_PCIA_PEFA_STEP2",
        "status": "FROZEN_BEFORE_OUTCOME",
        "interval": [0.0, 1.0],
        "schedules": schedules,
        "relu_node_contract": {
            "left_index_zero_based": 31,
            "left_t_expected": 0.48782485366828776,
            "right_index_zero_based": 32,
            "right_t_expected": 0.5121751463317122,
            "minimum_input_distance_to_zero": 0.024350292663424478,
            "hit_kink_construction": "-1.0+2.0*t64[31]",
        },
    }


def write_or_verify(path: Path, payload: dict[str, Any]) -> str:
    raw = canonical_bytes(payload)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"frozen registry drift: {path}")
    else:
        path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    result = {
        "fixture_registry_sha256": write_or_verify(HERE / "fixture_registry.json", fixture_registry()),
        "quadrature_registry_sha256": write_or_verify(HERE / "quadrature_registry.json", quadrature_registry()),
        "status": "PASS_STEP2_REGISTRIES_FROZEN",
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
