#!/usr/bin/env python3
"""Execute frozen NNS-CST N1 CPU-only analytic and mutation fixtures."""
from __future__ import annotations

import copy
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from nns_cst.core import (  # noqa: E402
    CSTError, FAIL_CUT_MISSING_EDGE, FAIL_MISSING_TOKEN,
    FAIL_NO_SEMANTIC_AFTER_CONTROL,
    FAIL_NEAR_SINGULAR_CONTROL_COVARIANCE, FAIL_NONFINITE_VALUE,
    FAIL_OFF_MANIFOLD, FAIL_PORT_IDENTITY, FAIL_PORT_DUPLICATE,
    FAIL_RANK_DEFICIENT, FAIL_RANK_EXCEEDS_DIMENSION, FAIL_TOKEN_ALIGNMENT,
    FAIL_ZERO_PAIR_DIFFERENCE, add, canonical_digest, complement, dot,
    estimate_subspace, intervene, matvec, norm, orthonormalize, project,
    projector_distance, projector_similarity, semantic_complement_attribution,
    subtract, transport_linear, transport_residual, transport_two_branch,
    validate_port_registry, validate_token_alignment,
)

EVIDENCE = HERE / "evidence/n1_r2_20260922"
ATOL = 1e-9
RTOL = 1e-7


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError("terminal evidence drift: " + str(path))
    if not path.exists():
        path.write_bytes(raw)


def close(actual, expected, label, atol=ATOL, rtol=RTOL):
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(label + " length")
        errors = [abs(float(a) - float(b)) for a, b in zip(actual, expected)]
        limits = [atol + rtol * abs(float(b)) for b in expected]
        passed = all(error <= limit for error, limit in zip(errors, limits))
        maximum = max(errors) if errors else 0.0
    else:
        maximum = abs(float(actual) - float(expected))
        passed = maximum <= atol + rtol * abs(float(expected))
    if not passed:
        raise AssertionError("%s max error %.12g" % (label, maximum))
    return {"assertion": label, "passed": True, "max_abs_error": maximum}


def ok(label, condition):
    if not condition:
        raise AssertionError(label)
    return {"assertion": label, "passed": True, "max_abs_error": 0.0}


def fixture_rotated_subspace():
    root = math.sqrt(0.5)
    reference = orthonormalize([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotated = orthonormalize([[root, root, 0.0], [-root, root, 0.0]])
    value = [2.0, -3.0, 4.0]
    assertions = [
        close(project(value, reference), project(value, rotated), "PROJECTION_INVARIANCE"),
        close(projector_distance(reference, rotated), 0.0, "PROJECTOR_INVARIANCE"),
        close(projector_similarity(reference, rotated), 1.0, "SIMILARITY_ONE"),
    ]
    return {"fixture_id": "ROTATED_TWO_DIMENSIONAL_SUBSPACE", "assertions": assertions}


def fixture_multi_head():
    head_a = [1.0, 0.0]
    head_b = [0.0, 1.0]
    semantic = [1.0, 1.0]
    threshold = 1.5
    assertions = [
        ok("EACH_SINGLE_HEAD_INCOMPLETE", dot(head_a, semantic) < threshold and dot(head_b, semantic) < threshold),
        ok("JOINT_SUBSPACE_COMPLETE", dot(add(head_a, head_b), semantic) >= threshold),
    ]
    return {"fixture_id": "DISTRIBUTED_MULTI_HEAD_SIGNAL", "assertions": assertions}


def fixture_brightness_control():
    result = estimate_subspace(
        [[10.0, 2.0, 0.0], [9.0, 2.0, 0.0], [11.0, 2.0, 0.0]], rank=1,
        control_differences=[[1.0, 0.0, 0.0]])
    semantic_basis = result["basis"]
    assertions = [
        close(norm(project([1.0, 0.0, 0.0], semantic_basis)), 0.0, "NUISANCE_REMOVED"),
        close(norm(project([0.0, 1.0, 0.0], semantic_basis)), 1.0, "SEMANTIC_RETAINED"),
    ]
    return {"fixture_id": "BRIGHTNESS_CONTROL_RESIDUALIZATION", "assertions": assertions}


def fixture_transport():
    value = [2.0, -1.0]
    identity = [[1.0, 0.0], [0.0, 1.0]]
    linear = [[2.0, 3.0], [-1.0, 4.0]]
    branch = [[0.5, -0.25], [0.25, 0.5]]
    first = [[1.0, 2.0], [0.0, -1.0]]
    second = [[-0.5, 0.0], [1.5, 2.0]]
    residual = transport_residual(identity, branch, value)
    routed = transport_two_branch(first, second, value)
    assertions = [
        close(transport_linear(identity, value), value, "IDENTITY"),
        close(transport_linear(linear, value), [1.0, -6.0], "LINEAR"),
        close(residual["merged"], add(value, matvec(branch, value)), "RESIDUAL"),
        close(routed["merged"], add(matvec(first, value), matvec(second, value)), "TWO_BRANCH"),
    ]
    return {"fixture_id": "ANALYTIC_TRANSPORT", "assertions": assertions}


def fixture_complete_cut():
    value = [2.0, -3.0, 4.0]
    gradient = [0.5, 2.0, -1.0]
    basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    result = semantic_complement_attribution(value, gradient, basis)
    assertions = [
        close(result["reconstructed"], result["complete_cut"], "COORDINATE_CLOSURE"),
        close(math.fsum(result["reconstructed"]), math.fsum(result["complete_cut"]), "SCALAR_CLOSURE"),
    ]
    return {"fixture_id": "SEMANTIC_COMPLEMENT_COMPLETE_CUT", "assertions": assertions}


def fixture_intervention():
    basis = [[1.0, 0.0, 0.0]]
    recipient = [2.0, 3.0, 4.0]
    donor = [-1.0, 7.0, 9.0]
    assertions = [
        close(intervene(recipient, basis, "inject", donor), [-1.0, 3.0, 4.0], "INJECTION"),
        close(intervene(recipient, basis, "erase"), [0.0, 3.0, 4.0], "ERASURE"),
        close(intervene(recipient, basis, "random_subspace"), [2.0, 0.0, 4.0], "RANDOM_SUBSPACE"),
        close(intervene(recipient, basis, "orthogonal_complement"), [0.0, 3.0, 4.0], "ORTHOGONAL_COMPLEMENT"),
    ]
    return {"fixture_id": "INTERVENTION_ANALYTICS", "assertions": assertions}


def expect_failure(label, code, function):
    try:
        function()
    except CSTError as exc:
        if exc.code != code:
            raise AssertionError("%s expected %s got %s" % (label, code, exc.code))
        return {"assertion": label, "passed": True, "observed_failure": exc.code, "max_abs_error": 0.0}
    raise AssertionError(label + " did not fail")


def fixture_typed_failures():
    assertions = [
        expect_failure("RANK_TOO_LARGE", FAIL_RANK_EXCEEDS_DIMENSION,
                       lambda: estimate_subspace([[1.0, 0.0]], 3)),
        expect_failure("RANK_DEFICIENT", FAIL_RANK_DEFICIENT,
                       lambda: estimate_subspace([[1.0, 0.0], [2.0, 0.0]], 2)),
        expect_failure("ZERO_DIFFERENCE", FAIL_ZERO_PAIR_DIFFERENCE,
                       lambda: estimate_subspace([[0.0, 0.0]], 1)),
        expect_failure("NONFINITE", FAIL_NONFINITE_VALUE,
                       lambda: estimate_subspace([[float("nan"), 0.0]], 1)),
        expect_failure("NEAR_SINGULAR_CONTROL", FAIL_NEAR_SINGULAR_CONTROL_COVARIANCE,
                       lambda: estimate_subspace([[1.0, 2.0]], 1, control_differences=[[1.0, 0.0], [2.0, 0.0]])),
        expect_failure("NO_SEMANTIC_AFTER_CONTROL", FAIL_NO_SEMANTIC_AFTER_CONTROL,
                       lambda: estimate_subspace([[1.0, 0.0], [2.0, 0.0]], 1, control_differences=[[1.0, 0.0]])),
        expect_failure("MISSING_TOKEN", FAIL_MISSING_TOKEN,
                       lambda: estimate_subspace([[1.0, 2.0], [1.0]], 1)),
        expect_failure("OFF_MANIFOLD", FAIL_OFF_MANIFOLD,
                       lambda: intervene([1.0, 0.0], [[1.0, 0.0]], "inject", [100.0, 0.0], manifold_limit=0.1)),
    ]
    return {"fixture_id": "TYPED_FAILURES", "assertions": assertions}


def run_mutations(port_registry):
    records = port_registry["records"]
    expected_ids = [row["port_id"] for row in records]
    output = []

    def mutation(mid, expected, mutated, function):
        try:
            function(mutated)
        except CSTError as exc:
            if exc.code != expected:
                raise AssertionError("%s expected %s got %s" % (mid, expected, exc.code))
            output.append({"mutation_id": mid, "expected_failure": expected,
                           "observed_failure": exc.code, "passed": True})
            return
        raise AssertionError(mid + " escaped")

    mutation("OMIT_EDGE", FAIL_CUT_MISSING_EDGE, records[:-1],
             lambda value: validate_port_registry(value, expected_ids))
    mutation("DUPLICATE_EDGE", FAIL_PORT_DUPLICATE, records + [copy.deepcopy(records[0])],
             lambda value: validate_port_registry(value, expected_ids))
    qkv = next(i for i, row in enumerate(records) if row["tensor_role"] == "attention_head_query_input")
    swapped = copy.deepcopy(records)
    swapped[qkv]["input_slot"] = 1
    mutation("SWAP_Q_K_V_SLOT", FAIL_PORT_IDENTITY, swapped,
             lambda value: validate_port_registry(value, expected_ids))
    mutation("WRONG_TOKEN_ALIGNMENT", FAIL_TOKEN_ALIGNMENT,
             [("CLS", "CLS"), ("P0", "P1"), ("P1", "P1")],
             lambda value: validate_token_alignment(["CLS", "P0", "P1"], ["CLS", "P0", "P1"], value))
    return output


def determinism_probe():
    results = []
    for seed in ("0", "99173"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        environment["CUDA_VISIBLE_DEVICES"] = ""
        process = subprocess.run(
            [sys.executable, str(HERE / "n1_determinism_probe.py")], cwd=str(HERE),
            env=environment, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True)
        row = json.loads(process.stdout)
        results.append({"pythonhashseed": seed, "digest": row["digest"]})
    if results[0]["digest"] != results[1]["digest"]:
        raise AssertionError("determinism digest mismatch")
    return {"status": "PASS", "process_count": 2, "runs": results,
            "common_digest": results[0]["digest"]}


def microbenchmark(iterations):
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    for _ in range(iterations):
        result = estimate_subspace([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [2.0, 3.0, 0.0]], rank=2)
        transport_residual([[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 0.5]], [2.0, -1.0])
        semantic_complement_attribution([2.0, -3.0, 4.0], [0.5, 2.0, -1.0], result["basis"])
    wall = time.perf_counter() - started_wall
    cpu = time.process_time() - started_cpu
    result = {
        "iterations": iterations, "wall_seconds": wall, "cpu_seconds": cpu,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "iterations_per_second": iterations / wall,
    }
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in result.values()):
        raise AssertionError("invalid benchmark result")
    return result


def execute_suite(write=True):
    fixture_registry = load(HERE / "manifests/n1_r2/n1_fixture_registry.json")
    port_registry = load(HERE / "manifests/n1_r2/model_port_registry.json")
    validate_port_registry(port_registry["records"], [row["port_id"] for row in port_registry["records"]])
    fixture_functions = [
        fixture_rotated_subspace, fixture_multi_head, fixture_brightness_control,
        fixture_transport, fixture_complete_cut, fixture_intervention, fixture_typed_failures,
    ]
    fixtures = [function() for function in fixture_functions]
    expected = {row["fixture_id"]: set(row["required_assertions"]) for row in fixture_registry["fixtures"]}
    actual = {row["fixture_id"]: {item["assertion"] for item in row["assertions"]} for row in fixtures}
    if expected != actual:
        raise AssertionError("fixture assertion coverage mismatch")
    mutations = run_mutations(port_registry)
    deterministic = determinism_probe()
    benchmark = microbenchmark(fixture_registry["microbenchmark"]["iterations"])
    result = {
        "schema_version": 1,
        "document_type": "nns_cst_n1_analytic_results",
        "stage_id": "N1",
        "status": "PASS_N1_ANALYTIC_FIXTURES",
        "environment": {
            "sys_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version.split()[0],
            "device": "CPU",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
            "third_party_runtime_dependencies": [],
        },
        "fixtures": fixtures,
        "mutations": mutations,
        "determinism": deterministic,
        "microbenchmark": benchmark,
        "scientific_payload_digest": deterministic["common_digest"],
        "execution_boundary": {
            "formal_image_decodes": 0, "model_loads": 0, "checkpoint_deserializations": 0,
            "gpu_or_cuda_uses": 0, "training_runs": 0, "inference_runs": 0,
            "formal_outcome_reads": 0, "n2_started": False,
        },
    }
    if write:
        write_json(EVIDENCE / "analytic_results.json", result)
    return result


def main():
    result = execute_suite(write=True)
    print(json.dumps({
        "status": result["status"], "fixture_count": len(result["fixtures"]),
        "mutation_count": len(result["mutations"]),
        "determinism_digest": result["scientific_payload_digest"],
        "microbenchmark_wall_seconds": result["microbenchmark"]["wall_seconds"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
