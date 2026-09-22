#!/usr/bin/env python3
"""Run NNS-CST N1-R3 including all inherited R2 gates."""
from __future__ import annotations

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

import run_n1  # noqa: E402
from nns_cst.core import readout_score  # noqa: E402
from nns_cst_r3.readout import fit_frozen_readout  # noqa: E402


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_or_identical(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError("terminal evidence drift: " + str(path))
    if not path.exists():
        path.write_bytes(raw)


def assertion(label, condition):
    if not condition:
        raise AssertionError(label)
    return {"assertion": label, "passed": True, "max_abs_error": 0.0}


def run_determinism():
    runs = []
    for seed in ("0", "99173"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        environment["CUDA_VISIBLE_DEVICES"] = ""
        process = subprocess.run(
            [sys.executable, str(HERE / "n1_r3_determinism_probe.py")], cwd=str(HERE),
            env=environment, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True)
        row = json.loads(process.stdout)
        runs.append({"pythonhashseed": seed, "digest": row["digest"]})
    if runs[0]["digest"] != runs[1]["digest"]:
        raise AssertionError("R3 determinism mismatch")
    return {"status": "PASS", "process_count": 2, "runs": runs, "common_digest": runs[0]["digest"]}


def main():
    registry = load(HERE / "manifests/n1_r3/n1_fixture_registry.json")
    ports = load(HERE / "manifests/n1_r3/model_port_registry.json")
    run_n1.validate_port_registry(ports["records"], [row["port_id"] for row in ports["records"]])
    inherited_functions = [
        run_n1.fixture_rotated_subspace, run_n1.fixture_multi_head,
        run_n1.fixture_brightness_control, run_n1.fixture_transport,
        run_n1.fixture_complete_cut, run_n1.fixture_intervention,
        run_n1.fixture_typed_failures,
    ]
    fixtures = [function() for function in inherited_functions]
    contract_sha = hashlib.sha256((HERE / "N1数学合同.md").read_bytes()).hexdigest()
    readout = fit_frozen_readout(
        [[0.0, 2.0, 0.0], [1.0, 3.0, 0.0]],
        [[0.0, -2.0, 0.0], [-1.0, -3.0, 0.0]],
        [[0.0, 1.0, 0.0]], "engineering_fixture", contract_sha)
    closed_fields = {
        "schema_version", "document_type", "method", "weights", "threshold",
        "rank", "dimension", "fit_split", "contract_sha256",
        "validation_refit_allowed", "confirmation_refit_allowed",
    }
    fixtures.append({
        "fixture_id": "FROZEN_READOUT_SERIALIZATION",
        "assertions": [
            assertion("CLOSED_FIELD_SET", set(readout) == closed_fields),
            assertion("FIT_SPLIT_BOUND", readout["fit_split"] == "engineering_fixture"),
            assertion("CONTRACT_DIGEST_BOUND", readout["contract_sha256"] == contract_sha),
            assertion("VALIDATION_REFIT_DISABLED", readout["validation_refit_allowed"] is False),
            assertion("CONFIRMATION_REFIT_DISABLED", readout["confirmation_refit_allowed"] is False),
            assertion("POSITIVE_SCORE_DIRECTION", readout_score(readout, [0.0, 4.0, 0.0]) > 0.0),
            assertion("NEGATIVE_SCORE_DIRECTION", readout_score(readout, [0.0, -4.0, 0.0]) < 0.0),
        ],
    })
    expected = {row["fixture_id"]: set(row["required_assertions"]) for row in registry["fixtures"]}
    actual = {row["fixture_id"]: {item["assertion"] for item in row["assertions"]} for row in fixtures}
    if actual != expected:
        raise AssertionError("R3 fixture coverage mismatch")
    mutations = run_n1.run_mutations(ports)
    deterministic = run_determinism()
    benchmark = run_n1.microbenchmark(registry["microbenchmark"]["iterations"])
    if not all(math.isfinite(float(value)) for value in benchmark.values()):
        raise AssertionError("nonfinite benchmark")
    result = {
        "schema_version": 1,
        "document_type": "nns_cst_n1_r3_analytic_results",
        "stage_id": "N1",
        "status": "PASS_N1_R3_ANALYTIC_FIXTURES",
        "environment": {
            "sys_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version.split()[0], "device": "CPU",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
            "third_party_runtime_dependencies": [],
        },
        "fixtures": fixtures,
        "frozen_readout": readout,
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
    write_new_or_identical(EVIDENCE / "analytic_results.json", result)
    print(json.dumps({"status": result["status"], "fixture_count": len(fixtures),
                      "assertion_count": sum(len(row["assertions"]) for row in fixtures),
                      "mutation_count": len(mutations), "determinism_digest": deterministic["common_digest"],
                      "microbenchmark_wall_seconds": benchmark["wall_seconds"]},
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
