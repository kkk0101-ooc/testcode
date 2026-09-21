#!/usr/bin/env python3
"""Freeze the prospective Step-5.1 protocol and all A0/A1 identities."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RUN_ID = os.environ.get("STEP5_1_RUN_ID", "step5_1_r1_20260921")
if RUN_ID not in {"step5_1_r1_20260921", "step5_1_r2_20260921", "step5_1_r3_20260921",
                  "step5_1_r4_20260921", "step5_1_r5_20260921", "step5_1_r6_20260921",
                  "step5_1_r7_20260921"}:
    raise RuntimeError("STEP5_1_RUN_ID")
RUN = ROOT / RUN_ID
EP0 = REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value); result["canonical_digest"] = hashlib.sha256(canonical(value)).hexdigest()
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, mode: str = "source_or_registry") -> dict[str, Any]:
    return {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "mode": mode}


def write_new(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    if RUN.exists():
        raise RuntimeError("STEP5_1_ROOT_EXISTS")
    registry = json.loads((EP0 / "model_registry.json").read_text())
    cohort = json.loads((EP0 / "cohort_registry.json").read_text())
    models = {row["model_id"]: row for row in registry["models"]}
    if set(models) != {"vit_b16", "deit_s16"}:
        raise RuntimeError("MODEL_REGISTRY_DRIFT")
    rows = {(row["split"], int(row["split_order"])): row for row in cohort["rows"]}
    discovery = rows[("discovery", 0)]
    discovery_path = REPO / "vit_pcia_pefa/step4_runtime/discovery" / discovery["sample_id"]
    if not discovery_path.is_file():
        raise RuntimeError("A0_DISCOVERY_ASSET_MISSING")

    checkpoints = []
    for model_id, model in models.items():
        path = REPO / "vitnns_rebuild" / model["checkpoint"]["path"]
        if path.stat().st_size != model["checkpoint"]["bytes"] or sha256(path) != model["checkpoint"]["sha256"]:
            raise RuntimeError("CHECKPOINT_IDENTITY:" + model_id)
        checkpoints.append({"model_id": model_id, **identity(path, "checkpoint_stream_hash_no_deserialization")})

    images = {}
    a1_tasks = []
    for order in range(8, 16):
        row = rows[("calibration", order)]
        path = REPO / "vitnns_rebuild/artifacts/nns_vit_endpoint_ep2_r1_20260918/calibration_images" / row["sample_id"]
        if not path.is_file():
            raise RuntimeError("A1_CALIBRATION_ASSET_MISSING:" + row["sample_id"])
        image = identity(path, "frozen_calibration_asset_no_decode_during_freeze")
        images[row["sample_id"]] = image
        for model_id in ("deit_s16", "vit_b16"):
            heads = int(models[model_id]["heads"])
            true_head = (order - 8) % heads
            payload = {
                "task_id": f"{model_id}__{order:02d}", "model_id": model_id,
                "sample_id": row["sample_id"], "class_id": int(row["class_id"]),
                "split": "calibration", "split_order": order,
                "image_path": image["path"], "image_bytes": image["bytes"],
                "image_sha256": image["sha256"], "block_index": 5,
                "true_head": true_head, "decoy_head": (true_head + 1) % heads,
                "head_count": heads, "top_k": int(math.ceil(0.25 * heads)),
            }
            a1_tasks.append(payload)

    nodes, weights = np.polynomial.legendre.leggauss(512)
    nodes = np.ascontiguousarray((nodes + 1.0) / 2.0, dtype="<f8")
    weights = np.ascontiguousarray(weights / 2.0, dtype="<f8")
    protocol = signed({
        "schema_version": 1, "run_id": RUN.name, "stage": "ENDPOINT_PCIA_PEFA_STEP5_1",
        "status": ("REPORTING_ONLY_RECOVERY_FROZEN" if RUN_ID.endswith("r7_20260921")
                   else "FROZEN_PROSPECTIVE"),
        "authorization": "USER_REQUEST_EXECUTE_STEP_5_1_20260921",
        "prior_results_immutable": {
            "step5": "NOT_SUPPORTED_EMPIRICAL_VALUE",
            "extension": "PARTIAL_NOT_SUPPORTED_WITH_E3_INVALID",
        },
        "evaluation_access": False, "step6_authorized": False,
        "models": ["deit_s16", "vit_b16"], "target": "ground_truth_raw_logit",
        "reference": "normalized_zero", "path": "straight",
        "cut_id": "HEADS_WITH_BYPASS_FULL", "real_block_index": 5,
        "wrapper": {
            "use_port": "attn.proj.input_slot_0_head_slice",
            "formula": "z + alpha*s*tanh(((z-z_star)/s)^2)",
            "scale": "selected_clean_head_rms_clamped_min_1e-6_detached",
            "clean_value_parity": True, "clean_jacobian_identity": True,
            "a0_strengths": [0.25, 0.5, 1.0], "a1_strengths": [0.5, 1.0],
            "controls": (["decoy_port_same_strength_0.5", "decoy_port_same_strength_1.0",
                          "linear_tangent_1.0", "zero_strength"]
                         if RUN_ID.endswith(("r3_20260921", "r4_20260921", "r5_20260921",
                                             "r6_20260921", "r7_20260921")) else
                         ["decoy_port_same_strength_1.0", "linear_tangent_1.0", "zero_strength"]),
        },
        "integration": {
            "rule": "Gauss-Legendre", "real_nodes": 512, "fixture_nodes": 64,
            "nodes_sha256": hashlib.sha256(nodes.tobytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        },
        "numeric": {
            "fixture": {"dtype": "float64", "atol": 1e-9, "rtol": 1e-7, "near_zero": 1e-12},
            "real": {"forward_dtype": "float32", "reduction_dtype": "float64",
                     "atol": 1e-5, "rtol": 1e-4, "near_zero": 1e-6,
                     "tf32": False, "autocast": False},
        },
        "a0": {
            "scientific_claim": False, "fixture_block": 1, "development_split": "discovery",
            "development_split_order": 0,
            "gate": "all parity and cut closure pass; controls zero; nonzero alpha=0.25 true-head signal; nondecreasing true-head PEFA-R delta",
        },
        "a1": {
            "split": "calibration", "split_order_range": [8, 16], "image_count": 8,
            "task_count": 16, "statistical_unit": "image",
            "methods": ["pefa_R_delta", "pcia_delta", "pcia_level",
                        "endpoint_delta", "endpoint_level", "gradient_delta",
                        "gradient_level", "activation_delta", "activation_level", "random"],
            "metrics": ["top1", "top_quarter", "rank", "reciprocal_rank",
                        "percentile_rank", "paired_difference", "delta_R_e", "delta_G_direction"],
            "controlled_gate_each_model_each_strength": {
                "pefa_top1_min": 0.75, "pefa_top_k_min": 0.875,
                "pefa_mean_percentile_min": 0.90, "decoy_same_thresholds": True,
                "control_signal_max": "numeric_atol",
            },
            "incremental_gate_each_model_each_strength": {
                "roles": ["primary", "decoy"],
                "mean_percentile_advantage_over_every_baseline": 0.05,
                "top1_rate_advantage_over_every_baseline": 0.125,
            },
        },
        "b1": {
            "gated_on_a1": True, "lambdas": [1.0, 0.5, 0.0],
            "nonlinear_modules": ["attention_softmax", "qk_matmul", "av_matmul", "gelu", "layernorm"],
            "low_curvature_controls": ["linear_projection", "residual_add"],
            "incremental_rank_correlation_margin": 0.05,
        },
        "c1": {
            "gated_on_valid_a0": True, "independent_decision": True,
            "held_out_mutation": "consumer_head_slice_rotation",
            "expected_failure": "FAIL_PORT_ID_SEMANTIC_MISMATCH",
            "expected_location": {"block": 5, "cut": "HEADS_WITH_BYPASS_FULL", "category": "head"},
        },
        "claims": ["CONTROLLED_CURVATURE_LOCALIZATION", "NATURAL_NONLINEARITY_LOCALIZATION",
                   "ATTRIBUTION_ERROR_LOCALIZATION", "INCREMENTAL_OVER_SIMPLE_BASELINES",
                   "NUMERICAL_VALIDITY_AND_COST"],
        "resources": {
            "step5_1_h20_hours_cap": 32, "cumulative_step5_h20_hours_cap": 64,
            "predecessor_h20_hours": (13.331727234210381 if RUN_ID.endswith("r7_20260921") else
                                      (13.104299663088688 if RUN_ID.endswith(("r5_20260921", "r6_20260921")) else
                                      (12.854291645377577 if RUN_ID.endswith(("r3_20260921", "r4_20260921")) else
                                      (12.761258363979469 if RUN_ID.endswith("r2_20260921") else 12.75807705888834)))),
            "maximum_gpus": 4,
            "minimum_free_gpu_mib": 32768, "maximum_gpu_utilization_percent": 10,
            "gpu_memory_cap_bytes": 40 * 1024**3, "cpu_rss_cap_bytes": 32 * 1024**3,
            "task_timeout_seconds": 7200, "max_attempts": 2,
            "retained_bytes_cap": 128 * 1024**3,
        },
        "failure_codes": ["FAIL_SOURCE_BINDING", "INVALID_INTERVENTION_CONSTRUCTION",
                          "FAIL_QUADRATURE_CLOSURE", "FAIL_NONFINITE",
                          "FAIL_PORT_ID_SEMANTIC_MISMATCH", "PAUSED_RESOURCE_CAP"],
    })
    manifest = signed({
        "schema_version": 1, "run_id": RUN.name,
        "a0_tasks": [{
            "task_id": f"a0__{model_id}", "model_id": model_id,
            "sample_id": discovery["sample_id"], "class_id": int(discovery["class_id"]),
            "split": "discovery", "split_order": 0,
            "image_path": discovery_path.relative_to(REPO).as_posix(),
            "image_bytes": discovery_path.stat().st_size, "image_sha256": sha256(discovery_path),
            "block_index": 5,
            "true_head": int.from_bytes(hashlib.sha256(f"ENDPOINT_PCIA_PEFA|step5_1|a0|{model_id}".encode()).digest()[:8], "big") % int(models[model_id]["heads"]),
        } for model_id in ("deit_s16", "vit_b16")],
        "a1_tasks": a1_tasks, "evaluation_access": False, "replacement_allowed": False,
    })
    source_paths = [
        ROOT / "执行规划.md", ROOT / "method_contract.md", ROOT / "step5_1_protocol.md",
        ROOT / "src/vit_adapter.py", ROOT / "src/pilot.py", ROOT / "src/step5_1_localization.py",
        ROOT / "src/step5_1_queue.py",
        ROOT / "tests/test_step4_vit_adapter.py", ROOT / "tests/test_step5_pilot.py",
        ROOT / "tests/test_step5_1_localization.py", ROOT / "freeze_step5_1.py",
        ROOT / "tests/test_step5_1_queue.py",
        ROOT / "run_step5_1_a0_fixture.py", ROOT / "run_step5_1_a0_worker.py",
        ROOT / "launch_step5_1_a0.py",
        ROOT / "finalize_step5_1_a0.py", ROOT / "validate_step5_1.py",
        ROOT / "run_step5_1_a1_worker.py", ROOT / "launch_step5_1_a1.py",
        ROOT / "run_step5_1_c1.py", ROOT / "finalize_step5_1.py",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_vit.py", EP0 / "model_registry.json",
        EP0 / "cohort_registry.json",
    ]
    if RUN_ID.endswith("r2_20260921"):
        source_paths.append(ROOT / "step5_1_r2_recovery_amendment.md")
    if RUN_ID.endswith("r3_20260921"):
        source_paths.extend((ROOT / "step5_1_r2_recovery_amendment.md",
                             ROOT / "step5_1_r3_recovery_amendment.md",
                             ROOT / "inherit_step5_1_r3.py"))
    if RUN_ID.endswith("r4_20260921"):
        source_paths.extend((ROOT / "step5_1_r2_recovery_amendment.md",
                             ROOT / "step5_1_r3_recovery_amendment.md",
                             ROOT / "step5_1_r4_recovery_amendment.md",
                             ROOT / "inherit_step5_1_r4.py"))
    if RUN_ID.endswith("r5_20260921"):
        source_paths.extend((ROOT / "step5_1_r2_recovery_amendment.md",
                             ROOT / "step5_1_r3_recovery_amendment.md",
                             ROOT / "step5_1_r4_recovery_amendment.md",
                             ROOT / "step5_1_r5_recovery_amendment.md",
                             ROOT / "inherit_step5_1_r5.py"))
    if RUN_ID.endswith("r6_20260921"):
        source_paths.extend((ROOT / "step5_1_r2_recovery_amendment.md", ROOT / "step5_1_r3_recovery_amendment.md",
                             ROOT / "step5_1_r4_recovery_amendment.md", ROOT / "step5_1_r5_recovery_amendment.md",
                             ROOT / "step5_1_r6_recovery_amendment.md", ROOT / "inherit_step5_1_r6.py"))
    if RUN_ID.endswith("r7_20260921"):
        source_paths.extend((ROOT / "step5_1_r2_recovery_amendment.md", ROOT / "step5_1_r3_recovery_amendment.md",
                             ROOT / "step5_1_r4_recovery_amendment.md", ROOT / "step5_1_r5_recovery_amendment.md",
                             ROOT / "step5_1_r6_recovery_amendment.md", ROOT / "step5_1_r7_recovery_amendment.md",
                             ROOT / "inherit_step5_1_r7.py", ROOT / "step5_completion_20260921/resource_ledger.json"))
    missing = [path.name for path in source_paths if not path.is_file()]
    if missing:
        raise RuntimeError("STEP5_1_SOURCE_MISSING:" + ",".join(missing))
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "vit_pcia_pefa/tests/test_step4_vit_adapter.py",
         "vit_pcia_pefa/tests/test_step5_pilot.py",
         "vit_pcia_pefa/tests/test_step5_1_localization.py",
         "vit_pcia_pefa/tests/test_step5_1_queue.py"],
        cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    )
    dirty = subprocess.check_output(["git", "status", "--porcelain=v2", "--untracked-files=all"], cwd=REPO, text=True)
    binding = signed({
        "schema_version": 1, "run_id": RUN.name,
        "status": ("REPORTING_ONLY_RECOVERY_FROZEN" if RUN_ID.endswith("r7_20260921")
                   else "FROZEN_PROSPECTIVE"),
        "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "producer_branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True).strip(),
        "producer_dirty": bool(dirty), "producer_dirty_status_sha256": hashlib.sha256(dirty.encode()).hexdigest(),
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "files": [identity(path) for path in source_paths], "checkpoints": checkpoints,
        "images": [identity(discovery_path, "A0_development_asset"), *images.values()],
        "calibration_selection_before_outcome": "split_order_8_through_15",
        "evaluation_member_reads": 0,
        "outcome_reads_before_freeze": (16 if RUN_ID.endswith("r7_20260921") else 0),
        "outcomes_known_before_reporting_freeze": RUN_ID.endswith("r7_20260921"),
        "preaccess_tests": tests.stdout.strip(),
        "recovery": ({
            "predecessor_root": "vit_pcia_pefa/step5_1_r1_20260921",
            "trigger": "A0 manifest omitted block_index; both workers failed before model deserialization or outcome computation",
            "scientific_contract_changed": False,
        } if RUN_ID.endswith("r2_20260921") else ({
            "predecessor_root": "vit_pcia_pefa/step5_1_r2_20260921",
            "trigger": "pre-A1 structured-control ambiguity; require same-strength decoy at both frozen primary strengths",
            "scientific_contract_changed": False, "a0_inherited": True,
        } if RUN_ID.endswith("r3_20260921") else ({
            "predecessor_root": "vit_pcia_pefa/step5_1_r3_20260921",
            "trigger": "A1 queue rejected flat closed manifest rows before queue schema creation or outcome computation",
            "scientific_contract_changed": False, "a0_and_c1_inherited": True,
        } if RUN_ID.endswith("r4_20260921") else ({
            "predecessor_root": "vit_pcia_pefa/step5_1_r4_20260921",
            "trigger": "A1 worker read task_id from payload instead of claim envelope; zero accepted A1 results",
            "scientific_contract_changed": False, "a0_and_c1_inherited": True,
        } if RUN_ID.endswith("r5_20260921") else ({
            "predecessor_root": "vit_pcia_pefa/step5_1_r5_20260921",
            "trigger": "reporting-only correction for r4 resource accounting and unrun B1 claim label",
            "scientific_contract_changed": False, "all_results_inherited": True,
        } if RUN_ID.endswith("r6_20260921") else ({
            "predecessor_root": "vit_pcia_pefa/step5_1_r6_20260921",
            "trigger": "completion audit found that r6 cumulative accounting omitted the previously sealed post-extension diagnosis and interrupted-attempt upper bound",
            "scientific_contract_changed": False, "all_results_inherited": True,
            "reporting_only": True, "outcomes_known": True,
        } if RUN_ID.endswith("r7_20260921") else None)))))),
    })
    RUN.mkdir()
    write_new(RUN / "protocol.json", protocol)
    write_new(RUN / "task_manifest.json", manifest)
    write_new(RUN / "source_binding.json", binding)
    write_new(RUN / "freeze_receipt.json", signed({
        "schema_version": 1, "run_id": RUN.name,
        "status": ("REPORTING_ONLY_RECOVERY_FROZEN" if RUN_ID.endswith("r7_20260921")
                   else "FROZEN_PROSPECTIVE"),
        "frozen_unix_ns": time.time_ns(), "protocol_digest": protocol["canonical_digest"],
        "task_manifest_digest": manifest["canonical_digest"],
        "source_binding_digest": binding["canonical_digest"],
        "a0_started": False, "a1_started": False, "b1_started": False,
        "c1_started": False, "evaluation_access": False, "step6_started": False,
    }))
    print(json.dumps({"status": protocol["status"], "root": RUN.relative_to(REPO).as_posix(),
                      "a0_tasks": 2, "a1_tasks": len(a1_tasks),
                      "protocol_digest": protocol["canonical_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
