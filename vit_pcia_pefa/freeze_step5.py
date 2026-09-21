#!/usr/bin/env python3
"""Freeze the Step-5 pilot before any pilot outcome is read."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PILOT = ROOT / "step5_pilot_r8_20260921"
EP0 = REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918"
EP2 = REPO / "vitnns_rebuild/reports/nns_vit_endpoint_ep2_r1_20260918"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def signed(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value); value["canonical_digest"] = digest(value); return value


def main() -> int:
    if PILOT.exists():
        raise RuntimeError("PILOT_ROOT_EXISTS")
    if shutil.disk_usage(REPO).free < 50 * 1024**3:
        raise RuntimeError("PAUSED_RESOURCE_CAP:DISK")
    predecessor = json.loads((ROOT / "vit_validation.json").read_text())
    cost = json.loads((ROOT / "cost_report.json").read_text())
    if predecessor["status"] != "PASS_VIT_PORT_FIDELITY_AND_IDENTITIES":
        raise RuntimeError("STEP4_NOT_PASS")
    if cost["status"] != "PASS_COST_GATE" or not cost["projected_production"]["pass"]:
        raise RuntimeError("STEP4_COST_NOT_PASS")
    tests = subprocess.run(
        [str(REPO / ".conda/envs/vitnns/bin/python"), "-m", "pytest", "-q",
         "vit_pcia_pefa/tests/test_step5_pilot.py", "vit_pcia_pefa/tests/test_step5_queue.py",
         "vit_pcia_pefa/tests/test_step4_vit_adapter.py"], cwd=REPO,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
             "PYTHONPATH": ".:vitnns_rebuild/src"}, capture_output=True, text=True,
    )
    if tests.returncode != 0:
        raise RuntimeError("STEP5_PREACCESS_TEST_FAIL:" + tests.stdout + tests.stderr)

    ep2_manifest_path = EP2 / "task_manifest.json"
    ep2_manifest = json.loads(ep2_manifest_path.read_text())
    tasks_by_key = {
        (row["payload"]["model_id"], int(row["payload"]["split_order"])): row["payload"]
        for row in ep2_manifest["tasks"]
    }
    pilot_tasks = []
    images: dict[str, dict[str, Any]] = {}
    for split_order in range(8):
        for model_id in ("deit_s16", "vit_b16"):
            source = tasks_by_key[(model_id, split_order)]
            image_path = REPO / "vitnns_rebuild" / source["cache_path"]
            if image_path.stat().st_size != source["image_bytes"] or sha256(image_path) != source["image_sha256"]:
                raise RuntimeError("CALIBRATION_IMAGE_DRIFT:" + source["sample_id"])
            image_relative = image_path.relative_to(REPO).as_posix()
            images[image_relative] = {
                "path": image_relative, "bytes": source["image_bytes"],
                "sha256": source["image_sha256"],
            }
            payload = {
                "model_id": model_id, "sample_id": source["sample_id"],
                "split": "calibration", "split_order": split_order,
                "class_id": int(source["class_id"]), "wnid": source["wnid"],
                "image_path": image_relative, "image_bytes": source["image_bytes"],
                "image_sha256": source["image_sha256"],
            }
            pilot_tasks.append({"task_id": f"{model_id}__{split_order:02d}", "payload": payload})

    source_paths = [
        ROOT / "执行规划.md", ROOT / "method_contract.md", ROOT / "step1_decision.md",
        ROOT / "step2_decision.md", ROOT / "step3_decision.md", ROOT / "step4_decision.md",
        ROOT / "vit_validation.json", ROOT / "cost_report.json",
        ROOT / "step4_attempt2_source_binding.json", ROOT / "vit_port_registry.json",
        ROOT / "src/vit_adapter.py", ROOT / "src/pilot.py", ROOT / "run_step5_worker.py",
        ROOT / "run_step5_omega_audit_worker.py", ROOT / "launch_step5_omega_audit.py",
        ROOT / "freeze_step5.py", ROOT / "launch_step5.py", ROOT / "finalize_step5.py",
        ROOT / "step5_protocol.md", ROOT / "step5_r2_recovery_amendment.md",
        ROOT / "step5_r3_recovery_amendment.md", ROOT / "step5_r4_recovery_amendment.md",
        ROOT / "step5_r5_recovery_amendment.md", ROOT / "inherit_step5_r2.py",
        ROOT / "step5_r6_recovery_amendment.md", ROOT / "step5_r7_recovery_amendment.md",
        ROOT / "step5_r8_recovery_amendment.md", ROOT / "step5_r6_contract_audit.md",
        ROOT / "step5_r7_discovery_probe_receipt.json",
        ROOT / "step5_r7_omega_diagnosis_receipt.json", ROOT / "diagnose_step5_r7_omega.py",
        ROOT / "inherit_step5_r3.py",
        ROOT / "inherit_step5_r4.py", ROOT / "inherit_step5_r5.py",
        ROOT / "step5_pilot_r1_20260921/failure_receipt.json",
        ROOT / "step5_pilot_r2_20260921/failure_receipt.json", ROOT / "tests/test_step5_pilot.py",
        ROOT / "step5_pilot_r3_20260921/failure_receipt.json",
        ROOT / "step5_pilot_r4_20260921/failure_receipt.json", ROOT / "validate_step5.py",
        ROOT / "step5_pilot_r5_20260921/failure_receipt.json",
        ROOT / "step5_pilot_r6_20260921/terminal_receipt.json",
        ROOT / "step5_pilot_r6_20260921/resource_ledger.json",
        ROOT / "step5_pilot_r7_20260921/failure_receipt.json",
        ROOT / "step5_pilot_r7_20260921/launch_terminal.json",
        ROOT / "step5_pilot_r7_20260921/source_binding.json",
        ROOT / "inherit_step5_r8.py",
        ROOT / "src/step5_queue.py", ROOT / "tests/test_step5_queue.py",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_nns.py",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_vit.py",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_ep1_queue.py",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_ep2.py",
        REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
        REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/cohort_registry.json",
        ep2_manifest_path,
    ]
    model_registry = json.loads((EP0 / "model_registry.json").read_text())
    checkpoints = []
    for row in model_registry["models"]:
        path = REPO / "vitnns_rebuild" / row["checkpoint"]["path"]
        observed = identity(path); observed["model_id"] = row["model_id"]
        if observed["bytes"] != row["checkpoint"]["bytes"] or observed["sha256"] != row["checkpoint"]["sha256"]:
            raise RuntimeError("CHECKPOINT_DRIFT:" + row["model_id"])
        checkpoints.append(observed)

    protocol = signed({
        "schema_version": 1, "stage": "ENDPOINT_PCIA_PEFA_STEP5",
        "run_id": "step5_pilot_r8_20260921", "status": "FROZEN_PROSPECTIVE",
        "predecessor_run_id": "step5_pilot_r7_20260921",
        "predecessor_status": "TERMINAL_MATHEMATICAL_CLOSURE_FAIL",
        "recovery_scope": "preserve float32 scientific Omega scores and move only the zero-field mathematical audit to a direct float64 complete-cut evaluation",
        "inherited_tasks": "14 canonical-valid r7 PASS_TASK results; rerun only vit_b16__02 and vit_b16__05; independently audit all 16 tasks in float64",
        "prior_outcome_known": True, "pilot_outcome_access_before_freeze": True,
        "outcome_access_during_execution": False,
        "recovery_trigger_outcome_independent": True,
        "authorization": "execute execution-plan step 5 frozen pilot",
        "step4_predecessor_status": "PASS_VIT_PORT_FIDELITY_AND_COST_GATE",
        "pilot_images": 8, "models": ["deit_s16", "vit_b16"],
        "task_count": 16, "split": "calibration", "split_order_prefix": [0, 8],
        "evaluation_access": False, "replacement_allowed": False,
        "target": "ground_truth_raw_logit",
        "references": ["normalized_zero_primary", "raw_rgb_black_sensitivity"],
        "paths": ["straight_primary", "quadrants_0_and_3_then_1_and_2_sensitivity"],
        "blocks": [0, 5, 11], "primary_intervention_block": 5,
        "primary_cut": "HEADS_WITH_BYPASS_FULL",
        "integration": "Gauss-Legendre", "integration_nodes": 512,
        "ih_nodes": 16, "forward_dtype": "float32", "reduction_dtype": "float64",
        "omega_scientific_score_dtype": "float32",
        "omega_zero_sum_gate_source": "independent_float64_complete_cut",
        "omega_zero_sum_audit_dtype": "float64",
        "omega_zero_sum_evaluation_components": ["identity", "attention"],
        "omega_zero_sum_declared_equivalence": "identity + sum(heads) == identity + attention",
        "closure": {"atol": 1e-5, "rtol": 1e-4, "near_zero": 1e-6},
        "representation": {
            "quantities": ["D", "Q_as_T_endpoint", "R", "Omega_directional"],
            "summaries": ["signed_mass", "absolute_mass", "cancellation_rate",
                          "routing_concentration_hhi", "worst_closure_sample"],
            "families": ["residual", "attention_branch", "query", "key", "value",
                         "head", "projection", "mlp"],
        },
        "contribution": {
            "endpoint_minus_ig_identity": True,
            "input_group_count": 4,
            "methods": ["endpoint", "ig", "conductance_pcia", "integrated_hessians",
                        "pefa_T", "pefa_R", "random", "input_norm"],
            "reference_conditions_reported_separately": True,
        },
        "interventions": {
            "input": "each frozen quadrant replaced by normalized-zero baseline",
            "port": "each head slice zeroed immediately before attention projection",
            "effects": ["endpoint_change_l1", "target_logit_abs_change",
                        "routing_total_variation"],
            "ranking_statistics": "within-image Spearman over all four groups or all heads",
            "correlation_is_not_causal_effect": True,
        },
        "stability": {
            "target": "clean_top1", "reference": "raw_rgb_black",
            "path": "registered_piecewise_linear", "model_randomization": "classifier_only_matched_scale",
            "small_perturbation": "iid_rgb_Gaussian_then_clip", "noise_sigma_rgb": 0.01,
        },
        "statistics": {
            "unit": "image", "aggregate_before_interval": True,
            "interval": "exact_enumeration_percentile_bootstrap_10000_seeded_if_n_ge_2",
            "minimum_valid_fraction": 1.0,
            "utility_gate": "PEFA_R mean within-image Spearman must exceed max(endpoint,IG,conductance-PCIA,Integrated-Hessians,random,input-norm) by 0.05 in both models for at least one preregistered causal outcome; descriptive only at n=8",
            "intervention_prediction_gate": "same criterion on a true input or head intervention outcome",
            "no_metric_switching_or_sample_removal": True,
        },
        "seed_namespace": "ENDPOINT_PCIA_PEFA|step5",
        "rng_derivation": {
            "task_seed": "uint64_big_endian(SHA256(seed_namespace + '|' + task_id)[:8])",
            "classifier_randomization_seed": "task_seed",
            "noise_seed": "task_seed",
            "omega_direction_seed": "task_seed + block_index",
            "random_baseline_namespace": "decimal_task_seed + '|input' or '|head'",
        },
        "claim_rules": {
            "states": ["SUPPORTED", "NOT_SUPPORTED", "INVALID"],
            "mathematical_closure": "SUPPORTED iff every task closure passes",
            "negative_controls": "SUPPORTED iff every task detects every frozen control",
            "cost": "SUPPORTED iff all caps pass",
            "incremental_value": "SUPPORTED iff utility_gate holds; else NOT_SUPPORTED when valid",
            "intervention_prediction": "SUPPORTED iff intervention_prediction_gate holds; else NOT_SUPPORTED when valid",
            "promotion": "all five claims SUPPORTED",
            "fallback": "NOT_SUPPORTED_EMPIRICAL_VALUE if valid closure but either empirical claim is not supported",
        },
        "negative_controls": ["omit_residual_identity", "duplicate_residual_identity",
                              "omit_attention_head", "swap_residual_input_slots",
                              "duplicate_serial_mlp_cuts", "detach_second_order_omega"],
        "resources": {
            "h20_hours_cap": 64, "wall_hours_cap": 72, "retained_bytes_cap": 128 * 1024**3,
            "gpu_memory_cap_bytes": 40 * 1024**3, "cpu_rss_cap_bytes": 32 * 1024**3,
            "task_timeout_seconds": 1800, "max_attempts": 2,
            "minimum_free_disk_bytes": 50 * 1024**3, "lease_seconds": 1800,
            "heartbeat_seconds": 25, "minimum_free_gpu_mib": 32768,
            "maximum_gpu_utilization_percent": 10, "maximum_gpus": 4,
            "dynamic_global_queue": True, "static_modulo_sharding": False,
        },
        "model_registry_path": "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
        "failure_codes": ["FAIL_SOURCE_BINDING", "INVALID_OUTCOME_BLINDNESS",
                          "FAIL_QUADRATURE_CLOSURE", "FAIL_NEGATIVE_CONTROL",
                          "FAIL_NONFINITE", "PAUSED_RESOURCE_CAP",
                          "NOT_SUPPORTED_EMPIRICAL_VALUE"],
        "step6_authorized": False,
    })
    task_manifest = signed({
        "schema_version": 1, "run_id": protocol["run_id"],
        "task_count": len(pilot_tasks), "tasks": pilot_tasks,
        "evaluation_access": False, "pilot_outcome_access_before_freeze": True,
        "outcome_access_during_execution": False,
    })
    dirty_status = subprocess.check_output(
        ["git", "status", "--porcelain=v2", "--untracked-files=all"], cwd=REPO, text=True,
    )
    binding = signed({
        "schema_version": 1, "run_id": protocol["run_id"],
        "status": "FROZEN_PROSPECTIVE",
        "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "producer_branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True).strip(),
        "producer_dirty": bool(dirty_status),
        "producer_dirty_status": dirty_status.splitlines(),
        "producer_dirty_status_sha256": hashlib.sha256(dirty_status.encode()).hexdigest(),
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "files": [identity(path) for path in source_paths],
        "checkpoints": checkpoints, "pilot_images": sorted(images.values(), key=lambda row: row["path"]),
        "forbidden_evaluation_access_count": 0, "pilot_outcome_access_before_freeze": True,
        "outcome_access_during_execution": False,
        "preaccess_tests": tests.stdout.strip(),
    })
    PILOT.mkdir(parents=True)
    write_new(PILOT / "protocol.json", protocol)
    write_new(PILOT / "task_manifest.json", task_manifest)
    write_new(PILOT / "source_binding.json", binding)
    write_new(PILOT / "freeze_receipt.json", signed({
        "schema_version": 1, "run_id": protocol["run_id"], "status": "FROZEN",
        "frozen_unix_ns": time.time_ns(), "protocol_digest": protocol["canonical_digest"],
        "task_manifest_digest": task_manifest["canonical_digest"],
        "source_binding_digest": binding["canonical_digest"],
        "pilot_outcome_access_before_freeze": True,
        "prior_outcome_known": True, "recovery_trigger_outcome_independent": True,
        "outcome_access_during_execution": False, "step6_started": False,
    }))
    print(json.dumps({
        "status": "FROZEN", "root": PILOT.relative_to(REPO).as_posix(),
        "task_count": len(pilot_tasks), "image_count": len(images),
        "protocol_digest": protocol["canonical_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
