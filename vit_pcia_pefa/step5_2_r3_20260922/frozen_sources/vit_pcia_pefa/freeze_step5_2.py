#!/usr/bin/env python3
"""Freeze Step 5.2 before any intervention outcome exists."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import identity, sha256, verify, write_new  # noqa: E402

RUN_ID = os.environ.get("STEP5_2_RUN_ID", "step5_2_r3_20260922")
if RUN_ID != "step5_2_r3_20260922":
    raise RuntimeError("STEP5_2_RUN_ID")
RUN = ROOT / RUN_ID
EP0 = REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918"
PYTHON = REPO / ".conda/envs/vitnns/bin/python3.10"


def main() -> int:
    if RUN.exists():
        raise RuntimeError("STEP5_2_ROOT_EXISTS")
    test_command = [str(PYTHON), "-m", "pytest", "-q",
                    "vit_pcia_pefa/tests/test_step4_vit_adapter.py",
                    "vit_pcia_pefa/tests/test_step5_1_localization.py",
                    "vit_pcia_pefa/tests/test_step5_2_prediction.py",
                    "vit_pcia_pefa/tests/test_step5_2_queue.py"]
    tests = subprocess.run(test_command, cwd=REPO, text=True, capture_output=True, check=True)
    registry = json.loads((EP0 / "model_registry.json").read_text())
    cohort = json.loads((EP0 / "cohort_registry.json").read_text())
    models = {row["model_id"]: row for row in registry["models"]}
    if set(models) != {"vit_b16", "deit_s16"}:
        raise RuntimeError("MODEL_REGISTRY_DRIFT")
    selected_models = ("deit_s16",)
    rows = {(row["split"], int(row["split_order"])): row for row in cohort["rows"]}
    discovery = rows[("discovery", 0)]
    discovery_path = ROOT / "step4_runtime/discovery" / discovery["sample_id"]
    if not discovery_path.is_file():
        raise RuntimeError("P0_DISCOVERY_ASSET_MISSING")
    checkpoints = []
    for model_id in selected_models:
        model = models[model_id]
        path = REPO / "vitnns_rebuild" / model["checkpoint"]["path"]
        if path.stat().st_size != model["checkpoint"]["bytes"] or sha256(path) != model["checkpoint"]["sha256"]:
            raise RuntimeError("CHECKPOINT_IDENTITY:" + model_id)
        checkpoints.append({"model_id": model_id, **identity(path, REPO, "checkpoint_stream_hash_no_deserialization")})
    images = {discovery["sample_id"]: identity(discovery_path, REPO, "P0_development_asset")}
    p1_tasks = []
    for order in range(16, 24):
        row = rows[("calibration", order)]
        path = REPO / "vitnns_rebuild/artifacts/nns_vit_endpoint_ep2_r1_20260918/calibration_images" / row["sample_id"]
        if not path.is_file():
            raise RuntimeError("P1_CALIBRATION_ASSET_MISSING:" + row["sample_id"])
        image = identity(path, REPO, "frozen_calibration_asset_no_decode_during_freeze"); images[row["sample_id"]] = image
        for model_id in selected_models:
            heads = int(models[model_id]["heads"]); task_id = f"{model_id}__{order:02d}"
            p1_tasks.append({"task_id": task_id, "model_id": model_id, "sample_id": row["sample_id"],
                "class_id": int(row["class_id"]), "split": "calibration", "split_order": order,
                "image_path": image["path"], "image_bytes": image["bytes"], "image_sha256": image["sha256"],
                "block_index": 5, "head_count": heads, "top_k": int(math.ceil(0.25 * heads))})
    coarse_nodes, coarse_weights = np.polynomial.legendre.leggauss(1024)
    fine_nodes, fine_weights = np.polynomial.legendre.leggauss(2048)
    coarse_nodes = np.ascontiguousarray((coarse_nodes + 1) / 2, dtype="<f8")
    coarse_weights = np.ascontiguousarray(coarse_weights / 2, dtype="<f8")
    fine_nodes = np.ascontiguousarray((fine_nodes + 1) / 2, dtype="<f8")
    fine_weights = np.ascontiguousarray(fine_weights / 2, dtype="<f8")
    prior = json.loads((ROOT / "step5_1_r7_20260921/resource_ledger.json").read_text())
    r1 = verify(ROOT / "step5_2_r1_20260922/p0_decision.json")
    r1_failure = verify(ROOT / "step5_2_r1_20260922/failure_receipt.json")
    if (r1["status"] != "INVALID_NUMERICAL_OR_INTERVENTION"
            or r1_failure["failure_code"] != "FAIL_QUADRATURE_CONVERGENCE"):
        raise RuntimeError("R1_PREDECESSOR_STATUS")
    diagnostic_paths = sorted(ROOT.glob("step5_2_r1_quadrature_diagnosis_*.json"))
    if len(diagnostic_paths) != 4:
        raise RuntimeError("R1_DIAGNOSTIC_COVERAGE")
    diagnostics = [verify(path) for path in diagnostic_paths]
    if (any(row["status"] != "P0_DISCOVERY_ONLY_DIAGNOSIS" or row["calibration_outcomes_read"]
            or row["evaluation_access"] for row in diagnostics)
            or {(row["model_id"], tuple(row["nodes"])) for row in diagnostics}
            != {(model, nodes) for model in ("deit_s16", "vit_b16")
                for nodes in ((512, 1024), (1024, 2048))}):
        raise RuntimeError("R1_DIAGNOSTIC_CONTRACT")
    diagnostic_seconds = math.fsum(float(row["resources"]["wall_seconds"]) for row in diagnostics)
    prior_step5_2_hours = float(r1["p0_h20_hours"]) + diagnostic_seconds / 3600.0
    r2_root = ROOT / "step5_2_r2_20260922"
    r2_failure = verify(r2_root / "failure_receipt.json")
    r2_protocol = verify(r2_root / "protocol.json")
    r2_p0 = verify(r2_root / "p0_decision.json")
    if (r2_failure["failure_code"] != "FAIL_PREDICTION_WORKER_TASK_ID_PLUMBING"
            or r2_failure["accepted_prediction_count"] != 0
            or r2_failure["intervention_started"]
            or r2_p0["status"] != "PASS_P0_NUMERICAL_FEASIBILITY"):
        raise RuntimeError("R2_REBIND_ELIGIBILITY")
    r2_binding = verify(r2_root / "source_binding.json")
    r2_bound_files = {row["path"]: row for row in r2_binding["files"]}
    p0_source_paths = (
        "vit_pcia_pefa/src/step5_2_io.py", "vit_pcia_pefa/src/step5_2_prediction.py",
        "vit_pcia_pefa/src/step5_1_localization.py", "vit_pcia_pefa/src/pilot.py",
        "vit_pcia_pefa/src/vit_adapter.py", "vit_pcia_pefa/run_step5_2_p0_fixture.py",
        "vit_pcia_pefa/run_step5_2_p0_worker.py", "vit_pcia_pefa/launch_step5_2_p0.py",
        "vit_pcia_pefa/finalize_step5_2_p0.py",
    )
    if any(path not in r2_bound_files or sha256(REPO / path) != r2_bound_files[path]["sha256"]
           for path in p0_source_paths):
        raise RuntimeError("R2_P0_SOURCE_DRIFT")
    r2_failed_hours = float(r2_failure["failed_attempt_gpu_seconds"]) / 3600.0
    prior_step5_2_hours += float(r2_p0["p0_h20_hours"]) + r2_failed_hours
    protocol = write_new(RUN / "protocol.json", {
        "schema_version": 1, "run_id": RUN_ID, "stage": "ENDPOINT_PCIA_PEFA_STEP5_2",
        "status": "FROZEN_PROSPECTIVE",
        "authorization": "USER_AUTHORIZED_DEIT_S16_ONLY_SCOPE_20260922",
        "amendments": ["step5_2_r2_amendment.md", "step5_2_r3_amendment.md"],
        "predecessor_runs": [
            {"run_id": "step5_2_r1_20260922", "status": "INVALID_NUMERICAL_OR_INTERVENTION",
             "failure_code": "FAIL_QUADRATURE_CONVERGENCE", "scientific_decision": "NONE_NUMERICALLY_INVALID"},
            {"run_id": "step5_2_r2_20260922", "status": "INVALID_INFRASTRUCTURE",
             "failure_code": "FAIL_PREDICTION_WORKER_TASK_ID_PLUMBING",
             "scientific_decision": "NONE_INFRASTRUCTURE_INVALID"}],
        "prior_results_immutable": {"step5": "NOT_SUPPORTED_EMPIRICAL_VALUE",
            "step5_1": "NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES"},
        "evaluation_access": False, "step6_authorized": False, "holdout_authorized": False,
        "models": list(selected_models), "model_scope": "DEIT_S16_ONLY",
        "cross_architecture_replication_claim": False, "target": "ground_truth_raw_logit",
        "reference": "normalized_zero", "path": "straight", "cut_id": "HEADS_WITH_BYPASS_FULL",
        "real_block_index": 5, "lambdas": [1.0, 0.5, 0.0],
        "head_mapping": "complete normalized residual input to one pre-projection attention-head output including Q,K,V,softmax,AV",
        "integration": {"rule": "Gauss-Legendre", "fixture_coarse_nodes": 32, "fixture_fine_nodes": 64,
            "real_coarse_nodes": 1024, "real_fine_nodes": 2048,
            "coarse_nodes_sha256": hashlib.sha256(coarse_nodes.tobytes()).hexdigest(),
            "coarse_weights_sha256": hashlib.sha256(coarse_weights.tobytes()).hexdigest(),
            "fine_nodes_sha256": hashlib.sha256(fine_nodes.tobytes()).hexdigest(),
            "fine_weights_sha256": hashlib.sha256(fine_weights.tobytes()).hexdigest(),
            "adaptive_per_sample_nodes": False},
        "numeric": {
            "fixture": {"dtype": "float64", "atol": 1e-9, "rtol": 1e-7, "near_zero": 1e-12,
                        "convergence_atol": 1e-7, "convergence_rtol": 1e-4},
            "real": {"forward_dtype": "float32", "reduction_dtype": "float64",
                     "atol": 1e-5, "rtol": 1e-4, "near_zero": 1e-6,
                     "convergence_atol": 1e-4, "convergence_rtol": 1e-2,
                     "convergence_rule": "global_relative_l2_or_absolute_near_zero",
                     "elementwise_convergence_is_diagnostic_only": True,
                     "tf32": False, "autocast": False},
            "outcome_epsilon": 1e-12, "dose_absolute_tolerance": 1e-8},
        "p0": {"scientific_claim": False, "fixture_block": 1, "development_split": "discovery",
               "development_split_order": 0, "real_test_head": 0,
               "gate": "analytic and graph fixtures; DeiT-S/16 only; convergence, clean parity, closure, nonzero Y and dose order"},
        "p1": {"split": "calibration", "split_order_range": [16, 24], "image_count": 8,
               "image_task_count": 8, "candidate_scope": "all block-5 attention heads in DeiT-S/16",
               "top_k_rule": "ceil(0.25*num_heads)", "prediction_frozen_before_intervention": True,
               "methods": ["pefa_R", "pcia_A", "endpoint_Q", "gradient_norm", "activation_norm", "random"],
               "primary_outcome": "Y_e_lambda0", "metrics": ["spearman", "top1_hit", "top_k_effect_capture"],
               "statistical_unit": "image", "bootstrap_resamples": 20000, "bootstrap_confidence": 0.95,
               "bootstrap_seed_namespace": "ENDPOINT_PCIA_PEFA|step5_2|bootstrap",
               "undefined_spearman_for_gate": 0.0, "undefined_direction_for_summary": 0.0,
               "positive_prediction": {"mean_spearman_min": 0.20, "lower_ci_strictly_above": 0.0},
               "increment_over_strongest_baseline": {"paired_mean_spearman_min": 0.10, "lower_ci_strictly_above": 0.0},
               "simple_baseline_tie_order": ["pcia_A", "endpoint_Q", "gradient_norm", "activation_norm", "random"],
               "dose_order_rate_min": 0.90,
               "direction_supported_if": "DeiT-S/16 cosine lower CI > 0 and sign-agreement lower CI > 0.5"},
        "claims": ["NATURAL_GAP_PREDICTION", "PREDICTIVE_INCREMENT_OVER_SIMPLE_BASELINES",
                   "DOSE_ORDER", "DIRECTIONAL_ALIGNMENT", "NUMERICAL_VALIDITY_AND_COST"],
        "resources": {"step5_2_h20_hours_cap": 32, "cumulative_step5_h20_hours_cap": 64,
            "predecessor_h20_hours": float(prior["cumulative_step5_h20_hours_upper_bound"]),
            "prior_step5_2_h20_hours": prior_step5_2_hours,
            "prior_step5_2_breakdown": {"r1_p0_h20_hours": float(r1["p0_h20_hours"]),
                "r1_diagnostic_h20_hours": diagnostic_seconds / 3600.0,
                "r2_p0_h20_hours": float(r2_p0["p0_h20_hours"]),
                "r2_failed_prediction_attempt_h20_hours": r2_failed_hours},
            "maximum_gpus": 4, "minimum_free_gpu_mib": 32768, "maximum_gpu_utilization_percent": 10,
            "gpu_memory_cap_bytes": 51539607552, "cpu_rss_cap_bytes": 34359738368,
            "task_timeout_seconds": 14400, "max_attempts": 2, "retained_bytes_cap": 137438953472},
        "failure_codes": ["FAIL_SOURCE_BINDING", "INVALID_NUMERICAL_OR_INTERVENTION",
            "FAIL_QUADRATURE_CONVERGENCE", "FAIL_COMPLETE_CUT_CLOSURE", "FAIL_NONFINITE", "PAUSED_RESOURCE_CAP"]})
    for key in ("models", "target", "reference", "path", "cut_id", "real_block_index",
                "lambdas", "head_mapping", "integration", "numeric", "p0"):
        if protocol[key] != r2_protocol[key]:
            raise RuntimeError("R2_P0_PROTOCOL_DRIFT:" + key)
    p0_tasks = [{"task_id": f"p0__{model_id}", "model_id": model_id,
        "sample_id": discovery["sample_id"], "class_id": int(discovery["class_id"]),
        "image_path": images[discovery["sample_id"]]["path"],
        "image_bytes": images[discovery["sample_id"]]["bytes"],
        "image_sha256": images[discovery["sample_id"]]["sha256"], "block_index": 5, "head_index": 0}
        for model_id in selected_models]
    manifest = write_new(RUN / "task_manifest.json", {"schema_version": 1, "run_id": RUN_ID,
        "p0_tasks": p0_tasks, "prediction_tasks": p1_tasks, "intervention_tasks": p1_tasks,
        "evaluation_access": False})
    source_names = ["执行规划.md", "step5_2_protocol.md", "step5_2_r2_amendment.md",
        "step5_2_r3_amendment.md",
        "src/step5_2_io.py", "src/step5_2_prediction.py",
        "src/step5_2_queue.py", "tests/test_step5_2_prediction.py", "tests/test_step5_2_queue.py",
        "freeze_step5_2.py", "run_step5_2_p0_fixture.py", "run_step5_2_p0_worker.py",
        "launch_step5_2_p0.py", "finalize_step5_2_p0.py", "run_step5_2_prediction_worker.py",
        "run_step5_2_intervention_worker.py", "launch_step5_2.py", "freeze_step5_2_predictions.py",
        "finalize_step5_2.py", "validate_step5_2.py", "audit_step5_2_completion.py",
        "seal_step5_2_r1_failure.py", "seal_step5_2_r2_failure.py",
        "src/step5_1_localization.py", "src/pilot.py", "src/vit_adapter.py"]
    files = [identity(ROOT / name, REPO, "frozen_source") for name in source_names]
    for path in (EP0 / "model_registry.json", EP0 / "cohort_registry.json",
                 ROOT / "step5_1_completion_r2_20260921/terminal_receipt.json",
                 ROOT / "step5_1_r7_20260921/resource_ledger.json",
                 ROOT / "step5_2_r1_20260922/failure_receipt.json",
                 r2_root / "protocol.json", r2_root / "source_binding.json",
                 r2_root / "p0_fixture_result.json", r2_root / "p0/deit_s16.json",
                 r2_root / "p0_decision.json", r2_root / "failure_receipt.json",
                 *diagnostic_paths):
        files.append(identity(path, REPO, "frozen_registry_or_prior_evidence"))
    source = write_new(RUN / "source_binding.json", {"schema_version": 1, "run_id": RUN_ID,
        "status": "FROZEN_BEFORE_OUTCOMES", "files": files, "checkpoints": checkpoints,
        "images": list(images.values()),
        "calibration_selection_before_outcome": "split_order_16_through_23_deit_s16_only",
        "outcome_reads_before_freeze": 0, "r2_outcome_reads_before_freeze": 0,
        "r1_evidence_reads_allowed": True, "evaluation_member_reads": 0,
        "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "plan_worktree_sha256_at_freeze": sha256(ROOT / "执行规划.md"),
        "sys_executable_realpath": str(PYTHON.resolve()), "preaccess_tests": tests.stdout.strip()})
    receipt = write_new(RUN / "freeze_receipt.json", {"schema_version": 1, "run_id": RUN_ID,
        "status": "PASS_STEP5_2_PROSPECTIVE_FREEZE", "protocol_digest": protocol["canonical_digest"],
        "task_manifest_digest": manifest["canonical_digest"], "source_binding_digest": source["canonical_digest"],
        "frozen_unix_ns": time.time_ns(), "prediction_started": False, "intervention_started": False,
        "evaluation_access": False, "step6_started": False})
    p0_rebind = write_new(RUN / "p0_rebind_receipt.json", {
        "schema_version": 1, "run_id": RUN_ID, "status": "PASS_P0_REBOUND_FROM_R2",
        "source_run_id": r2_protocol["run_id"], "source_p0_status": r2_p0["status"],
        "source_protocol_digest": r2_protocol["canonical_digest"],
        "source_p0_decision_digest": r2_p0["canonical_digest"],
        "scientific_claim": False, "prediction_authorized_by_gate": True,
        "p0_h20_hours": 0.0, "source_p0_h20_hours_already_in_prior_ledger": r2_p0["p0_h20_hours"],
        "verified_p0_source_paths": list(p0_source_paths),
        "changed_code_outside_p0_scope": ["run_step5_2_prediction_worker.py",
            "run_step5_2_intervention_worker.py", "tests/test_step5_2_queue.py", "freeze_step5_2.py"],
        "evaluation_access": False, "step6_started": False, "rebound_unix_ns": time.time_ns()})
    print(json.dumps({"status": receipt["status"], "run": RUN_ID, "p1_tasks": len(p1_tasks),
                      "protocol_digest": protocol["canonical_digest"],
                      "p0_rebind_digest": p0_rebind["canonical_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
