#!/usr/bin/env python3
"""Freeze a prospective Step-5 mechanism-extension root."""
from __future__ import annotations

import argparse
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
EP0 = REPO / "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918"
R8 = ROOT / "step5_pilot_r8_20260921"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value); value["canonical_digest"] = digest(value); return value


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("E1_E2", "E3A"), required=True)
    parser.add_argument("--recovery-r2", action="store_true")
    args = parser.parse_args()
    if args.recovery_r2 and args.phase != "E1_E2":
        raise RuntimeError("RECOVERY_PHASE")
    run_id = ("step5_extension_r2_20260921" if args.recovery_r2 else
              "step5_extension_r1_20260921" if args.phase == "E1_E2" else
              "step5_extension_r3_20260921")
    output = ROOT / run_id
    if output.exists():
        raise RuntimeError("EXTENSION_ROOT_EXISTS")
    if shutil.disk_usage(REPO).free < 50 * 1024**3:
        raise RuntimeError("PAUSED_RESOURCE_CAP:DISK")
    terminal = json.loads((R8 / "terminal_receipt.json").read_text())
    validation = json.loads((R8 / "validation_report.json").read_text())
    if terminal["status"] != "NOT_SUPPORTED_EMPIRICAL_VALUE" or terminal["step6_started"]:
        raise RuntimeError("R8_TERMINAL_BOUNDARY")
    validation_body = dict(validation); validation_digest = validation_body.pop("canonical_digest", None)
    if (validation_digest != digest(validation_body)
            or terminal.get("validation_report_digest") != validation_digest
            or validation["status"] != terminal["status"]):
        raise RuntimeError("R8_VALIDATION")
    extension_predecessor = None
    predecessor_root = None
    if args.recovery_r2:
        predecessor_root = ROOT / "step5_extension_r1_20260921"
        extension_predecessor = json.loads((predecessor_root / "terminal_receipt.json").read_text())
        predecessor_validation = json.loads((predecessor_root / "validation_report.json").read_text())
        if extension_predecessor["phase"] != "E1_E2" or extension_predecessor["status"] != "INVALID":
            raise RuntimeError("E1_E2_R1_NOT_INVALID")
        if predecessor_validation["status"] != "PASS_STEP5_EXTENSION_AUDIT":
            raise RuntimeError("E1_E2_R1_VALIDATION")
    elif args.phase == "E3A":
        predecessor_root = ROOT / "step5_extension_r2_20260921"
        extension_predecessor = json.loads((predecessor_root / "terminal_receipt.json").read_text())
        predecessor_validation = json.loads((predecessor_root / "validation_report.json").read_text())
        if extension_predecessor["phase"] != "E1_E2" or extension_predecessor["status"] == "INVALID":
            raise RuntimeError("E1_E2_NOT_VALID")
        if predecessor_validation["status"] != "PASS_STEP5_EXTENSION_AUDIT":
            raise RuntimeError("E1_E2_VALIDATION")
    tests = subprocess.run(
        [str(REPO / ".conda/envs/vitnns/bin/python"), "-m", "pytest", "-q",
         "vit_pcia_pefa/tests/test_step5_extension.py",
         "vit_pcia_pefa/tests/test_step5_queue.py"], cwd=REPO,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
             "PYTHONPATH": ".:vitnns_rebuild/src"}, capture_output=True, text=True,
    )
    if tests.returncode != 0:
        raise RuntimeError("EXTENSION_PREACCESS_TEST_FAIL:" + tests.stdout + tests.stderr)

    r8_manifest = json.loads((R8 / "task_manifest.json").read_text())
    tasks, images = [], {}
    for row in r8_manifest["tasks"]:
        payload = dict(row["payload"])
        if args.phase == "E3A" and int(payload["split_order"]) >= 2:
            continue
        payload["phase"] = args.phase
        task_id = f"{args.phase.lower()}__{payload['model_id']}__{int(payload['split_order']):02d}"
        tasks.append({"task_id": task_id, "payload": payload})
        image_path = REPO / payload["image_path"]
        if image_path.stat().st_size != payload["image_bytes"] or sha256(image_path) != payload["image_sha256"]:
            raise RuntimeError("CALIBRATION_IMAGE_DRIFT:" + payload["sample_id"])
        images[payload["image_path"]] = {
            "path": payload["image_path"], "bytes": payload["image_bytes"],
            "sha256": payload["image_sha256"],
        }

    source_paths = [
        ROOT / "执行规划.md", ROOT / "method_contract.md",
        ROOT / "step5_extension_protocol.md", ROOT / "src/vit_adapter.py",
        ROOT / "src/pilot.py", ROOT / "src/step5_extension.py",
        ROOT / "src/step5_extension_queue.py", ROOT / "run_step5_extension_worker.py",
        ROOT / "freeze_step5_extension.py", ROOT / "launch_step5_extension.py",
        ROOT / "finalize_step5_extension.py", ROOT / "validate_step5_extension.py",
        ROOT / "tests/test_step5_extension.py", ROOT / "tests/test_step5_queue.py",
        R8 / "terminal_receipt.json", R8 / "validation_report.json",
        R8 / "task_manifest.json", R8 / "resource_ledger.json",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_nns.py",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_vit.py",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_ep1_queue.py",
        REPO / "vitnns_rebuild/src/vitnns/endpoint_ep2.py",
        EP0 / "model_registry.json",
    ]
    if predecessor_root is not None:
        source_paths.extend([predecessor_root / "terminal_receipt.json",
                             predecessor_root / "resource_ledger.json",
                             predecessor_root / "validation_report.json"])
    diagnosis_hours = 0.0
    if args.recovery_r2:
        diagnosis_paths = [ROOT / "step5_extension_r1_quadrature_diagnosis.json",
                           ROOT / "step5_extension_r1_quadrature_diagnosis_high_nodes.json"]
        source_paths.extend([ROOT / "step5_extension_r2_recovery_amendment.md",
                             ROOT / "diagnose_step5_extension_r1.py",
                             ROOT / "diagnose_step5_extension_r1_high_nodes.py", *diagnosis_paths])
        diagnosis_hours = sum(json.loads(path.read_text())["resources"]["h20_hours"]
                              for path in diagnosis_paths)
    predecessor_ledger_path = ((predecessor_root or R8) / "resource_ledger.json")
    predecessor_ledger = json.loads(predecessor_ledger_path.read_text())
    predecessor_hours = predecessor_ledger.get("cumulative_step5_h20_hours",
                                                predecessor_ledger.get("aggregate_h20_hours"))
    registry = json.loads((EP0 / "model_registry.json").read_text())
    checkpoints = []
    for row in registry["models"]:
        path = REPO / "vitnns_rebuild" / row["checkpoint"]["path"]
        observed = identity(path); observed["model_id"] = row["model_id"]
        if observed["bytes"] != row["checkpoint"]["bytes"] or observed["sha256"] != row["checkpoint"]["sha256"]:
            raise RuntimeError("CHECKPOINT_DRIFT:" + row["model_id"])
        checkpoints.append(observed)

    from vit_pcia_pefa.src.step5_extension_queue import SEED_NAMESPACE
    protocol = signed({
        "schema_version": 1, "stage": "ENDPOINT_PCIA_PEFA_STEP5_EXTENSION",
        "run_id": run_id, "phase": args.phase, "status": "FROZEN_PROSPECTIVE",
        "authorization": "execute Step-5 mechanism extension only",
        "predecessor_run_id": (extension_predecessor["run_id"] if extension_predecessor else "step5_pilot_r8_20260921"),
        "predecessor_status": (extension_predecessor["status"] if extension_predecessor else terminal["status"]),
        "original_pilot_run_id": "step5_pilot_r8_20260921",
        "original_pilot_status": terminal["status"],
        "prior_outcome_known": True, "new_hypothesis_after_original_pilot": True,
        "original_terminal_immutable": True, "confirmatory_claim_allowed": False,
        "outcome_access_before_freeze": bool(args.recovery_r2),
        "outcome_access_during_execution": False,
        "recovery_trigger_outcome_independent": bool(args.recovery_r2),
        "recovery_scope": ("increase edge quadrature from 8 to 128 nodes after conservative-field validity failure"
                           if args.recovery_r2 else None),
        "predecessor_h20_hours_including_diagnosis": predecessor_hours + diagnosis_hours,
        "split": "calibration", "split_orders": list(range(8)) if args.phase == "E1_E2" else [0, 1],
        "evaluation_access": False, "models": ["deit_s16", "vit_b16"],
        "target": "ground_truth_raw_logit", "block": 5, "forward_dtype": "float32",
        "reduction_dtype": "float64", "seed_namespace": SEED_NAMESPACE,
        "loop": {
            "object": "oriented closed port-field line integral; never same-endpoint Endpoint difference",
            "edge_nodes": 128 if args.recovery_r2 else 8,
            "epsilon_l2": [0.25, 0.5, 1.0], "primary_epsilon_l2": 0.5,
            "direction_generation": "two deterministic Gram-Schmidt orthonormal Gaussian directions",
            "metrics": ["absolute_rank_spearman", "signed_cosine", "sign_agreement",
                        "relative_slope_error", "circulation_scaling_exponent",
                        "rank_advantage_over_random"],
            "thresholds": {"absolute_rank_spearman": 0.50, "signed_cosine": 0.80,
                           "sign_agreement": 0.75, "relative_slope_error_max": 0.50,
                           "scaling_exponent_min": 1.5, "scaling_exponent_max": 2.5,
                           "rank_advantage_over_random": 0.10},
            "validity": {"identical_endpoint_max_abs": 1e-6,
                         "complete_cut_vs_full_abs_error": 1e-4,
                         "full_gradient_circulation_abs": 1e-4},
        },
        "nonlinearity": {
            "condition": "one-step ground-truth-versus-clean-best-other margin-gradient sign",
            "boundary_epsilons_rgb": [1/255, 2/255, 4/255, 8/255],
            "thresholds": {"margin_reduced_or_crossed_fraction": 0.75,
                           "boundary_to_clean_curl_norm_ratio": 1.25,
                           "boundary_to_clean_circulation_density_ratio": 1.25,
                           "boundary_signed_cosine": 0.50,
                           "low_margin_exceeds_high_margin": True},
        },
        "source_resolution": {
            "grids": [4, 7], "nodes": 16, "images_per_model": 2,
            "primary_effect": "endpoint_change_l1",
            "secondary_effect": "target_logit_abs_change",
            "method": "head-port source-resolved PEFA-R absolute mass",
            "comparators": ["endpoint", "input_norm", "conductance_pcia", "random"],
            "advance_to_14x14": {"minimum_baseline_advantage": 0.05,
                                  "minimum_7x7_over_4x4": 0.05,
                                  "every_image_improves": True},
        },
        "claim_states": ["SUPPORTED", "NOT_SUPPORTED", "INVALID"],
        "resources": {
            "extension_h20_hours_cap": 24, "cumulative_step5_h20_hours_cap": 64,
            "wall_hours_cap": 72, "retained_bytes_cap": 128 * 1024**3,
            "gpu_memory_cap_bytes": 40 * 1024**3, "cpu_rss_cap_bytes": 32 * 1024**3,
            "task_timeout_seconds": 3600, "max_attempts": 2,
            "minimum_free_disk_bytes": 50 * 1024**3, "lease_seconds": 3600,
            "heartbeat_seconds": 25, "minimum_free_gpu_mib": 45000,
            "maximum_gpu_utilization_percent": 10, "maximum_gpus": 4,
            "dynamic_global_queue": True, "static_modulo_sharding": False,
        },
        "model_registry_path": "vitnns_rebuild/manifests/nns_vit_endpoint/ep0_r1_20260918/model_registry.json",
        "task_count": len(tasks), "step6_authorized": False,
    })
    manifest = signed({
        "schema_version": 1, "run_id": run_id, "phase": args.phase,
        "task_count": len(tasks), "tasks": tasks, "evaluation_access": False,
        "outcome_access_before_freeze": False, "outcome_access_during_execution": False,
    })
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v2", "--untracked-files=all"], cwd=REPO, text=True,
    )
    binding = signed({
        "schema_version": 1, "run_id": run_id, "status": "FROZEN_PROSPECTIVE",
        "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "producer_branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True).strip(),
        "producer_dirty": bool(dirty), "producer_dirty_status": dirty.splitlines(),
        "producer_dirty_status_sha256": hashlib.sha256(dirty.encode()).hexdigest(),
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "files": [identity(path) for path in source_paths], "checkpoints": checkpoints,
        "pilot_images": sorted(images.values(), key=lambda row: row["path"]),
        "forbidden_evaluation_access_count": 0,
        "outcome_access_before_freeze": bool(args.recovery_r2),
        "outcome_access_during_execution": False, "preaccess_tests": tests.stdout.strip(),
    })
    output.mkdir(parents=True)
    write_new(output / "protocol.json", protocol)
    write_new(output / "task_manifest.json", manifest)
    write_new(output / "source_binding.json", binding)
    write_new(output / "freeze_receipt.json", signed({
        "schema_version": 1, "run_id": run_id, "phase": args.phase, "status": "FROZEN",
        "frozen_unix_ns": time.time_ns(), "protocol_digest": protocol["canonical_digest"],
        "task_manifest_digest": manifest["canonical_digest"],
        "source_binding_digest": binding["canonical_digest"],
        "prior_original_pilot_outcome_known": True,
        "extension_outcome_access_before_freeze": bool(args.recovery_r2),
        "recovery_trigger_outcome_independent": bool(args.recovery_r2),
        "step6_started": False,
    }))
    print(json.dumps({"status": "FROZEN", "root": output.relative_to(REPO).as_posix(),
                      "phase": args.phase, "task_count": len(tasks),
                      "protocol_digest": protocol["canonical_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
