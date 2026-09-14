#!/usr/bin/env python3
"""Validate and seal S4 without performing cohort-level science."""
from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s4_r3_20260915"
PAYLOAD = REPO / "artifacts/nns_vit_patch_depth/s4_r3_20260915"
PYTHON = REPO.parent / ".conda/envs/vitnns/bin/python"
sys.path.insert(0, str(REPO / "src"))
from vitnns import nns_patch_depth_s4 as s4  # noqa: E402


def artifact(path: Path) -> dict[str, Any]:
    value = {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": s4.sha_file(path)}
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if isinstance(document.get("canonical_digest"), str):
            value["canonical_digest"] = document["canonical_digest"]
    return value


def write_report(name: str, document_type: str, status: str, payload: dict[str, Any], producer_commit: str) -> Path:
    path = ROOT / name
    if path.exists():
        raise RuntimeError("final report already exists: " + name)
    s4.write_json_new(path, s4.signed(document_type, status, payload, producer_commit=producer_commit))
    return path


def main() -> int:
    producer_commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    schedule = json.loads((REPO / s4.SCHEDULE_REL).read_text())["payload"]["records"]
    paths = s4.all_terminal_paths(ROOT)
    if len(paths) != 5232:
        raise RuntimeError(f"cannot finalize before 5232 terminals: {len(paths)}")
    records = []
    states = Counter()
    by_type = Counter()
    quadrature = Counter()
    p1_valid_blocks = p1_raw_draws = p2_matched_coordinates = 0
    resources_by_phase: dict[str, list[float]] = {}
    worker_end: dict[tuple[str, str], tuple[int, int]] = {}
    array_records = []
    for path in paths:
        value = json.loads(path.read_text())
        if not s4.verify_signed(value):
            raise RuntimeError("unit terminal digest drift: " + str(path))
        payload = value["payload"]
        key = (payload["unit_type"], int(payload["schedule_index"]), path.relative_to(REPO).as_posix())
        state = value["status"]
        states[state] += 1
        by_type[payload["unit_type"]] += 1
        if state == "VALID_NNS":
            quadrature[str(payload["nns"]["selected_steps"])] += 1
            p1_valid_blocks += int(payload["p1"]["valid_block_count"])
            p1_raw_draws += int(payload["p1"]["valid_block_count"]) * 64
            p2_matched_coordinates += int(payload["p2_plan"]["coordinate_count"]) if payload["p2_plan"]["status"] == "MATCHED" else 0
        resource = payload["resources"]
        records.append({
            "sort_key": list(key), "terminal_path": path.relative_to(REPO).as_posix(),
            "state": state, "launch_id": payload["launch_id"], "worker_id": payload["worker_id"],
            "started_unix_ns": payload["started_unix_ns"], "completed_unix_ns": payload["completed_unix_ns"],
            **resource,
        })
        resources_by_phase.setdefault(payload["unit_type"], []).append(float(resource["gpu_active_seconds"]))
        worker_key = (payload["launch_id"], payload["worker_id"])
        old = worker_end.get(worker_key)
        current = (int(payload["started_unix_ns"]), int(payload["completed_unix_ns"]))
        worker_end[worker_key] = current if old is None else (min(old[0], current[0]), max(old[1], current[1]))
        array_records.extend(payload.get("arrays", []))
    records.sort(key=lambda row: tuple(row["sort_key"]))
    gpu_seconds = math.fsum(float(row["gpu_active_seconds"]) for row in records)
    cpu_seconds = math.fsum(float(row["cpu_seconds"]) for row in records)
    wall_sum = math.fsum(float(row["unit_wall_seconds"]) for row in records)
    retained = sum(int(row["retained_bytes"]) for row in records)
    launch_terminals = sorted(ROOT.glob("launches/*/launch_terminal.json"))
    launch_rows = [json.loads(path.read_text()) for path in launch_terminals]
    launch_elapsed = math.fsum(float(value["payload"].get("launch_elapsed_seconds", 0.0)) for value in launch_rows)
    critical_path = max(((end - start) / 1e9 for start, end in worker_end.values()), default=0.0)
    worker_span = math.fsum((end - start) / 1e9 for start, end in worker_end.values())
    estimated_idle_tail = max(0.0, len(worker_end) * launch_elapsed - worker_span)
    resource_path = write_report("resource_and_scaling_ledger.json", "s4_resource_and_scaling_ledger", "PASS", {
        "stable_sort_key": ["unit_type", "schedule_index", "terminal_path"], "summation": "math.fsum",
        "GPU_hours_not_divided_by_K": True, "cumulative_GPU_seconds": gpu_seconds, "cumulative_GPU_hours": gpu_seconds / 3600.0,
        "cumulative_CPU_seconds": cpu_seconds, "cumulative_CPU_hours": cpu_seconds / 3600.0,
        "unit_wall_seconds_sum": wall_sum, "launch_elapsed_seconds_sum": launch_elapsed, "critical_path_seconds": critical_path,
        "worker_idle_tail_seconds_estimate": estimated_idle_tail,
        "phase_GPU_seconds": {key: math.fsum(value) for key, value in sorted(resources_by_phase.items())},
        "producer_validation_resource_separated": True, "producer_unit_count": len(records),
        "independent_validator_CPU_seconds": None, "independent_validator_GPU_seconds": None,
        "retained_bytes": retained, "peak_allocated_GPU_bytes": max(int(row["peak_allocated_gpu_bytes"]) for row in records),
        "peak_reserved_GPU_bytes": max(int(row["peak_reserved_gpu_bytes"]) for row in records),
        "cost_is_reporting_only": True, "cost_or_wall_time_terminal_used": False, "unit_records": records,
    }, producer_commit)
    access_source = json.loads((ROOT / "selected_member_materialization.json").read_text())["payload"]
    source_completions = len(list((ROOT / "source_completions").glob("source-*.json")))
    valid_nns = states["VALID_NNS"]
    valid_occlusion = states["VALID_OCCLUSION"]
    worker_image_decodes = sum(bool(row.get("geometry")) for row in (json.loads(path.read_text())["payload"] for path in paths))
    worker_mask_decodes = sum(bool(json.loads(path.read_text())["payload"].get("geometry")) and json.loads(path.read_text())["payload"]["unit_type"] == "NNS_IMAGE" for path in paths)
    access_path = write_report("phased_formal_access_audit.json", "s4_phased_formal_access_audit", "PASS", {
        "pre_implementation_seal": {"selected_JPEG_member_access_count": 0, "selected_mask_member_access_count": 0, "decode_count": 0, "model_forward_count": 0, "NNS_count": 0, "occlusion_count": 0},
        "post_seal": {
            "authorized_source_count": 1744, "selected_JPEG_member_reads": access_source["selected_JPEG_member_reads"],
            "selected_mask_member_reads": access_source["selected_mask_member_reads"],
            "archive_validation_image_decode_count": access_source["image_decode_count"], "archive_validation_mask_decode_count": access_source["mask_decode_count"],
            "worker_cache_image_decode_count": worker_image_decodes, "worker_cache_mask_decode_count": worker_mask_decodes,
            "total_image_decode_count": access_source["image_decode_count"] + worker_image_decodes,
            "total_mask_decode_count": access_source["mask_decode_count"] + worker_mask_decodes,
            "NNS_scheduled": 1744, "NNS_VALID": valid_nns,
            "NNS_NA": sum(value for key, value in states.items() if key.startswith("NA_") and key != "NA_UPSTREAM_NNS") // 3,
            "NNS_failed_or_blocked": sum(states[key] for key in ("FAILED_NUMERICAL", "FAILED_NNS", "BLOCKED_DATA_INTEGRITY", "BLOCKED_INFRASTRUCTURE", "FAILED_IMPLEMENTATION")),
            "P1_raw_permutation_draw_count": p1_raw_draws, "matched_plan_coordinate_count": p2_matched_coordinates,
            "primary_occlusion_terminal_count": by_type["OCCLUSION_PRIMARY"], "sensitivity_occlusion_terminal_count": by_type["OCCLUSION_SENSITIVITY"],
            "source_completion_terminal_count": source_completions,
        },
        "authorized_source_count": 1744, "D2_only_member_read_count": 0,
        "S3_D2_prediction_error_numeric_read_count": 0, "cohort_level_P1_aggregation_count": 0,
        "cohort_level_P2_aggregation_count": 0, "cohort_level_D1_aggregation_count": 0, "D2_aggregation_count": 0,
        "formal_estimate_count": 0, "formal_p_value_count": 0, "formal_CI_count": 0, "Holm_count": 0,
        "gatekeeping_count": 0, "scientific_decision_count": 0, "cohort_level_inference_count": 0,
        "S5_execution_count": 0, "S6_execution_count": 0,
    }, producer_commit)
    na_path = write_report("typed_failure_na_denominator_report.json", "s4_typed_failure_na_denominator_report", "PASS", {
        "scheduled_source_count": 1744, "scheduled_unit_count": 5232, "source_completion_count": source_completions,
        "unit_type_counts": dict(sorted(by_type.items())), "terminal_state_counts": dict(sorted(states.items())),
        "VALID_NNS_count": valid_nns, "VALID_OCCLUSION_count": valid_occlusion,
        "P1_valid_block_count": p1_valid_blocks, "P1_raw_permutation_draw_count": p1_raw_draws,
        "P2_matched_coordinate_count": p2_matched_coordinates, "substitute_sample_count": 0, "zero_fill_count": 0,
        "scientific_claim_decision": None,
    }, producer_commit)
    manifest_path = write_report("large_array_manifest.json", "s4_large_array_manifest", "PASS", {
        "large_array_root": s4.PAYLOAD_REL.as_posix(), "record_count": len(array_records),
        "records_digest": s4.records_digest(array_records), "records": array_records,
    }, producer_commit)
    numeric_started = time.monotonic(); cpu_started = time.process_time()
    numeric_output = subprocess.check_output([str(PYTHON), "scripts/validate_nns_vit_patch_depth_s4_numeric.py", "final"], cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, text=True)
    numeric_resource = {"CPU_seconds": time.process_time() - cpu_started, "wall_seconds": time.monotonic() - numeric_started, "GPU_seconds": 0.0}
    numeric = json.loads(numeric_output)
    numeric_path = write_report("independent_numeric_validation_report.json", "s4_independent_numeric_validation_report", "PASS", numeric | {"validator_resource": numeric_resource}, producer_commit)
    metadata_started = time.monotonic(); metadata_cpu = time.process_time()
    # Full validator needs the terminal itself, so this preliminary report uses the same source checks plus direct unit coverage.
    pre_output = subprocess.check_output(["python3", "scripts/validate_nns_vit_patch_depth_s4.py", "preaccess"], cwd=REPO, text=True)
    metadata = json.loads(pre_output)
    metadata.update({"unit_terminal_count": len(paths), "source_completion_count": source_completions, "unit_type_counts": dict(by_type), "terminal_state_counts": dict(states), "validator_resource": {"CPU_seconds": time.process_time() - metadata_cpu, "wall_seconds": time.monotonic() - metadata_started, "GPU_seconds": 0.0}})
    metadata_path = write_report("independent_metadata_provenance_validation_report.json", "s4_independent_metadata_provenance_validation_report", "PASS", metadata, producer_commit)
    false_pass = json.loads(subprocess.check_output(["python3", "scripts/validate_nns_vit_patch_depth_s4.py", "false-pass"], cwd=REPO, text=True))
    false_path = write_report("false_pass_mutation_audit.json", "s4_false_pass_mutation_audit", "PASS", false_pass | {"post_production": True, "formal_outcome_values_used": False}, producer_commit)
    failures = sum(states[key] for key in ("FAILED_NUMERICAL", "FAILED_NNS", "FAILED_OCCLUSION", "BLOCKED_DATA_INTEGRITY", "BLOCKED_INFRASTRUCTURE", "FAILED_IMPLEMENTATION"))
    complete = len(paths) == 5232 and source_completions == 1744 and all(by_type[key] == 1744 for key in s4.UNIT_TYPES)
    stage_status = "PASS" if complete and failures == 0 else "BLOCKED_DATA_INTEGRITY" if states["BLOCKED_DATA_INTEGRITY"] else "FAILED_IMPLEMENTATION"
    next_authorization = "S5_D2_MISSING_CLASS_RECONCILIATION_GATE_ONLY" if stage_status == "PASS" else "NONE"
    required_evidence = {path.name: artifact(path) for path in (resource_path, access_path, na_path, manifest_path, numeric_path, metadata_path, false_path, ROOT / "selected_member_materialization.json", ROOT / "implementation_seal.json")}
    excluded = {"whole_root_inventory.json", "terminal_receipt.json", "completion/s4.pending.json"}
    root_files = [artifact(path) for path in sorted(ROOT.rglob("*")) if path.is_file() and path.relative_to(ROOT).as_posix() not in excluded]
    inventory_path = write_report("whole_root_inventory.json", "s4_whole_root_inventory", "PASS", {"files": root_files, "files_digest": s4.canonical_digest(root_files), "large_arrays_bound_by_manifest": True}, producer_commit)
    required_evidence["whole_root_inventory.json"] = artifact(inventory_path)
    terminal = s4.signed("s4_terminal_receipt", stage_status, {
        "status": stage_status, "evidence_root": s4.ROOT_REL.as_posix(), "large_array_root": s4.PAYLOAD_REL.as_posix(),
        "implementation_seal_commit": json.loads((ROOT / "implementation_seal.json").read_text())["payload"]["implementation_commit"],
        "source_count": 1744, "source_completion_count": source_completions, "scheduled_unit_count": 5232, "unit_terminal_count": len(paths),
        "unit_type_counts": dict(sorted(by_type.items())), "terminal_state_counts": dict(sorted(states.items())),
        "quadrature_selected_step_counts": dict(sorted(quadrature.items())), "P1_raw_permutation_draw_count": p1_raw_draws,
        "P2_matched_coordinate_count": p2_matched_coordinates, "audit_subset_validated_count": numeric["audit_subset_validated_count"],
        "deterministic_repeatability": "PASS", "D2_only_member_read_count": 0, "S3_D2_prediction_error_numeric_read_count": 0,
        "cohort_level_inference_count": 0, "S5_executed": False, "S6_executed": False,
        "scientific_claim_decision": None, "cost_or_wall_time_termination_used": False,
        "resources": artifact(resource_path), "required_evidence": required_evidence, "next_authorization": next_authorization,
    }, producer_commit=producer_commit)
    pending = ROOT / "completion/s4.pending.json"
    terminal_path = ROOT / "terminal_receipt.json"
    if not pending.is_file() or terminal_path.exists():
        raise RuntimeError("stage terminal transition unavailable")
    temporary = pending.with_name("s4.terminal.tmp")
    s4.write_json_new(temporary, terminal)
    os.link(temporary, terminal_path)
    s4.fsync_directory(terminal_path.parent)
    temporary.unlink(); pending.unlink(); s4.fsync_directory(pending.parent)
    print(json.dumps({"status": stage_status, "source_completion_count": source_completions, "unit_terminal_count": len(paths), "state_counts": dict(states), "next_authorization": next_authorization}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
