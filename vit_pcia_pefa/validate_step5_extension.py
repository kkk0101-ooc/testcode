#!/usr/bin/env python3
"""CPU-only prompt-to-artifact audit for a Step-5 extension root."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_extension_queue import audit_coverage, task_seed  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value); value["canonical_digest"] = digest(value); return value


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if digest(value) != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def write_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_step5_extension.py EXTENSION_ROOT")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "VALIDATOR_MUST_BE_CPU_ONLY")
    root = Path(sys.argv[1]).resolve()
    output = root / "validation_report.json"
    if output.exists():
        raise RuntimeError("VALIDATION_ALREADY_EXISTS")
    protocol = verify_signed(root / "protocol.json"); manifest = verify_signed(root / "task_manifest.json")
    binding = verify_signed(root / "source_binding.json"); freeze = verify_signed(root / "freeze_receipt.json")
    launch = verify_signed(root / "launch_terminal.json"); analysis = verify_signed(root / "analysis.json")
    claims = verify_signed(root / "claim_decisions.json"); ledger = verify_signed(root / "resource_ledger.json")
    terminal = verify_signed(root / "terminal_receipt.json"); artifacts = verify_signed(root / "artifact_manifest.json")
    require(protocol["stage"] == "ENDPOINT_PCIA_PEFA_STEP5_EXTENSION", "STAGE")
    require(protocol["status"] == "FROZEN_PROSPECTIVE" and freeze["status"] == "FROZEN", "FREEZE")
    require(protocol["prior_outcome_known"] and protocol["new_hypothesis_after_original_pilot"], "DISCLOSURE")
    require(protocol["original_terminal_immutable"] and not protocol["confirmatory_claim_allowed"], "CLAIM_BOUNDARY")
    require(not protocol["evaluation_access"] and not protocol["step6_authorized"], "STAGE_OR_SPLIT_BOUNDARY")
    honest_recovery = (freeze["extension_outcome_access_before_freeze"]
                       and freeze.get("recovery_trigger_outcome_independent") is True
                       and protocol["prior_outcome_known"] is True
                       and protocol["outcome_access_during_execution"] is False)
    require((not freeze["extension_outcome_access_before_freeze"] or honest_recovery)
            and not freeze["step6_started"], "OUTCOME_BOUNDARY")
    require(manifest["task_count"] == protocol["task_count"] == len(manifest["tasks"]), "TASK_COUNT")
    require(all(row["payload"]["split"] == "calibration" for row in manifest["tasks"]), "SPLIT")
    require(all(task_seed(row["task_id"]) >= 0 for row in manifest["tasks"]), "SEED")
    require(freeze["protocol_digest"] == protocol["canonical_digest"], "PROTOCOL_BINDING")
    require(freeze["task_manifest_digest"] == manifest["canonical_digest"], "MANIFEST_BINDING")
    require(freeze["source_binding_digest"] == binding["canonical_digest"], "SOURCE_BINDING")
    for row in (*binding["files"], *binding["checkpoints"], *binding["pilot_images"]):
        path = REPO / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"], "BOUND_FILE_SIZE:" + row["path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "BOUND_FILE_HASH:" + row["path"])
    coverage = audit_coverage(root / "queue.sqlite3", root / "results")
    require(coverage["status"] == "PASS" and launch["status"] == "PASS_EXTENSION_EXECUTION", "EXECUTION")
    require(claims["original_r8_decision_unchanged"] == "NOT_SUPPORTED_EMPIRICAL_VALUE", "ORIGINAL_DECISION")
    require(not claims["confirmatory_claim_allowed"] and not claims["step6_production_authorized"], "PROMOTION_BOUNDARY")
    require(terminal["analysis_digest"] == analysis["canonical_digest"], "ANALYSIS_BINDING")
    require(terminal["claim_decisions_digest"] == claims["canonical_digest"], "CLAIMS_BINDING")
    require(terminal["resource_ledger_digest"] == ledger["canonical_digest"], "LEDGER_BINDING")
    require(not terminal["step6_started"], "STEP6_STARTED")
    require(all(row["decision"] in {"SUPPORTED", "NOT_SUPPORTED", "INVALID"}
                for row in claims["decisions"]), "CLAIM_VOCABULARY")
    decisions = [row["decision"] for row in claims["decisions"]]
    expected_terminal = "INVALID" if "INVALID" in decisions or ledger["status"] == "INVALID" else (
        "SUPPORTED" if all(value == "SUPPORTED" for value in decisions) else "NOT_SUPPORTED")
    require(terminal["status"] == expected_terminal, "TERMINAL_STATUS")
    require(ledger["cumulative_step5_h20_hours"] <= protocol["resources"]["cumulative_step5_h20_hours_cap"], "H20_CAP")
    require(ledger["peak_reserved_gpu_bytes"] <= protocol["resources"]["gpu_memory_cap_bytes"], "GPU_CAP")
    require(ledger["peak_cpu_rss_bytes"] <= protocol["resources"]["cpu_rss_cap_bytes"], "CPU_CAP")
    require(protocol["loop"]["object"].startswith("oriented closed port-field line integral"), "LOOP_OBJECT")
    require(protocol["source_resolution"]["grids"] == [4, 7], "RESOLUTION_ORDER")
    if protocol["phase"] == "E3A":
        require(analysis["pixel_level_authorized"] is False, "PIXEL_LEVEL_BOUNDARY")
        require((terminal["next_authorization"] == "FRESH_E3B_14X14_ROOT_ONLY")
                == bool(analysis["advance_to_14x14"]), "E3_ADVANCE")
    manifest_paths = set()
    for row in artifacts["artifacts"]:
        require(row["path"] not in manifest_paths, "DUPLICATE_ARTIFACT")
        manifest_paths.add(row["path"]); path = root / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"], "ARTIFACT_SIZE:" + row["path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "ARTIFACT_HASH:" + row["path"])
    require(artifacts["artifact_count"] == len(artifacts["artifacts"]), "ARTIFACT_COUNT")
    tests = subprocess.run(
        [str(REPO / ".conda/envs/vitnns/bin/python"), "-m", "pytest", "-q",
         "vit_pcia_pefa/tests/test_step5_extension.py", "vit_pcia_pefa/tests/test_step5_queue.py"],
        cwd=REPO, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                       "PYTHONPATH": ".:vitnns_rebuild/src"}, capture_output=True, text=True,
    )
    require(tests.returncode == 0, "UNIT_TESTS:" + tests.stdout + tests.stderr)
    report = signed({
        "schema_version": 1, "run_id": protocol["run_id"], "phase": protocol["phase"],
        "status": "PASS_STEP5_EXTENSION_AUDIT", "tasks_verified": protocol["task_count"],
        "claims_verified": len(claims["decisions"]), "artifacts_verified": len(manifest_paths),
        "terminal_status": terminal["status"], "original_r8_decision_unchanged": True,
        "evaluation_access": False, "step6_started": False, "unit_tests": tests.stdout.strip(),
    })
    write_new(output, report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
