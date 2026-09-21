#!/usr/bin/env python3
"""Seal the immutable Step-5.2 r2 prediction infrastructure failure."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import verify, write_new  # noqa: E402
from vit_pcia_pefa.src.step5_2_queue import audit_coverage  # noqa: E402


def main() -> int:
    run = ROOT / "step5_2_r2_20260922"
    protocol = verify(run / "protocol.json")
    p0 = verify(run / "p0_decision.json")
    launch = verify(run / "prediction_launch_terminal.json")
    coverage = audit_coverage(run / "prediction_queue.sqlite3", run / "prediction_results", "prediction")
    if p0["status"] != "PASS_P0_NUMERICAL_FEASIBILITY":
        raise RuntimeError("R2_P0_NOT_PASS")
    if launch["status"] != "FAIL_PREDICTION_TASK_EXECUTION" or coverage["completed_count"] != 0:
        raise RuntimeError("R2_PREDICTION_FAILURE_STATE")
    if (run / "prediction_freeze_receipt.json").exists() or (run / "intervention_queue.sqlite3").exists():
        raise RuntimeError("R2_PHASE_BOUNDARY_VIOLATION")
    with sqlite3.connect(run / "prediction_queue.sqlite3") as connection:
        attempts = connection.execute(
            "SELECT task_id,attempt,claimed_ns,completed_ns,status FROM attempts ORDER BY task_id,attempt"
        ).fetchall()
        failures = [json.loads(value) for (value,) in connection.execute(
            "SELECT value FROM metadata WHERE key LIKE 'failure:%' ORDER BY key"
        ).fetchall()]
    expected_attempts = int(protocol["p1"]["image_task_count"]) * int(protocol["resources"]["max_attempts"])
    if (len(attempts) != expected_attempts or len(failures) != expected_attempts
            or any(row[4] not in {"RETRYABLE", "FAILED"} for row in attempts)
            or any(row.get("type") != "KeyError" or row.get("message") != "'task_id'" for row in failures)):
        raise RuntimeError("R2_FAILURE_NOT_UNIFORM")
    failed_seconds = math.fsum((int(row[3]) - int(row[2])) / 1e9 for row in attempts)
    receipt = write_new(run / "failure_receipt.json", {
        "schema_version": 1, "run_id": protocol["run_id"],
        "status": "INVALID_INFRASTRUCTURE",
        "failure_code": "FAIL_PREDICTION_WORKER_TASK_ID_PLUMBING",
        "failed_phase": "P1_PREDICTION",
        "root_immutable": True, "p0_status": p0["status"],
        "failed_attempt_count": len(attempts), "failed_attempt_gpu_seconds": failed_seconds,
        "accepted_prediction_count": coverage["completed_count"],
        "prediction_artifacts_frozen": False, "intervention_started": False,
        "scientific_decision": "NONE_INFRASTRUCTURE_INVALID",
        "recovery": "fresh_root_fix_queue_task_id_plumbing_and_rebind_valid_r2_p0",
        "prediction_launch_digest": launch["canonical_digest"],
        "evaluation_access": False, "step6_started": False,
        "completed_unix_ns": time.time_ns(),
    })
    print(json.dumps({"status": receipt["status"], "failure_code": receipt["failure_code"],
                      "failed_attempt_gpu_hours": failed_seconds / 3600.0,
                      "digest": receipt["canonical_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
