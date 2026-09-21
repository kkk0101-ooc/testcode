#!/usr/bin/env python3
"""Seal the development-only Step-5.2 P0 gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import verify, write_new  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: finalize_step5_2_p0.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json")
    fixture = verify(run / "p0_fixture_result.json"); launch = verify(run / "p0_launch_terminal.json")
    results = [verify(run / "p0" / f"{model}.json") for model in protocol["models"]]
    h20_hours = sum(float(row["resources"]["wall_seconds"]) for row in results) / 3600.0
    cap = protocol["resources"]
    prior_step5_2 = float(cap.get("prior_step5_2_h20_hours", 0.0))
    resources = prior_step5_2 + h20_hours <= cap["step5_2_h20_hours_cap"]
    resources = resources and cap["predecessor_h20_hours"] + prior_step5_2 + h20_hours <= cap["cumulative_step5_h20_hours_cap"]
    resources = resources and max(row["resources"]["peak_reserved_gpu_bytes"] for row in results) <= cap["gpu_memory_cap_bytes"]
    resources = resources and max(row["resources"]["peak_cpu_rss_bytes"] for row in results) <= cap["cpu_rss_cap_bytes"]
    passed = fixture["status"] == "PASS_P0_FIXTURE" and launch["status"] == "PASS_P0_TASK_EXECUTION"
    passed = passed and all(row["status"] == "PASS_P0_REAL_MODEL" for row in results) and resources
    result = write_new(run / "p0_decision.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "PASS_P0_NUMERICAL_FEASIBILITY" if passed else "INVALID_NUMERICAL_OR_INTERVENTION",
        "scientific_claim": False, "fixture_status": fixture["status"],
        "model_statuses": {row["model_id"]: row["status"] for row in results},
        "resource_status": "SUPPORTED" if resources else "INVALID", "p0_h20_hours": h20_hours,
        "prior_step5_2_h20_hours": prior_step5_2,
        "cumulative_step5_h20_hours": cap["predecessor_h20_hours"] + prior_step5_2 + h20_hours,
        "prediction_authorized_by_gate": passed, "intervention_authorized": False,
        "evaluation_access": False, "step6_started": False, "completed_unix_ns": time.time_ns()})
    print(json.dumps({"status": result["status"], "p0_h20_hours": h20_hours}, sort_keys=True)); return 0 if passed else 3


if __name__ == "__main__": raise SystemExit(main())
