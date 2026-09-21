#!/usr/bin/env python3
"""Seal the development-only A0 gate without making a science claim."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed: raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    body = dict(value); body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest()
    payload = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: finalize_step5_1_a0.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json")
    fixture = verify(run / "a0_fixture_result.json"); launch = verify(run / "a0_launch_terminal.json")
    results = [verify(run / "a0" / f"{model}.json") for model in ("deit_s16", "vit_b16")]
    wall_seconds = sum(float(row["resources"]["wall_seconds"]) for row in results)
    h20_hours = wall_seconds / 3600.0
    cap = protocol["resources"]
    resources_pass = (h20_hours <= cap["step5_1_h20_hours_cap"]
                      and cap["predecessor_h20_hours"] + h20_hours <= cap["cumulative_step5_h20_hours_cap"]
                      and max(row["resources"]["peak_reserved_gpu_bytes"] for row in results) <= cap["gpu_memory_cap_bytes"]
                      and max(row["resources"]["peak_cpu_rss_bytes"] for row in results) <= cap["cpu_rss_cap_bytes"])
    gate = (fixture["status"] == "PASS_A0_FIXTURE" and launch["status"] == "PASS_A0_TASK_EXECUTION"
            and all(row["status"] == "PASS_A0_REAL_MODEL" for row in results) and resources_pass)
    write_new(run / "a0_decision.json", {
        "schema_version": 1, "run_id": protocol["run_id"],
        "status": "PASS_A0_NUMERICAL_FEASIBILITY" if gate else "INVALID_A0_NUMERICAL_FEASIBILITY",
        "scientific_claim": False, "fixture_status": fixture["status"],
        "model_statuses": {row["model_id"]: row["status"] for row in results},
        "resource_status": "SUPPORTED" if resources_pass else "INVALID",
        "a0_h20_hours": h20_hours, "cumulative_step5_h20_hours": cap["predecessor_h20_hours"] + h20_hours,
        "a1_authorized_by_gate": gate, "b1_authorized_by_gate": False,
        "evaluation_access": False, "step6_started": False, "completed_unix_ns": time.time_ns(),
    })
    print(json.dumps({"status": "PASS_A0_NUMERICAL_FEASIBILITY" if gate else "INVALID_A0_NUMERICAL_FEASIBILITY",
                      "a0_h20_hours": h20_hours}, sort_keys=True))
    return 0 if gate else 3


if __name__ == "__main__": raise SystemExit(main())
