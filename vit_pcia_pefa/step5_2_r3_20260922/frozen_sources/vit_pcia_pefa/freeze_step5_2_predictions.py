#!/usr/bin/env python3
"""Seal clean-model predictions before any Step-5.2 intervention starts."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent; REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_2_io import sha256, verify, write_new  # noqa: E402
from vit_pcia_pefa.src.step5_2_queue import audit_coverage  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: freeze_step5_2_predictions.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json"); launch = verify(run / "prediction_launch_terminal.json")
    if (run / "intervention_startup_snapshot.json").exists() or (run / "intervention_queue.sqlite3").exists():
        raise RuntimeError("INTERVENTION_ALREADY_STARTED")
    coverage = audit_coverage(run / "prediction_queue.sqlite3", run / "prediction_results", "prediction")
    if launch["status"] != "PASS_PREDICTION_TASK_EXECUTION" or coverage["status"] != "PASS": raise RuntimeError("PREDICTION_COVERAGE")
    with sqlite3.connect(run / "prediction_queue.sqlite3") as connection:
        rows = connection.execute("SELECT task_id,result_path,result_digest FROM tasks ORDER BY task_id").fetchall()
    predictions = []
    for task_id, result_path, result_digest in rows:
        path = run / "prediction_results" / result_path; document = json.loads(path.read_text())
        if document["result"]["status"] != "PASS_PREDICTION_TASK": raise RuntimeError("PREDICTION_TASK_STATUS")
        array = document["result"]["prediction_array"]; array_path = run / array["path"]
        if array_path.stat().st_size != array["bytes"] or sha256(array_path) != array["sha256"]: raise RuntimeError("PREDICTION_ARRAY_IDENTITY")
        predictions.append({"task_id": task_id, "result_path": (Path("prediction_results") / result_path).as_posix(),
            "result_digest": result_digest, "result_sha256": sha256(path), "array_path": array["path"],
            "array_bytes": array["bytes"], "array_sha256": array["sha256"]})
    receipt = write_new(run / "prediction_freeze_receipt.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "PASS_PREDICTIONS_FROZEN_BEFORE_INTERVENTION", "prediction_count": len(predictions),
        "prediction_launch_digest": launch["canonical_digest"], "predictions": predictions,
        "intervention_started_before_freeze": False, "frozen_unix_ns": time.time_ns(),
        "evaluation_access": False, "step6_started": False})
    print(json.dumps({"status": receipt["status"], "prediction_count": len(predictions),
                      "digest": receipt["canonical_digest"]}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
