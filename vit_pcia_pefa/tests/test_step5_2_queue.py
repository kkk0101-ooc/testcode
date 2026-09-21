from __future__ import annotations

import json
from pathlib import Path

from vit_pcia_pefa.src.step5_2_queue import audit_coverage, complete_task, create_queue, claim_task, task_seed


def test_phase_specific_queue_and_result_audit(tmp_path: Path):
    queue = tmp_path / "queue.sqlite3"; results = tmp_path / "results"
    receipt = create_queue(queue, [{"task_id": "x", "value": 1}], "prediction")
    assert receipt["task_count"] == 1
    claim = claim_task(queue, worker_id="w", device_uuid="u", lease_seconds=30, max_attempts=2)
    assert claim is not None and int(claim["seed"]) == task_seed("prediction", "x")
    assert claim["task_id"] == "x" and "task_id" not in claim["payload"]
    complete_task(queue, results, claim, {"status": "PASS_PREDICTION_TASK"})
    assert audit_coverage(queue, results, "prediction")["status"] == "PASS"
    assert audit_coverage(queue, results, "intervention")["status"] == "FAIL"
