from __future__ import annotations

import hashlib
import sqlite3

from vit_pcia_pefa.src.step5_queue import (
    SEED_NAMESPACE, audit_coverage, claim_task, complete_task, create_queue, task_seed,
)


def test_step5_seed_exact_contract_and_worker_independence(tmp_path):
    task_id = "vit_b16__00"
    expected = int.from_bytes(
        hashlib.sha256(b"ENDPOINT_PCIA_PEFA|step5|vit_b16__00").digest()[:8], "big"
    )
    assert SEED_NAMESPACE == "ENDPOINT_PCIA_PEFA|step5"
    assert task_seed(task_id) == expected
    queue = tmp_path / "queue.sqlite3"; results = tmp_path / "results"
    create_queue(queue, [{"task_id": task_id, "payload": {"value": 1}}])
    claim = claim_task(queue, worker_id="worker-arbitrary", device_uuid="GPU-arbitrary")
    assert claim["seed"] == expected
    complete_task(queue, results, claim, {"ok": True})
    assert audit_coverage(queue, results)["status"] == "PASS"


def test_step5_seed_drift_fails_closed(tmp_path):
    queue = tmp_path / "queue.sqlite3"
    create_queue(queue, [{"task_id": "fixed", "payload": {}}])
    with sqlite3.connect(queue) as connection:
        connection.execute("UPDATE tasks SET seed='1' WHERE task_id='fixed'")
    assert any(row["code"] == "SEED_DRIFT" for row in audit_coverage(queue, tmp_path / "results")["invalid"])
