from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from vitnns.endpoint_ep1_queue import (
    QueueContractError, audit_coverage, claim_task, complete_task, create_queue,
    heartbeat, task_seed,
)


def tasks(count=6):
    return [{"task_id": f"task-{index:03d}", "payload": {"index": index}} for index in range(count)]


def drain(queue, results, workers):
    def run(worker):
        completed = []
        while True:
            claim = claim_task(queue, worker_id=worker, device_uuid=f"GPU-{worker}")
            if claim is None:
                return completed
            complete_task(queue, results, claim, {"value": claim["payload"]["index"] ** 2})
            completed.append((claim["task_id"], claim["seed"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [row for group in pool.map(run, (f"w{i}" for i in range(workers))) for row in group]


def test_single_and_multi_worker_have_identical_tasks_seeds_and_results(tmp_path):
    manifests = tasks()
    observations = []
    for worker_count in (1, 3):
        root = tmp_path / str(worker_count)
        queue, results = root / "queue.sqlite3", root / "results"
        identity = create_queue(queue, manifests)
        done = drain(queue, results, worker_count)
        audit = audit_coverage(queue, results)
        assert audit["status"] == "PASS" and audit["completed_count"] == len(manifests)
        values = {}
        for path in results.rglob("*.json"):
            row = json.loads(path.read_text()); values[row["task_id"]] = (row["seed"], row["result"])
        observations.append((identity, sorted(done), values))
    assert observations[0] == observations[1]


def test_expired_attempt_is_fenced_and_recovered(tmp_path):
    queue, results = tmp_path / "queue.sqlite3", tmp_path / "results"
    create_queue(queue, tasks(1))
    old = claim_task(queue, worker_id="old", device_uuid="GPU-0", lease_seconds=1, now_ns=1)
    new = claim_task(queue, worker_id="new", device_uuid="GPU-1", lease_seconds=1, now_ns=2_000_000_002)
    assert new["attempt"] == 2 and new["fencing_token"] > old["fencing_token"]
    with pytest.raises(QueueContractError, match="STALE_FENCING_TOKEN"):
        complete_task(queue, results, old, {"value": "stale"}, now_ns=2_000_000_003)
    complete_task(queue, results, new, {"value": "accepted"}, now_ns=2_000_000_004)
    assert audit_coverage(queue, results)["status"] == "PASS"


def test_expired_final_attempt_becomes_failed_not_reclaimable(tmp_path):
    queue = tmp_path / "queue.sqlite3"
    create_queue(queue, tasks(1))
    claim_task(queue, worker_id="old", device_uuid="GPU-0", lease_seconds=1, now_ns=1)
    assert claim_task(
        queue, worker_id="new", device_uuid="GPU-1", lease_seconds=1,
        max_attempts=1, now_ns=2_000_000_002,
    ) is None
    audit = audit_coverage(queue, tmp_path / "results")
    assert audit["status"] == "FAIL" and audit["missing_task_ids"] == ["task-000"]


def test_heartbeat_and_immutable_result(tmp_path):
    queue, results = tmp_path / "queue.sqlite3", tmp_path / "results"
    create_queue(queue, tasks(1))
    claim = claim_task(queue, worker_id="w", device_uuid="GPU-0", now_ns=10)
    assert heartbeat(queue, claim, now_ns=20) > 20
    destination = complete_task(queue, results, claim, {"ok": True}, now_ns=30)
    before = destination.read_bytes()
    with pytest.raises(QueueContractError):
        complete_task(queue, results, claim, {"ok": False}, now_ns=40)
    assert destination.read_bytes() == before


def test_exact_coverage_detects_missing_and_tampered_results(tmp_path):
    queue, results = tmp_path / "queue.sqlite3", tmp_path / "results"
    create_queue(queue, tasks(2))
    claim = claim_task(queue, worker_id="w", device_uuid="GPU-0")
    result_path = complete_task(queue, results, claim, {"ok": True})
    audit = audit_coverage(queue, results)
    assert audit["status"] == "FAIL" and audit["missing_task_ids"] == ["task-001"]
    result_path.write_text("{}\n")
    assert any(row["code"] == "RESULT_DIGEST" for row in audit_coverage(queue, results)["invalid"])


def test_manifest_drift_and_seed_independence_fail_closed(tmp_path):
    queue = tmp_path / "queue.sqlite3"
    create_queue(queue, tasks(1))
    with pytest.raises(QueueContractError, match="TASK_MANIFEST_DRIFT"):
        create_queue(queue, tasks(2))
    assert task_seed("fixed") == task_seed("fixed")
    with sqlite3.connect(queue) as connection:
        connection.execute("UPDATE tasks SET seed='1' WHERE task_id='task-000'")
    assert any(row["code"] == "SEED_DRIFT" for row in audit_coverage(queue, tmp_path / "results")["invalid"])
