"""Step-5 queue adapter with the method-contract seed namespace."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from vitnns.endpoint_ep1_queue import (
    QueueContractError, claim_task, complete_task, heartbeat, canonical, digest,
    create_queue as _create_queue,
)

SEED_NAMESPACE = "ENDPOINT_PCIA_PEFA|step5"


def task_seed(task_id: str) -> int:
    if not isinstance(task_id, str) or not task_id:
        raise QueueContractError("TASK_ID_EMPTY")
    return int.from_bytes(hashlib.sha256(f"{SEED_NAMESPACE}|{task_id}".encode()).digest()[:8], "big")


def create_queue(path: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create the base queue, then bind every seed to the Step-5 namespace."""
    receipt = _create_queue(path, tasks)
    with sqlite3.connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            "UPDATE tasks SET seed=? WHERE task_id=?",
            [(str(task_seed(row["task_id"])), row["task_id"]) for row in tasks],
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('seed_namespace',?)",
            (SEED_NAMESPACE,),
        )
        connection.commit()
    return {**receipt, "seed_namespace": SEED_NAMESPACE}


def audit_coverage(path: Path, result_root: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        namespace = connection.execute(
            "SELECT value FROM metadata WHERE key='seed_namespace'"
        ).fetchone()
        rows = connection.execute(
            "SELECT task_id,payload_json,payload_digest,seed,state,attempt,result_path,result_digest "
            "FROM tasks ORDER BY task_id"
        ).fetchall()
        attempts = connection.execute(
            "SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt"
        ).fetchall()
    missing, invalid, completed = [], [], []
    if namespace is None or namespace[0] != SEED_NAMESPACE:
        invalid.append({"task_id": "__queue__", "code": "SEED_NAMESPACE_DRIFT"})
    for row in rows:
        if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_digest"]:
            invalid.append({"task_id": row["task_id"], "code": "PAYLOAD_DIGEST"})
        if int(row["seed"]) != task_seed(row["task_id"]):
            invalid.append({"task_id": row["task_id"], "code": "SEED_DRIFT"})
        if row["state"] != "completed":
            missing.append(row["task_id"]); continue
        result_path = result_root.resolve() / row["result_path"]
        try:
            document = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            invalid.append({"task_id": row["task_id"], "code": "RESULT_MISSING_OR_JSON"}); continue
        body = dict(document); observed_digest = body.pop("canonical_digest", None)
        if digest(body) != observed_digest or observed_digest != row["result_digest"]:
            invalid.append({"task_id": row["task_id"], "code": "RESULT_DIGEST"}); continue
        if document["task_id"] != row["task_id"] or int(document["attempt"]) != int(row["attempt"]):
            invalid.append({"task_id": row["task_id"], "code": "RESULT_IDENTITY"}); continue
        completed.append(row["task_id"])
    accepted = [(row["task_id"], int(row["attempt"])) for row in attempts if row["status"] == "COMPLETED"]
    duplicates = sorted(task for task in set(completed) if completed.count(task) > 1)
    status = "PASS" if not missing and not invalid and not duplicates and len(accepted) == len(rows) else "FAIL"
    return {
        "status": status, "task_count": len(rows), "completed_count": len(completed),
        "missing_task_ids": missing, "invalid": invalid, "duplicate_task_ids": duplicates,
        "accepted_attempts": accepted, "seed_namespace": SEED_NAMESPACE,
    }


__all__ = ["SEED_NAMESPACE", "audit_coverage", "claim_task", "complete_task",
           "create_queue", "heartbeat", "task_seed"]
