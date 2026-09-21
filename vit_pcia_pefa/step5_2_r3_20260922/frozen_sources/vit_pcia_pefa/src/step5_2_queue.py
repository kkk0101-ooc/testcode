"""Dynamic queues for prospective Step-5.2 prediction and intervention."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from vitnns.endpoint_ep1_queue import (
    QueueContractError, claim_task, complete_task, heartbeat, create_queue as _create_queue, digest,
)


def task_seed(phase: str, task_id: str) -> int:
    if phase not in {"prediction", "intervention"} or not task_id:
        raise QueueContractError("STEP5_2_TASK_ID")
    return int.from_bytes(hashlib.sha256(f"ENDPOINT_PCIA_PEFA|step5_2|{phase}|{task_id}".encode()).digest()[:8], "big")


def create_queue(path: Path, tasks: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
    normalized = [{"task_id": row["task_id"],
                   "payload": row["payload"] if set(row) == {"task_id", "payload"}
                   else {key: value for key, value in row.items() if key != "task_id"}} for row in tasks]
    receipt = _create_queue(path, normalized)
    namespace = f"ENDPOINT_PCIA_PEFA|step5_2|{phase}"
    with sqlite3.connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany("UPDATE tasks SET seed=? WHERE task_id=?",
                               [(str(task_seed(phase, row["task_id"])), row["task_id"]) for row in normalized])
        connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('seed_namespace',?)", (namespace,))
        connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('phase',?)", (phase,))
        connection.commit()
    return {**receipt, "seed_namespace": namespace, "phase": phase}


def audit_coverage(path: Path, result_root: Path, phase: str) -> dict[str, Any]:
    namespace = f"ENDPOINT_PCIA_PEFA|step5_2|{phase}"
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        metadata = dict(connection.execute("SELECT key,value FROM metadata").fetchall())
        rows = connection.execute(
            "SELECT task_id,payload_json,payload_digest,seed,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id"
        ).fetchall()
        attempts = connection.execute("SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt").fetchall()
    missing: list[str] = []; invalid: list[dict[str, str]] = []; completed: list[str] = []
    if metadata.get("seed_namespace") != namespace or metadata.get("phase") != phase:
        invalid.append({"task_id": "__queue__", "code": "PHASE_OR_SEED_NAMESPACE_DRIFT"})
    for row in rows:
        if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_digest"]:
            invalid.append({"task_id": row["task_id"], "code": "PAYLOAD_DIGEST"})
        if int(row["seed"]) != task_seed(phase, row["task_id"]):
            invalid.append({"task_id": row["task_id"], "code": "SEED_DRIFT"})
        if row["state"] != "completed":
            missing.append(row["task_id"]); continue
        result_path = result_root.resolve() / row["result_path"]
        try:
            document = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            invalid.append({"task_id": row["task_id"], "code": "RESULT_MISSING_OR_JSON"}); continue
        body = dict(document); observed = body.pop("canonical_digest", None)
        if digest(body) != observed or observed != row["result_digest"]:
            invalid.append({"task_id": row["task_id"], "code": "RESULT_DIGEST"}); continue
        if document["task_id"] != row["task_id"] or int(document["attempt"]) != int(row["attempt"]):
            invalid.append({"task_id": row["task_id"], "code": "RESULT_IDENTITY"}); continue
        completed.append(row["task_id"])
    accepted = [(row["task_id"], int(row["attempt"])) for row in attempts if row["status"] == "COMPLETED"]
    status = "PASS" if not missing and not invalid and len(completed) == len(rows) == len(accepted) else "FAIL"
    return {"status": status, "phase": phase, "task_count": len(rows), "completed_count": len(completed),
            "missing_task_ids": missing, "invalid": invalid, "accepted_attempts": accepted,
            "seed_namespace": namespace}


__all__ = ["audit_coverage", "claim_task", "complete_task", "create_queue", "heartbeat", "task_seed"]
