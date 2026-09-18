"""Minimal SQLite WAL task queue for Endpoint Scanner EP1 and later stages."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


NAMESPACE = "NNS_VIT_ENDPOINT_QUEUE_V1"


class QueueContractError(ValueError):
    """The queue or an attempted state transition violates its contract."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def task_seed(task_id: str) -> int:
    if not isinstance(task_id, str) or not task_id:
        raise QueueContractError("TASK_ID_EMPTY")
    return int.from_bytes(hashlib.sha256(f"{NAMESPACE}|{task_id}".encode()).digest()[:8], "big")


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, isolation_level=None, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def create_queue(path: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create or verify a fixed task manifest without changing task identity."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for task in tasks:
        if set(task) != {"task_id", "payload"}:
            raise QueueContractError("TASK_SCHEMA_OPEN_OR_INCOMPLETE")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise QueueContractError("TASK_ID_DUPLICATE_OR_EMPTY")
        seen.add(task_id)
        payload_json = canonical(task["payload"]).decode()
        rows.append((task_id, payload_json, hashlib.sha256(payload_json.encode()).hexdigest(), task_seed(task_id)))
    rows.sort(key=lambda row: row[0])
    manifest_digest = digest([{"task_id": row[0], "payload": json.loads(row[1])} for row in rows])
    with _connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks(
              task_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, payload_digest TEXT NOT NULL,
              seed TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', attempt INTEGER NOT NULL DEFAULT 0,
              fencing_token INTEGER, worker_id TEXT, device_uuid TEXT, lease_until_ns INTEGER,
              heartbeat_ns INTEGER, result_path TEXT, result_digest TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts(
              task_id TEXT NOT NULL, attempt INTEGER NOT NULL, fencing_token INTEGER NOT NULL,
              worker_id TEXT NOT NULL, device_uuid TEXT NOT NULL, claimed_ns INTEGER NOT NULL,
              lease_until_ns INTEGER NOT NULL, status TEXT NOT NULL, completed_ns INTEGER,
              PRIMARY KEY(task_id, attempt), FOREIGN KEY(task_id) REFERENCES tasks(task_id)
            );
            """
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute("SELECT value FROM metadata WHERE key='manifest_digest'").fetchone()
            if existing is None:
                connection.execute("INSERT INTO metadata VALUES('manifest_digest', ?)", (manifest_digest,))
                connection.executemany(
                    "INSERT INTO tasks(task_id,payload_json,payload_digest,seed) VALUES(?,?,?,?)",
                    [(a, b, c, str(d)) for a, b, c, d in rows],
                )
            else:
                if existing[0] != manifest_digest:
                    raise QueueContractError("TASK_MANIFEST_DRIFT")
                actual = connection.execute("SELECT task_id,payload_json,payload_digest,seed FROM tasks ORDER BY task_id").fetchall()
                if [tuple(row) for row in actual] != [(a, b, c, str(d)) for a, b, c, d in rows]:
                    raise QueueContractError("TASK_TABLE_DRIFT")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {"task_count": len(rows), "manifest_digest": manifest_digest}


def claim_task(
    path: Path, *, worker_id: str, device_uuid: str, lease_seconds: int = 180,
    max_attempts: int = 2, now_ns: int | None = None,
) -> dict[str, Any] | None:
    if not worker_id or not device_uuid or lease_seconds <= 0 or max_attempts <= 0:
        raise QueueContractError("INVALID_CLAIM_IDENTITY_OR_LEASE")
    now = time.time_ns() if now_ns is None else int(now_ns)
    lease_until = now + int(lease_seconds * 1_000_000_000)
    with _connect(path.resolve()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            expired = connection.execute(
                "SELECT task_id,attempt FROM tasks WHERE state='claimed' AND lease_until_ns < ?", (now,),
            ).fetchall()
            for row in expired:
                connection.execute(
                    "UPDATE attempts SET status='EXPIRED' WHERE task_id=? AND attempt=? AND status='CLAIMED'",
                    (row["task_id"], row["attempt"]),
                )
            connection.execute(
                "UPDATE tasks SET state='pending',fencing_token=NULL,worker_id=NULL,device_uuid=NULL,"
                "lease_until_ns=NULL,heartbeat_ns=NULL WHERE state='claimed' AND lease_until_ns < ? AND attempt < ?",
                (now, max_attempts),
            )
            connection.execute(
                "UPDATE tasks SET state='failed',fencing_token=NULL,worker_id=NULL,device_uuid=NULL,"
                "lease_until_ns=NULL,heartbeat_ns=NULL WHERE state='claimed' AND lease_until_ns < ? AND attempt >= ?",
                (now, max_attempts),
            )
            row = connection.execute(
                "SELECT task_id,payload_json,seed,attempt FROM tasks WHERE state='pending' ORDER BY task_id LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            attempt = int(row["attempt"]) + 1
            token = attempt
            connection.execute(
                "UPDATE tasks SET state='claimed',attempt=?,fencing_token=?,worker_id=?,device_uuid=?,"
                "lease_until_ns=?,heartbeat_ns=? WHERE task_id=? AND state='pending'",
                (attempt, token, worker_id, device_uuid, lease_until, now, row["task_id"]),
            )
            connection.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,'CLAIMED',NULL)",
                (row["task_id"], attempt, token, worker_id, device_uuid, now, lease_until),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return {
        "task_id": row["task_id"], "payload": json.loads(row["payload_json"]),
        "seed": int(row["seed"]), "attempt": attempt, "fencing_token": token,
        "worker_id": worker_id, "device_uuid": device_uuid, "lease_until_ns": lease_until,
    }


def heartbeat(path: Path, claim: Mapping[str, Any], *, lease_seconds: int = 180, now_ns: int | None = None) -> int:
    now = time.time_ns() if now_ns is None else int(now_ns)
    lease_until = now + int(lease_seconds * 1_000_000_000)
    with _connect(path.resolve()) as connection:
        cursor = connection.execute(
            "UPDATE tasks SET heartbeat_ns=?,lease_until_ns=? WHERE task_id=? AND state='claimed' "
            "AND attempt=? AND fencing_token=? AND worker_id=? AND device_uuid=? AND lease_until_ns>=?",
            (now, lease_until, claim["task_id"], claim["attempt"], claim["fencing_token"],
             claim["worker_id"], claim["device_uuid"], now),
        )
        if cursor.rowcount != 1:
            raise QueueContractError("STALE_FENCING_TOKEN")
    return lease_until


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise QueueContractError("IMMUTABLE_RESULT_EXISTS") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def complete_task(
    path: Path, result_root: Path, claim: Mapping[str, Any], result: Mapping[str, Any],
    *, now_ns: int | None = None,
) -> Path:
    now = time.time_ns() if now_ns is None else int(now_ns)
    result_root = result_root.resolve()
    relative = Path("attempts") / claim["task_id"] / f"attempt-{int(claim['attempt']):02d}.json"
    destination = result_root / relative
    document = {
        "schema_version": 1, "task_id": claim["task_id"], "attempt": int(claim["attempt"]),
        "fencing_token": int(claim["fencing_token"]), "seed": int(claim["seed"]),
        "result": dict(result),
    }
    document["canonical_digest"] = digest(document)
    with _connect(path.resolve()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            current = connection.execute(
                "SELECT state,attempt,fencing_token,worker_id,device_uuid,lease_until_ns FROM tasks WHERE task_id=?",
                (claim["task_id"],),
            ).fetchone()
            valid = current is not None and current["state"] == "claimed" and (
                int(current["attempt"]), int(current["fencing_token"]), current["worker_id"], current["device_uuid"]
            ) == (int(claim["attempt"]), int(claim["fencing_token"]), claim["worker_id"], claim["device_uuid"])
            if not valid or int(current["lease_until_ns"]) < now:
                raise QueueContractError("STALE_FENCING_TOKEN")
            _write_new(destination, document)
            connection.execute(
                "UPDATE tasks SET state='completed',result_path=?,result_digest=? WHERE task_id=?",
                (relative.as_posix(), document["canonical_digest"], claim["task_id"]),
            )
            connection.execute(
                "UPDATE attempts SET status='COMPLETED',completed_ns=? WHERE task_id=? AND attempt=?",
                (now, claim["task_id"], int(claim["attempt"])),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return destination


def audit_coverage(path: Path, result_root: Path) -> dict[str, Any]:
    with _connect(path.resolve()) as connection:
        rows = connection.execute(
            "SELECT task_id,payload_json,payload_digest,seed,state,attempt,result_path,result_digest FROM tasks ORDER BY task_id"
        ).fetchall()
        attempts = connection.execute(
            "SELECT task_id,attempt,status FROM attempts ORDER BY task_id,attempt"
        ).fetchall()
    missing, invalid, completed = [], [], []
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
        signed = dict(document); observed_digest = signed.pop("canonical_digest", None)
        if digest(signed) != observed_digest or observed_digest != row["result_digest"]:
            invalid.append({"task_id": row["task_id"], "code": "RESULT_DIGEST"}); continue
        if document["task_id"] != row["task_id"] or int(document["attempt"]) != int(row["attempt"]):
            invalid.append({"task_id": row["task_id"], "code": "RESULT_IDENTITY"}); continue
        completed.append(row["task_id"])
    accepted_attempts = [(row["task_id"], int(row["attempt"])) for row in attempts if row["status"] == "COMPLETED"]
    duplicates = sorted(task for task in set(completed) if completed.count(task) > 1)
    status = "PASS" if not missing and not invalid and not duplicates and len(accepted_attempts) == len(rows) else "FAIL"
    return {
        "status": status, "task_count": len(rows), "completed_count": len(completed),
        "missing_task_ids": missing, "invalid": invalid, "duplicate_task_ids": duplicates,
        "accepted_attempts": accepted_attempts,
    }
