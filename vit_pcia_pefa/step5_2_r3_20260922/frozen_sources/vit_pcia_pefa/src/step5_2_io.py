"""Small fail-closed I/O helpers shared by Step 5.2 executables."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import time
from typing import Any, Mapping

import numpy as np


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value); body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest(); return body


def verify(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest", None)
    if claimed is None or hashlib.sha256(canonical(value)).hexdigest() != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def write_new(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    body = signed(value); payload = json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return body


def identity(path: Path, repo: Path, mode: str) -> dict[str, Any]:
    return {"path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "mode": mode}


def verify_binding(binding: Mapping[str, Any], repo: Path) -> None:
    for row in (*binding["files"], *binding["checkpoints"], *binding["images"]):
        path = repo / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError("FAIL_SOURCE_BINDING:" + row["path"])


def rng_state(torch_module, device) -> dict[str, Any]:
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"bit_generator": state[0], "keys": state[1].tolist(), "position": int(state[2]),
                      "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
            "torch_cpu_base64": base64.b64encode(torch_module.get_rng_state().cpu().numpy().tobytes()).decode(),
            "torch_cuda_base64": base64.b64encode(torch_module.cuda.get_rng_state(device).cpu().numpy().tobytes()).decode()}


def abandon(queue: Path, claim: Mapping[str, Any], error: Mapping[str, Any], max_attempts: int) -> None:
    now = time.time_ns()
    with sqlite3.connect(queue, timeout=30) as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = connection.execute(
            "SELECT state,attempt,fencing_token,worker_id,device_uuid FROM tasks WHERE task_id=?",
            (claim["task_id"],),
        ).fetchone()
        valid = current is not None and current[0] == "claimed" and tuple(current[1:]) == (
            claim["attempt"], claim["fencing_token"], claim["worker_id"], claim["device_uuid"])
        if not valid:
            raise RuntimeError("STALE_FENCING_TOKEN")
        terminal = int(claim["attempt"]) >= int(max_attempts)
        connection.execute("UPDATE attempts SET status=?,completed_ns=? WHERE task_id=? AND attempt=?",
                           ("FAILED" if terminal else "RETRYABLE", now, claim["task_id"], claim["attempt"]))
        connection.execute("UPDATE tasks SET state=?,fencing_token=NULL,worker_id=NULL,device_uuid=NULL,"
                           "lease_until_ns=NULL,heartbeat_ns=NULL WHERE task_id=?",
                           ("failed" if terminal else "pending", claim["task_id"]))
        connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                           (f"failure:{claim['task_id']}:{claim['attempt']}", json.dumps(error, sort_keys=True)))
        connection.commit()
