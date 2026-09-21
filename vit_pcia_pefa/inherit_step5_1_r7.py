#!/usr/bin/env python3
"""Bind and copy sealed r6 evidence into reporting-only r7."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "step5_1_r6_20260921"
TARGET = ROOT / "step5_1_r7_20260921"
NAMES = (
    "a0_fixture_result.json", "a0_startup_snapshot.json", "a0_launch_terminal.json",
    "a0/deit_s16.json", "a0/vit_b16.json", "a0_decision.json", "c1_result.json",
    "a1_startup_snapshot.json", "a1_launch_terminal.json", "a1_queue.sqlite3",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def verify(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: dict[str, Any]) -> None:
    body = dict(value)
    body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(body, sort_keys=True, indent=2,
                                allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    protocol = verify(TARGET / "protocol.json")
    predecessor = verify(SOURCE / "terminal_receipt.json")
    if (protocol["run_id"] != TARGET.name
            or protocol["status"] != "REPORTING_ONLY_RECOVERY_FROZEN"
            or predecessor["status"] != "NOT_SUPPORTED_INCREMENTAL_OVER_SIMPLE_BASELINES"):
        raise RuntimeError("R6_NOT_INHERITABLE")
    records = []
    for name in NAMES:
        source = SOURCE / name
        destination = TARGET / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".json"):
            verify(source)
        if destination.exists():
            raise RuntimeError("INHERIT_DESTINATION_EXISTS:" + name)
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError("INHERIT_COPY_DIGEST:" + name)
        records.append({"path": name, "bytes": destination.stat().st_size,
                        "sha256": sha256(destination)})
    source_results = SOURCE / "a1_results"
    target_results = TARGET / "a1_results"
    shutil.copytree(source_results, target_results)
    for path in sorted(target_results.rglob("*.json")):
        verify(path)
        source = source_results / path.relative_to(target_results)
        if sha256(source) != sha256(path):
            raise RuntimeError("RESULT_COPY_DIGEST")
        records.append({"path": path.relative_to(TARGET).as_posix(),
                        "bytes": path.stat().st_size, "sha256": sha256(path)})
    with sqlite3.connect(TARGET / "a1_queue.sqlite3") as connection:
        states = dict(connection.execute(
            "SELECT state,count(*) FROM tasks GROUP BY state").fetchall())
        attempts = dict(connection.execute(
            "SELECT status,count(*) FROM attempts GROUP BY status").fetchall())
    if states != {"completed": 16} or attempts != {"COMPLETED": 16}:
        raise RuntimeError("QUEUE_NOT_COMPLETE")
    write_new(TARGET / "inheritance_receipt.json", {
        "schema_version": 1, "run_id": TARGET.name,
        "status": "PASS_ALL_RESULT_INHERITANCE", "source_root": SOURCE.name,
        "source_terminal_digest": predecessor["canonical_digest"],
        "artifacts": records, "accepted_a1_results": 16,
        "outcomes_known": True, "reporting_only": True,
        "completed_unix_ns": time.time_ns(), "evaluation_access": False,
        "b1_started": False, "step6_started": False,
    })
    shutil.copyfile(SOURCE / "a0_inheritance_receipt.json",
                    TARGET / "a0_inheritance_receipt.json")
    print(json.dumps({"status": "PASS_ALL_RESULT_INHERITANCE",
                      "artifact_count": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
