#!/usr/bin/env python3
"""Verify and inherit the 14 canonical-valid r7 task results into r8."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_queue import create_queue, digest  # noqa: E402

R7 = ROOT / "step5_pilot_r7_20260921"
R8 = ROOT / "step5_pilot_r8_20260921"


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    claimed = value.pop("canonical_digest")
    if canonical(value) != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed
    return value


def write_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    manifest = signed(R8 / "task_manifest.json")
    freeze = signed(R8 / "freeze_receipt.json")
    if freeze["outcome_access_during_execution"] is not False:
        raise RuntimeError("OUTCOME_ACCESS_CONTRACT")
    failure = json.loads((R7 / "failure_receipt.json").read_text())
    if failure["status"] != "TERMINAL_MATHEMATICAL_CLOSURE_FAIL":
        raise RuntimeError("R7_NOT_TERMINAL_CLOSURE_FAIL")
    queue = R8 / "queue.sqlite3"
    info = create_queue(queue, manifest["tasks"])
    source = sqlite3.connect(R7 / "queue.sqlite3")
    target = sqlite3.connect(queue)
    inherited = []
    for task_id, attempt, relative, result_digest in source.execute(
        "SELECT task_id,attempt,result_path,result_digest FROM tasks WHERE state='completed' ORDER BY task_id"
    ):
        source_path = R7 / "results" / relative
        document = signed(source_path)
        if (document["canonical_digest"] != result_digest or document["task_id"] != task_id
                or document["result"]["status"] != "PASS_TASK"):
            raise RuntimeError("R7_RESULT_INVALID:" + task_id)
        destination = R8 / "results" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        if signed(destination)["canonical_digest"] != result_digest:
            raise RuntimeError("COPY_DIGEST:" + task_id)
        target.execute(
            "UPDATE tasks SET state='completed',attempt=?,result_path=?,result_digest=? WHERE task_id=?",
            (attempt, relative, result_digest, task_id),
        )
        target.execute(
            "INSERT INTO attempts(task_id,attempt,fencing_token,worker_id,device_uuid,claimed_ns,lease_until_ns,status,completed_ns) VALUES(?,?,?,?,?,?,?,?,?)",
            (task_id, attempt, attempt, "INHERITED_R7", "INHERITED_R7", 0, 0, "COMPLETED", 0),
        )
        inherited.append({
            "task_id": task_id, "attempt": attempt,
            "source_path": source_path.relative_to(REPO).as_posix(),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "canonical_digest": result_digest,
        })
    target.commit(); target.close(); source.close()
    expected_missing = ["vit_b16__02", "vit_b16__05"]
    if len(inherited) != 14 or sorted(set(row["task_id"] for row in inherited) & set(expected_missing)):
        raise RuntimeError("INHERITANCE_SCOPE")
    receipt = {
        "schema_version": 1, "run_id": manifest["run_id"],
        "status": "PASS_INHERITANCE", "source_run_id": "step5_pilot_r7_20260921",
        "source_status": failure["status"],
        "source_failure_receipt_sha256": hashlib.sha256((R7 / "failure_receipt.json").read_bytes()).hexdigest(),
        "inherited_task_ids": [row["task_id"] for row in inherited],
        "records": inherited, "queue_manifest_digest": info["manifest_digest"],
        "scientific_values_changed": False, "missing_tasks": expected_missing,
        "omega_float64_audit_required_for_all_tasks": True,
    }
    receipt["canonical_digest"] = digest(receipt)
    write_new(R8 / "inheritance_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "inherited": 14, "missing": expected_missing}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
