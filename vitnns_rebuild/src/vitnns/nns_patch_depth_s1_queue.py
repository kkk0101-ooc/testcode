"""Atomic fixed-GPU dynamic unit queue for S1-R1-A2.

Claims are launch-scoped and immutable.  A recovery uses a fresh launch root,
keeps stale claims as evidence, and skips every index that already has an
immutable unit terminal anywhere in the stage root.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


class QueueError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {"schema_version": 1, "document_type": document_type, "study_id": "NNS_VIT_PATCH_DEPTH_SEMANTICS", "stage_id": "S1-R1", "run_id": "s1_r1_pilot_20260914", "status": status, "payload": dict(payload)}
    value["canonical_digest"] = digest(value); return value


def write_exclusive(path: Path, value: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except Exception:
        try: path.unlink()
        except OSError: pass
        raise
    return True


def terminal_index(path: Path) -> int:
    value = json.loads(path.read_text()); payload = value.get("payload", {}); identity = payload.get("identity", {})
    if value.get("document_type") != "s1_unit_terminal" or type(identity.get("schedule_index")) is not int or value.get("status") == "PENDING":
        raise QueueError("malformed immutable unit terminal: " + str(path))
    return int(identity["schedule_index"])


def completed_terminals(stage_root: Path) -> dict[int, Path]:
    patterns = (
        "smoke/shards/shard-*/units/unit-*/completion/unit.json",
        "intermediate/shards/shard-*/units/unit-*/completion/unit.json",
        "full_pilot/shards/shard-*/units/unit-*/completion/unit.json",
        "dynamic_launches/*/dynamic_units/unit-*/attempts/*/completion/unit.json",
    )
    result: dict[int, Path] = {}
    for pattern in patterns:
        for path in sorted(stage_root.glob(pattern)):
            index = terminal_index(path)
            if index in result and result[index].resolve() != path.resolve():
                raise QueueError(f"duplicate immutable terminals for index {index}: {result[index]} and {path}")
            result[index] = path
    return result


def claim_next(launch_root: Path, candidates: Sequence[int], stage_root: Path, worker: Mapping[str, Any], source_by_index: Mapping[int, Mapping[str, Any]], now_ns: int | None = None) -> dict[str, Any] | None:
    completed = completed_terminals(stage_root)
    for index in sorted(set(int(item) for item in candidates)):
        if index in completed:
            continue
        if index not in source_by_index:
            raise QueueError(f"candidate index absent from schedule: {index}")
        source = source_by_index[index]
        claimed_ns = time.time_ns() if now_ns is None else int(now_ns)
        payload = {
            "launch_id": launch_root.name, "schedule_index": index,
            "source_image_id": source["source_image_id"], "image_sha256": source["image_sha256"],
            "worker_id": worker["worker_id"], "worker_pid": int(worker["worker_pid"]),
            "physical_gpu_index": int(worker["physical_gpu_index"]), "gpu_uuid": worker["gpu_uuid"],
            "claimed_unix_ns": claimed_ns, "claim_scope": "THIS_LAUNCH_ONLY",
            "scientific_seed_depends_on_worker_or_gpu": False,
        }
        value = signed("s1_r1_a2_unit_claim", "CLAIMED", payload)
        path = launch_root / "claims" / f"unit-{index:04d}.claim.json"
        if write_exclusive(path, value):
            return {**payload, "claim_path": path.as_posix(), "claim_canonical_digest": value["canonical_digest"]}
    return None


def missing_indices(stage_root: Path, candidates: Sequence[int]) -> list[int]:
    completed = completed_terminals(stage_root)
    return [index for index in sorted(set(int(item) for item in candidates)) if index not in completed]


def aggregate_unit_resources(stage_root: Path, candidates: Sequence[int]) -> dict[str, Any]:
    terminals = completed_terminals(stage_root); selected = []
    for index in sorted(set(int(item) for item in candidates)):
        path = terminals.get(index)
        if path is None:
            continue
        value = json.loads(path.read_text()); resources = value.get("payload", {}).get("resources")
        if resources is not None:
            selected.append((index, path.as_posix(), resources))
    return {
        "accounting_unit": "unit_terminal", "unit_count": len(selected),
        "gpu_active_seconds": math.fsum(float(item[2]["gpu_active_seconds"]) for item in selected),
        "cpu_seconds": math.fsum(float(item[2]["cpu_seconds"]) for item in selected),
        "wall_seconds_sum": math.fsum(float(item[2]["wall_seconds"]) for item in selected),
        "retained_bytes": sum(int(item[2]["retained_bytes"]) for item in selected),
        "peak_reserved_gpu_bytes": max((int(item[2]["peak_reserved_gpu_bytes"]) for item in selected), default=0),
        "unit_records": [{"schedule_index": item[0], "terminal_path": item[1]} for item in selected],
        "static_shard_inference_used": False,
    }
