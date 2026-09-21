#!/usr/bin/env python3
"""Bind and copy the validated r2 A0 artifacts into the prospective r3 root."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "step5_1_r2_20260921"
TARGET = ROOT / "step5_1_r3_20260921"
NAMES = ("a0_fixture_result.json", "a0_startup_snapshot.json", "a0_launch_terminal.json",
         "a0/deit_s16.json", "a0/vit_b16.json", "a0_decision.json")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed: raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: dict[str, Any]) -> None:
    body = dict(value); body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest()
    payload = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    protocol = verify(TARGET / "protocol.json")
    expected = ["decoy_port_same_strength_0.5", "decoy_port_same_strength_1.0",
                "linear_tangent_1.0", "zero_strength"]
    if protocol["run_id"] != TARGET.name or protocol["wrapper"]["controls"] != expected:
        raise RuntimeError("R3_PROTOCOL_NOT_CLARIFIED")
    decision = verify(SOURCE / "a0_decision.json")
    if decision["status"] != "PASS_A0_NUMERICAL_FEASIBILITY" or not decision["a1_authorized_by_gate"]:
        raise RuntimeError("R2_A0_NOT_INHERITABLE")
    records = []
    for name in NAMES:
        source = SOURCE / name; verify(source)
        destination = TARGET / name; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): raise RuntimeError("INHERIT_DESTINATION_EXISTS:" + name)
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination): raise RuntimeError("INHERIT_COPY_DIGEST:" + name)
        records.append({"path": name, "bytes": destination.stat().st_size, "sha256": sha256(destination)})
    write_new(TARGET / "a0_inheritance_receipt.json", {
        "schema_version": 1, "run_id": TARGET.name, "status": "PASS_A0_INHERITANCE",
        "source_root": SOURCE.name, "source_a0_decision_digest": decision["canonical_digest"],
        "artifacts": records, "a0_recomputed": False, "a0_gate_unchanged": True,
        "completed_unix_ns": time.time_ns(), "evaluation_access": False, "step6_started": False,
    })
    print(json.dumps({"status": "PASS_A0_INHERITANCE", "artifact_count": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
