#!/usr/bin/env python3
"""Run the frozen C1 held-out port-ID mutation audit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(REPO), str(REPO / "vitnns_rebuild/src")]
from vit_pcia_pefa.src.step5_1_localization import audit_head_port_semantics  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if hashlib.sha256(canonical(value)).hexdigest() != claimed: raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    body = dict(value); body["canonical_digest"] = hashlib.sha256(canonical(body)).hexdigest()
    payload = json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: run_step5_1_c1.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve(); protocol = verify(run / "protocol.json")
    inherited = verify(run / "a0_inheritance_receipt.json")
    if inherited["status"] != "PASS_A0_INHERITANCE": raise RuntimeError("C1_A0_GATE")
    registry = json.loads((ROOT / "vit_port_registry.json").read_text())
    rows = []
    for model_id in protocol["models"]:
        original = next(row for row in registry["cuts"] if row["model_id"] == model_id
                        and row["block_index"] == 5 and row["cut_id"] == "HEADS_WITH_BYPASS_FULL")
        valid = audit_head_port_semantics(original)
        mutated = {**original, "port_ids": [dict(port) for port in original["port_ids"]]}
        slices = [port["slice_spec"] for port in mutated["port_ids"][1:]]
        for port, rotated in zip(mutated["port_ids"][1:], slices[1:] + slices[:1]):
            port["slice_spec"] = rotated
        detected = audit_head_port_semantics(mutated)
        rows.append({"model_id": model_id, "development_control_original_pass": valid,
                     "held_out_mutation": "consumer_head_slice_rotation",
                     "ordinary_output_unchanged_by_registry_only_mutation": True,
                     "ordinary_ig_completeness_unchanged_by_registry_only_mutation": True,
                     "endpoint_single_point_unchanged_by_registry_only_mutation": True,
                     "contract_detection": detected})
    supported = all(row["development_control_original_pass"]["pass"]
                    and not row["contract_detection"]["pass"]
                    and row["contract_detection"]["block_index"] == 5
                    and row["contract_detection"]["cut_id"] == "HEADS_WITH_BYPASS_FULL"
                    and row["contract_detection"]["category"] == "head"
                    and {error["code"] for error in row["contract_detection"]["errors"]}
                        == {"FAIL_PORT_ID_SEMANTIC_MISMATCH"} for row in rows)
    write_new(run / "c1_result.json", {"schema_version": 1, "run_id": protocol["run_id"],
        "status": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "claim_id": "ATTRIBUTION_ERROR_LOCALIZATION", "rows": rows,
        "natural_mechanism_claim": False, "evaluation_access": False,
        "step6_started": False, "completed_unix_ns": time.time_ns()})
    print(json.dumps({"status": "SUPPORTED" if supported else "NOT_SUPPORTED",
                      "claim_id": "ATTRIBUTION_ERROR_LOCALIZATION"}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
