#!/usr/bin/env python3
"""Validate immutable digests and current Step-5.1 gate coverage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def verify(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest", None)
    if claimed is None or hashlib.sha256(canonical(value)).hexdigest() != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 2: raise SystemExit("usage: validate_step5_1.py RUN_ROOT")
    run = Path(sys.argv[1]).resolve()
    protocol = verify(run / "protocol.json"); manifest = verify(run / "task_manifest.json")
    source = verify(run / "source_binding.json"); freeze = verify(run / "freeze_receipt.json")
    if protocol["stage"] != "ENDPOINT_PCIA_PEFA_STEP5_1" or protocol["evaluation_access"]:
        raise RuntimeError("STAGE_OR_DATA_BOUNDARY")
    if manifest["evaluation_access"] or source["evaluation_member_reads"] != 0 or freeze["step6_started"]:
        raise RuntimeError("EVALUATION_OR_STEP6_BOUNDARY")
    if len(manifest["a0_tasks"]) != 2 or len(manifest["a1_tasks"]) != 16:
        raise RuntimeError("TASK_COVERAGE")
    orders = sorted({int(row["split_order"]) for row in manifest["a1_tasks"]})
    if orders != list(range(8, 16)) or any(row["split"] != "calibration" for row in manifest["a1_tasks"]):
        raise RuntimeError("A1_SELECTION")
    repo = Path(__file__).resolve().parent.parent
    for row in (*source["files"], *source["checkpoints"], *source["images"]):
        path = repo / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError("SOURCE_BINDING:" + row["path"])
    status = "PASS_STEP5_1_FREEZE"
    if (run / "a0_decision.json").exists():
        decision = verify(run / "a0_decision.json")
        status = "PASS_STEP5_1_A0" if decision["status"] == "PASS_A0_NUMERICAL_FEASIBILITY" else "INVALID_STEP5_1_A0"
    if (run / "terminal_receipt.json").exists():
        terminal = verify(run / "terminal_receipt.json"); decisions = verify(run / "claim_decisions.json")
        analysis = verify(run / "a1_analysis.json"); resources = verify(run / "resource_ledger.json")
        evidence = verify(run / "evidence_index.json"); c1 = verify(run / "c1_result.json")
        expected = {"CONTROLLED_CURVATURE_LOCALIZATION", "NATURAL_NONLINEARITY_LOCALIZATION",
                    "ATTRIBUTION_ERROR_LOCALIZATION", "INCREMENTAL_OVER_SIMPLE_BASELINES",
                    "NUMERICAL_VALIDITY_AND_COST"}
        if {row["claim_id"] for row in decisions["decisions"]} != expected:
            raise RuntimeError("CLAIM_COVERAGE")
        if terminal["claim_decisions_digest"] != decisions["canonical_digest"] or terminal["analysis_digest"] != analysis["canonical_digest"] or terminal["resource_ledger_digest"] != resources["canonical_digest"] or terminal["evidence_index_digest"] != evidence["canonical_digest"]:
            raise RuntimeError("TERMINAL_LINKAGE")
        if c1["claim_id"] != "ATTRIBUTION_ERROR_LOCALIZATION" or terminal["evaluation_access"] or terminal["step6_started"]:
            raise RuntimeError("TERMINAL_BOUNDARY")
        status = "PASS_STEP5_1_TERMINAL_COVERAGE"
    print(json.dumps({"status": status, "protocol_digest": protocol["canonical_digest"],
                      "a0_task_count": 2, "a1_task_count": 16, "evaluation_access": False,
                      "step6_started": False}, sort_keys=True))
    return 0 if status.startswith("PASS") else 3


if __name__ == "__main__": raise SystemExit(main())
