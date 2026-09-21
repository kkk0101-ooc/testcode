#!/usr/bin/env python3
"""Create the final immutable Step-5 completion seal."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = ROOT / "step5_completion_20260921"
INTERRUPTED_VIT_B_FLOAT64_DIAGNOSIS_H20_HOURS_UPPER_BOUND = 0.2


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value); value["canonical_digest"] = digest(value); return value


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()); claimed = value.pop("canonical_digest")
    if digest(value) != claimed:
        raise RuntimeError("CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed; return value


def identity(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("STEP5_COMPLETION_EXISTS")
    r8 = ROOT / "step5_pilot_r8_20260921"
    r2 = ROOT / "step5_extension_r2_20260921"
    r3 = ROOT / "step5_extension_r3_20260921"
    seal = ROOT / "step5_extension_seal_20260921"
    r8_terminal = verify_signed(r8 / "terminal_receipt.json")
    r2_terminal = verify_signed(r2 / "terminal_receipt.json")
    r3_terminal = verify_signed(r3 / "terminal_receipt.json")
    extension_terminal = verify_signed(seal / "terminal_receipt.json")
    extension_claims = verify_signed(seal / "claim_decisions.json")
    extension_ledger = verify_signed(seal / "resource_ledger.json")
    float64_deit = verify_signed(ROOT / "step5_extension_r3_float64_diagnosis_deit_s16.json")
    if r8_terminal["status"] != "NOT_SUPPORTED_EMPIRICAL_VALUE":
        raise RuntimeError("R8_STATUS")
    if r2_terminal["status"] != "NOT_SUPPORTED" or r3_terminal["status"] != "NOT_SUPPORTED":
        raise RuntimeError("EXTENSION_ROOT_STATUS")
    if extension_terminal["status"] != "PARTIAL_NOT_SUPPORTED_WITH_E3_INVALID":
        raise RuntimeError("EXTENSION_SEAL_STATUS")
    if float64_deit["model_dtype"] != "float64" or float64_deit["scientific_rank_claim_evaluated"]:
        raise RuntimeError("FLOAT64_DIAGNOSIS")
    cumulative = (extension_ledger["cumulative_step5_h20_hours"]
                  + float64_deit["resources"]["h20_hours"]
                  + INTERRUPTED_VIT_B_FLOAT64_DIAGNOSIS_H20_HOURS_UPPER_BOUND)
    claims = signed({
        "schema_version": 1, "run_id": "step5_completion_20260921",
        "decisions": [
            {"claim_id": "ORIGINAL_STEP5_EMPIRICAL_VALUE", "decision": "NOT_SUPPORTED",
             "evidence_root": "step5_pilot_r8_20260921"},
            {"claim_id": "E1_LOCAL_CURL_PREDICTS_LOOP_CIRCULATION", "decision": "NOT_SUPPORTED",
             "evidence_root": "step5_extension_r2_20260921"},
            {"claim_id": "E2_NONLINEAR_CONDITION_AMPLIFIES_VALID_CURVATURE_EFFECT", "decision": "NOT_SUPPORTED",
             "evidence_root": "step5_extension_r2_20260921"},
            {"claim_id": "E3A_SOURCE_RESOLUTION_STAIRCASE", "decision": "INVALID",
             "reason": "INVALID_SOURCE_INTEGRAL_CONVERGENCE"},
            {"claim_id": "STEP5_COST_AFFORDABLE", "decision": "SUPPORTED"},
        ],
        "mainline_position": "ENDPOINT_EXTENSION_ONLY",
        "evaluation_access": False, "step6_production_authorized": False,
        "advance_to_14x14": False, "pixel_level_authorized": False,
    })
    ledger = signed({
        "schema_version": 1, "run_id": "step5_completion_20260921",
        "through_extension_seal_h20_hours": extension_ledger["cumulative_step5_h20_hours"],
        "completed_float64_deit_diagnosis_h20_hours": float64_deit["resources"]["h20_hours"],
        "interrupted_float64_vit_b_diagnosis_h20_hours_upper_bound": INTERRUPTED_VIT_B_FLOAT64_DIAGNOSIS_H20_HOURS_UPPER_BOUND,
        "interrupted_diagnosis_scientific_result_used": False,
        "cumulative_step5_h20_hours_upper_bound": cumulative, "cap_h20_hours": 64,
        "retained_bytes": sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file()),
        "retained_bytes_cap": 128 * 1024**3,
        "status": "SUPPORTED" if cumulative <= 64 else "INVALID",
    })
    evidence_paths = [
        ROOT / "执行规划.md", ROOT / "method_contract.md", ROOT / "step5_extension_protocol.md",
        r8 / "terminal_receipt.json", r8 / "claim_decisions.json", r8 / "resource_ledger.json",
        r8 / "validation_report.json", r2 / "terminal_receipt.json", r2 / "analysis.json",
        r2 / "claim_decisions.json", r2 / "resource_ledger.json", r2 / "validation_report.json",
        r3 / "terminal_receipt.json", r3 / "analysis.json", r3 / "claim_decisions.json",
        r3 / "resource_ledger.json", r3 / "validation_report.json",
        seal / "terminal_receipt.json", seal / "claim_decisions.json",
        seal / "resource_ledger.json", seal / "validation_report.json",
        ROOT / "step5_extension_r3_float64_diagnosis_deit_s16.json",
    ]
    evidence = signed({"schema_version": 1, "run_id": "step5_completion_20260921",
                       "files": [identity(path) for path in evidence_paths],
                       "interrupted_attempt": {
                           "task": "step5_extension_r3_float64_diagnosis_vit_b16",
                           "status": "STOPPED_ON_USER_REQUEST", "artifact_written": False,
                           "scientific_result_used": False,
                           "resource_upper_bound_h20_hours": INTERRUPTED_VIT_B_FLOAT64_DIAGNOSIS_H20_HOURS_UPPER_BOUND,
                       }})
    terminal = signed({
        "schema_version": 1, "run_id": "step5_completion_20260921",
        "stage": "ENDPOINT_PCIA_PEFA_STEP5",
        "status": "STEP5_COMPLETE_NOT_SUPPORTED_WITH_E3_INVALID",
        "completed_unix_ns": time.time_ns(), "claim_decisions_digest": claims["canonical_digest"],
        "resource_ledger_digest": ledger["canonical_digest"],
        "evidence_index_digest": evidence["canonical_digest"],
        "original_r8_decision_unchanged": "NOT_SUPPORTED_EMPIRICAL_VALUE",
        "evaluation_access": False, "next_authorization": "NONE",
        "all_step5_gpu_processes_stopped": True, "step6_started": False,
    })
    OUTPUT.mkdir(parents=True)
    write_new(OUTPUT / "claim_decisions.json", claims)
    write_new(OUTPUT / "resource_ledger.json", ledger)
    write_new(OUTPUT / "evidence_index.json", evidence)
    write_new(OUTPUT / "terminal_receipt.json", terminal)
    print(json.dumps({"status": terminal["status"], "claims": claims["decisions"],
                      "cumulative_h20_hours_upper_bound": cumulative,
                      "step6_started": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
