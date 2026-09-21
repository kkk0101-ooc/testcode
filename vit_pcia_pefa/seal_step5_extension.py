#!/usr/bin/env python3
"""Seal the cross-root Step-5 extension conclusion after diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = ROOT / "step5_extension_seal_20260921"


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


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("SEAL_EXISTS")
    r8 = ROOT / "step5_pilot_r8_20260921"
    r1 = ROOT / "step5_extension_r1_20260921"
    r2 = ROOT / "step5_extension_r2_20260921"
    r3 = ROOT / "step5_extension_r3_20260921"
    terminals = {name: verify_signed(path / "terminal_receipt.json")
                 for name, path in (("r8", r8), ("r1", r1), ("r2", r2), ("r3", r3))}
    validations = {name: verify_signed(path / "validation_report.json")
                   for name, path in (("r8", r8), ("r1", r1), ("r2", r2), ("r3", r3))}
    if terminals["r8"]["status"] != "NOT_SUPPORTED_EMPIRICAL_VALUE":
        raise RuntimeError("R8_STATUS")
    if terminals["r1"]["status"] != "INVALID":
        raise RuntimeError("R1_STATUS")
    if terminals["r2"]["status"] != "NOT_SUPPORTED":
        raise RuntimeError("R2_STATUS")
    if terminals["r3"]["status"] != "NOT_SUPPORTED":
        raise RuntimeError("R3_PRE_DIAGNOSIS_STATUS")
    if any(report.get("step6_started") for report in validations.values()):
        raise RuntimeError("STEP6_STARTED")
    r2_claims = verify_signed(r2 / "claim_decisions.json")
    if [row["decision"] for row in r2_claims["decisions"]] != ["NOT_SUPPORTED", "NOT_SUPPORTED"]:
        raise RuntimeError("R2_CLAIMS")
    diagnosis_paths = [
        ROOT / "step5_extension_r3_source_diagnosis_deit_s16.json",
        ROOT / "step5_extension_r3_source_diagnosis_vit_b16.json",
        ROOT / "step5_extension_r3_jvp_diagnosis_deit_s16.json",
        ROOT / "step5_extension_r3_jvp_diagnosis_vit_b16.json",
    ]
    diagnoses = [verify_signed(path) for path in diagnosis_paths]
    for diagnosis in diagnoses:
        if diagnosis["status"] != "DIAGNOSIS_COMPLETE" or not diagnosis["scientific_rank_claim_evaluated"] is False:
            raise RuntimeError("SOURCE_DIAGNOSIS_STATUS")
    terminal_errors = {
        diagnosis["task_id"]: {str(row["nodes"]): row["terminal_T_convergence"]
                                 for row in diagnosis["rows"]}
        for diagnosis in diagnoses[:2]
    }
    if any(terminal_errors[task]["512"]["relative_l2"] <= 1e-4 for task in terminal_errors):
        raise RuntimeError("SOURCE_CONVERGENCE_UNEXPECTED_PASS")
    r3_ledger = verify_signed(r3 / "resource_ledger.json")
    diagnosis_hours = sum(row["resources"]["h20_hours"] for row in diagnoses)
    cumulative_hours = r3_ledger["cumulative_step5_h20_hours"] + diagnosis_hours
    decisions = signed({
        "schema_version": 1, "run_id": "step5_extension_seal_20260921",
        "decisions": [
            {"claim_id": "E1_LOCAL_CURL_PREDICTS_LOOP_CIRCULATION", "decision": "NOT_SUPPORTED",
             "evidence_root": "step5_extension_r2_20260921"},
            {"claim_id": "E2_NONLINEAR_CONDITION_AMPLIFIES_VALID_CURVATURE_EFFECT", "decision": "NOT_SUPPORTED",
             "evidence_root": "step5_extension_r2_20260921"},
            {"claim_id": "E3A_7X7_PEFA_INCREMENTAL_RESOLUTION_VALUE", "decision": "INVALID",
             "reason": "INVALID_SOURCE_INTEGRAL_CONVERGENCE",
             "supersedes_only": "step5_extension_r3_20260921 post-hoc scientific interpretation; artifacts remain immutable"},
        ],
        "original_r8_decision_unchanged": "NOT_SUPPORTED_EMPIRICAL_VALUE",
        "confirmatory_claim_allowed": False, "step6_production_authorized": False,
        "advance_to_14x14": False, "pixel_level_authorized": False,
    })
    ledger = signed({
        "schema_version": 1, "run_id": "step5_extension_seal_20260921",
        "through_r3_h20_hours": r3_ledger["cumulative_step5_h20_hours"],
        "post_r3_source_diagnosis_h20_hours": diagnosis_hours,
        "cumulative_step5_h20_hours": cumulative_hours, "cap_h20_hours": 64,
        "status": "SUPPORTED" if cumulative_hours <= 64 else "INVALID",
    })
    evidence_paths = [
        ROOT / "step5_extension_protocol.md", ROOT / "step5_extension_r2_recovery_amendment.md",
        r8 / "terminal_receipt.json", r8 / "validation_report.json",
        r2 / "analysis.json", r2 / "claim_decisions.json", r2 / "terminal_receipt.json",
        r2 / "validation_report.json", r2 / "resource_ledger.json",
        r3 / "analysis.json", r3 / "claim_decisions.json", r3 / "terminal_receipt.json",
        r3 / "validation_report.json", r3 / "resource_ledger.json", *diagnosis_paths,
    ]
    evidence = signed({"schema_version": 1, "run_id": "step5_extension_seal_20260921",
                       "files": [identity(path) for path in evidence_paths]})
    terminal = signed({
        "schema_version": 1, "run_id": "step5_extension_seal_20260921",
        "status": "PARTIAL_NOT_SUPPORTED_WITH_E3_INVALID", "completed_unix_ns": time.time_ns(),
        "claim_decisions_digest": decisions["canonical_digest"],
        "resource_ledger_digest": ledger["canonical_digest"],
        "evidence_index_digest": evidence["canonical_digest"],
        "original_r8_decision_unchanged": "NOT_SUPPORTED_EMPIRICAL_VALUE",
        "next_authorization": "NONE", "step6_started": False,
    })
    OUTPUT.mkdir(parents=True)
    write_new(OUTPUT / "claim_decisions.json", decisions)
    write_new(OUTPUT / "resource_ledger.json", ledger)
    write_new(OUTPUT / "evidence_index.json", evidence)
    write_new(OUTPUT / "terminal_receipt.json", terminal)
    print(json.dumps({"status": terminal["status"], "decisions": decisions["decisions"],
                      "cumulative_step5_h20_hours": cumulative_hours, "step6_started": False},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
