#!/usr/bin/env python3
"""CPU-only independent audit of the immutable Step-5 completion seal."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SEAL = ROOT / "step5_completion_20260921"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value)
    value["canonical_digest"] = digest(value)
    return value


def verify_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    claimed = value.pop("canonical_digest")
    require(digest(value) == claimed, "CANONICAL_DIGEST:" + str(path))
    value["canonical_digest"] = claimed
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "",
            "VALIDATOR_MUST_BE_CPU_ONLY")
    output = SEAL / "validation_report.json"
    require(not output.exists(), "VALIDATION_ALREADY_EXISTS")

    claims = verify_signed(SEAL / "claim_decisions.json")
    ledger = verify_signed(SEAL / "resource_ledger.json")
    evidence = verify_signed(SEAL / "evidence_index.json")
    terminal = verify_signed(SEAL / "terminal_receipt.json")

    require(terminal["status"] == "STEP5_COMPLETE_NOT_SUPPORTED_WITH_E3_INVALID",
            "TERMINAL_STATUS")
    require(terminal["claim_decisions_digest"] == claims["canonical_digest"],
            "CLAIMS_BINDING")
    require(terminal["resource_ledger_digest"] == ledger["canonical_digest"],
            "LEDGER_BINDING")
    require(terminal["evidence_index_digest"] == evidence["canonical_digest"],
            "EVIDENCE_BINDING")

    expected = {
        "ORIGINAL_STEP5_EMPIRICAL_VALUE": "NOT_SUPPORTED",
        "E1_LOCAL_CURL_PREDICTS_LOOP_CIRCULATION": "NOT_SUPPORTED",
        "E2_NONLINEAR_CONDITION_AMPLIFIES_VALID_CURVATURE_EFFECT": "NOT_SUPPORTED",
        "E3A_SOURCE_RESOLUTION_STAIRCASE": "INVALID",
        "STEP5_COST_AFFORDABLE": "SUPPORTED",
    }
    observed = {row["claim_id"]: row["decision"] for row in claims["decisions"]}
    require(observed == expected, "CLAIM_DECISIONS")
    e3 = next(row for row in claims["decisions"]
              if row["claim_id"] == "E3A_SOURCE_RESOLUTION_STAIRCASE")
    require(e3["reason"] == "INVALID_SOURCE_INTEGRAL_CONVERGENCE", "E3_REASON")
    require(claims["mainline_position"] == "ENDPOINT_EXTENSION_ONLY",
            "MAINLINE_POSITION")
    require(not claims["evaluation_access"] and
            not claims["advance_to_14x14"] and
            not claims["pixel_level_authorized"] and
            not claims["step6_production_authorized"], "STAGE_BOUNDARY")
    require(not terminal["evaluation_access"] and
            terminal["next_authorization"] == "NONE" and
            not terminal["step6_started"], "TERMINAL_BOUNDARY")
    require(terminal["original_r8_decision_unchanged"] ==
            "NOT_SUPPORTED_EMPIRICAL_VALUE", "R8_BOUNDARY")

    for row in evidence["files"]:
        path = REPO / row["path"]
        require(path.is_file(), "EVIDENCE_MISSING:" + row["path"])
        require(path.stat().st_size == row["bytes"], "EVIDENCE_SIZE:" + row["path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"],
                "EVIDENCE_HASH:" + row["path"])

    extension = verify_signed(ROOT / "step5_extension_seal_20260921/resource_ledger.json")
    float64_deit = verify_signed(
        ROOT / "step5_extension_r3_float64_diagnosis_deit_s16.json")
    interrupted = evidence["interrupted_attempt"]
    require(interrupted["status"] == "STOPPED_ON_USER_REQUEST" and
            not interrupted["artifact_written"] and
            not interrupted["scientific_result_used"], "INTERRUPTED_ATTEMPT")
    require(not (ROOT / "step5_extension_r3_float64_diagnosis_vit_b16.json").exists(),
            "INTERRUPTED_ARTIFACT_PRESENT")
    recomputed = math.fsum([
        extension["cumulative_step5_h20_hours"],
        float64_deit["resources"]["h20_hours"],
        interrupted["resource_upper_bound_h20_hours"],
    ])
    require(math.isclose(recomputed, ledger["cumulative_step5_h20_hours_upper_bound"],
                         rel_tol=0, abs_tol=1e-12), "RESOURCE_RECOMPUTATION")
    require(ledger["status"] == "SUPPORTED" and recomputed <= ledger["cap_h20_hours"],
            "RESOURCE_CAP")
    require(not ledger["interrupted_diagnosis_scientific_result_used"],
            "INTERRUPTED_RESULT_USED")

    process_table = subprocess.run(
        ["/usr/bin/ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    active = [line.strip() for line in process_table
              if "run_step5" in line or "diagnose_step5" in line]
    require(not active, "ACTIVE_STEP5_COMPUTE_PROCESS:" + " | ".join(active))

    report = signed({
        "schema_version": 1,
        "run_id": "step5_completion_20260921",
        "status": "PASS_STEP5_PROMPT_TO_ARTIFACT_AUDIT",
        "claims_verified": len(observed),
        "evidence_files_verified": len(evidence["files"]),
        "cumulative_step5_h20_hours_upper_bound_recomputed": recomputed,
        "resource_cap_h20_hours": ledger["cap_h20_hours"],
        "evaluation_access": False,
        "advance_to_14x14": False,
        "pixel_level_authorized": False,
        "all_step5_gpu_processes_stopped": True,
        "step6_started": False,
    })
    write_new(output, report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
