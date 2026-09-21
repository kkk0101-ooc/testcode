#!/usr/bin/env python3
"""CPU-only independent audit of the cross-root Step-5 extension seal."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SEAL = ROOT / "step5_extension_seal_20260921"


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "VALIDATOR_MUST_BE_CPU_ONLY")
    output = SEAL / "validation_report.json"
    if output.exists():
        raise RuntimeError("VALIDATION_ALREADY_EXISTS")
    claims = verify_signed(SEAL / "claim_decisions.json")
    ledger = verify_signed(SEAL / "resource_ledger.json")
    evidence = verify_signed(SEAL / "evidence_index.json")
    terminal = verify_signed(SEAL / "terminal_receipt.json")
    require(terminal["status"] == "PARTIAL_NOT_SUPPORTED_WITH_E3_INVALID", "TERMINAL")
    require(terminal["claim_decisions_digest"] == claims["canonical_digest"], "CLAIMS_BINDING")
    require(terminal["resource_ledger_digest"] == ledger["canonical_digest"], "LEDGER_BINDING")
    require(terminal["evidence_index_digest"] == evidence["canonical_digest"], "EVIDENCE_BINDING")
    require(terminal["original_r8_decision_unchanged"] == "NOT_SUPPORTED_EMPIRICAL_VALUE", "R8_BOUNDARY")
    require(terminal["next_authorization"] == "NONE" and not terminal["step6_started"], "STAGE_BOUNDARY")
    expected = {
        "E1_LOCAL_CURL_PREDICTS_LOOP_CIRCULATION": "NOT_SUPPORTED",
        "E2_NONLINEAR_CONDITION_AMPLIFIES_VALID_CURVATURE_EFFECT": "NOT_SUPPORTED",
        "E3A_7X7_PEFA_INCREMENTAL_RESOLUTION_VALUE": "INVALID",
    }
    observed = {row["claim_id"]: row["decision"] for row in claims["decisions"]}
    require(observed == expected, "CLAIM_DECISIONS")
    e3 = next(row for row in claims["decisions"] if row["claim_id"].startswith("E3A"))
    require(e3["reason"] == "INVALID_SOURCE_INTEGRAL_CONVERGENCE", "E3_REASON")
    require(not claims["advance_to_14x14"] and not claims["pixel_level_authorized"], "RESOLUTION_STOP")
    require(not claims["confirmatory_claim_allowed"] and not claims["step6_production_authorized"], "PROMOTION_BOUNDARY")
    for row in evidence["files"]:
        path = REPO / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"], "EVIDENCE_SIZE:" + row["path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "EVIDENCE_HASH:" + row["path"])
    r3 = verify_signed(ROOT / "step5_extension_r3_20260921/resource_ledger.json")
    diagnosis_paths = [
        ROOT / "step5_extension_r3_source_diagnosis_deit_s16.json",
        ROOT / "step5_extension_r3_source_diagnosis_vit_b16.json",
        ROOT / "step5_extension_r3_jvp_diagnosis_deit_s16.json",
        ROOT / "step5_extension_r3_jvp_diagnosis_vit_b16.json",
    ]
    diagnoses = [verify_signed(path) for path in diagnosis_paths]
    recomputed = r3["cumulative_step5_h20_hours"] + math.fsum(
        row["resources"]["h20_hours"] for row in diagnoses
    )
    require(math.isclose(recomputed, ledger["cumulative_step5_h20_hours"],
                         rel_tol=0, abs_tol=1e-12), "RESOURCE_RECOMPUTATION")
    require(ledger["status"] == "SUPPORTED" and recomputed <= ledger["cap_h20_hours"], "RESOURCE_CAP")
    source = {row["task_id"]: row for row in diagnoses[:2]}
    require(set(source) == {"e3a__deit_s16__00", "e3a__vit_b16__00"}, "DIAGNOSIS_MODELS")
    require(all(next(item for item in row["rows"] if item["nodes"] == 512)
                ["terminal_T_convergence"]["relative_l2"] > 1.0 for row in source.values()),
            "SOURCE_FAILURE_NOT_REPRODUCED")
    tests = subprocess.run(
        [str(REPO / ".conda/envs/vitnns/bin/python"), "-m", "pytest", "-q",
         "vit_pcia_pefa/tests/test_step5_extension.py", "vit_pcia_pefa/tests/test_step5_queue.py"],
        cwd=REPO, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                       "PYTHONPATH": ".:vitnns_rebuild/src"}, capture_output=True, text=True,
    )
    require(tests.returncode == 0, "UNIT_TESTS:" + tests.stdout + tests.stderr)
    report = signed({
        "schema_version": 1, "run_id": "step5_extension_seal_20260921",
        "status": "PASS_STEP5_EXTENSION_SEAL_AUDIT", "evidence_files_verified": len(evidence["files"]),
        "claims_verified": len(claims["decisions"]),
        "cumulative_step5_h20_hours_recomputed": recomputed,
        "original_r8_decision_unchanged": True, "evaluation_access": False,
        "resolution_staircase_stopped_before_14x14": True, "pixel_level_started": False,
        "step6_started": False, "unit_tests": tests.stdout.strip(),
    })
    write_new(output, report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
