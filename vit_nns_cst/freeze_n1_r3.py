#!/usr/bin/env python3
"""Freeze fresh NNS-CST N1-R3 artifacts before R3 execution."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import freeze_n1


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVIDENCE = HERE / "evidence/n1_r3_20260922"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_or_identical(path: Path, value: Any) -> None:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError("frozen artifact drift: " + str(path))
    if not path.exists():
        path.write_bytes(raw)


def main() -> None:
    ports = freeze_n1.build_port_registry()
    contract = freeze_n1.build_subspace_contract()
    fixtures = freeze_n1.build_fixture_registry()
    fixtures["fixtures"].append({
        "fixture_id": "FROZEN_READOUT_SERIALIZATION",
        "required_assertions": [
            "CLOSED_FIELD_SET", "FIT_SPLIT_BOUND", "CONTRACT_DIGEST_BOUND",
            "VALIDATION_REFIT_DISABLED", "CONFIRMATION_REFIT_DISABLED",
            "POSITIVE_SCORE_DIRECTION", "NEGATIVE_SCORE_DIRECTION",
        ],
    })
    write_new_or_identical(HERE / "manifests/n1_r3/model_port_registry.json", ports)
    write_new_or_identical(HERE / "config/n1_r3/subspace_contract.json", contract)
    write_new_or_identical(HERE / "manifests/n1_r3/n1_fixture_registry.json", fixtures)

    frozen_paths = [
        "config/n1_r3/subspace_contract.json",
        "manifests/n1_r3/model_port_registry.json",
        "manifests/n1_r3/n1_fixture_registry.json",
        "schemas/n1_port_registry.schema.json",
        "schemas/n1_subspace_contract.schema.json",
        "schemas/n1_r3_fixture_registry.schema.json",
        "schemas/n1_r3_frozen_readout.schema.json",
        "schemas/n1_r3_analytic_results.schema.json",
        "schemas/n1_r3_freeze_receipt.schema.json",
        "schemas/n1_r3_validation.schema.json",
        "schemas/n1_r3_completion_audit.schema.json",
        "schemas/n1_r3_terminal_receipt.schema.json",
        "N1数学合同.md", "N1-R3恢复修订.md",
        "src/nns_cst/__init__.py", "src/nns_cst/core.py", "src/nns_cst/interfaces.py",
        "src/nns_cst_r3/__init__.py", "src/nns_cst_r3/readout.py",
        "run_n1.py", "n1_determinism_probe.py", "freeze_n1.py",
        "freeze_n1_r3.py", "run_n1_r3.py", "n1_r3_determinism_probe.py",
        "validate_n1_r3.py", "finalize_n1_r3.py", "audit_n1_r3.py",
        "tests/test_n1.py", "tests/test_n1_r3.py",
    ]
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=str(REPO)).decode().strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
    receipt = {
        "schema_version": 1,
        "document_type": "nns_cst_n1_r3_freeze_receipt",
        "stage_id": "N1",
        "attempt_id": "N1_R3_20260922",
        "status": "FROZEN_BEFORE_N1_R3_EXECUTION",
        "authorization": {
            "source": "explicit_current_thread_user_request",
            "authorized_stage": "N1",
            "n2_authorized": False,
        },
        "predecessors": {
            "n0_decision": "GO_REQUEST_N1",
            "n0_decision_sha256": sha256(HERE / "n0_decision.json"),
            "n1_r1_status": "FAIL_N1_R1_SCHEMA_FREEZE_INCOMPLETE",
            "n1_r1_terminal_sha256": sha256(HERE / "evidence/n1_r1_20260922/terminal_receipt.json"),
            "n1_r2_status": "FAIL_N1_R2_READOUT_SERIALIZATION_INCOMPLETE",
            "n1_r2_terminal_sha256": sha256(HERE / "evidence/n1_r2_20260922/terminal_receipt.json"),
        },
        "authority": {
            "execution_plan_sha256": sha256(HERE / "执行规划.md"),
            "research_goal_sha256": sha256(HERE / "研究目标.md"),
            "mathematical_contract_sha256": sha256(HERE / "N1数学合同.md"),
            "recovery_amendment_sha256": sha256(HERE / "N1-R3恢复修订.md"),
        },
        "frozen_artifacts": [{"path": path, "sha256": sha256(HERE / path)} for path in frozen_paths],
        "repository": {"branch": branch, "head_commit": head},
        "execution_boundary": {
            "formal_image_decodes": 0, "model_loads": 0, "checkpoint_deserializations": 0,
            "gpu_or_cuda_uses": 0, "training_runs": 0, "inference_runs": 0,
            "formal_outcome_reads": 0, "n2_started": False,
        },
    }
    write_new_or_identical(EVIDENCE / "freeze_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "frozen_artifact_count": len(frozen_paths),
                      "fixture_count": len(fixtures["fixtures"]), "port_count": ports["record_count"]},
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
