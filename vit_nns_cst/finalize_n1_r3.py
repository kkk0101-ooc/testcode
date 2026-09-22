#!/usr/bin/env python3
"""Issue the immutable NNS-CST N1-R3 terminal receipt after full audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence/n1_r3_20260922"


def sha256(relative):
    return hashlib.sha256((HERE / relative).read_bytes()).hexdigest()


def load(relative):
    return json.loads((HERE / relative).read_text(encoding="utf-8"))


def main():
    validation = load("evidence/n1_r3_20260922/validation.json")
    audit = load("evidence/n1_r3_20260922/completion_audit.json")
    if validation["status"] != "PASS_N1_R3_INDEPENDENT_VALIDATION":
        raise RuntimeError("validation not PASS")
    if audit["status"] != "PASS_N1_R3_COMPLETION_AUDIT" or not audit["all_requirements_passed"] or audit["missing_requirements"]:
        raise RuntimeError("completion audit not PASS")
    if audit["prompt_coverage"]["total_items"] < 45:
        raise RuntimeError("completion audit coverage too small")
    paths = [
        "N1数学合同.md", "N1-R3恢复修订.md",
        "config/n1_r3/subspace_contract.json",
        "manifests/n1_r3/model_port_registry.json",
        "manifests/n1_r3/n1_fixture_registry.json",
        "schemas/n1_r3_frozen_readout.schema.json",
        "evidence/n1_r3_20260922/freeze_receipt.json",
        "evidence/n1_r3_20260922/analytic_results.json",
        "evidence/n1_r3_20260922/validation.json",
        "evidence/n1_r3_20260922/completion_audit.json",
    ]
    receipt = {
        "schema_version": 1,
        "document_type": "nns_cst_n1_r3_terminal_receipt",
        "stage_id": "N1",
        "status": "PASS_N1_ANALYTIC_CONTRACT_AND_SKELETON",
        "terminal": True,
        "predecessors": {
            "r1_status": "FAIL_N1_R1_SCHEMA_FREEZE_INCOMPLETE",
            "r1_terminal_sha256": sha256("evidence/n1_r1_20260922/terminal_receipt.json"),
            "r2_status": "FAIL_N1_R2_READOUT_SERIALIZATION_INCOMPLETE",
            "r2_terminal_sha256": sha256("evidence/n1_r2_20260922/terminal_receipt.json"),
        },
        "gate_results": {
            "n0_predecessor": True, "closed_schemas": True,
            "frozen_readout_serialization": True, "fixture_assertions": True,
            "mutation_tests": True, "determinism_probe": True,
            "microbenchmark": True, "unit_tests": True,
            "independent_validator": True, "prompt_to_artifact_audit": True,
        },
        "artifact_bindings": [{"path": path, "sha256": sha256(path)} for path in paths],
        "validator_output_sha256": sha256("evidence/n1_r3_20260922/validation.json"),
        "completion_audit_sha256": sha256("evidence/n1_r3_20260922/completion_audit.json"),
        "execution_boundary": {"formal_image_decodes": 0, "model_loads": 0, "gpu_or_cuda_uses": 0, "n2_started": False},
        "claims_not_established": [
            "REAL_MODEL_ADAPTER_FIDELITY", "REAL_MODEL_JVP_VJP", "SEMANTIC_SUBSPACE_EXISTS",
            "INPUT_BEHAVIOR_GATE", "REAL_NNS_LOCALIZATION", "CAUSAL_MEDIATION",
            "CROSS_MODEL_REPLICATION",
        ],
        "next_gate": "N2_REQUIRES_EXPLICIT_USER_REQUEST",
    }
    raw = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path = EVIDENCE / "terminal_receipt.json"
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError("terminal receipt drift")
    if not path.exists():
        path.write_bytes(raw)
    print(json.dumps({"status": receipt["status"], "artifact_binding_count": len(paths),
                      "completion_audit_items": audit["prompt_coverage"]["total_items"],
                      "next_gate": receipt["next_gate"]},
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
