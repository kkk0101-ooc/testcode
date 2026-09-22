#!/usr/bin/env python3
"""Write the N2 prompt-to-artifact completion audit."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_r1c_20260922/completion_audit.json"
# Preserve the virtual-environment entry point when launching child checks.
# Resolving this symlink would bypass the venv site-packages.
PYTHON = Path(sys.executable).absolute()
PYTHON_REALPATH = Path(os.path.realpath(str(PYTHON)))


def item(identifier: str, requirement: str, evidence: list[str], verification: str, status: str = "PASS") -> dict:
    return {"requirement_id": identifier, "requirement": requirement, "evidence": evidence, "verification": verification, "status": status}


def main() -> int:
    validation = json.loads(subprocess.check_output([str(PYTHON), str(ROOT / "validate_n2.py")], text=True, cwd=ROOT))
    if validation["status"] != "PASS_N2_TERMINAL_VALIDATION" or len(validation["checks"]) != 33:
        raise RuntimeError("N2_TERMINAL_VALIDATION_NOT_CURRENT")
    test = subprocess.run([str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, text=True, capture_output=True, check=True)
    if "Ran 22 tests" not in test.stderr or "OK" not in test.stderr:
        raise RuntimeError("UNIT_TEST_COVERAGE_DRIFT")
    qc = json.loads((ROOT / "evidence/n2_r1c_20260922/blind_qc.json").read_text())
    terminal = json.loads((ROOT / "evidence/n2_r1c_20260922/terminal_receipt.json").read_text())
    checks = [
        item("N2-COMMON-01", "N1 predecessor is valid", ["evidence/n1_r3_20260922/terminal_receipt.json", "validate_n1_r3.py"], "Current N1 validator and 45-item audit passed"),
        item("N2-COMMON-02", "Only N2 is authorized", ["config/n2_r1/pilot_contract.json"], "N2_ONLY and n3_authorized=false"),
        item("N2-5.1-01", "Only pilot_discovery data are used", ["evidence/n2_r1c_20260922/pair_manifest.json"], "24 unique sources are pilot/S1_PILOT_ONLY"),
        item("N2-5.1-02", "Confirmation and replication remain unread", ["evidence/n2_r1c_20260922/terminal_receipt.json"], "Both payload read counters are zero"),
        item("N2-5.1-03", "Layers, ranks and semantic axes are finite and predeclared", ["config/n2_r1/pilot_contract.json"], "One axis, four blocks and three ranks"),
        item("N2-5.1-04", "Pilot selections and recoveries are logged", ["evidence/n2_r1_20260922/qc_rejection_receipt.json", "evidence/n2_r1b_20260922/qc_rejection_receipt.json", "N2-R1C最终生产器恢复修订.md"], "All changes occurred before model outcome access"),
        item("N2-ID-01", "Semantic and matched-null four-arm pairs are constructed", ["evidence/n2_r1c_20260922/pair_inputs.npz", "evidence/n2_r1c_20260922/pair_manifest.json"], "Eight sources by four arms; raw digests recomputed"),
        item("N2-ID-02", "Source lineages do not cross protected splits", ["evidence/n2_r1c_20260922/validation.json"], "Source-group intersection is zero"),
        item("N2-ID-03", "Recipient foreground is unchanged", ["evidence/n2_r1c_20260922/validation.json"], "Maximum eroded-mask uint8 difference is zero"),
        item("N2-ID-04", "Low-level matching is independently recomputed", ["evidence/n2_r1c_20260922/validation.json"], "Six of eight matched; unmatched rate equals frozen 25 percent cap"),
        item("N2-ID-05", "Background semantic reliability is measured", ["evidence/n2_r1c_20260922/blind_qc.json"], "Required two-reviewer agreement is not established; fail closed"),
        item("N2-ID-06", "Blind edit quality and artifact parity are measured", ["evidence/n2_r1c_20260922/blind_qc.json"], "Seven of eight pairs failed artifact-free review"),
        item("N2-5.2-01", "Behavior is evaluated only after identifiability", ["N2隔离Pilot协议.md", "evidence/n2_r1c_20260922/blind_qc.json"], "Behavior gate correctly not run after identifiability failure", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.2-02", "Semantic, null and low-level controls enter the behavior gate", ["config/n2_r1/pilot_contract.json", "evidence/n2_r1c_20260922/pair_manifest.json"], "Inputs are present but no model outcome was read", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.3-01", "Paired activations and finite subspace search", ["evidence/n2_r1c_20260922/blind_qc.json"], "Not run because the mandatory prior gate failed", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.3-02", "Basis invariance and independent recomputation", ["tests/test_n2.py", "src/nns_cst_n2/core.py"], "Engineering fixtures pass; real pilot correctly not run", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.3-03", "Single-head, PCA, random and equal-capacity probe comparisons", ["evidence/n2_r1c_20260922/blind_qc.json"], "Not run because the mandatory prior gate failed", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.3-04", "Held-out stability and topology checks", ["config/n2_r1/pilot_contract.json", "evidence/n2_r1c_20260922/blind_qc.json"], "Frozen but not run after prior gate failure", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.4-01", "Endpoint connectivity and cost pilot", ["evidence/n2_r1c_20260922/terminal_receipt.json"], "Endpoint run count is zero after prior gate failure", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.4-02", "Path NNS convergence, references and mask alignment", ["evidence/n2_r1c_20260922/terminal_receipt.json"], "Path NNS run count is zero after prior gate failure", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.4-03", "Complete-cut semantic/complement closure", ["evidence/n2_r1c_20260922/terminal_receipt.json"], "Not run after prior gate failure", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.4-04", "Attention, MLP and residual JVP transport", ["evidence/n2_r1c_20260922/terminal_receipt.json"], "JVP run count is zero after prior gate failure", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.4-05", "Injection, erasure, doses and off-manifold checks", ["evidence/n2_r1c_20260922/terminal_receipt.json"], "Intervention run count is zero after prior gate failure", "CORRECTLY_NOT_RUN_AFTER_GATE_FAILURE"),
        item("N2-5.4-06", "Task resource accounting", ["evidence/n2_r1c_20260922/terminal_receipt.json"], "GPU, checkpoint, inference and downstream counts are explicitly zero"),
        item("N2-5.5-01", "Terminal state belongs to the allowed N2 set", ["执行规划.md", "evidence/n2_r1c_20260922/terminal_receipt.json"], "NO_GO_NOT_IDENTIFIABLE is explicitly allowed"),
        item("N2-5.5-02", "No extensive concept/layer fishing", ["N2-R1C最终生产器恢复修订.md", "evidence/n2_r1c_20260922/terminal_receipt.json"], "Stopped at frozen final producer attempt before any outcome"),
        item("N2-COMMON-03", "Producer and validator are separated", ["build_n2_pairs.py", "validate_n2.py"], "Validator independently recomputes raw NPZ evidence"),
        item("N2-COMMON-04", "Schemas are closed", ["schemas/n2_pilot_contract.schema.json", "schemas/n2_pair_manifest.schema.json", "schemas/n2_blind_qc.schema.json", "schemas/n2_terminal_receipt.schema.json", "schemas/n2_completion_audit.schema.json"], "Five Draft-07 schemas self-check and every object is closed"),
        item("N2-COMMON-05", "N2 implementation tests pass", ["tests/test_n2.py"], "Seven N2 tests and twenty-two repository tests pass"),
        item("N2-COMMON-06", "N3 is not started", ["config/n2_r1/pilot_contract.json", "evidence/n2_r1c_20260922/terminal_receipt.json"], "n3_authorized=false and terminal next gate starts STOP_N2"),
    ]
    result = {"schema_version": 1, "stage_id": "N2", "attempt_id": "N2_R1C_20260922", "status": "PASS_N2_COMPLETION_AUDIT", "objective": "Execute N2 of testcode/vit_nns_cst/执行规划.md through its first mandatory terminal gate", "terminal_decision": terminal["status"], "all_requirements_handled": True, "missing_requirements": [], "checklist": checks, "verification_summary": {"checklist_items": len(checks), "n2_validator_checks": 33, "unit_tests": 22, "n1_completion_items": 45, "model_outcome_reads": qc["review_boundary"]["model_checkpoint_deserializations_before_review"], "gpu_uses": terminal["access_and_resource_accounting"]["gpu_uses"], "n3_started": False, "python_executable": str(PYTHON), "python_executable_realpath": str(PYTHON_REALPATH)}}
    schema = json.loads((ROOT / "schemas/n2_completion_audit.schema.json").read_text())
    jsonschema.Draft7Validator.check_schema(schema)
    jsonschema.validate(result, schema)
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if OUT.exists():
        if OUT.read_text(encoding="utf-8") != text:
            raise RuntimeError("COMPLETION_AUDIT_EVIDENCE_DRIFT")
    else:
        descriptor = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "checklist_items": len(checks), "terminal_decision": result["terminal_decision"]}, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
