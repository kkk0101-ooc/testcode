#!/usr/bin/env python3
"""Seal N2-V2 I2 at the mandatory human-review stop gate."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_i2_20260922"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""): h.update(chunk)
    return h.hexdigest()

def main():
    path = OUT / "terminal_receipt.json"
    if path.exists(): raise RuntimeError("I2_TERMINAL_ALREADY_EXISTS")
    validation = json.loads((OUT / "validation.json").read_text())
    if validation["status"] != "PASS_REVIEW_PREPARATION_VALIDATION": raise RuntimeError("I2_VALIDATION_NOT_PASS")
    receipt = {
        "schema_version": 1, "study_version": "N2_V2", "stage_id": "N2_V2_INPUT_IDENTIFIABILITY",
        "attempt_id": "N2_V2_I2_20260922", "terminal": True, "status": "BLOCKED_AWAITING_TWO_INDEPENDENT_BLIND_REVIEWS",
        "reason": "Licensed scene candidates, blinded cards, and twelve class-aligned recipient candidates are ready; two independent human review bundles are required before acceptance or matching.",
        "gate_results": {"legacy_n2_closure": "PASS", "v2_protocol_freeze": "PASS", "legacy_lineage_exclusion": "PASS", "scene_candidate_preparation": "PASS_102_CANDIDATES", "recipient_candidate_preparation": "PASS_12_CANDIDATES", "independent_reviewers": "ASSIGNED_BY_USER_TWO_HUMANS", "blind_review": "BLOCKED_AWAITING_TWO_SEALED_BUNDLES", "accepted_scene_only_registry": "NOT_CREATED_BEFORE_REVIEW", "low_level_matching": "NOT_RUN_PRECEDING_GATE", "identifiability_freeze": "NOT_ISSUED", "behavior_gate": "NOT_AUTHORIZED", "n3": "NOT_AUTHORIZED"},
        "access_accounting": {"scene_image_payload_reads_for_preparation": 102, "recipient_image_payload_reads_for_preparation": 12, "recipient_mask_payload_reads_for_preparation": 12, "reviewer_bundles_sealed": 0, "checkpoint_opens": 0, "checkpoint_deserializations": 0, "model_loads": 0, "inference_runs": 0, "behavior_outcome_reads": 0, "activation_extractions": 0, "endpoint_runs": 0, "path_nns_runs": 0, "jvp_runs": 0, "intervention_runs": 0, "gpu_uses": 0},
        "artifact_bindings": {"protocol_sha256": sha(ROOT / "N2-V2输入可识别性恢复协议.md"), "contract_sha256": sha(ROOT / "config/n2_v2/input_identifiability_contract.json"), "legacy_lineage_ledger_sha256": sha(ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json"), "scene_candidate_registry_sha256": sha(OUT / "private/scene_candidate_registry.json"), "recipient_candidate_registry_sha256": sha(OUT / "private/recipient_candidate_registry.json"), "review_card_manifest_sha256": sha(OUT / "review/review_card_manifest.json"), "reviewer_a_template_sha256": sha(OUT / "review/templates/reviewer_A.json"), "reviewer_b_template_sha256": sha(OUT / "review/templates/reviewer_B.json"), "review_instructions_sha256": sha(OUT / "review/README.md"), "preparation_receipt_sha256": sha(OUT / "preparation_receipt.json"), "validation_sha256": sha(OUT / "validation.json"), "validator_sha256": sha(ROOT / "validate_n2_v2_i2.py")},
        "next_gate": "STOP_I2_WAIT_FOR_TWO_INDEPENDENT_HUMAN_REVIEW_BUNDLES_NEW_FRESH_ROOT_REQUIRED_AFTER_REVIEWS",
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, ensure_ascii=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": receipt["status"], "bindings": len(receipt["artifact_bindings"]), "model_accesses": 0}, sort_keys=True))

if __name__ == "__main__": main()
