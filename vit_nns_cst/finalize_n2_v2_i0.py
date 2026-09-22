#!/usr/bin/env python3
"""Seal the blocked N2-V2 I0 terminal receipt after independent validation."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_i0_20260922"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    terminal_path = OUT / "terminal_receipt.json"
    if terminal_path.exists(): raise RuntimeError("I0_TERMINAL_ALREADY_EXISTS")
    validation = json.loads((OUT / "validation.json").read_text())
    if validation["status"] != "PASS_I0_FAIL_CLOSED_VALIDATION" or validation["summary"]["check_count"] != 27:
        raise RuntimeError("I0_VALIDATION_NOT_CURRENT")
    access = {
        "new_image_payload_reads": 0, "new_mask_payload_reads": 0, "confirmation_payload_reads": 0, "replication_payload_reads": 0,
        "reviewer_bundles_sealed": 0, "checkpoint_opens": 0, "checkpoint_deserializations": 0, "model_loads": 0,
        "inference_runs": 0, "behavior_outcome_reads": 0, "activation_extractions": 0, "endpoint_runs": 0,
        "path_nns_runs": 0, "jvp_runs": 0, "intervention_runs": 0, "gpu_uses": 0,
    }
    paths = {
        "protocol_sha256": ROOT / "N2-V2输入可识别性恢复协议.md",
        "contract_sha256": ROOT / "config/n2_v2/input_identifiability_contract.json",
        "contract_schema_sha256": ROOT / "schemas/n2_v2_input_contract.schema.json",
        "lineage_ledger_sha256": ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json",
        "lineage_schema_sha256": ROOT / "schemas/n2_v2_lineage_exclusions.schema.json",
        "review_schema_sha256": ROOT / "schemas/n2_v2_review_bundle.schema.json",
        "preflight_schema_sha256": ROOT / "schemas/n2_v2_i0_preflight.schema.json",
        "validation_schema_sha256": ROOT / "schemas/n2_v2_i0_validation.schema.json",
        "terminal_schema_sha256": ROOT / "schemas/n2_v2_i0_terminal.schema.json",
        "legacy_closure_sha256": OUT / "legacy_closure.json",
        "asset_preflight_sha256": OUT / "asset_preflight.json",
        "reviewer_preflight_sha256": OUT / "reviewer_preflight.json",
        "validation_sha256": OUT / "validation.json",
        "validator_sha256": ROOT / "validate_n2_v2_i0.py",
    }
    receipt = {
        "schema_version": 1, "study_version": "N2_V2", "stage_id": "N2_V2_INPUT_IDENTIFIABILITY",
        "attempt_id": "N2_V2_I0_20260922", "terminal": True, "status": "BLOCKED_INPUT_ASSETS_AND_REVIEWERS",
        "reason": "No audited natural scene-only asset registry and no two independent sealed reviewer bundles are available; all model-side work remains prohibited.",
        "gate_results": {
            "legacy_n2_closure": "PASS", "v2_protocol_freeze": "PASS", "legacy_lineage_exclusion": "PASS",
            "scene_only_asset_registry": "BLOCKED_NOT_AVAILABLE", "independent_reviewers": "BLOCKED_ZERO_OF_TWO",
            "blind_review": "NOT_RUN_PRECEDING_BLOCK", "low_level_matching": "NOT_RUN_PRECEDING_BLOCK",
            "identifiability_freeze": "NOT_ISSUED", "behavior_gate": "NOT_AUTHORIZED", "n3": "NOT_AUTHORIZED",
        },
        "access_accounting": access, "artifact_bindings": {key: sha(path) for key, path in paths.items()},
        "next_gate": "STOP_I0_NEW_FRESH_ROOT_REQUIRES_AUDITED_SCENE_ONLY_ASSETS_AND_TWO_INDEPENDENT_REVIEWERS",
    }
    schema = json.loads((ROOT / "schemas/n2_v2_i0_terminal.schema.json").read_text())
    jsonschema.validate(receipt, schema)
    descriptor = os.open(terminal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": receipt["status"], "bindings": len(paths), "model_accesses": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
