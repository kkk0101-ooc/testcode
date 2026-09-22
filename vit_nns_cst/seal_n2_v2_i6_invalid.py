#!/usr/bin/env python3
"""Seal the interrupted N2-V2 I6 root as an invalid protocol attempt."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_i6_20260922"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    terminal_path = OUT / "terminal_receipt.json"
    if terminal_path.exists():
        raise RuntimeError("I6_TERMINAL_ALREADY_EXISTS")
    if not OUT.exists():
        raise RuntimeError("I6_ROOT_NOT_PRESENT")
    receipt = {
        "schema_version": 1,
        "study_version": "N2_V2",
        "stage_id": "N2_V2_INPUT_IDENTIFIABILITY",
        "attempt_id": "N2_V2_I6_20260922",
        "terminal": True,
        "status": "INVALID_PROTOCOL",
        "reason": "The 0.85 / 18 threshold amendment was applied, but the first I6 evaluator still used the inherited accepted-asset schema with minItems=48. The run stopped before accepted registry, matching, model access, or downstream gates.",
        "superseded_by_attempt_id": "N2_V2_I7_20260922",
        "gate_results": {
            "review_bundle_binding": "PASS",
            "threshold_amendment": "PRESENT",
            "accepted_registry_schema": "INVALID_OLD_MINITEMS_48_FOR_AMENDED_MIN_PER_CONTEXT_18",
            "accepted_scene_only_registry": "NOT_CREATED",
            "low_level_matching": "NOT_RUN_PRECEDING_INVALID_PROTOCOL",
            "identifiability_freeze": "NOT_ISSUED",
            "behavior_gate": "NOT_AUTHORIZED",
            "n3": "NOT_AUTHORIZED"
        },
        "access_accounting": {
            "review_bundle_reads": 2,
            "checkpoint_opens": 0,
            "checkpoint_deserializations": 0,
            "model_loads": 0,
            "inference_runs": 0,
            "behavior_outcome_reads": 0,
            "activation_extractions": 0,
            "endpoint_runs": 0,
            "path_nns_runs": 0,
            "jvp_runs": 0,
            "intervention_runs": 0,
            "gpu_uses": 0
        },
        "artifact_bindings": {
            "threshold_amendment_sha256": sha(ROOT / "N2-V2-I6输入门阈值修订.md"),
            "source_terminal_i5_sha256": sha(ROOT / "evidence/n2_v2_i5_20260922/terminal_receipt.json"),
            "reviewer_a_bundle_sha256": sha(OUT / "review/reviewer_A.json"),
            "reviewer_b_bundle_sha256": sha(OUT / "review/reviewer_B.json"),
            "review_card_manifest_sha256": sha(OUT / "review/review_card_manifest.json"),
            "old_accepted_schema_sha256": sha(ROOT / "schemas/n2_v2_accepted_asset_registry.schema.json"),
            "invalid_seal_script_sha256": sha(ROOT / "seal_n2_v2_i6_invalid.py")
        },
        "next_gate": "SUPERSEDED_BY_I7_DO_NOT_CONTINUE_FROM_I6"
    }
    descriptor = os.open(terminal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": receipt["status"], "superseded_by": receipt["superseded_by_attempt_id"], "model_accesses": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
