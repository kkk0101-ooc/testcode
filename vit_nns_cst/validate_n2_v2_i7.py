#!/usr/bin/env python3
"""Validate the amended N2-V2 I7 blind-review acceptance root."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_i7_20260922"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, code: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(code)
    checks.append(code)


def main() -> int:
    checks: list[str] = []
    receipt = json.loads((OUT / "terminal_receipt.json").read_text())
    evaluation = json.loads((OUT / "review_evaluation.json").read_text())
    accepted = json.loads((OUT / "accepted_scene_only_assets.json").read_text())
    allowed = json.loads((ROOT / "config/n2_v2/input_identifiability_contract.json").read_text())["allowed_input_stage_statuses"]
    check(receipt["status"] == "PASS_INPUT_IDENTIFIABILITY_FROZEN" and receipt["status"] in allowed, "TERMINAL_STATUS_ALLOWED", checks)
    check(evaluation["status"] == "PASS_AMENDED_BLIND_REVIEW", "EVALUATION_STATUS_PASS", checks)
    check(evaluation["minimum_raw_semantic_agreement"] == 0.85 and evaluation["minimum_accepted_per_context"] == 18, "AMENDED_THRESHOLDS_BOUND", checks)
    check(evaluation["raw_semantic_agreement"] >= 0.85, "RAW_AGREEMENT_PASSES_AMENDMENT", checks)
    check(evaluation["cohen_kappa"] >= 0.80, "KAPPA_PASSES", checks)
    check(evaluation["accepted_by_context_label"] == {"AQUATIC_CONTEXT": 19, "NON_AQUATIC_CONTEXT": 24}, "ACCEPTED_CONTEXT_COUNTS", checks)
    check(len(accepted["records"]) == 43, "ACCEPTED_REGISTRY_RECORD_COUNT", checks)
    check(accepted["asset_type"] == "NATURAL_SCENE_ONLY_NO_FOREGROUND_REMOVAL", "ACCEPTED_ASSET_TYPE", checks)
    check(accepted["threshold_amendment_id"] == "N2_V2_I7_REVIEW_THRESHOLD_AMENDMENT_20260922" and accepted["minimum_accepted_per_context"] == 18, "ACCEPTED_REGISTRY_AMENDMENT_BOUND", checks)
    check(all(row["scene_only"] and not row["artifact_union_reject"] and not row["legacy_lineage_overlap"] for row in accepted["records"]), "ACCEPTED_RECORD_FLAGS", checks)
    check({row["context_label"] for row in accepted["records"]} == {"AQUATIC_CONTEXT", "NON_AQUATIC_CONTEXT"}, "ACCEPTED_CONTEXT_LABELS_CLOSED", checks)
    check(receipt["gate_results"]["low_level_matching"] == "NOT_RUN_AWAITING_NEXT_FRESH_ROOT", "MATCHING_NOT_RUN", checks)
    check(receipt["gate_results"]["behavior_gate"] == "NOT_AUTHORIZED", "BEHAVIOR_NOT_AUTHORIZED", checks)
    check(all(value == 0 for key, value in receipt["access_accounting"].items() if key.endswith("uses") or key.endswith("runs") or key in {"checkpoint_opens", "checkpoint_deserializations", "model_loads", "behavior_outcome_reads", "activation_extractions"}), "MODEL_SIDE_ACCESS_ZERO", checks)
    path_map = {
        "protocol_sha256": ROOT / "N2-V2输入可识别性恢复协议.md",
        "threshold_amendment_sha256": ROOT / "N2-V2-I7输入门阈值修订.md",
        "contract_sha256": ROOT / "config/n2_v2/input_identifiability_contract.json",
        "legacy_lineage_ledger_sha256": ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json",
        "source_terminal_i5_sha256": ROOT / "evidence/n2_v2_i5_20260922/terminal_receipt.json",
        "scene_candidate_registry_sha256": OUT / "private/scene_candidate_registry.json",
        "recipient_candidate_registry_sha256": OUT / "private/recipient_candidate_registry.json",
        "mask_archive_bindings_sha256": OUT / "private/openimages_mask_archive_bindings.json",
        "review_card_manifest_sha256": OUT / "review/review_card_manifest.json",
        "reviewer_a_bundle_sha256": OUT / "review/reviewer_A.json",
        "reviewer_b_bundle_sha256": OUT / "review/reviewer_B.json",
        "accepted_scene_registry_sha256": OUT / "accepted_scene_only_assets.json",
        "review_evaluation_sha256": OUT / "review_evaluation.json",
        "review_schema_sha256": ROOT / "schemas/n2_v2_review_bundle.schema.json",
        "accepted_registry_schema_sha256": ROOT / "schemas/n2_v2_i7_accepted_asset_registry.schema.json",
        "evaluator_sha256": ROOT / "finalize_n2_v2_i7_review.py",
    }
    for key, expected in receipt["artifact_bindings"].items():
        check(sha(path_map[key]) == expected, "BINDING_" + key, checks)
    result = {
        "schema_version": 1,
        "study_version": "N2_V2",
        "attempt_id": "N2_V2_I7_20260922",
        "status": "PASS_I7_AMENDED_REVIEW_VALIDATION",
        "check_count": len(checks),
        "checks": checks,
        "terminal_status": receipt["status"],
        "accepted_scene_asset_count": len(accepted["records"]),
        "model_access_count": 0,
    }
    if "--write" in sys.argv:
        (OUT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
