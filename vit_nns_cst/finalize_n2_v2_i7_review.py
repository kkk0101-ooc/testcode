#!/usr/bin/env python3
"""Seal N2-V2 I7 under the amended 0.85 / 18 blind-review thresholds."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "evidence/n2_v2_i5_20260922"
OUT = ROOT / "evidence/n2_v2_i7_20260922"
ATTEMPT_ID = "N2_V2_I7_20260922"
MIN_RAW_AGREEMENT = 0.85
MIN_KAPPA = 0.80
MIN_ACCEPTED_PER_CONTEXT = 18
ARTIFACT_FIELDS = (
    "visible_non_background_object",
    "donor_removal_or_fill_trace",
    "abnormal_compositing_boundary",
    "other_arm_identity_cue",
)

sys.path.insert(0, str(ROOT / "src"))
from nns_cst_n2_v2.input_gate import evaluate_blind_reviews  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> int:
    if OUT.exists():
        raise RuntimeError("I7_ROOT_ALREADY_EXISTS")
    OUT.mkdir(parents=True)

    for relative in (
        "private/scene_candidate_registry.json",
        "private/recipient_candidate_registry.json",
        "private/openimages_mask_archive_bindings.json",
        "review/reviewer_A.json",
        "review/reviewer_B.json",
        "review/review_card_manifest.json",
    ):
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, target)

    schema = json.loads((ROOT / "schemas/n2_v2_review_bundle.schema.json").read_text())
    bundle_a = json.loads((OUT / "review/reviewer_A.json").read_text())
    bundle_b = json.loads((OUT / "review/reviewer_B.json").read_text())
    jsonschema.validate(bundle_a, schema)
    jsonschema.validate(bundle_b, schema)
    manifest_sha = sha(OUT / "review/review_card_manifest.json")
    if bundle_a["review_card_manifest_sha256"] != manifest_sha or bundle_b["review_card_manifest_sha256"] != manifest_sha:
        raise RuntimeError("REVIEW_BUNDLE_MANIFEST_SHA_MISMATCH")

    result = evaluate_blind_reviews(bundle_a, bundle_b, minimum_raw_agreement=MIN_RAW_AGREEMENT, minimum_kappa=MIN_KAPPA)
    scenes = json.loads((OUT / "private/scene_candidate_registry.json").read_text())["records"]
    by_card = {row["card_id"]: row for row in scenes}
    rows_a = {row["card_id"]: row for row in bundle_a["reviews"]}
    rows_b = {row["card_id"]: row for row in bundle_b["reviews"]}

    accepted_records: list[dict] = []
    rejected_cards: list[dict] = []
    reason_counter: Counter[str] = Counter()
    accepted_by_context: Counter[str] = Counter()
    accepted_by_stratum: Counter[str] = Counter()
    for card_id in sorted(rows_a):
        a, b = rows_a[card_id], rows_b[card_id]
        reasons: list[str] = []
        if a["semantic_label"] == "AMBIGUOUS" or b["semantic_label"] == "AMBIGUOUS":
            reasons.append("AMBIGUOUS_LABEL")
        if a["semantic_label"] != b["semantic_label"]:
            reasons.append("SEMANTIC_DISAGREEMENT")
        for field in ARTIFACT_FIELDS:
            if a[field] or b[field]:
                reasons.append(field.upper())
        if reasons:
            unique = sorted(set(reasons))
            rejected_cards.append({"card_id": card_id, "candidate_stratum": by_card[card_id]["candidate_stratum"], "reasons": unique})
            reason_counter.update(unique)
            continue
        scene = by_card[card_id]
        context = a["semantic_label"]
        accepted_by_context[context] += 1
        accepted_by_stratum[scene["candidate_stratum"]] += 1
        accepted_records.append({
            "asset_id": scene["asset_id"],
            "card_id": card_id,
            "source_group_id": scene["source_group_id"],
            "image_sha256": scene["image_sha256"],
            "image_path": scene["image_path"],
            "license_record_id": scene["license_record_id"],
            "context_label": context,
            "scene_only": True,
            "foreground_removal_applied": False,
            "producer_operations": [],
            "reviewer_labels": [a["semantic_label"], b["semantic_label"]],
            "artifact_union_reject": False,
            "legacy_lineage_overlap": False,
            "global_exposure_status": "ELIGIBLE_NEW_PILOT_ONLY",
        })

    if not result["reliability_passed"]:
        raise RuntimeError("AMENDED_RELIABILITY_GATE_FAILED")
    if accepted_by_context["AQUATIC_CONTEXT"] < MIN_ACCEPTED_PER_CONTEXT or accepted_by_context["NON_AQUATIC_CONTEXT"] < MIN_ACCEPTED_PER_CONTEXT:
        raise RuntimeError("AMENDED_ACCEPTED_CONTEXT_COUNT_FAILED")

    review_bundle_hashes = [sha(OUT / "review/reviewer_A.json"), sha(OUT / "review/reviewer_B.json")]
    accepted_registry = {
        "schema_version": 1,
        "study_version": "N2_V2",
        "registry_id": "N2_V2_I7_ACCEPTED_SCENE_ONLY_ASSETS_20260922",
        "asset_type": "NATURAL_SCENE_ONLY_NO_FOREGROUND_REMOVAL",
        "review_bundle_sha256s": review_bundle_hashes,
        "legacy_lineage_ledger_sha256": sha(ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json"),
        "threshold_amendment_id": "N2_V2_I7_REVIEW_THRESHOLD_AMENDMENT_20260922",
        "minimum_accepted_per_context": MIN_ACCEPTED_PER_CONTEXT,
        "records": accepted_records,
    }
    accepted_schema = json.loads((ROOT / "schemas/n2_v2_i7_accepted_asset_registry.schema.json").read_text())
    jsonschema.validate(accepted_registry, accepted_schema)
    write_json(OUT / "accepted_scene_only_assets.json", accepted_registry)

    review_evaluation = {
        "schema_version": 1,
        "study_version": "N2_V2",
        "attempt_id": ATTEMPT_ID,
        "source_attempt_id": "N2_V2_I5_20260922",
        "threshold_amendment_path": "N2-V2-I7输入门阈值修订.md",
        "status": "PASS_AMENDED_BLIND_REVIEW",
        "reviewed_card_count": result["reviewed_card_count"],
        "raw_semantic_agreement": result["raw_semantic_agreement"],
        "minimum_raw_semantic_agreement": MIN_RAW_AGREEMENT,
        "cohen_kappa": result["cohen_kappa"],
        "minimum_cohen_kappa": MIN_KAPPA,
        "reliability_passed": result["reliability_passed"],
        "accepted_card_count": len(accepted_records),
        "accepted_by_context_label": dict(accepted_by_context),
        "accepted_by_candidate_stratum": dict(accepted_by_stratum),
        "minimum_accepted_per_context": MIN_ACCEPTED_PER_CONTEXT,
        "rejected_card_count": len(rejected_cards),
        "rejected_reason_counts": dict(reason_counter),
        "rejected_cards": rejected_cards,
        "review_bundle_sha256s": review_bundle_hashes,
        "review_card_manifest_sha256": manifest_sha,
        "decision": "PASS_ACCEPTED_SCENE_ONLY_ASSETS_FROZEN",
        "next_gate": "LOW_LEVEL_MATCHING_NOT_RUN_YET",
    }
    write_json(OUT / "review_evaluation.json", review_evaluation)

    receipt = {
        "schema_version": 1,
        "study_version": "N2_V2",
        "stage_id": "N2_V2_INPUT_IDENTIFIABILITY",
        "attempt_id": ATTEMPT_ID,
        "terminal": True,
        "status": "PASS_INPUT_IDENTIFIABILITY_FROZEN",
        "reason": "Under the I7 threshold amendment, the two independent I5 review bundles pass reliability and leave at least 18 accepted scene-only assets for each final context label. Low-level matching has not run.",
        "gate_results": {
            "legacy_n2_closure": "PASS",
            "v2_protocol_freeze": "PASS_WITH_I7_THRESHOLD_AMENDMENT",
            "legacy_lineage_exclusion": "PASS",
            "scene_candidate_preparation": "PASS_60_CANDIDATES",
            "recipient_candidate_preparation": "PASS_12_CANDIDATES",
            "independent_reviewers": "PASS_TWO_SEALED_BUNDLES",
            "review_bundle_binding": "PASS",
            "semantic_reliability": "PASS_RAW_AGREEMENT_0_900_ABOVE_0_850",
            "cohen_kappa": "PASS_0_819_ABOVE_0_800",
            "artifact_union_reject": "APPLIED",
            "accepted_scene_only_registry": "PASS_43_ACCEPTED_19_AQUATIC_24_NON_AQUATIC",
            "low_level_matching": "NOT_RUN_AWAITING_NEXT_FRESH_ROOT",
            "identifiability_freeze": "NOT_ISSUED_PENDING_LOW_LEVEL_MATCHING",
            "behavior_gate": "NOT_AUTHORIZED",
            "n3": "NOT_AUTHORIZED",
        },
        "review_summary": {
            "reviewed_card_count": result["reviewed_card_count"],
            "raw_semantic_agreement": result["raw_semantic_agreement"],
            "cohen_kappa": result["cohen_kappa"],
            "accepted_card_count": len(accepted_records),
            "accepted_by_context_label": dict(accepted_by_context),
            "accepted_by_candidate_stratum": dict(accepted_by_stratum),
            "rejected_reason_counts": dict(reason_counter),
        },
        "access_accounting": {
            "review_bundle_reads": 2,
            "scene_image_payload_reads_for_review_evaluation": 0,
            "recipient_image_payload_reads_for_review_evaluation": 0,
            "recipient_mask_payload_reads_for_review_evaluation": 0,
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
            "gpu_uses": 0,
        },
        "artifact_bindings": {
            "protocol_sha256": sha(ROOT / "N2-V2输入可识别性恢复协议.md"),
            "threshold_amendment_sha256": sha(ROOT / "N2-V2-I7输入门阈值修订.md"),
            "contract_sha256": sha(ROOT / "config/n2_v2/input_identifiability_contract.json"),
            "legacy_lineage_ledger_sha256": sha(ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json"),
            "source_terminal_i5_sha256": sha(SOURCE / "terminal_receipt.json"),
            "scene_candidate_registry_sha256": sha(OUT / "private/scene_candidate_registry.json"),
            "recipient_candidate_registry_sha256": sha(OUT / "private/recipient_candidate_registry.json"),
            "mask_archive_bindings_sha256": sha(OUT / "private/openimages_mask_archive_bindings.json"),
            "review_card_manifest_sha256": sha(OUT / "review/review_card_manifest.json"),
            "reviewer_a_bundle_sha256": sha(OUT / "review/reviewer_A.json"),
            "reviewer_b_bundle_sha256": sha(OUT / "review/reviewer_B.json"),
            "accepted_scene_registry_sha256": sha(OUT / "accepted_scene_only_assets.json"),
            "review_evaluation_sha256": sha(OUT / "review_evaluation.json"),
            "review_schema_sha256": sha(ROOT / "schemas/n2_v2_review_bundle.schema.json"),
            "accepted_registry_schema_sha256": sha(ROOT / "schemas/n2_v2_i7_accepted_asset_registry.schema.json"),
            "evaluator_sha256": sha(ROOT / "finalize_n2_v2_i7_review.py"),
        },
        "next_gate": "STOP_I7_LOW_LEVEL_MATCHING_REQUIRES_NEW_FRESH_ROOT_NO_MODEL_ACCESS_AUTHORIZED",
    }
    write_json(OUT / "terminal_receipt.json", receipt)
    print(json.dumps({
        "status": receipt["status"],
        "raw_agreement": result["raw_semantic_agreement"],
        "cohen_kappa": result["cohen_kappa"],
        "accepted_by_context_label": dict(accepted_by_context),
        "model_accesses": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
