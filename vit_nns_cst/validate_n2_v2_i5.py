#!/usr/bin/env python3
"""Independent validator for the compact N2-V2 I5 human-review handoff."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_i5_20260922"


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
    prep = json.loads((OUT / "preparation_receipt.json").read_text())
    scenes = json.loads((OUT / "private/scene_candidate_registry.json").read_text())
    recipients = json.loads((OUT / "private/recipient_candidate_registry.json").read_text())
    cards = json.loads((OUT / "review/review_card_manifest.json").read_text())
    legacy = json.loads((ROOT / "manifests/n2_v2/legacy_lineage_exclusions.json").read_text())
    archives = json.loads((OUT / "private/openimages_mask_archive_bindings.json").read_text())
    allowed = json.loads((ROOT / "config/n2_v2/input_identifiability_contract.json").read_text())["allowed_input_stage_statuses"]

    check(prep["status"] == "BLOCKED_INDEPENDENT_REVIEWERS" and prep["status"] in allowed, "CONTRACT_STATUS_ENUM_VALID", checks)
    check(prep["model_access_count"] == prep["checkpoint_access_count"] == prep["gpu_use_count"] == 0, "MODEL_AND_GPU_ACCESS_ZERO", checks)
    check(prep["review_card_count"] == prep["scene_candidate_count"] == 60, "COMPACT_REVIEW_COUNT_60", checks)
    check(prep["candidate_stratum_counts"] == {"AQUATIC_CANDIDATE_STRATUM": 30, "NON_AQUATIC_CANDIDATE_STRATUM": 30}, "COMPACT_STRATUM_COUNTS_30_30", checks)
    check(prep["minimum_required_per_context"] == 24 and prep["reserve_per_context_before_review"] == 6, "RESERVE_MARGIN_DOCUMENTED", checks)
    check(len(scenes["records"]) == len(cards["records"]) == 60, "SCENE_AND_CARD_RECORD_COUNT", checks)
    check(Counter(row["candidate_stratum"] for row in scenes["records"]) == Counter({"AQUATIC_CANDIDATE_STRATUM": 30, "NON_AQUATIC_CANDIDATE_STRATUM": 30}), "SCENE_STRATUM_COUNTS", checks)
    check(len(recipients["records"]) == 12, "RECIPIENT_COUNT_12", checks)
    check(cards["blinding"]["display_transform"] == "FULL_FRAME_ASPECT_PRESERVING_MAX_960_GREY_PAD_NO_CROP", "FULL_FRAME_NO_CROP_DISPLAY", checks)

    old_ids = {row["source_group_id"] for row in legacy["records"]}
    old_hashes = {row["image_sha256"] for row in legacy["records"]}
    check(not old_ids.intersection(row["source_group_id"] for row in scenes["records"] + recipients["records"]), "NO_LEGACY_SOURCE_ID_OVERLAP", checks)
    check(not old_hashes.intersection(row["image_sha256"] for row in scenes["records"] + recipients["records"]), "NO_LEGACY_IMAGE_HASH_OVERLAP", checks)

    card_by_id = {row["card_id"]: row for row in cards["records"]}
    check(len(card_by_id) == 60, "BLIND_CARD_IDS_UNIQUE", checks)
    for row in scenes["records"]:
        image = OUT / row["image_path"]
        card = OUT / row["review_image_path"]
        check(image.is_file() and sha(image) == row["image_sha256"], "SCENE_IMAGE_BINDING_" + row["asset_id"], checks)
        check(card.is_file() and sha(card) == row["review_image_sha256"], "CARD_IMAGE_BINDING_" + row["card_id"], checks)
        check(card_by_id[row["card_id"]]["image_sha256"] == row["review_image_sha256"], "CARD_MANIFEST_BINDING_" + row["card_id"], checks)
        check(row["foreground_removal_applied"] is False and row["producer_operations"] == [], "NO_SCENE_EDIT_" + row["asset_id"], checks)
        author_ok = bool(row["artist"]) or row["license_short_name"] in {"Public domain", "PDM"}
        check(bool(row["license_short_name"] and row["description_url"] and author_ok), "SCENE_LICENSE_" + row["asset_id"], checks)

    for row in recipients["records"]:
        image = OUT / row["image_path"]
        mask = OUT / row["mask_path"]
        check(image.is_file() and sha(image) == row["image_sha256"], "RECIPIENT_IMAGE_" + row["recipient_id"], checks)
        check(mask.is_file() and sha(mask) == row["mask_sha256"], "RECIPIENT_MASK_" + row["recipient_id"], checks)
        check(row["target_model_class_wnid"] == "n01443537" and row["target_model_class_index"] == 1, "RECIPIENT_TARGET_MAPPING_" + row["recipient_id"], checks)
        check(row["class_alignment_qc"] == "PASS_VISUAL_GOLDFISH_COMPATIBLE", "RECIPIENT_CLASS_QC_" + row["recipient_id"], checks)
        check(row["license_url"] == "https://creativecommons.org/licenses/by/2.0/", "RECIPIENT_IMAGE_LICENSE_" + row["recipient_id"], checks)

    check(archives["annotation_license"] == "CC_BY_4_0" and len(archives["archives"]) == 7, "MASK_ARCHIVE_BINDINGS", checks)
    manifest_sha = sha(OUT / "review/review_card_manifest.json")
    for template in sorted((OUT / "review/templates").glob("*.json")):
        data = json.loads(template.read_text())
        check(data["review_card_manifest_sha256"] == manifest_sha, "TEMPLATE_MANIFEST_BINDING_" + template.stem, checks)
        check(len(data["reviews"]) == 60 and {row["card_id"] for row in data["reviews"]} == set(card_by_id), "TEMPLATE_CARD_SET_" + template.stem, checks)
    for reviewer in ("reviewer_A", "reviewer_B"):
        check((OUT / "review" / f"{reviewer}.html").is_file(), "OFFLINE_FORM_PRESENT_" + reviewer, checks)

    result = {
        "schema_version": 1,
        "study_version": "N2_V2",
        "attempt_id": "N2_V2_I5_20260922",
        "status": "PASS_REVIEW_PREPARATION_VALIDATION",
        "check_count": len(checks),
        "checks": checks,
        "scene_candidate_count": 60,
        "candidate_stratum_counts": {"AQUATIC_CANDIDATE_STRATUM": 30, "NON_AQUATIC_CANDIDATE_STRATUM": 30},
        "recipient_candidate_count": 12,
        "review_card_count": 60,
        "sealed_review_bundle_count": 0,
        "model_access_count": 0,
        "behavior_gate": "NOT_AUTHORIZED",
    }
    if "--write" in sys.argv:
        (OUT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
