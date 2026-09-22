#!/usr/bin/env python3
"""Independent input-only validator for N2-V2 I2 review preparation."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence/n2_v2_i2_20260922"

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def check(condition: bool, code: str, checks: list[str]) -> None:
    if not condition: raise RuntimeError(code)
    checks.append(code)

def main() -> int:
    checks=[]
    receipt=json.loads((OUT/"preparation_receipt.json").read_text())
    scenes=json.loads((OUT/"private/scene_candidate_registry.json").read_text())
    recipients=json.loads((OUT/"private/recipient_candidate_registry.json").read_text())
    cards=json.loads((OUT/"review/review_card_manifest.json").read_text())
    legacy=json.loads((ROOT/"manifests/n2_v2/legacy_lineage_exclusions.json").read_text())
    check(receipt["status"]=="READY_FOR_TWO_INDEPENDENT_BLIND_REVIEWS","PREPARATION_STATUS_READY",checks)
    check(receipt["model_access_count"]==receipt["checkpoint_access_count"]==receipt["gpu_use_count"]==0,"MODEL_AND_GPU_ACCESS_ZERO",checks)
    check(len(scenes["records"])==102,"SCENE_CANDIDATE_COUNT_102",checks)
    counts=Counter(r["candidate_stratum"] for r in scenes["records"]); check(counts==Counter({"AQUATIC_CANDIDATE_STRATUM":54,"NON_AQUATIC_CANDIDATE_STRATUM":48}),"SCENE_STRATUM_COUNTS",checks)
    check(len(recipients["records"])==12,"RECIPIENT_COUNT_12",checks)
    check(len(cards["records"])==102 and len({r["card_id"] for r in cards["records"]})==102,"BLIND_CARD_COUNT_AND_IDS",checks)
    old_ids={r["source_group_id"] for r in legacy["records"]}; old_hashes={r["image_sha256"] for r in legacy["records"]}
    check(not old_ids.intersection(r["source_group_id"] for r in scenes["records"]+recipients["records"]),"NO_LEGACY_SOURCE_ID_OVERLAP",checks)
    check(not old_hashes.intersection(r["image_sha256"] for r in scenes["records"]+recipients["records"]),"NO_LEGACY_IMAGE_HASH_OVERLAP",checks)
    for row in scenes["records"]:
        image=OUT/row["image_path"]; card=OUT/row["review_image_path"]
        check(image.is_file() and sha(image)==row["image_sha256"],"SCENE_IMAGE_BINDING_"+row["asset_id"],checks)
        check(card.is_file() and sha(card)==row["review_image_sha256"],"CARD_IMAGE_BINDING_"+row["card_id"],checks)
        check(row["foreground_removal_applied"] is False and row["producer_operations"]==[],"NO_SCENE_EDIT_"+row["asset_id"],checks)
        author_ok = bool(row["artist"]) or row["license_short_name"] in {"Public domain", "PDM"}
        check(bool(row["license_short_name"] and row["description_url"] and author_ok),"SCENE_LICENSE_PROVENANCE_"+row["asset_id"],checks)
    for row in recipients["records"]:
        image=OUT/row["image_path"]; mask=OUT/row["mask_path"]
        check(image.is_file() and sha(image)==row["image_sha256"],"RECIPIENT_IMAGE_BINDING_"+row["recipient_id"],checks)
        check(mask.is_file() and sha(mask)==row["mask_sha256"],"RECIPIENT_MASK_BINDING_"+row["recipient_id"],checks)
        check(row["target_model_class_wnid"]=="n01443537" and row["target_model_class_index"]==1 and row["class_alignment_qc"]=="PASS_VISUAL_GOLDFISH_COMPATIBLE","RECIPIENT_CLASS_ALIGNMENT_"+row["recipient_id"],checks)
        check(row["license_url"]=="https://creativecommons.org/licenses/by/2.0/","RECIPIENT_IMAGE_LICENSE_"+row["recipient_id"],checks)
        archive=OUT/"private/openimages_mask_archives"/("train-masks-"+row["openimages_image_id"][0]+".zip")
        with zipfile.ZipFile(archive) as bundle:
            check(hashlib.sha256(bundle.read(Path(row["mask_path"]).name)).hexdigest()==row["mask_sha256"],"RECIPIENT_MASK_ARCHIVE_BINDING_"+row["recipient_id"],checks)
    manifest_sha=sha(OUT/"review/review_card_manifest.json")
    for template in sorted((OUT/"review/templates").glob("*.json")):
        data=json.loads(template.read_text()); check(data["review_card_manifest_sha256"]==manifest_sha,"TEMPLATE_MANIFEST_BINDING_"+template.stem,checks)
        check(len(data["reviews"])==102 and {r["card_id"] for r in data["reviews"]}=={r["card_id"] for r in cards["records"]},"TEMPLATE_CARD_SET_"+template.stem,checks)
    result={"schema_version":1,"study_version":"N2_V2","attempt_id":"N2_V2_I2_20260922","status":"PASS_REVIEW_PREPARATION_VALIDATION","check_count":len(checks),"checks":checks,"scene_candidate_count":102,"recipient_candidate_count":12,"review_card_count":102,"sealed_review_bundle_count":0,"model_access_count":0,"behavior_gate":"NOT_AUTHORIZED"}
    if "--write" in sys.argv: (OUT/"validation.json").write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
